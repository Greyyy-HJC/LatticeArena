from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from core.source_integrity import capture_source_snapshot
from tasks.pion_projection.anchor_scalar_reader import (
    CORRELATOR_SCHEMA,
    FEATURE_VECTOR_RESULT_SCHEMA,
    MAP_HASH,
    MAP_ID,
    MANIFEST_SCHEMA,
    anchor_scalar,
    read_anchor_scalar_snapshot,
)
from tasks.pion_projection.anchor_spectral_interval import (
    read_anchor_ground_mass_interval_snapshot,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_sha(array: np.ndarray) -> str:
    return _sha(np.ascontiguousarray(array.astype("<i8", copy=False)).tobytes())


def _fixture(tmp_path, *, n=12, invalid=False, split="anchor-open-calibration", complex_data=False):
    protocol = (Path(__file__).resolve().parents[1]
                / "tasks/pion_projection/protocol.json").read_bytes()
    ids = np.arange(500, 500 + n, dtype=np.int64)
    t = np.arange(16, dtype=float)
    correlators = np.broadcast_to(
        np.exp(-0.5 * t) + np.exp(-0.5 * (16 - t)), (n, 16),
    ).copy()
    if invalid:
        correlators[-1] = 1.0
    if complex_data:
        correlators = correlators.astype(np.complex128) + 1j * 1e-8
    buffer = BytesIO()
    np.savez_compressed(
        buffer, schema=np.asarray(CORRELATOR_SCHEMA),
        state_ids=ids, pp_correlators=correlators,
    )
    data = buffer.getvalue()
    manifest = {
        "schema": MANIFEST_SCHEMA, "split_id": split,
        "chain_id": "anchor-calibration-A",
        "state_unit": "saved gauge configuration",
        "geometry": "S16T16", "kappa": 0.14, "temporal_extent": 16,
        "source_protocol_sha256": _sha(protocol),
        "state_ids_sha256": _array_sha(ids),
        "correlator_data_sha256": _sha(data),
    }
    encoded = json.dumps(manifest, sort_keys=True).encode()
    (tmp_path / "manifest.json").write_bytes(encoded)
    (tmp_path / "anchor.npz").write_bytes(data)
    (tmp_path / "protocol.json").write_bytes(protocol)
    snapshot = capture_source_snapshot(
        tmp_path,
        {"anchor_manifest": _sha(encoded), "anchor_correlator_data": _sha(data),
         "anchor_protocol": _sha(protocol)},
        {"anchor_manifest": "manifest.json", "anchor_correlator_data": "anchor.npz",
         "anchor_protocol": "protocol.json"},
    )
    return snapshot


def test_anchor_scalar_recovers_one_cosh_mass_without_clipping():
    t = np.arange(16, dtype=float)
    correlator = np.exp(-0.5 * t) + np.exp(-0.5 * (16 - t))
    assert anchor_scalar(correlator) == pytest.approx(0.5, abs=1e-12)
    assert MAP_ID == "pion-folded-pp-cosh-effective-mass-t4-v1"
    assert len(MAP_HASH) == 64


def test_snapshot_reader_uses_captured_bytes_and_retains_complete_denominator(tmp_path):
    snapshot = _fixture(tmp_path, n=12, invalid=True)
    (tmp_path / "anchor.npz").write_bytes(b"replaced after capture")
    result = read_anchor_scalar_snapshot(snapshot)
    assert result["status"] == "ABSTAIN_INVALID_ANCHOR_STATES"
    assert result["state_denominator"] == {
        "unit": "anchor_chain_state", "planned": 12,
        "valid": 11, "failed": 1, "abstained": 0,
    }
    assert result["scalar_values"][:-1] == pytest.approx([0.5] * 11)
    assert result["scalar_values"][-1] is None
    assert result["mean_scalar_if_all_valid"] is None
    assert result["physical_score"] is None


def test_snapshot_reader_accepts_open_anchor_fixture_and_expected_count(tmp_path):
    snapshot = _fixture(tmp_path, n=20)
    result = read_anchor_scalar_snapshot(snapshot, expected_states=20)
    assert result["status"] == "diagnostic_only"
    assert result["state_denominator"]["planned"] == 20
    assert result["mean_scalar_if_all_valid"] == pytest.approx(0.5, abs=1e-12)
    with pytest.raises(ValueError, match="predeclared denominator"):
        read_anchor_scalar_snapshot(snapshot, expected_states=4000)


def test_snapshot_reader_can_return_ordered_pp_mean_features(tmp_path):
    snapshot = _fixture(tmp_path, n=12)
    result = read_anchor_scalar_snapshot(snapshot, include_feature_vector=True)

    assert result["schema"] == FEATURE_VECTOR_RESULT_SCHEMA
    feature_vector = result["feature_vector"]
    assert feature_vector["feature_order"] == [
        "C_fold_3", "C_fold_4", "C_fold_5",
    ]
    assert feature_vector["denominator"] == {
        "unit": "saved gauge configuration feature row",
        "planned": 12, "valid": 12, "failed": 0, "abstained": 0,
    }
    assert np.asarray(feature_vector["matrix"]).shape == (12, 3)
    assert np.asarray(feature_vector["matrix"])[:, 1] == pytest.approx(
        [np.exp(-2.0) + np.exp(-6.0)] * 12,
    )
    assert "not per-state spectral masses" in feature_vector["note"]


def test_snapshot_reader_feature_path_keeps_rows_when_scalar_map_abstains(tmp_path):
    snapshot = _fixture(tmp_path, n=12, invalid=True)
    result = read_anchor_scalar_snapshot(snapshot, include_feature_vector=True)

    assert result["status"] == "ABSTAIN_INVALID_ANCHOR_STATES"
    assert result["state_denominator"] == {
        "unit": "anchor_chain_state", "planned": 12,
        "valid": 11, "failed": 1, "abstained": 0,
    }
    assert result["feature_vector"]["denominator"] == {
        "unit": "saved gauge configuration feature row",
        "planned": 12, "valid": 12, "failed": 0, "abstained": 0,
    }


def test_snapshot_spectral_interval_path_is_conditional_and_hash_bound(tmp_path):
    snapshot = _fixture(tmp_path, n=4000)
    (tmp_path / "anchor.npz").write_bytes(b"changed after immutable capture")

    result = read_anchor_ground_mass_interval_snapshot(
        snapshot,
        support_bounds=np.asarray([[0.22, 0.23], [0.13, 0.15], [0.08, 0.10]]),
        gamma_ps_lower=0.2,
        alpha_anchor=0.01,
        initial_law_tv_bound=0.0,
        burn_in_units=0,
        stationary_start=True,
        score_scale_R=2.9,
        expected_states=4000,
    )

    lower, upper = result["identified_ground_mass_interval"]
    assert lower <= 0.5 <= upper
    assert result["source_read_mode"] == "immutable_snapshot"
    assert result["split_id"] == "anchor-open-calibration"
    assert set(result["source_hashes"]) == {
        "anchor_manifest", "anchor_correlator_data", "anchor_protocol",
    }
    assert result["feature_denominator"] == {
        "unit": "saved gauge configuration feature row",
        "planned": 4000, "valid": 4000, "failed": 0, "abstained": 0,
    }
    assert result["input_markov_mean_box"]["joint_coverage_probability_lower_bound"] == 0.99
    assert result["target_bridge_status"] == "unverified_for_independent_E_reference"
    assert result["common_bias_bound_C"] is None
    assert result["physical_score"] is None and not result["certified"]


def test_snapshot_spectral_interval_path_rejects_sealed_split(tmp_path):
    snapshot = _fixture(tmp_path, n=12, split="anchor-sealed-confirmation")
    with pytest.raises(ValueError, match="open, explicitly named anchor split"):
        read_anchor_ground_mass_interval_snapshot(
            snapshot,
            support_bounds=np.asarray([[0.22, 0.23], [0.13, 0.15], [0.08, 0.10]]),
            gamma_ps_lower=0.2,
            alpha_anchor=0.01,
            initial_law_tv_bound=0.0,
            burn_in_units=0,
            stationary_start=True,
            score_scale_R=2.9,
            expected_states=12,
        )


def test_snapshot_reader_rejects_sealed_split_and_complex_correlators(tmp_path):
    with pytest.raises(ValueError, match="open, explicitly named anchor split"):
        read_anchor_scalar_snapshot(_fixture(tmp_path, split="anchor-sealed-confirmation"))
    complex_root = tmp_path / "complex"
    complex_root.mkdir()
    with pytest.raises(ValueError, match="finite and explicitly real"):
        read_anchor_scalar_snapshot(_fixture(complex_root, complex_data=True))


@pytest.mark.parametrize("bad", [
    np.ones(16),
    np.full(16, np.nan),
    np.zeros(16),
])
def test_scalar_map_rejects_invalid_support_or_denominator(bad):
    with pytest.raises(ValueError):
        anchor_scalar(bad)


@pytest.mark.parametrize(("amplitude", "tolerance"), [
    (1e308, 1e-12),
    # At subnormal scale the input itself has only a few relative bits.
    (1e-320, 7e-4),
])
def test_scalar_map_is_stable_under_extreme_common_correlator_scale(amplitude, tolerance):
    mass = 0.2
    correlator = np.zeros(16, dtype=np.float64)
    for time in (3, 4, 5):
        value = amplitude * np.cosh(mass * (8 - time))
        correlator[time] = value
        correlator[16 - time] = value

    assert anchor_scalar(correlator) == pytest.approx(mass, abs=tolerance)
