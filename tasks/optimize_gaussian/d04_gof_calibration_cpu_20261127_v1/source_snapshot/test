from __future__ import annotations

import hashlib
import json
from io import BytesIO

import numpy as np
import pytest

from core.source_integrity import capture_source_snapshot
from tasks.optimize_gaussian.d04_raw_reference import (
    CORRELATOR_SCHEMA,
    MANIFEST_SCHEMA,
    PROTOCOL_SCHEMA,
    evaluate_d04_raw_snapshot,
    fold_a4p_odd,
    fold_pp_even,
)


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest_array(array: np.ndarray) -> str:
    return _digest_bytes(np.ascontiguousarray(array).tobytes())


def _synthetic_payload(*, complex_pp: bool = False, bad_ids: bool = False):
    ncfg = 240
    config_ids = np.arange(100, 100 + ncfg, dtype=np.int64)
    trajectories = 7 * config_ids + 3
    if bad_ids:
        config_ids[20] = config_ids[19]
    rng = np.random.default_rng(20261126)
    t = np.arange(16, dtype=float)
    masses = (0.32, 0.91)
    pp = np.zeros((ncfg, 3, 16), dtype=np.float64)
    a4p = np.zeros((ncfg, 16), dtype=np.float64)
    for channel, (a0, a1) in enumerate(((1.0, 0.22), (0.7, -0.12), (1.3, 0.30))):
        amp0 = a0 * (1.0 + rng.normal(0, 0.015, ncfg))
        amp1 = a1 * (1.0 + rng.normal(0, 0.04, ncfg))
        model = (amp0[:, None] * (np.exp(-masses[0] * t)
                                  + np.exp(-masses[0] * (16 - t)))
                 + amp1[:, None] * (np.exp(-masses[1] * t)
                                    + np.exp(-masses[1] * (16 - t))))
        pp[:, channel, :] = model + rng.normal(0, 2e-5, model.shape)
    amp_a0 = 0.18 * (1 + rng.normal(0, 0.015, ncfg))
    amp_a1 = -0.035 * (1 + rng.normal(0, 0.04, ncfg))
    a4p[:] = (amp_a0[:, None] * (np.exp(-masses[0] * t)
                                 - np.exp(-masses[0] * (16 - t)))
              + amp_a1[:, None] * (np.exp(-masses[1] * t)
                                   - np.exp(-masses[1] * (16 - t)))
              + rng.normal(0, 1e-5, (ncfg, 16)))
    if complex_pp:
        pp = pp.astype(np.complex128) + 1j * 1e-9
    buf = BytesIO()
    np.savez_compressed(
        buf, schema=np.asarray(CORRELATOR_SCHEMA),
        configuration_ids=config_ids, trajectory_ids=trajectories,
        pp=pp, a4p=a4p,
    )
    data = buf.getvalue()
    protocol = {
        "schema": PROTOCOL_SCHEMA, "geometry": "S16T16",
        "action": "quenched_wilson", "kappa": 0.14,
        "temporal_correlator_boundary": "periodic",
        "pp_order": ["PP_n0", "PP_n8", "PP_n16"],
        "dimension_order": "configuration,channel,time", "temporal_extent": 16,
    }
    protocol_bytes = json.dumps(protocol, sort_keys=True).encode()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "split_id": "E-reference-calibration-01",
        "geometry": "S16T16", "action": "quenched_wilson", "kappa": 0.14,
        "temporal_correlator_boundary": "periodic",
        "source_protocol_sha256": _digest_bytes(protocol_bytes),
        "configuration_ids_sha256": _digest_array(config_ids),
        "trajectory_ids_sha256": _digest_array(trajectories),
        "correlator_data_sha256": _digest_bytes(data),
        "pp_order": ["PP_n0", "PP_n8", "PP_n16"],
        "dimension_order": "configuration,channel,time",
    }
    return (config_ids, trajectories, json.dumps(manifest, sort_keys=True).encode(),
            data, protocol_bytes)


def _capture(tmp_path, *, split_mutation=None, complex_pp=False, bad_ids=False):
    config_ids, trajectories, manifest, data, protocol = _synthetic_payload(
        complex_pp=complex_pp, bad_ids=bad_ids,
    )
    if split_mutation:
        value = json.loads(manifest)
        value["split_id"] = split_mutation
        manifest = json.dumps(value, sort_keys=True).encode()
    (tmp_path / "manifest.json").write_bytes(manifest)
    (tmp_path / "correlators.npz").write_bytes(data)
    (tmp_path / "source_protocol.json").write_bytes(protocol)
    snapshot = capture_source_snapshot(
        tmp_path,
        {"manifest": _digest_bytes(manifest), "correlator_data": _digest_bytes(data),
         "source_protocol": _digest_bytes(protocol)},
        {"manifest": "manifest.json", "correlator_data": "correlators.npz",
         "source_protocol": "source_protocol.json"},
    )
    return snapshot, config_ids, trajectories, data


def test_even_and_odd_folding_keep_midpoint_once_and_force_a4p_midpoint_zero():
    pp = np.arange(16, dtype=float)
    a4p = np.arange(16, dtype=float)
    even = fold_pp_even(pp)
    odd = fold_a4p_odd(a4p)
    assert even.shape == (9,)
    assert even[0] == 0.0 and even[8] == 8.0
    assert even[3] == (3 + 13) / 2
    assert odd.shape == (9,)
    assert odd[0] == 0.0 and odd[8] == 0.0
    assert odd[3] == (3 - 13) / 2


def test_d04_reader_fits_six_joint_variants_from_the_captured_bytes(tmp_path):
    snapshot, _, _, _ = _capture(tmp_path)
    # The reader must consume the immutable snapshot, not reopen this path.
    (tmp_path / "correlators.npz").write_bytes(b"replaced after capture")
    result = evaluate_d04_raw_snapshot(snapshot)
    assert result["status"] in {"diagnostic_only", "diagnostic_incomplete"}
    assert result["certified"] is False and result["physical_score"] is None
    assert result["configuration_ids"]["count"] == 240
    assert result["fit_denominator"]["planned"] == 6
    assert result["fit_denominator"]["valid"] + result["fit_denominator"]["failed"] == 6
    main = next(item for item in result["fit_variants"]
                if item["tmin"] == 3 and item["nstates"] == 2)
    assert main["status"] == "valid"
    assert abs(main["reference_mass"] - 0.32) < 0.015
    assert main["gof_p_approx"] is not None
    assert len(main["attempts"]) == 3
    assert "no dependence-aware block bootstrap" in result["open_gates"][0]


def test_d04_mbb_is_seeded_paired_and_reports_channel_and_model_shifts(tmp_path):
    snapshot, *_ = _capture(tmp_path)
    first = evaluate_d04_raw_snapshot(
        snapshot, block_bootstrap_replicates=4,
        block_lengths=(1, 4), block_bootstrap_seed=901,
    )
    second = evaluate_d04_raw_snapshot(
        snapshot, block_bootstrap_replicates=4,
        block_lengths=(1, 4), block_bootstrap_seed=901,
    )
    diagnostic = first["block_bootstrap"]
    assert diagnostic["status"] == "diagnostic_only"
    assert diagnostic == second["block_bootstrap"]
    assert diagnostic["resampling"]["block_lengths"] == [1, 4]
    assert "not estimated dependence scales" in diagnostic["resampling"]["interpretation"]
    for length in ("1", "4"):
        item = diagnostic["by_block_length"][length]
        assert item["resample_covariance_denominator"] == {
            "planned": 4, "valid": 4, "failed": 0, "abstained": 0,
        }
        assert item["fit_denominators"]["joint_tmin3_nstates2"] == {
            "planned": 4, "valid": 4, "failed": 0, "abstained": 0,
        }
        assert item["primary_mass"]["draw_denominator"] == {
            "planned": 4, "valid": 4, "failed": 0, "abstained": 0,
        }
        assert item["contrasts"]["pp_only_tmin3_nstates2"][
            "bootstrap_paired_shift"]["draw_denominator"]["valid"] == 4
        assert item["contrasts"]["a4p_only_tmin3_nstates2"][
            "bootstrap_paired_shift"]["draw_denominator"]["valid"] == 4
        assert item["contrasts"]["joint_tmin4_nstates3"][
            "paired_shift_over_se"] is not None


def test_d04_mbb_rejects_invalid_block_lengths(tmp_path):
    snapshot, *_ = _capture(tmp_path)
    with pytest.raises(ValueError, match="unique integers"):
        evaluate_d04_raw_snapshot(
            snapshot, block_bootstrap_replicates=2, block_lengths=(1, 1),
        )


def test_d04_reader_rejects_closed_splits_and_manifest_mismatch(tmp_path):
    snapshot, *_ = _capture(tmp_path, split_mutation="E-gap-sealed")
    with pytest.raises(ValueError, match="refuses sealed"):
        evaluate_d04_raw_snapshot(snapshot)


def test_d04_reader_rejects_nonascending_ids_and_complex_values(tmp_path):
    snapshot, *_ = _capture(tmp_path, bad_ids=True)
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate_d04_raw_snapshot(snapshot)
    other = tmp_path / "complex"
    other.mkdir()
    snapshot, *_ = _capture(other, complex_pp=True)
    with pytest.raises(ValueError, match="explicitly real"):
        evaluate_d04_raw_snapshot(snapshot)


def test_d04_reader_rejects_stale_or_wrong_axis_npz(tmp_path):
    snapshot, *_ = _capture(tmp_path)
    # A changed captured manifest must not be silently accepted under an old digest.
    altered = snapshot.source_bytes["manifest"].replace(b"0.14", b"0.15")
    from dataclasses import replace
    forged = replace(snapshot, source_bytes={**snapshot.source_bytes, "manifest": altered})
    with pytest.raises(ValueError, match="captured bytes do not match"):
        evaluate_d04_raw_snapshot(forged)


def test_d04_reader_requires_the_captured_protocol_bytes_to_match_manifest(tmp_path):
    snapshot, *_ = _capture(tmp_path)
    protocol = snapshot.source_bytes["source_protocol"] + b" "
    from dataclasses import replace
    forged = replace(snapshot, source_bytes={**snapshot.source_bytes,
                                            "source_protocol": protocol})
    with pytest.raises(ValueError, match="captured bytes do not match"):
        evaluate_d04_raw_snapshot(forged)


def test_complex_fold_helpers_do_not_silently_drop_imaginary_parts():
    with pytest.raises(ValueError, match="finite real"):
        fold_pp_even(np.ones((2, 16), dtype=np.complex128))
    with pytest.raises(ValueError, match="finite real"):
        fold_a4p_odd(np.ones((2, 16), dtype=np.complex128))


def test_d04_mbb_source_map_binds_snapshot_reader_implementation():
    from pathlib import Path
    from tasks.optimize_gaussian import d04_block_bootstrap_diagnostic as diagnostic

    expected = Path(__file__).resolve().parents[1] / "core/source_integrity.py"
    assert diagnostic._SOURCE_PATHS["source_integrity"].resolve() == expected
    npz_safety = Path(__file__).resolve().parents[1] / "core/npz_safety.py"
    assert diagnostic._SOURCE_PATHS["npz_safety"].resolve() == npz_safety
