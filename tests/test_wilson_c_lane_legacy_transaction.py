from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json

import numpy as np

from core.io import fingerprint
from core.source_integrity import capture_source_snapshot
from tasks.wilson_loop import c_lane_open_development_diagnostic as c_diagnostic
from tasks.wilson_loop import c_lane_snapshot_reader as c_reader
from tasks.wilson_loop.c_lane_legacy_transaction import (
    _PINNED_ARCHIVE_SOURCES,
    adapt_legacy_calibration_snapshot,
)
from tasks.wilson_loop.lane_contract import selected_protocol


_ROLES = {
    "producer_npz": "calibration.npz",
    "producer_provenance": "calibration_provenance.json",
    "producer_commit": "calibration.commit.json",
    "input_manifest": "hmc_manifest.json",
}
_DEPENDENCIES = {
    name: _PINNED_ARCHIVE_SOURCES[name]
    for name in ("kernels.py", "statistics.py", "measure.py", "common.py", "identity.py")
}


def _legacy_package(
    tmp_path,
    *,
    include_calibration_opt_in=True,
    selected_kernel=True,
    full_stencil=True,
    wrong_trajectory=False,
):
    ids = list(range(100, 112))
    r_values = [1, 2, 3, 4] if full_stencil else [1, 2, 3]
    t_values = [4, 5, 6, 7, 8]
    kernels = [
        {"name": "identity", "kind": "identity"},
        ({"name": "stout_rho0.1_n5", "kind": "stout", "rho": 0.1, "steps": 5}
         if selected_kernel else {"name": "ape_a0.5_n5", "alpha": 0.5, "steps": 5}),
    ]
    configs = [
        {
            "id": cfg,
            "trajectory": 500 + 4 * (cfg - ids[0]),
            "sha256": sha256(f"raw-gauge-{cfg}".encode()).hexdigest(),
        }
        for cfg in ids
    ]
    hmc_manifest = {
        "schema": "hmc-input-manifest-v1",
        "ensemble": "S16T16",
        "action": "Wilson gauge action, beta=6",
        "lattice": [16, 16, 16, 16],
        "beta": 6.0,
        "seed": 1776,
        "configs": configs,
    }
    input_manifest_bytes = json.dumps(
        hmc_manifest, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    protocol = {
        "version": "legacy-wilson-measurement-v1",
        "measurement_splits": (["calibration"] if include_calibration_opt_in
                               else ["development", "gap", "validation"]),
        "calibration": [ids[0], ids[-1]],
        "geometry": [16, 16, 16, 16],
        "beta": 6.0,
        "action": "Wilson gauge action, beta=6",
        "boundary_conditions": "periodic_all_directions",
        "r_values": r_values,
        "t_values": t_values,
        "kernels": kernels,
        "source_hashes": dict(_PINNED_ARCHIVE_SOURCES),
        "input_manifest": {
            "path": "hmc_manifest.json",
            "sha256": sha256(input_manifest_bytes).hexdigest(),
            "schema": "hmc-input-manifest-v1",
            "ensemble": "S16T16",
            "beta": 6.0,
            "hmc_seed": 1776,
        },
    }
    rows, cols = np.meshgrid(r_values, t_values, indexing="ij")
    loops = np.empty((len(ids), len(kernels), len(r_values), len(t_values)), dtype=np.float64)
    for kernel_index, sigma in enumerate((0.12, 0.10)):
        loops[:, kernel_index] = np.exp(-sigma * rows * cols)[None, :, :]
    if not selected_kernel:
        loops[:, 1] = loops[:, 0]
    # The generation follows the archived measurement_spec fingerprint rule.
    generation_spec = {
        "split": "calibration",
        "ids": ids,
        "protocol_hash": fingerprint(protocol),
        "kernel_sha256": _DEPENDENCIES["kernels.py"],
        "dependencies": _DEPENDENCIES,
    }
    generation = fingerprint(generation_spec)
    npz_stream = BytesIO()
    np.savez_compressed(
        npz_stream,
        ids=np.asarray(ids, dtype=np.int64),
        loops=loops,
        protocol_hash=np.asarray(fingerprint(protocol)),
        generation=np.asarray(generation),
    )
    npz_bytes = npz_stream.getvalue()
    records = []
    for index, cfg in enumerate(ids):
        records.append({
            "id": cfg,
            "hmc_trajectory": configs[index]["trajectory"] + (1 if wrong_trajectory else 0),
            "gauge_sha256": configs[index]["sha256"],
            "loops_sha256": sha256(np.ascontiguousarray(loops[index]).tobytes()).hexdigest(),
        })
    provenance = {
        "protocol": protocol,
        "kernel_sha256": _DEPENDENCIES["kernels.py"],
        "input_manifest_sha256": sha256(input_manifest_bytes).hexdigest(),
        "generation": generation,
        "dependencies": _DEPENDENCIES,
        "records": records,
        "git": {"commit": "a" * 40, "dirty": False},
    }
    provenance_bytes = json.dumps(
        provenance, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    commit = {
        "generation": generation,
        "split": "calibration",
        "npz_sha256": sha256(npz_bytes).hexdigest(),
        "provenance_sha256": sha256(provenance_bytes).hexdigest(),
        "loops_sha256": sha256(np.ascontiguousarray(loops).tobytes()).hexdigest(),
    }
    commit_bytes = json.dumps(
        commit, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    payloads = {
        "producer_npz": npz_bytes,
        "producer_provenance": provenance_bytes,
        "producer_commit": commit_bytes,
        "input_manifest": input_manifest_bytes,
    }
    paths = {}
    expected = {}
    for role, filename in _ROLES.items():
        (tmp_path / filename).write_bytes(payloads[role])
        paths[role] = filename
        expected[role] = sha256(payloads[role]).hexdigest()
    snapshot = capture_source_snapshot(tmp_path, expected, paths)
    return snapshot, {role: role for role in _ROLES}


def _recapture(root):
    paths = {role: filename for role, filename in _ROLES.items()}
    expected = {
        role: sha256((root / filename).read_bytes()).hexdigest()
        for role, filename in paths.items()
    }
    return capture_source_snapshot(root, expected, paths)


def _shorten_bootstrap(monkeypatch):
    selected = selected_protocol()
    selected["familywise_inference"]["block_bootstrap_draws"] = 8
    monkeypatch.setattr(c_reader, "selected_protocol", lambda: selected)
    monkeypatch.setattr(c_reader, "N_DRAWS", 8)
    monkeypatch.setattr(c_diagnostic, "N_DRAWS", 8)


def test_legacy_calibration_adapter_reads_one_snapshot_and_derives_current_schema(
    tmp_path, monkeypatch,
):
    _shorten_bootstrap(monkeypatch)
    snapshot, names = _legacy_package(tmp_path)
    original_hashes = dict(snapshot.expected_hashes)
    for filename in _ROLES.values():
        (tmp_path / filename).write_bytes(b"mutated after capture")

    result = adapt_legacy_calibration_snapshot(
        snapshot, source_names=names, calibration_opt_in=True,
    )

    assert result["status"] == "LEGACY_TRANSACTION_ADAPTED_DIAGNOSTIC_ONLY"
    assert result["certified"] is False and result["physical_score"] is None
    assert result["source_hashes"] == {
        role: original_hashes[role] for role in names
    }
    assert result["source_denominator"] == {
        "unit": "captured legacy input file", "planned": 4,
        "valid": 4, "failed": 0, "abstained": 0,
    }
    assert result["configuration_denominator"] == {
        "unit": "paired saved gauge configuration", "planned": 12,
        "valid": 12, "failed": 0, "abstained": 0,
    }
    assert result["draw_denominator"]["planned"] == 3 * 8
    assert result["current_reader"]["summary_consistency"] == "MATCHED_TO_CAPTURED_LOOP_MATRIX"
    assert result["current_reader"]["status"] == "ARRAY_REDUCED_NOT_CERTIFIED"
    assert result["current_reader"]["physical_score"] is None
    assert result["producer_hash_claims"]["source_bytes_for_producer_code_captured"] is False
    assert result["text_flow"]
    canonical = result["canonical_snapshot"]
    assert canonical.read_bytes("manifest")
    assert canonical.read_bytes("loop_measurements")
    assert canonical.read_bytes("interval_summary")


def test_legacy_calibration_adapter_requires_explicit_opt_in(tmp_path):
    snapshot, names = _legacy_package(tmp_path)
    result = adapt_legacy_calibration_snapshot(snapshot, source_names=names)
    assert result["status"] == "ABSTAIN_C_LEGACY_OPT_IN"
    assert result["source_denominator"]["planned"] == 4
    assert result["source_denominator"]["valid"] == 0
    assert result["physical_score"] is None


def test_archived_default_measurement_split_is_not_reinterpreted_as_calibration(tmp_path):
    snapshot, names = _legacy_package(tmp_path, include_calibration_opt_in=False)
    result = adapt_legacy_calibration_snapshot(
        snapshot, source_names=names, calibration_opt_in=True,
    )
    assert result["status"] == "ABSTAIN_C_LEGACY_OPT_IN"
    assert "does not explicitly allow calibration" in result["reason"]
    assert result["source_denominator"]["valid"] == 4


def test_legacy_calibration_adapter_rejects_missing_selected_stencil(tmp_path):
    snapshot, names = _legacy_package(tmp_path, full_stencil=False)
    result = adapt_legacy_calibration_snapshot(
        snapshot, source_names=names, calibration_opt_in=True,
    )
    assert result["status"] == "ABSTAIN_C_LEGACY_STENCIL"
    assert "R axis" in result["reason"]


def test_legacy_calibration_adapter_rejects_wrong_kernel_and_hmc_join(tmp_path):
    tmp_path.joinpath("kernel").mkdir()
    wrong_kernel, names = _legacy_package(tmp_path / "kernel", selected_kernel=False)
    result = adapt_legacy_calibration_snapshot(
        wrong_kernel, source_names=names, calibration_opt_in=True,
    )
    assert result["status"] == "ABSTAIN_C_LEGACY_KERNEL"

    tmp_path.joinpath("trajectory").mkdir()
    wrong_join, names = _legacy_package(
        tmp_path / "trajectory", wrong_trajectory=True,
    )
    result = adapt_legacy_calibration_snapshot(
        wrong_join, source_names=names, calibration_opt_in=True,
    )
    assert result["status"] == "ABSTAIN_C_LEGACY_PROVENANCE"
    assert "trajectory/raw-gauge" in result["reason"]


def test_legacy_calibration_adapter_closes_missing_and_failed_file_denominators(tmp_path):
    snapshot, names = _legacy_package(tmp_path)
    missing = capture_source_snapshot(
        tmp_path,
        dict(snapshot.expected_hashes),
        {role: filename for role, filename in _ROLES.items() if role != "input_manifest"},
    )
    result = adapt_legacy_calibration_snapshot(
        missing, source_names=names, calibration_opt_in=True,
    )
    assert result["status"] == "ABSTAIN_C_LEGACY_SOURCE_BYTES"
    assert result["source_denominator"] == {
        "unit": "captured legacy input file", "planned": 4,
        "valid": 3, "failed": 0, "abstained": 1,
    }

    (tmp_path / _ROLES["producer_commit"]).write_bytes(b"corrupt bytes")
    failed = capture_source_snapshot(
        tmp_path,
        dict(snapshot.expected_hashes),
        {role: filename for role, filename in _ROLES.items()},
    )
    result = adapt_legacy_calibration_snapshot(
        failed, source_names=names, calibration_opt_in=True,
    )
    assert result["status"] == "ABSTAIN_C_LEGACY_SOURCE_BYTES"
    assert result["source_denominator"] == {
        "unit": "captured legacy input file", "planned": 4,
        "valid": 3, "failed": 1, "abstained": 0,
    }


def test_legacy_calibration_adapter_rechecks_transaction_hashes_after_capture(tmp_path):
    _snapshot, names = _legacy_package(tmp_path)
    commit_path = tmp_path / _ROLES["producer_commit"]
    commit = json.loads(commit_path.read_text())
    commit["npz_sha256"] = "0" * 64
    commit_path.write_text(json.dumps(commit, sort_keys=True, separators=(",", ":")))
    changed_snapshot = _recapture(tmp_path)

    result = adapt_legacy_calibration_snapshot(
        changed_snapshot, source_names=names, calibration_opt_in=True,
    )

    assert result["status"] == "ABSTAIN_C_LEGACY_COMMIT"
    assert "does not bind" in result["reason"]
    assert result["source_denominator"]["valid"] == 4
