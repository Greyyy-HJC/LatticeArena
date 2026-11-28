from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json

import numpy as np
import pytest

from core.source_integrity import capture_source_snapshot
from tasks.wilson_loop import f_lane_raw_matrix_snapshot_diagnostic as diagnostic


def _transaction(tmp_path, *, split="synthetic_calibration_open", mutate_npz=None,
                 manifest_overrides=None):
    arrays = {
        "block_ids": np.arange(8, dtype=np.int64),
        "correlator_matrices": diagnostic._synthetic_matrices(8),
        "operator_basis_ids": np.asarray(diagnostic.OPERATOR_BASIS_IDS, dtype="U32"),
        "R_values": np.asarray(diagnostic.R_VALUES, dtype=np.int64),
        "time_slices": np.asarray(diagnostic.TIME_SLICES, dtype=np.int64),
        "split_id": np.asarray(split),
    }
    if mutate_npz is not None:
        mutate_npz(arrays)
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    npz_bytes = stream.getvalue()
    definition = (
        "C_ij(R,t)=<O_i(R,t) O_j^dagger(R,0)>; rows are sink operator i, "
        "columns are source operator j; all cross terms must be measured"
    )
    manifest = {
        "schema": diagnostic.SCHEMA,
        "classification": diagnostic.CLASSIFICATION,
        "split_id": split,
        "geometry": [8, 8, 8, 16],
        "action": "synthetic_forward_fixture",
        "boundary_conditions": "periodic_fixture",
        "temporal_extent": 16,
        "operator_basis_ids": list(diagnostic.OPERATOR_BASIS_IDS),
        "R_values": list(diagnostic.R_VALUES),
        "time_slices": list(diagnostic.TIME_SLICES),
        "matrix_definition": definition,
        "matrix_npz_sha256": sha256(npz_bytes).hexdigest(),
        "time_units": "lattice_time_slices",
        "block_unit": "configuration_block",
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (tmp_path / "matrix.npz").write_bytes(npz_bytes)
    (tmp_path / "manifest.json").write_bytes(manifest_bytes)
    expected = {
        "manifest_bytes": sha256(manifest_bytes).hexdigest(),
        "matrix_bytes": sha256(npz_bytes).hexdigest(),
    }
    paths = {"manifest_bytes": "manifest.json", "matrix_bytes": "matrix.npz"}
    snapshot = capture_source_snapshot(tmp_path, expected, paths)
    return snapshot, manifest, arrays


def _read(snapshot, split="synthetic_calibration_open"):
    return diagnostic.read_raw_matrix_snapshot(
        snapshot,
        source_names={"manifest": "manifest_bytes", "matrix_npz": "matrix_bytes"},
        split_id=split,
    )


def test_reader_projects_from_immutable_snapshot_and_reports_no_physical_score(tmp_path):
    snapshot, _, _ = _transaction(tmp_path)
    (tmp_path / "matrix.npz").write_bytes(b"replacement after capture")
    (tmp_path / "manifest.json").write_bytes(b"replacement after capture")
    result = _read(snapshot)

    assert result["status"] == "RAW_MATRIX_GEVP_PROJECTION_ONLY"
    assert result["certified"] is False and result["physical_score"] is None
    assert result["projected_correlators"].shape == (8, 2, 7)
    assert result["denominators"]["R_time_values"] == {
        "planned": 14, "valid": 14, "failed": 0, "abstained": 0,
    }
    assert result["denominators"]["physical_q_interval_or_certification"]["abstained"] == 1
    assert all(abs(row["metric_norm"] - 1) < 1e-9 for row in result["matrix_condition"].values())


def test_reader_rejects_closed_split(tmp_path):
    snapshot, _, _ = _transaction(tmp_path, split="calibration_validation_holdout")
    with pytest.raises(ValueError, match="closed gap/validation/holdout"):
        _read(snapshot, split="calibration_validation_holdout")


def test_reader_rejects_reordered_operator_axis(tmp_path):
    def reorder(arrays):
        arrays["operator_basis_ids"] = arrays["operator_basis_ids"][[1, 0, 2]]
    snapshot, _, _ = _transaction(tmp_path, mutate_npz=reorder)
    with pytest.raises(ValueError, match="operator axis is missing, reordered"):
        _read(snapshot)


def test_reader_rejects_geometry_temporal_extent_mismatch(tmp_path):
    snapshot, _, _ = _transaction(tmp_path, manifest_overrides={"geometry": [8, 8, 8, 12]})
    with pytest.raises(ValueError, match="temporal_extent must match geometry"):
        _read(snapshot)


def test_reader_does_not_require_each_noisy_block_projection_to_be_positive(tmp_path):
    def one_negative_t0_block(arrays):
        arrays["correlator_matrices"][0, 0, 0] *= -0.5
    snapshot, _, _ = _transaction(tmp_path, mutate_npz=one_negative_t0_block)
    result = _read(snapshot)
    projected = result["projected_correlators"][:, 0, 0]
    assert projected[0] < 0
    assert projected.mean() > 0


@pytest.mark.parametrize("kind, message", [
    ("non_hermitian", "must be Hermitian"),
    ("nonfinite", "must be finite"),
    ("singular_metric", "not positive definite"),
])
def test_reader_rejects_invalid_matrix_values(tmp_path, kind, message):
    def mutate(arrays):
        matrices = arrays["correlator_matrices"]
        if kind == "non_hermitian":
            matrices[0, 0, 0, 0, 1] += 0.1j
        elif kind == "nonfinite":
            matrices[0, 0, 0, 0, 0] = complex(float("nan"), 0)
        elif kind == "singular_metric":
            matrices[:, 0, 0, :, :] = 0
            matrices[:, 0, 1, :, :] = 0
    snapshot, _, _ = _transaction(tmp_path, mutate_npz=mutate)
    with pytest.raises(ValueError, match=message):
        _read(snapshot)


def test_reader_requires_complete_snapshot_source_map(tmp_path):
    snapshot, _, _ = _transaction(tmp_path)
    with pytest.raises(ValueError, match="map exactly manifest and matrix_npz"):
        diagnostic.read_raw_matrix_snapshot(snapshot, source_names={"manifest": "manifest_bytes"},
                                            split_id="synthetic_calibration_open")


def test_reader_rejects_empty_split_id(tmp_path):
    snapshot, _, _ = _transaction(tmp_path)
    with pytest.raises(ValueError, match="split_id must be a nonempty string"):
        diagnostic.read_raw_matrix_snapshot(
            snapshot,
            source_names={"manifest": "manifest_bytes", "matrix_npz": "matrix_bytes"},
            split_id="",
        )


def test_report_hashes_figure_and_keeps_certification_gate_abstained(tmp_path):
    output = tmp_path / "report"
    summary = diagnostic.build(output)
    assert summary["status"] == "RAW_MATRIX_GEVP_PROJECTION_ONLY"
    assert summary["physical_score"] is None
    assert summary["physical_measurement_inputs"] == {
        "unit": "raw Wilson correlator or independent spectrum receipt",
        "planned": 0, "valid": 0, "failed": 0, "abstained": 0,
    }
    assert (output / "projected_correlators.png").is_file()
    assert (output / "README.md").is_file()
    hashes = json.loads((output / "artifact_hashes.json").read_text())["files"]
    for name, expected in hashes.items():
        assert sha256((output / name).read_bytes()).hexdigest() == expected
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        diagnostic.build(output)
