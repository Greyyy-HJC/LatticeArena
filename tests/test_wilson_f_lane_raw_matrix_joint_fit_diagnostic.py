import csv
import hashlib
import json

from tasks.wilson_loop import f_lane_raw_matrix_joint_fit_diagnostic as diagnostic


def test_raw_snapshot_matrix_bytes_feed_candidate_fit_without_path_reread(tmp_path):
    snapshot, expected, paths = diagnostic._write_synthetic_snapshot(tmp_path, n_blocks=80)
    (tmp_path / paths["synthetic_matrix_npz"]).write_bytes(b"changed after capture")
    (tmp_path / paths["synthetic_manifest"]).write_bytes(b"changed after capture")
    result = diagnostic.analyze_snapshot(
        snapshot,
        source_names={
            "manifest": "synthetic_manifest",
            "matrix_npz": "synthetic_matrix_npz",
        },
        split_id=diagnostic.SPLIT_ID,
    )
    assert result["source_hashes"] == {
        "manifest": expected["synthetic_manifest"],
        "matrix_npz": expected["synthetic_matrix_npz"],
    }
    assert result["denominators"]["joint_matrix_fits"] == {
        "planned": 12, "valid": 12, "failed": 0, "abstained": 0,
    }
    assert result["denominators"]["kernel_fit_variants"] == {
        "planned": 36, "valid": 36, "failed": 0, "abstained": 0,
    }
    assert result["denominators"]["q_candidate_cells"] == {
        "planned": 72, "valid": 72, "failed": 0, "abstained": 0,
    }
    assert result["not_connected_to_spectral_q_reader"]
    assert result["not_connected_to_evidence_gate"]
    assert result["physical_score"] is None


def test_insufficient_blocks_abstain_without_dropping_variants(tmp_path):
    output = tmp_path / "small-report"
    summary = diagnostic.build(output, n_blocks=8)
    assert summary["denominators"]["joint_matrix_fits"] == {
        "planned": 12, "valid": 0, "failed": 0, "abstained": 12,
    }
    assert summary["denominators"]["kernel_fit_variants"] == {
        "planned": 36, "valid": 0, "failed": 0, "abstained": 36,
    }
    assert summary["denominators"]["q_candidate_cells"] == {
        "planned": 72, "valid": 0, "failed": 0, "abstained": 72,
    }


def test_report_binds_fit_variants_figure_and_uncertified_gate(tmp_path):
    output = tmp_path / "report"
    summary = diagnostic.build(output, n_blocks=80)
    assert summary["status"] == "RAW_MATRIX_JOINT_FIT_CANDIDATE_ONLY"
    assert summary["certified"] is False and summary["physical_score"] is None
    assert summary["lane_status"] == "GROUND_STATE_FORCE_UNCERTIFIED"
    assert summary["not_connected_to_spectral_q_reader"]
    assert summary["not_connected_to_evidence_gate"]
    for name in ("fit_status.csv", "fit_variants.csv", "q_candidates.csv"):
        with (output / name).open() as stream:
            rows = list(csv.DictReader(stream))
        assert rows
    with (output / "q_candidates.csv").open() as stream:
        q_rows = list(csv.DictReader(stream))
    assert len(q_rows) == 72
    assert all(row["interval_status"] == "UNCALIBRATED_DELTA_METHOD" for row in q_rows)
    hashes = json.loads((output / "artifact_hashes.json").read_text())["files"]
    for name, expected in hashes.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["method_boundary"]["not_connected_to_spectral_q_reader"]
    assert protocol["method_boundary"]["not_connected_to_evidence_gate"]
    assert (output / "q_fit_variants.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
