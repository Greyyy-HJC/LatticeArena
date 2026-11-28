from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from core.source_integrity import capture_source_snapshot
from tasks.optimize_gaussian.bounded_block_risk import ARMS, TIMES
from tasks.optimize_gaussian.source_reader import (
    SCHEMA, read_bounded_block_risk_snapshot, read_json_object_snapshot,
)


def _npz_payload(*, split_id="D06-calibration-open", arm_order=None, invalid_block=None):
    n_blocks = 12
    q = np.full((n_blocks, len(ARMS), len(TIMES)), 0.8, dtype=np.float64)
    failures = np.zeros((n_blocks, len(ARMS)), dtype=bool)
    failures[0, 1] = True
    q[0, 1, :] = np.nan
    if invalid_block is not None:
        q[invalid_block, 2, 0] = np.nan
    stream = BytesIO()
    np.savez_compressed(
        stream,
        schema=np.asarray(SCHEMA),
        split_id=np.asarray(split_id),
        block_ids=np.arange(100, 100 + n_blocks, dtype=np.int64),
        arm_order=np.asarray(ARMS if arm_order is None else arm_order),
        times=np.asarray(TIMES, dtype=np.int64),
        q_by_block_arm_time=q,
        arm_failure_mask=failures,
    )
    return stream.getvalue()


def _snapshot(tmp_path, payload):
    (tmp_path / "scores.npz").write_bytes(payload)
    return capture_source_snapshot(
        tmp_path,
        {"paired_scores": sha256(payload).hexdigest()},
        {"paired_scores": "scores.npz"},
    )


def test_optimize_d06_reader_parses_captured_paired_scores_and_penalizes_failed_arm(tmp_path):
    payload = _npz_payload()
    snapshot = _snapshot(tmp_path, payload)
    (tmp_path / "scores.npz").write_bytes(b"changed after capture")

    result = read_bounded_block_risk_snapshot(
        snapshot, source_name="paired_scores", split_id="D06-calibration-open",
    )

    assert result["status"] == "ARRAY_REDUCED_NOT_CERTIFIED"
    assert result["source_read_mode"] == "immutable_snapshot"
    assert result["source_sha256"] == sha256(payload).hexdigest()
    assert result["source_denominator"] == {
        "unit": "snapshot_source_file", "planned": 1,
        "valid": 1, "failed": 0, "abstained": 0,
    }
    assert result["block_denominator"] == {
        "unit": "paired_macroblock", "planned": 12,
        "valid": 12, "failed": 0, "abstained": 0,
    }
    assert result["assigned_arm_run_denominator"] == {
        "unit": "assigned_solver_arm_run", "planned": 36,
        "valid": 35, "failed": 1, "abstained": 0,
    }
    assert result["block_scores"][0][1] == 1.0
    assert result["decision"]["physical_score"] is None
    assert result["physical_score"] is None and result["certified"] is False


def test_optimize_d06_reader_abstains_closed_split_without_falling_back_to_path(tmp_path):
    snapshot = _snapshot(tmp_path, _npz_payload())
    result = read_bounded_block_risk_snapshot(
        snapshot, source_name="paired_scores", split_id="D06-validation-sealed",
    )
    assert result["status"] == "ABSTAIN_CLOSED_SPLIT"
    assert result["source_denominator"]["planned"] == 1
    assert result["source_denominator"]["valid"] == 0
    assert result["source_denominator"]["abstained"] == 1


def test_optimize_d06_reader_rejects_arm_reordering_and_invalid_unfailed_blocks(tmp_path):
    reordered = _snapshot(tmp_path, _npz_payload(arm_order=tuple(reversed(ARMS))))
    wrong_order = read_bounded_block_risk_snapshot(
        reordered, source_name="paired_scores", split_id="D06-calibration-open",
    )
    assert wrong_order["status"] == "ABSTAIN_ARM_ORDER_MISMATCH"

    invalid = _snapshot(tmp_path, _npz_payload(invalid_block=4))
    result = read_bounded_block_risk_snapshot(
        invalid, source_name="paired_scores", split_id="D06-calibration-open",
    )
    assert result["status"] == "ABSTAIN_INVALID_PAIRED_BLOCKS"
    assert result["block_denominator"] == {
        "unit": "paired_macroblock", "planned": 12,
        "valid": 11, "failed": 1, "abstained": 0,
    }
    assert result["invalid_block_ids"] == [104]


def test_optimize_d06_reader_abstains_before_loading_inflated_npz_arrays(tmp_path):
    import numpy as np
    header = BytesIO()
    np.lib.format.write_array_header_1_0(
        header,
        {"descr": np.dtype("<f8").str, "fortran_order": False,
         "shape": (10**8,)},
    )
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("scores.npy", header.getvalue() + b"\\x00")
    snapshot = _snapshot(tmp_path, stream.getvalue())

    result = read_bounded_block_risk_snapshot(
        snapshot, source_name="paired_scores", split_id="D06-calibration-open",
    )

    assert result["status"] == "ABSTAIN_NPZ_PARSE_FAILED"
    assert result["source_denominator"]["failed"] == 1
    assert result["physical_score"] is None


@pytest.mark.parametrize(
    "payload",
    [b'{"x":1,"x":2}', b'{"x":NaN}', b'[]'],
)
def test_json_snapshot_reader_rejects_duplicate_nonfinite_and_nonobject_roots(tmp_path, payload):
    (tmp_path / "summary.json").write_bytes(payload)
    snapshot = capture_source_snapshot(
        tmp_path,
        {"summary": sha256(payload).hexdigest()},
        {"summary": "summary.json"},
    )
    with pytest.raises(ValueError, match="JSON snapshot source"):
        read_json_object_snapshot(snapshot, source_name="summary")
