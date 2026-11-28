from __future__ import annotations

import numpy as np
import pytest

from tasks.pion_projection.bounded_block_risk_setwise import (
    bounded_block_risk_setwise,
)


def test_single_block_score_set_exposes_target_induced_ranking_flip():
    args = dict(
        reference_values=[0.0],
        candidate_values=[1.0],
        valid_blocks=np.asarray([True]),
        score_scale_R=1.0,
    )
    full = bounded_block_risk_setwise(target_interval=[0.0, 1.0], **args)
    left = bounded_block_risk_setwise(target_interval=[0.0, 0.4], **args)
    right = bounded_block_risk_setwise(target_interval=[0.6, 1.0], **args)

    assert full["normalized_score_interval"] == pytest.approx([-1.0, 1.0])
    assert full["ranking"] == "ordering_unresolved_over_target_set"
    assert left["normalized_score_interval"] == pytest.approx([-1.0, -0.2])
    assert left["ranking"] == "reference_better_for_all_targets_in_set"
    assert right["normalized_score_interval"] == pytest.approx([0.2, 1.0])
    assert right["ranking"] == "candidate_better_for_all_targets_in_set"
    assert full["physical_score"] is None and not full["certified"]
    assert full["confidence_coverage"] is None


def test_exact_breakpoint_range_matches_dense_target_scan_and_keeps_failures():
    reference = np.asarray([0.0, 2.0, 1.0, -1.0, np.nan])
    candidate = np.asarray([1.0, 0.0, 1.5, 1.0, np.nan])
    valid = np.asarray([True, True, True, True, False])
    interval = [-0.2, 1.7]
    scale = 1.5
    result = bounded_block_risk_setwise(
        reference, candidate,
        valid_blocks=valid,
        target_interval=interval,
        score_scale_R=scale,
    )

    targets = np.linspace(interval[0], interval[1], 100_001)
    per_target_scores = []
    for target in targets:
        raw = ((target - reference[valid]) ** 2
               - (target - candidate[valid]) ** 2) / scale**2
        scores = np.full(len(reference), -1.0)
        scores[valid] = np.clip(raw, -1.0, 1.0)
        per_target_scores.append(scores.mean())
    dense_min, dense_max = min(per_target_scores), max(per_target_scores)
    exact_min, exact_max = result["normalized_score_interval"]

    assert exact_min <= dense_min + 1e-12
    assert exact_max >= dense_max - 1e-12
    assert exact_min == pytest.approx(dense_min, abs=2e-5)
    assert exact_max == pytest.approx(dense_max, abs=2e-5)
    assert result["denominators"] == {
        "planned_blocks": 5, "valid_blocks": 4,
        "failed_blocks": 1, "abstained_blocks": 0,
    }
    assert result["result_denominators"] == {
        "planned": 1, "valid": 1, "failed": 0, "abstained": 0,
    }


def test_failed_blocks_remain_minus_one_in_planned_mean():
    result = bounded_block_risk_setwise(
        [0.0, np.nan], [1.0, np.nan],
        valid_blocks=np.asarray([True, False]),
        target_interval=[1.0, 1.0],
        score_scale_R=1.0,
    )

    assert result["normalized_score_interval"] == pytest.approx([0.0, 0.0])
    assert result["denominators"]["planned_blocks"] == 2
    assert result["denominators"]["failed_blocks"] == 1


def test_extreme_finite_observables_do_not_overflow_score_evaluation():
    result = bounded_block_risk_setwise(
        [1e150], [-1e150],
        valid_blocks=np.asarray([True]),
        target_interval=[0.0, 1e150],
        score_scale_R=1e150,
    )

    assert result["normalized_score_interval"] == pytest.approx([-1.0, 0.0])


def test_all_failed_blocks_abstain_and_malformed_inputs_reject():
    result = bounded_block_risk_setwise(
        [np.nan], [np.nan],
        valid_blocks=np.asarray([False]),
        target_interval=[0.0, 1.0],
        score_scale_R=1.0,
    )
    assert result["status"] == "ABSTAIN_NO_VALID_BLOCKS"
    assert result["normalized_score_interval"] is None
    assert result["physical_score"] is None

    with pytest.raises(ValueError, match="target_interval"):
        bounded_block_risk_setwise(
            [0.0], [1.0], valid_blocks=np.asarray([True]),
            target_interval=[1.0, 0.0], score_scale_R=1.0,
        )
    with pytest.raises(ValueError, match="boolean array"):
        bounded_block_risk_setwise(
            [0.0], [1.0], valid_blocks=np.asarray([1]),
            target_interval=[0.0, 1.0], score_scale_R=1.0,
        )


@pytest.mark.parametrize("invalid_values", [
    np.asarray([True]),
    np.asarray(["1.0"]),
    np.asarray([1.0 + 2.0j]),
    [10**400],
])
@pytest.mark.parametrize("input_name", ["reference_values", "candidate_values"])
def test_prediction_inputs_reject_lossy_numeric_coercions(invalid_values, input_name):
    arguments = {
        "reference_values": [0.0],
        "candidate_values": [1.0],
        "valid_blocks": np.asarray([True]),
        "target_interval": [0.0, 1.0],
        "score_scale_R": 1.0,
    }
    arguments[input_name] = invalid_values

    with pytest.raises(ValueError, match=input_name):
        bounded_block_risk_setwise(**arguments)


@pytest.mark.parametrize("target_interval,score_scale_R", [
    ([0, (1 << 53) + 1], 1.0),
    ([0.0, 1.0], (1 << 53) + 1),
])
def test_scalar_inputs_reject_inexact_large_integer_coercions(
    target_interval, score_scale_R,
):
    with pytest.raises(ValueError, match="not exactly representable"):
        bounded_block_risk_setwise(
            [0.0], [1.0],
            valid_blocks=np.asarray([True]),
            target_interval=target_interval,
            score_scale_R=score_scale_R,
        )
