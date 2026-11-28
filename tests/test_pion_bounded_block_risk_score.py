import math

import numpy as np
import pytest

from tasks.pion_projection.bounded_block_risk_score import (
    bounded_block_scores,
    conditional_block_risk_lcb,
)


def calibration():
    return dict(
        score_scale_R=2.0,
        chromatic_number=2,
        family=2,
        alpha_score=0.04,
        alpha_anchor=0.01,
        alpha_total=0.05,
        relative_anchor_bound_B_over_R=0.03,
    )


def test_clips_valid_scores_and_keeps_failed_blocks_in_planned_mean():
    result = conditional_block_risk_lcb(
        [8.0, -8.0, 2.0, np.nan],
        valid_blocks=np.array([True, True, True, False]),
        **calibration(),
    )

    assert result["status"] == "CONDITIONAL_BOUND_ONLY"
    assert result["physical_score"] is None
    assert result["normalized_block_scores"] == [1.0, -1.0, 0.5, -1.0]
    assert result["denominators"] == {
        "planned_blocks": 4,
        "valid_blocks": 3,
        "failed_blocks": 1,
        "abstained_blocks": 0,
    }
    assert result["result_unit"] == "one planned candidate-reference comparison"
    assert result["result_denominators"] == {
        "planned": 1, "valid": 1, "failed": 0, "abstained": 0,
    }
    assert result["valid_block_mean"] == pytest.approx(1.0 / 6.0)
    assert result["planned_block_mean"] == pytest.approx(-0.125)
    expected_lcb = (
        -0.125
        - math.sqrt(2 * 2 * math.log(2 / 0.04) / 4)
        - 2 * 0.03
    )
    assert result["normalized_lower_bound"] == pytest.approx(expected_lcb)


def test_all_failed_blocks_abstain_without_a_lower_bound():
    result = conditional_block_risk_lcb(
        [np.nan, np.nan],
        valid_blocks=np.array([False, False]),
        **calibration(),
    )

    assert result["status"] == "ABSTAIN_NO_VALID_BLOCKS"
    assert result["normalized_lower_bound"] is None
    assert result["planned_block_mean"] == -1.0
    assert result["denominators"] == {
        "planned_blocks": 2,
        "valid_blocks": 0,
        "failed_blocks": 2,
        "abstained_blocks": 0,
    }
    assert result["result_denominators"] == {
        "planned": 1, "valid": 0, "failed": 0, "abstained": 1,
    }
    assert result["physical_score"] is None


def test_batched_score_transform_uses_last_axis_and_preserves_failures():
    raw = np.array([
        [[8.0, 2.0, np.nan], [-4.0, 0.0, 6.0]],
        [[-2.0, 4.0, 0.0], [2.0, np.nan, 2.0]],
    ])
    valid = np.array([
        [[True, True, False], [True, True, True]],
        [[True, True, True], [True, False, True]],
    ])
    scores = bounded_block_scores(raw, valid_blocks=valid, score_scale_R=2.0)
    np.testing.assert_array_equal(scores, [
        [[1.0, 0.5, -1.0], [-1.0, 0.0, 1.0]],
        [[-0.5, 1.0, 0.0], [0.5, -1.0, 0.5]],
    ])


@pytest.mark.parametrize("differences, mask, kwargs", [
    ([1.0, 2.0], np.array([True]), {}),
    ([np.nan], np.array([True]), {}),
    ([1.0], np.array([1]), {}),
    ([1.0], np.array([True]), {"score_scale_R": 0.0}),
    ([1.0], np.array([True]), {"score_scale_R": 1e308}),
    ([1.0], np.array([True]), {"score_scale_R": 1e-300}),
    ([1.0], np.array([True]), {"alpha_score": 0.04, "alpha_anchor": 0.02}),
])
def test_malformed_scores_masks_or_calibration_contract_reject(differences, mask, kwargs):
    args = calibration()
    args.update(kwargs)
    with pytest.raises(ValueError):
        conditional_block_risk_lcb(differences, valid_blocks=mask, **args)


@pytest.mark.parametrize(
    "differences",
    [
        np.array([1.0 + 2.0j]),
        [10**400],
        ["1.0"],
        [True],
    ],
)
def test_score_inputs_reject_lossy_or_unrepresentable_real_coercions(differences):
    with pytest.raises(ValueError):
        bounded_block_scores(
            differences,
            valid_blocks=np.ones(np.shape(differences), dtype=bool),
            score_scale_R=2.0,
        )
    with pytest.raises(ValueError):
        conditional_block_risk_lcb(
            differences,
            valid_blocks=np.ones(np.shape(differences), dtype=bool),
            **calibration(),
        )
