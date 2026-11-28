import math

import numpy as np
import pytest

from tasks.optimize_gaussian.bounded_block_risk_markov import (
    bounded_block_risk_markov_decision,
    markov_bernstein_radius,
)


def test_markov_radius_matches_inverted_theorem_formula():
    n = 400
    gamma = 0.2
    alpha = 0.01
    lower, upper = -10.0 / 9.0, 1.0
    width = upper - lower
    variance = width**2 / 4.0
    centered = width
    log_tail = math.log(2.0 / alpha)
    linear = 20.0 * centered * log_tail
    variance_term = 8.0 * log_tail * (n + 1.0 / gamma) * variance
    expected = (
        linear + math.sqrt(linear**2 + 4.0 * gamma * variance_term)
    ) / (2.0 * gamma * n)
    assert markov_bernstein_radius(
        blocks=n, gamma_ps_lower=gamma, alpha=alpha,
        lower=lower, upper=upper,
    ) == pytest.approx(expected, rel=1e-14)


def test_markov_radius_is_monotone_in_sample_size_gap_and_alpha():
    common = dict(alpha=0.01, lower=0.0, upper=1.0)
    small_n = markov_bernstein_radius(blocks=100, gamma_ps_lower=0.2, **common)
    large_n = markov_bernstein_radius(blocks=1000, gamma_ps_lower=0.2, **common)
    weak_gap = markov_bernstein_radius(blocks=1000, gamma_ps_lower=0.02, **common)
    stricter_alpha = markov_bernstein_radius(
        blocks=1000, gamma_ps_lower=0.2, alpha=0.001,
        lower=0.0, upper=1.0,
    )
    assert large_n < small_n
    assert weak_gap > large_n
    assert stricter_alpha > large_n


def test_radius_scales_linearly_with_support_width():
    unit = markov_bernstein_radius(
        blocks=250, gamma_ps_lower=0.3, alpha=0.02, lower=0.0, upper=1.0,
    )
    width = 10.0 / 9.0 + 1.0
    contrast = markov_bernstein_radius(
        blocks=250, gamma_ps_lower=0.3, alpha=0.02,
        lower=-10.0 / 9.0, upper=1.0,
    )
    assert contrast == pytest.approx(unit * width, rel=1e-14)


def test_missing_gap_receipt_abstains_and_keeps_scored_denominator():
    scores = np.full((12, 3), 0.2)
    result = bounded_block_risk_markov_decision(
        scores, gamma_ps_lower=None, burn_in_blocks=8,
    )
    assert result["status"] == "ABSTAIN_MISSING_BLOCK_CHAIN_GAP_RECEIPT"
    assert result["planned_scored_blocks"] == 12
    assert result["scored_blocks"] == 12
    assert result["decision_denominators"]["abstained"] == 1
    assert result["physical_score"] is None
    assert result["certified"] is False


def test_tv_budget_is_paid_once_before_three_event_family_bound():
    scores = np.full((12, 3), 0.2)
    result = bounded_block_risk_markov_decision(
        scores, gamma_ps_lower=0.5, tv_bound_after_burn_in=0.05,
    )
    assert result["status"] == "ABSTAIN_TV_BUDGET_EXHAUSTS_ALPHA"
    assert result["tv_probability_cost"] == pytest.approx(0.05)
    assert result["decision_denominators"]["abstained"] == 1


def test_strong_frozen_effect_can_pass_conditionally_but_never_certifies_physics():
    scores = np.tile([0.05, 0.20, 0.20], (100_000, 1))
    result = bounded_block_risk_markov_decision(
        scores, gamma_ps_lower=1.0, tv_bound_after_burn_in=0.0,
    )
    assert result["status"] == "CONDITIONAL_MARKOV_PASS_BOTH_COMPARISONS"
    assert result["alpha_per_event_after_tv"] == pytest.approx(0.05 / 3)
    assert all(item["lower_bound"] > 0 for item in result["comparisons"].values())
    assert result["bounds"]["trainable_w_mean"]["lower_bound"] > 0
    assert result["certified"] is False
    assert result["physical_score"] is None


def test_algorithm_failures_remain_in_denominator_at_maximum_loss():
    scores = np.tile([0.05, 0.20, 0.20], (100_000, 1))
    failures = np.zeros(scores.shape, dtype=bool)
    failures[0, 0] = True
    scores[0, 0] = 1.0
    result = bounded_block_risk_markov_decision(
        scores, gamma_ps_lower=1.0, arm_failure_mask=failures,
    )
    assert result["planned_scored_blocks"] == 100_000
    assert result["scored_blocks"] == 100_000
    assert result["algorithm_failure_counts"]["trainable_w"] == 1


@pytest.mark.parametrize("gamma", [0.0, -0.1, 1.1, float("nan")])
def test_invalid_pseudo_gap_is_rejected(gamma):
    with pytest.raises(ValueError):
        markov_bernstein_radius(
            blocks=100, gamma_ps_lower=gamma, alpha=0.01,
            lower=0.0, upper=1.0,
        )


@pytest.mark.parametrize(
    "scores,failure_mask",
    [
        (np.full((3, 2), 0.2), None),
        (np.full((3, 3), np.nan), None),
        (np.full((3, 3), 1.1), None),
        (np.full((3, 3), 0.2), np.zeros((3, 3), dtype=int)),
        (np.full((3, 3), 0.2), np.array([[True, False, False]] * 2)),
    ],
)
def test_invalid_scores_or_failure_mask_are_rejected(scores, failure_mask):
    with pytest.raises(ValueError):
        bounded_block_risk_markov_decision(
            scores, gamma_ps_lower=0.5, arm_failure_mask=failure_mask,
        )


def test_failed_arm_must_have_score_one():
    scores = np.full((3, 3), 0.2)
    failures = np.zeros(scores.shape, dtype=bool)
    failures[0, 1] = True
    with pytest.raises(ValueError, match="maximum loss score 1"):
        bounded_block_risk_markov_decision(
            scores, gamma_ps_lower=0.5, arm_failure_mask=failures,
        )


@pytest.mark.parametrize(
    "scores",
    [
        np.full((3, 3), 0.2 + 0.1j),
        [[10**400, 0, 0]],
        [["0.2", "0.2", "0.2"]],
        [[True, False, True]],
    ],
)
def test_score_array_rejects_lossy_or_unrepresentable_coercions(scores):
    with pytest.raises(ValueError):
        bounded_block_risk_markov_decision(scores, gamma_ps_lower=0.5)
