import numpy as np
import pytest

from tasks.optimize_gaussian.bounded_block_risk import (
    ALPHA_TOTAL,
    BOUND_EVENTS,
    DELTA_PER_BOUND,
    R_STAR,
    bounded_block_risk_decision,
    bounded_block_scores,
)
from tasks.optimize_gaussian.d06_bounded_block_risk_diagnostic import build


def test_block_score_clips_squared_deviation_and_averages_time_within_block():
    q = np.ones((2, 3, 4), dtype=float)
    q[0, 0] = [0.5, 1.5, 3.0, -4.0]
    scores = bounded_block_scores(q)
    assert scores.shape == (2, 3)
    assert scores[0, 0] == pytest.approx((0.25 + 0.25 + 1.0 + 1.0) / 4)
    assert np.all(scores[1] == 0)


def test_block_score_rejects_bad_shape_nonfinite_complex_and_handles_huge_finite_q():
    with pytest.raises(ValueError, match="shape"):
        bounded_block_scores(np.ones((5, 3)))
    with pytest.raises(ValueError, match="non-finite"):
        bounded_block_scores(np.full((2, 3, 4), np.nan))
    with pytest.raises(ValueError, match="real-valued"):
        bounded_block_scores(np.ones((2, 3, 4), dtype=complex) * (1 + 1j))
    huge = np.full((2, 3, 4), 1e300)
    assert np.all(bounded_block_scores(huge) == 1.0)


def test_failed_assigned_arm_is_scored_at_one_and_keeps_the_block():
    q = np.full((10, 3, 4), 1.2)
    failures = np.zeros((10, 3), dtype=bool)
    failures[0, 1] = True
    q[0, 1] = np.nan  # a failed solver has no q result
    scores = bounded_block_scores(q, arm_failure_mask=failures)
    assert scores[0, 1] == 1.0
    assert np.all(scores[1:, 1] == pytest.approx(0.04))

    decision = bounded_block_risk_decision(
        scores, arm_failure_mask=failures,
    )
    assert decision["planned_blocks"] == decision["scored_blocks"] == 10
    assert decision["failed_blocks"] == 0  # no observation was dropped
    assert decision["arm_failure_counts"]["same_budget_slsqp"] == 1
    assert decision["algorithm_failure_blocks_any_arm"] == 1
    assert decision["comparisons"]["same_budget_slsqp"]["empirical_risk"] == pytest.approx(0.136)


def test_missing_q_without_declared_algorithm_failure_abstains_by_rejection():
    q = np.full((2, 3, 4), 1.2)
    q[0, 1, 0] = np.nan
    with pytest.raises(ValueError, match="outside a failed arm"):
        bounded_block_scores(q)


def test_failure_mask_shape_and_dropped_failure_score_are_rejected():
    q = np.full((4, 3, 4), 1.2)
    with pytest.raises(ValueError, match="boolean with shape"):
        bounded_block_scores(q, arm_failure_mask=np.ones((4, 3), dtype=int))
    failures = np.zeros((4, 3), dtype=bool)
    failures[0, 2] = True
    scores = bounded_block_scores(q)  # does not carry the failure penalty
    with pytest.raises(ValueError, match="maximum loss score 1"):
        bounded_block_risk_decision(scores, arm_failure_mask=failures)


def test_alpha_split_covers_two_contrasts_and_shared_positive_denominator():
    assert BOUND_EVENTS == 3
    assert DELTA_PER_BOUND == pytest.approx(ALPHA_TOTAL / 3)
    assert BOUND_EVENTS * DELTA_PER_BOUND == pytest.approx(ALPHA_TOTAL)


def test_empirical_bernstein_releases_large_fixed_effect_and_reports_no_physical_score():
    scores = np.tile([0.04, 0.075, 0.08], (20000, 1))
    decision = bounded_block_risk_decision(scores)
    assert decision["status"] == "PASS_BOTH_EFFECT_THRESHOLDS"
    assert decision["trainable_w"]["denominator_proved_positive"]
    assert decision["comparisons"]["same_budget_slsqp"]["contrast"]["lower_bound"] > 0
    assert decision["comparisons"]["degree4_chebyshev"]["contrast"]["lower_bound"] > 0
    assert decision["physical_score"] is None


@pytest.mark.parametrize("method", ["hoeffding", "empirical_bernstein"])
def test_boundary_ratio_does_not_release(method):
    scores = np.tile([0.05, R_STAR * 0.05, R_STAR * 0.05], (20000, 1))
    decision = bounded_block_risk_decision(scores, method=method)
    assert decision["status"] == "NO_FAMILYWISE_RELEASE"
    assert not decision["comparisons"]["same_budget_slsqp"]["threshold_pass"]
    assert not decision["comparisons"]["degree4_chebyshev"]["threshold_pass"]


def test_zero_or_unresolved_trainable_risk_abstains_instead_of_forming_ratio():
    scores = np.tile([0.0, 0.8, 0.9], (1000, 1))
    decision = bounded_block_risk_decision(scores)
    assert decision["status"] == "ABSTAIN_TRAINABLE_RISK_NOT_SEPARATED_FROM_ZERO"
    assert decision["trainable_w"]["empirical_risk"] == 0
    assert decision["comparisons"]["same_budget_slsqp"]["empirical_ratio_if_defined"] is None
    assert decision["valid_blocks"] == 0
    assert decision["abstained_blocks"] == 1000


@pytest.mark.parametrize(
    "scores",
    [
        np.full((3, 3), 0.2 + 0.1j),
        [[10**400, 0, 0]],
        [["0.2", "0.2", "0.2"]],
        [[True, False, True]],
    ],
)
def test_risk_decision_rejects_lossy_or_unrepresentable_coercions(scores):
    with pytest.raises(ValueError):
        bounded_block_risk_decision(scores)


def test_decision_rejects_nonfinite_or_out_of_support_scores():
    with pytest.raises(ValueError, match=r"finite and lie in \[0, 1\]"):
        bounded_block_risk_decision(np.full((10, 3), np.nan))
    with pytest.raises(ValueError, match=r"finite and lie in \[0, 1\]"):
        bounded_block_risk_decision(np.full((10, 3), 1.1))


def test_small_report_has_full_denominator_and_iid_plus_dependence_cells(tmp_path):
    out = tmp_path / "bounded-risk-report"
    protocol = build(out, replicates=100, dependence_replicates=100,
                     seed=71, batch_size=16)
    assert protocol["iid_stress_design"]["panel_denominator_per_method"] == 3 * 3 * 4 * 100
    assert protocol["dependence_negative_control"]["panel_denominator_per_method"] == 6 * 100
    assert protocol["physical_score"] is None
    assert (out / "coverage_release_power.png").is_file()
    assert (out / "artifact_hashes.json").is_file()
