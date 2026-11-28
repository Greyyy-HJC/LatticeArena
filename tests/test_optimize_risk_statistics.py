import numpy as np
import pytest

from tasks.optimize_gaussian.risk_statistics import (
    D06_BLOCK_LENGTHS,
    D06_R_STAR,
    d04_bootstrap_risk,
    d06_paired_ratio_hull,
)


def test_d04_reducer_matches_hand_calculated_mean_variance_and_bias():
    q = np.array([
        [1.0, 0.5, 1.0, 1.0],
        [2.0, 0.5, 1.0, 0.0],
        [3.0, 0.5, 2.0, 1.0],
    ])

    result = d04_bootstrap_risk(q, planned_draws=3)

    assert result["status"] == "D04_RISK_REDUCED_ARRAY_ONLY"
    assert result["times"] == [2, 3, 4, 5]
    assert result["draw_denominators"] == {
        "planned": 3, "valid": 3, "failed": 0, "abstained": 0,
    }
    assert result["result_denominators"] == {
        "planned": 1, "valid": 1, "failed": 0, "abstained": 0,
    }
    expected_contributions = [2.0, 0.25, 4.0 / 9.0, 4.0 / 9.0]
    assert [row["risk_contribution"] for row in result["per_time"]] == pytest.approx(
        expected_contributions,
    )
    assert result["risk"] == pytest.approx(sum(expected_contributions) / 4)
    assert result["physical_score"] is None
    assert result["certified"] is False


def test_d04_preserves_failed_draws_and_abstains_instead_of_dropping_them():
    q = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [1.1, 0.9, 1.0, 1.0],
        [np.nan, 1.0, 1.0, 1.0],
    ])

    result = d04_bootstrap_risk(q, planned_draws=4)

    assert result["status"] == "ABSTAIN_INCOMPLETE_OR_INVALID_Q_DRAWS"
    assert result["draw_denominators"] == {
        "planned": 4, "valid": 2, "failed": 2, "abstained": 0,
    }
    assert result["result_denominators"]["abstained"] == 1
    assert result["risk"] is None


def test_d04_nonfinite_arithmetic_abstains_without_reclassifying_finite_input_draws():
    q = np.full((2, 4), 1e308)

    result = d04_bootstrap_risk(q, planned_draws=2)

    assert result["status"] == "ABSTAIN_NONFINITE_D04_RISK_RESULT"
    assert result["draw_denominators"] == {
        "planned": 2, "valid": 2, "failed": 0, "abstained": 0,
    }
    assert result["result_denominators"]["abstained"] == 1
    assert result["risk"] is None


def _d06_inputs(ratios_by_length, *, planned=4):
    ids = np.arange(planned, dtype=np.int64)
    comp = {length: np.asarray(ratios_by_length[length], dtype=float) for length in D06_BLOCK_LENGTHS}
    cand = {length: np.ones(planned) for length in D06_BLOCK_LENGTHS}
    comp_ids = {length: ids.copy() for length in D06_BLOCK_LENGTHS}
    cand_ids = {length: ids.copy() for length in D06_BLOCK_LENGTHS}
    return comp, cand, comp_ids, cand_ids


def test_d06_pairs_each_arm_and_uses_the_worst_block_length_lower_quantile():
    comp, cand, comp_ids, cand_ids = _d06_inputs({
        2: [2.0, 2.1, 2.2, 2.3],
        5: [1.8, 1.9, 2.0, 2.1],
        10: [1.6, 1.7, 1.8, 1.9],
    })

    result = d06_paired_ratio_hull(
        comp, cand, comp_ids, cand_ids, planned_draws_per_length=4,
    )

    expected = {
        length: float(np.quantile(comp[length] / cand[length], 0.025, method="linear"))
        for length in D06_BLOCK_LENGTHS
    }
    assert result["status"] == "D06_THRESHOLD_CLEARED_ARRAY_ONLY"
    assert result["hull_lower"] == pytest.approx(min(expected.values()))
    assert result["hull_lower"] > D06_R_STAR
    assert result["draw_denominators"] == {
        "planned": 12, "valid": 12, "failed": 0, "abstained": 0,
    }
    assert result["physical_score"] is None
    assert result["certified"] is False


def test_d06_any_block_length_below_threshold_prevents_clearance():
    comp, cand, comp_ids, cand_ids = _d06_inputs({
        2: [1.8, 1.9, 2.0, 2.1],
        5: [1.8, 1.9, 2.0, 2.1],
        10: [1.0, 1.1, 1.2, 1.3],
    })

    result = d06_paired_ratio_hull(
        comp, cand, comp_ids, cand_ids, planned_draws_per_length=4,
    )

    assert result["status"] == "D06_THRESHOLD_NOT_CLEARED"
    assert result["hull_lower"] < D06_R_STAR
    assert result["result_denominators"]["valid"] == 1


def test_d06_zero_risk_is_not_repaired_with_epsilon_and_is_counted():
    comp, cand, comp_ids, cand_ids = _d06_inputs({
        2: [2.0, 2.0, 2.0, 2.0],
        5: [2.0, 2.0, 2.0, 2.0],
        10: [2.0, 0.0, 2.0, 2.0],
    })

    result = d06_paired_ratio_hull(
        comp, cand, comp_ids, cand_ids, planned_draws_per_length=4,
    )

    assert result["status"] == "ABSTAIN_INVALID_OR_UNPAIRED_RISK_DRAWS"
    assert result["block_results"]["10"]["status"] == "ABSTAIN_NONPOSITIVE_OR_NONFINITE_RISK"
    assert result["block_results"]["10"]["valid"] == 3
    assert result["draw_denominators"] == {
        "planned": 12, "valid": 11, "failed": 1, "abstained": 0,
    }
    assert result["hull_lower"] is None


def test_d06_rejects_unpaired_draw_ids_and_nonfinite_ratios():
    comp, cand, comp_ids, cand_ids = _d06_inputs({
        2: [2.0, 2.0, 2.0, 2.0],
        5: [2.0, 2.0, 2.0, 2.0],
        10: [1e308, 1e308, 1e308, 1e308],
    })
    cand[10] = np.full(4, 1e-308)
    cand_ids[5] = np.array([0, 1, 2, 9])

    result = d06_paired_ratio_hull(
        comp, cand, comp_ids, cand_ids, planned_draws_per_length=4,
    )

    assert result["status"] == "ABSTAIN_INVALID_OR_UNPAIRED_RISK_DRAWS"
    assert result["block_results"]["5"]["status"] == "ABSTAIN_UNPAIRED_OR_MALFORMED_DRAWS"
    assert result["block_results"]["10"]["status"] == "ABSTAIN_NONFINITE_RISK_RATIO"
    assert result["draw_denominators"] == {
        "planned": 12, "valid": 4, "failed": 8, "abstained": 0,
    }


def test_d06_does_not_allow_callers_to_change_the_selected_contract():
    comp, cand, comp_ids, cand_ids = _d06_inputs({
        2: [2.0] * 4, 5: [2.0] * 4, 10: [2.0] * 4,
    })

    with pytest.raises(ValueError, match="fixed by the selected F=2 contract"):
        d06_paired_ratio_hull(
            comp, cand, comp_ids, cand_ids, planned_draws_per_length=4,
            alpha_per_comparison=0.05,
        )
    with pytest.raises(ValueError, match="fixed by the selected contract"):
        d06_paired_ratio_hull(
            comp, cand, comp_ids, cand_ids, planned_draws_per_length=4,
            r_star=1.0,
        )
