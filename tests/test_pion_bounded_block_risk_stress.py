import math

import pytest

from tasks.pion_projection.bounded_block_risk_stress import (
    classify_scenario,
    denominator_summary,
    make_grid,
    planned_failure_count,
)


def test_grid_crosses_all_declared_calibration_dimensions():
    rows = make_grid()
    assert len(rows) == 1728
    assert {row["planned_blocks"] for row in rows} == {20, 100, 400}
    assert {row["chromatic_number_upper_bound"] for row in rows} == {1, 2, 3, 5}
    assert {row["alpha_anchor"] for row in rows} == {0.0, 0.01, 0.02}
    assert {row["relative_anchor_bound_B_over_R"] for row in rows} == {
        0.0, 0.025, 0.05, 0.1,
    }
    assert {row["requested_failure_fraction"] for row in rows} == {
        0.0, 0.05, 0.1, 0.25,
    }
    assert {row["normalized_threshold_delta_over_R2"] for row in rows} == {
        0.0, 0.05, 0.1,
    }
    assert all(row["alpha_s"] + row["alpha_anchor"] == pytest.approx(0.05)
               for row in rows)


def test_strict_boundary_at_one_abstains_instead_of_clipping():
    row = classify_scenario(
        blocks=20,
        chromatic_number=20,
        alpha_total=0.05,
        alpha_anchor=0.0,
        relative_anchor_bound=0.0,
        requested_failure_fraction=0.0,
        normalized_threshold=0.0,
    )
    assert row["required_valid_mean_boundary"] == pytest.approx(
        math.sqrt(2 * math.log(40))
    )
    assert row["required_valid_mean_boundary"] > 1.0
    assert row["status"] == "ABSTAIN_UNATTAINABLE_MEAN"


def test_exactly_one_boundary_abstains_under_strict_certification_rule():
    penalty = math.sqrt(2 * math.log(2 / 0.05) / 20)
    row = classify_scenario(
        blocks=20,
        chromatic_number=1,
        alpha_total=0.05,
        alpha_anchor=0.0,
        relative_anchor_bound=0.0,
        requested_failure_fraction=0.0,
        normalized_threshold=1.0 - penalty,
    )
    assert row["required_valid_mean_boundary"] == pytest.approx(1.0)
    assert row["status"] == "ABSTAIN_UNATTAINABLE_MEAN"


def test_all_failed_planned_blocks_abstain_without_dropping_denominator():
    row = classify_scenario(
        blocks=20,
        chromatic_number=1,
        alpha_total=0.05,
        alpha_anchor=0.0,
        relative_anchor_bound=0.0,
        requested_failure_fraction=0.999,
        normalized_threshold=0.0,
    )
    assert row["planned_blocks"] == 20
    assert row["valid_blocks"] == 0
    assert row["failed_blocks"] == 20
    assert row["status"] == "ABSTAIN_NO_VALID_BLOCKS"


def test_fraction_rounding_preserves_conservative_integer_failures():
    assert planned_failure_count(20, 0.0) == 0
    assert planned_failure_count(20, 0.051) == 2
    with pytest.raises(ValueError):
        planned_failure_count(20, 1.0)


def test_summary_keeps_scenario_and_hypothetical_block_denominators_separate():
    rows = [
        classify_scenario(
            blocks=20,
            chromatic_number=1,
            alpha_total=0.05,
            alpha_anchor=0.0,
            relative_anchor_bound=0.0,
            requested_failure_fraction=0.0,
            normalized_threshold=0.0,
        ),
        classify_scenario(
            blocks=20,
            chromatic_number=20,
            alpha_total=0.05,
            alpha_anchor=0.0,
            relative_anchor_bound=0.0,
            requested_failure_fraction=0.1,
            normalized_threshold=0.0,
        ),
    ]
    scenarios, blocks = denominator_summary(rows)
    assert scenarios["planned"] == 2
    assert scenarios["valid"] + scenarios["failed"] + scenarios["abstained"] == 2
    assert scenarios["valid"] == 1
    assert scenarios["abstained"] == 1
    assert blocks["planned"] == 40
    assert blocks["valid"] == 38
    assert blocks["failed"] == 2
    assert blocks["abstained"] == 0
