import csv

import pytest

from tasks.pion_projection.anchor_bias_feasibility_diagnostic import build, selected_inputs


def test_selected_anchor_bridge_is_inside_the_necessary_best_case_ceiling():
    inputs = selected_inputs()
    assert inputs["anchor_method"] == "markov_pseudo_gap_bernstein"
    assert inputs["n_anchor"] == 4000
    assert inputs["anchor_gamma_ps_lower_assumed"] == pytest.approx(0.20)
    assert inputs["anchor_burn_in_units_assumed"] == 8
    assert inputs["anchor_tv_bound_assumed"] == pytest.approx(0.001)
    assert inputs["anchor_stationary_alpha_after_tv"] == pytest.approx(0.009)
    assert inputs["anchor_random_radius_over_R"] == pytest.approx(0.2020396155)
    assert inputs["selected_C_over_R"] < inputs["C_over_R_max_for_any_best_case_pass"]
    assert inputs["best_case_lcb_before_C"] - 2 * inputs["selected_C_over_R"] > (
        inputs["delta_star_over_R2"]
    )


def test_range_only_common_bias_bound_cannot_reach_positive_effect_threshold():
    inputs = selected_inputs()
    best_case = inputs["best_case_lcb_before_C"] - 2 * inputs["trivial_range_only_C_over_R"]
    assert best_case < inputs["delta_star_over_R2"]
    assert inputs["trivial_range_only_C_over_R"] == pytest.approx(1.0)


def test_report_is_analytic_only_and_keeps_physical_gate_abstained(tmp_path):
    output = tmp_path / "bias-report"
    protocol = build(output)
    assert protocol["schema"] == "pion-anchor-common-bias-feasibility-v2"
    assert protocol["physical_score"] is None
    assert protocol["physical_evidence_denominator"] == {
        "planned": 1, "valid": 0, "failed": 0, "abstained": 1,
        "reason": "no anchor scalar-map, common-bias, independence, reference, or chain receipts",
    }
    with (output / "bias_sensitivity.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 101
    assert all(int(row["planned"]) == 1 and int(row["valid"]) == 1 for row in rows)
    assert protocol["source_hashes"]["bounded_block_risk_anchor_markov.py"]
    report = (output / "README.md").read_text()
    assert "conditional Markov-anchor random radius/R" in report
    assert "alpha_A=0.01" in report
