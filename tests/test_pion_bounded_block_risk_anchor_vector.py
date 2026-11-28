import numpy as np
import pytest
import hashlib
import json

from tasks.pion_projection.bounded_block_risk_anchor_vector import (
    markov_anchor_mean_box,
)
from tasks.pion_projection.bounded_block_risk_anchor_vector_diagnostic import build_report
from tasks.pion_projection.bounded_block_risk_markov_gap import markov_gap_radius


def test_vector_box_uses_one_joint_tv_budget_and_componentwise_family_bound():
    samples = np.array([
        [0.2, -0.5, 0.8],
        [0.4, 0.1, 0.7],
        [0.3, 0.2, 0.9],
        [0.6, -0.2, 0.6],
    ])
    support = np.array([[0.0, 1.0], [-1.0, 1.0], [0.0, 1.0]])

    result = markov_anchor_mean_box(
        samples,
        support_bounds=support,
        gamma_ps_lower=0.4,
        alpha_anchor=0.05,
        initial_law_tv_bound=0.01,
        burn_in_units=8,
        stationary_start=False,
    )

    assert result["status"] == "CONDITIONAL_MARKOV_VECTOR_MEAN_BOX_ONLY"
    assert result["stationary_alpha_after_tv"] == pytest.approx(0.04)
    assert result["joint_coverage_probability_lower_bound"] == pytest.approx(0.95)
    assert result["fit_target_interval"] is None
    assert result["common_bias_bound_C"] is None
    assert result["denominators"]["vector_box"] == {
        "planned": 1, "valid": 1, "failed": 0, "abstained": 0,
    }
    assert result["denominators"]["feature_coordinates"] == {
        "planned": 3, "valid": 3, "failed": 0, "abstained": 0,
    }

    for j, item in enumerate(result["feature_intervals"]):
        expected_radius = markov_gap_radius(
            blocks=len(samples),
            gamma_ps_lower=0.4,
            family=samples.shape[1],
            alpha_s=0.04,
            variance_bound=(support[j, 1] - support[j, 0]) ** 2 / 4.0,
            centered_abs_bound=support[j, 1] - support[j, 0],
        )
        assert item["raw_bernstein_radius"] == pytest.approx(expected_radius)
        lower, upper = item["confidence_interval"]
        assert support[j, 0] <= lower <= upper <= support[j, 1]


def test_vector_box_retains_coordinates_and_intervals_cover_the_sample_means():
    samples = np.array([
        [0.2, -0.5],
        [0.4, 0.1],
        [0.3, 0.2],
        [0.6, -0.2],
    ])
    result = markov_anchor_mean_box(
        samples,
        support_bounds=[(0.0, 1.0), (-1.0, 1.0)],
        gamma_ps_lower=0.5,
        alpha_anchor=0.05,
    )
    for j, item in enumerate(result["feature_intervals"]):
        lower, upper = item["confidence_interval"]
        assert lower <= samples[:, j].mean() <= upper
        assert item["feature_index"] == j


@pytest.mark.parametrize("samples,support,match", [
    (np.array([[0.2, 1.1]]), [(0.0, 1.0), (0.0, 1.0)], "outside"),
    (np.array([[0.2, np.nan]]), [(0.0, 1.0), (0.0, 1.0)], "finite"),
    (np.array([[0.2 + 0.1j]]), [(0.0, 1.0)], "real-valued"),
    (np.array([0.2, 0.4]), [(0.0, 1.0)], "shape"),
    (np.array([[0.2]]), [(1.0, 1.0)], "increasing"),
])
def test_vector_box_rejects_invalid_observations_or_support(samples, support, match):
    with pytest.raises(ValueError, match=match):
        markov_anchor_mean_box(
            samples,
            support_bounds=support,
            gamma_ps_lower=0.5,
            alpha_anchor=0.05,
        )


def test_vector_box_fails_closed_when_tv_spends_anchor_alpha():
    with pytest.raises(ValueError, match="initial_law_tv_bound"):
        markov_anchor_mean_box(
            np.array([[0.2], [0.3]]),
            support_bounds=[(0.0, 1.0)],
            gamma_ps_lower=0.5,
            alpha_anchor=0.01,
            initial_law_tv_bound=0.01,
            burn_in_units=8,
            stationary_start=False,
        )


def test_vector_box_rejects_unrepresentable_numeric_parameters():
    with pytest.raises(ValueError, match="gamma_ps_lower"):
        markov_anchor_mean_box(
            np.array([[0.2], [0.3]]),
            support_bounds=[(0.0, 1.0)],
            gamma_ps_lower=10**10000,
            alpha_anchor=0.05,
        )


def test_vector_box_requires_consistent_startup_convention():
    with pytest.raises(ValueError, match="stationary start"):
        markov_anchor_mean_box(
            np.array([[0.2], [0.3]]),
            support_bounds=[(0.0, 1.0)],
            gamma_ps_lower=0.5,
            alpha_anchor=0.05,
            initial_law_tv_bound=0.001,
            stationary_start=True,
        )


def test_vector_box_report_has_hashed_text_flow_and_keeps_fit_target_open(tmp_path):
    output = tmp_path / "anchor-vector"
    report = build_report(output)
    manifest = json.loads((output / "artifact_hashes.json").read_text())
    receipt = json.loads((output / "run_receipt.json").read_text())

    assert report["status"] == "CONDITIONAL_VECTOR_MEAN_BOX_ONLY"
    assert report["physical_score"] is None
    assert report["target_fit_interval"] is None
    assert report["common_bias_bound_C"] is None
    readme = (output / "README.md").read_text()
    assert "Text flow:" in readme
    assert "not configurations, a Markov path" in readme
    assert receipt["status"] == "complete"
    assert receipt["denominators"]["vector_box"]["planned"] == 1
    assert "bounded_block_risk_anchor_vector.py" in receipt["scenario"]["source_hashes"]
    for name, expected in receipt["artifact_hashes"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert set(manifest) == {
        "README.md", "protocol.json", "result.json", "run_receipt.json",
    }
    for name, expected in manifest.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
