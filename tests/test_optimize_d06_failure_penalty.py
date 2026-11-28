import json

import numpy as np
import pytest

from tasks.optimize_gaussian.bounded_block_risk import ARMS, R_STAR
from tasks.optimize_gaussian.d06_failure_penalty_diagnostic import (
    FAILURE_SCENARIOS,
    SUCCESS_RISKS,
    build,
    expected_risks,
)


def test_failure_mixture_formula_and_fixed_threshold_sensitivity():
    assert np.allclose(expected_risks([0.0, 0.0, 0.0]), SUCCESS_RISKS)
    equal_failures = expected_risks([0.05, 0.05, 0.05])
    assert np.allclose(equal_failures, [0.088, 0.126, 0.12125])
    assert equal_failures[1] / equal_failures[0] > R_STAR
    trainable_only = expected_risks([0.05, 0.0, 0.0])
    assert trainable_only[1] / trainable_only[0] < 1.0
    comparator_only = expected_risks([0.0, 0.05, 0.05])
    assert comparator_only[1] / comparator_only[0] > equal_failures[1] / equal_failures[0]


@pytest.mark.parametrize("bad", [[-0.1, 0, 0], [0, 1.1, 0], [0, float("nan"), 0]])
def test_failure_probabilities_must_be_finite_and_bounded(bad):
    with pytest.raises(ValueError, match="failure probabilities"):
        expected_risks(bad)


def test_failure_sensitivity_report_is_explicitly_analytic_and_keeps_denominator(tmp_path):
    out = tmp_path / "failure-penalty"
    protocol = build(out)
    assert len(ARMS) == 3
    assert set(protocol["failure_scenarios"]) == set(FAILURE_SCENARIOS)
    assert protocol["formula_scenario_denominator"] == {
        "planned": 303,
        "valid": 303,
        "failed": 0,
        "abstained": 0,
        "unit": "one exact probability and failure-pattern scenario; not a dataset or measurement",
    }
    assert protocol["physical_score"] is None
    assert (out / "failure_sensitivity.csv").is_file()
    assert (out / "failure_effect.png").is_file()
    manifest = json.loads((out / "artifact_hashes.json").read_text())
    assert "README.md" in manifest
