import numpy as np
import pytest

from tasks.optimize_gaussian.d06_bootstrap_coverage_diagnostic import generate_paired_ar1_panel
from tasks.optimize_gaussian.d06_hac_coverage_diagnostic import (
    d04_risk_and_influence,
    hac_ratio_lowers,
    newey_west_long_run_variance,
)


def test_d04_influence_matches_finite_difference_of_empirical_risk():
    panel = generate_paired_ar1_panel(12345, phi=0.4, n=40)
    risk, influence = d04_risk_and_influence(panel, block_length=5)

    configuration, arm = 7, 1
    epsilon = 1e-7
    shifted = panel.copy()
    shifted[configuration, arm, :] += epsilon
    shifted_risk, _ = d04_risk_and_influence(shifted, block_length=5)
    finite_difference = (shifted_risk[arm] - risk[arm]) / epsilon

    assert finite_difference == pytest.approx(
        influence[configuration, arm] / panel.shape[0], rel=2e-5, abs=2e-8,
    )


def test_newey_west_constant_series_has_zero_long_run_variance():
    assert newey_west_long_run_variance(np.ones(20), bandwidth=5) == pytest.approx(0.0)


def test_hac_ratio_hull_has_every_selected_length_bandwidth_pair():
    panel = generate_paired_ar1_panel(4567, phi=0.8, n=60)
    lower, point = hac_ratio_lowers(panel)

    expected_keys = {f"L{length}_b{bandwidth}"
                     for length in (2, 5, 10) for bandwidth in (2, 5, 10, 20, 30)}
    assert set(lower["SLSQP"]) == expected_keys
    assert set(lower["Chebyshev"]) == expected_keys
    assert set(point["SLSQP"]) == {"L2", "L5", "L10"}
    assert set(point["Chebyshev"]) == {"L2", "L5", "L10"}
    assert all(np.isfinite(list(lower[name].values())).all() for name in lower)
    assert all(value > 0 for by_length in point.values() for value in by_length.values())
