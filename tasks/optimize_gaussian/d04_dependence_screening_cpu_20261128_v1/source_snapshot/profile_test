from __future__ import annotations

import numpy as np
import pytest

from tasks.optimize_gaussian.d04_ar1_bootstrap_diagnostic import (
    PROFILE_RHO_GRID_POINTS,
    _profile_common_rho_candidate,
    _profile_matrix_normal_deviance,
)


@pytest.fixture
def fixed_panel() -> np.ndarray:
    # Deterministic full-rank algebra fixture; this is not a calibration panel.
    return np.asarray([
        [0.0, 1.0, 4.0],
        [2.0, 1.0, 0.0],
        [1.0, 3.0, 2.0],
        [4.0, 0.0, 1.0],
        [2.0, 2.0, 5.0],
        [5.0, 4.0, 3.0],
        [3.0, 6.0, 2.0],
        [7.0, 2.0, 4.0],
    ])


@pytest.mark.parametrize("rho", [0.35, -0.35])
def test_profile_deviance_matches_direct_matrix_normal_profile(fixed_panel, rho):
    n, n_features = fixed_panel.shape
    lag = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    row_covariance = rho**lag
    inverse = np.linalg.inv(row_covariance)
    ones = np.ones(n)
    mean = (ones @ inverse @ fixed_panel) / (ones @ inverse @ ones)
    residual = fixed_panel - ones[:, None] * mean[None, :]
    scatter = residual.T @ inverse @ residual
    _, logdet_rows = np.linalg.slogdet(row_covariance)
    _, logdet_features = np.linalg.slogdet(scatter / n)
    expected = n_features * logdet_rows + n * logdet_features

    assert _profile_matrix_normal_deviance(fixed_panel, rho) == pytest.approx(expected)


def test_profile_rho_candidate_is_invariant_to_feature_affine_reparameterization(
    fixed_panel,
):
    transformed = (fixed_panel * np.asarray([2.0, -3.0, 0.5])
                   + np.asarray([10.0, -4.0, 1.0]))

    original = _profile_common_rho_candidate(fixed_panel)
    reparameterized = _profile_common_rho_candidate(transformed)

    assert reparameterized["rho_candidate"] == pytest.approx(
        original["rho_candidate"], abs=2.0e-6
    )
    assert original["grid_points"] == PROFILE_RHO_GRID_POINTS
    assert len(original["profile_deviance_grid"]) == PROFILE_RHO_GRID_POINTS
    assert np.all(np.isfinite(original["profile_deviance_grid"]))
    assert not original["boundary_hit"]
    assert not reparameterized["boundary_hit"]


def test_profile_deviance_rejects_singular_profiled_feature_covariance():
    singular = np.arange(18, dtype=float).reshape(6, 3)

    with pytest.raises(ValueError, match="not positive definite"):
        _profile_matrix_normal_deviance(singular, rho=0.2)
