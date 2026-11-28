from __future__ import annotations

import numpy as np
import pytest

from tasks.optimize_gaussian import d04_gof_calibration_diagnostic as generator
from tasks.optimize_gaussian.d04_heldout_dependence_diagnostic import (
    DGP_CELLS,
    METHODS,
    _circular_moving_block_sample,
    _draw_panel,
    _finite_n_mean_variance_factor,
    _lag_correlation,
    _seed,
)


def test_heldout_cell_acf_and_finite_n_mean_variance_contract():
    cells = {cell["name"]: cell for cell in DGP_CELLS}

    assert _lag_correlation(cells["gaussian_iid"], 1) == 0.0
    assert _lag_correlation(cells["gaussian_ar1_rho030"], 1) == pytest.approx(0.3)
    assert _lag_correlation(cells["multiscale_gaussian_mixture"], 1) == pytest.approx(0.36)
    assert _finite_n_mean_variance_factor(cells["gaussian_iid"], 128) == 1.0
    assert _finite_n_mean_variance_factor(cells["gaussian_ar1_rho060"], 128) > 1.0


@pytest.mark.parametrize("cell_name", [
    "student_t5_innovation_ar1_rho030",
    "multiscale_gaussian_mixture",
])
def test_new_heldout_dgp_draws_are_reproducible_and_finite(cell_name):
    cell = next(cell for cell in DGP_CELLS if cell["name"] == cell_name)
    mean = generator._truth_mean(
        generator.MASS_NULL,
        generator.PP_AMPLITUDES_NULL,
        generator.A4_AMPLITUDES_NULL,
    )
    covariance = generator._noise_covariance(mean)
    first = _draw_panel(cell, np.random.default_rng(81021), mean, covariance)
    second = _draw_panel(cell, np.random.default_rng(81021), mean, covariance)

    assert first.shape == (128, 27)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_seed_manifest_streams_are_disjoint_for_the_frozen_grid():
    seeds = [
        _seed(cell_index, panel, stream)
        for cell_index in range(len(DGP_CELLS))
        for panel in range(200)
        for stream in range(len(METHODS) + 1)
    ]

    assert len(seeds) == len(set(seeds))
    assert all(seed >= 0 for seed in seeds)


def test_circular_block_bootstrap_preserves_a_full_length_residual_path():
    sample = np.arange(30, dtype=float).reshape(10, 3)
    fitted_mean = np.asarray([2.0, -3.0, 5.0])
    residual = sample - sample.mean(axis=0, keepdims=True)
    result = _circular_moving_block_sample(
        np.random.default_rng(9), sample, fitted_mean, block_length=len(sample),
    )

    assert result.shape == sample.shape
    assert np.isfinite(result).all()
    centered_result = result - fitted_mean
    assert any(np.allclose(centered_result, np.roll(residual, shift, axis=0))
               for shift in range(len(sample)))


@pytest.mark.parametrize("block_length", [0, -1, 11, True, 1.5])
def test_circular_block_bootstrap_rejects_invalid_length(block_length):
    sample = np.arange(30, dtype=float).reshape(10, 3)
    with pytest.raises(ValueError, match="block_length"):
        _circular_moving_block_sample(
            np.random.default_rng(1), sample, np.zeros(3), block_length=block_length,
        )
