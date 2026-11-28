import numpy as np
import pytest

from tasks.wilson_loop.f_lane_gap_overlap_force_screen import (
    GAP_GRID,
    KERNELS,
    R_VALUES,
    TIME_STARTS,
    TOY_E0,
    TOY_GAP,
    TOY_OVERLAP_RATIO,
    _toy_effective_energies,
    candidate_reference_effective_bias_bound,
    effective_energy_interval,
    excited_energy_correction_bound,
    force_interval,
)
from tasks.wilson_loop.f_lane_finite_time_ground_force_nonidentifiability import (
    EFFECTIVE_ENERGY_TIMES,
    _build_witness,
)


def test_q_gap_conditional_energy_and_force_intervals_contain_toy_truth():
    energies = _toy_effective_energies()
    q_bounds = np.full((len(KERNELS), len(R_VALUES)), 0.25)
    gap_bounds = np.full_like(q_bounds, 0.20)
    lower, upper = effective_energy_interval(energies, q_bounds, gap_bounds)
    force_lower, force_upper = force_interval(lower, upper)

    assert np.all(lower <= TOY_E0)
    assert np.all(TOY_E0 <= upper)
    assert force_lower <= 0.20 <= force_upper
    assert lower[0] < 0.0  # no E0 >= 0 assumption is hidden in the bound


def test_common_static_energy_shift_leaves_force_interval_unchanged():
    energies = _toy_effective_energies()
    q_bounds = np.full((len(KERNELS), len(R_VALUES)), 0.25)
    gap_bounds = np.full_like(q_bounds, 0.20)
    bounds = effective_energy_interval(energies, q_bounds, gap_bounds)
    shifted = effective_energy_interval(energies + 7.25, q_bounds, gap_bounds)

    assert np.allclose(shifted[0] - 7.25, bounds[0], rtol=0.0, atol=1e-14)
    assert np.allclose(shifted[1] - 7.25, bounds[1], rtol=0.0, atol=1e-14)
    assert force_interval(*shifted) == pytest.approx(force_interval(*bounds))


def test_four_corner_bias_bound_covers_finite_time_kernel_contrast_proxy():
    energies = _toy_effective_energies()
    bound = candidate_reference_effective_bias_bound(TOY_OVERLAP_RATIO, TOY_GAP)
    t5 = TIME_STARTS.index(5)
    per_kernel_force = energies[:, 1, t5] - energies[:, 0, t5]
    observed_contrast = float(per_kernel_force[0] - per_kernel_force[1])

    assert bound > 0.0
    assert abs(observed_contrast) <= bound
    assert not np.isclose(observed_contrast, 0.0)


def test_preceding_witness_changes_absolute_force_but_not_shared_kernel_contrast():
    result = _build_witness()
    for matrices_key in ("base_matrices", "alt_matrices"):
        matrices = result[matrices_key]
        kernel_forces = []
        for operator_index in (0, 2):
            effective_by_radius = []
            for radius_index in range(2):
                values = [
                    np.log(
                        matrices[radius_index, time_index, operator_index, operator_index]
                        / matrices[radius_index, time_index + 1, operator_index, operator_index]
                    )
                    for time_index, _time in enumerate(EFFECTIVE_ENERGY_TIMES)
                ]
                effective_by_radius.append(float(np.mean(values)))
            kernel_forces.append(effective_by_radius[1] - effective_by_radius[0])
        assert kernel_forces[0] == pytest.approx(kernel_forces[1], abs=1e-14)


def test_correction_bound_decreases_with_time_and_gap():
    by_time = [excited_energy_correction_bound(0.5, 0.2, t) for t in (3, 4, 5, 6)]
    by_gap = [excited_energy_correction_bound(0.5, gap, 5) for gap in (0.1, 0.2, 0.4)]

    assert np.all(np.diff(by_time) < 0.0)
    assert np.all(np.diff(by_gap) < 0.0)


def test_nonreal_and_bool_arrays_fail_closed():
    energies = _toy_effective_energies()
    q_bounds = np.full((len(KERNELS), len(R_VALUES)), 0.25)
    gap_bounds = np.full_like(q_bounds, 0.20)
    with pytest.raises(ValueError, match="real integer or floating dtype"):
        effective_energy_interval(energies.astype(complex) + 1.0j, q_bounds, gap_bounds)
    with pytest.raises(ValueError, match="real integer or floating dtype"):
        effective_energy_interval(energies, q_bounds.astype(bool), gap_bounds)


def test_invalid_time_and_empty_conditional_intersection_are_rejected():
    with pytest.raises(ValueError, match="integer >= 1"):
        excited_energy_correction_bound(0.1, 0.2, True)
    energies = _toy_effective_energies()
    too_tight_q = np.full((len(KERNELS), len(R_VALUES)), 1e-8)
    too_large_gap = np.full_like(too_tight_q, 2.0)
    with pytest.raises(ValueError, match="do not intersect"):
        effective_energy_interval(energies, too_tight_q, too_large_gap)


def test_gap_grid_is_a_sensitivity_axis_not_a_receipt():
    assert GAP_GRID[0] == pytest.approx(0.01)
    assert GAP_GRID[-1] == pytest.approx(1.0)
    assert len(GAP_GRID) == 61
