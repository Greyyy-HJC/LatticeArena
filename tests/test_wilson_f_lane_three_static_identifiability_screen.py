from __future__ import annotations

import numpy as np

from tasks.wilson_loop.f_lane_three_static_identifiability_screen import (
    STRATA,
    _case_parameters,
    _generate_panel,
)
from tasks.wilson_loop.f_lane_two_sector_fit_candidate import NT_VALUES,R_VALUES,T_VALUES


def test_stress_strata_are_explicit_and_have_distinct_truth_or_noise_contracts():
    assert len(STRATA) == 5
    near = _case_parameters("near_degenerate")
    assert np.isclose(near["static_energies"][2][1] - near["static_energies"][2][0], 0.035)
    weak = _case_parameters("weak_ground_overlap")
    base = _case_parameters("baseline")
    assert np.allclose(weak["overlaps"][:, :, 0, :], base["overlaps"][:, :, 0, :] * 0.05)
    omitted = _case_parameters("omitted_vacuum_truth")
    assert omitted["vacuum_energies"].shape == (1,)
    dependent = _case_parameters("dependent_unequal_variance")
    assert dependent["rho"] == 0.75
    assert dependent["scale_by_T"]["6"] == 1.8


def test_panel_generator_is_deterministic_finite_and_hermitian():
    first, spec = _generate_panel(2026114299, "baseline")
    second, _ = _generate_panel(2026114299, "baseline")
    assert first.shape == (96, len(NT_VALUES), len(R_VALUES), len(T_VALUES), 3, 3)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.allclose(first, first.conj().swapaxes(-1, -2), rtol=1e-12, atol=1e-13)
    assert spec["force_truth"] == 0.14999999999999997


def test_omitted_vacuum_generator_keeps_finite_nt_matrix_axes():
    matrices, spec = _generate_panel(2026114203, "omitted_vacuum_truth")
    assert matrices.shape == (96, len(NT_VALUES), len(R_VALUES), len(T_VALUES), 3, 3)
    assert spec["vacuum_state_count_truth"] == 1
    assert np.allclose(matrices, matrices.conj().swapaxes(-1, -2), rtol=1e-12, atol=1e-13)
