from __future__ import annotations

import numpy as np
import pytest

from tasks.wilson_loop import f_lane_finite_nt_trace_diagnostic as toy
from tasks.wilson_loop import f_lane_three_static_two_vacuum_candidate as three_state
from tasks.wilson_loop import f_lane_two_sector_fit_candidate as two_state
from tasks.wilson_loop.f_lane_two_sector_fit_candidate import NT_VALUES,R_VALUES,T_VALUES


def _toy_parameter_vector() -> np.ndarray:
    values = [three_state._inverse_softplus(float(toy.VACUUM_ENERGIES[1]))]
    for radius_index, radius in enumerate(R_VALUES):
        energies = toy.STATIC_ENERGIES[radius]
        increments = np.diff(np.concatenate(([0.0], energies)))
        values.extend(three_state._inverse_softplus(float(value)) for value in increments)
        for vacuum_state in range(three_state.VACUUM_STATES):
            for static_state in range(three_state.STATIC_STATES):
                values.extend(two_state._pack_amplitude(
                    toy.TRANSITION_OVERLAPS[radius_index, vacuum_state, static_state]
                ))
    return np.asarray(values)


def _toy_observed_vector() -> np.ndarray:
    return np.stack([
        three_state.two_state._matrix_vector(toy.finite_trace_matrix(radius, time, nt))
        for radius in R_VALUES for nt in NT_VALUES for time in T_VALUES
    ])


def test_three_static_two_vacuum_model_matches_exact_finite_nt_toy():
    parameters = _toy_parameter_vector()
    predicted = three_state._model_vectors(parameters)
    observed = _toy_observed_vector()
    assert predicted.shape == (len(R_VALUES) * len(NT_VALUES) * len(T_VALUES), 9)
    assert np.allclose(predicted, observed, rtol=2e-13, atol=2e-15)


def test_decode_enforces_parameter_dimension_and_finiteness():
    params = _toy_parameter_vector()
    with pytest.raises(ValueError, match="parameter vector"):
        three_state._decode(params[:-1])
    params[0] = np.nan
    with pytest.raises(ValueError, match="parameter vector"):
        three_state._decode(params)


def test_initializer_rejects_nonordered_static_energy_guesses():
    energies = [toy.STATIC_ENERGIES[radius].copy() for radius in R_VALUES]
    overlaps = [np.ones((three_state.STATIC_STATES, three_state.OPERATOR_COUNT), dtype=complex)
                for _ in R_VALUES]
    energies[0][2] = energies[0][1]
    with pytest.raises(ValueError, match="positive ordered energies"):
        three_state._initial_parameters(energies, overlaps, 0.3, 0.25)


def test_force_gradient_depends_only_on_the_two_ground_energy_coordinates():
    params = _toy_parameter_vector()
    gradient = three_state._absolute_force_gradient(params)
    epsilon, static, _ = three_state._decode(params)
    radius_stride = three_state.STATIC_STATES + three_state.VACUUM_STATES * three_state.STATIC_STATES * 5
    expected = np.zeros_like(params)
    expected[1] = -1.0 / (1.0 + np.exp(-params[1]))
    expected[1 + radius_stride] = 1.0 / (1.0 + np.exp(-params[1 + radius_stride]))
    assert epsilon > 0
    assert np.all(static[:, 0] > 0)
    assert np.allclose(gradient, expected)
