from __future__ import annotations

import numpy as np
import pytest

from tasks.wilson_loop import f_lane_three_static_pivot_chart as pivot_chart
from tasks.wilson_loop import f_lane_three_static_two_vacuum_candidate as current
from tasks.wilson_loop import f_lane_two_sector_fit_candidate as two_state


def _current_parameters():
    parameters = np.zeros(current.PARAMETER_COUNT, dtype=np.float64)
    parameters[0] = current._inverse_softplus(0.27)
    for radius_index in range(len(two_state.R_VALUES)):
        radius_start = 1 + radius_index * pivot_chart.RADIUS_STRIDE
        energies = (0.29 + 0.02 * radius_index, 0.58 + 0.03 * radius_index,
                    0.96 + 0.04 * radius_index)
        increments = np.diff(np.concatenate(([0.0], energies)))
        parameters[radius_start:radius_start + current.STATIC_STATES] = [
            current._inverse_softplus(float(value)) for value in increments
        ]
        amplitude_start = radius_start + current.STATIC_STATES
        for state_index in range(current.VACUUM_STATES * current.STATIC_STATES):
            block = amplitude_start + state_index * 5
            parameters[block:block + 5] = [
                np.log(0.30 + 0.02 * state_index),
                0.12 + 0.01 * state_index, -0.04 + 0.005 * state_index,
                -0.08 + 0.003 * state_index, 0.06 - 0.002 * state_index,
            ]
    radius0_amplitude_start = 1 + current.STATIC_STATES
    parameters[radius0_amplitude_start:radius0_amplitude_start + 5] = [
        np.log(0.08), 0.80, 0.20, -0.30, 0.10,
    ]
    radius1_amplitude_start = 1 + pivot_chart.RADIUS_STRIDE + current.STATIC_STATES
    radius1_last_transition = radius1_amplitude_start + (current.STATIC_STATES * 2 - 1) * 5
    parameters[radius1_last_transition:radius1_last_transition + 5] = [
        np.log(0.10), 0.20, 0.10, 1.10, -0.30,
    ]
    return parameters


def _manual_vectors(epsilon, static, overlaps):
    rows = []
    vacuum = np.asarray([0.0, epsilon])
    for radius_index in range(len(two_state.R_VALUES)):
        for nt in two_state.NT_VALUES:
            partition = np.sum(np.exp(-vacuum * nt))
            for time_extent in two_state.T_VALUES:
                weights = np.exp(
                    -vacuum[:, None] * (nt - time_extent)
                    -static[radius_index][None, :] * time_extent,
                ) / partition
                matrix = np.einsum(
                    "mn,mni,mnj->ij", weights, overlaps[radius_index],
                    overlaps[radius_index].conj(), optimize=True,
                )
                rows.append(two_state._matrix_vector(matrix))
    return np.stack(rows)


def test_rechart_preserves_existing_finite_nt_model_exactly():
    parameters = _current_parameters()
    recoded, pivots = pivot_chart.encode_current_parameters(parameters)

    assert recoded.shape == parameters.shape
    assert pivots.shape == (len(two_state.R_VALUES), current.VACUUM_STATES,
                            current.STATIC_STATES)
    np.testing.assert_allclose(
        pivot_chart.model_vectors(recoded, pivots),
        current._model_vectors(parameters), rtol=2e-14, atol=2e-14,
    )


def test_chart_supports_a_transition_with_zero_operator_zero_overlap():
    epsilon = 0.31
    static = np.asarray([[0.28, 0.61, 0.99], [0.32, 0.67, 1.04]])
    overlaps = np.zeros((2, 2, 3, 3), dtype=np.complex128)
    for index in np.ndindex(overlaps.shape[:-1]):
        overlaps[index] = np.asarray([0.4 + 0.01 * sum(index),
                                     0.2 + 0.03j, -0.1 + 0.02j])
    overlaps[0, 1, 2] = np.asarray([0.0, 0.70 + 0.10j, -0.20 + 0.10j])

    parameters, pivots = pivot_chart.encode_components(epsilon, static, overlaps)
    _epsilon, _static, decoded = pivot_chart._decode(parameters, pivots)

    assert pivots[0, 1, 2] == 1
    np.testing.assert_allclose(pivot_chart.model_vectors(parameters, pivots),
                               _manual_vectors(epsilon, static, decoded),
                               rtol=2e-14, atol=2e-14)
    with pytest.raises(ValueError, match="operator-0 overlap"):
        two_state._pack_amplitude(overlaps[0, 1, 2])


def test_pivot_chart_analytic_jacobian_matches_central_difference():
    parameters, pivots = pivot_chart.encode_current_parameters(_current_parameters())
    analytic = pivot_chart.model_jacobian(parameters, pivots)
    numeric = np.empty_like(analytic)
    for index, value in enumerate(parameters):
        step = 1e-6 * max(1.0, abs(float(value)))
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += step
        minus[index] -= step
        numeric[:, index] = (
            pivot_chart.model_vectors(plus, pivots).ravel()
            - pivot_chart.model_vectors(minus, pivots).ravel()
        ) / (2.0 * step)

    np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=2e-8)


def test_fixed_operator_zero_chart_matches_zero_pivot_map_and_jacobian():
    parameters = _current_parameters()
    pivots = np.zeros((len(two_state.R_VALUES), current.VACUUM_STATES,
                       current.STATIC_STATES), dtype=np.int64)
    np.testing.assert_allclose(
        pivot_chart.model_vectors(parameters, pivots),
        current._model_vectors(parameters), rtol=2e-14, atol=2e-14,
    )

    analytic = pivot_chart.model_jacobian(parameters, pivots)
    numeric = np.empty_like(analytic)
    for index, value in enumerate(parameters):
        step = 1e-6 * max(1.0, abs(float(value)))
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += step
        minus[index] -= step
        numeric[:, index] = (
            current._model_vectors(plus).ravel()
            - current._model_vectors(minus).ravel()
        ) / (2.0 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=2e-8)


def test_all_zero_transition_has_no_identifiable_phase_chart():
    with pytest.raises(ValueError, match="all-zero transition"):
        pivot_chart._pack_vector(np.zeros(3, dtype=np.complex128))


def test_jacobian_rejects_noninteger_pivot_map_instead_of_truncating():
    parameters, pivots = pivot_chart.encode_current_parameters(_current_parameters())
    malformed = pivots.astype(np.float64)
    malformed[0, 0, 0] = 0.5

    with pytest.raises(ValueError, match="pivot map"):
        pivot_chart.model_jacobian(parameters, malformed)
