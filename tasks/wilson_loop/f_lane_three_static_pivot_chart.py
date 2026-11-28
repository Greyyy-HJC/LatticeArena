"""Alternative phase chart for finite-Nt multi-operator spectral amplitudes.

The current candidate fixes operator 0 as the positive real phase anchor for
every transition vector. This prototype chooses the largest-magnitude
component as the per-vector phase anchor. It represents the same positive
spectral model and changes no Wilson estimand or certification rule.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import expit

from . import f_lane_three_static_two_vacuum_candidate as current
from . import f_lane_two_sector_fit_candidate as two_state


SCHEMA = "wilson-f-three-static-largest-component-phase-chart-v1"
VACUUM_STATES = current.VACUUM_STATES
STATIC_STATES = current.STATIC_STATES
PARAMETER_COUNT = current.PARAMETER_COUNT
RADIUS_STRIDE = STATIC_STATES + VACUUM_STATES * STATIC_STATES * 5


def _softplus(value: float) -> float:
    return float(np.logaddexp(0.0, value))


def _inverse_softplus(value: float) -> float:
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError("inverse softplus requires a positive finite value")
    return float(np.log(np.expm1(value)))


def _pack_vector(vector: object) -> tuple[list[float], int]:
    z = np.asarray(vector, dtype=np.complex128)
    if z.shape != (current.OPERATOR_COUNT,) or not np.isfinite(z).all():
        raise ValueError("transition overlap must be a finite three-operator vector")
    pivot = int(np.argmax(np.abs(z)))
    pivot_abs = float(abs(z[pivot]))
    if pivot_abs <= 0.0:
        raise ValueError("an all-zero transition vector has no identifiable phase chart")
    rotated = z * np.exp(-1j * np.angle(z[pivot]))
    values = [math.log(pivot_abs)]
    for operator in range(current.OPERATOR_COUNT):
        if operator != pivot:
            values.extend((float(rotated[operator].real), float(rotated[operator].imag)))
    return values, pivot


def encode_components(epsilon: float, static_energies: object,
                      overlaps: object) -> tuple[np.ndarray, np.ndarray]:
    """Encode physical components in a largest-component phase chart.

    ``static_energies`` has axes (R, n); ``overlaps`` has axes (R, m, n, i).
    Every transition vector must have at least one nonzero component.
    """
    static = np.asarray(static_energies, dtype=np.float64)
    amplitudes = np.asarray(overlaps, dtype=np.complex128)
    expected_static = (len(two_state.R_VALUES), STATIC_STATES)
    expected_amplitudes = (len(two_state.R_VALUES), VACUUM_STATES,
                           STATIC_STATES, current.OPERATOR_COUNT)
    if (not math.isfinite(float(epsilon)) or epsilon <= 0.0
            or static.shape != expected_static or amplitudes.shape != expected_amplitudes
            or not np.isfinite(static).all() or not np.isfinite(amplitudes).all()):
        raise ValueError("finite-Nt components have malformed axes or non-finite values")
    if np.any(static[:, 0] <= 0.0) or np.any(np.diff(static, axis=1) <= 0.0):
        raise ValueError("static energies must be positive and strictly ordered at each R")

    parameters = np.empty(PARAMETER_COUNT, dtype=np.float64)
    pivots = np.empty(expected_amplitudes[:-1], dtype=np.int64)
    parameters[0] = _inverse_softplus(float(epsilon))
    for radius_index in range(len(two_state.R_VALUES)):
        radius_start = 1 + radius_index * RADIUS_STRIDE
        increments = np.diff(np.concatenate(([0.0], static[radius_index])))
        parameters[radius_start:radius_start + STATIC_STATES] = [
            _inverse_softplus(float(value)) for value in increments
        ]
        amplitude_start = radius_start + STATIC_STATES
        for vacuum_state in range(VACUUM_STATES):
            for static_state in range(STATIC_STATES):
                block = amplitude_start + (vacuum_state * STATIC_STATES + static_state) * 5
                packed, pivot = _pack_vector(amplitudes[radius_index, vacuum_state, static_state])
                parameters[block:block + 5] = packed
                pivots[radius_index, vacuum_state, static_state] = pivot
    return parameters, pivots


def encode_current_parameters(parameters: object) -> tuple[np.ndarray, np.ndarray]:
    """Re-chart a point from the current operator-0 convention exactly."""
    epsilon, static, overlaps = current._decode(parameters)
    return encode_components(epsilon, static, overlaps)


def _decode(parameters: object, pivots: object
            ) -> tuple[float, np.ndarray, np.ndarray]:
    p = np.asarray(parameters, dtype=np.float64)
    pivot_map = np.asarray(pivots)
    if p.shape != (PARAMETER_COUNT,) or not np.isfinite(p).all():
        raise ValueError(f"pivot-chart parameter vector must contain {PARAMETER_COUNT} finite values")
    expected_pivots = (len(two_state.R_VALUES), VACUUM_STATES, STATIC_STATES)
    if (pivot_map.shape != expected_pivots or pivot_map.dtype.kind not in "iu"
            or np.any(pivot_map < 0) or np.any(pivot_map >= current.OPERATOR_COUNT)):
        raise ValueError("pivot map must contain one valid operator index per transition vector")
    epsilon = _softplus(float(p[0]))
    static = np.empty((len(two_state.R_VALUES), STATIC_STATES), dtype=np.float64)
    overlaps = np.empty((len(two_state.R_VALUES), VACUUM_STATES, STATIC_STATES,
                         current.OPERATOR_COUNT), dtype=np.complex128)
    for radius_index in range(len(two_state.R_VALUES)):
        radius_start = 1 + radius_index * RADIUS_STRIDE
        increments = np.asarray([
            _softplus(float(value))
            for value in p[radius_start:radius_start + STATIC_STATES]
        ])
        static[radius_index] = np.cumsum(increments)
        amplitude_start = radius_start + STATIC_STATES
        for vacuum_state in range(VACUUM_STATES):
            for static_state in range(STATIC_STATES):
                block = amplitude_start + (vacuum_state * STATIC_STATES + static_state) * 5
                pivot = int(pivot_map[radius_index, vacuum_state, static_state])
                vector = np.zeros(current.OPERATOR_COUNT, dtype=np.complex128)
                vector[pivot] = math.exp(float(p[block]))
                cursor = block + 1
                for operator in range(current.OPERATOR_COUNT):
                    if operator != pivot:
                        vector[operator] = p[cursor] + 1j * p[cursor + 1]
                        cursor += 2
                overlaps[radius_index, vacuum_state, static_state] = vector
    return epsilon, static, overlaps


def model_vectors(parameters: object, pivots: object) -> np.ndarray:
    """Evaluate the selected two-vacuum × three-static finite-Nt model."""
    epsilon, static, overlaps = _decode(parameters, pivots)
    vacuum = np.asarray((0.0, epsilon), dtype=np.float64)
    rows = []
    for radius_index in range(len(two_state.R_VALUES)):
        for temporal_extent in two_state.NT_VALUES:
            partition = float(np.sum(np.exp(-vacuum * temporal_extent)))
            for time_extent in two_state.T_VALUES:
                weights = np.exp(
                    -vacuum[:, None] * (temporal_extent - time_extent)
                    -static[radius_index][None, :] * time_extent
                ) / partition
                matrix = np.einsum(
                    "mn,mni,mnj->ij", weights, overlaps[radius_index],
                    overlaps[radius_index].conj(), optimize=True,
                )
                rows.append(two_state._matrix_vector(matrix))
    return np.stack(rows)


def model_jacobian(parameters: object, pivots: object) -> np.ndarray:
    """Analytic Jacobian in pivot-chart coordinates, flattened like model_vectors."""
    p = np.asarray(parameters, dtype=np.float64)
    pivot_map = np.asarray(pivots)
    epsilon, static, overlaps = _decode(p, pivot_map)
    nt_values = np.asarray(two_state.NT_VALUES, dtype=np.float64)
    vacuum_excited_probability = np.exp(-epsilon * nt_values)
    vacuum_excited_probability /= 1.0 + vacuum_excited_probability
    model = model_vectors(p, pivot_map)
    jacobian = np.zeros((model.size, PARAMETER_COUNT), dtype=np.float64)
    row_index = 0
    for radius_index in range(len(two_state.R_VALUES)):
        radius_start = 1 + radius_index * RADIUS_STRIDE
        energy_indices = tuple(radius_start + index for index in range(STATIC_STATES))
        amplitude_start = radius_start + STATIC_STATES
        for nt_index, nt in enumerate(two_state.NT_VALUES):
            p_excited = float(vacuum_excited_probability[nt_index])
            partition = 1.0 + math.exp(-epsilon * nt)
            for time_extent in two_state.T_VALUES:
                weights = np.exp(
                    -np.asarray([0.0, epsilon])[:, None] * (nt - time_extent)
                    -static[radius_index][None, :] * time_extent,
                ) / partition
                derivatives: dict[int, np.ndarray] = {}
                for vacuum_state in range(VACUUM_STATES):
                    for static_state in range(STATIC_STATES):
                        weight = float(weights[vacuum_state, static_state])
                        amplitude = overlaps[radius_index, vacuum_state, static_state]
                        pivot = int(pivot_map[radius_index, vacuum_state, static_state])
                        outer = np.outer(amplitude, amplitude.conj())
                        dlog_weight_depsilon = (
                            -(nt - time_extent) * (vacuum_state == 1)
                            + nt * p_excited
                        ) * float(expit(p[0]))
                        derivatives.setdefault(0, np.zeros((3, 3), dtype=np.complex128))
                        derivatives[0] += weight * dlog_weight_depsilon * outer
                        for increment_index, parameter_index in enumerate(energy_indices):
                            if increment_index <= static_state:
                                derivatives.setdefault(
                                    parameter_index, np.zeros((3, 3), dtype=np.complex128),
                                )
                                derivatives[parameter_index] += (
                                    -time_extent * float(expit(p[parameter_index]))
                                    * weight * outer
                                )
                        block = amplitude_start + (
                            vacuum_state * STATIC_STATES + static_state
                        ) * 5
                        dz_terms = [(block, pivot, amplitude[pivot])]
                        cursor = block + 1
                        for operator in range(current.OPERATOR_COUNT):
                            if operator != pivot:
                                dz_terms.append((cursor, operator, 1.0 + 0.0j))
                                dz_terms.append((cursor + 1, operator, 1.0j))
                                cursor += 2
                        for parameter_index, operator, dz in dz_terms:
                            direction = np.zeros(3, dtype=np.complex128)
                            direction[operator] = dz
                            derivative = weight * (
                                np.outer(direction, amplitude.conj())
                                + np.outer(amplitude, direction.conj())
                            )
                            derivatives.setdefault(
                                parameter_index, np.zeros((3, 3), dtype=np.complex128),
                            )
                            derivatives[parameter_index] += derivative
                for parameter_index, matrix in derivatives.items():
                    jacobian[row_index:row_index + 9, parameter_index] = two_state._matrix_vector(matrix)
                row_index += 9
    return jacobian
