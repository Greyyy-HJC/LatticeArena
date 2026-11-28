from __future__ import annotations

import numpy as np

from tasks.wilson_loop import f_lane_finite_nt_trace_diagnostic as toy
from tasks.wilson_loop import f_lane_three_static_two_vacuum_candidate as candidate
from tasks.wilson_loop.f_lane_three_static_nonlinear_overlap_profile import (
    FIXED_FORCE_INDEX,
    GROUND_A0_INDICES,
    _expand_force_coordinate,
    _model_jacobian,
    _start_vectors,
)


def _parameters():
    static = [toy.STATIC_ENERGIES[radius].copy() for radius in (2, 3)]
    overlaps = [toy.TRANSITION_OVERLAPS[index, 0].copy() for index in range(2)]
    return candidate._initial_parameters(static, overlaps, epsilon=0.30,
                                         excited_scale=0.35)


def test_analytic_finite_nt_model_jacobian_matches_central_difference():
    base = _parameters()
    rng = np.random.default_rng(20261128)
    for _ in range(4):
        parameters = base + rng.normal(0.0, 0.025, size=base.size)
        analytic = _model_jacobian(parameters)
        numeric = np.empty_like(analytic)
        for index, value in enumerate(parameters):
            step = 1e-6 * max(1.0, abs(float(value)))
            plus, minus = parameters.copy(), parameters.copy()
            plus[index] += step
            minus[index] -= step
            numeric[:, index] = (
                candidate._model_vectors(plus).ravel()
                - candidate._model_vectors(minus).ravel()
            ) / (2.0 * step)
        assert analytic.shape == (
            candidate._model_vectors(parameters).size, candidate.PARAMETER_COUNT,
        )
        assert np.isfinite(analytic).all()
        assert np.max(np.abs(analytic - numeric)) < 2e-6


def test_exact_force_elimination_and_chain_jacobian():
    parameters = _parameters()
    reduced_indices = [index for index in range(candidate.PARAMETER_COUNT)
                       if index != FIXED_FORCE_INDEX]
    reduced = np.delete(parameters, FIXED_FORCE_INDEX)
    target_force = 0.15
    full, chain = _expand_force_coordinate(reduced, target_force)
    energies = candidate._decode(full)[1]
    assert np.isclose(energies[1, 0] - energies[0, 0], target_force, atol=1e-14)

    column = reduced_indices.index(1)
    step = 1e-6
    plus, minus = reduced.copy(), reduced.copy()
    plus[column] += step
    minus[column] -= step
    numeric = (
        candidate._model_vectors(_expand_force_coordinate(plus, target_force)[0]).ravel()
        - candidate._model_vectors(_expand_force_coordinate(minus, target_force)[0]).ravel()
    ) / (2.0 * step)
    analytic = _model_jacobian(full) @ chain[:, column]
    assert np.max(np.abs(analytic - numeric)) < 2e-6


def test_neighbor_continuation_replaces_secondary_start_without_mutating_seed():
    base = _parameters()
    continuation = base.copy()
    continuation[0] += 0.1
    continuation[2] += 0.2
    before = continuation.copy()
    starts = _start_vectors(base, floor=0.9, continuation_parameters=continuation)

    assert [name for name, _ in starts] == ["captured_fit", "neighbor_force_continuation"]
    assert np.array_equal(continuation, before)
    assert starts[1][1] is not continuation
    assert all(starts[1][1][index] >= np.log(0.9) + 0.02
               for index in GROUND_A0_INDICES)


def test_neighbor_continuation_rejects_wrong_shape_and_nonfinite_seed():
    base = _parameters()
    try:
        _start_vectors(base, floor=0.9, continuation_parameters=base[:-1])
    except ValueError as exc:
        assert "full parameter vector" in str(exc)
    else:
        raise AssertionError("wrong-size continuation start should fail closed")

    invalid = base.copy()
    invalid[0] = np.nan
    try:
        _start_vectors(base, floor=0.9, continuation_parameters=invalid)
    except ValueError as exc:
        assert "full parameter vector" in str(exc)
    else:
        raise AssertionError("nonfinite continuation start should fail closed")
