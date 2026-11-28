"""Finite-Nt two-vacuum × three-static-state GLS prototype.

This candidate extends the current absolute-force fit in the same finite-Nt
forward model. It is CPU/synthetic research code only and does not certify
finite-sample coverage or production Wilson conventions.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares
from scipy.special import expit

from . import f_lane_joint_spectral_gls_diagnostic as forward
from . import f_lane_two_sector_fit_candidate as two_state
from .f_lane_two_sector_fit_candidate import NT_VALUES, R_VALUES, T_VALUES


VACUUM_STATES = 2
STATIC_STATES = 3
OPERATOR_COUNT = two_state.OPERATOR_COUNT
PARAMETER_COUNT = 1 + len(R_VALUES) * (STATIC_STATES + VACUUM_STATES * STATIC_STATES * 5)
FIT_SCHEMA = "wilson-f-two-vacuum-three-static-fit-candidate-v1"


def _softplus(value: float) -> float:
    return float(np.logaddexp(0.0, value))


def _inverse_softplus(value: float) -> float:
    if value <= 0 or not math.isfinite(value):
        raise ValueError("inverse softplus requires a positive finite value")
    return float(np.log(np.expm1(value)))


def _decode(parameters: object) -> tuple[float, np.ndarray, np.ndarray]:
    p = np.asarray(parameters, dtype=float)
    if p.shape != (PARAMETER_COUNT,) or not np.isfinite(p).all():
        raise ValueError(f"three-static parameter vector must contain {PARAMETER_COUNT} finite values")
    epsilon = _softplus(float(p[0]))
    static = np.empty((len(R_VALUES), STATIC_STATES), dtype=float)
    overlaps = np.empty(
        (len(R_VALUES), VACUUM_STATES, STATIC_STATES, OPERATOR_COUNT), dtype=complex,
    )
    cursor = 1
    for radius_index in range(len(R_VALUES)):
        increments = np.asarray([_softplus(float(value)) for value in p[cursor:cursor + STATIC_STATES]])
        static[radius_index] = np.cumsum(increments)
        cursor += STATIC_STATES
        for vacuum_state in range(VACUUM_STATES):
            for static_state in range(STATIC_STATES):
                log_a0, a1r, a1i, a2r, a2i = p[cursor:cursor + 5]
                overlaps[radius_index, vacuum_state, static_state] = (
                    np.exp(log_a0), a1r + 1j * a1i, a2r + 1j * a2i,
                )
                cursor += 5
    return epsilon, static, overlaps


def _initial_parameters(static_guesses: list[np.ndarray], overlap_guesses: list[np.ndarray],
                        epsilon: float, excited_scale: float) -> np.ndarray:
    if (len(static_guesses) != len(R_VALUES) or len(overlap_guesses) != len(R_VALUES)
            or not math.isfinite(excited_scale) or excited_scale <= 0):
        raise ValueError("three-static initializer has invalid radius axes or overlap scale")
    values = [_inverse_softplus(epsilon)]
    for radius_index in range(len(R_VALUES)):
        energies = np.asarray(static_guesses[radius_index], dtype=float)
        overlaps = np.asarray(overlap_guesses[radius_index], dtype=complex)
        if (energies.shape != (STATIC_STATES,) or overlaps.shape != (STATIC_STATES, OPERATOR_COUNT)
                or not np.isfinite(energies).all() or not np.isfinite(overlaps).all()
                or energies[0] <= 0 or np.any(np.diff(energies) <= 0)):
            raise ValueError("three-static initializer requires positive ordered energies and finite overlaps")
        increments = np.diff(np.concatenate(([0.0], energies)))
        values.extend(_inverse_softplus(float(value)) for value in increments)
        for vacuum_state in range(VACUUM_STATES):
            scale = 1.0 if vacuum_state == 0 else excited_scale
            for static_state in range(STATIC_STATES):
                values.extend(two_state._pack_amplitude(scale * overlaps[static_state]))
    initial = np.asarray(values, dtype=float)
    if initial.shape != (PARAMETER_COUNT,) or not np.isfinite(initial).all():
        raise ValueError("three-static initializer produced malformed parameters")
    return initial


def _model_vectors(parameters: object) -> np.ndarray:
    epsilon, static, overlaps = _decode(parameters)
    vacuum = np.asarray((0.0, epsilon), dtype=float)
    rows = []
    for radius_index in range(len(R_VALUES)):
        for temporal_extent in NT_VALUES:
            partition = float(np.sum(np.exp(-vacuum * temporal_extent)))
            for time_extent in T_VALUES:
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


def _initial_from_snapshot(matrices: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    static_guesses, overlap_guesses, receipts = [], [], []
    nt_index = NT_VALUES.index(32)
    for radius_index, radius in enumerate(R_VALUES):
        block_vectors = np.stack([
            np.concatenate([
                forward._matrix_vector(matrices[block, nt_index, radius_index, time_index])
                for time_index in range(len(T_VALUES))
            ])
            for block in range(matrices.shape[0])
        ])
        fit = forward.fit_joint_spectrum(
            block_vectors, time_slices=T_VALUES, n_states=STATIC_STATES,
            operator_indices=(0, 1, 2),
        )
        receipts.append({
            "R": radius, "success": bool(fit.get("success")),
            "rank": fit.get("rank"), "parameters": fit.get("parameter_count"),
            "nfev": fit.get("nfev"), "reduced_chi_square": fit.get("reduced_chi_square"),
        })
        if not fit.get("success"):
            raise ValueError(f"Nt=32 three-state initializer failed at R={radius}")
        static_guesses.append(np.asarray(fit["energies"], dtype=float))
        overlap_pairs = np.asarray(fit["overlaps"], dtype=float)
        if overlap_pairs.shape != (STATIC_STATES, OPERATOR_COUNT, 2):
            raise ValueError(f"Nt=32 initializer overlap axes are invalid at R={radius}")
        overlap_guesses.append(overlap_pairs[..., 0] + 1j * overlap_pairs[..., 1])
    return static_guesses, overlap_guesses, receipts


def _absolute_force_gradient(parameters: object) -> np.ndarray:
    p = np.asarray(parameters, dtype=float)
    if p.shape != (PARAMETER_COUNT,) or not np.isfinite(p).all():
        raise ValueError("three-static force gradient received malformed parameters")
    cursor = 1
    ground_indices = []
    radius_stride = STATIC_STATES + VACUUM_STATES * STATIC_STATES * 5
    for _ in R_VALUES:
        ground_indices.append(cursor)
        cursor += radius_stride
    gradient = np.zeros_like(p)
    gradient[ground_indices[0]] = -float(expit(p[ground_indices[0]]))
    gradient[ground_indices[1]] = float(expit(p[ground_indices[1]]))
    return gradient


def fit_three_static_snapshot(snapshot) -> dict[str, Any]:
    """Fit the same two-vacuum finite-Nt trace with three static states."""
    manifest, matrices = two_state._validate_raw_snapshot(
        snapshot, manifest_name="manifest", matrix_name="raw_matrices",
        contract_name="lane_contract",
    )
    blocks = two_state._observed_vectors(matrices)
    mean = blocks.mean(axis=0)
    covariance, covariance_diagnostics = two_state._regularized_mean_covariance(blocks)
    covariance_cholesky = cholesky(covariance, lower=True, check_finite=True)
    static_guesses, overlap_guesses, initializer_receipts = _initial_from_snapshot(matrices)

    lower = np.full(PARAMETER_COUNT, -np.inf)
    upper = np.full(PARAMETER_COUNT, np.inf)
    lower[0] = -8.0
    upper[0] = 2.5
    cursor = 1
    for _ in R_VALUES:
        lower[cursor:cursor + STATIC_STATES] = -8.0
        upper[cursor:cursor + STATIC_STATES] = 2.5
        cursor += STATIC_STATES
        for _ in range(VACUUM_STATES * STATIC_STATES):
            lower[cursor] = -12.0
            upper[cursor] = 4.0
            lower[cursor + 1:cursor + 5] = -5.0
            upper[cursor + 1:cursor + 5] = 5.0
            cursor += 5

    def residual(parameters: np.ndarray) -> np.ndarray:
        raw = _model_vectors(parameters).ravel() - mean
        return solve_triangular(covariance_cholesky, raw, lower=True, check_finite=True)

    attempts, successful = [], []
    for epsilon_guess, scale in ((0.15, 0.20), (0.30, 0.35), (0.55, 0.50)):
        initial = _initial_parameters(static_guesses, overlap_guesses, epsilon_guess, scale)
        if np.any(initial < lower) or np.any(initial > upper):
            attempts.append({
                "epsilon_start": epsilon_guess, "excited_overlap_scale_start": scale,
                "success": False, "status": "ABSTAIN_INITIALIZER_OUT_OF_BOUNDS",
                "nfev": 0,
            })
            continue
        try:
            result = least_squares(
                residual, initial, bounds=(lower, upper), method="trf",
                x_scale="jac", max_nfev=8000, ftol=1e-10, xtol=1e-10, gtol=1e-10,
            )
            singular = np.linalg.svd(result.jac, compute_uv=False)
            tolerance = (singular[0] * max(result.jac.shape) * np.finfo(float).eps
                         if singular.size else 0.0)
            rank = int(np.count_nonzero(singular > tolerance))
            finite = bool(np.isfinite(result.fun).all() and np.isfinite(result.x).all())
            valid = bool(result.success and finite and rank == PARAMETER_COUNT)
            attempt = {
                "epsilon_start": epsilon_guess, "excited_overlap_scale_start": scale,
                "success": valid, "optimizer_success": bool(result.success),
                "rank": rank, "parameter_count": PARAMETER_COUNT,
                "objective": float(np.dot(result.fun, result.fun)) if finite else None,
                "nfev": int(result.nfev), "message": str(result.message),
            }
            attempts.append(attempt)
            if valid:
                successful.append((float(np.dot(result.fun, result.fun)), result, singular, rank))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            attempts.append({
                "epsilon_start": epsilon_guess, "excited_overlap_scale_start": scale,
                "success": False, "status": "FAILED_OPTIMIZATION",
                "nfev": 0, "message": type(exc).__name__,
            })

    if not successful:
        return {
            "schema": FIT_SCHEMA, "status": "three_static_fit_failed",
            "certified": False, "physical_score": None,
            "split_id": manifest["split_id"],
            "denominator": {"planned": 1, "valid": 0, "failed": 1, "abstained": 0},
            "initializer_fits": initializer_receipts, "fit_attempts": attempts,
            "open_gates": ["no full-rank finite-Nt 2-vacuum × 3-static fit converged"],
        }

    objective, best, singular, rank = min(successful, key=lambda row: row[0])
    epsilon, static, _ = _decode(best.x)
    force = float(static[1, 0] - static[0, 0])
    _, _, right = np.linalg.svd(best.jac, full_matrices=False)
    parameter_covariance = (right.T * (1.0 / singular ** 2)) @ right
    gradient = _absolute_force_gradient(best.x)
    force_variance = float(gradient @ parameter_covariance @ gradient)
    fit = {
        "parameter_count": PARAMETER_COUNT,
        "jacobian_rank": rank,
        "jacobian_condition_number": float(singular[0] / singular[-1]),
        "degrees_of_freedom": int(best.fun.size - PARAMETER_COUNT),
        "gls_objective": objective,
        "reduced_gls_objective": objective / (best.fun.size - PARAMETER_COUNT),
        "vacuum_energies": [0.0, epsilon],
        "static_energies_by_R": {str(radius): static[index].tolist()
                                  for index, radius in enumerate(R_VALUES)},
        "absolute_stout_ground_force": force,
        "force_delta_standard_error_diagnostic": (
            math.sqrt(force_variance) if math.isfinite(force_variance) and force_variance > 0 else None
        ),
        "force_uncertainty_calibrated": False,
        "fit_coordinates": best.x.astype(float).tolist(),
    }
    return {
        "schema": FIT_SCHEMA, "status": "three_static_fit_candidate",
        "certified": False, "physical_score": None,
        "split_id": manifest["split_id"],
        "denominator": {"planned": 1, "valid": 1, "failed": 0, "abstained": 0},
        "covariance_diagnostics": covariance_diagnostics,
        "initializer_fits": initializer_receipts,
        "fit_attempts": attempts, "fit": fit,
        "open_gates": [
            "the endpoint/source-sink factorization remains a draft convention",
            "the inverse-Jacobian standard error is an asymptotic diagnostic, not calibrated coverage",
            "no difficult-strata, production exporter, thermal/mixed contamination, reference force, review or authorization receipts",
        ],
    }
