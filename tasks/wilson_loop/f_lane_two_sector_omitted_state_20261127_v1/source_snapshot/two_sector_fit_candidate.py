"""Proposed finite-Nt two-sector fit candidate for synthetic Wilson matrices.

The fitted trace is exactly the DRAFT formula in ``lane_contract_v2.json``.
This module is a CPU method prototype: it does not establish exporter endpoint
conventions, finite-sample coverage, or production evidence.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import math
from typing import Any

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares
from scipy.stats import norm

from core.npz_safety import validate_npz_payload
from core.source_integrity import SourceByteSnapshot
from . import f_lane_joint_spectral_gls_diagnostic as forward
from .f_lane_transfer_trace_reader import (
    _component_values,
    _validate_contract,
)


RAW_SCHEMA = "wilson-f-two-sector-raw-matrices-v1"
RAW_MANIFEST_SCHEMA = "wilson-f-two-sector-raw-manifest-v1"
FIT_SCHEMA = "wilson-f-two-sector-spectrum-fit-v1"
FIT_RESULT_SCHEMA = "wilson-f-two-sector-fit-candidate-v1"
R_VALUES = (2, 3)
NT_VALUES = (16, 32)
T_VALUES = (3, 4, 5, 6, 7)
OPERATOR_COUNT = 3
VACUUM_STATES = 2
STATIC_STATES = 2
_MANIFEST_KEYS = {
    "schema", "split_id", "geometry", "action", "boundary_conditions",
    "radius_order", "temporal_extents", "time_slices", "dimension_order",
    "matrix_sha256", "lane_contract_sha256",
}
GLS_DIAGONAL_SHRINKAGE = 0.25
GLS_EIGENVALUE_FLOOR_RELATIVE = 1e-10
QHX_POINTWISE_ALPHA = 0.025
QHX_SIMULTANEOUS_ALPHA = 0.05


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(snapshot: SourceByteSnapshot, name: str) -> tuple[dict[str, Any], bytes]:
    payload = snapshot.read_bytes(name)
    if sha256(payload).hexdigest() != snapshot.expected_hashes.get(name):
        raise ValueError(f"{name}: captured bytes do not match their expected hash")
    value = json.loads(
        payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {token}")),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{name}: JSON root must be an object")
    return value, payload


def _softplus(value: float) -> float:
    return float(np.logaddexp(0.0, value))


def _inverse_softplus(value: float) -> float:
    if value <= 0 or not math.isfinite(value):
        raise ValueError("inverse softplus requires a positive finite value")
    return float(np.log(np.expm1(value)))


def _pack_amplitude(vector: np.ndarray) -> list[float]:
    z = np.asarray(vector, dtype=complex)
    if z.shape != (OPERATOR_COUNT,) or not np.isfinite(z).all():
        raise ValueError("each transition overlap must be a finite three-operator vector")
    if abs(z[0]) <= 1e-12:
        raise ValueError("the overlap phase convention requires nonzero operator-0 overlap")
    z = z * np.exp(-1j * np.angle(z[0]))
    if z[0].real <= 0:
        raise ValueError("overlap phase convention failed to make operator-0 overlap positive")
    return [float(np.log(z[0].real)), float(z[1].real), float(z[1].imag),
            float(z[2].real), float(z[2].imag)]


def _initial_parameters(static_guesses: list[np.ndarray], overlap_guesses: list[np.ndarray],
                        epsilon: float, excited_scale: float) -> np.ndarray:
    values = [_inverse_softplus(epsilon)]
    for radius_index in range(len(R_VALUES)):
        static = np.asarray(static_guesses[radius_index], dtype=float)
        overlaps = np.asarray(overlap_guesses[radius_index], dtype=complex)
        if static.shape != (STATIC_STATES,) or overlaps.shape != (STATIC_STATES, OPERATOR_COUNT):
            raise ValueError("forward-fit initializer has the wrong state or operator axes")
        values.extend((_inverse_softplus(float(static[0])),
                       _inverse_softplus(float(static[1] - static[0]))))
        for vacuum_state in range(VACUUM_STATES):
            for static_state in range(STATIC_STATES):
                scale = 1.0 if vacuum_state == 0 else excited_scale
                values.extend(_pack_amplitude(scale * overlaps[static_state]))
    return np.asarray(values, dtype=float)


def _decode(parameters: object) -> tuple[float, np.ndarray, np.ndarray]:
    p = np.asarray(parameters, dtype=float)
    expected = 1 + len(R_VALUES) * (2 + VACUUM_STATES * STATIC_STATES * 5)
    if p.shape != (expected,) or not np.isfinite(p).all():
        raise ValueError(f"two-sector parameter vector must contain {expected} finite values")
    epsilon = _softplus(float(p[0]))
    static = np.empty((len(R_VALUES), STATIC_STATES), dtype=float)
    overlaps = np.empty((len(R_VALUES), VACUUM_STATES, STATIC_STATES, OPERATOR_COUNT),
                        dtype=complex)
    cursor = 1
    for radius_index in range(len(R_VALUES)):
        ground = _softplus(float(p[cursor]))
        gap = _softplus(float(p[cursor + 1]))
        cursor += 2
        static[radius_index] = (ground, ground + gap)
        for vacuum_state in range(VACUUM_STATES):
            for static_state in range(STATIC_STATES):
                log_a0, a1r, a1i, a2r, a2i = p[cursor:cursor + 5]
                overlaps[radius_index, vacuum_state, static_state] = (
                    np.exp(log_a0), a1r + 1j * a1i, a2r + 1j * a2i,
                )
                cursor += 5
    vacuum = np.asarray((0.0, epsilon), dtype=float)
    return epsilon, static, overlaps


def _matrix_vector(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=complex)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("raw finite-Nt matrix must be finite complex 3x3")
    if not np.allclose(value, value.conj().T, rtol=1e-10, atol=1e-12):
        raise ValueError("raw finite-Nt matrix must be Hermitian")
    return np.asarray([
        value[0, 0].real, value[1, 1].real, value[2, 2].real,
        value[0, 1].real, value[0, 1].imag,
        value[0, 2].real, value[0, 2].imag,
        value[1, 2].real, value[1, 2].imag,
    ], dtype=float)


def _model_vectors(parameters: np.ndarray) -> np.ndarray:
    epsilon, static, overlaps = _decode(parameters)
    vacuum = np.asarray((0.0, epsilon), dtype=float)
    rows = []
    for radius_index in range(len(R_VALUES)):
        for nt in NT_VALUES:
            partition = float(np.sum(np.exp(-vacuum * nt)))
            for time_extent in T_VALUES:
                weights = np.exp(
                    -vacuum[:, None] * (nt - time_extent)
                    -static[radius_index][None, :] * time_extent
                ) / partition
                matrix = np.einsum(
                    "mn,mni,mnj->ij", weights, overlaps[radius_index],
                    overlaps[radius_index].conj(), optimize=True,
                )
                rows.append(_matrix_vector(matrix))
    return np.stack(rows)


def _observed_vectors(matrices: np.ndarray) -> np.ndarray:
    rows = []
    for radius_index in range(len(R_VALUES)):
        for nt_index in range(len(NT_VALUES)):
            for time_index in range(len(T_VALUES)):
                rows.append(np.stack([
                    _matrix_vector(matrices[block, nt_index, radius_index, time_index])
                    for block in range(matrices.shape[0])
                ]))
    # (cell, block, component) -> (block, flattened cell/component)
    return np.transpose(np.stack(rows), (1, 0, 2)).reshape(matrices.shape[0], -1)


def _regularized_mean_covariance(block_vectors: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Estimate a regularized covariance of the block mean for the GLS pilot.

    The raw covariance is rank deficient when the number of blocks is below the
    number of matrix components. Fixed diagonal shrinkage and an eigenvalue
    floor make the numerical metric positive definite; they do not calibrate
    its finite-sample coverage.
    """
    blocks = np.asarray(block_vectors, dtype=float)
    if blocks.ndim != 2 or blocks.shape[0] < 3 or not np.isfinite(blocks).all():
        raise ValueError("block_vectors must be a finite matrix with at least three blocks")
    raw = np.cov(blocks, rowvar=False, ddof=1) / blocks.shape[0]
    diagonal = np.diag(np.diag(raw))
    shrunk = ((1.0 - GLS_DIAGONAL_SHRINKAGE) * raw
              + GLS_DIAGONAL_SHRINKAGE * diagonal)
    raw_values = np.linalg.eigvalsh(0.5 * (raw + raw.T))
    values, vectors = np.linalg.eigh(0.5 * (shrunk + shrunk.T))
    largest = float(values[-1])
    if largest <= 0.0 or not math.isfinite(largest):
        raise ValueError("estimated mean covariance has no positive eigenvalue")
    floor = GLS_EIGENVALUE_FLOOR_RELATIVE * largest
    floored = np.maximum(values, floor)
    covariance = (vectors * floored) @ vectors.T
    return covariance, {
        "block_count": int(blocks.shape[0]),
        "component_count": int(blocks.shape[1]),
        "diagonal_shrinkage": GLS_DIAGONAL_SHRINKAGE,
        "eigenvalue_floor_relative": GLS_EIGENVALUE_FLOOR_RELATIVE,
        "raw_min_eigenvalue": float(raw_values[0]),
        "shrunk_min_eigenvalue": float(values[0]),
        "regularized_min_eigenvalue": float(floored[0]),
        "max_eigenvalue": largest,
        "condition_number": float(floored[-1] / floored[0]),
        "floor_active_count": int(np.count_nonzero(values < floor)),
        "covariance_of": "sample mean",
    }


def _qhx_vector(parameters: np.ndarray) -> np.ndarray:
    epsilon, static, amplitudes = _decode(parameters)
    values = []
    for radius_index in range(len(R_VALUES)):
        for operator_index in range(OPERATOR_COUNT):
            for time_extent in (5, 6):
                for temporal_extent in NT_VALUES:
                    cell = _component_values(
                        np.asarray((0.0, epsilon)), static[radius_index],
                        amplitudes[radius_index], operator_index,
                        time_extent, temporal_extent,
                    )
                    values.extend(cell[name] for name in ("q", "h", "x"))
    return np.asarray(values, dtype=float)


def _qhx_delta_intervals(parameters: np.ndarray,
                         parameter_covariance: np.ndarray) -> list[dict[str, float]]:
    point = _qhx_vector(parameters)
    gradient = np.empty((point.size, parameters.size), dtype=float)
    for index, value in enumerate(parameters):
        step = 2e-6 * max(1.0, abs(float(value)))
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += step
        minus[index] -= step
        gradient[:, index] = (_qhx_vector(plus) - _qhx_vector(minus)) / (2.0 * step)
    variances = np.einsum("ij,jk,ik->i", gradient, parameter_covariance, gradient)
    standard_errors = np.sqrt(np.maximum(variances, 0.0))
    pointwise_z = float(norm.isf(QHX_POINTWISE_ALPHA))
    simultaneous_z = float(norm.isf(QHX_SIMULTANEOUS_ALPHA / point.size))
    rows = []
    for index, (estimate, standard_error) in enumerate(zip(point, standard_errors)):
        rows.append({
            "estimate": float(estimate),
            "delta_standard_error": float(standard_error),
            "upper_97_5_pointwise": float(estimate + pointwise_z * standard_error),
            "upper_95_bonferroni_72": float(estimate + simultaneous_z * standard_error),
        })
    return rows


def _validate_raw_snapshot(snapshot: SourceByteSnapshot, *, manifest_name: str,
                           matrix_name: str, contract_name: str):
    if not isinstance(snapshot, SourceByteSnapshot):
        raise ValueError("snapshot must be a SourceByteSnapshot")
    manifest, _ = _strict_json(snapshot, manifest_name)
    if set(manifest) != _MANIFEST_KEYS or manifest.get("schema") != RAW_MANIFEST_SCHEMA:
        raise ValueError("two-sector manifest keys or schema mismatch")
    split_id = manifest.get("split_id")
    if (not isinstance(split_id, str) or not split_id.upper().startswith("F")
            or any(marker in split_id.lower()
                   for marker in ("gap", "validation", "holdout", "sealed", "confirmation"))):
        raise ValueError("two-sector reader accepts only an open F development/calibration split")
    if (manifest.get("geometry") != "S16T16"
            or manifest.get("action") != "wilson_gauge_beta6"
            or manifest.get("boundary_conditions") != "periodic"
            or manifest.get("radius_order") != list(R_VALUES)
            or manifest.get("temporal_extents") != list(NT_VALUES)
            or manifest.get("time_slices") != list(T_VALUES)
            or manifest.get("dimension_order") != "block,Nt,R,T,i,j"):
        raise ValueError("two-sector manifest identity, geometry or axis order mismatch")
    contract, contract_bytes = _strict_json(snapshot, contract_name)
    _validate_contract(contract)
    if sha256(contract_bytes).hexdigest() != manifest.get("lane_contract_sha256"):
        raise ValueError("captured lane contract does not match manifest hash")
    matrix_bytes = snapshot.read_bytes(matrix_name)
    matrix_hash = sha256(matrix_bytes).hexdigest()
    if matrix_hash != snapshot.expected_hashes.get(matrix_name):
        raise ValueError("matrix bytes do not match their captured hash")
    if matrix_hash != manifest.get("matrix_sha256"):
        raise ValueError("matrix bytes do not match the manifest hash")
    try:
        validate_npz_payload(matrix_bytes)
    except ValueError as exc:
        raise ValueError(f"two-sector NPZ preflight failed: {exc}") from exc
    try:
        archive = np.load(BytesIO(matrix_bytes), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("two-sector NPZ is invalid") from exc
    with archive:
        if set(archive.files) != {
            "schema", "block_ids", "correlator_matrices", "R_values",
            "temporal_extents", "time_slices",
        }:
            raise ValueError("two-sector NPZ keys mismatch")
        if str(np.asarray(archive["schema"]).item()) != RAW_SCHEMA:
            raise ValueError("two-sector raw matrix schema mismatch")
        block_ids = np.asarray(archive["block_ids"])
        matrices = np.asarray(archive["correlator_matrices"])
        radii = np.asarray(archive["R_values"])
        temporal_extents = np.asarray(archive["temporal_extents"])
        times = np.asarray(archive["time_slices"])
    if (block_ids.ndim != 1 or block_ids.dtype.kind not in "iu" or len(block_ids) < 64
            or np.any(np.diff(block_ids) <= 0)):
        raise ValueError("block IDs must be at least 64 strictly increasing integers")
    expected_shape = (len(block_ids), len(NT_VALUES), len(R_VALUES), len(T_VALUES), 3, 3)
    if (matrices.shape != expected_shape or matrices.dtype.kind != "c"
            or not np.isfinite(matrices).all()):
        raise ValueError(f"correlator matrices must be finite complex data of shape {expected_shape}")
    if (radii.tolist() != list(R_VALUES) or temporal_extents.tolist() != list(NT_VALUES)
            or times.tolist() != list(T_VALUES)):
        raise ValueError("two-sector NPZ axis order differs from the frozen reader contract")
    for matrix in matrices.reshape(-1, 3, 3):
        if not np.allclose(matrix, matrix.conj().T, rtol=1e-10, atol=1e-12):
            raise ValueError("raw finite-Nt matrices contain non-Hermitian entries")
    return manifest, matrices


def fit_two_sector_snapshot(
    snapshot: SourceByteSnapshot, *, manifest_name: str = "manifest",
    matrix_name: str = "raw_matrices", contract_name: str = "lane_contract",
    objective_mode: str = "diagonal",
) -> dict[str, Any]:
    """Fit a 2-vacuum x 2-static trace to one captured raw-matrix snapshot.

    ``objective_mode='diagonal'`` preserves the original pilot. The optional
    ``'regularized_full_covariance'`` mode estimates the covariance of the
    block mean, applies fixed 25% diagonal shrinkage and a 1e-10 relative
    eigenvalue floor, and reports asymptotic delta intervals for q/h/x. Those
    intervals are diagnostic only and have no finite-sample coverage claim.
    """
    if objective_mode not in {"diagonal", "regularized_full_covariance"}:
        raise ValueError("objective_mode must be diagonal or regularized_full_covariance")
    manifest, matrices = _validate_raw_snapshot(
        snapshot, manifest_name=manifest_name,
        matrix_name=matrix_name, contract_name=contract_name,
    )
    blocks = _observed_vectors(matrices)
    mean = blocks.mean(axis=0)
    covariance_diagnostics = None
    covariance_cholesky = None
    if objective_mode == "diagonal":
        standard_error = blocks.std(axis=0, ddof=1) / math.sqrt(len(blocks))
        positive = standard_error[standard_error > 0]
        scale_floor = (float(np.max(positive)) * 1e-8 if positive.size
                       else max(float(np.max(np.abs(mean))) * 1e-10, 1e-15))
        weights = 1.0 / np.maximum(standard_error, scale_floor)
    else:
        mean_covariance, covariance_diagnostics = _regularized_mean_covariance(blocks)
        covariance_cholesky = cholesky(mean_covariance, lower=True, check_finite=True)

    # Initialize the m=0 static spectrum from the matched-Nt=32 forward fit.
    static_guesses, overlap_guesses = [], []
    fit_starts = []
    for radius_index in range(len(R_VALUES)):
        nt_index = NT_VALUES.index(32)
        forward_blocks = np.stack([
            np.concatenate([
                _matrix_vector(matrices[block, nt_index, radius_index, time_index])
                for time_index in range(len(T_VALUES))
            ])
            for block in range(len(blocks))
        ])
        forward_fit = forward.fit_joint_spectrum(
            forward_blocks, time_slices=T_VALUES, n_states=STATIC_STATES,
            operator_indices=(0, 1, 2),
        )
        fit_starts.append({"R": R_VALUES[radius_index], **{
            key: forward_fit.get(key) for key in
            ("success", "message", "nfev", "rank", "parameter_count")
        }})
        if not forward_fit.get("success"):
            return {
                "schema": FIT_RESULT_SCHEMA, "status": "fit_initialization_failed",
                "certified": False, "physical_score": None,
                "split_id": manifest["split_id"],
                "source_hashes": {
                    manifest_name: snapshot.expected_hashes[manifest_name],
                    matrix_name: snapshot.expected_hashes[matrix_name],
                    contract_name: snapshot.expected_hashes[contract_name],
                },
                "denominator": {"planned": 1, "valid": 0, "failed": 1, "abstained": 0},
                "qhx_denominator": {
                    "planned": len(R_VALUES) * OPERATOR_COUNT * 2 * 2,
                    "valid": 0, "failed": 0,
                    "abstained": len(R_VALUES) * OPERATOR_COUNT * 2 * 2,
                },
                "initializer_fits": fit_starts,
                "fit": None, "qhx_rows": [],
                "open_gates": [
                    "forward Nt=32 initializer did not yield a full-rank 2-state fit",
                    "no two-sector optimization was attempted",
                    "source/sink endpoint convention and coverage remain unreviewed",
                ],
            }
        static_guesses.append(np.asarray(forward_fit["energies"], dtype=float))
        overlap_pairs = np.asarray(forward_fit["overlaps"], dtype=float)
        if overlap_pairs.shape != (STATIC_STATES, OPERATOR_COUNT, 2):
            raise ValueError("forward-fit overlap initializer has an invalid complex-pair shape")
        overlaps = overlap_pairs[..., 0] + 1j * overlap_pairs[..., 1]
        overlap_guesses.append(overlaps)

    lower = np.full(1 + len(R_VALUES) * (2 + 20), -np.inf)
    upper = np.full_like(lower, np.inf)
    lower[0] = -8.0
    upper[0] = 2.5
    cursor = 1
    for _ in R_VALUES:
        lower[cursor:cursor + 2] = -8.0
        upper[cursor:cursor + 2] = 2.5
        cursor += 2
        for _ in range(VACUUM_STATES * STATIC_STATES):
            lower[cursor] = -12.0
            upper[cursor] = 4.0
            lower[cursor + 1:cursor + 5] = -5.0
            upper[cursor + 1:cursor + 5] = 5.0
            cursor += 5

    attempts, successful = [], []
    for epsilon_guess, scale in ((0.15, 0.20), (0.30, 0.35), (0.55, 0.50)):
        initial = _initial_parameters(static_guesses, overlap_guesses,
                                      epsilon_guess, scale)

        def residual(parameters):
            raw = _model_vectors(parameters).ravel() - mean
            if covariance_cholesky is None:
                return raw * weights
            return solve_triangular(covariance_cholesky, raw, lower=True,
                                    check_finite=True)

        try:
            result = least_squares(
                residual, initial, bounds=(lower, upper), method="trf",
                x_scale="jac", max_nfev=5000, ftol=1e-10, xtol=1e-10, gtol=1e-10,
            )
            objective = float(np.dot(result.fun, result.fun))
            finite = bool(np.isfinite(objective) and np.isfinite(result.x).all())
            converged = bool(result.success and finite)
            singular = np.linalg.svd(result.jac, compute_uv=False)
            tolerance = (singular[0] * max(result.jac.shape) * np.finfo(float).eps
                         if singular.size else 0.0)
            rank = int(np.count_nonzero(singular > tolerance))
            attempt = {
                "epsilon_start": epsilon_guess, "excited_overlap_scale_start": scale,
                "optimizer_success": bool(result.success), "finite": finite,
                "converged": converged, "status_code": int(result.status),
                "nfev": int(result.nfev), "chi_square_diagonal": objective if finite else None,
                "jacobian_rank": rank, "parameter_count": int(result.x.size),
                "message": str(result.message),
            }
            attempts.append(attempt)
            if converged:
                successful.append((objective, result, singular, rank))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            attempts.append({
                "epsilon_start": epsilon_guess, "excited_overlap_scale_start": scale,
                "optimizer_success": False, "finite": False, "converged": False,
                "status_code": "exception", "nfev": 0, "chi_square_diagonal": None,
                "jacobian_rank": 0, "parameter_count": int(lower.size),
                "message": type(exc).__name__,
            })

    if not successful:
        return {
            "schema": FIT_RESULT_SCHEMA, "status": "two_sector_fit_failed",
            "certified": False, "physical_score": None,
            "split_id": manifest["split_id"],
            "source_hashes": {
                manifest_name: snapshot.expected_hashes[manifest_name],
                matrix_name: snapshot.expected_hashes[matrix_name],
                contract_name: snapshot.expected_hashes[contract_name],
            },
            "denominator": {"planned": 1, "valid": 0, "failed": 1, "abstained": 0},
            "qhx_denominator": {
                "planned": len(R_VALUES) * OPERATOR_COUNT * 2 * 2,
                "valid": 0, "failed": 0,
                "abstained": len(R_VALUES) * OPERATOR_COUNT * 2 * 2,
            },
            "initializer_fits": fit_starts, "fit_attempts": attempts,
            "fit": None, "qhx_rows": [],
            "open_gates": [
                "no optimizer start converged",
                "source/sink endpoint convention and coverage remain unreviewed",
            ],
        }

    objective, best, singular, rank = min(successful, key=lambda row: row[0])
    epsilon, static, amplitudes = _decode(best.x)
    full_rank = rank == best.x.size
    parameter_covariance = None
    qhx_intervals = None
    if objective_mode == "regularized_full_covariance" and full_rank:
        _, singular_values, right = np.linalg.svd(best.jac, full_matrices=False)
        parameter_covariance = (right.T * (1.0 / singular_values ** 2)) @ right
        qhx_intervals = _qhx_delta_intervals(best.x, parameter_covariance)
    qhx_rows = []
    qhx_failed = 0
    for radius_index, radius in enumerate(R_VALUES):
        for operator_index in range(OPERATOR_COUNT):
            for time_extent in (5, 6):
                for temporal_extent in NT_VALUES:
                    try:
                        cell = _component_values(
                            np.asarray((0.0, epsilon)), static[radius_index],
                            amplitudes[radius_index], operator_index,
                            time_extent, temporal_extent,
                        )
                    except (ValueError, OverflowError, FloatingPointError) as exc:
                        qhx_failed += 1
                        qhx_rows.append({
                            "R": radius, "operator_index": operator_index,
                            "T": time_extent, "N_t": temporal_extent,
                            "status": "failed", "reason": str(exc),
                        })
                    else:
                        qhx_rows.append({
                            "R": radius, "operator_index": operator_index,
                            "T": time_extent, "N_t": temporal_extent,
                            "status": "valid", **cell,
                        })
    if qhx_intervals is not None and len(qhx_rows) == 24:
        for cell_index, row in enumerate(qhx_rows):
            row["delta_intervals"] = {
                name: qhx_intervals[3 * cell_index + offset]
                for offset, name in enumerate(("q", "h", "x"))
            }
    denominator = {"planned": 1, "valid": 1 if full_rank else 0,
                   "failed": 0 if full_rank else 1, "abstained": 0}
    return {
        "schema": FIT_RESULT_SCHEMA,
        "fit_schema": FIT_SCHEMA,
        "status": "two_sector_fit_candidate" if full_rank else "fit_rank_deficient",
        "certified": False, "physical_score": None,
        "split_id": manifest["split_id"],
        "source_hashes": {
            manifest_name: snapshot.expected_hashes[manifest_name],
            matrix_name: snapshot.expected_hashes[matrix_name],
            contract_name: snapshot.expected_hashes[contract_name],
        },
        "denominator": denominator,
        "qhx_denominator": {
            "planned": len(R_VALUES) * OPERATOR_COUNT * 2 * 2,
            "valid": len(qhx_rows) - qhx_failed,
            "failed": qhx_failed, "abstained": 0,
        },
        "fit_contract": {
            "vacuum_state_count": VACUUM_STATES,
            "static_state_count": STATIC_STATES,
            "radius_order": list(R_VALUES), "temporal_extents": list(NT_VALUES),
            "time_slices": list(T_VALUES),
            "objective": ("diagonal inverse-SE weighted least squares over Hermitian 3x3 components"
                          if objective_mode == "diagonal" else
                          "regularized full-covariance GLS over Hermitian 3x3 components"),
            "objective_mode": objective_mode,
            "covariance_diagnostics": covariance_diagnostics,
            "parameter_count": int(best.x.size), "data_components": int(best.fun.size),
            "degrees_of_freedom": int(best.fun.size - best.x.size),
            "jacobian_rank": rank,
            "jacobian_condition_number": (float(singular[0] / singular[-1])
                                           if full_rank and singular[-1] > 0 else None),
            "chi_square_diagonal": objective,
            "goodness_of_fit_calibrated": False,
            "parameter_covariance_method": (
                "inverse whitened Jacobian information; asymptotic diagnostic only"
                if parameter_covariance is not None else None),
            "qhx_interval_method": (
                "finite-difference delta method; pointwise 97.5% and 95% Bonferroni across 72 q/h/x terms"
                if qhx_intervals is not None else None),
            "qhx_interval_coverage_calibrated": False,
        },
        "spectrum_fit": {
            "vacuum_energies": [0.0, float(epsilon)],
            "static_energies_by_R": {
                str(radius): static[index].tolist()
                for index, radius in enumerate(R_VALUES)
            },
            "transition_overlaps_by_R": {
                str(radius): [[
                    [[float(value.real), float(value.imag)] for value in vector]
                    for vector in amplitudes[index, m]
                ] for m in range(VACUUM_STATES)]
                for index, radius in enumerate(R_VALUES)
            },
        },
        "fit_attempts": attempts,
        "initializer_fits": fit_starts,
        "qhx_rows": qhx_rows,
        "open_gates": [
            "the two-sector endpoint factorization is a draft convention, not an independently reviewed exporter mapping",
            "diagonal fit weights ignore cross-component covariance and are not a calibrated uncertainty method",
            "two states per sector are a synthetic candidate; omitted-state and interval-coverage checks remain open",
            "source-file hashes do not authenticate provenance, reviewer identity, or authorization",
        ],
    }
