"""Synthetic joint positive-spectrum GLS and q-bound calibration pilot.

This is a candidate CPU method diagnostic, not a Wilson measurement reader or
physical evidence producer. It fits all independent Hermitian components of a
3x3 matrix-valued correlator jointly, with one shared set of energies and bare
operator overlaps, using an estimated block covariance for GLS whitening.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky, eigh, solve_triangular
from scipy.optimize import least_squares


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = TASK_DIR / "f_lane_joint_spectral_gls_20261126_v1"
TIME_SLICES = tuple(range(1, 8))
BLOCK_COUNTS = (64, 128)
NOISE_LEVELS = (0.01, 0.04)
PANELS_PER_CELL = 200
SHRINKAGE = 0.25
EIGENVALUE_FLOOR_RELATIVE = 1e-10
Z_UPPER_97_5 = 1.959963984540054
CALIBRATION_ALPHA = 0.025
TRUE_ENERGIES = np.asarray([0.18, 0.32, 0.49], dtype=float)
TRUE_OVERLAPS = np.asarray([
    [1.00, 0.80 + 0.04j, 0.55 - 0.03j],
    [0.72, -0.35 + 0.02j, 0.60 + 0.05j],
    [0.45, 0.62 - 0.06j, -0.50 + 0.03j],
], dtype=complex)
Q_OPERATORS = (0, 2)
Q_TIMES = (5, 6)


def _matrix_vector(matrix: object) -> np.ndarray:
    value = np.asarray(matrix, dtype=complex)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError("each correlator matrix must be finite complex 3x3")
    if not np.allclose(value, value.conj().T, rtol=1e-11, atol=1e-13):
        raise ValueError("correlator matrix must be Hermitian")
    return np.asarray([
        value[0, 0].real, value[1, 1].real, value[2, 2].real,
        value[0, 1].real, value[0, 1].imag,
        value[0, 2].real, value[0, 2].imag,
        value[1, 2].real, value[1, 2].imag,
    ], dtype=float)


def _vector_matrices(vector: object, times: tuple[int, ...] = TIME_SLICES) -> np.ndarray:
    raw = np.asarray(vector, dtype=float)
    if raw.shape != (9 * len(times),) or not np.all(np.isfinite(raw)):
        raise ValueError("matrix-vector payload has the wrong shape or non-finite values")
    matrices = []
    for chunk in raw.reshape(len(times), 9):
        matrix = np.zeros((3, 3), dtype=complex)
        matrix[0, 0], matrix[1, 1], matrix[2, 2] = chunk[:3]
        matrix[0, 1] = chunk[3] + 1j * chunk[4]
        matrix[0, 2] = chunk[5] + 1j * chunk[6]
        matrix[1, 2] = chunk[7] + 1j * chunk[8]
        matrix[1, 0] = matrix[0, 1].conjugate()
        matrix[2, 0] = matrix[0, 2].conjugate()
        matrix[2, 1] = matrix[1, 2].conjugate()
        matrices.append(matrix)
    return np.stack(matrices)


def _spectrum_matrices(energies: object, overlaps: object,
                       times: tuple[int, ...] = TIME_SLICES) -> np.ndarray:
    energy = np.asarray(energies, dtype=float)
    z = np.asarray(overlaps, dtype=complex)
    if (energy.ndim != 1 or energy.size not in (2, 3)
            or z.shape != (energy.size, 3)
            or not np.isfinite(energy).all() or not np.isfinite(z).all()
            or np.any(energy <= 0.0) or np.any(np.diff(energy) <= 0.0)):
        raise ValueError("spectrum requires 2/3 finite increasing positive energies and matching overlaps")
    result = []
    for time in times:
        weights = np.exp(-energy * int(time))
        result.append(np.einsum("n,ni,nj->ij", weights, z, z.conj()))
    return np.asarray(result)


def _unpack_parameters(parameters: object, n_states: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(parameters, dtype=float)
    inferred_states = p.size // 6 if p.ndim == 1 and p.size % 6 == 0 else 0
    n_states = inferred_states if n_states is None else n_states
    if (n_states not in (2, 3) or p.shape != (6 * n_states,)
            or not np.isfinite(p).all()):
        raise ValueError("joint fit parameter vector must contain 12 or 18 finite values")
    energies = np.cumsum(p[:n_states])
    overlaps = np.zeros((n_states, 3), dtype=complex)
    for state in range(n_states):
        start = n_states + 5 * state
        overlaps[state, 0] = p[start]
        overlaps[state, 1] = p[start + 1] + 1j * p[start + 2]
        overlaps[state, 2] = p[start + 3] + 1j * p[start + 4]
    return energies, overlaps


def _pack_parameters(energies: np.ndarray, overlaps: np.ndarray) -> np.ndarray:
    n_states = int(np.asarray(energies).size)
    if n_states not in (2, 3) or np.asarray(overlaps).shape != (n_states, 3):
        raise ValueError("joint fit supports two or three states with three operator overlaps")
    packed = [float(energies[0])]
    packed.extend(float(energies[index] - energies[index - 1])
                  for index in range(1, n_states))
    z = np.asarray(overlaps, dtype=complex).copy()
    for state in range(n_states):
        z[state] *= np.exp(-1j * np.angle(z[state, 0]))
        packed.extend([float(z[state, 0].real), float(z[state, 1].real),
                       float(z[state, 1].imag), float(z[state, 2].real),
                       float(z[state, 2].imag)])
    return np.asarray(packed, dtype=float)


def _model_vector(parameters: object, times: tuple[int, ...] = TIME_SLICES,
                  n_states: int | None = None) -> np.ndarray:
    energies, overlaps = _unpack_parameters(parameters, n_states=n_states)
    matrices = _spectrum_matrices(energies, overlaps, times=times)
    return np.concatenate([_matrix_vector(matrix) for matrix in matrices])


def _initial_from_gevp(mean_vector: np.ndarray, times: tuple[int, ...] = TIME_SLICES,
                       n_states: int = 3) -> np.ndarray:
    matrices = _vector_matrices(mean_vector, times=times)
    c0, c1 = matrices[0], matrices[1]
    metric_values, metric_vectors = eigh(c0, check_finite=True)
    metric_order = np.argsort(metric_values)[::-1][:n_states]
    retained_metric_values = metric_values[metric_order]
    if np.any(retained_metric_values <= 0.0) or not np.all(np.isfinite(retained_metric_values)):
        raise ValueError("C(t0) has fewer than n_states positive metric directions")
    basis = metric_vectors[:, metric_order]
    reduced_c0 = basis.conj().T @ c0 @ basis
    reduced_c1 = basis.conj().T @ c1 @ basis
    lower = cholesky(reduced_c0, lower=True, check_finite=True)
    left = solve_triangular(lower, reduced_c1, lower=True, check_finite=True)
    whitened = solve_triangular(lower, left.conj().T, lower=True,
                                 check_finite=True).conj().T
    eigenvalues, eigenvectors = eigh(whitened, check_finite=True)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    eigenvalues = eigenvalues[:n_states]
    eigenvectors = eigenvectors[:, :n_states]
    if np.any(eigenvalues <= 0.0) or not np.all(np.isfinite(eigenvalues)):
        raise ValueError("t0/t0+1 GEVP has non-positive or non-finite eigenvalues")
    energies = -np.log(eigenvalues)
    if np.any(energies <= 0.0) or np.any(np.diff(energies) <= 0.0):
        raise ValueError("GEVP initialization did not yield ordered positive energies")
    overlaps = (basis @ lower @ eigenvectors * np.exp(0.5 * energies)[None, :]).T
    return _pack_parameters(energies, overlaps)


def _regularized_covariance(block_vectors: object) -> tuple[np.ndarray, dict[str, float]]:
    blocks = np.asarray(block_vectors, dtype=float)
    if blocks.ndim != 2 or blocks.shape[0] < 3 or not np.isfinite(blocks).all():
        raise ValueError("block_vectors must be a finite matrix with at least three blocks")
    raw = np.cov(blocks, rowvar=False, ddof=1) / blocks.shape[0]
    diagonal = np.diag(np.diag(raw))
    shrunk = (1.0 - SHRINKAGE) * raw + SHRINKAGE * diagonal
    values, vectors = np.linalg.eigh(0.5 * (shrunk + shrunk.T))
    largest = float(values[-1])
    if largest <= 0.0 or not math.isfinite(largest):
        raise ValueError("estimated mean covariance has no positive eigenvalue")
    floor = EIGENVALUE_FLOOR_RELATIVE * largest
    floored = np.maximum(values, floor)
    covariance = (vectors * floored) @ vectors.T
    condition = float(floored[-1] / floored[0])
    return covariance, {
        "raw_min_eigenvalue": float(values[0]),
        "regularized_min_eigenvalue": float(floored[0]),
        "max_eigenvalue": largest,
        "condition_number": condition,
        "floor_active": float(np.count_nonzero(values < floor)),
    }


def _q_value(parameters: np.ndarray, operator_index: int, time: int,
              n_states: int | None = None) -> float:
    energies, overlaps = _unpack_parameters(parameters, n_states=n_states)
    ground = abs(overlaps[0, operator_index])
    if not math.isfinite(ground) or ground <= 1e-12:
        raise ValueError("bare ground-state overlap is zero or unresolved")
    ratios = np.abs(overlaps[1:, operator_index] / overlaps[0, operator_index]) ** 2
    gaps = energies[1:] - energies[0]
    return float(np.sum(ratios * np.exp(-gaps * time)))


def _q_gradient(parameters: np.ndarray, operator_index: int, time: int,
                n_states: int | None = None) -> np.ndarray:
    gradient = np.empty_like(parameters)
    for index in range(parameters.size):
        step = 2e-6 * max(1.0, abs(float(parameters[index])))
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (_q_value(plus, operator_index, time, n_states=n_states)
                           - _q_value(minus, operator_index, time, n_states=n_states)) / (2.0 * step)
    return gradient


def fit_joint_spectrum(block_vectors: object, *,
                       time_slices: tuple[int, ...] = TIME_SLICES,
                       n_states: int = 3,
                       operator_indices: tuple[int, ...] = Q_OPERATORS) -> dict[str, Any]:
    """Fit a common 2/3-state spectrum with full real/imag GLS whitening."""
    blocks = np.asarray(block_vectors, dtype=float)
    times = tuple(time_slices)
    dimension = 9 * len(times)
    n_parameters = 6 * n_states if n_states in (2, 3) else -1
    if (len(times) < 3 or tuple(sorted(set(times))) != times
            or any(type(time) is not int or time < 1 for time in times)):
        raise ValueError("time_slices must contain at least three increasing unique positive integers")
    if (blocks.ndim != 2 or blocks.shape[1] != dimension or not np.isfinite(blocks).all()):
        raise ValueError(f"block_vectors must have shape (n_blocks,{dimension}) and finite values")
    if blocks.shape[0] <= dimension:
        raise ValueError(f"at least {dimension + 1} blocks are required for the full raw sample covariance")
    if (n_parameters < 0 or not operator_indices
            or any(type(index) is not int or index not in (0, 1, 2) for index in operator_indices)
            or len(set(operator_indices)) != len(operator_indices)):
        raise ValueError("n_states must be 2/3 and operator_indices must be unique basis indices 0..2")
    mean = blocks.mean(axis=0)
    covariance, covariance_diagnostics = _regularized_covariance(blocks)
    covariance_cholesky = cholesky(covariance, lower=True, check_finite=True)
    initial = _initial_from_gevp(mean, times=times, n_states=n_states)
    lower = np.full(n_parameters, -np.inf)
    upper = np.full(n_parameters, np.inf)
    lower[:n_states] = [0.005] + [0.002] * (n_states - 1)
    upper[:n_states] = 2.0
    for state in range(n_states):
        lower[n_states + 5 * state] = 1e-8
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("GEVP initialization falls outside the declared fit domain")

    def residual(parameters: np.ndarray) -> np.ndarray:
        raw = _model_vector(parameters, times=times, n_states=n_states) - mean
        return solve_triangular(covariance_cholesky, raw, lower=True,
                                check_finite=True)

    fit = least_squares(residual, initial, bounds=(lower, upper), method="trf",
                        x_scale="jac", max_nfev=1000, ftol=1e-11,
                        xtol=1e-11, gtol=1e-11)
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    tolerance = singular[0] * max(fit.jac.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(singular > tolerance))
    dof = int(fit.fun.size - fit.x.size)
    reduced_chi_square = (float(np.dot(fit.fun, fit.fun) / dof) if dof > 0 else math.inf)
    result: dict[str, Any] = {
        "success": bool(fit.success and rank == fit.x.size),
        "optimizer_success": bool(fit.success),
        "message": str(fit.message),
        "nfev": int(fit.nfev),
        "rank": rank,
        "parameter_count": int(fit.x.size),
        "n_states": n_states,
        "time_slices": list(times),
        "dof": dof,
        "chi_square": float(np.dot(fit.fun, fit.fun)),
        "reduced_chi_square": reduced_chi_square,
        "jacobian_condition_number": (float(singular[0] / singular[-1])
                                       if rank == fit.x.size else None),
        "covariance_diagnostics": covariance_diagnostics,
        "parameters": fit.x.tolist(),
    }
    if not result["success"]:
        result["q_cells"] = []
        return result

    _, right = np.linalg.svd(fit.jac, full_matrices=False)[1:]
    parameter_covariance = (right.T * (1.0 / singular ** 2)) @ right
    q_cells = []
    for operator_index in operator_indices:
        for time in Q_TIMES:
            if time not in times:
                continue
            estimate = _q_value(fit.x, operator_index, time, n_states=n_states)
            gradient = _q_gradient(fit.x, operator_index, time, n_states=n_states)
            variance = float(gradient @ parameter_covariance @ gradient)
            standard_error = math.sqrt(max(variance, 0.0))
            q_cells.append({
                "operator_index": operator_index,
                "time": time,
                "q_estimate": estimate,
                "q_standard_error_delta": standard_error,
                "q_upper_97_5_pointwise": estimate + Z_UPPER_97_5 * standard_error,
            })
    result["q_cells"] = q_cells
    result["energies"] = _unpack_parameters(fit.x, n_states=n_states)[0].tolist()
    result["overlaps"] = [[
        [float(value.real), float(value.imag)] for value in row
    ] for row in _unpack_parameters(fit.x, n_states=n_states)[1]]
    return result


def _design_covariance(seed: int, n_blocks: int, noise_level: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    means = _spectrum_matrices(TRUE_ENERGIES, TRUE_OVERLAPS)
    center = np.concatenate([_matrix_vector(matrix) for matrix in means])
    per_time_scale = np.asarray([
        max(float(np.linalg.norm(center[i * 9:(i + 1) * 9])) / 3.0, 1e-10)
        for i in range(len(TIME_SLICES))
    ])
    scales = np.repeat(noise_level * per_time_scale, 9)
    rt = np.fromfunction(lambda i, j: 0.35 ** np.abs(i - j),
                         (len(TIME_SLICES), len(TIME_SLICES)))
    re = np.fromfunction(lambda i, j: 0.20 ** np.abs(i - j), (9, 9))
    correlation = np.kron(rt, re)
    covariance = np.outer(scales, scales) * correlation
    return rng.multivariate_normal(np.zeros(center.size), covariance,
                                   size=n_blocks) + center


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("invalid binomial denominator")
    z = 1.959963984540054
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def _split_conformal_upper_multiplier(scores: object,
                                      alpha: float = CALIBRATION_ALPHA) -> float | None:
    """Finite-sample one-sided split-conformal quantile; None means +infinity."""
    values = np.asarray(scores, dtype=float)
    if (values.ndim != 1 or values.size == 0 or not math.isfinite(alpha)
            or not 0.0 < alpha < 1.0 or np.isnan(values).any()
            or np.isneginf(values).any()):
        raise ValueError("scores and alpha are invalid for a conformal upper quantile")
    rank = int(math.ceil((values.size + 1) * (1.0 - alpha)))
    if rank > values.size:
        return None
    quantile = float(np.sort(values)[rank - 1])
    return None if math.isinf(quantile) else quantile


def _coverage_plot(summary_rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = {64: "#0072B2", 128: "#D55E00"}
    markers = {0.01: "o", 0.04: "s"}
    for row_index, operator_index in enumerate(Q_OPERATORS):
        for col_index, metric in enumerate(("coverage", "upper_ratio")):
            ax = axes[row_index, col_index]
            for n_blocks in BLOCK_COUNTS:
                for noise in NOISE_LEVELS:
                    subset = [row for row in summary_rows
                              if row["operator_index"] == operator_index
                              and row["n_blocks"] == n_blocks
                              and row["noise_level"] == noise]
                    subset.sort(key=lambda row: row["time"])
                    xs = np.asarray([row["time"] for row in subset], dtype=float)
                    if metric == "coverage":
                        values = np.asarray([row["upper_coverage"] for row in subset])
                        lows = np.asarray([row["coverage_mc95_low"] for row in subset])
                        highs = np.asarray([row["coverage_mc95_high"] for row in subset])
                        if any(value is None for value in values):
                            ax.text(0.5, 0.5, "no valid q fits", transform=ax.transAxes,
                                    ha="center", fontsize=9)
                            continue
                        ax.errorbar(xs, values, yerr=[values - lows, highs - values],
                                    color=colors[n_blocks], marker=markers[noise],
                                    linestyle="-" if noise == 0.01 else "--",
                                    label=f"blocks={n_blocks}, noise={noise:g}", capsize=2)
                    else:
                        values = np.asarray([row["median_upper_over_true_q"] for row in subset])
                        if any(value is None for value in values):
                            ax.text(0.5, 0.5, "no valid q fits", transform=ax.transAxes,
                                    ha="center", fontsize=9)
                            continue
                        ax.plot(xs, values, color=colors[n_blocks], marker=markers[noise],
                                linestyle="-" if noise == 0.01 else "--",
                                label=f"blocks={n_blocks}, noise={noise:g}")
            if metric == "coverage":
                ax.axhline(0.975, color="#555555", linestyle=":", label="pointwise target 97.5%")
                ax.set_ylim(0.75, 1.005)
                ax.set_ylabel("empirical one-sided upper coverage")
                ax.set_title(f"bare operator {operator_index}: pointwise MC coverage")
            else:
                ax.axhline(1.0, color="#555555", linestyle=":", label="upper = truth")
                ax.set_ylabel("median q upper / true q")
                ax.set_title(f"bare operator {operator_index}: bound conservatism")
            ax.set_xticks(Q_TIMES)
            ax.set_xlabel("T")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
    fig.suptitle("Synthetic independent-block GLS pilot; no finite-Nt or physical data")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _calibration_plot(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = {64: "#0072B2", 128: "#D55E00"}
    markers = {0.01: "o", 0.04: "s"}
    for row_index, operator_index in enumerate(Q_OPERATORS):
        for col_index, time in enumerate(Q_TIMES):
            ax = axes[row_index, col_index]
            for n_blocks in BLOCK_COUNTS:
                for noise in NOISE_LEVELS:
                    row = next(item for item in rows
                               if item["operator_index"] == operator_index
                               and item["time"] == time
                               and item["n_blocks"] == n_blocks
                               and item["noise_level"] == noise)
                    x = n_blocks + (0.0 if noise == 0.01 else 8.0)
                    if row["baseline_holdout_coverage"] is not None:
                        ax.errorbar(
                            x - 2.0, row["baseline_holdout_coverage"],
                            yerr=[[max(row["baseline_holdout_coverage"] - row["baseline_mc95_low"], 0.0)],
                                  [max(row["baseline_mc95_high"] - row["baseline_holdout_coverage"], 0.0)]],
                            color=colors[n_blocks], marker=markers[noise], linestyle="none",
                            capsize=3, label="delta method" if (n_blocks, noise) == (64, 0.01) else None,
                        )
                    if row["conformal_holdout_coverage"] is not None:
                        ax.errorbar(
                            x + 2.0, row["conformal_holdout_coverage"],
                            yerr=[[max(row["conformal_holdout_coverage"] - row["conformal_mc95_low"], 0.0)],
                                  [max(row["conformal_mc95_high"] - row["conformal_holdout_coverage"], 0.0)]],
                            color=colors[n_blocks], marker=markers[noise], linestyle="none",
                            markerfacecolor="white", capsize=3,
                            label="split conformal" if (n_blocks, noise) == (64, 0.01) else None,
                        )
            ax.axhline(1.0 - CALIBRATION_ALPHA, color="#555555", linestyle=":",
                       label="target 97.5%" if (row_index, col_index) == (0, 0) else None)
            ax.set_ylim(0.70, 1.01)
            ax.set_xticks([64, 72, 128, 136], ["64/.01", "64/.04", "128/.01", "128/.04"])
            ax.set_xlabel("blocks / relative noise")
            ax.set_ylabel("holdout empirical upper coverage")
            ax.set_title(f"bare operator {operator_index}, T={time}")
            ax.grid(alpha=0.2)
            if row_index == 0 and col_index == 0:
                ax.legend(fontsize=8)
            if all(row["conformal_holdout_coverage"] is None for row in rows):
                ax.text(0.5, 0.45, "Conformal abstains: calibration sample too small",
                        transform=ax.transAxes, ha="center", fontsize=8)
    fig.suptitle("Independent synthetic calibration/holdout split; not physical or simultaneous coverage")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(output: str | Path = DEFAULT_OUT, *, panels_per_cell: int = PANELS_PER_CELL) -> dict:
    if isinstance(panels_per_cell, bool) or not isinstance(panels_per_cell, int) or panels_per_cell < 20:
        raise ValueError("panels_per_cell must be an integer of at least 20")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol = {
        "schema": "wilson-f-joint-spectral-gls-pilot-v1",
        "classification": "synthetic CPU method diagnostic; not a Wilson reader, receipt, or certification",
        "physical_score": None,
        "forward_model": "C_ij(t)=sum_n Z_ni Z*_nj exp(-E_n t), exactly three positive states, one shared spectrum across all 3x3 matrix elements",
        "time_slices": list(TIME_SLICES),
        "bare_operator_indices_for_q": list(Q_OPERATORS),
        "q_times": list(Q_TIMES),
        "true_energies": TRUE_ENERGIES.tolist(),
        "true_overlaps_real_imag": [[[float(z.real), float(z.imag)] for z in row] for row in TRUE_OVERLAPS],
        "block_counts": list(BLOCK_COUNTS),
        "relative_noise_levels": list(NOISE_LEVELS),
        "panels_per_cell": panels_per_cell,
        "covariance": {
            "synthetic_time_correlation": "0.35^abs(t_i-t_j)",
            "synthetic_matrix_component_correlation": "0.20^abs(a_i-a_j)",
            "estimated_from_same_independent_blocks": True,
            "diagonal_shrinkage": SHRINKAGE,
            "eigenvalue_floor_relative": EIGENVALUE_FLOOR_RELATIVE,
            "meaning": "illustrative numerical regularization only; not an approved production rule",
        },
        "interval": "pointwise one-sided q_upper=qhat+1.95996398454*delta_method_se; not simultaneous and not bootstrap calibrated",
        "split_conformal_calibration": {
            "alpha": CALIBRATION_ALPHA,
            "training_replicates": "replicate indices below floor(panels_per_cell/2)",
            "validation_replicates": "remaining replicate indices; disjoint deterministic seeds",
            "score": "(q_true - q_hat) / delta_method_se, with failed or missing calibration q mapped to +infinity",
            "quantile_rank": "ceil((n_calibration + 1) * (1-alpha)); abstain if rank exceeds planned calibration count",
            "scope": "marginal within each exact synthetic noise/block/operator/time stratum under exchangeability; not familywise or physical calibration",
        },
        "seeds": [2026112600 + ni * 100000 + bi * 10000 + replicate
                  for ni in range(len(NOISE_LEVELS))
                  for bi in range(len(BLOCK_COUNTS))
                  for replicate in range(panels_per_cell)],
        "source_sha256": {Path(__file__).name: source_hash},
        "omitted": ["finite-Nt thermal h", "mixed-sector x", "N_t=32 matched data", "force/reference propagation", "chain dependence", "source exporter review", "physical source bytes"],
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    shutil.copyfile(__file__, out / f"snapshot_{Path(__file__).name}")

    replicate_rows: list[dict] = []
    planned_panels = len(NOISE_LEVELS) * len(BLOCK_COUNTS) * panels_per_cell
    for noise_index, noise in enumerate(NOISE_LEVELS):
        for block_index, n_blocks in enumerate(BLOCK_COUNTS):
            for replicate in range(panels_per_cell):
                seed = 2026112600 + noise_index * 100000 + block_index * 10000 + replicate
                blocks = _design_covariance(seed, n_blocks, noise)
                try:
                    fit = fit_joint_spectrum(blocks)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    fit = {"success": False, "message": str(exc), "q_cells": [],
                           "nfev": 0, "rank": 0, "jacobian_condition_number": None,
                           "reduced_chi_square": None,
                           "covariance_diagnostics": {}}
                fit_by_key = {(row["operator_index"], row["time"]): row
                              for row in fit.get("q_cells", [])}
                for operator_index in Q_OPERATORS:
                    for time in Q_TIMES:
                        cell = fit_by_key.get((operator_index, time))
                        q_true = _q_value(_pack_parameters(TRUE_ENERGIES, TRUE_OVERLAPS),
                                          operator_index, time)
                        replicate_rows.append({
                            "seed": seed, "noise_level": noise, "n_blocks": n_blocks,
                            "replicate": replicate, "operator_index": operator_index,
                            "time": time, "fit_valid": bool(fit.get("success", False)),
                            "fit_message": str(fit.get("message", "")),
                            "fit_nfev": fit.get("nfev", 0), "fit_rank": fit.get("rank", 0),
                            "fit_condition_number": fit.get("jacobian_condition_number"),
                            "reduced_chi_square": fit.get("reduced_chi_square"),
                            "q_true": q_true,
                            "q_estimate": None if cell is None else cell["q_estimate"],
                            "q_delta_se": None if cell is None else cell["q_standard_error_delta"],
                            "q_upper_97_5": None if cell is None else cell["q_upper_97_5_pointwise"],
                            "covered": None if cell is None else cell["q_upper_97_5_pointwise"] >= q_true,
                            "planned_q_cell": True,
                        })

    summary_rows = []
    for operator_index in Q_OPERATORS:
        for time in Q_TIMES:
            for n_blocks in BLOCK_COUNTS:
                for noise in NOISE_LEVELS:
                    subset = [row for row in replicate_rows
                              if row["operator_index"] == operator_index
                              and row["time"] == time and row["n_blocks"] == n_blocks
                              and row["noise_level"] == noise]
                    valid = [row for row in subset if row["fit_valid"] and row["covered"] is not None]
                    covered = sum(bool(row["covered"]) for row in valid)
                    low, high = _wilson_interval(covered, len(valid)) if valid else (None, None)
                    q_true = subset[0]["q_true"]
                    ratios = [row["q_upper_97_5"] / q_true for row in valid]
                    estimates = [row["q_estimate"] / q_true for row in valid]
                    summary_rows.append({
                        "operator_index": operator_index, "time": time,
                        "n_blocks": n_blocks, "noise_level": noise,
                        "planned": len(subset), "valid": len(valid),
                        "failed": sum(not row["fit_valid"] for row in subset),
                        "abstained": len(subset) - len(valid) - sum(not row["fit_valid"] for row in subset),
                        "upper_coverage": covered / len(valid) if valid else None,
                        "coverage_mc95_low": low, "coverage_mc95_high": high,
                        "median_q_estimate_over_true": float(np.median(estimates)) if estimates else None,
                        "median_upper_over_true_q": float(np.median(ratios)) if ratios else None,
                        "median_fit_condition_number": float(np.median([
                            row["fit_condition_number"] for row in valid
                            if row["fit_condition_number"] is not None
                        ])) if valid else None,
                        "classification": "synthetic exact-three-state independent-block pointwise delta-method pilot",
                    })

    calibration_rows = []
    split_at = panels_per_cell // 2
    for operator_index in Q_OPERATORS:
        for time in Q_TIMES:
            for n_blocks in BLOCK_COUNTS:
                for noise in NOISE_LEVELS:
                    subset = [row for row in replicate_rows
                              if row["operator_index"] == operator_index
                              and row["time"] == time and row["n_blocks"] == n_blocks
                              and row["noise_level"] == noise]
                    training = [row for row in subset if row["replicate"] < split_at]
                    validation = [row for row in subset if row["replicate"] >= split_at]
                    scores = []
                    calibration_valid = calibration_failed = calibration_abstained = 0
                    for row in training:
                        if not row["fit_valid"]:
                            calibration_failed += 1
                            scores.append(math.inf)
                        elif (row["q_estimate"] is None or row["q_delta_se"] is None
                              or not math.isfinite(float(row["q_estimate"]))
                              or not math.isfinite(float(row["q_delta_se"]))):
                            calibration_abstained += 1
                            scores.append(math.inf)
                        else:
                            calibration_valid += 1
                            se = float(row["q_delta_se"])
                            scores.append((float(row["q_true"]) - float(row["q_estimate"])) / se
                                          if se > 0.0 else math.inf)
                    multiplier = _split_conformal_upper_multiplier(scores)

                    validation_failed = validation_fit_valid = 0
                    baseline_covered = conformal_covered = 0
                    conformal_valid = conformal_abstained = 0
                    baseline_valid = 0
                    baseline_upper_ratios = []
                    conformal_upper_ratios = []
                    for row in validation:
                        if not row["fit_valid"]:
                            validation_failed += 1
                            continue
                        if (row["q_estimate"] is None or row["q_delta_se"] is None
                                or not math.isfinite(float(row["q_estimate"]))
                                or not math.isfinite(float(row["q_delta_se"]))):
                            continue
                        validation_fit_valid += 1
                        baseline_valid += 1
                        baseline_upper = (float(row["q_estimate"])
                                          + Z_UPPER_97_5 * float(row["q_delta_se"]))
                        baseline_covered += int(baseline_upper >= float(row["q_true"]))
                        baseline_upper_ratios.append(baseline_upper / float(row["q_true"]))
                        if multiplier is None:
                            continue
                        conformal_valid += 1
                        conformal_upper = (float(row["q_estimate"])
                                           + multiplier * float(row["q_delta_se"]))
                        conformal_covered += int(conformal_upper >= float(row["q_true"]))
                        conformal_upper_ratios.append(conformal_upper / float(row["q_true"]))
                    validation_abstained = len(validation) - validation_failed - validation_fit_valid
                    conformal_failed = validation_failed
                    if multiplier is None:
                        conformal_abstained += validation_fit_valid
                    else:
                        conformal_abstained += validation_abstained
                    baseline_low, baseline_high = (
                        _wilson_interval(baseline_covered, baseline_valid)
                        if baseline_valid else (None, None)
                    )
                    conformal_low, conformal_high = (
                        _wilson_interval(conformal_covered, conformal_valid)
                        if conformal_valid else (None, None)
                    )
                    calibration_rows.append({
                        "operator_index": operator_index, "time": time,
                        "n_blocks": n_blocks, "noise_level": noise,
                        "alpha": CALIBRATION_ALPHA,
                        "training_replicates": f"0:{split_at}",
                        "validation_replicates": f"{split_at}:{panels_per_cell}",
                        "calibration_planned": len(training),
                        "calibration_valid": calibration_valid,
                        "calibration_failed": calibration_failed,
                        "calibration_abstained": calibration_abstained,
                        "conformal_multiplier": multiplier,
                        "calibration_status": ("FINITE_SPLIT_CONFORMAL_MULTIPLIER"
                                               if multiplier is not None
                                               else "ABSTAIN_NO_FINITE_MULTIPLIER"),
                        "validation_planned": len(validation),
                        "validation_fit_valid": validation_fit_valid,
                        "baseline_valid": baseline_valid,
                        "baseline_failed": validation_failed,
                        "baseline_abstained": validation_abstained,
                        "baseline_holdout_coverage": (baseline_covered / baseline_valid
                                                       if baseline_valid else None),
                        "baseline_mc95_low": baseline_low,
                        "baseline_mc95_high": baseline_high,
                        "baseline_median_upper_over_true_q": (
                            float(np.median(baseline_upper_ratios)) if baseline_upper_ratios else None
                        ),
                        "conformal_valid": conformal_valid,
                        "conformal_failed": conformal_failed,
                        "conformal_abstained": conformal_abstained,
                        "conformal_holdout_coverage": (conformal_covered / conformal_valid
                                                        if conformal_valid else None),
                        "conformal_mc95_low": conformal_low,
                        "conformal_mc95_high": conformal_high,
                        "conformal_median_upper_over_true_q": (
                            float(np.median(conformal_upper_ratios)) if conformal_upper_ratios else None
                        ),
                        "classification": "independent synthetic split-calibration versus holdout",
                    })

    with (out / "replicates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(replicate_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(replicate_rows)
    with (out / "coverage_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(summary_rows)
    with (out / "calibration_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(calibration_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(calibration_rows)
    _coverage_plot(summary_rows, out / "coverage_and_interval_width.png")
    _calibration_plot(calibration_rows, out / "calibration_holdout_coverage.png")

    fit_planned = planned_panels
    panel_rows = [row for row in replicate_rows
                  if row["operator_index"] == Q_OPERATORS[0]
                  and row["time"] == Q_TIMES[0]]
    panel_status = {int(row["seed"]): bool(row["fit_valid"]) for row in panel_rows}
    if len(panel_status) != fit_planned:
        raise RuntimeError("panel denominator does not reconcile with unique deterministic seeds")
    fit_valid = sum(panel_status.values())
    q_valid = sum(row["fit_valid"] and row["covered"] is not None for row in replicate_rows)
    q_failed = sum(not row["fit_valid"] for row in replicate_rows)
    q_abstained = len(replicate_rows) - q_valid - q_failed
    denominators = {
        "synthetic_panels_planned": fit_planned,
        "synthetic_panels_valid": fit_valid,
        "synthetic_panels_failed": fit_planned - fit_valid,
        "synthetic_panels_abstained": 0,
        "q_cells_planned": len(replicate_rows),
        "q_cells_valid": q_valid,
        "q_cells_failed": q_failed,
        "q_cells_abstained": q_abstained,
        "physical_measurements_planned": 0,
        "physical_measurements_valid": 0,
        "physical_measurements_failed": 0,
        "physical_measurements_abstained": 0,
        "conformal_calibration_cells_planned": len(calibration_rows),
        "conformal_calibration_cells_finite_multiplier": sum(
            row["conformal_multiplier"] is not None for row in calibration_rows
        ),
        "conformal_calibration_cells_abstained": sum(
            row["conformal_multiplier"] is None for row in calibration_rows
        ),
        "conformal_validation_q_cells_planned": sum(
            row["validation_planned"] for row in calibration_rows
        ),
        "conformal_validation_q_cells_valid": sum(
            row["conformal_valid"] for row in calibration_rows
        ),
        "conformal_validation_q_cells_failed": sum(
            row["conformal_failed"] for row in calibration_rows
        ),
        "conformal_validation_q_cells_abstained": sum(
            row["conformal_abstained"] for row in calibration_rows
        ),
        "certification_planned": 1,
        "certification_valid": 0,
        "certification_failed": 0,
        "certification_abstained": 1,
    }
    summary = {
        "schema": "wilson-f-joint-spectral-gls-summary-v1",
        "classification": "synthetic CPU method diagnostic only",
        "physical_score": None,
        "F_lane_status": "GROUND_STATE_FORCE_UNCERTIFIED",
        "denominators": denominators,
        "coverage_rows": summary_rows,
        "calibration_rows": calibration_rows,
        "files": ["README.md", "protocol.json", "replicates.csv", "coverage_summary.csv",
                  "calibration_summary.csv", "coverage_and_interval_width.png",
                  "calibration_holdout_coverage.png",
                  "snapshot_f_lane_joint_spectral_gls_diagnostic.py",
                  "summary.json", "artifact_hashes.json"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    low_coverages = [row["upper_coverage"] for row in summary_rows if row["upper_coverage"] is not None]
    baseline_holdout = [row["baseline_holdout_coverage"] for row in calibration_rows
                        if row["baseline_holdout_coverage"] is not None]
    conformal_holdout = [row["conformal_holdout_coverage"] for row in calibration_rows
                         if row["conformal_holdout_coverage"] is not None]
    conformal_median_ratios = [row["conformal_median_upper_over_true_q"]
                               for row in calibration_rows
                               if row["conformal_median_upper_over_true_q"] is not None]
    conformal_calibrated_cells = sum(row["conformal_multiplier"] is not None
                                     for row in calibration_rows)
    delta_point_below_target = sum(row["upper_coverage"] < 0.975
                                   for row in summary_rows if row["upper_coverage"] is not None)
    delta_mc_upper_below_target = sum(row["coverage_mc95_high"] < 0.975
                                      for row in summary_rows if row["coverage_mc95_high"] is not None)
    conformal_mc_upper_below_target = sum(row["conformal_mc95_high"] < 0.975
                                          for row in calibration_rows
                                          if row["conformal_mc95_high"] is not None)
    lines = [
        "# Wilson F-lane synthetic joint spectral GLS pilot",
        "",
        "**SYNTHETIC METHOD DIAGNOSTIC ONLY.** No raw Wilson inputs, source snapshots, sealed splits, GPU or QUDA data were accessed. `physical_score=null`; F remains `GROUND_STATE_FORCE_UNCERTIFIED`.",
        "",
        "## Model and calculation",
        "",
        "The generator uses exactly three positive forward states and three bare operators, with one shared energy spectrum across every real and imaginary component of the 3×3 matrices at `t=1..7`. Each synthetic panel has independent Gaussian block vectors; the fit estimates the full 63×63 covariance from those same blocks, applies the explicitly illustrative 25% diagonal shrinkage and `1e-10` relative eigenvalue floor, whitens the joint residual, and fits 18 common energy/overlap parameters. GEVP at `t0=1` initializes the fit. The pointwise q upper is `qhat + 1.95996*delta_method_SE`, with parameter covariance from the inverse whitened information matrix.",
        "",
        "The q cells use bare basis operators 0 and 2 at `T=5,6`, matching the selected object semantics. The delta-method parameter covariance is the inverse whitened information matrix; it is not multiplied by reduced chi-square because the fitted GLS covariance already estimates the mean covariance. The resulting intervals are pointwise, not familywise; this experiment measures coverage only under the stated synthetic design and does not establish general or physical coverage. It does not resample fitted spectra or include model-selection uncertainty.",
        "",
        "![Pointwise q upper coverage and conservatism](coverage_and_interval_width.png)",
        "",
        f"The grid has {fit_planned} synthetic panels ({panels_per_cell} per noise × block-count cell) and {len(replicate_rows)} q cells; panel planned/valid/failed/abstained = `{fit_planned}/{fit_valid}/{fit_planned-fit_valid}/0`, q-cell planned/valid/failed/abstained = `{len(replicate_rows)}/{q_valid}/{q_failed}/{q_abstained}`. The pointwise delta upper covers `{f'{min(low_coverages):.3f}–{max(low_coverages):.3f}' if low_coverages else 'undefined (no valid q fits)'}` against a 0.975 target; exact per-cell Wilson Monte Carlo intervals and upper/truth ratios are in `coverage_summary.csv`.",
        f"All `{delta_point_below_target}/{len(low_coverages)}` pointwise delta coverage estimates are below 0.975; `{delta_mc_upper_below_target}/{len(low_coverages)}` cells also have 95% Wilson Monte Carlo upper limits below target. This is evidence against using this uncalibrated delta rule for the chosen synthetic regime.",
        "",
        "## Independent split-calibration check",
        "",
        f"Within each exact operator/time/noise/block stratum, replicate IDs `0:{split_at}` calibrate the studentized one-sided score `(q_true - qhat)/SE`; IDs `{split_at}:{panels_per_cell}` form a disjoint holdout. The finite-sample rank is `ceil((n_cal+1)*0.975)`. Failed or missing calibration q values enter as `+infinity`; if the rank exceeds the planned calibration count or the selected quantile is infinite, the interval method abstains rather than returning a finite bound. The diagnostic produced finite conformal multipliers for `{conformal_calibrated_cells}/{len(calibration_rows)}` strata.",
        "",
        "![Independent calibration versus holdout coverage](calibration_holdout_coverage.png)",
        "",
        (f"On valid holdout q cells, ordinary delta upper coverage is `{f'{min(baseline_holdout):.3f}–{max(baseline_holdout):.3f}' if baseline_holdout else 'undefined (no valid q fits)'}`. "
         + (f"Split-conformal upper coverage is `{min(conformal_holdout):.3f}–{max(conformal_holdout):.3f}` across finite-calibrator strata."
            if conformal_holdout else "No stratum produced a finite conformal multiplier; those holdout bounds abstained.")),
        f"For the split-conformal holdout, `{conformal_mc_upper_below_target}/{len(conformal_holdout)}` strata have a 95% Wilson Monte Carlo upper limit below 0.975; preserve that finite-holdout uncertainty alongside the exchangeability-based marginal guarantee.",
        (f"Across finite-calibrator strata, median `q_upper/q_true` ranges from `{min(conformal_median_ratios):.3f}` to `{max(conformal_median_ratios):.3f}`; cell-level conservatism is recorded in `calibration_summary.csv`."
         if conformal_median_ratios else "No finite conformal q upper was available for a conservatism summary."),
        "The split-conformal statement is marginal and only applies under exchangeability within these exact synthetic strata. Calibration panels use simulator truth unavailable for a real physical force; this is a candidate calibration pattern, not an authorized Wilson receipt or simultaneous q/h/x bound.",
        "",
        "## Limits and next work",
        "",
        "This best-case generator has exactly the fitted state count, forward-only infinite-time spectrum, known stationary independent synthetic blocks and Gaussian noise. It does not model omitted levels, finite-`N_t` thermal `h`, mixed `x`, shared-chain dependence, production endpoint semantics, raw exporter layout, force/reference propagation or the full familywise q/h/x budget. The 25% shrinkage/floor and delta method are candidate numerical choices, not an approved production covariance or interval rule. Even better same-generator split-conformal coverage would not certify the physical lane.",
        "",
        "The next software-only stress should add an omitted fourth state, near-degenerate gaps, near-zero bare ground overlaps, and covariance estimates with fewer effective blocks; retain failures and compare profile/bootstrap intervals before any reader/gate connection. Production integration still requires an independently reviewed exporter, authorized immutable raw snapshot and calibration split. Keep `NOT_READY/UNCERTIFIED`, `GROUND_STATE_FORCE_UNCERTIFIED`, and `physical_score=null`.",
        "",
        "Text flow: `exact 3-state source -> correlated synthetic block matrices -> same-block covariance/shrinkage -> whiten all matrix residuals -> joint ordered-spectrum fit -> pointwise bare-operator q upper -> per-cell Monte Carlo coverage + denominator figure -> certification abstains`.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    payload = [out / name for name in summary["files"] if name not in {"artifact_hashes.json"}]
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in payload if path.is_file()}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--panels-per-cell", type=int, default=PANELS_PER_CELL)
    args = parser.parse_args()
    result = run(args.output, panels_per_cell=args.panels_per_cell)
    print(json.dumps({"output": str(args.output), "denominators": result["denominators"]}, indent=2))


if __name__ == "__main__":
    main()
