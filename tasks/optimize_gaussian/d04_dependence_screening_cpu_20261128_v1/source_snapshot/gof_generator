"""Finite-sample pilot for the approximate D04 GLS goodness-of-fit p-value."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tasks.optimize_gaussian import d04_raw_reference as reader


TASK_DIR = Path(__file__).resolve().parent
ROOT = TASK_DIR.parents[1]
DEFAULT_OUT = TASK_DIR / "d04_gof_calibration_cpu_20261127_v1"
PANELS_PER_CELL = 300
CONFIGURATION_COUNTS = (64, 128)
BASE_SEED = 2026112712
NOISE_FRACTION = 0.03
MASS_NULL = (0.32, 0.91)
MASS_ALTERNATIVE = (0.32, 0.91, 1.45)
PP_AMPLITUDES_NULL = ((1.0, 0.22), (0.7, -0.12), (1.3, 0.30))
PP_AMPLITUDES_ALTERNATIVE = ((1.0, 0.22, 0.14), (0.7, -0.12, 0.11),
                             (1.3, 0.30, 0.18))
A4_AMPLITUDES_NULL = (0.18, -0.035)
A4_AMPLITUDES_ALTERNATIVE = (0.18, -0.035, 0.022)
PP_AMPLITUDES_ALTERNATIVE_STRONG = ((1.0, 0.22, 0.65), (0.7, -0.12, 0.45),
                                    (1.3, 0.30, 0.75))
A4_AMPLITUDES_ALTERNATIVE_STRONG = (0.18, -0.035, 0.085)
P_THRESHOLDS = (0.01, 0.05)
_SOURCE_PATHS = {
    "diagnostic": Path(__file__),
    "reader": TASK_DIR / "d04_raw_reference.py",
    "contract": TASK_DIR / "research_contract_v1.json",
    "test": ROOT / "tests/test_optimize_d04_raw_reference.py",
    "source_integrity": ROOT / "core/source_integrity.py",
    "npz_safety": ROOT / "core/npz_safety.py",
}


def _sha(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _feature_descriptors():
    return ([ (f"PP{channel}", time) for channel in range(3) for time in reader.TIMES ]
            + [("A4P", time) for time in reader.TIMES[:-1]])


def _truth_mean(masses, pp_amplitudes, a4_amplitudes):
    rows = []
    for channel in range(3):
        for time in reader.TIMES:
            value = sum(
                amplitude * (math.exp(-mass * time)
                             + math.exp(-mass * (reader.TEMPORAL_EXTENT - time)))
                for mass, amplitude in zip(masses, pp_amplitudes[channel])
            )
            rows.append(float(value))
    for time in reader.TIMES[:-1]:
        value = sum(
            amplitude * (math.exp(-mass * time)
                         - math.exp(-mass * (reader.TEMPORAL_EXTENT - time)))
            for mass, amplitude in zip(masses, a4_amplitudes)
        )
        rows.append(float(value))
    return np.asarray(rows, dtype=float)


def _noise_covariance(mean: np.ndarray) -> np.ndarray:
    features = _feature_descriptors()
    channel_names = ("PP0", "PP1", "PP2", "A4P")
    channel_index = {name: index for index, name in enumerate(channel_names)}
    channel_corr = np.full((4, 4), 0.2)
    np.fill_diagonal(channel_corr, 1.0)
    correlation = np.empty((len(features), len(features)), dtype=float)
    for i, (channel_i, time_i) in enumerate(features):
        for j, (channel_j, time_j) in enumerate(features):
            temporal = 0.65 ** abs(time_i - time_j)
            correlation[i, j] = temporal * channel_corr[
                channel_index[channel_i], channel_index[channel_j]]
    scales = np.maximum(np.abs(mean) * NOISE_FRACTION, 1e-8)
    covariance = np.outer(scales, scales) * correlation
    # Construction is a principal submatrix of a positive AR(1) x channel
    # compound-symmetry covariance, but check before passing to the sampler.
    if np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
        raise ArithmeticError("synthetic feature covariance is not positive definite")
    return covariance


def _multivariate_t_draws(rng: np.random.Generator, covariance: np.ndarray,
                          count: int, degrees_of_freedom: float) -> np.ndarray:
    if degrees_of_freedom <= 2.0:
        raise ValueError("Student-t innovations must have finite variance (df > 2)")
    scale = covariance * ((degrees_of_freedom - 2.0) / degrees_of_freedom)
    factor = np.linalg.cholesky(scale)
    normal = rng.standard_normal((count, covariance.shape[0])) @ factor.T
    chi_square = rng.chisquare(degrees_of_freedom, size=count)
    return normal / np.sqrt(chi_square / degrees_of_freedom)[:, None]


def _sample_panel(rng: np.random.Generator, mean: np.ndarray,
                  covariance: np.ndarray, count: int, *,
                  dependence_rho: float, innovation_df: float | None) -> np.ndarray:
    if not 0.0 <= dependence_rho < 1.0:
        raise ValueError("AR(1) dependence rho must lie in [0, 1)")
    if innovation_df is not None and innovation_df <= 2.0:
        raise ValueError("Student-t innovations must have finite variance (df > 2)")
    if dependence_rho == 0.0:
        if innovation_df is None:
            return rng.multivariate_normal(mean, covariance, size=count)
        return mean + _multivariate_t_draws(rng, covariance, count, innovation_df)

    # The innovation covariance makes covariance(mean-centered state)=covariance
    # at stationarity. Burn in from zero before retaining the finite chain.
    burn_in = 200
    innovation_covariance = covariance * (1.0 - dependence_rho**2)
    draws = count + burn_in
    if innovation_df is None:
        innovations = rng.multivariate_normal(
            np.zeros_like(mean), innovation_covariance, size=draws)
    else:
        innovations = _multivariate_t_draws(
            rng, innovation_covariance, draws, innovation_df)
    state = np.zeros_like(mean)
    samples = np.empty((count, len(mean)), dtype=float)
    for index, innovation in enumerate(innovations):
        state = dependence_rho * state + innovation
        if index >= burn_in:
            samples[index - burn_in] = mean + state
    return samples


def _mean_variance_inflation(rho: float, count: int) -> float:
    """Exact stationary AR(1) Var(mean) / (Sigma / n) multiplier."""
    lags = np.arange(1, count, dtype=float)
    return float(1.0 + 2.0 * np.sum((1.0 - lags / count) * rho**lags))


def _effective_shrinkage(covariance_mode: str,
                         shrinkage_override: float | None) -> float:
    if covariance_mode == "estimated_shrunk":
        return reader.SHRINKAGE
    if covariance_mode in {"oracle_full", "oracle_long_run"}:
        return 0.0
    if covariance_mode == "oracle_shrunk":
        return reader.SHRINKAGE if shrinkage_override is None else shrinkage_override
    raise ValueError("unsupported covariance_mode")


def _generator_name(alternative: str, dependence_rho: float,
                    innovation_df: float | None) -> str:
    if alternative != "null":
        return f"three_state_omission_{alternative}"
    if innovation_df is None:
        if dependence_rho == 0.0:
            return "exact_two_state_null"
        return f"two_state_ar1_gaussian_rho_{dependence_rho:.2f}"
    if dependence_rho == 0.0:
        return f"two_state_iid_student_t_df_{innovation_df:g}"
    return (f"two_state_ar1_student_t_df_{innovation_df:g}_"
            f"rho_{dependence_rho:.2f}")


def _folded_arrays(samples: np.ndarray):
    descriptors = _feature_descriptors()
    folded_pp = np.zeros((len(samples), 3, reader.TEMPORAL_EXTENT // 2 + 1), dtype=float)
    folded_a4p = np.zeros((len(samples), reader.TEMPORAL_EXTENT // 2 + 1), dtype=float)
    for column, (channel, time) in enumerate(descriptors):
        if channel.startswith("PP"):
            folded_pp[:, int(channel[2]), time] = samples[:, column]
        else:
            folded_a4p[:, time] = samples[:, column]
    return folded_pp, folded_a4p


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials**2)) / denom
    if successes == 0:
        return 0.0, min(1.0, center + half)
    if successes == trials:
        return max(0.0, center - half), 1.0
    return max(0.0, center - half), min(1.0, center + half)


def _simulate_cell(*, label: str, configurations: int, panels: int, seed: int,
                   alternative: str, covariance_mode: str = "estimated_shrunk",
                   shrinkage_override: float | None = None,
                   dependence_rho: float = 0.0,
                   innovation_df: float | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if alternative == "null":
        masses, pp_amplitudes, a4_amplitudes = (
            MASS_NULL, PP_AMPLITUDES_NULL, A4_AMPLITUDES_NULL)
    elif alternative == "mild":
        masses, pp_amplitudes, a4_amplitudes = (
            MASS_ALTERNATIVE, PP_AMPLITUDES_ALTERNATIVE, A4_AMPLITUDES_ALTERNATIVE)
    elif alternative == "strong":
        masses, pp_amplitudes, a4_amplitudes = (
            MASS_ALTERNATIVE, PP_AMPLITUDES_ALTERNATIVE_STRONG,
            A4_AMPLITUDES_ALTERNATIVE_STRONG)
    else:
        raise ValueError("alternative must be null, mild or strong")
    mean = _truth_mean(masses, pp_amplitudes, a4_amplitudes)
    covariance = _noise_covariance(mean)
    rng = np.random.default_rng(seed)
    rows = []
    failures = boundaries = near_degenerate = 0
    for panel in range(panels):
        sample = _sample_panel(
            rng, mean, covariance, configurations,
            dependence_rho=dependence_rho, innovation_df=innovation_df)
        folded_pp, folded_a4p = _folded_arrays(sample)
        fit_mean, fit_covariance, columns = reader._fit_inputs(folded_pp, folded_a4p)
        if covariance_mode == "oracle_full":
            fit_covariance = covariance / configurations
        elif covariance_mode == "oracle_shrunk":
            alpha = reader.SHRINKAGE if shrinkage_override is None else shrinkage_override
            if not 0.0 <= alpha <= 1.0:
                raise ValueError("oracle shrinkage alpha must lie in [0, 1]")
            diagonal = np.diag(np.diag(covariance))
            fit_covariance = ((1.0 - alpha) * covariance
                              + alpha * diagonal) / configurations
        elif covariance_mode == "oracle_long_run":
            inflation = _mean_variance_inflation(dependence_rho, configurations)
            fit_covariance = covariance * inflation / configurations
        elif covariance_mode != "estimated_shrunk":
            raise ValueError("unsupported covariance_mode")
        fit = reader._fit_variant(fit_mean, fit_covariance, columns,
                                  tmin=3, nstates=2)
        valid = fit["status"] == "valid"
        failures += int(not valid)
        if valid:
            boundaries += int(fit["boundary_hit_count"] > 0)
            near_degenerate += int(fit["near_degenerate_state_gap_below_0p05"])
        rows.append({
            "cell": label, "panel": panel, "seed": seed,
            "sample_sha256": _sha(np.asarray(sample, dtype=np.float64).tobytes()),
            "configurations": configurations,
            "generator": _generator_name(alternative, dependence_rho, innovation_df),
            "is_null": alternative == "null",
            "covariance_mode": covariance_mode,
            "covariance_shrinkage": _effective_shrinkage(covariance_mode, shrinkage_override),
            "dependence_rho": dependence_rho,
            "innovation_df": innovation_df,
            "mean_variance_inflation_factor": _mean_variance_inflation(
                dependence_rho, configurations),
            "fit_status": fit["status"],
            "reference_mass": fit.get("reference_mass"),
            "chi2": fit.get("chi2"), "dof": fit.get("dof"),
            "gof_p_approx": fit.get("gof_p_approx"),
            "boundary_hit_count": fit.get("boundary_hit_count"),
            "near_degenerate": fit.get("near_degenerate_state_gap_below_0p05"),
        })
    valid_rows = [row for row in rows if row["gof_p_approx"] is not None]
    summary = {
        "cell": label,
        "generator": _generator_name(alternative, dependence_rho, innovation_df),
        "is_null": alternative == "null",
        "covariance_mode": covariance_mode,
        "covariance_shrinkage": _effective_shrinkage(covariance_mode, shrinkage_override),
        "dependence_rho": dependence_rho,
        "innovation_df": innovation_df,
        "mean_variance_inflation_factor": _mean_variance_inflation(
            dependence_rho, configurations),
        "configurations_per_panel": configurations, "panels": panels,
        "denominator": {"planned": panels, "valid": len(valid_rows),
                        "failed": failures, "abstained": 0},
        "boundary_hit_panels": boundaries, "near_degenerate_fit_panels": near_degenerate,
        "reference_mass_true": float(masses[0]),
        "fit_state_count": 2, "true_state_count": len(masses),
    }
    if valid_rows:
        pvalues = np.asarray([row["gof_p_approx"] for row in valid_rows], dtype=float)
        summary["pvalue_summary"] = {
            "median": float(np.median(pvalues)),
            "mean": float(np.mean(pvalues)),
            "ks_distance_to_uniform": float(np.max(np.abs(
                np.sort(pvalues) - (np.arange(len(pvalues)) + 0.5) / len(pvalues)))),
            "rejection": {},
        }
        for threshold in P_THRESHOLDS:
            count = int(np.count_nonzero(pvalues < threshold))
            interval = _wilson(count, len(pvalues))
            summary["pvalue_summary"]["rejection"][str(threshold)] = {
                "count": count, "rate": count / len(pvalues),
                "wilson_95": list(interval),
                "nominal_rate": threshold,
                "mcse": float(math.sqrt(threshold * (1.0 - threshold) / len(pvalues))),
            }
    else:
        summary["pvalue_summary"] = None
    return rows, summary


def _paired_contrast(rows: list[dict[str, Any]], left_cell: str,
                     right_cell: str) -> dict[str, Any]:
    """Compare fit metrics for the exact same simulated panel samples."""
    left = {row["panel"]: row for row in rows if row["cell"] == left_cell}
    right = {row["panel"]: row for row in rows if row["cell"] == right_cell}
    if set(left) != set(right):
        raise ValueError(f"paired cells have different panel IDs: {left_cell}, {right_cell}")
    paired = []
    for panel in sorted(left):
        lhs, rhs = left[panel], right[panel]
        if lhs["sample_sha256"] != rhs["sample_sha256"]:
            raise ValueError(f"paired cell samples differ at panel {panel}")
        if lhs["gof_p_approx"] is not None and rhs["gof_p_approx"] is not None:
            paired.append((lhs, rhs))
    if not paired:
        raise ValueError(f"paired cells have no valid fit pairs: {left_cell}, {right_cell}")
    delta_p = np.asarray([rhs["gof_p_approx"] - lhs["gof_p_approx"]
                          for lhs, rhs in paired])
    delta_chi2 = np.asarray([rhs["chi2"] - lhs["chi2"] for lhs, rhs in paired])
    return {
        "left_cell": left_cell,
        "right_cell": right_cell,
        "interpretation": "right minus left; identical simulated panel samples verified by SHA-256",
        "planned_pairs": len(left), "valid_pairs": len(paired),
        "failed_or_missing_pairs": len(left) - len(paired),
        "mean_delta_gof_p_approx": float(np.mean(delta_p)),
        "median_delta_gof_p_approx": float(np.median(delta_p)),
        "fraction_right_pvalue_lower": float(np.mean(delta_p < 0.0)),
        "mean_delta_chi2": float(np.mean(delta_chi2)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows: list[dict[str, Any]], summaries: list[dict[str, Any]]):
    groups = []
    for cell in summaries:
        if cell["cell"].startswith("chain_"):
            continue
        values = np.asarray([row["gof_p_approx"] for row in rows
                             if row["cell"] == cell["cell"]
                             and row["gof_p_approx"] is not None])
        if values.size:
            groups.append((cell, np.sort(values)))
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.4), layout="constrained")
    ax = axes[0]
    epsilon = math.sqrt(math.log(2.0 / 0.05) / (2.0 * PANELS_PER_CELL))
    x = np.linspace(0.0, 1.0, 101)
    ax.fill_between(x, np.maximum(0.0, x - epsilon), np.minimum(1.0, x + epsilon),
                    color="#cccccc", alpha=0.45, label="95% DKW band (null reference)")
    ax.plot(x, x, color="black", linestyle="--", label="Uniform reference")
    colors = ("#2b78e4", "#4f9d69", "#333333", "#e6ab02", "#cc6677",
              "#9966aa", "#168aad")
    null_groups = [(cell, values) for cell, values in groups
                   if cell["generator"] == "exact_two_state_null"]
    for (cell, values), color in zip(null_groups, colors):
        ecdf = np.arange(1, len(values) + 1) / len(values)
        label = cell["cell"]
        if label.startswith("null_n128_oracle_shrunk_a"):
            label = "oracle shrinkage " + label.rsplit("a", 1)[1]
        elif label == "null_n128_oracle_full":
            label = "oracle full covariance"
        elif label == "null_n128":
            label = "estimated + 25% shrinkage"
        ax.step(values, ecdf, where="post", color=color,
                label=label)
    for threshold in P_THRESHOLDS:
        ax.axvline(threshold, color="#666666", alpha=0.35, linewidth=0.8)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="approximate GLS chi-square p-value",
           ylabel="empirical CDF", title="Null p-value calibration across covariance metrics")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)

    ax = axes[1]
    left_cell = "omitted_state_strong_n128"
    right_cell = "omitted_state_strong_oracle_full_n128"
    left = {row["panel"]: row for row in rows if row["cell"] == left_cell}
    right = {row["panel"]: row for row in rows if row["cell"] == right_cell}
    if left and right:
        panels = sorted(set(left) & set(right))
        paired = [(left[p]["gof_p_approx"], right[p]["gof_p_approx"])
                  for p in panels if left[p]["gof_p_approx"] is not None
                  and right[p]["gof_p_approx"] is not None]
        x_values, y_values = zip(*paired) if paired else ([], [])
        ax.scatter(x_values, y_values, s=15, alpha=0.38, color="#168aad",
                   edgecolors="none")
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1,
                label="same p-value")
        ax.set(xlim=(0, 1), ylim=(0, 1),
               xlabel="estimated + 25% shrinkage p-value",
               ylabel="known full-covariance p-value",
               title="Strong omitted-state alternative\npaired on identical samples")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    ax = axes[2]
    labels, positions = [], []
    short_labels = {
        "null_n64": "null N=64",
        "null_n128": "estimated α=.25",
        "null_n128_oracle_full": "oracle α=0",
        "null_n128_oracle_shrunk_a25": "oracle α=.25",
        "null_n128_oracle_shrunk_a05": "oracle α=.05",
        "null_n128_oracle_shrunk_a10": "oracle α=.10",
        "null_n128_oracle_shrunk_a50": "oracle α=.50",
        "omitted_state_mild_n128": "mild omission",
        "omitted_state_strong_n128": "strong omission",
        "omitted_state_strong_oracle_full_n128": "strong oracle α=0",
    }
    for index, (cell, _) in enumerate(groups):
        for threshold in P_THRESHOLDS:
            item = cell["pvalue_summary"]["rejection"][str(threshold)]
            low, high = item["wilson_95"]
            ax.errorbar(index * 2 + (0 if threshold == 0.01 else 0.45), item["rate"],
                        yerr=[[item["rate"] - low], [high - item["rate"]]],
                        fmt="o", capsize=3,
                        color="#2b78e4" if threshold == 0.01 else "#e69138")
        labels.append(short_labels.get(cell["cell"], cell["cell"]))
        positions.append(index * 2 + 0.225)
    ax.axhline(0.01, color="#2b78e4", linestyle="--", alpha=0.7,
               label="nominal 1% null size")
    ax.axhline(0.05, color="#e69138", linestyle="--", alpha=0.7,
               label="nominal 5% null size")
    ax.set_xticks(positions, labels, rotation=35, ha="right", fontsize=7)
    upper_bounds = [cell["pvalue_summary"]["rejection"][str(threshold)]["wilson_95"][1]
                    for cell, _ in groups for threshold in P_THRESHOLDS]
    ax.set_ylim(0, min(1.0, max(0.10, 1.12 * max(upper_bounds, default=0.05))))
    ax.set_ylabel("observed rejection fraction")
    ax.set_title("Wilson intervals; null size vs alternative power")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_dependence(path: Path, rows: list[dict[str, Any]],
                     summaries: list[dict[str, Any]]):
    selected_summaries = [cell for cell in summaries
                          if cell["cell"].startswith("chain_")]
    groups = []
    for cell in selected_summaries:
        values = np.asarray([row["gof_p_approx"] for row in rows
                             if row["cell"] == cell["cell"]
                             and row["gof_p_approx"] is not None])
        if values.size:
            groups.append((cell, np.sort(values)))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), layout="constrained")
    ax = axes[0]
    x = np.linspace(0.0, 1.0, 101)
    epsilon = math.sqrt(math.log(2.0 / 0.05) / (2.0 * PANELS_PER_CELL))
    ax.fill_between(x, np.maximum(0.0, x - epsilon), np.minimum(1.0, x + epsilon),
                    color="#cccccc", alpha=0.4, label="95% DKW reference band")
    ax.plot(x, x, color="black", linestyle="--", label="uniform reference")
    palette = ("#2b78e4", "#cc6677", "#333333", "#4f9d69", "#e6ab02", "#168aad")
    label_map = {
        "chain_iid_gaussian_n128": "iid Gaussian, estimated covariance",
        "chain_ar1_gaussian_est_n128": "AR(1) Gaussian, estimated covariance",
        "chain_ar1_gaussian_oracle_lrv_n128": "AR(1) Gaussian, oracle long-run covariance",
        "chain_iid_student_t5_n128": "iid Student-t(5), estimated covariance",
        "chain_ar1_student_t5_est_n128": "AR(1) Student-t(5), estimated covariance",
        "chain_ar1_student_t5_oracle_lrv_n128": "AR(1) Student-t(5), oracle long-run covariance",
    }
    for (cell, values), color in zip(groups, palette):
        ecdf = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, ecdf, where="post", color=color,
                label=label_map.get(cell["cell"], cell["cell"]))
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="approximate GLS chi-square p-value",
           ylabel="empirical CDF", title="Null calibration under dependence / heavy tails")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)

    ax = axes[1]
    positions, labels = [], []
    short_labels = {
        "chain_iid_gaussian_n128": "iid Gaussian",
        "chain_ar1_gaussian_est_n128": "AR1 Gaussian estimated",
        "chain_ar1_gaussian_oracle_lrv_n128": "AR1 Gaussian oracle",
        "chain_iid_student_t5_n128": "iid t(5)",
        "chain_ar1_student_t5_est_n128": "AR1 t(5) estimated",
        "chain_ar1_student_t5_oracle_lrv_n128": "AR1 t(5) oracle",
    }
    for index, (cell, _) in enumerate(groups):
        for threshold in P_THRESHOLDS:
            item = cell["pvalue_summary"]["rejection"][str(threshold)]
            low, high = item["wilson_95"]
            ax.errorbar(index * 2 + (0 if threshold == 0.01 else 0.45), item["rate"],
                        yerr=[[item["rate"] - low], [high - item["rate"]]],
                        fmt="o", capsize=3,
                        color="#2b78e4" if threshold == 0.01 else "#e69138")
        labels.append(short_labels.get(cell["cell"], cell["cell"]))
        positions.append(index * 2 + 0.225)
    ax.axhline(0.01, color="#2b78e4", linestyle="--", alpha=0.7,
               label="nominal 1% size")
    ax.axhline(0.05, color="#e69138", linestyle="--", alpha=0.7,
               label="nominal 5% size")
    ax.set_xticks(positions, labels, rotation=30, ha="right", fontsize=7)
    upper_bounds = [cell["pvalue_summary"]["rejection"][str(threshold)]["wilson_95"][1]
                    for cell, _ in groups for threshold in P_THRESHOLDS]
    ax.set_ylim(0, min(1.0, max(0.10, 1.12 * max(upper_bounds, default=0.05))))
    ax.set_ylabel("observed rejection fraction")
    ax.set_title("Nominal size and Wilson 95% intervals")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _artifact_hashes(output: Path):
    files = sorted(path for path in output.rglob("*")
                    if path.is_file() and path.name != "artifact_hashes.json")
    values = {str(path.relative_to(output)): _sha(path.read_bytes()) for path in files}
    (output / "artifact_hashes.json").write_bytes(_json_bytes({
        "schema": "optimize-d04-gof-artifact-hashes-v1", "artifacts": values,
    }))
    return values


def build(output: str | Path = DEFAULT_OUT):
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {out}")
    source_bytes = {name: path.read_bytes() for name, path in _SOURCE_PATHS.items()}
    source_hashes = {name: _sha(value) for name, value in source_bytes.items()}
    out.mkdir(parents=True)
    cell_specs = (
        ("null_n64", 64, "null", "estimated_shrunk", BASE_SEED, None, 0.0, None),
        ("null_n128", 128, "null", "estimated_shrunk", BASE_SEED + 1000, None, 0.0, None),
        ("null_n128_oracle_full", 128, "null", "oracle_full", BASE_SEED + 1000, 0.0, 0.0, None),
        ("null_n128_oracle_shrunk_a25", 128, "null", "oracle_shrunk", BASE_SEED + 1000, 0.25, 0.0, None),
        ("null_n128_oracle_shrunk_a05", 128, "null", "oracle_shrunk", BASE_SEED + 1000, 0.05, 0.0, None),
        ("null_n128_oracle_shrunk_a10", 128, "null", "oracle_shrunk", BASE_SEED + 1000, 0.10, 0.0, None),
        ("null_n128_oracle_shrunk_a50", 128, "null", "oracle_shrunk", BASE_SEED + 1000, 0.50, 0.0, None),
        ("omitted_state_mild_n128", 128, "mild", "estimated_shrunk", BASE_SEED + 4000, None, 0.0, None),
        ("omitted_state_strong_n128", 128, "strong", "estimated_shrunk", BASE_SEED + 5000, None, 0.0, None),
        ("omitted_state_strong_oracle_full_n128", 128, "strong", "oracle_full", BASE_SEED + 5000, 0.0, 0.0, None),
        ("chain_iid_gaussian_n128", 128, "null", "estimated_shrunk", BASE_SEED + 8000, None, 0.0, None),
        ("chain_ar1_gaussian_est_n128", 128, "null", "estimated_shrunk", BASE_SEED + 9000, None, 0.30, None),
        ("chain_ar1_gaussian_oracle_lrv_n128", 128, "null", "oracle_long_run", BASE_SEED + 9000, 0.0, 0.30, None),
        ("chain_iid_student_t5_n128", 128, "null", "estimated_shrunk", BASE_SEED + 10000, None, 0.0, 5.0),
        ("chain_ar1_student_t5_est_n128", 128, "null", "estimated_shrunk", BASE_SEED + 11000, None, 0.30, 5.0),
        ("chain_ar1_student_t5_oracle_lrv_n128", 128, "null", "oracle_long_run", BASE_SEED + 11000, 0.0, 0.30, 5.0),
    )
    all_rows, summaries = [], []
    for label, count, alternative, covariance_mode, seed, shrinkage, rho, df in cell_specs:
        rows, summary = _simulate_cell(
            label=label, configurations=count, panels=PANELS_PER_CELL,
            seed=seed,
            alternative=alternative, covariance_mode=covariance_mode,
            shrinkage_override=shrinkage,
            dependence_rho=rho, innovation_df=df,
        )
        all_rows.extend(rows)
        summaries.append(summary)
    paired_contrasts = [
        _paired_contrast(all_rows, "null_n128", "null_n128_oracle_full"),
        _paired_contrast(all_rows, "null_n128", "null_n128_oracle_shrunk_a25"),
        _paired_contrast(all_rows, "null_n128_oracle_shrunk_a05",
                         "null_n128_oracle_shrunk_a10"),
        _paired_contrast(all_rows, "null_n128_oracle_shrunk_a25",
                         "null_n128_oracle_shrunk_a50"),
        _paired_contrast(all_rows, "omitted_state_strong_n128",
                         "omitted_state_strong_oracle_full_n128"),
        _paired_contrast(all_rows, "chain_ar1_gaussian_est_n128",
                         "chain_ar1_gaussian_oracle_lrv_n128"),
        _paired_contrast(all_rows, "chain_ar1_student_t5_est_n128",
                         "chain_ar1_student_t5_oracle_lrv_n128"),
    ]
    unique_sample_keys = {(count, alternative, seed, rho, df)
                          for _, count, alternative, _, seed, _, rho, df in cell_specs}
    unique_sample_panels = len(unique_sample_keys) * PANELS_PER_CELL
    summary_payload = {
        "schema": "optimize-d04-gof-calibration-summary-v1",
        "classification": "SYNTHETIC_GOF_SIZE_AND_POWER_PILOT_NOT_PHYSICAL_REFERENCE",
        "fit": {"tmin": 3, "true_mass_ground": MASS_NULL[0], "fit_state_count": 2,
                "fit_dof": 13, "pp_a4p_joint": True,
                "covariance_shrinkage": reader.SHRINKAGE,
                "covariance_eigenvalue_floor": reader.EIGENVALUE_FLOOR},
        "generator": {
            "kind": "synthetic_two_state_null_and_omitted_state_panels",
            "feature_count": 27, "noise_fraction": NOISE_FRACTION,
            "temporal_correlation": 0.65,
            "cross_channel_correlation": 0.2,
            "unique_sample_panels": unique_sample_panels,
            "fit_rows": len(all_rows),
            "panels_per_cell": PANELS_PER_CELL,
            "chain_stress": {
                "ar1_rho": 0.30, "student_t_degrees_of_freedom": 5,
                "burn_in_configurations": 200,
                "innovation_covariance": "(1-rho^2) * stationary feature covariance; Student-t scaled to the same finite covariance",
                "oracle_long_run_covariance": "finite-n multiplier 1 + 2*sum_{k=1}^{n-1} (1-k/n)*rho^k applied to stationary covariance/n",
            },
            "cells": summaries,
        },
        "paired_contrasts": paired_contrasts,
        "physical_e_chain": {"planned": 0, "valid": 0, "failed": 0, "abstained": 0},
        "reference_gate": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
        "physical_score": None,
        "open_gates": [
            "default p-values use an approximate chi-square reference with data-estimated, 25%-shrunk covariance; paired oracle cells use known full covariance or known-covariance diagonal shrinkage alpha sensitivity",
            "six chain-stress cells add one rho=0.30 AR(1) and one iid/AR(1) Student-t(5) design; this does not certify any production Markov chain",
            "oracle long-run covariance is synthetic truth and is not an available production estimator",
            "only sixteen synthetic cells are evaluated; multiple covariance cells reuse identical raw panels",
            "the alternative is an omitted third state and the rejection fractions are power diagnostics only",
            "no HMC chain dependence, estimator selection, or candidate-blind production E-chain provenance is modeled",
            "these results cannot replace the independent D04 reference receipt or statistical review",
        ],
    }
    (out / "summary.json").write_bytes(_json_bytes(summary_payload))
    _write_csv(out / "panel_results.csv", all_rows)
    _plot(out / "gof_pvalue_calibration.png", all_rows, summaries)
    _plot_dependence(out / "gof_dependence_stress.png", all_rows, summaries)
    protocol = {
        "schema": "optimize-d04-gof-calibration-protocol-v1",
        "classification": summary_payload["classification"],
        "source_hashes": source_hashes,
        "fit_rule": "reader._fit_variant(mean,cov,columns,tmin=3,nstates=2)",
        "fit_settings": {"starts": 3, "shrinkage": reader.SHRINKAGE,
                         "eigenvalue_floor": reader.EIGENVALUE_FLOOR,
                         "pvalue": "chi2.sf(fitted GLS residual, dof=13)"},
        "panels_per_cell": PANELS_PER_CELL,
        "seeds": {cell: seed for cell, _, _, _, seed, _, _, _ in cell_specs},
        "covariance_modes": {cell: mode for cell, _, _, mode, _, _, _, _ in cell_specs},
        "covariance_shrinkage": {
            cell: _effective_shrinkage(mode, alpha)
            for cell, _, _, mode, _, alpha, _, _ in cell_specs
        },
        "dependence": {
            cell: {"rho": rho, "innovation_df": df,
                   "mean_variance_inflation_factor": _mean_variance_inflation(rho, count)}
            for cell, count, _, _, _, _, rho, df in cell_specs
        },
        "paired_contrasts": paired_contrasts,
        "thresholds": list(P_THRESHOLDS),
        "design_note": "The strong omitted-state cell was added after an initial mild-alternative pilot showed weak separation. The rho=0.30 AR(1) and finite-variance Student-t(5) panels are predeclared synthetic chain-stress designs; covariance-oracle cells are paired by sample hash and do not tune a production threshold.",
        "generation": "all samples generated from the source bytes of this diagnostic; no input file read",
        "physical_score": None,
    }
    (out / "protocol.json").write_bytes(_json_bytes(protocol))
    source_dir = out / "source_snapshot"
    source_dir.mkdir()
    for name, payload in source_bytes.items():
        (source_dir / name.replace("/", "_" )).write_bytes(payload)
    (out / "source_hashes.json").write_bytes(_json_bytes({
        "schema": "optimize-d04-gof-source-hashes-v1", "sources": source_hashes,
    }))
    readme = [
        "# D04 approximate chi-square GOF calibration pilot",
        "",
        "**Status:** synthetic CPU diagnostic. It measures null rejection size and an omitted-state alternative for the current `gof_p_approx`; it is not a candidate-blind E-chain reference or a physical GO decision.",
        "",
        "## Frozen procedure",
        "",
        f"- Sixteen cells, `{PANELS_PER_CELL}` planned fits per cell: the first ten cover Gaussian-null covariance calibration and omitted-state alternatives; six additional cells stress iid Gaussian, AR(1) Gaussian, iid Student-t(5), and AR(1) Student-t(5), all at `Ncfg=128`.",
        f"- The covariance-alpha sweep reuses the same `Ncfg=128` null samples. Across all cells there are `{unique_sample_panels}` unique generated panels and `{len(all_rows)}` planned fits; paired covariance fits are not independent samples.",
        "- The mild omitted-state truth uses third-state PP overlaps 0.11–0.18 and A4P overlap 0.022; the strong stress uses PP overlaps 0.45–0.75 and A4P overlap 0.085. Both are distinct from the two-state null.",
        "- Covariance-oracle cells reuse exact sample panels of their comparison cells; sample SHA-256 values are checked before paired summaries are calculated. The full oracle removes covariance estimation and shrinkage; the alpha sweep isolates the metric under known covariance. For dependent cells, the oracle uses exact finite-N stationary AR(1) `Var(mean)` inflation; it is a simulation truth, not a deployable estimator.",
        f"- Each configuration is a 27-component vector around folded PP-even/A4P-odd correlator means; feature covariance is fixed from 0.65 within-vector temporal correlation and 0.2 cross-channel correlation, with relative noise `{NOISE_FRACTION}`. Chain-stress sequences use stationary-variance AR(1) `rho=0.30`, 200 burn-in configurations, and either Gaussian or Student-t(5) innovations scaled to finite covariance.",
        f"- Each panel uses the existing D04 fit path (`tmin=3`, two states, three starts); the default covariance is estimated from the same panel with {reader.SHRINKAGE:.0%} diagonal shrinkage and relative eigenvalue floor `{reader.EIGENVALUE_FLOOR:g}`. Independent-null oracle cells use known covariance or diagonal-shrunk versions at alpha 0%, 5%, 10%, 25% and 50%. AR(1) oracle cells use the exact finite-N long-run covariance of the synthetic process. All p-values use the approximate chi-square survival probability with 13 nominal degrees of freedom.",
        "- All panel fits remain in the planned denominator. The null cells estimate the false rejection frequency at p<0.01 and p<0.05; the omitted-state cell is a misspecification power diagnostic.",
        "- The strong omitted-state cell was added after the initial mild-alternative pilot showed weak separation. It is an exploratory power stress, not a preregistered confirmatory test; null thresholds and seeds were not changed.",
        "",
        "![D04 p-value calibration](gof_pvalue_calibration.png)",
        "",
        "![D04 chain dependence stress](gof_dependence_stress.png)",
        "",
        "In the first figure, the left panel shows the seven iid Gaussian null covariance cells' empirical p-value CDFs against the uniform reference; the gray band is a 95% Dvoretzky–Kiefer–Wolfowitz band for a 300-panel sample. The middle panel compares strong-alternative p-values for identical samples under estimated/shrunk and known full covariance. The right panel gives the first ten cells' observed rejection fractions and Wilson intervals at 1% and 5%. For null cells, a well-calibrated p-value should track the diagonal and rejection rates should be compatible with nominal sizes; omitted-state rejection is power, not type-I error.",
        "In the dependence-stress figure, iid Gaussian and iid Student-t(5) estimated-covariance p-values are conservative in this realization, while rho=0.30 AR(1) panels with iid-style covariance-of-mean are anti-conservative: `p<0.05` rates are `0.217` for Gaussian and `0.163` for Student-t(5). The paired exact finite-N long-run covariance oracle lowers them to `0.063` and `0.067`. This demonstrates sensitivity to dependence under the chosen synthetic process; it does not identify a production HAC/block estimator or certify an HMC chain.",
        "",
        "## Interpretation limits and next step",
        "",
        "The chain-stress panel adds only one synthetic `rho=0.30` AR(1) process and finite-variance Student-t(5) innovations; it does not model actual HMC dependence, channel/source convention errors, fit-window or optimizer-start selection, or a real E-chain. The 300-panel size gives MCSE about 0.0056 at a 1% event and 0.0126 at 5%, so tail estimates remain coarse. The DKW band is descriptive per 300-panel cell, not multiplicity control. Results must not be used to tune and validate a replacement threshold on the same panels. Next, preregister a larger independent grid over dependence strength and tail weight, then compare the approximate chi-square reference with a parametric bootstrap that refits covariance and spectra on each simulated panel. Keep `D04 NOT_READY/UNCERTIFIED` and `physical_score=null` until an authorized independent E-chain package and review exist.",
        "",
        "## Run denominators",
        "",
    ]
    for cell in summaries:
        readme.append(f"- `{cell['cell']}`: P/V/F/A = `{cell['denominator']['planned']}/{cell['denominator']['valid']}/{cell['denominator']['failed']}/{cell['denominator']['abstained']}`; rejection summaries `{cell.get('pvalue_summary')}`.")
    readme.extend(["", "## Paired fit-metric contrasts", ""])
    for contrast in paired_contrasts:
        readme.append(f"- `{contrast['left_cell']}` → `{contrast['right_cell']}`: `{contrast}`.")
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    artifacts = _artifact_hashes(out)
    return {"output": str(out), "cells": len(summaries), "panels": len(all_rows),
            "artifact_count": len(artifacts), "valid": sum(
                cell["denominator"]["valid"] for cell in summaries)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--panels", type=int, default=PANELS_PER_CELL)
    args = parser.parse_args()
    if args.panels != PANELS_PER_CELL:
        raise ValueError("this frozen pilot's panel count is fixed in the source")
    print(json.dumps(build(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
