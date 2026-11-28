"""D04 dependence-aware parametric-bootstrap stress diagnostic (synthetic only)."""
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
from scipy.optimize import minimize_scalar

from tasks.optimize_gaussian import d04_gof_calibration_diagnostic as generator
from tasks.optimize_gaussian import d04_parametric_bootstrap_diagnostic as bootstrap
from tasks.optimize_gaussian import d04_raw_reference as reader


TASK_DIR = Path(__file__).resolve().parent
ROOT = TASK_DIR.parents[1]
DEFAULT_OUT = TASK_DIR / "d04_ar1_bootstrap_cpu_20261127_v1"
OUTER_PANELS = 160
BOOTSTRAP_REPLICATES = 199
CONFIGURATIONS = 128
TRUE_RHO = 0.30
MIN_VALID_FRACTION = 0.90
BASE_SEED = 2026112799
P_THRESHOLDS = (0.01, 0.05)
PROFILE_RHO_GRID_POINTS = 81
_SOURCE_PATHS = {
    "diagnostic": Path(__file__),
    "bootstrap_helpers": TASK_DIR / "d04_parametric_bootstrap_diagnostic.py",
    "gof_generator": TASK_DIR / "d04_gof_calibration_diagnostic.py",
    "reader": TASK_DIR / "d04_raw_reference.py",
    "contract": TASK_DIR / "research_contract_v1.json",
    "test": ROOT / "tests/test_optimize_d04_raw_reference.py",
}


def _sha(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write_csv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path.name}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sample_stationary_ar1(rng: np.random.Generator, mean: np.ndarray,
                           covariance: np.ndarray, count: int,
                           rho: float) -> np.ndarray:
    """Draw a stationary multivariate AR(1), allowing signed rho."""
    if not -0.95 <= rho <= 0.95:
        raise ValueError("AR(1) rho must be in [-0.95, 0.95]")
    factor = math.sqrt(1.0 - rho * rho)
    state = rng.multivariate_normal(np.zeros_like(mean), covariance)
    sample = np.empty((count, len(mean)), dtype=float)
    for index in range(count):
        if index:
            innovation = rng.multivariate_normal(np.zeros_like(mean), covariance)
            state = rho * state + factor * innovation
        sample[index] = mean + state
    return sample


def _estimate_common_rho(sample: np.ndarray) -> float:
    """Pooled lag-1 OLS estimate after removing each feature's sample mean."""
    centered = sample - np.mean(sample, axis=0, keepdims=True)
    denominator = float(np.sum(centered[:-1] ** 2))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("cannot estimate AR(1) rho from a constant or invalid panel")
    numerator = float(np.sum(centered[1:] * centered[:-1]))
    estimate = numerator / denominator
    return float(np.clip(estimate, -0.95, 0.95))


def _profile_matrix_normal_deviance(sample: np.ndarray, rho: float) -> float:
    """Return the profile deviance for a stationary matrix-normal AR(1).

    Rows are ordered configurations and columns are features. The mean vector
    and the unrestricted feature covariance are profiled out at each rho.
    Constants independent of rho are omitted, so this is a comparison
    objective rather than a calibrated likelihood-ratio statistic.
    """
    values = np.asarray(sample, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("rho profile input must be a non-empty 2D panel")
    if not np.all(np.isfinite(values)):
        raise ValueError("rho profile input must be finite")
    if isinstance(rho, (bool, np.bool_)) or not np.isfinite(rho):
        raise ValueError("profile rho must be a finite scalar")
    rho = float(rho)
    if not -0.95 <= rho <= 0.95:
        raise ValueError("profile rho must lie in [-0.95, 0.95]")

    n_configurations, n_features = values.shape
    variance_factor = 1.0 - rho * rho
    if variance_factor <= 0.0:
        raise ValueError("stationary AR(1) row covariance is not positive definite")
    diagonal = np.full(n_configurations, 1.0 + rho * rho)
    diagonal[[0, -1]] = 1.0
    diagonal /= variance_factor
    off_diagonal = -rho / variance_factor

    def apply_precision(matrix: np.ndarray) -> np.ndarray:
        weighted = diagonal[:, None] * matrix
        weighted[:-1] += off_diagonal * matrix[1:]
        weighted[1:] += off_diagonal * matrix[:-1]
        return weighted

    ones = np.ones((n_configurations, 1), dtype=float)
    precision_one = apply_precision(ones)
    precision_values = apply_precision(values)
    mean_denominator = float(np.sum(precision_one))
    if not np.isfinite(mean_denominator) or mean_denominator <= 0.0:
        raise ValueError("profiled GLS mean is unidentifiable")
    mean = np.sum(precision_values, axis=0) / mean_denominator
    residual = values - mean[None, :]
    feature_scatter = residual.T @ apply_precision(residual)
    feature_scatter = 0.5 * (feature_scatter + feature_scatter.T)
    sign, logdet_scatter = np.linalg.slogdet(feature_scatter / n_configurations)
    if sign <= 0.0 or not np.isfinite(logdet_scatter):
        raise ValueError("profiled feature covariance is not positive definite")

    # det(R_N(rho)) = (1-rho**2)**(N-1) for stationary unit-variance AR(1).
    logdet_row_covariance = (n_configurations - 1) * np.log1p(-rho * rho)
    deviance = (n_features * logdet_row_covariance
                + n_configurations * float(logdet_scatter))
    if not np.isfinite(deviance):
        raise ValueError("profile deviance is non-finite")
    return float(deviance)


def _profile_common_rho_candidate(sample: np.ndarray) -> dict[str, Any]:
    """Fit a bounded scalar rho candidate under the stationary Gaussian model.

    A deterministic grid brackets every resolved local minimum before local
    refinement. A boundary solution must be treated as an abstention by callers.
    """
    lower, upper = -0.95, 0.95
    objective = lambda value: _profile_matrix_normal_deviance(sample, float(value))
    grid = np.linspace(lower, upper, PROFILE_RHO_GRID_POINTS)
    try:
        grid_deviance = np.asarray([objective(value) for value in grid], dtype=float)
        if not np.all(np.isfinite(grid_deviance)):
            raise ValueError("profile grid contains non-finite values")
        endpoint_candidates = [
            {"rho": float(grid[0]), "deviance": float(grid_deviance[0])},
            {"rho": float(grid[-1]), "deviance": float(grid_deviance[-1])},
        ]
        candidates = [(entry["rho"], entry["deviance"], "endpoint")
                      for entry in endpoint_candidates]
        local_minima = [
            index for index in range(1, len(grid) - 1)
            if (grid_deviance[index] <= grid_deviance[index - 1]
                and grid_deviance[index] <= grid_deviance[index + 1]
                and (grid_deviance[index] < grid_deviance[index - 1]
                     or grid_deviance[index] < grid_deviance[index + 1]))
        ]
        refined_minima = []
        for index in local_minima:
            bracket = (float(grid[index - 1]), float(grid[index + 1]))
            result = minimize_scalar(
                objective, bounds=bracket,
                method="bounded", options={"xatol": 1.0e-8, "maxiter": 500},
            )
            if not result.success or not np.isfinite(result.fun):
                raise ValueError("profile local-minimum refinement did not converge")
            minimum = {
                "grid_rho": float(grid[index]),
                "bracket": [bracket[0], bracket[1]],
                "rho": float(result.x),
                "deviance": float(result.fun),
            }
            refined_minima.append(minimum)
            candidates.append((minimum["rho"], minimum["deviance"], "refined_grid_minimum"))
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        raise ValueError("stationary matrix-normal rho profiling failed") from exc

    rho_hat, deviance, candidate_kind = min(candidates, key=lambda item: item[1])
    boundary_tolerance = 1.0e-6
    return {
        "rho_candidate": float(rho_hat),
        "profile_deviance": float(deviance),
        "global_candidate_kind": candidate_kind,
        "boundary_hit": bool(rho_hat <= lower + boundary_tolerance
                              or rho_hat >= upper - boundary_tolerance),
        "optimizer_success": True,
        "grid_points": int(len(grid)),
        "grid_spacing": float(grid[1] - grid[0]),
        "local_minima_refined": int(len(local_minima)),
        "refined_local_minima": refined_minima,
        "endpoint_candidates": endpoint_candidates,
        "profile_rho_grid": grid.tolist(),
        "profile_deviance_grid": grid_deviance.tolist(),
        "interpretation": (
            "stationary common-rho Gaussian candidate; not calibrated; lowest "
            "among endpoints and grid-resolved local minima only, so a narrow "
            "unsampled minimum is not excluded"
        ),
    }


def _rho_mean_variance_inflation(rho: float, count: int) -> float:
    lags = np.arange(1, count, dtype=float)
    return float(1.0 + 2.0 * np.sum((1.0 - lags / count) * rho**lags))


def _bootstrap_panel(rng: np.random.Generator, mean: np.ndarray,
                     covariance: np.ndarray, rho: float) -> np.ndarray:
    if rho == 0.0:
        return rng.multivariate_normal(mean, covariance, size=CONFIGURATIONS)
    return _sample_stationary_ar1(rng, mean, covariance, CONFIGURATIONS, rho)


def _plot(path: Path, panels: list[dict[str, Any]], summary: dict[str, Any]):
    method_specs = (
        ("chi_square_approx", "Approximate chi-square", "#2b78e4"),
        ("iid_bootstrap", "iid bootstrap", "#dd8452"),
        ("estimated_ar1_bootstrap", "Estimated-AR(1) bootstrap", "#55a868"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9), layout="constrained")
    ax = axes[0]
    x = np.linspace(0.0, 1.0, 101)
    n = summary["pvalue_summaries"]["estimated_ar1_bootstrap"]["count"]
    epsilon = math.sqrt(math.log(2.0 / 0.05) / (2.0 * max(n, 1)))
    ax.fill_between(x, np.maximum(0.0, x - epsilon), np.minimum(1.0, x + epsilon),
                    color="#cccccc", alpha=0.38, label=f"95% DKW band (n={n})")
    ax.plot(x, x, color="black", linestyle="--", label="uniform reference")
    for key, label, color in method_specs:
        values = sorted(float(row[key]) for row in panels if row[key] is not None)
        if values:
            ax.step(values, np.arange(1, len(values) + 1) / len(values),
                    where="post", color=color, label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="null p-value", ylabel="empirical CDF",
           title="AR(1) null panels")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[1]
    positions = {"chi_square_approx": 0, "iid_bootstrap": 1,
                 "estimated_ar1_bootstrap": 2}
    for key, _, color in method_specs:
        method_summary = summary["pvalue_summaries"][key]
        if method_summary is None:
            continue
        for threshold_index, threshold in enumerate(P_THRESHOLDS):
            result = method_summary["rejection"][str(threshold)]
            low, high = result["wilson_95"]
            offset = -0.09 if threshold_index == 0 else 0.09
            ax.errorbar(positions[key] + offset, result["rate"],
                        yerr=[[result["rate"] - low], [high - result["rate"]]],
                        fmt="o", capsize=3, color="#4c72b0" if threshold == 0.01 else color,
                        label=f"nominal {threshold:.0%}" if key == method_specs[0][0] else None)
    ax.axhline(0.01, color="#4c72b0", linestyle="--", alpha=0.65)
    ax.axhline(0.05, color="#dd8452", linestyle="--", alpha=0.65)
    ax.set_xticks([0, 1, 2], ["approx.\nchi-square", "iid\nbootstrap", "estimated AR(1)\nbootstrap"])
    upper = [entry["wilson_95"][1]
             for item in summary["pvalue_summaries"].values() if item is not None
             for entry in item["rejection"].values()]
    ax.set_ylim(0.0, max(0.12, 1.08 * max(upper, default=0.05)))
    ax.set_ylabel("observed rejection fraction")
    ax.set_title("Size estimates and Wilson 95% intervals")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[2]
    estimates = np.asarray([float(row["rho_hat"]) for row in panels
                            if row["rho_hat"] is not None])
    if len(estimates):
        ax.hist(estimates, bins=18, color="#8172b3", alpha=0.82, density=True)
        ax.axvline(TRUE_RHO, color="black", linestyle="--", label=f"true rho={TRUE_RHO:.2f}")
        ax.axvline(float(np.mean(estimates)), color="#c44e52", linestyle=":",
                   label=f"mean estimate={np.mean(estimates):.3f}")
    ax.set(xlabel="pooled lag-1 rho estimate", ylabel="density",
           title="Dependence parameter recovery")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _artifact_hashes(output: Path):
    files = sorted(path for path in output.rglob("*")
                    if path.is_file() and path.name != "artifact_hashes.json")
    values = {str(path.relative_to(output)): _sha(path.read_bytes()) for path in files}
    (output / "artifact_hashes.json").write_bytes(_json_bytes({
        "schema": "optimize-d04-ar1-bootstrap-artifact-hashes-v1",
        "artifacts": values,
    }))
    return values


def build(output: str | Path = DEFAULT_OUT):
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {out}")
    source_bytes = {name: path.read_bytes() for name, path in _SOURCE_PATHS.items()}
    source_hashes = {name: _sha(payload) for name, payload in source_bytes.items()}
    out.mkdir(parents=True)

    truth_mean = generator._truth_mean(generator.MASS_NULL,
                                       generator.PP_AMPLITUDES_NULL,
                                       generator.A4_AMPLITUDES_NULL)
    truth_covariance = generator._noise_covariance(truth_mean)
    outer_rng = np.random.default_rng(BASE_SEED)
    panels: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []
    outer_failures = rho_estimation_failures = 0
    draw_valid = {"iid_bootstrap": 0, "estimated_ar1_bootstrap": 0}
    minimum_valid = math.ceil(MIN_VALID_FRACTION * BOOTSTRAP_REPLICATES)

    for panel in range(OUTER_PANELS):
        sample = generator._sample_panel(
            outer_rng, truth_mean, truth_covariance, CONFIGURATIONS,
            dependence_rho=TRUE_RHO, innovation_df=None)
        row: dict[str, Any] = {
            "panel": panel,
            "sample_sha256": _sha(np.asarray(sample, dtype=np.float64).tobytes()),
            "configurations": CONFIGURATIONS,
            "true_rho": TRUE_RHO,
            "rho_hat": None,
            "fit_status": "failed",
            "chi_square_approx": None,
            "iid_bootstrap": None,
            "estimated_ar1_bootstrap": None,
        }
        try:
            rho_hat = _estimate_common_rho(sample)
            row["rho_hat"] = rho_hat
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            rho_estimation_failures += 1
            rho_hat = None
        try:
            fit, mean, fit_covariance, columns = bootstrap._fit_sample(sample)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            fit = {"status": "failed"}
        if fit.get("status") != "valid":
            outer_failures += 1
            panels.append(row)
            continue
        row["fit_status"] = "valid"
        row["chi_square_approx"] = float(fit["gof_p_approx"])
        try:
            fitted_mean = bootstrap._fitted_null_mean(fit, mean, fit_covariance, columns)
            sample_covariance, covariance_floor = bootstrap._regularize_sample_covariance(sample)
            row["sample_covariance_floor"] = covariance_floor
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            panels.append(row)
            continue

        for method_index, (method, rho_boot) in enumerate((
                ("iid_bootstrap", 0.0),
                ("estimated_ar1_bootstrap", rho_hat))):
            if rho_boot is None:
                continue
            seed = BASE_SEED + 100_000 + panel * 10 + method_index
            rng = np.random.default_rng(seed)
            exceedances = valid = failed = 0
            for replicate in range(BOOTSTRAP_REPLICATES):
                try:
                    bootstrap_sample = _bootstrap_panel(
                        rng, fitted_mean, sample_covariance, float(rho_boot))
                    boot_fit, _, _, _ = bootstrap._fit_sample(bootstrap_sample)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    boot_fit = {"status": "failed"}
                if boot_fit.get("status") == "valid":
                    valid += 1
                    draw_valid[method] += 1
                    exceedance = float(boot_fit["chi2"]) >= float(fit["chi2"])
                    exceedances += int(exceedance)
                    draw_rows.append({
                        "panel": panel, "method": method, "replicate": replicate,
                        "seed": seed, "rho_used": float(rho_boot),
                        "fit_status": "valid", "chi2": float(boot_fit["chi2"]),
                    })
                else:
                    failed += 1
                    draw_rows.append({
                        "panel": panel, "method": method, "replicate": replicate,
                        "seed": seed, "rho_used": float(rho_boot),
                        "fit_status": "failed", "chi2": None,
                    })
            row[method] = bootstrap._bootstrap_upper_tail_pvalue(
                exceedances=exceedances,
                valid_draws=valid,
                failed_draws=failed,
                planned_draws=BOOTSTRAP_REPLICATES,
                minimum_valid_draws=minimum_valid,
            )
            row[f"{method}_planned"] = BOOTSTRAP_REPLICATES
            row[f"{method}_valid"] = valid
            row[f"{method}_failed"] = failed
            row[f"{method}_seed"] = seed
        panels.append(row)

    approximate = [float(row["chi_square_approx"]) for row in panels
                   if row["chi_square_approx"] is not None]
    iid_values = [float(row["iid_bootstrap"]) for row in panels
                  if row["iid_bootstrap"] is not None]
    ar_values = [float(row["estimated_ar1_bootstrap"]) for row in panels
                 if row["estimated_ar1_bootstrap"] is not None]
    rho_values = np.asarray([float(row["rho_hat"]) for row in panels
                             if row["rho_hat"] is not None])
    delta_iid = [row["iid_bootstrap"] - row["chi_square_approx"] for row in panels
                 if row["iid_bootstrap"] is not None and row["chi_square_approx"] is not None]
    delta_ar = [row["estimated_ar1_bootstrap"] - row["chi_square_approx"] for row in panels
                if row["estimated_ar1_bootstrap"] is not None
                and row["chi_square_approx"] is not None]
    delta_ar_iid = [row["estimated_ar1_bootstrap"] - row["iid_bootstrap"] for row in panels
                    if row["estimated_ar1_bootstrap"] is not None
                    and row["iid_bootstrap"] is not None]
    summary = {
        "schema": "optimize-d04-ar1-bootstrap-summary-v1",
        "classification": "SYNTHETIC_AR1_DEPENDENCE_BOOTSTRAP_STRESS_NOT_PHYSICAL_REFERENCE",
        "generator": {
            "kind": "stationary_multivariate_gaussian_AR1",
            "outer_panels": OUTER_PANELS,
            "configurations_per_panel": CONFIGURATIONS,
            "true_rho": TRUE_RHO,
            "finite_n_mean_variance_inflation": _rho_mean_variance_inflation(TRUE_RHO, CONFIGURATIONS),
            "feature_count": len(truth_mean),
            "bootstrap_replicates_per_method_panel": BOOTSTRAP_REPLICATES,
            "bootstrap_methods": ["iid_bootstrap", "estimated_ar1_bootstrap"],
            "rho_estimator": "pooled lag-1 OLS on each raw feature centered by its panel mean, clipped to [-0.95, 0.95]",
            "bootstrap_covariance": "unshrunk raw within-panel sample covariance with eigenvalue floor 1e-12 of its largest eigenvalue",
            "bootstrap_rule": "re-estimate folded covariance with 25% shrinkage and refit the two-state spectrum in every replicate",
            "minimum_valid_fraction": MIN_VALID_FRACTION,
            "minimum_valid_draws": minimum_valid,
            "pvalue": "(1 + valid chi2 exceedances + failed refits) / (planned bootstrap draws + 1); failed refits count as conservative upper-tail exceedances",
        },
        "fit": {"tmin": 3, "nstates": 2, "nominal_dof": 13,
                "pp_a4p_joint": True, "multistarts": 3,
                "covariance_shrinkage": reader.SHRINKAGE,
                "covariance_eigenvalue_floor": reader.EIGENVALUE_FLOOR},
        "denominators": {
            "outer_fit": {"planned": OUTER_PANELS, "valid": len(approximate),
                          "failed": outer_failures, "abstained": 0},
            "rho_estimate": {"planned": OUTER_PANELS, "valid": len(rho_values),
                             "failed": rho_estimation_failures,
                             "abstained": 0},
            "iid_bootstrap_outer": {"planned": OUTER_PANELS,
                                     "valid": len(iid_values), "failed": outer_failures,
                                     "abstained": sum(row["iid_bootstrap"] is None for row in panels)},
            "estimated_ar1_bootstrap_outer": {"planned": OUTER_PANELS,
                                               "valid": len(ar_values), "failed": outer_failures,
                                               "abstained": sum(row["estimated_ar1_bootstrap"] is None for row in panels)},
        },
        "pvalue_summaries": {
            "chi_square_approx": bootstrap._pvalue_summary(approximate),
            "iid_bootstrap": bootstrap._pvalue_summary(iid_values),
            "estimated_ar1_bootstrap": bootstrap._pvalue_summary(ar_values),
        },
        "rho_estimate_summary": {
            "count": len(rho_values),
            "mean": float(np.mean(rho_values)) if len(rho_values) else None,
            "median": float(np.median(rho_values)) if len(rho_values) else None,
            "sd": float(np.std(rho_values, ddof=1)) if len(rho_values) > 1 else None,
            "bias": float(np.mean(rho_values) - TRUE_RHO) if len(rho_values) else None,
            "q05_q95": [float(v) for v in np.quantile(rho_values, [0.05, 0.95])] if len(rho_values) else None,
        },
        "paired_comparisons": {
            "iid_bootstrap_minus_approx": {"valid": len(delta_iid), "mean": float(np.mean(delta_iid)) if delta_iid else None,
                                             "fraction_bootstrap_lower": float(np.mean(np.asarray(delta_iid) < 0.0)) if delta_iid else None},
            "estimated_ar1_bootstrap_minus_approx": {"valid": len(delta_ar), "mean": float(np.mean(delta_ar)) if delta_ar else None,
                                                      "fraction_bootstrap_lower": float(np.mean(np.asarray(delta_ar) < 0.0)) if delta_ar else None},
            "estimated_ar1_bootstrap_minus_iid_bootstrap": {"valid": len(delta_ar_iid), "mean": float(np.mean(delta_ar_iid)) if delta_ar_iid else None,
                                                            "fraction_ar1_pvalue_lower": float(np.mean(np.asarray(delta_ar_iid) < 0.0)) if delta_ar_iid else None},
        },
        "physical_e_chain": {"planned": 0, "valid": 0, "failed": 0, "abstained": 0},
        "reference_gate": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
        "physical_score": None,
        "open_gates": [
            "only one stationary Gaussian AR(1) rho=0.30 generator is evaluated",
            "rho-hat uncertainty is represented only through the per-panel plug-in estimate; no block/HAC or multiscale chain bootstrap is evaluated",
            "finite-sample rejection intervals remain Monte Carlo limited and the method uses one independent outer validation set only",
            "a synthetic process does not establish production chain stationarity, source provenance, or candidate blindness",
            "an independent reviewer must inspect the rho estimator, fitted-mean reconstruction, and covariance convention",
        ],
    }
    # Correct draw denominators from the explicit per-method planned/valid/failed totals.
    for method, key in (("iid_bootstrap", "iid_bootstrap_draws"),
                        ("estimated_ar1_bootstrap", "estimated_ar1_bootstrap_draws")):
        failed = sum(row["method"] == method and row["fit_status"] == "failed" for row in draw_rows)
        planned = OUTER_PANELS * BOOTSTRAP_REPLICATES
        summary["denominators"][key] = {
            "planned": planned,
            "valid": draw_valid[method],
            "failed": failed,
            "abstained": planned - draw_valid[method] - failed,
        }
    summary["denominators"]["bootstrap_outer_total"] = {
        "planned": 2 * OUTER_PANELS,
        "valid": len(iid_values) + len(ar_values),
        "failed": 2 * outer_failures,
        "abstained": (OUTER_PANELS - len(iid_values)) + (OUTER_PANELS - len(ar_values)) - outer_failures * 2,
    }
    (out / "summary.json").write_bytes(_json_bytes(summary))
    _write_csv(out / "panel_results.csv", panels)
    _write_csv(out / "bootstrap_draw_results.csv", draw_rows)
    _plot(out / "ar1_bootstrap_calibration.png", panels, summary)
    protocol = {
        "schema": "optimize-d04-ar1-bootstrap-protocol-v1",
        "classification": summary["classification"],
        "source_hashes": source_hashes,
        "base_seed": BASE_SEED,
        "outer_seed_rule": "single PCG64 stream; separate seed from the 20261127 iid pilot; no input file read",
        "outer_panels": OUTER_PANELS,
        "configurations_per_panel": CONFIGURATIONS,
        "true_rho": TRUE_RHO,
        "bootstrap_replicates_per_method_panel": BOOTSTRAP_REPLICATES,
        "bootstrap_seed_rule": "BASE_SEED + 100000 + panel*10 + method index; iid index 0, estimated AR(1) index 1",
        "fit_rule": "same D04 joint PP-even/A4P-odd two-state fit, tmin=3, three starts",
        "methods": {
            "approximate": "reader's existing 13-dof approximate chi-square p-value",
            "iid_bootstrap": "plug-in sample covariance and rho=0 synthetic replicates",
            "estimated_ar1_bootstrap": "same plug-in covariance with pooled lag-1 OLS rho estimated from that outer panel",
        },
        "pvalue_rule": summary["generator"]["pvalue"],
        "finite_n_variance_inflation": summary["generator"]["finite_n_mean_variance_inflation"],
        "physical_score": None,
    }
    (out / "protocol.json").write_bytes(_json_bytes(protocol))
    source_dir = out / "source_snapshot"
    source_dir.mkdir()
    for name, payload in source_bytes.items():
        (source_dir / name.replace("/", "_")).write_bytes(payload)
    (out / "source_hashes.json").write_bytes(_json_bytes({
        "schema": "optimize-d04-ar1-bootstrap-source-hashes-v1",
        "sources": source_hashes,
    }))
    readme = [
        "# D04 AR(1) dependence-aware bootstrap stress test",
        "",
        "**Status:** synthetic CPU validation only; it does not create a physical E-chain reference or change `D04 NOT_READY/UNCERTIFIED`.",
        "",
        "## Frozen design",
        "",
        f"- Generate `{OUTER_PANELS}` independent held-out null panels with `{CONFIGURATIONS}` ordered configurations each from a stationary 27-feature Gaussian AR(1), true `rho={TRUE_RHO:.2f}`. The exact finite-N variance inflation for a sample mean is `{summary['generator']['finite_n_mean_variance_inflation']:.4f}`.",
        "- On each same outer panel, compare the current approximate chi-square p-value, an iid plug-in bootstrap, and an estimated-AR(1) plug-in bootstrap. Both bootstrap methods use the same panel-specific fitted null mean and raw feature covariance; only the serial-dependence parameter differs.",
        f"- The dependence estimate is pooled lag-1 OLS over all 27 raw feature series after subtracting each series' panel mean, clipped to `[-0.95, 0.95]`. Bootstrap samples use that estimate for AR(1), or zero for the iid comparator. Each replicate refolds, re-estimates/shrinks covariance by `{reader.SHRINKAGE:.0%}`, and refits with three starts.",
        f"- Each method plans `{BOOTSTRAP_REPLICATES}` inner refits per outer panel. The conservative upper-tail p-value is `(1 + valid exceedances + failed refits)/(planned draws + 1)`; failed refits count as exceedances, and fewer than `{minimum_valid}` valid draws abstain.",
        "",
        "![AR(1) bootstrap calibration diagnostics](ar1_bootstrap_calibration.png)",
        "",
        "The left panel compares the three null p-value empirical CDFs with the uniform reference. The center panel shows 1% and 5% rejection rates with Wilson 95% intervals. The right panel shows how well the pooled rho estimator recovers the known synthetic value.",
        "",
        "## Results",
        "",
        f"- Denominators: `{json.dumps(summary['denominators'], sort_keys=True)}`.",
        f"- P-value summaries: `{json.dumps(summary['pvalue_summaries'], sort_keys=True)}`.",
        f"- Estimated rho summary: `{json.dumps(summary['rho_estimate_summary'], sort_keys=True)}`.",
        f"- Paired contrasts: `{json.dumps(summary['paired_comparisons'], sort_keys=True)}`.",
        "",
        "## Interpretation and next work",
        "",
        "This design isolates one failure mode already visible in the earlier stress report: treating correlated configurations as iid can make the nominal GOF test over-reject. A better result for the estimated-AR(1) bootstrap would support only this synthetic stationary Gaussian mechanism. It would not validate an HMC chain, select a production dependence model, or establish calibration after fit-window/model selection. The plug-in covariance, pooled rho estimator, and fitted-mean reconstruction still need independent statistical review. The next software comparison should freeze a wider rho/tail grid and compare AR(1), circular block, and HAC-based bootstrap under an untouched outer set. Before a physical D04 reference can be made, obtain an authorized candidate-blind E-chain snapshot with provenance/dependence receipts and production source/sink conventions. Keep `physical_score=null`; do not open sealed data or infer GPU/QUDA evidence from this CPU run.",
        "",
        "## Reproduction and integrity",
        "",
        "Run from the repository root: `python -m tasks.optimize_gaussian.d04_ar1_bootstrap_diagnostic`. The report refuses to overwrite an existing output directory. Full panel and per-draw tables, the frozen protocol, source snapshots/hashes, and artifact hashes are saved alongside the figure.",
        "- SHA-256 manifests are written after the README, tables, figure, protocol, and source snapshots are complete.",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    artifacts = _artifact_hashes(out)
    return {"output": str(out), "outer_panels": len(panels),
            "draw_rows": len(draw_rows), "artifact_count": len(artifacts),
            "source_count": len(source_hashes), "draw_valid": draw_valid}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
