"""Nested parametric-bootstrap pilot for the D04 approximate GLS GOF p-value."""
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

from tasks.optimize_gaussian import d04_gof_calibration_diagnostic as generator
from tasks.optimize_gaussian import d04_raw_reference as reader


TASK_DIR = Path(__file__).resolve().parent
ROOT = TASK_DIR.parents[1]
DEFAULT_OUT = TASK_DIR / "d04_parametric_bootstrap_cpu_20261127_v1"
OUTER_PANELS = 60
BOOTSTRAP_REPLICATES = 199
MIN_VALID_FRACTION = 0.90
CONFIGURATIONS = 128
BASE_SEED = 2026112753
P_THRESHOLDS = (0.01, 0.05)
_SOURCE_PATHS = {
    "diagnostic": Path(__file__),
    "generator": TASK_DIR / "d04_gof_calibration_diagnostic.py",
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


def _fit_sample(sample: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[tuple[str, int]]]:
    folded_pp, folded_a4p = generator._folded_arrays(sample)
    mean, covariance, columns = reader._fit_inputs(folded_pp, folded_a4p)
    fit = reader._fit_variant(mean, covariance, columns, tmin=3, nstates=2)
    return fit, mean, covariance, columns


def _fitted_null_mean(fit: dict[str, Any], mean: np.ndarray,
                      covariance: np.ndarray,
                      columns: list[tuple[str, int]]) -> np.ndarray:
    """Reconstruct all 27 fitted means using GLS amplitudes at fitted masses."""
    masses = np.asarray(fit["state_masses"], dtype=float)
    selected = reader._selected_indices(columns, 3)
    design, labels = reader._design(masses, 2, 3)
    if labels != [columns[index] for index in selected]:
        raise ValueError("D04 fitted-null design and selected observation order differ")
    fit_covariance = covariance[np.ix_(selected, selected)]
    eigenvalues, eigenvectors = np.linalg.eigh(fit_covariance)
    scale = float(np.max(eigenvalues))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("D04 fit covariance is unidentifiable for bootstrap mean")
    eigenvalues = np.maximum(eigenvalues, scale * reader.EIGENVALUE_FLOOR)
    root = np.sqrt(eigenvalues)
    whitened_design = (eigenvectors.T @ design) / root[:, None]
    whitened_y = (eigenvectors.T @ mean[selected]) / root
    amplitudes, *_ = np.linalg.lstsq(whitened_design, whitened_y, rcond=None)
    full_design, full_labels = reader._design(masses, 2, 2)
    if full_labels != generator._feature_descriptors():
        raise ValueError("D04 full fitted-null design does not match the generator feature order")
    fitted_mean = full_design @ amplitudes
    if not np.all(np.isfinite(fitted_mean)):
        raise ValueError("D04 fitted-null mean contains non-finite values")
    return fitted_mean


def _regularize_sample_covariance(sample: np.ndarray) -> tuple[np.ndarray, float]:
    covariance = np.atleast_2d(np.cov(sample, rowvar=False, ddof=1))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = float(np.max(eigenvalues))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("plug-in raw sample covariance is unidentifiable")
    floor = scale * 1.0e-12
    covariance = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    return covariance, floor


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials**2)) / denominator
    if successes == 0:
        return 0.0, center + half
    if successes == trials:
        return center - half, 1.0
    return center - half, center + half


def _pvalue_summary(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    result: dict[str, Any] = {
        "count": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "ks_distance_to_uniform": float(np.max(np.abs(
            np.sort(array) - (np.arange(len(array)) + 0.5) / len(array)))),
        "rejection": {},
    }
    for threshold in P_THRESHOLDS:
        count = int(np.count_nonzero(array < threshold))
        result["rejection"][str(threshold)] = {
            "count": count, "rate": count / len(array),
            "wilson_95": list(_wilson(count, len(array))),
            "nominal_rate": threshold,
            "mcse": math.sqrt(threshold * (1.0 - threshold) / len(array)),
        }
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path.name}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, panel_rows: list[dict[str, Any]], summary: dict[str, Any]):
    methods = (
        ("chi_square_approx", "approximate chi-square", "#2b78e4"),
        ("parametric_bootstrap", "parametric bootstrap", "#cc6677"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), layout="constrained")
    ax = axes[0]
    x = np.linspace(0.0, 1.0, 101)
    n = min(summary["denominators"]["bootstrap_outer"]["valid"],
            summary["denominators"]["chi_square"]["valid"])
    epsilon = math.sqrt(math.log(2.0 / 0.05) / (2.0 * max(n, 1)))
    ax.fill_between(x, np.maximum(0.0, x - epsilon), np.minimum(1.0, x + epsilon),
                    color="#cccccc", alpha=0.4, label=f"95% DKW band (n={n})")
    ax.plot(x, x, color="black", linestyle="--", label="uniform reference")
    for key, label, color in methods:
        values = sorted(float(row[key]) for row in panel_rows if row[key] is not None)
        if values:
            ecdf = np.arange(1, len(values) + 1) / len(values)
            ax.step(values, ecdf, where="post", color=color, label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="null p-value",
           ylabel="empirical CDF", title="Outer null panels")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[1]
    positions = {"chi_square_approx": 0, "parametric_bootstrap": 1}
    for key, label, color in methods:
        method_summary = summary["pvalue_summaries"][key]
        if method_summary is None:
            continue
        for threshold_index, threshold in enumerate(P_THRESHOLDS):
            item = method_summary["rejection"][str(threshold)]
            low, high = item["wilson_95"]
            offset = -0.08 if threshold_index == 0 else 0.08
            ax.errorbar(positions[key] + offset, item["rate"],
                        yerr=[[item["rate"] - low], [high - item["rate"]]],
                        fmt="o", capsize=4,
                        color="#2b78e4" if threshold == 0.01 else "#e69138",
                        label=f"nominal {threshold:.0%}" if key == methods[0][0] else None)
    ax.axhline(0.01, color="#2b78e4", linestyle="--", alpha=0.65)
    ax.axhline(0.05, color="#e69138", linestyle="--", alpha=0.65)
    ax.set_xticks([0, 1], ["approximate\nchi-square", "parametric\nbootstrap"])
    upper = [item["wilson_95"][1]
             for method in summary["pvalue_summaries"].values() if method is not None
             for item in method["rejection"].values()]
    ax.set_ylim(0, max(0.12, 1.12 * max(upper, default=0.05)))
    ax.set_ylabel("observed rejection fraction")
    ax.set_title("Size estimates with Wilson 95% intervals")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _artifact_hashes(output: Path):
    files = sorted(path for path in output.rglob("*")
                    if path.is_file() and path.name != "artifact_hashes.json")
    values = {str(path.relative_to(output)): _sha(path.read_bytes()) for path in files}
    (output / "artifact_hashes.json").write_bytes(_json_bytes({
        "schema": "optimize-d04-parametric-bootstrap-artifact-hashes-v1",
        "artifacts": values,
    }))
    return values


def build(output: str | Path = DEFAULT_OUT):
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {out}")
    source_bytes = {name: path.read_bytes() for name, path in _SOURCE_PATHS.items()}
    source_hashes = {name: _sha(value) for name, value in source_bytes.items()}
    out.mkdir(parents=True)

    truth_mean = generator._truth_mean(
        generator.MASS_NULL, generator.PP_AMPLITUDES_NULL,
        generator.A4_AMPLITUDES_NULL)
    truth_covariance = generator._noise_covariance(truth_mean)
    outer_rng = np.random.default_rng(BASE_SEED)
    outer_rows, draw_rows = [], []
    outer_failures = bootstrap_outer_abstentions = 0
    bootstrap_draw_abstentions = bootstrap_failures = bootstrap_valid = 0
    minimum_valid = math.ceil(MIN_VALID_FRACTION * BOOTSTRAP_REPLICATES)

    for panel in range(OUTER_PANELS):
        sample = outer_rng.multivariate_normal(
            truth_mean, truth_covariance, size=CONFIGURATIONS)
        outer_row = {
            "panel": panel, "sample_seed": BASE_SEED,
            "sample_sha256": _sha(np.asarray(sample, dtype=np.float64).tobytes()),
            "configurations": CONFIGURATIONS,
            "chi_square_approx": None, "parametric_bootstrap": None,
            "fit_status": "failed", "bootstrap_planned": BOOTSTRAP_REPLICATES,
            "bootstrap_valid": 0, "bootstrap_failed": 0,
            "bootstrap_abstained": False, "bootstrap_abstained_draws": 0,
            "bootstrap_seed": BASE_SEED + 50000 + panel,
            "sample_covariance_floor": None,
        }
        try:
            fit, mean, fit_covariance, columns = _fit_sample(sample)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            fit = {"status": "failed"}
        if fit.get("status") != "valid":
            outer_failures += 1
            outer_row["bootstrap_abstained"] = True
            outer_row["bootstrap_abstained_draws"] = BOOTSTRAP_REPLICATES
            bootstrap_draw_abstentions += BOOTSTRAP_REPLICATES
            outer_rows.append(outer_row)
            continue

        outer_row["fit_status"] = "valid"
        outer_row["chi_square_approx"] = float(fit["gof_p_approx"])
        try:
            fitted_mean = _fitted_null_mean(fit, mean, fit_covariance, columns)
            sample_covariance, covariance_floor = _regularize_sample_covariance(sample)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            outer_row["bootstrap_abstained"] = True
            outer_row["bootstrap_abstained_draws"] = BOOTSTRAP_REPLICATES
            bootstrap_outer_abstentions += 1
            bootstrap_draw_abstentions += BOOTSTRAP_REPLICATES
            outer_rows.append(outer_row)
            continue
        outer_row["sample_covariance_floor"] = covariance_floor
        bootstrap_rng = np.random.default_rng(outer_row["bootstrap_seed"])
        replicate_chi2 = []

        for replicate in range(BOOTSTRAP_REPLICATES):
            try:
                bootstrap_sample = bootstrap_rng.multivariate_normal(
                    fitted_mean, sample_covariance, size=CONFIGURATIONS)
                boot_fit, _, _, _ = _fit_sample(bootstrap_sample)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                boot_fit = {"status": "failed"}
            valid = boot_fit.get("status") == "valid"
            if valid:
                value = float(boot_fit["chi2"])
                replicate_chi2.append(value)
                bootstrap_valid += 1
            else:
                value = None
                bootstrap_failures += 1
            draw_rows.append({
                "panel": panel, "replicate": replicate,
                "bootstrap_seed": outer_row["bootstrap_seed"],
                "fit_status": "valid" if valid else "failed",
                "chi2": value,
            })

        valid_draws = len(replicate_chi2)
        failed_draws = BOOTSTRAP_REPLICATES - valid_draws
        outer_row["bootstrap_valid"] = valid_draws
        outer_row["bootstrap_failed"] = failed_draws
        if valid_draws >= minimum_valid:
            outer_row["parametric_bootstrap"] = (
                1.0 + sum(value >= float(fit["chi2"]) for value in replicate_chi2)
            ) / (valid_draws + 1.0)
        else:
            outer_row["bootstrap_abstained"] = True
            bootstrap_outer_abstentions += 1
        outer_rows.append(outer_row)

    approx_values = [float(row["chi_square_approx"]) for row in outer_rows
                     if row["chi_square_approx"] is not None]
    bootstrap_values = [float(row["parametric_bootstrap"]) for row in outer_rows
                        if row["parametric_bootstrap"] is not None]
    bootstrap_outer_failed = sum(row["fit_status"] != "valid" for row in outer_rows)
    delta = [row["parametric_bootstrap"] - row["chi_square_approx"]
             for row in outer_rows if row["parametric_bootstrap"] is not None
             and row["chi_square_approx"] is not None]
    paired = {
        "planned": OUTER_PANELS,
        "valid": len(delta),
        "failed": bootstrap_outer_failed,
        "abstained": bootstrap_outer_abstentions,
        "mean_delta_bootstrap_minus_chi_square": float(np.mean(delta)) if delta else None,
        "median_delta_bootstrap_minus_chi_square": float(np.median(delta)) if delta else None,
        "fraction_bootstrap_pvalue_lower": float(np.mean(np.asarray(delta) < 0.0)) if delta else None,
    }
    summary = {
        "schema": "optimize-d04-parametric-bootstrap-summary-v1",
        "classification": "SYNTHETIC_IID_GAUSSIAN_PARAMETRIC_BOOTSTRAP_PILOT_NOT_PHYSICAL_REFERENCE",
        "fit": {"tmin": 3, "nstates": 2, "nominal_dof": 13,
                "pp_a4p_joint": True, "multistarts": 3,
                "covariance_shrinkage": reader.SHRINKAGE,
                "covariance_eigenvalue_floor": reader.EIGENVALUE_FLOOR},
        "generator": {"kind": "independent_multivariate_Gaussian_configuration_panels",
                       "configurations_per_panel": CONFIGURATIONS,
                       "feature_count": len(truth_mean), "outer_panels": OUTER_PANELS,
                       "bootstrap_replicates_per_panel": BOOTSTRAP_REPLICATES,
                       "bootstrap_minimum_valid_fraction": MIN_VALID_FRACTION,
                       "bootstrap_minimum_valid_replicates": minimum_valid,
                       "bootstrap_covariance": "unshrunk raw within-panel sample covariance, eigenvalue floor 1e-12 of max eigenvalue; every replicate refits the spectrum and re-estimates/shrinks covariance",
                       "bootstrap_pvalue": "(1 + count(bootstrap chi2 >= observed chi2)) / (valid bootstrap draws + 1)"},
        "denominators": {
            "chi_square": {"planned": OUTER_PANELS, "valid": len(approx_values),
                           "failed": outer_failures, "abstained": 0},
            "bootstrap_outer": {"planned": OUTER_PANELS,
                                 "valid": len(bootstrap_values),
                                 "failed": bootstrap_outer_failed,
                                 "abstained": bootstrap_outer_abstentions},
            "bootstrap_draws": {"planned": OUTER_PANELS * BOOTSTRAP_REPLICATES,
                                "valid": bootstrap_valid,
                                "failed": bootstrap_failures,
                                "abstained": bootstrap_draw_abstentions},
        },
        "pvalue_summaries": {
            "chi_square_approx": _pvalue_summary(approx_values),
            "parametric_bootstrap": _pvalue_summary(bootstrap_values),
        },
        "paired_comparison": paired,
        "physical_e_chain": {"planned": 0, "valid": 0, "failed": 0, "abstained": 0},
        "reference_gate": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
        "physical_score": None,
        "open_gates": [
            "the outer null panel count is 60 and the 1%/5% rejection MC errors are coarse",
            "only the iid Gaussian configuration model is assessed; no AR(1), heavy-tail, chain-selection or production E-chain validation is performed here",
            "bootstrap draws use a plug-in raw sample covariance and are nested within each outer panel; independent calibration/validation panels are still required",
            "these synthetic results cannot replace a candidate-blind E-chain reference receipt or independent statistical review",
        ],
    }
    (out / "summary.json").write_bytes(_json_bytes(summary))
    _write_csv(out / "panel_results.csv", outer_rows)
    _write_csv(out / "bootstrap_draw_results.csv", draw_rows)
    _plot(out / "bootstrap_calibration.png", outer_rows, summary)
    protocol = {
        "schema": "optimize-d04-parametric-bootstrap-protocol-v1",
        "classification": summary["classification"],
        "source_hashes": source_hashes,
        "base_seed": BASE_SEED,
        "outer_panels": OUTER_PANELS,
        "configurations_per_panel": CONFIGURATIONS,
        "bootstrap_replicates_per_panel": BOOTSTRAP_REPLICATES,
        "bootstrap_seed_rule": "BASE_SEED + 50000 + outer panel index",
        "minimum_valid_draws": minimum_valid,
        "fit_rule": "reader._fit_variant(mean, covariance_of_mean, columns, tmin=3, nstates=2)",
        "bootstrap_rule": summary["generator"]["bootstrap_pvalue"],
        "outer_generator": {"truth_mean_sha256": _sha(np.asarray(truth_mean, dtype=np.float64).tobytes()),
                            "truth_covariance_sha256": _sha(np.asarray(truth_covariance, dtype=np.float64).tobytes()),
                            "temporal_feature_correlation": 0.65,
                            "cross_channel_correlation": 0.2,
                            "relative_noise": generator.NOISE_FRACTION},
        "outer_seed_rule": "single PCG64 stream initialized at BASE_SEED; no input file read",
        "physical_score": None,
    }
    (out / "protocol.json").write_bytes(_json_bytes(protocol))
    source_dir = out / "source_snapshot"
    source_dir.mkdir()
    for name, payload in source_bytes.items():
        (source_dir / name.replace("/", "_")).write_bytes(payload)
    (out / "source_hashes.json").write_bytes(_json_bytes({
        "schema": "optimize-d04-parametric-bootstrap-source-hashes-v1",
        "sources": source_hashes,
    }))
    readme = [
        "# D04 nested parametric-bootstrap pilot",
        "",
        "**Status:** synthetic CPU diagnostic only. This does not provide a candidate-blind E-chain reference or change `D04 NOT_READY/UNCERTIFIED`.",
        "",
        "## Frozen procedure",
        "",
        f"- Generate `{OUTER_PANELS}` independent exact two-state Gaussian null panels with `{CONFIGURATIONS}` configurations each; fit the current D04 joint PP-even/A4P-odd two-state model (`tmin=3`, 13 nominal dof, three deterministic starts).",
        f"- For every valid outer fit, estimate the plug-in raw feature covariance from that panel, reconstruct the fitted null mean by solving GLS amplitudes at fitted masses, and draw `{BOOTSTRAP_REPLICATES}` parametric panels. Every bootstrap panel is folded, its covariance is re-estimated with the current `{reader.SHRINKAGE:.0%}` shrinkage and eigenvalue floor, and the spectrum is refit with all three starts.",
        f"- The bootstrap p-value is `(1 + exceedances)/(valid draws + 1)`. A panel needs at least `{minimum_valid}` valid bootstrap fits (`{MIN_VALID_FRACTION:.0%}`) or its bootstrap p-value abstains; failed draws stay in their planned denominator.",
        "- Bootstrap covariance uses the unshrunk within-panel sample covariance, with only an eigenvalue floor `1e-12` of its largest eigenvalue to draw from a positive-definite plug-in model. This covariance is not the fit covariance; the latter is re-estimated and shrunk in every replicate.",
        "",
        "![D04 approximate and bootstrap GOF comparison](bootstrap_calibration.png)",
        "",
        "The left panel compares the empirical p-value CDFs from the same outer null panels; the right panel shows nominal 1%/5% rejection rates and Wilson intervals. The bootstrap and chi-square p-values are paired panel-by-panel.",
        "",
        "## Observed results",
        "",
    ]
    for key, label in (("chi_square_approx", "Approximate chi-square"),
                       ("parametric_bootstrap", "Parametric bootstrap")):
        readme.append(f"- {label}: denominator `{summary['denominators']['bootstrap_outer' if key == 'parametric_bootstrap' else 'chi_square']}`; p-value summary `{summary['pvalue_summaries'][key]}`.")
    readme.extend([
        f"- Paired bootstrap minus chi-square comparison: `{paired}`.",
        "",
        "## Interpretation limits and next step",
        "",
        "At 60 outer panels, the rejection-rate Monte Carlo standard error is about `0.0128` at 1% and `0.0281` at 5%. The B=199 bootstrap resolution is 0.005 before accounting for within-bootstrap Monte Carlo noise. This pilot can expose major miscalibration and exercise failure accounting, but cannot certify size or coverage. It assumes an iid Gaussian configuration generator, uses one plug-in covariance per outer panel, and has no AR(1), Student-t, HMC dependence, fit-window selection, or production source/sink conventions. The next calibration must use larger, independent outer/inner panels; compare iid and dependence-aware resampling under a wider rho/tail grid; and keep a separate untouched validation set. Keep `physical_score=null` until an authorized, candidate-blind E-chain snapshot and independent review close the D04 gate.",
        "",
        "## Run denominators",
        "",
        f"- Outer panels: P/V/F/A = `{summary['denominators']['chi_square']['planned']}/{summary['denominators']['chi_square']['valid']}/{summary['denominators']['chi_square']['failed']}/{summary['denominators']['chi_square']['abstained']}` for the approximate statistic; bootstrap outer p-values retain `{summary['denominators']['bootstrap_outer']}`.",
        f"- Nested bootstrap fits: P/V/F/A = `{summary['denominators']['bootstrap_draws']['planned']}/{summary['denominators']['bootstrap_draws']['valid']}/{summary['denominators']['bootstrap_draws']['failed']}/{summary['denominators']['bootstrap_draws']['abstained']}`.",
    ])
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    artifacts = _artifact_hashes(out)
    return {"output": str(out), "panels": len(outer_rows),
            "bootstrap_draws": len(draw_rows), "artifact_count": len(artifacts),
            "bootstrap_valid": bootstrap_valid}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
