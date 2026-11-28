"""Compare HAC ratio-delta and HAC contrast-inversion lower bounds.

CPU-only synthetic coverage stress. It uses known finite-sample D04 risk truths
for Gaussian, heavy-tailed, and periodically heteroskedastic AR(1) panels, and
does not read lattice data or authorize a physical comparison.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .d06_bootstrap_coverage_diagnostic import ARMS, MEANS, SIGMAS, _wilson_interval
from .d06_hac_coverage_diagnostic import (
    COMPARISONS,
    HAC_BANDWIDTHS,
    d04_risk_and_influence,
    newey_west_long_run_variance,
)
from .risk_statistics import D06_ALPHA_PER_COMPARISON, D06_BLOCK_LENGTHS, D06_R_STAR


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/optimize_gaussian/d06_hac_fieller_stress_cpu_20261125_v1"
N_SCENARIOS = (40, 60, 100)
PHI_SCENARIOS = (0.0, 0.4, 0.8)
FAMILIES = ("gaussian", "student_t5", "periodic_volatility")
COMMON_LOADINGS = (0.0, 0.8)
MC_REPLICATES = 1000
BASE_SEED = 2026112511


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _standard_innovation(rng: np.random.Generator, family: str,
                          shape: tuple[int, ...]) -> np.ndarray:
    if family == "student_t5":
        return rng.standard_t(5, size=shape) / math.sqrt(5.0 / 3.0)
    if family in ("gaussian", "periodic_volatility"):
        return rng.normal(size=shape)
    raise ValueError(f"unknown innovation family {family!r}")


def _innovation_variances(n: int, family: str) -> np.ndarray:
    if family == "periodic_volatility":
        return np.asarray([0.5 if index % 2 == 1 else 1.5 for index in range(n)])
    if family in ("gaussian", "student_t5"):
        return np.ones(n, dtype=float)
    raise ValueError(f"unknown innovation family {family!r}")


def _ar1(rng: np.random.Generator, n: int, trailing: tuple[int, ...],
         phi: float, family: str) -> np.ndarray:
    if n < 2 or not 0 <= phi < 1:
        raise ValueError("n must be >=2 and phi in [0,1)")
    variances = _innovation_variances(n, family)
    if family == "periodic_volatility":
        # Period-2 stationary variances for innovation variance (1.5, 0.5).
        s0, s1 = 1.5, 0.5
        denominator = 1.0 - phi**4
        initial_variance = (s0 + phi**2 * s1) / denominator
    else:
        initial_variance = 1.0 / (1.0 - phi**2)
    values = np.empty((n, *trailing), dtype=float)
    values[0] = _standard_innovation(rng, family, trailing) * math.sqrt(initial_variance)
    for index in range(1, n):
        innovation = _standard_innovation(rng, family, trailing)
        values[index] = phi * values[index - 1] + math.sqrt(variances[index]) * innovation
    return values


def generate_panel(seed: int, phi: float, n: int, family: str,
                   common_loading: float) -> np.ndarray:
    """Return paired q[configuration, arm, time] with declared cross-arm noise."""
    if not 0 <= common_loading < 1:
        raise ValueError("common_loading must be in [0,1)")
    rng = np.random.default_rng(seed)
    common = _ar1(rng, n, (len(MEANS[0]),), phi, family)
    independent = _ar1(rng, n, (len(ARMS), len(MEANS[0])), phi, family)
    noise = (common_loading * common[:, None, :]
             + math.sqrt(1.0 - common_loading**2) * independent)
    return MEANS[None, :, :] + SIGMAS[None, :, None] * noise


def true_risks(phi: float, n: int, family: str) -> dict[str, float]:
    """Exact finite-n D04 targets; only second moments enter the risk truth."""
    innovation_variance = _innovation_variances(n, family)
    if family == "periodic_volatility":
        s0, s1 = 1.5, 0.5
        denominator = 1.0 - phi**4
        current = (s0 + phi**2 * s1) / denominator
    else:
        current = 1.0 / (1.0 - phi**2)
    marginal_variance = np.empty(n, dtype=float)
    marginal_variance[0] = current
    for index in range(1, n):
        marginal_variance[index] = phi**2 * marginal_variance[index - 1] + innovation_variance[index]
    variance_of_mean = float(np.sum(marginal_variance) / (n * n))
    for lag in range(1, n):
        covariance_sum = float(np.sum(marginal_variance[:-lag])) * phi**lag
        variance_of_mean += 2.0 * covariance_sum / (n * n)
    return {
        arm: float(np.mean(np.square(MEANS[index] - 1.0)
                           + SIGMAS[index] ** 2 * variance_of_mean))
        for index, arm in enumerate(ARMS)
    }


def _hac_contrast_coefficients(if_comparator: np.ndarray, if_candidate: np.ndarray,
                               bandwidth: int) -> tuple[float, float, float]:
    """Return omega_cc, omega_cw, omega_ww for IFc - r*IFw."""
    c = np.asarray(if_comparator, dtype=float)
    w = np.asarray(if_candidate, dtype=float)
    if c.ndim != 1 or w.shape != c.shape or c.size < 2:
        raise ValueError("influence arrays must be same-size one-dimensional vectors")
    c = c - np.mean(c)
    w = w - np.mean(w)
    n = c.size

    def bilinear(left: np.ndarray, right: np.ndarray) -> float:
        total = float(np.dot(left, right) / n)
        for lag in range(1, min(bandwidth, n - 1) + 1):
            weight = 1.0 - lag / (bandwidth + 1.0)
            total += weight * (
                float(np.dot(left[:-lag], right[lag:]) / n)
                + float(np.dot(right[:-lag], left[lag:]) / n)
            )
        return total

    return bilinear(c, c), bilinear(c, w), bilinear(w, w)


def fieller_hac_lower(risk_comparator: float, risk_candidate: float,
                      omega_cc: float, omega_cw: float, omega_ww: float,
                      n: int, z: float) -> float:
    """Invert a one-sided HAC studentized contrast for a positive risk ratio."""
    values = (risk_comparator, risk_candidate, omega_cc, omega_cw, omega_ww, z)
    if (n < 2 or any(not math.isfinite(float(value)) for value in values)
            or risk_comparator <= 0 or risk_candidate <= 0 or z <= 0):
        raise ValueError("Fieller HAC inputs must be finite and risks positive")

    def signed_margin(ratio: float) -> float:
        variance = omega_cc - 2.0 * ratio * omega_cw + ratio * ratio * omega_ww
        tolerance = 1e-12 * max(1.0, abs(omega_cc), abs(omega_cw), abs(omega_ww))
        if variance < -tolerance:
            return math.nan
        se = math.sqrt(max(variance, 0.0) / n)
        return risk_comparator - ratio * risk_candidate - z * se

    if signed_margin(0.0) <= 0:
        return 0.0
    low = 0.0
    high = max(1.0, risk_comparator / risk_candidate)
    for _ in range(100):
        margin = signed_margin(high)
        if not math.isfinite(margin):
            raise ValueError("HAC contrast variance is invalid")
        if margin <= 0:
            break
        high *= 2.0
    else:
        raise ValueError("could not bracket the Fieller HAC lower endpoint")
    for _ in range(64):
        middle = (low + high) / 2.0
        margin = signed_margin(middle)
        if not math.isfinite(margin):
            raise ValueError("HAC contrast variance is invalid")
        if margin > 0:
            low = middle
        else:
            high = middle
    return low


def _panel_lowers(panel: np.ndarray, z: float) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {method: {} for method in ("ratio_delta", "fieller_contrast")}
    for length in D06_BLOCK_LENGTHS:
        risks, influences = d04_risk_and_influence(panel, length)
        for name, comparator_index in COMPARISONS:
            candidate = float(risks[0])
            comparator = float(risks[comparator_index])
            ratio = comparator / candidate
            ratio_if = (influences[:, comparator_index] - ratio * influences[:, 0]) / candidate
            for bandwidth in HAC_BANDWIDTHS:
                key = f"{name}_L{length}_b{bandwidth}"
                omega_ratio = newey_west_long_run_variance(ratio_if, bandwidth)
                if not math.isfinite(omega_ratio) or omega_ratio < 0:
                    raise ValueError("ratio HAC variance is invalid")
                result["ratio_delta"][key] = max(0.0, ratio - z * math.sqrt(omega_ratio / panel.shape[0]))
                omega_cc, omega_cw, omega_ww = _hac_contrast_coefficients(
                    influences[:, comparator_index], influences[:, 0], bandwidth
                )
                result["fieller_contrast"][key] = fieller_hac_lower(
                    comparator, candidate, omega_cc, omega_cw, omega_ww,
                    panel.shape[0], z,
                )
    return result


def _summarize(values: list[bool], planned: int) -> dict[str, Any]:
    valid = len(values)
    successes = sum(values)
    coverage = successes / valid if valid else None
    return {
        "planned": planned, "valid": valid, "failed": 0,
        "abstained": planned - valid,
        "successes": successes,
        "coverage": coverage,
        "wilson_95": _wilson_interval(successes, valid) if valid else None,
        "mc_standard_error": (
            math.sqrt(coverage * (1.0 - coverage) / valid) if valid else None
        ),
    }


def _plot(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    families = list(FAMILIES)
    colors = {"ratio_delta": "#3d85c6", "fieller_contrast": "#e69138"}
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
    for row_index, family in enumerate(families):
        for col_index, metric in enumerate(("family_coverage", "family_release_rate")):
            axis = axes[row_index, col_index]
            for method in ("ratio_delta", "fieller_contrast"):
                for phi in PHI_SCENARIOS:
                    for loading in COMMON_LOADINGS:
                        subset = [entry for entry in summary_rows
                                  if entry["family"] == family
                                  and entry["phi"] == phi
                                  and entry["common_loading"] == loading
                                  and entry["method"] == method]
                        subset.sort(key=lambda entry: entry["n_configurations"])
                        axis.plot(
                            [entry["n_configurations"] for entry in subset],
                            [entry[metric] for entry in subset],
                            color=colors[method],
                            alpha=0.38 + 0.18 * (loading > 0),
                            linestyle="-" if phi == 0.0 else "--" if phi == 0.4 else ":",
                            marker="o" if loading == 0 else "s",
                            label=(f"{method}, phi={phi:g}, common loading={loading:g}"
                                   if row_index == 0 and col_index == 0 else None),
                        )
            axis.axhline(0.95 if metric == "family_coverage" else 0.0,
                         color="#333333", linewidth=0.8, linestyle="-." if metric == "family_coverage" else ":")
            axis.set_ylim(-0.03, 1.03)
            axis.grid(True, alpha=0.25)
            axis.set_ylabel("fraction of panels")
            axis.set_title(f"{family}: {metric.replace('_', ' ')}")
            if row_index == 2:
                axis.set_xlabel("configuration count n")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("D06 HAC ratio bounds under synthetic dependence and tail stress", y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build(output: Path = DEFAULT_OUT, *, replicates: int = MC_REPLICATES,
          base_seed: int = BASE_SEED, n_scenarios: tuple[int, ...] = N_SCENARIOS,
          phi_scenarios: tuple[float, ...] = PHI_SCENARIOS,
          families: tuple[str, ...] = FAMILIES,
          common_loadings: tuple[float, ...] = COMMON_LOADINGS) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if replicates < 2:
        raise ValueError("coverage summary needs at least two independent panels")
    if not n_scenarios or any(n < 20 or any(n % length for length in D06_BLOCK_LENGTHS)
                              for n in n_scenarios):
        raise ValueError("all n scenarios must be >=20 and divisible by L=2,5,10")
    if any(family not in FAMILIES for family in families):
        raise ValueError("unknown DGP family")
    output.mkdir(parents=True)
    z = NormalDist().inv_cdf(1.0 - D06_ALPHA_PER_COMPARISON)
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(families):
        for n_index, n in enumerate(n_scenarios):
            for phi_index, phi in enumerate(phi_scenarios):
                for load_index, loading in enumerate(common_loadings):
                    cell_id = f"{family}_n{n}_phi{phi:g}_common{loading:g}"
                    true = true_risks(phi, n, family)
                    truth_ratios = {
                        name: true[ARMS[index]] / true[ARMS[0]]
                        for name, index in COMPARISONS
                    }
                    cell = {method: {name: [] for name, _ in COMPARISONS}
                            for method in ("ratio_delta", "fieller_contrast")}
                    cell_family_cover = {method: [] for method in cell}
                    cell_family_release = {method: [] for method in cell}
                    for replicate in range(replicates):
                        seed = (base_seed + family_index * 100_000_000
                                + n_index * 10_000_000 + phi_index * 1_000_000
                                + load_index * 100_000 + replicate)
                        panel = generate_panel(seed, phi, n, family, loading)
                        lowers = _panel_lowers(panel, z)
                        row: dict[str, Any] = {
                            "cell_id": cell_id, "family": family, "n_configurations": n,
                            "phi": phi, "common_loading": loading,
                            "replicate": replicate, "seed": seed,
                        }
                        for method in cell:
                            all_covered = []
                            all_release = []
                            for comparison, _ in COMPARISONS:
                                values = [value for key, value in lowers[method].items()
                                          if key.startswith(f"{comparison}_")]
                                if len(values) != len(D06_BLOCK_LENGTHS) * len(HAC_BANDWIDTHS):
                                    raise ValueError("incomplete method hull")
                                lower = min(values)
                                truth_ratio = truth_ratios[comparison]
                                covered = lower <= truth_ratio
                                released = lower > D06_R_STAR
                                cell[method][comparison].append(covered)
                                all_covered.append(covered)
                                all_release.append(released)
                                row[f"truth_{comparison}"] = truth_ratio
                                row[f"lower_{method}_{comparison}"] = lower
                                row[f"covered_{method}_{comparison}"] = int(covered)
                                row[f"clears_r_star_{method}_{comparison}"] = int(released)
                            family_covered = all(all_covered)
                            family_release = all(all_release)
                            cell_family_cover[method].append(family_covered)
                            cell_family_release[method].append(family_release)
                            row[f"family_covered_{method}"] = int(family_covered)
                            row[f"family_clears_r_star_{method}"] = int(family_release)
                        raw_rows.append(row)
                    for method in cell:
                        summary_rows.append({
                            "cell_id": cell_id, "family": family, "n_configurations": n,
                            "phi": phi, "common_loading": loading, "method": method,
                            "r_star": D06_R_STAR,
                            "marginal_SLSQP": _summarize(cell[method]["SLSQP"], replicates),
                            "marginal_Chebyshev": _summarize(cell[method]["Chebyshev"], replicates),
                            "family_coverage": _summarize(cell_family_cover[method], replicates)["coverage"],
                            "family_coverage_wilson_95": _summarize(cell_family_cover[method], replicates)["wilson_95"],
                            "family_release_rate": _summarize(cell_family_release[method], replicates)["coverage"],
                            "family_release_wilson_95": _summarize(cell_family_release[method], replicates)["wilson_95"],
                            "planned": replicates, "valid": replicates, "failed": 0, "abstained": 0,
                        })

    def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    write_csv("replicates.csv", raw_rows)
    summary_flat = []
    for row in summary_rows:
        summary_flat.append({
            key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        })
    write_csv("coverage_summary.csv", summary_flat)
    _plot(output / "coverage_release.png", summary_rows)
    worst = {}
    aggregate = {}
    for family in families:
        for method in ("ratio_delta", "fieller_contrast"):
            subset = [row for row in summary_rows
                      if row["family"] == family and row["method"] == method]
            worst[f"{family}:{method}"] = {
                "minimum_family_coverage": min(row["family_coverage"] for row in subset),
                "minimum_family_release_rate": min(row["family_release_rate"] for row in subset),
                "cells_with_wilson_lower_ge_0.95": sum(
                    row["family_coverage_wilson_95"][0] >= 0.95 for row in subset
                ),
                "cell_count": len(subset),
                "mean_family_coverage_over_cells": math.fsum(
                    row["family_coverage"] for row in subset
                ) / len(subset),
                "mean_family_release_rate_over_cells": math.fsum(
                    row["family_release_rate"] for row in subset
                ) / len(subset),
            }
    for method in ("ratio_delta", "fieller_contrast"):
        subset = [row for row in summary_rows if row["method"] == method]
        aggregate[method] = {
            "cells_with_wilson_lower_ge_0.95": sum(
                row["family_coverage_wilson_95"][0] >= 0.95 for row in subset
            ),
            "cell_count": len(subset),
            "equal_cell_mean_family_coverage": math.fsum(
                row["family_coverage"] for row in subset
            ) / len(subset),
            "equal_cell_mean_family_release_rate": math.fsum(
                row["family_release_rate"] for row in subset
            ) / len(subset),
        }
    aggregate_table = "\n".join(
        f"|{family}|{method}|{worst[f'{family}:{method}']['mean_family_coverage_over_cells']:.4f}|"
        f"{worst[f'{family}:{method}']['mean_family_release_rate_over_cells']:.4f}|"
        f"{worst[f'{family}:{method}']['cells_with_wilson_lower_ge_0.95']}/"
        f"{worst[f'{family}:{method}']['cell_count']}|"
        for family in families for method in ("ratio_delta", "fieller_contrast")
    )
    protocol = {
        "schema": "optimize-d06-hac-fieller-stress-v1",
        "evidence_class": "synthetic_known_risk_gaussian_t5_and_periodic_heteroskedastic_ar1",
        "status": "METHOD_STRESS_NOT_CALIBRATED_FOR_PHYSICAL_USE",
        "n_configuration_scenarios": list(n_scenarios),
        "phi_scenarios": list(phi_scenarios),
        "families": list(families),
        "common_noise_loadings": list(common_loadings),
        "common_arm_correlation": [value**2 for value in common_loadings],
        "mc_replicates_per_cell": replicates,
        "mc_standard_error_at_95pct_nominal": math.sqrt(0.95 * 0.05 / replicates),
        "one_sided_alpha_per_contrast": D06_ALPHA_PER_COMPARISON,
        "family_size": 2,
        "r_star": D06_R_STAR,
        "block_lengths": list(D06_BLOCK_LENGTHS),
        "hac_bandwidths": list(HAC_BANDWIDTHS),
        "methods": [
            "ratio_delta: HAC standard error of the plug-in risk-ratio influence function",
            "fieller_contrast: invert HAC-studentized R_comparator-r*R_trainableW contrasts; solve for the upper root",
        ],
        "truth": "exact finite-n expected D04 risk from declared means and covariance; Student-t innovations have finite variance and periodic heteroskedastic covariance is propagated exactly",
        "denominator": {
            "planned": len(raw_rows), "valid": len(raw_rows), "failed": 0, "abstained": 0,
            "unit": "independent synthetic panels; all method/contrast endpoints finite",
        },
        "physical_score": None,
        "worst_cell_summary": worst,
        "aggregate_cell_summary": aggregate,
        "limits": [
            "No lattice or sealed data, GPU, QUDA, production reference fit, or optimizer failures were modeled.",
            "HAC relies on a studentized weak-dependence approximation and is not a finite-sample theorem.",
            "This compares HAC ratio constructions; it does not select a production interval or replace the archived percentile method.",
        ],
        "source_hashes": {
            path.name: _sha256(path) for path in (
                Path(__file__), Path(__file__).with_name("d06_hac_coverage_diagnostic.py"),
                Path(__file__).with_name("d06_bootstrap_coverage_diagnostic.py"),
                Path(__file__).with_name("risk_statistics.py"),
            )
        },
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    readme = f"""# D06 HAC ratio interval stress: delta method vs contrast inversion

**CPU-only synthetic method stress.** It does not use gauge data, a production reference fit, sealed split, GPU, or QUDA. The physical D06 rule remains `UNCALIBRATED/NOT_READY`, with `physical_score=null`.

## Question and methods

Both methods form one-sided lower confidence bounds for `rho=R_comparator/R_trainableW` with per-comparison alpha `{D06_ALPHA_PER_COMPARISON}` and family size 2. `ratio_delta` applies HAC to the plug-in ratio influence. `fieller_contrast` directly inverts the HAC-studentized contrasts `R_comparator - r*R_trainableW`; it is a Fieller-style lower endpoint and avoids treating the denominator as fixed. Both take the minimum across block lengths `{list(D06_BLOCK_LENGTHS)}` and Bartlett bandwidths `{list(HAC_BANDWIDTHS)}`. Release additionally requires both family lower bounds to exceed `r_star={D06_R_STAR:.6f}`.

We used known finite-`n` D04 risk truth for Gaussian AR(1), finite-variance Student-t5 AR(1), and period-2 heteroskedastic AR(1) innovations. Each family spans `n={list(n_scenarios)}`, `phi={list(phi_scenarios)}`, and common-noise loading `{list(common_loadings)}` (arm correlation is loading squared), with `{replicates}` independent panels per cell. Expected D04 risk is computed from the exact declared mean and covariance, not an estimated Monte Carlo truth. The nominal-coverage MC standard error at 95% is `{math.sqrt(0.95*0.05/replicates):.4f}`.

![Coverage and release-rate stress](coverage_release.png)

## Findings and boundaries

`coverage_summary.csv` provides every cell and Wilson interval, and `replicates.csv` keeps paired panel outcomes. Across all 54 cells, ratio-delta's familywise Wilson lower endpoint reaches 0.95 in {aggregate['ratio_delta']['cells_with_wilson_lower_ge_0.95']}/54 cells; contrast inversion does so in {aggregate['fieller_contrast']['cells_with_wilson_lower_ge_0.95']}/54. Equal-cell averages below are descriptive summaries of this declared grid, not estimates for a lattice ensemble:

|DGP|method|mean family coverage|mean family release rate|cells with Wilson lower >=0.95|
|---|---|---:|---:|---:|
{aggregate_table}

Worst-cell minima and the full denominator by DGP/method are recorded in `protocol.json`:

```text
{json.dumps(worst, indent=2, sort_keys=True)}
```

The experimental unit denominator is planned/valid/failed/abstained = `{len(raw_rows)}/{len(raw_rows)}/0/0` independent synthetic panels. Numeric coverage is not calibration evidence for a lattice chain. HAC remains an asymptotic candidate; no method is promoted from this sweep. The stress does not include uncertainty in D04 reference fitting, nonstationary mean drift, optimizer failures, or the production candidate-generation process.

Text flow: `declared DGP and exact risk truth -> paired three-arm panel -> D04 risk/influence at L=2/5/10 -> delta-HAC vs inverted-HAC contrast lower bounds over fixed bandwidth hull -> family coverage and r_star release rate -> record undercoverage or conservative abstention; no physical score`.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    hashes = {path.name: _sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
    (output / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replicates", type=int, default=MC_REPLICATES)
    args = parser.parse_args()
    print(json.dumps(build(args.output, replicates=args.replicates), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
