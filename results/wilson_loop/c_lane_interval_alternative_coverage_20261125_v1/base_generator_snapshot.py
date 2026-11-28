"""Synthetic coverage stress for the finite-T Creutz circular-block interval."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEED = 20261125
N_CONFIG = 100
PRIMARY_BLOCK_LENGTHS = (2, 5, 10)
EXTENDED_BLOCK_LENGTHS = (2, 5, 10, 20, 30)
RHO_GRID = (0.0, 0.3, 0.6, 0.8, 0.9)
LOG_SIGMA_GRID = (0.10, 0.35)
N_REPLICATES = 240
N_BOOTSTRAP = 800
ALPHA_LANE = 0.025
HALFWIDTH_TARGET = 0.02
LOOP_MEANS = np.asarray([0.020, 0.060, 0.035, 0.040], dtype=np.float64)
LOG_SIGMA_SHAPE = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
LOG_CORRELATION = 0.75


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def circular_block_indices(n: int, length: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    blocks_per_draw = math.ceil(n / length)
    starts = rng.integers(0, n, size=(draws, blocks_per_draw))
    offsets = np.arange(length, dtype=np.int64)
    return ((starts[..., None] + offsets) % n).reshape(draws, -1)[:, :n]


def creutz_from_means(means: np.ndarray) -> np.ndarray:
    """Input loop order is W(3,6), W(2,5), W(3,5), W(2,6)."""
    return -np.log(means[..., 0]) - np.log(means[..., 1]) \
        + np.log(means[..., 2]) + np.log(means[..., 3])


def lognormal_ar1(rho: float, marginal_sigma: float, rng: np.random.Generator) -> np.ndarray:
    corr = np.full((4, 4), LOG_CORRELATION, dtype=np.float64)
    np.fill_diagonal(corr, 1.0)
    log_sigma = marginal_sigma * LOG_SIGMA_SHAPE
    covariance = corr * np.outer(log_sigma, log_sigma)
    innovations = rng.multivariate_normal(np.zeros(4), covariance, size=N_CONFIG)
    z = np.empty_like(innovations)
    z[0] = innovations[0]
    scale = math.sqrt(1.0 - rho * rho)
    for i in range(1, N_CONFIG):
        z[i] = rho * z[i - 1] + scale * innovations[i]
    log_location = np.log(LOOP_MEANS) - 0.5 * np.diag(covariance)
    return np.exp(log_location + z)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return center - half, center + half


def interval_for_length(data: np.ndarray, length: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    indices = circular_block_indices(len(data), length, N_BOOTSTRAP, rng)
    means = data[indices].mean(axis=1)
    estimates = creutz_from_means(means)
    valid = np.isfinite(estimates)
    lower, upper = np.quantile(
        estimates[valid], [ALPHA_LANE / 2.0, 1.0 - ALPHA_LANE / 2.0], method="linear"
    )
    return {
        "block_length": length,
        "lower": float(lower),
        "upper": float(upper),
        "halfwidth": float((upper - lower) / 2.0),
        "planned_draws": N_BOOTSTRAP,
        "valid_draws": int(np.count_nonzero(valid)),
        "invalid_draws": int(N_BOOTSTRAP - np.count_nonzero(valid)),
    }


def make_plot(rows: list[dict], path: Path) -> None:
    methods = ([f"L={x}" for x in EXTENDED_BLOCK_LENGTHS]
               + ["hull L=2/5/10", "hull L=2/5/10/20/30"])
    colors = ["#3B82F6", "#F97316", "#16A34A", "#A855F7", "#DC2626",
              "#111827", "#0F766E"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    for row_index, marginal_sigma in enumerate(LOG_SIGMA_GRID):
        coverage_ax, width_ax = axes[row_index]
        for method, color in zip(methods, colors):
            subset = [r for r in rows if r["method"] == method
                      and r["log_loop_sigma"] == marginal_sigma]
            subset.sort(key=lambda r: r["rho"])
            xs = np.asarray([r["rho"] for r in subset])
            coverage = np.asarray([r["coverage_fraction"] for r in subset])
            lo = np.asarray([r["coverage_ci95_lower"] for r in subset])
            hi = np.asarray([r["coverage_ci95_upper"] for r in subset])
            coverage_ax.errorbar(xs, coverage, yerr=[coverage - lo, hi - coverage], marker="o",
                                 capsize=2, linewidth=1.4, color=color, label=method)
            widths = np.asarray([r["median_halfwidth"] for r in subset])
            width_ax.plot(xs, widths, marker="o", linewidth=1.4, color=color, label=method)
        coverage_ax.axhline(1.0 - ALPHA_LANE, color="#6B7280", linestyle="--", linewidth=1.0,
                            label="nominal 97.5%")
        coverage_ax.set(xlabel="AR(1) correlation", ylabel="empirical coverage",
                        ylim=(0.35, 1.01), title=f"Coverage, log-loop SD={marginal_sigma} (Wilson 95% MC intervals)")
        width_ax.axhline(HALFWIDTH_TARGET, color="#6B7280", linestyle="--", linewidth=1.0,
                         label="numeric half-width target 0.02")
        width_ax.set(xlabel="AR(1) correlation", ylabel="median interval half-width",
                     title=f"Precision, log-loop SD={marginal_sigma}")
        for ax in (coverage_ax, width_ax):
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("Synthetic finite-T Creutz block-bootstrap stress (n=100)")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    rep_rows = []
    true_chi = float(creutz_from_means(LOOP_MEANS))
    for sigma_index, marginal_sigma in enumerate(LOG_SIGMA_GRID):
      for rho_index, rho in enumerate(RHO_GRID):
        for replicate in range(N_REPLICATES):
            data_rng = np.random.default_rng(SEED + 1_000_000 * sigma_index
                                             + 100_000 * rho_index + replicate)
            data = lognormal_ar1(rho, marginal_sigma, data_rng)
            intervals = [interval_for_length(data, length,
                          SEED + 10_000_000 * sigma_index + 1_000_000 * rho_index
                          + 10_000 * replicate + length)
                         for length in EXTENDED_BLOCK_LENGTHS]
            for item in intervals:
                rep_rows.append({
                    "rho": rho, "log_loop_sigma": marginal_sigma,
                    "replicate": replicate, "method": f"L={item['block_length']}",
                    **item, "true_chi": true_chi,
                    "covered": item["lower"] <= true_chi <= item["upper"],
                    "precision_pass": item["halfwidth"] <= HALFWIDTH_TARGET,
                })
            for label, lengths in [
                ("hull L=2/5/10", PRIMARY_BLOCK_LENGTHS),
                ("hull L=2/5/10/20/30", EXTENDED_BLOCK_LENGTHS),
            ]:
                selected = [x for x in intervals if x["block_length"] in lengths
                            and x["valid_draws"] > 0]
                lower = min(x["lower"] for x in selected)
                upper = max(x["upper"] for x in selected)
                halfwidth = (upper - lower) / 2.0
                planned_draws = sum(x["planned_draws"] for x in intervals
                                    if x["block_length"] in lengths)
                valid_draws = sum(x["valid_draws"] for x in intervals
                                  if x["block_length"] in lengths)
                invalid_draws = planned_draws - valid_draws
                rep_rows.append({
                    "rho": rho, "log_loop_sigma": marginal_sigma,
                    "replicate": replicate, "method": label,
                    "block_length": "hull", "lower": lower, "upper": upper,
                    "halfwidth": halfwidth, "planned_draws": planned_draws,
                    "valid_draws": valid_draws, "invalid_draws": invalid_draws,
                    "true_chi": true_chi, "covered": lower <= true_chi <= upper,
                    "precision_pass": halfwidth <= HALFWIDTH_TARGET,
                })

    summary_rows = []
    for marginal_sigma in LOG_SIGMA_GRID:
      for rho in RHO_GRID:
        for method in ([f"L={x}" for x in EXTENDED_BLOCK_LENGTHS]
                       + ["hull L=2/5/10", "hull L=2/5/10/20/30"]):
            subset = [r for r in rep_rows if r["rho"] == rho
                      and r["log_loop_sigma"] == marginal_sigma and r["method"] == method]
            successes = sum(bool(r["covered"]) for r in subset)
            lo, hi = wilson_interval(successes, len(subset))
            summary_rows.append({
                "rho": rho, "log_loop_sigma": marginal_sigma, "method": method,
                "planned_replicates": N_REPLICATES, "valid_replicates": len(subset),
                "failed_replicates": 0, "abstained_replicates": 0,
                "covered_replicates": successes,
                "coverage_fraction": successes / len(subset),
                "coverage_ci95_lower": lo, "coverage_ci95_upper": hi,
                "precision_pass_replicates": sum(bool(r["precision_pass"]) for r in subset),
                "median_halfwidth": float(np.median([r["halfwidth"] for r in subset])),
                "median_invalid_draw_fraction": float(np.median([
                    r["invalid_draws"] / r["planned_draws"] for r in subset
                ])),
            })

    import csv
    for name, rows in [("coverage_summary.csv", summary_rows), ("replicates.csv", rep_rows)]:
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    source_path = Path(__file__).resolve()
    shutil.copy2(source_path, output / "source_snapshot.py")
    make_plot(summary_rows, output / "coverage_and_precision.png")
    protocol = {
        "schema": "wilson-c-lane-block-bootstrap-coverage-stress-v1",
        "classification": "synthetic_method_calibration_only",
        "physical_score": None,
        "data_access": "No Wilson measurement, development split, sealed split, GPU, or QUDA data read.",
        "estimator": "absolute finite-T Creutz -log(mean W36)-log(mean W25)+log(mean W35)+log(mean W26)",
        "sampling_model": "four-variate stationary Gaussian AR(1) log-loops; exponentiated to positive loop observations",
        "ar1_rho_grid": RHO_GRID,
        "log_loop_marginal_sd_grid": LOG_SIGMA_GRID,
        "n_configurations": N_CONFIG,
        "planned_replicates_per_rho": N_REPLICATES,
        "bootstrap_draws_per_length": N_BOOTSTRAP,
        "primary_block_lengths": PRIMARY_BLOCK_LENGTHS,
        "extended_sensitivity_block_lengths": EXTENDED_BLOCK_LENGTHS,
        "interval": "equal-tail percentile interval at lane alpha 0.025; quantiles 0.0125 and 0.9875; compare primary L=2/5/10 hull with extended L=2/5/10/20/30 hull",
        "true_loop_arithmetic_means_order": ["W(3,6)", "W(2,5)", "W(3,5)", "W(2,6)"],
        "true_loop_arithmetic_means": LOOP_MEANS.tolist(),
        "log_loop_marginal_sd": "each of four log loops uses the selected grid value",
        "log_loop_cross_correlation": LOG_CORRELATION,
        "true_chi": true_chi,
        "precision_halfwidth_target": HALFWIDTH_TARGET,
        "seeds": {"base": SEED, "data_rule": "base + 1000000*sigma_index + 100000*rho_index + replicate",
                  "bootstrap_rule": "base + 10000000*sigma_index + 1000000*rho_index + 10000*replicate + block_length"},
        "interpretation_limit": "Coverage is conditional on this declared AR(1) lognormal family; it does not establish physical dependence or production coverage.",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema": protocol["schema"], "classification": protocol["classification"],
        "physical_score": None, "rho_levels": len(RHO_GRID),
        "noise_levels": len(LOG_SIGMA_GRID),
        "planned_replicates": len(LOG_SIGMA_GRID) * len(RHO_GRID) * N_REPLICATES,
        "denominators_by_scenario": {
            "planned": N_REPLICATES, "valid": N_REPLICATES, "failed": 0, "abstained": 0
        },
        "true_chi": true_chi,
        "coverage_rows": summary_rows,
        "data_access": protocol["data_access"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    readme = f"""# Wilson C-lane synthetic block-bootstrap coverage stress

**Status: conditional synthetic-method diagnostic only.** `physical_score=null`; this run reads no Wilson measurements, development or sealed split, GPU, or QUDA input.

## Design

The estimand is the absolute finite-$T$ four-loop Creutz combination. Synthetic positive loop observations are generated by exponentiating a four-variate Gaussian AR(1) process with adjacent-configuration correlations `rho={list(RHO_GRID)}`, `n={N_CONFIG}`, marginal log standard deviations `{list(LOG_SIGMA_GRID)}`, and cross-loop log correlation `{LOG_CORRELATION}`. The arithmetic loop means and hence the true Creutz value are known analytically (`{true_chi:.8f}`). Each of {N_REPLICATES} independently generated datasets per `(rho, log-loop SD)` cell is analyzed with circular moving-block bootstrap lengths 2, 5 and 10, {N_BOOTSTRAP} draws per length, 97.5% equal-tail percentile intervals and their hull. Precision is separately summarized against half-width 0.02.

![Empirical interval coverage and precision by dependence strength](coverage_and_precision.png)

## Denominators and interpretation

There are {len(LOG_SIGMA_GRID)*len(RHO_GRID)*N_REPLICATES} planned synthetic datasets (`{len(RHO_GRID)*N_REPLICATES}` per marginal log-loop SD), with `planned/valid/failed/abstained = {len(LOG_SIGMA_GRID)*len(RHO_GRID)*N_REPLICATES}/{len(LOG_SIGMA_GRID)*len(RHO_GRID)*N_REPLICATES}/0/0`. Each row in `coverage_summary.csv` gives empirical coverage, a Wilson 95% Monte Carlo interval, median half-width and precision-pass count; `replicates.csv` retains every dataset/interval result and invalid-draw denominator.

This checks whether the selected L=2/5/10 interval hull has nominal coverage inside one explicit AR(1) lognormal family at the current n=100 scale, and whether adding L=20/30 is sufficient. Results do not identify the actual HMC dependence, calibrate physical coverage, or justify opening confirmation data. If even the extended hull undercovers, the C-lane rule needs a defensible dependence bound or abstention; simply increasing bootstrap draws cannot fix the missing dependence information.

Text flow: `known positive loop means + correlated lognormal AR(1) generator -> 240 independent datasets per (rho, log-loop SD) -> paired four-loop resamples at L=2/5/10/20/30 -> primary and extended interval hulls -> empirical coverage/precision plot + full planned denominator`.

Rebuild into a fresh directory with `python -m tasks.wilson_loop.c_lane_block_bootstrap_coverage_stress --output NEW_DIRECTORY`. The exact source is copied to `source_snapshot.py`; hashes are listed in `artifact_hashes.json`.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    payload_files = ["README.md", "coverage_and_precision.png", "coverage_summary.csv",
                     "replicates.csv", "protocol.json", "summary.json", "source_snapshot.py"]
    manifest = {name: sha256(output / name) for name in payload_files}
    (output / "artifact_hashes.json").write_text(
        json.dumps({"algorithm": "sha256", "files": manifest}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "true_chi": true_chi,
                      "planned_datasets": len(LOG_SIGMA_GRID) * len(RHO_GRID) * N_REPLICATES,
                      "rows": summary_rows}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("results/wilson_loop/c_lane_block_bootstrap_coverage_stress_20261125_v3"))
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
