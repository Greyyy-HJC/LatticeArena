"""Compare alternative synthetic intervals for the finite-T Creutz estimator."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tasks.wilson_loop import c_lane_block_bootstrap_coverage_stress as base

HAC_BANDWIDTHS = (5, 10, 20, 30)
STATIONARY_MEAN_BLOCKS = (5, 10, 20)
N_STATIONARY_DRAWS = 800
Z_CRIT = 2.241402727604947


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stationary_indices(n: int, mean_block: int, draws: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap with geometric block lengths."""
    idx = np.empty((draws, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=draws)
    p = 1.0 / mean_block
    for t in range(1, n):
        restart = rng.random(draws) < p
        fresh = rng.integers(0, n, size=draws)
        idx[:, t] = np.where(restart, fresh, (idx[:, t - 1] + 1) % n)
    return idx


def grad(means: np.ndarray) -> np.ndarray:
    return np.asarray([-1 / means[0], -1 / means[1],
                       1 / means[2], 1 / means[3]], dtype=np.float64)


def hac_interval(data: np.ndarray, bandwidth: int) -> dict:
    n = len(data)
    means = data.mean(axis=0)
    centered = data - means
    omega = centered.T @ centered / n
    max_lag = min(bandwidth, n - 1)
    for lag in range(1, max_lag + 1):
        gamma = centered[lag:].T @ centered[:-lag] / n
        omega += (1 - lag / (max_lag + 1)) * (gamma + gamma.T)
    variance = float(grad(means) @ omega @ grad(means) / n)
    point = float(base.creutz_from_means(means))
    if not math.isfinite(variance) or variance <= 0:
        return {"lower": None, "upper": None, "halfwidth": None,
                "planned_draws": 0, "valid_draws": 0, "invalid_draws": 0,
                "abstained": True}
    half = Z_CRIT * math.sqrt(variance)
    return {"lower": point-half, "upper": point+half, "halfwidth": half,
            "planned_draws": 0, "valid_draws": 0, "invalid_draws": 0,
            "abstained": False}


def oracle_interval(data: np.ndarray, rho: float, marginal_sigma: float) -> dict:
    """Delta interval with exact finite-n mean covariance under known DGP."""
    n = len(data)
    corr = np.full((4, 4), base.LOG_CORRELATION, dtype=np.float64)
    np.fill_diagonal(corr, 1.0)
    sigma = marginal_sigma * base.LOG_SIGMA_SHAPE
    log_cov = corr * np.outer(sigma, sigma)
    mean_outer = np.outer(base.LOOP_MEANS, base.LOOP_MEANS)
    cov_sum = n * mean_outer * np.expm1(log_cov)
    for lag in range(1, n):
        cov_sum += 2 * (n-lag) * mean_outer * np.expm1(log_cov * rho**lag)
    cov_mean = cov_sum / n**2
    variance = float(grad(base.LOOP_MEANS) @ cov_mean @ grad(base.LOOP_MEANS))
    point = float(base.creutz_from_means(data.mean(axis=0)))
    half = Z_CRIT * math.sqrt(max(variance, 0.0))
    return {"lower": point-half, "upper": point+half, "halfwidth": half,
            "planned_draws": 0, "valid_draws": 0, "invalid_draws": 0,
            "abstained": False}


def stationary_interval(data: np.ndarray, mean_block: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = stationary_indices(len(data), mean_block, N_STATIONARY_DRAWS, rng)
    estimates = base.creutz_from_means(data[idx].mean(axis=1))
    valid = np.isfinite(estimates)
    if not np.any(valid):
        return {"lower": None, "upper": None, "halfwidth": None,
                "planned_draws": N_STATIONARY_DRAWS, "valid_draws": 0,
                "invalid_draws": N_STATIONARY_DRAWS, "abstained": True}
    lo, hi = np.quantile(estimates[valid],
                         [base.ALPHA_LANE/2, 1-base.ALPHA_LANE/2], method="linear")
    return {"lower": float(lo), "upper": float(hi),
            "halfwidth": float((hi-lo)/2),
            "planned_draws": N_STATIONARY_DRAWS,
            "valid_draws": int(np.count_nonzero(valid)),
            "invalid_draws": int(N_STATIONARY_DRAWS-np.count_nonzero(valid)),
            "abstained": False}


def make_plot(rows: list[dict], output: Path) -> None:
    methods = (["known-rho covariance"] + [f"HAC bw={b}" for b in HAC_BANDWIDTHS]
               + [f"stationary mean-L={b}" for b in STATIONARY_MEAN_BLOCKS])
    colors = ["#111827", "#3B82F6", "#F97316", "#16A34A", "#7C3AED",
              "#DC2626", "#0891B2", "#A16207"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), constrained_layout=True)
    for i, sigma in enumerate(base.LOG_SIGMA_GRID):
        cov_ax, width_ax = axes[i]
        for method, color in zip(methods, colors):
            subset = sorted([r for r in rows if r["method"] == method
                             and r["log_loop_sigma"] == sigma], key=lambda r: r["rho"])
            xs = np.asarray([r["rho"] for r in subset])
            cov = np.asarray([r["coverage_fraction"] for r in subset])
            lo = np.asarray([r["coverage_ci95_lower"] for r in subset])
            hi = np.asarray([r["coverage_ci95_upper"] for r in subset])
            cov_ax.errorbar(xs, cov, yerr=[cov-lo, hi-cov], marker="o",
                            capsize=2, linewidth=1.2, color=color, label=method)
            width_ax.plot(xs, [r["median_halfwidth"] for r in subset], marker="o",
                          linewidth=1.2, color=color, label=method)
        cov_ax.axhline(1-base.ALPHA_LANE, color="#6B7280", linestyle="--",
                       label="nominal 97.5%")
        cov_ax.set(xlabel="AR(1) correlation", ylabel="empirical coverage",
                   ylim=(0.35, 1.01), title=f"Coverage, log-loop SD={sigma}")
        width_ax.axhline(base.HALFWIDTH_TARGET, color="#6B7280", linestyle="--",
                         label="half-width target 0.02")
        width_ax.set(xlabel="AR(1) correlation", ylabel="median interval half-width",
                     title=f"Precision, log-loop SD={sigma}")
        for ax in (cov_ax, width_ax):
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7.5)
    fig.suptitle("Alternative finite-T Creutz intervals (n=100)")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True)
    true_chi = float(base.creutz_from_means(base.LOOP_MEANS))
    methods = (["known-rho covariance"] + [f"HAC bw={b}" for b in HAC_BANDWIDTHS]
               + [f"stationary mean-L={b}" for b in STATIONARY_MEAN_BLOCKS])
    replicates = []
    for si, sigma in enumerate(base.LOG_SIGMA_GRID):
        for ri, rho in enumerate(base.RHO_GRID):
            for rep in range(base.N_REPLICATES):
                seed = base.SEED + 1_000_000*si + 100_000*ri + rep
                data = base.lognormal_ar1(rho, sigma, np.random.default_rng(seed))
                intervals = {"known-rho covariance": oracle_interval(data, rho, sigma)}
                intervals.update({f"HAC bw={b}": hac_interval(data, b)
                                  for b in HAC_BANDWIDTHS})
                for mean_block in STATIONARY_MEAN_BLOCKS:
                    boot_seed = (base.SEED + 40_000_000 + 1_000_000*si
                                 + 100_000*ri + 10_000*rep + mean_block)
                    intervals[f"stationary mean-L={mean_block}"] = stationary_interval(
                        data, mean_block, boot_seed)
                for method, interval in intervals.items():
                    valid = not interval["abstained"]
                    covered = bool(valid and interval["lower"] <= true_chi <= interval["upper"])
                    replicates.append({
                        "log_loop_sigma": sigma, "rho": rho, "replicate": rep,
                        "method": method, **interval, "true_chi": true_chi,
                        "covered": covered,
                        "precision_pass": bool(valid and interval["halfwidth"] <= base.HALFWIDTH_TARGET),
                    })

    summaries = []
    for sigma in base.LOG_SIGMA_GRID:
        for rho in base.RHO_GRID:
            for method in methods:
                rows = [r for r in replicates if r["log_loop_sigma"] == sigma
                        and r["rho"] == rho and r["method"] == method]
                valid = [r for r in rows if not r["abstained"]]
                covered = sum(bool(r["covered"]) for r in valid)
                lo, hi = base.wilson_interval(covered, len(valid))
                summaries.append({
                    "log_loop_sigma": sigma, "rho": rho, "method": method,
                    "planned_replicates": len(rows), "valid_replicates": len(valid),
                    "failed_replicates": 0, "abstained_replicates": len(rows)-len(valid),
                    "covered_replicates": covered,
                    "coverage_fraction": covered/len(valid) if valid else None,
                    "coverage_ci95_lower": lo, "coverage_ci95_upper": hi,
                    "precision_pass_replicates": sum(bool(r["precision_pass"]) for r in valid),
                    "median_halfwidth": float(np.median([r["halfwidth"] for r in valid])) if valid else None,
                    "median_invalid_draw_fraction": float(np.median([
                        r["invalid_draws"]/r["planned_draws"] if r["planned_draws"] else 0
                        for r in rows])),
                })
    for name, rows in [("coverage_summary.csv", summaries), ("replicates.csv", replicates)]:
        with (output/name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    this_source = Path(__file__).resolve()
    base_source = Path(base.__file__).resolve()
    shutil.copy2(this_source, output/"source_snapshot.py")
    shutil.copy2(base_source, output/"base_generator_snapshot.py")
    make_plot(summaries, output/"coverage_and_precision.png")
    protocol = {
        "schema": "wilson-c-lane-alternative-interval-coverage-v1",
        "classification": "synthetic_method_comparison_only", "physical_score": None,
        "data_access": "No Wilson measurement, development/sealed split, GPU or QUDA data read.",
        "base_protocol_sha256": sha256(Path("results/wilson_loop/c_lane_block_bootstrap_coverage_stress_20261125_v3/protocol.json")),
        "base_source_sha256": sha256(base_source),
        "known_rho_covariance_benchmark": "normal delta interval using exact finite-n mean covariance under known rho and lognormal scale; non-deployable benchmark",
        "hac": {"kernel": "Newey-West Bartlett", "bandwidths": HAC_BANDWIDTHS},
        "stationary_bootstrap": {"mean_block_lengths": STATIONARY_MEAN_BLOCKS,
                                 "draws": N_STATIONARY_DRAWS, "interval": "97.5% equal-tail percentile"},
        "outer_grid": {"rho": base.RHO_GRID, "log_loop_sd": base.LOG_SIGMA_GRID,
                       "n": base.N_CONFIG, "replicates_per_cell": base.N_REPLICATES},
        "loop_mean_order": ["W(3,6)", "W(2,5)", "W(3,5)", "W(2,6)"],
        "loop_arithmetic_means": base.LOOP_MEANS.tolist(), "true_chi": true_chi,
        "precision_halfwidth_target": base.HALFWIDTH_TARGET,
        "interpretation_limit": "Conditional synthetic coverage only; no estimate of actual HMC dependence.",
    }
    (output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n", encoding="utf-8")
    summary = {"schema": protocol["schema"], "classification": protocol["classification"],
               "physical_score": None, "planned_datasets": len(base.LOG_SIGMA_GRID)*len(base.RHO_GRID)*base.N_REPLICATES,
               "methods": methods, "coverage_rows": summaries, "data_access": protocol["data_access"]}
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    readme = f"""# Wilson C-lane alternative interval coverage comparison

**Conditional synthetic method comparison only.** physical_score is null. No Wilson, development or sealed data, GPU or QUDA input was read.

![Coverage and interval width for alternative methods](coverage_and_precision.png)

## Model and methods

This regenerates the same known-truth four-loop lognormal AR(1) datasets as the block-bootstrap stress report: n={base.N_CONFIG}, 240 datasets per rho/log-loop-SD cell, rho={list(base.RHO_GRID)}, log-loop SD={list(base.LOG_SIGMA_GRID)}, with true Creutz value {true_chi:.8f}. Shared outer seeds make results paired to the existing stress report.

The intervals are: a known-rho covariance benchmark using exact finite-n covariance and a normal delta interval; Newey-West/Bartlett HAC delta intervals at bandwidths {list(HAC_BANDWIDTHS)}; and stationary-bootstrap percentile intervals with mean block lengths {list(STATIONARY_MEAN_BLOCKS)} and {N_STATIONARY_DRAWS} draws. All use the 97.5% lane interval. The benchmark receives the true synthetic dependence and scale, so it is not deployable; it is also a delta approximation for the nonlinear Creutz estimator, not an exact finite-sample confidence interval.

## Denominators and limits

There are 2,400 planned datasets per method. coverage_summary.csv reports empirical coverage, Wilson Monte Carlo intervals, precision-pass counts and median widths. replicates.csv contains every dataset/method row and abstention/invalid-draw counts. The figure shows coverage and width against the 0.02 precision target.

This comparison tests these methods only inside the declared synthetic family. It does not identify physical dependence or certify production coverage. A method whose Monte Carlo interval excludes nominal 97.5% coverage in a dependence scenario judged plausible cannot enter the confirmation protocol without an independent dependence restriction or fail-closed abstention.

Text flow: fixed outer seeds -> positive loop means with known correlated-lognormal law -> known-rho covariance benchmark, HAC and stationary-bootstrap intervals -> paired method coverage table -> plot with complete denominators.

Rebuild into a fresh directory with python -m tasks.wilson_loop.c_lane_interval_alternative_coverage --output NEW_DIRECTORY. Both source snapshots and the artifact hash manifest are included.
"""
    (output/"README.md").write_text(readme, encoding="utf-8")
    files = ["README.md", "coverage_and_precision.png", "coverage_summary.csv", "replicates.csv",
             "protocol.json", "summary.json", "source_snapshot.py", "base_generator_snapshot.py"]
    hashes = {name: sha256(output/name) for name in files}
    (output/"artifact_hashes.json").write_text(
        json.dumps({"algorithm":"sha256","files":hashes}, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output":str(output), "planned_datasets":summary["planned_datasets"],
                      "coverage_rows":summaries}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("results/wilson_loop/c_lane_interval_alternative_coverage_20261125_v1"))
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
