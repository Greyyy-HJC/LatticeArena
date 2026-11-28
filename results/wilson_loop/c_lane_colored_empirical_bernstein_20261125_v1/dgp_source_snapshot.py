"""Stress-test a bounded finite-dependence Creutz interval against block bootstrap.

No Wilson measurements are read. Synthetic observations are generated from a
four-loop Gaussian-copula Beta process with exact finite dependence range m.
"""
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
from scipy.stats import beta as beta_distribution
from scipy.stats import norm


SEED = 20261125
N_CONFIG = 100
DEPENDENCE_RANGES = (0, 5, 10)
LATENT_CROSS_CORRELATIONS = (0.0, 0.75)
BETA_CONCENTRATIONS = (5.0, 50.0)
N_REPLICATES = 250
N_BOOTSTRAP = 500
BOOTSTRAP_BLOCK_LENGTHS = (2, 5, 10)
LANE_ALPHA = 0.025
INVALID_DRAW_LIMIT = 0.01
HALFWIDTH_TARGET = 0.02
LOOP_MEANS = np.asarray([0.020, 0.060, 0.035, 0.040], dtype=np.float64)
LOOP_ORDER = ("W(3,6)", "W(2,5)", "W(3,5)", "W(2,6)")
LOOP_SUPPORT_LOWER = -1.0 / 3.0
LOOP_SUPPORT_UPPER = 1.0
LOOP_SUPPORT_WIDTH = LOOP_SUPPORT_UPPER - LOOP_SUPPORT_LOWER


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def creutz_from_means(means: np.ndarray) -> np.ndarray:
    means = np.asarray(means, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (-np.log(means[..., 0]) - np.log(means[..., 1])
                  + np.log(means[..., 2]) + np.log(means[..., 3]))
    return np.where(np.all(means > 0, axis=-1), result, np.nan)


def m_dependent_beta_series(n: int, m: int, cross_corr: float,
                            concentration: float,
                            rng: np.random.Generator) -> np.ndarray:
    """Stationary m-dependent Gaussian-copula Beta series in [0,1]^4."""
    corr = np.full((4, 4), cross_corr, dtype=np.float64)
    np.fill_diagonal(corr, 1.0)
    # Each row is a four-loop vector innovation. A moving sum of m+1 vectors
    # gives exact independence for temporal separations greater than m.
    innovations = rng.multivariate_normal(np.zeros(4), corr, size=n + m)
    latent = np.empty((n, 4), dtype=np.float64)
    scale = math.sqrt(m + 1)
    for i in range(n):
        latent[i] = innovations[i:i + m + 1].sum(axis=0) / scale
    u = np.clip(norm.cdf(latent), np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    beta_means = (LOOP_MEANS - LOOP_SUPPORT_LOWER) / LOOP_SUPPORT_WIDTH
    a = beta_means * concentration
    b = (1.0 - beta_means) * concentration
    beta_values = np.column_stack([
        beta_distribution.ppf(u[:, j], a[j], b[j]) for j in range(4)
    ])
    return LOOP_SUPPORT_LOWER + LOOP_SUPPORT_WIDTH * beta_values


def circular_block_indices(n: int, length: int, draws: int,
                           rng: np.random.Generator) -> np.ndarray:
    blocks = math.ceil(n / length)
    starts = rng.integers(0, n, size=(draws, blocks))
    offsets = np.arange(length, dtype=np.int64)
    return ((starts[..., None] + offsets) % n).reshape(draws, -1)[:, :n]


def block_hull_interval(data: np.ndarray, seed: int) -> dict:
    intervals = []
    planned = valid = 0
    invalid_fractions = []
    for length in BOOTSTRAP_BLOCK_LENGTHS:
        rng = np.random.default_rng(seed + length)
        indices = circular_block_indices(len(data), length, N_BOOTSTRAP, rng)
        means = data[indices].mean(axis=1)
        estimates = creutz_from_means(means)
        good = np.isfinite(estimates)
        planned += N_BOOTSTRAP
        valid += int(np.count_nonzero(good))
        invalid_fractions.append(1.0 - float(np.mean(good)))
        if np.any(good):
            intervals.append(np.quantile(
                estimates[good], [LANE_ALPHA / 2, 1 - LANE_ALPHA / 2], method="linear"
            ))
    if not intervals:
        return {"lower": None, "upper": None, "halfwidth": None,
                "planned_draws": planned, "valid_draws": valid,
                "invalid_draws": planned - valid, "computed": False,
                "max_invalid_draw_fraction": max(invalid_fractions, default=1.0),
                "invalid_draw_gate_pass": False}
    lower = float(min(x[0] for x in intervals))
    upper = float(max(x[1] for x in intervals))
    return {"lower": lower, "upper": upper, "halfwidth": (upper - lower) / 2,
            "planned_draws": planned, "valid_draws": valid,
            "invalid_draws": planned - valid, "computed": True,
            "max_invalid_draw_fraction": max(invalid_fractions, default=1.0),
            "invalid_draw_gate_pass": max(invalid_fractions, default=1.0) <= INVALID_DRAW_LIMIT}


def finite_dependence_hoeffding(data: np.ndarray, m: int) -> dict:
    """Bonferroni simultaneous loop-mean intervals, propagated through logs.

    A length-(m+1) interval graph is a valid dependency graph for the moving
    average construction. It can be colored with chi=m+1 independent sets.
    Four means and two tails split the lane alpha into eight Hoeffding events.
    """
    n = len(data)
    chi = m + 1
    alpha_cell = LANE_ALPHA / 8.0
    radius = LOOP_SUPPORT_WIDTH * math.sqrt(
        chi * math.log(2.0 / alpha_cell) / (2.0 * n)
    )
    means = data.mean(axis=0)
    lower_means = np.maximum(0.0, means - radius)
    upper_means = np.minimum(LOOP_SUPPORT_UPPER, means + radius)
    if np.any(upper_means <= 0.0):
        return {"lower": None, "upper": None, "halfwidth": None,
                "mean_radius": radius, "chromatic_number_bound": chi,
                "alpha_per_one_sided_loop_event": alpha_cell,
                "computed": False}
    # Monotonicity of -log for the first two terms and +log for the last two.
    lower = (float("-inf") if np.any(lower_means[2:] <= 0.0) else
             -math.log(upper_means[0]) - math.log(upper_means[1])
             + math.log(lower_means[2]) + math.log(lower_means[3]))
    upper = (float("inf") if np.any(lower_means[:2] <= 0.0) else
             -math.log(lower_means[0]) - math.log(lower_means[1])
             + math.log(upper_means[2]) + math.log(upper_means[3]))
    halfwidth = ((upper - lower) / 2.0
                 if math.isfinite(lower) and math.isfinite(upper) else float("inf"))
    return {"lower": lower, "upper": upper, "halfwidth": halfwidth,
            "mean_radius": radius, "chromatic_number_bound": chi,
            "alpha_per_one_sided_loop_event": alpha_cell,
            "computed": True}


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z*z/total
    center = (p + z*z/(2*total))/denom
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denom
    return center-half, center+half


def target_radius_for_halfwidth(target: float) -> float:
    """Solve the optimistic population-centered log-box width equation."""
    lo, hi = 0.0, float(np.min(LOOP_MEANS)) * (1 - 1e-12)
    for _ in range(120):
        mid = (lo + hi) / 2
        width = 0.5 * sum(math.log((mu + mid) / (mu - mid)) for mu in LOOP_MEANS)
        if width > target:
            hi = mid
        else:
            lo = mid
    return lo


def make_coverage_plot(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    for row, concentration in enumerate(BETA_CONCENTRATIONS):
        for col, cross_corr in enumerate(LATENT_CROSS_CORRELATIONS):
            ax = axes[row, col]
            subset = [r for r in rows if r["beta_concentration"] == concentration
                      and r["latent_cross_correlation"] == cross_corr]
            subset.sort(key=lambda r: r["dependence_range_m"])
            x = sorted({r["dependence_range_m"] for r in subset})
            block_rows = [next(r for r in subset if r["dependence_range_m"] == m
                               and r["method"] == "block_hull_L2_5_10") for m in x]
            graph_rows = [next(r for r in subset if r["dependence_range_m"] == m
                               and r["method"] == "finite_dependence_Hoeffding") for m in x]
            cov = np.asarray([r["coverage_fraction"] for r in block_rows])
            low = np.asarray([r["coverage_mc95_low"] for r in block_rows])
            high = np.asarray([r["coverage_mc95_high"] for r in block_rows])
            ax.errorbar(x, cov, yerr=[np.maximum(cov-low, 0), np.maximum(high-cov, 0)],
                        marker="o", capsize=3,
                        linewidth=1.5, color="#2563EB", label="block hull L=2/5/10")
            graph_cov = np.asarray([r["coverage_fraction"] for r in graph_rows])
            g_low = np.asarray([r["coverage_mc95_low"] for r in graph_rows])
            g_high = np.asarray([r["coverage_mc95_high"] for r in graph_rows])
            ax.errorbar(x, graph_cov,
                        yerr=[np.maximum(graph_cov-g_low, 0), np.maximum(g_high-graph_cov, 0)],
                        marker="s", capsize=3, linewidth=1.4, color="#B91C1C",
                        label="finite-dependence Hoeffding")
            ax.axhline(1-LANE_ALPHA, color="#6B7280", linestyle="--", linewidth=1,
                       label="nominal 97.5%")
            ax.set(xlabel="true dependence range m", ylabel="empirical interval coverage",
                   ylim=(0.65, 1.015), title=f"Beta concentration={concentration:g}, latent cross-corr={cross_corr:g}")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("Wilson C-lane synthetic finite-dependence coverage (n=100)")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_precision_plot(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    for row, concentration in enumerate(BETA_CONCENTRATIONS):
        for col, cross_corr in enumerate(LATENT_CROSS_CORRELATIONS):
            ax = axes[row, col]
            subset = [r for r in rows if r["beta_concentration"] == concentration
                      and r["latent_cross_correlation"] == cross_corr]
            subset.sort(key=lambda r: r["dependence_range_m"])
            x = np.asarray(sorted({r["dependence_range_m"] for r in subset}))
            block_rows = [next(r for r in subset if r["dependence_range_m"] == m
                               and r["method"] == "block_hull_L2_5_10") for m in x]
            graph_rows = [next(r for r in subset if r["dependence_range_m"] == m
                               and r["method"] == "finite_dependence_Hoeffding") for m in x]
            bw = np.asarray([r["median_halfwidth"] for r in block_rows])
            bb = np.asarray([r["precision_pass_fraction"] for r in block_rows])
            gb = np.asarray([r["precision_pass_fraction"] for r in graph_rows])
            ax.plot(x, bw, marker="o", color="#2563EB", label="block hull median half-width")
            ax.axhline(HALFWIDTH_TARGET, color="#6B7280", linestyle="--",
                       label="precision target 0.02")
            ax.set(xlabel="true dependence range m", ylabel="Creutz interval half-width",
                   title=f"Beta concentration={concentration:g}, latent cross-corr={cross_corr:g}")
            ax.grid(alpha=0.2)
            second = ax.twinx()
            second.plot(x, bb, marker="s", linestyle=":", color="#0F766E",
                        label="block-hull release fraction")
            second.plot(x, gb, marker="^", linestyle="--", color="#B91C1C",
                        label="Hoeffding release fraction")
            second.set_ylabel("fraction passing precision gate", color="#374151")
            second.set_ylim(-0.03, 1.03)
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = second.get_legend_handles_labels()
            ax.legend(h1+h2, l1+l2, fontsize=7.5, loc="upper left")
    fig.suptitle("Coverage-precision tradeoff; precision gate half-width ≤ 0.02")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_sample_size_plot(rows: list[dict], path: Path) -> None:
    rows = sorted(rows, key=lambda r: r["dependence_range_m"])
    x = np.asarray([r["dependence_range_m"] for r in rows])
    y = np.asarray([r["optimistic_n_for_halfwidth_target"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.plot(x, y, marker="o", color="#7C3AED")
    ax.set_yscale("log")
    ax.set(xlabel="assumed true dependence range m", ylabel="required configurations (log scale)",
           title="Optimistic finite-dependence Hoeffding planning bound")
    ax.grid(alpha=0.25, which="both")
    for index, (xi, yi) in enumerate(zip(x, y)):
        offset = (4, 7) if index % 2 == 0 else (4, -12)
        ax.annotate(f"{yi:,.0f}", (xi, yi), xytext=offset,
                    textcoords="offset points", fontsize=7.5)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    true_chi = float(creutz_from_means(LOOP_MEANS))
    replicate_rows: list[dict] = []
    for k_idx, concentration in enumerate(BETA_CONCENTRATIONS):
        for c_idx, cross_corr in enumerate(LATENT_CROSS_CORRELATIONS):
            for m_idx, m in enumerate(DEPENDENCE_RANGES):
                for rep in range(N_REPLICATES):
                    seed = SEED + 10_000_000*k_idx + 1_000_000*c_idx + 10_000*m_idx + rep
                    data = m_dependent_beta_series(
                        N_CONFIG, m, cross_corr, concentration, np.random.default_rng(seed)
                    )
                    boot = block_hull_interval(data, seed + 50_000_000)
                    finite = finite_dependence_hoeffding(data, m)
                    for name, interval in (("block_hull_L2_5_10", boot),
                                           ("finite_dependence_Hoeffding", finite)):
                        valid = bool(interval["computed"])
                        covered = bool(valid and interval["lower"] <= true_chi <= interval["upper"])
                        halfwidth = interval["halfwidth"]
                        invalid_gate_pass = bool(interval.get("invalid_draw_gate_pass", True))
                        precision_pass = bool(valid and invalid_gate_pass
                                              and math.isfinite(halfwidth)
                                              and halfwidth <= HALFWIDTH_TARGET)
                        invalid_gate_abstain = bool(valid and not invalid_gate_pass)
                        precision_gate_abstain = bool(valid and invalid_gate_pass
                                                      and not precision_pass)
                        replicate_rows.append({
                            "beta_concentration": concentration,
                            "latent_cross_correlation": cross_corr,
                            "dependence_range_m": m,
                            "replicate": rep,
                            "method": name,
                            "planned_replicates": 1,
                            "interval_computed": valid,
                            "failed": not valid,
                            "abstained": bool(valid and not precision_pass),
                            "invalid_draw_gate_abstained": invalid_gate_abstain,
                            "precision_gate_abstained": precision_gate_abstain,
                            "lower": interval["lower"], "upper": interval["upper"],
                            "halfwidth": (halfwidth if halfwidth is not None
                                          and math.isfinite(halfwidth) else None),
                            "unbounded_interval": bool(halfwidth is not None
                                                       and not math.isfinite(halfwidth)),
                            "covered": covered,
                            "precision_pass": precision_pass,
                            "planned_bootstrap_draws": interval.get("planned_draws", 0),
                            "valid_bootstrap_draws": interval.get("valid_draws", 0),
                            "invalid_bootstrap_draws": interval.get("invalid_draws", 0),
                            "max_invalid_draw_fraction": interval.get("max_invalid_draw_fraction", 0.0),
                            "invalid_draw_gate_pass": invalid_gate_pass,
                            "mean_radius": interval.get("mean_radius"),
                            "chromatic_number_bound": interval.get("chromatic_number_bound"),
                            "true_chi": true_chi,
                        })

    summary_rows: list[dict] = []
    for concentration in BETA_CONCENTRATIONS:
        for cross_corr in LATENT_CROSS_CORRELATIONS:
            for m in DEPENDENCE_RANGES:
                for method in ("block_hull_L2_5_10", "finite_dependence_Hoeffding"):
                    rows = [r for r in replicate_rows
                            if r["beta_concentration"] == concentration
                            and r["latent_cross_correlation"] == cross_corr
                            and r["dependence_range_m"] == m
                            and r["method"] == method]
                    computed = [r for r in rows if r["interval_computed"]]
                    covered = sum(r["covered"] for r in computed)
                    cov_lo, cov_hi = wilson_interval(covered, len(computed))
                    precision = sum(r["precision_pass"] for r in computed)
                    widths = [r["halfwidth"] for r in computed
                              if r["halfwidth"] is not None and math.isfinite(r["halfwidth"])]
                    summary_rows.append({
                        "beta_concentration": concentration,
                        "latent_cross_correlation": cross_corr,
                        "dependence_range_m": m,
                        "method": method,
                        "planned": len(rows), "interval_computed": len(computed),
                        "failed": sum(r["failed"] for r in rows),
                        "abstained": sum(r["abstained"] for r in rows),
                        "invalid_draw_gate_abstained": sum(r["invalid_draw_gate_abstained"] for r in rows),
                        "precision_gate_abstained": sum(r["precision_gate_abstained"] for r in rows),
                        "valid_release": precision,
                        "covered_intervals": covered,
                        "coverage_fraction": covered / len(computed) if computed else None,
                        "coverage_mc95_low": cov_lo, "coverage_mc95_high": cov_hi,
                        "precision_pass_fraction": precision / len(computed) if computed else None,
                        "median_halfwidth": float(np.median(widths)) if widths else None,
                        "unbounded_interval_count": sum(r["unbounded_interval"] for r in computed),
                        "median_max_invalid_draw_fraction": float(np.median([
                            r["max_invalid_draw_fraction"] for r in rows])),
                        "median_invalid_bootstrap_fraction": float(np.median([
                            r["invalid_bootstrap_draws"] / r["planned_bootstrap_draws"]
                            if r["planned_bootstrap_draws"] else 0.0 for r in rows])),
                    })

    # A population-centered planning calculation; it uses the known synthetic
    # means and is therefore optimistic, not a data-dependent sample-size claim.
    radius_target = target_radius_for_halfwidth(HALFWIDTH_TARGET)
    event_alpha = LANE_ALPHA / 8.0
    log_factor = math.log(2.0 / event_alpha)
    sample_rows = []
    for m in range(0, 11):
        chi = m + 1
        n_required = math.ceil(
            LOOP_SUPPORT_WIDTH**2 * chi * log_factor / (2.0 * radius_target**2)
        )
        sample_rows.append({
            "dependence_range_m": m,
            "chromatic_number_bound": chi,
            "alpha_per_one_sided_loop_event": event_alpha,
            "optimistic_required_mean_radius": radius_target,
            "optimistic_n_for_halfwidth_target": n_required,
            "assumption": "sample means equal the known synthetic population means; actual precision may require more",
        })

    for name, rows in (("coverage_summary.csv", summary_rows),
                       ("replicates.csv", replicate_rows),
                       ("sample_size_planning.csv", sample_rows)):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    source = Path(__file__).resolve()
    shutil.copy2(source, output / "source_snapshot.py")
    make_coverage_plot(summary_rows, output / "coverage.png")
    make_precision_plot(summary_rows, output / "precision.png")
    make_sample_size_plot(sample_rows, output / "sample_size.png")
    protocol = {
        "schema": "wilson-c-lane-finite-dependence-stress-v1",
        "classification": "synthetic_conditional_method_comparison_only",
        "physical_score": None,
        "data_access": "No Wilson measurements, development/sealed split, GPU or QUDA data read.",
        "estimator": "absolute finite-T Creutz -log(mean W36)-log(mean W25)+log(mean W35)+log(mean W26)",
        "true_loop_means_order": LOOP_ORDER,
        "true_loop_means": LOOP_MEANS.tolist(),
        "true_chi": true_chi,
        "observation_support": {"lower": LOOP_SUPPORT_LOWER, "upper": LOOP_SUPPORT_UPPER,
                                "basis": "real normalized SU(3) trace lies in [-1/3,1]; the three-direction average preserves this support"},
        "marginal_family": "affinely transformed Beta(mean=mu_j, concentration=kappa)",
        "concentrations": BETA_CONCENTRATIONS,
        "temporal_process": "stationary moving-average of m+1 iid Gaussian vectors, followed by coordinatewise Gaussian-copula Beta transform; exact m-dependence",
        "dependence_ranges": DEPENDENCE_RANGES,
        "latent_cross_correlations": LATENT_CROSS_CORRELATIONS,
        "n_configurations": N_CONFIG,
        "replicates_per_cell": N_REPLICATES,
        "block_hull": {"lengths": BOOTSTRAP_BLOCK_LENGTHS, "draws_per_length": N_BOOTSTRAP,
                        "interval": "hull of equal-tail intervals at alpha=0.025"},
        "finite_dependence_method": {
            "chromatic_number_bound": "chi=m+1 for the finite-range interval dependency graph",
            "bound": "radius=(4/3)*sqrt(chi*log(2/alpha_cell)/(2*n))",
            "alpha_cell": "lane_alpha/(2 tails * 4 loop means)",
            "propagation": "componentwise loop mean intervals through monotone log Creutz expression",
            "precision_gate": HALFWIDTH_TARGET,
            "invalid_draw_fraction_max": INVALID_DRAW_LIMIT,
        },
        "sample_size_planning": "population-centered optimistic radius calculation; not an achieved-coverage or physical N recommendation",
        "observable_source": {"git_revision": "54f0e97ba85b7f8f724c31d9f9d3512f040583a9",
                               "measure_py_sha256": "5fdee931db8a0b9bc74debb6f8c53e659ec6ccef7a42b0443788db4593e84ef6",
                               "normalization": "sum ReTr(W) over 3 spatial orientations and volume, divided by 9*volume"},
        "seed_rule": "20261125 + 10000000*kappa_index + 1000000*cross_corr_index + 10000*m_index + replicate; bootstrap adds 50000000 and block length",
        "planned_data_sets": len(BETA_CONCENTRATIONS) * len(LATENT_CROSS_CORRELATIONS) * len(DEPENDENCE_RANGES) * N_REPLICATES,
        "interpretation_limit": "Synthetic coverage is conditional on the exact finite-m dependence graph and declared Beta-copula DGP; it does not estimate physical HMC dependence or authorize production data.",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema": protocol["schema"], "classification": protocol["classification"],
        "physical_score": None, "planned_datasets": protocol["planned_data_sets"],
        "dataset_denominator": {"planned": protocol["planned_data_sets"],
                                "valid": protocol["planned_data_sets"], "failed": 0, "abstained": 0},
        "method_rows": summary_rows,
        "finite_dependence_precision_gate_abstains_all_n100": all(
            r["valid_release"] == 0 for r in summary_rows
            if r["method"] == "finite_dependence_Hoeffding"),
        "data_access": protocol["data_access"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    readme = f"""# Wilson C-lane finite-dependence interval stress

**Conditional synthetic-method comparison only.** No physical Wilson, development, sealed, GPU, or QUDA inputs were read. `physical_score` remains null.

![Coverage](coverage.png)

![Precision and release rate](precision.png)

![Optimistic sample-size planning](sample_size.png)

## Design

The target is the absolute finite-T Creutz quantity with ordered loop means `{list(LOOP_ORDER)}` and synthetic truth `{true_chi:.8f}`. The measured observable averages real traces over three spatial orientations and volume, with division by `9*volume`; each normalized real SU(3) trace lies in `[-1/3,1]`, so the average has the same support. This normalization is verified in `measure.py` at Git revision `54f0e97` (SHA-256 `{protocol['observable_source']['measure_py_sha256']}`). Marginals are affinely transformed Beta distributions with means `{LOOP_MEANS.tolist()}` and concentrations `{list(BETA_CONCENTRATIONS)}`. Four-dimensional Gaussian innovations have latent equicorrelation `{list(LATENT_CROSS_CORRELATIONS)}`. A moving sum of `m+1` innovation vectors, followed by the coordinatewise Gaussian-copula transform, produces a stationary process with exact temporal dependence range `m={list(DEPENDENCE_RANGES)}`. The crossed grid has {len(BETA_CONCENTRATIONS)*len(LATENT_CROSS_CORRELATIONS)*len(DEPENDENCE_RANGES)} cells and {N_REPLICATES} independent datasets per cell (`{protocol['planned_data_sets']}` planned datasets total).

The selected-proposal block-bootstrap method is the hull of circular moving-block percentile intervals at lengths `{list(BOOTSTRAP_BLOCK_LENGTHS)}`, with {N_BOOTSTRAP} draws per length. The comparison bound uses the known synthetic dependency graph, `chi=m+1`, a distribution-free graph-color Hoeffding radius for each loop mean with support width `4/3`, Bonferroni alpha `{event_alpha:.6f}` per one-sided loop event, and monotone interval propagation through the four logarithms. Under the declared m-dependence and `[-1/3,1]` support, this is a conditional finite-sample coverage route; it is not deployable unless an external source establishes the dependence range.

## Results and limits

Coverage and Monte Carlo Wilson intervals, median widths, precision passes, invalid-draw gates, failures and abstentions appear by cell in `coverage_summary.csv`; every replicate-level interval and draw denominator is in `replicates.csv`. Planned data-generation denominator is `{summary['dataset_denominator']}`. For each method, `planned = valid_release + failed + abstained`; `valid_release` counts intervals that pass both the invalid-draw limit `{INVALID_DRAW_LIMIT}` and the proposed half-width target `{HALFWIDTH_TARGET}`. A computed interval that fails either gate is explicitly abstained. The coverage plot shows raw candidate-interval coverage before those release gates, which must be read together with the precision/abstention plot.

The graph-color bound has nominal coverage conditional on the exact assumed dependency graph, but its log-propagated interval can be unbounded when any lower confidence limit for a loop mean reaches zero. The sample-size table solves for the maximum mean radius that would give half-width `{HALFWIDTH_TARGET}` if sample means were exactly equal to the known population means. This is an optimistic planning calculation, not an achieved precision guarantee; it can reveal that the distribution-free route is statistically valid but unusable at the present budget. The bootstrap's empirical coverage is limited to the declared Beta-copula moving-average scenarios and does not establish a physical HMC coverage claim. Individual bootstrap resamples with nonpositive required loop means remain invalid in the denominator; a block-length invalid fraction above `{INVALID_DRAW_LIMIT}` forces abstention, as specified in the method proposal.

Text flow: `bounded loop means + known synthetic m-dependence -> block-hull vs graph-color Hoeffding intervals -> nonlinear log propagation -> coverage/width/release tables -> use theorem route only with independently justified m; otherwise abstain`.

Rebuild in a fresh directory with `python -m tasks.wilson_loop.c_lane_finite_dependence_stress --output NEW_DIRECTORY`. `source_snapshot.py`, the protocol, replicate CSV, summary and artifact hash manifest are included.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    tracked = ["README.md", "coverage.png", "precision.png", "sample_size.png",
               "coverage_summary.csv", "replicates.csv", "sample_size_planning.csv",
               "protocol.json", "summary.json", "source_snapshot.py"]
    hashes = {name: sha256(output / name) for name in tracked}
    (output / "artifact_hashes.json").write_text(
        json.dumps({"algorithm": "sha256", "files": hashes}, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"output": str(output), "planned_datasets": protocol["planned_data_sets"],
                      "sample_size_planning": sample_rows, "method_rows": summary_rows}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
