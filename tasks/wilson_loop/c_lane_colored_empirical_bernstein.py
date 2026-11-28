"""Compare dependency-color empirical-Bernstein and Hoeffding C-lane bounds.

This synthetic study uses the exact finite-range DGP already specified by
``c_lane_finite_dependence_stress``. It adds one variance-adaptive method and a
sample-size sweep; it does not read Wilson measurements or infer HMC dependence.
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

from . import c_lane_finite_dependence_stress as base


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/wilson_loop/c_lane_colored_empirical_bernstein_20261125_v1"
SAMPLE_SIZES = (100, 500, 2000, 5000)
REPLICATES = 250


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def colored_empirical_bernstein(data: np.ndarray, m: int) -> dict:
    """Construct a two-sided simultaneous log-Creutz interval.

    The predeclared colors are indices modulo ``m+1``. For the moving-sum
    synthetic process, each fixed loop coordinate is independent within each
    such class. The interval applies Maurer--Pontil Theorem 11 separately to
    each class and loop, with Bonferroni over both tails, four loop means and
    all color classes. This theorem condition is not inferred from data.
    """
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("data must be a finite n-by-4 loop array")
    if np.any(values < base.LOOP_SUPPORT_LOWER) or np.any(values > base.LOOP_SUPPORT_UPPER):
        raise ValueError("loop values fall outside the predeclared support")
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 0:
        raise ValueError("m must be a nonnegative integer")

    n = len(values)
    chi = int(m) + 1
    colors = np.arange(n) % chi
    alpha_event = base.LANE_ALPHA / (2.0 * 4.0 * chi)
    log_term = math.log(2.0 / alpha_event)
    lower_means = np.zeros(4, dtype=np.float64)
    upper_means = np.zeros(4, dtype=np.float64)
    class_diagnostics = []
    for color in range(chi):
        indices = np.flatnonzero(colors == color)
        n_color = len(indices)
        if n_color == 0:
            raise ValueError("every predeclared color class must contain an observation")
        class_values = values[indices]
        mean = class_values.mean(axis=0)
        if n_color >= 2:
            variance = class_values.var(axis=0, ddof=1)
            # Theorem 11 on [0,1], rescaled by the support width 4/3.
            radius = (np.sqrt(2.0 * variance * log_term / n_color)
                      + 7.0 * base.LOOP_SUPPORT_WIDTH * log_term
                      / (3.0 * (n_color - 1)))
            method = "empirical_bernstein"
        else:
            variance = np.zeros(4, dtype=np.float64)
            radius = np.full(
                4,
                base.LOOP_SUPPORT_WIDTH
                * math.sqrt(math.log(1.0 / alpha_event) / (2.0 * n_color)),
            )
            method = "hoeffding_singleton"
        lower_means += (n_color / n) * np.maximum(base.LOOP_SUPPORT_LOWER, mean - radius)
        upper_means += (n_color / n) * np.minimum(base.LOOP_SUPPORT_UPPER, mean + radius)
        class_diagnostics.append({
            "color": color,
            "n": n_color,
            "method": method,
            "sample_means": mean.tolist(),
            "sample_variances": variance.tolist(),
            "radii": radius.tolist(),
        })

    # Creutz is decreasing in the first two loop means and increasing in the
    # final two; interval propagation preserves that monotonicity.
    if np.any(upper_means <= 0.0):
        return {
            "lower": None, "upper": None, "halfwidth": None,
            "computed": False, "reason": "nonpositive upper loop-mean bound",
            "mean_lower": lower_means.tolist(), "mean_upper": upper_means.tolist(),
            "color_class_count": chi, "alpha_per_one_sided_event": alpha_event,
            "classes": class_diagnostics,
        }
    lower = (None if np.any(lower_means[2:] <= 0.0) else
             -math.log(upper_means[0]) - math.log(upper_means[1])
             + math.log(lower_means[2]) + math.log(lower_means[3]))
    upper = (None if np.any(lower_means[:2] <= 0.0) else
             -math.log(lower_means[0]) - math.log(lower_means[1])
             + math.log(upper_means[2]) + math.log(upper_means[3]))
    finite = lower is not None and upper is not None
    halfwidth = (float(upper - lower) / 2.0) if finite else float("inf")
    return {
        "lower": float(lower) if lower is not None else None,
        "upper": float(upper) if upper is not None else None,
        "halfwidth": halfwidth,
        "computed": finite,
        "reason": None if finite else "at least one propagated log bound reaches zero",
        "mean_lower": lower_means.tolist(), "mean_upper": upper_means.tolist(),
        "color_class_count": chi, "alpha_per_one_sided_event": alpha_event,
        "classes": class_diagnostics,
    }


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / total
    denom = 1.0 + z*z/total
    center = (p + z*z/(2*total))/denom
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denom
    return center-half, center+half


def make_plots(rows: list[dict], out: Path) -> None:
    facets = [(k, c) for k in base.BETA_CONCENTRATIONS
              for c in base.LATENT_CROSS_CORRELATIONS]
    colors = {0: "#2563EB", 5: "#0F766E", 10: "#B91C1C"}
    methods = {
        "colored_empirical_Bernstein": ("o", "colored empirical-Bernstein"),
        "graph_color_Hoeffding": ("s", "graph-color Hoeffding"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    for ax, (concentration, cross_corr) in zip(axes.flat, facets):
        subset = [r for r in rows if r["beta_concentration"] == concentration
                  and r["latent_cross_correlation"] == cross_corr]
        for m in base.DEPENDENCE_RANGES:
            for method, (marker, label) in methods.items():
                curve = sorted((r for r in subset if r["dependence_range_m"] == m
                                and r["method"] == method), key=lambda r: r["n_configurations"])
                x = np.asarray([r["n_configurations"] for r in curve], dtype=float)
                y = np.asarray([r["coverage_fraction"] for r in curve], dtype=float)
                lo = np.asarray([r["coverage_mc95_low"] for r in curve], dtype=float)
                hi = np.asarray([r["coverage_mc95_high"] for r in curve], dtype=float)
                if np.any(np.isfinite(y)):
                    finite = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
                    ax.errorbar(x[finite], y[finite],
                                yerr=[np.maximum(y[finite]-lo[finite], 0),
                                      np.maximum(hi[finite]-y[finite], 0)],
                                marker=marker, color=colors[m],
                                linestyle="-" if method.startswith("colored") else "--",
                                capsize=2, linewidth=1.2,
                                label=f"m={m}: {label}")
        ax.axhline(1-base.LANE_ALPHA, color="#4B5563", linestyle=":", linewidth=1,
                   label="nominal 97.5%")
        ax.set_xscale("log")
        ax.set_ylim(0.5, 1.01)
        ax.set(xlabel="number of configurations", ylabel="coverage among finite intervals",
               title=f"Beta concentration={concentration:g}, cross-correlation={cross_corr:g}")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6.8, ncol=2)
    fig.suptitle("Finite-dependence interval coverage (synthetic; known m)")
    fig.savefig(out / "coverage.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    for ax, (concentration, cross_corr) in zip(axes.flat, facets):
        subset = [r for r in rows if r["beta_concentration"] == concentration
                  and r["latent_cross_correlation"] == cross_corr]
        for m in base.DEPENDENCE_RANGES:
            for method, (marker, label) in methods.items():
                curve = sorted((r for r in subset if r["dependence_range_m"] == m
                                and r["method"] == method), key=lambda r: r["n_configurations"])
                x = np.asarray([r["n_configurations"] for r in curve], dtype=float)
                y = np.asarray([r["valid_release_fraction"] for r in curve], dtype=float)
                ax.plot(x, y, marker=marker, color=colors[m], linestyle="-" if method.startswith("colored") else "--",
                        linewidth=1.2, label=f"m={m}: {label}")
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set(xlabel="number of configurations", ylabel="fraction meeting half-width <= 0.02",
               title=f"Beta concentration={concentration:g}, cross-correlation={cross_corr:g}")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6.8, ncol=2)
    fig.suptitle("Precision-gate release fraction under conditional bounds")
    fig.savefig(out / "precision_release.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for method, marker, label in (
        ("colored_empirical_Bernstein", "o", "colored empirical-Bernstein"),
        ("graph_color_Hoeffding", "s", "graph-color Hoeffding"),
    ):
        means = []
        finite_counts = []
        for n in SAMPLE_SIZES:
            subset = [r for r in rows if r["n_configurations"] == n and r["method"] == method]
            widths = [r["median_halfwidth"] for r in subset
                      if r["median_halfwidth"] is not None]
            means.append(float(np.median(widths)) if widths else float("nan"))
            finite_counts.append(sum(int(r["interval_finite_count"]) for r in subset))
        if any(math.isfinite(value) for value in means):
            ax.plot(SAMPLE_SIZES, means, marker=marker, linewidth=1.7, label=label)
        else:
            ax.text(0.02, 0.95 if method.startswith("colored") else 0.87,
                    f"{label}: no finite intervals in this grid",
                    transform=ax.transAxes, va="top", fontsize=8)
    ax.axhline(base.HALFWIDTH_TARGET, color="#4B5563", linestyle=":", label="target 0.02")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="number of configurations", ylabel="median finite interval half-width",
           title="Pooled precision by sample size (12 synthetic cells)")
    ax.grid(alpha=0.2, which="both")
    ax.legend()
    fig.savefig(out / "halfwidth_vs_n.png", dpi=170)
    plt.close(fig)


def build_report(output: str | Path = DEFAULT_OUT) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    truth = float(base.creutz_from_means(base.LOOP_MEANS))
    replicate_rows = []
    for k_idx, concentration in enumerate(base.BETA_CONCENTRATIONS):
        for c_idx, cross_corr in enumerate(base.LATENT_CROSS_CORRELATIONS):
            for m_idx, m in enumerate(base.DEPENDENCE_RANGES):
                for n in SAMPLE_SIZES:
                    for rep in range(REPLICATES):
                        seed = (base.SEED + 10_000_000*k_idx + 1_000_000*c_idx
                                + 10_000*m_idx + rep)
                        data = base.m_dependent_beta_series(
                            n, m, cross_corr, concentration, np.random.default_rng(seed)
                        )
                        intervals = {
                            "colored_empirical_Bernstein": colored_empirical_bernstein(data, m),
                            "graph_color_Hoeffding": base.finite_dependence_hoeffding(data, m),
                        }
                        for method, interval in intervals.items():
                            computed = bool(interval["computed"])
                            finite = bool(computed and interval["lower"] is not None
                                          and interval["upper"] is not None
                                          and math.isfinite(interval["lower"])
                                          and math.isfinite(interval["upper"]))
                            covered = bool(finite and interval["lower"] <= truth <= interval["upper"])
                            halfwidth = interval["halfwidth"]
                            precision = bool(finite and math.isfinite(halfwidth)
                                              and halfwidth <= base.HALFWIDTH_TARGET)
                            replicate_rows.append({
                                "beta_concentration": concentration,
                                "latent_cross_correlation": cross_corr,
                                "dependence_range_m": m,
                                "n_configurations": n,
                                "replicate": rep,
                                "method": method,
                                "planned": 1,
                                "interval_computed": computed,
                                "interval_finite": finite,
                                "failed": False,
                                "abstained": not precision,
                                "valid_release": precision,
                                "covered": covered,
                                "lower": interval["lower"],
                                "upper": interval["upper"],
                                "halfwidth": halfwidth if halfwidth is not None and math.isfinite(halfwidth) else None,
                                "unbounded": not finite,
                                "reason": interval.get("reason"),
                                "true_creutz": truth,
                                "chromatic_number": m + 1,
                            })

    summary_rows = []
    for concentration in base.BETA_CONCENTRATIONS:
        for cross_corr in base.LATENT_CROSS_CORRELATIONS:
            for m in base.DEPENDENCE_RANGES:
                for n in SAMPLE_SIZES:
                    for method in ("colored_empirical_Bernstein", "graph_color_Hoeffding"):
                        cell = [r for r in replicate_rows
                                if r["beta_concentration"] == concentration
                                and r["latent_cross_correlation"] == cross_corr
                                and r["dependence_range_m"] == m
                                and r["n_configurations"] == n
                                and r["method"] == method]
                        computed = [r for r in cell if r["interval_computed"]]
                        finite = [r for r in cell if r["interval_finite"]]
                        hits = sum(r["covered"] for r in finite)
                        low, high = wilson_interval(hits, len(finite))
                        widths = [r["halfwidth"] for r in finite if r["halfwidth"] is not None]
                        summary_rows.append({
                            "beta_concentration": concentration,
                            "latent_cross_correlation": cross_corr,
                            "dependence_range_m": m,
                            "n_configurations": n,
                            "method": method,
                            "planned": len(cell),
                            "interval_computed": len(computed),
                            "interval_finite_count": len(finite),
                            "failed": sum(r["failed"] for r in cell),
                            "abstained": sum(r["abstained"] for r in cell),
                            "valid_release": sum(r["valid_release"] for r in cell),
                            "coverage_fraction": hits / len(finite) if finite else None,
                            "coverage_mc95_low": low,
                            "coverage_mc95_high": high,
                            "valid_release_fraction": sum(r["valid_release"] for r in cell) / len(cell),
                            "median_halfwidth": float(np.median(widths)) if widths else None,
                            "unbounded_count": sum(r["unbounded"] for r in cell),
                        })

    for name, rows in (("replicates.csv", replicate_rows), ("coverage_summary.csv", summary_rows)):
        with (out / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    shutil.copy2(Path(__file__).resolve(), out / "source_snapshot.py")
    shutil.copy2(Path(base.__file__).resolve(), out / "dgp_source_snapshot.py")
    make_plots(summary_rows, out)
    totals = {
        method: {
            "planned": len(rows),
            "valid": sum(r["valid_release"] for r in rows),
            "failed": sum(r["failed"] for r in rows),
            "abstained": sum(r["abstained"] for r in rows),
            "valid_release": sum(r["valid_release"] for r in rows),
            "finite_interval_count": sum(r["interval_finite"] for r in rows),
            "unbounded_count": sum(r["unbounded"] for r in rows),
        }
        for method in ("colored_empirical_Bernstein", "graph_color_Hoeffding")
        for rows in [[r for r in replicate_rows if r["method"] == method]]
    }
    protocol = {
        "schema": "wilson-c-lane-colored-empirical-bernstein-v1",
        "classification": "synthetic_conditional_method_comparison_only",
        "physical_score": None,
        "data_access": "no Wilson measurement, development/sealed split, GPU or QUDA data read",
        "dgp": {
            "source": "the same Gaussian-copula Beta moving-sum DGP as the prior finite-dependence report; this study adds a bound and sample-size comparison",
            "dependence_ranges": list(base.DEPENDENCE_RANGES),
            "cross_correlations": list(base.LATENT_CROSS_CORRELATIONS),
            "beta_concentrations": list(base.BETA_CONCENTRATIONS),
            "sample_sizes": list(SAMPLE_SIZES),
            "replicates_per_cell": REPLICATES,
            "total_datasets": len(base.DEPENDENCE_RANGES) * len(base.LATENT_CROSS_CORRELATIONS) * len(base.BETA_CONCENTRATIONS) * len(SAMPLE_SIZES) * REPLICATES,
            "mean_seed_matched_across_sample_sizes": True,
        },
        "estimand": {
            "name": "absolute finite-T Creutz",
            "truth": truth,
            "loop_order": list(base.LOOP_ORDER),
            "support": [base.LOOP_SUPPORT_LOWER, base.LOOP_SUPPORT_UPPER],
            "precision_halfwidth_target": base.HALFWIDTH_TARGET,
        },
        "methods": {
            "colored_empirical_Bernstein": {
                "source": "Maurer and Pontil (2009), Theorem 11, independent not necessarily identically distributed variables; arXiv:0907.3740",
                "coloring": "index modulo (m+1), valid for this synthetic moving-sum DGP only",
                "alpha_allocation": "LANE_ALPHA / (2 tails * 4 means * chi color classes)",
                "log_propagation": "monotone interval arithmetic through the four log means; zero-crossing intervals abstain",
            },
            "graph_color_Hoeffding": {
                "source": "same conditional finite-dependence bound as the prior report",
                "purpose": "same-sample baseline for adaptive variance interval",
            },
        },
        "method_denominators": totals,
        "limitations": [
            "simulation coverage is conditional on the exact synthetic dependence graph and support",
            "the true m is supplied by the synthetic generator and is not estimated from HMC data",
            "the study does not validate theorem assumptions or coverage on physical Wilson configurations",
            "a valid finite-sample interval may remain too wide to pass the 0.02 precision gate",
        ],
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    method_lines = "\n".join(
        f"| {method} | {d['planned']} | {d['finite_interval_count']} | "
        f"{d['unbounded_count']} | {d['failed']} | {d['abstained']} | "
        f"{d['valid_release']} |"
        for method, d in totals.items()
    )
    readme = f"""# Wilson C-lane colored empirical-Bernstein diagnostic

**Conditional synthetic method comparison only.** `physical_score=null`; no Wilson measurements, sealed data, GPU or QUDA were accessed.

![Coverage across sample size](coverage.png)

![Precision release fraction](precision_release.png)

![Median half-width by sample size](halfwidth_vs_n.png)

## Method and assumptions

This study adds a color-class empirical-Bernstein interval and a sample-size sweep to the already declared finite-range moving-sum Gaussian-copula Beta generator. For a true synthetic dependence range `m`, indices modulo `m+1` form independent color classes for each fixed loop coordinate. Within each class, Maurer–Pontil Theorem 11 supplies a variance-adaptive one-sided bound for independent bounded variables, including non-identically distributed variables. We split the lane alpha over two tails, four loop means and `chi=m+1` color classes, then take a weighted union of the class intervals. Monotone propagation through the four logarithms yields the Creutz interval; any loop-mean interval touching zero makes it unbounded and therefore abstained.

The theorem is conditional on a valid predeclared coloring, class-wise independence and fixed supports. Here these are guaranteed by construction, because the generator uses an exact m-dependent moving sum. They are not known for a production HMC chain. The method receives the synthetic `m`; it does not estimate it. Source: [Maurer and Pontil, *Empirical Bernstein Bounds and Sample Variance Penalization*, Theorem 11](https://arxiv.org/abs/0907.3740).

The design has 12 parameter cells, four sample sizes `{SAMPLE_SIZES}`, and {REPLICATES} independent datasets per cell: **{protocol['dgp']['total_datasets']} planned datasets**. Random seeds are paired across sample sizes so each larger chain extends the same shorter prefix. Replicate-level interval status and coverage are in `replicates.csv`; cell summaries with Wilson Monte Carlo intervals are in `coverage_summary.csv`.

## Denominators

| method | planned | finite intervals | unbounded | failed | abstained | precision passes |
|---|---:|---:|---:|---:|---:|---:|
{method_lines}

`valid` counts intervals that pass both finiteness and the precision threshold. `abstained` includes computed but unbounded intervals and finite intervals wider than the threshold; `failed` is zero unless numerical evaluation itself fails. These status counts refer to synthetic datasets, not physical configurations. Coverage is calculated only over finite intervals; unbounded intervals are reported separately and cannot pass the release gate.

Text flow: `bounded loop traces + exact synthetic m -> color classes -> per-color empirical-Bernstein intervals (or Hoeffding comparator) -> weighted loop-mean box -> monotone four-log Creutz interval -> coverage/precision/abstention plots`.

## Interpretation limits

In this grid, the variance-adaptive method produced some finite intervals in `m=0` cells where graph-color Hoeffding stayed unbounded, but all finite half-widths were far above `0.02`; it produced no finite intervals for `m=5` or `m=10`. Its finite-sample remainder still scales with the number of color classes and confidence allocation. The simulation only examines the specified Beta-copula moving-sum scenarios; it does not establish real HMC dependence range, coverage, a justified `m`, or usable production precision. If production dependence remains uncertified, the correct outcome is `ABSTAIN`.

Rebuild with `python -m tasks.wilson_loop.c_lane_colored_empirical_bernstein --output NEW_DIRECTORY`. Source snapshots and all output hashes are recorded in `artifact_hashes.json`.
"""
    (out / "README.md").write_text(readme)
    payloads = sorted(path for path in out.iterdir() if path.is_file())
    hashes = {path.name: sha256(path) for path in payloads if path.name != "artifact_hashes.json"}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    return {"status": "SYNTHETIC_CONDITIONAL_METHOD_COMPARISON", "denominators": totals,
            "output": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build_report(args.output), indent=2))


if __name__ == "__main__":
    main()
