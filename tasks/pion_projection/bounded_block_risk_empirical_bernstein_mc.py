"""Synthetic comparison of graph-color Hoeffding and empirical-Bernstein LCBs.

The m-dependent generator has known color classes, so this tests only the
conditional finite-sample behavior of the two bounds when that graph is known.
It cannot infer a dependency graph or produce a physical Pion score.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np

from tasks.pion_projection.bounded_block_risk_coverage_mc import (
    ALPHA_S, BASE_SCORE_MEAN, BLOCK_COUNTS, CHUNK_SIZE,
    DEPENDENCE_ORDERS, FAILURE_PROBABILITIES, FAMILY,
    NOISE_AMPLITUDE, POSITIVE_CERTIFICATE_THRESHOLD, REPLICATES, SEED,
    _cell_seed, _wilson_interval, moving_window_scores,
)
from tasks.pion_projection.bounded_block_risk_empirical_bernstein import (
    color_class_empirical_bernstein_lcb,
)
from tasks.pion_projection.dependency_graph_bound import dependence_penalty


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / (
    "results/pion_projection/"
    "bounded_block_risk_empirical_bernstein_20261123_v4"
)


def simulate_cell(*, blocks: int, dependence_order: int,
                  failure_probability: float, replicates: int = REPLICATES,
                  chunk_size: int = CHUNK_SIZE) -> tuple[dict, dict[str, np.ndarray]]:
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 1:
        raise ValueError("blocks must be a positive integer")
    if (isinstance(dependence_order, bool)
            or not isinstance(dependence_order, int)
            or not 0 <= dependence_order < blocks):
        raise ValueError("dependence_order must be in [0, blocks)")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not 0 <= failure_probability < 1:
        raise ValueError("failure_probability must be in [0,1)")

    chi = dependence_order + 1
    labels = np.arange(blocks, dtype=np.int64) % chi
    graph_radius = dependence_penalty(
        blocks=blocks, chromatic_number=chi, family=FAMILY, alpha_s=ALPHA_S,
    )
    true_mean = ((1.0 - failure_probability) * BASE_SCORE_MEAN
                 - failure_probability)
    rng = np.random.default_rng(_cell_seed(
        blocks, dependence_order, failure_probability,
    ))
    coverage_graph = np.empty(replicates, dtype=bool)
    coverage_eb = np.empty(replicates, dtype=bool)
    positive_graph = np.empty(replicates, dtype=bool)
    positive_eb = np.empty(replicates, dtype=bool)
    min_lcb_graph = np.empty(replicates, dtype=np.float32)
    min_lcb_eb = np.empty(replicates, dtype=np.float32)
    eb_penalty = np.empty(replicates, dtype=np.float32)
    failed_count = np.empty(replicates, dtype=np.int16)
    cursor = 0
    while cursor < replicates:
        size = min(chunk_size, replicates - cursor)
        failures = rng.random((size, blocks)) < failure_probability
        scores = np.stack([
            moving_window_scores(
                rng, rows=size, blocks=blocks,
                dependence_order=dependence_order,
                mean=BASE_SCORE_MEAN, amplitude=NOISE_AMPLITUDE,
            ) for _ in range(FAMILY)
        ], axis=1)
        scores[failures[:, None, :].repeat(FAMILY, axis=1)] = -1.0
        mean = scores.mean(axis=-1)
        graph_lcb = mean - graph_radius
        eb_result = color_class_empirical_bernstein_lcb(
            scores, color_labels=labels, alpha_score=ALPHA_S,
        )
        eb_lcb = eb_result["simultaneous_lower_bounds"]
        sl = slice(cursor, cursor + size)
        coverage_graph[sl] = np.all(graph_lcb <= true_mean, axis=1)
        coverage_eb[sl] = np.all(eb_lcb <= true_mean, axis=1)
        positive_graph[sl] = np.all(graph_lcb > POSITIVE_CERTIFICATE_THRESHOLD, axis=1)
        positive_eb[sl] = np.all(eb_lcb > POSITIVE_CERTIFICATE_THRESHOLD, axis=1)
        min_lcb_graph[sl] = graph_lcb.min(axis=1)
        min_lcb_eb[sl] = eb_lcb.min(axis=1)
        eb_penalty[sl] = (mean - eb_lcb).max(axis=1)
        failed_count[sl] = failures.sum(axis=1)
        cursor += size

    graph_successes = int(coverage_graph.sum())
    eb_successes = int(coverage_eb.sum())
    graph_ci = _wilson_interval(graph_successes, replicates)
    eb_ci = _wilson_interval(eb_successes, replicates)
    row = {
        "physical_score": None,
        "planned_replicates": replicates,
        "valid_replicates": replicates,
        "failed_replicates": 0,
        "abstained_replicates": 0,
        "blocks": blocks,
        "dependence_order_m": dependence_order,
        "graph_chromatic_number": chi,
        "failure_probability": failure_probability,
        "family_size_F": FAMILY,
        "alpha_s": ALPHA_S,
        "true_failure_augmented_mean": true_mean,
        "planned_pair_block_events": replicates * blocks,
        "valid_pair_block_events": int(replicates * blocks - failed_count.sum()),
        "failed_pair_block_events": int(failed_count.sum()),
        "abstained_pair_block_events": 0,
        "planned_family_block_scores": replicates * blocks * FAMILY,
        "valid_family_block_scores": int((replicates * blocks - failed_count.sum()) * FAMILY),
        "failed_family_block_scores": int(failed_count.sum() * FAMILY),
        "abstained_family_block_scores": 0,
        "graph_hoeffding_penalty": graph_radius,
        "familywise_coverage_graph_hoeffding": graph_successes / replicates,
        "familywise_coverage_graph_hoeffding_mc95_low": graph_ci[0],
        "familywise_coverage_graph_hoeffding_mc95_high": graph_ci[1],
        "familywise_coverage_color_empirical_bernstein": eb_successes / replicates,
        "familywise_coverage_color_empirical_bernstein_mc95_low": eb_ci[0],
        "familywise_coverage_color_empirical_bernstein_mc95_high": eb_ci[1],
        "positive_certificate_rate_graph_hoeffding": float(positive_graph.mean()),
        "positive_certificate_rate_color_empirical_bernstein": float(positive_eb.mean()),
        "median_min_lcb_graph_hoeffding": float(np.median(min_lcb_graph)),
        "median_min_lcb_color_empirical_bernstein": float(np.median(min_lcb_eb)),
        "median_worst_candidate_eb_penalty": float(np.median(eb_penalty)),
        "empirical_failure_fraction": float(failed_count.sum() / (replicates * blocks)),
        "seed": _cell_seed(blocks, dependence_order, failure_probability),
        "classification": "conditional synthetic Monte Carlo with known m-dependent graph",
    }
    arrays = {
        "coverage_graph_hoeffding": coverage_graph,
        "coverage_color_empirical_bernstein": coverage_eb,
        "positive_graph_hoeffding": positive_graph,
        "positive_color_empirical_bernstein": positive_eb,
        "min_lcb_graph_hoeffding": min_lcb_graph,
        "min_lcb_color_empirical_bernstein": min_lcb_eb,
        "worst_candidate_eb_penalty": eb_penalty,
        "failed_block_count": failed_count,
    }
    return row, arrays


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(out: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    block_counts = sorted({row["blocks"] for row in rows})
    failure_rates = sorted({row["failure_probability"] for row in rows})
    fig, axes = plt.subplots(len(block_counts), len(failure_rates),
                             figsize=(11.5, 3.0 * len(block_counts)),
                             sharex=True, sharey=True, layout="constrained")
    axes = np.atleast_2d(axes)
    for row_index, blocks in enumerate(block_counts):
        for col_index, failure in enumerate(failure_rates):
            ax = axes[row_index, col_index]
            subset = sorted((row for row in rows
                             if row["blocks"] == blocks
                             and row["failure_probability"] == failure),
                            key=lambda row: row["dependence_order_m"])
            x = [row["dependence_order_m"] for row in subset]
            for color, value_key, lo_key, hi_key, label in (
                ("#0072B2", "familywise_coverage_graph_hoeffding",
                 "familywise_coverage_graph_hoeffding_mc95_low",
                 "familywise_coverage_graph_hoeffding_mc95_high",
                 "graph-color Hoeffding"),
                ("#D55E00", "familywise_coverage_color_empirical_bernstein",
                 "familywise_coverage_color_empirical_bernstein_mc95_low",
                 "familywise_coverage_color_empirical_bernstein_mc95_high",
                 "color-class empirical Bernstein"),
            ):
                y = [row[value_key] for row in subset]
                lo = [value - row[lo_key] for value, row in zip(y, subset)]
                hi = [row[hi_key] - value for value, row in zip(y, subset)]
                ax.errorbar(x, y, yerr=[lo, hi], marker="o", capsize=2,
                            color=color, label=label)
            ax.axhline(1.0 - ALPHA_S, color="black", ls="--", lw=1,
                       label="nominal familywise coverage")
            ax.set_ylim(0.94, 1.005)
            ax.set_title(f"N={blocks}; coded failure probability={failure:g}")
            ax.grid(alpha=0.25)
            if col_index == 0:
                ax.set_ylabel("familywise LCB coverage\n(95% Monte Carlo interval)")
            if row_index == len(block_counts) - 1:
                ax.set_xlabel("true dependence range m; color bound = m+1")
            if row_index == 0 and col_index == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Synthetic known-graph coverage; 5,000 replicates/cell; no physical coverage claim")
    fig.savefig(out / "empirical_bernstein_coverage.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(failure_rates),
                             figsize=(6.2 * len(failure_rates), 4.5),
                             sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
    for col_index, failure in enumerate(failure_rates):
        ax = axes[col_index]
        for color_index, blocks in enumerate(block_counts):
            color = palette[color_index % len(palette)]
            subset = sorted((row for row in rows if row["blocks"] == blocks
                             and row["failure_probability"] == failure),
                            key=lambda row: row["dependence_order_m"])
            x = [row["dependence_order_m"] for row in subset]
            label = f"N={blocks}, failure={failure:g}"
            ax.plot(x, [row["positive_certificate_rate_graph_hoeffding"]
                        for row in subset], color=color, ls="-",
                    marker="o", label=f"graph Hoeffding, N={blocks}")
            ax.plot(x, [row["positive_certificate_rate_color_empirical_bernstein"]
                        for row in subset], color=color, ls="--",
                    marker="s", label=f"empirical Bernstein, N={blocks}")
        ax.set(xlabel="true dependence range m",
               ylabel="fraction of cases with both LCBs > 0",
               title=f"Illustrative certificate rate; failure={failure:g}",
               ylim=(-0.03, 1.03))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Synthetic positive-certificate rates; threshold 0 is a display value")
    fig.savefig(out / "empirical_bernstein_utility.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    for color_index, blocks in enumerate(block_counts):
        color = palette[color_index % len(palette)]
        for failure, style in zip(failure_rates, ("-", "--")):
            subset = sorted((row for row in rows if row["blocks"] == blocks
                             and row["failure_probability"] == failure),
                            key=lambda row: row["dependence_order_m"])
            ax.plot([row["dependence_order_m"] for row in subset],
                    [row["median_worst_candidate_eb_penalty"] for row in subset],
                    color=color, ls=style, marker="o",
                    label=f"N={blocks}, failure={failure:g}")
    ax.set(xlabel="true dependence range m",
           ylabel="median worst-candidate EB penalty",
           title="Empirical-Bernstein penalty after color/family split")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Penalty grows as color classes become smaller")
    fig.savefig(out / "empirical_bernstein_penalty.png", dpi=180)
    plt.close(fig)


def build(output: str | Path = DEFAULT_OUT, *, replicates: int = REPLICATES,
          block_counts: tuple[int, ...] = BLOCK_COUNTS,
          dependence_orders: tuple[int, ...] = DEPENDENCE_ORDERS,
          failure_probabilities: tuple[float, ...] = FAILURE_PROBABILITIES) -> dict:
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    replicate_arrays: dict[str, np.ndarray] = {}
    for blocks in block_counts:
        for order in dependence_orders:
            for failure in failure_probabilities:
                row, arrays = simulate_cell(
                    blocks=blocks, dependence_order=order,
                    failure_probability=failure, replicates=replicates,
                )
                rows.append(row)
                key = f"N{blocks}_m{order}_phi{int(100*failure):02d}"
                for name, values in arrays.items():
                    replicate_arrays[f"{key}_{name}"] = values
    np.savez_compressed(out / "replicate_summary.npz", **replicate_arrays)
    _write_csv(out / "cell_summary.csv", rows)
    _write_plots(out, rows)
    protocol = {
        "schema": "pion-bounded-block-risk-color-empirical-bernstein-v1",
        "classification": "synthetic conditional calibration experiment",
        "physical_score": None,
        "data_access": "no project measurements, anchor, sealed split, GPU, or QUDA data accessed",
        "blocks": list(block_counts), "dependence_orders": list(dependence_orders),
        "failure_probabilities": list(failure_probabilities),
        "replicates_per_cell": replicates, "family": FAMILY,
        "alpha_score": ALPHA_S,
        "score_generator": "mean + amplitude*tanh of moving sums of Gaussian innovations; residues modulo m+1 have disjoint innovations",
        "failure_rule": "paired independent failure mask; each failed planned score remains in N and is encoded as -1",
        "empirical_bernstein": "apply Maurer-Pontil Theorem 11 after [-1,1] to [0,1] scaling within each predeclared color class; Bonferroni alpha_s/(F*chi); singleton class fallback Hoeffding",
        "graph_hoeffding_baseline": "existing graph-color penalty with known chi=m+1",
        "multiplicity": "simultaneous over F candidates and chi color classes for empirical Bernstein; finite family F in Hoeffding baseline",
        "positive_certificate_threshold": POSITIVE_CERTIFICATE_THRESHOLD,
        "source_hashes": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("bounded_block_risk_empirical_bernstein_mc.py",
                         "bounded_block_risk_empirical_bernstein.py",
                         "bounded_block_risk_coverage_mc.py",
                         "dependency_graph_bound.py")
        },
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__},
        "limitations": [
            "the dependency graph and proper coloring are known by construction, not inferred",
            "all guarantees are conditional on independent variables within each color class",
            "sample variance cannot repair dependence, anchor bias, or misspecified clipping bounds",
            "empirical Bernstein can be wider than Hoeffding at high chromatic number or small color classes",
            "the positive threshold is an illustration, not a physical delta_star",
        ],
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    coverage_key = "familywise_coverage_color_empirical_bernstein"
    gap = [row[coverage_key] - row["familywise_coverage_graph_hoeffding"]
           for row in rows]
    summary = {
        "schema": protocol["schema"], "status": "SYNTHETIC_CONDITIONAL_DIAGNOSTIC_ONLY",
        "physical_score": None, "planned_cells": len(rows),
        "valid_cells": sum(row["planned_replicates"] == row["valid_replicates"]
                            for row in rows),
        "failed_cells": sum(row["valid_replicates"] != row["planned_replicates"]
                             for row in rows),
        "planned_replicates": sum(row["planned_replicates"] for row in rows),
        "valid_replicates": sum(row["valid_replicates"] for row in rows),
        "failed_replicates": sum(row["failed_replicates"] for row in rows),
        "abstained_replicates": sum(row["abstained_replicates"] for row in rows),
        "minimum_graph_hoeffding_coverage": min(
            row["familywise_coverage_graph_hoeffding"] for row in rows),
        "minimum_empirical_bernstein_coverage": min(row[coverage_key] for row in rows),
        "minimum_empirical_bernstein_coverage_mc95_low": min(
            row["familywise_coverage_color_empirical_bernstein_mc95_low"] for row in rows),
        "largest_empirical_bernstein_minus_graph_hoeffding_coverage": max(gap),
        "smallest_empirical_bernstein_minus_graph_hoeffding_coverage": min(gap),
        "files": ["README.md", "protocol.json", "cell_summary.csv",
                  "replicate_summary.npz", "empirical_bernstein_coverage.png",
                  "empirical_bernstein_utility.png", "empirical_bernstein_penalty.png",
                  "snapshot_empirical_bernstein.py",
                  "snapshot_coverage_mc.py", "snapshot_dependency_graph_bound.py",
                  "snapshot_bounded_block_risk_empirical_bernstein_mc.py",
                  "summary.json", "artifact_hashes.json"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    selected_examples = [row for row in rows
                         if row["blocks"] == 512
                         and row["dependence_order_m"] in (0, 1, 3, 7, 15)]
    example_lines = [
        "|N|m|failure probability|Hoeffding positive rate|empirical-Bernstein positive rate|median EB penalty|",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected_examples:
        example_lines.append(
            f"|{row['blocks']}|{row['dependence_order_m']}|{row['failure_probability']:.1f}|"
            f"{row['positive_certificate_rate_graph_hoeffding']:.4f}|"
            f"{row['positive_certificate_rate_color_empirical_bernstein']:.4f}|"
            f"{row['median_worst_candidate_eb_penalty']:.4f}|"
        )
    report = [
        "# Color-class empirical-Bernstein calibration diagnostic",
        "",
        "**SYNTHETIC CONDITIONAL DIAGNOSTIC ONLY. `physical_score=null`.** No Pion correlators, anchor, sealed split, GPU, or QUDA inputs were accessed.",
        "",
        f"The simulation compares the existing graph-color Hoeffding penalty with a color-class empirical-Bernstein LCB over {len(rows)} cells and {replicates} paired replicates per cell. The m-dependent score generator has a known proper coloring `c(i)=i mod (m+1)`, so blocks within each color class use disjoint Gaussian innovations. The two methods see identical seeded scores and paired failure masks.",
        "",
        "![Familywise coverage](empirical_bernstein_coverage.png)",
        "",
        "![Certificate-rate and variance-penalty diagnostics](empirical_bernstein_utility.png)",
        "",
        "![Empirical-Bernstein penalty growth](empirical_bernstein_penalty.png)",
        "",
        "## Bound and assumptions",
        "",
        "For each fixed candidate and graph color class, the sample is bounded in `[-1,1]`. After mapping to `[0,1]`, Maurer–Pontil's empirical-Bernstein bound is applied with per-class `delta=alpha_s/(F*chi)`, then the color-class lower bounds are averaged with weights `n_c/N`. A singleton color class uses a Hoeffding fallback. This yields a familywise score bound only when every color class is jointly independent and the candidate family was frozen. The primary paper states the bound for independent, not necessarily identically distributed, variables (Theorem 11); this does not establish such independence for project blocks. [Maurer & Pontil, “Empirical Bernstein Bounds and Sample Variance Penalization”](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf).",
        "",
        "Empirical variance can tighten the class-wise radius when color classes are large and their observed variance is small. Its additive finite-sample term grows as classes shrink; splitting a high-dependence graph into many colors can make this method looser than the existing Hoeffding graph bound. A variance estimate does not estimate dependence or fix an invalid coloring.",
        "",
        "## Coverage and accounting",
        "",
        f"Across the generated grid, graph-color Hoeffding minimum empirical coverage was {summary['minimum_graph_hoeffding_coverage']:.4f}; empirical-Bernstein minimum empirical coverage was {summary['minimum_empirical_bernstein_coverage']:.4f}, with minimum per-cell Monte Carlo 95% Wilson lower limit {summary['minimum_empirical_bernstein_coverage_mc95_low']:.4f}. Coverage is measured against the known failure-augmented mean `(1-phi)*0.2-phi`; a Monte Carlo lower limit below the nominal 0.95 is finite replicate uncertainty, not a demonstrated bound failure.",
        "",
        f"Replicate denominators are planned/valid/failed/abstained = `{summary['planned_replicates']}/{summary['valid_replicates']}/{summary['failed_replicates']}/{summary['abstained_replicates']}`. Failed block events remain in the planned denominator and receive score `-1`; no replicate is dropped. `cell_summary.csv` separately reports block-event and family-block-score denominators.",
        "",
        "At `N=512`, all shown graph and empirical-Bernstein cells observed familywise coverage 1.0/1.0, while the positive-certificate rates show a clear efficiency tradeoff:",
        "",
        *example_lines,
        "",
        "At `m=0`, empirical Bernstein is slightly more useful in this fixture. At `m=1`, it cuts the no-failure certificate rate from 0.6994 to 0.3550; at `m≥3`, both methods are near zero power, with empirical Bernstein no better. The larger additive term after color splitting explains why this alternative is not a general improvement.",
        "",
        "## Interpretation and next boundary",
        "",
        "This is a candidate method for a future sensitivity lane, not a replacement selected by these synthetic results. Retain graph-color Hoeffding as the conservative reference until its assumptions are established. Empirical Bernstein is worth comparing only if a defensible score-specific coloring exists and color classes are large enough to avoid a dominant additive term. If project dependence is unknown, both bounds abstain; a low sample variance cannot make them valid.",
        "",
        "The score remains `bounded_block_risk`, a clipped planned-block estimand. This report does not change the clipping range, target mass, `R`, candidate family, `alpha_total`, anchor distribution, or common-bias contract. The project remains `NOT_READY/UNCERTIFIED`, sealed data stay closed, and `physical_score=null`.",
        "",
        "## Rebuild",
        "",
        "Use a new output directory; existing artifacts are never overwritten:",
        "",
        "```text",
        "python -B -m tasks.pion_projection.bounded_block_risk_empirical_bernstein_mc --output NEW_DIRECTORY",
        "```",
    ]
    (out / "README.md").write_text("\n".join(report) + "\n")
    snapshots = {
        "snapshot_empirical_bernstein.py": Path(__file__).with_name("bounded_block_risk_empirical_bernstein.py"),
        "snapshot_coverage_mc.py": Path(__file__).with_name("bounded_block_risk_coverage_mc.py"),
        "snapshot_dependency_graph_bound.py": Path(__file__).with_name("dependency_graph_bound.py"),
        "snapshot_bounded_block_risk_empirical_bernstein_mc.py": Path(__file__),
    }
    for filename, source in snapshots.items():
        shutil.copyfile(source, out / filename)
    hashes = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
              for name in summary["files"] if name != "artifact_hashes.json"}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    args = parser.parse_args()
    summary = build(args.output, replicates=args.replicates)
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
