"""Synthetic coverage study for a graph-color bounded block-risk lower bound.

The score generator is exactly m-dependent by construction. This validates the
conditional calibration formula under a known synthetic graph; it cannot infer
dependence, anchor coverage, or a physical Pion score from project data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

from tasks.pion_projection.dependency_graph_bound import dependence_penalty


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/pion_projection/bounded_block_risk_coverage_mc_20261123_v2"
BLOCK_COUNTS = (32, 128, 512)
DEPENDENCE_ORDERS = (0, 1, 3, 7, 15)
FAILURE_PROBABILITIES = (0.0, 0.1)
REPLICATES = 5000
FAMILY = 2
ALPHA_S = 0.05
BASE_SCORE_MEAN = 0.2
NOISE_AMPLITUDE = 0.8
POSITIVE_CERTIFICATE_THRESHOLD = 0.0
SEED = 2026112307
CHUNK_SIZE = 250


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def moving_window_scores(rng: np.random.Generator, *, rows: int,
                         blocks: int, dependence_order: int,
                         mean: float = BASE_SCORE_MEAN,
                         amplitude: float = NOISE_AMPLITUDE) -> np.ndarray:
    """Generate bounded scores with graph edges at distances <= m.

    For block i, the transformed score uses innovations i,...,i+m. Blocks
    separated by more than m therefore use disjoint innovations; residues
    modulo m+1 form jointly independent color classes.
    """
    rows = _positive_integer(rows, "rows")
    blocks = _positive_integer(blocks, "blocks")
    if (isinstance(dependence_order, bool)
            or not isinstance(dependence_order, int)
            or dependence_order < 0 or dependence_order >= blocks):
        raise ValueError("dependence_order must be an integer in [0, blocks)")
    if not math.isfinite(mean) or not math.isfinite(amplitude):
        raise ValueError("mean and amplitude must be finite")
    if amplitude < 0 or abs(mean) + amplitude > 1:
        raise ValueError("mean and amplitude must keep scores in [-1, 1]")
    width = dependence_order + 1
    innovations = rng.normal(size=(rows, blocks + dependence_order))
    rolling = np.zeros((rows, blocks), dtype=float)
    for offset in range(width):
        rolling += innovations[:, offset:offset + blocks]
    signal = np.tanh(rolling / math.sqrt(width))
    return mean + amplitude * signal


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("Wilson interval requires 0 <= successes <= positive trials")
    rate = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (rate + z2 / (2.0 * trials)) / denominator
    half = z * math.sqrt(rate * (1.0 - rate) / trials
                         + z2 / (4.0 * trials * trials)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _cell_seed(blocks: int, dependence_order: int,
               failure_probability: float) -> int:
    return (SEED + 1009 * blocks + 9176 * dependence_order
            + 17713 * round(100 * failure_probability))


def simulate_cell(*, blocks: int, dependence_order: int,
                  failure_probability: float, replicates: int = REPLICATES,
                  chunk_size: int = CHUNK_SIZE) -> tuple[dict, dict[str, np.ndarray]]:
    """Run one paired-family Monte Carlo cell and retain replicate summaries."""
    blocks = _positive_integer(blocks, "blocks")
    replicates = _positive_integer(replicates, "replicates")
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    if (isinstance(dependence_order, bool)
            or not isinstance(dependence_order, int)
            or not 0 <= dependence_order < blocks):
        raise ValueError("dependence_order must be in [0, blocks)")
    if not math.isfinite(failure_probability) or not 0 <= failure_probability < 1:
        raise ValueError("failure_probability must be finite and in [0, 1)")

    chi = dependence_order + 1
    epsilon_iid = dependence_penalty(
        blocks=blocks, chromatic_number=1, family=FAMILY, alpha_s=ALPHA_S)
    epsilon_graph = dependence_penalty(
        blocks=blocks, chromatic_number=chi, family=FAMILY, alpha_s=ALPHA_S)
    true_mean = (1.0 - failure_probability) * BASE_SCORE_MEAN - failure_probability
    rng = np.random.default_rng(_cell_seed(blocks, dependence_order,
                                           failure_probability))
    failed_count = np.empty(replicates, dtype=np.int16)
    valid_count = np.empty(replicates, dtype=np.int16)
    min_iid_lcb = np.empty(replicates, dtype=np.float32)
    min_graph_lcb = np.empty(replicates, dtype=np.float32)
    familywise_cover_iid = np.empty(replicates, dtype=bool)
    familywise_cover_graph = np.empty(replicates, dtype=bool)
    positive_iid = np.empty(replicates, dtype=bool)
    positive_graph = np.empty(replicates, dtype=bool)
    all_failed = np.empty(replicates, dtype=bool)
    replicate_coded_mean = np.empty(replicates, dtype=np.float32)
    replicate_valid_mean = np.full(replicates, np.nan, dtype=np.float32)
    cursor = 0
    while cursor < replicates:
        size = min(chunk_size, replicates - cursor)
        failures = rng.random((size, blocks)) < failure_probability
        scores = np.stack([
            moving_window_scores(rng, rows=size, blocks=blocks,
                                 dependence_order=dependence_order)
            for _ in range(FAMILY)
        ], axis=1)
        scores[failures[:, None, :].repeat(FAMILY, axis=1)] = -1.0
        means = scores.mean(axis=2)
        lower_iid = means - epsilon_iid
        lower_graph = means - epsilon_graph
        failed = failures.sum(axis=1)
        valid = blocks - failed
        sl = slice(cursor, cursor + size)
        failed_count[sl] = failed
        valid_count[sl] = valid
        min_iid_lcb[sl] = lower_iid.min(axis=1)
        min_graph_lcb[sl] = lower_graph.min(axis=1)
        familywise_cover_iid[sl] = np.all(lower_iid <= true_mean, axis=1)
        familywise_cover_graph[sl] = np.all(lower_graph <= true_mean, axis=1)
        positive_iid[sl] = np.all(lower_iid > POSITIVE_CERTIFICATE_THRESHOLD, axis=1)
        positive_graph[sl] = np.all(lower_graph > POSITIVE_CERTIFICATE_THRESHOLD, axis=1)
        all_failed[sl] = valid == 0
        replicate_coded_mean[sl] = means.mean(axis=1)
        valid_score_sum = np.where(failures[:, None, :], 0.0, scores).sum(axis=(1, 2))
        replicate_valid_mean[sl] = np.divide(
            valid_score_sum, valid * FAMILY,
            out=np.full(size, np.nan, dtype=float), where=valid > 0,
        )
        cursor += size

    valid_runs = ~all_failed
    cover_iid = int(familywise_cover_iid.sum())
    cover_graph = int(familywise_cover_graph.sum())
    cov_iid_lo, cov_iid_hi = _wilson_interval(cover_iid, replicates)
    cov_graph_lo, cov_graph_hi = _wilson_interval(cover_graph, replicates)
    row = {
        "blocks": blocks,
        "dependence_order_m": dependence_order,
        "graph_chromatic_number": chi,
        "failure_probability": failure_probability,
        "family_size_F": FAMILY,
        "alpha_s": ALPHA_S,
        "planned_replicates": replicates,
        "valid_replicates": int(valid_runs.sum()),
        "failed_replicates": 0,
        "abstained_all_blocks_failed": int(all_failed.sum()),
        "planned_pair_block_events": replicates * blocks,
        "valid_pair_block_events": int(valid_count.sum()),
        "failed_pair_block_events": int(failed_count.sum()),
        "abstained_pair_block_events": 0,
        "planned_family_block_scores": replicates * blocks * FAMILY,
        "valid_family_block_scores": int(valid_count.sum()) * FAMILY,
        "failed_family_block_scores": int(failed_count.sum()) * FAMILY,
        "abstained_family_block_scores": 0,
        "true_failure_augmented_mean": true_mean,
        "empirical_valid_score_mean": float(np.nanmean(replicate_valid_mean)),
        "empirical_coded_score_mean": float(replicate_coded_mean.mean()),
        "empirical_coded_mean_error": float(replicate_coded_mean.mean() - true_mean),
        "empirical_failure_fraction": float(failed_count.sum()
                                             / (replicates * blocks)),
        "iid_chi_assumption": 1,
        "graph_epsilon": epsilon_graph,
        "iid_epsilon": epsilon_iid,
        "familywise_coverage_iid": cover_iid / replicates,
        "familywise_coverage_iid_mc95_low": cov_iid_lo,
        "familywise_coverage_iid_mc95_high": cov_iid_hi,
        "familywise_coverage_graph": cover_graph / replicates,
        "familywise_coverage_graph_mc95_low": cov_graph_lo,
        "familywise_coverage_graph_mc95_high": cov_graph_hi,
        "positive_certificate_rate_iid": float(positive_iid.mean()),
        "positive_certificate_rate_graph": float(positive_graph.mean()),
        "median_min_lcb_iid": float(np.median(min_iid_lcb)),
        "median_min_lcb_graph": float(np.median(min_graph_lcb)),
        "seed": _cell_seed(blocks, dependence_order, failure_probability),
        "classification": "conditional synthetic Monte Carlo, not physical coverage",
    }
    arrays = {
        "failed_count": failed_count,
        "valid_count": valid_count,
        "min_iid_lcb": min_iid_lcb,
        "min_graph_lcb": min_graph_lcb,
        "familywise_cover_iid": familywise_cover_iid,
        "familywise_cover_graph": familywise_cover_graph,
        "positive_iid": positive_iid,
        "positive_graph": positive_graph,
        "all_failed": all_failed,
        "replicate_coded_mean": replicate_coded_mean,
        "replicate_valid_mean": replicate_valid_mean,
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

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 9.0), sharex=True,
                             sharey=True, layout="constrained")
    for row_index, blocks in enumerate(BLOCK_COUNTS):
        for col_index, failure_probability in enumerate(FAILURE_PROBABILITIES):
            ax = axes[row_index, col_index]
            subset = sorted((row for row in rows
                             if row["blocks"] == blocks
                             and row["failure_probability"] == failure_probability),
                            key=lambda row: row["dependence_order_m"])
            m_values = [row["dependence_order_m"] for row in subset]
            for method, color, lower_key, high_key, label in (
                ("IID assumption", "#D55E00", "familywise_coverage_iid_mc95_low",
                 "familywise_coverage_iid_mc95_high", "naive IID penalty (chi=1)"),
                ("Known graph", "#0072B2", "familywise_coverage_graph_mc95_low",
                 "familywise_coverage_graph_mc95_high", "graph penalty (chi=m+1)"),
            ):
                rate_key = ("familywise_coverage_iid" if method == "IID assumption"
                            else "familywise_coverage_graph")
                rate = [row[rate_key] for row in subset]
                lower = [row[rate_key] - row[lower_key] for row in subset]
                upper = [row[high_key] - row[rate_key] for row in subset]
                ax.errorbar(m_values, rate, yerr=[lower, upper], marker="o",
                            capsize=2, color=color, label=label)
            ax.axhline(1 - ALPHA_S, color="black", ls="--", lw=1,
                       label="nominal familywise coverage")
            ax.set_ylim(0.85, 1.005)
            ax.set_title(f"N={blocks}, coded failure probability={failure_probability:g}")
            ax.grid(alpha=0.25)
            if col_index == 0:
                ax.set_ylabel("familywise LCB coverage\n(95% Monte Carlo interval)")
            if row_index == len(BLOCK_COUNTS) - 1:
                ax.set_xlabel("true dependency range m; graph color bound = m+1")
            if row_index == 0 and col_index == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Known m-dependent synthetic scores; F=2, alpha_s=0.05; no physical coverage claim")
    fig.savefig(out / "coverage_by_dependence.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for blocks, color in zip(BLOCK_COUNTS, ("#D55E00", "#0072B2", "#009E73")):
        for failure_probability, style in zip(FAILURE_PROBABILITIES, ("-", "--")):
            subset = sorted((row for row in rows
                             if row["blocks"] == blocks
                             and row["failure_probability"] == failure_probability),
                            key=lambda row: row["dependence_order_m"])
            x = [row["dependence_order_m"] for row in subset]
            label = f"N={blocks}, failure={failure_probability:g}"
            axes[0].plot(x, [row["positive_certificate_rate_iid"] for row in subset],
                         color=color, ls=style, marker="o", label=label)
            axes[1].plot(x, [row["positive_certificate_rate_graph"] for row in subset],
                         color=color, ls=style, marker="o", label=label)
    for ax, title in zip(axes, ("IID penalty", "Known graph-color penalty")):
        ax.set(xlabel="true dependency range m",
               ylabel="fraction with both lower bounds > 0",
               title=title, ylim=(-0.03, 1.03))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Illustrative positive-certificate rate; power is not a physical threshold selection")
    fig.savefig(out / "positive_certificate_rate.png", dpi=180)
    plt.close(fig)


def build(output: str | Path = DEFAULT_OUT,
          replicates: int = REPLICATES) -> dict:
    replicates = _positive_integer(replicates, "replicates")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    replicate_arrays: dict[str, np.ndarray] = {}
    for blocks in BLOCK_COUNTS:
        for order in DEPENDENCE_ORDERS:
            for failure_probability in FAILURE_PROBABILITIES:
                row, arrays = simulate_cell(
                    blocks=blocks,
                    dependence_order=order,
                    failure_probability=failure_probability,
                    replicates=replicates,
                )
                rows.append(row)
                key = f"N{blocks}_m{order}_phi{int(100*failure_probability):02d}"
                for name, values in arrays.items():
                    replicate_arrays[f"{key}_{name}"] = values

    protocol = {
        "schema": "pion-bounded-block-risk-coverage-mc-v1",
        "classification": "synthetic conditional calibration experiment",
        "physical_score": None,
        "blocks": list(BLOCK_COUNTS),
        "dependence_orders": list(DEPENDENCE_ORDERS),
        "graph_chromatic_number": "m + 1; residues modulo m+1 use disjoint innovation windows",
        "failure_probabilities": list(FAILURE_PROBABILITIES),
        "replicates_per_cell": replicates,
        "family": FAMILY,
        "alpha_s": ALPHA_S,
        "valid_score_mean": BASE_SCORE_MEAN,
        "bounded_noise": "0.8*tanh(sum of m+1 iid standard-normal innovations / sqrt(m+1))",
        "failure_policy": "shared paired-block failure mask across F arms; score coded to -1, retained in N",
        "estimand_mean": "(1-phi)*valid_score_mean - phi",
        "comparison": ["naive IID chi=1", "known graph-color bound chi=m+1"],
        "anchor": "none; alpha_A=0 and anchor bias bound=0 for this algebra-only study",
        "positive_certificate_threshold": POSITIVE_CERTIFICATE_THRESHOLD,
        "seed_rule": "seed = base + 1009*N + 9176*m + 17713*round(100*phi)",
        "source_sha256": {
            "bounded_block_risk_coverage_mc.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependency_graph_bound.py": hashlib.sha256(
                Path(__file__).with_name("dependency_graph_bound.py").read_bytes()).hexdigest(),
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (out / "snapshot_bounded_block_risk_coverage_mc.py").write_bytes(
        Path(__file__).read_bytes())
    dependency_source = Path(__file__).with_name("dependency_graph_bound.py")
    (out / "snapshot_dependency_graph_bound.py").write_bytes(
        dependency_source.read_bytes())
    _write_csv(out / "cell_summary.csv", rows)
    np.savez_compressed(out / "mc_replicate_summary.npz", **replicate_arrays)
    _write_plots(out, rows)

    planned_replicates = len(rows) * replicates
    valid_replicates = sum(row["valid_replicates"] for row in rows)
    abstained_replicates = sum(row["abstained_all_blocks_failed"] for row in rows)
    planned_pair_blocks = sum(row["planned_pair_block_events"] for row in rows)
    valid_pair_blocks = sum(row["valid_pair_block_events"] for row in rows)
    failed_pair_blocks = sum(row["failed_pair_block_events"] for row in rows)
    planned_family_scores = sum(row["planned_family_block_scores"] for row in rows)
    valid_family_scores = sum(row["valid_family_block_scores"] for row in rows)
    failed_family_scores = sum(row["failed_family_block_scores"] for row in rows)
    denominator_rows = [
        {"unit": "synthetic Monte Carlo cell", "planned": len(rows),
         "valid": len(rows), "failed": 0, "abstained": 0,
         "interpretation": "All prespecified simulation cells completed."},
        {"unit": "Monte Carlo replicate", "planned": planned_replicates,
         "valid": valid_replicates, "failed": 0,
         "abstained": abstained_replicates,
         "interpretation": "An all-failure replicate is operationally abstained; its coded-score algebraic bound remains in replicate arrays."},
        {"unit": "paired planned block event", "planned": planned_pair_blocks,
         "valid": valid_pair_blocks, "failed": failed_pair_blocks,
         "abstained": 0,
         "interpretation": "Failures are hypothetical Bernoulli events, not observed measurement failures; all remain in N and are scored -1."},
        {"unit": "candidate-family block score", "planned": planned_family_scores,
         "valid": valid_family_scores, "failed": failed_family_scores,
         "abstained": 0,
         "interpretation": "F=2 scores share paired failure events; this denominator is not a physical block count."},
    ]
    _write_csv(out / "denominator_summary.csv", denominator_rows)
    summary = {
        "schema": "pion-bounded-block-risk-coverage-mc-summary-v1",
        "status": "SYNTHETIC_M_DEPENDENT_COVERAGE_DIAGNOSTIC",
        "physical_score": None,
        "data_access": "No physical score, correlator, anchor, sealed split, GPU, or QUDA data were accessed.",
        "cells": len(rows),
        "replicates_per_cell": replicates,
        "nominal_familywise_coverage": 1.0 - ALPHA_S,
        "graph_coverage_min": min(row["familywise_coverage_graph"] for row in rows),
        "iid_coverage_min": min(row["familywise_coverage_iid"] for row in rows),
        "abstained_all_failure_replicates": abstained_replicates,
        "denominator_summary": denominator_rows,
        "source_sha256": protocol["source_sha256"],
        "files": ["README.md", "protocol.json", "cell_summary.csv",
                  "denominator_summary.csv", "mc_replicate_summary.npz",
                  "coverage_by_dependence.png", "positive_certificate_rate.png",
                  "snapshot_bounded_block_risk_coverage_mc.py",
                  "snapshot_dependency_graph_bound.py", "summary.json",
                  "artifact_hashes.json"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    readme = [
        "# Bounded block-risk conditional coverage Monte Carlo",
        "",
        "**SYNTHETIC METHOD DIAGNOSTIC ONLY. `physical_score=null`.** The generator has a known finite dependency graph and known score mean. This study does not estimate LatticeArena/Pion dependence, anchor coverage, effect size, or physical risk.",
        "",
        "## Prespecified generator and estimand",
        "",
        f"The grid crosses `N={list(BLOCK_COUNTS)}`, dependence ranges `m={list(DEPENDENCE_ORDERS)}`, and hypothetical failure probabilities `{list(FAILURE_PROBABILITIES)}` with `{replicates}` Monte Carlo replicates per cell. There are `F={FAMILY}` candidate score families and `alpha_s={ALPHA_S}`. Each valid score is `mu + 0.8*tanh(mean of m+1 iid Gaussian innovations)` with `mu={BASE_SCORE_MEAN}`; this lies in `[-1,1]` and has expectation `mu` by symmetry. Blocks at distance greater than `m` use disjoint innovations, so the graph with edges up to range `m` is valid and its residue coloring uses `chi=m+1` jointly independent classes.",
        "",
        "A Bernoulli failure mask is shared across the paired candidate families. Failed planned blocks are retained and assigned score `-1`, so the target mean is `(1-phi)*mu - phi`. No anchor error is included (`alpha_A=0`). The study compares the lower bound using the correct graph color count with the same formula incorrectly assuming IID (`chi=1`). The two arms share block-failure indicators; Bonferroni `F=2` is used regardless of cross-family dependence.",
        "",
        "## Figures",
        "",
        "![Coverage by dependence order](coverage_by_dependence.png)",
        "",
        "Points are simulated familywise coverage for both one-sided lower bounds; whiskers are 95% Monte Carlo Wilson intervals, not physics confidence intervals. The dashed line is the nominal 95% familywise level. The graph-color result tests the formula under the constructed m-dependent process. The IID curve is a deliberate misspecification control; any undercoverage is conditional on this synthetic generator.",
        "",
        "![Positive lower-bound rate](positive_certificate_rate.png)",
        "",
        "This plot shows how often both lower bounds exceed zero. It is only the power/utility tradeoff for the arbitrary synthetic mean and zero threshold: a valid graph bound can retain coverage while abstaining from a positive conclusion. Zero is not a chosen Pion effect threshold.",
        "",
        "## Denominators and limits",
        "",
        "`cell_summary.csv` reports every cell, both coverage rates and Monte Carlo intervals, positive-certificate rate, and planned/valid/failed/abstained counts at replicate, paired-block, and family-score units. `mc_replicate_summary.npz` retains each replicate's block failure count and lower-bound/coverage outcomes. All-failure replicates are operational abstentions; the algebraic coded-score bound is separately retained for coverage auditing.",
        "",
        "The coverage claim is conditional on a correct dependency graph and score bound. This simulated graph is known by construction; real trajectory spacing, block bootstrap or observed autocorrelation does not establish it. The shared-anchor bias correction is omitted here and still requires its independent coverage receipt. Clipping and coding failures define the simulated estimand; neither operation restores the original unbounded ensemble-risk.",
        "",
        "The CSV also reports empirical valid-score means, failure-coded means, and the coded-mean deviation from its known target. These are generator diagnostics; they do not calibrate any project-specific score distribution.",
        "",
        "## Rebuild",
        "",
        "Use a fresh output directory; existing paths are never overwritten:",
        "",
        "```text",
        "python -B -m tasks.pion_projection.bounded_block_risk_coverage_mc --output NEW_DIRECTORY",
        "```",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n")
    artifact_names = ["README.md", "protocol.json", "cell_summary.csv",
                      "denominator_summary.csv", "mc_replicate_summary.npz",
                      "coverage_by_dependence.png", "positive_certificate_rate.png",
                      "snapshot_bounded_block_risk_coverage_mc.py",
                      "snapshot_dependency_graph_bound.py", "summary.json"]
    hashes = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
              for name in artifact_names}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    args = parser.parse_args()
    summary = build(args.output, args.replicates)
    print(json.dumps({
        "status": summary["status"],
        "cells": summary["cells"],
        "replicates_per_cell": summary["replicates_per_cell"],
        "graph_coverage_min": summary["graph_coverage_min"],
        "iid_coverage_min": summary["iid_coverage_min"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
