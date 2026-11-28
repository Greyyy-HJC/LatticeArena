"""Synthetic coverage study with both dependent scores and a shared anchor.

Goal: exercise the alpha_s/alpha_anchor split under a known m-dependent score
      process and an independent, finite-sample anchor confidence interval.
Inputs: fixed synthetic grid, seed 2026112311, 5,000 replicates per cell, CPU only.
Deliverable: no-clobber report with figures, raw replicate outcomes and hashes.
Check: unit tests cover the Lipschitz construction, denominators, and artifact set.

This study cannot estimate real block dependence, anchor error, or a physical
Pion score. It deliberately leaves physical_score null.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
from pathlib import Path
from statistics import NormalDist

import numpy as np

from .bounded_block_risk_coverage_mc import moving_window_scores
from .bounded_block_risk_score import bounded_block_scores
from .dependency_graph_bound import dependence_penalty


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/pion_projection/bounded_block_risk_anchor_coverage_mc_20261123_v1"
BLOCK_COUNTS = (128, 512)
DEPENDENCE_ORDERS = (0, 15)
FAILURE_PROBABILITIES = (0.0, 0.1)
ANCHOR_SAMPLE_SIZES = (32, 128)
REPLICATES = 5000
FAMILY = 2
ALPHA_TOTAL = 0.05
ALPHA_SCORE = 0.04
ALPHA_ANCHOR = 0.01
ANCHOR_SIGMA = 0.05
ANCHOR_TARGET = 0.0
SCORE_SCALE_R = 1.0
BASE_SCORE_MEAN = 0.2
NOISE_AMPLITUDE = 0.8
POSITIVE_THRESHOLD = 0.0
SEED = 2026112311
CHUNK_SIZE = 250


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _wilson_interval(successes: int, trials: int,
                     z: float = 1.959963984540054) -> tuple[float, float]:
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
               failure_probability: float, anchor_samples: int) -> int:
    return (SEED + 1009 * blocks + 9176 * dependence_order
            + 17713 * round(100 * failure_probability) + 997 * anchor_samples)


def anchor_bound(*, samples: int, sigma: float = ANCHOR_SIGMA,
                 alpha_anchor: float = ALPHA_ANCHOR) -> float:
    """Two-sided exact-normal error radius for a mean with known sigma."""
    n = _positive_integer(samples, "samples")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(alpha_anchor) or not 0 < alpha_anchor < 1:
        raise ValueError("alpha_anchor must be finite and in (0, 1)")
    z = NormalDist().inv_cdf(1.0 - alpha_anchor / 2.0)
    return z * sigma / math.sqrt(n)


def simulate_cell(*, blocks: int, dependence_order: int,
                  failure_probability: float, anchor_samples: int,
                  replicates: int = REPLICATES,
                  chunk_size: int = CHUNK_SIZE) -> tuple[dict, dict[str, np.ndarray]]:
    """Simulate one cell and retain every replicate-level diagnostic."""
    blocks = _positive_integer(blocks, "blocks")
    anchor_samples = _positive_integer(anchor_samples, "anchor_samples")
    replicates = _positive_integer(replicates, "replicates")
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    if (isinstance(dependence_order, bool)
            or not isinstance(dependence_order, int)
            or not 0 <= dependence_order < blocks):
        raise ValueError("dependence_order must be in [0, blocks)")
    if (not math.isfinite(failure_probability)
            or not 0 <= failure_probability < 1):
        raise ValueError("failure_probability must be finite and in [0, 1)")
    if not math.isclose(ALPHA_SCORE + ALPHA_ANCHOR, ALPHA_TOTAL):
        raise ValueError("score and anchor alpha allocations must sum to alpha_total")

    chi_graph = dependence_order + 1
    epsilon_iid = dependence_penalty(
        blocks=blocks, chromatic_number=1, family=FAMILY, alpha_s=ALPHA_SCORE,
    )
    epsilon_graph = dependence_penalty(
        blocks=blocks, chromatic_number=chi_graph,
        family=FAMILY, alpha_s=ALPHA_SCORE,
    )
    anchor_se = ANCHOR_SIGMA / math.sqrt(anchor_samples)
    anchor_radius = anchor_bound(samples=anchor_samples)
    anchor_penalty = 2.0 * anchor_radius / SCORE_SCALE_R
    true_mean = ((1.0 - failure_probability) * BASE_SCORE_MEAN
                 - failure_probability)
    rng = np.random.default_rng(_cell_seed(
        blocks, dependence_order, failure_probability, anchor_samples,
    ))

    anchor_error = np.empty(replicates, dtype=np.float64)
    failed_count = np.empty(replicates, dtype=np.int32)
    anchor_covered = np.empty(replicates, dtype=bool)
    abstained_all_blocks = np.empty(replicates, dtype=bool)
    minimum_graph_lcb = np.empty(replicates, dtype=np.float32)
    minimum_iid_lcb = np.empty(replicates, dtype=np.float32)
    false_certificate_graph = np.empty(replicates, dtype=bool)
    false_certificate_iid = np.empty(replicates, dtype=bool)
    false_graph_given_anchor = np.empty(replicates, dtype=bool)
    false_iid_given_anchor = np.empty(replicates, dtype=bool)
    positive_certificate_graph = np.empty(replicates, dtype=bool)
    positive_certificate_iid = np.empty(replicates, dtype=bool)

    cursor = 0
    while cursor < replicates:
        size = min(chunk_size, replicates - cursor)
        failures = rng.random((size, blocks)) < failure_probability
        anchor_errors = rng.normal(0.0, anchor_se, size=size)
        family_scores = np.stack([
            moving_window_scores(
                rng,
                rows=size,
                blocks=blocks,
                dependence_order=dependence_order,
                mean=BASE_SCORE_MEAN,
                amplitude=NOISE_AMPLITUDE,
            )
            for _ in range(FAMILY)
        ], axis=1)

        # A*=0, y_candidate-y_reference=R, and D(A*)=R^2*q imply
        # D(A_hat)/R^2 = q + 2*(A_hat-A*)/R. The anchor draw is shared by F.
        raw_differences = (
            SCORE_SCALE_R**2 * family_scores
            + 2.0 * SCORE_SCALE_R * anchor_errors[:, None, None]
        )
        valid_mask = np.broadcast_to(~failures[:, None, :], raw_differences.shape)
        scores = bounded_block_scores(
            raw_differences,
            valid_blocks=valid_mask,
            score_scale_R=SCORE_SCALE_R,
        )
        means = scores.mean(axis=-1)
        lower_graph = means - epsilon_graph - anchor_penalty
        lower_iid = means - epsilon_iid - anchor_penalty
        n_failed = failures.sum(axis=1)
        n_valid = blocks - n_failed
        abstain = n_valid == 0
        issue_bound = ~abstain
        graph_false = issue_bound & np.any(lower_graph > true_mean, axis=1)
        iid_false = issue_bound & np.any(lower_iid > true_mean, axis=1)
        anchor_ok = np.abs(anchor_errors) <= anchor_radius

        sl = slice(cursor, cursor + size)
        anchor_error[sl] = anchor_errors
        failed_count[sl] = n_failed
        anchor_covered[sl] = anchor_ok
        abstained_all_blocks[sl] = abstain
        minimum_graph_lcb[sl] = lower_graph.min(axis=1)
        minimum_iid_lcb[sl] = lower_iid.min(axis=1)
        false_certificate_graph[sl] = graph_false
        false_certificate_iid[sl] = iid_false
        false_graph_given_anchor[sl] = graph_false & anchor_ok
        false_iid_given_anchor[sl] = iid_false & anchor_ok
        positive_certificate_graph[sl] = (
            issue_bound & np.all(lower_graph > POSITIVE_THRESHOLD, axis=1)
        )
        positive_certificate_iid[sl] = (
            issue_bound & np.all(lower_iid > POSITIVE_THRESHOLD, axis=1)
        )
        cursor += size

    issued_count = int((~abstained_all_blocks).sum())
    anchor_covered_count = int(anchor_covered.sum())
    graph_false_count = int(false_certificate_graph.sum())
    iid_false_count = int(false_certificate_iid.sum())
    graph_false_anchor_count = int(false_graph_given_anchor.sum())
    iid_false_anchor_count = int(false_iid_given_anchor.sum())
    anchor_coverage_ci = _wilson_interval(anchor_covered_count, replicates)
    graph_false_ci = _wilson_interval(graph_false_count, replicates)
    iid_false_ci = _wilson_interval(iid_false_count, replicates)
    graph_false_anchor_ci = _wilson_interval(
        graph_false_anchor_count, max(anchor_covered_count, 1),
    )
    iid_false_anchor_ci = _wilson_interval(
        iid_false_anchor_count, max(anchor_covered_count, 1),
    )
    issued_graph_successes = issued_count - graph_false_count
    issued_iid_successes = issued_count - iid_false_count
    graph_issued_ci = (
        _wilson_interval(issued_graph_successes, issued_count)
        if issued_count else (None, None)
    )
    iid_issued_ci = (
        _wilson_interval(issued_iid_successes, issued_count)
        if issued_count else (None, None)
    )
    failed_events = int(failed_count.sum())
    planned_family_scores = replicates * blocks * FAMILY
    row = {
        "blocks": blocks,
        "dependence_order_m": dependence_order,
        "graph_chromatic_number": chi_graph,
        "failure_probability": failure_probability,
        "anchor_sample_size": anchor_samples,
        "anchor_sigma_known": ANCHOR_SIGMA,
        "anchor_standard_error": anchor_se,
        "anchor_radius_B": anchor_radius,
        "relative_anchor_bound_B_over_R": anchor_radius / SCORE_SCALE_R,
        "alpha_total": ALPHA_TOTAL,
        "alpha_score": ALPHA_SCORE,
        "alpha_anchor": ALPHA_ANCHOR,
        "nominal_familywise_error_bound": ALPHA_TOTAL,
        "nominal_anchor_coverage": 1.0 - ALPHA_ANCHOR,
        "family_size_F": FAMILY,
        "planned_replicates": replicates,
        "valid_replicates": issued_count,
        "failed_replicates": 0,
        "abstained_all_blocks_failed": int(abstained_all_blocks.sum()),
        "planned_family_block_scores": planned_family_scores,
        "valid_family_block_scores": (replicates * blocks - failed_events) * FAMILY,
        "failed_family_block_scores": failed_events * FAMILY,
        "abstained_family_block_scores": 0,
        "planned_paired_block_events": replicates * blocks,
        "valid_paired_block_events": replicates * blocks - failed_events,
        "failed_paired_block_events": failed_events,
        "abstained_paired_block_events": 0,
        "planned_anchor_estimates": replicates,
        "valid_anchor_estimates": replicates,
        "failed_anchor_estimates": 0,
        "abstained_anchor_estimates": 0,
        "planned_anchor_observations": replicates * anchor_samples,
        "valid_anchor_observations": replicates * anchor_samples,
        "failed_anchor_observations": 0,
        "abstained_anchor_observations": 0,
        "true_failure_augmented_mean": true_mean,
        "empirical_failure_fraction": failed_events / (replicates * blocks),
        "epsilon_iid": epsilon_iid,
        "epsilon_graph": epsilon_graph,
        "anchor_penalty_2B_over_R": anchor_penalty,
        "anchor_interval_coverage": anchor_covered_count / replicates,
        "anchor_interval_coverage_mc95_low": anchor_coverage_ci[0],
        "anchor_interval_coverage_mc95_high": anchor_coverage_ci[1],
        "graph_false_certificate_count": graph_false_count,
        "graph_false_certificate_rate_planned": graph_false_count / replicates,
        "graph_false_certificate_mc95_low": graph_false_ci[0],
        "graph_false_certificate_mc95_high": graph_false_ci[1],
        "graph_false_certificate_given_anchor_covered": (
            graph_false_anchor_count / anchor_covered_count
            if anchor_covered_count else None
        ),
        "graph_false_given_anchor_mc95_low": graph_false_anchor_ci[0],
        "graph_false_given_anchor_mc95_high": graph_false_anchor_ci[1],
        "graph_numeric_coverage_given_issued": (
            issued_graph_successes / issued_count if issued_count else None
        ),
        "graph_numeric_coverage_issued_mc95_low": graph_issued_ci[0],
        "graph_numeric_coverage_issued_mc95_high": graph_issued_ci[1],
        "iid_false_certificate_count": iid_false_count,
        "iid_false_certificate_rate_planned": iid_false_count / replicates,
        "iid_false_certificate_mc95_low": iid_false_ci[0],
        "iid_false_certificate_mc95_high": iid_false_ci[1],
        "iid_false_certificate_given_anchor_covered": (
            iid_false_anchor_count / anchor_covered_count
            if anchor_covered_count else None
        ),
        "iid_false_given_anchor_mc95_low": iid_false_anchor_ci[0],
        "iid_false_given_anchor_mc95_high": iid_false_anchor_ci[1],
        "iid_numeric_coverage_given_issued": (
            issued_iid_successes / issued_count if issued_count else None
        ),
        "iid_numeric_coverage_issued_mc95_low": iid_issued_ci[0],
        "iid_numeric_coverage_issued_mc95_high": iid_issued_ci[1],
        "graph_positive_certificate_rate": float(positive_certificate_graph.mean()),
        "iid_positive_certificate_rate": float(positive_certificate_iid.mean()),
        "median_min_lcb_graph": float(np.median(minimum_graph_lcb)),
        "median_min_lcb_iid": float(np.median(minimum_iid_lcb)),
        "seed": _cell_seed(blocks, dependence_order,
                           failure_probability, anchor_samples),
        "classification": "synthetic joint coverage and anchor uncertainty diagnostic",
    }
    arrays = {
        "anchor_error": anchor_error,
        "failed_count": failed_count,
        "anchor_bound_covered": anchor_covered,
        "abstained_all_blocks": abstained_all_blocks,
        "minimum_graph_lcb": minimum_graph_lcb,
        "minimum_iid_lcb": minimum_iid_lcb,
        "false_certificate_graph": false_certificate_graph,
        "false_certificate_iid": false_certificate_iid,
        "false_graph_given_anchor": false_graph_given_anchor,
        "false_iid_given_anchor": false_iid_given_anchor,
        "positive_certificate_graph": positive_certificate_graph,
        "positive_certificate_iid": positive_certificate_iid,
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


def _write_denominators(path: Path, rows: list[dict]) -> list[dict]:
    denominator_rows = []
    for row in rows:
        cell = (f"N{row['blocks']}_m{row['dependence_order_m']}"
                f"_phi{row['failure_probability']}_nA{row['anchor_sample_size']}")
        for unit, planned_key, valid_key, failed_key, abstained_key in (
            ("MC replicate", "planned_replicates", "valid_replicates",
             "failed_replicates", "abstained_all_blocks_failed"),
            ("paired block event", "planned_paired_block_events",
             "valid_paired_block_events", "failed_paired_block_events",
             "abstained_paired_block_events"),
            ("family block score", "planned_family_block_scores",
             "valid_family_block_scores", "failed_family_block_scores",
             "abstained_family_block_scores"),
            ("anchor estimate", "planned_anchor_estimates",
             "valid_anchor_estimates", "failed_anchor_estimates",
             "abstained_anchor_estimates"),
            ("anchor observation", "planned_anchor_observations",
             "valid_anchor_observations", "failed_anchor_observations",
             "abstained_anchor_observations"),
        ):
            denominator_rows.append({
                "cell": cell,
                "unit": unit,
                "planned": row[planned_key],
                "valid": row[valid_key],
                "failed": row[failed_key],
                "abstained": row[abstained_key],
                "interpretation": (
                    "All-failed MC replicates abstain; failed block scores remain -1 in planned N."
                    if unit == "MC replicate"
                    else "Synthetic denominator; not a measured physical sample count."
                ),
            })
    _write_csv(path, denominator_rows)
    return denominator_rows


def _write_plot(out: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    conditions = sorted({(row["blocks"], row["dependence_order_m"],
                          row["failure_probability"]) for row in rows})
    anchor_sizes = list(ANCHOR_SAMPLE_SIZES)
    row_labels = [f"N={n}, m={m}, phi={phi:g}" for n, m, phi in conditions]

    def matrix(rate_key: str) -> np.ndarray:
        lookup = {(r["blocks"], r["dependence_order_m"],
                   r["failure_probability"], r["anchor_sample_size"]): r
                  for r in rows}
        return np.asarray([[lookup[(n, m, phi, n_a)][rate_key]
                            for n_a in anchor_sizes]
                           for n, m, phi in conditions], dtype=float)

    def ci_matrices(low_key: str, high_key: str) -> tuple[np.ndarray, np.ndarray]:
        lookup = {(r["blocks"], r["dependence_order_m"],
                   r["failure_probability"], r["anchor_sample_size"]): r
                  for r in rows}
        low = np.asarray([[lookup[(n, m, phi, n_a)][low_key]
                           for n_a in anchor_sizes]
                          for n, m, phi in conditions], dtype=float)
        high = np.asarray([[lookup[(n, m, phi, n_a)][high_key]
                            for n_a in anchor_sizes]
                           for n, m, phi in conditions], dtype=float)
        return low, high

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), layout="constrained")
    panels = [
        (axes[0, 0], "graph_false_certificate_rate_planned",
         "graph_false_certificate_mc95_low", "graph_false_certificate_mc95_high",
         "Graph bound: false-certificate rate", 0.10),
        (axes[0, 1], "iid_false_certificate_rate_planned",
         "iid_false_certificate_mc95_low", "iid_false_certificate_mc95_high",
         "IID misspecification: false-certificate rate", 0.20),
        (axes[1, 0], "graph_positive_certificate_rate", None, None,
         "Graph bound: positive certificate rate (threshold 0)", 1.0),
    ]
    for ax, rate_key, low_key, high_key, title, vmax in panels:
        values = matrix(rate_key)
        im = ax.imshow(values, vmin=0, vmax=vmax, cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_yticks(np.arange(len(row_labels)), labels=row_labels, fontsize=8)
        ax.set_xticks(np.arange(len(anchor_sizes)),
                      labels=[f"n_A={n}" for n in anchor_sizes])
        if low_key is None:
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    ax.text(j, i, f"{100*values[i,j]:.1f}%",
                            ha="center", va="center",
                            color="black" if values[i,j] > vmax * .45 else "white",
                            fontsize=8)
        else:
            lows, highs = ci_matrices(low_key, high_key)
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    ax.text(j, i,
                            f"{100*values[i,j]:.2f}%\n[{100*lows[i,j]:.2f}%,{100*highs[i,j]:.2f}%]",
                            ha="center", va="center",
                            color="black" if values[i,j] > vmax * .45 else "white",
                            fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     format=PercentFormatter(xmax=1.0))

    ax = axes[1, 1]
    anchor_rates = []
    anchor_lows = []
    anchor_highs = []
    for n_a in anchor_sizes:
        selected = [r for r in rows if r["anchor_sample_size"] == n_a]
        total = sum(r["planned_replicates"] for r in selected)
        covered = round(sum(r["anchor_interval_coverage"]
                            * r["planned_replicates"] for r in selected))
        lo, hi = _wilson_interval(covered, total)
        anchor_rates.append(covered / total)
        anchor_lows.append(lo)
        anchor_highs.append(hi)
    x = np.arange(len(anchor_sizes))
    ax.errorbar(x, anchor_rates,
                yerr=[np.asarray(anchor_rates)-anchor_lows,
                      np.asarray(anchor_highs)-anchor_rates],
                fmt="o", capsize=4, color="#0072B2")
    ax.axhline(1.0 - ALPHA_ANCHOR, color="#D55E00", ls="--",
               label=f"nominal {1-ALPHA_ANCHOR:.0%}")
    ax.set(xticks=x, xticklabels=[f"n_A={n}" for n in anchor_sizes],
           ylim=(0.97, 1.002), ylabel="empirical coverage",
           title="Shared anchor interval coverage (pooled cells)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.suptitle(
        "Synthetic bounded block-risk calibration with anchor uncertainty\n"
        "F=2, alpha_score=0.04, alpha_anchor=0.01; all rates are Monte Carlo diagnostics",
        fontsize=12,
    )
    fig.savefig(out / "coverage_and_utility.png", dpi=180)
    plt.close(fig)


def build(output: str | Path = DEFAULT_OUT,
          replicates: int = REPLICATES) -> dict:
    replicates = _positive_integer(replicates, "replicates")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    replicate_arrays: dict[str, list[np.ndarray]] = {}
    cell_order = []
    for n in BLOCK_COUNTS:
        for m in DEPENDENCE_ORDERS:
            for phi in FAILURE_PROBABILITIES:
                for n_a in ANCHOR_SAMPLE_SIZES:
                    row, arrays = simulate_cell(
                        blocks=n,
                        dependence_order=m,
                        failure_probability=phi,
                        anchor_samples=n_a,
                        replicates=replicates,
                    )
                    rows.append(row)
                    cell_order.append([n, m, phi, n_a])
                    for name, values in arrays.items():
                        replicate_arrays.setdefault(name, []).append(values)

    _write_csv(out / "cell_summary.csv", rows)
    denominator_rows = _write_denominators(out / "denominator_summary.csv", rows)
    np.savez_compressed(
        out / "replicate_summary.npz",
        cell_order=np.asarray(cell_order, dtype=np.float64),
        **{name: np.stack(values) for name, values in replicate_arrays.items()},
    )
    protocol = {
        "schema": "pion-bounded-block-risk-anchor-coverage-mc-protocol-v1",
        "status": "SYNTHETIC_METHOD_DIAGNOSTIC_ONLY",
        "physical_score": None,
        "data_access": "No physical correlator, anchor, ensemble, sealed split, GPU, or QUDA data accessed.",
        "grid": {
            "planned_blocks_N": list(BLOCK_COUNTS),
            "known_m_dependence_order": list(DEPENDENCE_ORDERS),
            "hypothetical_failure_probability": list(FAILURE_PROBABILITIES),
            "independent_anchor_sample_size": list(ANCHOR_SAMPLE_SIZES),
            "replicates_per_cell": replicates,
        },
        "estimand": {
            "families_F": FAMILY,
            "true_anchor_A_star": ANCHOR_TARGET,
            "score_scale_R": SCORE_SCALE_R,
            "valid_score_mean": BASE_SCORE_MEAN,
            "noise_amplitude": NOISE_AMPLITUDE,
            "failure_score": -1.0,
            "true_failure_augmented_mean": "(1-phi)*0.2 - phi",
        },
        "anchor_model": {
            "iid_normal_observations": True,
            "known_sigma": ANCHOR_SIGMA,
            "two_sided_alpha_A": ALPHA_ANCHOR,
            "bound": "B=z_(1-alpha_A/2)*sigma/sqrt(n_anchor)",
            "independent_of_score_innovations": True,
        },
        "confidence_allocation": {
            "alpha_total": ALPHA_TOTAL,
            "alpha_score": ALPHA_SCORE,
            "alpha_anchor": ALPHA_ANCHOR,
            "family_size": FAMILY,
        },
        "comparators": ["known graph chi=m+1", "deliberately misspecified IID chi=1"],
        "positive_threshold": POSITIVE_THRESHOLD,
        "seed": SEED,
        "chunk_size": CHUNK_SIZE,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "theoretical_event": (
            "P(any issued graph LCB exceeds its true failure-augmented mean) <= "
            "alpha_score + alpha_anchor under the declared synthetic assumptions"
        ),
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    source_paths = {
        "snapshot_anchor_coverage_mc.py": Path(__file__),
        "snapshot_bounded_block_risk_score.py": Path(__file__).with_name("bounded_block_risk_score.py"),
        "snapshot_bounded_block_risk_coverage_mc.py": Path(__file__).with_name("bounded_block_risk_coverage_mc.py"),
        "snapshot_dependency_graph_bound.py": Path(__file__).with_name("dependency_graph_bound.py"),
        "snapshot_core_numerics.py": (
            Path(__file__).resolve().parents[2] / "core/numerics.py"
        ),
    }
    for target, source in source_paths.items():
        shutil.copyfile(source, out / target)

    _write_plot(out, rows)
    planned_replicates = sum(r["planned_replicates"] for r in rows)
    summary = {
        "schema": "pion-bounded-block-risk-anchor-coverage-mc-summary-v1",
        "status": "SYNTHETIC_JOINT_DEPENDENCE_AND_ANCHOR_COVERAGE_DIAGNOSTIC",
        "physical_score": None,
        "data_access": protocol["data_access"],
        "cells": len(rows),
        "replicates_per_cell": replicates,
        "planned_replicates": planned_replicates,
        "alpha_total": ALPHA_TOTAL,
        "alpha_score": ALPHA_SCORE,
        "alpha_anchor": ALPHA_ANCHOR,
        "graph_false_certificate_rate_max": max(r["graph_false_certificate_rate_planned"] for r in rows),
        "graph_false_certificate_mc95_upper_max": max(r["graph_false_certificate_mc95_high"] for r in rows),
        "iid_false_certificate_rate_max": max(r["iid_false_certificate_rate_planned"] for r in rows),
        "anchor_interval_coverage_min": min(r["anchor_interval_coverage"] for r in rows),
        "anchor_interval_coverage_max": max(r["anchor_interval_coverage"] for r in rows),
        "denominator_summary": denominator_rows,
        "files": [
            "README.md", "protocol.json", "cell_summary.csv",
            "denominator_summary.csv", "replicate_summary.npz",
            "coverage_and_utility.png", "snapshot_anchor_coverage_mc.py",
            "snapshot_bounded_block_risk_score.py",
            "snapshot_bounded_block_risk_coverage_mc.py",
            "snapshot_dependency_graph_bound.py", "snapshot_core_numerics.py",
            "summary.json",
            "artifact_hashes.json",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    readme = [
        "# Bounded block-risk coverage with shared anchor uncertainty",
        "",
        "**SYNTHETIC METHOD DIAGNOSTIC ONLY. `physical_score=null`.** No project measurement, correlator, E anchor, sealed split, GPU, or QUDA data are read.",
        "",
        "## Model and prespecified calibration",
        "",
        f"The grid has `N={list(BLOCK_COUNTS)}`, known dependence range `m={list(DEPENDENCE_ORDERS)}`, hypothetical failures `{list(FAILURE_PROBABILITIES)}`, independent anchor sample sizes `{list(ANCHOR_SAMPLE_SIZES)}`, and `{replicates}` replicates per cell. There are `F={FAMILY}` paired families with `alpha_score={ALPHA_SCORE}`, `alpha_anchor={ALPHA_ANCHOR}`, and `alpha_total={ALPHA_TOTAL}`.",
        "",
        f"For each family, the true-anchor valid score is `q_i=0.2+0.8*tanh(mean of m+1 iid Gaussian innovations)`; it is bounded in `[-1,1]` and has mean 0.2. Innovation windows overlap for at most `m` positions, so residue classes modulo `m+1` are jointly independent. A shared Bernoulli failure mask is used for both families; failures remain in planned `N` and receive `-1`, giving true mean `(1-phi)*0.2-phi`.",
        "",
        f"The synthetic anchor target is `A*=0`. Each replicate obtains `n_A` independent normal observations with known `sigma={ANCHOR_SIGMA}` and estimates `A_hat` by their mean. The two-sided exact-normal radius is `B=z_(1-alpha_anchor/2)*sigma/sqrt(n_A)`, so its nominal coverage is `{1-ALPHA_ANCHOR:.0%}`. The model fixes `y_candidate-y_reference=R` with `R={SCORE_SCALE_R}` and constructs `D(A*)=R^2*q`; hence `D(A_hat)/R^2=q+2*(A_hat-A*)/R`. On the event `|A_hat-A*|<=B`, clipping is 1-Lipschitz and the anchor correction `2B/R` covers the score shift.",
        "",
        "The graph-color LCB uses `chi=m+1`; the comparator deliberately assumes `chi=1`. The two one-sided events are budgeted by a union bound: graph-score failure at most `alpha_score`, plus anchor interval failure at most `alpha_anchor`. This theoretical statement is conditional on the fully specified generator, independence of anchor and score innovations, and the exact known-sigma normal model.",
        "",
        "## Figure and reading guide",
        "",
        "![Joint score and anchor calibration diagnostics](coverage_and_utility.png)",
        "",
        "The top-left heatmap shows the planned-run probability that an issued graph LCB exceeds the true mean (false certificate); its cells include Monte Carlo Wilson intervals. The top-right repeats the diagnostic with the intentionally wrong IID graph. The lower-left reports the illustrative positive-bound rate at threshold zero, which is power/utility rather than a physical threshold. The lower-right checks the anchor interval coverage against its 99% nominal line. Numeric LCB coverage conditional on issuing a bound is also tabulated, but conditioning on non-abstention is descriptive and is not the unconditional theorem.",
        "",
        "## Denominators and limitations",
        "",
        "`cell_summary.csv` and `denominator_summary.csv` separate planned/valid/failed/abstained MC replicates, paired block events, family-block scores, anchor estimates, and anchor observations. `replicate_summary.npz` stores each anchor error, failure count, interval-coverage indicator, bound/abstention result, and certificate event. An all-failed score replicate abstains and cannot issue a bound. Monte Carlo Wilson intervals quantify only simulation error.",
        "",
        "This simulation closes the software/method question of how a shared anchor interval can enter the bounded block-risk correction under a known independent normal anchor and known score graph. It does not estimate the project's `B`, `chi`, anchor distribution, or score; it cannot establish independence of actual E and D measurements or repair the failed reference/model-identification gate. The task stays `NOT_READY/UNCERTIFIED` with `physical_score=null`.",
        "",
        "## Rebuild",
        "",
        "Use a fresh output directory; existing artifacts are never overwritten:",
        "",
        "```text",
        "python -B -m tasks.pion_projection.bounded_block_risk_anchor_coverage_mc --output NEW_DIRECTORY",
        "```",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n")
    artifact_names = [name for name in summary["files"] if name != "artifact_hashes.json"]
    hashes = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
              for name in artifact_names}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.replicates), indent=2))


if __name__ == "__main__":
    main()
