"""Joint anchor-set and setwise Markov coverage screen (synthetic CPU only).

The independent anchor sample creates a random Clopper-Pearson target interval.
A separate, exactly stationary reversible Markov chain supplies planned-block
scores. The screen is deliberately not a physical calibration receipt.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta

from .bounded_block_risk_markov_coverage_screen import _wilson
from .bounded_block_risk_markov_varying import (
    time_varying_markov_radius_from_gap_bound,
)
from .bounded_block_risk_setwise_markov import _target_grid


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
DEFAULT_OUT = TASK_DIR / "bounded_block_risk_independent_anchor_coverage_20261128_v1"
TARGET_PROBABILITY = 0.4
ANCHOR_SAMPLE_SIZE = 256
GAMMA_GRID = (0.5, 0.75, 1.0)
PLANNED_BLOCKS = 16_384
REPLICATES_PER_GAMMA = 1_000
ALPHA_ANCHOR = 0.025
ALPHA_SCORE = 0.025
ALPHA_TOTAL = 0.05
PREDICTION_DOMAIN = (0.0, 1.0)
SCORE_SCALE_R = 1.0
MESH_STEP_R = 0.1
ANCHOR_SEED_BASE = 202611282300
SCORE_SEED_BASE = 202611287300


def _cp_interval(successes: int, trials: int, alpha: float) -> tuple[float, float]:
    """Two-sided exact Clopper-Pearson interval for a binomial probability."""
    if type(successes) is not int or type(trials) is not int:
        raise ValueError("successes and trials must be integers")
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("require trials >= 1 and 0 <= successes <= trials")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    lower = 0.0 if successes == 0 else float(
        beta.ppf(alpha / 2.0, successes, trials - successes + 1)
    )
    upper = 1.0 if successes == trials else float(
        beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes)
    )
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError("failed to construct a finite Clopper-Pearson interval")
    return lower, upper


def _draw_anchor_samples(*, seed: int, replicates: int,
                         trials: int = ANCHOR_SAMPLE_SIZE) -> np.ndarray:
    if type(replicates) is not int or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if type(trials) is not int or trials < 1:
        raise ValueError("trials must be a positive integer")
    rng = np.random.default_rng(seed)
    return rng.random((replicates, trials)) < TARGET_PROBABILITY


def _draw_reversible_chain(*, gamma_ps: float, blocks: int,
                           replicates: int, seed: int) -> np.ndarray:
    """Draw exact-stationary two-state trajectories with the requested gap.

    For positive second eigenvalue lambda=sqrt(1-gamma_ps), detailed balance
    holds at pi(1)=p and the normal-kernel pseudo-gap is 1-lambda**2.
    """
    if not math.isfinite(gamma_ps) or not 0.0 < gamma_ps <= 1.0:
        raise ValueError("gamma_ps must lie in (0, 1]")
    if type(blocks) is not int or blocks < 2:
        raise ValueError("blocks must be an integer >= 2")
    if type(replicates) is not int or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    eigenvalue = math.sqrt(1.0 - gamma_ps)
    p01 = (1.0 - eigenvalue) * TARGET_PROBABILITY
    p10 = (1.0 - eigenvalue) * (1.0 - TARGET_PROBABILITY)
    rng = np.random.default_rng(seed)
    states = np.empty((replicates, blocks), dtype=np.bool_)
    states[:, 0] = rng.random(replicates) < TARGET_PROBABILITY
    for index in range(1, blocks):
        uniforms = rng.random(replicates)
        previous = states[:, index - 1]
        states[:, index] = np.where(previous, uniforms >= p10, uniforms < p01)
    return states


def _expected_score(target: float, probability_state1: float) -> float:
    """Exact stationary expected normalized score at target ``target``."""
    # Scores are target - 1/4 in state 0 and 3/4 - target in state 1.
    return (1.0 - 2.0 * probability_state1) * target - 0.25 + probability_state1


def _exact_interval_minimum(lower: float, upper: float) -> float:
    return min(_expected_score(lower, TARGET_PROBABILITY),
               _expected_score(upper, TARGET_PROBABILITY))


def _uniform_lcb_from_state1_counts(
    state1_counts: np.ndarray,
    intervals: np.ndarray,
    *,
    gamma_ps: float,
    blocks: int,
    alpha_score: float = ALPHA_SCORE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized equivalent of the production helper for this two-state case."""
    if state1_counts.ndim != 1 or intervals.shape != (state1_counts.size, 2):
        raise ValueError("counts and target intervals have incompatible shapes")
    fractions = state1_counts / blocks
    lower_bounds = np.empty(state1_counts.size, dtype=float)
    true_minima = np.empty(state1_counts.size, dtype=float)
    point_counts = np.empty(state1_counts.size, dtype=np.int64)
    radius_cache: dict[int, float] = {}
    for index, (lower, upper) in enumerate(intervals):
        grid, mesh_penalty, _ = _target_grid(
            float(lower), float(upper), score_scale_R=SCORE_SCALE_R,
            mesh_step_R=MESH_STEP_R, max_grid_points=4096,
        )
        family_size = len(grid)
        point_counts[index] = family_size
        if family_size not in radius_cache:
            radius_cache[family_size] = time_varying_markov_radius_from_gap_bound(
                blocks=blocks,
                gamma_ps_lower=gamma_ps,
                family=family_size,
                alpha_stationary=alpha_score,
            )["radius"]
        # E_hat[s(m)] = m - 1/4 + q_hat*(1 - 2m).
        point_means = [
            grid_value - 0.25 + fractions[index] * (1.0 - 2.0 * grid_value)
            for grid_value in grid
        ]
        raw_lcb = min(point_means) - radius_cache[family_size] - mesh_penalty
        lower_bounds[index] = max(-1.0, math.nextafter(raw_lcb, -math.inf))
        true_minima[index] = _exact_interval_minimum(float(lower), float(upper))
    return lower_bounds, true_minima, point_counts


def _cell(gamma_ps: float, *, replicates: int = REPLICATES_PER_GAMMA,
          blocks: int = PLANNED_BLOCKS, anchor_seed: int | None = None,
          score_seed: int | None = None) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    gap_index = GAMMA_GRID.index(gamma_ps)
    anchor_seed = (ANCHOR_SEED_BASE + 10_000 * gap_index
                   if anchor_seed is None else anchor_seed)
    score_seed = (SCORE_SEED_BASE + 10_000 * gap_index
                  if score_seed is None else score_seed)
    anchors = _draw_anchor_samples(seed=anchor_seed, replicates=replicates)
    states = _draw_reversible_chain(
        gamma_ps=gamma_ps, blocks=blocks, replicates=replicates,
        seed=score_seed,
    )
    anchor_counts = anchors.sum(axis=1, dtype=np.int64)
    intervals = np.asarray([
        _cp_interval(int(count), ANCHOR_SAMPLE_SIZE, ALPHA_ANCHOR)
        for count in anchor_counts
    ], dtype=float)
    state1_counts = states.sum(axis=1, dtype=np.int64)
    uniform_lcbs, true_minima, grid_points = _uniform_lcb_from_state1_counts(
        state1_counts, intervals, gamma_ps=gamma_ps, blocks=blocks,
    )
    anchor_covered = (intervals[:, 0] <= TARGET_PROBABILITY) & (
        TARGET_PROBABILITY <= intervals[:, 1]
    )
    score_covered = uniform_lcbs <= true_minima
    joint_covered = anchor_covered & score_covered
    widths = intervals[:, 1] - intervals[:, 0]
    slacks = true_minima - uniform_lcbs

    rows = [
        {
            "gamma_ps": gamma_ps,
            "replicate": index,
            "anchor_seed": anchor_seed,
            "score_chain_seed": score_seed,
            "planned_anchor_observations": ANCHOR_SAMPLE_SIZE,
            "valid_anchor_observations": ANCHOR_SAMPLE_SIZE,
            "failed_anchor_observations": 0,
            "anchor_successes": int(anchor_counts[index]),
            "target_interval_lower": float(intervals[index, 0]),
            "target_interval_upper": float(intervals[index, 1]),
            "target_interval_width": float(widths[index]),
            "planned_score_blocks": blocks,
            "valid_score_blocks": blocks,
            "failed_score_blocks": 0,
            "state1_count": int(state1_counts[index]),
            "state1_fraction": float(state1_counts[index] / blocks),
            "target_grid_points": int(grid_points[index]),
            "uniform_lcb": float(uniform_lcbs[index]),
            "exact_interval_minimum_score": float(true_minima[index]),
            "lcb_slack": float(slacks[index]),
            "anchor_covered": bool(anchor_covered[index]),
            "score_uniformly_covered": bool(score_covered[index]),
            "jointly_covered": bool(joint_covered[index]),
            "positive_lcb": bool(uniform_lcbs[index] > 0.0),
        }
        for index in range(replicates)
    ]

    def stats(mask: np.ndarray) -> dict[str, Any]:
        covered = int(mask.sum())
        low, high = _wilson(covered, replicates)
        return {
            "successes": covered,
            "trials": replicates,
            "rate": covered / replicates,
            "wilson95_low": low,
            "wilson95_high": high,
        }

    summary = {
        "gamma_ps": gamma_ps,
        "second_eigenvalue": math.sqrt(1.0 - gamma_ps),
        "transition_p01": (1.0 - math.sqrt(1.0 - gamma_ps)) * TARGET_PROBABILITY,
        "transition_p10": (1.0 - math.sqrt(1.0 - gamma_ps)) * (1.0 - TARGET_PROBABILITY),
        "planned_replicates": replicates,
        "valid_replicates": replicates,
        "failed_replicates": 0,
        "abstained_replicates": 0,
        "planned_anchor_observations": replicates * ANCHOR_SAMPLE_SIZE,
        "valid_anchor_observations": replicates * ANCHOR_SAMPLE_SIZE,
        "failed_anchor_observations": 0,
        "planned_score_blocks": replicates * blocks,
        "valid_score_blocks": replicates * blocks,
        "failed_score_blocks": 0,
        "anchor_coverage": stats(anchor_covered),
        "score_uniform_coverage": stats(score_covered),
        "joint_coverage": stats(joint_covered),
        "nominal_anchor_coverage": 1.0 - ALPHA_ANCHOR,
        "nominal_score_coverage": 1.0 - ALPHA_SCORE,
        "union_bound_joint_coverage_lower": 1.0 - ALPHA_TOTAL,
        "positive_lcb_replicates": int(np.count_nonzero(uniform_lcbs > 0.0)),
        "positive_lcb_rate": float(np.mean(uniform_lcbs > 0.0)),
        "median_target_interval_width": float(np.median(widths)),
        "median_uniform_lcb": float(np.median(uniform_lcbs)),
        "median_exact_interval_minimum_score": float(np.median(true_minima)),
        "median_lcb_slack": float(np.median(slacks)),
        "median_grid_points": float(np.median(grid_points)),
        "anchor_seed": anchor_seed,
        "score_chain_seed": score_seed,
    }
    raw = {
        "anchor_samples": anchors,
        "score_chain_states": states,
        "anchor_counts": anchor_counts,
        "intervals": intervals,
        "state1_counts": state1_counts,
    }
    return raw, rows, summary


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plot(summary: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gaps = [item["gamma_ps"] for item in summary]
    metrics = [
        ("anchor_coverage", "random anchor interval"),
        ("score_uniform_coverage", "uniform score lower bound"),
        ("joint_coverage", "joint event"),
    ]
    colors = ["#3c7d69", "#4776a8", "#c47937"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), constrained_layout=True)
    ax = axes[0]
    offsets = (-0.035, 0.0, 0.035)
    for (key, label), color, offset in zip(metrics, colors, offsets):
        rates = [item[key]["rate"] for item in summary]
        lows = [item[key]["wilson95_low"] for item in summary]
        highs = [item[key]["wilson95_high"] for item in summary]
        errors = [[rate - low for rate, low in zip(rates, lows)],
                  [high - rate for rate, high in zip(rates, highs)]]
        ax.errorbar(np.asarray(gaps) + offset, rates, yerr=errors,
                    marker="o", capsize=3, linewidth=1.4, color=color,
                    label=label)
    ax.axhline(1.0 - ALPHA_TOTAL, color="#7a3b3b", linestyle="--",
               linewidth=1.2, label="joint union-bound floor = 95%")
    ax.set(xticks=gaps, ylim=(0.90, 1.005), xlabel=r"known pseudo-gap $\gamma_{ps}$",
           ylabel="empirical coverage across 1,000 replicates",
           title="Independent anchor + Markov score chain")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.7, loc="lower right")

    ax = axes[1]
    med_lcb = [item["median_uniform_lcb"] for item in summary]
    med_truth = [item["median_exact_interval_minimum_score"] for item in summary]
    ax.plot(gaps, med_truth, marker="s", color="#333333",
            label="median exact minimum over random interval")
    ax.plot(gaps, med_lcb, marker="o", color="#6a4c93",
            label="median uniform LCB")
    ax.fill_between(gaps, med_lcb, med_truth, color="#9b8ab8", alpha=0.18,
                    label="median vertical gap (guide)")
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1)
    ax.set(xticks=gaps, xlabel=r"known pseudo-gap $\gamma_{ps}$",
           ylabel="normalized bounded-risk score",
           title="Lower-bound sharpness on random target sets")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.4, loc="best")
    fig.suptitle("Synthetic joint-coverage screen; not physical calibration")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_report(output: str | Path = DEFAULT_OUT) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    raw_by_gap: dict[str, dict[str, np.ndarray]] = {}
    all_rows: list[dict] = []
    summaries: list[dict] = []
    for gamma in GAMMA_GRID:
        raw, rows, summary = _cell(gamma)
        raw_by_gap[f"gamma_{gamma:g}"] = raw
        all_rows.extend(rows)
        summaries.append(summary)

    with (out / "replicate_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    summary_rows = []
    for row in summaries:
        summary_rows.append({
            "gamma_ps": row["gamma_ps"],
            "planned_replicates": row["planned_replicates"],
            "anchor_coverage": row["anchor_coverage"]["rate"],
            "anchor_wilson95_low": row["anchor_coverage"]["wilson95_low"],
            "anchor_wilson95_high": row["anchor_coverage"]["wilson95_high"],
            "score_uniform_coverage": row["score_uniform_coverage"]["rate"],
            "score_wilson95_low": row["score_uniform_coverage"]["wilson95_low"],
            "score_wilson95_high": row["score_uniform_coverage"]["wilson95_high"],
            "joint_coverage": row["joint_coverage"]["rate"],
            "joint_wilson95_low": row["joint_coverage"]["wilson95_low"],
            "joint_wilson95_high": row["joint_coverage"]["wilson95_high"],
            "median_target_interval_width": row["median_target_interval_width"],
            "median_uniform_lcb": row["median_uniform_lcb"],
            "median_exact_interval_minimum_score": row["median_exact_interval_minimum_score"],
            "median_lcb_slack": row["median_lcb_slack"],
            "positive_lcb_rate": row["positive_lcb_rate"],
        })
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    input_arrays: dict[str, np.ndarray] = {}
    raw_shapes: dict[str, dict[str, list[int]]] = {}
    for gap_name, raw in raw_by_gap.items():
        anchor = raw["anchor_samples"]
        chain = raw["score_chain_states"]
        input_arrays[f"{gap_name}_anchor_samples_packed"] = np.packbits(
            anchor, axis=1, bitorder="little",
        )
        input_arrays[f"{gap_name}_score_chain_states_packed"] = np.packbits(
            chain, axis=1, bitorder="little",
        )
        input_arrays[f"{gap_name}_anchor_counts"] = raw["anchor_counts"]
        input_arrays[f"{gap_name}_target_intervals"] = raw["intervals"]
        input_arrays[f"{gap_name}_state1_counts"] = raw["state1_counts"]
        raw_shapes[gap_name] = {
            "anchor_samples": list(anchor.shape),
            "score_chain_states": list(chain.shape),
            "packing": "numpy.packbits(axis=1, bitorder='little')",
        }
    np.savez_compressed(out / "synthetic_inputs.npz", **input_arrays)

    source_files = {
        "coverage_screen": Path(__file__).resolve(),
        "fixed_markov_screen": TASK_DIR / "bounded_block_risk_markov_coverage_screen.py",
        "setwise_markov": TASK_DIR / "bounded_block_risk_setwise_markov.py",
        "setwise_scores": TASK_DIR / "bounded_block_risk_setwise.py",
        "markov_varying": TASK_DIR / "bounded_block_risk_markov_varying.py",
        "markov_gap": TASK_DIR / "bounded_block_risk_markov_gap.py",
        "core_numerics": REPO_ROOT / "core" / "numerics.py",
    }
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for name, source in source_files.items():
        if name == "core_numerics":
            destination = snapshot / "core" / "numerics.py"
            destination.parent.mkdir(parents=True, exist_ok=True)
        else:
            destination = snapshot / "tasks" / "pion_projection" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    source_hashes = {
        f"{destination.relative_to(out).as_posix()}": _sha(destination)
        for destination in snapshot.rglob("*.py")
    }
    protocol = {
        "schema": "pion-independent-anchor-setwise-markov-coverage-v1",
        "classification": "SYNTHETIC_JOINT_COVERAGE_SCREEN_NOT_PHYSICAL_CALIBRATION",
        "generated_on": "2026-11-28",
        "anchor": {
            "sampling": "independent iid Bernoulli draws from exact target mass p",
            "target_probability": TARGET_PROBABILITY,
            "sample_size": ANCHOR_SAMPLE_SIZE,
            "interval": "two-sided exact Clopper-Pearson",
            "alpha_anchor": ALPHA_ANCHOR,
        },
        "score_chain": {
            "family": "two-state reversible Markov chains with stationary mass P(state=1)=p",
            "pseudo_spectral_gap_grid": list(GAMMA_GRID),
            "second_eigenvalue_rule": "lambda=sqrt(1-gamma_ps); P01=(1-lambda)*p; P10=(1-lambda)*(1-p)",
            "initial_law": "exact stationary draw; initial TV=0",
            "planned_blocks_per_chain": PLANNED_BLOCKS,
            "reference_prediction": "score-chain state 0 or 1",
            "candidate_prediction": 0.5,
        },
        "estimand": {
            "score": "clip(((m-reference)^2-(m-candidate)^2)/R^2,-1,1)",
            "target_set": "random anchor interval, common across planned score-chain blocks",
            "prediction_domain": list(PREDICTION_DOMAIN),
            "score_scale_R": SCORE_SCALE_R,
            "grid_mesh_step_R": MESH_STEP_R,
            "exact_stationary_mean": "(1-2p)*m - 1/4 + p",
            "joint_success": "anchor interval contains p AND uniform setwise Markov LCB <= exact minimum expected score over that interval",
        },
        "error_allocation": {
            "alpha_anchor": ALPHA_ANCHOR,
            "alpha_score": ALPHA_SCORE,
            "alpha_total": ALPHA_TOTAL,
            "union_bound_joint_coverage_lower": 1.0 - ALPHA_TOTAL,
            "reason": "independent anchor set permits the score-chain simultaneous bound conditional on every realized set; union bound combines anchor and score failures",
        },
        "replicates_per_gap": REPLICATES_PER_GAMMA,
        "seed_rules": {
            "anchor": f"{ANCHOR_SEED_BASE} + 10000*gap_index",
            "score_chain": f"{SCORE_SEED_BASE} + 10000*gap_index",
            "indices": "gap_index is zero-based; separate NumPy Generator instances use disjoint seed ranges",
        },
        "raw_input_archive": {
            "file": "synthetic_inputs.npz",
            "arrays": raw_shapes,
            "bit_order": "little",
        },
        "source_snapshot_sha256": source_hashes,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "matplotlib": __import__("matplotlib").__version__,
        },
        "no_tuning": True,
        "sealed_data_access": False,
        "gpu_quda": False,
        "physical_score": None,
        "certified": False,
    }
    (out / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    _plot(summaries, out / "joint_coverage_and_sharpness.png")
    report = {
        "schema": protocol["schema"],
        "classification": protocol["classification"],
        "physical_score": None,
        "certified": False,
        "denominators": {
            "planned_replicates": len(GAMMA_GRID) * REPLICATES_PER_GAMMA,
            "valid_replicates": len(GAMMA_GRID) * REPLICATES_PER_GAMMA,
            "failed_replicates": 0,
            "abstained_replicates": 0,
            "planned_anchor_observations": len(GAMMA_GRID) * REPLICATES_PER_GAMMA * ANCHOR_SAMPLE_SIZE,
            "valid_anchor_observations": len(GAMMA_GRID) * REPLICATES_PER_GAMMA * ANCHOR_SAMPLE_SIZE,
            "failed_anchor_observations": 0,
            "planned_score_blocks": len(GAMMA_GRID) * REPLICATES_PER_GAMMA * PLANNED_BLOCKS,
            "valid_score_blocks": len(GAMMA_GRID) * REPLICATES_PER_GAMMA * PLANNED_BLOCKS,
            "failed_score_blocks": 0,
        },
        "summary": summaries,
    }
    (out / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    readme = [
        "# Pion independent-anchor joint coverage screen",
        "",
        "**Status:** `SYNTHETIC_JOINT_COVERAGE_SCREEN_NOT_PHYSICAL_CALIBRATION`; `physical_score=null`; no lattice, E-reference, gauge, GPU/QUDA, or sealed input was read.",
        "",
        "## Frozen protocol",
        "",
        "Each replicate has two separate random streams. The anchor stream draws 256 IID Bernoulli observations with exact target mass `p=0.4`, then forms a two-sided exact Clopper-Pearson interval at `alpha_anchor=0.025`. The score stream draws 16,384 states from an independent, exactly stationary two-state reversible chain. The chain uses `lambda=sqrt(1-gamma_ps)`, `P01=(1-lambda)*0.4`, and `P10=(1-lambda)*0.6`; its normal pseudo-gap is the selected `gamma_ps` in `{0.5, 0.75, 1}`.",
        "",
        "The reference prediction is the chain state and the candidate is `0.5`, with prediction domain `[0,1]`, `R=1`, and target mesh step `0.1`. The exact stationary risk curve is `E[s(m)]=(1-2p)m-1/4+p = 0.2m+0.15`, so its minimum on each random interval is computed exactly from the interval endpoints. The production exact-grid builder and support-gap Eq. (3.26) radius are used for the uniform lower bound. The joint event requires both that the random anchor interval contains `p` and that the score LCB is no greater than the exact minimum over that interval.",
        "",
        "The preallocated failure probabilities are `alpha_anchor=0.025` and `alpha_score=0.025`. Conditional on any realized anchor interval, the independent score-chain guarantee applies simultaneously over its grid; a union bound therefore gives a planned joint lower coverage of 0.95 under the protocol assumptions.",
        "",
        "## Results",
        "",
        "| gamma_ps | anchor coverage | score-uniform coverage | joint coverage | joint Wilson 95% interval | median interval width | median LCB | median exact minimum | median slack | positive LCB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        readme.append(
            f"| {row['gamma_ps']:.2f} | {row['anchor_coverage']['rate']:.3f} | "
            f"{row['score_uniform_coverage']['rate']:.3f} | {row['joint_coverage']['rate']:.3f} | "
            f"[{row['joint_coverage']['wilson95_low']:.3f}, {row['joint_coverage']['wilson95_high']:.3f}] | "
            f"{row['median_target_interval_width']:.3f} | {row['median_uniform_lcb']:.3f} | "
            f"{row['median_exact_interval_minimum_score']:.3f} | {row['median_lcb_slack']:.3f} | "
            f"{row['positive_lcb_rate']:.3f} |"
        )
    readme += [
        "",
        "![Joint coverage and lower-bound sharpness](joint_coverage_and_sharpness.png)",
        "",
        "## Text diagnostic and interpretation",
        "",
        "```text",
        "independent anchor Bernoulli sample (256) -> exact random target interval (alpha_anchor=.025)",
        "                                                                  |",
        "separate exact-stationary Markov chain (N=16,384) -> production exact grid + support-gap LCB",
        "                                                                  |",
        "                      anchor contains p AND uniform LCB <= exact interval minimum",
        "                                                                  |",
        "                   count anchor / score / joint events; Wilson MC intervals + sharpness",
        "```",
        "",
        "The figure's left panel shows the three event rates against the planned 95% joint lower bound. The right panel compares the median uniform LCB with the median exact interval minimum and shows the median slack. The screen uses 1,000 independent PRNG replicates per gap, so its Wilson intervals describe Monte Carlo uncertainty; they do not replace the analytic 95% union-bound guarantee. If a small MC interval endpoint falls below 0.95, that is a finite simulation fluctuation, not by itself a contradiction of the theorem.",
        "",
        "## Denominators and reproducibility",
        "",
        f"Planned and valid denominators: {report['denominators']['planned_replicates']} anchor/score replicate pairs; {report['denominators']['planned_anchor_observations']:,} anchor observations; and {report['denominators']['planned_score_blocks']:,} score-chain states. Failed or abstained units: 0. Raw synthetic Bernoulli arrays and full chain trajectories are preserved bit-packed in `synthetic_inputs.npz`; `replicate_results.csv` has one row per planned replicate and records both seed streams, interval endpoints, state counts, coverage events, and LCB slack. `protocol.json` records source snapshot hashes and runtime versions.",
        "",
        "These results test only a known two-state reversible model with exact `p`, exact pseudo-gap, exact stationary starts, and an anchor sample that is IID by construction. They do not validate HMC independence, production `gamma_ps`/TV receipts, target mass/source correspondence, score-anchor common bias, anchor-to-E bridge, prediction support, physical units, or an E-reference gate. The anchor interval is not an estimate of any physical mass. Pion remains `NOT_READY/UNCERTIFIED`, `physical_score=null`, and sealed inputs closed.",
        "",
        "To reproduce from the repository root: `python -m tasks.pion_projection.bounded_block_risk_independent_anchor_coverage_screen`. The report builder fails if the output folder already exists, preserving prior evidence.",
        "",
    ]
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    artifact_names = [
        "protocol.json", "summary.json", "summary.csv", "replicate_results.csv",
        "synthetic_inputs.npz", "joint_coverage_and_sharpness.png", "README.md",
    ]
    (out / "artifact_hashes.json").write_text(
        json.dumps({name: _sha(out / name) for name in artifact_names},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
