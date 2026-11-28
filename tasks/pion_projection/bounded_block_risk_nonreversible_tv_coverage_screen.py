"""Nonreversible, post-burn-in coverage screen for Pion setwise Markov LCB.

This finite-state screen has an exact pseudo-gap and exact initial-law TV
distance.  It is a conditional synthetic diagnostic, not a production receipt.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from .bounded_block_risk_markov_coverage_screen import _wilson
from .bounded_block_risk_setwise_markov import (
    _target_grid,
    setwise_markov_score_lcb_from_gap_bound,
)
from .bounded_block_risk_markov_varying import time_varying_markov_radius_from_gap_bound


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = TASK_DIR / "bounded_block_risk_nonreversible_tv_coverage_20261128_v1"
MOVE_PROBABILITY = 0.5
PSEUDO_GAP = 0.75
BURN_IN = 8
PLANNED_BLOCKS = 16_384
REPLICATES = 1_000
SEED = 202611281709
TARGET_INTERVAL = (0.45, 0.55)
PREDICTION_DOMAIN = (0.0, 1.0)
SCORE_SCALE_R = 1.0
MESH_STEP_R = 0.10
ALPHA_SCORE = 0.04
ALPHA_ANCHOR = 0.01
ALPHA_TOTAL = 0.05
TRUE_UNIFORM_MEAN = 1.0 / 6.0


def _transition() -> np.ndarray:
    """P[i,j] = 1/2 stay at i or move one step forward modulo three."""
    return np.asarray([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
    ], dtype=np.float64)


def _tv_after_burn_in(burn_in: int = BURN_IN) -> Fraction:
    if type(burn_in) is not int or burn_in < 0:
        raise ValueError("burn_in must be a nonnegative integer")
    transition = ((Fraction(1, 2), Fraction(1, 2), Fraction(0)),
                  (Fraction(0), Fraction(1, 2), Fraction(1, 2)),
                  (Fraction(1, 2), Fraction(0), Fraction(1, 2)))
    distribution = [Fraction(1), Fraction(0), Fraction(0)]
    stationary = Fraction(1, 3)
    for _ in range(burn_in):
        distribution = [
            sum(distribution[i] * transition[i][j] for i in range(3))
            for j in range(3)
        ]
    return sum(abs(value - stationary) for value in distribution) / 2


def _burned_cycle_counts(
    *, blocks: int = PLANNED_BLOCKS,
    replicates: int = REPLICATES,
    seed: int = SEED,
    burn_in: int = BURN_IN,
) -> np.ndarray:
    """Draw scored-state counts after starting at 0 and discarding burn-in."""
    if type(blocks) is not int or blocks < 2:
        raise ValueError("blocks must be an integer >= 2")
    if type(replicates) is not int or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if type(burn_in) is not int or burn_in < 0:
        raise ValueError("burn_in must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    transitions = rng.random((replicates, burn_in + blocks - 1)) < MOVE_PROBABILITY
    states = np.empty((replicates, burn_in + blocks), dtype=np.int8)
    states[:, 0] = 0
    states[:, 1:] = (np.cumsum(transitions, axis=1) % 3).astype(np.int8)
    scored = states[:, burn_in:burn_in + blocks]
    return np.stack([np.count_nonzero(scored == state, axis=1)
                     for state in range(3)], axis=1)


def _state_score(target: float, state: int) -> float:
    reference = (0.0, 0.5, 1.0)[state]
    candidate = 0.5
    return float(np.clip(
        ((target - reference) ** 2 - (target - candidate) ** 2)
        / SCORE_SCALE_R**2,
        -1.0,
        1.0,
    ))


def _cell(*, blocks: int = PLANNED_BLOCKS,
          replicates: int = REPLICATES,
          seed: int = SEED,
          burn_in: int = BURN_IN) -> tuple[list[dict], dict]:
    counts = _burned_cycle_counts(
        blocks=blocks, replicates=replicates, seed=seed, burn_in=burn_in,
    )
    tv_exact = _tv_after_burn_in(burn_in)
    tv_upper = math.nextafter(float(tv_exact), math.inf) if tv_exact else 0.0
    grid, mesh_penalty, _ = _target_grid(
        *TARGET_INTERVAL, score_scale_R=SCORE_SCALE_R,
        mesh_step_R=MESH_STEP_R, max_grid_points=4096,
    )
    radius_info = time_varying_markov_radius_from_gap_bound(
        blocks=blocks,
        gamma_ps_lower=PSEUDO_GAP,
        family=len(grid),
        alpha_stationary=ALPHA_SCORE - len(grid) * tv_upper,
    )
    radius = radius_info["radius"]
    means = np.zeros((len(grid), replicates), dtype=np.float64)
    for target_index, target in enumerate(grid):
        scores = np.asarray([_state_score(target, state) for state in range(3)])
        means[target_index] = counts @ scores / blocks
    pointwise_lcb = means - radius - 2.0 * tv_upper
    uniform_lcb = np.maximum(
        -1.0,
        np.nextafter(np.min(pointwise_lcb, axis=0) - mesh_penalty, -np.inf),
    )
    covered = uniform_lcb <= TRUE_UNIFORM_MEAN
    rows = []
    for i in range(replicates):
        rows.append({
            "replicate": i,
            "seed": seed,
            "burn_in_transitions": burn_in,
            "planned_scored_states": blocks,
            "valid_scored_states": blocks,
            "failed_scored_states": 0,
            "state0_count": int(counts[i, 0]),
            "state1_count": int(counts[i, 1]),
            "state2_count": int(counts[i, 2]),
            "initial_law_tv_exact": str(tv_exact),
            "initial_law_tv_upper": tv_upper,
            "pseudo_gap": PSEUDO_GAP,
            "markov_radius": radius,
            "target_grid_points": len(grid),
            "mesh_penalty": mesh_penalty,
            "uniform_lcb": float(uniform_lcb[i]),
            "true_uniform_mean": TRUE_UNIFORM_MEAN,
            "covered": bool(covered[i]),
            "positive_lcb": bool(uniform_lcb[i] > 0.0),
        })
    successes = int(np.count_nonzero(covered))
    cover_ci = _wilson(successes, replicates)
    summary = {
        "chain": "three-state directed cycle with 1/2 self-loop and 1/2 forward move",
        "pseudo_gap": PSEUDO_GAP,
        "burn_in_transitions": burn_in,
        "initial_law_tv_exact": str(tv_exact),
        "initial_law_tv_upper": tv_upper,
        "planned_scored_states_per_chain": blocks,
        "planned_replicates": replicates,
        "valid_replicates": replicates,
        "failed_replicates": 0,
        "abstained_replicates": 0,
        "planned_scored_states": blocks * replicates,
        "true_uniform_mean": TRUE_UNIFORM_MEAN,
        "empirical_uniform_lcb_coverage": successes / replicates,
        "coverage_wilson95_low": cover_ci[0],
        "coverage_wilson95_high": cover_ci[1],
        "nominal_conditional_coverage": 1.0 - ALPHA_SCORE,
        "positive_lcb_replicates": int(np.count_nonzero(uniform_lcb > 0.0)),
        "positive_lcb_rate": float(np.mean(uniform_lcb > 0.0)),
        "median_uniform_lcb": float(np.median(uniform_lcb)),
        "minimum_uniform_lcb": float(np.min(uniform_lcb)),
        "maximum_uniform_lcb": float(np.max(uniform_lcb)),
        "markov_radius": radius,
        "target_grid_points": len(grid),
        "mesh_penalty": mesh_penalty,
        "seed": seed,
    }
    return rows, summary


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plot(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rate = summary["empirical_uniform_lcb_coverage"]
    low, high = summary["coverage_wilson95_low"], summary["coverage_wilson95_high"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    axes[0].errorbar([0], [rate], yerr=[[rate - low], [high - rate]],
                     marker="o", capsize=5, color="#276c91",
                     label="empirical coverage with Wilson 95% MC interval")
    axes[0].axhline(1.0 - ALPHA_SCORE, color="#a13b36", linestyle="--",
                    label="nominal conditional level 96%")
    axes[0].set_xlim(-0.55, 0.55)
    axes[0].set_xticks([0], ["nonreversible\npost-burn-in"])
    axes[0].set_ylim(0.90, 1.005)
    axes[0].set_ylabel("coverage across independent chains")
    axes[0].set_title("Uniform lower-bound coverage")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower right")

    axes[1].bar([0], [summary["median_uniform_lcb"]], color="#6a4c93", width=0.5)
    axes[1].axhline(TRUE_UNIFORM_MEAN, color="black", linestyle=":",
                    label="exact stationary risk difference = 1/6")
    axes[1].axhline(0.0, color="#777777", linestyle="--", linewidth=1)
    axes[1].set_xlim(-0.6, 0.6)
    axes[1].set_xticks([0], ["TV-adjusted\nsetwise LCB"])
    axes[1].set_ylabel("normalized score units")
    axes[1].set_title("Lower-bound sharpness")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    axes[1].text(0, summary["median_uniform_lcb"],
                 f"  positive in {summary['positive_lcb_rate']:.1%} of chains",
                 va="center", fontsize=8)
    fig.suptitle("Conditional synthetic screen; not physical calibration")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def build_report(output: str | Path = DEFAULT_OUT) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows, summary = _cell()
    with (out / "replicate_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary), lineterminator="\n")
        writer.writeheader()
        writer.writerow(summary)
    grid, mesh_penalty, _ = _target_grid(
        *TARGET_INTERVAL, score_scale_R=SCORE_SCALE_R,
        mesh_step_R=MESH_STEP_R, max_grid_points=4096,
    )
    sources = {
        "coverage_screen": Path(__file__).resolve(),
        "coverage_statistics": TASK_DIR / "bounded_block_risk_markov_coverage_screen.py",
        "setwise_markov": TASK_DIR / "bounded_block_risk_setwise_markov.py",
        "markov_varying": TASK_DIR / "bounded_block_risk_markov_varying.py",
        "markov_gap": TASK_DIR / "bounded_block_risk_markov_gap.py",
        "core_numerics": TASK_DIR.parents[1] / "core" / "numerics.py",
    }
    protocol = {
        "schema": "pion-nonreversible-tv-setwise-coverage-screen-v1",
        "classification": "CONDITIONAL_SYNTHETIC_MARKOV_COVERAGE_SCREEN_NOT_CALIBRATION",
        "chain": {
            "transition_matrix": _transition().tolist(),
            "stationary_distribution": [1 / 3, 1 / 3, 1 / 3],
            "reversibility": False,
            "pseudo_gap_derivation": (
                "eigenvalue magnitudes are 1 and 1/2; for this normal circulant P, "
                "max_k (1-(1/2)^(2k))/k = 1-(1/2)^2 = 3/4"
            ),
            "initial_state": 0,
            "burn_in_transitions": BURN_IN,
            "exact_tv_after_burn_in": str(_tv_after_burn_in()),
            "score_predictions_by_state": [0.0, 0.5, 1.0],
            "candidate_prediction": 0.5,
        },
        "target_interval": list(TARGET_INTERVAL),
        "prediction_domain": list(PREDICTION_DOMAIN),
        "score_scale_R": SCORE_SCALE_R,
        "score": "clip(((m-reference)^2-(m-candidate)^2)/R^2,-1,1)",
        "true_uniform_mean": TRUE_UNIFORM_MEAN,
        "planned_blocks_per_chain": PLANNED_BLOCKS,
        "replicates": REPLICATES,
        "seed": SEED,
        "pseudo_gap_lower_bound": PSEUDO_GAP,
        "alpha_score": ALPHA_SCORE,
        "alpha_anchor_reserved_but_not_used_for_fixed_target_set": ALPHA_ANCHOR,
        "alpha_total": ALPHA_TOTAL,
        "family_size_outside_grid": 1,
        "simultaneous_family_size": len(grid),
        "target_grid": grid,
        "maximum_grid_gap_over_R_upper_bound": mesh_penalty,
        "mesh_step_R": MESH_STEP_R,
        "failed_score_rule": "none in this exact finite-state simulation",
        "data_access": "synthetic finite-state chains only; no lattice, gauge, anchor, E-reference, or sealed bytes",
        "physical_score": None,
        "certified": False,
        "source_sha256": {
            str(path.relative_to(TASK_DIR.parents[1])): _sha(path)
            for path in sources.values()
        },
        "numpy_version": np.__version__,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    _plot(summary, out / "coverage_and_sharpness.png")
    readme = [
        "# Nonreversible post-burn-in Pion setwise Markov coverage screen",
        "",
        "**Status:** `CONDITIONAL_SYNTHETIC_MARKOV_COVERAGE_SCREEN_NOT_CALIBRATION`; `physical_score=null`. No lattice, anchor, E-reference, gauge, or sealed input was read.",
        "",
        "## Frozen construction",
        "",
        "The three-state directed cycle has transition matrix `[[1/2,1/2,0],[0,1/2,1/2],[1/2,0,1/2]]`. It is doubly stochastic but nonreversible. Its nontrivial eigenvalues have magnitude `1/2`; normality gives `gamma_ps=max_k(1-(1/2)^(2k))/k=3/4`. Chains start at state 0, discard eight transitions, then score 16,384 planned states. The exact post-burn-in total-variation distance from uniform is `1/384`; the helper receives the next representable float above it.",
        "",
        "The reference prediction by state is `(0, 0.5, 1)` and the candidate is `0.5`. On every target `m` in `[0.45,0.55]`, stationary risk difference is exactly `1/6`: the state-0 and state-2 scores sum to `1/2`, and state 1 contributes zero. The report uses the production exact grid, simultaneous support-gap radius, initial-TV alpha cost, `2*TV` mean shift, and `h/R` Lipschitz cover penalty.",
        "",
        "## Results",
        "",
        "| pseudo-gap | burn-in | exact initial TV | chains | empirical coverage | Wilson 95% MC interval | median LCB | positive-LCB rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {PSEUDO_GAP:.2f} | {BURN_IN} | 1/384 | {REPLICATES} | {summary['empirical_uniform_lcb_coverage']:.3f} | [{summary['coverage_wilson95_low']:.3f}, {summary['coverage_wilson95_high']:.3f}] | {summary['median_uniform_lcb']:.3f} | {summary['positive_lcb_rate']:.3f} |",
        "",
        "![Nonreversible TV-adjusted coverage and sharpness](coverage_and_sharpness.png)",
        "",
        "## Interpretation and limits",
        "",
        "This adds a nonreversible transition operator and a known nonstationary starting law to the earlier reversible stationary screen. Coverage counts the event that the helper's interval-uniform LCB is no greater than the exact stationary mean. The Wilson interval measures Monte Carlo uncertainty across the 1,000 independently seeded chains; it is not the helper's conditional confidence interval.",
        "",
        "The screen does not validate a production trajectory's pseudo-gap, its TV receipt, target-set coverage, anchor-to-E bridge, score support, block schedule, or unit. The anchor alpha is reserved but no random anchor event is present. No production conclusions follow; retain `NOT_READY/UNCERTIFIED`, `physical_score=null`, and sealed inputs closed.",
        "",
        "Data flow: `fixed state 0 -> 8 directed-cycle transitions -> exact TV=1/384 -> 16,384 scored states -> exact setwise grid -> simultaneous gap bound with TV-adjusted alpha and mean shift -> interval cover penalty -> coverage/LCB figure`. All 1,000 chains and all planned score states remain in the denominator.",
        "",
        "Artifact digests are recorded in `artifact_hashes.json`; `protocol.json` freezes the generator, exact gap/TV derivation, and seed.",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    artifact_names = ["protocol.json", "summary.json", "summary.csv",
                      "replicate_results.csv", "coverage_and_sharpness.png", "README.md"]
    (out / "artifact_hashes.json").write_text(
        json.dumps({name: _sha(out / name) for name in artifact_names},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return {"protocol": protocol, "summary": summary}


def main() -> None:
    result = build_report()
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
