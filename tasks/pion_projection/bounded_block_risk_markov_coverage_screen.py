"""Finite-state coverage screen for the conditional setwise Markov LCB.

This is a controlled synthetic check, not a calibration receipt.  It uses a
known reversible two-state chain and a deterministic target interval so the
coverage event has an exact reference value.  No lattice or sealed input is
read and the generated bound never becomes a physical score.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .bounded_block_risk_markov_varying import (
    time_varying_markov_radius_from_gap_bound,
)
from .bounded_block_risk_setwise_markov import _target_grid


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = TASK_DIR / "bounded_block_risk_markov_coverage_20261128_v1"
GAMMA_GRID = (0.25, 0.5, 1.0)
PLANNED_BLOCKS = 4096
REPLICATES_PER_GAMMA = 1000
SEED_BASE = 202611281103
TARGET_INTERVAL = (0.45, 0.55)
PREDICTION_DOMAIN = (0.0, 1.0)
SCORE_SCALE_R = 1.0
MESH_STEP_R = 0.10
ALPHA_SCORE = 0.04
ALPHA_ANCHOR_RESERVED = 0.01
ALPHA_TOTAL = 0.05


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("Wilson interval requires 0 <= successes <= trials and trials > 0")
    rate = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (rate + z2 / (2.0 * trials)) / denominator
    half = z * math.sqrt(rate * (1.0 - rate) / trials + z2 / (4.0 * trials * trials)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _chain_zero_counts(*, gamma: float, blocks: int, replicates: int, seed: int) -> np.ndarray:
    """Draw counts of state 0 from the stationary reversible two-state chain.

    The transition matrix has off-diagonal probability gamma/2, eigenvalues
    1 and 1-gamma, and pseudo-spectral gap gamma for gamma in (0, 1].
    """
    if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")
    if type(blocks) is not int or blocks < 2:
        raise ValueError("blocks must be an integer >= 2")
    if type(replicates) is not int or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    rng = np.random.default_rng(seed)
    flips = rng.random((replicates, blocks - 1)) < gamma / 2.0
    states = np.empty((replicates, blocks), dtype=np.int8)
    states[:, 0] = rng.integers(0, 2, size=replicates, dtype=np.int8)
    states[:, 1:] = (states[:, :1] + np.cumsum(flips, axis=1)) % 2
    return np.count_nonzero(states == 0, axis=1)


def _score_at_target(target: float, state: int) -> float:
    reference = float(state)
    candidate = 0.5
    return float(np.clip(
        ((target - reference) ** 2 - (target - candidate) ** 2)
        / SCORE_SCALE_R**2,
        -1.0,
        1.0,
    ))


def _cell(gamma: float, *, blocks: int = PLANNED_BLOCKS,
          replicates: int = REPLICATES_PER_GAMMA,
          seed: int | None = None) -> tuple[list[dict], dict]:
    seed = SEED_BASE + round(gamma * 1000) if seed is None else seed
    zero_counts = _chain_zero_counts(
        gamma=gamma, blocks=blocks, replicates=replicates, seed=seed,
    )
    grid, mesh_penalty, _ = _target_grid(
        *TARGET_INTERVAL,
        score_scale_R=SCORE_SCALE_R,
        mesh_step_R=MESH_STEP_R,
        max_grid_points=4096,
    )
    radius = time_varying_markov_radius_from_gap_bound(
        blocks=blocks,
        gamma_ps_lower=gamma,
        family=len(grid),  # one comparison times every exact target-grid point
        alpha_stationary=ALPHA_SCORE,
    )["radius"]
    p_zero = zero_counts / blocks
    target_lcbs = []
    for target in grid:
        mean = (
            p_zero * _score_at_target(target, 0)
            + (1.0 - p_zero) * _score_at_target(target, 1)
        )
        target_lcbs.append(mean - radius)
    # Use the same exact-rational grid gap and upward cover cost as the helper.
    uniform_lcb = np.maximum(
        -1.0,
        np.nextafter(np.min(np.stack(target_lcbs), axis=0) - mesh_penalty,
                     -np.inf),
    )
    true_uniform_mean = 0.25
    covered = uniform_lcb <= true_uniform_mean
    rows = [
        {
            "gamma_ps": gamma,
            "replicate": index,
            "seed": seed,
            "planned_blocks": blocks,
            "valid_blocks": blocks,
            "failed_blocks": 0,
            "state0_count": int(count),
            "state0_fraction": float(p_zero[index]),
            "markov_radius": radius,
            "target_grid_points": len(grid),
            "mesh_penalty": mesh_penalty,
            "uniform_lcb": float(uniform_lcb[index]),
            "true_uniform_mean": true_uniform_mean,
            "covered": bool(covered[index]),
            "positive_lcb": bool(uniform_lcb[index] > 0.0),
        }
        for index, count in enumerate(zero_counts)
    ]
    successes = int(covered.sum())
    cover_ci = _wilson(successes, replicates)
    positive = int(np.count_nonzero(uniform_lcb > 0.0))
    summary = {
        "gamma_ps": gamma,
        "transition_off_diagonal": gamma / 2.0,
        "planned_blocks_per_chain": blocks,
        "planned_replicates": replicates,
        "valid_replicates": replicates,
        "failed_replicates": 0,
        "abstained_replicates": 0,
        "planned_chain_states": blocks * replicates,
        "valid_chain_states": blocks * replicates,
        "failed_chain_states": 0,
        "true_uniform_mean": true_uniform_mean,
        "empirical_uniform_lcb_coverage": successes / replicates,
        "coverage_wilson95_low": cover_ci[0],
        "coverage_wilson95_high": cover_ci[1],
        "nominal_conditional_coverage": 1.0 - ALPHA_SCORE,
        "positive_lcb_replicates": positive,
        "positive_lcb_rate": positive / replicates,
        "median_uniform_lcb": float(np.median(uniform_lcb)),
        "minimum_uniform_lcb": float(np.min(uniform_lcb)),
        "maximum_uniform_lcb": float(np.max(uniform_lcb)),
        "markov_radius": radius,
        "target_grid_points": len(grid),
        "max_grid_gap_over_R_upper_bound": mesh_penalty,
        "seed": seed,
    }
    return rows, summary


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plot(summary: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gammas = [row["gamma_ps"] for row in summary]
    rates = [row["empirical_uniform_lcb_coverage"] for row in summary]
    low = [row["coverage_wilson95_low"] for row in summary]
    high = [row["coverage_wilson95_high"] for row in summary]
    errors = [[rate - lo for rate, lo in zip(rates, low)],
              [hi - rate for rate, hi in zip(rates, high)]]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    ax = axes[0]
    ax.errorbar(gammas, rates, yerr=errors, marker="o", capsize=4,
                color="#276c91", label="empirical lower-bound coverage")
    ax.axhline(1.0 - ALPHA_SCORE, color="#a13b36", linestyle="--",
               label="conditional nominal 96%")
    ax.set_ylim(0.90, 1.005)
    ax.set_xlabel(r"known pseudo-spectral gap $\gamma_{ps}$")
    ax.set_ylabel("coverage across 1,000 independent chains")
    ax.set_title("Known two-state reversible chain")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    medians = [row["median_uniform_lcb"] for row in summary]
    positives = [row["positive_lcb_rate"] for row in summary]
    ax.plot(gammas, medians, marker="s", color="#6a4c93",
            label="median setwise LCB")
    ax.axhline(0.25, color="black", linestyle=":", label="true uniform mean = 0.25")
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel(r"known pseudo-spectral gap $\gamma_{ps}$")
    ax.set_ylabel("score units")
    ax.set_title("Coverage can be conservative and uninformative")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.text(0.02, 0.02,
            "Positive-LCB rate: " + ", ".join(f"{p:.1%}" for p in positives),
            transform=ax.transAxes, fontsize=8, va="bottom")
    fig.suptitle("Conditional synthetic screen; not physical calibration")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def build_report(output: str | Path = DEFAULT_OUT) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    all_rows: list[dict] = []
    summary: list[dict] = []
    for gamma in GAMMA_GRID:
        rows, row_summary = _cell(gamma)
        all_rows.extend(rows)
        summary.append(row_summary)

    with (out / "replicate_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)

    report_grid, report_mesh_penalty, _ = _target_grid(
        *TARGET_INTERVAL,
        score_scale_R=SCORE_SCALE_R,
        mesh_step_R=MESH_STEP_R,
        max_grid_points=4096,
    )
    source_files = {
        "coverage_screen": Path(__file__).resolve(),
        "setwise_markov": TASK_DIR / "bounded_block_risk_setwise_markov.py",
        "markov_varying": TASK_DIR / "bounded_block_risk_markov_varying.py",
        "markov_gap": TASK_DIR / "bounded_block_risk_markov_gap.py",
        "core_numerics": TASK_DIR.parents[1] / "core" / "numerics.py",
    }
    protocol = {
        "schema": "pion-markov-setwise-coverage-screen-v1",
        "classification": "CONDITIONAL_SYNTHETIC_MARKOV_COVERAGE_SCREEN_NOT_CALIBRATION",
        "chain": {
            "transition_matrix": "[[1-gamma/2,gamma/2],[gamma/2,1-gamma/2]]",
            "stationary_distribution": [0.5, 0.5],
            "pseudo_spectral_gap": "gamma for gamma in (0,1]",
            "initial_law": "exact stationary draw; no burn-in",
            "score_states": {"state_0": 0.0, "state_1": 1.0},
            "candidate_prediction": 0.5,
        },
        "target_interval": list(TARGET_INTERVAL),
        "prediction_domain": list(PREDICTION_DOMAIN),
        "score_scale_R": SCORE_SCALE_R,
        "score": "clip(((m-reference)^2-(m-candidate)^2)/R^2,-1,1)",
        "true_uniform_mean": 0.25,
        "planned_blocks_per_chain": PLANNED_BLOCKS,
        "replicates_per_gamma": REPLICATES_PER_GAMMA,
        "gamma_grid": list(GAMMA_GRID),
        "seed_rule": "202611281103 + round(1000*gamma_ps)",
        "alpha_score": ALPHA_SCORE,
        "alpha_anchor_reserved_but_not_used_for_fixed_target_set": ALPHA_ANCHOR_RESERVED,
        "alpha_total": ALPHA_TOTAL,
        "family_size_outside_grid": 1,
        "simultaneous_family_size": len(report_grid),
        "grid_endpoints": list(TARGET_INTERVAL),
        "target_grid": report_grid,
        "maximum_grid_gap_over_R_upper_bound": report_mesh_penalty,
        "mesh_step_R": MESH_STEP_R,
        "failed_block_rule": "none in this exact-state simulation; production failures remain score -1",
        "data_access": "synthetic two-state chains only; no lattice, gauge, anchor, E-reference, or sealed bytes",
        "physical_score": None,
        "certified": False,
        "source_sha256": {
            str(path.relative_to(TASK_DIR.parents[1])): _sha(path)
            for path in source_files.values()
        },
        "numpy_version": np.__version__,
    }
    (out / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    _plot(summary, out / "coverage_and_sharpness.png")
    report = {
        "schema": protocol["schema"],
        "classification": protocol["classification"],
        "physical_score": None,
        "certified": False,
        "denominators": {
            "planned_chains": len(GAMMA_GRID) * REPLICATES_PER_GAMMA,
            "valid_chains": len(GAMMA_GRID) * REPLICATES_PER_GAMMA,
            "failed_chains": 0,
            "abstained_chains": 0,
        },
        "summary": summary,
    }
    (out / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    artifact_names = [
        "protocol.json", "summary.json", "summary.csv", "replicate_results.csv",
        "coverage_and_sharpness.png",
    ]
    (out / "artifact_hashes.json").write_text(
        json.dumps({name: _sha(out / name) for name in artifact_names},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = [
        "# Pion setwise Markov lower-bound coverage screen",
        "",
        "**Status:** `CONDITIONAL_SYNTHETIC_MARKOV_COVERAGE_SCREEN_NOT_CALIBRATION`. `physical_score=null`; no lattice, anchor, E-reference, gauge, or sealed input was read.",
        "",
        "## Frozen construction",
        "",
        "A two-state reversible chain uses transition matrix `[[1-gamma/2,gamma/2],[gamma/2,1-gamma/2]]`, stationary law `(1/2,1/2)`, exact stationary starts, and pseudo-spectral gap `gamma`. Each independent replicate contains 4,096 planned states. The reference prediction is the current state (0 or 1), the candidate prediction is 0.5, `R=1`, and the fixed common target interval is `[0.45,0.55]`.",
        "",
        "For either stationary state, the normalized risk difference has expectation exactly `0.25` for every target `m` in the interval. The screen calls the production helper's exact-rational grid builder, counts every generated grid point in the simultaneous family, applies the support-gap Eq. (3.26) radius, and subtracts its upward-rounded `h/R` Lipschitz cover penalty. It counts a replicate as covered only when the uniform LCB does not exceed the exact value `0.25`.",
        "",
        "## Results",
        "",
        "| gamma_ps | chains | empirical coverage | Wilson 95% interval | median uniform LCB | positive-LCB rate |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        readme.append(
            f"| {row['gamma_ps']:.2f} | {row['planned_replicates']} | "
            f"{row['empirical_uniform_lcb_coverage']:.3f} | "
            f"[{row['coverage_wilson95_low']:.3f}, {row['coverage_wilson95_high']:.3f}] | "
            f"{row['median_uniform_lcb']:.3f} | {row['positive_lcb_rate']:.3f} |"
        )
    readme += [
        "",
        "![Coverage and lower-bound sharpness](coverage_and_sharpness.png)",
        "",
        "## Interpretation and limits",
        "",
        "The left panel checks conditional lower-bound coverage under a known, reversible two-state chain with correctly supplied gap. The right panel shows informativeness: high coverage alone is insufficient if the LCB remains below zero. This checks only one exact finite-state family and the implemented grid/cover arithmetic; it does not validate an HMC trajectory's gap, initial TV, block schedule, target-set coverage, anchor-to-E bridge, or real score support. `alpha_anchor=0.01` is reserved but no random anchor set is used in this fixed-interval check.",
        "",
        "Data flow: `frozen two-state transition -> independent exact-stationary chains -> statewise bounded scores on the production helper's full target grid -> support-gap Markov radius with the exact grid family -> h/R cover penalty -> coverage/LCB ledger and figure`. No failed draws occurred; all planned states and replicates remain in the denominator.",
        "",
        "This does not close Pion's target mass/source correspondence, common-bias bound, production dependence receipts, independent review, or TaskSpec/evaluator integration. Keep `NOT_READY/UNCERTIFIED`, `physical_score=null`, and sealed inputs closed.",
        "",
        "Artifact digests are recorded in `artifact_hashes.json`; `protocol.json` freezes seeds and assumptions.",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    # Include the human-readable report in the immutable artifact digest set.
    hashes = {name: _sha(out / name) for name in artifact_names + ["README.md"]}
    (out / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
