"""Conditional Markov pseudo-spectral-gap bound sensitivity for Pion scores.

This is an analytic planning diagnostic. It samples no scores and reads no
physical, anchor, validation, holdout, GPU, or QUDA inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from numbers import Real
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/pion_projection/bounded_block_risk_markov_gap_20261124_v3"
BLOCKS_GRID = (20, 50, 100, 200, 500)
GAP_GRID = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
FAMILY = 2
ALPHA_S = 0.05
VARIANCE_BOUND = 1.0
CENTERED_ABS_BOUND = 2.0
INITIAL_TV_GRID = (0.0, 0.001, 0.005, 0.01, 0.02, 0.025)
INITIAL_TV_BLOCKS = (20, 500)


def _finite_real(value: object) -> bool:
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def markov_gap_radius(*, blocks: int, gamma_ps_lower: float,
                       family: int = FAMILY, alpha_s: float = ALPHA_S,
                       variance_bound: float = VARIANCE_BOUND,
                       centered_abs_bound: float = CENTERED_ABS_BOUND) -> float:
    """Return a conservative simultaneous score-mean radius.

    Solves Paulin, arXiv:1212.2015v5, Theorem 3.11 / Eq. (3.25)'s
    non-reversible Bernstein exponent using the worst-case bounds Var(Z)<=1
    and |Z-EZ|<=2 for Z in [-1, 1]. The author notes this version corrects an
    incomplete proof of the general-state Bernstein inequalities in previous
    versions, including the published version.
    ``gamma_ps_lower`` must be a justified lower bound for the Markov chain
    whose states are the predeclared block observations. The theorem's single
    function f must be the same fixed score map for every block; the
    time-varying f_i extension in Eq. (3.26) is not implemented here.
    """
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 1:
        raise ValueError("blocks must be a positive integer")
    if isinstance(family, bool) or not isinstance(family, int) or family < 1:
        raise ValueError("family must be a positive integer")
    if (not _finite_real(gamma_ps_lower)
            or not 0 < gamma_ps_lower <= 1):
        raise ValueError("gamma_ps_lower must be finite and in (0, 1]")
    if not _finite_real(alpha_s) or not 0 < alpha_s < 1:
        raise ValueError("alpha_s must be finite and in (0, 1)")
    if (not _finite_real(variance_bound) or variance_bound < 0
            or not _finite_real(centered_abs_bound)
            or centered_abs_bound <= 0):
        raise ValueError("invalid Bernstein range or variance bound")

    log_tail = math.log(2 * family / alpha_s)
    gamma = float(gamma_ps_lower)
    variance_bound = float(variance_bound)
    centered_abs_bound = float(centered_abs_bound)
    linear = 20 * centered_abs_bound * log_tail
    variance = 8 * log_tail * (blocks + 1 / gamma) * variance_bound
    sum_deviation = (linear + math.sqrt(linear * linear + 4 * gamma * variance)) / (2 * gamma)
    radius = sum_deviation / blocks
    if not math.isfinite(radius):
        raise ValueError("Markov pseudo-gap radius is non-finite for these settings")
    return radius


def markov_gap_radius_after_initial_tv(
    *, blocks: int, gamma_ps_lower: float, family: int, alpha_score: float,
    initial_law_tv_bound: float,
) -> tuple[float, float]:
    """Return the radius and stationary alpha left after an initial-law TV term.

    ``alpha_score`` is a family-wise score error budget. The supplied TV bound
    is per comparison after the declared burn-in, so the family union bound
    spends ``family * initial_law_tv_bound`` before applying the stationary
    Bernstein inequality. The score-mean LCB also subtracts ``2 * TV``.
    """
    if (not _finite_real(initial_law_tv_bound)
            or not 0 <= initial_law_tv_bound <= 1):
        raise ValueError("initial_law_tv_bound must be finite and in [0, 1]")
    if (isinstance(family, bool) or not isinstance(family, int)
            or family < 1):
        raise ValueError("family must be a positive integer")
    if not _finite_real(alpha_score) or not 0 < alpha_score < 1:
        raise ValueError("alpha_score must be finite and in (0, 1)")
    stationary_alpha = alpha_score - family * float(initial_law_tv_bound)
    if stationary_alpha <= 0:
        raise ValueError(
            "family * initial_law_tv_bound must be smaller than alpha_score"
        )
    radius = markov_gap_radius(
        blocks=blocks,
        gamma_ps_lower=gamma_ps_lower,
        family=family,
        alpha_s=stationary_alpha,
    )
    return radius, stationary_alpha


def _validate_alpha_allocation(*, alpha_score: float, alpha_anchor: float,
                               alpha_total: float) -> None:
    if not _finite_real(alpha_score) or not 0 < alpha_score < 1:
        raise ValueError("alpha_score must be finite and in (0, 1)")
    if not _finite_real(alpha_anchor) or not 0 <= alpha_anchor < 1:
        raise ValueError("alpha_anchor must be finite and in [0, 1)")
    if not _finite_real(alpha_total) or not 0 < alpha_total < 1:
        raise ValueError("alpha_total must be finite and in (0, 1)")
    if alpha_score + alpha_anchor > alpha_total:
        raise ValueError("alpha_score + alpha_anchor must not exceed alpha_total")


def conditional_block_risk_markov_lcb(
    raw_differences: object,
    *,
    valid_blocks: object,
    score_scale_R: float,
    gamma_ps_lower: float,
    family: int,
    alpha_score: float,
    alpha_anchor: float,
    alpha_total: float,
    relative_anchor_bound_B_over_R: float,
    burn_in_blocks: int,
    initial_law_tv_bound: float,
) -> dict:
    """Compute a conditional Markov-chain LCB over the full planned block set.

    This is a math helper only. A valid lower bound on the *block-chain* pseudo
    spectral gap and every theorem condition must be supplied by the caller.
    The same frozen score function must be applied to every scored block.
    The initial-law TV bound is per comparison after ``burn_in_blocks``; it
    pays both a familywise probability cost and a conservative 2*TV shift in
    the finite-plan score mean. Failed blocks stay in planned N and receive
    score -1. An all-failed comparison abstains. No physical score is emitted.
    """
    import numpy as np

    from .bounded_block_risk_score import bounded_block_scores

    differences = np.asarray(raw_differences, dtype=float)
    valid = np.asarray(valid_blocks)
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("raw_differences must be a nonempty one-dimensional array")
    if valid.ndim != 1 or valid.shape != differences.shape:
        raise ValueError("valid_blocks must match the planned block array")
    scores = bounded_block_scores(
        differences, valid_blocks=valid, score_scale_R=score_scale_R,
    )
    if (isinstance(family, bool) or not isinstance(family, int)
            or family < 1):
        raise ValueError("family must be a positive integer")
    _validate_alpha_allocation(
        alpha_score=alpha_score, alpha_anchor=alpha_anchor,
        alpha_total=alpha_total,
    )
    if (not _finite_real(relative_anchor_bound_B_over_R)
            or relative_anchor_bound_B_over_R < 0):
        raise ValueError("relative_anchor_bound_B_over_R must be finite and nonnegative")
    if (isinstance(burn_in_blocks, bool) or not isinstance(burn_in_blocks, int)
            or burn_in_blocks < 0):
        raise ValueError("burn_in_blocks must be a nonnegative integer")
    if (not _finite_real(initial_law_tv_bound)
            or not 0 <= initial_law_tv_bound <= 1):
        raise ValueError("initial_law_tv_bound must be finite and in [0, 1]")

    planned = int(differences.size)
    valid_count = int(valid.sum())
    failed_count = planned - valid_count
    radius, stationary_alpha = markov_gap_radius_after_initial_tv(
        blocks=planned,
        gamma_ps_lower=gamma_ps_lower,
        family=family,
        alpha_score=alpha_score,
        initial_law_tv_bound=initial_law_tv_bound,
    )
    planned_mean = float(scores.mean())
    anchor_penalty = 2.0 * float(relative_anchor_bound_B_over_R)
    initial_law_mean_penalty = 2.0 * float(initial_law_tv_bound)
    denominators = {
        "planned_blocks": planned,
        "valid_blocks": valid_count,
        "failed_blocks": failed_count,
        "abstained_blocks": 0,
    }
    common = {
        "estimand": "bounded_block_risk",
        "dependence_method": "markov_pseudo_gap_bernstein",
        "gamma_ps_lower_bound_assumed": float(gamma_ps_lower),
        "alpha_allocation": {
            "score": float(alpha_score),
            "anchor": float(alpha_anchor),
            "total": float(alpha_total),
        },
        "initial_law": {
            "burn_in_blocks": burn_in_blocks,
            "tv_bound_per_comparison": float(initial_law_tv_bound),
            "familywise_probability_cost": family * float(initial_law_tv_bound),
            "stationary_score_alpha_after_tv": float(stationary_alpha),
            "score_mean_shift_penalty": initial_law_mean_penalty,
        },
        "certified": False,
        "data_access": "none",
        "physical_score": None,
        "result_unit": "one planned candidate-reference comparison",
        "denominators": denominators,
        "result_denominators": {
            "planned": 1,
            "valid": int(valid_count > 0),
            "failed": 0,
            "abstained": int(valid_count == 0),
        },
        "normalized_block_scores": scores.tolist(),
        "planned_block_mean": planned_mean,
        "relative_anchor_bound_B_over_R": float(relative_anchor_bound_B_over_R),
        "anchor_penalty": anchor_penalty,
        "assumptions": [
            "the predeclared block chain is homogeneous and satisfies the corrected theorem conditions with a stationary law",
            "the same fixed, predeclared bounded score function is applied to every block-chain state",
            "gamma_ps_lower is a valid auditable lower bound for that block chain",
            "the theorem's range/variance bounds apply to all planned coded scores in [-1, 1]",
            "after the declared burn-in, each comparison's initial block-state law is within the supplied TV bound of stationarity; TV contraction preserves this for later scored blocks",
            "the finite-plan score mean is within 2*TV of the stationary mean because scores lie in [-1, 1]",
            "the score concentration bound is valid conditional on a frozen anchor, or a valid joint bound accounts for any score-anchor dependence",
            "the supplied anchor bound has coverage at least 1-alpha_anchor",
            "|y_candidate-y_reference| <= score_scale_R for the squared-error correction",
        ],
    }
    if valid_count == 0:
        return {
            **common,
            "status": "ABSTAIN_NO_VALID_BLOCKS",
            "valid_block_mean": None,
            "concentration_penalty": None,
            "initial_law_mean_penalty": initial_law_mean_penalty,
            "normalized_lower_bound": None,
        }

    valid_mean = float(scores[valid].mean())
    lower_bound = (
        planned_mean - radius - anchor_penalty - initial_law_mean_penalty
    )
    return {
        **common,
        "status": "CONDITIONAL_MARKOV_PSEUDOGAP_BOUND_ONLY",
        "valid_block_mean": valid_mean,
        "concentration_penalty": radius,
        "initial_law_mean_penalty": initial_law_mean_penalty,
        "normalized_lower_bound": lower_bound,
    }


def graph_color_radius(*, blocks: int, chi: int, family: int = FAMILY,
                       alpha_s: float = ALPHA_S) -> float:
    """Existing graph-color Hoeffding radius, included as conditional context."""
    return math.sqrt(2 * chi * math.log(family / alpha_s) / blocks)


def make_grid() -> list[dict]:
    rows = []
    for blocks in BLOCKS_GRID:
        for gamma in GAP_GRID:
            radius = markov_gap_radius(blocks=blocks, gamma_ps_lower=gamma)
            rows.append({
                "planned_blocks": blocks,
                "gamma_ps_lower_assumed": gamma,
                "family": FAMILY,
                "alpha_s_assumed": ALPHA_S,
                "variance_bound": VARIANCE_BOUND,
                "centered_abs_bound": CENTERED_ABS_BOUND,
                "log_tail": math.log(2 * FAMILY / ALPHA_S),
                "score_radius": radius,
                "zero_threshold_reachable_at_best_mean": radius < 1.0,
                "status": ("CONDITIONALLY_FEASIBLE" if radius < 1.0
                           else "ABSTAIN_UNATTAINABLE_AT_ZERO_THRESHOLD"),
            })
    return rows


def make_initial_tv_sensitivity() -> list[dict]:
    """Illustrate the corrected theorem's nonstationary initial-law cost."""
    rows = []
    for blocks in INITIAL_TV_BLOCKS:
        for tv_bound in INITIAL_TV_GRID:
            remaining_alpha = ALPHA_S - FAMILY * tv_bound
            if remaining_alpha <= 0:
                rows.append({
                    "planned_blocks": blocks,
                    "gamma_ps_lower_assumed": 1.0,
                    "family": FAMILY,
                    "alpha_score_total": ALPHA_S,
                    "initial_law_tv_per_comparison": tv_bound,
                    "familywise_tv_cost": FAMILY * tv_bound,
                    "remaining_stationary_alpha": 0.0,
                    "stationary_radius": None,
                    "mean_shift_penalty_2tv": 2 * tv_bound,
                    "total_score_penalty": None,
                    "zero_threshold_reachable_at_best_mean": False,
                    "status": "ABSTAIN_NO_STATIONARY_ALPHA_LEFT",
                })
                continue
            radius = markov_gap_radius(
                blocks=blocks,
                gamma_ps_lower=1.0,
                family=FAMILY,
                alpha_s=remaining_alpha,
            )
            total_penalty = radius + 2 * tv_bound
            rows.append({
                "planned_blocks": blocks,
                "gamma_ps_lower_assumed": 1.0,
                "family": FAMILY,
                "alpha_score_total": ALPHA_S,
                "initial_law_tv_per_comparison": tv_bound,
                "familywise_tv_cost": FAMILY * tv_bound,
                "remaining_stationary_alpha": remaining_alpha,
                "stationary_radius": radius,
                "mean_shift_penalty_2tv": 2 * tv_bound,
                "total_score_penalty": total_penalty,
                "zero_threshold_reachable_at_best_mean": total_penalty < 1.0,
                "status": (
                    "CONDITIONALLY_FEASIBLE" if total_penalty < 1.0
                    else "ABSTAIN_UNATTAINABLE_AT_ZERO_THRESHOLD"
                ),
            })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(out: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    lookup = {(row["planned_blocks"], row["gamma_ps_lower_assumed"]):
              row["score_radius"] for row in rows}
    matrix = np.asarray([
        [lookup[(n, gamma)] for gamma in GAP_GRID]
        for n in BLOCKS_GRID
    ])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    mesh = axes[0].pcolormesh(
        GAP_GRID, BLOCKS_GRID, matrix,
        shading="nearest", norm=LogNorm(vmin=matrix.min(), vmax=matrix.max()),
        cmap="magma_r",
    )
    axes[0].set_xscale("log")
    axes[0].set_xticks(GAP_GRID, labels=[f"{v:g}" for v in GAP_GRID])
    axes[0].set_yticks(BLOCKS_GRID, labels=[str(v) for v in BLOCKS_GRID])
    axes[0].set_xlabel(r"assumed lower bound $\underline{\gamma}_{ps}$")
    axes[0].set_ylabel("planned blocks N")
    axes[0].set_title("Markov-chain radius (worst-case bounded variance)")
    axes[0].contour(
        GAP_GRID, BLOCKS_GRID, matrix, levels=[1.0], colors=["cyan"],
        linewidths=1.8,
    )
    for row_i, n in enumerate(BLOCKS_GRID):
        for col_i, gamma in enumerate(GAP_GRID):
            axes[0].text(gamma, n, f"{matrix[row_i, col_i]:.2g}",
                         ha="center", va="center", fontsize=7,
                         color="white" if matrix[row_i, col_i] > 30 else "black")
    fig.colorbar(mesh, ax=axes[0], label=r"score radius $\epsilon_{ps}$")

    for gamma in (1.0, 0.2, 0.05):
        values = [lookup[(n, gamma)] for n in BLOCKS_GRID]
        axes[1].plot(BLOCKS_GRID, values, marker="o",
                     label=fr"Markov bound, $\underline{{\gamma}}_{{ps}}={gamma:g}$")
    for chi in (1, 2, 4):
        values = [graph_color_radius(blocks=n, chi=chi) for n in BLOCKS_GRID]
        axes[1].plot(BLOCKS_GRID, values, marker="s", linestyle="--",
                     label=fr"graph Hoeffding, $\chi={chi}$")
    axes[1].axhline(1.0, color="black", linewidth=1.0, linestyle=":",
                    label="best possible sample mean at zero threshold")
    axes[1].set_yscale("log")
    axes[1].set_xticks(BLOCKS_GRID, labels=[str(v) for v in BLOCKS_GRID])
    axes[1].set_xlabel("planned blocks N")
    axes[1].set_ylabel("simultaneous score penalty")
    axes[1].set_title("Conditional routes; assumptions differ")
    axes[1].legend(fontsize=7, ncol=1)
    fig.suptitle("Analytic sensitivity only — no Pion scores or chain receipts")
    fig.savefig(out / "markov_gap_sensitivity.png", dpi=180)
    plt.close(fig)


def _write_initial_tv_plot(out: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for blocks in INITIAL_TV_BLOCKS:
        group = [row for row in rows if row["planned_blocks"] == blocks]
        x = [row["initial_law_tv_per_comparison"] for row in group
             if row["total_score_penalty"] is not None]
        y = [row["total_score_penalty"] for row in group
             if row["total_score_penalty"] is not None]
        ax.plot(x, y, marker="o", label=f"N={blocks}")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1,
               label="zero-threshold reachability boundary")
    ax.axvline(ALPHA_S / FAMILY, color="red", linestyle="--", linewidth=1,
               label="no stationary alpha remains")
    ax.set_yscale("log")
    ax.set_xticks(INITIAL_TV_GRID, labels=[f"{v:g}" for v in INITIAL_TV_GRID])
    ax.set_xlabel("assumed post-burn-in TV bound per comparison")
    ax.set_ylabel("Markov radius + finite-plan mean-shift penalty")
    ax.set_title(
        "Initial-law correction sensitivity (F=2, alpha_score=0.05, gamma=1)"
    )
    ax.legend()
    fig.savefig(out / "initial_law_tv_sensitivity.png", dpi=180)
    plt.close(fig)


def _synthetic_lcb_demo() -> dict:
    """Exercise stationarity, finite-TV, and all-failed synthetic panels."""
    import numpy as np

    calibration = {
        "score_scale_R": 2.0,
        "gamma_ps_lower": 1.0,
        "family": 2,
        "alpha_score": 0.04,
        "alpha_anchor": 0.01,
        "alpha_total": 0.05,
        "relative_anchor_bound_B_over_R": 0.03,
        "burn_in_blocks": 0,
        "initial_law_tv_bound": 0.0,
    }
    partially_failed = conditional_block_risk_markov_lcb(
        [8.0, -8.0, 2.0, np.nan],
        valid_blocks=np.asarray([True, True, True, False]),
        **calibration,
    )
    all_failed = conditional_block_risk_markov_lcb(
        [np.nan, np.nan],
        valid_blocks=np.asarray([False, False]),
        **calibration,
    )
    nonstationary_calibration = {
        **calibration,
        "burn_in_blocks": 8,
        "initial_law_tv_bound": 0.005,
    }
    nonstationary = conditional_block_risk_markov_lcb(
        [8.0, -8.0, 2.0, np.nan],
        valid_blocks=np.asarray([True, True, True, False]),
        **nonstationary_calibration,
    )
    return {
        "classification": "SYNTHETIC_HELPER_INTEGRATION_ONLY",
        "physical_score": None,
        "certified": False,
        "data_access": "none",
        "calibration": calibration,
        "input_encoding": (
            "JSON null in raw_differences denotes a failed block and is passed to the helper as NaN; "
            "the explicit valid_blocks mask controls failure coding."
        ),
        "cases": [
            {
                "name": "partially_failed_panel",
                "raw_differences": [8.0, -8.0, 2.0, None],
                "valid_blocks": [True, True, True, False],
                "result": partially_failed,
            },
            {
                "name": "all_failed_panel",
                "raw_differences": [None, None],
                "valid_blocks": [False, False],
                "result": all_failed,
            },
            {
                "name": "partially_failed_panel_with_initial_tv_bound",
                "raw_differences": [8.0, -8.0, 2.0, None],
                "valid_blocks": [True, True, True, False],
                "calibration": nonstationary_calibration,
                "result": nonstationary,
            },
        ],
        "comparison_denominators": {
            "planned": 3,
            "valid": 2,
            "failed": 0,
            "abstained": 1,
        },
        "block_denominators": {
            "planned": 10,
            "valid": 6,
            "failed": 4,
            "abstained": 0,
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out}")
    out.mkdir(parents=True)
    rows = make_grid()
    tv_rows = make_initial_tv_sensitivity()
    demo = _synthetic_lcb_demo()
    feasible = sum(row["status"] == "CONDITIONALLY_FEASIBLE" for row in rows)
    tv_feasible = sum(
        row["status"] == "CONDITIONALLY_FEASIBLE" for row in tv_rows
    )
    summary = {
        "classification": "SYNTHETIC_ANALYTIC_SENSITIVITY_ONLY",
        "physical_score": None,
        "scenario_denominators": {
            "planned": len(rows), "valid": feasible,
            "failed": 0, "abstained": len(rows) - feasible,
        },
        "physical_block_score_denominators": {
            "planned": 0, "valid": 0, "failed": 0, "abstained": 0,
        },
        "synthetic_helper_demo_comparisons": demo["comparison_denominators"],
        "synthetic_helper_demo_blocks": demo["block_denominators"],
        "initial_law_tv_sensitivity_denominators": {
            "planned": len(tv_rows),
            "valid": tv_feasible,
            "failed": 0,
            "abstained": len(tv_rows) - tv_feasible,
        },
        "scenario_weighted_hypothetical_planned_blocks": sum(
            row["planned_blocks"] for row in rows
        ),
        "feasible_scenarios": feasible,
        "unattainable_zero_threshold_scenarios": len(rows) - feasible,
        "smallest_planned_blocks": min(BLOCKS_GRID),
        "largest_planned_blocks": max(BLOCKS_GRID),
        "maximum_assumed_pseudo_gap_lower_bound": max(GAP_GRID),
        "baseline_at_N20_gamma1": next(
            row["score_radius"] for row in rows
            if row["planned_blocks"] == 20
            and row["gamma_ps_lower_assumed"] == 1.0
        ),
        "best_case_at_N500_gamma1": next(
            row["score_radius"] for row in rows
            if row["planned_blocks"] == 500
            and row["gamma_ps_lower_assumed"] == 1.0
        ),
        "warning": (
            "Feasibility is only hypothetical and assumes zero target, zero anchor "
            "correction, stationary initialization, a known valid lower "
            "bound on the block-chain pseudo-spectral gap, and all theorem conditions."
        ),
    }
    protocol = {
        "classification": summary["classification"],
        "inputs_read": [],
        "data_access": "none; no physical or sealed inputs were opened",
        "formula_source": {
            "paper": "Paulin, arXiv:1212.2015v5 (2018), Theorem 3.11 / Eq. (3.25) and Proposition 3.15 / Eq. (3.29)",
            "url": "https://arxiv.org/html/1212.2015v5",
            "bound": "P(|S-E S|>=t) <= 2 exp(-t^2 gamma_ps / (8(N+1/gamma_ps)V_f + 20 t C))",
            "version_note": "The author notes that an argument used for the general-state Bernstein proofs in previous versions, including the published version, was incomplete; this report follows the corrected arXiv v5 statement.",
        },
        "block_chain_contract": [
            "Predeclare non-overlapping score blocks and represent each block as one state of a homogeneous block Markov chain.",
            "The reference block chain must be homogeneous, phi-irreducible, aperiodic, Polish, and have a unique stationary law as required by Assumption 3.1.",
            "Theorem 3.11 / Eq. (3.25) uses one fixed score function f across all chain states; a time-varying score sequence requires the separate Eq. (3.26) route and is outside this helper.",
            "For nonstationary starts, provide an audited post-burn-in total-variation bound delta per comparison; spend F*delta from alpha_score and subtract 2*delta from the normalized finite-plan mean bound.",
            "Use an auditable lower confidence bound gamma_ps_lower > 0 for this block chain; trajectory spacing and empirical ESS are not such a bound.",
            "For Z_i in [-1,1], use C=2 and V_f<=1; failures remain mapped to -1 and stay in planned N.",
            "The score-family union bound uses log(2F/alpha_s), because the cited theorem is two-sided.",
            "The score bound is conditional on a frozen anchor; if anchor and score share trajectory dependence, the Markov score bound must remain valid conditionally or a valid joint bound must be supplied.",
            "Anchor coverage/bias and the target threshold remain separate from the Markov and initial-law corrections.",
            "Synthetic fixtures exercise stationary, nonstationary-TV, and all-failed code paths; they are not project data or a coverage test.",
        ],
        "grid": {
            "planned_blocks": list(BLOCKS_GRID),
            "gamma_ps_lower_assumed": list(GAP_GRID),
            "family": FAMILY,
            "alpha_s": ALPHA_S,
            "variance_bound": VARIANCE_BOUND,
            "centered_abs_bound": CENTERED_ABS_BOUND,
            "burn_in_blocks": 0,
            "initial_law_tv_bound": 0.0,
            "target_threshold": 0.0,
            "anchor_correction": 0.0,
            "failure_fraction": 0.0,
            "warning": "All values are hypothetical, not frozen or estimated project inputs.",
        },
    }
    _write_csv(out / "sensitivity_grid.csv", rows)
    _write_csv(out / "initial_law_tv_sensitivity.csv", tv_rows)
    scenario_denominators = [
        {
            "unit": "analytic planning scenario",
            "planned": summary["scenario_denominators"]["planned"],
            "valid": summary["scenario_denominators"]["valid"],
            "failed": summary["scenario_denominators"]["failed"],
            "abstained": summary["scenario_denominators"]["abstained"],
            "interpretation": "Valid means a zero-threshold LCB is mathematically reachable at the maximum possible sample mean under the illustrative inputs; not an observed or certified result.",
        },
        {
            "unit": "physical block-score outcome",
            "planned": 0, "valid": 0, "failed": 0, "abstained": 0,
            "interpretation": "No physical block score was sampled; these are not measurement denominators.",
        },
        {
            "unit": "synthetic Markov-LCB comparison fixtures",
            **demo["comparison_denominators"],
            "interpretation": "Three fixed software fixtures: two conditional helper results (stationary and initial-TV adjusted) and one all-failed abstention; no random experiment or physical sample.",
        },
        {
            "unit": "synthetic Markov-LCB planned block slots",
            **demo["block_denominators"],
            "interpretation": "The three software fixtures contain ten hypothetical slots; six valid and four explicitly failed slots, with no block removed.",
        },
        {
            "unit": "initial-law TV sensitivity scenario",
            **summary["initial_law_tv_sensitivity_denominators"],
            "interpretation": "A 12-cell deterministic what-if grid at N=20/500 and assumed per-comparison TV bounds; valid means zero-threshold reachability at the best possible mean, not an observed result.",
        },
    ]
    _write_csv(out / "denominator_summary.csv", scenario_denominators)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "markov_lcb_demo.json").write_text(json.dumps(demo, indent=2) + "\n")
    _write_plot(out, rows)
    _write_initial_tv_plot(out, tv_rows)
    shutil.copy2(Path(__file__), out / "source_snapshot.py")

    readme = f"""# Pion bounded-risk Markov gap sensitivity

**SYNTHETIC ANALYTIC SENSITIVITY ONLY. `physical_score=null`.** This report evaluates a published Markov-chain concentration inequality on hypothetical block counts, pseudo-spectral-gap lower bounds, and initial-law TV bounds. It samples no scores and reads no ensemble, anchor, gap, validation, holdout, GPU, or QUDA input.

## Corrected theorem source

This report follows Paulin's corrected arXiv v5 statement: Theorem 3.11 / Eq. (3.25) for the stationary non-reversible Bernstein bound and Proposition 3.15 / Eq. (3.29) for a non-stationary initial law. The author notes that the proof argument for general-state Bernstein inequalities in earlier versions, including the published version, was incomplete and is corrected in v5. The v1/v2 outputs used the older 2015 / Theorem 3.10 citation; their displayed concentration constants match the corrected statement, but cite this corrected source note going forward. See [Paulin, arXiv:1212.2015v5](https://arxiv.org/html/1212.2015v5).

## Conditional stationary bound

If each planned score `Z_i` is the same fixed bounded function `f` of a predeclared, non-overlapping block state, and the reference block process is a homogeneous, `phi`-irreducible, aperiodic Markov chain on a Polish space with a unique stationary law, then the corrected Theorem 3.11 gives, for `S=sum_i f(X_i)`,

```text
P_pi(|S-E_pi S| >= t) <= 2 exp[-t^2 gamma_ps / (8 (N + 1/gamma_ps) V_f + 20 t C)]
```

For `Z_i in [-1,1]`, `C=2` and `V_f<=1`. With a family of `F` comparisons and family-wise score error budget `alpha_s`, set `L=log(2F/alpha_s)`. Solving the exponent gives

```text
epsilon_ps = [20 C L + sqrt((20 C L)^2 + 32 gamma_ps L (N+1/gamma_ps) V_f)] / (2 gamma_ps N).
```

`gamma_ps_lower` must be an auditable lower bound for the **block chain itself**. The score bound must be valid conditional on a frozen anchor, or a valid joint argument must account for score-anchor dependence. An anchor estimated from the same trajectory does not become independent merely because the code conditions on it.

The v1 helper deliberately accepts only this fixed-score route. Paulin's Eq. (3.26) also gives a stationary bound for `sum_i f_i(X_i)` with time-varying functions, using the variance sum `V_sum=sum_i Var_pi(f_i)` and an additional `M/k_ps` range factor. Supporting that route would require new reviewed inputs and validation for `k_ps`, variance-sum upper bounds, per-index centered ranges, and a finite justified upper bound on `M`; no such values are inferred here. Until that contract and implementation exist, block-dependent score maps must use another established bound or abstain.

## Non-stationary initial-law correction

Suppose the first scored block state, after discarding `burn_in_blocks`, has law `q_t` with an externally justified bound `delta >= d_TV(q_t, pi)`. Proposition 3.15 / Eq. (3.29) adds `delta` to the stationary event probability for each comparison. With `F` comparisons, this uses `F*delta` from `alpha_s`; the remaining stationary family budget is `alpha_s - F*delta`, so the radius uses `L=log(2F/(alpha_s-F*delta))`. If `F*delta >= alpha_s`, the helper fails closed. For the operational finite-plan target, TV contraction and the score range `[-1,1]` give a conservative `2*delta` mean shift; the LCB subtracts it too. The contract therefore requires an external receipt for the post-burn-in TV bound, rather than treating burn-in as automatically sufficient.

![Initial-law TV sensitivity](initial_law_tv_sensitivity.png)

The companion table `initial_law_tv_sensitivity.csv` uses `F=2`, `alpha_s=0.05`, `gamma_ps_lower=1`, `N={list(INITIAL_TV_BLOCKS)}`, and per-comparison `delta={list(INITIAL_TV_GRID)}`. It reports both the familywise probability cost and the `2*delta` mean shift. Its planned/valid/failed/abstained scenario denominator is `{summary['initial_law_tv_sensitivity_denominators']['planned']}/{summary['initial_law_tv_sensitivity_denominators']['valid']}/{summary['initial_law_tv_sensitivity_denominators']['failed']}/{summary['initial_law_tv_sensitivity_denominators']['abstained']}`. Valid means only that a zero-threshold LCB is reachable if every score is one; these are what-if cells, not measured starts or TV estimates.

## Figure and denominators

![Pseudo-spectral-gap sensitivity under bounded worst-case variance](markov_gap_sensitivity.png)

The left panel spans `N={list(BLOCKS_GRID)}` and assumed `gamma_ps` lower bounds `{list(GAP_GRID)}`; cyan marks the optimistic zero-threshold boundary `epsilon_ps=1`. The right panel overlays graph-color Hoeffding penalties for `chi={1,2,4}` as conditional context under different assumptions.

The stationary-grid denominator is planned/valid/failed/abstained = `{summary['scenario_denominators']['planned']}/{summary['scenario_denominators']['valid']}/{summary['scenario_denominators']['failed']}/{summary['scenario_denominators']['abstained']}`. “Valid” means only that the hypothetical stationary LCB could exceed zero at a sample mean of one. Physical block-score outcomes are `0/0/0/0`; no physical scores were sampled. At `N=20,gamma_ps=1`, the stationary radius remains `{summary['baseline_at_N20_gamma1']:.3f}`. At `N=500,gamma_ps=1`, it is `{summary['best_case_at_N500_gamma1']:.3f}`. This is bound conservatism, not an estimate of the project's mixing rate.

## Synthetic helper integration

`markov_lcb_demo.json` exercises three fixed fixtures: a partially failed stationary panel, an all-failed abstention, and a partially failed panel with `burn_in_blocks=8` and illustrative post-burn-in `delta=0.005`. The comparison denominator is planned/valid/failed/abstained=`{demo['comparison_denominators']['planned']}/{demo['comparison_denominators']['valid']}/{demo['comparison_denominators']['failed']}/{demo['comparison_denominators']['abstained']}`; planned block-slot denominator is `{demo['block_denominators']['planned']}/{demo['block_denominators']['valid']}/{demo['block_denominators']['failed']}/{demo['block_denominators']['abstained']}`. Failed slots stay in `N` and are scored `-1`; the all-failed comparison returns no LCB. These are deterministic software fixtures, not sampled observations or coverage tests. All inputs are illustrative; the helper sets `certified=false` and `physical_score=null` and cannot verify any receipt.

## Text flow

```text
freeze block map, score/failure map, and candidate family
  -> establish the corrected stationary block-chain conditions, fixed score map, and gamma_ps lower bound
  -> justify delta after the declared burn-in, or establish delta=0 stationarity
  -> spend F*delta, solve the stationary radius, and subtract the 2*delta mean shift
  -> apply anchor coverage/common-bias correction under a valid conditional or joint bound
  -> compare with delta_star/R^2 and separately pass the physical reference gate
  -> if any input/receipt is absent: ABSTAIN / NOT_READY; physical_score stays null
```

The implementation and synthetic report close only the software/theorem-mapping boundary. They do not supply a block-chain pseudo-gap receipt, a post-burn-in TV bound, score-anchor dependence evidence, anchor coverage/common-bias evidence, or physical reference. Pion remains `NOT_READY/UNCERTIFIED`, with sealed data closed.

## Rebuild

```text
python -B -m tasks.pion_projection.bounded_block_risk_markov_gap --output NEW_DIRECTORY
```
"""
    (out / "README.md").write_text(readme)

    manifest = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "artifact_hashes.json":
            manifest[path.name] = _sha256(path)
    (out / "artifact_hashes.json").write_text(
        json.dumps({"algorithm": "sha256", "files": manifest}, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = build_report(args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
