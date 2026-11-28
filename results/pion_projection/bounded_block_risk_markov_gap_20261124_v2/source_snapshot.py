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
DEFAULT_OUT = ROOT / "results/pion_projection/bounded_block_risk_markov_gap_20261124_v2"
BLOCKS_GRID = (20, 50, 100, 200, 500)
GAP_GRID = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
FAMILY = 2
ALPHA_S = 0.05
VARIANCE_BOUND = 1.0
CENTERED_ABS_BOUND = 2.0


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

    Solves Paulin (2015), Theorem 3.10's non-reversible Bernstein exponent
    using the worst-case bounds Var(Z)<=1 and |Z-EZ|<=2 for Z in [-1, 1].
    ``gamma_ps_lower`` must be a justified lower bound for the Markov chain
    whose states are the predeclared block observations.
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
) -> dict:
    """Compute a conditional Markov-chain LCB over the full planned block set.

    This is a math helper only. A valid lower bound on the *block-chain* pseudo
    spectral gap and every theorem condition must be supplied by the caller.
    Failed blocks stay in the planned denominator and receive score -1. An
    all-failed comparison abstains. This function never emits a physical score.
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

    planned = int(differences.size)
    valid_count = int(valid.sum())
    failed_count = planned - valid_count
    radius = markov_gap_radius(
        blocks=planned,
        gamma_ps_lower=gamma_ps_lower,
        family=family,
        alpha_s=alpha_score,
    )
    planned_mean = float(scores.mean())
    anchor_penalty = 2.0 * float(relative_anchor_bound_B_over_R)
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
            "the predeclared block process is a stationary homogeneous Markov chain satisfying the theorem conditions",
            "gamma_ps_lower is a valid auditable lower bound for that block chain",
            "the theorem's range/variance bounds apply to all planned coded scores in [-1, 1]",
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
            "normalized_lower_bound": None,
        }

    valid_mean = float(scores[valid].mean())
    lower_bound = planned_mean - radius - anchor_penalty
    return {
        **common,
        "status": "CONDITIONAL_MARKOV_PSEUDOGAP_BOUND_ONLY",
        "valid_block_mean": valid_mean,
        "concentration_penalty": radius,
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


def _synthetic_lcb_demo() -> dict:
    """Exercise the Markov LCB integration on two explicitly synthetic panels."""
    import numpy as np

    calibration = {
        "score_scale_R": 2.0,
        "gamma_ps_lower": 1.0,
        "family": 2,
        "alpha_score": 0.04,
        "alpha_anchor": 0.01,
        "alpha_total": 0.05,
        "relative_anchor_bound_B_over_R": 0.03,
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
        ],
        "comparison_denominators": {
            "planned": 2,
            "valid": 1,
            "failed": 0,
            "abstained": 1,
        },
        "block_denominators": {
            "planned": 6,
            "valid": 3,
            "failed": 3,
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
    demo = _synthetic_lcb_demo()
    feasible = sum(row["status"] == "CONDITIONALLY_FEASIBLE" for row in rows)
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
            "correction, no extra nonstationarity correction, a known valid lower "
            "bound on the block-chain pseudo-spectral gap, and all theorem conditions."
        ),
    }
    protocol = {
        "classification": summary["classification"],
        "inputs_read": [],
        "data_access": "none; no physical or sealed inputs were opened",
        "formula_source": {
            "paper": "Paulin, Concentration inequalities for Markov chains by Marton couplings and spectral methods (2015), Theorem 3.10",
            "url": "https://arxiv.org/abs/1212.2015",
            "bound": "P(|S-E S|>=t) <= 2 exp(-t^2 gamma_ps / (8(N+1/gamma_ps)V_f + 20 t C))",
        },
        "block_chain_contract": [
            "Predeclare non-overlapping score blocks and represent each block as one state of a homogeneous block Markov chain.",
            "The block chain itself must be stationary, phi-irreducible, aperiodic, Polish, and have a unique stationary law as required by the paper.",
            "Use an auditable lower confidence bound gamma_ps_lower > 0 for this block chain; trajectory spacing and empirical ESS are not such a bound.",
            "For Z_i in [-1,1], use C=2 and V_f<=1; failures remain mapped to -1 and stay in planned N.",
            "The score-family union bound uses log(2F/alpha_s), because the cited theorem is two-sided.",
            "The score bound is conditional on a fixed anchor; if the anchor shares trajectory dependence, include it in a proved joint process or use a valid joint bound.",
            "Anchor coverage/bias, target threshold, and nonstationary-start correction remain separate; none is included in this sensitivity radius.",
            "A separate synthetic integration fixture exercises the implemented LCB helper and failure denominators; it is not project data or a coverage test.",
        ],
        "grid": {
            "planned_blocks": list(BLOCKS_GRID),
            "gamma_ps_lower_assumed": list(GAP_GRID),
            "family": FAMILY,
            "alpha_s": ALPHA_S,
            "variance_bound": VARIANCE_BOUND,
            "centered_abs_bound": CENTERED_ABS_BOUND,
            "target_threshold": 0.0,
            "anchor_correction": 0.0,
            "failure_fraction": 0.0,
            "warning": "All values are hypothetical, not frozen or estimated project inputs.",
        },
    }
    _write_csv(out / "sensitivity_grid.csv", rows)
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
            "interpretation": "Two fixed software fixtures: one conditional helper result and one all-failed abstention; no random experiment or physical sample.",
        },
        {
            "unit": "synthetic Markov-LCB planned block slots",
            **demo["block_denominators"],
            "interpretation": "The two software fixtures contain six hypothetical slots; three valid and three explicitly failed slots, with no block removed.",
        },
    ]
    _write_csv(out / "denominator_summary.csv", scenario_denominators)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "markov_lcb_demo.json").write_text(json.dumps(demo, indent=2) + "\n")
    _write_plot(out, rows)
    shutil.copy2(Path(__file__), out / "source_snapshot.py")

    readme = f"""# Pion bounded-risk Markov gap sensitivity

**SYNTHETIC ANALYTIC SENSITIVITY ONLY. `physical_score=null`.** The script evaluates a published Markov-chain concentration inequality on hypothetical block counts and pseudo-spectral-gap lower bounds. It samples no scores and reads no ensemble, anchor, gap, validation, holdout, GPU, or QUDA input.

## Conditional route

If each planned score `Z_i` is a fixed bounded function of a predeclared, non-overlapping block state and the resulting block process is a stationary, homogeneous, `phi`-irreducible, aperiodic Markov chain on a Polish space with unique stationary law, Paulin's Theorem 3.10 gives, for `S=sum_i Z_i`,

```text
P(|S-E S| >= t) <= 2 exp[-t^2 gamma_ps / (8 (N + 1/gamma_ps) V_f + 20 t C)]
```

For `Z_i in [-1,1]`, the worst-case choices are `C=2` and `V_f<=1`. With a simultaneous family of `F` comparisons and score error budget `alpha_s`, set `L=log(2F/alpha_s)`. Solving the exponent for `t` gives the normalized score radius

```text
epsilon_ps = [20 C L + sqrt((20 C L)^2 + 32 gamma_ps L (N+1/gamma_ps) V_f)] / (2 gamma_ps N).
```

The grid uses `V_f=1`, `F=2`, `alpha_s=0.05`, no failures, zero target, and zero anchor correction. These inputs deliberately give the most favorable possible decision threshold while retaining the theorem's bounded-range term. They are not frozen protocol values. For certification, `gamma_ps` must be replaced with an auditable positive lower bound for the **block chain itself**. If initialization is not stationary, a separately justified burn-in or initial-law correction is required. The score bound is conditional on a fixed anchor; if the anchor shares trajectory dependence, it must be included in a proved joint process or handled with a valid joint bound. Shared-anchor coverage and the `2B/R` bias term remain separate from this score penalty. All failures continue to map to `-1` and remain in planned `N`; the existing all-failed panel policy still abstains.

The source states that Theorem 3.10 assumes a stationary Markov chain and defines the non-reversible pseudo-spectral gap; the paper's Section 3 also states the chain regularity conditions used here. See [Paulin (2015), Theorem 3.10](https://arxiv.org/abs/1212.2015).

## Figure and denominators

![Pseudo-spectral-gap sensitivity under bounded worst-case variance](markov_gap_sensitivity.png)

The left panel spans `N={list(BLOCKS_GRID)}` and assumed `gamma_ps` lower bounds `{list(GAP_GRID)}`; cyan marks the optimistic zero-threshold attainability boundary `epsilon_ps=1`. The right panel overlays the graph-color Hoeffding formula for `chi={1,2,4}` as conditional context. They require different dependence assumptions and are not an empirical head-to-head test.

The scenario denominator is planned/valid/failed/abstained = `{summary['scenario_denominators']['planned']}/{summary['scenario_denominators']['valid']}/{summary['scenario_denominators']['failed']}/{summary['scenario_denominators']['abstained']}`. “Valid” means only that the hypothetical LCB could exceed zero if every score were at its maximum value of one; it does not indicate data availability. Physical block-score outcomes are `0/0/0/0`, because none were sampled. The scenario-weighted planning grid contains `{summary['scenario_weighted_hypothetical_planned_blocks']}` hypothetical block slots.

At the candidate planning count `N=20`, even the hypothetical `gamma_ps=1` case has radius `{summary['baseline_at_N20_gamma1']:.3f}`, so no zero-threshold certificate is attainable with this worst-case version. At `N=500, gamma_ps=1`, the radius is `{summary['best_case_at_N500_gamma1']:.3f}`; only `{summary['feasible_scenarios']}/{len(rows)}` grid cases are mathematically reachable. This quantifies conservatism under the declared theorem/variance bound, not the project's actual mixing rate or future utility.

## Synthetic helper integration

`markov_lcb_demo.json` exercises the implemented `conditional_block_risk_markov_lcb` with a fixed four-block example and an all-failed example. The first maps normalized scores to `[1, -1, 0.5, -1]`, retains the failed fourth block in `N=4`, and returns only `CONDITIONAL_MARKOV_PSEUDOGAP_BOUND_ONLY`; the all-failed case returns `ABSTAIN_NO_VALID_BLOCKS` and no lower bound. Across these two fixed software fixtures, comparison denominators are planned/valid/failed/abstained=`{demo['comparison_denominators']['planned']}/{demo['comparison_denominators']['valid']}/{demo['comparison_denominators']['failed']}/{demo['comparison_denominators']['abstained']}`; block-slot denominators are `{demo['block_denominators']['planned']}/{demo['block_denominators']['valid']}/{demo['block_denominators']['failed']}/{demo['block_denominators']['abstained']}`. These counts are deterministic fixture cases, not sampled observations. Alpha allocation, the assumed block-chain gap, and `B/R` are illustrative only; the helper marks `certified=false`, leaves `physical_score=null`, and cannot verify any theorem receipt.

## Text flow

```text
freeze non-overlapping score blocks and failure map
  -> establish stationary Markov assumptions for the block process
  -> obtain a reviewed lower bound on block-chain gamma_ps
  -> compute simultaneous epsilon_ps from Theorem 3.10
  -> subtract separate anchor-bias correction and compare with delta_star/R^2
  -> also pass the independent physical reference gate
  -> if any input/bound is absent: ABSTAIN / NOT_READY; physical_score stays null
```

This route supplies explicit constants for the TODO's alternative-theorem option. It does not supply the required block-chain pseudo-gap receipt or anchor/reference evidence. The Pion task therefore remains `NOT_READY/UNCERTIFIED`, with sealed data closed.

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
