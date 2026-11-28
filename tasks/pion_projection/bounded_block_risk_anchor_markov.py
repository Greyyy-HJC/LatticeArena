"""Conditional Markov-chain coverage bounds for a scalar anchor summary.

This module applies Paulin's corrected non-reversible Bernstein inequality to
a *fixed bounded scalar emitted by each anchor-chain state*. It does not turn
a nonlinear fit to an ensemble-averaged correlator into such a scalar; that
mapping, its target correspondence, and its bias bound require independent
receipts. No measurements are read and every result remains uncertified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

from core.receipts import run_receipt, scenario_spec
from .bounded_block_risk_markov_gap import markov_gap_radius


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / (
    "results/pion_projection/bounded_block_risk_anchor_markov_20261125_v1"
)
SUPPORT = (0.1, 3.0)
WIDTH = SUPPORT[1] - SUPPORT[0]
ALPHA_ANCHOR = 0.01
COMMON_BIAS_OVER_R = 0.01
SCORE_FAMILY = 3
ALPHA_SCORE = 0.04
SCORE_BLOCKS = 4000
SCORE_GAMMA = 0.20
SCORE_TV = 0.001
ANCHOR_COUNTS = (1000, 2000, 4000, 8000, 16000, 32000, 64000)
ANCHOR_GAMMAS = (0.05, 0.1, 0.2, 0.5, 1.0)
ANCHOR_TVS = (0.0, 0.001, 0.005, 0.009, 0.01)


def _finite(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def markov_anchor_bound(*, sample_count: int, gamma_ps_lower: float,
                        alpha_anchor: float, support: tuple[float, float],
                        common_bias_bound: float,
                        initial_law_tv_bound: float = 0.0,
                        burn_in_units: int = 0,
                        stationary_start: bool = True,
                        score_scale_R: float | None = None) -> dict[str, Any]:
    """Compute a conditional interval radius for one shared anchor estimate.

    The chain state must emit one predeclared scalar in ``support``. The
    stationary Bernstein tail uses family=1 because this interval is shared by
    all score comparisons. For a post-burn-in start, Proposition 3.15 / Eq.
    (3.29) spends ``delta_TV`` from the anchor error probability; it does not
    add a score-chain mean-shift penalty to this coverage interval.
    """
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
    if (not _finite(gamma_ps_lower) or not 0 < gamma_ps_lower <= 1):
        raise ValueError("gamma_ps_lower must be finite and in (0, 1]")
    if not _finite(alpha_anchor) or not 0 < alpha_anchor < 1:
        raise ValueError("alpha_anchor must be finite and in (0, 1)")
    if (not isinstance(support, tuple) or len(support) != 2
            or not all(_finite(x) for x in support) or support[0] >= support[1]):
        raise ValueError("support must be a finite increasing pair")
    if not _finite(common_bias_bound) or common_bias_bound < 0:
        raise ValueError("common_bias_bound must be finite and nonnegative")
    if (not _finite(initial_law_tv_bound)
            or not 0 <= initial_law_tv_bound < alpha_anchor):
        raise ValueError("initial_law_tv_bound must be in [0, alpha_anchor)")
    if (isinstance(burn_in_units, bool) or not isinstance(burn_in_units, int)
            or burn_in_units < 0):
        raise ValueError("burn_in_units must be a nonnegative integer")
    if stationary_start and (burn_in_units != 0 or initial_law_tv_bound != 0):
        raise ValueError("stationary start requires zero burn-in and zero TV bound")
    if not stationary_start and burn_in_units < 1:
        raise ValueError("post-burn-in start requires a positive burn-in count")
    if score_scale_R is None:
        score_scale_R = support[1] - support[0]
    if not _finite(score_scale_R) or score_scale_R <= 0:
        raise ValueError("score_scale_R must be finite and positive")

    lower, upper = map(float, support)
    width = upper - lower
    delta = float(initial_law_tv_bound)
    alpha_stationary = float(alpha_anchor) - delta
    raw_radius = markov_gap_radius(
        blocks=sample_count,
        gamma_ps_lower=float(gamma_ps_lower),
        family=1,
        alpha_s=alpha_stationary,
        variance_bound=width * width / 4.0,
        centered_abs_bound=width,
    )
    # Both the sample mean and stationary mean lie in the declared support.
    random_radius = min(width, raw_radius)
    total_error = random_radius + float(common_bias_bound)
    return {
        "status": "CONDITIONAL_MARKOV_ANCHOR_BOUND_ONLY",
        "sample_count_n_A": sample_count,
        "gamma_ps_lower_assumed": float(gamma_ps_lower),
        "alpha_anchor": float(alpha_anchor),
        "anchor_tv_probability_cost": delta,
        "stationary_alpha_after_tv": alpha_stationary,
        "burn_in_units": burn_in_units,
        "support_interval": [lower, upper],
        "random_coverage_radius": random_radius,
        "raw_bernstein_radius_before_support_cap": raw_radius,
        "common_bias_bound_C": float(common_bias_bound),
        "total_error_bound_B": total_error,
        "score_scale_R": float(score_scale_R),
        "normalized_score_shift_2B_over_R": 2.0 * total_error / float(score_scale_R),
        "coverage_probability_lower_bound": 1.0 - float(alpha_anchor),
        "shared_across_comparison_family": True,
        "family_for_anchor_interval": 1,
        "certified": False,
        "physical_score": None,
        "data_access": "none",
        "required_assumptions": [
            "one fixed measurable scalar function maps each anchor-chain state into the declared support",
            "the anchor state sequence is a time-homogeneous Markov chain satisfying Paulin corrected Theorem 3.11 assumptions",
            "gamma_ps_lower is a justified lower bound for this anchor chain, not the separate score block chain",
            "stationary start or the post-burn-in path law is within initial_law_tv_bound of stationarity",
            "the scalar anchor mean matches the stated m_star target up to common_bias_bound_C",
            "the single anchor interval is shared by all three comparisons and is conditionally independent of score-chain outcomes",
        ],
    }


def _scenario_rows() -> list[dict[str, Any]]:
    score_radius = markov_gap_radius(
        blocks=SCORE_BLOCKS,
        gamma_ps_lower=SCORE_GAMMA,
        family=SCORE_FAMILY,
        alpha_s=ALPHA_SCORE - SCORE_FAMILY * SCORE_TV,
    )
    rows = []
    for n in ANCHOR_COUNTS:
        for gamma in ANCHOR_GAMMAS:
            for tv in ANCHOR_TVS:
                planned = 1
                valid = int(tv < ALPHA_ANCHOR)
                failed = 0
                abstained = 1 - valid
                radius = total = penalty = best_lcb = None
                reach = False
                if valid:
                    result = markov_anchor_bound(
                        sample_count=n,
                        gamma_ps_lower=gamma,
                        alpha_anchor=ALPHA_ANCHOR,
                        support=SUPPORT,
                        common_bias_bound=COMMON_BIAS_OVER_R * WIDTH,
                        initial_law_tv_bound=tv,
                        burn_in_units=8 if tv > 0 else 0,
                        stationary_start=(tv == 0),
                        score_scale_R=WIDTH,
                    )
                    radius = result["random_coverage_radius"] / WIDTH
                    total = result["total_error_bound_B"] / WIDTH
                    penalty = result["normalized_score_shift_2B_over_R"]
                    best_lcb = 1.0 - score_radius - 2.0 * SCORE_TV - penalty
                    reach = best_lcb > 0.05
                rows.append({
                    "planned_anchor_units": n,
                    "gamma_ps_anchor_assumed": gamma,
                    "anchor_tv_bound_assumed": tv,
                    "alpha_anchor": ALPHA_ANCHOR,
                    "stationary_alpha": ALPHA_ANCHOR - tv if valid else None,
                    "random_radius_over_R": radius,
                    "total_B_over_R": total,
                    "score_anchor_penalty_2B_over_R": penalty,
                    "fixed_score_plan_radius_assumed": score_radius,
                    "best_case_lcb_assuming_score_mean_1": best_lcb,
                    "best_case_exceeds_0_05": reach,
                    "planned": planned,
                    "valid": valid,
                    "failed": failed,
                    "abstained": abstained,
                    "status": "FORMULA_SCENARIO_ONLY" if valid else "ABSTAIN_TV_SPENDS_ALPHA_ANCHOR",
                })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _build_figure(out: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    selected = [row for row in rows if row["anchor_tv_bound_assumed"] == 0.001]
    matrix = np.full((len(ANCHOR_GAMMAS), len(ANCHOR_COUNTS)), np.nan)
    for row in selected:
        i = ANCHOR_GAMMAS.index(row["gamma_ps_anchor_assumed"])
        j = ANCHOR_COUNTS.index(row["planned_anchor_units"])
        matrix[i, j] = row["best_case_lcb_assuming_score_mean_1"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    mesh = axes[0].imshow(matrix, aspect="auto", origin="lower", cmap="viridis",
                          extent=(-0.5, len(ANCHOR_COUNTS)-0.5,
                                  -0.5, len(ANCHOR_GAMMAS)-0.5))
    axes[0].set_xticks(range(len(ANCHOR_COUNTS)), labels=[str(n) for n in ANCHOR_COUNTS], rotation=35)
    axes[0].set_yticks(range(len(ANCHOR_GAMMAS)), labels=[str(g) for g in ANCHOR_GAMMAS])
    axes[0].set_xlabel("anchor Markov states n_A")
    axes[0].set_ylabel("assumed anchor pseudo-gap lower bound")
    axes[0].set_title("Best-case score LCB with anchor Markov bound\nassuming anchor TV = 0.001")
    fig.colorbar(mesh, ax=axes[0], label="LCB after score + anchor penalties")

    for tv in (0.0, 0.001, 0.005, 0.009):
        curve = [row for row in rows
                 if row["gamma_ps_anchor_assumed"] == 0.2
                 and row["anchor_tv_bound_assumed"] == tv]
        axes[1].plot(
            [row["planned_anchor_units"] for row in curve],
            [row["score_anchor_penalty_2B_over_R"] for row in curve],
            marker="o", label=f"TV={tv:g}",
        )
    iid_n1000 = math.sqrt(math.log(2.0 / ALPHA_ANCHOR) / (2 * 1000)) + COMMON_BIAS_OVER_R
    axes[1].axhline(2 * iid_n1000, linestyle="--", color="black",
                    label="IID Hoeffding penalty, n_A=1000")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(ANCHOR_COUNTS, labels=[str(n) for n in ANCHOR_COUNTS], rotation=35)
    axes[1].set_xlabel("anchor Markov states n_A")
    axes[1].set_ylabel("score penalty 2B/R")
    axes[1].set_title("Score-anchor penalty vs anchor-chain length")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Conditional planning arithmetic only; no anchor or score measurements", fontsize=11)
    fig.savefig(out / "anchor_markov_sensitivity.png", dpi=180)
    fig.savefig(out / "anchor_markov_sensitivity.svg")
    plt.close(fig)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(output: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows = _scenario_rows()
    _write_csv(out / "anchor_markov_sensitivity.csv", rows)
    _build_figure(out, rows)
    totals = {key: sum(row[key] for row in rows) for key in ("planned", "valid", "failed", "abstained")}
    selected = next(row for row in rows if row["planned_anchor_units"] == 4000
                    and row["gamma_ps_anchor_assumed"] == 0.2
                    and row["anchor_tv_bound_assumed"] == 0.001)
    protocol = {
        "schema": "pion-markov-anchor-coverage-plan-v1",
        "status": "CONDITIONAL_ANALYTIC_ONLY_NOT_READY",
        "physical_score": None,
        "certified": False,
        "data_access": "none",
        "anchor_method": "markov_pseudo_gap_bernstein",
        "anchor_family": 1,
        "anchor_is_shared_across_score_comparisons": True,
        "support_interval": list(SUPPORT),
        "support_width_R": WIDTH,
        "planned_anchor_units_candidate": 4000,
        "gamma_ps_anchor_lower_bound_candidate": 0.2,
        "burn_in_anchor_units_candidate": 8,
        "post_burn_in_tv_candidate": 0.001,
        "alpha_anchor": ALPHA_ANCHOR,
        "common_bias_C_over_R_candidate": COMMON_BIAS_OVER_R,
        "score_chain_plan": {
            "family": SCORE_FAMILY,
            "alpha_score": ALPHA_SCORE,
            "blocks": SCORE_BLOCKS,
            "gamma_ps_assumed": SCORE_GAMMA,
            "post_burn_in_tv_assumed": SCORE_TV,
        },
        "candidate_point": selected,
        "required_receipts": [
            "bounded scalar map from each anchor-chain state to a_t m_pi units",
            "proof/review that the stationary scalar expectation targets the selected m_star up to C",
            "anchor-chain homogeneous Markov and state-space assumptions",
            "anchor-chain gamma_ps lower-bound receipt (separate from score chain)",
            "anchor post-burn-in TV receipt, if nonstationary start is used",
            "support, common-bias C, and conditional-independence receipts",
        ],
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema": "pion-markov-anchor-coverage-summary-v1",
        "status": "CONDITIONAL_ANALYTIC_ONLY_NOT_READY",
        "physical_score": None,
        "certified": False,
        "data_access": "none",
        "scenario_denominators": totals,
        "physical_anchor_units": 0,
        "physical_score_units": 0,
        "candidate_point": selected,
        "iid_hoeffding_selected_plan_n1000_score_penalty": 2 * (
            math.sqrt(math.log(2.0 / ALPHA_ANCHOR) / (2 * 1000)) + COMMON_BIAS_OVER_R
        ),
        "theorem": "Paulin corrected arXiv v5 Theorem 3.11 / Eq. (3.25), with Proposition 3.15 / Eq. (3.29) initial-law TV probability cost",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    rows_for_receipt = [{
        "scenario": f"n{r['planned_anchor_units']}_g{r['gamma_ps_anchor_assumed']}_tv{r['anchor_tv_bound_assumed']}",
        "unit": "one analytic assumption combination",
        "planned": r["planned"], "valid": r["valid"],
        "failed": r["failed"], "abstained": r["abstained"],
        "reason": r["status"],
    } for r in rows]
    _write_csv(out / "denominator_summary.csv", rows_for_receipt)
    flow = (
        "predeclared bounded scalar per independent anchor-chain state -> require its target map, "
        "separate anchor gamma_ps and post-burn-in TV receipts -> spend delta_TV from alpha_anchor "
        "for one shared anchor interval -> corrected Bernstein radius + common-bias C -> "
        "subtract 2B/R from the score LCB -> missing map/receipt means NOT_READY; physical_score=null"
    )
    (out / "README.md").write_text("\n".join([
        "# Markov-aware anchor coverage: conditional planning diagnostic",
        "",
        "This report gives the selected conditional Markov-anchor planning route. Its formula scenarios do not establish a valid state-to-scalar map, support, chain gap, TV bound, common-bias bridge, or any physical receipt; the anchor evaluator must abstain until those inputs are independently reviewed.",
        "",
        "## Bound and scope",
        "",
        "For a stationary anchor chain and one fixed scalar map `f(X_i) in [L,U]`, use Paulin corrected arXiv v5 Theorem 3.11 / Eq. (3.25), with worst-case `Var(f)<= (U-L)^2/4` and centered bound `C=U-L`. Solve the two-sided tail bound for the sample-mean radius. If the scored chain starts within TV `delta_A` after burn-in, Proposition 3.15 / Eq. (3.29) adds `delta_A` to the event probability; therefore the stationary theorem receives `alpha_A-delta_A`. The anchor interval is shared once across the three comparisons, so its interval uses family size one. Total anchor error is `B = random_radius + common_bias_C`, and its paired squared-error score penalty is `2B/R`.",
        "",
        "The scalar map requirement is substantive. A global multi-channel fit to ensemble-averaged PP/A4P correlators is not automatically a per-state function `f(X_i)`. It needs a separately reviewed bounded state map and a target/bias argument; otherwise this theorem route cannot be applied to that fit.",
        "",
        "## Selected planning point (assumptions only)",
        "",
        f"At `n_A=4000`, assumed anchor `gamma_ps>=0.20`, assumed post-burn-in `TV<=0.001`, `alpha_anchor=0.01`, support `[0.1,3.0]`, and `C/R=0.01`, the formula gives `random radius/R={selected['random_radius_over_R']:.6f}`, `B/R={selected['total_B_over_R']:.6f}`, and `2B/R={selected['score_anchor_penalty_2B_over_R']:.6f}`. With the separate score plan's assumed `N=4000`, `gamma_ps=0.20`, and `TV=0.001`, even a score mean of one yields best-case LCB `{selected['best_case_lcb_assuming_score_mean_1']:.6f}`. These are assumptions and arithmetic only; neither chain has these receipts.",
        "",
        f"For comparison, the selected IID-Hoeffding `n_A=1000` route has planning penalty `2B/R={summary['iid_hoeffding_selected_plan_n1000_score_penalty']:.6f}` under its much stronger IID assumption. The Markov route does not claim to be narrower; it provides a conditional option when dependence is explicit.",
        "",
        "![Markov anchor coverage sensitivity](anchor_markov_sensitivity.png)",
        "",
        "## Text flow",
        "",
        "```text",
        flow,
        "```",
        "",
        f"Analytic assumption cells planned/valid/failed/abstained = {totals['planned']}/{totals['valid']}/{totals['failed']}/{totals['abstained']}. This is not a Monte Carlo count or measured sample denominator. Physical anchor and score denominators are 0/0.",
        "",
        "## Required before use",
        "",
        "- Define a fixed bounded scalar emitted by each anchor-chain state and explain how its stationary mean estimates the selected `m_star`.",
        "- Obtain a separate anchor-chain receipt, a valid `gamma_ps` lower bound, and a post-burn-in TV receipt (or prove stationarity). Do not reuse score-chain receipts.",
        "- Bound common bias `C`, establish independence from the score chain, and have the target mapping independently reviewed.",
        "- Keep the global candidate-blind multi-channel fit as the physical reference gate; a scalar-map route needs a quantitative bridge to it.",
        "- No sealed input, confirmation split, GPU/QUDA data, or measurement was accessed; status remains `NOT_READY/UNCERTIFIED`, `certified=false`, `physical_score=null`.",
        "",
        "## Source",
        "",
        "Paulin, *Concentration inequalities for Markov chains by Marton couplings and spectral methods*, corrected arXiv v5, Theorem 3.11 Eq. (3.25), Proposition 3.15 Eq. (3.29): https://arxiv.org/html/1212.2015v5",
        "",
    ]))
    source_snapshot = out / "snapshot_bounded_block_risk_anchor_markov.py"
    shutil.copy2(Path(__file__), source_snapshot)
    payloads = {p.relative_to(out).as_posix(): _sha256(p)
                for p in sorted(out.rglob("*")) if p.is_file()}
    denominators = {
        row["scenario"]: {
            "unit": row["unit"],
            "planned": row["planned"],
            "valid": row["valid"],
            "failed": row["failed"],
            "abstained": row["abstained"],
        }
        for row in rows_for_receipt
    }
    reasons = [{
        "unit": row["scenario"],
        "state": "abstained",
        "count": 1,
        "reason": "assumed anchor TV is not smaller than alpha_anchor",
    } for row in rows_for_receipt if row["abstained"]]
    receipt = run_receipt(
        scenario_spec(
            "pion-markov-anchor-analytic-sensitivity",
            "anchor count, anchor pseudo-gap assumption, and post-burn-in TV assumption",
            "Conditional formula grid for Markov-dependent scalar anchor estimates.",
            assumptions=[
                "each anchor chain state emits one fixed bounded scalar in the declared support",
                "the anchor chain satisfies corrected Markov pseudo-gap Bernstein assumptions",
                "the anchor scalar expectation maps to m_star up to the supplied common bias bound",
                "one shared anchor interval is used across the score comparison family",
            ],
            source_hashes={
                "anchor_helper": _sha256(Path(__file__)),
                "markov_gap_helper": _sha256(Path(__file__).with_name("bounded_block_risk_markov_gap.py")),
            },
        ),
        denominators,
        artifact_hashes=payloads,
        text_flow=flow,
        reason_rows=reasons,
    )
    (out / "run_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    hashes = {p.relative_to(out).as_posix(): _sha256(p)
              for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    summary = build_report(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
