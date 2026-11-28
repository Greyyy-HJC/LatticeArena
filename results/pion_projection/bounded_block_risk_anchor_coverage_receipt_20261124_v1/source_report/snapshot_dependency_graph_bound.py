"""Analytic dependence sensitivity for bounded block-MSE planning.

This module has no access to physics data. The graph-coloring result is
conditional on a valid dependency graph whose color classes are jointly
independent (also conditional on any shared anchor and frozen protocol).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/pion_projection/bounded_block_risk_calibration_20261123_v2"
BLOCK_COUNTS = (20, 100, 400)
FAMILY = 2
ALPHA_S = 0.05
ALPHA_TOTAL = 0.05


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def dependence_penalty(*, blocks: int, chromatic_number: int,
                       family: int = FAMILY, alpha_s: float = ALPHA_S) -> float:
    """Return the normalized one-sided Hoeffding penalty.

    For normalized scores Z_i in [-1, 1], a dependency graph with a proper
    coloring into chi jointly independent color classes gives
    sqrt(2*chi*log(F/alpha_s)/N). A proven upper bound on the chromatic number
    may be used; it must not be inferred from trajectory spacing alone.
    """
    n = _positive_integer(blocks, "blocks")
    chi = _positive_integer(chromatic_number, "chromatic_number")
    f = _positive_integer(family, "family")
    if chi > n:
        raise ValueError("chromatic_number cannot exceed the number of blocks")
    if not math.isfinite(alpha_s) or not 0 < alpha_s < 1:
        raise ValueError("alpha_s must be finite and in (0, 1)")
    return math.sqrt(2.0 * chi * math.log(f / alpha_s) / n)


def _validate_alpha_allocation(*, alpha_s: float, alpha_anchor: float,
                               alpha_total: float) -> None:
    if not math.isfinite(alpha_s) or not 0 < alpha_s < 1:
        raise ValueError("alpha_s must be finite and in (0, 1)")
    if not math.isfinite(alpha_anchor) or not 0 <= alpha_anchor < 1:
        raise ValueError("alpha_anchor must be finite and in [0, 1)")
    if not math.isfinite(alpha_total) or not 0 < alpha_total < 1:
        raise ValueError("alpha_total must be finite and in (0, 1)")
    if alpha_s + alpha_anchor > alpha_total:
        raise ValueError("alpha_s + alpha_anchor must not exceed alpha_total")


def normalized_lower_bound(valid_mean: float, *, blocks: int,
                           chromatic_number: int, family: int,
                           alpha_s: float, alpha_anchor: float,
                           alpha_total: float,
                           relative_anchor_bound: float = 0.0,
                           failure_fraction: float = 0.0) -> float:
    """Return a lower bound for the bounded block-risk estimand.

    ``valid_mean`` is the observed mean over valid planned blocks; failures are
    coded as -1, so the planned-sample mean is
    ``(1-failure_fraction)*valid_mean - failure_fraction``. The dependency
    graph and concentration bound apply to all planned blocks, including coded
    failures. ``relative_anchor_bound`` is B/R and its factor 2 is valid for a
    squared-error contrast only when ``|y_candidate-y_reference| <= R``.
    """
    if not math.isfinite(valid_mean) or not -1 <= valid_mean <= 1:
        raise ValueError("valid_mean must be finite and in [-1, 1]")
    if (not math.isfinite(relative_anchor_bound)
            or relative_anchor_bound < 0):
        raise ValueError("relative_anchor_bound must be finite and nonnegative")
    if not math.isfinite(failure_fraction) or not 0 <= failure_fraction < 1:
        raise ValueError("failure_fraction must be finite and in [0, 1)")
    _validate_alpha_allocation(alpha_s=alpha_s, alpha_anchor=alpha_anchor,
                               alpha_total=alpha_total)
    penalty = dependence_penalty(
        blocks=blocks, chromatic_number=chromatic_number,
        family=family, alpha_s=alpha_s,
    )
    coded_mean = (1.0 - failure_fraction) * valid_mean - failure_fraction
    return coded_mean - penalty - 2.0 * relative_anchor_bound


def required_valid_mean(*, blocks: int, chromatic_number: int,
                        family: int, alpha_s: float,
                        alpha_anchor: float, alpha_total: float,
                        relative_anchor_bound: float = 0.0,
                        normalized_threshold: float = 0.0,
                        failure_fraction: float = 0.0) -> float:
    """Return the observed valid-block mean boundary for a threshold.

    The equality point is a planning boundary, not an observed score; a strict
    certification rule requires the observed bound to exceed the threshold.
    Values above 1 are unattainable for scores bounded in [-1, 1].
    """
    if not math.isfinite(normalized_threshold):
        raise ValueError("normalized_threshold must be finite")
    if not math.isfinite(relative_anchor_bound) or relative_anchor_bound < 0:
        raise ValueError("relative_anchor_bound must be finite and nonnegative")
    if not math.isfinite(failure_fraction) or not 0 <= failure_fraction < 1:
        raise ValueError("failure_fraction must be finite and in [0, 1)")
    _validate_alpha_allocation(alpha_s=alpha_s, alpha_anchor=alpha_anchor,
                               alpha_total=alpha_total)
    penalty = dependence_penalty(
        blocks=blocks, chromatic_number=chromatic_number,
        family=family, alpha_s=alpha_s,
    )
    return (normalized_threshold + penalty + 2.0 * relative_anchor_bound
            + failure_fraction) / (1.0 - failure_fraction)


def _write_plot(path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    ax = axes[0]
    n = 20
    chis = list(range(1, n + 1))
    penalties = [dependence_penalty(blocks=n, chromatic_number=chi)
                 for chi in chis]
    ax.plot(chis, penalties, color="#0072B2", lw=2, marker="o", ms=3)
    ax.axhline(1.0, color="#D55E00", ls="--", lw=1.2,
               label="maximum possible normalized improvement = 1")
    ax.set(xlabel="assumed chromatic-number upper bound  chi",
           ylabel="one-sided penalty / R^2",
           title="N=20 planned blocks")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    colors = ("#0072B2", "#009E73", "#CC79A7")
    for n, color in zip(BLOCK_COUNTS, colors):
        chis = list(range(1, n + 1))
        required = [required_valid_mean(
            blocks=n, chromatic_number=chi, family=FAMILY,
            alpha_s=ALPHA_S, alpha_anchor=0.0, alpha_total=ALPHA_TOTAL,
        ) for chi in chis]
        ax.plot(chis, required, color=color, lw=2,
                label=f"N={n} planned blocks")
    ax.axhline(1.0, color="black", ls="--", lw=1.1,
               label="largest possible valid mean")
    ax.set(xlabel="assumed chromatic-number upper bound  chi",
           ylabel="required valid mean / R^2",
           title="Zero failures, zero anchor/threshold penalties")
    ax.set_ylim(0, 3.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.suptitle("Dependency-graph sensitivity: analytic illustration only\n"
                 "F=2, alpha_total=alpha_s=0.05, alpha_A=0; no physical score or dependence estimate",
                 fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build(output: str | Path = DEFAULT_OUT) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for n in BLOCK_COUNTS:
        for chi in range(1, n + 1):
            penalty = dependence_penalty(blocks=n, chromatic_number=chi)
            rows.append({
                "planned_blocks": n,
                "chromatic_number_upper_bound": chi,
                "family": FAMILY,
                "alpha_s": ALPHA_S,
                "alpha_anchor": 0.0,
                "alpha_total": ALPHA_TOTAL,
                "union_bound_alpha": ALPHA_S,
                "normalized_penalty": penalty,
                "effective_block_count_algebraic_only": n / chi,
                "required_valid_mean_over_R2_zero_failure": required_valid_mean(
                    blocks=n, chromatic_number=chi, family=FAMILY,
                    alpha_s=ALPHA_S, alpha_anchor=0.0,
                    alpha_total=ALPHA_TOTAL,
                ),
                "feasible_at_maximum_valid_mean": required_valid_mean(
                    blocks=n, chromatic_number=chi, family=FAMILY,
                    alpha_s=ALPHA_S, alpha_anchor=0.0,
                    alpha_total=ALPHA_TOTAL,
                ) <= 1.0,
                "classification": "analytic sensitivity; not a measured dependence bound",
            })
    with (out / "sensitivity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol = {
        "schema": "pion-bounded-block-risk-calibration-v2",
        "status": "ANALYTIC_SENSITIVITY_ONLY",
        "physical_score": None,
        "data_access": "No correlator, anchor, validation, gap, holdout, GPU, or QUDA input was read.",
        "assumption": "Conditional on a valid dependency graph for the planned normalized block scores, whose color classes are jointly independent (conditional on the frozen shared anchor and protocol).",
        "normalized_score_range": [-1, 1],
        "formula": {
            "penalty": "sqrt(2*chi*log(F/alpha_s)/N)",
            "coded_sample_mean": "zbar=(1-phi_hat)*zbar_valid-phi_hat",
            "lower_bound": "L/R^2=zbar-sqrt(2*chi*log(F/alpha_s)/N)-2*B/R",
            "required_mean": "zbar_valid>(delta_star/R^2+sqrt(2*chi*log(F/alpha_s)/N)+2*B/R+phi_hat)/(1-phi_hat)",
            "anchor_error_condition": "|D(A)-D(m*)|=2*|A-m*|*|y_candidate-y_reference| <= 2*B*R",
            "algebraic_effective_count": "N/chi (notation for this bound only; not an estimated ESS)",
        },
        "illustration_only": {
            "planned_blocks": list(BLOCK_COUNTS),
            "family": FAMILY,
            "alpha_s": ALPHA_S,
            "alpha_anchor": 0.0,
            "alpha_total": ALPHA_TOTAL,
            "relative_anchor_bound": 0.0,
            "normalized_threshold": 0.0,
            "failure_fraction": 0.0,
            "note": "Optimistic sensitivity values only; none is frozen for Pion.",
        },
        "rows": len(rows),
        "source_sha256": source_hash,
        "files": ["README.md", "sensitivity.csv", "dependence_sensitivity.png", "artifact_hashes.json"],
    }
    (out / "summary.json").write_text(json.dumps(protocol, indent=2) + "\n")
    _write_plot(out / "dependence_sensitivity.png")
    readme = [
        "# Bounded block-MSE: dependence sensitivity",
        "",
        "**ANALYTIC SENSITIVITY ONLY. `physical_score=null`.** This CPU-only calculation reads no correlator, E anchor, validation, gap, shared holdout, GPU or QUDA data. It does not estimate the actual Pion score dependence and does not certify independence.",
        "",
        "## Why this check was added",
        "",
        "The receipt audit documents 100 development configurations with trajectory spacing 5, grouped into 20 reproducible length-5 candidate blocks. It does not provide a score-specific autocorrelation or mixing bound. The earlier independent-block Hoeffding calibration is therefore conditional on a block-independence assumption that has not been established from the available receipts.",
        "",
        "## Bound and assumptions",
        "",
        "Let `D_i(A)` be the predeclared paired squared-error contrast on planned block `i`, and define `Z_i=clip(D_i(A)/R^2,-1,1)`. The estimand is the finite-plan mean `theta_N=(1/N)*sum_i E[Z_i | frozen protocol, anchor]`; stationarity may simplify it to one common block mean but is not needed for the bounded concentration inequality. `N` is the planned block count, `F` the fixed candidate family, and `chi` an upper bound on the chromatic number of a dependency graph. If each color class consists of jointly independent scores conditional on the shared anchor and frozen protocol, coloring plus Holder's inequality gives the conservative one-sided penalty",
        "",
        "```text",
        "sqrt(2*chi*log(F/alpha_s)/N)",
        "```",
        "",
        "`alpha_s` is the score-concentration error budget. If the asserted anchor bound `|A-m*|<=B` itself holds only with probability at least `1-alpha_A`, combine the events by a union bound: total noncoverage is at most `alpha_s+alpha_A`. Allocate `alpha_s` from the total budget before using the formula. This illustration uses `alpha_A=0` (an exact/deterministic anchor-bound assumption); it does not calibrate an uncertain anchor.",
        "",
        "For centered `X_i=Z_i-E[Z_i]` and color classes `C_r`, Holder gives `E exp(lambda sum_i X_i) <= product_r {E exp(chi*lambda sum_{i in C_r} X_i)}^(1/chi)`. Within each color class the variables are jointly independent; Hoeffding's lemma for a width-2 variable then bounds the right side by `exp(chi*lambda^2*N/2)`. Chernoff's inequality gives a one-sided tail `exp(-N*epsilon^2/(2*chi))`; a union bound over the fixed family `F` yields the displayed penalty at tail budget `alpha_s`.",
        "",
        "This is a bounded block-risk estimand, not the ensemble-risk estimand. Let `\u03c6_hat` be the planned-sample failure fraction, `zbar_valid` the observed valid-block mean, and code each failed planned block as `-1`; then `zbar=(1-\u03c6_hat)*zbar_valid-\u03c6_hat`. Concentration targets the mean of the coded scores over all `N` planned blocks, not only the survivors. A failure rate selected after observing results is not a new denominator.",
        "For a squared-error contrast `D(A)=(A-y_ref)^2-(A-y_candidate)^2`, the anchor perturbation is exactly `D(A)-D(m*)=2(A-m*)(y_candidate-y_ref)`. Therefore `2B/R` after division by `R^2` is justified only if `|A-m*|<=B` and the predeclared candidate/reference prediction difference satisfies `|y_candidate-y_ref|<=R`. If the score has additional variance, fitting, clipping, or nonlinear terms, derive and add their own Lipschitz bounds; do not reuse `2B/R` by default.",
        "Allocate the total noncoverage budget before evaluation. Conditional score error is `alpha_s`; anchor coverage error is `alpha_A`; the implementation requires `alpha_s + alpha_A <= alpha_total` and the union bound is at most that sum. The figure uses the optimistic special case `alpha_s=alpha_total=0.05`, `alpha_A=0`.",
        "With `phi_hat` failures coded as the worst normalized improvement `-1`, relative anchor bias bound `B/R`, and normalized threshold `delta_star/R^2`, the planning inequality is",
        "",
        "```text",
        "L/R^2 = (1-phi_hat)*zbar_valid - phi_hat - sqrt(2*chi*log(F/alpha_s)/N) - 2*B/R",
        "zbar_valid > (delta_star/R^2 + sqrt(2*chi*log(F/alpha_s)/N) + 2*B/R + phi_hat)/(1-phi_hat)",
        "```",
        "",
        "The dependency graph must describe the actual planned score variables, including whatever worst-case coding is used for failures. A finite trajectory gap alone does not establish such a graph. A finite-state Markov chain is not automatically finitely dependent; mixing, autocorrelation, or spectral-gap arguments need their own score-specific assumptions and bound. If no justified graph bound is available, the dependence-adjusted certificate is unavailable and the decision remains `ABSTAIN/NOT_READY`.",
        "The bounded score must be clipped or otherwise range-limited before data access, with the threshold and physical interpretation frozen. Clipping protects concentration from unbounded tails but changes the estimand; it does not make correlated blocks independent. A paired block bootstrap, HAC/effective-sample-size estimate, or median-of-means result may be reported as a sensitivity diagnostic, but cannot replace a justified dependence bound or certify the physical score on its own.",
        "",
        "## Illustration",
        "",
        "![Dependency sensitivity](dependence_sensitivity.png)",
        "",
        "The plot uses only `F=2`, `alpha_s=0.05`, `alpha_A=0`, `B/R=0`, zero normalized threshold, and zero failures. In the `N=20` panel, chi=1 reproduces the independent-block penalty (about 0.607); chi=2 raises it to about 0.859; chi=3 exceeds the maximum possible normalized improvement of 1. Thus even this optimistic bound cannot certify a positive threshold at chi≥3 for N=20. In the second panel the same required-mean boundary is shown for N=20, 100 and 400. These curves are algebraic what-if values, not observed effects, chosen tolerances, or proof that a particular chi applies.",
        "",
        "`effective_block_count_algebraic_only=N/chi` in the CSV is merely a rewrite of this bound's penalty. It is not an empirical effective sample size, integrated autocorrelation estimate, or evidence that the trajectory blocks are independent.",
        "",
        "## Next steps and stop rule",
        "",
        "1. Keep the current evidence status `NOT_READY/UNCERTIFIED`; retain sealed E/gap/validation/holdout data and `physical_score=null`.",
        "2. Before claiming a dependence-adjusted bound, obtain an independently reviewable, score-specific justification for a dependency graph / chromatic-number upper bound, or propose a different concentration theorem with explicit mixing or spectral-gap inputs and finite-sample penalty.",
        "3. If only a dependence diagnostic is available, report it as diagnostic and abstain from a calibrated physical certificate. Do not infer a graph color count from trajectory spacing, bootstrap resamples, RHS count, or this figure.",
        "4. Any later calibration must keep planned/valid/failed/abstained denominators separate and include a dependence stress figure plus a plain-language interpretation of the actual bound and its assumptions.",
        "",
        "Deterministic rebuild into a new directory: `python3 -B -m tasks.pion_projection.dependency_graph_bound --output NEW_DIRECTORY`.",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n")
    artifact_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (out / "README.md", out / "summary.json", out / "sensitivity.csv",
                  out / "dependence_sensitivity.png")
    }
    (out / "artifact_hashes.json").write_text(json.dumps(artifact_hashes, indent=2) + "\n")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
