"""Reproducible Eq. (3.26) sensitivity report; reads no physical data."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

from .bounded_block_risk_markov_gap import markov_gap_radius
from .bounded_block_risk_markov_varying import (
    markov_time_varying_score_lcb,
    time_varying_markov_radius,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/pion_projection/bounded_block_risk_markov_varying_20261124_v1"
N_GRID = (20, 100, 500)
K_GRID = (1, 2, 4, 8)
GAMMA_GRID = (0.05, 0.2, 0.8)
FAMILY = 2
ALPHA_SCORE = 0.04


def _receipt(label: str) -> str:
    return hashlib.sha256(f"SYNTHETIC_ONLY:{label}".encode()).hexdigest()


def _class_counts(blocks: int, k_ps: int) -> list[int]:
    return [len(range(residue, blocks, k_ps)) for residue in range(k_ps)]


def _variance_bounds(blocks: int, k_ps: int, profile: str) -> tuple[list[float], list[float]]:
    counts = _class_counts(blocks, k_ps)
    if profile == "balanced_low_variance_upper":
        per_score = [0.05] * k_ps
    elif profile == "imbalanced_positive_upper":
        per_score = [0.005] + ([0.2] * (k_ps - 1))
    else:
        raise ValueError(f"unknown variance profile: {profile}")
    return [count * value for count, value in zip(counts, per_score)], per_score


def _make_figure(rows: list[dict], tv_rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    ax = axes[0]
    styles = {
        "balanced_low_variance_upper": ("tab:blue", "balanced upper bounds"),
        "imbalanced_positive_upper": ("tab:orange", "imbalanced upper bounds"),
    }
    selected_k = 4
    selected_gamma = 0.2
    for profile, (color, label) in styles.items():
        subset = [row for row in rows if row["profile"] == profile
                  and row["k_ps"] == selected_k
                  and row["gamma_ps_lower"] == selected_gamma]
        subset.sort(key=lambda row: row["blocks"])
        ax.plot([row["blocks"] for row in subset],
                [row["varying_radius"] for row in subset],
                marker="o", color=color, label=f"Eq. (3.26), {label}")
        ax.plot([row["blocks"] for row in subset],
                [row["fixed_radius_comparator"] for row in subset],
                marker="x", linestyle="--", color=color,
                label=f"Eq. (3.25) comparator, {label}")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2,
               label="best-case positive-LCB boundary")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("planned block states N")
    ax.set_ylabel("concentration radius (score units)")
    ax.set_title("Hypothetical radius at k_ps=4, gamma_ps=0.2")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    selected = [row for row in rows if row["blocks"] == 500
                and row["gamma_ps_lower"] == 0.2]
    for profile, (color, label) in styles.items():
        subset = sorted((row for row in selected if row["profile"] == profile),
                        key=lambda row: row["k_ps"])
        ax.plot([row["k_ps"] for row in subset],
                [row["M_over_k_ps"] for row in subset], marker="o",
                color=color, label=label)
    ax.set_xlabel("k_ps (assumed maximizing block length)")
    ax.set_ylabel("M / k_ps")
    ax.set_title("The retained range-factor cost")
    ax.grid(True, alpha=0.25)
    ax.legend()
    figure.suptitle("Time-varying Markov score sensitivity (synthetic assumptions)")
    figure.savefig(output, dpi=170)
    plt.close(figure)


def build_report(output: str | Path = DEFAULT_OUT) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    failed = 0
    for blocks in N_GRID:
        for k_ps in K_GRID:
            for gamma in GAMMA_GRID:
                for profile in ("balanced_low_variance_upper",
                                "imbalanced_positive_upper"):
                    class_bounds, per_score = _variance_bounds(blocks, k_ps, profile)
                    try:
                        varying = time_varying_markov_radius(
                            blocks=blocks,
                            gamma_ps_lower=gamma,
                            k_ps=k_ps,
                            variance_upper_by_residue=class_bounds,
                            family=FAMILY,
                            alpha_stationary=ALPHA_SCORE,
                        )
                        fixed = markov_gap_radius(
                            blocks=blocks,
                            gamma_ps_lower=gamma,
                            family=FAMILY,
                            alpha_s=ALPHA_SCORE,
                            variance_bound=max(per_score),
                        )
                        valid = varying["radius"] < 1.0
                        rows.append({
                            "blocks": blocks,
                            "k_ps": k_ps,
                            "gamma_ps_lower": gamma,
                            "profile": profile,
                            "per_score_variance_upper": json.dumps(per_score),
                            "variance_upper_by_residue": json.dumps(class_bounds),
                            "variance_sum_upper": varying["variance_sum_upper"],
                            "M": varying["M"],
                            "M_over_k_ps": varying["M_over_k_ps"],
                            "varying_radius": varying["radius"],
                            "fixed_radius_comparator": fixed,
                            "best_case_zero_threshold_reachable": valid,
                            "planned": 1,
                            "valid": int(valid),
                            "failed": 0,
                            "abstained": int(not valid),
                        })
                    except (ValueError, OverflowError, ZeroDivisionError):
                        failed += 1

    tv_rows: list[dict] = []
    for delta in (0.0, 0.001, 0.005, 0.01, 0.015, 0.02):
        if FAMILY * delta >= ALPHA_SCORE:
            tv_rows.append({
                "delta": delta,
                "stationary_alpha": 0.0,
                "radius": None,
                "tv_mean_shift": 2.0 * delta,
                "best_case_lcb_without_anchor": None,
                "status": "ABSTAIN_ALPHA_EXHAUSTED",
            })
            continue
        bounds, _ = _variance_bounds(100, 4, "balanced_low_variance_upper")
        radius = time_varying_markov_radius(
            blocks=100,
            gamma_ps_lower=0.2,
            k_ps=4,
            variance_upper_by_residue=bounds,
            family=FAMILY,
            alpha_stationary=ALPHA_SCORE - FAMILY * delta,
        )["radius"]
        best_case_lcb = 1.0 - radius - 2.0 * delta
        tv_rows.append({
            "delta": delta,
            "stationary_alpha": ALPHA_SCORE - FAMILY * delta,
            "radius": radius,
            "tv_mean_shift": 2.0 * delta,
            "best_case_lcb_without_anchor": best_case_lcb,
            "status": "BEST_CASE_POSITIVE_LCB_REACHABLE" if best_case_lcb > 0
            else "ABSTAIN_BOUND_EXCEEDS_MAX_SCORE",
        })

    fixture_scores = [
        -0.70 + ((index * 7) % 17) / 17 * 1.50
        for index in range(20)
    ]
    valid = [index not in {4, 13} for index in range(20)]
    # Failed slots may contain NaN; they are replaced with -1 and kept in N.
    import numpy as np

    demo_values = np.asarray(fixture_scores, dtype=float)
    demo_values[~np.asarray(valid, dtype=bool)] = np.nan
    demo_result = markov_time_varying_score_lcb(
        demo_values,
        valid_blocks=np.asarray(valid, dtype=bool),
        gamma_ps_lower=0.2,
        k_ps=2,
        variance_upper_by_residue=[10.0, 10.0],
        family=FAMILY,
        alpha_score=ALPHA_SCORE,
        alpha_anchor=0.01,
        alpha_total=0.05,
        initial_law_mode="post_burn_in_tv",
        burn_in_blocks=8,
        initial_law_tv_bound=0.001,
        anchor_score_shift_bound=0.02,
        initial_law_receipt_sha256=_receipt("demo-tv"),
        anchor_shift_receipt_sha256=_receipt("demo-anchor-shift"),
        score_schedule_receipt_sha256=_receipt("demo-schedule"),
        variance_receipt_sha256=_receipt("demo-variance"),
        k_ps_receipt_sha256=_receipt("demo-kps"),
    )
    all_failed_result = markov_time_varying_score_lcb(
        [float("nan")] * 4,
        valid_blocks=np.asarray([False] * 4),
        gamma_ps_lower=0.2,
        k_ps=2,
        variance_upper_by_residue=[2.0, 2.0],
        family=FAMILY,
        alpha_score=ALPHA_SCORE,
        alpha_anchor=0.01,
        alpha_total=0.05,
        initial_law_mode="stationary",
        burn_in_blocks=0,
        initial_law_tv_bound=0.0,
        anchor_score_shift_bound=0.0,
        initial_law_receipt_sha256=_receipt("demo-stationary"),
        anchor_shift_receipt_sha256=_receipt("demo-zero-anchor"),
        score_schedule_receipt_sha256=_receipt("all-failed-schedule"),
        variance_receipt_sha256=_receipt("all-failed-variance"),
        k_ps_receipt_sha256=_receipt("all-failed-kps"),
    )

    fieldnames = list(rows[0])
    with (out / "scenario_grid.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (out / "tv_sensitivity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tv_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(tv_rows)
    figure = out / "markov_varying_sensitivity.png"
    _make_figure(rows, tv_rows, figure)

    scenario_counts = {
        key: sum(row[key] for row in rows) + (failed if key == "failed" else 0)
        for key in ("planned", "valid", "failed", "abstained")
    }
    demo = {
        "partial_failure_fixture": demo_result,
        "all_failed_fixture": all_failed_result,
        "partial_failure_fixture_denominators": demo_result["denominators"],
        "partial_failure_result_denominators": demo_result["result_denominators"],
        "all_failed_fixture_denominators": all_failed_result["denominators"],
        "all_failed_result_denominators": all_failed_result["result_denominators"],
    }
    (out / "fixtures.json").write_text(json.dumps(demo, indent=2, allow_nan=False) + "\n")
    protocol = {
        "status": "SYNTHETIC_ANALYTIC_SENSITIVITY_ONLY",
        "theorem": "Paulin arXiv:1212.2015v5 Theorem 3.11 / Eq. (3.26)",
        "stationary_alpha": ALPHA_SCORE,
        "family": FAMILY,
        "n_grid": N_GRID,
        "k_ps_grid": K_GRID,
        "gamma_ps_lower_grid": GAMMA_GRID,
        "variance_profiles_are_hypothetical_upper_bounds": True,
        "fixed_comparator": "Paulin Eq. (3.25) radius using the maximum per-index variance bound for each profile; formula-only context, not a simultaneous theorem for varying f_i",
        "coverage_simulation": False,
        "physical_data_access": "none",
        "sealed_inputs_accessed": False,
        "gpu_quda": False,
        "physical_score": None,
        "certified": False,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    report = [
        "# Pion time-varying Markov score sensitivity",
        "",
        "**ANALYTIC/SOFTWARE DIAGNOSTIC ONLY. `physical_score=null`; no measurements or sealed inputs were read.**",
        "",
        "## Theorem and implemented route",
        "",
        "Paulin's corrected arXiv v5 Theorem 3.11, Eq. (3.26), covers a predeclared sequence `f_i(X_i)` of time-varying functions on a stationary Markov chain and the finite-plan estimand `theta_N=N^-1 sum_i E_pi[f_i(X_i)]`. This is not automatically the same target as the fixed-score route; a human must review that the schedule preserves the accepted target. It uses `V_sum=sum_i Var_pi(f_i)` and, for residue classes modulo the smallest maximizing `k_ps`, `V_r=sum_j Var_pi(f_{r+j k_ps})`; `M=(sum_r sqrt(V_r))/min_r sqrt(V_r)`. The exponent denominator is `8 V_sum + 20 t C M/k_ps`. The paper explicitly says it is unknown whether `M/k_ps` can be replaced by 1, so this implementation retains it.",
        "",
        "The helper takes positive reviewed upper bounds `V'_r >= V_r` for each residue class and computes `V'_sum=sum_r V'_r`, `M'=(sum_r sqrt(V'_r))/min_r sqrt(V'_r)`. As allowed by the paper's Remark 3.12, the same inflated class bounds are used in both terms. Zero bounds fail closed; an explicitly justified positive conservative bound is needed for every class. `k_ps` must be at most N so no empty residue class is silently accepted.",
        "",
        "For `L=log(2F/alpha_stationary)`, the radius solves `gamma_ps t^2 - 20 C (M/k_ps) L t - 8 V_sum L = 0`, and is `t/N`. A post-burn-in TV bound spends `F*delta` from `alpha_score`; the operational mean also subtracts `2*delta`. An externally justified anchor-induced score shift is separately subtracted. The code cannot verify any receipt or establish that `k_ps` truly maximizes the pseudo-gap.",
        "",
        "Primary source: [Paulin, Concentration inequalities for Markov chains, corrected arXiv v5](https://arxiv.org/html/1212.2015v5), Theorem 3.11 / Eqs. (3.25)-(3.26), Remark 3.12, Proposition 3.15 / Eq. (3.29). The author says at the start of the paper that the previous general-state Bernstein proof, including the published version, was incomplete and presents a correction.",
        "",
        "## Synthetic analytic grid",
        "",
        f"The grid has {scenario_counts['planned']} planned formula-only cells over `N={N_GRID}`, `k_ps={K_GRID}`, `gamma_ps_lower={GAMMA_GRID}`, and two explicitly hypothetical variance-upper profiles. The balanced profile assigns per-score upper bound 0.05; the imbalanced profile assigns 0.005 to the first residue and 0.2 to the others. These are not estimated from project data. `valid` means only that a zero-threshold LCB is reachable if every score equals 1; radii at least 1 count as abstentions. Grid denominators planned/valid/failed/abstained are `{scenario_counts['planned']}/{scenario_counts['valid']}/{scenario_counts['failed']}/{scenario_counts['abstained']}`.",
        "",
        "The dashed Eq. (3.25) comparator uses a common per-index variance upper bound equal to the largest profile value. It provides formula-only scale context and is not a valid theorem for the varying-score schedule. Where Eq. (3.26) is narrower, the change depends on its extra variance-upper and `k_ps` receipts; this plot is not evidence that learned or block-varying scores improve physical performance.",
        "",
        "![Eq. (3.26) radius and M/k_ps sensitivity](markov_varying_sensitivity.png)",
        "",
        "Left panel: radius versus planned N at assumed `k_ps=4` and `gamma_ps_lower=0.2`, with Eq. (3.25) shown as a formula comparator. The horizontal line is the maximum possible mean for scores in `[-1,1]`; radii above it preclude a positive zero-threshold LCB even in the best case. Right panel: the actual `M/k_ps` term under balanced and imbalanced variance upper bounds; the imbalanced profile illustrates why setting that term to 1 is not allowed.",
        "",
        "## Initial-law sensitivity and deterministic fixtures",
        "",
        f"The separate TV table fixes `N=100`, `k_ps=4`, `gamma_ps_lower=0.2`; it shows `delta`, remaining stationary alpha, radius, and the best-case LCB before anchor correction. Scenarios are planned/valid/failed/abstained `{len(tv_rows)}/{sum(row['status'] == 'BEST_CASE_POSITIVE_LCB_REACHABLE' for row in tv_rows)}/{sum(row['status'].startswith('ERROR') for row in tv_rows)}/{sum(row['status'].startswith('ABSTAIN') for row in tv_rows)}`. It is an analytic sensitivity, not an estimate of actual burn-in TV.",
        "",
        f"A deterministic partial-failure fixture uses 20 illustrative score values with two failed slots; block denominators are `{demo_result['denominators']}` and comparison-result denominators are `{demo_result['result_denominators']}`. The all-failed fixture has block denominators `{all_failed_result['denominators']}` and comparison-result denominators `{all_failed_result['result_denominators']}`; it emits no LCB. These fixtures validate data-flow and denominator behavior only, not finite-sample coverage.",
        "",
        "## Decision flow",
        "",
        "```text",
        "freeze the entire time-varying f_i schedule and block map before observing scores",
        "  -> obtain reviewed block-chain assumptions and gamma_ps lower bound",
        "  -> provide a receipt that k_ps is the smallest maximizing pseudo-gap index",
        "  -> provide positive upper bounds for every residue-class variance sum and their receipt",
        "  -> retain M/k_ps; allocate F*delta from alpha_score when the scored chain is nonstationary",
        "  -> subtract 2*delta and an externally justified anchor-score shift",
        "  -> if any condition/receipt is absent or F*delta >= alpha_score: ABSTAIN",
        "  -> keep physical_score=null until the separate target, anchor, and reference gates pass",
        "```",
        "",
        "## Boundaries and reproducibility",
        "",
        "The schedule and variance bounds in the fixtures are synthetic. No coverage Monte Carlo was run because the project-specific chain, `k_ps`, variance receipts, anchor shift, target, and reference are unavailable. The current fixed-score path remains the default; using the time-varying helper defines a schedule-specific finite-plan mean and requires review that it is equivalent to the accepted estimand. This diagnostic does not change task status or produce a physical score. Pion remains `NOT_READY/UNCERTIFIED`, data remain sealed, and `physical_score=null`.",
        "",
        "Rebuild with `python -B -m tasks.pion_projection.bounded_block_risk_markov_varying_diagnostic --output NEW_DIRECTORY`.",
        "",
    ]
    (out / "README.md").write_text("\n".join(report))
    snapshots = {
        "snapshot_markov_varying.py": Path(__file__).with_name("bounded_block_risk_markov_varying.py"),
        "snapshot_markov_varying_diagnostic.py": Path(__file__),
        "snapshot_contract_inspector.py": Path(__file__).with_name("bounded_block_risk_contract.py"),
        "snapshot_contract_v1.json": Path(__file__).with_name("bounded_block_risk_contract_v1.json"),
    }
    for name, source in snapshots.items():
        shutil.copyfile(source, out / name)
    payloads = [
        "scenario_grid.csv", "tv_sensitivity.csv", "fixtures.json",
        "protocol.json", "markov_varying_sensitivity.png", "README.md",
        *snapshots,
    ]
    manifest = {
        "algorithm": "sha256",
        "files": {
            name: hashlib.sha256((out / name).read_bytes()).hexdigest()
            for name in payloads
        },
    }
    (out / "artifact_hashes.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "SYNTHETIC_ANALYTIC_SENSITIVITY_ONLY",
        "physical_score": None,
        "scenario_denominators": scenario_counts,
        "scenario_count": len(rows),
        "tv_row_count": len(tv_rows),
        "demo_denominators": demo_result["denominators"],
        "all_failed_status": all_failed_result["status"],
        "files": payloads + ["artifact_hashes.json"],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    arguments = parser.parse_args()
    print(json.dumps(build_report(arguments.output), indent=2))


if __name__ == "__main__":
    main()
