"""F=3 analytic-only sensitivity for the selected Pion block-risk plan.

No observations, correlators, receipts, sealed splits, or GPU/QUDA inputs are
read. The score mean is set to its mathematical maximum of one only to show an
optimistic best-case LCB; it is not a simulated or measured score.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bounded_block_risk_contract import selected_protocol
from .bounded_block_risk_markov_gap import markov_gap_radius_after_initial_tv


DEFAULT_OUT = Path(__file__).resolve().parent / "F3_markov_sensitivity_v1"
BLOCKS = (20, 100, 500, 1000, 2000, 4000, 8000)
GAMMAS = (0.05, 0.1, 0.2, 0.5, 1.0)
TV_BOUNDS = (0.0, 0.001, 0.005, 0.01, 0.013333333333333334, 0.015, 0.02)


def _anchor_score_penalty(plan: dict) -> float:
    anchor = plan["anchor_design"]
    alpha = float(anchor["alpha"])
    n = int(anchor["independent_stationary_units_required"])
    common_bias = float(anchor["common_bias_bound_C_max_over_R"])
    return 2.0 * (math.sqrt(math.log(2.0 / alpha) / (2.0 * n)) + common_bias)


def rows() -> list[dict]:
    plan = selected_protocol()
    family = int(plan["comparison_family"]["F"])
    alpha_score = float(plan["comparison_family"]["alpha_score"])
    anchor_penalty = _anchor_score_penalty(plan)
    threshold = float(plan["score"]["normalized_effect_threshold"])
    out = []
    for n in BLOCKS:
        for gamma in GAMMAS:
            for tv in TV_BOUNDS:
                row = {
                    "planned_blocks": n,
                    "gamma_ps_lower_assumed": gamma,
                    "initial_tv_per_comparison_assumed": tv,
                    "family_F": family,
                    "alpha_score": alpha_score,
                    "anchor_score_penalty_2B_over_R": anchor_penalty,
                    "normalized_effect_threshold": threshold,
                    "mean_assumed_for_best_case_only": 1.0,
                    "score_radius": None,
                    "stationary_alpha_after_family_tv": None,
                    "initial_law_mean_shift_2TV": 2.0 * tv,
                    "best_case_normalized_lcb": None,
                    "best_case_threshold_reachable": False,
                    "planned": 1,
                    "valid": 0,
                    "failed": 0,
                    "abstained": 0,
                    "status": None,
                }
                if family * tv >= alpha_score:
                    row.update({"abstained": 1, "status": "ABSTAIN_TV_SPENDS_SCORE_ALPHA"})
                else:
                    radius, stationary_alpha = markov_gap_radius_after_initial_tv(
                        blocks=n,
                        gamma_ps_lower=gamma,
                        family=family,
                        alpha_score=alpha_score,
                        initial_law_tv_bound=tv,
                    )
                    best_lcb = 1.0 - radius - 2.0 * tv - anchor_penalty
                    row.update({
                        "score_radius": radius,
                        "stationary_alpha_after_family_tv": stationary_alpha,
                        "best_case_normalized_lcb": best_lcb,
                        "best_case_threshold_reachable": best_lcb > threshold,
                        "valid": 1,
                        "status": ("BEST_CASE_THRESHOLD_REACHABLE" if best_lcb > threshold
                                   else "FORMULA_DEFINED_THRESHOLD_UNREACHABLE"),
                    })
                out.append(row)
    return out


def build(output: str | Path = DEFAULT_OUT) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    plan = selected_protocol()
    data = rows()
    fields = list(data[0])
    with (out / "sensitivity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)

    valid_rows = [row for row in data if row["valid"]]
    tv_plot = 0.001
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    ax = axes[0]
    for gamma in GAMMAS:
        chosen = [row for row in valid_rows
                  if row["gamma_ps_lower_assumed"] == gamma
                  and row["initial_tv_per_comparison_assumed"] == tv_plot]
        ax.plot([row["planned_blocks"] for row in chosen],
                [row["score_radius"] for row in chosen], marker="o",
                label=fr"$\gamma_{{ps}}={gamma:g}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("planned scored blocks N")
    ax.set_ylabel("Markov score radius")
    ax.set_title(rf"F=3, $\alpha_s=0.04$, TV={tv_plot:g}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    matrix = np.full((len(GAMMAS), len(TV_BOUNDS)), np.nan)
    for row in valid_rows:
        if row["planned_blocks"] == 4000:
            i = GAMMAS.index(row["gamma_ps_lower_assumed"])
            j = TV_BOUNDS.index(row["initial_tv_per_comparison_assumed"])
            matrix[i, j] = row["best_case_normalized_lcb"]
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, aspect="auto", origin="lower", cmap="viridis",
                      extent=(-0.5, len(TV_BOUNDS)-0.5, -0.5, len(GAMMAS)-0.5))
    ax.set_xticks(range(len(TV_BOUNDS)), [f"{v:g}" for v in TV_BOUNDS], rotation=35)
    ax.set_yticks(range(len(GAMMAS)), [f"{v:g}" for v in GAMMAS])
    ax.set_xlabel("assumed per-comparison post-burn-in TV")
    ax.set_ylabel(r"assumed $\gamma_{ps}$ lower bound")
    ax.set_title("best-case normalized LCB, N=4000")
    fig.colorbar(image, ax=ax, label="LCB after 2TV and anchor correction")
    ax.contour(np.arange(len(TV_BOUNDS)), np.arange(len(GAMMAS)),
               np.nan_to_num(matrix, nan=-10.0), levels=[0.05], colors="white",
               linewidths=1.5)

    ax = axes[2]
    gamma = 0.2
    for n in (20, 500, 4000):
        chosen = [row for row in valid_rows if row["planned_blocks"] == n
                  and row["gamma_ps_lower_assumed"] == gamma]
        ax.plot([row["initial_tv_per_comparison_assumed"] for row in chosen],
                [row["best_case_normalized_lcb"] for row in chosen], marker="o",
                label=f"N={n}")
    ax.axhline(plan["score"]["normalized_effect_threshold"], color="black",
               linestyle="--", label="minimum effect 0.05")
    ax.set_xlabel("assumed per-comparison post-burn-in TV")
    ax.set_ylabel("best-case normalized LCB")
    ax.set_title(r"$\gamma_{ps}=0.2$, F=3")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.suptitle("Pion bounded block-risk: selected F=3 method sensitivity (analytic only)")
    svg_path = out / "sensitivity.svg"
    fig.savefig(svg_path)
    plt.close(fig)
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")

    denominator = {
        "formula_scenarios": {
            "planned": len(data),
            "valid": sum(row["valid"] for row in data),
            "failed": sum(row["failed"] for row in data),
            "abstained": sum(row["abstained"] for row in data),
            "unit": "one analytic parameter combination; not a sample or measurement",
        },
        "physical_measurements": {
            "planned": 0, "valid": 0, "failed": 0, "abstained": 0,
            "unit": "new physical scored blocks or anchor observations",
        },
    }
    selected = next(row for row in data if row["planned_blocks"] == 4000
                    and row["gamma_ps_lower_assumed"] == 0.2
                    and row["initial_tv_per_comparison_assumed"] == 0.001)
    README = [
        "# Pion F=3 block-risk sensitivity",
        "",
        "**Analytic planning only. No score blocks, correlators, anchor observations, receipts, sealed splits, GPU, or QUDA inputs were read. `physical_score=null`.**",
        "",
        "## Selected design and formula",
        "",
        "The figure and table use the selected three-comparison family (`F=3`), `alpha_score=0.04`, `alpha_anchor=0.01`, normalized effect threshold `0.05`, and bounded-IID anchor requirement of 1000 independent stationary units. The anchor correction is computed from the stated Hoeffding radius and `C/R=0.01`; it is a design budget, not an observed anchor error.",
        "",
        "For each assumed `N`, pseudo-gap lower bound and per-comparison TV bound, the existing corrected fixed-score Markov helper evaluates the stationary Bernstein radius using remaining score alpha `0.04 - F*TV`. The optimistic best-case LCB is `1 - epsilon - 2*TV - 2B/R`, with score mean fixed at its mathematical maximum 1. A parameter combination is abstained if `F*TV >= alpha_score`; an LCB that is defined but below 0.05 is reported as unreachable even under this best-case mean.",
        "",
        "![F=3 analytic sensitivity](sensitivity.svg)",
        "",
        "Panel 1 varies `N` and `gamma_ps` at assumed `TV=0.001`. Panel 2 shows the best-case normalized LCB at `N=4000` over `gamma_ps` and TV; blank cells are alpha-budget abstentions. Panel 3 isolates TV sensitivity for `gamma_ps=0.2` at three block counts. None of these values estimates the project chain's gap, TV, anchor bias, or coverage.",
        "",
        "## Flow and denominators",
        "",
        "```text",
        "frozen E-chain reference and score map -> reviewed length-5 block-chain conditions",
        "  -> gamma_ps lower bound + burn-in/TV receipt -> reserve F*TV and compute Markov radius",
        "  -> independent bounded-IID anchor coverage + C bound -> subtract 2B/R",
        "  -> evaluate full N planned blocks (failed blocks stay -1) -> LCB > 0.05 or ABSTAIN",
        "  -> separate confirmation authorization and physical review; score remains null until then",
        "```",
        "",
        f"Formula-scenario denominators: planned/valid/failed/abstained = {denominator['formula_scenarios']['planned']}/{denominator['formula_scenarios']['valid']}/{denominator['formula_scenarios']['failed']}/{denominator['formula_scenarios']['abstained']}. These count formula combinations only. New physical measurement denominators are 0/0/0/0. At the selected planning point `N=4000`, assumed `gamma_ps=0.20`, assumed `TV=0.001`, score radius is `{selected['score_radius']:.6g}` and best-case normalized LCB is `{selected['best_case_normalized_lcb']:.6g}`. These are conditional arithmetic outputs, not evidence that the assumed values hold.",
        "",
        "## Still required before a certificate",
        "",
        "- Independent candidate-blind E-chain correlator receipt and successful reference/model gate.",
        "- Reviewed score-specific block-chain and `gamma_ps >= 0.20` lower-bound evidence; block spacing alone is insufficient.",
        "- Burn-in 8 / post-burn-in TV `<=0.001` receipt for every comparison.",
        "- 1000 independent stationary bounded anchor units, coverage receipt, common-bias `C/R<=0.01` evidence, and conditional-independence review.",
        "- Authorized unsealed confirmation data and independent review. Production data and GPU/QUDA remain unauthorized.",
        "",
    ]
    (out / "README.md").write_text("\n".join(README))
    protocol = {
        "schema": "pion-f3-markov-sensitivity-v1",
        "status": "ANALYTIC_ONLY_NOT_READY",
        "selected_protocol": plan,
        "grid": {"blocks": BLOCKS, "gamma_ps_lower_assumed": GAMMAS,
                 "initial_tv_assumed": TV_BOUNDS},
        "formula_scenario_denominators": denominator["formula_scenarios"],
        "physical_measurement_denominators": denominator["physical_measurements"],
        "selected_planning_point": selected,
        "physical_score": None,
        "certified": False,
        "data_access": "none",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    (out / "snapshot_bounded_block_risk_f3_plan.py").write_bytes(Path(__file__).read_bytes())
    artifacts = ["README.md", "sensitivity.csv", "sensitivity.svg", "protocol.json",
                 "snapshot_bounded_block_risk_f3_plan.py"]
    hashes = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
              for name in artifacts}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    return {"output": str(out), "denominators": denominator,
            "selected_planning_point": selected, "files": artifacts + ["artifact_hashes.json"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
