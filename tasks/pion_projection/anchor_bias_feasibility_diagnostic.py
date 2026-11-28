"""Analytic feasibility of the selected Pion anchor common-bias bridge.

This diagnostic uses only the frozen method contract and concentration formula.
It does not read correlators, receipts, hidden splits, or physical measurements.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bounded_block_risk_contract import selected_protocol
from .bounded_block_risk_anchor_markov import markov_anchor_bound
from .bounded_block_risk_markov_gap import markov_gap_radius_after_initial_tv


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/pion_projection/anchor_bias_feasibility_markov_anchor_20261126_v2"
CONTRACT = ROOT / "tasks/pion_projection/bounded_block_risk_contract_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_inputs() -> dict:
    plan = selected_protocol()
    target = plan["target"]
    score = plan["score"]
    anchor = plan["anchor_design"]
    family = plan["comparison_family"]
    dependence = plan["dependence_design"]
    support = [float(value) for value in anchor["support_interval"]]
    width = support[1] - support[0]
    n_anchor = int(anchor["planned_anchor_units"])
    alpha_anchor = float(anchor["alpha"])
    anchor_gamma = float(anchor["gamma_ps_lower_bound_requirement"])
    anchor_tv = float(anchor["post_burn_in_tv_bound"])
    anchor_burn_in = int(anchor["burn_in_units"])
    anchor_result = markov_anchor_bound(
        sample_count=n_anchor,
        gamma_ps_lower=anchor_gamma,
        alpha_anchor=alpha_anchor,
        support=tuple(support),
        common_bias_bound=0.0,
        initial_law_tv_bound=anchor_tv,
        burn_in_units=anchor_burn_in,
        stationary_start=False,
        score_scale_R=width,
    )
    random_radius_over_R = anchor_result["random_coverage_radius"] / width
    tv = float(dependence["post_burn_in_tv_bound_per_comparison"])
    radius, _ = markov_gap_radius_after_initial_tv(
        blocks=int(dependence["planned_scored_blocks"]),
        gamma_ps_lower=float(dependence["gamma_ps_lower_bound_requirement"]),
        family=int(family["F"]),
        alpha_score=float(family["alpha_score"]),
        initial_law_tv_bound=tv,
    )
    delta_star_over_R2 = float(score["normalized_effect_threshold"])
    selected_c_over_R = float(anchor["common_bias_bound_C_max_over_R"])
    best_case_before_c = 1.0 - radius - 2.0 * tv - 2.0 * random_radius_over_R
    feasibility_ceiling = (best_case_before_c - delta_star_over_R2) / 2.0
    return {
        "target": target["observable"],
        "units": score["units"],
        "support_interval": support,
        "support_width_R": width,
        "n_anchor": n_anchor,
        "alpha_anchor": alpha_anchor,
        "anchor_gamma_ps_lower_assumed": anchor_gamma,
        "anchor_tv_bound_assumed": anchor_tv,
        "anchor_burn_in_units_assumed": anchor_burn_in,
        "anchor_stationary_alpha_after_tv": anchor_result["stationary_alpha_after_tv"],
        "anchor_method": anchor["method"],
        "anchor_random_radius_over_R": random_radius_over_R,
        "planned_score_blocks": int(dependence["planned_scored_blocks"]),
        "gamma_ps_lower_assumed": float(dependence["gamma_ps_lower_bound_requirement"]),
        "post_burn_in_tv_assumed": tv,
        "family_F": int(family["F"]),
        "alpha_score": float(family["alpha_score"]),
        "score_radius": radius,
        "delta_star_over_R2": delta_star_over_R2,
        "selected_C_over_R": selected_c_over_R,
        "best_case_lcb_before_C": best_case_before_c,
        "C_over_R_max_for_any_best_case_pass": max(0.0, feasibility_ceiling),
        "selected_C_units": selected_c_over_R * width,
        "trivial_range_only_C_over_R": 1.0,
    }


def _plot(path: Path, rows: list[dict], inputs: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot([row["C_over_R"] for row in rows],
            [row["best_case_normalized_lcb"] for row in rows],
            color="#1769aa", linewidth=2.4,
            label="best possible LCB (assume score mean = 1)")
    ax.axhline(inputs["delta_star_over_R2"], color="#b71c1c", linestyle="--",
               label="required minimum effect 0.05")
    ax.axvline(inputs["selected_C_over_R"], color="#388e3c", linestyle=":",
               label="selected bridge budget C/R = 0.01")
    ax.axvline(inputs["C_over_R_max_for_any_best_case_pass"], color="#e07a16",
               linestyle="-.", label="largest C/R with any best-case pass")
    ax.axvline(1.0, color="#555555", linestyle="--",
               label="trivial range-only bound C/R = 1")
    ax.fill_between([0, 1], -2, 0.05, color="#ef9a9a", alpha=0.12)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.7, 0.65)
    ax.set_xlabel("common-bias bound C / score scale R")
    ax.set_ylabel("optimistic normalized lower confidence bound")
    ax.set_title("Pion anchor bridge: bounded support alone is not informative")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def build(output: Path = DEFAULT_OUT) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    inputs = selected_inputs()
    ratios = np.linspace(0.0, 1.0, 101)
    rows = []
    for ratio in ratios:
        lcb = (inputs["best_case_lcb_before_C"] - 2.0 * float(ratio))
        rows.append({
            "C_over_R": float(ratio),
            "best_case_score_mean": 1.0,
            "best_case_normalized_lcb": lcb,
            "threshold": inputs["delta_star_over_R2"],
            "threshold_reachable_even_at_max_score_mean": int(
                lcb > inputs["delta_star_over_R2"]
            ),
            "planned": 1, "valid": 1, "failed": 0, "abstained": 0,
        })
    with (output / "bias_sensitivity.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    selected_lcb = inputs["best_case_lcb_before_C"] - 2 * inputs["selected_C_over_R"]
    trivial_lcb = inputs["best_case_lcb_before_C"] - 2.0
    protocol = {
        "schema": "pion-anchor-common-bias-feasibility-v2",
        "evidence_class": "analytic_selected_contract_sensitivity_only",
        "inputs": inputs,
        "formula": "best_case_LCB=1-score_radius-2*TV-2*(anchor_random_radius/R+C/R)",
        "selected_C_over_R_best_case_lcb": selected_lcb,
        "range_only_C_over_R_best_case_lcb": trivial_lcb,
        "formula_scenario_denominator": {
            "planned": len(rows), "valid": len(rows), "failed": 0, "abstained": 0,
            "unit": "one deterministic C/R value; not a measurement",
        },
        "physical_evidence_denominator": {
            "planned": 1, "valid": 0, "failed": 0, "abstained": 1,
            "reason": "no anchor scalar-map, common-bias, independence, reference, or chain receipts",
        },
        "physical_score": None,
        "status": "NOT_READY_UNCERTIFIED",
        "source_hashes": {
            "bounded_block_risk_contract_v1.json": _sha256(CONTRACT),
            "bounded_block_risk_contract.py": _sha256(Path(__file__).with_name("bounded_block_risk_contract.py")),
            "bounded_block_risk_markov_gap.py": _sha256(Path(__file__).with_name("bounded_block_risk_markov_gap.py")),
            "bounded_block_risk_anchor_markov.py": _sha256(Path(__file__).with_name("bounded_block_risk_anchor_markov.py")),
            "anchor_bias_feasibility_diagnostic.py": _sha256(Path(__file__)),
        },
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    _plot(output / "bias_feasibility.png", rows, inputs)
    readme = f"""# Pion common-bias feasibility under the selected bounded block-risk plan

**Analytic planning diagnostic only.** It uses the selected contract defaults and corrected Markov concentration helper. No correlators, receipts, cached propagators, sealed splits, GPU, or QUDA data were read. The score remains `NOT_READY/UNCERTIFIED`, `physical_score=null`.

## Calculation

The optimistic best-case normalized LCB sets the unknown score mean to its mathematical maximum 1:

```text
LCB_max = 1 - score_radius - 2*TV
             - 2*(anchor_random_radius/R + C/R).
```

The selected plan gives score radius `{inputs['score_radius']:.9f}`, `2*TV_score={2*inputs['post_burn_in_tv_assumed']:.6f}`, and conditional Markov-anchor random radius/R `{inputs['anchor_random_radius_over_R']:.9f}` from `n_A={inputs['n_anchor']}`, `gamma_A={inputs['anchor_gamma_ps_lower_assumed']}`, `TV_A={inputs['anchor_tv_bound_assumed']}`, burn-in `{inputs['anchor_burn_in_units_assumed']}`, and `alpha_A={inputs['alpha_anchor']}`. The anchor TV allowance is spent from `alpha_A`, leaving stationary tail level `{inputs['anchor_stationary_alpha_after_tv']:.6f}`; it is not added as a score-mean shift. Even under this best possible score mean, the threshold `delta_star/R^2={inputs['delta_star_over_R2']:.3f}` can only be reached if `C/R < {inputs['C_over_R_max_for_any_best_case_pass']:.6f}`. This is a necessary feasibility ceiling, not a sufficient certificate or a recommended estimate of `C`.

The selected design budget `C/R={inputs['selected_C_over_R']:.3f}` gives best-case LCB `{selected_lcb:.6f}`. If one uses only the trivial range fact that both the clipped anchor scalar and `m*` lie in `{inputs['support_interval']}`, then `C<=R={inputs['support_width_R']:.3f}` and `C/R=1`; the best-case LCB becomes `{trivial_lcb:.6f}`. Thus clipping or bounded support alone cannot support the chosen score. The required `C/R<=0.01` is a separate bridge that needs a physical estimator/bias argument and an independent receipt.

![Best-case lower bound versus common-bias allowance](bias_feasibility.png)

## Denominators and decision flow

- Formula sensitivity: planned/valid/failed/abstained = `101/101/0/0`.
- Physical evidence gate: `1/0/0/1`; anchor map, `C` coverage, independence, and chain/reference receipts are absent.

```text
declared support [0.1,3.0] + Markov receipts (n_A=4000, gamma_A>=0.20, burn-in=8, TV_A<=0.001)
  -> conditional random radius/R from corrected Markov Bernstein
  -> trivial support-only C/R <= 1 still gives best-case LCB below threshold
  -> need an estimator-specific common-bias bridge C/R <= 0.01 with receipt
  -> independently review scalar map, target correspondence, and score/anchor conditional independence
  -> only then consider a fresh authorized confirmation; otherwise ABSTAIN
```

The computation does not change `m*`, any selected alpha/sample-size defaults, or the status. The existing selected C budget is not inferred from D development data; this calculation shows why its proof is essential.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    files = sorted(path for path in output.iterdir() if path.is_file())
    hashes = {path.name: _sha256(path) for path in files}
    (output / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
