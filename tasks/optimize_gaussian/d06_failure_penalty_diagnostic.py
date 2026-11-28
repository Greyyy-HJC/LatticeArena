"""Exact sensitivity of bounded block-risk ratios to assigned-arm failures."""
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

from .bounded_block_risk import ARMS, R_STAR


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/optimize_gaussian/d06_failure_penalty_cpu_20261126_v1"
SUCCESS_RISKS = np.asarray((0.04, 0.08, 0.075), dtype=np.float64)
FAILURE_SCENARIOS = {
    "same_failure_rate_all_arms": (1.0, 1.0, 1.0),
    "trainable_w_only": (1.0, 0.0, 0.0),
    "comparators_only": (0.0, 1.0, 1.0),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_risks(failure_probabilities: object) -> np.ndarray:
    probabilities = np.asarray(failure_probabilities, dtype=np.float64)
    if probabilities.shape != (len(ARMS),):
        raise ValueError("failure_probabilities must have one value per arm")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("failure probabilities must be finite and lie in [0, 1]")
    # A declared algorithm failure is an outcome with maximum bounded loss 1.
    return probabilities + (1.0 - probabilities) * SUCCESS_RISKS


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario, multipliers in FAILURE_SCENARIOS.items():
        for probability in np.linspace(0.0, 0.5, 101):
            failure_probabilities = probability * np.asarray(multipliers)
            risks = expected_risks(failure_probabilities)
            ratio_slsqp = float(risks[1] / risks[0])
            ratio_chebyshev = float(risks[2] / risks[0])
            rows.append({
                "scenario": scenario,
                "common_probability_scale": float(probability),
                "p_failure_trainable_w": float(failure_probabilities[0]),
                "p_failure_slsqp": float(failure_probabilities[1]),
                "p_failure_chebyshev": float(failure_probabilities[2]),
                "risk_trainable_w": float(risks[0]),
                "risk_slsqp": float(risks[1]),
                "risk_chebyshev": float(risks[2]),
                "ratio_slsqp_over_trainable_w": ratio_slsqp,
                "ratio_chebyshev_over_trainable_w": ratio_chebyshev,
                "family_min_ratio": min(ratio_slsqp, ratio_chebyshev),
                "family_effect_gate_pass": int(
                    ratio_slsqp > R_STAR and ratio_chebyshev > R_STAR
                ),
                "planned": 1,
                "valid": 1,
                "failed": 0,
                "abstained": 0,
            })
    return rows


def _plot(path: Path, rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    symmetric = [row for row in rows if row["scenario"] == "same_failure_rate_all_arms"]
    p = [float(row["common_probability_scale"]) for row in symmetric]
    for arm, field in zip(ARMS, ("risk_trainable_w", "risk_slsqp", "risk_chebyshev")):
        axes[0].plot(p, [float(row[field]) for row in symmetric], linewidth=2,
                     label=arm.replace("_", " "))
    axes[0].set_title("Expected bounded risk when all arms fail equally")
    axes[0].set_xlabel("assigned-arm failure probability")
    axes[0].set_ylabel("expected block risk")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    colors = {"same_failure_rate_all_arms": "#1769aa",
              "trainable_w_only": "#c62828",
              "comparators_only": "#2e7d32"}
    for scenario, color in colors.items():
        selected = [row for row in rows if row["scenario"] == scenario]
        x = [float(row["common_probability_scale"]) for row in selected]
        y = [float(row["family_min_ratio"]) for row in selected]
        axes[1].plot(x, y, color=color, linewidth=2, label=scenario.replace("_", " "))
    axes[1].axhline(R_STAR, color="#111111", linestyle="--", linewidth=1.4,
                    label=rf"required ratio $r_\star={R_STAR:.3f}$")
    axes[1].set_title("Weaker of the two primary effect ratios")
    axes[1].set_xlabel("failure probability in the scenario arms")
    axes[1].set_ylabel(r"$\min(R_{SLSQP}/R_W, R_{Cheb}/R_W)$")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.suptitle("D06 failure policy: failed arm-blocks receive loss 1 and stay in the denominator")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def build(output: Path = DEFAULT_OUT) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    rows = _rows()
    with (output / "failure_sensitivity.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _plot(output / "failure_effect.png", rows)

    protocol = {
        "schema": "optimize-d06-failure-penalty-diagnostic-v1",
        "status": "ANALYTIC_SENSITIVITY_ONLY_NOT_CALIBRATION",
        "evidence_class": "exact_mixture_sensitivity",
        "data_access": "none; no gauge, reference-fit, sealed split, GPU, or QUDA input",
        "success_risk_examples": dict(zip(ARMS, SUCCESS_RISKS.tolist())),
        "failure_scenarios": {name: list(values) for name, values in FAILURE_SCENARIOS.items()},
        "estimand_rule": "X=1 for a declared assigned-arm optimizer failure; otherwise X is the predeclared clipped q loss",
        "mixture_formula": "mu_a = p_fail,a + (1-p_fail,a)*mu_success,a",
        "r_star": R_STAR,
        "formula_scenario_denominator": {
            "planned": len(rows), "valid": len(rows), "failed": 0, "abstained": 0,
            "unit": "one exact probability and failure-pattern scenario; not a dataset or measurement",
        },
        "invalid_input_rule": "missing/corrupt q outside a declared optimizer failure is not imputed and causes ABSTAIN at the runner",
        "optimizer_failure_rule": "the whole paired arm-macroblock receives loss 1 and remains in the planned denominator",
        "physical_score": None,
        "source_hashes": {
            "bounded_block_risk.py": _sha256(Path(__file__).with_name("bounded_block_risk.py")),
            "d06_failure_penalty_diagnostic.py": _sha256(Path(__file__)),
        },
        "limits": [
            "success risks and failure patterns are illustrative assumptions, not measured rates",
            "does not establish independence of physical macroblocks or a physical failure probability",
            "does not include D04 reference-fit uncertainty or production solver receipts",
        ],
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")

    p = 0.05
    symmetric = expected_risks(np.full(3, p))
    trainable_only = expected_risks(np.asarray((p, 0.0, 0.0)))
    comparator_only = expected_risks(np.asarray((0.0, p, p)))
    readme = f"""# D06 assigned-arm failure penalty: exact sensitivity (v1)

**Analytic sensitivity only.** The successful bounded-risk means `{SUCCESS_RISKS[0]:.3f}`, `{SUCCESS_RISKS[1]:.3f}`, `{SUCCESS_RISKS[2]:.3f}` and the failure probabilities are illustrative inputs, not measured optimizer outcomes. No gauge, reference-fit, sealed split, GPU, or QUDA input was read; `physical_score=null`.

## Proposed failure semantics

For one preassigned arm inside one paired macroblock:

```text
accepted finite optimizer result -> average the four clipped q losses
optimizer failure / budget exhaustion / failed residual gate -> set that arm's whole block score to 1; retain the block
missing or corrupt input without a declared optimizer failure -> do not impute; the run abstains
```

The first failure rule makes algorithm reliability part of the arm's risk estimand. If successful runs have mean block risk `mu_success` and the assigned-arm failure probability is `p_fail`, the exact mixture mean is `mu = p_fail + (1-p_fail)*mu_success`. Keeping failed assignments at score 1 preserves `[0,1]` support; under independent paired macroblocks, the bounded-mean theorem still applies to the composite outcome. It does not turn correlated HMC blocks into independent observations.

![Exact risk and ratio sensitivity to assigned-arm failures](failure_effect.png)

At the illustrative `p_fail=0.05`, expected risks in the `same_failure_rate_all_arms` case are `{symmetric[0]:.4f}`, `{symmetric[1]:.4f}`, `{symmetric[2]:.4f}` and the SLSQP/trainable-W and Chebyshev/trainable-W ratios are `{symmetric[1]/symmetric[0]:.4f}` and `{symmetric[2]/symmetric[0]:.4f}`. If only trainable-W fails at 5%, the ratios become `{trainable_only[1]/trainable_only[0]:.4f}` and `{trainable_only[2]/trainable_only[0]:.4f}`. If only the comparators fail at 5%, they become `{comparator_only[1]/comparator_only[0]:.4f}` and `{comparator_only[2]/comparator_only[0]:.4f}`. The direction and size of a comparison can therefore change substantially when failures are dropped; the proposed score rule prevents that exclusion.

The 303 rows in `failure_sensitivity.csv` are exact formula scenarios (`planned/valid/failed/abstained=303/303/0/0`), not Monte Carlo panels. The method helper now accepts a boolean arm-failure mask, permits absent `q` only for a declared failed arm, assigns that arm score 1, and reports failures separately while keeping all blocks scored. Unknown, malformed, or nonfinite inputs outside the declared failure mask are rejected.

## Next gate

The rule is a proposed method default, not independent review or production authorization. Before evaluator use, freeze what constitutes an optimizer failure (including residual and hard-timeout receipts), ensure one q-to-score map per arm, and verify that missing measurement/reference inputs abstain rather than become solver failures. Then calibrate the complete pipeline with its actual failure process and dependence contract. Current D04 remains unchanged; same-budget SLSQP and degree-4 Chebyshev remain both primary comparators.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    files = sorted(path for path in output.iterdir() if path.is_file())
    hashes = {path.name: _sha256(path) for path in files}
    (output / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
