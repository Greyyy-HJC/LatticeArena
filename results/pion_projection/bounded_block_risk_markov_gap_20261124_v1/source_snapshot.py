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
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/pion_projection/bounded_block_risk_markov_gap_20261124_v1"
BLOCKS_GRID = (20, 50, 100, 200, 500)
GAP_GRID = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
FAMILY = 2
ALPHA_S = 0.05
VARIANCE_BOUND = 1.0
CENTERED_ABS_BOUND = 2.0


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
    if not math.isfinite(gamma_ps_lower) or not 0 < gamma_ps_lower <= 1:
        raise ValueError("gamma_ps_lower must be finite and in (0, 1]")
    if not math.isfinite(alpha_s) or not 0 < alpha_s < 1:
        raise ValueError("alpha_s must be finite and in (0, 1)")
    if (not math.isfinite(variance_bound) or variance_bound < 0
            or not math.isfinite(centered_abs_bound)
            or centered_abs_bound <= 0):
        raise ValueError("invalid Bernstein range or variance bound")

    log_tail = math.log(2 * family / alpha_s)
    gamma = gamma_ps_lower
    linear = 20 * centered_abs_bound * log_tail
    variance = 8 * log_tail * (blocks + 1 / gamma) * variance_bound
    sum_deviation = (linear + math.sqrt(linear * linear + 4 * gamma * variance)) / (2 * gamma)
    return sum_deviation / blocks


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out}")
    out.mkdir(parents=True)
    rows = make_grid()
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
    ]
    _write_csv(out / "denominator_summary.csv", scenario_denominators)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
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
