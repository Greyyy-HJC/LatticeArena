"""Synthetic sensitivity grid for the bounded block-risk planning rule.

This module consumes no physical scores. Grid rows are hypothetical planning
scenarios; ``CONDITIONALLY_FEASIBLE`` means only that the required normalized
valid-block mean is strictly below its mathematical maximum of one.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from tasks.pion_projection.dependency_graph_bound import required_valid_mean


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/pion_projection/bounded_block_risk_stress_20261123_v2"
ALPHA_TOTAL = 0.05
FAMILY = 2
BLOCKS_GRID = (20, 100, 400)
CHI_GRID = (1, 2, 3, 5)
ALPHA_ANCHOR_GRID = (0.0, 0.01, 0.02)
RELATIVE_ANCHOR_GRID = (0.0, 0.025, 0.05, 0.1)
FAILURE_RATE_GRID = (0.0, 0.05, 0.1, 0.25)
THRESHOLD_GRID = (0.0, 0.05, 0.1)


def planned_failure_count(blocks: int, requested_failure_fraction: float) -> int:
    """Round a hypothetical failure fraction upward to whole planned blocks."""
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 1:
        raise ValueError("blocks must be a positive integer")
    if (not math.isfinite(requested_failure_fraction)
            or not 0 <= requested_failure_fraction < 1):
        raise ValueError("requested_failure_fraction must be finite and in [0, 1)")
    return min(blocks, math.ceil(requested_failure_fraction * blocks))


def classify_scenario(*, blocks: int, chromatic_number: int,
                      alpha_total: float, alpha_anchor: float,
                      relative_anchor_bound: float,
                      requested_failure_fraction: float,
                      normalized_threshold: float,
                      family: int = FAMILY) -> dict:
    """Return one hypothetical grid row, preserving block and scenario counts."""
    failed = planned_failure_count(blocks, requested_failure_fraction)
    valid = blocks - failed
    row = {
        "planned_blocks": blocks,
        "valid_blocks": valid,
        "failed_blocks": failed,
        "abstained_blocks": 0,
        "chromatic_number_upper_bound": chromatic_number,
        "family": family,
        "alpha_total": alpha_total,
        "alpha_anchor": alpha_anchor,
        "alpha_s": alpha_total - alpha_anchor,
        "relative_anchor_bound_B_over_R": relative_anchor_bound,
        "requested_failure_fraction": requested_failure_fraction,
        "coded_failure_fraction": failed / blocks,
        "normalized_threshold_delta_over_R2": normalized_threshold,
        "required_valid_mean_boundary": None,
        "status": None,
        "status_reason": None,
    }
    if valid == 0:
        row["status"] = "ABSTAIN_NO_VALID_BLOCKS"
        row["status_reason"] = "all planned blocks are coded as failures"
        return row

    boundary = required_valid_mean(
        blocks=blocks,
        chromatic_number=chromatic_number,
        family=family,
        alpha_s=row["alpha_s"],
        alpha_anchor=alpha_anchor,
        alpha_total=alpha_total,
        relative_anchor_bound=relative_anchor_bound,
        normalized_threshold=normalized_threshold,
        failure_fraction=failed / blocks,
    )
    row["required_valid_mean_boundary"] = boundary
    # The certification inequality is strict. At a boundary of exactly 1,
    # no admissible normalized score can exceed it, so abstention is required.
    if boundary >= 1.0:
        row["status"] = "ABSTAIN_UNATTAINABLE_MEAN"
        row["status_reason"] = (
            "required mean is at or above the strict score maximum"
        )
    else:
        row["status"] = "CONDITIONALLY_FEASIBLE"
        row["status_reason"] = (
            "mathematically attainable only if every stated assumption holds"
        )
    return row


def make_grid() -> list[dict]:
    """Enumerate the frozen synthetic planning grid; no scores are sampled."""
    rows = []
    for n in BLOCKS_GRID:
        for chi in CHI_GRID:
            for alpha_anchor in ALPHA_ANCHOR_GRID:
                for relative_anchor in RELATIVE_ANCHOR_GRID:
                    for failure_fraction in FAILURE_RATE_GRID:
                        for threshold in THRESHOLD_GRID:
                            rows.append(classify_scenario(
                                blocks=n,
                                chromatic_number=chi,
                                alpha_total=ALPHA_TOTAL,
                                alpha_anchor=alpha_anchor,
                                relative_anchor_bound=relative_anchor,
                                requested_failure_fraction=failure_fraction,
                                normalized_threshold=threshold,
                            ))
    return rows


def denominator_summary(rows: Iterable[dict]) -> list[dict]:
    """Summarize scenario and hypothetical-block denominators separately."""
    items = list(rows)
    valid_statuses = {"CONDITIONALLY_FEASIBLE"}
    abstained_statuses = {
        "ABSTAIN_NO_VALID_BLOCKS", "ABSTAIN_UNATTAINABLE_MEAN",
    }
    failed_statuses = {"FAILED_INVALID_INPUT", "FAILED_NUMERICAL"}
    planned_scenarios = len(items)
    valid_scenarios = sum(row["status"] in valid_statuses for row in items)
    failed_scenarios = sum(row["status"] in failed_statuses for row in items)
    abstained_scenarios = sum(row["status"] in abstained_statuses for row in items)
    scenario_rows = [{
        "unit": "synthetic planning scenario",
        "planned": planned_scenarios,
        "valid": valid_scenarios,
        "failed": failed_scenarios,
        "abstained": abstained_scenarios,
        "interpretation": (
            "Valid means the bound is mathematically attainable under stated "
            "assumptions; it is not an observed or certified physical result."
        ),
    }]
    block_rows = [{
        "unit": "hypothetical blocks summed over grid rows",
        "planned": sum(row["planned_blocks"] for row in items),
        "valid": sum(row["valid_blocks"] for row in items),
        "failed": sum(row["failed_blocks"] for row in items),
        "abstained": sum(row["abstained_blocks"] for row in items),
        "interpretation": (
            "Scenario-weighted hypothetical denominator; not a measurement "
            "sample and not a count of independent blocks."
        ),
    }]
    return scenario_rows + block_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    failure_rates = FAILURE_RATE_GRID
    chis = CHI_GRID
    relative_bounds = (0.0, 0.05, 0.1)
    alpha_anchor = 0.01
    threshold = 0.05
    fig, axes = plt.subplots(
        len(BLOCKS_GRID), len(relative_bounds),
        figsize=(11.5, 9.0), sharex=True, sharey=True,
        layout="constrained",
    )
    image = None
    for row_index, n in enumerate(BLOCKS_GRID):
        for col_index, relative_bound in enumerate(relative_bounds):
            matrix = np.empty((len(chis), len(failure_rates)))
            for i, chi in enumerate(chis):
                for j, requested_failure in enumerate(failure_rates):
                    row = classify_scenario(
                        blocks=n,
                        chromatic_number=chi,
                        alpha_total=ALPHA_TOTAL,
                        alpha_anchor=alpha_anchor,
                        relative_anchor_bound=relative_bound,
                        requested_failure_fraction=requested_failure,
                        normalized_threshold=threshold,
                    )
                    matrix[i, j] = row["required_valid_mean_boundary"]
            ax = axes[row_index, col_index]
            image = ax.imshow(matrix, origin="lower", aspect="auto",
                              vmin=0, vmax=1.5, cmap="RdYlGn_r")
            ax.set_xticks(range(len(failure_rates)),
                          [f"{x:.0%}" for x in failure_rates])
            ax.set_yticks(range(len(chis)), [str(x) for x in chis])
            ax.set_title(f"N={n}, B/R={relative_bound:g}")
            if col_index == 0:
                ax.set_ylabel("assumed graph color bound chi")
            if row_index == len(BLOCKS_GRID) - 1:
                ax.set_xlabel("requested failure fraction")
            if matrix.min() <= 1.0 <= matrix.max():
                ax.contour(matrix, levels=[1.0], colors="black",
                           linewidths=1.1)
    fig.colorbar(image, ax=axes, shrink=0.84,
                 label="required valid mean / R^2 (>=1: ABSTAIN)")
    fig.suptitle(
        "Synthetic planning sensitivity; alpha_total=0.05, alpha_A=0.01, "
        "alpha_s=0.04, delta*/R^2=0.05\n"
        "No physical scores, dependence graph, or anchor bound are estimated"
    )
    fig.savefig(out / "failure_dependence_anchor_surface.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    alpha_anchor_values = (0.0, 0.005, 0.01, 0.02, 0.03, 0.04)
    colors = ("#D55E00", "#0072B2", "#009E73")
    for n, color in zip(BLOCKS_GRID, colors):
        values = []
        for alpha_a in alpha_anchor_values:
            values.append(required_valid_mean(
                blocks=n,
                chromatic_number=2,
                family=FAMILY,
                alpha_s=ALPHA_TOTAL - alpha_a,
                alpha_anchor=alpha_a,
                alpha_total=ALPHA_TOTAL,
                relative_anchor_bound=0.05,
                normalized_threshold=0.05,
                failure_fraction=0.1,
            ))
        ax.plot([x / ALPHA_TOTAL for x in alpha_anchor_values], values,
                marker="o", color=color, label=f"N={n}")
    ax.axhline(1.0, color="black", ls="--", lw=1.1,
               label="strict maximum: ABSTAIN at or above")
    ax.set(xlabel="anchor share of total alpha budget",
           ylabel="required valid mean / R^2",
           title="Tradeoff at chi=2, B/R=0.05, failure=10%, delta*/R²=0.05")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(out / "alpha_allocation_sensitivity.png", dpi=180)
    plt.close(fig)


def build(output: str | Path = DEFAULT_OUT) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows = make_grid()
    summary_rows = denominator_summary(rows)
    _write_csv(out / "sensitivity_grid.csv", rows)
    _write_csv(out / "denominator_summary.csv", summary_rows)
    _write_plots(out)
    status_counts = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    summary = {
        "schema": "pion-bounded-block-risk-stress-v1",
        "status": "SYNTHETIC_ANALYTIC_SENSITIVITY_ONLY",
        "physical_score": None,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_access": "No physical scores, correlators, anchor, sealed split, GPU, or QUDA data were read.",
        "strict_feasibility_rule": "required_valid_mean_boundary < 1; equality is ABSTAIN because the certification inequality is strict",
        "grid": {
            "blocks": list(BLOCKS_GRID),
            "chromatic_number_upper_bounds": list(CHI_GRID),
            "alpha_total": ALPHA_TOTAL,
            "alpha_anchor": list(ALPHA_ANCHOR_GRID),
            "alpha_s": [ALPHA_TOTAL - value for value in ALPHA_ANCHOR_GRID],
            "relative_anchor_bound_B_over_R": list(RELATIVE_ANCHOR_GRID),
            "requested_failure_fraction": list(FAILURE_RATE_GRID),
            "normalized_threshold_delta_over_R2": list(THRESHOLD_GRID),
            "family": FAMILY,
        },
        "scenario_count": len(rows),
        "status_counts": status_counts,
        "denominator_summary": summary_rows,
        "files": [
            "README.md", "sensitivity_grid.csv", "denominator_summary.csv",
            "failure_dependence_anchor_surface.png",
            "alpha_allocation_sensitivity.png", "summary.json",
            "artifact_hashes.json",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    readme = [
        "# Bounded block-risk synthetic calibration stress grid",
        "",
        "**SYNTHETIC ANALYTIC SENSITIVITY ONLY. `physical_score=null`.** This report evaluates formulas on a fixed hypothetical parameter grid. It reads no physical score, correlator, anchor, sealed split, GPU, or QUDA input. It neither estimates the Pion effect nor certifies a dependence or anchor bound.",
        "",
        "## Estimand and rule",
        "",
        "The planning target remains the bounded finite-plan block-risk from the parent calibration report. For `N` planned blocks, `F=2`, a justified dependency-graph color bound `chi`, score error budget `alpha_s`, relative anchor bound `B/R`, normalized target `delta_star/R^2`, and planned failure fraction `phi`, the required valid-block mean boundary is",
        "",
        "```text",
        "m_required = (delta_star/R^2 + sqrt(2*chi*log(F/alpha_s)/N) + 2*B/R + phi)/(1-phi)",
        "```",
        "",
        "Each hypothetical failed block is retained in the planned denominator and coded as `-1`. The strict certificate requires the observed valid mean to exceed `m_required`; therefore `m_required >= 1` is `ABSTAIN`, including equality, because a score in `[-1,1]` cannot exceed one. `CONDITIONALLY_FEASIBLE` means only that the inequality is mathematically attainable if every stated assumption holds. It does not mean a result was observed, calibrated, or certified.",
        "",
        "## Grid and denominator accounting",
        "",
        "The 1,728 rows cross `N={20,100,400}`, `chi={1,2,3,5}`, `alpha_A={0,0.01,0.02}` with `alpha_s=0.05-alpha_A`, `B/R={0,0.025,0.05,0.1}`, requested failure fractions `{0,0.05,0.1,0.25}`, and `delta_star/R^2={0,0.05,0.1}`. Failed-block counts are rounded upward to whole blocks; the actual coded fraction is included per row. All values are illustrative, not frozen research choices.",
        "",
        "`denominator_summary.csv` separates two units. The scenario row reports planned, mathematically feasible, computation-failed, and abstained grid configurations. The block row sums hypothetical planned/valid/failed blocks over scenarios and is scenario-weighted; it is not a measurement denominator or an independent-block count. Grid construction has zero computation failures, and no block is itself marked abstained because no experiment ran.",
        "",
        "## Figures",
        "",
        "![Failure, dependence, and anchor sensitivity](failure_dependence_anchor_surface.png)",
        "",
        "Each panel fixes `alpha_total=0.05`, `alpha_A=0.01`, `alpha_s=0.04`, and `delta_star/R^2=0.05`, then varies planned `N`, assumed `chi`, requested failure fraction, and `B/R`. The black contour marks the strict feasibility boundary at one; cells at or above it are abstentions. `chi` is only an axis value and has not been inferred from receipts.",
        "",
        "![Alpha allocation sensitivity](alpha_allocation_sensitivity.png)",
        "",
        "This panel holds `chi=2`, `B/R=0.05`, failure fraction `0.1`, and `delta_star/R^2=0.05` fixed while reallocating the illustrative total alpha budget. Increasing the anchor share reduces `alpha_s`, enlarging the concentration penalty. This is a planning tradeoff, not a recommendation for the physical budget.",
        "",
        "## Denominators and interpretation",
        "",
        "The grid has no observed sample outcomes: valid/failed block counts are only hypothetical consequences of each requested failure rate. `ABSTAIN` is counted at the scenario level when a requested mean is unattainable, while failed blocks remain in the formula as worst-case scores. These two distinct denominators are not interchangeable.",
        "The underlying receipt audit still supports only 20 reproducible length-5 candidate blocks among 100 development records; it does not establish block independence. There is no inspected E-anchor correlator receipt or common-bias coverage bound. Thus no plotted cell is a Pion certificate and the project remains `NOT_READY/UNCERTIFIED`.",
        "",
        "## Rebuild",
        "",
        "Use a new output directory; the builder refuses to overwrite an existing path:",
        "",
        "```text",
        "python -B -m tasks.pion_projection.bounded_block_risk_stress --output NEW_DIRECTORY",
        "```",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n")
    artifact_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (out / "README.md", out / "sensitivity_grid.csv",
                     out / "denominator_summary.csv",
                     out / "failure_dependence_anchor_surface.png",
                     out / "alpha_allocation_sensitivity.png",
                     out / "summary.json")
    }
    (out / "artifact_hashes.json").write_text(
        json.dumps(artifact_hashes, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = build(args.output)
    print(json.dumps({
        "status": summary["status"],
        "scenario_count": summary["scenario_count"],
        "status_counts": summary["status_counts"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
