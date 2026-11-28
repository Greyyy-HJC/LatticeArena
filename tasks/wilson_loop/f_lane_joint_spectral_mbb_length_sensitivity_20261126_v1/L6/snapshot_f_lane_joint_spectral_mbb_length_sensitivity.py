"""Run the pre-registered L={6,8,10} synthetic Wilson MBB sensitivity scan.

The L=8 B=40 report is reused as the baseline; this runner only creates L=6
and L=10 reports. All three lengths use the same deterministic holdout panels
and bootstrap seed keys. This is descriptive CPU evidence, not calibration or
physical Wilson evidence.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tasks.wilson_loop import f_lane_joint_spectral_gls_diagnostic as joint
from tasks.wilson_loop import f_lane_joint_spectral_mbb_diagnostic as mbb
from tasks.wilson_loop import f_lane_joint_spectral_stress_diagnostic as stress


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_BASE = TASK_DIR / "f_lane_joint_spectral_mbb_20261126_v1"
DEFAULT_OUT = TASK_DIR / "f_lane_joint_spectral_mbb_length_sensitivity_20261126_v1"
LENGTHS = (6, 8, 10)
NEW_LENGTHS = (6, 10)
OUTER_PANELS_PER_SCENARIO = 40
BOOTSTRAP_REPLICATES = 40
Q_CELLS_PER_SCENARIO = len(joint.Q_OPERATORS) * len(joint.Q_TIMES)
WRAPPER_NAME = Path(__file__).name
COLORS = {6: "#0072B2", 8: "#D55E00", 10: "#009E73"}
SCENARIO_LABELS = {
    "matched_3state_independent": "matched",
    "omitted_4th_state": "4th omitted",
    "near_degenerate_first_gap": "near gap",
    "small_ground_overlap_operator_2": "small Z0",
    "ar1_block_rho_0p75": "AR(1) .75",
    "student_t5_independent": "Student-t5",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _verified_report(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = root / "artifact_hashes.json"
    protocol_path = root / "protocol.json"
    summary_path = root / "summary.json"
    for path in (manifest_path, protocol_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"incomplete MBB report: {path}")
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest.items():
        artifact = root / name
        if not artifact.is_file() or _sha256(artifact) != digest:
            raise ValueError(f"MBB report artifact hash mismatch: {name}")
    return json.loads(protocol_path.read_text()), json.loads(summary_path.read_text())


def _validate_baseline(base: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    protocol, summary = _verified_report(base)
    if (protocol.get("schema") != "wilson-f-joint-spectral-mbb-v1"
            or protocol.get("block_length") != 8
            or protocol.get("bootstrap_replicates_per_outer_panel") != BOOTSTRAP_REPLICATES
            or protocol.get("scenarios") != list(stress.SCENARIOS)
            or summary.get("physical_score") is not None):
        raise ValueError("L=8 input is not the expected synthetic B=40 baseline")
    denominators = summary["denominators"]
    expected_panels = len(stress.SCENARIOS) * OUTER_PANELS_PER_SCENARIO
    expected_cells = expected_panels * Q_CELLS_PER_SCENARIO
    if (denominators["outer_holdout_panels_planned"] != expected_panels
            or denominators["bootstrap_refits_planned"] != expected_panels * BOOTSTRAP_REPLICATES
            or denominators["moving_block_intervals_planned"] != expected_cells):
        raise ValueError("L=8 baseline denominators do not match the frozen scan")
    rows = _read_csv(base / "bootstrap_replicates.csv")
    if len(rows) != expected_panels * BOOTSTRAP_REPLICATES:
        raise ValueError("L=8 bootstrap row count does not match its denominator")
    return protocol, summary, rows


def _render_child_readme(root: Path, summary: dict[str, Any], length: int) -> str:
    den = summary["denominators"]
    comparison = [row for row in summary["comparison_rows"]
                  if row["method"] == "moving_block_bootstrap"]
    rows = [
        "# Wilson F-lane MBB length-sensitivity cell",
        "",
        "**Synthetic CPU diagnostic only.** This report reuses the hash-verified six-scenario stress bundle. It reads no Wilson measurements, sealed split, GPU, or QUDA data. `physical_score=null`; F remains `GROUND_STATE_FORCE_UNCERTIFIED`.",
        "",
        f"This is the pre-registered `L={length}` cell in the fixed `L={{6,8,10}}` scan. The `L=8` baseline was inspected before this scan; this child is not an independent confirmation and cannot select a preferred length. All cells use the same deterministic holdout panel seeds, 40 calibration panels and 40 holdout panels per scenario, `B=40`, and upper order-statistic rank 40. Bootstrap seed keys are shared across lengths; the sampled circular chunk layout changes with L.",
        "",
        "## Results",
        "",
        "![Pointwise interval coverage by method](method_coverage.png)",
        "",
        f"Planned/valid/failed/abstained: holdout panels `{den['outer_holdout_panels_planned']}/{den['outer_holdout_panels_valid']}/{den['outer_holdout_panels_failed']}/{den['outer_holdout_panels_abstained']}`; bootstrap refits `{den['bootstrap_refits_planned']}/{den['bootstrap_refits_valid']}/{den['bootstrap_refits_failed']}/{den['bootstrap_refits_abstained']}`; moving-block intervals `{den['moving_block_intervals_planned']}/{den['moving_block_intervals_valid']}/{den['moving_block_intervals_failed']}/{den['moving_block_intervals_abstained']}`. Interval coverage is conditional on valid intervals; the table retains the full denominator.",
        "",
        "| Scenario | Operator | T | Planned/valid/failed/abstained | Coverage (95% Wilson MC interval) | Median upper / truth |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        coverage = "NA" if row["coverage"] is None else f"{row['coverage']:.3f}"
        low = "NA" if row["mc95_low"] is None else f"{row['mc95_low']:.3f}"
        high = "NA" if row["mc95_high"] is None else f"{row['mc95_high']:.3f}"
        ratio = ("NA" if row["median_upper_over_true"] is None
                 else f"{row['median_upper_over_true']:.3f}")
        rows.append(
            f"| `{row['scenario']}` | {row['operator_index']} | {row['time']} | "
            f"{row['planned']}/{row['valid']}/{row['failed']}/{row['abstained']} | "
            f"{coverage} [{low}, {high}] | {ratio} |"
        )
    rows.extend([
        "",
        "## Interpretation and limits",
        "",
        f"This cell varies only the circular block length around the previously used `L=8`. It uses the same already-inspected synthetic holdout panels. Any coverage change is descriptive sensitivity to the resampling scheme under this generator; it is not a new coverage guarantee, not a tuning target, and not evidence about HMC-chain correlation. Failed/missing bootstrap q estimates remain `+infinity` when forming the upper order statistic, and all planned intervals remain in the denominator.",
        "",
        "No result from this report may enter the q reader or force gate. `L=6`, `L=8`, and `L=10` remain diagnostic-only regardless of which cell looks best. The pre-registered scan stops after these three values; further block-length or B tuning on this generator/holdout is out of scope. Real F-lane evidence still requires authorized raw cross-correlator snapshots, independently reviewed spectrum/overlap, finite-`N_t` thermal `h`, mixed-sector `x`, paired force/reference uncertainty, contamination bounds, and independent review.",
        "",
        "Text flow: `verify stress-bundle hashes -> regenerate the fixed holdout panels from shared seeds -> circular MBB with this pre-fixed L and B=40 -> refit joint 3-state GLS -> retain failed q draws as +infinity -> compare pointwise coverage, Wilson MC intervals, width, and abstention -> certification abstains`.",
        "",
        f"Reproduce this child through `PYTHONPATH=. python -B -m tasks.wilson_loop.f_lane_joint_spectral_mbb_length_sensitivity --output tasks/wilson_loop/f_lane_joint_spectral_mbb_length_sensitivity_20261126_v1`. Source and artifact hashes are recorded in `protocol.json`, `summary.json`, and `artifact_hashes.json`.",
    ])
    return "\n".join(rows) + "\n"


def _finalize_child(root: Path, length: int) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads((root / "protocol.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if (protocol.get("block_length") != length
            or protocol.get("bootstrap_replicates_per_outer_panel") != BOOTSTRAP_REPLICATES):
        raise ValueError(f"generated child did not match fixed L={length}, B=40 protocol")
    wrapper_hash = _sha256(Path(__file__))
    protocol["comparison_methods"]["moving_block_bootstrap"] = (
        f"empirical one-sided percentile of bootstrap refit q estimates using circular block length {length}"
    )
    protocol["length_sensitivity"] = {
        "fixed_lengths": list(LENGTHS),
        "this_length": length,
        "baseline_length": 8,
        "baseline_was_examined_before_scan": True,
        "holdout_panels_reused_across_lengths": True,
        "outer_and_bootstrap_seed_keys_shared": True,
        "independent_confirmation": False,
        "selection_or_gate_use": False,
    }
    protocol["source_sha256"][WRAPPER_NAME] = wrapper_hash
    wrapper_snapshot = f"snapshot_{WRAPPER_NAME}"
    shutil.copyfile(Path(__file__), root / wrapper_snapshot)
    if wrapper_snapshot not in summary["files"]:
        summary["files"].append(wrapper_snapshot)
    summary["source_sha256"][WRAPPER_NAME] = wrapper_hash
    summary["length_sensitivity"] = protocol["length_sensitivity"]
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (root / "README.md").write_text(_render_child_readme(root, summary, length))
    hashes = {name: _sha256(root / name) for name in summary["files"]}
    (root / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, allow_nan=False) + "\n")
    return protocol, summary


def _verify_shared_seed_keys(reports: dict[int, list[dict[str, str]]]) -> int:
    keyed: dict[int, dict[tuple[str, int, int], tuple[int, int]]] = {}
    for length, rows in reports.items():
        per_length: dict[tuple[str, int, int], tuple[int, int]] = {}
        for row in rows:
            key = (row["scenario"], int(row["outer_replicate"]),
                   int(row["bootstrap_replicate"]))
            if key in per_length:
                raise ValueError(f"duplicate bootstrap key for L={length}: {key}")
            per_length[key] = (int(row["outer_seed"]), int(row["bootstrap_seed"]))
        keyed[length] = per_length
    baseline = keyed[8]
    expected_count = len(stress.SCENARIOS) * OUTER_PANELS_PER_SCENARIO * BOOTSTRAP_REPLICATES
    if len(baseline) != expected_count:
        raise ValueError("baseline seed-key count does not match the fixed scan")
    for length in LENGTHS:
        if keyed[length].keys() != baseline.keys():
            raise ValueError(f"holdout/bootstrap identifiers differ at L={length}")
        if keyed[length] != baseline:
            raise ValueError(f"common random seed keys differ at L={length}")
    return expected_count


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _collect_rows(reports: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_cell_l8 = {
        (row["scenario"], int(row["operator_index"]), int(row["time"])): row
        for row in reports[8]
    }
    output = []
    for length in LENGTHS:
        rows = [row for row in reports[length] if row["method"] == "moving_block_bootstrap"]
        if len(rows) != len(stress.SCENARIOS) * Q_CELLS_PER_SCENARIO:
            raise ValueError(f"expected 24 MBB comparison rows at L={length}, got {len(rows)}")
        for row in rows:
            key = (row["scenario"], int(row["operator_index"]), int(row["time"]))
            baseline = by_cell_l8[key]
            coverage = _number(row["coverage"])
            baseline_coverage = _number(baseline["coverage"])
            planned, valid = int(row["planned"]), int(row["valid"])
            output.append({
                "block_length": length,
                "scenario": row["scenario"],
                "operator_index": int(row["operator_index"]),
                "time": int(row["time"]),
                "method": row["method"],
                "planned": planned,
                "valid": valid,
                "failed": int(row["failed"]),
                "abstained": int(row["abstained"]),
                "yield": valid / planned if planned else None,
                "coverage": coverage,
                "mc95_low": _number(row["mc95_low"]),
                "mc95_high": _number(row["mc95_high"]),
                "coverage_delta_vs_L8": (coverage - baseline_coverage
                                          if coverage is not None and baseline_coverage is not None else None),
                "median_upper_over_true": _number(row["median_upper_over_true"]),
                "bootstrap_replicates": int(row["bootstrap_replicates"]),
                "baseline_L8_valid": int(baseline["valid"]),
                "baseline_L8_abstained": int(baseline["abstained"]),
            })
    return output


def _plot(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
    scenarios = list(stress.SCENARIOS)
    positions = np.arange(len(scenarios), dtype=float)
    offsets = {6: -0.22, 8: 0.0, 10: 0.22}
    for ax, (operator_index, time) in zip(axes.flat,
                                           ((op, time) for op in joint.Q_OPERATORS
                                            for time in joint.Q_TIMES)):
        for length in LENGTHS:
            subset = {(row["scenario"]): row for row in rows
                      if row["block_length"] == length
                      and row["operator_index"] == operator_index
                      and row["time"] == time}
            for index, scenario in enumerate(scenarios):
                row = subset[scenario]
                x = positions[index] + offsets[length]
                coverage, low, high = row["coverage"], row["mc95_low"], row["mc95_high"]
                if coverage is None or low is None or high is None:
                    ax.scatter(x, 0.5, marker="x", color=COLORS[length], s=65,
                               label=f"L={length} (no valid interval)" if index == 0 else None)
                else:
                    ax.errorbar(x, coverage,
                                yerr=[[coverage - low], [high - coverage]],
                                fmt="o", color=COLORS[length], capsize=3,
                                label=f"L={length}" if index == 0 else None)
        ax.axhline(0.975, color="black", linestyle="--", linewidth=1,
                   label="nominal 97.5%" if operator_index == joint.Q_OPERATORS[0]
                   and time == joint.Q_TIMES[0] else None)
        ax.set_xticks(positions, [SCENARIO_LABELS.get(name, name) for name in scenarios],
                      rotation=18, ha="right")
        ax.set_ylim(0.35, 1.03)
        ax.set_ylabel("conditional pointwise coverage (95% Wilson MC interval)")
        ax.set_title(f"bare operator {operator_index}, T={time}")
        ax.grid(alpha=0.2)
    fig.suptitle("Pre-fixed MBB block-length sensitivity on shared synthetic holdouts")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _cell_table(rows: list[dict[str, Any]]) -> list[str]:
    table = [
        "| Scenario | L | Planned/valid/failed/abstained | Yield | Cell coverages (op0 T5/T6; op2 T5/T6) |",
        "|---|---:|---:|---:|---|",
    ]
    for scenario in stress.SCENARIOS:
        for length in LENGTHS:
            subset = [row for row in rows if row["scenario"] == scenario
                      and row["block_length"] == length]
            planned = sum(row["planned"] for row in subset)
            valid = sum(row["valid"] for row in subset)
            failed = sum(row["failed"] for row in subset)
            abstained = sum(row["abstained"] for row in subset)
            coverage_parts = []
            for row in sorted(subset, key=lambda item: (item["operator_index"], item["time"])):
                label = f"op{row['operator_index']} T{row['time']}"
                if row["coverage"] is None:
                    value = "NA (0 valid)"
                else:
                    value = (f"{row['coverage']:.3f} [{row['mc95_low']:.3f}, "
                             f"{row['mc95_high']:.3f}], n={row['valid']}/40")
                coverage_parts.append(f"{label}: {value}")
            table.append(
                f"| `{scenario}` | {length} | {planned}/{valid}/{failed}/{abstained} | "
                f"{valid / planned:.3f} | " + "; ".join(coverage_parts) + " |"
            )
    return table


def _top_readme(summary: dict[str, Any], rows: list[dict[str, Any]], seed_keys: int) -> str:
    per_length = summary["per_length_denominators"]
    total = summary["combined_interval_denominators"]
    lines = [
        "# Wilson F-lane MBB block-length sensitivity",
        "",
        "**Synthetic CPU diagnostic only.** This fixed scan reuses the same hash-verified six-scenario stress bundle and the same 240 deterministic holdout panels. It reads no Wilson measurements, sealed split, GPU, or QUDA data. `physical_score=null`; F remains `GROUND_STATE_FORCE_UNCERTIFIED`.",
        "",
        "## Frozen design",
        "",
        "The length family was frozen before execution as `L={6,8,10}`. The already-reviewed B=40 `L=8` report is reused; only `L=6` and `L=10` were newly run. Each length uses the same first 40 calibration and last 40 holdout panels per scenario, `B=40`, rank 40, the same outer panel seeds, and the same bootstrap seed keys. The circular chunk layouts differ by L, as intended. This is a descriptive sensitivity scan on the same holdout, not an independent confirmation or a new calibration.",
        "",
        f"The seed-key audit matched `{seed_keys}/{seed_keys}` `(scenario, holdout panel, bootstrap index)` keys across all three lengths. The figure gives per-cell conditional coverage and 95% Wilson Monte Carlo intervals. An `x` means zero valid intervals; its displayed y=0.5 position is only a plotting placeholder, not a 50% estimate. The dashed line is the nominal pointwise 97.5% target; no familywise `q/h/x` guarantee is tested.",
        "",
        "![Coverage by block length and scenario](coverage_by_L.png)",
        "",
        "## Full cell diagnostics",
        "",
        *(_cell_table(rows)),
        "",
        "Counts use planned/valid/failed/abstained interval denominators. Coverage and its MC interval are conditional on valid intervals, while yield exposes how many planned intervals remain reportable. `length_sensitivity.csv` contains every individual q cell, including coverage differences from L=8 and median upper/truth.",
        "",
        "## Denominators",
        "",
        "| L | Planned/valid/failed/abstained intervals |",
        "|---:|---:|",
    ]
    for length in LENGTHS:
        d = per_length[str(length)]
        lines.append(f"| {length} | {d['planned']}/{d['valid']}/{d['failed']}/{d['abstained']} |")
    lines.extend([
        f"| Combined three-length sensitivity entries | {total['planned']}/{total['valid']}/{total['failed']}/{total['abstained']} |",
        "",
        "The combined row counts repeated analyses of shared holdout intervals; it does not represent 720 independent observations. The 240 outer holdout panels are valid once, not three times. Physical measurements remain `0/0/0/0`; certification remains `1/0/0/1` with abstention.",
        "",
        "## Interpretation and stopping rule",
        "",
        "Any change across L describes the sensitivity of this MBB procedure under the chosen synthetic generator. It does not establish the true HMC correlation length. Failed or missing bootstrap q estimates remain `+infinity` in the upper order statistic, and their associated intervals stay in the planned denominator. A higher conditional coverage at one L cannot compensate for poor yield or justify selecting that L after viewing the holdout.",
        "",
        "The scan stops at the pre-fixed `L={6,8,10}` family. No length is promoted, no further B/L tuning is performed on this generator/holdout, and none of these bounds enters the q reader or force gate. Physical F-lane work still needs authorized raw cross-correlator snapshots, independently reviewed spectrum/overlap, finite-`N_t` thermal `h`, mixed-sector `x`, paired force/reference uncertainty, contamination bounds, and independent review. Keep `NOT_READY/UNCERTIFIED`, `GROUND_STATE_FORCE_UNCERTIFIED`, and `physical_score=null`.",
        "",
        "Text flow: `verify existing L8 B40 report and fixed stress bundle -> run only L6 and L10 using the same deterministic holdouts -> compare all 24 q cells per length -> report MC intervals and P/V/F/A counts -> stop the synthetic scan -> physical certification abstains`.",
        "",
        "The child directories contain the full per-length bootstrap draws, interval tables, plots, source snapshots, and hash manifests. The top-level `protocol.json`, `summary.json`, and `artifact_hashes.json` bind the combined analysis and its inputs.",
    ])
    return "\n".join(lines) + "\n"


def run(output: str | Path = DEFAULT_OUT, *, base: str | Path = DEFAULT_BASE,
        report_progress: bool = True) -> dict[str, Any]:
    """Run the frozen sensitivity package; never overwrites an existing report."""
    base_path = Path(base)
    base_protocol, base_summary, base_bootstrap_rows = _validate_baseline(base_path)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "run_status.json").write_text(json.dumps({
        "status": "RUNNING", "fixed_lengths": list(LENGTHS),
        "completed_new_lengths": [], "next_length": NEW_LENGTHS[0],
    }, indent=2) + "\n")

    bootstrap_rows: dict[int, list[dict[str, str]]] = {8: base_bootstrap_rows}
    comparison_rows: dict[int, list[dict[str, Any]]] = {
        8: base_summary["comparison_rows"],
    }
    per_length_summaries: dict[int, dict[str, Any]] = {8: base_summary}
    try:
        for length in NEW_LENGTHS:
            child = out / f"L{length}"
            result = mbb.run(
                child, outer_panels_per_scenario=OUTER_PANELS_PER_SCENARIO,
                bootstrap_replicates=BOOTSTRAP_REPLICATES, block_length=length,
                scenarios=stress.SCENARIOS, report_progress=report_progress,
            )
            _finalize_child(child, length)
            bootstrap_rows[length] = _read_csv(child / "bootstrap_replicates.csv")
            comparison_rows[length] = result["comparison_rows"]
            per_length_summaries[length] = result
            (out / "run_status.json").write_text(json.dumps({
                "status": "RUNNING", "fixed_lengths": list(LENGTHS),
                "completed_new_lengths": [value for value in NEW_LENGTHS if value in bootstrap_rows],
                "next_length": next((value for value in NEW_LENGTHS if value not in bootstrap_rows), None),
            }, indent=2) + "\n")
            if report_progress:
                print(f"completed pre-fixed MBB block length L={length}", flush=True)

        seed_keys = _verify_shared_seed_keys(bootstrap_rows)
        rows = _collect_rows(comparison_rows)
        _write_csv(out / "length_sensitivity.csv", rows)
        _plot(rows, out / "coverage_by_L.png")

        source_hashes = {
            WRAPPER_NAME: _sha256(Path(__file__)),
            Path(mbb.__file__).name: _sha256(Path(mbb.__file__)),
            Path(stress.__file__).name: _sha256(Path(stress.__file__)),
            Path(joint.__file__).name: _sha256(Path(joint.__file__)),
        }
        shutil.copyfile(Path(__file__), out / f"snapshot_{WRAPPER_NAME}")
        per_length_denominators = {}
        for length in LENGTHS:
            d = per_length_summaries[length]["denominators"]
            per_length_denominators[str(length)] = {
                "planned": d["moving_block_intervals_planned"],
                "valid": d["moving_block_intervals_valid"],
                "failed": d["moving_block_intervals_failed"],
                "abstained": d["moving_block_intervals_abstained"],
                "bootstrap_refits_planned": d["bootstrap_refits_planned"],
                "bootstrap_refits_valid": d["bootstrap_refits_valid"],
                "bootstrap_refits_failed": d["bootstrap_refits_failed"],
            }
        combined = {
            key: sum(per_length_denominators[str(length)][key]
                     for length in LENGTHS)
            for key in ("planned", "valid", "failed", "abstained")
        }
        top_files = ["README.md", "protocol.json", "length_sensitivity.csv",
                     "coverage_by_L.png", "summary.json", f"snapshot_{WRAPPER_NAME}"]
        protocol = {
            "schema": "wilson-f-joint-spectral-mbb-length-sensitivity-v1",
            "classification": "pre-registered synthetic CPU descriptive sensitivity only",
            "physical_score": None,
            "fixed_lengths": list(LENGTHS),
            "newly_run_lengths": list(NEW_LENGTHS),
            "baseline_length": 8,
            "baseline_was_examined_before_scan": True,
            "same_stress_bundle_and_holdout_panels": True,
            "calibration_panel_replicates": "0:40",
            "holdout_panel_replicates": "40:80",
            "bootstrap_replicates_per_panel": BOOTSTRAP_REPLICATES,
            "upper_order_statistic_rank": 40,
            "failed_or_missing_q_draw_rule": "+infinity; retain planned interval in denominator",
            "seed_key_coupling": "same scenario/outer-panel/bootstrap-index seed keys across L; circular chunk layout varies with L",
            "shared_seed_key_count": seed_keys,
            "per_length_denominators": per_length_denominators,
            "combined_interval_denominators_reuse_same_holdout": combined,
            "baseline_b40_artifact_hashes": json.loads((base_path / "artifact_hashes.json").read_text()),
            "baseline_protocol_source_sha256": base_protocol["source_sha256"],
            "input_stress_artifact_hashes": json.loads((stress.DEFAULT_OUT / "artifact_hashes.json").read_text()),
            "source_sha256": source_hashes,
            "physical_measurements": {"planned": 0, "valid": 0, "failed": 0, "abstained": 0},
            "certification": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
            "physical_reader_or_gate_use": False,
            "next_action": "stop tuning L or B on this generator/holdout; await new independent calibration panels and authorized production receipts",
        }
        (out / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
        summary = {
            "schema": "wilson-f-joint-spectral-mbb-length-sensitivity-summary-v1",
            "classification": protocol["classification"],
            "physical_score": None,
            "F_lane_status": "GROUND_STATE_FORCE_UNCERTIFIED",
            "fixed_lengths": list(LENGTHS),
            "newly_run_lengths": list(NEW_LENGTHS),
            "shared_seed_key_count": seed_keys,
            "per_length_denominators": per_length_denominators,
            "combined_interval_denominators": combined,
            "physical_measurements": {"planned": 0, "valid": 0, "failed": 0, "abstained": 0},
            "certification": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
            "files": top_files,
            "source_sha256": source_hashes,
            "comparison_rows": rows,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        (out / "README.md").write_text(_top_readme(summary, rows, seed_keys))
        hashes = {name: _sha256(out / name) for name in top_files}
        (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, allow_nan=False) + "\n")
        (out / "run_status.json").write_text(json.dumps({
            "status": "COMPLETE", "fixed_lengths": list(LENGTHS),
            "completed_new_lengths": list(NEW_LENGTHS), "next_length": None,
            "seed_key_audit": f"{seed_keys}/{seed_keys}",
            "combined_interval_denominators": combined,
        }, indent=2) + "\n")
        return summary
    except Exception as exc:
        (out / "run_status.json").write_text(json.dumps({
            "status": "FAILED", "fixed_lengths": list(LENGTHS),
            "completed_new_lengths": [value for value in NEW_LENGTHS if value in bootstrap_rows],
            "error_type": type(exc).__name__, "error": str(exc),
        }, indent=2) + "\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = run(args.output, base=args.base, report_progress=not args.quiet)
    print(json.dumps({
        "output": str(args.output),
        "fixed_lengths": result["fixed_lengths"],
        "combined_interval_denominators": result["combined_interval_denominators"],
    }, indent=2))


if __name__ == "__main__":
    main()
