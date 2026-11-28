"""Extend the verified B=40 Wilson synthetic MBB pilot to B=80."""
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
DEFAULT_OUT = TASK_DIR / "f_lane_joint_spectral_mbb_b80_20261126_v1"
BASE_REPLICATES = 40
DEFAULT_TOTAL_REPLICATES = 80

Q_COLUMNS = {
    (0, 5): "q_op0_T5", (0, 6): "q_op0_T6",
    (2, 5): "q_op2_T5", (2, 6): "q_op2_T6",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_verified_b40(base: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = base / "artifact_hashes.json"
    protocol_path = base / "protocol.json"
    summary_path = base / "summary.json"
    for path in (manifest_path, protocol_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"verified B=40 report is incomplete: {path}")
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest.items():
        path = base / name
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"B=40 artifact hash mismatch: {name}")
    protocol = json.loads(protocol_path.read_text())
    summary = json.loads(summary_path.read_text())
    if (protocol.get("schema") != "wilson-f-joint-spectral-mbb-v1"
            or int(protocol.get("bootstrap_replicates_per_outer_panel", -1)) != BASE_REPLICATES
            or summary.get("physical_score") is not None):
        raise ValueError("input report is not the expected B=40 synthetic MBB pilot")
    expected_sources = {
        Path(mbb.__file__).name: _sha256(Path(mbb.__file__)),
        Path(stress.__file__).name: _sha256(Path(stress.__file__)),
        Path(joint.__file__).name: _sha256(Path(joint.__file__)),
    }
    if protocol.get("source_sha256") != expected_sources:
        raise ValueError("B=40 report source snapshot does not match current producer code")
    for filename in ("bootstrap_replicates.csv", "bootstrap_intervals.csv"):
        if filename not in manifest:
            raise ValueError(f"B=40 artifact manifest omits {filename}")
    with (base / "bootstrap_replicates.csv").open(newline="") as stream:
        bootstrap_rows = list(csv.DictReader(stream))
    with (base / "bootstrap_intervals.csv").open(newline="") as stream:
        interval_rows = list(csv.DictReader(stream))
    planned_refits = int(summary["denominators"]["bootstrap_refits_planned"])
    if len(bootstrap_rows) != planned_refits or planned_refits != 6 * 40 * BASE_REPLICATES:
        raise ValueError("B=40 refit rows do not match the recorded denominator")
    if len(interval_rows) != int(summary["denominators"]["moving_block_intervals_planned"]):
        raise ValueError("B=40 interval rows do not match the recorded denominator")
    keys = [(r["scenario"], int(r["outer_replicate"]), int(r["bootstrap_replicate"]))
            for r in bootstrap_rows]
    if len(set(keys)) != len(keys):
        raise ValueError("B=40 report contains duplicate panel/bootstrap IDs")
    return protocol, summary, bootstrap_rows, interval_rows


def _as_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _as_bool(value: object) -> bool:
    return str(value).lower() == "true"


def _bootstrap_seed(scenario: str, replicate: int, bootstrap_index: int) -> int:
    scenario_index = stress.SCENARIOS.index(scenario)
    seed = 2026112700 + scenario_index * 10000 + replicate
    return seed + mbb.SEED_OFFSET + bootstrap_index


def _plot(rows: list[dict[str, Any]], path: Path, scenarios: tuple[str, ...]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
    labels = {
        "matched_3state_independent": "matched",
        "omitted_4th_state": "4th omitted",
        "near_degenerate_first_gap": "near gap",
        "small_ground_overlap_operator_2": "small Z0",
        "ar1_block_rho_0p75": "AR(1) .75",
        "student_t5_independent": "Student-t5",
    }
    methods = (
        ("delta", -0.30, "#0072B2", "o"),
        ("split_conformal", -0.10, "#D55E00", "s"),
        ("moving_block_B40", 0.10, "#009E73", "^"),
        ("moving_block_B80", 0.30, "#CC79A7", "D"),
    )
    positions = np.arange(len(scenarios), dtype=float)
    for row_index, operator_index in enumerate(joint.Q_OPERATORS):
        for col_index, time in enumerate(joint.Q_TIMES):
            ax = axes[row_index, col_index]
            for method, offset, color, marker in methods:
                for position, scenario in enumerate(scenarios):
                    item = next(r for r in rows if r["scenario"] == scenario
                                and r["operator_index"] == operator_index
                                and r["time"] == time and r["method"] == method)
                    value = item["coverage"]
                    if value is None:
                        ax.scatter(position + offset, 0.5, marker="x", color=color, s=38,
                                   label=method.replace("_", " ") if position == 0 else None)
                        continue
                    low, high = item["mc95_low"], item["mc95_high"]
                    ax.errorbar(
                        position + offset, value,
                        yerr=[[max(value - low, 0.0)], [max(high - value, 0.0)]],
                        color=color, marker=marker,
                        markerfacecolor="white" if method == "split_conformal" else color,
                        linestyle="none", capsize=3,
                        label=method.replace("_", " ") if position == 0 else None,
                    )
            ax.axhline(0.975, color="#444444", linestyle=":",
                       label="97.5% target" if (row_index, col_index) == (0, 0) else None)
            ax.set_ylim(0.4, 1.01)
            ax.set_xticks(positions, [labels[s] for s in scenarios], rotation=18, ha="right")
            ax.set_ylabel("holdout pointwise coverage (95% Wilson MC interval)")
            ax.set_title(f"bare operator {operator_index}, T={time}")
            ax.grid(alpha=0.2)
            if row_index == 0 and col_index == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Synthetic MBB tail-resolution comparison; not familywise or physical coverage")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(output: str | Path = DEFAULT_OUT, *, base: str | Path = DEFAULT_BASE,
        outer_panels_per_scenario: int = 40,
        total_bootstrap_replicates: int = DEFAULT_TOTAL_REPLICATES,
        scenarios: tuple[str, ...] = stress.SCENARIOS,
        report_progress: bool = False) -> dict[str, Any]:
    if (isinstance(outer_panels_per_scenario, bool)
            or not isinstance(outer_panels_per_scenario, int)
            or not 1 <= outer_panels_per_scenario <= 40):
        raise ValueError("outer_panels_per_scenario must be in [1,40]")
    if (isinstance(total_bootstrap_replicates, bool)
            or not isinstance(total_bootstrap_replicates, int)
            or total_bootstrap_replicates <= BASE_REPLICATES):
        raise ValueError("total_bootstrap_replicates must be an integer greater than 40")
    if (not scenarios or len(set(scenarios)) != len(scenarios)
            or any(s not in stress.SCENARIOS for s in scenarios)):
        raise ValueError("scenarios must be unique names from the verified stress report")

    base_protocol, base_summary, base_bootstrap, base_intervals = _read_verified_b40(Path(base))
    stress_protocol, stress_rows = mbb._read_stress_bundle()
    split_at = int(stress_protocol["panels_per_scenario"]) // 2
    if split_at != 40 or outer_panels_per_scenario > split_at:
        raise ValueError("base stress bundle calibration/holdout split changed")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)

    base_boot_map = {
        (r["scenario"], int(r["outer_replicate"]), int(r["bootstrap_replicate"])): r
        for r in base_bootstrap
    }
    base_interval_map = {
        (r["scenario"], int(r["outer_replicate"]), int(r["operator_index"]), int(r["time"])): r
        for r in base_intervals
    }
    base_panel_rows = {
        (r["scenario"], int(r["replicate"]), int(r["operator_index"]), int(r["time"])): r
        for r in stress_rows
    }
    multipliers, calibration_receipts = mbb._calibration_multipliers(stress_rows, scenarios)

    additional_rows: list[dict[str, Any]] = []
    combined_draws: dict[tuple[str, int, int, int], list[float]] = {}
    fit_valid = fit_failed = q_valid = q_failed = q_abstained = 0
    base_fit_planned = base_fit_valid = base_fit_failed = 0
    base_q_planned = base_q_valid = base_q_failed = base_q_abstained = 0
    panel_planned = len(scenarios) * outer_panels_per_scenario
    n_blocks = int(base_protocol["n_blocks"])
    block_length = int(base_protocol["block_length"])
    scenario_index = {s: i for i, s in enumerate(stress.SCENARIOS)}
    for scenario in scenarios:
        for local_index in range(outer_panels_per_scenario):
            replicate = split_at + local_index
            for bootstrap_index in range(BASE_REPLICATES):
                row = base_boot_map.get((scenario, replicate, bootstrap_index))
                if row is None:
                    raise ValueError(f"B=40 input misses {scenario}/{replicate}/{bootstrap_index}")
                base_fit_planned += 1
                base_fit_valid += int(_as_bool(row["fit_valid"]))
                base_fit_failed += int(not _as_bool(row["fit_valid"]))
                for key, column in Q_COLUMNS.items():
                    value = _as_float(row[column]) if _as_bool(row["fit_valid"]) else None
                    base_q_planned += 1
                    if not _as_bool(row["fit_valid"]):
                        base_q_failed += 1
                    elif value is None:
                        base_q_abstained += 1
                    else:
                        base_q_valid += 1
                    draws = combined_draws.setdefault((scenario, replicate, key[0], key[1]), [])
                    draws.append(math.inf if value is None else value)
            panel = stress._generate_panel(
                2026112700 + scenario_index[scenario] * 10000 + replicate,
                n_blocks=n_blocks, scenario=scenario,
            )
            for bootstrap_index in range(BASE_REPLICATES, total_bootstrap_replicates):
                seed = _bootstrap_seed(scenario, replicate, bootstrap_index)
                rng = np.random.default_rng(seed)
                indices = mbb._circular_moving_block_indices(rng, n_blocks, block_length)
                try:
                    fit = joint.fit_joint_spectrum(panel[indices], n_states=stress.FIT_STATES)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    fit = {"success": False, "message": str(exc), "q_cells": [], "nfev": 0}
                valid = bool(fit.get("success", False))
                fit_valid += int(valid)
                fit_failed += int(not valid)
                q_by_key = {(r["operator_index"], r["time"]): r
                            for r in fit.get("q_cells", [])}
                for key, column in Q_COLUMNS.items():
                    cell = q_by_key.get(key)
                    value = (float(cell["q_estimate"])
                             if valid and cell is not None
                             and math.isfinite(float(cell["q_estimate"])) else None)
                    if not valid:
                        q_failed += 1
                    elif value is None:
                        q_abstained += 1
                    else:
                        q_valid += 1
                    combined_draws[(scenario, replicate, key[0], key[1])].append(
                        math.inf if value is None else value
                    )
                additional_rows.append({
                    "scenario": scenario, "outer_replicate": replicate,
                    "outer_seed": 2026112700 + scenario_index[scenario] * 10000 + replicate,
                    "bootstrap_replicate": bootstrap_index,
                    "bootstrap_seed": seed, "planned_blocks": n_blocks,
                    "block_length": block_length, "fit_valid": valid,
                    "fit_nfev": int(fit.get("nfev", 0)),
                    "fit_message": str(fit.get("message", "")),
                    **{column: (float((q_by_key.get(key) or {}).get("q_estimate"))
                                if valid and q_by_key.get(key) is not None
                                and math.isfinite(float(q_by_key[key]["q_estimate"])) else None)
                       for key, column in Q_COLUMNS.items()},
                })
            if report_progress and (local_index + 1) % 5 == 0:
                print(f"B80 scenario={scenario} panels={local_index + 1}/{outer_panels_per_scenario}",
                      flush=True)

    interval_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for local_index in range(outer_panels_per_scenario):
            replicate = split_at + local_index
            for (operator_index, time), column in Q_COLUMNS.items():
                draws = combined_draws[(scenario, replicate, operator_index, time)]
                upper, rank = mbb._upper_order_statistic(draws, alpha=mbb.ALPHA)
                base_interval = base_interval_map[(scenario, replicate, operator_index, time)]
                panel_row = base_panel_rows[(scenario, replicate, operator_index, time)]
                truth = float(panel_row["q_true"])
                valid_draws = sum(math.isfinite(x) for x in draws)
                failed_draws = sum(math.isinf(x) for x in draws)
                interval_rows.append({
                    "scenario": scenario, "outer_replicate": replicate,
                    "operator_index": operator_index, "time": time,
                    "planned_bootstrap_refits": total_bootstrap_replicates,
                    "valid_bootstrap_refits": valid_draws,
                    "failed_or_missing_bootstrap_refits": failed_draws,
                    "required_order_rank": rank, "bootstrap_upper": upper,
                    "b40_upper": _as_float(base_interval["bootstrap_upper"]),
                    "q_true": truth,
                    "base_q_estimate": _as_float(panel_row["q_estimate"]),
                    "covered": upper >= truth if upper is not None else None,
                    "status": "VALID" if upper is not None else "ABSTAIN_BOOTSTRAP_QUANTILE",
                })

    comparison_rows: list[dict[str, Any]] = []
    methods = ("delta", "split_conformal", "moving_block_B40", "moving_block_B80")
    for scenario in scenarios:
        for (operator_index, time), _column in Q_COLUMNS.items():
            values_by_method: dict[str, list[tuple[float, float]]] = {m: [] for m in methods}
            failed = {m: 0 for m in methods}
            abstained = {m: 0 for m in methods}
            for local_index in range(outer_panels_per_scenario):
                replicate = split_at + local_index
                row = base_panel_rows[(scenario, replicate, operator_index, time)]
                if not _as_bool(row["fit_valid"]):
                    for method in methods:
                        failed[method] += 1
                    continue
                truth = float(row["q_true"])
                estimate = _as_float(row["q_estimate"])
                error = _as_float(row["q_delta_se"])
                if estimate is None or error is None:
                    abstained["delta"] += 1
                    abstained["split_conformal"] += 1
                else:
                    values_by_method["delta"].append(
                        (truth, estimate + joint.Z_UPPER_97_5 * error))
                    multiplier = multipliers[(scenario, operator_index, time)]
                    if multiplier is None:
                        abstained["split_conformal"] += 1
                    else:
                        values_by_method["split_conformal"].append(
                            (truth, estimate + multiplier * error))
                b40 = base_interval_map[(scenario, replicate, operator_index, time)]
                for method, upper in (("moving_block_B40", _as_float(b40["bootstrap_upper"])),
                                      ("moving_block_B80", _as_float(interval_rows[
                                          (scenarios.index(scenario) * outer_panels_per_scenario
                                           + local_index) * 4
                                          + list(Q_COLUMNS).index((operator_index, time))]["bootstrap_upper"]))):
                    if upper is None:
                        abstained[method] += 1
                    else:
                        values_by_method[method].append((truth, upper))
            for method in methods:
                values = values_by_method[method]
                valid = len(values)
                failed_n = failed[method]
                abstained_n = abstained[method]
                if valid + failed_n + abstained_n != outer_panels_per_scenario:
                    raise RuntimeError(f"{method} denominator does not reconcile for {scenario}")
                covered = sum(upper >= truth for truth, upper in values)
                low, high = mbb._wilson_interval(covered, valid)
                ratios = [upper / truth for truth, upper in values]
                comparison_rows.append({
                    "scenario": scenario, "operator_index": operator_index,
                    "time": time, "method": method,
                    "planned": outer_panels_per_scenario, "valid": valid,
                    "failed": failed_n, "abstained": abstained_n,
                    "coverage": covered / valid if valid else None,
                    "mc95_low": low, "mc95_high": high,
                    "median_upper_over_true": float(np.median(ratios)) if ratios else None,
                    "block_length": block_length if method.startswith("moving_block") else None,
                    "bootstrap_replicates": (BASE_REPLICATES if method == "moving_block_B40"
                                              else total_bootstrap_replicates
                                              if method == "moving_block_B80" else None),
                    "classification": "synthetic pointwise conditional coverage; no familywise or physical guarantee",
                })

    additional_fit_planned = panel_planned * (total_bootstrap_replicates - BASE_REPLICATES)
    additional_q_planned = additional_fit_planned * len(Q_COLUMNS)
    interval_planned = panel_planned * len(Q_COLUMNS)
    valid_intervals = sum(r["status"] == "VALID" for r in interval_rows)
    abstained_intervals = interval_planned - valid_intervals
    denominators = {
        "outer_holdout_panels_planned": panel_planned,
        "outer_holdout_panels_valid": panel_planned,
        "outer_holdout_panels_failed": 0,
        "outer_holdout_panels_abstained": 0,
        "additional_refits_planned": additional_fit_planned,
        "additional_refits_valid": fit_valid,
        "additional_refits_failed": fit_failed,
        "additional_refits_abstained": 0,
        "additional_q_estimates_planned": additional_q_planned,
        "additional_q_estimates_valid": q_valid,
        "additional_q_estimates_failed": q_failed,
        "additional_q_estimates_abstained": q_abstained,
        "b80_refits_planned": base_fit_planned + additional_fit_planned,
        "b80_refits_valid": base_fit_valid + fit_valid,
        "b80_refits_failed": base_fit_failed + fit_failed,
        "b80_refits_abstained": 0,
        "b80_q_estimates_planned": base_q_planned + additional_q_planned,
        "b80_q_estimates_valid": base_q_valid + q_valid,
        "b80_q_estimates_failed": base_q_failed + q_failed,
        "b80_q_estimates_abstained": base_q_abstained + q_abstained,
        "b80_intervals_planned": interval_planned,
        "b80_intervals_valid": valid_intervals,
        "b80_intervals_failed": 0,
        "b80_intervals_abstained": abstained_intervals,
        "physical_measurements_planned": 0,
        "physical_measurements_valid": 0,
        "physical_measurements_failed": 0,
        "physical_measurements_abstained": 0,
        "certification_planned": 1,
        "certification_valid": 0,
        "certification_failed": 0,
        "certification_abstained": 1,
    }
    for prefix, planned in (("additional_refits", additional_fit_planned),
                            ("additional_q_estimates", additional_q_planned),
                            ("b80_refits", panel_planned * total_bootstrap_replicates),
                            ("b80_q_estimates", panel_planned * total_bootstrap_replicates * len(Q_COLUMNS)),
                            ("b80_intervals", interval_planned)):
        if sum(denominators[f"{prefix}_{k}"] for k in ("valid", "failed", "abstained")) != planned:
            raise RuntimeError(f"{prefix} denominator does not reconcile")

    source_hashes = {
        Path(__file__).name: _sha256(Path(__file__)),
        Path(mbb.__file__).name: _sha256(Path(mbb.__file__)),
        Path(stress.__file__).name: _sha256(Path(stress.__file__)),
        Path(joint.__file__).name: _sha256(Path(joint.__file__)),
    }
    protocol = {
        "schema": "wilson-f-joint-spectral-mbb-b80-v1",
        "classification": "synthetic CPU bootstrap stability extension only",
        "physical_score": None,
        "base_report": str(Path(base)),
        "base_report_artifact_hashes": json.loads((Path(base) / "artifact_hashes.json").read_text()),
        "base_bootstrap_replicates": BASE_REPLICATES,
        "total_bootstrap_replicates": total_bootstrap_replicates,
        "additional_bootstrap_replicates": total_bootstrap_replicates - BASE_REPLICATES,
        "scenarios": list(scenarios),
        "holdout_replicates": f"{split_at}:{split_at + outer_panels_per_scenario}",
        "block_length": block_length,
        "alpha": mbb.ALPHA,
        "seed_extension": "use the original deterministic bootstrap seed formula at indices 40 through B-1; retain all hash-verified B=40 rows unchanged",
        "upper_order_statistic": "rank=ceil((B+1)*0.975); missing or failed q estimates are +infinity",
        "source_sha256": source_hashes,
        "input_stress_artifact_hashes": json.loads((stress.DEFAULT_OUT / "artifact_hashes.json").read_text()),
        "warning": "B=80 improves empirical tail resolution but does not supply a finite-sample coverage guarantee; all results are conditional synthetic pointwise diagnostics",
        "physical_score": None,
        "omitted": ["profile likelihood", "finite-Nt thermal h", "mixed-sector x", "familywise q/h/x", "production snapshots", "independent contamination receipts"],
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    for module_path in (Path(__file__), Path(mbb.__file__), Path(stress.__file__), Path(joint.__file__)):
        shutil.copyfile(module_path, out / f"snapshot_{module_path.name}")
    for filename, rows in (("additional_bootstrap_replicates.csv", additional_rows),
                           ("b80_intervals.csv", interval_rows),
                           ("method_comparison.csv", comparison_rows),
                           ("calibration_summary.csv", calibration_receipts)):
        with (out / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    _plot(comparison_rows, out / "method_coverage.png", scenarios)
    summary = {
        "schema": "wilson-f-joint-spectral-mbb-b80-summary-v1",
        "classification": "synthetic CPU bootstrap stability extension only",
        "physical_score": None,
        "F_lane_status": "GROUND_STATE_FORCE_UNCERTIFIED",
        "denominators": denominators,
        "scenarios": list(scenarios),
        "comparison_rows": comparison_rows,
        "files": ["README.md", "protocol.json", "additional_bootstrap_replicates.csv",
                  "b80_intervals.csv", "method_comparison.csv", "calibration_summary.csv",
                  "method_coverage.png", "summary.json",
                  *[f"snapshot_{Path(p).name}" for p in
                    (Path(__file__), Path(mbb.__file__), Path(stress.__file__), Path(joint.__file__))]],
        "source_sha256": source_hashes,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    findings = []
    for scenario in scenarios:
        method_parts = []
        for method in methods:
            group = [r for r in comparison_rows if r["scenario"] == scenario and r["method"] == method]
            cov = [r["coverage"] for r in group if r["coverage"] is not None]
            valid_denoms = sorted({(r["valid"], r["planned"]) for r in group})
            interval = f"{min(cov):.3f}–{max(cov):.3f}" if cov else "abstained"
            method_parts.append(f"{method} coverage `{interval}`, valid/planned `{valid_denoms}`")
        findings.append(f"- `{scenario}`: " + "; ".join(method_parts) + ".")
    readme = [
        "# Wilson F-lane moving-block bootstrap B=80 stability extension", "",
        "**Synthetic CPU diagnostic only.** The hash-verified B=40 report is reused and extended with deterministic bootstrap indices 40–79; no measurement, sealed split, GPU or QUDA data were read. `physical_score=null`; F remains `GROUND_STATE_FORCE_UNCERTIFIED`.", "",
        "## Fixed design", "",
        f"The same {len(scenarios)} scenarios, 40-panel calibration/holdout split and length-{block_length} circular blocks are retained. Existing B=40 refits are checked by artifact SHA-256 and reused without modification; only {additional_fit_planned} additional full 3-state GLS refits are run. B=80 uses rank `ceil(81×0.975)=79`; failed/missing q draws remain `+infinity`. The same holdout panels compare delta, split-conformal, MBB B=40 and MBB B=80. This is a tail-resolution sensitivity check, not an independent coverage validation.", "",
        "## Results", "", "![B=40 and B=80 pointwise coverage](method_coverage.png)", "",
        f"Additional refits planned/valid/failed/abstained = `{additional_fit_planned}/{fit_valid}/{fit_failed}/0`; additional q estimates = `{additional_q_planned}/{q_valid}/{q_failed}/{q_abstained}`; B=80 intervals = `{interval_planned}/{valid_intervals}/0/{abstained_intervals}`. Cell-level MC intervals, widths and abstentions are in `method_comparison.csv`.", "",
        *findings, "",
        "## Interpretation and limits", "",
        "B=80 can show whether B=40's maximum order statistic materially changes, but it cannot repair structural bias from fitting three states to a four-state truth, near-degenerate identifiability, or unmodeled chain dependence. Coverage is conditional on finite, reported intervals; abstentions remain in the denominators. No q/h/x familywise guarantee is formed, and this report does not authorize a bound for the physical gate.", "",
        "Text flow: `verify B=40 artifact/source hashes -> preserve indices 0:40 -> add deterministic indices 40:80 -> refit same full GLS on fixed holdout panels -> B=40/B=80/delta/conformal coverage, widths and abstentions -> certification abstains`.", "",
        "Reproduce with `PYTHONPATH=. python -B -m tasks.wilson_loop.f_lane_joint_spectral_mbb_b80_diagnostic --output tasks/wilson_loop/f_lane_joint_spectral_mbb_b80_20261126_v1`. Source and output hashes are in `protocol.json`, `summary.json` and `artifact_hashes.json`.",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n")
    hashes = {name: _sha256(out / name) for name in summary["files"] if (out / name).is_file()}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--outer-panels-per-scenario", type=int, default=40)
    parser.add_argument("--total-bootstrap-replicates", type=int, default=DEFAULT_TOTAL_REPLICATES)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = run(args.output, base=args.base,
                 outer_panels_per_scenario=args.outer_panels_per_scenario,
                 total_bootstrap_replicates=args.total_bootstrap_replicates,
                 report_progress=not args.quiet)
    print(json.dumps(result["denominators"], sort_keys=True))


if __name__ == "__main__":
    main()
