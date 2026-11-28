"""Compare circular moving-block bootstrap bounds with delta and conformal.

The input is the tracked synthetic six-scenario joint-spectrum stress bundle.
This is a CPU method diagnostic only; it is not a Wilson production reader or
physical evidence producer.
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
from tasks.wilson_loop import f_lane_joint_spectral_stress_diagnostic as stress


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = TASK_DIR / "f_lane_joint_spectral_mbb_20261126_v1"
DEFAULT_OUTER_PANELS = 40
DEFAULT_BOOTSTRAP_REPLICATES = 40
DEFAULT_BLOCK_LENGTH = 8
ALPHA = 0.025
CALIBRATION_SPLIT_AT = 40
SEED_OFFSET = 530_000_000


def _circular_moving_block_indices(rng: np.random.Generator, n: int,
                                   block_length: int) -> np.ndarray:
    if (isinstance(n, bool) or not isinstance(n, int) or n < 1
            or isinstance(block_length, bool) or not isinstance(block_length, int)
            or block_length < 1 or block_length > n):
        raise ValueError("n and block_length must satisfy 1 <= block_length <= n")
    n_chunks = math.ceil(n / block_length)
    starts = rng.integers(0, n, size=n_chunks)
    chunks = [(int(start) + np.arange(block_length)) % n for start in starts]
    return np.concatenate(chunks)[:n].astype(int)


def _upper_order_statistic(values: object, alpha: float = ALPHA) -> tuple[float | None, int]:
    sample = np.asarray(values, dtype=float)
    if (sample.ndim != 1 or sample.size < 1 or not math.isfinite(alpha)
            or not 0.0 < alpha < 1.0 or np.isnan(sample).any()
            or np.isneginf(sample).any()):
        raise ValueError("bootstrap values and alpha are invalid")
    rank = int(math.ceil((sample.size + 1) * (1.0 - alpha)))
    if rank > sample.size:
        return None, rank
    value = float(np.sort(sample)[rank - 1])
    return (value if math.isfinite(value) else None), rank


def _read_stress_bundle() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = stress.DEFAULT_OUT
    summary_path = root / "summary.json"
    protocol_path = root / "protocol.json"
    manifest_path = root / "artifact_hashes.json"
    for path in (summary_path, protocol_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"required tracked stress artifact is missing: {path}")
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest.items():
        artifact = root / name
        if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
            raise ValueError(f"stress artifact hash mismatch: {name}")
    protocol = json.loads(protocol_path.read_text())
    expected_sources = {
        Path(stress.__file__).name: hashlib.sha256(Path(stress.__file__).read_bytes()).hexdigest(),
        Path(joint.__file__).name: hashlib.sha256(Path(joint.__file__).read_bytes()).hexdigest(),
    }
    if protocol.get("source_sha256") != expected_sources:
        raise ValueError("stress report was generated from different source code")
    summary = json.loads(summary_path.read_text())
    if (summary.get("physical_score") is not None
            or summary.get("classification") != "synthetic CPU stress diagnostic only"):
        raise ValueError("input bundle is not the expected synthetic stress report")
    with (root / "panel_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != int(summary["denominators"]["q_cells_planned"]):
        raise ValueError("stress panel-results denominator does not match summary")
    return protocol, rows


def _float_or_none(value: str | float | None) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _calibration_multipliers(rows: list[dict[str, Any]], scenarios: tuple[str, ...]
                             ) -> tuple[dict[tuple[str, int, int], float | None], list[dict[str, Any]]]:
    multipliers: dict[tuple[str, int, int], float | None] = {}
    receipts = []
    for scenario in scenarios:
        for operator_index in joint.Q_OPERATORS:
            for time in joint.Q_TIMES:
                training = [row for row in rows
                            if row["scenario"] == scenario
                            and int(row["replicate"]) < CALIBRATION_SPLIT_AT
                            and int(row["operator_index"]) == operator_index
                            and int(row["time"]) == time]
                scores = []
                valid = failed = abstained = 0
                for row in training:
                    estimate = _float_or_none(row["q_estimate"])
                    error = _float_or_none(row["q_delta_se"])
                    if row["fit_valid"].lower() != "true":
                        failed += 1
                        scores.append(math.inf)
                    elif estimate is None or error is None:
                        abstained += 1
                        scores.append(math.inf)
                    else:
                        valid += 1
                        scores.append((float(row["q_true"]) - estimate) / error
                                      if error > 0.0 else math.inf)
                planned = int(protocol_calibration_size(rows, scenario))
                if len(training) != planned:
                    raise ValueError(f"incomplete calibration IDs for {scenario}")
                multiplier = joint._split_conformal_upper_multiplier(scores, alpha=ALPHA)
                key = (scenario, operator_index, time)
                multipliers[key] = multiplier
                receipts.append({
                    "scenario": scenario, "operator_index": operator_index, "time": time,
                    "planned": len(training), "valid": valid, "failed": failed,
                    "abstained": abstained, "multiplier": multiplier,
                    "status": "FINITE" if multiplier is not None else "ABSTAIN_NO_FINITE_MULTIPLIER",
                })
    return multipliers, receipts


def protocol_calibration_size(rows: list[dict[str, Any]], scenario: str) -> int:
    counts = {int(row["replicate"]) for row in rows if row["scenario"] == scenario
              and int(row["replicate"]) < CALIBRATION_SPLIT_AT}
    if not counts:
        return CALIBRATION_SPLIT_AT
    return len(counts)


def _wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    if total < 1:
        return None, None
    low, high = joint._wilson_interval(successes, total)
    return low, high


def _comparison_plot(rows: list[dict[str, Any]], path: Path,
                     scenarios: tuple[str, ...]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
    scenario_labels = {
        "matched_3state_independent": "matched",
        "omitted_4th_state": "4th omitted",
        "near_degenerate_first_gap": "near gap",
        "small_ground_overlap_operator_2": "small Z0",
        "ar1_block_rho_0p75": "AR(1) .75",
        "student_t5_independent": "Student-t5",
    }
    labels = [scenario_labels[scenario] for scenario in scenarios]
    positions = np.arange(len(scenarios), dtype=float)
    methods = (
        ("delta", -0.22, "#0072B2", "o"),
        ("split_conformal", 0.0, "#D55E00", "s"),
        ("moving_block_bootstrap", 0.22, "#009E73", "^"),
    )
    for row_index, operator_index in enumerate(joint.Q_OPERATORS):
        for col_index, time in enumerate(joint.Q_TIMES):
            ax = axes[row_index, col_index]
            subset = [row for row in rows
                      if row["operator_index"] == operator_index and row["time"] == time]
            for method, offset, color, marker in methods:
                for position, scenario in enumerate(scenarios):
                    item = next(row for row in subset
                                if row["scenario"] == scenario and row["method"] == method)
                    value = item["coverage"]
                    if value is None:
                        ax.scatter(position + offset, 0.5, marker="x", color=color, s=38,
                                   label=method.replace("_", " ") if position == 0 else None)
                        continue
                    low, high = item["mc95_low"], item["mc95_high"]
                    ax.errorbar(
                        position + offset, value,
                        yerr=[[max(value - low, 0.0)], [max(high - value, 0.0)]],
                        color=color, marker=marker, markerfacecolor=("white" if method == "split_conformal" else color),
                        linestyle="none", capsize=3,
                        label=method.replace("_", " ") if position == 0 else None,
                    )
            ax.axhline(0.975, color="#444444", linestyle=":",
                       label="97.5% target" if (row_index, col_index) == (0, 0) else None)
            ax.set_ylim(0.4, 1.01)
            ax.set_xticks(positions, labels, rotation=18, ha="right")
            ax.set_ylabel("holdout one-sided upper coverage (95% Wilson MC interval)")
            ax.set_title(f"bare operator {operator_index}, T={time}")
            ax.grid(alpha=0.2)
            if row_index == 0 and col_index == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Synthetic pointwise q upper comparison; not familywise or physical coverage")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(output: str | Path = DEFAULT_OUT, *,
        outer_panels_per_scenario: int = DEFAULT_OUTER_PANELS,
        bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
        block_length: int = DEFAULT_BLOCK_LENGTH,
        scenarios: tuple[str, ...] = stress.SCENARIOS,
        report_progress: bool = False) -> dict[str, Any]:
    if (isinstance(outer_panels_per_scenario, bool)
            or not isinstance(outer_panels_per_scenario, int)
            or not 1 <= outer_panels_per_scenario <= 40):
        raise ValueError("outer_panels_per_scenario must be an integer in [1,40]")
    if (isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int)
            or bootstrap_replicates < 1):
        raise ValueError("bootstrap_replicates must be a positive integer")
    if (isinstance(block_length, bool) or not isinstance(block_length, int)
            or not 1 <= block_length <= stress.N_BLOCKS):
        raise ValueError("block_length must be in [1,n_blocks]")
    if (not scenarios or len(set(scenarios)) != len(scenarios)
            or any(scenario not in stress.SCENARIOS for scenario in scenarios)):
        raise ValueError("scenarios must be unique names from the verified stress report")
    base_protocol, base_rows = _read_stress_bundle()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    n_blocks = int(base_protocol["n_blocks_per_panel"])
    if block_length > n_blocks:
        raise ValueError("block_length exceeds the verified stress panel size")
    split_at = int(base_protocol["panels_per_scenario"]) // 2
    if split_at != CALIBRATION_SPLIT_AT or outer_panels_per_scenario > split_at:
        raise ValueError("stress bundle calibration/holdout split does not match this protocol")
    multipliers, calibration_rows = _calibration_multipliers(base_rows, scenarios)
    scenario_index = {name: i for i, name in enumerate(stress.SCENARIOS)}
    base_by_key = {
        (row["scenario"], int(row["replicate"]), int(row["operator_index"]), int(row["time"])): row
        for row in base_rows
    }
    source_hashes = {
        Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        Path(stress.__file__).name: hashlib.sha256(Path(stress.__file__).read_bytes()).hexdigest(),
        Path(joint.__file__).name: hashlib.sha256(Path(joint.__file__).read_bytes()).hexdigest(),
    }
    protocol = {
        "schema": "wilson-f-joint-spectral-mbb-v1",
        "classification": "synthetic CPU interval-method comparison; not physical evidence",
        "physical_score": None,
        "input_stress_schema": base_protocol["schema"],
        "input_stress_artifact_hashes": json.loads((stress.DEFAULT_OUT / "artifact_hashes.json").read_text()),
        "scenarios": list(scenarios),
        "outer_panel_replicates": f"{split_at}:{split_at + outer_panels_per_scenario}",
        "calibration_replicates": f"0:{split_at}",
        "n_blocks": n_blocks,
        "block_length": block_length,
        "bootstrap_replicates_per_outer_panel": bootstrap_replicates,
        "alpha": ALPHA,
        "moving_block_rule": "circular moving blocks sampled with replacement; concatenate until n_blocks; refit full 3-state GLS for every resample",
        "upper_order_statistic": "rank=ceil((B+1)*(1-alpha)); failed or missing bootstrap q estimates are +infinity; insufficient rank or infinite selected value abstains",
        "comparison_methods": {
            "delta": "qhat + 1.95996398454 * same-fit delta-method SE",
            "split_conformal": "qhat + per-scenario/operator/time multiplier calibrated on the disjoint first 40 panels",
            "moving_block_bootstrap": "empirical one-sided percentile of bootstrap refit q estimates using circular block length 8",
        },
        "source_sha256": source_hashes,
        "warning": "B=40 has coarse 97.5% tail resolution; bootstrap coverage is an empirical synthetic pilot and has no finite-sample guarantee",
        "omitted": ["finite-Nt thermal h", "mixed-sector x", "familywise q/h/x envelope", "production Wilson snapshots", "independent contamination receipts"],
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    for module_path in (Path(__file__), Path(stress.__file__), Path(joint.__file__)):
        shutil.copyfile(module_path, out / f"snapshot_{module_path.name}")

    raw_bootstrap_rows: list[dict[str, Any]] = []
    bound_rows: list[dict[str, Any]] = []
    fit_valid = fit_failed = 0
    q_boot_valid = q_boot_failed = q_boot_abstained = 0
    outer_panel_planned = len(scenarios) * outer_panels_per_scenario
    for scenario in scenarios:
        for local_index in range(outer_panels_per_scenario):
            replicate = split_at + local_index
            panel_key_rows = {
                (operator_index, time): base_by_key[(scenario, replicate, operator_index, time)]
                for operator_index in joint.Q_OPERATORS for time in joint.Q_TIMES
            }
            first = next(iter(panel_key_rows.values()))
            if first["fit_valid"].lower() != "true":
                raise ValueError(f"base stress fit failed for holdout panel {scenario}/{replicate}")
            seed = 2026112700 + scenario_index[scenario] * 10000 + replicate
            blocks = stress._generate_panel(seed, n_blocks=n_blocks, scenario=scenario)
            q_draws = {(op, time): [] for op in joint.Q_OPERATORS for time in joint.Q_TIMES}
            panel_fit_failed = 0
            for bootstrap_index in range(bootstrap_replicates):
                rng = np.random.default_rng(seed + SEED_OFFSET + bootstrap_index)
                indices = _circular_moving_block_indices(rng, n_blocks, block_length)
                try:
                    fit = joint.fit_joint_spectrum(blocks[indices], n_states=stress.FIT_STATES)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    fit = {"success": False, "message": str(exc), "q_cells": [], "nfev": 0}
                valid = bool(fit.get("success", False))
                fit_valid += int(valid)
                fit_failed += int(not valid)
                panel_fit_failed += int(not valid)
                q_by_key = {(row["operator_index"], row["time"]): row
                            for row in fit.get("q_cells", [])}
                q_values = {}
                for operator_index in joint.Q_OPERATORS:
                    for time in joint.Q_TIMES:
                        cell = q_by_key.get((operator_index, time))
                        value = (float(cell["q_estimate"])
                                 if valid and cell is not None
                                 and math.isfinite(float(cell["q_estimate"])) else None)
                        q_values[(operator_index, time)] = value
                        q_draws[(operator_index, time)].append(
                            math.inf if value is None else value
                        )
                        if not valid:
                            q_boot_failed += 1
                        elif value is None:
                            q_boot_abstained += 1
                        else:
                            q_boot_valid += 1
                raw_bootstrap_rows.append({
                    "scenario": scenario, "outer_replicate": replicate,
                    "outer_seed": seed, "bootstrap_replicate": bootstrap_index,
                    "bootstrap_seed": seed + SEED_OFFSET + bootstrap_index,
                    "planned_blocks": n_blocks, "block_length": block_length,
                    "fit_valid": valid, "fit_nfev": int(fit.get("nfev", 0)),
                    "fit_message": str(fit.get("message", "")),
                    "q_op0_T5": q_values[(0, 5)], "q_op0_T6": q_values[(0, 6)],
                    "q_op2_T5": q_values[(2, 5)], "q_op2_T6": q_values[(2, 6)],
                })
            for operator_index in joint.Q_OPERATORS:
                for time in joint.Q_TIMES:
                    base_row = panel_key_rows[(operator_index, time)]
                    draws = q_draws[(operator_index, time)]
                    upper, rank = _upper_order_statistic(draws, alpha=ALPHA)
                    valid_draws = sum(math.isfinite(value) for value in draws)
                    failed_draws = sum(math.isinf(value) for value in draws)
                    status = "VALID" if upper is not None else "ABSTAIN_BOOTSTRAP_QUANTILE"
                    bound_rows.append({
                        "scenario": scenario, "outer_replicate": replicate,
                        "operator_index": operator_index, "time": time,
                        "planned_bootstrap_refits": bootstrap_replicates,
                        "valid_bootstrap_refits": valid_draws,
                        "failed_or_missing_bootstrap_refits": failed_draws,
                        "required_order_rank": rank,
                        "bootstrap_upper": upper,
                        "q_true": float(base_row["q_true"]),
                        "base_q_estimate": _float_or_none(base_row["q_estimate"]),
                        "base_delta_upper": _float_or_none(base_row["delta_upper"]),
                        "base_delta_se": _float_or_none(base_row["q_delta_se"]),
                        "base_fit_valid": base_row["fit_valid"].lower() == "true",
                        "upper_over_truth": (upper / float(base_row["q_true"])
                                              if upper is not None else None),
                        "covered": (upper >= float(base_row["q_true"])
                                    if upper is not None else None),
                        "status": status,
                    })
            completed = local_index + 1
            if report_progress and completed % 5 == 0:
                print(f"MBB scenario={scenario} outer_panels={completed}/{outer_panels_per_scenario}",
                      flush=True)

    calibration_by_key = {(row["scenario"], row["operator_index"], row["time"]): row
                          for row in calibration_rows}
    bounds_by_key = {(row["scenario"], row["outer_replicate"],
                      row["operator_index"], row["time"]): row for row in bound_rows}
    comparison_rows = []
    methods = ("delta", "split_conformal", "moving_block_bootstrap")
    for scenario in scenarios:
        for operator_index in joint.Q_OPERATORS:
            for time in joint.Q_TIMES:
                observations = [base_by_key[(scenario, split_at + offset, operator_index, time)]
                                for offset in range(outer_panels_per_scenario)]
                multiplier = multipliers[(scenario, operator_index, time)]
                method_values: dict[str, list[tuple[float, float]]] = {method: [] for method in methods}
                method_failed: dict[str, int] = {method: 0 for method in methods}
                method_abstained: dict[str, int] = {method: 0 for method in methods}
                for offset, base_row in enumerate(observations):
                    replicate = split_at + offset
                    if base_row["fit_valid"].lower() != "true":
                        for method in methods:
                            method_failed[method] += 1
                        continue
                    estimate = _float_or_none(base_row["q_estimate"])
                    error = _float_or_none(base_row["q_delta_se"])
                    truth = float(base_row["q_true"])
                    if estimate is None or error is None:
                        method_abstained["delta"] += 1
                        method_abstained["split_conformal"] += 1
                    else:
                        delta_upper = estimate + joint.Z_UPPER_97_5 * error
                        method_values["delta"].append((truth, delta_upper))
                        if multiplier is None:
                            method_abstained["split_conformal"] += 1
                        else:
                            method_values["split_conformal"].append(
                                (truth, estimate + multiplier * error)
                            )
                    bound = bounds_by_key[(scenario, replicate, operator_index, time)]
                    if bound["bootstrap_upper"] is None:
                        method_abstained["moving_block_bootstrap"] += 1
                    else:
                        method_values["moving_block_bootstrap"].append(
                            (truth, float(bound["bootstrap_upper"]))
                        )
                for method in methods:
                    values = method_values[method]
                    valid = len(values)
                    failed = method_failed[method]
                    abstained = method_abstained[method]
                    planned = outer_panels_per_scenario
                    if valid + failed + abstained != planned:
                        raise RuntimeError(f"{method} denominator does not reconcile for {scenario}")
                    covered = sum(upper >= truth for truth, upper in values)
                    low, high = _wilson_interval(covered, valid)
                    ratios = [upper / truth for truth, upper in values]
                    comparison_rows.append({
                        "scenario": scenario, "operator_index": operator_index, "time": time,
                        "method": method, "planned": planned, "valid": valid,
                        "failed": failed, "abstained": abstained,
                        "coverage": covered / valid if valid else None,
                        "mc95_low": low, "mc95_high": high,
                        "median_upper_over_true": float(np.median(ratios)) if ratios else None,
                        "conformal_multiplier": multiplier if method == "split_conformal" else None,
                        "bootstrap_block_length": block_length if method == "moving_block_bootstrap" else None,
                        "bootstrap_replicates": (bootstrap_replicates
                                                  if method == "moving_block_bootstrap" else None),
                        "classification": "synthetic pointwise interval comparison; no familywise or physical guarantee",
                    })

    fit_planned = outer_panel_planned * bootstrap_replicates
    q_boot_planned = fit_planned * len(joint.Q_OPERATORS) * len(joint.Q_TIMES)
    interval_planned = outer_panel_planned * len(joint.Q_OPERATORS) * len(joint.Q_TIMES)
    bb_valid = sum(row["status"] == "VALID" for row in bound_rows)
    bb_failed = sum(not row["base_fit_valid"] for row in bound_rows)
    bb_abstained = interval_planned - bb_valid - bb_failed
    denominators = {
        "outer_holdout_panels_planned": outer_panel_planned,
        "outer_holdout_panels_valid": outer_panel_planned,
        "outer_holdout_panels_failed": 0,
        "outer_holdout_panels_abstained": 0,
        "bootstrap_refits_planned": fit_planned,
        "bootstrap_refits_valid": fit_valid,
        "bootstrap_refits_failed": fit_failed,
        "bootstrap_refits_abstained": 0,
        "bootstrap_q_estimates_planned": q_boot_planned,
        "bootstrap_q_estimates_valid": q_boot_valid,
        "bootstrap_q_estimates_failed": q_boot_failed,
        "bootstrap_q_estimates_abstained": q_boot_abstained,
        "moving_block_intervals_planned": interval_planned,
        "moving_block_intervals_valid": bb_valid,
        "moving_block_intervals_failed": bb_failed,
        "moving_block_intervals_abstained": bb_abstained,
        "physical_measurements_planned": 0,
        "physical_measurements_valid": 0,
        "physical_measurements_failed": 0,
        "physical_measurements_abstained": 0,
        "certification_planned": 1,
        "certification_valid": 0,
        "certification_failed": 0,
        "certification_abstained": 1,
    }
    for prefix, planned in (("bootstrap_refits", fit_planned),
                            ("bootstrap_q_estimates", q_boot_planned),
                            ("moving_block_intervals", interval_planned)):
        if (denominators[f"{prefix}_valid"] + denominators[f"{prefix}_failed"]
                + denominators[f"{prefix}_abstained"] != planned):
            raise RuntimeError(f"{prefix} denominator does not reconcile")

    for name, rows in (("bootstrap_replicates.csv", raw_bootstrap_rows),
                       ("bootstrap_intervals.csv", bound_rows),
                       ("method_comparison.csv", comparison_rows),
                       ("calibration_summary.csv", calibration_rows)):
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    _comparison_plot(comparison_rows, out / "method_coverage.png", scenarios)
    summary = {
        "schema": "wilson-f-joint-spectral-mbb-summary-v1",
        "classification": "synthetic CPU interval-method comparison only",
        "physical_score": None,
        "F_lane_status": "GROUND_STATE_FORCE_UNCERTIFIED",
        "denominators": denominators,
        "scenarios": list(scenarios),
        "comparison_rows": comparison_rows,
        "files": ["README.md", "protocol.json", "bootstrap_replicates.csv",
                  "bootstrap_intervals.csv", "method_comparison.csv",
                  "calibration_summary.csv", "method_coverage.png", "summary.json",
                  f"snapshot_{Path(__file__).name}",
                  f"snapshot_{Path(stress.__file__).name}",
                  f"snapshot_{Path(joint.__file__).name}"],
        "source_sha256": source_hashes,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    findings = []
    for scenario in scenarios:
        parts = []
        for method in methods:
            group = [row for row in comparison_rows
                     if row["scenario"] == scenario and row["method"] == method]
            coverage = [float(row["coverage"]) for row in group if row["coverage"] is not None]
            widths = [float(row["median_upper_over_true"])
                      for row in group if row["median_upper_over_true"] is not None]
            cov_text = f"{min(coverage):.3f}–{max(coverage):.3f}" if coverage else "abstained"
            width_text = f"{min(widths):.3f}–{max(widths):.3f}" if widths else "unavailable"
            parts.append(f"{method}: coverage `{cov_text}`, median upper/truth `{width_text}`")
        findings.append(f"- `{scenario}`: " + "; ".join(parts) + ".")
    lines = [
        "# Wilson F-lane moving-block bootstrap interval pilot",
        "",
        "**Synthetic CPU diagnostic only.** The input is the hash-verified tracked six-scenario synthetic stress bundle. No Wilson measurement, sealed split, GPU or QUDA data were read. `physical_score=null`; F remains `GROUND_STATE_FORCE_UNCERTIFIED`.",
        "",
        "## Fixed design and interval rules",
        "",
        f"For each of {len(scenarios)} scenarios, the run uses the 40 disjoint holdout panels from the six-scenario report, and draws `{bootstrap_replicates}` circular moving-block resamples per panel with fixed block length `{block_length}` from `{n_blocks}` ordered blocks. Length 8 is chosen to span the approximate AR(1) integrated correlation scale at rho 0.75 (about 7 blocks); the same rule is held fixed across all scenarios. Each resample refits the full three-state joint GLS model. Pointwise q cells are bare operators 0/2 at T=5/6.",
        "",
        f"The moving-block upper is the empirical order statistic at rank `ceil((B+1)*0.975)` among refit q estimates. A failed/missing bootstrap q estimate is +infinity; if the selected order statistic is infinite, the interval abstains. With B={bootstrap_replicates}, tail resolution is coarse and the selected statistic is the sample maximum; report this as a pilot only. Comparators use the same holdout panels: pointwise delta upper, split-conformal upper calibrated on the separate first 40 synthetic panels, and moving-block bootstrap upper.",
        "",
        "## Results",
        "",
        "![Pointwise coverage comparison](method_coverage.png)",
        "",
        f"Outer panels planned/valid/failed/abstained = `{outer_panel_planned}/{outer_panel_planned}/0/0`; bootstrap refits = `{fit_planned}/{fit_valid}/{fit_failed}/0`; bootstrap q estimates = `{q_boot_planned}/{q_boot_valid}/{q_boot_failed}/{q_boot_abstained}`; moving-block intervals = `{interval_planned}/{bb_valid}/{bb_failed}/{bb_abstained}`. Full per-cell 95% Wilson MC intervals, widths and calibration values are in `method_comparison.csv` and `calibration_summary.csv`.",
        "",
        *findings,
        "",
        "## Interpretation and limits",
        "",
        "The comparison is pointwise across independent synthetic panels within each exact scenario. Bootstrap refits are conditional on the generated panel; they do not correct omitted-state bias or prove positive-spectrum assumptions. The B=40 upper-tail estimate is highly discrete, and this run does not tune block length or account for uncertainty in choosing it. Split-conformal has a separate finite-sample exchangeability argument across panels; moving-block bootstrap does not inherit that guarantee. None of these intervals controls simultaneous q/h/x familywise coverage.",
        "",
        "Text flow: `hash-check six-scenario stress bundle -> use fixed 40-panel holdout -> regenerate each deterministic panel from its seed -> circular length-8 block resamples -> refit 3-state full-covariance GLS -> retain every bootstrap failure as +infinity -> compare interval coverage/width/abstention -> certification abstains`.",
        "",
        "No bound from this report may enter `spectral_q_bound_reader` or the force gate. Remaining work includes a larger-bootstrap stability scan, block-length sensitivity fixed before evaluation, a profile-likelihood comparator, familywise q/h/x coverage, and real authorized calibration snapshots with independent contamination receipts. Keep `NOT_READY/UNCERTIFIED`, `GROUND_STATE_FORCE_UNCERTIFIED`, and `physical_score=null`.",
        "",
        f"Reproduce with `PYTHONPATH=. python -B -m tasks.wilson_loop.f_lane_joint_spectral_mbb_diagnostic --output tasks/wilson_loop/f_lane_joint_spectral_mbb_20261126_v1 --outer-panels-per-scenario 40 --bootstrap-replicates 40 --block-length 8`. Source and artifact hashes are recorded in `protocol.json`, `summary.json`, and `artifact_hashes.json`.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    hashes = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
              for name in summary["files"] if (out / name).is_file()}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--outer-panels-per-scenario", type=int, default=DEFAULT_OUTER_PANELS)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = run(args.output, outer_panels_per_scenario=args.outer_panels_per_scenario,
                 bootstrap_replicates=args.bootstrap_replicates,
                 block_length=args.block_length, report_progress=not args.quiet)
    print(json.dumps({"output": str(args.output), "denominators": result["denominators"]}, indent=2))


if __name__ == "__main__":
    main()
