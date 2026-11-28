"""Stress synthetic Wilson F-lane joint-spectrum intervals beyond the matched model.

This is a CPU-only method diagnostic. It deliberately tests model misspecification,
weak bare overlaps, near-degenerate levels and within-panel block dependence. It is
not a Wilson measurement reader, calibration receipt, or physical certification.
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


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = TASK_DIR / "f_lane_joint_spectral_stress_20261126_v1"
PANELS_PER_SCENARIO = 80
N_BLOCKS = 96
RELATIVE_NOISE = 0.04
CALIBRATION_ALPHA = joint.CALIBRATION_ALPHA
SPLIT_AT = PANELS_PER_SCENARIO // 2
FIT_STATES = 3
SCENARIOS = (
    "matched_3state_independent",
    "omitted_4th_state",
    "near_degenerate_first_gap",
    "small_ground_overlap_operator_2",
    "ar1_block_rho_0p75",
    "student_t5_independent",
)


def _scenario(name: str) -> tuple[np.ndarray, np.ndarray, float, str]:
    energies = joint.TRUE_ENERGIES.copy()
    overlaps = joint.TRUE_OVERLAPS.copy()
    rho = 0.0
    description = "three-state truth, three-state fit, independent Gaussian blocks"
    if name == "omitted_4th_state":
        energies = np.asarray([0.18, 0.32, 0.49, 0.67], dtype=float)
        overlaps = np.vstack([
            overlaps,
            np.asarray([0.26, 0.38 + 0.04j, -0.31 + 0.02j]),
        ])
        description = "four-state truth fitted with three states; fourth state omitted"
    elif name == "near_degenerate_first_gap":
        energies = np.asarray([0.18, 0.205, 0.49], dtype=float)
        description = "three-state truth with E1-E0=0.025 fitted with three states"
    elif name == "small_ground_overlap_operator_2":
        overlaps[0, 2] = 0.04 + 0.0j
        description = "three-state truth with |Z0(operator 2)|=0.04"
    elif name == "ar1_block_rho_0p75":
        rho = 0.75
        description = "three-state truth; stationary AR(1) dependence rho=0.75 within each panel"
    elif name == "student_t5_independent":
        description = "three-state truth; independent multivariate Student-t innovations with 5 degrees of freedom"
    elif name != "matched_3state_independent":
        raise ValueError(f"unknown stress scenario: {name}")
    return energies, overlaps, rho, description


def _spectrum_matrices(energies: np.ndarray, overlaps: np.ndarray) -> np.ndarray:
    energy = np.asarray(energies, dtype=float)
    z = np.asarray(overlaps, dtype=complex)
    if (energy.ndim != 1 or energy.size not in (3, 4)
            or z.shape != (energy.size, 3)
            or not np.isfinite(energy).all() or not np.isfinite(z).all()
            or np.any(energy <= 0.0) or np.any(np.diff(energy) <= 0.0)):
        raise ValueError("stress truth must have 3/4 ordered positive states and 3 operators")
    return np.asarray([
        np.einsum("n,ni,nj->ij", np.exp(-energy * time), z, z.conj())
        for time in joint.TIME_SLICES
    ])


def _true_q(energies: np.ndarray, overlaps: np.ndarray,
            operator_index: int, time: int) -> float:
    ground = abs(overlaps[0, operator_index])
    if not math.isfinite(float(ground)) or ground <= 1e-12:
        raise ValueError("truth requires a nonzero bare ground-state overlap")
    ratios = np.abs(overlaps[1:, operator_index] / overlaps[0, operator_index]) ** 2
    return float(np.sum(ratios * np.exp(-(energies[1:] - energies[0]) * time)))


def _generate_panel(seed: int, *, n_blocks: int, scenario: str) -> np.ndarray:
    energies, overlaps, rho, _ = _scenario(scenario)
    matrices = _spectrum_matrices(energies, overlaps)
    center = np.concatenate([joint._matrix_vector(matrix) for matrix in matrices])
    per_time_scale = np.asarray([
        max(float(np.linalg.norm(center[index * 9:(index + 1) * 9])) / 3.0, 1e-10)
        for index in range(len(joint.TIME_SLICES))
    ])
    scales = np.repeat(RELATIVE_NOISE * per_time_scale, 9)
    time_correlation = np.fromfunction(
        lambda i, j: 0.35 ** np.abs(i - j),
        (len(joint.TIME_SLICES), len(joint.TIME_SLICES)),
    )
    component_correlation = np.fromfunction(
        lambda i, j: 0.20 ** np.abs(i - j), (9, 9),
    )
    correlation = np.kron(time_correlation, component_correlation)
    innovation_covariance = np.outer(scales, scales) * correlation
    rng = np.random.default_rng(seed)
    if scenario == "student_t5_independent":
        factor = np.linalg.cholesky(innovation_covariance)
        normal = rng.standard_normal(size=(n_blocks, center.size)) @ factor.T
        chi_square = rng.chisquare(5, size=n_blocks)
        innovations = normal * np.sqrt(3.0 / chi_square)[:, None]
    else:
        innovations = rng.multivariate_normal(
            np.zeros(center.size), innovation_covariance, size=n_blocks,
        )
    if rho:
        deviations = np.empty_like(innovations)
        deviations[0] = innovations[0]
        innovation_scale = math.sqrt(1.0 - rho * rho)
        for index in range(1, n_blocks):
            deviations[index] = rho * deviations[index - 1] + innovation_scale * innovations[index]
    else:
        deviations = innovations
    return deviations + center


def _coverage_plot(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    labels = ["matched", "4th omitted", "near gap", "small Z0", "AR(1) 0.75", "Student-t5"]
    positions = np.arange(len(SCENARIOS), dtype=float)
    colors = {"delta": "#0072B2", "split_conformal": "#D55E00"}
    offsets = {"delta": -0.09, "split_conformal": 0.09}
    for row_index, operator_index in enumerate(joint.Q_OPERATORS):
        for col_index, time in enumerate(joint.Q_TIMES):
            ax = axes[row_index, col_index]
            subset = [row for row in rows
                      if row["operator_index"] == operator_index and row["time"] == time]
            for method in ("delta", "split_conformal"):
                prefix = "delta" if method == "delta" else "conformal"
                for position, scenario in enumerate(SCENARIOS):
                    item = next(row for row in subset if row["scenario"] == scenario)
                    value = item[f"{prefix}_coverage"]
                    if value is None:
                        ax.scatter(position + offsets[method], 0.52, marker="x",
                                   color=colors[method], s=40,
                                   label=f"{method}: abstain" if position == 0 else None)
                        continue
                    low = item[f"{prefix}_mc95_low"]
                    high = item[f"{prefix}_mc95_high"]
                    ax.errorbar(
                        position + offsets[method], value,
                        yerr=[[max(value - low, 0.0)], [max(high - value, 0.0)]],
                        color=colors[method], marker="o" if method == "delta" else "s",
                        markerfacecolor=colors[method] if method == "delta" else "white",
                        linestyle="none", capsize=3,
                        label=method.replace("_", " ") if position == 0 else None,
                    )
            ax.axhline(1.0 - CALIBRATION_ALPHA, color="#444444", linestyle=":",
                       label="97.5% target" if (row_index, col_index) == (0, 0) else None)
            ax.set_ylim(0.45, 1.01)
            ax.set_xticks(positions, labels, rotation=18, ha="right")
            ax.set_ylabel("holdout upper coverage (95% Wilson interval)")
            ax.set_title(f"bare operator {operator_index}, T={time}")
            ax.grid(alpha=0.2)
            if row_index == 0 and col_index == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Joint GLS stress: panel-level split calibration; synthetic CPU only")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _conservatism_plot(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    labels = ["matched", "4th omitted", "near gap", "small Z0", "AR(1) 0.75", "Student-t5"]
    positions = np.arange(len(SCENARIOS), dtype=float)
    for row_index, operator_index in enumerate(joint.Q_OPERATORS):
        for col_index, time in enumerate(joint.Q_TIMES):
            ax = axes[row_index, col_index]
            subset = [row for row in rows
                      if row["operator_index"] == operator_index and row["time"] == time]
            for method, offset, color in (
                ("delta", -0.09, "#0072B2"),
                ("split_conformal", 0.09, "#D55E00"),
            ):
                for position, scenario in enumerate(SCENARIOS):
                    item = next(row for row in subset if row["scenario"] == scenario)
                    prefix = "delta" if method == "delta" else "conformal"
                    value = item[f"{prefix}_median_upper_over_true"]
                    if value is None:
                        ax.scatter(position + offset, 0.0, marker="x", color=color, s=40)
                    else:
                        ax.scatter(position + offset, value, marker="o" if method == "delta" else "s",
                                   facecolors=color if method == "delta" else "white",
                                   edgecolors=color, label=method.replace("_", " ") if position == 0 else None)
            ax.axhline(1.0, color="#444444", linestyle=":")
            ax.set_yscale("log")
            ax.set_xticks(positions, labels, rotation=18, ha="right")
            ax.set_ylabel("median upper / truth (log scale)")
            ax.set_title(f"bare operator {operator_index}, T={time}")
            ax.grid(alpha=0.2, which="both")
            if row_index == 0 and col_index == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Synthetic interval conservatism by stress scenario")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def run(output: str | Path = DEFAULT_OUT, *,
        panels_per_scenario: int = PANELS_PER_SCENARIO,
        n_blocks: int = N_BLOCKS) -> dict[str, Any]:
    if (isinstance(panels_per_scenario, bool) or not isinstance(panels_per_scenario, int)
            or panels_per_scenario < 20 or panels_per_scenario % 2):
        raise ValueError("panels_per_scenario must be an even integer >= 20")
    if isinstance(n_blocks, bool) or not isinstance(n_blocks, int) or n_blocks <= 63:
        raise ValueError("n_blocks must exceed the 63 raw covariance components")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    split_at = panels_per_scenario // 2
    source_hashes = {
        Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        Path(joint.__file__).name: hashlib.sha256(Path(joint.__file__).read_bytes()).hexdigest(),
    }
    scenario_specs = {}
    for name in SCENARIOS:
        energies, overlaps, rho, description = _scenario(name)
        scenario_specs[name] = {
            "description": description,
            "truth_states": int(energies.size),
            "fit_states": FIT_STATES,
            "true_energies": energies.tolist(),
            "first_gap": float(energies[1] - energies[0]),
            "overlaps_real_imag": [[[float(z.real), float(z.imag)] for z in row]
                                     for row in overlaps],
            "within_panel_ar1_rho": rho,
            "approx_effective_blocks": float(n_blocks * (1.0 - rho) / (1.0 + rho)),
            "innovation_distribution": ("multivariate Student-t df=5 scaled to the declared covariance"
                                        if name == "student_t5_independent" else "multivariate Gaussian"),
        }
    protocol = {
        "schema": "wilson-f-joint-spectral-stress-v1",
        "classification": "synthetic CPU stress diagnostic; no physical Wilson evidence",
        "physical_score": None,
        "fit_model": "three positive forward states with a shared 3x3 Hermitian spectrum",
        "time_slices": list(joint.TIME_SLICES),
        "bare_operator_indices_for_q": list(joint.Q_OPERATORS),
        "q_times": list(joint.Q_TIMES),
        "relative_noise": RELATIVE_NOISE,
        "n_blocks_per_panel": n_blocks,
        "panels_per_scenario": panels_per_scenario,
        "calibration_replicates": f"0:{split_at}",
        "holdout_replicates": f"{split_at}:{panels_per_scenario}",
        "alpha": CALIBRATION_ALPHA,
        "intervals": {
            "delta": "qhat + 1.95996398454 * same-fit delta-method SE; pointwise",
            "split_conformal": "panel-level score (q_true-qhat)/SE calibrated on the disjoint first half; marginal per scenario/operator/time under exchangeable panels",
            "failure_policy": "fit failure or missing calibration value contributes +infinity; nonfinite calibrator quantile abstains",
        },
        "scenarios": scenario_specs,
        "source_sha256": source_hashes,
        "omitted": ["finite-Nt thermal h", "mixed-sector x", "full familywise q/h/x envelope",
                    "force/reference propagation", "production exporter", "authorized physical snapshots"],
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    shutil.copyfile(__file__, out / f"snapshot_{Path(__file__).name}")
    shutil.copyfile(joint.__file__, out / f"snapshot_{Path(joint.__file__).name}")

    panel_rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        energies, overlaps, rho, _ = _scenario(scenario)
        for replicate in range(panels_per_scenario):
            seed = 2026112700 + scenario_index * 10000 + replicate
            blocks = _generate_panel(seed, n_blocks=n_blocks, scenario=scenario)
            try:
                fit = joint.fit_joint_spectrum(blocks, n_states=FIT_STATES)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                fit = {"success": False, "message": str(exc), "q_cells": [], "nfev": 0,
                       "rank": 0, "jacobian_condition_number": None,
                       "reduced_chi_square": None}
            fit_by_key = {(row["operator_index"], row["time"]): row
                          for row in fit.get("q_cells", [])}
            for operator_index in joint.Q_OPERATORS:
                for time in joint.Q_TIMES:
                    cell = fit_by_key.get((operator_index, time))
                    panel_rows.append({
                        "seed": seed, "scenario": scenario, "replicate": replicate,
                        "n_blocks": n_blocks, "within_panel_ar1_rho": rho,
                        "truth_states": int(energies.size), "fit_states": FIT_STATES,
                        "fit_valid": bool(fit.get("success", False)),
                        "fit_message": str(fit.get("message", "")),
                        "fit_nfev": fit.get("nfev", 0), "fit_rank": fit.get("rank", 0),
                        "fit_condition_number": fit.get("jacobian_condition_number"),
                        "reduced_chi_square": fit.get("reduced_chi_square"),
                        "operator_index": operator_index, "time": time,
                        "q_true": _true_q(energies, overlaps, operator_index, time),
                        "q_estimate": None if cell is None else cell["q_estimate"],
                        "q_delta_se": None if cell is None else cell["q_standard_error_delta"],
                        "delta_upper": None if cell is None else cell["q_upper_97_5_pointwise"],
                    })

    summary_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for operator_index in joint.Q_OPERATORS:
            for time in joint.Q_TIMES:
                subset = [row for row in panel_rows
                          if row["scenario"] == scenario
                          and row["operator_index"] == operator_index and row["time"] == time]
                training = [row for row in subset if row["replicate"] < split_at]
                holdout = [row for row in subset if row["replicate"] >= split_at]
                scores: list[float] = []
                calibration_valid = calibration_failed = calibration_abstained = 0
                for row in training:
                    if not row["fit_valid"]:
                        calibration_failed += 1
                        scores.append(math.inf)
                    elif row["q_estimate"] is None or row["q_delta_se"] is None:
                        calibration_abstained += 1
                        scores.append(math.inf)
                    else:
                        estimate, error = float(row["q_estimate"]), float(row["q_delta_se"])
                        if not math.isfinite(estimate) or not math.isfinite(error):
                            calibration_abstained += 1
                            scores.append(math.inf)
                        else:
                            calibration_valid += 1
                            scores.append((float(row["q_true"]) - estimate) / error
                                          if error > 0.0 else math.inf)
                multiplier = joint._split_conformal_upper_multiplier(scores)
                delta_valid = delta_failed = delta_abstained = 0
                conformal_valid = conformal_failed = conformal_abstained = 0
                delta_covered = conformal_covered = 0
                delta_ratios: list[float] = []
                conformal_ratios: list[float] = []
                for row in holdout:
                    if not row["fit_valid"]:
                        delta_failed += 1
                        conformal_failed += 1
                        continue
                    if row["q_estimate"] is None or row["q_delta_se"] is None:
                        delta_abstained += 1
                        conformal_abstained += 1
                        continue
                    estimate = float(row["q_estimate"])
                    error = float(row["q_delta_se"])
                    truth = float(row["q_true"])
                    if not math.isfinite(estimate) or not math.isfinite(error):
                        delta_abstained += 1
                        conformal_abstained += 1
                        continue
                    delta_upper = estimate + joint.Z_UPPER_97_5 * error
                    delta_valid += 1
                    delta_covered += int(delta_upper >= truth)
                    delta_ratios.append(delta_upper / truth)
                    if multiplier is None:
                        conformal_abstained += 1
                    else:
                        conformal_upper = estimate + multiplier * error
                        if math.isfinite(conformal_upper):
                            conformal_valid += 1
                            conformal_covered += int(conformal_upper >= truth)
                            conformal_ratios.append(conformal_upper / truth)
                        else:
                            conformal_abstained += 1
                delta_low, delta_high = (joint._wilson_interval(delta_covered, delta_valid)
                                         if delta_valid else (None, None))
                conformal_low, conformal_high = (
                    joint._wilson_interval(conformal_covered, conformal_valid)
                    if conformal_valid else (None, None)
                )
                summary_rows.append({
                    "scenario": scenario, "operator_index": operator_index, "time": time,
                    "alpha": CALIBRATION_ALPHA,
                    "calibration_planned": len(training), "calibration_valid": calibration_valid,
                    "calibration_failed": calibration_failed,
                    "calibration_abstained": calibration_abstained,
                    "conformal_multiplier": multiplier,
                    "conformal_calibration_status": ("FINITE" if multiplier is not None else "ABSTAIN_NO_FINITE_MULTIPLIER"),
                    "holdout_planned": len(holdout),
                    "delta_valid": delta_valid, "delta_failed": delta_failed,
                    "delta_abstained": delta_abstained,
                    "delta_coverage": delta_covered / delta_valid if delta_valid else None,
                    "delta_mc95_low": delta_low, "delta_mc95_high": delta_high,
                    "delta_median_upper_over_true": float(np.median(delta_ratios)) if delta_ratios else None,
                    "conformal_valid": conformal_valid, "conformal_failed": conformal_failed,
                    "conformal_abstained": conformal_abstained,
                    "conformal_coverage": conformal_covered / conformal_valid if conformal_valid else None,
                    "conformal_mc95_low": conformal_low, "conformal_mc95_high": conformal_high,
                    "conformal_median_upper_over_true": (float(np.median(conformal_ratios))
                                                          if conformal_ratios else None),
                    "classification": "synthetic panel-level stress; not physical/q-h-x familywise coverage",
                })

    with (out / "panel_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(panel_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(panel_rows)
    with (out / "coverage_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    _coverage_plot(summary_rows, out / "stress_coverage.png")
    _conservatism_plot(summary_rows, out / "interval_conservatism.png")

    fit_panel_rows = [row for row in panel_rows
                      if row["operator_index"] == joint.Q_OPERATORS[0]
                      and row["time"] == joint.Q_TIMES[0]]
    fit_valid = sum(bool(row["fit_valid"]) for row in fit_panel_rows)
    fit_failed = len(fit_panel_rows) - fit_valid
    q_planned = len(panel_rows)
    q_failed = sum(not row["fit_valid"] for row in panel_rows)
    q_valid = sum(row["fit_valid"] and row["q_estimate"] is not None for row in panel_rows)
    q_abstained = q_planned - q_valid - q_failed
    validation_planned = sum(row["holdout_planned"] for row in summary_rows)
    delta_valid = sum(row["delta_valid"] for row in summary_rows)
    delta_failed = sum(row["delta_failed"] for row in summary_rows)
    delta_abstained = sum(row["delta_abstained"] for row in summary_rows)
    conformal_valid = sum(row["conformal_valid"] for row in summary_rows)
    conformal_failed = sum(row["conformal_failed"] for row in summary_rows)
    conformal_abstained = sum(row["conformal_abstained"] for row in summary_rows)
    denominators = {
        "synthetic_panels_planned": len(SCENARIOS) * panels_per_scenario,
        "synthetic_panels_valid": fit_valid, "synthetic_panels_failed": fit_failed,
        "synthetic_panels_abstained": 0,
        "q_cells_planned": q_planned, "q_cells_valid": q_valid,
        "q_cells_failed": q_failed, "q_cells_abstained": q_abstained,
        "conformal_strata_planned": len(summary_rows),
        "conformal_strata_finite_multiplier": sum(row["conformal_multiplier"] is not None
                                                   for row in summary_rows),
        "conformal_strata_abstained": sum(row["conformal_multiplier"] is None
                                          for row in summary_rows),
        "holdout_q_cells_planned": validation_planned,
        "delta_holdout_valid": delta_valid, "delta_holdout_failed": delta_failed,
        "delta_holdout_abstained": delta_abstained,
        "conformal_holdout_valid": conformal_valid,
        "conformal_holdout_failed": conformal_failed,
        "conformal_holdout_abstained": conformal_abstained,
        "physical_measurements_planned": 0, "physical_measurements_valid": 0,
        "physical_measurements_failed": 0, "physical_measurements_abstained": 0,
        "certification_planned": 1, "certification_valid": 0,
        "certification_failed": 0, "certification_abstained": 1,
    }
    for prefix in ("q_cells", "delta_holdout", "conformal_holdout"):
        planned = denominators["q_cells_planned"] if prefix == "q_cells" else validation_planned
        if (denominators[f"{prefix}_valid"] + denominators[f"{prefix}_failed"]
                + denominators[f"{prefix}_abstained"] != planned):
            raise RuntimeError(f"{prefix} denominator does not reconcile")
    if fit_valid + fit_failed != denominators["synthetic_panels_planned"]:
        raise RuntimeError("panel fit denominator does not reconcile")
    summary = {
        "schema": "wilson-f-joint-spectral-stress-summary-v1",
        "classification": "synthetic CPU stress diagnostic only",
        "physical_score": None,
        "F_lane_status": "GROUND_STATE_FORCE_UNCERTIFIED",
        "denominators": denominators,
        "scenario_names": list(SCENARIOS),
        "scenario_summaries": summary_rows,
        "source_sha256": source_hashes,
        "files": ["README.md", "protocol.json", "panel_results.csv", "coverage_summary.csv",
                  "stress_coverage.png", "interval_conservatism.png",
                  f"snapshot_{Path(__file__).name}",
                  f"snapshot_{Path(joint.__file__).name}", "summary.json"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    scenario_findings = []
    for scenario in SCENARIOS:
        scenario_rows = [row for row in summary_rows if row["scenario"] == scenario]
        finding_parts = []
        for method in ("delta", "conformal"):
            prefix = "delta" if method == "delta" else "conformal"
            covered_rows = [row for row in scenario_rows if row[f"{prefix}_coverage"] is not None]
            coverage_values = [float(row[f"{prefix}_coverage"]) for row in covered_rows]
            ratio_values = [float(row[f"{prefix}_median_upper_over_true"])
                            for row in covered_rows
                            if row[f"{prefix}_median_upper_over_true"] is not None]
            below_target = sum(float(row[f"{prefix}_mc95_high"]) < 0.975
                               for row in covered_rows
                               if row[f"{prefix}_mc95_high"] is not None)
            coverage_text = (f"{min(coverage_values):.3f}–{max(coverage_values):.3f}"
                             if coverage_values else "no valid intervals")
            ratio_text = (f"{min(ratio_values):.3f}–{max(ratio_values):.3f}"
                          if ratio_values else "unavailable")
            finding_parts.append(
                f"{method} coverage `{coverage_text}`, median upper/truth `{ratio_text}`, "
                f"Wilson-MC upper below target `{below_target}/{len(scenario_rows)}`"
            )
        scenario_findings.append(f"- `{scenario}`: " + "; ".join(finding_parts) + ".")
    lines = [
        "# Wilson F-lane joint-spectrum stress report",
        "",
        "**Synthetic CPU diagnostic only.** No Wilson measurement, sealed input, GPU or QUDA data were read. `physical_score=null`; F remains `GROUND_STATE_FORCE_UNCERTIFIED`.",
        "",
        "## Design and fixed run conditions",
        "",
        f"The run has {len(SCENARIOS)} truth scenarios, `{panels_per_scenario}` independent panels per scenario, `{n_blocks}` blocks per panel, relative noise `{RELATIVE_NOISE}`, and a fixed three-state GLS fit over `t=1..7`. Bare q cells are operators `{joint.Q_OPERATORS}` at `T={joint.Q_TIMES}`. The first `{split_at}` panel IDs in each scenario calibrate the one-sided split-conformal score; the remaining `{panels_per_scenario - split_at}` are holdout. The two intervals compared are pointwise `qhat+1.95996*SE_delta` and split-conformal `qhat+c*SE_delta`.",
        "",
        "The stress cases are: matched three-state independent Gaussian blocks; a four-state truth fitted with three states; a near-degenerate first gap `E1-E0=0.025`; a small bare ground overlap `|Z0(operator 2)|=0.04`; stationary AR(1) block dependence `rho=0.75`; and independent multivariate Student-t innovations with 5 degrees of freedom, rescaled to the same covariance. The AR(1) case has approximate effective block count `N(1-rho)/(1+rho)`; the GLS standard error still uses the existing iid sample-covariance rule, so this is an intentional stress of that assumption.",
        "",
        "## Results",
        "",
        "![Coverage by stress scenario](stress_coverage.png)",
        "",
        "![Interval conservatism by stress scenario](interval_conservatism.png)",
        "",
        f"Synthetic panel planned/valid/failed/abstained = `{denominators['synthetic_panels_planned']}/{fit_valid}/{fit_failed}/0`; q-cell planned/valid/failed/abstained = `{q_planned}/{q_valid}/{q_failed}/{q_abstained}`. Conformal strata planned/finite/abstained = `{denominators['conformal_strata_planned']}/{denominators['conformal_strata_finite_multiplier']}/{denominators['conformal_strata_abstained']}`. Holdout q cells planned = `{validation_planned}`; delta valid/failed/abstained = `{delta_valid}/{delta_failed}/{delta_abstained}`; conformal valid/failed/abstained = `{conformal_valid}/{conformal_failed}/{conformal_abstained}`. Cell-wise 95% Wilson Monte Carlo intervals and median upper/truth ratios are in `coverage_summary.csv`.",
        "",
        "Scenario-wise findings across the four bare-operator/time q cells:",
        *scenario_findings,
        "",
        "Interpret coverage only within each exact synthetic scenario and q cell. At 40 holdout panels, Wilson intervals remain broad; split conformal is marginal across exchangeable synthetic panels and does not guarantee each finite holdout stratum will hit 97.5%. A finite calibrator in this design is not a physical calibration receipt.",
        "",
        "## Data flow and limits",
        "",
        "Text flow: `fixed truth scenario -> 96 same-panel correlated matrix blocks -> fit three-state full-covariance GLS -> compute q estimate/delta SE -> calibrate only on panel IDs 0:40 -> evaluate on disjoint IDs 40:80 -> preserve fit, calibration, holdout and abstention denominators -> certification abstains`.",
        "",
        "This tests only interval behavior for controlled positive spectra. It does not validate the assumed positive spectral model for production Wilson loops, finite-`N_t` thermal `h`, mixed-sector `x`, force/reference propagation, chain receipts, or simultaneous familywise q/h/x coverage. The generator uses independent panels; dependence is within each AR(1) panel, while Student-t tails are independent across blocks. Full state profile likelihood, moving-block bootstrap, model-averaged intervals, and q/h/x familywise calibration remain open software research. No output from this report may enter `spectral_q_bound_reader` or the force gate.",
        "",
        "Next software work, if the chosen interval is still under study: add a moving-block bootstrap/profile-likelihood comparator and report its failed-fit and interval-abstention denominators under the same six scenarios. Production work still requires reviewed definitions, authorized immutable calibration bytes and independent contamination receipts. Keep `NOT_READY/UNCERTIFIED`, `GROUND_STATE_FORCE_UNCERTIFIED`, and `physical_score=null`.",
        "",
        "Reproduce with `PYTHONPATH=. python -B -m tasks.wilson_loop.f_lane_joint_spectral_stress_diagnostic --output tasks/wilson_loop/f_lane_joint_spectral_stress_20261126_v1 --panels-per-scenario 80 --n-blocks 96`. Source/artifact hashes and full scenario values are recorded in `protocol.json`, `summary.json`, and `artifact_hashes.json`.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    output_files = [out / name for name in summary["files"]]
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in output_files if path.is_file() and path.name != "artifact_hashes.json"}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--panels-per-scenario", type=int, default=PANELS_PER_SCENARIO)
    parser.add_argument("--n-blocks", type=int, default=N_BLOCKS)
    args = parser.parse_args()
    result = run(args.output, panels_per_scenario=args.panels_per_scenario,
                 n_blocks=args.n_blocks)
    print(json.dumps({"output": str(args.output), "denominators": result["denominators"]}, indent=2))


if __name__ == "__main__":
    main()
