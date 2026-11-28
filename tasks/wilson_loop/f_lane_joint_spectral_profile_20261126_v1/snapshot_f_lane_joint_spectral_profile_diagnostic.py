"""Compare a plug-in profile-likelihood q upper with synthetic GLS bounds."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize

from tasks.wilson_loop import f_lane_joint_spectral_gls_diagnostic as joint
from tasks.wilson_loop import f_lane_joint_spectral_mbb_b80_diagnostic as b80
from tasks.wilson_loop import f_lane_joint_spectral_mbb_diagnostic as mbb
from tasks.wilson_loop import f_lane_joint_spectral_stress_diagnostic as stress


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_B80 = TASK_DIR / "f_lane_joint_spectral_mbb_b80_20261126_v1"
DEFAULT_OUT = TASK_DIR / "f_lane_joint_spectral_profile_20261126_v1"
PROFILE_DELTA_CHI2 = 3.841458820694124
Q_KEYS = tuple((operator, time) for operator in joint.Q_OPERATORS for time in joint.Q_TIMES)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_jacobian(parameters: np.ndarray, times: tuple[int, ...],
                    n_states: int) -> np.ndarray:
    """Analytic Jacobian of the packed Hermitian spectral model."""
    energies, overlaps = joint._unpack_parameters(parameters, n_states=n_states)
    jacobian = np.zeros((9 * len(times), parameters.size), dtype=float)

    def pack(matrix: np.ndarray) -> np.ndarray:
        return np.asarray([
            matrix[0, 0].real, matrix[1, 1].real, matrix[2, 2].real,
            matrix[0, 1].real, matrix[0, 1].imag,
            matrix[0, 2].real, matrix[0, 2].imag,
            matrix[1, 2].real, matrix[1, 2].imag,
        ], dtype=float)

    for time_index, time in enumerate(times):
        weights = np.exp(-energies * time)
        terms = weights[:, None, None] * overlaps[:, :, None] * overlaps[:, None, :].conj()
        rows = slice(9 * time_index, 9 * (time_index + 1))
        for energy_parameter in range(n_states):
            jacobian[rows, energy_parameter] = pack(
                -time * np.sum(terms[energy_parameter:], axis=0)
            )
        for state in range(n_states):
            start = n_states + 5 * state
            for operator_index in range(3):
                basis = np.zeros(3, dtype=complex)
                basis[operator_index] = 1.0
                components = ((start, basis),)
                if operator_index in (1, 2):
                    components = (
                        (start + (1 if operator_index == 1 else 3), basis),
                        (start + (2 if operator_index == 1 else 4), 1j * basis),
                    )
                for parameter_index, dz in components:
                    derivative = weights[state] * (
                        np.outer(dz, overlaps[state].conj())
                        + np.outer(overlaps[state], dz.conj())
                    )
                    jacobian[rows, parameter_index] = pack(derivative)
    return jacobian


def _ground_real_index(n_states: int, operator_index: int) -> int:
    offsets = {0: 0, 1: 1, 2: 3}
    if n_states not in (2, 3) or operator_index not in offsets:
        raise ValueError("profile parameterization requires 2/3 states and operator index 0..2")
    return n_states + offsets[operator_index]


def _profile_vector(parameters: object, q_target: float, operator_index: int) -> np.ndarray:
    p = np.asarray(parameters, dtype=float)
    n_states = p.size // 6 if p.ndim == 1 and p.size % 6 == 0 else 0
    real_index = _ground_real_index(n_states, operator_index)
    if not math.isfinite(q_target) or q_target <= 0.0 or not np.isfinite(p).all():
        raise ValueError("q_target and fit parameters must be finite and q_target positive")
    keep = [i for i in range(p.size) if i != real_index]
    values = p[keep].copy()
    if operator_index in (1, 2):
        imag_index = real_index + 1
        z_ground = p[real_index] + 1j * p[imag_index]
        values[keep.index(imag_index)] = np.angle(z_ground)
    return np.concatenate(([q_target], values))


def _parameters_at_profile_q(profile_parameters: object, operator_index: int,
                              time: int, n_states: int) -> np.ndarray:
    y = np.asarray(profile_parameters, dtype=float)
    if y.shape != (6 * n_states,) or not np.isfinite(y).all():
        raise ValueError("profile parameter vector has the wrong shape or non-finite values")
    real_index = _ground_real_index(n_states, operator_index)
    keep = [i for i in range(6 * n_states) if i != real_index]
    full = np.empty(6 * n_states, dtype=float)
    full[keep] = y[1:]
    q_target = float(y[0])
    if not math.isfinite(q_target) or q_target <= 0.0:
        raise ValueError("profile q must be finite and positive")
    energies, overlaps = joint._unpack_parameters(full, n_states=n_states)
    numerator = float(np.sum(
        np.abs(overlaps[1:, operator_index]) ** 2
        * np.exp(-(energies[1:] - energies[0]) * time)
    ))
    if not math.isfinite(numerator) or numerator <= 0.0:
        raise ValueError("profile q numerator is zero or non-finite")
    magnitude = math.sqrt(numerator / q_target)
    if operator_index == 0:
        full[real_index] = magnitude
    else:
        imag_index = real_index + 1
        phase = full[imag_index]
        full[real_index] = magnitude * math.cos(phase)
        full[imag_index] = magnitude * math.sin(phase)
    return full


def _profile_transform_jacobian(profile_parameters: object, operator_index: int,
                                time: int, n_states: int) -> np.ndarray:
    """Return d(full fit parameters)/d(profile coordinates), including q's numerator."""
    y = np.asarray(profile_parameters, dtype=float)
    full = _parameters_at_profile_q(y, operator_index, time, n_states)
    real_index = _ground_real_index(n_states, operator_index)
    keep = [index for index in range(6 * n_states) if index != real_index]
    energies, overlaps = joint._unpack_parameters(full, n_states=n_states)
    gaps = energies[1:] - energies[0]
    exponentials = np.exp(-gaps * time)
    excited = overlaps[1:, operator_index]
    terms = np.abs(excited) ** 2 * exponentials
    numerator = float(np.sum(terms))
    magnitude = math.sqrt(numerator / float(y[0]))
    dnumerator = np.zeros(6 * n_states, dtype=float)
    for gap_parameter in range(1, n_states):
        dnumerator[gap_parameter] = -time * float(np.sum(terms[gap_parameter - 1:]))
    for excited_state in range(1, n_states):
        decay = math.exp(-float(gaps[excited_state - 1]) * time)
        start = n_states + 5 * excited_state
        if operator_index == 0:
            dnumerator[start] = 2.0 * float(full[start]) * decay
        else:
            real_offset = 1 if operator_index == 1 else 3
            real_component = start + real_offset
            imag_component = real_component + 1
            dnumerator[real_component] = 2.0 * float(full[real_component]) * decay
            dnumerator[imag_component] = 2.0 * float(full[imag_component]) * decay

    transform = np.zeros((6 * n_states, y.size), dtype=float)
    transform[real_index, 0] = -0.5 * full[real_index] / float(y[0])
    phase_index = real_index + 1 if operator_index in (1, 2) else None
    if phase_index is not None:
        transform[phase_index, 0] = -0.5 * full[phase_index] / float(y[0])
        phase = math.atan2(full[phase_index], full[real_index])
    for column, parameter_index in enumerate(keep, start=1):
        if phase_index is not None and parameter_index == phase_index:
            transform[real_index, column] = -magnitude * math.sin(phase)
            transform[phase_index, column] = magnitude * math.cos(phase)
            continue
        transform[parameter_index, column] = 1.0
        dmag = 0.5 * magnitude * dnumerator[parameter_index] / numerator
        if operator_index == 0:
            transform[real_index, column] = dmag
        else:
            transform[real_index, column] = dmag * math.cos(phase)
            transform[phase_index, column] = dmag * math.sin(phase)
    return transform


def _read_verified_b80(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = root / "artifact_hashes.json"
    for path in (manifest_path, root / "protocol.json", root / "summary.json"):
        if not path.is_file():
            raise FileNotFoundError(f"verified B=80 report is incomplete: {path}")
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest.items():
        path = root / name
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"B=80 artifact hash mismatch: {name}")
    protocol = json.loads((root / "protocol.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if (protocol.get("schema") != "wilson-f-joint-spectral-mbb-b80-v1"
            or int(protocol.get("total_bootstrap_replicates", -1)) != 80
            or summary.get("physical_score") is not None):
        raise ValueError("input bundle is not the expected B=80 synthetic report")
    sources = {
        Path(b80.__file__).name: _sha256(Path(b80.__file__)),
        Path(mbb.__file__).name: _sha256(Path(mbb.__file__)),
        Path(stress.__file__).name: _sha256(Path(stress.__file__)),
        Path(joint.__file__).name: _sha256(Path(joint.__file__)),
    }
    if protocol.get("source_sha256") != sources:
        raise ValueError("B=80 report source snapshot does not match current code")
    with (root / "b80_intervals.csv").open(newline="") as stream:
        intervals = list(csv.DictReader(stream))
    if len(intervals) != int(summary["denominators"]["b80_intervals_planned"]):
        raise ValueError("B=80 interval rows do not match their denominator")
    return protocol, intervals


def _profile_upper(blocks: np.ndarray, operator_index: int, time: int,
                   max_iterations: int = 1000) -> dict[str, Any]:
    fit = joint.fit_joint_spectrum(blocks, n_states=stress.FIT_STATES)
    if not fit.get("success"):
        return {"status": "FAILED_BASE_FIT", "q_estimate": None,
                "profile_upper": None, "chi2_min": None, "chi2_at_upper": None,
                "chi2_delta": None, "optimizer_success": False,
                "optimizer_iterations": 0, "boundary_hits": [],
                "message": str(fit.get("message", "base fit failed"))}

    n_states = stress.FIT_STATES
    parameters = np.asarray(fit["parameters"], dtype=float)
    mean = np.asarray(blocks, dtype=float).mean(axis=0)
    covariance, covariance_diagnostics = joint._regularized_covariance(blocks)
    covariance_cholesky = cholesky(covariance, lower=True, check_finite=True)

    def residual(full_parameters: np.ndarray) -> np.ndarray:
        raw = joint._model_vector(full_parameters, n_states=n_states) - mean
        return solve_triangular(covariance_cholesky, raw, lower=True,
                                check_finite=True)

    def chi_square(profile_parameters: np.ndarray) -> float:
        full = _parameters_at_profile_q(profile_parameters, operator_index, time, n_states)
        r = residual(full)
        return float(r @ r)

    q_hat = joint._q_value(parameters, operator_index, time, n_states=n_states)
    q_cell = next(row for row in fit["q_cells"]
                  if row["operator_index"] == operator_index and row["time"] == time)
    q_se = float(q_cell["q_standard_error_delta"])
    chi2_min = float(np.dot(residual(parameters), residual(parameters)))
    y0 = _profile_vector(parameters, q_hat, operator_index)
    q_start = max(q_hat + joint.Z_UPPER_97_5 * q_se, q_hat * (1.0 + 1e-5), q_hat + 1e-8)
    y0[0] = q_start
    start_chi2 = chi_square(y0)
    # Back off the delta-method start if the nonlinear profile is already outside the LR set.
    while (start_chi2 - chi2_min > PROFILE_DELTA_CHI2
           and y0[0] > q_hat * (1.0 + 1e-7)):
        y0[0] = 0.5 * (y0[0] + q_hat)
        start_chi2 = chi_square(y0)
    if (not math.isfinite(start_chi2)
            or start_chi2 - chi2_min > PROFILE_DELTA_CHI2 + 1e-5):
        return {"status": "FAILED_NO_FEASIBLE_START", "q_estimate": q_hat,
                "profile_upper": None, "chi2_min": chi2_min,
                "chi2_at_upper": None, "chi2_delta": None,
                "optimizer_success": False, "optimizer_iterations": 0,
                "boundary_hits": [], "message": "could not initialize within LR set"}

    n_profile = y0.size
    def constraint(y: np.ndarray) -> float:
        return PROFILE_DELTA_CHI2 - (chi_square(y) - chi2_min)

    def constraint_jacobian(y: np.ndarray) -> np.ndarray:
        full = _parameters_at_profile_q(y, operator_index, time, n_states)
        residual_value = residual(full)
        full_jacobian = solve_triangular(
            covariance_cholesky,
            _model_jacobian(full, tuple(joint.TIME_SLICES), n_states),
            lower=True, check_finite=True,
        )
        transform = _profile_transform_jacobian(y, operator_index, time, n_states)
        profile_jacobian = full_jacobian @ transform
        return -2.0 * (profile_jacobian.T @ residual_value)

    # The first profile coordinate is q itself; remaining coordinates are the
    # original fit parameters with one ground-state magnitude eliminated.
    full_lower = np.full(parameters.size, -np.inf)
    full_upper = np.full(parameters.size, np.inf)
    full_lower[:n_states] = [0.005] + [0.002] * (n_states - 1)
    full_upper[:n_states] = 2.0
    for state in range(n_states):
        full_lower[n_states + 5 * state] = 1e-8
    real_index = _ground_real_index(n_states, operator_index)
    keep = [index for index in range(parameters.size) if index != real_index]
    bounds = [(q_hat, None)]
    for index in keep:
        if operator_index in (1, 2) and index == real_index + 1:
            bounds.append((None, None))  # this stored component is the overlap phase
        else:
            low = None if not math.isfinite(full_lower[index]) else float(full_lower[index])
            high = None if not math.isfinite(full_upper[index]) else float(full_upper[index])
            bounds.append((low, high))
    result = minimize(
        lambda y: -float(y[0]), y0,
        jac=lambda y: np.r_[-1.0, np.zeros(n_profile - 1)],
        method="SLSQP", bounds=bounds,
        constraints=[{"type": "ineq", "fun": constraint, "jac": constraint_jacobian}],
        options={"ftol": 1e-8, "maxiter": max_iterations},
    )
    try:
        full_at_upper = _parameters_at_profile_q(result.x, operator_index, time, n_states)
        chi2_upper = float(np.dot(residual(full_at_upper), residual(full_at_upper)))
        q_upper = float(result.x[0])
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return {"status": "FAILED_PROFILE_DECODE", "q_estimate": q_hat,
                "profile_upper": None, "chi2_min": chi2_min,
                "chi2_at_upper": None, "chi2_delta": None,
                "optimizer_success": bool(result.success),
                "optimizer_iterations": int(getattr(result, "nit", 0)),
                "boundary_hits": [], "message": str(exc)}

    tolerance = 1e-5
    boundary_hits = []
    for index, bound in enumerate(full_lower):
        if math.isfinite(bound) and full_at_upper[index] - bound <= tolerance * max(1.0, abs(bound)):
            boundary_hits.append(f"parameter[{index}].lower")
    for index, bound in enumerate(full_upper):
        if math.isfinite(bound) and bound - full_at_upper[index] <= tolerance * max(1.0, abs(bound)):
            boundary_hits.append(f"parameter[{index}].upper")
    q_reconstructed = joint._q_value(full_at_upper, operator_index, time, n_states=n_states)
    feasible = (math.isfinite(chi2_upper)
                and chi2_upper - chi2_min <= PROFILE_DELTA_CHI2 + 1e-4
                and abs(q_reconstructed - q_upper) <= 1e-5 * max(1.0, q_upper))
    if not result.success or not feasible:
        status = "FAILED_PROFILE_OPTIMIZER"
    elif boundary_hits:
        status = "ABSTAIN_PARAMETER_BOUNDARY"
    else:
        status = "VALID"
    return {
        "status": status, "q_estimate": q_hat,
        "profile_upper": q_upper if status == "VALID" else None,
        "chi2_min": chi2_min, "chi2_at_upper": chi2_upper,
        "chi2_delta": chi2_upper - chi2_min,
        "optimizer_success": bool(result.success),
        "optimizer_iterations": int(getattr(result, "nit", 0)),
        "boundary_hits": boundary_hits,
        "covariance_condition_number": covariance_diagnostics["condition_number"],
        "message": str(result.message),
    }


def _plot(rows: list[dict[str, Any]], path: Path,
          scenarios: tuple[str, ...]) -> None:
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
        ("moving_block_B80", 0.10, "#009E73", "^"),
        ("profile_likelihood", 0.30, "#CC79A7", "D"),
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
            ax.set_ylabel("conditional pointwise coverage (95% Wilson MC interval)")
            ax.set_title(f"bare operator {operator_index}, T={time}")
            ax.grid(alpha=0.2)
            if row_index == 0 and col_index == 0:
                handles, _ = ax.get_legend_handles_labels()
                handles.append(Line2D(
                    [], [], marker="x", color="#444444", linestyle="none",
                    label="no valid intervals (marker at 0.5 is placeholder)",
                ))
                ax.legend(handles=handles, fontsize=8)
    fig.suptitle(
        "Synthetic profile-likelihood pilot; x marks no valid intervals (y=0.5 placeholder)"
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _refresh_derived_report(output: str | Path) -> dict[str, Any]:
    """Refresh plot, explanatory README, source snapshots and hashes without refitting."""
    out = Path(output)
    for name in ("summary.json", "protocol.json", "profile_intervals.csv",
                 "method_comparison.csv", "artifact_hashes.json"):
        if not (out / name).is_file():
            raise FileNotFoundError(f"profile report is incomplete: {out / name}")

    summary = json.loads((out / "summary.json").read_text())
    protocol = json.loads((out / "protocol.json").read_text())
    with (out / "profile_intervals.csv").open(newline="") as stream:
        profile_rows = list(csv.DictReader(stream))
    with (out / "method_comparison.csv").open(newline="") as stream:
        comparison_rows = list(csv.DictReader(stream))
    planned = int(summary["denominators"]["profile_intervals_planned"])
    if len(profile_rows) != planned:
        raise ValueError("profile rows do not match their frozen denominator")

    module_paths = (Path(__file__), Path(b80.__file__), Path(mbb.__file__),
                    Path(stress.__file__), Path(joint.__file__))
    source_hashes = {module_path.name: _sha256(module_path)
                     for module_path in module_paths}
    for module_path in module_paths:
        shutil.copyfile(module_path, out / f"snapshot_{module_path.name}")
    protocol["source_sha256"] = source_hashes
    protocol["optimizer"] = (
        "SLSQP constrained to delta_chi2 <= chi2_1(0.95), using analytic model, "
        "profile-transform and constraint Jacobians; unsuccessful/infeasible runs "
        "are FAILED, successful fit-domain boundary hits are ABSTAIN"
    )
    protocol["derived_report_refresh"] = (
        "plot, README and source/artifact hashes refreshed from frozen interval rows; "
        "no fit or resampling was rerun"
    )
    summary["source_sha256"] = source_hashes

    _plot(summary["comparison_rows"], out / "method_coverage.png",
          tuple(summary["scenarios"]))
    profile_status = {scenario: Counter() for scenario in summary["scenarios"]}
    for row in profile_rows:
        profile_status[row["scenario"]][row["status"]] += 1
    cells = {}
    for row in comparison_rows:
        if row["method"] != "profile_likelihood":
            continue
        key = (row["scenario"], int(row["operator_index"]), int(row["time"]))
        cells[key] = row
    panels_per_cell = planned // (len(summary["scenarios"]) * len(Q_KEYS))
    profile_cell_lines = []
    for scenario in summary["scenarios"]:
        cell_text = []
        for operator_index, time in Q_KEYS:
            row = cells[(scenario, operator_index, time)]
            coverage = row["coverage"]
            if coverage in ("", None):
                result = "no valid interval"
            else:
                result = (f"{float(coverage):.3f} "
                          f"({int(row['valid'])}/{panels_per_cell} valid)")
            cell_text.append(f"op{operator_index},T{time}: {result}")
        counts = profile_status[scenario]
        profile_cell_lines.append(
            f"| `{scenario}` | {counts.get('VALID', 0)} | "
            f"{counts.get('FAILED_PROFILE_OPTIMIZER', 0)} | "
            f"{counts.get('ABSTAIN_PARAMETER_BOUNDARY', 0)} | "
            f"{'<br>'.join(cell_text)} |"
        )
    failures = Counter(row["message"] for row in profile_rows
                       if row["status"].startswith("FAILED"))
    boundary_failures = sum(
        bool(row["boundary_hits"]) and row["status"].startswith("FAILED")
        for row in profile_rows
    )
    iteration_counts = [int(row["optimizer_iterations"]) for row in profile_rows]
    denominators = summary["denominators"]
    readme = [
        "# Wilson F-lane profile-likelihood upper-bound pilot", "",
        "**Synthetic CPU method diagnostic only.** It reuses the verified six-scenario GLS stress inputs and B=80 report for comparisons. No Wilson measurement, sealed split, GPU or QUDA data were read. `physical_score=null`; F remains `GROUND_STATE_FORCE_UNCERTIFIED`.", "",
        "## Fixed method", "",
        f"The same first-40 calibration / next-{panels_per_cell}-panel holdout design is used. For each panel, a three-state GLS fit estimates the full regularized covariance once; the profile keeps that covariance fixed and maximizes q under `chi2(q)-chi2_min <= {PROFILE_DELTA_CHI2:.12f}`, the plug-in one-degree-of-freedom chi-square cutoff corresponding to a two-sided 95% likelihood-ratio set / one-sided 97.5% endpoint under regular Wilks conditions. For bare operator 0 the ground overlap is eliminated algebraically; for operator 2 its complex phase is retained and magnitude reconstructed. SLSQP uses analytic model, profile-transform and constraint Jacobians. Failed or infeasible optimizer results are `FAILED`; a successful solution hitting a fit-domain parameter boundary is `ABSTAIN`.", "",
        "![Profile, B=80, conformal and delta coverage](method_coverage.png)", "",
        "In the figure, an `x` marks a cell with no valid intervals; its y-position at 0.5 is only a plotting placeholder, not measured 50% coverage.", "",
        f"Profile interval denominators planned/valid/failed/abstained = `{denominators['profile_intervals_planned']}/{denominators['profile_intervals_valid']}/{denominators['profile_intervals_failed']}/{denominators['profile_intervals_abstained']}`. The {denominators['holdout_panels_valid']} holdout panels all generated, but only {denominators['profile_intervals_valid']}/{planned} profile limits were valid. Of {denominators['profile_intervals_failed']} optimizer failures, messages were `{failures.get('Iteration limit reached', 0)} iteration-limit`, `{failures.get('Positive directional derivative for linesearch', 0)} positive-directional-derivative`, `{failures.get('Inequality constraints incompatible', 0)} incompatible-constraint`, and `{failures.get('Singular matrix E in LSQ subproblem', 0)} singular-subproblem`; {boundary_failures} failed runs also touched at least one parameter bound. Iteration count median/max was `{statistics.median(iteration_counts):.0f}/{max(iteration_counts)}`. Failure status takes precedence over the boundary counter, hence zero separately classified abstentions.", "",
        "## Per-scenario denominators and conditional pointwise coverage", "",
        "Each coverage value is conditional on valid intervals only; the valid count is shown beside it, out of the planned number per operator/time cell. It is not an unconditional coverage estimate when optimizer failures are common.", "",
        "| Scenario | valid | failed | abstained | profile coverage by cell |\n|---|---:|---:|---:|---|\n" + "\n".join(profile_cell_lines), "",
        "The raw rows include pointwise Wilson Monte Carlo intervals, optimizer status, chi-square displacement, boundary hits and interval widths in `profile_intervals.csv` and `method_comparison.csv`.", "",
        "## Limits", "",
        "This is not a finite-sample guarantee. The covariance is estimated then treated as fixed, the profile optimizer is local, the likelihood-ratio cutoff assumes regularity, and near-zero overlaps or active bounds violate those conditions. Omitted-state bias, AR(1) block dependence, finite-Nt thermal h, mixed x and familywise q/h/x control remain outside the profile. Do not judge coverage without the failure denominator. No interval may enter the spectral q reader or F gate.", "",
        "Text flow: `verify B80/stress hashes -> hold same synthetic panels and per-panel covariance -> fit three-state GLS -> eliminate selected ground-overlap magnitude using q -> maximize q inside fixed chi-square LR set -> retain optimizer failures/bound hits in P/V/F/A -> compare conditional coverage -> physical certification abstains`.", "",
        "Reproduce the fits with `PYTHONPATH=. python -B -m tasks.wilson_loop.f_lane_joint_spectral_profile_diagnostic --output tasks/wilson_loop/f_lane_joint_spectral_profile_20261126_v1`. To refresh only plot, explanation and hashes from frozen CSV rows without rerunning fits, add `--refresh-derived`. Source/artifact hashes are recorded in the report.",
    ]
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (out / "README.md").write_text("\n".join(readme) + "\n")
    artifact_hashes = {
        name: _sha256(out / name) for name in summary["files"]
        if (out / name).is_file()
    }
    (out / "artifact_hashes.json").write_text(
        json.dumps(artifact_hashes, indent=2, allow_nan=False) + "\n"
    )
    return summary


def run(output: str | Path = DEFAULT_OUT, *, b80_root: str | Path = DEFAULT_B80,
        outer_panels_per_scenario: int = 40,
        scenarios: tuple[str, ...] = stress.SCENARIOS,
        max_iterations: int = 1000,
        report_progress: bool = False) -> dict[str, Any]:
    if (isinstance(outer_panels_per_scenario, bool)
            or not isinstance(outer_panels_per_scenario, int)
            or not 1 <= outer_panels_per_scenario <= 40):
        raise ValueError("outer_panels_per_scenario must be in [1,40]")
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
            or max_iterations < 1):
        raise ValueError("max_iterations must be a positive integer")
    if (not scenarios or len(set(scenarios)) != len(scenarios)
            or any(s not in stress.SCENARIOS for s in scenarios)):
        raise ValueError("scenarios must be unique names from the verified stress report")
    b80_protocol, b80_intervals = _read_verified_b80(Path(b80_root))
    stress_protocol, stress_rows = mbb._read_stress_bundle()
    split_at = int(stress_protocol["panels_per_scenario"]) // 2
    if split_at != 40 or outer_panels_per_scenario > split_at:
        raise ValueError("stress bundle calibration/holdout split changed")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)

    b80_map = {(r["scenario"], int(r["outer_replicate"]), int(r["operator_index"]), int(r["time"])): r
               for r in b80_intervals}
    panel_map = {(r["scenario"], int(r["replicate"]), int(r["operator_index"]), int(r["time"])): r
                 for r in stress_rows}
    multipliers, calibration_receipts = mbb._calibration_multipliers(stress_rows, scenarios)
    scenario_index = {scenario: index for index, scenario in enumerate(stress.SCENARIOS)}

    profile_rows: list[dict[str, Any]] = []
    planned = len(scenarios) * outer_panels_per_scenario * len(Q_KEYS)
    for scenario in scenarios:
        for local_index in range(outer_panels_per_scenario):
            replicate = split_at + local_index
            seed = 2026112700 + scenario_index[scenario] * 10000 + replicate
            blocks = stress._generate_panel(seed, n_blocks=stress.N_BLOCKS, scenario=scenario)
            for operator_index, time in Q_KEYS:
                result = _profile_upper(blocks, operator_index, time,
                                        max_iterations=max_iterations)
                base = panel_map[(scenario, replicate, operator_index, time)]
                profile_rows.append({
                    "scenario": scenario, "outer_replicate": replicate,
                    "seed": seed, "operator_index": operator_index, "time": time,
                    "q_true": float(base["q_true"]),
                    "q_estimate": result.get("q_estimate"),
                    "profile_upper": result.get("profile_upper"),
                    "chi2_min": result.get("chi2_min"),
                    "chi2_at_upper": result.get("chi2_at_upper"),
                    "chi2_delta": result.get("chi2_delta"),
                    "optimizer_success": result.get("optimizer_success"),
                    "optimizer_iterations": result.get("optimizer_iterations"),
                    "covariance_condition_number": result.get("covariance_condition_number"),
                    "boundary_hits": ";".join(result.get("boundary_hits", [])),
                    "status": result["status"], "message": result.get("message", ""),
                })
            if report_progress and (local_index + 1) % 5 == 0:
                print(f"profile scenario={scenario} panels={local_index + 1}/{outer_panels_per_scenario}",
                      flush=True)

    methods = ("delta", "split_conformal", "moving_block_B80", "profile_likelihood")
    comparison_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for operator_index, time in Q_KEYS:
            values_by_method: dict[str, list[tuple[float, float]]] = {method: [] for method in methods}
            failed = {method: 0 for method in methods}
            abstained = {method: 0 for method in methods}
            for local_index in range(outer_panels_per_scenario):
                replicate = split_at + local_index
                base = panel_map[(scenario, replicate, operator_index, time)]
                profile = next(r for r in profile_rows if r["scenario"] == scenario
                               and r["outer_replicate"] == replicate
                               and r["operator_index"] == operator_index and r["time"] == time)
                if not str(base["fit_valid"]).lower() == "true":
                    for method in methods:
                        failed[method] += 1
                    continue
                truth = float(base["q_true"])
                estimate = float(base["q_estimate"])
                error = float(base["q_delta_se"])
                values_by_method["delta"].append((truth, estimate + joint.Z_UPPER_97_5 * error))
                multiplier = multipliers[(scenario, operator_index, time)]
                if multiplier is None:
                    abstained["split_conformal"] += 1
                else:
                    values_by_method["split_conformal"].append((truth, estimate + multiplier * error))
                b80_row = b80_map[(scenario, replicate, operator_index, time)]
                b80_upper = mbb._float_or_none(b80_row["bootstrap_upper"])
                if b80_upper is None:
                    abstained["moving_block_B80"] += 1
                else:
                    values_by_method["moving_block_B80"].append((truth, b80_upper))
                if profile["status"] == "VALID":
                    values_by_method["profile_likelihood"].append((truth, float(profile["profile_upper"])))
                elif profile["status"].startswith("FAILED"):
                    failed["profile_likelihood"] += 1
                else:
                    abstained["profile_likelihood"] += 1
            for method in methods:
                values = values_by_method[method]
                valid, failed_n, abstained_n = len(values), failed[method], abstained[method]
                if valid + failed_n + abstained_n != outer_panels_per_scenario:
                    raise RuntimeError(f"{method} denominator does not reconcile for {scenario}")
                covered = sum(upper >= truth for truth, upper in values)
                low, high = mbb._wilson_interval(covered, valid)
                ratios = [upper / truth for truth, upper in values]
                comparison_rows.append({
                    "scenario": scenario, "operator_index": operator_index, "time": time,
                    "method": method, "planned": outer_panels_per_scenario,
                    "valid": valid, "failed": failed_n, "abstained": abstained_n,
                    "coverage": covered / valid if valid else None,
                    "mc95_low": low, "mc95_high": high,
                    "median_upper_over_true": float(np.median(ratios)) if ratios else None,
                    "classification": "synthetic pointwise conditional coverage; LR/profile assumptions are not certified",
                })

    valid_profile = sum(row["status"] == "VALID" for row in profile_rows)
    failed_profile = sum(row["status"].startswith("FAILED") for row in profile_rows)
    abstained_profile = planned - valid_profile - failed_profile
    denominators = {
        "profile_intervals_planned": planned,
        "profile_intervals_valid": valid_profile,
        "profile_intervals_failed": failed_profile,
        "profile_intervals_abstained": abstained_profile,
        "holdout_panels_planned": len(scenarios) * outer_panels_per_scenario,
        "holdout_panels_valid": len(scenarios) * outer_panels_per_scenario,
        "holdout_panels_failed": 0,
        "holdout_panels_abstained": 0,
        "physical_measurements_planned": 0,
        "physical_measurements_valid": 0,
        "physical_measurements_failed": 0,
        "physical_measurements_abstained": 0,
        "certification_planned": 1,
        "certification_valid": 0,
        "certification_failed": 0,
        "certification_abstained": 1,
    }
    if valid_profile + failed_profile + abstained_profile != planned:
        raise RuntimeError("profile interval denominator does not reconcile")

    source_hashes = {
        Path(__file__).name: _sha256(Path(__file__)),
        Path(b80.__file__).name: _sha256(Path(b80.__file__)),
        Path(mbb.__file__).name: _sha256(Path(mbb.__file__)),
        Path(stress.__file__).name: _sha256(Path(stress.__file__)),
        Path(joint.__file__).name: _sha256(Path(joint.__file__)),
    }
    protocol = {
        "schema": "wilson-f-joint-spectral-profile-v1",
        "classification": "synthetic CPU profile-likelihood interval pilot only",
        "physical_score": None,
        "input_b80_artifact_hashes": json.loads((Path(b80_root) / "artifact_hashes.json").read_text()),
        "input_stress_artifact_hashes": json.loads((stress.DEFAULT_OUT / "artifact_hashes.json").read_text()),
        "scenarios": list(scenarios),
        "calibration_replicates": "0:40",
        "holdout_replicates": f"40:{40 + outer_panels_per_scenario}",
        "n_states": stress.FIT_STATES,
        "pointwise_alpha": mbb.ALPHA,
        "profile_cutoff_delta_chi2": PROFILE_DELTA_CHI2,
        "profile_parameterization": "maximize q by replacing the selected ground-overlap magnitude with q; for operator 2, retain the ground-overlap phase and reconstruct its complex magnitude",
        "objective": "fixed per-panel regularized covariance GLS chi-square, with no refitting of the covariance during the profile",
        "optimizer": "SLSQP constrained to delta_chi2 <= chi2_1(0.95); analytic model/profile-transform/constraint Jacobians; unsuccessful or infeasible runs are FAILED; successful parameter-boundary hits ABSTAIN",
        "comparators": ["delta", "split_conformal", "moving_block_B80"],
        "source_sha256": source_hashes,
        "warning": "chi-square cutoff is a plug-in asymptotic profile rule; local optimizer and model misspecification can invalidate coverage",
        "omitted": ["familywise q/h/x", "finite-Nt thermal h", "mixed x", "production correlators", "independent contamination receipts"],
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    for module_path in (Path(__file__), Path(b80.__file__), Path(mbb.__file__),
                        Path(stress.__file__), Path(joint.__file__)):
        shutil.copyfile(module_path, out / f"snapshot_{module_path.name}")
    for filename, rows in (("profile_intervals.csv", profile_rows),
                           ("method_comparison.csv", comparison_rows),
                           ("calibration_summary.csv", calibration_receipts)):
        with (out / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    _plot(comparison_rows, out / "method_coverage.png", scenarios)
    summary = {
        "schema": "wilson-f-joint-spectral-profile-summary-v1",
        "classification": "synthetic CPU profile-likelihood interval pilot only",
        "physical_score": None,
        "F_lane_status": "GROUND_STATE_FORCE_UNCERTIFIED",
        "denominators": denominators,
        "scenarios": list(scenarios),
        "comparison_rows": comparison_rows,
        "files": ["README.md", "protocol.json", "profile_intervals.csv",
                  "method_comparison.csv", "calibration_summary.csv",
                  "method_coverage.png", "summary.json",
                  *[f"snapshot_{Path(p).name}" for p in
                    (Path(__file__), Path(b80.__file__), Path(mbb.__file__),
                     Path(stress.__file__), Path(joint.__file__))]],
        "source_sha256": source_hashes,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    profile_lines = []
    for scenario in scenarios:
        group = [r for r in comparison_rows
                 if r["scenario"] == scenario and r["method"] == "profile_likelihood"]
        cov = [r["coverage"] for r in group if r["coverage"] is not None]
        den = sorted({(r["valid"], r["planned"], r["failed"], r["abstained"]) for r in group})
        cov_text = f"{min(cov):.3f}–{max(cov):.3f}" if cov else "no finite intervals"
        profile_lines.append(f"- `{scenario}`: profile coverage `{cov_text}`, cell denominators `{den}`.")
    readme = [
        "# Wilson F-lane profile-likelihood upper-bound pilot", "",
        "**Synthetic CPU method diagnostic only.** It reuses the verified six-scenario GLS stress inputs and B=80 report for comparisons. No Wilson measurement, sealed split, GPU or QUDA data were read. `physical_score=null`; F remains `GROUND_STATE_FORCE_UNCERTIFIED`.", "",
        "## Fixed method", "",
        f"The same first-40 calibration / next-{outer_panels_per_scenario} holdout panels are used. For each panel, a three-state GLS fit estimates the full regularized covariance once; the profile keeps that covariance fixed and maximizes q under `chi2(q)-chi2_min <= {PROFILE_DELTA_CHI2:.12f}`, the plug-in one-degree-of-freedom chi-square cutoff corresponding to a two-sided 95% likelihood-ratio set / one-sided 97.5% endpoint under regular Wilks conditions. For bare operator 0 the ground overlap is eliminated algebraically; for operator 2 its complex phase is retained and magnitude reconstructed. A local SLSQP solution, failed optimization, or fit-domain boundary hit is recorded as failed/abstained rather than silently accepted.", "",
        "![Profile, B=80, conformal and delta coverage](method_coverage.png)", "",
        f"Profile interval denominators planned/valid/failed/abstained = `{planned}/{valid_profile}/{failed_profile}/{abstained_profile}`. Full per-cell coverage, Wilson Monte Carlo intervals, optimizer status, chi-square displacement, boundary hits and interval widths are in `profile_intervals.csv` and `method_comparison.csv`.", "",
        *profile_lines, "",
        "## Limits", "",
        "This is not a finite-sample guarantee. The covariance is estimated then treated as fixed, the profile optimizer is local, the likelihood-ratio cutoff assumes regularity, and near-zero overlaps or active bounds violate those conditions. Omitted-state bias, AR(1) block dependence, finite-Nt thermal h, mixed x and familywise q/h/x control remain outside the profile. Results are conditional on reportable intervals; abstentions stay visible in denominators. No interval may enter the spectral q reader or F gate.", "",
        "Text flow: `verify B80/stress hashes -> hold same synthetic panels and per-panel covariance -> fit three-state GLS -> eliminate selected ground-overlap magnitude using q -> maximize q inside fixed chi-square LR set -> record optimizer/boundary failures -> compare coverage and abstention -> physical certification abstains`.", "",
        "Reproduce with `PYTHONPATH=. python -B -m tasks.wilson_loop.f_lane_joint_spectral_profile_diagnostic --output tasks/wilson_loop/f_lane_joint_spectral_profile_20261126_v1`. Source/artifact hashes are recorded in the report.",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n")
    hashes = {name: _sha256(out / name) for name in summary["files"] if (out / name).is_file()}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, allow_nan=False) + "\n")
    return _refresh_derived_report(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b80-root", type=Path, default=DEFAULT_B80)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--outer-panels-per-scenario", type=int, default=40)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--refresh-derived", action="store_true",
                        help="refresh an existing report's plot/text/hashes without refitting")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.refresh_derived:
        result = _refresh_derived_report(args.output)
        print(json.dumps(result["denominators"], sort_keys=True))
        return
    result = run(args.output, b80_root=args.b80_root,
                 outer_panels_per_scenario=args.outer_panels_per_scenario,
                 max_iterations=args.max_iterations,
                 report_progress=not args.quiet)
    print(json.dumps(result["denominators"], sort_keys=True))


if __name__ == "__main__":
    main()
