"""Paired point-fit screen for the Wilson largest-component phase chart.

This compares coordinate charts only on already-saved open synthetic panels.
It does not estimate profile coverage or produce physical force bounds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares

from core.source_integrity import capture_source_snapshot
from tasks.wilson_loop import f_lane_three_static_pivot_chart as pivot
from tasks.wilson_loop import f_lane_three_static_two_vacuum_candidate as current
from tasks.wilson_loop import f_lane_two_sector_fit_candidate as two_state


TASK_DIR = Path(__file__).resolve().parent
INPUT_REPORT = TASK_DIR / "f_lane_three_static_stratified_profile_20261128_v1"
DEFAULT_OUT = TASK_DIR / "f_lane_pivot_chart_fit_comparison_20261128_v2"
PANEL_NAMES = (
    "06_baseline", "07_baseline", "08_baseline",
    "09_near_degenerate", "10_near_degenerate", "11_near_degenerate",
    "12_omitted_vacuum_truth", "13_omitted_vacuum_truth", "14_omitted_vacuum_truth",
    "15_dependent_unequal_variance", "16_dependent_unequal_variance",
    "17_dependent_unequal_variance",
)
STARTS = ((0.15, 0.20), (0.30, 0.35), (0.55, 0.50))
MAX_NFEV = 8000
TOLERANCE = 1e-10
METHODS = ("fixed_operator0", "largest_component_pivot")
FIT_ACCEPTANCE = "scipy success, finite residual and coordinates, numerical Jacobian rank 67"
PROFILE_ADVANCE_RULE = (
    "Advance to a paired profile comparison only if the pivot chart recovers at least one "
    "baseline-invalid start, or reduces median solver evaluations (nfev+njev) by >=20% in at least two "
    "strata among pairs valid in both methods, without any per-stratum valid-count loss."
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha(path.read_bytes())


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _protocol(artifact_manifest_sha256: str) -> dict[str, Any]:
    return {
        "schema": "wilson-f-pivot-chart-paired-point-fit-screen-v2",
        "classification": "runtime-motivated re-execution; post-hoc method feasibility diagnostic; not physical calibration",
        "revision_reason": (
            "The v1 finite-difference run was cancelled after 23m with zero completed fits. "
            "No fit outcomes were observed. Use the already implemented analytic chart Jacobian "
            "for both equivalent parameterizations to retain the paired chart comparison at feasible cost."
        ),
        "input_report": "f_lane_three_static_stratified_profile_20261128_v1",
        "input_panel_ids": list(PANEL_NAMES),
        "input_artifact_manifest_sha256": artifact_manifest_sha256,
        "methods": list(METHODS),
        "shared_model": "finite-Nt two-vacuum x three-static positive spectral trace; 67 coordinates",
        "shared_covariance": "same block vectors; 25% diagonal shrinkage and 1e-10 relative eigenvalue floor",
        "shared_initialization": "same Nt=32 three-static forward initializer; epsilon/scale starts are fixed below",
        "starts_epsilon_excited_scale": [list(row) for row in STARTS],
        "solver": "scipy least_squares(method=trf, jac=analytic chart-specific derivative, x_scale=jac)",
        "max_nfev_per_start": MAX_NFEV,
        "ftol_xtol_gtol": TOLERANCE,
        "primary_validity": FIT_ACCEPTANCE,
        "reported_metrics": [
            "planned/valid/failed/abstained starts by chart and stratum",
            "residual calls, analytic Jacobian calls, total solver evaluations (nfev+njev), elapsed seconds",
            "best GLS objective and point-force difference between charts",
        ],
        "profile_advance_rule": PROFILE_ADVANCE_RULE,
        "not_claimed": [
            "force interval coverage", "global identifiability", "production overlap floor",
            "physical q/h/x bounds", "C-lane finite-T Creutz result",
        ],
        "physical_score": None,
        "certified": False,
    }


def _bounds() -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(current.PARAMETER_COUNT, -np.inf)
    upper = np.full(current.PARAMETER_COUNT, np.inf)
    lower[0], upper[0] = -8.0, 2.5
    cursor = 1
    for _ in two_state.R_VALUES:
        lower[cursor:cursor + current.STATIC_STATES] = -8.0
        upper[cursor:cursor + current.STATIC_STATES] = 2.5
        cursor += current.STATIC_STATES
        for _ in range(current.VACUUM_STATES * current.STATIC_STATES):
            lower[cursor], upper[cursor] = -12.0, 4.0
            lower[cursor + 1:cursor + 5] = -5.0
            upper[cursor + 1:cursor + 5] = 5.0
            cursor += 5
    return lower, upper


def _pivot_start(static: list[np.ndarray], overlaps: list[np.ndarray],
                 epsilon: float, excited_scale: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    amplitudes = np.empty(
        (len(two_state.R_VALUES), current.VACUUM_STATES,
         current.STATIC_STATES, current.OPERATOR_COUNT), dtype=np.complex128,
    )
    for radius_index in range(len(two_state.R_VALUES)):
        amplitudes[radius_index, 0] = overlaps[radius_index]
        amplitudes[radius_index, 1] = excited_scale * overlaps[radius_index]
    return pivot.encode_components(epsilon, static, amplitudes)


def _run_one(method: str, initial: np.ndarray, pivot_map: np.ndarray | None,
             mean: np.ndarray, covariance_cholesky: np.ndarray,
             lower: np.ndarray, upper: np.ndarray) -> dict[str, Any]:
    if np.any(initial < lower) or np.any(initial > upper):
        return {"status": "ABSTAIN_INITIALIZER_OUT_OF_BOUNDS", "valid": False,
                "residual_calls": 0, "jacobian_calls": 0, "model_evaluations": 0,
                "solver_evaluations": 0, "elapsed_seconds": 0.0,
                "nfev": 0, "njev": 0, "objective": None,
                "force": None, "rank": 0, "condition_number": None,
                "error": "initial coordinates exceed shared parameter bounds"}
    residual_calls = 0
    jacobian_calls = 0
    model_evaluations = 0
    fixed_pivots = np.zeros(
        (len(two_state.R_VALUES), current.VACUUM_STATES, current.STATIC_STATES),
        dtype=np.int64,
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        nonlocal residual_calls, model_evaluations
        residual_calls += 1
        model_evaluations += 1
        if method == "fixed_operator0":
            model = current._model_vectors(parameters)
        else:
            assert pivot_map is not None
            model = pivot.model_vectors(parameters, pivot_map)
        return solve_triangular(
            covariance_cholesky, model.ravel() - mean, lower=True, check_finite=True,
        )

    def jacobian(parameters: np.ndarray) -> np.ndarray:
        nonlocal jacobian_calls, model_evaluations
        jacobian_calls += 1
        model_evaluations += 1
        pivots = fixed_pivots if method == "fixed_operator0" else pivot_map
        assert pivots is not None
        raw = pivot.model_jacobian(parameters, pivots)
        return solve_triangular(covariance_cholesky, raw, lower=True, check_finite=True)

    try:
        started = time.perf_counter()
        fit = least_squares(
            residual, initial, jac=jacobian, bounds=(lower, upper), method="trf",
            x_scale="jac", max_nfev=MAX_NFEV,
            ftol=TOLERANCE, xtol=TOLERANCE, gtol=TOLERANCE,
        )
        singular = np.linalg.svd(fit.jac, compute_uv=False)
        tolerance = (singular[0] * max(fit.jac.shape) * np.finfo(float).eps
                     if singular.size else 0.0)
        rank = int(np.count_nonzero(singular > tolerance))
        finite = bool(np.isfinite(fit.fun).all() and np.isfinite(fit.x).all())
        if method == "fixed_operator0":
            force = float(current._decode(fit.x)[1][1, 0]
                          - current._decode(fit.x)[1][0, 0])
        else:
            assert pivot_map is not None
            decoded = pivot._decode(fit.x, pivot_map)
            force = float(decoded[1][1, 0] - decoded[1][0, 0])
        objective = float(np.dot(fit.fun, fit.fun)) if finite else None
        valid = bool(fit.success and finite and rank == current.PARAMETER_COUNT)
        elapsed_seconds = time.perf_counter() - started
        return {
            "status": "VALID_FULL_RANK_FIT" if valid else "FAILED_OR_ABSTAINED_FIT",
            "valid": valid, "optimizer_success": bool(fit.success),
            "residual_calls": residual_calls, "jacobian_calls": jacobian_calls,
            "model_evaluations": model_evaluations,
            "solver_evaluations": int(fit.nfev + (fit.njev or 0)),
            "elapsed_seconds": elapsed_seconds,
            "nfev": int(fit.nfev),
            "njev": int(fit.njev or 0), "objective": objective,
            "force": force if finite else None, "rank": rank,
            "condition_number": float(singular[0] / singular[-1]) if singular.size else None,
            "error": None if valid else str(fit.message),
        }
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
        return {
            "status": "FAILED_OPTIMIZATION", "valid": False,
            "residual_calls": residual_calls, "jacobian_calls": jacobian_calls,
            "model_evaluations": model_evaluations,
            "solver_evaluations": residual_calls + jacobian_calls,
            "elapsed_seconds": time.perf_counter() - started,
            "nfev": 0, "njev": 0,
            "objective": None, "force": None, "rank": 0,
            "condition_number": None, "error": f"{type(exc).__name__}: {exc}",
        }


def _start_abstention(status: str, error: str) -> dict[str, Any]:
    return {
        "status": status, "valid": False, "residual_calls": 0,
        "jacobian_calls": 0, "model_evaluations": 0, "solver_evaluations": 0,
        "elapsed_seconds": 0.0, "nfev": 0, "njev": 0,
        "objective": None, "force": None,
        "rank": 0, "condition_number": None, "error": error,
    }


def _stratum(panel_id: str) -> str:
    for value in ("baseline", "near_degenerate", "omitted_vacuum_truth",
                  "dependent_unequal_variance"):
        if value in panel_id:
            return value
    raise ValueError(f"unrecognized panel stratum: {panel_id}")


def _load_panel(panel_id: str, artifacts: dict[str, str]):
    names = ("manifest", "raw_matrices", "lane_contract")
    filenames = {"manifest": "manifest.json", "raw_matrices": "raw_matrices.npz",
                 "lane_contract": "lane_contract.json"}
    relative = {name: f"panels/{panel_id}/{filenames[name]}" for name in names}
    expected = {name: artifacts[relative[name]] for name in names}
    snapshot = capture_source_snapshot(
        INPUT_REPORT, expected, relative, max_total_bytes=64 * 1024 * 1024,
    )
    _manifest, matrices = two_state._validate_raw_snapshot(
        snapshot, manifest_name="manifest", matrix_name="raw_matrices",
        contract_name="lane_contract",
    )
    return snapshot, matrices


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(output: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    artifact_manifest_bytes = (INPUT_REPORT / "artifact_hashes.json").read_bytes()
    artifact_manifest = json.loads(artifact_manifest_bytes)
    artifacts = artifact_manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("input artifact manifest has an invalid schema")
    protocol = _protocol(_sha(artifact_manifest_bytes))
    _write_json(out / "protocol.json", protocol)
    progress_path = out / "execution_progress.jsonl"
    progress_path.write_text("", encoding="utf-8")

    lower, upper = _bounds()
    rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    source_digests: dict[str, str] = {
        "tasks/wilson_loop/f_lane_pivot_chart_fit_comparison.py": _sha_file(Path(__file__)),
        "tasks/wilson_loop/f_lane_three_static_pivot_chart.py": _sha_file(TASK_DIR / "f_lane_three_static_pivot_chart.py"),
        "tasks/wilson_loop/f_lane_three_static_two_vacuum_candidate.py": _sha_file(TASK_DIR / "f_lane_three_static_two_vacuum_candidate.py"),
        "tasks/wilson_loop/f_lane_two_sector_fit_candidate.py": _sha_file(TASK_DIR / "f_lane_two_sector_fit_candidate.py"),
        "tasks/wilson_loop/f_lane_joint_spectral_gls_diagnostic.py": _sha_file(TASK_DIR / "f_lane_joint_spectral_gls_diagnostic.py"),
        "core/source_integrity.py": _sha_file(REPO_ROOT / "core/source_integrity.py"),
        "core/npz_safety.py": _sha_file(REPO_ROOT / "core/npz_safety.py"),
    }
    panel_capture_rows = []
    for panel_id in PANEL_NAMES:
        snapshot, matrices = _load_panel(panel_id, artifacts)
        capture = snapshot.capture_entries
        panel_capture_rows.extend(
            {"panel_id": panel_id, "source_name": name, "status": entry["status"],
             "sha256": entry["observed_sha256"]}
            for name, entry in capture.items()
        )
        manifest, _ = two_state._strict_json(snapshot, "manifest")
        block_vectors = two_state._observed_vectors(matrices)
        mean = block_vectors.mean(axis=0)
        covariance, covariance_diagnostics = two_state._regularized_mean_covariance(block_vectors)
        chol = cholesky(covariance, lower=True, check_finite=True)
        try:
            static, overlaps, initializer_receipts = current._initial_from_snapshot(matrices)
            initializer_error = None
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
            static, overlaps, initializer_receipts = None, None, []
            initializer_error = f"{type(exc).__name__}: {exc}"
        methods_for_panel: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
        for start_index, (epsilon, excited_scale) in enumerate(STARTS):
            if initializer_error is not None:
                failed = {"status": "ABSTAIN_SHARED_INITIALIZER_FAILED", "valid": False,
                          "residual_calls": 0, "jacobian_calls": 0,
                          "model_evaluations": 0, "solver_evaluations": 0,
                          "elapsed_seconds": 0.0, "nfev": 0, "njev": 0,
                          "objective": None, "force": None, "rank": 0,
                          "condition_number": None, "error": initializer_error}
                fixed_result = dict(failed)
                pivot_result = dict(failed)
                fixed_initial = None
                pivot_initial = None
                pivot_map = None
                initial_model_diff = None
            else:
                assert static is not None and overlaps is not None
                fixed_initial = None
                pivot_initial = None
                pivot_map = None
                initial_model_diff = None
                fixed_result = None
                pivot_result = None
                try:
                    fixed_initial = current._initial_parameters(
                        static, overlaps, epsilon, excited_scale,
                    )
                except (ValueError, FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
                    fixed_result = _start_abstention(
                        "ABSTAIN_FIXED_CHART_INITIALIZER_FAILED",
                        f"{type(exc).__name__}: {exc}",
                    )
                try:
                    pivot_initial, pivot_map = _pivot_start(
                        static, overlaps, epsilon, excited_scale,
                    )
                except (ValueError, FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
                    pivot_result = _start_abstention(
                        "ABSTAIN_PIVOT_CHART_INITIALIZER_FAILED",
                        f"{type(exc).__name__}: {exc}",
                    )
                if fixed_initial is not None and pivot_initial is not None:
                    initial_model_diff = float(np.max(np.abs(
                        current._model_vectors(fixed_initial)
                        - pivot.model_vectors(pivot_initial, pivot_map)
                    )))
                    if initial_model_diff > 5e-13:
                        message = "paired charts do not encode the same initial model"
                        fixed_result = _start_abstention("ABSTAIN_PAIRED_START_MISMATCH", message)
                        pivot_result = _start_abstention("ABSTAIN_PAIRED_START_MISMATCH", message)
                        fixed_initial = None
                        pivot_initial = None
                if fixed_initial is not None and fixed_result is None:
                    fixed_result = _run_one(
                        "fixed_operator0", fixed_initial, None, mean, chol, lower, upper,
                    )
                if pivot_initial is not None and pivot_result is None:
                    pivot_result = _run_one(
                        "largest_component_pivot", pivot_initial, pivot_map,
                        mean, chol, lower, upper,
                    )
                assert fixed_result is not None and pivot_result is not None
            for method, result in ((METHODS[0], fixed_result), (METHODS[1], pivot_result)):
                methods_for_panel[method].append(result)
            row = {
                "panel_id": panel_id, "stratum": _stratum(panel_id),
                "split_id": manifest["split_id"], "start_index": start_index,
                "epsilon_start": epsilon, "excited_overlap_scale_start": excited_scale,
                "initial_model_max_abs_difference": initial_model_diff,
                "fixed_status": fixed_result["status"], "fixed_valid": fixed_result["valid"],
                "fixed_residual_calls": fixed_result["residual_calls"],
                "fixed_jacobian_calls": fixed_result["jacobian_calls"],
                "fixed_model_evaluations": fixed_result["model_evaluations"],
                "fixed_solver_evaluations": fixed_result["solver_evaluations"],
                "fixed_elapsed_seconds": fixed_result["elapsed_seconds"],
                "fixed_nfev": fixed_result["nfev"], "fixed_njev": fixed_result["njev"],
                "fixed_objective": fixed_result["objective"], "fixed_force": fixed_result["force"],
                "fixed_rank": fixed_result["rank"],
                "fixed_condition_number": fixed_result["condition_number"],
                "fixed_error": fixed_result["error"],
                "pivot_status": pivot_result["status"], "pivot_valid": pivot_result["valid"],
                "pivot_residual_calls": pivot_result["residual_calls"],
                "pivot_jacobian_calls": pivot_result["jacobian_calls"],
                "pivot_model_evaluations": pivot_result["model_evaluations"],
                "pivot_solver_evaluations": pivot_result["solver_evaluations"],
                "pivot_elapsed_seconds": pivot_result["elapsed_seconds"],
                "pivot_nfev": pivot_result["nfev"], "pivot_njev": pivot_result["njev"],
                "pivot_objective": pivot_result["objective"], "pivot_force": pivot_result["force"],
                "pivot_rank": pivot_result["rank"],
                "pivot_condition_number": pivot_result["condition_number"],
                "pivot_error": pivot_result["error"],
            }
            rows.append(row)
            progress_record = {
                "panel_id": panel_id, "start_index": start_index,
                "fixed_status": fixed_result["status"], "fixed_valid": fixed_result["valid"],
                "fixed_solver_evaluations": fixed_result["solver_evaluations"],
                "pivot_status": pivot_result["status"], "pivot_valid": pivot_result["valid"],
                "pivot_solver_evaluations": pivot_result["solver_evaluations"],
            }
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(progress_record, sort_keys=True) + "\n")
                stream.flush()
            print(
                f"completed {len(rows)}/{len(PANEL_NAMES) * len(STARTS)} paired starts: "
                f"{panel_id} start={start_index} fixed={fixed_result['status']} "
                f"pivot={pivot_result['status']}",
                flush=True,
            )
        best = {}
        for method in METHODS:
            valid_rows = [row for row in methods_for_panel[method] if row["valid"]]
            best[method] = min(valid_rows, key=lambda item: item["objective"]) if valid_rows else None
        fixed_best, pivot_best = best[METHODS[0]], best[METHODS[1]]
        panel_rows.append({
            "panel_id": panel_id, "stratum": _stratum(panel_id),
            "split_id": manifest["split_id"],
            "planned_starts_per_method": len(STARTS),
            "fixed_valid_starts": sum(row["valid"] for row in methods_for_panel[METHODS[0]]),
            "pivot_valid_starts": sum(row["valid"] for row in methods_for_panel[METHODS[1]]),
            "fixed_best_objective": fixed_best["objective"] if fixed_best else None,
            "pivot_best_objective": pivot_best["objective"] if pivot_best else None,
            "best_objective_relative_difference": (
                (pivot_best["objective"] - fixed_best["objective"])
                / max(abs(fixed_best["objective"]), 1e-12)
                if fixed_best and pivot_best else None
            ),
            "fixed_best_force": fixed_best["force"] if fixed_best else None,
            "pivot_best_force": pivot_best["force"] if pivot_best else None,
            "best_force_difference": (
                pivot_best["force"] - fixed_best["force"]
                if fixed_best and pivot_best else None
            ),
            "covariance_condition_number": covariance_diagnostics["condition_number"],
            "initializer_status": "valid" if initializer_error is None else "failed",
            "initializer_error": initializer_error,
        })

    _write_csv(out / "paired_start_results.csv", list(rows[0]), rows)
    _write_csv(out / "panel_summary.csv", list(panel_rows[0]), panel_rows)
    _write_csv(out / "captured_inputs.csv", list(panel_capture_rows[0]), panel_capture_rows)
    _write_json(out / "source_hashes.json", source_digests)

    by_method: dict[str, dict[str, int]] = {}
    by_stratum: dict[str, dict[str, dict[str, int]]] = {}
    for method, valid_key in ((METHODS[0], "fixed_valid"), (METHODS[1], "pivot_valid")):
        valids = [row[valid_key] for row in rows]
        by_method[method] = {
            "planned": len(valids), "valid": sum(valids),
            "failed_or_abstained": len(valids) - sum(valids),
        }
    for stratum in sorted({_stratum(panel) for panel in PANEL_NAMES}):
        by_stratum[stratum] = {}
        group = [row for row in rows if row["stratum"] == stratum]
        for method, valid_key in ((METHODS[0], "fixed_valid"), (METHODS[1], "pivot_valid")):
            valid_count = sum(bool(row[valid_key]) for row in group)
            by_stratum[stratum][method] = {
                "planned": len(group), "valid": valid_count,
                "failed_or_abstained": len(group) - valid_count,
            }
    paired_both_valid = [row for row in rows if row["fixed_valid"] and row["pivot_valid"]]
    evaluation_savings: dict[str, float | None] = {}
    for stratum in by_stratum:
        group = [row for row in paired_both_valid if row["stratum"] == stratum]
        if group:
            fixed_median = float(np.median([row["fixed_solver_evaluations"] for row in group]))
            pivot_median = float(np.median([row["pivot_solver_evaluations"] for row in group]))
            evaluation_savings[stratum] = (
                (fixed_median - pivot_median) / fixed_median if fixed_median > 0 else None
            )
        else:
            evaluation_savings[stratum] = None
    recovered = sum(
        not row["fixed_valid"] and row["pivot_valid"] for row in rows
    )
    valid_count_loss = {
        stratum: (by_stratum[stratum][METHODS[1]]["valid"]
                  < by_stratum[stratum][METHODS[0]]["valid"])
        for stratum in by_stratum
    }
    strata_meeting_call_rule = sum(
        savings is not None and savings >= 0.20
        for savings in evaluation_savings.values()
    )
    profile_advance = bool(
        recovered >= 1
        or (strata_meeting_call_rule >= 2 and not any(valid_count_loss.values()))
    )
    summary = {
        "schema": "wilson-f-pivot-chart-paired-point-fit-summary-v2",
        "status": "METHOD_SCREEN_COMPLETE",
        "physical_score": None,
        "certified": False,
        "panel_denominator": {
            "planned": len(PANEL_NAMES), "valid": sum(row["initializer_status"] == "valid" for row in panel_rows),
            "failed": sum(row["initializer_status"] == "failed" for row in panel_rows),
            "abstained": 0,
        },
        "start_denominators": by_method,
        "start_denominators_by_stratum": by_stratum,
        "paired_starts_valid_in_both": len(paired_both_valid),
        "pivot_recovers_baseline_invalid_starts": recovered,
        "median_solver_evaluation_savings_by_stratum": evaluation_savings,
        "strata_meeting_20_percent_savings": strata_meeting_call_rule,
        "any_stratum_valid_count_loss": any(valid_count_loss.values()),
        "advance_to_profile_comparison": profile_advance,
        "profile_advance_rule": PROFILE_ADVANCE_RULE,
        "best_force_difference_lattice_units": [
            row["best_force_difference"] for row in panel_rows
            if row["best_force_difference"] is not None
        ],
        "source_capture_denominator": {
            "planned": len(panel_capture_rows),
            "valid": sum(row["status"] == "valid" for row in panel_capture_rows),
            "failed": sum(row["status"] == "failed" for row in panel_capture_rows),
            "abstained": sum(row["status"] == "abstained" for row in panel_capture_rows),
        },
        "all_initial_model_pair_differences_max": max(
            row["initial_model_max_abs_difference"] for row in rows
            if row["initial_model_max_abs_difference"] is not None
        ),
        "fit_acceptance": FIT_ACCEPTANCE,
        "covariance": "same observed block vectors per panel, same fixed shrinkage/floor for both charts",
        "limitations": [
            "point-fit feasibility only; no force-profile intervals or coverage",
            "small reused synthetic panel set; results are not independent confirmation",
            "current panels do not include the exact-zero-operator0 stress fixture",
            "no production matrices, sealed data, GPU or QUDA",
        ],
    }
    _write_json(out / "summary.json", summary)
    _plot(out / "fit_comparison.png", rows)
    _write_readme(out / "README.md", summary)

    artifacts = ["README.md", "protocol.json", "summary.json", "execution_progress.jsonl",
                 "paired_start_results.csv",
                 "panel_summary.csv", "captured_inputs.csv", "fit_comparison.png",
                 "source_hashes.json"]
    _write_json(out / "artifact_hashes.json", {
        name: _sha_file(out / name) for name in artifacts
    })
    return summary


def _plot(path: Path, rows: list[dict[str, Any]]) -> None:
    strata = sorted({row["stratum"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    x = np.arange(len(strata))
    width = 0.36
    for offset, method, key, label, color in (
        (-width / 2, METHODS[0], "fixed_valid", "fixed operator 0", "#3978a8"),
        (width / 2, METHODS[1], "pivot_valid", "largest-component pivot", "#d07a2d"),
    ):
        counts = [sum(bool(row[key]) for row in rows if row["stratum"] == stratum)
                  for stratum in strata]
        axes[0].bar(x + offset, counts, width, label=label, color=color)
    axes[0].set_xticks(x, [value.replace("_", "\n") for value in strata])
    axes[0].set_ylim(0, 10)
    axes[0].set_ylabel("valid full-rank starts / 9 planned")
    axes[0].set_title("Point-fit completion by stress stratum")
    axes[0].legend(frameon=False, fontsize=8)

    paired = [row for row in rows if row["fixed_valid"] and row["pivot_valid"]]
    if paired:
        fixed = np.asarray([row["fixed_solver_evaluations"] for row in paired], dtype=float)
        pivot_calls = np.asarray([row["pivot_solver_evaluations"] for row in paired], dtype=float)
        colors = [strata.index(row["stratum"]) for row in paired]
        axes[1].scatter(fixed, pivot_calls, c=colors, cmap="viridis", s=24, alpha=0.8)
        maxv = max(float(fixed.max()), float(pivot_calls.max()))
        axes[1].plot([0, maxv], [0, maxv], color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("fixed operator-0 solver evaluations (nfev+njev)")
    axes[1].set_ylabel("largest-pivot solver evaluations (nfev+njev)")
    axes[1].set_title("Paired starts valid in both methods")
    axes[1].grid(alpha=0.2)
    fig.suptitle("Same synthetic matrices and solver budget; diagnostic only")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _write_readme(path: Path, summary: dict[str, Any]) -> None:
    methods = summary["start_denominators"]
    capture = summary["source_capture_denominator"]
    lines = [
        "# Paired Wilson point-fit screen: fixed and largest-component phase charts",
        "",
        "**Classification:** exploratory CPU method-feasibility screen on reused open synthetic data. It is not profile coverage calibration or a physical force result.",
        "",
        "![Paired chart fit results](fit_comparison.png)",
        "",
        "## Frozen comparison",
        "",
        "The v1 finite-difference run was stopped after 23 minutes during the first fit; zero fit outcomes were observed. This v2 rerun keeps all 12 panels, the same raw matrix snapshot, block covariance, Nt=32 three-static initializer, three epsilon/overlap-scale starts, TRF solver, `x_scale='jac'`, `max_nfev=8000`, `1e-10` tolerances, and rank-67 validity rule. The only protocol revision is using the already implemented analytic Jacobian for each chart; fixed operator 0 is the largest-pivot chart with all pivot indices set to zero, so both Jacobians share one tested derivative implementation in their own coordinates. Every start remains in the planned denominator.",
        "",
        "Profile-follow-up rule: continue to a paired profile comparison only if the pivot chart rescues at least one baseline-invalid start, or saves at least 20% median solver evaluations (`nfev+njev`) in at least two strata among pairs valid in both methods, with no valid-start count loss in any stratum.",
        "",
        "## Results",
        "",
        f"- Panel initializer denominator: `{summary['panel_denominator']['planned']}/{summary['panel_denominator']['valid']}/{summary['panel_denominator']['failed']}/{summary['panel_denominator']['abstained']}` (planned/valid/failed/abstained).",
        f"- Captured source files: `{capture['planned']}/{capture['valid']}/{capture['failed']}/{capture['abstained']}` (planned/valid/failed/abstained).",
        f"- Fixed operator-0 chart starts: `{methods[METHODS[0]]['planned']}/{methods[METHODS[0]]['valid']}/{methods[METHODS[0]]['failed_or_abstained']}` (planned/valid/failed-or-abstained).",
        f"- Largest-component chart starts: `{methods[METHODS[1]]['planned']}/{methods[METHODS[1]]['valid']}/{methods[METHODS[1]]['failed_or_abstained']}` (planned/valid/failed-or-abstained).",
        f"- Paired starts valid in both: `{summary['paired_starts_valid_in_both']}`; pivot rescues over baseline: `{summary['pivot_recovers_baseline_invalid_starts']}`.",
        f"- Median solver-evaluation savings by stratum: `{json.dumps(summary['median_solver_evaluation_savings_by_stratum'], sort_keys=True)}`.",
        f"- Largest max absolute difference between paired initial model vectors: `{summary['all_initial_model_pair_differences_max']:.3g}`.",
        f"- Frozen profile-follow-up rule result: `advance={summary['advance_to_profile_comparison']}`.",
        "",
        "## Interpretation and limits",
        "",
        "This screen compares point fits only. It does not calculate force intervals, truth containment, finite-sample coverage, q/h/x contamination, or physical score. The panels were previously used in method development, so they are not held-out confirmation data. The exact-zero-operator0 fixture is a separate algebraic representation check and is not among these panels. Source hashes bind local bytes but do not authenticate a data owner or independent review. No production Wilson, sealed input, GPU, or QUDA was accessed. C finite-T Creutz and F ground-state force stay separate; physical statuses remain NOT_READY/UNCERTIFIED and `physical_score=null`.",
        "",
        "## Reproduction and audit",
        "",
        "Run `python tasks/wilson_loop/f_lane_pivot_chart_fit_comparison.py --output <new-empty-directory>`. `protocol.json` freezes panel IDs, starts, solver, budget, acceptance rule and profile-follow-up threshold. `paired_start_results.csv`, `panel_summary.csv` and `captured_inputs.csv` retain full denominators and byte digests; source and artifact SHA-256 maps are included. The v1 runtime-cancelled attempt is retained in its own folder and must not be counted as fit evidence.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    summary = build(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
