"""Nonlinear conditional force profiles with an analytic finite-Nt Jacobian.

The only input is the previously saved synthetic weak-overlap panel. Results
are method-feasibility diagnostics, not physical bounds or calibrated intervals.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares
from scipy.special import expit

from tasks.wilson_loop import f_lane_absolute_force_bootstrap_diagnostic as bootstrap
from tasks.wilson_loop import f_lane_three_static_overlap_profile as local_profile
from tasks.wilson_loop import f_lane_three_static_two_vacuum_candidate as candidate
from tasks.wilson_loop import f_lane_two_sector_fit_candidate as two_state


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
INPUT_REPORT = TASK_DIR / "f_lane_three_static_identifiability_20261128_v1"
DEFAULT_OUT = TASK_DIR / "f_lane_three_static_nonlinear_overlap_profile_20261128_v1"
FLOORS = (0.0, 0.045, 0.075)
FORCE_GRID = (0.14, 0.15, 0.15757559165100204, 0.16, 0.17, 0.18)
LR_REFERENCE = 3.84
FIXED_FORCE_INDEX = 34
GROUND_A0_INDICES = (4, 37)
RADIUS_STRIDE = candidate.STATIC_STATES + candidate.VACUUM_STATES * candidate.STATIC_STATES * 5


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _model_jacobian(parameters: object) -> np.ndarray:
    """Analytic derivative of flattened candidate model vectors by parameters."""
    p = np.asarray(parameters, dtype=float)
    epsilon, static, amplitudes = candidate._decode(p)
    vacuum_excited_probability = np.exp(-epsilon * np.asarray(two_state.NT_VALUES, dtype=float))
    vacuum_excited_probability /= 1.0 + vacuum_excited_probability
    model = candidate._model_vectors(p)
    jacobian = np.zeros((model.size, p.size), dtype=float)
    row_index = 0
    for radius_index, _radius in enumerate(two_state.R_VALUES):
        radius_start = 1 + radius_index * RADIUS_STRIDE
        energy_parameter_indices = tuple(radius_start + index
                                         for index in range(candidate.STATIC_STATES))
        amplitude_start = radius_start + candidate.STATIC_STATES
        for nt_index, nt in enumerate(two_state.NT_VALUES):
            p_excited = float(vacuum_excited_probability[nt_index])
            partition = 1.0 + math.exp(-epsilon * nt)
            for time_index, time_extent in enumerate(two_state.T_VALUES):
                time_weights = np.exp(
                    -np.asarray([0.0, epsilon])[:, None] * (nt - time_extent)
                    -static[radius_index][None, :] * time_extent,
                ) / partition
                derivatives: dict[int, np.ndarray] = {}
                for vacuum_state in range(candidate.VACUUM_STATES):
                    for static_state in range(candidate.STATIC_STATES):
                        weight = float(time_weights[vacuum_state, static_state])
                        amplitude = amplitudes[radius_index, vacuum_state, static_state]
                        outer = np.outer(amplitude, amplitude.conj())
                        dlog_weight_depsilon = (-(nt - time_extent) * (vacuum_state == 1)
                                                + nt * p_excited) * float(expit(p[0]))
                        derivatives.setdefault(0, np.zeros((3, 3), dtype=complex))
                        derivatives[0] += weight * dlog_weight_depsilon * outer
                        for increment_index, parameter_index in enumerate(energy_parameter_indices):
                            if increment_index <= static_state:
                                derivatives.setdefault(parameter_index,
                                                       np.zeros((3, 3), dtype=complex))
                                derivatives[parameter_index] += (
                                    -time_extent * float(expit(p[parameter_index]))
                                    * weight * outer
                                )
                        state_start = amplitude_start + (
                            vacuum_state * candidate.STATIC_STATES + static_state
                        ) * 5
                        derivative_amplitudes = (
                            (state_start, np.asarray([amplitude[0], 0.0, 0.0], dtype=complex)),
                            (state_start + 1, np.asarray([0.0, 1.0, 0.0], dtype=complex)),
                            (state_start + 2, np.asarray([0.0, 1.0j, 0.0], dtype=complex)),
                            (state_start + 3, np.asarray([0.0, 0.0, 1.0], dtype=complex)),
                            (state_start + 4, np.asarray([0.0, 0.0, 1.0j], dtype=complex)),
                        )
                        for parameter_index, d_amplitude in derivative_amplitudes:
                            if parameter_index == state_start:
                                d_amplitude = np.asarray([amplitude[0], 0.0, 0.0], dtype=complex)
                            derivative = weight * (
                                np.outer(d_amplitude, amplitude.conj())
                                + np.outer(amplitude, d_amplitude.conj())
                            )
                            derivatives.setdefault(parameter_index,
                                                   np.zeros((3, 3), dtype=complex))
                            derivatives[parameter_index] += derivative
                for parameter_index, derivative_matrix in derivatives.items():
                    jacobian[row_index:row_index + 9, parameter_index] = two_state._matrix_vector(
                        derivative_matrix,
                    )
                row_index += 9
    return jacobian


def _expand_force_coordinate(reduced: np.ndarray, target_force: float) -> tuple[np.ndarray, np.ndarray]:
    full = np.insert(np.asarray(reduced, dtype=float), FIXED_FORCE_INDEX, 0.0)
    energy_r2 = candidate._softplus(float(full[1]))
    energy_r3 = energy_r2 + target_force
    if energy_r3 <= 0 or not math.isfinite(energy_r3):
        raise ValueError("profile force must preserve positive ordered ground energies")
    full[FIXED_FORCE_INDEX] = candidate._inverse_softplus(energy_r3)
    reduced_indices = [index for index in range(candidate.PARAMETER_COUNT)
                       if index != FIXED_FORCE_INDEX]
    chain = np.zeros((candidate.PARAMETER_COUNT, len(reduced_indices)), dtype=float)
    for column, original_index in enumerate(reduced_indices):
        chain[original_index, column] = 1.0
    p1_column = reduced_indices.index(1)
    chain[FIXED_FORCE_INDEX, p1_column] = (
        float(expit(full[1])) / float(expit(full[FIXED_FORCE_INDEX]))
    )
    return full, chain


def _parameter_bounds(floor: float, target_force: float) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(candidate.PARAMETER_COUNT, -np.inf)
    upper = np.full(candidate.PARAMETER_COUNT, np.inf)
    lower[0], upper[0] = -8.0, 2.5
    cursor = 1
    for _ in two_state.R_VALUES:
        lower[cursor:cursor + candidate.STATIC_STATES] = -8.0
        upper[cursor:cursor + candidate.STATIC_STATES] = 2.5
        cursor += candidate.STATIC_STATES
        for _ in range(candidate.VACUUM_STATES * candidate.STATIC_STATES):
            lower[cursor], upper[cursor] = -12.0, 4.0
            lower[cursor + 1:cursor + 5] = -5.0
            upper[cursor + 1:cursor + 5] = 5.0
            cursor += 5
    if floor > 0:
        for index in GROUND_A0_INDICES:
            lower[index] = max(lower[index], math.log(floor))
    maximum_e0_r3 = candidate._softplus(float(upper[FIXED_FORCE_INDEX]))
    maximum_e0_r2 = maximum_e0_r3 - target_force
    if maximum_e0_r2 <= candidate._softplus(float(lower[1])):
        raise ValueError("target force falls outside the candidate energy bounds")
    upper[1] = min(upper[1], candidate._inverse_softplus(maximum_e0_r2))
    return np.delete(lower, FIXED_FORCE_INDEX), np.delete(upper, FIXED_FORCE_INDEX)


def _start_vectors(fit_parameters: np.ndarray, floor: float,
                   continuation_parameters: np.ndarray | None = None
                   ) -> list[tuple[str, np.ndarray]]:
    base = np.asarray(fit_parameters, dtype=float).copy()
    for index in GROUND_A0_INDICES:
        if floor > 0:
            base[index] = max(base[index], math.log(floor) + 0.02)
    starts = [("captured_fit", base.copy())]
    if continuation_parameters is None:
        secondary = base.copy()
        secondary[0] = candidate._inverse_softplus(0.30)
        for radius_index in range(len(two_state.R_VALUES)):
            amplitude_start = 1 + radius_index * RADIUS_STRIDE + candidate.STATIC_STATES
            for static_state in range(candidate.STATIC_STATES):
                state_start = amplitude_start + (candidate.STATIC_STATES + static_state) * 5
                secondary[state_start] += math.log(0.65)
                secondary[state_start + 1:state_start + 5] *= 0.65
        starts.append(("lower_excited_vacuum_start", secondary))
    else:
        secondary = np.asarray(continuation_parameters, dtype=float)
        if secondary.shape != (candidate.PARAMETER_COUNT,) or not np.isfinite(secondary).all():
            raise ValueError("continuation start must be a finite full parameter vector")
        secondary = secondary.copy()
        for index in GROUND_A0_INDICES:
            if floor > 0:
                secondary[index] = max(secondary[index], math.log(floor) + 0.02)
        starts.append(("neighbor_force_continuation", secondary))
    return starts


def _profile_at_force(snapshot: Any, fit_parameters: np.ndarray,
                      floor: float, target_force: float, *,
                      continuation_parameters: np.ndarray | None = None) -> dict[str, Any]:
    _manifest, matrices = two_state._validate_raw_snapshot(
        snapshot, manifest_name="manifest", matrix_name="raw_matrices",
        contract_name="lane_contract",
    )
    blocks = two_state._observed_vectors(matrices)
    mean = blocks.mean(axis=0)
    covariance, covariance_diagnostics = two_state._regularized_mean_covariance(blocks)
    chol = cholesky(covariance, lower=True, check_finite=True)
    lower, upper = _parameter_bounds(floor, target_force)
    starts = _start_vectors(fit_parameters, floor, continuation_parameters)
    attempts = []
    valid = []
    reduced_indices = [index for index in range(candidate.PARAMETER_COUNT)
                       if index != FIXED_FORCE_INDEX]
    for start_name, full_start in starts:
        try:
            _full, _chain = _expand_force_coordinate(
                np.delete(full_start, FIXED_FORCE_INDEX), target_force,
            )
            initial = np.delete(_full, FIXED_FORCE_INDEX)
            initial = np.minimum(np.maximum(initial, lower + 1e-9), upper - 1e-9)

            def residual(reduced: np.ndarray) -> np.ndarray:
                full, _ = _expand_force_coordinate(reduced, target_force)
                return solve_triangular(
                    chol, candidate._model_vectors(full).ravel() - mean,
                    lower=True, check_finite=True,
                )

            def jacobian(reduced: np.ndarray) -> np.ndarray:
                full, chain = _expand_force_coordinate(reduced, target_force)
                raw_jac = _model_jacobian(full) @ chain
                return solve_triangular(chol, raw_jac, lower=True, check_finite=True)

            fit = least_squares(
                residual, initial, jac=jacobian, bounds=(lower, upper),
                method="trf", x_scale="jac", max_nfev=1600,
                ftol=1e-9, xtol=1e-9, gtol=1e-9,
            )
            singular = np.linalg.svd(fit.jac, compute_uv=False)
            tolerance = (singular[0] * max(fit.jac.shape) * np.finfo(float).eps
                         if singular.size else 0.0)
            rank = int(np.count_nonzero(singular > tolerance))
            full, _ = _expand_force_coordinate(fit.x, target_force)
            energy = candidate._decode(full)[1]
            force_error = abs(float(energy[1, 0] - energy[0, 0]) - target_force)
            amplitudes = candidate._decode(full)[2]
            ground_a0 = [float(abs(amplitudes[r, 0, 0, 0]))
                         for r in range(len(two_state.R_VALUES))]
            valid_fit = bool(fit.success and np.isfinite(fit.fun).all()
                             and rank == len(reduced_indices)
                             and force_error <= 1e-10
                             and (floor == 0 or min(ground_a0) >= floor - 1e-8))
            objective = float(np.dot(fit.fun, fit.fun)) if np.isfinite(fit.fun).all() else None
            attempt = {
                "start": start_name, "valid": valid_fit,
                "optimizer_success": bool(fit.success), "rank": rank,
                "parameter_count": len(reduced_indices),
                "condition_number": float(singular[0] / singular[-1]) if singular.size else None,
                "objective": objective, "nfev": int(fit.nfev),
                "force_constraint_error": force_error,
                "ground_a0_by_R": ground_a0,
                "covariance_diagnostics": covariance_diagnostics,
                "fit_coordinates": full.astype(float).tolist(),
                "message": str(fit.message),
            }
            attempts.append(attempt)
            if valid_fit:
                valid.append(attempt)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
            attempts.append({
                "start": start_name, "valid": False, "status": "FAILED_NONLINEAR_PROFILE",
                "error_type": type(exc).__name__, "error_message": str(exc),
            })
    best = min(valid, key=lambda row: float(row["objective"])) if valid else None
    return {
        "status": "VALID_NONLINEAR_PROFILE" if best else "ABSTAIN_NO_VALID_START",
        "valid": best is not None, "target_force": target_force,
        "overlap_floor": floor, "planned_starts": len(starts),
        "valid_starts": len(valid), "failed_or_abstained_starts": len(starts) - len(valid),
        "objective": best["objective"] if best else None,
        "condition_number": best["condition_number"] if best else None,
        "nfev_total": sum(int(row.get("nfev", 0)) for row in attempts),
        "attempts": attempts,
    }


def _profile_panel(snapshot: Any, fit_parameters: np.ndarray) -> dict[str, Any]:
    rows = [_profile_at_force(snapshot, fit_parameters, floor, force)
            for floor in FLOORS for force in FORCE_GRID]
    summaries = []
    for floor in FLOORS:
        group = [row for row in rows if row["overlap_floor"] == floor]
        valid = [row for row in group if row["valid"]]
        if len(valid) != len(FORCE_GRID):
            summaries.append({
                "overlap_floor": floor, "status": "ABSTAIN_INCOMPLETE_PROFILE_GRID",
                "planned": len(FORCE_GRID), "valid": len(valid),
                "failed_or_abstained": len(group) - len(valid), "grid_interval": None,
            })
            continue
        minimum = min(float(row["objective"]) for row in valid)
        accepted = [row["target_force"] for row in valid
                    if float(row["objective"]) - minimum <= LR_REFERENCE]
        summaries.append({
            "overlap_floor": floor,
            "status": "CONDITIONAL_UNCALIBRATED_NONLINEAR_PROFILE_SET" if accepted
            else "ABSTAIN_EMPTY_PROFILE_SET",
            "planned": len(FORCE_GRID), "valid": len(valid), "failed_or_abstained": 0,
            "minimum_objective": minimum,
            "profile_reference_delta_objective": LR_REFERENCE,
            "grid_interval": [float(min(accepted)), float(max(accepted))] if accepted else None,
            "grid_boundary_clipped": bool(accepted and (
                min(accepted) == FORCE_GRID[0] or max(accepted) == FORCE_GRID[-1]
            )),
        })
    return {"rows": rows, "per_floor": summaries}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = [{key: value for key, value in row.items() if key != "attempts"} for row in rows]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)


def _plot(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    for ax, floor in zip(axes, FLOORS):
        group = sorted([row for row in rows if row["overlap_floor"] == floor],
                       key=lambda row: row["target_force"])
        valid = [row for row in group if row["valid"]]
        if valid:
            minimum = min(float(row["objective"]) for row in valid)
            ax.plot([row["target_force"] for row in valid],
                    [float(row["objective"]) - minimum for row in valid],
                    marker="o", color="#0072b2")
        for row in group:
            if not row["valid"]:
                ax.scatter([row["target_force"]], [0.0], marker="x", color="#d55e00")
        ax.axhline(LR_REFERENCE, color="black", linestyle="--", linewidth=1)
        ax.axvline(0.15, color="#555555", linestyle=":", linewidth=1)
        ax.set_title(f"A0 floor={floor:g}" if floor else "No floor")
        ax.set_xlabel("fixed absolute force")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("profile Δobjective (lattice toy)")
    fig.suptitle("Nonlinear conditional profiles; crosses mark failed/abstained grids")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(output: Path = DEFAULT_OUT) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty report directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    matrices, contract_bytes, origin_hash, source_manifest_bytes, source_fit_bytes = (
        local_profile._read_weak_panel()
    )
    source_fit = json.loads(source_fit_bytes)
    fit = source_fit.get("fit")
    if not isinstance(fit, dict) or not isinstance(fit.get("fit_coordinates"), list):
        raise ValueError("captured weak-overlap fit receipt lacks parameters")
    fit_parameters = np.asarray(fit["fit_coordinates"], dtype=float)
    input_snapshot_dir = output / "input_snapshot"
    snapshot, snapshot_hashes = bootstrap._snapshot_from_matrices(
        input_snapshot_dir, matrices, contract_bytes,
        "F-synthetic-open-weak-overlap-nonlinear-profile",
    )
    (output / "source_panel_manifest.json").write_bytes(source_manifest_bytes)
    (output / "source_fit_receipt.json").write_bytes(source_fit_bytes)

    # Validate analytic derivatives against central finite differences before any profile run.
    analytic = _model_jacobian(fit_parameters)
    finite_difference = np.empty_like(analytic)
    for index, value in enumerate(fit_parameters):
        step = 1e-6 * max(1.0, abs(float(value)))
        plus, minus = fit_parameters.copy(), fit_parameters.copy()
        plus[index] += step
        minus[index] -= step
        finite_difference[:, index] = (
            candidate._model_vectors(plus).ravel() - candidate._model_vectors(minus).ravel()
        ) / (2.0 * step)
    jacobian_error = {
        "max_abs": float(np.max(np.abs(analytic - finite_difference))),
        "max_relative": float(np.max(
            np.abs(analytic - finite_difference) / np.maximum(1.0, np.abs(finite_difference)),
        )),
        "all_finite": bool(np.isfinite(analytic).all()),
    }
    profile = _profile_panel(snapshot, fit_parameters)
    _write_csv(output / "profile_grid.csv", profile["rows"])
    _write_json(output / "profile_fit_receipts.json", {
        "schema": "wilson-nonlinear-overlap-profile-fit-receipts-v1",
        "rows": profile["rows"],
    })
    gate = local_profile.assess_overlap_floor_receipt(None)
    summary = {
        "schema": "wilson-f-lane-nonlinear-overlap-profile-summary-v1",
        "classification": "CONDITIONAL_SYNTHETIC_NONLINEAR_PROFILE_FEASIBILITY",
        "certified": False, "physical_score": None,
        "source_raw_matrix_sha256": origin_hash,
        "analysis_snapshot_hashes": snapshot_hashes,
        "jacobian_validation": jacobian_error,
        "force_grid": list(FORCE_GRID), "floor_grid": list(FLOORS),
        "overlap_floor_definition": local_profile.OVERLAP_DEFINITION,
        "profile_by_floor": profile["per_floor"],
        "gate_without_independent_floor_receipt": gate,
        "denominator": {
            "planned_profile_cells": len(profile["rows"]),
            "valid_profile_cells": sum(row["valid"] for row in profile["rows"]),
            "failed_or_abstained_profile_cells": sum(not row["valid"] for row in profile["rows"]),
            "planned_start_fits": sum(row["planned_starts"] for row in profile["rows"]),
            "valid_start_fits": sum(row["valid_starts"] for row in profile["rows"]),
            "failed_or_abstained_start_fits": sum(
                row["failed_or_abstained_starts"] for row in profile["rows"]
            ),
        },
        "scope_limits": [
            "one reused synthetic weak-overlap panel is a conditional method screen, not independent calibration",
            "the endpoint-normalized overlap floors are hypothetical and have no physical unit receipt",
            "each grid cell has two starts only; nonlinear optimizer success and full rank do not prove global identifiability",
            "the nominal delta-objective threshold is not finite-sample calibrated",
            "production force remains uncertified without production matrices, independent floor/gap bounds, reference, review and authorization",
        ],
    }
    _write_json(output / "summary.json", summary)
    _plot(output / "nonlinear_profiles.png", profile["rows"])
    source_map = {
        "nonlinear_overlap_profile.py": Path(__file__),
        "local_overlap_profile_gate.py": Path(local_profile.__file__),
        "three_static_candidate.py": Path(candidate.__file__),
        "two_static_candidate.py": Path(two_state.__file__),
        "bootstrap_snapshot_helper.py": Path(bootstrap.__file__),
        "source_integrity.py": REPO_ROOT / "core/source_integrity.py",
        "npz_safety.py": REPO_ROOT / "core/npz_safety.py",
    }
    source_dir = output / "source_snapshot"
    source_dir.mkdir()
    source_hashes = {}
    for name, source_path in source_map.items():
        payload = source_path.read_bytes()
        source_hashes[name] = _sha(payload)
        (source_dir / name).write_bytes(payload)
    summary["source_hashes"] = source_hashes
    _write_json(output / "summary.json", summary)
    _write_json(output / "protocol.json", {
        "schema": "wilson-nonlinear-overlap-profile-protocol-v1",
        "source_panel": "02_weak_ground_overlap; intentionally reused from the prior frozen synthetic screen",
        "source_raw_matrix_sha256": origin_hash,
        "floor_values": list(FLOORS), "force_grid": list(FORCE_GRID),
        "profile_method": "exact finite-Nt forward model; analytically differentiated residual; eliminate E0(R=3) coordinate to fix target force exactly; two retained starts per cell",
        "jacobian_check": jacobian_error,
        "uncertainty_rule": "nominal objective difference <= 3.84 only; no finite-sample calibration claim",
        "physical_gate_without_receipt": gate["status"],
        "physical_status": "NOT_READY/UNCERTIFIED; F=GROUND_STATE_FORCE_UNCERTIFIED",
    })
    (output / "lane_contract.json").write_bytes(contract_bytes)
    readme = "\n".join([
        "# Nonlinear weak-overlap force profile", "",
        "**Classification:** synthetic conditional profile feasibility; not a calibrated interval or physical force result.", "",
        "![Nonlinear conditional profiles](nonlinear_profiles.png)", "",
        "## Method", "",
        "This reuses the earlier weak-overlap panel, fixes `F=E0(R=3)-E0(R=2)` exactly on the declared grid, and nonlinearly refits nuisance energies/overlaps under each hypothetical lower bound on `min_R |A[m=0,n=0,operator=0]|`. The finite-`Nt` model Jacobian is analytic; the report checks it against central finite differences before running the profiles. Each cell retains two starts and all failure/abstention outcomes.", "",
        f"Analytic Jacobian check: max absolute difference `{jacobian_error['max_abs']:.3g}`, max scaled difference `{jacobian_error['max_relative']:.3g}`, all finite `{jacobian_error['all_finite']}`.", "",
        "## Results and interpretation", "",
        f"Profile cells valid: `{summary['denominator']['valid_profile_cells']}/{summary['denominator']['planned_profile_cells']}`; failed/abstained: `{summary['denominator']['failed_or_abstained_profile_cells']}`. Start fits valid: `{summary['denominator']['valid_start_fits']}/{summary['denominator']['planned_start_fits']}`; failed/abstained starts: `{summary['denominator']['failed_or_abstained_start_fits']}`.",
        f"Without an independent production floor receipt the physical gate is `{gate['status']}`. No floor in this report can be used as physical evidence.",
        "Per-floor grid profile sets and clipping flags are in `summary.json`; every start's objective, rank, condition, nfev, force constraint and fit coordinates are in `profile_fit_receipts.json`.", "",
        "The plotted `Δobjective=3.84` rule is a nominal likelihood-ratio reference only. One reused panel, two starts per cell, covariance regularization, and grid spacing do not establish coverage or global identifiability. No production Wilson matrices, sealed split, GPU or QUDA was read; `physical_score=null` and F remains `GROUND_STATE_FORCE_UNCERTIFIED`.",
    ]) + "\n"
    (output / "README.md").write_text(readme, encoding="utf-8")
    artifacts = {
        str(path.relative_to(output)): _sha(path.read_bytes())
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_json(output / "artifact_hashes.json", {
        "schema": "wilson-nonlinear-overlap-profile-artifact-hashes-v1",
        "artifacts": artifacts,
    })
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({
        "jacobian_validation": result["jacobian_validation"],
        "denominator": result["denominator"],
        "gate_without_floor_receipt": result["gate_without_independent_floor_receipt"]["status"],
        "per_floor": result["profile_by_floor"],
    }, indent=2, sort_keys=True))
