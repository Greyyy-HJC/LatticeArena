"""Conditional overlap-floor profile diagnostic for the Wilson F-lane toy.

This module evaluates one already-published synthetic weak-overlap panel. Its
conditional profile curves are not calibrated intervals and cannot certify a
production force. A physical overlap floor is admissible only with a separate,
hash-bound, independently reviewed production receipt.
"""
from __future__ import annotations

import csv
import hashlib
from io import BytesIO
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

from core.npz_safety import validate_npz_payload
from tasks.wilson_loop import f_lane_absolute_force_bootstrap_diagnostic as bootstrap
from tasks.wilson_loop import f_lane_three_static_identifiability_screen as stress
from tasks.wilson_loop import f_lane_three_static_two_vacuum_candidate as candidate
from tasks.wilson_loop import f_lane_two_sector_fit_candidate as two_state


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
INPUT_REPORT = TASK_DIR / "f_lane_three_static_identifiability_20261128_v1"
DEFAULT_OUT = TASK_DIR / "f_lane_three_static_overlap_profile_20261128_v1"
FLOORS = (0.0, 0.045, 0.075)
FORCE_GRID = (0.14, 0.15, 0.15757559165100204, 0.16, 0.17, 0.18)
LR_REFERENCE = 3.84
LOCAL_NONLINEAR_MISMATCH_MAX = 0.10
FORCE_LINEARIZATION_ABS_TOL = 0.0025
LOCAL_MAX_ABS_PARAMETER_SHIFT = 0.25
GROUND_A0_PARAMETER_INDICES = (4, 37)
OVERLAP_DEFINITION = "min_R abs(A[m=0,n=0,operator=0]) in the toy's normalized endpoint convention"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _read_weak_panel() -> tuple[np.ndarray, bytes, str, bytes, bytes]:
    artifact_manifest = json.loads((INPUT_REPORT / "artifact_hashes.json").read_text())
    relative = "panels/02_weak_ground_overlap/raw_matrices.npz"
    raw_bytes = (INPUT_REPORT / relative).read_bytes()
    if _sha(raw_bytes) != artifact_manifest["artifacts"].get(relative):
        raise ValueError("weak-overlap source panel does not match its captured artifact digest")
    validate_npz_payload(raw_bytes)
    with np.load(BytesIO(raw_bytes), allow_pickle=False) as archive:
        if set(archive.files) != {
            "schema", "block_ids", "correlator_matrices", "R_values",
            "temporal_extents", "time_slices",
        }:
            raise ValueError("weak-overlap source NPZ keys mismatch")
        matrices = np.asarray(archive["correlator_matrices"], dtype=np.complex128)
    contract_bytes = (INPUT_REPORT / "lane_contract.json").read_bytes()
    source_manifest_bytes = (INPUT_REPORT / "panels/02_weak_ground_overlap/manifest.json").read_bytes()
    source_fit_bytes = (INPUT_REPORT / "panels/02_weak_ground_overlap/fit.json").read_bytes()
    for name, payload in (
        ("lane_contract.json", contract_bytes),
        ("panels/02_weak_ground_overlap/manifest.json", source_manifest_bytes),
        ("panels/02_weak_ground_overlap/fit.json", source_fit_bytes),
    ):
        if _sha(payload) != artifact_manifest["artifacts"].get(name):
            raise ValueError(f"weak-overlap source receipt does not match its captured artifact digest: {name}")
    return matrices, contract_bytes, _sha(raw_bytes), source_manifest_bytes, source_fit_bytes


def _linearized_profile_context(snapshot: Any, fit_coordinates: np.ndarray) -> dict[str, Any]:
    _manifest, matrices = two_state._validate_raw_snapshot(
        snapshot, manifest_name="manifest", matrix_name="raw_matrices",
        contract_name="lane_contract",
    )
    blocks = two_state._observed_vectors(matrices)
    mean = blocks.mean(axis=0)
    covariance, cov_diag = two_state._regularized_mean_covariance(blocks)
    chol = cholesky(covariance, lower=True, check_finite=True)
    def residual(parameters: np.ndarray) -> np.ndarray:
        raw = candidate._model_vectors(parameters).ravel() - mean
        return solve_triangular(chol, raw, lower=True, check_finite=True)

    point = np.asarray(fit_coordinates, dtype=float)
    base = residual(point)
    jacobian = np.empty((base.size, point.size), dtype=float)
    for index in range(point.size):
        step = 1e-5 * max(1.0, abs(float(point[index])))
        plus, minus = point.copy(), point.copy()
        plus[index] += step
        minus[index] -= step
        jacobian[:, index] = (residual(plus) - residual(minus)) / (2.0 * step)
    energies = candidate._decode(point)[1]
    return {
        "point": point, "residual": base, "jacobian": jacobian,
        "force": float(energies[1, 0] - energies[0, 0]),
        "force_gradient": candidate._absolute_force_gradient(point),
        "covariance_diagnostics": cov_diag,
        "jacobian_rank": int(np.linalg.matrix_rank(jacobian)),
        "residual_at": residual,
    }


def _constrained_quadratic_profile(context: dict[str, Any], floor: float,
                                   target_force: float) -> dict[str, Any]:
    """Solve a local Gauss-Newton profile by enumerating active floor bounds."""
    point, residual, jacobian = context["point"], context["residual"], context["jacobian"]
    gradient = context["force_gradient"]
    hessian = jacobian.T @ jacobian
    linear = jacobian.T @ residual
    rhs_force = target_force - context["force"]
    floor_rhs = [math.log(floor) - float(point[index]) if floor > 0 else None
                 for index in GROUND_A0_PARAMETER_INDICES]
    candidates = []
    for active_mask in range(1 << len(GROUND_A0_PARAMETER_INDICES)):
        rows, values, active_indices = [gradient], [rhs_force], []
        for bit, index in enumerate(GROUND_A0_PARAMETER_INDICES):
            if active_mask & (1 << bit) and floor > 0:
                unit = np.zeros_like(point)
                unit[index] = 1.0
                rows.append(unit)
                values.append(float(floor_rhs[bit]))
                active_indices.append(index)
        constraints = np.stack(rows)
        value_array = np.asarray(values, dtype=float)
        try:
            unconstrained = np.linalg.solve(hessian, -linear)
            hessian_constraint = np.linalg.solve(hessian, constraints.T)
            constraint_gram = constraints @ hessian_constraint
            lagrange = np.linalg.solve(
                constraint_gram, value_array - constraints @ unconstrained,
            )
            delta = unconstrained + hessian_constraint @ lagrange
        except np.linalg.LinAlgError:
            continue
        equality_error = float(np.max(np.abs(constraints @ delta - values)))
        if equality_error > 1e-6:
            continue
        if floor > 0 and any(
            point[index] + delta[index] < math.log(floor) - 1e-7
            for index in GROUND_A0_PARAMETER_INDICES
        ):
            continue
        profiled_residual = residual + jacobian @ delta
        objective = float(np.dot(profiled_residual, profiled_residual))
        base_objective = float(np.dot(residual, residual))
        max_abs_shift = float(np.max(np.abs(delta)))
        try:
            nonlinear_residual = context["residual_at"](point + delta)
            nonlinear_objective = float(np.dot(nonlinear_residual, nonlinear_residual))
            nonlinear_mismatch = abs(nonlinear_objective - objective) / max(
                abs(objective - base_objective), LR_REFERENCE,
            )
            nonlinear_force = float(candidate._decode(point + delta)[1][1, 0]
                                     - candidate._decode(point + delta)[1][0, 0])
            force_error = abs(nonlinear_force - target_force)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, OverflowError):
            nonlinear_mismatch, force_error, nonlinear_objective = None, None, None
        locally_valid = (nonlinear_mismatch is not None and force_error is not None
                         and nonlinear_mismatch <= LOCAL_NONLINEAR_MISMATCH_MAX
                         and force_error <= FORCE_LINEARIZATION_ABS_TOL
                         and max_abs_shift <= LOCAL_MAX_ABS_PARAMETER_SHIFT)
        candidates.append((objective, active_indices, delta, equality_error,
                           nonlinear_mismatch, force_error, locally_valid,
                           nonlinear_objective, max_abs_shift))
    if not candidates:
        return {"status": "ABSTAIN_NO_FEASIBLE_LOCAL_PROFILE", "valid": False,
                "target_force": target_force, "overlap_floor": floor}
    (objective, active, delta, equality_error, nonlinear_mismatch, force_error,
     locally_valid, nonlinear_objective, max_abs_shift) = min(
        candidates, key=lambda item: item[0],
    )
    return {
        "status": ("VALID_LOCAL_LINEARIZED_PROFILE" if locally_valid
                   else "ABSTAIN_LOCAL_LINEARIZATION_BREAKDOWN"),
        "valid": locally_valid,
        "target_force": target_force, "overlap_floor": floor,
        "objective": objective, "active_floor_parameter_indices": active,
        "force_constraint_linearization_error": equality_error,
        "nonlinear_residual_mismatch_fraction": nonlinear_mismatch,
        "nonlinear_objective": nonlinear_objective,
        "nonlinear_force_constraint_error": force_error,
        "max_abs_parameter_shift": max_abs_shift,
        "local_approximation_thresholds": {
            "residual_mismatch_fraction_max": LOCAL_NONLINEAR_MISMATCH_MAX,
            "force_error_absolute_max": FORCE_LINEARIZATION_ABS_TOL,
            "max_abs_parameter_shift": LOCAL_MAX_ABS_PARAMETER_SHIFT,
        },
        "estimated_ground_a0_min_linearized": min(
            math.exp(float(point[index] + delta[index]))
            for index in GROUND_A0_PARAMETER_INDICES
        ),
    }


def profile_weak_overlap_panel(snapshot: Any, unconstrained_fit: dict[str, Any],
                               floors: tuple[float, ...] = FLOORS,
                               force_grid: tuple[float, ...] = FORCE_GRID) -> dict[str, Any]:
    start = np.asarray(unconstrained_fit["fit_coordinates"], dtype=float)
    context = _linearized_profile_context(snapshot, start)
    rows = [_constrained_quadratic_profile(context, floor, force)
            for floor in floors for force in force_grid]
    per_floor = []
    for floor in floors:
        subset = [row for row in rows if row["overlap_floor"] == floor]
        valid = [row for row in subset if row["valid"]]
        if len(valid) != len(force_grid):
            per_floor.append({
                "overlap_floor": floor, "status": "ABSTAIN_INCOMPLETE_PROFILE_GRID",
                "planned": len(force_grid), "valid": len(valid),
                "failed": sum(row["status"].startswith("FAILED") for row in subset),
                "abstained": len(subset) - len(valid), "grid_interval": None,
            })
            continue
        minimum = min(float(row["objective"]) for row in valid)
        accepted = [row["target_force"] for row in valid
                    if float(row["objective"]) - minimum <= LR_REFERENCE]
        if not accepted:
            status, interval = "ABSTAIN_EMPTY_PROFILE_SET", None
        else:
            status = "CONDITIONAL_UNCALIBRATED_LOCAL_PROFILE_SET"
            interval = [float(min(accepted)), float(max(accepted))]
        per_floor.append({
            "overlap_floor": floor, "status": status,
            "planned": len(force_grid), "valid": len(valid), "failed": 0,
            "abstained": len(subset) - len(valid),
            "minimum_objective": minimum,
            "profile_reference_delta_objective": LR_REFERENCE,
            "grid_interval": interval,
            "grid_boundary_clipped": bool(interval and (
                interval[0] == force_grid[0] or interval[1] == force_grid[-1]
            )),
        })
    return {
        "rows": rows, "per_floor": per_floor,
        "linearization": {
            "center_force": context["force"],
            "jacobian_rank": context["jacobian_rank"],
            "parameter_count": int(context["point"].size),
            "covariance_diagnostics": context["covariance_diagnostics"],
            "method": "central finite-difference whitened Jacobian plus active-set Gauss-Newton KKT",
        },
    }


def assess_overlap_floor_receipt(receipt: object) -> dict[str, Any]:
    """Fail closed unless an independent production receipt supports the floor."""
    base = {"physical_profile_permitted": False, "physical_interval": None}
    if not isinstance(receipt, dict):
        return {**base, "status": "ABSTAIN_NO_INDEPENDENT_OVERLAP_FLOOR"}
    hashes = receipt.get("source_sha256_by_radius")
    review_hash = receipt.get("independent_review_sha256")
    valid_hash = lambda item: (isinstance(item, str) and len(item) == 64
                               and all(char in "0123456789abcdef" for char in item))
    bounds = receipt.get("lower_bound_by_radius")
    valid = (
        receipt.get("schema") == "wilson-ground-overlap-floor-receipt-v1"
        and receipt.get("source_kind") == "production_open_calibration"
        and receipt.get("review_status") == "accepted"
        and receipt.get("independent_of_fit") is True
        and receipt.get("overlap_definition") == OVERLAP_DEFINITION
        and isinstance(receipt.get("split_id"), str)
        and receipt["split_id"].startswith("F")
        and not any(marker in receipt["split_id"].lower()
                    for marker in ("synthetic", "sealed", "holdout", "confirmation"))
        and isinstance(bounds, dict)
        and all(isinstance(bounds.get(str(radius)), (int, float))
                and math.isfinite(float(bounds[str(radius)]))
                and float(bounds[str(radius)]) > 0 for radius in two_state.R_VALUES)
        and isinstance(hashes, dict)
        and all(valid_hash(hashes.get(str(radius))) for radius in two_state.R_VALUES)
        and valid_hash(review_hash)
    )
    if not valid:
        return {**base, "status": "ABSTAIN_INVALID_OR_NONINDEPENDENT_FLOOR_RECEIPT"}
    return {
        "status": "CONDITIONAL_PROFILE_ALLOWED_REQUIRES_CALIBRATION",
        "physical_profile_permitted": True,
        "physical_interval": None,
        "condition": "profile only at min_R floor=min_r lower_bound_by_radius[r]",
        "certified": False,
    }


def _plot(path: Path, rows: list[dict[str, Any]], floors: tuple[float, ...]) -> None:
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    groups = (floors[:-1], floors[-1:])
    for ax, floor_group in zip(axes, groups):
        for floor in floor_group:
            subset = sorted((row for row in rows
                             if row["overlap_floor"] == floor and row.get("objective") is not None),
                            key=lambda row: row["target_force"])
            if not subset:
                continue
            minimum = min(float(row["objective"]) for row in subset)
            xs = [row["target_force"] for row in subset]
            ys = [float(row["objective"]) - minimum for row in subset]
            label = f"A0 floor={floor:g}" if floor else "no floor"
            ax.plot(xs, ys, alpha=0.6, label=label)
            for row, x_value, y_value in zip(subset, xs, ys):
                ax.scatter([x_value], [y_value], marker="o" if row["valid"] else "x",
                           s=36, zorder=3)
        ax.axhline(LR_REFERENCE, color="black", linestyle="--", linewidth=1,
                   label="nominal Δobjective=3.84")
        ax.axvline(0.15, color="#555555", linestyle=":", linewidth=1.4,
                   label="synthetic truth 0.15")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], color="#555555", marker="x", linestyle="None",
                               label="fails local-fidelity screen"))
        labels.append("fails local-fidelity screen")
        ax.set_xlabel("linearized absolute force (lattice units)")
        ax.grid(alpha=0.22)
        ax.legend(handles, labels, fontsize=8)
    axes[0].set_ylabel("local profile Δobjective")
    axes[0].set_title("Floors 0 and 0.045")
    axes[1].set_title("Floor 0.075; crosses fail local-fidelity screen")
    fig.suptitle("Conditional overlap-floor profile (independent vertical scales)")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path = DEFAULT_OUT) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty report directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    matrices, contract_bytes, origin_hash, source_manifest_bytes, source_fit_bytes = _read_weak_panel()
    source_fit = json.loads(source_fit_bytes)
    fit = source_fit.get("fit")
    if not isinstance(fit, dict) or not isinstance(fit.get("fit_coordinates"), list):
        raise ValueError("captured weak-overlap fit receipt lacks fit coordinates")
    panel_dir = output / "input_snapshot"
    snapshot, snapshot_hashes = bootstrap._snapshot_from_matrices(
        panel_dir, matrices, contract_bytes,
        "F-synthetic-open-weak-overlap-conditional-profile",
    )
    (output / "source_panel_manifest.json").write_bytes(source_manifest_bytes)
    (output / "source_fit_receipt.json").write_bytes(source_fit_bytes)
    profile = profile_weak_overlap_panel(snapshot, fit)
    _write_csv(output / "profile_grid.csv", profile["rows"])
    gate = assess_overlap_floor_receipt(None)
    summary = {
        "schema": "wilson-f-lane-overlap-profile-summary-v1",
        "classification": "CONDITIONAL_SYNTHETIC_PROFILE_SENSITIVITY",
        "certified": False, "physical_score": None,
        "source_raw_matrix_sha256": origin_hash,
        "analysis_snapshot_hashes": snapshot_hashes,
        "overlap_definition": OVERLAP_DEFINITION,
        "floor_values": list(FLOORS), "force_grid": list(FORCE_GRID),
        "profile_reference_delta_objective": LR_REFERENCE,
        "local_linearization": profile["linearization"],
        "local_approximation_thresholds": {
            "nonlinear_objective_mismatch_fraction_max": LOCAL_NONLINEAR_MISMATCH_MAX,
            "force_constraint_absolute_error_max": FORCE_LINEARIZATION_ABS_TOL,
            "maximum_parameter_shift": LOCAL_MAX_ABS_PARAMETER_SHIFT,
        },
        "gate_without_independent_floor_receipt": gate,
        "profile_by_floor": profile["per_floor"],
        "denominator": {
            "planned_profile_points": len(profile["rows"]),
            "valid_profile_points": sum(row["valid"] for row in profile["rows"]),
            "failed_profile_points": sum(row["status"].startswith("FAILED")
                                          for row in profile["rows"]),
            "abstained_profile_points": sum(not row["valid"] and not row["status"].startswith("FAILED")
                                             for row in profile["rows"]),
        },
        "scope_limits": [
            "the source is one reused synthetic weak-overlap panel and is not independent calibration",
            "the overlap floor is in the toy endpoint normalization and is not a physical receipt",
            "the local Gauss-Newton profile and nominal delta-objective threshold are not finite-sample calibrated",
            "the local profile is a first-order approximation, not a nonlinear nuisance refit",
            "the fixed-force grid is coarse and may clip a conditional profile set",
            "without an independent open-calibration production floor receipt, the physical gate abstains",
        ],
    }
    _write_json(output / "summary.json", summary)
    _plot(output / "conditional_profiles.png", profile["rows"], FLOORS)
    source_map = {
        "overlap_profile.py": Path(__file__),
        "three_static_candidate.py": Path(candidate.__file__),
        "identifiability_screen.py": Path(stress.__file__),
        "two_static_candidate.py": Path(two_state.__file__),
        "bootstrap_snapshot_helper.py": Path(bootstrap.__file__),
        "source_integrity.py": REPO_ROOT / "core/source_integrity.py",
        "npz_safety.py": REPO_ROOT / "core/npz_safety.py",
    }
    snapshot_dir = output / "source_snapshot"
    snapshot_dir.mkdir()
    source_hashes = {}
    for name, path in source_map.items():
        payload = path.read_bytes()
        source_hashes[name] = _sha(payload)
        (snapshot_dir / name).write_bytes(payload)
    summary["source_hashes"] = source_hashes
    _write_json(output / "summary.json", summary)
    _write_json(output / "protocol.json", {
        "schema": "wilson-overlap-profile-protocol-v1",
        "input_origin": str(INPUT_REPORT.relative_to(REPO_ROOT)),
        "source_panel": "02_weak_ground_overlap; same previously analyzed panel, intentionally reused",
        "source_matrix_sha256": origin_hash,
        "overlap_definition": OVERLAP_DEFINITION,
        "floor_values": list(FLOORS), "fixed_force_grid": list(FORCE_GRID),
        "profile_rule": "linearize whitened residuals at the captured weak-overlap fit; impose first-order force equality and active overlap-floor inequalities; solve a Gauss-Newton KKT system at each grid point; report objective-minus-minimum <= 3.84",
        "optimizer_starts": "no nonlinear optimizer; one local expansion around the captured weak-overlap fit coordinates",
        "local_approximation_thresholds": {
            "nonlinear_objective_mismatch_fraction_max": LOCAL_NONLINEAR_MISMATCH_MAX,
            "force_constraint_absolute_error_max": FORCE_LINEARIZATION_ABS_TOL,
            "maximum_parameter_shift": LOCAL_MAX_ABS_PARAMETER_SHIFT,
        },
        "receipt_gate": "production floor is not accepted without independent open-calibration source and review hashes",
        "physical_status": "NOT_READY/UNCERTIFIED; F=GROUND_STATE_FORCE_UNCERTIFIED",
    })
    _write_json(output / "lane_contract.json", json.loads(contract_bytes))
    readme = "\n".join([
        "# Conditional weak-overlap force profile", "",
        "**Classification:** one-panel synthetic sensitivity analysis. The floor values are hypothetical in the toy endpoint normalization. This is not physical calibration or a production confidence interval.", "",
        "![Conditional profile curves](conditional_profiles.png)", "",
        "## Decision rule", "",
        "```text",
        "no independent open-calibration overlap-floor receipt",
        "  -> ABSTAIN_NO_INDEPENDENT_OVERLAP_FLOOR",
        "  -> do not emit a physical force interval or physical score",
        "```", "",
        "The synthetic curve computation is a conditional local-sensitivity exercise. It reuses the weak-ground-overlap panel from the preceding screen, computes one central finite-difference whitened residual Jacobian at its captured fit, and constrains the linearized force `F=E0(R=3)-E0(R=2)` on a coarse grid. For each floor/grid point, an active-set Gauss-Newton KKT solve profiles the local nuisance displacement, with the floor applied to `min_R |A[m=0,n=0,operator=0]|`. This is a **linearized local profile**, not a nonlinear refit. A point is retained as locally valid only if its nonlinear objective differs from the linearized objective by at most 10% of the larger profile change or 3.84, its exact force mismatch is at most `0.0025`, and its largest parameter shift is at most `0.25`. The nominal `Δobjective=3.84` line is not calibrated for 95% coverage.", "",
        "## Receipts and limits", "",
        f"- Raw-matrix origin hash: `{origin_hash}`",
        f"- Local profile points: `{summary['denominator']['valid_profile_points']}/{summary['denominator']['planned_profile_points']}` valid; failed=`{summary['denominator']['failed_profile_points']}`; abstained=`{summary['denominator']['abstained_profile_points']}`",
        f"- The independent-floor gate is `{gate['status']}`. No production overlap floor or review receipt exists in this report.",
        "- Per-floor conditional grid sets and clipping flags are in `summary.json`; pointwise profile rows are in `profile_grid.csv`.",
        "- The source matrix, captured fit, contract, profile data, code snapshots, and checksums are retained for audit.", "",
        "No production Wilson data, sealed split, GPU, or QUDA was read. `physical_score=null`; both lanes remain `NOT_READY/UNCERTIFIED`, F remains `GROUND_STATE_FORCE_UNCERTIFIED`.",
    ]) + "\n"
    (output / "README.md").write_text(readme, encoding="utf-8")
    artifacts = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_hashes.json":
            artifacts[str(path.relative_to(output))] = _sha(path.read_bytes())
    _write_json(output / "artifact_hashes.json", {
        "schema": "wilson-overlap-profile-artifact-hashes-v1", "artifacts": artifacts,
    })
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({
        "denominator": result["denominator"],
        "gate_without_floor_receipt": result["gate_without_independent_floor_receipt"]["status"],
        "per_floor": result["profile_by_floor"],
    }, indent=2, sort_keys=True))
