"""Paired dogbox-versus-trf check on saved synthetic Wilson profile failures.

This is a bounded solver experiment on the same six panels, force grid,
covariance, overlap floor and start vectors. It is not interval calibration.
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
import scipy
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares

from tasks.wilson_loop import f_lane_three_static_profile_continuation_diagnostic as continuation
from tasks.wilson_loop import f_lane_three_static_nonlinear_overlap_profile as nonlinear
from tasks.wilson_loop import f_lane_three_static_two_vacuum_candidate as candidate
from tasks.wilson_loop import f_lane_two_sector_fit_candidate as two_state


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
INPUT_REPORT = TASK_DIR / "f_lane_three_static_stratified_profile_20261128_v1"
CONTINUATION_REPORT = TASK_DIR / "f_lane_three_static_profile_continuation_20261128_v1"
DEFAULT_OUT = TASK_DIR / "f_lane_three_static_profile_dogbox_20261128_v1"
TARGET_STRATA = ("near_degenerate", "omitted_vacuum_truth")
FLOOR = 0.9
MAX_NFEV = 1600
GTOL = 1e-9


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _verify_report(root: Path) -> dict[str, Any]:
    payload = (root / "artifact_hashes.json").read_bytes()
    manifest = json.loads(payload)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"invalid artifact manifest: {root}")
    for relative, expected in artifacts.items():
        path = (root / relative).resolve(strict=True)
        if root.resolve() not in path.parents:
            raise ValueError(f"artifact escapes report directory: {relative}")
        if _sha(path.read_bytes()) != expected:
            raise ValueError(f"artifact hash mismatch: {root.name}/{relative}")
    return {"manifest_sha256": _sha(payload), "planned": len(artifacts),
            "verified": len(artifacts), "failed": 0}


def _closest_valid_start(cells: list[dict[str, Any]], target_force: float
                         ) -> np.ndarray | None:
    options = []
    for cell in cells:
        if not cell.get("valid"):
            continue
        force = float(cell["target_force"])
        for attempt in cell.get("attempts", []):
            if attempt.get("valid") and isinstance(attempt.get("fit_coordinates"), list):
                options.append((abs(force - target_force), force,
                                float(attempt["objective"]), attempt["fit_coordinates"]))
    if not options:
        return None
    return np.asarray(min(options, key=lambda row: row[:3])[3], dtype=float)


def _reduced_start(full_start: np.ndarray, force: float, lower: np.ndarray,
                   upper: np.ndarray) -> np.ndarray:
    expanded, _chain = nonlinear._expand_force_coordinate(
        np.delete(np.asarray(full_start, dtype=float), nonlinear.FIXED_FORCE_INDEX), force,
    )
    initial = np.delete(expanded, nonlinear.FIXED_FORCE_INDEX)
    return np.minimum(np.maximum(initial, lower + 1e-9), upper - 1e-9)


def _solve_dogbox(start: np.ndarray, force: float, mean: np.ndarray,
                  covariance_cholesky: np.ndarray) -> dict[str, Any]:
    lower, upper = nonlinear._parameter_bounds(FLOOR, force)
    initial = _reduced_start(start, force, lower, upper)

    def residual(reduced: np.ndarray) -> np.ndarray:
        full, _chain = nonlinear._expand_force_coordinate(reduced, force)
        return solve_triangular(
            covariance_cholesky, candidate._model_vectors(full).ravel() - mean,
            lower=True, check_finite=True,
        )

    def jacobian(reduced: np.ndarray) -> np.ndarray:
        full, chain = nonlinear._expand_force_coordinate(reduced, force)
        return solve_triangular(
            covariance_cholesky, nonlinear._model_jacobian(full) @ chain,
            lower=True, check_finite=True,
        )

    fit = least_squares(
        residual, initial, jac=jacobian, bounds=(lower, upper), method="dogbox",
        x_scale="jac", max_nfev=MAX_NFEV, ftol=1e-9, xtol=1e-9, gtol=GTOL,
    )
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    tolerance = (singular[0] * max(fit.jac.shape) * np.finfo(float).eps
                 if singular.size else 0.0)
    rank = int(np.count_nonzero(singular > tolerance))
    full, _chain = nonlinear._expand_force_coordinate(fit.x, force)
    energy = candidate._decode(full)[1]
    amplitudes = candidate._decode(full)[2]
    force_error = abs(float(energy[1, 0] - energy[0, 0]) - force)
    ground_a0 = [float(abs(amplitudes[r, 0, 0, 0]))
                 for r in range(len(two_state.R_VALUES))]
    finite = bool(np.isfinite(fit.fun).all())
    objective = float(fit.fun @ fit.fun) if finite else None
    valid = bool(fit.success and finite and rank == fit.x.size
                 and force_error <= 1e-10 and min(ground_a0) >= FLOOR - 1e-8)
    return {
        "valid": valid,
        "optimizer_success": bool(fit.success),
        "message": str(fit.message),
        "nfev": int(fit.nfev),
        "objective": objective,
        "rank": rank,
        "parameter_count": int(fit.x.size),
        "condition_number": float(singular[0] / singular[-1]) if singular.size else None,
        "scaled_optimality": float(fit.optimality),
        "active_mask_count": int(np.count_nonzero(fit.active_mask)),
        "force_constraint_error": force_error,
        "minimum_ground_a0": min(ground_a0),
        "fit_coordinates": full.astype(float).tolist(),
    }


def _profile_from_invalid_cells(cold_cells: list[dict[str, Any]],
                                replacements: dict[float, list[dict[str, Any]]],
                                panel_summary: dict[str, Any]) -> dict[str, Any]:
    combined = []
    for cold_cell in cold_cells:
        force = float(cold_cell["target_force"])
        if cold_cell.get("valid"):
            combined.append(cold_cell)
            continue
        candidates = [row for row in replacements.get(force, []) if row.get("valid")]
        if candidates:
            best = min(candidates, key=lambda row: float(row["objective"]))
            combined.append({"valid": True, "target_force": force,
                             "objective": float(best["objective"])})
        else:
            combined.append({"valid": False, "target_force": force})
    return continuation._interval(combined, panel_summary)


def _plot(path: Path, panel_rows: list[dict[str, Any]], rescue_counts: dict[str, int]) -> None:
    methods = ("cold_trf", "neighbor_trf", "cold_dogbox", "neighbor_dogbox")
    labels = ("Cold TRF", "Neighbor TRF", "Cold dogbox", "Neighbor dogbox")
    colors = ("#999999", "#56b4e9", "#0072b2", "#d55e00")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), layout="constrained")
    axes[0].bar(range(4), [rescue_counts[key] for key in methods], color=colors)
    axes[0].set_xticks(range(4), labels, rotation=16, ha="right")
    axes[0].set_ylim(0, 13)
    axes[0].set_yticks(range(0, 14, 2))
    axes[0].set_ylabel("recovered cold-invalid force cells (of 13)")
    axes[0].set_title("Cell rescue under the same force grid")
    axes[0].grid(axis="y", alpha=0.22)
    for phase_index, (phase, label, color) in enumerate((
        ("cold_dogbox", "Dogbox from cold starts", "#0072b2"),
        ("neighbor_dogbox", "Dogbox from neighbor starts", "#d55e00"),
    )):
        strata = []
        complete = []
        for stratum in TARGET_STRATA:
            selected = [row for row in panel_rows if row["stratum"] == stratum]
            strata.append(stratum.replace("_", "\n"))
            complete.append(sum(bool(row[phase]["complete"]) for row in selected))
        positions = np.arange(len(TARGET_STRATA)) + (-0.18 if phase_index == 0 else 0.18)
        axes[1].bar(positions, complete, width=0.34, color=color, label=label)
    axes[1].set_xticks(range(len(TARGET_STRATA)), [s.replace("_", "\n") for s in TARGET_STRATA])
    axes[1].set_ylim(0, 3.2)
    axes[1].set_yticks(range(4))
    axes[1].set_ylabel("complete profiles (of 3 panels)")
    axes[1].set_title("Dogbox-completed profiles")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.22)
    fig.suptitle("Paired bounded solver screen on reused synthetic Wilson panels")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(output: Path = DEFAULT_OUT) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty report directory: {output}")
    input_integrity = {
        "stratified_profile_report": _verify_report(INPUT_REPORT),
        "continuation_report": _verify_report(CONTINUATION_REPORT),
    }
    source_manifest = json.loads((INPUT_REPORT / "artifact_hashes.json").read_text())
    continuation_receipts = json.loads(
        (CONTINUATION_REPORT / "rescue_receipts.json").read_text()
    )["panels"]
    jobs: dict[tuple[int, float, str], dict[str, Any]] = {}
    panel_data: dict[int, dict[str, Any]] = {}
    for rescue_entry in continuation_receipts:
        if rescue_entry["stratum"] not in TARGET_STRATA:
            raise ValueError("unexpected stratum in continuation report")
        panel_id = int(rescue_entry["panel"])
        panel_dir = (REPO_ROOT / rescue_entry["input_panel"]).resolve()
        snapshot, matrices, panel_hashes = continuation._capture_panel(panel_dir, source_manifest)
        cold_receipts = json.loads(snapshot.read_bytes("profile_receipts"))
        fit_receipt = json.loads(snapshot.read_bytes("fit"))
        point_fit = np.asarray(fit_receipt["fit"]["fit_coordinates"], dtype=float)
        blocks = two_state._observed_vectors(matrices)
        mean = blocks.mean(axis=0)
        covariance, _diagnostics = two_state._regularized_mean_covariance(blocks)
        covariance_cholesky = cholesky(covariance, lower=True, check_finite=True)
        cold_cells = [cell for cell in cold_receipts["profile_cells"]
                      if math.isclose(float(cell["overlap_floor"]), FLOOR,
                                      rel_tol=0.0, abs_tol=1e-12)]
        panel_data[panel_id] = {
            "entry": rescue_entry, "cold_receipts": cold_receipts,
            "panel_summary": cold_receipts["panel_summary"],
            "cold_cells": cold_cells, "fit_coordinates": point_fit,
            "mean": mean, "covariance_cholesky": covariance_cholesky,
            "panel_hashes": panel_hashes,
        }
        cold_valid = [cell for cell in cold_cells if cell.get("valid")]
        cold_invalid = [cell for cell in cold_cells if not cell.get("valid")]
        for rescue_cell in rescue_entry["cells"]:
            if rescue_cell["cold_valid"]:
                continue
            force = float(rescue_cell["target_force"])
            neighbor = _closest_valid_start(cold_valid, force)
            if neighbor is None:
                raise ValueError("failed target has no cold-valid neighbor to reconstruct")
            start_groups = {
                "cold": nonlinear._start_vectors(point_fit, FLOOR, None),
                "neighbor": nonlinear._start_vectors(point_fit, FLOOR, neighbor),
            }
            reference_attempts = {
                "cold": {row["start"]: row for row in rescue_cell["cold_attempts"]},
                "neighbor": {row["start"]: row for row in
                             rescue_cell["continuation_result"]["attempts"]},
            }
            for scope, starts in start_groups.items():
                for start_name, full_start in starts:
                    lower, upper = nonlinear._parameter_bounds(FLOOR, force)
                    reduced_start = _reduced_start(full_start, force, lower, upper)
                    key = (panel_id, force, _sha(np.asarray(reduced_start, dtype="<f8").tobytes()))
                    job = jobs.setdefault(key, {
                        "panel_id": panel_id, "force": force,
                        "stratum": rescue_entry["stratum"],
                        "replicate": int(rescue_entry["replicate"]),
                        "seed": int(rescue_entry["seed"]),
                        "start_name": start_name,
                        "start_sha256": key[2],
                        "start_coordinates": np.asarray(full_start, dtype=float),
                        "scopes": {},
                    })
                    baseline = reference_attempts[scope].get(start_name)
                    if baseline is None:
                        raise ValueError(f"saved TRF attempt is missing: {scope}/{start_name}")
                    job["scopes"][scope] = baseline

    job_rows = []
    for job in jobs.values():
        panel = panel_data[job["panel_id"]]
        result = _solve_dogbox(job["start_coordinates"], job["force"], panel["mean"],
                               panel["covariance_cholesky"])
        job["dogbox_result"] = result
        for scope, baseline in sorted(job["scopes"].items()):
            job_rows.append({
                "panel": job["panel_id"], "stratum": job["stratum"],
                "replicate": job["replicate"], "seed": job["seed"],
                "target_force": job["force"], "scope": scope,
                "start": job["start_name"], "start_sha256": job["start_sha256"],
                "trf_valid": bool(baseline["valid"]),
                "trf_nfev": int(baseline["nfev"]),
                "trf_objective": baseline.get("objective"),
                "dogbox_valid": bool(result["valid"]),
                "dogbox_nfev": result["nfev"],
                "dogbox_objective": result["objective"],
                "dogbox_rank": result["rank"],
                "dogbox_condition_number": result["condition_number"],
                "dogbox_scaled_optimality": result["scaled_optimality"],
                "dogbox_message": result["message"],
            })

    replacements = {"cold": {}, "neighbor": {}}
    for job in jobs.values():
        result = job["dogbox_result"]
        for scope in job["scopes"]:
            replacements[scope].setdefault((job["panel_id"], job["force"]), []).append(result)
    panel_rows = []
    rescue_counts = {"cold_trf": 0, "neighbor_trf": 0,
                     "cold_dogbox": 0, "neighbor_dogbox": 0}
    for panel_id, panel in sorted(panel_data.items()):
        entry = panel["entry"]
        for rescue_cell in entry["cells"]:
            if rescue_cell["cold_valid"]:
                continue
            force = float(rescue_cell["target_force"])
            if rescue_cell["continuation_result"].get("valid"):
                rescue_counts["neighbor_trf"] += 1
            for scope in ("cold", "neighbor"):
                results = replacements[scope].get((panel_id, force), [])
                if any(row["valid"] for row in results):
                    rescue_counts[f"{scope}_dogbox"] += 1
        cold_profile = continuation._interval(panel["cold_cells"], panel["panel_summary"])
        cold_dogbox = _profile_from_invalid_cells(
            panel["cold_cells"], {
                float(cell["target_force"]): replacements["cold"].get(
                    (panel_id, float(cell["target_force"])), []
                ) for cell in panel["cold_cells"] if not cell.get("valid")
            }, panel["panel_summary"],
        )
        neighbor_dogbox = _profile_from_invalid_cells(
            panel["cold_cells"], {
                float(cell["target_force"]): replacements["neighbor"].get(
                    (panel_id, float(cell["target_force"])), []
                ) for cell in panel["cold_cells"] if not cell.get("valid")
            }, panel["panel_summary"],
        )
        panel_rows.append({
            "panel": panel_id, "stratum": entry["stratum"],
            "replicate": int(entry["replicate"]), "seed": int(entry["seed"]),
            "cold_trf": {"complete": cold_profile.get("interval") is not None,
                         **cold_profile},
            "neighbor_trf": {"complete": entry["continuation_profile_summary"].get("interval") is not None,
                             **entry["continuation_profile_summary"]},
            "cold_dogbox": {"complete": cold_dogbox.get("interval") is not None,
                            **cold_dogbox},
            "neighbor_dogbox": {"complete": neighbor_dogbox.get("interval") is not None,
                                **neighbor_dogbox},
        })
    profile_summaries = {}
    for method in ("cold_trf", "neighbor_trf", "cold_dogbox", "neighbor_dogbox"):
        profile_summaries[method] = {}
        for stratum in TARGET_STRATA:
            selected = [row for row in panel_rows if row["stratum"] == stratum]
            complete = [row[method] for row in selected if row[method]["complete"]]
            unclipped = [row for row in complete if not row["grid_clipped"]]
            profile_summaries[method][stratum] = {
                "planned_panels": len(selected), "complete_profiles": len(complete),
                "unclipped_profiles": len(unclipped),
                "truth_contained_unclipped": sum(bool(row["contains_truth"]) for row in unclipped),
            }
    source_map = {
        "profile_dogbox_comparison.py": Path(__file__),
        "profile_continuation_diagnostic.py": Path(continuation.__file__),
        "nonlinear_overlap_profile.py": Path(nonlinear.__file__),
        "three_static_candidate.py": Path(candidate.__file__),
        "two_sector_fit_candidate.py": Path(two_state.__file__),
        "joint_spectral_gls.py": Path(candidate.forward.__file__),
        "source_integrity.py": REPO_ROOT / "core/source_integrity.py",
        "npz_safety.py": REPO_ROOT / "core/npz_safety.py",
    }
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / "source_snapshot"
    source_dir.mkdir()
    source_hashes = {}
    for name, source in source_map.items():
        payload = source.read_bytes()
        source_hashes[name] = _sha(payload)
        (source_dir / name).write_bytes(payload)
    csv_path = output / "paired_start_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(job_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(job_rows)
    _plot(output / "solver_comparison.png", panel_rows, rescue_counts)
    summary = {
        "schema": "wilson-f-lane-profile-dogbox-comparison-v1",
        "classification": "PAIRED_SYNTHETIC_BOUNDED_SOLVER_COMPARISON",
        "certified": False, "physical_score": None,
        "input_integrity": input_integrity,
        "input_reports": [str(INPUT_REPORT.relative_to(REPO_ROOT)),
                          str(CONTINUATION_REPORT.relative_to(REPO_ROOT))],
        "panel_count": len(panel_rows), "target_strata": list(TARGET_STRATA),
        "unique_solver_jobs": len(jobs),
        "paired_result_rows": len(job_rows),
        "solver": {
            "reference": "scipy.optimize.least_squares(method='trf', x_scale='jac') saved receipts",
            "candidate": "scipy.optimize.least_squares(method='dogbox', x_scale='jac')",
            "same_objective_covariance_bounds_starts_and_max_nfev": True,
            "max_nfev_per_start": MAX_NFEV,
            "ftol_xtol_gtol": [1e-9, 1e-9, GTOL],
            "floor": FLOOR,
        },
        "software_versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "failed_cell_rescues": rescue_counts,
        "profile_summaries": profile_summaries,
        "panels": panel_rows,
        "scope_limits": [
            "all start vectors and matrices are reused from the six existing synthetic panels",
            "no random seeds, production measurements, sealed split, GPU or QUDA are used",
            "each solver path keeps the existing point-fit/neighbor force starts and the fixed evaluation cap",
            "three panels per stratum are not enough to calibrate an interval or infer physical coverage",
            "dogbox is a numerical comparison only; a recovered fit does not certify identifiability",
        ],
        "physical_status": "NOT_READY/UNCERTIFIED; F=GROUND_STATE_FORCE_UNCERTIFIED",
        "source_hashes": source_hashes,
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "protocol.json", {
        "schema": "wilson-f-lane-profile-dogbox-comparison-protocol-v1",
        "input_strata": list(TARGET_STRATA), "panels": 6,
        "overlap_floor": FLOOR, "force_grid": "unchanged saved report grid",
        "comparison": "saved TRF starts and results versus Dogbox re-fit from byte-hash-bound reconstructed start vectors",
        "dogbox_method": "least_squares(method='dogbox', x_scale='jac')",
        "max_nfev_per_start": MAX_NFEV,
        "ftol_xtol_gtol": [1e-9, 1e-9, GTOL],
        "optimizer_start_vectors": ["captured_fit", "lower_excited_vacuum_start", "neighbor_force_continuation"],
        "optimizer_rerun_policy": "candidate Dogbox only; exact saved matrix/covariance/objective/bounds/start pairs",
        "random_seeds_added": 0,
        "physical_gate": "ABSTAIN_NO_INDEPENDENT_OVERLAP_FLOOR",
        "no_clobber": True,
    })
    readme = [
        "# Wilson bounded solver comparison: dogbox versus saved TRF", "",
        "**Classification:** paired numerical experiment on six reused synthetic panels; not a calibration study.", "",
        "![Profile cell recovery and completeness](solver_comparison.png)", "",
        f"The comparison reuses {len(panel_rows)} panels, the oracle toy floor `{FLOOR}`, the same force grid, full-covariance objective, analytic Jacobian, bounds, starts and `max_nfev={MAX_NFEV}`. Only the bounded least-squares method changes from the saved `trf` runs to `dogbox`; both use `x_scale='jac'`. The candidate solver ran {len(jobs)} unique start/force cases; duplicate captured-fit cases shared between the cold and neighbor branches were run once, then paired against both receipts.",
        "",
        "| Failed-cell branch | TRF recovered | Dogbox recovered |",
        "| --- | ---: | ---: |",
        f"| Cold captured + lower-excited starts | 0/13 | {rescue_counts['cold_dogbox']}/13 |",
        f"| Captured + nearest-force neighbor starts | {rescue_counts['neighbor_trf']}/13 | {rescue_counts['neighbor_dogbox']}/13 |",
        "",
        "| Method | Stratum | Complete profiles | Unclipped truth-containing profiles |",
        "| --- | --- | ---: | ---: |",
    ]
    method_labels = {"cold_trf": "Cold TRF", "neighbor_trf": "Neighbor TRF",
                     "cold_dogbox": "Cold dogbox", "neighbor_dogbox": "Neighbor dogbox"}
    for method, strata in profile_summaries.items():
        for stratum, item in strata.items():
            readme.append(
                f"| {method_labels[method]} | {stratum} | {item['complete_profiles']}/3 | "
                f"{item['truth_contained_unclipped']}/{item['unclipped_profiles']} |"
            )
    readme += [
        "",
        "These six panels only support a solver-failure comparison on the selected toy strata. If dogbox recovers cells, it changes numerical completion but does not establish that the `Δobjective<=3.84` profile set has valid coverage; the single existing near-degenerate interval remains a miss unless an actual reported result shows otherwise. No floor, start, force-grid, objective, covariance, eligibility or truth rule was changed.",
        "",
        "The paired attempt table records start-vector SHA, TRF and dogbox validity, objective, evaluation count, rank, condition number and dogbox scaled optimality. Source snapshots and input artifact hashes make this run auditable. No production Wilson input, sealed data, GPU or QUDA was touched. C finite-T Creutz and F ground-state force remain separate; status stays `NOT_READY/UNCERTIFIED`, F=`GROUND_STATE_FORCE_UNCERTIFIED`, `physical_score=null`.",
    ]
    (output / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    artifact_hashes = {
        str(path.relative_to(output)): _sha(path.read_bytes())
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_json(output / "artifact_hashes.json", {
        "schema": "wilson-profile-dogbox-artifact-hashes-v1",
        "artifacts": artifact_hashes,
    })
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"unique_solver_jobs": result["unique_solver_jobs"],
                      "failed_cell_rescues": result["failed_cell_rescues"],
                      "profile_summaries": result["profile_summaries"]},
                     indent=2, sort_keys=True))
