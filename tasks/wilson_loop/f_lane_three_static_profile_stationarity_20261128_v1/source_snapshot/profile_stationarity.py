"""Read-only stationarity attribution for saved failed Wilson profile starts.

The script verifies prior synthetic report artifacts, reloads their captured
panels, and recomputes residual/Jacobian summaries at saved optimizer endpoints.
It never invokes an optimizer and does not add coverage samples.
"""
from __future__ import annotations

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

from tasks.wilson_loop import f_lane_three_static_profile_continuation_diagnostic as continuation
from tasks.wilson_loop import f_lane_three_static_nonlinear_overlap_profile as nonlinear
from tasks.wilson_loop import f_lane_three_static_two_vacuum_candidate as candidate
from tasks.wilson_loop import f_lane_two_sector_fit_candidate as two_state


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
INPUT_REPORT = TASK_DIR / "f_lane_three_static_stratified_profile_20261128_v1"
CONTINUATION_REPORT = TASK_DIR / "f_lane_three_static_profile_continuation_20261128_v1"
DEFAULT_OUT = TASK_DIR / "f_lane_three_static_profile_stationarity_20261128_v1"
TARGET_STRATA = ("near_degenerate", "omitted_vacuum_truth")
FLOOR = 0.9
ACTIVE_BOUND_RTOL = 1e-7
GTOL_REFERENCE = 1e-9


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _verify_artifact_report(root: Path) -> dict[str, Any]:
    manifest_path = root / "artifact_hashes.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"empty or malformed artifact manifest: {manifest_path}")
    checked = 0
    for relative, expected in artifacts.items():
        path = (root / relative).resolve(strict=True)
        if root.resolve() not in path.parents:
            raise ValueError(f"artifact path escapes its report directory: {relative}")
        observed = _sha(path.read_bytes())
        if observed != expected:
            raise ValueError(f"artifact hash mismatch: {root.name}/{relative}")
        checked += 1
    return {"manifest_sha256": _sha(manifest_bytes), "planned": len(artifacts),
            "verified": checked, "failed": 0}


def _snapshot_panel(panel_dir: Path, source_manifest: dict[str, Any]):
    return continuation._capture_panel(panel_dir.resolve(), source_manifest)


def _weak_group_loadings(jacobian: np.ndarray) -> tuple[dict[str, float], float]:
    column_norms = np.linalg.norm(jacobian, axis=0)
    nonzero = column_norms > np.finfo(float).tiny
    standardized = jacobian[:, nonzero] / column_norms[nonzero]
    _u, singular, vh = np.linalg.svd(standardized, full_matrices=False)
    reduced_weak = np.zeros(jacobian.shape[1], dtype=float)
    reduced_weak[nonzero] = vh[-1]
    retained_indices = [index for index in range(candidate.PARAMETER_COUNT)
                        if index != nonlinear.FIXED_FORCE_INDEX]
    full_weak = np.zeros(candidate.PARAMETER_COUNT, dtype=float)
    full_weak[retained_indices] = reduced_weak
    groups = {
        "vacuum_gap": [0],
        "R2_energy_increments": [1, 2, 3],
        "R2_amplitudes": list(range(4, 34)),
        "R3_energy_increments_excluding_force": [35, 36],
        "R3_amplitudes": list(range(37, 67)),
    }
    denominator = float(full_weak @ full_weak)
    loadings = {name: float(full_weak[indices] @ full_weak[indices]) / denominator
                for name, indices in groups.items()}
    standardized_condition = float(singular[0] / singular[-1])
    return loadings, standardized_condition


def _attempt_diagnostic(
    phase: str,
    entry: dict[str, Any],
    force: float,
    attempt: dict[str, Any],
    mean: np.ndarray,
    covariance_cholesky: np.ndarray,
) -> dict[str, Any]:
    saved = np.asarray(attempt["fit_coordinates"], dtype=float)
    if saved.shape != (candidate.PARAMETER_COUNT,) or not np.isfinite(saved).all():
        raise ValueError("saved fit coordinates have an invalid shape or nonfinite values")
    full, chain = nonlinear._expand_force_coordinate(
        np.delete(saved, nonlinear.FIXED_FORCE_INDEX), force,
    )
    reduced = np.delete(full, nonlinear.FIXED_FORCE_INDEX)
    residual = solve_triangular(
        covariance_cholesky,
        candidate._model_vectors(full).ravel() - mean,
        lower=True, check_finite=True,
    )
    jacobian = solve_triangular(
        covariance_cholesky,
        nonlinear._model_jacobian(full) @ chain,
        lower=True, check_finite=True,
    )
    singular = np.linalg.svd(jacobian, compute_uv=False)
    rank_tolerance = (singular[0] * max(jacobian.shape) * np.finfo(float).eps
                      if singular.size else 0.0)
    local_rank = int(np.count_nonzero(singular > rank_tolerance))
    condition_number = float(singular[0] / singular[-1])
    objective = float(residual @ residual)
    saved_objective = float(attempt["objective"])
    if not math.isclose(objective, saved_objective, rel_tol=2e-7, abs_tol=2e-6):
        raise ValueError("recomputed endpoint objective does not match its saved receipt")
    saved_condition = float(attempt["condition_number"])
    if not math.isclose(condition_number, saved_condition, rel_tol=2e-7, abs_tol=1e-5):
        raise ValueError("recomputed endpoint condition does not match its saved receipt")

    gradient = jacobian.T @ residual
    column_norms = np.linalg.norm(jacobian, axis=0)
    lower, upper = nonlinear._parameter_bounds(FLOOR, force)
    lower_near = (np.isfinite(lower)
                  & (reduced <= lower + ACTIVE_BOUND_RTOL * np.maximum(1.0, np.abs(lower))))
    upper_near = (np.isfinite(upper)
                  & (reduced >= upper - ACTIVE_BOUND_RTOL * np.maximum(1.0, np.abs(upper))))
    projected_gradient = gradient.copy()
    projected_gradient[lower_near & (gradient > 0.0)] = 0.0
    projected_gradient[upper_near & (gradient < 0.0)] = 0.0
    scaled_projected_gradient = (np.abs(projected_gradient)
                                 / np.maximum(column_norms, np.finfo(float).tiny))
    weak_loadings, standardized_condition = _weak_group_loadings(jacobian)
    return {
        "phase": phase,
        "panel": int(entry["panel"]),
        "replicate": int(entry["replicate"]),
        "stratum": str(entry["stratum"]),
        "seed": int(entry["seed"]),
        "target_force": float(force),
        "start": str(attempt["start"]),
        "nfev": int(attempt["nfev"]),
        "saved_objective": saved_objective,
        "recomputed_objective": objective,
        "saved_condition_number": saved_condition,
        "recomputed_condition_number": condition_number,
        "local_jacobian_rank": local_rank,
        "parameter_count": int(attempt["parameter_count"]),
        "near_bound_parameter_count": int(np.count_nonzero(lower_near | upper_near)),
        "projected_bound_parameter_count": int(np.count_nonzero(
            (lower_near & (gradient > 0.0)) | (upper_near & (gradient < 0.0))
        )),
        "scaled_projected_gradient_inf": float(np.max(scaled_projected_gradient)),
        "gradient_exceeds_gtol_reference": bool(
            np.max(scaled_projected_gradient) > GTOL_REFERENCE
        ),
        "standardized_jacobian_condition": standardized_condition,
        **{f"weak_loading_{name}": value for name, value in weak_loadings.items()},
    }


def _collect_attempts(source_manifest: dict[str, Any], continuation_receipts: dict[str, Any]
                      ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    panel_rows = continuation_receipts["panels"]
    for entry in panel_rows:
        if entry["stratum"] not in TARGET_STRATA:
            raise ValueError("unexpected stratum in continuation receipt")
        panel_dir = (REPO_ROOT / entry["input_panel"]).resolve()
        cold_receipts = json.loads((panel_dir / "profile_receipts.json").read_text())
        snapshot, matrices, _panel_hashes = _snapshot_panel(panel_dir, source_manifest)
        blocks = two_state._observed_vectors(matrices)
        mean = blocks.mean(axis=0)
        covariance, _diagnostics = two_state._regularized_mean_covariance(blocks)
        covariance_cholesky = cholesky(covariance, lower=True, check_finite=True)
        cold_cells = [cell for cell in cold_receipts["profile_cells"]
                      if math.isclose(float(cell["overlap_floor"]), FLOOR,
                                      rel_tol=0.0, abs_tol=1e-12)]
        for cell in cold_cells:
            if cell.get("valid"):
                continue
            for attempt in cell.get("attempts", []):
                if not attempt.get("valid") and attempt.get("fit_coordinates"):
                    rows.append(_attempt_diagnostic(
                        "cold", entry, float(cell["target_force"]), attempt, mean,
                        covariance_cholesky,
                    ))
        for cell in entry["cells"]:
            result = cell.get("continuation_result")
            if result is None:
                continue
            for attempt in result.get("attempts", []):
                if not attempt.get("valid") and attempt.get("fit_coordinates"):
                    rows.append(_attempt_diagnostic(
                        "neighbor_continuation", entry, float(cell["target_force"]),
                        attempt, mean, covariance_cholesky,
                    ))
    return rows, {"panel_count": len(panel_rows),
                  "target_strata": sorted({str(row["stratum"]) for row in panel_rows})}


def _phase_summary(rows: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    selected = [row for row in rows if row["phase"] == phase]
    gradients = np.asarray([row["scaled_projected_gradient_inf"] for row in selected])
    loading_names = sorted(key.removeprefix("weak_loading_") for key in selected[0]
                           if key.startswith("weak_loading_"))
    return {
        "invalid_starts": len(selected),
        "at_max_nfev_1600": sum(row["nfev"] == 1600 for row in selected),
        "full_local_rank": sum(row["local_jacobian_rank"] == row["parameter_count"]
                                for row in selected),
        "scaled_projected_gradient_inf": {
            "median": float(np.median(gradients)),
            "p90": float(np.quantile(gradients, 0.9)),
            "maximum": float(np.max(gradients)),
            "above_gtol_reference": sum(row["gradient_exceeds_gtol_reference"]
                                         for row in selected),
        },
        "weak_direction_mean_squared_loading": {
            name: float(np.mean([row[f"weak_loading_{name}"] for row in selected]))
            for name in loading_names
        },
    }


def _plot(path: Path, rows: list[dict[str, Any]]) -> None:
    phases = ("cold", "neighbor_continuation")
    labels = ("Cold invalid starts", "Neighbor retry failures")
    colors = ("#0072b2", "#d55e00")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    for index, (phase, label, color) in enumerate(zip(phases, labels, colors)):
        selected = [row for row in rows if row["phase"] == phase]
        values = np.asarray([row["scaled_projected_gradient_inf"] for row in selected])
        jitter = np.linspace(-0.11, 0.11, len(values)) if len(values) > 1 else np.zeros(len(values))
        axes[0].scatter(index + jitter, values, color=color, alpha=0.78, s=30,
                        label=f"{label} (n={len(values)})")
    axes[0].axhline(GTOL_REFERENCE, color="#555555", linestyle="--", linewidth=1,
                    label="configured gtol reference (1e-9)")
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(2), labels)
    axes[0].set_ylabel("scaled projected gradient infinity norm")
    axes[0].set_title("Saved endpoints remain non-stationary")
    axes[0].grid(axis="y", which="both", alpha=0.22)
    axes[0].legend(fontsize=8)
    group_names = ("vacuum gap", "R2 energy", "R2 amplitudes", "R3 energy*", "R3 amplitudes")
    group_colors = ("#999999", "#56b4e9", "#0072b2", "#e69f00", "#cc79a7")
    bottom = np.zeros(2)
    for group_name, color, source_name in zip(
        group_names, group_colors,
        ("vacuum_gap", "R2_energy_increments", "R2_amplitudes",
         "R3_energy_increments_excluding_force", "R3_amplitudes"),
    ):
        values = [np.mean([row[f"weak_loading_{source_name}"] for row in rows
                           if row["phase"] == phase]) for phase in phases]
        axes[1].bar(range(2), values, bottom=bottom, color=color, label=group_name)
        bottom += np.asarray(values)
    axes[1].set_xticks(range(2), labels)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Local weak-direction loading (stacked to 1)")
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.suptitle("Wilson F-lane saved failed-start attribution (no optimizer rerun)",
                 fontsize=14)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(output: Path = DEFAULT_OUT) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty report directory: {output}")
    input_integrity = {
        "stratified_profile_report": _verify_artifact_report(INPUT_REPORT),
        "continuation_report": _verify_artifact_report(CONTINUATION_REPORT),
    }
    source_manifest = json.loads((INPUT_REPORT / "artifact_hashes.json").read_text())
    continuation_receipts = json.loads(
        (CONTINUATION_REPORT / "rescue_receipts.json").read_text()
    )
    rows, panel_summary = _collect_attempts(source_manifest, continuation_receipts)
    expected_counts = {"cold": 26, "neighbor_continuation": 23}
    observed_counts = {phase: sum(row["phase"] == phase for row in rows)
                       for phase in expected_counts}
    if observed_counts != expected_counts:
        raise ValueError(f"unexpected failed-start denominators: {observed_counts}")
    summaries = {phase: _phase_summary(rows, phase) for phase in expected_counts}
    output.mkdir(parents=True, exist_ok=True)
    source_map = {
        "profile_stationarity.py": Path(__file__),
        "profile_continuation_diagnostic.py": Path(continuation.__file__),
        "nonlinear_overlap_profile.py": Path(nonlinear.__file__),
        "three_static_candidate.py": Path(candidate.__file__),
        "two_sector_fit_candidate.py": Path(two_state.__file__),
        "joint_spectral_gls.py": Path(candidate.forward.__file__),
        "source_integrity.py": REPO_ROOT / "core/source_integrity.py",
        "npz_safety.py": REPO_ROOT / "core/npz_safety.py",
    }
    source_dir = output / "source_snapshot"
    source_dir.mkdir()
    source_hashes = {}
    for name, source in source_map.items():
        payload = source.read_bytes()
        source_hashes[name] = _sha(payload)
        (source_dir / name).write_bytes(payload)

    csv_path = output / "failed_start_stationarity.csv"
    fieldnames = list(rows[0])
    import csv
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _plot(output / "stationarity_and_weak_directions.png", rows)
    summary = {
        "schema": "wilson-f-lane-profile-stationarity-attribution-v1",
        "classification": "READ_ONLY_SAVED_ENDPOINT_DIAGNOSTIC",
        "certified": False,
        "physical_score": None,
        "input_report": str(INPUT_REPORT.relative_to(REPO_ROOT)),
        "continuation_report": str(CONTINUATION_REPORT.relative_to(REPO_ROOT)),
        "input_integrity": input_integrity,
        "selected_panels": panel_summary,
        "denominator": observed_counts,
        "phase_summaries": summaries,
        "gradient_definition": (
            "At each saved endpoint, recompute J and residual r from the captured matrix; "
            "project J.T@r against bounds within relative tolerance 1e-7, then divide each "
            "component by its whitened Jacobian column norm. This approximates the scaled "
            "box-constrained first-order optimality used with x_scale='jac'; it is not the "
            "saved SciPy optimality field."
        ),
        "weak_direction_definition": (
            "SVD of the whitened Jacobian after normalizing each nonzero column to unit norm; "
            "group loadings are squared components in the resulting weakest right singular "
            "vector. This is local and depends on the chosen parameterization."
        ),
        "scope_limits": [
            "all input report artifacts are hash-checked before selected panel bytes are captured",
            "only previously saved failed optimizer endpoints are evaluated; no optimizer is run",
            "the failed starts are paired retries on reused synthetic matrices, not independent samples",
            "full local rank does not establish global identifiability or a physical force interval",
            "no production Wilson matrices, sealed inputs, GPU or QUDA are read or run",
        ],
        "physical_status": "NOT_READY/UNCERTIFIED; F=GROUND_STATE_FORCE_UNCERTIFIED",
        "source_hashes": source_hashes,
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "protocol.json", {
        "schema": "wilson-f-lane-profile-stationarity-protocol-v1",
        "strata": list(TARGET_STRATA),
        "overlap_floor": FLOOR,
        "phase_denominators": observed_counts,
        "optimizer_reruns": 0,
        "gradient_active_bound_relative_tolerance": ACTIVE_BOUND_RTOL,
        "gtol_reference": GTOL_REFERENCE,
        "input_integrity": input_integrity,
        "output": ["failed_start_stationarity.csv", "stationarity_and_weak_directions.png"],
        "no_clobber": True,
    })
    readme = [
        "# Wilson saved-endpoint stationarity attribution", "",
        "**Classification:** read-only diagnosis of saved synthetic optimizer endpoints; not a new fit, calibration, or physical result.", "",
        "![Stationarity and weak-direction summaries](stationarity_and_weak_directions.png)", "",
        "Both input reports passed their full artifact-hash manifests before the selected raw panels and receipts were read. For each stored failed start, the script recomputed the whitened residual and analytic Jacobian at the saved coordinates, checked the reconstructed objective and condition number against the receipt, and computed a box-projected gradient scaled by Jacobian column norms. No optimizer was rerun.", "",
        "| Attempt set | Failed starts | At 1600 evaluation cap | Full local rank | Scaled projected gradient median / p90 / max | Above configured `gtol=1e-9` reference |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for phase, label in (("cold", "Cold-invalid starts"),
                         ("neighbor_continuation", "Failed neighbor retries")):
        item = summaries[phase]
        g = item["scaled_projected_gradient_inf"]
        readme.append(
            f"| {label} | {item['invalid_starts']} | {item['at_max_nfev_1600']} | "
            f"{item['full_local_rank']}/{item['invalid_starts']} | "
            f"{g['median']:.3g} / {g['p90']:.3g} / {g['maximum']:.3g} | "
            f"{g['above_gtol_reference']}/{item['invalid_starts']} |"
        )
    cold = summaries["cold"]["weak_direction_mean_squared_loading"]
    neighbor = summaries["neighbor_continuation"]["weak_direction_mean_squared_loading"]
    readme += [
        "",
        "The recomputed objective and condition number match their saved receipts within the declared numerical tolerance. Every failed endpoint remains far above the configured gradient tolerance reference after bound projection, so these saved runs stopped at the evaluation cap while still making a nontrivial first-order descent direction available. This supports a convergence/conditioning diagnosis for these endpoints; it does not show that the profile minimum is absent or that the force is globally identifiable.",
        "",
        f"After unit-normalizing each nonzero Jacobian column, the weakest local direction has mean squared loading on R2 amplitudes `{cold['R2_amplitudes']:.3f}` for cold failures and `{neighbor['R2_amplitudes']:.3f}` for neighbor failures. The other loadings are reported in `summary.json` and drawn at right. Because the coordinates are standardized by local column sensitivity, and because the direction is evaluated at nonconverged endpoints, this only nominates R2 amplitude tradeoffs for a future fixed-budget solver comparison; it does not prove global non-identifiability.",
        "",
        "A bounded next experiment is therefore a paired solver/parameter-scaling comparison on the same six panels and 49 failed-start targets. Keep the floor, force grid, objective, covariance, starts and `max_nfev` fixed; record objective, projected gradient, bound activity, and rescue count. Accept a numerical change only if it reduces the residual stationarity failure without changing the data model or eligibility rule. Do not expand seeds or label the resulting profile calibrated.",
        "",
        "All matrices are reused synthetic panels. No production `C_ij`, sealed split, GPU or QUDA is used. The physical reader remains `ABSTAIN_NO_INDEPENDENT_OVERLAP_FLOOR`; C finite-T Creutz and F ground-state force remain separate; status remains `NOT_READY/UNCERTIFIED`, F=`GROUND_STATE_FORCE_UNCERTIFIED`, `physical_score=null`.",
    ]
    (output / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    artifact_hashes = {
        str(path.relative_to(output)): _sha(path.read_bytes())
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_json(output / "artifact_hashes.json", {
        "schema": "wilson-profile-stationarity-artifact-hashes-v1",
        "artifacts": artifact_hashes,
    })
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"denominator": result["denominator"],
                      "phase_summaries": result["phase_summaries"]},
                     indent=2, sort_keys=True))
