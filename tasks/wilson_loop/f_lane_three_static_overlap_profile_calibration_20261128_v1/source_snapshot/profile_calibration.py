"""Small conditional coverage screen for nonlinear weak-overlap profiles."""
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

from tasks.wilson_loop import f_lane_absolute_force_bootstrap_diagnostic as bootstrap
from tasks.wilson_loop import f_lane_three_static_identifiability_screen as generator
from tasks.wilson_loop import f_lane_three_static_nonlinear_overlap_profile as nonlinear
from tasks.wilson_loop import f_lane_three_static_two_vacuum_candidate as candidate


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
INPUT_REPORT = TASK_DIR / "f_lane_three_static_identifiability_20261128_v1"
DEFAULT_OUT = TASK_DIR / "f_lane_three_static_overlap_profile_calibration_20261128_v1"
PANEL_COUNT = 6
SEED_BASE = 2026114300
FLOORS = (0.0, 0.045)
BASE_FORCE_GRID = (0.14, 0.15, 0.16, 0.17, 0.18)
LR_REFERENCE = 3.84


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float] | None:
    if trials == 0:
        return None
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half = (z * math.sqrt(proportion * (1.0 - proportion) / trials
                          + z * z / (4.0 * trials * trials)) / denominator)
    return [max(0.0, center - half), min(1.0, center + half)]


def _force_grid(center: float) -> tuple[float, ...]:
    return tuple(sorted(set(BASE_FORCE_GRID + (float(center),))))


def _panel_profile(snapshot: Any, fit_receipt: dict[str, Any], truth: float,
                   panel_id: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fit = fit_receipt.get("fit")
    if (fit_receipt.get("denominator", {}).get("valid") != 1
            or not isinstance(fit, dict) or not isinstance(fit.get("fit_coordinates"), list)):
        row = {"panel": panel_id, "panel_valid": False,
               "failure_reason": "missing_candidate_fit_parameters",
               "force_truth": truth, "force_point": None, "grid": [],
               "planned_profile_cells": 0, "valid_profile_cells": 0,
               "failed_or_abstained_profile_cells": len(FLOORS) * len(BASE_FORCE_GRID),
               "planned_start_fits": 0, "valid_start_fits": 0,
               "failed_or_abstained_start_fits": 0,
               "floor_results": [{
                   "floor": floor, "status": "ABSTAIN_CANDIDATE_FIT",
                   "planned_cells": len(BASE_FORCE_GRID), "valid_cells": 0,
                   "failed_or_abstained_cells": len(BASE_FORCE_GRID),
                   "interval": None, "contains_truth": None, "grid_clipped": None,
               } for floor in FLOORS]}
        return [], row
    center = float(fit["absolute_stout_ground_force"])
    grid = _force_grid(center)
    cells = [nonlinear._profile_at_force(snapshot, np.asarray(fit["fit_coordinates"], dtype=float),
                                         floor, force)
             for floor in FLOORS for force in grid]
    floor_results = []
    for floor in FLOORS:
        group = [cell for cell in cells if cell["overlap_floor"] == floor]
        valid = [cell for cell in group if cell["valid"]]
        if len(valid) != len(grid):
            floor_results.append({
                "floor": floor, "status": "ABSTAIN_INCOMPLETE_GRID",
                "planned_cells": len(grid), "valid_cells": len(valid),
                "failed_or_abstained_cells": len(group) - len(valid),
                "interval": None, "contains_truth": None, "grid_clipped": None,
            })
            continue
        minimum = min(float(cell["objective"]) for cell in valid)
        accepted = [float(cell["target_force"]) for cell in valid
                    if float(cell["objective"]) - minimum <= LR_REFERENCE]
        interval = [min(accepted), max(accepted)] if accepted else None
        clipped = bool(interval and (interval[0] == grid[0] or interval[1] == grid[-1]))
        floor_results.append({
            "floor": floor,
            "status": "CONDITIONAL_UNCALIBRATED_PROFILE_SET" if interval
            else "ABSTAIN_EMPTY_PROFILE_SET",
            "planned_cells": len(grid), "valid_cells": len(valid),
            "failed_or_abstained_cells": 0,
            "minimum_objective": minimum,
            "interval": interval,
            "contains_truth": bool(interval and interval[0] <= truth <= interval[1]),
            "grid_clipped": clipped,
        })
    panel_row = {
        "panel": panel_id, "seed": SEED_BASE + panel_id,
        "panel_valid": fit_receipt.get("denominator", {}).get("valid") == 1,
        "force_truth": truth, "force_point": center,
        "grid": list(grid), "floor_results": floor_results,
        "planned_profile_cells": len(cells),
        "valid_profile_cells": sum(cell["valid"] for cell in cells),
        "failed_or_abstained_profile_cells": sum(not cell["valid"] for cell in cells),
        "planned_start_fits": sum(cell["planned_starts"] for cell in cells),
        "valid_start_fits": sum(cell["valid_starts"] for cell in cells),
        "failed_or_abstained_start_fits": sum(
            cell["failed_or_abstained_starts"] for cell in cells
        ),
    }
    return cells, {"panel_summary": panel_row, "profile_cells": cells}


def _coverage_summary(panel_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for floor in FLOORS:
        statuses = [next(item for item in panel["floor_results"] if item["floor"] == floor)
                    for panel in panel_summaries]
        complete = [item for item in statuses if item["interval"] is not None]
        eligible = [item for item in complete if not item["grid_clipped"]]
        covered = sum(bool(item["contains_truth"]) for item in eligible)
        clipped = sum(bool(item["grid_clipped"]) for item in complete)
        summaries.append({
            "floor": floor,
            "planned_panels": len(statuses),
            "profile_complete_panels": len(complete),
            "unclipped_profiles_used_for_coverage": len(eligible),
            "truth_contained_panels": covered,
            "conditional_containment_fraction": covered / len(eligible) if eligible else None,
            "wilson_95_interval": _wilson_interval(covered, len(eligible)),
            "clipped_intervals": clipped,
            "failed_or_abstained_panels": len(statuses) - len(complete),
        })
    return summaries


def _plot(path: Path, panel_summaries: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, len(FLOORS), figsize=(12, 4.6), sharey=True,
                             layout="constrained")
    for ax, floor in zip(axes, FLOORS):
        for panel_index, panel in enumerate(panel_summaries):
            result = next(item for item in panel["floor_results"] if item["floor"] == floor)
            interval = result["interval"]
            if interval is None:
                ax.scatter([0.15], [panel_index], marker="x", color="#d55e00")
                continue
            color = "#0072b2" if result["contains_truth"] else "#d55e00"
            ax.plot(interval, [panel_index, panel_index], color=color, linewidth=2)
            ax.scatter([panel["force_point"]], [panel_index], color=color, s=22, zorder=3)
            if result["grid_clipped"]:
                ax.scatter([interval[1]], [panel_index], marker=">", color=color, zorder=4)
        ax.axvline(0.15, color="black", linestyle="--", label="exact toy truth")
        ax.set_title(f"floor={floor:g}" if floor else "no floor")
        ax.set_xlabel("conditional force set (lattice units)")
        ax.set_yticks(range(len(panel_summaries)),
                      [str(panel["panel"]) for panel in panel_summaries])
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_ylabel("synthetic panel")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Six-panel weak-overlap profile screen (red misses toy truth; > marks clipped)")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(output: Path = DEFAULT_OUT) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty report directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    panel_summaries, all_cells, panel_artifacts = [], [], []
    contract_path = INPUT_REPORT / "lane_contract.json"
    contract_bytes = contract_path.read_bytes()
    for panel_index in range(PANEL_COUNT):
        seed = SEED_BASE + panel_index
        matrices, spec = generator._generate_panel(seed, "weak_ground_overlap")
        panel_dir = output / "panels" / f"{panel_index:02d}_weak_overlap"
        snapshot, snapshot_hashes = bootstrap._snapshot_from_matrices(
            panel_dir, matrices, contract_bytes,
            f"F-synthetic-open-weak-overlap-calibration-{panel_index:02d}",
        )
        raw_hash = snapshot_hashes["raw_matrices"]
        try:
            fit_receipt = candidate.fit_three_static_snapshot(snapshot)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            fit_receipt = {
                "certified": False, "physical_score": None,
                "denominator": {"planned": 1, "valid": 0, "failed": 1, "abstained": 0},
                "error_type": type(exc).__name__, "error_message": str(exc),
            }
        _write_json(panel_dir / "fit.json", fit_receipt)
        cells, payload = _panel_profile(snapshot, fit_receipt, float(spec["force_truth"]), panel_index)
        _write_json(panel_dir / "profile_receipts.json", payload)
        panel_summary = payload["panel_summary"]
        panel_summary.update({
            "raw_matrices_sha256": raw_hash,
            "truth_overlap_floor_min_R": min(
                abs(float(generator._case_parameters("weak_ground_overlap")["overlaps"][r, 0, 0, 0]))
                for r in range(2)
            ),
            "truth_definition": "one positive 2-vacuum × 3-static weak-overlap DGP; 96 ordered AR(1) blocks; rho=0.5; relative noise=0.003",
        })
        panel_summaries.append(panel_summary)
        all_cells.extend({key: value for key, value in {"panel": panel_index, **cell}.items()
                          if key != "attempts"} for cell in cells)
        panel_artifacts.append({"panel": panel_index, "snapshot_hashes": snapshot_hashes,
                                "source_spec": spec})

    coverage = _coverage_summary(panel_summaries)
    summary = {
        "schema": "wilson-weak-overlap-profile-calibration-summary-v1",
        "classification": "SMALL_CONDITIONAL_SYNTHETIC_COVERAGE_SCREEN",
        "certified": False, "physical_score": None,
        "panel_count": PANEL_COUNT, "seed_base": SEED_BASE,
        "data_generation": {
            "stratum": "weak_ground_overlap", "blocks": generator.N_BLOCKS,
            "rho": 0.5, "relative_noise": generator.RELATIVE_NOISE,
            "truth_force": 0.15,
        },
        "profile": {"floors": list(FLOORS), "base_force_grid": list(BASE_FORCE_GRID),
                    "profile_threshold": LR_REFERENCE, "starts_per_cell": 2},
        "panel_summaries": panel_summaries,
        "coverage_by_floor": coverage,
        "denominator": {
            "planned_panels": PANEL_COUNT,
            "valid_candidate_fit_panels": sum(panel["panel_valid"] for panel in panel_summaries),
            "planned_profile_cells": sum(panel["planned_profile_cells"] for panel in panel_summaries),
            "valid_profile_cells": sum(panel["valid_profile_cells"] for panel in panel_summaries),
            "failed_or_abstained_profile_cells": sum(
                panel["failed_or_abstained_profile_cells"] for panel in panel_summaries
            ),
            "planned_start_fits": sum(panel["planned_start_fits"] for panel in panel_summaries),
            "valid_start_fits": sum(panel["valid_start_fits"] for panel in panel_summaries),
            "failed_or_abstained_start_fits": sum(
                panel["failed_or_abstained_start_fits"] for panel in panel_summaries
            ),
        },
        "physical_gate_without_independent_floor_receipt":
            "ABSTAIN_NO_INDEPENDENT_OVERLAP_FLOOR",
        "scope_limits": [
            "six panels from one synthetic DGP are a small conditional screen, not publication-grade calibration",
            "the 0.045 floor equals the generator's toy minimum overlap and is an oracle simulation condition, not independent production evidence",
            "nominal delta-objective threshold and coarse grid do not prove 95 percent coverage",
            "two-start success and full local rank do not prove global identifiability",
            "production force remains uncertified and no physical score is produced",
        ],
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "panel_artifacts.json", {"panels": panel_artifacts})
    _write_csv(output / "panel_summary.csv", panel_summaries)
    _write_csv(output / "profile_grid.csv", all_cells)
    _plot(output / "profile_intervals.png", panel_summaries)
    source_map = {
        "profile_calibration.py": Path(__file__),
        "nonlinear_overlap_profile.py": Path(nonlinear.__file__),
        "identifiability_generator.py": Path(generator.__file__),
        "three_static_candidate.py": Path(candidate.__file__),
        "bootstrap_snapshot_helper.py": Path(bootstrap.__file__),
        "source_integrity.py": REPO_ROOT / "core/source_integrity.py",
        "npz_safety.py": REPO_ROOT / "core/npz_safety.py",
    }
    source_dir = output / "source_snapshot"
    source_dir.mkdir()
    hashes = {}
    for name, source in source_map.items():
        payload = source.read_bytes()
        hashes[name] = _sha(payload)
        (source_dir / name).write_bytes(payload)
    summary["source_hashes"] = hashes
    _write_json(output / "summary.json", summary)
    _write_json(output / "protocol.json", {
        "schema": "wilson-weak-overlap-profile-calibration-protocol-v1",
        "panel_count": PANEL_COUNT, "seed_rule": "2026114300 + panel_index",
        "DGP": "weak_ground_overlap; 96 blocks; AR(1) rho=0.5; relative noise=0.003; exact force 0.15",
        "floors": list(FLOORS), "base_force_grid": list(BASE_FORCE_GRID),
        "adaptive_grid_point": "each panel's nonlinear candidate force estimate, added to the fixed shared grid",
        "profile_method": "exact-force 2-vacuum × 3-static nonlinear nuisance refit; analytic residual Jacobian; two starts",
        "oracle_floor_note": "0.045 is the known minimum overlap in the synthetic generator only; never an independent production receipt",
        "profile_rule": "objective minus per-floor grid minimum <= 3.84; clipped endpoints flagged",
        "physical_status": "NOT_READY/UNCERTIFIED; F=GROUND_STATE_FORCE_UNCERTIFIED",
    })
    _write_json(output / "lane_contract.json", json.loads(contract_bytes))
    readme = "\n".join([
        "# Weak-overlap nonlinear profile calibration screen", "",
        "**Classification:** six-panel, single-DGP synthetic coverage screen. This is not publication-grade calibration or a physical force result.", "",
        "![Per-panel force profile intervals](profile_intervals.png)", "",
        "## Results", "",
        f"Candidate-fit panels valid: `{summary['denominator']['valid_candidate_fit_panels']}/{PANEL_COUNT}`. Nonlinear profile cells valid: `{summary['denominator']['valid_profile_cells']}/{summary['denominator']['planned_profile_cells']}`; failed/abstained `{summary['denominator']['failed_or_abstained_profile_cells']}`. Start fits valid: `{summary['denominator']['valid_start_fits']}/{summary['denominator']['planned_start_fits']}`; failed/abstained `{summary['denominator']['failed_or_abstained_start_fits']}`.", "",
        "Each interval is the hull of grid values with `Δobjective<=3.84`; profiles add the panel's candidate point estimate to a fixed shared grid. Red intervals miss exact toy force `0.15`; blue intervals contain it; right-pointing markers mean the grid endpoint clipped the conditional set. Per-floor coverage numerators, eligible denominators and Wilson 95% binomial intervals are in `summary.json`.", "",
        "The `0.045` floor equals the known minimum overlap generated in this toy. It is an oracle condition for simulation only, not a measured or independently reviewed production lower bound. Without an independent floor receipt, the physical gate remains `ABSTAIN_NO_INDEPENDENT_OVERLAP_FLOOR`.", "",
        "Six replicates from one positive-spectrum weak-overlap DGP have wide Monte Carlo uncertainty. The nominal threshold is not calibrated; two starts do not establish global identifiability. No production Wilson matrices, sealed split, GPU or QUDA was accessed. `physical_score=null`; C/F remain `NOT_READY/UNCERTIFIED` and F remains `GROUND_STATE_FORCE_UNCERTIFIED`.",
    ]) + "\n"
    (output / "README.md").write_text(readme, encoding="utf-8")
    artifacts = {
        str(path.relative_to(output)): _sha(path.read_bytes())
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_json(output / "artifact_hashes.json", {
        "schema": "wilson-weak-overlap-profile-calibration-artifact-hashes-v1",
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
        "denominator": result["denominator"],
        "coverage_by_floor": result["coverage_by_floor"],
        "physical_gate_without_floor_receipt": result["physical_gate_without_independent_floor_receipt"],
    }, indent=2, sort_keys=True))
