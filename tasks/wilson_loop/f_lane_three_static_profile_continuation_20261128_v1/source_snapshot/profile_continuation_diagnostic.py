"""Paired nearest-neighbor-start rescue screen for failed Wilson force profiles.

This reuses only already-saved, hash-verified synthetic profiles. It does not
add independent samples or calibrate a physical force interval.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.npz_safety import validate_npz_payload
from core.source_integrity import capture_source_snapshot
from tasks.wilson_loop import f_lane_three_static_nonlinear_overlap_profile as nonlinear
from tasks.wilson_loop import f_lane_three_static_two_vacuum_candidate as candidate


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
INPUT_REPORT = TASK_DIR / "f_lane_three_static_stratified_profile_20261128_v1"
DEFAULT_OUT = TASK_DIR / "f_lane_three_static_profile_continuation_20261128_v1"
TARGET_STRATA = ("near_degenerate", "omitted_vacuum_truth")
ORACLE_FLOOR = 0.9
LR_REFERENCE = 3.84


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _closest_valid_start(cells: list[dict[str, Any]], target_force: float
                         ) -> dict[str, Any] | None:
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
    _distance, force, objective, coordinates = min(options, key=lambda row: row[:3])
    return {"source_force": force, "source_objective": objective,
            "fit_coordinates": list(coordinates)}


def _capture_panel(panel_dir: Path, artifact_manifest: dict[str, Any]):
    relative_dir = panel_dir.relative_to(INPUT_REPORT)
    names = ("raw_matrices.npz", "manifest.json", "lane_contract.json",
             "fit.json", "profile_receipts.json")
    payloads = {name: (panel_dir / name).read_bytes() for name in names}
    source_artifacts = artifact_manifest["artifacts"]
    source_hashes = {}
    for name, payload in payloads.items():
        relative = str(relative_dir / name)
        digest = _sha(payload)
        if source_artifacts.get(relative) != digest:
            raise ValueError(f"stratified report artifact hash mismatch: {relative}")
        source_hashes[name] = digest
    snapshot = capture_source_snapshot(
        panel_dir,
        {
            "raw_matrices": source_hashes["raw_matrices.npz"],
            "manifest": source_hashes["manifest.json"],
            "lane_contract": source_hashes["lane_contract.json"],
            "fit": source_hashes["fit.json"],
            "profile_receipts": source_hashes["profile_receipts.json"],
        },
        {
            "raw_matrices": "raw_matrices.npz", "manifest": "manifest.json",
            "lane_contract": "lane_contract.json", "fit": "fit.json",
            "profile_receipts": "profile_receipts.json",
        },
    )
    raw = snapshot.read_bytes("raw_matrices")
    validate_npz_payload(raw)
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        matrices = np.asarray(archive["correlator_matrices"], dtype=complex)
    return snapshot, matrices, source_hashes


def _interval(cells: list[dict[str, Any]], panel_summary: dict[str, Any]) -> dict[str, Any]:
    planned = len(cells)
    valid = [cell for cell in cells if cell.get("valid")]
    if len(valid) != planned:
        return {"status": "ABSTAIN_INCOMPLETE_GRID", "planned_cells": planned,
                "valid_cells": len(valid), "interval": None,
                "contains_truth": None, "grid_clipped": None}
    minimum = min(float(cell["objective"]) for cell in valid)
    accepted = [float(cell["target_force"]) for cell in valid
                if float(cell["objective"]) - minimum <= LR_REFERENCE]
    interval = [min(accepted), max(accepted)] if accepted else None
    grid = [float(item) for item in panel_summary["grid"]]
    clipped = bool(interval and (interval[0] == min(grid) or interval[1] == max(grid)))
    truth = float(panel_summary["force_truth"])
    return {
        "status": "CONDITIONAL_UNCALIBRATED_PROFILE_SET" if interval
        else "ABSTAIN_EMPTY_PROFILE_SET",
        "planned_cells": planned, "valid_cells": len(valid),
        "minimum_objective": minimum, "interval": interval,
        "contains_truth": bool(interval and interval[0] <= truth <= interval[1]),
        "grid_clipped": clipped,
    }


def _wilson_interval(successes: int, trials: int) -> list[float] | None:
    if trials == 0:
        return None
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / trials
                         + z * z / (4.0 * trials * trials)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _plot(path: Path, receipts: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, len(TARGET_STRATA), figsize=(12, 4.5), sharey=True,
                             layout="constrained")
    colors = {"cold_valid": "#999999", "rescued": "#0072b2", "still_abstains": "#d55e00"}
    for ax, stratum in zip(axes, TARGET_STRATA):
        group = [row for row in receipts if row["stratum"] == stratum]
        for row_index, panel in enumerate(group):
            for cell in panel["cells"]:
                if cell["cold_valid"]:
                    state = "cold_valid"
                elif cell.get("continuation_result", {}).get("valid"):
                    state = "rescued"
                else:
                    state = "still_abstains"
                ax.scatter(float(cell["target_force"]), row_index,
                           marker="s", s=60, color=colors[state])
        ax.set_title(f"{stratum}: oracle floor {ORACLE_FLOOR}")
        ax.set_xlabel("target force (lattice units)")
        ax.grid(axis="x", alpha=0.25)
        ax.set_yticks(range(len(group)), [str(row["replicate"]) for row in group])
    axes[0].set_ylabel("panel replicate")
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=color,
                          label=label.replace("_", " "))
               for label, color in colors.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    fig.suptitle("Paired profile-cell rescue using a neighboring converged force fit")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(output: Path = DEFAULT_OUT) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty report directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    artifact_manifest = json.loads((INPUT_REPORT / "artifact_hashes.json").read_text())
    input_summary = json.loads((INPUT_REPORT / "summary.json").read_text())
    all_panel_roots = sorted((INPUT_REPORT / "panels").iterdir())
    panel_receipts = []
    planned_rescues = []
    for panel_dir in all_panel_roots:
        profile_path = panel_dir / "profile_receipts.json"
        if not profile_path.exists():
            continue
        cold_payload = json.loads(profile_path.read_text())
        panel_summary = cold_payload["panel_summary"]
        stratum = panel_summary.get("stratum")
        if stratum not in TARGET_STRATA:
            continue
        floor_cells = [cell for cell in cold_payload["profile_cells"]
                       if math.isclose(float(cell["overlap_floor"]), ORACLE_FLOOR,
                                       rel_tol=0.0, abs_tol=1e-12)]
        if not floor_cells:
            raise ValueError(f"target panel has no oracle-floor cells: {panel_dir.name}")
        snapshot, matrices, input_hashes = _capture_panel(panel_dir, artifact_manifest)
        fit = json.loads(snapshot.read_bytes("fit"))
        fit_coordinates = np.asarray(fit["fit"]["fit_coordinates"], dtype=float)
        cold_valid = [cell for cell in floor_cells if cell.get("valid")]
        panel_cells = []
        for cold_cell in sorted(floor_cells, key=lambda row: float(row["target_force"])):
            target_force = float(cold_cell["target_force"])
            result = None
            continuation = None
            if not cold_cell.get("valid"):
                continuation = _closest_valid_start(cold_valid, target_force)
                if continuation is None:
                    result = {"status": "ABSTAIN_NO_NEIGHBOR_VALID_START", "valid": False,
                              "target_force": target_force, "overlap_floor": ORACLE_FLOOR,
                              "planned_starts": 0, "valid_starts": 0,
                              "failed_or_abstained_starts": 0, "attempts": []}
                else:
                    result = nonlinear._profile_at_force(
                        snapshot, fit_coordinates, ORACLE_FLOOR, target_force,
                        continuation_parameters=np.asarray(continuation["fit_coordinates"],
                                                           dtype=float),
                    )
            panel_cells.append({
                "target_force": target_force,
                "cold_valid": bool(cold_cell.get("valid")),
                "cold_status": cold_cell.get("status"),
                "cold_nfev_total": int(cold_cell.get("nfev_total", 0)),
                "cold_attempts": cold_cell.get("attempts", []),
                "neighbor_seed": None if continuation is None else {
                    "source_force": continuation["source_force"],
                    "source_objective": continuation["source_objective"],
                    "fit_coordinates_sha256": _sha(np.asarray(
                        continuation["fit_coordinates"], dtype="<f8").tobytes()),
                },
                "continuation_result": result,
            })
            if not cold_cell.get("valid"):
                planned_rescues.append(panel_cells[-1])
        combined_cells = []
        for cold_cell, row in zip(sorted(floor_cells, key=lambda item: float(item["target_force"])),
                                  panel_cells):
            combined_cells.append(row["continuation_result"] if not cold_cell.get("valid")
                                  and row["continuation_result"] is not None
                                  and row["continuation_result"].get("valid") else cold_cell)
        old_floor_summary = next(item for item in panel_summary["floor_results"]
                                 if math.isclose(float(item["floor"]), ORACLE_FLOOR,
                                                 rel_tol=0.0, abs_tol=1e-12))
        new_floor_summary = _interval(combined_cells, panel_summary)
        panel_receipts.append({
            "panel": int(panel_summary["panel"]), "replicate": int(panel_summary["replicate"]),
            "stratum": stratum, "seed": int(panel_summary["seed"]),
            "oracle_truth_floor": float(panel_summary["oracle_truth_floor"]),
            "force_truth": float(panel_summary["force_truth"]),
            "input_panel": str(panel_dir.relative_to(REPO_ROOT)),
            "input_artifact_hashes": input_hashes,
            "cold_profile_summary": old_floor_summary,
            "continuation_profile_summary": new_floor_summary,
            "cells": panel_cells,
            "planned_rescue_cells": sum(not row["cold_valid"] for row in panel_cells),
            "valid_continuation_cells": sum(bool(row["continuation_result"]
                                                 and row["continuation_result"].get("valid"))
                                             for row in panel_cells),
            "valid_start_fits": sum(int(row["continuation_result"].get("valid_starts", 0))
                                     for row in panel_cells if row["continuation_result"]),
            "planned_start_fits": sum(int(row["continuation_result"].get("planned_starts", 0))
                                       for row in panel_cells if row["continuation_result"]),
        })

    if len(panel_receipts) != len(TARGET_STRATA) * 3:
        raise ValueError(f"expected three saved panels per target stratum; got {len(panel_receipts)}")
    coverage = []
    for stratum in TARGET_STRATA:
        group = [row for row in panel_receipts if row["stratum"] == stratum]
        for phase, key in (("cold", "cold_profile_summary"),
                           ("neighbor_continuation", "continuation_profile_summary")):
            intervals = [row[key] for row in group]
            complete = [item for item in intervals if item.get("interval") is not None]
            unclipped = [item for item in complete if not item["grid_clipped"]]
            successes = sum(bool(item["contains_truth"]) for item in unclipped)
            coverage.append({
                "stratum": stratum, "phase": phase,
                "planned_panels": len(group), "complete_profiles": len(complete),
                "unclipped_profiles": len(unclipped), "truth_contained": successes,
                "wilson_95_interval": _wilson_interval(successes, len(unclipped)),
                "failed_or_abstained_profiles": len(group) - len(complete),
            })

    rescue_rows = []
    for panel in panel_receipts:
        for cell in panel["cells"]:
            result = cell["continuation_result"]
            if result is None:
                continue
            rescue_rows.append({
                "panel": panel["panel"], "stratum": panel["stratum"],
                "replicate": panel["replicate"], "seed": panel["seed"],
                "target_force": cell["target_force"], "floor": ORACLE_FLOOR,
                "cold_valid": cell["cold_valid"],
                "neighbor_force": None if cell["neighbor_seed"] is None
                else cell["neighbor_seed"]["source_force"],
                "continuation_valid": bool(result.get("valid")),
                "valid_starts": result.get("valid_starts", 0),
                "planned_starts": result.get("planned_starts", 0),
                "nfev_total": result.get("nfev_total", 0),
                "status": result.get("status"),
            })

    summary = {
        "schema": "wilson-f-lane-profile-continuation-screen-v1",
        "classification": "PAIRED_SYNTHETIC_FAILED_CELL_START_RESCUE_SCREEN",
        "certified": False, "physical_score": None,
        "input_report": str(INPUT_REPORT.relative_to(REPO_ROOT)),
        "input_report_artifact_count": len(artifact_manifest["artifacts"]),
        "reused_strata": list(TARGET_STRATA), "oracle_floor": ORACLE_FLOOR,
        "paired_coverage": coverage,
        "profile_cell_denominator": {
            "planned_failed_cold_cells": sum(int(panel["planned_rescue_cells"])
                                               for panel in panel_receipts),
            "valid_neighbor_continuation_cells": sum(int(panel["valid_continuation_cells"])
                                                       for panel in panel_receipts),
            "failed_or_abstained_continuation_cells": sum(
                int(panel["planned_rescue_cells"] - panel["valid_continuation_cells"])
                for panel in panel_receipts),
        },
        "start_fit_denominator": {
            "planned": sum(int(panel["planned_start_fits"]) for panel in panel_receipts),
            "valid": sum(int(panel["valid_start_fits"]) for panel in panel_receipts),
            "failed_or_abstained": sum(
                int(panel["planned_start_fits"] - panel["valid_start_fits"])
                for panel in panel_receipts),
        },
        "panels": panel_receipts,
        "scope_limits": [
            "failed cells only are rerun; the neighbor start is not an independent start or new sample",
            "one neighboring cold-valid profile supplies the continuation start, paired with the captured point-fit start",
            "profile force grid and nominal delta-objective threshold 3.84 remain uncalibrated",
            "all inputs are synthetic and reused; this cannot establish production coverage",
            "no production Wilson input, sealed split, GPU or QUDA is read or run",
        ],
        "physical_gate": "ABSTAIN_NO_INDEPENDENT_OVERLAP_FLOOR",
        "physical_status": "NOT_READY/UNCERTIFIED; F=GROUND_STATE_FORCE_UNCERTIFIED",
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "protocol.json", {
        "schema": "wilson-f-lane-profile-continuation-protocol-v1",
        "input_report": summary["input_report"],
        "target_strata": list(TARGET_STRATA), "floor": ORACLE_FLOOR,
        "selection": "only cold-invalid cells under the per-panel toy oracle floor in near_degenerate and omitted_vacuum_truth",
        "neighbor_rule": "choose nearest force target with a valid cold-start fit; tie-break by target force then objective",
        "fit_starts": "captured point-fit start plus neighboring cold-valid cell parameters; each start is force-eliminated at target force",
        "max_nfev_per_start": 1600,
        "profile_threshold": LR_REFERENCE,
        "physical_gate": "ABSTAIN_NO_INDEPENDENT_OVERLAP_FLOOR",
    })
    _write_json(output / "rescue_receipts.json", {"panels": panel_receipts})
    with (output / "rescue_cells.csv").open("w", encoding="utf-8", newline="") as stream:
        if rescue_rows:
            writer = csv.DictWriter(stream, fieldnames=list(rescue_rows[0]),
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(rescue_rows)
        else:
            stream.write("\n")
    _plot(output / "continuation_cell_status.png", panel_receipts)

    source_map = {
        "profile_continuation_diagnostic.py": Path(__file__),
        "nonlinear_overlap_profile.py": Path(nonlinear.__file__),
        "stratified_profile_screen.py": TASK_DIR / "f_lane_three_static_stratified_profile_screen.py",
        "three_static_candidate.py": Path(candidate.__file__),
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
    summary["source_hashes"] = source_hashes
    _write_json(output / "summary.json", summary)

    containment = []
    for row in coverage:
        rate = (f"{row['truth_contained']}/{row['unclipped_profiles']}"
                if row["unclipped_profiles"] else "NA")
        interval = row["wilson_95_interval"]
        interval_text = ("NA" if interval is None
                         else f"[{interval[0]:.3f}, {interval[1]:.3f}]")
        containment.append(
            f"| {row['stratum']} | {row['phase']} | {row['complete_profiles']}/3 | "
            f"{rate} | {row['failed_or_abstained_profiles']} | {interval_text} |"
        )
    readme = "\n".join([
        "# Wilson profile continuation start-rescue screen", "",
        "**Classification:** paired numerical-rescue experiment on existing synthetic panels; not calibration.", "",
        "![Cold-valid, neighbor-rescued, and still-abstaining cells](continuation_cell_status.png)", "",
        f"The screen reruns {summary['profile_cell_denominator']['planned_failed_cold_cells']} failed cells from three near-degenerate and three omitted-vacuum panels. Each rerun keeps the original captured point fit as one start and adds a second start copied from the nearest cold-valid force cell at the same floor.", "",
        f"Neighbor continuation recovered {summary['profile_cell_denominator']['valid_neighbor_continuation_cells']} cells; continuation starts valid **{summary['start_fit_denominator']['valid']}/{summary['start_fit_denominator']['planned']}**. These are paired retries on the same matrices, not independent fits or new coverage samples.", "",
        "| Stratum | Phase | Complete profiles | Truth containment | Incomplete profiles | Wilson 95% interval |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        *containment, "",
        "Containment uses complete, unclipped intervals only; `NA` means no eligible interval. The panel count is three per stratum, so binomial intervals are descriptive and very wide. The force grid and nominal `Δobjective<=3.84` rule remain uncalibrated.", "",
        "All raw matrices and cold receipts were reused after checking their source report's artifact hashes and capturing one immutable byte snapshot per panel. No new RNG panels, production Wilson matrices, sealed inputs, GPU or QUDA were used. The physical reader still returns `ABSTAIN_NO_INDEPENDENT_OVERLAP_FLOOR`; both lanes remain `NOT_READY/UNCERTIFIED`, F=`GROUND_STATE_FORCE_UNCERTIFIED`, and `physical_score=null`.",
    ]) + "\n"
    (output / "README.md").write_text(readme, encoding="utf-8")
    artifacts = {
        str(path.relative_to(output)): _sha(path.read_bytes())
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_json(output / "artifact_hashes.json", {
        "schema": "wilson-profile-continuation-artifact-hashes-v1",
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
        "profile_cell_denominator": result["profile_cell_denominator"],
        "start_fit_denominator": result["start_fit_denominator"],
        "paired_coverage": result["paired_coverage"],
    }, indent=2, sort_keys=True))
