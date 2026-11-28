"""Multi-stratum synthetic stress screen for nonlinear weak-overlap profiles."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tasks.wilson_loop import f_lane_absolute_force_bootstrap_diagnostic as bootstrap
from tasks.wilson_loop import f_lane_three_static_identifiability_screen as generator
from tasks.wilson_loop import f_lane_three_static_nonlinear_overlap_profile as nonlinear
from tasks.wilson_loop import f_lane_three_static_overlap_profile_calibration as calibration
from tasks.wilson_loop import f_lane_three_static_two_vacuum_candidate as candidate
from tasks.wilson_loop import f_lane_two_sector_fit_candidate as two_state


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
WEAK_REPORT = TASK_DIR / "f_lane_three_static_overlap_profile_calibration_20261128_v1"
DEFAULT_OUT = TASK_DIR / "f_lane_three_static_stratified_profile_20261128_v1"
OTHER_STRATA = (
    "baseline", "near_degenerate", "omitted_vacuum_truth", "dependent_unequal_variance",
)
REPLICATES_PER_STRATUM = 3
SEED_BASE = 2026114400


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


def _checked_weak_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((WEAK_REPORT / "artifact_hashes.json").read_text())
    summary = json.loads((WEAK_REPORT / "summary.json").read_text())
    panel_summaries = []
    panel_receipts = []
    for panel_index in range(6):
        panel_root = WEAK_REPORT / "panels" / f"{panel_index:02d}_weak_overlap"
        payloads = {}
        for name in ("raw_matrices.npz", "manifest.json", "fit.json", "profile_receipts.json"):
            relative = f"panels/{panel_index:02d}_weak_overlap/{name}"
            data = (panel_root / name).read_bytes()
            if _sha(data) != manifest["artifacts"].get(relative):
                raise ValueError(f"reused weak-overlap panel artifact hash mismatch: {relative}")
            payloads[name] = _sha(data)
        panel_summaries.append(summary["panel_summaries"][panel_index])
        panel_receipts.append({
            "panel": panel_index, "source_report": str(WEAK_REPORT.relative_to(REPO_ROOT)),
            "artifact_hashes": payloads,
            "profile_receipts": json.loads((panel_root / "profile_receipts.json").read_text()),
        })
    return panel_summaries, panel_receipts, {
        "source_report": str(WEAK_REPORT.relative_to(REPO_ROOT)),
        "artifact_manifest_schema": manifest["schema"],
        "source_artifact_count": len(manifest["artifacts"]),
    }


def _truth_floor(stratum: str) -> float:
    spec = generator._case_parameters(stratum)
    return float(min(
        abs(complex(spec["overlaps"][radius_index, 0, 0, 0]))
        for radius_index in range(len(two_state.R_VALUES))
    ))


def _run_new_panel(output: Path, global_panel: int, stratum: str,
                   replicate: int, seed: int, contract_bytes: bytes
                   ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    matrices, spec = generator._generate_panel(seed, stratum)
    floor = _truth_floor(stratum)
    panel_dir = output / "panels" / f"{global_panel:02d}_{stratum}"
    snapshot, snapshot_hashes = bootstrap._snapshot_from_matrices(
        panel_dir, matrices, contract_bytes,
        f"F-synthetic-open-stratified-profile-{stratum}-{replicate:02d}",
    )
    try:
        fit_receipt = candidate.fit_three_static_snapshot(snapshot)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        fit_receipt = {
            "certified": False, "physical_score": None,
            "denominator": {"planned": 1, "valid": 0, "failed": 1, "abstained": 0},
            "error_type": type(exc).__name__, "error_message": str(exc),
        }
    _write_json(panel_dir / "fit.json", fit_receipt)
    cells, payload = calibration._panel_profile(
        snapshot, fit_receipt, float(spec["force_truth"]), global_panel,
        floors=(0.0, floor),
    )
    panel_summary = payload["panel_summary"]
    panel_summary.update({
        "stratum": stratum, "replicate": replicate, "seed": seed,
        "oracle_truth_floor": floor,
        "raw_matrices_sha256": snapshot_hashes["raw_matrices"],
        "source_spec": spec,
    })
    payload["panel_summary"] = panel_summary
    _write_json(panel_dir / "profile_receipts.json", payload)
    cell_rows = [{key: value for key, value in cell.items() if key != "attempts"}
                 for cell in cells]
    return panel_summary, {"panel": global_panel, "profile_cells": cells}, {
        "panel": global_panel, "stratum": stratum, "replicate": replicate,
        "snapshot_hashes": snapshot_hashes,
        "fit_status": fit_receipt.get("status"),
        "denominator": fit_receipt.get("denominator"),
        "profile_cell_receipt_count": len(cell_rows),
    }


def _stratum_coverage(panels: list[dict[str, Any]], stratum: str) -> list[dict[str, Any]]:
    selected = [panel for panel in panels if panel["stratum"] == stratum]
    results = []
    for category, floor_selector in (("no_floor", lambda panel: 0.0),
                                     ("oracle_truth_floor", lambda panel: panel["oracle_truth_floor"])):
        rows = []
        for panel in selected:
            floor = float(floor_selector(panel))
            rows.append(next(item for item in panel["floor_results"]
                             if math_isclose(float(item["floor"]), floor)))
        complete = [row for row in rows if row.get("interval") is not None]
        unclipped = [row for row in complete if not row["grid_clipped"]]
        successes = sum(bool(row["contains_truth"]) for row in unclipped)
        results.append({
            "stratum": stratum, "floor_condition": category,
            "floor_values": sorted({float(row["floor"]) for row in rows}),
            "planned_panels": len(rows), "profile_complete_panels": len(complete),
            "unclipped_profiles_used_for_coverage": len(unclipped),
            "truth_contained_panels": successes,
            "observed_containment_fraction": successes / len(unclipped) if unclipped else None,
            "wilson_95_interval": calibration._wilson_interval(successes, len(unclipped)),
            "clipped_intervals": sum(bool(row["grid_clipped"]) for row in complete),
            "failed_or_abstained_panels": len(rows) - len(complete),
        })
    return results


def math_isclose(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-12 * max(1.0, abs(left), abs(right))


def _plot(path: Path, panels: list[dict[str, Any]]) -> None:
    strata = ("weak_ground_overlap",) + OTHER_STRATA
    fig, axes = plt.subplots(len(strata), 2, figsize=(12, 13), sharex=True,
                             layout="constrained")
    for row_index, stratum in enumerate(strata):
        group = [panel for panel in panels if panel["stratum"] == stratum]
        for col_index, category in enumerate(("no_floor", "oracle_truth_floor")):
            ax = axes[row_index, col_index]
            for panel_index, panel in enumerate(group):
                floor = 0.0 if category == "no_floor" else panel["oracle_truth_floor"]
                result = next(item for item in panel["floor_results"]
                              if math_isclose(float(item["floor"]), floor))
                interval = result.get("interval")
                if interval is None:
                    ax.scatter([0.15], [panel_index], marker="x", color="#d55e00")
                    continue
                color = "#0072b2" if result.get("contains_truth") else "#d55e00"
                ax.plot(interval, [panel_index, panel_index], color=color, linewidth=2)
                if result.get("grid_clipped"):
                    if math_isclose(float(interval[0]), float(panel["grid"][0])):
                        ax.scatter([interval[0]], [panel_index], marker="<", color=color)
                    if math_isclose(float(interval[1]), float(panel["grid"][-1])):
                        ax.scatter([interval[1]], [panel_index], marker=">", color=color)
            ax.axvline(0.15, color="black", linestyle="--", linewidth=1)
            ax.set_title(f"{stratum}: {category}")
            ax.set_yticks(range(len(group)), [str(panel["replicate"]) for panel in group])
            ax.grid(axis="x", alpha=0.2)
            if row_index == len(strata) - 1:
                ax.set_xlabel("conditional force set (lattice units)")
    axes[0, 0].set_ylabel("replicate")
    fig.suptitle("Stratified nonlinear force-profile screen (blue contains toy truth; red misses)")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(output: Path = DEFAULT_OUT) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty report directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    weak_panels, weak_receipts, weak_source_info = _checked_weak_inputs()
    panel_summaries = [dict(row) for row in weak_panels]
    for replicate, panel in enumerate(panel_summaries):
        panel["stratum"] = "weak_ground_overlap"
        panel["replicate"] = replicate
        panel["oracle_truth_floor"] = float(panel["truth_overlap_floor_min_R"])
    all_cells = []
    for receipt in weak_receipts:
        all_cells.extend({"panel": receipt["panel"], **{
            key: value for key, value in cell.items() if key != "attempts"
        }}
                         for cell in receipt["profile_receipts"]["profile_cells"])
    panel_artifacts = [{"panel": receipt["panel"], "reused_from": receipt["source_report"],
                        "input_artifact_hashes": receipt["artifact_hashes"]}
                       for receipt in weak_receipts]

    contract_bytes = (WEAK_REPORT / "lane_contract.json").read_bytes()
    global_index = len(panel_summaries)
    for stratum_index, stratum in enumerate(OTHER_STRATA):
        for replicate in range(REPLICATES_PER_STRATUM):
            seed = SEED_BASE + stratum_index * 100 + replicate
            panel, receipt, artifact = _run_new_panel(
                output, global_index, stratum, replicate, seed, contract_bytes,
            )
            panel_summaries.append(panel)
            all_cells.extend({"panel": receipt["panel"], **{
                key: value for key, value in cell.items() if key != "attempts"
            }}
                             for cell in receipt["profile_cells"])
            panel_artifacts.append(artifact)
            global_index += 1

    coverage = [row for stratum in (("weak_ground_overlap",) + OTHER_STRATA)
                for row in _stratum_coverage(panel_summaries, stratum)]
    summary = {
        "schema": "wilson-f-lane-stratified-profile-screen-v1",
        "classification": "SMALL_MULTI_STRATUM_SYNTHETIC_PROFILE_SCREEN",
        "certified": False, "physical_score": None,
        "strata": ("weak_ground_overlap",) + OTHER_STRATA,
        "panel_count": len(panel_summaries),
        "weak_overlap_panels_reused": len(weak_panels),
        "new_panels": len(panel_summaries) - len(weak_panels),
        "replicates_per_new_stratum": REPLICATES_PER_STRATUM,
        "coverage_by_stratum_and_floor_condition": coverage,
        "panel_summaries": panel_summaries,
        "denominator": {
            "planned_panels": len(panel_summaries),
            "valid_candidate_fit_panels": sum(bool(panel["panel_valid"]) for panel in panel_summaries),
            "planned_profile_cells": sum(int(panel.get("planned_profile_cells", 0))
                                          for panel in panel_summaries),
            "valid_profile_cells": sum(int(panel.get("valid_profile_cells", 0))
                                        for panel in panel_summaries),
            "failed_or_abstained_profile_cells": sum(
                int(panel.get("failed_or_abstained_profile_cells", 0)) for panel in panel_summaries
            ),
            "planned_start_fits": sum(int(panel.get("planned_start_fits", 0))
                                       for panel in panel_summaries),
            "valid_start_fits": sum(int(panel.get("valid_start_fits", 0))
                                     for panel in panel_summaries),
            "failed_or_abstained_start_fits": sum(
                int(panel.get("failed_or_abstained_start_fits", 0)) for panel in panel_summaries
            ),
        },
        "weak_panel_source": weak_source_info,
        "physical_gate_without_independent_floor_receipt":
            "ABSTAIN_NO_INDEPENDENT_OVERLAP_FLOOR",
        "scope_limits": [
            "three new panels per added stratum are a small screening sample and cannot establish coverage",
            "nonweak-stratum oracle floors are read from each toy generator truth, not independent production evidence",
            "the nominal profile threshold and coarse force grid are uncalibrated",
            "two starts and local full rank do not prove global identifiability",
            "no physical Wilson input, sealed data, GPU or QUDA is used",
        ],
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "panel_artifacts.json", {"panels": panel_artifacts})
    _write_csv(output / "profile_grid.csv", all_cells)
    _plot(output / "stratified_profile_intervals.png", panel_summaries)
    source_map = {
        "stratified_profile_screen.py": Path(__file__),
        "profile_calibration.py": Path(calibration.__file__),
        "nonlinear_overlap_profile.py": Path(nonlinear.__file__),
        "identifiability_generator.py": Path(generator.__file__),
        "three_static_candidate.py": Path(candidate.__file__),
        "bootstrap_snapshot_helper.py": Path(bootstrap.__file__),
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
    _write_json(output / "protocol.json", {
        "schema": "wilson-f-lane-stratified-profile-protocol-v1",
        "reused_weak_panels": weak_source_info,
        "new_strata": list(OTHER_STRATA),
        "replicates_per_new_stratum": REPLICATES_PER_STRATUM,
        "seed_rule": "2026114400 + 100*stratum_index + replicate_index",
        "new_panel_DGP": "96 ordered blocks; each stress profile uses the previously declared exact positive-spectrum generator setting",
        "profile": "no floor and per-panel oracle truth floor; fixed force grid plus fitted center; two starts per cell; nominal delta objective 3.84",
        "physical_gate": "ABSTAIN_NO_INDEPENDENT_OVERLAP_FLOOR",
        "physical_status": "NOT_READY/UNCERTIFIED; F=GROUND_STATE_FORCE_UNCERTIFIED",
    })
    _write_json(output / "lane_contract.json", json.loads(contract_bytes))
    readme = "\n".join([
        "# Stratified Wilson nonlinear force-profile screen", "",
        "**Classification:** small synthetic stress map; not physical calibration.", "",
        "![Profile intervals across stress strata](stratified_profile_intervals.png)", "",
        "The screen reuses six hash-verified weak-overlap panels from the prior report and adds three new panels each for baseline, near-degenerate, omitted-vacuum truth, and dependent/unequal-variance strata. Each panel compares no floor with a floor equal to the generator's known toy overlap minimum; that oracle value is not an independent production receipt.", "",
        f"Panel fits valid: `{summary['denominator']['valid_candidate_fit_panels']}/{summary['denominator']['planned_panels']}`. Profile cells valid: `{summary['denominator']['valid_profile_cells']}/{summary['denominator']['planned_profile_cells']}`; failed/abstained cells `{summary['denominator']['failed_or_abstained_profile_cells']}`. Start fits valid: `{summary['denominator']['valid_start_fits']}/{summary['denominator']['planned_start_fits']}`; failed/abstained `{summary['denominator']['failed_or_abstained_start_fits']}`.", "",
        "The per-stratum table in `summary.json` reports complete, unclipped profile sets, toy-truth containment and Wilson 95% binomial intervals. Small replicate counts mean these are only screening diagnostics. The nominal objective threshold and grid are uncalibrated; the profiles do not show global identifiability.", "",
        "Without an independently reviewed open-calibration floor receipt the physical force reader remains `ABSTAIN_NO_INDEPENDENT_OVERLAP_FLOOR`. No production Wilson matrices, sealed data, GPU or QUDA were used; both lanes remain `NOT_READY/UNCERTIFIED`, F remains `GROUND_STATE_FORCE_UNCERTIFIED`, and `physical_score=null`.",
    ]) + "\n"
    (output / "README.md").write_text(readme, encoding="utf-8")
    artifacts = {
        str(path.relative_to(output)): _sha(path.read_bytes())
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_json(output / "artifact_hashes.json", {
        "schema": "wilson-stratified-profile-artifact-hashes-v1", "artifacts": artifacts,
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
        "coverage": result["coverage_by_stratum_and_floor_condition"],
    }, indent=2, sort_keys=True))
