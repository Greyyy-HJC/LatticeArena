"""Run the frozen five-stratum, CPU-only Wilson F-lane screening diagnostic.

This is a small screening study for scenario coverage, not a calibration or a
physical Wilson measurement. Each panel retains raw Hermitian matrices,
fit-attempt records, coverage denominators, and artifact hashes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tasks.wilson_loop import f_lane_finite_nt_trace_diagnostic as toy
from tasks.wilson_loop import f_lane_two_sector_omitted_coverage_diagnostic as base


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
PROTOCOL_PATH = TASK_DIR / "f_lane_two_sector_multistratum_screen_protocol.json"
DEFAULT_OUT = TASK_DIR / "f_lane_two_sector_multistratum_screen_20261128_v1"
STRATUM_IDS = (
    "S1_high_overlap_omitted_static",
    "S2_near_degenerate_static",
    "S3_weak_operator_ground_overlap",
    "S4_omitted_vacuum_state",
    "S5_dependent_unequal_variance_blocks",
)
SOURCE_PATHS = {
    "multistratum_screen.py": Path(__file__),
    "multistratum_screen_protocol.json": PROTOCOL_PATH,
    "omitted_coverage_diagnostic.py": Path(base.__file__),
    "finite_nt_trace_diagnostic.py": Path(toy.__file__),
    "two_sector_fit_candidate.py": TASK_DIR / "f_lane_two_sector_fit_candidate.py",
    "transfer_trace_reader.py": TASK_DIR / "f_lane_transfer_trace_reader.py",
    "lane_contract_v2.json": TASK_DIR / "lane_contract_v2.json",
    "source_integrity.py": REPO_ROOT / "core/source_integrity.py",
    "npz_safety.py": REPO_ROOT / "core/npz_safety.py",
}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _as_bool(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.lower() == "true")


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _load_protocol() -> tuple[dict[str, Any], bytes]:
    payload = PROTOCOL_PATH.read_bytes()
    document = json.loads(payload)
    if (document.get("schema") != "wilson-f-two-sector-multistratum-screen-protocol-v1"
            or document.get("classification") != "SYNTHETIC_SCREENING_ONLY_NOT_CALIBRATION"):
        raise ValueError("screening protocol schema/classification mismatch")
    rows = document.get("strata")
    if not isinstance(rows, list) or tuple(row.get("id") for row in rows) != STRATUM_IDS:
        raise ValueError("the five frozen strata or their order changed")
    if document.get("panels_per_stratum") != 12 or document.get("blocks_per_panel") != base.N_BLOCKS:
        raise ValueError("panel or block count differs from the frozen screening plan")
    return document, payload


@contextmanager
def _scenario_truth(scenario: dict[str, Any]) -> Iterator[None]:
    """Install one exact synthetic truth temporarily, restoring module state."""
    original_vacuum = toy.VACUUM_ENERGIES.copy()
    original_static = {radius: values.copy() for radius, values in toy.STATIC_ENERGIES.items()}
    original_overlaps = toy.TRANSITION_OVERLAPS.copy()
    try:
        transform = scenario["transform"]
        scenario_id = scenario["id"]
        if scenario_id == STRATUM_IDS[0]:
            amplitudes = toy.TRANSITION_OVERLAPS.copy()
            amplitudes[:, :, int(transform["static_state_index"]), :] *= float(
                transform["transition_amplitude_multiplier"])
            toy.TRANSITION_OVERLAPS = amplitudes
        elif scenario_id == STRATUM_IDS[1]:
            toy.STATIC_ENERGIES = {
                int(radius): np.asarray(values, dtype=float)
                for radius, values in transform["static_energies_by_R"].items()
            }
        elif scenario_id == STRATUM_IDS[2]:
            amplitudes = toy.TRANSITION_OVERLAPS.copy()
            amplitudes[:, :, int(transform["static_state_index"]),
                       int(transform["operator_index"])] *= float(
                transform["transition_amplitude_multiplier"])
            toy.TRANSITION_OVERLAPS = amplitudes
        elif scenario_id == STRATUM_IDS[3]:
            source_index = int(transform["added_vacuum_amplitude_source_index"])
            multiplier = float(transform["added_vacuum_amplitude_multiplier"])
            extra = multiplier * toy.TRANSITION_OVERLAPS[:, source_index:source_index + 1]
            toy.TRANSITION_OVERLAPS = np.concatenate(
                [toy.TRANSITION_OVERLAPS, extra], axis=1)
            toy.VACUUM_ENERGIES = np.asarray(transform["vacuum_energies"], dtype=float)
        elif scenario_id != STRATUM_IDS[4]:
            raise ValueError(f"unknown frozen scenario {scenario_id}")
        yield
    finally:
        toy.VACUUM_ENERGIES = original_vacuum
        toy.STATIC_ENERGIES = original_static
        toy.TRANSITION_OVERLAPS = original_overlaps


def _noise_spec(scenario: dict[str, Any]) -> dict[str, Any]:
    noise = scenario["noise"]
    spec = {
        "model": noise["model"],
        "scale_by_T": noise.get("innovation_scale_by_T", noise.get("scale_by_T", {})),
    }
    if "rho" in noise:
        spec["rho"] = float(noise["rho"])
    return spec


def _truth_description(scenario: dict[str, Any]) -> str:
    identifier = scenario["id"]
    if identifier == STRATUM_IDS[3]:
        return "3 vacuum x 3 static truth; fitted model remains 2 vacuum x 2 static"
    return str(scenario["truth"])


def _screen_metadata(protocol: dict[str, Any], scenario: dict[str, Any],
                     stratum_index: int) -> dict[str, Any]:
    seed_base = int(protocol["seed_rule"].split("+")[0].strip()) + 100 * stratum_index
    return {
        "classification": protocol["classification"],
        "stratum_id": scenario["id"],
        "stratum_index_zero_based": stratum_index,
        "split_prefix": scenario["id"].lower(),
        "truth": _truth_description(scenario),
        "fit_model": "2 vacuum x 2 static (fixed across all strata)",
        "seed_rule": f"numpy default_rng({seed_base} + panel_index); base follows 2026112900 + 100*stratum_index",
        "open_gates": [
            "twelve independent panels per stratum are diagnostic screening only and cannot certify nominal coverage",
            "no method, interval, fit window, covariance shrinkage or DGP parameter is selected from these screening results",
            "synthetic source does not validate production exporter conventions, matched-Nt thermal/mixed receipts, static-force reference or physical uncertainty",
        ],
        "interpretation": (
            f"This is the frozen 12-panel screen for `{scenario['id']}`. It is used to identify qualitative failure modes and exercise the reproducible report path. "
            "The familywise Wilson interval and panel-cluster pointwise intervals are descriptive Monte Carlo summaries at this small n; they do not establish calibration or pass the separately frozen five-stratum confirmatory rule (606 panels and at least 588 familywise successes per stratum)."
        ),
    }


def _screen_plots(output: Path, summaries: list[dict[str, Any]],
                  panel_rows: list[dict[str, Any]], cell_rows: list[dict[str, Any]]) -> None:
    ids = [row["stratum_id"] for row in summaries]
    labels = [identifier.split("_", 1)[0] for identifier in ids]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    family_rates = [row["coverage"]["familywise_bonferroni72"]["rate"] for row in summaries]
    family_ci = [row["coverage"]["familywise_bonferroni72"]["wilson95"] for row in summaries]
    family_err = np.asarray([
        [max(0.0, rate - interval[0]) for rate, interval in zip(family_rates, family_ci)],
        [max(0.0, interval[1] - rate) for rate, interval in zip(family_rates, family_ci)],
    ])
    axes[0].errorbar(np.arange(len(ids)), family_rates, yerr=family_err,
                     fmt="o", capsize=4, color="#6f4e7c")
    axes[0].axhline(0.95, color="#e28e2c", linestyle="--", linewidth=1,
                    label="nominal familywise 95%")
    axes[0].set_xticks(np.arange(len(ids)), labels)
    axes[0].set_ylim(0, 1.03)
    axes[0].set_title("Familywise 72/72 containment · Wilson 95% MC CI")
    axes[0].set_ylabel("fraction of independent panels")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8, loc="lower left")

    colors = {"q": "#3478bf", "h": "#e28e2c", "x": "#59a14f"}
    offsets = {"q": -0.18, "h": 0.0, "x": 0.18}
    for term in base.TERMS:
        rates = []
        intervals = []
        for summary in summaries:
            entry = summary["coverage"]["pointwise_by_term"][term]
            rates.append(entry["cluster_mean"])
            intervals.append(entry["cluster_t95"])
        errors = np.asarray([
            [max(0.0, rate - interval[0]) for rate, interval in zip(rates, intervals)],
            [max(0.0, interval[1] - rate) for rate, interval in zip(rates, intervals)],
        ])
        axes[1].errorbar(np.arange(len(ids)) + offsets[term], rates, yerr=errors,
                         fmt="o", capsize=3, color=colors[term], label=term.upper())
    axes[1].axhline(0.975, color="#3478bf", linestyle="--", linewidth=1,
                    label="nominal pointwise 97.5%")
    axes[1].set_xticks(np.arange(len(ids)), labels)
    axes[1].set_ylim(0, 1.03)
    axes[1].set_title("Pointwise containment · panel-cluster t 95% CI")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(fontsize=8, loc="lower left", ncol=2)
    fig.suptitle("Wilson F-lane five-stratum screening · synthetic only")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / "screening_coverage.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(ids))
    medians = [row["fit_diagnostics"]["median_jacobian_condition_number"] for row in summaries]
    maxima = [row["fit_diagnostics"]["max_jacobian_condition_number"] for row in summaries]
    axis.scatter(x - 0.08, medians, marker="o", label="median condition number")
    axis.scatter(x + 0.08, maxima, marker="^", label="maximum condition number")
    for row in panel_rows:
        index = ids.index(row["stratum_id"])
        value = row["condition_number"]
        if value is not None:
            axis.scatter(index + np.linspace(-0.035, 0.035, 1)[0], float(value),
                         alpha=0.22, s=15, color="#777777")
    axis.set_yscale("log")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Jacobian condition number (log scale)")
    axis.set_title("Fit conditioning across the 12-panel screens")
    axis.grid(axis="y", which="both", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "condition_numbers.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5.2))
    width = 0.22
    for term_index, term in enumerate(base.TERMS):
        rates = []
        intervals = []
        for identifier in ids:
            panel_means = []
            for panel_index in range(12):
                values = [
                    float(row["estimate"]) - float(row["truth"])
                    for row in cell_rows
                    if row["stratum_id"] == identifier
                    and int(row["panel"]) == panel_index
                    and row["term"] == term
                    and row["estimate"] is not None
                ]
                if values:
                    panel_means.append(float(np.mean(values)))
            rates.append(float(np.mean(panel_means)) if panel_means else 0.0)
            intervals.append(base._cluster_t_interval(panel_means))
        xloc = x + (term_index - 1) * width
        low = [max(0.0, rate - ci[0]) if ci[0] is not None else 0.0
               for rate, ci in zip(rates, intervals)]
        high = [max(0.0, ci[1] - rate) if ci[1] is not None else 0.0
                for rate, ci in zip(rates, intervals)]
        axis.bar(xloc, rates, width=width, label=term.upper(),
                 color=colors[term], alpha=0.85)
        axis.errorbar(xloc, rates, yerr=np.asarray([low, high]), fmt="none",
                      ecolor="#333333", capsize=2)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels)
    axis.set_ylabel("panel-mean estimate − exact scenario truth")
    axis.set_title("Signed q/h/x bias with panel-cluster t 95% intervals")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "signed_bias_by_stratum.png", dpi=180)
    plt.close(fig)


def _summarize_stratum(identifier: str, report_dir: Path,
                       panel_rows_out: list[dict[str, Any]],
                       cell_rows_out: list[dict[str, Any]]) -> dict[str, Any]:
    child = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    panels = _read_csv(report_dir / "panel_results.csv")
    cells = _read_csv(report_dir / "cell_coverage.csv")
    panel_rows = []
    for row in panels:
        converted = {
            "stratum_id": identifier,
            "panel": int(row["panel"]),
            "seed": int(row["seed"]),
            "fit_valid": _as_bool(row["fit_valid"]),
            "fit_status": row["fit_status"],
            "rank": int(row["rank"]) if row["rank"] else None,
            "condition_number": _as_float(row["condition_number"]),
            "objective": _as_float(row["objective"]),
            "input_matrix_sha256": row["input_matrix_sha256"],
            "valid_qhx_cells": int(row["valid_qhx_cells"]),
            "familywise_bonferroni72_contains": _as_bool(
                row["familywise_bonferroni72_contains"]),
        }
        panel_rows.append(converted)
        panel_rows_out.append(converted)
    cell_rows = []
    for row in cells:
        converted = {
            "stratum_id": identifier,
            "panel": int(row["panel"]),
            "seed": int(row["seed"]),
            "R": int(row["R"]),
            "operator_index": int(row["operator_index"]),
            "T": int(row["T"]),
            "N_t": int(row["N_t"]),
            "term": row["term"],
            "truth": float(row["truth"]),
            "estimate": _as_float(row["estimate"]),
            "delta_se": _as_float(row["delta_se"]),
            "pointwise_upper": _as_float(row["pointwise_upper"]),
            "bonferroni72_upper": _as_float(row["bonferroni72_upper"]),
            "pointwise_contains": _as_bool(row["pointwise_contains"]),
            "bonferroni72_contains": _as_bool(row["bonferroni72_contains"]),
            "fit_cell_valid": _as_bool(row["fit_cell_valid"]),
        }
        cell_rows.append(converted)
        cell_rows_out.append(converted)

    fit_valid = sum(row["fit_valid"] for row in panel_rows)
    family_success = sum(row["familywise_bonferroni72_contains"] for row in panel_rows)
    pointwise_by_term = {}
    for term in base.TERMS:
        panel_rates = []
        for panel_index in range(len(panel_rows)):
            selected = [row for row in cell_rows
                        if row["panel"] == panel_index and row["term"] == term]
            panel_rates.append(sum(row["pointwise_contains"] for row in selected) / 24)
        pointwise_by_term[term] = {
            "cluster_mean": float(np.mean(panel_rates)),
            "cluster_t95": base._cluster_t_interval(panel_rates),
            "n_independent_panels": len(panel_rows),
        }
    condition = [row["condition_number"] for row in panel_rows
                 if row["condition_number"] is not None]
    valid_cells = sum(row["fit_cell_valid"] for row in cell_rows)
    planned_cells = len(cell_rows)
    family_rate = family_success / len(panel_rows)
    summary = {
        "stratum_id": identifier,
        "classification": child["classification"],
        "truth": child["generator"]["truth"],
        "noise_model": child["generator"]["noise_model"],
        "denominators": {
            "panels": {"planned": len(panel_rows), "valid": fit_valid,
                       "failed": len(panel_rows) - fit_valid, "abstained": 0},
            "q_h_x_cells": {"planned": planned_cells, "valid": valid_cells,
                            "failed": child["denominators"]["q_h_x_cells"]["failed"],
                            "abstained": child["denominators"]["q_h_x_cells"]["abstained"]},
            "unique_raw_matrix_hashes": len({row["input_matrix_sha256"] for row in panel_rows}),
        },
        "coverage": {
            "familywise_bonferroni72": {
                "successes": family_success, "planned_panels": len(panel_rows),
                "rate": family_rate,
                "wilson95": base._wilson_interval(family_success, len(panel_rows)),
            },
            "pointwise_by_term": pointwise_by_term,
        },
        "fit_diagnostics": {
            "median_jacobian_condition_number": float(np.median(condition)) if condition else None,
            "max_jacobian_condition_number": max(condition) if condition else None,
            "n_condition_numbers": len(condition),
            "failed_fit_status_counts": {
                status: sum(row["fit_status"] == status for row in panel_rows if not row["fit_valid"])
                for status in sorted({row["fit_status"] for row in panel_rows if not row["fit_valid"]})
            },
        },
        "child_report": str(report_dir.name),
        "child_artifact_hashes_sha256": _sha((report_dir / "artifact_hashes.json").read_bytes()),
    }
    return summary


def _artifact_hashes(output: Path) -> dict[str, str]:
    paths = sorted(path for path in output.rglob("*")
                   if path.is_file() and path.name != "artifact_hashes.json")
    hashes = {str(path.relative_to(output)): _sha(path.read_bytes()) for path in paths}
    (output / "artifact_hashes.json").write_bytes(_json_bytes({
        "schema": "wilson-f-two-sector-multistratum-artifact-hashes-v1",
        "artifacts": hashes,
    }))
    return hashes


def build(output: str | Path = DEFAULT_OUT, *, resume: bool = False) -> dict[str, Any]:
    out = Path(output)
    protocol, protocol_bytes = _load_protocol()
    source_payloads = {name: path.read_bytes() for name, path in SOURCE_PATHS.items()}
    source_hashes = {name: _sha(payload) for name, payload in source_payloads.items()}
    run_protocol = {
        "schema": "wilson-f-two-sector-multistratum-screen-run-v1",
        "classification": protocol["classification"],
        "frozen_protocol_sha256": _sha(protocol_bytes),
        "source_hashes": source_hashes,
        "seed_rule": protocol["seed_rule"],
        "strata_order": [row["id"] for row in protocol["strata"]],
        "panels_per_stratum": protocol["panels_per_stratum"],
        "confirmatory_boundary": protocol["interpretation_boundary"],
        "physical_score": None,
        "certified": False,
    }
    run_protocol_bytes = _json_bytes(run_protocol)
    if out.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite existing screen report: {out}")
        saved = json.loads((out / "run_protocol.json").read_text(encoding="utf-8"))
        if saved != run_protocol:
            raise ValueError("resume protocol/source hashes differ from the saved run")
    else:
        out.mkdir(parents=True)
        (out / "protocol.json").write_bytes(protocol_bytes)
        (out / "run_protocol.json").write_bytes(run_protocol_bytes)
        source_dir = out / "source_snapshot"
        source_dir.mkdir()
        for name, payload in source_payloads.items():
            (source_dir / name).write_bytes(payload)

    summaries: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    base_seed = int(protocol["seed_rule"].split("+")[0].strip())
    panels_per_stratum = int(protocol["panels_per_stratum"])
    for stratum_index, scenario in enumerate(protocol["strata"]):
        identifier = scenario["id"]
        child_dir = out / identifier
        seed_base = base_seed + 100 * stratum_index
        metadata = _screen_metadata(protocol, scenario, stratum_index)
        with _scenario_truth(scenario):
            base.build(
                child_dir, panels_per_stratum, resume=child_dir.exists(),
                seed_base=seed_base,
                noise_spec=_noise_spec(scenario),
                report_metadata=metadata,
            )
        summaries.append(_summarize_stratum(identifier, child_dir, panel_rows, cell_rows))

    fit_valid = sum(row["fit_valid"] for row in panel_rows)
    fit_planned = len(panel_rows)
    cell_valid = sum(row["fit_cell_valid"] for row in cell_rows)
    cell_planned = len(cell_rows)
    family_success = sum(row["familywise_bonferroni72_contains"] for row in panel_rows)
    total_family_denominator = sum(row["denominators"]["panels"]["planned"] for row in summaries)
    aggregate = {
        "schema": "wilson-f-two-sector-multistratum-screen-summary-v1",
        "classification": protocol["classification"],
        "physical_score": None,
        "certified": False,
        "f_lane_status": "GROUND_STATE_FORCE_UNCERTIFIED",
        "physical_measurements": {"planned": 0, "valid": 0, "failed": 0, "abstained": 0},
        "certification_gate": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
        "protocol": run_protocol,
        "denominators": {
            "panels": {"planned": fit_planned, "valid": fit_valid,
                       "failed": fit_planned - fit_valid, "abstained": 0},
            "q_h_x_cells": {"planned": cell_planned, "valid": cell_valid,
                            "failed": sum(row["denominators"]["q_h_x_cells"]["failed"] for row in summaries),
                            "abstained": sum(row["denominators"]["q_h_x_cells"]["abstained"] for row in summaries)},
            "familywise_events": {"planned": total_family_denominator,
                                  "valid": fit_planned, "failed": fit_planned - fit_valid,
                                  "abstained": 0},
            "unique_raw_matrix_hashes": len({row["input_matrix_sha256"] for row in panel_rows}),
        },
        "coverage": {
            "familywise_72": {
                "successes": family_success, "planned_panels": total_family_denominator,
                "pooled_rate_descriptive_only": family_success / total_family_denominator,
            },
            "by_stratum": summaries,
        },
        "interpretation_boundary": protocol["interpretation_boundary"],
        "summary": (
            "This 5x12 screen is a synthetic diagnostic under five predeclared stresses. "
            "The Monte Carlo intervals are deliberately descriptive at n=12 and do not alter the frozen 606-panel/588-success confirmatory plan."
        ),
    }
    (out / "panel_results.csv").write_text("", encoding="utf-8")
    _write_csv(out / "panel_results.csv", panel_rows)
    _write_csv(out / "cell_coverage.csv", cell_rows)
    (out / "summary.json").write_bytes(_json_bytes(aggregate))

    bias_rows = []
    for summary in summaries:
        identifier = summary["stratum_id"]
        for term in base.TERMS:
            valid_biases = [row["estimate"] - row["truth"] for row in cell_rows
                            if row["stratum_id"] == identifier and row["term"] == term
                            and row["estimate"] is not None]
            bias_rows.append({
                "stratum_id": identifier,
                "term": term,
                "valid_cells": len(valid_biases),
                "planned_cells": panels_per_stratum * 24,
                "mean_signed_bias_valid_cells": float(np.mean(valid_biases)) if valid_biases else None,
                "median_signed_bias_valid_cells": float(np.median(valid_biases)) if valid_biases else None,
            })
    _write_csv(out / "signed_bias_summary.csv", bias_rows)
    _screen_plots(out, summaries, panel_rows, cell_rows)

    lines = [
        "# Wilson F lane: five-stratum synthetic screen",
        "",
        "**Status:** `SYNTHETIC_SCREENING_ONLY_NOT_CALIBRATION`. The screen does not open sealed data, run GPU/QUDA, create a physical score, or certify the ground-state static force. `physical_score=null`; F remains `GROUND_STATE_FORCE_UNCERTIFIED`.",
        "",
        "## Frozen design",
        "",
        "- Five predeclared strata, 12 independent panels per stratum, 96 blocks per panel; 60 panels total.",
        "- Seeds follow `2026112900 + 100*stratum_index + panel_index`, disjoint from the previous 24-panel run (`2026112800..2026112823`). Each raw matrix NPZ and its input manifest/hash are retained under the corresponding stratum directory.",
        "- Every stratum uses the same fixed 2-vacuum × 2-static fitted model, 25% diagonal shrinkage, relative eigenvalue floor `1e-10`, three starts, 72 q/h/x cells, one-sided 97.5% pointwise uppers, and Bonferroni-72 familywise 95% uppers.",
        "- The 12-panel results are a screening-only signal. They cannot certify coverage and cannot be used to tune the interval, shrinkage, fit window, or DGP.",
        "",
        "## Per-stratum result ledger",
        "",
        "| Stratum | Fit valid / planned | q/h/x cells valid / planned | Familywise successes / panels | Familywise Wilson 95% MC interval | Median Jacobian condition |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        coverage = row["coverage"]["familywise_bonferroni72"]
        lo, hi = coverage["wilson95"]
        lines.append(
            f"| {row['stratum_id']} | {row['denominators']['panels']['valid']}/{row['denominators']['panels']['planned']} | "
            f"{row['denominators']['q_h_x_cells']['valid']}/{row['denominators']['q_h_x_cells']['planned']} | "
            f"{coverage['successes']}/{coverage['planned_panels']} ({coverage['rate']:.3f}) | [{lo:.3f}, {hi:.3f}] | "
            f"{row['fit_diagnostics']['median_jacobian_condition_number']:.4g}"
            if row["fit_diagnostics"]["median_jacobian_condition_number"] is not None else
            f"| {row['stratum_id']} | {row['denominators']['panels']['valid']}/{row['denominators']['panels']['planned']} | "
            f"{row['denominators']['q_h_x_cells']['valid']}/{row['denominators']['q_h_x_cells']['planned']} | "
            f"{coverage['successes']}/{coverage['planned_panels']} ({coverage['rate']:.3f}) | [{lo:.3f}, {hi:.3f}] | NA |"
        )
    lines.extend([
        "",
        "![Familywise and pointwise screening coverage](screening_coverage.png)",
        "",
        "![Fit conditioning by stratum](condition_numbers.png)",
        "",
        "![Signed q/h/x bias](signed_bias_by_stratum.png)",
        "",
        "## Data flow",
        "",
        "```text",
        "freeze five truth/noise transforms -> assign disjoint stratum/panel seeds -> draw 60 Hermitian block panels -> hash raw input -> fit the same 2x2 GLS and compute the same 72 uppers -> retain fit failures/abstentions and cell-level bias -> plot panel-cluster and familywise summaries -> leave physical F gate abstaining",
        "```",
        "",
        "## Reading the screen",
        "",
        "Use the per-stratum rows, CSVs, and figures to locate distinct mechanisms: omitted-state amplitude bias, near-degenerate fit conditioning, unstable weak-overlap ratios, missing-vacuum finite-Nt contamination, or dependent/unequal-variance blocks. The pointwise bars are averaged within each panel before the t interval, so the independent sampling unit is the panel. Familywise bars count a panel as success only when the fit is valid and all 72 q/h/x upper bounds contain their exact synthetic truth; the Wilson interval also has only 12 independent trials per stratum.",
        "",
        "The separately frozen confirmation calculation remains `606` independent panels per stratum and at least `588/606` familywise successes per stratum, under its declared `p=0.98` planning alternative and five-stratum multiplicity rule. This screen is not a substitute. The production F lane still lacks the production correlator/exporter convention evidence, matched-`Nt=32` thermal/mixed receipts, force/reference error budget, and independent review required for physical interpretation. The finite-`T` Creutz C lane remains separate.",
        "",
        "All graphics have a corresponding text interpretation above and raw data/results are retained. No GPU/QUDA or sealed data were used.",
    ])
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashes = _artifact_hashes(out)
    return {
        "output": str(out),
        "panels": fit_planned,
        "fit_valid": fit_valid,
        "fit_failed": fit_planned - fit_valid,
        "q_h_x_cells": f"{cell_valid}/{cell_planned}",
        "familywise": f"{family_success}/{total_family_denominator}",
        "artifacts": len(hashes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true",
                        help="resume deterministic panel inputs in a matching saved report")
    args = parser.parse_args()
    print(json.dumps(build(args.output, resume=args.resume), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
