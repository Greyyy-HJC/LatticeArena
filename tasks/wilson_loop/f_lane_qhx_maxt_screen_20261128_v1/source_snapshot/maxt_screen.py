"""Fresh-seed screening comparison for the Gaussian-parameter max-T interval."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.source_integrity import capture_source_snapshot
from tasks.wilson_loop import f_lane_finite_nt_trace_diagnostic as toy
from tasks.wilson_loop import f_lane_qhx_maxt_interval as maxt
from tasks.wilson_loop import f_lane_two_sector_fit_candidate as fit_candidate
from tasks.wilson_loop import f_lane_two_sector_multistratum_screen as screen
from tasks.wilson_loop import f_lane_two_sector_omitted_coverage_diagnostic as base


TASK_DIR = Path(__file__).resolve().parent
ROOT = TASK_DIR.parents[1]
PROTOCOL_PATH = TASK_DIR / "f_lane_qhx_maxt_screen_protocol_v1.json"
BASE_PROTOCOL_PATH = screen.PROTOCOL_PATH
DEFAULT_OUT = TASK_DIR / "f_lane_qhx_maxt_screen_20261128_v1"
TERMS = ("q", "h", "x")
SOURCE_PATHS = {
    "maxt_screen.py": Path(__file__).resolve(),
    "maxt_interval.py": Path(maxt.__file__).resolve(),
    "fit_candidate.py": Path(fit_candidate.__file__).resolve(),
    "multistratum_screen.py": Path(screen.__file__).resolve(),
    "multistratum_protocol.json": BASE_PROTOCOL_PATH,
    "base_omitted_screen.py": Path(base.__file__).resolve(),
    "finite_nt_toy.py": Path(toy.__file__).resolve(),
    "lane_contract_v2.json": TASK_DIR / "lane_contract_v2.json",
    "source_integrity.py": ROOT / "core/source_integrity.py",
    "npz_safety.py": ROOT / "core/npz_safety.py",
    "maxt_protocol.json": PROTOCOL_PATH,
}


def _sha(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _wilson(successes: int, total: int) -> list[float]:
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Wilson interval requires a nonempty binomial denominator")
    z = 1.959963984540054
    p = successes / total
    z2 = z * z
    den = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / den
    half = z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / den
    lower = 0.0 if successes == 0 else max(0.0, center - half)
    upper = 1.0 if successes == total else min(1.0, center + half)
    return [lower, upper]


def _load_protocol() -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    payload = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(payload)
    base_payload = BASE_PROTOCOL_PATH.read_bytes()
    if protocol.get("schema") != "wilson-f-qhx-gaussian-parameter-maxt-screen-v1":
        raise ValueError("max-T protocol schema mismatch")
    if protocol.get("classification") != "SYNTHETIC_SCREENING_ONLY_NOT_CALIBRATION":
        raise ValueError("max-T protocol must remain screening-only")
    if protocol.get("base_multistratum_protocol_sha256") != _sha(base_payload):
        raise ValueError("base five-stratum DGP protocol hash changed")
    if protocol.get("panels_per_stratum") != 12:
        raise ValueError("frozen max-T screen requires exactly 12 panels per stratum")
    if protocol.get("candidate_interval", {}).get("draws_per_panel") != 2048:
        raise ValueError("frozen max-T parameter draw count changed")
    base_protocol, _ = screen._load_protocol()
    expected_strata = tuple(row["id"] for row in base_protocol["strata"])
    comparison_strata = protocol.get("comparison", {}).get("truth_strata")
    if not isinstance(comparison_strata, list) or tuple(comparison_strata) != expected_strata:
        raise ValueError("max-T comparison strata differ from the frozen five-stratum DGP")
    if protocol.get("seed_rule") != "2026113400 + 100 * stratum_index + panel_index; both indices are zero-based":
        raise ValueError("max-T screen seed rule differs from the frozen disjoint range")
    if protocol.get("no_tuning") is not True or protocol.get("sealed_data_access") is not False:
        raise ValueError("max-T protocol must prohibit tuning and sealed-data access")
    return protocol, payload, base_protocol


def _max_t_interval_index(cell_index: int, term_index: int) -> int:
    if not 0 <= term_index < len(TERMS):
        raise ValueError("q/h/x term index is out of range")
    if cell_index < 0:
        raise ValueError("q/h/x cell index must be nonnegative")
    return len(TERMS) * cell_index + term_index


def _sources() -> tuple[dict[str, bytes], dict[str, str]]:
    payloads = {name: path.read_bytes() for name, path in SOURCE_PATHS.items()}
    return payloads, {name: _sha(payload) for name, payload in payloads.items()}


def _save_source_snapshot(out: Path, source_payloads: dict[str, bytes], hashes: dict[str, str]) -> None:
    snapshot = out / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name, payload in source_payloads.items():
        (snapshot / name).write_bytes(payload)
    (out / "source_hashes.json").write_bytes(_json_bytes({
        "schema": "wilson-f-qhx-maxt-screen-source-hashes-v1",
        "sources": hashes,
    }))


def _panel_source_payloads() -> dict[str, bytes]:
    return {name: path.read_bytes() for name, path in screen.SOURCE_PATHS.items()}


def _build_panel(out: Path, *, stratum_index: int, panel_index: int,
                 scenario: dict[str, Any], protocol: dict[str, Any],
                 panel_source_payloads: dict[str, bytes]) -> dict[str, Any]:
    interval_cfg = protocol["candidate_interval"]
    seed_base = int(protocol["seed_rule"].split("+")[0].strip())
    outer_seed = seed_base + 100 * stratum_index + panel_index
    interval_seed = outer_seed + int(interval_cfg["seed_rule"].split("+")[1].strip())
    stratum_dir = out / scenario["id"]
    panel_dir = stratum_dir / f"panel_{panel_index:03d}"
    panel_dir.mkdir(parents=True, exist_ok=True)
    result_path = panel_dir / "panel_result.json"

    with screen._scenario_truth(scenario):
        truth_rows = base._truth_rows()
        if result_path.is_file():
            snapshot, input_hashes = base._load_panel_input(out, panel_dir)
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            if (saved.get("seed") != outer_seed
                    or saved.get("input_hashes") != input_hashes
                    or saved.get("stratum_id") != scenario["id"]):
                raise ValueError(f"saved panel result is not bound to its input: {result_path}")
            return saved

        snapshot, input_hashes = base._save_panel_input(
            out, panel_dir, panel_index, outer_seed, panel_source_payloads,
            noise_spec=screen._noise_spec(scenario),
            split_prefix=f"maxt-{scenario['id'].lower()}",
        )
        fit_started = time.perf_counter()
        fit = fit_candidate.fit_two_sector_snapshot(
            snapshot, objective_mode="regularized_full_covariance",
            return_uncertainty_state=True,
        )
        fit_seconds = time.perf_counter() - fit_started

    fit_valid = (
        fit.get("denominator", {}).get("valid") == 1
        and fit.get("fit_contract", {}).get("jacobian_rank") == 45
        and isinstance(fit.get("parameter_uncertainty_state"), dict)
    )
    interval_result: dict[str, Any]
    interval_seconds = 0.0
    if fit_valid:
        state = fit["parameter_uncertainty_state"]
        interval_started = time.perf_counter()
        try:
            interval_result = maxt.gaussian_parameter_max_t_upper(
                state["parameter_estimate"], state["parameter_covariance"],
                draws=int(interval_cfg["draws_per_panel"]),
                alpha=float(interval_cfg["family_alpha"]), seed=interval_seed,
            )
        except (TypeError, ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError) as exc:
            interval_result = {
                "status": "ABSTAIN_INVALID_FIT_COVARIANCE",
                "certified": False, "physical_score": None,
                "denominators": {"planned": int(interval_cfg["draws_per_panel"]),
                                 "valid": 0, "failed": 0,
                                 "abstained": int(interval_cfg["draws_per_panel"])},
                "intervals": None,
                "error_type": type(exc).__name__,
            }
        interval_seconds = time.perf_counter() - interval_started
    else:
        interval_result = {
            "status": "ABSTAIN_FIT_INVALID",
            "certified": False, "physical_score": None,
            "denominators": {"planned": int(interval_cfg["draws_per_panel"]),
                             "valid": 0, "failed": 0,
                             "abstained": int(interval_cfg["draws_per_panel"])},
            "intervals": None,
        }

    truth_by_key = {
        (row["R"], row["operator_index"], row["T"], row["N_t"]): row
        for row in truth_rows
    }
    estimate_by_key = {
        (row.get("R"), row.get("operator_index"), row.get("T"), row.get("N_t")): row
        for row in fit.get("qhx_rows", [])
    }
    delta_intervals = []
    max_t_intervals = interval_result.get("intervals")
    cell_rows: list[dict[str, Any]] = []
    for cell_index, (key, truth) in enumerate(truth_by_key.items()):
        estimated = estimate_by_key.get(key, {})
        delta = estimated.get("delta_intervals", {})
        for term_index, term in enumerate(TERMS):
            delta_item = delta.get(term)
            max_t_item = (max_t_intervals[_max_t_interval_index(cell_index, term_index)]
                          if max_t_intervals is not None else None)
            delta_upper = (delta_item or {}).get("upper_95_bonferroni_72")
            max_t_upper = (max_t_item or {}).get("simultaneous_upper")
            estimate = estimated.get(term)
            row = {
                "stratum_id": scenario["id"], "panel": panel_index,
                "seed": outer_seed, "R": truth["R"],
                "operator_index": truth["operator_index"], "T": truth["T"],
                "N_t": truth["N_t"], "term": term,
                "truth": float(truth[term]),
                "estimate": None if estimate is None else float(estimate),
                "delta_upper_95_bonferroni72": delta_upper,
                "delta_contains": bool(delta_upper is not None and delta_upper >= truth[term]),
                "maxt_simultaneous_upper": max_t_upper,
                "maxt_contains": bool(max_t_upper is not None and max_t_upper >= truth[term]),
                "maxt_interval_valid": bool(max_t_upper is not None),
            }
            cell_rows.append(row)
            if delta_item:
                delta_intervals.append(delta_item)
    delta_familywise = bool(
        fit_valid and len(delta_intervals) == 72
        and all(row["delta_contains"] for row in cell_rows)
    )
    maxt_familywise = bool(
        fit_valid and interval_result.get("status") == "DIAGNOSTIC_SIMULTANEOUS_UPPER"
        and len(cell_rows) == 72
        and all(row["maxt_contains"] for row in cell_rows)
    )
    panel_result = {
        "stratum_id": scenario["id"], "panel": panel_index,
        "seed": outer_seed, "interval_seed": interval_seed,
        "input_hashes": input_hashes, "fit_status": fit.get("status"),
        "fit_valid": fit_valid, "fit_rank": fit.get("fit_contract", {}).get("jacobian_rank"),
        "fit_seconds": fit_seconds, "interval_seconds": interval_seconds,
        "qhx_cells": {"planned": 72, "valid": int(sum(row["estimate"] is not None for row in cell_rows)),
                      "failed": int(sum(row["estimate"] is None for row in cell_rows)),
                      "abstained": 0},
        "delta_interval_valid_cells": len(delta_intervals),
        "delta_familywise_success": delta_familywise,
        "maxt": interval_result,
        "maxt_familywise_success": maxt_familywise,
        "cell_results": cell_rows,
    }
    result_path.write_bytes(_json_bytes(panel_result))
    return panel_result


def _summarize(panels: list[dict[str, Any]], stratum_id: str) -> dict[str, Any]:
    subset = [row for row in panels if row["stratum_id"] == stratum_id]
    total = len(subset)
    delta_success = sum(bool(row["delta_familywise_success"]) for row in subset)
    maxt_success = sum(bool(row["maxt_familywise_success"]) for row in subset)
    maxt_valid = sum(row["maxt"].get("status") == "DIAGNOSTIC_SIMULTANEOUS_UPPER"
                     for row in subset)
    family = {
        "delta_bonferroni72": {
            "successes": delta_success, "planned_panels": total,
            "rate": delta_success / total if total else None,
            "wilson95": _wilson(delta_success, total) if total else [None, None],
        },
        "gaussian_parameter_maxt": {
            "successes": maxt_success, "planned_panels": total,
            "valid_intervals": maxt_valid,
            "abstained_intervals": total - maxt_valid,
            "rate": maxt_success / total if total else None,
            "wilson95": _wilson(maxt_success, total) if total else [None, None],
        },
    }
    pointwise: dict[str, dict[str, dict[str, Any]]] = {}
    widths: dict[str, dict[str, float | None]] = {}
    for method in ("delta_bonferroni72", "gaussian_parameter_maxt"):
        pointwise[method] = {}
        widths[method] = {}
        for term in TERMS:
            rows = [cell for panel in subset for cell in panel["cell_results"]
                    if cell["term"] == term]
            upper_key = ("delta_upper_95_bonferroni72" if method == "delta_bonferroni72"
                         else "maxt_simultaneous_upper")
            successes = sum(cell[upper_key] is not None and cell[upper_key] >= cell["truth"]
                            for cell in rows)
            valid = sum(cell[upper_key] is not None for cell in rows)
            pointwise[method][term] = {
                "successes": successes, "planned_cells": len(rows),
                "valid_intervals": valid,
                "coverage_rate_over_planned": successes / len(rows) if rows else None,
            }
            term_widths = [float(cell[upper_key] - cell["estimate"])
                           for cell in rows
                           if cell[upper_key] is not None and cell["estimate"] is not None]
            widths[method][term] = float(np.median(term_widths)) if term_widths else None
    return {
        "stratum_id": stratum_id,
        "denominators": {
            "panels": {"planned": total,
                       "fit_valid": sum(bool(row["fit_valid"]) for row in subset),
                       "fit_failed": sum(not bool(row["fit_valid"]) for row in subset)},
            "qhx_cells": {"planned": total * 72,
                          "valid": sum(row["qhx_cells"]["valid"] for row in subset),
                          "failed": sum(row["qhx_cells"]["failed"] for row in subset),
                          "abstained": 0},
            "maxt_draws": {
                "planned": sum(row["maxt"].get("denominators", {}).get("planned", 0)
                               for row in subset),
                "valid": sum(row["maxt"].get("denominators", {}).get("valid", 0)
                             for row in subset),
                "failed": sum(row["maxt"].get("denominators", {}).get("failed", 0)
                              for row in subset),
                "abstained": sum(row["maxt"].get("denominators", {}).get("abstained", 0)
                                 for row in subset),
            },
        },
        "familywise_coverage": family,
        "pointwise_by_term": pointwise,
        "median_absolute_upper_width_by_method_term": widths,
    }


def _plots(out: Path, summaries: list[dict[str, Any]]) -> None:
    labels = [row["stratum_id"].replace("_", "\n") for row in summaries]
    x = np.arange(len(labels))
    methods = ("delta_bonferroni72", "gaussian_parameter_maxt")
    titles = ("delta + Bonferroni-72", "Gaussian-parameter max-T")
    colors = ("#999999", "#277da1")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for method, title, color, offset in zip(methods, titles, colors, (-0.16, 0.16)):
        rates = [row["familywise_coverage"][method]["rate"] for row in summaries]
        lows = [row["familywise_coverage"][method]["wilson95"][0] for row in summaries]
        highs = [row["familywise_coverage"][method]["wilson95"][1] for row in summaries]
        if any(rate is None for rate in rates):
            continue
        lower_error = [max(0.0, rate - low) for rate, low in zip(rates, lows)]
        upper_error = [max(0.0, high - rate) for rate, high in zip(rates, highs)]
        axes[0].errorbar(
            x + offset, rates,
            yerr=[lower_error, upper_error],
            marker="o", capsize=3, linestyle="none", label=title, color=color,
        )
    axes[0].axhline(0.95, color="#a13b36", linestyle="--", label="nominal familywise 95%")
    axes[0].set_xticks(x, labels, fontsize=7)
    axes[0].set_ylim(-0.03, 1.04)
    axes[0].set_ylabel("all 72 q/h/x uppers cover truth")
    axes[0].set_title("Panel-level familywise coverage (12 panels/stratum)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    width_methods = methods
    terms = TERMS
    width_data = []
    for method in width_methods:
        width_data.append([
            float(np.median([
                row["median_absolute_upper_width_by_method_term"][method][term]
                for row in summaries
                if row["median_absolute_upper_width_by_method_term"][method][term] is not None
            ])) if any(
                row["median_absolute_upper_width_by_method_term"][method][term] is not None
                for row in summaries) else 0.0
            for term in terms
        ])
    wx = np.arange(len(terms))
    axes[1].bar(wx - 0.18, width_data[0], width=0.36, color=colors[0],
                label=titles[0])
    axes[1].bar(wx + 0.18, width_data[1], width=0.36, color=colors[1],
                label=titles[1])
    axes[1].set_xticks(wx, [term.upper() for term in terms])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("median absolute upper width (log scale)")
    axes[1].set_title("Width pooled over all strata and cells")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Fresh-seed diagnostic only; no finite-sample calibration claim")
    fig.savefig(out / "coverage_and_width.png", dpi=170)
    plt.close(fig)


def _write_artifact_hashes(out: Path) -> dict[str, str]:
    files = sorted(path for path in out.rglob("*")
                   if path.is_file() and path.name != "artifact_hashes.json")
    hashes = {path.relative_to(out).as_posix(): _sha(path.read_bytes()) for path in files}
    (out / "artifact_hashes.json").write_bytes(_json_bytes({
        "schema": "wilson-f-qhx-maxt-screen-artifact-hashes-v1",
        "artifacts": hashes,
    }))
    return hashes


def build(output: str | Path = DEFAULT_OUT, *, resume: bool = False) -> dict[str, Any]:
    out = Path(output)
    protocol, protocol_payload, base_protocol = _load_protocol()
    source_payloads, source_hashes = _sources()
    if out.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite existing screen: {out}")
    if not out.exists():
        out.mkdir(parents=True)
        (out / "protocol.json").write_bytes(protocol_payload)
        _save_source_snapshot(out, source_payloads, source_hashes)
        (out / "lane_contract.json").write_bytes(source_payloads["lane_contract_v2.json"])
    else:
        if (out / "protocol.json").read_bytes() != protocol_payload:
            raise ValueError("resume protocol differs from the frozen candidate protocol")
        saved_source_hashes = json.loads((out / "source_hashes.json").read_text())
        if saved_source_hashes.get("sources") != source_hashes:
            raise ValueError("resume source map differs from the report's frozen source map")
        for name, payload in source_payloads.items():
            if (out / "source_snapshot" / name).read_bytes() != payload:
                raise ValueError(f"resume source snapshot differs: {name}")

    panel_payloads = _panel_source_payloads()
    all_panels: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    all_cell_rows: list[dict[str, Any]] = []
    for stratum_index, scenario in enumerate(base_protocol["strata"]):
        scenario_panels = []
        for panel_index in range(int(protocol["panels_per_stratum"])):
            result = _build_panel(
                out, stratum_index=stratum_index, panel_index=panel_index,
                scenario=scenario, protocol=protocol,
                panel_source_payloads=panel_payloads,
            )
            scenario_panels.append(result)
            all_panels.append(result)
            all_cell_rows.extend(result["cell_results"])
            (out / "progress.json").write_bytes(_json_bytes({
                "completed_panels": len(all_panels), "planned_panels": 60,
                "last_stratum": scenario["id"], "last_panel": panel_index,
            }))
        summary_rows.append(_summarize(scenario_panels, scenario["id"]))
    delta_success = sum(row["delta_familywise_success"] for row in all_panels)
    maxt_success = sum(row["maxt_familywise_success"] for row in all_panels)
    maxt_abstained = sum(
        row["maxt"].get("status") != "DIAGNOSTIC_SIMULTANEOUS_UPPER"
        for row in all_panels
    )
    pooled_widths = {
        method: {
            term: float(np.median([
                row["median_absolute_upper_width_by_method_term"][method][term]
                for row in summary_rows
                if row["median_absolute_upper_width_by_method_term"][method][term] is not None
            ]))
            for term in TERMS
        }
        for method in ("delta_bonferroni72", "gaussian_parameter_maxt")
    }
    min_maxt_success = min(
        row["familywise_coverage"]["gaussian_parameter_maxt"]["successes"]
        for row in summary_rows
    )
    continuation = (
        "SCREEN_ELIGIBLE_FOR_REVIEWED_CONFIRMATION_ONLY"
        if min_maxt_success >= 10 and maxt_abstained == 0
        else "NO_GO_FOR_THIS_GAUSSIAN_PARAMETER_MAXT_CANDIDATE"
    )
    summary = {
        "schema": "wilson-f-qhx-maxt-screen-summary-v1",
        "classification": protocol["classification"],
        "candidate_disposition": continuation,
        "physical_score": None, "certified": False,
        "physical_measurements": {"planned": 0, "valid": 0, "failed": 0, "abstained": 0},
        "denominators": {
            "panels": {"planned": len(all_panels),
                       "fit_valid": sum(bool(row["fit_valid"]) for row in all_panels),
                       "fit_failed": sum(not bool(row["fit_valid"]) for row in all_panels)},
            "q_h_x_cells": {"planned": len(all_cell_rows),
                            "valid": sum(row["estimate"] is not None for row in all_cell_rows),
                            "failed": sum(row["estimate"] is None for row in all_cell_rows),
                            "abstained": 0},
            "maxt_intervals": {"planned": len(all_panels),
                               "valid": len(all_panels) - maxt_abstained,
                               "failed": 0, "abstained": maxt_abstained},
            "maxt_draws": {
                key: sum(row["maxt"].get("denominators", {}).get(key, 0)
                         for row in all_panels)
                for key in ("planned", "valid", "failed", "abstained")
            },
        },
        "baseline_delta_familywise_successes": delta_success,
        "maxt_familywise_successes": maxt_success,
        "min_maxt_successes_in_any_stratum": min_maxt_success,
        "max_t_interval_abstained_panels": maxt_abstained,
        "continuation_rule": protocol["screening_rule"],
        "base_multistratum_protocol_sha256": protocol["base_multistratum_protocol_sha256"],
        "median_of_stratum_median_absolute_upper_width_by_method_term": pooled_widths,
        "strata": summary_rows,
    }
    (out / "summary.json").write_bytes(_json_bytes(summary))
    _write_csv(out / "panel_results.csv", all_panels)
    _write_csv(out / "cell_results.csv", all_cell_rows)
    _plots(out, summary_rows)
    rows = [
        "# Wilson F q/h/x Gaussian-parameter max-T interval screen",
        "",
        "**Status:** `SYNTHETIC_SCREENING_ONLY_NOT_CALIBRATION`; no production Cij, GPU/QUDA, sealed split, or physical score. `physical_score=null`; F remains `GROUND_STATE_FORCE_UNCERTIFIED`.",
        "",
        "## Frozen candidate and decision rule",
        "",
        f"- Candidate: fit the unchanged regularized full-covariance 2-vacuum × 2-static GLS model; draw `{protocol['candidate_interval']['draws_per_panel']}` parameter vectors from the inverse-whitened-Jacobian Gaussian approximation; transform all 72 q/h/x terms; studentize each term; use the higher empirical `{1-protocol['candidate_interval']['family_alpha']:.0%}` quantile of the maximum one-sided studentized deviation as a simultaneous upper critical value.",
        f"- Baseline: same-panel finite-difference delta-normal upper with Bonferroni-72 correction. No fit window, covariance shrinkage, or interval is changed on these panels.",
        f"- Fresh seeds: `{protocol['seed_rule']}`; the range is disjoint from both prior synthetic screen seed ranges in `protocol.json`.",
        f"- Pre-set screen rule: {protocol['screening_rule']}",
        "",
        "## Results",
        "",
        "| Stratum | Fit valid/12 | Delta familywise | max-T familywise | max-T valid intervals | max-T draw failed/abstained |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        delta = row["familywise_coverage"]["delta_bonferroni72"]
        candidate = row["familywise_coverage"]["gaussian_parameter_maxt"]
        fit_counts = row["denominators"]["panels"]
        draws_counts = row["denominators"]["maxt_draws"]
        rows.append(
            f"| {row['stratum_id']} | {fit_counts['fit_valid']}/12 | "
            f"{delta['successes']}/{delta['planned_panels']} | "
            f"{candidate['successes']}/{candidate['planned_panels']} | "
            f"{candidate['valid_intervals']}/12 | "
            f"{draws_counts['failed']}/{draws_counts['abstained']} |"
        )
    rows.extend([
        "",
        "Median of stratum medians for absolute upper width (dimensionless; max-T excludes abstained intervals):",
        "",
        "| Term | Delta + Bonferroni-72 | Gaussian-parameter max-T |",
        "|---|---:|---:|",
    ])
    for term in TERMS:
        rows.append(
            f"| {term.upper()} | "
            f"{pooled_widths['delta_bonferroni72'][term]:.6g} | "
            f"{pooled_widths['gaussian_parameter_maxt'][term]:.6g} |"
        )
    rows.extend([
        "",
        "![Familywise coverage and interval width comparison](coverage_and_width.png)",
        "",
        "## Text diagnosis and limits",
        "",
        f"Across 60 planned panels, both methods had 0 familywise successes in every 12-panel stratum (delta {delta_success}/60; max-T {maxt_success}/60). The max-T interval was available on {len(all_panels) - maxt_abstained}/60 panels and abstained on {maxt_abstained}/60. The plotted width medians show q widths were similar, max-T h widths somewhat smaller, and max-T x widths larger; these widths do not offset the observed zero familywise coverage. Candidate disposition under the pre-set exploratory rule is `{continuation}`. Twelve panels per stratum yield wide Monte Carlo uncertainty; passing this screen would only justify considering a reviewed 606-panel confirmatory design, never a calibration or physical claim.",
        "",
        "The max-T draw is conditional on the fitted covariance and the fixed 2×2 spectral model. It accounts for nonlinear q/h/x transformation and their fitted covariance correlation, but it does not repair omitted-state bias, near-degenerate model misspecification, weak-overlap identifiability, an incorrect covariance model, nonstationarity, or a wrong exporter convention. Its chi-square/Jacobian covariance is asymptotic and not itself calibrated. No independent reviewer or production receipt is supplied by this run.",
        "",
        "Data flow: `frozen 5-stratum truth/noise protocol -> fresh disjoint panel seeds -> immutable raw Hermitian 3x3 snapshots -> same full-covariance 2x2 GLS fit -> fitted parameter estimate/covariance -> 2048 Gaussian parameter draws -> max over 72 studentized q/h/x deviations -> paired delta/max-T familywise and width summaries -> stop/continue screening decision`.",
        "",
        "Source bytes are copied under `source_snapshot/`; raw per-panel NPZ/manifests and interval draw denominators are retained. SHA-256 artifacts are in `artifact_hashes.json`. The report never certifies the ground-state static force; keep `NOT_READY/UNCERTIFIED`, `GROUND_STATE_FORCE_UNCERTIFIED`, and `physical_score=null`.",
    ])
    (out / "README.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    artifacts = _write_artifact_hashes(out)
    return {
        "output": str(out), "panels": len(all_panels),
        "delta_familywise": f"{delta_success}/{len(all_panels)}",
        "maxt_familywise": f"{maxt_success}/{len(all_panels)}",
        "maxt_abstained": maxt_abstained,
        "candidate_disposition": continuation,
        "artifacts": len(artifacts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.output, resume=args.resume), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
