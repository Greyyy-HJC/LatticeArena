"""Attach shared receipts to the saved synthetic Eq. (3.26) report.

This wrapper verifies and copies the existing diagnostic; it does not rerun its
formula grid, estimate project chain inputs, or access physical/sealed data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from core.receipts import run_receipt, scenario_spec


ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPORT = ROOT / "results/pion_projection/bounded_block_risk_markov_varying_20261124_v1"
DEFAULT_OUT = ROOT / "results/pion_projection/bounded_block_risk_markov_varying_receipt_20261124_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _manifest_files(source: Path) -> dict[str, str]:
    raw = json.loads((source / "artifact_hashes.json").read_text())
    if not isinstance(raw, dict) or raw.get("algorithm") != "sha256":
        raise ValueError("source manifest must declare sha256")
    files = raw.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("source manifest must contain a nonempty files mapping")
    for name, expected in files.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ValueError("source manifest entries must map paths to hashes")
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe source manifest path: {name}")
        path = source / rel
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"source report artifact failed SHA-256 check: {name}")
    return files


def _counts(rows: list[dict[str, str]], *, unit_name: str,
            valid_predicate, abstain_predicate=None) -> dict[str, Any]:
    result = {key: 0 for key in ("planned", "valid", "failed", "abstained")}
    reasons: dict[str, dict[str, int]] = {"failed": {}, "abstained": {}}
    for row in rows:
        planned = int(row.get("planned", "1"))
        valid = int(row.get("valid", "1" if valid_predicate(row) else "0"))
        failed = int(row.get("failed", "0"))
        if "abstained" in row:
            abstained = int(row["abstained"])
        elif abstain_predicate is not None and abstain_predicate(row):
            abstained = planned - valid - failed
        else:
            abstained = 0
        if planned != valid + failed + abstained:
            raise ValueError(f"{unit_name} denominator does not close")
        for key, value in zip(result, (planned, valid, failed, abstained)):
            result[key] += value
        if failed:
            reason = row.get("failure_reason", "Synthetic formula fixture failure.")
            reasons["failed"][reason] = reasons["failed"].get(reason, 0) + failed
        if abstained:
            reason = row.get("abstention_reason", "Synthetic bound does not reach a positive zero-threshold LCB.")
            reasons["abstained"][reason] = reasons["abstained"].get(reason, 0) + abstained
    return {"unit": unit_name, **result, "reasons": reasons}


def _read_source(source: Path) -> tuple[dict[str, str], dict[str, Any]]:
    manifest = _manifest_files(source)
    protocol = json.loads((source / "protocol.json").read_text())
    if protocol.get("status") != "SYNTHETIC_ANALYTIC_SENSITIVITY_ONLY":
        raise ValueError("source report is not the expected synthetic analytic diagnostic")
    if protocol.get("physical_score") is not None or protocol.get("physical_data_access") != "none":
        raise ValueError("source report must have no physical score or data access")

    with (source / "scenario_grid.csv").open(newline="") as stream:
        grid_rows = list(csv.DictReader(stream))
    expected_grid = (len(protocol["n_grid"]) * len(protocol["k_ps_grid"])
                     * len(protocol["gamma_ps_lower_grid"]) * 2)
    if len(grid_rows) != expected_grid:
        raise ValueError(f"expected {expected_grid} analytic grid rows")
    grid = _counts(
        grid_rows, unit_name="hypothetical Eq. (3.26) analytic cell",
        valid_predicate=lambda row: row.get("best_case_zero_threshold_reachable") == "True",
        abstain_predicate=lambda row: row.get("best_case_zero_threshold_reachable") == "False",
    )

    with (source / "tv_sensitivity.csv").open(newline="") as stream:
        tv_rows = list(csv.DictReader(stream))
    if len(tv_rows) != 6:
        raise ValueError("expected six initial-law TV sensitivity rows")
    tv_counts = {key: 0 for key in ("planned", "valid", "failed", "abstained")}
    tv_reasons: dict[str, dict[str, int]] = {"failed": {}, "abstained": {}}
    for row in tv_rows:
        status = row.get("status", "")
        if status == "BEST_CASE_POSITIVE_LCB_REACHABLE":
            state = "valid"
        elif status.startswith("ERROR"):
            state = "failed"
        elif status.startswith("ABSTAIN"):
            state = "abstained"
        else:
            raise ValueError(f"unexpected TV sensitivity status: {status!r}")
        tv_counts["planned"] += 1
        tv_counts[state] += 1
        if state != "valid":
            tv_reasons[state][status] = tv_reasons[state].get(status, 0) + 1
    tv = {"unit": "hypothetical post-burn-in TV sensitivity cell",
          **tv_counts, "reasons": tv_reasons}

    fixtures = json.loads((source / "fixtures.json").read_text())
    fixture_units: dict[str, Any] = {}
    fixture_fields = (
        ("partial_block_slots", "partial_failure_fixture", "denominators",
         "partial_failure_fixture_denominators"),
        ("partial_comparison_result", "partial_failure_fixture", "result_denominators",
         "partial_failure_result_denominators"),
        ("all_failed_block_slots", "all_failed_fixture", "denominators",
         "all_failed_fixture_denominators"),
        ("all_failed_comparison_result", "all_failed_fixture", "result_denominators",
         "all_failed_result_denominators"),
    )
    for name, fixture_name, field, duplicate_key in fixture_fields:
        counts = fixtures[fixture_name][field]
        if counts != fixtures.get(duplicate_key):
            raise ValueError(f"duplicated fixture denominator disagrees: {duplicate_key}")
        fixture_units[name] = {
            "unit": name.replace("_", " "),
            **{key: int(counts[key]) for key in ("planned", "valid", "failed", "abstained")},
            "reasons": {"failed": {}, "abstained": {}},
        }
        if fixture_units[name]["planned"] != sum(
                fixture_units[name][key] for key in ("valid", "failed", "abstained")):
            raise ValueError(f"fixture denominator does not close: {name}")
        if fixture_units[name]["failed"]:
            fixture_units[name]["reasons"]["failed"] = {
                "Synthetic illustrative failure retained in the planned denominator; not a measurement failure.":
                    fixture_units[name]["failed"]
            }
        if fixture_units[name]["abstained"]:
            fixture_units[name]["reasons"]["abstained"] = {
                "All planned illustrative blocks failed; no conditional LCB is emitted.":
                    fixture_units[name]["abstained"]
            }

    return manifest, {"protocol": protocol, "grid": grid, "tv": tv,
                      "fixtures": fixtures, "fixture_units": fixture_units}


def build_report(output: str | Path = DEFAULT_OUT, *,
                 source: str | Path = SOURCE_REPORT) -> dict[str, Any]:
    source_path = Path(source).resolve()
    out = Path(output)
    source_manifest, parsed = _read_source(source_path)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source_path, out / "source_report")

    snapshots = {
        "snapshot_markov_varying.py": Path(__file__).with_name("bounded_block_risk_markov_varying.py"),
        "snapshot_markov_varying_diagnostic.py": Path(__file__).with_name("bounded_block_risk_markov_varying_diagnostic.py"),
        "snapshot_contract.py": Path(__file__).with_name("bounded_block_risk_contract.py"),
        "snapshot_contract_v1.json": Path(__file__).with_name("bounded_block_risk_contract_v1.json"),
        "snapshot_core_receipts.py": ROOT / "core/receipts.py",
        "snapshot_markov_varying_receipt.py": Path(__file__),
    }
    for name, path in snapshots.items():
        (out / name).write_bytes(path.read_bytes())

    source_hashes = {
        "tasks/pion_projection/bounded_block_risk_markov_varying.py": _sha256(out / "snapshot_markov_varying.py"),
        "tasks/pion_projection/bounded_block_risk_markov_varying_diagnostic.py": _sha256(out / "snapshot_markov_varying_diagnostic.py"),
        "tasks/pion_projection/bounded_block_risk_contract.py": _sha256(out / "snapshot_contract.py"),
        "tasks/pion_projection/bounded_block_risk_contract_v1.json": _sha256(out / "snapshot_contract_v1.json"),
        "core/receipts.py": _sha256(out / "snapshot_core_receipts.py"),
        "tasks/pion_projection/bounded_block_risk_markov_varying_receipt.py": _sha256(out / "snapshot_markov_varying_receipt.py"),
        "source_report/artifact_hashes.json": _sha256(out / "source_report/artifact_hashes.json"),
    }
    units = {
        "analytic_grid_cells": parsed["grid"],
        "tv_sensitivity_cells": parsed["tv"],
        **parsed["fixture_units"],
    }
    denominators = {
        name: {key: value[key] for key in ("unit", "planned", "valid", "failed", "abstained")}
        for name, value in units.items()
    }
    reason_rows = [
        {"unit": name, "state": state, "count": count, "reason": reason}
        for name, value in units.items()
        for state in ("failed", "abstained")
        for reason, count in value["reasons"][state].items()
    ]
    source_manifest_hash = _sha256(source_path / "artifact_hashes.json")
    protocol = {
        "schema": "pion-markov-varying-receipt-v1",
        "status": "SAVED_SYNTHETIC_REPORT_RECEIPT_ONLY",
        "physical_score": None,
        "simulation_or_formula_grid_rerun": False,
        "physical_or_sealed_data_access": False,
        "source_report": "source_report/",
        "source_report_manifest_sha256": source_manifest_hash,
        "source_report_manifest_entries": len(source_manifest),
        "analysis_units": list(denominators),
        "receipt_status_meaning": "all planned diagnostic units are accounted for; not theorem validity, coverage, or scientific success",
    }
    _write_json(out / "protocol.json", protocol)

    grid, tv = parsed["grid"], parsed["tv"]
    lines = [
        "# Pion Eq. (3.26) sensitivity with shared receipt", "",
        "**SAVED SYNTHETIC/ANALYTIC DIAGNOSTIC ONLY. `physical_score=null`. The verified report was copied; no formula grid, Monte Carlo, physical data, or sealed input was rerun/read.**", "",
        "## Text data flow", "",
        "```text",
        "verify saved Eq. (3.26) report and nested SHA-256 manifest",
        "  -> keep 72 hypothetical analytic cells and 6 TV what-if cells separate",
        "  -> retain partial/all-failed fixture block slots and result statuses separately",
        "  -> attach exact synthetic abstention/failure reasons and source/output hashes",
        "  -> complete means accounting only, not theorem applicability or coverage",
        "  -> physical_score remains null; dependence/anchor/reference/authorization gates stay open",
        "```", "",
        "## Denominators", "",
        "|unit|planned|valid|failed|abstained|not_run|", "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in units.items():
        not_run = value["planned"] - value["valid"] - value["failed"] - value["abstained"]
        lines.append(f"|{value['unit']}|{value['planned']}|{value['valid']}|{value['failed']}|{value['abstained']}|{not_run}|")
    lines.extend([
        "",
        f"The analytic grid has `{grid['planned']}/{grid['valid']}/{grid['failed']}/{grid['abstained']}` planned/valid/failed/abstained cells. A cell is valid only if its synthetic best-case score could reach a positive zero-threshold LCB; this is not observed performance. The TV table has `{tv['planned']}/{tv['valid']}/{tv['failed']}/{tv['abstained']}` cells; its abstentions are formula bounds under hypothetical TV values, not estimates of actual burn-in distance.",
        "",
        "## Saved figure and limits", "",
        "![Eq. (3.26) radius and M/k_ps sensitivity](source_report/markov_varying_sensitivity.png)", "",
        "The left panel compares the Eq. (3.26) radius with a formula-only Eq. (3.25) scale comparator under assumed `k_ps=4`, `gamma_ps_lower=0.2`, and hypothetical variance bounds. The right panel shows the retained `M/k_ps` factor for balanced and imbalanced hypothetical variance profiles. No plotted input estimates the project chain, maximizing `k_ps`, residue-class variance, initial-law TV, anchor shift, or target equivalence.",
        "",
        "The source report's 10-entry SHA-256 manifest is copied and rechecked. `run_receipt.json` retains separate analytic-grid, TV, fixture-slot and result denominators with reason counts. A complete receipt proves accounting of these saved diagnostic units only; it does not prove the theorem assumptions, finite-sample coverage, physical validity, or input independence.",
        "",
        "## Rebuild the overlay", "",
        "```text",
        "python -B -m tasks.pion_projection.bounded_block_risk_markov_varying_receipt --source results/pion_projection/bounded_block_risk_markov_varying_20261124_v1 --output NEW_DIRECTORY",
        "```", "",
    ])
    (out / "README.md").write_text("\n".join(lines))
    summary = {
        "schema": "pion-markov-varying-receipt-summary-v1",
        "status": protocol["status"],
        "physical_score": None,
        "simulation_or_formula_grid_rerun": False,
        "source_report_manifest_sha256": source_manifest_hash,
        "source_report_manifest_entries": len(source_manifest),
        "denominators": denominators,
    }
    _write_json(out / "summary.json", summary)

    payload_hashes = {
        path.relative_to(out).as_posix(): _sha256(path)
        for path in sorted(out.rglob("*")) if path.is_file()
    }
    receipt = run_receipt(
        scenario_spec(
            "pion-time-varying-markov-saved-diagnostic",
            "analytic cell, TV sensitivity cell, fixture block slot, and result",
            "Receipt overlay for a verified synthetic Eq. (3.26) report; no diagnostic rerun.",
            assumptions=[
                "gamma, k_ps, variance, TV, and score schedules in the source report are hypothetical",
                "fixture receipts are synthetic placeholders and do not establish project assumptions",
                "the time-varying finite-plan mean may differ from the accepted target unless reviewed",
            ],
            source_hashes=source_hashes,
        ),
        denominators,
        artifact_hashes=payload_hashes,
        text_flow=(
            "verified saved Eq. (3.26) diagnostic -> separate analytic/TV cells and fixture units "
            "-> exact abstention/failure reasons -> hashed accounting receipt; no diagnostic rerun "
            "and complete does not mean theorem validity or coverage"
        ),
        reason_rows=reason_rows,
    )
    _write_json(out / "run_receipt.json", receipt)
    output_hashes = {
        path.relative_to(out).as_posix(): _sha256(path)
        for path in sorted(out.rglob("*")) if path.is_file()
    }
    _write_json(out / "artifact_hashes.json", {"algorithm": "sha256", "files": output_hashes})
    return {"status": protocol["status"], "physical_score": None,
            "denominators": denominators, "receipt": receipt,
            "files": list(output_hashes) + ["artifact_hashes.json"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = build_report(args.output, source=args.source)
    print(json.dumps({
        "status": result["status"], "physical_score": result["physical_score"],
        "diagnostic_rerun": False, "denominator_units": list(result["denominators"]),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
