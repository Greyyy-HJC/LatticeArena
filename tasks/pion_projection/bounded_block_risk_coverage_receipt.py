"""Attach shared run receipts to the saved known-graph coverage report.

This wrapper copies and verifies an existing synthetic report. It does not run
the Monte Carlo generator or access physical/sealed data.
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
SOURCE_REPORT = ROOT / "results/pion_projection/bounded_block_risk_coverage_mc_20261123_v2"
DEFAULT_OUT = ROOT / "results/pion_projection/bounded_block_risk_coverage_receipt_20261124_v1"
UNIT_NAMES = {
    "synthetic Monte Carlo cell": "scenario_cell",
    "Monte Carlo replicate": "monte_carlo_replicate",
    "paired planned block event": "paired_planned_block_event",
    "candidate-family block score": "candidate_family_block_score",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_verified_source(source: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = source / "artifact_hashes.json"
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("source artifact manifest must be a nonempty mapping")
    for name, expected in manifest.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ValueError("source artifact manifest entries must map names to hashes")
        path = source / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"source report artifact failed SHA-256 check: {name}")

    with (source / "denominator_summary.csv").open(newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    rows = []
    for row in source_rows:
        unit = row.get("unit")
        if unit not in UNIT_NAMES:
            raise ValueError(f"unexpected source denominator unit: {unit!r}")
        counts = {
            key: int(row[key])
            for key in ("planned", "valid", "failed", "abstained")
        }
        if counts["planned"] != counts["valid"] + counts["failed"] + counts["abstained"]:
            raise ValueError(f"source denominator does not close: {unit}")
        rows.append({**row, **counts, "receipt_name": UNIT_NAMES[unit]})
    if set(row["unit"] for row in rows) != set(UNIT_NAMES):
        raise ValueError("source denominator summary is missing a required analysis unit")
    return manifest, rows


def build_report(output: str | Path = DEFAULT_OUT, *,
                 source: str | Path = SOURCE_REPORT) -> dict[str, Any]:
    """Copy an existing coverage report and attach a versioned shared receipt."""
    source_path = Path(source).resolve()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    source_manifest, rows = _load_verified_source(source_path)
    source_summary = json.loads((source_path / "summary.json").read_text())
    if source_summary.get("physical_score") is not None:
        raise ValueError("source report must keep physical_score null")

    source_copy = out / "source_report"
    shutil.copytree(source_path, source_copy)
    snapshots = {
        "snapshot_bounded_block_risk_coverage_mc.py": Path(__file__).with_name(
            "bounded_block_risk_coverage_mc.py"),
        "snapshot_dependency_graph_bound.py": Path(__file__).with_name(
            "dependency_graph_bound.py"),
        "snapshot_core_receipts.py": ROOT / "core/receipts.py",
        "snapshot_coverage_receipt_wrapper.py": Path(__file__),
    }
    for name, path in snapshots.items():
        (out / name).write_bytes(path.read_bytes())

    source_hashes = {
        "tasks/pion_projection/bounded_block_risk_coverage_mc.py": _sha256(
            out / "snapshot_bounded_block_risk_coverage_mc.py"),
        "tasks/pion_projection/dependency_graph_bound.py": _sha256(
            out / "snapshot_dependency_graph_bound.py"),
        "core/receipts.py": _sha256(out / "snapshot_core_receipts.py"),
        "tasks/pion_projection/bounded_block_risk_coverage_receipt.py": _sha256(
            out / "snapshot_coverage_receipt_wrapper.py"),
    }
    scenario = scenario_spec(
        "pion-known-graph-coverage-saved-report",
        "scenario cell, Monte Carlo replicate, paired block event, and family score",
        "Receipt overlay for the verified 2026-11-23 synthetic report; no simulations rerun.",
        assumptions=[
            "the known graph and bounded score are true only by synthetic construction",
            "failure events are hypothetical Bernoulli coding events, not measured failures",
            "the source report contains no anchor term or project-specific dependence evidence",
        ],
        source_hashes=source_hashes,
    )
    denominators = {}
    reason_rows = []
    for row in rows:
        name = row["receipt_name"]
        denominators[name] = {
            "unit": row["unit"],
            "planned": row["planned"],
            "valid": row["valid"],
            "failed": row["failed"],
            "abstained": row["abstained"],
        }
        if row["failed"]:
            reason_rows.append({
                "unit": name,
                "state": "failed",
                "count": row["failed"],
                "reason": row["interpretation"],
            })

    source_manifest_hash = _sha256(source_path / "artifact_hashes.json")
    protocol = {
        "schema": "pion-known-graph-coverage-receipt-v1",
        "status": "SAVED_SYNTHETIC_REPORT_RECEIPT_ONLY",
        "physical_score": None,
        "data_access": "verified saved synthetic report only; no simulation rerun",
        "source_report": "source_report/",
        "source_report_manifest_sha256": source_manifest_hash,
        "source_report_manifest_entries": len(source_manifest),
        "receipt_schema": "latticearena-run-receipt-v1",
        "receipt_status_meaning": "all planned units are accounted for; not scientific success",
        "analysis_units": list(UNIT_NAMES.values()),
    }
    _write_json(out / "protocol.json", protocol)
    readme = [
        "# Pion known-graph coverage report with shared receipt",
        "",
        "**SAVED SYNTHETIC DIAGNOSTIC ONLY. `physical_score=null`. This report copies the verified v2 output; the 150,000-replicate Monte Carlo was not rerun.**",
        "",
        "The original report and its artifact manifest are preserved under `source_report/`. The overlay adds a versioned `ScenarioSpec` and `RunReceipt` for the existing scenario-cell, replicate, paired-block-event, and candidate-family-score denominators. It does not infer project dependence or anchor coverage.",
        "",
        "## Text data flow",
        "",
        "```text",
        "verified saved v2 report + nested SHA-256 manifest",
        "  -> read the four existing denominator rows (no calls to the simulation generator)",
        "  -> retain separate scenario cell / replicate / paired block event / family score units",
        "  -> attach exact failure-reason counts and source/output hashes",
        "  -> complete means all planned units are accounted for, not scientific success",
        "  -> physical_score remains null; project evidence and authorization gates remain open",
        "```",
        "",
        "## Denominator summary",
        "",
        "|unit|planned|valid|failed|abstained|not_run|interpretation|",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        not_run = row["planned"] - row["valid"] - row["failed"] - row["abstained"]
        readme.append(
            f"|{row['unit']}|{row['planned']}|{row['valid']}|{row['failed']}|"
            f"{row['abstained']}|{not_run}|{row['interpretation']}|"
        )
    readme.extend([
        "",
        "The paired block-event and family-score failures are synthetic Bernoulli failures that are retained in the planned denominator and coded as `-1`; they are not measurement failures. The Monte Carlo replicate row has zero all-block-failed abstentions. Reasons in `run_receipt.json` account exactly for the failed block-event and family-score counts.",
        "",
        "## Saved figures and interpretation",
        "",
        "![Coverage by dependence order](source_report/coverage_by_dependence.png)",
        "",
        "The saved v2 summary reports minimum familywise coverage 1.0000 under the constructed graph-color bound and 0.8362 under the deliberately misspecified IID penalty. These are finite synthetic Monte Carlo results only, not coverage evidence for LatticeArena data.",
        "",
        "![Positive-certificate rate](source_report/positive_certificate_rate.png)",
        "",
        "The original source report, all its raw replicate summaries, and its 10-entry manifest remain intact in `source_report/`. The outer `artifact_hashes.json` covers every copied source file, receipt payload and `run_receipt.json`; the receipt itself lists the hashes of all nonrecursive payloads.",
        "",
        "## Rebuild the receipt overlay",
        "",
        "This command copies and verifies the saved report; it does not rerun Monte Carlo:",
        "",
        "```text",
        "python -B -m tasks.pion_projection.bounded_block_risk_coverage_receipt --source results/pion_projection/bounded_block_risk_coverage_mc_20261123_v2 --output NEW_DIRECTORY",
        "```",
        "",
    ])
    (out / "README.md").write_text("\n".join(readme))
    summary = {
        "schema": "pion-known-graph-coverage-receipt-summary-v1",
        "status": "SAVED_SYNTHETIC_REPORT_RECEIPT_ONLY",
        "physical_score": None,
        "simulation_rerun": False,
        "source_report_manifest_sha256": source_manifest_hash,
        "denominators": denominators,
        "source_report_summary": {
            "cells": source_summary["cells"],
            "replicates_per_cell": source_summary["replicates_per_cell"],
            "graph_coverage_min": source_summary["graph_coverage_min"],
            "iid_coverage_min": source_summary["iid_coverage_min"],
        },
    }
    _write_json(out / "summary.json", summary)

    payload_hashes = {
        path.relative_to(out).as_posix(): _sha256(path)
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }
    receipt = run_receipt(
        scenario, denominators, artifact_hashes=payload_hashes,
        text_flow=(
            "verified saved synthetic coverage report -> separate existing analysis-unit "
            "denominators -> exact hypothetical-failure reasons -> hashed receipt overlay; "
            "no Monte Carlo rerun, and complete means accounting only, not scientific success"
        ),
        reason_rows=reason_rows,
    )
    _write_json(out / "run_receipt.json", receipt)
    output_hashes = {
        path.relative_to(out).as_posix(): _sha256(path)
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }
    _write_json(out / "artifact_hashes.json", {
        "algorithm": "sha256",
        "files": output_hashes,
    })
    return {
        "status": protocol["status"],
        "physical_score": None,
        "denominators": denominators,
        "receipt": receipt,
        "files": list(output_hashes) + ["artifact_hashes.json"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = build_report(args.output, source=args.source)
    print(json.dumps({
        "status": result["status"],
        "physical_score": result["physical_score"],
        "simulation_rerun": False,
        "denominator_units": list(result["denominators"]),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
