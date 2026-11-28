"""Attach shared receipts to the saved synthetic score/anchor coverage report.

This is a read-only report wrapper: it validates and copies saved artifacts but
does not call the Monte Carlo generator or access physical/sealed data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from core.receipts import run_receipt, scenario_spec


ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPORT = ROOT / "results/pion_projection/bounded_block_risk_anchor_coverage_mc_20261123_v1"
DEFAULT_OUT = ROOT / "results/pion_projection/bounded_block_risk_anchor_coverage_receipt_20261124_v1"
UNIT_NAMES = {
    "MC replicate": "mc_replicate",
    "paired block event": "paired_block_event",
    "family block score": "family_block_score",
    "anchor estimate": "anchor_estimate",
    "anchor observation": "anchor_observation",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _read_source(source: Path):
    manifest = json.loads((source / "artifact_hashes.json").read_text())
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("source artifact manifest must be a nonempty mapping")
    for name, expected in manifest.items():
        path = source / name
        if not isinstance(name, str) or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"source report artifact failed SHA-256 check: {name}")
    with (source / "denominator_summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    cells = sorted({row["cell"] for row in rows})
    if len(cells) != 16 or len(rows) != len(cells) * len(UNIT_NAMES):
        raise ValueError("expected 16 scenario cells with five denominator units each")
    normalized = []
    for row in rows:
        if row.get("unit") not in UNIT_NAMES:
            raise ValueError(f"unexpected source denominator unit: {row.get('unit')!r}")
        counts = {key: int(row[key]) for key in
                  ("planned", "valid", "failed", "abstained")}
        if counts["planned"] != counts["valid"] + counts["failed"] + counts["abstained"]:
            raise ValueError(f"source denominator does not close: {row['cell']}/{row['unit']}")
        normalized.append({**row, **counts})
    expected_pairs = {(cell, unit) for cell in cells for unit in UNIT_NAMES}
    observed_pairs = {(row["cell"], row["unit"]) for row in normalized}
    if len(observed_pairs) != len(normalized) or observed_pairs != expected_pairs:
        raise ValueError("source denominator rows must contain each cell/unit pair exactly once")
    return manifest, cells, normalized


def build_report(output: str | Path = DEFAULT_OUT, *,
                 source: str | Path = SOURCE_REPORT) -> dict[str, Any]:
    source_path = Path(source).resolve()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    source_manifest, cells, rows = _read_source(source_path)
    source_summary = json.loads((source_path / "summary.json").read_text())
    if source_summary.get("physical_score") is not None:
        raise ValueError("source report must keep physical_score null")
    shutil.copytree(source_path, out / "source_report")

    snapshots = {
        "snapshot_anchor_coverage_mc.py": Path(__file__).with_name(
            "bounded_block_risk_anchor_coverage_mc.py"),
        "snapshot_bounded_block_risk_score.py": Path(__file__).with_name(
            "bounded_block_risk_score.py"),
        "snapshot_bounded_block_risk_coverage_mc.py": Path(__file__).with_name(
            "bounded_block_risk_coverage_mc.py"),
        "snapshot_dependency_graph_bound.py": Path(__file__).with_name(
            "dependency_graph_bound.py"),
        "snapshot_core_receipts.py": ROOT / "core/receipts.py",
        "snapshot_core_numerics.py": ROOT / "core/numerics.py",
        "snapshot_coverage_receipt_wrapper.py": Path(__file__),
    }
    for name, source_file in snapshots.items():
        (out / name).write_bytes(source_file.read_bytes())
    source_hashes = {
        "tasks/pion_projection/bounded_block_risk_anchor_coverage_mc.py": _sha256(
            out / "snapshot_anchor_coverage_mc.py"),
        "tasks/pion_projection/bounded_block_risk_score.py": _sha256(
            out / "snapshot_bounded_block_risk_score.py"),
        "tasks/pion_projection/bounded_block_risk_coverage_mc.py": _sha256(
            out / "snapshot_bounded_block_risk_coverage_mc.py"),
        "tasks/pion_projection/dependency_graph_bound.py": _sha256(
            out / "snapshot_dependency_graph_bound.py"),
        "core/receipts.py": _sha256(out / "snapshot_core_receipts.py"),
        "core/numerics.py": _sha256(out / "snapshot_core_numerics.py"),
        "tasks/pion_projection/bounded_block_risk_anchor_coverage_receipt.py": _sha256(
            out / "snapshot_coverage_receipt_wrapper.py"),
    }

    denominators = {
        "scenario_cells": {
            "unit": "synthetic joint-coverage scenario cell",
            "planned": len(cells), "valid": len(cells), "failed": 0,
            "abstained": 0,
        }
    }
    reason_rows = []
    totals = defaultdict(lambda: {key: 0 for key in
                                  ("planned", "valid", "failed", "abstained")})
    for row in rows:
        unit = UNIT_NAMES[row["unit"]]
        name = f"{unit}::{row['cell']}"
        denominators[name] = {
            "unit": row["unit"],
            "planned": row["planned"], "valid": row["valid"],
            "failed": row["failed"], "abstained": row["abstained"],
        }
        for key in totals[unit]:
            totals[unit][key] += row[key]
        if row["failed"]:
            reason_rows.append({
                "unit": name,
                "state": "failed",
                "count": row["failed"],
                "reason": row["interpretation"],
            })

    source_manifest_hash = _sha256(source_path / "artifact_hashes.json")
    protocol = {
        "schema": "pion-joint-score-anchor-coverage-receipt-v1",
        "status": "SAVED_SYNTHETIC_REPORT_RECEIPT_ONLY",
        "physical_score": None,
        "data_access": "verified saved synthetic report only; no simulation rerun",
        "source_report": "source_report/",
        "source_report_manifest_sha256": source_manifest_hash,
        "source_report_manifest_entries": len(source_manifest),
        "scenario_cell_count": len(cells),
        "cell_unit_rows": len(rows),
        "receipt_schema": "latticearena-run-receipt-v1",
        "receipt_status_meaning": "all planned units are accounted for; not scientific success",
    }
    _write_json(out / "protocol.json", protocol)

    total_rows = [
        ("MC replicate", totals["mc_replicate"]),
        ("paired block event", totals["paired_block_event"]),
        ("family block score", totals["family_block_score"]),
        ("anchor estimate", totals["anchor_estimate"]),
        ("anchor observation", totals["anchor_observation"]),
    ]
    readme = [
        "# Pion joint score/anchor coverage with shared receipt",
        "",
        "**SAVED SYNTHETIC DIAGNOSTIC ONLY. `physical_score=null`. This report copies the verified v1 artifacts; 80,000 Monte Carlo replicates were not rerun.**",
        "",
        "The original report and its manifest are preserved under `source_report/`. The overlay records 16 scenario cells and retains all 80 cell-by-analysis-unit denominator rows; it does not combine score blocks, anchor observations, or comparisons into one sample count.",
        "",
        "## Text data flow",
        "",
        "```text",
        "verified saved joint score/anchor report + nested SHA-256 manifest",
        "  -> 16 prespecified scenario cells -> 80 cell-by-unit denominator rows",
        "  -> separate replicate / paired block / family score / anchor estimate / anchor observation counts",
        "  -> reason rows for each hypothetical failed score event -> hashed receipt",
        "  -> complete means all planned units are accounted for, not scientific success",
        "  -> physical_score remains null; dependence/anchor/reference and authorization gates remain open",
        "```",
        "",
        "## Aggregate denominator summary",
        "",
        "|unit|planned|valid|failed|abstained|",
        "|---|---:|---:|---:|---:|",
    ]
    for unit, counts in total_rows:
        readme.append(
            f"|{unit}|{counts['planned']}|{counts['valid']}|{counts['failed']}|{counts['abstained']}|"
        )
    readme.extend([
        "",
        "The aggregate table is a convenience summary; `run_receipt.json` retains one entry for every cell/unit pair. Paired block and family-score failures are synthetic Bernoulli events retained in planned `N` and coded `-1`, not observed measurement failures.",
        "",
        "## Saved figure and interpretation",
        "",
        "![Joint score and anchor coverage](source_report/coverage_and_utility.png)",
        "",
        f"The verified source summary reports graph-color false-certificate rate max `{source_summary['graph_false_certificate_rate_max']:.4g}` with Monte Carlo upper limit `{source_summary['graph_false_certificate_mc95_upper_max']:.6g}`; the deliberately misspecified IID route reaches `{source_summary['iid_false_certificate_rate_max']:.3g}`. Anchor interval coverage ranges from `{source_summary['anchor_interval_coverage_min']:.4f}` to `{source_summary['anchor_interval_coverage_max']:.4f}`. These results are conditional on the synthetic generator and known iid anchor model; they do not establish project anchor support, independence, common-bias coverage, or Pion dependence.",
        "",
        "The source report's `source_report/artifact_hashes.json` verifies its 11 payloads. The outer manifest covers all copied source files, receipt payloads, and `run_receipt.json`; the receipt binds every nonrecursive payload hash.",
        "",
        "## Rebuild the receipt overlay",
        "",
        "This command copies and verifies the saved report; it does not rerun Monte Carlo:",
        "",
        "```text",
        "python -B -m tasks.pion_projection.bounded_block_risk_anchor_coverage_receipt --source results/pion_projection/bounded_block_risk_anchor_coverage_mc_20261123_v1 --output NEW_DIRECTORY",
        "```",
        "",
    ])
    (out / "README.md").write_text("\n".join(readme))
    summary = {
        "schema": "pion-joint-score-anchor-coverage-receipt-summary-v1",
        "status": "SAVED_SYNTHETIC_REPORT_RECEIPT_ONLY",
        "physical_score": None,
        "simulation_rerun": False,
        "source_report_manifest_sha256": source_manifest_hash,
        "scenario_cells": len(cells),
        "cell_unit_rows": len(rows),
        "aggregate_denominators": dict(totals),
        "source_report_summary": {
            name: source_summary[name]
            for name in (
                "planned_replicates", "graph_false_certificate_rate_max",
                "graph_false_certificate_mc95_upper_max", "iid_false_certificate_rate_max",
                "anchor_interval_coverage_min", "anchor_interval_coverage_max",
            )
        },
    }
    _write_json(out / "summary.json", summary)

    payload_hashes = {
        path.relative_to(out).as_posix(): _sha256(path)
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }
    receipt = run_receipt(
        scenario_spec(
            "pion-joint-score-anchor-saved-coverage-report",
            "scenario cell, cell-by-unit count, and analysis unit",
            "Receipt overlay for a verified synthetic score/anchor report; no simulations rerun.",
            assumptions=[
                "the graph and failure process are known only by synthetic construction",
                "the anchor observations follow the report's independent known-sigma synthetic model",
                "receipt hashes do not establish project anchor independence or common-bias coverage",
            ],
            source_hashes=source_hashes,
        ),
        denominators,
        artifact_hashes=payload_hashes,
        text_flow=(
            "verified saved joint score/anchor report -> 16 scenario cells and 80 cell-unit "
            "rows -> separate score, block, replicate, and anchor counts -> exact synthetic "
            "failure reasons -> complete accounting only, not scientific success"
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
        "scenario_cells": 16,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
