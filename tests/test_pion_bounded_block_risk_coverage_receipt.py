import hashlib
import json

from tasks.pion_projection.bounded_block_risk_coverage_receipt import (
    SOURCE_REPORT,
    build_report,
)


def test_receipt_overlay_reuses_saved_report_without_running_mc(tmp_path, monkeypatch):
    from tasks.pion_projection import bounded_block_risk_coverage_mc as mc

    def unexpected_simulation(*args, **kwargs):
        raise AssertionError("receipt overlay must not rerun the Monte Carlo")

    monkeypatch.setattr(mc, "simulate_cell", unexpected_simulation)
    output = tmp_path / "coverage-receipt"
    result = build_report(output, source=SOURCE_REPORT)

    assert result["status"] == "SAVED_SYNTHETIC_REPORT_RECEIPT_ONLY"
    assert result["physical_score"] is None
    assert result["receipt"]["status"] == "complete"
    assert result["receipt"]["scenario"]["seed"] is None
    assert result["receipt"]["scenario"]["assumptions"]

    denominators = result["receipt"]["denominators"]
    assert denominators == {
        "scenario_cell": {
            "unit": "synthetic Monte Carlo cell",
            "planned": 30, "valid": 30, "failed": 0, "abstained": 0,
            "not_run": 0,
        },
        "monte_carlo_replicate": {
            "unit": "Monte Carlo replicate",
            "planned": 150000, "valid": 150000, "failed": 0, "abstained": 0,
            "not_run": 0,
        },
        "paired_planned_block_event": {
            "unit": "paired planned block event",
            "planned": 33600000, "valid": 31920007,
            "failed": 1679993, "abstained": 0, "not_run": 0,
        },
        "candidate_family_block_score": {
            "unit": "candidate-family block score",
            "planned": 67200000, "valid": 63840014,
            "failed": 3359986, "abstained": 0, "not_run": 0,
        },
    }
    assert sum(row["count"] for row in result["receipt"]["reason_rows"]
               if row["state"] == "failed") == 5039979

    source_manifest = json.loads(
        (output / "source_report/artifact_hashes.json").read_text()
    )
    for name, expected in source_manifest.items():
        assert hashlib.sha256(
            (output / "source_report" / name).read_bytes()
        ).hexdigest() == expected

    receipt = json.loads((output / "run_receipt.json").read_text())
    manifest = json.loads((output / "artifact_hashes.json").read_text())
    assert len(receipt["artifact_hashes"]) == 18
    assert len(manifest["files"]) == 19
    for name, expected in receipt["artifact_hashes"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected

    report = (output / "README.md").read_text()
    assert "150,000-replicate Monte Carlo was not rerun" in report
    assert "scenario cell / replicate / paired block event / family score" in report
    assert "![Coverage by dependence order](source_report/coverage_by_dependence.png)" in report
    assert "complete means all planned units are accounted for, not scientific success" in report
