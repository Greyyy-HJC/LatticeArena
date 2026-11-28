import hashlib
import json

from tasks.pion_projection.bounded_block_risk_anchor_coverage_receipt import (
    SOURCE_REPORT,
    build_report,
)


def test_joint_score_anchor_receipt_uses_saved_report_without_running_mc(
    tmp_path, monkeypatch,
):
    from tasks.pion_projection import bounded_block_risk_anchor_coverage_mc as mc

    def unexpected_simulation(*args, **kwargs):
        raise AssertionError("receipt overlay must not rerun the Monte Carlo")

    monkeypatch.setattr(mc, "simulate_cell", unexpected_simulation)
    output = tmp_path / "anchor-coverage-receipt"
    result = build_report(output, source=SOURCE_REPORT)

    assert result["status"] == "SAVED_SYNTHETIC_REPORT_RECEIPT_ONLY"
    assert result["physical_score"] is None
    receipt = json.loads((output / "run_receipt.json").read_text())
    assert receipt["status"] == "complete"
    denominators = receipt["denominators"]
    assert len(denominators) == 81
    assert denominators["scenario_cells"] == {
        "unit": "synthetic joint-coverage scenario cell",
        "planned": 16, "valid": 16, "failed": 0, "abstained": 0,
        "not_run": 0,
    }

    expected_totals = {
        "mc_replicate": (80000, 80000, 0, 0),
        "paired_block_event": (25600000, 24321134, 1278866, 0),
        "family_block_score": (51200000, 48642268, 2557732, 0),
        "anchor_estimate": (80000, 80000, 0, 0),
        "anchor_observation": (6400000, 6400000, 0, 0),
    }
    for unit, expected in expected_totals.items():
        rows = [value for key, value in denominators.items()
                if key.startswith(unit + "::")]
        assert len(rows) == 16
        assert tuple(sum(row[name] for row in rows)
                     for name in ("planned", "valid", "failed", "abstained")) == expected
    assert sum(row["count"] for row in receipt["reason_rows"]
               if row["state"] == "failed") == 3836598
    assert "core/receipts.py" in receipt["scenario"]["source_hashes"]
    assert "core/numerics.py" in receipt["scenario"]["source_hashes"]
    helper_hash = receipt["scenario"]["source_hashes"]["core/numerics.py"]
    assert hashlib.sha256((output / "snapshot_core_numerics.py").read_bytes()).hexdigest() == helper_hash

    source_manifest = json.loads(
        (output / "source_report/artifact_hashes.json").read_text()
    )
    for name, expected in source_manifest.items():
        assert hashlib.sha256(
            (output / "source_report" / name).read_bytes()
        ).hexdigest() == expected
    manifest = json.loads((output / "artifact_hashes.json").read_text())
    assert len(receipt["artifact_hashes"]) == 22
    assert len(manifest["files"]) == 23
    for name, expected in receipt["artifact_hashes"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected

    report = (output / "README.md").read_text()
    assert "80,000 Monte Carlo replicates were not rerun" in report
    assert "retains all 80 cell-by-analysis-unit denominator rows" in report
    assert "![Joint score and anchor coverage](source_report/coverage_and_utility.png)" in report
    assert "do not establish project anchor support" in report
