import hashlib
import json

from tasks.pion_projection.bounded_block_risk_markov_varying_receipt import (
    SOURCE_REPORT,
    build_report,
)


def test_saved_markov_varying_receipt_does_not_rerun_diagnostic(tmp_path, monkeypatch):
    from tasks.pion_projection import bounded_block_risk_markov_varying_diagnostic as diagnostic

    def unexpected_rerun(*args, **kwargs):
        raise AssertionError("receipt overlay must not rerun the formula diagnostic")

    monkeypatch.setattr(diagnostic, "build_report", unexpected_rerun)
    output = tmp_path / "markov-varying-receipt"
    result = build_report(output, source=SOURCE_REPORT)

    assert result["status"] == "SAVED_SYNTHETIC_REPORT_RECEIPT_ONLY"
    assert result["physical_score"] is None
    receipt = json.loads((output / "run_receipt.json").read_text())
    assert receipt["status"] == "complete"
    expected = {
        "analytic_grid_cells": (72, 5, 0, 67),
        "tv_sensitivity_cells": (6, 0, 0, 6),
        "partial_block_slots": (20, 18, 2, 0),
        "partial_comparison_result": (1, 1, 0, 0),
        "all_failed_block_slots": (4, 0, 4, 0),
        "all_failed_comparison_result": (1, 0, 0, 1),
    }
    assert set(receipt["denominators"]) == set(expected)
    for name, counts in expected.items():
        row = receipt["denominators"][name]
        assert tuple(row[key] for key in ("planned", "valid", "failed", "abstained")) == counts
    reason_totals = {}
    for row in receipt["reason_rows"]:
        key = (row["unit"], row["state"])
        reason_totals[key] = reason_totals.get(key, 0) + row["count"]
    assert reason_totals[("analytic_grid_cells", "abstained")] == 67
    assert reason_totals[("tv_sensitivity_cells", "abstained")] == 6
    assert reason_totals[("partial_block_slots", "failed")] == 2
    assert reason_totals[("all_failed_block_slots", "failed")] == 4
    assert reason_totals[("all_failed_comparison_result", "abstained")] == 1

    source_manifest = json.loads((output / "source_report/artifact_hashes.json").read_text())
    for name, expected_hash in source_manifest["files"].items():
        assert hashlib.sha256((output / "source_report" / name).read_bytes()).hexdigest() == expected_hash
    manifest = json.loads((output / "artifact_hashes.json").read_text())
    assert len(receipt["artifact_hashes"]) == 20
    assert len(manifest["files"]) == 21
    for name, expected_hash in receipt["artifact_hashes"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected_hash
    for name, expected_hash in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected_hash

    readme = (output / "README.md").read_text()
    assert "![Eq. (3.26) radius and M/k_ps sensitivity](source_report/markov_varying_sensitivity.png)" in readme
    assert "complete means accounting only" in readme
    assert "physical_score remains null" in readme
