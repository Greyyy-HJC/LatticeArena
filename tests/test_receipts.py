import pytest

from core.receipts import denominator, run_receipt, scenario_spec


HASH = "a" * 64


def test_scenario_spec_requires_auditable_fields_and_hashes():
    spec = scenario_spec(
        "fixture-a", "one panel", "synthetic contract fixture",
        assumptions=["no physical data"], seed=7, source_hashes={"code.py": HASH},
    )
    assert spec["schema"] == "latticearena-scenario-v1"
    assert spec["source_hashes"] == {"code.py": HASH}
    with pytest.raises(ValueError, match="SHA-256"):
        scenario_spec("fixture-a", "panel", "desc", source_hashes={"code.py": "short"})
    with pytest.raises(ValueError, match="source_hashes"):
        scenario_spec("fixture-a", "panel", "desc", source_hashes=[])
    with pytest.raises(ValueError, match="seed"):
        scenario_spec("fixture-a", "panel", "desc", seed=True)


def test_denominator_exposes_not_run_and_rejects_impossible_counts():
    result = denominator("panels", planned=5, valid=2, failed=1, abstained=1)
    assert result == {
        "unit": "panels", "planned": 5, "valid": 2, "failed": 1,
        "abstained": 1, "not_run": 1,
    }
    with pytest.raises(ValueError, match="cannot exceed planned"):
        denominator("panels", planned=2, valid=2, failed=1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        denominator("panels", planned=2, valid=True)


def test_run_receipt_binds_denominators_and_artifacts():
    scenario = scenario_spec(
        "fixture-a", "one panel", "synthetic contract fixture",
        source_hashes={"code.py": HASH},
    )
    receipt = run_receipt(
        scenario,
        {"panels": denominator("panels", planned=3, valid=2, failed=1)},
        artifact_hashes={"README.md": HASH},
        text_flow="plan -> evaluate -> retain valid/failed/not-run counts",
        reason_rows=[{"unit": "panels", "state": "failed", "count": 1,
                      "reason": "solver did not converge"}],
    )
    assert receipt["schema"] == "latticearena-run-receipt-v1"
    assert receipt["status"] == "complete"
    assert receipt["denominators"]["panels"]["not_run"] == 0
    assert receipt["reason_rows"][0]["reason"] == "solver did not converge"
    with pytest.raises(ValueError, match="inconsistent not_run"):
        run_receipt(
            scenario,
            {"panels": {"unit": "panels", "planned": 2, "valid": 1,
                         "failed": 0, "abstained": 0, "not_run": 0}},
            artifact_hashes={}, text_flow="flow",
        )
    with pytest.raises(ValueError, match="not_run must be a nonnegative integer"):
        run_receipt(
            scenario,
            {"panels": {"unit": "panels", "planned": 1, "valid": 0,
                         "failed": 0, "abstained": 0, "not_run": True}},
            artifact_hashes={}, text_flow="flow",
            reason_rows=[{"unit": "panels", "state": "not_run", "count": 1,
                          "reason": "not started"}],
        )


def test_run_receipt_marks_unstarted_planned_units_partial():
    scenario = scenario_spec("fixture-a", "panel", "fixture")
    receipt = run_receipt(
        scenario,
        {"measurements": denominator("measurements", planned=4, valid=0)},
        artifact_hashes={}, text_flow="planned -> not run",
        reason_rows=[{"unit": "measurements", "state": "not_run", "count": 4,
                      "reason": "data access not authorized"}],
    )
    assert receipt["status"] == "partial"
    assert receipt["denominators"]["measurements"]["not_run"] == 4


def test_run_receipt_requires_reason_accounting_for_failures_and_abstentions():
    scenario = scenario_spec("fixture-a", "panel", "fixture")
    counts = {"panels": denominator("panels", planned=3, valid=1,
                                     failed=1, abstained=1)}
    with pytest.raises(ValueError, match="expected 1"):
        run_receipt(scenario, counts, artifact_hashes={}, text_flow="flow")
    receipt = run_receipt(
        scenario,
        counts,
        artifact_hashes={},
        text_flow="flow",
        reason_rows=[
            {"unit": "panels", "state": "failed", "count": 1, "reason": "fit error"},
            {"unit": "panels", "state": "abstained", "count": 1,
             "reason": "reference missing"},
        ],
    )
    assert len(receipt["reason_rows"]) == 2
