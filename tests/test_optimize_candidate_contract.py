import hashlib
import json

import pytest

from tasks.optimize_gaussian.candidate_contract import (
    build_report,
    candidate_spec_for,
    open_gate_run_state,
    operator_contract_for,
)


def test_optimize_candidate_contract_is_registered_but_explicitly_diagnostic():
    operator = operator_contract_for()
    candidate = candidate_spec_for()

    assert operator["name"] == "optimize-chebyshev-diagnostic-v1"
    assert operator["constraints"]["physical_contract_frozen"] is False
    assert operator["constraints"]["selected_method_defaults_recorded"] is True
    assert operator["constraints"]["production_evaluator_connected"] is False
    assert candidate["builder"] == (
        "tasks.optimize_gaussian.chebyshev_baseline:apply_hopping_filter_cpu"
    )
    assert candidate["parameters"]["control_points"] == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_optimize_candidate_contract_rejects_invalid_controls():
    with pytest.raises(ValueError, match="nondecreasing"):
        candidate_spec_for([0.0, 0.6, 0.4, 0.8, 1.0])
    with pytest.raises(ValueError, match="fixed to 1"):
        candidate_spec_for([0.0, 0.25, 0.5, 0.75, 0.9])


def test_open_physical_gates_project_to_abstained_state_without_score():
    state = open_gate_run_state()

    assert state["physical_ready"] is False
    assert state["score"] is None
    assert set(state["status"].values()) == {"abstained"}
    assert all(state["evidence"][stage] for stage in state["status"])
    assert "E-chain" in state["evidence"]["reference"][0]
    assert "confirmation IDs" in state["evidence"]["comparison"][0]


def test_candidate_contract_report_is_hashed_and_has_text_flow(tmp_path):
    output = tmp_path / "candidate-contract"
    report = build_report(output)
    manifest = json.loads((output / "artifact_hashes.json").read_text())
    receipt = json.loads((output / "run_receipt.json").read_text())

    assert report["status"] == "DIAGNOSTIC_SCHEMA_ONLY"
    assert report["physical_score"] is None
    assert report["data_access"] == "none"
    assert report["run_state"]["physical_ready"] is False
    assert report["run_state"]["score"] is None
    assert set(report["run_state"]["status"].values()) == {"abstained"}
    assert report["denominators"]["coefficient_fit"]["planned"] == 0
    assert report["denominators"]["run_state_gates"] == {
        "planned": 5, "valid": 0, "failed": 0, "abstained": 5,
        "unit": "one physical certification stage in core.RunState",
    }
    assert "selected D11 method defaults are recorded" in (output / "README.md").read_text()
    assert receipt["schema"] == "latticearena-run-receipt-v1"
    assert receipt["status"] == "complete"
    assert receipt["denominators"]["schema_fixture"]["not_run"] == 0
    assert receipt["denominators"]["coefficient_fit"]["planned"] == 0
    assert receipt["denominators"]["run_state_gates"]["abstained"] == 5
    assert receipt["reason_rows"] == [{
        "unit": "run_state_gates", "state": "abstained", "count": 5,
        "reason": "physical certification awaits independent receipts and explicit authorization",
    }]
    assert "core/receipts.py" in receipt["scenario"]["source_hashes"]
    assert "core/numerics.py" in receipt["scenario"]["source_hashes"]
    assert "core/state.py" in receipt["scenario"]["source_hashes"]
    assert "research_contract_v1.json" in receipt["scenario"]["source_hashes"]
    for name, expected in receipt["artifact_hashes"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert set(manifest) == {
        "README.md", "contract.json", "snapshot_candidate_contract.py",
        "run_receipt.json", "snapshot_chebyshev_baseline.py",
        "snapshot_research_contract_v1.json",
        "snapshot_core_receipts.py", "snapshot_core_specs.py",
        "snapshot_core_state.py", "snapshot_core_numerics.py",
        "snapshot_kernels.py",
    }
    for name, expected in manifest.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
