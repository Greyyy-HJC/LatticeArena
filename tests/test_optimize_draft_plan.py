import hashlib
import json

from tasks.optimize_gaussian.draft_plan import build_plan, build_report


def test_selected_f2_contrasts_are_bound_to_fail_closed_no_launch_plan():
    plan = build_plan()
    task = plan["task_spec"]
    run = plan["run_spec"]
    comparisons = [
        "trainable_W_vs_same_budget_SLSQP",
        "trainable_W_vs_fixed_degree4_Chebyshev",
    ]

    assert task["metric"]["comparison_family"] == comparisons
    assert run["task"]["metric"]["comparison_family"] == comparisons
    assert task["budget"]["methods"] == [
        "trainable_W", "same_budget_SLSQP", "fixed_degree4_Chebyshev",
    ]
    assert run["task"]["budget"]["objective_gradient_call_cap_per_arm"] == 160
    assert run["task"]["metric"]["D06"]["r_star"] == 10 / 9
    assert run["task"]["metric"]["D06"]["family_size"] == 2

    assert plan["status"] == "DRAFT_NO_LAUNCH"
    assert run["status"] == "DRAFT_NO_LAUNCH"
    assert plan["launch_allowed"] is False
    assert run["launch_allowed"] is False
    assert "risk_statistics.py" in plan["source_hashes"]
    assert "source_reader.py" in plan["source_hashes"]
    assert "core/npz_safety.py" in plan["source_hashes"]
    assert "core/source_integrity.py" in plan["source_hashes"]
    assert "core/numerics.py" in plan["source_hashes"]
    assert "bounded_block_risk_markov.py" in plan["source_hashes"]
    assert "paired_cost_accounting.py" in plan["source_hashes"]
    assert any("paired_cost_accounting.py" in item for item in plan["text_flow"])
    assert "trainable_w.py" in plan["source_hashes"]
    assert "features.py" in plan["source_hashes"]
    assert "d04_raw_reference.py" in plan["source_hashes"]
    assert "evidence_adapter.py" in plan["source_hashes"]
    assert "core/evidence.py" in plan["source_hashes"]
    assert "core/receipts.py" in plan["source_hashes"]
    assert "an evaluator that generates paired outer block-bootstrap and inner D04 draws" in plan["not_established"]
    assert "a complete comparable paired cost ledger with setup/training charged to the selected one-task batch" in plan["not_established"]
    assert "production D04/D11 summary generation from raw correlator/link inputs" in plan["not_established"]
    assert any("D04/D11 summary JSON" in item for item in plan["text_flow"])
    assert any("D04 raw E-chain snapshot reader" in item for item in plan["text_flow"])
    assert plan["physical_score"] is None
    assert run["physical_score"] is None
    assert run["data"] == {}
    assert run["authorization"] == {"production_data": False, "gpu_quda": False}
    assert plan["denominators"]["physical_gates"] == {
        "planned": 5, "valid": 0, "failed": 0, "abstained": 5,
        "unit": "operator/reference/numerical/statistical/comparison certification stages",
    }


def test_draft_report_copies_the_hashed_snapshot_reader_dependencies(tmp_path):
    output = tmp_path / "draft-report"
    result = build_report(output)
    plan = json.loads((output / "run_plan.json").read_text())
    artifacts = json.loads((output / "artifact_hashes.json").read_text())

    assert result["status"] == "DRAFT_NO_LAUNCH"
    assert "core/npz_safety.py" in plan["source_hashes"]
    assert "snapshot_core_numerics.py" in artifacts
    assert "snapshot_core_npz_safety.py" in artifacts
    assert "snapshot_core_source_integrity.py" in artifacts
    assert "snapshot_trainable_w.py" in artifacts
    assert "snapshot_d04_raw_reference.py" in artifacts
    assert "snapshot_features.py" in artifacts
    assert "snapshot_evidence_adapter.py" in artifacts
    assert "snapshot_core_evidence.py" in artifacts
    assert "snapshot_core_receipts.py" in artifacts
    assert "snapshot_paired_cost_accounting.py" in artifacts
    for source_name, expected in plan["source_hashes"].items():
        snapshot_name = f"snapshot_{source_name.replace('/', '_')}"
        assert snapshot_name in artifacts
        assert hashlib.sha256((output / snapshot_name).read_bytes()).hexdigest() == expected
    for name, expected in artifacts.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
