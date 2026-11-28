from __future__ import annotations

import copy

from tasks.optimize_gaussian.paired_cost_accounting import (
    ARMS,
    PHASES,
    SCHEMA,
    summarize_paired_cost_ledger,
)


def _phase(**values):
    return {phase: float(values.get(phase, 0.0)) for phase in PHASES}


def _block(block_id, status, *, solve=0.0, inference=0.0, recovery=0.0,
           objective=0, gradient=0):
    return {
        "block_id": block_id,
        "status": status,
        "phase_seconds": _phase(solve=solve, inference=inference,
                                failure_recovery=recovery),
        "calls": {"objective": objective, "gradient": gradient},
    }


def _ledger():
    environment = {
        "device": "cpu-model-A",
        "precision": "float64",
        "thread_count": 1,
        "software_versions": "python=3.x; scipy=1.x",
        "clock_policy": "fixed-performance-mode",
    }
    arms = {
        "trainable_W": {
            "arm_protocol_sha256": "a" * 64,
            "environment": dict(environment),
            "shared_phase_seconds": _phase(setup=0.1, training=0.4),
            "shared_calls": {"objective": 2, "gradient": 1},
            "blocks": [
                _block("mb-01", "valid", solve=0.2, inference=0.05,
                       objective=4, gradient=4),
                _block("mb-02", "failed", solve=0.3, inference=0.02,
                       recovery=0.1, objective=1),
            ],
        },
        "same_budget_SLSQP": {
            "arm_protocol_sha256": "b" * 64,
            "environment": dict(environment),
            "shared_phase_seconds": _phase(setup=0.05),
            "shared_calls": {"objective": 0, "gradient": 0},
            "blocks": [
                _block("mb-01", "valid", solve=0.1, inference=0.03,
                       objective=3, gradient=2),
                _block("mb-02", "valid", solve=0.2, inference=0.03,
                       objective=4, gradient=3),
            ],
        },
        "fixed_degree4_Chebyshev": {
            "arm_protocol_sha256": "c" * 64,
            "environment": dict(environment),
            "shared_phase_seconds": _phase(setup=0.04),
            "shared_calls": {"objective": 0, "gradient": 0},
            "blocks": [
                _block("mb-01", "valid", inference=0.05),
                _block("mb-02", "valid", inference=0.05),
            ],
        },
    }
    return {
        "schema": SCHEMA,
        "planned_block_ids": ["mb-01", "mb-02"],
        "input_manifest_sha256": "1" * 64,
        "seed_plan_sha256": "2" * 64,
        "arms": arms,
    }


def test_complete_ledger_charges_shared_training_and_keeps_failed_block_cost():
    result = summarize_paired_cost_ledger(_ledger())

    assert result["status"] == "COST_LEDGER_COMPLETE"
    assert result["comparison_ready"] is True
    assert result["certified"] is False
    assert result["physical_score"] is None
    learned = result["arms"]["trainable_W"]
    assert learned["denominator"] == {
        "planned": 2, "valid": 1, "failed": 1, "abstained": 0,
        "not_run": 0, "unit": "paired_macroblock",
    }
    assert learned["shared_phase_seconds"]["training"] == 0.4
    assert learned["full_batch_phase_seconds"]["failure_recovery"] == 0.1
    assert learned["full_task_batch_seconds"] == 1.17
    assert learned["objective_calls_total"] == 7
    assert learned["gradient_calls_total"] == 5
    assert learned["max_per_block_calls"] == 8
    assert result["cost_ratios_to_trainable_W"]["same_budget_SLSQP"] < 1.0


def test_missing_planned_block_abstains_and_emits_no_cost_ratio():
    ledger = _ledger()
    ledger["arms"]["fixed_degree4_Chebyshev"]["blocks"].pop()

    result = summarize_paired_cost_ledger(ledger)

    assert result["status"] == "ABSTAIN_COST_LEDGER_INCOMPLETE"
    assert result["comparison_ready"] is False
    assert result["denominators"]["fixed_degree4_Chebyshev"]["not_run"] == 1
    assert result["cost_ratios_to_trainable_W"] == {
        "same_budget_SLSQP": None, "fixed_degree4_Chebyshev": None,
    }


def test_environment_mismatch_abstains_even_when_all_cost_rows_are_present():
    ledger = _ledger()
    ledger["arms"]["same_budget_SLSQP"]["environment"]["precision"] = "float32"

    result = summarize_paired_cost_ledger(ledger)

    assert result["status"] == "ABSTAIN_COST_ENVIRONMENT_MISMATCH"
    assert result["environment_comparable"] is False
    assert all(value is None for value in result["cost_ratios_to_trainable_W"].values())


def test_invalid_costs_calls_and_duplicate_block_ids_fail_closed():
    cases = []
    negative = _ledger()
    negative["arms"]["trainable_W"]["blocks"][0]["phase_seconds"]["solve"] = -0.01
    cases.append(negative)
    boolean_call = _ledger()
    boolean_call["arms"]["trainable_W"]["blocks"][0]["calls"]["gradient"] = True
    cases.append(boolean_call)
    duplicate = _ledger()
    duplicate["arms"]["trainable_W"]["blocks"][1]["block_id"] = "mb-01"
    cases.append(duplicate)

    for ledger in cases:
        result = summarize_paired_cost_ledger(copy.deepcopy(ledger))
        assert result["status"] == "ABSTAIN_COST_LEDGER_INVALID"
        assert result["comparison_ready"] is False
        assert result["physical_score"] is None


def test_finite_per_row_costs_that_overflow_batch_total_abstain():
    ledger = _ledger()
    rows = ledger["arms"]["trainable_W"]["blocks"]
    rows[0]["phase_seconds"]["solve"] = 1e308
    rows[1]["phase_seconds"]["solve"] = 1e308

    result = summarize_paired_cost_ledger(ledger)

    assert result["status"] == "ABSTAIN_COST_LEDGER_INVALID"
    assert result["comparison_ready"] is False
    assert all(value is None for value in result["cost_ratios_to_trainable_W"].values())
    assert any("aggregate seconds overflow" in reason for reason in result["reasons"])


def test_overflowing_cost_ratio_abstains_instead_of_returning_infinity():
    ledger = _ledger()
    learned = ledger["arms"]["trainable_W"]
    learned["shared_phase_seconds"] = _phase(setup=1e-308)
    for row in learned["blocks"]:
        row["phase_seconds"] = _phase()
    comparison = ledger["arms"]["same_budget_SLSQP"]
    comparison["shared_phase_seconds"] = _phase(setup=1e308)
    for row in comparison["blocks"]:
        row["phase_seconds"] = _phase()
    chebyshev = ledger["arms"]["fixed_degree4_Chebyshev"]
    chebyshev["shared_phase_seconds"] = _phase()
    for row in chebyshev["blocks"]:
        row["phase_seconds"] = _phase()

    result = summarize_paired_cost_ledger(ledger)

    assert result["status"] == "ABSTAIN_COST_RATIO_UNDEFINED"
    assert result["comparison_ready"] is False
    assert result["cost_ratios_to_trainable_W"]["same_budget_SLSQP"] is None
    assert result["cost_ratios_to_trainable_W"]["fixed_degree4_Chebyshev"] == 0.0
    assert any("ratio is not finite" in reason for reason in result["reasons"])
