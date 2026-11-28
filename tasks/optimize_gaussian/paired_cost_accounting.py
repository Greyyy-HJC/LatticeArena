"""Fail-closed cost accounting for paired Optimize arm receipts.

The reducer keeps shared setup/training costs separate from per-block work and
charges the complete shared cost to this one task batch. It validates ledger
shape and comparable execution metadata; it does not authenticate receipts or
turn cost comparability into a physics result.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from numbers import Integral
from typing import Any

from core.numerics import as_finite_real


SCHEMA = "optimize-paired-cost-ledger-v1"
ARMS = ("trainable_W", "same_budget_SLSQP", "fixed_degree4_Chebyshev")
PHASES = ("setup", "training", "solve", "inference", "io", "failure_recovery")
STATUSES = ("valid", "failed", "abstained")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENVIRONMENT_FIELDS = ("device", "precision", "thread_count", "software_versions", "clock_policy")


def _invalid(reasons: list[str], denominators: dict[str, dict[str, int]] | None = None) -> dict[str, Any]:
    return {
        "schema": "optimize-paired-cost-summary-v1",
        "status": "ABSTAIN_COST_LEDGER_INVALID",
        "comparison_ready": False,
        "certified": False,
        "physical_score": None,
        "reasons": reasons,
        "denominators": denominators or {},
        "arms": {},
        "cost_ratios_to_trainable_W": {},
        "text_flow": (
            "paired input/seed hashes + three arm receipts -> strict schema, block-ID, "
            "cost and environment checks -> complete P/V/F/A ledger -> total task-batch "
            "costs including setup/training -> ratios only if complete and comparable"
        ),
    }


def _is_nonnegative_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0


def _phase_map(value: object, *, label: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(PHASES):
        raise ValueError(f"{label} must contain exactly {list(PHASES)}")
    normalized: dict[str, float] = {}
    for phase in PHASES:
        seconds = as_finite_real(value[phase], name=f"{label}.{phase}")
        if seconds < 0.0:
            raise ValueError(f"{label}.{phase} must be nonnegative seconds")
        normalized[phase] = seconds
    return normalized


def _calls(value: object, *, label: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != {"objective", "gradient"}:
        raise ValueError(f"{label} must contain exactly objective and gradient counts")
    if any(not _is_nonnegative_integer(value[name]) for name in ("objective", "gradient")):
        raise ValueError(f"{label} counts must be nonnegative integers")
    return {name: int(value[name]) for name in ("objective", "gradient")}


def _add_finite_seconds(total: float, increment: float, *, label: str) -> float:
    """Add individually finite durations without allowing aggregate overflow."""
    try:
        result = math.fsum((total, increment))
    except OverflowError as exc:
        raise ValueError(f"{label} aggregate seconds overflow float64") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} aggregate seconds are non-finite")
    return result


def _environment(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(_ENVIRONMENT_FIELDS):
        raise ValueError(f"{label} must contain exactly {list(_ENVIRONMENT_FIELDS)}")
    result = dict(value)
    for name in ("device", "precision", "software_versions", "clock_policy"):
        if not isinstance(result[name], str) or not result[name].strip():
            raise ValueError(f"{label}.{name} must be a nonempty string")
    if not isinstance(result["thread_count"], int) or isinstance(result["thread_count"], bool) or result["thread_count"] < 1:
        raise ValueError(f"{label}.thread_count must be a positive integer")
    return result


def summarize_paired_cost_ledger(ledger: object) -> dict[str, Any]:
    """Validate and summarize a paired three-arm task-batch cost ledger.

    Required root fields are ``planned_block_ids``, shared input/seed hashes,
    and one receipt per selected arm. Each arm separates one-time shared costs
    from per-block costs. Missing/duplicate blocks, incomplete phase fields,
    invalid counts, or non-comparable environments fail closed. Solver failures
    and abstentions remain in the planned denominator and their observed costs
    are still charged.
    """
    if not isinstance(ledger, dict):
        return _invalid(["ledger must be an object"])
    reasons: list[str] = []
    if ledger.get("schema") != SCHEMA:
        reasons.append(f"schema must equal {SCHEMA!r}")
    block_ids = ledger.get("planned_block_ids")
    if (not isinstance(block_ids, list) or not block_ids
            or any(not isinstance(value, str) or not value.strip() for value in block_ids)
            or len(set(block_ids)) != len(block_ids)):
        reasons.append("planned_block_ids must be a nonempty list of unique nonempty strings")
        planned_ids: list[str] = []
    else:
        planned_ids = list(block_ids)
    for name in ("input_manifest_sha256", "seed_plan_sha256"):
        value = ledger.get(name)
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            reasons.append(f"{name} must be a lowercase SHA-256 identifier")
    arm_values = ledger.get("arms")
    if not isinstance(arm_values, dict) or set(arm_values) != set(ARMS):
        reasons.append(f"arms must contain exactly {list(ARMS)}")
        arm_values = {}
    if reasons:
        return _invalid(reasons)

    summaries: dict[str, dict[str, Any]] = {}
    environments: list[dict[str, Any]] = []
    invalid = False
    incomplete = False
    for arm in ARMS:
        source = arm_values[arm]
        try:
            if not isinstance(source, dict):
                raise ValueError("arm receipt must be an object")
            protocol_hash = source.get("arm_protocol_sha256")
            if not isinstance(protocol_hash, str) or not _SHA256.fullmatch(protocol_hash):
                raise ValueError("arm_protocol_sha256 must be a lowercase SHA-256 identifier")
            environment = _environment(source.get("environment"), label=f"{arm}.environment")
            environments.append(environment)
            shared_seconds = _phase_map(source.get("shared_phase_seconds"), label=f"{arm}.shared_phase_seconds")
            shared_calls = _calls(source.get("shared_calls"), label=f"{arm}.shared_calls")
            rows = source.get("blocks")
            if not isinstance(rows, list):
                raise ValueError("blocks must be a list")
            seen: set[str] = set()
            counts: Counter[str] = Counter()
            phase_totals = dict(shared_seconds)
            block_phase_totals = {phase: 0.0 for phase in PHASES}
            calls_total = dict(shared_calls)
            max_block_calls = 0
            reasons_for_arm: list[str] = []
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("each block receipt must be an object")
                block_id = row.get("block_id")
                if not isinstance(block_id, str) or block_id not in planned_ids:
                    raise ValueError("each block_id must occur in planned_block_ids")
                if block_id in seen:
                    raise ValueError(f"duplicate block_id {block_id!r}")
                seen.add(block_id)
                status = row.get("status")
                if status not in STATUSES:
                    raise ValueError(f"block {block_id!r} status must be one of {list(STATUSES)}")
                counts[status] += 1
                phase_seconds = _phase_map(row.get("phase_seconds"), label=f"{arm}.{block_id}.phase_seconds")
                block_calls = _calls(row.get("calls"), label=f"{arm}.{block_id}.calls")
                for phase in PHASES:
                    block_phase_totals[phase] = _add_finite_seconds(
                        block_phase_totals[phase], phase_seconds[phase],
                        label=f"{arm}.{phase} per-block",
                    )
                    phase_totals[phase] = _add_finite_seconds(
                        phase_totals[phase], phase_seconds[phase],
                        label=f"{arm}.{phase} full-batch",
                    )
                for kind in ("objective", "gradient"):
                    calls_total[kind] += block_calls[kind]
                max_block_calls = max(max_block_calls, block_calls["objective"] + block_calls["gradient"])
            missing_ids = [value for value in planned_ids if value not in seen]
            extra_count = len(rows) - len(seen)
            if missing_ids:
                reasons_for_arm.append(f"missing {len(missing_ids)} planned block receipts")
            if extra_count:
                reasons_for_arm.append("block list contains repeated rows")
            planned = len(planned_ids)
            completed = counts["valid"] + counts["failed"] + counts["abstained"]
            try:
                total_seconds = math.fsum(phase_totals.values())
            except OverflowError as exc:
                raise ValueError(f"{arm} full task-batch seconds overflow float64") from exc
            if not math.isfinite(total_seconds):
                raise ValueError(f"{arm} full task-batch seconds are non-finite")
            block_means = {
                phase: (block_phase_totals[phase] / planned
                        if not reasons_for_arm and completed == planned else None)
                for phase in PHASES
            }
            summaries[arm] = {
                "arm_protocol_sha256": protocol_hash,
                "denominator": {
                    "planned": planned,
                    "valid": counts["valid"],
                    "failed": counts["failed"],
                    "abstained": counts["abstained"],
                    "not_run": planned - completed,
                    "unit": "paired_macroblock",
                },
                "ledger_complete": not reasons_for_arm and completed == planned,
                "reasons": reasons_for_arm,
                "shared_phase_seconds": shared_seconds,
                "per_block_phase_seconds_total": block_phase_totals,
                "full_batch_phase_seconds": phase_totals,
                "full_task_batch_seconds": total_seconds,
                "per_planned_block_seconds_excluding_shared": block_means,
                "objective_calls_total": calls_total["objective"],
                "gradient_calls_total": calls_total["gradient"],
                "max_per_block_calls": max_block_calls,
            }
            if reasons_for_arm or completed != planned:
                incomplete = True
        except (ValueError, TypeError, OverflowError) as exc:
            reasons.append(f"{arm}: {exc}")
            invalid = True

    # Environment metadata must be identical before wall-time ratios are compared.
    same_environment = len(environments) == len(ARMS) and all(
        environment == environments[0] for environment in environments[1:]
    )
    if not same_environment:
        reasons.append("execution environment differs across arms")

    complete = not invalid and not incomplete and all(
        summaries.get(arm, {}).get("ledger_complete", False) for arm in ARMS
    )
    ratios: dict[str, float | None] = {arm: None for arm in ARMS[1:]}
    if complete and same_environment:
        denominator_cost = summaries["trainable_W"]["full_task_batch_seconds"]
        if denominator_cost > 0.0:
            for arm in ARMS[1:]:
                ratio = summaries[arm]["full_task_batch_seconds"] / denominator_cost
                if math.isfinite(ratio):
                    ratios[arm] = ratio
                else:
                    reasons.append(f"{arm} cost ratio is not finite")
        else:
            reasons.append("trainable_W full task-batch wall time must be positive to form cost ratios")

    if invalid:
        status = "ABSTAIN_COST_LEDGER_INVALID"
    elif not complete:
        status = "ABSTAIN_COST_LEDGER_INCOMPLETE"
    elif not same_environment:
        status = "ABSTAIN_COST_ENVIRONMENT_MISMATCH"
    elif any(value is None for value in ratios.values()):
        status = "ABSTAIN_COST_RATIO_UNDEFINED"
    else:
        status = "COST_LEDGER_COMPLETE"

    return {
        "schema": "optimize-paired-cost-summary-v1",
        "status": status,
        "comparison_ready": status == "COST_LEDGER_COMPLETE",
        "certified": False,
        "physical_score": None,
        "input_manifest_sha256": ledger["input_manifest_sha256"],
        "seed_plan_sha256": ledger["seed_plan_sha256"],
        "environment_comparable": same_environment,
        "denominators": {arm: summary["denominator"] for arm, summary in summaries.items()},
        "arms": summaries,
        "cost_ratios_to_trainable_W": ratios,
        "reasons": reasons,
        "hashes_are_syntactically_checked_not_authenticated": True,
        "text_flow": (
            "shared input/seed identities -> three paired arm ledgers -> match device, "
            "precision, threads, software and clock policy -> retain every planned "
            "block status/cost -> sum setup+training+solve+inference+I/O+failure recovery "
            "for this one task -> compare full-batch wall time only when complete"
        ),
    }
