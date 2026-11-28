"""Structural draft for the Optimize Gaussian research contract.

This inspector records the two user-selected contrasts and enumerates the
still-open D04/D06/D11 and fairness-budget inputs. It reads no measurements,
does not authorize GPU work, and never produces a physical score.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


SCHEMA = "optimize-gaussian-dual-contrast-contract-v1"
COMPARISONS = (
    "trainable_W_vs_same_budget_SLSQP",
    "trainable_W_vs_fixed_degree4_Chebyshev",
)
DEFAULT_OUT = (Path(__file__).resolve().parents[2]
               / "results/optimize_gaussian/contract_diagnostic_20261124_v1")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REFERENCE_FIELDS = (
    "observable", "units", "physical_interpretation", "reference_method",
    "data_source", "pass_rule", "failure_rule", "independent_review_receipt",
)
OPERATOR_FIELDS = (
    "H_definition", "spectral_interval", "spectral_mapping",
    "identity_normalization", "coefficient_constraints",
    "positivity_projection_constraints", "failure_rule",
    "independent_review_receipt",
)
OPERATOR_STRING_FIELDS = tuple(
    field for field in OPERATOR_FIELDS if field != "spectral_interval"
)
BUDGET_STRING_FIELDS = (
    "objective_id", "instance_plan_sha256", "fit_manifest_sha256",
    "heldout_manifest_sha256", "seed_plan_sha256", "initialization_policy",
    "restart_policy", "stopping_rule", "hardware_environment",
)
COST_FIELDS = (
    "training", "setup", "solve", "inference", "io",
    "failure_recovery", "total_walltime",
)


def empty_contract() -> dict[str, Any]:
    """Return the current null draft without freezing scientific inputs."""
    return {
        "schema": SCHEMA,
        "status": "DRAFT_NOT_READY",
        "physical_score": None,
        "certified": False,
        "data_access": "none",
        "authorization": {"production_data": False, "gpu_quda": False},
        "comparison_family": list(COMPARISONS),
        "D04_reference": {
            "observable": None,
            "units": None,
            "physical_interpretation": None,
            "reference_method": None,
            "data_source": None,
            "data_receipt_sha256": None,
            "pass_rule": None,
            "failure_rule": None,
            "independent_review_receipt": None,
        },
        "D06_effect_threshold": {
            "r_star": None,
            "units": None,
            "direction_and_interpretation": None,
            "prespecified_before_confirmation": None,
        },
        "D11_strong_spectral_operator": {
            "family": "Chebyshev",
            "degree": 4,
            "H_definition": None,
            "spectral_interval": None,
            "spectral_mapping": None,
            "identity_normalization": None,
            "coefficient_constraints": None,
            "positivity_projection_constraints": None,
            "failure_rule": None,
            "independent_review_receipt": None,
        },
        "fair_budget": {
            "objective_id": None,
            "instance_plan_sha256": None,
            "fit_manifest_sha256": None,
            "heldout_manifest_sha256": None,
            "seed_plan_sha256": None,
            "initialization_policy": None,
            "restart_policy": None,
            "stopping_rule": None,
            "hardware_environment": None,
            "objective_gradient_call_cap_per_arm": None,
            "amortization_task_count": None,
            "costs": {
                name: {"amount": None, "unit": None} for name in COST_FIELDS
            },
        },
    }


def _mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def _require_strings(obj: Any, fields: tuple[str, ...], label: str) -> list[str]:
    if not _mapping(obj):
        return [f"{label}: expected an object"]
    return [f"{label}.{name}: missing nonempty value"
            for name in fields if not _nonempty(obj.get(name))]


def _check_reference(reference: Any) -> list[str]:
    label = "D04_reference"
    issues = _require_strings(reference, REFERENCE_FIELDS, label)
    if not _mapping(reference):
        return issues
    if not _sha256(reference.get("data_receipt_sha256")):
        issues.append(f"{label}.data_receipt_sha256: expected a SHA-256 receipt")
    return issues


def _check_threshold(threshold: Any) -> list[str]:
    label = "D06_effect_threshold"
    if not _mapping(threshold):
        return [f"{label}: expected an object"]
    issues: list[str] = []
    if not _number(threshold.get("r_star")) or threshold["r_star"] <= 0:
        issues.append(f"{label}.r_star: expected a positive finite value")
    for field in ("units", "direction_and_interpretation"):
        if not _nonempty(threshold.get(field)):
            issues.append(f"{label}.{field}: missing nonempty value")
    if threshold.get("prespecified_before_confirmation") is not True:
        issues.append(f"{label}.prespecified_before_confirmation: must be true")
    return issues


def _check_operator(operator: Any) -> list[str]:
    label = "D11_strong_spectral_operator"
    if not _mapping(operator):
        return [f"{label}: expected an object"]
    issues: list[str] = []
    if operator.get("family") != "Chebyshev":
        issues.append(f"{label}.family: expected the selected Chebyshev family")
    degree = operator.get("degree")
    if not isinstance(degree, int) or isinstance(degree, bool) or degree != 4:
        issues.append(f"{label}.degree: expected degree 4")
    for field in OPERATOR_STRING_FIELDS:
        if not _nonempty(operator.get(field)):
            issues.append(f"{label}.{field}: missing nonempty value")
    interval = operator.get("spectral_interval")
    if (not isinstance(interval, list) or len(interval) != 2
            or not all(_number(value) for value in interval)
            or interval[0] >= interval[1]):
        issues.append(f"{label}.spectral_interval: expected two finite increasing endpoints")
    receipt = operator.get("independent_review_receipt")
    if not _sha256(receipt):
        issues.append(f"{label}.independent_review_receipt: expected a SHA-256 receipt")
    return issues


def _check_budget(budget: Any) -> list[str]:
    label = "fair_budget"
    issues = _require_strings(budget, BUDGET_STRING_FIELDS, label)
    if not _mapping(budget):
        return issues
    for name in ("instance_plan_sha256", "fit_manifest_sha256",
                 "heldout_manifest_sha256", "seed_plan_sha256"):
        if not _sha256(budget.get(name)):
            issues.append(f"{label}.{name}: expected a SHA-256 receipt")
    for name in ("objective_gradient_call_cap_per_arm", "amortization_task_count"):
        if not _positive_int(budget.get(name)):
            issues.append(f"{label}.{name}: expected a positive integer")
    costs = budget.get("costs")
    if not _mapping(costs):
        return issues + [f"{label}.costs: expected an object"]
    for name in COST_FIELDS:
        cost = costs.get(name)
        if not _mapping(cost):
            issues.append(f"{label}.costs.{name}: expected amount and unit")
            continue
        amount, unit = cost.get("amount"), cost.get("unit")
        if not _number(amount) or amount < 0:
            issues.append(f"{label}.costs.{name}.amount: expected a finite nonnegative value")
        if not _nonempty(unit):
            issues.append(f"{label}.costs.{name}.unit: missing nonempty value")
    return issues


def inspect_contract(contract: Any) -> dict[str, Any]:
    """Check metadata shape; even a complete fixture requires human review."""
    if not _mapping(contract):
        return {
            "schema_valid": False,
            "status": "NOT_READY",
            "structurally_complete": False,
            "certified": False,
            "physical_score": None,
            "issues": ["contract: expected an object"],
        }
    issues: list[str] = []
    schema_valid = contract.get("schema") == SCHEMA
    if not schema_valid:
        issues.append(f"schema: expected {SCHEMA}")
    if contract.get("status") != "DRAFT_NOT_READY":
        issues.append("status: input contract must remain DRAFT_NOT_READY")
    if "physical_score" not in contract or contract["physical_score"] is not None:
        issues.append("physical_score: field must exist and remain null in this structural inspector")
    if contract.get("certified") is not False:
        issues.append("certified: must remain false in this structural inspector")
    if contract.get("data_access") != "none":
        issues.append("data_access: this inspector must declare none")
    authorization = contract.get("authorization")
    if not _mapping(authorization):
        issues.append("authorization: expected explicit gates")
    else:
        for field in ("production_data", "gpu_quda"):
            if not isinstance(authorization.get(field), bool):
                issues.append(f"authorization.{field}: expected an explicit boolean")

    comparisons = contract.get("comparison_family")
    if not isinstance(comparisons, list) or comparisons != list(COMPARISONS):
        issues.append("comparison_family: must retain both selected contrasts in the declared order")
    issues.extend(_check_reference(contract.get("D04_reference")))
    issues.extend(_check_threshold(contract.get("D06_effect_threshold")))
    issues.extend(_check_operator(contract.get("D11_strong_spectral_operator")))
    issues.extend(_check_budget(contract.get("fair_budget")))
    complete = schema_valid and not issues
    return {
        "schema_valid": schema_valid,
        "status": ("CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW"
                   if complete else "NOT_READY"),
        "structurally_complete": complete,
        "certified": False,
        "physical_score": None,
        "issues": issues,
    }


def _synthetic_complete_contract() -> dict[str, Any]:
    """Return a shape fixture with visibly synthetic values only."""
    contract = empty_contract()
    contract["D04_reference"] = {
        "observable": "SYNTHETIC_ONLY_observable",
        "units": "SYNTHETIC_ONLY_units",
        "physical_interpretation": "SYNTHETIC_ONLY_interpretation",
        "reference_method": "SYNTHETIC_ONLY_reference_method",
        "data_source": "SYNTHETIC_ONLY_source",
        "data_receipt_sha256": hashlib.sha256(b"SYNTHETIC_ONLY_D04").hexdigest(),
        "pass_rule": "SYNTHETIC_ONLY_pass_rule",
        "failure_rule": "SYNTHETIC_ONLY_failure_rule",
        "independent_review_receipt": "SYNTHETIC_ONLY_review",
    }
    contract["D06_effect_threshold"] = {
        "r_star": 1.0,
        "units": "SYNTHETIC_ONLY_units",
        "direction_and_interpretation": "SYNTHETIC_ONLY_effect",
        "prespecified_before_confirmation": True,
    }
    contract["D11_strong_spectral_operator"] = {
        "family": "Chebyshev", "degree": 4,
        "H_definition": "SYNTHETIC_ONLY_H",
        "spectral_interval": [-1.0, 1.0],
        "spectral_mapping": "SYNTHETIC_ONLY_affine_map",
        "identity_normalization": "SYNTHETIC_ONLY_identity_term",
        "coefficient_constraints": "SYNTHETIC_ONLY_coefficient_set",
        "positivity_projection_constraints": "SYNTHETIC_ONLY_constraint",
        "failure_rule": "SYNTHETIC_ONLY_failure_rule",
        "independent_review_receipt": hashlib.sha256(
            b"SYNTHETIC_ONLY_D11_review").hexdigest(),
    }
    budget = contract["fair_budget"]
    for name in BUDGET_STRING_FIELDS:
        budget[name] = f"SYNTHETIC_ONLY_{name}"
    for name in ("instance_plan_sha256", "fit_manifest_sha256",
                 "heldout_manifest_sha256", "seed_plan_sha256"):
        budget[name] = hashlib.sha256(f"SYNTHETIC_ONLY_{name}".encode()).hexdigest()
    budget["objective_gradient_call_cap_per_arm"] = 8
    budget["amortization_task_count"] = 1
    for name, cost in budget["costs"].items():
        cost["amount"] = 1.0
        cost["unit"] = f"SYNTHETIC_ONLY_{name}_unit"
    return contract


def build(output: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    """Write a source-closed text diagnostic for the open research contract."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    complete = _synthetic_complete_contract()
    missing_comparison = _synthetic_complete_contract()
    missing_comparison["comparison_family"] = [COMPARISONS[0]]
    invalid_threshold = _synthetic_complete_contract()
    invalid_threshold["D06_effect_threshold"]["r_star"] = 0
    missing_operator = _synthetic_complete_contract()
    missing_operator["D11_strong_spectral_operator"]["positivity_projection_constraints"] = None
    fixtures = {
        "current_null_draft": empty_contract(),
        "synthetic_complete_requires_review": complete,
        "synthetic_missing_comparison": missing_comparison,
        "synthetic_invalid_r_star": invalid_threshold,
        "synthetic_incomplete_D11": missing_operator,
    }
    results = {name: inspect_contract(value) for name, value in fixtures.items()}
    rows = [{
        "fixture": name,
        "unit": "one contract check",
        "planned": 1,
        "valid": int(result["structurally_complete"]),
        "failed": int(not result["structurally_complete"]),
        "abstained": 0,
        "status": result["status"],
        "certified": result["certified"],
        "physical_score": result["physical_score"],
        "issue_count": len(result["issues"]),
    } for name, result in results.items()]
    (out / "contract_diagnostics.json").write_text(json.dumps({
        "schema": SCHEMA, "data_access": "none", "physical_score": None,
        "certified": False, "fixtures": results,
    }, indent=2, allow_nan=False) + "\n")
    with (out / "contract_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Optimize Gaussian contract diagnostic",
        "",
        "**STRUCTURE CHECK ONLY. `physical_score=null`; no measurements, sealed data, GPU or QUDA inputs were read.**",
        "",
        "The draft preserves both user-selected comparisons: trainable-W versus same-budget SLSQP, and trainable-W versus the fixed degree-4 Chebyshev spectral baseline. D04 reference, D06 `r_star`, D11 operator details and the production fairness budget remain open.",
        "",
        "## Text decision flow",
        "",
        "```text",
        "null D04/D06/D11/budget contract",
        "  -> require both selected comparisons and all provenance/units/constraint fields",
        "  -> missing or malformed fields: NOT_READY with exact paths",
        "  -> shape complete: CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW only",
        "  -> independent reference + predeclared effect + operator review + authorization still required",
        "  -> physical_score remains null throughout this inspector",
        "```",
        "",
        "`contract_summary.csv` uses one contract check as its unit; it is not a solver run or measurement denominator. The complete fixture contains only `SYNTHETIC_ONLY` values. It checks shape handling and cannot fill any physical decision.",
        "",
        "|fixture|planned|valid|failed|abstained|status|blockers|",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"|{row['fixture']}|{row['planned']}|{row['valid']}|{row['failed']}|{row['abstained']}|{row['status']}|{row['issue_count']}|"
        )
    lines.extend([
        "",
        "## What remains unset",
        "",
        "- **D04:** independent physical observable/reference, units, data provenance, pass/failure rules and independent review receipt.",
        "- **D06:** minimum meaningful effect `r_star`, units, direction and interpretation, frozen before confirmation.",
        "- **D11:** exact degree-4 Chebyshev `H`, spectral interval and map, identity normalization, coefficient and positivity/projection constraints, failure rule and review receipt.",
        "- **Fair budget:** same objective and split identities, initialization/restarts, per-arm objective/gradient cap, training/setup/solve/inference/I/O/recovery/total-walltime costs, hardware and amortization count.",
        "- Production data and GPU/QUDA authorization are separate gates. No contract-complete fixture passes them.",
        "",
        "This diagnostic is a metadata gate. It does not rank the methods, validate a physical reference, confirm `r_star`, or assert that the current Chebyshev candidate is the final baseline contract.",
        "",
    ])
    (out / "README.md").write_text("\n".join(lines))
    protocol = {
        "schema": SCHEMA,
        "status": "STRUCTURE_DIAGNOSTIC_ONLY",
        "data_access": "none",
        "physical_score": None,
        "certified": False,
        "comparisons": list(COMPARISONS),
        "fixtures": list(fixtures),
        "synthetic_fixture_values_are_physical": False,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (out / "snapshot_optimize_contract.py").write_bytes(Path(__file__).read_bytes())
    files = ["README.md", "contract_diagnostics.json", "contract_summary.csv",
             "protocol.json", "snapshot_optimize_contract.py"]
    hashes = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
              for name in files}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return {"status": protocol["status"], "physical_score": None,
            "fixtures": results, "files": files + ["artifact_hashes.json"]}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = build(args.output)
    print(json.dumps({
        "status": summary["status"],
        "physical_score": summary["physical_score"],
        "fixture_statuses": {
            name: result["status"] for name, result in summary["fixtures"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
