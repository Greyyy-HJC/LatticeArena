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
               / "results/optimize_gaussian/contract_diagnostic_20261125_v3")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONTRACT_PATH = Path(__file__).with_name("research_contract_v1.json")
REFERENCE_FIELDS = (
    "observable", "units", "physical_interpretation", "reference_method",
    "data_source", "pass_rule", "failure_rule", "independent_review_receipt",
)
OPERATOR_FIELDS = (
    "H_definition", "mathematical_link_domain", "physical_link_gate",
    "spectral_interval", "spectral_mapping", "identity_normalization", "coefficient_constraints",
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
    """Return selected method defaults while leaving empirical receipts null."""
    return {
        "schema": SCHEMA,
        "status": "DRAFT_NOT_READY",
        "physical_score": None,
        "certified": False,
        "data_access": "none",
        "authorization": {"production_data": False, "gpu_quda": False},
        "method_status": "METHOD_DEFAULTS_SELECTED_EVIDENCE_NOT_READY",
        "comparison_family": list(COMPARISONS),
        "D04_reference": {
            "observable": "lowest zero-momentum valence isovector pseudoscalar energy on S16T16 quenched Wilson, kappa=0.14",
            "units": "a_t m_pi",
            "physical_interpretation": "independent target for the dimensionless Pion fixed-protocol risk objective",
            "reference_method": "candidate-blind joint two-state PP(n=0,8,16)+local A4P spectral fit",
            "data_source": "independent E-chain; correlator receipt not yet supplied",
            "data_receipt_sha256": None,
            "fit_protocol": {
                "main_window": [3, 8], "tmin_checks": [2, 3, 4],
                "state_count_checks": [2, 3], "PP_fold": "even",
                "A4P_fold": "odd", "shared_states": True,
                "covariance_diagonal_shrinkage": 0.25,
                "covariance_eigenvalue_floor": 1e-10,
                "all_arms_share_windows_covariance_blocks_and_reference_draws": True,
            },
            "risk_objective": {
                "q_theta": "C_theta(t)/(A_theta*f0(t))",
                "R_theta": "mean_t=2..5[Var_boot(q_theta(t))+|E_boot(q_theta(t))-1|^2]",
                "amplitude_window": [6, 8],
                "f0_source": "independent E-chain reference",
                "units": "dimensionless",
            },
            "pass_rule": "relative mass error<=0.05; p>=0.01; PP/A4P and window/model shifts<=2.5 paired SE; no boundary hits",
            "failure_rule": "any failed reference gate makes D04 unresolved; do not substitute synthetic operator loss",
            "independent_review_receipt": None,
        },
        "D06_effect_threshold": {
            "r_star": 1.1111111111111112,
            "units": "dimensionless risk ratio",
            "direction_and_interpretation": "rho=R_comparator/R_trainableW; rho>1 favors trainable-W; require at least 10% risk reduction",
            "prespecified_before_confirmation": True,
            "family_size": 2,
            "alpha_total_one_sided": 0.05,
            "alpha_per_comparison_bonferroni": 0.025,
            "paired_block_bootstrap_draws": 10000,
            "primary_block_length": 5,
            "sensitivity_block_lengths": [2, 10],
            "decision_rule": "hull lower bound across block lengths must exceed r_star for both comparisons",
            "invalid_or_zero_risk_rule": "ABSTAIN_RATIO_UNDEFINED; do not add epsilon or change to an absolute-difference estimand",
        },
        "D11_strong_spectral_operator": {
            "family": "Chebyshev",
            "degree": 4,
            "basis_parameterization": "monotone Bernstein controls mapped by a fixed exact dyadic rational degree-4 transform, evaluated in float64, to a Chebyshev series for Clenshaw application",
            "H_definition": "H=sum_i(T_i+T_i^dagger-2I), three periodic spatial directions, equal-weight unitary 3x3 links; U(3) unitarity suffices for the stated spectral bound",
            "mathematical_link_domain": "unitary U(3) links suffice for the Hermitian spectral bound",
            "physical_link_gate": "production QCD inputs must carry an SU(3) provenance or determinant-unitarity receipt; the CPU helper checks U-dagger-U only",
            "spectral_interval": [-12.0, 0.0],
            "spectral_mapping": "y=I+H/12 in [0,1]; x=2y-1 in [-1,1]",
            "identity_normalization": "P4(y=1)=b4=1 at the upper spectral endpoint; this is an H=0 eigenmode normalization only when such a mode exists; outer unnormalized G^2 P4(H) G^2 is common to all arms",
            "coefficient_constraints": "P4(y)=sum_i b_i*C(4,i)*y^i*(1-y)^(4-i), 0<=b0<=b1<=b2<=b3<=b4=1",
            "positivity_projection_constraints": "Bernstein basis gives 0<=P4<=1; monotone controls imply dP4/dy>=0; no post-fit clipping or projection",
            "outer_gaussian_sandwich": {"G": "I+0.10*sum_i(T_i+T_i^dagger)", "power_m": 2,
                                         "G_relation": "G=1.6I+0.1H",
                                         "G_spectrum": [0.4, 1.6],
                                         "raw_zero_mode_factor": 6.5536,
                                         "raw_zero_mode_factor_condition": "conditional on an H=0 eigenmode; not guaranteed for a general gauge field",
                                         "per_step_normalization": False},
            "failure_rule": "reject nonunitary links, wrong shape/boundary, nonfinite values, or infeasible controls; never repair by clipping; physical production also requires independently checked SU(3) link provenance",
            "independent_review_receipt": None,
        },
        "fair_budget": {
            "objective_id": "D04_fixed_Pion_risk_v1",
            "instance_plan_sha256": None,
            "fit_manifest_sha256": None,
            "heldout_manifest_sha256": None,
            "seed_plan_sha256": None,
            "initialization_policy": "freeze and hash one prespecified initialization plan; use matched starts where parameterizations permit",
            "restart_policy": "one prespecified start per arm; no unbudgeted restarts; retain failed runs in assigned-arm denominator",
            "stopping_rule": "stop at 160 objective/gradient calls or the prespecified solver tolerance; count all objective and gradient evaluations",
            "hardware_environment": "not yet measured; record CPU/GPU model, thread count, precision, software versions and clock policy",
            "objective_gradient_call_cap_per_arm": 160,
            "amortization_task_count": 1,
            "walltime_rule": "charge full training/setup/selection cost to one task until a supported deployment task count and cost ledger exist",
            "costs": {
                name: {"amount": None, "unit": "seconds"} for name in COST_FIELDS
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


def _check_selected_method_defaults(contract: dict[str, Any]) -> list[str]:
    """Prevent evidence updates from silently changing the selected protocol."""
    expected = empty_contract()
    issues: list[str] = []
    if contract.get("comparison_family") != expected["comparison_family"]:
        issues.append("comparison_family: differs from the selected two-contrast family")
    reference = contract.get("D04_reference")
    if not _mapping(reference):
        return issues
    for field in ("observable", "units", "physical_interpretation", "reference_method",
                  "data_source", "fit_protocol", "risk_objective", "pass_rule",
                  "failure_rule"):
        if reference.get(field) != expected["D04_reference"][field]:
            issues.append(f"D04_reference.{field}: differs from selected method defaults")
    threshold = contract.get("D06_effect_threshold")
    if _mapping(threshold):
        expected_threshold = expected["D06_effect_threshold"]
        if threshold != expected_threshold:
            issues.append("D06_effect_threshold: differs from selected effect and inference defaults")
    operator = contract.get("D11_strong_spectral_operator")
    if _mapping(operator):
        expected_operator = expected["D11_strong_spectral_operator"]
        for field, value in expected_operator.items():
            if field != "independent_review_receipt" and operator.get(field) != value:
                issues.append(f"D11_strong_spectral_operator.{field}: differs from selected method defaults")
    budget = contract.get("fair_budget")
    if _mapping(budget):
        for field in ("objective_id", "initialization_policy", "restart_policy",
                      "stopping_rule", "hardware_environment",
                      "objective_gradient_call_cap_per_arm", "amortization_task_count",
                      "walltime_rule"):
            if budget.get(field) != expected["fair_budget"][field]:
                issues.append(f"fair_budget.{field}: differs from selected planning rule")
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
    if contract.get("method_status") != "METHOD_DEFAULTS_SELECTED_EVIDENCE_NOT_READY":
        issues.append("method_status: selected defaults must remain evidence-not-ready")
    issues.extend(_check_selected_method_defaults(contract))
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
    contract["D04_reference"]["data_receipt_sha256"] = hashlib.sha256(
        b"SYNTHETIC_ONLY_D04").hexdigest()
    contract["D04_reference"]["independent_review_receipt"] = "SYNTHETIC_ONLY_review"
    contract["D11_strong_spectral_operator"]["independent_review_receipt"] = hashlib.sha256(
        b"SYNTHETIC_ONLY_D11_review").hexdigest()
    budget = contract["fair_budget"]
    for name in ("instance_plan_sha256", "fit_manifest_sha256",
                 "heldout_manifest_sha256", "seed_plan_sha256"):
        budget[name] = hashlib.sha256(f"SYNTHETIC_ONLY_{name}".encode()).hexdigest()
    for name, cost in budget["costs"].items():
        cost["amount"] = 1.0
        cost["unit"] = f"SYNTHETIC_ONLY_{name}_unit"
    return contract


def build(output: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    """Write a source-closed text diagnostic for the open research contract."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    current_contract = json.loads(CONTRACT_PATH.read_text())
    complete = _synthetic_complete_contract()
    missing_comparison = _synthetic_complete_contract()
    missing_comparison["comparison_family"] = [COMPARISONS[0]]
    invalid_threshold = _synthetic_complete_contract()
    invalid_threshold["D06_effect_threshold"]["r_star"] = 0
    missing_operator = _synthetic_complete_contract()
    missing_operator["D11_strong_spectral_operator"]["positivity_projection_constraints"] = None
    fixtures = {
        "current_selected_method_draft": current_contract,
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
        "The method defaults are recorded in the contract: D04 uses the candidate-blind independent E-chain joint spectrum fit and dimensionless Pion risk; D06 uses comparator/trainable-W risk ratios with `r_star=1.111111...`, `F=2`, Bonferroni one-sided `alpha=0.025`, and paired length-5 bootstrap with lengths 2/10 as sensitivity; D11 uses a positive monotone degree-4 Bernstein polynomial mapped by a fixed dyadic rational transform (symbolic identity, float64 evaluation) to Chebyshev form, with a common unnormalized `G^2 P4(H) G^2` sandwich. The two selected contrasts remain trainable-W versus same-budget SLSQP and trainable-W versus this fixed strong spectral baseline. Independent data receipts, measured costs, production review and authorization remain open.",
        "",
        "## Text decision flow",
        "",
        "```text",
        "selected D04/D06/D11 method defaults -> retain open reference, data, review, split and measured-cost receipts",
        "  -> fit W, same-budget SLSQP and degree-4 positive spectral baseline on common fit IDs",
        "  -> freeze outputs and evaluate both risk ratios on new paired confirmation IDs",
        "  -> require both simultaneous ratio-hull lower bounds > 1.111111...",
        "  -> missing or malformed fields: NOT_READY with exact paths",
        "  -> shape complete: CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW only",
        "  -> missing receipts/budget or failed reference: NOT_READY; synthetic operator loss is not D04 evidence",
        "  -> independent review + production/GPU authorization remain separate later gates",
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
        "## Evidence and measured inputs still missing",
        "",
        "- **D04:** independent E-chain correlator data receipt, fitted target value, successful reference gates, and independent review receipt.",
        "- **D06:** no validation result is available; confirmation must use the frozen ratio direction, 10,000 paired block draws, Bonferroni allocation and interval hull.",
        "- **D11:** independent operator-contract review receipt and real-link CPU/GPU operator evidence; the selected math constraints do not establish production-link correctness.",
        "- **Fair budget:** fit/heldout/split/seed hashes, measured per-arm training/setup/solve/inference/I/O/recovery/total seconds, matched hardware and independent cost review. The chosen cap is 160 objective/gradient calls and amortization count is conservatively one, but cost amounts remain null.",
        "- Production data and GPU/QUDA authorization are separate gates. No contract-complete fixture passes them.",
        "",
        "This diagnostic records the chosen method contract and tests its schema; it does not rank the methods, establish the physical reference, verify real-link operator properties, or provide GPU timing. The target remains `NOT_READY/UNCERTIFIED`, with `physical_score=null` and sealed data closed.",
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
        "method_status": empty_contract()["method_status"],
        "selected_method_contract": current_contract,
        "fixtures": list(fixtures),
        "synthetic_fixture_values_are_physical": False,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (out / "snapshot_optimize_contract.py").write_bytes(Path(__file__).read_bytes())
    (out / "research_contract_v1.json").write_bytes(CONTRACT_PATH.read_bytes())
    files = ["README.md", "contract_diagnostics.json", "contract_summary.csv",
             "protocol.json", "snapshot_optimize_contract.py",
             "research_contract_v1.json"]
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
