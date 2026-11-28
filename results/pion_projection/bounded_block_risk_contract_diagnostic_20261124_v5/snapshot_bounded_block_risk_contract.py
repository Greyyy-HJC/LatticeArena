"""Fail-closed structural inspector for the Pion bounded block-risk contract.

The inspector reads only the versioned contract draft, never measurement or
sealed inputs. A structurally complete fixture still requires human review and
cannot set ``physical_score`` or authorize data/GPU access.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


SCHEMA = "pion-bounded-block-risk-contract-v1"
DEFAULT_OUT = (Path(__file__).resolve().parents[2]
               / "results/pion_projection/bounded_block_risk_contract_diagnostic_20261124_v5")
CONTRACT_PATH = Path(__file__).with_name("bounded_block_risk_contract_v1.json")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEPENDENCE_METHODS = {"graph_color_hoeffding", "markov_pseudo_gap_bernstein"}
ANCHOR_METHODS = {"known_sigma_normal_interval", "bounded_iid_hoeffding"}
FAILURE_POLICY = "failed_block=-1; all_failed_comparison=abstain"


def empty_contract() -> dict[str, Any]:
    """Return a draft with every study-specific input unset."""
    return {
        "schema": SCHEMA,
        "status": "DRAFT_NOT_READY",
        "physical_score": None,
        "certified": False,
        "data_access": "none",
        "authorization": {"production_data": False, "gpu_quda": False},
        "estimand": {
            "name": "bounded_block_risk",
            "target": {
                "m_star": None,
                "observable": None,
                "units": None,
                "physical_interpretation": None,
            },
            "score_component": "paired_squared_error_difference",
            "score_scale_R": None,
            "scale_units": None,
            "prediction_difference_abs_bound": None,
            "prediction_difference_receipt_sha256": None,
            "clipping_interval": None,
            "failure_policy": None,
            "planned_blocks_N": None,
            "block_plan_receipt_sha256": None,
            "delta_star": None,
            "delta_star_units": None,
            "result_unit": "one planned candidate-reference comparison",
            "candidate_family_ids": None,
            "reference_gate": {
                "method": None,
                "data_source_receipt_sha256": None,
                "pass_rule": None,
                "failure_rule": None,
                "independent_review_receipt_sha256": None,
            },
        },
        "dependence": {
            "method": None,
            "score_specific_block_map": None,
            "evidence_receipt_sha256": None,
            "chromatic_number": None,
            "color_class_independence_receipt_sha256": None,
            "gamma_ps_lower_bound": None,
            "block_chain_assumptions": None,
            "block_chain_receipt_sha256": None,
        },
        "anchor": {
            "estimate_receipt_sha256": None,
            "target_match_statement": None,
            "units": None,
            "total_error_bound_B": None,
            "common_bias_bound_C": None,
            "bound_combination_rule": None,
            "coverage_method": None,
            "coverage_parameters": None,
            "coverage_probability_lower_bound": None,
            "coverage_receipt_sha256": None,
            "common_bias_coverage_receipt_sha256": None,
            "conditional_independence_receipt_sha256": None,
        },
        "inference": {
            "comparison_family_ids": None,
            "F": None,
            "alpha_total": None,
            "alpha_score": None,
            "alpha_anchor": None,
        },
    }


def _mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def _required_strings(obj: Any, fields: tuple[str, ...], label: str) -> list[str]:
    if not _mapping(obj):
        return [f"{label}: expected an object"]
    return [f"{label}.{field}: missing nonempty value"
            for field in fields if not _nonempty(obj.get(field))]


def _required_receipt(obj: Any, field: str, label: str) -> list[str]:
    if not _mapping(obj):
        return [f"{label}: expected an object"]
    if not _sha256(obj.get(field)):
        return [f"{label}.{field}: expected a SHA-256 receipt"]
    return []


def _check_reference_gate(gate: Any) -> list[str]:
    label = "estimand.reference_gate"
    issues = _required_strings(gate, ("method", "pass_rule", "failure_rule"), label)
    issues.extend(_required_receipt(gate, "data_source_receipt_sha256", label))
    issues.extend(_required_receipt(gate, "independent_review_receipt_sha256", label))
    return issues


def _check_estimand(estimand: Any) -> list[str]:
    label = "estimand"
    if not _mapping(estimand):
        return [f"{label}: expected an object"]
    issues: list[str] = []
    if estimand.get("name") != "bounded_block_risk":
        issues.append(f"{label}.name: expected bounded_block_risk")
    target = estimand.get("target")
    issues.extend(_required_strings(
        target, ("observable", "units", "physical_interpretation"),
        f"{label}.target",
    ))
    if not _mapping(target) or not _finite_number(target.get("m_star")):
        issues.append(f"{label}.target.m_star: expected an explicit finite target value")
    if estimand.get("score_component") != "paired_squared_error_difference":
        issues.append(f"{label}.score_component: v1 supports only paired squared-error difference")
    scale = estimand.get("score_scale_R")
    if not _finite_number(scale) or scale <= 0:
        issues.append(f"{label}.score_scale_R: expected a finite positive value")
    elif not math.isfinite(float(scale) * float(scale)):
        issues.append(f"{label}.score_scale_R: squared scale must remain finite")
    scale_units = estimand.get("scale_units")
    if not _nonempty(scale_units):
        issues.append(f"{label}.scale_units: missing nonempty value")
    elif _mapping(target) and _nonempty(target.get("units")) and scale_units != target["units"]:
        issues.append(f"{label}.scale_units: must match estimand.target.units")

    prediction_bound = estimand.get("prediction_difference_abs_bound")
    if not _finite_number(prediction_bound) or prediction_bound < 0:
        issues.append(f"{label}.prediction_difference_abs_bound: expected a finite nonnegative value")
    elif _finite_number(scale) and prediction_bound > scale:
        issues.append(f"{label}.prediction_difference_abs_bound: exceeds score_scale_R")
    issues.extend(_required_receipt(
        estimand, "prediction_difference_receipt_sha256", label,
    ))
    if estimand.get("clipping_interval") != [-1, 1]:
        issues.append(f"{label}.clipping_interval: v1 helper requires [-1, 1]; human confirmation remains required")
    if estimand.get("failure_policy") != FAILURE_POLICY:
        issues.append(f"{label}.failure_policy: must preserve failed-block and all-failed rules")
    if not _positive_int(estimand.get("planned_blocks_N")):
        issues.append(f"{label}.planned_blocks_N: expected a positive integer")
    issues.extend(_required_receipt(estimand, "block_plan_receipt_sha256", label))
    delta_star = estimand.get("delta_star")
    if not _finite_number(delta_star) or delta_star <= 0:
        issues.append(f"{label}.delta_star: expected a positive finite predeclared effect")
    if not _nonempty(estimand.get("delta_star_units")):
        issues.append(f"{label}.delta_star_units: missing nonempty value")
    if estimand.get("result_unit") != "one planned candidate-reference comparison":
        issues.append(f"{label}.result_unit: v1 counts one planned candidate-reference comparison")
    family_ids = estimand.get("candidate_family_ids")
    if (not isinstance(family_ids, list) or not family_ids
            or any(not _nonempty(value) for value in family_ids)
            or len(set(family_ids)) != len(family_ids)):
        issues.append(f"{label}.candidate_family_ids: expected unique nonempty comparison IDs")
    issues.extend(_check_reference_gate(estimand.get("reference_gate")))
    return issues


def _check_dependence(dependence: Any, *, planned_blocks: Any) -> list[str]:
    label = "dependence"
    if not _mapping(dependence):
        return [f"{label}: expected an object"]
    issues: list[str] = []
    method = dependence.get("method")
    if not isinstance(method, str) or method not in DEPENDENCE_METHODS:
        issues.append(f"{label}.method: expected one of {sorted(DEPENDENCE_METHODS)}")
    if not _nonempty(dependence.get("score_specific_block_map")):
        issues.append(f"{label}.score_specific_block_map: missing score-to-block definition")
    issues.extend(_required_receipt(dependence, "evidence_receipt_sha256", label))
    if method == "graph_color_hoeffding":
        chi = dependence.get("chromatic_number")
        if not _positive_int(chi):
            issues.append(f"{label}.chromatic_number: expected a positive integer")
        elif _positive_int(planned_blocks) and chi > planned_blocks:
            issues.append(f"{label}.chromatic_number: cannot exceed planned_blocks_N")
        issues.extend(_required_receipt(
            dependence, "color_class_independence_receipt_sha256", label,
        ))
    elif method == "markov_pseudo_gap_bernstein":
        gamma = dependence.get("gamma_ps_lower_bound")
        if not _finite_number(gamma) or not 0 < gamma <= 1:
            issues.append(f"{label}.gamma_ps_lower_bound: expected a value in (0, 1]")
        if not _nonempty(dependence.get("block_chain_assumptions")):
            issues.append(f"{label}.block_chain_assumptions: missing explicit chain assumptions")
        issues.extend(_required_receipt(
            dependence, "block_chain_receipt_sha256", label,
        ))
    return issues


def _check_anchor(anchor: Any, *, target_units: Any,
                  alpha_anchor: Any) -> list[str]:
    label = "anchor"
    if not _mapping(anchor):
        return [f"{label}: expected an object"]
    issues: list[str] = []
    issues.extend(_required_receipt(anchor, "estimate_receipt_sha256", label))
    if not _nonempty(anchor.get("target_match_statement")):
        issues.append(f"{label}.target_match_statement: missing target/anchor correspondence")
    units = anchor.get("units")
    if not _nonempty(units):
        issues.append(f"{label}.units: missing nonempty value")
    elif _nonempty(target_units) and units != target_units:
        issues.append(f"{label}.units: must match estimand target units")
    total_bound = anchor.get("total_error_bound_B")
    if not _finite_number(total_bound) or total_bound < 0:
        issues.append(f"{label}.total_error_bound_B: expected a finite nonnegative bound")
    common_bias = anchor.get("common_bias_bound_C")
    if not _finite_number(common_bias) or common_bias < 0:
        issues.append(f"{label}.common_bias_bound_C: expected a finite nonnegative bound")
    elif _finite_number(total_bound) and common_bias > total_bound:
        issues.append(f"{label}.common_bias_bound_C: cannot exceed total_error_bound_B")
    if not _nonempty(anchor.get("bound_combination_rule")):
        issues.append(f"{label}.bound_combination_rule: state how random and common-bias terms form total B")
    method = anchor.get("coverage_method")
    if not isinstance(method, str) or method not in ANCHOR_METHODS:
        issues.append(f"{label}.coverage_method: expected one of {sorted(ANCHOR_METHODS)}")
    parameters = anchor.get("coverage_parameters")
    random_coverage_radius = None
    sample_count = None
    if not _mapping(parameters):
        issues.append(f"{label}.coverage_parameters: expected method-specific assumptions")
    elif method == "known_sigma_normal_interval":
        if not _positive_int(parameters.get("sample_count_n_A")):
            issues.append(f"{label}.coverage_parameters.sample_count_n_A: expected a positive integer")
        sample_count = parameters.get("sample_count_n_A")
        if not _positive_int(sample_count):
            issues.append(f"{label}.coverage_parameters.sample_count_n_A: expected a positive integer")
        sigma = parameters.get("known_sigma")
        if not _finite_number(sigma) or sigma <= 0:
            issues.append(f"{label}.coverage_parameters.known_sigma: expected a positive finite value")
        if not _nonempty(parameters.get("normality_assumption")):
            issues.append(f"{label}.coverage_parameters.normality_assumption: missing explicit assumption")
        if not _sha256(parameters.get("parameter_receipt_sha256")):
            issues.append(f"{label}.coverage_parameters.parameter_receipt_sha256: expected a SHA-256 receipt")
        if (not _finite_number(alpha_anchor) or alpha_anchor <= 0):
            issues.append(f"{label}.coverage_parameters: alpha_anchor must be positive for a sampling interval")
        elif (_positive_int(sample_count) and _finite_number(sigma) and sigma > 0):
            probability = 1.0 - alpha_anchor / 2.0
            if not 0.5 < probability < 1.0:
                issues.append(f"{label}.coverage_parameters: alpha_anchor is outside normal-quantile precision")
            else:
                random_coverage_radius = (
                    statistics.NormalDist().inv_cdf(probability)
                    * sigma / math.sqrt(sample_count)
                )
    elif method == "bounded_iid_hoeffding":
        if not _positive_int(parameters.get("sample_count_n_A")):
            issues.append(f"{label}.coverage_parameters.sample_count_n_A: expected a positive integer")
        sample_count = parameters.get("sample_count_n_A")
        if not _positive_int(sample_count):
            issues.append(f"{label}.coverage_parameters.sample_count_n_A: expected a positive integer")
        support = parameters.get("support_interval")
        support_valid = (isinstance(support, list) and len(support) == 2
                         and all(_finite_number(value) for value in support)
                         and support[0] < support[1])
        if not support_valid:
            issues.append(f"{label}.coverage_parameters.support_interval: expected finite increasing bounds")
        if not _nonempty(parameters.get("iid_unit_statement")):
            issues.append(f"{label}.coverage_parameters.iid_unit_statement: missing explicit iid unit")
        if not _sha256(parameters.get("support_receipt_sha256")):
            issues.append(f"{label}.coverage_parameters.support_receipt_sha256: expected a SHA-256 receipt")
        if not _finite_number(alpha_anchor) or alpha_anchor <= 0:
            issues.append(f"{label}.coverage_parameters: alpha_anchor must be positive for a sampling interval")
        elif _positive_int(sample_count) and support_valid:
            random_coverage_radius = (
                (support[1] - support[0])
                * math.sqrt(math.log(2.0 / alpha_anchor) / (2.0 * sample_count))
            )
    coverage = anchor.get("coverage_probability_lower_bound")
    if not _finite_number(coverage) or not 0 < coverage <= 1:
        issues.append(f"{label}.coverage_probability_lower_bound: expected a value in (0, 1]")
    elif _finite_number(alpha_anchor) and coverage < 1 - alpha_anchor - 1e-12:
        issues.append(f"{label}.coverage_probability_lower_bound: does not meet alpha_anchor")
    for field in (
        "coverage_receipt_sha256",
        "common_bias_coverage_receipt_sha256",
        "conditional_independence_receipt_sha256",
    ):
        issues.extend(_required_receipt(anchor, field, label))
    if (random_coverage_radius is not None and _finite_number(common_bias)
            and _finite_number(total_bound)):
        required_total = random_coverage_radius + common_bias
        tolerance = 1e-12 * max(1.0, abs(required_total))
        if total_bound + tolerance < required_total:
            issues.append(
                f"{label}.total_error_bound_B: must cover random radius "
                "plus common_bias_bound_C"
            )
    return issues


def _check_inference(inference: Any, *, candidate_family_ids: Any) -> list[str]:
    label = "inference"
    if not _mapping(inference):
        return [f"{label}: expected an object"]
    issues: list[str] = []
    family_ids = inference.get("comparison_family_ids")
    if (not isinstance(family_ids, list) or not family_ids
            or any(not _nonempty(value) for value in family_ids)
            or len(set(family_ids)) != len(family_ids)):
        issues.append(f"{label}.comparison_family_ids: expected unique nonempty IDs")
    elif family_ids != candidate_family_ids:
        issues.append(f"{label}.comparison_family_ids: must match estimand.candidate_family_ids")
    family = inference.get("F")
    if not _positive_int(family):
        issues.append(f"{label}.F: expected a positive integer")
    elif isinstance(family_ids, list) and family != len(family_ids):
        issues.append(f"{label}.F: must equal comparison_family_ids length")
    alpha_total = inference.get("alpha_total")
    alpha_score = inference.get("alpha_score")
    alpha_anchor = inference.get("alpha_anchor")
    if not _finite_number(alpha_total) or not 0 < alpha_total < 1:
        issues.append(f"{label}.alpha_total: expected a value in (0, 1)")
    if not _finite_number(alpha_score) or not 0 < alpha_score < 1:
        issues.append(f"{label}.alpha_score: expected a value in (0, 1)")
    if not _finite_number(alpha_anchor) or not 0 <= alpha_anchor < 1:
        issues.append(f"{label}.alpha_anchor: expected a value in [0, 1)")
    if (_finite_number(alpha_total) and _finite_number(alpha_score)
            and _finite_number(alpha_anchor)
            and alpha_score + alpha_anchor > alpha_total + 1e-12):
        issues.append(f"{label}.alpha_score + alpha_anchor: exceeds alpha_total")
    return issues


def inspect_contract(contract: Any) -> dict[str, Any]:
    """Check schema shape only; complete inputs still require independent review."""
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
        issues.append("status: must remain DRAFT_NOT_READY for this structural inspector")
    if "physical_score" not in contract or contract["physical_score"] is not None:
        issues.append("physical_score: must exist and remain null")
    if contract.get("certified") is not False:
        issues.append("certified: must remain false")
    if contract.get("data_access") != "none":
        issues.append("data_access: inspector must use none")
    authorization = contract.get("authorization")
    if not _mapping(authorization):
        issues.append("authorization: expected explicit gates")
    else:
        for field in ("production_data", "gpu_quda"):
            if not isinstance(authorization.get(field), bool):
                issues.append(f"authorization.{field}: expected an explicit boolean")

    estimand = contract.get("estimand")
    issues.extend(_check_estimand(estimand))
    planned_blocks = estimand.get("planned_blocks_N") if _mapping(estimand) else None
    issues.extend(_check_dependence(
        contract.get("dependence"), planned_blocks=planned_blocks,
    ))
    target = estimand.get("target") if _mapping(estimand) else None
    target_units = target.get("units") if _mapping(target) else None
    inference = contract.get("inference")
    alpha_anchor = inference.get("alpha_anchor") if _mapping(inference) else None
    issues.extend(_check_anchor(
        contract.get("anchor"), target_units=target_units,
        alpha_anchor=alpha_anchor,
    ))
    candidate_family_ids = (estimand.get("candidate_family_ids")
                            if _mapping(estimand) else None)
    issues.extend(_check_inference(
        inference, candidate_family_ids=candidate_family_ids,
    ))
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


def _synthetic_receipt(label: str) -> str:
    return hashlib.sha256(f"SYNTHETIC_ONLY:{label}".encode()).hexdigest()


def _synthetic_complete_contract(method: str = "graph_color_hoeffding") -> dict[str, Any]:
    """Build a schema fixture whose values are visibly synthetic only."""
    contract = empty_contract()
    family_ids = ["SYNTHETIC_ONLY_candidate_A", "SYNTHETIC_ONLY_candidate_B"]
    contract["estimand"].update({
        "target": {
            "m_star": 0.14,
            "observable": "SYNTHETIC_ONLY_observable",
            "units": "SYNTHETIC_ONLY_unit",
            "physical_interpretation": "SYNTHETIC_ONLY_interpretation",
        },
        "score_scale_R": 2.0,
        "scale_units": "SYNTHETIC_ONLY_unit",
        "prediction_difference_abs_bound": 1.5,
        "prediction_difference_receipt_sha256": _synthetic_receipt("prediction-bound"),
        "clipping_interval": [-1, 1],
        "failure_policy": FAILURE_POLICY,
        "planned_blocks_N": 20,
        "block_plan_receipt_sha256": _synthetic_receipt("block-plan"),
        "delta_star": 0.1,
        "delta_star_units": "SYNTHETIC_ONLY_unit_squared",
        "candidate_family_ids": family_ids,
        "reference_gate": {
            "method": "SYNTHETIC_ONLY_reference_method",
            "data_source_receipt_sha256": _synthetic_receipt("reference-source"),
            "pass_rule": "SYNTHETIC_ONLY_pass_rule",
            "failure_rule": "SYNTHETIC_ONLY_failure_rule",
            "independent_review_receipt_sha256": _synthetic_receipt("reference-review"),
        },
    })
    dependence = contract["dependence"]
    dependence.update({
        "method": method,
        "score_specific_block_map": "SYNTHETIC_ONLY fixed block map",
        "evidence_receipt_sha256": _synthetic_receipt("dependence-evidence"),
    })
    if method == "graph_color_hoeffding":
        dependence.update({
            "chromatic_number": 2,
            "color_class_independence_receipt_sha256": _synthetic_receipt("color-independence"),
        })
    elif method == "markov_pseudo_gap_bernstein":
        dependence.update({
            "gamma_ps_lower_bound": 0.25,
            "block_chain_assumptions": "SYNTHETIC_ONLY stationary block-chain assumptions",
            "block_chain_receipt_sha256": _synthetic_receipt("block-chain"),
        })
    anchor = contract["anchor"]
    anchor.update({
        "estimate_receipt_sha256": _synthetic_receipt("anchor-estimate"),
        "target_match_statement": "SYNTHETIC_ONLY matched anchor target",
        "units": "SYNTHETIC_ONLY_unit",
        "total_error_bound_B": 0.30,
        "common_bias_bound_C": 0.01,
        "bound_combination_rule": "SYNTHETIC_ONLY total B includes the random coverage radius and common-bias C",
        "coverage_method": "bounded_iid_hoeffding",
        "coverage_parameters": {
            "sample_count_n_A": 40,
            "support_interval": [-0.5, 0.5],
            "iid_unit_statement": "SYNTHETIC_ONLY independent anchor observations",
            "support_receipt_sha256": _synthetic_receipt("anchor-support"),
        },
        "coverage_probability_lower_bound": 0.99,
        "coverage_receipt_sha256": _synthetic_receipt("anchor-coverage"),
        "common_bias_coverage_receipt_sha256": _synthetic_receipt("common-bias"),
        "conditional_independence_receipt_sha256": _synthetic_receipt("anchor-score-independence"),
    })
    contract["inference"].update({
        "comparison_family_ids": family_ids.copy(),
        "F": len(family_ids),
        "alpha_total": 0.05,
        "alpha_score": 0.04,
        "alpha_anchor": 0.01,
    })
    return contract


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(output: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    """Write a source-closed diagnostic using only the null contract and fixtures."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    current = json.loads(CONTRACT_PATH.read_text())
    valid_graph = _synthetic_complete_contract("graph_color_hoeffding")
    valid_markov = _synthetic_complete_contract("markov_pseudo_gap_bernstein")
    alpha_over = _synthetic_complete_contract()
    alpha_over["inference"]["alpha_score"] = 0.05
    bound_over = _synthetic_complete_contract()
    bound_over["estimand"]["prediction_difference_abs_bound"] = 2.1
    missing_anchor = _synthetic_complete_contract()
    missing_anchor["anchor"]["common_bias_coverage_receipt_sha256"] = None
    fixtures = {
        "current_null_contract": current,
        "synthetic_graph_color_complete": valid_graph,
        "synthetic_markov_gap_complete": valid_markov,
        "synthetic_alpha_over_budget": alpha_over,
        "synthetic_prediction_bound_over_R": bound_over,
        "synthetic_missing_anchor_coverage": missing_anchor,
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
        "schema": SCHEMA,
        "data_access": "none",
        "physical_score": None,
        "certified": False,
        "fixtures": {
            name: {"contract": fixtures[name], "result": result}
            for name, result in results.items()
        },
    }, indent=2, allow_nan=False) + "\n")
    with (out / "contract_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Pion bounded block-risk contract diagnostic",
        "",
        "**STRUCTURE CHECK ONLY. `physical_score=null`; no correlator, anchor, gap, validation, holdout, GPU, or QUDA inputs were read.**",
        "",
        "The inspector checks whether a future contract contains the fields required by the v1 estimand: bounded paired squared-error scores, a score-specific dependence route, anchor coverage/common-bias evidence, preallocated alpha, and a separate physical reference gate.",
        "",
        "## Text decision flow",
        "",
        "```text",
        "load the null v1 contract; do not load measurements",
        "  -> check estimand, R/prediction bound, clipping, failure policy, and planned-block receipt",
        "  -> require either a score-specific graph/color receipt or block-chain gamma_ps receipt",
        "  -> require matched anchor, bias/coverage/common-bias, and conditional-independence receipts",
        "  -> check comparison family F and alpha_score + alpha_anchor <= alpha_total",
        "  -> check independent physical reference/model-identification gate",
        "  -> missing/malformed field: NOT_READY with exact paths",
        "  -> complete synthetic shape: CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW only",
        "  -> physical_score remains null throughout",
        "```",
        "",
        "For the two-sided known-sigma normal route, the random radius is `z_(1-alpha_A/2) * sigma / sqrt(n_A)`; for bounded iid Hoeffding it is `(U-L) * sqrt(log(2/alpha_A) / (2*n_A))`. The v1 conservative combination requires total `B >= random_coverage_radius + C`, where `C` bounds common bias.",
        "",
        "Receipts are checked for presence and SHA-256 syntax only; their scientific contents and independence are not verified. `valid` below means structurally complete, not evidence-valid, measured, or certified. Each row counts one contract check; it is not a score block or physical sample.",
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
        "## Remaining researcher-supplied inputs",
        "",
        "- Target `m*`, observable, interpretation and units; positive `R`, the all-included prediction-difference bound and its receipt; clipping and failure-policy approval.",
        "- Planned block identities; score-specific graph/color proof or block-chain assumptions and `gamma_ps` lower-bound receipt; candidate comparison family `F`.",
        "- Matched E-anchor receipt, total anchor-error bound `B`, explicit common-bias sub-bound `C <= B`, a justified combination rule, method-specific coverage assumptions, score/anchor conditional-independence evidence, and the preallocated alpha split. For the v1 conservative combination, the inspector also checks `B >= random_coverage_radius + C`.",
        "- Minimum effect `delta_star` with units and rationale, plus an independently reviewed physical reference/model-identification gate.",
        "- Even a structurally complete contract remains subject to human evidence review and separate data/GPU authorization. No fixture changes readiness or emits a score.",
        "",
    ])
    (out / "README.md").write_text("\n".join(lines))
    protocol = {
        "schema": SCHEMA,
        "status": "STRUCTURE_DIAGNOSTIC_ONLY",
        "input_contract_sha256": _sha256_file(CONTRACT_PATH),
        "data_access": "none",
        "physical_score": None,
        "certified": False,
        "fixtures": list(fixtures),
        "synthetic_fixture_values_are_physical": False,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    snapshots = {
        "snapshot_bounded_block_risk_contract.py": Path(__file__),
        "snapshot_bounded_block_risk_contract_v1.json": CONTRACT_PATH,
        "snapshot_bounded_block_risk_score.py": Path(__file__).with_name("bounded_block_risk_score.py"),
    }
    for name, source in snapshots.items():
        (out / name).write_bytes(source.read_bytes())
    payloads = [
        "README.md", "contract_diagnostics.json", "contract_summary.csv",
        "protocol.json", *snapshots,
    ]
    hashes = {name: _sha256_file(out / name) for name in payloads}
    (out / "artifact_hashes.json").write_text(json.dumps({
        "algorithm": "sha256", "files": hashes,
    }, indent=2) + "\n")
    counts = {
        key: sum(row[key] for row in rows)
        for key in ("planned", "valid", "failed", "abstained")
    }
    return {
        "status": "STRUCTURE_DIAGNOSTIC_ONLY",
        "physical_score": None,
        "fixture_results": results,
        "denominators": counts,
        "files": payloads + ["artifact_hashes.json"],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = build_report(args.output)
    print(json.dumps({
        "status": result["status"],
        "physical_score": result["physical_score"],
        "denominators": result["denominators"],
        "fixture_statuses": {
            name: value["status"] for name, value in result["fixture_results"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
