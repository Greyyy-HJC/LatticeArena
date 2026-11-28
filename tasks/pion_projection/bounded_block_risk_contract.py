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

from core.receipts import run_receipt, scenario_spec
from core.state import RunState
from .bounded_block_risk_anchor_markov import markov_anchor_bound


SCHEMA = "pion-bounded-block-risk-contract-v1"
DEFAULT_OUT = (Path(__file__).resolve().parents[2]
               / "results/pion_projection/bounded_block_risk_contract_diagnostic_20261126_v15")
CONTRACT_PATH = Path(__file__).with_name("bounded_block_risk_contract_v1.json")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEPENDENCE_METHODS = {
    "graph_color_hoeffding",
    "markov_pseudo_gap_bernstein",
    "markov_time_varying_bernstein",
    "markov_time_varying_support_gap_bernstein",
}
ANCHOR_METHODS = {
    "known_sigma_normal_interval",
    "bounded_iid_hoeffding",
    "markov_pseudo_gap_bernstein",
}
FAILURE_POLICY = "failed_block=-1; all_failed_comparison=abstain"
OPEN_RUN_GATES = {
    "operator": "candidate scores are not physically interpretable until an independent m_star reference passes",
    "reference": "candidate-blind E-chain reference measurement and independent review receipts are missing",
    "numerical": "anchor feature supports and certified PP/A4P fit-image calculation are not available",
    "statistical": "score-chain dependence and anchor coverage/common-bias receipts are missing",
    "comparison": "authorized fresh confirmation inputs and score-family evaluation are not available",
}


def open_gate_run_state() -> dict[str, Any]:
    """Project current evidence blockers into the generic no-certificate state."""
    state = RunState()
    for stage, reason in OPEN_RUN_GATES.items():
        state.abstain(stage, reason)
    return state.summary()


def selected_protocol() -> dict[str, Any]:
    """Return the selected method defaults, keeping every empirical receipt open."""
    return {
        "protocol_id": "pion-bounded-block-risk-selected-v1",
        "status": "METHOD_DEFAULTS_SELECTED_EVIDENCE_NOT_READY",
        "target": {
            "observable": "lowest zero-momentum valence isovector pseudoscalar energy",
            "units": "a_t m_pi",
            "ensemble": "S16T16 quenched Wilson, kappa=0.14",
            "m_star_measurement": None,
        },
        "reference_fit": {
            "channels": ["PP_n0", "PP_n8", "PP_n16", "local_A4P"],
            "candidate_blind": True,
            "shared_states": 2,
            "fit_window": [3, 8],
            "tmin_checks": [2, 3, 4],
            "state_count_checks": [2, 3],
            "covariance_diagonal_shrinkage": 0.25,
            "covariance_eigenvalue_floor": 1e-10,
            "folding": {"PP": "even", "A4P": "odd"},
            "pass_gates": {
                "relative_mass_error_max": 0.05,
                "p_value_min": 0.01,
                "window_model_shift_max_paired_se": 2.5,
                "require_PP_A4P_compatibility": True,
                "forbid_parameter_boundary_hits": True,
            },
            "measurement_receipt_sha256": None,
            "independent_review_receipt_sha256": None,
        },
        "score": {
            "prediction_domain": [0.1, 3.0],
            "units": "a_t m_pi",
            "R": 2.9,
            "prediction_difference_abs_bound": 2.9,
            "formula": "clip(((A-y_ref)^2-(A-y_candidate)^2)/R^2, -1, 1)",
            "clipping_interval": [-1, 1],
            "failure_policy": FAILURE_POLICY,
            "delta_star": 0.4205,
            "delta_star_units": "(a_t m_pi)^2",
            "normalized_effect_threshold": 0.05,
        },
        "comparison_family": {
            "ids": ["candidate_vs_local_identity", "candidate_vs_tuned_H32",
                    "candidate_vs_frozen_signed_SLSQP"],
            "F": 3,
            "alpha_total": 0.05,
            "alpha_score": 0.04,
            "alpha_anchor": 0.01,
        },
        "dependence_design": {
            "method": "markov_pseudo_gap_bernstein",
            "fixed_score_map_across_blocks": True,
            "block_length_configurations": 5,
            "planned_scored_blocks": 4000,
            "gamma_ps_lower_bound_requirement": 0.20,
            "burn_in_blocks": 8,
            "post_burn_in_tv_bound_per_comparison": 0.001,
            "evidence_receipt_sha256": None,
            "block_chain_receipt_sha256": None,
            "gamma_ps_receipt_sha256": None,
            "initial_law_tv_receipt_sha256": None,
        },
        "anchor_design": {
            "method": "markov_pseudo_gap_bernstein",
            "planned_anchor_units": 4000,
            "gamma_ps_lower_bound_requirement": 0.20,
            "burn_in_units": 8,
            "post_burn_in_tv_bound": 0.001,
            "support_interval": [0.1, 3.0],
            "alpha": 0.01,
            "coverage_radius_formula": "markov_gap_radius(n_A=4000,gamma_ps=0.20,family=1,alpha_stationary=0.009,variance=(3.0-0.1)^2/4,centered_abs_bound=3.0-0.1)",
            "common_bias_bound_C_max_over_R": 0.01,
            "total_error_B_over_R_formula": "(random_coverage_radius+0.029)/2.9",
            "coverage_receipt_sha256": None,
            "common_bias_receipt_sha256": None,
            "score_anchor_independence_receipt_sha256": None,
            "shared_anchor_interval_family": 1,
            "state_map": "UNRESOLVED: no per-configuration pion-mass proxy is accepted; the ensemble-averaged nonlinear fit is not a scalar state map",
            "receipts_required": [
                "anchor_scalar_map_receipt_sha256",
                "anchor_chain_receipt_sha256",
                "gamma_ps_lower_receipt_sha256",
                "initial_law_tv_receipt_sha256",
                "common_bias_coverage_receipt_sha256",
                "conditional_independence_receipt_sha256",
            ],
            "iid_fallback": {
                "method": "bounded_iid_hoeffding",
                "independent_stationary_units_required": 1000,
                "support_interval": [0.1, 3.0],
                "alpha": 0.01,
                "precondition": "Use only if every anchor unit is receipt-proven IID from the target stationary distribution and the same fixed scalar-map/support/common-bias bridge is reviewed",
            },
        },
        "gates": {
            "production_data_authorized": False,
            "gpu_quda_authorized": False,
            "sealed_confirmation_open": False,
            "physical_score": None,
        },
    }


def empty_contract() -> dict[str, Any]:
    """Return a draft with selected method defaults and no empirical receipts."""
    return {
        "schema": SCHEMA,
        "status": "DRAFT_NOT_READY",
        "physical_score": None,
        "certified": False,
        "data_access": "none",
        "authorization": {"production_data": False, "gpu_quda": False},
        "selected_protocol": selected_protocol(),
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
            "gamma_ps_lower_receipt_sha256": None,
            "pseudo_gap_k_ps": None,
            "pseudo_gap_k_ps_receipt_sha256": None,
            "block_chain_assumptions": None,
            "block_chain_receipt_sha256": None,
            "score_function_time_homogeneous": None,
            "score_schedule_receipt_sha256": None,
            "variance_upper_by_residue": None,
            "variance_upper_receipt_sha256": None,
            "initial_law_mode": None,
            "burn_in_blocks": None,
            "initial_law_tv_bound": None,
            "initial_law_receipt_sha256": None,
        },
        "anchor": {
            "estimate_receipt_sha256": None,
            "target_match_statement": None,
            "units": None,
            "total_error_bound_B": None,
            "common_bias_bound_C": None,
            "bound_combination_rule": None,
            "time_varying_score_shift_bound": None,
            "time_varying_score_shift_receipt_sha256": None,
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
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


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


def _check_dependence(dependence: Any, *, planned_blocks: Any,
                      comparison_family: Any, alpha_score: Any) -> list[str]:
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
    elif method in {
        "markov_pseudo_gap_bernstein",
        "markov_time_varying_bernstein",
        "markov_time_varying_support_gap_bernstein",
    }:
        gamma = dependence.get("gamma_ps_lower_bound")
        if not _finite_number(gamma) or not 0 < gamma <= 1:
            issues.append(f"{label}.gamma_ps_lower_bound: expected a value in (0, 1]")
        if not _nonempty(dependence.get("block_chain_assumptions")):
            issues.append(f"{label}.block_chain_assumptions: missing explicit chain assumptions")
        issues.extend(_required_receipt(
            dependence, "block_chain_receipt_sha256", label,
        ))
        issues.extend(_required_receipt(
            dependence, "gamma_ps_lower_receipt_sha256", label,
        ))
        homogeneous = dependence.get("score_function_time_homogeneous")
        if method == "markov_pseudo_gap_bernstein":
            if homogeneous is not True:
                issues.append(
                    f"{label}.score_function_time_homogeneous: fixed-score route requires one fixed score function across blocks"
                )
        else:
            if homogeneous is not False:
                issues.append(
                    f"{label}.score_function_time_homogeneous: time-varying route requires an explicitly varying score schedule"
                )
            if not _positive_int(planned_blocks):
                issues.append(f"{label}.planned_blocks_N: must be positive for the time-varying Markov route")
            issues.extend(_required_receipt(
                dependence, "score_schedule_receipt_sha256", label,
            ))
            if method == "markov_time_varying_bernstein":
                k_ps = dependence.get("pseudo_gap_k_ps")
                if not _positive_int(k_ps):
                    issues.append(f"{label}.pseudo_gap_k_ps: expected a positive integer")
                elif _positive_int(planned_blocks) and k_ps > planned_blocks:
                    issues.append(f"{label}.pseudo_gap_k_ps: cannot exceed planned_blocks_N")
                issues.extend(_required_receipt(
                    dependence, "pseudo_gap_k_ps_receipt_sha256", label,
                ))
                variance_bounds = dependence.get("variance_upper_by_residue")
                if not isinstance(variance_bounds, list):
                    issues.append(f"{label}.variance_upper_by_residue: expected a list")
                elif _positive_int(k_ps):
                    if len(variance_bounds) != k_ps:
                        issues.append(f"{label}.variance_upper_by_residue: length must equal pseudo_gap_k_ps")
                    elif any(not _finite_number(value) or value <= 0
                             for value in variance_bounds):
                        issues.append(f"{label}.variance_upper_by_residue: every bound must be finite and strictly positive")
                issues.extend(_required_receipt(
                    dependence, "variance_upper_receipt_sha256", label,
                ))
        initial_mode = dependence.get("initial_law_mode")
        if (not isinstance(initial_mode, str)
                or initial_mode not in {"stationary", "post_burn_in_tv"}):
            issues.append(
                f"{label}.initial_law_mode: expected 'stationary' or 'post_burn_in_tv'"
            )
        burn_in = dependence.get("burn_in_blocks")
        if (not isinstance(burn_in, int) or isinstance(burn_in, bool)
                or burn_in < 0):
            issues.append(f"{label}.burn_in_blocks: expected a nonnegative integer")
        tv_bound = dependence.get("initial_law_tv_bound")
        if not _finite_number(tv_bound) or not 0 <= tv_bound <= 1:
            issues.append(f"{label}.initial_law_tv_bound: expected a finite value in [0, 1]")
        issues.extend(_required_receipt(
            dependence, "initial_law_receipt_sha256", label,
        ))
        if initial_mode == "stationary":
            if isinstance(burn_in, int) and not isinstance(burn_in, bool) and burn_in != 0:
                issues.append(f"{label}.burn_in_blocks: stationary mode requires zero burn-in")
            if _finite_number(tv_bound) and tv_bound != 0:
                issues.append(f"{label}.initial_law_tv_bound: stationary mode requires zero TV bound")
        elif initial_mode == "post_burn_in_tv":
            if (_finite_number(tv_bound) and 0 <= tv_bound <= 1
                    and _positive_int(comparison_family)
                    and _finite_number(alpha_score)):
                try:
                    familywise_tv_cost = comparison_family * tv_bound
                except OverflowError:
                    familywise_tv_cost = math.inf
                if familywise_tv_cost >= alpha_score:
                    issues.append(
                        f"{label}.initial_law_tv_bound: F * delta must be smaller than inference.alpha_score"
                    )
    return issues


def _check_anchor(anchor: Any, *, target_units: Any,
                  alpha_anchor: Any,
                  dependence_method: Any,
                  score_scale_R: Any) -> list[str]:
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
    if dependence_method in {
        "markov_time_varying_bernstein",
        "markov_time_varying_support_gap_bernstein",
    }:
        shift = anchor.get("time_varying_score_shift_bound")
        if not _finite_number(shift) or shift < 0:
            issues.append(
                f"{label}.time_varying_score_shift_bound: expected a finite nonnegative normalized-score shift bound"
            )
        elif (_finite_number(total_bound) and total_bound >= 0
              and _finite_number(score_scale_R) and score_scale_R > 0):
            try:
                fixed_score_floor = 2.0 * total_bound / score_scale_R
            except OverflowError:
                fixed_score_floor = math.inf
            if not math.isfinite(fixed_score_floor) or shift + 1e-12 < fixed_score_floor:
                issues.append(
                    f"{label}.time_varying_score_shift_bound: must cover the fixed paired-squared-error floor 2B/R"
                )
        issues.extend(_required_receipt(
            anchor, "time_varying_score_shift_receipt_sha256", label,
        ))
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
    elif method == "markov_pseudo_gap_bernstein":
        sample_count = parameters.get("sample_count_n_A")
        if not _positive_int(sample_count):
            issues.append(f"{label}.coverage_parameters.sample_count_n_A: expected a positive integer")
        support = parameters.get("support_interval")
        support_valid = (isinstance(support, list) and len(support) == 2
                         and all(_finite_number(value) for value in support)
                         and support[0] < support[1])
        if not support_valid:
            issues.append(f"{label}.coverage_parameters.support_interval: expected finite increasing bounds")
        gamma = parameters.get("gamma_ps_lower_bound")
        if not _finite_number(gamma) or not 0 < gamma <= 1:
            issues.append(f"{label}.coverage_parameters.gamma_ps_lower_bound: expected a value in (0, 1]")
        if not _nonempty(parameters.get("anchor_scalar_map_statement")):
            issues.append(f"{label}.coverage_parameters.anchor_scalar_map_statement: define one fixed bounded scalar per chain state")
        if not _nonempty(parameters.get("anchor_chain_assumptions")):
            issues.append(f"{label}.coverage_parameters.anchor_chain_assumptions: missing Markov-chain assumptions")
        for field in (
            "support_receipt_sha256",
            "anchor_scalar_map_receipt_sha256",
            "anchor_chain_receipt_sha256",
            "gamma_ps_lower_receipt_sha256",
            "initial_law_receipt_sha256",
        ):
            if not _sha256(parameters.get(field)):
                issues.append(f"{label}.coverage_parameters.{field}: expected a SHA-256 receipt")
        burn_in = parameters.get("burn_in_units")
        if (isinstance(burn_in, bool) or not isinstance(burn_in, int) or burn_in < 0):
            issues.append(f"{label}.coverage_parameters.burn_in_units: expected a nonnegative integer")
        initial_mode = parameters.get("initial_law_mode")
        if initial_mode not in {"stationary", "post_burn_in_tv"}:
            issues.append(f"{label}.coverage_parameters.initial_law_mode: expected 'stationary' or 'post_burn_in_tv'")
        tv_bound = parameters.get("initial_law_tv_bound")
        if not _finite_number(tv_bound) or not 0 <= tv_bound <= 1:
            issues.append(f"{label}.coverage_parameters.initial_law_tv_bound: expected a value in [0, 1]")
        if initial_mode == "stationary":
            if isinstance(burn_in, int) and not isinstance(burn_in, bool) and burn_in != 0:
                issues.append(f"{label}.coverage_parameters.burn_in_units: stationary mode requires zero burn-in")
            if _finite_number(tv_bound) and tv_bound != 0:
                issues.append(f"{label}.coverage_parameters.initial_law_tv_bound: stationary mode requires zero TV")
        elif initial_mode == "post_burn_in_tv":
            if isinstance(burn_in, int) and not isinstance(burn_in, bool) and burn_in < 1:
                issues.append(f"{label}.coverage_parameters.burn_in_units: post-burn-in mode requires positive burn-in")
            if (not _finite_number(alpha_anchor) or not _finite_number(tv_bound)
                    or tv_bound >= alpha_anchor):
                issues.append(f"{label}.coverage_parameters.initial_law_tv_bound: must be smaller than alpha_anchor")
        if (not _finite_number(alpha_anchor) or alpha_anchor <= 0):
            issues.append(f"{label}.coverage_parameters: alpha_anchor must be positive for a sampling interval")
        elif (_positive_int(sample_count) and support_valid
              and _finite_number(gamma) and 0 < gamma <= 1
              and _finite_number(tv_bound) and 0 <= tv_bound < alpha_anchor
              and _finite_number(common_bias) and common_bias >= 0):
            try:
                bound = markov_anchor_bound(
                    sample_count=sample_count,
                    gamma_ps_lower=gamma,
                    alpha_anchor=alpha_anchor,
                    support=(float(support[0]), float(support[1])),
                    common_bias_bound=float(common_bias),
                    initial_law_tv_bound=float(tv_bound),
                    burn_in_units=burn_in,
                    stationary_start=initial_mode == "stationary",
                    score_scale_R=float(score_scale_R),
                )
                random_coverage_radius = bound["random_coverage_radius"]
            except (TypeError, ValueError, OverflowError):
                issues.append(f"{label}.coverage_parameters: invalid Markov anchor-bound inputs")
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
    protocol = contract.get("selected_protocol")
    if not _mapping(protocol):
        issues.append("selected_protocol: missing selected method defaults")
    else:
        if protocol != selected_protocol():
            issues.append("selected_protocol: differs from the frozen method defaults or evidence-open fields")
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
    inference = contract.get("inference")
    comparison_family = inference.get("F") if _mapping(inference) else None
    alpha_score = inference.get("alpha_score") if _mapping(inference) else None
    issues.extend(_check_dependence(
        contract.get("dependence"), planned_blocks=planned_blocks,
        comparison_family=comparison_family, alpha_score=alpha_score,
    ))
    target = estimand.get("target") if _mapping(estimand) else None
    target_units = target.get("units") if _mapping(target) else None
    alpha_anchor = inference.get("alpha_anchor") if _mapping(inference) else None
    issues.extend(_check_anchor(
        contract.get("anchor"), target_units=target_units,
        alpha_anchor=alpha_anchor,
        dependence_method=(contract.get("dependence", {}).get("method")
                           if _mapping(contract.get("dependence")) else None),
        score_scale_R=estimand.get("score_scale_R") if _mapping(estimand) else None,
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
            "gamma_ps_lower_receipt_sha256": _synthetic_receipt("gamma-ps-lower-bound"),
            "block_chain_assumptions": "SYNTHETIC_ONLY stationary block-chain assumptions",
            "block_chain_receipt_sha256": _synthetic_receipt("block-chain"),
            "score_function_time_homogeneous": True,
            "initial_law_mode": "stationary",
            "burn_in_blocks": 0,
            "initial_law_tv_bound": 0.0,
            "initial_law_receipt_sha256": _synthetic_receipt("stationary-initial-law"),
        })
    elif method == "markov_time_varying_bernstein":
        dependence.update({
            "gamma_ps_lower_bound": 0.25,
            "gamma_ps_lower_receipt_sha256": _synthetic_receipt("gamma-ps-lower-bound"),
            "pseudo_gap_k_ps": 2,
            "pseudo_gap_k_ps_receipt_sha256": _synthetic_receipt("pseudo-gap-k-ps"),
            "block_chain_assumptions": "SYNTHETIC_ONLY stationary block-chain assumptions",
            "block_chain_receipt_sha256": _synthetic_receipt("block-chain"),
            "score_function_time_homogeneous": False,
            "score_schedule_receipt_sha256": _synthetic_receipt("score-schedule"),
            "variance_upper_by_residue": [0.2, 0.2],
            "variance_upper_receipt_sha256": _synthetic_receipt("class-variance-bounds"),
            "initial_law_mode": "stationary",
            "burn_in_blocks": 0,
            "initial_law_tv_bound": 0.0,
            "initial_law_receipt_sha256": _synthetic_receipt("stationary-initial-law"),
        })
    elif method == "markov_time_varying_support_gap_bernstein":
        dependence.update({
            "gamma_ps_lower_bound": 0.25,
            "gamma_ps_lower_receipt_sha256": _synthetic_receipt("gamma-ps-lower-bound"),
            "block_chain_assumptions": "SYNTHETIC_ONLY stationary Markov block-chain assumptions",
            "block_chain_receipt_sha256": _synthetic_receipt("block-chain"),
            "score_function_time_homogeneous": False,
            "score_schedule_receipt_sha256": _synthetic_receipt("score-schedule"),
            "initial_law_mode": "stationary",
            "burn_in_blocks": 0,
            "initial_law_tv_bound": 0.0,
            "initial_law_receipt_sha256": _synthetic_receipt("stationary-initial-law"),
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
    if method == "markov_time_varying_bernstein":
        anchor.update({
            "time_varying_score_shift_bound": 0.30,
            "time_varying_score_shift_receipt_sha256": _synthetic_receipt("time-varying-anchor-shift"),
        })
    elif method == "markov_time_varying_support_gap_bernstein":
        anchor.update({
            "time_varying_score_shift_bound": 0.30,
            "time_varying_score_shift_receipt_sha256": _synthetic_receipt("time-varying-anchor-shift"),
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
    valid_markov_varying = _synthetic_complete_contract("markov_time_varying_bernstein")
    valid_markov_support_gap = _synthetic_complete_contract(
        "markov_time_varying_support_gap_bernstein",
    )
    valid_markov_anchor = _synthetic_complete_contract()
    valid_markov_anchor["anchor"].update({
        "total_error_bound_B": 0.30,
        "coverage_method": "markov_pseudo_gap_bernstein",
        "coverage_parameters": {
            "sample_count_n_A": 4000,
            "support_interval": [-0.5, 0.5],
            "gamma_ps_lower_bound": 0.25,
            "initial_law_mode": "stationary",
            "burn_in_units": 0,
            "initial_law_tv_bound": 0.0,
            "anchor_scalar_map_statement": "SYNTHETIC_ONLY one fixed bounded scalar per anchor Markov state",
            "anchor_chain_assumptions": "SYNTHETIC_ONLY homogeneous chain with unique stationary law",
            "support_receipt_sha256": _synthetic_receipt("markov-anchor-support"),
            "anchor_scalar_map_receipt_sha256": _synthetic_receipt("markov-anchor-scalar-map"),
            "anchor_chain_receipt_sha256": _synthetic_receipt("markov-anchor-chain"),
            "gamma_ps_lower_receipt_sha256": _synthetic_receipt("markov-anchor-gap"),
            "initial_law_receipt_sha256": _synthetic_receipt("markov-anchor-stationary-law"),
        },
    })
    missing_anchor_map = json.loads(json.dumps(valid_markov_anchor))
    missing_anchor_map["anchor"]["coverage_parameters"]["anchor_scalar_map_receipt_sha256"] = None
    anchor_alpha_spent = json.loads(json.dumps(valid_markov_anchor))
    anchor_alpha_spent["anchor"]["coverage_parameters"].update({
        "initial_law_mode": "post_burn_in_tv",
        "burn_in_units": 8,
        "initial_law_tv_bound": 0.01,
    })
    valid_markov_tv = _synthetic_complete_contract("markov_pseudo_gap_bernstein")
    valid_markov_tv["dependence"].update({
        "block_chain_assumptions": "SYNTHETIC_ONLY homogeneous Markov block chain with unique stationary law",
        "initial_law_mode": "post_burn_in_tv",
        "burn_in_blocks": 8,
        "initial_law_tv_bound": 0.005,
        "initial_law_receipt_sha256": _synthetic_receipt("post-burn-in-tv"),
    })
    alpha_over = _synthetic_complete_contract()
    alpha_over["inference"]["alpha_score"] = 0.05
    bound_over = _synthetic_complete_contract()
    bound_over["estimand"]["prediction_difference_abs_bound"] = 2.1
    missing_anchor = _synthetic_complete_contract()
    missing_anchor["anchor"]["common_bias_coverage_receipt_sha256"] = None
    missing_initial_law = _synthetic_complete_contract("markov_pseudo_gap_bernstein")
    missing_initial_law["dependence"]["initial_law_receipt_sha256"] = None
    tv_over_budget = _synthetic_complete_contract("markov_pseudo_gap_bernstein")
    tv_over_budget["dependence"].update({
        "initial_law_mode": "post_burn_in_tv",
        "burn_in_blocks": 8,
        "initial_law_tv_bound": 0.02,
        "initial_law_receipt_sha256": _synthetic_receipt("post-burn-in-tv-over-budget"),
    })
    changing_score = _synthetic_complete_contract("markov_pseudo_gap_bernstein")
    changing_score["dependence"]["score_function_time_homogeneous"] = False
    missing_varying_receipt = _synthetic_complete_contract("markov_time_varying_bernstein")
    missing_varying_receipt["dependence"]["score_schedule_receipt_sha256"] = None
    invalid_varying_variance = _synthetic_complete_contract("markov_time_varying_bernstein")
    invalid_varying_variance["dependence"]["variance_upper_by_residue"] = [0.2, 0.0]
    underbounded_varying_anchor = _synthetic_complete_contract("markov_time_varying_bernstein")
    underbounded_varying_anchor["anchor"]["time_varying_score_shift_bound"] = 0.1
    missing_markov_gamma_receipt = _synthetic_complete_contract(
        "markov_pseudo_gap_bernstein",
    )
    missing_markov_gamma_receipt["dependence"]["gamma_ps_lower_receipt_sha256"] = None
    fixtures = {
        "current_null_contract": current,
        "synthetic_graph_color_complete": valid_graph,
        "synthetic_markov_gap_complete": valid_markov,
        "synthetic_markov_time_varying_complete": valid_markov_varying,
        "synthetic_markov_time_varying_support_gap_complete": valid_markov_support_gap,
        "synthetic_markov_anchor_complete": valid_markov_anchor,
        "synthetic_markov_anchor_missing_scalar_map_receipt": missing_anchor_map,
        "synthetic_markov_anchor_tv_spends_alpha": anchor_alpha_spent,
        "synthetic_markov_post_burn_in_tv_complete": valid_markov_tv,
        "synthetic_alpha_over_budget": alpha_over,
        "synthetic_prediction_bound_over_R": bound_over,
        "synthetic_missing_anchor_coverage": missing_anchor,
        "synthetic_missing_initial_law_receipt": missing_initial_law,
        "synthetic_initial_tv_spends_score_alpha": tv_over_budget,
        "synthetic_time_varying_score_function": changing_score,
        "synthetic_time_varying_missing_schedule_receipt": missing_varying_receipt,
        "synthetic_time_varying_zero_class_variance": invalid_varying_variance,
        "synthetic_time_varying_anchor_shift_under_2B_over_R": underbounded_varying_anchor,
        "synthetic_markov_missing_gamma_receipt": missing_markov_gamma_receipt,
    }
    results = {name: inspect_contract(value) for name, value in fixtures.items()}
    run_state = open_gate_run_state()
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
        "run_state": run_state,
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
        "The selected method defaults are recorded in `bounded_block_risk_contract_v1.json`: target `a_t m_pi`, `R=2.9`, `delta_star/R^2=0.05`, three comparisons with `alpha_score=0.04` and `alpha_anchor=0.01`, length-5 score blocks with `N=4000`, score-chain `gamma_ps>=0.20`, burn-in 8 and per-comparison TV `<=0.001`; the primary anchor route uses a separate Markov chain with `n_A=4000`, `gamma_A>=0.20`, burn-in 8, TV `<=0.001`, and a separately evidenced common-bias limit. The fixed bounded anchor scalar map remains unresolved, so no anchor interval can yet be evaluated. The 1000-unit bounded-IID Hoeffding design is conditional fallback only. All are planning choices, not measurements or receipts.",
        "",
        "## Selected protocol flow",
        "",
        "```text",
        "independent candidate-blind E-chain joint fit -> pass mass/model/folding gates -> freeze bounded score and F=3 family",
        "  -> review score-specific length-5 block-chain, gamma_ps lower bound, and burn-in/TV receipts",
        "  -> review the independent anchor Markov-chain scalar map, gamma_ps/TV receipts, support, common-bias C, and score/anchor independence",
        "  -> use IID Hoeffding only if 1000 truly IID stationary units and the same scalar-map/support/C bridge are independently receipted",
        "  -> only then evaluate the full N=4000 planned-block LCB; retain failed slots as -1",
        "  -> require normalized LCB > 0.05; otherwise ABSTAIN; physical_score remains null until all gates are separately authorized",
        "```",
        "",
        "## Current no-data RunState projection",
        "",
        "The generic `RunState` projection records all five physical-certification stages as `abstained` with explicit evidence blockers in `contract_diagnostics.json`. It reports `physical_ready=false`, `score=null`, and planned/valid/failed/abstained=`5/0/0/5`. This counts missing evidence and authorization, not failed physical measurements; every future authorized run must start with a fresh state.",
        "",
        "|stage|state|current blocker|",
        "|---|---|---|",
    ]
    for stage in ("operator", "reference", "numerical", "statistical", "comparison"):
        lines.append(f"|{stage}|{run_state['status'][stage]}|{run_state['evidence'][stage][0]}|")
    lines.extend([
        "",
        "The inspector checks whether a future contract contains the fields required by the v1 estimand: bounded paired squared-error scores, a score-specific dependence route, anchor coverage/common-bias evidence, preallocated alpha, and a separate physical reference gate.",
        "",
        "The fixed-score Markov route follows Paulin arXiv v5 Theorem 3.11 / Eq. (3.25). The optional time-varying routes follow Eq. (3.26) for predeclared `f_i(X_i)` and `theta_N=N^-1 sum_i E_pi[f_i(X_i)]`; a human must review whether this schedule preserves the accepted target. The exact-`k_ps` route requires the smallest maximizing `k_ps`, a receipt for that choice, positive upper bounds for every residue-class variance sum, a variance receipt, and a score-schedule receipt. A second support-only route uses `Var_pi(f_i)<=1` for scores in `[-1,1]` plus a separately receipted positive pseudo-gap lower bound; it removes the exact-`k_ps` and per-class variance receipts by using `V_sum<=N+ceil(1/gamma_ps_lower)-1` and `M/k_ps=1` after uniform residue-bound inflation. This may be conservative and does not validate the gap or chain assumptions. Both routes require review of the schedule-specific finite-plan estimand. See [Paulin, arXiv:1212.2015v5](https://arxiv.org/html/1212.2015v5).",
        "",
        "For a planned family with `F` comparisons, a post-burn-in TV bound `delta` spends `F*delta` from `alpha_score`; the remaining stationary alpha enters the Eq. (3.26) radius. The LCB subtracts `2*delta` for the bounded-score mean shift and an externally justified anchor shift in normalized score units; for the existing paired-squared-error component the inspector requires this shift to cover at least `2B/R`. Zero class-variance bounds are not accepted: a strictly positive conservative upper bound is needed to keep `M` finite.",
        "",
        "## Text decision flow",
        "",
        "```text",
        "load the null v1 contract; do not load measurements",
        "  -> check estimand, R/prediction bound, clipping, failure policy, and planned-block receipt",
        "  -> require a score-specific block map and dependence-evidence receipt",
        "  -> graph-color route: prove color-class independence and provide its receipt",
        "  -> every Markov route: provide block-chain and gamma_ps-lower-bound receipts",
        "  -> fixed-score Markov: freeze one score function; time-varying Markov: receipt the full score schedule",
        "  -> exact-k Eq. (3.26): check k_ps/argmax receipt, positive variance upper bound per residue, variance receipt, and score-schedule receipt",
        "  -> support-only Eq. (3.26): check [-1,1] score support, block-chain/gamma lower-bound receipts, and score-schedule receipt; exact k_ps and class-variance receipts are unnecessary",
        "  -> for the Markov route, require stationary-start evidence or post-burn-in TV evidence and F*delta < alpha_score",
        "  -> require matched anchor, bias/coverage/common-bias, and conditional-independence receipts",
        "  -> check comparison family F and alpha_score + alpha_anchor <= alpha_total",
        "  -> check independent physical reference/model-identification gate",
        "  -> missing/malformed field: NOT_READY with exact paths",
        "  -> complete synthetic shape: CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW only",
        "  -> physical_score remains null throughout",
        "```",
        "",
        "For the two-sided known-sigma normal route, the random radius is `z_(1-alpha_A/2) * sigma / sqrt(n_A)`; for bounded iid Hoeffding it is `(U-L) * sqrt(log(2/alpha_A) / (2*n_A))`. The Markov anchor route applies corrected Theorem 3.11 to one fixed scalar per anchor-chain state, with worst-case variance `(U-L)^2/4`, centered bound `U-L`, and that anchor chain's own pseudo-gap. A post-burn-in TV bound spends `delta_A` from the single shared anchor alpha, so the stationary Bernstein radius uses `alpha_A-delta_A`. Each route requires total `B >= random_coverage_radius + C`, where `C` bounds common bias.",
        "",
        "Receipts are checked for presence and SHA-256 syntax only; their scientific contents and independence are not verified. `valid` below means structurally complete, not evidence-valid, measured, or certified. Each row counts one contract check; it is not a score block or physical sample.",
        "",
        f"`run_receipt.json` records these {sum(row['planned'] for row in rows)} synthetic contract checks plus a separate five-stage physical `RunState` denominator of `5/0/0/5` (planned/valid/failed/abstained). Each incomplete fixture and the five abstained evidence gates have explicit reasons. Receipt `complete` means all planned units are accounted for, not that any dependence, anchor, reference, coverage or physical claim succeeded. `artifact_hashes.json` covers the run receipt and all report payloads; source hashes bind the inspector, null contract, score helpers and shared receipt/state implementations.",
        "",
        "|fixture|planned|valid|failed|abstained|status|blockers|",
        "|---|---:|---:|---:|---:|---|---:|",
    ])
    for row in rows:
        lines.append(
            f"|{row['fixture']}|{row['planned']}|{row['valid']}|{row['failed']}|{row['abstained']}|{row['status']}|{row['issue_count']}|"
        )
    lines.extend([
        "",
        "## Remaining evidence receipts after method defaults were selected",
        "",
        "- The target observable, units, score transform, `R`, clipping, `delta_star`, `F=3`, and alpha allocation are selected. The fitted numerical `m_star` remains unknown until the independent candidate-blind E-chain fit passes its fixed model/window gates and yields measurement plus review receipts.",
        "- For the selected length-5 score block chain, provide authorized block IDs/provenance, a score-specific pseudo-gap lower-bound receipt, stationary-start or post-burn-in TV evidence, and the complete fixed score schedule. Require `F*delta < alpha_score`; do not infer independence from trajectory spacing or bootstrap count.",
        "- For the selected anchor Markov route, provide 4000 anchor-chain states, a fixed bounded scalar map and support proof, an independently justified anchor-chain `gamma_ps` lower bound, burn-in/post-burn-in TV evidence, common-bias `C` bound, and score/anchor conditional-independence review. The 1000-unit IID Hoeffding route is an alternative only with receipt-proven IID stationary units and the same scalar-map/support/C bridge.",
        "- For the vector route, additionally prove finite support for every PP/A4P feature, fix or simultaneously cover covariance/regularization inputs, and certify a full-box interval image of the two-state PP/A4P fit over every global minimizer. An unbounded, non-unique, boundary-hitting or unclosed fit image abstains; a vector mean box alone does not identify `m_star`.",
        "- Complete the reference gate, anchor/common-bias review and score-chain receipts before any physical LCB. New confirmation inputs, data access and GPU/QUDA each require their corresponding owner authorization. No structural fixture or RunState receipt grants those permissions or emits a score.",
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
        "run_state": run_state,
        "selected_protocol": selected_protocol(),
        "fixtures": list(fixtures),
        "synthetic_fixture_values_are_physical": False,
        "receipt": {
            "schema": "latticearena-run-receipt-v1",
            "unit": "contract fixture checks plus a separately counted physical-gate projection",
            "complete_means": "all planned structural checks and abstained evidence gates are accounted for; not scientific success",
        },
        "markov_theorem_route": {
            "stationary": "Paulin arXiv:1212.2015v5 Theorem 3.11 / Eq. (3.25), one fixed score function f",
            "nonstationary": "Proposition 3.15 / Eq. (3.29), requires d_TV(q P^t0, pi) <= delta",
            "time_varying_scores_exact_k": "Eq. (3.26), requires k_ps maximizer receipt, positive variance upper bound per residue, score schedule receipt, and explicit M/k_ps factor",
            "time_varying_scores_support_gap": "Eq. (3.26) with f_i in [-1,1], gamma_ps lower-bound receipt, uniform V'_r=ceil(N/k_ps), M/k_ps=1, and V_sum<=N+ceil(1/gamma_ps_lower)-1",
            "markov_anchor": "Theorem 3.11 / Eq. (3.25) for a fixed bounded scalar per anchor-chain state; Proposition 3.15 / Eq. (3.29) spends post-burn-in TV from the shared anchor alpha",
        },
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    snapshots = {
        "snapshot_bounded_block_risk_contract.py": Path(__file__),
        "snapshot_bounded_block_risk_contract_v1.json": CONTRACT_PATH,
        "snapshot_bounded_block_risk_score.py": Path(__file__).with_name("bounded_block_risk_score.py"),
        "snapshot_bounded_block_risk_markov_varying.py": Path(__file__).with_name("bounded_block_risk_markov_varying.py"),
        "snapshot_bounded_block_risk_anchor_markov.py": Path(__file__).with_name("bounded_block_risk_anchor_markov.py"),
        "snapshot_core_receipts.py": Path(__file__).resolve().parents[2] / "core/receipts.py",
        "snapshot_core_numerics.py": Path(__file__).resolve().parents[2] / "core/numerics.py",
        "snapshot_core_state.py": Path(__file__).resolve().parents[2] / "core/state.py",
    }
    for name, source in snapshots.items():
        (out / name).write_bytes(source.read_bytes())
    payloads = [
        "README.md", "contract_diagnostics.json", "contract_summary.csv",
        "protocol.json", *snapshots,
    ]
    source_hashes = {
        "tasks/pion_projection/bounded_block_risk_contract.py": _sha256_file(
            out / "snapshot_bounded_block_risk_contract.py"),
        "tasks/pion_projection/bounded_block_risk_contract_v1.json": _sha256_file(
            out / "snapshot_bounded_block_risk_contract_v1.json"),
        "tasks/pion_projection/bounded_block_risk_score.py": _sha256_file(
            out / "snapshot_bounded_block_risk_score.py"),
        "tasks/pion_projection/bounded_block_risk_markov_varying.py": _sha256_file(
            out / "snapshot_bounded_block_risk_markov_varying.py"),
        "tasks/pion_projection/bounded_block_risk_anchor_markov.py": _sha256_file(
            out / "snapshot_bounded_block_risk_anchor_markov.py"),
        "core/receipts.py": _sha256_file(out / "snapshot_core_receipts.py"),
        "core/numerics.py": _sha256_file(out / "snapshot_core_numerics.py"),
        "core/state.py": _sha256_file(out / "snapshot_core_state.py"),
    }
    scenario = scenario_spec(
        "pion-bounded-block-risk-contract-fixtures",
        "one contract fixture inspection",
        "Null draft and synthetic schema fixtures only; no correlator, anchor or sealed inputs.",
        assumptions=[
            "contract completeness is metadata shape only and requires independent human review",
            "fixtures are software checks and are not independent physical observations",
            "receipt hashes do not authenticate dependence, anchor coverage or reference evidence",
        ],
        source_hashes=source_hashes,
    )
    denominators = {}
    reason_rows = []
    for row in rows:
        name = row["fixture"]
        denominators[name] = {
            "unit": "one contract fixture inspection",
            "planned": row["planned"],
            "valid": row["valid"],
            "failed": row["failed"],
            "abstained": row["abstained"],
        }
        if row["failed"]:
            reason_rows.append({
                "unit": name,
                "state": "failed",
                "count": row["failed"],
                "reason": "; ".join(results[name]["issues"])
                or "structural validation failed",
            })
    denominators["physical_run_state_gates"] = {
        "unit": "one physical certification stage in core.RunState",
        "planned": len(OPEN_RUN_GATES),
        "valid": 0,
        "failed": 0,
        "abstained": len(OPEN_RUN_GATES),
    }
    reason_rows.append({
        "unit": "physical_run_state_gates",
        "state": "abstained",
        "count": len(OPEN_RUN_GATES),
        "reason": "physical Pion scoring awaits independent reference, anchor and dependence receipts plus authorization",
    })
    receipt_payload_hashes = {
        name: _sha256_file(out / name) for name in payloads
    }
    receipt = run_receipt(
        scenario, denominators, artifact_hashes=receipt_payload_hashes,
        text_flow=(
            "null v1 contract plus synthetic graph/Markov/anchor fixtures -> "
            "independent contract inspection per fixture -> explicit denominator "
            "plus five RunState abstentions -> hashed report; complete means accounting only, "
            "not scientific success; physical review and authorized data access "
            "remain later gates"
        ),
        reason_rows=reason_rows,
    )
    (out / "run_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    hashes = {
        name: _sha256_file(out / name) for name in payloads + ["run_receipt.json"]
    }
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
        "run_state": run_state,
        "fixture_results": results,
        "denominators": counts,
        "receipt": receipt,
        "files": payloads + ["run_receipt.json", "artifact_hashes.json"],
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
        "run_state": result["run_state"],
        "denominators": result["denominators"],
        "fixture_statuses": {
            name: value["status"] for name, value in result["fixture_results"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
