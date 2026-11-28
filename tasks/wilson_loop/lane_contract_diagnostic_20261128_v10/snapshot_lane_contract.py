"""Schema checks for the separate finite-T Creutz and ground-state-force lanes.

This module checks whether a prospective contract is structurally complete. It
does not validate evidence contents, read Wilson data, certify physics, or set
``physical_score``. Synthetic fixtures are provided only to exercise the schema.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from core.receipts import run_receipt, scenario_spec
from core.state import RunState


SCHEMA = "wilson-two-lane-contract-v2"
DEFAULT_OUT = (Path(__file__).resolve().parents[2]
               / "results/wilson_loop/lane_contract_diagnostic_20261125_v9")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONTRACT_PATH = Path(__file__).with_name("lane_contract_v2.json")
LATTICE_FORCE_UNITS = "dimensionless lattice force a_s*a_t*F"
F_BOUND_COMPONENTS = (
    "excited_state_force", "finite_T_thermal", "mixed",
    "spectral_model_truncation",
)
QHX_TERMS = ("q", "h", "x")
QHX_UNITS = "dimensionless"
CREUTZ_UNITS = "dimensionless"
OPEN_RUN_GATES = {
    "operator": "selected kernel descriptors lack independent production-link validation",
    "reference": "F-lane positive stout-force scale lower-bound receipt is missing",
    "numerical": "authorized C-loop means and independent F-lane spectrum/thermal receipts are missing",
    "statistical": "paired C/F interval results and full failed-draw denominators are missing",
    "comparison": "fresh confirmation IDs and data/resource owner authorization are missing",
}


def q_overlap_contract() -> dict[str, Any]:
    """Return the selected kernel-indexed q object and its operator mapping."""
    return {
        "definition_id": "bare_basis_squared_overlap_ratio_v1",
        "kernel_to_basis_operator": {
            "identity": "identity",
            "ape_alpha0.4_n5": "APE",
            "spatial_stout_rho0.10_n5": "stout",
        },
        "matrix_convention": (
            "C_ij(R,t)=sum_n Z_ni(R) * conjugate(Z_nj(R)) * exp(-E_n(R)*t)"
        ),
        "operator_overlap_convention": "Z_nk(R)=<n|O_k(R)|0> for the ordered bare smearing operator O_k",
        "ratio_upper_semantics": "one-sided upper bound on |Z_nk(R)/Z_0k(R)|^2",
        "q_definition": (
            "q_k(R,T)=sum_{n>0} |Z_nk(R)/Z_0k(R)|^2 * "
            "exp(-(E_n(R)-E_0(R))*T)"
        ),
        "fit_source": "joint finite-Nt spectral fit to the measured 3x3 C_ij(R,t) matrix",
        "gevp_role": (
            "GEVP may initialize, label, and stress-test joint spectral fits; its "
            "single composite projected operator is separate and must not be copied "
            "into identity, APE, or stout kernel-indexed q cells"
        ),
        "scope": (
            "positive forward-spectrum q term only; finite-Nt backward/thermal, "
            "mixed-sector, fit-truncation, and interval-coverage bounds remain separate"
        ),
        "finite_Nt_trace_decomposition": {
            "definition_id": "two_sector_transfer_trace_qhx_v1",
            "trace_formula": (
                "C_ij(R,T;N_t)=Z0(N_t)^-1 * sum_(m,n) "
                "exp[-eps_m*(N_t-T)] * exp[-V_n(R)*T] "
                "* A_i,mn(R) * conjugate(A_j,mn(R))"
            ),
            "partition_function": "Z0(N_t)=sum_m exp[-eps_m*N_t]",
            "component_sectors": {
                "q": "m=0,n>0: vacuum ground state with excited static-source states",
                "h": "m>0,n=0: excited vacuum-sector states with static-source ground state",
                "x": "m>0,n>0: excited states in both transfer-matrix sectors",
            },
            "diagonal_normalization": (
                "C_kk/C_kk^(m=0,n=0)=1+q_k+h_k+x_k; "
                "delta_log=log1p(q_k+h_k+x_k)"
            ),
            "endpoint_convention": (
                "proposed time-local source/sink convention with constant endpoint "
                "factors absorbed into A; production exporter time origin and any "
                "insertion offsets require independent review"
            ),
            "positivity_assumption": (
                "q/h/x are nonnegative only under a positive transfer-matrix "
                "spectral representation and the stated endpoint factorization"
            ),
            "status": "PROPOSED_TRACE_INTERPRETATION_REQUIRES_INDEPENDENT_REVIEW",
        },
        "status": "PROPOSED_METHOD_DEFAULT_REQUIRES_INDEPENDENT_REVIEW",
    }


def open_gate_run_state() -> dict[str, Any]:
    """Project the current two-lane physical blockers into the generic state model."""
    state = RunState()
    for stage, reason in OPEN_RUN_GATES.items():
        state.abstain(stage, reason)
    return state.summary()


def selected_protocol() -> dict[str, Any]:
    """Return selected C/F method defaults without inventing evidence receipts."""
    return {
        "protocol_id": "wilson-two-lane-selected-v3",
        "status": "METHOD_DEFAULTS_SELECTED_EVIDENCE_NOT_READY",
        "ensemble_design": {
            "geometry": [16, 16, 16, 16],
            "action": "Wilson gauge action, beta=6",
            "boundary_conditions": {"spatial": "periodic", "temporal": "periodic"},
            "temporal_extent_Nt": 16,
            "primary_kernel": "spatial_stout_rho0.10_n5",
            "descriptive_kernel": "ape_alpha0.4_n5",
            "kernel_pair_interpretation": "weak-field matched candidate pair only; no nonperturbative radius equivalence asserted",
        },
        "familywise_inference": {
            "alpha_total": 0.05,
            "co_primary_lanes": ["finite_T_Creutz", "ground_state_discrete_static_force"],
            "alpha_each_lane": 0.025,
            "block_bootstrap_draws": 10000,
            "block_bootstrap_seed_base": 20261125,
            "block_bootstrap_seed_rule": "seed = seed_base + block_length",
            "block_bootstrap_invalid_draw_fraction_max": 0.01,
            "primary_circular_block_length": 5,
            "sensitivity_block_lengths": [2, 10],
            "interval_rule": "hull of the required block-length intervals",
            "failed_draw_rule": "invalid required loop mean invalidates the paired draw; abstain if >1% invalid draws or precision gate fails",
        },
        "familywise_alpha_allocation": {
            "alpha_total": 0.05,
            "alpha_creutz_lane": 0.025,
            "alpha_force_lane": 0.025,
            "alpha_force_statistics": 0.0125,
            "alpha_force_systematics": 0.0125,
            "alpha_force_primary_interval": 0.00625,
            "alpha_positive_force_scale_lower_bound": 0.00625,
            "force_systematic_bound_cells": 28,
            "alpha_per_force_systematic_cell": 0.0125 / 28,
        },
        "finite_T_Creutz": {
            "estimand": "absolute finite-T chi_stout(R=2,T=5), not a ground-state force",
            "R": 2, "T": 5,
            "kernel_id": "spatial_stout_rho0.10_n5",
            "formula": "-[log Wbar(3,6)+log Wbar(2,5)-log Wbar(3,5)-log Wbar(2,6)]",
            "loop_mean_rule": "compute paired ensemble means first; take logs only for positive means",
            "precision_halfwidth_target": 0.02,
            "precision_units": "dimensionless",
            "bootstrap_seed_base": 20261125,
            "bootstrap_seed_rule": "seed = seed_base + block_length",
            "bootstrap_invalid_draw_fraction_max": 0.01,
            "data_receipt_sha256": None,
            "paired_split_manifest_sha256": None,
        },
        "ground_state_discrete_static_force": {
            "estimand": "absolute stout ground-state discrete static force at midpoint R=2.5",
            "stencil_R": 2,
            "midpoint_R": 2.5,
            "candidate_kernel_id": "spatial_stout_rho0.10_n5",
            "reference_kernel_id": "identity",
            "formula": "U0_stout(3)-U0_stout(2)",
            "reference_formula": "U0_identity(3)-U0_identity(2)",
            "reference_role": (
                "identity is an operator-overlap and shared-spectrum consistency check; "
                "the ground-state force difference is not the primary estimand"
            ),
            "units": LATTICE_FORCE_UNITS,
            "spectral_evidence_plan": {
                "basis": ["identity", "APE", "stout"],
                "q_overlap_contract": q_overlap_contract(),
                "gevp_dimension": 3,
                "t0": 1, "td": 2,
                "R_values": [2, 3],
                "joint_finite_Nt_state_count_checks": [2, 3],
                "fit_window": [3, 7],
                "tmin_checks": [3, 4, 5],
                "contamination_stencil": {
                    "primary_effective_energy_T": 5,
                    "loop_time_extents": [5, 6],
                    "rationale": (
                        "T=5 and T+1=6 lie inside every selected fit window "
                        "[tmin,7] for tmin=3,4,5"
                    ),
                    "q_envelope": (
                        "for each selected kernel and R, take the maximum over "
                        "all tmin=3,4,5 and 2/3-state variants of "
                        "sum_n overlap_ratio_upper[n]*exp(-gap_lower[n]*T); "
                        "overlap_ratio_upper bounds |Z_nk/Z_0k|^2 for that bare basis operator"
                    ),
                    "within_cell_interval_family": (
                        "Bonferroni over 24 endpoint slots per q cell "
                        "(6 fit variants x up to 2 excited states x 2 one-sided "
                        "endpoints); 2-state variants use one excited state, "
                        "unused slots are not reassigned, and "
                        "alpha_endpoint=systematic_cell_alpha/24"
                    ),
                    "status": "PROPOSED_METHOD_DEFAULT_REQUIRES_INDEPENDENT_REVIEW",
                },
                "matched_thermal_Nt": 32,
            },
            "relative_error_budget": {
                "reference": "positive simultaneous lower confidence bound for f_reference is required; otherwise abstain",
                "excited_state_q_max_fraction": 0.015,
                "thermal_h_max_fraction": 0.015,
                "mixed_x_max_fraction": 0.01,
                "spectral_model_truncation_max_fraction": 0.01,
                "systematic_total_max_fraction": 0.05,
                "statistical_halfwidth_max_fraction": 0.05,
                "linear_total_max_fraction": 0.10,
                "q_h_x_cells": 24,
                "model_truncation_cells": 4,
                "systematic_cell_alpha": 0.0125 / 28,
            },
            "error_propagation": "use log(1+q+h+x) at the four stout R=2,3 and T=5,6 corners, then apply Delta_R Delta_t before the force-unit gate; identity corners are comparison diagnostics only",
            "evidence_receipts": {
                "gevp_and_overlap": None, "excited_state_q": None,
                "thermal_h_Nt32": None, "mixed_x": None,
                "spectral_model_truncation": None,
                "independent_physics_review": None,
            },
        },
        "gates": {
            "production_data_authorized": False,
            "gpu_quda_authorized": False,
            "sealed_confirmation_open": False,
            "finite_T_Creutz_certified": False,
            "ground_state_force_certified": False,
            "physical_score": None,
        },
    }


def empty_contract() -> dict[str, Any]:
    """Return a selected C/F plan with all production evidence left null."""
    return {
        "schema": SCHEMA,
        "status": "DRAFT_NOT_READY",
        "physical_score": None,
        "certified": False,
        "selected_protocol": selected_protocol(),
        "common": {
            "metadata_status": "PROPOSED_METHOD_METADATA_NO_PRODUCTION_RECEIPT",
            "action_id": "wilson_gauge_beta6_proposed",
            "static_source_action_id": "fundamental_rectangular_wilson_loop_proposed",
            "geometry": {"spatial_extents": [16, 16, 16], "temporal_extent": 16},
            "boundary_conditions": {"spatial": "periodic", "temporal": "periodic"},
            "lattice_spacing": {
                "a_s": None, "a_t": None,
                "length_unit": None, "energy_unit": None,
                "scale_status": "not_required_for_dimensionless_lattice_estimands",
            },
            "kernel_ids": [
                "spatial_stout_rho0.10_n5", "ape_alpha0.4_n5", "identity",
            ],
            "split_manifest_sha256": None,
            "paired_block_plan_sha256": None,
            "independent_review_receipt": None,
        },
        "lanes": {
            "finite_T_Creutz": {
                "estimand": "finite_T_Creutz",
                "primary": {"R": 2, "T": 5,
                            "kernel_id": "spatial_stout_rho0.10_n5"},
                "stencil": [
                    {"R": 2, "T": 5, "coefficient": -1},
                    {"R": 3, "T": 5, "coefficient": 1},
                    {"R": 2, "T": 6, "coefficient": 1},
                    {"R": 3, "T": 6, "coefficient": -1},
                ],
                "log_rule": "log_positive_ensemble_mean",
                "invalid_point_rule": (
                    "invalidate a paired draw if any required loop mean is nonpositive or nonfinite; "
                    "abstain if invalid fraction exceeds 0.01 or the precision gate fails"
                ),
                "comparison_family": (
                    "co-primary family with the separate ground_state_discrete_static_force lane; "
                    "alpha allocated to this lane is 0.025"
                ),
                "inference": {
                    "interval_method": (
                        "paired circular block bootstrap: 10000 draws; primary block length 5; "
                        "report hull across lengths 2, 5, 10; coverage calibration remains open"
                    ),
                    "alpha_total": 0.025,
                    "alpha_allocations": {"finite_T_Creutz_two_sided_interval": 0.025},
                    "bootstrap_seed_base": 20261125,
                    "bootstrap_seed_rule": "seed = seed_base + block_length",
                    "bootstrap_invalid_draw_fraction_max": 0.01,
                    "precision_target": 0.02,
                    "precision_units": CREUTZ_UNITS,
                    "multiplicity_method": "Bonferroni allocation across the two co-primary lanes",
                },
            },
            "ground_state_discrete_static_force": {
                "estimand": "absolute_ground_state_discrete_static_force",
                "primary": {"R": 2, "midpoint_R": 2.5,
                            "candidate_kernel_id": "spatial_stout_rho0.10_n5",
                            "reference_kernel_id": "identity"},
                "force_stencil": [
                    {"R": 3, "kernel_id": "spatial_stout_rho0.10_n5", "coefficient": 1},
                    {"R": 2, "kernel_id": "spatial_stout_rho0.10_n5", "coefficient": -1},
                ],
                "reference_force_stencil": [
                    {"R": 3, "kernel_id": "identity", "coefficient": 1},
                    {"R": 2, "kernel_id": "identity", "coefficient": -1},
                ],
                "formula": "U0_stout(3)-U0_stout(2)",
                "reference_formula": "U0_identity(3)-U0_identity(2)",
                "reference_role": (
                    "identity is an operator-overlap and shared-spectrum consistency check; "
                    "the ground-state force difference is not the primary estimand"
                ),
                "units": LATTICE_FORCE_UNITS,
                "force_bound_propagation": {
                    "assumption": "q,h,x are nonnegative loop-level upper bounds and each correction is log(1+q+h+x)",
                    "corner_count": 4,
                    "corner_layout": [
                        "stout_R2_T5", "stout_R2_T6",
                        "stout_R3_T5", "stout_R3_T6",
                    ],
                    "component_force_bound": "sum over the four stout corners of log1p(component_bound)",
                    "uniform_raw_loop_cap": "expm1(force_fraction*positive_force_scale_lower/4)",
                    "identity_comparison_corners": 4,
                    "evidence_status": "FORMULA_ONLY_INPUT_RECEIPTS_MISSING",
                },
                "spectral_bounds": [
                    {"component": "q", "bound": None, "units": QHX_UNITS,
                     "proposed_relative_cap": 0.015, "evidence_sha256": None,
                     "scope": "proposed 1.5% post-propagation absolute stout-force component budget; derive raw corner cap from the positive force-scale lower bound",
                     "uniform_raw_loop_cap_formula": "expm1(0.015*positive_force_scale_lower/4)",
                     "independent_review_receipt": None},
                    {"component": "h", "bound": None, "units": QHX_UNITS,
                     "proposed_relative_cap": 0.015, "evidence_sha256": None,
                     "scope": "proposed 1.5% post-propagation absolute stout-force component budget; derive raw corner cap from the positive force-scale lower bound",
                     "uniform_raw_loop_cap_formula": "expm1(0.015*positive_force_scale_lower/4)",
                     "independent_review_receipt": None},
                    {"component": "x", "bound": None, "units": QHX_UNITS,
                     "proposed_relative_cap": 0.01, "evidence_sha256": None,
                     "scope": "proposed 1.0% post-propagation absolute stout-force component budget; derive raw corner cap from the positive force-scale lower bound",
                     "uniform_raw_loop_cap_formula": "expm1(0.01*positive_force_scale_lower/4)",
                     "independent_review_receipt": None},
                ],
                "systematic_bounds": [
                    {"component": "excited_state_force", "bound": None,
                     "units": LATTICE_FORCE_UNITS, "proposed_relative_cap": 0.015,
                     "evidence_sha256": None,
                     "scope": "force-unit component bound after propagation through the four stout loop corners",
                     "independent_review_receipt": None},
                    {"component": "finite_T_thermal", "bound": None,
                     "units": LATTICE_FORCE_UNITS, "proposed_relative_cap": 0.015,
                     "evidence_sha256": None,
                     "scope": "force-unit component bound after propagation through the four stout loop corners",
                     "independent_review_receipt": None},
                    {"component": "mixed", "bound": None,
                     "units": LATTICE_FORCE_UNITS, "proposed_relative_cap": 0.01,
                     "evidence_sha256": None,
                     "scope": "force-unit component bound after propagation through the four stout loop corners",
                     "independent_review_receipt": None},
                    {"component": "spectral_model_truncation", "bound": None,
                     "units": LATTICE_FORCE_UNITS, "proposed_relative_cap": 0.01,
                     "evidence_sha256": None,
                     "scope": "four stout-corner model-truncation bounds combined in force units",
                     "independent_review_receipt": None},
                ],
                "spectral_evidence_plan": {
                    "basis": ["identity", "APE", "stout"], "gevp_dimension": 3,
                    "q_overlap_contract": q_overlap_contract(),
                    "t0": 1, "td": 2, "R_values": [2, 3],
                    "joint_finite_Nt_state_count_checks": [2, 3],
                    "fit_window": [3, 7], "tmin_checks": [3, 4, 5],
                    "contamination_stencil": {
                        "primary_effective_energy_T": 5,
                        "loop_time_extents": [5, 6],
                        "rationale": (
                            "T=5 and T+1=6 lie inside every selected fit window "
                            "[tmin,7] for tmin=3,4,5"
                        ),
                        "q_envelope": (
                            "for each selected kernel and R, take the maximum over "
                            "all tmin=3,4,5 and 2/3-state variants of "
                            "sum_n overlap_ratio_upper[n]*exp(-gap_lower[n]*T); "
                            "overlap_ratio_upper bounds |Z_nk/Z_0k|^2 for that bare basis operator"
                        ),
                        "within_cell_interval_family": (
                            "Bonferroni over 24 endpoint slots per q cell "
                            "(6 fit variants x up to 2 excited states x 2 one-sided "
                            "endpoints); 2-state variants use one excited state, "
                            "unused slots are not reassigned, and "
                            "alpha_endpoint=systematic_cell_alpha/24"
                        ),
                        "status": "PROPOSED_METHOD_DEFAULT_REQUIRES_INDEPENDENT_REVIEW",
                    },
                    "matched_thermal_Nt": 32,
                },
                "inference": {
                    "interval_method": (
                        "paired circular block bootstrap: 10000 draws; primary block length 5; "
                        "report hull across lengths 2, 5, 10; coverage calibration remains open"
                    ),
                    "alpha_total": 0.025,
                    "alpha_allocations": {
                        "stout_force_two_sided_interval": 0.00625,
                        "positive_force_scale_one_sided_lower_bound": 0.00625,
                        "systematic_family_28_cells": 0.0125,
                    },
                    "statistical_precision_target": None,
                    "proposed_relative_statistical_halfwidth_cap": 0.05,
                    "precision_units": LATTICE_FORCE_UNITS,
                    "multiplicity_method": (
                        "Bonferroni: 24 q/h/x cells and 4 spectral-model cells; "
                        "alpha=0.0125/28 per systematic cell"
                    ),
                },
                "total_tolerance": None,
                "proposed_relative_total_tolerance": 0.10,
                "scope": (
                    "fixed lattice spacing, volume, action, and static-source convention; "
                    "continuum and infinite-volume extrapolations are outside this estimand"
                ),
            },
        },
    }


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def _mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _get_path(root: Any, path: str) -> Any:
    current = root
    for token in path.split("."):
        if not isinstance(current, dict) or token not in current:
            return None
        current = current[token]
    return current


def _check_common(common: Any, *, dimensionless_lattice_force: bool = False) -> list[str]:
    issues: list[str] = []
    if not _mapping(common):
        return ["common: expected an object"]
    for field in ("action_id", "static_source_action_id",
                  "independent_review_receipt"):
        if not _nonempty(common.get(field)):
            issues.append(f"common.{field}: missing nonempty receipt/identifier")
    geometry = common.get("geometry")
    if not _mapping(geometry):
        issues.append("common.geometry: expected an object")
    else:
        extents = geometry.get("spatial_extents")
        if (not isinstance(extents, list) or len(extents) != 3
                or any(not _positive_int(v) for v in extents)):
            issues.append("common.geometry.spatial_extents: expected three positive integers")
        if not _positive_int(geometry.get("temporal_extent")):
            issues.append("common.geometry.temporal_extent: expected a positive integer")
    boundaries = common.get("boundary_conditions")
    if not _mapping(boundaries):
        issues.append("common.boundary_conditions: expected an object")
    else:
        for field in ("spatial", "temporal"):
            if not _nonempty(boundaries.get(field)):
                issues.append(f"common.boundary_conditions.{field}: missing")
    spacing = common.get("lattice_spacing")
    if not _mapping(spacing):
        issues.append("common.lattice_spacing: expected an object")
    else:
        if (dimensionless_lattice_force
                and spacing.get("scale_status")
                == "not_required_for_dimensionless_lattice_estimands"
                and spacing.get("a_s") is None and spacing.get("a_t") is None):
            pass
        else:
            for field in ("a_s", "a_t"):
                if not _is_number(spacing.get(field)) or spacing[field] <= 0:
                    issues.append(f"common.lattice_spacing.{field}: expected a positive finite value")
            for field in ("length_unit", "energy_unit"):
                if not _nonempty(spacing.get(field)):
                    issues.append(f"common.lattice_spacing.{field}: missing unit")
    kernels = common.get("kernel_ids")
    if (not isinstance(kernels, list) or not kernels
            or any(not _nonempty(v) for v in kernels)
            or len(set(kernels)) != len(kernels)):
        issues.append("common.kernel_ids: expected unique nonempty identifiers")
    for field in ("split_manifest_sha256", "paired_block_plan_sha256"):
        if not _sha256(common.get(field)):
            issues.append(f"common.{field}: expected a SHA-256 receipt")
    return issues


def _check_stencil(rows: Any, *, expected: set[tuple[int, int]],
                   coefficient_pattern: dict[tuple[int, int], int],
                   label: str) -> list[str]:
    if not isinstance(rows, list) or len(rows) != 4:
        return [f"{label}: expected four explicit corners"]
    issues: list[str] = []
    observed: dict[tuple[int, int], Any] = {}
    for index, row in enumerate(rows):
        if not _mapping(row):
            issues.append(f"{label}[{index}]: expected an object")
            continue
        r, t, coefficient = row.get("R"), row.get("T"), row.get("coefficient")
        if not _positive_int(r) or not _positive_int(t):
            issues.append(f"{label}[{index}]: R and T must be positive integers")
            continue
        if (r, t) in observed:
            issues.append(f"{label}: duplicate corner {(r, t)}")
        observed[(r, t)] = coefficient
    if set(observed) != expected:
        issues.append(f"{label}: corners do not match the declared primary R/T stencil")
    elif all(_is_number(v) for v in observed.values()):
        actual = {key: int(value) if float(value).is_integer() else value
                  for key, value in observed.items()}
        if actual != coefficient_pattern:
            issues.append(f"{label}: coefficients violate the prescribed signed four-point convention")
    else:
        issues.append(f"{label}: coefficients must be finite numbers")
    return issues


def _check_interval(inference: Any, *, precision_field: str,
                    expected_precision_units: str | None = None,
                    label: str) -> list[str]:
    if not _mapping(inference):
        return [f"{label}: expected an object"]
    issues: list[str] = []
    for field in ("interval_method", "multiplicity_method"):
        if not _nonempty(inference.get(field)):
            issues.append(f"{label}.{field}: missing")
    alpha = inference.get("alpha_total")
    if not _is_number(alpha) or not 0 < alpha < 1:
        issues.append(f"{label}.alpha_total: expected a value in (0,1)")
    allocations = inference.get("alpha_allocations")
    if not isinstance(allocations, dict) or not allocations:
        issues.append(f"{label}.alpha_allocations: list the prospective familywise allocations")
    else:
        values = list(allocations.values())
        if any(not _is_number(value) or value <= 0 for value in values):
            issues.append(f"{label}.alpha_allocations: values must be positive finite numbers")
        elif _is_number(alpha) and sum(values) > alpha + 1e-12:
            issues.append(f"{label}.alpha_allocations: sum exceeds alpha_total")
    precision = inference.get(precision_field)
    if not _is_number(precision) or precision <= 0:
        issues.append(f"{label}.{precision_field}: expected a positive finite value")
    precision_units = inference.get("precision_units")
    if not _nonempty(precision_units):
        issues.append(f"{label}.precision_units: missing")
    elif (expected_precision_units is not None
          and precision_units != expected_precision_units):
        issues.append(
            f"{label}.precision_units: expected {expected_precision_units!r}"
        )
    return issues


def _check_creutz(lane: Any, common: Any) -> list[str]:
    label = "finite_T_Creutz"
    if not _mapping(lane):
        return [f"{label}: expected an object"]
    issues: list[str] = []
    if lane.get("estimand") != "finite_T_Creutz":
        issues.append(f"{label}.estimand: mismatch")
    primary = lane.get("primary")
    if not _mapping(primary):
        issues.append(f"{label}.primary: expected an object")
        primary = {}
    r, t, kernel = primary.get("R"), primary.get("T"), primary.get("kernel_id")
    if not _positive_int(r):
        issues.append(f"{label}.primary.R: expected a positive integer")
    if not _positive_int(t):
        issues.append(f"{label}.primary.T: expected a positive integer")
    kernels_value = common.get("kernel_ids") if _mapping(common) else None
    kernels = kernels_value if isinstance(kernels_value, list) else []
    if not _nonempty(kernel) or kernel not in kernels:
        issues.append(f"{label}.primary.kernel_id: not declared in common.kernel_ids")
    geometry = common.get("geometry", {}) if _mapping(common) else {}
    if _positive_int(r) and _mapping(geometry):
        extents = geometry.get("spatial_extents")
        if isinstance(extents, list) and len(extents) == 3 and all(_positive_int(v) for v in extents):
            if r + 1 > min(extents):
                issues.append(f"{label}.primary.R: R+1 exceeds the declared spatial extent")
    if _positive_int(t) and _mapping(geometry):
        nt = geometry.get("temporal_extent")
        if _positive_int(nt) and t + 1 > nt:
            issues.append(f"{label}.primary.T: T+1 exceeds the declared temporal extent")
    if _positive_int(r) and _positive_int(t):
        pattern = {(r, t): -1, (r + 1, t): 1,
                   (r, t + 1): 1, (r + 1, t + 1): -1}
        issues.extend(_check_stencil(
            lane.get("stencil"), expected=set(pattern),
            coefficient_pattern=pattern, label=f"{label}.stencil",
        ))
    if lane.get("log_rule") != "log_positive_ensemble_mean":
        issues.append(f"{label}.log_rule: must explicitly be log of positive ensemble means")
    if not _nonempty(lane.get("invalid_point_rule")):
        issues.append(f"{label}.invalid_point_rule: missing failure rule")
    if not _nonempty(lane.get("comparison_family")):
        issues.append(f"{label}.comparison_family: missing")
    issues.extend(_check_interval(lane.get("inference"),
                                  precision_field="precision_target",
                                  expected_precision_units=CREUTZ_UNITS,
                                  label=f"{label}.inference"))
    return issues


def _check_evidence_rows(rows: Any, expected_names: tuple[str, ...],
                         *, label: str, units: str | None = None) -> list[str]:
    if not isinstance(rows, list):
        return [f"{label}: expected evidence rows"]
    issues: list[str] = []
    by_name: dict[str, dict] = {}
    for index, row in enumerate(rows):
        if not _mapping(row):
            issues.append(f"{label}[{index}]: expected an object")
            continue
        name = row.get("component", row.get("term"))
        if not _nonempty(name) or name in by_name:
            issues.append(f"{label}[{index}]: missing or duplicate component/term")
            continue
        by_name[name] = row
        bound = row.get("bound")
        if not _is_number(bound) or bound < 0:
            issues.append(f"{label}.{name}.bound: expected finite nonnegative value")
        row_units = row.get("units")
        if not _nonempty(row_units) or (units is not None and row_units != units):
            issues.append(f"{label}.{name}.units: missing or inconsistent")
        if not _sha256(row.get("evidence_sha256")):
            issues.append(f"{label}.{name}.evidence_sha256: missing evidence hash")
        if not _nonempty(row.get("scope")):
            issues.append(f"{label}.{name}.scope: missing")
        if not _nonempty(row.get("independent_review_receipt")):
            issues.append(f"{label}.{name}.independent_review_receipt: missing")
    missing = sorted(set(expected_names) - set(by_name))
    for name in missing:
        issues.append(f"{label}.{name}: missing required evidence")
    return issues


def _check_force(lane: Any, common: Any) -> list[str]:
    label = "ground_state_discrete_static_force"
    if not _mapping(lane):
        return [f"{label}: expected an object"]
    issues: list[str] = []
    if lane.get("estimand") != "absolute_ground_state_discrete_static_force":
        issues.append(f"{label}.estimand: mismatch")
    primary = lane.get("primary")
    if not _mapping(primary):
        issues.append(f"{label}.primary: expected an object")
        primary = {}
    r = primary.get("R")
    candidate, reference = primary.get("candidate_kernel_id"), primary.get("reference_kernel_id")
    kernels_value = common.get("kernel_ids") if _mapping(common) else None
    kernels = kernels_value if isinstance(kernels_value, list) else []
    if not _positive_int(r):
        issues.append(f"{label}.primary.R: expected a positive integer")
    for name, value in (("candidate_kernel_id", candidate),
                        ("reference_kernel_id", reference)):
        if not _nonempty(value) or value not in kernels:
            issues.append(f"{label}.primary.{name}: not declared in common.kernel_ids")
    if candidate == reference:
        issues.append(f"{label}.primary: candidate and reference kernels must differ")
    geometry = common.get("geometry", {}) if _mapping(common) else {}
    if _positive_int(r) and _mapping(geometry):
        extents = geometry.get("spatial_extents")
        if (isinstance(extents, list) and len(extents) == 3
                and all(_positive_int(v) for v in extents)
                and r + 1 > min(extents)):
            issues.append(f"{label}.primary.R: force stencil R+1 exceeds the declared spatial extent")
    if not _nonempty(lane.get("units")):
        issues.append(f"{label}.units: missing")
    def check_two_term_stencil(field: str, kernel: str) -> None:
        rows = lane.get(field)
        if not isinstance(rows, list) or len(rows) != 2:
            issues.append(f"{label}.{field}: expected two explicit discrete-force terms")
            return
        observed: dict[tuple[int, str], Any] = {}
        for index, row in enumerate(rows):
            if not _mapping(row):
                issues.append(f"{label}.{field}[{index}]: expected an object")
                continue
            rr, kk, coeff = row.get("R"), row.get("kernel_id"), row.get("coefficient")
            if not _positive_int(rr) or not _nonempty(kk):
                issues.append(f"{label}.{field}[{index}]: invalid R or kernel_id")
                continue
            key = (rr, kk)
            if key in observed:
                issues.append(f"{label}.{field}: duplicate term {key}")
            observed[key] = coeff
        expected = {(r, kernel), (r + 1, kernel)}
        pattern = {(r + 1, kernel): 1, (r, kernel): -1}
        if set(observed) != expected:
            issues.append(f"{label}.{field}: terms do not match its declared kernel and R stencil")
        elif all(_is_number(value) for value in observed.values()):
            actual = {key: int(value) if float(value).is_integer() else value
                      for key, value in observed.items()}
            if actual != pattern:
                issues.append(f"{label}.{field}: coefficients must encode R+1 minus R")
        else:
            issues.append(f"{label}.{field}: coefficients must be finite numbers")

    if _positive_int(r) and _nonempty(candidate) and _nonempty(reference):
        check_two_term_stencil("force_stencil", candidate)
        check_two_term_stencil("reference_force_stencil", reference)
    expected_protocol = selected_protocol()["ground_state_discrete_static_force"]
    is_synthetic = (
        _mapping(common)
        and isinstance(common.get("action_id"), str)
        and common["action_id"].startswith("SYNTHETIC_ONLY")
    )
    if not is_synthetic:
        if candidate != expected_protocol["candidate_kernel_id"]:
            issues.append(f"{label}.primary.candidate_kernel_id: must match the selected stout kernel")
        if reference != expected_protocol["reference_kernel_id"]:
            issues.append(f"{label}.primary.reference_kernel_id: must match the identity comparison kernel")
    if lane.get("formula") != expected_protocol["formula"]:
        issues.append(f"{label}.formula: must encode the absolute stout ground-state force")
    if lane.get("reference_formula") != expected_protocol["reference_formula"]:
        issues.append(f"{label}.reference_formula: must encode the identity comparison force")
    if lane.get("reference_role") != expected_protocol["reference_role"]:
        issues.append(f"{label}.reference_role: identity must remain a comparison diagnostic")
    propagation = lane.get("force_bound_propagation")
    if not _mapping(propagation) or propagation.get("corner_count") != 4:
        issues.append(f"{label}.force_bound_propagation: absolute force requires four stout corners")
    elif propagation.get("uniform_raw_loop_cap") != (
        "expm1(force_fraction*positive_force_scale_lower/4)"
    ):
        issues.append(f"{label}.force_bound_propagation: raw loop cap must use four stout corners")

    qhx = lane.get("spectral_bounds")
    issues.extend(_check_evidence_rows(qhx, QHX_TERMS,
                                       label=f"{label}.spectral_bounds",
                                       units=QHX_UNITS))
    systematic = lane.get("systematic_bounds")
    issues.extend(_check_evidence_rows(systematic, F_BOUND_COMPONENTS,
                                       label=f"{label}.systematic_bounds",
                                       units=lane.get("units") if _nonempty(lane.get("units")) else None))
    if isinstance(systematic, list) and _is_number(lane.get("total_tolerance")):
        systematic_sum = 0.0
        for row in systematic:
            if _mapping(row) and _is_number(row.get("bound")) and row["bound"] >= 0:
                systematic_sum += float(row["bound"])
        inference = lane.get("inference")
        precision = (_get_path(inference, "statistical_precision_target")
                     if _mapping(inference) else None)
        precision_units = (_get_path(inference, "precision_units")
                           if _mapping(inference) else None)
        precision_in_force_units = (
            _is_number(precision) and precision >= 0
            and _nonempty(lane.get("units"))
            and precision_units == lane.get("units")
        )
        total_budget = systematic_sum + (
            float(precision) if precision_in_force_units else 0.0
        )
        if total_budget > lane["total_tolerance"]:
            issues.append(f"{label}.systematic_bounds: force-level systematic plus statistical budgets exceed total_tolerance")
    tolerance = lane.get("total_tolerance")
    if not _is_number(tolerance) or tolerance <= 0:
        issues.append(f"{label}.total_tolerance: expected a positive finite value")
    issues.extend(_check_interval(lane.get("inference"),
                                  precision_field="statistical_precision_target",
                                  expected_precision_units=(
                                      lane.get("units") if _nonempty(lane.get("units"))
                                      else None
                                  ),
                                  label=f"{label}.inference"))
    return issues


def inspect_contract(contract: Any) -> dict[str, Any]:
    """Validate structure and blockers without granting physics readiness."""
    if not _mapping(contract):
        return {
            "schema_valid": False, "status": "DRAFT_NOT_READY",
            "physical_score": None, "certified": False,
            "common_issues": ["contract: expected an object"],
            "lanes": {},
        }
    common = contract.get("common")
    raw_lanes = contract.get("lanes")
    force_lane = (raw_lanes.get("ground_state_discrete_static_force")
                  if _mapping(raw_lanes) else None)
    dimensionless_lattice_force = (
        _mapping(force_lane) and force_lane.get("units") == LATTICE_FORCE_UNITS
    )
    common_issues = _check_common(
        common, dimensionless_lattice_force=dimensionless_lattice_force,
    )
    protocol = contract.get("selected_protocol")
    if not _mapping(protocol):
        common_issues.append("selected_protocol: missing selected C/F method defaults")
    elif protocol != selected_protocol():
        common_issues.append(
            "selected_protocol: differs from frozen C/F defaults or evidence-open fields"
        )
    if contract.get("schema") != SCHEMA:
        common_issues.append(f"contract.schema: expected {SCHEMA}")
    if not _mapping(raw_lanes):
        raw_lanes = {}
    c_issues = common_issues + _check_creutz(raw_lanes.get("finite_T_Creutz"), common)
    f_issues = common_issues + _check_force(
        raw_lanes.get("ground_state_discrete_static_force"), common,
    )
    lanes = {}
    for name, issues in (("finite_T_Creutz", c_issues),
                         ("ground_state_discrete_static_force", f_issues)):
        lanes[name] = {
            "status": ("CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW"
                       if not issues else "NOT_READY"),
            "structurally_complete": not issues,
            "certified": False,
            "physical_score": None,
            "issues": issues,
        }
    schema_valid = contract.get("schema") == SCHEMA
    return {
        "schema_valid": schema_valid,
        "status": "DRAFT_NOT_READY",
        "physical_score": None,
        "certified": False,
        "common_issues": common_issues,
        "lanes": lanes,
    }


def _synthetic_contract(*, complete_creutz: bool,
                        complete_force: bool) -> dict[str, Any]:
    """Return explicitly synthetic shape fixtures; never evidence receipts."""
    contract = empty_contract()
    contract["common"] = {
        "action_id": "SYNTHETIC_ONLY_action_example",
        "static_source_action_id": "SYNTHETIC_ONLY_source_example",
        "geometry": {"spatial_extents": [12, 12, 12], "temporal_extent": 16},
        "boundary_conditions": {"spatial": "SYNTHETIC_ONLY_periodic",
                                 "temporal": "SYNTHETIC_ONLY_periodic"},
        "lattice_spacing": {"a_s": 0.1, "a_t": 0.1,
                            "length_unit": "SYNTHETIC_ONLY_fm",
                            "energy_unit": "SYNTHETIC_ONLY_GeV"},
        "kernel_ids": ["SYNTHETIC_ONLY_candidate", "SYNTHETIC_ONLY_reference"],
        "split_manifest_sha256": "a" * 64,
        "paired_block_plan_sha256": "b" * 64,
        "independent_review_receipt": "SYNTHETIC_ONLY_review_receipt",
    }
    if complete_creutz:
        contract["lanes"]["finite_T_Creutz"] = {
            "estimand": "finite_T_Creutz",
            "primary": {"R": 2, "T": 3,
                        "kernel_id": "SYNTHETIC_ONLY_candidate"},
            "stencil": [
                {"R": 2, "T": 3, "coefficient": -1},
                {"R": 3, "T": 3, "coefficient": 1},
                {"R": 2, "T": 4, "coefficient": 1},
                {"R": 3, "T": 4, "coefficient": -1},
            ],
            "log_rule": "log_positive_ensemble_mean",
            "invalid_point_rule": "SYNTHETIC_ONLY_fail closed if any mean is nonpositive",
            "comparison_family": "SYNTHETIC_ONLY_candidate versus reference",
            "inference": {
                "interval_method": "SYNTHETIC_ONLY_paired_block_interval",
                "alpha_total": 0.05, "alpha_allocations": {"primary": 0.05},
                "precision_target": 0.01,
                "precision_units": CREUTZ_UNITS,
                "multiplicity_method": "SYNTHETIC_ONLY_maxT",
            },
        }
    if complete_force:
        force_units = "SYNTHETIC_ONLY_GeV"
        review = "SYNTHETIC_ONLY_not_an_independent_physics_review"

        def evidence_rows(names: tuple[str, ...], units: str) -> list[dict[str, Any]]:
            return [{
                "component": name, "bound": 0.001,
                "units": units,
                "evidence_sha256": hashlib.sha256(
                    f"SYNTHETIC_ONLY:{name}".encode()).hexdigest(),
                "scope": "SYNTHETIC_ONLY one candidate contrast and stencil",
                "independent_review_receipt": review,
            } for name in names]

        contract["lanes"]["ground_state_discrete_static_force"] = {
            "estimand": "absolute_ground_state_discrete_static_force",
            "primary": {"R": 2,
                        "candidate_kernel_id": "SYNTHETIC_ONLY_candidate",
                        "reference_kernel_id": "SYNTHETIC_ONLY_reference"},
            "force_stencil": [
                {"R": 3, "kernel_id": "SYNTHETIC_ONLY_candidate", "coefficient": 1},
                {"R": 2, "kernel_id": "SYNTHETIC_ONLY_candidate", "coefficient": -1},
            ],
            "reference_force_stencil": [
                {"R": 3, "kernel_id": "SYNTHETIC_ONLY_reference", "coefficient": 1},
                {"R": 2, "kernel_id": "SYNTHETIC_ONLY_reference", "coefficient": -1},
            ],
            "formula": "U0_stout(3)-U0_stout(2)",
            "reference_formula": "U0_identity(3)-U0_identity(2)",
            "reference_role": (
                "identity is an operator-overlap and shared-spectrum consistency check; "
                "the ground-state force difference is not the primary estimand"
            ),
            "force_bound_propagation": {
                "corner_count": 4,
                "uniform_raw_loop_cap": "expm1(force_fraction*positive_force_scale_lower/4)",
            },
            "units": force_units,
            "spectral_bounds": evidence_rows(QHX_TERMS, QHX_UNITS),
            "systematic_bounds": evidence_rows(F_BOUND_COMPONENTS, force_units),
            "inference": {
                "interval_method": "SYNTHETIC_ONLY_paired_block_interval",
                "alpha_total": 0.05, "alpha_allocations": {"primary": 0.05},
                "statistical_precision_target": 0.001,
                "precision_units": force_units,
                "multiplicity_method": "SYNTHETIC_ONLY_maxT",
            },
            "total_tolerance": 0.01,
        }
    return contract


def build(output: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    """Write a source-closed report for the current draft and synthetic fixtures."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    c_reversed = _synthetic_contract(complete_creutz=True, complete_force=True)
    for row in c_reversed["lanes"]["finite_T_Creutz"]["stencil"]:
        row["coefficient"] *= -1
    f_reversed = _synthetic_contract(complete_creutz=True, complete_force=True)
    for row in f_reversed["lanes"]["ground_state_discrete_static_force"]["force_stencil"]:
        row["coefficient"] *= -1
    f_wrong_precision_units = _synthetic_contract(
        complete_creutz=True, complete_force=True,
    )
    f_inference = f_wrong_precision_units["lanes"][
        "ground_state_discrete_static_force"]["inference"]
    f_inference["statistical_precision_target"] = 100.0
    f_inference["precision_units"] = QHX_UNITS
    c_wrong_precision_units = _synthetic_contract(
        complete_creutz=True, complete_force=True,
    )
    c_inference = c_wrong_precision_units["lanes"]["finite_T_Creutz"]["inference"]
    c_inference["precision_units"] = "SYNTHETIC_ONLY_GeV"
    null_kernels = _synthetic_contract(complete_creutz=True, complete_force=True)
    null_kernels["common"]["kernel_ids"] = None
    current_contract = json.loads(CONTRACT_PATH.read_text())
    fixtures = {
        "current_selected_method_draft": current_contract,
        "synthetic_C_only_complete": _synthetic_contract(
            complete_creutz=True, complete_force=False,
        ),
        "synthetic_C_and_F_schema_complete": _synthetic_contract(
            complete_creutz=True, complete_force=True,
        ),
        "synthetic_C_reversed_sign": c_reversed,
        "synthetic_F_reversed_sign": f_reversed,
        "synthetic_F_wrong_precision_units": f_wrong_precision_units,
        "synthetic_C_wrong_precision_units": c_wrong_precision_units,
        "synthetic_null_common_kernel_ids": null_kernels,
    }
    results = {name: inspect_contract(contract)
               for name, contract in fixtures.items()}
    summary_rows = []
    for fixture, result in results.items():
        for lane, row in result["lanes"].items():
            summary_rows.append({
                "fixture": fixture,
                "lane": lane,
                "planned": 1,
                "valid": int(row["structurally_complete"]),
                "failed": int(not row["structurally_complete"]),
                "abstained": 0,
                "status": row["status"],
                "certified": row["certified"],
                "physical_score": row["physical_score"],
                "issue_count": len(row["issues"]),
            })
    run_state = open_gate_run_state()
    (out / "contract_diagnostics.json").write_text(
        json.dumps({"schema": SCHEMA, "data_access": "none",
                    "physical_score": None, "certified": False,
                    "run_state": run_state,
                    "selected_protocol": selected_protocol(),
                    "fixtures": results}, indent=2, allow_nan=False) + "\n"
    )
    with (out / "lane_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    lines = [
        "# Wilson two-lane contract diagnostic",
        "",
        "**STRUCTURE CHECK ONLY. `physical_score=null`; no Wilson measurements, sealed data, GPU, or QUDA inputs were accessed.**",
        "",
        "The selected protocol is recorded in `selected_protocol()`: Lane C is absolute stout `chi(2,5)` on periodic 16^3x16 Wilson beta=6 data, with dimensionless halfwidth 0.02; Lane F is the absolute stout ground-state discrete force `U0(3)-U0(2)` at midpoint R=2.5. Identity is retained as a shared-spectrum/operator-overlap comparison, not subtracted from the primary force. Each lane receives alpha=0.025 from the total 0.05 family budget. Both remain distinct estimands. These are design choices; measurement, spectrum, thermal, split, paired-block and review receipts remain absent.",
        "",
        "## Selected C/F decision flow",
        "",
        "```text",
        "paired loop IDs and 10,000 circular block-bootstrap draws (length 5; lengths 2/10 sensitivity)",
        "  -> C: positive ensemble means -> logs -> finite-T stout chi(2,5) -> dimensionless halfwidth <=0.02",
        "  -> F: independent 3x3 GEVP + Nt16/Nt32 joint spectrum -> q/h/x and model bounds",
        "  -> propagate log(1+q+h+x) through the four stout Delta_R Delta_t corners -> compare force-unit systematic/statistical budget",
        "  -> missing receipt, nonpositive stout force-scale lower bound, >1% invalid draws, or failed precision => lane-specific ABSTAIN",
        "  -> independent review and data/GPU authorization remain separate; physical_score stays null",
        "```",
        "",
        "For Lane F, q/h/x bounds propagate through the four stout corners only. The identity four-corner propagation is retained as a comparison diagnostic and does not double the primary budget. The q/h/x limits are relative to a positive simultaneous lower confidence bound for the stout-force scale; they are not themselves force units. The 28 simultaneous systematic/model cells each receive alpha=0.0125/28. Statistical alpha=0.0125 is split equally between the stout-force interval and its positive scale lower-bound gate.",
        "",
        "Structural completeness is not physics validation. Even a fully populated synthetic fixture can only reach `CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW`; the tool never sets a certified or ready state.",
        "",
        "## Current no-data RunState projection",
        "",
        "The separate generic `RunState` projection records all five overall physical-certification stages as `abstained`; each stage has an explicit reason in `contract_diagnostics.json`. It reports `physical_ready=false`, `score=null`, and planned/valid/failed/abstained=`5/0/0/5`. This is a snapshot of missing receipts/authorization, not five failed Wilson measurements. A future authorized calibration or confirmation must start a fresh run state.",
        "",
        "|stage|state|current blocker|",
        "|---|---|---|",
    ]
    for stage in ("operator", "reference", "numerical", "statistical", "comparison"):
        reason = run_state["evidence"][stage][0]
        lines.append(f"|{stage}|{run_state['status'][stage]}|{reason}|")
    lines.extend([
        "",
        "## Text diagnostic and decision flow",
        "",
        "```text",
        "selected protocol defaults -> acquire shared action, geometry, boundary, split and paired-block receipts",
        "  -> Lane C: stout chi(2,5), positive-mean logs, 0.02 dimensionless precision, 97.5% interval",
        "  -> Lane F: absolute stout ground-state force at R=2.5, with identity shared-spectrum comparison, 3x3 GEVP and matched Nt=32 thermal check",
        "  -> populate independently reviewed q/h/x/model bounds and their force-unit propagation",
        "  -> schema-complete? no: remain NOT_READY and list blockers",
        "  -> schema-complete? yes: CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW only",
        "  -> independent physics review + authorized measurements + sealed confirmation are separate later gates",
        "  -> physical_score remains null until those gates are independently satisfied",
        "```",
        "",
        f"`lane_summary.csv` reports planned/valid/failed/abstained counts for each synthetic fixture and lane. `run_receipt.json` records those {len(summary_rows)} structural checks plus a separate five-stage physical `RunState` denominator of `5/0/0/5` (planned/valid/failed/abstained), source and artifact hashes, and exact blocker reasons. `complete` means all planned units are accounted for, not that a lane or method succeeded. `artifact_hashes.json` also covers the receipt. The C interval precision target must be labeled `{CREUTZ_UNITS}` because its signed log-Creutz observable is dimensionless. A missing bound is never treated as zero. Bound rows require a scope, hash-shaped evidence receipt and review receipt; those strings are checked only for shape and are not opened or authenticated here. q/h/x receipts are dimensionless spectral inputs; the force-budget check requires their separately propagated excited-state, thermal, mixed, and spectral-model-truncation bounds in `{LATTICE_FORCE_UNITS}` and the statistical precision target. Since the selected force is itself a fixed-lattice/finite-volume estimand, finite-a and finite-volume extrapolation corrections are outside its scope; any continuum claim needs a separately versioned estimand. Familywise alpha allocations must sum to no more than `alpha_total`.",
        "",
        "## Fixture results",
        "",
        "|fixture|lane|planned|valid|failed|abstained|status|certified|score|blocker count|",
        "|---|---|---:|---:|---:|---:|---|---|---|---:|",
    ])
    for row in summary_rows:
        lines.append(
            f"|{row['fixture']}|{row['lane']}|{row['planned']}|{row['valid']}|{row['failed']}|{row['abstained']}|{row['status']}|{row['certified']}|{row['physical_score']}|{row['issue_count']}|"
        )
    lines.extend([
        "",
        "### What the fixtures establish",
        "",
        "- The current draft stays `NOT_READY` and exposes unset values as blockers.",
        "- A schema-complete synthetic C lane does not make Lane F ready.",
        "- Synthetic C+F completeness validates stencil and budget bookkeeping only; it does not provide real Q/H/X spectra, independent bounds, action receipts, uncertainty calibration, or physics certification.",
        "- Selected R/T/kernel, alpha, precision and contamination budgets are method defaults; their data, spectral and independent-review receipts remain missing, so both production lanes remain `NOT_READY/UNCERTIFIED` and the ground-state force stays uncertified.",
        "",
        "### Fail-closed regression diagnostics",
        "",
        "- Reversing all four C coefficients is rejected because the declared convention is `chi = -[log W(R+1,T+1)+log W(R,T)-log W(R+1,T)-log W(R,T+1)]`, matching `finite_nt_toy.py`.",
        "- Reversing the two stout F coefficients is rejected because the primary force is U0_stout(R+1)-U0_stout(R); identity has its own two-term comparison stencil.",
        "- A C statistical precision target labeled with energy/force units is rejected because the four-point log-Creutz estimator is dimensionless.",
        "- An F statistical precision target labeled dimensionless is rejected against the force-unit estimand and is excluded from the force-unit budget sum until units match.",
        "- A null shared `kernel_ids` field produces explicit common blockers; the inspector does not crash or mark either lane complete.",
        "- These are synthetic schema mutations. They diagnose contract handling only and do not read Wilson data or estimate a physical uncertainty.",
        "",
        "## Next work after the selected protocol defaults",
        "",
        "1. Keep the selected C estimand fixed: obtain an independently authorized calibration manifest for stout `chi(2,5)` and preserve four paired loop means, positive-mean/log checks, failed-draw reasons, the 10,000-draw length-5 bootstrap, and the length-2/10 interval hull. Do not reuse observed D0–99 as confirmation.",
        "2. Keep the absolute stout F estimand fixed: obtain separate calibration receipts for the R=2,3 three-operator GEVP, two/three-state and time-window diagnostics, matched Nt=32 thermal comparison, mixed-sector bound, and all force-unit propagated error cells. Propagate q/h/x and model truncation through the four stout corners; save identity-kernel results as a shared-spectrum consistency diagnostic with planned/valid/failed/abstained counts.",
        "3. Have an independent reviewer check coverage, force-unit propagation, correlated terms and any double counting; only after calibration passes request separate authorization for fresh, unobserved confirmation IDs and GPU/QUDA resources.",
        "4. Until these receipts and authorizations exist, retain both lane statuses as `NOT_READY/UNCERTIFIED`, keep the force lane `GROUND_STATE_FORCE_UNCERTIFIED`, do not open sealed data, and leave `physical_score=null`.",
        "",
    ])
    (out / "README.md").write_text("\n".join(lines))
    protocol = {
        "schema": SCHEMA,
        "status": "STRUCTURE_DIAGNOSTIC_ONLY",
        "data_access": "none",
        "physical_score": None,
        "certified": False,
        "run_state": run_state,
        "selected_protocol": selected_protocol(),
        "lane_estimands": ["finite_T_Creutz",
                           "absolute_ground_state_discrete_static_force"],
        "fixed_sign_conventions": {
            "finite_T_Creutz": "minus the R-T rectangular log-W difference, matching finite_nt_toy.py",
            "absolute_ground_state_discrete_static_force": "U0_stout(R+1)-U0_stout(R); identity is a comparison diagnostic",
        },
        "fixtures": list(fixtures),
        "meaning_of_valid": "schema complete only; requires human physics review and does not certify",
        "receipt": {
            "schema": "latticearena-run-receipt-v1",
            "unit": "one fixture-lane structural check",
            "complete_means": "all planned schema checks are accounted for; not scientific success",
        },
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (out / "lane_contract_v2.json").write_bytes(CONTRACT_PATH.read_bytes())
    source_snapshot = out / "snapshot_lane_contract.py"
    source_snapshot.write_bytes(Path(__file__).read_bytes())
    receipt_source = Path(__file__).resolve().parents[2] / "core/receipts.py"
    (out / "snapshot_core_receipts.py").write_bytes(receipt_source.read_bytes())
    state_source = Path(__file__).resolve().parents[2] / "core/state.py"
    (out / "snapshot_core_state.py").write_bytes(state_source.read_bytes())
    files = ["README.md", "contract_diagnostics.json", "lane_summary.csv",
             "protocol.json", "lane_contract_v2.json", "snapshot_lane_contract.py",
             "snapshot_core_receipts.py", "snapshot_core_state.py"]
    source_hashes = {
        "tasks/wilson_loop/lane_contract.py": hashlib.sha256(
            source_snapshot.read_bytes()).hexdigest(),
        "core/receipts.py": hashlib.sha256(
            (out / "snapshot_core_receipts.py").read_bytes()).hexdigest(),
        "core/state.py": hashlib.sha256(
            (out / "snapshot_core_state.py").read_bytes()).hexdigest(),
    }
    scenario = scenario_spec(
        "wilson-two-lane-contract-fixtures",
        "one fixture-lane structural check",
        "Synthetic contract validation only; no Wilson measurements or production inputs.",
        assumptions=[
            "fixtures are software checks and are not independent physical observations",
            "structural validity does not imply physics validity or certification",
        ],
        source_hashes=source_hashes,
    )
    denominators = {}
    reason_rows = []
    for row in summary_rows:
        unit = f"{row['fixture']}::{row['lane']}"
        denominators[unit] = {
            "unit": "one fixture-lane structural check",
            "planned": row["planned"],
            "valid": row["valid"],
            "failed": row["failed"],
            "abstained": row["abstained"],
        }
        if row["failed"]:
            issues = results[row["fixture"]]["lanes"][row["lane"]]["issues"]
            reason_rows.append({
                "unit": unit,
                "state": "failed",
                "count": row["failed"],
                "reason": "; ".join(issues) or "structural validation failed",
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
        "reason": "physical certification awaits independent C/F receipts and explicit authorization",
    })
    receipt_payload_hashes = {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in files
    }
    receipt = run_receipt(
        scenario, denominators, artifact_hashes=receipt_payload_hashes,
        text_flow=(
            "synthetic contract fixture -> separate C/F lane validation -> "
            "fixture-lane denominator and blocker reasons -> hashed report; "
            "complete means accounting only, not scientific success; human physics "
            "review and authorized measurement remain later gates"
        ),
        reason_rows=reason_rows,
    )
    (out / "run_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    hashes = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
              for name in files + ["run_receipt.json"]}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return {"status": protocol["status"], "physical_score": None,
            "run_state": run_state,
            "fixtures": results,
            "receipt": receipt,
            "files": files + ["run_receipt.json", "artifact_hashes.json"]}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = build(args.output)
    concise = {"status": summary["status"],
               "physical_score": summary["physical_score"],
               "lane_statuses": {
                   fixture: {lane: row["status"] for lane, row in result["lanes"].items()}
                   for fixture, result in summary["fixtures"].items()
               }}
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
