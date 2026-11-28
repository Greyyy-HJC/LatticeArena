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


SCHEMA = "wilson-two-lane-contract-v2"
DEFAULT_OUT = (Path(__file__).resolve().parents[2]
               / "results/wilson_loop/lane_contract_diagnostic_20261124_v5")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
F_BOUND_COMPONENTS = (
    "excited_state_force", "finite_T_thermal", "mixed", "finite_a",
    "finite_volume", "other_systematic",
)
QHX_TERMS = ("q", "h", "x")
QHX_UNITS = "dimensionless"
CREUTZ_UNITS = "dimensionless"


def empty_contract() -> dict[str, Any]:
    """Return a draft that preserves all unfrozen scientific choices as null."""
    return {
        "schema": SCHEMA,
        "status": "DRAFT_NOT_READY",
        "physical_score": None,
        "certified": False,
        "common": {
            "action_id": None,
            "static_source_action_id": None,
            "geometry": {"spatial_extents": None, "temporal_extent": None},
            "boundary_conditions": {"spatial": None, "temporal": None},
            "lattice_spacing": {"a_s": None, "a_t": None,
                                "length_unit": None, "energy_unit": None},
            "kernel_ids": None,
            "split_manifest_sha256": None,
            "paired_block_plan_sha256": None,
            "independent_review_receipt": None,
        },
        "lanes": {
            "finite_T_Creutz": {
                "estimand": "finite_T_Creutz",
                "primary": {"R": None, "T": None, "kernel_id": None},
                "stencil": None,
                "log_rule": None,
                "invalid_point_rule": None,
                "comparison_family": None,
                "inference": {
                    "interval_method": None, "alpha_total": None,
                    "alpha_allocations": None,
                    "precision_target": None, "precision_units": None,
                    "multiplicity_method": None,
                },
            },
            "ground_state_discrete_static_force": {
                "estimand": "ground_state_discrete_static_force_difference",
                "primary": {"R": None, "candidate_kernel_id": None,
                            "reference_kernel_id": None},
                "force_stencil": None,
                "units": None,
                "spectral_bounds": None,
                "systematic_bounds": None,
                "inference": {
                    "interval_method": None, "alpha_total": None,
                    "alpha_allocations": None,
                    "statistical_precision_target": None,
                    "precision_units": None,
                    "multiplicity_method": None,
                },
                "total_tolerance": None,
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


def _check_common(common: Any) -> list[str]:
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
    if lane.get("estimand") != "ground_state_discrete_static_force_difference":
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
    if _positive_int(r) and _nonempty(candidate) and _nonempty(reference):
        expected = {(r, candidate), (r + 1, candidate),
                    (r, reference), (r + 1, reference)}
        pattern = {(r + 1, candidate): 1, (r, candidate): -1,
                   (r + 1, reference): -1, (r, reference): 1}
        rows = lane.get("force_stencil")
        if not isinstance(rows, list) or len(rows) != 4:
            issues.append(f"{label}.force_stencil: expected four explicit force terms")
        else:
            observed: dict[tuple[int, str], Any] = {}
            for index, row in enumerate(rows):
                if not _mapping(row):
                    issues.append(f"{label}.force_stencil[{index}]: expected an object")
                    continue
                rr, kk, coeff = row.get("R"), row.get("kernel_id"), row.get("coefficient")
                if not _positive_int(rr) or not _nonempty(kk):
                    issues.append(f"{label}.force_stencil[{index}]: invalid R or kernel_id")
                    continue
                key = (rr, kk)
                if key in observed:
                    issues.append(f"{label}.force_stencil: duplicate term {key}")
                observed[key] = coeff
            if set(observed) != expected:
                issues.append(f"{label}.force_stencil: terms do not match candidate/reference force contrast")
            elif all(_is_number(value) for value in observed.values()):
                actual = {key: int(value) if float(value).is_integer() else value
                          for key, value in observed.items()}
                if actual != pattern:
                    issues.append(
                        f"{label}.force_stencil: coefficients must encode candidate minus reference"
                    )
            else:
                issues.append(f"{label}.force_stencil: coefficients must be finite numbers")

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
    common_issues = _check_common(common)
    if contract.get("schema") != SCHEMA:
        common_issues.append(f"contract.schema: expected {SCHEMA}")
    raw_lanes = contract.get("lanes")
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
            "estimand": "ground_state_discrete_static_force_difference",
            "primary": {"R": 2,
                        "candidate_kernel_id": "SYNTHETIC_ONLY_candidate",
                        "reference_kernel_id": "SYNTHETIC_ONLY_reference"},
            "force_stencil": [
                {"R": 3, "kernel_id": "SYNTHETIC_ONLY_candidate", "coefficient": 1},
                {"R": 2, "kernel_id": "SYNTHETIC_ONLY_candidate", "coefficient": -1},
                {"R": 3, "kernel_id": "SYNTHETIC_ONLY_reference", "coefficient": -1},
                {"R": 2, "kernel_id": "SYNTHETIC_ONLY_reference", "coefficient": 1},
            ],
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
    fixtures = {
        "current_null_draft": empty_contract(),
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
    (out / "contract_diagnostics.json").write_text(
        json.dumps({"schema": SCHEMA, "data_access": "none",
                    "physical_score": None, "certified": False,
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
        "The contract preserves the user-selected lanes separately: finite-`T` Creutz as its own observable and the ground-state discrete static-force difference as a distinct target. Structural completeness is not physics validation. Even a fully populated synthetic fixture can only reach `CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW`; the tool never sets a certified or ready state.",
        "",
        "## Text diagnostic and decision flow",
        "",
        "```text",
        "CURRENT NULL DRAFT",
        "  -> identify shared action, geometry, boundaries, units, kernels, split and paired-block receipts",
        "  -> Lane C: chi = -[log W(R+1,T+1)+log W(R,T)-log W(R+1,T)-log W(R,T+1)]; require dimensionless precision units and freeze primary R/T/kernel and remaining inference rules",
        "  -> Lane F: freeze candidate-minus-reference force stencil and units, q/h/x receipts, thermal/mixed/cutoff/volume bounds, alpha allocation and total error budget",
        "  -> schema-complete? no: remain NOT_READY and list blockers",
        "  -> schema-complete? yes: CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW only",
        "  -> independent physics review + authorized measurements + sealed confirmation are separate later gates",
        "  -> physical_score remains null until those gates are independently satisfied",
        "```",
        "",
        f"`lane_summary.csv` reports planned/valid/failed/abstained counts for each fixture and lane. `run_receipt.json` records the same {len(summary_rows)} fixture-lane checks, source hashes, artifact hashes, and one reason row for every failed schema check; `complete` means all planned checks are accounted for, not that a lane or method succeeded. `artifact_hashes.json` also covers the receipt. The C interval precision target must be labeled `{CREUTZ_UNITS}` because its signed log-Creutz observable is dimensionless. A missing bound is never treated as zero. Bound rows require a scope, hash-shaped evidence receipt and review receipt; those strings are checked only for shape and are not opened or authenticated here. q/h/x receipts are dimensionless spectral inputs; the force-budget check requires their separately propagated `excited_state_force` bound and sums that with thermal/mixed/finite-a/volume/other force-level systematic bounds and the statistical precision target. Familywise alpha allocations must sum to no more than `alpha_total`.",
        "",
        "## Fixture results",
        "",
        "|fixture|lane|planned|valid|failed|abstained|status|certified|score|blocker count|",
        "|---|---|---:|---:|---:|---:|---|---|---|---:|",
    ]
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
        "- The accepted primary `R/T/kernel`, precision/multiplicity choices and physical contamination limits remain unfrozen, so production `NOT_READY/UNCERTIFIED` is unchanged.",
        "",
        "### Fail-closed regression diagnostics",
        "",
        "- Reversing all four C coefficients is rejected because the declared convention is `chi = -[log W(R+1,T+1)+log W(R,T)-log W(R+1,T)-log W(R,T+1)]`, matching `finite_nt_toy.py`.",
        "- Reversing all four F coefficients is rejected because the contrast is candidate force minus reference force.",
        "- A C statistical precision target labeled with energy/force units is rejected because the four-point log-Creutz estimator is dimensionless.",
        "- An F statistical precision target labeled dimensionless is rejected against the force-unit estimand and is excluded from the force-unit budget sum until units match.",
        "- A null shared `kernel_ids` field produces explicit common blockers; the inspector does not crash or mark either lane complete.",
        "- These are synthetic schema mutations. They diagnose contract handling only and do not read Wilson data or estimate a physical uncertainty.",
        "",
        "## Next work",
        "",
        "1. Review the physical definitions and freeze C's primary rectangle, sign convention, kernel family, positive-loop rule, interval and multiplicity design.",
        "2. Freeze F's discrete-force contrast, then obtain independent per-kernel/per-R q/h/x, finite-temporal, mixed, lattice-spacing and volume evidence with a prospective error budget.",
        "3. Populate receipts from authorized production provenance; review them independently before any sealed evaluation.",
        "",
    ])
    (out / "README.md").write_text("\n".join(lines))
    protocol = {
        "schema": SCHEMA,
        "status": "STRUCTURE_DIAGNOSTIC_ONLY",
        "data_access": "none",
        "physical_score": None,
        "certified": False,
        "lane_estimands": ["finite_T_Creutz",
                           "ground_state_discrete_static_force_difference"],
        "fixed_sign_conventions": {
            "finite_T_Creutz": "minus the R-T rectangular log-W difference, matching finite_nt_toy.py",
            "ground_state_discrete_static_force_difference": "candidate force minus reference force",
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
    source_snapshot = out / "snapshot_lane_contract.py"
    source_snapshot.write_bytes(Path(__file__).read_bytes())
    receipt_source = Path(__file__).resolve().parents[2] / "core/receipts.py"
    (out / "snapshot_core_receipts.py").write_bytes(receipt_source.read_bytes())
    files = ["README.md", "contract_diagnostics.json", "lane_summary.csv",
             "protocol.json", "snapshot_lane_contract.py",
             "snapshot_core_receipts.py"]
    source_hashes = {
        "tasks/wilson_loop/lane_contract.py": hashlib.sha256(
            source_snapshot.read_bytes()).hexdigest(),
        "core/receipts.py": hashlib.sha256(
            (out / "snapshot_core_receipts.py").read_bytes()).hexdigest(),
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
