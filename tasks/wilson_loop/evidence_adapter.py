"""Lane-specific gates for Wilson finite-T Creutz and ground-state force.

The C and F estimands use separate receipts, expected fields, denominators,
and RunState instances. By default each gate requires explicit source paths or
immutable snapshots before a complete bundle can advance only to ``draft``.
A reader can consume and pass the same snapshot to its gate. Neither route
authenticates source authority, proves reviewer independence, or certifies
either physical result. Metadata-only mode is for synthetic tests.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from numbers import Real
from pathlib import Path
from typing import Any

from core.evidence import (
    apply_evidence_bundle_gate,
    apply_evidence_bundle_gate_with_source_bytes,
    apply_evidence_gate,
    apply_evidence_gate_with_source_bytes,
)
from core.receipts import RECEIPT_SCHEMA, run_receipt as normalize_run_receipt
from core.source_integrity import SOURCE_VERIFICATION_SCHEMA, SourceByteSnapshot
from .lane_contract import LATTICE_FORCE_UNITS, selected_protocol


TASK = "wilson_loop"
CONTRACT_PATH = Path(__file__).with_name("lane_contract_v2.json")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_DRAWS_PER_BLOCK_LENGTH = 10_000
_REQUIRED_BLOCK_LENGTHS = 3
_PAIRED_DRAWS = _DRAWS_PER_BLOCK_LENGTH * _REQUIRED_BLOCK_LENGTHS
_C_ALPHA = 0.025
_F_ALPHA = 0.025
_F_SYSTEMATIC_ALPHA = 0.0125
_F_SYSTEMATIC_CELLS = 28
_C_SOURCE_KEYS = frozenset({
    "manifest", "loop_map", "loop_measurements", "dependence_receipt",
    "coverage_calibration_receipt", "interval_summary", "review_receipt",
    "authorization_receipt",
})
_F_ROUTES = (
    "force_interval", "positive_force_scale", "q_excited",
    "h_thermal_nt32", "x_mixed", "spectral_model_truncation",
)
_F_SOURCE_KEYS = {
    "force_interval": frozenset({
        "manifest", "observable_map", "data_receipt", "force_summary",
        "dependence_receipt", "coverage_calibration_receipt",
        "review_receipt", "authorization_receipt",
    }),
    "positive_force_scale": frozenset({
        "manifest", "observable_map", "data_receipt", "force_summary",
        "dependence_receipt", "coverage_calibration_receipt",
        "review_receipt", "authorization_receipt",
    }),
    "q_excited": frozenset({
        "manifest", "observable_map", "data_receipt", "gevp_receipt",
        "component_receipt", "propagation_receipt", "force_summary",
        "review_receipt", "authorization_receipt",
    }),
    "h_thermal_nt32": frozenset({
        "manifest", "observable_map", "data_receipt", "gevp_receipt",
        "matched_nt32_receipt", "component_receipt", "propagation_receipt",
        "force_summary", "review_receipt", "authorization_receipt",
    }),
    "x_mixed": frozenset({
        "manifest", "observable_map", "data_receipt", "gevp_receipt",
        "component_receipt", "propagation_receipt", "force_summary",
        "review_receipt", "authorization_receipt",
    }),
    "spectral_model_truncation": frozenset({
        "manifest", "observable_map", "data_receipt", "gevp_receipt",
        "component_receipt", "propagation_receipt", "force_summary",
        "review_receipt", "authorization_receipt",
    }),
}
_F_UNITS = {
    "force_interval": LATTICE_FORCE_UNITS,
    "positive_force_scale": LATTICE_FORCE_UNITS,
    "q_excited": "dimensionless q bound",
    "h_thermal_nt32": "dimensionless h bound",
    "x_mixed": "dimensionless x bound",
    "spectral_model_truncation": LATTICE_FORCE_UNITS,
}
_F_KINDS = {
    "force_interval": "ground_state_force_interval",
    "positive_force_scale": "positive_absolute_force_scale",
    "q_excited": "excited_state_q_envelope",
    "h_thermal_nt32": "matched_nt32_thermal_h_envelope",
    "x_mixed": "mixed_sector_x_envelope",
    "spectral_model_truncation": "spectral_model_truncation_envelope",
}


def _canonical_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def creutz_map_sha256() -> str:
    """Hash the selected finite-T Creutz point, stencil, and sign convention."""
    protocol = selected_protocol()
    return _canonical_hash({
        "protocol_id": protocol["protocol_id"],
        "finite_T_Creutz": protocol["finite_T_Creutz"],
        "lane_contract": json.loads(CONTRACT_PATH.read_text())["lanes"]["finite_T_Creutz"],
    })


def force_map_sha256(route: str) -> str:
    """Hash the selected force stencil plus one lane-specific evidence role."""
    if route not in _F_ROUTES:
        raise ValueError(f"unknown Wilson F route {route!r}")
    protocol = selected_protocol()
    force = protocol["ground_state_discrete_static_force"]
    return _canonical_hash({
        "protocol_id": protocol["protocol_id"],
        "estimand": force["estimand"],
        "formula": force["formula"],
        "reference_formula": force["reference_formula"],
        "reference_role": force["reference_role"],
        "units": force["units"],
        "error_propagation": force["error_propagation"],
        "spectral_evidence_plan": force["spectral_evidence_plan"],
        "lane_contract": json.loads(CONTRACT_PATH.read_text())[
            "lanes"]["ground_state_discrete_static_force"],
        "role": route,
        "support_unit": _F_UNITS[route],
    })


def _finite(value: object) -> bool:
    return (isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _hash(value: object) -> bool:
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _abstain(run_state: object, reasons: list[str], status: str):
    stage = "statistical"
    if (not hasattr(run_state, "status") or stage not in run_state.status
            or not hasattr(run_state, "summary") or not hasattr(run_state, "abstain")):
        raise ValueError("run_state must be a core.RunState-compatible object")
    if run_state.status[stage] != "not_run":
        return {
            "status": "ABSTAIN_RUNSTATE_STAGE_ALREADY_USED",
            "reasons": [f"{stage} is already {run_state.status[stage]}"],
            "run_state": run_state.summary(), "certified": False,
            "physical_score": None,
        }
    run_state.abstain(stage, "; ".join(reasons) or status)
    return {
        "status": status, "reasons": reasons,
        "run_state": run_state.summary(), "certified": False,
        "physical_score": None,
    }


def _check_sources(value: object, required: frozenset[str], label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} source hashes must be a mapping"]
    reasons = []
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required)
    if missing:
        reasons.append(f"{label} source hashes missing {missing}")
    if extra:
        reasons.append(f"{label} source hashes contain unexpected keys {extra}")
    for name, digest in value.items():
        if not _hash(digest):
            reasons.append(f"{label} source hash {name!r} is not lowercase SHA-256")
    return reasons


def _missing_source_byte_receipt(value: object) -> dict[str, Any]:
    """Account for every expected source when no root/path map was supplied."""
    expected: dict[str, object] = {}
    if isinstance(value, dict):
        if value and all(isinstance(item, dict) for item in value.values()):
            for route, hashes in value.items():
                expected.update({f"{route}::{name}": digest for name, digest in hashes.items()})
        else:
            expected = {str(name): digest for name, digest in value.items()}
    entries = {
        name: {
            "status": "abstained",
            "path": None,
            "expected_sha256": digest if _hash(digest) else None,
            "observed_sha256": None,
            "reason": "source_root_or_path_map_missing",
        }
        for name, digest in expected.items()
    }
    planned = len(entries)
    return {
        "schema": SOURCE_VERIFICATION_SCHEMA,
        "status": "incomplete",
        "denominator": {
            "unit": "source_file", "planned": planned, "valid": 0,
            "failed": 0, "abstained": planned, "not_run": 0,
        },
        "entries": entries,
    }


def _check_receipt(receipt: object, expected: dict[str, int]) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
        return None, [f"receipt.schema must be {RECEIPT_SCHEMA!r}"]
    try:
        normalized = normalize_run_receipt(
            receipt.get("scenario"), receipt.get("denominators"),
            artifact_hashes=receipt.get("artifact_hashes"),
            text_flow=receipt.get("text_flow"),
            reason_rows=receipt.get("reason_rows"),
        )
    except (TypeError, ValueError) as exc:
        return None, [str(exc)]
    reasons = []
    if receipt.get("status") != normalized["status"]:
        reasons.append("receipt.status does not match denominator counts")
    denominators = normalized["denominators"]
    if set(denominators) != set(expected):
        reasons.append("receipt denominator names do not exactly match the lane plan")
    for name, planned in expected.items():
        item = denominators.get(name)
        if item is None:
            continue
        if item["planned"] != planned:
            reasons.append(f"{name}: planned {item['planned']}, expected {planned}")
        if item["not_run"]:
            reasons.append(f"{name}: {item['not_run']} units remain not_run")
    return normalized, reasons


def _link_sources(envelope: object, hashes: object, *, manifest_key: str,
                  map_key: str, summary_key: str) -> list[str]:
    if not isinstance(envelope, dict) or not isinstance(hashes, dict):
        return ["envelope and source hashes must be mappings"]
    chain = envelope.get("chain")
    observable = envelope.get("observable")
    review = envelope.get("review")
    source = envelope.get("source_hashes")
    if not all(isinstance(item, dict) for item in (chain, observable, review, source)):
        return ["envelope chain, observable, review, and source_hashes must be mappings"]
    reasons = []
    if hashes.get(manifest_key) != chain.get("manifest_sha256"):
        reasons.append(f"{manifest_key} does not bind the chain manifest")
    if hashes.get(map_key) != observable.get("map_sha256"):
        reasons.append(f"{map_key} does not bind the observable map")
    if hashes.get("review_receipt") != review.get("receipt_sha256"):
        reasons.append("review_receipt does not bind the envelope review")
    if hashes.get(summary_key) not in source.values():
        reasons.append(f"{summary_key} is not included in envelope source_hashes")
    if source != hashes:
        reasons.append("envelope source_hashes differ from the required source-hash map")
    return reasons


def _summary_hash_matches(hashes: object, summary: object, key: str = "summary_sha256") -> list[str]:
    if not isinstance(hashes, dict) or not isinstance(summary, dict):
        return ["lane summary or source hashes are missing"]
    if not _hash(summary.get(key)) or hashes.get("interval_summary", hashes.get("force_summary")) != summary[key]:
        return ["lane summary hash does not match its source receipt"]
    return []


def _prerequisites(run_state: object, required: tuple[str, ...]) -> list[str]:
    reasons = []
    for stage in required:
        if getattr(run_state, "status", {}).get(stage) != "validated":
            reasons.append(f"upstream {stage} stage must be validated")
    return reasons


def _open_calibration_split(envelope: object) -> list[str]:
    if not isinstance(envelope, dict) or not isinstance(envelope.get("chain"), dict):
        return ["chain metadata is missing"]
    split_id = str(envelope["chain"].get("split_id", "")).lower()
    forbidden = ("confirmation", "validation", "shared-holdout", "shared_holdout", "sealed", "gap")
    if any(term in split_id for term in forbidden):
        return ["sealed, gap, validation, shared-holdout, and confirmation splits are closed"]
    if "calibration" not in split_id:
        return ["lane evidence must use an explicitly named calibration split"]
    return []


def apply_wilson_creutz_gate(
    run_state: object,
    envelope: object,
    receipt: object,
    interval_summary: object,
    *,
    required_source_hashes: object,
    source_root: str | Path | None = None,
    source_paths: dict[str, str] | None = None,
    source_snapshot: SourceByteSnapshot | None = None,
    require_source_bytes: bool = True,
) -> dict[str, Any]:
    """Check the selected finite-T Creutz C lane, keeping it separate from F."""
    reasons = _prerequisites(run_state, ("operator", "numerical"))
    reasons.extend(_check_sources(required_source_hashes, _C_SOURCE_KEYS, "Wilson C"))
    normalized, receipt_reasons = _check_receipt(
        receipt,
        {"four_required_loop_means": 4, "paired_bootstrap_draws": _PAIRED_DRAWS},
    )
    reasons.extend(receipt_reasons)
    if normalized is None or not isinstance(normalized.get("scenario"), dict) or normalized["scenario"].get("name") != "wilson-finite-t-creutz-calibration":
        reasons.append("C receipt scenario must be wilson-finite-t-creutz-calibration")
    counts = {} if normalized is None else normalized["denominators"]
    means = counts.get("four_required_loop_means")
    draws = counts.get("paired_bootstrap_draws")
    if means and (means["valid"] != 4 or means["failed"] or means["abstained"]):
        reasons.append("all four C-lane ensemble loop means must be valid and positive")
    if not isinstance(interval_summary, dict):
        reasons.append("C interval summary is missing")
    else:
        if interval_summary.get("units") != "dimensionless":
            reasons.append("C interval summary units must be dimensionless")
        keys = ("hull_lower", "hull_upper", "halfwidth", "invalid_draws", "planned_draws")
        for key in keys:
            if not _finite(interval_summary.get(key)):
                reasons.append(f"C interval summary {key} must be finite")
        if all(_finite(interval_summary.get(key)) for key in keys):
            lower = float(interval_summary["hull_lower"])
            upper = float(interval_summary["hull_upper"])
            halfwidth = float(interval_summary["halfwidth"])
            invalid_value = interval_summary["invalid_draws"]
            planned_value = interval_summary["planned_draws"]
            if (isinstance(invalid_value, bool) or not isinstance(invalid_value, int)
                    or isinstance(planned_value, bool) or not isinstance(planned_value, int)):
                reasons.append("C invalid_draws and planned_draws must be integers")
                invalid, planned = -1, -1
            else:
                invalid, planned = invalid_value, planned_value
            if upper < lower or abs(halfwidth - (upper - lower) / 2.0) > 1e-10:
                reasons.append("C reported halfwidth must equal half the interval-hull width")
            if planned != _PAIRED_DRAWS or (draws and planned != draws["planned"]):
                reasons.append("C interval summary planned draw count differs from the frozen 3x10,000 plan")
            if invalid < 0 or invalid / _PAIRED_DRAWS > 0.01:
                reasons.append("C invalid paired-draw fraction exceeds the selected 1% limit")
            if halfwidth > 0.02:
                reasons.append("C interval halfwidth exceeds the selected 0.02 dimensionless target")
            if draws and draws["failed"] + draws["abstained"] != invalid:
                reasons.append("C invalid draw summary differs from failed plus abstained denominator")
    reasons.extend(_summary_hash_matches(required_source_hashes, interval_summary))
    reasons.extend(_open_calibration_split(envelope))
    reasons.extend(_link_sources(
        envelope, required_source_hashes, manifest_key="manifest", map_key="loop_map",
        summary_key="interval_summary",
    ))
    if isinstance(envelope, dict):
        chain = envelope.get("chain", {})
        observable = envelope.get("observable", {})
        method = envelope.get("method", {})
        confidence = envelope.get("confidence", {})
        dependence = envelope.get("dependence", {})
        if not isinstance(chain, dict) or chain.get("state_unit") != "paired saved gauge configuration":
            reasons.append("C chain state_unit must be paired saved gauge configuration")
        if not isinstance(observable, dict):
            reasons.append("C observable metadata is missing")
        else:
            support = observable.get("support", {})
            if not isinstance(support, dict):
                reasons.append("C observable support must be a mapping")
                support = {}
            if observable.get("map_id") != "wilson-finite-t-creutz-R2-T5-stout-v1":
                reasons.append("C observable map_id differs from selected R=2,T=5 stout Creutz point")
            if observable.get("map_sha256") != creutz_map_sha256():
                reasons.append("C observable map hash differs from selected C-lane formula and sign")
            if support.get("unit") != "dimensionless finite-T Creutz observable":
                reasons.append("C observable must be dimensionless finite-T Creutz")
        if not isinstance(method, dict) or method.get("name") != "paired_circular_block_bootstrap_hull":
            reasons.append("C method must bind the selected paired block-bootstrap interval hull")
        if not isinstance(confidence, dict) or (
            confidence.get("family_id"), confidence.get("alpha_total"),
            confidence.get("alpha_allocated"), confidence.get("family_size"),
        ) != ("wilson-two-lane-selected-v3", 0.05, _C_ALPHA, 2):
            reasons.append("C confidence allocation must be the selected 0.025 of the two-lane 0.05 family")
        if not isinstance(dependence, dict) or dependence.get("mode") not in ("markov", "finite_dependence"):
            reasons.append("C interval requires a reviewed Markov-gap or finite-dependence route")
        if isinstance(dependence, dict) and dependence.get("mode") == "markov":
            if (not _finite(dependence.get("gamma_ps_lower"))
                    or not 0 < float(dependence["gamma_ps_lower"]) <= 1
                    or not _finite(dependence.get("tv_bound_after_burn_in"))
                    or not 0 <= float(dependence["tv_bound_after_burn_in"]) < 0.025):
                reasons.append("C Markov route needs a positive gap and TV bound below allocated alpha")
        if isinstance(dependence, dict) and dependence.get("mode") == "finite_dependence":
            if not isinstance(dependence.get("dependence_radius"), int):
                reasons.append("C finite-dependence route needs an integer dependence radius")
    if reasons:
        return _abstain(run_state, reasons, "ABSTAIN_WILSON_C_LANE")
    expected = {
        "task": TASK,
        "evidence_kind": "finite_t_creutz_interval",
        "observable.map_id": "wilson-finite-t-creutz-R2-T5-stout-v1",
        "observable.map_sha256": creutz_map_sha256(),
        "observable.support.unit": "dimensionless finite-T Creutz observable",
        "method.name": "paired_circular_block_bootstrap_hull",
        "confidence.family_id": "wilson-two-lane-selected-v3",
        "confidence.alpha_total": 0.05,
        "confidence.alpha_allocated": _C_ALPHA,
        "confidence.family_size": 2,
    }
    if (require_source_bytes is False and source_root is None and source_paths is None
            and source_snapshot is None):
        return apply_evidence_gate(
            run_state, "statistical", envelope, receipt, expected=expected,
            required_source_hashes=required_source_hashes,
        )
    if source_snapshot is not None:
        if source_root is not None or source_paths is not None:
            result = _abstain(
                run_state,
                ["Wilson C pass a source snapshot or source_root/source_paths, not both"],
                "ABSTAIN_WILSON_C_LANE",
            )
            result["source_verification"] = _missing_source_byte_receipt(required_source_hashes)
            return result
        return apply_evidence_gate_with_source_bytes(
            run_state, "statistical", envelope, receipt,
            source_snapshot=source_snapshot,
            expected=expected, required_source_hashes=required_source_hashes,
        )
    if source_root is None or source_paths is None:
        result = _abstain(
            run_state,
            ["Wilson C source-byte verification requires source_root and an explicit source_paths map"],
            "ABSTAIN_WILSON_C_LANE",
        )
        result["source_verification"] = _missing_source_byte_receipt(required_source_hashes)
        return result
    return apply_evidence_gate_with_source_bytes(
        run_state, "statistical", envelope, receipt,
        source_root=str(source_root), source_paths=source_paths,
        expected=expected, required_source_hashes=required_source_hashes,
    )


def _force_expected(route: str, envelope: object) -> dict[str, Any]:
    dependence = envelope.get("dependence", {}) if isinstance(envelope, dict) else {}
    if route in ("force_interval", "positive_force_scale"):
        confidence_alpha = 0.00625
        confidence_total = 0.0125
        confidence_family = "wilson-two-lane-selected-v3/F/statistical"
        family_size = 2
    else:
        cells = 4 if route == "spectral_model_truncation" else 8
        confidence_alpha = _F_SYSTEMATIC_ALPHA * cells / _F_SYSTEMATIC_CELLS
        confidence_total = _F_SYSTEMATIC_ALPHA
        confidence_family = "wilson-two-lane-selected-v3/F/systematic"
        family_size = cells
    method_name = (
        "paired_circular_block_bootstrap_hull"
        if route in ("force_interval", "positive_force_scale")
        else "joint_3x3_finite_Nt_spectral_envelope"
    )
    # Bound envelopes are deterministic transformations of the referenced
    # spectral/thermal evidence; chain dependence remains encoded in receipts.
    mode = dependence.get("mode") if isinstance(dependence, dict) else None
    return {
        "task": TASK,
        "evidence_kind": _F_KINDS[route],
        "observable.map_id": f"wilson-ground-state-force-R2.5-v2/{route}",
        "observable.map_sha256": force_map_sha256(route),
        "observable.support.unit": _F_UNITS[route],
        "method.name": method_name,
        "confidence.family_id": confidence_family,
        "confidence.alpha_total": confidence_total,
        "confidence.alpha_allocated": confidence_alpha,
        "confidence.family_size": family_size,
        "dependence.mode": mode,
    }


def apply_wilson_force_gate(
    run_state: object,
    envelopes: object,
    receipt: object,
    force_summary: object,
    *,
    required_source_hashes_by_envelope: object,
    source_root: str | Path | None = None,
    source_paths_by_envelope: dict[str, dict[str, str]] | None = None,
    source_snapshots_by_envelope: dict[str, SourceByteSnapshot] | None = None,
    require_source_bytes: bool = True,
) -> dict[str, Any]:
    """Check the F-lane bundle with force-unit propagated bounds.

    The receipt must account for both 30,000-draw interval hulls, all 36
    planned GEVP fit variants, all 28 simultaneous systematic cells, and the
    matched ``N_t=32`` thermal evidence. It uses a lane-specific RunState.
    """
    reasons: list[str] = _prerequisites(run_state, ("operator", "reference", "numerical"))
    if not isinstance(envelopes, dict) or set(envelopes) != set(_F_ROUTES):
        reasons.append("F bundle must contain all six selected force/spectrum/bound routes")
        envelopes = envelopes if isinstance(envelopes, dict) else {}
    source_routes = required_source_hashes_by_envelope
    if not isinstance(source_routes, dict) or set(source_routes) != set(_F_ROUTES):
        reasons.append("F source-hash bundle must contain exactly the seven selected routes")
        source_routes = source_routes if isinstance(source_routes, dict) else {}
    for route in _F_ROUTES:
        reasons.extend(_check_sources(source_routes.get(route), _F_SOURCE_KEYS[route], f"F {route}"))

    normalized, receipt_reasons = _check_receipt(
        receipt,
        {
            "force_interval_draws": _PAIRED_DRAWS,
            "positive_force_scale_draws": _PAIRED_DRAWS,
            "gevp_fit_variants": 36,
            "force_systematic_cells": _F_SYSTEMATIC_CELLS,
            "matched_nt32_thermal_receipts": 1,
        },
    )
    reasons.extend(receipt_reasons)
    if normalized is None or not isinstance(normalized.get("scenario"), dict) or normalized["scenario"].get("name") != "wilson-ground-state-force-calibration":
        reasons.append("F receipt scenario must be wilson-ground-state-force-calibration")
    counts = {} if normalized is None else normalized["denominators"]
    for name in ("force_interval_draws", "positive_force_scale_draws"):
        item = counts.get(name)
        if item and item["failed"] + item["abstained"] > 0.01 * item["planned"]:
            reasons.append(f"{name}: invalid draw fraction exceeds the selected 1% limit")
    if isinstance(force_summary, dict):
        invalid_summary = {
            "force_interval_draws": force_summary.get("invalid_force_interval_draws"),
            "positive_force_scale_draws": force_summary.get("invalid_force_scale_draws"),
        }
        for name, value in invalid_summary.items():
            item = counts.get(name)
            if isinstance(value, bool) or not isinstance(value, int):
                reasons.append(f"F summary {name} invalid count must be an integer")
            elif item and value != item["failed"] + item["abstained"]:
                reasons.append(f"F summary {name} disagrees with its receipt denominator")
    for name in ("gevp_fit_variants", "force_systematic_cells", "matched_nt32_thermal_receipts"):
        item = counts.get(name)
        if item and (item["valid"] != item["planned"] or item["failed"] or item["abstained"]):
            reasons.append(f"{name}: every planned F-lane evidence unit must be valid")

    reasons.extend(_summary_hash_matches(
        source_routes.get("force_interval"), force_summary, key="summary_sha256"
    ))
    if not isinstance(force_summary, dict):
        reasons.append("F force summary is missing")
    else:
        if force_summary.get("units") != LATTICE_FORCE_UNITS:
            reasons.append(f"F summary units must be {LATTICE_FORCE_UNITS!r}")
        reference = force_summary.get("positive_force_scale_lower")
        estimate = force_summary.get("stout_force_estimate")
        interval_lower = force_summary.get("stout_force_interval_lower")
        interval_upper = force_summary.get("stout_force_interval_upper")
        stat = force_summary.get("statistical_halfwidth")
        components = force_summary.get("force_component_bounds")
        expected_components = {
            "excited_state_force": 0.015,
            "finite_T_thermal": 0.015,
            "mixed": 0.01,
            "spectral_model_truncation": 0.01,
        }
        if not _finite(reference) or float(reference) <= 0:
            reasons.append("F lane requires a positive simultaneous stout-force scale lower bound")
        if (not _finite(estimate) or not _finite(interval_lower)
                or not _finite(interval_upper)
                or float(interval_lower) <= 0
                or not float(interval_lower) <= float(estimate) <= float(interval_upper)):
            reasons.append("F summary must provide a positive stout-force interval containing its estimate")
        elif _finite(reference) and float(reference) > float(interval_lower):
            reasons.append("F force-scale lower bound must not exceed the stout interval lower bound")
        if not _finite(stat) or float(stat) < 0:
            reasons.append("F statistical halfwidth must be finite and nonnegative in lattice-force units")
        elif (_finite(estimate) and _finite(interval_lower) and _finite(interval_upper)
              and abs(float(stat) - max(float(estimate) - float(interval_lower),
                                        float(interval_upper) - float(estimate)))
              > 1e-10 * max(1.0, abs(float(stat)))):
            reasons.append("F statistical halfwidth must equal the stout interval's maximum endpoint deviation")
        if not isinstance(components, dict) or set(components) != set(expected_components):
            reasons.append("F summary must report all four propagated force-unit systematic components")
        if _finite(reference) and float(reference) > 0 and _finite(stat) and isinstance(components, dict):
            ref = float(reference)
            component_values = []
            for name, fraction in expected_components.items():
                value = components.get(name)
                if not _finite(value) or float(value) < 0:
                    reasons.append(f"F component {name} must be finite, nonnegative force")
                    continue
                value = float(value)
                component_values.append(value)
                if value > fraction * ref:
                    reasons.append(f"F component {name} exceeds its {fraction:.1%} reference-force budget")
            total_systematic = force_summary.get("systematic_total_bound")
            total_linear = force_summary.get("total_linear_bound")
            if not _finite(total_systematic) or not _finite(total_linear):
                reasons.append("F systematic and total linear bounds must be finite")
            elif len(component_values) == len(expected_components):
                expected_systematic = sum(component_values)
                if abs(float(total_systematic) - expected_systematic) > 1e-10 * max(1.0, expected_systematic):
                    reasons.append("F systematic total must equal the sum of its four force-unit components")
                if float(total_systematic) > 0.05 * ref:
                    reasons.append("F total systematic bound exceeds 5% of the positive reference-force lower bound")
                if float(stat) > 0.05 * ref:
                    reasons.append("F statistical halfwidth exceeds 5% of the reference-force lower bound")
                if abs(float(total_linear) - (expected_systematic + float(stat))) > 1e-10 * max(1.0, expected_systematic + float(stat)):
                    reasons.append("F total linear bound must equal systematic plus statistical bounds")
                if float(total_linear) > 0.10 * ref:
                    reasons.append("F total linear error exceeds 10% of the reference-force lower bound")

    expected_by_route: dict[str, dict[str, Any]] = {}
    if set(envelopes) == set(_F_ROUTES) and set(source_routes) == set(_F_ROUTES):
        for route in _F_ROUTES:
            envelope = envelopes[route]
            hashes = source_routes[route]
            reasons.extend(_link_sources(
                envelope, hashes, manifest_key="manifest", map_key="observable_map",
                summary_key="force_summary",
            ))
            if not isinstance(envelope, dict):
                continue
            chain = envelope.get("chain", {})
            observable = envelope.get("observable", {})
            split_id = str(chain.get("split_id", "")).lower() if isinstance(chain, dict) else ""
            if not split_id or "calibration" not in split_id:
                reasons.append(f"F {route} must use an explicitly named calibration split")
            if isinstance(chain, dict) and any(
                token in split_id
                for token in ("confirmation", "validation", "shared_holdout", "shared-holdout", "sealed", "gap")
            ):
                reasons.append(f"F {route} uses a closed split")
            if not isinstance(observable, dict):
                reasons.append(f"F {route} observable metadata is missing")
            else:
                if observable.get("map_id") != f"wilson-ground-state-force-R2.5-v2/{route}":
                    reasons.append(f"F {route} observable map_id is not the selected route")
                if observable.get("map_sha256") != force_map_sha256(route):
                    reasons.append(f"F {route} map hash does not bind the selected force stencil")
                support = observable.get("support", {})
                if not isinstance(support, dict):
                    reasons.append(f"F {route} observable support must be a mapping")
                    support = {}
                if support.get("unit") != _F_UNITS[route]:
                    reasons.append(f"F {route} support unit must be {_F_UNITS[route]!r}")
                if not _finite(support.get("lower")) or not _finite(support.get("upper")) or support["upper"] <= support["lower"]:
                    reasons.append(f"F {route} support must be finite and nonempty")
                if route in ("positive_force_scale", "q_excited", "h_thermal_nt32", "x_mixed") and _finite(support.get("lower")) and support["lower"] < 0:
                    reasons.append(f"F {route} support must be nonnegative")
            expected = _force_expected(route, envelope)
            expected_by_route[route] = expected
        if set(source_routes) == set(_F_ROUTES):
            for key in ("force_summary", "review_receipt", "authorization_receipt"):
                if len({source_routes[route].get(key) for route in _F_ROUTES}) != 1:
                    reasons.append(f"F bundle must share one lane-level {key}")
        for route in ("force_interval", "positive_force_scale"):
            env = envelopes.get(route)
            if isinstance(env, dict):
                dep = env.get("dependence", {})
                if not isinstance(dep, dict) or dep.get("mode") not in ("markov", "finite_dependence"):
                    reasons.append(f"F {route} requires a reviewed Markov-gap or finite-dependence route")
        for route in ("q_excited", "x_mixed", "spectral_model_truncation"):
            env = envelopes.get(route)
            if isinstance(env, dict):
                dep = env.get("dependence", {})
                if not isinstance(dep, dict) or dep.get("mode") != "spectral":
                    reasons.append(f"F {route} must declare the spectral evidence route")
        thermal = envelopes.get("h_thermal_nt32")
        if isinstance(thermal, dict):
            dep = thermal.get("dependence", {})
            if not isinstance(dep, dict) or dep.get("mode") not in (
                "markov", "finite_dependence", "spectral"
            ):
                reasons.append("F matched Nt=32 thermal evidence must declare its dependence route")
        if (set(source_routes) == set(_F_ROUTES)
                and all(isinstance(source_routes.get(route), dict) for route in _F_ROUTES)):
            gevp_hashes = {
                source_routes[route].get("gevp_receipt")
                for route in ("q_excited", "h_thermal_nt32", "x_mixed", "spectral_model_truncation")
            }
            if len(gevp_hashes) != 1:
                reasons.append("all F spectral-contamination routes must bind the same 3x3 GEVP receipt")
            thermal_hash = source_routes["h_thermal_nt32"].get("matched_nt32_receipt")
            if not _hash(thermal_hash):
                reasons.append("F thermal route must bind the matched Nt=32 receipt")
        force_chain = envelopes.get("force_interval")
        reference_chain = envelopes.get("positive_force_scale")
        if isinstance(force_chain, dict) and isinstance(reference_chain, dict):
            left = force_chain.get("chain", {})
            right = reference_chain.get("chain", {})
            if isinstance(left, dict) and isinstance(right, dict):
                if (left.get("chain_id"), left.get("manifest_sha256"), left.get("split_id")) != (
                    right.get("chain_id"), right.get("manifest_sha256"), right.get("split_id")
                ):
                    reasons.append("F interval and positive-force-scale routes must preserve their paired source chain")

    if reasons:
        return _abstain(run_state, reasons, "ABSTAIN_WILSON_F_LANE")
    if (require_source_bytes is False and source_root is None
            and source_paths_by_envelope is None and source_snapshots_by_envelope is None):
        return apply_evidence_bundle_gate(
            run_state, "statistical", envelopes, receipt,
            expected_by_envelope=expected_by_route,
            required_source_hashes_by_envelope=source_routes,
        )
    if source_snapshots_by_envelope is not None:
        if source_root is not None or source_paths_by_envelope is not None:
            result = _abstain(
                run_state,
                ["Wilson F pass per-envelope snapshots or source_root/source_paths, not both"],
                "ABSTAIN_WILSON_F_LANE",
            )
            result["source_verification"] = _missing_source_byte_receipt(source_routes)
            return result
        return apply_evidence_bundle_gate_with_source_bytes(
            run_state, "statistical", envelopes, receipt,
            source_snapshots_by_envelope=source_snapshots_by_envelope,
            expected_by_envelope=expected_by_route,
            required_source_hashes_by_envelope=source_routes,
        )
    if source_root is None or source_paths_by_envelope is None:
        result = _abstain(
            run_state,
            ["Wilson F source-byte verification requires source_root and an explicit per-route source_paths map"],
            "ABSTAIN_WILSON_F_LANE",
        )
        result["source_verification"] = _missing_source_byte_receipt(source_routes)
        return result
    return apply_evidence_bundle_gate_with_source_bytes(
        run_state, "statistical", envelopes, receipt,
        source_root=str(source_root),
        source_paths_by_envelope=source_paths_by_envelope,
        expected_by_envelope=expected_by_route,
        required_source_hashes_by_envelope=source_routes,
    )
