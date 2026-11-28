"""Optimize task gates for D04, bounded-block D06, and D11 evidence.

These adapters bind task metadata and denominators to the selected Optimize
contracts. By default they also require either an explicit root/path map or
immutable source snapshots before a successful transition can reach ``draft``.
Snapshot callers can use the same retained bytes in their reader and gate. The
explicit metadata-only option is for synthetic fixtures. Neither byte
matching nor metadata validates scientific content, source authority, or
reviewer independence; ``physical_score`` remains unset.
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
from core.source_integrity import (
    SOURCE_VERIFICATION_SCHEMA, SourceByteSnapshot, verify_source_snapshot,
)
from .bounded_block_risk import R_STAR
from .source_reader import read_json_object_snapshot


TASK = "optimize_gaussian"
CONTRACT_PATH = Path(__file__).with_name("research_contract_v1.json")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_RATIO = 10.0 / 9.0
_D06_ALPHA = 0.05
_D06_EVENTS = 3
_D06_ALPHA_PER_EVENT = _D06_ALPHA / _D06_EVENTS
_D04_FIT_VARIANTS = 3 * 2
_D06_ROUTES = ("same_budget_slsqp", "degree4_chebyshev", "trainable_w")
_D06_SOURCE_KEYS = frozenset({
    "manifest", "score_map", "score_matrix", "block_chain_receipt",
    "gamma_ps_receipt", "initial_law_tv_receipt", "review_receipt",
    "authorization_receipt", "budget_receipt", "cost_ledger",
})
_D04_SOURCE_KEYS = frozenset({
    "manifest", "correlator_receipt", "fit_map", "fit_diagnostics",
    "reference_summary", "dependence_receipt", "review_receipt",
    "authorization_receipt",
})
_D11_SOURCE_KEYS = frozenset({
    "link_manifest", "operator_map", "operator_summary", "kernel_source",
    "su3_receipt", "review_receipt", "authorization_receipt",
})
_D04_SUMMARY_FIELDS = frozenset({
    "reference_mass", "reference_mass_error", "relative_mass_error_max",
    "gof_p_min", "max_channel_or_window_shift_paired_se", "fit_variants",
    "boundary_hit_count", "summary_content_sha256",
})
_D11_SUMMARY_FIELDS = frozenset({
    "production_links", "su3_provenance_valid", "shape_and_periodic_boundary_valid",
    "unitarity_valid", "finite_values_valid", "bernstein_constraints_valid",
    "spectral_interval_valid", "clenshaw_reference_agrees", "summary_content_sha256",
})


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text())


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def summary_content_sha256(summary: dict[str, Any]) -> str:
    """Hash summary fields without a self-referential digest field."""
    if not isinstance(summary, dict):
        raise TypeError("summary must be a mapping")
    return _canonical_hash({
        key: value for key, value in summary.items()
        if key != "summary_content_sha256"
    })


def _check_summary_content(summary: object, *, fields: frozenset[str], label: str) -> list[str]:
    if not isinstance(summary, dict):
        return [f"{label} summary is missing"]
    reasons = []
    if set(summary) != fields:
        reasons.append(f"{label} summary keys do not match its frozen schema")
    digest = summary.get("summary_content_sha256")
    if not _valid_hash(digest):
        reasons.append(f"{label} summary_content_sha256 must be lowercase SHA-256")
    else:
        try:
            observed = summary_content_sha256(summary)
        except (TypeError, ValueError, OverflowError):
            reasons.append(f"{label} summary cannot be canonically serialized")
        else:
            if digest != observed:
                reasons.append(f"{label} summary content hash does not match its fields")
    return reasons


def d04_reference_map_sha256() -> str:
    """Hash the selected D04 reference fit and target definition."""
    contract = _contract()["D04_reference"]
    return _canonical_hash({
        key: contract[key]
        for key in ("observable", "units", "reference_method", "fit_protocol")
    })


def d11_operator_map_sha256() -> str:
    """Hash the selected D11 mathematical operator contract."""
    contract = _contract()["D11_strong_spectral_operator"]
    return _canonical_hash({
        key: value for key, value in contract.items()
        if key not in ("independent_review_receipt",)
    })


def bounded_risk_map_sha256(route: str) -> str:
    """Hash one bounded-block score or contrast map, including its role."""
    if route not in _D06_ROUTES:
        raise ValueError(f"unknown D06 route {route!r}")
    support = [0.0, 1.0] if route == "trainable_w" else [-R_STAR, 1.0]
    role = "trainable_w_mean" if route == "trainable_w" else f"contrast:{route}"
    return _canonical_hash({
        "schema": "optimize-bounded-block-risk-markov-v1",
        "role": role,
        "times": [2, 3, 4, 5],
        "score": "mean_t min((q-1)^2,1); assigned-arm solver failure scores 1",
        "contrast": None if route == "trainable_w" else "X_comparator - (10/9)*X_trainable_w",
        "r_star": _RATIO,
        "support": support,
    })


def _valid_hash(value: object) -> bool:
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _finite(value: object) -> bool:
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _abstain(run_state: object, stage: str, reasons: list[str], status: str):
    if (not hasattr(run_state, "status") or stage not in run_state.status
            or not hasattr(run_state, "summary") or not hasattr(run_state, "abstain")):
        raise ValueError("run_state must be a core.RunState-compatible object")
    if run_state.status[stage] != "not_run":
        return {
            "status": "ABSTAIN_RUNSTATE_STAGE_ALREADY_USED",
            "reasons": [f"{stage} is already {run_state.status[stage]}"],
            "run_state": run_state.summary(),
            "certified": False,
            "physical_score": None,
        }
    run_state.abstain(stage, "; ".join(reasons) or status)
    return {
        "status": status,
        "reasons": reasons,
        "run_state": run_state.summary(),
        "certified": False,
        "physical_score": None,
    }


def _source_map(value: object, required: frozenset[str], label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} source hashes must be a mapping"]
    reasons = []
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required)
    if missing:
        reasons.append(f"{label} source hashes missing {missing}")
    if extra:
        reasons.append(f"{label} source hashes contain unexpected keys {extra}")
    for key, digest in value.items():
        if not _valid_hash(digest):
            reasons.append(f"{label} source hash {key!r} is not lowercase SHA-256")
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
            "expected_sha256": digest if _valid_hash(digest) else None,
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


def _snapshot_source_blocker(run_state: object, stage: str, hashes: object,
                             snapshot: SourceByteSnapshot | None,
                             status: str) -> dict[str, Any] | None:
    if snapshot is None or not isinstance(hashes, dict):
        return None
    try:
        verification = verify_source_snapshot(snapshot, hashes)
    except (TypeError, ValueError):
        return None
    if verification["status"] == "verified":
        return None
    result = _abstain(
        run_state, stage,
        ["source snapshot does not contain all matching task evidence bytes"], status,
    )
    result["source_verification"] = verification
    return result


def _check_receipt(receipt: object, expected: dict[str, int] | None = None,
                   *, dynamic: frozenset[str] = frozenset()) -> tuple[dict[str, Any] | None, list[str]]:
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
    if expected is not None and set(denominators) != set(expected) | set(dynamic):
        reasons.append("receipt denominator names do not exactly match the task plan")
    for name, count in (expected or {}).items():
        item = denominators.get(name)
        if item is not None and item["planned"] != count:
            reasons.append(f"{name}: planned {item['planned']}, expected {count}")
    for name in dynamic:
        item = denominators.get(name)
        if item is not None and item["planned"] < 1:
            reasons.append(f"{name}: at least one planned unit is required")
    for name, item in denominators.items():
        if item["not_run"]:
            reasons.append(f"{name}: {item['not_run']} units remain not_run")
    return normalized, reasons


def _link_sources(envelope: object, hashes: object, *, manifest_key: str,
                  map_key: str, summary_key: str) -> list[str]:
    if not isinstance(envelope, dict) or not isinstance(hashes, dict):
        return ["envelope and route source hashes must be mappings"]
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


def _scenario_name(receipt: dict[str, Any] | None, name: str) -> list[str]:
    if receipt is None or not isinstance(receipt.get("scenario"), dict):
        return ["receipt scenario is missing"]
    if receipt["scenario"].get("name") != name:
        return [f"receipt scenario name must be {name!r}"]
    return []


def _closed_split(envelope: object, *, require_calibration: bool = False) -> list[str]:
    if not isinstance(envelope, dict) or not isinstance(envelope.get("chain"), dict):
        return ["chain metadata is missing"]
    split_id = str(envelope["chain"].get("split_id", "")).lower()
    forbidden = ("confirmation", "validation", "shared-holdout", "shared_holdout", "sealed", "gap")
    if any(token in split_id for token in forbidden):
        return ["sealed, gap, validation, shared-holdout, and confirmation splits are closed"]
    if require_calibration and "calibration" not in split_id:
        return ["D06 production evidence must use an explicitly named calibration split"]
    return []


def apply_optimize_d04_reference_gate(
    run_state: object,
    envelope: object,
    receipt: object,
    reference_summary: object,
    *,
    required_source_hashes: object,
    source_root: str | Path | None = None,
    source_paths: dict[str, str] | None = None,
    source_snapshot: SourceByteSnapshot | None = None,
    require_source_bytes: bool = True,
) -> dict[str, Any]:
    """Check the selected independent E-chain D04 fit and advance to draft.

    Six planned joint fits cover three ``tmin`` checks crossed with two/three
    states. When a source snapshot is supplied, ``reference_summary`` is
    parsed from its captured ``reference_summary`` bytes; a conflicting caller
    object or unreadable source abstains. Any fit failure or failed frozen
    reference criterion abstains.
    """
    stage = "reference"
    if not hasattr(run_state, "status") or stage not in run_state.status:
        raise ValueError("run_state must provide the reference stage")
    hashes = required_source_hashes
    blocked = _snapshot_source_blocker(
        run_state, stage, hashes, source_snapshot, "ABSTAIN_SOURCE_BYTES",
    )
    if blocked is not None:
        return blocked
    summary_read_error = None
    if source_snapshot is not None:
        try:
            captured_summary = read_json_object_snapshot(
                source_snapshot, source_name="reference_summary"
            )
        except ValueError as exc:
            summary_read_error = str(exc)
            reference_summary = None
        else:
            if reference_summary is not None and reference_summary != captured_summary:
                summary_read_error = "caller-provided D04 summary differs from captured summary bytes"
                reference_summary = None
            else:
                reference_summary = captured_summary
    reasons = _source_map(hashes, _D04_SOURCE_KEYS, "D04")
    if summary_read_error is not None:
        reasons.append(summary_read_error)
    normalized, receipt_reasons = _check_receipt(
        receipt, {"reference_fit_variants": _D04_FIT_VARIANTS}
    )
    reasons.extend(receipt_reasons)
    reasons.extend(_scenario_name(normalized, "optimize-d04-reference"))
    denominators = {} if normalized is None else normalized["denominators"]
    fits = denominators.get("reference_fit_variants")
    if fits and (fits["valid"] != _D04_FIT_VARIANTS or fits["failed"]
                 or fits["abstained"]):
        reasons.append("all six planned D04 fit variants must be valid")

    reasons.extend(_check_summary_content(
        reference_summary, fields=_D04_SUMMARY_FIELDS, label="D04",
    ))
    if isinstance(reference_summary, dict):
        numeric = (
            "reference_mass", "reference_mass_error",
            "relative_mass_error_max", "gof_p_min",
            "max_channel_or_window_shift_paired_se", "fit_variants",
            "boundary_hit_count",
        )
        for key in numeric:
            if not _finite(reference_summary.get(key)):
                reasons.append(f"D04 summary {key} must be finite")
        if all(_finite(reference_summary.get(key)) for key in numeric):
            if reference_summary["reference_mass"] <= 0:
                reasons.append("D04 reference mass must be positive")
            if reference_summary["reference_mass_error"] < 0:
                reasons.append("D04 reference mass uncertainty must be nonnegative")
            if reference_summary["relative_mass_error_max"] > 0.05:
                reasons.append("D04 relative mass error exceeds 0.05")
            if not 0 <= reference_summary["gof_p_min"] <= 1:
                reasons.append("D04 goodness-of-fit p value must lie in [0, 1]")
            elif reference_summary["gof_p_min"] < 0.01:
                reasons.append("D04 minimum goodness-of-fit p value is below 0.01")
            if reference_summary["relative_mass_error_max"] < 0:
                reasons.append("D04 relative mass error must be nonnegative")
            if reference_summary["max_channel_or_window_shift_paired_se"] < 0:
                reasons.append("D04 channel/window shift must be nonnegative")
            if (isinstance(reference_summary["fit_variants"], bool)
                    or not isinstance(reference_summary["fit_variants"], int)):
                reasons.append("D04 fit_variants must be an integer denominator")
            if reference_summary["max_channel_or_window_shift_paired_se"] > 2.5:
                reasons.append("D04 channel/window/model shift exceeds 2.5 paired SE")
            if reference_summary["fit_variants"] != _D04_FIT_VARIANTS:
                reasons.append("D04 summary does not cover all six fit variants")
            if (isinstance(reference_summary["boundary_hit_count"], bool)
                    or not isinstance(reference_summary["boundary_hit_count"], int)):
                reasons.append("D04 boundary_hit_count must be an integer count")
            if reference_summary["boundary_hit_count"] != 0:
                reasons.append("D04 fit hit a frozen parameter boundary")
    if isinstance(envelope, dict):
        chain = envelope.get("chain", {})
        observable = envelope.get("observable", {})
        method = envelope.get("method", {})
        if not isinstance(chain, dict) or not str(chain.get("split_id", "")).upper().startswith("E"):
            reasons.append("D04 chain split_id must identify the independent E-chain")
        reasons.extend(_closed_split(envelope))
        if isinstance(observable, dict):
            support = observable.get("support", {})
            if not isinstance(support, dict):
                reasons.append("D04 observable support must be a mapping")
                support = {}
            if support.get("unit") != "a_t m_pi":
                reasons.append("D04 observable support unit must be a_t m_pi")
            if not _finite(support.get("lower")) or support["lower"] < 0:
                reasons.append("D04 reference mass support must be nonnegative")
            if (not _finite(support.get("upper"))
                    or (_finite(support.get("lower")) and support["upper"] <= support["lower"])):
                reasons.append("D04 observable support must be finite and nonempty")
            if (isinstance(reference_summary, dict)
                    and _finite(reference_summary.get("reference_mass"))
                    and _finite(support.get("lower")) and _finite(support.get("upper"))
                    and not support["lower"] <= reference_summary["reference_mass"] <= support["upper"]):
                reasons.append("D04 fitted reference mass lies outside the declared observable support")
        if not isinstance(method, dict) or method.get("name") != "candidate_blind_joint_PP_A4P_two_state_fit":
            reasons.append("D04 method must be the selected candidate-blind joint PP/A4P fit")
        if isinstance(observable, dict):
            if observable.get("map_id") != "optimize-d04-joint-reference-v1":
                reasons.append("D04 observable map_id differs from the selected reference contract")
            if observable.get("map_sha256") != d04_reference_map_sha256():
                reasons.append("D04 observable map hash differs from the selected reference contract")
    reasons.extend(_link_sources(
        envelope, hashes, manifest_key="manifest", map_key="fit_map",
        summary_key="reference_summary",
    ))
    if reasons:
        return _abstain(run_state, stage, reasons, "ABSTAIN_OPTIMIZE_D04_REFERENCE")
    expected = {
        "task": TASK,
        "evidence_kind": "d04_e_chain_reference",
        "observable.map_id": "optimize-d04-joint-reference-v1",
        "observable.map_sha256": d04_reference_map_sha256(),
        "observable.support.unit": "a_t m_pi",
        "method.name": "candidate_blind_joint_PP_A4P_two_state_fit",
    }
    if (require_source_bytes is False and source_root is None and source_paths is None
            and source_snapshot is None):
        return apply_evidence_gate(
            run_state, stage, envelope, receipt, expected=expected,
            required_source_hashes=hashes,
        )
    if source_snapshot is not None:
        if source_root is not None or source_paths is not None:
            result = _abstain(
                run_state, stage,
                ["D04 pass a source snapshot or source_root/source_paths, not both"],
                "ABSTAIN_OPTIMIZE_D04_REFERENCE",
            )
            result["source_verification"] = _missing_source_byte_receipt(hashes)
            return result
        return apply_evidence_gate_with_source_bytes(
            run_state, stage, envelope, receipt,
            source_snapshot=source_snapshot,
            expected=expected, required_source_hashes=hashes,
        )
    if source_root is None or source_paths is None:
        result = _abstain(
            run_state, stage,
            ["D04 source-byte verification requires source_root and an explicit source_paths map"],
            "ABSTAIN_OPTIMIZE_D04_REFERENCE",
        )
        result["source_verification"] = _missing_source_byte_receipt(hashes)
        return result
    return apply_evidence_gate_with_source_bytes(
        run_state, stage, envelope, receipt,
        source_root=str(source_root), source_paths=source_paths,
        expected=expected, required_source_hashes=hashes,
    )


def apply_optimize_d11_operator_gate(
    run_state: object,
    envelope: object,
    receipt: object,
    operator_summary: object,
    *,
    required_source_hashes: object,
    source_root: str | Path | None = None,
    source_paths: dict[str, str] | None = None,
    source_snapshot: SourceByteSnapshot | None = None,
    require_source_bytes: bool = True,
) -> dict[str, Any]:
    """Bind D11 production-link/operator receipts; success remains draft-only.

    When a source snapshot is supplied, ``operator_summary`` is parsed from
    its captured ``operator_summary`` bytes. A conflicting caller object or
    unreadable source abstains.
    """
    stage = "operator"
    if not hasattr(run_state, "status") or stage not in run_state.status:
        raise ValueError("run_state must provide the operator stage")
    hashes = required_source_hashes
    blocked = _snapshot_source_blocker(
        run_state, stage, hashes, source_snapshot, "ABSTAIN_SOURCE_BYTES",
    )
    if blocked is not None:
        return blocked
    summary_read_error = None
    if source_snapshot is not None:
        try:
            captured_summary = read_json_object_snapshot(
                source_snapshot, source_name="operator_summary"
            )
        except ValueError as exc:
            summary_read_error = str(exc)
            operator_summary = None
        else:
            if operator_summary is not None and operator_summary != captured_summary:
                summary_read_error = "caller-provided D11 summary differs from captured summary bytes"
                operator_summary = None
            else:
                operator_summary = captured_summary
    reasons = _source_map(hashes, _D11_SOURCE_KEYS, "D11")
    if summary_read_error is not None:
        reasons.append(summary_read_error)
    normalized, receipt_reasons = _check_receipt(receipt, dynamic=frozenset({"operator_cases"}))
    reasons.extend(receipt_reasons)
    reasons.extend(_scenario_name(normalized, "optimize-d11-production-operator"))
    denominators = {} if normalized is None else normalized["denominators"]
    cases = denominators.get("operator_cases")
    if cases and (cases["valid"] != cases["planned"] or cases["failed"] or cases["abstained"]):
        reasons.append("every planned D11 production operator case must be valid")
    reasons.extend(_check_summary_content(
        operator_summary, fields=_D11_SUMMARY_FIELDS, label="D11",
    ))
    if isinstance(operator_summary, dict):
        flags = (
            "production_links", "su3_provenance_valid", "shape_and_periodic_boundary_valid",
            "unitarity_valid", "finite_values_valid", "bernstein_constraints_valid",
            "spectral_interval_valid", "clenshaw_reference_agrees",
        )
        for key in flags:
            if operator_summary.get(key) is not True:
                reasons.append(f"D11 summary requires {key}=true")
    if isinstance(envelope, dict):
        chain = envelope.get("chain", {})
        observable = envelope.get("observable", {})
        method = envelope.get("method", {})
        if not isinstance(chain, dict) or chain.get("state_unit") != "production gauge-link field":
            reasons.append("D11 chain state_unit must identify a production gauge-link field")
        if isinstance(chain, dict) and any(
            term in str(chain.get("split_id", "")).lower()
            for term in ("synthetic", "fixture", "toy")
        ):
            reasons.append("D11 production operator gate rejects synthetic split IDs")
        if not isinstance(observable, dict):
            reasons.append("D11 observable is missing")
        else:
            support = observable.get("support", {})
            if not isinstance(support, dict):
                reasons.append("D11 observable support must be a mapping")
                support = {}
            if observable.get("map_id") != "optimize-d11-degree4-operator-v1":
                reasons.append("D11 observable map_id differs from the selected contract")
            if observable.get("map_sha256") != d11_operator_map_sha256():
                reasons.append("D11 observable map hash differs from the selected contract")
            if (support.get("lower"), support.get("upper"), support.get("unit")) != (
                -12.0, 0.0, "dimensionless operator spectrum"
            ):
                reasons.append("D11 spectral support must be [-12, 0] in dimensionless operator units")
        if not isinstance(method, dict) or method.get("name") != "degree4_bernstein_chebyshev_clenshaw":
            reasons.append("D11 method must use the selected degree-4 Bernstein/Chebyshev operator")
        dependence = envelope.get("dependence", {})
        if not isinstance(dependence, dict) or dependence.get("mode") != "deterministic":
            reasons.append("D11 operator evidence must declare deterministic algebraic dependence")
    reasons.extend(_link_sources(
        envelope, hashes, manifest_key="link_manifest", map_key="operator_map",
        summary_key="operator_summary",
    ))
    if reasons:
        return _abstain(run_state, stage, reasons, "ABSTAIN_OPTIMIZE_D11_OPERATOR")
    expected = {
        "task": TASK,
        "evidence_kind": "d11_production_operator",
        "observable.map_id": "optimize-d11-degree4-operator-v1",
        "observable.map_sha256": d11_operator_map_sha256(),
        "method.name": "degree4_bernstein_chebyshev_clenshaw",
        "dependence.mode": "deterministic",
    }
    if (require_source_bytes is False and source_root is None and source_paths is None
            and source_snapshot is None):
        return apply_evidence_gate(
            run_state, stage, envelope, receipt, expected=expected,
            required_source_hashes=hashes,
        )
    if source_snapshot is not None:
        if source_root is not None or source_paths is not None:
            result = _abstain(
                run_state, stage,
                ["D11 pass a source snapshot or source_root/source_paths, not both"],
                "ABSTAIN_OPTIMIZE_D11_OPERATOR",
            )
            result["source_verification"] = _missing_source_byte_receipt(hashes)
            return result
        return apply_evidence_gate_with_source_bytes(
            run_state, stage, envelope, receipt,
            source_snapshot=source_snapshot,
            expected=expected, required_source_hashes=hashes,
        )
    if source_root is None or source_paths is None:
        result = _abstain(
            run_state, stage,
            ["D11 source-byte verification requires source_root and an explicit source_paths map"],
            "ABSTAIN_OPTIMIZE_D11_OPERATOR",
        )
        result["source_verification"] = _missing_source_byte_receipt(hashes)
        return result
    return apply_evidence_gate_with_source_bytes(
        run_state, stage, envelope, receipt,
        source_root=str(source_root), source_paths=source_paths,
        expected=expected, required_source_hashes=hashes,
    )


def apply_optimize_bounded_block_risk_gate(
    run_state: object,
    envelopes: object,
    receipt: object,
    *,
    required_source_hashes_by_envelope: object,
    source_root: str | Path | None = None,
    source_paths_by_envelope: dict[str, dict[str, str]] | None = None,
    source_snapshots_by_envelope: dict[str, SourceByteSnapshot] | None = None,
    require_source_bytes: bool = True,
) -> dict[str, Any]:
    """Atomically gate both bounded-risk contrasts and their shared denominator.

    The D04 risk-ratio estimand remains separate and uncalibrated. This adapter
    covers only the accepted clipped bounded-block-risk estimand, with one
    common paired macroblock chain, three simultaneous Markov bounds, and the
    two agreed comparators.
    """
    stage = "statistical"
    if not hasattr(run_state, "status") or stage not in run_state.status:
        raise ValueError("run_state must provide the statistical stage")
    reasons: list[str] = []
    for prerequisite in ("operator", "reference", "numerical"):
        if run_state.status.get(prerequisite) != "validated":
            reasons.append(f"upstream {prerequisite} stage must be validated")
    if not isinstance(envelopes, dict) or set(envelopes) != set(_D06_ROUTES):
        reasons.append("D06 bundle must contain SLSQP, degree-4 Chebyshev, and trainable-W envelopes")
        envelopes = envelopes if isinstance(envelopes, dict) else {}
    source_routes = required_source_hashes_by_envelope
    if not isinstance(source_routes, dict) or set(source_routes) != set(_D06_ROUTES):
        reasons.append("D06 source hashes must contain exactly the three selected routes")
        source_routes = source_routes if isinstance(source_routes, dict) else {}
    for route in _D06_ROUTES:
        reasons.extend(_source_map(source_routes.get(route), _D06_SOURCE_KEYS, f"D06 {route}"))

    normalized, receipt_reasons = _check_receipt(
        receipt,
        {},
        dynamic=frozenset({"paired_macroblocks", "assigned_arm_runs"}),
    )
    # The number of future scored blocks is not frozen by the method contract;
    # require a positive plan and bind exactly three assigned arm-runs per block.
    reasons.extend(receipt_reasons)
    reasons.extend(_scenario_name(normalized, "optimize-d06-bounded-block-risk"))
    denominators = {} if normalized is None else normalized["denominators"]
    macroblocks = denominators.get("paired_macroblocks")
    arm_runs = denominators.get("assigned_arm_runs")
    if macroblocks:
        if macroblocks["planned"] < 2:
            reasons.append("D06 Markov bound requires at least two planned paired macroblocks")
        if (macroblocks["valid"] != macroblocks["planned"]
                or macroblocks["failed"] or macroblocks["abstained"]):
            reasons.append("every D06 paired macroblock must have a complete bounded score")
    if macroblocks and arm_runs and arm_runs["planned"] != 3 * macroblocks["planned"]:
        reasons.append("assigned_arm_runs planned count must equal three arms per paired macroblock")

    expected_by_route: dict[str, dict[str, Any]] = {}
    if set(envelopes) == set(_D06_ROUTES) and set(source_routes) == set(_D06_ROUTES):
        chain_identities = []
        dependence_records = []
        shared_hash_keys = ("manifest", "score_matrix", "block_chain_receipt",
                            "gamma_ps_receipt", "initial_law_tv_receipt", "budget_receipt")
        for route in _D06_ROUTES:
            envelope = envelopes[route]
            hashes = source_routes[route]
            reasons.extend(_link_sources(
                envelope, hashes, manifest_key="manifest", map_key="score_map",
                summary_key="score_matrix",
            ))
            if not isinstance(envelope, dict):
                continue
            chain = envelope.get("chain", {})
            observable = envelope.get("observable", {})
            method = envelope.get("method", {})
            dependence = envelope.get("dependence", {})
            confidence = envelope.get("confidence", {})
            if isinstance(chain, dict):
                chain_identities.append((chain.get("chain_id"), chain.get("manifest_sha256"),
                                         chain.get("split_id"), chain.get("state_unit")))
                reasons.extend(_closed_split(envelope, require_calibration=True))
            if isinstance(dependence, dict):
                dependence_records.append((dependence.get("gamma_ps_lower"),
                                           dependence.get("tv_bound_after_burn_in"),
                                           dependence.get("burn_in")))
            role = "trainable_w_mean" if route == "trainable_w" else f"contrast:{route}"
            lower, upper = ((0.0, 1.0) if route == "trainable_w" else (-_RATIO, 1.0))
            map_id = f"optimize-bounded-block-risk-v1/{role}"
            expected_by_route[route] = {
                "task": TASK,
                "evidence_kind": "bounded_block_risk",
                "chain.state_unit": "paired length-5 saved-configuration macroblock",
                "observable.map_id": map_id,
                "observable.map_sha256": bounded_risk_map_sha256(route),
                "observable.support.lower": lower,
                "observable.support.upper": upper,
                "observable.support.unit": "bounded block-risk summand",
                "method.name": "corrected_markov_pseudo_gap_bernstein_support_only",
                "dependence.mode": "markov",
                "confidence.family_id": "optimize-bounded-block-risk-v1/F3",
                "confidence.alpha_total": _D06_ALPHA,
                "confidence.family_size": _D06_EVENTS,
            }
        if chain_identities and (len(set(chain_identities)) != 1):
            reasons.append("all D06 paired-block envelopes must bind the same chain, manifest, split, and state unit")
        if dependence_records and (len(set(json.dumps(item, sort_keys=True) for item in dependence_records)) != 1):
            reasons.append("all D06 envelopes must share one gamma, TV, and burn-in receipt")
        if dependence_records:
            gamma, tv, _burn = dependence_records[0]
            if not _finite(gamma) or not 0 < float(gamma) <= 1:
                reasons.append("D06 requires a positive gamma_ps lower-bound receipt")
            if not _finite(tv) or not 0 <= float(tv) < _D06_ALPHA:
                reasons.append("D06 post-burn-in TV bound must be below alpha_total")
            alpha = ((_D06_ALPHA - float(tv)) / _D06_EVENTS
                     if _finite(tv) and 0 <= float(tv) < _D06_ALPHA else None)
            if alpha is not None:
                for route in _D06_ROUTES:
                    expected_by_route[route]["confidence.alpha_allocated"] = alpha
        if source_routes and all(isinstance(source_routes.get(route), dict)
                                for route in _D06_ROUTES):
            for key in shared_hash_keys:
                values = {source_routes[route].get(key) for route in _D06_ROUTES}
                if len(values) != 1:
                    reasons.append(f"D06 {key} must be shared across the paired bundle")

    if reasons:
        return _abstain(run_state, stage, reasons, "ABSTAIN_OPTIMIZE_D06_BOUNDED_RISK")
    if (require_source_bytes is False and source_root is None
            and source_paths_by_envelope is None and source_snapshots_by_envelope is None):
        return apply_evidence_bundle_gate(
            run_state, stage, envelopes, receipt,
            expected_by_envelope=expected_by_route,
            required_source_hashes_by_envelope=source_routes,
        )
    if source_snapshots_by_envelope is not None:
        if source_root is not None or source_paths_by_envelope is not None:
            result = _abstain(
                run_state, stage,
                ["D06 pass per-envelope snapshots or source_root/source_paths, not both"],
                "ABSTAIN_OPTIMIZE_D06_BOUNDED_RISK",
            )
            result["source_verification"] = _missing_source_byte_receipt(source_routes)
            return result
        return apply_evidence_bundle_gate_with_source_bytes(
            run_state, stage, envelopes, receipt,
            source_snapshots_by_envelope=source_snapshots_by_envelope,
            expected_by_envelope=expected_by_route,
            required_source_hashes_by_envelope=source_routes,
        )
    if source_root is None or source_paths_by_envelope is None:
        result = _abstain(
            run_state, stage,
            ["D06 source-byte verification requires source_root and an explicit per-route source_paths map"],
            "ABSTAIN_OPTIMIZE_D06_BOUNDED_RISK",
        )
        result["source_verification"] = _missing_source_byte_receipt(source_routes)
        return result
    return apply_evidence_bundle_gate_with_source_bytes(
        run_state, stage, envelopes, receipt,
        source_root=str(source_root),
        source_paths_by_envelope=source_paths_by_envelope,
        expected_by_envelope=expected_by_route,
        required_source_hashes_by_envelope=source_routes,
    )
