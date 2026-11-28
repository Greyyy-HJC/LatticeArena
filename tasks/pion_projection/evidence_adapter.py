"""Pion gate for the selected bounded block-risk protocol.

The adapter binds the shared evidence schema to the frozen F=3 score/anchor
defaults and planned denominators. By default it requires either explicit
source-root/path mappings or per-route immutable source snapshots before a
complete bundle can reach ``draft``. A snapshot can also be consumed by the
task reader so the gate rechecks the same retained bytes. This matches supplied
digests only; it does not authenticate digest authority, reviewer identity,
independence, or physical validity, and it does not compute an LCB or certify a
physical score. A passing result is still draft-only.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from core.evidence import (
    apply_evidence_bundle_gate,
    apply_evidence_bundle_gate_with_source_bytes,
    inspect_evidence_envelope,
)
from core.receipts import RECEIPT_SCHEMA, run_receipt as normalize_run_receipt
from core.source_integrity import (
    SourceByteSnapshot,
    capture_source_snapshot,
    verify_source_snapshot,
)
from .bounded_block_risk_contract import selected_protocol


STAGE = "statistical"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCORE_SOURCE_KEYS = frozenset({
    "manifest", "score_map", "block_chain_receipt", "gamma_ps_receipt",
    "initial_law_tv_receipt", "review_receipt",
})
_ANCHOR_SOURCE_KEYS = frozenset({
    "manifest", "scalar_map", "coverage_receipt", "common_bias_receipt",
    "anchor_chain_receipt", "gamma_ps_receipt", "initial_law_tv_receipt",
    "support_receipt", "score_anchor_independence_receipt", "review_receipt",
})
_COMMON_BIAS_RECEIPT_SCHEMA = "pion-anchor-common-bias-receipt-v1"
_COMMON_BIAS_RECEIPT_KEYS = frozenset({
    "schema", "status", "map_id", "map_sha256", "target_reference_sha256",
    "review_scope_sha256", "quantity", "bound_over_R", "support_interval",
})
_COMMON_BIAS_QUANTITY = "abs(E[anchor_scalar_map]-m_star)/R"


def _sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def score_map_sha256() -> str:
    """Hash the selected bounded score definition in canonical JSON form."""
    protocol = selected_protocol()
    score = protocol["score"]
    payload = {
        "protocol_id": protocol["protocol_id"],
        "prediction_domain": score["prediction_domain"],
        "units": score["units"],
        "R": score["R"],
        "formula": score["formula"],
        "clipping_interval": score["clipping_interval"],
        "failure_policy": score["failure_policy"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _expected_fields(anchor_scalar_map_id: str,
                     anchor_scalar_map_sha256: str) -> dict[str, dict[str, Any]]:
    protocol = selected_protocol()
    score = protocol["score"]
    family = protocol["comparison_family"]
    dependence = protocol["dependence_design"]
    anchor = protocol["anchor_design"]
    score_hash = score_map_sha256()
    score_family_id = f"{protocol['protocol_id']}/score-F{family['F']}"
    anchor_family_id = f"{protocol['protocol_id']}/anchor"
    return {
        "score": {
            "task": "pion_projection",
            "evidence_kind": "score_dependence",
            "chain.state_unit": "length-5 saved-configuration macroblock",
            "observable.map_id": f"{protocol['protocol_id']}/bounded-score-map",
            "observable.map_sha256": score_hash,
            "observable.support.lower": -1.0,
            "observable.support.upper": 1.0,
            "observable.support.unit": "bounded score",
            "method.name": dependence["method"],
            "dependence.mode": "markov",
            "confidence.family_id": score_family_id,
            "confidence.alpha_total": family["alpha_total"],
            "confidence.alpha_allocated": family["alpha_score"],
            "confidence.family_size": family["F"],
        },
        "anchor": {
            "task": "pion_projection",
            "evidence_kind": "anchor_coverage",
            "chain.state_unit": "saved gauge-configuration state",
            "observable.map_id": anchor_scalar_map_id,
            "observable.map_sha256": anchor_scalar_map_sha256,
            "observable.support.lower": anchor["support_interval"][0],
            "observable.support.upper": anchor["support_interval"][1],
            "observable.support.unit": score["units"],
            "method.name": anchor["method"],
            "dependence.mode": "markov",
            "confidence.family_id": anchor_family_id,
            "confidence.alpha_total": family["alpha_total"],
            "confidence.alpha_allocated": family["alpha_anchor"],
            "confidence.family_size": 1,
        },
    }


def _validate_denominators(receipt: object) -> list[str]:
    reasons: list[str] = []
    if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
        return [f"receipt.schema must be {RECEIPT_SCHEMA!r}"]
    try:
        normalized = normalize_run_receipt(
            receipt.get("scenario"), receipt.get("denominators"),
            artifact_hashes=receipt.get("artifact_hashes"),
            text_flow=receipt.get("text_flow"),
            reason_rows=receipt.get("reason_rows"),
        )
    except (TypeError, ValueError) as exc:
        return [str(exc)]
    if receipt.get("status") != normalized["status"]:
        reasons.append("receipt.status does not match its denominator counts")
    protocol = selected_protocol()
    family_size = protocol["comparison_family"]["F"]
    planned = {
        "score_family_blocks": (
            protocol["dependence_design"]["planned_scored_blocks"] * family_size
        ),
        "anchor_units": protocol["anchor_design"]["planned_anchor_units"],
    }
    if set(normalized["denominators"]) != set(planned):
        reasons.append(
            "statistical receipt denominators must be exactly score_family_blocks and anchor_units"
        )
    for name, expected_count in planned.items():
        counts = normalized["denominators"].get(name)
        if counts is None:
            continue
        if counts["planned"] != expected_count:
            reasons.append(
                f"{name}: planned {counts['planned']}, expected {expected_count}"
            )
        if counts["not_run"] != 0:
            reasons.append(f"{name}: {counts['not_run']} planned units remain not_run")
        if counts["valid"] != expected_count or counts["failed"] or counts["abstained"]:
            reasons.append(
                f"{name}: expected all {expected_count} units to yield a bounded task value"
            )
    return reasons


def _validate_source_hash_routes(value: object) -> list[str]:
    if not isinstance(value, dict) or set(value) != {"score", "anchor"}:
        return ["required_source_hashes_by_envelope must contain exactly score and anchor routes"]
    reasons: list[str] = []
    for route, required in (("score", _SCORE_SOURCE_KEYS), ("anchor", _ANCHOR_SOURCE_KEYS)):
        hashes = value.get(route)
        if not isinstance(hashes, dict):
            reasons.append(f"{route}: required source hashes must be a mapping")
            continue
        missing = sorted(required - set(hashes))
        if missing:
            reasons.append(f"{route}: missing required source hashes {missing}")
        for name, digest in hashes.items():
            if not _sha256(digest):
                reasons.append(f"{route}: source hash {name!r} is not lowercase SHA-256")
    return reasons


def _validate_source_hash_links(envelopes: object,
                                required_hashes: object) -> list[str]:
    if not isinstance(envelopes, dict) or not isinstance(required_hashes, dict):
        return []
    reasons: list[str] = []
    for route in ("score", "anchor"):
        if route not in required_hashes:
            continue
        result = inspect_evidence_envelope(envelopes.get(route), require_review=True)
        if result["status"] != "SCHEMA_COMPLETE_NOT_CERTIFIED":
            continue
        normalized = result["normalized_envelope"]
        hashes = required_hashes[route]
        required = _SCORE_SOURCE_KEYS if route == "score" else _ANCHOR_SOURCE_KEYS
        if not isinstance(hashes, dict) or not required.issubset(hashes):
            continue
        if hashes["manifest"] != normalized["chain"]["manifest_sha256"]:
            reasons.append(f"{route}: required manifest hash does not bind the chain manifest")
        if hashes["review_receipt"] != normalized["review"]["receipt_sha256"]:
            reasons.append(f"{route}: required review hash does not bind the review receipt")
        map_key = "score_map" if route == "score" else "scalar_map"
        if hashes[map_key] != normalized["observable"]["map_sha256"]:
            reasons.append(f"{route}: required observable-map hash does not bind the envelope map")
        if route == "anchor" and normalized["dependence"]["mode"] != "markov":
            reasons.append("anchor: selected route requires a Markov-chain envelope")
    return reasons


def _snapshot_source_verification(
    snapshots: dict[str, SourceByteSnapshot],
    required_hashes: dict[str, dict[str, str]],
) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    for route in ("score", "anchor"):
        verification = verify_source_snapshot(snapshots[route], required_hashes[route])
        for name, entry in verification["entries"].items():
            entries[f"{route}::{name}"] = dict(entry)
    all_entries = list(entries.values())
    counts = {
        key: sum(entry.get("status") == key for entry in all_entries)
        for key in ("valid", "failed", "abstained")
    }
    return {
        "schema": "latticearena-source-byte-verification-v1",
        "status": "verified" if counts["valid"] == len(all_entries) else "incomplete",
        "read_mode": "immutable_snapshot",
        "denominator": {
            "unit": "source_file", "planned": len(all_entries),
            **counts, "not_run": 0,
        },
        "entries": entries,
    }


def _common_bias_receipt_reasons(
    payload: bytes,
    *,
    envelopes: object,
    anchor_scalar_map_id: str,
    anchor_scalar_map_sha256: str,
    common_bias_bound_over_R: float,
    target_reference_sha256: object,
) -> list[str]:
    """Check that the byte-bound C/R claim matches the current map/reference.

    A successful check verifies internal links only. It does not authenticate
    the reviewer or prove that the claimed common-bias bound is scientifically
    valid; the evidence bundle can still advance only to draft.
    """
    reasons: list[str] = []

    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        receipt = json.loads(
            payload.decode("utf-8"), object_pairs_hook=no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return [f"anchor common-bias receipt is not strict JSON: {exc}"]
    if not isinstance(receipt, dict) or set(receipt) != _COMMON_BIAS_RECEIPT_KEYS:
        return ["anchor common-bias receipt keys do not match the selected schema"]
    if receipt.get("schema") != _COMMON_BIAS_RECEIPT_SCHEMA:
        reasons.append("anchor common-bias receipt schema mismatch")
    if receipt.get("status") != "reviewed_claim":
        reasons.append("anchor common-bias receipt must carry reviewed_claim status")
    if receipt.get("map_id") != anchor_scalar_map_id:
        reasons.append("anchor common-bias receipt map ID mismatch")
    if receipt.get("map_sha256") != anchor_scalar_map_sha256:
        reasons.append("anchor common-bias receipt map hash mismatch")
    if (not _sha256(target_reference_sha256)
            or receipt.get("target_reference_sha256") != target_reference_sha256):
        reasons.append("anchor common-bias receipt does not bind the validated reference stage")
    if receipt.get("quantity") != _COMMON_BIAS_QUANTITY:
        reasons.append("anchor common-bias receipt quantity mismatch")
    support = selected_protocol()["anchor_design"]["support_interval"]
    if receipt.get("support_interval") != support:
        reasons.append("anchor common-bias receipt support interval mismatch")
    bound = receipt.get("bound_over_R")
    if (not isinstance(bound, (int, float)) or isinstance(bound, bool)
            or not math.isfinite(float(bound)) or bound < 0):
        reasons.append("anchor common-bias receipt bound_over_R must be finite and nonnegative")
    elif not math.isclose(float(bound), float(common_bias_bound_over_R),
                          rel_tol=0.0, abs_tol=1e-15):
        reasons.append("caller C/R differs from the captured common-bias receipt")
    anchor_result = inspect_evidence_envelope(
        envelopes.get("anchor") if isinstance(envelopes, dict) else None,
        require_review=True,
    )
    if anchor_result["status"] == "SCHEMA_COMPLETE_NOT_CERTIFIED":
        expected_scope = anchor_result["normalized_envelope"]["review"]["scope_sha256"]
        if receipt.get("review_scope_sha256") != expected_scope:
            reasons.append("anchor common-bias receipt review scope does not match the envelope")
    else:
        reasons.append("anchor envelope is unavailable for common-bias receipt binding")
    return reasons


def _common_bias_receipt_verification(reasons: list[str]) -> dict[str, Any]:
    valid = not reasons
    return {
        "schema": "pion-anchor-common-bias-consistency-v1",
        "status": "internally_consistent_claim" if valid else "rejected",
        "claim_status": "reviewed_claim",
        "authority": "not_authenticated",
        "denominator": {
            "unit": "captured_common_bias_receipt_claim",
            "planned": 1,
            "valid": int(valid),
            "failed": int(not valid),
            "abstained": 0,
            "not_run": 0,
        },
        "reasons": list(reasons),
    }


def _validate_protocol_claims(envelopes: object) -> list[str]:
    if not isinstance(envelopes, dict) or set(envelopes) != {"score", "anchor"}:
        return ["evidence bundle must contain exactly score and anchor envelopes"]
    reasons: list[str] = []
    results = {
        route: inspect_evidence_envelope(envelope, require_review=True)
        for route, envelope in envelopes.items()
    }
    for route, result in results.items():
        if result["status"] != "SCHEMA_COMPLETE_NOT_CERTIFIED":
            reasons.extend(f"{route}: {reason}" for reason in result["reasons"])
    if reasons:
        return reasons

    protocol = selected_protocol()
    score = results["score"]["normalized_envelope"]
    anchor = results["anchor"]["normalized_envelope"]
    design = protocol["dependence_design"]
    score_dependence = score["dependence"]
    if score_dependence["gamma_ps_lower"] < design["gamma_ps_lower_bound_requirement"]:
        reasons.append("score: gamma_ps lower bound is below the selected minimum")
    if score_dependence["burn_in"]["count"] < design["burn_in_blocks"]:
        reasons.append("score: burn-in is shorter than the selected block count")
    if score_dependence["tv_bound_after_burn_in"] > design[
        "post_burn_in_tv_bound_per_comparison"
    ]:
        reasons.append("score: post-burn-in TV bound exceeds the selected maximum")
    anchor_dependence = anchor["dependence"]
    anchor_design = protocol["anchor_design"]
    if anchor_dependence["mode"] != "markov":
        reasons.append("anchor: selected primary route requires a bounded Markov chain")
    else:
        if anchor_dependence["gamma_ps_lower"] < anchor_design[
            "gamma_ps_lower_bound_requirement"
        ]:
            reasons.append("anchor: gamma_ps lower bound is below the selected minimum")
        if anchor_dependence["burn_in"]["count"] < anchor_design["burn_in_units"]:
            reasons.append("anchor: burn-in is shorter than the selected state count")
        if anchor_dependence["tv_bound_after_burn_in"] > anchor_design[
            "post_burn_in_tv_bound"
        ]:
            reasons.append("anchor: post-burn-in TV bound exceeds the selected maximum")
    return reasons


def apply_pion_statistical_evidence_gate(
    run_state: object,
    envelopes: object,
    receipt: object,
    *,
    anchor_scalar_map_id: str,
    anchor_scalar_map_sha256: str,
    common_bias_bound_over_R: float,
    required_source_hashes_by_envelope: object,
    source_root: str | Path | None = None,
    source_paths_by_envelope: object = None,
    source_snapshots_by_envelope: dict[str, SourceByteSnapshot] | None = None,
    require_source_bytes: bool = True,
) -> dict[str, Any]:
    """Apply the Pion statistical metadata gate and optionally verify bytes.

    The anchor map and evidence source hashes must come from externally reviewed
    receipts. When source bytes are required, the common-bias JSON claim is read
    from the same immutable snapshot as the byte gate and checked against the
    map, validated reference-stage hash, review scope, and caller C/R value.
    Those consistency checks do not authenticate hash authority or prove the
    scientific bound. The adapter requires the operator, reference and
    numerical stages to be validated first and accepts only the selected
    12,000 family-block scores plus 4,000 anchor Markov states. A complete
    result advances ``statistical`` to ``draft`` only.
    """
    if (not hasattr(run_state, "status") or not hasattr(run_state, "summary")
            or not hasattr(run_state, "set") or not hasattr(run_state, "abstain")):
        raise ValueError("run_state must provide status, summary, set and abstain")
    if STAGE not in run_state.status:
        raise ValueError(f"unknown RunState stage {STAGE!r}")
    if run_state.status[STAGE] != "not_run":
        return {
            "status": "ABSTAIN_RUNSTATE_STAGE_ALREADY_USED",
            "reasons": [f"{STAGE} is already {run_state.status[STAGE]}"],
            "run_state": run_state.summary(),
            "certified": False,
            "physical_score": None,
        }
    if not isinstance(require_source_bytes, bool):
        reason = "require_source_bytes must be an explicit boolean"
        run_state.abstain(STAGE, reason)
        return {
            "status": "ABSTAIN_PION_STATISTICAL_ADAPTER",
            "reasons": [reason],
            "run_state": run_state.summary(),
            "certified": False,
            "physical_score": None,
        }

    reasons: list[str] = []
    for stage in ("operator", "reference", "numerical"):
        if run_state.status.get(stage) != "validated":
            reasons.append(f"upstream {stage} stage must be validated before statistical evidence")

    if not isinstance(anchor_scalar_map_id, str) or not anchor_scalar_map_id.strip():
        reasons.append("anchor scalar map ID is missing")
    if not _sha256(anchor_scalar_map_sha256):
        reasons.append("anchor scalar map hash is missing or invalid")
    if (not isinstance(common_bias_bound_over_R, (int, float))
            or isinstance(common_bias_bound_over_R, bool)
            or not math.isfinite(float(common_bias_bound_over_R))
            or common_bias_bound_over_R < 0):
        reasons.append("common-bias bound C/R must be a finite nonnegative number")
    elif common_bias_bound_over_R > selected_protocol()["anchor_design"][
        "common_bias_bound_C_max_over_R"
    ]:
        reasons.append("common-bias bound C/R exceeds the selected maximum")

    reasons.extend(_validate_source_hash_routes(required_source_hashes_by_envelope))
    reasons.extend(_validate_source_hash_links(
        envelopes, required_source_hashes_by_envelope
    ))
    reasons.extend(_validate_denominators(receipt))
    reasons.extend(_validate_protocol_claims(envelopes))
    if reasons:
        run_state.abstain(STAGE, "; ".join(reasons))
        return {
            "status": "ABSTAIN_PION_STATISTICAL_ADAPTER",
            "reasons": reasons,
            "run_state": run_state.summary(),
            "certified": False,
            "physical_score": None,
        }

    expected = _expected_fields(anchor_scalar_map_id, anchor_scalar_map_sha256)
    score_hash = score_map_sha256()
    source_hashes = required_source_hashes_by_envelope
    if source_hashes["anchor"].get("scalar_map") != anchor_scalar_map_sha256:
        reasons.append("anchor scalar map hash does not match its required source hash")
    if source_hashes["score"].get("score_map") != score_hash:
        reasons.append("score map hash does not match the selected bounded score definition")
    if reasons:
        run_state.abstain(STAGE, "; ".join(reasons))
        return {
            "status": "ABSTAIN_PION_STATISTICAL_ADAPTER",
            "reasons": reasons,
            "run_state": run_state.summary(),
            "certified": False,
            "physical_score": None,
        }

    paths_supplied = (source_root is not None or source_paths_by_envelope is not None
                      or source_snapshots_by_envelope is not None)
    if require_source_bytes or paths_supplied:
        if (source_snapshots_by_envelope is not None
                and (source_root is not None or source_paths_by_envelope is not None)):
            reason = "pass per-envelope source snapshots or source_root/source_paths, not both"
            run_state.abstain(STAGE, reason)
            return {
                "status": "ABSTAIN_PION_STATISTICAL_ADAPTER",
                "reasons": [reason],
                "run_state": run_state.summary(),
                "certified": False,
                "physical_score": None,
            }

        snapshots = source_snapshots_by_envelope
        if snapshots is None and (source_root is not None or source_paths_by_envelope is not None):
            if source_root is None or source_paths_by_envelope is None:
                reason = "source-byte gate requires both source_root and source_paths_by_envelope"
                run_state.abstain(STAGE, reason)
                return {
                    "status": "ABSTAIN_PION_STATISTICAL_ADAPTER",
                    "reasons": [reason],
                    "run_state": run_state.summary(),
                    "certified": False,
                    "physical_score": None,
                }
            try:
                if (not isinstance(source_paths_by_envelope, dict)
                        or set(source_paths_by_envelope) != {"score", "anchor"}):
                    raise ValueError("source path map must contain score and anchor routes")
                snapshots = {
                    route: capture_source_snapshot(
                        source_root, source_hashes[route], source_paths_by_envelope[route],
                    )
                    for route in ("score", "anchor")
                }
            except (OSError, TypeError, ValueError) as exc:
                return apply_evidence_bundle_gate_with_source_bytes(
                    run_state,
                    STAGE,
                    envelopes,
                    receipt,
                    source_root=source_root,
                    source_paths_by_envelope=source_paths_by_envelope,
                    expected_by_envelope=expected,
                    required_source_hashes_by_envelope=source_hashes,
                    distinct_chain_pairs=(("anchor", "score"),),
                )

        if snapshots is None:
            if not require_source_bytes:
                return apply_evidence_bundle_gate(
                    run_state,
                    STAGE,
                    envelopes,
                    receipt,
                    expected_by_envelope=expected,
                    required_source_hashes_by_envelope=source_hashes,
                    distinct_chain_pairs=(("anchor", "score"),),
                )
            reason = (
                "source-byte gate requires both source_root/source_paths_by_envelope "
                "or per-envelope source snapshots"
            )
            run_state.abstain(STAGE, reason)
            return {
                "status": "ABSTAIN_PION_STATISTICAL_ADAPTER",
                "reasons": [reason],
                "run_state": run_state.summary(),
                "certified": False,
                "physical_score": None,
            }

        common_bias_verification = None
        if (isinstance(snapshots, dict) and set(snapshots) == {"score", "anchor"}
                and all(isinstance(snapshots.get(route), SourceByteSnapshot)
                        for route in ("score", "anchor"))):
            anchor_snapshot = snapshots["anchor"]
            source_verification = _snapshot_source_verification(snapshots, source_hashes)
            source_entry = source_verification["entries"].get("anchor::common_bias_receipt")
            if source_entry is not None and source_entry.get("status") == "valid":
                try:
                    bias_bytes = anchor_snapshot.read_bytes("common_bias_receipt")
                except ValueError:
                    bias_bytes = None
                if bias_bytes is not None:
                    target_hash = run_state.evidence_hash.get("reference")
                    bias_reasons = _common_bias_receipt_reasons(
                        bias_bytes,
                        envelopes=envelopes,
                        anchor_scalar_map_id=anchor_scalar_map_id,
                        anchor_scalar_map_sha256=anchor_scalar_map_sha256,
                        common_bias_bound_over_R=float(common_bias_bound_over_R),
                        target_reference_sha256=target_hash,
                    )
                    common_bias_verification = _common_bias_receipt_verification(
                        bias_reasons
                    )
                    if bias_reasons:
                        run_state.abstain(STAGE, "; ".join(bias_reasons))
                        return {
                            "status": "ABSTAIN_PION_COMMON_BIAS_RECEIPT",
                            "reasons": bias_reasons,
                            "run_state": run_state.summary(),
                            "source_verification": source_verification,
                            "common_bias_receipt_verification": common_bias_verification,
                            "certified": False,
                            "physical_score": None,
                        }

        result = apply_evidence_bundle_gate_with_source_bytes(
            run_state,
            STAGE,
            envelopes,
            receipt,
            source_snapshots_by_envelope=snapshots,
            expected_by_envelope=expected,
            required_source_hashes_by_envelope=source_hashes,
            distinct_chain_pairs=(("anchor", "score"),),
        )
        if common_bias_verification is not None:
            result["common_bias_receipt_verification"] = common_bias_verification
        return result

    return apply_evidence_bundle_gate(
        run_state,
        STAGE,
        envelopes,
        receipt,
        expected_by_envelope=expected,
        required_source_hashes_by_envelope=source_hashes,
        distinct_chain_pairs=(("anchor", "score"),),
    )
