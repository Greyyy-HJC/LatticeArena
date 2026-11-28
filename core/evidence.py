"""Typed, fail-closed metadata contracts for scientific evidence receipts.

These helpers verify that a receipt names the expected chain, observable,
support, method, confidence allocation, sources and review artifact. Review
role and independence are recorded as claims; the schema does not authenticate
those claims or certify the underlying physics.
"""
from __future__ import annotations

import math
import re
from numbers import Real
from typing import Any

from .receipts import RECEIPT_SCHEMA, run_receipt as normalize_run_receipt
from .source_integrity import (
    SOURCE_VERIFICATION_SCHEMA,
    SourceByteSnapshot,
    verify_source_bytes,
    verify_source_snapshot,
)


EVIDENCE_SCHEMA = "latticearena-evidence-envelope-v2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DEPENDENCE_MODES = frozenset(("iid", "finite_dependence", "markov", "spectral", "deterministic"))


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return value


def _finite(value: object, label: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{label} must be a finite real number")
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite real number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be a finite real number")
    return converted


def _hashes(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a nonempty mapping of names to SHA-256 hashes")
    result = {}
    for name, digest in value.items():
        result[_text(name, f"{label} name")] = _sha(digest, f"{label}[{name!r}]")
    return result


def evidence_envelope(
    task: str,
    evidence_kind: str,
    *,
    chain: dict[str, Any],
    observable: dict[str, Any],
    method: dict[str, Any],
    dependence: dict[str, Any],
    confidence: dict[str, Any],
    source_hashes: dict[str, str],
    review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a versioned envelope while preserving its conditional status.

    ``chain_id`` identifies the sampling chain and ``manifest_sha256`` binds
    its saved-configuration/state manifest. ``observable.support.unit`` is the
    unit of the quantity carried by this envelope; for example, Wilson F-lane
    contamination can be recorded in lattice-force units after propagation.
    """
    task = _text(task, "task")
    evidence_kind = _text(evidence_kind, "evidence_kind")
    if not isinstance(chain, dict):
        raise ValueError("chain must be a mapping")
    normalized_chain = {
        "chain_id": _text(chain.get("chain_id"), "chain.chain_id"),
        "split_id": _text(chain.get("split_id"), "chain.split_id"),
        "state_unit": _text(chain.get("state_unit"), "chain.state_unit"),
        "manifest_sha256": _sha(chain.get("manifest_sha256"), "chain.manifest_sha256"),
    }

    if not isinstance(observable, dict) or not isinstance(observable.get("support"), dict):
        raise ValueError("observable and observable.support must be mappings")
    support = observable["support"]
    lower = _finite(support.get("lower"), "observable.support.lower")
    upper = _finite(support.get("upper"), "observable.support.upper")
    if upper <= lower:
        raise ValueError("observable.support.upper must exceed lower")
    normalized_observable = {
        "map_id": _text(observable.get("map_id"), "observable.map_id"),
        "map_sha256": _sha(observable.get("map_sha256"), "observable.map_sha256"),
        "support": {
            "lower": lower,
            "upper": upper,
            "unit": _text(support.get("unit"), "observable.support.unit"),
        },
    }

    if not isinstance(method, dict):
        raise ValueError("method must be a mapping")
    assumptions = method.get("assumptions")
    if isinstance(assumptions, str) or not isinstance(assumptions, (list, tuple)):
        raise ValueError("method.assumptions must be a sequence of nonempty strings")
    if not assumptions:
        raise ValueError("method.assumptions must be nonempty")
    normalized_method = {
        "name": _text(method.get("name"), "method.name"),
        "version": _text(method.get("version"), "method.version"),
        "assumptions": [_text(item, "method assumption") for item in assumptions],
    }

    if not isinstance(dependence, dict):
        raise ValueError("dependence must be a mapping")
    mode = dependence.get("mode")
    if mode not in _DEPENDENCE_MODES:
        raise ValueError(f"dependence.mode must be one of {sorted(_DEPENDENCE_MODES)}")
    dependence_assumptions = dependence.get("assumptions")
    if (isinstance(dependence_assumptions, str)
            or not isinstance(dependence_assumptions, (list, tuple))):
        raise ValueError("dependence.assumptions must be a sequence of nonempty strings")
    normalized_dependence: dict[str, Any] = {
        "mode": mode,
        "assumptions": list(dependence_assumptions),
    }
    if not normalized_dependence["assumptions"]:
        raise ValueError("dependence.assumptions must be nonempty")
    for item in normalized_dependence["assumptions"]:
        _text(item, "dependence assumption")
    if mode == "markov":
        gamma = _finite(dependence.get("gamma_ps_lower"), "dependence.gamma_ps_lower")
        tv = _finite(dependence.get("tv_bound_after_burn_in"),
                     "dependence.tv_bound_after_burn_in")
        burn_in = dependence.get("burn_in")
        if not isinstance(burn_in, dict):
            raise ValueError("Markov dependence requires a burn_in mapping")
        burn_count = burn_in.get("count")
        if (not isinstance(burn_count, int) or isinstance(burn_count, bool)
                or burn_count < 0):
            raise ValueError("dependence.burn_in.count must be a nonnegative integer")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("dependence.gamma_ps_lower must be in (0, 1]")
        if not 0.0 <= tv < 1.0:
            raise ValueError("dependence.tv_bound_after_burn_in must be in [0, 1)")
        normalized_dependence.update({
            "gamma_ps_lower": gamma,
            "tv_bound_after_burn_in": tv,
            "burn_in": {
                "unit": _text(burn_in.get("unit"), "dependence.burn_in.unit"),
                "count": burn_count,
            },
        })
    elif mode == "iid":
        normalized_dependence["independence_receipt_sha256"] = _sha(
            dependence.get("independence_receipt_sha256"),
            "dependence.independence_receipt_sha256",
        )
    elif mode == "finite_dependence":
        radius = dependence.get("dependence_radius")
        if not isinstance(radius, int) or isinstance(radius, bool) or radius < 0:
            raise ValueError("finite_dependence requires nonnegative dependence_radius")
        normalized_dependence["dependence_radius"] = radius
        normalized_dependence["graph_sha256"] = _sha(
            dependence.get("graph_sha256"), "dependence.graph_sha256"
        )

    if not isinstance(confidence, dict):
        raise ValueError("confidence must be a mapping")
    alpha_total = _finite(confidence.get("alpha_total"), "confidence.alpha_total")
    alpha_allocated = _finite(confidence.get("alpha_allocated"),
                              "confidence.alpha_allocated")
    family_size = confidence.get("family_size")
    if not 0.0 < alpha_allocated <= alpha_total < 1.0:
        raise ValueError("confidence must satisfy 0 < alpha_allocated <= alpha_total < 1")
    if not isinstance(family_size, int) or isinstance(family_size, bool) or family_size < 1:
        raise ValueError("confidence.family_size must be a positive integer")
    normalized_confidence = {
        "family_id": _text(confidence.get("family_id"), "confidence.family_id"),
        "alpha_total": alpha_total,
        "alpha_allocated": alpha_allocated,
        "family_size": family_size,
    }

    if review is None:
        review = {"status": "pending"}
    if not isinstance(review, dict) or review.get("status") not in ("pending", "reviewed"):
        raise ValueError("review.status must be 'pending' or 'reviewed'")
    normalized_review = {"status": review["status"]}
    if review["status"] == "reviewed":
        normalized_review.update({
            "receipt_sha256": _sha(review.get("receipt_sha256"), "review.receipt_sha256"),
            "scope_sha256": _sha(review.get("scope_sha256"), "review.scope_sha256"),
            "reviewer_role_claim": _text(
                review.get("reviewer_role_claim"), "review.reviewer_role_claim"
            ),
            "independence_claim_sha256": _sha(
                review.get("independence_claim_sha256"),
                "review.independence_claim_sha256",
            ),
        })

    return {
        "schema": EVIDENCE_SCHEMA,
        "task": task,
        "evidence_kind": evidence_kind,
        "chain": normalized_chain,
        "observable": normalized_observable,
        "method": normalized_method,
        "dependence": normalized_dependence,
        "confidence": normalized_confidence,
        "source_hashes": _hashes(source_hashes, "source_hashes"),
        "review": normalized_review,
        "certified": False,
        "physical_score": None,
    }


def inspect_evidence_envelope(
    envelope: object,
    *,
    expected: dict[str, Any] | None = None,
    required_source_hashes: dict[str, str] | None = None,
    require_review: bool = True,
) -> dict[str, Any]:
    """Check schema and caller-frozen identities; mismatches fail closed.

    A successful result is only ``SCHEMA_COMPLETE_NOT_CERTIFIED``. External
    hashes must still be checked against their actual bytes by the receipt
    consumer. A role and independence claim in the review record are metadata,
    not proof of reviewer identity, conflict status or scientific correctness.
    """
    try:
        if not isinstance(envelope, dict) or envelope.get("schema") != EVIDENCE_SCHEMA:
            raise ValueError(f"envelope.schema must be {EVIDENCE_SCHEMA!r}")
        normalized = evidence_envelope(
            envelope.get("task"), envelope.get("evidence_kind"),
            chain=envelope.get("chain"), observable=envelope.get("observable"),
            method=envelope.get("method"), dependence=envelope.get("dependence"),
            confidence=envelope.get("confidence"),
            source_hashes=envelope.get("source_hashes"), review=envelope.get("review"),
        )
    except (TypeError, ValueError) as exc:
        return {
            "status": "ABSTAIN_INVALID_EVIDENCE_ENVELOPE",
            "reasons": [str(exc)],
            "certified": False,
            "physical_score": None,
        }

    if expected is None:
        expected = {}
    if not isinstance(expected, dict):
        raise ValueError("expected must be a mapping of dotted field paths to values")
    reasons = []
    for path, wanted in expected.items():
        actual: Any = normalized
        for component in _text(path, "expected field path").split("."):
            if not isinstance(actual, dict) or component not in actual:
                actual = None
                break
            actual = actual[component]
        if actual != wanted:
            reasons.append(f"field mismatch {path}: expected {wanted!r}, observed {actual!r}")
    if required_source_hashes is not None:
        required = _hashes(required_source_hashes, "required_source_hashes")
        for name, digest in required.items():
            actual = normalized["source_hashes"].get(name)
            if actual != digest:
                reasons.append(f"source hash mismatch {name!r}: expected {digest}, observed {actual!r}")
    if require_review:
        review = normalized["review"]
        if review["status"] != "reviewed":
            reasons.append("independent review receipt is missing")
        elif review["reviewer_role_claim"] != "independent":
            reasons.append("reviewer role claim is not independent")
    return {
        "status": "ABSTAIN_EVIDENCE_MISMATCH" if reasons else "SCHEMA_COMPLETE_NOT_CERTIFIED",
        "reasons": reasons,
        "normalized_envelope": normalized,
        "certified": False,
        "physical_score": None,
    }


def inspect_distinct_chain_identities(first: object, second: object) -> dict[str, Any]:
    """Reject colliding chain IDs/manifests without claiming independence.

    Identity separation is a necessary bookkeeping check, not proof of
    statistical independence or absence of common-mode bias.
    """
    left = inspect_evidence_envelope(first, require_review=False)
    right = inspect_evidence_envelope(second, require_review=False)
    reasons = []
    if left["status"].startswith("ABSTAIN"):
        reasons.append("first envelope is invalid")
    if right["status"].startswith("ABSTAIN"):
        reasons.append("second envelope is invalid")
    if not reasons:
        a, b = left["normalized_envelope"]["chain"], right["normalized_envelope"]["chain"]
        if a["chain_id"] == b["chain_id"]:
            reasons.append("chain_id collision")
        if a["manifest_sha256"] == b["manifest_sha256"]:
            reasons.append("chain manifest collision")
    return {
        "status": "ABSTAIN_CHAIN_IDENTITY_COLLISION" if reasons else "IDENTITIES_DISTINCT_NOT_INDEPENDENCE_PROOF",
        "reasons": reasons,
        "identity_distinct": not reasons,
        "independence_proven": False,
        "physical_score": None,
    }


def apply_evidence_gate(
    run_state: object,
    stage: str,
    envelope: object,
    receipt: object,
    *,
    expected: dict[str, Any] | None = None,
    required_source_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Route envelope/denominator failures to RunState abstention.

    A complete schema plus complete denominator advances only to ``draft``.
    Task-specific evaluators must still verify source bytes, thresholds,
    assumptions and reviewer scope before they can validate a stage.
    """
    if not hasattr(run_state, "status") or not hasattr(run_state, "set") or not hasattr(run_state, "abstain"):
        raise ValueError("run_state must provide status, set and abstain")
    if stage not in run_state.status:
        raise ValueError(f"unknown RunState stage {stage!r}")
    if run_state.status[stage] != "not_run":
        return {
            "status": "ABSTAIN_RUNSTATE_STAGE_ALREADY_USED",
            "reasons": [f"{stage} is already {run_state.status[stage]}"],
            "run_state": run_state.summary(),
            "physical_score": None,
        }

    reasons = []
    envelope_result = inspect_evidence_envelope(
        envelope, expected=expected,
        required_source_hashes=required_source_hashes,
        require_review=True,
    )
    if envelope_result["status"] != "SCHEMA_COMPLETE_NOT_CERTIFIED":
        reasons.extend(envelope_result["reasons"])

    try:
        if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
            raise ValueError(f"receipt.schema must be {RECEIPT_SCHEMA!r}")
        normalized_receipt = normalize_run_receipt(
            receipt.get("scenario"), receipt.get("denominators"),
            artifact_hashes=receipt.get("artifact_hashes"),
            text_flow=receipt.get("text_flow"),
            reason_rows=receipt.get("reason_rows"),
        )
        if receipt.get("status") != normalized_receipt["status"]:
            raise ValueError("receipt.status does not match its denominator counts")
        if not any(item["planned"] > 0 for item in normalized_receipt["denominators"].values()):
            raise ValueError("run receipt has no planned units")
        if normalized_receipt["status"] != "complete":
            reasons.append("run receipt has planned units that are not run")
    except (TypeError, ValueError) as exc:
        reasons.append(str(exc))

    if reasons:
        run_state.abstain(stage, "; ".join(reasons))
        return {
            "status": "ABSTAIN_EVIDENCE_GATE",
            "reasons": reasons,
            "run_state": run_state.summary(),
            "certified": False,
            "physical_score": None,
        }

    run_state.set(
        stage,
        "draft",
        evidence="metadata and denominator contracts are complete; source and scientific validation remain",
    )
    return {
        "status": "DRAFT_PENDING_TASK_VALIDATION",
        "reasons": [],
        "run_state": run_state.summary(),
        "certified": False,
        "physical_score": None,
    }


def apply_evidence_bundle_gate(
    run_state: object,
    stage: str,
    envelopes: dict[str, object],
    receipt: object,
    *,
    expected_by_envelope: dict[str, dict[str, Any]],
    required_source_hashes_by_envelope: dict[str, dict[str, str]] | None = None,
    distinct_chain_pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...] = (),
) -> dict[str, Any]:
    """Atomically check multiple evidence envelopes before one RunState transition.

    This supports task gates that need separate receipts for distinct chains or
    estimands. Different chain identifiers are only an identity check and never
    prove statistical independence. A complete bundle advances only to
    ``draft``; each task evaluator still verifies source bytes and claims.
    """
    if not hasattr(run_state, "status") or not hasattr(run_state, "set") or not hasattr(run_state, "abstain"):
        raise ValueError("run_state must provide status, set and abstain")
    if stage not in run_state.status:
        raise ValueError(f"unknown RunState stage {stage!r}")
    if run_state.status[stage] != "not_run":
        return {
            "status": "ABSTAIN_RUNSTATE_STAGE_ALREADY_USED",
            "reasons": [f"{stage} is already {run_state.status[stage]}"],
            "run_state": run_state.summary(),
            "certified": False,
            "physical_score": None,
        }

    reasons: list[str] = []
    if not isinstance(envelopes, dict) or not envelopes:
        reasons.append("evidence bundle must be a nonempty mapping")
        envelopes = {}
    if not isinstance(expected_by_envelope, dict) or set(expected_by_envelope) != set(envelopes):
        reasons.append("expected_by_envelope keys must exactly match the evidence bundle")
        expected_by_envelope = expected_by_envelope if isinstance(expected_by_envelope, dict) else {}
    if required_source_hashes_by_envelope is not None:
        if (not isinstance(required_source_hashes_by_envelope, dict)
                or not set(required_source_hashes_by_envelope).issubset(envelopes)):
            reasons.append("required source-hash routes must be a mapping over bundle members")
            required_source_hashes_by_envelope = {}
    if not isinstance(distinct_chain_pairs, (list, tuple)):
        reasons.append("distinct_chain_pairs must be a sequence of route-name pairs")
        distinct_chain_pairs = ()

    envelope_results: dict[str, dict[str, Any]] = {}
    normalized: dict[str, dict[str, Any]] = {}
    for route, envelope in envelopes.items():
        try:
            route_name = _text(route, "evidence bundle route")
        except ValueError as exc:
            reasons.append(str(exc))
            continue
        expected = expected_by_envelope.get(route_name)
        if not isinstance(expected, dict):
            reasons.append(f"{route_name}: expected fields must be a mapping")
            continue
        required_hashes = None
        if required_source_hashes_by_envelope is not None:
            if route_name in required_source_hashes_by_envelope:
                required_hashes = required_source_hashes_by_envelope[route_name]
            if (route_name in required_source_hashes_by_envelope
                    and (not isinstance(required_hashes, dict) or not required_hashes)):
                reasons.append(f"{route_name}: required source hashes must be a mapping")
                continue
        result = inspect_evidence_envelope(
            envelope,
            expected=expected,
            required_source_hashes=required_hashes,
            require_review=True,
        )
        envelope_results[route_name] = result
        if result["status"] != "SCHEMA_COMPLETE_NOT_CERTIFIED":
            reasons.extend(f"{route_name}: {reason}" for reason in result["reasons"])
        else:
            normalized[route_name] = result["normalized_envelope"]

    identity_checks = []
    for pair in distinct_chain_pairs:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(not isinstance(name, str) for name in pair)):
            reasons.append("each distinct-chain check must name exactly two bundle routes")
            continue
        left_name, right_name = pair
        if left_name not in normalized or right_name not in normalized:
            reasons.append(f"chain identity check {left_name!r}/{right_name!r} lacks a valid envelope")
            continue
        left = normalized[left_name]["chain"]
        right = normalized[right_name]["chain"]
        collisions = []
        if left["chain_id"] == right["chain_id"]:
            collisions.append("chain_id collision")
        if left["manifest_sha256"] == right["manifest_sha256"]:
            collisions.append("chain manifest collision")
        identity_checks.append({
            "routes": [left_name, right_name],
            "status": "ABSTAIN_CHAIN_IDENTITY_COLLISION" if collisions else "IDENTITIES_DISTINCT_NOT_INDEPENDENCE_PROOF",
            "reasons": collisions,
            "identity_distinct": not collisions,
            "independence_proven": False,
        })
        if collisions:
            reasons.extend(
                f"chain identity {left_name!r}/{right_name!r}: {reason}"
                for reason in collisions
            )

    if reasons:
        run_state.abstain(stage, "; ".join(reasons))
        return {
            "status": "ABSTAIN_EVIDENCE_BUNDLE",
            "reasons": reasons,
            "envelope_results": envelope_results,
            "chain_identity_checks": identity_checks,
            "run_state": run_state.summary(),
            "certified": False,
            "physical_score": None,
        }

    primary_route = next(iter(envelopes))
    gate_result = apply_evidence_gate(
        run_state,
        stage,
        envelopes[primary_route],
        receipt,
        expected=expected_by_envelope[primary_route],
        required_source_hashes=(
            required_source_hashes_by_envelope.get(primary_route)
            if required_source_hashes_by_envelope is not None else None
        ),
    )
    return {
        **gate_result,
        "envelope_results": envelope_results,
        "chain_identity_checks": identity_checks,
        "certified": False,
        "physical_score": None,
    }


def _source_verification_reasons(receipt: object) -> list[str]:
    if not isinstance(receipt, dict) or receipt.get("status") != "verified":
        if isinstance(receipt, dict) and isinstance(receipt.get("entries"), dict):
            reasons = [
                f"{name}: {entry.get('reason') or entry.get('status')}"
                for name, entry in receipt["entries"].items()
                if not isinstance(entry, dict) or entry.get("status") != "valid"
            ]
            if reasons:
                return reasons
        return ["source bytes are not fully verified"]
    return []


def _abstain_source_bytes(
    run_state: object,
    stage: str,
    reasons: list[str],
    verification: dict[str, Any],
) -> dict[str, Any]:
    run_state.abstain(stage, "; ".join(reasons))
    return {
        "status": "ABSTAIN_SOURCE_BYTES",
        "reasons": reasons,
        "source_verification": verification,
        "run_state": run_state.summary(),
        "certified": False,
        "physical_score": None,
    }


def apply_evidence_gate_with_source_bytes(
    run_state: object,
    stage: str,
    envelope: object,
    receipt: object,
    *,
    source_root: str | None = None,
    source_paths: dict[str, str] | None = None,
    source_snapshot: SourceByteSnapshot | None = None,
    expected: dict[str, Any] | None = None,
    required_source_hashes: dict[str, str],
) -> dict[str, Any]:
    """Verify explicit source bytes before applying the metadata-only gate.

    A successful call still advances at most to ``draft``. Callers may pass
    either paths for preflight hashing or an immutable snapshot also consumed
    by their reader. Neither authenticates expected-hash authority or validates
    scientific content.
    """
    if (not hasattr(run_state, "status") or not hasattr(run_state, "set")
            or not hasattr(run_state, "abstain")):
        raise ValueError("run_state must provide status, set and abstain")
    if stage not in run_state.status:
        raise ValueError(f"unknown RunState stage {stage!r}")
    if run_state.status[stage] != "not_run":
        return {
            "status": "ABSTAIN_RUNSTATE_STAGE_ALREADY_USED",
            "reasons": [f"{stage} is already {run_state.status[stage]}"],
            "run_state": run_state.summary(),
            "certified": False,
            "physical_score": None,
        }

    metadata = inspect_evidence_envelope(
        envelope,
        expected=expected,
        required_source_hashes=required_source_hashes,
        require_review=True,
    )
    if metadata["status"] != "SCHEMA_COMPLETE_NOT_CERTIFIED":
        result = apply_evidence_gate(
            run_state,
            stage,
            envelope,
            receipt,
            expected=expected,
            required_source_hashes=required_source_hashes,
        )
        result["source_verification"] = None
        return result

    try:
        if source_snapshot is not None:
            if source_root is not None or source_paths is not None:
                raise ValueError("pass either source_snapshot or source_root/source_paths")
            verification = verify_source_snapshot(source_snapshot, required_source_hashes)
        else:
            verification = verify_source_bytes(
                source_root, required_source_hashes, source_paths,
            )
    except (TypeError, ValueError) as exc:
        verification = {
            "schema": SOURCE_VERIFICATION_SCHEMA,
            "status": "invalid",
            "reasons": [str(exc)],
        }
        return _abstain_source_bytes(
            run_state, stage, [f"source-byte preflight is invalid: {exc}"], verification,
        )

    source_reasons = _source_verification_reasons(verification)
    if source_reasons:
        return _abstain_source_bytes(run_state, stage, source_reasons, verification)

    result = apply_evidence_gate(
        run_state,
        stage,
        envelope,
        receipt,
        expected=expected,
        required_source_hashes=required_source_hashes,
    )
    result["source_verification"] = verification
    result["source_byte_identity_verified"] = True
    return result


def apply_evidence_bundle_gate_with_source_bytes(
    run_state: object,
    stage: str,
    envelopes: dict[str, object],
    receipt: object,
    *,
    source_root: str | None = None,
    source_paths_by_envelope: dict[str, dict[str, str]] | None = None,
    source_snapshots_by_envelope: dict[str, SourceByteSnapshot] | None = None,
    expected_by_envelope: dict[str, dict[str, Any]],
    required_source_hashes_by_envelope: dict[str, dict[str, str]],
    distinct_chain_pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...] = (),
) -> dict[str, Any]:
    """Verify a path map or per-route snapshots before one atomic transition."""
    if (not hasattr(run_state, "status") or not hasattr(run_state, "set")
            or not hasattr(run_state, "abstain")):
        raise ValueError("run_state must provide status, set and abstain")
    if stage not in run_state.status:
        raise ValueError(f"unknown RunState stage {stage!r}")
    if run_state.status[stage] != "not_run":
        return {
            "status": "ABSTAIN_RUNSTATE_STAGE_ALREADY_USED",
            "reasons": [f"{stage} is already {run_state.status[stage]}"],
            "run_state": run_state.summary(),
            "certified": False,
            "physical_score": None,
        }

    routes = set(envelopes) if isinstance(envelopes, dict) else set()
    snapshot_mode = source_snapshots_by_envelope is not None
    if (not routes or not isinstance(expected_by_envelope, dict)
            or set(expected_by_envelope) != routes
            or not isinstance(required_source_hashes_by_envelope, dict)
            or set(required_source_hashes_by_envelope) != routes
            or (snapshot_mode and (
                not isinstance(source_snapshots_by_envelope, dict)
                or set(source_snapshots_by_envelope) != routes
                or source_root is not None or source_paths_by_envelope is not None
            ))
            or (not snapshot_mode and (
                not isinstance(source_paths_by_envelope, dict)
                or not set(source_paths_by_envelope).issubset(routes)
            ))):
        verification = {
            "schema": SOURCE_VERIFICATION_SCHEMA,
            "status": "invalid",
            "reasons": ["bundle route maps must cover the evidence envelopes exactly"],
        }
        return _abstain_source_bytes(
            run_state, stage, verification["reasons"], verification,
        )

    flattened_hashes: dict[str, str] = {}
    flattened_paths: dict[str, str] = {}
    try:
        for route in envelopes:
            route_hashes = required_source_hashes_by_envelope[route]
            route_paths = (source_paths_by_envelope or {}).get(route, {})
            if not isinstance(route_hashes, dict) or not route_hashes:
                raise ValueError(f"{route}: required source hashes must be nonempty")
            if not isinstance(route_paths, dict):
                raise ValueError(f"{route}: source paths must be a mapping")
            unexpected_paths = set(route_paths) - set(route_hashes)
            if unexpected_paths:
                raise ValueError(f"{route}: source paths contain unregistered names")
            for name, digest in route_hashes.items():
                flattened_name = f"{route}::{name}"
                flattened_hashes[flattened_name] = digest
                if name in route_paths:
                    flattened_paths[flattened_name] = route_paths[name]
    except (TypeError, ValueError) as exc:
        verification = {
            "schema": SOURCE_VERIFICATION_SCHEMA,
            "status": "invalid",
            "reasons": [str(exc)],
        }
        return _abstain_source_bytes(
            run_state, stage, [f"source-byte bundle is invalid: {exc}"], verification,
        )

    try:
        if snapshot_mode:
            entries: dict[str, Any] = {}
            for route in envelopes:
                route_verification = verify_source_snapshot(
                    source_snapshots_by_envelope[route],
                    required_source_hashes_by_envelope[route],
                )
                for name, entry in route_verification.get("entries", {}).items():
                    entries[f"{route}::{name}"] = entry
            counts = {
                status: sum(entry["status"] == status for entry in entries.values())
                for status in ("valid", "failed", "abstained")
            }
            planned = len(entries)
            verification = {
                "schema": SOURCE_VERIFICATION_SCHEMA,
                "status": "verified" if counts["valid"] == planned else "incomplete",
                "read_mode": "immutable_snapshot",
                "denominator": {
                    "unit": "source_file", "planned": planned,
                    "valid": counts["valid"], "failed": counts["failed"],
                    "abstained": counts["abstained"], "not_run": 0,
                },
                "entries": entries,
            }
        else:
            verification = verify_source_bytes(
                source_root, flattened_hashes, flattened_paths,
            )
    except (TypeError, ValueError) as exc:
        verification = {
            "schema": SOURCE_VERIFICATION_SCHEMA,
            "status": "invalid",
            "reasons": [str(exc)],
        }
        return _abstain_source_bytes(
            run_state, stage, [f"source-byte preflight is invalid: {exc}"], verification,
        )

    source_reasons = _source_verification_reasons(verification)
    if source_reasons:
        return _abstain_source_bytes(run_state, stage, source_reasons, verification)

    result = apply_evidence_bundle_gate(
        run_state,
        stage,
        envelopes,
        receipt,
        expected_by_envelope=expected_by_envelope,
        required_source_hashes_by_envelope=required_source_hashes_by_envelope,
        distinct_chain_pairs=distinct_chain_pairs,
    )
    result["source_verification"] = verification
    result["source_byte_identity_verified"] = True
    return result
