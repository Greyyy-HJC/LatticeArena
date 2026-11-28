"""Immutable-snapshot reader for paired Optimize bounded-block score arrays.

This CPU reader consumes a precomputed q matrix only. It does not run the
optimizer, generate a reference, or establish macroblock independence. Its
output is diagnostic input for later receipt/reviewer gates and never a
physical certification.
"""
from __future__ import annotations

import json
from hashlib import sha256
from io import BytesIO
from collections.abc import Mapping
from typing import Any

import numpy as np

from core.source_integrity import SourceByteSnapshot
from core.npz_safety import validate_npz_payload
from .bounded_block_risk import ARMS, TIMES, bounded_block_scores
from .bounded_block_risk_markov import bounded_block_risk_markov_decision


SCHEMA = "optimize-bounded-block-score-matrix-v1"
_REQUIRED_ARRAYS = frozenset({
    "schema", "split_id", "block_ids", "arm_order", "times",
    "q_by_block_arm_time", "arm_failure_mask",
})
_CLOSED_SPLIT_MARKERS = (
    "gap", "validation", "holdout", "sealed", "confirmation",
)
_BLOCK_CHAIN_RECEIPT_SCHEMA = "optimize-paired-macroblock-chain-receipt-v1"
_GAP_RECEIPT_SCHEMA = "optimize-markov-gap-receipt-v1"
_TV_RECEIPT_SCHEMA = "optimize-initial-law-tv-receipt-v1"


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def read_json_object_snapshot(snapshot: SourceByteSnapshot, *, source_name: str) -> dict[str, Any]:
    """Parse a strict JSON object from bytes retained by a source snapshot."""
    if not isinstance(snapshot, SourceByteSnapshot):
        raise ValueError("snapshot must be a SourceByteSnapshot")
    if not isinstance(source_name, str) or not source_name.strip():
        raise ValueError("source_name must be a nonempty string")
    try:
        payload = snapshot.read_bytes(source_name)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"JSON snapshot source is unavailable: {source_name}") from exc
    observed = sha256(payload).hexdigest()
    if observed != snapshot.expected_hashes.get(source_name):
        raise ValueError(f"JSON snapshot source hash mismatch: {source_name}")
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"JSON snapshot source is invalid: {source_name}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"JSON snapshot source root must be an object: {source_name}")
    return document


def _abstention(status: str, reason: str, split_id: object, *,
                source_planned: int = 1, source_abstained: int = 1,
                source_failed: int = 0, source_valid: int = 0,
                planned_blocks: int | None = None) -> dict[str, Any]:
    return {
        "schema": "optimize-bounded-block-snapshot-read-v1",
        "status": status,
        "split_id": split_id if isinstance(split_id, str) else None,
        "reason": reason,
        "source_denominator": {
            "unit": "snapshot_source_file", "planned": source_planned,
            "valid": source_valid, "failed": source_failed,
            "abstained": source_abstained,
        },
        "block_denominator": {
            "unit": "paired_macroblock", "planned": planned_blocks,
            "valid": 0, "failed": 0, "abstained": planned_blocks,
        },
        "physical_score": None,
        "certified": False,
    }


def read_bounded_block_risk_snapshot(
    snapshot: SourceByteSnapshot,
    *,
    source_name: str,
    split_id: str,
    gamma_ps_lower: float | None = None,
    tv_bound_after_burn_in: float = 0.0,
    burn_in_blocks: int = 0,
    method: str = "corrected_markov_pseudo_gap_bernstein",
) -> dict[str, Any]:
    """Parse one hash-matched D06 calibration NPZ from captured bytes.

    Expected NPZ keys are exact; IDs must be unique and increasing; arms and
    times must match the selected contract. A block-level optimization failure
    is permitted only through ``arm_failure_mask`` and is scored as loss 1 by
    the selected reducer. Malformed/nonfinite unfailed arms fail their whole
    paired macroblock. The function has no filesystem path fallback.
    """
    if not isinstance(snapshot, SourceByteSnapshot):
        return _abstention("ABSTAIN_INVALID_SNAPSHOT", "snapshot has the wrong type", split_id)
    if not isinstance(source_name, str) or not source_name.strip():
        return _abstention("ABSTAIN_SOURCE_NAME_MISSING", "source_name is required", split_id)
    if not isinstance(split_id, str) or not split_id.strip():
        return _abstention("ABSTAIN_SPLIT_ID_MISSING", "split_id is required", split_id)
    if method != "corrected_markov_pseudo_gap_bernstein":
        return _abstention(
            "ABSTAIN_METHOD_UNSUPPORTED",
            "D06 contract requires corrected Markov pseudo-gap Bernstein",
            split_id,
        )
    lowered = split_id.lower()
    if any(marker in lowered for marker in _CLOSED_SPLIT_MARKERS):
        return _abstention(
            "ABSTAIN_CLOSED_SPLIT",
            "D06 snapshot reader refuses gap, validation, holdout, sealed, and confirmation splits",
            split_id,
        )
    if "calibration" not in lowered:
        return _abstention(
            "ABSTAIN_NOT_CALIBRATION_SPLIT",
            "D06 snapshot reader requires an explicitly named calibration split",
            split_id,
        )

    try:
        payload = snapshot.read_bytes(source_name)
    except (ValueError, TypeError) as exc:
        return _abstention("ABSTAIN_SOURCE_UNAVAILABLE", str(exc), split_id)
    observed_sha256 = sha256(payload).hexdigest()
    if observed_sha256 != snapshot.expected_hashes.get(source_name):
        return _abstention(
            "ABSTAIN_SNAPSHOT_DIGEST_MISMATCH",
            "captured bytes no longer match the snapshot's expected source hash",
            split_id, source_failed=1, source_abstained=0,
        )

    try:
        validate_npz_payload(payload)
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != _REQUIRED_ARRAYS:
                return _abstention(
                    "ABSTAIN_NPZ_SCHEMA_MISMATCH",
                    f"NPZ keys must be exactly {sorted(_REQUIRED_ARRAYS)}",
                    split_id, source_failed=1, source_abstained=0,
                )
            file_schema = str(archive["schema"].item())
            file_split_id = str(archive["split_id"].item())
            block_ids = np.array(archive["block_ids"], copy=True)
            arm_order = np.array(archive["arm_order"], copy=True)
            times = np.array(archive["times"], copy=True)
            q_values = np.array(archive["q_by_block_arm_time"], copy=True)
            failures = np.array(archive["arm_failure_mask"], copy=True)
    except (OSError, ValueError, KeyError, EOFError) as exc:
        return _abstention(
            "ABSTAIN_NPZ_PARSE_FAILED", f"NPZ parse failed: {type(exc).__name__}",
            split_id, source_failed=1, source_abstained=0,
        )

    if file_schema != SCHEMA:
        return _abstention(
            "ABSTAIN_NPZ_SCHEMA_MISMATCH", "NPZ schema does not match the selected reader",
            split_id, source_failed=1, source_abstained=0,
        )
    if file_split_id != split_id:
        return _abstention(
            "ABSTAIN_SPLIT_ID_MISMATCH", "NPZ split_id differs from the caller's frozen split",
            split_id, source_failed=1, source_abstained=0,
        )
    if arm_order.ndim != 1 or arm_order.tolist() != list(ARMS):
        return _abstention(
            "ABSTAIN_ARM_ORDER_MISMATCH", "NPZ arm order differs from the selected paired contract",
            split_id, source_failed=1, source_abstained=0,
        )
    if times.ndim != 1 or times.dtype.kind not in "iu" or times.tolist() != list(TIMES):
        return _abstention(
            "ABSTAIN_TIME_ORDER_MISMATCH", "NPZ times must be the ordered selected points (2,3,4,5)",
            split_id, source_failed=1, source_abstained=0,
        )
    if (block_ids.ndim != 1 or block_ids.dtype.kind not in "iu"
            or block_ids.size < 2 or np.unique(block_ids).size != block_ids.size
            or (block_ids.dtype.kind == "u"
                and np.any(block_ids > np.iinfo(np.int64).max))
            or np.any(np.diff(block_ids.astype(np.int64, copy=False)) <= 0)):
        return _abstention(
            "ABSTAIN_BLOCK_ID_ORDER", "block IDs must be unique and strictly increasing integers",
            split_id, source_failed=1, source_abstained=0,
        )

    n_blocks = int(block_ids.size)
    if (q_values.ndim != 3 or q_values.shape != (n_blocks, len(ARMS), len(TIMES))
            or q_values.dtype.kind not in "iuf"):
        return _abstention(
            "ABSTAIN_Q_SHAPE", "q array must have shape (blocks, 3 arms, 4 ordered times)",
            split_id, source_failed=1, source_abstained=0, planned_blocks=n_blocks,
        )
    if failures.dtype.kind != "b" or failures.shape != (n_blocks, len(ARMS)):
        return _abstention(
            "ABSTAIN_FAILURE_MASK", "arm failure mask must be boolean with shape (blocks, 3 arms)",
            split_id, source_failed=1, source_abstained=0, planned_blocks=n_blocks,
        )

    values = np.asarray(q_values, dtype=np.float64)
    scores = np.full((n_blocks, len(ARMS)), np.nan, dtype=np.float64)
    valid_blocks = np.zeros(n_blocks, dtype=bool)
    for index in range(n_blocks):
        try:
            scores[index] = bounded_block_scores(
                values[index:index + 1], arm_failure_mask=failures[index:index + 1]
            )[0]
            valid_blocks[index] = True
        except (TypeError, ValueError, OverflowError, FloatingPointError):
            continue

    n_valid = int(np.count_nonzero(valid_blocks))
    n_failed = n_blocks - n_valid
    denominator = {
        "unit": "paired_macroblock", "planned": n_blocks,
        "valid": n_valid, "failed": n_failed, "abstained": 0,
    }
    if n_failed:
        return {
            "schema": "optimize-bounded-block-snapshot-read-v1",
            "status": "ABSTAIN_INVALID_PAIRED_BLOCKS",
            "split_id": split_id,
            "reason": "one or more paired macroblocks contain invalid q values outside declared failed arms",
            "source_sha256": observed_sha256,
            "source_read_mode": "immutable_snapshot",
            "source_denominator": {
                "unit": "snapshot_source_file", "planned": 1,
                "valid": 1, "failed": 0, "abstained": 0,
            },
            "block_denominator": denominator,
            "invalid_block_ids": block_ids[~valid_blocks].astype(int).tolist(),
            "physical_score": None,
            "certified": False,
        }

    try:
        decision = bounded_block_risk_markov_decision(
            scores,
            gamma_ps_lower=gamma_ps_lower,
            tv_bound_after_burn_in=tv_bound_after_burn_in,
            burn_in_blocks=burn_in_blocks,
            arm_failure_mask=failures,
        )
    except (TypeError, ValueError, OverflowError, FloatingPointError) as exc:
        result = _abstention(
            "ABSTAIN_MARKOV_PARAMETERS_INVALID",
            f"Markov decision inputs are invalid: {type(exc).__name__}",
            split_id,
            source_planned=1,
            source_abstained=0,
            source_valid=1,
            planned_blocks=n_blocks,
        )
        result["source_sha256"] = observed_sha256
        result["source_read_mode"] = "immutable_snapshot"
        result["block_denominator"] = {
            "unit": "paired_macroblock", "planned": n_blocks,
            "valid": 0, "failed": 0, "abstained": n_blocks,
        }
        return result
    score_matrix_sha256 = sha256(np.ascontiguousarray(scores).tobytes()).hexdigest()
    failure_counts = {
        arm: int(np.count_nonzero(failures[:, index]))
        for index, arm in enumerate(ARMS)
    }
    failed_arm_runs = sum(failure_counts.values())
    return {
        "schema": "optimize-bounded-block-snapshot-read-v1",
        "status": "ARRAY_REDUCED_NOT_CERTIFIED",
        "split_id": split_id,
        "source_name": source_name,
        "source_sha256": observed_sha256,
        "source_read_mode": "immutable_snapshot",
        "arm_order": list(ARMS),
        "times": list(TIMES),
        "block_ids": block_ids.astype(int).tolist(),
        "block_scores": scores.tolist(),
        "block_scores_sha256": score_matrix_sha256,
        "arm_failure_counts": failure_counts,
        "source_denominator": {
            "unit": "snapshot_source_file", "planned": 1,
            "valid": 1, "failed": 0, "abstained": 0,
        },
        "block_denominator": denominator,
        "assigned_arm_run_denominator": {
            "unit": "assigned_solver_arm_run", "planned": n_blocks * len(ARMS),
            "valid": n_blocks * len(ARMS) - failed_arm_runs,
            "failed": failed_arm_runs, "abstained": 0,
        },
        "decision": decision,
        "physical_score": None,
        "certified": False,
        "text_flow": [
            "explicit calibration split + captured NPZ bytes",
            "schema/order/ID preflight -> q(block, arm, t=2,3,4,5)",
            "declared failed arms -> bounded loss 1; invalid unfailed values -> block failure",
            "paired block score matrix -> corrected Markov pseudo-gap Bernstein reducer -> conditional diagnostic only",
        ],
        "limitations": [
            "does not launch trainable-W, SLSQP, or Chebyshev solvers",
            "does not prove macroblock independence, gamma_ps, burn-in, or TV bounds",
            "does not generate or validate D04 E-chain reference uncertainty",
            "does not produce a physical score or certification",
        ],
    }


def _reader_summary(parsed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for route, item in parsed.items():
        decision = item.get("decision", {})
        bounds = decision.get("bounds") if isinstance(decision, dict) else None
        comparisons = decision.get("comparisons") if isinstance(decision, dict) else None
        summary[route] = {
            "status": item.get("status"),
            "source_read_mode": item.get("source_read_mode"),
            "source_sha256": item.get("source_sha256"),
            "block_scores_sha256": item.get("block_scores_sha256"),
            "block_denominator": item.get("block_denominator"),
            "assigned_arm_run_denominator": item.get("assigned_arm_run_denominator"),
            "decision_status": decision.get("status") if isinstance(decision, dict) else None,
            "dependence_method": decision.get("dependence_method") if isinstance(decision, dict) else None,
            "gamma_ps_lower_assumed": decision.get("gamma_ps_lower_assumed") if isinstance(decision, dict) else None,
            "tv_bound_after_burn_in": decision.get("tv_bound_after_burn_in") if isinstance(decision, dict) else None,
            "burn_in_blocks": decision.get("burn_in_blocks_excluded_from_scored_denominator") if isinstance(decision, dict) else None,
            "trainable_w_lower_bound": (
                bounds.get("trainable_w_mean", {}).get("lower_bound")
                if isinstance(bounds, dict) and isinstance(bounds.get("trainable_w_mean"), dict)
                else None
            ),
            "comparison_lower_bounds": {
                name: row.get("lower_bound")
                for name, row in comparisons.items()
                if isinstance(row, dict)
            } if isinstance(comparisons, dict) else None,
        }
    return summary


def _read_markov_receipts(
    envelopes: dict[str, Any],
    hashes_by_route: dict[str, Any],
    snapshots_by_route: dict[str, Any],
    readers_by_route: dict[str, dict[str, Any]],
    routes: tuple[str, ...],
) -> tuple[dict[str, Any] | None, dict[str, int], list[str]]:
    """Parse the exact chain/gap/TV receipt bytes used for the Markov bound."""
    specs = {
        "block_chain_receipt": (
            _BLOCK_CHAIN_RECEIPT_SCHEMA,
            {
                "schema", "chain_id", "split_id", "state_unit",
                "block_length_configurations", "block_ids", "arm_order", "times",
                "review_receipt_sha256",
            },
        ),
        "gamma_ps_receipt": (
            _GAP_RECEIPT_SCHEMA,
            {"schema", "chain_id", "split_id", "gamma_ps_lower", "review_receipt_sha256"},
        ),
        "initial_law_tv_receipt": (
            _TV_RECEIPT_SCHEMA,
            {"schema", "chain_id", "split_id", "tv_bound_after_burn_in",
             "burn_in_unit", "burn_in_blocks", "review_receipt_sha256"},
        ),
    }
    states: dict[str, str] = {}
    records: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []

    for name, (schema, required_keys) in specs.items():
        payloads: list[bytes] = []
        missing_states: list[str] = []
        for route in routes:
            snapshot = snapshots_by_route.get(route)
            try:
                if not isinstance(snapshot, SourceByteSnapshot):
                    raise ValueError("invalid source snapshot")
                payload = snapshot.read_bytes(name)
            except (TypeError, ValueError):
                entry = (snapshot.capture_entries.get(name)
                         if isinstance(snapshot, SourceByteSnapshot) else None)
                missing_states.append(
                    entry.get("status", "abstained") if isinstance(entry, Mapping) else "abstained"
                )
                continue
            expected_hashes = hashes_by_route.get(route)
            expected_hash = (expected_hashes.get(name)
                             if isinstance(expected_hashes, dict) else None)
            if not isinstance(expected_hash, str) or sha256(payload).hexdigest() != expected_hash:
                missing_states.append("failed")
                continue
            payloads.append(payload)
        if missing_states:
            states[name] = "failed" if "failed" in missing_states else "abstained"
            reasons.append(f"{name}: a required route snapshot is missing or hash-mismatched")
            continue
        if len(payloads) != len(routes) or any(item != payloads[0] for item in payloads[1:]):
            states[name] = "failed"
            reasons.append(f"{name}: paired routes did not capture identical receipt bytes")
            continue
        try:
            record = json.loads(payloads[0])
        except (UnicodeDecodeError, json.JSONDecodeError):
            states[name] = "failed"
            reasons.append(f"{name}: captured receipt is not valid JSON")
            continue
        if (not isinstance(record, dict) or set(record) != required_keys
                or record.get("schema") != schema):
            states[name] = "failed"
            reasons.append(f"{name}: captured receipt schema or exact key set is invalid")
            continue
        states[name] = "valid"
        records[name] = record

    def fail(name: str, message: str) -> None:
        states[name] = "failed"
        reasons.append(f"{name}: {message}")

    if len(records) == len(specs):
        chain_receipt = records["block_chain_receipt"]
        gamma_receipt = records["gamma_ps_receipt"]
        tv_receipt = records["initial_law_tv_receipt"]
        first_envelope = envelopes.get(routes[0])
        first_chain = first_envelope.get("chain") if isinstance(first_envelope, dict) else None
        first_review = first_envelope.get("review") if isinstance(first_envelope, dict) else None
        first_hashes = hashes_by_route.get(routes[0])
        expected_chain_id = first_chain.get("chain_id") if isinstance(first_chain, dict) else None
        expected_split_id = first_chain.get("split_id") if isinstance(first_chain, dict) else None
        expected_state_unit = first_chain.get("state_unit") if isinstance(first_chain, dict) else None
        expected_review_sha = first_review.get("receipt_sha256") if isinstance(first_review, dict) else None
        expected_block_ids = readers_by_route.get(routes[0], {}).get("block_ids")

        expected_declarations = {}
        for route in routes:
            envelope = envelopes.get(route)
            chain = envelope.get("chain") if isinstance(envelope, dict) else None
            dependence = envelope.get("dependence") if isinstance(envelope, dict) else None
            burn = dependence.get("burn_in") if isinstance(dependence, dict) else None
            review = envelope.get("review") if isinstance(envelope, dict) else None
            hashes = hashes_by_route.get(route)
            expected_declarations[route] = {
                "chain_id": chain.get("chain_id") if isinstance(chain, dict) else None,
                "split_id": chain.get("split_id") if isinstance(chain, dict) else None,
                "state_unit": chain.get("state_unit") if isinstance(chain, dict) else None,
                "gamma_ps_lower": dependence.get("gamma_ps_lower") if isinstance(dependence, dict) else None,
                "tv_bound_after_burn_in": dependence.get("tv_bound_after_burn_in") if isinstance(dependence, dict) else None,
                "burn_in_unit": burn.get("unit") if isinstance(burn, dict) else None,
                "burn_in_blocks": burn.get("count") if isinstance(burn, dict) else None,
                "review_receipt_sha256": review.get("receipt_sha256") if isinstance(review, dict) else None,
                "source_review_receipt_sha256": hashes.get("review_receipt") if isinstance(hashes, dict) else None,
            }
        declarations = list(expected_declarations.values())
        if any(item != declarations[0] for item in declarations[1:]):
            fail("block_chain_receipt", "route envelope declarations are not identical")
        declared = declarations[0]

        for name, receipt in (("block_chain_receipt", chain_receipt),
                              ("gamma_ps_receipt", gamma_receipt),
                              ("initial_law_tv_receipt", tv_receipt)):
            if (receipt.get("chain_id"), receipt.get("split_id"),
                    receipt.get("review_receipt_sha256")) != (
                    expected_chain_id, expected_split_id, expected_review_sha):
                fail(name, "chain, split, or review receipt link differs from the envelope")
            if receipt.get("review_receipt_sha256") != declared["source_review_receipt_sha256"]:
                fail(name, "review receipt digest differs from the route source map")

        if chain_receipt.get("state_unit") != expected_state_unit:
            fail("block_chain_receipt", "state unit differs from the envelope")
        if (type(chain_receipt.get("block_length_configurations")) is not int
                or chain_receipt.get("block_length_configurations") != 5):
            fail("block_chain_receipt", "block length must be the selected five configurations")
        if chain_receipt.get("block_ids") != expected_block_ids:
            fail("block_chain_receipt", "captured block IDs differ from the parsed q matrix")
        if chain_receipt.get("arm_order") != list(ARMS) or chain_receipt.get("times") != list(TIMES):
            fail("block_chain_receipt", "arm/time order differs from the fixed D06 map")

        gamma = gamma_receipt.get("gamma_ps_lower")
        if (not isinstance(gamma, (int, float)) or isinstance(gamma, bool)
                or not np.isfinite(gamma) or not 0 < gamma <= 1
                or gamma != declared["gamma_ps_lower"]):
            fail("gamma_ps_receipt", "gamma_ps lower bound is invalid or differs from the envelope")

        tv = tv_receipt.get("tv_bound_after_burn_in")
        burn_count = tv_receipt.get("burn_in_blocks")
        if (not isinstance(tv, (int, float)) or isinstance(tv, bool)
                or not np.isfinite(tv) or not 0 <= tv < 0.05
                or tv != declared["tv_bound_after_burn_in"]):
            fail("initial_law_tv_receipt", "TV bound is invalid or differs from the envelope")
        if (type(burn_count) is not int or burn_count < 0
                or tv_receipt.get("burn_in_unit") != "paired macroblock"
                or burn_count != declared["burn_in_blocks"]
                or tv_receipt.get("burn_in_unit") != declared["burn_in_unit"]):
            fail("initial_law_tv_receipt", "burn-in unit/count differs from the envelope")

    denominator = {
        "unit": "D06_markov_receipt", "planned": len(specs),
        "valid": sum(value == "valid" for value in states.values()),
        "failed": sum(value == "failed" for value in states.values()),
        "abstained": sum(value == "abstained" for value in states.values()),
    }
    params = None
    if not reasons and len(records) == len(specs):
        params = {
            "gamma_ps_lower": records["gamma_ps_receipt"]["gamma_ps_lower"],
            "tv_bound_after_burn_in": records["initial_law_tv_receipt"]["tv_bound_after_burn_in"],
            "burn_in_blocks": records["initial_law_tv_receipt"]["burn_in_blocks"],
            "burn_in_unit": records["initial_law_tv_receipt"]["burn_in_unit"],
        }
    return params, denominator, reasons


def apply_bounded_block_risk_snapshot_gate(
    run_state: object,
    envelopes: object,
    receipt: object,
    *,
    required_source_hashes_by_envelope: object,
    source_snapshots_by_envelope: object,
    method: str = "corrected_markov_pseudo_gap_bernstein",
) -> dict[str, Any]:
    """Read paired q arrays, reconcile denominators, then call the D06 gate.

    All three routes must parse the same shared score-matrix bytes and produce
    identical block IDs/scores. The receipt macroblock and assigned-arm counts
    must equal the parser's observed counts. The downstream adapter still
    checks the independent evidence/source bytes and can advance only to
    ``draft``; it does not certify the bound or its assumptions.
    """
    from .evidence_adapter import (
        _D06_ROUTES, _D06_SOURCE_KEYS, apply_optimize_bounded_block_risk_gate,
    )

    stage = "statistical"
    if (not hasattr(run_state, "status") or stage not in run_state.status
            or not hasattr(run_state, "summary") or not hasattr(run_state, "abstain")):
        raise ValueError("run_state must provide the statistical stage and abstain method")
    if run_state.status[stage] != "not_run":
        return {
            "status": "ABSTAIN_RUNSTATE_STAGE_ALREADY_USED",
            "reasons": [f"{stage} is already {run_state.status[stage]}"],
            "run_state": run_state.summary(), "certified": False,
            "physical_score": None,
        }

    def abstain(reason: str, *, route_denominator: dict[str, Any] | None = None):
        run_state.abstain(stage, reason)
        result = {
            "status": "ABSTAIN_OPTIMIZE_D06_SNAPSHOT_READER",
            "reasons": [reason], "run_state": run_state.summary(),
            "certified": False, "physical_score": None,
        }
        if route_denominator is not None:
            result["reader_route_denominator"] = route_denominator
        return result

    def reader_denominator(parsed: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return {
            name: {
                "planned": 1,
                "valid": int(item.get("status") == "ARRAY_REDUCED_NOT_CERTIFIED"),
                "failed": int(item.get("source_denominator", {}).get("failed", 0) > 0),
                "abstained": int(item.get("source_denominator", {}).get("abstained", 0) > 0),
            }
            for name, item in parsed.items()
        }

    for prerequisite in ("operator", "reference", "numerical"):
        if run_state.status.get(prerequisite) != "validated":
            return abstain(f"upstream {prerequisite} stage must be validated before D06 source reading")
    if (not isinstance(envelopes, dict) or set(envelopes) != set(_D06_ROUTES)
            or not isinstance(source_snapshots_by_envelope, dict)
            or set(source_snapshots_by_envelope) != set(_D06_ROUTES)
            or not isinstance(required_source_hashes_by_envelope, dict)
            or set(required_source_hashes_by_envelope) != set(_D06_ROUTES)):
        return abstain("D06 snapshot reader requires exactly the three selected paired routes")

    expected_shared_hashes = {}
    for route in _D06_ROUTES:
        hashes = required_source_hashes_by_envelope[route]
        if not isinstance(hashes, dict) or set(hashes) != _D06_SOURCE_KEYS:
            return abstain(f"{route}: D06 source-hash map is incomplete or contains unexpected names")
    for name in ("manifest", "score_matrix", "block_chain_receipt",
                 "gamma_ps_receipt", "initial_law_tv_receipt", "budget_receipt"):
        values = {
            required_source_hashes_by_envelope[route].get(name)
            if isinstance(required_source_hashes_by_envelope[route], dict) else None
            for route in _D06_ROUTES
        }
        if len(values) != 1:
            return abstain(f"D06 {name} source hash must be shared across all paired routes")
        expected_shared_hashes[name] = next(iter(values))

    dependence_records = {}
    for route in _D06_ROUTES:
        envelope = envelopes[route]
        dependence = envelope.get("dependence") if isinstance(envelope, dict) else None
        burn_in = dependence.get("burn_in") if isinstance(dependence, dict) else None
        dependence_records[route] = {
            "gamma_ps_lower": dependence.get("gamma_ps_lower") if isinstance(dependence, dict) else None,
            "tv_bound_after_burn_in": (
                dependence.get("tv_bound_after_burn_in") if isinstance(dependence, dict) else None
            ),
            "burn_in_blocks": burn_in.get("count") if isinstance(burn_in, dict) else None,
            "burn_in_unit": burn_in.get("unit") if isinstance(burn_in, dict) else None,
        }
    if len({json.dumps(value, sort_keys=True) for value in dependence_records.values()}) != 1:
        return abstain("D06 envelopes do not provide one shared Markov gap, TV, and burn-in configuration")
    dependence = next(iter(dependence_records.values()))
    if dependence["burn_in_unit"] != "paired macroblock":
        return abstain("D06 source reader requires burn-in to be counted in paired macroblocks")
    if method != "corrected_markov_pseudo_gap_bernstein":
        return abstain("D06 source reader only supports the selected corrected Markov bound")

    parsed: dict[str, dict[str, Any]] = {}
    for route in _D06_ROUTES:
        envelope = envelopes[route]
        chain = envelope.get("chain") if isinstance(envelope, dict) else None
        split_id = chain.get("split_id") if isinstance(chain, dict) else None
        result = read_bounded_block_risk_snapshot(
            source_snapshots_by_envelope[route],
            source_name="score_matrix", split_id=split_id,
            gamma_ps_lower=dependence["gamma_ps_lower"],
            tv_bound_after_burn_in=dependence["tv_bound_after_burn_in"],
            burn_in_blocks=dependence["burn_in_blocks"],
            method=method,
        )
        parsed[route] = result
        if result.get("status") != "ARRAY_REDUCED_NOT_CERTIFIED":
            return abstain(
                f"{route}: snapshot reader returned {result.get('status')}: {result.get('reason', '')}",
                route_denominator=reader_denominator(parsed),
            )
        route_hashes = required_source_hashes_by_envelope[route]
        if route_hashes.get("score_matrix") != result.get("source_sha256"):
            return abstain(f"{route}: parsed score matrix bytes differ from the required source hash")

    summaries = {
        route: (
            parsed[route]["source_sha256"],
            tuple(parsed[route]["block_ids"]),
            parsed[route]["block_scores_sha256"],
            parsed[route]["arm_failure_counts"],
        )
        for route in _D06_ROUTES
    }
    if len({json.dumps(value, sort_keys=True) for value in summaries.values()}) != 1:
        return abstain("D06 routes did not parse identical paired block IDs, scores, and failures")

    receipt_params, markov_receipt_denominator, receipt_reasons = _read_markov_receipts(
        envelopes, required_source_hashes_by_envelope,
        source_snapshots_by_envelope, parsed, _D06_ROUTES,
    )
    if receipt_reasons or receipt_params is None:
        result = abstain(
            "; ".join(receipt_reasons) or "D06 Markov receipts could not be parsed",
            route_denominator=reader_denominator(parsed),
        )
        result["markov_receipt_denominator"] = markov_receipt_denominator
        result["reader_summary"] = _reader_summary(parsed)
        return result

    reference = parsed[_D06_ROUTES[0]]
    n_blocks = reference["block_denominator"]["planned"]
    failed_arms = sum(reference["arm_failure_counts"].values())
    try:
        denominators = receipt["denominators"]
        block_counts = denominators["paired_macroblocks"]
        arm_counts = denominators["assigned_arm_runs"]
        count_match = (
            block_counts["planned"] == n_blocks
            and block_counts["valid"] == n_blocks
            and block_counts["failed"] == 0
            and block_counts["abstained"] == 0
            and block_counts.get("not_run", 0) == 0
            and arm_counts["planned"] == n_blocks * len(ARMS)
            and arm_counts["valid"] == n_blocks * len(ARMS) - failed_arms
            and arm_counts["failed"] == failed_arms
            and arm_counts["abstained"] == 0
            and arm_counts.get("not_run", 0) == 0
        )
    except (AttributeError, TypeError, KeyError):
        count_match = False
    if not count_match:
        return abstain("D06 receipt denominators do not match the parsed paired matrix and arm failures")

    decision = reference["decision"]
    if decision.get("status") != "CONDITIONAL_MARKOV_PASS_BOTH_COMPARISONS":
        result = abstain(
            "D06 bounded-risk reader did not pass both selected effect thresholds and the positive denominator gate",
            route_denominator=reader_denominator(parsed),
        )
        result["reader_decision"] = decision
        result["reader_summary"] = _reader_summary(parsed)
        result["markov_receipt_denominator"] = markov_receipt_denominator
        return result

    result = apply_optimize_bounded_block_risk_gate(
        run_state, envelopes, receipt,
        required_source_hashes_by_envelope=required_source_hashes_by_envelope,
        source_snapshots_by_envelope=source_snapshots_by_envelope,
        require_source_bytes=True,
    )
    result["reader_decision"] = decision
    result["reader_summary"] = _reader_summary(parsed)
    result["markov_receipt_denominator"] = markov_receipt_denominator
    return result
