"""Snapshot-only readers for Pion anchor scalars and PP mean features.

The default map is diagnostic: each captured anchor-state PP correlator is
folded and mapped to one scalar, with invalid states retained in its planned
denominator. Optional feature-vector mode separately returns finite folded
PP values at t=3,4,5 for the ensemble-mean route. Neither mode proves a bias
bound, Markov assumptions, or independence from the score chain; neither
creates a certified physical score.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import numpy as np

from core.npz_safety import validate_npz_payload
from core.source_integrity import SourceByteSnapshot


MANIFEST_SCHEMA = "pion-anchor-pp-manifest-v1"
CORRELATOR_SCHEMA = "pion-anchor-pp-correlators-v1"
RESULT_SCHEMA = "pion-anchor-scalar-map-read-v1"
FEATURE_VECTOR_RESULT_SCHEMA = "pion-anchor-pp-feature-vector-read-v1"
TEMPORAL_EXTENT = 16
MAP_TIME = 4
SUPPORT = (0.1, 3.0)
MAP_ID = "pion-folded-pp-cosh-effective-mass-t4-v1"
MAP_HASH = sha256(json.dumps({
    "schema": MAP_ID,
    "formula": "acosh((C_fold(3)+C_fold(5))/(2*C_fold(4)))",
    "fold": "0.5*(C(t)+C(T-t))",
    "temporal_extent": TEMPORAL_EXTENT,
    "map_time": MAP_TIME,
    "support": list(SUPPORT),
    "invalid_policy": "ABSTAIN; no clipping or state dropping",
}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
IMPLEMENTATION_SHA256 = sha256(Path(__file__).read_bytes()).hexdigest()
_MANIFEST_KEYS = {
    "schema", "split_id", "chain_id", "state_unit", "geometry",
    "kappa", "temporal_extent", "source_protocol_sha256",
    "state_ids_sha256", "correlator_data_sha256",
}


def _strict_json(snapshot: SourceByteSnapshot, name: str) -> dict[str, Any]:
    payload = snapshot.read_bytes(name)
    if sha256(payload).hexdigest() != snapshot.expected_hashes.get(name):
        raise ValueError(f"{name}: captured bytes do not match expected hash")

    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(payload.decode("utf-8"), object_pairs_hook=no_duplicates,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           ValueError(f"non-finite JSON value {token}")))
    if not isinstance(value, dict):
        raise ValueError(f"{name}: JSON root must be an object")
    return value


def fold_pp_even(correlators: object) -> np.ndarray:
    values = np.asarray(correlators)
    if (values.ndim < 1 or values.shape[-1] != TEMPORAL_EXTENT
            or values.dtype.kind != "f" or not np.all(np.isfinite(values))):
        raise ValueError("anchor PP correlators must be finite real T=16 arrays")
    reflected_pairs = []
    for time in range(1, TEMPORAL_EXTENT // 2):
        left, right = values[..., time], values[..., TEMPORAL_EXTENT - time]
        same_sign = np.signbit(left) == np.signbit(right)
        # Use the difference form for same-sign values and the sum form for
        # opposite-sign values, so neither branch overflows before averaging.
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            same_sign_average = left + 0.5 * (right - left)
            opposite_sign_average = 0.5 * (left + right)
        reflected_pairs.append(np.where(same_sign, same_sign_average,
                                        opposite_sign_average))
    return np.stack(
        [values[..., 0]]
        + reflected_pairs
        + [values[..., TEMPORAL_EXTENT // 2]], axis=-1,
    )


def anchor_scalar(correlator: object) -> float:
    """Apply the fixed t=4 map; out-of-support values abstain without clipping."""
    folded = fold_pp_even(correlator)
    if folded.shape != (TEMPORAL_EXTENT // 2 + 1,):
        raise ValueError("one anchor state must provide exactly one PP time series")
    c3, c4, c5 = (float(folded[t]) for t in (3, 4, 5))
    if not all(np.isfinite(value) for value in (c3, c4, c5)) or c4 <= 0:
        raise ValueError("anchor scalar denominator is non-positive or non-finite")
    scale = max(abs(c3), abs(c4), abs(c5))
    scaled_c3, scaled_c4, scaled_c5 = c3 / scale, c4 / scale, c5 / scale
    scaled_denominator = 2.0 * scaled_c4
    if scaled_denominator <= 0:
        raise ValueError("anchor scalar denominator is numerically unresolved")
    argument = (scaled_c3 + scaled_c5) / scaled_denominator
    if not np.isfinite(argument) or argument < 1.0:
        raise ValueError("anchor scalar cosh argument is below one or non-finite")
    value = float(np.arccosh(argument))
    if not SUPPORT[0] <= value <= SUPPORT[1]:
        raise ValueError("anchor scalar lies outside the frozen support [0.1,3.0]")
    return value


def read_anchor_scalar_snapshot(
    snapshot: SourceByteSnapshot,
    *, manifest_name: str = "anchor_manifest",
    correlator_name: str = "anchor_correlator_data",
    protocol_name: str = "anchor_protocol",
    expected_states: int | None = None,
    include_feature_vector: bool = False,
) -> dict[str, Any]:
    """Read an anchor chain from captured bytes and report every state outcome."""
    if not isinstance(snapshot, SourceByteSnapshot):
        raise ValueError("snapshot must be a SourceByteSnapshot")
    if not isinstance(include_feature_vector, bool):
        raise ValueError("include_feature_vector must be boolean")
    manifest = _strict_json(snapshot, manifest_name)
    if set(manifest) != _MANIFEST_KEYS or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("anchor manifest keys or schema do not match the reader contract")
    split_id = manifest.get("split_id")
    if (not isinstance(split_id, str) or "anchor" not in split_id.lower()
            or any(marker in split_id.lower()
                   for marker in ("gap", "validation", "holdout", "sealed", "confirmation"))):
        raise ValueError("reader accepts only an open, explicitly named anchor split")
    if (manifest.get("geometry") != "S16T16" or manifest.get("kappa") != 0.14
            or manifest.get("temporal_extent") != TEMPORAL_EXTENT
            or manifest.get("state_unit") != "saved gauge configuration"):
        raise ValueError("anchor manifest geometry, kappa, extent or state unit mismatch")
    if not isinstance(manifest.get("chain_id"), str) or not manifest["chain_id"].strip():
        raise ValueError("anchor manifest chain_id is required")
    protocol_hash = manifest.get("source_protocol_sha256")
    if (not isinstance(protocol_hash, str) or len(protocol_hash) != 64
            or any(char not in "0123456789abcdef" for char in protocol_hash)):
        raise ValueError("anchor protocol hash must be lowercase SHA-256")
    protocol = _strict_json(snapshot, protocol_name)
    protocol_payload = snapshot.read_bytes(protocol_name)
    if sha256(protocol_payload).hexdigest() != protocol_hash:
        raise ValueError("anchor source protocol bytes do not match the manifest hash")
    if (protocol.get("geometry") != [16, 16, 16, 16]
            or protocol.get("kappa") != 0.14
            or protocol.get("t_boundary") != -1):
        raise ValueError("anchor source protocol geometry, kappa or temporal boundary mismatch")

    payload = snapshot.read_bytes(correlator_name)
    data_hash = sha256(payload).hexdigest()
    if (data_hash != snapshot.expected_hashes.get(correlator_name)
            or manifest.get("correlator_data_sha256") != data_hash):
        raise ValueError("anchor correlator bytes do not match the manifest/source hashes")
    validate_npz_payload(payload, max_uncompressed_bytes=64 * 1024 * 1024)
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != {"schema", "state_ids", "pp_correlators"}:
            raise ValueError("anchor correlator NPZ keys do not match the reader contract")
        schema = np.asarray(archive["schema"])
        if schema.shape != () or str(schema.item()) != CORRELATOR_SCHEMA:
            raise ValueError("anchor correlator NPZ schema mismatch")
        raw_ids = np.asarray(archive["state_ids"])
        correlators = np.asarray(archive["pp_correlators"])
    if raw_ids.ndim != 1 or raw_ids.dtype.kind not in "iu" or raw_ids.dtype.kind == "b":
        raise ValueError("anchor state IDs must be a one-dimensional integer array")
    state_ids = raw_ids.astype("<i8", copy=False)
    if len(state_ids) < 1 or np.any(np.diff(state_ids) <= 0):
        raise ValueError("anchor state IDs must be nonempty and strictly increasing")
    if expected_states is not None and (
        isinstance(expected_states, bool) or not isinstance(expected_states, int)
        or expected_states <= 0 or len(state_ids) != expected_states
    ):
        raise ValueError("anchor state count differs from the predeclared denominator")
    if correlators.shape != (len(state_ids), TEMPORAL_EXTENT):
        raise ValueError("anchor PP array axes must be (state, t=0..15)")
    if correlators.dtype.kind != "f" or not np.all(np.isfinite(correlators)):
        raise ValueError("anchor PP array must be finite and explicitly real")
    feature_values = None
    if include_feature_vector:
        folded = fold_pp_even(correlators)
        feature_values = folded[:, (3, 4, 5)]
        if not np.all(np.isfinite(feature_values)):
            raise ValueError("folded anchor PP feature matrix must be finite")
    id_hash = sha256(np.ascontiguousarray(state_ids).tobytes()).hexdigest()
    if manifest.get("state_ids_sha256") != id_hash:
        raise ValueError("anchor state ID hash mismatch")

    scalars: list[float | None] = []
    reasons: list[dict[str, Any]] = []
    for index, (state_id, row) in enumerate(zip(state_ids, correlators)):
        try:
            scalars.append(anchor_scalar(row))
        except ValueError as exc:
            scalars.append(None)
            reasons.append({"state_index": index, "state_id": int(state_id),
                            "reason": str(exc)})
    planned = len(state_ids)
    failed = len(reasons)
    valid = planned - failed
    result = {
        "schema": RESULT_SCHEMA,
        "status": "diagnostic_only" if failed == 0 else "ABSTAIN_INVALID_ANCHOR_STATES",
        "certified": False, "physical_score": None,
        "split_id": split_id, "chain_id": manifest["chain_id"],
        "scalar_map": {"map_id": MAP_ID, "map_sha256": MAP_HASH,
                       "implementation_sha256": IMPLEMENTATION_SHA256,
                       "formula": "acosh((C_fold(3)+C_fold(5))/(2*C_fold(4)))",
                       "support": list(SUPPORT), "invalid_policy": "state retained as failed; no clipping or row deletion"},
        "source_hashes": {manifest_name: snapshot.expected_hashes[manifest_name],
                          correlator_name: data_hash,
                          protocol_name: protocol_hash},
        "state_denominator": {"unit": "anchor_chain_state", "planned": planned,
                              "valid": valid, "failed": failed, "abstained": 0},
        "state_ids": state_ids.tolist(), "scalar_values": scalars,
        "failure_reasons": reasons,
        "mean_scalar_if_all_valid": float(np.mean(scalars)) if valid == planned else None,
        "open_gates": [
            "candidate scalar-map bias C/R<=0.01 is not established",
            "Markov gamma_ps, burn-in, post-burn-in TV and conditional independence are not verified",
            "this reader does not produce a Pion evidence envelope or certified score",
        ],
    }
    if feature_values is not None:
        result["schema"] = FEATURE_VECTOR_RESULT_SCHEMA
        result["feature_vector"] = {
            "feature_order": ["C_fold_3", "C_fold_4", "C_fold_5"],
            "matrix": feature_values.tolist(),
            "denominator": {
                "unit": "saved gauge configuration feature row",
                "planned": planned, "valid": planned, "failed": 0,
                "abstained": 0,
            },
            "note": "per-state features for an ensemble-mean vector bound; not per-state spectral masses",
        }
    return result
