"""Same-byte calibration reader for the selected finite-T Creutz C lane.

The reader accepts a caller-captured calibration transaction, recomputes the
fixed R=2,T=5 stout interval hull from its loop matrix, and compares the result
with the colocated summary bytes before passing the same snapshot to the lane
gate. It does not read gauge files or establish interval coverage.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from io import BytesIO
from typing import Any

import numpy as np

from core.source_integrity import SourceByteSnapshot
from core.npz_safety import validate_npz_payload

from .c_lane_open_development_diagnostic import (
    ALPHA_LANE,
    BLOCK_LENGTHS,
    HALFWIDTH_TARGET,
    INVALID_LIMIT,
    N_DRAWS,
    R_VALUES,
    T_VALUES,
    analyze as analyze_creutz_values,
)
from .evidence_adapter import (
    _C_SOURCE_KEYS,
    apply_wilson_creutz_gate,
)
from .lane_contract import selected_protocol


MEASUREMENT_SCHEMA = "wilson-c-calibration-loop-matrix-v1"
MANIFEST_SCHEMA = "wilson-c-calibration-manifest-v1"
SUMMARY_SCHEMA = "wilson-c-creutz-interval-summary-v2"
_MEASUREMENT_KEYS = frozenset({
    "schema", "split_id", "config_ids", "kernel_ids", "r_values", "t_values",
    "manifest_sha256", "loops",
})
_MANIFEST_KEYS = frozenset({
    "schema", "split_id", "config_ids", "geometry", "action",
    "boundary_conditions", "r_values", "t_values", "kernel_ids",
    "primary_kernel_id", "kernel_parameters",
})
_SUMMARY_KEYS = frozenset({
    "schema", "units", "hull_lower", "hull_upper", "halfwidth",
    "invalid_draws", "planned_draws", "resampling_protocol",
})
_NAMES = ("manifest", "loop_measurements", "interval_summary")
_READER_SOURCE_KEYS = frozenset(_NAMES)
_CLOSED_SPLIT_MARKERS = ("gap", "validation", "holdout", "sealed", "confirmation")


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must have a JSON object root")
    return value


def _finite(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and np.isfinite(value))


def _read_bytes(
    snapshot: SourceByteSnapshot,
    required_hashes: Mapping[str, str],
    name: str,
) -> bytes:
    expected = required_hashes.get(name)
    if not isinstance(expected, str) or snapshot.expected_hashes.get(name) != expected:
        raise ValueError(f"{name}: expected source hash differs from the captured snapshot")
    payload = snapshot.read_bytes(name)
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"{name}: captured bytes do not match the required SHA-256")
    return payload


def read_creutz_calibration_snapshot(
    snapshot: SourceByteSnapshot,
    *,
    split_id: str,
    required_source_hashes: object,
) -> dict[str, Any]:
    """Recompute C-lane bootstrap summary from one explicit calibration snapshot."""
    denominator = {"unit": "C calibration input file", "planned": len(_NAMES),
                   "valid": 0, "failed": 0, "abstained": 0}

    def abstain(status: str, reason: str, *, config_count: int | None = None,
                draw_denominator: dict[str, int] | None = None) -> dict[str, Any]:
        result = {
            "schema": "wilson-c-calibration-snapshot-read-v1",
            "status": status, "reason": reason,
            "source_denominator": denominator.copy(),
            "certified": False, "physical_score": None,
        }
        if config_count is not None:
            result["configuration_denominator"] = {
                "unit": "paired saved gauge configuration", "planned": config_count,
                "valid": 0, "failed": 0, "abstained": config_count,
            }
        if draw_denominator is not None:
            result["draw_denominator"] = draw_denominator
        return result

    if not isinstance(snapshot, SourceByteSnapshot):
        denominator["abstained"] = denominator["planned"]
        return abstain("ABSTAIN_C_INVALID_SNAPSHOT", "snapshot has the wrong type")
    if (not isinstance(required_source_hashes, Mapping)
            or set(required_source_hashes) not in (_C_SOURCE_KEYS, _READER_SOURCE_KEYS)):
        denominator["abstained"] = denominator["planned"]
        return abstain("ABSTAIN_C_SOURCE_MAP", "C reader requires its three data hashes; the evidence gate requires the full C source map")
    if not isinstance(split_id, str) or "calibration" not in split_id.lower():
        denominator["abstained"] = denominator["planned"]
        return abstain("ABSTAIN_C_NOT_CALIBRATION", "C reader requires an explicitly named calibration split")
    if any(marker in split_id.lower() for marker in _CLOSED_SPLIT_MARKERS):
        denominator["abstained"] = denominator["planned"]
        return abstain("ABSTAIN_C_CLOSED_SPLIT", "C reader refuses gap, validation, holdout, sealed, and confirmation")

    payloads = {}
    for name in _NAMES:
        try:
            payloads[name] = _read_bytes(snapshot, required_source_hashes, name)
            denominator["valid"] += 1
        except (KeyError, TypeError, ValueError) as exc:
            entry = snapshot.capture_entries.get(name, {})
            if entry.get("status") == "failed" or "hash" in str(exc):
                denominator["failed"] += 1
            else:
                denominator["abstained"] += 1
            return abstain("ABSTAIN_C_SOURCE_BYTES", str(exc))

    try:
        manifest = _strict_json(payloads["manifest"], "C calibration manifest")
        saved_summary = _strict_json(payloads["interval_summary"], "C interval summary")
    except ValueError as exc:
        return abstain("ABSTAIN_C_METADATA_PARSE", str(exc))
    if set(manifest) != _MANIFEST_KEYS or manifest.get("schema") != MANIFEST_SCHEMA:
        return abstain("ABSTAIN_C_MANIFEST_SCHEMA", "C manifest keys/schema differ from the frozen reader")
    if set(saved_summary) != _SUMMARY_KEYS or saved_summary.get("schema") != SUMMARY_SCHEMA:
        return abstain("ABSTAIN_C_SUMMARY_SCHEMA", "C summary keys/schema differ from the frozen reader")
    if manifest.get("split_id") != split_id:
        return abstain("ABSTAIN_C_SPLIT_MISMATCH", "manifest split_id differs from the selected calibration split")

    selected = selected_protocol()
    common = selected["ensemble_design"]
    c_lane = selected["finite_T_Creutz"]
    inference = selected["familywise_inference"]
    seed_base = c_lane["bootstrap_seed_base"]
    contract_lengths = (inference.get("primary_circular_block_length"),
                        *inference.get("sensitivity_block_lengths", []))
    if (inference.get("block_bootstrap_draws") != N_DRAWS
            or tuple(sorted(contract_lengths)) != tuple(sorted(BLOCK_LENGTHS))
            or inference.get("alpha_each_lane") != ALPHA_LANE
            or inference.get("block_bootstrap_invalid_draw_fraction_max") != INVALID_LIMIT
            or c_lane.get("bootstrap_invalid_draw_fraction_max") != INVALID_LIMIT
            or c_lane.get("precision_halfwidth_target") != HALFWIDTH_TARGET):
        return abstain("ABSTAIN_C_RESAMPLING_CONTRACT", "selected C draws/block lengths/alpha/precision differ from the implemented interval procedure")
    if (isinstance(seed_base, bool) or not isinstance(seed_base, int)
            or seed_base < 0):
        return abstain("ABSTAIN_C_RESAMPLING_CONTRACT", "selected C bootstrap seed base must be a nonnegative integer")
    resampling_protocol = {
        "method": "paired circular moving-block bootstrap",
        "block_lengths": list(BLOCK_LENGTHS),
        "draws_per_length": N_DRAWS,
        "alpha_lane": ALPHA_LANE,
        "invalid_fraction_max": INVALID_LIMIT,
        "precision_halfwidth_target": HALFWIDTH_TARGET,
        "seed_base": seed_base,
        "seed_rule": "seed = seed_base + block_length",
        "interval_level": 0.975,
    }
    if (seed_base != inference.get("block_bootstrap_seed_base")
            or c_lane.get("bootstrap_seed_rule") != inference.get("block_bootstrap_seed_rule")
            or c_lane.get("bootstrap_seed_rule") != resampling_protocol["seed_rule"]):
        return abstain("ABSTAIN_C_RESAMPLING_CONTRACT", "selected C seed rule differs from the family inference contract")
    if (manifest.get("geometry") != common["geometry"]
            or manifest.get("action") != "Wilson gauge action, beta=6"
            or manifest.get("boundary_conditions") != common["boundary_conditions"]
            or manifest.get("r_values") != list(R_VALUES)
            or manifest.get("t_values") != list(T_VALUES)
            or manifest.get("primary_kernel_id") != c_lane["kernel_id"]):
        return abstain("ABSTAIN_C_PROTOCOL_MISMATCH", "manifest geometry/action/boundary/axis/kernel differs from the selected C contract")
    kernel_parameters = manifest.get("kernel_parameters")
    if (not isinstance(kernel_parameters, dict)
            or kernel_parameters.get(c_lane["kernel_id"]) != {
                "kind": "stout", "rho": 0.1, "steps": 5,
            }):
        return abstain("ABSTAIN_C_KERNEL_MISMATCH", "selected stout kernel parameters differ from rho=0.1, n=5")

    try:
        validate_npz_payload(payloads["loop_measurements"])
        with np.load(BytesIO(payloads["loop_measurements"]), allow_pickle=False) as archive:
            if set(archive.files) != _MEASUREMENT_KEYS:
                return abstain("ABSTAIN_C_MEASUREMENT_SCHEMA", "measurement NPZ keys differ from the frozen reader")
            measurement_schema = str(archive["schema"].item())
            measurement_split = str(archive["split_id"].item())
            config_ids = np.array(archive["config_ids"], copy=True)
            kernel_ids = np.array(archive["kernel_ids"], copy=True)
            r_values = np.array(archive["r_values"], copy=True)
            t_values = np.array(archive["t_values"], copy=True)
            manifest_sha256 = str(archive["manifest_sha256"].item())
            loops = np.array(archive["loops"], copy=True)
    except (OSError, ValueError, KeyError, EOFError) as exc:
        return abstain("ABSTAIN_C_MEASUREMENT_PARSE", f"measurement NPZ parse failed: {type(exc).__name__}")

    if measurement_schema != MEASUREMENT_SCHEMA or measurement_split != split_id:
        return abstain("ABSTAIN_C_MEASUREMENT_IDENTITY", "measurement schema/split differs from the selected calibration")
    manifest_digest = hashlib.sha256(payloads["manifest"]).hexdigest()
    if manifest_sha256 != manifest_digest:
        return abstain("ABSTAIN_C_MANIFEST_HASH", "measurement NPZ does not bind the captured manifest bytes")
    expected_ids = manifest.get("config_ids")
    if (not isinstance(expected_ids, list) or len(expected_ids) < 2
            or any(type(value) is not int for value in expected_ids)
            or any(right <= left for left, right in zip(expected_ids, expected_ids[1:]))
            or config_ids.ndim != 1 or config_ids.dtype.kind not in "iu"
            or config_ids.tolist() != expected_ids):
        return abstain("ABSTAIN_C_CONFIGURATION_IDS", "configuration IDs must match the increasing manifest list",
                       config_count=len(expected_ids) if isinstance(expected_ids, list) else None)
    n_config = int(config_ids.size)
    if (kernel_ids.ndim != 1 or kernel_ids.dtype.kind not in "US"
            or kernel_ids.tolist() != manifest.get("kernel_ids")
            or r_values.ndim != 1 or r_values.dtype.kind not in "iu"
            or r_values.tolist() != list(R_VALUES)
            or t_values.ndim != 1 or t_values.dtype.kind not in "iu"
            or t_values.tolist() != list(T_VALUES)):
        return abstain("ABSTAIN_C_AXIS_MISMATCH", "kernel/R/T arrays differ from the captured manifest or selected contract",
                       config_count=n_config)
    if (loops.ndim != 4 or loops.shape != (n_config, len(kernel_ids), len(R_VALUES), len(T_VALUES))
            or loops.dtype.kind not in "iuf" or not np.isfinite(loops).all()):
        return abstain("ABSTAIN_C_LOOP_SHAPE_OR_FINITE", "loop array has the wrong shape/type or nonfinite entries",
                       config_count=n_config)
    if manifest.get("kernel_ids") != kernel_ids.tolist():
        return abstain("ABSTAIN_C_KERNEL_ORDER", "measurement kernel order differs from manifest",
                       config_count=n_config)
    kernel_matches = np.flatnonzero(kernel_ids == c_lane["kernel_id"])
    if kernel_matches.size != 1:
        return abstain("ABSTAIN_C_KERNEL_MISSING", "selected stout C kernel must appear exactly once",
                       config_count=n_config)

    try:
        analyzed = analyze_creutz_values(
            np.asarray(loops[:, int(kernel_matches[0]), :, :], dtype=np.float64),
            seed_base=seed_base,
        )
    except (TypeError, ValueError, OverflowError, FloatingPointError) as exc:
        return abstain("ABSTAIN_C_ANALYSIS_FAILED", f"selected Creutz analysis failed: {type(exc).__name__}",
                       config_count=n_config)
    means, point, point_valid, rows, _draws, hull_low, hull_high, hull_halfwidth, empirical_pass = analyzed
    planned_draws = len(rows) * N_DRAWS
    invalid_draws = sum(row["invalid_draws"] for row in rows)
    valid_draws = planned_draws - invalid_draws
    draw_denominator = {
        "unit": "paired circular-bootstrap draw",
        "planned": planned_draws, "valid": valid_draws,
        "failed": invalid_draws, "abstained": 0,
    }
    derived = {
        "units": "dimensionless",
        "hull_lower": hull_low,
        "hull_upper": hull_high,
        "halfwidth": hull_halfwidth,
        "invalid_draws": invalid_draws,
        "planned_draws": planned_draws,
        "resampling_protocol": resampling_protocol,
        "summary_sha256": hashlib.sha256(payloads["interval_summary"]).hexdigest(),
    }
    if not point_valid or hull_low is None or hull_high is None or hull_halfwidth is None:
        return abstain("ABSTAIN_C_NO_VALID_INTERVAL", "one or more required means or all interval endpoints are invalid",
                       config_count=n_config, draw_denominator=draw_denominator)
    for name in ("hull_lower", "hull_upper", "halfwidth"):
        if not _finite(saved_summary.get(name)) or abs(float(saved_summary[name]) - float(derived[name])) > 1e-12:
            return abstain("ABSTAIN_C_SUMMARY_MISMATCH", f"saved interval summary {name} differs from the captured loop analysis",
                           config_count=n_config, draw_denominator=draw_denominator)
    if saved_summary.get("resampling_protocol") != resampling_protocol:
        return abstain("ABSTAIN_C_SUMMARY_RESAMPLING", "saved interval summary does not declare the selected block lengths, draws, seed rule, and level",
                       config_count=n_config, draw_denominator=draw_denominator)
    if (saved_summary.get("units") != "dimensionless"
            or type(saved_summary.get("invalid_draws")) is not int
            or type(saved_summary.get("planned_draws")) is not int
            or saved_summary["invalid_draws"] != invalid_draws
            or saved_summary["planned_draws"] != planned_draws):
        return abstain("ABSTAIN_C_SUMMARY_DENOMINATOR", "saved interval summary units or draw denominator differs from recomputation",
                       config_count=n_config, draw_denominator=draw_denominator)
    if not empirical_pass:
        # Preserve calculated diagnostics while preventing an invalid block-length
        # interval from being promoted by the aggregate C gate's denominator.
        return abstain("ABSTAIN_C_NUMERIC_GATE", "point, per-length invalid-draw, or interval-width gate did not pass",
                       config_count=n_config, draw_denominator=draw_denominator)

    return {
        "schema": "wilson-c-calibration-snapshot-read-v1",
        "status": "ARRAY_REDUCED_NOT_CERTIFIED",
        "split_id": split_id,
        "source_read_mode": "immutable_snapshot",
        "source_hashes": {
            name: hashlib.sha256(payloads[name]).hexdigest() for name in _NAMES
        },
        "point_estimate_dimensionless": point,
        "four_required_loop_means": {
            "W(3,6)": float(means[2, 2]),
            "W(2,5)": float(means[1, 1]),
            "W(3,5)": float(means[2, 1]),
            "W(2,6)": float(means[1, 2]),
        },
        "block_intervals": rows,
        "interval_summary": derived,
        "summary_consistency": "MATCHED_TO_CAPTURED_LOOP_MATRIX",
        "resampling_protocol": {
            "method": "paired circular moving-block bootstrap",
            "same_indices_for_all_four_loop_means": True,
            "block_lengths": list(BLOCK_LENGTHS),
            "draws_per_length": N_DRAWS,
            "seed_base": seed_base,
            "seed_rule": resampling_protocol["seed_rule"],
            "interval_level": 0.975,
        "invalid_fraction_max": INVALID_LIMIT,
        "precision_halfwidth_target": HALFWIDTH_TARGET,
        "invalid_rule": "any required draw mean is nonpositive or nonfinite; retain in planned denominator",
        },
        "source_denominator": denominator.copy(),
        "configuration_denominator": {
            "unit": "paired saved gauge configuration", "planned": n_config,
            "valid": n_config, "failed": 0, "abstained": 0,
        },
        "draw_denominator": draw_denominator,
        "text_flow": [
            "explicit C calibration manifest + loop matrix + interval summary snapshots",
            "split/geometry/action/boundary/kernel/R/T/ID/hash preflight",
            "select stout rho=0.1,n=5 and compute four paired ensemble loop means",
            f"same circular-block indices at L={','.join(map(str, BLOCK_LENGTHS))}; retain invalid draws in all {planned_draws:,} planned slots",
            "recompute equal-tail intervals and hull; compare every summary value and denominator",
            "pass the same snapshot and derived summary to the C evidence gate; draft remains uncertified",
        ],
        "limitations": [
            "does not read raw gauge links or validate the measurement producer",
            "does not establish block-bootstrap coverage or dependence/gap/TV assumptions",
            "does not authenticate coverage, review, authorization, or HMC receipts",
            "is the finite-T Creutz C estimand and does not estimate the F ground-state force",
            "does not produce a physical score or certification",
        ],
        "certified": False,
        "physical_score": None,
    }


def apply_wilson_creutz_snapshot_gate(
    run_state: object,
    envelope: object,
    receipt: object,
    *,
    required_source_hashes: object,
    source_snapshot: SourceByteSnapshot,
) -> dict[str, Any]:
    """Recompute the C interval from captured bytes, then run its evidence gate."""
    split_id = None
    if isinstance(envelope, dict) and isinstance(envelope.get("chain"), dict):
        split_id = envelope["chain"].get("split_id")
    reader = read_creutz_calibration_snapshot(
        source_snapshot, split_id=split_id, required_source_hashes=required_source_hashes,
    )
    if reader.get("status") != "ARRAY_REDUCED_NOT_CERTIFIED":
        result = apply_wilson_creutz_gate(
            run_state, envelope, receipt, None,
            required_source_hashes=required_source_hashes,
            source_snapshot=source_snapshot,
        )
        result["creutz_snapshot_reader"] = reader
        return result
    result = apply_wilson_creutz_gate(
        run_state, envelope, receipt, reader["interval_summary"],
        required_source_hashes=required_source_hashes,
        source_snapshot=source_snapshot,
    )
    result["creutz_snapshot_reader"] = reader
    return result
