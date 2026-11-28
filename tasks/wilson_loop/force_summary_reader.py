"""Derive the Wilson F force summary from one immutable source snapshot.

This reader checks the deterministic q/h/x-to-force propagation in the
selected absolute-stout-force contract. It does not fit the force interval,
prove the positive force-scale lower bound, validate GEVP/thermal receipts,
or certify the physical bounds.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.source_integrity import SourceByteSnapshot

from .evidence_adapter import _F_ROUTES, apply_wilson_force_gate
from .force_budget import propagate_absolute_force_budget, propagate_force_budget
from .lane_contract import LATTICE_FORCE_UNITS
from .spectral_q_bound_reader import (
    FORCE_Q_CORNER_LABELS,
    FORCE_Q_KERNEL_ORDER,
    read_spectral_q_bounds,
)


SCHEMA = "wilson-f-force-summary-input-v3"
_REQUIRED = frozenset({
    "schema", "units", "stout_force_estimate", "stout_force_interval_lower",
    "stout_force_interval_upper", "positive_force_scale_lower",
    "statistical_halfwidth", "model_truncation_bound",
    "invalid_force_interval_draws", "invalid_force_scale_draws",
    "force_kernel_order", "force_corner_order", "qhx_corner_bounds",
})


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _finite_number(value: object, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _parse_summary_bytes(payload: bytes, source_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"force summary is not strict UTF-8 JSON: {type(exc).__name__}") from exc
    if not isinstance(raw, dict) or set(raw) != _REQUIRED:
        actual = sorted(raw) if isinstance(raw, dict) else "non-object root"
        raise ValueError(f"force summary schema keys differ: {actual}")
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"force summary schema must be {SCHEMA!r}")
    if raw.get("units") != LATTICE_FORCE_UNITS:
        raise ValueError(f"force summary units must be {LATTICE_FORCE_UNITS!r}")
    expected_kernel_order = list(FORCE_Q_KERNEL_ORDER)
    if raw.get("force_kernel_order") != expected_kernel_order:
        raise ValueError(
            f"force_kernel_order must be exactly {expected_kernel_order!r}"
        )
    expected_corner_order = list(FORCE_Q_CORNER_LABELS)
    if raw.get("force_corner_order") != expected_corner_order:
        raise ValueError(
            f"force_corner_order must be exactly {expected_corner_order!r}"
        )

    estimate_raw = raw["stout_force_estimate"]
    if isinstance(estimate_raw, bool) or not isinstance(estimate_raw, (int, float)):
        raise ValueError("stout_force_estimate must be a finite number")
    try:
        estimate = float(estimate_raw)
    except (OverflowError, ValueError) as exc:
        raise ValueError("stout_force_estimate must be a finite number") from exc
    if not math.isfinite(estimate):
        raise ValueError("stout_force_estimate must be a finite number")
    interval_lower = _finite_number(
        raw["stout_force_interval_lower"], "stout_force_interval_lower"
    )
    interval_upper = _finite_number(
        raw["stout_force_interval_upper"], "stout_force_interval_upper"
    )
    force_scale_lower = _finite_number(
        raw["positive_force_scale_lower"], "positive_force_scale_lower"
    )
    if not interval_lower <= estimate <= interval_upper:
        raise ValueError("stout force interval must contain its point estimate")
    if force_scale_lower > interval_lower:
        raise ValueError(
            "positive_force_scale_lower must not exceed the simultaneous stout interval lower bound"
        )
    statistical = _finite_number(
        raw["statistical_halfwidth"], "statistical_halfwidth", allow_zero=True
    )
    interval_halfwidth = max(estimate - interval_lower, interval_upper - estimate)
    if abs(statistical - interval_halfwidth) > 1e-10 * max(1.0, abs(statistical)):
        raise ValueError(
            "statistical_halfwidth must equal the stout interval's maximum endpoint deviation"
        )
    model = _finite_number(
        raw["model_truncation_bound"], "model_truncation_bound", allow_zero=True
    )
    invalid_counts = {}
    for name in ("invalid_force_interval_draws", "invalid_force_scale_draws"):
        value = raw[name]
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        invalid_counts[name] = value

    qhx = raw["qhx_corner_bounds"]
    if not isinstance(qhx, dict) or set(qhx) != {"q", "h", "x"}:
        raise ValueError("qhx_corner_bounds must contain exactly q, h, and x")
    try:
        primary_qhx = {name: qhx[name][0] for name in ("q", "h", "x")}
        propagated = propagate_absolute_force_budget(
            primary_qhx, force_scale_lower,
            model_truncation_bound=model,
            statistical_halfwidth=statistical,
        )
        comparison = propagate_force_budget(qhx, force_scale_lower)
    except (IndexError, TypeError, ValueError, OverflowError, FloatingPointError) as exc:
        raise ValueError(f"q/h/x corner inputs are invalid: {exc}") from exc
    components = {
        "excited_state_force": propagated["component_force_bounds"]["q"],
        "finite_T_thermal": propagated["component_force_bounds"]["h"],
        "mixed": propagated["component_force_bounds"]["x"],
        "spectral_model_truncation": model,
    }
    systematic = math.fsum(components.values())
    derived = {
        "units": LATTICE_FORCE_UNITS,
        "stout_force_estimate": estimate,
        "stout_force_interval_lower": interval_lower,
        "stout_force_interval_upper": interval_upper,
        "positive_force_scale_lower": force_scale_lower,
        "statistical_halfwidth": statistical,
        "force_kernel_order": expected_kernel_order,
        "force_corner_order": expected_corner_order,
        "qhx_corner_bounds": {
            name: [list(row) for row in qhx[name]] for name in ("q", "h", "x")
        },
        "force_component_bounds": components,
        "comparison_diagnostic": {
            "estimand": "stout_minus_identity_finite_time_contamination_envelope_only",
            "force_corner_count": comparison["force_corner_count"],
            "component_force_bounds": comparison["component_force_bounds"],
            "interpretation": (
                "identity corners are retained for a shared-spectrum consistency diagnostic; "
                "they are excluded from the absolute stout-force primary budget"
            ),
        },
        "systematic_total_bound": systematic,
        "total_linear_bound": systematic + statistical,
        **invalid_counts,
        # This is the digest of the exact captured input bytes. The evidence gate
        # separately checks that every route points to the same digest.
        "summary_sha256": source_sha256,
    }
    diagnostic = {
        "schema": "wilson-f-force-summary-read-v1",
        "status": "DERIVED_FROM_IMMUTABLE_SNAPSHOT",
        "source_read_mode": "immutable_snapshot",
        "source_sha256": source_sha256,
        "analysis_source_sha256": {
            "force_summary_reader.py": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "force_budget.py": hashlib.sha256(
                Path(__file__).with_name("force_budget.py").read_bytes()
            ).hexdigest(),
            "core/numerics.py": hashlib.sha256(
                Path(__file__).resolve().parents[2].joinpath("core/numerics.py").read_bytes()
            ).hexdigest(),
        },
        "stout_force_estimate": estimate,
        "stout_force_interval": [interval_lower, interval_upper],
        "positive_force_scale_lower": force_scale_lower,
        "statistical_halfwidth": statistical,
        "force_kernel_order": expected_kernel_order,
        "force_corner_order": expected_corner_order,
        "primary_force_kernel": expected_kernel_order[0],
        "primary_force_corner_count": 4,
        "q_h_x_force_unit_bounds": components,
        "comparison_diagnostic": derived["comparison_diagnostic"],
        "model_truncation_bound": model,
        "systematic_total_bound": systematic,
        "total_linear_bound": systematic + statistical,
        "force_budget_recalculation": {
            "component_gates_pass": propagated["component_gates_pass"],
            "model_gate_pass": propagated["model_gate_pass"],
            "statistical_gate_pass": propagated["statistical_gate_pass"],
            "total_gate_pass": propagated["total_gate_pass"],
        },
        "text_flow": [
            "six route snapshots -> parse the identical force-summary bytes",
            "validate the positive stout-force interval, positive scale lower bound, invalid-draw counts, explicit kernel/corner order, and 2x4 q/h/x corners",
            "propagate the four stout q/h/x corners into the absolute-force primary budget; retain identity corners only for a comparison diagnostic",
            "pass the derived summary and the same snapshots to the F evidence gate",
            "complete metadata and bytes can reach draft only; physical score remains null",
        ],
        "limitations": [
            "does not derive force intervals or positive force-scale bounds from raw loops",
            "does not validate the GEVP, matched-Nt32 thermal, mixed-sector, or model receipts",
            "schema v3 requires q/h/x rows in stout-then-identity order and columns R2_T5, R2_T6, R3_T5, R3_T6",
            "does not authenticate input authority, reviewer identity, or independence",
            "does not certify the ground-state force or set a physical score",
        ],
    }
    return derived, diagnostic


def read_force_summary_snapshots(
    snapshots_by_envelope: object,
    required_source_hashes_by_envelope: object,
) -> dict[str, Any]:
    """Read and reconcile the same strict force-summary JSON from all six routes."""
    planned = len(_F_ROUTES)
    if (not isinstance(snapshots_by_envelope, Mapping)
            or set(snapshots_by_envelope) != set(_F_ROUTES)
            or not isinstance(required_source_hashes_by_envelope, Mapping)
            or set(required_source_hashes_by_envelope) != set(_F_ROUTES)):
        return {
            "status": "ABSTAIN_F_SUMMARY_ROUTE_MAP",
            "reason": "F summary reader requires exactly the six selected route snapshots and hash maps",
            "denominator": {"unit": "F force-summary route", "planned": planned,
                            "valid": 0, "failed": 0, "abstained": planned},
            "certified": False, "physical_score": None,
        }

    parsed = []
    valid = failed = abstained = 0
    for route in _F_ROUTES:
        snapshot = snapshots_by_envelope[route]
        route_hashes = required_source_hashes_by_envelope[route]
        if not isinstance(snapshot, SourceByteSnapshot) or not isinstance(route_hashes, Mapping):
            abstained += 1
            continue
        expected = route_hashes.get("force_summary")
        if not isinstance(expected, str) or snapshot.expected_hashes.get("force_summary") != expected:
            failed += 1
            continue
        try:
            payload = snapshot.read_bytes("force_summary")
        except (TypeError, ValueError):
            entry = snapshot.capture_entries.get("force_summary", {})
            if entry.get("status") == "failed":
                failed += 1
            else:
                abstained += 1
            continue
        observed = hashlib.sha256(payload).hexdigest()
        if observed != expected:
            failed += 1
            continue
        valid += 1
        parsed.append((payload, observed))

    denominator = {"unit": "F force-summary route", "planned": planned,
                   "valid": valid, "failed": failed, "abstained": abstained}
    if valid != planned or failed or abstained:
        return {
            "status": "ABSTAIN_F_SUMMARY_SOURCE",
            "reason": "one or more route snapshots did not provide the hash-matched summary bytes",
            "denominator": denominator,
            "certified": False, "physical_score": None,
        }
    payloads = {payload for payload, _ in parsed}
    hashes = {digest for _, digest in parsed}
    if len(payloads) != 1 or len(hashes) != 1:
        return {
            "status": "ABSTAIN_F_SUMMARY_ROUTE_MISMATCH",
            "reason": "the six F routes do not capture one identical force-summary byte sequence",
            "denominator": denominator,
            "bundle_denominator": {"unit": "shared force-summary bundle", "planned": 1,
                                   "valid": 0, "failed": 1, "abstained": 0},
            "certified": False, "physical_score": None,
        }
    digest = parsed[0][1]
    try:
        summary, diagnostic = _parse_summary_bytes(parsed[0][0], digest)
    except ValueError as exc:
        return {
            "status": "ABSTAIN_F_SUMMARY_CONTENT",
            "reason": str(exc), "denominator": denominator,
            "bundle_denominator": {"unit": "shared force-summary bundle", "planned": 1,
                                   "valid": 0, "failed": 1, "abstained": 0},
            "certified": False, "physical_score": None,
        }
    return {
        "status": "F_SUMMARY_READ_NOT_CERTIFIED",
        "summary": summary,
        "diagnostic": diagnostic,
        "denominator": denominator,
        "bundle_denominator": {"unit": "shared force-summary bundle", "planned": 1,
                               "valid": 1, "failed": 0, "abstained": 0},
        "certified": False, "physical_score": None,
    }


def apply_wilson_force_snapshot_gate(
    run_state: object,
    envelopes: object,
    receipt: object,
    *,
    required_source_hashes_by_envelope: object,
    source_snapshots_by_envelope: object,
) -> dict[str, Any]:
    """Bind spectral q(T) and force-summary bounds to one F snapshot bundle."""
    routes = required_source_hashes_by_envelope
    snapshots = source_snapshots_by_envelope
    q_envelope = envelopes.get("q_excited") if isinstance(envelopes, Mapping) else None
    q_hashes = routes.get("q_excited") if isinstance(routes, Mapping) else None
    q_snapshot = snapshots.get("q_excited") if isinstance(snapshots, Mapping) else None
    q_chain = q_envelope.get("chain") if isinstance(q_envelope, Mapping) else None
    q_split = q_chain.get("split_id") if isinstance(q_chain, Mapping) else None
    q_receipt_hash = q_hashes.get("gevp_receipt") if isinstance(q_hashes, Mapping) else None
    if isinstance(q_snapshot, SourceByteSnapshot) and isinstance(q_receipt_hash, str):
        q_reader = read_spectral_q_bounds(
            q_snapshot,
            source_name="gevp_receipt",
            split_id=q_split,
            expected_source_sha256=q_receipt_hash,
        )
    else:
        q_reader = {
            "status": "ABSTAIN_F_SPECTRUM_ROUTE_MAP",
            "reason": "q_excited must bind a captured GEVP fit-support receipt",
            "source_denominator": {
                "unit": "immutable_snapshot_source_file", "planned": 1,
                "valid": 0, "failed": 0, "abstained": 1,
            },
            "selected_q_cell_denominator": {
                "unit": "selected_kernel_R_time_q_cell", "planned": 8,
                "valid": 0, "failed": 0, "abstained": 8,
            },
            "certified": False, "physical_score": None,
        }
    read = read_force_summary_snapshots(
        snapshots, routes
    )
    q_reconciliation = None
    if (q_reader.get("status") == "Q_BOUND_DERIVED_NOT_CERTIFIED"
            and read.get("status") == "F_SUMMARY_READ_NOT_CERTIFIED"):
        supplied = read["summary"].get("qhx_corner_bounds", {}).get("q")
        order = q_reader.get("force_qhx_corner_order")
        bounds = q_reader.get("force_q_corner_bounds")
        reasons = []
        valid_corners = 0
        failed_corners = 0
        abstained_corners = 0
        if (not isinstance(supplied, list) or len(supplied) != 2
                or not isinstance(order, list) or len(order) != 2
                or not isinstance(bounds, Mapping)):
            reasons.append("force q corners or spectral q corner map is malformed")
            abstained_corners = 8
        else:
            for kernel_index, kernel in enumerate(order):
                row = supplied[kernel_index]
                lower = bounds.get(kernel)
                if (not isinstance(row, list) or len(row) != 4
                        or not isinstance(lower, list) or len(lower) != 4):
                    reasons.append(f"{kernel}: expected four q corners in the selected force order")
                    abstained_corners += 4
                    continue
                for corner_index, (value, floor) in enumerate(zip(row, lower)):
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(float(value))
                            or isinstance(floor, bool) or not isinstance(floor, (int, float))
                            or not math.isfinite(float(floor))):
                        reasons.append(f"{kernel} q corner {corner_index} is not finite numeric input")
                        failed_corners += 1
                        continue
                    tolerance = 8.0 * math.ulp(max(1.0, abs(float(value)), abs(float(floor))))
                    if float(value) + tolerance < float(floor):
                        reasons.append(
                            f"{kernel} q corner {corner_index} is below the captured spectral q upper bound"
                        )
                        failed_corners += 1
                    else:
                        valid_corners += 1
        q_reconciliation = {
            "status": "Q_FORCE_CORNERS_RECONCILED" if not reasons else "ABSTAIN_Q_BELOW_SPECTRAL_BOUND",
            "kernel_order": order,
            "corner_denominator": {
                "unit": "selected_spectral_q_force_corner", "planned": 8,
                "valid": valid_corners, "failed": failed_corners,
                "abstained": abstained_corners,
            },
            "reasons": reasons,
            "certified": False,
            "physical_score": None,
        }
    else:
        q_reconciliation = {
            "status": "ABSTAIN_SPECTRAL_Q_READER",
            "kernel_order": None,
            "corner_denominator": {
                "unit": "selected_spectral_q_force_corner", "planned": 8,
                "valid": 0, "failed": 0, "abstained": 8,
            },
            "reasons": ["spectral q reader and force summary must both pass before F gating"],
            "certified": False,
            "physical_score": None,
        }

    if (read.get("status") != "F_SUMMARY_READ_NOT_CERTIFIED"
            or q_reader.get("status") != "Q_BOUND_DERIVED_NOT_CERTIFIED"
            or q_reconciliation.get("status") != "Q_FORCE_CORNERS_RECONCILED"):
        result = apply_wilson_force_gate(
            run_state, envelopes, receipt, None,
            required_source_hashes_by_envelope=routes,
            source_snapshots_by_envelope=snapshots,
        )
        result["force_summary_reader"] = read
        result["spectral_q_reader"] = q_reader
        result["spectral_q_reconciliation"] = q_reconciliation
        return result
    result = apply_wilson_force_gate(
        run_state, envelopes, receipt, read["summary"],
        required_source_hashes_by_envelope=routes,
        source_snapshots_by_envelope=snapshots,
    )
    result["force_summary_reader"] = {
        "status": read["status"],
        "diagnostic": read["diagnostic"],
        "denominator": read["denominator"],
        "bundle_denominator": read["bundle_denominator"],
        "certified": False,
        "physical_score": None,
    }
    result["spectral_q_reader"] = {
        "status": q_reader["status"],
        "source_sha256": q_reader.get("source_sha256"),
        "fit_variant_denominator": q_reader.get("fit_variant_denominator"),
        "selected_q_cell_denominator": q_reader.get("selected_q_cell_denominator"),
        "force_qhx_corner_order": q_reader.get("force_qhx_corner_order"),
        "force_qhx_corner_labels": q_reader.get("force_qhx_corner_labels"),
        "force_q_corner_bounds": q_reader.get("force_q_corner_bounds"),
        "text_flow": q_reader.get("text_flow"),
        "limitations": q_reader.get("limitations"),
        "certified": False,
        "physical_score": None,
    }
    result["spectral_q_reconciliation"] = q_reconciliation
    return result
