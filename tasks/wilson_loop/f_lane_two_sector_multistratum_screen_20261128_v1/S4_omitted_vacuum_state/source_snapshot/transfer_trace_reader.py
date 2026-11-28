"""Snapshot-only q/h/x reconstruction from a proposed two-sector fit summary.

This is a diagnostic consumer for fitted energies and transition overlaps. It
does not read raw Wilson correlators, produce their fit, estimate uncertainty,
or certify the proposed finite-Nt interpretation.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from core.source_integrity import SourceByteSnapshot


MANIFEST_SCHEMA = "wilson-f-transfer-fit-manifest-v1"
FIT_SCHEMA = "wilson-f-two-sector-spectrum-fit-v1"
RESULT_SCHEMA = "wilson-f-transfer-trace-reconstruction-v1"
RADIUS_ORDER = (2, 3)
KERNEL_ORDER = ("identity", "APE", "stout")
TIME_EXTENTS = (5, 6)
TEMPORAL_EXTENTS = (16, 32)
CONTRACT_PATH = Path(__file__).with_name("lane_contract_v2.json")
_MANIFEST_KEYS = {
    "schema", "split_id", "chain_id", "geometry", "action", "radius_order",
    "kernel_order", "time_extents", "temporal_extents", "fit_summary_sha256",
    "source_protocol_sha256",
}
_FIT_KEYS = {
    "schema", "radius_order", "kernel_order", "vacuum_energies",
    "static_energies_by_R", "transition_overlaps_by_R",
}


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(snapshot: SourceByteSnapshot, name: str) -> tuple[dict[str, Any], bytes]:
    payload = snapshot.read_bytes(name)
    if sha256(payload).hexdigest() != snapshot.expected_hashes.get(name):
        raise ValueError(f"{name}: captured bytes do not match expected hash")
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {token}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name}: invalid strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name}: JSON root must be an object")
    return value, payload


def _validate_contract(document: dict[str, Any]) -> None:
    if document.get("schema") != "wilson-two-lane-contract-v2":
        raise ValueError("captured lane contract schema mismatch")
    selected = document.get("selected_protocol")
    if not isinstance(selected, dict):
        raise ValueError("captured lane contract lacks selected_protocol object")
    try:
        force = selected["ground_state_discrete_static_force"]
        spectral = force["spectral_evidence_plan"]
        q_contract = spectral["q_overlap_contract"]
        trace = q_contract["finite_Nt_trace_decomposition"]
    except (KeyError, TypeError) as exc:
        raise ValueError("captured lane contract lacks the selected F-lane trace definition") from exc
    if not all(isinstance(value, dict) for value in (force, spectral, q_contract, trace)):
        raise ValueError("captured lane contract trace sections must be objects")
    if (trace.get("definition_id") != "two_sector_transfer_trace_qhx_v1"
            or q_contract.get("definition_id") != "bare_basis_squared_overlap_ratio_v1"
            or q_contract.get("kernel_to_basis_operator") != {
                "identity": "identity", "ape_alpha0.4_n5": "APE",
                "spatial_stout_rho0.10_n5": "stout",
            }
            or spectral.get("matched_thermal_Nt") != 32):
        raise ValueError("captured lane contract does not match q/h/x reader semantics")


def _complex_overlap_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.shape != (*shape, 2) or not np.all(np.isfinite(raw)):
        raise ValueError(f"{label} must be finite complex pairs with shape {shape + (2,)}")
    result = raw[..., 0] + 1j * raw[..., 1]
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} contains a non-finite complex overlap")
    return result


def _component_values(
    vacuum_energies: np.ndarray,
    static_energies: np.ndarray,
    amplitudes: np.ndarray,
    operator_index: int,
    time_extent: int,
    temporal_extent: int,
) -> dict[str, float]:
    ground_overlap = float(abs(amplitudes[0, 0, operator_index]) ** 2)
    if not math.isfinite(ground_overlap) or ground_overlap <= np.finfo(float).tiny:
        raise ValueError("bare operator has zero or non-finite ground-sector overlap")
    e0 = float(static_energies[0])
    q = sum(
        float(abs(amplitudes[0, n, operator_index]) ** 2) / ground_overlap
        * math.exp(-(float(static_energies[n]) - e0) * time_extent)
        for n in range(1, static_energies.size)
    )
    h = sum(
        float(abs(amplitudes[m, 0, operator_index]) ** 2) / ground_overlap
        * math.exp(-float(vacuum_energies[m]) * (temporal_extent - time_extent))
        for m in range(1, vacuum_energies.size)
    )
    x = sum(
        float(abs(amplitudes[m, n, operator_index]) ** 2) / ground_overlap
        * math.exp(-float(vacuum_energies[m]) * (temporal_extent - time_extent)
                   -(float(static_energies[n]) - e0) * time_extent)
        for m in range(1, vacuum_energies.size)
        for n in range(1, static_energies.size)
    )

    partition = float(np.sum(np.exp(-vacuum_energies * temporal_extent)))
    weights = np.exp(
        -vacuum_energies[:, None] * (temporal_extent - time_extent)
        -static_energies[None, :] * time_extent
    )
    full_diagonal = float(np.sum(weights * np.abs(amplitudes[:, :, operator_index]) ** 2)
                          / partition)
    ground = (ground_overlap * math.exp(-float(vacuum_energies[0])
                                        * (temporal_extent - time_extent)
                                        - e0 * time_extent) / partition)
    ratio = full_diagonal / ground
    reconstructed = 1.0 + q + h + x
    if not all(math.isfinite(value) and value >= 0 for value in (q, h, x, ratio)):
        raise ValueError("q/h/x reconstruction produced a negative or non-finite component")
    return {
        "q": float(q), "h": float(h), "x": float(x),
        "delta_log": float(math.log1p(q + h + x)),
        "full_to_ground_ratio": float(ratio),
        "reconstructed_ratio": float(reconstructed),
        "decomposition_abs_error": float(abs(ratio - reconstructed)),
    }


def read_transfer_trace_snapshot(
    snapshot: SourceByteSnapshot,
    *, manifest_name: str = "transfer_manifest",
    fit_name: str = "transfer_fit",
    protocol_name: str = "lane_contract",
) -> dict[str, Any]:
    """Read a captured two-sector fit summary and reconstruct the 24 q/h/x cells."""
    if not isinstance(snapshot, SourceByteSnapshot):
        raise ValueError("snapshot must be a SourceByteSnapshot")
    manifest, _ = _strict_json(snapshot, manifest_name)
    if set(manifest) != _MANIFEST_KEYS or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("Wilson transfer manifest keys or schema mismatch")
    split_id = manifest.get("split_id")
    if (not isinstance(split_id, str) or not split_id.upper().startswith("F")
            or any(marker in split_id.lower()
                   for marker in ("gap", "validation", "holdout", "sealed", "confirmation"))):
        raise ValueError("Wilson transfer reader accepts only an open F calibration/development split")
    if (not isinstance(manifest.get("chain_id"), str) or not manifest["chain_id"].strip()
            or manifest.get("geometry") != "S16T16"
            or manifest.get("action") != "wilson_gauge_beta6"
            or manifest.get("radius_order") != list(RADIUS_ORDER)
            or manifest.get("kernel_order") != list(KERNEL_ORDER)
            or manifest.get("time_extents") != list(TIME_EXTENTS)
            or manifest.get("temporal_extents") != list(TEMPORAL_EXTENTS)):
        raise ValueError("Wilson transfer manifest identity, geometry, action or axis order mismatch")

    protocol, protocol_bytes = _strict_json(snapshot, protocol_name)
    protocol_hash = sha256(protocol_bytes).hexdigest()
    if protocol_hash != manifest.get("source_protocol_sha256"):
        raise ValueError("Wilson transfer protocol bytes do not match the manifest hash")
    _validate_contract(protocol)

    fit, fit_bytes = _strict_json(snapshot, fit_name)
    fit_hash = sha256(fit_bytes).hexdigest()
    if fit_hash != manifest.get("fit_summary_sha256"):
        raise ValueError("Wilson transfer fit bytes do not match the manifest hash")
    if set(fit) != _FIT_KEYS or fit.get("schema") != FIT_SCHEMA:
        raise ValueError("Wilson transfer fit keys or schema mismatch")
    if fit.get("radius_order") != list(RADIUS_ORDER) or fit.get("kernel_order") != list(KERNEL_ORDER):
        raise ValueError("Wilson transfer fit semantic axis order mismatch")
    static_by_radius = fit.get("static_energies_by_R")
    overlaps_by_radius = fit.get("transition_overlaps_by_R")
    if (not isinstance(static_by_radius, dict)
            or set(static_by_radius) != {str(radius) for radius in RADIUS_ORDER}
            or not isinstance(overlaps_by_radius, dict)
            or set(overlaps_by_radius) != {str(radius) for radius in RADIUS_ORDER}):
        raise ValueError("Wilson transfer fit must contain exactly the selected R=2,3 rows")
    eps = np.asarray(fit.get("vacuum_energies"), dtype=np.float64)
    if (eps.ndim != 1 or eps.size < 2 or not np.all(np.isfinite(eps))
            or eps[0] != 0.0 or np.any(np.diff(eps) <= 0)):
        raise ValueError("vacuum energies must be finite, ordered and normalized to eps_0=0")

    rows = []
    failed = []
    for radius in RADIUS_ORDER:
        static = np.asarray(static_by_radius[str(radius)], dtype=np.float64)
        if (static.ndim != 1 or static.size < 2 or not np.all(np.isfinite(static))
                or static[0] <= 0 or np.any(np.diff(static) <= 0)):
            raise ValueError(f"R={radius}: static energies must be positive and strictly ordered")
        amplitudes = _complex_overlap_array(
            overlaps_by_radius[str(radius)],
            (eps.size, static.size, len(KERNEL_ORDER)), f"R={radius} transition overlaps",
        )
        for kernel_index, kernel in enumerate(KERNEL_ORDER):
            for time_extent in TIME_EXTENTS:
                for temporal_extent in TEMPORAL_EXTENTS:
                    planned_key = {"R": radius, "kernel_id": kernel,
                                   "T": time_extent, "N_t": temporal_extent}
                    try:
                        components = _component_values(
                            eps, static, amplitudes, kernel_index,
                            time_extent, temporal_extent,
                        )
                    except (ValueError, OverflowError, FloatingPointError) as exc:
                        failed.append({**planned_key, "reason": str(exc)})
                    else:
                        rows.append({**planned_key, **components})
    planned = len(RADIUS_ORDER) * len(KERNEL_ORDER) * len(TIME_EXTENTS) * len(TEMPORAL_EXTENTS)
    valid = len(rows)
    return {
        "schema": RESULT_SCHEMA,
        "status": "diagnostic_only" if valid == planned else "ABSTAIN_INVALID_TRANSFER_CELLS",
        "certified": False, "physical_score": None,
        "split_id": split_id, "chain_id": manifest["chain_id"],
        "source_file_denominator": {
            "unit": "captured_source_file", "planned": 3, "valid": 3,
            "failed": 0, "abstained": 0,
        },
        "source_hashes": {
            manifest_name: snapshot.expected_hashes[manifest_name],
            fit_name: fit_hash, protocol_name: protocol_hash,
        },
        "component_definition": {
            "q": "m=0,n>0; excited static-source states with vacuum ground",
            "h": "m>0,n=0; excited vacuum states with static-source ground",
            "x": "m>0,n>0; both transfer sectors excited",
            "normalization": "C_kk/C_kk^(0,0)=1+q+h+x; delta_log=log1p(q+h+x)",
            "fit_input": "caller-supplied fitted energies and transition overlaps; raw fits are not performed here",
        },
        "cell_denominator": {"unit": "R_kernel_T_Nt_cell", "planned": planned,
                             "valid": valid, "failed": len(failed), "abstained": 0},
        "physical_measurement_denominator": {
            "unit": "authorized_production_measurement", "planned": 0,
            "valid": 0, "failed": 0, "abstained": 0,
        },
        "certification_gate_denominator": {
            "unit": "F_lane_certification_gate", "planned": 1,
            "valid": 0, "failed": 0, "abstained": 1,
        },
        "cells": rows, "failure_reasons": failed,
        "open_gates": [
            "proposed endpoint factorization and source/sink timing still require independent review",
            "input is a fit summary; this reader does not parse or fit raw C_ij(R,t) correlators",
            "no covariance, simultaneous confidence interval, coverage, or q/h/x upper bound is computed",
            "no matched-Nt thermal receipt, force propagation, reviewer identity, or authorization is established",
        ],
    }
