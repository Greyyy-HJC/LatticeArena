"""CPU-only structural and numerical checks for Wilson lane adapters."""
from __future__ import annotations

import hashlib
import json
import math
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from core.evidence import evidence_envelope
from core.receipts import denominator, run_receipt, scenario_spec
from core.source_integrity import capture_source_snapshot
from core.state import RunState
from tasks.wilson_loop.evidence_adapter import (
    CONTRACT_PATH,
    _F_UNITS,
    _F_ROUTES,
    apply_wilson_creutz_gate,
    apply_wilson_force_gate,
    creutz_map_sha256,
    force_map_sha256,
)
from tasks.wilson_loop.lane_contract import selected_protocol
from tasks.wilson_loop.force_summary_reader import (
    apply_wilson_force_snapshot_gate,
)
from tasks.wilson_loop.spectral_q_bound_reader import (
    CONTAMINATION_TIMES,
    ENDPOINT_SLOTS_PER_Q_CELL,
    FORCE_Q_CORNER_LABELS,
    FORCE_Q_KERNEL_ORDER,
    FIT_WINDOW,
    KERNEL_TO_BASIS_OPERATOR,
    KERNELS,
    OVERLAP_DEFINITION_ID,
    R_VALUES,
    SCHEMA as Q_SUPPORT_SCHEMA,
    STATE_COUNTS,
    TMIN_VALUES,
    UNITS as Q_SUPPORT_UNITS,
)
from tasks.wilson_loop.c_lane_snapshot_reader import (
    MANIFEST_SCHEMA as C_MANIFEST_SCHEMA,
    MEASUREMENT_SCHEMA as C_MEASUREMENT_SCHEMA,
    SUMMARY_SCHEMA as C_SUMMARY_SCHEMA,
    apply_wilson_creutz_snapshot_gate,
)
from tasks.wilson_loop.c_lane_open_development_diagnostic import (
    BLOCK_LENGTHS,
    N_DRAWS as C_N_DRAWS,
    analyze as analyze_creutz_values,
)


H = [f"{i:064x}" for i in range(1, 60)]
PAIRED_DRAWS = 30_000


def _q_fit_support_payload(split_id="F-calibration-open"):
    cell_alpha = 0.0125 / 28
    variants = []
    for kernel in KERNELS:
        for radius in R_VALUES:
            for tmin in TMIN_VALUES:
                for n_states in STATE_COUNTS:
                    ratios = [0.0001]
                    gaps = [0.2]
                    if n_states == 3:
                        ratios.append(0.00005)
                        gaps.append(0.1)
                    variants.append({
                        "variant_id": f"{kernel}-R{radius}-tmin{tmin}-n{n_states}",
                        "kernel_id": kernel,
                        "basis_operator_id": KERNEL_TO_BASIS_OPERATOR[kernel],
                        "R": radius,
                        "n_states": n_states,
                        "tmin": tmin,
                        "fit_window": [tmin, FIT_WINDOW[1]],
                        "overlap_ratio_upper": ratios,
                        "gap_lower": gaps,
                    })
    document = {
        "schema": Q_SUPPORT_SCHEMA,
        "classification": "fit_receipt_reducer_input_not_raw_measurement",
        "split_id": split_id,
        "units": Q_SUPPORT_UNITS,
        "overlap_definition_id": OVERLAP_DEFINITION_ID,
        "positive_forward_spectral_sum_assumption": True,
        "gevp": {
            "dimension": 3,
            "basis_kernel_ids": list(KERNELS),
            "basis_operator_ids": [KERNEL_TO_BASIS_OPERATOR[kernel] for kernel in KERNELS],
            "t0": 1,
            "td": 2,
            "R_values": list(R_VALUES),
            "state_counts": list(STATE_COUNTS),
            "fit_window": list(FIT_WINDOW),
            "tmin_checks": list(TMIN_VALUES),
            "contamination_times": list(CONTAMINATION_TIMES),
        },
        "confidence_allocation": {
            "method": "Bonferroni_one_sided_marginal_bounds_within_each_q_cell",
            "systematic_cell_alpha": cell_alpha,
            "endpoint_slots_per_q_cell": ENDPOINT_SLOTS_PER_Q_CELL,
            "endpoint_alpha": cell_alpha / ENDPOINT_SLOTS_PER_Q_CELL,
            "unused_endpoint_slots_reassigned": False,
        },
        "fit_variants": variants,
    }
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _review(digest):
    return {
        "status": "reviewed", "receipt_sha256": digest,
        "scope_sha256": H[40], "reviewer_role_claim": "independent",
        "independence_claim_sha256": H[41],
    }


def _receipt(name, denominators):
    return run_receipt(
        scenario_spec(name, "planned lane adapter unit", "synthetic contract fixture"),
        denominators,
        artifact_hashes={"fixture": H[42]},
        text_flow="lane inputs -> task envelope(s) -> exact denominator check -> draft or abstain",
    )


def _ready_state(*stages):
    state = RunState()
    for stage in stages:
        state.set(stage, "draft", evidence=f"fixture {stage}")
        state.set(stage, "validated", evidence_hash=H[43])
    return state


def _creutz_fixture(*, split="C-calibration-open", halfwidth=0.01):
    hashes = {
        "manifest": H[0], "loop_map": creutz_map_sha256(),
        "loop_measurements": H[1], "dependence_receipt": H[2],
        "coverage_calibration_receipt": H[3], "interval_summary": H[4],
        "review_receipt": H[5], "authorization_receipt": H[6],
    }
    envelope = evidence_envelope(
        "wilson_loop", "finite_t_creutz_interval",
        chain={
            "chain_id": "C-lane-calibration-chain", "split_id": split,
            "state_unit": "paired saved gauge configuration",
            "manifest_sha256": hashes["manifest"],
        },
        observable={
            "map_id": "wilson-finite-t-creutz-R2-T5-stout-v1",
            "map_sha256": hashes["loop_map"],
            "support": {
                "lower": -1.0, "upper": 1.0,
                "unit": "dimensionless finite-T Creutz observable",
            },
        },
        method={
            "name": "paired_circular_block_bootstrap_hull",
            "version": "selected-10000x3-v1",
            "assumptions": ["four loop means are paired within each saved configuration"],
        },
        dependence={
            "mode": "markov", "assumptions": ["production chain is homogeneous"],
            "gamma_ps_lower": 0.3, "tv_bound_after_burn_in": 0.001,
            "burn_in": {"unit": "saved gauge configuration", "count": 8},
        },
        confidence={
            "family_id": "wilson-two-lane-selected-v3", "alpha_total": 0.05,
            "alpha_allocated": 0.025, "family_size": 2,
        },
        source_hashes=hashes, review=_review(hashes["review_receipt"]),
    )
    receipt = _receipt(
        "wilson-finite-t-creutz-calibration",
        {
            "four_required_loop_means": denominator("required paired loop mean", planned=4, valid=4),
            "paired_bootstrap_draws": denominator("paired circular-block draw", planned=PAIRED_DRAWS, valid=PAIRED_DRAWS),
        },
    )
    summary = {
        "hull_lower": 0.09, "hull_upper": 0.11,
        "halfwidth": halfwidth, "invalid_draws": 0,
        "planned_draws": PAIRED_DRAWS,
        "units": "dimensionless",
        "summary_sha256": hashes["interval_summary"],
    }
    return envelope, receipt, summary, hashes


def _creutz_snapshot_fixture(tmp_path, *, split="C-calibration-open", summary_delta=0.0,
                            summary_seed_base=None):
    envelope, receipt, _summary, _hashes = _creutz_fixture(split=split)
    protocol = selected_protocol()
    c_lane = protocol["finite_T_Creutz"]
    ids = list(range(12))
    kernel_ids = ["identity", "ape_alpha0.4_n5", c_lane["kernel_id"]]
    manifest = {
        "schema": C_MANIFEST_SCHEMA,
        "split_id": split,
        "config_ids": ids,
        "geometry": protocol["ensemble_design"]["geometry"],
        "action": "Wilson gauge action, beta=6",
        "boundary_conditions": protocol["ensemble_design"]["boundary_conditions"],
        "r_values": [1, 2, 3, 4],
        "t_values": [4, 5, 6, 7, 8],
        "kernel_ids": kernel_ids,
        "primary_kernel_id": c_lane["kernel_id"],
        "kernel_parameters": {c_lane["kernel_id"]: {
            "kind": "stout", "rho": 0.1, "steps": 5,
        }},
    }
    manifest_bytes = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    loops = np.full((len(ids), len(kernel_ids), 4, 5), 0.4, dtype=np.float64)
    stout = loops[:, kernel_ids.index(c_lane["kernel_id"])]
    stout[:, 2, 2] = 0.3  # W(3,6)
    stout[:, 1, 1] = 0.5  # W(2,5)
    stout[:, 2, 1] = 0.4  # W(3,5)
    stout[:, 1, 2] = 0.4  # W(2,6)
    npz_stream = BytesIO()
    np.savez_compressed(
        npz_stream,
        schema=np.asarray(C_MEASUREMENT_SCHEMA), split_id=np.asarray(split),
        config_ids=np.asarray(ids, dtype=np.int64), kernel_ids=np.asarray(kernel_ids),
        r_values=np.asarray([1, 2, 3, 4], dtype=np.int64),
        t_values=np.asarray([4, 5, 6, 7, 8], dtype=np.int64),
        manifest_sha256=np.asarray(hashlib.sha256(manifest_bytes).hexdigest()),
        loops=loops,
    )
    measurement_bytes = npz_stream.getvalue()
    seed_base = protocol["finite_T_Creutz"]["bootstrap_seed_base"]
    analyzed = analyze_creutz_values(stout, seed_base=seed_base)
    _means, _point, _point_valid, rows, _draws, lower, upper, halfwidth, _passed = analyzed
    invalid = sum(row["invalid_draws"] for row in rows)
    summary = {
        "schema": C_SUMMARY_SCHEMA,
        "units": "dimensionless",
        "hull_lower": lower + summary_delta,
        "hull_upper": upper,
        "halfwidth": (upper - lower) / 2 + summary_delta / 2,
        "invalid_draws": invalid,
        "planned_draws": len(rows) * C_N_DRAWS,
        "resampling_protocol": {
            "method": "paired circular moving-block bootstrap",
            "block_lengths": list(BLOCK_LENGTHS),
            "draws_per_length": C_N_DRAWS,
            "alpha_lane": protocol["familywise_inference"]["alpha_each_lane"],
            "invalid_fraction_max": protocol["familywise_inference"][
                "block_bootstrap_invalid_draw_fraction_max"
            ],
            "precision_halfwidth_target": c_lane["precision_halfwidth_target"],
            "seed_base": seed_base if summary_seed_base is None else summary_seed_base,
            "seed_rule": "seed = seed_base + block_length",
            "interval_level": 0.975,
        },
    }
    summary_bytes = json.dumps(
        summary, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    (tmp_path / "manifest.json").write_bytes(manifest_bytes)
    (tmp_path / "loops.npz").write_bytes(measurement_bytes)
    (tmp_path / "interval-summary.json").write_bytes(summary_bytes)
    common_bytes = b"synthetic C dependence, coverage, review, and authorization receipts"
    (tmp_path / "common.bin").write_bytes(common_bytes)
    map_bytes = json.dumps({
        "protocol_id": protocol["protocol_id"],
        "finite_T_Creutz": c_lane,
        "lane_contract": json.loads(CONTRACT_PATH.read_text())["lanes"]["finite_T_Creutz"],
    }, sort_keys=True, separators=(",", ":")).encode()
    (tmp_path / "loop-map.json").write_bytes(map_bytes)
    hashes = {
        "manifest": hashlib.sha256(manifest_bytes).hexdigest(),
        "loop_map": hashlib.sha256(map_bytes).hexdigest(),
        "loop_measurements": hashlib.sha256(measurement_bytes).hexdigest(),
        "dependence_receipt": hashlib.sha256(common_bytes).hexdigest(),
        "coverage_calibration_receipt": hashlib.sha256(common_bytes).hexdigest(),
        "interval_summary": hashlib.sha256(summary_bytes).hexdigest(),
        "review_receipt": hashlib.sha256(common_bytes).hexdigest(),
        "authorization_receipt": hashlib.sha256(common_bytes).hexdigest(),
    }
    envelope["chain"]["manifest_sha256"] = hashes["manifest"]
    envelope["source_hashes"] = hashes
    envelope["review"]["receipt_sha256"] = hashes["review_receipt"]
    paths = {
        "manifest": "manifest.json", "loop_map": "loop-map.json",
        "loop_measurements": "loops.npz", "interval_summary": "interval-summary.json",
        "dependence_receipt": "common.bin", "coverage_calibration_receipt": "common.bin",
        "review_receipt": "common.bin", "authorization_receipt": "common.bin",
    }
    snapshot = capture_source_snapshot(tmp_path, hashes, paths)
    return envelope, receipt, hashes, snapshot, measurement_bytes


def _force_fixture():
    envelopes = {}
    routes = {}
    summary_hash = H[7]
    review_hash = H[8]
    authorization_hash = H[9]
    gevp_hash = H[10]
    for index, route in enumerate(_F_ROUTES):
        hashes = {
            "manifest": H[11] if route in ("force_interval", "positive_force_scale") else H[11 + index],
            "observable_map": force_map_sha256(route),
            "data_receipt": H[19 + index],
            "force_summary": summary_hash,
            "review_receipt": review_hash,
            "authorization_receipt": authorization_hash,
        }
        if route in ("force_interval", "positive_force_scale"):
            hashes.update({"dependence_receipt": H[27], "coverage_calibration_receipt": H[28]})
        else:
            hashes.update({
                "gevp_receipt": gevp_hash,
                "component_receipt": H[29 + index],
                "propagation_receipt": H[30 + index],
            })
            if route == "h_thermal_nt32":
                hashes["matched_nt32_receipt"] = H[36]
        routes[route] = hashes
        if route in ("force_interval", "positive_force_scale"):
            chain_id = "paired-F-calibration-chain"
            split_id = "F-calibration-open"
            state_unit = "paired saved gauge configuration"
            mode = "markov"
            confidence = {
                "family_id": "wilson-two-lane-selected-v3/F/statistical",
                "alpha_total": 0.0125,
                "alpha_allocated": 0.00625,
                "family_size": 2,
            }
        else:
            chain_id = f"{route}-spectral-chain"
            split_id = f"F-calibration-{route}"
            state_unit = "spectral fit cell"
            mode = "markov" if route == "h_thermal_nt32" else "spectral"
            cells = 4 if route == "spectral_model_truncation" else 8
            confidence = {
                "family_id": "wilson-two-lane-selected-v3/F/systematic",
                "alpha_total": 0.0125,
                "alpha_allocated": 0.0125 * cells / 28,
                "family_size": cells,
            }
        if route == "positive_force_scale":
            support = {"lower": 0.95, "upper": 1.1, "unit": "dimensionless lattice force a_s*a_t*F"}
        elif route == "force_interval":
            support = {"lower": 0.99, "upper": 1.07, "unit": "dimensionless lattice force a_s*a_t*F"}
        elif route in ("q_excited", "h_thermal_nt32", "x_mixed"):
            unit = {
                "q_excited": "dimensionless q bound",
                "h_thermal_nt32": "dimensionless h bound",
                "x_mixed": "dimensionless x bound",
            }[route]
            support = {"lower": 0.0, "upper": 1.0, "unit": unit}
        else:
            support = {"lower": 0.0, "upper": 1.0,
                       "unit": "dimensionless lattice force a_s*a_t*F"}
        dependence = {
            "mode": mode,
            "assumptions": ["source receipt covers the declared route and saved-state unit"],
        }
        if mode == "markov":
            dependence.update({
                "gamma_ps_lower": 0.3,
                "tv_bound_after_burn_in": 0.001,
                "burn_in": {"unit": state_unit, "count": 8},
            })
        envelopes[route] = evidence_envelope(
            "wilson_loop", {
                "force_interval": "ground_state_force_interval",
                "positive_force_scale": "positive_absolute_force_scale",
                "q_excited": "excited_state_q_envelope",
                "h_thermal_nt32": "matched_nt32_thermal_h_envelope",
                "x_mixed": "mixed_sector_x_envelope",
                "spectral_model_truncation": "spectral_model_truncation_envelope",
            }[route],
            chain={
                "chain_id": chain_id, "split_id": split_id,
                "state_unit": state_unit, "manifest_sha256": hashes["manifest"],
            },
            observable={
                "map_id": f"wilson-ground-state-force-R2.5-v2/{route}",
                "map_sha256": hashes["observable_map"], "support": support,
            },
            method={
                "name": "paired_circular_block_bootstrap_hull"
                if route in ("force_interval", "positive_force_scale")
                else "joint_3x3_finite_Nt_spectral_envelope",
                "version": "selected-F-v1",
                "assumptions": ["primary force bounds use the four stout corners; identity corners are comparison diagnostics"],
            },
            dependence=dependence,
            confidence=confidence,
            source_hashes=hashes,
            review=_review(review_hash),
        )
    receipt = _receipt(
        "wilson-ground-state-force-calibration",
        {
            "force_interval_draws": denominator("force interval paired draw", planned=PAIRED_DRAWS, valid=PAIRED_DRAWS),
            "positive_force_scale_draws": denominator("positive force reference draw", planned=PAIRED_DRAWS, valid=PAIRED_DRAWS),
            "gevp_fit_variants": denominator("3x3 GEVP fit variant", planned=36, valid=36),
            "force_systematic_cells": denominator("simultaneous force systematic cell", planned=28, valid=28),
            "matched_nt32_thermal_receipts": denominator("matched Nt=32 thermal receipt", planned=1, valid=1),
        },
    )
    summary = {
        "units": "dimensionless lattice force a_s*a_t*F",
        "stout_force_estimate": 1.03,
        "stout_force_interval_lower": 0.99,
        "stout_force_interval_upper": 1.07,
        "positive_force_scale_lower": 0.95,
        "statistical_halfwidth": 0.04,
        "force_component_bounds": {
            "excited_state_force": 0.01,
            "finite_T_thermal": 0.01,
            "mixed": 0.005,
            "spectral_model_truncation": 0.005,
        },
        "systematic_total_bound": 0.03,
        "total_linear_bound": 0.07,
        "invalid_force_interval_draws": 0,
        "invalid_force_scale_draws": 0,
        "summary_sha256": summary_hash,
    }
    return envelopes, receipt, summary, routes


def _force_summary_snapshot_bundle(tmp_path, *, summary_override=None):
    envelopes, receipt, _summary, routes = _force_fixture()
    protocol = selected_protocol()
    raw_summary = {
        "schema": "wilson-f-force-summary-input-v3",
        "units": "dimensionless lattice force a_s*a_t*F",
        "stout_force_estimate": 1.03,
        "stout_force_interval_lower": 0.99,
        "stout_force_interval_upper": 1.07,
        "positive_force_scale_lower": 0.95,
        "statistical_halfwidth": 0.04,
        "model_truncation_bound": 0.005,
        "invalid_force_interval_draws": 0,
        "invalid_force_scale_draws": 0,
        "force_kernel_order": list(FORCE_Q_KERNEL_ORDER),
        "force_corner_order": list(FORCE_Q_CORNER_LABELS),
        "qhx_corner_bounds": {
            "q": [[0.001] * 4 for _ in range(2)],
            "h": [[0.001] * 4 for _ in range(2)],
            "x": [[0.0005] * 4 for _ in range(2)],
        },
    }
    if summary_override is not None:
        raw_summary = summary_override
    summary_bytes = json.dumps(
        raw_summary, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    (tmp_path / "force-summary.json").write_bytes(summary_bytes)
    common_bytes = b"synthetic shared Wilson F evidence receipts"
    (tmp_path / "common.bin").write_bytes(common_bytes)
    q_support_bytes = _q_fit_support_payload(
        split_id=envelopes["q_excited"]["chain"]["split_id"]
    )
    (tmp_path / "q-fit-support.json").write_bytes(q_support_bytes)
    summary_hash = hashlib.sha256(summary_bytes).hexdigest()
    common_hash = hashlib.sha256(common_bytes).hexdigest()
    q_support_hash = hashlib.sha256(q_support_bytes).hexdigest()
    source_paths = {}

    for route in _F_ROUTES:
        force = protocol["ground_state_discrete_static_force"]
        map_bytes = json.dumps({
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
        }, sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(map_bytes).hexdigest() == force_map_sha256(route)
        map_name = f"{route}-map.json"
        (tmp_path / map_name).write_bytes(map_bytes)
        route_hashes = {
            name: (force_map_sha256(route) if name == "observable_map"
                   else summary_hash if name == "force_summary"
                   else q_support_hash if name == "gevp_receipt" else common_hash)
            for name in routes[route]
        }
        routes[route] = route_hashes
        envelopes[route]["chain"]["manifest_sha256"] = common_hash
        envelopes[route]["source_hashes"] = route_hashes
        envelopes[route]["review"]["receipt_sha256"] = common_hash
        source_paths[route] = {
            name: (map_name if name == "observable_map"
                   else "force-summary.json" if name == "force_summary"
                   else "q-fit-support.json" if name == "gevp_receipt"
                   else "common.bin")
            for name in route_hashes
        }

    snapshots = {
        route: capture_source_snapshot(tmp_path, routes[route], source_paths[route])
        for route in _F_ROUTES
    }
    return envelopes, receipt, routes, snapshots, summary_bytes


def test_creutz_lane_checks_frozen_point_precision_and_keeps_score_null():
    envelope, receipt, summary, hashes = _creutz_fixture()
    state = _ready_state("operator", "numerical")
    result = apply_wilson_creutz_gate(
        state, envelope, receipt, summary, required_source_hashes=hashes,
        require_source_bytes=False,
    )
    assert result["status"] == "DRAFT_PENDING_TASK_VALIDATION"
    assert state.status["statistical"] == "draft"
    assert result["physical_score"] is None
    assert result["certified"] is False


def test_creutz_lane_requires_explicit_source_bytes_by_default():
    envelope, receipt, summary, hashes = _creutz_fixture()
    state = _ready_state("operator", "numerical")
    result = apply_wilson_creutz_gate(
        state, envelope, receipt, summary, required_source_hashes=hashes,
    )
    assert result["status"] == "ABSTAIN_WILSON_C_LANE"
    assert state.status["statistical"] == "abstained"
    assert any("source-byte verification requires" in reason for reason in result["reasons"])
    assert result["source_verification"]["denominator"] == {
        "unit": "source_file", "planned": 8, "valid": 0,
        "failed": 0, "abstained": 8, "not_run": 0,
    }


def test_creutz_lane_verifies_explicit_source_bytes_before_draft(tmp_path):
    envelope, receipt, summary, hashes = _creutz_fixture()
    common_bytes = b"synthetic shared Wilson C receipt"
    (tmp_path / "common.bin").write_bytes(common_bytes)
    common_hash = hashlib.sha256(common_bytes).hexdigest()
    protocol = selected_protocol()
    map_bytes = json.dumps({
        "protocol_id": protocol["protocol_id"],
        "finite_T_Creutz": protocol["finite_T_Creutz"],
        "lane_contract": json.loads(CONTRACT_PATH.read_text())["lanes"]["finite_T_Creutz"],
    }, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(map_bytes).hexdigest() == creutz_map_sha256()
    (tmp_path / "creutz-map.json").write_bytes(map_bytes)

    hashes = {
        name: (creutz_map_sha256() if name == "loop_map" else common_hash)
        for name in hashes
    }
    envelope["chain"]["manifest_sha256"] = common_hash
    envelope["source_hashes"] = hashes
    envelope["review"]["receipt_sha256"] = common_hash
    summary["summary_sha256"] = common_hash
    source_paths = {
        name: ("creutz-map.json" if name == "loop_map" else "common.bin")
        for name in hashes
    }

    state = _ready_state("operator", "numerical")
    result = apply_wilson_creutz_gate(
        state, envelope, receipt, summary,
        required_source_hashes=hashes,
        source_root=tmp_path,
        source_paths=source_paths,
    )
    assert result["status"] == "DRAFT_PENDING_TASK_VALIDATION"
    assert result["source_byte_identity_verified"] is True
    assert result["source_verification"]["denominator"] == {
        "unit": "source_file", "planned": 8, "valid": 8,
        "failed": 0, "abstained": 0, "not_run": 0,
    }

    snapshot = capture_source_snapshot(tmp_path, hashes, source_paths)
    assert snapshot.read_bytes("loop_map") == map_bytes
    (tmp_path / "common.bin").write_bytes(b"changed after the lane reader consumed its snapshot")
    state = _ready_state("operator", "numerical")
    result = apply_wilson_creutz_gate(
        state, envelope, receipt, summary,
        required_source_hashes=hashes,
        source_snapshot=snapshot,
    )
    assert result["status"] == "DRAFT_PENDING_TASK_VALIDATION"
    assert result["source_verification"]["denominator"]["valid"] == 8

    (tmp_path / "common.bin").write_bytes(b"tampered Wilson C receipt")
    state = _ready_state("operator", "numerical")
    result = apply_wilson_creutz_gate(
        state, envelope, receipt, summary,
        required_source_hashes=hashes,
        source_root=tmp_path,
        source_paths=source_paths,
    )
    assert result["status"] == "ABSTAIN_SOURCE_BYTES"
    assert result["source_verification"]["denominator"] == {
        "unit": "source_file", "planned": 8, "valid": 1,
        "failed": 7, "abstained": 0, "not_run": 0,
    }


def test_creutz_snapshot_reader_recomputes_interval_from_same_loop_bytes(tmp_path):
    envelope, receipt, hashes, snapshot, measurement_bytes = _creutz_snapshot_fixture(tmp_path)
    (tmp_path / "loops.npz").write_bytes(b"changed after immutable capture")

    result = apply_wilson_creutz_snapshot_gate(
        _ready_state("operator", "numerical"), envelope, receipt,
        required_source_hashes=hashes, source_snapshot=snapshot,
    )

    assert result["status"] == "DRAFT_PENDING_TASK_VALIDATION"
    assert result["physical_score"] is None and result["certified"] is False
    reader = result["creutz_snapshot_reader"]
    assert reader["status"] == "ARRAY_REDUCED_NOT_CERTIFIED"
    assert reader["source_hashes"]["loop_measurements"] == hashlib.sha256(measurement_bytes).hexdigest()
    assert reader["summary_consistency"] == "MATCHED_TO_CAPTURED_LOOP_MATRIX"
    selected = selected_protocol()
    assert reader["resampling_protocol"]["seed_base"] == selected[
        "finite_T_Creutz"]["bootstrap_seed_base"]
    assert reader["resampling_protocol"]["seed_rule"] == "seed = seed_base + block_length"
    assert reader["resampling_protocol"]["block_lengths"] == [2, 5, 10]
    assert reader["source_denominator"] == {
        "unit": "C calibration input file", "planned": 3,
        "valid": 3, "failed": 0, "abstained": 0,
    }
    assert reader["configuration_denominator"] == {
        "unit": "paired saved gauge configuration", "planned": 12,
        "valid": 12, "failed": 0, "abstained": 0,
    }
    assert reader["draw_denominator"] == {
        "unit": "paired circular-bootstrap draw", "planned": 30_000,
        "valid": 30_000, "failed": 0, "abstained": 0,
    }
    assert reader["interval_summary"]["halfwidth"] <= 1e-12
    assert reader["text_flow"]
    assert result["source_verification"]["denominator"] == {
        "unit": "source_file", "planned": 8, "valid": 8,
        "failed": 0, "abstained": 0, "not_run": 0,
    }


def test_creutz_snapshot_reader_rejects_stale_summary_and_closed_split(tmp_path):
    envelope, receipt, hashes, snapshot, _measurement_bytes = _creutz_snapshot_fixture(
        tmp_path, summary_delta=0.01,
    )
    result = apply_wilson_creutz_snapshot_gate(
        _ready_state("operator", "numerical"), envelope, receipt,
        required_source_hashes=hashes, source_snapshot=snapshot,
    )
    assert result["status"] == "ABSTAIN_WILSON_C_LANE"
    assert result["creutz_snapshot_reader"]["status"] == "ABSTAIN_C_SUMMARY_MISMATCH"
    assert result["physical_score"] is None


def test_creutz_snapshot_reader_rejects_unmatched_bootstrap_seed(tmp_path):
    selected_seed = selected_protocol()["finite_T_Creutz"]["bootstrap_seed_base"]
    envelope, receipt, hashes, snapshot, _measurement_bytes = _creutz_snapshot_fixture(
        tmp_path, summary_seed_base=selected_seed + 1,
    )

    result = apply_wilson_creutz_snapshot_gate(
        _ready_state("operator", "numerical"), envelope, receipt,
        required_source_hashes=hashes, source_snapshot=snapshot,
    )

    assert result["status"] == "ABSTAIN_WILSON_C_LANE"
    assert result["creutz_snapshot_reader"]["status"] == "ABSTAIN_C_SUMMARY_RESAMPLING"
    assert result["physical_score"] is None

    closed_root = tmp_path / "closed"
    closed_root.mkdir()
    envelope, receipt, hashes, snapshot, _measurement_bytes = _creutz_snapshot_fixture(
        closed_root, split="C-calibration-validation-sealed",
    )
    state = _ready_state("operator", "numerical")
    result = apply_wilson_creutz_snapshot_gate(
        state, envelope, receipt,
        required_source_hashes=hashes, source_snapshot=snapshot,
    )
    assert result["status"] == "ABSTAIN_WILSON_C_LANE"
    assert state.status["statistical"] == "abstained"
    assert result["creutz_snapshot_reader"]["status"] == "ABSTAIN_C_CLOSED_SPLIT"
    assert result["physical_score"] is None


def test_creutz_lane_abstains_on_bad_precision_or_closed_split():
    envelope, receipt, summary, hashes = _creutz_fixture(halfwidth=0.021)
    state = _ready_state("operator", "numerical")
    result = apply_wilson_creutz_gate(
        state, envelope, receipt, summary, required_source_hashes=hashes,
        require_source_bytes=False,
    )
    assert result["status"] == "ABSTAIN_WILSON_C_LANE"
    assert state.status["statistical"] == "abstained"
    assert any("halfwidth" in reason for reason in result["reasons"])

    envelope, receipt, summary, hashes = _creutz_fixture(split="C-sealed-confirmation")
    state = _ready_state("operator", "numerical")
    result = apply_wilson_creutz_gate(
        state, envelope, receipt, summary, required_source_hashes=hashes,
        require_source_bytes=False,
    )
    assert result["status"] == "ABSTAIN_WILSON_C_LANE"
    assert any("closed" in reason for reason in result["reasons"])


def test_force_lane_checks_28_systematic_cells_and_force_unit_budget():
    envelopes, receipt, summary, routes = _force_fixture()
    state = _ready_state("operator", "reference", "numerical")
    result = apply_wilson_force_gate(
        state, envelopes, receipt, summary,
        required_source_hashes_by_envelope=routes,
        require_source_bytes=False,
    )
    assert result["status"] == "DRAFT_PENDING_TASK_VALIDATION"
    assert state.status["statistical"] == "draft"
    assert len(result["envelope_results"]) == 6
    assert result["physical_score"] is None


def test_force_lane_requires_explicit_source_bytes_by_default():
    envelopes, receipt, summary, routes = _force_fixture()
    state = _ready_state("operator", "reference", "numerical")
    result = apply_wilson_force_gate(
        state, envelopes, receipt, summary,
        required_source_hashes_by_envelope=routes,
    )
    assert result["status"] == "ABSTAIN_WILSON_F_LANE"
    assert state.status["statistical"] == "abstained"
    assert any("source-byte verification requires" in reason for reason in result["reasons"])
    assert result["source_verification"]["denominator"]["planned"] == sum(
        len(route_hashes) for route_hashes in routes.values()
    )
    assert result["source_verification"]["denominator"]["abstained"] == sum(
        len(route_hashes) for route_hashes in routes.values()
    )


def test_force_lane_verifies_all_route_source_bytes_atomically(tmp_path):
    envelopes, receipt, summary, routes = _force_fixture()
    common_bytes = b"synthetic shared Wilson F receipt"
    (tmp_path / "common.bin").write_bytes(common_bytes)
    common_hash = hashlib.sha256(common_bytes).hexdigest()
    protocol = selected_protocol()
    force = protocol["ground_state_discrete_static_force"]
    source_paths = {}

    for route, envelope in envelopes.items():
        map_bytes = json.dumps({
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
        }, sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(map_bytes).hexdigest() == force_map_sha256(route)
        map_name = f"{route}-map.json"
        (tmp_path / map_name).write_bytes(map_bytes)
        route_hashes = {
            name: (force_map_sha256(route) if name == "observable_map" else common_hash)
            for name in routes[route]
        }
        routes[route] = route_hashes
        envelope["chain"]["manifest_sha256"] = common_hash
        envelope["source_hashes"] = route_hashes
        envelope["review"]["receipt_sha256"] = common_hash
        source_paths[route] = {
            name: (map_name if name == "observable_map" else "common.bin")
            for name in route_hashes
        }

    summary["summary_sha256"] = common_hash
    state = _ready_state("operator", "reference", "numerical")
    result = apply_wilson_force_gate(
        state, envelopes, receipt, summary,
        required_source_hashes_by_envelope=routes,
        source_root=tmp_path,
        source_paths_by_envelope=source_paths,
    )
    assert result["status"] == "DRAFT_PENDING_TASK_VALIDATION"
    assert result["source_byte_identity_verified"] is True
    assert result["source_verification"]["denominator"] == {
        "unit": "source_file", "planned": 53, "valid": 53,
        "failed": 0, "abstained": 0, "not_run": 0,
    }

    snapshots = {
        route: capture_source_snapshot(tmp_path, routes[route], source_paths[route])
        for route in _F_ROUTES
    }
    assert json.loads(
        snapshots["force_interval"].read_bytes("observable_map")
    )["role"] == "force_interval"
    (tmp_path / "common.bin").write_bytes(b"changed after the force reader consumed its snapshots")
    result = apply_wilson_force_gate(
        _ready_state("operator", "reference", "numerical"),
        envelopes, receipt, summary,
        required_source_hashes_by_envelope=routes,
        source_snapshots_by_envelope=snapshots,
    )
    assert result["status"] == "DRAFT_PENDING_TASK_VALIDATION"
    assert result["source_verification"]["denominator"]["valid"] == 53

    (tmp_path / "common.bin").write_bytes(b"tampered Wilson F receipt")
    state = _ready_state("operator", "reference", "numerical")
    result = apply_wilson_force_gate(
        state, envelopes, receipt, summary,
        required_source_hashes_by_envelope=routes,
        source_root=tmp_path,
        source_paths_by_envelope=source_paths,
    )
    assert result["status"] == "ABSTAIN_SOURCE_BYTES"
    assert result["source_verification"]["denominator"] == {
        "unit": "source_file", "planned": 53, "valid": 6,
        "failed": 47, "abstained": 0, "not_run": 0,
    }


def test_force_snapshot_reader_recomputes_qhx_force_bounds_from_captured_bytes(tmp_path):
    envelopes, receipt, routes, snapshots, summary_bytes = _force_summary_snapshot_bundle(tmp_path)
    (tmp_path / "force-summary.json").write_bytes(b"changed after immutable capture")

    result = apply_wilson_force_snapshot_gate(
        _ready_state("operator", "reference", "numerical"),
        envelopes, receipt,
        required_source_hashes_by_envelope=routes,
        source_snapshots_by_envelope=snapshots,
    )

    assert result["status"] == "DRAFT_PENDING_TASK_VALIDATION"
    assert result["physical_score"] is None and result["certified"] is False
    reader = result["force_summary_reader"]
    assert reader["status"] == "F_SUMMARY_READ_NOT_CERTIFIED"
    assert reader["denominator"] == {
        "unit": "F force-summary route", "planned": 6,
        "valid": 6, "failed": 0, "abstained": 0,
    }
    assert reader["diagnostic"]["source_sha256"] == hashlib.sha256(summary_bytes).hexdigest()
    assert reader["diagnostic"]["analysis_source_sha256"]["core/numerics.py"] == hashlib.sha256(
        (Path(__file__).resolve().parents[1] / "core/numerics.py").read_bytes()
    ).hexdigest()
    assert reader["diagnostic"]["source_read_mode"] == "immutable_snapshot"
    assert reader["diagnostic"]["force_kernel_order"] == list(FORCE_Q_KERNEL_ORDER)
    assert reader["diagnostic"]["force_corner_order"] == list(FORCE_Q_CORNER_LABELS)
    components = reader["diagnostic"]["q_h_x_force_unit_bounds"]
    assert abs(components["excited_state_force"] - 4 * math.log1p(0.001)) < 1e-14
    assert abs(components["finite_T_thermal"] - 4 * math.log1p(0.001)) < 1e-14
    assert abs(components["mixed"] - 4 * math.log1p(0.0005)) < 1e-14
    assert reader["diagnostic"]["comparison_diagnostic"]["force_corner_count"] == 8
    assert reader["diagnostic"]["primary_force_kernel"] == "spatial_stout_rho0.10_n5"
    assert reader["diagnostic"]["force_budget_recalculation"] == {
        "component_gates_pass": True, "model_gate_pass": True,
        "statistical_gate_pass": True, "total_gate_pass": True,
    }
    assert reader["diagnostic"]["text_flow"]
    q_reader = result["spectral_q_reader"]
    assert q_reader["status"] == "Q_BOUND_DERIVED_NOT_CERTIFIED"
    assert q_reader["fit_variant_denominator"] == {
        "unit": "prespecified_GEVP_fit_variant", "planned": 36,
        "valid": 36, "failed": 0, "abstained": 0,
    }
    assert q_reader["selected_q_cell_denominator"]["valid"] == 8
    assert q_reader["force_qhx_corner_order"] == list(FORCE_Q_KERNEL_ORDER)
    assert q_reader["force_qhx_corner_labels"] == list(FORCE_Q_CORNER_LABELS)
    assert result["spectral_q_reconciliation"] == {
        "status": "Q_FORCE_CORNERS_RECONCILED",
        "kernel_order": list(FORCE_Q_KERNEL_ORDER),
        "corner_denominator": {
            "unit": "selected_spectral_q_force_corner", "planned": 8,
            "valid": 8, "failed": 0, "abstained": 0,
        },
        "reasons": [], "certified": False, "physical_score": None,
    }
    assert result["source_verification"]["denominator"] == {
        "unit": "source_file", "planned": 53, "valid": 53,
        "failed": 0, "abstained": 0, "not_run": 0,
    }


def test_force_snapshot_gate_rejects_q_corners_below_same_snapshot_spectral_bound(tmp_path):
    summary = {
        "schema": "wilson-f-force-summary-input-v3",
        "units": "dimensionless lattice force a_s*a_t*F",
        "stout_force_estimate": 1.03,
        "stout_force_interval_lower": 0.99,
        "stout_force_interval_upper": 1.07,
        "positive_force_scale_lower": 0.95,
        "statistical_halfwidth": 0.04,
        "model_truncation_bound": 0.005,
        "invalid_force_interval_draws": 0,
        "invalid_force_scale_draws": 0,
        "force_kernel_order": list(FORCE_Q_KERNEL_ORDER),
        "force_corner_order": list(FORCE_Q_CORNER_LABELS),
        "qhx_corner_bounds": {
            "q": [[0.0] * 4 for _ in range(2)],
            "h": [[0.001] * 4 for _ in range(2)],
            "x": [[0.0005] * 4 for _ in range(2)],
        },
    }
    envelopes, receipt, routes, snapshots, _summary_bytes = _force_summary_snapshot_bundle(
        tmp_path, summary_override=summary,
    )
    state = _ready_state("operator", "reference", "numerical")
    result = apply_wilson_force_snapshot_gate(
        state, envelopes, receipt,
        required_source_hashes_by_envelope=routes,
        source_snapshots_by_envelope=snapshots,
    )

    assert result["status"] == "ABSTAIN_WILSON_F_LANE"
    assert state.status["statistical"] == "abstained"
    assert result["spectral_q_reader"]["status"] == "Q_BOUND_DERIVED_NOT_CERTIFIED"
    reconciliation = result["spectral_q_reconciliation"]
    assert reconciliation["status"] == "ABSTAIN_Q_BELOW_SPECTRAL_BOUND"
    assert reconciliation["corner_denominator"] == {
        "unit": "selected_spectral_q_force_corner", "planned": 8,
        "valid": 0, "failed": 8, "abstained": 0,
    }
    assert any("below the captured spectral q upper bound" in reason
               for reason in reconciliation["reasons"])
    assert result["physical_score"] is None and result["certified"] is False


def test_force_snapshot_reader_rejects_reversed_kernel_or_corner_order(tmp_path):
    base_summary = {
        "schema": "wilson-f-force-summary-input-v3",
        "units": "dimensionless lattice force a_s*a_t*F",
        "stout_force_estimate": 1.03,
        "stout_force_interval_lower": 0.99,
        "stout_force_interval_upper": 1.07,
        "positive_force_scale_lower": 0.95,
        "statistical_halfwidth": 0.04,
        "model_truncation_bound": 0.005,
        "invalid_force_interval_draws": 0,
        "invalid_force_scale_draws": 0,
        "force_kernel_order": list(FORCE_Q_KERNEL_ORDER),
        "force_corner_order": list(FORCE_Q_CORNER_LABELS),
        "qhx_corner_bounds": {
            "q": [[0.001, 0.002, 0.003, 0.004] for _ in range(2)],
            "h": [[0.001] * 4 for _ in range(2)],
            "x": [[0.0005] * 4 for _ in range(2)],
        },
    }
    for index, field in enumerate(("force_kernel_order", "force_corner_order")):
        case_dir = tmp_path / str(index)
        case_dir.mkdir()
        summary = dict(base_summary)
        summary[field] = list(reversed(base_summary[field]))
        envelopes, receipt, routes, snapshots, _ = _force_summary_snapshot_bundle(
            case_dir, summary_override=summary,
        )
        state = _ready_state("operator", "reference", "numerical")
        result = apply_wilson_force_snapshot_gate(
            state, envelopes, receipt,
            required_source_hashes_by_envelope=routes,
            source_snapshots_by_envelope=snapshots,
        )

        assert result["force_summary_reader"]["status"] == "ABSTAIN_F_SUMMARY_CONTENT"
        assert field in result["force_summary_reader"]["reason"]
        assert result["force_summary_reader"]["denominator"] == {
            "unit": "F force-summary route", "planned": 6,
            "valid": 6, "failed": 0, "abstained": 0,
        }
        assert result["force_summary_reader"]["bundle_denominator"] == {
            "unit": "shared force-summary bundle", "planned": 1,
            "valid": 0, "failed": 1, "abstained": 0,
        }
        assert result["spectral_q_reconciliation"]["corner_denominator"] == {
            "unit": "selected_spectral_q_force_corner", "planned": 8,
            "valid": 0, "failed": 0, "abstained": 8,
        }
        assert state.status["statistical"] == "abstained"


@pytest.mark.parametrize(
    ("field", "value", "reason_fragment"),
    [
        ("stout_force_interval_lower", 0.0, "stout_force_interval_lower"),
        ("positive_force_scale_lower", 1.0, "must not exceed"),
        ("statistical_halfwidth", 0.01, "maximum endpoint deviation"),
    ],
)
def test_force_summary_v3_rejects_nonpositive_or_inconsistent_stout_interval(
    tmp_path, field, value, reason_fragment,
):
    base = {
        "schema": "wilson-f-force-summary-input-v3",
        "units": "dimensionless lattice force a_s*a_t*F",
        "stout_force_estimate": 1.03,
        "stout_force_interval_lower": 0.99,
        "stout_force_interval_upper": 1.07,
        "positive_force_scale_lower": 0.95,
        "statistical_halfwidth": 0.04,
        "model_truncation_bound": 0.005,
        "invalid_force_interval_draws": 0,
        "invalid_force_scale_draws": 0,
        "force_kernel_order": list(FORCE_Q_KERNEL_ORDER),
        "force_corner_order": list(FORCE_Q_CORNER_LABELS),
        "qhx_corner_bounds": {
            "q": [[0.001] * 4 for _ in range(2)],
            "h": [[0.001] * 4 for _ in range(2)],
            "x": [[0.0005] * 4 for _ in range(2)],
        },
    }
    base[field] = value
    case_dir = tmp_path / field
    case_dir.mkdir()
    envelopes, receipt, routes, snapshots, _ = _force_summary_snapshot_bundle(
        case_dir, summary_override=base,
    )
    state = _ready_state("operator", "reference", "numerical")
    result = apply_wilson_force_snapshot_gate(
        state, envelopes, receipt,
        required_source_hashes_by_envelope=routes,
        source_snapshots_by_envelope=snapshots,
    )
    reader = result["force_summary_reader"]
    assert reader["status"] == "ABSTAIN_F_SUMMARY_CONTENT"
    assert reason_fragment in reader["reason"]
    assert state.status["statistical"] == "abstained"
    assert result["physical_score"] is None and result["certified"] is False


def test_force_snapshot_reader_rejects_mismatched_summary_routes_and_bad_qhx(tmp_path):
    envelopes, receipt, routes, snapshots, _summary_bytes = _force_summary_snapshot_bundle(tmp_path)
    routes["q_excited"]["force_summary"] = H[50]
    state = _ready_state("operator", "reference", "numerical")
    result = apply_wilson_force_snapshot_gate(
        state, envelopes, receipt,
        required_source_hashes_by_envelope=routes,
        source_snapshots_by_envelope=snapshots,
    )
    assert result["status"] == "ABSTAIN_WILSON_F_LANE"
    assert state.status["statistical"] == "abstained"
    assert result["force_summary_reader"]["status"] == "ABSTAIN_F_SUMMARY_SOURCE"
    assert result["physical_score"] is None

    bad_root = tmp_path / "bad"
    bad_root.mkdir()
    bad_summary = {
        "schema": "wilson-f-force-summary-input-v3",
        "units": "dimensionless lattice force a_s*a_t*F",
        "stout_force_estimate": 1.03,
        "stout_force_interval_lower": 0.99,
        "stout_force_interval_upper": 1.07,
        "positive_force_scale_lower": 0.95,
        "statistical_halfwidth": 0.04,
        "model_truncation_bound": 0.005,
        "invalid_force_interval_draws": 0,
        "invalid_force_scale_draws": 0,
        "force_kernel_order": list(FORCE_Q_KERNEL_ORDER),
        "force_corner_order": list(FORCE_Q_CORNER_LABELS),
        "qhx_corner_bounds": {"q": None, "h": [[0.001] * 4],
                              "x": [[0.0005] * 4]},
    }
    envelopes, receipt, routes, snapshots, _summary_bytes = _force_summary_snapshot_bundle(
        bad_root, summary_override=bad_summary,
    )
    state = _ready_state("operator", "reference", "numerical")
    result = apply_wilson_force_snapshot_gate(
        state, envelopes, receipt,
        required_source_hashes_by_envelope=routes,
        source_snapshots_by_envelope=snapshots,
    )
    assert result["status"] == "ABSTAIN_WILSON_F_LANE"
    assert state.status["statistical"] == "abstained"
    assert result["force_summary_reader"]["status"] == "ABSTAIN_F_SUMMARY_CONTENT"
    assert "q/h/x corner inputs are invalid" in result["force_summary_reader"]["reason"]
    assert result["physical_score"] is None


def test_force_lane_abstains_without_positive_force_scale_or_with_overbudget_bounds():
    envelopes, receipt, summary, routes = _force_fixture()
    summary["positive_force_scale_lower"] = 0.0
    state = _ready_state("operator", "reference", "numerical")
    result = apply_wilson_force_gate(
        state, envelopes, receipt, summary,
        required_source_hashes_by_envelope=routes,
        require_source_bytes=False,
    )
    assert result["status"] == "ABSTAIN_WILSON_F_LANE"
    assert any("positive simultaneous" in reason for reason in result["reasons"])

    envelopes, receipt, summary, routes = _force_fixture()
    summary["force_component_bounds"]["finite_T_thermal"] = 0.02
    summary["systematic_total_bound"] = 0.04
    summary["total_linear_bound"] = 0.07
    state = _ready_state("operator", "reference", "numerical")
    result = apply_wilson_force_gate(
        state, envelopes, receipt, summary,
        required_source_hashes_by_envelope=routes,
        require_source_bytes=False,
    )
    assert result["status"] == "ABSTAIN_WILSON_F_LANE"
    assert any("finite_T_thermal" in reason for reason in result["reasons"])


def test_force_lane_requires_complete_GEVP_thermal_denominators_and_separate_runstate():
    envelopes, receipt, summary, routes = _force_fixture()
    bad = run_receipt(
        receipt["scenario"],
        {
            **receipt["denominators"],
            "gevp_fit_variants": denominator("3x3 GEVP fit variant", planned=36, valid=35, failed=0, abstained=0),
        },
        artifact_hashes=receipt["artifact_hashes"], text_flow=receipt["text_flow"],
        reason_rows=[{
            "unit": "gevp_fit_variants", "state": "not_run", "count": 1,
            "reason": "one planned GEVP variant is missing",
        }],
    )
    state = _ready_state("operator", "reference", "numerical")
    result = apply_wilson_force_gate(
        state, envelopes, bad, summary,
        required_source_hashes_by_envelope=routes,
        require_source_bytes=False,
    )
    assert result["status"] == "ABSTAIN_WILSON_F_LANE"
    assert any("gevp_fit_variants" in reason for reason in result["reasons"])

    envelopes, receipt, summary, routes = _force_fixture()
    state = _ready_state("operator", "reference", "numerical")
    result = apply_wilson_force_gate(
        state, envelopes, receipt, summary,
        required_source_hashes_by_envelope=routes,
        require_source_bytes=False,
    )
    assert result["status"] == "DRAFT_PENDING_TASK_VALIDATION"
