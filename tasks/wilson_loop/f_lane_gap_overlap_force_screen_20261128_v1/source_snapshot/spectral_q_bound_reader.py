"""Derive positive-spectrum excited-state q envelopes from one frozen fit receipt.

This is a CPU-only reducer for already-computed one-sided overlap-ratio and
energy-gap bounds. It does not solve a GEVP, fit correlators, create confidence
intervals, or establish that the positive spectral representation applies to
the production observable. Successful parsing is explicitly uncertified.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from core.source_integrity import SourceByteSnapshot
from .lane_contract import q_overlap_contract, selected_protocol


SCHEMA = "wilson-f-bare-basis-overlap-support-v2"
RESULT_SCHEMA = "wilson-f-spectral-q-bound-read-v2"
UNITS = "squared overlap ratios dimensionless; energy gaps in a_t*DeltaE"
KERNELS = (
    "identity",
    "ape_alpha0.4_n5",
    "spatial_stout_rho0.10_n5",
)
KERNEL_TO_BASIS_OPERATOR = {
    "identity": "identity",
    "ape_alpha0.4_n5": "APE",
    "spatial_stout_rho0.10_n5": "stout",
}
OVERLAP_DEFINITION_ID = q_overlap_contract()["definition_id"]
SELECTED_FORCE_KERNELS = ("identity", "spatial_stout_rho0.10_n5")
FORCE_Q_KERNEL_ORDER = ("spatial_stout_rho0.10_n5", "identity")
R_VALUES = (2, 3)
TMIN_VALUES = (3, 4, 5)
STATE_COUNTS = (2, 3)
FIT_WINDOW = (3, 7)
CONTAMINATION_TIMES = (5, 6)
FORCE_Q_CORNER_ORDER = ((2, 5), (2, 6), (3, 5), (3, 6))
FORCE_Q_CORNER_LABELS = tuple(f"R{radius}_T{time_extent}"
                              for radius, time_extent in FORCE_Q_CORNER_ORDER)
ENDPOINT_SLOTS_PER_Q_CELL = 24
Q_CELL_COUNT = len(SELECTED_FORCE_KERNELS) * len(R_VALUES) * len(CONTAMINATION_TIMES)
EXPECTED_FIT_VARIANTS = len(KERNELS) * len(R_VALUES) * len(TMIN_VALUES) * len(STATE_COUNTS)
_CLOSED_SPLIT_MARKERS = ("gap", "validation", "holdout", "sealed", "confirmation")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _is_int(value: object, expected: int | None = None) -> bool:
    return (type(value) is int and (expected is None or value == expected))


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite nonnegative number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite nonnegative number") from exc
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return converted


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _q_upper(ratios: list[float], gaps: list[float], time_extent: int) -> float:
    """Evaluate sum(r_i exp(-gap_i T)) without overflowing intermediate terms."""
    log_terms = [
        math.log(ratio) - gap * time_extent
        for ratio, gap in zip(ratios, gaps) if ratio > 0
    ]
    if not log_terms:
        return 0.0
    largest = max(log_terms)
    if largest == -math.inf:
        return 0.0
    log_total = largest + math.log(math.fsum(math.exp(term - largest) for term in log_terms))
    if log_total > math.log(sys.float_info.max):
        raise OverflowError("derived q upper bound is outside finite float range")
    return math.exp(log_total)


def _abstain(status: str, reason: str, *, source_status: str = "valid",
             variants_valid: int = 0, variants_failed: int = 0,
             variants_abstained: int = EXPECTED_FIT_VARIANTS) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "status": status,
        "reason": reason,
        "fit_variant_denominator": {
            "unit": "prespecified_GEVP_fit_variant",
            "planned": EXPECTED_FIT_VARIANTS,
            "valid": variants_valid,
            "failed": variants_failed,
            "abstained": variants_abstained,
        },
        "source_denominator": {
            "unit": "immutable_snapshot_source_file", "planned": 1,
            "valid": int(source_status == "valid"),
            "failed": int(source_status == "failed"),
            "abstained": int(source_status == "abstained"),
        },
        "selected_q_cell_denominator": {
            "unit": "selected_kernel_R_time_q_cell", "planned": Q_CELL_COUNT,
            "valid": 0, "failed": 0, "abstained": Q_CELL_COUNT,
        },
        "physical_score": None,
        "certified": False,
    }


def _expected_gevp_metadata() -> dict[str, Any]:
    return {
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
    }


def _selected_cell_alpha() -> float:
    # Keep the family allocation single-sourced in the selected task contract.
    protocol = selected_protocol()
    force = protocol["ground_state_discrete_static_force"]
    family = protocol["familywise_alpha_allocation"]
    budget = force["relative_error_budget"]
    n_cells = budget["q_h_x_cells"] + budget["model_truncation_cells"]
    return family["alpha_force_systematics"] / n_cells


def _validate_top_level(document: object, split_id: str) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("root must be a JSON object")
    required = {
        "schema", "classification", "split_id", "units", "overlap_definition_id",
        "positive_forward_spectral_sum_assumption", "gevp",
        "confidence_allocation", "fit_variants",
    }
    if set(document) != required:
        raise ValueError(f"top-level keys must be exactly {sorted(required)}")
    if document["schema"] != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA!r}")
    if document["classification"] != "fit_receipt_reducer_input_not_raw_measurement":
        raise ValueError("classification must identify a derived fit-receipt input")
    if document["units"] != UNITS:
        raise ValueError(f"units must be {UNITS!r}")
    if document["overlap_definition_id"] != OVERLAP_DEFINITION_ID:
        raise ValueError("overlap definition differs from the selected bare-basis q contract")
    if document["split_id"] != split_id:
        raise ValueError("snapshot split_id differs from the caller's frozen split")
    lowered = split_id.lower()
    if any(marker in lowered for marker in _CLOSED_SPLIT_MARKERS):
        raise ValueError("closed gap/validation/holdout/sealed/confirmation split rejected")
    if "calibration" not in lowered:
        raise ValueError("an explicitly named calibration split is required")
    if document["positive_forward_spectral_sum_assumption"] is not True:
        raise ValueError("positive forward spectral-sum assumption must be explicit")
    gevp = document["gevp"]
    expected_gevp = _expected_gevp_metadata()
    if (not isinstance(gevp, dict) or set(gevp) != set(expected_gevp)
            or any(not _is_int(gevp.get(name), expected_gevp[name])
                   for name in ("dimension", "t0", "td"))
            or any(not isinstance(gevp.get(name), list)
                   or any(type(item) is not int for item in gevp[name])
                   for name in ("R_values", "state_counts", "tmin_checks", "contamination_times"))
            or any(not isinstance(gevp.get(name), list)
                   for name in ("basis_kernel_ids", "fit_window"))
            or any(not isinstance(item, str) for item in gevp.get("basis_kernel_ids", []))
            or any(type(item) is not int for item in gevp.get("fit_window", []))
            or gevp != expected_gevp):
        raise ValueError("GEVP, operator basis, fit window, or contamination times differ from the selected draft")

    alpha = _selected_cell_alpha()
    expected_confidence = {
        "method": "Bonferroni_one_sided_marginal_bounds_within_each_q_cell",
        "systematic_cell_alpha": alpha,
        "endpoint_slots_per_q_cell": ENDPOINT_SLOTS_PER_Q_CELL,
        "endpoint_alpha": alpha / ENDPOINT_SLOTS_PER_Q_CELL,
        "unused_endpoint_slots_reassigned": False,
    }
    confidence = document["confidence_allocation"]
    cell_alpha = _finite_float(confidence.get("systematic_cell_alpha")) if isinstance(confidence, dict) else None
    endpoint_alpha = _finite_float(confidence.get("endpoint_alpha")) if isinstance(confidence, dict) else None
    if (not isinstance(confidence, dict)
            or set(confidence) != set(expected_confidence)
            or confidence["method"] != expected_confidence["method"]
            or confidence["endpoint_slots_per_q_cell"] != ENDPOINT_SLOTS_PER_Q_CELL
            or confidence["unused_endpoint_slots_reassigned"] is not False
            or cell_alpha is None
            or not math.isclose(cell_alpha, alpha,
                                rel_tol=0, abs_tol=1e-16)
            or endpoint_alpha is None
            or not math.isclose(endpoint_alpha,
                                alpha / ENDPOINT_SLOTS_PER_Q_CELL,
                                rel_tol=0, abs_tol=1e-18)):
        raise ValueError("confidence allocation differs from the selected nested Bonferroni draft")
    return document


def _validate_variants(rows: object) -> tuple[dict[tuple[str, int, int, int], dict], int, int, int, int]:
    if not isinstance(rows, list):
        raise ValueError("fit_variants must be a list")
    by_key = {}
    failed_keys = set()
    seen_ids = set()
    extra_rows = 0
    expected_keys = {
        (k, r, start, count)
        for k in KERNELS for r in R_VALUES
        for start in TMIN_VALUES for count in STATE_COUNTS
    }
    required = {
        "variant_id", "kernel_id", "R", "n_states", "tmin", "fit_window",
        "basis_operator_id", "overlap_ratio_upper", "gap_lower",
    }
    for row in rows:
        key = None
        try:
            if not isinstance(row, dict):
                raise ValueError("fit variant has unexpected or missing fields")
            possible = (row.get("kernel_id"), row.get("R"),
                        row.get("tmin"), row.get("n_states"))
            if (possible[0] in KERNELS and _is_int(possible[1])
                    and _is_int(possible[2]) and _is_int(possible[3])):
                key = (possible[0], possible[1], possible[2], possible[3])
            if set(row) != required:
                raise ValueError("fit variant has unexpected or missing fields")
            variant_id = row["variant_id"]
            kernel = row["kernel_id"]
            radius, n_states, tmin = row["R"], row["n_states"], row["tmin"]
            if not isinstance(variant_id, str) or not variant_id.strip():
                raise ValueError("variant_id must be nonempty")
            if kernel not in KERNELS:
                raise ValueError("unknown GEVP basis operator")
            if row["basis_operator_id"] != KERNEL_TO_BASIS_OPERATOR[kernel]:
                raise ValueError("fit overlap object does not match the kernel's bare basis operator")
            if not _is_int(radius) or radius not in R_VALUES:
                raise ValueError("R must be one of the selected integer radii")
            if not _is_int(n_states) or n_states not in STATE_COUNTS:
                raise ValueError("n_states must be 2 or 3")
            if not _is_int(tmin) or tmin not in TMIN_VALUES:
                raise ValueError("tmin must be one of the selected fit starts")
            key = (kernel, radius, tmin, n_states)
            if key not in expected_keys:
                raise ValueError("fit variant key is outside the selected Cartesian design")
            if row["fit_window"] != [tmin, FIT_WINDOW[1]]:
                raise ValueError("fit window must be [tmin,7]")
            ratios = row["overlap_ratio_upper"]
            gaps = row["gap_lower"]
            n_excited = n_states - 1
            if (not isinstance(ratios, list) or not isinstance(gaps, list)
                    or len(ratios) != n_excited or len(gaps) != n_excited):
                raise ValueError("one-state/two-state excited ratio and gap arrays must match n_states")
            normalized_ratios = [
                _finite_nonnegative(value, f"{variant_id}.overlap_ratio_upper[{i}]")
                for i, value in enumerate(ratios)
            ]
            normalized_gaps = [
                _finite_nonnegative(value, f"{variant_id}.gap_lower[{i}]")
                for i, value in enumerate(gaps)
            ]
            if key in by_key or key in failed_keys or variant_id in seen_ids:
                raise ValueError("duplicate fit variant key or variant_id")
            seen_ids.add(variant_id)
            by_key[key] = {
                **row,
                "overlap_ratio_upper": normalized_ratios,
                "gap_lower": normalized_gaps,
            }
        except ValueError:
            if key in by_key:
                del by_key[key]
                failed_keys.add(key)
            elif key is not None:
                failed_keys.add(key)
            else:
                extra_rows += 1
    missing_keys = expected_keys - set(by_key) - failed_keys
    extra_rows += max(0, len(rows) - EXPECTED_FIT_VARIANTS)
    return by_key, len(by_key), len(failed_keys), len(missing_keys), extra_rows


def read_spectral_q_bounds(
    snapshot: SourceByteSnapshot,
    *,
    source_name: str,
    split_id: str,
    expected_source_sha256: str,
) -> dict[str, Any]:
    """Recompute q(T) bounds from one captured 36-variant joint-matrix fit receipt.

    Each kernel's q uses the squared excited/ground overlap ratios for its own
    bare basis operator, jointly fitted from the full 3x3 cross-correlator
    matrix. The GEVP composite operator is a distinct object and is never
    copied into these kernel-indexed cells. APE is retained as a descriptive
    basis diagnostic, not a force-systematic q cell. The parser verifies the
    selected confidence-allocation metadata; it cannot establish coverage of
    the upstream one-sided intervals.
    """
    if not isinstance(snapshot, SourceByteSnapshot):
        return _abstain("ABSTAIN_F_SPECTRUM_SNAPSHOT", "snapshot has the wrong type",
                        source_status="abstained")
    if not isinstance(split_id, str) or not split_id.strip():
        return _abstain("ABSTAIN_F_SPECTRUM_SPLIT", "split_id is required",
                        source_status="abstained")
    if not isinstance(source_name, str) or not source_name.strip():
        return _abstain("ABSTAIN_F_SPECTRUM_HASH_MAP", "source_name is missing",
                        source_status="abstained")
    expected = snapshot.expected_hashes.get(source_name)
    if (not isinstance(source_name, str) or not source_name.strip()
            or not isinstance(expected_source_sha256, str)
            or expected != expected_source_sha256):
        return _abstain("ABSTAIN_F_SPECTRUM_HASH_MAP", "expected source hash map is incomplete or inconsistent",
                        source_status="abstained")
    try:
        payload = snapshot.read_bytes(source_name)
    except (TypeError, ValueError) as exc:
        entry = snapshot.capture_entries.get(source_name, {})
        status = "failed" if entry.get("status") == "failed" else "abstained"
        return _abstain("ABSTAIN_F_SPECTRUM_SOURCE", str(exc), source_status=status)
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_source_sha256:
        return _abstain("ABSTAIN_F_SPECTRUM_SOURCE_HASH", "captured fit receipt differs from expected SHA-256",
                        source_status="failed")

    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
        _validate_top_level(document, split_id)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return _abstain("ABSTAIN_F_SPECTRUM_CONTENT", str(exc), source_status="failed")

    variants, valid, failed, missing, extra_rows = _validate_variants(document["fit_variants"])
    denominator = {
        "unit": "prespecified_GEVP_fit_variant", "planned": EXPECTED_FIT_VARIANTS,
        "valid": valid, "failed": failed, "abstained": missing,
    }
    if valid != EXPECTED_FIT_VARIANTS or failed or missing or extra_rows:
        result = _abstain(
            "ABSTAIN_F_SPECTRUM_VARIANTS",
            "all 36 selected kernel/R/tmin/state-count fit variants are required",
            source_status="valid", variants_valid=valid, variants_failed=failed,
            variants_abstained=missing,
        )
        result["fit_variant_denominator"] = denominator
        result["unexpected_variant_rows"] = extra_rows
        result["source_sha256"] = observed
        return result

    q_rows = []
    selected_valid = descriptive_valid = 0
    selected_lookup = {}
    for kernel in KERNELS:
        for radius in R_VALUES:
            cell_variants = [
                variants[(kernel, radius, tmin, n_states)]
                for tmin in TMIN_VALUES for n_states in STATE_COUNTS
            ]
            for time_extent in CONTAMINATION_TIMES:
                candidates = []
                for row in cell_variants:
                    try:
                        value = _q_upper(
                            row["overlap_ratio_upper"], row["gap_lower"], time_extent
                        )
                    except OverflowError as exc:
                        selected = kernel in SELECTED_FORCE_KERNELS
                        result = _abstain(
                            "ABSTAIN_F_SPECTRUM_Q_NUMERIC", str(exc),
                            source_status="valid", variants_valid=valid,
                            variants_failed=failed, variants_abstained=missing,
                        )
                        result["fit_variant_denominator"] = denominator
                        result["selected_q_cell_denominator"] = {
                            "unit": "selected_kernel_R_time_q_cell",
                            "planned": Q_CELL_COUNT,
                            "valid": 0,
                            "failed": int(selected),
                            "abstained": Q_CELL_COUNT - int(selected),
                        }
                        result["descriptive_ape_cell_denominator"] = {
                            "unit": "descriptive_APE_R_time_cell",
                            "planned": len(R_VALUES) * len(CONTAMINATION_TIMES),
                            "valid": 0, "failed": int(not selected),
                            "abstained": len(R_VALUES) * len(CONTAMINATION_TIMES) - int(not selected),
                        }
                        result["failed_q_cell"] = {
                            "kernel_id": kernel, "R": radius, "T": time_extent,
                            "variant_id": row["variant_id"],
                        }
                        result["source_sha256"] = observed
                        return result
                    candidates.append((value, row["variant_id"], row["tmin"], row["n_states"]))
                maximum = max(candidates, key=lambda item: item[0])
                q_rows.append({
                    "kernel_id": kernel,
                    "basis_operator_id": KERNEL_TO_BASIS_OPERATOR[kernel],
                    "overlap_definition_id": OVERLAP_DEFINITION_ID,
                    "R": radius,
                    "T": time_extent,
                    "q_upper": maximum[0],
                    "dominating_variant_id": maximum[1],
                    "dominating_tmin": maximum[2],
                    "dominating_state_count": maximum[3],
                    "variants_in_maximum": len(candidates),
                    "confidence_alpha_per_endpoint": _selected_cell_alpha() / ENDPOINT_SLOTS_PER_Q_CELL,
                    "use": "force_systematic_q_cell" if kernel in SELECTED_FORCE_KERNELS else "descriptive_basis_only",
                })
                if kernel in SELECTED_FORCE_KERNELS:
                    selected_valid += 1
                    selected_lookup[(kernel, radius, time_extent)] = maximum[0]
                else:
                    descriptive_valid += 1

    force_order = FORCE_Q_KERNEL_ORDER
    force_qhx_corners = {
        kernel: [
            selected_lookup[(kernel, radius, time_extent)]
            for radius, time_extent in FORCE_Q_CORNER_ORDER
        ]
        for kernel in force_order
    }

    return {
        "schema": RESULT_SCHEMA,
        "status": "Q_BOUND_DERIVED_NOT_CERTIFIED",
        "classification": "derived_from_synthetic_or_external_fit_receipt; not raw measurement",
        "split_id": split_id,
        "units": UNITS,
        "overlap_definition_id": OVERLAP_DEFINITION_ID,
        "source_read_mode": "immutable_snapshot",
        "source_sha256": observed,
        "fit_variant_denominator": denominator,
        "selected_q_cell_denominator": {
            "unit": "selected_kernel_R_time_q_cell", "planned": Q_CELL_COUNT,
            "valid": selected_valid, "failed": 0, "abstained": 0,
        },
        "descriptive_ape_cell_denominator": {
            "unit": "descriptive_APE_R_time_cell", "planned": len(R_VALUES) * len(CONTAMINATION_TIMES),
            "valid": descriptive_valid, "failed": 0, "abstained": 0,
        },
        "confidence_allocation": document["confidence_allocation"],
        "q_corner_bounds": q_rows,
        "force_qhx_corner_order": list(force_order),
        "force_qhx_corner_labels": list(FORCE_Q_CORNER_LABELS),
        "force_q_corner_bounds": force_qhx_corners,
        "physical_score": None,
        "certified": False,
        "text_flow": [
            "one hash-matched calibration fit-support JSON captured as immutable bytes",
            "strictly validate the 36 selected kernel-to-bare-operator/R/tmin/state-count variants and nested alpha rule",
            "derive each kernel q from that bare operator's squared overlap ratios; keep GEVP-composite q separate",
            "for each kernel/R/T=5,6 compute sum(ratio_upper*exp(-gap_lower*T))",
            "take the maximum across all six time-window/state-count variants",
            "return eight identity/stout force q cells plus four descriptive APE cells",
            "leave force gate abstained until the fit receipt, h/x bounds, force draws, and reviews are independently verified",
        ],
        "limitations": [
            "does not read raw correlation matrices, solve the GEVP, or fit the joint state-overlap matrix",
            "does not fit energies/overlaps or construct one-sided confidence bounds",
            "cannot authenticate the positive spectral representation or upstream interval coverage",
            "does not derive thermal h, mixed x, model truncation, force intervals, or positive reference force",
            "successful parsing is not physical evidence certification",
        ],
        "analysis_source_sha256": {
            Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "lane_contract.py": hashlib.sha256(Path(__file__).with_name("lane_contract.py").read_bytes()).hexdigest(),
            "lane_contract_v2.json": hashlib.sha256(Path(__file__).with_name("lane_contract_v2.json").read_bytes()).hexdigest(),
        },
    }
