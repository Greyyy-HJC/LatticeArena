import hashlib
import json
import math

import pytest

from core.source_integrity import capture_source_snapshot
from tasks.wilson_loop.spectral_q_bound_reader import (
    CONTAMINATION_TIMES,
    ENDPOINT_SLOTS_PER_Q_CELL,
    FIT_WINDOW,
    KERNELS,
    R_VALUES,
    SCHEMA,
    STATE_COUNTS,
    TMIN_VALUES,
    UNITS,
    read_spectral_q_bounds,
)


def _fit_support(*, split_id="F-calibration"):
    alpha_cell = 0.0125 / 28
    variants = []
    for kernel in KERNELS:
        for radius in R_VALUES:
            for tmin in TMIN_VALUES:
                for n_states in STATE_COUNTS:
                    ratios = [0.10 + 0.01 * tmin + 0.02 * (n_states - 2)]
                    gaps = [0.20 - 0.01 * tmin]
                    if n_states == 3:
                        ratios.append(0.05)
                        gaps.append(0.10)
                    variants.append({
                        "variant_id": f"{kernel}-R{radius}-tmin{tmin}-n{n_states}",
                        "kernel_id": kernel,
                        "basis_operator_id": {
                            "identity": "identity",
                            "ape_alpha0.4_n5": "APE",
                            "spatial_stout_rho0.10_n5": "stout",
                        }[kernel],
                        "R": radius,
                        "n_states": n_states,
                        "tmin": tmin,
                        "fit_window": [tmin, FIT_WINDOW[1]],
                        "overlap_ratio_upper": ratios,
                        "gap_lower": gaps,
                    })
    return {
        "schema": SCHEMA,
        "classification": "fit_receipt_reducer_input_not_raw_measurement",
        "split_id": split_id,
        "units": UNITS,
        "overlap_definition_id": "bare_basis_squared_overlap_ratio_v1",
        "positive_forward_spectral_sum_assumption": True,
        "gevp": {
            "dimension": 3,
            "basis_kernel_ids": list(KERNELS),
            "basis_operator_ids": [
                {"identity": "identity", "ape_alpha0.4_n5": "APE",
                 "spatial_stout_rho0.10_n5": "stout"}[kernel]
                for kernel in KERNELS
            ],
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
            "systematic_cell_alpha": alpha_cell,
            "endpoint_slots_per_q_cell": ENDPOINT_SLOTS_PER_Q_CELL,
            "endpoint_alpha": alpha_cell / ENDPOINT_SLOTS_PER_Q_CELL,
            "unused_endpoint_slots_reassigned": False,
        },
        "fit_variants": variants,
    }


def _read(tmp_path, document, *, split_id="F-calibration"):
    payload = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (tmp_path / "fit-support.json").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    snapshot = capture_source_snapshot(
        tmp_path, {"fit_support": digest}, {"fit_support": "fit-support.json"}
    )
    return read_spectral_q_bounds(
        snapshot, source_name="fit_support", split_id=split_id,
        expected_source_sha256=digest,
    )


def test_derives_q_from_all_fit_variants_and_keeps_lanes_uncertified(tmp_path):
    result = _read(tmp_path, _fit_support())
    assert result["status"] == "Q_BOUND_DERIVED_NOT_CERTIFIED"
    assert result["fit_variant_denominator"] == {
        "unit": "prespecified_GEVP_fit_variant", "planned": 36,
        "valid": 36, "failed": 0, "abstained": 0,
    }
    assert result["selected_q_cell_denominator"] == {
        "unit": "selected_kernel_R_time_q_cell", "planned": 8,
        "valid": 8, "failed": 0, "abstained": 0,
    }
    assert result["descriptive_ape_cell_denominator"]["planned"] == 4
    assert result["descriptive_ape_cell_denominator"]["valid"] == 4
    assert result["overlap_definition_id"] == "bare_basis_squared_overlap_ratio_v1"
    assert result["physical_score"] is None and not result["certified"]
    stout_r2_t5 = next(
        row for row in result["q_corner_bounds"]
        if row["kernel_id"] == "spatial_stout_rho0.10_n5"
        and row["R"] == 2 and row["T"] == 5
    )
    expected = 0.17 * math.exp(-0.15 * 5) + 0.05 * math.exp(-0.1 * 5)
    assert stout_r2_t5["dominating_tmin"] == 5
    assert stout_r2_t5["dominating_state_count"] == 3
    assert stout_r2_t5["basis_operator_id"] == "stout"
    assert stout_r2_t5["overlap_definition_id"] == "bare_basis_squared_overlap_ratio_v1"
    assert abs(stout_r2_t5["q_upper"] - expected) < 1e-15
    assert result["force_qhx_corner_order"] == [
        "spatial_stout_rho0.10_n5", "identity",
    ]
    assert result["force_q_corner_bounds"]["spatial_stout_rho0.10_n5"] == [
        row["q_upper"] for row in result["q_corner_bounds"]
        if row["kernel_id"] == "spatial_stout_rho0.10_n5"
    ]


def test_missing_variant_abstains_with_exact_fit_denominator(tmp_path):
    document = _fit_support()
    document["fit_variants"].pop()
    result = _read(tmp_path, document)
    assert result["status"] == "ABSTAIN_F_SPECTRUM_VARIANTS"
    assert result["fit_variant_denominator"] == {
        "unit": "prespecified_GEVP_fit_variant", "planned": 36,
        "valid": 35, "failed": 0, "abstained": 1,
    }
    assert result["selected_q_cell_denominator"]["abstained"] == 8
    assert result["physical_score"] is None


@pytest.mark.parametrize("split", [
    "F-gap-calibration", "F-validation-calibration", "F-sealed-calibration",
])
def test_closed_splits_are_rejected_even_when_named_calibration(tmp_path, split):
    result = _read(tmp_path, _fit_support(split_id=split), split_id=split)
    assert result["status"] == "ABSTAIN_F_SPECTRUM_CONTENT"
    assert result["source_denominator"]["failed"] == 1
    assert result["selected_q_cell_denominator"]["abstained"] == 8


def test_confidence_alpha_and_state_count_array_mismatch_fail_closed(tmp_path):
    document = _fit_support()
    document["confidence_allocation"]["endpoint_alpha"] *= 2
    result = _read(tmp_path, document)
    assert result["status"] == "ABSTAIN_F_SPECTRUM_CONTENT"

    document = _fit_support()
    document["fit_variants"][0]["gap_lower"] = [0.2, 0.1]
    result = _read(tmp_path, document)
    assert result["status"] == "ABSTAIN_F_SPECTRUM_VARIANTS"
    assert result["fit_variant_denominator"]["failed"] == 1


def test_kernel_indexed_q_requires_its_own_bare_basis_overlap_object(tmp_path):
    document = _fit_support()
    assert _read(tmp_path, document)["status"] == "Q_BOUND_DERIVED_NOT_CERTIFIED"

    document = _fit_support()
    document["fit_variants"][0]["basis_operator_id"] = "stout"
    result = _read(tmp_path, document)
    assert result["status"] == "ABSTAIN_F_SPECTRUM_VARIANTS"
    assert result["fit_variant_denominator"] == {
        "unit": "prespecified_GEVP_fit_variant", "planned": 36,
        "valid": 35, "failed": 1, "abstained": 0,
    }

    document = _fit_support()
    document["overlap_definition_id"] = "gevp_projected_composite_v1"
    result = _read(tmp_path, document)
    assert result["status"] == "ABSTAIN_F_SPECTRUM_CONTENT"


def test_duplicate_fit_variant_does_not_fill_a_missing_cell(tmp_path):
    document = _fit_support()
    document["fit_variants"][-1] = dict(document["fit_variants"][0])
    result = _read(tmp_path, document)
    assert result["status"] == "ABSTAIN_F_SPECTRUM_VARIANTS"
    denominator = result["fit_variant_denominator"]
    assert denominator["valid"] == 34
    assert denominator["failed"] == 1
    assert denominator["abstained"] == 1
    assert sum(denominator[key] for key in ("valid", "failed", "abstained")) == 36


def test_missing_source_and_mismatched_expected_digest_abstain(tmp_path):
    document = _fit_support()
    payload = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (tmp_path / "fit-support.json").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    missing = capture_source_snapshot(tmp_path, {"fit_support": digest}, {})
    result = read_spectral_q_bounds(
        missing, source_name="fit_support", split_id="F-calibration",
        expected_source_sha256=digest,
    )
    assert result["status"] == "ABSTAIN_F_SPECTRUM_SOURCE"
    assert result["source_denominator"]["abstained"] == 1
    assert result["selected_q_cell_denominator"]["abstained"] == 8

    complete = capture_source_snapshot(
        tmp_path, {"fit_support": digest}, {"fit_support": "fit-support.json"}
    )
    result = read_spectral_q_bounds(
        complete, source_name="fit_support", split_id="F-calibration",
        expected_source_sha256="0" * 64,
    )
    assert result["status"] == "ABSTAIN_F_SPECTRUM_HASH_MAP"
    assert result["source_denominator"]["abstained"] == 1


def test_unrepresentable_integer_ratio_is_counted_as_a_failed_variant(tmp_path):
    document = _fit_support()
    document["fit_variants"][0]["overlap_ratio_upper"] = [10**400]

    result = _read(tmp_path, document)

    assert result["status"] == "ABSTAIN_F_SPECTRUM_VARIANTS"
    assert result["fit_variant_denominator"] == {
        "unit": "prespecified_GEVP_fit_variant", "planned": 36,
        "valid": 35, "failed": 1, "abstained": 0,
    }
    assert result["selected_q_cell_denominator"]["abstained"] == 8


def test_overflowing_derived_q_abstains_with_cell_denominator(tmp_path):
    document = _fit_support()
    row = next(
        row for row in document["fit_variants"]
        if row["kernel_id"] == "identity" and row["R"] == 2
        and row["tmin"] == 3 and row["n_states"] == 3
    )
    row["overlap_ratio_upper"] = [1e308, 1e308]
    row["gap_lower"] = [0.0, 0.0]

    result = _read(tmp_path, document)

    assert result["status"] == "ABSTAIN_F_SPECTRUM_Q_NUMERIC"
    assert result["fit_variant_denominator"]["valid"] == 36
    assert result["selected_q_cell_denominator"] == {
        "unit": "selected_kernel_R_time_q_cell", "planned": 8,
        "valid": 0, "failed": 1, "abstained": 7,
    }
    assert result["failed_q_cell"] == {
        "kernel_id": "identity", "R": 2, "T": 5,
        "variant_id": row["variant_id"],
    }
    assert result["physical_score"] is None


def test_unrepresentable_confidence_integer_fails_closed(tmp_path):
    document = _fit_support()
    document["confidence_allocation"]["systematic_cell_alpha"] = 10**400

    result = _read(tmp_path, document)

    assert result["status"] == "ABSTAIN_F_SPECTRUM_CONTENT"
    assert result["source_denominator"]["failed"] == 1
    assert result["selected_q_cell_denominator"]["abstained"] == 8
