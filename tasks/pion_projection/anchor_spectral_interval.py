"""Set-valued ground-mass propagation from a joint PP mean-correlation box.

The interval reducer consumes simultaneous ensemble-mean bounds for folded PP
correlators at t=3,4,5. Its snapshot wrapper reads only immutable captured
bytes from an explicitly open anchor split, then applies the conditional
Markov vector-mean helper. Neither path authenticates chain/support receipts
or certifies a bridge to an independent target or physical score.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np

from .anchor_scalar_reader import (
    FEATURE_VECTOR_RESULT_SCHEMA,
    read_anchor_scalar_snapshot,
)
from .bounded_block_risk_anchor_vector import markov_anchor_mean_box


def _finite_real(value: object, label: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{label} must be a finite real number")
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite real number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be a finite real number")
    return converted


def _interval(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a two-element [lower, upper] interval")
    lower = _finite_real(value[0], f"{label}.lower")
    upper = _finite_real(value[1], f"{label}.upper")
    if lower > upper:
        raise ValueError(f"{label} lower endpoint exceeds upper endpoint")
    return lower, upper


def _log_positive_sum(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        raise ValueError("positive-spectrum PP mean upper endpoints must be positive")
    log_left, log_right = math.log(left), math.log(right)
    largest = max(log_left, log_right)
    return largest + math.log1p(math.exp(min(log_left, log_right) - largest))


def _acosh_from_log_ratio(log_ratio: float) -> float:
    """Stable acosh(exp(x)) for x >= 0, including ratios beyond float range."""
    if log_ratio < 0:
        raise ValueError("the PP mean box admits no positive-spectrum ratio >= 1")
    return log_ratio + math.log1p(math.sqrt(-math.expm1(-2.0 * log_ratio)))


def propagate_ground_mass_interval(
    *,
    c3_mean_interval: object,
    c4_mean_interval: object,
    c5_mean_interval: object,
    score_scale_R: float,
) -> dict[str, Any]:
    """Return a conditional identified set for the same ensemble's ground mass.

    For a positive periodic spectrum, ``C(t)=sum_n a_n cosh(E_n*(8-t))``
    with ``a_n>0`` and ``E_n>=0``. Then

    ``(C(3)+C(5))/(2*C(4))``

    is a positive weighted mean of ``cosh(E_n)``. Therefore the ground mass
    lies in ``[0, m_eff(4)]``. Given simultaneous boxes for the three means,
    this function uses the largest possible numerator and smallest positive
    denominator to form a conservative upper endpoint. A caller must still
    establish the vector-box coverage, the positive spectral representation,
    and equality of this ground state to the external ``m_star`` target.
    """
    c3_lower, c3_upper = _interval(c3_mean_interval, "c3_mean_interval")
    c4_lower, c4_upper = _interval(c4_mean_interval, "c4_mean_interval")
    c5_lower, c5_upper = _interval(c5_mean_interval, "c5_mean_interval")
    scale = _finite_real(score_scale_R, "score_scale_R")
    if scale <= 0:
        raise ValueError("score_scale_R must be positive")
    if c4_lower <= 0:
        raise ValueError("c4 mean interval must have a strictly positive lower endpoint")

    # Monotonicity of the ratio on C4>0 gives its upper endpoint. Log-space
    # arithmetic avoids overflow in C3+C5 and underflow in 2*C4.
    log_ratio_upper = (
        _log_positive_sum(c3_upper, c5_upper)
        - math.log(2.0)
        - math.log(c4_lower)
    )
    effective_mass_upper = _acosh_from_log_ratio(log_ratio_upper)
    identified_interval = [0.0, effective_mass_upper]
    minimax_radius = 0.5 * effective_mass_upper

    return {
        "status": "PARTIAL_IDENTIFICATION_NOT_CERTIFIED",
        "estimand": "same-ensemble PP spectral ground mass",
        "units": "a_t m_pi",
        "input_mean_intervals": {
            "C_fold_3": [c3_lower, c3_upper],
            "C_fold_4": [c4_lower, c4_upper],
            "C_fold_5": [c5_lower, c5_upper],
        },
        "log_ratio_upper": log_ratio_upper,
        "effective_mass_upper": effective_mass_upper,
        "identified_ground_mass_interval": identified_interval,
        "identified_set_diameter": effective_mass_upper,
        "midpoint_minimax_radius_for_same_ensemble_ground_mass": minimax_radius,
        "midpoint_minimax_radius_over_R": minimax_radius / scale,
        "score_scale_R": scale,
        "target_bridge_status": "unverified_for_independent_E_reference",
        "joint_mean_box_coverage": "conditional; not established by this reducer",
        "assumptions": [
            "the three input intervals jointly cover the ensemble means of folded PP C(3), C(4), C(5)",
            "the same ensemble correlator has a positive periodic cosh spectral sum with nonnegative energies",
            "the score target is the ground mass of that same ensemble correlator",
            "a separate review establishes the bridge to any independent PP/A4P m_star reference",
        ],
        "text_flow": [
            "same immutable anchor-chain snapshot -> fixed folded-PP feature vector (C3,C4,C5)",
            "simultaneous Markov mean box -> ratio upper (C3_upper+C5_upper)/(2*C4_lower)",
            "positive-spectrum identity -> m_ground in [0, acosh(ratio_upper)]",
            "propagate the set through candidate risks or rankings only after m_star bridge review",
            "keep physical_score null until coverage, source, dependence, and target-bridge gates pass",
        ],
        "common_bias_bound_C": None,
        "certified": False,
        "physical_score": None,
    }


def propagate_ground_mass_from_markov_box(
    mean_box: object,
    *,
    feature_order: object,
    score_scale_R: float,
) -> dict[str, Any]:
    """Connect the existing Markov vector-mean helper to spectral propagation.

    The caller must freeze feature order as ``(C_fold_3, C_fold_4,
    C_fold_5)`` before examining samples. This checks the reducer's output
    shape and order labels supplied by the caller; it does not authenticate
    feature construction, support bounds, chain receipts, or coverage.
    """
    expected_order = ("C_fold_3", "C_fold_4", "C_fold_5")
    if (not isinstance(feature_order, (list, tuple))
            or tuple(feature_order) != expected_order):
        raise ValueError("feature_order must be frozen as (C_fold_3, C_fold_4, C_fold_5)")
    if (not isinstance(mean_box, Mapping)
            or mean_box.get("status") != "CONDITIONAL_MARKOV_VECTOR_MEAN_BOX_ONLY"
            or mean_box.get("feature_count") != 3):
        raise ValueError("mean_box must be a three-feature conditional Markov vector box")
    rows = mean_box.get("feature_intervals")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("mean_box must contain exactly three feature intervals")
    intervals = []
    for index, row in enumerate(rows):
        if (not isinstance(row, Mapping)
                or type(row.get("feature_index")) is not int
                or row.get("feature_index") != index
                or "confidence_interval" not in row):
            raise ValueError("mean_box feature intervals must use the frozen ordered indices")
        intervals.append(row["confidence_interval"])
    coverage = _finite_real(
        mean_box.get("joint_coverage_probability_lower_bound"),
        "mean_box.joint_coverage_probability_lower_bound",
    )
    if not 0 <= coverage <= 1:
        raise ValueError("mean_box joint coverage lower bound must be in [0,1]")

    result = propagate_ground_mass_interval(
        c3_mean_interval=intervals[0],
        c4_mean_interval=intervals[1],
        c5_mean_interval=intervals[2],
        score_scale_R=score_scale_R,
    )
    result["input_markov_mean_box"] = {
        "status": mean_box["status"],
        "joint_coverage_probability_lower_bound": coverage,
        "coverage_authentication": "not performed by this range propagator",
    }
    return result


def read_anchor_ground_mass_interval_snapshot(
    snapshot: object,
    *,
    support_bounds: object,
    gamma_ps_lower: float,
    alpha_anchor: float,
    initial_law_tv_bound: float,
    burn_in_units: int,
    stationary_start: bool,
    score_scale_R: float,
    expected_states: int | None = None,
    manifest_name: str = "anchor_manifest",
    correlator_name: str = "anchor_correlator_data",
    protocol_name: str = "anchor_protocol",
) -> dict[str, Any]:
    """Read an open immutable anchor snapshot and propagate a Markov mean box.

    All chain and support parameters are explicit. Their presence here does
    not authenticate their receipts; this remains a conditional, diagnostic
    path. Closed splits are rejected by the underlying snapshot reader.
    """
    source = read_anchor_scalar_snapshot(
        snapshot,
        manifest_name=manifest_name,
        correlator_name=correlator_name,
        protocol_name=protocol_name,
        expected_states=expected_states,
        include_feature_vector=True,
    )
    if source.get("schema") != FEATURE_VECTOR_RESULT_SCHEMA:
        raise ValueError("anchor source reader did not return the frozen feature-vector schema")
    feature_record = source.get("feature_vector")
    if not isinstance(feature_record, Mapping):
        raise ValueError("anchor source reader omitted the feature vector")
    if feature_record.get("feature_order") != ["C_fold_3", "C_fold_4", "C_fold_5"]:
        raise ValueError("anchor source feature order differs from the spectral interval contract")
    feature_denominator = feature_record.get("denominator")
    planned = feature_denominator.get("planned") if isinstance(feature_denominator, Mapping) else None
    if (not isinstance(feature_denominator, Mapping)
            or feature_denominator.get("unit") != "saved gauge configuration feature row"
            or type(planned) is not int or planned < 1
            or type(feature_denominator.get("valid")) is not int
            or feature_denominator.get("valid") != planned
            or type(feature_denominator.get("failed")) is not int
            or feature_denominator.get("failed") != 0
            or type(feature_denominator.get("abstained")) is not int
            or feature_denominator.get("abstained") != 0):
        raise ValueError("anchor PP feature rows are incomplete")
    features = np.asarray(feature_record.get("matrix"), dtype=np.float64)
    if (features.ndim != 2 or features.shape[1] != 3
            or features.shape[0] != planned
            or not np.isfinite(features).all()):
        raise ValueError("anchor PP feature matrix does not match its denominator")

    mean_box = markov_anchor_mean_box(
        features,
        support_bounds=support_bounds,
        gamma_ps_lower=gamma_ps_lower,
        alpha_anchor=alpha_anchor,
        initial_law_tv_bound=initial_law_tv_bound,
        burn_in_units=burn_in_units,
        stationary_start=stationary_start,
    )
    result = propagate_ground_mass_from_markov_box(
        mean_box,
        feature_order=feature_record["feature_order"],
        score_scale_R=score_scale_R,
    )
    result.update({
        "source_read_mode": "immutable_snapshot",
        "split_id": source["split_id"],
        "chain_id": source["chain_id"],
        "source_hashes": source["source_hashes"],
        "feature_denominator": feature_denominator,
        "scalar_map_diagnostic_status": source["status"],
        "markov_mean_box": mean_box,
    })
    return result
