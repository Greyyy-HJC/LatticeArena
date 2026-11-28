"""Exact bounded block-risk range over a caller-supplied common target set.

This helper propagates target-set uncertainty through the already selected
clipped squared-error contrast. It is descriptive and does not certify the
coverage of the target set, the anchor-to-reference bridge, or score-chain
concentration assumptions.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.numerics import as_finite_real, as_real_float_array


def _finite_real(value: object, label: str) -> float:
    return as_finite_real(value, name=label)


def _target_interval(value: object) -> tuple[float, float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2):
        raise ValueError("target_interval must be a two-element [lower, upper] interval")
    lower = _finite_real(value[0], "target_interval.lower")
    upper = _finite_real(value[1], "target_interval.upper")
    if lower > upper:
        raise ValueError("target_interval lower endpoint exceeds its upper endpoint")
    return lower, upper


def _normalized_difference(
    target: float, reference: float, candidate: float, scale: float,
) -> float:
    """Evaluate the clipped squared-error contrast without forming huge squares."""
    coordinate_scale = max(abs(target), abs(reference), abs(candidate), scale)
    if coordinate_scale == 0.0:
        return 0.0
    target_scaled = target / coordinate_scale
    reference_scaled = reference / coordinate_scale
    candidate_scaled = candidate / coordinate_scale
    score_scale_scaled = scale / coordinate_scale
    numerator = ((candidate_scaled - reference_scaled)
                 * (2.0 * target_scaled - reference_scaled - candidate_scaled))
    denominator = score_scale_scaled * score_scale_scaled
    if denominator == 0.0:
        if numerator > 0.0:
            return 1.0
        if numerator < 0.0:
            return -1.0
        return 0.0
    value = numerator / denominator
    if value >= 1.0:
        return 1.0
    if value <= -1.0:
        return -1.0
    return float(value)


def _mean_score_at(
    target: float,
    reference: np.ndarray,
    candidate: np.ndarray,
    valid: np.ndarray,
    scale: float,
) -> float:
    scores = np.full(reference.shape, -1.0, dtype=float)
    for index in np.flatnonzero(valid):
        scores[index] = _normalized_difference(
            target, float(reference[index]), float(candidate[index]), scale,
        )
    return float(np.mean(scores))


def _breakpoints(
    lower: float,
    upper: float,
    reference: np.ndarray,
    candidate: np.ndarray,
    valid: np.ndarray,
    scale: float,
) -> list[float]:
    points: list[float] = []
    for index in np.flatnonzero(valid):
        ref = float(reference[index])
        cand = float(candidate[index])
        coordinate_scale = max(abs(lower), abs(upper), abs(ref), abs(cand), scale)
        lo_scaled = lower / coordinate_scale
        hi_scaled = upper / coordinate_scale
        ref_scaled = ref / coordinate_scale
        cand_scaled = cand / coordinate_scale
        scale_scaled = scale / coordinate_scale
        slope_half = cand_scaled - ref_scaled
        if slope_half == 0.0:
            continue
        threshold_squared = scale_scaled * scale_scaled
        if threshold_squared == 0.0:
            candidates = [(ref_scaled + cand_scaled) / 2.0]
        else:
            candidates = [
                (ref_scaled + cand_scaled
                 + sign * threshold_squared / slope_half) / 2.0
                for sign in (-1.0, 1.0)
            ]
        for point_scaled in candidates:
            if lo_scaled <= point_scaled <= hi_scaled:
                point = point_scaled * coordinate_scale
                if math.isfinite(point):
                    points.append(min(upper, max(lower, point)))
    return points


def bounded_block_risk_setwise(
    reference_values: object,
    candidate_values: object,
    *,
    valid_blocks: object,
    target_interval: object,
    score_scale_R: float,
) -> dict[str, Any]:
    """Return the exact planned-mean score range over one common target set.

    For each planned block ``i``, the score is

    ``clip(((m - reference_i)^2 - (m - candidate_i)^2) / R^2, -1, 1)``.

    The unknown target ``m`` is shared by every block and lies in the supplied
    closed interval. Failed blocks remain in the planned denominator as -1.
    Because each clipped contrast is piecewise linear in ``m``, exact extrema
    occur at an interval endpoint or a clipping breakpoint; all are evaluated.
    This returns only target-set variation of the finite planned-block mean,
    not a confidence bound over the score chain.
    """
    reference = as_real_float_array(reference_values, name="reference_values")
    candidate = as_real_float_array(candidate_values, name="candidate_values")
    valid = np.asarray(valid_blocks)
    if reference.ndim != 1 or reference.size == 0:
        raise ValueError("reference_values must be a nonempty one-dimensional planned-block array")
    if candidate.shape != reference.shape:
        raise ValueError("candidate_values must match the planned-block reference array")
    if valid.shape != reference.shape or valid.dtype != np.bool_:
        raise ValueError("valid_blocks must be a same-shape boolean array")
    if (not np.isfinite(reference[valid]).all()
            or not np.isfinite(candidate[valid]).all()):
        raise ValueError("valid-block reference and candidate values must be finite")
    scale = _finite_real(score_scale_R, "score_scale_R")
    if (scale <= 0.0 or not math.isfinite(scale * scale)
            or scale * scale <= 0.0):
        raise ValueError("score_scale_R must have a finite, positive square")
    lower, upper = _target_interval(target_interval)

    planned = int(reference.size)
    valid_count = int(valid.sum())
    failed_count = planned - valid_count
    denominators = {
        "planned_blocks": planned,
        "valid_blocks": valid_count,
        "failed_blocks": failed_count,
        "abstained_blocks": 0,
    }
    if valid_count == 0:
        return {
            "status": "ABSTAIN_NO_VALID_BLOCKS",
            "estimand": "bounded_block_risk_over_caller_supplied_common_target_set",
            "target_interval": [lower, upper],
            "normalized_score_interval": None,
            "ranking": "unresolved_no_valid_blocks",
            "denominators": denominators,
            "result_denominators": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
            "confidence_coverage": None,
            "target_bridge_status": "not_verified_here",
            "certified": False,
            "physical_score": None,
        }

    targets = [lower, upper]
    targets.extend(_breakpoints(lower, upper, reference, candidate, valid, scale))
    # Floating-point duplicates only waste work; exact set membership preserves
    # distinct corners and all mathematically relevant clipping points.
    targets = sorted(set(targets))
    evaluated = [
        (target, _mean_score_at(target, reference, candidate, valid, scale))
        for target in targets
    ]
    score_min = min(score for _, score in evaluated)
    score_max = max(score for _, score in evaluated)
    tolerance = 8.0 * np.finfo(float).eps
    minimizing = [target for target, score in evaluated
                  if abs(score - score_min) <= tolerance]
    maximizing = [target for target, score in evaluated
                  if abs(score - score_max) <= tolerance]
    if score_min > 0.0:
        ranking = "candidate_better_for_all_targets_in_set"
    elif score_max < 0.0:
        ranking = "reference_better_for_all_targets_in_set"
    else:
        ranking = "ordering_unresolved_over_target_set"

    return {
        "status": "SET_VALUED_SCORE_DIAGNOSTIC_NOT_CERTIFIED",
        "estimand": "bounded_block_risk_over_caller_supplied_common_target_set",
        "score_definition": "clip(((m-reference)^2-(m-candidate)^2)/R^2,-1,1); failed planned blocks score -1",
        "target_interval": [lower, upper],
        "target_set_coverage": "not established by this helper",
        "normalized_score_interval": [score_min, score_max],
        "minimizing_target_values": minimizing,
        "maximizing_target_values": maximizing,
        "evaluated_breakpoint_count": len(targets),
        "ranking": ranking,
        "denominators": denominators,
        "result_denominators": {"planned": 1, "valid": 1, "failed": 0, "abstained": 0},
        "confidence_coverage": None,
        "target_bridge_status": "not_verified_here",
        "requirements": [
            "the caller-supplied target interval covers the same scalar target used in every planned block",
            "candidate/reference values and R use one frozen scalar observable and unit",
            "a separate dependence-aware confidence bound is needed for score-chain sampling uncertainty",
            "a separate anchor-to-reference bridge and review are needed before using a Pion interval as this target set",
        ],
        "certified": False,
        "physical_score": None,
        "text_flow": [
            "candidate/reference block predictions + a common target interval",
            "evaluate clipped squared-error contrast at both endpoints and every clipping breakpoint",
            "failed blocks remain -1 in the planned mean at every target value",
            "report score-set range and whether its sign is uniform over the whole target set",
            "do not call this a confidence interval or physical score without separate chain and target-bridge evidence",
        ],
    }
