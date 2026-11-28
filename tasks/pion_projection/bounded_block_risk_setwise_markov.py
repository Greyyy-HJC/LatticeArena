"""Conditional uniform Markov LCB over a common target interval.

This composes the set-valued bounded-risk score with the support-only
time-varying Markov Bernstein route.  It is a CPU diagnostic: receipt hashes
are identifiers supplied by the caller, not authenticated evidence, and the
result never certifies a physical score.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import numpy as np

from core.numerics import as_finite_real, as_real_float_array
from .bounded_block_risk_markov_varying import (
    markov_time_varying_score_lcb_from_gap_bound,
)
from .bounded_block_risk_setwise import _normalized_difference

_HARD_MAX_GRID_POINTS = 4096


def _interval(value: object, *, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element [lower, upper] interval")
    lower = as_finite_real(value[0], name=f"{name}.lower")
    upper = as_finite_real(value[1], name=f"{name}.upper")
    if lower > upper:
        raise ValueError(f"{name} lower endpoint exceeds its upper endpoint")
    return lower, upper


def _sha256(value: object, *, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex string")
    return value


def _fraction_as_float_upper(value: Fraction) -> float:
    """Convert a nonnegative exact rational to float64, rounding upward."""
    result = float(value)
    if Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _target_grid(
    lower: float,
    upper: float,
    *,
    score_scale_R: float,
    mesh_step_R: float,
    max_grid_points: int,
) -> tuple[list[float], float, int]:
    if isinstance(max_grid_points, bool) or not isinstance(max_grid_points, int):
        raise ValueError("max_grid_points must be an integer")
    if max_grid_points < 1:
        raise ValueError("max_grid_points must be positive")
    if max_grid_points > _HARD_MAX_GRID_POINTS:
        raise ValueError(
            f"max_grid_points must not exceed the hard cap {_HARD_MAX_GRID_POINTS}"
        )
    width = upper - lower
    if not math.isfinite(width):
        raise ValueError("target interval width must be finite")
    if width == 0.0:
        return [lower], 0.0, 0

    exact_width = Fraction.from_float(upper) - Fraction.from_float(lower)
    exact_max_spacing = (
        Fraction.from_float(mesh_step_R) * Fraction.from_float(score_scale_R)
    )
    if exact_max_spacing <= 0:
        raise ValueError("mesh_step_R * score_scale_R must be positive")
    ratio = exact_width / exact_max_spacing
    segments = max(1, (ratio.numerator + ratio.denominator - 1) // ratio.denominator)
    if segments + 1 > max_grid_points:
        raise ValueError("target grid exceeds max_grid_points")

    raw_grid = np.linspace(lower, upper, num=segments + 1, dtype=float)
    grid = sorted(set(float(value) for value in raw_grid))
    if grid[0] != lower or grid[-1] != upper:
        raise ValueError("target grid failed to retain both interval endpoints")
    if len(grid) > 1:
        max_gap = max(
            Fraction.from_float(right) - Fraction.from_float(left)
            for left, right in zip(grid[:-1], grid[1:])
        )
        if max_gap > exact_max_spacing:
            raise ValueError("float64 target grid cannot satisfy the requested mesh spacing")
        exact_mesh_penalty = max_gap / Fraction.from_float(score_scale_R)
        mesh_penalty = (
            2.0 if exact_mesh_penalty >= 2
            else _fraction_as_float_upper(exact_mesh_penalty)
        )
    else:
        mesh_penalty = 0.0
    # Scores lie in [-1, 1], so their difference can never exceed two.
    mesh_penalty = min(2.0, mesh_penalty)
    return grid, mesh_penalty, len(grid) - 1


def setwise_markov_score_lcb_from_gap_bound(
    reference_values: object,
    candidate_values: object,
    *,
    valid_blocks: object,
    target_interval: object,
    prediction_domain: object,
    score_scale_R: float,
    mesh_step_R: float,
    max_grid_points: int = 4096,
    gamma_ps_lower: float,
    family: int,
    alpha_score: float,
    alpha_anchor: float,
    alpha_total: float,
    initial_law_mode: str,
    burn_in_blocks: int,
    initial_law_tv_bound: float,
    anchor_score_shift_bound: float,
    gamma_ps_lower_receipt_sha256: str,
    dependence_evidence_receipt_sha256: str,
    block_chain_receipt_sha256: str,
    initial_law_receipt_sha256: str,
    anchor_shift_receipt_sha256: str,
    score_schedule_receipt_sha256: str,
    target_set_coverage_receipt_sha256: str,
    prediction_domain_receipt_sha256: str,
) -> dict[str, Any]:
    """Bound the expected planned score uniformly over a target interval.

    The finite grid receives simultaneous Eq. (3.26) support-gap Markov LCBs.
    If the valid reference and candidate predictions both lie in a frozen
    domain of width at most ``R``, each clipped score is ``2/R``-Lipschitz in
    the common target.  Thus an interval grid with largest gap ``h`` needs an
    additional ``h/R`` lower-bound penalty to cover every target between grid
    nodes.

    ``family`` counts the candidate/reference comparisons outside this
    interval grid.  The Markov route applies a family size of
    ``family * number_of_grid_points``.  ``alpha_anchor`` must cover the joint
    target-set coverage and any probabilistic residual anchor bridge; if the
    target interval depends on this score chain, a valid joint/conditional
    argument is required.  This function only checks receipt-hash syntax.
    """
    reference = as_real_float_array(reference_values, name="reference_values")
    candidate = as_real_float_array(candidate_values, name="candidate_values")
    valid = np.asarray(valid_blocks)
    if reference.ndim != 1 or reference.size == 0:
        raise ValueError("reference_values must be a nonempty planned-block vector")
    if candidate.shape != reference.shape:
        raise ValueError("candidate_values must match reference_values")
    if valid.shape != reference.shape or valid.dtype != np.bool_:
        raise ValueError("valid_blocks must be a matching boolean vector")
    if (not np.isfinite(reference[valid]).all()
            or not np.isfinite(candidate[valid]).all()):
        raise ValueError("valid-block predictions must be finite")

    lower, upper = _interval(target_interval, name="target_interval")
    domain_lower, domain_upper = _interval(prediction_domain, name="prediction_domain")
    scale = as_finite_real(score_scale_R, name="score_scale_R")
    if (scale <= 0.0 or not math.isfinite(scale * scale)
            or scale * scale <= 0.0):
        raise ValueError("score_scale_R must have a finite, positive square")
    domain_width = domain_upper - domain_lower
    if not math.isfinite(domain_width):
        raise ValueError("prediction_domain width must be finite")
    exact_domain_width = (
        Fraction.from_float(domain_upper) - Fraction.from_float(domain_lower)
    )
    if exact_domain_width > Fraction.from_float(scale):
        raise ValueError("prediction_domain width must not exceed score_scale_R")
    if np.any(reference[valid] < domain_lower) or np.any(reference[valid] > domain_upper):
        raise ValueError("valid reference predictions fall outside prediction_domain")
    if np.any(candidate[valid] < domain_lower) or np.any(candidate[valid] > domain_upper):
        raise ValueError("valid candidate predictions fall outside prediction_domain")

    mesh_step = as_finite_real(mesh_step_R, name="mesh_step_R")
    if not 0.0 < mesh_step <= 1.0:
        raise ValueError("mesh_step_R must be in (0, 1]")
    if isinstance(family, bool) or not isinstance(family, int) or family < 1:
        raise ValueError("family must be a positive integer")

    target_set_receipt = _sha256(
        target_set_coverage_receipt_sha256,
        name="target_set_coverage_receipt_sha256",
    )
    domain_receipt = _sha256(
        prediction_domain_receipt_sha256,
        name="prediction_domain_receipt_sha256",
    )
    grid, mesh_penalty, segments = _target_grid(
        lower, upper, score_scale_R=scale, mesh_step_R=mesh_step,
        max_grid_points=max_grid_points,
    )
    simultaneous_family = family * len(grid)
    route = {
        "gamma_ps_lower": gamma_ps_lower,
        "family": simultaneous_family,
        "alpha_score": alpha_score,
        "alpha_anchor": alpha_anchor,
        "alpha_total": alpha_total,
        "initial_law_mode": initial_law_mode,
        "burn_in_blocks": burn_in_blocks,
        "initial_law_tv_bound": initial_law_tv_bound,
        "anchor_score_shift_bound": anchor_score_shift_bound,
        "gamma_ps_lower_receipt_sha256": gamma_ps_lower_receipt_sha256,
        "dependence_evidence_receipt_sha256": dependence_evidence_receipt_sha256,
        "block_chain_receipt_sha256": block_chain_receipt_sha256,
        "initial_law_receipt_sha256": initial_law_receipt_sha256,
        "anchor_shift_receipt_sha256": anchor_shift_receipt_sha256,
        "score_schedule_receipt_sha256": score_schedule_receipt_sha256,
    }
    point_bounds: list[dict[str, Any]] = []
    for target in grid:
        scores = np.full(reference.size, -1.0, dtype=float)
        for index in np.flatnonzero(valid):
            scores[index] = _normalized_difference(
                target, float(reference[index]), float(candidate[index]), scale,
            )
        point = markov_time_varying_score_lcb_from_gap_bound(
            scores, valid_blocks=valid, **route,
        )
        point_bounds.append({
            "target": target,
            "planned_mean": point["planned_mean"],
            "normalized_lower_bound": point["normalized_lower_bound"],
            "radius": point["radius"],
            "status": point["status"],
            "denominators": point["denominators"],
        })

    if not np.any(valid):
        uniform_lower = None
        status = "ABSTAIN_NO_VALID_BLOCKS"
    else:
        finite_lcbs = [row["normalized_lower_bound"] for row in point_bounds]
        raw_uniform_lower = min(float(value) for value in finite_lcbs) - mesh_penalty
        raw_uniform_lower = math.nextafter(raw_uniform_lower, -math.inf)
        uniform_lower = max(-1.0, raw_uniform_lower)
        status = "CONDITIONAL_UNIFORM_BOUND_ONLY"

    return {
        "status": status,
        "estimand": "expected_bounded_block_risk_uniform_over_common_target_interval",
        "score_definition": (
            "clip(((m-reference)^2-(m-candidate)^2)/R^2,-1,1); "
            "failed planned blocks score -1"
        ),
        "dependence_method": "finite_grid_support_gap_markov_eq_3_26_plus_lipschitz_cover",
        "target_interval": [lower, upper],
        "target_set_coverage_receipt_sha256": target_set_receipt,
        "prediction_domain": [domain_lower, domain_upper],
        "prediction_domain_receipt_sha256": domain_receipt,
        "score_scale_R": scale,
        "mesh_step_R_requested": mesh_step,
        "target_grid": grid,
        "target_grid_segments": segments,
        "target_grid_points": len(grid),
        "simultaneous_family_size": simultaneous_family,
        "pointwise_bounds": point_bounds,
        "lipschitz_constant_upper_bound": math.nextafter(2.0 / scale, math.inf),
        "maximum_grid_gap_over_R_upper_bound": mesh_penalty,
        "uniform_target_set_lower_bound": uniform_lower,
        "alpha_allocation": {
            "score_family": float(alpha_score),
            "anchor_joint_event": float(alpha_anchor),
            "total": float(alpha_total),
        },
        "conditional_joint_failure_probability_bound": float(alpha_total),
        "target_set_coverage": "not verified by this helper",
        "score_chain_assumptions": "caller supplied; not verified by this helper",
        "anchor_to_reference_bridge": "caller supplied; residual shift is subtracted at every grid point",
        "confidence_coverage": None,
        "certified": False,
        "physical_score": None,
        "data_access": "none",
        "denominators": {
            "planned_blocks": int(reference.size),
            "valid_blocks": int(valid.sum()),
            "failed_blocks": int(reference.size - valid.sum()),
            "abstained_blocks": 0,
        },
        "requirements": [
            "the common target interval covers the same scalar target in every planned block",
            "the target interval and grid are frozen independently of the score chain, or a valid joint/conditional bound accounts for their selection",
            "the target-set coverage event and any probabilistic residual anchor bridge fit inside alpha_anchor",
            "the caller-supplied prediction domain covers every valid reference/candidate prediction and has width no greater than R",
            "the support-gap pseudo-spectral-gap, block-chain, initial-law TV, and score-schedule assumptions hold for every grid score map",
            "the score-family receipt and dependence variance/support evidence cover all grid points and all comparisons counted in family",
            "receipt hashes identify evidence only; this helper does not authenticate their authority or truth",
        ],
        "text_flow": [
            "paired planned-block predictions + common target interval + frozen prediction domain",
            "validate domain width <= R; construct a target grid with requested maximum mesh",
            "at every grid target, code valid scores and keep failed blocks at -1",
            "apply support-gap Markov Eq. (3.26) with family size multiplied by grid-point count",
            "take the minimum simultaneous grid LCB and subtract the Lipschitz cover penalty h/R",
            "report a conditional interval-uniform lower bound; leave coverage and physical score unverified",
        ],
    }
