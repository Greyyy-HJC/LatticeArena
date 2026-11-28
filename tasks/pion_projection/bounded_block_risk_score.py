"""Pure CPU evaluation of the conditional bounded block-risk lower bound.

Goal: map planned paired block loss differences to the accepted bounded estimand.
Inputs: one raw difference and an explicit success flag per planned block, plus a
       fully specified normalization, graph bound, multiplicity, and alpha budget.
Deliverable: per-block bounded scores and a conditional lower bound; never a
             physical score or an evaluator-ready certification. Returned
             denominators distinguish planned blocks from the one comparison
             result, which can abstain when all planned blocks fail.
Check: focused tests cover clipping, fixed failure denominator, and abstention.
"""
from __future__ import annotations

import math

import numpy as np

from core.numerics import as_real_float_array

from .dependency_graph_bound import (
    dependence_penalty,
    normalized_lower_bound,
)


def bounded_block_scores(
    raw_differences: object,
    *,
    valid_blocks: object,
    score_scale_R: float,
) -> np.ndarray:
    """Clip paired differences on the last (planned-block) axis.

    Leading axes may batch families and Monte Carlo replicates. Every failed
    planned block receives score ``-1``; it is never removed from the array.
    """
    differences = as_real_float_array(raw_differences, name="raw_differences")
    valid = np.asarray(valid_blocks)
    if differences.ndim < 1 or differences.size == 0:
        raise ValueError("raw_differences must have at least one planned block")
    if valid.shape != differences.shape:
        raise ValueError("valid_blocks must match the planned block array")
    if valid.dtype != np.bool_:
        raise ValueError("valid_blocks must contain booleans")
    if not math.isfinite(score_scale_R) or score_scale_R <= 0:
        raise ValueError("score_scale_R must be finite and positive")
    scale_squared = score_scale_R * score_scale_R
    if not math.isfinite(scale_squared) or scale_squared <= 0:
        raise ValueError("score_scale_R squared must be finite and positive")
    if not np.all(np.isfinite(differences[valid])):
        raise ValueError("valid block differences must be finite")
    scores = np.full(differences.shape, -1.0, dtype=float)
    scores[valid] = np.clip(differences[valid] / scale_squared, -1.0, 1.0)
    return scores


def conditional_block_risk_lcb(
    raw_differences: object,
    *,
    valid_blocks: object,
    score_scale_R: float,
    chromatic_number: int,
    family: int,
    alpha_score: float,
    alpha_anchor: float,
    alpha_total: float,
    relative_anchor_bound_B_over_R: float,
) -> dict:
    """Compute a conditional LCB without turning it into a physical score.

    For each planned block, ``raw_differences[i]`` is
    ``(A-y_reference)^2 - (A-y_candidate)^2`` in squared observable units.
    Valid differences are normalized by ``R**2`` and clipped to ``[-1, 1]``.
    Failed blocks remain in the planned denominator and receive score ``-1``.

    The returned LCB is conditional on a valid dependency graph whose color
    classes are jointly independent given the frozen protocol and anchor, on
    the supplied anchor coverage bound, and on ``|y_candidate-y_reference| <= R``
    for the squared-error Lipschitz correction. This routine cannot establish
    those assumptions and deliberately always returns ``physical_score=None``.
    """
    differences = as_real_float_array(raw_differences, name="raw_differences")
    valid = np.asarray(valid_blocks)
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("raw_differences must be a nonempty one-dimensional array")
    if valid.ndim != 1 or valid.shape != differences.shape:
        raise ValueError("valid_blocks must match the planned block array")
    scores = bounded_block_scores(
        differences, valid_blocks=valid, score_scale_R=score_scale_R,
    )

    planned = int(differences.size)
    valid_count = int(valid.sum())
    failed_count = planned - valid_count

    # Validate the explicit calibration contract even when the data path must
    # abstain because every planned block failed. The dummy result is discarded.
    normalized_lower_bound(
        0.0,
        blocks=planned,
        chromatic_number=chromatic_number,
        family=family,
        alpha_s=alpha_score,
        alpha_anchor=alpha_anchor,
        alpha_total=alpha_total,
        relative_anchor_bound=relative_anchor_bound_B_over_R,
    )

    counts = {
        "planned_blocks": planned,
        "valid_blocks": valid_count,
        "failed_blocks": failed_count,
        "abstained_blocks": 0,
    }
    common = {
        "estimand": "bounded_block_risk",
        "physical_score": None,
        "result_unit": "one planned candidate-reference comparison",
        "denominators": counts,
        "result_denominators": {
            "planned": 1,
            "valid": int(valid_count > 0),
            "failed": 0,
            "abstained": int(valid_count == 0),
        },
        "normalized_block_scores": scores.tolist(),
        "relative_anchor_bound_B_over_R": float(relative_anchor_bound_B_over_R),
        "assumptions": [
            "chromatic_number is a valid score-specific dependency-graph upper bound",
            "each color class is jointly independent conditional on protocol and anchor",
            "the supplied anchor bound has coverage at least 1-alpha_anchor",
            "|y_candidate-y_reference| <= score_scale_R for the squared-error correction",
        ],
    }
    if valid_count == 0:
        return {
            **common,
            "status": "ABSTAIN_NO_VALID_BLOCKS",
            "valid_block_mean": None,
            "planned_block_mean": -1.0,
            "concentration_penalty": None,
            "anchor_penalty": 2.0 * relative_anchor_bound_B_over_R,
            "normalized_lower_bound": None,
        }

    failure_fraction = failed_count / planned
    valid_mean = float(scores[valid].mean())
    planned_mean = float(scores.mean())
    penalty = dependence_penalty(
        blocks=planned,
        chromatic_number=chromatic_number,
        family=family,
        alpha_s=alpha_score,
    )
    lower_bound = normalized_lower_bound(
        valid_mean,
        blocks=planned,
        chromatic_number=chromatic_number,
        family=family,
        alpha_s=alpha_score,
        alpha_anchor=alpha_anchor,
        alpha_total=alpha_total,
        relative_anchor_bound=relative_anchor_bound_B_over_R,
        failure_fraction=failure_fraction,
    )
    return {
        **common,
        "status": "CONDITIONAL_BOUND_ONLY",
        "valid_block_mean": valid_mean,
        "planned_block_mean": planned_mean,
        "concentration_penalty": penalty,
        "anchor_penalty": 2.0 * relative_anchor_bound_B_over_R,
        "normalized_lower_bound": lower_bound,
    }
