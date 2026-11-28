"""Conditional Markov-chain bounds for the bounded block-risk estimand.

This module is a CPU math helper only. A numerical pseudo-spectral-gap value
does not count as a receipt, and every result remains uncertified until the
exact paired-macroblock chain, stationarity/burn-in, and frozen score map are
independently justified.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.numerics import as_finite_real, as_real_float_array

from .bounded_block_risk import (
    ALPHA_TOTAL,
    ARMS,
    BOUND_EVENTS,
    R_STAR,
)


SCHEMA = "optimize-bounded-block-risk-markov-v1"


def _finite_real(value: object) -> bool:
    try:
        as_finite_real(value, name="numeric parameter")
    except ValueError:
        return False
    return True


def markov_bernstein_radius(
    *,
    blocks: int,
    gamma_ps_lower: float,
    alpha: float,
    lower: float,
    upper: float,
) -> float:
    """Invert Paulin's corrected pseudo-gap Bernstein inequality.

    The score support is ``[lower, upper]``. We use the support-only bounds
    ``Var_pi(f) <= (upper-lower)^2/4`` and
    ``|f-E_pi(f)| <= upper-lower``. The returned radius applies to a sample
    mean from a stationary, homogeneous Markov chain with a justified
    pseudo-spectral-gap lower bound for the *paired macroblock state chain*.
    The two-sided theorem is used for a lower confidence bound, so this is
    conservative but avoids relying on a one-sided variant.
    """
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 1:
        raise ValueError("blocks must be a positive integer")
    if not _finite_real(gamma_ps_lower) or not 0.0 < float(gamma_ps_lower) <= 1.0:
        raise ValueError("gamma_ps_lower must be finite and in (0, 1]")
    if not _finite_real(alpha) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    if (not _finite_real(lower) or not _finite_real(upper)
            or float(upper) <= float(lower)):
        raise ValueError("support must be finite with upper > lower")

    width = float(upper) - float(lower)
    gamma = float(gamma_ps_lower)
    log_tail = math.log(2.0 / float(alpha))
    variance_bound = width * width / 4.0
    centered_abs_bound = width
    linear = 20.0 * centered_abs_bound * log_tail
    variance_term = 8.0 * log_tail * (blocks + 1.0 / gamma) * variance_bound
    sum_radius = (
        linear + math.sqrt(linear * linear + 4.0 * gamma * variance_term)
    ) / (2.0 * gamma)
    radius = sum_radius / blocks
    if not math.isfinite(radius):
        raise ValueError("Markov Bernstein radius is non-finite")
    return radius


def bounded_block_risk_markov_decision(
    block_scores: object,
    *,
    gamma_ps_lower: float | None,
    tv_bound_after_burn_in: float = 0.0,
    burn_in_blocks: int = 0,
    alpha_total: float = ALPHA_TOTAL,
    arm_failure_mask: object | None = None,
) -> dict[str, Any]:
    """Compute a conditional, familywise lower-bound decision.

    Three events are union-bounded: the trainable-W mean in ``[0,1]`` and the
    two paired contrasts ``X_c-r_star*X_W`` in ``[-r_star,1]``. If the scored
    chain starts within ``tv_bound_after_burn_in`` of stationarity, TV is paid
    once against the union event, leaving ``alpha_total-TV`` for those three
    bounds. ``block_scores`` contains only post-burn-in scored macroblocks;
    all of them stay in the denominator, including explicitly failed solver
    outcomes already encoded as score 1.

    Supplying ``gamma_ps_lower`` is not evidence that the bound applies. The
    output is always ``certified=False`` and ``physical_score=None``.
    """
    scores = as_real_float_array(block_scores, name="block_scores")
    if scores.ndim != 2 or scores.shape[1] != len(ARMS):
        raise ValueError("block_scores must have shape (blocks, 3 arms)")
    if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("block_scores must be finite and lie in [0, 1]")
    n = int(scores.shape[0])
    if arm_failure_mask is None:
        failures = np.zeros(scores.shape, dtype=bool)
    else:
        failures = np.asarray(arm_failure_mask)
        if failures.dtype.kind != "b" or failures.shape != scores.shape:
            raise ValueError("arm_failure_mask must be boolean with shape matching block_scores")
        if np.any(scores[failures] != 1.0):
            raise ValueError("failed arm blocks must carry maximum loss score 1")

    if not _finite_real(alpha_total) or not 0.0 < float(alpha_total) < 1.0:
        raise ValueError("alpha_total must be finite and in (0, 1)")
    if (not _finite_real(tv_bound_after_burn_in)
            or not 0.0 <= float(tv_bound_after_burn_in) <= 1.0):
        raise ValueError("tv_bound_after_burn_in must be finite and in [0, 1]")
    if (isinstance(burn_in_blocks, bool) or not isinstance(burn_in_blocks, int)
            or burn_in_blocks < 0):
        raise ValueError("burn_in_blocks must be a nonnegative integer")

    failure_counts = {
        arm: int(np.count_nonzero(failures[:, index]))
        for index, arm in enumerate(ARMS)
    }
    common: dict[str, Any] = {
        "schema": SCHEMA,
        "estimand": "mean paired-macroblock average of clipped q risk over t=2,3,4,5",
        "dependence_method": "corrected_markov_pseudo_gap_bernstein_support_only",
        "r_star": R_STAR,
        "alpha_total": float(alpha_total),
        "family_event_count": BOUND_EVENTS,
        "tv_bound_after_burn_in": float(tv_bound_after_burn_in),
        "tv_probability_cost": float(tv_bound_after_burn_in),
        "burn_in_blocks_excluded_from_scored_denominator": burn_in_blocks,
        "gamma_ps_lower_assumed": (
            None if gamma_ps_lower is None else float(gamma_ps_lower)
        ),
        "planned_scored_blocks": n,
        "scored_blocks": n,
        "missing_or_corrupt_blocks": 0,
        "algorithm_failure_counts": failure_counts,
        "alpha_per_event_after_tv": None,
        "bounds": None,
        "comparisons": None,
        "certified": False,
        "physical_score": None,
        "data_access": "caller-supplied finite score matrix only; this helper reads no files",
        "assumptions": [
            "the scored paired-macroblock states form a homogeneous Markov chain with a stationary law",
            "gamma_ps_lower is an auditable lower bound for that exact scored block-state chain",
            "the same frozen bounded score map is applied to every chain state; stochastic solver randomness is included in the state or otherwise justified",
            "the support [0,1] for arm scores and [-r_star,1] for each paired contrast holds for every planned scored block",
            "the post-burn-in scored-chain initial law is within the supplied total-variation bound of stationarity",
            "the independent-arm failure rule is already encoded as score 1 and failures remain in the assigned-arm denominator",
            "reference-fit uncertainty and candidate-selection uncertainty are not covered by this method",
        ],
    }
    if n == 0:
        return {
            **common,
            "status": "ABSTAIN_NO_SCORED_BLOCKS",
            "decision_denominators": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
        }
    if gamma_ps_lower is None:
        return {
            **common,
            "status": "ABSTAIN_MISSING_BLOCK_CHAIN_GAP_RECEIPT",
            "decision_denominators": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
        }
    if not _finite_real(gamma_ps_lower) or not 0.0 < float(gamma_ps_lower) <= 1.0:
        raise ValueError("gamma_ps_lower must be finite and in (0, 1]")
    stationary_alpha = float(alpha_total) - float(tv_bound_after_burn_in)
    if stationary_alpha <= 0.0:
        return {
            **common,
            "status": "ABSTAIN_TV_BUDGET_EXHAUSTS_ALPHA",
            "decision_denominators": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
        }
    alpha_per_event = stationary_alpha / BOUND_EVENTS
    radius_w = markov_bernstein_radius(
        blocks=n, gamma_ps_lower=float(gamma_ps_lower), alpha=alpha_per_event,
        lower=0.0, upper=1.0,
    )
    mean_w = float(np.mean(scores[:, 0]))
    lower_w = mean_w - radius_w

    comparisons: dict[str, Any] = {}
    for index, arm in ((1, ARMS[1]), (2, ARMS[2])):
        contrast = scores[:, index] - R_STAR * scores[:, 0]
        radius = markov_bernstein_radius(
            blocks=n, gamma_ps_lower=float(gamma_ps_lower), alpha=alpha_per_event,
            lower=-R_STAR, upper=1.0,
        )
        sample_mean = float(np.mean(contrast))
        comparisons[arm] = {
            "sample_mean_contrast": sample_mean,
            "support": [-R_STAR, 1.0],
            "concentration_radius": radius,
            "lower_bound": sample_mean - radius,
            "threshold_pass": sample_mean - radius > 0.0,
        }

    positive_denominator = lower_w > 0.0
    if not positive_denominator:
        status = "ABSTAIN_TRAINABLE_RISK_NOT_SEPARATED_FROM_ZERO"
        valid = 0
    elif all(item["threshold_pass"] for item in comparisons.values()):
        status = "CONDITIONAL_MARKOV_PASS_BOTH_COMPARISONS"
        valid = 1
    else:
        status = "CONDITIONAL_NO_FAMILYWISE_RELEASE"
        valid = 1

    return {
        **common,
        "status": status,
        "alpha_per_event_after_tv": alpha_per_event,
        "bounds": {
            "trainable_w_mean": {
                "sample_mean": mean_w,
                "support": [0.0, 1.0],
                "concentration_radius": radius_w,
                "lower_bound": lower_w,
                "positive": positive_denominator,
            }
        },
        "comparisons": comparisons,
        "decision_denominators": {
            "planned": 1,
            "valid": valid,
            "failed": 0,
            "abstained": int(not positive_denominator),
        },
    }
