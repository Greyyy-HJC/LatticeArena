"""Distribution-free inference for a separate bounded block-risk estimand.

This module deliberately does not replace ``risk_statistics.py``'s D04/D06
estimand.  It defines a clipped-loss risk over independent macroblocks and
offers one-sided Hoeffding and empirical-Bernstein lower bounds for the
fixed D06 effect threshold.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


TIMES = (2, 3, 4, 5)
ARMS = ("trainable_w", "same_budget_slsqp", "degree4_chebyshev")
R_STAR = 10.0 / 9.0
ALPHA_TOTAL = 0.05
FAMILY_SIZE = 2
BOUND_EVENTS = FAMILY_SIZE + 1  # two paired contrasts plus one shared denominator bound
DELTA_PER_BOUND = ALPHA_TOTAL / BOUND_EVENTS
LOSS_CAP = 1.0


def _failure_mask(value: object | None, n_blocks: int) -> np.ndarray:
    if value is None:
        return np.zeros((n_blocks, len(ARMS)), dtype=bool)
    mask = np.asarray(value)
    if mask.dtype.kind != "b" or mask.shape != (n_blocks, len(ARMS)):
        raise ValueError("arm_failure_mask must be boolean with shape (blocks, 3 arms)")
    return mask


def bounded_block_scores(
    q_by_block_arm_time: object,
    *,
    arm_failure_mask: object | None = None,
) -> np.ndarray:
    """Map q[b, arm, t] into block scores in [0, 1].

    Each block is one inferential unit.  The four time points are averaged
    within that unit and never counted as four independent observations.
    The clipping is part of this alternative estimand:
    ``ell(q) = min((q - 1)**2, 1)``.
    """
    raw = np.asarray(q_by_block_arm_time)
    if raw.ndim != 3 or raw.shape[1:] != (len(ARMS), len(TIMES)):
        raise ValueError("q input must have shape (blocks, 3 arms, 4 times)")
    if not np.isrealobj(raw):
        raise ValueError("q input must be real-valued")
    values = np.asarray(raw, dtype=np.float64)
    failures = _failure_mask(arm_failure_mask, int(values.shape[0]))
    finite = np.isfinite(values)
    # Missing/nonfinite q is tolerated only when that entire assigned arm's
    # optimization failed.  The failure outcome itself is scored below as 1.
    if np.any(~finite & ~failures[:, :, None]):
        raise ValueError("q input contains a non-finite value outside a failed arm")
    values = np.where(finite, values, 1.0)
    # Clip the deviation before squaring so even very large finite q values
    # cannot overflow.  This is exactly min((q-1)^2, 1).
    deviation = np.minimum(np.abs(values - 1.0), 1.0)
    losses = deviation * deviation
    scores = np.mean(losses, axis=2)
    scores[failures] = LOSS_CAP
    return scores


def _one_sided_lower(
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
    delta: float,
    method: str,
) -> dict[str, float | str | int]:
    n = int(values.size)
    width = upper - lower
    if n < 2:
        raise ValueError("at least two independent blocks are required")
    if not (math.isfinite(lower) and math.isfinite(upper) and width > 0):
        raise ValueError("invalid support interval")
    if not (math.isfinite(delta) and 0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    if not np.isfinite(values).all() or np.any(values < lower) or np.any(values > upper):
        raise ValueError("observations fall outside the declared finite support")

    mean = float(np.mean(values))
    sample_variance = float(np.var(values, ddof=1))
    if method == "hoeffding":
        radius = width * math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    elif method == "empirical_bernstein":
        # Maurer-Pontil (2009), Theorem 4, rescaled from [0,1] to
        # [lower, upper].  The observations must be i.i.d. across blocks.
        log_term = math.log(2.0 / delta)
        radius = width * (
            math.sqrt(2.0 * (sample_variance / (width * width)) * log_term / n)
            + 7.0 * log_term / (3.0 * (n - 1))
        )
    else:
        raise ValueError("method must be 'hoeffding' or 'empirical_bernstein'")
    return {
        "method": method,
        "n_blocks": n,
        "mean": mean,
        "sample_variance": sample_variance,
        "delta": delta,
        "radius": radius,
        "lower_bound": mean - radius,
    }


def bounded_block_risk_decision(
    block_scores: object,
    *,
    method: str = "empirical_bernstein",
    arm_failure_mask: object | None = None,
) -> dict[str, Any]:
    """Test both fixed D06 effects for the bounded block-risk estimand.

    The familywise error budget is split over three one-sided bounds: the two
    paired contrasts ``E[X_c - r_star X_W]`` and a shared lower bound proving
    ``E[X_W] > 0``.  A positive contrast lower bound plus a positive W-risk
    lower bound is equivalent to a risk ratio above ``r_star``.  Failure to
    separate the denominator from zero is reported as abstention.
    """
    scores = np.asarray(block_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(ARMS):
        raise ValueError("block scores must have shape (blocks, 3 arms)")
    if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > LOSS_CAP):
        raise ValueError("block scores must be finite and lie in [0, 1]")
    n = int(scores.shape[0])
    failures = _failure_mask(arm_failure_mask, n)
    if np.any(scores[failures] != LOSS_CAP):
        raise ValueError("failed arm blocks must carry the maximum loss score 1")
    failure_counts = {
        arm: int(np.count_nonzero(failures[:, index]))
        for index, arm in enumerate(ARMS)
    }
    failure_summary = {
        "arm_failure_counts": failure_counts,
        "algorithm_failure_blocks_any_arm": int(np.count_nonzero(np.any(failures, axis=1))),
        "scored_blocks": n,
        "failure_score_rule": "an assigned arm optimization failure sets that arm's whole block score to 1; the block remains in n",
    }
    if n < 2:
        return {
            "schema": "optimize-bounded-block-risk-decision-v1",
            "status": "ABSTAIN_INSUFFICIENT_INDEPENDENT_BLOCKS",
            "method": method,
            "estimand": "mean block average of min((q-1)^2, 1) over t=2,3,4,5",
            "r_star": R_STAR,
            "planned_blocks": n,
            "valid_blocks": 0,
            "failed_blocks": 0,
            "abstained_blocks": n,
            **failure_summary,
            "physical_score": None,
        }

    delta = DELTA_PER_BOUND
    w_lower = _one_sided_lower(
        scores[:, 0], lower=0.0, upper=1.0, delta=delta, method=method,
    )
    comparisons: dict[str, Any] = {}
    for index, arm in ((1, ARMS[1]), (2, ARMS[2])):
        contrast = scores[:, index] - R_STAR * scores[:, 0]
        contrast_lower = _one_sided_lower(
            contrast, lower=-R_STAR, upper=1.0, delta=delta, method=method,
        )
        comparisons[arm] = {
            "empirical_risk": float(np.mean(scores[:, index])),
            "empirical_ratio_if_defined": (
                float(np.mean(scores[:, index]) / np.mean(scores[:, 0]))
                if float(np.mean(scores[:, 0])) > 0.0 else None
            ),
            "contrast": contrast_lower,
            "threshold_pass": float(contrast_lower["lower_bound"]) > 0.0,
        }

    positive_denominator = float(w_lower["lower_bound"]) > 0.0
    if not positive_denominator:
        status = "ABSTAIN_TRAINABLE_RISK_NOT_SEPARATED_FROM_ZERO"
        valid_blocks, abstained_blocks = 0, n
    else:
        all_pass = all(item["threshold_pass"] for item in comparisons.values())
        status = "PASS_BOTH_EFFECT_THRESHOLDS" if all_pass else "NO_FAMILYWISE_RELEASE"
        valid_blocks, abstained_blocks = n, 0

    return {
        "schema": "optimize-bounded-block-risk-decision-v1",
        "status": status,
        "method": method,
        "estimand": "mean block average of min((q-1)^2, 1) over t=2,3,4,5",
        "r_star": R_STAR,
        "family_size": FAMILY_SIZE,
        "alpha_total": ALPHA_TOTAL,
        "delta_per_bound": delta,
        "bound_event_count": BOUND_EVENTS,
        "planned_blocks": n,
        "valid_blocks": valid_blocks,
        "failed_blocks": 0,
        "abstained_blocks": abstained_blocks,
        **failure_summary,
        "trainable_w": {
            "empirical_risk": float(np.mean(scores[:, 0])),
            "positive_lower_bound": w_lower,
            "denominator_proved_positive": positive_denominator,
        },
        "comparisons": comparisons,
        "physical_score": None,
        "assumptions": [
            "macroblocks are independent draws from a fixed target distribution",
            "all three arms and four time points are paired inside each block",
            "optimizer failure is a scored arm outcome with maximum loss, not a dropped block",
            "missing/corrupt inputs outside a declared arm failure cause the run to abstain rather than being imputed",
            "candidate arms and the q-to-loss map are frozen before confirmation",
            "reference-fit uncertainty is handled separately and is not included here",
        ],
    }
