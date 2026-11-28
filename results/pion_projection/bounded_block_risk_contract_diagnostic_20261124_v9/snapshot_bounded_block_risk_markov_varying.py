"""Optional time-varying-score Markov Bernstein route (Paulin Eq. 3.26).

This is a conditional CPU helper for a predeclared sequence of bounded score
maps. It does not estimate a pseudo-spectral gap, variance bounds, TV distance,
or anchor error, and always leaves ``physical_score`` unset.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Callable

import numpy as np


def _finite_real(value: object) -> bool:
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def time_varying_markov_radius(
    *,
    blocks: int,
    gamma_ps_lower: float,
    k_ps: int,
    variance_upper_by_residue: object,
    family: int,
    alpha_stationary: float,
) -> dict:
    """Return the Eq. (3.26) Bernstein radius for a planned score mean.

    ``variance_upper_by_residue[r]`` is a positive, reviewed upper bound on
    ``sum_j Var_pi(f_{r+j*k_ps})``. These values may conservatively inflate
    zero or small true variances; the same inflated values are used to form
    both ``V_sum`` and ``M``. A finite positive value for every residue class
    is required because Eq. (3.26)'s ``M`` divides by the smallest square-root
    class variance. ``k_ps`` must be the reviewed smallest maximizer used by
    the theorem and cannot exceed the number of planned scored states.
    """
    blocks = _positive_integer(blocks, "blocks")
    family = _positive_integer(family, "family")
    k_ps = _positive_integer(k_ps, "k_ps")
    if k_ps > blocks:
        raise ValueError("k_ps cannot exceed planned blocks; empty residue classes are unsupported")
    if not _finite_real(gamma_ps_lower) or not 0 < gamma_ps_lower <= 1:
        raise ValueError("gamma_ps_lower must be finite and in (0, 1]")
    if not _finite_real(alpha_stationary) or not 0 < alpha_stationary < 1:
        raise ValueError("alpha_stationary must be finite and in (0, 1)")
    try:
        raw_bounds = list(variance_upper_by_residue)
    except TypeError as exc:
        raise ValueError("variance_upper_by_residue must be a finite sequence") from exc
    if len(raw_bounds) != k_ps:
        raise ValueError("variance_upper_by_residue length must equal k_ps")
    if any(not _finite_real(value) or value <= 0 for value in raw_bounds):
        raise ValueError("every residue-class variance upper bound must be finite and strictly positive")
    bounds = np.asarray(raw_bounds, dtype=float)
    try:
        variance_sum = math.fsum(float(value) for value in bounds)
        roots = [math.sqrt(float(value)) for value in bounds]
        M = math.fsum(roots) / min(roots)
        M_over_k = M / k_ps
        log_tail = math.log(2.0) + math.log(family) - math.log(alpha_stationary)
        centered_abs_bound = 2.0
        linear = 20.0 * centered_abs_bound * M_over_k * log_tail
        variance_term = 8.0 * log_tail * variance_sum
        gamma = float(gamma_ps_lower)
        root_variance = 2.0 * math.sqrt(gamma) * math.sqrt(variance_term)
        half_root = (linear + math.hypot(linear, root_variance)) / (2.0 * gamma)
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("time-varying Markov bound is outside finite floating-point range") from exc
    if not all(math.isfinite(value) for value in (
            variance_sum, M, M_over_k, log_tail, linear, variance_term, half_root)):
        raise ValueError("time-varying Markov bound is outside finite floating-point range")
    radius = half_root / blocks
    if not math.isfinite(radius):
        raise ValueError("time-varying Markov radius is non-finite")
    return {
        "radius": radius,
        "sum_deviation_radius": half_root,
        "log_tail": log_tail,
        "variance_sum_upper": variance_sum,
        "variance_upper_by_residue": bounds.tolist(),
        "M": M,
        "M_over_k_ps": M_over_k,
        "gamma_ps_lower_bound": float(gamma_ps_lower),
        "k_ps": k_ps,
        "centered_abs_bound": 2.0,
    }


def time_varying_markov_radius_from_gap_bound(
    *,
    blocks: int,
    gamma_ps_lower: float,
    family: int,
    alpha_stationary: float,
) -> dict:
    """Return a support-only Eq. (3.26) radius without exact ``k_ps``.

    For each score map with support in ``[-1, 1]``, ``Var_pi(f_i) <= 1``.
    If ``gamma_ps >= gamma_ps_lower > 0``, the pseudo-gap definition implies
    the (unknown) maximizing integer ``k_ps <= 1/gamma_ps_lower``.  The exact
    binary value of the supplied float is used to compute the conservative
    integer ceiling of its reciprocal, avoiding a rounded-down reciprocal.

    Inflating every residue-class variance to ``ceil(N/k_ps)`` makes
    ``M'/k_ps = 1`` and costs at most ``k_ps*ceil(N/k_ps) <= N+k_ps-1`` in
    the variance sum.  Taking the receipt-backed ``k_ps`` upper bound gives
    ``V_sum <= N + ceil(1/gamma_ps_lower) - 1``.  This is a formula-only,
    conditional alternative; it does not estimate or validate the gap.
    """
    blocks = _positive_integer(blocks, "blocks")
    family = _positive_integer(family, "family")
    if not _finite_real(gamma_ps_lower) or not 0 < gamma_ps_lower <= 1:
        raise ValueError("gamma_ps_lower must be finite and in (0, 1]")
    if not _finite_real(alpha_stationary) or not 0 < alpha_stationary < 1:
        raise ValueError("alpha_stationary must be finite and in (0, 1)")

    gamma = float(gamma_ps_lower)
    numerator, denominator = gamma.as_integer_ratio()
    k_upper = (denominator + numerator - 1) // numerator
    variance_sum_upper = blocks + k_upper - 1
    try:
        log_tail = math.log(2.0) + math.log(family) - math.log(alpha_stationary)
        linear = 40.0 * log_tail  # 20*C with centered |f-Ef| <= 2
        variance_term = 8.0 * log_tail * variance_sum_upper
        root_variance = 2.0 * math.sqrt(gamma) * math.sqrt(variance_term)
        sum_deviation_radius = (
            linear + math.hypot(linear, root_variance)
        ) / (2.0 * gamma)
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("gap-only Markov bound is outside finite floating-point range") from exc
    if not all(math.isfinite(value) for value in (
            log_tail, linear, variance_term, root_variance,
            sum_deviation_radius)):
        raise ValueError("gap-only Markov bound is outside finite floating-point range")
    radius = sum_deviation_radius / blocks
    if not math.isfinite(radius):
        raise ValueError("gap-only Markov radius is non-finite")
    return {
        "radius": radius,
        "sum_deviation_radius": sum_deviation_radius,
        "log_tail": log_tail,
        "variance_sum_upper": variance_sum_upper,
        "variance_upper_by_residue": None,
        "M": None,
        "M_over_k_ps": 1.0,
        "gamma_ps_lower_bound": gamma,
        "k_ps": None,
        "k_ps_upper_bound": k_upper,
        "variance_bound_method": "support_var_le_1_uniform_residue_inflation",
        "centered_abs_bound": 2.0,
        "score_abs_bound": 1.0,
    }


def _markov_time_varying_score_lcb_core(
    planned_scores: object,
    *,
    valid_blocks: object,
    gamma_ps_lower: float,
    family: int,
    alpha_score: float,
    alpha_anchor: float,
    alpha_total: float,
    initial_law_mode: str,
    burn_in_blocks: int,
    initial_law_tv_bound: float,
    anchor_score_shift_bound: float,
    initial_law_receipt_sha256: str,
    anchor_shift_receipt_sha256: str,
    score_schedule_receipt_sha256: str,
    route_receipts: dict[str, str],
    dependence_method: str,
    lcb_scope: str,
    radius_builder: Callable[[int, int, float], dict],
) -> dict:
    """Shared score, alpha, initial-law, and failure handling for Eq. (3.26)."""
    from .bounded_block_risk_markov_gap import _validate_alpha_allocation

    raw_values = np.asarray(planned_scores, dtype=object)
    valid = np.asarray(valid_blocks)
    if raw_values.ndim != 1 or raw_values.size == 0:
        raise ValueError("planned_scores must be a nonempty one-dimensional array")
    if valid.ndim != 1 or valid.shape != raw_values.shape or valid.dtype != np.bool_:
        raise ValueError("valid_blocks must be a matching boolean vector")
    if any(not _finite_real(value) for value in raw_values[valid]):
        raise ValueError("valid planned scores must be finite non-boolean real numbers")
    values = np.full(raw_values.size, np.nan, dtype=float)
    values[valid] = np.asarray(raw_values[valid], dtype=float)
    if np.any(values[valid] < -1.0) or np.any(values[valid] > 1.0):
        raise ValueError("valid planned scores must lie in [-1, 1]")
    receipts = {
        "score_schedule_receipt_sha256": score_schedule_receipt_sha256,
        "initial_law_receipt_sha256": initial_law_receipt_sha256,
        "anchor_shift_receipt_sha256": anchor_shift_receipt_sha256,
        **route_receipts,
    }
    if not all(isinstance(receipt, str) and len(receipt) == 64
               and all(char in "0123456789abcdef" for char in receipt)
               for receipt in receipts.values()):
        raise ValueError("all route receipts must be SHA-256 hex strings")
    _validate_alpha_allocation(
        alpha_score=alpha_score, alpha_anchor=alpha_anchor,
        alpha_total=alpha_total,
    )
    if not _finite_real(initial_law_tv_bound) or not 0 <= initial_law_tv_bound <= 1:
        raise ValueError("initial_law_tv_bound must be finite and in [0, 1]")
    if (not isinstance(initial_law_mode, str)
            or initial_law_mode not in {"stationary", "post_burn_in_tv"}):
        raise ValueError("initial_law_mode must be stationary or post_burn_in_tv")
    burn_in_blocks = (burn_in_blocks if isinstance(burn_in_blocks, int)
                      and not isinstance(burn_in_blocks, bool) else -1)
    if burn_in_blocks < 0:
        raise ValueError("burn_in_blocks must be a nonnegative integer")
    if initial_law_mode == "stationary" and (
            burn_in_blocks != 0 or initial_law_tv_bound != 0):
        raise ValueError("stationary initial law requires zero burn-in and zero TV bound")
    if not _finite_real(anchor_score_shift_bound) or anchor_score_shift_bound < 0:
        raise ValueError("anchor_score_shift_bound must be finite and nonnegative")

    planned = int(values.size)
    family = _positive_integer(family, "family")
    try:
        family_tv_cost = (0.0 if initial_law_tv_bound == 0
                          else family * float(initial_law_tv_bound))
    except OverflowError:
        family_tv_cost = math.inf
    if not math.isfinite(family_tv_cost) or family_tv_cost >= float(alpha_score):
        raise ValueError("family * initial_law_tv_bound must be smaller than alpha_score")
    stationary_alpha = float(alpha_score) - family_tv_cost
    if not math.isfinite(stationary_alpha) or stationary_alpha <= 0:
        raise ValueError("family * initial_law_tv_bound must be smaller than alpha_score")
    radius_details = radius_builder(planned, family, stationary_alpha)

    scores = np.full(planned, -1.0, dtype=float)
    scores[valid] = values[valid]
    valid_count = int(valid.sum())
    failed_count = planned - valid_count
    denominators = {
        "planned": planned,
        "valid": valid_count,
        "failed": failed_count,
        "abstained": 0,
    }
    result_denominators = {
        "planned": 1,
        "valid": int(valid_count > 0),
        "failed": 0,
        "abstained": int(valid_count == 0),
    }
    common = {
        "estimand": "mean of predeclared time-varying bounded block scores",
        "dependence_method": dependence_method,
        "status": "ABSTAIN_NO_VALID_BLOCKS" if valid_count == 0 else "CONDITIONAL_BOUND_ONLY",
        "certified": False,
        "physical_score": None,
        "data_access": "none",
        "denominators": denominators,
        "result_denominators": result_denominators,
        "coded_scores": scores.tolist(),
        "planned_mean": None if valid_count == 0 else float(scores.mean()),
        "radius": radius_details["radius"],
        "initial_law_tv_bound": float(initial_law_tv_bound),
        "initial_law_mode": initial_law_mode,
        "burn_in_blocks": burn_in_blocks,
        "initial_law_receipt_sha256": initial_law_receipt_sha256,
        "familywise_tv_cost": family_tv_cost,
        "stationary_alpha": stationary_alpha,
        "initial_law_mean_shift_penalty": 2.0 * float(initial_law_tv_bound),
        "anchor_score_shift_bound": float(anchor_score_shift_bound),
        "anchor_shift_receipt_sha256": anchor_shift_receipt_sha256,
        "lcb_scope": lcb_scope,
        "score_schedule_receipt_sha256": score_schedule_receipt_sha256,
        **route_receipts,
        **radius_details,
    }
    if valid_count == 0:
        return {**common, "normalized_lower_bound": None}
    lower_bound = (
        float(scores.mean()) - radius_details["radius"]
        - 2.0 * float(initial_law_tv_bound)
        - float(anchor_score_shift_bound)
    )
    if not math.isfinite(lower_bound):
        raise ValueError("time-varying Markov LCB is outside finite floating-point range")
    return {**common, "normalized_lower_bound": lower_bound}


def markov_time_varying_score_lcb(
    planned_scores: object,
    *,
    valid_blocks: object,
    gamma_ps_lower: float,
    k_ps: int,
    variance_upper_by_residue: object,
    family: int,
    alpha_score: float,
    alpha_anchor: float,
    alpha_total: float,
    initial_law_mode: str,
    burn_in_blocks: int,
    initial_law_tv_bound: float,
    anchor_score_shift_bound: float,
    initial_law_receipt_sha256: str,
    anchor_shift_receipt_sha256: str,
    score_schedule_receipt_sha256: str,
    variance_receipt_sha256: str,
    k_ps_receipt_sha256: str,
) -> dict:
    """Compute a conditional LCB using reviewed ``k_ps`` and class variances."""
    def radius_builder(blocks: int, family: int, alpha: float) -> dict:
        return time_varying_markov_radius(
            blocks=blocks,
            gamma_ps_lower=gamma_ps_lower,
            k_ps=k_ps,
            variance_upper_by_residue=variance_upper_by_residue,
            family=family,
            alpha_stationary=alpha,
        )

    return _markov_time_varying_score_lcb_core(
        planned_scores,
        valid_blocks=valid_blocks,
        gamma_ps_lower=gamma_ps_lower,
        family=family,
        alpha_score=alpha_score,
        alpha_anchor=alpha_anchor,
        alpha_total=alpha_total,
        initial_law_mode=initial_law_mode,
        burn_in_blocks=burn_in_blocks,
        initial_law_tv_bound=initial_law_tv_bound,
        anchor_score_shift_bound=anchor_score_shift_bound,
        initial_law_receipt_sha256=initial_law_receipt_sha256,
        anchor_shift_receipt_sha256=anchor_shift_receipt_sha256,
        score_schedule_receipt_sha256=score_schedule_receipt_sha256,
        route_receipts={
            "variance_receipt_sha256": variance_receipt_sha256,
            "k_ps_receipt_sha256": k_ps_receipt_sha256,
        },
        dependence_method="markov_time_varying_bernstein_eq_3_26",
        lcb_scope=("conditional on reviewed score schedule, chain/gap, residue variance, "
                   "initial-law TV, and anchor-shift assumptions"),
        radius_builder=radius_builder,
    )


def markov_time_varying_score_lcb_from_gap_bound(
    planned_scores: object,
    *,
    valid_blocks: object,
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
    initial_law_receipt_sha256: str,
    anchor_shift_receipt_sha256: str,
    score_schedule_receipt_sha256: str,
) -> dict:
    """Compute a conditional LCB from a gap bound and ``f_i in [-1, 1]``.

    This route removes the exact maximizing-index and residue-variance
    receipts, but still requires independent evidence for the pseudo-gap lower
    bound, chain conditions, score schedule, initial law, and anchor shift.
    """
    def radius_builder(blocks: int, family: int, alpha: float) -> dict:
        return time_varying_markov_radius_from_gap_bound(
            blocks=blocks,
            gamma_ps_lower=gamma_ps_lower,
            family=family,
            alpha_stationary=alpha,
        )

    return _markov_time_varying_score_lcb_core(
        planned_scores,
        valid_blocks=valid_blocks,
        gamma_ps_lower=gamma_ps_lower,
        family=family,
        alpha_score=alpha_score,
        alpha_anchor=alpha_anchor,
        alpha_total=alpha_total,
        initial_law_mode=initial_law_mode,
        burn_in_blocks=burn_in_blocks,
        initial_law_tv_bound=initial_law_tv_bound,
        anchor_score_shift_bound=anchor_score_shift_bound,
        initial_law_receipt_sha256=initial_law_receipt_sha256,
        anchor_shift_receipt_sha256=anchor_shift_receipt_sha256,
        score_schedule_receipt_sha256=score_schedule_receipt_sha256,
        route_receipts={
            "gamma_ps_lower_receipt_sha256": gamma_ps_lower_receipt_sha256,
        },
        dependence_method="markov_time_varying_support_gap_bernstein_eq_3_26",
        lcb_scope=("conditional on reviewed score schedule, support, pseudo-gap lower bound, "
                   "chain assumptions, initial-law TV, and anchor-shift assumptions"),
        radius_builder=radius_builder,
    )
def markov_time_varying_score_lcb(
    planned_scores: object,
    *,
    valid_blocks: object,
    gamma_ps_lower: float,
    k_ps: int,
    variance_upper_by_residue: object,
    family: int,
    alpha_score: float,
    alpha_anchor: float,
    alpha_total: float,
    initial_law_mode: str,
    burn_in_blocks: int,
    initial_law_tv_bound: float,
    anchor_score_shift_bound: float,
    initial_law_receipt_sha256: str,
    anchor_shift_receipt_sha256: str,
    score_schedule_receipt_sha256: str,
    variance_receipt_sha256: str,
    k_ps_receipt_sha256: str,
) -> dict:
    """Compute a conditional LCB from already-coded block-specific scores.

    Each planned score must be the value of its predeclared ``f_i(X_i)`` in
    ``[-1, 1]``. Failures remain in planned N and are coded as -1. The supplied
    ``anchor_score_shift_bound`` is in normalized score units and is subtracted
    without interpretation; for a physical bounded-block-risk use, an audited
    derivation must link it to the accepted anchor contract. The stationary
    start or post-burn-in TV receipt, plus the schedule, variance, and k_ps
    receipts, are required inputs. This routine only reports a conditional
    mathematical bound and never a physical score.
    """
    from .bounded_block_risk_markov_gap import _validate_alpha_allocation

    raw_values = np.asarray(planned_scores, dtype=object)
    valid = np.asarray(valid_blocks)
    if raw_values.ndim != 1 or raw_values.size == 0:
        raise ValueError("planned_scores must be a nonempty one-dimensional array")
    if valid.ndim != 1 or valid.shape != raw_values.shape or valid.dtype != np.bool_:
        raise ValueError("valid_blocks must be a matching boolean vector")
    if any(not _finite_real(value) for value in raw_values[valid]):
        raise ValueError("valid planned scores must be finite non-boolean real numbers")
    values = np.full(raw_values.size, np.nan, dtype=float)
    values[valid] = np.asarray(raw_values[valid], dtype=float)
    if np.any(values[valid] < -1.0) or np.any(values[valid] > 1.0):
        raise ValueError("valid planned scores must lie in [-1, 1]")
    receipts = (score_schedule_receipt_sha256, variance_receipt_sha256,
                k_ps_receipt_sha256, initial_law_receipt_sha256,
                anchor_shift_receipt_sha256)
    if not all(isinstance(receipt, str) and len(receipt) == 64
               and all(char in "0123456789abcdef" for char in receipt)
               for receipt in receipts):
        raise ValueError("score schedule, variance, k_ps, initial-law, and anchor-shift receipts must be SHA-256 hex strings")
    _validate_alpha_allocation(
        alpha_score=alpha_score, alpha_anchor=alpha_anchor,
        alpha_total=alpha_total,
    )
    if not _finite_real(initial_law_tv_bound) or not 0 <= initial_law_tv_bound <= 1:
        raise ValueError("initial_law_tv_bound must be finite and in [0, 1]")
    if (not isinstance(initial_law_mode, str)
            or initial_law_mode not in {"stationary", "post_burn_in_tv"}):
        raise ValueError("initial_law_mode must be stationary or post_burn_in_tv")
    burn_in_blocks = (burn_in_blocks if isinstance(burn_in_blocks, int)
                      and not isinstance(burn_in_blocks, bool) else -1)
    if burn_in_blocks < 0:
        raise ValueError("burn_in_blocks must be a nonnegative integer")
    if initial_law_mode == "stationary" and (
            burn_in_blocks != 0 or initial_law_tv_bound != 0):
        raise ValueError("stationary initial law requires zero burn-in and zero TV bound")
    if not _finite_real(anchor_score_shift_bound) or anchor_score_shift_bound < 0:
        raise ValueError("anchor_score_shift_bound must be finite and nonnegative")

    planned = int(values.size)
    family = _positive_integer(family, "family")
    try:
        family_tv_cost = (0.0 if initial_law_tv_bound == 0
                          else family * float(initial_law_tv_bound))
    except OverflowError:
        family_tv_cost = math.inf
    if not math.isfinite(family_tv_cost) or family_tv_cost >= float(alpha_score):
        raise ValueError("family * initial_law_tv_bound must be smaller than alpha_score")
    stationary_alpha = float(alpha_score) - family_tv_cost
    if not math.isfinite(stationary_alpha) or stationary_alpha <= 0:
        raise ValueError("family * initial_law_tv_bound must be smaller than alpha_score")
    radius_details = time_varying_markov_radius(
        blocks=planned,
        gamma_ps_lower=gamma_ps_lower,
        k_ps=k_ps,
        variance_upper_by_residue=variance_upper_by_residue,
        family=family,
        alpha_stationary=stationary_alpha,
    )

    scores = np.full(planned, -1.0, dtype=float)
    scores[valid] = values[valid]
    valid_count = int(valid.sum())
    failed_count = planned - valid_count
    denominators = {
        "planned": planned,
        "valid": valid_count,
        "failed": failed_count,
        "abstained": 0,
    }
    result_denominators = {
        "planned": 1,
        "valid": int(valid_count > 0),
        "failed": 0,
        "abstained": int(valid_count == 0),
    }
    common = {
        "estimand": "mean of predeclared time-varying bounded block scores",
        "dependence_method": "markov_time_varying_bernstein_eq_3_26",
        "status": "ABSTAIN_NO_VALID_BLOCKS" if valid_count == 0 else "CONDITIONAL_BOUND_ONLY",
        "certified": False,
        "physical_score": None,
        "data_access": "none",
        "denominators": denominators,
        "result_denominators": result_denominators,
        "coded_scores": scores.tolist(),
        "planned_mean": None if valid_count == 0 else float(scores.mean()),
        "radius": radius_details["radius"],
        "initial_law_tv_bound": float(initial_law_tv_bound),
        "initial_law_mode": initial_law_mode,
        "burn_in_blocks": burn_in_blocks,
        "initial_law_receipt_sha256": initial_law_receipt_sha256,
        "familywise_tv_cost": family_tv_cost,
        "stationary_alpha": stationary_alpha,
        "initial_law_mean_shift_penalty": 2.0 * float(initial_law_tv_bound),
        "anchor_score_shift_bound": float(anchor_score_shift_bound),
        "anchor_shift_receipt_sha256": anchor_shift_receipt_sha256,
        "lcb_scope": "conditional on all cited score-map, chain, variance, gap, TV, and anchor-shift assumptions",
        "variance_receipt_sha256": variance_receipt_sha256,
        "score_schedule_receipt_sha256": score_schedule_receipt_sha256,
        "k_ps_receipt_sha256": k_ps_receipt_sha256,
        **radius_details,
    }
    if valid_count == 0:
        return {**common, "normalized_lower_bound": None}
    lower_bound = (
        float(scores.mean()) - radius_details["radius"]
        - 2.0 * float(initial_law_tv_bound)
        - float(anchor_score_shift_bound)
    )
    if not math.isfinite(lower_bound):
        raise ValueError("time-varying Markov LCB is outside finite floating-point range")
    return {**common, "normalized_lower_bound": lower_bound}
