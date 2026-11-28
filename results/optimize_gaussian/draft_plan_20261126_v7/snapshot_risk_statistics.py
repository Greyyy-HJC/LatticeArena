"""Fail-closed array reducers for the selected Optimize D04/D06 statistics.

These functions do not fit correlators, generate bootstrap samples, access
gauge data, launch solvers, or certify a physical score. They reduce arrays
prepared by a separately reviewed evaluator and preserve planned denominators.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import numpy as np


D04_TIMES = (2, 3, 4, 5)
D04_VARIANCE_DDOF = 1
D04_PLANNED_BOOTSTRAP_DRAWS = 10_000
D06_BLOCK_LENGTHS = (2, 5, 10)
D06_PLANNED_DRAWS_PER_LENGTH = 10_000
D06_R_STAR = 10.0 / 9.0
D06_ALPHA_PER_COMPARISON = 0.025
D06_QUANTILE_METHOD = "linear"


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _number(value: object) -> bool:
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and math.isfinite(float(value)))


def _denominators(planned: int, valid: int, failed: int, abstained: int = 0) -> dict:
    if planned != valid + failed:
        raise AssertionError("draw denominator does not balance")
    return {"planned": planned, "valid": valid, "failed": failed,
            "abstained": abstained}


def d04_bootstrap_risk(q_draws: object, *, planned_draws: int = D04_PLANNED_BOOTSTRAP_DRAWS) -> dict:
    """Reduce the D04 bootstrap q draws to ``R_theta``.

    ``q_draws[b, j]`` must already equal ``C_theta(t_j)/(A_theta*f0(t_j))``
    for the same bootstrap draw ``b`` and ordered ``t_j=(2,3,4,5)``. The
    variance uses the sample convention ``ddof=1``. Missing/nonfinite rows are
    retained as failed planned draws; any failure abstains the risk instead of
    silently changing the bootstrap distribution.
    """
    planned = _positive_int(planned_draws, "planned_draws")
    if planned < 2:
        raise ValueError("planned_draws must be at least 2 for sample variance")
    raw = np.asarray(q_draws)
    if raw.ndim != 2 or raw.shape[1] != len(D04_TIMES):
        raise ValueError("q_draws must have shape (draws, 4) ordered by t=(2,3,4,5)")
    if raw.dtype.kind not in "iuf":
        raise ValueError("q_draws must be a real numeric array")
    values = np.asarray(raw, dtype=np.float64)
    observed = int(values.shape[0])
    if observed > planned:
        raise ValueError("observed q draws exceed the planned denominator")
    finite_rows = np.all(np.isfinite(values), axis=1)
    valid = int(finite_rows.sum())
    failed = planned - valid
    draw_denominators = _denominators(planned, valid, failed)
    result_denominators = {"planned": 1, "valid": 0, "failed": 0, "abstained": 1}
    common = {
        "schema": "optimize-d04-bootstrap-risk-v1",
        "status": "ABSTAIN_INCOMPLETE_OR_INVALID_Q_DRAWS",
        "times": list(D04_TIMES),
        "variance_ddof": D04_VARIANCE_DDOF,
        "draw_denominators": draw_denominators,
        "result_denominators": result_denominators,
        "risk": None,
        "per_time": None,
        "physical_score": None,
        "certified": False,
        "data_access": "caller_supplied_arrays_only",
    }
    if valid != planned or observed != planned:
        return common

    with np.errstate(over="ignore", invalid="ignore"):
        means = np.mean(values, axis=0)
        variances = np.var(values, axis=0, ddof=D04_VARIANCE_DDOF)
        bias_squared = np.square(means - 1.0)
        contributions = variances + bias_squared
        risk = float(np.mean(contributions))
    if not math.isfinite(risk) or risk < 0:
        return {**common, "status": "ABSTAIN_NONFINITE_D04_RISK_RESULT"}
    per_time = [
        {
            "t": time,
            "bootstrap_mean_q": float(means[index]),
            "bootstrap_variance_q": float(variances[index]),
            "squared_bias_from_one": float(bias_squared[index]),
            "risk_contribution": float(contributions[index]),
        }
        for index, time in enumerate(D04_TIMES)
    ]
    return {
        **common,
        "status": "D04_RISK_REDUCED_ARRAY_ONLY",
        "result_denominators": {"planned": 1, "valid": 1, "failed": 0, "abstained": 0},
        "risk": risk,
        "per_time": per_time,
    }


def _ids_array(value: object, expected: int) -> np.ndarray | None:
    ids = np.asarray(value)
    if (ids.ndim != 1 or ids.dtype.kind not in "iu"
            or ids.size != expected or np.unique(ids).size != expected):
        return None
    if ids.dtype.kind == "u" and np.any(ids > np.iinfo(np.int64).max):
        return None
    return ids.astype(np.int64, copy=False)


def d06_paired_ratio_hull(
    comparator_risks: Mapping[int, object],
    trainable_w_risks: Mapping[int, object],
    comparator_draw_ids: Mapping[int, object],
    trainable_w_draw_ids: Mapping[int, object],
    *,
    planned_draws_per_length: int = D06_PLANNED_DRAWS_PER_LENGTH,
    alpha_per_comparison: float = D06_ALPHA_PER_COMPARISON,
    r_star: float = D06_R_STAR,
) -> dict:
    """Compute the D06 one-sided lower-quantile hull for paired risk arrays.

    Each input risk value is assumed to be an upstream, per-draw D04 risk
    estimate. For each block length, both arms must provide the same unique
    integer draw IDs in the same order and all risk values must be finite and
    strictly positive. The function never drops bad draws or adds an epsilon.
    """
    planned_each = _positive_int(planned_draws_per_length, "planned_draws_per_length")
    if (not _number(alpha_per_comparison)
            or float(alpha_per_comparison) != D06_ALPHA_PER_COMPARISON):
        raise ValueError("alpha_per_comparison is fixed by the selected F=2 contract at 0.025")
    if not _number(r_star) or float(r_star) != D06_R_STAR:
        raise ValueError("r_star is fixed by the selected contract at 10/9")

    planned_total = planned_each * len(D06_BLOCK_LENGTHS)
    valid_total = 0
    block_results = {}
    valid_inputs = True
    for length in D06_BLOCK_LENGTHS:
        if not all(length in mapping for mapping in (
                comparator_risks, trainable_w_risks,
                comparator_draw_ids, trainable_w_draw_ids)):
            valid_inputs = False
            block_results[str(length)] = {
                "status": "ABSTAIN_MISSING_BLOCK_LENGTH_INPUT",
                "planned": planned_each, "valid": 0, "failed": planned_each,
                "lower_quantile": None, "median_ratio": None,
                "draw_id_sha256": None,
            }
            continue

        comp = np.asarray(comparator_risks[length])
        cand = np.asarray(trainable_w_risks[length])
        comp_ids = _ids_array(comparator_draw_ids[length], planned_each)
        cand_ids = _ids_array(trainable_w_draw_ids[length], planned_each)
        shape_ok = (comp.ndim == cand.ndim == 1
                    and comp.size == planned_each and cand.size == planned_each
                    and comp.dtype.kind in "iuf" and cand.dtype.kind in "iuf")
        ids_ok = (comp_ids is not None and cand_ids is not None
                  and np.array_equal(comp_ids, cand_ids))
        if not shape_ok or not ids_ok:
            valid_inputs = False
            block_results[str(length)] = {
                "status": "ABSTAIN_UNPAIRED_OR_MALFORMED_DRAWS",
                "planned": planned_each, "valid": 0, "failed": planned_each,
                "lower_quantile": None, "median_ratio": None,
                "draw_id_sha256": None,
            }
            continue

        comp = np.asarray(comp, dtype=np.float64)
        cand = np.asarray(cand, dtype=np.float64)
        valid_rows = (np.isfinite(comp) & np.isfinite(cand)
                      & (comp > 0) & (cand > 0))
        n_valid = int(valid_rows.sum())
        valid_total += n_valid
        if n_valid != planned_each:
            valid_inputs = False
            block_results[str(length)] = {
                "status": "ABSTAIN_NONPOSITIVE_OR_NONFINITE_RISK",
                "planned": planned_each, "valid": n_valid,
                "failed": planned_each - n_valid,
                "lower_quantile": None, "median_ratio": None,
                "draw_id_sha256": hashlib.sha256(
                    comp_ids.astype("<i8", copy=False).tobytes()
                ).hexdigest(),
            }
            continue

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            ratio = comp / cand
        if not np.all(np.isfinite(ratio)):
            valid_inputs = False
            block_results[str(length)] = {
                "status": "ABSTAIN_NONFINITE_RISK_RATIO",
                "planned": planned_each, "valid": 0, "failed": planned_each,
                "lower_quantile": None, "median_ratio": None,
                "draw_id_sha256": hashlib.sha256(
                    comp_ids.astype("<i8", copy=False).tobytes()
                ).hexdigest(),
            }
            valid_total -= planned_each
            continue
        lower = float(np.quantile(
            ratio, float(alpha_per_comparison), method=D06_QUANTILE_METHOD,
        ))
        block_results[str(length)] = {
            "status": "VALID_PAIRED_RATIO_DRAWS",
            "planned": planned_each, "valid": planned_each, "failed": 0,
            "lower_quantile": lower,
            "median_ratio": float(np.median(ratio)),
            "draw_id_sha256": hashlib.sha256(
                comp_ids.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
        }

    draw_denominators = _denominators(
        planned_total, valid_total, planned_total - valid_total,
    )
    if not valid_inputs or valid_total != planned_total:
        return {
            "schema": "optimize-d06-paired-risk-ratio-v1",
            "status": "ABSTAIN_INVALID_OR_UNPAIRED_RISK_DRAWS",
            "block_lengths": list(D06_BLOCK_LENGTHS),
            "alpha_per_comparison": float(alpha_per_comparison),
            "quantile_method": D06_QUANTILE_METHOD,
            "r_star": float(r_star),
            "block_results": block_results,
            "hull_lower": None,
            "draw_denominators": draw_denominators,
            "result_denominators": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
            "physical_score": None,
            "certified": False,
            "data_access": "caller_supplied_arrays_only",
        }

    hull_lower = min(block_results[str(length)]["lower_quantile"]
                     for length in D06_BLOCK_LENGTHS)
    cleared = hull_lower > float(r_star)
    return {
        "schema": "optimize-d06-paired-risk-ratio-v1",
        "status": "D06_THRESHOLD_CLEARED_ARRAY_ONLY" if cleared else "D06_THRESHOLD_NOT_CLEARED",
        "block_lengths": list(D06_BLOCK_LENGTHS),
        "alpha_per_comparison": float(alpha_per_comparison),
        "quantile_method": D06_QUANTILE_METHOD,
        "r_star": float(r_star),
        "block_results": block_results,
        "hull_lower": hull_lower,
        "draw_denominators": draw_denominators,
        "result_denominators": {"planned": 1, "valid": 1, "failed": 0, "abstained": 0},
        "physical_score": None,
        "certified": False,
        "data_access": "caller_supplied_arrays_only",
    }
