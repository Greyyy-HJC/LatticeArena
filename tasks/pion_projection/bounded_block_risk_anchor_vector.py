"""Conditional simultaneous Markov confidence boxes for vector anchor means.

This is an optional bridge for a nonlinear ensemble-level fit: it bounds the
mean of each predeclared bounded per-state feature, but does not claim that a
nonlinear fitted mass is itself a per-state scalar. A separate certified
range-propagation step is still required to map the vector box to that fit.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np

from .bounded_block_risk_markov_gap import markov_gap_radius


def _finite_real(value: object) -> bool:
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def markov_anchor_mean_box(
    samples: object,
    *,
    support_bounds: object,
    gamma_ps_lower: float,
    alpha_anchor: float,
    initial_law_tv_bound: float = 0.0,
    burn_in_units: int = 0,
    stationary_start: bool = True,
) -> dict[str, Any]:
    """Bound a vector of anchor-chain feature means simultaneously.

    ``samples`` has shape ``(n_anchor_states, n_features)``; all columns must
    be fixed measurable functions of the same chain state at every retained
    post-burn-in state. ``support_bounds[j]`` is a pre-established finite
    interval for feature ``j``. A common pseudo-gap lower bound applies to the
    same anchor chain for all coordinates.

    The corrected scalar Bernstein inequality is applied componentwise with
    family size ``n_features``. A single path-level initial-law TV budget is
    subtracted once from ``alpha_anchor`` before the union bound. The result
    covers the vector mean; it does not propagate the box through a nonlinear
    fit or establish the required correspondence to ``m_star``.
    """
    raw = np.asarray(samples)
    if raw.ndim != 2 or min(raw.shape) < 1:
        raise ValueError("samples must have shape (n_anchor_states, n_features)")
    if np.iscomplexobj(raw):
        if np.any(np.imag(raw) != 0):
            raise ValueError("samples must be real-valued features")
        raw = np.real(raw)
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError("samples must be numeric")
    try:
        values = raw.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("samples must be finite real numbers") from exc
    if not np.isfinite(values).all():
        raise ValueError("samples must be finite real numbers")

    raw_bounds = np.asarray(support_bounds)
    if (raw_bounds.shape != (values.shape[1], 2)
            or np.iscomplexobj(raw_bounds)
            or not np.issubdtype(raw_bounds.dtype, np.number)):
        raise ValueError("support_bounds must have shape (n_features, 2) and be real numeric")
    bounds = raw_bounds.astype(float, copy=False)
    if (not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError("each support interval must be finite and increasing")

    if np.any(values < bounds[:, 0]) or np.any(values > bounds[:, 1]):
        raise ValueError("samples fall outside the predeclared feature support")
    if not _finite_real(gamma_ps_lower) or not 0 < gamma_ps_lower <= 1:
        raise ValueError("gamma_ps_lower must be finite and in (0, 1]")
    if not _finite_real(alpha_anchor) or not 0 < alpha_anchor < 1:
        raise ValueError("alpha_anchor must be finite and in (0, 1)")
    if (not _finite_real(initial_law_tv_bound)
            or not 0 <= initial_law_tv_bound < alpha_anchor):
        raise ValueError("initial_law_tv_bound must be in [0, alpha_anchor)")
    if (isinstance(burn_in_units, (bool, np.bool_))
            or not isinstance(burn_in_units, (int, np.integer))
            or burn_in_units < 0):
        raise ValueError("burn_in_units must be a nonnegative integer")
    if not isinstance(stationary_start, (bool, np.bool_)):
        raise ValueError("stationary_start must be boolean")
    if stationary_start and (burn_in_units != 0 or initial_law_tv_bound != 0):
        raise ValueError("stationary start requires zero burn-in and zero TV bound")
    if not stationary_start and burn_in_units < 1:
        raise ValueError("post-burn-in start requires a positive burn-in count")

    n_states, n_features = values.shape
    alpha_stationary = float(alpha_anchor) - float(initial_law_tv_bound)
    means = values.mean(axis=0)
    feature_bounds = []
    for j in range(n_features):
        lower, upper = map(float, bounds[j])
        width = upper - lower
        raw_radius = markov_gap_radius(
            blocks=n_states,
            gamma_ps_lower=float(gamma_ps_lower),
            family=n_features,
            alpha_s=alpha_stationary,
            variance_bound=width * width / 4.0,
            centered_abs_bound=width,
        )
        radius = min(width, raw_radius)
        feature_bounds.append({
            "feature_index": j,
            "support": [lower, upper],
            "sample_mean": float(means[j]),
            "raw_bernstein_radius": float(raw_radius),
            "capped_radius": float(radius),
            "confidence_interval": [
                max(lower, float(means[j]) - radius),
                min(upper, float(means[j]) + radius),
            ],
            "coordinate_family_size": n_features,
            "stationary_alpha": alpha_stationary,
        })

    return {
        "status": "CONDITIONAL_MARKOV_VECTOR_MEAN_BOX_ONLY",
        "method": "componentwise corrected Markov Bernstein with a Bonferroni union bound",
        "sample_count_n_A": n_states,
        "feature_count": n_features,
        "gamma_ps_lower_assumed": float(gamma_ps_lower),
        "alpha_anchor": float(alpha_anchor),
        "initial_law_tv_bound": float(initial_law_tv_bound),
        "stationary_alpha_after_tv": alpha_stationary,
        "burn_in_units": int(burn_in_units),
        "stationary_start": bool(stationary_start),
        "feature_intervals": feature_bounds,
        "denominators": {
            "vector_box": {"planned": 1, "valid": 1, "failed": 0, "abstained": 0},
            "feature_coordinates": {
                "planned": n_features, "valid": n_features,
                "failed": 0, "abstained": 0,
            },
        },
        "joint_coverage_probability_lower_bound": 1.0 - float(alpha_anchor),
        "coverage_status": "conditional on all stated chain, support and receipt assumptions",
        "fit_target_interval": None,
        "common_bias_bound_C": None,
        "certified": False,
        "physical_score": None,
        "data_access": "in_memory_argument_only",
        "required_assumptions": [
            "each feature is a fixed measurable scalar of the same anchor-chain state",
            "all retained rows use the same time-homogeneous anchor chain",
            "each feature support is established independently of the observed sample values",
            "gamma_ps_lower is justified for this anchor chain",
            "stationary start or the post-burn-in path law has the declared TV bound",
            "alpha_anchor covers the single joint vector box and is not reused as per-coordinate alpha",
            "a separately certified fit-range map is required to connect this box to m_star",
        ],
    }
