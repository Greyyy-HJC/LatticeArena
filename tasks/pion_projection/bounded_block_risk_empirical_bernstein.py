"""Color-class empirical-Bernstein LCB for bounded block-risk scores.

This is an optional conditional bound, not a replacement for a valid
score-specific dependency graph. Color classes must be jointly independent;
the function cannot establish that condition from observed scores.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def color_class_empirical_bernstein_lcb(
    scores: object,
    *,
    color_labels: object,
    alpha_score: float,
) -> dict[str, Any]:
    """Return simultaneous lower bounds for finitely many bounded score means.

    ``scores`` has shape ``(..., F, N)``: optional replicate axes, then F
    candidate scores and N planned blocks. ``color_labels`` has length N and
    fixes a proper dependency-graph coloring before observing score values.
    Within each color class, block scores must be jointly independent and lie
    in [-1, 1]. No independence is assumed across color classes or candidates.

    For every family/color pair, the one-sided empirical-Bernstein bound of
    Maurer & Pontil (2009, Theorem 11) is applied after mapping [-1,1] to [0,1],
    with delta = alpha_score/(F*chi). Color-class LCBs are averaged with their
    planned-block weights, producing a bound on the finite planned mean of
    expectations. Singleton color classes use the corresponding Hoeffding
    bound because the empirical variance bound requires n >= 2.
    """
    values = np.asarray(scores, dtype=float)
    labels = np.asarray(color_labels)
    if values.ndim < 2 or values.shape[-1] == 0 or values.shape[-2] == 0:
        raise ValueError("scores must have shape (..., F, N) with F,N positive")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    if np.any((values < -1.0) | (values > 1.0)):
        raise ValueError("scores must lie in [-1, 1]")
    blocks = values.shape[-1]
    family = values.shape[-2]
    if labels.shape != (blocks,):
        raise ValueError("color_labels must have one fixed label per planned block")
    if labels.dtype.kind not in "iu" or labels.dtype.kind == "b":
        raise ValueError("color_labels must be integer color IDs")
    if np.any(labels < 0):
        raise ValueError("color IDs must be nonnegative")
    if (not math.isfinite(alpha_score) or not 0 < alpha_score < 1):
        raise ValueError("alpha_score must be finite and in (0, 1)")

    classes = np.unique(labels)
    chi = int(classes.size)
    delta = alpha_score / (family * chi)
    log_term = math.log(2.0 / delta)
    means = []
    variances = []
    radii = []
    class_lcbs = []
    class_sizes = []
    class_methods = []
    for color in classes:
        indices = np.flatnonzero(labels == color)
        class_scores = values[..., indices]
        n_color = int(indices.size)
        mean = class_scores.mean(axis=-1)
        if n_color >= 2:
            variance = class_scores.var(axis=-1, ddof=1)
            # Scaling the [0,1] Theorem 11 to [-1,1] doubles both terms.
            radius = (np.sqrt(2.0 * variance * log_term / n_color)
                      + 14.0 * log_term / (3.0 * (n_color - 1)))
            method = "empirical_bernstein"
        else:
            variance = np.zeros_like(mean)
            # Hoeffding for range length two, with the same class/family delta.
            radius = np.full_like(
                mean, math.sqrt(2.0 * math.log(1.0 / delta) / n_color),
            )
            method = "hoeffding_singleton"
        lcb = np.maximum(-1.0, mean - radius)
        means.append(mean)
        variances.append(variance)
        radii.append(radius)
        class_lcbs.append(lcb)
        class_sizes.append(n_color)
        class_methods.append(method)

    class_means = np.stack(means, axis=-1)
    class_variances = np.stack(variances, axis=-1)
    class_radii = np.stack(radii, axis=-1)
    class_lower_bounds = np.stack(class_lcbs, axis=-1)
    weights = np.asarray(class_sizes, dtype=float) / blocks
    combined = np.sum(class_lower_bounds * weights, axis=-1)
    empirical_mean = values.mean(axis=-1)
    return {
        "status": "CONDITIONAL_BOUND_ONLY",
        "physical_score": None,
        "planned_blocks": blocks,
        "family_size": family,
        "color_class_count": chi,
        "alpha_score": float(alpha_score),
        "per_family_color_delta": delta,
        "empirical_mean": empirical_mean,
        "color_class_sizes": class_sizes,
        "color_class_methods": class_methods,
        "color_class_means": class_means,
        "color_class_sample_variances": class_variances,
        "color_class_radii": class_radii,
        "color_class_lower_bounds": class_lower_bounds,
        "simultaneous_lower_bounds": combined,
        "assumptions": [
            "color labels are a proper coloring of a predeclared score-specific dependency graph",
            "block scores within each color class are jointly independent",
            "all planned block scores, including predeclared failure codes, lie in [-1,1]",
            "candidate family is fixed before confirmation and covered by the finite-family union bound",
        ],
    }
