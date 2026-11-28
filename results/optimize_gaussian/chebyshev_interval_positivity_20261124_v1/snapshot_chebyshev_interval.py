"""Numerically inspect a real Chebyshev polynomial on a closed interval.

The minimum candidates are the interval endpoints and all real stationary
points found from the derivative polynomial. This is a diagnostic based on
floating-point root finding; it is not an interval-arithmetic certificate.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def chebyshev_interval_minimum(
    coefficients: object,
    interval: object = (-1.0, 1.0),
    *,
    root_imag_tolerance: float = 1e-10,
    boundary_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Return the smallest evaluated value and its candidate point.

    ``coefficients[k]`` multiplies the degree-k Chebyshev polynomial. Only
    real-valued coefficient vectors are accepted. All returned candidate
    points are either endpoints or derivative roots inside the interval.
    """
    raw = np.asarray(coefficients)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("coefficients must be a nonempty one-dimensional array")
    if np.iscomplexobj(raw):
        if np.any(np.imag(raw) != 0):
            raise ValueError("coefficients must be real")
        raw = np.real(raw)
    try:
        coeff = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("coefficients must be finite real numbers") from exc
    if not np.all(np.isfinite(coeff)):
        raise ValueError("coefficients must be finite real numbers")

    bounds = np.asarray(interval, dtype=float)
    if (bounds.shape != (2,) or not np.all(np.isfinite(bounds))
            or bounds[0] >= bounds[1]):
        raise ValueError("interval must contain two finite increasing endpoints")
    if (not math.isfinite(root_imag_tolerance) or root_imag_tolerance < 0
            or not math.isfinite(boundary_tolerance) or boundary_tolerance < 0):
        raise ValueError("root and boundary tolerances must be finite and nonnegative")

    lower, upper = map(float, bounds)
    points: list[dict[str, float | str]] = [
        {"kind": "lower_endpoint", "x": lower},
        {"kind": "upper_endpoint", "x": upper},
    ]
    derivative = np.polynomial.chebyshev.chebder(coeff)
    derivative = np.trim_zeros(derivative, trim="b")
    if derivative.size > 1:
        roots = np.polynomial.chebyshev.chebroots(derivative)
        if not np.all(np.isfinite(roots)):
            raise FloatingPointError("derivative root calculation returned non-finite roots")
        for root in roots:
            real = float(np.real(root))
            imag = float(np.imag(root))
            if abs(imag) > root_imag_tolerance * max(1.0, abs(real)):
                continue
            if lower - boundary_tolerance <= real <= upper + boundary_tolerance:
                x = min(max(real, lower), upper)
                points.append({"kind": "stationary_point", "x": x})

    xs = np.asarray([point["x"] for point in points], dtype=float)
    values = np.polynomial.chebyshev.chebval(xs, coeff)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("polynomial evaluation returned non-finite values")
    for point, value in zip(points, values):
        point["value"] = float(value)
    index = int(np.argmin(values))
    return {
        "minimum": float(values[index]),
        "argmin": float(xs[index]),
        "candidate_points": points,
        "method": "endpoints_and_real_derivative_roots",
        "numerically_certified": False,
    }
