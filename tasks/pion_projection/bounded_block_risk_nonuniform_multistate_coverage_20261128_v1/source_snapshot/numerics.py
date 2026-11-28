"""Strict numeric conversion helpers shared by task-specific evaluators.

These helpers validate Python/NumPy input types and float64 representability.
They do not apply task units, support intervals, clipping, or physical meaning;
those checks remain in the owning task module.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np


def as_real_float_array(value: object, *, name: str) -> np.ndarray:
    """Convert an integer/float array to float64 without lossy coercions.

    Complex, boolean, object, and string arrays are refused so callers cannot
    silently project imaginary parts or parse text as measured values. Integer
    magnitudes above ``2**53`` are also refused because float64 cannot preserve
    every integer there. NaN and infinity are preserved for the task layer to
    handle according to its failure-mask and denominator policy.
    """
    try:
        raw = np.asarray(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular real numeric array") from exc
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be an integer or floating-point array")
    if raw.dtype.kind in "iu" and raw.size:
        exact_integer_limit = 1 << 53
        if raw.dtype.kind == "u":
            if int(raw.max()) > exact_integer_limit:
                raise ValueError(f"{name} contains an integer not exactly representable as float64")
        elif int(raw.min()) < -exact_integer_limit or int(raw.max()) > exact_integer_limit:
            raise ValueError(f"{name} contains an integer not exactly representable as float64")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.asarray(raw, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} cannot be represented as float64") from exc


def as_finite_real(value: object, *, name: str) -> float:
    """Return one finite real scalar, rejecting bool and implicit text casts."""
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number")
    if isinstance(value, Integral) and abs(int(value)) > (1 << 53):
        raise ValueError(f"{name} integer is not exactly representable as float64")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result
