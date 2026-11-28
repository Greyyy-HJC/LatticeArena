"""Conservative propagation of finite-Nt q/h/x envelopes into a force bound.

The helper is algebra-only. It accepts *assumed or independently measured*
nonnegative loop-level bounds and never reads Wilson data or certifies a lane.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from core.numerics import as_finite_real


COMPONENT_FRACTIONS = {"q": 0.015, "h": 0.015, "x": 0.01}
FORCE_CORNER_COUNT = 8  # 2 kernels x (R=2,3) x (t,t+1)
ABSOLUTE_FORCE_CORNER_COUNT = 4  # stout kernel x (R=2,3) x (t,t+1)


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    result = as_finite_real(value, name=name)
    invalid = result < 0 if allow_zero else result <= 0
    if invalid:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _flatten_corner_bounds(values: Sequence[Sequence[float]], name: str) -> list[float]:
    if isinstance(values, (str, bytes)) or len(values) != 2:
        raise ValueError(f"{name} must contain two kernels")
    flattened: list[float] = []
    for kernel_index, row in enumerate(values):
        if isinstance(row, (str, bytes)) or len(row) != 4:
            raise ValueError(f"{name}[{kernel_index}] must contain four loop corners")
        for corner_index, value in enumerate(row):
            flattened.append(_positive(
                value, f"{name}[{kernel_index}][{corner_index}]", allow_zero=True
            ))
    return flattened


def uniform_loop_cap(force_fraction: float, reference_force_lower: float) -> float:
    """Maximum equal per-corner q, h, or x bound for one force budget.

    If a component has the same upper bound `c` at all eight kernel/R/time
    corners, its propagated force envelope is `8*log1p(c)`. Solving
    `8*log1p(c) <= force_fraction*reference_force_lower` gives this cap.
    """
    fraction = _positive(force_fraction, "force_fraction", allow_zero=True)
    reference = _positive(reference_force_lower, "reference_force_lower")
    return math.expm1(fraction * reference / FORCE_CORNER_COUNT)


def _flatten_absolute_corner_bounds(values: Sequence[float], name: str) -> list[float]:
    if isinstance(values, (str, bytes)) or len(values) != ABSOLUTE_FORCE_CORNER_COUNT:
        raise ValueError(f"{name} must contain four stout loop corners")
    return [
        _positive(value, f"{name}[{corner_index}]", allow_zero=True)
        for corner_index, value in enumerate(values)
    ]


def uniform_absolute_loop_cap(force_fraction: float,
                              positive_force_scale_lower: float) -> float:
    """Equal per-corner cap for one absolute stout-force component budget."""
    fraction = _positive(force_fraction, "force_fraction", allow_zero=True)
    reference = _positive(positive_force_scale_lower, "positive_force_scale_lower")
    return math.expm1(fraction * reference / ABSOLUTE_FORCE_CORNER_COUNT)


def propagate_absolute_force_budget(
    qhx_bounds: Mapping[str, Sequence[float]],
    positive_force_scale_lower: float,
    *,
    model_truncation_bound: float = 0.0,
    statistical_halfwidth: float = 0.0,
    component_fractions: Mapping[str, float] | None = None,
    model_fraction: float = 0.01,
    statistical_fraction: float = 0.05,
    total_fraction: float = 0.10,
) -> dict:
    """Propagate q/h/x through the four stout corners of the absolute force.

    The primary force is ``U0_stout(R+1)-U0_stout(R)``. Identity-kernel
    corners belong to the separate shared-spectrum consistency diagnostic and
    must not double the primary uncertainty by entering this sum.
    """
    if not isinstance(qhx_bounds, Mapping) or set(qhx_bounds) != {"q", "h", "x"}:
        raise ValueError("qhx_bounds must contain exactly q, h, and x")
    reference = _positive(positive_force_scale_lower,
                          "positive_force_scale_lower")
    qhx = {name: _flatten_absolute_corner_bounds(qhx_bounds[name], name)
           for name in ("q", "h", "x")}
    fractions = dict(COMPONENT_FRACTIONS if component_fractions is None
                     else component_fractions)
    if set(fractions) != {"q", "h", "x"}:
        raise ValueError("component_fractions must specify exactly q, h, and x")
    fractions = {name: _positive(value, f"{name}_fraction", allow_zero=True)
                 for name, value in fractions.items()}
    model_fraction = _positive(model_fraction, "model_fraction", allow_zero=True)
    statistical_fraction = _positive(statistical_fraction, "statistical_fraction",
                                     allow_zero=True)
    total_fraction = _positive(total_fraction, "total_fraction", allow_zero=True)
    model = _positive(model_truncation_bound, "model_truncation_bound",
                      allow_zero=True)
    statistical = _positive(statistical_halfwidth, "statistical_halfwidth",
                             allow_zero=True)

    component_bounds = {
        name: math.fsum(math.log1p(value) for value in values)
        for name, values in qhx.items()
    }
    combined_corner_bound = math.fsum(
        math.log1p(q + h + x)
        for q, h, x in zip(qhx["q"], qhx["h"], qhx["x"])
    )
    component_sum = math.fsum(component_bounds.values())
    relative = {name: bound / reference
                for name, bound in component_bounds.items()}
    relative["model_truncation"] = model / reference
    relative["statistical"] = statistical / reference
    systematic_relative = (component_sum + model) / reference
    total_relative = (component_sum + model + statistical) / reference
    component_pass = all(relative[name] <= fractions[name] + 1e-14
                         for name in ("q", "h", "x"))
    model_pass = relative["model_truncation"] <= model_fraction + 1e-14
    statistical_pass = relative["statistical"] <= statistical_fraction + 1e-14
    total_pass = total_relative <= total_fraction + 1e-14
    return {
        "physical_score": None,
        "force_estimand": "absolute_stout_ground_state_discrete_static_force",
        "force_corner_count": ABSOLUTE_FORCE_CORNER_COUNT,
        "positive_force_scale_lower": reference,
        "component_force_bounds": component_bounds,
        "combined_log_corner_bound": combined_corner_bound,
        "factorized_qhx_bound": component_sum,
        "component_relative_bounds": relative,
        "systematic_relative_bound": systematic_relative,
        "total_relative_bound": total_relative,
        "limits": {
            "component_fractions": fractions,
            "model_fraction": model_fraction,
            "statistical_fraction": statistical_fraction,
            "total_fraction": total_fraction,
        },
        "component_gates_pass": component_pass,
        "model_gate_pass": model_pass,
        "statistical_gate_pass": statistical_pass,
        "total_gate_pass": total_pass,
        "assumption_only_numeric_gate_pass": (
            component_pass and model_pass and statistical_pass and total_pass
        ),
        "certification_status": "NOT_CERTIFIED_REQUIRES_INDEPENDENT_RECEIPTS_AND_REVIEW",
    }


def propagate_force_budget(
    qhx_bounds: Mapping[str, Sequence[Sequence[float]]],
    reference_force_lower: float,
    *,
    model_truncation_bound: float = 0.0,
    statistical_halfwidth: float = 0.0,
    component_fractions: Mapping[str, float] | None = None,
    model_fraction: float = 0.01,
    statistical_fraction: float = 0.05,
    total_fraction: float = 0.10,
) -> dict:
    """Propagate eight loop-level q/h/x bounds through the force contrast.

    For each kernel, the finite-Nt correction is
    `delta=log(1+q+h+x)`. The candidate-minus-reference force contrast has
    eight signed loop-corner terms. Triangle inequality and
    `log(1+q+h+x) <= log(1+q)+log(1+h)+log(1+x)` yield separate, conservative
    q/h/x force-unit envelopes suitable for comparison with allocated budgets.
    """
    if not isinstance(qhx_bounds, Mapping) or set(qhx_bounds) != {"q", "h", "x"}:
        raise ValueError("qhx_bounds must contain exactly q, h, and x")
    reference = _positive(reference_force_lower, "reference_force_lower")
    qhx = {name: _flatten_corner_bounds(qhx_bounds[name], name)
           for name in ("q", "h", "x")}
    fractions = dict(COMPONENT_FRACTIONS if component_fractions is None
                     else component_fractions)
    if set(fractions) != {"q", "h", "x"}:
        raise ValueError("component_fractions must specify exactly q, h, and x")
    fractions = {name: _positive(value, f"{name}_fraction", allow_zero=True)
                 for name, value in fractions.items()}
    model_fraction = _positive(model_fraction, "model_fraction", allow_zero=True)
    statistical_fraction = _positive(statistical_fraction, "statistical_fraction",
                                     allow_zero=True)
    total_fraction = _positive(total_fraction, "total_fraction", allow_zero=True)
    model = _positive(model_truncation_bound, "model_truncation_bound",
                      allow_zero=True)
    statistical = _positive(statistical_halfwidth, "statistical_halfwidth",
                             allow_zero=True)

    component_bounds = {
        name: math.fsum(math.log1p(value) for value in values)
        for name, values in qhx.items()
    }
    combined_corner_bound = math.fsum(
        math.log1p(q + h + x)
        for q, h, x in zip(qhx["q"], qhx["h"], qhx["x"])
    )
    component_sum = math.fsum(component_bounds.values())
    relative = {name: bound / reference
                for name, bound in component_bounds.items()}
    relative["model_truncation"] = model / reference
    relative["statistical"] = statistical / reference
    systematic_relative = (component_sum + model) / reference
    total_relative = (component_sum + model + statistical) / reference
    component_pass = all(relative[name] <= fractions[name] + 1e-14
                         for name in ("q", "h", "x"))
    model_pass = relative["model_truncation"] <= model_fraction + 1e-14
    statistical_pass = relative["statistical"] <= statistical_fraction + 1e-14
    total_pass = total_relative <= total_fraction + 1e-14
    return {
        "physical_score": None,
        "force_corner_count": FORCE_CORNER_COUNT,
        "reference_force_lower": reference,
        "component_force_bounds": component_bounds,
        "combined_log_corner_bound": combined_corner_bound,
        "factorized_qhx_bound": component_sum,
        "component_relative_bounds": relative,
        "systematic_relative_bound": systematic_relative,
        "total_relative_bound": total_relative,
        "limits": {
            "component_fractions": fractions,
            "model_fraction": model_fraction,
            "statistical_fraction": statistical_fraction,
            "total_fraction": total_fraction,
        },
        "component_gates_pass": component_pass,
        "model_gate_pass": model_pass,
        "statistical_gate_pass": statistical_pass,
        "total_gate_pass": total_pass,
        "assumption_only_numeric_gate_pass": (
            component_pass and model_pass and statistical_pass and total_pass
        ),
        "certification_status": "NOT_CERTIFIED_REQUIRES_INDEPENDENT_RECEIPTS_AND_REVIEW",
    }
