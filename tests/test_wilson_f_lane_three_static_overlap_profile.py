from __future__ import annotations

import math

import numpy as np

from tasks.wilson_loop import f_lane_three_static_two_vacuum_candidate as candidate
from tasks.wilson_loop.f_lane_three_static_overlap_profile import (
    GROUND_A0_PARAMETER_INDICES,
    _constrained_quadratic_profile,
    assess_overlap_floor_receipt,
)


def test_local_profile_enforces_force_and_overlap_floor_constraints():
    parameter_count = 67
    point = np.zeros(parameter_count)
    point[1] = candidate._inverse_softplus(0.44)
    point[34] = candidate._inverse_softplus(0.59)
    point[4] = math.log(0.08)
    point[37] = math.log(0.04)
    gradient = candidate._absolute_force_gradient(point)
    context = {
        "point": point,
        "residual": np.zeros(parameter_count),
        "jacobian": np.eye(parameter_count),
        "force": 0.15,
        "force_gradient": gradient,
        "residual_at": lambda value: value - point,
    }
    result = _constrained_quadratic_profile(context, floor=0.05, target_force=0.16)
    assert result["valid"] is True
    assert result["status"] == "VALID_LOCAL_LINEARIZED_PROFILE"
    assert result["force_constraint_linearization_error"] < 1e-8
    assert result["estimated_ground_a0_min_linearized"] >= 0.05 - 1e-8


def test_missing_or_synthetic_overlap_floor_receipt_abstains():
    missing = assess_overlap_floor_receipt(None)
    assert missing["status"] == "ABSTAIN_NO_INDEPENDENT_OVERLAP_FLOOR"
    assert missing["physical_interval"] is None

    synthetic = assess_overlap_floor_receipt({
        "schema": "wilson-ground-overlap-floor-receipt-v1",
        "source_kind": "synthetic",
        "review_status": "accepted",
        "independent_of_fit": True,
        "overlap_definition": "min_R abs(A[m=0,n=0,operator=0]) in the toy's normalized endpoint convention",
        "split_id": "F-synthetic-open-screen",
        "lower_bound_by_radius": {"2": 0.04, "3": 0.04},
        "source_sha256_by_radius": {"2": "a" * 64, "3": "b" * 64},
        "independent_review_sha256": "c" * 64,
    })
    assert synthetic["status"] == "ABSTAIN_INVALID_OR_NONINDEPENDENT_FLOOR_RECEIPT"
    assert synthetic["physical_profile_permitted"] is False
    assert synthetic["physical_interval"] is None
