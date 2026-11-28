from __future__ import annotations

import math

from tasks.wilson_loop.f_lane_absolute_force_state_count_envelope_screen import (
    _force_envelope,
)


def test_state_count_envelope_uses_both_finite_point_estimates():
    result = _force_envelope({2: 0.14, 3: 0.16})
    assert result["status"] == "VALID_POINT_ESTIMATE_ENVELOPE"
    assert result["lower"] == 0.14
    assert result["upper"] == 0.16
    assert math.isclose(result["width"], 0.02)
    assert result["finite_sample_coverage_calibrated"] is False


def test_missing_state_count_fails_closed():
    result = _force_envelope({2: 0.14})
    assert result == {
        "status": "ABSTAIN_MODEL_FIT_FAILURE", "lower": None, "upper": None,
    }


def test_nonfinite_or_boolean_estimate_fails_closed():
    for estimates in ({2: 0.1, 3: math.inf}, {2: 0.1, 3: True}):
        result = _force_envelope(estimates)
        assert result == {
            "status": "ABSTAIN_INVALID_MODEL_ESTIMATE", "lower": None, "upper": None,
        }


def test_envelope_is_order_invariant():
    assert _force_envelope({2: 0.14, 3: 0.16}) == _force_envelope({3: 0.16, 2: 0.14})
