from __future__ import annotations

from tasks.wilson_loop.f_lane_three_static_profile_continuation_diagnostic import (
    _closest_valid_start,
)


def test_neighbor_start_uses_nearest_force_and_deterministic_force_tie_break():
    cells = [
        {"valid": True, "target_force": 0.16, "attempts": [
            {"valid": True, "objective": 1.0, "fit_coordinates": [16.0]},
        ]},
        {"valid": True, "target_force": 0.14, "attempts": [
            {"valid": True, "objective": 2.0, "fit_coordinates": [14.0]},
        ]},
        {"valid": False, "target_force": 0.15, "attempts": []},
    ]
    result = _closest_valid_start(cells, 0.15)
    assert result == {
        "source_force": 0.14,
        "source_objective": 2.0,
        "fit_coordinates": [14.0],
    }


def test_neighbor_start_abstains_when_no_cold_cell_is_valid():
    assert _closest_valid_start([
        {"valid": False, "target_force": 0.14, "attempts": []},
    ], 0.15) is None
