from __future__ import annotations

from tasks.wilson_loop.f_lane_three_static_stratified_profile_screen import (
    _stratum_coverage,
    _truth_floor,
)


def test_oracle_overlap_floor_comes_from_each_declared_toy_truth():
    assert abs(_truth_floor("weak_ground_overlap") - 0.045) < 1e-12
    assert abs(_truth_floor("baseline") - 0.9) < 1e-12
    assert abs(_truth_floor("near_degenerate") - 0.9) < 1e-12


def test_stratum_coverage_excludes_clipped_sets_and_keeps_floor_categories_distinct():
    panels = [
        {
            "stratum": "baseline", "oracle_truth_floor": 0.9,
            "floor_results": [
                {"floor": 0.0, "interval": [0.15, 0.16], "contains_truth": True,
                 "grid_clipped": False},
                {"floor": 0.9, "interval": [0.14, 0.15], "contains_truth": True,
                 "grid_clipped": True},
            ],
        },
        {
            "stratum": "baseline", "oracle_truth_floor": 0.9,
            "floor_results": [
                {"floor": 0.0, "interval": [0.16, 0.17], "contains_truth": False,
                 "grid_clipped": False},
                {"floor": 0.9, "interval": [0.14, 0.15], "contains_truth": True,
                 "grid_clipped": False},
            ],
        },
    ]
    no_floor, oracle_floor = _stratum_coverage(panels, "baseline")
    assert no_floor["unclipped_profiles_used_for_coverage"] == 2
    assert no_floor["truth_contained_panels"] == 1
    assert oracle_floor["clipped_intervals"] == 1
    assert oracle_floor["unclipped_profiles_used_for_coverage"] == 1
    assert oracle_floor["truth_contained_panels"] == 1
