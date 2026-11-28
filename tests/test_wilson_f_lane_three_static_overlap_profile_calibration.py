from __future__ import annotations

from tasks.wilson_loop.f_lane_three_static_overlap_profile_calibration import (
    FLOORS,
    _coverage_summary,
    _force_grid,
    _panel_profile,
    _wilson_interval,
)


def test_adaptive_grid_contains_exact_truth_and_panel_fit_center():
    grid = _force_grid(0.15757559165100204)
    assert 0.15 in grid
    assert 0.15757559165100204 in grid
    assert len(grid) == len(set(grid))


def test_invalid_candidate_fit_abstains_all_floor_profiles():
    cells, panel = _panel_profile(
        snapshot=None,
        fit_receipt={"denominator": {"valid": 0}},
        truth=0.15,
        panel_id=0,
    )
    assert cells == []
    assert panel["panel_valid"] is False
    assert len(panel["floor_results"]) == len(FLOORS)
    assert all(item["status"] == "ABSTAIN_CANDIDATE_FIT" for item in panel["floor_results"])


def test_clipped_profile_is_excluded_from_empirical_coverage_denominator():
    panel = {
        "floor_results": [
            {"floor": 0.0, "interval": [0.145, 0.155], "contains_truth": True,
             "grid_clipped": False},
            {"floor": 0.045, "interval": [0.15, 0.18], "contains_truth": True,
             "grid_clipped": True},
        ],
    }
    result = _coverage_summary([panel])
    no_floor = result[0]
    assert no_floor["unclipped_profiles_used_for_coverage"] == 1
    assert no_floor["truth_contained_panels"] == 1
    oracle_floor = result[1]
    assert oracle_floor["clipped_intervals"] == 1
    assert oracle_floor["unclipped_profiles_used_for_coverage"] == 0
    assert oracle_floor["conditional_containment_fraction"] is None
    assert oracle_floor["wilson_95_interval"] is None


def test_wilson_interval_reports_small_sample_uncertainty_and_zero_denominator():
    interval = _wilson_interval(5, 6)
    assert interval is not None
    assert interval[0] < 5 / 6 < interval[1]
    assert _wilson_interval(0, 0) is None
