from tasks.wilson_loop import f_lane_qhx_maxt_screen as screen


def test_maxt_screen_protocol_matches_frozen_dgp_and_uses_disjoint_seeds():
    protocol, _, base = screen._load_protocol()
    expected_ids = [row["id"] for row in base["strata"]]
    assert protocol["comparison"]["truth_strata"] == expected_ids
    assert protocol["panels_per_stratum"] == 12
    assert protocol["seed_rule"].startswith("2026113400 +")
    assert protocol["sealed_data_access"] is False
    assert protocol["certified"] is False


def test_qhx_term_positions_advance_three_items_per_truth_cell():
    indices = [
        screen._max_t_interval_index(cell_index, term_index)
        for cell_index in range(24)
        for term_index in range(3)
    ]
    assert indices == list(range(72))


def test_wilson_interval_has_exact_binomial_boundary_endpoints():
    assert screen._wilson(0, 12)[0] == 0.0
    assert screen._wilson(12, 12)[1] == 1.0


def test_coverage_plot_handles_zero_success_boundary(tmp_path):
    summaries = [{
        "stratum_id": "S1_screen",
        "familywise_coverage": {
            method: {"rate": 0.0, "wilson95": screen._wilson(0, 12)}
            for method in ("delta_bonferroni72", "gaussian_parameter_maxt")
        },
        "median_absolute_upper_width_by_method_term": {
            method: {term: 0.1 for term in screen.TERMS}
            for method in ("delta_bonferroni72", "gaussian_parameter_maxt")
        },
    }]
    screen._plots(tmp_path, summaries)
    assert (tmp_path / "coverage_and_width.png").is_file()
