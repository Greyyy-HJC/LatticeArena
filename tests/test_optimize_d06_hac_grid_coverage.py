from tasks.optimize_gaussian.d06_hac_grid_coverage_diagnostic import (
    _paired_comparison,
    _summary,
)


def _panel(n, phi, percentile_fwer, hac_fwer, percentile_slsqp, hac_slsqp,
           percentile_cheb, hac_cheb):
    return {
        "n_configurations": n,
        "phi": phi,
        "percentile_familywise_covered": percentile_fwer,
        "hac_familywise_covered": hac_fwer,
        "hac_familywise_valid": 1,
        "percentile_covered_SLSQP": percentile_slsqp,
        "hac_covered_SLSQP": hac_slsqp,
        "hac_valid_SLSQP": 1,
        "percentile_covered_Chebyshev": percentile_cheb,
        "hac_covered_Chebyshev": hac_cheb,
        "hac_valid_Chebyshev": 1,
    }


def test_grid_summary_retains_each_method_cell_denominator():
    rows = [
        _panel(40, 0.0, 1, 1, 1, 1, 1, 1),
        _panel(40, 0.0, 0, 1, 0, 1, 1, 1),
        _panel(40, 0.4, 0, 0, 0, 0, 1, 1),
    ]

    result = _summary(rows)
    fwer = [item for item in result if item["unit"] == "familywise"]

    assert len(fwer) == 4
    cell_40_phi0 = [item for item in fwer if item["n_configurations"] == 40 and item["phi"] == 0.0]
    assert {(item["method"], item["planned"], item["valid"], item["coverage"])
            for item in cell_40_phi0} == {
                ("percentile", 2, 2, 0.5),
                ("hac", 2, 2, 1.0),
            }


def test_paired_grid_comparison_uses_same_panel_outcomes():
    rows = [
        _panel(40, 0.0, 0, 1, 0, 1, 0, 1),
        _panel(40, 0.0, 1, 1, 1, 1, 1, 1),
        _panel(40, 0.0, 1, 0, 1, 1, 1, 0),
    ]

    result = _paired_comparison(rows)
    familywise = next(item for item in result if item["unit"] == "familywise")

    assert familywise["planned"] == familywise["valid_paired_panels"] == 3
    assert familywise["percentile_only_covers"] == 1
    assert familywise["hac_only_covers"] == 1
    assert familywise["hac_minus_percentile_coverage"] == 0.0
    assert familywise["paired_panel_bootstrap_95"][0] <= 0.0
    assert familywise["paired_panel_bootstrap_95"][1] >= 0.0
