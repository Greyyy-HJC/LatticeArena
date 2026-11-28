import json

import pytest

from tasks.optimize_gaussian.d06_hac_fieller_diagnostic import (
    build,
    fieller_hac_lower,
    generate_panel,
    true_risks,
)


def test_fieller_contrast_inversion_returns_exact_ratio_with_zero_variance():
    assert fieller_hac_lower(2.4, 1.2, 0.0, 0.0, 0.0, 60, 1.96) == pytest.approx(2.0)


def test_fieller_contrast_lower_is_below_point_ratio_with_positive_variance():
    lower = fieller_hac_lower(2.4, 1.2, 0.04, 0.01, 0.03, 60, 1.96)
    assert 0 <= lower < 2.0


def test_fieller_rejects_invalid_risk_and_covariance_inputs():
    with pytest.raises(ValueError, match="finite and risks positive"):
        fieller_hac_lower(0.0, 1.0, 1.0, 0.0, 1.0, 60, 1.96)
    with pytest.raises(ValueError, match="HAC contrast variance is invalid"):
        fieller_hac_lower(1.0, 1.0, 1.0, 100.0, 1.0, 60, 1.96)


@pytest.mark.parametrize("family", ["gaussian", "student_t5", "periodic_volatility"])
def test_synthetic_generators_have_expected_shapes_and_finite_truth(family):
    panel = generate_panel(20261125, 0.8, 40, family, 0.8)
    assert panel.shape == (40, 3, 4)
    assert true_risks(0.8, 40, family)["trainable_W"] > 0


def test_small_report_build_preserves_denom_and_physical_null(tmp_path):
    output = tmp_path / "fieller-report"
    protocol = build(
        output, replicates=2, n_scenarios=(20,), phi_scenarios=(0.0,),
        families=("gaussian",), common_loadings=(0.0,),
    )
    assert protocol["denominator"] == {
        "planned": 2, "valid": 2, "failed": 0, "abstained": 0,
        "unit": "independent synthetic panels; all method/contrast endpoints finite",
    }
    assert protocol["physical_score"] is None
    summary = list(csv_rows(output / "coverage_summary.csv"))
    assert len(summary) == 2
    assert {row["method"] for row in summary} == {"ratio_delta", "fieller_contrast"}
    assert len(list(csv_rows(output / "replicates.csv"))) == 2


def csv_rows(path):
    import csv

    with path.open(newline="") as stream:
        yield from csv.DictReader(stream)
