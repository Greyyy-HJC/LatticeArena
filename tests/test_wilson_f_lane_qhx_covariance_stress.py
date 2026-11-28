import hashlib
import json

import numpy as np
import pytest

from tasks.wilson_loop.f_lane_qhx_covariance_stress_diagnostic import (
    N_T_CORRELATIONS,
    N_T_PLANS,
    R_VALUES,
    T_CORRELATIONS,
    T_VALUES,
    _scenario_rows,
    build,
    separable_correlation,
    whiten_design,
)
from tasks.wilson_loop.f_lane_qhx_design_identifiability import (
    analyze_design,
    correction_design_matrix,
)


def test_zero_correlation_whitening_reproduces_unweighted_design():
    for radius in R_VALUES:
        for plan in N_T_PLANS:
            matrix = correction_design_matrix(radius, plan, T_VALUES)
            covariance = separable_correlation(
                plan, T_VALUES, temporal_rho=0.0, cross_extent_rho=0.0,
            )
            weighted = analyze_design(whiten_design(matrix, covariance))
            unweighted = analyze_design(matrix)
            assert weighted["numerical_rank_unit_column_scaled"] == (
                unweighted["numerical_rank_unit_column_scaled"]
            )
            assert weighted["unit_column_scaled_condition_number"] == pytest.approx(
                unweighted["unit_column_scaled_condition_number"], rel=1e-12,
            )


def test_separable_covariance_is_symmetric_positive_definite_and_unit_diagonal():
    covariance = separable_correlation(
        (16, 32), T_VALUES, temporal_rho=-0.9, cross_extent_rho=0.9,
    )
    assert covariance.shape == (10, 10)
    assert np.allclose(covariance, covariance.T)
    assert np.allclose(np.diag(covariance), 1.0)
    assert np.linalg.eigvalsh(covariance)[0] > 0


@pytest.mark.parametrize(
    "matrix,covariance",
    [
        (np.ones((5, 5)), np.eye(4)),
        (np.ones((5, 5)), np.eye(5) + np.diag(np.full(4, 0.2), 1)),
        (np.ones((5, 5)), np.eye(5) + np.diag(np.full(4, 2.0), 1)
         + np.diag(np.full(4, 2.0), -1)),
    ],
)
def test_whitening_rejects_bad_covariance(matrix, covariance):
    with pytest.raises(ValueError):
        whiten_design(matrix, covariance)


def test_correlation_grid_keeps_paired_shape_condition_below_single_extent():
    rows = _scenario_rows()
    assert len(rows) == len(R_VALUES) * len(T_CORRELATIONS) * (
        len(N_T_PLANS) - 1 + len(N_T_CORRELATIONS)
    )
    paired = [row for row in rows if row["N_t_plan"] == "16+32"]
    assert len(paired) == len(R_VALUES) * len(T_CORRELATIONS) * len(N_T_CORRELATIONS)
    assert all(row["shape_condition_improvement_factor"] > 1.0 for row in paired)
    assert all(row["numerical_rank"] == 5 for row in paired)


def test_report_build_binds_sources_and_artifacts(tmp_path):
    output = tmp_path / "covariance-stress"
    summary = build(output)
    hashes = json.loads((output / "artifact_hashes.json").read_text())["files"]
    assert summary["status"] == "HYPOTHETICAL_COVARIANCE_CORRELATION_SENSITIVITY_ONLY"
    assert summary["scenario_denominator"] == {
        "planned": 50, "valid": 50, "failed": 0, "abstained": 0,
    }
    assert summary["physical_measurements"] == {
        "planned": 0, "valid": 0, "failed": 0, "abstained": 0,
    }
    for filename, expected in hashes.items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == expected
    protocol = json.loads((output / "protocol.json").read_text())
    source = "tasks/wilson_loop/f_lane_qhx_covariance_stress_diagnostic.py"
    assert protocol["source_sha256"]["covariance_diagnostic"] == hashlib.sha256(
        open(source, "rb").read()
    ).hexdigest()
    assert (output / "covariance_condition_heatmap.png").is_file()
