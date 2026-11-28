import hashlib
import json

import numpy as np
import pytest

from tasks.wilson_loop.f_lane_qhx_design_identifiability import (
    MODE_LABELS,
    analyze_design,
    build,
    correction_design_matrix,
)


@pytest.mark.parametrize("radius", [2, 3])
def test_matched_temporal_extents_improve_fixed_energy_mode_separation(radius):
    only_16 = analyze_design(correction_design_matrix(radius, (16,), range(3, 8)))
    only_32 = analyze_design(correction_design_matrix(radius, (32,), range(3, 8)))
    paired = analyze_design(correction_design_matrix(radius, (16, 32), range(3, 8)))

    assert only_16["numerical_rank_unit_column_scaled"] == 5
    assert only_32["numerical_rank_unit_column_scaled"] == 5
    assert paired["numerical_rank_unit_column_scaled"] == 5
    assert paired["unit_column_scaled_condition_number"] < (
        only_16["unit_column_scaled_condition_number"] / 100.0
    )
    assert paired["raw_condition_number"] < only_16["raw_condition_number"]
    assert paired["raw_condition_number"] < only_32["raw_condition_number"]


def test_fixed_nt32_suppresses_toy_thermal_and_mixed_column_norms():
    matrix_16 = correction_design_matrix(2, (16,), range(3, 8))
    matrix_32 = correction_design_matrix(2, (32,), range(3, 8))
    norms_16 = np.linalg.norm(matrix_16, axis=0)
    norms_32 = np.linalg.norm(matrix_32, axis=0)

    assert norms_32[0] == pytest.approx(norms_16[0])
    assert norms_32[1] == pytest.approx(norms_16[1])
    assert np.all(norms_32[2:] < norms_16[2:] * 2e-3)


def test_exactly_degenerate_static_gaps_make_q_and_x_modes_unidentifiable():
    matrix = correction_design_matrix(
        2, (16, 32), range(3, 8), static_energies=(0.44, 0.94, 0.94),
    )
    result = analyze_design(matrix)

    assert len(MODE_LABELS) == 5
    assert result["numerical_rank_unit_column_scaled"] == 3
    assert result["full_column_rank_unit_column_scaled"] is False
    assert result["unit_column_scaled_condition_number"] is None
    assert "rank deficient" in result["interpretation"]


def test_nearly_degenerate_static_gaps_are_numerically_full_rank_but_ill_conditioned():
    base = analyze_design(correction_design_matrix(2, (16, 32), range(3, 8)))
    near = analyze_design(correction_design_matrix(
        2, (16, 32), range(3, 8), static_energies=(0.44, 0.94, 0.940001),
    ))

    assert near["numerical_rank_unit_column_scaled"] == 5
    assert near["unit_column_scaled_condition_number"] > (
        base["unit_column_scaled_condition_number"] * 1e5
    )


@pytest.mark.parametrize(
    "radius,n_t,t_values,kwargs",
    [
        (1, (16,), range(3, 8), {}),
        (2, (), range(3, 8), {}),
        (2, (16, 16), range(3, 8), {}),
        (2, (16,), (3, 16), {}),
        (2, (16,), range(3, 8), {"static_energies": (0.44, 0.94)}),
        (2, (16,), range(3, 8), {"vacuum_energies": (0.0, float("nan"))}),
    ],
)
def test_invalid_design_inputs_fail_closed(radius, n_t, t_values, kwargs):
    with pytest.raises(ValueError):
        correction_design_matrix(radius, n_t, t_values, **kwargs)


def test_underdetermined_matrix_reports_null_direction():
    result = analyze_design(np.ones((3, 5)))
    assert result["numerical_rank_unit_column_scaled"] == 1
    assert result["full_column_rank_unit_column_scaled"] is False
    assert len(result["weakest_scaled_direction"]) == 5


def test_report_build_binds_source_and_artifact_hashes(tmp_path):
    out = tmp_path / "design-audit"
    summary = build(out)
    manifest = json.loads((out / "artifact_hashes.json").read_text())
    assert summary["status"] == "SYNTHETIC_DESIGN_IDENTIFIABILITY_DIAGNOSTIC_ONLY"
    assert summary["physical_score"] is None
    assert summary["diagnostic_cells"] == {
        "planned": 6, "valid": 6, "failed": 0, "abstained": 0,
    }
    assert summary["physical_measurements"] == {
        "planned": 0, "valid": 0, "failed": 0, "abstained": 0,
    }
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected
    protocol = json.loads((out / "protocol.json").read_text())
    for name, rel in {
        "design_diagnostic": "tasks/wilson_loop/f_lane_qhx_design_identifiability.py",
        "finite_trace_toy": "tasks/wilson_loop/f_lane_finite_nt_trace_diagnostic.py",
        "lane_contract": "tasks/wilson_loop/lane_contract.py",
        "lane_contract_json": "tasks/wilson_loop/lane_contract_v2.json",
    }.items():
        assert protocol["source_sha256"][name] == hashlib.sha256(
            open(rel, "rb").read()
        ).hexdigest()
    assert (out / "design_singular_values.png").is_file()
    assert (out / "design_metrics.csv").is_file()
