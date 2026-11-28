import numpy as np
import pytest

from tasks.wilson_loop.f_lane_finite_nt_trace_diagnostic import (
    EFFECTIVE_ENERGY_TIMES,
    OPERATOR_IDS,
    R_VALUES,
    TEMPORAL_EXTENTS,
    TIME_EXTENTS,
    effective_energy,
    finite_trace_matrix,
    trace_components,
)


def test_finite_nt_cross_correlator_matrix_is_hermitian_psd_and_complex():
    matrix = finite_trace_matrix(radius=2, time_extent=4, temporal_extent=16)

    assert matrix.shape == (3, 3)
    assert np.all(np.isfinite(matrix))
    assert np.allclose(matrix, matrix.conj().T, rtol=1e-12, atol=1e-14)
    assert np.linalg.eigvalsh(matrix)[0] >= -1e-12
    assert np.max(np.abs(np.imag(matrix - np.diag(np.diag(matrix))))) > 1e-8


def test_qhx_reconstructs_every_bare_operator_diagonal_cell():
    checked = 0
    for radius in R_VALUES:
        for temporal_extent in TEMPORAL_EXTENTS:
            for time_extent in TIME_EXTENTS:
                matrix = finite_trace_matrix(radius, time_extent, temporal_extent)
                for operator_id in OPERATOR_IDS:
                    row = trace_components(
                        radius, operator_id, time_extent, temporal_extent,
                    )
                    assert row["q"] >= 0.0
                    assert row["h"] >= 0.0
                    assert row["x"] >= 0.0
                    assert row["decomposition_abs_error"] < 1e-12
                    assert row["full_to_ground_ratio"] == pytest.approx(
                        row["reconstructed_to_ground_ratio"], rel=1e-12,
                    )
                    assert matrix[OPERATOR_IDS.index(operator_id),
                                  OPERATOR_IDS.index(operator_id)].real > 0.0
                    checked += 1

    assert checked == 60


def test_vacuum_ground_q_is_nt_independent_and_thermal_sectors_decay():
    for radius in R_VALUES:
        for operator_id in OPERATOR_IDS:
            for time_extent in TIME_EXTENTS:
                short = trace_components(radius, operator_id, time_extent, 16)
                long = trace_components(radius, operator_id, time_extent, 32)
                assert short["q"] == long["q"]
                assert long["h"] < short["h"]
                assert long["x"] < short["x"]


def test_effective_energy_bias_rows_are_finite_and_t_plus_one_is_in_range():
    assert EFFECTIVE_ENERGY_TIMES == (3, 4, 5, 6)
    for radius in R_VALUES:
        for temporal_extent in TEMPORAL_EXTENTS:
            for time_extent in EFFECTIVE_ENERGY_TIMES:
                row = effective_energy(radius, "stout", time_extent,
                                       temporal_extent)
                assert np.isfinite(row["effective_energy"])
                assert np.isfinite(row["bias"])
                assert row["ground_energy"] > 0.0


@pytest.mark.parametrize(
    "radius,operator_id,time_extent,temporal_extent",
    [
        (1, "stout", 3, 16),
        (2, "unknown", 3, 16),
        (2, "stout", 17, 16),
        (2, "stout", 3.0, 16),
        (2, "stout", 3, True),
    ],
)
def test_invalid_trace_inputs_fail_closed(radius, operator_id, time_extent,
                                         temporal_extent):
    with pytest.raises(ValueError):
        trace_components(radius, operator_id, time_extent, temporal_extent)


def test_effective_energy_rejects_t_at_temporal_boundary():
    with pytest.raises(ValueError, match=r"T\+1 <= N_t"):
        effective_energy(2, "stout", 16, 16)
