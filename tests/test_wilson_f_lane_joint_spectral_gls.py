import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tasks.wilson_loop import f_lane_joint_spectral_gls_diagnostic as diagnostic


def test_exact_spectrum_round_trips_and_gevp_recovers_energies():
    parameters = diagnostic._pack_parameters(
        diagnostic.TRUE_ENERGIES, diagnostic.TRUE_OVERLAPS,
    )
    expected = np.concatenate([
        diagnostic._matrix_vector(matrix)
        for matrix in diagnostic._spectrum_matrices(
            diagnostic.TRUE_ENERGIES, diagnostic.TRUE_OVERLAPS,
        )
    ])
    assert np.allclose(diagnostic._model_vector(parameters), expected, rtol=1e-13, atol=1e-14)
    initial = diagnostic._initial_from_gevp(expected)
    assert diagnostic._unpack_parameters(initial)[0] == pytest.approx(
        diagnostic.TRUE_ENERGIES, rel=1e-12, abs=1e-12,
    )


def test_same_block_covariance_regularization_is_positive_definite():
    blocks = np.random.default_rng(314159).normal(size=(64, 63))
    covariance, details = diagnostic._regularized_covariance(blocks)
    assert covariance.shape == (63, 63)
    assert np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14)
    assert np.linalg.eigvalsh(covariance)[0] > 0.0
    assert details["condition_number"] >= 1.0
    assert details["floor_active"] >= 0.0


def test_joint_fit_returns_finite_pointwise_q_upper_for_seeded_panel():
    blocks = diagnostic._design_covariance(2026112600, 64, 0.04)
    result = diagnostic.fit_joint_spectrum(blocks)
    assert result["success"]
    assert result["rank"] == result["parameter_count"] == 18
    assert result["energies"] == pytest.approx(diagnostic.TRUE_ENERGIES, abs=0.01)
    assert len(result["q_cells"]) == 4
    for row in result["q_cells"]:
        assert np.isfinite(row["q_standard_error_delta"])
        assert row["q_standard_error_delta"] >= 0.0
        assert row["q_upper_97_5_pointwise"] >= row["q_estimate"]


def test_joint_fit_supports_two_state_three_operator_subwindows():
    times = (3, 4, 5, 6, 7)
    energies = np.asarray([0.18, 0.42])
    overlaps = np.asarray([
        [1.0, 0.8 + 0.1j, 0.6 - 0.05j],
        [0.6, -0.4 + 0.2j, 0.5 - 0.1j],
    ])
    matrices = diagnostic._spectrum_matrices(energies, overlaps, times=times)
    center = np.concatenate([diagnostic._matrix_vector(matrix) for matrix in matrices])
    scales = np.repeat([
        0.004 * max(float(np.linalg.norm(matrix)) / 3.0, 1e-10)
        for matrix in matrices
    ], 9)
    blocks = center + np.random.default_rng(271828).normal(size=(80, center.size)) * scales
    result = diagnostic.fit_joint_spectrum(
        blocks, time_slices=times, n_states=2, operator_indices=(0, 1, 2),
    )
    assert result["success"]
    assert result["n_states"] == 2
    assert result["time_slices"] == list(times)
    assert result["energies"] == pytest.approx(energies, abs=0.01)
    assert len(result["q_cells"]) == 6


@pytest.mark.parametrize(
    "blocks,match",
    [
        (np.zeros((64, 62)), "shape"),
        (np.zeros((63, 63)), "at least 64 blocks"),
        (np.full((64, 63), np.nan), "finite values"),
    ],
)
def test_joint_fit_rejects_invalid_block_panels(blocks, match):
    with pytest.raises(ValueError, match=match):
        diagnostic.fit_joint_spectrum(blocks)


def test_split_conformal_uses_finite_sample_rank_and_abstains_when_too_small():
    assert diagnostic._split_conformal_upper_multiplier(np.arange(1.0, 101.0)) == 99.0
    assert diagnostic._split_conformal_upper_multiplier(np.arange(10.0)) is None
    assert diagnostic._split_conformal_upper_multiplier(
        np.concatenate([np.arange(1.0, 100.0), [np.inf]])
    ) == 99.0
    assert diagnostic._split_conformal_upper_multiplier(np.full(100, np.inf)) is None
    with pytest.raises(ValueError, match="invalid"):
        diagnostic._split_conformal_upper_multiplier([1.0, np.nan])


def test_report_binds_sources_artifacts_and_explicit_abstention(tmp_path):
    output = tmp_path / "joint-gls-report"
    summary = diagnostic.run(output, panels_per_cell=20)
    denominators = summary["denominators"]
    assert denominators["synthetic_panels_planned"] == 80
    assert denominators["synthetic_panels_valid"] + denominators["synthetic_panels_failed"] == 80
    assert denominators["synthetic_panels_abstained"] == 0
    assert denominators["q_cells_planned"] == 320
    assert (denominators["q_cells_valid"] + denominators["q_cells_failed"]
            + denominators["q_cells_abstained"] == 320)
    assert denominators["physical_measurements_planned"] == 0
    assert denominators["conformal_calibration_cells_planned"] == 16
    assert denominators["conformal_calibration_cells_finite_multiplier"] == 0
    assert denominators["conformal_validation_q_cells_planned"] == 160
    assert denominators["conformal_validation_q_cells_valid"] == 0
    assert denominators["conformal_validation_q_cells_abstained"] == 160
    assert denominators["certification_abstained"] == 1
    assert summary["physical_score"] is None
    assert summary["F_lane_status"] == "GROUND_STATE_FORCE_UNCERTIFIED"

    protocol = json.loads((output / "protocol.json").read_text())
    source_path = Path(diagnostic.__file__)
    assert protocol["source_sha256"][source_path.name] == hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    hashes = json.loads((output / "artifact_hashes.json").read_text())
    for name, digest in hashes.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert (output / "coverage_and_interval_width.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    assert (output / "calibration_holdout_coverage.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    with (output / "calibration_summary.csv").open() as stream:
        calibration_rows = list(csv.DictReader(stream))
    assert len(calibration_rows) == 16
    assert all(row["calibration_status"] == "ABSTAIN_NO_FINITE_MULTIPLIER"
               for row in calibration_rows)
    with pytest.raises(FileExistsError):
        diagnostic.run(output, panels_per_cell=20)
