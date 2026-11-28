import hashlib
import json

import numpy as np
import pytest

from tasks.wilson_loop import f_lane_joint_spectral_stress_diagnostic as stress
from tasks.wilson_loop import f_lane_joint_spectral_gls_diagnostic as joint


def test_fourth_state_truth_contains_the_omitted_state_contribution():
    energies, overlaps, _, _ = stress._scenario("omitted_4th_state")
    q_full = stress._true_q(energies, overlaps, operator_index=2, time=5)
    q_first_three = stress._true_q(energies[:3], overlaps[:3], operator_index=2, time=5)
    assert energies.size == 4
    assert q_full > q_first_three


def test_ar1_panel_generator_is_reproducible_and_serially_correlated():
    first = stress._generate_panel(731, n_blocks=stress.N_BLOCKS,
                                   scenario="ar1_block_rho_0p75")
    second = stress._generate_panel(731, n_blocks=stress.N_BLOCKS,
                                    scenario="ar1_block_rho_0p75")
    assert np.array_equal(first, second)
    energies, overlaps, _, _ = stress._scenario("ar1_block_rho_0p75")
    mean = np.concatenate([
        joint._matrix_vector(matrix)
        for matrix in stress._spectrum_matrices(energies, overlaps)
    ])
    residual = first - mean
    observed = float(np.corrcoef(residual[:-1, 0], residual[1:, 0])[0, 1])
    assert observed == pytest.approx(0.75, abs=0.22)


def test_student_t5_generator_is_reproducible_and_heavy_tailed():
    first = stress._generate_panel(991, n_blocks=1000,
                                   scenario="student_t5_independent")
    second = stress._generate_panel(991, n_blocks=1000,
                                    scenario="student_t5_independent")
    assert np.array_equal(first, second)
    energies, overlaps, _, _ = stress._scenario("student_t5_independent")
    mean = np.concatenate([
        joint._matrix_vector(matrix)
        for matrix in stress._spectrum_matrices(energies, overlaps)
    ])
    residual = first[:, 0] - mean[0]
    standardized = residual / np.std(residual, ddof=1)
    sample_kurtosis = float(np.mean(standardized ** 4))
    assert sample_kurtosis > 4.0


def test_small_sample_stress_report_preserves_abstention_and_hashes(tmp_path):
    output = tmp_path / "wilson-stress-report"
    result = stress.run(output, panels_per_scenario=20, n_blocks=stress.N_BLOCKS)
    denominator = result["denominators"]
    assert denominator["synthetic_panels_planned"] == len(stress.SCENARIOS) * 20
    assert denominator["q_cells_planned"] == len(stress.SCENARIOS) * 20 * 4
    assert (denominator["q_cells_valid"] + denominator["q_cells_failed"]
            + denominator["q_cells_abstained"] == len(stress.SCENARIOS) * 20 * 4)
    assert denominator["conformal_strata_planned"] == len(stress.SCENARIOS) * 4
    assert denominator["conformal_strata_finite_multiplier"] == 0
    assert denominator["conformal_strata_abstained"] == len(stress.SCENARIOS) * 4
    assert denominator["conformal_holdout_valid"] == 0
    assert (denominator["conformal_holdout_valid"]
            + denominator["conformal_holdout_failed"]
            + denominator["conformal_holdout_abstained"] == len(stress.SCENARIOS) * 10 * 4)
    assert denominator["physical_measurements_planned"] == 0
    assert denominator["certification_abstained"] == 1
    assert result["physical_score"] is None
    assert result["F_lane_status"] == "GROUND_STATE_FORCE_UNCERTIFIED"

    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["source_sha256"][stress.Path(stress.__file__).name] == hashlib.sha256(
        stress.Path(stress.__file__).read_bytes()
    ).hexdigest()
    hashes = json.loads((output / "artifact_hashes.json").read_text())
    for name, digest in hashes.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert (output / "stress_coverage.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (output / "interval_conservatism.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    readme = (output / "README.md").read_text()
    assert "Scenario-wise findings" in readme
    assert "omitted_4th_state" in readme
    with pytest.raises(FileExistsError):
        stress.run(output, panels_per_scenario=20, n_blocks=stress.N_BLOCKS)


@pytest.mark.parametrize("panels,n_blocks,match", [
    (19, 96, "even integer >= 20"),
    (20, 63, "exceed the 63 raw covariance components"),
])
def test_stress_run_rejects_invalid_design_before_writing(tmp_path, panels, n_blocks, match):
    output = tmp_path / "invalid-stress-report"
    with pytest.raises(ValueError, match=match):
        stress.run(output, panels_per_scenario=panels, n_blocks=n_blocks)
    assert not output.exists()
