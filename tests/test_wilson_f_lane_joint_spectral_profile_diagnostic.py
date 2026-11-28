import hashlib
import json

import numpy as np
import pytest

from tasks.wilson_loop import f_lane_joint_spectral_gls_diagnostic as joint
from tasks.wilson_loop import f_lane_joint_spectral_profile_diagnostic as profile
from tasks.wilson_loop import f_lane_joint_spectral_stress_diagnostic as stress


@pytest.mark.parametrize("operator_index", [0, 2])
def test_profile_parameterization_reconstructs_the_requested_q(operator_index):
    energies = np.array([0.31, 0.56, 0.91])
    overlaps = np.array([
        [0.7 + 0.0j, 0.22 + 0.08j, 0.31 - 0.17j],
        [0.16 + 0.0j, 0.13 - 0.05j, 0.12 + 0.09j],
        [0.09 + 0.0j, 0.05 + 0.03j, 0.08 - 0.04j],
    ])
    parameters = joint._pack_parameters(energies, overlaps)
    q_target = joint._q_value(parameters, operator_index, 5, n_states=3)
    profile_vector = profile._profile_vector(parameters, q_target, operator_index)
    reconstructed = profile._parameters_at_profile_q(
        profile_vector, operator_index, 5, n_states=3,
    )
    assert joint._q_value(reconstructed, operator_index, 5, n_states=3) == pytest.approx(q_target)
    profile_vector[0] = 1.2 * q_target
    analytic = profile._profile_transform_jacobian(profile_vector, operator_index, 5, 3)
    numeric = np.empty_like(analytic)
    for index in range(profile_vector.size):
        step = 1e-6 * max(1.0, abs(float(profile_vector[index])))
        plus, minus = profile_vector.copy(), profile_vector.copy()
        plus[index] += step
        minus[index] -= step
        numeric[:, index] = (
            profile._parameters_at_profile_q(plus, operator_index, 5, 3)
            - profile._parameters_at_profile_q(minus, operator_index, 5, 3)
        ) / (2.0 * step)
    assert analytic == pytest.approx(numeric, rel=2e-5, abs=2e-6)


def test_profile_upper_returns_auditable_status_and_likelihood_displacement():
    blocks = stress._generate_panel(
        2026112740, n_blocks=stress.N_BLOCKS,
        scenario="matched_3state_independent",
    )
    result = profile._profile_upper(blocks, operator_index=0, time=5)
    assert result["status"] in {"VALID", "ABSTAIN_PARAMETER_BOUNDARY"}
    assert result["optimizer_success"]
    assert result["chi2_delta"] == pytest.approx(profile.PROFILE_DELTA_CHI2, abs=1e-4)
    if result["status"] == "VALID":
        assert result["profile_upper"] > result["q_estimate"]


def test_verified_b80_report_hashes_and_denominator():
    protocol, intervals = profile._read_verified_b80(profile.DEFAULT_B80)
    assert protocol["total_bootstrap_replicates"] == 80
    assert len(intervals) == 960


def test_profile_report_smoke_reconciles_abstentions_and_hashes(tmp_path):
    output = tmp_path / "profile-report"
    result = profile.run(
        output, scenarios=("matched_3state_independent",),
        outer_panels_per_scenario=1,
    )
    den = result["denominators"]
    assert den["profile_intervals_planned"] == 4
    assert (den["profile_intervals_valid"] + den["profile_intervals_failed"]
            + den["profile_intervals_abstained"] == 4)
    assert den["physical_measurements_planned"] == 0
    assert den["certification_abstained"] == 1
    assert result["physical_score"] is None
    manifest = json.loads((output / "artifact_hashes.json").read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert (output / "method_coverage.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    readme = (output / "README.md").read_text()
    assert "Wilks" in readme
    assert "plotting placeholder" in readme
    assert "conditional pointwise coverage" in readme
    protocol = json.loads((output / "protocol.json").read_text())
    assert "analytic model" in protocol["optimizer"]
    assert "no fit or resampling was rerun" in protocol["derived_report_refresh"]
    with pytest.raises(FileExistsError):
        profile.run(output, scenarios=("matched_3state_independent",),
                    outer_panels_per_scenario=1)


@pytest.mark.parametrize("outer,match", [(0, r"\[1,40\]"), (41, r"\[1,40\]")])
def test_profile_report_rejects_invalid_panel_count_before_output(tmp_path, outer, match):
    output = tmp_path / "invalid-profile-report"
    with pytest.raises(ValueError, match=match):
        profile.run(output, scenarios=("matched_3state_independent",),
                    outer_panels_per_scenario=outer)
    assert not output.exists()
