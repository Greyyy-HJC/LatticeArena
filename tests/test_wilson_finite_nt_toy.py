import hashlib
import json
import math

import pytest

from tasks.wilson_loop.finite_nt_toy import (
    TOY_NT,
    TOY_R,
    TOY_R_PLUS_1,
    contamination_terms,
    build,
    finite_nt_creutz,
    log_loop,
)
from tasks.wilson_loop.ground_state_bound import creutz_excited_correction


def test_toy_report_hashes_and_retains_shared_numeric_conversion(tmp_path):
    output = tmp_path / "finite-nt-toy"
    summary = build(output)
    saved = json.loads((output / "summary.json").read_text())
    hashes = saved["analysis_source_sha256"]
    toy_bytes = (output / "snapshot_finite_nt_toy.py").read_bytes()
    helper_bytes = (output / "snapshot_core_numerics.py").read_bytes()
    assert summary["physical_score"] is None
    assert hashlib.sha256(toy_bytes).hexdigest() == hashes["finite_nt_toy.py"]
    assert hashlib.sha256(helper_bytes).hexdigest() == hashes["core/numerics.py"]
    artifacts = json.loads((output / "artifact_hashes.json").read_text())
    assert hashlib.sha256(toy_bytes).hexdigest() == artifacts["snapshot_finite_nt_toy.py"]
    assert hashlib.sha256(helper_bytes).hexdigest() == artifacts["snapshot_core_numerics.py"]
    assert "snapshot_finite_nt_toy.py" in saved["files"]
    assert "snapshot_core_numerics.py" in saved["files"]


def test_q_h_x_use_pair_and_complementary_temporal_extents():
    terms = contamination_terms(TOY_R, time_extent=6, temporal_extent=16)
    assert terms["q"] == pytest.approx(TOY_R["Q"] * math.exp(-TOY_R["gap_q"] * 6))
    assert terms["h"] == pytest.approx(TOY_R["H"] * math.exp(-TOY_R["gap_0"] * 10))
    assert terms["x"] == pytest.approx(
        TOY_R["X"] * math.exp(-TOY_R["gap_q"] * 6 - TOY_R["gap_0"] * 10)
    )


def test_finite_nt_four_point_sign_matches_exact_delta_cross():
    result = finite_nt_creutz(TOY_R, TOY_R_PLUS_1, 5, TOY_NT)
    assert result["bias_lattice"] == pytest.approx(-result["delta_R_delta_t"])
    assert result["bias_lattice"] == pytest.approx(result["identity_bias_lattice"])


def test_zero_contamination_recovers_ground_state_discrete_force():
    r = dict(TOY_R, Q=0.0, H=0.0, X=0.0)
    rp1 = dict(TOY_R_PLUS_1, Q=0.0, H=0.0, X=0.0)
    for t in range(1, TOY_NT):
        result = finite_nt_creutz(r, rp1, t, TOY_NT)
        assert result["chi_lattice"] == pytest.approx(result["target_force_lattice"])
        assert result["bias_lattice"] == pytest.approx(0.0)


def test_zero_temperature_limit_matches_existing_positive_spectrum_correction():
    r = dict(TOY_R, H=0.0, X=0.0)
    rp1 = dict(TOY_R_PLUS_1, H=0.0, X=0.0)
    for t in range(4, 9):
        result = finite_nt_creutz(r, rp1, t, TOY_NT)
        rho_r_t = r["Q"] * math.exp(-r["gap_q"] * t)
        rho_r_tp1 = r["Q"] * math.exp(-r["gap_q"] * (t + 1))
        rho_r1_t = rp1["Q"] * math.exp(-rp1["gap_q"] * t)
        rho_r1_tp1 = rp1["Q"] * math.exp(-rp1["gap_q"] * (t + 1))
        expected = creutz_excited_correction(
            rho_r_t, rho_r_tp1, rho_r1_t, rho_r1_tp1,
        )
        assert result["bias_lattice"] == pytest.approx(expected)


def test_four_point_envelope_covers_exact_q_h_x_bias_at_all_toy_times():
    for t in range(4, 9):
        result = finite_nt_creutz(TOY_R, TOY_R_PLUS_1, t, TOY_NT)
        assert result["envelope_covers_bias"]
        assert abs(result["bias_lattice"]) <= result["four_loop_envelope_lattice"]


def test_finite_nt_tradeoff_can_suppress_q_while_increasing_h():
    early = contamination_terms(TOY_R, 4, TOY_NT)
    later = contamination_terms(TOY_R, 8, TOY_NT)
    assert later["q"] < early["q"]
    assert later["h"] > early["h"]


def test_time_stencil_cannot_exceed_temporal_extent():
    with pytest.raises(ValueError, match="cannot exceed"):
        finite_nt_creutz(TOY_R, TOY_R_PLUS_1, TOY_NT, TOY_NT)
    with pytest.raises(ValueError, match="at least one"):
        finite_nt_creutz(TOY_R, TOY_R_PLUS_1, 0, TOY_NT)


@pytest.mark.parametrize("updates", [
    {"Q": -0.1}, {"H": float("nan")}, {"X": -0.2},
    {"gap_q": -0.1}, {"gap_0": float("inf")},
])
def test_negative_or_nonfinite_positive_spectrum_inputs_reject(updates):
    params = dict(TOY_R)
    params.update(updates)
    with pytest.raises(ValueError):
        contamination_terms(params, 4, TOY_NT)


def test_log_loop_requires_positive_overlap():
    params = dict(TOY_R, A0=0.0)
    with pytest.raises(ValueError, match="A0"):
        log_loop(params, 4, TOY_NT)


@pytest.mark.parametrize("value", [True, "0.4", 2**53 + 1, 10**400, 0.4 + 0.1j])
def test_positive_spectrum_parameters_reject_implicit_or_unbounded_coercion(value):
    params = dict(TOY_R, Q=value)
    with pytest.raises(ValueError):
        contamination_terms(params, 4, TOY_NT)


def test_large_finite_qhx_uses_stable_log_envelope():
    params = dict(TOY_R, Q=1e308, H=1e308, X=1e308, gap_q=0.0, gap_0=0.0)
    terms = contamination_terms(params, 0, TOY_NT)
    assert math.isfinite(terms["delta"])
    assert terms["delta"] == pytest.approx(math.log(1e308) + math.log(3.0), rel=1e-14)


def test_nonfinite_log_loop_result_fails_closed():
    params = dict(TOY_R, V0=-1e308)
    with pytest.raises(ValueError, match="finite floating-point"):
        log_loop(params, 4, TOY_NT)
