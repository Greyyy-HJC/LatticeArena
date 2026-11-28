import numpy as np
import pytest

from tasks.wilson_loop import f_lane_qhx_maxt_interval as maxt


def test_gaussian_parameter_max_t_is_reproducible_and_simultaneous(monkeypatch):
    point = np.linspace(0.02, 0.30, maxt.EXPECTED_QHX_FAMILY)

    def nonlinear_map(parameters):
        return point * np.exp(0.2 * parameters[0])

    monkeypatch.setattr(maxt.fit_candidate, "_qhx_vector", nonlinear_map)
    estimate = np.zeros(maxt.EXPECTED_PARAMETERS)
    covariance = np.zeros((maxt.EXPECTED_PARAMETERS, maxt.EXPECTED_PARAMETERS))
    covariance[0, 0] = 0.04
    first = maxt.gaussian_parameter_max_t_upper(
        estimate, covariance, draws=2048, alpha=0.05, seed=991,
    )
    second = maxt.gaussian_parameter_max_t_upper(
        estimate, covariance, draws=2048, alpha=0.05, seed=991,
    )
    assert first == second
    assert first["status"] == "DIAGNOSTIC_SIMULTANEOUS_UPPER"
    assert first["family_size"] == 72
    assert first["denominators"] == {
        "planned": 2048, "valid": 2048, "failed": 0, "abstained": 0,
    }
    assert first["max_t_critical"] >= 0.0
    for interval, truth in zip(first["intervals"], point):
        assert interval["estimate"] == pytest.approx(truth)
        assert interval["simultaneous_upper"] >= truth
    assert first["certified"] is False and first["physical_score"] is None
    assert first["confidence_coverage"] is None


def test_gaussian_parameter_max_t_rejects_non_psd_covariance(monkeypatch):
    monkeypatch.setattr(
        maxt.fit_candidate, "_qhx_vector",
        lambda parameters: np.full(maxt.EXPECTED_QHX_FAMILY, 0.1),
    )
    covariance = np.eye(maxt.EXPECTED_PARAMETERS)
    covariance[0, 0] = -1.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        maxt.gaussian_parameter_max_t_upper(
            np.zeros(maxt.EXPECTED_PARAMETERS), covariance, seed=1,
        )


def test_gaussian_parameter_max_t_keeps_invalid_draws_in_denominator(monkeypatch):
    point = np.full(maxt.EXPECTED_QHX_FAMILY, 0.1)

    def partially_invalid_map(parameters):
        return point + float(parameters[0])

    monkeypatch.setattr(maxt.fit_candidate, "_qhx_vector", partially_invalid_map)
    covariance = np.zeros((maxt.EXPECTED_PARAMETERS, maxt.EXPECTED_PARAMETERS))
    covariance[0, 0] = 1.0
    result = maxt.gaussian_parameter_max_t_upper(
        np.zeros(maxt.EXPECTED_PARAMETERS), covariance,
        draws=512, seed=33,
    )
    assert result["status"] == "ABSTAIN_INVALID_PARAMETER_DRAWS"
    assert result["denominators"]["planned"] == 512
    assert result["denominators"]["failed"] > 0
    assert result["denominators"]["valid"] + result["denominators"]["failed"] == 512
    assert result["intervals"] is None
