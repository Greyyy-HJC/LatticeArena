"""Asymptotic Gaussian-parameter max-T upper bound for the 72 q/h/x cells.

This is a diagnostic interval candidate. It propagates the fitted parameter
covariance through the nonlinear q/h/x map and uses the simulated maximum
studentized deviation for a simultaneous upper. It is not a finite-sample
calibration and cannot overcome spectral-model bias or invalid covariance.
"""
from __future__ import annotations

import math

import numpy as np

from tasks.wilson_loop import f_lane_two_sector_fit_candidate as fit_candidate


EXPECTED_PARAMETERS = 45
EXPECTED_QHX_FAMILY = 72


def gaussian_parameter_max_t_upper(
    parameter_estimate: object,
    parameter_covariance: object,
    *,
    draws: int = 4096,
    alpha: float = 0.05,
    seed: int,
) -> dict:
    """Return simultaneous max-T q/h/x uppers under a Gaussian fit approximation.

    The covariance is the inverse whitened-Jacobian information from the
    regularized full-covariance GLS fit. Its Gaussian interpretation is an
    explicit modeling approximation. All draws stay in the denominator; a
    single invalid nonlinear transform fails the candidate closed.
    """
    estimate = np.asarray(parameter_estimate)
    covariance = np.asarray(parameter_covariance)
    if estimate.dtype.kind not in "iuf" or estimate.shape != (EXPECTED_PARAMETERS,):
        raise ValueError("parameter_estimate must be 45 real fit coordinates")
    if covariance.dtype.kind not in "iuf" or covariance.shape != (
            EXPECTED_PARAMETERS, EXPECTED_PARAMETERS):
        raise ValueError("parameter_covariance must be a 45x45 real matrix")
    estimate = estimate.astype(np.float64)
    covariance = covariance.astype(np.float64)
    if not np.isfinite(estimate).all() or not np.isfinite(covariance).all():
        raise ValueError("fit estimate and covariance must be finite")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 128:
        raise ValueError("draws must be an integer >= 128")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or not math.isfinite(float(alpha)) or not 0.0 < alpha < 1.0):
        raise ValueError("alpha must be finite and in (0,1)")

    asymmetry = float(np.max(np.abs(covariance - covariance.T)))
    scale = max(1.0, float(np.max(np.abs(covariance))))
    if asymmetry > 1e-10 * scale:
        raise ValueError("parameter_covariance must be symmetric")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigen_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    negative_tolerance = 1e-10 * eigen_scale
    if float(eigenvalues[0]) < -negative_tolerance:
        raise ValueError("parameter_covariance must be positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    factor = eigenvectors * np.sqrt(eigenvalues)

    point = np.asarray(fit_candidate._qhx_vector(estimate), dtype=np.float64)
    if (point.shape != (EXPECTED_QHX_FAMILY,) or not np.isfinite(point).all()
            or np.any(point < 0.0)):
        raise ValueError("fitted q/h/x point must contain 72 finite nonnegative values")

    rng = np.random.default_rng(seed)
    parameter_draws = estimate + rng.standard_normal((draws, EXPECTED_PARAMETERS)) @ factor.T
    qhx_draws = np.empty((draws, EXPECTED_QHX_FAMILY), dtype=np.float64)
    invalid_draws = []
    for index, draw in enumerate(parameter_draws):
        try:
            transformed = np.asarray(fit_candidate._qhx_vector(draw), dtype=np.float64)
            if (transformed.shape != (EXPECTED_QHX_FAMILY,)
                    or not np.isfinite(transformed).all()
                    or np.any(transformed < 0.0)):
                raise ValueError("nonfinite, negative, or wrong-shaped q/h/x draw")
        except (ArithmeticError, FloatingPointError, OverflowError, ValueError) as exc:
            invalid_draws.append((index, type(exc).__name__))
            continue
        qhx_draws[index] = transformed

    denominator = {
        "planned": draws,
        "valid": draws - len(invalid_draws),
        "failed": len(invalid_draws),
        "abstained": 0,
    }
    common = {
        "schema": "wilson-f-lane-gaussian-parameter-maxt-v1",
        "method": "Gaussian fit-coordinate draws with simultaneous max-T studentization",
        "alpha_family": float(alpha),
        "family_size": EXPECTED_QHX_FAMILY,
        "parameter_count": EXPECTED_PARAMETERS,
        "planned_draws": draws,
        "seed": seed,
        "denominators": denominator,
        "certified": False,
        "physical_score": None,
        "confidence_coverage": None,
        "interpretation": (
            "nonlinear covariance propagation under an asymptotic Gaussian fit approximation; "
            "not finite-sample calibrated and not robust to omitted spectral states"
        ),
    }
    if invalid_draws:
        return {
            **common,
            "status": "ABSTAIN_INVALID_PARAMETER_DRAWS",
            "max_t_critical": None,
            "intervals": None,
            "invalid_draw_examples": [
                {"draw_index": index, "reason_type": reason}
                for index, reason in invalid_draws[:10]
            ],
        }

    standard_error = np.std(qhx_draws, axis=0, ddof=1)
    centered = qhx_draws - point[None, :]
    zero = standard_error == 0.0
    if np.any(zero & np.any(np.abs(centered) > 1e-12, axis=0)):
        return {
            **common,
            "status": "ABSTAIN_DEGENERATE_MAX_T_SCALE",
            "max_t_critical": None,
            "intervals": None,
            "invalid_draw_examples": [],
        }
    studentized = np.divide(
        centered, standard_error[None, :], out=np.zeros_like(centered),
        where=standard_error[None, :] > 0.0,
    )
    max_t = np.max(studentized, axis=1)
    critical_raw = float(np.quantile(max_t, 1.0 - float(alpha), method="higher"))
    critical = max(0.0, critical_raw)
    upper = np.maximum(0.0, point + critical * standard_error)
    intervals = [
        {
            "estimate": float(point[index]),
            "gaussian_draw_se": float(standard_error[index]),
            "simultaneous_upper": float(upper[index]),
        }
        for index in range(EXPECTED_QHX_FAMILY)
    ]
    return {
        **common,
        "status": "DIAGNOSTIC_SIMULTANEOUS_UPPER",
        "max_t_critical_raw": critical_raw,
        "max_t_critical": critical,
        "intervals": intervals,
        "invalid_draw_examples": [],
    }
