"""Benchmark scoring for gsfit_2pt submissions."""

from __future__ import annotations

from typing import Any

import numpy as np

from tasks.gsfit_2pt.dataset.synthetic import SyntheticCorrelatorCase, make_synthetic_cases
from tasks.gsfit_2pt.interface import GroundStateFitConfig, Pion2PtGroundStateFit, validate_config


def benchmark_case(
    case: SyntheticCorrelatorCase,
    config: GroundStateFitConfig,
    max_resample_fits: int | None = None,
) -> dict[str, Any]:
    """Benchmark one submission config on one synthetic case."""

    from tasks.gsfit_2pt.scripts.fit import fit_correlator

    validate_config(config, case.lt)

    mean_correlator = np.mean(case.samples, axis=0)
    covariance = np.atleast_2d(np.cov(case.samples, rowvar=False, ddof=1))
    central_result = fit_correlator(mean_correlator, covariance, config, case.lt)
    e0_hat = central_result["energies"][0]["mean"] if central_result["fit_success"] else float("nan")

    sample_limit = len(case.samples) if max_resample_fits is None else min(max_resample_fits, len(case.samples))
    fit_values: list[float] = []
    failure_count = 0
    for sample in case.samples[:sample_limit]:
        sample_result = fit_correlator(sample, covariance, config, case.lt)
        if sample_result["fit_success"] and np.isfinite(sample_result["energies"][0]["mean"]):
            fit_values.append(sample_result["energies"][0]["mean"])
        else:
            failure_count += 1

    sigma_e0 = float(np.std(fit_values, ddof=1)) if len(fit_values) >= 2 else float("inf")
    failure_rate = failure_count / max(sample_limit, 1)
    bias = abs(e0_hat - case.true_e0) if np.isfinite(e0_hat) else float("inf")
    rel_bias = bias / case.true_e0 if np.isfinite(bias) else float("inf")
    rel_sigma = sigma_e0 / case.true_e0 if np.isfinite(sigma_e0) else float("inf")
    chi2_term = abs(central_result["chi2_dof"] - 1.0) if np.isfinite(central_result["chi2_dof"]) else float("inf")
    penalty = 25.0 * rel_bias + 7.0 * rel_sigma + 1.5 * chi2_term + 12.0 * failure_rate
    score = 100.0 / (1.0 + penalty) if np.isfinite(penalty) else 0.0

    return {
        "case_name": case.name,
        "description": case.description,
        "true_e0": float(case.true_e0),
        "e0_hat": float(e0_hat),
        "sigma_e0": sigma_e0,
        "bias": float(bias),
        "relative_bias": float(rel_bias),
        "relative_sigma": float(rel_sigma),
        "chi2_dof": float(central_result["chi2_dof"]),
        "Q": float(central_result.get("Q", 0.0)),
        "failure_rate": float(failure_rate),
        "score": float(score),
        "fit_success": bool(central_result["fit_success"]),
    }


def benchmark_submission(
    submission: Pion2PtGroundStateFit,
    *,
    cases: list[SyntheticCorrelatorCase] | None = None,
    num_samples: int = 24,
    noise_multiplier: float = 1.0,
    max_resample_fits: int | None = None,
) -> dict[str, Any]:
    """Benchmark a submission across all synthetic cases."""

    validate_config(submission.config)
    return benchmark_config(
        submission.config,
        cases=cases,
        num_samples=num_samples,
        noise_multiplier=noise_multiplier,
        max_resample_fits=max_resample_fits,
    )


def benchmark_config(
    config: GroundStateFitConfig,
    *,
    cases: list[SyntheticCorrelatorCase] | None = None,
    num_samples: int = 24,
    noise_multiplier: float = 1.0,
    max_resample_fits: int | None = None,
) -> dict[str, Any]:
    """Benchmark a bare fit configuration across all synthetic cases."""

    validate_config(config)
    benchmark_cases = cases if cases is not None else make_synthetic_cases(
        num_samples=num_samples,
        noise_multiplier=noise_multiplier,
    )

    per_case = [benchmark_case(case, config, max_resample_fits=max_resample_fits) for case in benchmark_cases]
    overall_score = float(np.mean([case["score"] for case in per_case])) if per_case else 0.0
    aggregate_bias = float(np.mean([case["relative_bias"] for case in per_case])) if per_case else float("inf")
    aggregate_sigma = float(np.mean([case["relative_sigma"] for case in per_case])) if per_case else float("inf")
    aggregate_failure = float(np.mean([case["failure_rate"] for case in per_case])) if per_case else float("inf")

    return {
        "score": overall_score,
        "aggregate_relative_bias": aggregate_bias,
        "aggregate_relative_sigma": aggregate_sigma,
        "aggregate_failure_rate": aggregate_failure,
        "num_cases": len(per_case),
        "num_samples": len(benchmark_cases[0].samples) if benchmark_cases else 0,
        "cases": per_case,
    }
