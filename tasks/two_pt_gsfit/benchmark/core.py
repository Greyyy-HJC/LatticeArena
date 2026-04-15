"""Synthetic benchmark and fitting utilities for two_pt_gsfit."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from tasks.two_pt_gsfit.interface import GroundStateFitConfig, Pion2PtGroundStateFit, validate_config


@dataclass(frozen=True)
class SyntheticCorrelatorCase:
    """Synthetic pion 2pt benchmark case."""

    name: str
    description: str
    lt: int
    samples: np.ndarray
    true_energies: tuple[float, ...]
    amplitudes: tuple[float, ...]

    @property
    def true_e0(self) -> float:
        return float(self.true_energies[0])


@dataclass(frozen=True)
class CorrelatorFitResult:
    """Result of fitting one correlator sample."""

    success: bool
    e0: float
    energies: np.ndarray
    amplitudes: np.ndarray
    chi2_dof: float
    chi2: float
    objective: float
    transformed_params: np.ndarray
    message: str


def save_synthetic_cases(cases: list[SyntheticCorrelatorCase], output_path: str | Path) -> Path:
    """Save synthetic benchmark cases to a compressed ``.npz`` archive."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for idx, case in enumerate(cases):
        sample_key = f"case_{idx}_samples"
        energy_key = f"case_{idx}_true_energies"
        amplitude_key = f"case_{idx}_amplitudes"

        arrays[sample_key] = np.asarray(case.samples, dtype=np.float64)
        arrays[energy_key] = np.asarray(case.true_energies, dtype=np.float64)
        arrays[amplitude_key] = np.asarray(case.amplitudes, dtype=np.float64)
        manifest.append(
            {
                "name": case.name,
                "description": case.description,
                "lt": case.lt,
                "samples_key": sample_key,
                "true_energies_key": energy_key,
                "amplitudes_key": amplitude_key,
            }
        )

    arrays["manifest"] = np.array(json.dumps(manifest))
    np.savez_compressed(output, **arrays)
    return output


def load_synthetic_cases(dataset_path: str | Path) -> list[SyntheticCorrelatorCase]:
    """Load synthetic benchmark cases from a saved ``.npz`` archive."""

    dataset = Path(dataset_path)
    with np.load(dataset, allow_pickle=False) as payload:
        manifest = json.loads(payload["manifest"].tolist())
        cases = [
            SyntheticCorrelatorCase(
                name=entry["name"],
                description=entry["description"],
                lt=int(entry["lt"]),
                samples=np.asarray(payload[entry["samples_key"]], dtype=np.float64),
                true_energies=tuple(np.asarray(payload[entry["true_energies_key"]], dtype=np.float64).tolist()),
                amplitudes=tuple(np.asarray(payload[entry["amplitudes_key"]], dtype=np.float64).tolist()),
            )
            for entry in manifest
        ]

    return cases


def periodic_two_point(
    times: np.ndarray,
    lt: int,
    amplitudes: np.ndarray,
    energies: np.ndarray,
) -> np.ndarray:
    """Periodic multi-state pion 2pt correlator model."""

    data = np.zeros_like(times, dtype=np.float64)
    for amp, energy in zip(amplitudes, energies):
        data += amp * (np.exp(-energy * times) + np.exp(-energy * (lt - times)))
    return data


def _build_case(
    name: str,
    description: str,
    energies: tuple[float, ...],
    amplitudes: tuple[float, ...],
    noise_scale: float,
    corr_length: float,
    seed: int,
    lt: int,
    num_samples: int,
) -> SyntheticCorrelatorCase:
    times = np.arange(lt, dtype=np.float64)
    truth = periodic_two_point(times, lt, np.asarray(amplitudes), np.asarray(energies))

    std = noise_scale * truth * (1.0 + 0.05 * times)
    corr = np.exp(-np.abs(np.subtract.outer(times, times)) / corr_length)
    cov = np.outer(std, std) * corr
    cov += np.eye(lt) * max(np.mean(std**2), 1e-12) * 1e-5

    rng = np.random.default_rng(seed)
    raw = rng.multivariate_normal(mean=truth, cov=cov, size=num_samples, check_valid="warn")
    samples = raw.astype(np.float64)

    return SyntheticCorrelatorCase(
        name=name,
        description=description,
        lt=lt,
        samples=samples,
        true_energies=energies,
        amplitudes=amplitudes,
    )


def make_synthetic_cases(
    num_samples: int = 24,
    noise_multiplier: float = 1.0,
    lt: int = 48,
) -> list[SyntheticCorrelatorCase]:
    """Generate the deterministic synthetic benchmark cases."""

    return [
        _build_case(
            name="boosted_clean",
            description="Low-noise pion-like correlator with mild excited-state contamination",
            energies=(0.31, 0.82),
            amplitudes=(0.92, 0.18),
            noise_scale=0.010 * noise_multiplier,
            corr_length=5.0,
            seed=20260415,
            lt=lt,
            num_samples=num_samples,
        ),
        _build_case(
            name="boosted_mixed",
            description="Moderate-noise correlator with visible excited-state overlap",
            energies=(0.36, 0.69, 1.08),
            amplitudes=(0.74, 0.33, 0.10),
            noise_scale=0.018 * noise_multiplier,
            corr_length=4.0,
            seed=20260416,
            lt=lt,
            num_samples=num_samples,
        ),
        _build_case(
            name="boosted_hard",
            description="Hard case with stronger excited-state contamination and noisier late times",
            energies=(0.42, 0.63, 0.95),
            amplitudes=(0.56, 0.36, 0.22),
            noise_scale=0.028 * noise_multiplier,
            corr_length=3.0,
            seed=20260417,
            lt=lt,
            num_samples=num_samples,
        ),
    ]


def _unpack_transformed_params(values: np.ndarray, n_states: int) -> tuple[np.ndarray, np.ndarray]:
    amp_logs = values[:n_states]
    energy_logs = values[n_states:]

    amplitudes = np.exp(amp_logs)
    e0 = np.exp(energy_logs[0])
    deltas = np.exp(energy_logs[1:])

    energies = np.empty(n_states, dtype=np.float64)
    energies[0] = e0
    for idx in range(1, n_states):
        energies[idx] = energies[idx - 1] + deltas[idx - 1]

    return amplitudes, energies


def _pack_initial_guess(config: GroundStateFitConfig) -> np.ndarray:
    amp_init = [max(prior[0], 1e-6) for prior in config.amplitude_priors]
    energy_init = [max(config.e0_prior[0], 1e-6)]
    energy_init.extend(max(prior[0], 1e-6) for prior in config.delta_e_priors)
    return np.log(np.asarray(amp_init + energy_init, dtype=np.float64))


def _prior_penalty(config: GroundStateFitConfig, amplitudes: np.ndarray, energies: np.ndarray) -> float:
    penalty = ((energies[0] - config.e0_prior[0]) / config.e0_prior[1]) ** 2

    for idx in range(1, len(energies)):
        delta_e = energies[idx] - energies[idx - 1]
        mean, width = config.delta_e_priors[idx - 1]
        penalty += ((delta_e - mean) / width) ** 2

    for amp, prior in zip(amplitudes, config.amplitude_priors):
        mean, width = prior
        penalty += ((amp - mean) / width) ** 2

    return float(penalty)


def _fit_objective(
    transformed_params: np.ndarray,
    data: np.ndarray,
    times: np.ndarray,
    lt: int,
    config: GroundStateFitConfig,
    inv_cov: np.ndarray,
) -> float:
    amplitudes, energies = _unpack_transformed_params(transformed_params, config.n_states)
    model = periodic_two_point(times, lt, amplitudes, energies)
    resid = data - model
    chi2 = resid @ inv_cov @ resid
    return float(chi2 + _prior_penalty(config, amplitudes, energies))


def fit_correlator(
    correlator: np.ndarray,
    covariance: np.ndarray,
    lt: int,
    config: GroundStateFitConfig,
    initial_guess: np.ndarray | None = None,
) -> CorrelatorFitResult:
    """Run the framework-owned correlated Bayesian fit."""

    validate_config(config, lt)

    times = np.arange(config.t_min, config.t_max + 1, dtype=np.float64)
    data = np.asarray(correlator[config.t_min : config.t_max + 1], dtype=np.float64)

    cov_slice = np.asarray(covariance, dtype=np.float64)
    if cov_slice.ndim == 0:
        cov_slice = cov_slice.reshape(1, 1)
    if cov_slice.shape != (len(times), len(times)):
        raise ValueError("Covariance shape does not match the fit window.")

    diag_scale = max(float(np.trace(cov_slice)) / max(len(times), 1), 1e-12)
    stabilized_cov = cov_slice + np.eye(len(times), dtype=np.float64) * diag_scale * 1e-6
    inv_cov = np.linalg.pinv(stabilized_cov)

    start = _pack_initial_guess(config) if initial_guess is None else np.asarray(initial_guess, dtype=np.float64)

    result = minimize(
        _fit_objective,
        start,
        args=(data, times, lt, config, inv_cov),
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )

    if not np.all(np.isfinite(result.x)) or not np.isfinite(result.fun):
        return CorrelatorFitResult(
            success=False,
            e0=float("nan"),
            energies=np.array([], dtype=np.float64),
            amplitudes=np.array([], dtype=np.float64),
            chi2_dof=float("inf"),
            chi2=float("inf"),
            objective=float("inf"),
            transformed_params=start,
            message="Non-finite optimizer output.",
        )

    amplitudes, energies = _unpack_transformed_params(result.x, config.n_states)
    model = periodic_two_point(times, lt, amplitudes, energies)
    resid = data - model
    chi2 = float(resid @ inv_cov @ resid)
    n_params = 2 * config.n_states
    dof = max(len(times) - n_params, 1)

    return CorrelatorFitResult(
        success=bool(result.success),
        e0=float(energies[0]),
        energies=energies,
        amplitudes=amplitudes,
        chi2_dof=chi2 / dof,
        chi2=chi2,
        objective=float(result.fun),
        transformed_params=np.asarray(result.x, dtype=np.float64),
        message=str(result.message),
    )


def benchmark_case(
    case: SyntheticCorrelatorCase,
    config: GroundStateFitConfig,
    max_resample_fits: int | None = None,
) -> dict[str, Any]:
    """Benchmark one submission config on one synthetic case."""

    validate_config(config, case.lt)

    window = slice(config.t_min, config.t_max + 1)
    samples_window = case.samples[:, window]
    mean_correlator = np.mean(case.samples, axis=0)
    covariance = np.atleast_2d(np.cov(samples_window, rowvar=False, ddof=1))

    central_fit = fit_correlator(mean_correlator, covariance, case.lt, config)

    sample_limit = len(case.samples) if max_resample_fits is None else min(max_resample_fits, len(case.samples))
    fit_values: list[float] = []
    failure_count = 0
    initial_guess = central_fit.transformed_params if central_fit.success else None

    for sample in case.samples[:sample_limit]:
        sample_fit = fit_correlator(sample, covariance, case.lt, config, initial_guess=initial_guess)
        if sample_fit.success and np.isfinite(sample_fit.e0):
            fit_values.append(sample_fit.e0)
        else:
            failure_count += 1

    sigma_e0 = float(np.std(fit_values, ddof=1)) if len(fit_values) >= 2 else float("inf")
    failure_rate = failure_count / max(sample_limit, 1)
    bias = abs(central_fit.e0 - case.true_e0) if central_fit.success else float("inf")
    rel_bias = bias / case.true_e0 if np.isfinite(bias) else float("inf")
    rel_sigma = sigma_e0 / case.true_e0 if np.isfinite(sigma_e0) else float("inf")
    chi2_term = abs(central_fit.chi2_dof - 1.0) if np.isfinite(central_fit.chi2_dof) else float("inf")
    penalty = 25.0 * rel_bias + 7.0 * rel_sigma + 1.5 * chi2_term + 12.0 * failure_rate
    score = 100.0 / (1.0 + penalty) if np.isfinite(penalty) else 0.0

    return {
        "case_name": case.name,
        "description": case.description,
        "true_e0": float(case.true_e0),
        "e0_hat": float(central_fit.e0),
        "sigma_e0": sigma_e0,
        "bias": float(bias),
        "relative_bias": float(rel_bias),
        "relative_sigma": float(rel_sigma),
        "chi2_dof": float(central_fit.chi2_dof),
        "failure_rate": float(failure_rate),
        "score": float(score),
        "fit_success": bool(central_fit.success),
        "fit_message": central_fit.message,
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
    benchmark_cases = cases if cases is not None else make_synthetic_cases(
        num_samples=num_samples,
        noise_multiplier=noise_multiplier,
    )

    per_case = [benchmark_case(case, submission.config, max_resample_fits=max_resample_fits) for case in benchmark_cases]

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
