"""Benchmark scoring for boosted pion two-point correlators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tasks.pion_2pt.interface import PionInterpolatingOperator
from tasks.pion_2pt.scripts.measure import measure_dataset


def _effective_mass(correlator: np.ndarray) -> np.ndarray:
    masses = np.full(max(correlator.shape[-1] - 1, 0), np.nan, dtype=np.float64)
    if correlator.shape[-1] < 2:
        return masses

    numerator = correlator[:-1]
    denominator = correlator[1:]
    valid = (numerator > 0) & (denominator > 0)
    masses[valid] = -np.log(denominator[valid] / numerator[valid])
    return masses


def compute_metrics(measured: dict[str, Any]) -> dict[str, Any]:
    """Compute signal/noise and excited-state proxies from measured correlators."""

    per_config = np.asarray(measured["per_config"], dtype=np.complex128)
    mean_corr = np.asarray(measured["mean"], dtype=np.complex128)
    real_per_config = per_config.real
    mean_real = mean_corr.real

    n_configs, n_momenta, lt = real_per_config.shape
    corr_std = (
        np.std(real_per_config, axis=0, ddof=1)
        if n_configs > 1
        else np.ones_like(mean_real)
    )
    corr_std_safe = np.maximum(corr_std, 1e-12)

    t_min = 2 if lt > 4 else 1
    t_max = max(lt // 2, t_min + 1)
    snr_curve = np.abs(mean_real) / corr_std_safe
    signal_to_noise = float(np.nanmean(snr_curve[:, t_min:t_max]))

    mean_eff = np.asarray([_effective_mass(corr) for corr in mean_real], dtype=np.float64)
    per_config_eff = np.asarray(
        [[_effective_mass(corr) for corr in cfg] for cfg in real_per_config],
        dtype=np.float64,
    )
    eff_err = (
        np.nanstd(per_config_eff, axis=0, ddof=1) / np.sqrt(n_configs)
        if n_configs > 1
        else np.ones_like(mean_eff)
    )
    eff_err = np.where(np.isfinite(eff_err) & (eff_err > 1e-12), eff_err, 1.0)

    plateau_start = max(2, (lt - 1) // 3)
    plateau_stop = max(plateau_start + 2, lt // 2)
    plateau_chi2 = 0.0
    plateau_dof = 0
    excited_state_proxy_terms = []

    for momentum_idx in range(n_momenta):
        values = mean_eff[momentum_idx, plateau_start:plateau_stop]
        errors = eff_err[momentum_idx, plateau_start:plateau_stop]
        valid = np.isfinite(values)
        if np.count_nonzero(valid) < 2:
            excited_state_proxy_terms.append(float("inf"))
            continue

        plateau_level = float(np.nanmean(values[valid]))
        plateau_chi2 += float(np.sum(((values[valid] - plateau_level) / errors[valid]) ** 2))
        plateau_dof += max(np.count_nonzero(valid) - 1, 1)

        early_values = mean_eff[momentum_idx, 1 : 1 + np.count_nonzero(valid)]
        early_valid = np.isfinite(early_values)
        if np.any(early_valid):
            excited_state_proxy_terms.append(
                float(abs(np.nanmean(early_values[early_valid]) - plateau_level))
            )

    plateau_chi2_dof = plateau_chi2 / plateau_dof if plateau_dof else float("inf")
    excited_state_proxy = float(np.nanmean(excited_state_proxy_terms))
    snr_term = float(np.log1p(max(signal_to_noise, 0.0)))
    plateau_term = float(1.0 / (1.0 + max(plateau_chi2_dof, 0.0)))
    excited_state_term = float(1.0 / (1.0 + max(excited_state_proxy, 0.0)))
    score = 4.0 * plateau_term + 4.0 * excited_state_term + 2.0 * snr_term

    return {
        "n_configs": int(n_configs),
        "latt_size": [int(value) for value in np.asarray(measured["latt_size"], dtype=np.int64)],
        "source_times": [
            int(value) for value in np.asarray(measured["source_times"], dtype=np.int64)
        ],
        "momentum_modes": [
            [int(component) for component in mode]
            for mode in np.asarray(measured["momentum_modes"], dtype=np.int64)
        ],
        "mean_correlator_real": mean_real.tolist(),
        "mean_correlator_imag": mean_corr.imag.tolist(),
        "effective_mass": mean_eff.tolist(),
        "signal_to_noise": signal_to_noise,
        "plateau_chi2_dof": plateau_chi2_dof,
        "excited_state_proxy": excited_state_proxy,
        "score_terms": {
            "signal_to_noise_term": snr_term,
            "plateau_term": plateau_term,
            "excited_state_term": excited_state_term,
        },
        "score": float(score),
    }


def benchmark_submission(
    submission: PionInterpolatingOperator,
    dataset_path: str | Path,
    *,
    momentum_modes: list[tuple[int, int, int]] | None = None,
    source_times: list[int] | None = None,
    max_configs: int | None = None,
    resource_path: Path | None = None,
) -> dict[str, Any]:
    """Measure one submission and compute benchmark metrics."""

    measured = measure_dataset(
        dataset_path=Path(dataset_path),
        operator=submission,
        momentum_modes=momentum_modes,
        source_times=source_times,
        max_configs=max_configs,
        resource_path=resource_path,
    )
    metrics = compute_metrics(measured)
    metrics["ensemble_name"] = measured["ensemble_name"]
    metrics["files"] = list(measured["files"])
    return metrics
