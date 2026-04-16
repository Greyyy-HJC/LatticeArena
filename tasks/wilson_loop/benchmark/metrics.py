"""Wilson-loop benchmark metrics and scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tasks.wilson_loop.interface import SpatialOperator
from tasks.wilson_loop.scripts.measure import measure_dataset


def _effective_mass(correlator: np.ndarray) -> np.ndarray:
    """Compute log-ratio effective masses from a real correlator array."""

    masses = np.full(
        (correlator.shape[0], max(correlator.shape[1] - 1, 0)), np.nan, dtype=np.float64
    )
    if correlator.shape[1] < 2:
        return masses

    numerator = correlator[:, :-1]
    denominator = correlator[:, 1:]
    valid = (numerator > 0) & (denominator > 0)
    masses[valid] = -np.log(denominator[valid] / numerator[valid])
    return masses


def compute_metrics(
    measured: dict[str, np.ndarray | list[str] | list[int]],
) -> dict[str, Any]:
    """Compute v0 Wilson-loop benchmark metrics from measured correlators."""

    per_config = np.asarray(measured["per_config"], dtype=np.complex128)
    mean_corr = np.asarray(measured["mean"], dtype=np.complex128)
    real_per_config = per_config.real
    mean_real = mean_corr.real

    n_configs = real_per_config.shape[0]
    if n_configs > 1:
        corr_std = np.std(real_per_config, axis=0, ddof=1)
    else:
        corr_std = np.zeros_like(mean_real)

    corr_std_safe = np.maximum(corr_std, 1e-12)
    snr_points = np.clip(np.abs(mean_real) / corr_std_safe, 0.0, 1e3)

    mean_eff = _effective_mass(mean_real)
    per_config_eff = np.asarray(
        [_effective_mass(cfg) for cfg in real_per_config], dtype=np.float64
    )
    eff_err = (
        np.nanstd(per_config_eff, axis=0, ddof=1) / np.sqrt(max(n_configs, 1))
        if n_configs > 1
        else np.ones_like(mean_eff)
    )
    eff_err = np.where(np.isfinite(eff_err) & (eff_err > 1e-12), eff_err, 1.0)

    plateau_values = mean_eff[np.isfinite(mean_eff)]
    plateau_level = (
        float(np.nanmean(plateau_values)) if plateau_values.size else float("nan")
    )

    plateau_chi2 = 0.0
    plateau_dof = 0
    for ridx in range(mean_eff.shape[0]):
        valid = np.isfinite(mean_eff[ridx])
        if np.count_nonzero(valid) < 2:
            continue
        values = mean_eff[ridx, valid]
        errors = eff_err[ridx, valid]
        const = np.mean(values)
        plateau_chi2 += float(np.sum(((values - const) / errors) ** 2))
        plateau_dof += max(len(values) - 1, 1)
    plateau_chi2_dof = plateau_chi2 / plateau_dof if plateau_dof else float("inf")

    roughness = (
        float(np.nanmean(np.abs(np.diff(mean_eff, axis=1))))
        if mean_eff.shape[1] > 1
        else 0.0
    )
    signal_to_noise = (
        float(np.nanmean(snr_points[:, 1:]))
        if snr_points.shape[1] > 1
        else float(np.nanmean(snr_points))
    )

    overlap_proxy_values = []
    t_values = np.asarray(measured["t_values"], dtype=np.int64)
    for ridx in range(mean_real.shape[0]):
        valid = np.isfinite(mean_eff[ridx])
        if not np.any(valid) or mean_real[ridx, 0] <= 0:
            continue
        m_ref = float(np.nanmedian(mean_eff[ridx, valid]))
        ratios = mean_real[ridx, 1:] / mean_real[ridx, [0]]
        proxy = ratios * np.exp(m_ref * t_values[1:])
        overlap_proxy_values.extend(proxy[np.isfinite(proxy)].tolist())
    ground_state_overlap_proxy = (
        float(np.mean(overlap_proxy_values)) if overlap_proxy_values else 0.0
    )

    score_terms = {
        "signal_to_noise_term": float(np.log1p(max(signal_to_noise, 0.0))),
        "plateau_term": float(1.0 / (1.0 + max(plateau_chi2_dof, 0.0))),
        "roughness_term": float(1.0 / (1.0 + max(roughness, 0.0))),
        "overlap_term": float(max(ground_state_overlap_proxy, 0.0)),
    }
    score = (
        4.0 * score_terms["plateau_term"]
        + 3.0 * score_terms["roughness_term"]
        + 1.5 * score_terms["signal_to_noise_term"]
        + 1.0 * score_terms["overlap_term"]
    )

    return {
        "n_configs": int(n_configs),
        "r_values": [
            int(value) for value in np.asarray(measured["r_values"], dtype=np.int64)
        ],
        "t_values": [
            int(value) for value in np.asarray(measured["t_values"], dtype=np.int64)
        ],
        "mean_correlator_real": mean_real.tolist(),
        "mean_correlator_imag": mean_corr.imag.tolist(),
        "effective_mass": mean_eff.tolist(),
        "signal_to_noise": signal_to_noise,
        "plateau_chi2_dof": plateau_chi2_dof,
        "plateau_level": plateau_level,
        "effective_mass_roughness": roughness,
        "ground_state_overlap_proxy": ground_state_overlap_proxy,
        "score_terms": score_terms,
        "score": float(score),
    }


def benchmark_submission(
    operator: SpatialOperator,
    dataset_path: str | Path,
    *,
    r_values: list[int],
    t_values: list[int],
    max_configs: int | None = None,
) -> dict[str, Any]:
    """Measure correlators for one operator and compute benchmark metrics."""

    measured = measure_dataset(
        dataset_path=Path(dataset_path),
        operator=operator,
        r_values=r_values,
        t_values=t_values,
        max_configs=max_configs,
    )
    metrics = compute_metrics(measured)
    metrics["files"] = list(measured["files"])
    return metrics
