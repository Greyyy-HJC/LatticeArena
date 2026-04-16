"""Benchmark scoring for boosted pion two-point correlators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tasks.pion_2pt.benchmark.plots import save_effective_mass_plot
from tasks.pion_2pt.interface import PionInterpolatingOperator
from tasks.pion_2pt.scripts.measure import measure_dataset


def effective_mass_periodic(correlator: np.ndarray) -> np.ndarray:
    """Return periodic/cosh effective masses along the last axis."""

    correlator = np.asarray(correlator, dtype=np.float64)
    masses = np.full(
        (*correlator.shape[:-1], max(correlator.shape[-1] - 2, 0)),
        np.nan,
        dtype=np.float64,
    )
    if correlator.shape[-1] < 3:
        return masses

    center = correlator[..., 1:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (correlator[..., 2:] + correlator[..., :-2]) / (2.0 * center)
    valid = np.isfinite(ratio) & np.isfinite(center) & (center > 0) & (ratio >= 1.0)
    masses[valid] = np.arccosh(ratio[valid])
    return masses


def effective_mass_stderr(per_config_effective_mass: np.ndarray) -> np.ndarray:
    """Return config-to-config standard errors for effective masses."""

    per_config_effective_mass = np.asarray(per_config_effective_mass, dtype=np.float64)
    n_configs = per_config_effective_mass.shape[0]
    if n_configs <= 1:
        return np.zeros_like(per_config_effective_mass[0], dtype=np.float64)
    return np.nanstd(per_config_effective_mass, axis=0, ddof=1) / np.sqrt(n_configs)


def _plateau_window_stats(
    values: np.ndarray, errors: np.ndarray
) -> tuple[float, float]:
    """Return the weighted plateau level and chi^2/dof for one candidate window."""

    safe_errors = np.where(errors > 1e-12, errors, 1e-12)
    weights = 1.0 / np.square(safe_errors)
    plateau_level = float(np.sum(weights * values) / np.sum(weights))
    chi2 = float(np.sum(np.square((values - plateau_level) / safe_errors)))
    dof = max(values.size - 1, 1)
    return plateau_level, chi2 / dof


def detect_plateau_window(
    effective_mass: np.ndarray,
    effective_mass_stderr: np.ndarray,
    effective_mass_times: np.ndarray,
    *,
    min_window_points: int = 3,
    max_plateau_chi2_dof: float = 2.5,
) -> tuple[int, int, float, float]:
    """Detect the earliest stable effective-mass plateau window.

    The search scans forward from small `t_sep` and selects the earliest
    contiguous finite window whose weighted constant fit satisfies the chi^2/dof
    threshold. If no such stable window exists, it falls back to the valid
    window with the smallest chi^2/dof.
    """

    values = np.asarray(effective_mass, dtype=np.float64)
    errors = np.asarray(effective_mass_stderr, dtype=np.float64)
    times = np.asarray(effective_mass_times, dtype=np.int64)

    valid = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    candidates: list[tuple[int, int, float, float]] = []

    for start in range(values.shape[0]):
        if not valid[start]:
            continue
        for stop in range(start + min_window_points, values.shape[0] + 1):
            window = slice(start, stop)
            if not np.all(valid[window]):
                break
            plateau_level, chi2_dof = _plateau_window_stats(
                values[window], errors[window]
            )
            candidates.append((start, stop, plateau_level, chi2_dof))

    if not candidates:
        return 0, 0, float("nan"), float("inf")

    stable_candidates = [
        candidate
        for candidate in candidates
        if np.isfinite(candidate[3]) and candidate[3] <= max_plateau_chi2_dof
    ]
    if stable_candidates:
        stable_candidates.sort(
            key=lambda item: (
                int(times[item[0]]),
                -(item[1] - item[0]),
                item[3],
            )
        )
        return stable_candidates[0]

    candidates.sort(
        key=lambda item: (
            item[3],
            int(times[item[0]]),
            -(item[1] - item[0]),
        )
    )
    return candidates[0]


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

    mean_eff = effective_mass_periodic(mean_real)
    per_config_eff = effective_mass_periodic(real_per_config)
    eff_err = effective_mass_stderr(per_config_eff)
    eff_err_safe = np.where(np.isfinite(eff_err) & (eff_err > 1e-12), eff_err, 1.0)
    eff_times = np.arange(1, lt - 1, dtype=np.int64)

    plateau_chi2 = 0.0
    plateau_dof = 0
    excited_state_proxy_terms = []
    plateau_windows: list[dict[str, float | int]] = []

    for momentum_idx in range(n_momenta):
        start_idx, stop_idx, plateau_level, chi2_dof = detect_plateau_window(
            mean_eff[momentum_idx],
            eff_err_safe[momentum_idx],
            eff_times,
        )
        if stop_idx - start_idx < 2 or not np.isfinite(plateau_level):
            plateau_windows.append(
                {
                    "start_t": -1,
                    "stop_t": -1,
                    "n_points": 0,
                    "plateau_level": float("nan"),
                    "chi2_dof": float("inf"),
                }
            )
            excited_state_proxy_terms.append(float("inf"))
            continue

        window_length = stop_idx - start_idx
        plateau_windows.append(
            {
                "start_t": int(eff_times[start_idx]),
                "stop_t": int(eff_times[stop_idx - 1]),
                "n_points": int(window_length),
                "plateau_level": plateau_level,
                "chi2_dof": chi2_dof,
            }
        )
        plateau_chi2 += chi2_dof * max(window_length - 1, 1)
        plateau_dof += max(window_length - 1, 1)

        early_values = mean_eff[momentum_idx, :start_idx]
        early_valid = np.isfinite(early_values)
        if np.any(early_valid):
            excited_state_proxy_terms.append(
                float(abs(np.nanmean(early_values[early_valid]) - plateau_level))
            )
        else:
            excited_state_proxy_terms.append(0.0)

    plateau_chi2_dof = plateau_chi2 / plateau_dof if plateau_dof else float("inf")
    excited_state_proxy = float(np.nanmean(excited_state_proxy_terms))
    snr_term = float(np.log1p(max(signal_to_noise, 0.0)))
    plateau_term = float(1.0 / (1.0 + max(plateau_chi2_dof, 0.0)))
    excited_state_term = float(1.0 / (1.0 + max(excited_state_proxy, 0.0)))
    score = 4.0 * plateau_term + 4.0 * excited_state_term + 2.0 * snr_term

    return {
        "n_configs": int(n_configs),
        "latt_size": [
            int(value) for value in np.asarray(measured["latt_size"], dtype=np.int64)
        ],
        "source_times": [
            int(value) for value in np.asarray(measured["source_times"], dtype=np.int64)
        ],
        "momentum_modes": [
            [int(component) for component in mode]
            for mode in np.asarray(measured["momentum_modes"], dtype=np.int64)
        ],
        "effective_mass_times": [int(value) for value in eff_times],
        "mean_correlator_real": mean_real.tolist(),
        "mean_correlator_imag": mean_corr.imag.tolist(),
        "effective_mass": mean_eff.tolist(),
        "effective_mass_stderr": eff_err.tolist(),
        "plateau_windows": plateau_windows,
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
    artifact_dir: Path | None = None,
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
    artifacts: dict[str, str] = {}
    if artifact_dir is not None:
        plot_path = save_effective_mass_plot(
            metrics, Path(artifact_dir) / f"{submission.meta.name}_meff.pdf"
        )
        artifacts["effective_mass_plot"] = str(plot_path)
    metrics["artifacts"] = artifacts
    return metrics
