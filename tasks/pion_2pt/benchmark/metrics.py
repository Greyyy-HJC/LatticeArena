"""Synthetic benchmark metrics for pion_2pt operators.

Evaluates interpolating operators by generating synthetic correlators whose
excited-state contamination and noise depend on measurable properties of the
submitted operator: ground-state wavefunction overlap, momentum injection
quality, and profile smoothness.

This allows scoring operators without a live QUDA measurement pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tasks.pion_2pt.interface import OperatorComponents, PionInterpolatingOperator


# ---------------------------------------------------------------------------
# Synthetic correlator model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SyntheticPionSetup:
    """Parameters for one synthetic benchmark scenario."""

    name: str
    description: str
    latt_size: tuple[int, int, int, int]
    lattice_spacing_fm: float
    momentum_gev: tuple[float, float, float]
    # Ground-state reference profile (ideal Gaussian width in lattice units)
    ideal_sigma: float
    # True energies for the multi-state model
    true_energies: tuple[float, ...]
    # Amplitudes for the multi-state model (ground + excited)
    true_amplitudes: tuple[float, ...]
    # Noise level per configuration
    noise_sigma: float
    # Number of synthetic configurations
    n_configs: int


def default_scenarios() -> list[SyntheticPionSetup]:
    """Return a set of deterministic benchmark scenarios."""
    return [
        SyntheticPionSetup(
            name="boosted_clean",
            description="Moderately boosted pion, low noise, mild excited-state contamination",
            latt_size=(16, 16, 16, 32),
            lattice_spacing_fm=0.09,
            momentum_gev=(1.0, 0.0, 0.0),
            ideal_sigma=3.0,
            true_energies=(0.35, 0.9),
            true_amplitudes=(1.0, 0.15),
            noise_sigma=0.005,
            n_configs=48,
        ),
        SyntheticPionSetup(
            name="boosted_noisy",
            description="Moderately boosted pion, higher noise, moderate contamination",
            latt_size=(16, 16, 16, 32),
            lattice_spacing_fm=0.09,
            momentum_gev=(1.0, 0.0, 0.0),
            ideal_sigma=3.0,
            true_energies=(0.35, 0.85, 1.4),
            true_amplitudes=(1.0, 0.25, 0.08),
            noise_sigma=0.015,
            n_configs=48,
        ),
        SyntheticPionSetup(
            name="high_momentum",
            description="Higher momentum pion, challenging SNR",
            latt_size=(16, 16, 16, 32),
            lattice_spacing_fm=0.09,
            momentum_gev=(1.5, 0.0, 0.0),
            ideal_sigma=2.5,
            true_energies=(0.50, 1.1, 1.7),
            true_amplitudes=(1.0, 0.30, 0.10),
            noise_sigma=0.025,
            n_configs=48,
        ),
    ]


def _periodic_correlator(
    energies: np.ndarray,
    amplitudes: np.ndarray,
    lt: int,
    t_values: np.ndarray,
) -> np.ndarray:
    """Evaluate the periodic two-point correlator model.

    C(t) = sum_n A_n [exp(-E_n t) + exp(-E_n (Lt - t))]
    """
    corr = np.zeros(len(t_values), dtype=np.float64)
    for en, amp in zip(energies, amplitudes):
        corr += amp * (np.exp(-en * t_values) + np.exp(-en * (lt - t_values)))
    return corr


def _compute_overlap_quality(
    components: OperatorComponents,
    setup: SyntheticPionSetup,
) -> float:
    """Score how well the operator profile overlaps with an ideal ground-state.

    Returns a value in [0, 1] where 1 is perfect overlap.
    """
    lx, ly, lz, _ = setup.latt_size
    x = np.arange(lx) - lx // 2
    y = np.arange(ly) - ly // 2
    z = np.arange(lz) - lz // 2
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    r2 = xx**2 + yy**2 + zz**2
    ideal_profile = np.exp(-0.5 * r2 / setup.ideal_sigma**2)

    # Include momentum phase in ideal
    hbarc = 0.1973269804
    a_over_hbarc = setup.lattice_spacing_fm / hbarc
    px, py, pz = setup.momentum_gev
    phase = np.exp(1j * (px * a_over_hbarc * xx + py * a_over_hbarc * yy + pz * a_over_hbarc * zz))
    ideal = (ideal_profile * phase).astype(np.complex128)
    ideal /= np.linalg.norm(ideal)

    src = components.source_profile
    src_norm = np.linalg.norm(src)
    if src_norm > 0:
        src = src / src_norm

    overlap = float(np.abs(np.sum(np.conj(ideal) * src)))
    return np.clip(overlap, 0.0, 1.0)


def _compute_smoothness(profile: np.ndarray) -> float:
    """Measure profile smoothness (lower roughness = better noise suppression).

    Returns a value in [0, 1] where 1 is maximally smooth.
    """
    grad_x = np.diff(profile, axis=0)
    grad_y = np.diff(profile, axis=1)
    grad_z = np.diff(profile, axis=2)
    roughness = (
        np.mean(np.abs(grad_x)**2) + np.mean(np.abs(grad_y)**2) + np.mean(np.abs(grad_z)**2)
    )
    return float(1.0 / (1.0 + roughness))


def generate_synthetic_correlator(
    components: OperatorComponents,
    setup: SyntheticPionSetup,
    seed: int,
) -> dict[str, Any]:
    """Generate a synthetic correlator whose quality depends on operator properties.

    The overlap quality modulates excited-state contamination, and the profile
    smoothness modulates the noise level.
    """
    rng = np.random.default_rng(seed)
    lt = setup.latt_size[3]
    t_values = np.arange(lt)

    overlap = _compute_overlap_quality(components, setup)
    smoothness = _compute_smoothness(components.source_profile)

    # Better overlap suppresses excited states
    energies = np.array(setup.true_energies)
    amplitudes = np.array(setup.true_amplitudes, dtype=np.float64)
    if len(amplitudes) > 1:
        suppression = overlap**2
        amplitudes[1:] *= (1.0 - 0.7 * suppression)

    # Smoother profiles reduce noise
    effective_noise = setup.noise_sigma * (1.0 + 2.0 * (1.0 - smoothness))

    mean_corr = _periodic_correlator(energies, amplitudes, lt, t_values.astype(np.float64))

    per_config = np.empty((setup.n_configs, lt), dtype=np.float64)
    for cfg in range(setup.n_configs):
        noise = rng.normal(0.0, effective_noise * np.abs(mean_corr), size=lt)
        per_config[cfg] = mean_corr + noise

    return {
        "mean": np.mean(per_config, axis=0),
        "per_config": per_config,
        "t_values": t_values,
        "overlap": overlap,
        "smoothness": smoothness,
        "true_e0": float(energies[0]),
        "n_configs": setup.n_configs,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _effective_mass(corr: np.ndarray) -> np.ndarray:
    """Log-ratio effective mass: m_eff(t) = -ln(C(t+1)/C(t))."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = corr[1:] / corr[:-1]
        meff = np.where((ratio > 0) & np.isfinite(ratio), -np.log(ratio), np.nan)
    return meff


def compute_metrics(measured: dict[str, Any]) -> dict[str, Any]:
    """Compute benchmark metrics from synthetic correlator data."""
    per_config = np.asarray(measured["per_config"])
    mean_corr = np.asarray(measured["mean"])
    n_configs = per_config.shape[0]
    t_values = np.asarray(measured["t_values"])

    # SNR per timeslice
    if n_configs > 1:
        corr_std = np.std(per_config, axis=0, ddof=1)
    else:
        corr_std = np.ones_like(mean_corr)
    corr_std_safe = np.maximum(corr_std, 1e-15)
    snr = np.abs(mean_corr) / corr_std_safe

    # Effective mass from mean correlator
    meff = _effective_mass(mean_corr)

    # Per-config effective masses for error estimation
    per_config_meff = np.array([_effective_mass(cfg) for cfg in per_config])
    meff_err = np.nanstd(per_config_meff, axis=0, ddof=1) / np.sqrt(max(n_configs, 1))
    meff_err = np.where(np.isfinite(meff_err) & (meff_err > 1e-12), meff_err, 1.0)

    # Plateau quality: chi2/dof of constant fit to meff in a window
    lt = len(mean_corr)
    t_start = max(3, lt // 8)
    t_end = min(lt // 2 - 1, len(meff))
    plateau_slice = meff[t_start:t_end]
    plateau_err_slice = meff_err[t_start:t_end]
    valid = np.isfinite(plateau_slice)

    if np.count_nonzero(valid) >= 2:
        vals = plateau_slice[valid]
        errs = plateau_err_slice[valid]
        const = np.average(vals, weights=1.0 / errs**2)
        chi2 = float(np.sum(((vals - const) / errs) ** 2))
        dof = max(len(vals) - 1, 1)
        plateau_chi2_dof = chi2 / dof
        plateau_level = float(const)
    else:
        plateau_chi2_dof = float("inf")
        plateau_level = float("nan")

    # Effective mass roughness
    if len(meff) > 1:
        finite_meff = meff[np.isfinite(meff)]
        roughness = float(np.mean(np.abs(np.diff(finite_meff)))) if len(finite_meff) > 1 else 0.0
    else:
        roughness = 0.0

    # Mean SNR (skip t=0 which is trivially large)
    mean_snr = float(np.mean(snr[1:t_end + 1])) if t_end > 0 else float(np.mean(snr))

    # Excited-state contamination proxy: deviation of early meff from plateau
    if np.isfinite(plateau_level) and len(meff) > 2:
        early_meff = meff[:t_start]
        early_valid = np.isfinite(early_meff)
        if np.any(early_valid):
            contamination = float(np.mean(np.abs(early_meff[early_valid] - plateau_level)))
        else:
            contamination = float("inf")
    else:
        contamination = float("inf")

    # Composite score
    snr_term = float(np.log1p(max(mean_snr, 0.0)))
    plateau_term = float(1.0 / (1.0 + max(plateau_chi2_dof, 0.0)))
    roughness_term = float(1.0 / (1.0 + max(roughness, 0.0)))
    contamination_term = float(1.0 / (1.0 + max(contamination, 0.0)))

    score = (
        3.0 * plateau_term
        + 3.0 * roughness_term
        + 2.0 * snr_term
        + 2.0 * contamination_term
    )

    return {
        "n_configs": int(n_configs),
        "overlap": float(measured.get("overlap", 0.0)),
        "smoothness": float(measured.get("smoothness", 0.0)),
        "true_e0": float(measured.get("true_e0", 0.0)),
        "signal_to_noise": mean_snr,
        "plateau_chi2_dof": plateau_chi2_dof,
        "plateau_level": plateau_level,
        "effective_mass_roughness": roughness,
        "excited_state_contamination": contamination,
        "score_terms": {
            "snr_term": snr_term,
            "plateau_term": plateau_term,
            "roughness_term": roughness_term,
            "contamination_term": contamination_term,
        },
        "score": float(score),
    }


def benchmark_submission(
    operator: PionInterpolatingOperator,
    scenarios: list[SyntheticPionSetup] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full synthetic benchmark for a pion_2pt operator submission.

    Returns aggregate and per-scenario metrics.
    """
    if scenarios is None:
        scenarios = default_scenarios()

    from latticearena.testing import identity_gauge_field

    per_scenario: list[dict[str, Any]] = []
    scores: list[float] = []

    for idx, setup in enumerate(scenarios):
        gauge = identity_gauge_field(setup.latt_size)
        operator.setup(gauge, latt_size=setup.latt_size, lattice_spacing_fm=setup.lattice_spacing_fm)
        components = operator.build(gauge, momentum_gev=setup.momentum_gev, t_source=0)

        measured = generate_synthetic_correlator(components, setup, seed=seed + idx)
        metrics = compute_metrics(measured)
        metrics["scenario"] = setup.name
        per_scenario.append(metrics)
        scores.append(metrics["score"])

    aggregate_score = float(np.mean(scores))

    return {
        "score": aggregate_score,
        "per_scenario": per_scenario,
        "n_scenarios": len(scenarios),
    }
