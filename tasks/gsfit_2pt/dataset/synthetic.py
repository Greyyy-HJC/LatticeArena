"""Synthetic dataset definitions and I/O for gsfit_2pt."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


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


def save_synthetic_cases(
    cases: list[SyntheticCorrelatorCase], output_path: str | Path
) -> Path:
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
                true_energies=tuple(
                    np.asarray(
                        payload[entry["true_energies_key"]], dtype=np.float64
                    ).tolist()
                ),
                amplitudes=tuple(
                    np.asarray(
                        payload[entry["amplitudes_key"]], dtype=np.float64
                    ).tolist()
                ),
            )
            for entry in manifest
        ]

    return cases


def _periodic_two_point(
    times: np.ndarray,
    lt: int,
    amplitudes: np.ndarray,
    energies: np.ndarray,
) -> np.ndarray:
    """Periodic multi-state pion 2pt correlator model used for data generation."""

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
    truth = _periodic_two_point(times, lt, np.asarray(amplitudes), np.asarray(energies))

    std = noise_scale * truth * (1.0 + 0.05 * times)
    corr = np.exp(-np.abs(np.subtract.outer(times, times)) / corr_length)
    cov = np.outer(std, std) * corr
    cov += np.eye(lt) * max(np.mean(std**2), 1e-12) * 1e-5

    rng = np.random.default_rng(seed)
    raw = rng.multivariate_normal(
        mean=truth, cov=cov, size=num_samples, check_valid="warn"
    )
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
