"""Optimization interface for pion two-point ground-state fitting.

Contributors do not implement the fitter itself. Instead, they submit a fixed
fit configuration that the framework evaluates on synthetic pion 2pt
correlator samples with known truth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real


DEFAULT_TEMPORAL_EXTENT = 48


@dataclass(frozen=True)
class SubmissionMeta:
    """Metadata for a fit-configuration submission."""

    name: str
    description: str
    authors: list[str]


@dataclass(frozen=True)
class GroundStateFitConfig:
    """Fixed fit settings evaluated by the task benchmark."""

    t_min: int
    t_max: int
    n_states: int
    e0_prior: tuple[float, float]
    delta_e_priors: list[tuple[float, float]]
    amplitude_priors: list[tuple[float, float]]


def config_to_dict(config: GroundStateFitConfig) -> dict[str, object]:
    """Serialize a fit configuration to a JSON-friendly dict."""

    return {
        "t_min": config.t_min,
        "t_max": config.t_max,
        "n_states": config.n_states,
        "e0_prior": [float(config.e0_prior[0]), float(config.e0_prior[1])],
        "delta_e_priors": [[float(mean), float(width)] for mean, width in config.delta_e_priors],
        "amplitude_priors": [[float(mean), float(width)] for mean, width in config.amplitude_priors],
    }


def config_from_dict(payload: dict[str, object]) -> GroundStateFitConfig:
    """Load a fit configuration from a JSON-friendly dict."""

    return GroundStateFitConfig(
        t_min=int(payload["t_min"]),
        t_max=int(payload["t_max"]),
        n_states=int(payload["n_states"]),
        e0_prior=tuple(float(value) for value in payload["e0_prior"]),
        delta_e_priors=[tuple(float(value) for value in prior) for prior in payload["delta_e_priors"]],
        amplitude_priors=[tuple(float(value) for value in prior) for prior in payload["amplitude_priors"]],
    )


class Pion2PtGroundStateFit(ABC):
    """ABC for pion ground-state fit configuration submissions."""

    @property
    @abstractmethod
    def meta(self) -> SubmissionMeta:
        """Return submission metadata."""
        ...

    @property
    @abstractmethod
    def config(self) -> GroundStateFitConfig:
        """Return the fixed fit configuration to benchmark."""
        ...


def _validate_prior(name: str, prior: tuple[float, float]) -> None:
    if len(prior) != 2:
        raise ValueError(f"{name} must be a (mean, width) pair.")

    mean, width = prior
    if not isinstance(mean, Real) or not isinstance(width, Real):
        raise ValueError(f"{name} must contain numeric values.")
    if width <= 0:
        raise ValueError(f"{name} width must be strictly positive.")


def validate_config(config: GroundStateFitConfig, lt: int = DEFAULT_TEMPORAL_EXTENT) -> None:
    """Raise ``ValueError`` if a fit configuration is invalid."""

    if config.n_states < 1:
        raise ValueError("n_states must be at least 1.")
    if config.t_min < 0 or config.t_max < 0:
        raise ValueError("Fit range must be non-negative.")
    if config.t_min >= config.t_max:
        raise ValueError("Fit range must satisfy t_min < t_max.")
    if config.t_max >= lt:
        raise ValueError(f"Fit range must stay within the temporal extent Lt={lt}.")
    if len(config.delta_e_priors) != config.n_states - 1:
        raise ValueError("delta_e_priors length must equal n_states - 1.")
    if len(config.amplitude_priors) != config.n_states:
        raise ValueError("amplitude_priors length must equal n_states.")

    _validate_prior("e0_prior", config.e0_prior)
    for idx, prior in enumerate(config.delta_e_priors):
        _validate_prior(f"delta_e_priors[{idx}]", prior)
    for idx, prior in enumerate(config.amplitude_priors):
        _validate_prior(f"amplitude_priors[{idx}]", prior)
