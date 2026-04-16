"""Optimization interface for boosted pion two-point operators.

This task's submission contract keeps the PyQUDA solve and contraction inside
the framework while letting submissions design source, sink, and Dirac
structure explicitly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np


@dataclass
class SubmissionMeta:
    """Metadata for a pion 2pt submission."""

    name: str
    description: str
    authors: list[str]


@dataclass(frozen=True)
class PointSourceSpec:
    """A point source at one spatial site on the source timeslice."""

    position: tuple[int, int, int] = (0, 0, 0)


@dataclass(frozen=True)
class ProfileSourceSpec:
    """An arbitrary normalized spatial source profile."""

    profile: np.ndarray


@dataclass(frozen=True)
class PlaneWaveSinkSpec:
    """A plane-wave sink projection in lattice momentum units."""

    momentum_mode: tuple[int, int, int]


@dataclass(frozen=True)
class ProfileSinkSpec:
    """An arbitrary normalized spatial sink profile."""

    profile: np.ndarray


SourceSpec: TypeAlias = PointSourceSpec | ProfileSourceSpec
SinkSpec: TypeAlias = PlaneWaveSinkSpec | ProfileSinkSpec


class PionInterpolatingOperator(ABC):
    """ABC for boosted pion two-point interpolating operators.

    The pion correlator measured by this task is

        C_pi(p, t) = < O_pi(p, t0 + t) O_pi^dag(p, t0) >

    where

        O_pi(p, t) = sum_x exp(i p . x) \bar{d}(x, t) Gamma u(x, t)

    The optimization goal is to improve signal/noise and suppress excited-state
    contamination for boosted pions, with benchmark focus around |p| ~ 1 GeV.
    Submissions provide metadata plus one-time setup and explicit source, sink,
    and Dirac-structure design hooks.
    """

    @property
    @abstractmethod
    def meta(self) -> SubmissionMeta:
        """Return submission metadata."""
        ...

    @abstractmethod
    def setup(
        self,
        gauge_field: Any,
        latt_size: tuple[int, int, int, int],
        lattice_spacing_fm: float | None,
    ) -> None:
        """One-time setup per gauge configuration."""
        ...

    @abstractmethod
    def design_source(
        self,
        gauge_field: Any,
        momentum_mode: tuple[int, int, int],
        t_source: int,
    ) -> SourceSpec:
        """Describe the source to solve from for one momentum/timeslice."""
        ...

    @abstractmethod
    def design_sink(
        self,
        gauge_field: Any,
        momentum_mode: tuple[int, int, int],
        t_source: int,
    ) -> SinkSpec:
        """Describe the sink projection for one momentum/timeslice."""
        ...

    @abstractmethod
    def gamma_matrix(
        self,
        gauge_field: Any,
        momentum_mode: tuple[int, int, int],
        t_source: int,
    ) -> np.ndarray:
        """Return the bilinear Dirac matrix for one momentum/timeslice."""
        ...
