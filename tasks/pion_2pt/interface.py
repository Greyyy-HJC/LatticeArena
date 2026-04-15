"""Optimization interface for boosted pion two-point operators.

Contributors implement :class:`PionInterpolatingOperator` to design source/sink
interpolating operators for pion two-point correlators at finite momentum.
The flagship benchmark target is a pion with |p| ~= 1 GeV.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class OperatorMeta:
    """Metadata for an operator submission."""

    name: str
    description: str
    authors: list[str]


@dataclass
class OperatorComponents:
    """Building blocks used by the pion 2pt measurement pipeline.

    Attributes:
        source_profile:
            Complex profile on the source timeslice, shape (Lx, Ly, Lz).
            Includes spatial smearing and momentum phase factors.
        sink_profile:
            Complex profile on the sink timeslice, shape (Lx, Ly, Lz).
            Can differ from source_profile for variational/distillation-inspired
            designs.
        gamma_matrix:
            Dirac matrix defining the bilinear, shape (Ns, Ns) complex.
            Ns=4 in standard relativistic formulations.
    """

    source_profile: np.ndarray
    sink_profile: np.ndarray
    gamma_matrix: np.ndarray


class PionInterpolatingOperator(ABC):
    """ABC for boosted pion two-point interpolating operators.

    The pion correlator measured by this task is

        C_pi(p, t) = < O_pi(p, t0 + t) O_pi^dag(p, t0) >

    where

        O_pi(p, t) = sum_x exp(i p . x) \bar{d}(x, t) Gamma u(x, t)

    The optimization goal is to improve signal/noise and suppress excited-state
    contamination for boosted pions, with benchmark focus around |p| ~ 1 GeV.
    """

    @property
    @abstractmethod
    def meta(self) -> OperatorMeta:
        """Return operator metadata."""
        ...

    @abstractmethod
    def setup(
        self,
        gauge_field: np.ndarray,
        latt_size: tuple[int, int, int, int],
        lattice_spacing_fm: float,
    ) -> None:
        """One-time setup per gauge configuration.

        Args:
            gauge_field: Gauge links in natural ordering,
                shape (Nd, Lx, Ly, Lz, Lt, Nc, Nc), complex128.
            latt_size: (Lx, Ly, Lz, Lt).
            lattice_spacing_fm: Lattice spacing in femtometers.
        """
        ...

    @abstractmethod
    def build(
        self,
        gauge_field: np.ndarray,
        momentum_gev: tuple[float, float, float],
        t_source: int,
    ) -> OperatorComponents:
        """Build source/sink operator components for a given momentum.

        Args:
            gauge_field: Gauge links in natural ordering.
            momentum_gev: Target pion momentum (px, py, pz) in GeV.
            t_source: Source timeslice.

        Returns:
            OperatorComponents with source/sink profiles and gamma matrix.
        """
        ...
