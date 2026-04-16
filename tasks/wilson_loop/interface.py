"""Optimization interface for the Wilson loop task.

Contributors implement the SpatialOperator ABC to define custom spatial
Wilson line operators. The framework calls setup() once per gauge config,
then compute() for each (r, direction, t) to build the full Wilson loop
correlator.

Reference: arXiv:2602.02436 - "Wilson loops with neural networks"
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class SubmissionMeta:
    """Metadata for a Wilson-loop submission."""

    name: str
    description: str
    authors: list[str]


class SpatialOperator(ABC):
    """Abstract base class for spatial Wilson line operators.

    The spatial operator replaces the straight Wilson line S(x, x+r*e_dir, t)
    in the Wilson loop W_{r x t}. The Wilson loop correlator is:

        <tr W_{r x t}> = sum_n |c_n|^2 exp(-t * a * E_n(r))

    A better operator maximizes the ground-state overlap |c_0|^2, which
    produces an earlier effective-mass plateau and better signal-to-noise.

    Gauge covariance requirement:
        Under a gauge transformation U_mu(x) -> G(x) U_mu(x) G^dag(x+e_mu),
        the operator output must transform as:

            S_hat(x) -> G(x, t) @ S_hat(x) @ G^dag(x + r*e_dir, t)

        This is automatically satisfied by any product of gauge links along
        a path from x to x + r*e_dir, or any sum of such products.
    """

    @property
    @abstractmethod
    def meta(self) -> SubmissionMeta:
        """Return submission metadata."""
        ...

    @abstractmethod
    def setup(self, gauge_field: np.ndarray, latt_size: tuple[int, int, int, int]) -> None:
        """One-time setup for a gauge configuration.

        Called once before any compute() calls on this configuration.
        Use for gauge fixing, link smearing, precomputation, etc.

        Args:
            gauge_field: Gauge links in natural ordering,
                shape (Nd, Lx, Ly, Lz, Lt, Nc, Nc) complex128.
                Nd=4 directions (x, y, z, t), Nc=3 colors.
            latt_size: (Lx, Ly, Lz, Lt) lattice dimensions.
        """
        ...

    @abstractmethod
    def compute(
        self,
        gauge_field: np.ndarray,
        r: int,
        direction: int,
        t: int,
    ) -> np.ndarray:
        """Compute the spatial operator at all spatial sites on timeslice t.

        Args:
            gauge_field: Gauge links, shape (Nd, Lx, Ly, Lz, Lt, Nc, Nc).
            r: Spatial separation in lattice units (r >= 1).
            direction: Spatial direction index (0=x, 1=y, 2=z).
            t: Timeslice index (0 <= t < Lt).

        Returns:
            np.ndarray of shape (Lx, Ly, Lz, Nc, Nc) complex128.
            The 3x3 color matrix at each spatial site, representing the
            spatial connection from site x to site x + r*e_dir on timeslice t.
        """
        ...
