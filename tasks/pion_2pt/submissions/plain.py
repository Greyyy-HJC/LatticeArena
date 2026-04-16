"""Baseline boosted-pion interpolating operator.

This baseline uses a gauge-blind Gaussian profile multiplied by a plane-wave
phase, with the conventional pseudoscalar Dirac structure Gamma = gamma_5.
"""

from __future__ import annotations

import numpy as np

from tasks.pion_2pt.interface import OperatorComponents, PionInterpolatingOperator, SubmissionMeta


class PlainBoostedPion(PionInterpolatingOperator):
    """Simple Gaussian x plane-wave pion interpolating operator."""

    def __init__(self, sigma: float = 2.0) -> None:
        self.sigma = sigma
        self._latt_size: tuple[int, int, int, int] | None = None
        self._a_fm: float | None = None

    @property
    def meta(self) -> SubmissionMeta:
        return SubmissionMeta(
            name="plain",
            description="Gaussian profile with momentum phase and gamma_5 bilinear",
            authors=["LatticeArena"],
        )

    def setup(
        self,
        gauge_field: np.ndarray,
        latt_size: tuple[int, int, int, int],
        lattice_spacing_fm: float,
    ) -> None:
        self._latt_size = latt_size
        self._a_fm = lattice_spacing_fm

    def build(
        self,
        gauge_field: np.ndarray,
        momentum_gev: tuple[float, float, float],
        t_source: int,
    ) -> OperatorComponents:
        if self._latt_size is None or self._a_fm is None:
            raise RuntimeError("setup() must be called before build().")

        lx, ly, lz, _ = self._latt_size
        x = np.arange(lx) - lx // 2
        y = np.arange(ly) - ly // 2
        z = np.arange(lz) - lz // 2
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        radial2 = xx**2 + yy**2 + zz**2
        gaussian = np.exp(-0.5 * radial2 / (self.sigma**2))

        hbarc = 0.1973269804  # GeV*fm
        a_over_hbarc = self._a_fm / hbarc
        phase = np.exp(
            1j
            * (
                momentum_gev[0] * a_over_hbarc * xx
                + momentum_gev[1] * a_over_hbarc * yy
                + momentum_gev[2] * a_over_hbarc * zz
            )
        )

        profile = gaussian * phase
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm

        gamma5 = np.diag([1.0, 1.0, -1.0, -1.0]).astype(np.complex128)

        return OperatorComponents(
            source_profile=profile.astype(np.complex128),
            sink_profile=profile.astype(np.complex128),
            gamma_matrix=gamma5,
        )
