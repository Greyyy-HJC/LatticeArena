"""Momentum-smeared axial-style boosted-pion interpolating operator."""

from __future__ import annotations

from typing import Any

import numpy as np

from tasks.pion_2pt.interface import (
    OperatorComponents,
    PionInterpolatingOperator,
    SubmissionMeta,
)


def _gamma5() -> np.ndarray:
    return np.diag([1.0, 1.0, -1.0, -1.0]).astype(np.complex128)


def _gamma_spatial(index: int) -> np.ndarray:
    zero = 0.0 + 0.0j
    if index == 0:
        return np.array(
            [[zero, zero, zero, 1.0], [zero, zero, 1.0, zero], [zero, -1.0, zero, zero], [-1.0, zero, zero, zero]],
            dtype=np.complex128,
        )
    if index == 1:
        return np.array(
            [[zero, zero, zero, -1.0j], [zero, zero, 1.0j, zero], [zero, 1.0j, zero, zero], [-1.0j, zero, zero, zero]],
            dtype=np.complex128,
        )
    if index == 2:
        return np.array(
            [[zero, zero, 1.0, zero], [zero, zero, zero, -1.0], [-1.0, zero, zero, zero], [zero, 1.0, zero, zero]],
            dtype=np.complex128,
        )
    raise ValueError(f"Unsupported spatial gamma index: {index}")


class AxialSmearedBoostedPion(PionInterpolatingOperator):
    """Gaussian source/sink with an axial-style Dirac structure."""

    def __init__(self, sigma: float = 2.0) -> None:
        self.sigma = sigma
        self._latt_size: tuple[int, int, int, int] | None = None

    @property
    def meta(self) -> SubmissionMeta:
        return SubmissionMeta(
            name="axial_smeared",
            description="Gaussian momentum-smeared profile with a boost-aligned axial bilinear",
            authors=["LatticeArena"],
        )

    def setup(
        self,
        gauge_field: Any,
        latt_size: tuple[int, int, int, int],
        lattice_spacing_fm: float | None,
    ) -> None:
        del gauge_field, lattice_spacing_fm
        self._latt_size = latt_size

    def build(
        self,
        gauge_field: Any,
        momentum_mode: tuple[int, int, int],
        t_source: int,
    ) -> OperatorComponents:
        del gauge_field, t_source
        if self._latt_size is None:
            raise RuntimeError("setup() must be called before build().")

        lx, ly, lz, _ = self._latt_size
        x = np.arange(lx) - lx // 2
        y = np.arange(ly) - ly // 2
        z = np.arange(lz) - lz // 2
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        radial2 = xx**2 + yy**2 + zz**2
        gaussian = np.exp(-0.5 * radial2 / (self.sigma**2))
        phase = np.exp(
            2j
            * np.pi
            * (
                momentum_mode[0] * xx / lx
                + momentum_mode[1] * yy / ly
                + momentum_mode[2] * zz / lz
            )
        )

        profile = (gaussian * phase).astype(np.complex128)
        profile /= np.linalg.norm(profile)

        gamma5 = _gamma5()
        boost_direction = (_gamma_spatial(0) + _gamma_spatial(1) + _gamma_spatial(2)) / np.sqrt(3.0)
        gamma_matrix = boost_direction @ gamma5

        return OperatorComponents(
            source_profile=profile,
            sink_profile=profile,
            gamma_matrix=gamma_matrix,
        )
