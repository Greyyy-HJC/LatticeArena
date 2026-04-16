"""Baseline boosted-pion interpolating operator.

This baseline mirrors the simplest reference design: a pseudoscalar
interpolator with a point source and a plane-wave sink projection.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tasks.pion_2pt.interface import (
    OperatorComponents,
    PionInterpolatingOperator,
    SubmissionMeta,
)


class PlainBoostedPion(PionInterpolatingOperator):
    """Point-source, pseudoscalar pion interpolating operator."""

    def __init__(self) -> None:
        self._latt_size: tuple[int, int, int, int] | None = None

    @property
    def meta(self) -> SubmissionMeta:
        return SubmissionMeta(
            name="plain",
            description="Point source with plane-wave sink and gamma_5 bilinear",
            authors=["LatticeArena"],
        )

    def setup(
        self,
        gauge_field: Any,
        latt_size: tuple[int, int, int, int],
        lattice_spacing_fm: float | None,
    ) -> None:
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
        x = np.arange(lx)
        y = np.arange(ly)
        z = np.arange(lz)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        sink_profile = np.exp(
            2j
            * np.pi
            * (
                momentum_mode[0] * xx / lx
                + momentum_mode[1] * yy / ly
                + momentum_mode[2] * zz / lz
            )
        )
        sink_profile = sink_profile.astype(np.complex128)
        sink_profile /= np.linalg.norm(sink_profile)

        source_profile = np.zeros((lx, ly, lz), dtype=np.complex128)
        source_profile[0, 0, 0] = 1.0

        gamma5 = np.diag([1.0, 1.0, -1.0, -1.0]).astype(np.complex128)

        return OperatorComponents(
            source_profile=source_profile,
            sink_profile=sink_profile,
            gamma_matrix=gamma5,
        )
