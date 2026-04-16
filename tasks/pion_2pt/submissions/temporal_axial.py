"""Temporal-axial boosted-pion interpolating operator."""

from __future__ import annotations

from typing import Any

import numpy as np

from tasks.pion_2pt.dirac import gamma_t_gamma5_matrix
from tasks.pion_2pt.interface import (
    PlaneWaveSinkSpec,
    PionInterpolatingOperator,
    PointSourceSpec,
    SinkSpec,
    SourceSpec,
    SubmissionMeta,
)


class TemporalAxialBoostedPion(PionInterpolatingOperator):
    """Point-source, pseudoscalar pion interpolating operator."""

    def __init__(self) -> None:
        self._is_setup = False

    @property
    def meta(self) -> SubmissionMeta:
        return SubmissionMeta(
            name="temporal_axial",
            description="Point source with plane-wave sink and gamma_t gamma_5 bilinear",
            authors=["LatticeArena"],
        )

    def setup(
        self,
        gauge_field: Any,
        latt_size: tuple[int, int, int, int],
        lattice_spacing_fm: float | None,
    ) -> None:
        del gauge_field, latt_size, lattice_spacing_fm
        self._is_setup = True

    def design_source(
        self,
        gauge_field: Any,
        momentum_mode: tuple[int, int, int],
        t_source: int,
    ) -> SourceSpec:
        del gauge_field, momentum_mode, t_source
        if not self._is_setup:
            raise RuntimeError("setup() must be called before design_source().")
        return PointSourceSpec(position=(0, 0, 0))

    def design_sink(
        self,
        gauge_field: Any,
        momentum_mode: tuple[int, int, int],
        t_source: int,
    ) -> SinkSpec:
        del gauge_field, t_source
        if not self._is_setup:
            raise RuntimeError("setup() must be called before design_sink().")
        return PlaneWaveSinkSpec(momentum_mode=momentum_mode)

    def gamma_matrix(
        self,
        gauge_field: Any,
        momentum_mode: tuple[int, int, int],
        t_source: int,
    ) -> np.ndarray:
        del gauge_field, momentum_mode, t_source
        if not self._is_setup:
            raise RuntimeError("setup() must be called before gamma_matrix().")
        return gamma_t_gamma5_matrix()
