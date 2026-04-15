"""Baseline fit configuration for the gsfit_2pt task."""

from __future__ import annotations

from tasks.gsfit_2pt.interface import (
    FitSubmissionMeta,
    GroundStateFitConfig,
    Pion2PtGroundStateFit,
)


class PlainGroundStateFit(Pion2PtGroundStateFit):
    """Conservative two-state Bayesian fit baseline."""

    @property
    def meta(self) -> FitSubmissionMeta:
        return FitSubmissionMeta(
            name="plain",
            description="Conservative 2-state pion ground-state fit configuration",
            authors=["LatticeArena"],
        )

    @property
    def config(self) -> GroundStateFitConfig:
        return GroundStateFitConfig(
            t_min=4,
            t_max=16,
            n_states=2,
            e0_prior=(0.34, 0.18),
            delta_e_priors=[(0.45, 0.20)],
            amplitude_priors=[(0.90, 0.50), (0.25, 0.20)],
        )
