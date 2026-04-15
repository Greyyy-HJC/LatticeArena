"""Straight spatial Wilson-line baseline for the Wilson-loop task.

This implements the on-axis path in Eq. (3) of arXiv:2602.02436:

    S(x, x + r, t) = prod_{k=0}^{r-1} U_dir(x + k e_dir, t)

It is the simplest gauge-covariant baseline and provides the reference
operator against which more elaborate path superpositions can be compared.
"""

from __future__ import annotations

import numpy as np

from tasks.wilson_loop.interface import OperatorMeta, SpatialOperator


class PlainWilsonLine(SpatialOperator):
    """Straight product of links along one spatial direction."""

    def __init__(self) -> None:
        self._latt_size: tuple[int, int, int, int] | None = None

    @property
    def meta(self) -> OperatorMeta:
        return OperatorMeta(
            name="plain",
            description="Straight spatial Wilson line built from ordered link products",
            authors=["LatticeArena"],
        )

    def setup(self, gauge_field: np.ndarray, latt_size: tuple[int, int, int, int]) -> None:
        self._validate_gauge_field(gauge_field, latt_size)
        self._latt_size = latt_size

    def compute(
        self,
        gauge_field: np.ndarray,
        r: int,
        direction: int,
        t: int,
    ) -> np.ndarray:
        if self._latt_size is None:
            raise RuntimeError("setup() must be called before compute().")

        self._validate_gauge_field(gauge_field, self._latt_size)
        lx, ly, lz, lt = self._latt_size

        if not (1 <= r):
            raise ValueError(f"r must be >= 1, got {r}.")
        if direction not in (0, 1, 2):
            raise ValueError(f"direction must be one of (0, 1, 2), got {direction}.")
        if not (0 <= t < lt):
            raise ValueError(f"t must satisfy 0 <= t < {lt}, got {t}.")

        links_t = gauge_field[direction, :, :, :, t]
        out = np.broadcast_to(np.eye(3, dtype=np.complex128), (lx, ly, lz, 3, 3)).copy()
        for step in range(r):
            out = out @ np.roll(links_t, shift=-step, axis=direction)
        return out

    @staticmethod
    def _validate_gauge_field(gauge_field: np.ndarray, latt_size: tuple[int, int, int, int]) -> None:
        expected_shape = (4, *latt_size, 3, 3)
        if gauge_field.shape != expected_shape:
            raise ValueError(f"Expected gauge_field shape {expected_shape}, got {gauge_field.shape}.")
        if gauge_field.dtype != np.complex128:
            raise ValueError(f"Expected gauge_field dtype complex128, got {gauge_field.dtype}.")
