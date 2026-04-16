"""Validation helpers for pion 2pt submissions."""

from __future__ import annotations

import numpy as np

from core.testing import identity_gauge_field
from tasks.pion_2pt.interface import PionInterpolatingOperator


def validate_submission(
    submission: PionInterpolatingOperator,
    *,
    latt_size: tuple[int, int, int, int] = (8, 8, 8, 32),
    lattice_spacing_fm: float = 0.09,
    momentum_gev: tuple[float, float, float] = (1.0, 0.0, 0.0),
    t_source: int = 0,
    atol: float = 1e-12,
) -> bool:
    """Return ``True`` when a pion 2pt submission satisfies the basic contract."""

    gauge = identity_gauge_field(latt_size)
    try:
        submission.setup(
            gauge, latt_size=latt_size, lattice_spacing_fm=lattice_spacing_fm
        )
        components = submission.build(
            gauge, momentum_gev=momentum_gev, t_source=t_source
        )
    except Exception:
        return False

    if components.source_profile.shape != latt_size[:3]:
        return False
    if components.sink_profile.shape != latt_size[:3]:
        return False
    if components.gamma_matrix.shape != (4, 4):
        return False
    if not np.iscomplexobj(components.source_profile):
        return False
    if not np.iscomplexobj(components.sink_profile):
        return False

    src_norm = np.linalg.norm(components.source_profile)
    sink_norm = np.linalg.norm(components.sink_profile)
    if not np.isclose(src_norm, 1.0, atol=atol):
        return False
    if not np.isclose(sink_norm, 1.0, atol=atol):
        return False
    return True
