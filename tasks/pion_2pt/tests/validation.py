"""Validation helpers for pion 2pt submissions."""

from __future__ import annotations

import numpy as np

from core.testing import identity_gauge_field
from tasks.pion_2pt.interface import (
    PionInterpolatingOperator,
    PlaneWaveSinkSpec,
    PointSourceSpec,
    ProfileSinkSpec,
    ProfileSourceSpec,
)


def _valid_point_position(
    position: tuple[int, int, int],
    latt_size: tuple[int, int, int, int],
) -> bool:
    if len(position) != 3:
        return False
    return all(0 <= position[idx] < latt_size[idx] for idx in range(3))


def validate_submission(
    submission: PionInterpolatingOperator,
    *,
    latt_size: tuple[int, int, int, int] = (8, 8, 8, 32),
    lattice_spacing_fm: float = 0.09,
    momentum_mode: tuple[int, int, int] = (1, 0, 0),
    t_source: int = 0,
    atol: float = 1e-12,
) -> bool:
    """Return ``True`` when a pion 2pt submission satisfies the basic contract."""

    gauge = identity_gauge_field(latt_size)
    try:
        submission.setup(
            gauge, latt_size=latt_size, lattice_spacing_fm=lattice_spacing_fm
        )
        source_spec = submission.design_source(
            gauge, momentum_mode=momentum_mode, t_source=t_source
        )
        sink_spec = submission.design_sink(
            gauge, momentum_mode=momentum_mode, t_source=t_source
        )
        gamma_matrix = submission.gamma_matrix(
            gauge, momentum_mode=momentum_mode, t_source=t_source
        )
    except Exception:
        return False

    if isinstance(source_spec, PointSourceSpec):
        if not _valid_point_position(source_spec.position, latt_size):
            return False
    elif isinstance(source_spec, ProfileSourceSpec):
        if source_spec.profile.shape != latt_size[:3]:
            return False
        if not np.iscomplexobj(source_spec.profile):
            return False
        if not np.isclose(np.linalg.norm(source_spec.profile), 1.0, atol=atol):
            return False
    else:
        return False

    if isinstance(sink_spec, PlaneWaveSinkSpec):
        if len(sink_spec.momentum_mode) != 3:
            return False
        if not all(isinstance(component, int) for component in sink_spec.momentum_mode):
            return False
    elif isinstance(sink_spec, ProfileSinkSpec):
        if sink_spec.profile.shape != latt_size[:3]:
            return False
        if not np.iscomplexobj(sink_spec.profile):
            return False
        if not np.isclose(np.linalg.norm(sink_spec.profile), 1.0, atol=atol):
            return False
    else:
        return False

    if gamma_matrix.shape != (4, 4):
        return False
    if not np.iscomplexobj(gamma_matrix):
        return False
    return True
