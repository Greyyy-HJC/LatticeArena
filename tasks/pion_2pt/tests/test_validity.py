"""Validation tests for pion_2pt operators."""

from __future__ import annotations

import numpy as np

from tasks.pion_2pt.operators.plain import PlainBoostedPion


def _mock_gauge_field(latt_size: tuple[int, int, int, int]) -> np.ndarray:
    lx, ly, lz, lt = latt_size
    gauge = np.zeros((4, lx, ly, lz, lt, 3, 3), dtype=np.complex128)
    eye = np.eye(3, dtype=np.complex128)
    gauge[...] = eye
    return gauge


def test_plain_operator_shapes() -> None:
    latt_size = (8, 8, 8, 32)
    op = PlainBoostedPion(sigma=2.5)
    gauge = _mock_gauge_field(latt_size)

    op.setup(gauge, latt_size=latt_size, lattice_spacing_fm=0.09)
    components = op.build(gauge, momentum_gev=(1.0, 0.0, 0.0), t_source=0)

    assert components.source_profile.shape == (8, 8, 8)
    assert components.sink_profile.shape == (8, 8, 8)
    assert components.gamma_matrix.shape == (4, 4)
    assert np.iscomplexobj(components.source_profile)


def test_plain_operator_normalization() -> None:
    latt_size = (8, 8, 8, 32)
    op = PlainBoostedPion(sigma=2.0)
    gauge = _mock_gauge_field(latt_size)

    op.setup(gauge, latt_size=latt_size, lattice_spacing_fm=0.09)
    components = op.build(gauge, momentum_gev=(1.0, 0.0, 0.0), t_source=0)

    src_norm = np.linalg.norm(components.source_profile)
    sink_norm = np.linalg.norm(components.sink_profile)
    assert np.isclose(src_norm, 1.0, atol=1e-12)
    assert np.isclose(sink_norm, 1.0, atol=1e-12)


def test_plain_operator_requires_setup() -> None:
    op = PlainBoostedPion()
    gauge = _mock_gauge_field((4, 4, 4, 8))

    try:
        op.build(gauge, momentum_gev=(1.0, 0.0, 0.0), t_source=0)
    except RuntimeError:
        return
    raise AssertionError("Expected RuntimeError when build() is called before setup().")
