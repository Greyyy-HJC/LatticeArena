"""Validation tests for Wilson-loop spatial operators."""

from __future__ import annotations

import numpy as np

from tasks.wilson_loop.operators.plain import PlainWilsonLine


def _identity_gauge_field(latt_size: tuple[int, int, int, int]) -> np.ndarray:
    gauge = np.zeros((4, *latt_size, 3, 3), dtype=np.complex128)
    gauge[...] = np.eye(3, dtype=np.complex128)
    return gauge


def _random_su3_matrices(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_mats = int(np.prod(shape))
    mats = np.empty((n_mats, 3, 3), dtype=np.complex128)

    for i in range(n_mats):
        z = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        q, r = np.linalg.qr(z)
        phases = np.diag(r) / np.abs(np.diag(r))
        q = q @ np.diag(np.conjugate(phases))
        det_q = np.linalg.det(q)
        q *= det_q ** (-1 / 3)
        mats[i] = q

    return mats.reshape(*shape, 3, 3)


def _random_gauge_field(latt_size: tuple[int, int, int, int], seed: int) -> np.ndarray:
    return _random_su3_matrices((4, *latt_size), seed)


def _random_gauge_transform(latt_size: tuple[int, int, int, int], seed: int) -> np.ndarray:
    return _random_su3_matrices(latt_size, seed)


def _apply_gauge_transform(gauge_field: np.ndarray, gauge_transform: np.ndarray) -> np.ndarray:
    transformed = np.empty_like(gauge_field)
    for mu in range(4):
        shifted = np.roll(gauge_transform, shift=-1, axis=mu)
        transformed[mu] = gauge_transform @ gauge_field[mu] @ np.swapaxes(shifted.conj(), -1, -2)
    return transformed


def test_plain_operator_shape_and_dtype() -> None:
    latt_size = (4, 4, 4, 8)
    gauge = _identity_gauge_field(latt_size)
    op = PlainWilsonLine()

    op.setup(gauge, latt_size)
    spatial_line = op.compute(gauge, r=3, direction=1, t=2)

    assert spatial_line.shape == (4, 4, 4, 3, 3)
    assert spatial_line.dtype == np.complex128


def test_plain_operator_returns_identity_on_cold_config() -> None:
    latt_size = (4, 4, 4, 8)
    gauge = _identity_gauge_field(latt_size)
    op = PlainWilsonLine()

    op.setup(gauge, latt_size)
    spatial_line = op.compute(gauge, r=3, direction=0, t=5)

    expected = np.broadcast_to(np.eye(3, dtype=np.complex128), spatial_line.shape)
    assert np.allclose(spatial_line, expected)


def test_plain_operator_is_gauge_equivariant() -> None:
    latt_size = (4, 4, 4, 8)
    gauge = _random_gauge_field(latt_size, seed=1234)
    transform = _random_gauge_transform(latt_size, seed=5678)
    transformed_gauge = _apply_gauge_transform(gauge, transform)
    op = PlainWilsonLine()

    op.setup(gauge, latt_size)
    spatial_line = op.compute(gauge, r=2, direction=2, t=3)

    op.setup(transformed_gauge, latt_size)
    transformed_line = op.compute(transformed_gauge, r=2, direction=2, t=3)

    gt_start = transform[:, :, :, 3]
    gt_end = np.roll(gt_start, shift=-2, axis=2)
    expected = gt_start @ spatial_line @ np.swapaxes(gt_end.conj(), -1, -2)

    assert np.allclose(transformed_line, expected, atol=1e-10)


def test_plain_operator_requires_setup() -> None:
    op = PlainWilsonLine()
    gauge = _identity_gauge_field((4, 4, 4, 8))

    try:
        op.compute(gauge, r=1, direction=0, t=0)
    except RuntimeError:
        return
    raise AssertionError("Expected RuntimeError when compute() is called before setup().")
