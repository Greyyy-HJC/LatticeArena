"""Validation tests for Wilson-loop spatial operators."""

from __future__ import annotations

import numpy as np

from core.testing import (
    apply_gauge_transform,
    identity_gauge_field,
    random_gauge_field,
    random_gauge_transform,
)
from tasks.wilson_loop.submissions.plain import PlainWilsonLine


def test_plain_operator_shape_and_dtype() -> None:
    latt_size = (4, 4, 4, 8)
    gauge = identity_gauge_field(latt_size)
    op = PlainWilsonLine()

    op.setup(gauge, latt_size)
    spatial_line = op.compute(gauge, r=3, direction=1, t=2)

    assert spatial_line.shape == (4, 4, 4, 3, 3)
    assert spatial_line.dtype == np.complex128


def test_plain_operator_returns_identity_on_cold_config() -> None:
    latt_size = (4, 4, 4, 8)
    gauge = identity_gauge_field(latt_size)
    op = PlainWilsonLine()

    op.setup(gauge, latt_size)
    spatial_line = op.compute(gauge, r=3, direction=0, t=5)

    expected = np.broadcast_to(np.eye(3, dtype=np.complex128), spatial_line.shape)
    assert np.allclose(spatial_line, expected)


def test_plain_operator_is_gauge_equivariant() -> None:
    latt_size = (4, 4, 4, 8)
    gauge = random_gauge_field(latt_size, seed=1234)
    transform = random_gauge_transform(latt_size, seed=5678)
    transformed_gauge = apply_gauge_transform(gauge, transform)
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
    gauge = identity_gauge_field((4, 4, 4, 8))

    try:
        op.compute(gauge, r=1, direction=0, t=0)
    except RuntimeError:
        return
    raise AssertionError(
        "Expected RuntimeError when compute() is called before setup()."
    )
