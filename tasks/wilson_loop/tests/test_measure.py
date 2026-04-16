"""Measurement tests for Wilson-loop correlators."""

from __future__ import annotations

import numpy as np

from tasks.wilson_loop.submissions.plain import PlainWilsonLine
from tasks.wilson_loop.scripts.measure import measure_single_config


def _identity_gauge_field(latt_size: tuple[int, int, int, int]) -> np.ndarray:
    gauge = np.zeros((4, *latt_size, 3, 3), dtype=np.complex128)
    gauge[...] = np.eye(3, dtype=np.complex128)
    return gauge


def test_plain_measurement_on_cold_config_is_constant() -> None:
    gauge = _identity_gauge_field((4, 4, 4, 8))
    operator = PlainWilsonLine()

    correlator = measure_single_config(gauge, operator, r_values=[1, 2], t_values=[0, 1, 3])

    assert correlator.shape == (2, 3)
    assert np.allclose(correlator.imag, 0.0)
    assert np.allclose(correlator.real, 3.0)
