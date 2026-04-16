"""Tests for Wilson-loop dataset gauge I/O helpers."""

from __future__ import annotations

import numpy as np

from tasks.wilson_loop.scripts.gauge_io import (
    load_task_gauge_npy,
    pyquda_lexico_to_task_order,
    save_task_gauge_npy,
    task_order_to_pyquda_lexico,
)


def test_pyquda_lexico_to_task_order_shape() -> None:
    gauge = (
        np.arange(4 * 8 * 6 * 5 * 4 * 3 * 3, dtype=np.float64)
        .reshape(4, 8, 6, 5, 4, 3, 3)
        .astype(np.complex128)
    )

    converted = pyquda_lexico_to_task_order(gauge)

    assert converted.shape == (4, 4, 5, 6, 8, 3, 3)
    assert converted[2, 1, 3, 4, 5, 0, 1] == gauge[2, 5, 4, 3, 1, 0, 1]


def test_task_gauge_round_trip(tmp_path) -> None:
    gauge = (
        np.arange(4 * 7 * 6 * 5 * 4 * 3 * 3, dtype=np.float64)
        .reshape(4, 7, 6, 5, 4, 3, 3)
        .astype(np.complex128)
        + 1j
    )

    saved_path = save_task_gauge_npy(tmp_path / "cfg_0001.npy", gauge)
    loaded_task_order = load_task_gauge_npy(saved_path)
    recovered = task_order_to_pyquda_lexico(loaded_task_order)

    assert loaded_task_order.shape == (4, 4, 5, 6, 7, 3, 3)
    assert np.array_equal(recovered, gauge)
