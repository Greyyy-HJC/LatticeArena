"""Gauge I/O helpers for Wilson loop datasets.

The task stores gauge fields on disk in shape:
    (Nd, Lx, Ly, Lz, Lt, Nc, Nc)

PyQUDA's lexicographic host gauge arrays use:
    (Nd, Lt, Lz, Ly, Lx, Nc, Nc)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PYQUDA_LEXICO_AXES = (0, 4, 3, 2, 1, 5, 6)
TASK_GAUGE_SHAPE_LEN = 7


def _validate_gauge_array(name: str, gauge: np.ndarray) -> None:
    if gauge.ndim != TASK_GAUGE_SHAPE_LEN:
        raise ValueError(
            f"{name} must have {TASK_GAUGE_SHAPE_LEN} dimensions, got shape {gauge.shape}."
        )
    if gauge.shape[0] != 4:
        raise ValueError(f"{name} must have Nd=4 in axis 0, got shape {gauge.shape}.")
    if gauge.shape[-2:] != (3, 3):
        raise ValueError(
            f"{name} must end with (Nc, Nc) = (3, 3), got shape {gauge.shape}."
        )


def pyquda_lexico_to_task_order(gauge: np.ndarray) -> np.ndarray:
    """Convert a PyQUDA lexicographic gauge array to task on-disk ordering."""

    _validate_gauge_array("gauge", gauge)
    return np.ascontiguousarray(np.transpose(gauge, PYQUDA_LEXICO_AXES))


def task_order_to_pyquda_lexico(gauge: np.ndarray) -> np.ndarray:
    """Convert a task-order gauge array back to PyQUDA lexicographic ordering."""

    _validate_gauge_array("gauge", gauge)
    return np.ascontiguousarray(np.transpose(gauge, PYQUDA_LEXICO_AXES))


def save_task_gauge_npy(path: str | Path, gauge_lexico: np.ndarray) -> Path:
    """Save a PyQUDA lexicographic gauge array in task on-disk ordering."""

    output_path = Path(path)
    if output_path.suffix != ".npy":
        output_path = output_path.with_suffix(".npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, pyquda_lexico_to_task_order(gauge_lexico), allow_pickle=False)
    return output_path


def load_task_gauge_npy(path: str | Path) -> np.ndarray:
    """Load a task-order gauge array from disk."""

    gauge = np.load(Path(path), allow_pickle=False)
    _validate_gauge_array("gauge", gauge)
    return np.ascontiguousarray(gauge)
