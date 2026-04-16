"""Shared Dirac-matrix helpers for ``pion_2pt`` submissions and workflows."""

from __future__ import annotations

import numpy as np
from pyquda_utils import gamma


def _to_numpy(matrix: object) -> np.ndarray:
    """Convert a PyQUDA gamma matrix to a host NumPy array."""

    if hasattr(matrix, "get"):
        matrix = matrix.get()
    return np.asarray(matrix, dtype=np.complex128)


def gamma5_matrix() -> np.ndarray:
    """Return ``gamma_5`` from ``pyquda_utils.gamma``."""

    return _to_numpy(gamma.gamma(15))


def gamma_t_matrix() -> np.ndarray:
    """Return Euclidean ``gamma_t = gamma_4`` from ``pyquda_utils.gamma``."""

    return _to_numpy(gamma.gamma(8))


def gamma_t_gamma5_matrix() -> np.ndarray:
    """Return the temporal-axial bilinear ``gamma_t gamma_5``."""

    return gamma_t_matrix() @ gamma5_matrix()
