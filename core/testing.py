"""Shared test utilities for lattice QCD operator validation.

Provides helpers for generating gauge fields, SU(3) matrices, and gauge
transformations used across task validation test suites.
"""

from __future__ import annotations

import numpy as np


def identity_gauge_field(latt_size: tuple[int, int, int, int]) -> np.ndarray:
    """Create a unit (cold) gauge field in natural ordering.

    Returns:
        Array of shape (Nd, Lx, Ly, Lz, Lt, Nc, Nc), complex128.
    """
    gauge = np.zeros((4, *latt_size, 3, 3), dtype=np.complex128)
    gauge[...] = np.eye(3, dtype=np.complex128)
    return gauge


def random_su3_matrices(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Generate random SU(3) matrices via QR decomposition.

    Args:
        shape: Leading dimensions of the output (e.g. ``(4, Lx, Ly, Lz, Lt)``).
        seed: RNG seed for reproducibility.

    Returns:
        Array of shape ``(*shape, 3, 3)``, complex128.
    """
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


def random_gauge_field(latt_size: tuple[int, int, int, int], seed: int) -> np.ndarray:
    """Generate a random SU(3) gauge field in natural ordering.

    Returns:
        Array of shape (Nd, Lx, Ly, Lz, Lt, Nc, Nc), complex128.
    """
    return random_su3_matrices((4, *latt_size), seed)


def random_gauge_transform(
    latt_size: tuple[int, int, int, int], seed: int
) -> np.ndarray:
    """Generate a random SU(3) gauge transformation.

    Returns:
        Array of shape (Lx, Ly, Lz, Lt, Nc, Nc), complex128.
    """
    return random_su3_matrices(latt_size, seed)


def apply_gauge_transform(
    gauge_field: np.ndarray, gauge_transform: np.ndarray
) -> np.ndarray:
    """Apply a gauge transformation to a gauge field.

    Computes G(x) U_mu(x) G^dag(x + mu) for each link.

    Args:
        gauge_field: Shape (Nd, Lx, Ly, Lz, Lt, Nc, Nc).
        gauge_transform: Shape (Lx, Ly, Lz, Lt, Nc, Nc).

    Returns:
        Transformed gauge field with same shape.
    """
    transformed = np.empty_like(gauge_field)
    for mu in range(4):
        shifted = np.roll(gauge_transform, shift=-1, axis=mu)
        transformed[mu] = (
            gauge_transform @ gauge_field[mu] @ np.swapaxes(shifted.conj(), -1, -2)
        )
    return transformed
