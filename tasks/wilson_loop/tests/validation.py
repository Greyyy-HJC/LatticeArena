"""Validation helpers for Wilson-loop submissions."""

from __future__ import annotations

import numpy as np

from tasks.wilson_loop.interface import SpatialOperator


def identity_gauge_field(latt_size: tuple[int, int, int, int]) -> np.ndarray:
    """Return a cold/unit gauge field in task ordering."""

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
        q *= np.linalg.det(q) ** (-1 / 3)
        mats[i] = q

    return mats.reshape(*shape, 3, 3)


def random_gauge_field(latt_size: tuple[int, int, int, int], seed: int) -> np.ndarray:
    """Return a random SU(3) gauge field in task ordering."""

    return _random_su3_matrices((4, *latt_size), seed)


def random_gauge_transform(
    latt_size: tuple[int, int, int, int], seed: int
) -> np.ndarray:
    """Return a random local SU(3) gauge transformation field."""

    return _random_su3_matrices(latt_size, seed)


def apply_gauge_transform(
    gauge_field: np.ndarray, gauge_transform: np.ndarray
) -> np.ndarray:
    """Apply a local gauge transformation to a task-order gauge field."""

    transformed = np.empty_like(gauge_field)
    for mu in range(4):
        shifted = np.roll(gauge_transform, shift=-1, axis=mu)
        transformed[mu] = (
            gauge_transform @ gauge_field[mu] @ np.swapaxes(shifted.conj(), -1, -2)
        )
    return transformed


def validate_submission(
    submission: SpatialOperator,
    *,
    latt_size: tuple[int, int, int, int] = (4, 4, 4, 8),
    r: int = 2,
    direction: int = 2,
    t: int = 3,
    atol: float = 1e-10,
) -> list[str]:
    """Return a list of validation errors for one submission instance."""

    errors: list[str] = []
    cold = identity_gauge_field(latt_size)
    expected_identity = np.broadcast_to(
        np.eye(3, dtype=np.complex128), (latt_size[0], latt_size[1], latt_size[2], 3, 3)
    )

    try:
        submission.setup(cold, latt_size)
        cold_output = submission.compute(cold, r=r, direction=direction, t=t)
    except Exception as exc:
        return [f"submission raised during cold-config evaluation: {exc}"]

    if cold_output.shape != (latt_size[0], latt_size[1], latt_size[2], 3, 3):
        errors.append(f"output shape mismatch: got {cold_output.shape}")
    if cold_output.dtype != np.complex128:
        errors.append(f"output dtype mismatch: got {cold_output.dtype}")
    if not np.allclose(cold_output, expected_identity, atol=atol):
        errors.append("cold-config identity check failed")

    gauge = random_gauge_field(latt_size, seed=1234)
    transform = random_gauge_transform(latt_size, seed=5678)
    transformed_gauge = apply_gauge_transform(gauge, transform)

    try:
        submission.setup(gauge, latt_size)
        output = submission.compute(gauge, r=r, direction=direction, t=t)
        submission.setup(transformed_gauge, latt_size)
        transformed_output = submission.compute(
            transformed_gauge, r=r, direction=direction, t=t
        )
    except Exception as exc:
        errors.append(f"submission raised during gauge-equivariance check: {exc}")
        return errors

    gt_start = transform[:, :, :, t]
    gt_end = np.roll(gt_start, shift=-r, axis=direction)
    expected = gt_start @ output @ np.swapaxes(gt_end.conj(), -1, -2)
    if not np.allclose(transformed_output, expected, atol=atol):
        errors.append("gauge-equivariance check failed")

    return errors
