"""Reusable operator checks with explicit counterexample fixtures (todo F0).

The fixtures are deliberately bad closures/kernels used to show that a check
fails when it should; production kernels are never modified.
"""
import numpy as np


def linearity_residual(op, f, g, alpha=0.37, beta=-0.61):
    """||op(alpha f + beta g) - (alpha op(f) + beta op(g))|| / (|alpha| ||op(f)|| + |beta| ||op(g)||)."""
    combined = op(alpha * f + beta * g)
    separate = alpha * op(f) + beta * op(g)
    scale = abs(alpha) * np.linalg.norm(op(f)) + abs(beta) * np.linalg.norm(op(g))
    return float(np.linalg.norm(combined - separate) / scale)


def hermiticity_residual(bilinear, f, g):
    """|bilinear(f, g) - bilinear(g, f)*| / max(|bilinear(f, g)|, eps)."""
    forward = bilinear(f, g)
    backward = np.conj(bilinear(g, f))
    return float(abs(forward - backward) / max(abs(forward), 1e-300))


def covariance_residual(op, transform, f, tol_basis=None):
    """||op(T f) - T op(f)|| / ||op(f)|| for a fixed isometry transform.

    This checks that the operator commutes with the field transformation for a
    fixed link background. It is NOT a full gauge-covariance test: that needs the
    links to be transformed together with the field (transform must act on both).
    """
    return float(np.linalg.norm(op(transform(f)) - transform(op(f))) / max(np.linalg.norm(op(f)), 1e-300))


# --- counterexample fixtures (known bad, used by the tests) ---

def bad_gram_schmidt_projector(u):
    """Right-multiplication Gram-Schmidt: not gauge covariant."""
    out = u.copy()
    for i in range(3):
        v = out[..., :, i]
        for j in range(i):
            v = v - np.sum(np.conj(out[..., :, j]) * v, axis=-1, keepdims=True) * out[..., :, j]
        out[..., :, i] = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return out


def bad_nonhermitian_kernel(f):
    """A shift that is not Hermitian (forward hop only)."""
    return np.roll(f, 1, axis=0)


def good_hermitian_kernel(f):
    """Symmetric hop: Hermitian on a periodic lattice."""
    return 0.5 * (np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0))
