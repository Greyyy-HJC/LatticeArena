"""Generalized-eigenvalue reference for a matrix of correlators (todo C2).

Synthetic validation only so far: the real data contain one shared point source,
so a full operator matrix needs new inversions (NOT_RUN).
"""
import numpy as np
from scipy.linalg import eigh


def _hermitian(c):
    return np.allclose(c, c.conj().swapaxes(-1, -2), atol=1e-10)


def gevp_eigenvalues(c_t, c_t0):
    """Largest-first eigenvalues of c_t v = lambda c_t0 v, one row per block."""
    if c_t.ndim != 3 or c_t.shape != c_t0.shape:
        raise ValueError('expected (nblocks, N, N) matrices')
    if not (_hermitian(c_t) and _hermitian(c_t0)):
        raise ValueError('correlator matrices must be Hermitian')
    out = np.empty(c_t.shape[:2])
    for b in range(len(c_t)):
        try:
            values = eigh(c_t[b], c_t0[b], eigvals_only=True)
        except np.linalg.LinAlgError as exc:
            raise ValueError('c(t0) is not positive definite') from exc
        out[b] = values[::-1]
    return out


def gevp_energies(correlators, t0, times):
    """E_n(t) = log(lambda_n(t)/lambda_n(t+1)), block by block."""
    correlators = np.asarray(correlators)
    if correlators.ndim != 4:
        raise ValueError('expected (nblocks, N, N, T)')
    if not _hermitian(correlators[..., t0]):
        raise ValueError('c(t0) must be Hermitian')
    rows = {}
    for t in times:
        lam = gevp_eigenvalues(correlators[..., t], correlators[..., t0])
        lam_next = gevp_eigenvalues(correlators[..., t + 1], correlators[..., t0])
        if np.any(lam <= 0) or np.any(lam_next <= 0):
            raise ValueError('non-positive eigenvalues in the GEVP')
        rows[int(t)] = np.log(lam / lam_next)
    return rows


def synthetic(nblocks=200, n_ops=3, m0=0.8, m1=1.4, seed=0, noise=0.01, temporal=16):
    """Two-state correlator matrix with linearly independent overlaps."""
    rng = np.random.default_rng(seed)
    t = np.arange(temporal)
    a = np.array([1.0, 0.6, 0.25])[:n_ops]
    b = np.array([0.4, -0.25, 0.15])[:n_ops]
    mean = (np.outer(a, a)[..., None] * np.exp(-m0 * t)[None, None, :]
            + np.outer(b, b)[..., None] * np.exp(-m1 * t)[None, None, :]
            + 1e-3 * np.eye(n_ops)[..., None] * np.exp(-2.0 * t)[None, None, :])
    data = mean[None] * (1 + noise * rng.normal(size=(nblocks, 1, 1, temporal)))
    data = 0.5 * (data + data.conj().swapaxes(1, 2))
    return data


def summary(rows, state=0):
    values = np.array([row[:, state] for row in rows.values()])
    return {'energy': float(values.mean()),
            'se_boot': float(values.std(axis=1, ddof=1).mean() / np.sqrt(len(values))),
            'per_time': {int(t): float(rows[t][:, state].mean()) for t in rows}}
