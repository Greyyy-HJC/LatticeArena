"""Spatial link kernels for the static-potential pilot.

Links have shape (3, t, z, y, x, 3, 3); spatial axes are 1, 2, 3.
`build_kernel` is the submission interface: same rule for every time slice and
configuration, gauge covariant, no temporal links.
"""
import numpy as np


def _lib(a):
    if type(a).__module__.startswith('cupy'):
        import cupy
        return cupy
    return np


def shift(a, k, axis):
    """a[x + k] along axis (positive k moves up)."""
    return _lib(a).roll(a, -k, axis=axis)


def dagger(a):
    return a.swapaxes(-1, -2).conj()


def _inv_sqrt_hermitian(a, tol=1e-10, max_iter=80, min_iter=8):
    """Numerical inverse square root of a positive-definite Hermitian 3x3 batch.

    Newton-Schulz with a per-matrix norm (a batch-wide scale made mixed-scale
    batches fail) and an explicit convergence check, so an ill-conditioned
    matrix is reported instead of silently projected.
    """
    xp = _lib(a)
    eye = xp.broadcast_to(xp.eye(3, dtype=a.dtype), a.shape)
    norms = xp.abs(a).sum(axis=(-1, -2))
    if not xp.all(xp.isfinite(norms)) or xp.any(norms <= 0):
        raise ValueError('project: non-positive or non-finite metric')
    scale = norms[..., None, None]
    y = eye.copy()
    z = a / scale
    for k in range(max_iter):
        t = 0.5 * (3 * eye - z @ y)
        y = y @ t
        z = t @ z
        if k + 1 >= min_iter:
            error = xp.abs(z @ y - eye).max(axis=(-1, -2))
            if xp.all(error <= tol):
                break
    error = xp.abs(z @ y - eye).max(axis=(-1, -2))
    if xp.any(error > tol):
        bad = xp.argwhere(error > tol).ravel()
        count = int(bad.size)
        indices = bad[:5].tolist() if hasattr(bad, 'tolist') else [int(x) for x in bad[:5]]
        raise ValueError(f'project: Newton-Schulz did not converge for {count} matrices '
                         f'(indices {indices}, max error {float(xp.max(error)):.3e})')
    return y / xp.sqrt(scale)


def project(m):
    """Bidirectional polar projection onto SU(3): Q = M (M^dag M)^(-1/2), det -> 1.

    Gram-Schmidt only preserves left unitary transformations; the polar factor
    satisfies P(g M h) = g P(M) h for independent SU(3) g, h, which is the
    covariance the APE staple needs.
    """
    xp = _lib(m)
    m = xp.asarray(m, dtype=xp.complex128)
    herm = dagger(m) @ m
    q = m @ _inv_sqrt_hermitian(herm)
    det = xp.linalg.det(q)
    q = q * xp.exp(-1j * xp.arctan2(det.imag, det.real) / 3.0)[..., None, None]
    defect = xp.max(xp.abs(dagger(q) @ q - xp.eye(3, dtype=q.dtype)))
    if not xp.isfinite(defect) or defect > 1e-6:
        raise ValueError(f'project: singular or ill-conditioned link (unitarity defect {float(defect):.3e})')
    return q


def ape(links, alpha, steps):
    """APE smearing: F = Proj[(1-a)U + a/4 * sum(four staples)].

    With three spatial directions each link has four staples (two planes times
    two orientations), so the normalisation is 1/4, not 1/6. This changes the
    numeric meaning of ``alpha`` relative to the old pilot (v1); the old
    weight corresponds to alpha_new = 2*alpha/(3-alpha).
    """
    xp = _lib(links)
    u = links.copy()
    for _ in range(steps):
        new = xp.empty_like(u)
        for mu in range(3):
            staple = 0
            for nu in range(3):
                if nu == mu:
                    continue
                staple = staple + u[nu] @ shift(u[mu], 1, 3 - nu) @ dagger(shift(u[nu], 1, 3 - mu))
                staple = staple + dagger(shift(u[nu], -1, 3 - nu)) @ shift(u[mu], -1, 3 - nu) @ shift(shift(u[nu], 1, 3 - mu), -1, 3 - nu)
            new[mu] = project((1 - alpha) * u[mu] + alpha / 4 * staple)
        u = new
    return u


def build_kernel(spatial_links, parameters):
    return ape(spatial_links, parameters['alpha'], parameters['steps'])
