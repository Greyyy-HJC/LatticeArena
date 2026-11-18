"""Statistically explicit GEVP estimator (todo P0.2/P0.3).

Block-bootstrap replicas -> per-replica average matrix -> GEVP -> per-replica
generalized eigenvalues -> effective energies. Exact duplicate operators are
removed first (the power-0 operator appears in every family). No per-config
energies are averaged and no fake standard error is reported.

The forward estimator assumes lam(t) = exp(-E (t-t0)); the cosh estimator keeps
the finite-T wrap, cosh(E(t-T/2))/cosh(E(t0-T/2)) = lam(t).
"""
import numpy as np


def dedupe(matrix, atol=1e-12):
    """Drop operators whose matrix blocks are exact duplicates."""
    d = matrix.shape[1]
    keep = []
    for i in range(d):
        if not any(np.allclose(matrix[:, i, :, :], matrix[:, j, :, :], rtol=0, atol=atol)
                   for j in keep):
            keep.append(i)
    return matrix[:, keep][:, :, keep], keep


def replica_matrices(matrix, index):
    """matrix (ncfg,d,d,T) and block index (B,ncfg) -> replica averages (B,d,d,T)."""
    return matrix[index].mean(axis=1)


def _eigvals(a, b, rtol=1e-10, rank=None):
    """Generalized eigenvalues of (a, b), restricted to the resolved subspace of b."""
    values, vectors = np.linalg.eigh(0.5 * (b + b.conj().T))
    keep = values > rtol * max(values[-1], 1e-300)
    if rank is not None:
        keep = np.zeros_like(keep)
        keep[np.argsort(values)[-int(rank):]] = True
    if keep.sum() < 2:
        return np.array([])
    basis = vectors[:, keep] / np.sqrt(values[keep])
    reduced = basis.conj().T @ a @ basis
    return np.sort(np.linalg.eigvalsh(0.5 * (reduced + reduced.conj().T)))[::-1]


def effective_energy(lam, t, t0, T, model='forward'):
    """Effective energy from one generalized eigenvalue lam."""
    if model == 'forward':
        return float(-np.log(lam) / (t - t0)) if lam > 0 else np.nan
    from scipy.optimize import brentq
    def residual(e):
        return np.cosh(e * (t - T / 2)) / np.cosh(e * (t0 - T / 2)) - lam
    try:
        return float(brentq(residual, 1e-8, 5.0))
    except ValueError:
        return np.nan


def bootstrap_gevp(matrix, index, t0=2, times=(3, 4, 5, 6), model='forward', rank=None):
    """Return energies (ntimes, nmodes, B) and a list of failures."""
    replicas = replica_matrices(matrix, index)
    B, d, _, T = replicas.shape
    energies = np.full((len(times), d, B), np.nan)
    failures = []
    for b in range(B):
        for k, t in enumerate(times):
            lam = _eigvals(replicas[b, :, :, t], replicas[b, :, :, t0], rank=rank)
            if len(lam) == 0 or np.any(lam <= 0) or not np.all(np.isfinite(lam)):
                failures.append({'replica': b, 'time': int(t), 'reason': 'nonpositive eigenvalue'})
                continue
            for m in range(len(lam)):
                value = effective_energy(lam[m], t, t0, T, model)
                energies[k, m, b] = value
                if not np.isfinite(value):
                    failures.append({'replica': b, 'time': int(t), 'mode': m,
                                     'reason': 'energy undefined (cosh root or rank)'})
    return energies, failures


def summarise(energies, times, mode=0, require_all_times=True):
    """Plateau average with a fixed time set and explicit failure counts.

    A replica is valid only if every required plateau time is finite (the
    pre-declared whole-replica policy); partial times are never averaged.
    """
    values = energies[:, mode, :]
    finite = np.isfinite(values)
    local_failures = int((~finite).sum())
    ok = finite.all(axis=0) if require_all_times else finite.any(axis=0)
    per_time = np.array([values[i][finite[i]].mean() if finite[i].any() else np.nan
                         for i in range(values.shape[0])])
    se_time = np.array([values[i][finite[i]].std(ddof=1) if finite[i].sum() > 1 else np.nan
                        for i in range(values.shape[0])])
    per_replica = np.array([values[:, b][finite[:, b]].mean() if ok[b] else np.nan
                            for b in range(values.shape[1])])
    valid = per_replica[ok]
    estimate = float(valid.mean()) if valid.size else float('nan')
    se = float(valid.std(ddof=1)) if valid.size > 1 else float('nan')
    return {'energy': estimate, 'se': se,
            'per_time': {int(t): float(v) for t, v in zip(times, per_time)},
            'per_time_se': {int(t): float(v) for t, v in zip(times, se_time)},
            'replicas': int(ok.sum()), 'failures': int((~ok).sum()),
            'local_failures': local_failures,
            'valid_times_per_replica': finite.sum(axis=0).tolist()}


def synthetic_correlators(rng, ncfg=60, d=3, T=16, energies=(0.80, 1.30), noise=0.02,
                          wrap=True):
    """Known two-state spectrum, diagonalised amplitudes, multiplicative noise."""
    amplitudes = np.array([[1.0, 0.20], [0.60, 0.50], [0.30, 0.80]])[:d]
    t = np.arange(T)
    factors = np.exp(-np.asarray(energies)[:, None] * t)
    if wrap:
        factors = factors + np.exp(-np.asarray(energies)[:, None] * (T - t))
    signal = np.einsum('in,jn,nt->ijt', amplitudes, amplitudes, factors)
    return signal[None] * (1 + noise * rng.normal(size=(ncfg, d, d, T)))


def coverage_study(rng, nsim=100, ncfg=60, t0=2, times=(4, 5, 6), noise=0.02,
                   wrap=False, model='forward', truth=0.80):
    """Independent-dataset coverage of the E0 interval."""
    from scipy.stats import norm
    rows = []
    for _ in range(nsim):
        data = synthetic_correlators(rng, ncfg=ncfg, noise=noise, wrap=wrap)
        index = rng.integers(0, ncfg, (200, ncfg))
        energies, _ = bootstrap_gevp(data, index, t0=t0, times=times, model=model, rank=2)
        summary = summarise(energies, times, mode=0)
        rows.append({'energy': summary['energy'], 'se': summary['se']})
    est = np.array([r['energy'] for r in rows])
    se = np.array([r['se'] for r in rows])
    z = (est - truth) / se
    return {'estimates': est.tolist(), 'errors': se.tolist(),
            'mean_bias': float(est.mean() - truth),
            'bias_over_se': float((est.mean() - truth) / (est.std(ddof=1) / np.sqrt(nsim))),
            'coverage_95': float(np.mean(np.abs(z) <= norm.ppf(0.975))),
            'z': z.tolist()}
