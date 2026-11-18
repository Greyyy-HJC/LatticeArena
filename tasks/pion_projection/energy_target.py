"""Energy-error target with an independent anchor (todo C3).

Synthetic validation of the paired loss difference

    Delta = [(E_b - E_anchor)^2 - (E_c - E_anchor)^2] / tau_E^2,

and of the one-sided lower-bound certificate. Real-data use needs an
independent anchor (NOT_RUN).
"""
import numpy as np
from scipy.stats import norm


def energy_delta(baseline, candidate, anchor, tau_e=0.01):
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    if not (baseline.shape == candidate.shape == anchor.shape):
        raise ValueError('baseline, candidate and anchor draws must have the same shape')
    if tau_e <= 0:
        raise ValueError('tau_e must be positive')
    return ((baseline - anchor) ** 2 - (candidate - anchor) ** 2) / tau_e ** 2


def lower_bound(delta, alpha=0.05):
    """One-sided normal lower bound for the mean paired energy improvement."""
    delta = np.asarray(delta, dtype=float)
    if len(delta) < 10:
        raise ValueError('need at least ten draws for a lower bound')
    if not np.all(np.isfinite(delta)):
        raise ValueError('non-finite energy deltas')
    se = delta.std(ddof=1) / np.sqrt(len(delta))
    return float(delta.mean() - norm.ppf(1 - alpha) * se)


def experiment(e0, anchor_sigma, bias, sigma, n_draws, rng, tau_e):
    anchor = e0 + rng.normal(0, anchor_sigma, n_draws)
    baseline = e0 + bias[0] + rng.normal(0, sigma[0], n_draws)
    candidate = e0 + bias[1] + rng.normal(0, sigma[1], n_draws)
    return energy_delta(baseline, candidate, anchor, tau_e)


def coverage(n_rep=200, n_draws=200, alpha=0.05, seed=0, e0=0.8, **kwargs):
    rng = np.random.default_rng(seed)
    hits = 0
    medians = []
    for _ in range(n_rep):
        delta = experiment(e0=e0, rng=rng, n_draws=n_draws, **kwargs)
        hits += lower_bound(delta, alpha) > 0
        medians.append(float(np.median(delta)))
    return {'replicates': n_rep, 'draws': n_draws, 'alpha': alpha,
            'certified_fraction': hits / n_rep,
            'median_delta': float(np.mean(medians))}
