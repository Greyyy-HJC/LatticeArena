"""Block-level calibration of the reference p-value (todo Phase B9/B10).

Simulates per-block vectors from the fitted model, rebuilds the block means,
re-estimates the covariance (shrinkage + floor) exactly as the analysis does,
and refits. Also reports the fixed-metric variant and a coverage check.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

from .statistics import ReferenceFit, blocks, fold, inverse_covariance, shape

RUN = Path('results/pion_projection/s16t16_v1')
OUT = Path('results/pion_projection/score_diagnostics_v2')


def fit_with_metric(model, mean, inv):
    def objective(mass):
        f = shape(mass, model.window, model.p['geometry'][3])
        design = np.kron(np.eye(len(model.basis)), f[:, None])
        amp = np.linalg.solve(design.T @ inv @ design, design.T @ inv @ mean)
        r = mean - design @ amp
        return float(r @ inv @ r)

    fit = minimize_scalar(objective, bounds=model.p['mass_bounds'], method='bounded',
                          options={'xatol': 1e-9})
    return float(fit.x), float(fit.fun)


def fitted_moments(model, mean):
    f = shape(model.fit(mean)[0], model.window, model.p['geometry'][3])
    design = np.kron(np.eye(len(model.basis)), f[:, None])
    amp = np.linalg.solve(design.T @ model.inv @ design, design.T @ model.inv @ mean)
    return design @ amp


def block_covariance(model):
    return np.atleast_2d(np.cov(model.samples, rowvar=False, ddof=1))


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def calibrate(model, mean, nsim=300, seed=0):
    cost_obs = model.fit(mean)[1]
    mu = fitted_moments(model, mean)
    cov = block_covariance(model)
    chol = np.linalg.cholesky(cov + 1e-13 * np.eye(len(mu)))
    rng = np.random.default_rng(seed)
    fixed, rebuilt, invalid, costs = 0, 0, 0, []
    for _ in range(nsim):
        sim = mu + rng.normal(size=(len(model.samples), len(mu))) @ chol.T
        sim_mean = sim.mean(axis=0)
        try:
            _, c_fixed = fit_with_metric(model, sim_mean, model.inv)
            inv_sim = inverse_covariance(sim, model.p['covariance_shrinkage'])
            _, c_rebuilt = fit_with_metric(model, sim_mean, inv_sim)
        except (np.linalg.LinAlgError, ValueError):
            invalid += 1
            continue
        fixed += c_fixed >= cost_obs
        rebuilt += c_rebuilt >= cost_obs
        costs.append(c_fixed)
    n = nsim - invalid
    dof = len(mean) - len(model.basis) - 1
    return {'cost': cost_obs, 'nsim': nsim, 'invalid': invalid,
            'p_fixed': (1 + fixed) / (1 + n), 'p_rebuilt': (1 + rebuilt) / (1 + n),
            'p_fixed_ci': wilson(fixed, n), 'p_rebuilt_ci': wilson(rebuilt, n),
            'p_asymptotic': float(chi2.sf(cost_obs, dof)),
            'null_median_cost': float(np.median(costs)) if costs else None,
            'null_q95_cost': float(np.quantile(costs, 0.95)) if costs else None,
            'dof': dof}


def coverage(model, mean, nsim=100, nsim_p=100, seed=0):
    mu = fitted_moments(model, mean)
    cov = block_covariance(model)
    chol = np.linalg.cholesky(cov + 1e-13 * np.eye(len(mu)))
    rng = np.random.default_rng(seed + 1)
    fails = 0
    for i in range(nsim):
        sim = mu + rng.normal(size=(len(model.samples), len(mu))) @ chol.T
        entry = calibrate(model, sim.mean(axis=0), nsim=nsim_p, seed=seed + 2 + i)
        fails += entry['p_rebuilt'] < 0.01
    return {'simulated_datasets': nsim, 'false_fail_at_1pct': fails / nsim,
            'false_fail_ci': wilson(fails, nsim)}


def main():
    p = json.loads((RUN / 'development_provenance.json').read_text())['protocol']
    raw = np.load(RUN / 'development.npz')['correlators']
    data = blocks(fold(raw), p['block_length'])
    rows = {}
    for window in p['reference_windows']:
        model = ReferenceFit(data, p, window)
        mean = model.samples.mean(0)
        entry = calibrate(model, mean, nsim=300, seed=20261118)
        rows[str(window)] = entry
        print(window, json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                  for k, v in entry.items() if k not in ('p_fixed_ci', 'p_rebuilt_ci')}), flush=True)
    nominal = p['reference_windows'][p['reference_nominal_window']]
    model = ReferenceFit(data, p, nominal)
    cov = coverage(model, model.samples.mean(0), nsim=100, nsim_p=100, seed=20261118)
    print('coverage', cov, flush=True)
    (OUT / 'calibration_blocks.json').write_text(json.dumps(
        {'protocol_version': p.get('version'), 'n_blocks': len(data), 'windows': rows,
         'coverage': cov,
         'caveat': 'Plug-in block bootstrap: the simulated covariance is the sample covariance of the same '
                   '12 blocks, so the null is circular and too narrow; with 15 dimensions and 12 blocks a '
                   'proper inverse-Wishart posterior does not exist. These p-values are NOT usable.'},
        indent=2) + '\n')


if __name__ == '__main__':
    main()
