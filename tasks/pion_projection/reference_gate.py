"""Reference-gate calibration with the formal dof and a per-dataset null (C1).

Two experiments:
  asymptotic: p = chi2.sf(cost, dof) with dof = dims - nchan - 1 (3 amps + mass);
  null:       parametric bootstrap from the fitted single-cosh model, p =
              (1 + #{cost_null >= cost_obs}) / (N_null + 1).
Truth models: single (correct) and two-state (misspecified). Configuration-level
offsets and time-AR noise are generated separately.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from .statistics import ReferenceFit, blocks, shape

OUT = Path('results/pion_projection/reference_gate_calibration_v1')
PROTOCOL = json.loads(Path('results/pion_projection/s16t16_v1/development_provenance.json').read_text())['protocol']
BASE = np.array([1.0, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1])
WINDOW = PROTOCOL['reference_windows'][PROTOCOL['reference_nominal_window']]
BASIS = PROTOCOL['reference_basis']


def generate(rng, kind='single', ncfg=120, noise=0.02, cfg_sigma=0.01, time_rho=0.5):
    T = PROTOCOL['geometry'][3]
    t = np.arange(T)
    energies = (1.077,) if kind == 'single' else (1.077, 1.55)
    signal = np.zeros((len(BASE), T))
    for i, energy in enumerate(energies):
        amps = BASE * (1.0 if i == 0 else 0.3)
        signal += amps[:, None] * (np.exp(-energy * t) + np.exp(-energy * (T - t)))
    cfg = rng.normal(size=(ncfg, len(BASE), 1)) * cfg_sigma
    eps = rng.normal(size=(ncfg, len(BASE), T))
    for k in range(1, T):
        eps[:, :, k] = time_rho * eps[:, :, k - 1] + np.sqrt(1 - time_rho ** 2) * rng.normal(size=(ncfg, len(BASE)))
    return signal[None] * (1 + cfg + noise * eps)


def fit_gate(data, p=PROTOCOL):
    fit = ReferenceFit(blocks(data, p['block_length']), p, WINDOW)
    mean = fit.samples.mean(axis=0)
    mass, cost = fit.fit(mean)
    dof = fit.samples.shape[1] - len(BASIS) - 1
    return fit, mass, cost, dof


def fitted_model(fit, mass):
    """Per-channel fitted correlator in raw units over the requested window."""
    f = shape(mass, fit.window, fit.p['geometry'][3])
    design = np.kron(np.eye(len(BASIS)), f[:, None])
    amp = np.linalg.solve(design.T @ fit.inv @ design, design.T @ fit.inv @ fit.samples.mean(axis=0))
    return fit.scales[:, None] * (amp[:, None] * f[None, :]), design, amp


def null_pvalue(fit, mass, cost_observed, mean_full, rng, n_null=50, noise=0.02, cfg_sigma=0.01, time_rho=0.5):
    fitted, _, _ = fitted_model(fit, mass)
    model = np.array(mean_full, dtype=float, copy=True)
    model[np.ix_(BASIS, WINDOW)] = fitted
    T = model.shape[1]
    ncfg = len(fit.samples) * fit.p['block_length']
    null_costs = []
    for _ in range(n_null):
        cfg = rng.normal(size=(ncfg, len(BASE), 1)) * cfg_sigma
        eps = rng.normal(size=(ncfg, len(BASE), T))
        for k in range(1, T):
            eps[:, :, k] = time_rho * eps[:, :, k - 1] + np.sqrt(1 - time_rho ** 2) * rng.normal(size=(ncfg, len(BASE)))
        data = model[None] * (1 + cfg + noise * eps)
        null_fit = ReferenceFit(blocks(data, fit.p['block_length']), fit.p, WINDOW)
        _, null_cost = null_fit.fit(null_fit.samples.mean(axis=0))
        null_costs.append(null_cost)
    null_costs = np.asarray(null_costs)
    return float((1 + np.sum(null_costs >= cost_observed)) / (n_null + 1)), null_costs


def main(n_dataset=100, n_null=50, seed=20261119):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for kind in ('single', 'two-state'):
        for i in range(n_dataset):
            data = generate(rng, kind=kind)
            fit, mass, cost, dof = fit_gate(data)
            asymptotic_p = float(chi2.sf(cost, dof))
            null_p, _ = null_pvalue(fit, mass, cost, data.mean(axis=0), rng, n_null=n_null)
            rows.append({'kind': kind, 'dataset': i, 'mass': mass, 'cost': cost, 'dof': dof,
                         'asymptotic_p': asymptotic_p, 'null_p': null_p})
        print(kind, 'done', n_dataset)
    result = {'n_dataset': n_dataset, 'n_null': n_null, 'seed': seed, 'dof': rows[0]['dof'],
              'window': WINDOW, 'rows': rows,
              'rates': {}}
    for kind in ('single', 'two-state'):
        subset = [r for r in rows if r['kind'] == kind]
        for gate in ('asymptotic_p', 'null_p'):
            k = sum(r[gate] < 0.05 for r in subset)
            result['rates'][f'{kind}/{gate}'] = {'k': k, 'n': len(subset), 'rate': k / len(subset),
                                                 'corrected': (k + 1) / (len(subset) + 1)}
    (OUT / 'reference_gate_calibration.json').write_text(json.dumps(result, indent=2) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = list(result['rates'])
    axes[0].bar(range(len(labels)), [result['rates'][k]['rate'] for k in labels])
    axes[0].axhline(0.05, color='k', ls='--', lw=0.8, label='nominal 0.05')
    axes[0].set_xticks(range(len(labels)), labels, rotation=20, fontsize=7)
    axes[0].set(ylabel='rejection rate', title='reference gate (100 datasets)')
    axes[0].legend(fontsize=7)
    for gate, color in (('asymptotic_p', 'tab:blue'), ('null_p', 'tab:orange')):
        values = [r[gate] for r in rows if r['kind'] == 'single']
        axes[1].hist(values, bins=20, alpha=0.5, label=gate, color=color)
    axes[1].set(xlabel='p (single truth)', ylabel='count', title='p distribution under the correct model')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / 'reference_gate_calibration.png', dpi=150)
    print(json.dumps(result['rates'], indent=2))


if __name__ == '__main__':
    main()
