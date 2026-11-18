"""Known-spectrum energy-estimator experiment (todo 6.1).

Correlators -> single-cosh / two-state fits -> bias, variance, MSE, coverage;
ground-state-disappearance counterexample; anchor sensitivity; and the
bootstrap-vs-independent-repeat semantics of the energy difference.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .energy_target import energy_delta, lower_bound
from .reference import make_reference
from .statistics import ReferenceFit, blocks, bootstrap_indices, fold

OUT = Path('results/pion_projection/energy_target_v1')
RUN = Path('results/pion_projection/s16t16_gevp200')
AMPLITUDES = np.array([1.0, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1])
T_VALUES = 16


def protocol():
    return json.loads((RUN / 'development_provenance.json').read_text())['protocol']


def correlators(rng, ncfg=120, energies=(0.80, 1.30), state_amplitudes=None, noise=0.02):
    if state_amplitudes is None:
        state_amplitudes = [AMPLITUDES, 0.3 * AMPLITUDES]
    t = np.arange(T_VALUES)
    signal = np.zeros((len(AMPLITUDES), T_VALUES))
    for energy, amplitudes in zip(energies, state_amplitudes):
        signal += np.asarray(amplitudes)[:, None] * (np.exp(-energy * t) + np.exp(-energy * (T_VALUES - t)))
    return signal[None] * (1 + noise * rng.normal(size=(ncfg, len(AMPLITUDES), T_VALUES)))


def estimate(raw, p, kind='single', nboot=100, seed=1):
    data = blocks(fold(raw), p['block_length'])
    window = p['reference_windows'][p['reference_nominal_window']]
    if kind == 'single':
        ref = ReferenceFit(data, p, window)
    else:
        from .reference_twostate import TwoStateFit
        ref = TwoStateFit(data, p, window)
    index = bootstrap_indices(len(data), nboot, seed)
    fit, masses = ref.bootstrap(index)
    return float(fit['mass']), np.asarray(masses)


def bias_variance(p, kind, nrep=80, seed=3, e0=0.80, state_amplitudes=None):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(nrep):
        raw = correlators(rng, state_amplitudes=state_amplitudes)
        mass, masses = estimate(raw, p, kind, nboot=80, seed=int(rng.integers(1e6)))
        values.append({'mass': mass, 'se': float(masses.std(ddof=1))})
    masses = np.array([v['mass'] for v in values])
    ses = np.array([v['se'] for v in values])
    z = (masses - e0) / ses
    return {'estimator': kind, 'replicates': nrep, 'e0': e0,
            'mean': float(masses.mean()), 'bias': float(masses.mean() - e0),
            'variance': float(masses.var(ddof=1)), 'mse': float(np.mean((masses - e0) ** 2)),
            'mean_se': float(ses.mean()),
            'coverage': float(np.mean(np.abs(z) <= 1.96)),
            'values': masses.tolist(), 'ses': ses.tolist()}


def anchor_sensitivity(p, nrep=80, seed=7, e0=0.80):
    rng = np.random.default_rng(seed)
    rows = []
    cases = [(0.0, 0.0), (0.005, 0.0), (0.01, 0.0), (0.02, 0.0),
             (0.0, 0.01), (0.0, 0.02), (0.0, 0.05)]
    for sigma_anchor, bias_anchor in cases:
        certified = 0
        for _ in range(nrep):
            raw_b = correlators(rng)
            raw_c = correlators(rng, noise=0.01)
            e_b, sb = estimate(raw_b, p, 'single', 80, int(rng.integers(1e6)))
            e_c, sc = estimate(raw_c, p, 'single', 80, int(rng.integers(1e6)))
            anchor = e0 + bias_anchor + (rng.normal(0, sigma_anchor) if sigma_anchor else 0.0)
            delta = energy_delta(np.full(200, e_b), np.full(200, e_c), np.full(200, anchor), 0.01)
            certified += lower_bound(delta, 0.05) > 0
        rows.append({'anchor_sigma': sigma_anchor, 'anchor_bias': bias_anchor,
                     'certified_fraction': certified / nrep, 'replicates': nrep})
    return rows


def nboot_semantics(p, nrep=60, seed=11, e0=0.80):
    """Naive 1/sqrt(Nboot) shrinking vs the independent-dataset uncertainty."""
    rng = np.random.default_rng(seed)
    naive, dataset = [], []
    masses_all = []
    for _ in range(nrep):
        raw = correlators(rng)
        mass, masses = estimate(raw, p, 'single', nboot=800, seed=int(rng.integers(1e6)))
        masses_all.append(masses)
        dataset.append(mass)
    dataset_se = float(np.std(dataset, ddof=1) / np.sqrt(len(dataset)))
    for nboot in (50, 200, 800):
        means = np.array([m[:nboot].mean() for m in masses_all])
        naive.append({'nboot': nboot, 'naive_se': float(np.mean([m[:nboot].std(ddof=1) for m in masses_all]) / np.sqrt(nboot))})
    return {'dataset_level_se': dataset_se,
            'per_dataset_bootstrap_mean_se': float(np.mean([m.std(ddof=1) for m in masses_all])),
            'naive': naive}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = protocol()
    single = bias_variance(p, 'single')
    two = bias_variance(p, 'twostate')
    missing = bias_variance(p, 'single', state_amplitudes=[0.02 * AMPLITUDES, AMPLITUDES], seed=5)
    anchors = anchor_sensitivity(p)
    semantics = nboot_semantics(p)
    result = {'estimators': {'single': single, 'twostate': two, 'ground_state_missing': missing},
              'anchor_sensitivity': anchors, 'nboot_semantics': semantics}
    (OUT / 'energy_experiment.json').write_text(json.dumps(result, indent=2) + '\n')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, key in (('single', 'single'), ('two-state', 'twostate'), ('missing ground state', 'ground_state_missing')):
        values = np.array(result['estimators'][key]['values'])
        ses = np.array(result['estimators'][key]['ses'])
        axes[0].errorbar([name], [values.mean()], yerr=[ses.mean()], fmt='o', capsize=4, label=name)
    axes[0].axhline(0.80, color='k', ls='--', label='true $E_0$')
    axes[0].set(ylabel='$\\hat E$', title='estimators on known spectra')
    axes[0].legend(fontsize=7)
    for bias_anchor in (0.0, 0.02):
        rows = [r for r in anchors if r['anchor_bias'] == bias_anchor]
        axes[1].plot([r['anchor_sigma'] for r in rows], [r['certified_fraction'] for r in rows], 'o-',
                     label=f'anchor bias {bias_anchor}')
    axes[1].axhline(0.05, color='k', ls='--', lw=0.8, label='nominal 0.05')
    axes[1].set(xlabel='anchor $\\sigma$', ylabel='certified fraction', title='anchor sensitivity')
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / 'energy_bias_variance.png', dpi=150)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot([r['nboot'] for r in semantics['naive']], [r['naive_se'] for r in semantics['naive']], 'o-',
             label='naive SE (std/sqrt(Nboot))')
    ax2.axhline(semantics['dataset_level_se'], color='k', ls='--',
                label='independent-dataset SE')
    ax2.set(xscale='log', xlabel='$N_{boot}$', ylabel='reported SE', title='bootstrap vs independent repeats')
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(OUT / 'anchor_sensitivity.png', dpi=150)
    print(json.dumps({k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                          for kk, vv in v.items() if kk in ('bias', 'variance', 'mse', 'coverage', 'mean_se')}
                      for k, v in result['estimators'].items()}, indent=2))
    print('anchors', anchors)
    print('semantics', {k: v for k, v in semantics.items() if k != 'naive'})
    print('wrote', OUT)


if __name__ == '__main__':
    main()
