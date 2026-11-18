"""Double-layer calibration of the reference gate (todo P1.1/P1.2).

Outer layer: generate an independent dataset each simulation. Inner layer: fit
the same analysis pipeline (channel scales, covariance/shrinkage, reference) --
or, in 'fixed' mode, reuse the metric of a template dataset as a conditional
calibration. Reports the false-positive rate with (k+1)/(Nsim+1) resolution.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from .reference import make_reference
from .statistics import ReferenceFit, blocks

OUT = Path('results/pion_projection/score_calibration_v1')
AMPLITUDES = np.array([1.0, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1])
PROTOCOL = json.loads((Path('results/pion_projection/s16t16_v1/development_provenance.json')).read_text())['protocol']


def dataset(rng, ncfg=120, T=16, energies=(1.077,), state_amplitudes=None, noise=0.02,
            rho=0.5, heavy=False):
    """Synthetic seven-channel data with an optional two-state signal."""
    if state_amplitudes is None:
        state_amplitudes = [AMPLITUDES * (1.0 if i == 0 else 0.3) for i in range(len(energies))]
    t = np.arange(T)
    signal = np.zeros((len(AMPLITUDES), T))
    for energy, amplitudes in zip(energies, state_amplitudes):
        signal += np.asarray(amplitudes)[:, None] * (np.exp(-energy * t) + np.exp(-energy * (T - t)))
    eps = rng.normal(size=(ncfg, len(AMPLITUDES), T))
    if rho:
        for k in range(1, T):
            eps[:, :, k] = (rho * eps[:, :, k - 1]
                            + np.sqrt(1 - rho * rho) * rng.normal(size=eps[:, :, k].shape))
    if heavy:
        eps = eps / np.sqrt(rng.chisquare(3, size=eps.shape) / 3)
    return signal[None] * (1 + noise * eps)


class FixedMetricFit(ReferenceFit):
    """Use a template's channel scales and inverse covariance on new data."""

    def __init__(self, template, data_blocks, config, window):
        super().__init__(data_blocks, config, window)
        self.scales = template.scales
        self.samples = (data_blocks[:, self.basis][:, :, self.window]
                        / template.scales[None, :, None]).reshape(len(data_blocks), -1)
        self.inv = template.inv


def false_positive_rate(mode, truth, nsim, seed, blocks_=5, window=None, noise=0.02, rho=0.5, heavy=False):
    rng = np.random.default_rng(seed)
    window = window or PROTOCOL['reference_windows'][PROTOCOL['reference_nominal_window']]
    template = None
    hits = 0
    rows = []
    for _ in range(nsim):
        raw = dataset(rng, energies=(1.077,) if truth == 'single' else (1.077, 1.55),
                      noise=noise, rho=rho, heavy=heavy)
        data = blocks(raw, blocks_)
        if mode == 'fixed':
            if template is None:
                template = ReferenceFit(data, PROTOCOL, window)
            fit = FixedMetricFit(template, data, PROTOCOL, window)
        else:
            fit = ReferenceFit(data, PROTOCOL, window)
        mass, cost = fit.fit(fit.samples.mean(0))
        dof = fit.samples.shape[1] - 2
        pvalue = float(chi2.sf(cost, dof))
        hits += pvalue < 0.05
        rows.append({'mass': float(mass), 'chi2': float(cost), 'p': pvalue})
    k, n = hits, nsim
    from scipy.stats import beta
    low, high = beta.ppf([0.025, 0.975], k + 0.5, n - k + 0.5) if k > 0 else (0.0, beta.ppf(0.975, 0.5, n + 0.5))
    return {'mode': mode, 'truth': truth, 'nsim': n, 'hits': k,
            'rate': k / n, 'rate_corrected': (k + 1) / (n + 1),
            'interval': [float(low), float(high)], 'rows': rows}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for mode in ('full', 'fixed'):
        for truth in ('single', 'two-state'):
            result = false_positive_rate(mode, truth, nsim=400, seed=11 if truth == 'single' else 12)
            results.append(result)
            print(mode, truth, 'k/n %d/%d' % (result['hits'], result['nsim']),
                  'rate %.4f' % result['rate'], 'interval [%.3f, %.3f]' % tuple(result['interval']))
    scan = []
    for block_size, nblocks in ((20, 6), (10, 12), (5, 24)):
        for truth in ('single', 'two-state'):
            result = false_positive_rate('full', truth, nsim=200, seed=21,
                                         blocks_=block_size)
            scan.append({k: result[k] for k in ('mode', 'truth', 'nsim', 'hits', 'rate',
                                                'rate_corrected', 'interval')} | {'blocks': nblocks})
            print('scan', nblocks, truth, 'rate %.3f' % result['rate'])
    (OUT / 'calibration.json').write_text(json.dumps({'main': results, 'scan': scan}, indent=2) + '\n')

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f"{r['mode']}\n{r['truth']}" for r in results]
    rates = [r['rate'] for r in results]
    errors = np.array([[r['rate'] - r['interval'][0], r['interval'][1] - r['rate']] for r in results]).T
    ax.bar(range(len(results)), rates, yerr=errors, capsize=4)
    ax.axhline(0.05, color='k', ls='--', lw=0.8, label='nominal 0.05')
    ax.set_xticks(range(len(results)), labels)
    ax.set(ylabel='false-positive rate', title='reference gate calibration (400 datasets)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'score_calibration.png', dpi=150)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for truth, marker in (('single', 'o'), ('two-state', 's')):
        rows = [r for r in scan if r['truth'] == truth]
        ax2.errorbar([r['blocks'] for r in rows], [r['rate'] for r in rows],
                     yerr=[[r['rate'] - r['interval'][0] for r in rows],
                           [r['interval'][1] - r['rate'] for r in rows]],
                     fmt=marker + '-', capsize=3, label=truth)
    ax2.axhline(0.05, color='k', ls='--', lw=0.8, label='nominal 0.05')
    ax2.set(xlabel='blocks', ylabel='rejection rate', title='power vs blocks (full estimator, 200 datasets)')
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUT / 'score_calibration_blocks.png', dpi=150)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
