"""Physical-quality diagnostics: what the risk gate can and cannot see (todo P3)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .statistics import blocks, bootstrap_indices, potential, risk

OUT = Path('results/wilson_loop/spectral_quality_v1')
T_VALUES = [4, 5, 6, 7, 8]
WINDOWS = [[4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]]


def synthetic_flat_bias(seed=3, ncfg=400, bias=0.04, noise=0.01):
    """W(T) = exp(-(V0+bias) T): every window agrees, yet the value is biased."""
    rng = np.random.default_rng(seed)
    T = np.asarray(T_VALUES, dtype=float)
    signal = np.exp(-(0.5 + bias) * T)
    raw = signal[None, None, :] * (1 + noise * rng.normal(size=(ncfg, 1, len(T))))
    data = blocks(raw, 5)
    index = bootstrap_indices(len(data), 500, seed)
    v_ref = potential(data[:, 0].mean(0), WINDOWS[1], T_VALUES)
    result = risk(data, index, WINDOWS, T_VALUES, v_ref)
    return {'v_ref': float(v_ref), 'risk': result['risk'],
            'window_v': result['v_windows'], 'se': result['se'], 'spread': result['spread'],
            'true_potential': 0.5, 'bias': bias}


def run_diagnostics(run):
    root = Path(f'results/wilson_loop/{run}')
    sub = json.loads((root / 'submission.json').read_text())
    data = blocks(np.load(root / 'validation.npz')['loops'], 5)
    index = bootstrap_indices(len(data), 500, 11)
    v_ref = np.asarray(sub['identity']['v'])
    rows = []
    for k, spec in enumerate(json.loads((root / 'development_provenance.json').read_text())['protocol']['kernels']):
        r = risk(data[:, k], index, WINDOWS, T_VALUES, v_ref)
        rows.append({'kernel': spec['name'], 'risk': r['risk'],
                     'v_windows': r['v_windows'], 'se': r['se'], 'spread': r['spread']})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    synthetic = synthetic_flat_bias()
    runs = {run: run_diagnostics(run) for run in ('s16t16_v3', 's16t16_v4')}
    (OUT / 'spectral_quality.json').write_text(json.dumps(
        {'synthetic_flat_bias': synthetic, 'runs': runs, 'windows': WINDOWS,
         't_values': T_VALUES}, indent=2) + '\n')

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].errorbar([5], [synthetic['v_ref']], yerr=[float(np.mean(synthetic['se']))],
                     fmt='o', capsize=3, label='measured V(R)')
    axes[0].axhline(synthetic['true_potential'], color='k', ls='--', label='true $E_0$')
    axes[0].set(title='synthetic: flat but biased', xlabel='T', ylabel='$V_{eff}$')
    axes[0].legend(fontsize=7)
    for run, rows in runs.items():
        base = rows[0]
        v = np.asarray(base['v_windows'])
        se = np.asarray(base['se'])
        for r in range(v.shape[0]):
            axes[1].errorbar(np.arange(len(WINDOWS)), v[r], yerr=se[r], fmt='o-', capsize=3,
                             label=f'{run} R={r + 1}')
    axes[1].set(xticks=range(len(WINDOWS)), xticklabels=[str(w) for w in WINDOWS],
                title='identity V per window', ylabel='V')
    axes[1].tick_params(axis='x', labelsize=7)
    axes[1].legend(fontsize=7)
    for run, rows in runs.items():
        names = [r['kernel'] for r in rows]
        se = [float(np.mean((np.asarray(r['se']) / 1) ** 2)) for r in rows]
        sp = [float(np.mean((np.asarray(r['spread']) / 1) ** 2)) for r in rows]
        axes[2].bar(np.arange(len(names)) + (0.2 if run.endswith('v4') else -0.2), se, width=0.4,
                    label=run + ' SE term')
    axes[2].set(xticks=range(len(names)), xticklabels=names, title='risk SE term')
    axes[2].tick_params(axis='x', labelsize=6, labelrotation=20)
    fig.tight_layout()
    fig.savefig(OUT / 'spectral_quality.png', dpi=150)
    print(json.dumps({'synthetic': {k: synthetic[k] for k in ('v_ref', 'risk', 'spread', 'bias', 'se')},
                      'runs': {r: [x['kernel'] for x in rows] for r, rows in runs.items()}}, indent=2))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
