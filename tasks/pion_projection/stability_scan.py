"""Pre-declared block-length / covariance-shrinkage scan (todo Phase B5).

The grid below is fixed before looking at the results; it is development-only
and is not used to pick a validation winner.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .statistics import ReferenceFit, blocks, bootstrap_indices, fold

RUN = Path('results/pion_projection/s16t16_v1')
OUT = Path('results/pion_projection/score_diagnostics_v2')
BLOCK_GRID = [2, 3, 4, 5, 6]
SHRINK_GRID = [0.0, 0.1, 0.25, 0.5]


def main():
    p = json.loads((RUN / 'development_provenance.json').read_text())['protocol']
    raw = np.load(RUN / 'development.npz')['correlators']
    rows = []
    for length in BLOCK_GRID:
        data = blocks(fold(raw), length)
        for shrink in SHRINK_GRID:
            config = dict(p, block_length=length, covariance_shrinkage=shrink)
            for window in p['reference_windows']:
                model = ReferenceFit(data, config, window)
                result, _ = model.bootstrap(bootstrap_indices(len(data), 200, p['seed']))
                rows.append({'block_length': length, 'shrinkage': shrink, 'n_blocks': len(data),
                             'window': result['window'], 'mass': result['mass'], 'se': result['se'],
                             'chi2': result['chi2'], 'dof': result['dof'], 'p_approx': result['p_approx']})
            print(f'block {length} shrink {shrink}: done', flush=True)
    (OUT / 'stability_scan.json').write_text(json.dumps({'grid': {'block_length': BLOCK_GRID,
                                                                  'shrinkage': SHRINK_GRID},
                                                         'rows': rows}, indent=2) + '\n')

    nominal = p['reference_nominal_window']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for shrink in SHRINK_GRID:
        sel = [r for r in rows if r['shrinkage'] == shrink and r['window'] == p['reference_windows'][nominal]]
        axes[0].errorbar([r['block_length'] for r in sel], [r['mass'] for r in sel],
                         yerr=[r['se'] for r in sel], marker='o', capsize=3, label=f'shrink {shrink}')
        axes[1].plot([r['block_length'] for r in sel], [r['p_approx'] for r in sel], 'o-', label=f'shrink {shrink}')
    sel0 = [r for r in rows if r['shrinkage'] == 0.25]
    for window in p['reference_windows']:
        sel = [r for r in sel0 if r['window'] == window]
        axes[2].plot([r['block_length'] for r in sel], [r['mass'] for r in sel], 'o-', label=str(window))
    axes[0].set(xlabel='block length', ylabel='$m_0$ (nominal window)')
    axes[1].set(xlabel='block length', ylabel='$p$ (nominal window)')
    axes[2].set(xlabel='block length', ylabel='$m_0$ (shrinkage 0.25)', title='window dependence')
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / 'stability_scan.png', dpi=150)
    print('wrote', OUT / 'stability_scan.json', OUT / 'stability_scan.png')


if __name__ == '__main__':
    main()
