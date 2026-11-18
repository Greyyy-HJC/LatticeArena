"""Score sensitivity to the reference mass (todo: reference systematics).

Fixes the frozen candidate weights and the late-time metric, then scans m0 in
[1.00, 1.10] and reports the risk/score of each candidate. This isolates the
reference-mass systematic from the statistical uncertainty.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .statistics import blocks, bootstrap_indices, fold, risk

RUN = Path('results/pion_projection/s16t16_v1')
OUT = Path('results/pion_projection/score_diagnostics_v2')
GRID = np.arange(1.00, 1.101, 0.005)


def main():
    p = json.loads((RUN / 'development_provenance.json').read_text())['protocol']
    sub = json.loads((RUN / 'submission.json').read_text())
    dev = blocks(fold(np.load(RUN / 'development.npz')['correlators']), p['block_length'])
    val = blocks(fold(np.load(RUN / 'validation.npz')['correlators']), p['block_length'])
    vi = bootstrap_indices(len(val), p['bootstrap_samples'], p['seed'] + 1)
    rows = []
    for m0 in GRID:
        masses = np.full(p['bootstrap_samples'], m0)
        entry = {'m0': float(m0), 'risks': {}}
        for name, weights in sub['submissions'].items():
            value = risk(val, weights, m0, masses, vi, p, require_valid=False,
                         late_inverse=np.array(sub['late_inverse_covariances'][name]))
            entry['risks'][name] = value['risk']
        baseline = entry['risks']['local']
        entry['scores'] = {n: 100 * baseline / r for n, r in entry['risks'].items()}
        rows.append(entry)
    (OUT / 'm0_scan.json').write_text(json.dumps({'grid': GRID.tolist(), 'rows': rows}, indent=2) + '\n')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ('tuned_gaussian', 'agent_signed_mixture'):
        axes[0].plot(GRID, [r['risks'][name] for r in rows], 'o-', ms=3, label=name)
        axes[1].plot(GRID, [r['scores'][name] for r in rows], 'o-', ms=3, label=name)
    for ax in axes:
        ax.axvline(1.0861669411, color='k', ls='--', lw=0.8)
        ax.axvline(1.0570, color='k', ls=':', lw=0.8)
        ax.axvspan(1.023, 1.050, color='tab:green', alpha=0.1)
        ax.grid(alpha=0.2)
        ax.set_xlabel('$m_0$ (fixed reference mass)')
    axes[0].set(ylabel='validation risk', title='risk vs $m_0$')
    axes[1].set(ylabel='score (100 local / candidate)', title='score vs $m_0$')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'm0_scan.png', dpi=150)
    print('wrote', OUT / 'm0_scan.json', OUT / 'm0_scan.png')
    for at in (1.0862, 1.0570, 1.035):
        row = min(rows, key=lambda r: abs(r['m0'] - at))
        print(f"m0 {at:.4f}: tuned {row['scores']['tuned_gaussian']:.0f} "
              f"slsqp {row['scores']['agent_signed_mixture']:.0f} "
              f"tuned/slsqp risk {row['risks']['tuned_gaussian']/row['risks']['agent_signed_mixture']:.3f}")


if __name__ == '__main__':
    main()
