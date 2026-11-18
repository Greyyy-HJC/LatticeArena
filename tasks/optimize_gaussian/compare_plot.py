"""Plot the paired comparison (local / SLSQP / A1+)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CAND = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('results/pion_projection/s16t16_v1/candidates/f686a72d9c868716')


def main():
    r = json.loads((CAND / 'compare.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = ['local', 'slsqp', 'a1p']
    axes[0].bar(names, [r['point_risk'][n] for n in names], color=['#888', '#D28B26', '#087F8C'])
    axes[0].set_yscale('log')
    axes[0].set(ylabel='risk (log)', title='Validation risk on the same blocks')
    for i, n in enumerate(names):
        axes[0].text(i, r['point_risk'][n], f"{r['point_risk'][n]:.4g}", ha='center', va='bottom', fontsize=8)
    t = np.arange(2, 6)
    axes[1].plot(t, r['q']['slsqp'], 'o-', label='SLSQP mixture', color='#D28B26')
    axes[1].plot(t, r['q']['a1p'], 's-', label='A1+ learned', color='#087F8C')
    axes[1].axhline(1, color='k', lw=0.8, ls='--')
    axes[1].set(xlabel='$t/a$', ylabel='$q(t)$', title='Shape in the scoring window')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    out = CAND / 'fig_compare.png'
    fig.savefig(out, dpi=160)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
