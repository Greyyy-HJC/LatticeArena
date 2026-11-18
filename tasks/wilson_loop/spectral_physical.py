"""Two-state static-source counterexample for the window gate (P3).

W(T) = c0 exp(-E0 T) + c1 exp(-E1 T) (the zero-temperature form; the finite-Nt
wrap is discussed in doc/wilson_spectral_quality.md). Shows how a short-window
fit can pass the spread gate while its value is biased away from E0, scans the
contamination, and evaluates the unused window_z statistic.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .statistics import blocks, bootstrap_indices, potential, risk

OUT = Path('results/wilson_loop/spectral_physical_v1')
T_VALUES = [4, 5, 6, 7, 8]
WINDOWS = [[4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]]
E0, E1 = 0.5, 1.2


def world(rng, ncfg=80, c1=0.2, noise=0.005):
    t = np.asarray(T_VALUES, dtype=float)
    signal = np.exp(-E0 * t) + c1 * np.exp(-E1 * t)
    return signal[None, None, :] * (1 + noise * rng.normal(size=(ncfg, 1, len(t))))


def analyse(rng, c1, ncfg=80, noise=0.005):
    data = blocks(world(rng, ncfg=ncfg, c1=c1, noise=noise), 5)
    index = bootstrap_indices(len(data), 500, 11)
    v_ref = potential(data.mean(0), WINDOWS[1], T_VALUES)
    result = risk(data, index, WINDOWS, T_VALUES, np.asarray(v_ref))
    nominal = float(potential(data.mean(0), WINDOWS[1], T_VALUES)[0])
    bias = nominal - E0
    window_v = np.array(result['v_windows'])[0]
    boot = data[index].mean(axis=1)
    se_win = np.array([potential(boot, w, T_VALUES).std(axis=0, ddof=1)[0] for w in WINDOWS])
    z = (window_v - window_v[1]) / np.sqrt(se_win ** 2 + se_win[1] ** 2)
    se = se_win
    return {'c1': c1, 'bias': bias, 'se': float(np.mean(se)), 'bias_over_se': abs(bias) / float(np.mean(se)),
            'risk': result['risk'], 'spread_over_v': float(np.mean(np.asarray(result['spread']) / abs(v_ref))),
            'window_z': np.abs(z).tolist(), 'gate_spread_pass': bool(np.mean(np.asarray(result['spread']) / abs(v_ref)) <= 0.05)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261122)
    cases = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    rows = [analyse(rng, c1) for c1 in cases]
    result = {'E0': E0, 'E1': E1, 'windows': WINDOWS, 't_values': T_VALUES,
              'rows': rows,
              'window_z_threshold': 3.0,
              'note': ('bias is the short-window potential minus E0; the spread gate can pass '
                       'while bias_over_se is large. window_z compares the nominal window with '
                       'the others in paired SE units.')}
    (OUT / 'spectral_physical.json').write_text(json.dumps(result, indent=2) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    c1s = [r['c1'] for r in rows]
    axes[0].plot(c1s, [r['bias'] for r in rows], 'o-', label='bias')
    axes[0].plot(c1s, [r['se'] for r in rows], 's--', label='SE')
    axes[0].set(xlabel='$c_1$', ylabel='short-window $V-E_0$', title='two-state bias vs contamination')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    axes[1].plot(c1s, [max(r['window_z']) for r in rows], 'o-', label='max |window z|')
    axes[1].axhline(3.0, color='k', ls='--', lw=0.8, label='z=3')
    axes[1].plot(c1s, [r['spread_over_v'] for r in rows], 's--', label='spread/V')
    axes[1].set(xlabel='$c_1$', ylabel='gate statistic', title='spread gate vs window z')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'spectral_physical.png', dpi=150)
    for row in rows:
        print(row['c1'], 'bias %.4f' % row['bias'], 'bias/se %.1f' % row['bias_over_se'],
              'spread/V %.4f' % row['spread_over_v'], 'max|z| %.2f' % max(row['window_z']),
              'spread_pass', row['gate_spread_pass'])
    print('wrote', OUT)


if __name__ == '__main__':
    main()
