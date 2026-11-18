"""Reference diagnostics on the 120-development-config run (todo P1.3)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from .reference import make_reference
from .reference_twostate import TwoStateFit
from .statistics import ReferenceFit, blocks, fold

RUN = Path('results/pion_projection/s16t16_gevp200')
OUT = Path('results/pion_projection/score_diagnostics_v3')
WINDOWS = {'4-8': [4, 5, 6, 7, 8], '5-8': [5, 6, 7, 8], '6-8': [6, 7, 8]}
GRID = [(b, s) for b in (2, 3, 4, 5, 6) for s in (0.0, 0.1, 0.25, 0.5)]


def covariance_diagnostics(data, basis):
    rows = []
    for label, window in WINDOWS.items():
        samples = data[:, basis][:, :, window].reshape(len(data), -1)
        raw = np.cov(samples.T)
        scaled = np.cov((samples / np.abs(samples).mean(0)).T)
        entry = {'window': label, 'dim': samples.shape[1], 'blocks': len(data)}
        for name, matrix in (('raw', raw), ('scaled', scaled)):
            values = np.linalg.eigvalsh(matrix)[::-1]
            entry[name] = {
                'rank': int((values > 1e-10 * values[0]).sum()),
                'condition': float(values[0] / values[values > 1e-10 * values[0]][-1]),
                'floor_hits': int((values <= 1e-10 * values[0]).sum()),
                'spectrum': values.tolist(),
            }
        rows.append(entry)
    return rows


def scan(data, p, window):
    rows = []
    for block, shrink in GRID:
        config = dict(p, block_length=block, covariance_shrinkage=shrink)
        try:
            folded = blocks(fold(np.load(RUN / 'development.npz')['correlators']), block)
        except ValueError:
            rows.append({'block': block, 'shrinkage': shrink, 'p': None, 'reason': 'insufficient blocks'})
            continue
        fit = ReferenceFit(folded, config, window)
        mean = fit.samples.mean(0)
        slope = None
        try:
            mass, cost = fit.fit(mean)
            dof = fit.samples.shape[1] - 2
            rows.append({'block': block, 'shrinkage': shrink, 'mass': float(mass),
                         'chi2': float(cost), 'dof': int(dof),
                         'p': float(chi2.sf(cost, dof))})
        except Exception as exc:  # noqa: BLE001
            rows.append({'block': block, 'shrinkage': shrink, 'p': None, 'reason': str(exc)})
    return rows


def mass_window(data, p):
    rows = []
    for start in range(2, 6):
        window = list(range(start, start + 4))
        fit = ReferenceFit(data, p, window)
        index = np.random.default_rng(5).integers(0, len(data), (200, len(data)))
        mass, cost = fit.fit(fit.samples.mean(0))
        masses = np.array([fit.fit(row)[0] for row in fit.samples[index].mean(axis=1)])
        rows.append({'window': window, 'mass': float(mass),
                     'se': float(masses.std(ddof=1)), 'chi2': float(cost)})
    return rows


def twostate_profile(data, p, window):
    fit = TwoStateFit(data, p, window)
    mean = fit.samples.mean(0)
    best = fit.fit_full(mean)
    lo, hi = p['mass_bounds']
    grid = np.arange(0.98, 1.16, 0.005)
    profile = []
    for m0 in grid:
        best_cost = np.inf
        for m1 in np.arange(m0 + 0.05, hi, 0.01):
            cost = fit._cost(m0, m1, mean)
            if cost < best_cost:
                best_cost = cost
        profile.append(float(best_cost))
    profile = np.asarray(profile) - best['cost']
    inside = grid[profile <= 1.0]
    return {'mass': best['mass'], 'mass2': best['mass2'], 'gap': best['gap'],
            'cost': best['cost'], 'at_boundary': best['at_boundary'],
            'condition': best['condition'], 'grid': grid.tolist(),
            'delta_chi2': profile.tolist(),
            'interval_chi2_1': [float(inside.min()), float(inside.max())] if inside.size else None,
            'half_width': float((inside.max() - inside.min()) / 2) if inside.size else None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = json.loads((RUN / 'development_provenance.json').read_text())['protocol']
    raw = np.load(RUN / 'development.npz')['correlators']
    data = blocks(fold(raw), p['block_length'])
    window = p['reference_windows'][p['reference_nominal_window']]
    cov = covariance_diagnostics(data, p['reference_basis'])
    scan_rows = scan(data, p, window)
    windows = mass_window(data, p)
    profile = twostate_profile(data, p, window)
    fit = TwoStateFit(data, p, window)
    rng = np.random.default_rng(5)
    index = rng.integers(0, len(data), (200, len(data)))
    boot = np.array([fit.fit(row)[0] for row in fit.samples[index].mean(axis=1)])
    profile['bootstrap_se'] = float(boot.std(ddof=1))
    profile['half_width_over_se'] = (float(profile['half_width'] / profile['bootstrap_se'])
                                     if profile['half_width'] is not None else None)
    (OUT / 'diagnostics.json').write_text(json.dumps(
        {'covariance': cov, 'block_shrinkage': scan_rows, 'mass_window': windows,
         'twostate_profile': profile}, indent=2) + '\n')

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for entry in cov:
        axes[0, 0].semilogy(np.arange(1, len(entry['raw']['spectrum']) + 1),
                            entry['raw']['spectrum'], 'o-', label=entry['window'])
    axes[0, 0].set(title='raw covariance spectrum', xlabel='mode', ylabel='eigenvalue')
    axes[0, 0].legend(fontsize=8)
    grid_p = np.full((len({b for b, _ in GRID}), len({s for _, s in GRID})), np.nan)
    blocks_ = sorted({b for b, _ in GRID}); shrinks = sorted({s for _, s in GRID})
    for row in scan_rows:
        if row.get('p') is not None:
            grid_p[blocks_.index(row['block']), shrinks.index(row['shrinkage'])] = np.log10(max(row['p'], 1e-16))
    image = axes[0, 1].imshow(grid_p, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 1].set(xticks=range(len(shrinks)), xticklabels=[str(s) for s in shrinks],
                   yticks=range(len(blocks_)), yticklabels=[str(b) for b in blocks_],
                   title='single-cosh log10 p (block x shrinkage)', xlabel='shrinkage', ylabel='block')
    fig.colorbar(image, ax=axes[0, 1])
    axes[1, 0].errorbar([w['window'][0] for w in windows], [w['mass'] for w in windows],
                        yerr=[w['se'] for w in windows], fmt='o-', capsize=3)
    axes[1, 0].set(title='single-cosh mass vs window', xlabel='window start', ylabel='$m_0$')
    axes[1, 0].grid(alpha=0.2)
    axes[1, 1].plot(profile['grid'], profile['delta_chi2'])
    axes[1, 1].axhline(1.0, color='k', ls='--', lw=0.8)
    axes[1, 1].set(title='two-state profile $\\Delta\\chi^2$', xlabel='$m_0$', ylabel='$\\Delta\\chi^2$')
    axes[1, 1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / 'diagnostics.png', dpi=150)
    print(json.dumps({'covariance': cov, 'mass_window': windows,
                      'twostate': {k: profile[k] for k in ('mass', 'mass2', 'gap', 'cost',
                                                           'at_boundary', 'condition',
                                                           'interval_chi2_1', 'half_width',
                                                           'bootstrap_se', 'half_width_over_se')},
                      'scan': [r for r in scan_rows if r.get('p') is not None][:5]}, indent=2))


if __name__ == '__main__':
    main()
