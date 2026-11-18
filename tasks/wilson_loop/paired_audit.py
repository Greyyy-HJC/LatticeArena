"""Paired-statistics audit of the archived v2/v3/v4 runs (todo P2)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .statistics import blocks, bootstrap_indices, risk

RUNS = {'v2': 'results/wilson_loop/s16t16_v2', 'v3': 'results/wilson_loop/s16t16_v3',
        'v4': 'results/wilson_loop/s16t16_v4'}
OUT = Path('results/wilson_loop/paired_audit_v2')
BLOCK = 5
T_VALUES = [4, 5, 6, 7, 8]
WINDOWS = [[4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]]


def load(run):
    root = Path(RUNS[run])
    sub = json.loads((root / 'submission.json').read_text())
    loops = np.load(root / 'validation.npz')['loops']
    return blocks(loops, BLOCK), sub['windows'], sub['identity']['v']


def decomposition(run, data, windows, t_values, v_ref):
    index = bootstrap_indices(len(data), 500, 7)
    rows = []
    for k in range(data.shape[1]):
        r = risk(data[:, k], index, windows, T_VALUES, np.asarray(v_ref))
        rows.append({'kernel': k, 'risk': r['risk'],
                     'se_term': float(np.mean((np.asarray(r['se']) / np.asarray(v_ref)) ** 2)),
                     'spread_term': float(np.mean((np.asarray(r['spread']) / np.asarray(v_ref)) ** 2)),
                     'se': r['se'], 'spread': r['spread']})
    return rows


def nested(data, windows, v_ref, seed, outer=500, inner=500):
    rng = np.random.default_rng(seed)
    vi = bootstrap_indices(len(data), inner, seed + 1)
    ratios = np.zeros((outer, data.shape[1] - 1))
    num = np.zeros((outer, data.shape[1] - 1))
    den = np.zeros((outer, data.shape[1] - 1))
    unique = np.zeros(outer)
    invalid = 0
    for o in range(outer):
        rows = rng.integers(0, len(data), len(data))
        unique[o] = len(np.unique(rows))
        try:
            base = risk(data[rows, 0], vi, windows, T_VALUES, np.asarray(v_ref))['risk']
        except ValueError:
            invalid += 1
            continue
        for k in range(1, data.shape[1]):
            try:
                cand = risk(data[rows, k], vi, windows, T_VALUES, np.asarray(v_ref))['risk']
            except ValueError:
                invalid += 1
                continue
            num[o, k - 1], den[o, k - 1] = base, cand
            ratios[o, k - 1] = base / cand
    return {'ratios': ratios, 'num': num, 'den': den, 'unique': unique, 'invalid_draws': invalid}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for run in RUNS:
        data, windows, v = load(run)
        index = bootstrap_indices(len(data), 500, 7)
        archived = json.loads((Path(RUNS[run]) / 'validation_results.json').read_text())['results']
        rows = []
        for k in range(data.shape[1]):
            r = risk(data[:, k], index, windows, T_VALUES, v)
            rows.append({'kernel': k, 'risk': r['risk'],
                         'se_term': float(np.mean((np.asarray(r['se']) / np.asarray(v)) ** 2)),
                         'spread_term': float(np.mean((np.asarray(r['spread']) / np.asarray(v)) ** 2))})
        nested_result = nested(data, windows, v, seed=20261118)
        ratios = nested_result['ratios']
        names = list(archived)
        frozen_point = np.array([archived[names[0]]['risk'] / archived[name]['risk'] for name in names[1:]])
        point = np.array([rows[0]['risk'] / r['risk'] for r in rows[1:]])
        med = np.median(ratios, axis=0)
        lo, hi = np.quantile(ratios, [0.025, 0.975], axis=0)
        blocks_unique = np.bincount(nested_result['unique'].astype(int), minlength=len(data) + 1)[1:]
        jackknife = []
        for i in range(len(data)):
            keep = np.setdiff1d(np.arange(len(data)), [i])
            inner = bootstrap_indices(len(keep), 1000, 20261117)
            base = risk(data[keep, 0], inner, windows, T_VALUES, v)['risk']
            jackknife.append([base / risk(data[keep, k], inner, windows, T_VALUES, v)['risk']
                              for k in range(1, data.shape[1])])
        jackknife = np.asarray(jackknife)
        n = len(data)
        jk_mean = jackknife.mean(axis=0)
        jk_se = np.sqrt((n - 1) / n * ((jackknife - jk_mean) ** 2).sum(axis=0))
        result[run] = {'nested_invalid_draws': nested_result.get('invalid_draws'),
            'blocks': len(data), 'kernels': data.shape[1], 'point_risks': rows,
            'point_ratio': point.tolist(), 'median_ratio': med.tolist(),
            'ci95': [lo.tolist(), hi.tolist()],
            'unique_block_counts': blocks_unique.tolist(),
            'point_vs_median': (point / med).tolist(),
            'frozen_point_ratio': frozen_point.tolist(),
            'diagnostic_point_ratio': point.tolist(),
            'jackknife_ratios': jackknife.tolist(),
            'jackknife_mean': jk_mean.tolist(), 'jackknife_se': jk_se.tolist(),
            'corr_num_den': [float(np.corrcoef(nested_result['num'][:, j],
                                              nested_result['den'][:, j])[0, 1])
                             for j in range(data.shape[1] - 1)],
        }
        print(run, 'blocks', len(data), 'point', np.round(point, 3).tolist(),
              'median', np.round(med, 3).tolist(), 'ci', np.round(lo, 2).tolist(), np.round(hi, 2).tolist())
    # seed / block-length stability on v2 (the anomalous run)
    data, windows, v = load('v2')
    scan = []
    for block in (2, 3, 5, 6):
        try:
            data_b = blocks(np.load(Path(RUNS['v2']) / 'validation.npz')['loops'], block)
        except ValueError:
            continue
        for seed in (1, 2, 3):
            n = nested(data_b, windows, v, seed=seed, outer=300, inner=300)
            scan.append({'block_length': block, 'blocks': len(data_b), 'seed': seed,
                         'median': np.median(n['ratios'], axis=0).tolist()})
    result['v2_scan'] = scan

    def synthetic_world(rng, ncfg=80, base_noise=0.02, cand_noise=0.01, nR=3, nT=5):
        t = np.array([4, 5, 6, 7, 8], dtype=float)
        signal = np.exp(-0.5 * np.outer(np.arange(1, nR + 1), t))
        base = signal[None] * (1 + base_noise * rng.normal(size=(ncfg, nR, nT)))
        cand = signal[None] * (1 + cand_noise * rng.normal(size=(ncfg, nR, nT)))
        return base, cand

    def coverage_study(cand_noise, truth, sims=100, outer=100, seed=99):
        rng = np.random.default_rng(seed)
        hits = 0
        for _ in range(sims):
            base, cand = synthetic_world(rng, cand_noise=cand_noise)
            data = np.stack([blocks(base, 5), blocks(cand, 5)], axis=1)
            index = bootstrap_indices(len(data), 500, 11)
            ratios = []
            for _ in range(outer):
                rows = rng.integers(0, len(data), len(data))
                ratios.append(risk(data[rows, 0], index % len(data), WINDOWS, T_VALUES, [0.5] * 3)['risk']
                              / risk(data[rows, 1], index % len(data), WINDOWS, T_VALUES, [0.5] * 3)['risk'])
            lo, hi = np.quantile(ratios, [0.025, 0.975])
            hits += lo <= truth <= hi
        return hits / sims

    # truth for the risk ratio is approximately (base_noise / cand_noise)^2
    coverage_known = coverage_study(0.01, truth=4.0, sims=60)
    coverage_null = coverage_study(0.02, truth=1.0, sims=60)
    result['synthetic_coverage'] = {
        'known_gain': {'truth': 4.0, 'coverage': coverage_known, 'sims': 60},
        'null': {'truth': 1.0, 'coverage': coverage_null, 'sims': 60},
        'note': 'non-degenerate loops through the full mean->log->window->inner risk->outer ratio chain'}

    (OUT / 'paired_audit.json').write_text(json.dumps(result, indent=2) + '\n')

    fig, ax = plt.subplots(figsize=(7, 4))
    for run in RUNS:
        rows = result[run]
        ax.plot(np.arange(1, len(rows['point_ratio']) + 1), rows['point_ratio'], 'o-',
                label=f'{run} point')
        ax.plot(np.arange(1, len(rows['median_ratio']) + 1), rows['median_ratio'], 's--',
                label=f'{run} nested median')
    ax.set(xlabel='candidate kernel', ylabel='risk(identity)/risk(candidate)',
           title='point vs nested ratios (archived runs)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / 'point_vs_nested.png', dpi=150)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
