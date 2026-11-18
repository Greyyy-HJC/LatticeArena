"""Synthetic regression for the amplitude-source estimand (todo CPU package B).

Known model W(T) = A f0(T; m) (1 + eps), independent config noise, train/test
folds. Checks that point and bootstrap replicas use the same amplitude rule and
that the q=1 truth is covered by the replica interval at the nominal rate.
"""
import json
from pathlib import Path

import numpy as np

from tasks.pion_projection.statistics import blocks

from .unified import risk_with_fixed_amplitude

OUT = Path('results/optimize_gaussian/cpu_next_package_b/regression')
PROTOCOL = {'amplitude_window': [6, 7, 8], 'score_window': [2, 3, 4, 5],
            'geometry': [4, 4, 4, 9], 'covariance_shrinkage': 0.1}
M, A = 0.5, 1.0


def f0(t, m=M, T=9):
    return np.exp(-m * t) + np.exp(-m * (T - t))


def dataset(rng, ncfg=80, noise=0.01):
    t = np.arange(PROTOCOL['geometry'][3])
    signal = A * f0(t)
    return signal[None, None, :] * (1 + noise * rng.normal(size=(ncfg, 1, len(t))))


def run(n_dataset=60, n_boot=200, seed=20261123):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_dataset):
        data = blocks(dataset(rng), 5)
        split = rng.permutation(len(data))
        train, test = data[split[:3]], data[split[3:]]
        masses = np.full(n_boot, M) + rng.normal(0, 0.002, n_boot)
        index = rng.integers(0, len(test), (n_boot, len(test)))
        weights = np.ones(1)
        row = {}
        for source in ('train', 'test'):
            metric = risk_with_fixed_amplitude(test, weights, M, masses, index, PROTOCOL,
                                               A, 0.01, amplitude_source=source,
                                               return_draws=True)
            qb = np.asarray(metric['q_boot'])
            q = np.asarray(metric['q'])
            row[source] = {'q': q.tolist(),
                           'replica_mean': qb.mean(axis=0).tolist(),
                           'point_minus_replica': float(np.mean(qb.mean(axis=0) - q)),
                           'cover_q1': [bool(lo <= 1.0 <= hi) for lo, hi in
                                        np.quantile(qb, [0.025, 0.975], axis=0).T],
                           'risk': metric['risk'],
                           'late_covariance_source': metric['late_covariance_source']}
        rows.append(row)
    result = {'n_dataset': n_dataset, 'n_boot': n_boot, 'seed': seed, 'm': M, 'A': A,
              'rows': rows,
              'summary': {}}
    for source in ('train', 'test'):
        biases = [r[source]['point_minus_replica'] for r in rows]
        coverage = np.mean([np.mean(r[source]['cover_q1']) for r in rows])
        result['summary'][source] = {'bias_mean': float(np.mean(biases)),
                                     'bias_se': float(np.std(biases, ddof=1) / np.sqrt(len(biases))),
                                     'q1_coverage': float(coverage)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'regression.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))
    return result


if __name__ == '__main__':
    run()
