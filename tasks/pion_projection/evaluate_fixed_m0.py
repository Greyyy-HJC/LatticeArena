"""Held-out scores when the reference mass is fixed (diagnostic only).

This answers "what would the frozen candidates score if m0 were exactly X",
with a paired block bootstrap that only resamples the validation data. It is
not a replacement for a certified reference; it isolates the m0 systematic.
"""
import json
from pathlib import Path

import numpy as np

from .statistics import blocks, bootstrap_indices, fold, risk

RUN = Path('results/pion_projection/s16t16_v1')
MASSES = {'single': 1.0861669411, 'twostate': 1.0570, 'gevp': 1.0350}
OUTER, INNER = 200, 128


def paired(val, sub, m0, p, seed):
    rng = np.random.default_rng(seed)
    names = list(sub['submissions'])
    draws = np.zeros((OUTER, len(names)))
    for o in range(OUTER):
        rv = rng.integers(0, len(val), len(val))
        vi = rng.integers(0, len(val), (INNER, len(val)))
        for j, name in enumerate(names):
            draws[o, j] = risk(val[rv], sub['submissions'][name], m0, np.full(INNER, m0), vi, p,
                               require_valid=False,
                               late_inverse=np.array(sub['late_inverse_covariances'][name]))['risk']
    return draws, names


def main():
    p = json.loads((RUN / 'development_provenance.json').read_text())['protocol']
    sub = json.loads((RUN / 'submission.json').read_text())
    val = blocks(fold(np.load(RUN / 'validation.npz')['correlators']), p['block_length'])
    out = {}
    for label, m0 in MASSES.items():
        draws, names = paired(val, sub, m0, p, seed=20261118)
        point = {}
        idx = bootstrap_indices(len(val), INNER, p['seed'] + 1)
        for j, name in enumerate(names):
            point[name] = risk(val, sub['submissions'][name], m0, np.full(INNER, m0), idx, p,
                               require_valid=False,
                               late_inverse=np.array(sub['late_inverse_covariances'][name]))['risk']
        base = point['local']
        scores = {n: 100 * base / r for n, r in point.items()}
        j, k = names.index('agent_signed_mixture'), names.index('tuned_gaussian')
        ratio = draws[:, k] / draws[:, j]
        out[label] = {'m0': m0, 'risks': point, 'scores': scores,
                      'slsqp_over_tuned': float(np.median(ratio)),
                      'slsqp_over_tuned_95': np.quantile(ratio, [0.025, 0.975]).tolist()}
        print(label, 'm0', m0, {n: round(s, 0) for n, s in scores.items()},
              'slsqp/tuned', round(out[label]['slsqp_over_tuned'], 3),
              [round(x, 3) for x in out[label]['slsqp_over_tuned_95']])
    (RUN.parent / 'fixed_m0_scores.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
