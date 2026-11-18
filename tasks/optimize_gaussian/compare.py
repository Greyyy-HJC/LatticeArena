"""Paired comparison of local / SLSQP mixture / A1+ learned kernel on the same blocks."""
import json
import sys
from pathlib import Path

import numpy as np

from tasks.pion_projection.experiment import nested_intervals
from tasks.pion_projection.statistics import blocks, fold

RUN = Path('results/pion_projection/s16t16_v1')
CAND = Path(sys.argv[1]) if len(sys.argv) > 1 else RUN / 'candidates' / 'f686a72d9c868716'


def main():
    p = json.loads((RUN / 'development_provenance.json').read_text())['protocol']
    sub = json.loads((RUN / 'submission.json').read_text())
    frozen = json.loads((CAND / 'frozen.json').read_text())

    dev = np.concatenate([np.load(RUN / 'development.npz')['correlators'],
                          np.load(CAND / 'development.npz')['correlators']], axis=1)
    val = np.concatenate([np.load(RUN / 'validation.npz')['correlators'],
                          np.load(CAND / 'validation.npz')['correlators']], axis=1)
    dev_b = blocks(fold(dev), p['block_length'])
    val_b = blocks(fold(val), p['block_length'])

    weights = {'local': [1.0] + [0.0] * 7,
               'slsqp': list(sub['submissions']['agent_signed_mixture']) + [0.0],
               'a1p': [0.0] * 7 + [1.0]}
    late = {'local': np.array(sub['late_inverse_covariances']['local']),
            'slsqp': np.array(sub['late_inverse_covariances']['agent_signed_mixture']),
            'a1p': np.array(frozen['late_inverse'])}
    fake = {'submissions': weights, 'late_inverse_covariances': {k: v.tolist() for k, v in late.items()}}

    from tasks.pion_projection.statistics import ReferenceFit, bootstrap_indices, risk
    model = ReferenceFit(dev_b, p, p['reference_windows'][p['reference_nominal_window']])
    ref, masses = model.bootstrap(bootstrap_indices(len(dev_b), p['bootstrap_samples'], p['seed']))
    vi = bootstrap_indices(len(val_b), p['bootstrap_samples'], p['seed'] + 1)
    point = {k: risk(val_b, w, ref['mass'], masses, vi, p, require_valid=False, late_inverse=late[k])['risk']
             for k, w in weights.items()}
    draws, names = nested_intervals(dev_b, val_b, fake, p)
    np.savez_compressed(CAND / 'paired_draws.npz', draws=draws, names=np.array(names))
    out = {'point_risk': point,
           'ratio_a1p_over_slsqp': point['slsqp'] / point['a1p'],
           'ratio_95': np.quantile(draws[:, names.index('slsqp')] / draws[:, names.index('a1p')],
                                   [0.025, 0.975]).tolist(),
           'ratio_local_over_a1p': point['local'] / point['a1p'],
           'q': {k: risk(val_b, w, ref['mass'], masses, vi, p, require_valid=False,
                         late_inverse=late[k])['q'] for k, w in weights.items()},
           'snr': {k: risk(val_b, w, ref['mass'], masses, vi, p, require_valid=False,
                           late_inverse=late[k])['snr'] for k, w in weights.items()}}
    (CAND / 'compare.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'point': {k: round(v, 6) for k, v in point.items()},
                      'a1p/slsqp': round(out['ratio_a1p_over_slsqp'], 3),
                      'ratio_95': [round(x, 3) for x in out['ratio_95']]}, indent=2))


if __name__ == '__main__':
    main()
