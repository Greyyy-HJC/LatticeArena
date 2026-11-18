"""Score scale under two reference models (single-cosh vs two-state)."""
import json
from pathlib import Path

import numpy as np

from .reference_twostate import TwoStateFit
from .statistics import ReferenceFit, blocks, bootstrap_indices, fold, risk

RUN = Path('results/pion_projection/s16t16_v1')


def main():
    p = json.loads((RUN / 'development_provenance.json').read_text())['protocol']
    sub = json.loads((RUN / 'submission.json').read_text())
    dev = blocks(fold(np.load(RUN / 'development.npz')['correlators']), p['block_length'])
    val = blocks(fold(np.load(RUN / 'validation.npz')['correlators']), p['block_length'])
    window = p['reference_windows'][p['reference_nominal_window']]
    out = {}
    for name, cls in (('single', ReferenceFit), ('twostate', TwoStateFit)):
        model = cls(dev, p, window)
        idx = bootstrap_indices(len(dev), p['bootstrap_samples'], p['seed'])
        ref, masses = model.bootstrap(idx)
        vi = bootstrap_indices(len(val), p['bootstrap_samples'], p['seed'] + 1)
        row = {'mass': ref['mass'], 'se': ref['se'], 'chi2': ref['chi2'],
               'dof': ref['dof'], 'p': ref['p_approx'], 'risks': {}}
        for cand, w in sub['submissions'].items():
            late = np.array(sub['late_inverse_covariances'][cand])
            row['risks'][cand] = risk(val, w, ref['mass'], masses, vi, p,
                                      require_valid=False, late_inverse=late)['risk']
        row['score_tuned'] = 100 * row['risks']['local'] / row['risks']['tuned_gaussian']
        row['score_slsqp'] = 100 * row['risks']['local'] / row['risks']['agent_signed_mixture']
        out[name] = row
        print(name, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items() if k != 'risks'}))
        print('  risks', {k: round(v, 6) for k, v in row['risks'].items()})
    (RUN / 'reference_systematics.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
