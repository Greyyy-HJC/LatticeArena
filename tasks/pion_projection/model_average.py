"""Development-weighted model average of the reference (todo C1 + B7 profile).

Weights use the development fit only (AIC/BIC over the frozen nominal window);
the validation data are used only to display how far the candidate risks move
between the two models. This is a diagnostic, not a certified score.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from .reference_twostate import TwoStateFit
from .statistics import ReferenceFit, blocks, bootstrap_indices, fold, risk

RUN = Path('results/pion_projection/s16t16_v1')
OUT = Path('results/pion_projection/score_diagnostics_v2')


def profile(model, mean, lo=0.90, hi=1.25, step=0.005):
    grid = np.arange(lo, hi + step / 2, step)
    rows = []
    for m0 in grid:
        res = minimize_scalar(lambda m1: model._cost(m0, m1, mean), bounds=(m0 + 0.05, model.p['mass_bounds'][1]),
                              method='bounded', options={'xatol': 1e-6})
        rows.append({'mass': float(m0), 'cost': float(res.fun), 'mass2': float(res.x)})
    cost = np.array([r['cost'] for r in rows])
    delta = cost - cost.min()
    inside = grid[delta < 1.0]
    return {'grid': [float(x) for x in grid], 'delta_cost': delta.tolist(),
            'interval_1sigma': [float(inside.min()), float(inside.max())] if len(inside) else None,
            'best_mass': float(grid[int(np.argmin(cost))]),
            'gap_at_best': float(rows[int(np.argmin(cost))]['mass2'] - grid[int(np.argmin(cost))])}


def main():
    p = json.loads((RUN / 'development_provenance.json').read_text())['protocol']
    sub = json.loads((RUN / 'submission.json').read_text())
    dev = blocks(fold(np.load(RUN / 'development.npz')['correlators']), p['block_length'])
    val = blocks(fold(np.load(RUN / 'validation.npz')['correlators']), p['block_length'])
    window = p['reference_windows'][p['reference_nominal_window']]
    models = {'single': ReferenceFit(dev, p, window), 'twostate': TwoStateFit(dev, p, window)}
    index = bootstrap_indices(len(dev), p['bootstrap_samples'], p['seed'])
    vi = bootstrap_indices(len(val), p['bootstrap_samples'], p['seed'] + 1)
    fitted, masses = {}, {}
    for name, model in models.items():
        result, boot = model.bootstrap(index)
        fitted[name] = result
        masses[name] = boot
    n = len(models['single'].samples[0])
    k = {'single': len(models['single'].basis) + 1, 'twostate': 2 * len(models['twostate'].basis) + 2}
    aic = {name: fitted[name]['chi2'] + 2 * k[name] for name in models}
    bic = {name: fitted[name]['chi2'] + k[name] * np.log(n) for name in models}
    w_aic = {name: np.exp(-0.5 * (aic[name] - min(aic.values()))) for name in models}
    w_aic = {name: w / sum(w_aic.values()) for name, w in w_aic.items()}
    w_bic = {name: np.exp(-0.5 * (bic[name] - min(bic.values()))) for name in models}
    w_bic = {name: w / sum(w_bic.values()) for name, w in w_bic.items()}

    risks = {}
    for name, model in models.items():
        rows = {}
        for cand, weights in sub['submissions'].items():
            late = np.array(sub['late_inverse_covariances'][cand])
            rows[cand] = risk(val, weights, fitted[name]['mass'], masses[name], vi, p,
                              require_valid=False, late_inverse=late)['risk']
        risks[name] = rows
    average = {cand: sum(w_bic[name] * risks[name][cand] for name in models) for cand in risks['single']}
    spread = {cand: [min(risks[name][cand] for name in models), max(risks[name][cand] for name in models)]
              for cand in risks['single']}
    profile_rows = profile(models['twostate'], models['twostate'].samples.mean(0))
    out = {'weights': {'aic': w_aic, 'bic': w_bic},
           'fit': {name: {key: fitted[name][key] for key in ('mass', 'se', 'chi2', 'dof')} for name in models},
           'k': k, 'n_data': n,
           'risks': risks, 'bic_average': average, 'model_spread': spread,
           'ranking': {name: ('slsqp_better' if risks[name]['agent_signed_mixture'] < risks[name]['tuned_gaussian']
                              else 'tuned_better') for name in models},
           'profile': profile_rows}
    (OUT / 'model_average.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'weights_bic': {k: round(v, 3) for k, v in w_bic.items()},
                      'risks': {m: {c: round(v, 6) for c, v in rows.items()} for m, rows in risks.items()},
                      'average': {c: round(v, 6) for c, v in average.items()},
                      'ranking': out['ranking'],
                      'profile_1sigma': profile_rows['interval_1sigma'],
                      'profile_best': profile_rows['best_mass']}, indent=2))


if __name__ == '__main__':
    main()
