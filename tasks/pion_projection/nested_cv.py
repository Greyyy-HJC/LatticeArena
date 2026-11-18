"""Full-chain nested cross-validation on the development blocks (todo P1.4).

Each outer fold refits the reference, channel scales, covariance and the SLSQP
candidate on the training blocks, then evaluates on the held-out blocks. The
learned-kernel retraining is GPU-side and is recorded as NOT_RUN.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from .common import fingerprint
from .experiment import late_metric
from .reference import make_reference
from .statistics import blocks, bootstrap_indices, fold, risk

RUN = Path('results/pion_projection/s16t16_gevp200')
OUT = Path('results/pion_projection/nested_cv_v2')


def optimise(train, scales, mass, masses, index, p):
    basis = np.eye(len(p['powers']))
    normalized = train / scales[None, :, None]
    def objective(u):
        try:
            return risk(normalized, u, mass, masses, index, p)['risk'] + p['optimizer_ridge'] * float(u @ u)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return 1e10
    outcomes = []
    for start in [*basis, np.ones(len(basis)) / len(basis)][:p['optimizer_starts']]:
        fit = minimize(objective, start, method='SLSQP',
                       bounds=[(-p['optimizer_l1_bound'], p['optimizer_l1_bound'])] * len(basis),
                       constraints=[{'type': 'eq', 'fun': lambda u: u.sum() - 1},
                                    {'type': 'ineq', 'fun': lambda u: p['optimizer_l1_bound'] - abs(u).sum()}],
                       options={'maxiter': p['optimizer_max_iterations'], 'ftol': 1e-11})
        if fit.success and abs(fit.x.sum() - 1) < 1e-6 and abs(fit.x).sum() <= p['optimizer_l1_bound'] + 1e-6 and fit.fun < 1e9:
            outcomes.append(fit)
    if not outcomes:
        raise ValueError('no SLSQP start converged')
    fit = min(outcomes, key=lambda f: f.fun)
    return fit.x / scales, fit.x


def main(k=5, seed=20261120):
    OUT.mkdir(parents=True, exist_ok=True)
    p = json.loads((RUN / 'development_provenance.json').read_text())['protocol']
    raw = np.load(RUN / 'development.npz')['correlators']
    data = blocks(fold(raw), p['block_length'])
    window = p['reference_windows'][p['reference_nominal_window']]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data))
    folds = [np.sort(f) for f in np.array_split(order, k)]
    rows = []
    for i, test_ids in enumerate(folds):
        train = data[np.setdiff1d(np.arange(len(data)), test_ids)]
        test = data[test_ids]
        reference = make_reference(train, p, window, 'single')
        mass, cost = reference.fit(reference.samples.mean(axis=0))
        index = bootstrap_indices(len(train), p['bootstrap_samples'], seed + i)
        _, masses = reference.bootstrap(index)
        basis = np.eye(len(p['powers']))
        base = [risk(train, w, mass, masses, index, p) for w in basis]
        best = int(np.argmin([r['risk'] for r in base]))
        scales = np.array([r['amplitude'] for r in base])
        weights, u = optimise(train, scales, mass, masses, index, p)
        test_index = bootstrap_indices(len(test), p['bootstrap_samples'], seed + 100 + i)
        late = {name: late_metric(train, np.asarray(w, dtype=float), p).tolist()
                for name, w in (('local', basis[0]), ('tuned_gaussian', basis[best]), ('slsqp', weights))}
        test_risk = {name: risk(test, np.asarray(w, dtype=float), mass, masses, test_index, p,
                                late_inverse=np.asarray(late[name]))['risk']
                     for name, w in (('local', basis[0]), ('tuned_gaussian', basis[best]), ('slsqp', weights))}
        import hashlib
        rows.append({'fold': i, 'test_blocks': [int(x) for x in test_ids],
                     'train_blocks': [int(x) for x in np.setdiff1d(np.arange(len(data)), test_ids)],
                     'n_train': len(train), 'n_test': len(test), 'mass': float(mass),
                     'train_chi2': float(cost), 'best_basis': p['powers'][best],
                     'late_inverse_sha256': {name: hashlib.sha256(np.asarray(late[name]).tobytes()).hexdigest()
                                             for name in late},
                     'slsqp_weights': u.tolist(), 'test_risk': test_risk,
                     'ratio_local_slsqp': test_risk['local'] / test_risk['slsqp'],
                     'ratio_local_tuned': test_risk['local'] / test_risk['tuned_gaussian']})
        print('fold', i, 'mass %.4f' % mass, {k_: round(v, 5) for k_, v in test_risk.items()},
              'local/slsqp %.1f' % rows[-1]['ratio_local_slsqp'])
    ratios = np.array([r['ratio_local_slsqp'] for r in rows])
    result = {'protocol_hash': fingerprint(p), 'k': k, 'seed': seed,
              'bindings': {'split': 'development blocks', 'block_length': p['block_length'],
                           'bootstrap_samples': p['bootstrap_samples'],
                           'reference': {'kind': 'single',
                                         'window': p['reference_windows'][p['reference_nominal_window']]},
                           'baseline': 'local', 'score_spec': 'pion-score-v1',
                           'search_budget': {'optimizer_starts': p['optimizer_starts'], 'rounds': 1},
                           'fold_seed': seed, 'validation_observed': False},
              'folds': folds and [list(map(int, f)) for f in folds],
              'rows': rows,
              'mean_ratio_local_slsqp': float(ratios.mean()),
              'se_ratio': float(ratios.std(ddof=1) / np.sqrt(len(ratios))),
              'learned_kernel_retraining': 'NOT_RUN (GPU-side; fixed weights would leak)'}
    (OUT / 'fold_plan.json').write_text(json.dumps(result, indent=2) + '\n')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(rows)), [r['ratio_local_slsqp'] for r in rows])
    ax.set(xlabel='outer fold', ylabel='local / SLSQP test risk', title='nested CV (5 folds)')
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / 'nested_cv.png', dpi=150)
    print('mean ratio %.2f +- %.2f' % (ratios.mean(), result['se_ratio']))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
