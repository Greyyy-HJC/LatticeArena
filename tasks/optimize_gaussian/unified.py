"""Unified statistics entry for the frozen linear candidates (todo P1).

Recomputes the point risks from the frozen v1 data with one code path and then
does a fold-local evaluation (reference refitted on each training fold). The
learned BCB kernel needs the collected response data and is reported as
archived/PARTIAL, not re-evaluated here.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import minimize

from tasks.pion_projection.reference import make_reference
from tasks.pion_projection.statistics import blocks, bootstrap_indices, fold, inverse_covariance, risk, shape

RUN = Path('results/pion_projection/s16t16_v1')
OUT = Path('results/optimize_gaussian/cpu_next_package_b')


def load():
    p = json.loads((RUN / 'development_provenance.json').read_text())['protocol']
    sub = json.loads((RUN / 'submission.json').read_text())
    dev = blocks(fold(np.load(RUN / 'development.npz')['correlators']), p['block_length'])
    val = blocks(fold(np.load(RUN / 'validation.npz')['correlators']), p['block_length'])
    return p, sub, dev, val


def evaluate(data, weights, p, mass, masses, seed, late_inverse=None):
    index = bootstrap_indices(len(data), p['bootstrap_samples'], seed)
    return risk(data, np.asarray(weights), mass, masses, index, p,
                late_inverse=np.asarray(late_inverse) if late_inverse is not None else None)


def risk_with_fixed_amplitude(data, weights, mass, masses, index, config, amplitude, amplitude_se,
                              amplitude_source='train', late_inverse=None, return_draws=False):
    """Same metric as statistics.risk but with the amplitude fitted on the training fold."""
    values = np.einsum('nbt,b->nt', data, np.asarray(weights, dtype=float))
    mean = values.mean(axis=0)
    late, early = config['amplitude_window'], config['score_window']
    f = shape(mass, np.arange(values.shape[-1]), config['geometry'][3])
    boot_mean = values[index].mean(axis=1)
    fb = shape(masses, np.arange(values.shape[-1]), config['geometry'][3])
    if amplitude_source == 'train':
        # fixed training amplitude: point and replicas use the same value
        amp = float(amplitude)
        amp_se = float(amplitude_se)
        amps = np.full(len(boot_mean), amp)
        late_source = 'train'
    elif amplitude_source == 'test':
        # amplitude re-estimated on the test fold with the same algorithm in point and replicas
        inv = late_inverse if late_inverse is not None else inverse_covariance(
            values[:, late], config['covariance_shrinkage'])
        amp = float(np.einsum('i,ij,j->', f[late], inv, mean[late]) / np.einsum('i,ij,j->', f[late], inv, f[late]))
        amps = (np.einsum('bi,ij,bj->b', fb[:, late], inv, boot_mean[:, late])
                / np.einsum('bi,ij,bj->b', fb[:, late], inv, fb[:, late]))
        amp_se = float('nan')
        late_source = 'test'
    else:
        raise ValueError(f'unknown amplitude_source {amplitude_source!r}')
    z = abs(amp) / amp_se
    q = mean[early] / (amp * f[early])
    qb = boot_mean[:, early] / (amps[:, None] * fb[:, early])
    variance = float(qb.var(axis=0, ddof=1).mean())
    bias = float(np.mean((q - 1) ** 2))
    se = boot_mean.std(axis=0, ddof=1)
    snr = np.divide(abs(mean), se, out=np.zeros_like(mean), where=se > 0)
    result = {'risk': variance + bias, 'variance': variance, 'departure_squared': bias,
            'amplitude': amp, 'amplitude_se': amp_se, 'amplitude_z': z,
            'amplitude_source': amplitude_source, 'late_covariance_source': late_source,
            'estimand': ('training-frozen amplitude' if amplitude_source == 'train'
                         else 'test-estimated amplitude'), 'q': q.tolist(),
            'q_se': qb.std(axis=0, ddof=1).tolist(), 'mean': mean.tolist(),
            'se': se.tolist(), 'snr': snr.tolist()}
    if return_draws:
        result['q_boot'] = qb.tolist()
    return result


def select_on_train(train, p, mass, masses, index):
    """Re-select the SLSQP mixture and the tuned Gaussian power inside a fold."""
    basis = np.eye(len(p['powers']))
    base = [risk(train, w, mass, masses, index, p) for w in basis]
    scales = np.array([r['amplitude'] for r in base])
    best = int(np.argmin([r['risk'] for r in base]))
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
        return None, best, {'outcomes': 0, 'starts': len(starts)}
    fit = min(outcomes, key=lambda f: f.fun)
    info = {'outcomes': len(outcomes),
            'starts': len([*basis, np.ones(len(basis)) / len(basis)][:p['optimizer_starts']]),
            'nfev': [int(f.nfev) for f in outcomes],
            'best_nfev': int(fit.nfev), 'best_success': bool(fit.success),
            'feasible': bool(abs(fit.x.sum() - 1) < 1e-6 and abs(fit.x).sum() <= p['optimizer_l1_bound'] + 1e-6),
            'budget': p['optimizer_max_iterations']}
    return fit.x / scales, best, info


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p, sub, dev, val = load()
    window = p['reference_windows'][p['reference_nominal_window']]
    ref = make_reference(dev, p, window, 'single')
    m0, masses = ref.bootstrap(bootstrap_indices(len(dev), p['bootstrap_samples'], p['seed']))
    rows = {}
    for name, weights in sub['submissions'].items():
        late = sub['late_inverse_covariances'][name]
        rows[name] = evaluate(val, weights, p, m0['mass'], masses, p['seed'] + 1, late)
    archived = json.loads((RUN / 'candidates' / '63c33e619e06704a' / 'compare.json').read_text())['point_risk']
    mapping = {'local': 'local', 'agent_signed_mixture': 'slsqp', 'tuned_gaussian': None}
    checks = {}
    for name in rows:
        key = mapping.get(name)
        checks[name] = {'computed': rows[name]['risk'],
                        'archived': archived.get(key) if key else None,
                        'delta': (abs(rows[name]['risk'] - archived[key]) if key and key in archived else None)}
    folds = np.array_split(np.random.default_rng(5).permutation(len(dev)), 3)
    fold_rows = []
    for i, test_ids in enumerate(folds):
        train = dev[np.setdiff1d(np.arange(len(dev)), test_ids)]
        test = dev[test_ids]
        fold_ref = make_reference(train, p, window, 'single')
        fold_m0, fold_masses = fold_ref.bootstrap(bootstrap_indices(len(train), p['bootstrap_samples'], p['seed'] + i))
        fold = {'fold': i, 'mass': float(fold_m0['mass']),
                'train_blocks': [int(x) for x in np.setdiff1d(np.arange(len(dev)), test_ids)],
                'test_blocks': [int(x) for x in test_ids],
                'estimand': 'training-frozen amplitude',
                'reference': {'spec_kind': (sub.get('reference') or {}).get('kind') or 'legacy',
                              'mass': float(fold_m0['mass']),
                              'masses_sha256': hashlib.sha256(np.asarray(fold_masses).tobytes()).hexdigest(),
                              'passed': bool(sub.get('calibration', {}).get('reference_passed', False)),
                              'source_split': 'development'}}
        for name, weights in sub['submissions'].items():
            train_metric = evaluate(train, weights, p, fold_m0['mass'], fold_masses, p['seed'] + i)
            fixed = (train_metric['amplitude'], train_metric['amplitude_se'])
            metric = risk_with_fixed_amplitude(test, weights, fold_m0['mass'], fold_masses,
                                               bootstrap_indices(len(test), p['bootstrap_samples'], p['seed'] + 100 + i),
                                               p, fixed[0], fixed[1])
            fold[name] = metric['risk']
            fold[f'{name}_detail'] = {'variance': metric['variance'],
                                      'departure_squared': metric['departure_squared'],
                                      'amplitude': metric['amplitude'],
                                      'amplitude_z': metric['amplitude_z'],
                                      'amplitude_source': metric['amplitude_source']}
        fold['local_over_slsqp'] = fold['local'] / fold['agent_signed_mixture']
        fold_rows.append(fold)
    search_rows = []
    for i, test_ids in enumerate(folds):
        train = dev[np.setdiff1d(np.arange(len(dev)), test_ids)]
        test = dev[test_ids]
        fold_ref = make_reference(train, p, window, 'single')
        fold_m0, fold_masses = fold_ref.bootstrap(bootstrap_indices(len(train), p['bootstrap_samples'], p['seed'] + i))
        weights, best_power, search_info = select_on_train(train, p, fold_m0['mass'], fold_masses,
                                                            bootstrap_indices(len(train), p['bootstrap_samples'], p['seed'] + i))
        train_amp = evaluate(train, weights, p, fold_m0['mass'], fold_masses, p['seed'] + i)['amplitude']
        train_amp_se = evaluate(train, weights, p, fold_m0['mass'], fold_masses, p['seed'] + i)['amplitude_se']
        basis = np.eye(len(p['powers']))
        local_fixed = (evaluate(train, basis[0], p, fold_m0['mass'], fold_masses, p['seed'] + i)['amplitude'],
                       evaluate(train, basis[0], p, fold_m0['mass'], fold_masses, p['seed'] + i)['amplitude_se'])
        tuned_fixed = (evaluate(train, basis[best_power], p, fold_m0['mass'], fold_masses, p['seed'] + i)['amplitude'],
                       evaluate(train, basis[best_power], p, fold_m0['mass'], fold_masses, p['seed'] + i)['amplitude_se'])
        idx_test = bootstrap_indices(len(test), p['bootstrap_samples'], p['seed'] + 100 + i)
        local_risk = risk_with_fixed_amplitude(test, basis[0], fold_m0['mass'], fold_masses, idx_test, p, *local_fixed)['risk']
        tuned_risk = risk_with_fixed_amplitude(test, basis[best_power], fold_m0['mass'], fold_masses, idx_test, p, *tuned_fixed)['risk']
        slsqp_risk = risk_with_fixed_amplitude(test, weights, fold_m0['mass'], fold_masses, idx_test, p, train_amp, train_amp_se)['risk']
        search_rows.append({'fold': i, 'mass': float(fold_m0['mass']), 'best_power': int(best_power),
                            'best_power_value': float(p['powers'][best_power]),
                            'local': local_risk, 'tuned': tuned_risk, 'slsqp': slsqp_risk,
                            'tuned_over_slsqp': tuned_risk / slsqp_risk,
                            'train_blocks': [int(x) for x in np.setdiff1d(np.arange(len(dev)), test_ids)],
                            'test_blocks': [int(x) for x in test_ids],
                            'weights_sha256': hashlib.sha256(np.asarray(weights).tobytes()).hexdigest(),
                            'estimand': 'training-frozen amplitude (outer-training fit, no inner CV)',
                            'reference': {'mass': float(fold_m0['mass']),
                                          'masses_sha256': hashlib.sha256(np.asarray(fold_masses).tobytes()).hexdigest(),
                                          'passed': bool(sub.get('calibration', {}).get('reference_passed', False))},
                            'search': search_info})
    ratios = np.array([f['local_over_slsqp'] for f in fold_rows])
    failures = {'folds': len(fold_rows), 'search_folds_without_outcome': 0, 'reference_failed_folds': 0}
    for row in search_rows:
        if row['search'].get('outcomes', 1) == 0:
            failures['search_folds_without_outcome'] += 1
        if not row['reference']['passed']:
            failures['reference_failed_folds'] += 1
    result = {'failures': failures,
              'point': {name: {k: rows[name][k] for k in ('risk', 'variance', 'departure_squared', 'amplitude', 'amplitude_z')}
                        for name in rows},
              'archived_point_check': checks,
              'fold_local': fold_rows,
              'fold_local_mean_ratio': float(ratios.mean()),
              'fold_local_se_ratio': float(ratios.std(ddof=1) / np.sqrt(len(ratios))),
              'learned_bcb_kernel': 'archived only (needs the collected response data); not re-evaluated',
              'fold_local_search': search_rows,
              'fold_local_search_note': 'hyperparameters re-selected inside each outer training fold; '
                                        'the learned BCB kernel needs the collected responses and is NOT_RUN',
              'gate': 'reference not certified -> diagnostic_only'}
    (OUT / 'unified.json').write_text(json.dumps(result, indent=2) + '\n')
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    for j, name in enumerate(('local', 'agent_signed_mixture')):
        var = [f[f'{name}_detail']['variance'] for f in fold_rows]
        dep = [f[f'{name}_detail']['departure_squared'] for f in fold_rows]
        x = np.arange(len(fold_rows)) + (j - 0.5) * width
        ax.bar(x, var, width, label=f'{name} variance')
        ax.bar(x, dep, width, bottom=var, label=f'{name} departure')
    ax.set_xticks(range(len(fold_rows)), [f'fold {f["fold"]}' for f in fold_rows])
    ax.set(ylabel='risk', title='fold-local risk decomposition (outer-training amplitude)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_risk_decomposition.png', dpi=150)
    for tag, rows_ in (('frozen', fold_rows), ('reselected', search_rows)):
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        labels = [f'fold {row["fold"]}' for row in rows_]
        if tag == 'frozen':
            values = [row['local_over_slsqp'] for row in rows_]
            ax2.set(title='frozen-weight fold ratios (local/SLSQP)',
                    ylabel='local / SLSQP risk')
        else:
            values = [row['tuned_over_slsqp'] for row in rows_]
            ax2.set(title='fold-reselected tuned/SLSQP ratios', ylabel='tuned / SLSQP risk')
        ax2.bar(range(len(values)), values, color='tab:blue' if tag == 'frozen' else 'tab:orange')
        ax2.axhline(1.0, color='k', ls='--', lw=0.8)
        ax2.set_xticks(range(len(values)), labels)
        fig2.tight_layout()
        fig2.savefig(OUT / f'fig_folds_{tag}.png', dpi=150)
    print(json.dumps({'point': {n: round(v['risk'], 6) for n, v in rows.items()},
                      'archived_check': {n: (round(c['delta'], 8) if c['delta'] is not None else None)
                                         for n, c in checks.items()},
                      'fold_ratio': (round(ratios.mean(), 2), round(result['fold_local_se_ratio'], 2))}, indent=2))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
