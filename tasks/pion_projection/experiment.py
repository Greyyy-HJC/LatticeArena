"""Development-only optimization followed by frozen held-out evaluation."""
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from core.state import RunState

from core.io import digest_text

from .common import digest, fingerprint, load_data, protocol, write_json
from .identity import verify_run_identity
from .reference import make_reference, reference_kind, reference_spec, verify_reference
from .statistics import blocks, bootstrap_indices, fold, inverse_covariance, risk


def reflection_odd(raw):
    """C(t) - C(T-t) for t=1..T/2-1, valid for any even temporal extent."""
    temporal = raw.shape[-1]
    if temporal % 2:
        raise ValueError('Reflection check needs an even temporal extent')
    t = np.arange(1, temporal // 2)
    return raw[..., t] - raw[..., temporal - t]


def late_metric(data, weights, p):
    return inverse_covariance(np.einsum('nbt,b->nt', data, weights)[:, p['amplitude_window']],
                              p['covariance_shrinkage'])


def develop(output):
    output = Path(output)
    if (output / 'submission.json').exists() or (output / 'validation.npz').exists():
        raise ValueError('Development is closed for this run; use a new run and disclose prior validation access')
    p = protocol()
    spec = reference_spec()
    raw = load_data(output, 'development')
    data = blocks(fold(raw), p['block_length'])
    index = bootstrap_indices(len(data), p['bootstrap_samples'], p['seed'])
    windows, masses = [], []
    for window in p['reference_windows']:
        model = make_reference(data, p, window, spec['kind'])
        result, boot = model.bootstrap(index)
        windows.append(result)
        masses.append(boot)
    nominal = p['reference_nominal_window']
    reference = windows[nominal]
    m, mb = reference['mass'], masses[nominal]
    failures = []
    if reference['se']/m > p['reference_max_relative_error']:
        failures.append('Reference mass relative uncertainty exceeds 5%')
    if reference['p_approx'] < p['reference_min_fit_p']:
        failures.append('Nominal correlated reference fit fails the approximate goodness-of-fit gate')
    for j, window in enumerate(windows):
        if j != nominal:
            se = float((masses[j]-mb).std(ddof=1))
            z = abs(window['mass']-m)/se if se > 0 else 1e300
            window['paired_shift_z'] = z
            if z > p['reference_max_window_z']:
                failures.append(f'Reference window {window["window"]} shifts by {z:.2f} paired standard errors')
    if np.any(mb <= p['mass_bounds'][0]+1e-4) or np.any(mb >= p['mass_bounds'][1]-1e-4):
        failures.append('Reference bootstrap reaches mass bounds')
    odd = reflection_odd(raw)
    odd_blocks = blocks(odd, p['block_length'])
    odd_se = odd_blocks.std(axis=0, ddof=1)/np.sqrt(len(odd_blocks))
    odd_mean = odd_blocks.mean(axis=0)
    z = np.divide(np.abs(odd_mean), odd_se, out=np.zeros_like(odd_mean), where=odd_se > 0)
    reflection_z = float(np.max(z))
    if reflection_z > p['reflection_max_z']:
        failures.append('Forward/backward symmetry check exceeds the fixed z threshold')
    sensitivity = {}
    for length in [1, 2, 3, 5]:
        bs = blocks(fold(raw), length)
        sensitivity[str(length)] = (bs.std(axis=0, ddof=1)/np.sqrt(len(bs)))[:, 6].tolist()
    calibration = {'reference': reference, 'windows': windows, 'reference_passed': not failures,
                   'failure_reasons': failures, 'reflection_max_z': reflection_z,
                   'block_length_sensitivity_C6_se': sensitivity,
                   'independence_status': 'Trajectory spacing unknown; five-configuration blocks are a conservative pilot choice, not established decorrelation.',
                   'n_development_blocks': len(data)}
    print('CALIBRATION', json.dumps(calibration), flush=True)
    basis = np.eye(len(p['powers']))
    baseline_results = [risk(data, w, m, mb, index, p) for w in basis]
    best = min(range(len(basis)), key=lambda j: baseline_results[j]['risk'])
    scales = np.array([r['amplitude'] for r in baseline_results])
    normalized = data/scales[None, :, None]

    def objective(u):
        try:
            return risk(normalized, u, m, mb, index, p)['risk'] + p['optimizer_ridge']*float(u @ u)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return 1e10

    starts = [*basis, np.ones(len(basis))/len(basis)]
    outcomes = []
    for start in starts[:p['optimizer_starts']]:
        fit = minimize(objective, start, method='SLSQP', bounds=[(-p['optimizer_l1_bound'], p['optimizer_l1_bound'])]*len(basis),
                       constraints=[{'type': 'eq', 'fun': lambda u: u.sum()-1},
                                    {'type': 'ineq', 'fun': lambda u: p['optimizer_l1_bound']-abs(u).sum()}],
                       options={'maxiter': p['optimizer_max_iterations'], 'ftol': 1e-11})
        feasible = abs(fit.x.sum()-1) < 1e-6 and abs(fit.x).sum() <= p['optimizer_l1_bound']+1e-6
        if fit.success and feasible and fit.fun < 1e9:
            outcomes.append(fit)
    if not outcomes:
        raise ValueError('No optimizer start converged to a valid candidate')
    fit = min(outcomes, key=lambda f: f.fun)
    weights = fit.x/scales
    candidate_result = risk(data, weights, m, mb, index, p)
    submissions = {'local': basis[0].tolist(), 'tuned_gaussian': basis[best].tolist(),
                   'agent_signed_mixture': weights.tolist()}
    metrics = {name: late_metric(data, w, p).tolist() for name, w in submissions.items()}
    state = RunState()
    tests_path = Path(__file__).resolve().parents[2] / 'tests' / 'test_pion_projection.py'
    diagnostics = Path('results/pion_projection/score_diagnostics_v3/diagnostics.json')
    state.set('operator', 'draft', 'kernels.py')
    state.set('operator', 'validated', 'tests',
              evidence_hash=digest(tests_path) if tests_path.exists() else None)
    state.set('numerical', 'draft', 'condition numbers / floor diagnostics')
    state.set('numerical', 'validated', 'score_diagnostics_v3',
              evidence_hash=digest(diagnostics) if diagnostics.exists() else None)
    calibration_hash = digest_text(json.dumps(calibration, sort_keys=True))
    state.set('reference', 'draft', 'single-cosh calibration')
    if calibration['reference_passed']:
        state.set('reference', 'validated', 'reference gate passed', evidence_hash=calibration_hash)
    else:
        state.set('reference', 'failed', '; '.join(calibration['failure_reasons']))
    state.set('statistical', 'draft', 'block bootstrap')
    state.set('statistical', 'validated', f'{len(data)} blocks',
              evidence_hash=digest(output / 'development.npz'))
    state.set('comparison', 'draft', 'risk ratios')
    if not calibration['reference_passed']:
        state.set('comparison', 'failed', 'reference gate failed')
    frozen = {'protocol_hash': fingerprint(p), 'development_sha256': digest(output/'development.npz'),
              'statistics_sha256': digest(Path(__file__).with_name('statistics.py')),
              'kernel_sha256': digest(Path(__file__).with_name('kernels.py')),
              'powers': p['powers'], 'alpha': p['alpha'], 'submissions': submissions,
              'normalized_mixture_weights': fit.x.tolist(), 'basis_amplitude_scales': scales.tolist(),
              'late_inverse_covariances': metrics, 'selected_gaussian_power': p['powers'][best],
              'optimizer': {'converged_starts': len(outcomes), 'objective': float(fit.fun),
                            'iterations': int(fit.nit), 'l1': float(abs(fit.x).sum())},
              'calibration': calibration, 'reference': spec, 'run_state': state.summary(),
              'gate_artifact': {
                  'model': spec['kind'],
                  'window': p['reference_windows'][p['reference_nominal_window']],
                  'prior': {'mass_bounds': p['mass_bounds'], 'amplitude_z_min': p['amplitude_z_min'],
                            'min_fit_p': p['reference_min_fit_p']},
                  'reference_source': {'statistics': digest(Path(__file__).with_name('statistics.py')),
                                       'implementation': digest(Path(__file__).with_name('reference.py'))},
                  'development_sha256': digest(output / 'development.npz'),
                  'candidates': list(submissions),
                  'metric': {'score_window': p['score_window'], 'amplitude_window': p['amplitude_window'],
                             'covariance_shrinkage': p['covariance_shrinkage']},
                  'protocol_hash': fingerprint(p),
                  'calibration': {'passed': calibration['reference_passed'],
                                  'failures': calibration['failure_reasons'],
                                  'chi2': calibration['reference']['chi2'],
                                  'dof': 8 if spec['kind'] == 'single' else None}}}
    write_json(output/'development_results.json', {'basis': baseline_results, 'agent': candidate_result})
    write_json(output/'submission.json', frozen)
    np.savez_compressed(output/'reference_bootstrap.npz', masses=mb, indices=index)
    print('FROZEN_SUBMISSION', json.dumps(frozen), flush=True)


def nested_intervals(development, validation, frozen, p, kind=None):
    """Resample the entire conditional risk statistic, including its variance.

    Fit metrics are frozen at development, not re-estimated on validation or
    inside resamples. Shared reference and paired validation draws preserve
    correlations between all candidates. No replicate is silently discarded.
    """
    rng = np.random.default_rng(p['seed']+2)
    model = make_reference(development, p, p['reference_windows'][p['reference_nominal_window']], kind)
    names = list(frozen['submissions'])
    draws = []
    for outer in range(p['outer_samples']):
        rd = rng.integers(0, len(development), len(development))
        rv = rng.integers(0, len(validation), len(validation))
        ref_samples = model.samples[rd]
        mass = model.fit(ref_samples.mean(axis=0))[0]
        di = rng.integers(0, len(development), (p['inner_samples'], len(development)))
        vi = rng.integers(0, len(validation), (p['inner_samples'], len(validation)))
        masses = np.array([model.fit(mean)[0] for mean in ref_samples[di].mean(axis=1)])
        losses = []
        for name in names:
            value = risk(validation[rv], frozen['submissions'][name], mass, masses, vi, p,
                         require_valid=False, late_inverse=np.array(frozen['late_inverse_covariances'][name]))
            losses.append(value['risk'])
        draws.append(losses)
        if (outer+1) % 50 == 0:
            print(f'Nested bootstrap {outer+1}/{p["outer_samples"]}', flush=True)
    return np.asarray(draws), names


def evaluate(output):
    output = Path(output)
    p = protocol()
    frozen = json.loads((output/'submission.json').read_text())
    if frozen['protocol_hash'] != fingerprint(p) or frozen['development_sha256'] != digest(output/'development.npz'):
        raise ValueError('Frozen protocol or development data changed')
    if frozen['statistics_sha256'] != digest(Path(__file__).with_name('statistics.py')) or frozen['kernel_sha256'] != digest(Path(__file__).with_name('kernels.py')):
        raise ValueError('Frozen scoring/kernel implementation changed')
    spec = frozen.get('reference')
    if spec is None:
        if os.environ.get('LATTICEARENA_LEGACY_REFERENCE') != '1':
            raise ValueError('Frozen submission has no reference spec; set LATTICEARENA_LEGACY_REFERENCE=1 for a read-only legacy replay')
        kind = reference_kind()
    else:
        kind = verify_reference(spec)
    dev = blocks(fold(load_data(output, 'development')), p['block_length'])
    raw = load_data(output, 'validation')
    val = blocks(fold(raw), p['block_length'])
    model = make_reference(dev, p, p['reference_windows'][p['reference_nominal_window']], kind)
    reference, mb = model.bootstrap(bootstrap_indices(len(dev), p['bootstrap_samples'], p['seed']))
    vi = bootstrap_indices(len(val), p['bootstrap_samples'], p['seed']+1)
    state = RunState()
    frozen_state = frozen.get('run_state', {})
    for name in ('operator', 'reference', 'numerical', 'statistical'):
        state.status[name] = frozen_state.get('status', {}).get(name, 'draft')
    state.set('comparison', 'draft', 'evaluate')
    state.observe_validation(f'evaluate:{p["version"]}')
    if not frozen['calibration']['reference_passed']:
        state.set('comparison', 'failed', 'reference gate failed')
    kind_checked, identity_reasons = verify_run_identity(output, (kind,))
    certification = {'status': 'abstain', 'reasons': identity_reasons}
    certification_source = 'identity_check'
    joint_dir = output / 'joint'
    if not identity_reasons and (joint_dir / 'joint_manifest.json').exists() and (joint_dir / 'metric_results.json').exists():
        joint_manifest = json.loads((joint_dir / 'joint_manifest.json').read_text())
        compatible = (joint_manifest.get('protocol_hash') == fingerprint(p)
                      and joint_manifest.get('development_sha256') == digest(output / 'development.npz')
                      and joint_manifest.get('validation_sha256') == digest(output / 'validation.npz')
                      and joint_manifest.get('reference_kind') == kind)
        if compatible:
            certification = json.loads((joint_dir / 'metric_results.json').read_text())
            certification_source = 'joint'
        else:
            certification = {'status': 'abstain',
                             'reasons': ['joint manifest does not match this run identity']}
            certification_source = 'joint_rejected'
    elif not identity_reasons:
        certification = {'status': 'abstain',
                         'reasons': ['evaluate has no joint draws; run the joint command']}
        certification_source = 'no_joint'
    report = {'protocol': p, 'submission_sha256': digest(output/'submission.json'),
              'run_state': state.summary(), 'physical_score': state.score_or_null(),
              'certification': certification, 'certification_source': certification_source,
              'validation_sha256': digest(output/'validation.npz'), 'reference': reference,
              'reference_passed': frozen['calibration']['reference_passed'],
              'reference_failure_reasons': frozen['calibration']['failure_reasons'],
              'reference_spec': spec, 'reference_kind': kind,
              'n_validation_configurations': len(raw), 'n_validation_blocks': len(val),
              'results': {}, 'score_status': 'pilot_only' if frozen['calibration']['reference_passed'] else 'unscored_reference_failed'}
    for name, weights in frozen['submissions'].items():
        try:
            report['results'][name] = risk(val, weights, reference['mass'], mb, vi, p,
                                           late_inverse=np.array(frozen['late_inverse_covariances'][name]))
        except ValueError as exc:
            report['results'][name] = {'invalid_reason': str(exc), 'score': None}
    valid = all('risk' in value for value in report['results'].values())
    if valid:
        eps = p['score_epsilon']
        baseline = report['results']['local']['risk']
        for value in report['results'].values():
            value['diagnostic_loss_ratio_vs_local'] = (baseline+eps)/(value['risk']+eps)
            value['score'] = 100*value['diagnostic_loss_ratio_vs_local'] if report['reference_passed'] else None
        draws, names = nested_intervals(dev, val, frozen, p, kind)
        np.savez_compressed(output/'nested_bootstrap.npz', risks=draws, names=np.array(names))
        for j, name in enumerate(names):
            ratios = (draws[:, 0]+eps)/(draws[:, j]+eps)
            report['results'][name]['diagnostic_loss_ratio_95_interval'] = np.quantile(ratios, [.025, .975]).tolist()
            report['results'][name]['risk_95_interval'] = np.quantile(draws[:, j], [.025, .975]).tolist()
        j, k = names.index('agent_signed_mixture'), names.index('tuned_gaussian')
        report['agent_vs_tuned_gaussian'] = {
            'risk_ratio': report['results']['tuned_gaussian']['risk']/report['results']['agent_signed_mixture']['risk'],
            'risk_ratio_95_interval': np.quantile((draws[:, k]+eps)/(draws[:, j]+eps), [.025, .975]).tolist()}
    report['conditional_scores'] = {name: value.get('score') for name, value in report['results'].items()}
    write_json(output/'validation_results.json', report)
    print('VALIDATION', json.dumps(report), flush=True)
