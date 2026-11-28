"""Development-only optimization followed by frozen held-out evaluation."""
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from core.receipts import make as make_receipt
from core.state import RunState

from core.io import digest_text

from .common import digest, fingerprint, load_data, protocol, write_json
from .identity import verify_run_identity
from .reference import make_reference, reference_kind, reference_spec, verify_reference
from .statistics import blocks, bootstrap_indices, fold, inverse_covariance, risk

DEFAULT_SCORE_ALPHA = 0.05


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


def make_score_spec(p, candidate_count):
    """Freeze statistical confidence separately from the smear-kernel alpha."""
    score_alpha = float(p.get('score_alpha', DEFAULT_SCORE_ALPHA))
    if not 0 < score_alpha < 1:
        raise ValueError('score_alpha must be in (0, 1)')
    return {
        'alpha': score_alpha,
        'comparison_family': {'candidates': candidate_count, 'models': 1},
        'min_baseline_risk': 1e-6,
        'min_improvement': 1.0,
        'risk_floor': 0.0,
    }


def develop(output):
    output = Path(output)
    if any((output / f'{split}.npz').exists() for split in ('anchor', 'validation')):
        raise ValueError('Development is closed after a holdout measurement; use a new run and disclose prior access')
    if (output / 'submission.json').exists():
        raise ValueError('Development is already frozen for this run; use the existing submission')
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
    for length in [1, 2, 4, 5, 10, 20, 25]:
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
    fixed = p.get('synthetic_fixed_candidates')
    if fixed is not None:
        if fixed != {'baseline_channel': 1, 'candidate_channel': 2} or not p.get('synthetic'):
            raise ValueError('fixed candidate mode requires explicit synthetic channel1/channel2 spec')
        from types import SimpleNamespace
        weights = basis[2]
        fit = SimpleNamespace(x=basis[2]*scales, fun=0., nit=0)
        outcomes = []
        candidate_result = risk(data, weights, m, mb, index, p)
        submissions = {'local': basis[1].tolist(), 'fixed_candidate': weights.tolist()}
    else:
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
    code = {'kernels.py': digest(Path(__file__).with_name('kernels.py')),
            'statistics.py': digest(Path(__file__).with_name('statistics.py'))}
    inputs = {'development.npz': digest(output / 'development.npz')}
    receipts = {
        'operator': make_receipt('operator', code, inputs,
                                 {'submissions': list(submissions),
                                  'finite_late_metrics': bool(all(np.isfinite(v).all()
                                                                  for v in metrics.values()))}),
        'numerical': make_receipt('numerical', code, inputs,
                                  {'condition_ok': bool(np.all(np.isfinite(candidate_result['q']))) }),
        'reference': make_receipt('reference', code, inputs,
                                  {'passed': calibration['reference_passed'],
                                   'failure_reasons': calibration['failure_reasons'],
                                   'mass': calibration['reference']['mass'],
                                   'chi2': calibration['reference']['chi2']}),
        'statistical': make_receipt('statistical', code, inputs,
                                    {'seed': p['seed'], 'blocks': len(data),
                                     'bootstrap_samples': p['bootstrap_samples'],
                                     'candidate_risk': candidate_result['risk'],
                                     'late_metric_shapes': [list(np.asarray(v).shape)
                                                            for v in metrics.values()]}),
    }
    state.set('operator', 'draft', 'kernels.py')
    state.set('operator', 'validated', 'kernel receipt', receipt=receipts['operator'])
    state.set('numerical', 'draft', 'condition numbers / floor diagnostics')
    state.set('numerical', 'validated', 'numerical receipt', receipt=receipts['numerical'])
    state.set('reference', 'draft', 'single-cosh calibration')
    if calibration['reference_passed']:
        state.set('reference', 'validated', 'reference receipt', receipt=receipts['reference'])
    else:
        state.set('reference', 'failed', '; '.join(calibration['failure_reasons']))
    state.set('statistical', 'draft', 'block bootstrap')
    state.set('statistical', 'validated', 'statistical receipt', receipt=receipts['statistical'])
    state.set('comparison', 'draft', 'risk ratios')
    if not calibration['reference_passed']:
        state.set('comparison', 'failed', 'reference gate failed')
    frozen = {'protocol_hash': fingerprint(p), 'development_sha256': digest(output/'development.npz'),
              'statistics_sha256': digest(Path(__file__).with_name('statistics.py')),
              'kernel_sha256': digest(Path(__file__).with_name('kernels.py')),
              'powers': p['powers'], 'smear_alpha': p['alpha'], 'submissions': submissions,
              'synthetic_fixed_candidates': fixed, 'synthetic': bool(p.get('synthetic')),
              'normalized_mixture_weights': fit.x.tolist(), 'basis_amplitude_scales': scales.tolist(),
              'late_inverse_covariances': metrics, 'selected_gaussian_power': p['powers'][best],
              'optimizer': {'converged_starts': len(outcomes), 'objective': float(fit.fun),
                            'iterations': int(fit.nit), 'l1': float(abs(fit.x).sum())},
              'calibration': calibration, 'reference': spec, 'run_state': state.summary(),
              'receipts': receipts,
              'score_spec': make_score_spec(p, len(submissions) - 1),
              'gate_artifact': {
                  'model': spec['kind'],
                  'window': p['reference_windows'][p['reference_nominal_window']],
                  'prior': {'mass_bounds': p['mass_bounds'], 'amplitude_z_min': p['amplitude_z_min'],
                            'min_fit_p': p['reference_min_fit_p']},
                  'reference_source': {'statistics': digest(Path(__file__).with_name('statistics.py')),
                                       'implementation': digest(Path(__file__).with_name('reference.py'))},
                  'reference_spec_sha256': digest_text(json.dumps(spec, sort_keys=True)),
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


def nested_intervals(development, validation, frozen, p, kind=None, seed_offset=0):
    """Resample the entire conditional risk statistic, including its variance.

    Fit metrics are frozen at development, not re-estimated on validation or
    inside resamples. Shared reference and paired validation draws preserve
    correlations between all candidates. No replicate is silently discarded.
    """
    rng = np.random.default_rng(p['seed']+2+seed_offset)
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
        from .identity import verify_run_identity
        identity_kind, identity_reasons = verify_run_identity(output)
        if identity_kind != kind or identity_reasons:
            raise ValueError('Run identity check failed: ' + '; '.join(identity_reasons))
    dev = blocks(fold(load_data(output, 'development')), p['block_length'])
    raw = load_data(output, 'validation')
    val = blocks(fold(raw), p['block_length'])
    model = make_reference(dev, p, p['reference_windows'][p['reference_nominal_window']], kind)
    reference, mb = model.bootstrap(bootstrap_indices(len(dev), p['bootstrap_samples'], p['seed']))
    vi = bootstrap_indices(len(val), p['bootstrap_samples'], p['seed']+1)
    state = RunState()
    frozen_state = frozen.get('run_state', {})
    frozen_receipts = frozen.get('receipts') or {}
    receipt_status = {}
    for name in ('operator', 'reference', 'numerical', 'statistical'):
        status = frozen_state.get('status', {}).get(name, 'draft')
        receipt = frozen_receipts.get(name)
        if status == 'validated' and receipt is not None:
            from core.receipts import verify as verify_receipt
            try:
                verify_receipt(receipt)
            except ValueError as exc:
                receipt_status[name] = f'rejected: {exc}'
                state.set(name, 'draft')
            else:
                receipt_status[name] = 'verified'
                state.set(name, 'draft')
                state.set(name, 'validated', receipt=receipt)
        elif status == 'failed':
            state.set(name, 'failed')
        else:
            receipt_status[name] = 'missing receipt'
            state.set(name, 'draft')
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
              'robust_score_diagnostic': certification.get('robust_score_diagnostic',
                                                           {'status': 'unavailable', 'physical_score': None}),
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
        if 'agent_signed_mixture' in names and 'tuned_gaussian' in names:
            j, k = names.index('agent_signed_mixture'), names.index('tuned_gaussian')
            report['agent_vs_tuned_gaussian'] = {
                'risk_ratio': report['results']['tuned_gaussian']['risk']/report['results']['agent_signed_mixture']['risk'],
                'risk_ratio_95_interval': np.quantile((draws[:, k]+eps)/(draws[:, j]+eps), [.025, .975]).tolist()}
    report['conditional_scores'] = {name: value.get('score') for name, value in report['results'].items()}
    write_json(output/'validation_results.json', report)
    print('VALIDATION', json.dumps(report), flush=True)


def run_with_provenance(output, plan_path):
    """Run development analysis as a separately hashed amendment to a frozen measurement."""
    import hashlib
    import json
    from pathlib import Path

    output = Path(output)
    plan_path = Path(plan_path)
    p = protocol()
    measurement = output / 'development.npz'
    if not measurement.is_file():
        raise FileNotFoundError(measurement)
    with np.load(measurement, allow_pickle=False) as saved:
        generation = str(saved['generation'])
        protocol_hash = str(saved['protocol_hash'])
    amended_source = Path(__file__)
    base_source = amended_source.with_name('experiment.py')
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    inputs = {
        'development_npz_sha256': sha(measurement),
        'measurement_generation': generation,
        'measurement_protocol_hash': protocol_hash,
        'measurement_plan_sha256': sha(plan_path),
        'active_experiment_py_sha256': sha(base_source),
        'analysis_amendment_sha256': sha(amended_source),
    }
    if protocol_hash != fingerprint(p):
        raise ValueError('saved measurement protocol hash differs from selected protocol')
    develop(output)
    outputs = {}
    for name in ('submission.json', 'development_results.json', 'reference_bootstrap.npz'):
        path = output / name
        if not path.is_file():
            raise RuntimeError(f'analysis output missing: {name}')
        outputs[name] = sha(path)
    frozen = json.loads((output / 'submission.json').read_text())
    calibration = frozen['calibration']
    provenance = {
        'schema': 'pion-development-analysis-amendment-v1',
        'status': 'COMPLETED_DIAGNOSTIC_ONLY' if not calibration['reference_passed'] else 'COMPLETED_REFERENCE_GATE_PASSED',
        'inputs': inputs,
        'outputs_sha256': outputs,
        'sensitivity_block_lengths': [1, 2, 4, 5, 10, 20, 25],
        'omitted_block_length': {'length': 3, 'reason': '100 development configurations are not divisible by 3; complete equal-size blocks are required and rows are not dropped'},
        'scope': 'post-measurement analysis amendment; no measurement source or raw data changed',
        'reference_gate_passed': bool(calibration['reference_passed']),
        'reference_failure_reasons': calibration['failure_reasons'],
    }
    (output / 'analysis_provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    write_diagnostic(output, p, frozen)
    return provenance

def write_diagnostic(output, p, frozen):
    """Create a compact plot and text report from frozen development outputs."""
    import hashlib
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    output = Path(output)
    calibration = frozen['calibration']
    windows = calibration['windows']
    sensitivity = calibration['block_length_sensitivity_C6_se']
    lengths = sorted(map(int, sensitivity))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    x = [item['window'][0] for item in windows]
    y = [item['mass'] for item in windows]
    err = [item['se'] for item in windows]
    axes[0].errorbar(x, y, yerr=err, marker='o', capsize=4, color='#087F8C')
    nominal = p['reference_windows'][p['reference_nominal_window']][0]
    axes[0].axvline(nominal, color='#555555', linestyle='--', linewidth=1, label='nominal window')
    axes[0].set(xlabel=r'$t_{\min}/a$', ylabel=r'$a m_0$', title='Reference mass by fit window')
    axes[0].grid(alpha=.25)
    axes[0].legend()
    values = np.array([sensitivity[str(length)] for length in lengths])
    # C6 denotes the folded correlator at temporal separation t=6; the band spans sink powers.
    axes[1].plot(lengths, values.mean(axis=1), marker='o', color='#D28B26')
    axes[1].fill_between(lengths, values.min(axis=1), values.max(axis=1), color='#D28B26', alpha=.2, label='range across sink powers')
    axes[1].set(xlabel='block length (configurations)', ylabel='SE of folded C(t=6)',
                title='Block-length sensitivity (complete partitions only)', xscale='log')
    axes[1].grid(alpha=.25)
    axes[1].legend(fontsize=8)
    fig.suptitle('S16T16D development: diagnostic only; reference gate ' + ('PASS' if calibration['reference_passed'] else 'FAIL'))
    image = output / 'development_diagnostics.png'
    fig.savefig(image, dpi=180)
    plt.close(fig)
    report = [
        '# Pion S16T16D development analysis amendment',
        '',
        f"Status: **{ 'reference gate passed' if calibration['reference_passed'] else 'diagnostic only; reference gate failed' }**.",
        '',
        'The 100-config development measurement was preserved unchanged. The frozen analysis stopped because its block-length sensitivity loop required length 3, which cannot partition 100 rows into complete equal-size blocks. This amendment records that limitation and uses only complete partitions at lengths 1, 2, 4, 5, 10, 20, and 25. No row was dropped.',
        '',
        '![Reference-window and block-length diagnostics](development_diagnostics.png)',
        '',
        '## Reference gate',
        '',
        f"- Nominal fit: mass `{calibration['reference']['mass']:.8g}`, bootstrap SE `{calibration['reference']['se']:.3g}`, approximate fit p-value `{calibration['reference']['p_approx']:.4g}`.",
        f"- Maximum reflection-odd z: `{calibration['reflection_max_z']:.4g}` (limit `{p['reflection_max_z']}`).",
        f"- Failure reasons: {', '.join(calibration['failure_reasons']) if calibration['failure_reasons'] else 'none'}.",
        '- The figure shows fitted mass and bootstrap uncertainty across reference windows, plus the C(t=6) standard-error sensitivity. These are diagnostics from this chain, not an independent anchor or a physical score.',
        '',
        '## Measurement integrity',
        '',
        f"- Configurations: `{len(np.load(output / 'development.npz', allow_pickle=False)['ids'])}`; measurement generation: `{np.load(output / 'development.npz', allow_pickle=False)['generation'].item()}`.",
        '- See `analysis_provenance.json` for input, source, and output hashes.',
        '- The E-chain anchor and D validation/shared holdout were not opened by this amendment.',
        '',
    ]
    (output / 'README.md').write_text('\n'.join(report))
    provenance_path = output / 'analysis_provenance.json'
    provenance = json.loads(provenance_path.read_text())
    provenance['diagnostics'] = {
        'image': image.name,
        'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
        'report': 'README.md',
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
