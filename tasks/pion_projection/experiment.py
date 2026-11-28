"""Development-only optimization followed by frozen held-out evaluation."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from core.candidates import validate_weights
from core.source_integrity import SourceByteSnapshot

from .common import digest, fingerprint, load_data, protocol, write_json
from .statistics import (ReferenceFit, blocks, bootstrap_indices, fold, inverse_covariance, risk)
from .spec import bundle_for, task_spec_for


def late_metric(data, weights, p):
    return inverse_covariance(np.einsum('nbt,b->nt', data, weights)[:, p['amplitude_window']],
                              p['covariance_shrinkage'])


def optimize_mixture(data, scales, m, mb, index, p):
    normalized = data/scales[None, :, None]

    def objective(u):
        try:
            return risk(normalized, u, m, mb, index, p)['risk'] + p['optimizer_ridge']*float(u @ u)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return 1e10

    basis = np.eye(len(scales))
    starts = [*basis, np.ones(len(basis))/len(basis)]
    outcomes = []
    evaluations = 0
    for start in starts[:p['optimizer_starts']]:
        fit = minimize(objective, start, method='SLSQP', bounds=[(-p['optimizer_l1_bound'], p['optimizer_l1_bound'])]*len(basis),
                       constraints=[{'type': 'eq', 'fun': lambda u: u.sum()-1},
                                    {'type': 'ineq', 'fun': lambda u: p['optimizer_l1_bound']-abs(u).sum()}],
                       options={'maxiter': p['optimizer_max_iterations'], 'ftol': 1e-11})
        evaluations += int(fit.nfev)
        feasible, _ = validate_weights(fit.x, len(basis), p['optimizer_l1_bound'])
        if fit.success and feasible and fit.fun < 1e9:
            outcomes.append(fit)
    if not outcomes:
        raise ValueError('No optimizer start converged to a valid candidate')
    fit = min(outcomes, key=lambda f: f.fun)
    return fit, len(outcomes), evaluations


def develop(output, *, source_snapshot=None, source_names=None):
    output = Path(output)
    if source_snapshot is not None:
        if not isinstance(source_snapshot, SourceByteSnapshot):
            raise ValueError('source_snapshot must be a verified SourceByteSnapshot')
        if (not isinstance(source_names, dict)
                or set(source_names) != {
                    'development_provenance', 'development_measurements'
                }):
            raise ValueError(
                'source_names must map development_provenance and '
                'development_measurements only'
            )
        if output.exists():
            raise FileExistsError('snapshot development output must use a new directory')
        output_abs, snapshot_root = output.resolve(), Path(source_snapshot.root).resolve()
        if (output_abs == snapshot_root or output_abs in snapshot_root.parents
                or snapshot_root in output_abs.parents):
            raise ValueError('snapshot development output must be outside the source tree')
        captured_provenance = json.loads(
            source_snapshot.read_bytes(source_names['development_provenance'])
        )
        p = captured_provenance.get('protocol') if isinstance(captured_provenance, dict) else None
        if not isinstance(p, dict) or p != protocol():
            raise ValueError('captured development protocol differs from the checked-out protocol')
        measurement_bundle = bundle_for(
            source_snapshot=source_snapshot, source_names=source_names,
        )
        raw = measurement_bundle['arrays']['development']
        development_sha = measurement_bundle['manifest']['source_files'][
            'development_measurements'
        ]['sha256']
        source_hashes = {
            role: entry['sha256']
            for role, entry in measurement_bundle['manifest']['source_files'].items()
        }
        source_read_mode = 'immutable_snapshot'
    else:
        if source_names is not None:
            raise ValueError('source_names is only valid with source_snapshot')
        if (output / 'submission.json').exists() or (output / 'validation.npz').exists():
            raise ValueError('Development is closed for this run; use a new run and disclose prior validation access')
        p = protocol()
        raw = load_data(output, 'development')
        development_sha = digest(output / 'development.npz')
        source_hashes = {'development_measurements': development_sha}
        provenance = output / 'development_provenance.json'
        if provenance.is_file():
            source_hashes['development_provenance'] = digest(provenance)
        source_read_mode = 'path'

    data = blocks(fold(raw), p['block_length'])
    index = bootstrap_indices(len(data), p['bootstrap_samples'], p['seed'])
    windows, masses = [], []
    for window in p['reference_windows']:
        model = ReferenceFit(data, p, window)
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
    odd = raw[..., 1:8]-raw[..., 15:8:-1]
    odd_blocks = blocks(odd, p['block_length'])
    odd_se = odd_blocks.std(axis=0, ddof=1)/np.sqrt(len(odd_blocks))
    reflection_z = float(np.max(abs(odd_blocks.mean(axis=0))/odd_se))
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
    fit, converged, _ = optimize_mixture(data, scales, m, mb, index, p)
    weights = fit.x/scales
    candidate_result = risk(data, weights, m, mb, index, p)
    submissions = {'local': basis[0].tolist(), 'tuned_gaussian': basis[best].tolist(),
                   'agent_signed_mixture': weights.tolist()}
    metrics = {name: late_metric(data, w, p).tolist() for name, w in submissions.items()}
    frozen = {'protocol_hash': fingerprint(p), 'development_sha256': development_sha,
              'analysis_task_spec_sha256': fingerprint(task_spec_for(p)),
              'statistics_sha256': digest(Path(__file__).with_name('statistics.py')),
              'kernel_sha256': digest(Path(__file__).with_name('kernels.py')),
              'development_source_read_mode': source_read_mode,
              'development_source_hashes': source_hashes,
              'powers': p['powers'], 'alpha': p['alpha'], 'submissions': submissions,
              'normalized_mixture_weights': fit.x.tolist(), 'basis_amplitude_scales': scales.tolist(),
              'late_inverse_covariances': metrics, 'selected_gaussian_power': p['powers'][best],
              'optimizer': {'converged_starts': converged, 'objective': float(fit.fun),
                            'iterations': int(fit.nit), 'l1': float(abs(fit.x).sum())},
              'calibration': calibration}
    if source_snapshot is not None:
        output.mkdir(parents=True, exist_ok=False)
    write_json(output/'development_results.json', {'basis': baseline_results, 'agent': candidate_result})
    write_json(output/'submission.json', frozen)
    np.savez_compressed(output/'reference_bootstrap.npz', masses=mb, indices=index)
    if source_snapshot is not None:
        core_outputs = {
            'development_results.json': output/'development_results.json',
            'submission.json': output/'submission.json',
            'reference_bootstrap.npz': output/'reference_bootstrap.npz',
        }
        output_records = {
            name: {'sha256': digest(path), 'size_bytes': path.stat().st_size}
            for name, path in core_outputs.items()
        }
        risk_rows = [*baseline_results, candidate_result]
        valid_risks = sum(
            isinstance(row, dict)
            and isinstance(row.get('risk'), (int, float, np.number))
            and bool(np.isfinite(row['risk']))
            for row in risk_rows
        )
        denominators = {
            'captured_development_source_file': {
                'planned': 2, 'valid': len(source_hashes),
                'failed': 0, 'abstained': 2-len(source_hashes),
            },
            'development_configuration': {
                'planned': int(raw.shape[0]), 'valid': int(raw.shape[0]),
                'failed': 0, 'abstained': 0,
            },
            'development_block': {
                'planned': int(len(data)), 'valid': int(len(data)),
                'failed': 0, 'abstained': 0,
            },
            'finite_development_risk_summary': {
                'planned': len(risk_rows), 'valid': int(valid_risks),
                'failed': len(risk_rows)-int(valid_risks), 'abstained': 0,
            },
        }
        diagnostic = {
            'schema': 'pion-development-snapshot-diagnostic-v1',
            'classification': 'OPEN_DEVELOPMENT_DIAGNOSTIC_NOT_CONFIRMATION',
            'source_read_mode': 'immutable_snapshot',
            'protocol_sha256': fingerprint(p),
            'analysis_task_spec_sha256': frozen['analysis_task_spec_sha256'],
            'development_source_sha256': development_sha,
            'development_array_sha256': measurement_bundle['manifest']['inputs']['development'],
            'development_configuration_ids_sha256': measurement_bundle['manifest']['ids']['development'],
            'source_files': measurement_bundle['manifest']['source_files'],
            'development_configuration_count': int(raw.shape[0]),
            'development_block_count': int(len(data)),
            'reference': {
                'passed': bool(not failures),
                'nominal_window': reference['window'],
                'mass': float(m),
                'standard_error': float(reference['se']),
                'fit_p_approx': float(reference['p_approx']),
                'failure_reasons': failures,
            },
            'denominators': denominators,
            'output_files': output_records,
            'physical_score': None,
            'certified': False,
            'text_flow': [
                'two captured development source files -> protocol, configuration-ID, shape, finite-value and NPZ-budget checks',
                'captured correlators -> folded five-configuration blocks -> reference-window fits and paired bootstrap draws',
                'same development blocks -> basis and signed-mixture risk summaries -> hash-bound outputs',
                'diagnostic only; no validation, gap, anchor, confirmation or physical-score claim',
            ],
            'limitations': [
                'the source digest binds bytes but does not establish authorization, provenance authority, or statistical independence',
                'development outputs do not replace the independent reference, anchor, dependence, or common-bias evidence gates',
            ],
        }
        readme = [
            '# Pion development snapshot diagnostic',
            '',
            'Classification: `OPEN_DEVELOPMENT_DIAGNOSTIC_NOT_CONFIRMATION`.',
            '',
            'Text flow: captured provenance and development NPZ bytes are checked for protocol identity, configuration IDs, array shape, finite values, and decompression budget; the same captured correlators are folded into blocks, used for reference fits and risk summaries, and bound to the output hashes below.',
            '',
            f"- Captured development configurations: {raw.shape[0]}",
            f"- Development blocks: {len(data)}",
            f"- Nominal reference gate: {'pass' if not failures else 'fail'}",
            f"- Planned/valid/failed/abstained source files: 2/{len(source_hashes)}/{2-len(source_hashes)}/0",
            f"- Planned/valid/failed/abstained finite risk summaries: {len(risk_rows)}/{valid_risks}/{len(risk_rows)-valid_risks}/0",
            '- `physical_score`: `null`; certified: `false`.',
            '',
            'This development-only receipt does not authenticate the data owner, prove independent sampling, or provide anchor/reference calibration. It includes no validation, gap, holdout, or confirmation inputs.',
            '',
            'See `development_diagnostic.json` for source hashes, derived counts, artifact hashes, denominators, and limitations.',
        ]
        readme_path = output/'README.md'
        readme_path.write_text('\n'.join(readme)+'\n')
        output_records['README.md'] = {
            'sha256': digest(readme_path), 'size_bytes': readme_path.stat().st_size,
        }
        diagnostic['output_files'] = output_records
        write_json(output/'development_diagnostic.json', diagnostic)
        verify_snapshot_development_artifacts(
            output, source_snapshot=source_snapshot, source_names=source_names,
        )
    print('FROZEN_SUBMISSION', json.dumps(frozen), flush=True)


def verify_snapshot_development_artifacts(
    output, *, source_snapshot, source_names,
):
    """Check development-only input/output identity without reopening source paths."""
    output = Path(output)
    if not isinstance(source_snapshot, SourceByteSnapshot):
        raise ValueError('source_snapshot must be a verified SourceByteSnapshot')
    if (not isinstance(source_names, dict)
            or set(source_names) != {
                'development_provenance', 'development_measurements'
            }):
        raise ValueError('source_names must identify exactly the two development sources')
    try:
        diagnostic = json.loads((output/'development_diagnostic.json').read_text())
        frozen = json.loads((output/'submission.json').read_text())
        results = json.loads((output/'development_results.json').read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('development diagnostic, submission, or risk summary is unreadable') from exc
    if (not isinstance(diagnostic, dict)
            or diagnostic.get('schema') != 'pion-development-snapshot-diagnostic-v1'
            or diagnostic.get('classification') != 'OPEN_DEVELOPMENT_DIAGNOSTIC_NOT_CONFIRMATION'
            or diagnostic.get('source_read_mode') != 'immutable_snapshot'
            or diagnostic.get('certified') is not False
            or diagnostic.get('physical_score', 'missing') is not None):
        raise ValueError('development diagnostic is not explicitly development-only')

    measurement_bundle = bundle_for(
        source_snapshot=source_snapshot, source_names=source_names,
    )
    p = protocol()
    source_files = measurement_bundle['manifest']['source_files']
    source_hashes = {role: entry['sha256'] for role, entry in source_files.items()}
    raw = measurement_bundle['arrays']['development']
    data = blocks(fold(raw), p['block_length'])
    expected_fields = {
        'protocol_sha256': fingerprint(p),
        'analysis_task_spec_sha256': fingerprint(task_spec_for(p)),
        'development_source_sha256': source_hashes['development_measurements'],
        'development_array_sha256': measurement_bundle['manifest']['inputs']['development'],
        'development_configuration_ids_sha256': measurement_bundle['manifest']['ids']['development'],
        'source_files': source_files,
        'development_configuration_count': int(raw.shape[0]),
        'development_block_count': int(len(data)),
    }
    for key, value in expected_fields.items():
        if diagnostic.get(key) != value:
            raise ValueError(f'development diagnostic {key} differs from the captured snapshot')
    if (not isinstance(frozen, dict)
            or frozen.get('development_source_read_mode') != 'immutable_snapshot'
            or frozen.get('protocol_hash') != fingerprint(p)
            or frozen.get('analysis_task_spec_sha256') != fingerprint(task_spec_for(p))
            or frozen.get('development_sha256') != source_hashes['development_measurements']
            or frozen.get('development_source_hashes') != source_hashes
            or frozen.get('calibration', {}).get('n_development_blocks') != len(data)):
        raise ValueError('frozen submission does not bind the captured development inputs')
    expected_reference = frozen.get('calibration', {})
    expected_reference = expected_reference.get('reference')
    if not isinstance(expected_reference, dict):
        raise ValueError('frozen submission has no development reference summary')
    recorded_reference = diagnostic.get('reference')
    if (not isinstance(recorded_reference, dict)
            or recorded_reference.get('passed') is not frozen['calibration'].get('reference_passed')
            or recorded_reference.get('nominal_window') != expected_reference.get('window')
            or recorded_reference.get('mass') != expected_reference.get('mass')
            or recorded_reference.get('standard_error') != expected_reference.get('se')
            or recorded_reference.get('fit_p_approx') != expected_reference.get('p_approx')
            or recorded_reference.get('failure_reasons') != frozen['calibration'].get('failure_reasons')):
        raise ValueError('development reference diagnostic differs from the frozen submission')

    if not isinstance(results, dict) or not isinstance(results.get('basis'), list):
        raise ValueError('development risk summary has no basis result list')
    risk_rows = results['basis'] + [results.get('agent')]
    valid_risks = sum(
        isinstance(row, dict)
        and isinstance(row.get('risk'), (int, float, np.number))
        and not isinstance(row.get('risk'), (bool, np.bool_))
        and bool(np.isfinite(row['risk']))
        for row in risk_rows
    )
    expected_denominators = {
        'captured_development_source_file': {
            'planned': 2, 'valid': 2, 'failed': 0, 'abstained': 0,
        },
        'development_configuration': {
            'planned': int(raw.shape[0]), 'valid': int(raw.shape[0]),
            'failed': 0, 'abstained': 0,
        },
        'development_block': {
            'planned': int(len(data)), 'valid': int(len(data)),
            'failed': 0, 'abstained': 0,
        },
        'finite_development_risk_summary': {
            'planned': len(risk_rows), 'valid': int(valid_risks),
            'failed': len(risk_rows)-int(valid_risks), 'abstained': 0,
        },
    }
    if diagnostic.get('denominators') != expected_denominators:
        raise ValueError('development diagnostic denominators differ from parsed outputs')
    output_files = diagnostic.get('output_files')
    required_outputs = {
        'development_results.json', 'submission.json',
        'reference_bootstrap.npz', 'README.md',
    }
    if not isinstance(output_files, dict) or set(output_files) != required_outputs:
        raise ValueError('development diagnostic output hash map is incomplete or unexpected')
    for name, record in output_files.items():
        path = output/name
        if (not path.is_file() or not isinstance(record, dict)
                or digest(path) != record.get('sha256')
                or path.stat().st_size != record.get('size_bytes')):
            raise ValueError(f'development output hash mismatch: {name}')
    readme = (output/'README.md').read_text()
    if ('OPEN_DEVELOPMENT_DIAGNOSTIC_NOT_CONFIRMATION' not in readme
            or 'physical_score`: `null`' not in readme):
        raise ValueError('development README omits its non-certification boundary')
    return {
        'schema': 'pion-development-snapshot-artifact-verification-v1',
        'status': 'SNAPSHOT_ARTIFACTS_VERIFIED_NOT_CERTIFIED',
        'input_source_denominator': {
            'planned': 2, 'valid': 2, 'failed': 0, 'abstained': 0,
        },
        'output_artifact_denominator': {
            'planned': len(required_outputs), 'valid': len(required_outputs),
            'failed': 0, 'abstained': 0,
        },
        'certified': False,
        'physical_score': None,
    }


def nested_intervals(development, validation, frozen, p):
    """Resample the entire conditional risk statistic, including its variance.

    Fit metrics are frozen at development, not re-estimated on validation or
    inside resamples. Shared reference and paired validation draws preserve
    correlations between all candidates. No replicate is silently discarded.
    """
    rng = np.random.default_rng(p['seed']+2)
    model = ReferenceFit(development, p, p['reference_windows'][p['reference_nominal_window']])
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


def pion_mass_check(val, weights, ref_mass, ref_draws, val_indices, p, calibration):
    """Fit a free sink mass on validation and compare it to the development target."""
    reasons = list(calibration['failure_reasons'])
    if not calibration['reference_passed'] and not reasons:
        reasons.append('Development reference failed')
    # Resampling is paired across validation windows. Development and validation
    # indices are drawn independently, as in the conditional risk calculation.
    nominal = p['reference_windows'][p['reference_nominal_window']]
    windows = [nominal, *(w for w in p['reference_windows'] if w[0] > nominal[0])]
    results = []
    central = None
    def check_shift(label, difference, draws):
        error = float(draws.std(ddof=1))
        if abs(difference) > p['reference_max_window_z'] * error:
            reasons.append(label)
        return {'mass_difference': difference, 'bootstrap_se': error}

    try:
        channel = np.einsum('nbt,b->nt', val, weights)[:, None, :]
        for window in windows:
            fit = ReferenceFit(channel, dict(p, reference_basis=[0]), window)
            item, draws = fit.bootstrap(val_indices)
            results.append(item)
            lo, hi = p['mass_bounds']
            if (not np.isfinite(item['p_approx']) or item['p_approx'] < p['reference_min_fit_p']
                    or item['se'] > item['mass'] * p['reference_max_relative_error']
                    or np.any(draws <= lo + 1e-4) or np.any(draws >= hi - 1e-4)):
                reasons.append(f'Free-mass fit unresolved on {window}')
            item['vs_reference'] = check_shift(f'Free mass disagrees with development reference on {window}',
                                               item['mass'] - ref_mass, draws - ref_draws)
            if central is None:
                central = item['mass'], draws
            else:
                item['vs_nominal'] = check_shift(f'Free mass changes with fit window {window}',
                                                 item['mass'] - central[0], draws - central[1])
    except (ValueError, np.linalg.LinAlgError) as exc:
        reasons.append(f'Free-mass fit failed: {exc}')
    return {'status': 'unresolved' if reasons else 'consistent_with_reference',
            'development_reference_mass': ref_mass, 'fits': results, 'reasons': reasons}


def evaluate(output):
    output = Path(output)
    p = protocol()
    frozen = json.loads((output/'submission.json').read_text())
    if frozen.get('development_source_read_mode') == 'immutable_snapshot' and not (output/'development.npz').is_file():
        raise ValueError(
            'Snapshot-only development has no colocated NPZ for validation evaluation; '
            'supply an explicitly authorized, hash-matching development/validation transaction'
        )
    if frozen['protocol_hash'] != fingerprint(p) or frozen['development_sha256'] != digest(output/'development.npz'):
        raise ValueError('Frozen protocol or development data changed')
    if (frozen.get('analysis_task_spec_sha256') is not None
            and frozen['analysis_task_spec_sha256'] != fingerprint(task_spec_for(p))):
        raise ValueError('Frozen analysis task/source dependency contract changed')
    if frozen['statistics_sha256'] != digest(Path(__file__).with_name('statistics.py')) or frozen['kernel_sha256'] != digest(Path(__file__).with_name('kernels.py')):
        raise ValueError('Frozen scoring/kernel implementation changed')
    dev = blocks(fold(load_data(output, 'development')), p['block_length'])
    raw = load_data(output, 'validation')
    val = blocks(fold(raw), p['block_length'])
    model = ReferenceFit(dev, p, p['reference_windows'][p['reference_nominal_window']])
    reference, mb = model.bootstrap(bootstrap_indices(len(dev), p['bootstrap_samples'], p['seed']))
    vi = bootstrap_indices(len(val), p['bootstrap_samples'], p['seed']+1)
    report = {'protocol': p, 'submission_sha256': digest(output/'submission.json'),
              'validation_sha256': digest(output/'validation.npz'), 'reference': reference,
              'reference_passed': frozen['calibration']['reference_passed'],
              'reference_failure_reasons': frozen['calibration']['failure_reasons'],
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
        for name, value in report['results'].items():
            state = pion_mass_check(val, frozen['submissions'][name], reference['mass'], mb, vi, p, frozen['calibration'])
            value['pion_mass_check'] = state
            value['diagnostic_loss_ratio_vs_local'] = (baseline+eps)/(value['risk']+eps)
            value['score'] = 100*value['diagnostic_loss_ratio_vs_local'] if state['status'] == 'consistent_with_reference' else None
        draws, names = nested_intervals(dev, val, frozen, p)
        np.savez_compressed(output/'nested_bootstrap.npz', risks=draws, names=np.array(names))
        for j, name in enumerate(names):
            ratios = (draws[:, 0]+eps)/(draws[:, j]+eps)
            report['results'][name]['diagnostic_loss_ratio_95_interval'] = np.quantile(ratios, [.025, .975]).tolist()
            report['results'][name]['risk_95_interval'] = np.quantile(draws[:, j], [.025, .975]).tolist()
        j, k = names.index('agent_signed_mixture'), names.index('tuned_gaussian')
        report['agent_vs_tuned_gaussian'] = {
            'risk_ratio': report['results']['tuned_gaussian']['risk']/report['results']['agent_signed_mixture']['risk'],
            'risk_ratio_95_interval': np.quantile((draws[:, k]+eps)/(draws[:, j]+eps), [.025, .975]).tolist()}
    write_json(output/'validation_results.json', report)
    print('VALIDATION', json.dumps(report), flush=True)
