"""Apply a submitted time-slice kernel to the fixed cached propagators.

This is a trusted local Python plugin interface, not an adversarial sandbox.
The v1 executable contract accepts real Hermitian spin-scalar kernels; arbitrary
complex non-Hermitian observables require the future complex scoring extension.
"""
import importlib
import inspect
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .common import digest, fingerprint, indices, load_data, protocol, write_json
from .statistics import ReferenceFit, blocks, bootstrap_indices, fold, risk
from .experiment import late_metric, nested_intervals, pion_mass_check
from .spec import task_spec_for
from . import kernels, path_kernel


class CandidateRejected(ValueError):
    """The measured candidate cannot be scored under the frozen metric."""


def apply_kernel(gauge, prop, builder, parameters):
    import cupy as cp
    links = cp.asarray(gauge.lexico(force_numpy=False)[:3])
    fields = cp.asarray(prop.lexico(force_numpy=False))
    correlator = []
    for t in range(fields.shape[0]):
        shape = fields[t].shape[:3]
        field = fields[t].transpose(0, 1, 2, 5, 3, 4, 6).reshape(*shape, 3, 48)
        kernel = builder(links[:, t].copy(), parameters)
        out = kernel.apply(field.copy())
        if out.shape != field.shape or not bool(cp.all(cp.isfinite(out))):
            raise CandidateRejected('Kernel returned an invalid field')
        c = complex(cp.vdot(field, out).get())
        scale = float((cp.linalg.norm(field)*cp.linalg.norm(out)).get())
        if abs(c.imag) > 1e-10*max(scale, 1e-300):
            raise CandidateRejected('Non-Hermitian/complex correlator is outside the executable v1 contract')
        correlator.append(c.real)
    return np.asarray(correlator)


def run_candidate(output, definition_path, split, verify_only=False, *, candidate_root=None):
    output = Path(output)
    submission_path = output / 'submission.json'
    if submission_path.is_file():
        preliminary = json.loads(submission_path.read_text())
        frozen_spec_hash = preliminary.get('analysis_task_spec_sha256')
        if (frozen_spec_hash is not None
                and frozen_spec_hash != fingerprint(task_spec_for(protocol()))):
            raise ValueError('Frozen analysis task/source dependency contract changed')
        if (preliminary.get('development_source_read_mode') == 'immutable_snapshot'
                and not (output / 'development.npz').is_file()):
            raise ValueError(
                'Snapshot-only development has no colocated NPZ for candidate execution; '
                'provide an explicitly authorized, hash-matching development transaction'
            )

    import cupy as cp
    from mpi4py import MPI
    from pyquda_utils import core, io

    p = protocol()
    reference_frozen = json.loads((output/'submission.json').read_text())
    if reference_frozen['protocol_hash'] != fingerprint(p) or reference_frozen['development_sha256'] != digest(output/'development.npz'):
        raise ValueError('Frozen reference protocol/development data changed')
    if reference_frozen['statistics_sha256'] != digest(Path(__file__).with_name('statistics.py')):
        raise ValueError('Frozen scoring implementation changed')
    spec = json.loads(Path(definition_path).read_text())
    module, function = spec['builder'].split(':')
    builder = getattr(importlib.import_module(module), function)
    channel_by_construction = builder in (kernels.build_kernel, path_kernel.build_kernel)
    identity = {'definition': spec, 'builder_sha256': digest(inspect.getsourcefile(builder)),
                'protocol_hash': fingerprint(p), 'reference_submission_sha256': digest(output/'submission.json')}
    root = output / 'candidates' if candidate_root is None else Path(candidate_root)
    destination = root / fingerprint(identity)[:16]
    destination.mkdir(parents=True, exist_ok=True)
    frozen_path = destination/'frozen.json'
    if not verify_only and split == 'development' and (frozen_path.exists() or (destination/'validation.npz').exists()):
        raise ValueError('Candidate already frozen; do not overwrite development measurements')
    if split == 'validation' and not verify_only:
        frozen = json.loads(frozen_path.read_text())
        if frozen['identity'] != identity or frozen['development_sha256'] != digest(destination/'development.npz'):
            raise ValueError('Candidate changed after freeze')
    source = json.loads((output/f'{split}_provenance.json').read_text())
    if source['protocol'] != p:
        raise ValueError('Propagator provenance uses another protocol')
    expected_ids = indices(split).tolist()
    if [r['id'] for r in source['records']] != expected_ids:
        raise ValueError('Incomplete propagator cache manifest')
    Path('.cache/quda').mkdir(parents=True, exist_ok=True)
    core.init([1, 1, 1, 1], p['geometry'], resource_path='.cache/quda')
    if MPI.COMM_WORLD.Get_size() != 1:
        raise ValueError('One rank is required')
    cp.get_default_memory_pool().set_limit(size=p['gpu_memory_limit_gib']*1024**3)
    latt = core.LatticeInfo(p['geometry'], p['t_boundary'], 1.0)
    rows, timings = [], []
    selected = source['records'][:1] if verify_only else source['records']
    for record in selected:
        start = perf_counter()
        cfg = Path(p['ensemble'])/f'wilson_b6.{record["id"]}'
        if digest(cfg) != record['gauge_sha256'] or record['residual_max'] > p['tolerance']:
            raise ValueError('Gauge checksum/residual provenance mismatch')
        cached = Path('.cache/pion_projection/propagators')/f'{record["cache_key"]}.npy'
        metadata = json.loads(cached.with_suffix('.json').read_text())
        if (fingerprint(metadata['keydata']) != record['cache_key'] or
                metadata['keydata']['gauge'] != record['gauge_sha256'] or digest(cached) != metadata['array_sha256']):
            raise ValueError('Propagator checksum mismatch')
        gauge = io.readNERSCGauge(str(cfg))
        prop = core.LatticePropagator(latt, cp.asarray(np.load(cached, allow_pickle=False)))
        try:
            rows.append(apply_kernel(gauge, prop, builder, spec['parameters']))
        except cp.cuda.memory.OutOfMemoryError as exc:
            raise CandidateRejected('Operator exceeded GPU memory budget') from exc
        cp.cuda.Stream.null.synchronize()
        elapsed = perf_counter()-start
        if elapsed > p['wall_seconds_per_configuration']:
            raise CandidateRejected('Candidate exceeded the per-configuration budget')
        timings.append(elapsed)
        print(f'Candidate {destination.name}: cfg {record["id"]}, {elapsed:.2f}s', flush=True)
        del gauge, prop
    if verify_only:
        if spec['builder'] != 'tasks.pion_projection.kernels:build_kernel':
            raise ValueError('Verification against the basis is only available for the diffusion kernel')
        if spec['parameters']['powers'] != p['powers'] or spec['parameters']['alpha'] != p['alpha']:
            raise ValueError('Verification requires the measured diffusion basis')
        measured_basis = load_data(output, split)[0]
        expected = np.einsum('b,bt->t', spec['parameters']['weights'], measured_basis)
        np.testing.assert_allclose(rows[0], expected, rtol=1e-10, atol=1e-12)
        write_json(destination/f'{split}_verification.json', {'definition': spec, 'configuration': expected_ids[0],
                   'max_absolute_difference': float(np.max(abs(rows[0]-expected))),
                   'max_relative_difference': float(np.max(abs((rows[0]-expected)/expected))),
                   'seconds': timings[0], 'reference_submission_sha256': digest(output/'submission.json')})
        print('Direct submitted operator agrees with basis contraction.', flush=True)
        return
    np.savez_compressed(destination/f'{split}.npz', ids=expected_ids, correlators=np.asarray(rows)[:, None, :])
    write_json(destination/f'{split}_provenance.json', {'identity': identity, 'seconds_per_configuration': timings,
               'source_manifest_sha256': digest(output/f'{split}_provenance.json')})
    dev = blocks(fold(load_data(output, 'development')), p['block_length'])
    model = ReferenceFit(dev, p, p['reference_windows'][p['reference_nominal_window']])
    ref, masses = model.bootstrap(bootstrap_indices(len(dev), p['bootstrap_samples'], p['seed']))
    measured = blocks(fold(np.asarray(rows)[:, None, :]), p['block_length'])
    if split == 'development':
        try:
            metric = late_metric(measured, [1.], p)
            diagnostic = risk(measured, [1.], ref['mass'], masses,
                              bootstrap_indices(len(dev), p['bootstrap_samples'], p['seed']), p, late_inverse=metric)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise CandidateRejected(str(exc)) from exc
        write_json(frozen_path, {'identity': identity, 'development_sha256': digest(destination/'development.npz'),
                                'late_inverse': metric.tolist(), 'development_diagnostic': diagnostic})
    else:
        val = blocks(fold(load_data(output, 'validation')), p['block_length'])
        report = compare_candidate(dev, val, measured, reference_frozen, frozen['late_inverse'], ref['mass'], masses, p,
                                   channel_by_construction=channel_by_construction)
        report['validation_sha256'] = digest(destination/'validation.npz')
        report['baseline_validation_sha256'] = digest(output/'validation.npz')
        write_json(destination/'results.json', report)
    print(f'Candidate artifacts: {destination}', flush=True)
    return destination


def compare_candidate(dev, val, measured, frozen, late_inverse, mass, masses, p, *, channel_by_construction=False):
    """Evaluate each frozen operator on identical validation blocks and reference draws."""
    combined = np.concatenate([val, measured], axis=1)
    weights = {name: [*w, 0.] for name, w in frozen['submissions'].items()}
    weights['candidate'] = [0.] * val.shape[1] + [1.]
    metrics = dict(frozen['late_inverse_covariances'], candidate=late_inverse)
    vi = bootstrap_indices(len(val), p['bootstrap_samples'], p['seed']+1)
    results = {}
    valid = {}
    for name, w in weights.items():
        try:
            results[name] = risk(combined, w, mass, masses, vi, p, late_inverse=np.asarray(metrics[name]))
        except (ValueError, np.linalg.LinAlgError) as exc:
            results[name] = {'invalid_reason': str(exc)}
        else:
            valid[name] = w

    paired = {}
    interval_error = None
    if 'candidate' in valid and len(valid) > 1:
        eps = p['score_epsilon']
        for name in frozen['submissions']:
            if name in valid:
                paired[name] = {'risk_ratio': (results[name]['risk'] + eps) / (results['candidate']['risk'] + eps),
                                'risk_ratio_95_interval': None}
        try:
            draws, names = nested_intervals(dev, combined, {'submissions': valid, 'late_inverse_covariances': metrics}, p)
            for name in paired:
                ratios = (draws[:, names.index(name)] + eps) / (draws[:, names.index('candidate')] + eps)
                paired[name]['risk_ratio_95_interval'] = np.quantile(ratios, [.025, .975]).tolist()
        except (ValueError, np.linalg.LinAlgError) as exc:
            interval_error = str(exc)
    local = paired.get('local', {})
    ratio = local.get('risk_ratio')
    calibration = frozen['calibration']
    state = pion_mass_check(measured, [1.], mass, masses, vi, p, calibration)
    eligible = channel_by_construction and state['status'] == 'consistent_with_reference'
    return {'pion_channel': 'by_construction' if channel_by_construction else 'custom_unverified',
            'pion_mass_check': state, 'results': results, 'diagnostic_loss_ratio': ratio,
            'diagnostic_loss_ratio_95_interval': local.get('risk_ratio_95_interval'),
            'paired_vs_candidate': paired, 'interval_failure_reason': interval_error,
            'score': 100 * ratio if ratio is not None and eligible and interval_error is None else None,
            'reference_failure_reasons': calibration['failure_reasons']}
