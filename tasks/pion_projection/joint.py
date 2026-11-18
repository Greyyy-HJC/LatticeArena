"""Joint development/validation resampling and a conditional certificate (P0.6).

For each draw: resample development blocks, refit the reference, bootstrap its
masses inside the same resampled development set, and score every candidate on
the same resampled validation blocks with that draw's reference. The resulting
risk arrays share draw IDs, so the certificate can pair candidates and models.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .common import digest, fingerprint, write_json
from .reference import make_reference
from .scoring import improvement_certificate
from .statistics import blocks, fold, risk


def run(output, models=None, outer=200, inner=64, seed=20261119, dest=None):
    output = Path(output)
    dest = Path(dest) if dest else output / 'joint'
    if (dest / 'metric_results.json').exists():
        raise ValueError(f'{dest} already contains a joint result; choose a new output directory')
    dest.mkdir(parents=True, exist_ok=True)
    p = json.loads((output / 'development_provenance.json').read_text())['protocol']
    sub = json.loads((output / 'submission.json').read_text())
    report = json.loads((output / 'validation_results.json').read_text())
    from .identity import verify_run_identity
    frozen_kind, reasons = verify_run_identity(output, models)
    if frozen_kind is None:
        raise ValueError('; '.join(reasons))
    models = (frozen_kind,)
    if reasons:
        raise ValueError('; '.join(reasons))
    artifact = sub.get('gate_artifact')
    artifact_reason = None
    if artifact is None:
        artifact_reason = 'gate_artifact_missing'
    else:
        if artifact.get('model') != frozen_kind:
            raise ValueError('gate artifact model differs from the frozen kind')
        if artifact.get('development_sha256') != sub.get('development_sha256'):
            raise ValueError('gate artifact development hash differs from the submission')
        if artifact.get('protocol_hash') != sub.get('protocol_hash'):
            raise ValueError('gate artifact protocol hash differs from the submission')
        if artifact.get('window') != p['reference_windows'][p['reference_nominal_window']]:
            raise ValueError('gate artifact window differs from the protocol window')
    dev = blocks(fold(np.load(output / 'development.npz')['correlators']), p['block_length'])
    val = blocks(fold(np.load(output / 'validation.npz')['correlators']), p['block_length'])
    window = p['reference_windows'][p['reference_nominal_window']]
    names = list(sub['submissions'])
    rng = np.random.default_rng(seed)
    dev_index = rng.integers(0, len(dev), (outer, len(dev)))
    val_index = rng.integers(0, len(val), (outer, len(val)))
    dev_inner = rng.integers(0, len(dev), (outer, inner, len(dev)))
    val_inner = rng.integers(0, len(val), (outer, inner, len(val)))
    masses = {m: np.zeros(outer) for m in models}
    risks = {m: {n: np.zeros(outer) for n in names} for m in models}
    for r in range(outer):
        for m in models:
            ref = make_reference(dev[dev_index[r]], p, window, m)
            fit, boot_masses = ref.bootstrap(dev_inner[r])
            masses[m][r] = fit['mass']
            for name in names:
                value = risk(val[val_index[r]], sub['submissions'][name], fit['mass'], boot_masses,
                             val_inner[r], p, require_valid=False,
                             late_inverse=np.array(sub['late_inverse_covariances'][name]))
                risks[m][name][r] = value['risk']
    draw_ids = np.arange(outer)
    np.savez_compressed(dest / 'joint_draws.npz',
                        draw_id=draw_ids, dev_index=dev_index, val_index=val_index,
                        dev_inner=dev_inner, val_inner=val_inner,
                        models=np.asarray(list(models)), candidates=np.asarray(names),
                        **{f'mass_{m}': masses[m] for m in models},
                        **{f'risk_{m}_{n}': risks[m][n] for m in models for n in names})
    certificate = None
    if len(models) > 0:
        spec = {
            'schema': 'pion-score-v1', 'version': 'joint-draws-v1',
            'estimand': 'conditional perturbation risk improvement (joint draws)',
            'baseline_id': 'local', 'accepted_models': list(models),
            'candidate_ids': [n for n in names if n != 'local'],
            'analysis_version': 'joint-draws-v1', 'alpha': 0.05,
            'comparison_family': {'candidates': len(names) - 1, 'models': len(models)},
            'min_baseline_risk': 1e-6, 'invalid_tolerance': 0.05,
            'min_improvement': 1.0, 'risk_floor': 0.0,
            'require_draw_ids': True,
            'draw_ids_hash': hashlib.sha256(np.asarray(draw_ids, dtype=np.int64).tobytes()).hexdigest(),
        }
        certificate = improvement_certificate(
            {m: {n: risks[m][n] for n in names} for m in models}, spec,
            draw_ids=draw_ids)
    gate = report
    passed = bool(gate.get('reference_passed'))
    if certificate is not None:
        if artifact_reason:
            certificate['status'] = 'abstain'
            certificate['abstain'] = True
            certificate['failure_reasons'].append(artifact_reason)
        if not passed:
            certificate['status'] = 'abstain'
            certificate['abstain'] = True
            certificate['failure_reasons'].append('reference_gate_failed')
            for row in certificate['candidates'].values():
                row['certified'] = False
        certificate['reference_passed'] = passed
        certificate['gate_reasons'] = gate.get('reference_failure_reasons')
        certificate['protocol_hash'] = fingerprint(p)
        certificate['draws'] = {'outer': outer, 'inner': inner, 'seed': seed}
        certificate['bootstrap_mean_risks'] = {m: {n: float(risks[m][n].mean()) for n in names} for m in models}
        certificate['point_risks'] = {m: {n: gate.get('results', {}).get(n, {}).get('risk') for n in names}
                                      for m in models}
        write_json(dest / 'metric_results.json', certificate)
        write_json(dest / 'joint_manifest.json',
                   {'schema': 'pion-joint-manifest-v1',
                    'protocol_hash': fingerprint(p),
                    'development_sha256': digest(output / 'development.npz'),
                    'validation_sha256': digest(output / 'validation.npz'),
                    'reference_kind': sub.get('reference', {}).get('kind'),
                    'models': list(models), 'candidates': names,
                    'spec': {k: spec[k] for k in ('alpha', 'comparison_family', 'min_baseline_risk',
                                                  'min_improvement', 'risk_floor', 'require_draw_ids',
                                                  'draw_ids_hash')},
                    'draws': {'outer': outer, 'inner': inner, 'seed': seed},
                    'dependencies': {'statistics': digest(Path(__file__).with_name('statistics.py')),
                                     'reference': digest(Path(__file__).with_name('reference.py')),
                                     'scoring': digest(Path(__file__).with_name('scoring.py'))}})
    print(json.dumps(certificate, indent=2) if certificate else 'no models')
    return certificate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--models', nargs='+', default=['single'])
    parser.add_argument('--outer', type=int, default=200)
    parser.add_argument('--inner', type=int, default=64)
    parser.add_argument('--dest')
    args = parser.parse_args()
    run(args.output, tuple(args.models), args.outer, args.inner, dest=args.dest)
