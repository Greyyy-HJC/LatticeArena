"""TaskSpec and OperatorContract adapters for pion_projection."""
import hashlib
import json
from pathlib import Path

import numpy as np

from core.io import digest, fingerprint
from core.measurement import bundle
from core.specs import operator_contract, task_spec
from core.source_integrity import SourceByteSnapshot

from .common import load_data


def _source_dependencies():
    """Hash task, immutable-read, scoring, and evidence-gate implementations."""
    root = Path(__file__).resolve().parents[2]
    paths = {
        'tasks/pion_projection/spec.py': Path(__file__),
        'tasks/pion_projection/common.py': Path(__file__).with_name('common.py'),
        'tasks/pion_projection/experiment.py': Path(__file__).with_name('experiment.py'),
        'tasks/pion_projection/candidate.py': Path(__file__).with_name('candidate.py'),
        'tasks/pion_projection/path_kernel.py': Path(__file__).with_name('path_kernel.py'),
        'tasks/pion_projection/evidence_adapter.py': Path(__file__).with_name('evidence_adapter.py'),
        'tasks/pion_projection/anchor_scalar_reader.py': Path(__file__).with_name('anchor_scalar_reader.py'),
        'tasks/pion_projection/bounded_block_risk_contract.py': Path(__file__).with_name('bounded_block_risk_contract.py'),
        'tasks/pion_projection/bounded_block_risk_contract_v1.json': Path(__file__).with_name('bounded_block_risk_contract_v1.json'),
        'tasks/pion_projection/bounded_block_risk_score.py': Path(__file__).with_name('bounded_block_risk_score.py'),
        'tasks/pion_projection/bounded_block_risk_anchor_markov.py': Path(__file__).with_name('bounded_block_risk_anchor_markov.py'),
        'tasks/pion_projection/bounded_block_risk_markov_gap.py': Path(__file__).with_name('bounded_block_risk_markov_gap.py'),
        'core/io.py': root / 'core' / 'io.py',
        'core/candidates.py': root / 'core' / 'candidates.py',
        'core/evidence.py': root / 'core' / 'evidence.py',
        'core/measurement.py': root / 'core' / 'measurement.py',
        'core/receipts.py': root / 'core' / 'receipts.py',
        'core/specs.py': root / 'core' / 'specs.py',
        'core/source_integrity.py': root / 'core' / 'source_integrity.py',
        'core/npz_safety.py': root / 'core' / 'npz_safety.py',
        'core/numerics.py': root / 'core' / 'numerics.py',
        'core/state.py': root / 'core' / 'state.py',
    }
    return {name: digest(path) for name, path in paths.items()}


def task_spec_for(protocol):
    dependencies = {
        'protocol': fingerprint(protocol),
        'kernels': digest(Path(__file__).with_name('kernels.py')),
        'statistics': digest(Path(__file__).with_name('statistics.py')),
        **_source_dependencies(),
    }
    return task_spec(
        name='pion_projection', observable='zero-momentum pseudoscalar two-point function',
        geometry=list(protocol['geometry']),
        boundary={'spatial': 'periodic', 'temporal': 'antiperiodic (fermion)'},
        source={'kind': 'point', 'position': list(protocol['source'])},
        splits={'development': list(protocol['development']), 'gap': list(protocol['gap']),
                'validation': list(protocol['validation'])},
        reference={'kind': 'single', 'window': protocol['reference_windows'][protocol['reference_nominal_window']]},
        metric={'name': 'conditional perturbation risk', 'score_window': list(protocol['score_window'])},
        budget={'bootstrap_samples': protocol['bootstrap_samples'], 'block_length': protocol['block_length']},
        dependencies=dependencies,
    )


def operator_contract_for():
    return operator_contract(
        name='pion-kernel-v1', axes='(z,y,x,3,nrhs)', dtype='complex128',
        transformation='spin_scalar', support='time_slice',
        parameters={'schema_by_builder': {
            'tasks.pion_projection.path_kernel:build_kernel': {'terms': 'list[{path: signed spatial steps, weight: real or [real, imag]}]'},
            'tasks.pion_projection.kernels:build_kernel': {'powers': 'list[int]', 'weights': 'list[float]', 'alpha': 'float'}},
            'custom_builder': 'builder-defined JSON parameters'},
        dependencies={name: digest(Path(__file__).with_name(name + '.py')) for name in ('kernels', 'path_kernel')},
        constraints={'hermitian': True, 'zero_momentum': True, 'gauge_covariant': True,
                     'spatial_symmetry': 'A1+', 'bilinear_irrep': 'A1-', 'isospin': 1,
                     'G_parity': -1, 'neutral_partner_C': 1, 'translation_covariant': True, 'linear_in_field': True},
    )


def bundle_for(run=None, *, source_snapshot=None, source_names=None):
    """Build a Pion measurement bundle from paths or a development snapshot.

    Snapshot mode is development-only and reads no files outside the immutable
    snapshot. The path-based compatibility mode retains its pre-existing
    development/validation behavior and should not be used for sealed runs.
    """
    if source_snapshot is not None:
        if run is not None:
            raise ValueError('snapshot mode does not accept a run directory')
        if (not isinstance(source_names, dict)
                or set(source_names) != {
                    'development_provenance', 'development_measurements'
                }):
            raise ValueError(
                'snapshot source_names must map development_provenance and '
                'development_measurements only'
            )
        return bundle_for_development_snapshot(
            source_snapshot,
            provenance_source=source_names['development_provenance'],
            development_source=source_names['development_measurements'],
        )
    if source_names is not None:
        raise ValueError('source_names is only valid with source_snapshot')
    if run is None:
        raise ValueError('run is required unless source_snapshot is supplied')

    run = Path(run)
    protocol = json.loads((run / 'development_provenance.json').read_text())['protocol']
    arrays, splits, ids_hash, inputs = {}, {}, {}, {}
    for split in ('development', 'validation'):
        data = load_data(run, split, protocol)
        arrays[split] = data
        splits[split] = {'n': int(data.shape[0])}
        ids_hash[split] = fingerprint(list(range(protocol[split][0], protocol[split][1] + 1)))
        inputs[split] = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
    code = {name: digest(Path(__file__).with_name(name)) for name in
            ('kernels.py', 'statistics.py', 'experiment.py')}
    code.update(_source_dependencies())
    run_spec = task_spec_for(protocol)
    manifest = {'protocol_hash': fingerprint(protocol), 'splits': splits, 'ids': ids_hash,
                'inputs': inputs, 'code': code, 'powers': protocol['powers'],
                'alpha': protocol['alpha'],
                'run_spec': fingerprint(run_spec)}
    return bundle('pion_projection', arrays, manifest,
                  {'linear_response': True, 'cross_matrix': False, 'new_source': False})


def bundle_for_development_snapshot(snapshot, *, provenance_source, development_source):
    """Build a development-only bundle from the exact bytes captured by a reader.

    This entrypoint deliberately has no validation/gap argument. It is suitable
    for a development evaluator that passes the *same* snapshot to its later
    evidence gate; it cannot silently fall back to files on disk. Passing this
    bundle alone does not establish authorization, provenance authority, or a
    physical score.
    """
    if not isinstance(snapshot, SourceByteSnapshot):
        raise ValueError('snapshot must be a verified SourceByteSnapshot')
    if not isinstance(provenance_source, str) or not provenance_source.strip():
        raise ValueError('provenance_source must name a captured snapshot entry')
    if not isinstance(development_source, str) or not development_source.strip():
        raise ValueError('development_source must name a captured snapshot entry')

    try:
        provenance = json.loads(snapshot.read_bytes(provenance_source))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('captured development provenance is not valid JSON') from exc
    if not isinstance(provenance, dict) or not isinstance(provenance.get('protocol'), dict):
        raise ValueError('captured development provenance must contain a protocol object')
    protocol = provenance['protocol']

    data = load_data(
        snapshot.root, 'development', protocol,
        source_snapshot=snapshot, source_name=development_source,
    )
    split = 'development'
    names = {
        'development_provenance': provenance_source,
        'development_measurements': development_source,
    }
    source_bindings = {
        role: {
            'snapshot_name': name,
            'relative_path': snapshot.source_paths.get(name),
            'sha256': snapshot.expected_hashes.get(name),
        }
        for role, name in names.items()
    }
    arrays = {split: data}
    splits = {split: {'n': int(data.shape[0])}}
    ids_hash = {
        split: fingerprint(list(range(protocol[split][0], protocol[split][1] + 1)))
    }
    inputs = {
        split: hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
    }
    code = {name: digest(Path(__file__).with_name(name)) for name in
            ('kernels.py', 'statistics.py', 'experiment.py')}
    code.update(_source_dependencies())
    run_spec = task_spec_for(protocol)
    manifest = {
        'protocol_hash': fingerprint(protocol), 'splits': splits, 'ids': ids_hash,
        'inputs': inputs, 'code': code, 'powers': protocol['powers'],
        'alpha': protocol['alpha'], 'run_spec': fingerprint(run_spec),
        'source_files': source_bindings,
        'source_read_mode': 'immutable_snapshot',
    }
    return bundle('pion_projection', arrays, manifest,
                  {'linear_response': True, 'cross_matrix': False, 'new_source': False})
