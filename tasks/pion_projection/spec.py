"""TaskSpec and OperatorContract adapters for pion_projection (F1 slice 1-2)."""
from pathlib import Path

from core.io import digest
from core.specs import operator_contract, task_spec


def task_spec_for(protocol, reference_kind='single', protocol_path=None):
    return task_spec(
        name='pion_projection', observable='zero-momentum pseudoscalar two-point function',
        geometry=list(protocol['geometry']),
        boundary={'spatial': 'periodic', 'temporal': 'antiperiodic (fermion)'},
        source={'kind': 'point', 'time': protocol['source']},
        splits={'development': list(protocol['development']), 'gap': list(protocol['gap']),
                'validation': list(protocol['validation'])},
        reference={'kind': reference_kind, 'window': protocol['reference_windows'][protocol['reference_nominal_window']]},
        metric={'name': 'conditional perturbation risk', 'score_window': list(protocol['score_window'])},
        budget={'bootstrap_samples': protocol['bootstrap_samples'], 'block_length': protocol['block_length']},
        dependencies={'protocol': digest(protocol_path) if protocol_path else 'runtime',
                      'kernels': digest(Path(__file__).with_name('kernels.py')),
                      'statistics': digest(Path(__file__).with_name('statistics.py'))},
    )


def operator_contract_for():
    return operator_contract(
        name='pion-kernel-v1', axes='(z,y,x,3,nrhs)', dtype='complex128',
        transformation='spin_scalar', support='time_slice',
        parameters={'powers': 'list[int]', 'weights': 'list[float]', 'alpha': 'float'},
        dependencies={'kernels': digest(Path(__file__).with_name('kernels.py'))},
        constraints={'hermitian': True, 'zero_momentum': True, 'gauge_covariant': True},
    )


def bundle_for(run):
    """Trusted measurement bundle: load_data checks IDs/hash/shape/finite."""
    import hashlib
    import json

    import numpy as np

    from core.io import digest_text, fingerprint
    from core.measurement import bundle

    from .common import load_data

    run = Path(run)
    protocol = json.loads((run / 'development_provenance.json').read_text())['protocol']
    arrays, splits, ids_hash, inputs = {}, {}, {}, {}
    for split in ('development', 'validation'):
        data = load_data(run, split, protocol)
        arrays[split] = data
        splits[split] = {'n': int(data.shape[0])}
        ids_hash[split] = digest_text(str(list(range(protocol[split][0], protocol[split][1] + 1))))
        inputs[split] = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
    code = {name: digest(Path(__file__).with_name(name)) for name in
            ('kernels.py', 'statistics.py', 'experiment.py')}
    run_spec = task_spec_for(protocol)
    for split, info in splits.items():
        if info['n'] != run_spec['splits'][split][1] - run_spec['splits'][split][0] + 1:
            raise ValueError(f'{split}: bundle size {info["n"]} does not match the RunSpec range')
    manifest = {'protocol_hash': fingerprint(protocol), 'splits': splits, 'ids': ids_hash,
                'inputs': inputs, 'code': code, 'powers': protocol['powers'],
                'alpha': protocol['alpha'],
                'run_spec': digest_text(json.dumps(run_spec, sort_keys=True))}
    return bundle('pion_projection', arrays, manifest,
                  {'linear_response': True, 'cross_matrix': False, 'new_source': False})
