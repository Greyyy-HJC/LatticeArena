"""TaskSpec and OperatorContract adapters for wilson_loop (F1 slice 1-2)."""
from pathlib import Path

from core.io import digest
from core.specs import operator_contract, task_spec


def task_spec_for(protocol):
    return task_spec(
        name='wilson_loop', observable='static potential from Wilson loops',
        geometry=list(protocol['geometry']),
        boundary={'spatial': 'periodic', 'temporal': 'periodic'},
        source={'kind': 'static', 'origin': 'spatial origin'},
        splits={'development': list(protocol['development']), 'gap': list(protocol['gap']),
                'validation': list(protocol['validation'])},
        reference={'kind': 'identity kernel', 'windows': protocol['reference_windows']},
        metric={'name': 'precision plus window-spread risk', 'nominal_window_index': protocol['reference_nominal_window']},
        budget={'bootstrap_samples': protocol['bootstrap_samples'], 'block_length': protocol['block_length']},
        dependencies={'kernels': digest(Path(__file__).with_name('kernels.py')),
                      'statistics': digest(Path(__file__).with_name('statistics.py'))},
    )


def operator_contract_for():
    return operator_contract(
        name='wilson-kernel-v1', axes='(mu,t,z,y,x,3,3)', dtype='complex128',
        transformation='gauge_covariant', support='spacetime_local',
        parameters={'alpha': 'float', 'steps': 'int'},
        dependencies={'kernels': digest(Path(__file__).with_name('kernels.py'))},
        constraints={'temporal_links_unsmeared': True, 'left_right_covariant': True},
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
    protocol = json.loads((run / 'validation_provenance.json').read_text())['protocol']
    arrays, splits, ids_hash, inputs = {}, {}, {}, {}
    for split in ('development', 'validation'):
        data = load_data(run, split, protocol)
        arrays[split] = data
        splits[split] = {'n': int(data.shape[0])}
        ids_hash[split] = digest_text(str(list(range(protocol[split][0], protocol[split][1] + 1))))
        inputs[split] = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
    code = {name: digest(Path(__file__).with_name(name)) for name in
            ('kernels.py', 'statistics.py', 'measure.py', 'common.py')}
    run_spec = task_spec_for(protocol)
    for split, info in splits.items():
        if info['n'] != run_spec['splits'][split][1] - run_spec['splits'][split][0] + 1:
            raise ValueError(f'{split}: bundle size {info["n"]} does not match the RunSpec range')
    manifest = {'protocol_hash': fingerprint(protocol), 'splits': splits, 'ids': ids_hash,
                'inputs': inputs, 'code': code, 'r_values': protocol['r_values'],
                'kernels': [k['name'] for k in protocol['kernels']],
                'run_spec': digest_text(json.dumps(run_spec, sort_keys=True))}
    return bundle('wilson_loop', arrays, manifest,
                  {'linear_response': False, 'cross_matrix': False, 'new_source': False})
