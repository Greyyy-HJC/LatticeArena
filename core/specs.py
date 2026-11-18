"""Minimal TaskSpec/RunSpec and OperatorContract/CandidateSpec (todo F1, slices 1-2).

These are plain dicts with validation; they describe existing implementations
rather than introducing a registry. Two tasks use them through their adapters.
"""
TASK_FIELDS = ('name', 'observable', 'geometry', 'boundary', 'source', 'splits',
               'reference', 'metric', 'budget', 'dependencies')
OPERATOR_FIELDS = ('name', 'axes', 'dtype', 'transformation', 'support',
                   'parameters', 'dependencies', 'constraints')
AXES = ('(mu,t,z,y,x,3,3)', '(3,z,y,x,3,3)', '(t,z,y,x,4,4,3,3)', '(z,y,x,3,nrhs)')
TRANSFORMS = ('gauge_covariant', 'gauge_invariant', 'spin_scalar')
SUPPORT = ('spacetime_local', 'time_slice', 'point')


def task_spec(**fields):
    missing = [key for key in TASK_FIELDS if key not in fields]
    if missing:
        raise ValueError(f'TaskSpec is missing {missing}')
    geometry = fields['geometry']
    if len(geometry) != 4 or not all(isinstance(x, int) and x > 0 for x in geometry):
        raise ValueError('geometry must be four positive integers')
    splits = fields['splits']
    if not isinstance(splits, dict) or not {'development', 'validation'} <= set(splits):
        raise ValueError('splits must declare development and validation ranges')
    if not isinstance(fields['dependencies'], dict):
        raise ValueError('dependencies must be a mapping of name to hash')
    return dict(fields)


def run_spec(task, protocol_hash, data, **extra):
    if not isinstance(protocol_hash, str) or len(protocol_hash) < 8:
        raise ValueError('protocol_hash is required')
    spec = dict(task=task, protocol_hash=protocol_hash, data=data, **extra)
    if not isinstance(data, dict):
        raise ValueError('data must map split to hash')
    return spec


def operator_contract(**fields):
    missing = [key for key in OPERATOR_FIELDS if key not in fields]
    if missing:
        raise ValueError(f'OperatorContract is missing {missing}')
    if fields['axes'] not in AXES:
        raise ValueError(f'unsupported axes {fields["axes"]!r}')
    if fields['transformation'] not in TRANSFORMS:
        raise ValueError(f'unsupported transformation {fields["transformation"]!r}')
    if fields['support'] not in SUPPORT:
        raise ValueError(f'unsupported support {fields["support"]!r}')
    if fields['dtype'] not in ('complex128', 'float64', 'complex64'):
        raise ValueError(f'unsupported dtype {fields["dtype"]!r}')
    return dict(fields)


def candidate_spec(builder, parameters, contract, dependencies):
    if not isinstance(builder, str) or ':' not in builder:
        raise ValueError('builder must be a "module:function" string')
    if not isinstance(parameters, dict):
        raise ValueError('parameters must be a mapping')
    if contract['name'] not in ('pion-kernel-v1', 'wilson-kernel-v1'):
        raise ValueError(f'unknown operator contract {contract["name"]!r}')
    return {'builder': builder, 'parameters': parameters, 'contract': contract['name'],
            'dependencies': dependencies}
