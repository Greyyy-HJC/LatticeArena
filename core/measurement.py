"""Measurement bundle and capability declarations (todo F1, slice 3).

A bundle is arrays plus a manifest plus explicit capabilities. Asking for a
capability the bundle does not declare is an error, never a silent fallback.
"""
import numpy as np

CAPABILITIES = ('linear_response', 'cross_matrix', 'new_source')


class UnsupportedCapability(ValueError):
    pass


def bundle(task, arrays, manifest, capabilities, dtype='float64'):
    if not isinstance(arrays, dict) or not arrays:
        raise ValueError('arrays must be a non-empty mapping')
    if not isinstance(manifest, dict):
        raise ValueError('manifest must be a mapping')
    for key in CAPABILITIES:
        if key not in capabilities or not isinstance(capabilities[key], bool):
            raise ValueError(f'capability {key!r} must be declared as a bool')
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.size == 0:
            raise ValueError(f'array {name!r} is empty')
        if not np.all(np.isfinite(array)):
            raise ValueError(f'array {name!r} has non-finite entries')
    return {'task': task, 'arrays': arrays, 'manifest': manifest,
            'capabilities': dict(capabilities), 'dtype': dtype}


def require(bundle_, *names):
    missing = [name for name in names if not bundle_['capabilities'].get(name)]
    if missing:
        raise UnsupportedCapability(f'{bundle_["task"]} does not declare {missing}; '
                                    'do not infer it from a diagonal/single-source cache')
    return bundle_


def describe(bundle_):
    return {'task': bundle_['task'],
            'splits': sorted(bundle_['manifest'].get('splits', {})),
            'capabilities': bundle_['capabilities'],
            'shapes': {name: list(np.asarray(value).shape) for name, value in bundle_['arrays'].items()}}
