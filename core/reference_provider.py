"""ReferenceProvider: one entry point for the task's reference estimate (F1 slice 4).

The provider declares which kinds a task supports and returns fit draws; it does
not silently fall back to another kind.
"""
KINDS = {'pion_projection': ('single', 'twostate'), 'wilson_loop': ('identity',)}


def make_provider(task, kind, **params):
    if task not in KINDS:
        raise ValueError(f'unknown task {task!r}')
    if kind not in KINDS[task]:
        raise ValueError(f'{task} does not support reference kind {kind!r}')
    return {'task': task, 'kind': kind, 'params': params}


def provide(provider, data, index=None, seed=None):
    task = provider['task']
    params = provider['params']
    if task == 'pion_projection':
        from tasks.pion_projection.reference import make_reference
        reference = make_reference(data, params['protocol'], params['window'], provider['kind'])
        if index is None:
            raise ValueError('pion reference needs bootstrap indices')
        fit, masses = reference.bootstrap(index)
        return {'fit': fit, 'masses': masses}
    if task == 'wilson_loop':
        from tasks.wilson_loop.statistics import potential
        kernel = int(params.get('kernel', 0))
        value = potential(data[:, kernel].mean(0), params['window'], params['t_values'])
        return {'v_ref': value}
    raise ValueError(f'unknown task {task!r}')
