import json
from pathlib import Path

import numpy as np
import pytest

from core.specs import candidate_spec, operator_contract, task_spec


def test_task_spec_validation():
    with pytest.raises(ValueError, match='missing'):
        task_spec(name='x')
    good = dict(name='x', observable='y', geometry=[16, 16, 16, 16],
                boundary={'spatial': 'periodic'}, source={'kind': 'point'},
                splits={'development': [0, 9], 'validation': [10, 19]},
                reference={'kind': 'single'}, metric={'name': 'risk'},
                budget={'outer': 10}, dependencies={})
    task_spec(**good)
    with pytest.raises(ValueError, match='geometry'):
        task_spec(**dict(good, geometry=[16, 16]))
    with pytest.raises(ValueError, match='splits'):
        task_spec(**dict(good, splits={'development': [0, 9]}))


def test_operator_and_candidate_contracts():
    contract = operator_contract(name='pion-kernel-v1', axes='(z,y,x,3,nrhs)', dtype='complex128',
                                 transformation='spin_scalar', support='time_slice',
                                 parameters={}, dependencies={}, constraints={})
    with pytest.raises(ValueError, match='axes'):
        operator_contract(**dict(contract, axes='(bad)'))
    with pytest.raises(ValueError, match='transformation'):
        operator_contract(**dict(contract, transformation='nope'))
    spec = candidate_spec('tasks.pion_projection.kernels:build_kernel', {'alpha': 0.2}, contract, {})
    assert spec['contract'] == 'pion-kernel-v1'
    with pytest.raises(ValueError, match='builder'):
        candidate_spec('no-colon', {}, contract, {})


def test_task_adapters_match_protocols():
    from tasks.pion_projection.spec import operator_contract_for as pion_op
    from tasks.pion_projection.spec import task_spec_for as pion_spec
    from tasks.wilson_loop.spec import operator_contract_for as wilson_op
    from tasks.wilson_loop.spec import task_spec_for as wilson_spec
    p = json.loads(Path('tasks/pion_projection/protocol_gevp200.json').read_text())
    w = json.loads(Path('tasks/wilson_loop/protocol_v3.json').read_text())
    ps, ws = pion_spec(p), wilson_spec(w)
    assert ps['geometry'] == p['geometry'] and ws['geometry'] == w['geometry']
    assert ps['splits']['validation'] == p['validation']
    assert ws['splits']['validation'] == w['validation']
    assert pion_op()['axes'] == '(z,y,x,3,nrhs)'
    assert wilson_op()['axes'] == '(mu,t,z,y,x,3,3)'
    assert ps['dependencies']['kernels'] and ws['dependencies']['kernels']


def test_measurement_bundle_capabilities():
    from core.measurement import UnsupportedCapability, bundle, describe, require
    b = bundle('toy', {'x': np.ones((2, 3))}, {'splits': {'development': {}}},
               {'linear_response': True, 'cross_matrix': False, 'new_source': False})
    require(b, 'linear_response')
    with pytest.raises(UnsupportedCapability):
        require(b, 'cross_matrix')
    with pytest.raises(ValueError, match='capability'):
        bundle('toy', {'x': np.ones(3)}, {}, {'linear_response': True})
    assert describe(b)['shapes']['x'] == [2, 3]


def test_bundles_from_real_runs():
    from core.measurement import describe
    from tasks.pion_projection.spec import bundle_for as pion_bundle
    from tasks.wilson_loop.spec import bundle_for as wilson_bundle
    pb = pion_bundle('results/pion_projection/s16t16_gevp200')
    assert pb['arrays']['development'].ndim == 3
    assert describe(pb)['capabilities']['linear_response']
    wb = wilson_bundle('results/wilson_loop/s16t16_v3')
    assert wb['arrays']['validation'].ndim == 4
    assert not describe(wb)['capabilities']['linear_response']


def test_comparator_shares_draws():
    from core.comparator import check_draws, log_lower_bound, paired_ratio
    base = np.linspace(1.0, 2.0, 50)
    cand = np.linspace(0.5, 1.0, 50)
    result = paired_ratio(base, cand)
    assert result['median'] == pytest.approx(2.0) and result['draws'] == 50
    with pytest.raises(ValueError, match='different lengths'):
        check_draws({'a': np.ones(5), 'b': np.ones(6)})
    low, invalid = log_lower_bound(base, cand, 0.025)
    assert low > 0 and invalid == 0


def test_search_adapter_budget_and_rejections():
    from core.search import Budget, search
    proposals = [('a', 1.0), ('b', -1.0), ('c', 2.0), ('d', 3.0)]
    result = search(proposals, evaluate=lambda x: x,
                    validate=lambda x: (x > 0, 'must be positive'),
                    budget=Budget(3))
    assert [item['candidate'] for item in result.accepted] == [1.0, 2.0]
    assert result.rejected[0]['reason'] == 'must be positive'
    assert result.budget_exhausted and result.evaluations == 2

    broken = search([('x', 1.0)], evaluate=lambda x: (_ for _ in ()).throw(ValueError('bad')))
    assert broken.rejected and 'bad' in broken.rejected[0]['reason']
    assert broken.best() is None


def test_run_state_transitions_and_score_gating():
    from core.state import RunState
    state = RunState()
    with pytest.raises(ValueError, match='illegal transition'):
        state.set('statistical', 'validated', evidence_hash='h' * 12)
    with pytest.raises(ValueError, match='bound evidence hash'):
        state.set('operator', 'draft').set('operator', 'validated')
    state = RunState()
    for name in ('operator', 'numerical', 'statistical'):
        state.set(name, 'draft').set(name, 'validated', evidence_hash='h' * 12)
    assert state.score_or_null() is None
    state.set('reference', 'draft').set('reference', 'validated', evidence_hash='h' * 12)
    state.set('comparison', 'draft')
    state.score = 42.0
    assert state.score_or_null() is None       # comparison still draft
    state.set('comparison', 'validated', evidence_hash='h' * 12)
    assert state.score_or_null() == 42.0
    state.observe_validation('v3')
    assert state.validation_observed


def test_cost_ledger_phases_and_environment():
    from core.ledger import Ledger
    ledger = Ledger()
    ledger.add('setup', 1.0).add('solve', 2.0, cache='warm').add('analysis', 0.5)
    summary = ledger.summary()
    assert summary['total_seconds'] == 3.5
    assert summary['phases']['solve'] == 2.0
    with pytest.raises(ValueError, match='unknown phase'):
        ledger.add('magic', 1.0)
    with pytest.raises(ValueError, match='different devices'):
        ledger.merge(Ledger(device='dcu'))


def test_reference_provider_both_tasks():
    from core.reference_provider import make_provider, provide
    with pytest.raises(ValueError, match='unknown task'):
        make_provider('nope', 'single')
    with pytest.raises(ValueError, match='does not support'):
        make_provider('pion_projection', 'identity')
    p = json.loads(Path('results/pion_projection/s16t16_gevp200/development_provenance.json').read_text())['protocol']
    from tasks.pion_projection.statistics import blocks, fold
    raw = np.load('results/pion_projection/s16t16_gevp200/development.npz')['correlators']
    data = blocks(fold(raw), p['block_length'])
    provider = make_provider('pion_projection', 'single', protocol=p,
                             window=p['reference_windows'][p['reference_nominal_window']])
    out = provide(provider, data, index=np.zeros((4, len(data)), dtype=int))
    assert 'mass' in out['fit'] and out['masses'].shape == (4,)

    w = json.loads(Path('tasks/wilson_loop/protocol_v3.json').read_text())
    wp = make_provider('wilson_loop', 'identity',
                       window=w['reference_windows'][w['reference_nominal_window']],
                       t_values=w['t_values'], kernel=0)
    wdata = blocks(np.load('results/wilson_loop/s16t16_v3/validation.npz')['loops'], 5)
    v_ref = provide(wp, wdata)['v_ref']
    assert v_ref.shape == (len(w['r_values']),)


def test_check_draws_rejects_broadcast_and_nonfinite():
    from core.comparator import check_draws, paired_ratio
    with pytest.raises(ValueError, match='one-dimensional'):
        check_draws({'a': np.zeros((3, 1)), 'b': np.zeros(3)})
    with pytest.raises(ValueError, match='one-dimensional'):
        check_draws({'a': np.zeros(3), 'b': np.zeros((3, 1))})
    with pytest.raises(ValueError, match='non-finite'):
        check_draws({'a': np.array([1.0, np.nan, 2.0]), 'b': np.ones(3)})
    with pytest.raises(ValueError, match='different lengths'):
        check_draws({'a': np.zeros(3), 'b': np.zeros(4)})
    n, _ = check_draws({'a': np.zeros(3), 'b': np.ones(3)})
    assert n == 3
    with pytest.raises(ValueError, match='one-dimensional'):
        paired_ratio(np.zeros((3, 1)), np.zeros(3))


def test_search_rejects_nonfinite_evaluation():
    from core.search import search
    result = search([('a', 1.0), ('b', 2.0)], evaluate=lambda x: float('nan'))
    assert not result.accepted and len(result.rejected) == 2
    assert all('non-finite' in item['reason'] for item in result.rejected)


def test_bundle_uses_trusted_loading(tmp_path):
    import json
    import shutil
    import numpy as np
    from tasks.wilson_loop.spec import bundle_for
    run = Path('results/wilson_loop/s16t16_v3')
    if not (run / 'development.npz').exists():
        pytest.skip('v3 data not available')
    for name in ('development.npz', 'development_provenance.json', 'validation_provenance.json'):
        shutil.copy(run / name, tmp_path / name)
    with np.load(tmp_path / 'development.npz') as f:
        ids, loops = f['ids'], f['loops']
    np.savez(tmp_path / 'development.npz', ids=ids, loops=loops, protocol_hash=np.asarray('0' * 64))
    with pytest.raises(ValueError, match='Protocol changed'):
        bundle_for(tmp_path)
