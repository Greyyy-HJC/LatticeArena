import inspect
import json
import sys

import numpy as np
import pytest

from core.io import digest, fingerprint, write_json
from tasks.pion_projection.__main__ import main
from tasks.pion_projection import common
from tasks.pion_projection.common import protocol
from tasks.pion_projection.spec import bundle_for, operator_contract_for, task_spec_for
from tasks.pion_projection.statistics import shape


def make_split(output, p, split):
    ids = np.arange(p[split][0], p[split][1] + 1)
    t = np.arange(16)
    rng = np.random.default_rng(int(ids[0]) + 35)
    means = np.arange(1, 8)[:, None] * shape(.8, t, 16)
    values = means * (1 + .01 * rng.normal(size=(len(ids), 7, 16)))
    np.savez(output / f'{split}.npz', ids=ids, correlators=values, protocol_hash=fingerprint(p))
    write_json(output / f'{split}_provenance.json', {'protocol': p})


def test_agent_cli_uses_development_and_preserves_old_freeze(tmp_path, monkeypatch):
    from agents import loop

    p = dict(protocol(), bootstrap_samples=32)
    make_split(tmp_path, p, 'development')
    frozen = {'protocol_hash': fingerprint(p), 'development_sha256': digest(tmp_path / 'development.npz'),
              'statistics_sha256': digest(inspect.getfile(loop.ReferenceFit))}
    write_json(tmp_path / 'submission.json', frozen)
    good = [1, 0, 0, 0, 0, 0, 0]
    replies = iter(['bad JSON', json.dumps({'candidates': [['x'], good, [0] * 7, good]})])
    monkeypatch.setattr(loop, 'chat', lambda *_: next(replies))
    args = ['pion', '--output', str(tmp_path), 'agent', '--space', 'basis', '--rounds', '2', '--model', 'stub']
    monkeypatch.setattr(sys, 'argv', args)
    main()
    folder, = (tmp_path / 'agent').iterdir()
    summary = json.loads((folder / 'best.json').read_text())
    assert summary['validation_status'] == 'not_read' and summary['best']['candidate'] == good
    first, second = json.loads((folder / 'rounds.json').read_text())
    assert first['error'] and second['over_budget']
    assert len(second['rejections']) == 2
    make_split(tmp_path, p, 'validation')
    replies = iter([json.dumps({'candidates': [good]})] * 2)
    monkeypatch.setattr(sys, 'argv', args + ['--with-validation'])
    main()
    second_folder, = set((tmp_path / 'agent').iterdir()) - {folder}
    summary = json.loads((second_folder / 'best.json').read_text())
    assert summary['validation_status'] == 'post_hoc_diagnostic' and summary['validation_risk'] > 0
    assert json.loads((tmp_path / 'submission.json').read_text()) == frozen


def test_pion_measurement_bundle_checks_split_identity(tmp_path):
    p = protocol()
    for split in ('development', 'validation'):
        make_split(tmp_path, p, split)
    result = bundle_for(tmp_path)
    assert result['capabilities']['linear_response']
    task = task_spec_for(p)
    assert task['source']['position'] == p['source']
    assert 'core/npz_safety.py' in task['dependencies']
    assert 'core/numerics.py' in task['dependencies']
    assert 'core/source_integrity.py' in task['dependencies']
    assert 'core/specs.py' in task['dependencies']
    assert 'tasks/pion_projection/common.py' in task['dependencies']
    assert 'tasks/pion_projection/candidate.py' in task['dependencies']
    assert 'tasks/pion_projection/path_kernel.py' in task['dependencies']
    assert 'tasks/pion_projection/evidence_adapter.py' in task['dependencies']
    assert 'tasks/pion_projection/anchor_scalar_reader.py' in task['dependencies']
    assert 'tasks/pion_projection/bounded_block_risk_contract.py' in task['dependencies']
    assert 'tasks/pion_projection/bounded_block_risk_contract_v1.json' in task['dependencies']
    assert 'tasks/pion_projection/bounded_block_risk_score.py' in task['dependencies']
    assert 'tasks/pion_projection/bounded_block_risk_anchor_markov.py' in task['dependencies']
    assert 'tasks/pion_projection/bounded_block_risk_markov_gap.py' in task['dependencies']
    assert 'core/candidates.py' in task['dependencies']
    assert 'core/evidence.py' in task['dependencies']
    assert 'core/receipts.py' in task['dependencies']
    assert 'core/state.py' in task['dependencies']
    assert operator_contract_for()['axes'] == '(z,y,x,3,nrhs)'
    z = tmp_path / 'validation.npz'
    with np.load(z) as data:
        values = dict(data)
    np.savez(z, **dict(values, ids=values['ids'][::-1]))
    with pytest.raises(ValueError, match='reordered'):
        bundle_for(tmp_path)


def test_path_measurement_reader_preflights_npz_before_numpy_load(tmp_path, monkeypatch):
    (tmp_path / 'development.npz').write_bytes(b'captured only for preflight test')

    def reject_payload(_payload):
        raise ValueError('NPZ preflight stopped before allocation')

    def numpy_load_must_not_run(*args, **kwargs):
        raise AssertionError('numpy.load ran before NPZ preflight')

    monkeypatch.setattr(common, 'validate_npz_payload', reject_payload)
    monkeypatch.setattr(common.np, 'load', numpy_load_must_not_run)
    with pytest.raises(ValueError, match='NPZ preflight stopped before allocation'):
        common.load_data(tmp_path, 'development', protocol())


def test_path_measurement_reader_bounds_compressed_bytes_before_npz_preflight(tmp_path, monkeypatch):
    (tmp_path / 'development.npz').write_bytes(b'x' * 64)
    monkeypatch.setattr(common, '_MAX_COMPRESSED_NPZ_BYTES', 8)

    def npz_preflight_must_not_run(_payload):
        raise AssertionError('NPZ parser ran before compressed-size rejection')

    monkeypatch.setattr(common, 'validate_npz_payload', npz_preflight_must_not_run)
    with pytest.raises(ValueError, match='NPZ compressed size limit exceeded'):
        common.load_data(tmp_path, 'development', protocol())


def test_report_cli_uses_recorded_risk(tmp_path, monkeypatch):
    from agents import report

    data = {'results': {'invalid': {'invalid_reason': 'amplitude'},
                        'local': {'risk': 0.5, 'variance': 0.2, 'departure_squared': 0.3}},
            'reference_passed': False, 'reference_failure_reasons': ['unstable'],
            'score_status': 'unscored_reference_failed'}
    write_json(tmp_path / 'validation_results.json', data)
    monkeypatch.setattr(report, 'chat', lambda *_: '{"markdown": "model commentary"}')
    monkeypatch.setattr(sys, 'argv', ['pion', '--output', str(tmp_path), 'report', '--model', 'stub'])
    main()
    text = (tmp_path / 'REPORT.md').read_text()
    assert '| 1 | local | 0.5 |' in text and 'model commentary' in text and 'unstable' in text
