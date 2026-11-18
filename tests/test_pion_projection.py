import numpy as np
import pytest

from tasks.pion_projection.common import protocol
from tasks.pion_projection.kernels import build_kernel
from tasks.pion_projection.statistics import blocks, bootstrap_indices, fold, risk, shape


def random_unitary(rng, shape_):
    a = rng.normal(size=(*shape_, 3, 3)) + 1j*rng.normal(size=(*shape_, 3, 3))
    q, _ = np.linalg.qr(a)
    q[..., :, 0] /= np.linalg.det(q)[..., None]
    return q


def test_kernel_covariance_linearity_hermiticity_and_translation():
    rng = np.random.default_rng(4)
    u = random_unitary(rng, (3, 3, 3, 3))
    g = random_unitary(rng, (3, 3, 3))
    x = rng.normal(size=(3, 3, 3, 3, 2)) + 1j*rng.normal(size=(3, 3, 3, 3, 2))
    y = rng.normal(size=x.shape) + 1j*rng.normal(size=x.shape)
    args = {'powers': [0, 2, 4], 'weights': [1, -2, .5], 'alpha': .2}
    k = build_kernel(u, args)
    np.testing.assert_allclose(k.apply(2*x-3j*y), 2*k.apply(x)-3j*k.apply(y), atol=1e-13)
    np.testing.assert_allclose(np.vdot(x, k.apply(y)), np.vdot(k.apply(x), y), atol=1e-12)
    ug = np.empty_like(u)
    for mu, axis in enumerate((2, 1, 0)):
        ug[mu] = g @ u[mu] @ np.roll(g, -1, axis=axis).swapaxes(-1, -2).conj()
    gx = g @ x
    np.testing.assert_allclose(build_kernel(ug, args).apply(gx), g @ k.apply(x), atol=1e-13)
    np.testing.assert_allclose(np.vdot(gx, build_kernel(ug, args).apply(gx)), np.vdot(x, k.apply(x)), atol=1e-12)
    np.testing.assert_allclose(build_kernel(np.roll(u, 1, axis=2), args).apply(np.roll(x, 1, axis=1)),
                               np.roll(k.apply(x), 1, axis=1), atol=1e-13)
    # Swap x and z and their link directions (cubic symmetry).
    ur = u[[2, 1, 0]].transpose(0, 3, 2, 1, 4, 5)
    np.testing.assert_allclose(build_kernel(ur, args).apply(x.transpose(2, 1, 0, 3, 4)),
                               k.apply(x).transpose(2, 1, 0, 3, 4), atol=1e-13)
    def reflect(a):
        for axis in range(3):
            a = np.take(a, (-np.arange(3)) % 3, axis=axis)
        return a
    up = np.array([reflect(np.roll(u[mu], 1, axis=axis)).swapaxes(-1, -2).conj()
                   for mu, axis in enumerate((2, 1, 0))])
    np.testing.assert_allclose(build_kernel(up, args).apply(reflect(x)), reflect(k.apply(x)), atol=1e-13)


def synthetic(excited=0.0, noise=0.01):
    rng = np.random.default_rng(9)
    f = shape(.8, np.arange(9))
    mean = f + excited*shape(1.5, np.arange(9))
    return (mean[None, :]+noise*f[None, :]*rng.normal(size=(30, 9)))[:, None, :]


def evaluate_synthetic(data, weight=1.0):
    p = protocol()
    b = blocks(data, 5)
    idx = bootstrap_indices(len(b), 500, 9)
    return risk(b, [weight], .8, np.full(500, .8), idx, p)


def test_risk_scaling_noise_and_excited_state():
    data = synthetic()
    a = evaluate_synthetic(data)
    for w in [-10, .001, 15]:
        assert evaluate_synthetic(data, w)['risk'] == pytest.approx(a['risk'], rel=1e-10)
    assert evaluate_synthetic(synthetic(noise=.1))['variance'] > a['variance']*50
    for amplitude in [2., -0.5]:
        assert evaluate_synthetic(synthetic(excited=amplitude))['departure_squared'] > a['departure_squared']*10


def test_invalid_inputs_and_folding():
    with pytest.raises(ValueError):
        evaluate_synthetic(synthetic(), weight=0)
    with pytest.raises(ValueError):
        evaluate_synthetic(synthetic()*np.nan)
    with pytest.raises(ValueError):
        blocks(np.zeros((16, 1, 9)), 5)
    a = np.arange(16)[None, :].astype(float)
    np.testing.assert_array_equal(fold(a), [[0, 8, 8, 8, 8, 8, 8, 8, 8]])
    with pytest.raises(ValueError):
        build_kernel(np.zeros((4, 3, 3, 3, 3, 3)), {'powers': [0], 'weights': [1], 'alpha': .2})


def test_reference_recovers_synthetic_cosh():
    from tasks.pion_projection.statistics import ReferenceFit
    p = protocol()
    rng = np.random.default_rng(18)
    f = shape(.8, np.arange(9))
    amplitudes = np.arange(1., 8.)
    data = amplitudes[None, :, None]*f[None, None, :]*(1+.005*rng.normal(size=(30, 7, 9)))
    b = blocks(data, 5)
    result, boot = ReferenceFit(b, p, [5, 6, 7, 8]).bootstrap(bootstrap_indices(6, 100, 18))
    assert abs(result['mass']-.8) < .003
    assert np.all(np.isfinite(boot)) and result['se'] > 0


def test_block_bootstrap_degenerate_outer_draw_is_not_dropped():
    from tasks.pion_projection.statistics import inverse_covariance
    p = protocol()
    b = blocks(synthetic(), 5)
    inv = inverse_covariance(b[:, 0, p['amplitude_window']])
    repeated = np.repeat(b[:1], 6, axis=0)
    r = risk(repeated, [1], .8, np.full(100, .8), bootstrap_indices(6, 100, 9), p,
             require_valid=False, late_inverse=inv)
    assert np.isfinite(r['risk'])


def test_development_cannot_reopen_after_validation(tmp_path):
    from tasks.pion_projection.experiment import develop
    (tmp_path/'validation.npz').touch()
    with pytest.raises(ValueError, match='Development is closed'):
        develop(tmp_path)


def test_zero_overlap_rejected_and_missing_rows_not_ignored(tmp_path):
    from tasks.pion_projection.common import fingerprint, load_data
    x = synthetic()
    x -= x.mean(axis=0)
    with pytest.raises(ValueError, match='Amplitude|amplitude'):
        evaluate_synthetic(x)
    np.savez(tmp_path/'development.npz', ids=np.arange(29), correlators=np.ones((29, 7, 16)),
             protocol_hash=fingerprint(protocol()))
    with pytest.raises(ValueError, match='missing or reordered'):
        load_data(tmp_path, 'development')


def test_reference_spec_freezes_kind_and_rejects_tampering():
    from tasks.pion_projection.reference import make_reference, reference_spec, verify_reference
    from tasks.pion_projection.statistics import ReferenceFit
    p = protocol()
    spec = reference_spec('single')
    assert verify_reference(spec) == 'single'
    with pytest.raises(ValueError):
        verify_reference({**spec, 'statistics_sha256': '0' * 64})
    with pytest.raises(ValueError):
        verify_reference(dict(spec, kind='twostate'))
    with pytest.raises(ValueError):
        reference_spec('nope')
    data = blocks(np.repeat(synthetic(), 7, axis=1), 5)
    assert isinstance(make_reference(data, p, [5, 6, 7, 8], 'single'), ReferenceFit)
    from tasks.pion_projection.reference_twostate import TwoStateFit
    assert isinstance(make_reference(data, p, [5, 6, 7, 8], 'twostate'), TwoStateFit)


def test_reflection_odd_is_temporal_extent_general():
    from tasks.pion_projection.experiment import reflection_odd
    for temporal in (16, 32):
        raw = np.arange(temporal, dtype=float)[None, None, :]
        odd = reflection_odd(raw)
        t = np.arange(1, temporal // 2)
        np.testing.assert_array_equal(odd[0, 0], raw[0, 0, t] - raw[0, 0, temporal - t])
        assert odd.shape[-1] == temporal // 2 - 1
    with pytest.raises(ValueError):
        reflection_odd(np.zeros((1, 1, 15)))


def test_two_state_fit_reports_parameters_and_condition():
    from tasks.pion_projection.reference_twostate import TwoStateFit
    from tasks.pion_projection.statistics import shape
    p = protocol()
    rng = np.random.default_rng(21)
    t = np.arange(16)
    data = np.zeros((30, 7, 16))
    for k in range(7):
        data[:, k] = (shape(.8, t) + .3*shape(1.4, t))[None, :]*(1 + .005*rng.normal(size=(30, 1)))
    b = blocks(data, 5)
    fit = TwoStateFit(b, p, [5, 6, 7, 8])
    full = fit.fit_full(fit.samples.mean(0))
    assert abs(full['mass'] - .8) < .02
    assert abs(full['gap'] - .6) < .1
    assert full['condition'] > 0 and np.isfinite(full['condition'])
    assert len(full['amplitudes']) == 2*len(p['reference_basis'])
    result, masses = fit.bootstrap(bootstrap_indices(len(b), 50, 0))
    assert result['mass2'] > result['mass'] and result['gap'] > 0
    assert masses.shape == (50,)


def test_verify_reference_rejects_missing_spec():
    from tasks.pion_projection.reference import verify_reference
    for spec in (None, {}, {'schema': 'other'}):
        with pytest.raises(ValueError, match='legacy|schema'):
            verify_reference(spec)


def score_spec(**overrides):
    spec = dict(schema='pion-score-v1', version='test', estimand='x', baseline_id='local',
                accepted_models=['m'], candidate_ids=['c'], analysis_version='v2', alpha=0.05,
                comparison_family={'candidates': 1, 'models': 1}, min_baseline_risk=1e-6,
                invalid_tolerance=0.05, min_improvement=1.0)
    spec.update(overrides)
    return spec


def test_improvement_certificate_accepts_only_real_improvements():
    from tasks.pion_projection.scoring import improvement_certificate
    rng = np.random.default_rng(3)
    base = np.exp(rng.normal(0, 0.05, 4000))
    good = base * np.exp(-0.5)
    bad = base * np.exp(0.2)
    risks = {'m1': {'local': base, 'good': good, 'bad': bad},
             'm2': {'local': base, 'good': good, 'bad': bad}}
    spec = score_spec(accepted_models=['m1', 'm2'], candidate_ids=['good', 'bad'],
                      comparison_family={'candidates': 2, 'models': 2})
    out = improvement_certificate(risks, spec)
    assert out['status'] == 'certified'
    assert out['candidates']['good']['certified']
    assert not out['candidates']['bad']['certified']
    assert out['candidates']['good']['improvement_lower_bound'] > 1.4


def test_scoring_contract_rejects_bad_inputs():
    from tasks.pion_projection.scoring import improvement_certificate, validate_spec
    base = np.ones(50)
    with pytest.raises(ValueError, match='missing'):
        validate_spec({'schema': 'pion-score-v1'})
    with pytest.raises(ValueError, match='non-empty'):
        validate_spec(score_spec(accepted_models=[]))
    with pytest.raises(ValueError, match='accepted set'):
        improvement_certificate({'unapproved': {'local': base, 'c': base * 0.5}}, score_spec())
    with pytest.raises(ValueError, match='declared set'):
        improvement_certificate({'m': {'local': base, 'other': base * 0.5}}, score_spec())
    with pytest.raises(ValueError, match='one-dimensional'):
        improvement_certificate({'m': {'local': np.ones((50, 1)), 'c': np.ones((50, 1))}}, score_spec())
    with pytest.raises(ValueError, match='different lengths'):
        improvement_certificate({'m': {'local': base, 'c': np.ones(49)}}, score_spec())
    near_zero = improvement_certificate({'m': {'local': np.full(50, 1e-9), 'c': np.full(50, 1e-10)}}, score_spec())
    assert near_zero['abstain'] and not near_zero['candidates']['c']['certified']
    invalid = improvement_certificate({'m': {'local': base, 'c': np.full(50, np.nan)}}, score_spec())
    assert invalid['abstain'] and not invalid['candidates']['c']['certified']
    negative = improvement_certificate({'m': {'local': base, 'c': np.full(50, -0.1)}}, score_spec(risk_floor=1.0))
    assert negative['abstain'] and not negative['candidates']['c']['certified']
    weak = improvement_certificate({'m': {'local': base, 'c': base * 0.5}}, score_spec(min_improvement=3.0))
    assert not weak['candidates']['c']['certified']
    with pytest.raises(ValueError, match='requires explicit draw_ids'):
        improvement_certificate({'m': {'local': base, 'c': base * 0.5}},
                                score_spec(require_draw_ids=True))


def test_candidate_dependency_check(tmp_path, monkeypatch):
    from tasks.pion_projection.candidate import check_dependencies
    from tasks.pion_projection.common import digest
    helper = tmp_path / 'helper.py'
    helper.write_text('x = 1\n')
    monkeypatch.chdir(tmp_path)
    good = {'dependencies': {'helper.py': digest(helper)}}
    assert check_dependencies(good) == good['dependencies']
    helper.write_text('x = 2\n')
    with pytest.raises(ValueError, match='helper.py changed'):
        check_dependencies(good)
    assert check_dependencies({}) == {}


def test_gevp_recovers_a_two_state_matrix():
    from tasks.pion_projection.gevp import gevp_energies, summary, synthetic
    data = synthetic(nblocks=200, n_ops=3, m0=0.8, m1=1.4, seed=3, noise=0.01)
    rows = gevp_energies(data, t0=2, times=[5, 6, 7])
    result = summary(rows, state=0)
    assert abs(result['energy'] - 0.8) < 0.02
    assert result['se_boot'] > 0
    excited = summary(rows, state=1)
    assert abs(excited['energy'] - 1.4) < 0.1


def test_gevp_rejects_bad_matrices():
    from tasks.pion_projection.gevp import gevp_energies, synthetic
    data = synthetic(nblocks=4, n_ops=2, seed=4)
    with pytest.raises(ValueError, match='Hermitian'):
        gevp_energies(data * (1 + 0.1j), t0=2, times=[5])
    singular = data.copy()
    singular[..., 2] = 0.0
    with pytest.raises(ValueError, match='positive definite'):
        gevp_energies(singular, t0=2, times=[5])
    with pytest.raises(ValueError, match='nblocks'):
        gevp_energies(data[0], t0=2, times=[5])


def test_energy_delta_sign_and_validation():
    from tasks.pion_projection.energy_target import energy_delta, lower_bound
    rng = np.random.default_rng(0)
    anchor = np.full(500, 0.8)
    good = 0.8 + 0.005 * rng.normal(size=500)
    bad = 0.8 + 0.02 * rng.normal(size=500)
    assert np.median(energy_delta(bad, good, anchor)) > 0
    assert np.median(energy_delta(good, bad, anchor)) < 0
    with pytest.raises(ValueError, match='same shape'):
        energy_delta(np.zeros(3), np.zeros(4), np.zeros(3))
    with pytest.raises(ValueError, match='tau_e'):
        energy_delta(np.zeros(10), np.zeros(10), np.zeros(10), tau_e=0)
    with pytest.raises(ValueError, match='ten'):
        lower_bound(np.zeros(5))


def test_energy_target_certificate_behaviour():
    from tasks.pion_projection.energy_target import coverage
    better = coverage(n_rep=60, n_draws=100, anchor_sigma=0.005, bias=(0.0, 0.0),
                      sigma=(0.02, 0.005), tau_e=0.01, seed=1)
    same = coverage(n_rep=60, n_draws=100, anchor_sigma=0.005, bias=(0.0, 0.0),
                    sigma=(0.01, 0.01), tau_e=0.01, seed=2)
    worse = coverage(n_rep=60, n_draws=100, anchor_sigma=0.005, bias=(0.0, 0.02),
                     sigma=(0.01, 0.005), tau_e=0.01, seed=3)
    assert better['certified_fraction'] > 0.9
    assert same['certified_fraction'] < 0.15
    assert worse['certified_fraction'] < 0.1
    assert better['median_delta'] > 0 > worse['median_delta']


def test_hankel_matrix_uses_sum_channels():
    from tasks.pion_projection.hankel_gevp import build_hankel
    powers = [0, 2, 4, 8, 16, 32, 64]
    data = np.arange(2 * 7 * 5, dtype=float).reshape(2, 7, 5)
    matrix = build_hankel(data, powers, ops=(0, 8))
    assert matrix.shape == (2, 2, 2, 5)
    np.testing.assert_array_equal(matrix[:, 0, 0], data[:, 0])
    np.testing.assert_array_equal(matrix[:, 0, 1], data[:, 3])
    np.testing.assert_array_equal(matrix[:, 1, 1], data[:, 4])
    with pytest.raises(ValueError, match='not measured'):
        build_hankel(data, powers, ops=(8, 16))


def test_multisource_matrix_and_regularization():
    from tasks.pion_projection.multisource_gevp import build_matrix, regularize
    data = np.zeros((4, 3, 8, 16))
    for j in range(3):
        for k in range(8):
            data[:, j, k] = (j + 1) * (k + 1) * np.exp(-0.8 * np.arange(16))[None]
    matrix = build_matrix(data, [0, 2, 4, 8, 16, 24, 32, 64])
    assert matrix.shape == (4, 3, 3, 16)
    np.testing.assert_allclose(matrix, matrix.swapaxes(1, 2))
    bad = matrix.copy()
    bad[:, 0, 0, 2] = -1.0
    fixed = regularize(bad)
    values = np.linalg.eigvalsh(fixed[:, :, :, 2])
    assert np.all(values > 0)
    with pytest.raises(ValueError, match='match'):
        build_matrix(data, [0, 2, 4, 8, 16, 24, 32, 64], sinks=(0, 4))


def test_gram_gevp_floor_makes_positive_semidefinite():
    from tasks.pion_projection.gram_gevp import floored
    rng = np.random.default_rng(0)
    gram = np.zeros((3, 4, 4, 16))
    for t in range(16):
        a = rng.normal(size=(4, 4))
        gram[:, :, :, t] = a @ a.T
    gram[0, 0, 0, 5] = -1.0
    fixed = floored(gram)
    values = np.linalg.eigvalsh(fixed.transpose(0, 3, 1, 2).reshape(-1, 4, 4))
    assert np.all(values >= 0)


def test_agent_prompt_follows_the_protocol():
    from agents.loop import render_prompt
    p = protocol()
    q = dict(p, geometry=[16, 16, 16, 32], powers=[0, 2, 4], alpha=0.1, score_window=[3, 4, 5])
    text = render_prompt(q, [[0, 1.0, 0.0]], [])
    assert '32-t' in text and '0.1' in text and '[0, 2, 4]' in text
    assert '3, 4, 5' in text and '[[3 u values], [3 u values], [3 u values]]' in text


def test_gevp_dedupe_and_known_spectrum():
    from tasks.pion_projection.gevp_stats import bootstrap_gevp, dedupe, summarise, synthetic_correlators
    rng = np.random.default_rng(23)
    base = synthetic_correlators(rng, ncfg=60, noise=0.0)
    matrix, keep = dedupe(np.concatenate([base, base[:, :1]], axis=1))
    assert len(keep) == base.shape[1]
    data = synthetic_correlators(rng, ncfg=60, noise=0.01)
    index = rng.integers(0, 60, (200, 60))
    energies, failures = bootstrap_gevp(data, index, t0=2, times=(4, 5, 6), rank=2)
    summary = summarise(energies, (4, 5, 6), 0)
    assert summary['energy'] == pytest.approx(0.80, abs=0.01)
    assert summary['se'] > 0 and not failures


def test_joint_draws_share_ids_and_abstain_on_failed_gate(tmp_path):
    import json
    from pathlib import Path
    from tasks.pion_projection.joint import run
    run_dir = 'results/pion_projection/s16t16_gevp200'
    if not Path(run_dir, 'validation_results.json').exists():
        pytest.skip('run data not available')
    certificate = run(run_dir, models=('single',), outer=4, inner=4, dest=tmp_path)
    draws = np.load(tmp_path / 'joint_draws.npz')
    assert draws['draw_id'].tolist() == [0, 1, 2, 3]
    assert 'risk_single_local' in draws
    assert draws['dev_inner'].shape[0] == 4 and draws['val_inner'].shape[0] == 4
    manifest = json.loads((tmp_path / 'joint_manifest.json').read_text())
    assert manifest['require_draw_ids'] if False else manifest['spec']['require_draw_ids']
    assert certificate['status'] == 'abstain'
    assert set(certificate['failure_reasons']) == {'gate_artifact_missing', 'reference_gate_failed'}
    assert not any(row['certified'] for row in certificate['candidates'].values())


def test_agent_mock_rejects_illegal_candidates_without_reading_validation(tmp_path, monkeypatch):
    import json
    import shutil
    from pathlib import Path
    run_dir = Path('results/pion_projection/s16t16_gevp200')
    if not (run_dir / 'submission.json').exists():
        pytest.skip('run data not available')
    for name in ('development.npz', 'development_provenance.json', 'submission.json'):
        shutil.copy(run_dir / name, tmp_path / name)
    import agents.llm as llm
    from agents.loop import run
    llm.MOCK_QUEUE[:] = [
        json.dumps({'candidates': [[float('nan')] * 8, [1, 2, 3], [0.0] * 8, [5.0, -4.0] + [0.0] * 6]}),
        json.dumps({'candidates': [[1.0] + [0.0] * 7]}),
    ]
    original_load = np.load
    def guarded_load(path, *args, **kwargs):
        if 'validation' in str(path):
            raise AssertionError('validation was opened during search')
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', guarded_load)
    result = run(tmp_path, 'mock', 1, allow_validation=False)
    assert result['validation_status'] == 'not_read'
    reasons = json.loads((tmp_path / 'agent' / 'rejections.json').read_text())
    text = ' '.join(item['reason'] for item in reasons)
    assert 'non-finite' in text and 'length 8' in text and 'sum to 1' in text and 'L1' in text


def test_measurement_plan_schema():
    from tasks.pion_projection.measurement_plan import load_plan, validate_plan
    plan = load_plan('tasks/pion_projection/measurement_plan.example.json')
    assert plan['status'] == 'NOT_RUN' and plan['sources']
    with pytest.raises(ValueError, match='missing'):
        validate_plan({'schema': 'pion-measurement-plan-v1'})
    broken = dict(plan, identity={})
    with pytest.raises(ValueError, match='identity'):
        validate_plan(broken)
    with pytest.raises(ValueError, match='status'):
        validate_plan(dict(plan, status='MAYBE'))


def test_pion_develop_records_run_state(tmp_path, monkeypatch):
    import json
    import shutil
    from pathlib import Path
    from tasks.pion_projection import common, experiment
    run = Path('results/pion_projection/s16t16_v1')
    if not (run / 'development.npz').exists():
        pytest.skip('v1 development data not available')
    shutil.copy(run / 'development.npz', tmp_path / 'development.npz')
    shutil.copy(run / 'development_provenance.json', tmp_path / 'development_provenance.json')
    protocol = json.loads((run / 'protocol.json').read_text())
    monkeypatch.setattr(common, 'protocol', lambda: protocol)
    monkeypatch.setattr(experiment, 'protocol', lambda: protocol)
    experiment.develop(tmp_path)
    frozen = json.loads((tmp_path / 'submission.json').read_text())
    statuses = frozen['run_state']['status']
    assert statuses['operator'] == 'validated' and statuses['numerical'] == 'validated'
    assert statuses['reference'] in ('validated', 'failed')
    assert statuses['statistical'] == 'validated'
    artifact = frozen['gate_artifact']
    assert artifact['model'] == 'single'
    assert artifact['development_sha256'] == frozen['development_sha256']
    assert artifact['protocol_hash'] == frozen['protocol_hash']
    assert artifact['window'] == protocol['reference_windows'][protocol['reference_nominal_window']]


def test_joint_rejects_a_borrowed_gate(tmp_path):
    from pathlib import Path
    from tasks.pion_projection.joint import run
    run_dir = 'results/pion_projection/s16t16_twostate'
    if not (Path(run_dir) / 'submission.json').exists():
        pytest.skip('twostate run not available')
    with pytest.raises(ValueError, match='no frozen reference kind|cannot be borrowed'):
        run(run_dir, models=('single',), outer=2, inner=2, dest=tmp_path)


def test_gevp_summary_rejects_partial_plateaus():
    from tasks.pion_projection.gevp_stats import summarise
    energies = np.full((3, 2, 3), 1.0)   # times x modes x replicas
    energies[2, 0, 0] = np.nan
    energies[1, 0, 1] = np.nan
    summary = summarise(energies, (4, 5, 6), mode=0)
    assert summary['replicas'] == 1
    assert summary['failures'] == 2
    assert summary['local_failures'] == 2
    assert summary['valid_times_per_replica'] == [2, 2, 3]


def test_reference_gate_dof_and_null_pvalue():
    from tasks.pion_projection.reference_gate import fit_gate, generate, null_pvalue
    rng = np.random.default_rng(5)
    data = generate(rng, kind='single', ncfg=60, noise=0.01)
    fit, mass, cost, dof = fit_gate(data)
    assert dof == fit.samples.shape[1] - len(__import__('tasks.pion_projection.reference_gate', fromlist=['BASIS']).BASIS) - 1
    p, null_costs = null_pvalue(fit, mass, cost, data.mean(axis=0), rng, n_null=5)
    assert 0 < p <= 1 and len(null_costs) == 5


def test_certificate_calibration_chain():
    from tasks.pion_projection.certificate_calibration import run_simulation
    rng = np.random.default_rng(3)
    flat = np.zeros(7)
    same = flat.copy(); same[1] = same[2] = 0.5
    result = run_simulation(rng, same, same, outer=4, inner=4)
    assert 'abstain' in result
    if not result['abstain']:
        assert 'certified' in result and 'lower_bound' in result


def test_anchor_calibration_cell():
    from tasks.pion_projection.anchor_calibration import run_cell
    rng = np.random.default_rng(4)
    row = run_cell(rng, 0.02, 0.01, 0.0, n_dataset=3, n_boot=12)
    assert 0 <= row['naive_certified_fraction'] <= 1
    assert np.isfinite(row['dataset_mean']) and row['dataset_se'] >= 0


def test_verify_run_identity_rejects_tampering(tmp_path):
    import json
    import shutil
    from pathlib import Path
    from tasks.pion_projection.identity import verify_run_identity
    run = Path('results/pion_projection/s16t16_gevp200')
    if not (run / 'submission.json').exists():
        pytest.skip('run data not available')
    for name in ('development.npz', 'development_provenance.json', 'submission.json', 'validation_results.json'):
        shutil.copy(run / name, tmp_path / name)
    kind, reasons = verify_run_identity(tmp_path)
    assert kind == 'single' and not reasons
    sub = json.loads((tmp_path / 'submission.json').read_text())
    sub['development_sha256'] = '0' * 64
    (tmp_path / 'submission.json').write_text(json.dumps(sub))
    kind, reasons = verify_run_identity(tmp_path)
    assert any('development data hash' in reason for reason in reasons)


def test_draw_ids_reject_duplicates_and_reordering():
    from tasks.pion_projection.scoring import improvement_certificate
    base = np.ones(50)
    risks = {'m': {'local': base, 'c': base * 0.5}}
    spec = score_spec(require_draw_ids=True)
    ids = np.arange(50)
    out = improvement_certificate(risks, spec, draw_ids=ids)
    assert out['candidates']['c']['certified']
    with pytest.raises(ValueError, match='duplicates'):
        improvement_certificate(risks, spec, draw_ids=np.zeros(50, dtype=int))
    with pytest.raises(ValueError, match='reordered'):
        improvement_certificate(risks, spec, draw_ids=ids,
                                draw_ids_by_name={'c': ids[::-1]})


def test_evaluate_reuses_a_compatible_joint_certificate(tmp_path):
    import json
    import shutil
    from pathlib import Path
    from tasks.pion_projection import common, experiment
    from tasks.pion_projection.joint import run as joint_run
    run_dir = Path('results/pion_projection/s16t16_gevp200')
    if not (run_dir / 'submission.json').exists():
        pytest.skip('run data not available')
    for name in ('development.npz', 'validation.npz', 'development_provenance.json',
                 'validation_provenance.json', 'submission.json'):
        shutil.copy(run_dir / name, tmp_path / name)
    protocol = json.loads((run_dir / 'development_provenance.json').read_text())['protocol']
    common.protocol = lambda: protocol
    experiment.protocol = lambda: protocol
    experiment.evaluate(tmp_path)                      # first pass writes the report
    joint_run(tmp_path, models=('single',), outer=4, inner=4, dest=tmp_path / 'joint')
    experiment.evaluate(tmp_path)                      # second pass can reuse the certificate
    report = json.loads((tmp_path / 'validation_results.json').read_text())
    assert report['certification_source'] == 'joint'
    assert report['certification']['status'] == 'abstain'


def test_certificate_family_j2_k2():
    from tasks.pion_projection.certificate_calibration import run_family
    rng = np.random.default_rng(6)
    row = run_family(rng, n_dataset=2, outer=12, inner=8)
    assert row['n_dataset'] == 2 and row['valid'] >= 0
    assert 0 <= row['certified_null'] <= row['valid']


def test_certificate_wrong_excited_state():
    from tasks.pion_projection.certificate_calibration import run_wrong_state
    rng = np.random.default_rng(7)
    row = run_wrong_state(rng, n_dataset=2, outer=12, inner=8)
    assert row['gate_pass'] >= 1
    assert row['wrong_mass_gt_0.1'] >= 1
