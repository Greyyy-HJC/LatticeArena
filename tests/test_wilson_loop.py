import numpy as np
import pytest

from tasks.wilson_loop.kernels import ape, build_kernel
from tasks.wilson_loop.measure import loops
from tasks.wilson_loop.statistics import blocks, bootstrap_indices, potential, risk


def random_su3(rng, shape):
    a = rng.normal(size=(*shape, 3, 3)) + 1j * rng.normal(size=(*shape, 3, 3))
    q, _ = np.linalg.qr(a)
    return q / np.linalg.det(q)[..., None, None] ** (1 / 3)


def cold(latt):
    links = np.broadcast_to(np.eye(3, dtype=complex), (3, *latt, 3, 3)).copy()
    return links, np.broadcast_to(np.eye(3, dtype=complex), (*latt, 3, 3)).copy()


P = {'r_values': [1, 2], 't_values': [1, 2, 3]}


def test_cold_field_gives_unit_loops():
    links, temporal = cold((4, 4, 4, 4))
    w = loops(links, temporal, P)
    np.testing.assert_allclose(w, 1.0, atol=1e-13)


def test_gauge_covariance():
    rng = np.random.default_rng(3)
    latt = (4, 4, 4, 4)
    links = random_su3(rng, (3, *latt))
    temporal = random_su3(rng, latt)
    g = random_su3(rng, latt)
    gp = [np.roll(g, -1, axis=axis) for axis in (0, 1, 2, 3)]
    gt = g.swapaxes(-1, -2).conj()
    links_g = np.stack([g @ links[d] @ np.roll(gt, -1, axis=3 - d) for d in range(3)])
    temporal_g = g @ temporal @ np.roll(gt, -1, axis=0)
    np.testing.assert_allclose(loops(links_g, temporal_g, P), loops(links, temporal, P), rtol=1e-12)


def brute(links, temporal, R, T, d):
    lt, lz, ly, lx = temporal.shape[:4]
    dims = [lx, ly, lz]
    total = 0.0
    for t0 in range(lt):
        for z in range(lz):
            for y in range(ly):
                for x in range(lx):
                    c = [x, y, z]
                    def prod(c0, tt):
                        out = np.eye(3, dtype=complex)
                        for r in range(R):
                            cc = list(c0)
                            cc[d] = (cc[d] + r) % dims[d]
                            out = out @ links[d][tt, cc[2], cc[1], cc[0]]
                        return out
                    def time_line(c0, tt):
                        out = np.eye(3, dtype=complex)
                        for t in range(T):
                            out = out @ temporal[(tt + t) % lt, c0[2], c0[1], c0[0]]
                        return out
                    cR = list(c)
                    cR[d] = (cR[d] + R) % dims[d]
                    w = prod(c, t0) @ time_line(cR, t0) @ prod(c, (t0 + T) % lt).conj().T @ time_line(c, t0).conj().T
                    total += np.trace(w).real
    return total / (3 * lt * lz * ly * lx)


def test_loops_match_brute_force():
    rng = np.random.default_rng(7)
    latt = (4, 4, 4, 4)
    links = random_su3(rng, (3, *latt))
    temporal = random_su3(rng, latt)
    w = loops(links, temporal, P)
    for i, R in enumerate(P['r_values']):
        for j, T in enumerate(P['t_values']):
            reference = sum(brute(links, temporal, R, T, d) for d in range(3)) / 3
            assert w[i, j] == pytest.approx(reference, rel=1e-12)


def test_ape_keeps_links_unitary():
    rng = np.random.default_rng(5)
    links = random_su3(rng, (3, 4, 4, 4, 4))
    smeared = ape(links, 0.5, 3)
    eye = np.einsum('...ij,...kj->...ik', smeared.conj(), smeared)
    np.testing.assert_allclose(eye, np.broadcast_to(np.eye(3), eye.shape), atol=1e-10)
    det = np.linalg.det(smeared)
    np.testing.assert_allclose(det, 1.0, atol=1e-10)


def test_ape_limits():
    rng = np.random.default_rng(11)
    links = random_su3(rng, (3, 4, 4, 4, 4))
    np.testing.assert_allclose(build_kernel(links, {'alpha': 0.5, 'steps': 0}), links)
    np.testing.assert_allclose(ape(links, 0.0, 3), links)


def test_potential_recovers_known_slopes():
    v = np.array([0.4, 0.55, 0.7])
    t = np.arange(0, 9)
    w = np.exp(-np.outer(v, t))[None]
    np.testing.assert_allclose(potential(w, [4, 5, 6, 7], list(t))[0], v, rtol=1e-12)


def test_risk_penalizes_bias_and_noise():
    v = np.array([0.5, 0.6, 0.7])
    t = np.arange(1, 9)
    exact = np.exp(-np.outer(v, t))
    data = np.repeat(exact[None], 12, axis=0)[:, None]
    idx = bootstrap_indices(6, 50, 1)
    b = blocks(data, 2)
    tv = list(range(1, 9))
    windows = [[4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]]
    base = risk(b, idx, windows, tv, v)
    shift_t = np.exp(-0.02 * np.arange(8))
    worse = risk(b * shift_t, idx, windows, tv, v)
    assert base['risk'] < 1e-10
    assert worse['risk'] > base['risk']


def test_project_is_bidirectionally_covariant():
    from tasks.wilson_loop.kernels import project
    rng = np.random.default_rng(21)
    m = (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    g = random_su3(rng, ())
    h = random_su3(rng, ())
    q = project(m)
    np.testing.assert_allclose(g @ q @ h, project(g @ m @ h), atol=1e-12)
    np.testing.assert_allclose(q.conj().T @ q, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(q), 1.0, atol=1e-12)


def test_project_rejects_singular_and_nonfinite():
    from tasks.wilson_loop.kernels import project
    with pytest.raises(ValueError):
        project(np.zeros((3, 3), dtype=complex))
    with pytest.raises(ValueError):
        project(np.full((3, 3), np.nan, dtype=complex))


def test_ape_gauge_covariant_multiple_parameters():
    rng = np.random.default_rng(22)
    latt = (4, 4, 4, 4)
    u = random_su3(rng, (3, *latt))
    for alpha, steps in ((0.0, 2), (0.3, 1), (0.5, 2), (0.7, 1), (1.0, 1)):
        base = ape(u, alpha, steps)
        g = random_su3(rng, latt)
        gt = g.swapaxes(-1, -2).conj()
        ug = np.stack([g @ u[d] @ np.roll(gt, -1, axis=3 - d) for d in range(3)])
        want = np.stack([g @ base[d] @ np.roll(gt, -1, axis=3 - d) for d in range(3)])
        np.testing.assert_allclose(ape(ug, alpha, steps), want, atol=1e-10)


def _nersc(dims=(2, 2, 2, 2), datatype='4D_SU3_GAUGE_3x3', floating='1'):
    nx, ny, nz, nt = dims
    header = ('4D_SU3_GAUGE_3x3\n'
              f'DIMENSION_1={nx}\nDIMENSION_2={ny}\nDIMENSION_3={nz}\nDIMENSION_4={nt}\n'
              f'DATATYPE={datatype}\nFLOATING_POINT={floating}\nEND_HEADER\n')
    payload = np.zeros(nx * ny * nz * nt * 4 * 9, dtype='<c16').tobytes()
    return header.encode() + payload


def test_read_nersc_validates_header_and_payload(tmp_path):
    from tasks.wilson_loop.common import read_nersc
    path = tmp_path / 'gauge'
    path.write_bytes(_nersc())
    u = read_nersc(path)
    assert u.shape == (4, 2, 2, 2, 2, 3, 3)
    path.write_bytes(_nersc(floating='IEEE64LITTLE'))
    read_nersc(path)
    path.write_bytes(_nersc(datatype='4D_SU2_GAUGE_3x3'))
    with pytest.raises(ValueError, match='DATATYPE'):
        read_nersc(path)
    for bad in ('0', 'IEEE32LITTLE', 'IEEE64BIG'):
        path.write_bytes(_nersc(floating=bad))
        with pytest.raises(ValueError, match='FLOATING_POINT'):
            read_nersc(path)
    path.write_bytes(_nersc()[:-16])
    with pytest.raises(ValueError, match='payload'):
        read_nersc(path)


def test_load_data_validates_shape_dtype_and_finite(tmp_path):
    from tasks.pion_projection.common import fingerprint
    from tasks.wilson_loop.common import indices, load_data, protocol
    p = protocol()
    ids = indices('development')
    shape = (len(ids), len(p['kernels']), len(p['r_values']), len(p['t_values']))
    np.savez(tmp_path / 'development.npz', ids=ids, loops=np.ones(shape),
             protocol_hash=fingerprint(p))
    assert load_data(tmp_path, 'development').shape == shape
    for bad in (np.ones(shape[:-1]), np.full(shape, np.nan)):
        np.savez(tmp_path / 'development.npz', ids=ids, loops=bad,
                 protocol_hash=fingerprint(p))
        with pytest.raises(ValueError):
            load_data(tmp_path, 'development')


def _synthetic_loops(ids, v=(0.4, 0.55, 0.7)):
    from tasks.wilson_loop.common import protocol
    p = protocol()
    t = np.array(p['t_values'], dtype=float)
    base = np.exp(-np.outer(v, t))
    return np.repeat(base[None, None, :, :], len(ids), axis=0).repeat(len(p['kernels']), axis=1)


def test_develop_evaluate_roundtrip_and_tamper_rejection(tmp_path):
    import json
    from tasks.pion_projection.common import fingerprint
    from tasks.wilson_loop.common import indices, protocol
    from tasks.wilson_loop.experiment import develop, evaluate
    p = protocol()
    np.savez(tmp_path / 'development.npz', ids=indices('development'),
             loops=_synthetic_loops(indices('development')), protocol_hash=fingerprint(p))
    develop(tmp_path)
    frozen = json.loads((tmp_path / 'submission.json').read_text())
    assert frozen['reference_passed'] and 'kernel_sha256' in frozen
    np.savez(tmp_path / 'validation.npz', ids=indices('validation'),
             loops=_synthetic_loops(indices('validation')), protocol_hash=fingerprint(p))
    evaluate(tmp_path)
    report = json.loads((tmp_path / 'validation_results.json').read_text())
    assert report['reference_passed']
    assert report['results']['identity']['score'] == pytest.approx(100.0)
    # top-level score, RunState score and the comparison verdict must agree:
    # a conditional diagnostic is reported, never a physical score
    assert report['physical_score'] is None
    assert report['run_state']['score'] is None
    assert report['run_state']['status']['comparison'] in ('draft', 'failed')
    assert report['conditional_scores']['identity'] == pytest.approx(100.0)
    assert report['score_status'] == 'conditional_diagnostic'
    loops = _synthetic_loops(indices('development'))
    loops[0, 0, 0, 0] += 0.5
    np.savez(tmp_path / 'development.npz', ids=indices('development'), loops=loops,
             protocol_hash=fingerprint(p))
    with pytest.raises(ValueError, match='Development data changed'):
        evaluate(tmp_path)


def test_protocol_v2_documents_the_normalization_change():
    import json
    from pathlib import Path
    p = json.loads(Path('tasks/wilson_loop/protocol_v2.json').read_text())
    assert p['version'].endswith('-v2') and 'v1' in p['supersedes']
    assert p['normalization'].startswith('alpha/4')
    for kernel in p['kernels'][1:]:
        old = kernel['equivalent_v1_alpha']
        assert kernel['alpha'] == pytest.approx(2 * old / (3 - old), rel=1e-4)


def test_paired_ratios_are_one_for_identical_kernels_and_below_one_for_noise():
    from tasks.wilson_loop.common import protocol
    from tasks.wilson_loop.paired import paired_ratios
    p = protocol()
    t = np.array(p['t_values'], dtype=float)
    v = np.array([0.4, 0.55, 0.7])
    base = np.exp(-np.outer(v, t))
    blocks = 6
    same = np.repeat(base[None, None], blocks, axis=0).repeat(2, axis=1)
    rng = np.random.default_rng(0)
    noisy = same.copy()
    noisy[:, 1] *= np.exp(0.01 * rng.normal(size=noisy[:, 1].shape))
    r_same = paired_ratios(same, p['reference_windows'], p['t_values'], v, outer=40, inner=40, seed=1)
    r_noisy = paired_ratios(noisy, p['reference_windows'], p['t_values'], v, outer=40, inner=40, seed=1)
    np.testing.assert_allclose(r_same, 1.0, rtol=1e-12)
    assert np.median(r_noisy) < 1.0


def test_protocol_v3_is_consistent():
    import json
    from pathlib import Path
    p = json.loads(Path('tasks/wilson_loop/protocol_v3.json').read_text())
    assert p['version'].endswith('-v3')
    assert p['development'] == [0, 99] and p['validation'] == [120, 199]
    assert p['gap'] == [100, 119]
    assert p['kernels'] == json.loads(Path('tasks/wilson_loop/protocol_v2.json').read_text())['kernels']


def test_projection_batch_matches_single_matrices():
    from tasks.wilson_loop.kernels import project
    rng = np.random.default_rng(31)
    a = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    eye = np.eye(3, dtype=complex)
    single = [project(eye), project(1e4 * eye), project(a), project(1e-6 * a)]
    batch = project(np.stack([eye, 1e4 * eye, a, 1e-6 * a]))
    for i, expected in enumerate(single):
        np.testing.assert_allclose(batch[i], expected, atol=1e-10)


def test_projection_rejects_nonfinite_and_ill_conditioned():
    from tasks.wilson_loop.kernels import project
    bad = np.stack([np.eye(3, dtype=complex), np.full((3, 3), np.inf)])
    with pytest.raises(ValueError, match='non-positive or non-finite'):
        project(bad)


def test_nersc_header_geometry_and_payload(tmp_path):
    from tasks.wilson_loop.common import read_nersc
    header = (b'BEGIN_HEADER\nDIMENSION_1=1\nDIMENSION_2=1\nDIMENSION_3=1\nDIMENSION_4=1\n'
              b'DATATYPE=4D_SU3_GAUGE_3x3\nFLOATING_POINT=IEEE64LITTLE\nEND_HEADER\n')
    path = tmp_path / 'gauge'
    path.write_bytes(header + b'\x00' * (1 * 1 * 1 * 1 * 4 * 3 * 3 * 16))
    read_nersc(path, [1, 1, 1, 1])
    with pytest.raises(ValueError, match='geometry'):
        read_nersc(path, [2, 2, 2, 16])


def _make_transaction(tmp_path, spec, n=3):
    import json
    from tasks.wilson_loop.identity import generation, loops_sha256, write_marker
    ids = spec['ids'][:n]
    loops = np.zeros((n, 2, 2, 1))
    gen = generation(spec)
    np.savez(tmp_path / 'validation.npz', ids=np.asarray(ids), loops=loops,
             generation=np.asarray(gen))
    records = [{'id': int(cid), 'loops_sha256': loops_sha256(loops[row:row + 1])}
               for row, cid in enumerate(ids)]
    (tmp_path / 'validation_provenance.json').write_text(json.dumps(
        {'generation': gen, 'kernel_sha256': spec['kernel_sha256'], 'records': records}))
    write_marker(tmp_path, 'validation', tmp_path / 'validation.npz',
                 tmp_path / 'validation_provenance.json', spec, loops)
    return ids


def test_measurement_generation_and_resume(tmp_path):
    from tasks.wilson_loop.common import protocol
    from tasks.wilson_loop.identity import check_resume, measurement_spec
    spec = measurement_spec(protocol(), 'validation')
    assert check_resume(tmp_path, 'validation', spec, verify_gauge=False) == 0
    _make_transaction(tmp_path, spec)
    assert check_resume(tmp_path, 'validation', spec, verify_gauge=False) == 3
    with pytest.raises(ValueError, match='generation'):
        check_resume(tmp_path, 'validation', dict(spec, kernel_sha256='0' * 64), verify_gauge=False)


def test_resume_rejects_inconsistent_prefix(tmp_path):
    import json
    from tasks.wilson_loop.common import protocol
    from tasks.wilson_loop.identity import check_resume, measurement_spec
    spec = measurement_spec(protocol(), 'validation')
    ids = _make_transaction(tmp_path, spec)
    (tmp_path / 'validation.commit.json').unlink()
    with pytest.raises(ValueError, match='partial measurement transaction'):
        check_resume(tmp_path, 'validation', spec, verify_gauge=False)
    _make_transaction(tmp_path, spec)
    provenance = {'generation': json.loads((tmp_path / 'validation_provenance.json').read_text())['generation'],
                  'kernel_sha256': spec['kernel_sha256'],
                  'records': [{'id': int(i)} for i in ids[:2]]}
    (tmp_path / 'validation_provenance.json').write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match='commit marker|records versus|loops hash|no loops hash'):
        check_resume(tmp_path, 'validation', spec, verify_gauge=False)


def test_trust_status_does_not_use_the_protocol_name(monkeypatch):
    from pathlib import Path
    from tasks.wilson_loop.common import protocol
    from tasks.wilson_loop.identity import trust_status
    run = Path('results/wilson_loop/s16t16_v3')
    if not (run / 'submission.json').exists():
        pytest.skip('v3 run not available')
    import tasks.wilson_loop.common as common
    import json
    v3 = json.loads(Path('tasks/wilson_loop/protocol_v3.json').read_text())
    monkeypatch.setattr(common, 'protocol', lambda: v3)
    status = trust_status(run, v3, json.loads((run / 'submission.json').read_text()),
                          json.loads((run / 'validation_provenance.json').read_text()))
    assert status['status'] == 'archived_diagnostic'
    assert status['reasons']


def test_evidence_schema_and_summary(tmp_path):
    from core.evidence import load_evidence, record, summarise, validate, write_evidence
    with pytest.raises(ValueError, match='missing'):
        validate({'test_id': 'x'})
    with pytest.raises(ValueError, match='status'):
        record('x', 'wilson_loop', 'claim', 'cmd', 1, 1, 'MAYBE', 'synthetic_CPU')
    entry = record('x', 'wilson_loop', 'claim', 'cmd', 1, 1, 'PASS', 'synthetic_CPU')
    path = tmp_path / 'e.json'
    write_evidence(path, [entry])
    assert load_evidence(path)[0]['status'] == 'PASS'
    assert summarise([entry])[0]['task'] == 'wilson_loop'


def test_operator_checks_and_counterexamples():
    from core.operator_checks import (bad_gram_schmidt_projector, bad_nonhermitian_kernel,
                                      covariance_residual, good_hermitian_kernel,
                                      hermiticity_residual, linearity_residual)
    rng = np.random.default_rng(5)
    f = rng.normal(size=(8, 3)) + 1j * rng.normal(size=(8, 3))
    g = rng.normal(size=(8, 3)) + 1j * rng.normal(size=(8, 3))
    assert linearity_residual(lambda x: 2 * x - 0.5 * np.roll(x, 1, 0), f, g) < 1e-12
    assert linearity_residual(lambda x: np.abs(x), f, g) > 1e-3
    good = lambda a, b: np.vdot(a, good_hermitian_kernel(b))
    bad = lambda a, b: np.vdot(a, bad_nonhermitian_kernel(b))
    assert hermiticity_residual(good, f, g) < 1e-12
    assert hermiticity_residual(bad, f, g) > 1e-3
    u = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))[0]
    field = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    assert covariance_residual(lambda x: u @ x, lambda x: u @ x, field) < 1e-12
    assert covariance_residual(bad_gram_schmidt_projector, lambda x: x @ u, field) > 1e-6


def test_toy_leapfrog_is_second_order():
    from tasks.wilson_loop.hmc_quality import order_check
    rows = order_check((0.02, 0.01))
    ratio = rows[0]['delta_H'] / rows[1]['delta_H']
    assert 3.2 < ratio < 5.0, (ratio, rows)


def test_develop_records_run_state(tmp_path, monkeypatch):
    import json
    import shutil
    from pathlib import Path
    from tasks.wilson_loop import experiment
    run = Path('results/wilson_loop/s16t16_v3')
    if not (run / 'development.npz').exists():
        pytest.skip('v3 development data not available')
    shutil.copy(run / 'development.npz', tmp_path / 'development.npz')
    shutil.copy(run / 'development_provenance.json', tmp_path / 'development_provenance.json')
    v3 = json.loads(Path('tasks/wilson_loop/protocol_v3.json').read_text())
    import tasks.wilson_loop.common as common
    monkeypatch.setattr(common, 'protocol', lambda: v3)
    monkeypatch.setattr(experiment, 'protocol', lambda: v3)
    experiment.develop(tmp_path)
    frozen = json.loads((tmp_path / 'submission.json').read_text())
    assert frozen['run_state']['status']['operator'] == 'validated'
    assert frozen['run_state']['status']['numerical'] == 'validated'
    assert frozen['run_state']['status']['reference'] in ('validated', 'failed')
    assert frozen['run_state']['status']['comparison'] in ('draft', 'failed')



def test_invalid_sample_counting():
    from tasks.wilson_loop.invalid_audit import count_invalid
    data = np.ones((6, 2, 3, 5))
    data[:, 0, 0, 0] = -1.0        # negative mean at one (R, T)
    data[1, 1, 1, 2] = np.nan      # NaN
    index = np.arange(6)[:, None].repeat(6, axis=1)
    rows = count_invalid(data, index)
    assert rows[0]['mean_nonpositive'] == 1
    assert rows[1]['mean_nan'] >= 1
    assert rows[0]['bootstrap_nonpositive'] > 0


def test_measure_entry_end_to_end(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import tasks.wilson_loop.measure as measure_mod
    from tasks.wilson_loop import common
    p = json.loads(Path('tasks/wilson_loop/protocol_v3.json').read_text())
    p = dict(p, version='test-entry', geometry=[4, 4, 4, 16], r_values=[1, 2],
             kernels=p['kernels'][:2], ensemble=str(tmp_path / 'ensemble'))
    (tmp_path / 'ensemble').mkdir()
    for cfg in common.indices('development')[:3] if False else range(3):
        (tmp_path / 'ensemble' / f'wilson_b6.{cfg}').write_bytes(b'x')
    rng = np.random.default_rng(3)
    links = rng.normal(size=(4, 16, 4, 4, 4, 3, 3)) + 1j * rng.normal(size=(4, 16, 4, 4, 4, 3, 3))
    q, _ = np.linalg.qr(links)
    links = q / np.linalg.det(q)[..., None, None] ** (1 / 3)
    monkeypatch.setattr(measure_mod, 'read_nersc_with_status',
                        lambda path, expected_geometry=None: (links, 'absent'))
    monkeypatch.setattr(measure_mod, 'protocol', lambda: p)
    monkeypatch.setattr(common, 'protocol', lambda: p)
    out = tmp_path / 'out'
    measure_mod.measure(out, 'development', limit=2)
    with np.load(out / 'development.npz') as f:
        assert f['ids'].tolist() == [0, 1] and f['loops'].shape[0] == 2
    records = json.loads((out / 'development_provenance.json').read_text())['records']
    assert [r['id'] for r in records] == [0, 1]
    measure_mod.measure(out, 'development', limit=2)   # resume: no new rows
    with np.load(out / 'development.npz') as f:
        assert f['ids'].tolist() == [0, 1] and f['loops'].shape[0] == 2
    records = json.loads((out / 'development_provenance.json').read_text())['records']
    assert [r['id'] for r in records] == [0, 1]


def test_spectral_two_state_counterexample():
    from tasks.wilson_loop.spectral_physical import analyse
    rng = np.random.default_rng(7)
    row = analyse(rng, 0.2)
    assert row['gate_spread_pass']
    assert row['bias_over_se'] > 3


def test_projection_failure_reports_batch_index():
    from tasks.wilson_loop.kernels import project
    good = np.eye(3, dtype=complex)
    bad = np.diag([1.0, 1.0, 0.0]).astype(complex)
    with pytest.raises(ValueError, match=r'indices \[1\]'):
        project(np.stack([good, bad]))


def test_risk_fails_closed_on_invalid_w():
    from tasks.wilson_loop.statistics import blocks, bootstrap_indices, risk
    data = np.ones((6, 1, 5))
    data[:, 0, 0] = -1.0
    with pytest.raises(ValueError, match='non-positive W'):
        risk(blocks(data, 2), bootstrap_indices(3, 10, 1), [[4, 5, 6]], [4, 5, 6, 7, 8], [1.0], nominal=0)
    nan_data = np.ones((6, 1, 5))
    nan_data[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='non-finite W'):
        risk(blocks(nan_data, 2), bootstrap_indices(3, 10, 1), [[4, 5, 6]], [4, 5, 6, 7, 8], [1.0], nominal=0)


def test_nersc_checksum_trailer(tmp_path):
    import hashlib
    from tasks.wilson_loop.common import nersc_checksum_status, read_nersc
    rng = np.random.default_rng(3)
    data = (rng.normal(size=(2, 2, 2, 2, 4, 3, 3)) + 1j * rng.normal(size=(2, 2, 2, 2, 4, 3, 3))).astype('<c16')
    payload = data.tobytes()
    header = (b'4D_SU3_GAUGE_3x3\nDATATYPE=4D_SU3_GAUGE_3x3\n'
              b'DIMENSION_1=2\nDIMENSION_2=2\nDIMENSION_3=2\nDIMENSION_4=2\n'
              b'FLOATING_POINT=IEEE64LITTLE\nEND_HEADER\n')
    good = tmp_path / 'good'
    good.write_bytes(header + payload + hashlib.md5(payload).digest())
    assert nersc_checksum_status(good) == 'verified'
    array = read_nersc(good)
    assert array.shape == (4, 2, 2, 2, 2, 3, 3)
    absent = tmp_path / 'absent'
    absent.write_bytes(header + payload)
    assert nersc_checksum_status(absent) == 'absent'
    bad = tmp_path / 'bad'
    bad.write_bytes(header + payload + b'0' * 16)
    with pytest.raises(ValueError, match='MD5 mismatch'):
        read_nersc(bad)


def test_hmc_manifest_audit_reports_missing_fields():
    from tasks.wilson_loop.hmc_quality import audit_manifest
    result = audit_manifest({'seed': 1, 'tau_int': 0.5})
    assert result['status'] == 'archived_diagnostic'
    assert 'acceptance_rate_missing' in result['reasons']
    ok = audit_manifest({field: 0 for field in ('seed', 'first_trajectory', 'spacing',
                                                 'tau_int', 'n_configs', 'acceptance_rate',
                                                 'delta_h', 'thermalization')})
    assert ok['status'] == 'verified'
