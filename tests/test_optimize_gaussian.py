import numpy as np
import pytest

from tasks.optimize_gaussian.features import dagger, directional_features, plaquettes, shift
from tasks.optimize_gaussian.kernels import apply_kernel, correlator
from tasks.optimize_gaussian.submission import build_kernel
from tasks.optimize_gaussian.train import features, gaussian


def su3(rng, shape):
    a = rng.normal(size=(*shape, 3, 3)) + 1j * rng.normal(size=(*shape, 3, 3))
    q, _ = np.linalg.qr(a)
    q[..., :, 0] /= np.linalg.det(q)[..., None]
    return q


def gauge_links(u, g):
    return np.stack([g @ u[d] @ dagger(shift(g, 1, d)) for d in range(3)])


def gauge_field(f, g):
    return np.einsum('...ij,...jn->...in', g, f)


def params(seed=0, steps=1, hidden=4):
    rng = np.random.default_rng(seed)
    return {'W': rng.normal(size=(hidden, 2)).tolist(),
            'theta': rng.normal(size=1 + hidden).tolist(),
            'steps': steps, 'alpha': 0.09090909}


def field(rng, latt=(4, 4, 4)):
    return rng.normal(size=(*latt, 3, 2)) + 1j * rng.normal(size=(*latt, 3, 2))


def test_shift_matches_roll_and_is_periodic():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(4, 4, 4))
    for d, axis in ((0, 2), (1, 1), (2, 0)):
        np.testing.assert_array_equal(shift(a, 1, d), np.roll(a, -1, axis=axis))
        np.testing.assert_array_equal(shift(a, -1, d), np.roll(a, 1, axis=axis))
        np.testing.assert_array_equal(shift(shift(a, 1, d), -1, d), a)


def test_plaquettes_are_closed_and_gauge_invariant():
    rng = np.random.default_rng(2)
    latt = (4, 4, 4)
    u = su3(rng, (3, *latt))
    cold = np.broadcast_to(np.eye(3, dtype=complex), (3, *latt, 3, 3)).copy()
    np.testing.assert_allclose(plaquettes(cold), 1.0, atol=1e-13)
    g = su3(rng, latt)
    np.testing.assert_allclose(plaquettes(gauge_links(u, g)), plaquettes(u), atol=1e-12)
    # the loop must be closed: U_mu(x) U_nu(x+mu) U_mu^dag(x+nu) U_nu^dag(x)
    mu, nu = 0, 1
    loop = u[mu] @ shift(u[nu], 1, mu) @ dagger(shift(u[mu], 1, nu)) @ dagger(u[nu])
    np.testing.assert_allclose(plaquettes(u)[..., 0],
                               np.trace(loop, axis1=-2, axis2=-1).real / 3, atol=1e-12)


def test_kernel_is_hermitian_with_distinct_fields():
    rng = np.random.default_rng(3)
    u = su3(rng, (3, 4, 4, 4))
    f, g = field(rng), field(rng)
    a = rng.normal(size=(3, 4, 4, 4))
    np.testing.assert_allclose(np.vdot(g, apply_kernel(f, u, a)),
                               np.vdot(apply_kernel(g, u, a), f), atol=1e-12)


def test_kernel_is_gauge_covariant():
    rng = np.random.default_rng(4)
    latt = (4, 4, 4)
    u = su3(rng, (3, *latt))
    g = su3(rng, latt)
    f = field(rng)
    a = rng.normal(size=(3, *latt))
    np.testing.assert_allclose(apply_kernel(gauge_field(f, g), gauge_links(u, g), a),
                               gauge_field(apply_kernel(f, u, a), g), atol=1e-12)
    p = params(4, steps=2)
    np.testing.assert_allclose(build_kernel(gauge_links(u, g), p).apply(gauge_field(f, g)),
                               gauge_field(build_kernel(u, p).apply(f), g), atol=1e-12)


def test_kernel_cubic_symmetry():
    rng = np.random.default_rng(5)
    latt = (4, 4, 4)
    u = su3(rng, (3, *latt))
    f = field(rng)
    p = params(5, steps=1)
    swapped_u = u[[2, 1, 0]].transpose(0, 3, 2, 1, 4, 5)
    swapped_f = f.transpose(2, 1, 0, 3, 4)
    np.testing.assert_allclose(build_kernel(swapped_u, p).apply(swapped_f),
                               build_kernel(u, p).apply(f).transpose(2, 1, 0, 3, 4), atol=1e-12)


def test_linear_response_matches_direct_correlator():
    rng = np.random.default_rng(6)
    latt = (4, 4, 4)
    u = su3(rng, (3, *latt))
    f = field(rng)
    p = params(6, steps=2, hidden=4)
    W = np.asarray(p['W'])
    hidden = W.shape[0]
    base = gaussian(f, u, p['steps'], p['alpha'])
    zero = np.zeros((3, *latt))
    C0 = correlator(base, u, zero)
    Dc = correlator(base, u, np.ones((3, *latt))) - C0
    phi = features(u, W)
    Dh = [correlator(base, u, phi[:, ..., h]) - C0 for h in range(hidden)]
    theta = rng.normal(size=1 + hidden)
    a = np.full((3, *latt), theta[0])
    for h in range(hidden):
        a += theta[1 + h] * phi[:, ..., h]
    direct = correlator(base, u, a)
    linear = C0 + theta[0] * Dc + sum(theta[1 + h] * Dh[h] for h in range(hidden))
    np.testing.assert_allclose(linear, direct, atol=1e-11)


def test_features_are_gauge_invariant_and_zero_mean():
    rng = np.random.default_rng(7)
    latt = (4, 4, 4)
    u = su3(rng, (3, *latt))
    g = su3(rng, latt)
    W = np.asarray(params(7, hidden=3)['W'])
    np.testing.assert_allclose(features(gauge_links(u, g), W), features(u, W), atol=1e-11)
    phi = features(u, W)
    np.testing.assert_allclose(phi.mean(axis=(1, 2, 3)), 0.0, atol=1e-11)


def test_a1p_directional_pair_is_exchange_symmetric():
    rng = np.random.default_rng(8)
    latt = (4, 4, 4)
    u = su3(rng, (3, *latt))
    swapped = u[[0, 2, 1]].transpose(0, 2, 1, 3, 4, 5)
    f = directional_features(u).transpose(0, 2, 1, 3, 4)
    s = directional_features(swapped)
    # site axes are (z, y, x), so swapping y and z also transposes sites;
    # direction x sees (xy, xz), and the symmetric (sum, product) pair survives.
    np.testing.assert_allclose(f[0], s[0], atol=1e-12)
    np.testing.assert_allclose(f[1], s[2], atol=1e-12)
    np.testing.assert_allclose(f[2], s[1], atol=1e-12)


def test_invalid_shapes_are_rejected():
    with pytest.raises(Exception):
        apply_kernel(np.zeros((4, 4, 4, 3, 2)), np.zeros((2, 4, 4, 4, 3, 3)), np.zeros((3, 4, 4, 4)))


def test_basis_cache_rejects_identity_change(tmp_path):
    from tasks.optimize_gaussian.train import basis_key, load_basis, save_basis
    p = {'base_steps': 8, 'base_alpha': 0.09, 'hidden': 2}
    W = np.zeros((2, 2))
    key = basis_key(p, 7, W, [0, 1], [2, 3])
    save_basis(tmp_path / 'b.npz', key, np.zeros((2, 3, 4)), np.zeros((2, 3, 4)))
    load_basis(tmp_path / 'b.npz', key)
    for mutated in (basis_key(p, 8, W, [0, 1], [2, 3]),
                    basis_key(p, 7, W + 1, [0, 1], [2, 3]),
                    basis_key(p, 7, W, [0, 1], [2, 4]),
                    basis_key({**p, 'base_steps': 9}, 7, W, [0, 1], [2, 3]),
                    basis_key(p, 7, W, [0, 1], [2, 3], dependencies={'train.py': 'changed'}),
                    basis_key(p, 7, W, [0, 1], [2, 3], inputs={'gauge': {'0': 'changed'}})):
        with pytest.raises(ValueError, match='does not match'):
            load_basis(tmp_path / 'b.npz', mutated)
    np.savez(tmp_path / 'legacy.npz', C_fit=np.zeros(1), C_dev=np.zeros(1))
    with pytest.raises(ValueError, match='no identity record'):
        load_basis(tmp_path / 'legacy.npz', key)


def _cube_rotations():
    import itertools
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            r = np.zeros((3, 3), dtype=int)
            for k, j in enumerate(perm):
                r[k, j] = signs[k]
            if round(np.linalg.det(r)) == 1:
                out.append(r)
    return out


def _transform(field_or_links, r, shift, dagger):
    n = field_or_links.shape[1]
    zz, yy, xx = np.indices((n, n, n))
    v = np.stack([xx, yy, zz], axis=-1)
    w = np.einsum('ij,...j->...i', np.linalg.inv(r), v).round().astype(int) % n
    idx = (w[..., 2], w[..., 1], w[..., 0])
    if field_or_links.ndim == 5:
        return field_or_links[idx]
    out = []
    for k in range(3):
        j = int(np.argmax(np.abs(r[k])))
        sign = int(r[k, j])
        src = field_or_links[j] if sign > 0 else dagger(shift(field_or_links[j], -1, j))
        out.append(src[idx])
    return np.stack(out)


def test_full_proper_cubic_group_covariance():
    from tasks.optimize_gaussian.features import dagger, shift
    rng = np.random.default_rng(11)
    latt = (4, 4, 4)
    u = su3(rng, (3, *latt))
    f = field(rng)
    p = params(11, steps=2)
    base = build_kernel(u, p).apply(f)
    for r in _cube_rotations():
        ug = _transform(u, r, shift, dagger)
        fg = _transform(f, r, shift, dagger)
        np.testing.assert_allclose(build_kernel(ug, p).apply(fg), _transform(base, r, shift, dagger), atol=1e-11)


def test_parity_covariance():
    from tasks.optimize_gaussian.features import dagger, shift
    rng = np.random.default_rng(12)
    latt = (4, 4, 4)
    u = su3(rng, (3, *latt))
    f = field(rng)
    p = params(12, steps=2)
    r = -np.eye(3, dtype=int)
    ug = _transform(u, r, shift, dagger)
    fg = _transform(f, r, shift, dagger)
    np.testing.assert_allclose(build_kernel(ug, p).apply(fg),
                               _transform(build_kernel(u, p).apply(f), r, shift, dagger), atol=1e-11)


def test_run_manifest_records_dependencies_and_ids(tmp_path):
    from tasks.optimize_gaussian.train import run_manifest
    p = {'geometry': [16, 16, 16, 16], 'base_steps': 8, 'base_alpha': 0.09}
    m = run_manifest(p, np.zeros((2, 2)), np.zeros(3), [0, 1], [2], [3], {0: 'a', 1: 'b', 2: 'c', 3: 'd'})
    assert m['schema'] == 'optgauss-run-v1'
    assert m['ids'] == {'fit': [0, 1], 'dev_val': [2], 'test': [3]}
    assert m['propagator_cache'] == {'0': 'a', '1': 'b', '2': 'c', '3': 'd'}
    assert set(m['dependencies']) == {'features.py', 'kernels.py', 'submission.py', 'train.py'}
    assert all(len(v) == 64 for v in m['dependencies'].values())
    assert len(m['statistics_sha256']) == 64


def test_submission_matches_linear_response_and_is_hermitian():
    from tasks.optimize_gaussian.kernels import correlator
    from tasks.optimize_gaussian.submission import build_kernel
    from tasks.optimize_gaussian.train import gaussian
    rng = np.random.default_rng(41)
    latt = (3, 4, 5)  # non-cubic on purpose
    u = su3(rng, (3, *latt))
    f, g = field(rng, latt), field(rng, latt)
    p = params(41, steps=1, hidden=3)
    W = np.asarray(p['W'])
    hidden = W.shape[0]
    base = gaussian(f, u, p['steps'], p['alpha'])
    zero = np.zeros((3, *latt))
    C0 = correlator(base, u, zero)
    Dc = correlator(base, u, np.ones((3, *latt))) - C0
    phi = features(u, W)
    Dh = [correlator(base, u, phi[:, ..., h]) - C0 for h in range(hidden)]
    cases = [('zero', np.zeros(1 + hidden)),
             ('c_only', np.array([0.3] + [0.0] * hidden)),
             ('one_v', np.array([0.0, 0.4] + [0.0] * (hidden - 1))),
             ('full', rng.normal(size=1 + hidden))]
    for label, theta in cases:
        a = np.full((3, *latt), theta[0])
        for h in range(hidden):
            a += theta[1 + h] * phi[:, ..., h]
        linear = C0 + theta[0] * Dc + sum(theta[1 + h] * Dh[h] for h in range(hidden))
        direct = correlator(base, u, a)
        np.testing.assert_allclose(linear, direct, rtol=0, atol=1e-10)
        spec = {**p, 'theta': theta.tolist()}
        submitted = complex(np.vdot(f, build_kernel(u, spec).apply(f)))
        np.testing.assert_allclose(submitted.real, direct, rtol=0, atol=1e-10)
        # full BCB Hermiticity with distinct fields
        np.testing.assert_allclose(np.vdot(g, build_kernel(u, spec).apply(f)),
                                   np.vdot(build_kernel(u, spec).apply(g), f), atol=1e-10)


def test_teacher_seven_power_basis_gap():
    from tasks.optimize_gaussian.teacher_check import teacher_result
    result = teacher_result(shape=(2, 2, 2), seed=7)
    assert result['hermiticity_H'] < 1e-12 and result['hermiticity_Q'] < 1e-12
    assert result['augmented_error'] < 1e-10
    assert result['seven_power_error'] > result['spectral_floor']


def test_p3_teacher_spectral_vs_augmented():
    from tasks.optimize_gaussian.p3_teacher import run
    result = run(shape=(2, 2, 2), seed=11, n_probe=16)
    assert result['status'] == 'PASS'
    rows = result['rows']
    assert rows[0]['spectral_test_error'] < 1e-10
    assert rows[-1]['spectral_test_error'] > 1e-3
    assert rows[-1]['augmented_test_error'] < 1e-10


def test_validate_inputs_status(tmp_path, monkeypatch):
    import json
    import numpy as np
    from tasks.optimize_gaussian import train
    gauge = tmp_path / 'wilson_b6.7'
    gauge.write_bytes(b'gauge')
    cache_dir = tmp_path / 'props'
    cache_dir.mkdir()
    prop = cache_dir / 'k.npy'
    np.save(prop, np.zeros(3, dtype=np.complex128))
    import hashlib
    (cache_dir / 'k.json').write_text(json.dumps(
        {'array_sha256': hashlib.sha256(prop.read_bytes()).hexdigest(),
         'keydata': {'gauge': hashlib.sha256(gauge.read_bytes()).hexdigest(), 'residual_max': 1e-12,
                     'geometry': [4, 4, 4, 8], 'boundary': -1, 'tol': 1e-10, 'maxiter': 10000}}))
    monkeypatch.chdir(tmp_path)
    (tmp_path / '.cache/pion_projection/propagators').mkdir(parents=True)
    prop2 = tmp_path / '.cache/pion_projection/propagators/k.npy'
    prop2.write_bytes(prop.read_bytes())
    (prop2.with_suffix('.json')).write_text((cache_dir / 'k.json').read_text())
    p = {'ensemble': str(tmp_path), 'tolerance': 1e-10, 'geometry': [4, 4, 4, 8],
         't_boundary': -1, 'max_iterations': 10000}
    row = train.validate_inputs(p, [7], {7: 'k'})['7']
    assert row['status'] == 'verified' and row['gauge'] and row['propagator']
    assert row['geometry'] and row['boundary'] and row['precision']
    bad = dict(p, geometry=[8, 8, 8, 8])
    with pytest.raises(ValueError, match='protocol mismatch'):
        train.validate_inputs(bad, [7], {7: 'k'})


def test_crosslink_target_and_oracle():
    import numpy as np
    from tasks.optimize_gaussian.p3_crosslink import B_of, H_of, samples, su3, target
    rng = np.random.default_rng(2)
    links = su3(rng, 0.1) @ su3(rng, 0.5)
    h, b = H_of(links, (1, 1, 1, 1)), B_of(links, (1, 1, 1, 1))
    q = target(h, b, 0.0)
    assert np.linalg.norm(q - q.conj().T) < 1e-12 and abs(np.trace(q)) < 1e-12
    assert np.linalg.norm(q - (h @ h - np.trace(h @ h) / 3 * np.eye(3))) < 1e-12
    X, Y = samples(rng, links, 100, 0.1, augmented=True)
    assert X.shape[1] == 72 + 27 and Y.shape[1] == 9


def test_unified_estimator_amplitude_sources():
    import numpy as np
    from tasks.optimize_gaussian.unified import risk_with_fixed_amplitude
    rng = np.random.default_rng(0)
    t = np.arange(9)
    data = np.ones((6, 3, 9)) * (np.exp(-0.5 * t) + np.exp(-0.5 * (8 - t)))
    data *= 1 + 0.01 * rng.normal(size=data.shape)
    p = {'amplitude_window': [6, 7, 8], 'score_window': [2, 3, 4, 5], 'geometry': [4, 4, 4, 8],
         'covariance_shrinkage': 0.1}
    idx = np.array([[0, 1, 2], [3, 4, 5]])
    train = risk_with_fixed_amplitude(data, np.ones(3), 0.5, np.full(2, 0.5), idx, p, 1.0, 0.01,
                                      amplitude_source='train')
    test = risk_with_fixed_amplitude(data, np.ones(3), 0.5, np.full(2, 0.5), idx, p, 1.0, 0.01,
                                     amplitude_source='test')
    assert train['late_covariance_source'] == 'train' and train['amplitude'] == 1.0
    assert test['late_covariance_source'] == 'test'
    assert np.isfinite(train['risk']) and np.isfinite(test['risk'])


def test_estimator_regression_runs():
    from tasks.optimize_gaussian.estimator_regression import run
    result = run(n_dataset=5, n_boot=20)
    assert set(result['summary']) == {'train', 'test'}
    assert abs(result['summary']['train']['bias_mean']) < 1e-2


def test_adapt_instance_contract_and_search():
    from tasks.optimize_gaussian.adapt import contract_checks, search_theta
    import numpy as np
    rng = np.random.default_rng(0)
    checks = contract_checks(rng)
    assert all(checks['contract'][k] for k in ('linear_in_field', 'hermitian', 'identity_at_theta0'))
    info = search_theta(rng)
    assert info['accepted'] == 3 and info['rejected'] == 1
