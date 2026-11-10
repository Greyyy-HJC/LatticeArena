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
