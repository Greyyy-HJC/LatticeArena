"""Isolated trainable-W CPU parameterization tests; no production fixtures."""
import numpy as np
import pytest

from tasks.optimize_gaussian import trainable_w as model
from tasks.optimize_gaussian.features import directional_features
from tasks.optimize_gaussian.trainable_w_diagnostic import gradient_check, su3, symmetry_checks


def fixture_data():
    rng = np.random.default_rng(92021)
    links = [su3(rng, (3, 3, 3, 3)) for _ in range(2)]
    z = np.stack([directional_features(u) for u in links])
    W, v, c = np.array([[1.2, -.7], [-.4, 1.1]]), np.array([.2, -.3]), .06
    return links, z, W, v, c


def test_analytic_gradient_includes_spatial_centering():
    _, z, W, v, c = fixture_data()
    # Unrestricted real targets include spatial/direction mean offsets; this
    # catches the otherwise easy-to-miss derivative of feature mean removal.
    target = np.random.default_rng(92022).normal(size=z.shape[:-1]) * .3
    parameters = model.pack(W, v, c)
    check = gradient_check(parameters, z, target, 2)
    analytic = np.array([row['analytic'] for row in check['rows']])
    numeric = np.array([row['finite_difference'] for row in check['rows']])
    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-10)


def test_real_parameterization_symmetry_gauge_and_hermiticity():
    links, _, W, v, c = fixture_data()
    check = symmetry_checks(links[0], W, v, c, seed=92023)
    assert check['coefficient_spatial_mean_minus_c_max_abs'] < 1e-13
    assert check['feature_gauge_max_abs'] < 1e-12
    assert check['coefficient_gauge_max_abs'] < 1e-12
    assert check['kernel_gauge_relative'] < 1e-12
    assert check['hermitian_relative'] < 1e-12
    assert len(check['cubic_cases']) == 48
    assert sum(row['determinant'] == 1 for row in check['cubic_cases']) == 24
    assert check['cubic_max_relative'] < 1e-12
    assert check['cubic_coefficient_max_abs'] < 1e-12


def test_joint_training_updates_W_and_reduces_coefficient_loss():
    _, z, teacher_W, teacher_v, teacher_c = fixture_data()
    target = model.coefficient_field(z, teacher_W, teacher_v, teacher_c)
    W0, v0 = teacher_W * .3, np.array([.1, -.1])
    result = model.fit(z, target, W0, v0, -.03, maxiter=150)
    assert result['classification'] == 'diagnostic_only' and result['physical_score'] is None
    assert result['W_update_norm'] > 1e-5
    assert result['final_loss'] < result['initial_loss'] * .01
    assert len(result['trace']) == result['nfev']
    assert all(np.isfinite(row['loss']) for row in result['trace'])
    assert 'success' in result and 'message' in result  # retain nonconvergence too


def test_actual_hopping_correlator_gradient_matches_finite_differences():
    links, _, W, v, c = fixture_data()
    rng = np.random.default_rng(92024)
    field = (rng.normal(size=(3, 3, 3, 3, 4))
             + 1j * rng.normal(size=(3, 3, 3, 3, 4)))
    value, analytic = model.correlator_and_gradient(field, links[0], W, v, c)
    parameters = model.pack(W, v, c)
    numeric = np.empty_like(parameters)
    step = 2e-6
    for index in range(len(parameters)):
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += step
        minus[index] -= step
        Wp, vp, cp = model.unpack(plus, len(v))
        Wm, vm, cm = model.unpack(minus, len(v))
        fp = model.correlator_and_gradient(field, links[0], Wp, vp, cp)[0]
        fm = model.correlator_and_gradient(field, links[0], Wm, vm, cm)[0]
        numeric[index] = (fp - fm) / (2 * step)
    assert np.isfinite(value)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize('case', ['real_field', 'wrong_shape', 'nonfinite_links'])
def test_correlator_gradient_rejects_invalid_operator_inputs(case):
    links, _, W, v, c = fixture_data()
    rng = np.random.default_rng(92025)
    field = (rng.normal(size=(3, 3, 3, 3, 2))
             + 1j * rng.normal(size=(3, 3, 3, 3, 2)))
    if case == 'real_field':
        field = field.real
    elif case == 'wrong_shape':
        field = field[..., :2, :]
    else:
        links[0][0, 0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        model.correlator_and_gradient(field, links[0], W, v, c)


def test_zero_initialization_and_homogeneous_links_cannot_demonstrate_W_learning():
    _, z, W, v, c = fixture_data()
    target = model.coefficient_field(z, W, v, c)
    _, gradient = model.loss_and_gradient(model.pack(np.zeros((2, 2)), np.zeros(2), 0.), z, target, 2)
    np.testing.assert_array_equal(gradient[:-1], 0.)
    cold = np.broadcast_to(np.eye(3), (3, 3, 3, 3, 3, 3)).copy()
    a = model.coefficient_field(directional_features(cold)[None], W, v, c)
    np.testing.assert_allclose(a, c, atol=1e-15)


def test_hidden_sign_permutation_nonidentifiability():
    _, z, W, v, c = fixture_data()
    expected = model.coefficient_field(z, W, v, c)
    np.testing.assert_allclose(model.coefficient_field(z, -W, -v, c), expected, atol=1e-15)
    np.testing.assert_allclose(model.coefficient_field(z, W[::-1], v[::-1], c), expected, atol=1e-15)


@pytest.mark.parametrize('case', ['shape', 'complex', 'nonfinite', 'target_shape'])
def test_invalid_inputs_are_rejected(case):
    _, z, W, v, c = fixture_data()
    target = np.zeros(z.shape[:-1])
    if case == 'shape':
        z = z[0]
    elif case == 'complex':
        z = z.astype(complex) + 1j
    elif case == 'nonfinite':
        z.flat[0] = np.nan
    elif case == 'target_shape':
        target = target[:1]
    with pytest.raises(ValueError):
        model.loss_and_gradient(model.pack(W, v, c), z, target, 2)
