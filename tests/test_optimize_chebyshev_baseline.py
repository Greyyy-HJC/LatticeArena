"""CPU algebra checks for the proposed constrained Chebyshev baseline."""
import numpy as np
import pytest

from tasks.optimize_gaussian.chebyshev_baseline import (
    apply_filter,
    apply_gaussian_sandwich_cpu,
    apply_hopping_filter_cpu,
    bernstein_to_chebyshev,
    evaluate_scaled_laplacian,
    validate_control_points,
)
from tasks.optimize_gaussian.kernels import apply_kernel


def _small_unitary_links(shape):
    z, y, x = np.indices(shape)
    links = np.empty((3, *shape, 3, 3), dtype=complex)
    for d in range(3):
        theta0 = 0.13 * (1 + d + z + 2 * y + 3 * x)
        theta1 = -0.07 * (2 + 2 * d + 3 * z - y + x)
        phases = (theta0, theta1, -theta0 - theta1)
        links[d] = 0.0
        for color, phase in enumerate(phases):
            links[d, ..., color, color] = np.exp(1j * phase)
    return links


@pytest.mark.parametrize("controls", [
    np.ones(5),
    np.linspace(0.0, 1.0, 5),
    np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
    np.array([0.2, 0.3, 0.4, 0.7, 1.0]),
])
def test_response_is_a_monotone_positive_contraction(controls):
    lambdas = np.linspace(0.0, 1.0, 1001)
    response = evaluate_scaled_laplacian(controls, lambdas)
    assert response[0] == pytest.approx(1.0)
    assert response.min() >= -1e-14
    assert response.max() <= 1.0 + 1e-14
    # Response decreases as the Laplacian eigenvalue increases.
    assert np.all(np.diff(response) <= 1e-14)


def test_chebyshev_conversion_reproduces_bernstein_polynomial():
    controls = np.array([0.1, 0.25, 0.35, 0.8, 1.0])
    coefficients = bernstein_to_chebyshev(controls)
    fractions = np.linspace(0.0, 1.0, 101)
    x = 1.0 - 2.0 * fractions
    from_chebyshev = np.polynomial.chebyshev.chebval(x, coefficients)
    from_bernstein = evaluate_scaled_laplacian(controls, fractions)
    np.testing.assert_allclose(from_chebyshev, from_bernstein, rtol=2e-13, atol=2e-13)


def test_fixed_rational_basis_map_has_expected_constant_and_endpoint_columns():
    # These exact dyadic coefficients come from expanding the Bernstein basis
    # after y=(x+1)/2 and converting monomials to Chebyshev polynomials.
    np.testing.assert_array_equal(
        bernstein_to_chebyshev(np.ones(5)),
        np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        bernstein_to_chebyshev(np.array([0.0, 0.0, 0.0, 0.0, 1.0])),
        np.array([35.0, 56.0, 28.0, 8.0, 1.0]) / 128.0,
    )


def test_matrix_application_matches_spectral_function_and_is_contractive():
    eigenvalues = np.linspace(-12.0, 0.0, 11)
    h = np.diag(eigenvalues).astype(complex)
    rng = np.random.default_rng(2026112201)
    field = rng.normal(size=11) + 1j * rng.normal(size=11)
    controls = np.array([0.0, 0.0, 0.1, 0.7, 1.0])
    result = apply_filter(controls, h, field)
    expected = evaluate_scaled_laplacian(controls, -eigenvalues / 12.0) * field
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert np.linalg.norm(result) <= np.linalg.norm(field) + 1e-12


def test_positive_spectral_filter_stays_positive_in_gaussian_sandwich():
    """Algebraic fixture only; alpha and step count are not physical choices."""
    controls = np.array([0.0, 0.15, 0.4, 0.7, 1.0])
    h = np.diag([-12.0, -6.0, 0.0]).astype(complex)
    alpha, steps = 0.2, 3

    # For constant-alpha Gaussian steps, G=(1+6 alpha)I+alpha H.
    # This fixture makes G indefinite, so positivity cannot rely on G >= 0.
    g = (1.0 + 6.0 * alpha) * np.eye(3) + alpha * h
    assert np.linalg.eigvalsh(g)[0] < 0.0

    basis = np.eye(3, dtype=complex)
    p = np.column_stack([
        apply_filter(controls, h, basis[:, j]) for j in range(basis.shape[1])
    ])
    np.testing.assert_allclose(p, p.conj().T, rtol=0.0, atol=1e-12)
    sandwich = np.linalg.matrix_power(g, steps) @ p @ np.linalg.matrix_power(g, steps)
    assert np.linalg.eigvalsh((sandwich + sandwich.conj().T) / 2).min() >= -1e-12

    # P4(0)=1 for the zero mode; no per-step normalization is present.
    zero_mode_scale = (1.0 + 6.0 * alpha) ** (2 * steps)
    assert sandwich[2, 2].real == pytest.approx(zero_mode_scale)


def test_uniform_kernel_matches_affine_hopping_operator_on_periodic_links():
    """Check the proposed H/G identity against the repository's CPU kernel."""
    shape = (2, 2, 2)
    links = _small_unitary_links(shape)

    rng = np.random.default_rng(2026112203)
    field = rng.normal(size=(*shape, 3, 2)) + 1j * rng.normal(size=(*shape, 3, 2))
    unit_coefficients = np.ones((3, *shape))
    h_field = apply_kernel(field, links, unit_coefficients) - 7.0 * field

    alpha = 0.173  # Algebraic fixture, not a proposed production value.
    gaussian_coefficients = np.full((3, *shape), alpha)
    gaussian_field = apply_kernel(field, links, gaussian_coefficients)
    predicted = (1.0 + 6.0 * alpha) * field + alpha * h_field
    np.testing.assert_allclose(gaussian_field, predicted, rtol=1e-13, atol=1e-13)

    other = rng.normal(size=field.shape) + 1j * rng.normal(size=field.shape)
    h_other = apply_kernel(other, links, unit_coefficients) - 7.0 * other
    np.testing.assert_allclose(np.vdot(field, h_other), np.vdot(h_field, other),
                               rtol=1e-13, atol=1e-12)


def test_matrix_free_chebyshev_filter_matches_dense_operator_from_kernel():
    shape = (2, 2, 2)
    links = _small_unitary_links(shape)
    volume_color = int(np.prod(shape) * 3)
    basis = np.eye(volume_color, dtype=complex).reshape(*shape, 3, volume_color)
    unit_coefficients = np.ones((3, *shape))
    h_field = apply_kernel(basis, links, unit_coefficients) - 7.0 * basis
    h = h_field.reshape(volume_color, volume_color)

    controls = np.array([0.02, 0.18, 0.43, 0.72, 1.0])
    rng = np.random.default_rng(2026112204)
    field = rng.normal(size=(*shape, 3, 2)) + 1j * rng.normal(size=(*shape, 3, 2))
    dense = apply_filter(controls, h, field.reshape(volume_color, 2))
    matrix_free = apply_hopping_filter_cpu(controls, links, field)
    np.testing.assert_allclose(matrix_free.reshape(volume_color, 2), dense,
                               rtol=2e-11, atol=2e-11)
    assert np.linalg.norm(matrix_free) <= np.linalg.norm(field) + 1e-11


def test_matrix_free_chebyshev_filter_rejects_nonunitary_links():
    shape = (2, 2, 2)
    links = _small_unitary_links(shape)
    links[0, 0, 0, 0, 0, 0] *= 1.1
    field = np.ones((*shape, 3, 1), dtype=complex)
    with pytest.raises(ValueError, match="unitary"):
        apply_hopping_filter_cpu(np.linspace(0.0, 1.0, 5), links, field)


def test_matrix_free_gaussian_sandwich_matches_dense_operator():
    shape = (2, 2, 2)
    links = _small_unitary_links(shape)
    volume_color = int(np.prod(shape) * 3)
    basis = np.eye(volume_color, dtype=complex).reshape(*shape, 3, volume_color)
    h = (apply_kernel(basis, links, np.ones((3, *shape))) - 7.0 * basis).reshape(
        volume_color, volume_color)

    controls = np.array([0.0, 0.12, 0.38, 0.74, 1.0])
    alpha, steps = 0.137, 2
    g = (1.0 + 6.0 * alpha) * np.eye(volume_color) + alpha * h
    p = apply_filter(controls, h, np.eye(volume_color, dtype=complex))
    g_power = np.linalg.matrix_power(g, steps)
    operator = g_power @ p @ g_power
    np.testing.assert_allclose(operator, operator.conj().T, rtol=0.0, atol=2e-11)
    assert np.linalg.eigvalsh((operator + operator.conj().T) / 2).min() >= -2e-11

    rng = np.random.default_rng(2026112205)
    field = rng.normal(size=(*shape, 3, 2)) + 1j * rng.normal(size=(*shape, 3, 2))
    expected = (operator @ field.reshape(volume_color, 2)).reshape(field.shape)
    actual = apply_gaussian_sandwich_cpu(
        controls, links, field, alpha=alpha, steps=steps)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)


def test_polynomial_application_is_unitary_similarity_covariant():
    rng = np.random.default_rng(2026112202)
    raw = rng.normal(size=(7, 7)) + 1j * rng.normal(size=(7, 7))
    unitary, _ = np.linalg.qr(raw)
    spectrum = np.linspace(-11.5, -0.5, 7)
    h = np.diag(spectrum).astype(complex)
    field = rng.normal(size=7) + 1j * rng.normal(size=7)
    controls = np.array([0.05, 0.15, 0.4, 0.75, 1.0])
    direct = apply_filter(controls, h, field)
    transformed = apply_filter(controls, unitary @ h @ unitary.conj().T,
                               unitary @ field)
    np.testing.assert_allclose(transformed, unitary @ direct, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("controls", [
    [0, 0, 0, 0, 0],
    [0, 0.2, 0.1, 0.8, 1],
    [-0.1, 0.2, 0.4, 0.8, 1],
    [0, 0.2, 0.4, 1.1, 1],
    [0, 0.2, 0.4, 0.8, 0.99],
    [0, 0.2, float("nan"), 0.8, 1],
])
def test_invalid_control_points_reject(controls):
    with pytest.raises(ValueError):
        validate_control_points(controls)


@pytest.mark.parametrize("controls", [
    np.asarray([False, False, False, False, True]),
    np.asarray(["0", "0.2", "0.4", "0.8", "1"]),
    np.asarray([0.0, 0.2, 0.4, 0.8, 1.0 + 0.0j]),
])
def test_control_points_reject_implicit_numeric_coercion(controls):
    with pytest.raises(ValueError, match="control_points"):
        validate_control_points(controls)


@pytest.mark.parametrize("fraction", [True, "0.5", 0.5 + 0.0j])
def test_scaled_spectrum_coordinate_rejects_implicit_numeric_coercion(fraction):
    with pytest.raises(ValueError, match="lambda_fraction"):
        evaluate_scaled_laplacian([0.0, 0.2, 0.4, 0.8, 1.0], fraction)


def test_candidate_scalar_controls_reject_bool_and_string_values():
    links = _small_unitary_links((2, 2, 2))
    field = np.ones((2, 2, 2, 3, 1), dtype=complex)
    controls = [0.0, 0.2, 0.4, 0.8, 1.0]

    with pytest.raises(ValueError, match="alpha"):
        apply_gaussian_sandwich_cpu(controls, links, field, alpha="0.1", steps=1)
    with pytest.raises(ValueError, match="unitary_tolerance"):
        apply_hopping_filter_cpu(controls, links, field, unitary_tolerance=True)
    with pytest.raises(ValueError, match="hermitian_tolerance"):
        apply_filter(controls, np.diag([-1.0, 0.0]), np.ones(2),
                     hermitian_tolerance="1e-10")


def test_matrix_application_rejects_spectrum_or_hermiticity_violation():
    with pytest.raises(ValueError, match="spectrum"):
        apply_filter(np.linspace(0.0, 1.0, 5), np.diag([-13.0, -1.0]), np.ones(2))
    nonhermitian = np.array([[0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="Hermitian"):
        apply_filter(np.linspace(0.0, 1.0, 5), nonhermitian, np.ones(2))


def test_slsqp_control_fit_recovers_a_feasible_fixed_objective_minimum():
    from tasks.optimize_gaussian.chebyshev_baseline import fit_control_points_slsqp

    target = np.array([0.08, 0.29, 0.51, 0.77, 1.0])

    def objective(controls):
        return np.sum((controls - target) ** 2)

    result = fit_control_points_slsqp(
        objective, max_objective_calls=200, max_iterations=100, ftol=1e-12
    )

    assert result["solver_success"]
    assert result["status"] == "SLSQP_SUCCESS"
    assert result["objective_calls"] <= 200
    assert result["feasibility_residual"] == 0.0
    np.testing.assert_allclose(result["control_points"], target, atol=2e-6, rtol=0.0)
    validate_control_points(result["control_points"])


def test_slsqp_control_fit_stops_at_objective_call_cap_and_retains_best_feasible():
    from tasks.optimize_gaussian.chebyshev_baseline import fit_control_points_slsqp

    objective_calls = []

    def objective(controls):
        objective_calls.append(controls.copy())
        return float(np.sum(controls**2))

    result = fit_control_points_slsqp(
        objective, max_objective_calls=1, max_iterations=100
    )

    assert result["status"] == "OBJECTIVE_CALL_BUDGET_EXHAUSTED"
    assert not result["solver_success"]
    assert result["objective_calls"] == len(objective_calls) == 1
    assert result["feasibility_residual"] == 0.0
    validate_control_points(result["control_points"])


def test_slsqp_control_fit_rejects_nonfinite_objective_and_bad_budget():
    from tasks.optimize_gaussian.chebyshev_baseline import fit_control_points_slsqp

    with pytest.raises(ValueError, match="finite real scalar"):
        fit_control_points_slsqp(
            lambda _controls: np.nan,
            max_objective_calls=10,
            max_iterations=10,
        )
    with pytest.raises(ValueError, match="max_objective_calls"):
        fit_control_points_slsqp(
            lambda controls: float(np.sum(controls**2)),
            max_objective_calls=0,
            max_iterations=10,
        )
