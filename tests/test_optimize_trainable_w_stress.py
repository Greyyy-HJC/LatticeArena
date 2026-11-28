import numpy as np
import pytest

from tasks.optimize_gaussian.trainable_w_operator_diagnostic import (
    direct_loss,
    operator_loss,
    solve,
)
from tasks.optimize_gaussian.trainable_w_stress_diagnostic import (
    _bootstrap_median_interval,
    _feature_condition,
    _make_seed_plan,
    fit_local_polynomial_baseline,
    make_case,
    weak_field_links,
)
from tasks.optimize_gaussian import trainable_w as model


def test_weak_field_generator_is_su3_and_approaches_identity():
    rng = np.random.default_rng(51001)
    links = weak_field_links(rng, (3, 2, 2, 2), epsilon=0.03)
    identity = np.broadcast_to(np.eye(3), links.shape)
    np.testing.assert_allclose(links.conj().swapaxes(-1, -2) @ links,
                               identity, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(links), np.ones(links.shape[:-2]),
                               rtol=1e-12, atol=1e-12)
    assert np.max(np.abs(links - identity)) < 0.2


def test_local_target_is_link_varying_and_noncommuting_with_uniform_hopping():
    rng = np.random.default_rng(51002)
    local = make_case(rng, "haar_like", "local_environment", 0.0)
    spectral = make_case(rng, "haar_like", "spectral_heat", 0.0)
    assert local["target_checks"]["relative_commutator_with_uniform_hopping"] > 1e-5
    assert spectral["target_checks"]["relative_commutator_with_uniform_hopping"] < 1e-12
    assert local["target_checks"]["hermiticity_relative"] < 1e-12


def test_relative_operator_noise_has_declared_frobenius_scale():
    data = make_case(np.random.default_rng(51009), "haar_like",
                     "local_environment", 0.02)
    observed = np.linalg.norm(data["target"] - data["noiseless_target"])
    expected = 0.02 * np.linalg.norm(data["noiseless_target"])
    assert observed == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_weak_field_feature_design_reports_ill_conditioning():
    cases = [make_case(np.random.default_rng(seed), generator,
                       "local_environment", 0.0)
             for seed, generator in ((51010, "haar_like"),
                                    (51011, "weak_field_eps0.08"))]
    haar = _feature_condition([cases[0]])["condition_number"]
    weak = _feature_condition([cases[1]])["condition_number"]
    assert haar is not None and weak is not None
    assert weak > haar * 100


def test_fixed_local_polynomial_operator_baseline_recovers_noise_free_teacher():
    fit = [make_case(np.random.default_rng(seed), "weak_field_eps0.25",
                     "local_environment", 0.0) for seed in (51003, 51004)]
    heldout = [make_case(np.random.default_rng(seed), "weak_field_eps0.25",
                         "local_environment", 0.0) for seed in (51005, 51006)]
    result = fit_local_polynomial_baseline(fit, heldout)
    assert result["design_rank"] == result["parameters"]
    assert result["fit_loss"] < 1e-24
    assert result["heldout_loss"] < 1e-24


def test_operator_loss_compression_matches_dense_target_on_local_fixture():
    data = make_case(np.random.default_rng(51007), "haar_like",
                     "local_environment", 0.02)
    parameters = model.pack(np.array([[0.2, -0.1], [0.3, 0.15]]),
                            np.array([0.2, -0.1]), 0.03)
    compressed = operator_loss(parameters, [data], hidden=2)[0]
    dense = direct_loss(parameters, [data])
    assert compressed == pytest.approx(dense, abs=1e-12)


def test_solver_retains_call_cap_and_non_success_diagnostics():
    data = [make_case(np.random.default_rng(51008), "haar_like",
                      "spectral_heat", 0.0)]
    initial = model.pack(np.array([[0.1, 0.2], [-0.2, 0.1]]),
                         np.array([0.1, -0.1]), 0.02)
    result = solve(initial, data, data, "SLSQP", budget=1, joint=True)
    assert result["calls"] <= 1
    assert "budget_exhausted" in result
    assert "solver_success" in result
    assert np.isfinite(result["projected_gradient_norm"])


def test_replication_plan_is_balanced_and_seeded_per_cell():
    plan = _make_seed_plan(5)
    assert len(plan) == 3 * 2 * 2 * 5
    assert len({row["seed"] for row in plan}) == len(plan)
    cells = {}
    for row in plan:
        key = (row["generator"], row["target_family"], row["noise_level"])
        cells[key] = cells.get(key, 0) + 1
    assert set(cells.values()) == {5}
    with pytest.raises(ValueError, match="at least 2"):
        _make_seed_plan(1)


def test_panel_median_bootstrap_band_is_deterministic_and_validated():
    values = np.array([-0.3, -0.1, 0.2, 0.4, 0.8])
    first = _bootstrap_median_interval(values, seed=51012, draws=1000)
    second = _bootstrap_median_interval(values, seed=51012, draws=1000)
    assert first == second
    assert first[0] <= first[1]
    with pytest.raises(ValueError, match="at least two"):
        _bootstrap_median_interval([1.0], seed=1)
