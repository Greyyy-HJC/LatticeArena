import json

import numpy as np
import pytest

from tasks.wilson_loop import f_lane_finite_nt_trace_diagnostic as toy
from tasks.wilson_loop import f_lane_two_sector_multistratum_screen as screen
from tasks.wilson_loop import f_lane_two_sector_omitted_coverage_diagnostic as base


def test_frozen_screen_protocol_has_five_disjoint_seed_strata():
    protocol, _ = screen._load_protocol()
    assert [row["id"] for row in protocol["strata"]] == list(screen.STRATUM_IDS)
    assert protocol["classification"] == "SYNTHETIC_SCREENING_ONLY_NOT_CALIBRATION"
    seeds = {
        int(protocol["seed_rule"].split("+")[0].strip())
        + 100 * stratum_index + panel_index
        for stratum_index in range(5)
        for panel_index in range(protocol["panels_per_stratum"])
    }
    old_seeds = set(range(2026112800, 2026112824))
    assert len(seeds) == 60
    assert seeds.isdisjoint(old_seeds)
    assert protocol["interpretation_boundary"]["calibration_claim"] is False


@pytest.mark.parametrize("stratum_index", range(5))
def test_scenario_truth_transforms_are_exact_and_restored(stratum_index):
    protocol, _ = screen._load_protocol()
    original_vacuum = toy.VACUUM_ENERGIES.copy()
    original_static = {key: value.copy() for key, value in toy.STATIC_ENERGIES.items()}
    original_overlaps = toy.TRANSITION_OVERLAPS.copy()
    scenario = protocol["strata"][stratum_index]
    with screen._scenario_truth(scenario):
        if stratum_index == 0:
            np.testing.assert_allclose(
                toy.TRANSITION_OVERLAPS[:, :, 2, :],
                1.5 * original_overlaps[:, :, 2, :],
            )
        elif stratum_index == 1:
            np.testing.assert_allclose(toy.STATIC_ENERGIES[2], [0.44, 0.94, 0.965])
            np.testing.assert_allclose(toy.STATIC_ENERGIES[3], [0.59, 1.07, 1.095])
        elif stratum_index == 2:
            np.testing.assert_allclose(
                toy.TRANSITION_OVERLAPS[:, :, 0, 2],
                0.04 * original_overlaps[:, :, 0, 2],
            )
        elif stratum_index == 3:
            assert toy.TRANSITION_OVERLAPS.shape == (2, 3, 3, 3)
            np.testing.assert_allclose(toy.VACUUM_ENERGIES, [0.0, 0.42, 0.70])
            np.testing.assert_allclose(
                toy.TRANSITION_OVERLAPS[:, 2], 0.25 * original_overlaps[:, 0],
            )
        else:
            np.testing.assert_array_equal(toy.VACUUM_ENERGIES, original_vacuum)
    np.testing.assert_array_equal(toy.VACUUM_ENERGIES, original_vacuum)
    np.testing.assert_array_equal(toy.TRANSITION_OVERLAPS, original_overlaps)
    for radius in original_static:
        np.testing.assert_array_equal(toy.STATIC_ENERGIES[radius], original_static[radius])


def test_ar1_panel_is_reproducible_hermitian_and_correlated():
    protocol, _ = screen._load_protocol()
    scenario = protocol["strata"][4]
    noise = screen._noise_spec(scenario)
    with screen._scenario_truth(scenario):
        first = base._generate_matrices(2026113300, noise)
        repeated = base._generate_matrices(2026113300, noise)
        truth = toy.finite_trace_matrix(2, 3, 16)
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_allclose(first, np.swapaxes(first.conj(), -1, -2), atol=1e-12)
    sigma = max(float(np.max(np.abs(truth))) * base.NOISE_FRACTION * 0.5, 1e-9)
    residual = (first[:, 0, 0, 0, 0, 0].real - truth[0, 0].real) / sigma
    lag_one = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
    assert 0.15 < lag_one < 0.85


def test_generator_rejects_invalid_ar1_and_scale_contracts():
    with pytest.raises(ValueError, match="rho"):
        base._generate_matrices(1, {"model": "stationary_ar1", "rho": 1.0})
    with pytest.raises(ValueError, match="scale_by_T"):
        base._generate_matrices(1, {
            "model": "iid_gaussian", "scale_by_T": {"3": 0.0},
        })


def test_wilson_monte_carlo_interval_handles_boundary_counts_exactly():
    assert base._wilson_interval(0, 24)[0] == 0.0
    assert base._wilson_interval(24, 24)[1] == 1.0
    assert base._wilson_interval(0, 0) == [None, None]
