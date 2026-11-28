import math

import pytest

from tasks.wilson_loop.ground_state_bound import (
    creutz_excited_correction,
    excited_fraction_bound,
    force_bias_bound,
    minimum_gap_for_tolerance,
)


def test_excited_fraction_and_bias_bound_are_monotone_in_time_and_gap():
    rho4 = excited_fraction_bound(2.0, 0.2, 4)
    rho7 = excited_fraction_bound(2.0, 0.2, 7)
    assert rho4 > rho7 > 0
    assert force_bias_bound(2.0, 0.5, 7) < force_bias_bound(2.0, 0.2, 7)
    assert force_bias_bound(2.0, 0.2, 7) < force_bias_bound(2.0, 0.2, 4)


def test_minimum_gap_satisfies_requested_symbolic_tolerance():
    q, tau, t = 3.0, 0.01, 6
    gap = minimum_gap_for_tolerance(q, tau, t)
    assert force_bias_bound(q, gap, t) == pytest.approx(tau)
    assert minimum_gap_for_tolerance(0.001, tau, t) == 0.0


def test_creutz_excited_correction_obeys_log1p_envelope():
    rho_r_t, rho_r_tp1 = 0.30, 0.11
    rho_r1_t, rho_r1_tp1 = 0.17, 0.04
    delta = creutz_excited_correction(rho_r_t, rho_r_tp1, rho_r1_t, rho_r1_tp1)
    bound = max(math.log1p(rho_r_t), math.log1p(rho_r1_t))
    assert abs(delta) <= bound


@pytest.mark.parametrize("args", [
    (-1.0, 0.2, 4), (1.0, -0.1, 4), (1.0, 0.2, 0),
])
def test_invalid_bound_inputs_are_rejected(args):
    with pytest.raises(ValueError):
        excited_fraction_bound(*args)


def test_increasing_rho_in_time_is_rejected():
    with pytest.raises(ValueError):
        creutz_excited_correction(0.1, 0.2, 0.1, 0.05)
