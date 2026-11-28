import math

import pytest

from tasks.wilson_loop.c_lane_markov_support_bound import (
    FAMILY,
    LANE_ALPHA,
    creutz_box_interval,
    loop_mean_interval,
    markov_support_radius,
    required_blocks,
    worst_case_halfwidth,
)


def test_radius_shrinks_with_more_samples_and_larger_gap():
    n100 = markov_support_radius(blocks=100, gamma_ps_lower=0.5)["radius_normalized_score"]
    n1000 = markov_support_radius(blocks=1000, gamma_ps_lower=0.5)["radius_normalized_score"]
    better_gap = markov_support_radius(blocks=100, gamma_ps_lower=1.0)["radius_normalized_score"]
    assert n1000 < n100
    assert better_gap < n100


def test_radius_uses_two_sided_familywise_alpha():
    details = markov_support_radius(blocks=100, gamma_ps_lower=1.0)
    assert details["family"] == FAMILY == 4
    assert details["alpha_family"] == LANE_ALPHA == 0.025
    assert details["log_tail"] == pytest.approx(math.log(2 * FAMILY / LANE_ALPHA))


def test_support_map_and_monotone_creutz_interval_enclose_point():
    lo, hi = loop_mean_interval(0.5, 0.01)
    assert lo < 0.5 < hi
    result = creutz_box_interval([0.3, 0.4, 0.5, 0.6], 0.001)
    point = math.log(0.5) + math.log(0.6) - math.log(0.3) - math.log(0.4)
    assert result["status"] == "FINITE_SIMULTANEOUS_BOX"
    assert result["lower"] <= point <= result["upper"]
    assert result["halfwidth"] > 0


def test_nonpositive_loop_lower_bound_abstains_before_log():
    result = creutz_box_interval([0.3, 0.4, 0.5, 0.6], 2.0)
    assert result["status"] == "ABSTAIN_NONPOSITIVE_LOOP_LCB"
    assert result["lower"] is None
    assert result["upper"] is None


def test_required_sample_size_respects_gap_and_mean_floor_ordering():
    best = required_blocks(gamma_ps_lower=1.0, mean_floor=1.0)
    weaker_gap = required_blocks(gamma_ps_lower=0.1, mean_floor=1.0)
    smaller_mean = required_blocks(gamma_ps_lower=1.0, mean_floor=0.5)
    assert best == 222_616
    assert weaker_gap > best
    assert smaller_mean > best


@pytest.mark.parametrize("bad", [True, 0, -0.1, float("nan"), float("inf")])
def test_invalid_gap_values_fail_closed(bad):
    with pytest.raises(ValueError):
        markov_support_radius(blocks=100, gamma_ps_lower=bad)
