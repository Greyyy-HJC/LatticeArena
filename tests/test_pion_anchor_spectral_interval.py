from __future__ import annotations

import math

import numpy as np
import pytest

from tasks.pion_projection.anchor_spectral_interval import (
    propagate_ground_mass_from_markov_box,
    propagate_ground_mass_interval,
)
from tasks.pion_projection.bounded_block_risk_anchor_vector import (
    markov_anchor_mean_box,
)


def _periodic_positive_spectrum(masses, amplitudes, time):
    return sum(
        amplitude * math.cosh(mass * (8 - time))
        for mass, amplitude in zip(masses, amplitudes)
    )


def test_positive_two_state_spectrum_maps_to_ground_mass_set():
    ground, excited, effective = 0.15, 0.9, 0.42
    excited_weight = (
        math.cosh(effective) - math.cosh(ground)
    ) / (math.cosh(excited) - math.cosh(ground))
    amplitudes = (
        1.0,
        excited_weight / (1.0 - excited_weight)
        * math.cosh(4.0 * ground) / math.cosh(4.0 * excited),
    )
    c3, c4, c5 = (
        _periodic_positive_spectrum((ground, excited), amplitudes, time)
        for time in (3, 4, 5)
    )
    observed_effective = math.acosh((c3 + c5) / (2.0 * c4))

    result = propagate_ground_mass_interval(
        c3_mean_interval=(c3, c3),
        c4_mean_interval=(c4, c4),
        c5_mean_interval=(c5, c5),
        score_scale_R=2.9,
    )

    lower, upper = result["identified_ground_mass_interval"]
    assert observed_effective == pytest.approx(effective, abs=1e-14)
    assert lower <= ground <= upper
    assert upper == pytest.approx(observed_effective, abs=1e-14)
    assert result["midpoint_minimax_radius_over_R"] == pytest.approx(
        effective / (2.0 * 2.9)
    )
    assert result["midpoint_minimax_radius_over_R"] > 0.01
    assert result["status"] == "PARTIAL_IDENTIFICATION_NOT_CERTIFIED"
    assert result["target_bridge_status"] == "unverified_for_independent_E_reference"
    assert result["physical_score"] is None and not result["certified"]


@pytest.mark.parametrize(("amplitude", "tolerance"), [
    (1e308, 1e-12),
    (1e-320, 7e-4),
])
def test_log_ratio_propagation_is_stable_under_common_scale(amplitude, tolerance):
    mass = 0.2
    means = {
        time: amplitude * math.cosh(mass * (8 - time))
        for time in (3, 4, 5)
    }

    result = propagate_ground_mass_interval(
        c3_mean_interval=(means[3], means[3]),
        c4_mean_interval=(means[4], means[4]),
        c5_mean_interval=(means[5], means[5]),
        score_scale_R=2.9,
    )

    assert result["effective_mass_upper"] == pytest.approx(mass, abs=tolerance)


def test_mean_box_that_cannot_support_positive_spectrum_abstains():
    with pytest.raises(ValueError, match="strictly positive"):
        propagate_ground_mass_interval(
            c3_mean_interval=(0.1, 0.2),
            c4_mean_interval=(-0.1, 0.2),
            c5_mean_interval=(0.1, 0.2),
            score_scale_R=2.9,
        )

    with pytest.raises(ValueError, match="ratio >= 1"):
        propagate_ground_mass_interval(
            c3_mean_interval=(0.01, 0.02),
            c4_mean_interval=(1.0, 1.1),
            c5_mean_interval=(0.01, 0.02),
            score_scale_R=2.9,
        )


def test_markov_vector_box_connects_only_with_frozen_feature_order():
    mass = 0.4
    means = np.asarray([
        math.cosh(mass * (8 - time)) for time in (3, 4, 5)
    ])
    samples = np.tile(means, (4000, 1))
    support = np.column_stack((means * 0.999, means * 1.001))
    box = markov_anchor_mean_box(
        samples,
        support_bounds=support,
        gamma_ps_lower=0.2,
        alpha_anchor=0.01,
        stationary_start=True,
    )

    result = propagate_ground_mass_from_markov_box(
        box,
        feature_order=("C_fold_3", "C_fold_4", "C_fold_5"),
        score_scale_R=2.9,
    )

    lower, upper = result["identified_ground_mass_interval"]
    assert lower <= mass <= upper
    assert result["input_markov_mean_box"][
        "joint_coverage_probability_lower_bound"
    ] == pytest.approx(0.99)
    assert result["common_bias_bound_C"] is None
    assert result["physical_score"] is None

    with pytest.raises(ValueError, match="feature_order must be frozen"):
        propagate_ground_mass_from_markov_box(
            box,
            feature_order=("C_fold_4", "C_fold_3", "C_fold_5"),
            score_scale_R=2.9,
        )
