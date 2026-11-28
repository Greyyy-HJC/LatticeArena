import hashlib

import numpy as np
import pytest

from tasks.pion_projection.bounded_block_risk_setwise import (
    bounded_block_risk_setwise,
)
from tasks.pion_projection.bounded_block_risk_setwise_markov import (
    setwise_markov_score_lcb_from_gap_bound,
)
from tasks.pion_projection.bounded_block_risk_markov_varying import (
    markov_time_varying_score_lcb_from_gap_bound,
    time_varying_markov_radius_from_gap_bound,
)


def receipt(label):
    return hashlib.sha256(label.encode()).hexdigest()


def arguments():
    return {
        "valid_blocks": np.asarray([True, True, False, True]),
        "target_interval": [-0.3, 0.7],
        "prediction_domain": [-1.0, 1.0],
        "score_scale_R": 2.0,
        "mesh_step_R": 0.25,
        "gamma_ps_lower": 0.8,
        "family": 2,
        "alpha_score": 0.04,
        "alpha_anchor": 0.01,
        "alpha_total": 0.05,
        "initial_law_mode": "stationary",
        "burn_in_blocks": 0,
        "initial_law_tv_bound": 0.0,
        "anchor_score_shift_bound": 0.0,
        "gamma_ps_lower_receipt_sha256": receipt("gamma"),
        "dependence_evidence_receipt_sha256": receipt("dependence"),
        "block_chain_receipt_sha256": receipt("block-chain"),
        "initial_law_receipt_sha256": receipt("initial"),
        "anchor_shift_receipt_sha256": receipt("anchor-shift"),
        "score_schedule_receipt_sha256": receipt("score-schedule-family"),
        "target_set_coverage_receipt_sha256": receipt("target-set"),
        "prediction_domain_receipt_sha256": receipt("prediction-domain"),
    }


def test_uniform_markov_lcb_combines_simultaneous_grid_bounds_and_setwise_penalty():
    reference = np.asarray([0.0, 0.4, np.nan, -0.2])
    candidate = np.asarray([0.5, -0.3, np.nan, 0.8])
    kwargs = arguments()
    result = setwise_markov_score_lcb_from_gap_bound(
        reference, candidate, **kwargs,
    )

    assert result["status"] == "CONDITIONAL_UNIFORM_BOUND_ONLY"
    assert result["certified"] is False
    assert result["physical_score"] is None
    assert result["confidence_coverage"] is None
    assert result["data_access"] == "none"
    assert result["target_grid"][0] == kwargs["target_interval"][0]
    assert result["target_grid"][-1] == kwargs["target_interval"][1]
    assert result["maximum_grid_gap_over_R_upper_bound"] <= kwargs["mesh_step_R"]
    assert result["simultaneous_family_size"] == (
        kwargs["family"] * result["target_grid_points"]
    )
    grid_min = min(row["normalized_lower_bound"]
                   for row in result["pointwise_bounds"])
    expected_uniform_lcb = max(
        -1.0,
        np.nextafter(
            grid_min - result["maximum_grid_gap_over_R_upper_bound"], -np.inf,
        ),
    )
    assert result["uniform_target_set_lower_bound"] == pytest.approx(expected_uniform_lcb)
    assert result["denominators"] == {
        "planned_blocks": 4, "valid_blocks": 3,
        "failed_blocks": 1, "abstained_blocks": 0,
    }
    assert all(row["denominators"]["failed"] == 1
               for row in result["pointwise_bounds"])

    finite_plan = bounded_block_risk_setwise(
        reference, candidate,
        valid_blocks=kwargs["valid_blocks"],
        target_interval=kwargs["target_interval"],
        score_scale_R=kwargs["score_scale_R"],
    )
    # The Markov LCB is also below the finite-plan target-set minimum because
    # concentration, initial-law, and anchor corrections are nonnegative.
    assert result["uniform_target_set_lower_bound"] <= (
        finite_plan["normalized_score_interval"][0]
    )
    grid_mean_floor = min(row["planned_mean"] for row in result["pointwise_bounds"])
    assert grid_mean_floor - result["maximum_grid_gap_over_R_upper_bound"] <= (
        finite_plan["normalized_score_interval"][0]
    )
    expected_radius = time_varying_markov_radius_from_gap_bound(
        blocks=4,
        gamma_ps_lower=kwargs["gamma_ps_lower"],
        family=result["simultaneous_family_size"],
        alpha_stationary=kwargs["alpha_score"],
    )["radius"]
    assert all(row["radius"] == pytest.approx(expected_radius)
               for row in result["pointwise_bounds"])


def test_uniform_bound_abstains_when_every_planned_block_fails():
    kwargs = arguments()
    kwargs["valid_blocks"] = np.asarray([False, False])
    result = setwise_markov_score_lcb_from_gap_bound(
        [np.nan, np.nan], [np.nan, np.nan], **kwargs,
    )
    assert result["status"] == "ABSTAIN_NO_VALID_BLOCKS"
    assert result["uniform_target_set_lower_bound"] is None
    assert result["physical_score"] is None
    assert result["denominators"] == {
        "planned_blocks": 2, "valid_blocks": 0,
        "failed_blocks": 2, "abstained_blocks": 0,
    }


def test_singleton_target_set_matches_direct_markov_lcb_with_tv_and_anchor_shift():
    kwargs = arguments()
    kwargs.update({
        "valid_blocks": np.asarray([True, False]),
        "target_interval": [0.2, 0.2],
        "prediction_domain": [-0.5, 0.5],
        "mesh_step_R": 0.5,
        "family": 1,
        "initial_law_mode": "post_burn_in_tv",
        "burn_in_blocks": 3,
        "initial_law_tv_bound": 0.001,
        "anchor_score_shift_bound": 0.02,
    })
    reference = np.asarray([-0.25, np.nan])
    candidate = np.asarray([0.25, np.nan])
    result = setwise_markov_score_lcb_from_gap_bound(
        reference, candidate, **kwargs,
    )
    point_score = ((0.2 - (-0.25)) ** 2 - (0.2 - 0.25) ** 2) / 2.0**2
    direct = markov_time_varying_score_lcb_from_gap_bound(
        [point_score, -1.0],
        valid_blocks=kwargs["valid_blocks"],
        gamma_ps_lower=kwargs["gamma_ps_lower"],
        family=kwargs["family"],
        alpha_score=kwargs["alpha_score"],
        alpha_anchor=kwargs["alpha_anchor"],
        alpha_total=kwargs["alpha_total"],
        initial_law_mode=kwargs["initial_law_mode"],
        burn_in_blocks=kwargs["burn_in_blocks"],
        initial_law_tv_bound=kwargs["initial_law_tv_bound"],
        anchor_score_shift_bound=kwargs["anchor_score_shift_bound"],
        gamma_ps_lower_receipt_sha256=kwargs["gamma_ps_lower_receipt_sha256"],
        dependence_evidence_receipt_sha256=kwargs["dependence_evidence_receipt_sha256"],
        block_chain_receipt_sha256=kwargs["block_chain_receipt_sha256"],
        initial_law_receipt_sha256=kwargs["initial_law_receipt_sha256"],
        anchor_shift_receipt_sha256=kwargs["anchor_shift_receipt_sha256"],
        score_schedule_receipt_sha256=kwargs["score_schedule_receipt_sha256"],
    )

    assert result["target_grid"] == [0.2]
    assert result["maximum_grid_gap_over_R_upper_bound"] == 0.0
    assert result["simultaneous_family_size"] == 1
    assert result["pointwise_bounds"][0]["normalized_lower_bound"] == pytest.approx(
        direct["normalized_lower_bound"]
    )
    assert result["uniform_target_set_lower_bound"] == pytest.approx(
        max(-1.0, np.nextafter(direct["normalized_lower_bound"], -np.inf))
    )
    assert result["pointwise_bounds"][0]["denominators"] == {
        "planned": 2, "valid": 1, "failed": 1, "abstained": 0,
    }


@pytest.mark.parametrize("kwargs, message", [
    ({"prediction_domain": [-1.0, 1.1]}, "width must not exceed"),
    ({"prediction_domain": [-0.5, 0.5]}, "fall outside prediction_domain"),
    ({"mesh_step_R": 0.0}, "mesh_step_R must be in"),
    ({"max_grid_points": 2}, "exceeds max_grid_points"),
    ({"max_grid_points": 4097}, "hard cap 4096"),
    ({"target_set_coverage_receipt_sha256": "bad"}, "SHA-256"),
])
def test_uniform_route_fails_closed_on_missing_geometry_or_receipt(kwargs, message):
    arguments_ = arguments()
    arguments_.update(kwargs)
    with pytest.raises(ValueError, match=message):
        setwise_markov_score_lcb_from_gap_bound(
            [0.0, 0.4, np.nan, -0.2],
            [0.5, -0.3, np.nan, 0.8],
            **arguments_,
        )


def test_uniform_route_rejects_lossy_numeric_inputs_and_bad_prediction_domain():
    kwargs = arguments()
    kwargs["target_interval"] = [-(10**400), 0.7]
    with pytest.raises(ValueError, match="not exactly representable"):
        setwise_markov_score_lcb_from_gap_bound(
            [0.0, 0.4, np.nan, -0.2],
            [0.5, -0.3, np.nan, 0.8],
            **kwargs,
        )

    kwargs = arguments()
    kwargs["prediction_domain"] = [-1.0, 1.0 + 1e-12]
    with pytest.raises(ValueError, match="width must not exceed"):
        setwise_markov_score_lcb_from_gap_bound(
            [0.0, 0.4, np.nan, -0.2],
            [0.5, -0.3, np.nan, 0.8],
            **kwargs,
        )
