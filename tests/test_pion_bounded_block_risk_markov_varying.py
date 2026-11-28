import math

import numpy as np
import pytest

from tasks.pion_projection.bounded_block_risk_markov_varying import (
    markov_time_varying_score_lcb,
    markov_time_varying_score_lcb_from_gap_bound,
    time_varying_markov_radius,
    time_varying_markov_radius_from_gap_bound,
)


def receipt(label):
    import hashlib
    return hashlib.sha256(label.encode()).hexdigest()


def lcb_args():
    return {
        "gamma_ps_lower": 0.5,
        "k_ps": 2,
        "variance_upper_by_residue": [1.0, 1.0],
        "family": 2,
        "alpha_score": 0.04,
        "alpha_anchor": 0.01,
        "alpha_total": 0.05,
        "initial_law_mode": "post_burn_in_tv",
        "burn_in_blocks": 8,
        "initial_law_tv_bound": 0.005,
        "anchor_score_shift_bound": 0.03,
        "score_schedule_receipt_sha256": receipt("schedule"),
        "variance_receipt_sha256": receipt("variance"),
        "k_ps_receipt_sha256": receipt("k-ps"),
        "initial_law_receipt_sha256": receipt("initial-law"),
        "anchor_shift_receipt_sha256": receipt("anchor-shift"),
        "dependence_evidence_receipt_sha256": receipt("dependence-evidence"),
        "block_chain_receipt_sha256": receipt("block-chain"),
        "gamma_ps_lower_receipt_sha256": receipt("gamma-ps-lower"),
    }


def test_varying_radius_solves_eq_326_exponent_with_residue_bounds():
    n = 20
    gamma = 0.5
    family = 3
    alpha = 0.04
    details = time_varying_markov_radius(
        blocks=n,
        gamma_ps_lower=gamma,
        k_ps=2,
        variance_upper_by_residue=[1.0, 1.0],
        family=family,
        alpha_stationary=alpha,
    )
    t = n * details["radius"]
    exponent = (t * t * gamma) / (
        8 * 2.0 + 20 * t * 2.0 * details["M_over_k_ps"]
    )
    assert details["M"] == pytest.approx(2.0)
    assert details["M_over_k_ps"] == pytest.approx(1.0)
    assert exponent == pytest.approx(math.log(2 * family / alpha))


def test_gap_only_radius_uses_support_variance_and_conservative_integer_ceiling():
    details = time_varying_markov_radius_from_gap_bound(
        blocks=20,
        gamma_ps_lower=0.5,
        family=3,
        alpha_stationary=0.04,
    )
    assert details["k_ps"] is None
    assert details["k_ps_upper_bound"] == 2
    assert details["variance_sum_upper"] == 21
    assert details["variance_upper_by_residue"] is None
    assert details["M_over_k_ps"] == 1.0
    t = 20 * details["radius"]
    exponent = (t * t * 0.5) / (8 * 21 + 40 * t)
    assert exponent == pytest.approx(math.log(2 * 3 / 0.04))

    just_below_half = math.nextafter(0.5, 0.0)
    conservative = time_varying_markov_radius_from_gap_bound(
        blocks=20,
        gamma_ps_lower=just_below_half,
        family=1,
        alpha_stationary=0.05,
    )
    assert conservative["k_ps_upper_bound"] == 3


@pytest.mark.parametrize("gamma", [0.0, -0.1, 1.01, float("nan"), True])
def test_gap_only_radius_rejects_invalid_gap_inputs(gamma):
    with pytest.raises(ValueError, match="gamma_ps_lower"):
        time_varying_markov_radius_from_gap_bound(
            blocks=20, gamma_ps_lower=gamma, family=2, alpha_stationary=0.04,
        )


def test_varying_lcb_uses_all_planned_scores_tv_and_anchor_shift():
    result = markov_time_varying_score_lcb(
        [0.8, -0.1, np.nan, 0.5],
        valid_blocks=np.array([True, True, False, True]),
        **lcb_args(),
    )
    radius = time_varying_markov_radius(
        blocks=4,
        gamma_ps_lower=0.5,
        k_ps=2,
        variance_upper_by_residue=[1.0, 1.0],
        family=2,
        alpha_stationary=0.03,
    )["radius"]
    expected_mean = (0.8 - 0.1 - 1.0 + 0.5) / 4
    assert result["status"] == "CONDITIONAL_BOUND_ONLY"
    assert result["physical_score"] is None
    assert result["certified"] is False
    assert result["coded_scores"] == [0.8, -0.1, -1.0, 0.5]
    assert result["planned_mean"] == pytest.approx(expected_mean)
    assert result["radius"] == pytest.approx(radius)
    assert result["familywise_tv_cost"] == pytest.approx(0.01)
    assert result["stationary_alpha"] == pytest.approx(0.03)
    assert result["normalized_lower_bound"] == pytest.approx(
        expected_mean - radius - 0.01 - 0.03
    )
    assert result["denominators"] == {
        "planned": 4, "valid": 3, "failed": 1, "abstained": 0,
    }
    assert result["result_denominators"] == {
        "planned": 1, "valid": 1, "failed": 0, "abstained": 0,
    }


def test_varying_lcb_abstains_when_every_planned_block_fails():
    result = markov_time_varying_score_lcb(
        [np.nan, np.nan],
        valid_blocks=np.array([False, False]),
        **{**lcb_args(), "k_ps": 1, "variance_upper_by_residue": [0.1]},
    )
    assert result["status"] == "ABSTAIN_NO_VALID_BLOCKS"
    assert result["normalized_lower_bound"] is None
    assert result["denominators"] == {
        "planned": 2, "valid": 0, "failed": 2, "abstained": 0,
    }
    assert result["result_denominators"] == {
        "planned": 1, "valid": 0, "failed": 0, "abstained": 1,
    }


def test_gap_only_lcb_keeps_failures_and_never_sets_a_physical_score():
    args = {
        "gamma_ps_lower": 0.5,
        "family": 2,
        "alpha_score": 0.04,
        "alpha_anchor": 0.01,
        "alpha_total": 0.05,
        "initial_law_mode": "stationary",
        "burn_in_blocks": 0,
        "initial_law_tv_bound": 0.0,
        "anchor_score_shift_bound": 0.02,
        "gamma_ps_lower_receipt_sha256": receipt("gap-lower-bound"),
        "dependence_evidence_receipt_sha256": receipt("dependence-evidence"),
        "block_chain_receipt_sha256": receipt("block-chain"),
        "initial_law_receipt_sha256": receipt("stationary"),
        "anchor_shift_receipt_sha256": receipt("anchor-shift"),
        "score_schedule_receipt_sha256": receipt("schedule"),
    }
    result = markov_time_varying_score_lcb_from_gap_bound(
        [0.7, np.nan, -0.2],
        valid_blocks=np.array([True, False, True]),
        **args,
    )
    assert result["dependence_method"] == (
        "markov_time_varying_support_gap_bernstein_eq_3_26"
    )
    assert result["physical_score"] is None
    assert result["certified"] is False
    assert result["coded_scores"] == [0.7, -1.0, -0.2]
    assert result["denominators"] == {
        "planned": 3, "valid": 2, "failed": 1, "abstained": 0,
    }
    assert result["gamma_ps_lower_receipt_sha256"] == args[
        "gamma_ps_lower_receipt_sha256"
    ]
    assert "variance_receipt_sha256" not in result
    assert "k_ps_receipt_sha256" not in result


@pytest.mark.parametrize(
    "field",
    ["gamma_ps_lower_receipt_sha256", "dependence_evidence_receipt_sha256",
     "block_chain_receipt_sha256"],
)
def test_gap_only_lcb_requires_gap_chain_and_dependence_receipts(field):
    args = {
        "gamma_ps_lower": 0.5,
        "family": 1,
        "alpha_score": 0.04,
        "alpha_anchor": 0.01,
        "alpha_total": 0.05,
        "initial_law_mode": "stationary",
        "burn_in_blocks": 0,
        "initial_law_tv_bound": 0.0,
        "anchor_score_shift_bound": 0.0,
        "gamma_ps_lower_receipt_sha256": receipt("gap-lower-bound"),
        "dependence_evidence_receipt_sha256": receipt("dependence-evidence"),
        "block_chain_receipt_sha256": receipt("block-chain"),
        "initial_law_receipt_sha256": receipt("stationary"),
        "anchor_shift_receipt_sha256": receipt("anchor-shift"),
        "score_schedule_receipt_sha256": receipt("schedule"),
    }
    args[field] = "bad"
    with pytest.raises(ValueError, match="SHA-256"):
        markov_time_varying_score_lcb_from_gap_bound(
            [0.5], valid_blocks=np.array([True]), **args,
        )


@pytest.mark.parametrize("kwargs, message", [
    ({"k_ps": 5}, "cannot exceed planned blocks"),
    ({"k_ps": 1, "variance_upper_by_residue": [0.0]}, "strictly positive"),
    ({"variance_upper_by_residue": [1.0, True]}, "strictly positive"),
    ({"k_ps": 3, "variance_upper_by_residue": [1.0, 1.0]}, "length must equal"),
    ({"variance_upper_by_residue": [1.0, float("nan")]}, "strictly positive"),
])
def test_invalid_eq_326_inputs_fail_closed(kwargs, message):
    args = {
        "blocks": 4,
        "gamma_ps_lower": 0.5,
        "k_ps": 2,
        "variance_upper_by_residue": [1.0, 1.0],
        "family": 2,
        "alpha_stationary": 0.04,
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        time_varying_markov_radius(**args)


@pytest.mark.parametrize("kwargs, message", [
    ({"initial_law_mode": []}, "initial_law_mode"),
    ({"initial_law_mode": "stationary", "initial_law_tv_bound": 0.01}, "stationary initial law"),
    ({"initial_law_tv_bound": 0.02}, "smaller than alpha_score"),
    ({"anchor_score_shift_bound": -0.01}, "nonnegative"),
    ({"score_schedule_receipt_sha256": "bad"}, "receipts must be SHA-256"),
    ({"dependence_evidence_receipt_sha256": "bad"}, "receipts must be SHA-256"),
    ({"block_chain_receipt_sha256": "bad"}, "receipts must be SHA-256"),
    ({"gamma_ps_lower_receipt_sha256": "bad"}, "receipts must be SHA-256"),
])
def test_varying_lcb_rejects_unreviewed_contract_values(kwargs, message):
    args = lcb_args()
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        markov_time_varying_score_lcb(
            [0.5, 0.7], valid_blocks=np.array([True, True]), **args,
        )


def test_varying_score_values_must_be_bounded_and_mask_boolean():
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        markov_time_varying_score_lcb(
            [1.01, 0.7], valid_blocks=np.array([True, True]), **lcb_args(),
        )
    with pytest.raises(ValueError, match="boolean"):
        markov_time_varying_score_lcb(
            [0.5, 0.7], valid_blocks=np.array([1, 1]), **lcb_args(),
        )
    with pytest.raises(ValueError, match="non-boolean real"):
        markov_time_varying_score_lcb(
            [True, 0.7], valid_blocks=np.array([True, True]), **lcb_args(),
        )
