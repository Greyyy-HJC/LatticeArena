import hashlib
from fractions import Fraction

import numpy as np

from tasks.pion_projection.bounded_block_risk_nonreversible_tv_coverage_screen import (
    BURN_IN,
    PSEUDO_GAP,
    _burned_cycle_counts,
    _cell,
    _state_score,
    _transition,
    _tv_after_burn_in,
)
from tasks.pion_projection.bounded_block_risk_setwise_markov import (
    setwise_markov_score_lcb_from_gap_bound,
)


def _receipt(label):
    return hashlib.sha256(label.encode()).hexdigest()


def test_directed_cycle_has_exact_post_burn_in_tv_and_reproducible_counts():
    assert _tv_after_burn_in(0) == Fraction(2, 3)
    assert _tv_after_burn_in(BURN_IN) == Fraction(1, 384)
    first = _burned_cycle_counts(blocks=512, replicates=7, seed=113)
    second = _burned_cycle_counts(blocks=512, replicates=7, seed=113)
    assert np.array_equal(first, second)
    assert first.shape == (7, 3)
    assert np.all(first.sum(axis=1) == 512)


def test_directed_cycle_is_nonreversible_with_exact_normal_pseudo_gap_and_truth():
    transition = _transition()
    stationary = np.full(3, 1.0 / 3.0)
    assert np.allclose(stationary @ transition, stationary)
    flow = stationary[:, None] * transition
    assert not np.allclose(flow, flow.T)
    eigenvalues = np.linalg.eigvals(transition)
    nontrivial = sorted(abs(value) for value in eigenvalues if abs(value - 1) > 1e-10)
    assert np.allclose(nontrivial, [0.5, 0.5], atol=1e-12)
    gaps = [(1.0 - 0.25**k) / k for k in range(1, 129)]
    assert max(gaps) == PSEUDO_GAP
    for target in np.linspace(0.45, 0.55, 7):
        exact = sum(_state_score(float(target), state) for state in range(3)) / 3
        assert np.isclose(exact, 1.0 / 6.0, rtol=0.0, atol=1e-15)


def test_nonreversible_tv_screen_uses_tv_adjusted_production_helper():
    rows, summary = _cell(blocks=16_384, replicates=3, seed=1711)
    assert summary["pseudo_gap"] == PSEUDO_GAP
    assert summary["initial_law_tv_exact"] == "1/384"
    assert summary["failed_replicates"] == 0
    for row in rows:
        # The helper consumes the same planned scores. State ordering does not
        # alter the empirical average, and the TV correction is applied by the
        # production route rather than copied into this test.
        refs = np.repeat([0.0, 0.5, 1.0], [
            row["state0_count"], row["state1_count"], row["state2_count"],
        ])
        helper = setwise_markov_score_lcb_from_gap_bound(
            refs,
            np.full(refs.size, 0.5),
            valid_blocks=np.ones(refs.size, dtype=bool),
            target_interval=[0.45, 0.55],
            prediction_domain=[0.0, 1.0],
            score_scale_R=1.0,
            mesh_step_R=0.10,
            gamma_ps_lower=PSEUDO_GAP,
            family=1,
            alpha_score=0.04,
            alpha_anchor=0.01,
            alpha_total=0.05,
            initial_law_mode="post_burn_in_tv",
            burn_in_blocks=BURN_IN,
            initial_law_tv_bound=row["initial_law_tv_upper"],
            anchor_score_shift_bound=0.0,
            gamma_ps_lower_receipt_sha256=_receipt("cycle-gap"),
            dependence_evidence_receipt_sha256=_receipt("cycle-dependence"),
            block_chain_receipt_sha256=_receipt("cycle-chain"),
            initial_law_receipt_sha256=_receipt("cycle-initial-law"),
            anchor_shift_receipt_sha256=_receipt("cycle-anchor-shift"),
            score_schedule_receipt_sha256=_receipt("cycle-schedule"),
            target_set_coverage_receipt_sha256=_receipt("fixed-target-set"),
            prediction_domain_receipt_sha256=_receipt("prediction-domain"),
        )
        assert np.isclose(
            row["uniform_lcb"], helper["uniform_target_set_lower_bound"],
            rtol=0.0, atol=2e-14,
        )
        assert helper["simultaneous_family_size"] == 3
        assert row["initial_law_tv_upper"] > float(Fraction(1, 384))

