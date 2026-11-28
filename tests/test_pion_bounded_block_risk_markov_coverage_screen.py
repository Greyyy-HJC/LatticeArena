import hashlib

import numpy as np

from tasks.pion_projection.bounded_block_risk_markov_coverage_screen import (
    _cell,
    _chain_zero_counts,
)
from tasks.pion_projection.bounded_block_risk_setwise_markov import (
    setwise_markov_score_lcb_from_gap_bound,
)


def _receipt(label):
    return hashlib.sha256(label.encode()).hexdigest()


def test_two_state_chain_counts_are_reproducible_and_bounded():
    first = _chain_zero_counts(gamma=0.5, blocks=64, replicates=8, seed=9)
    second = _chain_zero_counts(gamma=0.5, blocks=64, replicates=8, seed=9)
    assert np.array_equal(first, second)
    assert first.shape == (8,)
    assert np.all((first >= 0) & (first <= 64))


def test_vectorized_screen_lcb_matches_production_setwise_helper():
    rows, summary = _cell(
        0.5, blocks=4096, replicates=3, seed=20261128,
    )
    assert summary["planned_replicates"] == 3
    assert summary["failed_replicates"] == 0
    for row in rows:
        count = row["state0_count"]
        reference = np.asarray([0.0] * count + [1.0] * (4096 - count))
        candidate = np.full(4096, 0.5)
        helper = setwise_markov_score_lcb_from_gap_bound(
            reference,
            candidate,
            valid_blocks=np.ones(4096, dtype=bool),
            target_interval=[0.45, 0.55],
            prediction_domain=[0.0, 1.0],
            score_scale_R=1.0,
            mesh_step_R=0.10,
            gamma_ps_lower=0.5,
            family=1,
            alpha_score=0.04,
            alpha_anchor=0.01,
            alpha_total=0.05,
            initial_law_mode="stationary",
            burn_in_blocks=0,
            initial_law_tv_bound=0.0,
            anchor_score_shift_bound=0.0,
            gamma_ps_lower_receipt_sha256=_receipt("gamma"),
            dependence_evidence_receipt_sha256=_receipt("dependence"),
            block_chain_receipt_sha256=_receipt("chain"),
            initial_law_receipt_sha256=_receipt("initial"),
            anchor_shift_receipt_sha256=_receipt("anchor"),
            score_schedule_receipt_sha256=_receipt("schedule"),
            target_set_coverage_receipt_sha256=_receipt("target"),
            prediction_domain_receipt_sha256=_receipt("domain"),
        )
        assert np.isclose(
            row["uniform_lcb"], helper["uniform_target_set_lower_bound"],
            rtol=0.0, atol=1e-14,
        )
        assert row["covered"] == (row["uniform_lcb"] <= 0.25)
