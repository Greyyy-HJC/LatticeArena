import hashlib

import numpy as np

from tasks.pion_projection.bounded_block_risk_independent_anchor_coverage_screen import (
    ALPHA_ANCHOR,
    ALPHA_SCORE,
    ALPHA_TOTAL,
    TARGET_PROBABILITY,
    _cell,
    _cp_interval,
    _draw_reversible_chain,
)
from tasks.pion_projection.bounded_block_risk_setwise_markov import (
    setwise_markov_score_lcb_from_gap_bound,
)


def _receipt(label):
    return hashlib.sha256(label.encode()).hexdigest()


def test_clopper_pearson_interval_handles_interior_and_boundary_counts():
    low, high = _cp_interval(102, 256, ALPHA_ANCHOR)
    assert 0.0 < low < 102 / 256 < high < 1.0
    assert _cp_interval(0, 256, ALPHA_ANCHOR)[0] == 0.0
    assert _cp_interval(256, 256, ALPHA_ANCHOR)[1] == 1.0


def test_reversible_score_chain_is_seed_reproducible_and_binary():
    first = _draw_reversible_chain(
        gamma_ps=0.75, blocks=128, replicates=7, seed=203,
    )
    second = _draw_reversible_chain(
        gamma_ps=0.75, blocks=128, replicates=7, seed=203,
    )
    assert np.array_equal(first, second)
    assert first.shape == (7, 128)
    assert first.dtype == np.bool_


def test_random_interval_vectorized_lcb_matches_production_helper():
    raw, rows, summary = _cell(
        0.5, replicates=3, blocks=512,
        anchor_seed=202611282333, score_seed=202611287333,
    )
    assert summary["planned_replicates"] == 3
    assert summary["valid_anchor_observations"] == 3 * 256
    assert summary["valid_score_blocks"] == 3 * 512
    assert summary["failed_replicates"] == 0
    assert np.all(raw["intervals"][:, 0] <= raw["intervals"][:, 1])

    for index, row in enumerate(rows):
        states = raw["score_chain_states"][index].astype(float)
        helper = setwise_markov_score_lcb_from_gap_bound(
            states,
            np.full(states.size, 0.5),
            valid_blocks=np.ones(states.size, dtype=bool),
            target_interval=raw["intervals"][index].tolist(),
            prediction_domain=[0.0, 1.0],
            score_scale_R=1.0,
            mesh_step_R=0.1,
            gamma_ps_lower=0.5,
            family=1,
            alpha_score=ALPHA_SCORE,
            alpha_anchor=ALPHA_ANCHOR,
            alpha_total=ALPHA_TOTAL,
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
            rtol=0.0, atol=2e-14,
        )
        assert row["score_uniformly_covered"] == (
            row["uniform_lcb"] <= row["exact_interval_minimum_score"]
        )
        assert row["jointly_covered"] == (
            row["anchor_covered"] and row["score_uniformly_covered"]
        )


def test_exact_mean_matches_statewise_score_formula():
    for target in (0.0, 0.2, 0.4, 0.8, 1.0):
        score0 = (target**2 - (target - 0.5) ** 2)
        score1 = ((target - 1.0) ** 2 - (target - 0.5) ** 2)
        direct = (1.0 - TARGET_PROBABILITY) * score0 + TARGET_PROBABILITY * score1
        formula = (1.0 - 2.0 * TARGET_PROBABILITY) * target - 0.25 + TARGET_PROBABILITY
        assert np.isclose(direct, formula, rtol=0.0, atol=1e-15)
