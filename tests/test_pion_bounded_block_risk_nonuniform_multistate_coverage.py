from __future__ import annotations

import numpy as np

from tasks.pion_projection import bounded_block_risk_nonuniform_multistate_coverage_screen as screen


def test_four_state_refresh_model_is_stationary_reversible_with_declared_spectrum():
    diagnostics = screen._model_diagnostics()
    transition = np.asarray(diagnostics["transition_matrix"], dtype=float)
    stationary = np.asarray(diagnostics["stationary_law"], dtype=float)

    assert np.allclose(transition.sum(axis=1), 1.0, rtol=0.0, atol=2e-15)
    assert np.allclose(stationary @ transition, stationary, rtol=0.0, atol=2e-15)
    assert np.allclose(
        stationary[:, None] * transition,
        (stationary[:, None] * transition).T,
        rtol=0.0,
        atol=2e-15,
    )
    assert np.allclose(
        diagnostics["eigenvalues_sorted"],
        [float(screen.REFRESH_EIGENVALUE)] * 3 + [1.0],
        rtol=0.0,
        atol=2e-14,
    )
    assert screen.PSEUDO_GAP == 1 - screen.REFRESH_EIGENVALUE**2


def test_nonstationary_start_tv_after_burnin_matches_exact_refresh_identity():
    transition = screen._transition_matrix()
    stationary = np.asarray([float(value) for value in screen.STATIONARY_LAW])
    post_burnin = np.linalg.matrix_power(transition, screen.BURN_IN)[0]
    measured_tv = 0.5 * np.abs(post_burnin - stationary).sum()

    assert np.isclose(measured_tv, float(screen.TV_EXACT), rtol=2e-12, atol=1e-16)
    assert screen.TV_UPPER >= float(screen.TV_EXACT)
    assert screen.TV_UPPER <= np.nextafter(float(screen.TV_EXACT), np.inf)


def test_small_joint_coverage_cell_is_reproducible_and_matches_production_helper():
    first_raw, first_rows, first_summary = screen._cell(
        replicates=4, blocks=256, anchor_seed=901, score_seed=902,
    )
    second_raw, second_rows, second_summary = screen._cell(
        replicates=4, blocks=256, anchor_seed=901, score_seed=902,
    )

    assert np.array_equal(first_raw["anchor_samples"], second_raw["anchor_samples"])
    assert np.array_equal(
        first_raw["score_chain_scored_states"],
        second_raw["score_chain_scored_states"],
    )
    assert np.array_equal(first_raw["uniform_lcbs"], second_raw["uniform_lcbs"])
    assert first_rows == second_rows
    assert first_summary["planned_score_blocks"] == 4 * 256
    assert first_summary["valid_score_blocks"] == 4 * 256
    assert first_summary["helper_equivalence"]["all_match"]
    assert first_summary["helper_equivalence"]["max_abs_difference"] <= 2e-14
