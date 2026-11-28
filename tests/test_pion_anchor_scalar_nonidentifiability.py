from __future__ import annotations

import math

import pytest

from tasks.pion_projection.anchor_scalar_nonidentifiability import (
    EXCITED_MASS,
    GROUND_MASSES,
    MAP_VALUE,
    R_SCALE,
    REQUIRED_C_OVER_R,
    _correlator,
    _models,
    solve_excited_amplitude_ratio,
)
from tasks.pion_projection.anchor_scalar_reader import anchor_scalar


def test_two_distinct_positive_spectra_have_same_scalar_map_value():
    rows = _models()
    assert [row["ground_mass"] for row in rows] == list(GROUND_MASSES)
    assert all(row["A1_over_A0"] > 0.0 for row in rows)
    assert [row["m_eff4"] for row in rows] == pytest.approx([MAP_VALUE, MAP_VALUE], abs=1e-12)
    assert rows[1]["ground_mass"] - rows[0]["ground_mass"] == pytest.approx(0.30)


def test_same_scalar_input_forces_minimax_bias_above_selected_common_bias_gate():
    separation = GROUND_MASSES[1] - GROUND_MASSES[0]
    minimax_error = separation / 2.0
    assert minimax_error == pytest.approx(0.15)
    assert minimax_error / R_SCALE > REQUIRED_C_OVER_R
    assert REQUIRED_C_OVER_R * R_SCALE == pytest.approx(0.029)


def test_amplitude_solution_rejects_non_bracketing_map_mass():
    with pytest.raises(ValueError, match="ground < map < excited"):
        solve_excited_amplitude_ratio(0.6, 0.5, EXCITED_MASS)


def test_constructed_spectrum_is_valid_under_the_strict_reader():
    for row in _models():
        correlator = _correlator(row["ground_mass"], row["excited_mass"], row["A1_over_A0"])
        assert math.isclose(anchor_scalar(correlator), MAP_VALUE, rel_tol=0.0, abs_tol=1e-12)
