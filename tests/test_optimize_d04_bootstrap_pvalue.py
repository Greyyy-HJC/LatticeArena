from __future__ import annotations

import pytest

from tasks.optimize_gaussian.d04_parametric_bootstrap_diagnostic import (
    _bootstrap_upper_tail_pvalue,
)


def test_bootstrap_pvalue_matches_standard_rank_rule_when_no_fit_fails():
    assert _bootstrap_upper_tail_pvalue(
        exceedances=7, valid_draws=199, failed_draws=0,
        planned_draws=199, minimum_valid_draws=180,
    ) == pytest.approx(8 / 200)


def test_failed_refits_are_conservative_upper_tail_exceedances():
    assert _bootstrap_upper_tail_pvalue(
        exceedances=5, valid_draws=197, failed_draws=2,
        planned_draws=199, minimum_valid_draws=180,
    ) == pytest.approx(8 / 200)


def test_bootstrap_pvalue_abstains_below_minimum_valid_draw_count():
    assert _bootstrap_upper_tail_pvalue(
        exceedances=5, valid_draws=179, failed_draws=20,
        planned_draws=199, minimum_valid_draws=180,
    ) is None


def test_bootstrap_pvalue_rejects_unclosed_denominators():
    with pytest.raises(ValueError, match="counts do not close"):
        _bootstrap_upper_tail_pvalue(
            exceedances=5, valid_draws=197, failed_draws=1,
            planned_draws=199, minimum_valid_draws=180,
        )
