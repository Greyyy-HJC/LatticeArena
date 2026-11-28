"""Pure-analytic checks for bounded-score dependence sensitivity."""
import math

import pytest

from tasks.pion_projection.dependency_graph_bound import (
    dependence_penalty,
    normalized_lower_bound,
    required_valid_mean,
)


def test_one_color_reproduces_independent_block_penalty():
    actual = dependence_penalty(
        blocks=20, chromatic_number=1, family=2, alpha_s=0.05,
    )
    expected = math.sqrt(2 * math.log(2 / 0.05) / 20)
    assert actual == pytest.approx(expected)


def test_penalty_grows_with_color_count_and_full_dependence_has_no_n_gain():
    values = [dependence_penalty(blocks=20, chromatic_number=chi)
              for chi in (1, 2, 5, 20)]
    assert values == sorted(values)
    assert values[-1] == pytest.approx(math.sqrt(2 * math.log(40)))


def test_failure_and_anchor_penalties_match_declared_formula():
    result = normalized_lower_bound(
        0.8, blocks=20, chromatic_number=2, family=2, alpha_s=0.05,
        alpha_anchor=0.0, alpha_total=0.05,
        relative_anchor_bound=0.03, failure_fraction=0.1,
    )
    expected = (0.9 * 0.8 - 0.1
                - math.sqrt(2 * 2 * math.log(40) / 20) - 0.06)
    assert result == pytest.approx(expected)


def test_required_mean_inverts_lower_bound_at_threshold():
    args = dict(
        blocks=100, chromatic_number=3, family=2, alpha_s=0.05,
        alpha_anchor=0.0, alpha_total=0.05,
        relative_anchor_bound=0.02, normalized_threshold=0.1,
        failure_fraction=0.05,
    )
    mean = required_valid_mean(**args)
    lower = normalized_lower_bound(
        mean, blocks=args["blocks"],
        chromatic_number=args["chromatic_number"], family=args["family"],
        alpha_s=args["alpha_s"],
        alpha_anchor=args["alpha_anchor"], alpha_total=args["alpha_total"],
        relative_anchor_bound=args["relative_anchor_bound"],
        failure_fraction=args["failure_fraction"],
    )
    assert lower == pytest.approx(args["normalized_threshold"])
    assert normalized_lower_bound(
        mean + 1e-5, blocks=args["blocks"],
        chromatic_number=args["chromatic_number"], family=args["family"],
        alpha_s=args["alpha_s"],
        alpha_anchor=args["alpha_anchor"], alpha_total=args["alpha_total"],
        relative_anchor_bound=args["relative_anchor_bound"],
        failure_fraction=args["failure_fraction"],
    ) > args["normalized_threshold"]


def test_anchor_and_score_error_budgets_must_fit_total_budget():
    with pytest.raises(ValueError, match=r"alpha_s \+ alpha_anchor"):
        normalized_lower_bound(
            0.5, blocks=20, chromatic_number=1, family=2, alpha_s=0.04,
            alpha_anchor=0.02, alpha_total=0.05,
        )
    with pytest.raises(ValueError, match=r"alpha_s \+ alpha_anchor"):
        required_valid_mean(
            blocks=20, chromatic_number=1, family=2, alpha_s=0.04,
            alpha_anchor=0.02, alpha_total=0.05,
        )


def test_reserved_anchor_budget_uses_score_budget_for_concentration():
    actual = normalized_lower_bound(
        0.8, blocks=20, chromatic_number=2, family=2, alpha_s=0.04,
        alpha_anchor=0.01, alpha_total=0.05,
    )
    expected = 0.8 - math.sqrt(2 * 2 * math.log(2 / 0.04) / 20)
    assert actual == pytest.approx(expected)


def test_anchor_squared_error_contrast_has_declared_lipschitz_bound():
    target, anchor, reference, candidate = 0.6, 0.58, 0.1, 0.8
    B, R = abs(anchor - target), 1.0
    d = (anchor - reference) ** 2 - (anchor - candidate) ** 2
    d_true = (target - reference) ** 2 - (target - candidate) ** 2
    assert abs(d - d_true) <= 2 * B * abs(candidate - reference)
    assert abs(d - d_true) / R**2 <= 2 * B / R


@pytest.mark.parametrize("kwargs", [
    {"blocks": 0, "chromatic_number": 1},
    {"blocks": 4, "chromatic_number": 0},
    {"blocks": 4, "chromatic_number": 5},
    {"blocks": True, "chromatic_number": 1},
    {"blocks": 4, "chromatic_number": 1, "family": 0},
    {"blocks": 4, "chromatic_number": 1, "alpha_s": 1.0},
])
def test_invalid_graph_or_confidence_inputs_reject(kwargs):
    args = dict(blocks=4, chromatic_number=1, family=2, alpha_s=0.05)
    args.update(kwargs)
    with pytest.raises(ValueError):
        dependence_penalty(**args)


@pytest.mark.parametrize("kwargs", [
    {"valid_mean": 1.1},
    {"valid_mean": float("nan")},
    {"valid_mean": 0.2, "failure_fraction": 1.0},
    {"valid_mean": 0.2, "relative_anchor_bound": -0.1},
])
def test_invalid_score_sensitivity_inputs_reject(kwargs):
    args = dict(valid_mean=0.2, blocks=4, chromatic_number=1,
                family=2, alpha_s=0.05, alpha_anchor=0.0,
                alpha_total=0.05)
    args.update(kwargs)
    with pytest.raises(ValueError):
        normalized_lower_bound(**args)
