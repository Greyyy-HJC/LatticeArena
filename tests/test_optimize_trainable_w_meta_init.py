import hashlib
import json

import numpy as np

from tasks.optimize_gaussian import trainable_w as model
from tasks.optimize_gaussian.trainable_w_meta_init_diagnostic import (
    _random_initial,
    build,
    fit_pooled_prior,
    make_local_case,
    minimum_break_even_tasks,
)


def test_pooled_prior_reports_cost_for_the_actual_panel_count():
    fit = [make_local_case(np.random.default_rng(83001),
                           np.array([0.045, 0.025, -0.018, 0.012, 0.008, -0.006]))]
    initial = _random_initial(np.random.default_rng(83002))
    prior = fit_pooled_prior(fit, initial, budget=1, panel_count=1)
    assert prior["calls"] <= 1
    assert prior["panel_equivalent_calls"] == prior["calls"]
    assert prior["total_panel_equivalent_evaluations"] == prior["calls"] + 1
    assert prior["total_objective_evaluations"] == prior["calls"] + 1
    assert prior["link_case_evaluations"] == prior["calls"]
    assert model.unpack(prior["parameters"], hidden=2)[0].shape == (2, 2)


def test_break_even_count_uses_strict_integer_cost_comparison():
    assert minimum_break_even_tasks(192, [7, 7, 7], [4, 4, 4]) == 65
    assert minimum_break_even_tasks(192, [4, 4], [4, 4]) is None


def test_small_meta_initialization_report_is_hash_closed_and_unsealed(tmp_path):
    output = tmp_path / "meta-init"
    summary = build(output, training_panels=1, evaluation_panels=2,
                    online_budgets=(32,), training_budget=1, seed=83003)
    assert summary["planned_training_panels"] == 1
    assert summary["planned_evaluation_panels"] == 2
    assert summary["planned_solver_runs"] == summary["returned_solver_runs"] == 6
    assert summary["physical_score"] is None
    assert summary["pooled_training_panel_equivalent_total_evaluations"] == (
        summary["pooled_training_calls"] + 1
    )
    hashes = json.loads((output / "artifact_hashes.json").read_text())
    for name, digest in hashes.items():
        if name == "artifact_hashes.json":
            continue
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    report = (output / "README.md").read_text()
    assert "No physical links, sealed split, GPU, or QUDA data are read" in report
    assert "degree-4 Chebyshev" in report
    assert "NOT_READY/UNCERTIFIED" in report
