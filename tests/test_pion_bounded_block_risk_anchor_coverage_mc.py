import hashlib
import json
from statistics import NormalDist

import numpy as np
import pytest

from tasks.pion_projection.bounded_block_risk_anchor_coverage_mc import (
    anchor_bound,
    build,
    simulate_cell,
)
from tasks.pion_projection.bounded_block_risk_score import bounded_block_scores


def test_anchor_bound_is_two_sided_normal_and_shrinks_with_sample_size():
    z = NormalDist().inv_cdf(0.995)
    assert anchor_bound(samples=32) == pytest.approx(z * 0.05 / np.sqrt(32))
    assert anchor_bound(samples=128) == pytest.approx(z * 0.05 / np.sqrt(128))
    assert anchor_bound(samples=32) == pytest.approx(2 * anchor_bound(samples=128))


def test_squared_error_anchor_shift_is_covered_by_declared_lipschitz_term():
    radius = 1.0
    q = np.array([-0.6, 0.2, 0.8, 1.0])
    target_anchor = 0.0
    anchor_estimate = 0.01
    anchor_bound_value = 0.02
    y_reference = -radius * (q + 1.0) / 2.0
    y_candidate = y_reference + radius

    d_true = ((target_anchor - y_reference) ** 2
              - (target_anchor - y_candidate) ** 2)
    d_observed = ((anchor_estimate - y_reference) ** 2
                  - (anchor_estimate - y_candidate) ** 2)
    z_true = bounded_block_scores(
        d_true, valid_blocks=np.ones(q.size, dtype=bool), score_scale_R=radius,
    )
    z_observed = bounded_block_scores(
        d_observed, valid_blocks=np.ones(q.size, dtype=bool), score_scale_R=radius,
    )
    assert np.max(np.abs(z_observed - z_true)) <= (
        2 * anchor_bound_value / radius + 1e-12
    )


def test_small_cell_is_reproducible_and_keeps_all_denominators():
    kwargs = dict(
        blocks=32,
        dependence_order=7,
        failure_probability=0.1,
        anchor_samples=32,
        replicates=120,
        chunk_size=40,
    )
    row_a, arrays_a = simulate_cell(**kwargs)
    row_b, arrays_b = simulate_cell(**kwargs)
    assert row_a == row_b
    assert row_a["planned_replicates"] == 120
    assert row_a["planned_family_block_scores"] == 120 * 32 * 2
    assert (row_a["valid_family_block_scores"]
            + row_a["failed_family_block_scores"]
            == row_a["planned_family_block_scores"])
    assert row_a["alpha_score"] + row_a["alpha_anchor"] == pytest.approx(
        row_a["alpha_total"]
    )
    assert (row_a["graph_false_certificate_rate_planned"]
            == np.mean(arrays_a["false_certificate_graph"]))
    for name in arrays_a:
        np.testing.assert_array_equal(arrays_a[name], arrays_b[name])


def test_small_build_writes_report_plot_denominators_and_hashes(tmp_path):
    output = tmp_path / "anchor-coverage"
    summary = build(output, replicates=2)
    assert summary["cells"] == 16
    assert summary["planned_replicates"] == 32
    assert summary["physical_score"] is None
    assert "No physical" in summary["data_access"]
    assert (output / "coverage_and_utility.png").is_file()
    denominators = (output / "denominator_summary.csv").read_text()
    assert "MC replicate" in denominators
    assert "paired block event" in denominators
    assert "family block score" in denominators
    assert "anchor observation" in denominators
    manifest = json.loads((output / "artifact_hashes.json").read_text())
    assert len(manifest) == 12
    for name, digest in manifest.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    readme = (output / "README.md").read_text()
    assert "SYNTHETIC METHOD DIAGNOSTIC ONLY" in readme
    assert "alpha_score=0.04" in readme
    assert "alpha_anchor=0.01" in readme
