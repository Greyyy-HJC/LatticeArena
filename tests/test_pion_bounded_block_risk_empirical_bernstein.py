import hashlib
import json

import numpy as np
import pytest

from tasks.pion_projection.bounded_block_risk_empirical_bernstein import (
    color_class_empirical_bernstein_lcb,
)
from tasks.pion_projection.bounded_block_risk_empirical_bernstein_mc import (
    build,
    simulate_cell,
)


def test_low_variance_color_class_can_tighten_the_graph_hoeffding_radius():
    scores = np.full((2, 128), 0.25)
    labels = np.zeros(128, dtype=int)
    result = color_class_empirical_bernstein_lcb(
        scores, color_labels=labels, alpha_score=0.05,
    )
    assert result["physical_score"] is None
    assert result["color_class_methods"] == ["empirical_bernstein"]
    assert np.all(result["color_class_sample_variances"] == 0.0)
    assert np.all(result["simultaneous_lower_bounds"] > 0.0)
    hoeffding_radius = np.sqrt(2.0 * np.log(2 * 2 / 0.05) / 128)
    assert result["color_class_radii"][0, 0] < hoeffding_radius


def test_singleton_color_classes_use_hoeffding_fallback_and_keep_all_blocks():
    scores = np.full((1, 4), 0.2)
    result = color_class_empirical_bernstein_lcb(
        scores, color_labels=np.arange(4), alpha_score=0.05,
    )
    assert result["planned_blocks"] == 4
    assert result["color_class_count"] == 4
    assert result["color_class_sizes"] == [1, 1, 1, 1]
    assert result["color_class_methods"] == ["hoeffding_singleton"] * 4
    np.testing.assert_allclose(result["simultaneous_lower_bounds"], [-1.0])


@pytest.mark.parametrize(
    "scores,labels,alpha",
    [
        (np.array([[1.01, 0.0]]), np.array([0, 0]), 0.05),
        (np.array([[0.0, 0.1]]), np.array([0]), 0.05),
        (np.array([[0.0, 0.1]]), np.array([0, 0]), 1.0),
    ],
)
def test_empirical_bernstein_rejects_invalid_contract(scores, labels, alpha):
    with pytest.raises(ValueError):
        color_class_empirical_bernstein_lcb(
            scores, color_labels=labels, alpha_score=alpha,
        )


def test_known_graph_simulation_is_paired_and_reproducible():
    first, first_arrays = simulate_cell(
        blocks=32, dependence_order=1, failure_probability=0.1,
        replicates=256, chunk_size=64,
    )
    second, second_arrays = simulate_cell(
        blocks=32, dependence_order=1, failure_probability=0.1,
        replicates=256, chunk_size=64,
    )
    assert first == second
    assert first["planned_replicates"] == first["valid_replicates"] == 256
    assert first["physical_score"] is None
    for key in first_arrays:
        np.testing.assert_array_equal(first_arrays[key], second_arrays[key])


def test_small_empirical_bernstein_report_is_hash_closed(tmp_path):
    output = tmp_path / "empirical-bernstein"
    summary = build(
        output, replicates=128, block_counts=(32,),
        dependence_orders=(0, 1), failure_probabilities=(0.0,),
    )
    assert summary["planned_cells"] == summary["valid_cells"] == 2
    assert summary["planned_replicates"] == summary["valid_replicates"] == 256
    assert summary["physical_score"] is None
    manifest = json.loads((output / "artifact_hashes.json").read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    report = (output / "README.md").read_text()
    assert "within each color class" in report
    assert "No Pion correlators" in report
