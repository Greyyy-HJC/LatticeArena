import hashlib
import json

import numpy as np
import pytest

from tasks.pion_projection.bounded_block_risk_coverage_mc import (
    build,
    moving_window_scores,
    simulate_cell,
)


def test_moving_window_scores_are_bounded_and_reproducible():
    first = moving_window_scores(
        np.random.default_rng(62001), rows=12, blocks=32,
        dependence_order=7,
    )
    second = moving_window_scores(
        np.random.default_rng(62001), rows=12, blocks=32,
        dependence_order=7,
    )
    np.testing.assert_array_equal(first, second)
    assert first.shape == (12, 32)
    assert np.all(first >= -1.0)
    assert np.all(first <= 1.0)


def test_invalid_dependence_range_rejects():
    with pytest.raises(ValueError, match="dependence_order"):
        moving_window_scores(
            np.random.default_rng(62002), rows=2, blocks=8,
            dependence_order=8,
        )
    with pytest.raises(ValueError, match="keep scores"):
        moving_window_scores(
            np.random.default_rng(62003), rows=2, blocks=8,
            dependence_order=1, mean=0.5, amplitude=0.6,
        )


def test_correct_graph_penalty_uses_more_colors_than_iid_misspecification():
    row, arrays = simulate_cell(
        blocks=32, dependence_order=7, failure_probability=0.1,
        replicates=200,
    )
    assert row["graph_chromatic_number"] == 8
    assert row["graph_epsilon"] > row["iid_epsilon"]
    assert len(arrays["familywise_cover_graph"]) == 200
    assert len(arrays["familywise_cover_iid"]) == 200
    assert np.all(arrays["min_graph_lcb"] <= arrays["min_iid_lcb"])


def test_failed_pairs_remain_in_the_planned_block_denominator():
    row, arrays = simulate_cell(
        blocks=32, dependence_order=3, failure_probability=0.1,
        replicates=100,
    )
    assert row["planned_pair_block_events"] == 3200
    assert (row["valid_pair_block_events"]
            + row["failed_pair_block_events"] == 3200)
    assert row["planned_family_block_scores"] == 6400
    assert (row["valid_family_block_scores"]
            + row["failed_family_block_scores"] == 6400)
    assert row["empirical_valid_score_mean"] == pytest.approx(0.2, abs=0.05)
    assert row["empirical_coded_mean_error"] == pytest.approx(0.0, abs=0.05)
    assert row["empirical_failure_fraction"] == pytest.approx(0.1, abs=0.04)
    assert int(arrays["failed_count"].sum()) == row["failed_pair_block_events"]
    assert row["abstained_all_blocks_failed"] == int(arrays["all_failed"].sum())


def test_small_build_writes_manifest_and_report(tmp_path):
    output = tmp_path / "mc-report"
    summary = build(output, replicates=2)
    assert summary["cells"] == 30
    assert summary["replicates_per_cell"] == 2
    manifest = json.loads((output / "artifact_hashes.json").read_text())
    assert len(manifest) == 10
    for name, digest in manifest.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert "SYNTHETIC METHOD DIAGNOSTIC ONLY" in (output / "README.md").read_text()
    assert (output / "coverage_by_dependence.png").is_file()
    assert (output / "positive_certificate_rate.png").is_file()
