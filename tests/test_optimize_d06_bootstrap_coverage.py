import json

import numpy as np
import pytest

from tasks.optimize_gaussian.d06_bootstrap_coverage_diagnostic import (
    _batch_bootstrap_risk,
    exact_ar1_sample_mean_variance,
    generate_paired_ar1_panel,
    build_report,
)


def test_analytic_inner_block_bootstrap_moments_match_direct_block_formula():
    samples = np.asarray([
        [[1.0, 0.8, 1.2, 1.0]],
        [[1.2, 1.0, 0.8, 1.1]],
        [[0.9, 1.1, 1.0, 0.9]],
        [[1.1, 0.9, 1.1, 1.2]],
    ])[None, ...]

    result = _batch_bootstrap_risk(samples, block_length=2)[0, 0]
    block_means = np.stack([
        0.5 * (samples[0, i, 0] + samples[0, (i + 1) % 4, 0])
        for i in range(4)
    ])
    bootstrap_variance = np.var(block_means, axis=0, ddof=0) / 2
    mean_q = samples[0, :, 0].mean(axis=0)
    expected = np.mean(bootstrap_variance + (mean_q - 1.0) ** 2)

    assert result == pytest.approx(expected)


def test_exact_ar1_mean_variance_matches_two_point_closed_form():
    assert exact_ar1_sample_mean_variance(2, sigma=1.0, phi=0.5) == pytest.approx(0.75)


def test_paired_generator_is_seed_reproducible_and_finite():
    left = generate_paired_ar1_panel(1234, phi=0.4)
    right = generate_paired_ar1_panel(1234, phi=0.4)

    assert left.shape == (40, 3, 4)
    assert np.array_equal(left, right)
    assert np.isfinite(left).all()


def test_small_coverage_report_keeps_panel_denominators_and_source_hashes(tmp_path):
    output = tmp_path / "coverage"

    result = build_report(
        output, mc_replicates=1, outer_draws=40, base_seed=10,
        n_scenarios=(40,), phi_scenarios=(0.0,),
    )
    protocol = json.loads((output / "protocol.json").read_text())
    summary = json.loads((output / "coverage_summary.json").read_text())

    assert result["physical_score"] is None
    assert len(summary) == 3
    assert all(row["planned"] == row["valid"] == 1 for row in summary)
    assert all(row["failed"] == row["abstained"] == 0 for row in summary)
    assert protocol["outer_draws_per_block_length"] == 40
    assert protocol["selected_contract_draws_per_block_length"] == 10_000
    assert protocol["csv_line_terminator"] == "LF"
    assert set(protocol["source_hashes"]) == {
        "d06_bootstrap_coverage_diagnostic.py", "risk_statistics.py",
        "research_contract_v1.json",
    }
    assert (output / "replicates.csv").is_file()
    assert b"\r" not in (output / "replicates.csv").read_bytes()
    assert (output / "README.md").is_file()
    assert (output / "artifact_hashes.json").is_file()
