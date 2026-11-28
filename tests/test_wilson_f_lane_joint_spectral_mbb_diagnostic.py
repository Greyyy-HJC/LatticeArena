import hashlib
import json

import numpy as np
import pytest

from tasks.wilson_loop import f_lane_joint_spectral_gls_diagnostic as joint
from tasks.wilson_loop import f_lane_joint_spectral_mbb_diagnostic as mbb


def test_circular_moving_block_indices_preserve_within_chunk_adjacency():
    rng = np.random.default_rng(20261126)
    first = mbb._circular_moving_block_indices(rng, n=20, block_length=5)
    second = mbb._circular_moving_block_indices(
        np.random.default_rng(20261126), n=20, block_length=5,
    )
    assert np.array_equal(first, second)
    assert first.shape == (20,)
    assert np.all((first >= 0) & (first < 20))
    for start in range(0, 20, 5):
        chunk = first[start:start + 5]
        assert np.array_equal((chunk[0] + np.arange(5)) % 20, chunk)


def test_bootstrap_upper_uses_finite_sample_rank_and_abstains():
    upper, rank = mbb._upper_order_statistic(np.arange(40.0))
    assert rank == 40
    assert upper == 39.0
    assert mbb._upper_order_statistic(np.arange(20.0)) == (None, 21)
    values = np.concatenate([np.arange(39.0), [np.inf]])
    assert mbb._upper_order_statistic(values) == (None, 40)
    with pytest.raises(ValueError, match="invalid"):
        mbb._upper_order_statistic([1.0, np.nan])


def test_small_mbb_report_binds_inputs_and_reconciles_all_denominators(tmp_path):
    output = tmp_path / "mbb-report"
    result = mbb.run(
        output, outer_panels_per_scenario=1, bootstrap_replicates=40,
        block_length=8, scenarios=("matched_3state_independent",),
    )
    denominators = result["denominators"]
    assert denominators["outer_holdout_panels_planned"] == 1
    assert denominators["bootstrap_refits_planned"] == 40
    assert (denominators["bootstrap_refits_valid"]
            + denominators["bootstrap_refits_failed"]
            + denominators["bootstrap_refits_abstained"] == 40)
    assert denominators["bootstrap_q_estimates_planned"] == 160
    assert (denominators["bootstrap_q_estimates_valid"]
            + denominators["bootstrap_q_estimates_failed"]
            + denominators["bootstrap_q_estimates_abstained"] == 160)
    assert denominators["moving_block_intervals_planned"] == 4
    assert (denominators["moving_block_intervals_valid"]
            + denominators["moving_block_intervals_failed"]
            + denominators["moving_block_intervals_abstained"] == 4)
    assert denominators["physical_measurements_planned"] == 0
    assert denominators["certification_abstained"] == 1
    assert result["physical_score"] is None

    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["block_length"] == 8
    assert protocol["bootstrap_replicates_per_outer_panel"] == 40
    assert protocol["source_sha256"][mbb.Path(mbb.__file__).name] == hashlib.sha256(
        mbb.Path(mbb.__file__).read_bytes()
    ).hexdigest()
    manifest = json.loads((output / "artifact_hashes.json").read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert len(result["comparison_rows"]) == 12
    assert (output / "method_coverage.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (output / "README.md").read_text().count("moving_block_bootstrap") > 0
    with pytest.raises(FileExistsError):
        mbb.run(
            output, outer_panels_per_scenario=1, bootstrap_replicates=40,
            block_length=8, scenarios=("matched_3state_independent",),
        )


@pytest.mark.parametrize("outer,boot,length,match", [
    (0, 40, 8, r"in \[1,40\]"),
    (1, 0, 8, "positive integer"),
    (1, 40, 97, r"in \[1,n_blocks\]"),
])
def test_mbb_run_rejects_invalid_design_before_creating_output(
    tmp_path, outer, boot, length, match,
):
    output = tmp_path / "invalid-mbb-report"
    with pytest.raises(ValueError, match=match):
        mbb.run(output, outer_panels_per_scenario=outer,
                bootstrap_replicates=boot, block_length=length,
                scenarios=("matched_3state_independent",))
    assert not output.exists()
