import hashlib
import json

import pytest

from tasks.wilson_loop import f_lane_joint_spectral_mbb_b80_diagnostic as b80


def test_verified_b40_bundle_matches_its_protocol_and_denominators():
    protocol, summary, bootstrap_rows, interval_rows = b80._read_verified_b40(b80.DEFAULT_BASE)
    assert protocol["bootstrap_replicates_per_outer_panel"] == 40
    assert len(bootstrap_rows) == summary["denominators"]["bootstrap_refits_planned"] == 9600
    assert len(interval_rows) == summary["denominators"]["moving_block_intervals_planned"] == 960


def test_b80_extension_reuses_b40_draws_and_reconciles_report(tmp_path):
    output = tmp_path / "b80-report"
    result = b80.run(
        output, scenarios=("matched_3state_independent",),
        outer_panels_per_scenario=1,
    )
    den = result["denominators"]
    assert den["additional_refits_planned"] == 40
    assert den["b80_refits_planned"] == 80
    assert (den["b80_refits_valid"] + den["b80_refits_failed"]
            + den["b80_refits_abstained"] == 80)
    assert den["b80_q_estimates_planned"] == 320
    assert (den["b80_q_estimates_valid"] + den["b80_q_estimates_failed"]
            + den["b80_q_estimates_abstained"] == 320)
    assert den["b80_intervals_planned"] == 4
    assert (den["b80_intervals_valid"] + den["b80_intervals_failed"]
            + den["b80_intervals_abstained"] == 4)
    assert den["physical_measurements_planned"] == 0
    assert den["certification_abstained"] == 1
    assert result["physical_score"] is None
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["base_bootstrap_replicates"] == 40
    assert protocol["total_bootstrap_replicates"] == 80
    manifest = json.loads((output / "artifact_hashes.json").read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert len(result["comparison_rows"]) == 16
    assert (output / "method_coverage.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "B=80" in (output / "README.md").read_text()
    with pytest.raises(FileExistsError):
        b80.run(output, scenarios=("matched_3state_independent",),
                outer_panels_per_scenario=1)


@pytest.mark.parametrize("outer,total,match", [
    (0, 80, r"\[1,40\]"),
    (1, 40, "greater than 40"),
])
def test_b80_extension_rejects_invalid_design_before_output(tmp_path, outer, total, match):
    output = tmp_path / "invalid-b80-report"
    with pytest.raises(ValueError, match=match):
        b80.run(output, scenarios=("matched_3state_independent",),
                outer_panels_per_scenario=outer,
                total_bootstrap_replicates=total)
    assert not output.exists()


def test_bootstrap_seed_extension_starts_after_original_prefix():
    scenario = "near_degenerate_first_gap"
    replicate = 47
    seeds = [b80._bootstrap_seed(scenario, replicate, k) for k in range(40, 80)]
    assert len(set(seeds)) == 40
    assert seeds[0] + 39 == seeds[-1]
    assert seeds[0] > b80._bootstrap_seed(scenario, replicate, 39)
