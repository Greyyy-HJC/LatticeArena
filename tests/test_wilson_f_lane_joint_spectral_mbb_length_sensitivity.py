import pytest

from tasks.wilson_loop import f_lane_joint_spectral_mbb_length_sensitivity as scan
from tasks.wilson_loop import f_lane_joint_spectral_mbb_diagnostic as mbb
from tasks.wilson_loop import f_lane_joint_spectral_stress_diagnostic as stress


def test_scan_family_and_shared_seed_keys_are_fixed(monkeypatch):
    assert scan.LENGTHS == (6, 8, 10)
    assert scan.NEW_LENGTHS == (6, 10)
    assert scan.OUTER_PANELS_PER_SCENARIO == 40
    assert scan.BOOTSTRAP_REPLICATES == 40

    monkeypatch.setattr(stress, "SCENARIOS", ("matched_3state_independent",))
    monkeypatch.setattr(scan, "OUTER_PANELS_PER_SCENARIO", 2)
    monkeypatch.setattr(scan, "BOOTSTRAP_REPLICATES", 3)
    reports = {}
    for length in scan.LENGTHS:
        reports[length] = [{
            "scenario": "matched_3state_independent",
            "outer_replicate": str(replicate),
            "bootstrap_replicate": str(index),
            "outer_seed": str(2026112700 + replicate),
            "bootstrap_seed": str(2026112700 + replicate + mbb.SEED_OFFSET + index),
        } for replicate in (40, 41) for index in range(3)]
    assert scan._verify_shared_seed_keys(reports) == 6

    reports[10][0]["bootstrap_seed"] = "0"
    with pytest.raises(ValueError, match="seed keys differ"):
        scan._verify_shared_seed_keys(reports)


def test_l8_baseline_report_is_hash_verified_and_matches_frozen_scan():
    protocol, summary, bootstrap_rows = scan._validate_baseline(scan.DEFAULT_BASE)
    assert protocol["block_length"] == 8
    assert protocol["bootstrap_replicates_per_outer_panel"] == 40
    assert summary["denominators"]["moving_block_intervals_planned"] == 960
    assert len(bootstrap_rows) == 9600


def test_collect_rows_keeps_all_cells_and_reports_differences_from_l8():
    reports = {}
    for length in scan.LENGTHS:
        rows = []
        for scenario in stress.SCENARIOS:
            for operator_index in (0, 2):
                for time in (5, 6):
                    rows.extend([
                        {"scenario": scenario, "operator_index": operator_index,
                         "time": time, "method": "delta", "coverage": 0.95},
                        {"scenario": scenario, "operator_index": operator_index,
                         "time": time, "method": "split_conformal", "coverage": 1.0},
                        {"scenario": scenario, "operator_index": operator_index,
                         "time": time, "method": "moving_block_bootstrap",
                         "planned": 40, "valid": 38 if length == 6 else 40,
                         "failed": 0, "abstained": 2 if length == 6 else 0,
                         "coverage": 0.9 if length == 6 else 0.95,
                         "mc95_low": 0.8, "mc95_high": 0.99,
                         "median_upper_over_true": 1.2,
                         "bootstrap_replicates": 40},
                    ])
        reports[length] = rows

    rows = scan._collect_rows(reports)
    assert len(rows) == 72
    l6_row = next(row for row in rows if row["block_length"] == 6)
    l8_row = next(row for row in rows if row["block_length"] == 8)
    assert l6_row["planned"] == 40
    assert l6_row["yield"] == 0.95
    assert l6_row["coverage_delta_vs_L8"] == pytest.approx(-0.05)
    assert l8_row["coverage_delta_vs_L8"] == 0.0


def test_existing_output_is_not_overwritten(tmp_path):
    output = tmp_path / "existing-report"
    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("keep\n")
    with pytest.raises(FileExistsError):
        scan.run(output)
    assert marker.read_text() == "keep\n"


def test_top_level_readme_explains_shared_holdout_and_no_valid_marker():
    rows = []
    for length in scan.LENGTHS:
        for scenario in stress.SCENARIOS:
            for operator_index in (0, 2):
                for time in (5, 6):
                    rows.append({
                        "block_length": length, "scenario": scenario,
                        "operator_index": operator_index, "time": time,
                        "planned": 40, "valid": 0, "failed": 0, "abstained": 40,
                        "yield": 0.0, "coverage": None, "mc95_low": None,
                        "mc95_high": None, "coverage_delta_vs_L8": None,
                        "median_upper_over_true": None,
                    })
    d = {"planned": 960, "valid": 0, "failed": 0, "abstained": 960}
    summary = {
        "per_length_denominators": {str(length): d for length in scan.LENGTHS},
        "combined_interval_denominators": {
            "planned": 2880, "valid": 0, "failed": 0, "abstained": 2880,
        },
    }
    readme = scan._top_readme(summary, rows, seed_keys=9600)
    assert "not an independent confirmation" in readme
    assert "x` means zero valid intervals" in readme
    assert "The scan stops at the pre-fixed" in readme


def test_plot_tolerates_wilson_upper_roundoff_without_hiding_real_mismatch(tmp_path):
    rows = []
    for length in scan.LENGTHS:
        for scenario in stress.SCENARIOS:
            for operator_index in (0, 2):
                for time in (5, 6):
                    special = (length == 10 and scenario == stress.SCENARIOS[0]
                               and operator_index == 0 and time == 5)
                    rows.append({
                        "block_length": length, "scenario": scenario,
                        "operator_index": operator_index, "time": time,
                        "coverage": 1.0 if special else 0.95,
                        "mc95_low": 0.9,
                        "mc95_high": 0.9999999999999999 if special else 0.99,
                    })
    output = tmp_path / "coverage.png"
    scan._plot(rows, output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    rows[0]["mc95_high"] = 0.8
    with pytest.raises(ValueError, match="does not contain"):
        scan._plot(rows, output)
