from pathlib import Path

from tasks.wilson_loop import f_lane_two_sector_omitted_coverage_diagnostic as diagnostic


def test_report_source_map_binds_transitive_fit_and_raw_reader_dependencies():
    expected = {
        "two_sector_fit_initializer.py": "f_lane_two_sector_fit_diagnostic.py",
        "gevp_overlap_object.py": "gevp_overlap_object_diagnostic.py",
        "gevp_t0_truncation.py": "gevp_t0_truncation_diagnostic.py",
    }
    for key, filename in expected.items():
        assert diagnostic.SOURCE_PATHS[key] == Path(diagnostic.TASK_DIR / filename)
