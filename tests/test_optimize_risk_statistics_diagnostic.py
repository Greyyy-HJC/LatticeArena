import json

from tasks.optimize_gaussian.risk_statistics_diagnostic import build_report


def test_risk_statistics_diagnostic_is_source_bound_and_explicitly_nonphysical(tmp_path):
    output = tmp_path / "diagnostic"

    result = build_report(output)
    payload = json.loads((output / "diagnostic.json").read_text())
    readme = (output / "README.md").read_text()

    assert result["all_physical_scores_null"] is True
    assert set(payload["source_hashes"]) == {
        "risk_statistics.py", "research_contract_v1.json",
    }
    assert payload["scenarios"]["d04_hand_calculation"]["risk"] == 0.7847222222222222
    assert payload["scenarios"]["d06_synthetic_hull_clears"]["draw_denominators"] == {
        "planned": 12, "valid": 12, "failed": 0, "abstained": 0,
    }
    assert payload["scenarios"]["d06_synthetic_zero_risk_abstains"]["hull_lower"] is None
    assert "do **not** show" in readme
    assert (output / "artifact_hashes.json").is_file()
