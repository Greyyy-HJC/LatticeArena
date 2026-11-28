from __future__ import annotations

from hashlib import sha256
import json

import pytest

from tasks.pion_projection import anchor_multitime_alias_diagnostic as diagnostic


def test_positive_spectra_alias_on_all_four_features_and_frozen_scalar():
    rows = diagnostic._models()
    assert [row["ground_mass"] for row in rows] == pytest.approx([0.15, 0.45])
    assert rows[1]["ground_mass"] - rows[0]["ground_mass"] == pytest.approx(0.30)
    for row in rows:
        assert all(value > 0.0 for value in row["amplitudes"])
        assert all(0.1 <= mass <= 3.0 for mass in row["masses"])
        assert sum(row["weights_at_t4"]) == pytest.approx(1.0)
        assert all(value > 0.0 for value in row["weights_at_t4"])
    assert rows[0]["feature_vector_four_time"] == pytest.approx(
        rows[1]["feature_vector_four_time"], abs=1e-12
    )
    assert rows[0]["reader_m_eff4"] == pytest.approx(
        rows[1]["reader_m_eff4"], abs=1e-12
    )


def test_report_preserves_hashes_denominators_and_abstained_physical_gate(tmp_path):
    output = tmp_path / "multi-time-alias"
    diagnostic.build(output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["physical_score"] is None
    assert summary["diagnostic"]["max_abs_unique_feature_gap"] <= 1e-12
    assert summary["diagnostic"]["ground_mass_gap"] == pytest.approx(0.30)
    assert summary["denominators"]["positive_spectra"] == {
        "planned": 2, "valid": 2, "failed": 0, "abstained": 0
    }
    assert summary["denominators"]["physical_anchor_inputs"] == {
        "planned": 0, "valid": 0, "failed": 0, "abstained": 0
    }
    assert summary["denominators"]["physical_gate"] == {
        "planned": 1, "valid": 0, "failed": 0, "abstained": 1
    }

    manifest = json.loads((output / "artifact_hashes.json").read_text())["artifacts"]
    assert len(manifest) == 13
    for relative_path, expected_hash in manifest.items():
        assert sha256((output / relative_path).read_bytes()).hexdigest() == expected_hash
    sources = json.loads((output / "source_hashes.json").read_text())["sources"]
    assert len(sources) == 6
    assert "core_npz_safety.py" in sources
    assert "core_source_integrity.py" in sources
    for name, expected_hash in sources.items():
        assert sha256((output / "source_snapshot" / name).read_bytes()).hexdigest() == expected_hash
    assert (output / "multitime_alias.png").is_file()
    assert "lowest masses differ" in (output / "README.md").read_text()


def test_report_refuses_to_overwrite_existing_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        diagnostic.build(output)
    assert marker.read_text() == "keep\n"
