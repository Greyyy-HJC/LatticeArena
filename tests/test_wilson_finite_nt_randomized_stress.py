import hashlib
import json

import pytest

from tasks.wilson_loop.finite_nt_randomized_stress import build, generate_cases


def test_randomized_positive_spectra_reproduce_identity_and_envelope():
    first = generate_cases(count=256, seed=71001)
    second = generate_cases(count=256, seed=71001)
    assert first == second
    assert len(first) == 256
    for row in first:
        assert 4 <= row["temporal_extent"] <= 64
        assert 1 <= row["time_extent"] < row["temporal_extent"]
        assert row["identity_abs_error"] < 1e-12
        assert row["envelope_covers_bias"]
        assert row["bias_to_envelope_ratio"] <= 1.0


def test_randomized_case_count_validation():
    with pytest.raises(ValueError, match="count"):
        generate_cases(count=0)


def test_small_randomized_report_has_hash_closed_artifacts(tmp_path):
    output = tmp_path / "wilson-stress"
    summary = build(output, count=64, seed=71002)
    assert summary["planned_cases"] == 64
    assert summary["envelope_covered_cases"] == 64
    assert summary["identity_max_abs_error"] < 1e-12
    manifest = json.loads((output / "artifact_hashes.json").read_text())
    assert len(manifest) == 9
    assert (output / "snapshot_core_numerics.py").is_file()
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["source_sha256"]["core/numerics.py"] == hashlib.sha256(
        (output / "snapshot_core_numerics.py").read_bytes()
    ).hexdigest()
    for name, digest in manifest.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert "not fitted spectra" in (output / "README.md").read_text()
