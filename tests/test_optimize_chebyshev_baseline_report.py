import hashlib
import json
from pathlib import Path

from tasks.optimize_gaussian.chebyshev_baseline import build


def test_constrained_baseline_report_is_hashed_and_labels_illustration_denominators(tmp_path):
    output = tmp_path / "constrained-baseline"
    contract = build(output)

    assert contract["status"] == "METHOD_DEFAULTS_SELECTED_EVIDENCE_NOT_READY"
    assert contract["selected_method_defaults"]["D06_effect_threshold"][
        "r_star"
    ] == 1.1111111111111112
    assert len(contract["open_gates"]) == 5
    assert contract["physical_score"] is None
    numerics_path = Path(__file__).resolve().parents[1] / "core/numerics.py"
    assert contract["numeric_validation_source_sha256"] == hashlib.sha256(
        numerics_path.read_bytes()
    ).hexdigest()
    assert contract["coefficient_fit_denominators"] == {
        "planned": 0, "valid": 0, "failed": 0, "abstained": 0,
        "unit": "no D04 callback supplied",
    }
    assert contract["illustration_denominators"] == {
        "planned": 3,
        "valid": 3,
        "failed": 0,
        "abstained": 0,
        "unit": "deterministic response examples, not fit runs or physical samples",
    }
    readme = (output / "README.md").read_text()
    assert "planned/valid/failed/abstained = `3/3/0/0`" in readme
    assert "No reference or physics objective" in readme
    assert "fitted-run denominator = 0/0/0/0" in readme
    assert "Text flow:" in readme
    assert "exact symbolic basis transform" in readme
    assert "float64" in readme

    manifest = json.loads((output / "artifact_hashes.json").read_text())
    assert set(manifest) == {
        "contract.json",
        "spectral_response.png",
        "README.md",
        "snapshot_chebyshev_baseline.py",
        "snapshot_kernels.py",
        "snapshot_core_numerics.py",
    }
    for name, expected in manifest.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert (output / "spectral_response.png").is_file()
