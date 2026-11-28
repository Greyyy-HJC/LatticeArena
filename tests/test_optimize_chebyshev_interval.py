import json
import hashlib

import numpy as np
import pytest

from tasks.optimize_gaussian.chebyshev_interval import chebyshev_interval_minimum
from tasks.optimize_gaussian.chebyshev_positivity_diagnostic import build


def test_finds_quartic_minimum_at_interior_stationary_point():
    # (x - 0.25)^2 + 0.2 = 0.7625 T0 - 0.5 T1 + 0.5 T2
    result = chebyshev_interval_minimum([0.7625, -0.5, 0.5])
    assert result["minimum"] == pytest.approx(0.2)
    assert result["argmin"] == pytest.approx(0.25)
    assert result["numerically_certified"] is False


def test_checks_interval_endpoints_for_linear_and_constant_polynomials():
    linear = chebyshev_interval_minimum([1.0, 1.0])
    assert linear["minimum"] == pytest.approx(0.0)
    assert linear["argmin"] == pytest.approx(-1.0)

    constant = chebyshev_interval_minimum([3.0])
    assert constant["minimum"] == pytest.approx(3.0)
    assert constant["argmin"] == pytest.approx(-1.0)


def test_rejects_invalid_coefficients_interval_and_tolerances():
    with pytest.raises(ValueError, match="nonempty one-dimensional"):
        chebyshev_interval_minimum([])
    with pytest.raises(ValueError, match="finite real"):
        chebyshev_interval_minimum([1.0, np.nan])
    with pytest.raises(ValueError, match="must be real"):
        chebyshev_interval_minimum([1.0 + 1e-4j])
    with pytest.raises(ValueError, match="increasing endpoints"):
        chebyshev_interval_minimum([1.0], (1.0, -1.0))
    with pytest.raises(ValueError, match="tolerances"):
        chebyshev_interval_minimum([1.0], root_imag_tolerance=-1.0)


def test_report_keeps_all_panel_denominators_and_hashes(tmp_path):
    input_path = tmp_path / "raw.json"
    input_path.write_text(json.dumps({
        "panels": [
            {"seed": 1, "spectral": [{"degree": 4, "coefficients": [1, 0, 0, 0, 0]}]},
            {"seed": 2, "spectral": [{"degree": 4, "coefficients": [1, 0, 0, 0, 0]}]},
            {"seed": 3, "spectral": [{"degree": 4, "coefficients": [1, 0, 0, 0, 0]}]},
        ]
    }))
    out = tmp_path / "report"
    summary = build(out, input_path)
    assert summary["denominators"] == {
        "planned": 3, "valid": 3, "failed": 0, "abstained": 0,
    }
    assert summary["physical_score"] is None
    manifest = json.loads((out / "artifact_hashes.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected
    assert (out / "chebyshev_positivity.png").is_file()
    assert "planned/valid/failed/abstained = 3/3/0/0" in (out / "README.md").read_text()
