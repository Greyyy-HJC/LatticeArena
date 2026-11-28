import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tasks.wilson_loop import gevp_t0_truncation_diagnostic as diagnostic
from tasks.wilson_loop.gevp_t0_truncation_diagnostic import (
    CONTRACT_PATH,
    OVERLAPS_4X3,
    TOY_ENERGIES_4,
    _calculate_rows,
    build,
    correlation_matrix,
    effective_ground_energy,
    principal_eigenvalues,
)


def test_full_three_state_basis_recovers_exact_ground_energy():
    energies = np.array([0.18, 0.46, 0.83])
    overlaps = np.array([
        [1.0, 0.3, 0.2],
        [0.2, 1.0, 0.4],
        [0.3, 0.2, 1.0],
    ])
    for t0, time in ((1, 2), (1, 4), (2, 5)):
        estimate, _, _ = effective_ground_energy(
            correlation_matrix(energies, overlaps, time),
            correlation_matrix(energies, overlaps, time + 1),
            correlation_matrix(energies, overlaps, t0),
        )
        assert estimate == pytest.approx(energies[0], abs=2e-12)


def test_fourth_omitted_state_produces_finite_basis_bias_in_selected_window():
    rows = _calculate_rows()
    selected = [row for row in rows if row["mode"] == "fixed_t0_1"]
    condition = [row for row in rows if row["mode"] == "paired_time_condition_t0_ge_half"]
    assert len(rows) == 19
    assert len(selected) == 5
    assert len(condition) == 5
    assert max(row["abs_bias"] for row in selected) > 0.10
    assert max(row["abs_bias"] for row in condition) < max(
        row["abs_bias"] for row in selected
    )


def test_gevp_rejects_nonhermitian_or_singular_reference_matrix():
    good = correlation_matrix(TOY_ENERGIES_4, OVERLAPS_4X3, 1)
    bad_nonhermitian = good.copy()
    bad_nonhermitian[0, 1] += 1e-3
    with pytest.raises(ValueError, match="Hermitian"):
        principal_eigenvalues(good, bad_nonhermitian)

    bad_singular = np.diag([1.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="positive definite"):
        principal_eigenvalues(good, bad_singular)


def test_report_binds_plot_protocol_and_does_not_overwrite(tmp_path):
    out = tmp_path / "gevp-report"
    summary = build(out)
    assert summary["status"] == "DIAGNOSTIC_ONLY"
    assert summary["physical_score"] is None
    assert summary["lane_status"] == "GROUND_STATE_FORCE_UNCERTIFIED"
    assert summary["denominators"]["deterministic_gevp_points"] == {
        "planned": 19, "valid": 19, "failed": 0, "abstained": 0,
        "unit": "one fixed-time pair (mode,t,t0)",
    }
    assert (out / "gevp_t0_sensitivity.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "meaning of `td`" in (out / "README.md").read_text(encoding="utf-8")

    hashes = json.loads((out / "artifact_hashes.json").read_text(encoding="utf-8"))
    for name, digest in hashes.items():
        if name == "gevp_t0_truncation_diagnostic.py":
            assert hashlib.sha256(Path(diagnostic.__file__).read_bytes()).hexdigest() == digest
            continue
        if name == "lane_contract_v2.json":
            assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == digest
            continue
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build(out)
