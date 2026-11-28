import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tasks.wilson_loop import gevp_overlap_object_diagnostic as diagnostic
from tasks.wilson_loop.gevp_overlap_object_diagnostic import (
    OVERLAPS_4X3,
    TOY_ENERGIES_4,
    _calculate_rows,
    build,
    correlation_matrix,
    principal_gevp_vector,
    projected_state_overlaps,
    spectral_q,
)


def test_exact_three_state_complete_basis_cancels_excited_overlaps():
    energies = np.array([0.18, 0.46, 0.83])
    overlaps = np.array([
        [1.0, 0.3, 0.2],
        [0.2, 1.0, 0.4],
        [0.3, 0.2, 1.0],
    ])
    v, _ = principal_gevp_vector(
        correlation_matrix(energies, overlaps, 2),
        correlation_matrix(energies, overlaps, 1),
    )
    projected = projected_state_overlaps(overlaps, v)
    assert spectral_q(energies, projected, 5) < 1e-24
    assert np.real(v.conj() @ correlation_matrix(energies, overlaps, 1) @ v) == pytest.approx(1.0)


def test_omitted_fourth_state_makes_q_depend_on_operator_object_and_td():
    rows = _calculate_rows()
    lookup = {(row["operator_object"], row["T"]): row["q_T"] for row in rows}
    bare_max_t5 = max(lookup[(f"bare_operator_{j}", 5)] for j in range(3))
    selected_t5 = lookup[("principal_gevp_projection_td_2", 5)]
    later_t5 = lookup[("principal_gevp_projection_td_5", 5)]
    assert selected_t5 > 5.0 * bare_max_t5
    assert selected_t5 > 10.0 * later_t5
    assert len(rows) == 14


def test_projected_q_is_invariant_to_vector_scale_and_phase():
    c0 = correlation_matrix(TOY_ENERGIES_4, OVERLAPS_4X3, 1)
    c2 = correlation_matrix(TOY_ENERGIES_4, OVERLAPS_4X3, 2)
    vector, _ = principal_gevp_vector(c2, c0)
    q = spectral_q(TOY_ENERGIES_4, projected_state_overlaps(OVERLAPS_4X3, vector), 5)
    scaled = vector * (2.7 * np.exp(0.63j))
    q_scaled = spectral_q(
        TOY_ENERGIES_4, projected_state_overlaps(OVERLAPS_4X3, scaled), 5
    )
    assert q_scaled == pytest.approx(q, rel=1e-13)


def test_projected_q_is_invariant_under_invertible_operator_basis_change():
    transform = np.array([
        [1.0, 0.2, 0.0],
        [0.0, 1.0, 0.3],
        [0.1, 0.0, 1.0],
    ])
    transformed_overlaps = OVERLAPS_4X3 @ transform.T
    original_c1 = correlation_matrix(TOY_ENERGIES_4, OVERLAPS_4X3, 1)
    original_c2 = correlation_matrix(TOY_ENERGIES_4, OVERLAPS_4X3, 2)
    transformed_c1 = transform @ original_c1 @ transform.T
    transformed_c2 = transform @ original_c2 @ transform.T
    original_vector, _ = principal_gevp_vector(original_c2, original_c1)
    transformed_vector, _ = principal_gevp_vector(transformed_c2, transformed_c1)
    original_q = spectral_q(
        TOY_ENERGIES_4,
        projected_state_overlaps(OVERLAPS_4X3, original_vector), 5,
    )
    transformed_q = spectral_q(
        TOY_ENERGIES_4,
        projected_state_overlaps(transformed_overlaps, transformed_vector), 5,
    )
    assert transformed_q == pytest.approx(original_q, rel=1e-12)


def test_rejects_singular_metric_and_zero_ground_overlap():
    with pytest.raises(ValueError, match="positive definite"):
        principal_gevp_vector(np.eye(3), np.diag([1.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match="ground-state overlap"):
        spectral_q([0.1, 0.2], [0.0, 1.0], 5)


def test_report_binds_artifacts_and_keeps_physics_gate_closed(tmp_path):
    out = tmp_path / "overlap-report"
    summary = build(out)
    assert summary["status"] == "DIAGNOSTIC_ONLY"
    assert summary["physical_score"] is None
    assert summary["denominators"]["physical_certification_gate"] == {
        "unit": "F-lane certification", "planned": 1, "valid": 0,
        "failed": 0, "abstained": 1,
    }
    assert (out / "overlap_object_q.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    hashes = json.loads((out / "artifact_hashes.json").read_text(encoding="utf-8"))
    for name, digest in hashes.items():
        if name == "gevp_overlap_object_diagnostic.py":
            assert hashlib.sha256(Path(diagnostic.__file__).read_bytes()).hexdigest() == digest
        elif name == "lane_contract_v2.json":
            assert hashlib.sha256(diagnostic.CONTRACT_PATH.read_bytes()).hexdigest() == digest
        elif name == "gevp_t0_truncation_diagnostic.py":
            assert hashlib.sha256(
                Path(diagnostic.__file__).with_name(name).read_bytes()
            ).hexdigest() == digest
        else:
            assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build(out)
