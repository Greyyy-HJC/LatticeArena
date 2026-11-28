from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from core.source_integrity import capture_source_snapshot
from tasks.wilson_loop import f_lane_transfer_trace_reader as reader


def _transaction(tmp_path, *, split="F_open_calibration", mutate_fit=None,
                 mutate_manifest=None, contract_bytes=None):
    contract = (reader.CONTRACT_PATH.read_bytes() if contract_bytes is None
                else contract_bytes)
    eps = [0.0, 0.18, 0.31]
    static = {
        "2": [0.42, 0.71, 1.03],
        "3": [0.51, 0.83, 1.17],
    }
    overlaps = {}
    for radius in reader.RADIUS_ORDER:
        values = np.empty((len(eps), len(static[str(radius)]),
                           len(reader.KERNEL_ORDER), 2), dtype=np.float64)
        for m in range(len(eps)):
            for n in range(len(static[str(radius)])):
                for k in range(len(reader.KERNEL_ORDER)):
                    values[m, n, k] = (1.0 if (m, n) == (0, 0)
                                       else 0.05 * (1 + m + n + k),
                                       0.01 * (m - n) * (k + 1))
        overlaps[str(radius)] = values.tolist()
    fit = {
        "schema": reader.FIT_SCHEMA,
        "radius_order": list(reader.RADIUS_ORDER),
        "kernel_order": list(reader.KERNEL_ORDER),
        "vacuum_energies": eps,
        "static_energies_by_R": static,
        "transition_overlaps_by_R": overlaps,
    }
    if mutate_fit is not None:
        mutate_fit(fit)
    fit_bytes = (json.dumps(fit, sort_keys=True, separators=(",", ":")) + "\n").encode()
    protocol_hash = sha256(contract).hexdigest()
    manifest = {
        "schema": reader.MANIFEST_SCHEMA,
        "split_id": split,
        "chain_id": "synthetic-chain-a",
        "geometry": "S16T16",
        "action": "wilson_gauge_beta6",
        "radius_order": list(reader.RADIUS_ORDER),
        "kernel_order": list(reader.KERNEL_ORDER),
        "time_extents": list(reader.TIME_EXTENTS),
        "temporal_extents": list(reader.TEMPORAL_EXTENTS),
        "fit_summary_sha256": sha256(fit_bytes).hexdigest(),
        "source_protocol_sha256": protocol_hash,
    }
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    files = {
        "manifest.json": manifest_bytes,
        "fit.json": fit_bytes,
        "lane_contract.json": contract,
    }
    for name, payload in files.items():
        (tmp_path / name).write_bytes(payload)
    names = {"transfer_manifest": "manifest.json", "transfer_fit": "fit.json",
             "lane_contract": "lane_contract.json"}
    hashes = {key: sha256(files[path]).hexdigest() for key, path in names.items()}
    snapshot = capture_source_snapshot(tmp_path, hashes, names)
    return snapshot, fit


def test_reader_reconstructs_all_qhx_cells_from_one_snapshot(tmp_path):
    snapshot, _ = _transaction(tmp_path)
    (tmp_path / "fit.json").write_bytes(b"changed after capture")
    (tmp_path / "lane_contract.json").write_bytes(b"changed after capture")

    result = reader.read_transfer_trace_snapshot(snapshot)

    assert result["status"] == "diagnostic_only"
    assert result["certified"] is False and result["physical_score"] is None
    assert result["cell_denominator"] == {
        "unit": "R_kernel_T_Nt_cell", "planned": 24, "valid": 24,
        "failed": 0, "abstained": 0,
    }
    assert result["source_file_denominator"] == {
        "unit": "captured_source_file", "planned": 3, "valid": 3,
        "failed": 0, "abstained": 0,
    }
    assert result["physical_measurement_denominator"] == {
        "unit": "authorized_production_measurement", "planned": 0,
        "valid": 0, "failed": 0, "abstained": 0,
    }
    assert result["certification_gate_denominator"] == {
        "unit": "F_lane_certification_gate", "planned": 1, "valid": 0,
        "failed": 0, "abstained": 1,
    }
    assert len(result["cells"]) == 24
    for row in result["cells"]:
        assert row["q"] >= 0 and row["h"] >= 0 and row["x"] >= 0
        assert row["decomposition_abs_error"] < 1e-12
        assert row["full_to_ground_ratio"] == pytest.approx(
            row["reconstructed_ratio"], rel=1e-12,
        )
    assert result["open_gates"]


def test_reader_keeps_zero_ground_overlap_cells_in_failed_denominator(tmp_path):
    def zero_one_bare_operator(fit):
        values = np.asarray(fit["transition_overlaps_by_R"]["2"], dtype=np.float64)
        values[0, 0, 0] = (0.0, 0.0)
        fit["transition_overlaps_by_R"]["2"] = values.tolist()

    snapshot, _ = _transaction(tmp_path, mutate_fit=zero_one_bare_operator)
    result = reader.read_transfer_trace_snapshot(snapshot)

    assert result["status"] == "ABSTAIN_INVALID_TRANSFER_CELLS"
    assert result["cell_denominator"] == {
        "unit": "R_kernel_T_Nt_cell", "planned": 24, "valid": 20,
        "failed": 4, "abstained": 0,
    }
    assert len(result["failure_reasons"]) == 4
    assert {row["kernel_id"] for row in result["failure_reasons"]} == {"identity"}
    assert {row["R"] for row in result["failure_reasons"]} == {2}


def test_reader_rejects_closed_or_sealed_split(tmp_path):
    snapshot, _ = _transaction(tmp_path, split="F_sealed_confirmation")
    with pytest.raises(ValueError, match="open F calibration/development"):
        reader.read_transfer_trace_snapshot(snapshot)


def test_reader_rejects_manifest_fit_hash_mismatch(tmp_path):
    snapshot, _ = _transaction(
        tmp_path, mutate_manifest=lambda manifest: manifest.update(
            fit_summary_sha256="0" * 64),
    )
    with pytest.raises(ValueError, match="fit bytes do not match"):
        reader.read_transfer_trace_snapshot(snapshot)


def test_reader_rejects_manifest_protocol_hash_mismatch(tmp_path):
    snapshot, _ = _transaction(
        tmp_path, mutate_manifest=lambda manifest: manifest.update(
            source_protocol_sha256="0" * 64),
    )
    with pytest.raises(ValueError, match="protocol bytes do not match"):
        reader.read_transfer_trace_snapshot(snapshot)


def test_reader_rejects_wrong_operator_mapping_contract(tmp_path):
    contract = json.loads(reader.CONTRACT_PATH.read_text())
    contract["selected_protocol"]["ground_state_discrete_static_force"][
        "spectral_evidence_plan"]["q_overlap_contract"]["kernel_to_basis_operator"][
            "spatial_stout_rho0.10_n5"] = "identity"
    payload = (json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n").encode()
    snapshot, _ = _transaction(tmp_path, contract_bytes=payload)
    with pytest.raises(ValueError, match="q/h/x reader semantics"):
        reader.read_transfer_trace_snapshot(snapshot)


@pytest.mark.parametrize("mutation, message", [
    (lambda fit: fit.update(vacuum_energies=[0.0, 0.0, 0.31]), "vacuum energies"),
    (lambda fit: fit["static_energies_by_R"].update({"2": [0.42, 0.42, 1.03]}),
     "static energies"),
    (lambda fit: fit["transition_overlaps_by_R"].update({"3": [[[[0.0, 0.0]]]]}),
     "transition overlaps"),
])
def test_reader_rejects_malformed_spectrum_or_overlap_shapes(tmp_path, mutation, message):
    snapshot, _ = _transaction(tmp_path, mutate_fit=mutation)
    with pytest.raises(ValueError, match=message):
        reader.read_transfer_trace_snapshot(snapshot)


def test_reader_rejects_extra_radius_row(tmp_path):
    snapshot, _ = _transaction(
        tmp_path, mutate_fit=lambda fit: fit["static_energies_by_R"].update(
            {"4": [0.8, 1.0]}),
    )
    with pytest.raises(ValueError, match="exactly the selected R=2,3 rows"):
        reader.read_transfer_trace_snapshot(snapshot)


def test_transfer_trace_report_hashes_match_retained_sources():
    repo_root = Path(__file__).resolve().parents[1]
    report_dir = repo_root / "tasks/wilson_loop/f_lane_transfer_trace_reader_cpu_20261126_v1"
    sources = json.loads((report_dir / "source_hashes.json").read_text())
    artifacts = json.loads((report_dir / "artifact_hashes.json").read_text())

    assert sources["denominator"] == {
        "unit": "source_file", "planned": 4, "valid": 4,
        "failed": 0, "abstained": 0,
    }
    for source_name, entry in sources["sources"].items():
        source_bytes = (repo_root / source_name).read_bytes()
        snapshot_bytes = (report_dir / entry["snapshot"]).read_bytes()
        assert sha256(source_bytes).hexdigest() == entry["sha256"]
        assert snapshot_bytes == source_bytes
    for name, expected in artifacts.items():
        assert sha256((report_dir / name).read_bytes()).hexdigest() == expected
