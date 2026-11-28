from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pytest

from core.source_integrity import capture_source_snapshot
from tasks.wilson_loop import f_lane_finite_nt_trace_diagnostic as toy
from tasks.wilson_loop.f_lane_two_sector_fit_candidate import (
    RAW_MANIFEST_SCHEMA,
    RAW_SCHEMA,
    NT_VALUES,
    R_VALUES,
    T_VALUES,
    _validate_raw_snapshot,
    fit_two_sector_snapshot,
)


def _digest(value: bytes) -> str:
    return sha256(value).hexdigest()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _make_snapshot(tmp_path: Path, *, split_id="F-synthetic-development",
                   dimension_order="block,Nt,R,T,i,j", nonhermitian=False):
    nblocks = 96
    rng = np.random.default_rng(2026112712)
    matrices = np.empty((nblocks, 2, 2, 5, 3, 3), dtype=np.complex128)
    for radius_index, radius in enumerate(R_VALUES):
        static = toy.STATIC_ENERGIES[radius][:2]
        amplitudes = toy.TRANSITION_OVERLAPS[radius_index, :, :2, :]
        for nt_index, temporal_extent in enumerate(NT_VALUES):
            partition = float(np.sum(np.exp(-toy.VACUUM_ENERGIES * temporal_extent)))
            for time_index, time_extent in enumerate(T_VALUES):
                weights = np.exp(
                    -toy.VACUUM_ENERGIES[:, None] * (temporal_extent - time_extent)
                    -static[None, :] * time_extent
                ) / partition
                mean = np.einsum("mn,mni,mnj->ij", weights, amplitudes,
                                 amplitudes.conj(), optimize=True)
                sigma = max(float(np.max(np.abs(mean))) * 0.003, 1e-9)
                for block in range(nblocks):
                    raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
                    noise = 0.5 * (raw + raw.conj().T)
                    noise[np.diag_indices(3)] = rng.normal(size=3)
                    matrices[block, nt_index, radius_index, time_index] = mean + sigma * noise
    if nonhermitian:
        matrices[0, 0, 0, 0, 0, 1] += 1e-3j

    data_stream = BytesIO()
    np.savez_compressed(
        data_stream, schema=np.asarray(RAW_SCHEMA), block_ids=np.arange(nblocks),
        correlator_matrices=matrices, R_values=np.asarray(R_VALUES),
        temporal_extents=np.asarray(NT_VALUES), time_slices=np.asarray(T_VALUES),
    )
    data = data_stream.getvalue()
    contract = (Path(__file__).resolve().parents[1]
                / "tasks/wilson_loop/lane_contract_v2.json").read_bytes()
    manifest = {
        "schema": RAW_MANIFEST_SCHEMA, "split_id": split_id,
        "geometry": "S16T16", "action": "wilson_gauge_beta6",
        "boundary_conditions": "periodic", "radius_order": list(R_VALUES),
        "temporal_extents": list(NT_VALUES), "time_slices": list(T_VALUES),
        "dimension_order": dimension_order,
        "matrix_sha256": _digest(data), "lane_contract_sha256": _digest(contract),
    }
    manifest_bytes = _json_bytes(manifest)
    (tmp_path / "manifest.json").write_bytes(manifest_bytes)
    (tmp_path / "raw.npz").write_bytes(data)
    (tmp_path / "lane_contract.json").write_bytes(contract)
    expected = {
        "manifest": _digest(manifest_bytes), "raw_matrices": _digest(data),
        "lane_contract": _digest(contract),
    }
    snapshot = capture_source_snapshot(
        tmp_path, expected,
        {"manifest": "manifest.json", "raw_matrices": "raw.npz",
         "lane_contract": "lane_contract.json"},
    )
    return snapshot, matrices


def test_two_sector_fit_recovers_full_rank_synthetic_candidate(tmp_path):
    snapshot, _ = _make_snapshot(tmp_path)
    (tmp_path / "raw.npz").write_bytes(b"replaced after capture")
    result = fit_two_sector_snapshot(snapshot)
    assert result["schema"] == "wilson-f-two-sector-fit-candidate-v1"
    assert result["status"] == "two_sector_fit_candidate"
    assert result["certified"] is False and result["physical_score"] is None
    assert result["denominator"] == {
        "planned": 1, "valid": 1, "failed": 0, "abstained": 0,
    }
    assert result["fit_contract"]["parameter_count"] == 45
    assert result["fit_contract"]["data_components"] == 180
    assert result["fit_contract"]["jacobian_rank"] == 45
    assert abs(result["spectrum_fit"]["vacuum_energies"][1]
               - toy.VACUUM_ENERGIES[1]) < 0.1
    for radius in R_VALUES:
        fitted = np.asarray(result["spectrum_fit"]["static_energies_by_R"][str(radius)])
        assert np.allclose(fitted, toy.STATIC_ENERGIES[radius][:2], atol=0.12, rtol=0)
    assert result["qhx_denominator"] == {
        "planned": 24, "valid": 24, "failed": 0, "abstained": 0,
    }
    assert len(result["qhx_rows"]) == 24
    assert max(row["decomposition_abs_error"] for row in result["qhx_rows"]) < 1e-10
    assert any("independently reviewed" in gate for gate in result["open_gates"])


def test_two_sector_reader_rejects_closed_or_wrong_axis_manifest(tmp_path):
    snapshot, _ = _make_snapshot(tmp_path, split_id="F-holdout-closed")
    with pytest.raises(ValueError, match="open F"):
        _validate_raw_snapshot(snapshot, manifest_name="manifest",
                               matrix_name="raw_matrices", contract_name="lane_contract")
    other = tmp_path / "axis"
    other.mkdir()
    snapshot, _ = _make_snapshot(other, dimension_order="block,R,Nt,T,i,j")
    with pytest.raises(ValueError, match="axis order"):
        _validate_raw_snapshot(snapshot, manifest_name="manifest",
                               matrix_name="raw_matrices", contract_name="lane_contract")


def test_two_sector_reader_rejects_nonhermitian_matrix_and_stale_snapshot(tmp_path):
    snapshot, _ = _make_snapshot(tmp_path, nonhermitian=True)
    with pytest.raises(ValueError, match="non-Hermitian"):
        _validate_raw_snapshot(snapshot, manifest_name="manifest",
                               matrix_name="raw_matrices", contract_name="lane_contract")
    good = tmp_path / "good"
    good.mkdir()
    snapshot, _ = _make_snapshot(good)
    corrupted = snapshot.source_bytes["raw_matrices"] + b" "
    forged = replace(snapshot, source_bytes={**snapshot.source_bytes, "raw_matrices": corrupted})
    with pytest.raises(ValueError, match="captured hash"):
        _validate_raw_snapshot(forged, manifest_name="manifest",
                               matrix_name="raw_matrices", contract_name="lane_contract")
