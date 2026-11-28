"""Strict immutable-snapshot reader for Wilson F-lane raw matrix GEVPs.

This module validates and projects raw block correlator matrices only.  It does
not fit energies/overlaps, construct confidence intervals, or connect to a
physical evidence gate.  The bundled report is driven by an exact synthetic
forward spectrum and is a schema/linear-algebra diagnostic, not measurement
evidence.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.npz_safety import validate_npz_payload
from core.source_integrity import SourceByteSnapshot, capture_source_snapshot
from .gevp_overlap_object_diagnostic import principal_gevp_vector
from .gevp_t0_truncation_diagnostic import (
    OVERLAPS_4X3, TOY_ENERGIES_4, correlation_matrix,
)


SCHEMA = "wilson-f-raw-cross-correlator-matrix-v1"
CLASSIFICATION = "RAW_MATRIX_CANDIDATE_INPUT_NOT_FIT_OR_CERTIFICATION"
OPERATOR_BASIS_IDS = ("identity", "ape_alpha0.4_n5", "spatial_stout_rho0.10_n5")
R_VALUES = (2, 3)
TIME_SLICES = tuple(range(1, 8))
T0 = 1
TD = 2
NPZ_KEYS = {"block_ids", "correlator_matrices", "operator_basis_ids", "R_values", "time_slices", "split_id"}
SOURCE_ROLES = {"manifest", "matrix_npz"}
DEFAULT_OUT = Path(__file__).resolve().parent / "f_lane_raw_matrix_snapshot_diagnostic_20261126_v1"
_CLOSED_MARKERS = ("gap", "validation", "holdout", "sealed", "confirmation")


def _sha_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _parse_json(payload: bytes) -> dict[str, Any]:
    try:
        result = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys,
                            parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("manifest must be valid UTF-8 JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("manifest root must be a JSON object")
    return result


def _validate_manifest(document: object, *, split_id: str, npz_sha256: str) -> dict[str, Any]:
    required = {
        "schema", "classification", "split_id", "geometry", "action",
        "boundary_conditions", "temporal_extent", "operator_basis_ids", "R_values",
        "time_slices", "matrix_definition", "matrix_npz_sha256", "time_units",
        "block_unit",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise ValueError(f"manifest keys must be exactly {sorted(required)}")
    if document["schema"] != SCHEMA or document["classification"] != CLASSIFICATION:
        raise ValueError("manifest schema/classification is not the raw-matrix candidate format")
    if document["split_id"] != split_id:
        raise ValueError("snapshot split_id differs from the requested split")
    lowered = split_id.lower()
    if any(marker in lowered for marker in _CLOSED_MARKERS):
        raise ValueError("closed gap/validation/holdout/sealed/confirmation split rejected")
    if "calibration" not in lowered:
        raise ValueError("an explicitly named calibration split is required")
    geometry = document["geometry"]
    if (not isinstance(geometry, list) or len(geometry) != 4
            or any(type(item) is not int or item <= 0 for item in geometry)):
        raise ValueError("geometry must be four positive integer extents")
    if (not isinstance(document["action"], str) or not document["action"].strip()
            or not isinstance(document["boundary_conditions"], str)
            or not document["boundary_conditions"].strip()):
        raise ValueError("action and boundary_conditions must be explicit nonempty strings")
    if (type(document["temporal_extent"]) is not int
            or document["temporal_extent"] != geometry[3]
            or document["temporal_extent"] < max(TIME_SLICES)):
        raise ValueError("temporal_extent must match geometry[3] and cover all requested time slices")
    if document["operator_basis_ids"] != list(OPERATOR_BASIS_IDS):
        raise ValueError("operator_basis_ids must match the frozen ordered three-operator basis")
    if document["R_values"] != list(R_VALUES) or document["time_slices"] != list(TIME_SLICES):
        raise ValueError("R/time axes differ from the reader contract")
    expected_definition = (
        "C_ij(R,t)=<O_i(R,t) O_j^dagger(R,0)>; rows are sink operator i, "
        "columns are source operator j; all cross terms must be measured"
    )
    if document["matrix_definition"] != expected_definition:
        raise ValueError("matrix_definition must state the ordered cross-correlator convention")
    if document["matrix_npz_sha256"] != npz_sha256:
        raise ValueError("manifest does not bind the captured NPZ bytes")
    if document["time_units"] != "lattice_time_slices" or document["block_unit"] != "configuration_block":
        raise ValueError("time_units or block_unit are not supported")
    return document


def read_raw_matrix_snapshot(
    snapshot: SourceByteSnapshot,
    *,
    source_names: dict[str, str],
    split_id: str,
) -> dict[str, Any]:
    """Read one manifest+NPZ transaction from exactly the captured bytes."""
    if not isinstance(snapshot, SourceByteSnapshot):
        raise ValueError("snapshot must be a SourceByteSnapshot")
    if not isinstance(split_id, str) or not split_id.strip():
        raise ValueError("split_id must be a nonempty string")
    if not isinstance(source_names, dict) or set(source_names) != SOURCE_ROLES:
        raise ValueError("source_names must map exactly manifest and matrix_npz roles")
    if any(not isinstance(name, str) or not name.strip() for name in source_names.values()):
        raise ValueError("source_names values must be nonempty snapshot source names")
    if len(set(source_names.values())) != 2:
        raise ValueError("manifest and matrix_npz must be distinct snapshot sources")
    payloads = {role: snapshot.read_bytes(name) for role, name in source_names.items()}
    for role, name in source_names.items():
        expected = snapshot.expected_hashes.get(name)
        if expected is None or _sha_bytes(payloads[role]) != expected:
            raise ValueError(f"{role} bytes do not match the captured expected digest")

    npz_bytes = payloads["matrix_npz"]
    validate_npz_payload(npz_bytes)
    try:
        with np.load(BytesIO(npz_bytes), allow_pickle=False) as archive:
            if set(archive.files) != NPZ_KEYS:
                raise ValueError(f"NPZ keys must be exactly {sorted(NPZ_KEYS)}")
            arrays = {name: archive[name].copy() for name in archive.files}
    except (OSError, ValueError, TypeError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("NPZ keys"):
            raise
        raise ValueError("matrix NPZ is unreadable or contains an invalid array") from exc

    manifest = _validate_manifest(
        _parse_json(payloads["manifest"]), split_id=split_id,
        npz_sha256=_sha_bytes(npz_bytes),
    )
    if arrays["split_id"].shape != () or str(arrays["split_id"].item()) != split_id:
        raise ValueError("NPZ split_id differs from the requested split")
    block_ids = arrays["block_ids"]
    if (block_ids.ndim != 1 or block_ids.dtype.kind not in "iu" or block_ids.size < 2
            or np.unique(block_ids).size != block_ids.size):
        raise ValueError("block_ids must be a unique integer vector with at least two blocks")
    if arrays["operator_basis_ids"].tolist() != list(OPERATOR_BASIS_IDS):
        raise ValueError("NPZ operator axis is missing, reordered, or unexpected")
    if (arrays["R_values"].dtype.kind not in "iu"
            or not np.array_equal(arrays["R_values"], np.asarray(R_VALUES, dtype=np.int64))):
        raise ValueError("NPZ R axis is missing, reordered, or unexpected")
    if (arrays["time_slices"].dtype.kind not in "iu"
            or not np.array_equal(arrays["time_slices"], np.asarray(TIME_SLICES, dtype=np.int64))):
        raise ValueError("NPZ time axis is missing, reordered, or unexpected")
    matrices = arrays["correlator_matrices"]
    expected_shape = (block_ids.size, len(R_VALUES), len(TIME_SLICES), 3, 3)
    if matrices.shape != expected_shape or matrices.dtype.kind != "c":
        raise ValueError(f"correlator_matrices must be complex with shape {expected_shape}")
    if not np.all(np.isfinite(matrices)):
        raise ValueError("correlator_matrices must be finite")
    if not np.allclose(matrices, matrices.swapaxes(-1, -2).conj(), rtol=1e-10, atol=1e-12):
        raise ValueError("correlator_matrices must be Hermitian; no silent symmetrization")

    projected = np.empty((block_ids.size, len(R_VALUES), len(TIME_SLICES)), dtype=float)
    records: list[dict[str, Any]] = []
    matrix_condition: dict[str, Any] = {}
    t0_index = TIME_SLICES.index(T0)
    td_index = TIME_SLICES.index(TD)
    for radius_index, radius in enumerate(R_VALUES):
        cell = matrices[:, radius_index]
        mean = np.mean(cell, axis=0)
        c_t0 = mean[t0_index]
        c_td = mean[td_index]
        if np.linalg.eigvalsh(c_t0)[0] <= 0 or np.linalg.eigvalsh(c_td)[0] <= 0:
            raise ValueError(f"mean C(t0)/C(td) is not positive definite: R={radius}")
        cond_t0 = float(np.linalg.cond(c_t0))
        cond_td = float(np.linalg.cond(c_td))
        if not (math.isfinite(cond_t0) and math.isfinite(cond_td)
                and max(cond_t0, cond_td) < 1e12):
            raise ValueError(f"mean C(t0)/C(td) is numerically singular: R={radius}")
        vector, principal_value = principal_gevp_vector(c_td, c_t0)
        values = np.einsum("i,btij,j->bt", vector.conj(), cell, vector, optimize=True)
        scale = np.maximum(1.0, np.max(np.abs(values.real), axis=1, keepdims=True))
        if np.any(np.abs(values.imag) > 1e-10 * scale):
            raise ValueError("projected correlator has a non-negligible imaginary component")
        projected[:, radius_index] = values.real
        key = f"R{radius}"
        matrix_condition[key] = {
            "condition_C_t0": cond_t0,
            "condition_C_td": cond_td,
            "principal_generalized_eigenvalue": principal_value,
            "eigenvector": [[float(v.real), float(v.imag)] for v in vector],
            "metric_norm": float(np.real(vector.conj() @ c_t0 @ vector)),
        }
        for time_index, time in enumerate(TIME_SLICES):
            block_values = projected[:, radius_index, time_index]
            records.append({
                "R": radius, "time": time,
                "mean_projected_C": float(np.mean(block_values)),
                "sd_across_synthetic_blocks_descriptive_only": float(np.std(block_values, ddof=1)),
                "p10_across_synthetic_blocks_descriptive_only": float(np.quantile(block_values, 0.10)),
                "p90_across_synthetic_blocks_descriptive_only": float(np.quantile(block_values, 0.90)),
                "n_blocks": int(block_ids.size),
            })

    return {
        "schema": "wilson-f-raw-matrix-gevp-projection-read-v1",
        "status": "RAW_MATRIX_GEVP_PROJECTION_ONLY",
        "certified": False,
        "physical_score": None,
        "split_id": split_id,
        "block_ids": block_ids.copy(),
        "operator_basis_ids": tuple(str(value) for value in arrays["operator_basis_ids"].tolist()),
        "R_values": tuple(int(value) for value in arrays["R_values"].tolist()),
        "time_slices": tuple(int(value) for value in arrays["time_slices"].tolist()),
        "correlator_matrices": matrices.copy(),
        "projected_correlators": projected,
        "records": records,
        "matrix_condition": matrix_condition,
        "source_hashes": {role: _sha_bytes(payloads[role]) for role in sorted(payloads)},
        "denominators": {
            "source_files": {"planned": 2, "valid": 2, "failed": 0, "abstained": 0},
            "R_projection_cells": {
                "planned": len(R_VALUES),
                "valid": len(R_VALUES), "failed": 0, "abstained": 0,
            },
            "R_time_values": {
                "planned": len(R_VALUES) * len(TIME_SLICES),
                "valid": len(R_VALUES) * len(TIME_SLICES),
                "failed": 0, "abstained": 0,
            },
            "physical_q_interval_or_certification": {
                "planned": 1, "valid": 0, "failed": 0, "abstained": 1,
            },
        },
    }


def _synthetic_matrices(n_blocks: int = 24) -> np.ndarray:
    result = np.empty((n_blocks, len(R_VALUES), len(TIME_SLICES), 3, 3), dtype=np.complex128)
    for block in range(n_blocks):
        for ri, _radius in enumerate(R_VALUES):
            overlap = OVERLAPS_4X3.copy()
            for state in range(overlap.shape[0]):
                for operator in range(overlap.shape[1]):
                    overlap[state, operator] *= (
                        1 + 0.035 * math.sin((block + 1) * (operator + 1) * 0.31 + state)
                        + 0.006 * ri
                    )
            energy_scale = 1 + 0.006 * (((block * 7) % 11) - 5) + 0.004 * ri
            energies = TOY_ENERGIES_4 * energy_scale
            for ti, time in enumerate(TIME_SLICES):
                result[block, ri, ti] = correlation_matrix(energies, overlap, time)
    return result


def _write_synthetic_input(output_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    npz_path = output_dir / "synthetic_raw_matrix_input.npz"
    manifest_path = output_dir / "synthetic_raw_matrix_manifest.json"
    npz_stream = BytesIO()
    np.savez_compressed(
        npz_stream,
        block_ids=np.arange(24, dtype=np.int64),
        correlator_matrices=_synthetic_matrices(),
        operator_basis_ids=np.asarray(OPERATOR_BASIS_IDS, dtype="U32"),
        R_values=np.asarray(R_VALUES, dtype=np.int64),
        time_slices=np.asarray(TIME_SLICES, dtype=np.int64),
        split_id=np.asarray("synthetic_calibration_open"),
    )
    npz_bytes = npz_stream.getvalue()
    npz_path.write_bytes(npz_bytes)
    manifest = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "split_id": "synthetic_calibration_open",
        "geometry": [16, 16, 16, 16],
        "action": "exact_four_state_forward_spectral_fixture",
        "boundary_conditions": "periodic_all_directions_in_fixture",
        "temporal_extent": 16,
        "operator_basis_ids": list(OPERATOR_BASIS_IDS),
        "R_values": list(R_VALUES),
        "time_slices": list(TIME_SLICES),
        "matrix_definition": (
            "C_ij(R,t)=<O_i(R,t) O_j^dagger(R,0)>; rows are sink operator i, "
            "columns are source operator j; all cross terms must be measured"
        ),
        "matrix_npz_sha256": _sha_bytes(npz_bytes),
        "time_units": "lattice_time_slices",
        "block_unit": "configuration_block",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    roles = {"manifest": "synthetic_manifest", "matrix_npz": "synthetic_matrix_npz"}
    hashes = {
        roles["manifest"]: _sha_bytes(manifest_path.read_bytes()),
        roles["matrix_npz"]: _sha_bytes(npz_path.read_bytes()),
    }
    paths = {roles["manifest"]: manifest_path.name, roles["matrix_npz"]: npz_path.name}
    return hashes, paths


def _plot(path: Path, result: dict[str, Any]) -> None:
    projected = result["projected_correlators"]
    fig, axes = plt.subplots(1, len(R_VALUES), figsize=(11, 4.5), sharex=True)
    for ri, radius in enumerate(R_VALUES):
        ax = axes[ri]
        data = projected[:, ri, :]
        normalized = data / data[:, :1]
        mean = normalized.mean(axis=0)
        low = np.quantile(normalized, 0.10, axis=0)
        high = np.quantile(normalized, 0.90, axis=0)
        ax.plot(TIME_SLICES, mean, marker="o", color="#2878a5")
        ax.fill_between(TIME_SLICES, low, high, color="#78b7c5", alpha=0.30,
                        label="10–90% synthetic block spread (descriptive)")
        info = result["matrix_condition"][f"R{radius}"]
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(f"R={radius}; 3-operator basis\ncond C(t0)={info['condition_C_t0']:.1f}; cond C(td)={info['condition_C_td']:.1f}")
        ax.set_xlabel("time slice t")
        ax.set_ylabel(r"$v^\dagger C(t)v / v^\dagger C(t_0)v$")
        if ri == 0:
            ax.legend(fontsize=8)
    fig.suptitle("F-lane raw-matrix → fixed GEVP projection (synthetic schema diagnostic)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def build(output_dir: Path = DEFAULT_OUT) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    hashes, paths = _write_synthetic_input(output_dir)
    snapshot = capture_source_snapshot(output_dir, hashes, paths)
    result = read_raw_matrix_snapshot(
        snapshot,
        source_names={"manifest": "synthetic_manifest", "matrix_npz": "synthetic_matrix_npz"},
        split_id="synthetic_calibration_open",
    )
    with (output_dir / "projected_correlators.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["records"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["records"])
    _plot(output_dir / "projected_correlators.png", result)

    reader_path = Path(__file__)
    npz_safety_path = reader_path.parents[2] / "core" / "npz_safety.py"
    source_integrity_path = reader_path.parents[2] / "core" / "source_integrity.py"
    source_copies = {
        "snapshot_f_lane_raw_matrix_reader.py": reader_path,
        "snapshot_core_npz_safety.py": npz_safety_path,
        "snapshot_core_source_integrity.py": source_integrity_path,
    }
    source_hashes = {}
    for name, source in source_copies.items():
        shutil.copyfile(source, output_dir / name)
        source_hashes[name] = _sha_bytes((output_dir / name).read_bytes())

    protocol = {
        "schema": "wilson-f-raw-matrix-snapshot-diagnostic-protocol-v1",
        "classification": "SYNTHETIC_SCHEMA_AND_LINEAR_ALGEBRA_DIAGNOSTIC_NOT_MEASUREMENT",
        "input": {
            "split_id": result["split_id"],
            "matrix_manifest_sha256": result["source_hashes"]["manifest"],
            "matrix_npz_sha256": result["source_hashes"]["matrix_npz"],
            "read_mode": "immutable SourceByteSnapshot; NPZ size preflight before numpy.load",
            "blocks": 24,
            "operator_basis_ids": list(OPERATOR_BASIS_IDS),
            "R_values": list(R_VALUES),
            "time_slices": list(TIME_SLICES),
        },
        "method": {
            "operator_basis_ids": list(OPERATOR_BASIS_IDS),
            "operator_matrix": "one C_ij(R,t)=<O_i(R,t) O_j^dagger(R,0)> per R; i is sink and j is source",
            "must_measure_cross_terms": True,
            "t0": T0,
            "td": TD,
            "gevp": "C(td)v=lambda C(t0)v; largest eigenvalue vector, C(t0)-normalized",
            "projection": "v^dagger C_block(R,t) v using one ensemble-mean vector per R cell",
            "output": "one projected composite operator per R plus condition diagnostics; no per-basis q, fit, or interval",
            "q_semantics": "Kernel-indexed q cells use the squared excited/ground overlap ratios of their mapped bare basis operator, obtained from a joint fit to C_ij; this GEVP projection is a separate composite-operator diagnostic",
        },
        "toy_model": {
            "spectrum": np.asarray(TOY_ENERGIES_4, dtype=float).tolist(),
            "overlaps_state_by_operator": np.asarray(OVERLAPS_4X3, dtype=float).tolist(),
            "temporal_extent": "finite metadata, but synthetic input contains forward terms only",
            "variation": "deterministic small block-dependent overlap/energy perturbations; no stochastic noise model",
        },
        "analysis_source_sha256": source_hashes,
        "limitations": [
            "synthetic input only; no Wilson measurement or production source bytes were read",
            "the repository has no raw cross-correlator producer; diagonal loop arrays cannot reconstruct missing off-diagonal elements",
            "the matrix basis uses three ordered smearing IDs; do not add an outer kernel axis or interpret one projected vector as three per-kernel q values",
            "the selected method default maps identity/APE/stout kernel IDs to their bare basis operators for q; independent review and production matrix-fit evidence remain open",
            "synthetic block spread is descriptive and is not a confidence interval or independence claim",
            "captured-byte digest equality does not authenticate who issued the expected digests",
            "no state-overlap/gap fit, q upper bound, coverage calibration, finite-Nt backward term, thermal h, mixed x, force propagation, or physical gate",
        ],
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema": "wilson-f-raw-matrix-snapshot-diagnostic-v1",
        "status": result["status"],
        "classification": protocol["classification"],
        "certified": False,
        "physical_score": None,
        "lane_status": "GROUND_STATE_FORCE_UNCERTIFIED",
        "source_hashes": result["source_hashes"],
        "denominators": result["denominators"],
        "physical_measurement_inputs": {
            "unit": "raw Wilson correlator or independent spectrum receipt",
            "planned": 0, "valid": 0, "failed": 0, "abstained": 0,
        },
        "matrix_condition": result["matrix_condition"],
        "limits": protocol["limitations"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme = f"""# F-lane raw matrix snapshot reader: synthetic diagnostic

This package validates a two-file immutable source snapshot (`manifest` + NPZ), checks NPZ size before allocation, and enforces one 3×3 cross-correlator matrix per `R`. The ordered matrix basis is `identity`, `ape_alpha0.4_n5`, `spatial_stout_rho0.10_n5`; the data shape is `[block, R, t, sink_operator, source_operator]`. It solves one fixed principal GEVP vector at `t0={T0}`, `td={TD}` per R and projects `v† C_block(R,t) v`.

![Projected correlator diagnostic](projected_correlators.png)

The panels show the two R values. Bands are the 10–90% spread across 24 deterministic synthetic blocks; they are descriptive, not confidence intervals and not evidence that blocks are independent. `physical_score=null`; the F lane remains `GROUND_STATE_FORCE_UNCERTIFIED`.

## Data-flow and limits

`captured manifest + NPZ bytes → strict schema/hash/split/axis validation → NPZ preflight → mean C(t0), C(td) per R → fixed principal GEVP → block projection → CSV/figure`.

The GEVP projection is one composite operator per R. It must not be copied into three separate identity/APE/stout q cells. The selected method default maps the kernel-indexed q cells to the corresponding bare basis operators; a future joint fit of `C_ij` supplies each operator's squared excited/ground overlap ratios. The GEVP remains a state-identification and sensitivity diagnostic, not the q object. This mapping is a proposal requiring independent review. No fit of energies or overlaps, q envelope, interval, coverage calibration, finite-`N_t` backward propagation, thermal `h`, mixed `x`, force propagation, or physical gate is implemented. The repository also lacks a raw cross-correlator producer; diagonal loop arrays do not determine off-diagonal matrix elements.

Denominators (planned/valid/failed/abstained): source files `{result['denominators']['source_files']['planned']}/{result['denominators']['source_files']['valid']}/{result['denominators']['source_files']['failed']}/{result['denominators']['source_files']['abstained']}`; R projection cells `{result['denominators']['R_projection_cells']['planned']}/{result['denominators']['R_projection_cells']['valid']}/{result['denominators']['R_projection_cells']['failed']}/{result['denominators']['R_projection_cells']['abstained']}`; R×time values `{result['denominators']['R_time_values']['planned']}/{result['denominators']['R_time_values']['valid']}/{result['denominators']['R_time_values']['failed']}/{result['denominators']['R_time_values']['abstained']}`; physical measurement inputs `0/0/0/0`; physical q/interval/certification `{result['denominators']['physical_q_interval_or_certification']['planned']}/{result['denominators']['physical_q_interval_or_certification']['valid']}/{result['denominators']['physical_q_interval_or_certification']['failed']}/{result['denominators']['physical_q_interval_or_certification']['abstained']}`.

See `protocol.json`, `summary.json`, and `artifact_hashes.json` for exact source identities and scope.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    artifact_hashes = {
        path.name: _sha_bytes(path.read_bytes())
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps({"schema": "latticearena-artifact-hashes-v1", "files": artifact_hashes}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
