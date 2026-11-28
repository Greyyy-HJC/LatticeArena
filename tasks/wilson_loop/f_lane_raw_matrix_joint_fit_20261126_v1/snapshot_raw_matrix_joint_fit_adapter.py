"""Connect the immutable raw-matrix reader to a candidate joint GLS fit.

This is a CPU-only diagnostic. It emits per-fit q estimates with uncalibrated
delta-method standard errors, but does not build the selected support envelope
or connect to the F evidence gate.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from io import BytesIO
from pathlib import Path
import shutil
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.source_integrity import SourceByteSnapshot, capture_source_snapshot
from . import f_lane_joint_spectral_gls_diagnostic as joint
from . import f_lane_raw_matrix_snapshot_diagnostic as raw_reader
from .spectral_q_bound_reader import (
    CONTAMINATION_TIMES, FIT_WINDOW, KERNELS, KERNEL_TO_BASIS_OPERATOR,
    R_VALUES, SELECTED_FORCE_KERNELS, STATE_COUNTS, TMIN_VALUES,
)


DEFAULT_OUT = Path(__file__).resolve().parent / "f_lane_raw_matrix_joint_fit_20261126_v1"
SPLIT_ID = "synthetic_calibration_open"
SYNTHETIC_BLOCKS = 80
SYNTHETIC_NOISE = 0.02
SELECTED_Q_OPERATORS = ("identity", "spatial_stout_rho0.10_n5")


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def _block_vectors(matrices: np.ndarray, *, radius_index: int,
                   time_indices: list[int]) -> np.ndarray:
    return np.stack([
        np.concatenate([
            joint._matrix_vector(matrices[block, radius_index, time_index])
            for time_index in time_indices
        ])
        for block in range(matrices.shape[0])
    ])


def analyze_snapshot(snapshot: SourceByteSnapshot, *,
                     source_names: dict[str, str], split_id: str) -> dict[str, Any]:
    """Fit selected 2/3-state, tmin=3/4/5 variants from captured matrix bytes."""
    source = raw_reader.read_raw_matrix_snapshot(
        snapshot, source_names=source_names, split_id=split_id,
    )
    if (tuple(source["operator_basis_ids"]) != tuple(KERNELS)
            or tuple(source["R_values"]) != tuple(R_VALUES)):
        raise ValueError("raw matrix basis/R axes differ from the selected q-fit contract")
    matrices = source["correlator_matrices"]
    times = tuple(source["time_slices"])
    variant_rows: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []
    fit_status_rows: list[dict[str, Any]] = []

    for radius_index, radius in enumerate(R_VALUES):
        for tmin in TMIN_VALUES:
            selected_times = tuple(time for time in times
                                   if tmin <= time <= FIT_WINDOW[1])
            time_indices = [times.index(time) for time in selected_times]
            blocks = _block_vectors(
                matrices, radius_index=radius_index, time_indices=time_indices,
            )
            for n_states in STATE_COUNTS:
                fit_id = f"R{radius}_tmin{tmin}_n{n_states}"
                planned_q_per_fit = len(KERNELS) * len(CONTAMINATION_TIMES)
                fit: dict[str, Any] | None = None
                if blocks.shape[0] <= 9 * len(selected_times):
                    fit_status = "abstained"
                    reason = (f"need more than {9 * len(selected_times)} blocks for the full raw covariance; "
                              f"received {blocks.shape[0]}")
                else:
                    try:
                        fit = joint.fit_joint_spectrum(
                            blocks,
                            time_slices=selected_times,
                            n_states=n_states,
                            operator_indices=tuple(range(len(KERNELS))),
                        )
                        fit_status = "valid" if fit["success"] else "failed"
                        reason = "" if fit["success"] else str(fit["message"])
                    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                        fit_status = "failed"
                        reason = str(exc)
                fit_status_rows.append({
                    "fit_id": fit_id, "R": radius, "tmin": tmin,
                    "n_states": n_states, "time_slices": list(selected_times),
                    "status": fit_status, "reason": reason,
                    "n_blocks": int(blocks.shape[0]),
                    "n_data_components": int(blocks.shape[1]),
                    "nfev": None if fit is None else fit.get("nfev"),
                    "rank": None if fit is None else fit.get("rank"),
                    "parameter_count": None if fit is None else fit.get("parameter_count"),
                    "dof": None if fit is None else fit.get("dof"),
                    "reduced_chi_square": None if fit is None else fit.get("reduced_chi_square"),
                    "jacobian_condition_number": (None if fit is None
                                                   else fit.get("jacobian_condition_number")),
                })
                fit_q = {} if fit is None else {
                    (row["operator_index"], row["time"]): row
                    for row in fit.get("q_cells", [])
                }
                for kernel in KERNELS:
                    operator_id = KERNEL_TO_BASIS_OPERATOR[kernel]
                    raw_operator_id = kernel
                    operator_index = source["operator_basis_ids"].index(raw_operator_id)
                    variant_rows.append({
                        "variant_id": f"{kernel}|{fit_id}",
                        "kernel_id": kernel,
                        "basis_operator_id": operator_id,
                        "raw_operator_basis_id": raw_operator_id,
                        "operator_index": operator_index,
                        "R": radius, "tmin": tmin, "n_states": n_states,
                        "fit_id": fit_id, "status": fit_status,
                        "fit_method": "joint_full_matrix_GLS_candidate",
                        "time_slices": list(selected_times),
                        "reason": reason,
                    })
                    for time in CONTAMINATION_TIMES:
                        cell = fit_q.get((operator_index, time))
                        cell_status = (fit_status if fit_status != "valid" else
                                       ("valid" if cell is not None else "abstained"))
                        cell_reason = (reason if fit_status != "valid" else
                                       ("joint fit did not emit this operator/time q cell"
                                        if cell is None else ""))
                        q_rows.append({
                            "variant_id": f"{kernel}|{fit_id}",
                            "kernel_id": kernel,
                            "basis_operator_id": operator_id,
                            "raw_operator_basis_id": raw_operator_id,
                            "operator_index": operator_index,
                            "R": radius, "T": time, "tmin": tmin,
                            "n_states": n_states, "fit_id": fit_id,
                            "status": cell_status,
                            "q_estimate": None if cell is None else cell["q_estimate"],
                            "q_delta_se_uncalibrated": (None if cell is None
                                                         else cell["q_standard_error_delta"]),
                            "q_upper_97_5_delta_uncalibrated": (
                                None if cell is None else cell["q_upper_97_5_pointwise"]
                            ),
                            "interval_status": "UNCALIBRATED_DELTA_METHOD",
                            "reason": cell_reason,
                        })

    n_fits = len(fit_status_rows)
    n_fit_valid = sum(row["status"] == "valid" for row in fit_status_rows)
    n_fit_failed = sum(row["status"] == "failed" for row in fit_status_rows)
    n_fit_abstained = sum(row["status"] == "abstained" for row in fit_status_rows)
    variant_valid = sum(row["status"] == "valid" for row in variant_rows)
    variant_failed = sum(row["status"] == "failed" for row in variant_rows)
    variant_abstained = sum(row["status"] == "abstained" for row in variant_rows)
    q_valid = sum(row["status"] == "valid" for row in q_rows)
    q_failed = sum(row["status"] == "failed" for row in q_rows)
    q_abstained = sum(row["status"] == "abstained" for row in q_rows)
    denominators = {
        "source_files": source["denominators"]["source_files"],
        "joint_matrix_fits": {
            "planned": n_fits, "valid": n_fit_valid,
            "failed": n_fit_failed, "abstained": n_fit_abstained,
        },
        "kernel_fit_variants": {
            "planned": len(variant_rows), "valid": variant_valid,
            "failed": variant_failed, "abstained": variant_abstained,
        },
        "q_candidate_cells": {
            "planned": len(q_rows), "valid": q_valid,
            "failed": q_failed, "abstained": q_abstained,
        },
        "physical_certification": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
    }
    return {
        "schema": "wilson-f-raw-matrix-joint-fit-candidate-v1",
        "status": "RAW_MATRIX_JOINT_FIT_CANDIDATE_ONLY",
        "certified": False,
        "physical_score": None,
        "split_id": split_id,
        "source_hashes": source["source_hashes"],
        "operator_basis_ids": list(source["operator_basis_ids"]),
        "R_values": list(source["R_values"]),
        "time_slices": list(source["time_slices"]),
        "fit_status_rows": fit_status_rows,
        "fit_variants": variant_rows,
        "q_candidates": q_rows,
        "denominators": denominators,
        "not_connected_to_spectral_q_reader": True,
        "not_connected_to_evidence_gate": True,
    }


def _synthetic_matrices(n_blocks: int = SYNTHETIC_BLOCKS) -> np.ndarray:
    matrices = np.empty(
        (n_blocks, len(R_VALUES), len(raw_reader.TIME_SLICES), 3, 3),
        dtype=np.complex128,
    )
    for radius_index in range(len(R_VALUES)):
        vectors = joint._design_covariance(
            2026112600 + 10000 * radius_index, n_blocks, SYNTHETIC_NOISE,
        )
        for block, vector in enumerate(vectors):
            matrices[block, radius_index] = joint._vector_matrices(
                vector, times=raw_reader.TIME_SLICES,
            )
    return matrices


def _write_synthetic_snapshot(output: Path, n_blocks: int = SYNTHETIC_BLOCKS):
    arrays = {
        "block_ids": np.arange(n_blocks, dtype=np.int64),
        "correlator_matrices": _synthetic_matrices(n_blocks),
        "operator_basis_ids": np.asarray(raw_reader.OPERATOR_BASIS_IDS, dtype="U32"),
        "R_values": np.asarray(R_VALUES, dtype=np.int64),
        "time_slices": np.asarray(raw_reader.TIME_SLICES, dtype=np.int64),
        "split_id": np.asarray(SPLIT_ID),
    }
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    npz_bytes = stream.getvalue()
    definition = (
        "C_ij(R,t)=<O_i(R,t) O_j^dagger(R,0)>; rows are sink operator i, "
        "columns are source operator j; all cross terms must be measured"
    )
    manifest = {
        "schema": raw_reader.SCHEMA,
        "classification": raw_reader.CLASSIFICATION,
        "split_id": SPLIT_ID,
        "geometry": [8, 8, 8, 16],
        "action": "synthetic_exact_three_state_forward_fixture",
        "boundary_conditions": "periodic_fixture",
        "temporal_extent": 16,
        "operator_basis_ids": list(raw_reader.OPERATOR_BASIS_IDS),
        "R_values": list(R_VALUES),
        "time_slices": list(raw_reader.TIME_SLICES),
        "matrix_definition": definition,
        "matrix_npz_sha256": _sha(npz_bytes),
        "time_units": "lattice_time_slices",
        "block_unit": "configuration_block",
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                      + "\n").encode()
    npz_path, manifest_path = output / "synthetic_raw_matrix_input.npz", output / "synthetic_raw_matrix_manifest.json"
    npz_path.write_bytes(npz_bytes)
    manifest_path.write_bytes(manifest_bytes)
    expected = {
        "synthetic_manifest": _sha(manifest_bytes),
        "synthetic_matrix_npz": _sha(npz_bytes),
    }
    paths = {
        "synthetic_manifest": manifest_path.name,
        "synthetic_matrix_npz": npz_path.name,
    }
    return capture_source_snapshot(output, expected, paths), expected, paths


def _plot(path: Path, result: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), layout="constrained")
    selected = [row for row in result["q_candidates"]
                if row["kernel_id"] in SELECTED_Q_OPERATORS]
    colors = {2: "#0072B2", 3: "#D55E00"}
    for ax, (kernel, radius, time) in zip(
        axes.flat,
        [(kernel, radius, time) for kernel in SELECTED_Q_OPERATORS
         for radius in R_VALUES for time in CONTAMINATION_TIMES],
    ):
        cell_rows = [row for row in selected
                     if row["kernel_id"] == kernel and row["R"] == radius
                     and row["T"] == time]
        for n_states in STATE_COUNTS:
            rows = sorted([row for row in cell_rows if row["n_states"] == n_states],
                          key=lambda row: row["tmin"])
            valid = [row for row in rows if row["status"] == "valid"]
            if not valid:
                continue
            xs = np.asarray([row["tmin"] for row in valid], dtype=float)
            ys = np.asarray([row["q_estimate"] for row in valid], dtype=float)
            errors = np.asarray([row["q_delta_se_uncalibrated"] for row in valid], dtype=float)
            ax.errorbar(xs, ys, yerr=errors, color=colors[n_states],
                        marker="o" if n_states == 2 else "s", capsize=2,
                        label=f"n_states={n_states}")
        if not any(row["status"] == "valid" for row in cell_rows):
            ax.text(0.5, 0.5, "no valid fit", transform=ax.transAxes,
                    ha="center", fontsize=9)
        ax.set_title(f"{kernel}, R={radius}, T={time}")
        ax.set_xlabel("fit tmin")
        ax.set_xticks(TMIN_VALUES)
        ax.set_ylabel("q estimate ± uncalibrated delta SE")
        ax.grid(alpha=0.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    fig.suptitle("Candidate raw-matrix fit variants; intervals are not calibrated")
    fig.savefig(path, dpi=165)
    plt.close(fig)


def build(output: str | Path = DEFAULT_OUT, *, n_blocks: int = SYNTHETIC_BLOCKS) -> dict[str, Any]:
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    if isinstance(n_blocks, bool) or not isinstance(n_blocks, int) or n_blocks < 8:
        raise ValueError("n_blocks must be an integer of at least eight for a raw-matrix fixture")
    out.mkdir(parents=True)
    snapshot, input_hashes, input_paths = _write_synthetic_snapshot(out, n_blocks=n_blocks)
    result = analyze_snapshot(
        snapshot,
        source_names={"manifest": "synthetic_manifest", "matrix_npz": "synthetic_matrix_npz"},
        split_id=SPLIT_ID,
    )
    with (out / "fit_status.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["fit_status_rows"][0]),
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(result["fit_status_rows"])
    with (out / "fit_variants.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["fit_variants"][0]),
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(result["fit_variants"])
    with (out / "q_candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["q_candidates"][0]),
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(result["q_candidates"])
    _plot(out / "q_fit_variants.png", result)

    repo = Path(__file__).resolve().parents[2]
    source_files = {
        "snapshot_raw_matrix_joint_fit_adapter.py": Path(__file__),
        "snapshot_raw_matrix_reader.py": repo / "tasks/wilson_loop/f_lane_raw_matrix_snapshot_diagnostic.py",
        "snapshot_joint_spectral_fit.py": repo / "tasks/wilson_loop/f_lane_joint_spectral_gls_diagnostic.py",
        "snapshot_spectral_q_reader.py": repo / "tasks/wilson_loop/spectral_q_bound_reader.py",
        "snapshot_lane_contract.json": repo / "tasks/wilson_loop/lane_contract_v2.json",
        "snapshot_source_integrity.py": repo / "core/source_integrity.py",
        "snapshot_npz_safety.py": repo / "core/npz_safety.py",
    }
    source_hashes = {}
    for name, source in source_files.items():
        shutil.copyfile(source, out / name)
        source_hashes[name] = _sha((out / name).read_bytes())

    fit_counts = result["denominators"]["joint_matrix_fits"]
    variant_counts = result["denominators"]["kernel_fit_variants"]
    q_counts = result["denominators"]["q_candidate_cells"]
    protocol = {
        "schema": "wilson-f-raw-matrix-joint-fit-diagnostic-protocol-v1",
        "classification": "SYNTHETIC_RAW_MATRIX_FIT_CANDIDATE_NOT_CALIBRATED_OR_CERTIFICATION",
        "source_snapshot": {
            "split_id": SPLIT_ID,
            "source_sha256": result["source_hashes"],
            "paths": input_paths,
            "expected_sha256": input_hashes,
            "block_count": n_blocks,
            "read_mode": "same captured SourceByteSnapshot passed to strict matrix reader and joint fitter",
        },
        "fit_contract": {
            "operator_basis_ids": list(KERNELS),
            "kernel_to_basis_operator": KERNEL_TO_BASIS_OPERATOR,
            "selected_q_kernels": list(SELECTED_FORCE_KERNELS),
            "R_values": list(R_VALUES),
            "tmin_values": list(TMIN_VALUES),
            "fit_window_end": FIT_WINDOW[1],
            "state_counts": list(STATE_COUNTS),
            "times_for_q": list(CONTAMINATION_TIMES),
            "covariance": f"same-block full covariance on 9*Nt real/imag matrix components; {joint.SHRINKAGE:.2f} diagonal shrinkage and {joint.EIGENVALUE_FLOOR_RELATIVE:g} eigen floor",
            "q_point_upper": "qhat + 1.95996398454 * delta-method SE; uncalibrated candidate only",
        },
        "method_boundary": {
            "not_connected_to_spectral_q_reader": True,
            "not_connected_to_evidence_gate": True,
            "reason": "candidate q intervals are not coverage-calibrated and do not emit ratio_upper/gap_lower support receipts or selected familywise envelope",
            "physical_score": None,
        },
        "analysis_source_sha256": source_hashes,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")
    summary = {
        "schema": result["schema"],
        "status": result["status"],
        "classification": protocol["classification"],
        "certified": False,
        "physical_score": None,
        "lane_status": "GROUND_STATE_FORCE_UNCERTIFIED",
        "source_hashes": result["source_hashes"],
        "denominators": result["denominators"],
        "fit_counts": {"joint_matrix_fits": fit_counts,
                       "kernel_fit_variants": variant_counts,
                       "q_candidate_cells": q_counts},
        "not_connected_to_spectral_q_reader": True,
        "not_connected_to_evidence_gate": True,
        "q_candidate_interval_status": "UNCALIBRATED_DELTA_METHOD",
        "physical_measurement_inputs": {"planned": 0, "valid": 0, "failed": 0, "abstained": 0},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8")
    readme = f"""# Wilson F-lane raw matrix → joint fit candidate

**Synthetic CPU diagnostic only.** Input bytes are generated from an exact three-state positive forward spectrum with independent Gaussian block noise. No Wilson measurement, sealed split, GPU or QUDA data were read. `physical_score=null`; F remains `GROUND_STATE_FORCE_UNCERTIFIED`.

## Calculation and plot

The existing strict raw-matrix reader validates and parses the captured manifest+NPZ snapshot, including the ordered basis `[identity, APE, stout]`, R/time axes, Hermitian matrix shape and split/hash preflight. This wrapper consumes the returned matrices from those same captured bytes. For each `R=2,3`, it fits one shared matrix spectrum for every `tmin=3,4,5` and `n_states=2,3`, on times `tmin..7`, using all three basis operators and same-block covariance. Each joint fit yields three bare-basis q estimates at `T=5,6`; the 36 kernel/R/tmin/state variants are derived from 12 common matrix fits.

![q fit variants](q_fit_variants.png)

Each panel fixes kernel, `R`, and `T`; the horizontal axis is `tmin`, while color/marker distinguish the two state counts. Error bars are pointwise delta-method standard errors, not calibrated intervals. The separate 800-panel coverage diagnostic in [`f_lane_joint_spectral_gls_20261126_v1/README.md`](f_lane_joint_spectral_gls_20261126_v1/README.md) found undercoverage in the same exact-three-state setting; therefore these q upper candidates are explicitly **not** suitable inputs to the selected q reader or force gate.

## Denominators and data flow

Joint matrix fits planned/valid/failed/abstained = `{fit_counts['planned']}/{fit_counts['valid']}/{fit_counts['failed']}/{fit_counts['abstained']}`; kernel fit variants = `{variant_counts['planned']}/{variant_counts['valid']}/{variant_counts['failed']}/{variant_counts['abstained']}`; q candidate cells = `{q_counts['planned']}/{q_counts['valid']}/{q_counts['failed']}/{q_counts['abstained']}`. Physical matrix inputs `0/0/0/0`; certification gate `1/0/0/1`.

Text flow: `synthetic manifest+matrix NPZ -> one immutable SourceByteSnapshot -> raw reader validates and returns matrices -> 12 common 2/3-state GLS fits -> 36 kernel-fit variants + 72 q candidates -> coverage calibration fails/unknown -> no selected q envelope or physical gate`.

## Scope boundary

This is not yet a production evaluator. It does not calculate ratio/gap support intervals, apply the selected endpoint alpha allocation across all fits, form the full familywise maximum envelope, model omitted states/finite-`N_t`/thermal `h`/mixed `x`/chain dependence, or recompute force/reference summaries. The q reader schema consumes upper ratios and lower gaps; this report has only exploratory direct q estimates and delta errors. The `1.95996` multiplier is known to under-cover in the linked independent-panel stress. Independent review, a production cross-correlator exporter, authorized immutable calibration snapshot, calibrated familywise intervals and contamination receipts remain required. Preserve `NOT_READY/UNCERTIFIED`, `GROUND_STATE_FORCE_UNCERTIFIED`, and `physical_score=null`.

See `fit_status.csv`, `fit_variants.csv`, `q_candidates.csv`, `protocol.json`, `summary.json` and `artifact_hashes.json` for all fit statuses and hashes.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    artifacts = {
        path.name: _sha(path.read_bytes())
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (out / "artifact_hashes.json").write_text(
        json.dumps({"schema": "latticearena-artifact-hashes-v1", "files": artifacts},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-blocks", type=int, default=SYNTHETIC_BLOCKS)
    args = parser.parse_args()
    print(json.dumps(build(args.output, n_blocks=args.n_blocks), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
