"""Run a self-contained synthetic raw-matrix fit for the proposed F-lane trace.

The output is a CPU method pilot. It uses the currently drafted endpoint
factorization only as an explicit toy assumption and does not certify a
Wilson force or a q/h/x confidence bound.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.source_integrity import capture_source_snapshot
from tasks.wilson_loop import f_lane_finite_nt_trace_diagnostic as toy
from tasks.wilson_loop.f_lane_two_sector_fit_candidate import (
    RAW_MANIFEST_SCHEMA, RAW_SCHEMA, NT_VALUES, R_VALUES, T_VALUES,
    fit_two_sector_snapshot,
)
from tasks.wilson_loop.f_lane_transfer_trace_reader import _component_values


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
DEFAULT_OUT = TASK_DIR / "f_lane_two_sector_fit_cpu_20261127_v1"
N_BLOCKS = 96
NOISE_FRACTION = 0.003
SEED = 2026112712
SOURCE_PATHS = {
    "two_sector_fit_candidate.py": REPO_ROOT / "tasks/wilson_loop/f_lane_two_sector_fit_candidate.py",
    "joint_forward_fit.py": REPO_ROOT / "tasks/wilson_loop/f_lane_joint_spectral_gls_diagnostic.py",
    "transfer_trace_reader.py": REPO_ROOT / "tasks/wilson_loop/f_lane_transfer_trace_reader.py",
    "finite_trace_toy.py": REPO_ROOT / "tasks/wilson_loop/f_lane_finite_nt_trace_diagnostic.py",
    "raw_matrix_reader.py": REPO_ROOT / "tasks/wilson_loop/f_lane_raw_matrix_snapshot_diagnostic.py",
    "source_integrity.py": REPO_ROOT / "core/source_integrity.py",
    "lane_contract_v2.json": TASK_DIR / "lane_contract_v2.json",
    "test_two_sector_fit.py": REPO_ROOT / "tests/test_wilson_f_lane_two_sector_fit_candidate.py",
    "fit_diagnostic.py": TASK_DIR / "f_lane_two_sector_fit_diagnostic.py",
}


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _synthetic_matrices() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    matrices = np.empty((N_BLOCKS, len(NT_VALUES), len(R_VALUES), len(T_VALUES), 3, 3),
                        dtype=np.complex128)
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
                sigma = max(float(np.max(np.abs(mean))) * NOISE_FRACTION, 1e-9)
                for block in range(N_BLOCKS):
                    raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
                    noise = 0.5 * (raw + raw.conj().T)
                    noise[np.diag_indices(3)] = rng.normal(size=3)
                    matrices[block, nt_index, radius_index, time_index] = mean + sigma * noise
    return matrices


def _write_input_bundle(output: Path) -> tuple[dict[str, str], dict[str, str]]:
    matrices = _synthetic_matrices()
    stream = BytesIO()
    np.savez_compressed(
        stream, schema=np.asarray(RAW_SCHEMA), block_ids=np.arange(N_BLOCKS),
        correlator_matrices=matrices, R_values=np.asarray(R_VALUES),
        temporal_extents=np.asarray(NT_VALUES), time_slices=np.asarray(T_VALUES),
    )
    matrix_bytes = stream.getvalue()
    contract_bytes = (TASK_DIR / "lane_contract_v2.json").read_bytes()
    manifest = {
        "schema": RAW_MANIFEST_SCHEMA, "split_id": "F-synthetic-development",
        "geometry": "S16T16", "action": "wilson_gauge_beta6",
        "boundary_conditions": "periodic", "radius_order": list(R_VALUES),
        "temporal_extents": list(NT_VALUES), "time_slices": list(T_VALUES),
        "dimension_order": "block,Nt,R,T,i,j",
        "matrix_sha256": _sha(matrix_bytes),
        "lane_contract_sha256": _sha(contract_bytes),
    }
    files = {
        "manifest.json": _json_bytes(manifest),
        "raw_matrices.npz": matrix_bytes,
        "lane_contract.json": contract_bytes,
    }
    for name, data in files.items():
        (output / name).write_bytes(data)
    expected = {
        "manifest": _sha(files["manifest.json"]),
        "raw_matrices": _sha(files["raw_matrices.npz"]),
        "lane_contract": _sha(files["lane_contract.json"]),
    }
    paths = {
        "manifest": "manifest.json", "raw_matrices": "raw_matrices.npz",
        "lane_contract": "lane_contract.json",
    }
    return expected, paths


def _truth_qhx() -> list[dict[str, Any]]:
    rows = []
    for radius_index, radius in enumerate(R_VALUES):
        static = toy.STATIC_ENERGIES[radius][:2]
        overlaps = toy.TRANSITION_OVERLAPS[radius_index, :, :2, :]
        for operator_index in range(3):
            for time_extent in (5, 6):
                for temporal_extent in NT_VALUES:
                    rows.append({
                        "R": radius, "operator_index": operator_index,
                        "T": time_extent, "N_t": temporal_extent,
                        **_component_values(toy.VACUUM_ENERGIES, static, overlaps,
                                            operator_index, time_extent, temporal_extent),
                    })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _plot(output: Path, result: dict[str, Any], truth: dict[str, Any]) -> None:
    fit = result["spectrum_fit"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")

    labels = [r"$\epsilon_1$", r"$V_0(R=2)$", r"$V_1(R=2)$",
              r"$V_0(R=3)$", r"$V_1(R=3)$"]
    true_values = [toy.VACUUM_ENERGIES[1]]
    fit_values = [fit["vacuum_energies"][1]]
    for radius in R_VALUES:
        true_values.extend(toy.STATIC_ENERGIES[radius][:2].tolist())
        fit_values.extend(fit["static_energies_by_R"][str(radius)])
    x = np.arange(len(labels))
    axes[0].scatter(x - 0.08, true_values, color="black", marker="o", label="injected toy value")
    axes[0].scatter(x + 0.08, fit_values, color="#2b78e4", marker="s", label="two-sector fit")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel(r"lattice energy ($a_t E$)")
    axes[0].set_title("Vacuum and static-sector energies")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    true_by_key = {(row["R"], row["operator_index"], row["T"], row["N_t"]): row
                   for row in truth["qhx_rows"]}
    colors = {"q": "#2b78e4", "h": "#e69138", "x": "#6aa84f"}
    for component, color in colors.items():
        true_values, fitted_values = [], []
        for row in result["qhx_rows"]:
            key = (row["R"], row["operator_index"], row["T"], row["N_t"])
            true = true_by_key[key][component]
            fitted = row.get(component)
            if true > 0 and fitted is not None and fitted > 0:
                true_values.append(true)
                fitted_values.append(fitted)
        axes[1].scatter(true_values, fitted_values, color=color, alpha=0.8,
                        label=f"{component}: {len(true_values)} cells")
    all_values = [value for row in truth["qhx_rows"] for value in (row["q"], row["h"], row["x"])
                  if value > 0]
    lower, upper = min(all_values) * 0.5, max(all_values) * 2.0
    axes[1].plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(lower, upper)
    axes[1].set_ylim(lower, upper)
    axes[1].set_xlabel("synthetic truth q/h/x")
    axes[1].set_ylabel("fitted q/h/x")
    axes[1].set_title("24 fitted q/h/x cells × 3 terms")
    axes[1].grid(which="both", alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.suptitle("Proposed two-sector fit · synthetic CPU only · no certification")
    fig.savefig(output / "two_sector_fit.png", dpi=170)
    plt.close(fig)


def _summary(fit_result: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    fit = fit_result.get("spectrum_fit")
    energy_errors = None
    qhx_errors = None
    if fit is not None:
        energy_errors = {
            "epsilon_1_abs_error": abs(fit["vacuum_energies"][1]
                                       - float(toy.VACUUM_ENERGIES[1])),
            "static_energy_abs_error_by_R": {
                str(radius): [
                    abs(float(value) - float(target))
                    for value, target in zip(fit["static_energies_by_R"][str(radius)],
                                             toy.STATIC_ENERGIES[radius][:2])
                ] for radius in R_VALUES
            },
        }
        truth_rows = {(row["R"], row["operator_index"], row["T"], row["N_t"]): row
                      for row in truth["qhx_rows"]}
        errors = {name: [] for name in ("q", "h", "x")}
        for row in fit_result["qhx_rows"]:
            key = (row["R"], row["operator_index"], row["T"], row["N_t"])
            for name in errors:
                errors[name].append(abs(float(row[name]) - float(truth_rows[key][name])))
        qhx_errors = {name: {"max_absolute_error": max(values),
                             "median_absolute_error": float(np.median(values))}
                      for name, values in errors.items()}
    return {
        "schema": "wilson-f-two-sector-fit-report-summary-v1",
        "status": fit_result["status"], "certified": False,
        "physical_score": None,
        "f_lane_status": "GROUND_STATE_FORCE_UNCERTIFIED",
        "synthetic_truth": {
            "vacuum_energies": [float(value) for value in toy.VACUUM_ENERGIES],
            "static_energies_by_R": {
                str(radius): toy.STATIC_ENERGIES[radius][:2].tolist()
                for radius in R_VALUES
            },
            "noise_fraction": NOISE_FRACTION, "blocks": N_BLOCKS,
            "seed": SEED,
        },
        "fit_result": fit_result,
        "truth_comparison": {"energy_errors": energy_errors, "qhx_errors": qhx_errors},
        "physical_measurements": {"planned": 0, "valid": 0, "failed": 0, "abstained": 0},
        "certification_gate": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
    }


def _write_readme(output: Path, summary: dict[str, Any], source_hashes: dict[str, str]) -> None:
    result = summary["fit_result"]
    fit = result.get("fit_contract") or {}
    errors = summary["truth_comparison"]
    lines = [
        "# Wilson F-lane proposed two-sector fit: synthetic CPU pilot",
        "",
        "**Status:** method prototype only. The input is generated from the proposed two-sector formula; no Wilson measurement, sealed split, GPU/QUDA data or physical receipt was used. `physical_score=null`; the ground-state force remains uncertified.",
        "",
        "## Fixed inputs and model",
        "",
        f"- `N_blocks={N_BLOCKS}`, deterministic seed `{SEED}`, relative per-block Hermitian noise `{NOISE_FRACTION:g}`; `R={list(R_VALUES)}`, `N_t={list(NT_VALUES)}`, `T={list(T_VALUES)}`.",
        "- Synthetic truth uses two vacuum states and two static-source states, with the first two states from the task's existing finite-trace toy. Parameters are not used to construct fit starts except the Nt=32 one-sector GEVP initializer; the vacuum-excitation start grid is fixed at 0.15/0.30/0.55.",
        "- The candidate fits the lane contract's draft trace `C_ij = Z0^-1 sum_mn exp[-eps_m(Nt-T)-V_n(R)T] A_i,mn A*_j,mn` and fixes `eps_0=0`, ordered positive energies, and one phase convention per overlap vector.",
        "- Objective: diagonal inverse-standard-error weighted least squares over 180 Hermitian matrix components; 45 parameters; three deterministic starting points. It ignores cross-component covariance and does not calculate parameter or q/h/x confidence intervals.",
        "",
        "## Result",
        "",
        f"- Fit status: `{result['status']}`; fit denominator `{result['denominator']}`; q/h/x cells `{result['qhx_denominator']}`.",
        f"- Parameter rank/size: `{fit.get('jacobian_rank')}/{fit.get('parameter_count')}`; Jacobian condition number `{fit.get('jacobian_condition_number')}`; diagonal chi-square `{fit.get('chi_square_diagonal')}` (not calibrated).",
        f"- Absolute energy errors versus the injected toy: `{errors['energy_errors']}`.",
        f"- Per-term q/h/x absolute errors across the 24 cells: `{errors['qhx_errors']}`.",
        "",
        "![two-sector fit diagnostic](two_sector_fit.png)",
        "",
        "The left panel compares the fitted vacuum/static energies to the toy truth. The right panel compares fitted q/h/x terms to the same toy truth; it is not a coverage plot. Internal `1+q+h+x` reconstruction is checked by the transfer-trace reader algebra, not by an independent measurement.",
        "",
        "## Data flow and limits",
        "",
        "```text",
        "synthetic finite-Nt matrix NPZ + manifest + selected lane contract",
        "  -> one immutable SourceByteSnapshot -> strict axes/hash/Hermitian checks",
        "  -> Nt=32 forward two-state GEVP initializer -> three fixed starts",
        "  -> joint two-sector diagonal-weight fit -> q/h/x reconstruction -> diagnostic only",
        "```",
        "",
        "This closes only the synthetic CPU fit path for the draft two-sector equation. It does not validate the endpoint/time-origin convention, positive factorization of the production exporter, full covariance, omitted-state truncation, simultaneous coverage, thermal/mixed error budget, reference force, or independent reviewer/authorization. The C lane remains the selected finite-T Creutz estimand and is not merged with this F-lane toy.",
        "",
        "## Source identity",
        "",
        "`source_hashes.json` records exact reader, fit, initializer, toy-generator, contract and test bytes. `artifact_hashes.json` covers all report files other than itself.",
        "",
        "| file | SHA-256 |",
        "|---|---|",
    ]
    lines.extend(f"| `{name}` | `{value}` |" for name, value in source_hashes.items())
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sources(output: Path) -> dict[str, str]:
    source_dir = output / "source_snapshot"
    source_dir.mkdir()
    hashes = {}
    for name, source in SOURCE_PATHS.items():
        payload = source.read_bytes()
        (source_dir / name).write_bytes(payload)
        hashes[name] = _sha(payload)
    (output / "source_hashes.json").write_bytes(_json_bytes({
        "schema": "wilson-f-two-sector-fit-source-hashes-v1", "sources": hashes,
    }))
    return hashes


def _write_artifact_hashes(output: Path) -> dict[str, str]:
    paths = sorted(path for path in output.rglob("*")
                    if path.is_file() and path.name != "artifact_hashes.json")
    hashes = {str(path.relative_to(output)): _sha(path.read_bytes()) for path in paths}
    (output / "artifact_hashes.json").write_bytes(_json_bytes({
        "schema": "wilson-f-two-sector-fit-artifact-hashes-v1", "artifacts": hashes,
    }))
    return hashes


def build(output: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {out}")
    out.mkdir(parents=True)
    expected, paths = _write_input_bundle(out)
    snapshot = capture_source_snapshot(out, expected, paths)
    fit_result = fit_two_sector_snapshot(snapshot)
    truth = {"qhx_rows": _truth_qhx()}
    summary = _summary(fit_result, truth)
    (out / "summary.json").write_bytes(_json_bytes(summary))
    _write_csv(out / "fit_attempts.csv", fit_result.get("fit_attempts", []))
    _write_csv(out / "qhx_comparison.csv", [
        {**row, **{"true_" + key: truth_row[key]
                   for key in ("q", "h", "x")}}
        for row, truth_row in zip(fit_result.get("qhx_rows", []), truth["qhx_rows"])
    ])
    _plot(out, fit_result, truth)
    protocol = {
        "schema": "wilson-f-two-sector-fit-pilot-protocol-v1",
        "classification": "SYNTHETIC_PROPOSED_TRACE_FIT_NOT_PHYSICS_EVIDENCE",
        "source_hashes": expected,
        "source_paths": paths,
        "fit": {
            "vacuum_state_count": 2, "static_state_count": 2,
            "R_values": list(R_VALUES), "N_t_values": list(NT_VALUES),
            "T_values": list(T_VALUES), "parameters": 45, "data_components": 180,
            "objective": "diagonal inverse-SE weighted least squares",
            "starts": [[0.15, 0.20], [0.30, 0.35], [0.55, 0.50]],
            "physical_formula_source": "selected lane_contract_v2.json draft q/h/x trace; production endpoint convention unreviewed",
        },
        "fit_status": fit_result["status"],
        "fit_denominator": fit_result["denominator"],
        "qhx_denominator": fit_result["qhx_denominator"],
        "certified": False, "physical_score": None,
    }
    (out / "protocol.json").write_bytes(_json_bytes(protocol))
    source_hashes = _write_sources(out)
    _write_readme(out, summary, source_hashes)
    artifact_hashes = _write_artifact_hashes(out)
    return {"output": str(out), "fit_status": fit_result["status"],
            "fit_denominator": fit_result["denominator"],
            "qhx_denominator": fit_result["qhx_denominator"],
            "source_files": len(source_hashes),
            "artifacts": len(artifact_hashes)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
