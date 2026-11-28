"""Single-panel omitted-static-state stress for the two-sector F-lane fit."""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import numpy as np

from core.source_integrity import capture_source_snapshot
from tasks.wilson_loop import f_lane_finite_nt_trace_diagnostic as toy
from tasks.wilson_loop import f_lane_two_sector_fit_candidate as candidate
from tasks.wilson_loop import f_lane_two_sector_gls_diagnostic as gls_report
from tasks.wilson_loop.f_lane_two_sector_fit_candidate import (
    RAW_MANIFEST_SCHEMA, RAW_SCHEMA, NT_VALUES, R_VALUES, T_VALUES,
    fit_two_sector_snapshot,
)
from tasks.wilson_loop.f_lane_transfer_trace_reader import _component_values


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
DEFAULT_OUT = TASK_DIR / "f_lane_two_sector_omitted_state_20261127_v1"
N_BLOCKS = 96
NOISE_FRACTION = 0.003
SEED = 2026112739
SOURCE_PATHS = {
    "two_sector_fit_candidate.py": TASK_DIR / "f_lane_two_sector_fit_candidate.py",
    "two_sector_gls_diagnostic.py": TASK_DIR / "f_lane_two_sector_gls_diagnostic.py",
    "two_sector_omitted_state_diagnostic.py": Path(__file__),
    "two_sector_fit_diagnostic.py": TASK_DIR / "f_lane_two_sector_fit_diagnostic.py",
    "joint_forward_fit.py": TASK_DIR / "f_lane_joint_spectral_gls_diagnostic.py",
    "transfer_trace_reader.py": TASK_DIR / "f_lane_transfer_trace_reader.py",
    "finite_trace_toy.py": TASK_DIR / "f_lane_finite_nt_trace_diagnostic.py",
    "raw_matrix_reader.py": TASK_DIR / "f_lane_raw_matrix_snapshot_diagnostic.py",
    "source_integrity.py": REPO_ROOT / "core/source_integrity.py",
    "npz_safety.py": REPO_ROOT / "core/npz_safety.py",
    "lane_contract_v2.json": TASK_DIR / "lane_contract_v2.json",
    "test_two_sector_fit.py": REPO_ROOT / "tests/test_wilson_f_lane_two_sector_fit_candidate.py",
}


def _sha(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _generate_matrices() -> np.ndarray:
    """Use the toy's full three-static-state truth; the fit remains 2x2."""
    rng = np.random.default_rng(SEED)
    matrices = np.empty((N_BLOCKS, len(NT_VALUES), len(R_VALUES), len(T_VALUES), 3, 3),
                        dtype=np.complex128)
    for radius_index, radius in enumerate(R_VALUES):
        static = toy.STATIC_ENERGIES[radius]  # all three truth states
        amplitudes = toy.TRANSITION_OVERLAPS[radius_index]
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


def _write_input(output: Path):
    matrices = _generate_matrices()
    stream = BytesIO()
    np.savez_compressed(
        stream, schema=np.asarray(RAW_SCHEMA), block_ids=np.arange(N_BLOCKS),
        correlator_matrices=matrices, R_values=np.asarray(R_VALUES),
        temporal_extents=np.asarray(NT_VALUES), time_slices=np.asarray(T_VALUES),
    )
    matrix_bytes = stream.getvalue()
    contract_bytes = (TASK_DIR / "lane_contract_v2.json").read_bytes()
    manifest = {
        "schema": RAW_MANIFEST_SCHEMA, "split_id": "F-synthetic-development-omitted-state",
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
    for name, payload in files.items():
        (output / name).write_bytes(payload)
    expected = {"manifest": _sha(files["manifest.json"]),
                "raw_matrices": _sha(matrix_bytes),
                "lane_contract": _sha(contract_bytes)}
    paths = {"manifest": "manifest.json", "raw_matrices": "raw_matrices.npz",
             "lane_contract": "lane_contract.json"}
    return expected, paths


def _truth_rows():
    rows = []
    for radius in R_VALUES:
        radius_index = R_VALUES.index(radius)
        for operator_index in range(3):
            for time_extent in (5, 6):
                for temporal_extent in NT_VALUES:
                    rows.append({
                        "R": radius, "operator_index": operator_index,
                        "T": time_extent, "N_t": temporal_extent,
                        **_component_values(
                            toy.VACUUM_ENERGIES, toy.STATIC_ENERGIES[radius],
                            toy.TRANSITION_OVERLAPS[radius_index], operator_index,
                            time_extent, temporal_extent,
                        ),
                    })
    return rows


def _artifact_hashes(output: Path):
    paths = sorted(path for path in output.rglob("*")
                   if path.is_file() and path.name != "artifact_hashes.json")
    hashes = {str(path.relative_to(output)): _sha(path.read_bytes()) for path in paths}
    (output / "artifact_hashes.json").write_bytes(_json_bytes({
        "schema": "wilson-f-two-sector-omitted-state-artifact-hashes-v1",
        "artifacts": hashes,
    }))
    return hashes


def build(output: str | Path = DEFAULT_OUT):
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {out}")
    source_payloads = {name: path.read_bytes() for name, path in SOURCE_PATHS.items()}
    source_hashes = {name: _sha(payload) for name, payload in source_payloads.items()}
    out.mkdir(parents=True)
    expected, paths = _write_input(out)
    snapshot = capture_source_snapshot(out, expected, paths)
    diagonal = fit_two_sector_snapshot(snapshot, objective_mode="diagonal")
    gls = fit_two_sector_snapshot(snapshot, objective_mode="regularized_full_covariance")
    truth_rows = _truth_rows()
    rows = gls_report._flatten(diagonal, gls, truth_rows)
    summary = gls_report._summary(diagonal, gls, rows)
    summary["classification"] = "SYNTHETIC_OMITTED_STATIC_STATE_SINGLE_PANEL_NOT_COVERAGE"
    summary["generator"] = {
        "vacuum_states": 2, "static_states": 3,
        "fit_vacuum_states": 2, "fit_static_states": 2,
        "blocks": N_BLOCKS, "noise_fraction": NOISE_FRACTION,
        "seed": SEED, "omitted_static_state": "third static-source level at each R",
    }
    summary["fit_bias_diagnostic"] = {
        "pointwise_upper_contains_truth": summary["single_panel_truth_diagnostic"][
            "pointwise_upper_contains_truth"],
        "bonferroni_upper_contains_truth": summary["single_panel_truth_diagnostic"][
            "bonferroni_upper_contains_truth"],
        "terms": 72,
        "interpretation": "one panel with deliberate state-count misspecification; not coverage",
    }
    (out / "summary.json").write_bytes(_json_bytes(summary))
    gls_report._write_csv(out / "qhx_comparison.csv", rows)
    gls_report._plot(out, rows)
    protocol = {
        "schema": "wilson-f-two-sector-omitted-state-protocol-v1",
        "classification": summary["classification"],
        "input_hashes": expected, "input_paths": paths,
        "analysis_source_hashes": source_hashes,
        "generator": summary["generator"],
        "fit": {
            "state_count": "2 vacuum x 2 static fitted to 2 vacuum x 3 static truth",
            "fit_parameters": 45, "data_components": 180,
            "objective_comparison": ["diagonal", "regularized_full_covariance"],
            "covariance_shrinkage": candidate.GLS_DIAGONAL_SHRINKAGE,
            "covariance_eigenvalue_floor_relative": candidate.GLS_EIGENVALUE_FLOOR_RELATIVE,
            "starts": [[0.15, 0.20], [0.30, 0.35], [0.55, 0.50]],
            "interval": "finite-difference delta; pointwise and Bonferroni-72 normal upper",
            "coverage_calibrated": False,
        },
        "denominators": {
            "diagonal_fit": diagonal["denominator"], "gls_fit": gls["denominator"],
            "diagonal_qhx": diagonal["qhx_denominator"], "gls_qhx": gls["qhx_denominator"],
            "truth_comparison": {"planned": 72, "valid": len(rows), "failed": 0,
                                 "abstained": 72 - len(rows)},
        },
        "certified": False, "physical_score": None,
    }
    (out / "protocol.json").write_bytes(_json_bytes(protocol))
    source_dir = out / "source_snapshot"
    source_dir.mkdir()
    for name, payload in source_payloads.items():
        (source_dir / name).write_bytes(payload)
    (out / "source_hashes.json").write_bytes(_json_bytes({
        "schema": "wilson-f-two-sector-omitted-state-source-hashes-v1",
        "sources": source_hashes,
    }))
    readme = [
        "# Wilson F-lane two-sector omitted-state stress",
        "",
        "**Status:** one synthetic CPU panel with deliberate state-count misspecification. This is a bias diagnostic, not repeated-sampling coverage or physical Wilson evidence. `physical_score=null`; F remains uncertified.",
        "",
        "## Frozen comparison",
        "",
        f"- Generator: `{N_BLOCKS}` iid Gaussian noisy blocks, seed `{SEED}`, relative noise `{NOISE_FRACTION}`; truth has two vacuum states and three static-source states.",
        "- Both fit candidates retain two vacuum and two static-source states (45 parameters, 180 Hermitian components), identical three starts and the same raw snapshot. The third static level is deliberately omitted from the fitted model.",
        "- Baseline uses diagonal inverse-SE weights. GLS uses block-mean covariance with fixed 25% diagonal shrinkage and `1e-10` relative eigenvalue floor; q/h/x intervals use a finite-difference delta method.",
        f"- Of 72 known-truth q/h/x terms, pointwise upper bounds contain `{summary['fit_bias_diagnostic']['pointwise_upper_contains_truth']}/72`; Bonferroni-72 upper bounds contain `{summary['fit_bias_diagnostic']['bonferroni_upper_contains_truth']}/72`.",
        "- The plot shows `(estimate - full three-state truth)/GLS delta SE`. Values below the one-sided upper-bound cutoff indicate missed truth in this one draw.",
        "",
        "![Omitted-state standardized errors](standardized_errors.png)",
        "",
        "These are single-panel containment counts, not coverage rates. The interval is conditional on a misspecified 2x2 model and treats estimated covariance as fixed; this stress does not calibrate the candidate interval. It shows why a full-rank fit and narrow asymptotic intervals can fail to protect against omitted spectral states.",
        "",
        "## Data flow and next diagnostic",
        "",
        "```text",
        "2x3 synthetic trace + iid block noise -> immutable raw-matrix snapshot -> same 2x2 model under diagonal and GLS weights -> q/h/x delta intervals -> 72 known-truth misspecification checks -> physical F gate abstains",
        "```",
        "",
        "The next calibration should repeat independent panels over predeclared omitted-level amplitudes/gaps, near-degeneracy, weak ground overlap and correlated blocks; retain fit failures and use Monte Carlo intervals for pointwise and familywise coverage. Do not tune the shrinkage/floor or interval rule against the same panels used for final calibration. Production endpoint conventions, raw `C_ij`, matched `N_t=32`, reference force, contamination/error receipts and independent review remain absent.",
        "",
        "## Source identity",
        "",
        "All diagnostic source bytes and SHA-256 digests are retained in this directory; `artifact_hashes.json` covers the report bundle.",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    artifacts = _artifact_hashes(out)
    return {"output": str(out), "classification": summary["classification"],
            "terms": len(rows), "pointwise_contains": summary["fit_bias_diagnostic"][
                "pointwise_upper_contains_truth"],
            "bonferroni_contains": summary["fit_bias_diagnostic"][
                "bonferroni_upper_contains_truth"], "artifacts": len(artifacts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
