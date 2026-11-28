"""Repeated-panel coverage stress for a two-sector fit with an omitted level.

This CPU-only diagnostic keeps the 2x3 synthetic generator fixed while drawing
independent noisy block panels.  The fitted model remains 2x2.  It measures
the candidate GLS delta upper-bound coverage under this one misspecified DGP;
it is not physical Wilson evidence or a general coverage certification.
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
from scipy.stats import norm, t as student_t

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
DEFAULT_OUT = TASK_DIR / "f_lane_two_sector_omitted_coverage_20261128_v1"
DEFAULT_PANELS = 24
N_BLOCKS = 96
NOISE_FRACTION = 0.003
SEED_BASE = 2026112800
TERMS = ("q", "h", "x")
SOURCE_PATHS = {
    "two_sector_fit_candidate.py": TASK_DIR / "f_lane_two_sector_fit_candidate.py",
    "two_sector_fit_initializer.py": TASK_DIR / "f_lane_two_sector_fit_diagnostic.py",
    "two_sector_gls_diagnostic.py": TASK_DIR / "f_lane_two_sector_gls_diagnostic.py",
    "two_sector_omitted_state_diagnostic.py": TASK_DIR / "f_lane_two_sector_omitted_state_diagnostic.py",
    "two_sector_omitted_coverage_diagnostic.py": Path(__file__),
    "two_sector_multistratum_screen.py": TASK_DIR / "f_lane_two_sector_multistratum_screen.py",
    "two_sector_multistratum_screen_protocol.json": TASK_DIR / "f_lane_two_sector_multistratum_screen_protocol.json",
    "joint_forward_fit.py": TASK_DIR / "f_lane_joint_spectral_gls_diagnostic.py",
    "transfer_trace_reader.py": TASK_DIR / "f_lane_transfer_trace_reader.py",
    "finite_trace_toy.py": TASK_DIR / "f_lane_finite_nt_trace_diagnostic.py",
    "raw_matrix_reader.py": TASK_DIR / "f_lane_raw_matrix_snapshot_diagnostic.py",
    "gevp_overlap_object.py": TASK_DIR / "gevp_overlap_object_diagnostic.py",
    "gevp_t0_truncation.py": TASK_DIR / "gevp_t0_truncation_diagnostic.py",
    "source_integrity.py": REPO_ROOT / "core/source_integrity.py",
    "npz_safety.py": REPO_ROOT / "core/npz_safety.py",
    "lane_contract_v2.json": TASK_DIR / "lane_contract_v2.json",
}


def _sha(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _hermitian_noise(rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    noise = 0.5 * (raw + raw.conj().T)
    noise[np.diag_indices(3)] = rng.normal(size=3)
    return noise


def _generate_matrices(seed: int, noise_spec: dict[str, Any] | None = None) -> np.ndarray:
    """Draw one independent 96-block panel from the current synthetic truth."""
    rng = np.random.default_rng(seed)
    noise_spec = noise_spec or {"model": "iid_gaussian", "scale_by_T": {}}
    model = noise_spec.get("model", "iid_gaussian")
    if model not in {"iid_gaussian", "stationary_ar1"}:
        raise ValueError("noise model must be iid_gaussian or stationary_ar1")
    rho = noise_spec.get("rho", 0.0)
    if (isinstance(rho, bool) or not isinstance(rho, (int, float))
            or not np.isfinite(rho) or not 0 <= rho < 1):
        raise ValueError("noise rho must be finite and satisfy 0 <= rho < 1")
    scale_by_t = noise_spec.get("scale_by_T", {})
    if not isinstance(scale_by_t, dict):
        raise ValueError("scale_by_T must map time extents to positive scales")
    for time, scale in scale_by_t.items():
        if (str(time) not in {str(value) for value in T_VALUES}
                or isinstance(scale, bool) or not isinstance(scale, (int, float))
                or not np.isfinite(scale) or scale <= 0):
            raise ValueError("scale_by_T contains an invalid time extent or scale")
    matrices = np.empty((N_BLOCKS, len(NT_VALUES), len(R_VALUES), len(T_VALUES), 3, 3),
                        dtype=np.complex128)
    for radius_index, radius in enumerate(R_VALUES):
        static = toy.STATIC_ENERGIES[radius]
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
                time_scale = float(scale_by_t.get(str(time_extent), 1.0))
                sigma = max(float(np.max(np.abs(mean))) * NOISE_FRACTION * time_scale, 1e-9)
                if model == "iid_gaussian":
                    # Preserve the legacy draw order for the original report.
                    for block in range(N_BLOCKS):
                        raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
                        noise = 0.5 * (raw + raw.conj().T)
                        noise[np.diag_indices(3)] = rng.normal(size=3)
                        matrices[block, nt_index, radius_index, time_index] = mean + sigma * noise
                else:
                    innovations = []
                    for _ in range(N_BLOCKS):
                        raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
                        noise = 0.5 * (raw + raw.conj().T)
                        noise[np.diag_indices(3)] = rng.normal(size=3)
                        innovations.append(noise)
                    sequence = np.empty((N_BLOCKS, 3, 3), dtype=np.complex128)
                    sequence[0] = innovations[0]
                    innovation_scale = np.sqrt(1.0 - float(rho) ** 2)
                    for block in range(1, N_BLOCKS):
                        sequence[block] = (float(rho) * sequence[block - 1]
                                           + innovation_scale * innovations[block])
                    matrices[:, nt_index, radius_index, time_index] = mean + sigma * sequence
    return matrices


def _truth_rows() -> list[dict[str, Any]]:
    rows = []
    for radius_index, radius in enumerate(R_VALUES):
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


def _wilson_interval(successes: int, total: int, alpha: float = 0.05):
    if total == 0:
        return [None, None]
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    z = float(norm.ppf(1 - alpha / 2))
    if successes == 0:
        upper = (z * z / total) / (1 + z * z / total)
        return [0.0, float(upper)]
    if successes == total:
        lower = 1 / (1 + z * z / total)
        return [float(lower), 1.0]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def _cluster_t_interval(panel_values: list[float]):
    values = np.asarray(panel_values, dtype=float)
    if values.size < 2:
        return [None, None]
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(values.size))
    critical = float(student_t.ppf(0.975, values.size - 1))
    return [float(max(0.0, mean - critical * se)),
            float(min(1.0, mean + critical * se))]


def _panel_source_paths(report_root: Path, panel_dir: Path):
    relative_panel = panel_dir.relative_to(report_root).as_posix()
    return {
        "manifest": f"{relative_panel}/manifest.json",
        "raw_matrices": f"{relative_panel}/raw_matrices.npz",
        "lane_contract": "lane_contract.json",
    }


def _save_panel_input(report_root: Path, panel_dir: Path, panel_index: int, seed: int,
                      source_payloads: dict[str, bytes],
                      noise_spec: dict[str, Any] | None = None,
                      split_prefix: str = "omitted-state"):
    matrices = _generate_matrices(seed, noise_spec)
    stream = BytesIO()
    np.savez_compressed(
        stream, schema=np.asarray(RAW_SCHEMA), block_ids=np.arange(N_BLOCKS),
        correlator_matrices=matrices, R_values=np.asarray(R_VALUES),
        temporal_extents=np.asarray(NT_VALUES), time_slices=np.asarray(T_VALUES),
    )
    matrix_bytes = stream.getvalue()
    contract_bytes = source_payloads["lane_contract_v2.json"]
    manifest = {
        "schema": RAW_MANIFEST_SCHEMA,
        "split_id": f"F-synthetic-open-screen-{split_prefix}-{panel_index:03d}",
        "geometry": "S16T16", "action": "wilson_gauge_beta6",
        "boundary_conditions": "periodic", "radius_order": list(R_VALUES),
        "temporal_extents": list(NT_VALUES), "time_slices": list(T_VALUES),
        "dimension_order": "block,Nt,R,T,i,j",
        "matrix_sha256": _sha(matrix_bytes),
        "lane_contract_sha256": _sha(contract_bytes),
    }
    payloads = {
        "manifest.json": _json_bytes(manifest),
        "raw_matrices.npz": matrix_bytes,
    }
    for name, payload in payloads.items():
        (panel_dir / name).write_bytes(payload)
    expected = {"manifest": _sha(payloads["manifest.json"]),
                "raw_matrices": _sha(matrix_bytes),
                "lane_contract": _sha(contract_bytes)}
    snapshot = capture_source_snapshot(
        report_root, expected, _panel_source_paths(report_root, panel_dir),
        max_total_bytes=8 * 1024 * 1024,
    )
    return snapshot, {name: digest for name, digest in expected.items()}


def _load_panel_input(report_root: Path, panel_dir: Path):
    manifest_bytes = (panel_dir / "manifest.json").read_bytes()
    matrix_bytes = (panel_dir / "raw_matrices.npz").read_bytes()
    contract_bytes = (report_root / "lane_contract.json").read_bytes()
    expected = {"manifest": _sha(manifest_bytes), "raw_matrices": _sha(matrix_bytes),
                "lane_contract": _sha(contract_bytes)}
    snapshot = capture_source_snapshot(
        report_root, expected, _panel_source_paths(report_root, panel_dir),
        max_total_bytes=8 * 1024 * 1024,
    )
    return snapshot, expected


def _plot(output: Path, rows: list[dict[str, Any]], panels: list[dict[str, Any]], summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1))
    colors = {"q": "#3478bf", "h": "#e28e2c", "x": "#59a14f"}
    for term in TERMS:
        term_rows = [row for row in rows if row["term"] == term]
        term_rows = [row for row in term_rows if row["estimate"] is not None]
        indices = np.arange(len(term_rows))
        axis = axes[0]
        axis.scatter(indices, [row["estimate"] - row["truth"] for row in term_rows],
                     s=11, alpha=0.55, color=colors[term], label=f"{term.upper()} estimate − truth")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Signed q/h/x bias across panel-cells")
    axes[0].set_xlabel("panel-cell index (panels grouped in order)")
    axes[0].set_ylabel("estimate − exact scenario truth")
    axes[0].legend(fontsize=8, loc="best")
    term_labels = [term.upper() for term in TERMS] + ["all 72 in panel"]
    rates = [summary["coverage"]["pointwise_by_term"][term]["cluster_mean"] for term in TERMS]
    rates.append(summary["coverage"]["familywise_bonferroni72"]["rate"])
    lows = [summary["coverage"]["pointwise_by_term"][term]["cluster_t95"][0] for term in TERMS]
    highs = [summary["coverage"]["pointwise_by_term"][term]["cluster_t95"][1] for term in TERMS]
    lows.append(summary["coverage"]["familywise_bonferroni72"]["wilson95"][0])
    highs.append(summary["coverage"]["familywise_bonferroni72"]["wilson95"][1])
    # Wilson bounds can be microscopically displaced from the raw estimate at
    # a boundary (e.g. 0/24); Matplotlib requires nonnegative error lengths.
    yerr = np.asarray([[max(0.0, r - lo) for r, lo in zip(rates, lows)],
                       [max(0.0, hi - r) for r, hi in zip(rates, highs)]])
    axes[1].errorbar(np.arange(4), rates, yerr=yerr, fmt="o", capsize=4,
                     color="#6f4e7c", ecolor="#777777")
    axes[1].axhline(0.975, color="#3478bf", linestyle="--", linewidth=1,
                    label="pointwise nominal 97.5%")
    axes[1].axhline(0.95, color="#e28e2c", linestyle=":", linewidth=1,
                    label="familywise nominal 95%")
    axes[1].set_xticks(np.arange(4), term_labels)
    axes[1].set_ylim(0, 1.03)
    axes[1].set_title("Pointwise and familywise containment")
    axes[1].set_ylabel("fraction containing exact truth")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(fontsize=8, loc="lower left")
    stratum = summary.get("report_metadata", {}).get("stratum_id", "omitted-state screen")
    fig.suptitle(f"F-lane repeated-panel diagnostic · {stratum}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output / "repeated_panel_coverage.png", dpi=180)
    plt.close(fig)

    force_rows = [row for row in panels if row.get("absolute_force_interval_valid")]
    force_fig, force_axis = plt.subplots(figsize=(9, 4.8))
    for row in force_rows:
        panel = int(row["panel"])
        estimate = float(row["absolute_force_estimate"])
        lower = float(row["absolute_force_lower_95"])
        upper = float(row["absolute_force_upper_95"])
        force_axis.errorbar(
            panel, estimate,
            yerr=[[max(0.0, estimate - lower)], [max(0.0, upper - estimate)]],
            fmt="o", color=("#3478bf" if row["absolute_force_interval_contains_truth"] else "#d62728"),
            ecolor="#777777", capsize=3,
        )
    truth = float(panels[0]["absolute_force_truth"]) if panels else 0.0
    force_axis.axhline(truth, color="black", linestyle="--", linewidth=1,
                       label=f"exact toy truth = {truth:.6g}")
    force_axis.set_xlabel("independent synthetic panel")
    force_axis.set_ylabel(r"$a_t[E_0(R=3)-E_0(R=2)]$ per $\Delta R=1$")
    force_axis.set_title("Absolute stout ground-force delta intervals")
    force_axis.grid(alpha=0.2)
    force_axis.legend(loc="best")
    force_fig.tight_layout()
    force_fig.savefig(output / "absolute_force_intervals.png", dpi=180)
    plt.close(force_fig)


def _artifact_hashes(output: Path):
    paths = sorted(path for path in output.rglob("*")
                   if path.is_file() and path.name != "artifact_hashes.json")
    hashes = {str(path.relative_to(output)): _sha(path.read_bytes()) for path in paths}
    (output / "artifact_hashes.json").write_bytes(_json_bytes({
        "schema": "wilson-f-two-sector-omitted-coverage-artifact-hashes-v1",
        "artifacts": hashes,
    }))
    return hashes


def build(output: str | Path = DEFAULT_OUT, n_panels: int = DEFAULT_PANELS,
          resume: bool = False, *, seed_base: int = SEED_BASE,
          noise_spec: dict[str, Any] | None = None,
          report_metadata: dict[str, Any] | None = None):
    out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite existing report: {out}")
    if isinstance(n_panels, bool) or not isinstance(n_panels, int) or n_panels < 2:
        raise ValueError("n_panels must be an integer >= 2")
    if isinstance(seed_base, bool) or not isinstance(seed_base, int) or seed_base < 0:
        raise ValueError("seed_base must be a nonnegative integer")
    noise_spec = noise_spec or {"model": "iid_gaussian", "scale_by_T": {}}
    report_metadata = dict(report_metadata or {})
    source_payloads = {name: path.read_bytes() for name, path in SOURCE_PATHS.items()}
    source_hashes = {name: _sha(payload) for name, payload in source_payloads.items()}
    if out.exists():
        progress_path = out / "progress.json"
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("planned_panels") != n_panels:
                raise ValueError("resume panel count differs from the existing run")
        # With no progress receipt, resume may still complete deterministic
        # inputs left between panel creation and the first completed fit.
    else:
        out.mkdir(parents=True)
    panel_root = out / "panels"
    panel_root.mkdir(exist_ok=True)
    shared_contract = out / "lane_contract.json"
    if resume:
        if not shared_contract.is_file():
            legacy_contract = panel_root / "panel_000" / "lane_contract.json"
            shared_contract.write_bytes(legacy_contract.read_bytes())
        expected_contract = _sha(source_payloads["lane_contract_v2.json"])
        if _sha(shared_contract.read_bytes()) != expected_contract:
            raise ValueError("saved shared lane contract differs from the captured source contract")
        for panel_contract in panel_root.glob("panel_*/lane_contract.json"):
            if _sha(panel_contract.read_bytes()) != expected_contract:
                raise ValueError("panel lane contract bytes are not identical to the shared contract")
    else:
        shared_contract.write_bytes(source_payloads["lane_contract_v2.json"])
    truth_rows = _truth_rows()
    truth_by_key = {(row["R"], row["operator_index"], row["T"], row["N_t"]): row
                    for row in truth_rows}
    panel_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    fit_attempt_rows: list[dict[str, Any]] = []
    for panel_index in range(n_panels):
        seed = seed_base + panel_index
        panel_dir = panel_root / f"panel_{panel_index:03d}"
        if (resume and (panel_dir / "manifest.json").is_file()
                and (panel_dir / "raw_matrices.npz").is_file()):
            snapshot, input_hashes = _load_panel_input(out, panel_dir)
        else:
            panel_dir.mkdir(exist_ok=True)
            snapshot, input_hashes = _save_panel_input(
                out, panel_dir, panel_index, seed, source_payloads,
                noise_spec=noise_spec,
                split_prefix=str(report_metadata.get("split_prefix", "omitted-state")),
            )
        fit = fit_two_sector_snapshot(snapshot, objective_mode="regularized_full_covariance")
        for start_index, attempt in enumerate(fit.get("fit_attempts", [])):
            fit_attempt_rows.append({
                "panel": panel_index, "seed": seed, "start_index": start_index,
                **{key: attempt.get(key) for key in (
                    "epsilon_start", "excited_overlap_scale_start", "optimizer_success",
                    "finite", "converged", "status_code", "nfev", "chi_square_diagonal",
                    "jacobian_rank", "parameter_count", "message",
                )},
            })
        fit_ok = fit.get("denominator", {}).get("valid") == 1 and fit.get("fit_contract", {}).get("jacobian_rank") == 45
        force_truth = float(toy.STATIC_ENERGIES[3][0] - toy.STATIC_ENERGIES[2][0])
        force_diagnostic = fit.get("force_estimand_diagnostic") or {}
        force_interval = force_diagnostic.get("absolute_stout_force_delta_interval") or {}
        force_values = [force_interval.get(key) for key in
                        ("estimate", "standard_error", "lower_95", "upper_95")]
        force_interval_valid = bool(
            fit_ok and all(value is not None and np.isfinite(value) for value in force_values)
            and force_interval.get("finite_sample_coverage_calibrated") is False
        )
        force_estimate = force_interval.get("estimate") if force_interval_valid else None
        force_lower = force_interval.get("lower_95") if force_interval_valid else None
        force_upper = force_interval.get("upper_95") if force_interval_valid else None
        force_contains = bool(force_interval_valid and force_lower <= force_truth <= force_upper)
        run_rows.append({
            "panel": panel_index, "seed": seed,
            "fit_status": fit.get("status"), "fit_valid": fit_ok,
            "rank": (fit.get("fit_contract") or {}).get("jacobian_rank"),
            "condition_number": (fit.get("fit_contract") or {}).get("jacobian_condition_number"),
            "objective": (fit.get("fit_contract") or {}).get("chi_square_diagonal"),
            "input_matrix_sha256": input_hashes["raw_matrices"],
            "qhx_fit_valid": fit.get("qhx_denominator", {}).get("valid", 0),
            "qhx_fit_failed": fit.get("qhx_denominator", {}).get("failed", 0),
            "absolute_force_truth": force_truth,
            "absolute_force_estimate": force_estimate,
            "absolute_force_standard_error": (force_interval.get("standard_error")
                                                if force_interval_valid else None),
            "absolute_force_lower_95": force_lower,
            "absolute_force_upper_95": force_upper,
            "absolute_force_interval_contains_truth": force_contains,
            "absolute_force_interval_valid": force_interval_valid,
        })
        by_key = {(row.get("R"), row.get("operator_index"), row.get("T"), row.get("N_t")): row
                  for row in fit.get("qhx_rows", [])}
        for truth_key, truth in truth_by_key.items():
            estimated = by_key.get(truth_key)
            for term in TERMS:
                interval = (estimated or {}).get("delta_intervals", {}).get(term)
                row = {
                    "panel": panel_index, "seed": seed,
                    "R": truth["R"], "operator_index": truth["operator_index"],
                    "T": truth["T"], "N_t": truth["N_t"], "term": term,
                    "truth": truth[term],
                    "fit_status": fit.get("status"),
                    "estimate": (estimated or {}).get(term),
                    "delta_se": (interval or {}).get("delta_standard_error"),
                    "pointwise_upper": (interval or {}).get("upper_97_5_pointwise"),
                    "bonferroni72_upper": (interval or {}).get("upper_95_bonferroni_72"),
                    "pointwise_contains": bool(interval and interval["upper_97_5_pointwise"] >= truth[term]),
                    "bonferroni72_contains": bool(interval and interval["upper_95_bonferroni_72"] >= truth[term]),
                    "fit_cell_valid": bool(interval),
                }
                cell_rows.append(row)
                if term == "q" and truth_key[1] == 0 and truth_key[2:] == (5, 16):
                    panel_rows.append(row)
        pointwise = sum(row["pointwise_contains"] for row in cell_rows if row["panel"] == panel_index)
        bonf = sum(row["bonferroni72_contains"] for row in cell_rows if row["panel"] == panel_index)
        valid_cells = sum(row["fit_cell_valid"] for row in cell_rows if row["panel"] == panel_index)
        run_rows[-1].update({
            "pointwise_cells_contained": pointwise,
            "bonferroni72_cells_contained": bonf,
            "valid_qhx_cells": valid_cells,
            "familywise_bonferroni72_contains": bool(fit_ok and valid_cells == 72 and bonf == 72),
        })
        # A progress receipt remains useful if a later panel fails or is interrupted.
        (out / "progress.json").write_bytes(_json_bytes({
            "completed_panels": panel_index + 1, "planned_panels": n_panels,
            "last_panel_status": fit.get("status"),
        }))
    _write_csv(out / "panel_results.csv", run_rows)
    _write_csv(out / "cell_coverage.csv", cell_rows)
    _write_csv(out / "fit_attempts.csv", fit_attempt_rows)

    by_panel_term = {}
    for term in TERMS:
        values = []
        for panel_index in range(n_panels):
            subset = [row for row in cell_rows
                      if row["panel"] == panel_index and row["term"] == term]
            values.append(sum(row["pointwise_contains"] for row in subset) / 24)
        by_panel_term[term] = {
            "cluster_mean": float(np.mean(values)),
            "cluster_t95": _cluster_t_interval(values),
            "n_independent_panels": n_panels,
        }
    pooled_success = sum(row["pointwise_contains"] for row in cell_rows)
    pooled_total = len(cell_rows)
    family_success = sum(row["familywise_bonferroni72_contains"] for row in run_rows)
    fit_success = sum(row["fit_valid"] for row in run_rows)
    force_rows = [row for row in run_rows if row["absolute_force_interval_valid"]]
    force_success = sum(row["absolute_force_interval_contains_truth"] for row in force_rows)
    force_truth = float(run_rows[0]["absolute_force_truth"])
    classification = str(report_metadata.get(
        "classification", "SYNTHETIC_REPEATED_PANEL_COVERAGE_UNDER_ONE_MISSPECIFIED_DGP"))
    truth_description = str(report_metadata.get(
        "truth", "2 vacuum x 3 static states; fit fixed at 2 x 2"))
    fit_description = str(report_metadata.get("fit_model", "2 vacuum x 2 static"))
    seed_rule = str(report_metadata.get(
        "seed_rule", f"numpy default_rng({seed_base} + panel_index)"))
    summary = {
        "schema": "wilson-f-two-sector-omitted-coverage-summary-v2",
        "classification": classification,
        "physical_score": None, "certified": False,
        "f_lane_status": "GROUND_STATE_FORCE_UNCERTIFIED",
        "physical_measurements": {"planned": 0, "valid": 0, "failed": 0, "abstained": 0},
        "certification_gate": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
        "generator": {
            "truth": truth_description,
            "panels": n_panels, "blocks_per_panel": N_BLOCKS,
            "relative_noise": NOISE_FRACTION,
            "seed_rule": seed_rule,
            "noise_model": noise_spec,
            "panel_independence": "fresh independent RNG stream per panel; blocks follow the declared within-panel noise model",
        },
        "fit": {
            "model": fit_description,
            "method": "regularized full-covariance GLS; q/h/x finite-difference delta uppers and absolute-force asymptotic normal delta interval",
            "covariance_shrinkage": candidate.GLS_DIAGONAL_SHRINKAGE,
            "covariance_eigenvalue_floor_relative": candidate.GLS_EIGENVALUE_FLOOR_RELATIVE,
            "pointwise_one_sided_level": 0.975,
            "simultaneous_method": "one-sided Bonferroni over 72 q/h/x terms at family alpha 0.05",
            "absolute_force_interval": "two-sided 95% normal delta interval from inverse whitened-Jacobian covariance; diagnostic and uncalibrated",
            "fit_parameter_count": 45,
            "interval_rule_frozen_across_panels": True,
        },
        "denominators": {
            "fit": {"planned": n_panels, "valid": fit_success,
                    "failed": n_panels - fit_success, "abstained": 0},
            "q_h_x_cells": {"planned": len(cell_rows),
                             "valid": sum(row["fit_cell_valid"] for row in cell_rows),
                             "failed": sum(not row["fit_cell_valid"] and row["fit_status"] not in ("two_sector_fit_candidate",) for row in cell_rows),
                             "abstained": sum(not row["fit_cell_valid"] and row["fit_status"] == "two_sector_fit_candidate" for row in cell_rows)},
            "absolute_force_intervals": {
                "planned": n_panels, "valid": len(force_rows),
                "failed": n_panels - fit_success,
                "abstained": fit_success - len(force_rows),
            },
            "unique_panel_matrix_hashes": len({row["input_matrix_sha256"] for row in run_rows}),
        },
        "coverage": {
            "pointwise_all_cells": {
                "successes": pooled_success, "planned_cells": pooled_total,
                "pooled_rate_descriptive_only": pooled_success / pooled_total,
                "wilson95_cellwise_not_cluster_adjusted": _wilson_interval(pooled_success, pooled_total),
                "warning": "cells within a panel share the same fit and are correlated; use panel-cluster summaries for inference",
            },
            "pointwise_by_term": by_panel_term,
            "absolute_stout_ground_force_normal_delta_95": {
                "successes": force_success,
                "valid_intervals": len(force_rows),
                "planned_panels": n_panels,
                "exact_toy_truth": force_truth,
                "observed_containment": (force_success / len(force_rows) if force_rows else None),
                "wilson95_over_valid_panels": _wilson_interval(force_success, len(force_rows)),
                "median_interval_width": (float(np.median([
                    row["absolute_force_upper_95"] - row["absolute_force_lower_95"]
                    for row in force_rows
                ])) if force_rows else None),
                "warning": "one fixed synthetic omitted-state design only; this is screening evidence, not calibrated physical coverage",
            },
            "average_pointwise_by_panel": {
                "rate": float(np.mean([sum(row["pointwise_contains"] for row in cell_rows if row["panel"] == i) / 72 for i in range(n_panels)])),
                "cluster_t95": _cluster_t_interval([
                    sum(row["pointwise_contains"] for row in cell_rows if row["panel"] == i) / 72
                    for i in range(n_panels)
                ]),
            },
            "familywise_bonferroni72": {
                "successes": family_success, "planned_panels": n_panels,
                "rate": family_success / n_panels,
                "wilson95": _wilson_interval(family_success, n_panels),
            },
        },
        "fit_diagnostics": {
            "median_jacobian_condition_number": float(np.median([
                row["condition_number"] for row in run_rows if row["condition_number"] is not None
            ])) if fit_success else None,
            "max_jacobian_condition_number": max([
                row["condition_number"] for row in run_rows if row["condition_number"] is not None
            ], default=None),
            "median_gls_objective": float(np.median([
                row["objective"] for row in run_rows if row["objective"] is not None
            ])) if fit_success else None,
        },
        "open_gates": [
            "one omitted-state truth design only; no near-degenerate, weak-ground-overlap, vacuum-omission, finite-Nt alternates, correlated-block, or unequal-covariance grid",
            "delta intervals condition on estimated covariance and the misspecified 2x2 model; no method selection or recalibration performed",
            "small number of independent panels gives wide Monte Carlo intervals",
            "production correlators, exporter convention, matched-Nt thermal/mixed receipts, force/reference uncertainty and independent review absent",
        ],
        "report_metadata": report_metadata,
    }
    if report_metadata.get("open_gates") is not None:
        summary["open_gates"] = list(report_metadata["open_gates"])
    (out / "summary.json").write_bytes(_json_bytes(summary))
    protocol = {
        "schema": "wilson-f-two-sector-omitted-coverage-protocol-v2",
        "classification": summary["classification"],
        "source_hashes": source_hashes,
        "generator": summary["generator"], "fit": summary["fit"],
        "report_metadata": report_metadata,
        "seed_manifest": [{"panel": row["panel"], "seed": row["seed"],
                            "matrix_sha256": row["input_matrix_sha256"]}
                           for row in run_rows],
        "denominators": summary["denominators"],
        "coverage": summary["coverage"],
        "certified": False, "physical_score": None,
    }
    (out / "protocol.json").write_bytes(_json_bytes(protocol))
    (out / "source_hashes.json").write_bytes(_json_bytes({"sources": source_hashes}))
    source_dir = out / "source_snapshot"
    source_dir.mkdir(exist_ok=True)
    for name, payload in source_payloads.items():
        (source_dir / name).write_bytes(payload)
    _plot(out, cell_rows, run_rows, summary)
    readme = [
        "# Wilson F lane: repeated-panel coverage diagnostic",
        "",
        f"**Status:** `{classification}`. This is not a physical Wilson measurement or certification; `physical_score=null`, F remains `GROUND_STATE_FORCE_UNCERTIFIED`.",
        "",
        "## Frozen design and result",
        "",
        f"- `{n_panels}` independent synthetic panels, `{N_BLOCKS}` blocks per panel; truth: `{truth_description}`; fit: `{fit_description}`; relative noise `{NOISE_FRACTION}`; noise model `{noise_spec}`; seed rule `{seed_rule}`.",
        "- Every panel uses the same regularized full-covariance GLS and finite-difference delta intervals: one-sided 97.5% pointwise, plus Bonferroni-72 familywise upper bounds. No interval/shrinkage/window was tuned on these panels.",
        f"- Fit denominator: `{fit_success}/{n_panels}` valid, `{n_panels-fit_success}` failed; q/h/x planned cells `{len(cell_rows)}`. Unique matrix hashes: `{summary['denominators']['unique_panel_matrix_hashes']}/{n_panels}`.",
        f"- Pointwise pooled descriptive containment: `{pooled_success}/{pooled_total} = {pooled_success/pooled_total:.3f}`; interpret with the panel-cluster term means and intervals in the summary because cells within a fit are correlated.",
        f"- Entire 72-term Bonferroni family contained in `{family_success}/{n_panels}` panels (`{family_success/n_panels:.3f}`), Wilson 95% MC interval `{summary['coverage']['familywise_bonferroni72']['wilson95']}`.",
        f"- Absolute stout-force normal-delta interval: `{force_success}/{len(force_rows)}` valid panels contain exact toy truth `{force_truth:.6g}`; planned/valid/failed/abstained `{n_panels}/{len(force_rows)}/{n_panels-fit_success}/{fit_success-len(force_rows)}`; Wilson 95% Monte Carlo interval `{summary['coverage']['absolute_stout_ground_force_normal_delta_95']['wilson95_over_valid_panels']}`.",
        "- The force interval uses the inverse whitened-Jacobian covariance and a two-sided normal delta approximation. It is included to expose estimator-specific behavior, not to claim finite-sample calibration.",
        "",
        "![Repeated panel bias and coverage](repeated_panel_coverage.png)",
        "",
        "![Absolute stout-force interval coverage](absolute_force_intervals.png)",
        "",
        "## Data flow",
        "",
        "```text",
        "freeze scenario truth/noise + 2x2 fit + interval rules -> disjoint seed per panel -> write and hash raw matrix snapshot -> same GLS fit -> compare 72 q/h/x uppers and absolute-force normal interval with exact toy truth -> keep separate coverage/failure denominators -> physical F gate abstains",
        "```",
        "",
        "## Interpretation and limits",
        "",
        str(report_metadata.get("interpretation", "The repeated panels estimate coverage only for this fixed synthetic design and noise model. The number of independent panels is small for tight Monte Carlo conclusions; cells are clustered by panel and the familywise event is the independent-panel unit.")),
        "",
        "The source/NPZ snapshots, fit-start ledger, panel/cell CSVs, protocol, figures and hashes are retained; full parameter vectors are omitted because the raw inputs and derived coverage cells are sufficient to replay and inspect this diagnostic. Synthetic CPU screening does not validate physical source/exporter/thermal/mixed/force receipts. No GPU/QUDA or sealed data were used.",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    (out / "fit_results.json").unlink(missing_ok=True)
    for cached_fit in panel_root.glob("panel_*/fit_result.json"):
        cached_fit.unlink()
    for panel_contract in panel_root.glob("panel_*/lane_contract.json"):
        panel_contract.unlink()
    (out / "progress.json").unlink(missing_ok=True)
    hashes = _artifact_hashes(out)
    return {"output": str(out), "panels": n_panels,
            "fit_valid": fit_success, "fit_failed": n_panels-fit_success,
            "pointwise_pooled": f"{pooled_success}/{pooled_total}",
            "familywise": f"{family_success}/{n_panels}",
            "artifacts": len(hashes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--panels", type=int, default=DEFAULT_PANELS)
    parser.add_argument("--resume", action="store_true",
                        help="resume report writing from saved deterministic panel inputs")
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.panels, args.resume), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
