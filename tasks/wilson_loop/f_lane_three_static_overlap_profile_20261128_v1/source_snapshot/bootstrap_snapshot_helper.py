"""Full-refit moving-block bootstrap screen for the absolute F-lane force.

Synthetic CPU diagnostic only. Every bootstrap sample refits the current
regularized two-sector GLS candidate; it does not certify physical coverage.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.source_integrity import capture_source_snapshot
from tasks.wilson_loop import f_lane_finite_nt_trace_diagnostic as toy
from tasks.wilson_loop import f_lane_two_sector_fit_candidate as candidate
from tasks.wilson_loop import f_lane_two_sector_omitted_coverage_diagnostic as generator
from tasks.wilson_loop.f_lane_two_sector_fit_candidate import (
    RAW_MANIFEST_SCHEMA, RAW_SCHEMA, NT_VALUES, R_VALUES, T_VALUES,
    fit_two_sector_snapshot,
)


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
DEFAULT_OUT = TASK_DIR / "f_lane_absolute_force_bootstrap_20261128_v1"
N_BLOCKS = 96
N_PANELS = 6
BOOTSTRAP_REPLICATES = 39
BLOCK_LENGTH = 8
ALPHA = 0.05
RHO = 0.5
SEED_BASE = 2026113800
BOOTSTRAP_SEED_OFFSET = 100_000_000

SOURCE_PATHS = {
    "absolute_force_bootstrap_diagnostic.py": Path(__file__),
    "two_sector_fit_candidate.py": Path(candidate.__file__),
    "two_sector_omitted_coverage_generator.py": Path(generator.__file__),
    "finite_trace_toy.py": Path(toy.__file__),
    "joint_forward_fit.py": TASK_DIR / "f_lane_joint_spectral_gls_diagnostic.py",
    "transfer_trace_reader.py": TASK_DIR / "f_lane_two_sector_gls_diagnostic.py",
    "lane_contract_v2.json": TASK_DIR / "lane_contract_v2.json",
    "source_integrity.py": REPO_ROOT / "core/source_integrity.py",
    "npz_safety.py": REPO_ROOT / "core/npz_safety.py",
}


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _circular_mbb_indices(rng: np.random.Generator, n: int, block_length: int) -> np.ndarray:
    if (isinstance(n, bool) or not isinstance(n, int) or n < 1
            or isinstance(block_length, bool) or not isinstance(block_length, int)
            or not 1 <= block_length <= n):
        raise ValueError("n and block_length must satisfy 1 <= block_length <= n")
    starts = rng.integers(0, n, size=math.ceil(n / block_length))
    return np.concatenate([
        (int(start) + np.arange(block_length)) % n for start in starts
    ])[:n].astype(int)


def _percentile_interval(values: list[float | None], alpha: float = ALPHA) -> dict[str, Any]:
    """Strict equal-tail interval: any failed refit abstains the whole interval."""
    if (not values or not math.isfinite(alpha) or not 0 < alpha < 1
            or len(values) < 2):
        raise ValueError("bootstrap draws and alpha are invalid")
    failed = sum(value is None or not math.isfinite(float(value)) for value in values)
    lower_rank = int(math.ceil((len(values) + 1) * alpha / 2))
    upper_rank = int(math.ceil((len(values) + 1) * (1 - alpha / 2)))
    result = {
        "method": "equal-tail percentile interval from full-refit circular MBB",
        "alpha": alpha,
        "planned_draws": len(values),
        "valid_draws": len(values) - failed,
        "failed_or_missing_draws": failed,
        "lower_order_rank_1based": lower_rank,
        "upper_order_rank_1based": upper_rank,
        "status": "ABSTAIN_BOOTSTRAP_FAILURES" if failed else "VALID",
        "lower": None,
        "upper": None,
        "finite_sample_coverage_calibrated": False,
    }
    if failed:
        return result
    if upper_rank > len(values):
        result["status"] = "ABSTAIN_INSUFFICIENT_TAIL_RESOLUTION"
        return result
    ordered = np.sort(np.asarray(values, dtype=float))
    result["lower"] = float(ordered[lower_rank - 1])
    result["upper"] = float(ordered[upper_rank - 1])
    return result


def _raw_bytes(matrices: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(
        stream, schema=np.asarray(RAW_SCHEMA), block_ids=np.arange(matrices.shape[0]),
        correlator_matrices=matrices, R_values=np.asarray(R_VALUES),
        temporal_extents=np.asarray(NT_VALUES), time_slices=np.asarray(T_VALUES),
    )
    return stream.getvalue()


def _snapshot_from_matrices(root: Path, matrices: np.ndarray, contract_bytes: bytes,
                            split_id: str, *, panel_prefix: str = ""):
    root.mkdir(parents=True, exist_ok=True)
    contract_path = root / "lane_contract.json"
    contract_path.write_bytes(contract_bytes)
    raw = _raw_bytes(matrices)
    raw_rel = f"{panel_prefix}raw_matrices.npz"
    manifest_rel = f"{panel_prefix}manifest.json"
    (root / raw_rel).parent.mkdir(parents=True, exist_ok=True)
    (root / raw_rel).write_bytes(raw)
    manifest = {
        "schema": RAW_MANIFEST_SCHEMA, "split_id": split_id,
        "geometry": "S16T16", "action": "wilson_gauge_beta6",
        "boundary_conditions": "periodic", "radius_order": list(R_VALUES),
        "temporal_extents": list(NT_VALUES), "time_slices": list(T_VALUES),
        "dimension_order": "block,Nt,R,T,i,j",
        "matrix_sha256": _sha(raw), "lane_contract_sha256": _sha(contract_bytes),
    }
    manifest_bytes = _json_bytes(manifest)
    (root / manifest_rel).write_bytes(manifest_bytes)
    expected = {
        "manifest": _sha(manifest_bytes), "raw_matrices": _sha(raw),
        "lane_contract": _sha(contract_bytes),
    }
    paths = {
        "manifest": manifest_rel, "raw_matrices": raw_rel,
        "lane_contract": "lane_contract.json",
    }
    snapshot = capture_source_snapshot(root, expected, paths, max_total_bytes=8 * 1024 * 1024)
    return snapshot, expected


def _fit_force(snapshot) -> tuple[dict[str, Any], float | None, dict[str, Any] | None]:
    fit = fit_two_sector_snapshot(snapshot, objective_mode="regularized_full_covariance")
    denominator = fit.get("denominator") or {}
    valid = (denominator.get("valid") == 1
             and (fit.get("fit_contract") or {}).get("jacobian_rank") == 45)
    diagnostic = fit.get("force_estimand_diagnostic") or {}
    estimate = diagnostic.get("absolute_stout_force_from_shared_spectrum") if valid else None
    interval = diagnostic.get("absolute_stout_force_delta_interval") if valid else None
    if estimate is not None and not math.isfinite(float(estimate)):
        estimate = None
    return fit, (float(estimate) if estimate is not None else None), interval


def _bootstrap_panel(args) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    panel_index, seed, matrices, contract_bytes = args
    force_truth = float(toy.STATIC_ENERGIES[3][0] - toy.STATIC_ENERGIES[2][0])
    bootstrap_values: list[float | None] = []
    bootstrap_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="latticearena_force_mbb_") as temp:
        root = Path(temp)
        original, original_hashes = _snapshot_from_matrices(
            root / "original", matrices, contract_bytes,
            f"F-synthetic-open-absolute-force-bootstrap-{panel_index:03d}",
        )
        point_fit, estimate, delta = _fit_force(original)
        point_valid = estimate is not None
        for draw_index in range(BOOTSTRAP_REPLICATES):
            draw_seed = seed + BOOTSTRAP_SEED_OFFSET + draw_index
            if not point_valid:
                fit = {"status": "ABSTAIN_OUTER_FIT_INVALID", "fit_contract": {}}
                draw_estimate = None
                attempted = False
            else:
                rng = np.random.default_rng(draw_seed)
                indices = _circular_mbb_indices(rng, N_BLOCKS, BLOCK_LENGTH)
                draw_snapshot, _ = _snapshot_from_matrices(
                    root / f"bootstrap_{draw_index:03d}", matrices[indices], contract_bytes,
                    f"F-synthetic-open-absolute-force-bootstrap-{panel_index:03d}-{draw_index:03d}",
                )
                try:
                    fit, draw_estimate, _ = _fit_force(draw_snapshot)
                    attempted = True
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    fit = {"status": f"FAILED_{type(exc).__name__}", "fit_contract": {}}
                    draw_estimate = None
                    attempted = True
            bootstrap_values.append(draw_estimate)
            bootstrap_rows.append({
                "panel": panel_index, "panel_seed": seed,
                "bootstrap_draw": draw_index, "bootstrap_seed": draw_seed,
                "block_length": BLOCK_LENGTH, "fit_status": fit.get("status"),
                "fit_attempted": attempted,
                "fit_valid": draw_estimate is not None,
                "jacobian_rank": (fit.get("fit_contract") or {}).get("jacobian_rank"),
                "force_estimate": draw_estimate,
            })
        interval = _percentile_interval(bootstrap_values)
    delta_result = None
    if delta:
        delta_result = {
            "lower": delta.get("lower_95"), "upper": delta.get("upper_95"),
            "standard_error": delta.get("standard_error"),
            "status": "VALID_UNCALIBRATED",
        }
    row = {
        "panel": panel_index, "seed": seed,
        "input_matrix_sha256": original_hashes["raw_matrices"],
        "outer_fit_status": point_fit.get("status"),
        "outer_fit_valid": point_valid,
        "jacobian_rank": (point_fit.get("fit_contract") or {}).get("jacobian_rank"),
        "force_truth": force_truth,
        "force_estimate": estimate,
        "delta_lower_95": (delta_result or {}).get("lower"),
        "delta_upper_95": (delta_result or {}).get("upper"),
        "delta_se": (delta_result or {}).get("standard_error"),
        "delta_interval_width": (
            delta_result["upper"] - delta_result["lower"] if delta_result else None
        ),
        "delta_contains_truth": bool(
            delta_result and delta_result["lower"] <= force_truth <= delta_result["upper"]
        ),
        "bootstrap_interval_status": interval["status"],
        "bootstrap_lower_95": interval["lower"],
        "bootstrap_upper_95": interval["upper"],
        "bootstrap_valid_draws": interval["valid_draws"],
        "bootstrap_failed_draws": interval["failed_or_missing_draws"],
        "bootstrap_contains_truth": bool(
            interval["status"] == "VALID"
            and interval["lower"] <= force_truth <= interval["upper"]
        ),
        "bootstrap_interval_width": (
            interval["upper"] - interval["lower"] if interval["status"] == "VALID" else None
        ),
    }
    return row, bootstrap_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, panel_rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    truth = float(panel_rows[0]["force_truth"])
    axes[0].axhline(truth, color="black", linestyle="--", linewidth=1, label=f"truth = {truth:.5g}")
    method_data = (
        ("delta", "delta_lower_95", "delta_upper_95", "#3478bf", -0.16),
        ("full-refit MBB", "bootstrap_lower_95", "bootstrap_upper_95", "#e28e2c", 0.16),
    )
    for row in panel_rows:
        panel = int(row["panel"])
        for label, lower_key, upper_key, color, offset in method_data:
            lower, upper = row[lower_key], row[upper_key]
            if lower is None or upper is None:
                marker_y = row["force_estimate"] if row["force_estimate"] is not None else truth
                axes[0].scatter(panel + offset, marker_y, marker="x", color="#777777", s=35)
                continue
            covers = lower <= truth <= upper
            draw_color = color if covers else "#c62828"
            axes[0].vlines(panel + offset, lower, upper, color=draw_color, linewidth=2.2)
            axes[0].scatter(panel + offset, row["force_estimate"], color=draw_color, s=20,
                            label=label if panel == int(panel_rows[0]["panel"]) else None)
    axes[0].set_xlabel("independent AR(1) synthetic panel")
    axes[0].set_ylabel(r"$a_t[E_0(R=3)-E_0(R=2)]$ per $\Delta R=1$")
    axes[0].set_title("95% interval candidates (red = misses toy truth)")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="best", fontsize=8)

    labels, widths = [], []
    for label, _, _, _, _ in method_data:
        key = "delta_interval_width" if label == "delta" else "bootstrap_interval_width"
        values = [row.get(key) for row in panel_rows if row.get(key) is not None]
        labels.append(label)
        widths.append(float(np.median(values)) if values else 0.0)
    bars = axes[1].bar(labels, widths, color=[item[3] for item in method_data])
    for bar, width in zip(bars, widths):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{width:.3g}",
                     ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("median interval width (lattice units)")
    axes[1].set_title("Interval width among valid panels")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Absolute-force MBB screen · 2×2 fit under 2×3 truth")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _artifact_hashes(output: Path) -> dict[str, str]:
    hashes = {
        str(path.relative_to(output)): _sha(path.read_bytes())
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output / "artifact_hashes.json").write_bytes(_json_bytes({
        "schema": "wilson-absolute-force-bootstrap-artifact-hashes-v1",
        "artifacts": hashes,
    }))
    return hashes


def run(output: str | Path = DEFAULT_OUT, *, max_workers: int = 4) -> dict[str, Any]:
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {out}")
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    out.mkdir(parents=True)
    contract_bytes = SOURCE_PATHS["lane_contract_v2.json"].read_bytes()
    (out / "lane_contract.json").write_bytes(contract_bytes)
    source_bytes = {name: path.read_bytes() for name, path in SOURCE_PATHS.items()}
    source_hashes = {name: _sha(data) for name, data in source_bytes.items()}
    source_dir = out / "source_snapshot"
    source_dir.mkdir()
    for name, data in source_bytes.items():
        (source_dir / name).write_bytes(data)

    args = []
    seed_manifest = []
    for panel_index in range(N_PANELS):
        seed = SEED_BASE + panel_index
        matrices = generator._generate_matrices(seed, {
            "model": "stationary_ar1", "rho": RHO, "scale_by_T": {},
        })
        snapshot, hashes = _snapshot_from_matrices(
            out, matrices, contract_bytes,
            f"F-synthetic-open-absolute-force-bootstrap-{panel_index:03d}",
            panel_prefix=f"panels/panel_{panel_index:03d}/",
        )
        seed_manifest.append({
            "panel": panel_index, "seed": seed,
            "matrix_sha256": hashes["raw_matrices"],
        })
        args.append((panel_index, seed, matrices, contract_bytes))

    results, draw_rows = [], []
    with ProcessPoolExecutor(max_workers=min(max_workers, N_PANELS)) as pool:
        futures = {pool.submit(_bootstrap_panel, item): item[0] for item in args}
        for done, future in enumerate(as_completed(futures), 1):
            panel_index = futures[future]
            try:
                row, rows = future.result()
            except Exception as exc:
                row = {
                    "panel": panel_index, "seed": SEED_BASE + panel_index,
                    "input_matrix_sha256": seed_manifest[panel_index]["matrix_sha256"],
                    "outer_fit_status": f"FAILED_{type(exc).__name__}",
                    "outer_fit_valid": False, "jacobian_rank": None,
                    "force_truth": float(toy.STATIC_ENERGIES[3][0] - toy.STATIC_ENERGIES[2][0]),
                    "force_estimate": None, "delta_lower_95": None, "delta_upper_95": None,
                    "delta_se": None, "delta_contains_truth": False,
                    "bootstrap_interval_status": "ABSTAIN_WORKER_FAILURE",
                    "bootstrap_lower_95": None, "bootstrap_upper_95": None,
                    "bootstrap_valid_draws": 0, "bootstrap_failed_draws": 0,
                    "bootstrap_contains_truth": False, "bootstrap_interval_width": None,
                    "worker_error": type(exc).__name__,
                }
                rows = [{
                    "panel": panel_index, "panel_seed": SEED_BASE + panel_index,
                    "bootstrap_draw": draw_index,
                    "bootstrap_seed": SEED_BASE + panel_index + BOOTSTRAP_SEED_OFFSET + draw_index,
                    "block_length": BLOCK_LENGTH,
                    "fit_status": f"ABSTAIN_WORKER_FAILURE_{type(exc).__name__}",
                    "fit_attempted": False, "fit_valid": False,
                    "jacobian_rank": None, "force_estimate": None,
                } for draw_index in range(BOOTSTRAP_REPLICATES)]
            results.append(row)
            draw_rows.extend(rows)
            print(f"completed bootstrap panel {done}/{N_PANELS}", flush=True)
    results.sort(key=lambda row: row["panel"])
    draw_rows.sort(key=lambda row: (row["panel"], row["bootstrap_draw"]))
    _write_csv(out / "panel_results.csv", results)
    _write_csv(out / "bootstrap_refits.csv", draw_rows)

    force_valid = sum(row["outer_fit_valid"] for row in results)
    interval_valid = sum(row["bootstrap_interval_status"] == "VALID" for row in results)
    delta_valid = sum(row["delta_lower_95"] is not None for row in results)
    delta_hits = sum(row["delta_contains_truth"] for row in results if row["delta_lower_95"] is not None)
    bootstrap_hits = sum(row["bootstrap_contains_truth"] for row in results
                         if row["bootstrap_interval_status"] == "VALID")
    refit_planned = N_PANELS * BOOTSTRAP_REPLICATES
    refit_valid = sum(row["fit_valid"] for row in draw_rows)
    refit_failed = sum(row["fit_attempted"] and not row["fit_valid"] for row in draw_rows)
    refit_abstained = refit_planned - refit_valid - refit_failed
    denominator = {
        "outer_panels": {"planned": N_PANELS, "valid": force_valid,
                         "failed": N_PANELS - force_valid, "abstained": 0},
        "delta_intervals": {"planned": N_PANELS, "valid": delta_valid,
                             "failed": N_PANELS - force_valid,
                             "abstained": force_valid - delta_valid},
        "bootstrap_refits": {"planned": refit_planned, "valid": refit_valid,
                             "failed": refit_failed,
                             "abstained": refit_abstained},
        "bootstrap_intervals": {"planned": N_PANELS, "valid": interval_valid,
                                "failed": N_PANELS - force_valid,
                                "abstained": force_valid - interval_valid},
        "physical_measurements": {"planned": 0, "valid": 0, "failed": 0, "abstained": 0},
        "certification": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
    }
    delta_widths = [row["delta_upper_95"] - row["delta_lower_95"] for row in results
                    if row["delta_lower_95"] is not None and row["delta_upper_95"] is not None]
    bootstrap_widths = [row["bootstrap_interval_width"] for row in results
                         if row["bootstrap_interval_width"] is not None]
    summary = {
        "schema": "wilson-absolute-force-bootstrap-summary-v1",
        "classification": "SYNTHETIC_SCREENING_ONLY_FULL_REFIT_MBB",
        "physical_score": None, "certified": False,
        "f_lane_status": "GROUND_STATE_FORCE_UNCERTIFIED",
        "truth": {
            "vacuum_states": 2, "static_states": 3,
            "fit_model": "2-vacuum x 2-static two-sector GLS",
            "exact_force": float(toy.STATIC_ENERGIES[3][0] - toy.STATIC_ENERGIES[2][0]),
            "force_unit": "a_t energy difference per one spatial lattice step (Delta R=1)",
        },
        "generator": {
            "panel_count": N_PANELS, "blocks_per_panel": N_BLOCKS,
            "noise_model": "stationary AR(1) across ordered blocks",
            "rho": RHO, "relative_noise": generator.NOISE_FRACTION,
            "panel_seed_rule": f"{SEED_BASE} + panel_index",
            "bootstrap_seed_rule": f"panel_seed + {BOOTSTRAP_SEED_OFFSET} + draw_index",
            "panel_seeds_disjoint_from_previous_force_delta_screen": True,
        },
        "bootstrap": {
            "replicates_per_panel": BOOTSTRAP_REPLICATES,
            "block_length": BLOCK_LENGTH,
            "resampling": "circular moving blocks with replacement; concatenate and truncate to 96 ordered block rows",
            "fit": "full regularized 2x2 finite-Nt GLS refit for every resample; no parameter draw approximation",
            "interval": "equal-tail 95% percentile using ranks ceil((B+1)*0.025) and ceil((B+1)*0.975); any failed refit abstains the panel interval",
            "tail_resolution": "B=39 makes the 95% interval the minimum-to-maximum bootstrap fit range; very coarse screening only",
            "comparison": "same outer panel asymptotic normal delta interval from inverse whitened-Jacobian covariance",
        },
        "denominators": denominator,
        "coverage": {
            "delta": {"successes": delta_hits, "valid": delta_valid,
                      "planned_panels": N_PANELS,
                      "observed_rate": delta_hits / delta_valid if delta_valid else None,
                      "median_width": float(np.median(delta_widths)) if delta_widths else None},
            "full_refit_mbb": {"successes": bootstrap_hits, "valid": interval_valid,
                                "planned_panels": N_PANELS,
                                "observed_rate": bootstrap_hits / interval_valid if interval_valid else None,
                                "median_width": float(np.median(bootstrap_widths)) if bootstrap_widths else None},
            "warning": "six outer panels and a 39-draw extreme-order interval are a method-feasibility screen, not a coverage calibration",
        },
        "seed_manifest": seed_manifest,
        "source_hashes": source_hashes,
        "open_gates": [
            "only one omitted-static-state positive-spectrum design; broader near-degenerate, weak-overlap, vacuum-omission and covariance strata remain untested",
            "the full-refit bootstrap cannot remove model misspecification bias; this run does not profile a 3-static-state model envelope",
            "independent production exporter, source/time/boundary conventions, N_t=32 thermal/mixed receipts, force scale and review are absent",
        ],
    }
    (out / "protocol.json").write_bytes(_json_bytes({
        "schema": "wilson-absolute-force-bootstrap-protocol-v1",
        **summary,
    }))
    (out / "summary.json").write_bytes(_json_bytes(summary))
    _plot(out / "absolute_force_bootstrap.png", results)
    report = [
        "# Absolute stout ground-force full-refit bootstrap screen",
        "",
        f"**Status:** `{summary['classification']}`. The fit uses a 2-vacuum x 2-static model on a fixed positive-spectrum 2-vacuum x 3-static synthetic truth; no physical score is produced.",
        "",
        "## Frozen method and outcome",
        "",
        f"- Independent panels: `{N_PANELS}`; `{N_BLOCKS}` ordered blocks/panel; AR(1) `rho={RHO}`, relative noise `{generator.NOISE_FRACTION}`; panel seeds `{SEED_BASE} + panel_index`.",
        f"- Each panel gets `{BOOTSTRAP_REPLICATES}` circular moving-block resamples of length `{BLOCK_LENGTH}` and a complete regularized 2x2 GLS refit. Any failed refit abstains that panel's percentile interval.",
        f"- Bootstrap refits denominator: `{refit_planned}/{refit_valid}/{refit_failed}/{denominator['bootstrap_refits']['abstained']}` planned/valid/failed/abstained; panel interval denominator `{N_PANELS}/{interval_valid}/{denominator['bootstrap_intervals']['failed']}/{denominator['bootstrap_intervals']['abstained']}`.",
        f"- Delta intervals: `{delta_hits}/{delta_valid}` contain the exact force; full-refit MBB intervals: `{bootstrap_hits}/{interval_valid}` contain it. Their median widths are `{summary['coverage']['delta']['median_width']}` and `{summary['coverage']['full_refit_mbb']['median_width']}` lattice units.",
        "- With `B=39`, the equal-tail 95% interval uses the sample minimum and maximum. This is deliberately reported as a coarse method screen, not a calibrated interval.",
        "",
        "![Delta and full-refit MBB intervals](absolute_force_bootstrap.png)",
        "",
        "## Data flow",
        "",
        "```text",
        "six disjoint AR(1) panels -> full 2x2 GLS fit -> circular length-8 block resample -> hash-bound raw snapshot -> full GLS refit -> retain failures -> 95% extreme-order interval vs exact 2x3 toy force",
        "```",
        "",
        "## Interpretation and limits",
        "",
        "The bootstrap reproduces sampling variation under the panel's empirical block process, but a refit of the same misspecified 2x2 model cannot remove omitted-state bias. Six panels and B=39 are too small for coverage claims. This result may reject a method candidate; it cannot certify physical force uncertainty or replace an independently justified model-truncation envelope.",
        "",
        "No Wilson measurements, GEVP data, sealed splits, GPU or QUDA were accessed. C/F stay `NOT_READY/UNCERTIFIED`; F stays `GROUND_STATE_FORCE_UNCERTIFIED`; `physical_score=null`.",
    ]
    (out / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    hashes = _artifact_hashes(out)
    return {
        "output": str(out), "outer_panels": N_PANELS,
        "bootstrap_refits_planned": refit_planned,
        "bootstrap_refits_valid": refit_valid,
        "bootstrap_refits_failed": refit_failed,
        "delta_coverage": f"{delta_hits}/{delta_valid}",
        "mbb_coverage": f"{bootstrap_hits}/{interval_valid}",
        "artifacts": len(hashes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.output, max_workers=args.workers), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
