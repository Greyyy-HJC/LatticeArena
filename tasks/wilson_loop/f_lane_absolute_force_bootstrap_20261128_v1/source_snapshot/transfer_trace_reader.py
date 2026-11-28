"""Compare diagonal and covariance-aware two-sector fits on one synthetic panel.

The report is a CPU method diagnostic, not production Wilson evidence or
finite-sample interval calibration.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from core.source_integrity import capture_source_snapshot
from tasks.wilson_loop import f_lane_two_sector_fit_candidate as candidate
from tasks.wilson_loop import f_lane_two_sector_fit_diagnostic as baseline


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
DEFAULT_OUT = TASK_DIR / "f_lane_two_sector_gls_20261127_v1"
SOURCE_PATHS = {
    "two_sector_fit_candidate.py": TASK_DIR / "f_lane_two_sector_fit_candidate.py",
    "two_sector_fit_diagnostic.py": TASK_DIR / "f_lane_two_sector_fit_diagnostic.py",
    "two_sector_gls_diagnostic.py": TASK_DIR / "f_lane_two_sector_gls_diagnostic.py",
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


def _capture_sources() -> tuple[dict[str, bytes], dict[str, str]]:
    payloads = {name: path.read_bytes() for name, path in SOURCE_PATHS.items()}
    return payloads, {name: _sha(payload) for name, payload in payloads.items()}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _flatten(diagonal: dict[str, Any], gls: dict[str, Any], truth_rows: list[dict[str, Any]]):
    by_key = lambda rows: {
        (row["R"], row["operator_index"], row["T"], row["N_t"]): row for row in rows
    }
    diagonal_rows = by_key(diagonal.get("qhx_rows", []))
    gls_rows = by_key(gls.get("qhx_rows", []))
    truth = by_key(truth_rows)
    rows = []
    for key, true_row in truth.items():
        drow, grow = diagonal_rows[key], gls_rows[key]
        for term in ("q", "h", "x"):
            interval = grow["delta_intervals"][term]
            rows.append({
                "R": key[0], "operator_index": key[1], "T": key[2], "N_t": key[3],
                "term": term, "truth": true_row[term],
                "diagonal_estimate": drow[term], "gls_estimate": grow[term],
                "gls_delta_se": interval["delta_standard_error"],
                "gls_pointwise_upper": interval["upper_97_5_pointwise"],
                "gls_bonferroni72_upper": interval["upper_95_bonferroni_72"],
                "pointwise_upper_contains_truth": (
                    interval["upper_97_5_pointwise"] >= true_row[term]),
                "bonferroni_upper_contains_truth": (
                    interval["upper_95_bonferroni_72"] >= true_row[term]),
            })
    return rows


def _plot(output: Path, rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharex=True)
    colors = {"q": "#2b78e4", "h": "#e69138", "x": "#6aa84f"}
    pointwise_z = 1.959963984540054
    simultaneous_z = float(norm.isf(0.05 / 72))
    for axis, term in zip(axes, ("q", "h", "x")):
        subset = [row for row in rows if row["term"] == term]
        x = np.arange(len(subset))
        for method, marker, offset in (("diagonal_estimate", "x", -0.07),
                                       ("gls_estimate", "o", 0.07)):
            score = np.asarray([
                (row[method] - row["truth"]) / max(row["gls_delta_se"], 1e-15)
                for row in subset
            ])
            axis.scatter(x + offset, score, s=21 if marker == "x" else 24,
                         marker=marker, color=colors[term], alpha=0.83,
                         label="diagonal estimate" if method.startswith("diagonal")
                         else "GLS estimate")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axhline(-pointwise_z, color="#666666", linestyle="--", linewidth=0.9,
                     label="pointwise truth-inclusion cutoff" if term == "q" else None)
        axis.axhline(-simultaneous_z, color="#aa3377", linestyle=":", linewidth=1.0,
                     label="Bonferroni-72 cutoff" if term == "q" else None)
        axis.set_title(f"{term.upper()} · 24 cells")
        axis.set_xlabel("cell index (R, operator, T, Nt)")
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("(estimate − toy truth) / GLS delta SE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 0.015))
    fig.suptitle("Two-sector diagonal vs regularized GLS · one synthetic panel only", y=0.985)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(output / "standardized_errors.png", dpi=170)
    plt.close(fig)


def _summary(diagonal: dict[str, Any], gls: dict[str, Any], rows):
    contains_pointwise = sum(row["pointwise_upper_contains_truth"] for row in rows)
    contains_bonferroni = sum(row["bonferroni_upper_contains_truth"] for row in rows)
    standardized = [
        abs(row["gls_estimate"] - row["truth"]) / max(row["gls_delta_se"], 1e-15)
        for row in rows
    ]
    return {
        "schema": "wilson-f-two-sector-gls-diagnostic-summary-v1",
        "classification": "SYNTHETIC_SINGLE_PANEL_METHOD_COMPARISON_NOT_COVERAGE",
        "physical_score": None,
        "certified": False,
        "f_lane_status": "GROUND_STATE_FORCE_UNCERTIFIED",
        "physical_measurements": {"planned": 0, "valid": 0, "failed": 0, "abstained": 0},
        "certification_gate": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
        "diagonal": {
            "status": diagonal["status"], "chi2": diagonal["fit_contract"]["chi_square_diagonal"],
            "rank": diagonal["fit_contract"]["jacobian_rank"],
            "condition_number": diagonal["fit_contract"]["jacobian_condition_number"],
        },
        "regularized_gls": {
            "status": gls["status"], "chi2": gls["fit_contract"]["chi_square_diagonal"],
            "rank": gls["fit_contract"]["jacobian_rank"],
            "condition_number": gls["fit_contract"]["jacobian_condition_number"],
            "covariance_diagnostics": gls["fit_contract"]["covariance_diagnostics"],
            "qhx_interval_method": gls["fit_contract"]["qhx_interval_method"],
            "qhx_interval_coverage_calibrated": False,
        },
        "single_panel_truth_diagnostic": {
            "terms": len(rows),
            "pointwise_upper_contains_truth": int(contains_pointwise),
            "bonferroni_upper_contains_truth": int(contains_bonferroni),
            "gls_standardized_absolute_error_median": float(np.median(standardized)),
            "gls_standardized_absolute_error_max": float(np.max(standardized)),
            "interpretation": "one known-truth synthetic draw; not an estimate of repeated-sampling coverage",
        },
        "open_gates": [
            "single-generator one-panel result does not calibrate interval coverage",
            "finite-difference delta intervals omit optimizer, model-selection and covariance-estimation uncertainty",
            "the generator has exactly two states per sector and independent Gaussian blocks",
            "the draft endpoint/time-origin convention is not validated against a production exporter",
            "no thermal, mixed, reference-force, contamination or production statistical budget was evaluated",
            "source hashes do not establish provenance, external review or authorization",
        ],
    }


def _write_readme(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], hashes):
    single = summary["single_panel_truth_diagnostic"]
    gls = summary["regularized_gls"]
    lines = [
        "# Wilson F-lane two-sector diagonal/GLS comparison",
        "",
        "**Status:** synthetic CPU method diagnostic from one known-truth panel. This is not interval-coverage evidence or a physical Wilson measurement; `physical_score=null` and the force remains uncertified.",
        "",
        "## Fixed comparison",
        "",
        "Both fits use the same immutable 96-block raw-matrix snapshot, two vacuum states, two static states, 45 parameters, 180 real Hermitian components, the same three starts, and the draft finite-`Nt` trace convention. The baseline uses diagonal inverse-SE weights. The GLS candidate estimates the block-mean covariance, applies fixed 25% diagonal shrinkage and a `1e-10` relative eigenvalue floor, then whitens the full residual.",
        "",
        f"- Diagonal: status `{summary['diagonal']['status']}`, rank `{summary['diagonal']['rank']}/45`, chi-square `{summary['diagonal']['chi2']:.6g}`, Jacobian condition `{summary['diagonal']['condition_number']:.6g}`.",
        f"- Regularized GLS: status `{gls['status']}`, rank `{gls['rank']}/45`, chi-square `{gls['chi2']:.6g}`, Jacobian condition `{gls['condition_number']:.6g}`.",
        f"- Covariance: {gls['covariance_diagnostics']}.",
        f"- Of the 72 known-truth q/h/x terms, the pointwise upper includes truth in `{single['pointwise_upper_contains_truth']}/72`; the Bonferroni-72 upper includes truth in `{single['bonferroni_upper_contains_truth']}/72`. This is a single draw and must not be called coverage.",
        f"- Median/max absolute GLS error in estimated delta-SE units: `{single['gls_standardized_absolute_error_median']:.4g}` / `{single['gls_standardized_absolute_error_max']:.4g}`.",
        "- The diagonal and GLS chi-square values use different residual metrics and are not directly comparable as goodness-of-fit tests; neither p-value nor finite-sample fit calibration is supplied.",
        "",
        "![Standardized errors](standardized_errors.png)",
        "",
        "The plot compares diagonal and GLS point-estimate errors in units of the GLS delta-method standard error. A point above the dashed `-1.96` line means the pointwise 97.5% one-sided upper includes truth in this single panel; a point above the dotted Bonferroni-72 cutoff means the analogous approximate simultaneous upper includes it. These are asymptotic diagnostics, not calibrated bounds. The plotted cells share fitted parameters and are correlated.",
        "",
        "## Data flow and limits",
        "",
        "```text",
        "one synthetic raw-matrix bundle -> immutable source snapshot -> same two-sector model and fixed starts",
        "  -> diagonal fit + regularized full-covariance GLS fit -> q/h/x delta-method bounds -> one-draw plot/text report",
        "```",
        "",
        "The covariance has 180 components but only 96 blocks, so its unregularized estimate is rank deficient. The fixed shrinkage yields a numerical metric; it does not establish covariance accuracy or finite-sample inference. The delta method treats the regularized covariance as fixed and omits optimizer-start, fit-window, state-count, covariance-estimation and dependence uncertainty. A next useful CPU step is a preregistered repeated-panel calibration that changes the data-generating mechanism (at minimum omitted third-sector levels, near-degenerate gaps, weak ground overlap, and block dependence), retaining all failures and simultaneous q/h/x denominators. Do not route these limits or this report to the physical F gate.",
        "",
        "## Source identity",
        "",
        "The protocol stores source SHA-256 values and exact source snapshots. The artifact manifest hashes every report file other than itself.",
        "",
        "| source | SHA-256 |",
        "|---|---|",
    ]
    lines.extend(f"| `{name}` | `{digest}` |" for name, digest in hashes.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sources(output: Path, source_payloads: dict[str, bytes], source_hashes):
    source_dir = output / "source_snapshot"
    source_dir.mkdir()
    for name, payload in source_payloads.items():
        (source_dir / name).write_bytes(payload)
    (output / "source_hashes.json").write_bytes(_json_bytes({
        "schema": "wilson-f-two-sector-gls-source-hashes-v1", "sources": source_hashes,
    }))


def _write_artifact_hashes(output: Path):
    paths = sorted(path for path in output.rglob("*")
                   if path.is_file() and path.name != "artifact_hashes.json")
    hashes = {str(path.relative_to(output)): _sha(path.read_bytes()) for path in paths}
    (output / "artifact_hashes.json").write_bytes(_json_bytes({
        "schema": "wilson-f-two-sector-gls-artifact-hashes-v1", "artifacts": hashes,
    }))
    return hashes


def build(output: str | Path = DEFAULT_OUT):
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {out}")
    source_payloads, source_hashes = _capture_sources()
    out.mkdir(parents=True)
    expected, paths = baseline._write_input_bundle(out)
    snapshot = capture_source_snapshot(out, expected, paths)
    diagonal = candidate.fit_two_sector_snapshot(snapshot, objective_mode="diagonal")
    gls = candidate.fit_two_sector_snapshot(
        snapshot, objective_mode="regularized_full_covariance",
    )
    truth_rows = baseline._truth_qhx()
    rows = _flatten(diagonal, gls, truth_rows)
    if len(rows) != 72:
        raise RuntimeError(f"expected 72 q/h/x comparisons, got {len(rows)}")
    summary = _summary(diagonal, gls, rows)
    (out / "summary.json").write_bytes(_json_bytes(summary))
    _write_csv(out / "qhx_comparison.csv", rows)
    _plot(out, rows)
    protocol = {
        "schema": "wilson-f-two-sector-gls-diagnostic-protocol-v1",
        "classification": summary["classification"],
        "input_hashes": expected,
        "input_paths": paths,
        "analysis_source_hashes": source_hashes,
        "fit": {
            "blocks": baseline.N_BLOCKS, "parameters": 45, "components": 180,
            "starts": [[0.15, 0.20], [0.30, 0.35], [0.55, 0.50]],
            "diagonal_objective": "diagonal inverse-SE weighted least squares",
            "gls_objective": gls["fit_contract"]["objective"],
            "gls_covariance_diagnostics": gls["fit_contract"]["covariance_diagnostics"],
            "gls_intervals": gls["fit_contract"]["qhx_interval_method"],
            "interval_calibration": "not performed; one synthetic known-truth panel only",
        },
        "fit_denominators": {
            "diagonal": diagonal["denominator"], "gls": gls["denominator"],
            "qhx_diagonal": diagonal["qhx_denominator"], "qhx_gls": gls["qhx_denominator"],
        },
        "certified": False, "physical_score": None,
    }
    (out / "protocol.json").write_bytes(_json_bytes(protocol))
    _write_sources(out, source_payloads, source_hashes)
    _write_readme(out / "README.md", summary, rows, source_hashes)
    artifact_hashes = _write_artifact_hashes(out)
    return {"output": str(out), "classification": summary["classification"],
            "diagonal_status": diagonal["status"], "gls_status": gls["status"],
            "terms": len(rows), "artifacts": len(artifact_hashes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
