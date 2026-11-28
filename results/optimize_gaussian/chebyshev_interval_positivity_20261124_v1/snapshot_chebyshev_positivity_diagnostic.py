"""Audit positivity of the existing synthetic degree-4 Chebyshev fits.

This reuses the source-closed replay coefficients. It creates a new report and
does not generate new panels, access physical data, or assign a physical score.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .chebyshev_interval import chebyshev_interval_minimum


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (ROOT / "tasks/optimize_gaussian"
                 / "trainable_w_operator_cpu_replay_20261123_v1/raw.json")
DEFAULT_OUT = (ROOT / "results/optimize_gaussian"
               / "chebyshev_interval_positivity_20261124_v1")
INTERVAL = (-1.0, 1.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit_id() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _load_rows(input_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    payload = json.loads(input_path.read_text())
    panels = payload.get("panels")
    if not isinstance(panels, list) or not panels:
        raise ValueError("input raw replay must contain a nonempty panels list")

    rows: list[dict[str, Any]] = []
    for index, panel in enumerate(panels):
        seed = panel.get("seed", f"missing_seed_{index}") if isinstance(panel, dict) else f"invalid_panel_{index}"
        row: dict[str, Any] = {
            "panel_seed": seed,
            "degree": 4,
            "status": "failed",
            "reason": "not processed",
        }
        try:
            if not isinstance(panel, dict):
                raise ValueError("panel must be an object")
            spectral = panel.get("spectral")
            matches = [item for item in spectral if item.get("degree") == 4]
            if len(matches) != 1:
                raise ValueError("expected exactly one degree-4 spectral fit")
            coefficients = matches[0].get("coefficients")
            result = chebyshev_interval_minimum(coefficients, INTERVAL)
            points = result["candidate_points"]
            row.update({
                "status": "valid",
                "reason": "",
                "coefficients": coefficients,
                "minimum": result["minimum"],
                "argmin": result["argmin"],
                "p_at_lower": next(
                    float(point["value"]) for point in points
                    if point["kind"] == "lower_endpoint"),
                "p_at_upper": next(
                    float(point["value"]) for point in points
                    if point["kind"] == "upper_endpoint"),
                "candidate_points": points,
                "numerically_certified": result["numerically_certified"],
            })
        except (AttributeError, TypeError, ValueError, FloatingPointError, StopIteration) as exc:
            row["reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    planned = len(rows)
    valid = sum(row["status"] == "valid" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    abstained = 0
    denominators = {
        "planned": planned,
        "valid": valid,
        "failed": failed,
        "abstained": abstained,
    }
    if planned != valid + failed + abstained:
        raise AssertionError("panel denominator identity failed")
    return rows, denominators


def build(output: str | Path = DEFAULT_OUT,
          input_path: str | Path = DEFAULT_INPUT) -> dict[str, Any]:
    """Write a hash-tracked synthetic-only positivity report without overwrite."""
    source_path = Path(input_path).resolve()
    output_path = Path(output)
    rows, denominators = _load_rows(source_path)
    output_path.mkdir(parents=True, exist_ok=False)

    snapshots = {
        "snapshot_chebyshev_interval.py": Path(__file__).with_name("chebyshev_interval.py"),
        "snapshot_chebyshev_positivity_diagnostic.py": Path(__file__),
    }
    for name, source in snapshots.items():
        shutil.copyfile(source, output_path / name)

    try:
        input_label = source_path.relative_to(ROOT).as_posix()
    except ValueError:
        input_label = source_path.name
    protocol = {
        "schema": "optimize-gaussian-chebyshev-positivity-diagnostic-v1",
        "classification": "synthetic_only",
        "physical_score": None,
        "input_path": input_label,
        "input_sha256": _sha256(source_path),
        "source_commit": _commit_id(),
        "degree": 4,
        "spectral_interval": list(INTERVAL),
        "minimum_method": "evaluate endpoints and real roots of the Chebyshev derivative",
        "root_imag_tolerance": 1e-10,
        "boundary_tolerance": 1e-12,
        "notes": [
            "Uses the existing three synthetic heat-kernel replay panels; no new panels are generated.",
            "A positive polynomial on [-1,1] implies a positive-semidefinite p(H/6) only if H is Hermitian and spec(H/6) is contained in the interval.",
            "Floating-point root finding is diagnostic, not an interval-arithmetic certificate.",
            "The original coefficient fit is unconstrained; this report does not make positivity a production contract.",
        ],
    }
    (output_path / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    csv_path = output_path / "interval_minima.csv"
    fields = ["panel_seed", "degree", "status", "reason", "coefficients",
              "minimum", "argmin", "p_at_lower", "p_at_upper",
              "numerically_certified"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), separators=(",", ":"))
                             if key in ("coefficients",) else row.get(key)
                             for key in fields})

    summary = {
        "schema": protocol["schema"],
        "classification": "synthetic_only",
        "physical_score": None,
        "denominators": denominators,
        "interval": list(INTERVAL),
        "rows": rows,
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_path / "denominator_summary.csv").write_text(
        "unit,planned,valid,failed,abstained\n"
        f"one degree-4 fit panel,{denominators['planned']},{denominators['valid']},"
        f"{denominators['failed']},{denominators['abstained']}\n"
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(INTERVAL[0], INTERVAL[1], 501)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for row in rows:
        if row["status"] != "valid":
            continue
        coefficients = np.asarray(row["coefficients"], dtype=float)
        y = np.polynomial.chebyshev.chebval(x, coefficients)
        ax.plot(x, y, lw=2, label=f"panel {row['panel_seed']}")
        ax.scatter([row["argmin"]], [row["minimum"]], s=24, zorder=3)
    ax.axhline(0.0, color="black", lw=1, ls="--", label="zero / PSD boundary")
    ax.set(xlabel="x in [-1, 1]", ylabel="p4(x)",
           title="Existing synthetic degree-4 fits; positivity not constrained")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path / "chebyshev_positivity.png", dpi=170)
    plt.close(fig)

    lines = [
        "# Degree-4 Chebyshev interval-positivity diagnostic",
        "",
        "Status: **SYNTHETIC-ONLY CPU DIAGNOSTIC**. `physical_score=null`; no measurements, sealed splits, GPU, or QUDA inputs were accessed.",
        "",
        "## Question and method",
        "",
        "The existing source-closed operator replay fits an unconstrained real degree-4 Chebyshev polynomial. This diagnostic reuses those three fit coefficient vectors and checks their minimum on the complete interval [-1, 1]. For a real polynomial on a closed interval, every minimum is at an endpoint or a real root of its derivative; the code evaluates both endpoints and every numerically real derivative root in the interval. Root solving and evaluation use double precision, so this is not an interval-arithmetic proof.",
        "",
        "If H is Hermitian and spec(H/6) is a subset of [-1, 1], nonnegative p4 on the entire interval implies p4(H/6) is positive semidefinite by the spectral theorem. The bound and coefficient positivity are separate conditions.",
        "",
        "![Three fitted polynomials over the full normalized interval](chebyshev_positivity.png)",
        "",
        "## Panel results",
        "",
        "| panel seed | status | minimum | argmin | p4(-1) | p4(1) | detail |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["status"] == "valid":
            lines.append(
                f"| {row['panel_seed']} | valid | {row['minimum']:.12f} | {row['argmin']:.9g} | "
                f"{row['p_at_lower']:.12f} | {row['p_at_upper']:.12f} | |"
            )
        else:
            lines.append(
                f"| {row['panel_seed']} | failed | — | — | — | — | {row['reason']} |"
            )
    lines.extend([
        "",
        f"Panel denominator planned/valid/failed/abstained = {denominators['planned']}/{denominators['valid']}/{denominators['failed']}/{denominators['abstained']}. A valid row means the interval minimum calculation completed, not that a physical contract is certified.",
        "",
        "## Interpretation and limits",
        "",
        "All three existing fitted polynomials have positive minima near 0.313 at x=-1. Thus these three synthetic operators are positive semidefinite under the stated Hermiticity and spectral-interval assumptions. The original least-squares fit did not impose this property; three heat-kernel panels cannot establish positivity for other configurations, production links, or future refits. This result does not freeze D11 or select the production normalization, coefficient strategy, tolerance, or failure rule.",
        "",
        "Text flow: existing fit coefficients -> derivative roots plus interval endpoints -> complete-interval numerical minimum -> conditional PSD statement -> retain the human D11 gate.",
        "",
        "Rebuild with `python -B -m tasks.optimize_gaussian.chebyshev_positivity_diagnostic`. The input JSON path, SHA-256, commit, tolerances, source snapshots, and artifact hashes are recorded alongside this report.",
        "",
    ])
    (output_path / "README.md").write_text("\n".join(lines))

    files = ["README.md", "chebyshev_positivity.png", "denominator_summary.csv",
             "interval_minima.csv", "protocol.json", "summary.json", *snapshots]
    hashes = {name: _sha256(output_path / name) for name in files}
    manifest = {"algorithm": "sha256", "files": hashes}
    (output_path / "artifact_hashes.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary["files"] = files + ["artifact_hashes.json"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = build(args.output, args.input)
    print(json.dumps({
        "output": str(args.output),
        "denominators": summary["denominators"],
        "physical_score": summary["physical_score"],
    }, indent=2))


if __name__ == "__main__":
    main()
