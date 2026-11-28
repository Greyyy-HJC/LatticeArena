"""Linear design audit for separating finite-Nt q/h/x trace components.

The audit fixes the hand-set two-sector energies from the companion toy and
asks only whether their five correction kernels are linearly distinguishable
on the selected (Nt,T) grid. It does not fit Wilson data, model covariance, or
certify a production spectrum.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "tasks/wilson_loop/f_lane_qhx_design_identifiability_20261126_v1"
R_VALUES = (2, 3)
N_T_PLANS = ((16,), (32,), (16, 32))
T_VALUES = tuple(range(3, 8))
MODE_LABELS = ("q_static_1", "q_static_2", "h_vacuum_1", "x_mixed_1", "x_mixed_2")
MODE_INDICES = ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2))


def _integer_tuple(values: Iterable[int], name: str, *, minimum: int = 1) -> tuple[int, ...]:
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonempty iterable of integers") from exc
    if not result or any(type(value) is not int or value < minimum for value in result):
        raise ValueError(f"{name} must contain integers >= {minimum}")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def correction_design_matrix(
    radius: int,
    temporal_extents: Iterable[int],
    time_extents: Iterable[int],
    *,
    static_energies: Iterable[float] | None = None,
    vacuum_energies: Iterable[float] | None = None,
) -> np.ndarray:
    """Return correction kernels after fixing the (m=0,n=0) coefficient to 1.

    Columns are ordered as two q modes, one h mode, then two mixed x modes.
    Energies use the same a_t E convention as the companion finite-trace toy.
    """
    if type(radius) is not int or radius not in R_VALUES:
        raise ValueError(f"radius must be one of {R_VALUES}")
    n_t_values = _integer_tuple(temporal_extents, "temporal_extents")
    t_values = _integer_tuple(time_extents, "time_extents", minimum=0)

    try:
        from .f_lane_finite_nt_trace_diagnostic import STATIC_ENERGIES, VACUUM_ENERGIES
    except ImportError:  # direct script execution from the repository root
        from tasks.wilson_loop.f_lane_finite_nt_trace_diagnostic import (
            STATIC_ENERGIES, VACUUM_ENERGIES,
        )

    static = np.asarray(
        STATIC_ENERGIES[radius] if static_energies is None else tuple(static_energies),
        dtype=float,
    )
    vacuum = np.asarray(
        VACUUM_ENERGIES if vacuum_energies is None else tuple(vacuum_energies),
        dtype=float,
    )
    if (static.shape != (3,) or vacuum.shape != (2,)
            or not np.isfinite(static).all() or not np.isfinite(vacuum).all()):
        raise ValueError("the toy audit requires three static and two vacuum energies")
    if static[0] < 0 or np.any(np.diff(static) < 0) or vacuum[0] != 0:
        raise ValueError("energies must be ordered, nonnegative, and use vacuum ground zero")
    if np.any(np.diff(vacuum) < 0):
        raise ValueError("vacuum energies must be ordered")
    if any(t >= nt for nt in n_t_values for t in t_values):
        raise ValueError("each selected T must be strictly smaller than every N_t")

    rows = []
    for nt in n_t_values:
        for time_extent in t_values:
            rows.append([
                math.exp(
                    -float(vacuum[m]) * (nt - time_extent)
                    -(float(static[n]) - float(static[0])) * time_extent
                )
                for m, n in MODE_INDICES
            ])
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(MODE_LABELS):
        raise ValueError("correction design matrix has an invalid shape")
    if not np.isfinite(matrix).all():
        raise ValueError("correction design matrix must be finite")
    return matrix


def analyze_design(matrix: object) -> dict[str, Any]:
    """Report raw and unit-column-scaled SVD diagnostics without a pass threshold."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != len(MODE_LABELS):
        raise ValueError("matrix must have shape (n_observations, 5)")
    if not np.isfinite(values).all():
        raise ValueError("matrix must be finite")
    column_norms = np.linalg.norm(values, axis=0)
    if np.any(column_norms <= np.finfo(float).tiny):
        raise ValueError("every correction mode must have nonzero support")
    scaled = values / column_norms
    raw_singular = np.linalg.svd(values, compute_uv=False)
    _, scaled_singular, scaled_vh = np.linalg.svd(scaled, full_matrices=True)
    if raw_singular.size < len(MODE_LABELS):
        raw_singular = np.pad(raw_singular, (0, len(MODE_LABELS) - raw_singular.size))
    if scaled_singular.size < len(MODE_LABELS):
        scaled_singular = np.pad(
            scaled_singular, (0, len(MODE_LABELS) - scaled_singular.size),
        )
    raw_tolerance = (raw_singular[0] * max(values.shape) * np.finfo(float).eps
                     if raw_singular.size else 0.0)
    scaled_tolerance = (scaled_singular[0] * max(scaled.shape) * np.finfo(float).eps
                        if scaled_singular.size else 0.0)
    raw_rank = int(np.count_nonzero(raw_singular > raw_tolerance))
    scaled_rank = int(np.count_nonzero(scaled_singular > scaled_tolerance))
    full_rank = scaled_rank == len(MODE_LABELS)
    return {
        "observation_count": int(values.shape[0]),
        "mode_count": len(MODE_LABELS),
        "numerical_rank_raw": raw_rank,
        "numerical_rank_unit_column_scaled": scaled_rank,
        "raw_rank_tolerance": float(raw_tolerance),
        "unit_column_scaled_rank_tolerance": float(scaled_tolerance),
        "full_column_rank_unit_column_scaled": full_rank,
        "raw_singular_values": raw_singular.tolist(),
        "unit_column_scaled_singular_values": scaled_singular.tolist(),
        "raw_condition_number": (
            float(raw_singular[0] / raw_singular[-1])
            if raw_rank == len(MODE_LABELS)
            else None
        ),
        "unit_column_scaled_condition_number": (
            float(scaled_singular[0] / scaled_singular[-1])
            if full_rank
            else None
        ),
        "column_norms": column_norms.tolist(),
        "weakest_scaled_direction": scaled_vh[-1].tolist(),
        "interpretation": (
            "full numerical column rank at the reported SVD tolerance; no covariance-weighted identifiability threshold was set"
            if full_rank else "rank deficient at the declared SVD tolerance; component weights are not separately identifiable"
        ),
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _designs() -> list[dict[str, Any]]:
    rows = []
    for radius in R_VALUES:
        for extents in N_T_PLANS:
            metrics = analyze_design(correction_design_matrix(radius, extents, T_VALUES))
            rows.append({
                "R": radius,
                "N_t_plan": "+".join(map(str, extents)),
                "T_values": ",".join(map(str, T_VALUES)),
                **metrics,
            })
    return rows


def _degeneracy_stress_cases() -> dict[str, Any]:
    cases = {}
    for label, energies in (
        ("near_degenerate", (0.44, 0.94, 0.940001)),
        ("exactly_degenerate", (0.44, 0.94, 0.94)),
    ):
        matrix = correction_design_matrix(
            2, (16, 32), T_VALUES, static_energies=energies,
        )
        cases[label] = {
            "static_energies_at": list(energies),
            "gap_separation_at": float(energies[2] - energies[1]),
            **analyze_design(matrix),
        }
    return cases


def _plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), constrained_layout=True)
    colors = {"16": "#2878a5", "32": "#d17b0f", "16+32": "#6b4c9a"}
    markers = {2: "o", 3: "s"}
    for row in rows:
        label = f"N_t={row['N_t_plan']}, R={row['R']}"
        axes[0].plot(
            range(1, len(row["unit_column_scaled_singular_values"]) + 1),
            row["unit_column_scaled_singular_values"],
            marker=markers[row["R"]], color=colors[row["N_t_plan"]],
            linestyle="-" if row["R"] == 2 else "--", label=label,
        )
        x = (int(row["N_t_plan"].split("+")[0])
             if "+" not in row["N_t_plan"] else 24)
        axes[1].scatter(
            [x], [row["unit_column_scaled_condition_number"]],
            marker=markers[row["R"]], s=65, color=colors[row["N_t_plan"]],
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Singular-value index")
    axes[0].set_ylabel("Singular value after unit column scaling")
    axes[0].set_title("Mode-shape separation")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_yscale("log")
    axes[1].set_xticks([16, 24, 32], ["16 only", "16 + 32", "32 only"])
    axes[1].set_xlabel("Temporal-extent plan")
    axes[1].set_ylabel("Unit-column-scaled condition number")
    axes[1].set_title("Conditioning after removing amplitude scale")
    axes[1].grid(True, which="both", alpha=0.25)
    fig.suptitle("Fixed-energy q/h/x design audit (synthetic; no covariance whitening)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build(output_dir: Path = DEFAULT_OUT) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    rows = _designs()
    stress_cases = _degeneracy_stress_cases()
    output_dir.mkdir(parents=True)

    csv_path = output_dir / "design_metrics.csv"
    columns = ["R", "N_t_plan", "T_values", "observation_count", "mode_count",
               "numerical_rank_raw", "numerical_rank_unit_column_scaled",
               "raw_rank_tolerance", "unit_column_scaled_rank_tolerance",
               "full_column_rank_unit_column_scaled",
               "raw_condition_number", "unit_column_scaled_condition_number",
               "column_norms", "raw_singular_values",
               "unit_column_scaled_singular_values", "weakest_scaled_direction",
               "interpretation"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key]) if isinstance(row[key], list)
                             else row[key] for key in columns})

    figure = output_dir / "design_singular_values.png"
    _plot(figure, rows)
    sources = {
        "design_diagnostic": Path(__file__),
        "finite_trace_toy": TASK_DIR / "f_lane_finite_nt_trace_diagnostic.py",
        "lane_contract": TASK_DIR / "lane_contract.py",
        "lane_contract_json": TASK_DIR / "lane_contract_v2.json",
    }
    protocol = {
        "schema": "wilson-f-lane-qhx-design-identifiability-v1",
        "classification": "FIXED_ENERGY_LINEAR_DESIGN_AUDIT_ONLY",
        "model": {
            "mode_order": list(MODE_LABELS),
            "mode_indices_mn": [list(pair) for pair in MODE_INDICES],
            "kernel": "K_mn(T,N_t)=exp[-eps_m*(N_t-T)-(V_n-V_0)*T]",
            "baseline": "(m=0,n=0) coefficient fixed to one; five correction coefficients remain",
            "radii": list(R_VALUES),
            "N_t_plans": [list(plan) for plan in N_T_PLANS],
            "T_values": list(T_VALUES),
            "energies_from": "hand-set finite-trace toy; a_t*E units",
        },
        "diagnostics": {
            "raw_svd": "unscaled design matrix; reports physical mode-amplitude suppression and shape together",
            "unit_column_scaled_svd": "each design column normalized to unit Euclidean norm; compares shape collinearity only",
            "numerical_rank": "rank uses max(matrix_shape)*machine_epsilon*s_max separately for raw and scaled matrices",
            "production_covariance": "not supplied; covariance-whitened information and acceptance threshold remain open",
        },
        "degeneracy_stress_cases": {
            "R": 2,
            "N_t_plans": [16, 32],
            "T_values": list(T_VALUES),
            "static_energies_at": "a_t*E",
        },
        "source_sha256": {name: _sha(path) for name, path in sources.items()},
        "limitations": [
            "spectral energies are fixed hand-set toy values, not fitted or measured",
            "linear design rank does not validate the nonlinear GEVP/spectral estimator",
            "unit-column scaling removes absolute signal size and cannot establish practical identifiability",
            "no measurement covariance, noise, bootstrap, interval coverage, or force bound is used",
            "the planned exporter time origin and endpoint factorization still require independent review",
        ],
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": "wilson-f-lane-qhx-design-identifiability-summary-v1",
        "status": "SYNTHETIC_DESIGN_IDENTIFIABILITY_DIAGNOSTIC_ONLY",
        "certified": False,
        "physical_score": None,
        "diagnostic_cells": {"planned": 6, "valid": len(rows), "failed": 0, "abstained": 0},
        "physical_measurements": {"planned": 0, "valid": 0, "failed": 0, "abstained": 0},
        "certification_gate": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
        "results": rows,
        "degeneracy_stress_cases": stress_cases,
    }
    summary["source_hashes"] = protocol["source_sha256"]
    summary["artifact_manifest"] = "artifact_hashes.json"
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    readme = f"""# Finite-`N_t` q/h/x design identifiability (fixed-energy toy)

**Classification: synthetic design audit only.** It evaluates the linear correction kernels implied by the two-sector trace proposal using the companion toy's hand-set energies. It does not fit measurements, estimate noise or covariance, or supply any physical contamination bound.

## Question and data flow

For fixed toy energies, the normalized diagonal trace has five unknown correction amplitudes after fixing the ground-sector coefficient to one:

```text
q_1, q_2: m=0, n=1,2
h:       m=1, n=0
x_1,x_2: m=1, n=1,2
K_mn(T,N_t) = exp[-eps_m*(N_t-T) - (V_n-V_0)*T]

selected (N_t,T) rows -> five-column kernel matrix -> raw SVD
                                           -> unit-column-scaled SVD
```

The selected toy grid uses `T=3..7`, `N_t=16`, `N_t=32`, and both `R=2,3`. It compares a single-extent grid with the matched pair `(16,32)`. The image shows singular values after column scaling and their condition number; the CSV also retains raw condition numbers, numerical rank tolerances, and column norms.

![q/h/x design singular values](design_singular_values.png)

For `R=2`, the unit-column-scaled condition number is `{rows[0]['unit_column_scaled_condition_number']:.3g}` for `N_t=16` alone and `{rows[1]['unit_column_scaled_condition_number']:.3g}` for `N_t=32` alone, versus `{rows[2]['unit_column_scaled_condition_number']:.3g}` for the paired `(16,32)` grid. The paired grid improves the shape-only condition by about `{rows[0]['unit_column_scaled_condition_number']/rows[2]['unit_column_scaled_condition_number']:.1f}x`. Its raw condition number is `{rows[2]['raw_condition_number']:.3g}`; raw single-extent values are `{rows[0]['raw_condition_number']:.3g}` and `{rows[1]['raw_condition_number']:.3g}`.

| `R` | `N_t` plan | Raw condition number | Unit-column-scaled condition number |
|---:|:---|---:|---:|
| 2 | 16 | {rows[0]['raw_condition_number']:.3g} | {rows[0]['unit_column_scaled_condition_number']:.3g} |
| 2 | 32 | {rows[1]['raw_condition_number']:.3g} | {rows[1]['unit_column_scaled_condition_number']:.3g} |
| 2 | 16+32 | {rows[2]['raw_condition_number']:.3g} | {rows[2]['unit_column_scaled_condition_number']:.3g} |
| 3 | 16 | {rows[3]['raw_condition_number']:.3g} | {rows[3]['unit_column_scaled_condition_number']:.3g} |
| 3 | 32 | {rows[4]['raw_condition_number']:.3g} | {rows[4]['unit_column_scaled_condition_number']:.3g} |
| 3 | 16+32 | {rows[5]['raw_condition_number']:.3g} | {rows[5]['unit_column_scaled_condition_number']:.3g} |

The improvement is conditional on the fixed energies and chosen `T` grid. For the hand-set toy, the `N_t=32` thermal/mixed columns are about 830 times smaller in norm than their `N_t=16` counterparts, so unit-column scaling removes a major practical signal penalty. A covariance-weighted information matrix could change both ordering and conditioning; none is supplied here. Numerical ranks use the reported scale-specific SVD tolerance; this is a floating-point rank diagnostic, not an acceptance threshold.

An explicit degeneracy stress check uses the paired grid at `R=2`. With the second and third static energies separated by `1e-6` in `a_t*E`, the five modes remain numerically full rank but the scaled condition number rises to `{stress_cases['near_degenerate']['unit_column_scaled_condition_number']:.3g}`. At exact degeneracy the scaled numerical rank falls to `{stress_cases['exactly_degenerate']['numerical_rank_unit_column_scaled']}` and the condition number is left undefined. This is a failure-mode check for the linear kernels; it does not establish the gap resolution or tolerance needed for production.

## Denominators and limits

- Fixed-energy design cells: planned/valid/failed/abstained = `6/6/0/0` (`R=2,3` × three `N_t` plans).
- Physical measurements: `0/0/0/0`; certification gate: `1/0/0/1`.
- The weakest singular vector and all singular values are in `design_metrics.csv`; source versions and generator parameters are in `protocol.json` and `artifact_hashes.json`.

This result suggests the next fitter should report raw and covariance-whitened profile information, plus column-scaled diagnostics, and should test exact/near-degenerate gaps. It does **not** validate the selected `N_t=32` thermal receipt, show that h/x are resolvable above production noise, or close the F-lane fit/coverage gate. Keep `NOT_READY/UNCERTIFIED`, `GROUND_STATE_FORCE_UNCERTIFIED`, and `physical_score=null`.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    hashes = {
        path.name: _sha(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps({"schema": "latticearena-artifact-hashes-v1", "files": hashes},
                   indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["artifact_hashes"] = hashes
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.out), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
