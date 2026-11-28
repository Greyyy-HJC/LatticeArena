"""Correlation-only covariance stress for the finite-N_t q/h/x design.

All row variances are fixed to one.  The covariance is a hypothetical
separable correlation model over T and, for the paired design, an equicorrelation
between N_t=16 and N_t=32 observations.  This is a design sensitivity study,
not an estimate of production covariance or a physical identifiability claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = TASK_DIR / "f_lane_qhx_covariance_stress_20261126_v1"
R_VALUES = (2, 3)
T_VALUES = tuple(range(3, 8))
N_T_PLANS = ((16,), (32,), (16, 32))
T_CORRELATIONS = (-0.9, -0.5, 0.0, 0.5, 0.9)
N_T_CORRELATIONS = (-0.9, 0.0, 0.9)


try:
    from .f_lane_qhx_design_identifiability import analyze_design, correction_design_matrix
except ImportError:  # direct script execution from the repository root
    from tasks.wilson_loop.f_lane_qhx_design_identifiability import (
        analyze_design, correction_design_matrix,
    )


def separable_correlation(
    temporal_extents: tuple[int, ...],
    time_extents: tuple[int, ...],
    *,
    temporal_rho: float,
    cross_extent_rho: float,
) -> np.ndarray:
    """Construct R_Nt ⊗ R_T with unit marginal variances."""
    if temporal_extents not in N_T_PLANS:
        raise ValueError("temporal_extents must be one of the registered N_t plans")
    if time_extents != T_VALUES:
        raise ValueError("time_extents must equal the selected T grid")
    for name, value in (("temporal_rho", temporal_rho), ("cross_extent_rho", cross_extent_rho)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number strictly between -1 and 1")
        if not math.isfinite(float(value)) or abs(float(value)) >= 1.0:
            raise ValueError(f"{name} must be a finite number strictly between -1 and 1")

    time_indices = np.asarray(time_extents, dtype=np.int64)
    time_corr = float(temporal_rho) ** np.abs(np.subtract.outer(time_indices, time_indices))
    if len(temporal_extents) == 1:
        extent_corr = np.ones((1, 1), dtype=float)
    else:
        extent_corr = np.asarray(
            [[1.0, float(cross_extent_rho)], [float(cross_extent_rho), 1.0]],
            dtype=float,
        )
    return np.kron(extent_corr, time_corr)


def whiten_design(matrix: object, covariance: object) -> np.ndarray:
    """Return L^-1 X for an explicitly supplied SPD observation covariance."""
    values = np.asarray(matrix, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != 5:
        raise ValueError("matrix must have shape (n_observations, 5)")
    if not np.isfinite(values).all():
        raise ValueError("matrix must be finite")
    if cov.shape != (values.shape[0], values.shape[0]) or not np.isfinite(cov).all():
        raise ValueError("covariance must be finite and square with one row per observation")
    if not np.allclose(cov, cov.T, rtol=1e-12, atol=1e-14):
        raise ValueError("covariance must be symmetric")
    try:
        lower = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite; no regularization is applied") from exc
    return np.linalg.solve(lower, values)


def _scenario_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for radius in R_VALUES:
        for rho_t in T_CORRELATIONS:
            single_reference = None
            for plan in N_T_PLANS:
                if len(plan) != 1:
                    continue
                design = correction_design_matrix(radius, plan, T_VALUES)
                covariance = separable_correlation(
                    plan, T_VALUES, temporal_rho=rho_t, cross_extent_rho=0.0,
                )
                metric = analyze_design(whiten_design(design, covariance))
                if single_reference is None:
                    single_reference = metric["unit_column_scaled_condition_number"]
                rows.append({
                    "R": radius,
                    "N_t_plan": str(plan[0]),
                    "temporal_rho": rho_t,
                    "cross_extent_rho": None,
                    "unit_column_scaled_condition_number": metric[
                        "unit_column_scaled_condition_number"
                    ],
                    "raw_condition_number": metric["raw_condition_number"],
                    "numerical_rank": metric["numerical_rank_unit_column_scaled"],
                    "single_extent_reference_condition_number": single_reference,
                    "shape_condition_improvement_factor": 1.0,
                })

            paired = (16, 32)
            design = correction_design_matrix(radius, paired, T_VALUES)
            for rho_nt in N_T_CORRELATIONS:
                covariance = separable_correlation(
                    paired, T_VALUES, temporal_rho=rho_t,
                    cross_extent_rho=rho_nt,
                )
                metric = analyze_design(whiten_design(design, covariance))
                rows.append({
                    "R": radius,
                    "N_t_plan": "16+32",
                    "temporal_rho": rho_t,
                    "cross_extent_rho": rho_nt,
                    "unit_column_scaled_condition_number": metric[
                        "unit_column_scaled_condition_number"
                    ],
                    "raw_condition_number": metric["raw_condition_number"],
                    "numerical_rank": metric["numerical_rank_unit_column_scaled"],
                    "single_extent_reference_condition_number": single_reference,
                    "shape_condition_improvement_factor": (
                        single_reference / metric["unit_column_scaled_condition_number"]
                    ),
                })
    expected_rows = len(R_VALUES) * len(T_CORRELATIONS) * (
        len(N_T_PLANS) - 1 + len(N_T_CORRELATIONS)
    )
    if len(rows) != expected_rows:
        raise RuntimeError("scenario denominator does not match the frozen grid")
    return rows


def _plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    rho_t_labels = [f"{value:g}" for value in T_CORRELATIONS]
    rho_nt_labels = [f"{value:g}" for value in N_T_CORRELATIONS]
    for row_index, radius in enumerate(R_VALUES):
        selected = [row for row in rows if row["R"] == radius and row["N_t_plan"] == "16+32"]
        conditions = np.asarray([
            [next(row["unit_column_scaled_condition_number"] for row in selected
                  if row["temporal_rho"] == rho_t and row["cross_extent_rho"] == rho_nt)
             for rho_nt in N_T_CORRELATIONS]
            for rho_t in T_CORRELATIONS
        ])
        improvements = np.asarray([
            [next(row["shape_condition_improvement_factor"] for row in selected
                  if row["temporal_rho"] == rho_t and row["cross_extent_rho"] == rho_nt)
             for rho_nt in N_T_CORRELATIONS]
            for rho_t in T_CORRELATIONS
        ])
        for column, (data, title, colorbar) in enumerate((
            (conditions, f"R={radius}: paired whitened condition", "Condition number"),
            (improvements, f"R={radius}: gain over one N_t", "Single / paired condition"),
        )):
            ax = axes[row_index, column]
            plotted = np.log10(data) if column == 0 else np.log10(data)
            image = ax.imshow(plotted, origin="lower", aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(rho_nt_labels)), rho_nt_labels)
            ax.set_yticks(range(len(rho_t_labels)), rho_t_labels)
            ax.set_xlabel("Assumed cross-N_t correlation")
            ax.set_ylabel("Assumed adjacent-T correlation")
            ax.set_title(title + " (log10 scale)")
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(j, i, f"{data[i, j]:.2g}", ha="center", va="center",
                            color="white" if plotted[i, j] < plotted.max() * 0.72 else "black",
                            fontsize=8)
            fig.colorbar(image, ax=ax, label=f"log10 {colorbar}")
    fig.suptitle("Hypothetical covariance-correlation stress (unit variances; synthetic design only)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output_dir: Path = DEFAULT_OUT) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    rows = _scenario_rows()
    output_dir.mkdir(parents=True)
    fields = [
        "R", "N_t_plan", "temporal_rho", "cross_extent_rho",
        "unit_column_scaled_condition_number", "raw_condition_number",
        "numerical_rank", "single_extent_reference_condition_number",
        "shape_condition_improvement_factor",
    ]
    with (output_dir / "covariance_scenarios.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _plot(output_dir / "covariance_condition_heatmap.png", rows)

    paired_rows = [row for row in rows if row["N_t_plan"] == "16+32"]
    improvement = [row["shape_condition_improvement_factor"] for row in paired_rows]
    source_paths = {
        "covariance_diagnostic": Path(__file__),
        "base_design_diagnostic": TASK_DIR / "f_lane_qhx_design_identifiability.py",
        "finite_trace_toy": TASK_DIR / "f_lane_finite_nt_trace_diagnostic.py",
    }
    protocol = {
        "schema": "wilson-f-lane-qhx-covariance-stress-v1",
        "classification": "HYPOTHETICAL_CORRELATION_ONLY_DESIGN_STRESS_NOT_PRODUCTION_COVARIANCE",
        "model": {
            "covariance": "Sigma = R_Nt ⊗ R_T, with unit marginal variances",
            "R_T": "rho_T^abs(T_i-T_j), using T=3..7",
            "R_Nt": "[[1,rho_Nt],[rho_Nt,1]] for paired N_t=16,32; identity for single extent",
            "rho_T_values": list(T_CORRELATIONS),
            "rho_Nt_values": list(N_T_CORRELATIONS),
            "radii": list(R_VALUES),
            "N_t_plans": [list(plan) for plan in N_T_PLANS],
            "whitening": "L^-1 X where Sigma=L L^T; no regularization or covariance fitting",
            "condition_metric": "unit-column-scaled condition number after covariance whitening; reports shape only",
        },
        "source_sha256": {name: _sha(path) for name, path in source_paths.items()},
        "limitations": [
            "all hypothetical observation variances are equal; only correlation changes",
            "the separable covariance family is illustrative and is not estimated from raw blocks",
            "unit-column scaling removes absolute q/h/x signal amplitude",
            "no measured covariance, fit, bootstrap, interval coverage, or force bound is used",
            "the underlying energies remain the hand-set values from the companion toy",
        ],
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": "wilson-f-lane-qhx-covariance-stress-summary-v1",
        "status": "HYPOTHETICAL_COVARIANCE_CORRELATION_SENSITIVITY_ONLY",
        "certified": False,
        "physical_score": None,
        "scenario_denominator": {
            "planned": len(rows), "valid": len(rows), "failed": 0, "abstained": 0,
        },
        "physical_measurements": {"planned": 0, "valid": 0, "failed": 0, "abstained": 0},
        "certification_gate": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
        "paired_shape_condition_improvement_factor_range": [min(improvement), max(improvement)],
        "results": rows,
        "source_hashes": protocol["source_sha256"],
        "artifact_manifest": "artifact_hashes.json",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = f"""# F-lane q/h/x covariance-correlation sensitivity

**Synthetic design stress only.** This report asks whether the paired `N_t=16+32` grid keeps a better column-shape condition number than either single-extent grid under a small, declared family of hypothetical observation correlations.

## Assumed covariance and calculation

All observations are assigned unit variance. The correlation model is separable:

```text
Sigma = R_Nt ⊗ R_T
R_T[i,j] = rho_T^|T_i-T_j|, T=3..7
R_Nt = [[1,rho_Nt],[rho_Nt,1]] for the paired extent grid
W = L^-1 K, where Sigma = L L^T
```

The scan uses `rho_T={list(T_CORRELATIONS)}` and `rho_Nt={list(N_T_CORRELATIONS)}`, both inside the positive-definite range. It varies correlation only; it does not guess row-dependent variances or tune the covariance to make either design look better. `covariance_scenarios.csv` includes raw and column-scaled whitened metrics for each radius and plan.

![Whitened design conditioning across assumed correlations](covariance_condition_heatmap.png)

Across the 30 paired scenarios, the shape-condition improvement factor (single extent / paired extents) ranges from **{min(improvement):.2f}× to {max(improvement):.2f}×**. This is conditional on equal marginal variances, the separable correlation family, and the companion toy energies. The heatmap's lower panels show the same factor; upper panels show the paired whitened condition number itself.

## Interpretation limits and denominators

This sensitivity scan says that the `N_t=16+32` column shapes remain better separated across these hypothetical correlation patterns. It does not say which pattern describes a Wilson ensemble. It also cannot restore absolute signal lost from the `N_t=32` thermal/mixed columns: column scaling removes amplitude, and the covariance variances were set to one by construction. Production decisions require covariance estimated from authorized raw matrix blocks and the existing force-unit budget.

- Hypothetical covariance scenarios: `{len(rows)}/{len(rows)}/0/0` (planned/valid/failed/abstained).
- Physical measurements: `0/0/0/0`; certification gate: `1/0/0/1`.
- No spectral fit, bootstrap coverage, q/h/x interval, or force bound is calculated.
- Keep `NOT_READY/UNCERTIFIED`, `GROUND_STATE_FORCE_UNCERTIFIED`, and `physical_score=null`.

The next fitter should estimate the block covariance from one immutable raw-matrix snapshot, whiten the full real/imaginary matrix-element residual vector, then report raw and whitened profile information alongside the unscaled mode norms. No covariance floor or acceptance threshold is selected by this report; singular or unstable production covariance must remain an abstention until a reviewed regularization rule exists.
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
