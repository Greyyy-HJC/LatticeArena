"""Analytic two-state bias sensitivity for a proposed Pion anchor scalar map.

This data-free diagnostic studies the fixed folded-PP cosh effective-mass
map at t=4 on T=16. It does not read measurements and does not establish the
anchor common-bias bound required by the bounded block-risk contract.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCHEMA = "pion-anchor-scalar-map-two-state-bias-v1"
T_EXTENT = 16
MAP_TIME = 4
SUPPORT = (0.1, 3.0)
RISK_SCALE = 2.9
COMMON_BIAS_LIMIT = 0.01 * RISK_SCALE
DEFAULT_OUT = Path(__file__).resolve().parent / "anchor_scalar_map_bias_20261126_v2"


def folded_pp(t: int, mass0: float, mass1: float, amplitude_ratio: float) -> float:
    """Unit-ground-amplitude positive two-state periodic PP correlator."""
    if (type(t) is not int or not 0 <= t <= T_EXTENT
            or not all(math.isfinite(x) for x in (mass0, mass1, amplitude_ratio))
            or mass0 <= 0 or mass1 < mass0 or amplitude_ratio < 0):
        raise ValueError("invalid positive two-state spectral parameters")

    def cosh_kernel(mass: float) -> float:
        return math.exp(-mass * t) + math.exp(-mass * (T_EXTENT - t))

    return cosh_kernel(mass0) + amplitude_ratio * cosh_kernel(mass1)


def effective_mass_at_four(mass0: float, gap: float, amplitude_ratio: float) -> float:
    """Return acosh[(C(3)+C(5))/(2 C(4))] for the fixed T=16 map."""
    if not math.isfinite(gap) or gap < 0:
        raise ValueError("gap must be finite and non-negative")
    mass1 = mass0 + gap
    if not SUPPORT[0] <= mass0 <= mass1 <= SUPPORT[1]:
        raise ValueError("both spectral masses must lie in the proposed anchor support")
    c3 = folded_pp(3, mass0, mass1, amplitude_ratio)
    c4 = folded_pp(4, mass0, mass1, amplitude_ratio)
    c5 = folded_pp(5, mass0, mass1, amplitude_ratio)
    argument = (c3 + c5) / (2.0 * c4)
    if not math.isfinite(argument) or argument < 1.0:
        raise ValueError("folded-PP cosh effective-mass argument is invalid")
    mapped = math.acosh(argument)
    if not SUPPORT[0] <= mapped <= SUPPORT[1]:
        raise ValueError("mapped anchor scalar falls outside the frozen support")
    return mapped


def max_overlap_ratio_for_bias(mass0: float, gap: float, limit: float) -> float:
    """Largest A1/A0 in [1e-12, 1e8] whose noiseless map bias is <= limit."""
    if gap <= limit:
        return 1e8
    lo, hi = -12.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        ratio = 10.0 ** mid
        bias = effective_mass_at_four(mass0, gap, ratio) - mass0
        if bias <= limit:
            lo = mid
        else:
            hi = mid
    return 10.0 ** lo


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing report directory: {output}")
    output.mkdir(parents=True)

    mass0_grid = (0.1, 0.25, 0.5, 1.0, 2.0)
    gap_grid = np.linspace(0.01, 2.9, 90)
    ratio_grid = np.logspace(-8, 4, 81)
    rows: list[dict[str, float | bool]] = []
    for mass0 in mass0_grid:
        for gap in gap_grid:
            if mass0 + gap > SUPPORT[1]:
                continue
            for ratio in ratio_grid:
                mapped = effective_mass_at_four(float(mass0), float(gap), float(ratio))
                bias = mapped - float(mass0)
                rows.append({
                    "mass0": float(mass0),
                    "gap": float(gap),
                    "amplitude_ratio_A1_over_A0": float(ratio),
                    "mapped_meff_t4": mapped,
                    "noiseless_spectral_bias": bias,
                    "passes_C_over_R_0p01": bool(bias <= COMMON_BIAS_LIMIT),
                })

    with (output / "bias_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    threshold_rows = []
    for mass0 in (0.1, 0.25, 0.5, 1.0, 2.0):
        max_gap = SUPPORT[1] - mass0
        for gap in np.linspace(0.005, max_gap, 120):
            threshold_rows.append({
                "mass0": mass0,
                "gap": float(gap),
                "max_A1_over_A0_for_C_le_0p029": max_overlap_ratio_for_bias(
                    mass0, float(gap), COMMON_BIAS_LIMIT,
                ),
                "gap_below_bias_limit": bool(gap <= COMMON_BIAS_LIMIT),
            })
    with (output / "overlap_thresholds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(threshold_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(threshold_rows)

    example_thresholds = []
    for mass0 in (0.25, 0.5, 1.0):
        for gap in (0.05, 0.10, 0.25, 0.50, 1.00):
            if mass0 + gap <= SUPPORT[1]:
                example_thresholds.append({
                    "mass0": mass0,
                    "gap": gap,
                    "max_A1_over_A0": max_overlap_ratio_for_bias(
                        mass0, gap, COMMON_BIAS_LIMIT,
                    ),
                })
    threshold_table = "\n".join(
        f"| {item['mass0']:.2f} | {item['gap']:.2f} | {item['max_A1_over_A0']:.3f} |"
        for item in example_thresholds
    )

    # Heatmap at a representative dimensionless ground mass; all values are
    # an analytic response surface, not measured ensemble estimates.
    representative_mass = 0.5
    gaps = np.linspace(0.01, min(2.0, SUPPORT[1] - representative_mass), 100)
    ratios = np.logspace(-8, 4, 121)
    bias_surface = np.empty((ratios.size, gaps.size))
    for iy, ratio in enumerate(ratios):
        for ix, gap in enumerate(gaps):
            bias_surface[iy, ix] = effective_mass_at_four(
                representative_mass, float(gap), float(ratio),
            ) - representative_mass

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    mesh = axes[0].pcolormesh(
        gaps, np.log10(ratios), bias_surface, shading="auto", cmap="viridis",
    )
    axes[0].contour(
        gaps, np.log10(ratios), bias_surface,
        levels=[COMMON_BIAS_LIMIT], colors=["white"], linewidths=2,
    )
    axes[0].axvline(COMMON_BIAS_LIMIT, color="#ffb000", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel(r"excited-state gap $\Delta=m_1-m_0$ ($a_t$ units)")
    axes[0].set_ylabel(r"$\log_{10}(A_1/A_0)$")
    axes[0].set_title(rf"Noiseless $m_{{eff}}(4)-m_0$, $m_0={representative_mass}$")
    colorbar = fig.colorbar(mesh, ax=axes[0])
    colorbar.set_label(r"bias ($a_t m_\pi$ units)")
    axes[0].text(
        0.03, 0.03, "white: bias = 0.029\norange: gap = 0.029",
        transform=axes[0].transAxes, color="white", fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 4},
    )

    for mass0 in (0.1, 0.25, 0.5, 1.0, 2.0):
        max_gap = SUPPORT[1] - mass0
        curve_gaps = np.linspace(0.005, max_gap, 160)
        thresholds = [
            max_overlap_ratio_for_bias(float(mass0), float(gap), COMMON_BIAS_LIMIT)
            for gap in curve_gaps
        ]
        axes[1].plot(curve_gaps, np.log10(thresholds), label=rf"$m_0={mass0:g}$")
    axes[1].axvline(COMMON_BIAS_LIMIT, color="#d62728", linestyle="--", linewidth=1.5,
                    label=r"$C=0.029$")
    axes[1].set_xlabel(r"excited-state gap $\Delta$ ($a_t$ units)")
    axes[1].set_ylabel(r"largest $\log_{10}(A_1/A_0)$ passing noiseless bias gate")
    axes[1].set_title("Required overlap suppression for C/R ≤ 0.01")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2, fontsize=8)
    figure = output / "anchor_map_bias.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    passing = sum(bool(row["passes_C_over_R_0p01"]) for row in rows)
    summary = {
        "schema": SCHEMA,
        "classification": "DATA_FREE_ANALYTIC_SENSITIVITY_ONLY",
        "map": {
            "formula": "acosh((C_fold(3)+C_fold(5))/(2*C_fold(4)))",
            "channel": "zero-momentum folded PP",
            "temporal_extent": T_EXTENT,
            "map_time": MAP_TIME,
            "support": list(SUPPORT),
            "invalid_policy": "ABSTAIN; no clipping and no dropped anchor states",
        },
        "spectral_model": "C_fold(t)=sum_{n=0,1} A_n*(exp(-m_n*t)+exp(-m_n*(16-t))); A0>0, A1>=0",
        "target": "m0, the exact two-state ground mass; compared with proposed common-bias cap C=0.029 a_t m_pi",
        "grid": {
            "mass0_values": len(mass0_grid),
            "gap_values": int(gap_grid.size),
            "overlap_ratio_values": int(ratio_grid.size),
            "valid_spectral_cells": len(rows),
            "cells_passing_noiseless_C_cap": passing,
            "pass_fraction": passing / len(rows),
        },
        "example_overlap_thresholds": example_thresholds,
        "limitations": [
            "The response surface uses the ensemble-mean exact two-state spectrum, not random per-block correlators.",
            "It omits stochastic nonlinearity E[m_eff(C_block)]-m_eff(E[C_block]), thermal/model mismatch, fit uncertainty, and chain dependence.",
            "Therefore it cannot certify the anchor common-bias cap C<=0.029 or Markov coverage.",
            "A real anchor map must be fixed per block, preserve all invalid states by abstaining, and receive an independent bias/support receipt.",
        ],
        "physical_score": None,
        "certified": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme = f"""# Pion anchor scalar-map bias sensitivity

**Data-free analytic sensitivity only.** No correlator, anchor chain, reference chain, validation/holdout split, propagator cache, GPU or QUDA input was read. This report does not certify `C<=0.029 a_t m_pi` or Markov coverage and keeps `physical_score=null`.

## Candidate map

For each predeclared anchor block, form the zero-momentum folded pseudoscalar correlator and calculate

```text
m_eff(4) = acosh((C_fold(3) + C_fold(5)) / (2 C_fold(4)))
```

The candidate support is `[0.1, 3.0]` in `a_t m_pi`. A nonpositive denominator, argument below one, nonfinite value, or mapped value outside support is an abstention; values are never clipped or silently dropped. This choice follows the fixed `T=16` folded-PP convention. It is only a candidate map for independent review.

## Two-state diagnostic

The analytic scan uses

```text
C_fold(t) = A0 [e^(-m0 t) + e^(-m0 (16-t))]
          + A1 [e^(-m1 t) + e^(-m1 (16-t))]
m1 = m0 + Delta,  A0 > 0, A1 >= 0
```

The proposed common-bias cap is `C=0.01 R=0.029 a_t m_pi`. The left panel shows the noiseless spectral shift `m_eff(4)-m0` at `m0=0.5`; the white contour marks the cap. The right panel gives the largest excited/ground amplitude ratio `A1/A0` that passes this **noiseless two-state** cap at each gap and selected `m0` values.

![Two-state bias surface and required excited-state suppression](anchor_map_bias.png)

The reproducible grid is in [`bias_grid.csv`](bias_grid.csv), with threshold curves in [`overlap_thresholds.csv`](overlap_thresholds.csv). Exact grid sizes, passing fraction, formulas and limitations are in [`summary.json`](summary.json). Rebuild in a new directory with:

```bash
python -m tasks.pion_projection.anchor_scalar_map_bias_diagnostic --output NEW_DIRECTORY
```

Selected noiseless thresholds are shown below. They are required upper limits on `A1/A0` for the two-state mean correlator to pass the proposed `C=0.029` cap; they are not measured overlap ratios.

| `m0` | `Delta` | maximum `A1/A0` |
|---:|---:|---:|
{threshold_table}

## Interpretation and gate

When `Delta > 0.029`, the map needs a sufficiently small `A1/A0`; the allowed ratio depends on both the gap and `m0`. The support interval by itself only gives a possible discrepancy as large as `2.9`, so it cannot establish the proposed cap. Even a passing point on this diagram is not a real receipt: random blockwise effective-mass nonlinearity, thermal/model mismatch, and the difference between this PP map and the candidate-blind joint PP/A4P reference remain outside this calculation. A reviewer must require an independent overlap/gap envelope and demonstrate that the **expectation of the fixed block map** differs from the joint-fit target by at most `0.029`; otherwise the anchor map remains rejected and the Pion gate abstains.

Text flow: `freeze folded-PP block map -> positive two-state T=16 scan -> plot bias and required A1/A0 envelope -> check missing stochastic/model terms -> no receipt => ABSTAIN`. Formula grid planned/valid/failed/abstained is `{len(rows)}/{len(rows)}/0/0`; physical anchor receipts are `0/0/0/1` and physical scoring remains withheld.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    shutil.copy2(Path(__file__), output / "source_snapshot.py")
    artifact_names = ["README.md", "anchor_map_bias.png", "bias_grid.csv",
                      "overlap_thresholds.csv", "summary.json", "source_snapshot.py"]
    hashes = {name: _sha256(output / name) for name in artifact_names}
    (output / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return {**summary, "artifact_hashes": hashes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.output)
    print(json.dumps({
        "schema": result["schema"],
        "classification": result["classification"],
        "grid": result["grid"],
        "physical_score": result["physical_score"],
        "artifacts": sorted(result["artifact_hashes"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
