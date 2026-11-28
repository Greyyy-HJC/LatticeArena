"""Exact anchor-interval coverage under bounded, non-Gaussian distributions.

This CPU-only diagnostic compares a known-variance Gaussian z interval with a
distribution-free Hoeffding interval for bounded iid anchor observations. The
finite two-point grids are enumerated exactly through their Binomial count
distribution; no random or physical measurements are read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
from pathlib import Path
from statistics import NormalDist

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / (
    "results/pion_projection/"
    "bounded_block_risk_anchor_robustness_exact_20261123_v1"
)
RARE_PROBABILITIES = {
    "symmetric_two_point": 0.5,
    "moderate_skew_two_point": 0.1,
    "rare_spike_two_point": 0.005,
}
ANCHOR_SAMPLE_SIZES = (16, 32, 128, 512)
ANCHOR_SIGMA = 0.05
ALPHA_ANCHOR = 0.01


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def two_point_support(*, rare_probability: float,
                      sigma: float = ANCHOR_SIGMA) -> tuple[float, float]:
    """Return two support points with mean zero and variance sigma squared.

    The upper point occurs with ``rare_probability``. These are synthetic
    dimensionless anchor errors; they are not inferred from a correlator.
    """
    if (not math.isfinite(rare_probability)
            or not 0 < rare_probability < 1):
        raise ValueError("rare_probability must be finite and in (0, 1)")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    upper = sigma * math.sqrt((1.0 - rare_probability) / rare_probability)
    lower = -sigma * math.sqrt(rare_probability / (1.0 - rare_probability))
    return lower, upper


def normal_z_radius(*, samples: int, sigma: float = ANCHOR_SIGMA,
                    alpha: float = ALPHA_ANCHOR) -> float:
    """Oracle-sigma Gaussian z radius; non-Gaussian use is a comparator only."""
    n = _positive_integer(samples, "samples")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and in (0, 1)")
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    return z * sigma / math.sqrt(n)


def hoeffding_radius(*, samples: int, lower: float, upper: float,
                     alpha: float = ALPHA_ANCHOR) -> float:
    """Two-sided Hoeffding radius for iid observations in [lower, upper]."""
    n = _positive_integer(samples, "samples")
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("support must be finite with lower < upper")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and in (0, 1)")
    width = upper - lower
    return width * math.sqrt(math.log(2.0 / alpha) / (2.0 * n))


def binomial_probabilities(*, samples: int,
                           rare_probability: float) -> np.ndarray:
    """Stable recurrence for P(K=k), K~Binomial(samples, p), normalized."""
    n = _positive_integer(samples, "samples")
    if (not math.isfinite(rare_probability)
            or not 0 < rare_probability < 1):
        raise ValueError("rare_probability must be finite and in (0, 1)")
    p = rare_probability
    probabilities = np.empty(n + 1, dtype=np.float64)
    probabilities[0] = (1.0 - p) ** n
    for k in range(n):
        probabilities[k + 1] = (
            probabilities[k] * (n - k) / (k + 1) * p / (1.0 - p)
        )
    total = float(probabilities.sum())
    if not math.isfinite(total) or total <= 0:
        raise ArithmeticError("binomial probability recurrence failed")
    probabilities /= total
    return probabilities


def exact_coverage(*, samples: int, rare_probability: float,
                   radius: float, sigma: float = ANCHOR_SIGMA
                   ) -> tuple[float, float]:
    """Enumerate exact count outcomes and return coverage and total mass."""
    n = _positive_integer(samples, "samples")
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    lower, upper = two_point_support(
        rare_probability=rare_probability, sigma=sigma,
    )
    weights = binomial_probabilities(
        samples=n, rare_probability=rare_probability,
    )
    counts = np.arange(n + 1, dtype=np.float64)
    sample_means = (counts * upper + (n - counts) * lower) / n
    covered = float(weights[np.abs(sample_means) <= radius].sum())
    return covered, float(weights.sum())


def analyze_grid() -> list[dict]:
    rows = []
    for distribution, p in RARE_PROBABILITIES.items():
        lower, upper = two_point_support(rare_probability=p)
        width = upper - lower
        mean = p * upper + (1.0 - p) * lower
        variance = p * (upper - mean) ** 2 + (1.0 - p) * (lower - mean) ** 2
        for n in ANCHOR_SAMPLE_SIZES:
            radius_z = normal_z_radius(samples=n)
            radius_h = hoeffding_radius(
                samples=n, lower=lower, upper=upper,
            )
            coverage_z, mass_z = exact_coverage(
                samples=n, rare_probability=p, radius=radius_z,
            )
            coverage_h, mass_h = exact_coverage(
                samples=n, rare_probability=p, radius=radius_h,
            )
            rows.append({
                "distribution": distribution,
                "rare_probability": p,
                "anchor_sample_size": n,
                "true_mean": mean,
                "true_variance": variance,
                "true_sigma": math.sqrt(variance),
                "support_lower": lower,
                "support_upper": upper,
                "support_width": width,
                "alpha_anchor": ALPHA_ANCHOR,
                "nominal_coverage": 1.0 - ALPHA_ANCHOR,
                "normal_z_radius_oracle_sigma": radius_z,
                "normal_z_exact_coverage": coverage_z,
                "normal_z_total_probability_mass": mass_z,
                "hoeffding_radius_known_support": radius_h,
                "hoeffding_exact_coverage": coverage_h,
                "hoeffding_total_probability_mass": mass_h,
                "hoeffding_to_normal_radius_ratio": radius_h / radius_z,
                "planned_anchor_observations": n,
                "valid_anchor_observations": n,
                "failed_anchor_observations": 0,
                "abstained_anchor_observations": 0,
                "planned_enumerated_count_states": n + 1,
                "valid_enumerated_count_states": n + 1,
                "failed_enumerated_count_states": 0,
                "abstained_enumerated_count_states": 0,
                "classification": "exact synthetic two-point distribution enumeration",
            })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_denominators(path: Path, rows: list[dict]) -> list[dict]:
    denominator_rows = []
    for row in rows:
        cell = f"{row['distribution']}_nA{row['anchor_sample_size']}"
        for unit, planned_key, valid_key, failed_key, abstained_key in (
            ("hypothetical anchor observation", "planned_anchor_observations",
             "valid_anchor_observations", "failed_anchor_observations",
             "abstained_anchor_observations"),
            ("exact binomial count state", "planned_enumerated_count_states",
             "valid_enumerated_count_states", "failed_enumerated_count_states",
             "abstained_enumerated_count_states"),
        ):
            denominator_rows.append({
                "cell": cell,
                "unit": unit,
                "planned": row[planned_key],
                "valid": row[valid_key],
                "failed": row[failed_key],
                "abstained": row[abstained_key],
                "probability_mass": (
                    row["normal_z_total_probability_mass"]
                    if unit == "exact binomial count state" else None
                ),
                "interpretation": (
                    "Hypothetical iid observations; not measured data."
                    if unit == "hypothetical anchor observation"
                    else "All n+1 count states are enumerated and probability weighted; not MC replicates."
                ),
            })
    _write_csv(path, denominator_rows)
    return denominator_rows


def _write_plot(out: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "symmetric_two_point": "#0072B2",
        "moderate_skew_two_point": "#D55E00",
        "rare_spike_two_point": "#009E73",
    }
    labels = {
        "symmetric_two_point": "symmetric (p=0.5)",
        "moderate_skew_two_point": "skewed (p=0.1)",
        "rare_spike_two_point": "rare spike (p=0.005)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
    for distribution in RARE_PROBABILITIES:
        selected = [r for r in rows if r["distribution"] == distribution]
        ns = [r["anchor_sample_size"] for r in selected]
        color = colors[distribution]
        axes[0].plot(
            ns, [r["normal_z_exact_coverage"] for r in selected],
            marker="o", linestyle="--", color=color,
            label=f"{labels[distribution]}: z comparator",
        )
        axes[0].plot(
            ns, [r["hoeffding_exact_coverage"] for r in selected],
            marker="s", linestyle="-", color=color,
            label=f"{labels[distribution]}: Hoeffding",
        )
        ratio = selected[0]["hoeffding_to_normal_radius_ratio"]
        axes[1].plot(
            ns, [ratio] * len(ns), marker="o", color=color,
            label=labels[distribution],
        )

    axes[0].axhline(1.0 - ALPHA_ANCHOR, color="black", ls=":",
                    label="nominal 99%")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(ANCHOR_SAMPLE_SIZES,
                       labels=[str(n) for n in ANCHOR_SAMPLE_SIZES])
    axes[0].set_ylim(0.88, 1.002)
    axes[0].set_xlabel("hypothetical independent anchor sample size")
    axes[0].set_ylabel("exact interval coverage")
    axes[0].set_title("Exact coverage under bounded non-Gaussian anchors")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(ANCHOR_SAMPLE_SIZES,
                       labels=[str(n) for n in ANCHOR_SAMPLE_SIZES])
    axes[1].set_xlabel("hypothetical independent anchor sample size")
    axes[1].set_ylabel("Hoeffding radius / Gaussian z radius")
    axes[1].set_title("Width cost when the support is known")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Synthetic anchor calibration: model-based z interval vs bounded-data bound\n"
        "Known variance and support are oracle inputs; alpha_anchor=0.01",
        fontsize=11,
    )
    fig.savefig(out / "coverage_and_radius_cost.png", dpi=180)
    plt.close(fig)


def build(output: str | Path = DEFAULT_OUT) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows = analyze_grid()
    _write_csv(out / "coverage_summary.csv", rows)
    denominator_rows = _write_denominators(
        out / "denominator_summary.csv", rows,
    )

    protocol = {
        "schema": "pion-bounded-anchor-robustness-protocol-v1",
        "status": "SYNTHETIC_EXACT_ENUMERATION_ONLY",
        "physical_score": None,
        "data_access": "No physical anchor, correlator, ensemble, sealed split, GPU, or QUDA data accessed.",
        "distributions": {
            name: {
                "rare_upper_point_probability": p,
                "mean": 0.0,
                "variance": ANCHOR_SIGMA**2,
                "support": list(two_point_support(rare_probability=p)),
            }
            for name, p in RARE_PROBABILITIES.items()
        },
        "sample_sizes": list(ANCHOR_SAMPLE_SIZES),
        "alpha_anchor": ALPHA_ANCHOR,
        "sigma_oracle": ANCHOR_SIGMA,
        "normal_comparator_radius": "z_(1-alpha_A/2)*sigma/sqrt(n)",
        "hoeffding_radius": "(upper-lower)*sqrt(log(2/alpha_A)/(2*n))",
        "coverage_method": "Enumerate all n+1 Binomial count states and sum exact probability mass.",
        "assumptions": [
            "iid anchor observations within each hypothetical sample",
            "two-point synthetic support is exactly known for Hoeffding radius",
            "oracle variance is exactly known for the Gaussian-z comparator",
        ],
        "limitations": [
            "neither support nor distribution is inferred from project data",
            "Hoeffding guarantee does not cover dependent observations without another bound",
            "anchor-score independence and the physical reference gate remain unresolved",
        ],
        "denominator_summary": denominator_rows,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    source_paths = {
        "snapshot_anchor_robustness_exact.py": Path(__file__),
        "snapshot_anchor_coverage_mc.py": Path(__file__).with_name(
            "bounded_block_risk_anchor_coverage_mc.py"
        ),
        "snapshot_bounded_block_risk_score.py": Path(__file__).with_name(
            "bounded_block_risk_score.py"
        ),
        "snapshot_dependency_graph_bound.py": Path(__file__).with_name(
            "dependency_graph_bound.py"
        ),
    }
    for target, source in source_paths.items():
        shutil.copyfile(source, out / target)

    _write_plot(out, rows)
    summary = {
        "schema": "pion-bounded-anchor-robustness-summary-v1",
        "status": "SYNTHETIC_EXACT_ENUMERATION_ONLY",
        "physical_score": None,
        "data_access": protocol["data_access"],
        "cells": len(rows),
        "exact_count_states_total": sum(
            r["planned_enumerated_count_states"] for r in rows
        ),
        "alpha_anchor": ALPHA_ANCHOR,
        "normal_z_coverage_min": min(
            r["normal_z_exact_coverage"] for r in rows
        ),
        "hoeffding_coverage_min": min(
            r["hoeffding_exact_coverage"] for r in rows
        ),
        "hoeffding_coverage_all_at_least_nominal": all(
            r["hoeffding_exact_coverage"] >= 1.0 - ALPHA_ANCHOR
            for r in rows
        ),
        "normal_z_coverage_rare_spike_n16": next(
            r["normal_z_exact_coverage"] for r in rows
            if r["distribution"] == "rare_spike_two_point"
            and r["anchor_sample_size"] == 16
        ),
        "hoeffding_to_normal_radius_ratio_max": max(
            r["hoeffding_to_normal_radius_ratio"] for r in rows
        ),
        "files": [
            "README.md", "protocol.json", "coverage_summary.csv",
            "denominator_summary.csv", "coverage_and_radius_cost.png",
            "snapshot_anchor_robustness_exact.py",
            "snapshot_anchor_coverage_mc.py",
            "snapshot_bounded_block_risk_score.py",
            "snapshot_dependency_graph_bound.py", "summary.json",
            "artifact_hashes.json",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    readme = [
        "# Robust anchor interval calibration: exact bounded-data stress test",
        "",
        "**SYNTHETIC EXACT ENUMERATION ONLY. `physical_score=null`.** No project anchor, correlator, ensemble, sealed split, GPU, or QUDA input is read.",
        "",
        "## Question and methods",
        "",
        "The previous joint calibration diagnostic used an iid known-variance normal anchor model. This report tests the remaining model-sensitivity boundary: how the corresponding Gaussian z radius behaves under bounded, non-Gaussian anchor errors, and what width a distribution-free bounded-data alternative pays.",
        "",
        "Each synthetic anchor error is a two-point variable with mean zero and variance `sigma^2=0.05^2`. The rare upper point has probability `p` and the lower point probability `1-p`; the three cases use `p=0.5`, `0.1`, and `0.005`. For each `n_A={list(ANCHOR_SAMPLE_SIZES)}`, all `n_A+1` possible counts of upper-point observations are enumerated exactly with their Binomial probability mass.",
        "",
        "The model-based comparator is `B_z=z_(1-alpha_A/2)*sigma/sqrt(n_A)`. It uses an oracle-known variance but a Gaussian critical value; under these non-Gaussian generators it has no finite-sample 99% guarantee. The distribution-free alternative is `B_H=(U-L)*sqrt(log(2/alpha_A)/(2*n_A))`, where `[L,U]` is the exact support. Hoeffding guarantees at least `1-alpha_A` coverage for iid bounded observations. Both formulae are two-sided.",
        "",
        "![Exact anchor coverage and interval-width tradeoff](coverage_and_radius_cost.png)",
        "",
        "## Results",
        "",
        f"The rare-spike case (`p=0.005`) has exact Gaussian-z coverage `{summary['normal_z_coverage_rare_spike_n16']:.4%}` at `n_A=16`, below the nominal 99%; the worst Gaussian-z coverage in the grid is `{summary['normal_z_coverage_min']:.4%}`. Exact Hoeffding coverage is at least `{summary['hoeffding_coverage_min']:.4%}` in every grid cell, consistent with its finite-sample guarantee. The largest Hoeffding/Gaussian radius ratio is `{summary['hoeffding_to_normal_radius_ratio_max']:.3f}` (rare-spike support). This width directly enlarges the Pion correction `2B/R` if the required independent and support assumptions can be established.",
        "",
        "`coverage_summary.csv` reports exact coverage, both radii, support and variance for every cell. `denominator_summary.csv` separates hypothetical anchor observations from the exact count states; the latter sum to probability mass one and are not Monte Carlo replicates or measured data.",
        "",
        "## Scope and next boundary",
        "",
        "This shows a real calibration choice: the Gaussian interval can be much narrower but relies on a distributional model; the Hoeffding interval is valid for any iid distribution within a proven support but may be too wide to certify a useful bounded block-risk lower bound. It does not show that project anchor errors are iid, bounded by these synthetic endpoints, or independent of score samples. If measurements are dependent, an independence or mixing bound is still required; if support is unknown, the Hoeffding radius is unavailable. The E-anchor receipt, common-bias bound, physical reference gate, and sealed-data authorization remain unresolved, so the task remains `NOT_READY/UNCERTIFIED` with `physical_score=null`.",
        "",
        "## Source note",
        "",
        "The known-variance Gaussian z interval is exact under the normal-population model stated by [NIST's confidence-interval reference](https://www.itl.nist.gov/div898/handbook/prc/section1/prc14.htm). The bounded independent-summand premise is from [Hoeffding (1963), *Probability Inequalities for Sums of Bounded Random Variables*](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf). The two-sided radius used here follows by applying the one-sided bounded-sum inequality to both tails and allocating `alpha_anchor/2` to each.",
        "",
        "## Rebuild",
        "",
        "Use a new output directory; existing artifacts are never overwritten:",
        "",
        "```text",
        "python -B -m tasks.pion_projection.bounded_block_risk_anchor_robustness_exact --output NEW_DIRECTORY",
        "```",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n")
    artifact_names = [name for name in summary["files"]
                      if name != "artifact_hashes.json"]
    hashes = {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in artifact_names
    }
    (out / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))


if __name__ == "__main__":
    main()
