"""Analytic feasibility and known-gap coverage stress for D06 Markov bounds.

This runner generates only synthetic bounded scores with an exactly known
two-state Markov driver. It reads no gauge, propagator, reference, or sealed
inputs and cannot certify the physical block-chain gap.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bounded_block_risk import ALPHA_TOTAL, ARMS, R_STAR
from .bounded_block_risk_markov import markov_bernstein_radius


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/optimize_gaussian/d06_markov_dependence_cpu_20261126_v1"
N_GRID = (100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000)
GAP_GRID = (1.0, 0.36, 0.0975, 0.02)
TV_GRID = (0.0, 0.005, 0.02, 0.04)
MARGIN_GRID = (0.01, 0.025, 0.05, 0.10)
MC_BLOCKS = (250, 2_000, 20_000)
MC_SWITCH_PROBABILITIES = (0.5, 0.1, 0.025)
MC_PANELS = 4_000
MC_BATCH = 64
MC_SEED = 20261126


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty diagnostic table")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    if trials < 1:
        raise ValueError("trials must be positive")
    z = 1.959963984540054
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials**2)) / denom
    if successes == 0:
        return 0.0, min(1.0, center + half)
    if successes == trials:
        return max(0.0, center - half), 1.0
    return max(0.0, center - half), min(1.0, center + half)


def _required_n(*, gap: float, target: float, alpha_per_event: float) -> int:
    """Find the smallest N meeting the monotone analytic radius target."""
    if target <= 0:
        raise ValueError("target must be positive")
    high = 1
    while markov_bernstein_radius(
        blocks=high, gamma_ps_lower=gap, alpha=alpha_per_event,
        lower=-R_STAR, upper=1.0,
    ) > target:
        high *= 2
        if high > 2**62:
            raise OverflowError("required block count exceeds the supported integer range")
    low = high // 2
    while low + 1 < high:
        mid = (low + high) // 2
        radius = markov_bernstein_radius(
            blocks=mid, gamma_ps_lower=gap, alpha=alpha_per_event,
            lower=-R_STAR, upper=1.0,
        )
        if radius <= target:
            high = mid
        else:
            low = mid
    return high


def _make_sensitivity_rows() -> list[dict]:
    rows: list[dict] = []
    for n in N_GRID:
        for gamma in GAP_GRID:
            for tv in TV_GRID:
                stationary_alpha = ALPHA_TOTAL - tv
                if stationary_alpha <= 0:
                    alpha_per_event = 0.0
                    radius_w = radius_d = math.inf
                    possible = False
                else:
                    alpha_per_event = stationary_alpha / 3.0
                    radius_w = markov_bernstein_radius(
                        blocks=n, gamma_ps_lower=gamma, alpha=alpha_per_event,
                        lower=0.0, upper=1.0,
                    )
                    radius_d = markov_bernstein_radius(
                        blocks=n, gamma_ps_lower=gamma, alpha=alpha_per_event,
                        lower=-R_STAR, upper=1.0,
                    )
                    max_contrast_given_positive_denominator = 1.0 - R_STAR * radius_w
                    possible = (
                        radius_w < 1.0
                        and radius_d < max_contrast_given_positive_denominator
                    )
                rows.append({
                    "planned_scored_blocks": n,
                    "gamma_ps_lower_assumed": gamma,
                    "post_burn_in_tv_assumed": tv,
                    "alpha_total": ALPHA_TOTAL,
                    "alpha_after_tv": stationary_alpha,
                    "alpha_per_family_event": alpha_per_event,
                    "trainable_w_radius": radius_w,
                    "paired_contrast_radius": radius_d,
                    "max_contrast_if_w_denominator_just_positive": (
                        1.0 - R_STAR * radius_w if math.isfinite(radius_w) else ""
                    ),
                    "best_case_support_allows_release": possible,
                    "formula_status": "CONDITIONAL_SCENARIO",
                })
    return rows


def _make_required_rows() -> list[dict]:
    rows = []
    for gap in GAP_GRID:
        for tv in (0.0, 0.005, 0.02):
            alpha_per_event = (ALPHA_TOTAL - tv) / 3.0
            for margin in MARGIN_GRID:
                rows.append({
                    "gamma_ps_lower_assumed": gap,
                    "post_burn_in_tv_assumed": tv,
                    "paired_contrast_margin_target": margin,
                    "alpha_per_event": alpha_per_event,
                    "required_scored_blocks": _required_n(
                        gap=gap, target=margin, alpha_per_event=alpha_per_event,
                    ),
                })
    return rows


def _simulate_cell(
    *, n: int, switch_probability: float, panels: int, batch_size: int,
    seed: int,
) -> dict:
    """Stress joint LCB coverage on a stationary hidden two-state chain.

    The augmented state is (two-state driver, fresh armwise uniforms). The
    bounded score vector is a fixed function of that state. The comparator 1
    has exactly rho=r_star under stationarity; comparator 2 has a strong,
    known positive contrast. This is a method stress only, not an HMC model.
    """
    if not 0.0 < switch_probability <= 0.5:
        raise ValueError("switch_probability must lie in (0, 0.5]")
    eigenvalue = 1.0 - 2.0 * switch_probability
    gamma = 1.0 - eigenvalue * eigenvalue
    alpha_per_event = ALPHA_TOTAL / 3.0
    radius_w = markov_bernstein_radius(
        blocks=n, gamma_ps_lower=gamma, alpha=alpha_per_event,
        lower=0.0, upper=1.0,
    )
    radius_d = markov_bernstein_radius(
        blocks=n, gamma_ps_lower=gamma, alpha=alpha_per_event,
        lower=-R_STAR, upper=1.0,
    )
    true_means = np.array([0.06, R_STAR * 0.06, 0.25])
    true_contrasts = np.array([
        true_means[1] - R_STAR * true_means[0],
        true_means[2] - R_STAR * true_means[0],
    ])

    rng = np.random.default_rng(seed)
    covered = 0
    family_released = 0
    used_panels = 0
    while used_panels < panels:
        count = min(batch_size, panels - used_panels)
        initial = rng.integers(0, 2, size=(count, 1), dtype=np.uint8)
        flips = (rng.random((count, n - 1)) < switch_probability).astype(np.uint8)
        states = np.concatenate(
            [initial, np.bitwise_xor(initial, np.bitwise_xor.accumulate(flips, axis=1))],
            axis=1,
        )
        p_w = np.where(states == 0, 0.04, 0.08)
        p_slsqp = R_STAR * p_w
        p_cheb = np.where(states == 0, 0.20, 0.30)
        probs = np.stack([p_w, p_slsqp, p_cheb], axis=2)
        scores = (rng.random((count, n, 3)) < probs).astype(np.float64)
        means = scores.mean(axis=1)
        contrast_slsqp = means[:, 1] - R_STAR * means[:, 0]
        contrast_cheb = means[:, 2] - R_STAR * means[:, 0]
        lower_w = means[:, 0] - radius_w
        lower_slsqp = contrast_slsqp - radius_d
        lower_cheb = contrast_cheb - radius_d
        covered += int(np.count_nonzero(
            (lower_w <= true_means[0])
            & (lower_slsqp <= true_contrasts[0])
            & (lower_cheb <= true_contrasts[1])
        ))
        family_released += int(np.count_nonzero(
            (lower_w > 0.0) & (lower_slsqp > 0.0) & (lower_cheb > 0.0)
        ))
        used_panels += count

    coverage_ci = _wilson(covered, panels)
    release_ci = _wilson(family_released, panels)
    return {
        "planned_panels": panels,
        "valid_panels": panels,
        "failed_panels": 0,
        "abstained_panels": 0,
        "scored_blocks_per_panel": n,
        "switch_probability_assumed": switch_probability,
        "driver_eigenvalue": eigenvalue,
        "known_gamma_ps": gamma,
        "alpha_per_event": alpha_per_event,
        "radius_w": radius_w,
        "radius_paired_contrast": radius_d,
        "true_mu_w": float(true_means[0]),
        "true_mu_slsqp": float(true_means[1]),
        "true_mu_chebyshev": float(true_means[2]),
        "true_D_slsqp_boundary": float(true_contrasts[0]),
        "true_D_chebyshev": float(true_contrasts[1]),
        "joint_lower_bound_coverage": covered / panels,
        "coverage_wilson95_low": coverage_ci[0],
        "coverage_wilson95_high": coverage_ci[1],
        "family_release_at_slsqp_boundary": family_released / panels,
        "release_wilson95_low": release_ci[0],
        "release_wilson95_high": release_ci[1],
        "mcse_coverage": math.sqrt((covered / panels) * (1 - covered / panels) / panels),
        "mcse_release": math.sqrt((family_released / panels) * (1 - family_released / panels) / panels),
    }


def _make_coverage_rows() -> list[dict]:
    rows = []
    for n in MC_BLOCKS:
        for switch in MC_SWITCH_PROBABILITIES:
            # Independent streams per cell make the replay key explicit.
            seed = MC_SEED + n * 100 + int(round(switch * 1000))
            rows.append(_simulate_cell(
                n=n, switch_probability=switch, panels=MC_PANELS,
                batch_size=MC_BATCH, seed=seed,
            ))
    return rows


def _plot(sensitivity: list[dict], coverage: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), constrained_layout=True)
    ax = axes[0]
    for gamma in GAP_GRID[:3]:
        selected = [row for row in sensitivity
                    if row["gamma_ps_lower_assumed"] == gamma
                    and row["post_burn_in_tv_assumed"] == 0.0]
        ns = [row["planned_scored_blocks"] for row in selected]
        ax.plot(ns, [row["trainable_w_radius"] for row in selected],
                marker="o", label=rf"$W$, $\gamma_{{ps}}={gamma:g}$")
        ax.plot(ns, [row["paired_contrast_radius"] for row in selected],
                marker="s", linestyle="--", label=rf"contrast, $\gamma_{{ps}}={gamma:g}$")
    for target in (0.05, 0.10):
        ax.axhline(target, color="#777777", linestyle=":", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("scored paired macroblocks n")
    ax.set_ylabel("support-only Markov Bernstein radius")
    ax.set_title("Conditional bound width")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    labels = [f"n={row['scored_blocks_per_panel']}\nγ={row['known_gamma_ps']:.3g}"
              for row in coverage]
    x = np.arange(len(coverage))
    coverage_y = np.array([row["joint_lower_bound_coverage"] for row in coverage])
    coverage_lo = np.array([row["coverage_wilson95_low"] for row in coverage])
    coverage_hi = np.array([row["coverage_wilson95_high"] for row in coverage])
    ax = axes[1]
    ax.bar(x, coverage_y, color="#3977a8", alpha=0.85)
    ax.errorbar(x, coverage_y, yerr=[coverage_y - coverage_lo, coverage_hi - coverage_y],
                fmt="none", ecolor="#222222", capsize=3, linewidth=1)
    ax.axhline(0.95, color="#ba4a00", linestyle="--", label="nominal 95%")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0.85, 1.005)
    ax.set_ylabel("joint lower-bound coverage")
    ax.set_title(f"Known-gap MC stress ({MC_PANELS:,} panels/cell)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[2]
    release = np.array([row["family_release_at_slsqp_boundary"] for row in coverage])
    release_hi = np.array([row["release_wilson95_high"] for row in coverage])
    release_lo = np.array([row["release_wilson95_low"] for row in coverage])
    ax.bar(x, release, color="#cc7a00", alpha=0.85)
    ax.errorbar(x, release, yerr=[release - release_lo, release_hi - release],
                fmt="none", ecolor="#222222", capsize=3, linewidth=1)
    ax.axhline(0.05, color="#ba4a00", linestyle="--", label="familywise α=0.05")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0.0, max(0.06, float(np.max(release_hi)) * 1.25))
    ax.set_ylabel("release rate when SLSQP is exactly at r⋆")
    ax.set_title("False family release at boundary")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    max_upper = float(np.max(release_hi))
    ax.text(
        0.03, 0.93,
        f"0 releases in every cell; max Wilson upper = {max_upper:.3%}",
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.9},
    )

    figure.suptitle("Optimize D06: conditional Markov dependence route")
    figure.savefig(output, dpi=170)
    plt.close(figure)


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    sensitivity = _make_sensitivity_rows()
    required = _make_required_rows()
    coverage = _make_coverage_rows()
    _write_csv(output / "sensitivity.csv", sensitivity)
    _write_csv(output / "required_blocks.csv", required)
    _write_csv(output / "known_gap_coverage_stress.csv", coverage)
    _plot(sensitivity, coverage, output / "markov_dependence_feasibility.png")

    protocol = {
        "schema": "optimize-d06-markov-dependence-diagnostic-v1",
        "status": "CONDITIONAL_METHOD_DIAGNOSTIC_ONLY",
        "estimand": "bounded paired macroblock risk; separate from primary D04 nested-bootstrap risk",
        "physical_score": None,
        "certified": False,
        "data_access": "no project measurements or sealed inputs",
        "selected_contract": {
            "arms": list(ARMS),
            "r_star": R_STAR,
            "alpha_total": ALPHA_TOTAL,
            "family_events": [
                "trainable-W mean positive denominator",
                "trainable-W vs same-budget SLSQP paired contrast",
                "trainable-W vs fixed degree-4 Chebyshev paired contrast",
            ],
            "post_burn_in_tv_grid": list(TV_GRID),
        },
        "support_only_inputs": {
            "arm_score_support": [0.0, 1.0],
            "paired_contrast_support": [-R_STAR, 1.0],
            "variance_bound": "support_width^2/4",
            "centered_absolute_bound": "support_width",
            "tail": "two-sided corrected non-reversible Bernstein inequality, conservatively inverted for lower bounds",
        },
        "analytic_grid": {
            "scored_block_counts": list(N_GRID),
            "gamma_ps_lower_assumptions": list(GAP_GRID),
            "post_burn_in_tv_assumptions": list(TV_GRID),
            "planned_formula_rows": len(sensitivity),
            "valid_formula_rows": len(sensitivity),
            "failed_formula_rows": 0,
            "abstained_formula_rows": 0,
            "best_case_support_feasible_rows": sum(
                bool(row["best_case_support_allows_release"]) for row in sensitivity
            ),
        },
        "coverage_stress": {
            "generator": "stationary symmetric two-state Markov driver plus independent fresh armwise uniforms",
            "known_gamma_ps": "1-(1-2p_switch)^2",
            "scored_block_counts": list(MC_BLOCKS),
            "switch_probabilities": list(MC_SWITCH_PROBABILITIES),
            "panels_per_cell": MC_PANELS,
            "planned_panels": len(coverage) * MC_PANELS,
            "valid_panels": len(coverage) * MC_PANELS,
            "failed_panels": 0,
            "abstained_panels": 0,
            "seed_base": MC_SEED,
            "boundary_effect": "same-budget SLSQP risk ratio equals r_star exactly under stationarity",
            "alternative_effect": "Chebyshev has a large positive paired contrast, included only to exercise both comparisons",
            "intervals": "Wilson 95% intervals quantify Monte Carlo uncertainty of these synthetic coverage and release rates",
        },
        "assumptions_not_verified_for_production": [
            "homogeneous Markov chain for the exact paired macroblock state",
            "stationary initialization or a justified post-burn-in total-variation bound",
            "valid audited lower bound on that chain's pseudo-spectral gap",
            "fixed score map; any blockwise solver randomness included in the Markov state or otherwise justified",
            "separate propagation of D04 reference-fit and candidate-selection uncertainty",
        ],
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )

    worst_coverage = min(float(row["joint_lower_bound_coverage"]) for row in coverage)
    max_release = max(float(row["family_release_at_slsqp_boundary"]) for row in coverage)
    min_required = min(int(row["required_scored_blocks"]) for row in required)
    max_required = max(int(row["required_scored_blocks"]) for row in required)
    coverage_lines = "\n".join(
        f"| {row['scored_blocks_per_panel']} | {row['known_gamma_ps']:.4f} | "
        f"{row['joint_lower_bound_coverage']:.4f} "
        f"[{row['coverage_wilson95_low']:.4f}, {row['coverage_wilson95_high']:.4f}] | "
        f"{row['family_release_at_slsqp_boundary']:.4f} "
        f"[{row['release_wilson95_low']:.4f}, {row['release_wilson95_high']:.4f}] |"
        for row in coverage
    )
    README = f"""# D06 Markov dependence correction: CPU feasibility and coverage diagnostic (v1)

**Conditional method diagnostic only.** No gauge, propagator, reference, held-out, sealed, GPU, or QUDA inputs were read. `physical_score=null`; the bounded block-risk estimand remains an alternative to D04 and is not a production result.

## Method

For each of three simultaneous events (trainable-W positive denominator and its paired contrasts against same-budget SLSQP and degree-4 Chebyshev), the script applies the corrected non-reversible Markov Bernstein inequality with a support-only variance bound. For a function supported on an interval of width `w`, it uses `Var_pi(f) <= w^2/4` and `|f-E_pi f| <= w`. The supports are `[0,1]` for arm risk and `[-r_star,1]` for `X_comparator-r_star X_W`. The two-sided tail is conservatively inverted for each lower bound.

The stationary family budget is `(alpha_total-TV)/3`. The supplied TV term is paid once for the union event: by the total-variation event inequality, this is valid only when the *joint scored-chain initial law after burn-in* is within that TV distance of stationarity. TV is not estimated from the data here. The gap and TV values in the tables are assumptions varied for feasibility, not observed properties.

The primary source is Paulin, corrected arXiv v5, Theorem 3.11, Eq. (3.25): [paper](https://arxiv.org/html/1212.2015v5). It states the bound for a stationary homogeneous Markov chain and a fixed summand function using its pseudo-spectral gap. This diagnostic does not establish those conditions for any LatticeArena chain.

## What the figure shows

![Conditional confidence widths and known-gap Monte Carlo checks](markov_dependence_feasibility.png)

- **Left:** support-only radius for the denominator and paired contrast over scored macroblock count, under three assumed pseudo-gaps. A contrast can pass only if its observed mean exceeds the contrast radius, while the trainable-W sample mean must separately exceed its own radius. At stricter gaps these necessary margins remain large.
- **Middle:** joint lower-bound coverage in 4,000 independent simulated panels per cell. The exact stationary two-state driver has the listed pseudo-gap; SLSQP is exactly at `r_star`, while Chebyshev is given a strong positive contrast. Wilson intervals show Monte Carlo uncertainty, not a production coverage guarantee.
- **Right:** family release rate when SLSQP is exactly at the threshold. Any release is a false family release for this synthetic boundary case; Wilson upper endpoints remain visible even when the observed count is zero.

| Scored blocks | Assumed `gamma_ps` | Joint coverage [MC Wilson 95%] | Boundary family release [MC Wilson 95%] |
|---:|---:|---:|---:|
{coverage_lines}

Coverage cells are `planned/valid/failed/abstained = {len(coverage)*MC_PANELS}/{len(coverage)*MC_PANELS}/0/0`. There are {len(sensitivity)} analytic scenarios (`planned/valid/failed/abstained = {len(sensitivity)}/{len(sensitivity)}/0/0`); valid means the formula evaluated, not that its assumptions hold. The best-case support-feasibility marker is true in {protocol['analytic_grid']['best_case_support_feasible_rows']}/{len(sensitivity)} formula rows. The paired-contrast radius needed for margins from 0.01 to 0.10 ranges from {min_required:,} to {max_required:,} scored blocks over the declared gap/TV grid; see `required_blocks.csv` for the exact conditional scenarios.

## Interpretation and next gate

This supplies a conservative, finite-sample dependence route for the **bounded block-risk estimand** if an independent reviewer can justify a homogeneous chain of the exact paired macroblock states, a pseudo-spectral-gap lower bound, the post-burn-in TV receipt, and a fixed score map. If block scores have stochastic solver noise, that randomness must be included in the state or otherwise justified. The toy chain confirms the implementation's behavior only for its known two-state model; it cannot infer a gauge-chain gap from trajectory spacing, empirical autocorrelation, or synthetic coverage.

The method is deliberately support-only and may require many blocks. A future comparison with tighter variance receipts is useful only if real, independent receipts can reduce the worst-case bounds. D04 reference-fit uncertainty, blind candidate selection, real trainable-W training, both comparators, and physical authorization remain open. Do not connect this helper to a confirmation evaluator until those contracts are reviewed.

Text flow: `freeze arms + clipped block score -> score paired macroblock chain -> verify homogeneous/stationary-or-TV + gamma_ps receipts -> simultaneous denominator and two contrast bounds -> both pass or ABSTAIN/no release`. A missing dependence receipt forces `ABSTAIN`; numeric sensitivity values never fill that receipt.
"""
    (output / "README.md").write_text(README, encoding="utf-8")

    source_files = [
        ROOT / "tasks/optimize_gaussian/bounded_block_risk.py",
        ROOT / "tasks/optimize_gaussian/bounded_block_risk_markov.py",
        Path(__file__).resolve(),
    ]
    (output / "source_hashes.json").write_text(json.dumps({
        str(path.relative_to(ROOT)): _sha256(path) for path in source_files
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payloads = [
        output / "README.md", output / "protocol.json", output / "source_hashes.json",
        output / "sensitivity.csv", output / "required_blocks.csv",
        output / "known_gap_coverage_stress.csv", output / "markov_dependence_feasibility.png",
    ]
    manifest = {
        "schema": "optimize-d06-markov-dependence-artifact-manifest-v1",
        "payload_sha256": {path.name: _sha256(path) for path in payloads},
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return {
        "output": str(output),
        "analytic_rows": len(sensitivity),
        "coverage_cells": len(coverage),
        "panels_per_cell": MC_PANELS,
        "worst_joint_coverage": worst_coverage,
        "max_boundary_family_release": max_release,
        "best_case_support_feasible_rows": protocol["analytic_grid"]["best_case_support_feasible_rows"],
        "artifact_payloads": len(manifest["payload_sha256"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
