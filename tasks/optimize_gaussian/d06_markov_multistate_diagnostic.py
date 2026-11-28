"""Known-gap non-reversible multi-state calibration for D06 Markov bounds.

This CPU-only diagnostic uses a finite cyclic chain with uniform refreshes.
The chain is non-reversible, its stationary distribution and pseudo-spectral
gap are known exactly, and the three bounded arm scores are fixed functions of
the paired state. It reads no project measurement or sealed input.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bounded_block_risk import ALPHA_TOTAL, ARMS, R_STAR
from .bounded_block_risk_markov import (
    bounded_block_risk_markov_decision,
    markov_bernstein_radius,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/optimize_gaussian/d06_markov_multistate_cpu_20261126_v1"
STATE_COUNT = 19
GAP_GRID = (0.36, 0.0975, 0.02)
BLOCK_GRID = (2_000, 20_000)
TV_GRID = (0.0, 0.02)
PANELS_PER_CELL = 1_200
BATCH_SIZE = 100
SEED_BASE = 2026112603


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


def _chain_contract(target_gamma: float, state_count: int = STATE_COUNT) -> dict:
    """Build and numerically audit P=(1-eta)C+eta J on a directed cycle.

    C is a one-step cyclic shift and J resets to the uniform law. On the
    mean-zero subspace, P is a scaled unitary shift, so the nontrivial
    eigenvalues of (P*)^k P^k are (1-eta)^(2k). Therefore
    gamma_ps=max_k [1-(1-eta)^(2k)]/k=1-(1-eta)^2.
    """
    if not 0.0 < target_gamma < 1.0:
        raise ValueError("target_gamma must be in (0, 1)")
    eta = 1.0 - math.sqrt(1.0 - target_gamma)
    cycle = np.zeros((state_count, state_count), dtype=float)
    for state in range(state_count):
        cycle[state, (state + 1) % state_count] = 1.0
    refresh = np.full_like(cycle, 1.0 / state_count)
    transition = (1.0 - eta) * cycle + eta * refresh
    stationary = np.full(state_count, 1.0 / state_count)
    stationary_error = float(np.max(np.abs(stationary @ transition - stationary)))
    detailed_balance_error = float(np.max(np.abs(
        stationary[:, None] * transition - stationary[None, :] * transition.T
    )))
    singular_squared = np.linalg.eigvalsh(transition.T @ transition)
    q = float(singular_squared[-2])
    numerical_gap_one = 1.0 - q
    numerical_gap_ps = max((1.0 - q**k) / k for k in range(1, 501))
    exact_gap = 1.0 - (1.0 - eta) ** 2
    if (stationary_error > 1e-12 or abs(numerical_gap_one - exact_gap) > 1e-11
            or abs(numerical_gap_ps - exact_gap) > 1e-11):
        raise ArithmeticError("finite-state numerical audit did not match exact gap")
    if detailed_balance_error < 1e-4:
        raise ArithmeticError("expected a genuinely non-reversible chain")
    return {
        "state_count": state_count,
        "reset_probability_eta": eta,
        "target_gamma_ps": target_gamma,
        "stationary_distribution": "uniform over all states",
        "stationary_max_residual": stationary_error,
        "detailed_balance_max_residual": detailed_balance_error,
        "largest_nontrivial_singular_value_squared": q,
        "numerical_gamma_ps_from_k_1_to_500": numerical_gap_ps,
        "exact_gamma_ps": exact_gap,
        "gamma_derivation": "max_k (1-(1-eta)^(2k))/k = 1-(1-eta)^2",
    }


def _score_map(state_count: int = STATE_COUNT) -> np.ndarray:
    """Return exact full-support boundary and positive-control arm scores."""
    if state_count != 19:
        raise ValueError("the exact r_star boundary fixture requires 19 states")
    states = np.arange(state_count)
    trainable = (states < 9).astype(float)
    same_budget_slsqp = (states >= 9).astype(float)
    chebyshev = np.ones(state_count, dtype=float)
    scores = np.stack([trainable, same_budget_slsqp, chebyshev], axis=1)
    means = scores.mean(axis=0)
    contrasts = means[1:] - R_STAR * means[0]
    if not np.array_equal(np.unique(scores), np.array([0.0, 1.0])):
        raise ArithmeticError("fixture must attain the complete [0,1] arm support")
    if not np.allclose(means, [9 / 19, 10 / 19, 1.0], rtol=0.0, atol=1e-15):
        raise ArithmeticError("unexpected stationary score means")
    if not np.allclose(contrasts, [0.0, 9 / 19], rtol=0.0, atol=1e-15):
        raise ArithmeticError("boundary or positive-control contrast is incorrect")
    return scores


def _simulate_cell(*, gamma: float, blocks: int, tv: float, panels: int,
                   batch_size: int, seed: int) -> dict:
    contract = _chain_contract(gamma)
    eta = float(contract["reset_probability_eta"])
    scores_by_state = _score_map()
    exact_means = scores_by_state.mean(axis=0)
    exact_contrasts = exact_means[1:] - R_STAR * exact_means[0]
    alpha_per_event = (ALPHA_TOTAL - tv) / 3.0
    radius_w = markov_bernstein_radius(
        blocks=blocks, gamma_ps_lower=gamma, alpha=alpha_per_event,
        lower=0.0, upper=1.0,
    )
    radius_d = markov_bernstein_radius(
        blocks=blocks, gamma_ps_lower=gamma, alpha=alpha_per_event,
        lower=-R_STAR, upper=1.0,
    )

    rng = np.random.default_rng(seed)
    # If nu=(1-rho)pi+rho*delta_0, then ||nu-pi||_TV=rho*(1-1/K).
    point_mass_mix = tv / (1.0 - 1.0 / STATE_COUNT)
    covered_joint = denominator_positive = 0
    slsqp_lower_positive = cheb_lower_positive = family_release = 0
    completed = 0
    while completed < panels:
        count = min(batch_size, panels - completed)
        states = rng.integers(0, STATE_COUNT, size=count)
        point_mass = rng.random(count) < point_mass_mix
        states[point_mass] = 0
        sums = np.zeros((count, len(ARMS)), dtype=np.float64)
        for step in range(blocks):
            sums += scores_by_state[states]
            if step + 1 < blocks:
                reset = rng.random(count) < eta
                fresh = rng.integers(0, STATE_COUNT, size=count)
                states = np.where(reset, fresh, (states + 1) % STATE_COUNT)
        means = sums / blocks
        contrasts = means[:, 1:] - R_STAR * means[:, [0]]
        lower_w = means[:, 0] - radius_w
        lower_d = contrasts - radius_d
        denominator_ok = lower_w > 0.0
        slsqp_ok = lower_d[:, 0] > 0.0
        cheb_ok = lower_d[:, 1] > 0.0
        covered = (
            (lower_w <= exact_means[0])
            & (lower_d[:, 0] <= exact_contrasts[0])
            & (lower_d[:, 1] <= exact_contrasts[1])
        )
        denominator_positive += int(np.count_nonzero(denominator_ok))
        slsqp_lower_positive += int(np.count_nonzero(slsqp_ok))
        cheb_lower_positive += int(np.count_nonzero(cheb_ok))
        family_release += int(np.count_nonzero(denominator_ok & slsqp_ok & cheb_ok))
        covered_joint += int(np.count_nonzero(covered))
        completed += count

    # Cross-check the vectorized calculation against the public decision helper
    # on one deterministic realized path from the same exact DGP.
    check_rng = np.random.default_rng(seed ^ 0x5A17)
    check_states = check_rng.integers(0, STATE_COUNT, size=blocks)
    if tv > 0.0 and check_rng.random() < point_mass_mix:
        check_states[0] = 0
    for step in range(blocks - 1):
        if check_rng.random() < eta:
            check_states[step + 1] = check_rng.integers(0, STATE_COUNT)
        else:
            check_states[step + 1] = (check_states[step] + 1) % STATE_COUNT
    checked = bounded_block_risk_markov_decision(
        scores_by_state[check_states], gamma_ps_lower=gamma,
        tv_bound_after_burn_in=tv,
    )
    checked_w = checked["bounds"]["trainable_w_mean"]["lower_bound"]
    checked_slsqp = checked["comparisons"][ARMS[1]]["lower_bound"]
    checked_cheb = checked["comparisons"][ARMS[2]]["lower_bound"]
    direct = scores_by_state[check_states].mean(axis=0)
    if not np.allclose(
        [checked_w, checked_slsqp, checked_cheb],
        [direct[0] - radius_w,
         direct[1] - R_STAR * direct[0] - radius_d,
         direct[2] - R_STAR * direct[0] - radius_d],
        rtol=0.0, atol=1e-12,
    ):
        raise ArithmeticError("vectorized simulator disagrees with decision helper")

    coverage_ci = _wilson(covered_joint, panels)
    denominator_ci = _wilson(denominator_positive, panels)
    slsqp_ci = _wilson(slsqp_lower_positive, panels)
    cheb_ci = _wilson(cheb_lower_positive, panels)
    family_ci = _wilson(family_release, panels)
    return {
        "state_count": STATE_COUNT,
        "target_gamma_ps": gamma,
        "verified_gamma_ps": contract["exact_gamma_ps"],
        "reset_probability_eta": eta,
        "scored_blocks": blocks,
        "post_burn_in_tv_bound": tv,
        "actual_initial_tv": tv,
        "panels_planned": panels,
        "panels_valid": panels,
        "panels_failed": 0,
        "panels_abstained_dgp": 0,
        "arm_score_means_exact": exact_means.tolist(),
        "paired_contrasts_exact": exact_contrasts.tolist(),
        "alpha_per_family_event": alpha_per_event,
        "trainable_risk_radius": radius_w,
        "contrast_radius": radius_d,
        "joint_lower_bound_coverage": covered_joint / panels,
        "joint_coverage_wilson95_low": coverage_ci[0],
        "joint_coverage_wilson95_high": coverage_ci[1],
        "denominator_positive_fraction": denominator_positive / panels,
        "denominator_positive_wilson95": list(denominator_ci),
        "slsqp_boundary_lower_positive_fraction": slsqp_lower_positive / panels,
        "slsqp_boundary_wilson95": list(slsqp_ci),
        "chebyshev_positive_control_lower_positive_fraction": cheb_lower_positive / panels,
        "chebyshev_positive_control_wilson95": list(cheb_ci),
        "family_release_at_slsqp_boundary_fraction": family_release / panels,
        "family_release_wilson95": list(family_ci),
        "mcse_joint_coverage": math.sqrt(
            (covered_joint / panels) * (1 - covered_joint / panels) / panels
        ),
        "public_helper_crosscheck": "exact agreement on one realized path",
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty diagnostic table")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict], path: Path) -> None:
    labels = [
        f"γ={row['target_gamma_ps']:g}\nn={row['scored_blocks']:,}\nTV={row['post_burn_in_tv_bound']:g}"
        for row in rows
    ]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.5), constrained_layout=True)
    coverage = np.array([row["joint_lower_bound_coverage"] for row in rows])
    coverage_lo = np.array([row["joint_coverage_wilson95_low"] for row in rows])
    coverage_hi = np.array([row["joint_coverage_wilson95_high"] for row in rows])
    axes[0].errorbar(x, coverage, yerr=[coverage - coverage_lo, coverage_hi - coverage],
                     fmt="o", color="#286e9e", capsize=3)
    axes[0].axhline(1.0 - ALPHA_TOTAL, color="#b34424", linestyle="--", label="nominal 95%")
    axes[0].set(title="Joint lower-bound coverage", ylabel="coverage", ylim=(0.9, 1.005))
    axes[0].legend(fontsize=8)

    cheb = np.array([row["chebyshev_positive_control_lower_positive_fraction"] for row in rows])
    axes[1].plot(x, cheb, "o-", color="#4b8c5a", label="positive Chebyshev contrast")
    denom = np.array([row["denominator_positive_fraction"] for row in rows])
    axes[1].plot(x, denom, "s--", color="#80589b", label="W risk lower bound > 0")
    axes[1].set(title="Useful positive-bound checks", ylabel="fraction of panels", ylim=(-0.03, 1.03))
    axes[1].legend(fontsize=8)

    family = np.array([row["family_release_at_slsqp_boundary_fraction"] for row in rows])
    family_hi = np.array([row["family_release_wilson95"][1] for row in rows])
    family_lo = np.array([row["family_release_wilson95"][0] for row in rows])
    axes[2].errorbar(x, family, yerr=[family - family_lo, family_hi - family],
                     fmt="o", color="#c47b12", capsize=3)
    axes[2].axhline(ALPHA_TOTAL, color="#b34424", linestyle="--", label="familywise α=0.05")
    axes[2].set(title="Family release at SLSQP boundary", ylabel="false family release rate",
                ylim=(-0.003, max(0.055, float(family_hi.max()) * 1.2)))
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.set_xticks(x, labels, fontsize=6.5)
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("D06 bounded block-risk: non-reversible 19-state calibration")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    source_files = [
        ROOT / "tasks/optimize_gaussian/bounded_block_risk.py",
        ROOT / "tasks/optimize_gaussian/bounded_block_risk_markov.py",
        Path(__file__).resolve(),
    ]
    protocol = {
        "schema": "optimize-d06-markov-multistate-diagnostic-v1",
        "status": "CONDITIONAL_METHOD_DIAGNOSTIC_ONLY",
        "estimand": "bounded paired-macroblock risk; distinct from primary D04 estimand",
        "physical_score": None,
        "certified": False,
        "data_access": "synthetic finite-state chains only; no project measurements, sealed inputs, GPU or QUDA",
        "generator": {
            "chain": "P=(1-eta) directed_cycle + eta uniform_refresh on 19 states",
            "stationary_law": "uniform",
            "reversible": False,
            "pseudo_gap": "gamma_ps=1-(1-eta)^2, analytically; verified against finite-state singular spectrum",
            "score_map": "W=1 on 9 states, SLSQP=1 on remaining 10 states, Chebyshev=1 on all states",
            "exact_means": [9 / 19, 10 / 19, 1.0],
            "exact_paired_contrasts": [0.0, 9 / 19],
            "tv_initialization": "nu=(1-rho)uniform+rho*delta_0 with rho=TV/(1-1/19)",
        },
        "grid": {
            "gamma_ps": list(GAP_GRID),
            "scored_blocks": list(BLOCK_GRID),
            "post_burn_in_tv": list(TV_GRID),
            "panels_per_cell": PANELS_PER_CELL,
            "planned_cells": len(GAP_GRID) * len(BLOCK_GRID) * len(TV_GRID),
            "seed_base": SEED_BASE,
            "confidence_rule": "Wilson 95% Monte Carlo intervals for coverage and release proportions",
        },
        "source_hashes": {str(path.relative_to(ROOT)): _sha256(path) for path in source_files},
        "text_flow": "known non-reversible chain -> exact stationary law/pseudo-gap -> bounded paired scores -> simultaneous Markov bounds -> coverage and release diagnostics",
        "assumptions_not_transferred_to_production": [
            "the production paired-macroblock process is homogeneous Markov",
            "a valid pseudo-spectral-gap lower bound applies to the exact production scored chain",
            "the production post-burn-in initial law has a justified TV bound",
            "the D04 reference-fit and candidate-selection uncertainty are separately controlled",
        ],
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows = []
    for gamma_index, gamma in enumerate(GAP_GRID):
        for blocks_index, blocks in enumerate(BLOCK_GRID):
            for tv_index, tv in enumerate(TV_GRID):
                seed = SEED_BASE + gamma_index * 100_000 + blocks_index * 10_000 + tv_index * 1_000
                rows.append(_simulate_cell(
                    gamma=gamma, blocks=blocks, tv=tv,
                    panels=PANELS_PER_CELL, batch_size=BATCH_SIZE, seed=seed,
                ))

    _write_csv(output / "coverage.csv", rows)
    _plot(rows, output / "multistate_markov_coverage.png")
    min_coverage = min(row["joint_lower_bound_coverage"] for row in rows)
    max_family_release = max(row["family_release_at_slsqp_boundary_fraction"] for row in rows)
    min_family_upper = max(row["family_release_wilson95"][1] for row in rows)
    pooled_macroblocks = sum(int(row["scored_blocks"]) for row in rows) * PANELS_PER_CELL
    readme = [
        "# D06 bounded block-risk: non-reversible multi-state Markov calibration",
        "",
        "**Conditional synthetic method diagnostic only.** `physical_score=null`; the bounded block-risk estimand remains separate from D04. No gauge, propagator, E-reference, validation, embargo, shared holdout, confirmation, GPU or QUDA input was read.",
        "",
        "## Generator and exact calibration target",
        "",
        "The paired macroblock state follows a 19-state directed cycle with a uniform refresh: `P=(1-eta) C + eta J`. The uniform law is stationary, the chain is irreducible and aperiodic, and detailed balance fails. On the mean-zero subspace, `P` is a scaled unitary shift, so the nontrivial eigenvalues of `(P*)^k P^k` are `(1-eta)^(2k)` and `gamma_ps=max_k [1-(1-eta)^(2k)]/k=1-(1-eta)^2`. The runner verifies stationarity and this gap numerically from the transition matrix for every grid value.",
        "",
        "The score map spans the full `[0,1]` support: trainable-W risk is one on 9 states and zero on 10; SLSQP reverses those values, giving exact risk ratio `10/9=r_star` and paired contrast zero. Chebyshev risk is one on every state, giving a positive control contrast `9/19`. Hence one contrast is exactly at the null boundary while the other is genuinely positive; all arms share the same chain state.",
        "",
        "For the nonstationary rows, the initial law is `nu=(1-rho)pi+rho*delta_0` with `rho=TV/(1-1/19)`, so its total-variation distance from stationarity is exactly the declared 0.02. The radius uses `(alpha_total-TV)/3`, matching the helper's one-time familywise TV correction. These are exact synthetic facts, not inferred project-chain receipts.",
        "",
        "## Figure and flow",
        "",
        "![Non-reversible multi-state Markov coverage and release](multistate_markov_coverage.png)",
        "",
        "Text flow: `19-state paired chain -> exact uniform stationary law and pseudo-gap -> fixed bounded three-arm score map -> simultaneous denominator + two contrast lower bounds -> coverage / false-release accounting`.",
        "",
        "The three panels show simultaneous lower-bound coverage with Wilson Monte Carlo intervals; the fraction with positive trainable-W denominator and Chebyshev control bounds; and family release frequency when SLSQP is exactly at `r_star`. A positive Chebyshev bound is a useful power-side check, while a family release at the SLSQP boundary is a false release.",
        "",
        f"All 12 planned cells completed: per cell planned/valid/failed/abstained panels = `{PANELS_PER_CELL}/{PANELS_PER_CELL}/0/0`; pooled panels = `{len(rows)*PANELS_PER_CELL}/{len(rows)*PANELS_PER_CELL}/0/0`. Across all panels, the experiment scored `{pooled_macroblocks:,}` paired macroblocks. No chain or panel was dropped. The public decision helper was cross-checked against the vectorized calculation on one additional deterministic realized path per cell.",
        "",
        "|gamma_ps|blocks|TV|joint coverage [Wilson 95%]|W denominator lower bound positive|Chebyshev control lower bound positive|family release at SLSQP boundary [Wilson 95%]|",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ci = row["joint_coverage_wilson95_low"], row["joint_coverage_wilson95_high"]
        fci = row["family_release_wilson95"]
        readme.append(
            f"|{row['target_gamma_ps']:g}|{row['scored_blocks']:,}|{row['post_burn_in_tv_bound']:g}|"
            f"{row['joint_lower_bound_coverage']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]|"
            f"{row['denominator_positive_fraction']:.4f}|"
            f"{row['chebyshev_positive_control_lower_positive_fraction']:.4f}|"
            f"{row['family_release_at_slsqp_boundary_fraction']:.4f} [{fci[0]:.4f}, {fci[1]:.4f}]|"
        )
    readme += [
        "",
        f"Observed minimum joint coverage was `{min_coverage:.4f}`; the largest observed family release rate at the SLSQP boundary was `{max_family_release:.4f}` (largest Wilson upper endpoint `{min_family_upper:.4f}`). Monte Carlo intervals only quantify simulation uncertainty for this finite-state generator.",
        "",
        "## Interpretation and limit",
        "",
        "This extends the earlier two-state reversible calibration with a non-reversible, 19-state chain, a full-support score map, exact threshold-boundary contrast, a positive-control contrast, and an exact nonstationary initial-TV case. It is a distinct DGP and does not repeat the earlier two-state generator. The experiment checks that the support-only Markov bounds remain conservative for these known-chain cases; it does not establish coverage for HMC or infer a production `gamma_ps` / TV receipt. Because the method uses worst-case support bounds, high coverage can coexist with low release power.",
        "",
        "D04 reference-fit uncertainty, candidate-selection uncertainty, actual trainable-W training, authorized production receipts, independent statistical review and all sealed splits remain open. Keep `NOT_READY/UNCERTIFIED`, `physical_score=null`, and the sealed data closed.",
        "",
        "Sources: Paulin, corrected arXiv v5, Theorem 3.11 / Eq. (3.25), for the pseudo-spectral-gap Bernstein inequality: [paper](https://arxiv.org/html/1212.2015v5). The stationary-law and pseudo-gap values here are derived from the stated finite transition matrix, not taken from that paper as empirical facts.",
    ]
    (output / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    payloads = [output / name for name in (
        "README.md", "protocol.json", "coverage.csv", "multistate_markov_coverage.png",
    )]
    manifest = {"schema": "optimize-d06-markov-multistate-artifact-manifest-v1",
                "payload_sha256": {path.name: _sha256(path) for path in payloads}}
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "output": str(output),
        "cells": len(rows),
        "panels_per_cell": PANELS_PER_CELL,
        "planned_panels": len(rows) * PANELS_PER_CELL,
        "minimum_joint_coverage": min_coverage,
        "maximum_family_release_at_boundary": max_family_release,
        "maximum_wilson_upper_release_at_boundary": min_family_upper,
        "artifacts_hashed": len(manifest["payload_sha256"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
