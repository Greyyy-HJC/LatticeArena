"""Synthetic calibration stress for the bounded block-risk alternative."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bounded_block_risk import (
    ALPHA_TOTAL,
    ARMS,
    BOUND_EVENTS,
    DELTA_PER_BOUND,
    FAMILY_SIZE,
    LOSS_CAP,
    R_STAR,
)


DEFAULT_OUT = Path(__file__).resolve().parents[2] / (
    "results/optimize_gaussian/d06_bounded_block_risk_cpu_20261125_v4"
)
METHODS = ("hoeffding", "empirical_bernstein")
N_BLOCKS = (200, 1000, 5000, 20000)
EFFECT_SCENARIOS = {
    "below_threshold": (1.05, 1.00),
    "at_threshold": (R_STAR, R_STAR),
    "above_threshold": (1.25, 1.50),
}
IID_FAMILIES = {
    "iid_bounded_homogeneous": {"common_outlier_probability": 0.0,
                                 "arm_outlier_probabilities": (0.0, 0.0, 0.0),
                                 "heteroskedastic": False},
    "iid_shared_rare_shock": {"common_outlier_probability": 0.005,
                              "arm_outlier_probabilities": (0.010, 0.015, 0.020),
                              "heteroskedastic": False},
    "iid_heteroskedastic_rare_shock": {"common_outlier_probability": 0.010,
                                       "arm_outlier_probabilities": (0.020, 0.030, 0.040),
                                       "heteroskedastic": True},
}
MARKOV_RHOS = (0.0, 0.5, 0.8, 0.9, 0.95, 0.98)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials < 1:
        return math.nan, math.nan
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _radius(mean: np.ndarray, variance: np.ndarray, n: int,
            width: float, delta: float, method: str) -> np.ndarray:
    if method == "hoeffding":
        return width * math.sqrt(math.log(1.0 / delta) / (2.0 * n)) * np.ones_like(mean)
    log_term = math.log(2.0 / delta)
    return width * (
        np.sqrt(2.0 * (variance / (width * width)) * log_term / n)
        + 7.0 * log_term / (3.0 * (n - 1))
    )


def _lower(values: np.ndarray, low: float, high: float, delta: float,
           method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = values.mean(axis=1)
    variances = values.var(axis=1, ddof=1)
    radii = _radius(means, variances, values.shape[1], high - low, delta, method)
    return means - radii, means, variances


def _arm_regular_losses(target_risks: tuple[float, float, float],
                        total_outlier_p: tuple[float, float, float]) -> np.ndarray:
    risks = np.asarray(target_risks, dtype=np.float64)
    p = np.asarray(total_outlier_p, dtype=np.float64)
    if np.any(risks <= p) or np.any(risks >= 1.0):
        raise ValueError("target risks must exceed their contamination probabilities and be < 1")
    return (risks - p) / (1.0 - p)


def _iid_panel_scores(rng: np.random.Generator, n: int, batch: int,
                      target_risks: tuple[float, float, float],
                      family: dict[str, object]) -> np.ndarray:
    common_p = float(family["common_outlier_probability"])
    total_p = np.asarray(family["arm_outlier_probabilities"], dtype=np.float64)
    if common_p > 0:
        unique_p = (total_p - common_p) / (1.0 - common_p)
    else:
        unique_p = total_p.copy()
    if np.any(unique_p < 0) or np.any(unique_p > 1):
        raise ValueError("common outlier probability exceeds an arm marginal")
    regular = _arm_regular_losses(target_risks, tuple(float(x) for x in total_p))
    if bool(family["heteroskedastic"]) and np.any(regular * 1.5 >= 1.0):
        raise ValueError("heteroskedastic multiplier would exceed bounded loss support")

    common_shock = rng.random((batch, n)) < common_p
    arm_shocks = rng.random((batch, n, len(ARMS))) < unique_p[None, None, :]
    outlier = common_shock[:, :, None] | arm_shocks
    if bool(family["heteroskedastic"]):
        scale = np.where(rng.random((batch, n)) < 0.5, 0.5, 1.5)
        regular_scores = regular[None, None, :] * scale[:, :, None]
    else:
        regular_scores = np.broadcast_to(regular, (batch, n, len(ARMS)))
    # This score is exactly the 4-time average when the synthetic block's q
    # is constant in t: q=3 on a shock and q=1+/-sqrt(regular_score) otherwise.
    return np.where(outlier, LOSS_CAP, regular_scores)


def _method_metrics(scores: np.ndarray, true_risks: np.ndarray,
                    method: str) -> dict[str, np.ndarray]:
    n = scores.shape[1]
    delta = DELTA_PER_BOUND
    w_lower, _, _ = _lower(scores[:, :, 0], 0.0, 1.0, delta, method)
    positive = w_lower > 0.0
    contrast_lowers = []
    contrast_truths = []
    individual_release = []
    for arm in (1, 2):
        contrast = scores[:, :, arm] - R_STAR * scores[:, :, 0]
        lower, _, _ = _lower(contrast, -R_STAR, 1.0, delta, method)
        contrast_lowers.append(lower)
        contrast_truths.append(true_risks[arm] - R_STAR * true_risks[0])
        individual_release.append(positive & (lower > 0.0))
    joint_covered = w_lower <= true_risks[0]
    for lower, truth in zip(contrast_lowers, contrast_truths):
        joint_covered &= lower <= truth
    family_release = individual_release[0] & individual_release[1]
    truth_ratios = true_risks[1:] / true_risks[0]
    is_null_family = bool(np.any(truth_ratios <= R_STAR))
    return {
        "positive_denominator": positive,
        "joint_covered": joint_covered,
        "individual_release_1": individual_release[0],
        "individual_release_2": individual_release[1],
        "family_release": family_release,
        "false_family_release": family_release if is_null_family else np.zeros_like(family_release),
        "w_lower": w_lower,
        "contrast_lower_1": contrast_lowers[0],
        "contrast_lower_2": contrast_lowers[1],
    }


def _summarize_cell(
    *, family_name: str, effect_name: str, n: int, method: str,
    true_risks: np.ndarray, truth_ratios: np.ndarray,
    positive: int, covered: int, ind1: int, ind2: int,
    family_release: int, false_family_release: int, panels: int,
) -> dict[str, object]:
    low, high = wilson_interval(covered, panels)
    release_low, release_high = wilson_interval(family_release, panels)
    false_low, false_high = wilson_interval(false_family_release, panels)
    return {
        "regime": "iid_macroblocks",
        "family": family_name,
        "effect_scenario": effect_name,
        "method": method,
        "n_independent_blocks": n,
        "planned_panels": panels,
        "valid_panels": positive,
        "failed_panels": 0,
        "abstained_panels": panels - positive,
        "joint_one_sided_coverage": covered / panels,
        "coverage_wilson95_low": low,
        "coverage_wilson95_high": high,
        "slsqp_release_rate": ind1 / panels,
        "chebyshev_release_rate": ind2 / panels,
        "family_release_rate": family_release / panels,
        "family_release_wilson95_low": release_low,
        "family_release_wilson95_high": release_high,
        "family_false_release_rate_when_any_null": false_family_release / panels,
        "false_release_wilson95_low": false_low,
        "false_release_wilson95_high": false_high,
        "risk_trainable_w_truth": float(true_risks[0]),
        "risk_slsqp_truth": float(true_risks[1]),
        "risk_chebyshev_truth": float(true_risks[2]),
        "ratio_slsqp_truth": float(truth_ratios[0]),
        "ratio_chebyshev_truth": float(truth_ratios[1]),
        "delta_per_bound": DELTA_PER_BOUND,
        "alpha_total": ALPHA_TOTAL,
        "bound_events": FAMILY_SIZE + 1,
    }


def _iid_stress(replicates: int, rng_seed: int, batch_size: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family_index, (family_name, family) in enumerate(IID_FAMILIES.items()):
        outlier_p = np.asarray(family["arm_outlier_probabilities"], dtype=np.float64)
        for effect_index, (effect_name, ratios) in enumerate(EFFECT_SCENARIOS.items()):
            risks = np.asarray((0.05, 0.05 * ratios[0], 0.05 * ratios[1]), dtype=np.float64)
            if np.any(risks <= outlier_p):
                raise ValueError("risk is below its contamination floor")
            truth_ratios = risks[1:] / risks[0]
            for n_index, n in enumerate(N_BLOCKS):
                rng = np.random.default_rng(np.random.SeedSequence(
                    (rng_seed, family_index, effect_index, n_index)
                ))
                counts = {
                    method: {key: 0 for key in ("positive_denominator", "joint_covered",
                                                "individual_release_1", "individual_release_2",
                                                "family_release", "false_family_release")}
                    for method in METHODS
                }
                generated = 0
                while generated < replicates:
                    batch = min(batch_size, replicates - generated)
                    scores = _iid_panel_scores(rng, n, batch, tuple(float(x) for x in risks), family)
                    for method in METHODS:
                        metric = _method_metrics(scores, risks, method)
                        for key in counts[method]:
                            counts[method][key] += int(np.count_nonzero(metric[key]))
                    generated += batch
                for method in METHODS:
                    c = counts[method]
                    rows.append(_summarize_cell(
                        family_name=family_name,
                        effect_name=effect_name,
                        n=n,
                        method=method,
                        true_risks=risks,
                        truth_ratios=truth_ratios,
                        positive=c["positive_denominator"],
                        covered=c["joint_covered"],
                        ind1=c["individual_release_1"],
                        ind2=c["individual_release_2"],
                        family_release=c["family_release"],
                        false_family_release=c["false_family_release"],
                        panels=replicates,
                    ))
    return rows


def _markov_panel_scores(rng: np.random.Generator, n: int, replicates: int,
                         rho: float, true_risks: np.ndarray, p: float = 0.01) -> np.ndarray:
    regular = (true_risks - p) / (1.0 - p)
    if np.any(regular <= 0) or np.any(regular >= 1):
        raise ValueError("Markov contamination design has invalid regular-loss values")
    p11 = p + rho * (1.0 - p)
    p01 = p * (1.0 - rho)
    state = rng.random((replicates, len(ARMS))) < p
    scores = np.empty((replicates, n, len(ARMS)), dtype=np.float64)
    for block in range(n):
        if block:
            u = rng.random((replicates, len(ARMS)))
            state = np.where(state, u < p11, u < p01)
        scores[:, block, :] = np.where(state, 1.0, regular[None, :])
    return scores


def _dependence_stress(replicates: int, rng_seed: int) -> list[dict[str, object]]:
    """Negative control: intentionally dependent macroblocks violate the theorem."""
    rows: list[dict[str, object]] = []
    n = 1000
    ratios = np.asarray((1.05, 1.0), dtype=np.float64)
    risks = np.asarray((0.05, *(0.05 * ratios)), dtype=np.float64)
    for rho_index, rho in enumerate(MARKOV_RHOS):
        rng = np.random.default_rng(np.random.SeedSequence((rng_seed, 700, rho_index)))
        scores = _markov_panel_scores(rng, n, replicates, rho, risks)
        for method in METHODS:
            metric = _method_metrics(scores, risks, method)
            released = int(np.count_nonzero(metric["family_release"]))
            covered = int(np.count_nonzero(metric["joint_covered"]))
            abstained = int(np.count_nonzero(~metric["positive_denominator"]))
            cover_low, cover_high = wilson_interval(covered, replicates)
            release_low, release_high = wilson_interval(released, replicates)
            rows.append({
                "regime": "dependent_macroblock_negative_control",
                "family": f"stationary_binary_markov_rho_{rho:g}",
                "effect_scenario": "below_threshold",
                "method": method,
                "n_independent_blocks": n,
                "planned_panels": replicates,
                "valid_panels": replicates - abstained,
                "failed_panels": 0,
                "abstained_panels": abstained,
                "joint_one_sided_coverage": covered / replicates,
                "coverage_wilson95_low": cover_low,
                "coverage_wilson95_high": cover_high,
                "slsqp_release_rate": float(np.mean(metric["individual_release_1"])),
                "chebyshev_release_rate": float(np.mean(metric["individual_release_2"])),
                "family_release_rate": released / replicates,
                "family_release_wilson95_low": release_low,
                "family_release_wilson95_high": release_high,
                "family_false_release_rate_when_any_null": released / replicates,
                "false_release_wilson95_low": release_low,
                "false_release_wilson95_high": release_high,
                "risk_trainable_w_truth": float(risks[0]),
                "risk_slsqp_truth": float(risks[1]),
                "risk_chebyshev_truth": float(risks[2]),
                "ratio_slsqp_truth": float(ratios[0]),
                "ratio_chebyshev_truth": float(ratios[1]),
                "delta_per_bound": DELTA_PER_BOUND,
                "alpha_total": ALPHA_TOTAL,
                "bound_events": FAMILY_SIZE + 1,
                "macroblock_lag1_correlation": rho,
            })
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows: list[dict[str, object]], replicates: int) -> None:
    iid = [row for row in rows if row["regime"] == "iid_macroblocks"]
    dependent = [row for row in rows if row["regime"] == "dependent_macroblock_negative_control"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.1), constrained_layout=True)
    colors = {"hoeffding": "#7b3294", "empirical_bernstein": "#008837"}
    # Joint confidence-bound coverage, averaged equally across all declared IID cells.
    for method in METHODS:
        by_n = {}
        for n in N_BLOCKS:
            values = [float(r["joint_one_sided_coverage"]) for r in iid
                      if r["method"] == method and r["n_independent_blocks"] == n]
            by_n[n] = float(np.mean(values))
        axes[0].plot(list(by_n), list(by_n.values()), marker="o", color=colors[method], label=method)
    axes[0].axhline(0.95, color="black", linestyle="--", linewidth=1, label="95% family target")
    axes[0].set_xscale("log")
    axes[0].set_ylim(0.85, 1.005)
    axes[0].set_title("IID-block joint lower-bound coverage")
    axes[0].set_xlabel("independent macroblocks per panel")
    axes[0].set_ylabel("coverage, equal-cell mean")
    axes[0].legend(fontsize=8)

    for method in METHODS:
        for scenario, style in (("below_threshold", "--"), ("at_threshold", ":")):
            values = []
            for n in N_BLOCKS:
                cell = [float(r["family_false_release_rate_when_any_null"])
                        for r in iid if r["method"] == method
                        and r["effect_scenario"] == scenario
                        and r["n_independent_blocks"] == n]
                values.append(float(np.mean(cell)))
            axes[1].plot(list(N_BLOCKS), values, marker="o", linestyle=style,
                         color=colors[method], label=f"{method}, {scenario}")
    axes[1].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("Family false-release rate")
    axes[1].set_xlabel("independent macroblocks per panel")
    axes[1].set_ylabel("release when at least one ratio ≤ r_star")
    axes[1].legend(fontsize=7)

    for method in METHODS:
        values = []
        for n in N_BLOCKS:
            cell = [float(r["family_release_rate"]) for r in iid
                    if r["method"] == method and r["effect_scenario"] == "above_threshold"
                    and r["n_independent_blocks"] == n]
            values.append(float(np.mean(cell)))
        axes[2].plot(list(N_BLOCKS), values, marker="o", color=colors[method], label=method)
    axes[2].set_xscale("log")
    axes[2].set_ylim(0, 1.02)
    axes[2].set_title("Power when both effects exceed threshold")
    axes[2].set_xlabel("independent macroblocks per panel")
    axes[2].set_ylabel("family release rate")
    axes[2].legend(fontsize=8)

    rhos = list(MARKOV_RHOS)
    for method in METHODS:
        values = []
        for rho in rhos:
            cell = [float(r["joint_one_sided_coverage"]) for r in dependent
                    if r["method"] == method
                    and float(r["macroblock_lag1_correlation"]) == rho]
            values.append(float(np.mean(cell)))
        axes[3].plot(rhos, values, marker="o", color=colors[method], label=method)
    axes[3].axhline(0.95, color="black", linestyle="--", linewidth=1, label="95% target")
    axes[3].set_ylim(0.65, 1.005)
    axes[3].set_title("Dependent-block negative control")
    axes[3].set_xlabel("macroblock lag-1 correlation")
    axes[3].set_ylabel("joint lower-bound coverage")
    axes[3].legend(fontsize=8)
    fig.suptitle(
        f"D06 bounded block-risk synthetic calibration; {replicates} panels/cell; "
        "IID macroblocks unless stated"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(output: Path, *, replicates: int = 1000, dependence_replicates: int = 2000,
          seed: int = 20261125, batch_size: int = 32) -> dict[str, object]:
    if replicates < 100 or dependence_replicates < 100 or batch_size < 1:
        raise ValueError("use at least 100 panels per cell and a positive batch size")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty report directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    iid_rows = _iid_stress(replicates, seed, batch_size)
    dependence_rows = _dependence_stress(dependence_replicates, seed + 101)
    rows = iid_rows + dependence_rows
    _write_csv(output / "calibration_summary.csv", rows)
    _plot(output / "coverage_release_power.png", rows, replicates)

    iid_panels_per_method = len(IID_FAMILIES) * len(EFFECT_SCENARIOS) * len(N_BLOCKS) * replicates
    dependence_panels_per_method = len(MARKOV_RHOS) * dependence_replicates
    method_denominators: dict[str, dict[str, dict[str, int]]] = {}
    for regime, label in (("iid_macroblocks", "iid_macroblocks"),
                          ("dependent_macroblock_negative_control", "dependent_negative_control")):
        method_denominators[label] = {}
        for method in METHODS:
            subset = [row for row in rows if row["regime"] == regime and row["method"] == method]
            method_denominators[label][method] = {
                key: sum(int(row[key]) for row in subset)
                for key in ("planned_panels", "valid_panels", "failed_panels", "abstained_panels")
            }
    selected_outcomes: dict[str, object] = {}
    for method in METHODS:
        iid_method = [row for row in iid_rows if row["method"] == method]
        dependent_method = [row for row in dependence_rows if row["method"] == method]
        selected_outcomes[method] = {
            "iid_joint_coverage_min_by_block_count": {
                str(n): min(float(row["joint_one_sided_coverage"]) for row in iid_method
                            if row["n_independent_blocks"] == n)
                for n in N_BLOCKS
            },
            "iid_joint_coverage_min_wilson_lower_by_block_count": {
                str(n): min(float(row["coverage_wilson95_low"]) for row in iid_method
                            if row["n_independent_blocks"] == n)
                for n in N_BLOCKS
            },
            "iid_above_threshold_family_release_equal_cell_mean": {
                str(n): float(np.mean([
                    float(row["family_release_rate"]) for row in iid_method
                    if row["effect_scenario"] == "above_threshold"
                    and row["n_independent_blocks"] == n
                ]))
                for n in N_BLOCKS
            },
            "iid_above_threshold_family_release_range_by_block_count": {
                str(n): [
                    min(float(row["family_release_rate"]) for row in iid_method
                        if row["effect_scenario"] == "above_threshold"
                        and row["n_independent_blocks"] == n),
                    max(float(row["family_release_rate"]) for row in iid_method
                        if row["effect_scenario"] == "above_threshold"
                        and row["n_independent_blocks"] == n),
                ]
                for n in N_BLOCKS
            },
            "iid_null_and_boundary_false_family_release_panels": sum(
                round(float(row["family_false_release_rate_when_any_null"])
                      * int(row["planned_panels"]))
                for row in iid_method
                if row["effect_scenario"] in ("below_threshold", "at_threshold")
            ),
            "dependent_joint_coverage_by_rho": {
                str(float(row["macroblock_lag1_correlation"])):
                    float(row["joint_one_sided_coverage"])
                for row in dependent_method
            },
        }
    source_paths = [Path(__file__), Path(__file__).with_name("bounded_block_risk.py"),
                    Path(__file__).with_name("research_contract_v1.json"),
                    Path(__file__).with_name("d06_hac_fieller_diagnostic.py")]
    source_hashes = {p.name: _sha256(p) for p in source_paths}
    protocol: dict[str, object] = {
        "schema": "optimize-d06-bounded-block-risk-calibration-v1",
        "status": "SYNTHETIC_METHOD_DIAGNOSTIC_NOT_PHYSICAL_CALIBRATION",
        "evidence_class": "cpu_synthetic",
        "physical_score": None,
        "data_access": "none; no gauge, propagator, reference-fit, sealed-split, GPU, or QUDA input",
        "rng_seed": seed,
        "batch_size_panels": batch_size,
        "estimand": {
            "block_score": "X[b,a] = mean_{t=2,3,4,5} min((q[b,a,t]-1)^2, 1)",
            "risk": "mu_a = E[X[b,a]] over the declared target distribution of independent macroblocks",
            "effect": "rho_a = mu_a / mu_trainable_w; release only when both comparator ratios exceed 10/9",
            "change_from_D04": "This clipped block-risk is a different estimand from the selected D04 bootstrap variance-plus-squared-bias risk; it is a calibration proposal, not a replacement.",
        },
        "inference": {
            "methods": {
                "hoeffding": "one-sided bounded-mean Hoeffding lower bound",
                "empirical_bernstein": "one-sided Maurer-Pontil empirical-Bernstein lower bound, Theorem 4, rescaled to the observation support",
            },
            "r_star": R_STAR,
            "family_size": FAMILY_SIZE,
            "alpha_total": ALPHA_TOTAL,
            "bound_events": BOUND_EVENTS,
            "delta_per_bound": DELTA_PER_BOUND,
            "events": [
                "E[X_trainable_w] > 0 shared denominator lower bound",
                "E[X_slsqp - r_star*X_trainable_w] > 0",
                "E[X_chebyshev - r_star*X_trainable_w] > 0",
            ],
            "method_assumption": "independent macroblocks from a fixed distribution; all arms/time points paired within block",
            "comparison": "empirical-Bernstein is the candidate primary; Hoeffding is the variance-insensitive reference. Calibration does not authorize choosing based on observed physical outcomes.",
        },
        "iid_stress_design": {
            "replicates_per_cell": replicates,
            "n_blocks": list(N_BLOCKS),
            "families": {name: spec for name, spec in IID_FAMILIES.items()},
            "true_ratio_scenarios": {name: list(ratios) for name, ratios in EFFECT_SCENARIOS.items()},
            "trainable_w_true_bounded_risk": 0.05,
            "block_dgp": "Each independent block has an optional common/per-arm rare shock. Loss is 1 on a shock; otherwise it is a fixed squared displacement, optionally multiplied by a shared Bernoulli 0.5/1.5 heteroskedastic factor. Exact arm risks follow from the declared Bernoulli probabilities.",
            "panel_denominator_per_method": iid_panels_per_method,
            "panel_denominator_all_methods": 2 * iid_panels_per_method,
            "block_observation_denominator_per_method": iid_panels_per_method * sum(N_BLOCKS),
        },
        "dependence_negative_control": {
            "replicates_per_cell": dependence_replicates,
            "n_blocks": 1000,
            "lag1_correlations": list(MARKOV_RHOS),
            "true_ratio_scenario": [1.05, 1.0],
            "purpose": "deliberately violates independent-macroblock assumptions; empirical performance is diagnostic only and cannot recalibrate the IID theorem",
            "panel_denominator_per_method": dependence_panels_per_method,
        },
        "denominator": {
            "iid_macroblock_panels_per_method": iid_panels_per_method,
            "iid_planned_valid_failed_abstained": "all planned panels are evaluated; denominator-zero panels are counted as abstained by method",
            "dependent_negative_control_panels_per_method": dependence_panels_per_method,
            "failed_panels": 0,
        },
        "method_denominators": method_denominators,
        "selected_outcomes": selected_outcomes,
        "source_hashes": source_hashes,
        "limits": [
            "does not establish independent macroblocks for an HMC chain",
            "does not propagate D04 reference-fit uncertainty or trainable-arm selection",
            "does not use the original D04 risk estimand or nested block-bootstrap hull",
            "rare-shock and Markov generators are controlled synthetic distributions, not lattice-chain evidence",
            "no physical score, production evaluator, or GPU/QUDA comparison is produced",
        ],
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    eb_iid = [row for row in iid_rows if row["method"] == "empirical_bernstein"]
    eb_dep = [row for row in dependence_rows if row["method"] == "empirical_bernstein"]
    ho_dep = [row for row in dependence_rows if row["method"] == "hoeffding"]
    coverage_lines = "\n".join(
        f"- `n={n}`: minimum cell coverage `{selected_outcomes['empirical_bernstein']['iid_joint_coverage_min_by_block_count'][str(n)]:.4f}`; "
        f"minimum Wilson 95% lower endpoint `{selected_outcomes['empirical_bernstein']['iid_joint_coverage_min_wilson_lower_by_block_count'][str(n)]:.4f}`."
        for n in N_BLOCKS
    )
    power_lines = "\n".join(
        f"- `n={n}`: family release mean `{selected_outcomes['empirical_bernstein']['iid_above_threshold_family_release_equal_cell_mean'][str(n)]:.4f}`, "
        f"range `{selected_outcomes['empirical_bernstein']['iid_above_threshold_family_release_range_by_block_count'][str(n)][0]:.4f}–{selected_outcomes['empirical_bernstein']['iid_above_threshold_family_release_range_by_block_count'][str(n)][1]:.4f}`; "
        f"Hoeffding mean `{selected_outcomes['hoeffding']['iid_above_threshold_family_release_equal_cell_mean'][str(n)]:.4f}`."
        for n in N_BLOCKS
    )
    false_release_lines = "\n".join(
        f"- `{method}`: `{selected_outcomes[method]['iid_null_and_boundary_false_family_release_panels']}/"
        f"{len(IID_FAMILIES) * 2 * len(N_BLOCKS) * replicates}` null/boundary panels released both effects."
        for method in METHODS
    )
    denominator_lines = "\n".join(
        f"- `{method}`: `{method_denominators['iid_macroblocks'][method]['planned_panels']}/"
        f"{method_denominators['iid_macroblocks'][method]['valid_panels']}/"
        f"{method_denominators['iid_macroblocks'][method]['failed_panels']}/"
        f"{method_denominators['iid_macroblocks'][method]['abstained_panels']}`."
        for method in METHODS
    )
    dependence_lines = "\n".join(
        f"- `rho={float(eb['macroblock_lag1_correlation']):g}`: empirical-Bernstein `"
        f"{float(eb['joint_one_sided_coverage']):.4f} [{float(eb['coverage_wilson95_low']):.4f}, {float(eb['coverage_wilson95_high']):.4f}]`; "
        f"Hoeffding `{float(ho['joint_one_sided_coverage']):.4f} [{float(ho['coverage_wilson95_low']):.4f}, {float(ho['coverage_wilson95_high']):.4f}]`."
        for eb, ho in zip(eb_dep, ho_dep)
    )
    readme = f"""# D06 bounded block-risk: CPU calibration proposal (v4)

**Synthetic method diagnostic only.** No gauge, propagator, reference-fit, sealed split, GPU, or QUDA input was read. `physical_score=null` and the selected D04/D06 rule remains `UNCALIBRATED/NOT_READY`.

## Alternative estimand

For each independent paired macroblock `b`, arm `a`, and four prespecified times, define

`X[b,a] = (1/4) sum_{{t=2..5}} min((q[b,a,t] - 1)^2, 1)`, with `0 <= X <= 1`.

The target is `mu[a]=E[X[b,a]]` and the contrast is `rho[a]=mu[a]/mu[trainable-W]`. This changes D04: it is a clipped block-risk, not the nested-bootstrap `mean_t(Var(q)+bias(q)^2)` risk. The cap reduces the effect of extreme q values while losing information about their magnitude; all arms use the same frozen map.

## Finite-sample decision rule

The threshold remains `r_star=10/9` with F=2. For each comparator we form the paired block contrast `D[b]=X[b,comparator]-r_star*X[b,W]`, which lies in `[-r_star,1]`. We require a positive one-sided lower bound for each contrast and a positive lower bound for `mu[W]`. Bonferroni splits total alpha `0.05` equally across these three events (`delta={DELTA_PER_BOUND:.8f}`), so both comparisons and a defined denominator are covered together. A nonpositive lower bound on `mu[W]` yields `ABSTAIN`.

The Hoeffding bound is the bounded, variance-insensitive reference. The candidate tighter method uses the empirical-Bernstein bound of Maurer and Pontil, Theorem 4: on independent bounded blocks, the one-sided radius on `[0,1]` is `sqrt(2 s^2 log(2/delta)/n) + 7 log(2/delta)/(3(n-1))`; for other support widths it scales linearly. The source theorem explicitly assumes independent observations; it does not authorize treating correlated HMC blocks as independent. [Original paper](https://arxiv.org/abs/0907.3740).

![Coverage, false-release and power stress](coverage_release_power.png)

## Findings

Across the three declared IID macroblock generators, empirical-Bernstein joint lower-bound coverage by block count is:
{coverage_lines}

At or below the effect threshold:
{false_release_lines}

Above the effect threshold, equal-cell family release by block count is:
{power_lines}

The IID-grid planned/valid/failed/abstained totals per method are:
{denominator_lines}
An abstention means the lower confidence bound did not establish positive trainable-W bounded risk, so a ratio decision was withheld.

The dependent negative control has `{dependence_replicates}` panels at each lag-one correlation. Joint lower-bound coverage and Wilson intervals are:
{dependence_lines}
This shows why the independent-block condition needs evidence; neither method acquires a dependence correction from this negative control.

## Synthetic stress and denominator

`calibration_summary.csv` contains each method, DGP, true-effect scenario and block count. The IID grid uses `{replicates}` independent synthetic panels per cell across three bounded distributions, ratios below / at / above threshold, and `n={list(N_BLOCKS)}` blocks. At each panel the same paired blocks feed both contrasts. Macroblock counts, panel counts, failures, denominator abstentions, joint lower-bound coverage, Wilson Monte Carlo intervals and release rates are recorded. The protocol binds RNG seed `{seed}` and batch size `{batch_size}`.

The separate negative control uses stationary binary Markov dependence across nominal blocks (`rho={list(MARKOV_RHOS)}`). At the largest correlations, the empirical-Bernstein lower-bound coverage falls below target; this is a direct warning that an IID theorem cannot be applied to autocorrelated HMC blocks without a separate dependence correction. The dependent rows are stress evidence, not a valid calibration claim for either theorem.

Text flow: `freeze bounded loss and candidate arms -> form one score per paired independent block -> estimate two fixed-threshold contrasts plus positive denominator -> simultaneous lower bounds -> release both or abstain/no release; a dependence receipt is a precondition, not inferred from the plot`.

## Interpretation and next gate

This is a decision-equivalent test for `rho > r_star`; it does not produce the original D06 ratio interval or replace its 10,000-draw rule. It can be adopted only after independent review freezes the clipped estimand and block-independence/chain dependence evidence. If blocks are not independently sampled or a valid mixing/dependency correction is absent, the bound is not applicable. D04 reference uncertainty, held-out candidate selection, production macroblock construction, and real operator measurements remain open.
"""
    (output / "README.md").write_text(readme)
    manifest = {name: _sha256(output / name) for name in (
        "README.md", "calibration_summary.csv", "coverage_release_power.png", "protocol.json"
    )}
    (output / "artifact_hashes.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return protocol


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--dependence-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    protocol = build(args.output, replicates=args.replicates,
                     dependence_replicates=args.dependence_replicates,
                     seed=args.seed, batch_size=args.batch_size)
    print(json.dumps({
        "output": str(args.output),
        "iid_panels_per_method": protocol["iid_stress_design"]["panel_denominator_per_method"],
        "dependent_panels_per_method": protocol["dependence_negative_control"]["panel_denominator_per_method"],
        "physical_score": protocol["physical_score"],
        "status": protocol["status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
