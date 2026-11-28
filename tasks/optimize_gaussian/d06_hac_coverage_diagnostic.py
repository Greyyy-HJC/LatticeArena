"""Compare an influence-HAC lower bound with D06's percentile hull.

The study is CPU-only and reuses the already archived synthetic AR(1) panels.
The HAC interval is a method candidate, not a production-calibrated interval.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .d06_bootstrap_coverage_diagnostic import (
    ARMS,
    COMMON_NOISE_FRACTION,
    MEANS,
    SIGMAS,
    TIME_POINTS,
    _wilson_interval,
    generate_paired_ar1_panel,
    true_risks,
)
from .risk_statistics import D04_TIMES, D06_ALPHA_PER_COMPARISON, D06_BLOCK_LENGTHS


ROOT = Path(__file__).resolve().parents[2]
BASE_REPORT_DEFAULT = ROOT / "results/optimize_gaussian/d06_percentile_hull_coverage_cpu_20261125_v8"
DEFAULT_OUT = ROOT / "results/optimize_gaussian/d06_hac_coverage_cpu_20261125_v1"
HAC_BANDWIDTHS = (2, 5, 10, 20, 30)
COMPARISONS = (("SLSQP", 1), ("Chebyshev", 2))
PAIRED_BOOTSTRAP_DRAWS = 10_000
PAIRED_BOOTSTRAP_SEED = 2026112501


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def d04_risk_and_influence(panel: np.ndarray, block_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return finite-L empirical D04 risk and n-scaled per-configuration IF.

    ``panel`` is q[cfg, arm, t]. The risk uses exact moments of the empirical
    circular-block bootstrap law, the same inner-risk convention as the
    archived percentile pilot. The returned influence sums to the first-order
    perturbation of the estimator divided by n after centering.
    """
    if panel.ndim != 3:
        raise ValueError("panel must have shape (configurations, arms, times)")
    n, arms, times = panel.shape
    if n % block_length or block_length < 1:
        raise ValueError("n must be divisible by a positive block_length")
    mean_q = np.mean(panel, axis=0)
    block_sum = np.zeros_like(panel)
    for offset in range(block_length):
        block_sum += np.roll(panel, -offset, axis=0)
    block_means = block_sum / block_length
    block_deviation = block_means - mean_q[None, :, :]
    empirical_bootstrap_variance = (
        block_length / n * np.mean(np.square(block_deviation), axis=0)
    )
    risk_by_time = np.square(mean_q - 1.0) + empirical_bootstrap_variance
    risk = np.mean(risk_by_time, axis=1)

    # For each observation i, sum deviations of all circular blocks containing i.
    indices = np.arange(n)
    containing_deviation = np.zeros_like(panel)
    for offset in range(block_length):
        containing_deviation += block_deviation[(indices - offset) % n]
    per_configuration_per_time = (
        2.0 * (mean_q[None, :, :] - 1.0)
        + (2.0 / n) * containing_deviation
    )
    influence = np.mean(per_configuration_per_time, axis=2)
    if (not np.isfinite(risk).all() or np.any(risk <= 0)
            or not np.isfinite(influence).all()):
        raise ValueError("risk/influence is non-finite or risk is nonpositive")
    return risk, influence


def newey_west_long_run_variance(values: np.ndarray, bandwidth: int) -> float:
    """Bartlett-kernel Newey-West estimate with divisor n at every lag."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or bandwidth < 0:
        raise ValueError("values must be a 1-D series and bandwidth nonnegative")
    x = x - np.mean(x)
    n = x.size
    omega = float(np.dot(x, x) / n)
    for lag in range(1, min(bandwidth, n - 1) + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        gamma = float(np.dot(x[:-lag], x[lag:]) / n)
        omega += 2.0 * weight * gamma
    return omega


def hac_ratio_lowers(
    panel: np.ndarray,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Build L x HAC-bandwidth ratio lower bounds for both D06 contrasts."""
    z = NormalDist().inv_cdf(1.0 - D06_ALPHA_PER_COMPARISON)
    lowers: dict[str, dict[str, float]] = {name: {} for name, _ in COMPARISONS}
    point_by_comparison: dict[str, dict[str, float]] = {
        name: {} for name, _ in COMPARISONS
    }
    for length in D06_BLOCK_LENGTHS:
        risks, influences = d04_risk_and_influence(panel, length)
        for name, comparator_index in COMPARISONS:
            candidate_risk = float(risks[0])
            ratio_hat = float(risks[comparator_index] / candidate_risk)
            ratio_if = (
                influences[:, comparator_index] - ratio_hat * influences[:, 0]
            ) / candidate_risk
            point_by_comparison[name][f"L{length}"] = ratio_hat
            for bandwidth in HAC_BANDWIDTHS:
                key = f"L{length}_b{bandwidth}"
                omega = newey_west_long_run_variance(ratio_if, bandwidth)
                if not math.isfinite(omega) or omega <= 0:
                    lowers[name][key] = math.nan
                    continue
                standard_error = math.sqrt(omega / panel.shape[0])
                raw_lower = ratio_hat - z * standard_error
                # Risk ratios are positive; intersecting the lower confidence
                # set with the parameter space at zero adds no epsilon.
                lowers[name][key] = max(0.0, raw_lower)
    return lowers, point_by_comparison


def _coverage_summary(values: list[bool], planned: int, unit: str) -> dict[str, Any]:
    valid = len(values)
    successes = sum(values)
    return {
        "unit": unit,
        "planned": planned,
        "valid": valid,
        "failed": 0,
        "abstained": planned - valid,
        "coverage": successes / valid if valid else None,
        "wilson_95": _wilson_interval(successes, valid) if valid else None,
        "mc_standard_error": (
            math.sqrt((successes / valid) * (1.0 - successes / valid) / valid)
            if valid else None
        ),
    }


def _load_baseline(
    baseline_report: str | Path = BASE_REPORT_DEFAULT,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    base_report = Path(baseline_report)
    if not base_report.is_absolute():
        base_report = ROOT / base_report
    protocol_path = base_report / "protocol.json"
    manifest_path = base_report / "artifact_hashes.json"
    replicate_path = base_report / "replicates.csv"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("replicates.csv") != _sha256(replicate_path):
        raise ValueError("baseline replicate CSV does not match its artifact hash manifest")
    n_scenarios = protocol.get("n_configuration_scenarios")
    phi_scenarios = protocol.get("phi_scenarios")
    if (not isinstance(n_scenarios, list) or not isinstance(phi_scenarios, list)
            or not n_scenarios or not phi_scenarios):
        raise ValueError("baseline protocol must list n and phi scenarios")
    with replicate_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    per_cell = int(protocol["mc_replicates_per_phi"])
    expected = len(n_scenarios) * len(phi_scenarios) * per_cell
    if len(rows) != expected or per_cell != 250:
        raise ValueError("baseline must retain 250 planned panels in every cell")
    identities = {
        (int(row["n_configurations"]), float(row["phi"]), int(row["replicate"]))
        for row in rows
    }
    if len(identities) != expected:
        raise ValueError("baseline (n, phi, replicate) identities must be unique")
    allowed = {(int(n), float(phi)) for n in n_scenarios for phi in phi_scenarios}
    if any((int(row["n_configurations"]), float(row["phi"])) not in allowed for row in rows):
        raise ValueError("baseline contains an undeclared n/phi cell")
    return rows, protocol, base_report


def _build_rows(
    baseline_report: str | Path = BASE_REPORT_DEFAULT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_rows, baseline_protocol, base_report = _load_baseline(baseline_report)
    rows = []
    denominators: dict[str, dict[str, int]] = {}
    for source in baseline_rows:
        seed = int(source["seed"])
        n = int(source["n_configurations"])
        phi = float(source["phi"])
        panel = generate_paired_ar1_panel(seed, phi, n)
        truths = true_risks(phi, n)
        truth_ratios = {
            name: truths[ARMS[comparator_arm]] / truths[ARMS[0]]
            for name, comparator_arm in COMPARISONS
        }
        for name, _ in COMPARISONS:
            if not math.isclose(
                float(source[f"true_ratio_{name}" if name == "SLSQP" else "true_ratio_Chebyshev"]),
                truth_ratios[name], rel_tol=1e-12, abs_tol=1e-12,
            ):
                raise ValueError("baseline true ratio differs from the declared generator truth")

        lower_by_comparison, point_ratios = hac_ratio_lowers(panel)
        cell_key = f"n{n}_phi{phi:g}"
        denominator = denominators.setdefault(
            cell_key, {"planned": 0, "valid": 0, "failed": 0, "abstained": 0},
        )
        denominator["planned"] += 1
        row: dict[str, Any] = {
            "replicate": int(source["replicate"]),
            "seed": seed,
            "n_configurations": n,
            "phi": phi,
        }
        all_methods_valid = True
        for name, _ in COMPARISONS:
            truth = truth_ratios[name]
            percentile_lower = float(source[f"lower_{name}"])
            lower_values = lower_by_comparison[name]
            valid_lowers = [value for value in lower_values.values() if math.isfinite(value)]
            hac_valid = len(valid_lowers) == len(D06_BLOCK_LENGTHS) * len(HAC_BANDWIDTHS)
            all_methods_valid &= hac_valid
            hac_lower = min(valid_lowers) if hac_valid else math.nan
            row[f"true_ratio_{name}"] = truth
            row[f"percentile_hull_lower_{name}"] = percentile_lower
            row[f"percentile_covered_{name}"] = int(percentile_lower <= truth)
            for length, point_ratio in point_ratios[name].items():
                row[f"hac_point_ratio_{name}_{length}"] = point_ratio
            row[f"hac_hull_lower_{name}"] = hac_lower
            row[f"hac_covered_{name}"] = int(hac_lower <= truth) if hac_valid else None
            row[f"hac_valid_{name}"] = int(hac_valid)
            for key, lower in lower_values.items():
                row[f"hac_lower_{name}_{key}"] = lower
        row["percentile_familywise_covered"] = int(
            bool(row["percentile_covered_SLSQP"] and row["percentile_covered_Chebyshev"])
        )
        row["hac_familywise_covered"] = int(
            bool(row["hac_covered_SLSQP"] and row["hac_covered_Chebyshev"])
            if all_methods_valid else False
        )
        row["hac_familywise_valid"] = int(all_methods_valid)
        if all_methods_valid:
            denominator["valid"] += 1
        else:
            denominator["abstained"] += 1
        rows.append(row)
    for denominator in denominators.values():
        if denominator["planned"] != denominator["valid"] + denominator["failed"] + denominator["abstained"]:
            raise AssertionError("panel denominator does not balance")
    return rows, {
        "baseline_protocol": baseline_protocol,
        "baseline_report": base_report,
        "hac_panel_denominators": denominators,
    }


def _plot(summaries: list[dict[str, Any]], rows: list[dict[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.23, wspace=0.14)
    units = ["SLSQP", "Chebyshev", "familywise"]
    x = np.arange(len(units))
    width = 0.34
    colors = ("#2563EB", "#D97706")
    for method_index, method in enumerate(("percentile", "hac")):
        subset = [next(item for item in summaries
                       if item["method"] == method and item["unit"] == unit)
                  for unit in units]
        coverage = np.asarray([item["coverage"] for item in subset], dtype=float)
        lows = np.asarray([item["wilson_95"][0] for item in subset], dtype=float)
        highs = np.asarray([item["wilson_95"][1] for item in subset], dtype=float)
        xpos = x + (method_index - 0.5) * width
        method_label = "Percentile hull" if method == "percentile" else "Influence-HAC hull"
        axes[0].bar(xpos, coverage, width=width, color=colors[method_index], label=method_label)
        axes[0].errorbar(xpos, coverage, yerr=[coverage-lows, highs-coverage],
                         fmt="none", ecolor="#111827", capsize=3, linewidth=1)
    axes[0].axhline(1.0 - D06_ALPHA_PER_COMPARISON, color="#6B7280", linestyle="--",
                    linewidth=1, label="marginal nominal 97.5%")
    axes[0].axhline(0.95, color="#9CA3AF", linestyle=":", linewidth=1,
                    label="F=2 familywise target 95%")
    axes[0].set_xticks(x, units)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_title("Same 250 AR(1) panels")
    from matplotlib.lines import Line2D
    for index, comparison in enumerate(("SLSQP", "Chebyshev")):
        truth = float(rows[0][f"true_ratio_{comparison}"])
        percentile = np.asarray([r[f"percentile_hull_lower_{comparison}"] for r in rows], dtype=float)
        hac = np.asarray([r[f"hac_hull_lower_{comparison}"] for r in rows], dtype=float)
        pos = index + 1
        axes[1].scatter(np.full(percentile.size, pos-0.13), percentile, s=7, alpha=0.25,
                        color=colors[0], edgecolors="none")
        axes[1].scatter(np.full(hac.size, pos+0.13), hac, s=7, alpha=0.25,
                        color=colors[1], edgecolors="none")
        axes[1].hlines(truth, pos-0.32, pos+0.32, color=("#111827" if index == 0 else "#4B5563"),
                       linestyle=("--" if index == 0 else "-."), linewidth=1.2,
                       label=f"{comparison} true ratio")
    axes[1].set_xticks([1, 2], ["SLSQP", "Chebyshev"])
    axes[1].set_ylabel("Lower endpoint")
    axes[1].set_title("Panel lower bounds vs known truth")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(0.26, 0.105, "Bars: blue = percentile hull; orange = influence-HAC hull",
             ha="center", fontsize=8)
    fig.text(0.26, 0.065, "Lines: dashed = 97.5% marginal target; dotted = 95% F=2 target",
             ha="center", fontsize=8)
    fig.text(0.76, 0.105, "Right panel: dashed / dash-dot show each known true ratio",
             ha="center", fontsize=8)
    fig.suptitle("Optimize D06: percentile vs influence-HAC (synthetic only)")
    fig.savefig(out / "coverage_and_bounds.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(output: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    out = Path(output)
    rows, source_context = _build_rows()
    out.mkdir(parents=True, exist_ok=False)
    planned = len(rows)
    summaries: list[dict[str, Any]] = []
    for method, flag in (("percentile", "percentile_covered"), ("hac", "hac_covered")):
        for unit, _ in (*COMPARISONS, ("familywise", -1)):
            column = f"{flag}_{unit}" if unit != "familywise" else f"{method}_familywise_covered"
            if method == "hac" and unit != "familywise":
                is_valid = lambda row, u=unit: bool(row[f"hac_valid_{u}"])
            elif method == "hac":
                is_valid = lambda row: bool(row["hac_familywise_valid"])
            else:
                is_valid = lambda row: True
            included = [bool(row[column]) for row in rows if is_valid(row)]
            summaries.append({"method": method, **_coverage_summary(included, planned, unit)})

    paired_rows = []
    for index, unit in enumerate(("SLSQP", "Chebyshev", "familywise")):
        percentile_column = (
            "percentile_familywise_covered" if unit == "familywise"
            else f"percentile_covered_{unit}"
        )
        hac_column = "hac_familywise_covered" if unit == "familywise" else f"hac_covered_{unit}"
        valid_rows = [row for row in rows if row["hac_familywise_valid"]]
        percentile_values = np.asarray([row[percentile_column] for row in valid_rows], dtype=np.int8)
        hac_values = np.asarray([row[hac_column] for row in valid_rows], dtype=np.int8)
        difference = hac_values - percentile_values
        rng = np.random.default_rng(PAIRED_BOOTSTRAP_SEED + index)
        sample_indices = rng.integers(
            0, len(valid_rows), size=(PAIRED_BOOTSTRAP_DRAWS, len(valid_rows)),
        )
        bootstrap_difference = np.mean(difference[sample_indices], axis=1)
        paired_rows.append({
            "unit": unit,
            "planned": planned,
            "valid_paired_panels": len(valid_rows),
            "both_cover": int(np.sum((percentile_values == 1) & (hac_values == 1))),
            "percentile_only_covers": int(np.sum((percentile_values == 1) & (hac_values == 0))),
            "hac_only_covers": int(np.sum((percentile_values == 0) & (hac_values == 1))),
            "neither_covers": int(np.sum((percentile_values == 0) & (hac_values == 0))),
            "hac_minus_percentile_coverage": float(np.mean(difference)),
            "paired_panel_bootstrap_95": [
                float(np.quantile(bootstrap_difference, 0.025, method="linear")),
                float(np.quantile(bootstrap_difference, 0.975, method="linear")),
            ],
            "bootstrap_resamples": PAIRED_BOOTSTRAP_DRAWS,
            "bootstrap_seed": PAIRED_BOOTSTRAP_SEED + index,
        })

    summary_path = out / "coverage_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(summaries[0]), lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summaries)
    replicate_path = out / "replicates.csv"
    with replicate_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    paired_path = out / "paired_method_comparison.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(paired_rows[0]), lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(paired_rows)
    _plot(summaries, rows, out)

    protocol = {
        "schema": "optimize-d06-hac-coverage-diagnostic-v1",
        "evidence_class": "synthetic_stationary_gaussian_ar1_only",
        "status": "HAC_METHOD_CANDIDATE_NOT_PRODUCTION_CALIBRATION",
        "data_access": "reused_open_synthetic_panel_seeds_only",
        "physical_score": None,
        "n_configurations": 60,
        "phi": 0.8,
        "mc_panels": len(rows),
        "paired_arm_order": list(ARMS),
        "time_points": list(TIME_POINTS),
        "q_means_by_arm_and_time": MEANS.tolist(),
        "q_sigma_by_arm": SIGMAS.tolist(),
        "cross_arm_common_noise_fraction": COMMON_NOISE_FRACTION,
        "risk_truth": "mean_t[(mu_arm,t-1)^2 + exact Var(sample_mean_t)]",
        "true_risk_ratios": {
            name: true_risks(0.8, 60)[ARMS[index]] / true_risks(0.8, 60)[ARMS[0]]
            for name, index in COMPARISONS
        },
        "empirical_d04_risk": "exact moments of the empirical circular-block bootstrap law, separately for L=2,5,10",
        "hac_method": {
            "risk_influence": "analytic derivative of finite-L empirical D04 risk; ratio influence uses paired arms",
            "kernel": "Newey-West Bartlett",
            "bandwidths": list(HAC_BANDWIDTHS),
            "block_lengths": list(D06_BLOCK_LENGTHS),
            "normal_critical_value": NormalDist().inv_cdf(1.0-D06_ALPHA_PER_COMPARISON),
            "lower_endpoint": "max(0, ratio_hat - z_0.975 * sqrt(omega_hat / n))",
            "hull": "minimum over all L x HAC bandwidth combinations; invalid cell forces panel abstention",
        },
        "percentile_comparator": {
            "source_report": str(BASE_REPORT.relative_to(ROOT)),
            "outer_draws_per_block_length": 10_000,
            "alpha_per_comparison": D06_ALPHA_PER_COMPARISON,
            "block_lengths": list(D06_BLOCK_LENGTHS),
        },
        "family_size": 2,
        "paired_coverage_difference": {
            "resamples": PAIRED_BOOTSTRAP_DRAWS,
            "seed_rule": "2026112501 + unit_index in SLSQP, Chebyshev, familywise order",
            "interval": "percentile bootstrap over the same paired Monte Carlo panels",
        },
        "source_hashes": {
            "d06_hac_coverage_diagnostic.py": _sha256(Path(__file__)),
            "d06_bootstrap_coverage_diagnostic.py": _sha256(Path(__file__).with_name("d06_bootstrap_coverage_diagnostic.py")),
            "risk_statistics.py": _sha256(Path(__file__).with_name("risk_statistics.py")),
            "baseline_protocol.json": _sha256(BASE_REPORT / "protocol.json"),
            "baseline_replicates.csv": _sha256(BASE_REPORT / "replicates.csv"),
        },
        "baseline_artifact_hashes": json.loads((BASE_REPORT / "artifact_hashes.json").read_text()),
        "panel_denominator": source_context["hac_panel_denominator"],
        "not_inferred": [
            "actual gauge-chain autocorrelation or dependence envelope",
            "finite-sample validity of Newey-West under the production chain",
            "independent D04 reference-fit uncertainty",
            "production D06 coverage or physical improvement",
        ],
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    method_rows = {row["method"]: row for row in summaries if row["unit"] == "familywise"}
    readme = f"""# Optimize D06 influence-HAC interval comparison

**Evidence class:** a synthetic stationary Gaussian AR(1) calibration candidate using the same 250 `n=60, phi=0.8` panels and seeds as the archived 10,000-draw percentile-hull report. No lattice data, fits, GPU/QUDA, or sealed split was accessed. This does not calibrate production coverage.

## Text flow

```text
the 250 open synthetic panel seeds from percentile report v8
  -> regenerate the same paired q(t) panels and verify the known risk ratios
  -> for each D04 L=2,5,10, compute exact empirical circular-bootstrap risk and analytic influence series
  -> paired risk-ratio influence -> Newey-West Bartlett HAC variance, bandwidth 2/5/10/20/30
  -> one-sided normal lower endpoint at alpha=0.025; take the minimum across L x bandwidth
  -> compare with the existing percentile hull on the identical panels and known generating truth
```

For each time `t`, the finite-`L` empirical D04 risk is
`(mean(q_t)-1)^2 + (L/n) mean_s[(blockmean_(s,L,t)-mean(q_t))^2]`.
The code differentiates this quantity with respect to each configuration, averages over `t=2..5`, and forms the paired ratio influence
`(IF_comparator - ratio_hat * IF_trainableW) / risk_trainableW`. Newey-West uses the Bartlett kernel. The one-sided lower endpoint is truncated at zero using the known positive parameter space; no epsilon is added.

## Coverage and denominator

![Coverage and panel-level lower endpoints](coverage_and_bounds.png)

| method | unit | coverage | Wilson 95% Monte Carlo interval | planned/valid/failed/abstained |
| --- | --- | ---: | --- | --- |
{chr(10).join(f"| {r['method']} | {r['unit']} | {r['coverage']:.3f} | [{r['wilson_95'][0]:.3f}, {r['wilson_95'][1]:.3f}] | {r['planned']}/{r['valid']}/{r['failed']}/{r['abstained']} |" for r in summaries)}

The familywise rows give the rate that both contrasts cover simultaneously. For this finite-dependence stress point the percentile familywise coverage is `{method_rows['percentile']['coverage']:.3f}` and the HAC-hull coverage is `{method_rows['hac']['coverage']:.3f}`. See `coverage_and_bounds.png` for Wilson Monte Carlo intervals and panel-level lower endpoints against the known truth.

The marginal target is 97.5% per comparison; the Bonferroni familywise target is at least 95%. A paired resampling of the same Monte Carlo panels gives HAC-minus-percentile coverage changes: {', '.join(f"{item['unit']} {100*item['hac_minus_percentile_coverage']:+.1f} pp (95% paired-panel interval {100*item['paired_panel_bootstrap_95'][0]:+.1f} to {100*item['paired_panel_bootstrap_95'][1]:+.1f} pp)" for item in paired_rows)}. These paired intervals quantify Monte Carlo panel variation for this declared DGP only.

## Interpretation boundary

This is a candidate comparison, not an endorsement of HAC as the D06 production interval. Newey-West requires an asymptotic linear approximation and suitable weak-dependence/moment conditions; the observed 250-panel Monte Carlo rate only checks this one Gaussian AR(1) point. The HAC bandwidth hull is deliberately conservative, but taking a minimum cannot repair a violated CLT or an unrepresented chain. It does not include independent D04 fit/reference uncertainty or optimizer failure modes. The percentile baseline separately undercovered in this same declared synthetic family. Neither method is certified; retain `UNCALIBRATED/NOT_READY`, `physical_score=null`, and the sealed-data boundary.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    shutil.copyfile(Path(__file__), out / "source_snapshot.py")
    payloads = sorted(p for p in out.iterdir() if p.is_file())
    manifest = {p.name: _sha256(p) for p in payloads}
    (out / "artifact_hashes.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"output": str(out), "summaries": summaries, "physical_score": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = build_report(args.output)
    print(json.dumps({"output": result["output"], "summaries": result["summaries"],
                      "physical_score": result["physical_score"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
