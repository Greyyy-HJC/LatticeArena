"""Pilot coverage calibration for the selected D04/D06 percentile-hull rule.

This is a synthetic stationary Gaussian AR(1) study. It does not read lattice
data, fit correlators, or authorize a physical comparison. The bootstrap risk
inner variance is computed analytically for the empirical circular-block
resampling distribution, which is the infinite-inner-draw limit of D04.
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

from .risk_statistics import D06_BLOCK_LENGTHS, D06_R_STAR, d06_paired_ratio_hull


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/optimize_gaussian/d06_percentile_hull_coverage_cpu_20261125_v7"
ARMS = ("trainable_W", "same_budget_SLSQP", "fixed_degree4_Chebyshev")
N_CONFIGURATIONS = 40
N_CONFIGURATION_SCENARIOS = (40, 60, 100)
OUTER_DRAWS = 1000
MC_REPLICATES = 250
BASE_SEED = 20261125
COMMON_NOISE_FRACTION = 0.60
TIME_POINTS = (2, 3, 4, 5)
MEANS = np.asarray([
    [1.075, 1.080, 1.085, 1.080],
    [1.078, 1.084, 1.087, 1.083],
    [1.090, 1.092, 1.100, 1.095],
], dtype=float)
SIGMAS = np.asarray([0.040, 0.042, 0.050], dtype=float)
PHI_SCENARIOS = (0.0, 0.4, 0.8)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ar1(rng: np.random.Generator, n: int, trailing_shape: tuple[int, ...], phi: float) -> np.ndarray:
    values = np.empty((n, *trailing_shape), dtype=float)
    values[0] = rng.normal(size=trailing_shape)
    innovation_scale = math.sqrt(1.0 - phi * phi)
    for index in range(1, n):
        values[index] = phi * values[index - 1] + innovation_scale * rng.normal(size=trailing_shape)
    return values


def generate_paired_ar1_panel(seed: int, phi: float, n: int = N_CONFIGURATIONS) -> np.ndarray:
    """Return q[cfg, arm, t] with shared noise to preserve pairing."""
    rng = np.random.default_rng(seed)
    common = _ar1(rng, n, (len(TIME_POINTS),), phi)
    independent = _ar1(rng, n, (len(ARMS), len(TIME_POINTS)), phi)
    independent_fraction = math.sqrt(1.0 - COMMON_NOISE_FRACTION**2)
    noise = (COMMON_NOISE_FRACTION * common[:, None, :]
             + independent_fraction * independent)
    return MEANS[None, :, :] + SIGMAS[None, :, None] * noise


def _batch_bootstrap_risk(samples: np.ndarray, block_length: int) -> np.ndarray:
    """Analytic D04 risk for each empirical circular-block bootstrap law.

    If k=n/L circular length-L blocks are sampled with replacement, the mean
    of the resampled configuration series is the mean of k sampled block
    means. Its conditional variance is Var(block means)/k. Averaging all n
    circular starts gives the exact empirical bootstrap law's first two
    moments, avoiding a second random inner bootstrap in this calibration.
    """
    if samples.ndim != 4:
        raise ValueError("samples must have shape (batch, configurations, arms, times)")
    batch, n, arms, times = samples.shape
    if n % block_length:
        raise ValueError("configuration count must be divisible by block_length")
    blocks_per_draw = n // block_length
    block_sum = np.zeros_like(samples)
    for offset in range(block_length):
        block_sum += np.roll(samples, -offset, axis=1)
    circular_block_means = block_sum / block_length
    mean_q = np.mean(samples, axis=1)
    bootstrap_variance = np.var(circular_block_means, axis=1, ddof=0) / blocks_per_draw
    risk = np.mean(bootstrap_variance + np.square(mean_q - 1.0), axis=-1)
    if risk.shape != (batch, arms) or not np.isfinite(risk).all() or np.any(risk <= 0):
        raise ValueError("synthetic risk must be finite and strictly positive")
    return risk


def _outer_bootstrap_risks(
    panel: np.ndarray, *, block_length: int, draws: int, seed: int,
) -> np.ndarray:
    n = panel.shape[0]
    blocks_per_draw = n // block_length
    if n % block_length:
        raise ValueError("configuration count must be divisible by block_length")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(draws, blocks_per_draw))
    within = np.arange(block_length)
    indices = ((starts[:, :, None] + within[None, None, :]) % n).reshape(draws, n)
    bootstrap_panels = panel[indices]
    return _batch_bootstrap_risk(bootstrap_panels, block_length)


def exact_ar1_sample_mean_variance(n: int, sigma: float, phi: float) -> float:
    """Exact Var(mean) for a stationary Gaussian AR(1) series of length n."""
    lag_sum = sum((n - lag) * phi**lag for lag in range(1, n))
    return sigma * sigma * (n + 2.0 * lag_sum) / (n * n)


def true_risks(phi: float, n: int = N_CONFIGURATIONS) -> dict[str, float]:
    result = {}
    for arm_index, arm in enumerate(ARMS):
        variance = exact_ar1_sample_mean_variance(
            n, float(SIGMAS[arm_index]), phi,
        )
        result[arm] = float(np.mean(np.square(MEANS[arm_index] - 1.0) + variance))
    return result


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _one_panel_decisions(
    panel: np.ndarray, phi: float, seed: int, outer_draws: int,
) -> dict[str, Any]:
    risks_by_comparison = {
        "SLSQP": ({length: [] for length in D06_BLOCK_LENGTHS},
                  {length: [] for length in D06_BLOCK_LENGTHS}),
        "Chebyshev": ({length: [] for length in D06_BLOCK_LENGTHS},
                      {length: [] for length in D06_BLOCK_LENGTHS}),
    }
    for length in D06_BLOCK_LENGTHS:
        risk_draws = _outer_bootstrap_risks(
            panel, block_length=length, draws=outer_draws,
            seed=seed + 10_000 + length,
        )
        for comparison, comparator_index in (("SLSQP", 1), ("Chebyshev", 2)):
            comp_map, candidate_map = risks_by_comparison[comparison]
            comp_map[length] = risk_draws[:, comparator_index]
            candidate_map[length] = risk_draws[:, 0]

    outputs = {}
    ids = {length: np.arange(outer_draws, dtype=np.int64) for length in D06_BLOCK_LENGTHS}
    truths = true_risks(phi, panel.shape[0])
    for comparison, comparator_arm in (("SLSQP", ARMS[1]), ("Chebyshev", ARMS[2])):
        comp_map, candidate_map = risks_by_comparison[comparison]
        result = d06_paired_ratio_hull(
            comp_map, candidate_map, ids, ids,
            planned_draws_per_length=outer_draws,
        )
        outputs[comparison] = {
            "true_risk_ratio": truths[comparator_arm] / truths[ARMS[0]],
            "hull_lower": result["hull_lower"],
            "covered": result["hull_lower"] <= truths[comparator_arm] / truths[ARMS[0]],
            "clears_r_star": result["hull_lower"] > D06_R_STAR,
            "block_length_lowers": {
                length: result["block_results"][str(length)]["lower_quantile"]
                for length in D06_BLOCK_LENGTHS
            },
            "status": result["status"],
            "draw_denominators": result["draw_denominators"],
        }
    outputs["familywise_covered"] = all(outputs[name]["covered"] for name in ("SLSQP", "Chebyshev"))
    outputs["familywise_clears_r_star"] = all(outputs[name]["clears_r_star"] for name in ("SLSQP", "Chebyshev"))
    outputs["planned_outer_draws_each_comparison"] = outer_draws * len(D06_BLOCK_LENGTHS)
    return outputs


def build_report(
    output: str | Path = DEFAULT_OUT, *, mc_replicates: int = MC_REPLICATES,
    outer_draws: int = OUTER_DRAWS, base_seed: int = BASE_SEED,
    n_scenarios: tuple[int, ...] = N_CONFIGURATION_SCENARIOS,
    phi_scenarios: tuple[float, ...] = PHI_SCENARIOS,
) -> dict:
    if mc_replicates < 1 or outer_draws < 40:
        raise ValueError("need positive MC replicates and at least 40 outer draws")
    draw_count = int(outer_draws)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    summaries = []
    if not n_scenarios or any(n < 10 or any(n % length for length in D06_BLOCK_LENGTHS)
                              for n in n_scenarios):
        raise ValueError("each n_scenario must be >=10 and divisible by L=2,5,10")
    if not phi_scenarios or any(not 0 <= phi < 1 for phi in phi_scenarios):
        raise ValueError("phi_scenarios must be nonempty values in [0,1)")
    for n_index, n in enumerate(n_scenarios):
        for phi_index, phi in enumerate(phi_scenarios):
            scenario_index = n_index * len(phi_scenarios) + phi_index
            scenario_rows = []
            for replicate in range(mc_replicates):
                seed = base_seed + scenario_index * 1_000_000 + replicate
                panel = generate_paired_ar1_panel(seed, phi, n)
                decisions = _one_panel_decisions(panel, phi, seed, draw_count)
                row = {
                    "n_configurations": n,
                    "phi": phi,
                    "replicate": replicate,
                    "seed": seed,
                    "true_ratio_SLSQP": decisions["SLSQP"]["true_risk_ratio"],
                    "lower_SLSQP": decisions["SLSQP"]["hull_lower"],
                    "covered_SLSQP": int(decisions["SLSQP"]["covered"]),
                    "clears_SLSQP": int(decisions["SLSQP"]["clears_r_star"]),
                    "true_ratio_Chebyshev": decisions["Chebyshev"]["true_risk_ratio"],
                    "lower_Chebyshev": decisions["Chebyshev"]["hull_lower"],
                    "covered_Chebyshev": int(decisions["Chebyshev"]["covered"]),
                    "clears_Chebyshev": int(decisions["Chebyshev"]["clears_r_star"]),
                    "familywise_covered": int(decisions["familywise_covered"]),
                    "familywise_clears_r_star": int(decisions["familywise_clears_r_star"]),
                }
                for comparison in ("SLSQP", "Chebyshev"):
                    for length, lower in decisions[comparison]["block_length_lowers"].items():
                        row[f"lower_{comparison}_L{length}"] = lower
                rows.append(row)
                scenario_rows.append(decisions)

            for key, extractor in (
                ("SLSQP", lambda d: d["SLSQP"]["covered"]),
                ("Chebyshev", lambda d: d["Chebyshev"]["covered"]),
                ("familywise", lambda d: d["familywise_covered"]),
            ):
                successes = sum(bool(extractor(decision)) for decision in scenario_rows)
                summaries.append({
                    "n_configurations": n,
                    "phi": phi,
                    "unit": key,
                    "planned": mc_replicates,
                    "valid": mc_replicates,
                    "failed": 0,
                    "abstained": 0,
                    "coverage": successes / mc_replicates,
                    "wilson_95": _wilson_interval(successes, mc_replicates),
                    "mc_standard_error": math.sqrt(
                        (successes / mc_replicates) * (1.0 - successes / mc_replicates) / mc_replicates,
                    ),
                })

    block_length_summaries = []
    for n in n_scenarios:
        for phi in phi_scenarios:
            cell_rows = [row for row in rows
                         if row["n_configurations"] == n and row["phi"] == phi]
            for length in D06_BLOCK_LENGTHS:
                for comparison in ("SLSQP", "Chebyshev"):
                    successes = sum(
                        row[f"true_ratio_{comparison}"] >= row[f"lower_{comparison}_L{length}"]
                        for row in cell_rows
                    )
                    block_length_summaries.append({
                        "n_configurations": n, "phi": phi,
                        "unit": comparison, "block_length": length,
                        "planned": mc_replicates, "valid": mc_replicates,
                        "failed": 0, "abstained": 0,
                        "coverage": successes / mc_replicates,
                        "wilson_95": _wilson_interval(successes, mc_replicates),
                    })
                family_successes = sum(
                    row["true_ratio_SLSQP"] >= row[f"lower_SLSQP_L{length}"]
                    and row["true_ratio_Chebyshev"] >= row[f"lower_Chebyshev_L{length}"]
                    for row in cell_rows
                )
                block_length_summaries.append({
                    "n_configurations": n, "phi": phi,
                    "unit": "familywise", "block_length": length,
                    "planned": mc_replicates, "valid": mc_replicates,
                    "failed": 0, "abstained": 0,
                    "coverage": family_successes / mc_replicates,
                    "wilson_95": _wilson_interval(family_successes, mc_replicates),
                })

    protocol = {
        "schema": "optimize-d06-percentile-hull-coverage-pilot-v1",
        "evidence_class": "synthetic_stationary_gaussian_ar1_only",
        "status": "PILOT_CALIBRATION_NOT_PHYSICAL_EVIDENCE",
        "n_configuration_scenarios": list(n_scenarios),
        "mc_replicates_per_phi": mc_replicates,
        "outer_draws_per_block_length": draw_count,
        "selected_contract_draws_per_block_length": 10_000,
        "block_lengths": list(D06_BLOCK_LENGTHS),
        "quantile": 0.025,
        "quantile_method": "linear",
        "r_star": D06_R_STAR,
        "family_size": 2,
        "phi_scenarios": list(phi_scenarios),
        "paired_arms": list(ARMS),
        "time_points": list(TIME_POINTS),
        "q_means_by_arm_and_time": MEANS.tolist(),
        "q_sigma_by_arm": SIGMAS.tolist(),
        "cross_arm_common_noise_fraction": COMMON_NOISE_FRACTION,
        "risk_truth": "mean_t[(mu_arm,t-1)^2 + exact Var(sample_mean_t)] for a stationary AR(1) with known marginal sigma and phi",
        "inner_bootstrap": "analytic infinite-draw circular-block empirical bootstrap variance; avoids finite inner-bootstrap Monte Carlo noise",
        "seed_rule": "base_seed + scenario_index*1000000 + replicate; outer draw seed adds 10000 + block_length",
        "base_seed": base_seed,
        "data_access": "none",
        "csv_line_terminator": "LF",
        "physical_score": None,
        "source_hashes": {
            "d06_bootstrap_coverage_diagnostic.py": _sha256(Path(__file__)),
            "risk_statistics.py": _sha256(Path(__file__).with_name("risk_statistics.py")),
            "research_contract_v1.json": _sha256(Path(__file__).with_name("research_contract_v1.json")),
        },
    }
    with (out / "replicates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (out / "block_length_coverage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(block_length_summaries[0]), lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(block_length_summaries)
    (out / "coverage_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (out / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )

    table = "\n".join(
        f"| {row['n_configurations']} | {row['phi']:.1f} | {row['unit']} | {row['coverage']:.3f} | "
        f"[{row['wilson_95'][0]:.3f}, {row['wilson_95'][1]:.3f}] | "
        f"{row['planned']}/{row['valid']}/{row['failed']}/{row['abstained']} |"
        for row in summaries
    )
    targeted_length_rows = [row for row in block_length_summaries
                            if row["n_configurations"] == 60 and row["phi"] == 0.8]
    length_table = "\n".join(
        f"| {row['unit']} | {row['block_length']} | {row['coverage']:.3f} | "
        f"[{row['wilson_95'][0]:.3f}, {row['wilson_95'][1]:.3f}] | "
        f"{row['planned']}/{row['valid']}/{row['failed']}/{row['abstained']} |"
        for row in targeted_length_rows
    )
    length_section = (
        "## Individual block-length coverage at the targeted high-dependence point\n\n"
        "The L=2/5/10 rows below are the un-hulled marginal intervals at `n=60, phi=0.8`. "
        "The complete per-cell/per-length results are in `block_length_coverage.csv`.\n\n"
        "| interval family | L | coverage | Monte Carlo Wilson 95% | panels P/V/F/A |\n"
        "| --- | ---: | ---: | ---: | ---: |\n"
        f"{length_table}"
        if targeted_length_rows else
        "The selected `n=60, phi=0.8` sensitivity cell is not included in this run; "
        "per-length results for the requested cells are in `block_length_coverage.csv`."
    )
    targeted_contract_count = (
        draw_count == 10_000 and tuple(n_scenarios) == (60,) and tuple(phi_scenarios) == (0.8,)
    )
    draw_description = (
        "This targeted high-correlation cell uses the selected 10,000 draws per length, but it covers only one synthetic `(n, phi)` setting."
        if targeted_contract_count else
        f"This pilot uses {draw_count} outer draws per length; the selected contract uses 10,000."
    )
    readme = f"""# Optimize D06 paired percentile-hull coverage pilot

**Evidence class:** synthetic stationary Gaussian AR(1) only. No lattice data, correlators, fits, GPU, QUDA, or sealed splits were accessed. This pilot does not certify physical coverage. It used {draw_count} outer draws and {mc_replicates} Monte Carlo panels per `(n, phi)` cell; the selected production protocol requires 10,000 paired outer draws per block length.

## Text flow

```text
known stationary paired Gaussian AR(1) process for q(theta,t), t=2..5
  -> n in {list(n_scenarios)} and phi in {list(phi_scenarios)}, shared noise across all three arms
  -> for each L=2,5,10, paired circular-block outer resamples
  -> recompute empirical D04 risk using exact moments of the inner circular-block bootstrap law
  -> paired comparator/trainable-W risk ratios; linear 0.025 quantile per L
  -> take the minimum L lower quantile; check it against known generating-process ratio
  -> evaluate marginal and F=2 simultaneous coverage; retain all planned panels
```

The inner block-bootstrap variance is computed analytically: if `k=n/L` circular blocks are drawn with replacement, the conditional variance of their mean is the population variance of the `n` circular length-`L` block means divided by `k`. This is the infinite-inner-draw limit of the selected D04 bootstrap statistic; it isolates outer percentile-hull behavior from finite inner-bootstrap noise. The outer procedure uses {draw_count} finite draws per block length. {draw_description}

For each arm and time, the truth is `bias^2 + Var(sample mean)`; the AR(1) variance is evaluated exactly as `sigma^2/n^2 * [n + 2*sum_(h=1)^(n-1)((n-h)*phi^h)]`. The same stationary `phi` is used for all arms; a common Gaussian component keeps the arms paired. Both user-selected contrasts are evaluated jointly, with `alpha=0.025` per contrast and `r_star=10/9`.

## Empirical coverage

The table reports the fraction of Monte Carlo panels whose lower hull did not exceed the known risk ratio, Wilson 95% intervals for the Monte Carlo coverage rate, and `planned/valid/failed/abstained` panel counts.

| n | phi | interval family | coverage | Monte Carlo Wilson 95% | panels P/V/F/A |
| ---: | ---: | --- | ---: | ---: | ---: |
{table}

{length_section}

The raw per-panel ratios and each block-length lower quantile are in `replicates.csv`; exact generator inputs and source hashes are in `protocol.json`. The Monte Carlo interval quantifies finite simulation noise only. These Gaussian AR(1) cases do not represent the gauge-chain dependence, fitted-reference uncertainty, optimizer failures, or the production evaluator. The observed undercoverage is a warning for this nested percentile implementation under the specified AR(1) model, not a measured property of the lattice chain and not a PASS for either comparison.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    payloads = sorted(path for path in out.iterdir() if path.is_file())
    hashes = {path.name: _sha256(path) for path in payloads}
    (out / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return {
        "output": str(out),
        "mc_replicates_per_phi": mc_replicates,
        "outer_draws_per_length": draw_count,
        "coverage_summary": summaries,
        "physical_score": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mc-replicates", type=int, default=MC_REPLICATES)
    parser.add_argument("--outer-draws", type=int, default=OUTER_DRAWS)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--n-scenarios", type=int, nargs="+", default=list(N_CONFIGURATION_SCENARIOS))
    parser.add_argument("--phi-scenarios", type=float, nargs="+", default=list(PHI_SCENARIOS))
    args = parser.parse_args()
    result = build_report(
        args.output, mc_replicates=args.mc_replicates,
        outer_draws=args.outer_draws, base_seed=args.seed,
        n_scenarios=tuple(args.n_scenarios), phi_scenarios=tuple(args.phi_scenarios),
    )
    print(json.dumps({
        "output": result["output"],
        "mc_replicates_per_phi": result["mc_replicates_per_phi"],
        "outer_draws_per_length": result["outer_draws_per_length"],
        "physical_score": result["physical_score"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
