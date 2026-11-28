"""Extend the D06 influence-HAC comparison over the archived AR(1) grid."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .d06_bootstrap_coverage_diagnostic import (
    ARMS,
    _wilson_interval,
    generate_paired_ar1_panel,
    true_risks,
)
from .d06_hac_coverage_diagnostic import (
    HAC_BANDWIDTHS,
    PAIRED_BOOTSTRAP_DRAWS,
    PAIRED_BOOTSTRAP_SEED,
    hac_ratio_lowers,
)
from .risk_statistics import D06_ALPHA_PER_COMPARISON, D06_BLOCK_LENGTHS


ROOT = Path(__file__).resolve().parents[2]
BASE_REPORT = ROOT / "results/optimize_gaussian/d06_percentile_hull_coverage_cpu_20261125_v7"
DEFAULT_OUT = ROOT / "results/optimize_gaussian/d06_hac_grid_coverage_cpu_20261125_v1"
COMPARISONS = (("SLSQP", 1), ("Chebyshev", 2))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_baseline() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol_path = BASE_REPORT / "protocol.json"
    replicate_path = BASE_REPORT / "replicates.csv"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    hashes = json.loads((BASE_REPORT / "artifact_hashes.json").read_text(encoding="utf-8"))
    if hashes.get("replicates.csv") != _sha256(replicate_path):
        raise ValueError("baseline CSV does not match its artifact hash manifest")
    if protocol.get("n_configuration_scenarios") != [40, 60, 100]:
        raise ValueError("expected the archived n=40/60/100 baseline grid")
    if protocol.get("phi_scenarios") != [0.0, 0.4, 0.8]:
        raise ValueError("expected the archived phi=0/0.4/0.8 baseline grid")
    if protocol.get("outer_draws_per_block_length") != 1_000:
        raise ValueError("expected the declared 1,000 outer draws per block length")
    with replicate_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    per_cell = int(protocol["mc_replicates_per_phi"])
    expected = len(protocol["n_configuration_scenarios"]) * len(protocol["phi_scenarios"]) * per_cell
    if len(rows) != expected or per_cell != 250:
        raise ValueError("baseline must retain all 250 panels in each declared cell")
    identities = {
        (int(row["n_configurations"]), float(row["phi"]), int(row["replicate"]))
        for row in rows
    }
    if len(identities) != expected:
        raise ValueError("baseline panel identities are not unique")
    return rows, protocol


def _build_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_rows, baseline_protocol = _load_baseline()
    rows = []
    cell_denominators: dict[str, dict[str, int]] = {}
    for source in baseline_rows:
        n = int(source["n_configurations"])
        phi = float(source["phi"])
        replicate = int(source["replicate"])
        seed = int(source["seed"])
        panel = generate_paired_ar1_panel(seed, phi=phi, n=n)
        true = true_risks(phi, n)
        true_ratios = {
            name: true[ARMS[index]] / true[ARMS[0]] for name, index in COMPARISONS
        }
        for name, _ in COMPARISONS:
            if not np.isclose(float(source[f"true_ratio_{name}"]), true_ratios[name], rtol=1e-12, atol=1e-12):
                raise ValueError("baseline known truth does not match the generator")
        lower_by_comparison, point_ratios = hac_ratio_lowers(panel)
        cell_key = f"n{n}_phi{phi:g}"
        denom = cell_denominators.setdefault(
            cell_key, {"planned": 0, "valid": 0, "failed": 0, "abstained": 0},
        )
        denom["planned"] += 1
        row: dict[str, Any] = {
            "n_configurations": n,
            "phi": phi,
            "replicate": replicate,
            "seed": seed,
        }
        all_valid = True
        for name, _ in COMPARISONS:
            percentile_lower = float(source[f"lower_{name}"])
            lower_map = lower_by_comparison[name]
            valid_lowers = [value for value in lower_map.values() if np.isfinite(value)]
            is_valid = len(valid_lowers) == len(D06_BLOCK_LENGTHS) * len(HAC_BANDWIDTHS)
            all_valid &= is_valid
            hac_lower = min(valid_lowers) if is_valid else float("nan")
            row[f"true_ratio_{name}"] = true_ratios[name]
            row[f"percentile_hull_lower_{name}"] = percentile_lower
            row[f"percentile_covered_{name}"] = int(percentile_lower <= true_ratios[name])
            row[f"hac_hull_lower_{name}"] = hac_lower
            row[f"hac_covered_{name}"] = int(hac_lower <= true_ratios[name]) if is_valid else None
            row[f"hac_valid_{name}"] = int(is_valid)
            for length, ratio_hat in point_ratios[name].items():
                row[f"hac_point_ratio_{name}_{length}"] = ratio_hat
            for label, lower in lower_map.items():
                row[f"hac_lower_{name}_{label}"] = lower
        row["percentile_familywise_covered"] = int(
            bool(row["percentile_covered_SLSQP"] and row["percentile_covered_Chebyshev"])
        )
        row["hac_familywise_covered"] = int(
            bool(row["hac_covered_SLSQP"] and row["hac_covered_Chebyshev"])
            if all_valid else False
        )
        row["hac_familywise_valid"] = int(all_valid)
        if all_valid:
            denom["valid"] += 1
        else:
            denom["abstained"] += 1
        rows.append(row)
    for denom in cell_denominators.values():
        if denom["planned"] != denom["valid"] + denom["failed"] + denom["abstained"]:
            raise AssertionError("cell denominator does not balance")
    return rows, {"baseline_protocol": baseline_protocol, "cell_denominators": cell_denominators}


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    cells = sorted({(int(row["n_configurations"]), float(row["phi"])) for row in rows})
    for n, phi in cells:
        cell = [row for row in rows if row["n_configurations"] == n and row["phi"] == phi]
        planned = len(cell)
        for method in ("percentile", "hac"):
            for unit in ("SLSQP", "Chebyshev", "familywise"):
                flag = (
                    f"{method}_familywise_covered" if unit == "familywise"
                    else f"{method}_covered_{unit}"
                )
                valid_rows = [
                    row for row in cell
                    if (row["hac_familywise_valid"] if method == "hac" and unit == "familywise"
                        else row[f"hac_valid_{unit}"] if method == "hac" else True)
                ]
                values = [bool(row[flag]) for row in valid_rows]
                successes = sum(values)
                valid = len(values)
                summaries.append({
                    "n_configurations": n,
                    "phi": phi,
                    "method": method,
                    "unit": unit,
                    "planned": planned,
                    "valid": valid,
                    "failed": 0,
                    "abstained": planned - valid,
                    "coverage": successes / valid if valid else None,
                    "wilson_95": _wilson_interval(successes, valid) if valid else None,
                })
    return summaries


def _paired_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    cells = sorted({(int(row["n_configurations"]), float(row["phi"])) for row in rows})
    unit_names = ("SLSQP", "Chebyshev", "familywise")
    unit_index = {name: i for i, name in enumerate(unit_names)}
    for cell_index, (n, phi) in enumerate(cells):
        cell = [row for row in rows if row["n_configurations"] == n and row["phi"] == phi]
        valid_rows = [row for row in cell if row["hac_familywise_valid"]]
        for unit in unit_names:
            p_col = f"percentile_familywise_covered" if unit == "familywise" else f"percentile_covered_{unit}"
            h_col = f"hac_familywise_covered" if unit == "familywise" else f"hac_covered_{unit}"
            p = np.asarray([row[p_col] for row in valid_rows], dtype=np.int8)
            h = np.asarray([row[h_col] for row in valid_rows], dtype=np.int8)
            delta = h - p
            seed = PAIRED_BOOTSTRAP_SEED + cell_index * 10 + unit_index[unit]
            rng = np.random.default_rng(seed)
            indices = rng.integers(0, len(valid_rows), size=(PAIRED_BOOTSTRAP_DRAWS, len(valid_rows)))
            delta_boot = np.mean(delta[indices], axis=1)
            results.append({
                "n_configurations": n,
                "phi": phi,
                "unit": unit,
                "planned": len(cell),
                "valid_paired_panels": len(valid_rows),
                "percentile_only_covers": int(np.sum((p == 1) & (h == 0))),
                "hac_only_covers": int(np.sum((p == 0) & (h == 1))),
                "hac_minus_percentile_coverage": float(np.mean(delta)),
                "paired_panel_bootstrap_95": [
                    float(np.quantile(delta_boot, 0.025, method="linear")),
                    float(np.quantile(delta_boot, 0.975, method="linear")),
                ],
                "bootstrap_resamples": PAIRED_BOOTSTRAP_DRAWS,
                "bootstrap_seed": seed,
            })
    return results


def _plot(summaries: list[dict[str, Any]], out: Path) -> None:
    n_values = sorted({int(item["n_configurations"]) for item in summaries})
    phi_values = sorted({float(item["phi"]) for item in summaries})
    fig, axes = plt.subplots(len(n_values), 2, figsize=(12.5, 3.8 * len(n_values)), squeeze=False)
    colors = {"percentile": "#2563EB", "hac": "#D97706"}
    bar_width = 0.32
    for row_index, n in enumerate(n_values):
        fwer_ax, marginal_ax = axes[row_index]
        x = np.arange(len(phi_values))
        for method_index, method in enumerate(("percentile", "hac")):
            fwer_rows = [next(item for item in summaries if item["n_configurations"] == n
                               and item["phi"] == phi and item["method"] == method
                               and item["unit"] == "familywise") for phi in phi_values]
            values = np.asarray([item["coverage"] for item in fwer_rows], dtype=float)
            low = np.asarray([item["wilson_95"][0] for item in fwer_rows], dtype=float)
            high = np.asarray([item["wilson_95"][1] for item in fwer_rows], dtype=float)
            xpos = x + (method_index - 0.5) * bar_width
            fwer_ax.bar(xpos, values, width=bar_width, color=colors[method],
                        label="Percentile hull" if method == "percentile" else "Influence-HAC hull")
            fwer_ax.errorbar(xpos, values, yerr=[values-low, high-values], fmt="none",
                             ecolor="#111827", capsize=3, linewidth=1)
        fwer_ax.axhline(0.95, color="#6B7280", linestyle="--", linewidth=1)
        fwer_ax.set_xticks(x, [f"{phi:g}" for phi in phi_values])
        fwer_ax.set_ylim(0, 1.02)
        fwer_ax.set_ylabel("Empirical familywise coverage")
        fwer_ax.set_title(f"n={n}: both contrasts cover")

        markers = {"percentile": "o", "hac": "s"}
        line_styles = {"SLSQP": "-", "Chebyshev": "--"}
        for unit in ("SLSQP", "Chebyshev"):
            for method in ("percentile", "hac"):
                entries = [next(item for item in summaries if item["n_configurations"] == n
                                and item["phi"] == phi and item["method"] == method
                                and item["unit"] == unit) for phi in phi_values]
                values = np.asarray([item["coverage"] for item in entries], dtype=float)
                low = np.asarray([item["wilson_95"][0] for item in entries], dtype=float)
                high = np.asarray([item["wilson_95"][1] for item in entries], dtype=float)
                marginal_ax.errorbar(phi_values, values, yerr=[values-low, high-values],
                                     color=colors[method], linestyle=line_styles[unit],
                                     marker=markers[method], capsize=2, linewidth=1.3,
                                     label=f"{unit} / {method}")
        marginal_ax.axhline(1.0-D06_ALPHA_PER_COMPARISON, color="#6B7280", linestyle=":", linewidth=1)
        marginal_ax.set_xticks(phi_values, [f"{phi:g}" for phi in phi_values])
        marginal_ax.set_ylim(0, 1.02)
        marginal_ax.set_ylabel("Marginal coverage")
        marginal_ax.set_title(f"n={n}: per-comparison intervals")
    axes[-1, 0].set_xlabel("AR(1) phi")
    axes[-1, 1].set_xlabel("AR(1) phi")
    fig.suptitle("Optimize D06 coverage: percentile vs influence-HAC across AR(1) grid", y=0.995)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=colors["percentile"], label="Percentile hull"),
        Patch(facecolor=colors["hac"], label="Influence-HAC hull"),
        Line2D([0], [0], color=colors["percentile"], marker="o", linestyle="-",
               label="SLSQP / percentile"),
        Line2D([0], [0], color=colors["hac"], marker="s", linestyle="-",
               label="SLSQP / HAC"),
        Line2D([0], [0], color=colors["percentile"], marker="o", linestyle="--",
               label="Chebyshev / percentile"),
        Line2D([0], [0], color=colors["hac"], marker="s", linestyle="--",
               label="Chebyshev / HAC"),
        Line2D([0], [0], color="#6B7280", linestyle="--", label="Familywise target 95%"),
        Line2D([0], [0], color="#6B7280", linestyle=":", label="Marginal target 97.5%"),
    ]
    fig.tight_layout(rect=(0, 0.10, 1, 0.98))
    fig.legend(legend_handles, [handle.get_label() for handle in legend_handles],
               frameon=False, fontsize=8, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.005))
    fig.savefig(out / "coverage_by_cell.png", dpi=180)
    plt.close(fig)


def build_report(output: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    out = Path(output)
    rows, source_context = _build_rows()
    summaries = _summary(rows)
    paired = _paired_comparison(rows)
    out.mkdir(parents=True, exist_ok=False)
    for filename, data in (("coverage_summary.csv", summaries),
                           ("paired_method_comparison.csv", paired),
                           ("replicates.csv", rows)):
        with (out / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    _plot(summaries, out)
    base_protocol = source_context["baseline_protocol"]
    paired_fwer = [item for item in paired if item["unit"] == "familywise"]
    fwer_summary = [item for item in summaries if item["unit"] == "familywise"]
    protocol = {
        "schema": "optimize-d06-hac-grid-coverage-diagnostic-v1",
        "evidence_class": "synthetic_stationary_gaussian_ar1_only",
        "status": "HAC_METHOD_CANDIDATE_NOT_PRODUCTION_CALIBRATION",
        "data_access": "reused_open_synthetic_panel_seeds_only",
        "physical_score": None,
        "grid": {
            "n_configuration_scenarios": base_protocol["n_configuration_scenarios"],
            "phi_scenarios": base_protocol["phi_scenarios"],
            "panels_per_cell": base_protocol["mc_replicates_per_phi"],
            "outer_draws_per_block_length_for_percentile_baseline": base_protocol["outer_draws_per_block_length"],
        },
        "paired_arm_order": list(ARMS),
        "time_points": [2, 3, 4, 5],
        "hac_block_lengths": list(D06_BLOCK_LENGTHS),
        "hac_bandwidths": list(HAC_BANDWIDTHS),
        "hac_interval": "paired ratio influence, Newey-West Bartlett long-run variance, normal one-sided 0.975 lower endpoint, zero-truncated positive-ratio hull",
        "family_size": 2,
        "paired_coverage_difference": {
            "resamples_per_cell_unit": PAIRED_BOOTSTRAP_DRAWS,
            "seed_rule": "2026112501 + sorted_cell_index*10 + unit_index",
        },
        "baseline_report": str(BASE_REPORT.relative_to(ROOT)),
        "baseline_hashes": {
            "protocol.json": _sha256(BASE_REPORT / "protocol.json"),
            "replicates.csv": _sha256(BASE_REPORT / "replicates.csv"),
            "artifact_hashes.json": _sha256(BASE_REPORT / "artifact_hashes.json"),
        },
        "source_hashes": {
            "d06_hac_grid_coverage_diagnostic.py": _sha256(Path(__file__)),
            "d06_hac_coverage_diagnostic.py": _sha256(Path(__file__).with_name("d06_hac_coverage_diagnostic.py")),
            "d06_bootstrap_coverage_diagnostic.py": _sha256(Path(__file__).with_name("d06_bootstrap_coverage_diagnostic.py")),
        },
        "hac_cell_denominators": source_context["cell_denominators"],
        "familywise_coverage": [
            {k: item[k] for k in ("n_configurations", "phi", "method", "planned", "valid", "failed", "abstained", "coverage", "wilson_95")}
            for item in fwer_summary
        ],
        "paired_familywise_differences": [
            {k: item[k] for k in ("n_configurations", "phi", "hac_minus_percentile_coverage", "paired_panel_bootstrap_95")}
            for item in paired_fwer
        ],
        "interpretation_limits": [
            "The percentile baseline uses 1,000 rather than the selected 10,000 outer draws; this is a method stress comparison, not final quantile Monte Carlo precision.",
            "The HAC lower bound relies on asymptotic linearization and weak-dependence conditions not established for a production gauge chain.",
            "The synthetic Gaussian AR(1) grid does not include heavy tails, nonstationarity, reference-fit uncertainty, or optimizer failures.",
            "No physical score, gauge-chain dependence estimate, or production calibration is inferred.",
        ],
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    familywise_table = "\n".join(
        f"| {r['n_configurations']} | {r['phi']:g} | {r['method']} | {r['coverage']:.3f} | "
        f"[{r['wilson_95'][0]:.3f}, {r['wilson_95'][1]:.3f}] | {r['planned']}/{r['valid']}/{r['failed']}/{r['abstained']} |"
        for r in fwer_summary
    )
    paired_table = "\n".join(
        f"| {r['n_configurations']} | {r['phi']:g} | {r['hac_minus_percentile_coverage']:+.3f} | "
        f"[{r['paired_panel_bootstrap_95'][0]:+.3f}, {r['paired_panel_bootstrap_95'][1]:+.3f}] |"
        for r in paired_fwer
    )
    readme = f"""# Optimize D06 influence-HAC coverage over the AR(1) grid

**Evidence class:** synthetic stationary Gaussian AR(1) only. This reuses the open `{base_protocol['mc_replicates_per_phi']}`-panel-per-cell baseline grid and regenerates the same q-panels from saved seeds. It reads no lattice data, fits, GPU/QUDA, or sealed split. The percentile comparator has `{base_protocol['outer_draws_per_block_length']}` outer draws per block length, below the selected 10,000-draw protocol.

## Text flow

```text
archived n={base_protocol['n_configuration_scenarios']} x phi={base_protocol['phi_scenarios']} x {base_protocol['mc_replicates_per_phi']} synthetic AR(1) panels/cell
  -> regenerate paired arms and verify exact generating risk ratios
  -> finite-L D04 risk + analytic configuration influence, L=2/5/10
  -> paired ratio influence -> HAC bandwidth 2/5/10/20/30 -> lower-bound hull
  -> compare with percentile-hull rows from the same panels
  -> cellwise marginal/F=2 coverage, Monte Carlo intervals, paired coverage changes
```

## F=2 familywise coverage

The familywise target is at least 95%; Wilson intervals quantify the Monte Carlo error in each cell. Full marginal coverage, every panel lower bound, and cell denominators are in `coverage_summary.csv` and `replicates.csv`.

| n | phi | method | coverage | Wilson 95% MC interval | P/V/F/A |
| ---: | ---: | --- | ---: | --- | --- |
{familywise_table}

## Paired familywise coverage change

Positive values favor HAC in this synthetic grid. Intervals bootstrap the paired MC panels within each cell and do not represent physical uncertainty.

| n | phi | HAC - percentile | paired panel bootstrap 95% |
| ---: | ---: | ---: | --- |
{paired_table}

## Interpretation boundary

The high-correlation `n=60, phi=0.8` targeted 10,000-draw comparison is reported separately in [`d06_hac_coverage_cpu_20261125_v4`](d06_hac_coverage_cpu_20261125_v4/README.md). This grid checks whether its apparent improvement persists over smaller `n` and lower correlations, but does not validate HAC as a D06 production interval. Newey-West relies on an asymptotic linear approximation and weak-dependence/moment conditions; neither the synthetic grid nor the finite-draw percentile comparator establishes those conditions for a gauge chain. Heavy tails, nonstationarity, fit/reference uncertainty and optimizer failures remain untested. Retain `UNCALIBRATED/NOT_READY`, `physical_score=null`, and the sealed-data boundary.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    shutil.copyfile(Path(__file__), out / "source_snapshot.py")
    payloads = sorted(path for path in out.iterdir() if path.is_file())
    manifest = {path.name: _sha256(path) for path in payloads}
    (out / "artifact_hashes.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"output": str(out), "familywise": fwer_summary, "physical_score": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = build_report(args.output)
    print(json.dumps({"output": result["output"], "familywise": result["familywise"],
                      "physical_score": result["physical_score"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
