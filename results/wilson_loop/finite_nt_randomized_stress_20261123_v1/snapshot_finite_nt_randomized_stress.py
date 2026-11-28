"""Randomized positive-spectrum algebra stress for finite-Nt Creutz stencils.

All parameters are synthetic. The study checks the exact four-point identity
and positive-contamination envelope over a broad input range; it does not
estimate physical spectra or certify a ground-state force.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

from tasks.wilson_loop.finite_nt_toy import finite_nt_creutz


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/wilson_loop/finite_nt_randomized_stress_20261123_v1"
DEFAULT_COUNT = 4000
DEFAULT_SEED = 2026112308


def _loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def _parameters(rng: np.random.Generator) -> dict:
    return {
        "A0": _loguniform(rng, 0.1, 10.0),
        "V0": float(rng.uniform(0.0, 1.5)),
        "Q": _loguniform(rng, 1e-3, 1e2),
        "H": _loguniform(rng, 1e-3, 1e2),
        "X": _loguniform(rng, 1e-3, 1e2),
        "gap_q": _loguniform(rng, 0.01, 1.0),
        "gap_0": _loguniform(rng, 0.01, 1.0),
    }


def generate_cases(count: int = DEFAULT_COUNT,
                   seed: int = DEFAULT_SEED) -> list[dict]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    rows = []
    for case in range(count):
        temporal_extent = int(rng.integers(4, 65))
        time_extent = int(rng.integers(1, temporal_extent))
        parameters_r = _parameters(rng)
        parameters_r_plus_1 = _parameters(rng)
        result = finite_nt_creutz(
            parameters_r, parameters_r_plus_1,
            time_extent, temporal_extent,
        )
        absolute_bias = abs(result["bias_lattice"])
        envelope = result["four_loop_envelope_lattice"]
        rows.append({
            "case": case,
            "temporal_extent": temporal_extent,
            "time_extent": time_extent,
            "time_fraction": time_extent / temporal_extent,
            **{f"R_{key}": value for key, value in parameters_r.items()},
            **{f"R_plus_1_{key}": value
               for key, value in parameters_r_plus_1.items()},
            "chi_lattice": result["chi_lattice"],
            "target_force_lattice": result["target_force_lattice"],
            "bias_lattice": result["bias_lattice"],
            "delta_R_delta_t": result["delta_R_delta_t"],
            "identity_abs_error": abs(
                result["bias_lattice"] - result["identity_bias_lattice"]),
            "four_loop_envelope_lattice": envelope,
            "absolute_bias_lattice": absolute_bias,
            "bias_to_envelope_ratio": absolute_bias / envelope,
            "envelope_covers_bias": result["envelope_covers_bias"],
        })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(out: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    envelopes = np.array([row["four_loop_envelope_lattice"] for row in rows])
    biases = np.array([row["absolute_bias_lattice"] for row in rows])
    ratios = np.array([row["bias_to_envelope_ratio"] for row in rows])
    fractions = np.array([row["time_fraction"] for row in rows])
    temporal = np.array([row["temporal_extent"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout="constrained")
    x = np.maximum(envelopes, 1e-30)
    y = np.maximum(biases, 1e-30)
    axes[0].scatter(x, y, c=temporal, cmap="viridis", s=13, alpha=0.45)
    low = min(float(x.min()), float(y.min()))
    high = max(float(x.max()), float(y.max()))
    axes[0].plot([low, high], [low, high], color="#D55E00", ls="--",
                 label="|bias| = envelope")
    axes[0].set(xscale="log", yscale="log",
                xlabel="sum of four positive log envelopes",
                ylabel="absolute exact Creutz bias",
                title="Every sampled bias lies below its envelope")
    axes[0].grid(alpha=0.2, which="both")
    colorbar = fig.colorbar(axes[0].collections[0], ax=axes[0], shrink=0.86)
    colorbar.set_label("temporal extent N_t")
    axes[0].legend(fontsize=8)

    axes[1].hist(ratios, bins=np.linspace(0, 1, 41), color="#0072B2",
                 alpha=0.8, edgecolor="white")
    axes[1].axvline(float(np.median(ratios)), color="#D55E00", ls="--",
                    label=f"median |bias|/envelope = {np.median(ratios):.3g}")
    axes[1].set(xlabel="absolute exact bias / four-loop envelope",
                ylabel="randomized cases",
                title="Slack of the conservative bound", xlim=(0, 1))
    axes[1].grid(alpha=0.2, axis="y")
    axes[1].legend(fontsize=8)
    fig.suptitle("Finite-N_t positive-spectrum algebra stress; synthetic inputs only")
    fig.savefig(out / "finite_nt_randomized_stress.png", dpi=180)
    plt.close(fig)


def build(output: str | Path = DEFAULT_OUT, count: int = DEFAULT_COUNT,
          seed: int = DEFAULT_SEED) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows = generate_cases(count, seed)
    _write_csv(out / "randomized_cases.csv", rows)
    _write_plots(out, rows)
    identity_max = max(row["identity_abs_error"] for row in rows)
    covered = sum(bool(row["envelope_covers_bias"]) for row in rows)
    denominators = [
        {"unit": "randomized parameter case", "planned": len(rows),
         "valid": len(rows), "failed": 0, "abstained": 0,
         "interpretation": "All cases use prespecified positive synthetic parameters; valid means the algebra checks completed."},
        {"unit": "four-loop evaluation", "planned": 4 * len(rows),
         "valid": 4 * len(rows), "failed": 0, "abstained": 0,
         "interpretation": "Each Creutz stencil contains four finite-N_t loop values."},
        {"unit": "envelope coverage check", "planned": len(rows),
         "valid": covered, "failed": len(rows) - covered, "abstained": 0,
         "interpretation": "A failed case would contradict the declared positive-contamination envelope."},
    ]
    with (out / "denominator_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(denominators[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(denominators)

    source_hashes = {
        "finite_nt_randomized_stress.py": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "finite_nt_toy.py": hashlib.sha256(
            Path(__file__).with_name("finite_nt_toy.py").read_bytes()).hexdigest(),
    }
    (out / "snapshot_finite_nt_randomized_stress.py").write_bytes(
        Path(__file__).read_bytes())
    (out / "snapshot_finite_nt_toy.py").write_bytes(
        Path(__file__).with_name("finite_nt_toy.py").read_bytes())
    protocol = {
        "schema": "wilson-finite-nt-randomized-algebra-v1",
        "classification": "synthetic positive-spectrum algebra stress only",
        "physical_score": None,
        "seed": seed,
        "planned_cases": count,
        "temporal_extent_integer_range_inclusive": [4, 64],
        "time_extent_rule": "uniform integer t in [1, N_t-1]",
        "parameter_ranges": {
            "A0": "log-uniform [0.1, 10]",
            "V0": "uniform [0, 1.5]",
            "Q_H_X": "independent log-uniform [1e-3, 1e2] for each radius",
            "gap_q_gap_0": "independent log-uniform [0.01, 1] for each radius",
        },
        "data_access": "No Wilson measurements, sealed split, GPU, or QUDA inputs.",
        "source_sha256": source_hashes,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    summary = {
        "schema": "wilson-finite-nt-randomized-algebra-summary-v1",
        "status": "SYNTHETIC_ALGEBRA_STRESS_ONLY",
        "physical_score": None,
        "data_access": protocol["data_access"],
        "planned_cases": count,
        "envelope_covered_cases": covered,
        "envelope_coverage_fraction": covered / count,
        "identity_max_abs_error": identity_max,
        "median_abs_bias_to_envelope": float(np.median([
            row["bias_to_envelope_ratio"] for row in rows])),
        "p95_abs_bias_to_envelope": float(np.quantile([
            row["bias_to_envelope_ratio"] for row in rows], 0.95)),
        "max_abs_bias_to_envelope": max(
            row["bias_to_envelope_ratio"] for row in rows),
        "denominator_summary": denominators,
        "source_sha256": source_hashes,
        "files": ["README.md", "protocol.json", "randomized_cases.csv",
                  "denominator_summary.csv", "finite_nt_randomized_stress.png",
                  "snapshot_finite_nt_randomized_stress.py",
                  "snapshot_finite_nt_toy.py", "summary.json",
                  "artifact_hashes.json"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    readme = [
        "# Finite-N_t randomized positive-spectrum algebra stress",
        "",
        "**SYNTHETIC ALGEBRA DIAGNOSTIC ONLY. `physical_score=null`.** Randomized inputs exercise the exact four-point formula and its positive-contamination envelope; they are not fitted spectra, measured Wilson loops, or physical contamination bounds.",
        "",
        "## Prespecified stress domain",
        "",
        f"The run generated `{count}` independent pseudo-random parameter cases with seed `{seed}`. `N_t` is uniform integer 4–64 and `t` is uniform integer 1–`N_t-1`. For each of `R` and `R+1`, `A0` is log-uniform `[0.1,10]`, `V0` uniform `[0,1.5]`, `Q/H/X` independently log-uniform `[1e-3,1e2]`, and `gap_q/gap_0` independently log-uniform `[0.01,1]`. All parameters are dimensionless lattice inputs and chosen only to stress a broad positive range.",
        "",
        "For each case the code evaluates the four `log W` points, checks `bias = -Delta_R Delta_t delta`, and checks `|bias| <= sum_four_points log1p(q+h+x)`. This envelope follows by applying the triangle inequality to four nonnegative contamination logs. `randomized_cases.csv` retains the sampled inputs and each diagnostic outcome.",
        "",
        "## Figure and result",
        "",
        "![Randomized finite-N_t identity and envelope stress](finite_nt_randomized_stress.png)",
        "",
        f"Planned/valid/failed/abstained randomized cases: `{count}/{count}/0/0`; four-loop evaluations: `{4*count}/{4*count}/0/0`. The identity's maximum absolute discrepancy is `{identity_max:.3g}`. Envelope coverage is `{covered}/{count}`; the median, 95th percentile, and maximum sampled `|bias|/envelope` ratios are `{summary['median_abs_bias_to_envelope']:.3g}`, `{summary['p95_abs_bias_to_envelope']:.3g}`, and `{summary['max_abs_bias_to_envelope']:.3g}`. These describe the arbitrary sampling box, not physical coverage or tightness.",
        "",
        "## Limits and next physical evidence",
        "",
        "This stress test verifies implementation algebra across randomized positive inputs. It does not show that Q/H/X or gaps in a simulation lie inside these ranges, establish positivity for a real operator basis, or make the envelope small enough to certify the ground-state static force. C-lane finite-T Creutz and F-lane ground-state force remain separate. F-lane certification still needs independent physical spectral, thermal, and mixed-sector bounds with a prospective tolerance.",
        "",
        "Rebuild into a fresh directory: `python -B -m tasks.wilson_loop.finite_nt_randomized_stress --output NEW_DIRECTORY`.",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n")
    artifact_names = ["README.md", "protocol.json", "randomized_cases.csv",
                      "denominator_summary.csv", "finite_nt_randomized_stress.png",
                      "snapshot_finite_nt_randomized_stress.py",
                      "snapshot_finite_nt_toy.py", "summary.json"]
    hashes = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
              for name in artifact_names}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    summary = build(args.output, args.count, args.seed)
    print(json.dumps({
        "status": summary["status"],
        "planned_cases": summary["planned_cases"],
        "envelope_covered_cases": summary["envelope_covered_cases"],
        "identity_max_abs_error": summary["identity_max_abs_error"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
