"""Conditional pseudo-gap confidence boxes for finite-T Creutz means.

This CPU diagnostic uses only the bounded support of normalized real SU(3)
Wilson-loop traces. It reads no measurements and does not estimate a Markov
gap. The interval is a conditional alternative to bootstrap coverage claims.
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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/wilson_loop/c_lane_markov_support_bound_20261125_v1"
LOOP_LOWER = -1.0 / 3.0
LOOP_UPPER = 1.0
LANE_ALPHA = 0.025
FAMILY = 4
HALFWIDTH_TARGET = 0.02
N_GRID = (100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000)
GAP_GRID = (1.0, 0.5, 0.1, 0.01)
MEAN_FLOOR_GRID = (0.1, 0.25, 0.5, 1.0)


def _finite_real(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def markov_support_radius(*, blocks: int, gamma_ps_lower: float,
                          family: int = FAMILY,
                          alpha: float = LANE_ALPHA) -> dict[str, float]:
    """Two-sided simultaneous radius for fixed scores in [-1, 1].

    Inverts Paulin's corrected arXiv v5 Theorem 3.11, Eq. (3.25), with
    ``Var_pi(f) <= 1`` and ``|f-E_pi f| <= 2``. ``blocks`` is the number of
    states in the saved-configuration chain; ``gamma_ps_lower`` must be an
    externally justified lower bound for that exact chain.
    """
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 1:
        raise ValueError("blocks must be a positive integer")
    if isinstance(family, bool) or not isinstance(family, int) or family < 1:
        raise ValueError("family must be a positive integer")
    if not _finite_real(gamma_ps_lower) or not 0 < gamma_ps_lower <= 1:
        raise ValueError("gamma_ps_lower must be finite and in (0, 1]")
    if not _finite_real(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and in (0, 1)")

    gamma = float(gamma_ps_lower)
    log_tail = math.log(2.0 * family / float(alpha))
    centered_bound = 2.0
    variance_bound = 1.0
    linear = 20.0 * centered_bound * log_tail
    variance_term = 8.0 * log_tail * (blocks + 1.0 / gamma) * variance_bound
    root = (linear + math.sqrt(linear * linear + 4.0 * gamma * variance_term))
    sum_radius = root / (2.0 * gamma)
    radius = sum_radius / blocks
    if not math.isfinite(radius):
        raise ValueError("Markov support radius is not finite")
    return {
        "radius_normalized_score": radius,
        "radius_normalized_sum": sum_radius,
        "log_tail": log_tail,
        "gamma_ps_lower": gamma,
        "family": family,
        "alpha_family": float(alpha),
        "variance_bound": variance_bound,
        "centered_abs_bound": centered_bound,
    }


def loop_mean_interval(sample_mean: float, score_radius: float) -> tuple[float, float]:
    """Map the normalized-score radius to the real normalized-trace support."""
    if not _finite_real(sample_mean) or not LOOP_LOWER <= sample_mean <= LOOP_UPPER:
        raise ValueError("sample_mean must lie in the normalized SU(3) trace support")
    if not _finite_real(score_radius) or score_radius < 0:
        raise ValueError("score_radius must be finite and nonnegative")
    loop_radius = (LOOP_UPPER - LOOP_LOWER) * 0.5 * float(score_radius)
    return (max(LOOP_LOWER, float(sample_mean) - loop_radius),
            min(LOOP_UPPER, float(sample_mean) + loop_radius))


def creutz_box_interval(sample_means: object, score_radius: float) -> dict:
    """Propagate four simultaneous loop-mean intervals through the log ratio.

    Ordered means are ``W(3,6), W(2,5), W(3,5), W(2,6)`` and
    ``chi = log W(3,5) + log W(2,6) - log W(3,6) - log W(2,5)``.
    """
    means = np.asarray(sample_means, dtype=float)
    if means.shape != (4,) or not np.all(np.isfinite(means)):
        raise ValueError("sample_means must contain four finite loop means")
    bounds = [loop_mean_interval(float(mean), score_radius) for mean in means]
    lower_means = [bound[0] for bound in bounds]
    upper_means = [bound[1] for bound in bounds]
    if any(value <= 0 for value in lower_means):
        return {
            "status": "ABSTAIN_NONPOSITIVE_LOOP_LCB",
            "lower": None,
            "upper": None,
            "halfwidth": None,
            "loop_intervals": [list(bound) for bound in bounds],
        }

    point = (math.log(float(means[2])) + math.log(float(means[3]))
             - math.log(float(means[0])) - math.log(float(means[1])))
    lower = (math.log(lower_means[2]) + math.log(lower_means[3])
             - math.log(upper_means[0]) - math.log(upper_means[1]))
    upper = (math.log(upper_means[2]) + math.log(upper_means[3])
             - math.log(lower_means[0]) - math.log(lower_means[1]))
    return {
        "status": "FINITE_SIMULTANEOUS_BOX",
        "point": point,
        "lower": lower,
        "upper": upper,
        "halfwidth": 0.5 * (upper - lower),
        "loop_intervals": [list(bound) for bound in bounds],
    }


def worst_case_halfwidth(mean_floor: float, score_radius: float) -> float | None:
    """Bound propagated half-width if every observed loop mean is >= floor.

    With four independent marginal boxes and monotone log propagation, the
    worst box-width occurs when all four means equal the smallest allowed
    sample mean. This is a planning sensitivity, not a measured mean bound.
    """
    if not _finite_real(mean_floor) or not 0 < mean_floor <= LOOP_UPPER:
        raise ValueError("mean_floor must be in (0, 1]")
    if not _finite_real(score_radius) or score_radius < 0:
        raise ValueError("score_radius must be finite and nonnegative")
    delta = (LOOP_UPPER - LOOP_LOWER) * 0.5 * float(score_radius)
    if delta >= mean_floor:
        return None
    upper = min(LOOP_UPPER, mean_floor + delta)
    return 2.0 * math.log(upper / (mean_floor - delta))


def required_blocks(*, gamma_ps_lower: float, mean_floor: float,
                    halfwidth_target: float = HALFWIDTH_TARGET) -> int:
    """Smallest N whose support-only interval meets the requested width."""
    if not _finite_real(halfwidth_target) or halfwidth_target <= 0:
        raise ValueError("halfwidth_target must be finite and positive")

    def passes(n: int) -> bool:
        radius = markov_support_radius(
            blocks=n, gamma_ps_lower=gamma_ps_lower,
        )["radius_normalized_score"]
        half = worst_case_halfwidth(mean_floor, radius)
        return half is not None and half <= halfwidth_target

    low, high = 1, 1
    while not passes(high):
        high *= 2
        if high > 10**14:
            raise ValueError("required sample size exceeds supported planning range")
    while low < high:
        mid = (low + high) // 2
        if passes(mid):
            high = mid
        else:
            low = mid + 1
    return low


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty sensitivity table")
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


def _make_plot(rows: list[dict], required: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), constrained_layout=True)
    means = list(MEAN_FLOOR_GRID)
    gaps = list(GAP_GRID)
    values = np.asarray([
        [math.log10(next(row["required_blocks"] for row in required
                        if row["gamma_ps_lower"] == gap
                        and row["mean_floor"] == mean)) for mean in means]
        for gap in gaps
    ])
    image = axes[0].imshow(values, aspect="auto", cmap="viridis")
    axes[0].set_xticks(range(len(means)), [f"{mean:g}" for mean in means])
    axes[0].set_yticks(range(len(gaps)), [f"{gap:g}" for gap in gaps])
    axes[0].set_xlabel("assumed minimum observed loop mean")
    axes[0].set_ylabel(r"assumed lower bound $\gamma_{ps}$")
    axes[0].set_title("log10 sample size needed for half-width 0.02")
    for iy, gap in enumerate(gaps):
        for ix, mean in enumerate(means):
            count = next(row["required_blocks"] for row in required
                         if row["gamma_ps_lower"] == gap
                         and row["mean_floor"] == mean)
            axes[0].text(ix, iy, f"{count:.2g}", ha="center", va="center",
                         color="white" if values[iy, ix] > 7.0 else "black",
                         fontsize=8)
    figure.colorbar(image, ax=axes[0], label=r"$\log_{10} N$")

    n_values = np.geomspace(100, 1e9, 180).astype(int)
    mean_floor = 0.5
    for gap in (1.0, 0.1, 0.01):
        curve = []
        for n in n_values:
            radius = markov_support_radius(
                blocks=int(n), gamma_ps_lower=gap,
            )["radius_normalized_score"]
            curve.append(worst_case_halfwidth(mean_floor, radius))
        finite_x = [n for n, half in zip(n_values, curve) if half is not None]
        finite_y = [half for half in curve if half is not None]
        axes[1].plot(finite_x, finite_y, label=rf"$\gamma_{{ps}}\geq {gap:g}$")
    axes[1].axhline(HALFWIDTH_TARGET, color="black", linestyle="--",
                    label="selected 0.02 target")
    axes[1].axvline(100, color="#6b7280", linestyle=":",
                    label="n=100 scale")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("saved configurations N")
    axes[1].set_ylabel("worst-case Creutz interval half-width")
    axes[1].set_title("Assumed minimum sample loop mean = 0.5")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("Conditional Markov support bound: four bounded loop means")
    figure.savefig(output, dpi=170)
    plt.close(figure)


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    rows = []
    for n in N_GRID:
        for gap in GAP_GRID:
            details = markov_support_radius(blocks=n, gamma_ps_lower=gap)
            for mean_floor in MEAN_FLOOR_GRID:
                delta = (LOOP_UPPER - LOOP_LOWER) * 0.5 * details["radius_normalized_score"]
                halfwidth = worst_case_halfwidth(mean_floor, details["radius_normalized_score"])
                if halfwidth is None:
                    status, reason = "ABSTAIN", "support interval reaches zero; logarithm is not bounded"
                elif halfwidth > HALFWIDTH_TARGET:
                    status, reason = "ABSTAIN", "simultaneous interval misses the 0.02 precision target"
                else:
                    status, reason = "CONDITIONAL_VALID", "conditional support and pseudo-gap assumptions meet precision"
                rows.append({
                    "planned_n": n,
                    "gamma_ps_lower_assumed": gap,
                    "mean_floor_assumed": mean_floor,
                    "score_radius": details["radius_normalized_score"],
                    "loop_mean_radius": delta,
                    "worst_case_halfwidth": "" if halfwidth is None else halfwidth,
                    "status": status,
                    "reason": reason,
                })

    required = []
    for gap in GAP_GRID:
        for mean_floor in MEAN_FLOOR_GRID:
            required.append({
                "gamma_ps_lower": gap,
                "mean_floor": mean_floor,
                "halfwidth_target": HALFWIDTH_TARGET,
                "required_blocks": required_blocks(
                    gamma_ps_lower=gap, mean_floor=mean_floor,
                ),
            })

    denominators = {
        "unit": "conditional analytic planning scenario; no datasets are generated",
        "planned": len(rows),
        "valid": sum(row["status"] == "CONDITIONAL_VALID" for row in rows),
        "failed": 0,
        "abstained": sum(row["status"] == "ABSTAIN" for row in rows),
    }
    reason_counts = {
        reason: sum(row["reason"] == reason for row in rows)
        for reason in sorted({row["reason"] for row in rows})
    }
    best_case_n = required_blocks(gamma_ps_lower=1.0, mean_floor=1.0)
    protocol = {
        "status": "CONDITIONAL_ANALYTIC_SENSITIVITY_ONLY",
        "estimand": "finite-T Creutz chi_stout(R=2,T=5), separate from ground-state force",
        "loop_order": ["W(3,6)", "W(2,5)", "W(3,5)", "W(2,6)"],
        "support_assumption": "each saved-configuration observable is ReTr(U_loop)/3 in [-1/3,1]",
        "normalization_is_a_production_receipt": False,
        "bound": "Paulin corrected arXiv v5 Theorem 3.11 / Eq. (3.25), fixed-score Bernstein with V_f<=1 and C=2",
        "bound_url": "https://arxiv.org/html/1212.2015v5",
        "family": FAMILY,
        "alpha_lane": LANE_ALPHA,
        "multiple_testing": "Bonferroni across four two-sided marginal mean intervals; interval alpha=0.025",
        "sample_chain": "saved-configuration chain; gamma_ps receipt must apply to this exact transition, not inferred from trajectory spacing",
        "gap_grid_is_hypothetical": True,
        "mean_floor_grid_is_hypothetical": True,
        "n_grid": list(N_GRID),
        "gamma_ps_grid": list(GAP_GRID),
        "mean_floor_grid": list(MEAN_FLOOR_GRID),
        "halfwidth_target": HALFWIDTH_TARGET,
        "initial_law": "stationary, the optimistic case; burn-in TV correction would further widen intervals and spend alpha",
        "denominators": denominators,
        "abstention_reason_counts": reason_counts,
        "best_possible_gap_and_mean_floor_required_n": best_case_n,
        "physical_data_access": "none",
        "sealed_inputs_accessed": False,
        "gpu_quda": False,
        "physical_score": None,
        "certified": False,
    }
    source_paths = [Path(__file__), Path(__file__).with_name("lane_contract_v2.json")]
    protocol["source_hashes"] = {path.name: _sha256(path) for path in source_paths}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    _write_csv(output / "sensitivity_grid.csv", rows)
    _write_csv(output / "required_sample_size.csv", required)
    _make_plot(rows, required, output / "markov_bound_sensitivity.png")
    readme = f"""# Wilson C-lane conditional Markov support-bound sensitivity

**Method-only result.** No Wilson measurements, development split, sealed input, GPU or QUDA data were read. `physical_score=null`; the result is not a production interval or coverage certificate.

## Candidate route

For the selected finite-$T$ Creutz value, ordered loop means are `W(3,6), W(2,5), W(3,5), W(2,6)` and

```text
chi = log W(3,5) + log W(2,6) - log W(3,6) - log W(2,5).
```

Conditional on each per-configuration measurement being the real normalized SU(3) trace `ReTr(U_loop)/3`, each loop value is in `[-1/3,1]`. Map it to `f=(W-1/3)/(2/3) in [-1,1]`. For the fixed-score saved-configuration chain, Paulin's corrected arXiv v5 Theorem 3.11, Eq. (3.25), with `Var_pi(f)<=1`, centered bound `C=2`, family `F=4`, and lane alpha `0.025`, gives

```text
L = log(2 F / alpha)
r_N = [40 L + sqrt((40 L)^2 + 32 gamma_ps L (N + 1/gamma_ps))]
      / (2 gamma_ps N)
delta_W = (2/3) r_N
```

Four simultaneous two-sided mean intervals receive Bonferroni coverage `1-alpha`. Their rectangular box is mapped to a Creutz interval by monotonicity. Any nonpositive loop-mean lower endpoint produces `ABSTAIN`; no log is evaluated outside the positive domain. The rectangle is conservative because it ignores cross-loop covariance. The theorem is conditional on a valid fixed, homogeneous Markov chain and a justified lower bound for the pseudo-spectral gap of the *saved-configuration chain*. Trajectory spacing, an empirical autocorrelation time, or bootstrap draws do not supply that bound. The table uses stationary initialization, the most favorable case; a burn-in/TV correction would consume alpha and widen the interval.

Primary source: [Paulin, corrected arXiv v5](https://arxiv.org/html/1212.2015v5). The paper explicitly says the proof in earlier versions, including its published version, was incomplete and gives the correction in v5.

## Numerical sensitivity, not data

The 96 planned scenarios cross `N={list(N_GRID)}`, hypothetical `gamma_ps` lower bounds `{list(GAP_GRID)}`, and assumed *observed sample-mean floors* `{list(MEAN_FLOOR_GRID)}`. The denominators are planned/valid/failed/abstained=`{denominators['planned']}/{denominators['valid']}/{denominators['failed']}/{denominators['abstained']}`. Here `valid` means only that a hypothetical support interval remains positive and meets the 0.02 half-width target. It is not an observed sample or certificate. Abstention counts by reason: `{reason_counts}`.

![Conditional pseudo-gap interval sensitivity](markov_bound_sensitivity.png)

Even with the largest possible gap lower bound `gamma_ps=1` and the mathematically best possible observed mean floor `1`, the support-only route needs **{best_case_n:,} saved configurations** to meet half-width 0.02. For smaller means, weaker gaps, or nonstationary initialization it needs more. Thus this method is a robust conditional reference and a useful stopping rule, but it does not make the current 100-configuration development preflight confirmatory and is unlikely to replace a calibrated, dependence-aware interval at practical sizes without tighter reviewed variance bounds.

Text flow: `saved configuration -> four normalized real loop traces -> four simultaneous Markov mean boxes -> positive-endpoint check -> monotone log propagation -> precision gate or ABSTAIN`. The current schema still lacks the production normalization receipt, saved-chain pseudo-gap receipt, stationary/burn-in receipt, and independent reviewer sign-off; lane status remains `NOT_READY`.
"""
    (output / "README.md").write_text(readme)
    shutil.copy2(Path(__file__), output / "source_snapshot.py")
    files = sorted(path for path in output.iterdir() if path.is_file())
    hashes = {path.name: _sha256(path) for path in files}
    (output / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    return {"output": str(output), "denominators": denominators,
            "best_case_required_n": best_case_n, "physical_score": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
