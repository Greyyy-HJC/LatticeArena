"""Conditional gap/overlap envelope screen for finite-time Wilson energies.

This deterministic synthetic screen connects the existing positive-spectrum
q/gap bound to effective-energy and force intervals. It does not calibrate the
joint spectral-fit estimator or establish production Q/gap receipts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
DEFAULT_OUT = TASK_DIR / "f_lane_gap_overlap_force_screen_20261128_v1"
KERNELS = ("spatial_stout_rho0.10_n5", "identity")
R_VALUES = (2, 3)
TIME_STARTS = (3, 4, 5, 6)
CONTAMINATION_BUDGET_TIME = 5
REFERENCE_FORCE_LOWER_TOY = 0.20
MODEL_TRUNCATION_FRACTION = 0.01
RADIUS_STEP = 1.0

# Positive one-excited-state fixture. The common E0(R) is shared by both
# operators, as required by a common transfer matrix. The negative E0 offset
# deliberately avoids relying on an unverified nonnegative static-energy
# subtraction convention; only the force difference is used.
TOY_E0 = np.asarray([-0.25, -0.05], dtype=float)
TOY_OVERLAP_RATIO = np.asarray([[0.20, 0.10], [0.08, 0.16]], dtype=float)
TOY_GAP = np.asarray([[0.35, 0.45], [0.25, 0.40]], dtype=float)
Q_GRID = np.geomspace(1.0e-3, 1.0e2, 61)
GAP_GRID = np.linspace(0.01, 1.0, 61)


def _nonnegative_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite nonnegative number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite nonnegative number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return result


def _finite_real_array(values: object, label: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{label} must use a real integer or floating dtype")
    try:
        result = raw.astype(float, copy=False)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and representable as float") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result


def excited_energy_correction_bound(q_max: float, gap_min: float,
                                     time_extent: int) -> float:
    """Return log(1 + Q_max exp(-Delta_min T)) for a positive spectrum."""
    q = _nonnegative_finite(q_max, "q_max")
    gap = _nonnegative_finite(gap_min, "gap_min")
    if type(time_extent) is not int or time_extent < 1:
        raise ValueError("time_extent must be an integer >= 1")
    return math.log1p(q * math.exp(-gap * time_extent))


def effective_energy_interval(
    effective_energies: np.ndarray,
    q_max_by_corner: np.ndarray,
    gap_min_by_corner: np.ndarray,
    time_starts: tuple[int, ...] = TIME_STARTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect per-operator E0 intervals under a shared spectrum per R.

    Inputs are indexed [kernel, R, time]. For each observation,
    E_eff(T)-log(1+Q_max exp(-Delta_min*T)) <= E0 <= E_eff(T).
    Intersections across kernels use the common-Hamiltonian premise.
    """
    energies = _finite_real_array(effective_energies, "effective_energies")
    q_values = _finite_real_array(q_max_by_corner, "q_max_by_corner")
    gaps = _finite_real_array(gap_min_by_corner, "gap_min_by_corner")
    expected = (len(KERNELS), len(R_VALUES), len(time_starts))
    if energies.shape != expected:
        raise ValueError(f"effective_energies must be finite with shape {expected}")
    corner_shape = expected[:2]
    if (q_values.shape != corner_shape or gaps.shape != corner_shape
            or np.any(q_values < 0.0) or np.any(gaps < 0.0)):
        raise ValueError(f"q/gap bounds must be finite nonnegative arrays of shape {corner_shape}")
    if (not time_starts or any(type(t) is not int or t < 1 for t in time_starts)
            or tuple(sorted(set(time_starts))) != time_starts):
        raise ValueError("time_starts must be unique increasing positive integers")

    lower_by_radius = []
    upper_by_radius = []
    for r_index in range(len(R_VALUES)):
        lows = []
        highs = []
        for k_index in range(len(KERNELS)):
            for t_index, time_extent in enumerate(time_starts):
                correction = excited_energy_correction_bound(
                    float(q_values[k_index, r_index]),
                    float(gaps[k_index, r_index]),
                    time_extent,
                )
                observed = float(energies[k_index, r_index, t_index])
                lows.append(observed - correction)
                highs.append(observed)
        lower = max(lows)
        upper = min(highs)
        if lower > upper + 1.0e-12:
            raise ValueError(f"conditional E0 intervals do not intersect at R={R_VALUES[r_index]}")
        lower_by_radius.append(lower)
        upper_by_radius.append(upper)
    return np.asarray(lower_by_radius), np.asarray(upper_by_radius)


def force_interval(energy_lower: np.ndarray, energy_upper: np.ndarray,
                   radius_step: float = RADIUS_STEP) -> tuple[float, float]:
    """Propagate independent endpoint intervals to (E0(R3)-E0(R2))/DeltaR."""
    lower = _finite_real_array(energy_lower, "energy_lower")
    upper = _finite_real_array(energy_upper, "energy_upper")
    if (lower.shape != (len(R_VALUES),) or upper.shape != lower.shape
            or np.any(lower > upper)):
        raise ValueError("energy endpoints must be finite ordered bounds for each radius")
    if (isinstance(radius_step, bool) or not isinstance(radius_step, (int, float))
            or not math.isfinite(radius_step) or radius_step <= 0):
        raise ValueError("radius_step must be positive and finite")
    return (
        float((lower[1] - upper[0]) / radius_step),
        float((upper[1] - lower[0]) / radius_step),
    )


def candidate_reference_effective_bias_bound(
    q_max_by_corner: np.ndarray,
    gap_min_by_corner: np.ndarray,
    time_extent: int = CONTAMINATION_BUDGET_TIME,
) -> float:
    """Bound the absolute four-corner bias of a same-T kernel-force contrast.

    This is for the effective-energy proxy only, not for a multi-state fit
    estimator. The common exact ground-state force cancels between kernels.
    """
    q_values = _finite_real_array(q_max_by_corner, "q_max_by_corner")
    gaps = _finite_real_array(gap_min_by_corner, "gap_min_by_corner")
    expected = (len(KERNELS), len(R_VALUES))
    if (q_values.shape != expected or gaps.shape != expected
            or np.any(q_values < 0.0) or np.any(gaps < 0.0)):
        raise ValueError(f"q/gap bounds must be finite nonnegative arrays of shape {expected}")
    return float(math.fsum(
        excited_energy_correction_bound(float(q_values[k, r]), float(gaps[k, r]), time_extent)
        for k in range(len(KERNELS)) for r in range(len(R_VALUES))
    ))


def _toy_effective_energies() -> np.ndarray:
    result = np.empty((len(KERNELS), len(R_VALUES), len(TIME_STARTS)), dtype=float)
    for k_index in range(len(KERNELS)):
        for r_index in range(len(R_VALUES)):
            q = float(TOY_OVERLAP_RATIO[k_index, r_index])
            gap = float(TOY_GAP[k_index, r_index])
            for t_index, t in enumerate(TIME_STARTS):
                now = math.log1p(q * math.exp(-gap * t))
                following = math.log1p(q * math.exp(-gap * (t + 1)))
                result[k_index, r_index, t_index] = TOY_E0[r_index] + now - following
    return result


def _toy_metrics(q_max: float, gap_min: float) -> dict[str, Any]:
    q_bounds = np.full((len(KERNELS), len(R_VALUES)), q_max, dtype=float)
    gap_bounds = np.full_like(q_bounds, gap_min)
    eff = _toy_effective_energies()
    e_lower, e_upper = effective_energy_interval(eff, q_bounds, gap_bounds)
    f_lower, f_upper = force_interval(e_lower, e_upper)
    t5_index = TIME_STARTS.index(CONTAMINATION_BUDGET_TIME)
    force_by_kernel = eff[:, 1, t5_index] - eff[:, 0, t5_index]
    observed_difference = float(force_by_kernel[0] - force_by_kernel[1])
    bound = candidate_reference_effective_bias_bound(q_bounds, gap_bounds)
    budget = MODEL_TRUNCATION_FRACTION * REFERENCE_FORCE_LOWER_TOY
    return {
        "energy_lower": e_lower,
        "energy_upper": e_upper,
        "force_interval": (f_lower, f_upper),
        "absolute_ground_force_truth": float((TOY_E0[1] - TOY_E0[0]) / RADIUS_STEP),
        "candidate_minus_reference_ground_force_truth": 0.0,
        "finite_time_candidate_minus_reference_force": observed_difference,
        "finite_time_candidate_minus_reference_bias_abs": abs(observed_difference),
        "four_corner_bias_bound": bound,
        "one_percent_toy_budget": budget,
        "budget_ratio": bound / budget,
        "observed_proxy_bias_to_budget_ratio": abs(observed_difference) / budget,
    }


def _grid_rows() -> list[dict[str, Any]]:
    rows = []
    true_q_max = float(np.max(TOY_OVERLAP_RATIO))
    true_gap_min = float(np.min(TOY_GAP))
    for gap_min in GAP_GRID:
        for q_max in Q_GRID:
            premises_hold = bool(q_max >= true_q_max and gap_min <= true_gap_min)
            q_bounds = np.full((len(KERNELS), len(R_VALUES)), q_max, dtype=float)
            gap_bounds = np.full_like(q_bounds, gap_min)
            budget_bound = candidate_reference_effective_bias_bound(q_bounds, gap_bounds)
            budget = MODEL_TRUNCATION_FRACTION * REFERENCE_FORCE_LOWER_TOY
            try:
                metrics = _toy_metrics(float(q_max), float(gap_min))
                f_lo, f_hi = metrics["force_interval"]
                interval_status = "VALID"
                contains_truth = bool(
                    f_lo <= metrics["absolute_ground_force_truth"] <= f_hi
                )
            except ValueError:
                f_lo, f_hi = None, None
                interval_status = "ABSTAIN"
                contains_truth = False
            rows.append({
                "Q_max_assumed": float(q_max),
                "Delta_min_assumed": float(gap_min),
                "toy_premises_hold": premises_hold,
                "conditional_interval_status": interval_status,
                "absolute_force_lower": f_lo,
                "absolute_force_upper": f_hi,
                "absolute_force_interval_width": None if f_lo is None else f_hi - f_lo,
                "candidate_reference_effective_bias_bound": budget_bound,
                "one_percent_toy_reference_budget": budget,
                "budget_ratio_bound_over_budget": budget_bound / budget,
                "passes_one_percent_toy_budget": budget_bound <= budget,
                "conditional_interval_contains_toy_force": contains_truth,
            })
    return rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plot(rows: list[dict[str, Any]], metrics: dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), layout="constrained")
    ax = axes[0]
    f_lo, f_hi = metrics["force_interval"]
    truth = metrics["absolute_ground_force_truth"]
    ax.hlines(1.0, f_lo, f_hi, color="#286e5d", linewidth=5, label="conditional interval")
    ax.scatter([truth], [1.0], color="#b74a3a", s=55, zorder=3, label="toy truth")
    ax.set(yticks=[1.0], yticklabels=["Qmax=0.25, Delta_min=0.20"],
           xlabel="absolute ground force (lattice units)",
           title="Q/gap-conditioned force interval")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    q_values = np.asarray([row["Q_max_assumed"] for row in rows]).reshape(len(GAP_GRID), len(Q_GRID))
    gap_values = np.asarray([row["Delta_min_assumed"] for row in rows]).reshape(len(GAP_GRID), len(Q_GRID))
    ratio = np.asarray([row["budget_ratio_bound_over_budget"] for row in rows]).reshape(len(GAP_GRID), len(Q_GRID))
    premises = np.asarray([row["toy_premises_hold"] for row in rows]).reshape(len(GAP_GRID), len(Q_GRID))
    masked = np.ma.masked_where(~premises, np.log10(ratio))
    image = ax.pcolormesh(Q_GRID, GAP_GRID, masked, shading="auto", cmap="viridis")
    ax.set_xscale("log")
    ax.contour(q_values, gap_values, premises.astype(float), levels=[0.5], colors=["white"], linewidths=1.2)
    masked_log_ratio = np.ma.masked_where(~premises, np.log10(ratio))
    if np.min(ratio[premises]) <= 1.0 <= np.max(ratio[premises]):
        ax.contour(Q_GRID, GAP_GRID, masked_log_ratio, levels=[0.0], colors=["#d43f3a"], linewidths=1.4)
    ax.set(xlabel="assumed total Q_max", ylabel="assumed minimum gap Delta_min",
           title="Four-corner bias bound / 1% toy budget")
    fig.colorbar(image, ax=ax, label="log10(bound / budget)")
    ax.text(0.03, 0.97, "white boundary: premises hold\n(red contour: budget ratio = 1, if present)",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"})

    ax = axes[2]
    ax.axis("off")
    entries = [
        ("shared-spectrum ground-force contrast", "0.000000"),
        ("finite-T effective-energy contrast", f"{metrics['finite_time_candidate_minus_reference_force']:.6f}"),
        ("observed proxy bias / budget", f"{metrics['observed_proxy_bias_to_budget_ratio']:.2f}x"),
        ("four-corner conditional bound", f"{metrics['four_corner_bias_bound']:.6f}"),
        ("1% of exact toy reference force", f"{metrics['one_percent_toy_budget']:.6f}"),
        ("bound / budget", f"{metrics['budget_ratio']:.1f}x"),
    ]
    y = 0.90
    for label, value in entries:
        ax.text(0.02, y, label, fontsize=9, va="top")
        ax.text(0.98, y, value, fontsize=9, va="top", ha="right", family="monospace")
        y -= 0.13
    ax.text(0.02, 0.06,
            "Synthetic positive one-excited-state fixture.\nThe bound covers the effective-energy proxy,\nnot the joint multi-state fit estimator.",
            fontsize=8, va="bottom", color="#444444")
    ax.set_title("Estimator scope and decision")
    fig.suptitle("Conditional Wilson F-lane gap/overlap screen — synthetic only")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_report(output: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    source_dir = out / "source_snapshot"
    source_dir.mkdir()
    source_paths = {
        "f_lane_gap_overlap_force_screen.py": Path(__file__),
        "ground_state_bound.py": TASK_DIR / "ground_state_bound.py",
        "spectral_q_bound_reader.py": TASK_DIR / "spectral_q_bound_reader.py",
    }
    for name, source in source_paths.items():
        shutil.copyfile(source, source_dir / name)
    source_hashes = {name: _sha(source_dir / name) for name in source_paths}

    eff = _toy_effective_energies()
    q_bounds = np.full((len(KERNELS), len(R_VALUES)), 0.25, dtype=float)
    gap_bounds = np.full_like(q_bounds, 0.20)
    metrics = _toy_metrics(0.25, 0.20)
    rows = _grid_rows()
    premise_rows = [row for row in rows if row["toy_premises_hold"]]
    interval_valid = [row for row in premise_rows
                      if row["conditional_interval_status"] == "VALID"]
    covered = [row for row in premise_rows if row["conditional_interval_contains_toy_force"]]
    passing = [row for row in premise_rows
               if row["conditional_interval_status"] == "VALID"
               and row["passes_one_percent_toy_budget"]]
    grid_summary = {
        "planned": len(rows),
        "premise_holds": len(premise_rows),
        "premise_violated": len(rows) - len(premise_rows),
        "conditional_interval_valid": len(interval_valid),
        "conditional_interval_abstained": len(premise_rows) - len(interval_valid),
        "truth_covered_when_premise_holds": len(covered),
        "one_percent_budget_pass": len(passing),
        "one_percent_budget_fail": len(premise_rows) - len(passing),
        "interpretation": "deterministic parameter grid, not Monte Carlo coverage",
    }
    with (out / "conditional_gap_overlap_grid.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    protocol = {
        "classification": "CONDITIONAL_POSITIVE_SPECTRUM_EFFECTIVE_ENERGY_SCREEN_NOT_PHYSICAL_RECEIPT",
        "estimand_note": "The selected repository F lane is stout-minus-identity ground-state force difference; this fixture separately reports absolute force identifiability and the finite-time effective-energy contrast bias.",
        "kernels_in_order": list(KERNELS),
        "radii": list(R_VALUES),
        "effective_energy_start_times": list(TIME_STARTS),
        "contamination_budget_time": CONTAMINATION_BUDGET_TIME,
        "positive_spectrum_equations": {
            "correlator": "C_k,R(T)=A0_k,R exp(-E0(R) T) [1+rho_k,R(T)]",
            "effective_energy": "Eeff_k,R(T)=E0(R)+log[(1+rho(T))/(1+rho(T+1))]",
            "condition": "0<=rho(T)<=Qmax exp(-Delta_min*T)",
            "energy_interval": "Eeff(T)-log1p(Qmax exp(-Delta_min*T)) <= E0(R) <= Eeff(T)",
            "force_interval": "[L_E0(R3)-U_E0(R2), U_E0(R3)-L_E0(R2)]/DeltaR",
            "kernel_contrast_proxy_bias": "sum_{kernel,R} log1p(Qmax[k,R] exp(-Delta_min[k,R]*T))",
        },
        "toy_truth": {
            "E0_by_R": {str(r): float(TOY_E0[i]) for i, r in enumerate(R_VALUES)},
            "absolute_force": metrics["absolute_ground_force_truth"],
            "shared_spectrum_ground_force_difference": 0.0,
            "overlap_ratio_by_kernel_R": {
                k: {str(r): float(TOY_OVERLAP_RATIO[ki, ri])
                    for ri, r in enumerate(R_VALUES)}
                for ki, k in enumerate(KERNELS)
            },
            "gap_by_kernel_R": {
                k: {str(r): float(TOY_GAP[ki, ri])
                    for ri, r in enumerate(R_VALUES)}
                for ki, k in enumerate(KERNELS)
            },
        },
        "display_assumptions": {"Q_max": 0.25, "Delta_min": 0.20},
        "toy_reference_force_lower": REFERENCE_FORCE_LOWER_TOY,
        "model_truncation_fraction": MODEL_TRUNCATION_FRACTION,
        "source_hashes": source_hashes,
        "production_data": "not accessed",
        "physical_score": None,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (out / "summary.json").write_text(json.dumps({
        "classification": protocol["classification"],
        "synthetic_denominator": {"planned": 4, "valid": 4, "failed": 0, "abstained": 0,
                                  "unit": "kernel-radius positive-spectrum corner"},
        "grid_denominator": grid_summary,
        "display_metrics": {
            key: (np.asarray(value).tolist()
                  if isinstance(value, (tuple, np.ndarray)) else value)
            for key, value in metrics.items()
        },
        "physical_score": None,
    }, indent=2, sort_keys=True) + "\n")
    _plot(rows, metrics, out / "conditional_force_envelope.png")

    report = [
        "# Conditional gap/overlap force screen",
        "",
        "**Classification:** `CONDITIONAL_POSITIVE_SPECTRUM_EFFECTIVE_ENERGY_SCREEN_NOT_PHYSICAL_RECEIPT`. The inputs are a deterministic positive one-excited-state toy. No Wilson measurement, sealed split, GPU, or QUDA data were read; `physical_score=null`.",
        "",
        "## Scope correction for the preceding finite-time witness",
        "",
        "The repository's selected F-lane estimand is the stout-minus-identity difference in ground-state force. In a shared transfer matrix, the ground-state energy at each R is common to all interpolating operators, so that exact ground-state difference is identically zero. The preceding non-identifiability witness varies the **absolute** force between two spectra; within each spectrum its operator directions share the same energies, so the selected stout-minus-identity difference remains zero in both. That witness therefore establishes non-identifiability of the absolute force, but it does not by itself demonstrate non-identifiability of the selected kernel-difference estimand or calibrate its joint-fit truncation error.",
        "",
        "This screen keeps those quantities separate: it gives a conditional interval for the absolute force, reports the exact shared-spectrum ground-force contrast as zero, and bounds the finite-time effective-energy contrast's excited-state bias. The last bound is **not** a validated bound for the repository's multi-state joint-fit estimator.",
        "",
        "## Formula and conditions",
        "",
        "For positive forward spectral weights, write `C_k,R(T)=A0_k,R exp(-E0(R) T) [1+rho_k,R(T)]`, where `rho(T)=sum_n r_n exp(-Delta_n T)`. If the external conditions `sum_n r_n <= Q_max` and `Delta_n >= Delta_min` hold for each kernel/radius, then:",
        "",
        "```text",
        "0 <= E_eff,k,R(T)-E0(R)",
        "  = log[(1+rho(T))/(1+rho(T+1))]",
        "  <= log1p(Q_max exp(-Delta_min T)) = b_k,R(T)",
        "E0(R) in intersection_{k,T} [E_eff,k,R(T)-b_k,R(T), E_eff,k,R(T)]",
        "F_abs in [L_E0(R=3)-U_E0(R=2), U_E0(R=3)-L_E0(R=2)] / DeltaR",
        "|F_eff,stout-F_eff,identity| <= sum_{k in {stout,identity}, R in {2,3}} b_k,R(5)",
        "```",
        "",
        "The interval is invariant under a common additive static-energy subtraction; it does not assume `E0>=0`. The `Q_max` and `Delta_min` values remain assumptions until independently bounded from an authorized spectrum receipt. No Monte Carlo coverage is claimed.",
        "",
        "## Toy screen result",
        "",
        f"The fixture uses shared `E0(R=2,3)=({TOY_E0[0]:.3f},{TOY_E0[1]:.3f})`, giving absolute force `{metrics['absolute_ground_force_truth']:.3f}` and exact stout-minus-identity ground-force difference `0`. Kernel/radius overlap ratios range from `{float(np.min(TOY_OVERLAP_RATIO)):.2f}` to `{float(np.max(TOY_OVERLAP_RATIO)):.2f}`; gaps range from `{float(np.min(TOY_GAP)):.2f}` to `{float(np.max(TOY_GAP)):.2f}`. With display assumptions `Q_max=0.25`, `Delta_min=0.20`, the conditional absolute-force interval is `[{metrics['force_interval'][0]:.6f}, {metrics['force_interval'][1]:.6f}]` and contains the toy truth.",
        f"At T=5 the finite-time effective-energy contrast is `{metrics['finite_time_candidate_minus_reference_force']:.6f}` while the true shared-spectrum contrast is zero. Its observed absolute toy bias is `{metrics['finite_time_candidate_minus_reference_bias_abs']:.6f}`, or `{metrics['observed_proxy_bias_to_budget_ratio']:.2f}x` the budget. The four-corner conditional bias bound is `{metrics['four_corner_bias_bound']:.6f}`. Against the toy exact reference-force lower value `{REFERENCE_FORCE_LOWER_TOY:.3f}`, the one-percent model-truncation budget is `{metrics['one_percent_toy_budget']:.6f}`; the conservative bound is `{metrics['budget_ratio']:.1f}x` that limit and therefore does not pass this toy screen.",
        "",
        "![Conditional force interval and Q/gap budget screen](conditional_force_envelope.png)",
        "",
        f"The deterministic grid has {grid_summary['planned']} planned `(Q_max, Delta_min)` points. The toy premises hold at {grid_summary['premise_holds']} points; all such points cover the toy absolute force, and {grid_summary['one_percent_budget_pass']} meet the one-percent proxy-bias budget. This is algebraic sensitivity, not a statistical coverage rate.",
        "",
        "## Limits and handoff",
        "",
        "This result connects the scalar `Q exp(-Delta_min T)` envelope in `ground_state_bound.py` to effective-energy intervals and to the sum of four kernel/radius proxy-bias terms. The `spectral_q_bound_reader.py` supplies a different, fit-receipt-dependent sum over statewise overlap-ratio and gap bounds; its successful parsing remains conditional, not proof that the bounds have coverage. Neither route validates the multi-state joint-fit estimator, finite-Nt thermal/mixed contributions, omitted-state coverage, or the selected physical force contrast.",
        "",
        "For a production advance, the independent receipt must establish each kernel/radius `Q_max` and `Delta_min`, their simultaneous coverage, and the shared transfer-spectrum convention. Then an estimator-specific bridge must compare the actual joint-fit force contrast with this conservative effective-energy proxy on disjoint calibration inputs. Until then keep both Wilson lanes `NOT_READY/UNCERTIFIED`, F=`GROUND_STATE_FORCE_UNCERTIFIED`, and `physical_score=null`; sealed inputs remain closed.",
        "",
        "## Reproduction",
        "",
        "Run from the repository root: `python -m tasks.wilson_loop.f_lane_gap_overlap_force_screen`. The output directory is no-clobber. `protocol.json`, `conditional_gap_overlap_grid.csv`, `summary.json`, the figure, and `source_snapshot/` bind the assumptions and source bytes. The synthetic denominator is 4/4/0/0 kernel-radius corners; the parameter-grid denominator is 3,721 deterministic cells.",
    ]
    (out / "README.md").write_text("\n".join(report) + "\n")
    artifact_paths = [p for p in out.rglob("*") if p.is_file()]
    hashes = {str(p.relative_to(out)): _sha(p) for p in artifact_paths}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    return {
        "output": str(out),
        "metrics": {
            key: (np.asarray(value).tolist()
                  if isinstance(value, (tuple, np.ndarray)) else value)
            for key, value in metrics.items()
        },
        "grid_denominator": grid_summary,
        "artifact_hashes": len(hashes),
        "source_hashes": source_hashes,
        "finite_time_energies": eff.tolist(),
        "q_bounds": q_bounds.tolist(),
        "gap_bounds": gap_bounds.tolist(),
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
