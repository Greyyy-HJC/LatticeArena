"""Cross-task pooled initialization stress for trainable-W + SLSQP.

This CPU diagnostic tests a mechanism absent from the per-panel solver stress:
whether a parameter vector trained on independent synthetic tasks gives a
useful warm start on new tasks. It keeps the joint-W width-2 class and online
SLSQP fixed, and includes a fixed degree-4 Chebyshev comparator. It is not a
physical smear score or production benchmark.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize

from . import trainable_w as model
from . import trainable_w_stress_diagnostic as stress
from .features import directional_features
from .kernels import apply_kernel
from .trainable_w_operator_diagnostic import operator_loss, solve, spectral


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/optimize_gaussian/trainable_w_meta_init_20261123_v1"
BASE_TEACHER = stress.LOCAL_TEACHER_COEFFICIENTS.copy()
TEACHER_RELATIVE_SD = 0.20
TRAINING_PANELS = 12
EVALUATION_PANELS = 12
ONLINE_BUDGETS = (8, 32, 128)
TRAINING_BUDGET = 256
# Shared per-panel thresholds are fixed fractions of the seeded cold-start fit
# loss; they quantify convergence effort, not physical effect size.
SYNTHETIC_FIT_FRACTIONS = (0.10, 0.01)
SEED = 2026112401
HIDDEN = 2
SIZE = stress.SIZE


class ObjectiveCallCap(Exception):
    pass


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _identity_field(size: int) -> np.ndarray:
    dimension = 3 * size**3
    return np.eye(dimension).reshape(size, size, size, 3, dimension).astype(complex)


def make_local_case(rng: np.random.Generator,
                    teacher_coefficients: np.ndarray,
                    generator: str = "weak_field_eps0.08") -> dict:
    """Generate one held-out-task sample under a varying local teacher."""
    coefficients = np.asarray(teacher_coefficients, dtype=float)
    if coefficients.shape != (6,) or not np.all(np.isfinite(coefficients)):
        raise ValueError("teacher_coefficients must be a finite six-vector")
    row = stress.make_case(rng, generator, "local_environment", 0.0)
    links, z = row["links"], row["z"]
    size = links.shape[1]
    dimension = 3 * size**3
    identity = _identity_field(size)
    basis = stress.local_polynomial_basis(z)
    target_field = np.einsum("dzyxp,p->dzyx", basis, coefficients)
    target = apply_kernel(identity, links, target_field).reshape(dimension, dimension)

    linear = np.empty((3, size, size, size), dtype=float)
    for direction in range(3):
        for zpos, ypos, xpos in np.ndindex((size, size, size)):
            xyz = [xpos, ypos, zpos]
            xyz[direction] = (xyz[direction] + 1) % size
            j = ((xyz[2] * size + xyz[1]) * size + xyz[0]) * 3
            i = ((zpos * size + ypos) * size + xpos) * 3
            linear[direction, zpos, ypos, xpos] = -2 * np.vdot(
                links[direction, zpos, ypos, xpos],
                target[i:i + 3, j:j + 3],
            ).real
    denominator = float(np.linalg.norm(target) ** 2)
    if denominator <= 0:
        raise ValueError("synthetic target norm must be positive")
    row.update({
        "target": target,
        "linear": linear,
        "constant": float(np.linalg.norm(np.eye(dimension) - target) ** 2),
        "denominator": denominator,
        "teacher_coefficients": coefficients.tolist(),
    })
    return row


def _sample_teacher(rng: np.random.Generator) -> np.ndarray:
    return BASE_TEACHER * (1.0 + rng.normal(0.0, TEACHER_RELATIVE_SD, size=6))


def _random_initial(rng: np.random.Generator) -> np.ndarray:
    return model.pack(
        rng.normal(size=(HIDDEN, 2)) * 0.4,
        rng.normal(size=HIDDEN) * 0.2,
        0.04,
    )


def _fixed_initial() -> np.ndarray:
    """Predeclared nonzero control start; avoids the zero-v stationary point."""
    return model.pack(
        np.array([[0.40, 0.10], [-0.15, 0.35]]),
        np.array([0.20, -0.08]),
        0.04,
    )


def fit_pooled_prior(training_fit: list[dict], initial: np.ndarray,
                     *, budget: int = TRAINING_BUDGET,
                     panel_count: int | None = None) -> dict:
    """Fit one shared joint-W vector from training fit samples only."""
    budget = _positive_integer(budget, "budget")
    if not training_fit:
        raise ValueError("training_fit must be nonempty")
    if panel_count is None:
        if len(training_fit) % 2:
            raise ValueError("panel_count is required when fit links are not paired")
        panel_count = len(training_fit) // 2
    panel_count = _positive_integer(panel_count, "panel_count")
    start = np.asarray(initial, dtype=float)
    if start.shape != (3 * HIDDEN + 1,) or not np.all(np.isfinite(start)):
        raise ValueError("initial must contain seven finite joint-W parameters")
    initial_loss, _ = operator_loss(start, training_fit, HIDDEN)
    best_parameters = start.copy()
    best_loss = initial_loss
    trace: list[dict] = []
    started = time.perf_counter()

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal best_parameters, best_loss
        if len(trace) >= budget:
            raise ObjectiveCallCap
        loss, gradient = operator_loss(parameters, training_fit, HIDDEN)
        trace.append({
            "call": len(trace) + 1,
            "loss": loss,
            "gradient_norm": float(np.linalg.norm(gradient)),
            "parameters": np.asarray(parameters).tolist(),
        })
        if loss < best_loss:
            best_loss = loss
            best_parameters = np.asarray(parameters, dtype=float).copy()
        return loss, gradient

    success = False
    status = None
    message = ""
    cap_stop = False
    try:
        result = minimize(
            objective, start, jac=True, method="SLSQP",
            bounds=[(-3.0, 3.0)] * len(start),
            options={"maxiter": 2000, "ftol": 1e-13},
        )
        success = bool(result.success)
        status = int(result.status)
        message = str(result.message)
    except ObjectiveCallCap:
        cap_stop = True
        message = "pooled_objective_call_cap"
    return {
        "method": "pooled training-fit SLSQP",
        "planned_calls": budget,
        "calls": len(trace),
        "optimizer_objective_evaluations": len(trace),
        "initial_diagnostic_evaluations": 1,
        "total_objective_evaluations": len(trace) + 1,
        "panel_equivalent_calls": len(trace) * panel_count,
        "total_panel_equivalent_evaluations": (len(trace) + 1) * panel_count,
        "link_case_evaluations": len(trace) * len(training_fit),
        "success": success,
        "status": status,
        "budget_exhausted": cap_stop,
        "message": message,
        "initial_loss": initial_loss,
        "best_training_fit_loss": best_loss,
        "parameters": best_parameters.tolist(),
        "optimization_seconds": time.perf_counter() - started,
        "trace": trace,
    }


def _calls_to_fit_tolerance(initial: np.ndarray, fit: list[dict],
                            trace: list[dict], tolerance: float) -> int | None:
    start_loss, _ = operator_loss(initial, fit, HIDDEN)
    if start_loss <= tolerance:
        return 0
    for row in trace:
        if row["loss"] <= tolerance:
            return int(row["call"])
    return None


def minimum_break_even_tasks(total_training_panel_equivalent: int,
                             cold_start_calls: list[int],
                             learned_init_calls: list[int]) -> int | None:
    """First integer K with pooled training cost strictly amortized.

    Calls are summed over the same observed panels, so the integer inequality
    avoids floating-point rounding at an exact break-even boundary.
    """
    total_training_panel_equivalent = _positive_integer(
        total_training_panel_equivalent, "total_training_panel_equivalent",
    )
    if (not cold_start_calls or len(cold_start_calls) != len(learned_init_calls)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in cold_start_calls + learned_init_calls)):
        raise ValueError("call totals must be equal-length lists of nonnegative integers")
    saving = sum(cold_start_calls) - sum(learned_init_calls)
    if saving <= 0:
        return None
    numerator = total_training_panel_equivalent * len(cold_start_calls)
    return numerator // saving + 1


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _denominators(training_panels: list[dict], evaluation_panels: list[dict],
                  runs: list[dict], spectral_rows: list[dict],
                  local_baseline_rows: list[dict],
                  tolerance_rows: list[dict]) -> list[dict]:
    rows = [
        {"unit": "training panel", "planned": len(training_panels),
         "valid": len(training_panels), "failed": 0, "abstained": 0,
         "interpretation": "Synthetic tasks used only through fit link samples."},
        {"unit": "training fit link target", "planned": sum(len(p["fit"]) for p in training_panels),
         "valid": sum(len(p["fit"]) for p in training_panels), "failed": 0,
         "abstained": 0, "interpretation": "Input to one pooled prior fit."},
        {"unit": "evaluation panel", "planned": len(evaluation_panels),
         "valid": len(evaluation_panels), "failed": 0, "abstained": 0,
         "interpretation": "Independent synthetic panels; heldout links are never used for training."},
        {"unit": "evaluation fit link target", "planned": sum(len(p["fit"]) for p in evaluation_panels),
         "valid": sum(len(p["fit"]) for p in evaluation_panels), "failed": 0,
         "abstained": 0, "interpretation": "Online SLSQP adaptation only."},
        {"unit": "evaluation heldout link target", "planned": sum(len(p["heldout"]) for p in evaluation_panels),
         "valid": sum(len(p["heldout"]) for p in evaluation_panels), "failed": 0,
         "abstained": 0, "interpretation": "Paired final diagnostics; excluded from fitting and prior training."},
        {"unit": "online solver run", "planned": len(runs),
         "valid": sum(r["returned"] for r in runs),
         "failed": sum(not r["returned"] for r in runs), "abstained": 0,
         "interpretation": "Budget stops and solver non-success are counted separately, not excluded."},
        {"unit": "degree-4 spectral baseline", "planned": len(spectral_rows),
         "valid": sum(r["returned"] for r in spectral_rows),
         "failed": sum(not r["returned"] for r in spectral_rows), "abstained": 0,
         "interpretation": "Fit on evaluation fit links, scored once on independent heldout links."},
        {"unit": "target-matched fixed local degree-2 baseline",
         "planned": len(local_baseline_rows),
         "valid": sum(r["returned"] for r in local_baseline_rows),
         "failed": sum(not r["returned"] for r in local_baseline_rows),
         "abstained": 0,
         "interpretation": "Synthetic target-family control; fit on evaluation fit links, score on heldout."},
    ]
    for row in tolerance_rows:
        rows.append({
            "unit": f"{row['method']} reached {row['fit_fraction_of_cold_initial']} of cold-start initial fit loss",
            "planned": row["planned_panels"], "valid": row["reached_panels"],
            "failed": row["not_reached_panels"], "abstained": 0,
            "interpretation": "Shared synthetic training-fit contraction target only; not a physical r_star or heldout stop rule.",
        })
    return rows


def _plot(out: Path, rows: list[dict], tolerance_rows: list[dict],
          spectral_rows: list[dict], local_baseline_rows: list[dict],
          training_prior: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"pooled_learned_init": "#0072B2",
              "random_cold_start": "#D55E00",
              "fixed_nonzero_control": "#009E73"}
    labels = {"pooled_learned_init": "pooled train-only init",
              "random_cold_start": "seeded random start",
              "fixed_nonzero_control": "fixed nonzero control"}
    arms = list(colors)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    caps = sorted({r["budget"] for r in rows})
    for arm in arms:
        medians = []
        for cap in caps:
            values = [r["heldout_loss"] for r in rows
                      if r["arm"] == arm and r["budget"] == cap]
            medians.append(float(np.median(values)) if values else float("nan"))
            axes[0].scatter([cap] * len(values), values, color=colors[arm],
                            alpha=0.23, s=12)
        axes[0].plot(caps, medians, marker="o", color=colors[arm], label=labels[arm])
    for index, row in enumerate(spectral_rows):
        axes[0].scatter([max(caps)], [row["heldout_loss"]],
                        marker="x", color="#CC79A7", alpha=0.45, s=18,
                        label="degree-4 Chebyshev" if index == 0 else None)
    for index, row in enumerate(local_baseline_rows):
        axes[0].scatter([max(caps)], [row["heldout_loss"]],
                        marker="^", color="#F0E442", edgecolor="#333333",
                        alpha=0.55, s=22,
                        label="target-matched local degree-2" if index == 0 else None)
    axes[0].set(xlabel="online SLSQP objective/gradient call cap",
                ylabel="independent heldout relative operator loss",
                title="Same joint-W class; separate evaluation tasks",
                yscale="log")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.2)

    cap = max(caps)
    for panel_id in sorted({r["panel_id"] for r in rows}):
        selected = [r for r in rows if r["panel_id"] == panel_id
                    and r["budget"] == cap]
        for row in selected:
            start = row["initial_heldout_loss"]
            end = row["heldout_loss"]
            axes[1].plot([0, 1], [start, end], color=colors[row["arm"]],
                         alpha=0.24, marker="o")
    for arm in arms:
        selected = [r for r in rows if r["arm"] == arm and r["budget"] == cap]
        axes[1].plot([0, 1], [np.median([r["initial_heldout_loss"] for r in selected]),
                               np.median([r["heldout_loss"] for r in selected])],
                     color=colors[arm], marker="o", linewidth=2,
                     label=labels[arm])
    axes[1].set_xticks([0, 1], ["initial", f"adapted ({cap} cap)"])
    axes[1].set(ylabel="independent heldout relative operator loss",
                title="Paired initial-to-adapted diagnostic", yscale="log")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.2)

    successful_tolerances = [r for r in tolerance_rows
                             if r["reached_panels"] == r["planned_panels"]]
    ks = np.arange(1, 129)
    for fraction in sorted({r["fit_fraction_of_cold_initial"] for r in successful_tolerances}):
        selected = [r for r in successful_tolerances
                    if r["fit_fraction_of_cold_initial"] == fraction]
        for method in ("pooled_learned_init", "random_cold_start"):
            row = next((r for r in selected if r["method"] == method), None)
            if row is None:
                continue
            if method == "pooled_learned_init":
                values = (training_prior["total_panel_equivalent_evaluations"]
                          + ks * row["mean_calls_to_tolerance"])
            else:
                values = ks * row["mean_calls_to_tolerance"]
            axes[2].plot(ks, values, label=f"{labels[method]}, target={fraction:g}x cold initial",
                         color=colors[method],
                         linestyle="-" if method == "pooled_learned_init" else "--")
    if not successful_tolerances:
        axes[2].text(0.5, 0.5,
                     "No shared tolerance reached\nNo break-even estimated",
                     ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set(xlabel="number of independent evaluation tasks K",
                ylabel="panel-equivalent objective/gradient calls",
                title="Training cost amortization at shared fit contraction")
    handles, _ = axes[2].get_legend_handles_labels()
    if handles:
        axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.2)
    fig.suptitle("Synthetic pooled initialization + SLSQP; no physical-score claim")
    fig.savefig(out / "meta_init_diagnostic.png", dpi=180)
    plt.close(fig)


def build(output: str | Path = DEFAULT_OUT, *,
          training_panels: int = TRAINING_PANELS,
          evaluation_panels: int = EVALUATION_PANELS,
          online_budgets: tuple[int, ...] = ONLINE_BUDGETS,
          training_budget: int = TRAINING_BUDGET,
          seed: int = SEED) -> dict:
    training_panels = _positive_integer(training_panels, "training_panels")
    evaluation_panels = _positive_integer(evaluation_panels, "evaluation_panels")
    training_budget = _positive_integer(training_budget, "training_budget")
    if (not online_budgets or any(isinstance(n, bool) or not isinstance(n, int) or n < 1
                                  for n in online_budgets)
            or tuple(sorted(set(online_budgets))) != tuple(online_budgets)):
        raise ValueError("online_budgets must be strictly increasing positive integers")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)

    spec = {
        "schema": "trainable-w-pooled-meta-initialization-v1",
        "status": "SYNTHETIC_CROSS_TASK_INITIALIZATION_DIAGNOSTIC_ONLY",
        "physical_score": None,
        "data_access": "No physical gauge links, sealed data, GPU, or QUDA data accessed.",
        "training_panels": training_panels,
        "evaluation_panels": evaluation_panels,
        "fit_links_per_panel": 2,
        "heldout_links_per_evaluation_panel": 2,
        "heldout_use": "evaluation-only; never used to fit pooled prior or online SLSQP",
        "target_family": "local_environment polynomial teacher with task-level coefficient variation",
        "base_teacher_coefficients": BASE_TEACHER.tolist(),
        "teacher_relative_sd": TEACHER_RELATIVE_SD,
        "feature_generator": "weak_field_eps0.08 SU(3)-like synthetic links",
        "parameterization": "joint trainable-W width-2; seven real parameters in [-3,3]",
        "pooled_training_solver": "SLSQP with analytic gradient",
        "pooled_training_objective_call_cap": training_budget,
        "online_solver": "SLSQP with analytic gradient, same class and box for every arm",
        "online_budgets": list(online_budgets),
        "arms": ["pooled_learned_init", "random_cold_start", "fixed_nonzero_control"],
        "strong_baseline": "degree-4 Chebyshev in uniform hopping H, fit per evaluation panel",
        "target_matched_control": "fixed degree-2 local polynomial operator basis, fit per evaluation panel",
        "synthetic_fit_loss_fractions_of_cold_initial": list(SYNTHETIC_FIT_FRACTIONS),
        "tolerance_warning": "Panel-specific shared training-fit contraction thresholds only; not D06 physical r_star or heldout selection rules.",
        "seed": seed,
        "objective_call_accounting": {
            "pooled_call_link_cases": 2 * training_panels,
            "panel_equivalent_pooled_optimizer_cost": "optimizer calls * training_panels",
            "panel_equivalent_total_cost": "(optimizer calls + one initial diagnostic evaluation) * training_panels",
            "prior_inference": "copy seven frozen parameters; counted per evaluation task, no objective calls",
            "optimizer_and_baseline_fit_walltime": "reported separately; synthetic input/target generation is excluded and not timed",
        },
        "source_hashes": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "trainable_w_meta_init_diagnostic.py",
                "trainable_w_stress_diagnostic.py",
                "trainable_w_operator_diagnostic.py",
                "trainable_w.py", "features.py", "kernels.py",
            )
        },
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "scipy": scipy.__version__},
        "limitations": [
            "synthetic task distribution and shared teacher family only",
            "no physical reference or D06 effect threshold",
            "degree-4 spectral comparator is a different operator family",
            "small panel counts do not support population inference",
        ],
    }
    (out / "protocol.json").write_text(json.dumps(spec, indent=2) + "\n")

    training = []
    evaluation = []
    arrays: dict[str, np.ndarray] = {}
    for i in range(training_panels):
        rng = np.random.default_rng(seed + 1000 + i)
        teacher = _sample_teacher(rng)
        fit = [make_local_case(rng, teacher) for _ in range(2)]
        training.append({"panel_id": f"train_{i:02d}", "seed": seed + 1000 + i,
                         "teacher_coefficients": teacher.tolist(), "fit": fit})
        arrays[f"train_{i:02d}_fit_links"] = np.stack([row["links"] for row in fit])
        arrays[f"train_{i:02d}_fit_targets"] = np.stack([row["target"] for row in fit])
    for i in range(evaluation_panels):
        rng = np.random.default_rng(seed + 10000 + i)
        teacher = _sample_teacher(rng)
        fit = [make_local_case(rng, teacher) for _ in range(2)]
        heldout = [make_local_case(rng, teacher) for _ in range(2)]
        evaluation.append({"panel_id": f"eval_{i:02d}", "seed": seed + 10000 + i,
                           "teacher_coefficients": teacher.tolist(),
                           "fit": fit, "heldout": heldout})
        arrays[f"eval_{i:02d}_fit_links"] = np.stack([row["links"] for row in fit])
        arrays[f"eval_{i:02d}_fit_targets"] = np.stack([row["target"] for row in fit])
        arrays[f"eval_{i:02d}_heldout_links"] = np.stack([row["links"] for row in heldout])
        arrays[f"eval_{i:02d}_heldout_targets"] = np.stack([row["target"] for row in heldout])

    training_fit = [row for panel in training for row in panel["fit"]]
    prior_rng = np.random.default_rng(seed + 20000)
    initial_training = _random_initial(prior_rng)
    prior = fit_pooled_prior(
        training_fit, initial_training, budget=training_budget,
        panel_count=training_panels,
    )
    learned_initial = np.asarray(prior["parameters"], dtype=float)
    fixed_initial = _fixed_initial()

    runs = []
    spectral_rows = []
    local_baseline_rows = []
    tolerance_rows = []
    for panel in evaluation:
        fit, heldout = panel["fit"], panel["heldout"]
        init_rng = np.random.default_rng(panel["seed"] + 777)
        initials = {
            "pooled_learned_init": learned_initial,
            "random_cold_start": _random_initial(init_rng),
            "fixed_nonzero_control": fixed_initial,
        }
        cold_initial_fit_loss, _ = operator_loss(
            initials["random_cold_start"], fit, HIDDEN,
        )
        panel_spectral = spectral(fit, heldout, degree=4)
        spectral_rows.append({"panel_id": panel["panel_id"], **panel_spectral,
                              "returned": True})
        local_started = time.perf_counter()
        panel_local = stress.fit_local_polynomial_baseline(fit, heldout)
        local_baseline_rows.append({"panel_id": panel["panel_id"], **panel_local,
                                    "setup_and_fit_seconds": time.perf_counter() - local_started,
                                    "returned": True})
        for arm, initial in initials.items():
            initial_fit_loss, _ = operator_loss(initial, fit, HIDDEN)
            initial_heldout_loss, _ = operator_loss(initial, heldout, HIDDEN)
            cap_runs = {}
            for budget in online_budgets:
                run = solve(initial, fit, heldout, "SLSQP", budget, True)
                row = {
                    "panel_id": panel["panel_id"],
                    "panel_seed": panel["seed"],
                    "teacher_coefficients": panel["teacher_coefficients"],
                    "arm": arm,
                    "budget": budget,
                    "initial_fit_loss": initial_fit_loss,
                    "initial_heldout_loss": initial_heldout_loss,
                    "returned": True,
                    **run,
                }
                runs.append(row)
                cap_runs[budget] = row
            cap = max(online_budgets)
            trace = cap_runs[cap]["trace"]
            for fraction in SYNTHETIC_FIT_FRACTIONS:
                target = fraction * cold_initial_fit_loss
                tolerance_rows.append({
                    "panel_id": panel["panel_id"],
                    "arm": arm,
                    "fit_fraction": fraction,
                    "common_fit_loss_target": target,
                    "initial_fit_loss": initial_fit_loss,
                    "calls_to_tolerance": _calls_to_fit_tolerance(
                        initial, fit, trace, target,
                    ),
                    "planned_panels": 1,
                })

    calls_to_tolerance_summary = []
    for fraction in SYNTHETIC_FIT_FRACTIONS:
        for arm in ("pooled_learned_init", "random_cold_start",
                    "fixed_nonzero_control"):
            selected = [r for r in tolerance_rows
                        if r["fit_fraction"] == fraction and r["arm"] == arm]
            reached = [r["calls_to_tolerance"] for r in selected
                       if r["calls_to_tolerance"] is not None]
            summary_row = {
                "method": arm,
                "fit_fraction_of_cold_initial": fraction,
                "mean_common_fit_loss_target": (
                    float(np.mean([r["common_fit_loss_target"] for r in selected]))
                    if selected else None
                ),
                "planned_panels": len(selected),
                "reached_panels": len(reached),
                "not_reached_panels": len(selected) - len(reached),
                "total_calls_to_tolerance": (sum(reached)
                                               if len(reached) == len(selected) else None),
                "mean_calls_to_tolerance": (
                    float(np.mean(reached)) if len(reached) == len(selected) and reached
                    else None
                ),
                "break_even_K_vs_cold": None,
            }
            calls_to_tolerance_summary.append(summary_row)
        learned_row = next(r for r in calls_to_tolerance_summary
                           if r["fit_fraction_of_cold_initial"] == fraction
                           and r["method"] == "pooled_learned_init")
        cold_row = next(r for r in calls_to_tolerance_summary
                        if r["fit_fraction_of_cold_initial"] == fraction
                        and r["method"] == "random_cold_start")
        if (learned_row["mean_calls_to_tolerance"] is not None
                and cold_row["mean_calls_to_tolerance"] is not None):
            selected_learned = [r for r in tolerance_rows
                                if r["fit_fraction"] == fraction
                                and r["arm"] == "pooled_learned_init"]
            selected_cold = [r for r in tolerance_rows
                             if r["fit_fraction"] == fraction
                             and r["arm"] == "random_cold_start"]
            if all(r["calls_to_tolerance"] is not None for r in selected_learned + selected_cold):
                learned_row["break_even_K_vs_cold"] = minimum_break_even_tasks(
                    prior["total_panel_equivalent_evaluations"],
                    [int(r["calls_to_tolerance"]) for r in selected_cold],
                    [int(r["calls_to_tolerance"]) for r in selected_learned],
                )
    denominators = _denominators(training, evaluation, runs, spectral_rows,
                                 local_baseline_rows,
                                 calls_to_tolerance_summary)
    _write_csv(out / "denominator_summary.csv", denominators)
    _write_csv(out / "paired_summary.csv", runs)
    _write_csv(out / "fit_tolerance_summary.csv", calls_to_tolerance_summary)
    _write_csv(out / "local_polynomial_baseline.csv", local_baseline_rows)
    np.savez_compressed(out / "synthetic_inputs.npz", **arrays)
    _plot(out, runs, calls_to_tolerance_summary, spectral_rows,
          local_baseline_rows, prior)

    raw = {
        "protocol": spec,
        "training_prior": prior,
        "training_panel_receipts": [
            {"panel_id": p["panel_id"], "seed": p["seed"],
             "teacher_coefficients": p["teacher_coefficients"],
             "fit_link_count": len(p["fit"])} for p in training
        ],
        "evaluation_panel_receipts": [
            {"panel_id": p["panel_id"], "seed": p["seed"],
             "teacher_coefficients": p["teacher_coefficients"],
             "fit_link_count": len(p["fit"]),
             "heldout_link_count": len(p["heldout"])} for p in evaluation
        ],
        "solver_runs": runs,
        "spectral_baselines": spectral_rows,
        "target_matched_local_baselines": local_baseline_rows,
        "tolerance_sensitivity": tolerance_rows,
        "denominators": denominators,
    }
    (out / "raw.json").write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
    summary = {
        "schema": "trainable-w-pooled-meta-init-summary-v1",
        "status": spec["status"],
        "physical_score": None,
        "data_access": spec["data_access"],
        "planned_training_panels": training_panels,
        "planned_evaluation_panels": evaluation_panels,
        "planned_solver_runs": len(runs),
        "returned_solver_runs": sum(row["returned"] for row in runs),
        "budget_exhausted_runs": sum(row["budget_exhausted"] for row in runs),
        "solver_non_success_runs": sum(not row["solver_success"] for row in runs),
        "pooled_training_calls": prior["calls"],
        "pooled_training_panel_equivalent_optimizer_calls": prior["panel_equivalent_calls"],
        "pooled_training_panel_equivalent_total_evaluations": prior["total_panel_equivalent_evaluations"],
        "median_heldout_loss_by_arm_at_max_budget": {
            arm: float(np.median([row["heldout_loss"] for row in runs
                                  if row["arm"] == arm and row["budget"] == max(online_budgets)]))
            for arm in ("pooled_learned_init", "random_cold_start",
                        "fixed_nonzero_control")
        },
        "median_spectral_degree4_heldout_loss": float(
            np.median([row["heldout_loss"] for row in spectral_rows])
        ),
        "median_target_matched_local_degree2_heldout_loss": float(
            np.median([row["heldout_loss"] for row in local_baseline_rows])
        ),
        "fit_tolerance_summary": calls_to_tolerance_summary,
        "files": [
            "README.md", "protocol.json", "raw.json", "paired_summary.csv",
            "fit_tolerance_summary.csv", "local_polynomial_baseline.csv",
            "denominator_summary.csv",
            "synthetic_inputs.npz", "meta_init_diagnostic.png",
            "snapshot_trainable_w_meta_init_diagnostic.py",
            "snapshot_trainable_w_operator_diagnostic.py",
            "snapshot_trainable_w_stress_diagnostic.py",
            "snapshot_trainable_w.py", "snapshot_features.py",
            "snapshot_kernels.py", "summary.json", "artifact_hashes.json",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    tolerance_table = [
        "|fraction of cold initial fit loss|arm|total calls to target|mean calls per panel|minimum K where pooled cost is strictly lower|",
        "|---:|---|---:|---:|---:|",
    ]
    for row in calls_to_tolerance_summary:
        break_even = (row["break_even_K_vs_cold"]
                      if row["method"] == "pooled_learned_init" else None)
        tolerance_table.append(
            f"|{row['fit_fraction_of_cold_initial']:.2f}|{row['method']}|"
            f"{row['total_calls_to_tolerance']}|{row['mean_calls_to_tolerance']}|{break_even}|"
        )
    readme = [
        "# Trainable-W cross-task pooled initialization stress",
        "",
        "**SYNTHETIC CPU DIAGNOSTIC ONLY. `physical_score=null`.** No physical links, sealed split, GPU, or QUDA data are read.",
        "",
        "## Question and split",
        "",
        f"This tests a mechanism missing from the previous per-panel stress: a shared width-2 trainable-W parameter vector is fitted on {training_panels} independent training panels and then used only as a warm start on {evaluation_panels} independent evaluation panels. Each panel draws a six-coefficient local-environment teacher around the frozen base vector with relative SD {TEACHER_RELATIVE_SD}; each training task supplies 2 fit links, and each evaluation task supplies 2 fit plus 2 untouched heldout links. The target family is intentionally synthetic and narrow.",
        "",
        "Training flow: `training seeds -> fit links only -> one pooled SLSQP prior -> freeze seven parameters`. Evaluation flow: `independent evaluation fit links -> learned/cold/fixed initialization -> same joint-W SLSQP caps -> heldout links scored once`. The degree-4 Chebyshev family is fitted on each evaluation fit split and scored on the same heldout links.",
        "",
        "![Pooled initialization, online fit and amortization diagnostics](meta_init_diagnostic.png)",
        "",
        "## Cost and failure accounting",
        "",
        f"The pooled prior used {prior['calls']} pooled optimizer objective/gradient calls over {training_panels*2} training link cases, or {prior['panel_equivalent_calls']} single-panel-equivalent optimizer calls. One additional initial-loss diagnostic is reported separately ({prior['total_panel_equivalent_evaluations']} total panel-equivalent objective evaluations including it). Every pooled evaluation visits all training fit cases; it is not counted as one ordinary task call. The initializer is a copied seven-parameter vector (no learned inference model). Optimizer and baseline fit seconds are saved separately; synthetic input/target generation is excluded and not timed, so no production walltime comparison is claimed. Online SLSQP caps are `{list(online_budgets)}` for all three initialization arms.",
        "",
        f"Two shared training-fit targets are fixed at fractions `{SYNTHETIC_FIT_FRACTIONS}` of each panel's seeded cold-start initial fit loss. They require substantial synthetic fit-loss contraction but are not D06 physical `r_star`; heldout loss does not choose or stop training. A minimum amortizing task count is estimated only if every panel reaches the same target for both pooled and cold starts and pooled initialization uses fewer online calls. It is the first integer K where observed cold-start call savings strictly exceed the {prior['total_panel_equivalent_evaluations']} panel-equivalent training/initial-diagnostic cost. If all arms already start below a target, or a target is missed on any panel, no break-even value is estimated for that target.",
        "",
        *tolerance_table,
        "",
        "",
        "`denominator_summary.csv` retains training/evaluation panels, fit/heldout links, solver runs, strong-baseline fits, and per-arm tolerance hits as planned/valid/failed/abstained. Solver cap stops and non-success outcomes remain in the raw trace and are not excluded.",
        "",
        "## Results and limitations",
        "",
        f"At the largest online cap, median heldout losses are pooled learned init `{summary['median_heldout_loss_by_arm_at_max_budget']['pooled_learned_init']:.6g}`, random cold start `{summary['median_heldout_loss_by_arm_at_max_budget']['random_cold_start']:.6g}`, fixed nonzero control `{summary['median_heldout_loss_by_arm_at_max_budget']['fixed_nonzero_control']:.6g}`, degree-4 Chebyshev `{summary['median_spectral_degree4_heldout_loss']:.6g}`, and target-matched fixed local degree-2 `{summary['median_target_matched_local_degree2_heldout_loss']:.6g}`. These descriptive medians over {evaluation_panels} synthetic tasks are not population estimates or physical claims.",
        "",
        "A pooled initialization is a *hybrid* with SLSQP, not a competing optimizer guarantee. Chebyshev is the requested fixed spectral comparator, but this diagnostic's local-environment teacher is deliberately non-spectral; the fixed degree-2 local baseline is a target-matched synthetic family control. Neither synthetic baseline selects the physical D11 operator. The experiment can support a finite-budget warm-start effect only under this declared synthetic task distribution. It does not resolve D04 physical reference, D06 minimum effect, D11 operator/positivity contract, GPU timing, or sealed confirmation. Keep the Optimize task `NOT_READY/UNCERTIFIED` and `physical_score=null`.",
        "",
        "## Rebuild",
        "",
        "Use a new output directory; existing artifacts are never overwritten:",
        "",
        "```text",
        "python -B -m tasks.optimize_gaussian.trainable_w_meta_init_diagnostic --output NEW_DIRECTORY",
        "```",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n")

    source_paths = {
        "snapshot_trainable_w_meta_init_diagnostic.py": Path(__file__),
        "snapshot_trainable_w_operator_diagnostic.py": Path(__file__).with_name("trainable_w_operator_diagnostic.py"),
        "snapshot_trainable_w_stress_diagnostic.py": Path(__file__).with_name("trainable_w_stress_diagnostic.py"),
        "snapshot_trainable_w.py": Path(__file__).with_name("trainable_w.py"),
        "snapshot_features.py": Path(__file__).with_name("features.py"),
        "snapshot_kernels.py": Path(__file__).with_name("kernels.py"),
    }
    for target, source in source_paths.items():
        shutil.copyfile(source, out / target)
    hashes = {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in summary["files"] if name != "artifact_hashes.json"
    }
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--training-panels", type=int, default=TRAINING_PANELS)
    parser.add_argument("--evaluation-panels", type=int, default=EVALUATION_PANELS)
    parser.add_argument("--training-budget", type=int, default=TRAINING_BUDGET)
    args = parser.parse_args()
    summary = build(
        args.output, training_panels=args.training_panels,
        evaluation_panels=args.evaluation_panels,
        training_budget=args.training_budget,
    )
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"files", "fit_tolerance_summary"}}, indent=2))


if __name__ == "__main__":
    main()
