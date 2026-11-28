"""Source-closed synthetic stress tests for trainable-W operator fitting.

The fixtures vary gauge-feature generators, target families, label noise, and
call budgets. They are algorithmic diagnostics, not physical smear scores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
from pathlib import Path

import numpy as np
import scipy

from . import trainable_w as model
from .features import directional_features
from .kernels import apply_kernel
from .trainable_w_diagnostic import su3
from .trainable_w_operator_diagnostic import operator_loss, solve, spectral


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/optimize_gaussian/trainable_w_stress_20261123_v4"
GENERATORS = (
    ("haar_like", None),
    ("weak_field_eps0.08", 0.08),
    ("weak_field_eps0.25", 0.25),
)
TARGET_FAMILIES = ("spectral_heat", "local_environment")
NOISE_LEVELS = (0.0, 0.02)
REPLICATES = 3
FIT_LINKS = 2
HELDOUT_LINKS = 2
CALL_BUDGETS = (32, 128)
HIDDEN = 2
SIZE = 3


def weak_field_links(rng: np.random.Generator, shape=(3, 3, 3, 3),
                     epsilon=0.08) -> np.ndarray:
    """Generate determinant-one links near identity from traceless Hermitian fields."""
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    links = np.empty((*shape, 3, 3), dtype=complex)
    identity = np.eye(3)
    for index in np.ndindex(shape):
        raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        hermitian = 0.5 * (raw + raw.conj().T)
        hermitian -= np.trace(hermitian).real / 3.0 * identity
        values, vectors = np.linalg.eigh(hermitian)
        links[index] = (vectors * np.exp(1j * epsilon * values)) @ vectors.conj().T
    return links


def _generator_links(rng: np.random.Generator, generator: str) -> np.ndarray:
    for name, epsilon in GENERATORS:
        if name == generator:
            if epsilon is None:
                return su3(rng, (3, SIZE, SIZE, SIZE))
            return weak_field_links(rng, (3, SIZE, SIZE, SIZE), epsilon)
    raise ValueError(f"unknown synthetic link generator: {generator}")


def local_polynomial_basis(z: np.ndarray) -> np.ndarray:
    """Return shared A1+ link-local degree-2 features (d,z,y,x,6)."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 5 or z.shape[0] != 3 or z.shape[-1] != 2:
        raise ValueError("z must have shape (3,z,y,x,2)")
    z0, z1 = z[..., 0], z[..., 1]
    c0 = z0 - z0.mean(axis=(1, 2, 3), keepdims=True)
    c1 = z1 - z1.mean(axis=(1, 2, 3), keepdims=True)
    terms = (np.ones_like(c0), c0, c1,
             c0 * c0 - (c0 * c0).mean(axis=(1, 2, 3), keepdims=True),
             c0 * c1 - (c0 * c1).mean(axis=(1, 2, 3), keepdims=True),
             c1 * c1 - (c1 * c1).mean(axis=(1, 2, 3), keepdims=True))
    return np.stack(terms, axis=-1)


LOCAL_TEACHER_COEFFICIENTS = np.array([0.045, 0.025, -0.018, 0.012, 0.008, -0.006])


def local_teacher_coefficients(z: np.ndarray) -> np.ndarray:
    return np.einsum("dzyxp,p->dzyx", local_polynomial_basis(z),
                     LOCAL_TEACHER_COEFFICIENTS)


def _identity(size: int) -> np.ndarray:
    dim = 3 * size**3
    return np.eye(dim).reshape(size, size, size, 3, dim).astype(complex)


def make_case(rng: np.random.Generator, generator: str, target_family: str,
              noise_level: float, *, size: int = SIZE) -> dict:
    """Construct one gauge-covariant synthetic operator target and loss data."""
    if target_family not in TARGET_FAMILIES:
        raise ValueError(f"target_family must be one of {TARGET_FAMILIES}")
    if not math.isfinite(noise_level) or noise_level < 0:
        raise ValueError("noise_level must be finite and nonnegative")
    links = _generator_links(rng, generator)
    z = directional_features(links)
    dim = 3 * size**3
    identity_matrix = np.eye(dim, dtype=complex)
    identity_field = _identity(size)
    hopping = apply_kernel(identity_field, links,
                           np.ones((3, size, size, size))) - identity_field
    h = hopping.reshape(dim, dim)
    if target_family == "spectral_heat":
        values, vectors = np.linalg.eigh(h)
        weights = np.exp(0.18 * values)
        target = (vectors * weights) @ vectors.conj().T / weights.mean()
    else:
        target_field = local_teacher_coefficients(z)
        target = apply_kernel(identity_field, links, target_field).reshape(dim, dim)

    noiseless_target = target.copy()
    if noise_level:
        random_field = rng.normal(size=(3, size, size, size))
        noise_operator = (apply_kernel(identity_field, links, random_field)
                          - identity_field).reshape(dim, dim)
        noise_norm = np.linalg.norm(noise_operator)
        if noise_norm <= 1e-14:
            raise ValueError("synthetic noise operator has zero norm")
        target = target + noise_level * np.linalg.norm(target) / noise_norm * noise_operator

    linear = np.empty((3, size, size, size), dtype=float)
    for direction in range(3):
        for zpos, ypos, xpos in np.ndindex((size, size, size)):
            xyz = [xpos, ypos, zpos]
            xyz[direction] = (xyz[direction] + 1) % size
            j = ((xyz[2] * size + xyz[1]) * size + xyz[0]) * 3
            i = ((zpos * size + ypos) * size + xpos) * 3
            linear[direction, zpos, ypos, xpos] = -2 * np.vdot(
                links[direction, zpos, ypos, xpos], target[i:i + 3, j:j + 3]
            ).real
    target_hermiticity = float(
        np.linalg.norm(target - target.conj().T) / max(np.linalg.norm(target), 1e-30))
    commutator = target @ h - h @ target
    commutator_relative = float(
        np.linalg.norm(commutator)
        / max(np.linalg.norm(target) * np.linalg.norm(h), 1e-30))
    return {
        "links": links,
        "z": z,
        "target": target,
        "noiseless_target": noiseless_target,
        "h": h,
        "linear": linear,
        "constant": float(np.linalg.norm(identity_matrix - target) ** 2),
        "denominator": float(np.linalg.norm(target) ** 2),
        "target_checks": {
            "hermiticity_relative": target_hermiticity,
            "minimum_eigenvalue": float(np.linalg.eigvalsh(target).min()),
            "relative_commutator_with_uniform_hopping": commutator_relative,
        },
        "target_family": target_family,
        "generator": generator,
        "noise_level": float(noise_level),
    }


def _feature_condition(fit: list[dict]) -> dict:
    design = np.concatenate([
        local_polynomial_basis(row["z"]).reshape(-1, 6) for row in fit
    ], axis=0)
    singular = np.linalg.svd(design, compute_uv=False)
    tol = singular[0] * max(design.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(singular > tol))
    condition = (float(singular[0] / singular[-1])
                 if rank == design.shape[1] else None)
    return {"rank": rank, "columns": int(design.shape[1]),
            "condition_number": condition,
            "smallest_singular_value": float(singular[-1])}


def _polynomial_operator_basis(row: dict) -> np.ndarray:
    links = row["links"]
    size = links.shape[1]
    dim = 3 * size**3
    ident = _identity(size)
    basis = local_polynomial_basis(row["z"])
    matrices = []
    for column in range(basis.shape[-1]):
        op = apply_kernel(ident, links, basis[..., column]).reshape(dim, dim)
        matrices.append(op - np.eye(dim, dtype=complex))
    return np.stack(matrices, axis=-1)


def fit_local_polynomial_baseline(fit: list[dict], heldout: list[dict]) -> dict:
    """Fit a fixed degree-2 local feature map by global operator least squares."""
    dim = fit[0]["target"].shape[0]
    identity = np.eye(dim, dtype=complex)
    design_blocks, target_blocks = [], []
    fit_bases = [_polynomial_operator_basis(row) for row in fit]
    for row, matrices in zip(fit, fit_bases):
        scale = math.sqrt(row["denominator"])
        x = matrices.reshape(-1, matrices.shape[-1]) / scale
        y = (row["target"] - identity).reshape(-1) / scale
        design_blocks.append(np.concatenate([x.real, x.imag], axis=0))
        target_blocks.append(np.concatenate([y.real, y.imag]))
    x_all = np.concatenate(design_blocks, axis=0)
    y_all = np.concatenate(target_blocks, axis=0)
    coefficients, _, rank, singular = np.linalg.lstsq(x_all, y_all, rcond=None)

    def score(rows):
        losses = []
        for row in rows:
            matrices = _polynomial_operator_basis(row)
            prediction = identity + np.einsum("ijp,p->ij", matrices, coefficients)
            losses.append(float(np.linalg.norm(prediction - row["target"]) ** 2
                                / row["denominator"]))
        return float(np.mean(losses))

    cond = (float(singular[0] / singular[-1])
            if singular.size == x_all.shape[1] and singular[-1] > 0 else None)
    return {
        "baseline": "fixed_local_degree2_operator_basis",
        "parameters": int(len(coefficients)),
        "coefficients": coefficients.tolist(),
        "design_rank": int(rank),
        "design_condition_number": cond,
        "fit_loss": score(fit),
        "heldout_loss": score(heldout),
    }


def _case_key(generator: str, target: str, noise: float, replicate: int) -> str:
    return f"{generator}__{target}__noise{noise:.2f}__rep{replicate}"


def _build_plots(out: Path, panels: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = {}
    for panel in panels:
        key = (panel["generator"], panel["target_family"], panel["noise_level"])
        runs = panel["solver_runs"]
        joint = next(run for run in runs if run["arm"] == "joint_W_SLSQP"
                     and run["budget"] == max(CALL_BUDGETS))
        fixed2 = next(run for run in runs if run["arm"] == "fixed_W_width2_SLSQP"
                      and run["budget"] == max(CALL_BUDGETS))
        fixed6 = next(run for run in runs if run["arm"] == "fixed_W_width6_SLSQP"
                      and run["budget"] == max(CALL_BUDGETS))
        ratio2 = max(joint["heldout_loss"], 1e-30) / max(fixed2["heldout_loss"], 1e-30)
        ratio6 = max(joint["heldout_loss"], 1e-30) / max(fixed6["heldout_loss"], 1e-30)
        grouped.setdefault(key, {"fixed_width2": [], "fixed_width6": []})
        grouped[key]["fixed_width2"].append(math.log10(ratio2))
        grouped[key]["fixed_width6"].append(math.log10(ratio6))

    fig, ax = plt.subplots(figsize=(14, 5.5), layout="constrained")
    keys = sorted(grouped)
    labels = []
    for index, key in enumerate(keys):
        for contrast, offset, color, marker, label in (
            ("fixed_width2", -0.12, "#0072B2", "o", "joint W / fixed W width 2"),
            ("fixed_width6", 0.12, "#CC79A7", "s", "joint W / fixed W width 6"),
        ):
            values = grouped[key][contrast]
            ax.scatter(np.full(len(values), index + offset), values,
                       color=color, marker=marker, alpha=0.75,
                       label=label if index == 0 else None)
            ax.plot([index + offset - 0.06, index + offset + 0.06],
                    [float(np.median(values))] * 2, color=color, lw=2)
        labels.append(f"{key[0]}\n{key[1]}\nσ={key[2]:g}")
    ax.axhline(0.0, color="black", ls="--", lw=1,
               label="equal heldout error; below favors joint W")
    ax.set_xticks(range(len(keys)), labels, rotation=0, fontsize=7)
    ax.set_ylabel("log10(heldout loss: joint W / fixed W arm)")
    ax.set_title("Paired heldout contrasts; SLSQP, call cap 128; independent synthetic panels")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(out / "paired_loss_ratios.png", dpi=175)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(GENERATORS), figsize=(14, 8),
                             sharex=True, sharey="row", layout="constrained")
    arm_specs = (
        ("joint_W_SLSQP", "joint W / SLSQP", "#0072B2"),
        ("fixed_W_width2_SLSQP", "fixed W width 2 / SLSQP", "#D55E00"),
        ("joint_W_L-BFGS-B", "joint W / L-BFGS-B", "#009E73"),
        ("fixed_W_width6_SLSQP", "fixed W width 6 / SLSQP", "#CC79A7"),
    )
    for row_index, target in enumerate(TARGET_FAMILIES):
        for col_index, (generator, _) in enumerate(GENERATORS):
            ax = axes[row_index, col_index]
            subset = [panel for panel in panels
                      if panel["generator"] == generator
                      and panel["target_family"] == target
                      and panel["noise_level"] == max(NOISE_LEVELS)]
            for arm, label, color in arm_specs:
                medians = []
                for budget in CALL_BUDGETS:
                    losses = [next(run["heldout_loss"] for run in panel["solver_runs"]
                                   if run["arm"] == arm and run["budget"] == budget)
                              for panel in subset]
                    medians.append(float(np.median(losses)))
                ax.plot(CALL_BUDGETS, medians, marker="o", color=color, label=label)
                if arm == arm_specs[0][0]:
                    optimizer_range = list(medians)
                else:
                    optimizer_range.extend(medians)
            baseline_key = "spectral_baselines" if target == "spectral_heat" else "local_baseline"
            if target == "spectral_heat":
                baseline_losses = [next(row["heldout_loss"] for row in panel[baseline_key]
                                        if row["degree"] == 4) for panel in subset]
                baseline_label = "Chebyshev degree 4"
            else:
                baseline_losses = [panel[baseline_key]["heldout_loss"] for panel in subset]
                baseline_label = "fixed local degree-2 baseline"
            baseline_median = float(np.median(baseline_losses))
            if min(optimizer_range) <= baseline_median <= max(optimizer_range):
                ax.axhline(baseline_median, ls="--", color="black",
                           label=baseline_label)
            else:
                ax.text(0.98, 0.04,
                        f"{baseline_label}: {baseline_median:.2e} (outside optimizer scale)",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
            ax.set_yscale("log")
            ax.set_title(f"{target}; {generator}; noise=2%")
            ax.set_xlabel("objective + gradient call cap")
            if col_index == 0:
                ax.set_ylabel("median heldout relative squared operator error")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle("Budget sensitivity; medians over three synthetic panels, no confidence claim")
    fig.savefig(out / "budget_convergence.png", dpi=175)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), layout="constrained")
    for index, (generator, _) in enumerate(GENERATORS):
        values = [panel["feature_condition"]["condition_number"] for panel in panels
                  if panel["generator"] == generator
                  and panel["target_family"] == "local_environment"
                  and panel["noise_level"] == 0.0
                  and panel["feature_condition"]["condition_number"] is not None]
        axes[0].scatter(np.full(len(values), index), values, alpha=0.75)
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(GENERATORS)), [item[0] for item in GENERATORS], rotation=18)
    axes[0].set_ylabel("degree-2 local feature design condition number")
    axes[0].set_title("Conditioning by synthetic link family")
    axes[0].grid(axis="y", alpha=0.25)

    for index, (generator, _) in enumerate(GENERATORS):
        spectral_losses = [next(row["heldout_loss"] for row in panel["spectral_baselines"]
                                if row["degree"] == 4)
                           for panel in panels if panel["generator"] == generator
                           and panel["target_family"] == "spectral_heat"
                           and panel["noise_level"] == 0.0]
        local_losses = [panel["local_baseline"]["heldout_loss"]
                        for panel in panels if panel["generator"] == generator
                        and panel["target_family"] == "local_environment"
                        and panel["noise_level"] == 0.0]
        axes[1].scatter(np.full(len(spectral_losses), index - 0.08), spectral_losses,
                        color="#0072B2", label="Chebyshev-4 / spectral target" if index == 0 else None)
        axes[1].scatter(np.full(len(local_losses), index + 0.08), local_losses,
                        color="#D55E00", label="local polynomial / local target" if index == 0 else None)
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(GENERATORS)), [item[0] for item in GENERATORS], rotation=18)
    axes[1].set_ylabel("heldout relative squared operator error")
    axes[1].set_title("Target-matched fixed baselines")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.suptitle("Synthetic diagnostics only; baseline fit and heldout loss are reported separately")
    fig.savefig(out / "conditioning_and_baselines.png", dpi=175)
    plt.close(fig)


def _group_summary(panels: list[dict]) -> list[dict]:
    groups = {}
    for panel in panels:
        key = (panel["generator"], panel["target_family"], panel["noise_level"])
        groups.setdefault(key, []).append(panel)
    rows = []
    for (generator, target, noise), members in sorted(groups.items()):
        joint_runs, fixed2_runs, fixed6_runs = [], [], []
        for panel in members:
            joint_runs.append(next(row for row in panel["solver_runs"]
                                   if row["arm"] == "joint_W_SLSQP"
                                   and row["budget"] == max(CALL_BUDGETS)))
            fixed2_runs.append(next(row for row in panel["solver_runs"]
                                    if row["arm"] == "fixed_W_width2_SLSQP"
                                    and row["budget"] == max(CALL_BUDGETS)))
            fixed6_runs.append(next(row for row in panel["solver_runs"]
                                    if row["arm"] == "fixed_W_width6_SLSQP"
                                    and row["budget"] == max(CALL_BUDGETS)))
        log_ratios2 = [math.log10(max(j["heldout_loss"], 1e-30)
                                  / max(f["heldout_loss"], 1e-30))
                       for j, f in zip(joint_runs, fixed2_runs)]
        log_ratios6 = [math.log10(max(j["heldout_loss"], 1e-30)
                                  / max(f["heldout_loss"], 1e-30))
                       for j, f in zip(joint_runs, fixed6_runs)]
        rows.append({
            "generator": generator,
            "target_family": target,
            "noise_level_relative_frobenius": noise,
            "independent_panels": len(members),
            "joint_slsqp_median_heldout_loss_cap128": float(np.median([r["heldout_loss"] for r in joint_runs])),
            "fixed_w_width2_slsqp_median_heldout_loss_cap128": float(np.median([r["heldout_loss"] for r in fixed2_runs])),
            "fixed_w_width6_slsqp_median_heldout_loss_cap128": float(np.median([r["heldout_loss"] for r in fixed6_runs])),
            "median_log10_joint_over_fixed_width2": float(np.median(log_ratios2)),
            "min_log10_joint_over_fixed_width2": float(np.min(log_ratios2)),
            "max_log10_joint_over_fixed_width2": float(np.max(log_ratios2)),
            "median_log10_joint_over_fixed_width6_equal_scalar_count": float(np.median(log_ratios6)),
            "min_log10_joint_over_fixed_width6_equal_scalar_count": float(np.min(log_ratios6)),
            "max_log10_joint_over_fixed_width6_equal_scalar_count": float(np.max(log_ratios6)),
            "joint_slsqp_median_calls": float(np.median([r["calls"] for r in joint_runs])),
            "joint_slsqp_median_optimizer_seconds": float(np.median([r["optimization_seconds"] for r in joint_runs])),
            "joint_slsqp_median_projected_gradient": float(np.median([r["projected_gradient_norm"] for r in joint_runs])),
            "classification": "descriptive synthetic panel summary; not an interval or significance test",
        })
    return rows


def run(output: str | Path = DEFAULT_OUT) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    seed_plan = []
    for gi, (generator, _) in enumerate(GENERATORS):
        for ti, target in enumerate(TARGET_FAMILIES):
            for ni, noise in enumerate(NOISE_LEVELS):
                for replicate in range(REPLICATES):
                    seed_plan.append({
                        "generator": generator,
                        "target_family": target,
                        "noise_level": noise,
                        "replicate": replicate,
                        "seed": 2026112300 + gi * 10000 + ti * 1000 + ni * 100 + replicate,
                    })
    source_files = ["trainable_w_stress_diagnostic.py", "trainable_w.py",
                    "trainable_w_operator_diagnostic.py", "trainable_w_diagnostic.py",
                    "features.py", "kernels.py"]
    protocol = {
        "schema": "trainable-w-synthetic-stress-v1",
        "classification": "synthetic algorithm diagnostic only",
        "physical_score": None,
        "generator_families": [
            {"name": name, "weak_field_epsilon": epsilon}
            for name, epsilon in GENERATORS
        ],
        "target_families": list(TARGET_FAMILIES),
        "target_noise_relative_frobenius": list(NOISE_LEVELS),
        "replicates_per_cell": REPLICATES,
        "fit_links_per_panel": FIT_LINKS,
        "heldout_links_per_panel": HELDOUT_LINKS,
        "call_budgets": list(CALL_BUDGETS),
        "model_arms": ["joint_W_SLSQP", "joint_W_L-BFGS-B",
                       "fixed_W_width2_SLSQP", "fixed_W_width6_SLSQP"],
        "initialization": "paired within each panel; fixed width-6 arm shares width-2 prefix and adds four seeded features",
        "operator_loss": "full identity-basis relative squared Frobenius loss; same target and fit/heldout IDs across arms",
        "spectral_baseline": "unconstrained degree-1/degree-4 Chebyshev fit on fitting panels; relevant strong baseline only for spectral_heat target",
        "local_baseline": "fixed degree-2 centered local feature operator basis by weighted linear least squares; a target-matched synthetic baseline, not the selected physical strong baseline",
        "seed_plan": seed_plan,
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in source_files
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__},
        "limitations": "3 panels per cell; no physical gauge ensemble, inferential interval, GPU timing, or production contract; optimizer seconds exclude input/target setup and postchecks",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for name in source_files:
        shutil.copyfile(Path(__file__).with_name(name), out / f"snapshot_{name}")

    rng_inputs = {}
    panels = []
    for item in seed_plan:
        seed = item["seed"]
        rng = np.random.default_rng(seed)
        fit = [make_case(rng, item["generator"], item["target_family"],
                         item["noise_level"]) for _ in range(FIT_LINKS)]
        heldout = [make_case(rng, item["generator"], item["target_family"],
                             item["noise_level"]) for _ in range(HELDOUT_LINKS)]
        init_rng = np.random.default_rng(seed + 700000)
        W2 = init_rng.normal(size=(2, 2)) * 0.35
        v2 = init_rng.normal(size=2) * 0.18
        c2 = 0.04
        initial2 = model.pack(W2, v2, c2)
        W6 = np.vstack([W2, init_rng.normal(size=(4, 2)) * 0.35])
        v6 = np.r_[v2, init_rng.normal(size=4) * 0.18]
        initial6 = model.pack(W6, v6, c2)

        solver_runs = []
        for budget in CALL_BUDGETS:
            for method, arm in (("SLSQP", "joint_W_SLSQP"),
                                ("L-BFGS-B", "joint_W_L-BFGS-B")):
                result = solve(initial2, fit, heldout, method, budget, True)
                result["arm"] = arm
                solver_runs.append(result)
            fixed = solve(initial2, fit, heldout, "SLSQP", budget, False)
            fixed["arm"] = "fixed_W_width2_SLSQP"
            solver_runs.append(fixed)
            fixed6 = solve(initial6, fit, heldout, "SLSQP", budget, False)
            fixed6["arm"] = "fixed_W_width6_SLSQP"
            solver_runs.append(fixed6)

        spectral_baselines = ([spectral(fit, heldout, degree)
                               for degree in (1, 4)]
                              if item["target_family"] == "spectral_heat" else [])
        local_baseline = (fit_local_polynomial_baseline(fit, heldout)
                          if item["target_family"] == "local_environment" else None)
        panel = {
            **item,
            "feature_condition": _feature_condition(fit),
            "fit_target_checks": [row["target_checks"] for row in fit],
            "heldout_target_checks": [row["target_checks"] for row in heldout],
            "solver_runs": solver_runs,
            "spectral_baselines": spectral_baselines,
            "local_baseline": local_baseline,
            "denominators": {
                "planned_fit_links": FIT_LINKS,
                "valid_fit_links": FIT_LINKS,
                "failed_fit_links": 0,
                "abstained_fit_links": 0,
                "planned_heldout_links": HELDOUT_LINKS,
                "valid_heldout_links": HELDOUT_LINKS,
                "failed_heldout_links": 0,
                "abstained_heldout_links": 0,
                "planned_solver_runs": len(solver_runs),
                "returned_solver_runs": len(solver_runs),
                "budget_stops": sum(row["budget_exhausted"] for row in solver_runs),
                "solver_non_success": sum(not row["solver_success"] for row in solver_runs),
                "excluded": 0,
            },
        }
        panels.append(panel)
        for population, rows in (("fit", fit), ("heldout", heldout)):
            rng_inputs[f"{_case_key(item['generator'], item['target_family'], item['noise_level'], item['replicate'])}_{population}_links"] = np.stack([row["links"] for row in rows])
            rng_inputs[f"{_case_key(item['generator'], item['target_family'], item['noise_level'], item['replicate'])}_{population}_targets"] = np.stack([row["target"] for row in rows])
            rng_inputs[f"{_case_key(item['generator'], item['target_family'], item['noise_level'], item['replicate'])}_{population}_features"] = np.stack([row["z"] for row in rows])

    np.savez_compressed(out / "synthetic_inputs.npz", **rng_inputs)
    _build_plots(out, panels)
    paired = _group_summary(panels)
    with (out / "paired_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(paired)
    all_runs = [run for panel in panels for run in panel["solver_runs"]]
    denominator_summary = {
        "planned_panels": len(seed_plan),
        "valid_panels": len(panels),
        "failed_panels": 0,
        "abstained_panels": 0,
        "planned_fit_links": len(panels) * FIT_LINKS,
        "valid_fit_links": len(panels) * FIT_LINKS,
        "failed_fit_links": 0,
        "abstained_fit_links": 0,
        "planned_heldout_links": len(panels) * HELDOUT_LINKS,
        "valid_heldout_links": len(panels) * HELDOUT_LINKS,
        "failed_heldout_links": 0,
        "abstained_heldout_links": 0,
        "planned_solver_runs": len(all_runs),
        "returned_solver_runs": len(all_runs),
        "budget_stops": sum(row["budget_exhausted"] for row in all_runs),
        "solver_non_success": sum(not row["solver_success"] for row in all_runs),
        "excluded_samples": 0,
    }
    raw = {
        "protocol": protocol,
        "denominators": denominator_summary,
        "panels": panels,
        "paired_summary": paired,
    }
    (out / "raw.json").write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
    summary = {
        "schema": "trainable-w-synthetic-stress-summary-v1",
        "classification": "synthetic algorithm diagnostic only",
        "physical_score": None,
        "data_access": "No production gauge ensemble, sealed IDs, GPU, or QUDA data were accessed.",
        "denominators": denominator_summary,
        "cells": len(paired),
        "files": ["README.md", "protocol.json", "raw.json", "synthetic_inputs.npz",
                  "paired_summary.csv", "paired_loss_ratios.png",
                  "budget_convergence.png", "conditioning_and_baselines.png",
                  "artifact_hashes.json"],
        "source_sha256": protocol["source_sha256"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Trainable-W synthetic generator/noise/conditioning stress diagnostic",
        "",
        "**SYNTHETIC ALGORITHM DIAGNOSTIC ONLY. `physical_score=null`.** No production gauge ensemble, sealed IDs, GPU, or QUDA data were accessed. Three panels per cell support descriptive stress testing only; no confidence or significance claim is made.",
        "",
        "## Design",
        "",
        "The fixed grid crosses three independent link generators (`haar_like`, and weak-field links with `epsilon=0.08` or `0.25`), two independently defined targets (`spectral_heat`, and a gauge-covariant local-environment polynomial that varies over links), two relative operator-noise levels (`0`, `0.02`), and three seeds per cell. Each panel has two fit and two independently generated heldout links. Noise is a Hermitian link-hopping operator perturbation scaled to the requested fraction of target Frobenius norm.",
        "",
        "Every panel pairs the same fit/heldout inputs, initial parameters, and bounds across joint trainable-W SLSQP, joint trainable-W L-BFGS-B, fixed-W width-2 SLSQP, and fixed-W width-6 SLSQP at call caps 32 and 128. Width-6 fixed W matches the seven fitted scalar count of joint width-2, though it changes the feature realization. Objective/gradient calls are capped per solver fit; optimizer seconds exclude data/target construction and postchecks. All traces, status, calls, projected-gradient residuals, target diagnostics, and raw inputs are retained.",
        "",
        "For spectral targets, degree-1/degree-4 Chebyshev fixed spectral baselines are fitted on the same fit links and evaluated on heldout links. For local-environment targets, a fixed degree-2 centered local operator basis is fit by weighted linear least squares as a target-matched capacity diagnostic; it is not substituted for the selected physical strong spectral baseline. The local target's commutator with the uniform hopping operator is reported as a direct check that this family is not a function of that one global hopping matrix.",
        "",
        "## Paired results",
        "",
        "![Paired heldout loss ratios](paired_loss_ratios.png)",
        "",
        "Each point is one independent synthetic panel at the 128-call cap. Blue circles compare joint-W SLSQP with fixed-W width 2 (three fitted scalars); pink squares compare with fixed-W width 6 (seven fitted scalars, matching joint width 2). Values below zero favor joint W. The second comparison controls parameter count but changes the random feature realization. Median ticks are descriptive only. See `paired_summary.csv` for both contrasts, ranges, calls, fit times, and projected-gradient residuals; ranges are not confidence intervals.",
        "",
        "![Budget convergence by target and generator](budget_convergence.png)",
        "",
        "Panels show median heldout loss at call caps 32 and 128 for the 2%-noise cases. The target-matched fixed baseline is shown as a dashed line when it lies within the optimizer scale; otherwise its median is annotated in the panel. The budget curves compare observed finite-budget fits, not equal walltime including setup, baseline construction, or amortized training.",
        "",
        "![Conditioning and target-matched fixed baselines](conditioning_and_baselines.png)",
        "",
        "The local-basis feature design condition number varies with the synthetic link family. Target-matched baselines are expected controls: degree-4 Chebyshev is aligned with the spectral target; the fixed local polynomial oracle is aligned with the local target. Neither is a physical ranking.",
        "",
        "## Denominators and interpretation",
        "",
        f"Planned/valid/failed/abstained synthetic panels: `{denominator_summary['planned_panels']}/{denominator_summary['valid_panels']}/{denominator_summary['failed_panels']}/{denominator_summary['abstained_panels']}`. Planned/returned solver fits: `{denominator_summary['planned_solver_runs']}/{denominator_summary['returned_solver_runs']}`; budget stops: `{denominator_summary['budget_stops']}`; solver non-success: `{denominator_summary['solver_non_success']}`; excluded samples: `{denominator_summary['excluded_samples']}`. Link counts and solver counts are separate denominators in `raw.json`.",
        "",
        "A small panel count cannot determine a population effect or select a physical loss. Synthetic noise and feature conditioning only probe robustness of this implementation and objective. The selected F=2 comparisons remain trainable-W against same-budget SLSQP and against the fixed strong spectral baseline; these data do not satisfy D04 reference, D06 `r_star`, D11 operator contract, or production GPU authorization.",
        "",
        "## Rebuild",
        "",
        "Generate a fresh output directory with `python -B -m tasks.optimize_gaussian.trainable_w_stress_diagnostic --output NEW_DIRECTORY`.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (out / "README.md", out / "protocol.json", out / "raw.json",
                     out / "synthetic_inputs.npz", out / "paired_summary.csv",
                     out / "paired_loss_ratios.png", out / "budget_convergence.png",
                     out / "conditioning_and_baselines.png", out / "summary.json")
    }
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = run(args.output)
    print(json.dumps({
        "classification": summary["classification"],
        "denominators": summary["denominators"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
