"""Exact positive-spectrum witness for finite-time ground-force non-identifiability.

Two positive transfer spectra reproduce the same 3x3 correlator matrices on
the current F-lane time slices T=3..7, but have different ground-state forces.
This deterministic CPU witness is not a production measurement.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shutil
from pathlib import Path
from typing import Any

import numpy as np


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
DEFAULT_OUT = TASK_DIR / "f_lane_finite_time_ground_force_nonidentifiability_20261128_v1"
R_VALUES = (2, 3)
TIME_SLICES = (3, 4, 5, 6, 7)
EFFECTIVE_ENERGY_TIMES = (3, 4, 5, 6)
BASE_NODES = np.asarray([0.15, 0.32, 0.51, 0.70, 0.90], dtype=float)
BASE_WEIGHTS = np.asarray([0.90, 0.70, 0.50, 0.35, 0.20], dtype=float)
OPERATOR_WEIGHTS = np.asarray([1.0, 0.8, 0.6], dtype=float)
HIDDEN_AMPLITUDE = 1.0e-6
HIDDEN_GROUND_ENERGY = {2: 0.03, 3: 0.08}  # a_t * E, illustrative


def _moment_matrix(nodes: np.ndarray, times: tuple[int, ...]) -> np.ndarray:
    return np.asarray([[float(node) ** time for node in nodes]
                       for time in times], dtype=float)


def _positivity_only_force_interval(
    effective_energies: np.ndarray, *, radius_step: float,
) -> tuple[float, float, list[float]]:
    """Broad force interval from positive spectra and an explicit E0 >= 0 prior.

    Positive spectral weights imply E_eff(R,T) >= E0(R), so the smallest
    measured effective energy is an upper bound on E0. The lower bound zero is
    an additional energy-convention assumption and is not inferred here.
    """
    values = np.asarray(effective_energies, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(R_VALUES) or values.shape[1] < 1:
        raise ValueError("effective_energies must have one nonempty row per radius")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("effective_energies must be finite and nonnegative")
    if not math.isfinite(radius_step) or radius_step <= 0.0:
        raise ValueError("radius_step must be positive and finite")
    upper_by_radius = np.min(values, axis=1)
    return (
        float(-upper_by_radius[0] / radius_step),
        float(upper_by_radius[1] / radius_step),
        upper_by_radius.tolist(),
    )


def _build_witness() -> dict[str, Any]:
    """Return two spectra and their finite-time matrix correlators."""
    if len(BASE_NODES) != len(TIME_SLICES):
        raise ValueError("the square moment construction needs one base node per time slice")
    if not np.all(np.diff(BASE_NODES) > 0.0) or not np.all((BASE_NODES > 0) & (BASE_NODES < 1)):
        raise ValueError("base transfer eigenvalues must be strictly ordered in (0,1)")
    if not np.all(BASE_WEIGHTS > 0.0) or not np.all(OPERATOR_WEIGHTS > 0.0):
        raise ValueError("all base spectral and operator weights must be positive")
    if not math.isfinite(HIDDEN_AMPLITUDE) or HIDDEN_AMPLITUDE <= 0.0:
        raise ValueError("hidden-state amplitude must be positive and finite")

    moment = _moment_matrix(BASE_NODES, TIME_SLICES)
    base_scalar = moment @ BASE_WEIGHTS
    base_matrices = np.asarray([
        [value * np.diag(OPERATOR_WEIGHTS) for value in base_scalar]
        for _radius in R_VALUES
    ])

    alt_weights_by_radius = []
    hidden_nodes = []
    alt_scalar_by_radius = []
    for radius in R_VALUES:
        energy = float(HIDDEN_GROUND_ENERGY[radius])
        hidden_node = math.exp(-energy)
        if not BASE_NODES[-1] < hidden_node < 1.0:
            raise ValueError("hidden state must lie below the base ground energy and above zero")
        hidden_moments = np.asarray([hidden_node**time for time in TIME_SLICES])
        # Solve V w_alt + epsilon v_hidden = V w_base, exactly at measured T.
        alt_weights = np.linalg.solve(
            moment, base_scalar - HIDDEN_AMPLITUDE * hidden_moments,
        )
        if not np.all(np.isfinite(alt_weights)) or not np.all(alt_weights > 0.0):
            raise ValueError("the selected witness does not preserve positive base weights")
        alt_scalar = _moment_matrix(BASE_NODES, TIME_SLICES) @ alt_weights
        alt_scalar += HIDDEN_AMPLITUDE * hidden_moments
        alt_weights_by_radius.append(alt_weights)
        hidden_nodes.append(hidden_node)
        alt_scalar_by_radius.append(alt_scalar)
    alt_matrices = np.asarray([
        [value * np.diag(OPERATOR_WEIGHTS) for value in values]
        for values in alt_scalar_by_radius
    ])

    residual = alt_matrices - base_matrices
    relative_residual = float(np.max(np.abs(residual)) / np.max(np.abs(base_matrices)))
    base_ground_energy = -math.log(float(BASE_NODES[-1]))
    base_ground_energies = np.asarray([base_ground_energy] * len(R_VALUES))
    alt_ground_energies = np.asarray([HIDDEN_GROUND_ENERGY[r] for r in R_VALUES], dtype=float)
    radius_step = float(R_VALUES[1] - R_VALUES[0])
    base_ground_force = float(np.diff(base_ground_energies)[0] / radius_step)
    alt_ground_force = float(np.diff(alt_ground_energies)[0] / radius_step)

    # E_eff(R,T)=log[C(R,T)/C(R,T+1)] for the common scalar diagonal factor.
    time_to_index = {time: index for index, time in enumerate(TIME_SLICES)}
    base_effective = np.asarray([
        [math.log(base_scalar[time_to_index[t]] / base_scalar[time_to_index[t + 1]])
         for t in EFFECTIVE_ENERGY_TIMES]
        for _radius in R_VALUES
    ])
    alt_effective = np.asarray([
        [math.log(values[time_to_index[t]] / values[time_to_index[t + 1]])
         for t in EFFECTIVE_ENERGY_TIMES]
        for values in alt_scalar_by_radius
    ])
    finite_time_force = np.mean(alt_effective[1] - alt_effective[0])
    positivity_force_lower, positivity_force_upper, positivity_energy_upper = (
        _positivity_only_force_interval(base_effective, radius_step=radius_step)
    )

    return {
        "moment_matrix": moment,
        "base_scalar": base_scalar,
        "alt_scalar_by_radius": np.asarray(alt_scalar_by_radius),
        "base_matrices": base_matrices,
        "alt_matrices": alt_matrices,
        "alt_weights_by_radius": np.asarray(alt_weights_by_radius),
        "hidden_nodes": np.asarray(hidden_nodes),
        "base_ground_energies": base_ground_energies,
        "alt_ground_energies": alt_ground_energies,
        "base_ground_force": base_ground_force,
        "alt_ground_force": alt_ground_force,
        "base_effective_energies": base_effective,
        "alt_effective_energies": alt_effective,
        "finite_time_force": float(finite_time_force),
        "positivity_only_force_interval": [
            positivity_force_lower, positivity_force_upper,
        ],
        "positivity_only_ground_energy_upper_by_radius": positivity_energy_upper,
        "max_matrix_abs_residual": float(np.max(np.abs(residual))),
        "max_matrix_relative_residual": relative_residual,
        "minimum_base_weight_after_adjustment": float(np.min(alt_weights_by_radius)),
        "minimum_matrix_eigenvalue": float(min(
            np.linalg.eigvalsh(matrix)[0]
            for matrix in np.concatenate((base_matrices, alt_matrices), axis=0).reshape(-1, 3, 3)
        )),
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plot(result: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.8), constrained_layout=True)
    ax = axes[0]
    times = np.asarray(TIME_SLICES)
    styles = [("#287a63", "-", "R=2 base"),
              ("#287a63", "--", "R=2 + hidden"),
              ("#7b4a94", "-", "R=3 base"),
              ("#7b4a94", "--", "R=3 + hidden")]
    scalar_series = [result["base_scalar"], result["alt_scalar_by_radius"][0],
                     result["base_scalar"], result["alt_scalar_by_radius"][1]]
    for values, (color, linestyle, label) in zip(scalar_series, styles):
        ax.semilogy(times, values, marker="o", color=color,
                    linestyle=linestyle, linewidth=1.4, label=label)
    ax.set(xlabel="measured temporal slice T", ylabel=r"$C_{11}(T)$ (toy units)",
           title="Correlator values coincide at T=3…7")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[1]
    eff_times = np.asarray(EFFECTIVE_ENERGY_TIMES)
    for radius_index, (radius, color) in enumerate(zip(R_VALUES, ("#287a63", "#7b4a94"))):
        ax.plot(eff_times, result["base_effective_energies"][radius_index],
                marker="o", color=color, linestyle="-", label=f"R={radius} base")
        ax.plot(eff_times, result["alt_effective_energies"][radius_index],
                marker="x", color=color, linestyle="--", label=f"R={radius} + hidden")
    ax.set(xlabel="effective-energy start time T", ylabel=r"$a_t E_{eff}(T)$",
           title="All measured effective energies coincide")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[2]
    labels = ["base ground\nforce", "hidden ground\nforce", "from measured\nT=3…7"]
    values = [result["base_ground_force"], result["alt_ground_force"],
              result["finite_time_force"]]
    colors = ["#287a63", "#c56c32", "#5579ad"]
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.axhline(0.0, color="#555555", linewidth=0.9)
    ax.set(ylabel=r"lattice-coordinate slope $\Delta(a_t E)/\Delta R$",
           title="Ground force differs despite identical inputs")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.3f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4 if value >= 0 else -12), textcoords="offset points",
                    ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    ax = axes[3]
    lower, upper = result["positivity_only_force_interval"]
    midpoint = 0.5 * (lower + upper)
    halfwidth = 0.5 * (upper - lower)
    ax.errorbar([midpoint], [1.0], xerr=[halfwidth], fmt="o", capsize=6,
                color="#333333")
    witness_forces = [result["base_ground_force"], result["alt_ground_force"],
                      result["finite_time_force"]]
    ax.scatter(witness_forces, [0.0, 0.0, 0.0], s=[45, 55, 45],
               c=["#287a63", "#c56c32", "#5579ad"], zorder=3)
    ax.axvline(0.0, color="#777777", linewidth=0.8)
    ax.set_xlim(lower - 0.03, upper + 0.03)
    ax.set(yticks=[0.0, 1.0], yticklabels=["base / hidden / measured", "broad interval"],
           xlabel=r"lattice-coordinate slope $\Delta(a_t E)/\Delta R$",
           title="Positivity gives only a broad envelope")
    ax.grid(axis="x", alpha=0.25)
    ax.annotate(f"{lower:.3f}", (lower, 1.0), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=8)
    ax.annotate(f"{upper:.3f}", (upper, 1.0), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=8)
    fig.suptitle("Finite-time positive spectra do not identify ground-state force")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_report(output: str | Path = DEFAULT_OUT) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    result = _build_witness()
    np.savez_compressed(
        out / "spectra_and_correlators.npz",
        times=np.asarray(TIME_SLICES, dtype=np.int64),
        radii=np.asarray(R_VALUES, dtype=np.int64),
        base_nodes=BASE_NODES,
        base_weights=BASE_WEIGHTS,
        operator_weights=OPERATOR_WEIGHTS,
        hidden_amplitude=np.asarray(HIDDEN_AMPLITUDE),
        hidden_nodes=result["hidden_nodes"],
        hidden_ground_energies=np.asarray([HIDDEN_GROUND_ENERGY[r] for r in R_VALUES]),
        adjusted_base_weights=result["alt_weights_by_radius"],
        base_matrices=result["base_matrices"],
        hidden_matrices=result["alt_matrices"],
    )
    with (out / "finite_time_observables.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        fields = ["radius", "time", "base_C11", "hidden_C11", "absolute_difference",
                  "base_effective_energy", "hidden_effective_energy"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for radius_index, radius in enumerate(R_VALUES):
            for time_index, time in enumerate(TIME_SLICES):
                base_value = float(result["base_scalar"][time_index])
                hidden_value = float(result["alt_scalar_by_radius"][radius_index, time_index])
                effective_index = time_index if time in EFFECTIVE_ENERGY_TIMES else None
                writer.writerow({
                    "radius": radius,
                    "time": time,
                    "base_C11": base_value,
                    "hidden_C11": hidden_value,
                    "absolute_difference": abs(base_value - hidden_value),
                    "base_effective_energy": (
                        float(result["base_effective_energies"][radius_index, effective_index])
                        if effective_index is not None else ""
                    ),
                    "hidden_effective_energy": (
                        float(result["alt_effective_energies"][radius_index, effective_index])
                        if effective_index is not None else ""
                    ),
                })

    parameters = {
        "radii": list(R_VALUES),
        "measured_time_slices": list(TIME_SLICES),
        "base_transfer_eigenvalues_x_equals_exp_minus_a_t_E": BASE_NODES.tolist(),
        "base_positive_spectral_weights": BASE_WEIGHTS.tolist(),
        "operator_diagonal_weights": OPERATOR_WEIGHTS.tolist(),
        "hidden_state_amplitude": HIDDEN_AMPLITUDE,
        "hidden_ground_energy_a_t_E_by_radius": {
            str(radius): HIDDEN_GROUND_ENERGY[radius] for radius in R_VALUES
        },
        "adjusted_old_state_weights_by_radius": result["alt_weights_by_radius"].tolist(),
        "base_ground_energy_a_t_E_by_radius": result["base_ground_energies"].tolist(),
        "hidden_model_ground_energy_a_t_E_by_radius": result["alt_ground_energies"].tolist(),
        "base_ground_force_lattice_coordinate": result["base_ground_force"],
        "hidden_ground_force_lattice_coordinate": result["alt_ground_force"],
        "finite_time_force_from_measured_effective_energies": result["finite_time_force"],
        "positivity_only_force_interval_assuming_nonnegative_ground_energies": result[
            "positivity_only_force_interval"
        ],
        "positivity_only_ground_energy_upper_by_radius": result[
            "positivity_only_ground_energy_upper_by_radius"
        ],
        "max_matrix_absolute_residual": result["max_matrix_abs_residual"],
        "max_matrix_relative_residual": result["max_matrix_relative_residual"],
        "minimum_adjusted_old_weight": result["minimum_base_weight_after_adjustment"],
        "minimum_correlator_matrix_eigenvalue": result["minimum_matrix_eigenvalue"],
        "classification": "DETERMINISTIC_POSITIVE_SPECTRUM_NONIDENTIFIABILITY_WITNESS_NOT_PHYSICAL_FORCE",
        "physical_score": None,
        "certified": False,
        "data_access": "no measurement or sealed data; finite-dimensional analytic toy only",
    }
    (out / "witness_parameters.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )

    source_snapshot = out / "source_snapshot"
    source_snapshot.mkdir()
    source_copy = source_snapshot / "f_lane_finite_time_ground_force_nonidentifiability.py"
    shutil.copy2(Path(__file__).resolve(), source_copy)
    protocol = {
        "schema": "wilson-finite-time-ground-force-nonidentifiability-v1",
        "classification": parameters["classification"],
        "construction": {
            "positive_spectral_form": "C_ij(t)=sum_n Z_i^(n) Z_j^(n)* exp(-a_t E_n t)",
            "base_state_count": len(BASE_NODES) * len(OPERATOR_WEIGHTS),
            "hidden_model_state_count": (len(BASE_NODES) + 1) * len(OPERATOR_WEIGHTS),
            "base_nodes": BASE_NODES.tolist(),
            "base_weights": BASE_WEIGHTS.tolist(),
            "hidden_amplitude": HIDDEN_AMPLITUDE,
            "hidden_ground_energy_a_t_E_by_radius": {
                str(radius): HIDDEN_GROUND_ENERGY[radius] for radius in R_VALUES
            },
            "operator_overlap_gram": "diag(1.0,0.8,0.6), repeated for each degenerate spectral level",
        },
        "observation_contract": {
            "radii": list(R_VALUES),
            "time_slices": list(TIME_SLICES),
            "matrix_shape": [len(R_VALUES), len(TIME_SLICES), 3, 3],
            "current_f_lane_source": "f_lane_two_sector_fit_candidate.py: T_VALUES=(3,4,5,6,7); synthetic witness only",
            "match_tolerance_relative": 1.0e-14,
        },
        "force_definition": "[a_t E0(R=3)-a_t E0(R=2)]/(3-2), lattice-coordinate slope",
        "inference_claim": "finite-time moments do not identify a unique ground-state force or a tight project-specific model-truncation budget without additional spectral restrictions; positivity alone may still imply a broad envelope under an explicit energy convention",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "source_sha256": {
            "source_snapshot/f_lane_finite_time_ground_force_nonidentifiability.py": _sha(source_copy),
        },
        "sealed_data_access": False,
        "gpu_quda": False,
        "physical_score": None,
        "certified": False,
    }
    (out / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    _plot(result, out / "finite_time_force_nonidentifiability.png")
    report = {
        "schema": protocol["schema"],
        "classification": protocol["classification"],
        "finite_matrix_points": len(R_VALUES) * len(TIME_SLICES),
        "planned": len(R_VALUES) * len(TIME_SLICES),
        "valid": len(R_VALUES) * len(TIME_SLICES),
        "failed": 0,
        "abstained": 0,
        "base_ground_force": result["base_ground_force"],
        "hidden_ground_force": result["alt_ground_force"],
        "finite_time_force": result["finite_time_force"],
        "positivity_only_force_interval": result["positivity_only_force_interval"],
        "max_matrix_relative_residual": result["max_matrix_relative_residual"],
        "minimum_adjusted_old_weight": result["minimum_base_weight_after_adjustment"],
        "physical_score": None,
        "certified": False,
    }
    (out / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    readme = f"""# Finite-time positive-spectrum non-identifiability witness

**Classification:** `{protocol['classification']}`. This is a deterministic analytic toy; no measured or sealed data, GPU, or QUDA input was used. `physical_score=null`.

## Frozen construction

The measured positive-time slices are `T={list(TIME_SLICES)}` at radii `R={list(R_VALUES)}`, matching the current two-sector F-lane time slices. For each operator direction, the base correlator is a positive spectral sum with transfer eigenvalues `x_n=exp(-a_t E_n)={BASE_NODES.tolist()}` and positive weights `{BASE_WEIGHTS.tolist()}`. The three operator overlap directions are orthogonal with positive Gram weights `{OPERATOR_WEIGHTS.tolist()}`, making every 3×3 correlator matrix positive definite.

For each radius `R`, the alternative adds a previously absent positive-amplitude ground state with amplitude `epsilon={HIDDEN_AMPLITUDE:g}` and `a_t E_0(R)={HIDDEN_GROUND_ENERGY[2]:.3f}` at R=2, `{HIDDEN_GROUND_ENERGY[3]:.3f}` at R=3. Its transfer eigenvalue is larger than every base eigenvalue. The five old-state weights are adjusted by solving the square moment equations on exactly `T=3…7`; all adjusted weights remain positive (minimum `{result['minimum_base_weight_after_adjustment']:.6f}`).

Because `V[t,n]=x_n**t` is invertible for these distinct positive nodes and consecutive time slices, the alternative weights satisfy `V w_alt + epsilon v_hidden = V w_base`. Thus the complete 3×3 matrices at all ten `(R,T)` points agree to relative residual `{result['max_matrix_relative_residual']:.3e}` (the exact-real construction agrees identically). Yet the base ground energies are both `{result['base_ground_energies'][0]:.6f}` and its R=2→3 ground force is `{result['base_ground_force']:.3f}`; the alternative ground energies are `{result['alt_ground_energies'].tolist()}` and force is `{result['alt_ground_force']:.3f}`. The finite-time effective-energy force formed from the measured slices is `{result['finite_time_force']:.3e}` for both.

![Finite-time correlators versus differing ground forces](finite_time_force_nonidentifiability.png)

## Text diagnostic

```text
positive base spectrum -> 3x3 C_ij(R,T), R=2,3 and T=3..7
                         | exact five-time moment equations
add tiny positive lower-energy state at each R
                         | solve for adjusted old weights (all stay positive)
same ten observed matrices / same effective energies
                         | different R-dependence of E0
ground force 0.000 vs 0.050 while observed finite-time force remains 0.000
```

## Interpretation and limits

This is a constructive non-identifiability result for the chosen finite set of positive-time correlator matrices. More generally, for a finite moment vector in the interior of the positive moment cone, a sufficiently small new positive atom can be added at a transfer eigenvalue above the assumed largest node, while nearby positive weights on the old nodes preserve every observed moment exactly. The hidden transfer eigenvalue can approach 1 while its positive amplitude shrinks, so these finite observations do not establish a positive lower gap or an overlap floor. Positivity together with a declared nonnegative-energy convention can still imply a broad envelope; this witness shows that the finite-time data alone do not identify a unique ground force or establish the tight project-specific model-truncation budget required by the F-lane contract. An independently justified overlap/gap/support restriction or equivalent receipt is needed to narrow that envelope. With noisy data the indistinguishable set only grows.

Under the **additional convention** that every ground energy is nonnegative, positive spectral weights imply `0 <= E0(R) <= min_T E_eff(R,T)`. Applied to this toy, that gives only `ground force in [{result['positivity_only_force_interval'][0]:.6f}, {result['positivity_only_force_interval'][1]:.6f}]`, a width of `{result['positivity_only_force_interval'][1] - result['positivity_only_force_interval'][0]:.6f}` in lattice-coordinate units. This contains both witnesses, but it is broad; the nonnegative-energy convention itself must be checked for the production energy normalization, especially if an additive static-energy subtraction is used.

The witness does **not** say the hidden spectrum is realized in the project, quantify the actual force error, include finite-`N_t` winding/thermal or mixed-sector `h/x`, identify the production exporter convention, or establish the size of any valid spectral prior. The positivity-only range therefore does not set the Wilson force budget. It instead shows why an independently sourced spectral restriction is required before the F-lane model-truncation component can be made tight enough for the selected budget. Keep F=`GROUND_STATE_FORCE_UNCERTIFIED`, both Wilson lanes `NOT_READY/UNCERTIFIED`, `physical_score=null`, and sealed inputs closed.

## Reproduction and evidence

Run from repository root with `python -m tasks.wilson_loop.f_lane_finite_time_ground_force_nonidentifiability`. The output directory is no-clobber. `spectra_and_correlators.npz` retains the base and alternative 3×3 matrices; `finite_time_observables.csv` retains every planned radius-time row; `witness_parameters.json`, `protocol.json`, `summary.json`, source snapshot and artifact hashes record the complete construction. The deterministic denominator is **10/10/0/0** `(R,T)` matrix points (planned/valid/failed/abstained) for each spectrum. These are toy algebra points, not Monte Carlo panels or physical observations.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    artifact_names = [
        "protocol.json", "witness_parameters.json", "summary.json",
        "finite_time_observables.csv", "spectra_and_correlators.npz",
        "finite_time_force_nonidentifiability.png", "README.md",
        "source_snapshot/f_lane_finite_time_ground_force_nonidentifiability.py",
    ]
    (out / "artifact_hashes.json").write_text(
        json.dumps({name: _sha(out / name) for name in artifact_names},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return report


def main() -> None:
    print(json.dumps(build_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
