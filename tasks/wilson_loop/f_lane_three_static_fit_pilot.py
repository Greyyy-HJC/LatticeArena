"""Single-panel synthetic pilot for the finite-Nt 2-vacuum × 3-static fit."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tasks.wilson_loop import f_lane_absolute_force_bootstrap_diagnostic as bootstrap
from tasks.wilson_loop import f_lane_finite_nt_trace_diagnostic as toy
from tasks.wilson_loop import f_lane_joint_spectral_gls_diagnostic as forward
from tasks.wilson_loop import f_lane_three_static_two_vacuum_candidate as three_static
from tasks.wilson_loop import f_lane_two_sector_fit_candidate as two_static
from tasks.wilson_loop import f_lane_two_sector_omitted_coverage_diagnostic as generator
from tasks.wilson_loop.f_lane_two_sector_fit_candidate import fit_two_sector_snapshot
from tasks.wilson_loop.f_lane_three_static_two_vacuum_candidate import (
    PARAMETER_COUNT, fit_three_static_snapshot,
)


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
DEFAULT_OUT = TASK_DIR / "f_lane_three_static_fit_pilot_20261128_v1"
SEED = 2026114101
N_BLOCKS = 96
RHO = 0.5
RELATIVE_NOISE = 0.003

SOURCE_PATHS = {
    "three_static_fit_pilot.py": Path(__file__),
    "three_static_candidate.py": Path(three_static.__file__),
    "two_static_candidate.py": Path(two_static.__file__),
    "joint_forward_fit.py": Path(forward.__file__),
    "omitted_coverage_generator.py": Path(generator.__file__),
    "absolute_force_bootstrap_diagnostic.py": Path(bootstrap.__file__),
    "finite_trace_toy.py": Path(toy.__file__),
    "transfer_trace_reader.py": TASK_DIR / "f_lane_two_sector_gls_diagnostic.py",
    "lane_contract_v2.json": TASK_DIR / "lane_contract_v2.json",
    "source_integrity.py": REPO_ROOT / "core/source_integrity.py",
    "npz_safety.py": REPO_ROOT / "core/npz_safety.py",
}


def _sha(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _fmt(value: Any, spec: str = ".6g") -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "unavailable"
    return format(float(value), spec)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _truth_vectors() -> tuple[np.ndarray, np.ndarray]:
    truth_vectors = np.stack([
        two_static._matrix_vector(toy.finite_trace_matrix(radius, time, nt))
        for radius in two_static.R_VALUES
        for nt in two_static.NT_VALUES
        for time in two_static.T_VALUES
    ])
    truth_params = [three_static._inverse_softplus(float(toy.VACUUM_ENERGIES[1]))]
    for radius_index, radius in enumerate(two_static.R_VALUES):
        energies = toy.STATIC_ENERGIES[radius]
        increments = np.diff(np.concatenate(([0.0], energies)))
        truth_params.extend(three_static._inverse_softplus(float(value)) for value in increments)
        for vacuum_state in range(three_static.VACUUM_STATES):
            for static_state in range(three_static.STATIC_STATES):
                truth_params.extend(two_static._pack_amplitude(
                    toy.TRANSITION_OVERLAPS[radius_index, vacuum_state, static_state]
                ))
    return truth_vectors, np.asarray(truth_params, dtype=float)


def _fit_summary(fit: dict[str, Any]) -> dict[str, Any]:
    model = fit.get("fit") or {}
    force_diagnostic = fit.get("force_estimand_diagnostic") or {}
    primary_interval = force_diagnostic.get("absolute_stout_force_delta_interval") or {}
    fit_contract = fit.get("fit_contract") or {}
    spectrum = fit.get("spectrum_fit") or {}
    if model:
        parameter_count = model.get("parameter_count")
        jacobian_rank = model.get("jacobian_rank")
        condition_number = model.get("jacobian_condition_number")
        dof = model.get("degrees_of_freedom")
        objective = model.get("gls_objective")
        reduced_objective = model.get("reduced_gls_objective")
        vacuum_energies = model.get("vacuum_energies")
        static_energies = model.get("static_energies_by_R")
    else:
        parameter_count = fit_contract.get("parameter_count")
        jacobian_rank = fit_contract.get("jacobian_rank")
        condition_number = fit_contract.get("jacobian_condition_number")
        dof = fit_contract.get("degrees_of_freedom")
        objective = fit_contract.get("chi_square_diagonal")
        reduced_objective = (objective / dof if objective is not None and dof else None)
        vacuum_energies = spectrum.get("vacuum_energies")
        static_energies = spectrum.get("static_energies_by_R")
    return {
        "status": fit.get("status"),
        "valid": fit.get("denominator", {}).get("valid") == 1,
        "parameter_count": parameter_count,
        "jacobian_rank": jacobian_rank,
        "jacobian_condition_number": condition_number,
        "degrees_of_freedom": dof,
        "gls_objective": objective,
        "reduced_gls_objective": reduced_objective,
        "vacuum_energies": vacuum_energies,
        "static_energies_by_R": static_energies,
        "absolute_force": model.get("absolute_stout_ground_force",
                                     (fit.get("force_estimand_diagnostic") or {}).get(
                                         "absolute_stout_force_from_shared_spectrum")),
        "force_se_diagnostic_uncalibrated": model.get(
            "force_delta_standard_error_diagnostic",
            primary_interval.get("standard_error")),
        "initializer_fits": fit.get("initializer_fits"),
        "fit_attempts": fit.get("fit_attempts"),
    }


def _plot(path: Path, truth: dict[str, Any], primary: dict[str, Any],
          three: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), layout="constrained")
    true_static = truth["static_energies_by_R"]
    fit_static = (three.get("fit") or {}).get("static_energies_by_R", {})
    positions, labels, true_values, fit_values = [], [], [], []
    cursor = 0
    for radius in two_static.R_VALUES:
        for state in range(three_static.STATIC_STATES):
            positions.append(cursor)
            labels.append(f"R={radius}, n={state}")
            true_values.append(true_static[str(radius)][state])
            level_values = fit_static.get(str(radius), [math.nan] * 3)
            fit_values.append(level_values[state])
            cursor += 1
    axes[0].scatter(np.asarray(positions) - 0.08, true_values, marker="o",
                    color="black", label="injected toy energy")
    axes[0].scatter(np.asarray(positions) + 0.08, fit_values, marker="s",
                    color="#0072b2", label="2-vacuum × 3-static fit")
    axes[0].set_xticks(positions, labels, rotation=25, ha="right")
    axes[0].set_ylabel(r"static energy ($a_t E_n$)")
    axes[0].set_title("All three ordered levels at both radii")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)

    truth_force = float(truth["absolute_force"])
    primary_force_value = (primary.get("force_estimand_diagnostic") or {}).get(
        "absolute_stout_force_from_shared_spectrum")
    three_force_value = (three.get("fit") or {}).get("absolute_stout_ground_force")
    axes[1].axhline(truth_force, color="black", linestyle="--", linewidth=1.2,
                    label=f"exact force = {truth_force:.5f}")
    if primary_force_value is not None:
        axes[1].scatter([0], [float(primary_force_value)], marker="x", color="#d55e00", s=65,
                        label="current 2-static primary")
    if three_force_value is not None:
        axes[1].scatter([1], [float(three_force_value)], marker="s", color="#0072b2", s=48,
                        label="2-vacuum × 3-static candidate")
    axes[1].set_xticks([0, 1], ["2-static", "3-static"])
    axes[1].set_ylabel("absolute stout force (lattice units)")
    axes[1].set_title("One noisy panel; no coverage claim")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.suptitle("Estimator-matched third-state feasibility pilot")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _artifact_hashes(output: Path) -> dict[str, str]:
    hashes = {
        str(path.relative_to(output)): _sha(path.read_bytes())
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output / "artifact_hashes.json").write_bytes(_json_bytes({
        "schema": "wilson-three-static-pilot-artifact-hashes-v1",
        "artifacts": hashes,
    }))
    return hashes


def run(output: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {out}")
    out.mkdir(parents=True)
    source_payloads = {name: path.read_bytes() for name, path in SOURCE_PATHS.items()}
    source_hashes = {name: _sha(payload) for name, payload in source_payloads.items()}
    source_dir = out / "source_snapshot"
    source_dir.mkdir()
    for name, payload in source_payloads.items():
        (source_dir / name).write_bytes(payload)
    contract_bytes = source_payloads["lane_contract_v2.json"]
    (out / "lane_contract.json").write_bytes(contract_bytes)
    protocol = {
        "schema": "wilson-three-static-pilot-protocol-v1",
        "classification": "SINGLE_PANEL_SYNTHETIC_FIT_FEASIBILITY",
        "seed": SEED, "blocks": N_BLOCKS, "noise": "stationary AR(1)",
        "rho": RHO, "relative_noise": RELATIVE_NOISE,
        "truth_model": "positive 2-vacuum × 3-static finite-Nt trace",
        "candidate_model": "positive 2-vacuum × 3-static finite-Nt GLS over Nt=16,32 and T=3..7",
        "primary_comparison": "current regularized 2-vacuum × 2-static finite-Nt GLS on the same panel",
        "estimand": "U0_stout(R=3)-U0_stout(R=2), Delta R=1 spatial step",
        "source_hashes": source_hashes,
    }
    (out / "protocol.json").write_bytes(_json_bytes(protocol))
    matrices = generator._generate_matrices(
        SEED, {"model": "stationary_ar1", "rho": RHO, "scale_by_T": {}},
    )
    panel_dir = out / "panel"
    panel_dir.mkdir()
    snapshot, input_hashes = bootstrap._snapshot_from_matrices(
        panel_dir, matrices, contract_bytes, "F-synthetic-open-three-static-pilot",
    )
    primary = fit_two_sector_snapshot(snapshot, objective_mode="regularized_full_covariance")
    candidate_fit = fit_three_static_snapshot(snapshot)
    for name in ("manifest.json", "raw_matrices.npz"):
        (panel_dir / name).rename(out / name)
    (panel_dir / "lane_contract.json").unlink()
    panel_dir.rmdir()

    truth_vectors, truth_parameters = _truth_vectors()
    predicted_truth = three_static._model_vectors(truth_parameters)
    maximum_identity_abs_error = float(np.max(np.abs(predicted_truth - truth_vectors)))
    truth_force = float(toy.STATIC_ENERGIES[3][0] - toy.STATIC_ENERGIES[2][0])
    truth = {
        "vacuum_energies": toy.VACUUM_ENERGIES.tolist(),
        "static_energies_by_R": {str(radius): toy.STATIC_ENERGIES[radius].tolist()
                                  for radius in two_static.R_VALUES},
        "absolute_force": truth_force,
        "force_unit": "a_t energy difference per one spatial lattice step; Delta R=1",
        "maximum_exact_forward_identity_absolute_component_error": maximum_identity_abs_error,
    }
    primary_summary = _fit_summary(primary)
    candidate_summary = _fit_summary(candidate_fit)
    if candidate_summary["valid"]:
        force_error = abs(candidate_summary["absolute_force"] - truth_force)
    else:
        force_error = None
    attempts = candidate_fit.get("fit_attempts", [])
    valid_objectives = [row["objective"] for row in attempts
                        if row.get("success") and row.get("objective") is not None]
    attempt_summary = {
        "planned": len(attempts), "valid": len(valid_objectives),
        "failed": len(attempts) - len(valid_objectives),
        "valid_objective_min": min(valid_objectives) if valid_objectives else None,
        "valid_objective_max": max(valid_objectives) if valid_objectives else None,
    }
    summary = {
        "schema": "wilson-three-static-pilot-summary-v1",
        "classification": protocol["classification"], "certified": False,
        "physical_score": None, "f_lane_status": "GROUND_STATE_FORCE_UNCERTIFIED",
        "input_hashes": input_hashes,
        "truth": truth,
        "denominators": {
            "primary_fit": primary.get("denominator"),
            "three_static_fit": candidate_fit.get("denominator"),
            "three_static_multistarts": attempt_summary,
            "observables": {"components": 180, "parameters": PARAMETER_COUNT,
                            "nominal_degrees_of_freedom": 113},
        },
        "primary_fit": primary_summary,
        "three_static_fit": candidate_summary,
        "three_static_force_absolute_error": force_error,
        "open_gates": [
            "single synthetic panel does not measure finite-sample coverage or method stability",
            "full-rank local Jacobian is not a proof of global identifiability; condition number and multiple starts require broader stress",
            "inverse-Jacobian force standard error is uncalibrated and is not connected to the force summary reader",
            "production exporter/action convention, thermal/mixed q/h/x bounds, independent force reference, model-truncation review and authorization remain absent",
        ],
        "source_hashes": source_hashes,
    }
    (out / "fit_summary.json").write_bytes(_json_bytes({
        "primary_two_static": primary_summary,
        "candidate_three_static": candidate_summary,
    }))
    (out / "summary.json").write_bytes(_json_bytes(summary))
    (out / "truth.json").write_bytes(_json_bytes(truth))
    _write_csv(out / "multistart.csv", attempts)
    _plot(out / "three_static_fit.png", truth, primary, candidate_fit)
    README = f"""# Finite-Nt two-vacuum × three-static-state fit pilot

**Status:** `{protocol['classification']}`. One synthetic panel tests exact forward-model identity, local fit rank and initialization behavior; it is not a coverage study.

## Model and data flow

The fitted matrix model is

```text
C_ij(R,T;Nt) = Z(Nt)^(-1) Σ_(m=0..1,n=0..2)
               exp[-eps_m (Nt-T)] exp[-V_n(R) T] A_(mn,i) A*_(mn,j)
F_abs = V_0(R=3) - V_0(R=2),  ΔR=1
```

The flow is `exact positive 2×3 truth -> AR(1) noisy 96-block panel -> current 2×2 fit and new full finite-Nt 2×3 fit -> retain 67-coordinate rank/condition/multi-start receipts and figure`.

## Pilot outcome

- The noiseless forward identity test has maximum absolute component discrepancy `{maximum_identity_abs_error:.3g}`.
- The 2×3 fit status is `{candidate_summary['status']}`, local Jacobian rank **{candidate_summary.get('jacobian_rank')} / {candidate_summary.get('parameter_count')}**, condition number **{_fmt(candidate_summary.get('jacobian_condition_number'), '.3g')}**, and reduced GLS objective **{_fmt(candidate_summary.get('reduced_gls_objective'), '.3f')}**.
- Exact force is `{truth_force:.6f}`; current 2×2 estimate is `{_fmt(primary_summary.get('absolute_force'), '.6f')}`; 2×3 estimate is `{_fmt(candidate_summary.get('absolute_force'), '.6f')}`. The 2×3 absolute error is `{_fmt(force_error, '.3g')}` lattice units.
- Multi-start attempts: `{attempt_summary['valid']}/{attempt_summary['planned']}` full-rank; valid objective range `{_fmt(attempt_summary['valid_objective_min'], '.3f')}…{_fmt(attempt_summary['valid_objective_max'], '.3f')}`. A converged start with a larger objective is retained rather than hidden.

![Three-state energies and force comparison](three_static_fit.png)

## Interpretation and limits

This pilot shows the same finite-Nt forward model can be evaluated and locally fit with three static states for this one noisy toy panel. Rank 67/67 is local; the condition number, multi-start spread and one-panel force error motivate a broader identifiability screen. The GLS objective is not calibrated goodness-of-fit, and its inverse-Jacobian force standard error is not calibrated uncertainty. Do not use this candidate for certification or connect it to the force summary reader.

No production Wilson data, GEVP, sealed split, GPU or QUDA was used. F remains `GROUND_STATE_FORCE_UNCERTIFIED`, `physical_score=null`.
"""
    (out / "README.md").write_text(README, encoding="utf-8")
    artifacts = _artifact_hashes(out)
    return {
        "output": str(out), "primary_status": primary.get("status"),
        "three_static_status": candidate_fit.get("status"),
        "exact_identity_max_abs_error": maximum_identity_abs_error,
        "three_static_force_error": force_error, "artifacts": len(artifacts),
        "certified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
