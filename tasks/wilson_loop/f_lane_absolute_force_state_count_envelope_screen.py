"""Compare 2- and 3-static-state absolute-force fits on matched Nt=32 toys.

This is a CPU synthetic model-sensitivity screen. The per-radius Nt=32
spectral fits are not the production F-lane estimator, which also models two
finite-Nt vacuum sectors across Nt=16 and 32. Results therefore cannot be used
as a confidence interval or a physical model-truncation bound.
"""
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
from tasks.wilson_loop import f_lane_two_sector_fit_candidate as candidate
from tasks.wilson_loop import f_lane_two_sector_omitted_coverage_diagnostic as generator
from tasks.wilson_loop.f_lane_joint_spectral_gls_diagnostic import fit_joint_spectrum
from tasks.wilson_loop.f_lane_two_sector_fit_candidate import (
    NT_VALUES, R_VALUES, T_VALUES, fit_two_sector_snapshot,
)


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
DEFAULT_OUT = TASK_DIR / "f_lane_absolute_force_state_count_20261128_v1"
N_PANELS = 12
SEED_BASE = 2026114000
N_BLOCKS = 96
AR1_RHO = 0.5
NOISE_FRACTION = 0.003
STATE_COUNTS = (2, 3)

SOURCE_PATHS = {
    "state_count_envelope_screen.py": Path(__file__),
    "absolute_force_bootstrap_diagnostic.py": Path(bootstrap.__file__),
    "two_sector_fit_candidate.py": Path(candidate.__file__),
    "two_sector_omitted_coverage_generator.py": Path(generator.__file__),
    "joint_forward_fit.py": Path(forward.__file__),
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _force_envelope(force_by_state_count: dict[int, float]) -> dict[str, Any]:
    """Return a point-estimate envelope, abstaining on missing/invalid models."""
    if set(force_by_state_count) != set(STATE_COUNTS):
        return {"status": "ABSTAIN_MODEL_FIT_FAILURE", "lower": None, "upper": None}
    values = [force_by_state_count[state] for state in STATE_COUNTS]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) for value in values):
        return {"status": "ABSTAIN_INVALID_MODEL_ESTIMATE", "lower": None, "upper": None}
    return {
        "status": "VALID_POINT_ESTIMATE_ENVELOPE",
        "lower": float(min(values)), "upper": float(max(values)),
        "width": float(max(values) - min(values)),
        "finite_sample_coverage_calibrated": False,
    }


def _nt32_blocks(matrices: np.ndarray, radius_index: int) -> np.ndarray:
    nt_index = NT_VALUES.index(32)
    return np.stack([
        np.concatenate([
            forward._matrix_vector(matrices[block, nt_index, radius_index, time_index])
            for time_index in range(len(T_VALUES))
        ])
        for block in range(matrices.shape[0])
    ])


def _panel_fit(panel_dir: Path, panel_index: int, seed: int,
               contract_bytes: bytes) -> dict[str, Any]:
    matrices = generator._generate_matrices(
        seed, {"model": "stationary_ar1", "rho": AR1_RHO, "scale_by_T": {}}
    )
    snapshot, hashes = bootstrap._snapshot_from_matrices(
        panel_dir, matrices, contract_bytes,
        f"F-synthetic-open-state-count-{panel_index:03d}",
    )
    primary = fit_two_sector_snapshot(
        snapshot, objective_mode="regularized_full_covariance",
    )
    primary_diag = primary.get("force_estimand_diagnostic") or {}
    primary_force = primary_diag.get("absolute_stout_force_from_shared_spectrum")
    state_fits: dict[int, dict[int, dict[str, Any]]] = {2: {}, 3: {}}
    radius_ground: dict[int, dict[int, float]] = {2: {}, 3: {}}
    for radius_index, radius in enumerate(R_VALUES):
        blocks = _nt32_blocks(matrices, radius_index)
        for state_count in STATE_COUNTS:
            fit = fit_joint_spectrum(
                blocks, time_slices=T_VALUES, n_states=state_count,
                operator_indices=(0, 1, 2),
            )
            state_fits[state_count][radius] = fit
            energies = fit.get("energies")
            if fit.get("success") and isinstance(energies, list) and energies:
                radius_ground[state_count][radius] = float(energies[0])

    forces: dict[int, float] = {}
    for state_count in STATE_COUNTS:
        if set(radius_ground[state_count]) == set(R_VALUES):
            forces[state_count] = (
                radius_ground[state_count][R_VALUES[1]]
                - radius_ground[state_count][R_VALUES[0]]
            )
    envelope = _force_envelope(forces)
    truth = float(toy.STATIC_ENERGIES[3][0] - toy.STATIC_ENERGIES[2][0])
    fit_quality = {
        str(state_count): {
            str(radius): {
                "success": bool(fit.get("success")),
                "reduced_chi_square": fit.get("reduced_chi_square"),
                "rank": fit.get("rank"),
                "parameters": fit.get("parameter_count"),
                "condition_number": fit.get("jacobian_condition_number"),
            }
            for radius, fit in state_fits[state_count].items()
        }
        for state_count in STATE_COUNTS
    }
    primary_fit_contract = primary.get("fit_contract") or {}
    result = {
        "panel": panel_index, "seed": seed,
        "input_matrix_sha256": hashes["raw_matrices"],
        "primary_fit_status": primary.get("status"),
        "primary_fit_valid": primary.get("denominator", {}).get("valid") == 1,
        "primary_force": primary_force,
        "primary_reduced_gls_objective_per_dof": (
            primary_fit_contract.get("chi_square_diagonal", None)
            / primary_fit_contract.get("degrees_of_freedom", 1)
            if primary_fit_contract.get("degrees_of_freedom") else None
        ),
        "force_2state_nt32": forces.get(2),
        "force_3state_nt32": forces.get(3),
        "state_count_envelope_status": envelope["status"],
        "state_count_envelope_lower": envelope["lower"],
        "state_count_envelope_upper": envelope["upper"],
        "state_count_envelope_width": envelope.get("width"),
        "state_count_envelope_contains_truth": (
            envelope["status"] == "VALID_POINT_ESTIMATE_ENVELOPE"
            and envelope["lower"] <= truth <= envelope["upper"]
        ),
        "truth": truth,
        "fit_quality_json": json.dumps(fit_quality, sort_keys=True),
        "model_scope": "per-R Nt32 positive spectrum; not full finite-Nt two-vacuum primary",
    }
    return result


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    panels = [row for row in rows if row["state_count_envelope_status"] == "VALID_POINT_ESTIMATE_ENVELOPE"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), layout="constrained")
    truth = float(rows[0]["truth"])
    axes[0].axhline(truth, color="black", linestyle="--", linewidth=1.2,
                    label=f"exact toy force = {truth:.3f}")
    for row in panels:
        x = int(row["panel"])
        low, high = row["state_count_envelope_lower"], row["state_count_envelope_upper"]
        axes[0].vlines(x, low, high, color="#7b61a8", alpha=0.72, linewidth=3,
                       label="2/3-state Nt=32 point envelope" if x == int(panels[0]["panel"]) else None)
        axes[0].scatter(x, row["force_2state_nt32"], marker="o", color="#d55e00",
                        s=28, label="2-state Nt=32" if x == int(panels[0]["panel"]) else None)
        axes[0].scatter(x, row["force_3state_nt32"], marker="s", color="#0072b2",
                        s=28, label="3-state Nt=32" if x == int(panels[0]["panel"]) else None)
        if row["primary_force"] is not None:
            axes[0].scatter(x, row["primary_force"], marker="x", color="#333333",
                            s=40, label="full finite-Nt 2×2 primary" if x == int(panels[0]["panel"]) else None)
    axes[0].set_xlabel("fresh synthetic panel")
    axes[0].set_ylabel("force estimate (lattice units)")
    axes[0].set_title("Model point estimates and their state-count span")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    quality = {2: [], 3: []}
    for row in rows:
        data = json.loads(row["fit_quality_json"])
        for state_count in STATE_COUNTS:
            values = [data[str(state_count)][str(radius)]["reduced_chi_square"]
                      for radius in R_VALUES
                      if data[str(state_count)][str(radius)]["reduced_chi_square"] is not None]
            if values:
                quality[state_count].append(float(np.mean(values)))
    for state_count, color, marker in ((2, "#d55e00", "o"), (3, "#0072b2", "s")):
        values = np.asarray(quality[state_count], dtype=float)
        if values.size:
            axes[1].scatter(np.full(values.size, state_count), values, color=color,
                            alpha=0.68, marker=marker, label=f"{state_count}-state fits")
            axes[1].plot([state_count - 0.08, state_count + 0.08],
                         [np.median(values)] * 2, color=color, linewidth=2)
    axes[1].set_yscale("log")
    axes[1].set_xticks([2, 3])
    axes[1].set_xlabel("static-state count in Nt=32 diagnostic model")
    axes[1].set_ylabel("mean reduced chi-square across R=2,3")
    axes[1].set_title("Nt=32 spectral fit residual diagnostic")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.suptitle("Synthetic state-count sensitivity · not interval calibration")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _artifact_hashes(output: Path) -> dict[str, str]:
    values = {
        str(path.relative_to(output)): _sha(path.read_bytes())
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output / "artifact_hashes.json").write_bytes(_json_bytes({
        "schema": "wilson-absolute-force-state-count-artifact-hashes-v1",
        "artifacts": values,
    }))
    return values


def run(output: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {out}")
    out.mkdir(parents=True)
    source_bytes = {name: path.read_bytes() for name, path in SOURCE_PATHS.items()}
    source_hashes = {name: _sha(payload) for name, payload in source_bytes.items()}
    source_dir = out / "source_snapshot"
    source_dir.mkdir()
    for name, payload in source_bytes.items():
        (source_dir / name).write_bytes(payload)
    contract_bytes = source_bytes["lane_contract_v2.json"]
    (out / "lane_contract.json").write_bytes(contract_bytes)
    protocol = {
        "schema": "wilson-absolute-force-state-count-screen-protocol-v1",
        "classification": "SYNTHETIC_MODEL_SENSITIVITY_SCREEN_ONLY",
        "panels": N_PANELS, "blocks_per_panel": N_BLOCKS,
        "seed_rule": f"{SEED_BASE} + panel_index",
        "noise": {"process": "stationary AR(1)", "rho": AR1_RHO,
                  "relative_scale": NOISE_FRACTION},
        "truth": {"force": float(toy.STATIC_ENERGIES[3][0] - toy.STATIC_ENERGIES[2][0]),
                  "unit": "a_t energy difference per spatial lattice step; Delta R=1"},
        "primary_estimator": "regularized full-covariance 2-vacuum x 2-static fit over Nt={16,32}, T=3..7",
        "diagnostic_estimators": {
            "2state": "independent per-R 2-static-state positive-spectrum GLS on Nt=32, T=3..7",
            "3state": "independent per-R 3-static-state positive-spectrum GLS on Nt=32, T=3..7",
        },
        "envelope": "min/max of the two Nt=32 model point estimates; not a confidence interval",
        "scope_warning": "diagnostic models do not match the primary finite-Nt two-vacuum estimator",
        "failure_rule": "if either state-count fit is missing/rank-deficient at either R, abstain the panel envelope",
        "source_hashes": source_hashes,
    }
    (out / "protocol.json").write_bytes(_json_bytes(protocol))
    rows = []
    for panel_index in range(N_PANELS):
        seed = SEED_BASE + panel_index
        panel_dir = out / "panels" / f"panel_{panel_index:03d}"
        panel_dir.mkdir(parents=True)
        rows.append(_panel_fit(panel_dir, panel_index, seed, contract_bytes))
        print(f"completed state-count panel {panel_index + 1}/{N_PANELS}", flush=True)
    # Every panel manifest points at the same contract digest; keep one shared
    # immutable copy in the report instead of twelve byte-identical duplicates.
    for panel_index in range(N_PANELS):
        (out / "panels" / f"panel_{panel_index:03d}" / "lane_contract.json").unlink()
    _write_csv(out / "panel_results.csv", rows)
    _plot(rows, out / "state_count_envelope.png")
    valid = [row for row in rows if row["state_count_envelope_status"] == "VALID_POINT_ESTIMATE_ENVELOPE"]
    contained = sum(bool(row["state_count_envelope_contains_truth"]) for row in valid)
    state_quality = {}
    for state_count in STATE_COUNTS:
        valid_values = []
        valid_radius_fits = 0
        for row in rows:
            per_radius = json.loads(row["fit_quality_json"])[str(state_count)]
            radius_values = []
            for radius in R_VALUES:
                fit_quality = per_radius[str(radius)]
                if fit_quality["success"]:
                    valid_radius_fits += 1
                    radius_values.append(fit_quality["reduced_chi_square"])
            if len(radius_values) == len(R_VALUES):
                valid_values.append(float(np.mean(radius_values)))
        state_quality[str(state_count)] = {
            "valid_radius_fits": valid_radius_fits,
            "planned_radius_fits": N_PANELS * len(R_VALUES),
            "valid_two_radius_panels": len(valid_values),
            "median_mean_reduced_chi_square": (
                float(np.median(valid_values)) if valid_values else None
            ),
        }
    summary = {
        "schema": "wilson-absolute-force-state-count-summary-v1",
        "classification": protocol["classification"], "certified": False,
        "physical_score": None, "f_lane_status": "GROUND_STATE_FORCE_UNCERTIFIED",
        "denominator": {"planned": N_PANELS, "valid": len(valid),
                        "failed": N_PANELS - len(valid), "abstained": 0},
        "state_count_envelope": {
            "successes_containing_truth": contained,
            "valid_panel_count": len(valid),
            "observed_fraction": contained / len(valid) if valid else None,
            "median_width": float(np.median([
                row["state_count_envelope_width"] for row in valid
            ])) if valid else None,
            "interpretation": "descriptive span of two model point estimates; not interval coverage",
        },
        "truth": {
            "force": float(toy.STATIC_ENERGIES[3][0] - toy.STATIC_ENERGIES[2][0]),
            "unit": "a_t energy difference per spatial lattice step; Delta R=1",
        },
        "fit_quality": state_quality,
        "source_hashes": source_hashes,
        "open_gates": [
            "Nt=32 per-R 2/3-state diagnostic does not match the full finite-Nt two-vacuum F primary estimator",
            "point-estimate span has no within-model confidence construction and is not a model-truncation bound",
            "only one omitted-static-state positive-spectrum generator and one AR(1) dependence/noise stratum were screened",
            "production exporter/action convention, thermal/mixed q/h/x bounds, independent reference-force interval, review and authorization remain absent",
        ],
    }
    (out / "summary.json").write_bytes(_json_bytes(summary))
    README = f"""# Absolute stout force: state-count sensitivity screen

**Status:** `{protocol['classification']}`. This report compares point estimates from two Nt=32 spectral models; its span is not a confidence interval.

## Protocol and outcome

- {N_PANELS} fresh panels; {N_BLOCKS} ordered blocks/panel; stationary AR(1) `rho={AR1_RHO}`; relative noise `{NOISE_FRACTION}`; seeds `{SEED_BASE} + panel_index`.
- The primary reference fit is the current regularized 2-vacuum × 2-static finite-Nt GLS over `Nt={{16,32}}`, `T=3…7`. The sensitivity fits separately model each radius using matched `Nt=32`, `T=3…7`, with two or three static states.
- The plotted state-count span is simply `[min(F_2state,F_3state), max(F_2state,F_3state)]`. Any failed/rank-deficient state-count fit abstains that panel envelope.
- Exact synthetic primary force: `{summary['truth']['force']}` lattice units. The two-point-estimate span contains it in `{contained}/{len(valid)}` valid panels; this fraction is descriptive and is not interval coverage. Median span width is `{summary['state_count_envelope']['median_width']}` lattice units.

![State-count point-estimate envelope and residual diagnostics](state_count_envelope.png)

## Data flow

```text
fresh positive-spectrum 2-vacuum × 3-static matrices
  -> current full finite-Nt 2×2 primary fit
  -> matched Nt=32, per-R 2-state and 3-state GLS fits
  -> force point-estimate span + fit residual diagnostics
  -> retain every panel, failures, raw matrices, one shared lane contract, source/artifact hashes
```

## Interpretation and limits

The matched-Nt 2/3-state fits are not the same estimator as the full finite-Nt two-vacuum primary. Their point span has no within-model uncertainty construction, and this one generator/noise stratum cannot establish coverage. A useful next implementation must either extend the finite-Nt primary to a controlled third static state or build an estimator-matched constrained profile, then compare statistical intervals and truncation envelopes separately across the prespecified difficult strata.

No production Wilson matrices, GEVP data, sealed split, GPU or QUDA were accessed. The physical score remains null; F remains `GROUND_STATE_FORCE_UNCERTIFIED`.
"""
    (out / "README.md").write_text(README, encoding="utf-8")
    artifact_hashes = _artifact_hashes(out)
    return {
        "output": str(out), "panels": N_PANELS,
        "valid_envelopes": len(valid), "contains_truth": contained,
        "artifacts": len(artifact_hashes), "certified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
