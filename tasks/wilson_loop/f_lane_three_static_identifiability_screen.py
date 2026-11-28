"""Deterministic synthetic identifiability and initialization stress for 2×3 fit."""
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
from tasks.wilson_loop import f_lane_three_static_two_vacuum_candidate as candidate
from tasks.wilson_loop import f_lane_two_sector_omitted_coverage_diagnostic as generator
from tasks.wilson_loop.f_lane_three_static_two_vacuum_candidate import fit_three_static_snapshot
from tasks.wilson_loop.f_lane_two_sector_fit_candidate import NT_VALUES,R_VALUES,T_VALUES


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
DEFAULT_OUT = TASK_DIR / "f_lane_three_static_identifiability_20261128_v1"
N_BLOCKS = 96
RELATIVE_NOISE = 0.003
SEED_BASE = 2026114200

STRATA = (
    "baseline",
    "near_degenerate",
    "weak_ground_overlap",
    "omitted_vacuum_truth",
    "dependent_unequal_variance",
)

SOURCE_PATHS = {
    "identifiability_screen.py": Path(__file__),
    "three_static_candidate.py": Path(candidate.__file__),
    "three_static_fit_pilot.py": TASK_DIR / "f_lane_three_static_fit_pilot.py",
    "two_static_candidate.py": TASK_DIR / "f_lane_two_sector_fit_candidate.py",
    "joint_forward_fit.py": Path(forward.__file__),
    "bootstrap_snapshot_helper.py": Path(bootstrap.__file__),
    "omitted_coverage_generator.py": Path(generator.__file__),
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


def _case_parameters(stratum: str) -> dict[str, Any]:
    if stratum not in STRATA:
        raise ValueError(f"unknown stratum: {stratum}")
    energies = {radius: toy.STATIC_ENERGIES[radius].copy() for radius in R_VALUES}
    overlaps = toy.TRANSITION_OVERLAPS.copy()
    vacuum = toy.VACUUM_ENERGIES.copy()
    rho = 0.5
    scale_by_t = {str(time): 1.0 for time in T_VALUES}
    if stratum == "near_degenerate":
        energies[2] = np.asarray([0.44, 0.475, 1.29])
        energies[3] = np.asarray([0.59, 0.625, 1.41])
    elif stratum == "weak_ground_overlap":
        overlaps[:, :, 0, :] *= 0.05
    elif stratum == "omitted_vacuum_truth":
        vacuum = np.asarray([0.0])
        overlaps = overlaps[:, :1, :, :]
    elif stratum == "dependent_unequal_variance":
        rho = 0.75
        scale_by_t = {"3": 0.50, "4": 1.40, "5": 0.80, "6": 1.80, "7": 0.65}
    return {
        "stratum": stratum, "static_energies": energies,
        "overlaps": overlaps, "vacuum_energies": vacuum,
        "rho": rho, "scale_by_T": scale_by_t,
    }


def _generate_panel(seed: int, stratum: str) -> tuple[np.ndarray, dict[str, Any]]:
    spec = _case_parameters(stratum)
    rng = np.random.default_rng(seed)
    vacuum = spec["vacuum_energies"]
    overlaps = spec["overlaps"]
    matrices = np.empty((N_BLOCKS, len(NT_VALUES), len(R_VALUES), len(T_VALUES), 3, 3),
                        dtype=np.complex128)
    for radius_index, radius in enumerate(R_VALUES):
        static = spec["static_energies"][radius]
        amplitudes = overlaps[radius_index]
        for nt_index, nt in enumerate(NT_VALUES):
            partition = float(np.sum(np.exp(-vacuum * nt)))
            for time_index, time_extent in enumerate(T_VALUES):
                weights = np.exp(
                    -vacuum[:, None] * (nt - time_extent)
                    -static[None, :] * time_extent
                ) / partition
                mean = np.einsum("mn,mni,mnj->ij", weights, amplitudes,
                                 amplitudes.conj(), optimize=True)
                sigma = max(
                    float(np.max(np.abs(mean))) * RELATIVE_NOISE
                    * float(spec["scale_by_T"][str(time_extent)]), 1e-9,
                )
                innovations = np.stack([
                    generator._hermitian_noise(rng) for _ in range(N_BLOCKS)
                ])
                sequence = np.empty_like(innovations)
                sequence[0] = innovations[0]
                innovation_scale = math.sqrt(1.0 - float(spec["rho"]) ** 2)
                for block in range(1, N_BLOCKS):
                    sequence[block] = (
                        float(spec["rho"]) * sequence[block - 1]
                        + innovation_scale * innovations[block]
                    )
                matrices[:, nt_index, radius_index, time_index] = mean + sigma * sequence
    truth_force = float(spec["static_energies"][3][0] - spec["static_energies"][2][0])
    spec_summary = {
        "stratum": stratum, "seed": seed, "blocks": N_BLOCKS,
        "vacuum_state_count_truth": int(len(vacuum)),
        "vacuum_energies_truth": vacuum.tolist(),
        "static_energies_truth_by_R": {
            str(radius): spec["static_energies"][radius].tolist() for radius in R_VALUES
        },
        "rho": float(spec["rho"]), "relative_noise": RELATIVE_NOISE,
        "scale_by_T": spec["scale_by_T"], "force_truth": truth_force,
    }
    return matrices, spec_summary


def _fit_panel(output: Path, panel_index: int, stratum: str, seed: int,
               contract_bytes: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    matrices, spec = _generate_panel(seed, stratum)
    panel_dir = output / "panels" / f"{panel_index:02d}_{stratum}"
    panel_dir.mkdir(parents=True)
    split = f"F-synthetic-open-identifiability-{panel_index:02d}-{stratum}"
    snapshot, input_hashes = bootstrap._snapshot_from_matrices(
        panel_dir, matrices, contract_bytes, split,
    )
    try:
        fit = fit_three_static_snapshot(snapshot)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        fit = {
            "status": "screen_fit_exception", "certified": False,
            "physical_score": None,
            "denominator": {"planned": 1, "valid": 0, "failed": 1, "abstained": 0},
            "error_type": type(exc).__name__, "error_message": str(exc),
            "initializer_fits": [], "fit_attempts": [], "fit": None,
        }
    (panel_dir / "lane_contract.json").unlink()
    fit_payload = _json_bytes(fit)
    (panel_dir / "fit.json").write_bytes(fit_payload)
    model = fit.get("fit") or {}
    valid = fit.get("denominator", {}).get("valid") == 1
    row = {
        "panel": panel_index, "stratum": stratum, "seed": seed,
        "input_matrix_sha256": input_hashes["raw_matrices"],
        "split_id": split, "status": fit.get("status"), "valid": valid,
        "failed": fit.get("denominator", {}).get("failed", 0),
        "abstained": fit.get("denominator", {}).get("abstained", 0),
        "parameter_count": model.get("parameter_count"),
        "jacobian_rank": model.get("jacobian_rank"),
        "jacobian_condition_number": model.get("jacobian_condition_number"),
        "reduced_gls_objective": model.get("reduced_gls_objective"),
        "force_truth": spec["force_truth"],
        "force_estimate": model.get("absolute_stout_ground_force"),
        "absolute_force_error": (
            abs(float(model["absolute_stout_ground_force"]) - spec["force_truth"])
            if valid else None
        ),
        "initializer_fits_json": json.dumps(fit.get("initializer_fits", []), sort_keys=True),
        "fit_attempts_json": json.dumps(fit.get("fit_attempts", []), sort_keys=True),
    }
    return row, {"spec": spec, "fit": fit}


def _plot(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained")
    x = np.arange(len(rows))
    truth = np.asarray([row["force_truth"] for row in rows], dtype=float)
    axes[0].scatter(x, truth, marker="_", color="black", s=140, linewidths=2,
                    label="exact force")
    valid_rows = [i for i, row in enumerate(rows) if row["valid"]]
    if valid_rows:
        estimates = np.asarray([rows[i]["force_estimate"] for i in valid_rows], dtype=float)
        truths = np.asarray([rows[i]["force_truth"] for i in valid_rows], dtype=float)
        axes[0].scatter(valid_rows, estimates, marker="s", color="#0072b2", s=45,
                        label="2-vacuum × 3-static point")
        for index, estimate, exact in zip(valid_rows, estimates, truths):
            axes[0].plot([index, index], [exact, estimate], color="#888888", alpha=0.6)
    for index, row in enumerate(rows):
        if not row["valid"]:
            axes[0].text(index, truth[index], "FAIL", ha="center", va="bottom",
                         color="#d55e00", fontsize=8)
    axes[0].set_xticks(x, [row["stratum"].replace("_", "\n") for row in rows])
    axes[0].set_ylabel("absolute force (lattice units)")
    axes[0].set_title("One deterministic panel per identifiability stratum")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)

    valid_condition = [row["jacobian_condition_number"] for row in rows
                       if row["valid"] and row["jacobian_condition_number"] is not None]
    valid_condition_x = [i for i, row in enumerate(rows)
                         if row["valid"] and row["jacobian_condition_number"] is not None]
    if valid_condition:
        axes[1].scatter(valid_condition_x, valid_condition, color="#cc79a7", marker="o",
                        label="local Jacobian condition number")
    for index, row in enumerate(rows):
        if row["jacobian_rank"] is not None:
            axes[1].text(index, row["jacobian_condition_number"],
                         f"rank {row['jacobian_rank']}/67", fontsize=7,
                         ha="center", va="bottom")
        else:
            axes[1].text(index, 1.2, "no full fit", fontsize=8, rotation=90,
                         ha="center", va="bottom")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, [row["stratum"].replace("_", "\n") for row in rows])
    axes[1].set_ylabel("Jacobian condition number (log scale)")
    axes[1].set_title("Local fit rank / conditioning; no coverage inference")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Finite-Nt three-static fit identifiability screen")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _artifact_hashes(output: Path) -> dict[str, str]:
    values = {
        str(path.relative_to(output)): _sha(path.read_bytes())
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output / "artifact_hashes.json").write_bytes(_json_bytes({
        "schema": "wilson-three-static-identifiability-artifact-hashes-v1",
        "artifacts": values,
    }))
    return values


def _fmt(value: Any, spec: str = ".4g") -> str:
    if value is None or isinstance(value, bool) or value == "":
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "—"
    return format(number, spec) if math.isfinite(number) else "—"


def finalize_existing(output: str | Path) -> dict[str, Any]:
    """Finish a report from persisted fits after a presentation-only failure."""
    out = Path(output)
    required = ("strata_results.csv", "summary.json", "fit_receipts.json",
                "identifiability_profiles.png", "protocol.json")
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot resume report; missing {missing}")
    source_payloads = {name: path.read_bytes() for name, path in SOURCE_PATHS.items()}
    source_hashes = {name: _sha(payload) for name, payload in source_payloads.items()}
    source_dir = out / "source_snapshot"
    source_dir.mkdir(exist_ok=True)
    for name, payload in source_payloads.items():
        (source_dir / name).write_bytes(payload)
    protocol_path = out / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["source_hashes"] = source_hashes
    recovery = (
        "All strata fits and receipts were persisted. The first README render failed on an unescaped literal dict brace; "
        "this recovery changes report formatting/source hashes only and reuses the saved matrices and fit results without refitting."
    )
    protocol["report_recovery"] = recovery
    protocol_path.write_bytes(_json_bytes(protocol))
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source_hashes"] = source_hashes
    summary["report_recovery"] = recovery
    summary_path.write_bytes(_json_bytes(summary))
    rows = list(csv.DictReader((out / "strata_results.csv").open(encoding="utf-8", newline="")))
    table = []
    for row in rows:
        table.append(
            f"| {row['stratum']} | {row['status']} | "
            f"{row['jacobian_rank'] or '—'}/67 | {_fmt(row['jacobian_condition_number'])} | "
            f"{_fmt(row['force_estimate'])} | {_fmt(row['absolute_force_error'])} |"
        )
    README = f"""# Finite-Nt three-static fit identifiability screen

**Status:** `{protocol['classification']}`. One deterministic fresh panel per stress stratum is a feasibility and negative-control screen; it is not a coverage study.

## Strata and data flow

- Baseline positive 2-vacuum × 3-static trace.
- Near-degenerate first static gap: `E_1-E_0=0.035` at each radius.
- Weak ground overlap: all ground-transition overlap vectors scaled by `0.05`.
- Omitted-vacuum truth: one vacuum state generated, two fit.
- Dependent/unequal-variance blocks: AR(1) `rho=0.75`, with time scales `{{3:0.50, 4:1.40, 5:0.80, 6:1.80, 7:0.65}}`.

Each stratum uses 96 blocks, relative noise `0.003`, and seed `2026114200 + stratum_index`. Data flow:

```text
frozen positive-spectrum stress truth
  -> raw finite-Nt matrices with the declared block/noise process
  -> captured manifest + source snapshot
  -> Nt=32 3-state initializer -> full finite-Nt 2-vacuum × 3-static GLS
  -> retain rank, condition, force error, all starts/failures and raw matrices
```

![Force error and local conditioning by stratum](identifiability_profiles.png)

## Results

| Stratum | Fit result | Local rank | Condition number | Force point | Absolute point error |
|---|---|---:|---:|---:|---:|
{chr(10).join(table)}

There are **{summary['denominator']['valid']}/{summary['denominator']['planned']}** valid panel fits, **{summary['denominator']['failed']}** failed and **{summary['denominator']['abstained']}** abstained. One panel per stress cannot estimate coverage or failure rates. In particular, local rank is not global identifiability. The weak-ground-overlap profile has the largest visible force error, while the near-degenerate profile has the poorest conditioning; neither observation is a general bound.

## Interpretation and limits

These tests assess candidate initialization and local fit stability only. The GLS objective and inverse-Jacobian force SE are uncalibrated. Do not connect this candidate to the physical force summary reader or interpret its point errors as a confidence interval.

No production Wilson matrices, GEVP data, sealed split, GPU or QUDA were accessed. F remains `GROUND_STATE_FORCE_UNCERTIFIED` and `physical_score=null`.

**Report recovery note:** the fit pass completed and persisted all panel receipts; the first README render stopped on an unescaped literal dictionary brace. This finalized report updates the README, source hash and artifact manifest using the already-saved raw matrices and fit results; no panel was refit or removed.
"""
    (out / "README.md").write_text(README, encoding="utf-8")
    hashes = _artifact_hashes(out)
    return {
        "output": str(out), "planned": len(rows),
        "valid": int(summary["denominator"]["valid"]),
        "failed": int(summary["denominator"]["failed"]),
        "artifacts": len(hashes), "resumed_from_saved_fits": True,
    }


def run(output: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {out}")
    out.mkdir(parents=True)
    payloads = {name: path.read_bytes() for name, path in SOURCE_PATHS.items()}
    source_hashes = {name: _sha(value) for name, value in payloads.items()}
    source_dir = out / "source_snapshot"
    source_dir.mkdir()
    for name, payload in payloads.items():
        (source_dir / name).write_bytes(payload)
    contract = payloads["lane_contract_v2.json"]
    (out / "lane_contract.json").write_bytes(contract)
    protocol = {
        "schema": "wilson-three-static-identifiability-protocol-v1",
        "classification": "DETERMINISTIC_SYNTHETIC_IDENTIFIABILITY_SCREEN",
        "seed_rule": f"{SEED_BASE} + stratum_index", "blocks_per_stratum": N_BLOCKS,
        "relative_noise": RELATIVE_NOISE, "strata": list(STRATA),
        "failure_rule": "retain each panel; no model fit or rank failure is converted to a point estimate",
        "coverage_claim": False,
        "source_hashes": source_hashes,
    }
    (out / "protocol.json").write_bytes(_json_bytes(protocol))
    rows, detailed = [], []
    for panel_index, stratum in enumerate(STRATA):
        row, detail = _fit_panel(out, panel_index, stratum, SEED_BASE + panel_index, contract)
        rows.append(row)
        detailed.append(detail)
        print(f"completed identifiability profile {panel_index + 1}/{len(STRATA)}: {stratum}",
              flush=True)
    _write_csv(out / "strata_results.csv", rows)
    _plot(out / "identifiability_profiles.png", rows)
    valid = [row for row in rows if row["valid"]]
    summary = {
        "schema": "wilson-three-static-identifiability-summary-v1",
        "classification": protocol["classification"], "certified": False,
        "physical_score": None, "f_lane_status": "GROUND_STATE_FORCE_UNCERTIFIED",
        "denominator": {"planned": len(rows), "valid": len(valid),
                        "failed": len(rows) - len(valid), "abstained": 0},
        "force_abs_error": {
            "valid_panels": len(valid),
            "median": float(np.median([row["absolute_force_error"] for row in valid]))
            if valid else None,
            "maximum": float(max(row["absolute_force_error"] for row in valid))
            if valid else None,
        },
        "strata": [{
            "stratum": row["stratum"], "status": row["status"],
            "valid": row["valid"], "rank": row["jacobian_rank"],
            "condition_number": row["jacobian_condition_number"],
            "force_truth": row["force_truth"], "force_estimate": row["force_estimate"],
            "force_error": row["absolute_force_error"],
        } for row in rows],
        "scope_limits": [
            "one deterministic panel per stratum does not estimate sampling coverage or failure rates",
            "the five stress variations are method counterexamples, not physical parameter bounds",
            "local Jacobian rank is not global identifiability and the candidate has no calibrated interval",
            "production exporter/action convention, thermal/mixed bounds, force reference, independent review and authorization remain absent",
        ],
        "source_hashes": source_hashes,
    }
    (out / "summary.json").write_bytes(_json_bytes(summary))
    (out / "fit_receipts.json").write_bytes(_json_bytes(detailed))
    README = f"""# Finite-Nt three-static fit identifiability screen

**Status:** `{protocol['classification']}`. One deterministic fresh panel per stress stratum is a feasibility/negative-control screen; it is not a coverage study.

## Strata and data flow

- Baseline omitted-third-state-free 2-vacuum × 3-static positive trace.
- Near-degenerate first gap (`E_1-E_0=0.035` at both radii).
- Weak ground overlap (all ground-transition overlap vectors scaled by 0.05).
- One-vacuum truth fitted by the two-vacuum candidate.
- Stationary AR(1) `rho=0.75` plus time-dependent noise scales `{{3:0.50,4:1.40,5:0.80,6:1.80,7:0.65}}`.

Each stratum gets 96 blocks, relative noise `0.003`, and a separate seed `2026114200 + index`. The pipeline is `frozen stress truth -> positive finite-Nt matrix panel -> captured raw snapshot -> 3-state Nt=32 initializer -> full finite-Nt 2-vacuum × 3-static GLS -> retain rank, condition, force, starts and failures`.

![Force error and local conditioning by stratum](identifiability_profiles.png)

## Interpretation

The figure reports point errors and local Jacobian condition numbers. A missing point is a failed/abstained fit retained in the denominator. No empirical coverage or population failure rate can be inferred from one panel per stratum. Any full-rank result is local to the converged basin; the multi-start objective spread and candidate's uncalibrated force SE remain open.

No Wilson production matrices, GEVP data, sealed split, GPU or QUDA were accessed. F remains `GROUND_STATE_FORCE_UNCERTIFIED` and `physical_score=null`.
"""
    (out / "README.md").write_text(README, encoding="utf-8")
    artifacts = _artifact_hashes(out)
    return {
        "output": str(out), "planned": len(rows), "valid": len(valid),
        "failed": len(rows) - len(valid), "artifacts": len(artifacts),
        "certified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--finalize-existing", action="store_true",
                        help="rebuild README and hashes from persisted results, without refitting")
    args = parser.parse_args()
    result = finalize_existing(args.output) if args.finalize_existing else run(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
