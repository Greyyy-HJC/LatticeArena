"""Exact two-state counterexample for the proposed Pion anchor scalar map."""
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

from tasks.pion_projection import anchor_scalar_reader as reader


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
DEFAULT_OUT = TASK_DIR / "anchor_scalar_nonidentifiability_20261128_v1"
TEMPORAL_EXTENT = 16
MAP_TIME = 4
MAP_VALUE = 0.50
EXCITED_MASS = 1.20
GROUND_MASSES = (0.15, 0.45)
R_SCALE = 2.9
REQUIRED_C_OVER_R = 0.01
SOURCE_PATHS = {
    "diagnostic": Path(__file__),
    "anchor_scalar_reader": TASK_DIR / "anchor_scalar_reader.py",
    "bounded_risk_contract": TASK_DIR / "bounded_block_risk_contract_v1.json",
}


def _sha(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _kernel(mass: float, time: int) -> float:
    return math.exp(-mass * time) + math.exp(-mass * (TEMPORAL_EXTENT - time))


def solve_excited_amplitude_ratio(ground_mass: float, map_mass: float,
                                  excited_mass: float, *, map_time: int = MAP_TIME) -> float:
    """Choose positive A1/A0 so the mixture's local cosh mass equals map_mass."""
    if not (0.1 <= ground_mass < map_mass < excited_mass <= 3.0):
        raise ValueError("masses must satisfy support and ground < map < excited")
    ground_cosh = math.cosh(ground_mass)
    map_cosh = math.cosh(map_mass)
    excited_cosh = math.cosh(excited_mass)
    excited_fraction_at_map_time = (map_cosh - ground_cosh) / (excited_cosh - ground_cosh)
    if not 0.0 < excited_fraction_at_map_time < 1.0:
        raise ArithmeticError("target map mass does not induce a positive mixture fraction")
    ground_kernel = _kernel(ground_mass, map_time)
    excited_kernel = _kernel(excited_mass, map_time)
    return (excited_fraction_at_map_time / (1.0 - excited_fraction_at_map_time)
            * ground_kernel / excited_kernel)


def _correlator(ground_mass: float, excited_mass: float,
                excited_amplitude_ratio: float) -> np.ndarray:
    times = np.arange(TEMPORAL_EXTENT, dtype=float)
    return (np.exp(-ground_mass * times)
            + np.exp(-ground_mass * (TEMPORAL_EXTENT - times))
            + excited_amplitude_ratio * np.exp(-excited_mass * times)
            + excited_amplitude_ratio * np.exp(-excited_mass * (TEMPORAL_EXTENT - times)))


def _local_effective_mass(correlator: np.ndarray, time: int) -> float:
    argument = (correlator[time - 1] + correlator[time + 1]) / (2.0 * correlator[time])
    return float(np.arccosh(argument))


def _models() -> list[dict[str, Any]]:
    rows = []
    for index, ground_mass in enumerate(GROUND_MASSES, start=1):
        amplitude_ratio = solve_excited_amplitude_ratio(
            ground_mass, MAP_VALUE, EXCITED_MASS)
        correlator = _correlator(ground_mass, EXCITED_MASS, amplitude_ratio)
        mapped = reader.anchor_scalar(correlator)
        if not math.isclose(mapped, MAP_VALUE, rel_tol=0.0, abs_tol=1e-12):
            raise ArithmeticError("constructed spectrum does not reproduce the target map value")
        rows.append({
            "model_id": f"spectrum_{index}",
            "ground_mass": ground_mass,
            "excited_mass": EXCITED_MASS,
            "A1_over_A0": amplitude_ratio,
            "excited_fraction_at_t4": (
                amplitude_ratio * _kernel(EXCITED_MASS, MAP_TIME)
                / (_kernel(ground_mass, MAP_TIME)
                   + amplitude_ratio * _kernel(EXCITED_MASS, MAP_TIME))),
            "m_eff4": mapped,
            "raw_map_bias": mapped - ground_mass,
            "raw_abs_map_bias_over_R": abs(mapped - ground_mass) / R_SCALE,
            "correlator": correlator,
            "local_effective_masses": {
                time: _local_effective_mass(correlator, time)
                for time in range(1, TEMPORAL_EXTENT // 2)
            },
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]):
    fields = ["model_id", "ground_mass", "excited_mass", "A1_over_A0",
              "excited_fraction_at_t4", "m_eff4", "raw_map_bias",
              "raw_abs_map_bias_over_R"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def _plot(path: Path, rows: list[dict[str, Any]], minimax_error: float):
    colors = ("#2b78e4", "#d95f02")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
    times = np.arange(TEMPORAL_EXTENT // 2 + 1)
    for row, color in zip(rows, colors):
        correlator = row["correlator"]
        folded = np.asarray([correlator[0]] + [
            0.5 * (correlator[t] + correlator[TEMPORAL_EXTENT - t])
            for t in range(1, TEMPORAL_EXTENT // 2)
        ] + [correlator[TEMPORAL_EXTENT // 2]])
        axes[0].plot(times, folded / folded[MAP_TIME], marker="o", ms=3,
                     color=color, label=f"ground mass {row['ground_mass']:.2f}")
        effective_times = sorted(row["local_effective_masses"])
        axes[1].plot(effective_times,
                     [row["local_effective_masses"][t] for t in effective_times],
                     marker="o", ms=3, color=color,
                     label=f"local mass, ground {row['ground_mass']:.2f}")
        axes[1].axhline(row["ground_mass"], color=color, alpha=0.35, linestyle=":")
    axes[0].set(xlabel="folded Euclidean time t", ylabel=r"$C_{fold}(t)/C_{fold}(4)$",
                title="Different spectra, same local map at t=4")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[1].axhline(MAP_VALUE, color="black", linestyle="--",
                    label=r"shared $m_{eff}(4)=0.50$")
    axes[1].set(xlabel="local effective-mass time t", ylabel=r"$m_{eff}(t)$",
                title="The shared point does not identify the ground state")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _artifact_hashes(output: Path):
    files = sorted(path for path in output.rglob("*")
                    if path.is_file() and path.name != "artifact_hashes.json")
    hashes = {str(path.relative_to(output)): _sha(path.read_bytes()) for path in files}
    (output / "artifact_hashes.json").write_bytes(_json_bytes({
        "schema": "pion-anchor-scalar-nonidentifiability-artifact-hashes-v1",
        "artifacts": hashes,
    }))
    return hashes


def build(output: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {out}")
    source_payloads = {name: path.read_bytes() for name, path in SOURCE_PATHS.items()}
    source_hashes = {name: _sha(payload) for name, payload in source_payloads.items()}
    out.mkdir(parents=True)
    rows = _models()
    target_gap = abs(rows[1]["ground_mass"] - rows[0]["ground_mass"])
    minimax_error = target_gap / 2.0
    required_abs_bias = REQUIRED_C_OVER_R * R_SCALE
    if minimax_error <= required_abs_bias:
        raise ArithmeticError("counterexample must exceed the proposed common-bias allowance")
    summary = {
        "schema": "pion-anchor-scalar-nonidentifiability-summary-v1",
        "classification": "EXACT_NOISELESS_TWO_STATE_IDENTIFIABILITY_COUNTEREXAMPLE_NOT_PHYSICAL_DATA",
        "map": {"map_id": reader.MAP_ID, "map_sha256": reader.MAP_HASH,
                "formula": "acosh((C_fold(3)+C_fold(5))/(2*C_fold(4)))",
                "map_time": MAP_TIME, "support": list(reader.SUPPORT)},
        "assumptions": {
            "temporal_extent": TEMPORAL_EXTENT,
            "truth_model": "positive two-state periodic PP cosh, A0=1 and A1>0",
            "unknowns_not_restricted_by_current_map_contract": ["ground mass", "excited mass", "excited overlap"],
            "no_statistical_noise": True,
            "no_chain_or_physical_input_read": True,
        },
        "examples": [{key: value for key, value in row.items()
                      if key not in {"correlator", "local_effective_masses"}}
                     for row in rows],
        "identifiability": {
            "shared_map_value": MAP_VALUE,
            "ground_mass_separation": target_gap,
            "any_single_output_minimax_abs_error_lower_bound": minimax_error,
            "R": R_SCALE,
            "required_C_over_R": REQUIRED_C_OVER_R,
            "required_abs_bias": required_abs_bias,
            "minimax_C_over_R_lower_bound": minimax_error / R_SCALE,
            "required_bound_is_impossible_for_this_model_class": minimax_error > required_abs_bias,
            "proof": "same map input implies same scalar output for both spectra; triangle inequality makes worst-case error at least half the ground-mass separation",
        },
        "denominator": {"planned": len(rows), "valid": len(rows), "failed": 0, "abstained": 0,
                        "unit": "exact analytic counterexample spectra"},
        "physical_anchor_input": {"planned": 0, "valid": 0, "failed": 0, "abstained": 0},
        "physical_score": None,
        "reference_gate": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
        "source_hashes": source_hashes,
    }
    (out / "summary.json").write_bytes(_json_bytes(summary))
    _write_csv(out / "counterexample_spectra.csv", rows)
    protocol = {
        "schema": "pion-anchor-scalar-nonidentifiability-protocol-v1",
        "classification": summary["classification"],
        "source_hashes": source_hashes,
        "target_map_mass": MAP_VALUE,
        "excited_mass": EXCITED_MASS,
        "ground_masses": list(GROUND_MASSES),
        "amplitude_solution": "w4=(cosh(M)-cosh(m0))/(cosh(m1)-cosh(m0)); A1/A0=(w4/(1-w4))*K(m0,4)/K(m1,4)",
        "kernel": "K(m,t)=exp(-m*t)+exp(-m*(16-t))",
        "R": R_SCALE,
        "required_C_over_R": REQUIRED_C_OVER_R,
        "formula_check": "reader.anchor_scalar(correlator) == target_map_mass within absolute tolerance 1e-12 for every row",
        "physical_score": None,
    }
    (out / "protocol.json").write_bytes(_json_bytes(protocol))
    source_dir = out / "source_snapshot"
    source_dir.mkdir()
    for name, payload in source_payloads.items():
        (source_dir / name.replace("/", "_")).write_bytes(payload)
    (out / "source_hashes.json").write_bytes(_json_bytes({
        "schema": "pion-anchor-scalar-nonidentifiability-source-hashes-v1",
        "sources": source_hashes,
    }))
    _plot(out / "nonidentifiability.png", rows, minimax_error)
    readme = [
        "# Pion anchor scalar map: exact identifiability counterexample",
        "",
        "**Classification:** exact noiseless synthetic two-state counterexample. No Pion configuration, anchor/reference bytes, sealed split, GPU or QUDA were accessed. `physical_score=null`.",
        "",
        "## Construction",
        "",
        "The proposed reader maps one state to `M=m_eff(4)=acosh((C_fold(3)+C_fold(5))/(2 C_fold(4)))`. For the positive periodic two-state spectrum `C(t)=A0 K(m0,t)+A1 K(m1,t)`, `K(m,t)=exp(-mt)+exp(-m(16-t))`, the identity `K(m,3)+K(m,5)=2 cosh(m) K(m,4)` gives:",
        "",
        "```text",
        "cosh(M) = (1-w4) cosh(m0) + w4 cosh(m1)",
        "w4 = A1 K(m1,4) / (A0 K(m0,4) + A1 K(m1,4))",
        "```",
        "",
        "Solving this equation lets distinct ground masses produce the same map value with positive finite overlaps.",
        "",
        "| spectrum | ground mass m0 | excited mass m1 | A1/A0 | returned m_eff(4) | |m_eff(4)-m0|/R |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        readme.append(
            f"| {row['model_id']} | {row['ground_mass']:.2f} | {row['excited_mass']:.2f} | "
            f"{row['A1_over_A0']:.6f} | {row['m_eff4']:.12f} | "
            f"{row['raw_abs_map_bias_over_R']:.6f} |"
        )
    readme.extend([
        "",
        "![Two spectra with the same t=4 local map but different ground masses](nonidentifiability.png)",
        "",
        f"The two valid reader inputs have the same map value `{MAP_VALUE}`, while their ground masses differ by `{target_gap:.3f}`. Any deterministic correction that sees only that scalar must return the same value for both; by the triangle inequality its worst-case absolute error is at least half the separation, `{minimax_error:.3f}`. The selected bound requires `C <= {required_abs_bias:.3f}` (`C/R <= {REQUIRED_C_OVER_R:.3f}`, `R={R_SCALE}`), so that guarantee is impossible over this two-state class without additional restrictions or observables. This remains true even though both map values and both ground masses lie inside `[0.1,3.0]` and both spectra are noiseless, positive, and exactly periodic.",
        "",
        "This is a deliberately favorable submodel in which the target joint-fit mass is exactly the PP ground-state mass `m0`; it does not prove that the actual anchor ensemble contains either spectrum. It proves that the current map formula plus support check alone cannot establish `C/R<=0.01`. An independently supported restriction on the excitation gap/overlap, or a richer per-state observable tied to the frozen joint PP/A4P target, is required. Noise, model mismatch, chain dependence and finite-sample invalid-state rates can only require further validation; they cannot repair this noiseless non-identifiability by themselves.",
        "",
        "## Data flow and decision",
        "",
        "```text",
        "choose two ground masses in support -> solve positive overlap so both map to M=0.50 -> verify with the production scalar-reader formula -> apply triangle-inequality minimax lower bound -> reject a universal C/R<=0.01 claim from this scalar alone -> keep anchor gate ABSTAIN",
        "```",
        "",
        "Analytic example denominator: `2/2/0/0`; production anchor input denominator: `0/0/0/0`; reference gate: `1/0/0/1`. The next step is an independent review of this no-go argument and a decision whether to replace the scalar map with a multi-time/per-channel estimator or require an externally justified spectral/overlap restriction. Do not infer a physical bias, open sealed data, or promote the gate.",
        "",
        "The protocol, example CSV, source snapshots, source hashes, and artifact hashes are included in this folder.",
    ])
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    artifacts = _artifact_hashes(out)
    return {"output": str(out), "spectra": len(rows),
            "map_value": MAP_VALUE, "ground_mass_gap": target_gap,
            "minimax_C_over_R_lower_bound": minimax_error / R_SCALE,
            "artifact_count": len(artifacts), "source_count": len(source_hashes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
