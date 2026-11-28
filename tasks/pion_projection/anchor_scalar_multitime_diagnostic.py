"""Check whether a predeclared multi-time PP feature separates the exact scalar collision."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tasks.pion_projection import anchor_scalar_nonidentifiability as collision


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = TASK_DIR / "anchor_scalar_multitime_20261128_v1"
FEATURE_TIMES = (2, 3, 5, 6)
EFFECTIVE_MASS_TIMES = (2, 3, 4, 5, 6)
SOURCE_PATHS = {
    "diagnostic": Path(__file__),
    "scalar_collision_diagnostic": TASK_DIR / "anchor_scalar_nonidentifiability.py",
    "anchor_scalar_reader": TASK_DIR / "anchor_scalar_reader.py",
}


def _sha(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fold(correlator: np.ndarray) -> np.ndarray:
    extent = collision.TEMPORAL_EXTENT
    return np.asarray([
        correlator[0],
        *[0.5 * (correlator[t] + correlator[extent - t])
          for t in range(1, extent // 2)],
        correlator[extent // 2],
    ], dtype=float)


def _build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spectra = collision._models()
    signatures = []
    timeseries = []
    for spectrum in spectra:
        folded = _fold(spectrum["correlator"])
        normalized = folded / folded[collision.MAP_TIME]
        for time in FEATURE_TIMES:
            signatures.append({
                "model_id": spectrum["model_id"],
                "ground_mass": spectrum["ground_mass"],
                "feature": f"C_fold({time})/C_fold(4)",
                "time": time,
                "value": float(normalized[time]),
            })
        for time in range(0, collision.TEMPORAL_EXTENT // 2 + 1):
            timeseries.append({
                "model_id": spectrum["model_id"],
                "ground_mass": spectrum["ground_mass"],
                "time": time,
                "normalized_correlator": float(normalized[time]),
                "local_effective_mass": (
                    float(spectrum["local_effective_masses"][time])
                    if time in spectrum["local_effective_masses"] else None
                ),
            })
    return signatures, timeseries


def _feature_vectors(signatures: list[dict[str, Any]]):
    models = sorted({row["model_id"] for row in signatures})
    vectors = {}
    for model in models:
        rows = sorted((row for row in signatures if row["model_id"] == model),
                      key=lambda row: row["time"])
        vectors[model] = np.asarray([row["value"] for row in rows], dtype=float)
    return vectors


def _plot(output: Path, signatures: list[dict[str, Any]], timeseries: list[dict[str, Any]],
          separation: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
    colors = {"spectrum_1": "#2b78e4", "spectrum_2": "#d95f02"}
    for model in sorted(colors):
        rows = sorted((row for row in timeseries if row["model_id"] == model),
                      key=lambda row: row["time"])
        axes[0].plot([row["time"] for row in rows],
                     [row["normalized_correlator"] for row in rows],
                     marker="o", ms=4, color=colors[model], label=model)
        mass_rows = [row for row in rows if row["local_effective_mass"] is not None]
        axes[1].plot([row["time"] for row in mass_rows],
                     [row["local_effective_mass"] for row in mass_rows],
                     marker="o", ms=4, color=colors[model], label=f"{model}: local mass")
        axes[1].axhline(mass_rows[0]["ground_mass"], color=colors[model],
                        linestyle=":", alpha=0.55, label=f"{model}: ground mass")
    axes[0].axvline(collision.MAP_TIME, color="black", linestyle="--", alpha=0.5,
                    label="shared scalar uses t=4")
    axes[0].set(xlabel="folded time t", ylabel=r"$C_{fold}(t)/C_{fold}(4)$",
                title="A multi-time signature separates the scalar collision")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[1].axvline(collision.MAP_TIME, color="black", linestyle="--", alpha=0.5)
    axes[1].set(xlabel="local effective-mass time t", ylabel=r"$m_{eff}(t)$",
                title=f"Same t=4 value; four-time feature distance={separation:.3f}")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=7, ncol=2)
    fig.savefig(output / "multitime_separation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _artifact_hashes(output: Path):
    files = sorted(path for path in output.rglob("*")
                   if path.is_file() and path.name != "artifact_hashes.json")
    hashes = {str(path.relative_to(output)): _sha(path.read_bytes()) for path in files}
    (output / "artifact_hashes.json").write_bytes(_json_bytes({
        "schema": "pion-anchor-scalar-multitime-artifact-hashes-v1",
        "artifacts": hashes,
    }))
    return hashes


def build(output: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {out}")
    source_payloads = {name: path.read_bytes() for name, path in SOURCE_PATHS.items()}
    source_hashes = {name: _sha(payload) for name, payload in source_payloads.items()}
    signatures, timeseries = _build_rows()
    vectors = _feature_vectors(signatures)
    model_ids = sorted(vectors)
    difference = vectors[model_ids[0]] - vectors[model_ids[1]]
    l2_separation = float(np.linalg.norm(difference))
    linf_separation = float(np.max(np.abs(difference)))
    if l2_separation <= 0.0:
        raise ArithmeticError("multi-time feature did not separate the exact collision pair")
    scalar_gap = abs(collision._models()[0]["m_eff4"] - collision._models()[1]["m_eff4"])
    summary = {
        "schema": "pion-anchor-scalar-multitime-summary-v1",
        "classification": "EXACT_TWO_STATE_PAIR_SEPARATED_BY_MULTITIME_FEATURE_NOT_GLOBAL_IDENTIFIABILITY",
        "physical_score": None,
        "scalar_collision": {
            "map_id": "pion-folded-pp-cosh-effective-mass-t4-v1",
            "feature": "m_eff(4)", "absolute_difference": scalar_gap,
        },
        "multi_time_feature": {
            "definition": [f"C_fold({time})/C_fold(4)" for time in FEATURE_TIMES],
            "feature_times": list(FEATURE_TIMES),
            "vector_spectrum_1": vectors[model_ids[0]].tolist(),
            "vector_spectrum_2": vectors[model_ids[1]].tolist(),
            "l2_separation": l2_separation,
            "linf_separation": linf_separation,
            "conditional_noise_statement": "if each observed feature vector has deterministic L2 error <= epsilon and 2*epsilon < the exact vector separation, these two neighborhoods do not overlap; no noise/error bound is supplied here",
        },
        "models": [
            {"model_id": row["model_id"], "ground_mass": row["ground_mass"],
             "excited_mass": row["excited_mass"], "A1_over_A0": row["A1_over_A0"],
             "m_eff4": row["m_eff4"]}
            for row in collision._models()
        ],
        "denominators": {
            "counterexample_spectra": {"planned": 2, "valid": 2, "failed": 0, "abstained": 0},
            "feature_vectors": {"planned": 2, "valid": 2, "failed": 0, "abstained": 0},
            "physical_anchor_inputs": {"planned": 0, "valid": 0, "failed": 0, "abstained": 0},
            "physical_gate": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
        },
        "source_hashes": source_hashes,
    }
    out.mkdir(parents=True)
    (out / "summary.json").write_bytes(_json_bytes(summary))
    _write_csv(out / "feature_signatures.csv", signatures)
    _write_csv(out / "normalized_timeseries.csv", timeseries)
    protocol = {
        "schema": "pion-anchor-scalar-multitime-protocol-v1",
        "classification": summary["classification"],
        "source_hashes": source_hashes,
        "source_scalar_collision_report": "../anchor_scalar_nonidentifiability_20261128_v1",
        "feature_times": list(FEATURE_TIMES),
        "feature_definition": "C_fold(t)/C_fold(4), t in {2,3,5,6}",
        "models": summary["models"],
        "l2_separation": l2_separation,
        "linf_separation": linf_separation,
        "physical_score": None,
    }
    (out / "protocol.json").write_bytes(_json_bytes(protocol))
    source_dir = out / "source_snapshot"
    source_dir.mkdir()
    for name, payload in source_payloads.items():
        (source_dir / name.replace("/", "_")).write_bytes(payload)
    (out / "source_hashes.json").write_bytes(_json_bytes({"sources": source_hashes}))
    _plot(out, signatures, timeseries, l2_separation)
    readme = [
        "# Pion anchor map: multi-time feature separation of the exact collision pair",
        "",
        "**Classification:** exact, noiseless, two-state periodic PP examples only. The production scalar returns the same `m_eff(4)=0.50` for both spectra; no anchor/reference data, sealed split, GPU or QUDA were accessed. `physical_score=null`.",
        "",
        "## Candidate feature and result",
        "",
        "Use the predeclared shape vector `x=(C_fold(2),C_fold(3),C_fold(5),C_fold(6))/C_fold(4)`. For the two exact counterexamples in the companion non-identifiability report, the vectors are:",
        "",
        f"- ground mass `0.15`: `{np.array2string(vectors[model_ids[0]], precision=6)}`",
        f"- ground mass `0.45`: `{np.array2string(vectors[model_ids[1]], precision=6)}`",
        f"- exact L2 separation `{l2_separation:.6f}`, L-infinity separation `{linf_separation:.6f}`; the scalar difference remains `{scalar_gap:.1e}`.",
        "",
        "![Multi-time feature separation](multitime_separation.png)",
        "",
        "For these two known spectra only, retaining the neighboring time points distinguishes a pair that the single t=4 scalar collapses. If both observed feature vectors had a deterministic L2 error at most `epsilon` and `2 epsilon` were below the exact separation, their error balls could not overlap; this report supplies no such data error bound.",
        "",
        "## Data flow and limits",
        "",
        "```text",
        "reuse the two exact scalar-collision spectra -> fold T=16 PP correlators -> normalize C(t) by C(4) at t={2,3,5,6} -> compare four-dimensional signatures -> distinguish this pair only -> keep physical anchor gate abstained",
        "```",
        "",
        "This does not establish global identifiability for arbitrary positive spectra, identify an unknown number of states, prove per-configuration positivity, or bound the map-to-joint-PP/A4P target bias. A richer feature vector needs a frozen estimator, failure/support rules, noise and chain-dependence calibration, omitted-state validation and an explicit bridge to `m*` before it can replace the scalar anchor.",
        "",
        "The exact model inputs, feature rows, protocol, source snapshots and hashes are included here. Reproduce from the repository root with:",
        "",
        "```bash",
        "python -m tasks.pion_projection.anchor_scalar_multitime_diagnostic \\",
        "  --output tasks/pion_projection/anchor_scalar_multitime_replay_v1",
        "```",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    artifacts = _artifact_hashes(out)
    return {"output": str(out), "feature_l2_separation": l2_separation,
            "feature_linf_separation": linf_separation, "artifacts": len(artifacts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
