"""Construct two positive periodic PP spectra with the same multi-time map."""
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

from tasks.pion_projection import anchor_scalar_reader as reader


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = TASK_DIR / "anchor_multitime_alias_20261127_v1"
TEMPORAL_EXTENT = 16
FEATURE_TIMES = (2, 3, 5, 6)
UNIQUE_FEATURE_TIMES = FEATURE_TIMES
GROUND_SPECTRUM_MASSES = (0.15, 0.60, 0.90, 1.20)
GROUND_SPECTRUM_WEIGHTS_T4 = (0.03, 0.97 / 3, 0.97 / 3, 0.97 / 3)
ALIAS_SPECTRUM_MASSES = (0.45, 0.70, 1.00, 1.30, 1.50)
SOURCE_PATHS = {
    "diagnostic": Path(__file__),
    "anchor_scalar_reader": TASK_DIR / "anchor_scalar_reader.py",
    "multitime_separation_diagnostic": TASK_DIR / "anchor_scalar_multitime_diagnostic.py",
    "scalar_collision_diagnostic": TASK_DIR / "anchor_scalar_nonidentifiability.py",
}


def _sha(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _kernel(mass: float, time: int) -> float:
    return float(np.exp(-mass * time) + np.exp(-mass * (TEMPORAL_EXTENT - time)))


def _correlator(masses: tuple[float, ...], weights_t4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kernels_t4 = np.asarray([_kernel(mass, 4) for mass in masses])
    amplitudes = weights_t4 / kernels_t4
    values = np.asarray([
        sum(amplitude * _kernel(mass, time)
            for mass, amplitude in zip(masses, amplitudes))
        for time in range(TEMPORAL_EXTENT)
    ], dtype=float)
    return values, amplitudes


def _feature_vector(correlator: np.ndarray) -> np.ndarray:
    folded = reader.fold_pp_even(correlator)
    return np.asarray([folded[time] / folded[4] for time in UNIQUE_FEATURE_TIMES])


def _models() -> list[dict[str, Any]]:
    first_masses = GROUND_SPECTRUM_MASSES
    first_weights = np.asarray(GROUND_SPECTRUM_WEIGHTS_T4, dtype=float)
    first_correlator, first_amplitudes = _correlator(first_masses, first_weights)
    target_features = _feature_vector(first_correlator)

    # Solve for a positive five-state alias exactly in the four ratios plus
    # the t=4 weight normalization. Folding only relates t to T-t, so all four
    # listed times (2, 3, 5, 6) are distinct in the half-lattice range.
    columns = []
    for mass in ALIAS_SPECTRUM_MASSES:
        columns.append(np.r_[1.0, [
            _kernel(mass, time) / _kernel(mass, 4)
            for time in UNIQUE_FEATURE_TIMES
        ]])
    alias_weights = np.linalg.solve(np.asarray(columns).T,
                                    np.r_[1.0, target_features])
    if np.any(alias_weights <= 0) or not np.isclose(alias_weights.sum(), 1.0,
                                                   rtol=0.0, atol=1e-12):
        raise ArithmeticError("constructed alias spectrum is not strictly positive and normalized")
    alias_correlator, alias_amplitudes = _correlator(ALIAS_SPECTRUM_MASSES,
                                                     alias_weights)

    rows = []
    for model_id, masses, weights, amplitudes, correlator in (
        ("ground_mass_0p15", first_masses, first_weights, first_amplitudes,
         first_correlator),
        ("ground_mass_0p45", ALIAS_SPECTRUM_MASSES, alias_weights,
         alias_amplitudes, alias_correlator),
    ):
        feature = _feature_vector(correlator)
        scalar = reader.anchor_scalar(correlator)
        rows.append({
            "model_id": model_id,
            "ground_mass": min(masses),
            "masses": list(masses),
            "weights_at_t4": np.asarray(weights, dtype=float).tolist(),
            "amplitudes": np.asarray(amplitudes, dtype=float).tolist(),
            "feature_vector_unique": feature.tolist(),
            "feature_vector_four_time": [
                float(reader.fold_pp_even(correlator)[time]
                      / reader.fold_pp_even(correlator)[4])
                for time in FEATURE_TIMES
            ],
            "reader_m_eff4": scalar,
            "correlator": correlator,
            "normalized_timeseries": (
                reader.fold_pp_even(correlator)
                / reader.fold_pp_even(correlator)[4]
            ),
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(output: Path, rows: list[dict[str, Any]], separation: float) -> None:
    colors = ("#2b78e4", "#d95f02")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), layout="constrained")
    times = np.arange(TEMPORAL_EXTENT // 2 + 1)
    for row, color in zip(rows, colors):
        axes[0].plot(times, row["normalized_timeseries"], marker="o", ms=3,
                     color=color, label=f"ground mass {row['ground_mass']:.2f}")
        axes[1].bar(np.asarray(row["masses"]) + (0.012 if color == colors[0] else -0.012),
                    row["weights_at_t4"], width=0.018, color=color, alpha=0.78,
                    label=f"ground mass {row['ground_mass']:.2f}")
    for time in FEATURE_TIMES:
        axes[0].axvline(time, color="black", alpha=0.10, linewidth=0.8)
    axes[0].set(xlabel="folded Euclidean time t", ylabel=r"$C_{fold}(t)/C_{fold}(4)$",
                title=f"Four-time signatures overlap (max gap={separation:.1e})")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel=r"spectral mass $m$", ylabel=r"$t=4$ contribution weight $w_m$",
                title="Different lowest masses, positive spectral weights")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.savefig(output / "multitime_alias.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _artifact_hashes(output: Path) -> dict[str, str]:
    files = sorted(path for path in output.rglob("*")
                   if path.is_file() and path.name != "artifact_hashes.json")
    hashes = {str(path.relative_to(output)): _sha(path.read_bytes()) for path in files}
    (output / "artifact_hashes.json").write_bytes(_json_bytes({
        "schema": "pion-anchor-multitime-alias-artifact-hashes-v1",
        "artifacts": hashes,
    }))
    return hashes


def build(output: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {out}")
    source_payloads = {name: path.read_bytes() for name, path in SOURCE_PATHS.items()}
    source_hashes = {name: _sha(payload) for name, payload in source_payloads.items()}
    rows = _models()
    feature_gap = np.asarray(rows[0]["feature_vector_unique"]) - np.asarray(
        rows[1]["feature_vector_unique"])
    max_abs_gap = float(np.max(np.abs(feature_gap)))
    l2_gap = float(np.linalg.norm(feature_gap))
    scalar_gap = abs(rows[0]["reader_m_eff4"] - rows[1]["reader_m_eff4"])
    if max_abs_gap > 1e-12 or scalar_gap > 1e-12:
        raise ArithmeticError("the constructed positive spectra do not collide under the frozen map")
    if not (rows[0]["ground_mass"] < rows[1]["ground_mass"]):
        raise ArithmeticError("counterexample ground masses must differ")

    out.mkdir(parents=True)
    summary = {
        "schema": "pion-anchor-multitime-alias-summary-v1",
        "classification": "POSITIVE_PERIODIC_SPECTRA_COLLIDE_ON_FOUR_TIME_FEATURE_AND_SCALAR",
        "physical_score": None,
        "kernel": "K(m,t)=exp(-m*t)+exp(-m*(16-t))",
        "feature_definition": "C_fold(t)/C_fold(4), t in {2,3,5,6}",
        "reflection_symmetry": "C_fold(t)=C_fold(16-t); the four listed times are distinct within t=0..8",
        "models": [{key: value for key, value in row.items()
                    if key not in {"correlator", "normalized_timeseries"}}
                   for row in rows],
        "diagnostic": {
            "max_abs_unique_feature_gap": max_abs_gap,
            "l2_unique_feature_gap": l2_gap,
            "reader_m_eff4_gap": scalar_gap,
            "ground_mass_gap": rows[1]["ground_mass"] - rows[0]["ground_mass"],
            "amplitudes_strictly_positive": all(
                all(value > 0 for value in row["amplitudes"]) for row in rows),
            "t4_weights_strictly_positive": all(
                all(value > 0 for value in row["weights_at_t4"]) for row in rows),
        },
        "denominators": {
            "positive_spectra": {"planned": 2, "valid": 2, "failed": 0, "abstained": 0},
            "feature_vectors": {"planned": 2, "valid": 2, "failed": 0, "abstained": 0},
            "physical_anchor_inputs": {"planned": 0, "valid": 0, "failed": 0, "abstained": 0},
            "physical_gate": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
        },
        "source_hashes": source_hashes,
    }
    (out / "summary.json").write_bytes(_json_bytes(summary))
    spectrum_rows = []
    for row in rows:
        for index, (mass, amplitude, weight) in enumerate(zip(
                row["masses"], row["amplitudes"], row["weights_at_t4"])):
            spectrum_rows.append({"model_id": row["model_id"], "state_index": index,
                                 "mass": mass, "amplitude": amplitude,
                                 "weight_at_t4": weight,
                                 "ground_mass": row["ground_mass"]})
    _write_csv(out / "spectral_components.csv", spectrum_rows)
    timeseries_rows = []
    for row in rows:
        for time, value in enumerate(row["normalized_timeseries"]):
            timeseries_rows.append({"model_id": row["model_id"], "ground_mass": row["ground_mass"],
                                    "time": time, "normalized_correlator": float(value)})
    _write_csv(out / "normalized_timeseries.csv", timeseries_rows)
    protocol = {
        "schema": "pion-anchor-multitime-alias-protocol-v1",
        "classification": summary["classification"],
        "source_hashes": source_hashes,
        "feature_definition": summary["feature_definition"],
        "models": summary["models"],
        "residual_tolerance": 1e-12,
        "construction": "normalize C_fold(4)=1; represent each spectrum by its positive t=4 contribution weights; solve five linear constraints (normalization plus four distinct ratios) for the alias spectrum",
        "physical_score": None,
    }
    (out / "protocol.json").write_bytes(_json_bytes(protocol))
    source_dir = out / "source_snapshot"
    source_dir.mkdir()
    for name, payload in source_payloads.items():
        (source_dir / name.replace("/", "_" )).write_bytes(payload)
    (out / "source_hashes.json").write_bytes(_json_bytes({"sources": source_hashes}))
    _plot(out, rows, max_abs_gap)
    readme = [
        "# Pion anchor: a positive-spectrum alias survives the four-time feature",
        "",
        "**Classification:** constructive noiseless periodic-spectrum counterexample only. It uses finite positive spectral amplitudes and checks the frozen scalar reader; it does not use gauge configurations, an anchor/reference chain, sealed inputs, GPU or QUDA. `physical_score=null`.",
        "",
        "## Construction and result",
        "",
        "The first spectrum has masses `[0.15, 0.60, 0.90, 1.20]`; its positive contributions to `C_fold(4)` are `[0.03, 0.97/3, 0.97/3, 0.97/3]`. A second spectrum with masses `[0.45, 0.70, 1.00, 1.30, 1.50]` is obtained by solving for its five positive `t=4` weights so that normalization and all four distinct ratios match. The actual positive amplitudes are `A_i=w_i/K(m_i,4)`.",
        "",
        f"The lowest masses differ by `{rows[1]['ground_mass'] - rows[0]['ground_mass']:.2f}`. Both spectra match all four listed ratios within `{max_abs_gap:.2e}`, and the production `anchor_scalar` reader returns the same `m_eff(4)` within `{scalar_gap:.2e}`.",
        "",
        "![Positive-spectrum multi-time alias](multitime_alias.png)",
        "",
        "## Interpretation and limits",
        "",
        "This is a direct no-go for treating this particular finite-time feature vector plus positivity and the `[0.1,3.0]` mass support as a globally identifying ground-state map. It does not show that the physical ensemble contains either constructed spectrum. It also does not rule out a useful anchor method under an independently justified finite-state, gap, overlap or amplitude-ratio restriction; such a restriction and a uniform bridge to the candidate-blind PP/A4P `m*` would need evidence and independent review.",
        "",
        "Data flow: `positive periodic spectra -> t=4 contribution weights -> solve finite feature-matching constraints -> verify positive amplitudes -> production scalar and four-time feature agree -> physical gate abstains`. Planned/valid/failed/abstained denominators are in `summary.json`.",
        "",
        "Reproduce from the repository root:",
        "",
        "```bash",
        "python -m tasks.pion_projection.anchor_multitime_alias_diagnostic \\",
        "  --output tasks/pion_projection/anchor_multitime_alias_replay_v1",
        "```",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    artifacts = _artifact_hashes(out)
    return {"output": str(out), "max_abs_feature_gap": max_abs_gap,
            "scalar_gap": scalar_gap, "artifact_count": len(artifacts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
