"""Reproducible CPU pilot for dependence-sensitive D04 reference fits.

All correlators are synthetic. The report compares a joint PP/A4P fit with
same-draw channel ablations and the frozen window/model variants; it cannot
produce a physical reference receipt or certify bootstrap coverage.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.source_integrity import capture_source_snapshot
from tasks.optimize_gaussian.d04_raw_reference import (
    CORRELATOR_SCHEMA,
    MANIFEST_SCHEMA,
    PROTOCOL_SCHEMA,
    evaluate_d04_raw_snapshot,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path(__file__).resolve().parent / "d04_mbb_cpu_20261127_v1"
BLOCK_LENGTHS = (1, 4, 8)
BOOTSTRAP_REPLICATES = 24
BASE_SEED = 2026112715
SYNTHETIC_CONFIGURATIONS = 240
_HERE = Path(__file__).resolve().parent
_SOURCE_PATHS = {
    "reader": ROOT / "tasks/optimize_gaussian/d04_raw_reference.py",
    "diagnostic": _HERE / "d04_block_bootstrap_diagnostic.py",
    "test": ROOT / "tests/test_optimize_d04_raw_reference.py",
    "contract": _HERE / "research_contract_v1.json",
    "npz_safety": ROOT / "core/npz_safety.py",
    "source_integrity": ROOT / "core/source_integrity.py",
}


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _ar1(rng: np.random.Generator, n: int, rho: float) -> np.ndarray:
    values = np.empty(n, dtype=float)
    values[0] = rng.normal()
    innovations = rng.normal(size=n - 1)
    for index, innovation in enumerate(innovations, start=1):
        values[index] = rho * values[index - 1] + np.sqrt(1.0 - rho * rho) * innovation
    return values


def _synthetic_bundle(output: Path) -> dict[str, str]:
    ncfg = SYNTHETIC_CONFIGURATIONS
    config_ids = np.arange(100, 100 + ncfg, dtype=np.int64)
    trajectory_ids = 7 * config_ids + 3
    rng = np.random.default_rng(BASE_SEED)
    time = np.arange(16, dtype=float)
    masses = (0.32, 0.91)
    pp = np.empty((ncfg, 3, 16), dtype=np.float64)
    a4p = np.empty((ncfg, 16), dtype=np.float64)
    amplitudes = ((1.0, 0.22), (0.7, -0.12), (1.3, 0.30))
    for channel, (amp0, amp1) in enumerate(amplitudes):
        fluct0 = _ar1(rng, ncfg, 0.65)
        fluct1 = _ar1(rng, ncfg, 0.65)
        ground = amp0 * (1.0 + 0.015 * fluct0)
        excited = amp1 * (1.0 + 0.04 * fluct1)
        model = (
            ground[:, None] * (np.exp(-masses[0] * time)
                               + np.exp(-masses[0] * (16 - time)))
            + excited[:, None] * (np.exp(-masses[1] * time)
                                  + np.exp(-masses[1] * (16 - time)))
        )
        pp[:, channel, :] = model + rng.normal(0.0, 2e-5, model.shape)
    a_ground = 0.18 * (1.0 + 0.015 * _ar1(rng, ncfg, 0.65))
    a_excited = -0.035 * (1.0 + 0.04 * _ar1(rng, ncfg, 0.65))
    a4_model = (
        a_ground[:, None] * (np.exp(-masses[0] * time)
                             - np.exp(-masses[0] * (16 - time)))
        + a_excited[:, None] * (np.exp(-masses[1] * time)
                                - np.exp(-masses[1] * (16 - time)))
    )
    a4p[:] = a4_model + rng.normal(0.0, 1e-5, a4_model.shape)

    data_buffer = BytesIO()
    np.savez_compressed(
        data_buffer, schema=np.asarray(CORRELATOR_SCHEMA),
        configuration_ids=config_ids, trajectory_ids=trajectory_ids,
        pp=pp, a4p=a4p,
    )
    data_bytes = data_buffer.getvalue()
    protocol = {
        "schema": PROTOCOL_SCHEMA, "geometry": "S16T16",
        "action": "quenched_wilson", "kappa": 0.14,
        "temporal_correlator_boundary": "periodic",
        "pp_order": ["PP_n0", "PP_n8", "PP_n16"],
        "dimension_order": "configuration,channel,time", "temporal_extent": 16,
    }
    protocol_bytes = _json_bytes(protocol)
    manifest = {
        "schema": MANIFEST_SCHEMA, "split_id": "E-synthetic-development-only",
        "geometry": "S16T16", "action": "quenched_wilson", "kappa": 0.14,
        "temporal_correlator_boundary": "periodic",
        "source_protocol_sha256": _sha(protocol_bytes),
        "configuration_ids_sha256": _sha(np.ascontiguousarray(config_ids.astype("<i8")).tobytes()),
        "trajectory_ids_sha256": _sha(np.ascontiguousarray(trajectory_ids.astype("<i8")).tobytes()),
        "correlator_data_sha256": _sha(data_bytes),
        "pp_order": ["PP_n0", "PP_n8", "PP_n16"],
        "dimension_order": "configuration,channel,time",
    }
    files = {
        "manifest.json": _json_bytes(manifest),
        "source_protocol.json": protocol_bytes,
        "synthetic_correlators.npz": data_bytes,
    }
    for name, payload in files.items():
        (output / name).write_bytes(payload)
    return {name: _sha(payload) for name, payload in files.items()}


def _plot(output: Path, result: dict[str, Any]) -> None:
    lengths = BLOCK_LENGTHS
    colors = {1: "#555555", 4: "#2b78e4", 8: "#e69138"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    for index, length in enumerate(lengths):
        summary = result["block_bootstrap"]["by_block_length"][str(length)]["primary_mass"]
        estimate = result["fit_variants"]
        main = next(item["reference_mass"] for item in estimate
                    if item["tmin"] == 3 and item["nstates"] == 2)
        lower, upper = summary["lower_95_percentile"], summary["upper_95_percentile"]
        if lower is None or upper is None:
            ax.scatter(index, main, marker="x", color=colors[length], label=f"L={length}: abstain")
            continue
        ax.vlines(index, lower, upper, color=colors[length], linewidth=2)
        ax.scatter(index, main, color=colors[length],
                   label=f"L={length}; bootstrap SE={summary['bootstrap_se']:.3g}")
    ax.axhline(0.32, color="black", linestyle="--", linewidth=1,
               label="synthetic injected mass")
    ax.set_xticks(range(len(lengths)), [f"L={length}" for length in lengths])
    ax.set_ylabel(r"$a_t m_0$")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_title("Primary joint fit: percentile interval by block length")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    contrast_ids = (
        "pp_only_tmin3_nstates2", "a4p_only_tmin3_nstates2",
        "joint_tmin2_nstates2", "joint_tmin2_nstates3",
        "joint_tmin3_nstates3", "joint_tmin4_nstates2",
        "joint_tmin4_nstates3",
    )
    labels = (
        "PP only", "A4P only", "joint window tmin=2, 2 states",
        "joint window tmin=2, 3 states", "joint window tmin=3, 3 states",
        "joint window tmin=4, 2 states", "joint window tmin=4, 3 states",
    )
    y = np.arange(len(contrast_ids))
    for offset, length in enumerate(lengths):
        contrasts = result["block_bootstrap"]["by_block_length"][str(length)]["contrasts"]
        shift = [contrasts[key]["paired_shift_over_se"] for key in contrast_ids]
        x = [2.68 if value is None else value for value in shift]
        markers = ["x" if value is None else "o" for value in shift]
        for value, xpos, ypos, marker in zip(shift, x, y + (offset - 1) * 0.18, markers):
            ax.scatter([xpos], [ypos], color=colors[length], marker=marker,
                       label=f"L={length}" if ypos == y[0] + (offset - 1) * 0.18 else None)
    ax.axvline(2.5, color="#b00020", linestyle="--", linewidth=1,
               label="2.5 reference-contract diagnostic threshold")
    ax.scatter([], [], color="black", marker="x", label="x = abstained interval (position only)")
    ax.set_xlim(left=0, right=2.9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("absolute point shift / paired bootstrap SE")
    ax.set_title("Paired shifts vs primary joint (tmin=3, 2 states)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.suptitle("D04 synthetic CPU pilot · conditional MBB only · not a gate")
    fig.tight_layout()
    fig.savefig(output / "mbb_diagnostics.png", dpi=160)
    plt.close(fig)


def _summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for length in BLOCK_LENGTHS:
        item = result["block_bootstrap"]["by_block_length"][str(length)]
        main = item["primary_mass"]
        rows.append({
            "metric": "primary_joint_mass", "block_length": length,
            "planned": main["draw_denominator"]["planned"],
            "valid": main["draw_denominator"]["valid"],
            "failed": main["draw_denominator"]["failed"],
            "abstained": main["denominator"]["abstained"],
            "estimate": next(
                row["reference_mass"] for row in result["fit_variants"]
                if row["tmin"] == 3 and row["nstates"] == 2
            ),
            "lower": main["lower_95_percentile"], "upper": main["upper_95_percentile"],
            "bootstrap_se": main["bootstrap_se"],
            "fit_planned": item["fit_denominators"]["joint_tmin3_nstates2"]["planned"],
            "fit_valid": item["fit_denominators"]["joint_tmin3_nstates2"]["valid"],
            "fit_failed": item["fit_denominators"]["joint_tmin3_nstates2"]["failed"],
            "fit_abstained": item["fit_denominators"]["joint_tmin3_nstates2"]["abstained"],
        })
        for variant_id, contrast in item["contrasts"].items():
            interval = contrast["bootstrap_paired_shift"]
            fit_denominator = item["fit_denominators"][variant_id]
            rows.append({
                "metric": f"paired_shift:{variant_id}", "block_length": length,
                "planned": interval["draw_denominator"]["planned"],
                "valid": interval["draw_denominator"]["valid"],
                "failed": interval["draw_denominator"]["failed"],
                "abstained": interval["denominator"]["abstained"],
                "estimate": contrast["point_shift_from_primary"],
                "lower": interval["lower_95_percentile"],
                "upper": interval["upper_95_percentile"],
                "bootstrap_se": interval["bootstrap_se"],
                "abs_shift_over_se": contrast["paired_shift_over_se"],
                "fit_planned": fit_denominator["planned"],
                "fit_valid": fit_denominator["valid"],
                "fit_failed": fit_denominator["failed"],
                "fit_abstained": fit_denominator["abstained"],
            })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(output: Path, result: dict[str, Any], source_hashes: dict[str, str]) -> None:
    lines = [
        "# D04 circular moving-block bootstrap pilot",
        "",
        "**Status:** synthetic CPU method diagnostic only. No physical E-chain bytes, sealed data, GPU/QUDA, reference receipt, or certification were used.",
        "",
        "## Frozen setup",
        "",
        f"- Synthetic configurations: {SYNTHETIC_CONFIGURATIONS}; two-state truth `a_t m=(0.32, 0.91)`. Ground/excited amplitude fluctuations use a stationary AR(1) generator with `rho=0.65`; additive noise is independent.",
        f"- Circular moving-block bootstrap lengths: `{list(BLOCK_LENGTHS)}`; `{BOOTSTRAP_REPLICATES}` replicates per length; seed base `{BASE_SEED}`, per-length seed = base + L.",
        "- `L=1` is the IID bootstrap baseline; `L=4,8` are declared sensitivity choices, not estimated decorrelation lengths. Each resample is applied once and shared by joint, PP-only, A4P-only, window, and state-count fits.",
        "- The 95% intervals are empirical percentile summaries from a small pilot. They carry no finite-sample coverage guarantee; L selection is not tuned to these results.",
        "",
        "## Diagnostic flow",
        "",
        "```text",
        "synthetic ordered PP/A4P bytes -> immutable source snapshot -> fold + shared covariance ->",
        "circular block resample (same indices for all fits) -> fit every variant -> retain failed draws ->",
        "primary mass interval + paired channel/window/model shifts -> diagnostic only / physical gate abstains",
        "```",
        "",
        "## Results and denominators",
        "",
        "See `summary.csv` for each block length and contrast. Every fit records `planned/valid/failed/abstained`; each paired shift requires both fits from the same resample. The `tmin=2`, three-state contrast has 21/24 valid paired draws at `L=1`, so its interval abstains under the frozen 90% reporting rule; other lengths retain 23/24 valid draws. The 2.5 paired-SE line is shown only as a reference-contract diagnostic, not a pass/fail decision. A plotted `x` marks an abstained interval at a visual placeholder x-position; it is not a ratio value.",
        "",
        "![Primary mass interval and paired fit shifts](mbb_diagnostics.png)",
        "",
        "## Interpretation and limits",
        "",
        "This pilot makes dependence sensitivity visible: the same synthetic input is analyzed with three fixed block sizes, while PP/A4P ablations and window/model changes are paired within each draw. A narrow interval at one L does not identify true dependence; the generator has a known AR(1) amplitude process, but production chain order, stationarity, gap, and dependence scale are not evidenced. An A4P-only shift is a diagnostic ablation and is not a replacement for the contract's joint PP/A4P reference.",
        "",
        "The MBB percentiles do not establish 95% coverage, and the toy data do not validate the production source/sink or sign conventions. D04 remains `NOT_READY/UNCERTIFIED`; the source/reference/authorization gates remain open, sealed data stay closed, and `physical_score=null`.",
        "",
        "`summary.csv` and the figure can be rebuilt from the saved `summary.json` with `--refresh-derived`; that operation verifies the existing artifact hashes first and does not refit or reread inputs.",
        "",
        "## Source identity",
        "",
        "The report's `source_hashes.json` binds the reader, report generator, test, selected contract, shared NPZ preflight, and source-snapshot implementation used at generation time. Hashes identify bytes only; they do not validate scientific correctness or authority.",
        "",
        "| source | SHA-256 |",
        "|---|---|",
    ]
    lines.extend(f"| `{name}` | `{digest}` |" for name, digest in source_hashes.items())
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(output: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {out}")
    out.mkdir(parents=True)
    source_hashes = {name: _sha(path.read_bytes()) for name, path in _SOURCE_PATHS.items()}
    _synthetic_bundle(out)
    expected = {name: _sha((out / filename).read_bytes()) for name, filename in (
        ("manifest", "manifest.json"),
        ("source_protocol", "source_protocol.json"),
        ("correlator_data", "synthetic_correlators.npz"),
    )}
    snapshot = capture_source_snapshot(
        out, expected,
        {"manifest": "manifest.json", "source_protocol": "source_protocol.json",
         "correlator_data": "synthetic_correlators.npz"},
    )
    result = evaluate_d04_raw_snapshot(
        snapshot, block_bootstrap_replicates=BOOTSTRAP_REPLICATES,
        block_lengths=BLOCK_LENGTHS, block_bootstrap_seed=BASE_SEED,
    )
    (out / "summary.json").write_bytes(_json_bytes(result))
    _write_csv(out / "summary.csv", _summary_rows(result))
    _plot(out, result)
    (out / "source_hashes.json").write_bytes(_json_bytes({
        "schema": "d04-mbb-source-hashes-v1", "sources": source_hashes,
    }))
    _write_readme(out, result, source_hashes)
    hashes = _artifact_hash_map(out)
    (out / "artifact_hashes.json").write_bytes(_json_bytes({
        "schema": "d04-mbb-artifact-hashes-v1", "artifacts": hashes,
    }))
    return {
        "output": str(out),
        "artifact_count_hashed": len(hashes),
        "source_count": len(source_hashes),
        "block_lengths": list(BLOCK_LENGTHS),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "status": result["status"],
    }


def _artifact_hash_map(output: Path) -> dict[str, str]:
    artifact_paths = sorted(path for path in output.iterdir()
                            if path.is_file() and path.name != "artifact_hashes.json")
    return {path.name: _sha(path.read_bytes()) for path in artifact_paths}


def refresh_derived(output: str | Path) -> dict[str, Any]:
    """Refresh CSV/plot/text from the frozen summary without rerunning fits."""
    out = Path(output)
    manifest_path = out / "artifact_hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "d04-mbb-artifact-hashes-v1":
        raise ValueError("derived refresh requires the expected artifact hash schema")
    current = _artifact_hash_map(out)
    if current != manifest.get("artifacts"):
        raise ValueError("derived refresh refused: existing report artifacts fail hash verification")
    result = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    if (result.get("schema") != "optimize-d04-joint-fit-diagnostic-v1"
            or "block_bootstrap" not in result):
        raise ValueError("derived refresh requires the frozen D04 MBB summary")
    source_hashes = {name: _sha(path.read_bytes()) for name, path in _SOURCE_PATHS.items()}
    _write_csv(out / "summary.csv", _summary_rows(result))
    _plot(out, result)
    (out / "source_hashes.json").write_bytes(_json_bytes({
        "schema": "d04-mbb-source-hashes-v1", "sources": source_hashes,
    }))
    _write_readme(out, result, source_hashes)
    hashes = _artifact_hash_map(out)
    manifest_path.write_bytes(_json_bytes({
        "schema": "d04-mbb-artifact-hashes-v1", "artifacts": hashes,
    }))
    return {"output": str(out), "artifacts_verified_before_refresh": len(current),
            "artifact_count_hashed": len(hashes), "refit": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--refresh-derived", action="store_true")
    args = parser.parse_args()
    result = refresh_derived(args.output) if args.refresh_derived else build(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
