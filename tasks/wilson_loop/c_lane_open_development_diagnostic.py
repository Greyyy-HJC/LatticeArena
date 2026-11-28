"""Retrospective C-lane preflight on the already-open Wilson D development split.

This tool reads exactly one frozen development transaction (IDs 0--99). It
does not read validation, gap, holdout, or confirmation files and cannot emit a
physical certificate. The input hashes are intentionally pinned to that
transaction so a different split fails closed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import source_integrity as source_integrity_module
from core.io import fingerprint
from core.source_integrity import SourceByteSnapshot
from core.npz_safety import validate_npz_payload
from core import npz_safety as npz_safety_module


GENERATION = "dd38c39f21982307c5abc2d5093b93c2aa12ef8848d6f9136519b1f8054b619e"
EXPECTED_NPZ_SHA256 = "da8cc5d72481475a8cd6d821692db2e21555c0bd5b1849747dc75a946dbd7209"
EXPECTED_PROVENANCE_SHA256 = "082ad7060e776f6e724ee02ac1de9c94e706332eb875019f3b9d1c45d4e9e82b"
EXPECTED_LOOPS_SHA256 = "513b67dde074d96c89c82395151a0079ca746be63c7dd5ac5c6ea1ae6d243dcc"
# These hashes bind the development package's measurement implementation.
EXPECTED_SOURCE_SHA256 = {
    "measure.py": "5fdee931db8a0b9bc74debb6f8c53e659ec6ccef7a42b0443788db4593e84ef6",
    "kernels.py": "a6b54eb606f1b7c790de596185d8c0190601fde4495d9f26eba89ba16abdd7dc",
    "stout.py": "7d9f5f8ea0c0e006b09f58abc087e5825fd74c5efecf085db176c7c23ed69f35",
    "statistics.py": "27c691dad427deb5e46447e8a9d6de6eea9027b65d5c99f0290f8ec2590f1c86",
    "common.py": "74de1e8cfaafbae2744855e85992561681057832f52bf537255f9d35230dd29e",
    "identity.py": "d15841a933c22502a616a9e3d56d12dff13dc11227d26343a3da325f03b65fbe",
}
KERNEL_ID = "stout_rho0.1_n5"
R_VALUES = (1, 2, 3, 4)
T_VALUES = (4, 5, 6, 7, 8)
BLOCK_LENGTHS = (2, 5, 10)
N_DRAWS = 10_000
SEED_BASE = 20_261_125
ALPHA_LANE = 0.025
INVALID_LIMIT = 0.01
HALFWIDTH_TARGET = 0.02


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_transaction_bytes(npz_bytes: bytes, provenance_bytes: bytes,
                             commit_bytes: bytes, *, read_mode: str):
    """Validate the pinned open-development transaction from retained bytes."""
    npz_sha = hashlib.sha256(npz_bytes).hexdigest()
    provenance_sha = hashlib.sha256(provenance_bytes).hexdigest()
    if npz_sha != EXPECTED_NPZ_SHA256:
        raise ValueError("development.npz hash differs from the pinned open D transaction")
    if provenance_sha != EXPECTED_PROVENANCE_SHA256:
        raise ValueError("provenance hash differs from the pinned open D transaction")

    try:
        commit = json.loads(commit_bytes)
        provenance = json.loads(provenance_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("development transaction metadata is not valid JSON") from exc
    if commit.get("split") != "development" or commit.get("generation") != GENERATION:
        raise ValueError("transaction marker is not the pinned development generation")
    if commit.get("npz_sha256") != npz_sha or commit.get("provenance_sha256") != provenance_sha:
        raise ValueError("transaction marker does not bind the development files")
    if commit.get("loops_sha256") != EXPECTED_LOOPS_SHA256:
        raise ValueError("transaction marker has an unexpected loop-array fingerprint")
    if provenance.get("generation") != GENERATION:
        raise ValueError("provenance generation mismatch")

    protocol = provenance.get("protocol", {})
    if protocol.get("development") != [0, 99]:
        raise ValueError("development ID range differs from the pinned open range")
    if protocol.get("geometry") != [16, 16, 16, 16] or protocol.get("beta") != 6.0:
        raise ValueError("unexpected geometry or beta in development provenance")
    if protocol.get("boundary_conditions") != "periodic_all_directions":
        raise ValueError("unexpected boundary conditions in development provenance")
    if protocol.get("r_values") != list(R_VALUES) or protocol.get("t_values") != list(T_VALUES):
        raise ValueError("R/T axes do not match the selected C-lane stencil")
    for name, expected in EXPECTED_SOURCE_SHA256.items():
        observed = protocol.get("source_hashes", {}).get(name)
        if observed != expected:
            raise ValueError(f"protocol source fingerprint mismatch for {name}")
        if name != "stout.py" and provenance.get("dependencies", {}).get(name) != expected:
            raise ValueError(f"provenance dependency fingerprint mismatch for {name}")
    kernels = protocol.get("kernels", [])
    matches = [i for i, item in enumerate(kernels) if item.get("name") == KERNEL_ID]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one frozen kernel named {KERNEL_ID}")
    kernel = kernels[matches[0]]
    if (kernel.get("kind"), kernel.get("rho"), kernel.get("steps")) != ("stout", 0.1, 5):
        raise ValueError("kernel parameters do not match the selected stout C lane")

    validate_npz_payload(npz_bytes)
    with np.load(BytesIO(npz_bytes), allow_pickle=False) as archive:
        ids = np.array(archive["ids"], copy=True)
        loops = np.array(archive["loops"], copy=True)
        archive_generation = str(archive["generation"].item())
        protocol_hash = str(archive["protocol_hash"].item())
    if archive_generation != GENERATION:
        raise ValueError("NPZ generation does not match the transaction marker")
    if protocol_hash != fingerprint(protocol):
        raise ValueError("NPZ protocol fingerprint differs from its captured provenance")
    expected_ids = np.arange(100, dtype=np.int64)
    if not np.array_equal(ids, expected_ids):
        raise ValueError("the open development NPZ must contain exactly IDs 0--99")
    if loops.shape != (100, len(kernels), len(R_VALUES), len(T_VALUES)):
        raise ValueError(f"unexpected loop array shape {loops.shape}")
    if not np.all(np.isfinite(loops)):
        raise ValueError("development loop array contains nonfinite values")
    loops_sha = hashlib.sha256(np.ascontiguousarray(loops).tobytes()).hexdigest()
    if loops_sha != EXPECTED_LOOPS_SHA256:
        raise ValueError("loop-array fingerprint differs from the transaction marker")
    records = provenance.get("records", [])
    record_ids = [row.get("id") for row in records]
    if record_ids != expected_ids.tolist():
        raise ValueError("provenance records do not match development IDs 0--99")
    trajectories = [row.get("hmc_trajectory") for row in records]
    if any(type(value) is not int for value in trajectories):
        raise ValueError("development provenance is missing HMC trajectory metadata")

    data_hashes = {
        "development_npz_sha256": npz_sha,
        "development_provenance_sha256": provenance_sha,
        "development_commit_sha256": hashlib.sha256(commit_bytes).hexdigest(),
        "loops_array_sha256": loops_sha,
        "read_mode": read_mode,
    }
    return ids, loops[:, matches[0], :, :], protocol_hash, trajectories, data_hashes


def load_transaction(input_dir: Path | None = None, *,
                     source_snapshot: SourceByteSnapshot | None = None,
                     source_names: dict[str, str] | None = None):
    """Load the open D-development transaction from paths or captured bytes.

    Snapshot mode reads exactly the three named development files and has no
    path fallback. Validation, gap, holdout, and confirmation are not valid
    inputs to this reader.
    """
    expected_names = {
        "development_npz", "development_provenance", "development_commit",
    }
    if source_snapshot is not None:
        if input_dir is not None:
            raise ValueError("snapshot mode does not accept an input directory")
        if not isinstance(source_snapshot, SourceByteSnapshot):
            raise ValueError("source_snapshot must be a verified SourceByteSnapshot")
        if not isinstance(source_names, dict) or set(source_names) != expected_names:
            raise ValueError("source_names must map exactly the three development transaction files")
        try:
            payloads = {
                role: source_snapshot.read_bytes(name)
                for role, name in source_names.items()
            }
        except ValueError as exc:
            raise ValueError(f"incomplete captured development transaction: {exc}") from exc
        return _parse_transaction_bytes(
            payloads["development_npz"], payloads["development_provenance"],
            payloads["development_commit"], read_mode="immutable_snapshot",
        )

    if source_names is not None:
        raise ValueError("source_names is only valid together with source_snapshot")
    if input_dir is None:
        raise ValueError("input_dir is required unless source_snapshot is supplied")
    input_dir = Path(input_dir)
    path_by_role = {
        "development_npz": input_dir / "development.npz",
        "development_provenance": input_dir / "development_provenance.json",
        "development_commit": input_dir / "development.commit.json",
    }
    payloads = {}
    for role, path in path_by_role.items():
        try:
            payloads[role] = path.read_bytes()
        except FileNotFoundError as exc:
            raise ValueError(
                f"required development transaction file is missing: {path.name}"
            ) from exc
    return _parse_transaction_bytes(
        payloads["development_npz"], payloads["development_provenance"],
        payloads["development_commit"], read_mode="path",
    )


def creutz(samples: np.ndarray) -> np.ndarray:
    """Apply -[log W(3,6)+log W(2,5)-log W(3,5)-log W(2,6)]."""
    return -np.log(samples[..., 2, 2]) - np.log(samples[..., 1, 1]) \
        + np.log(samples[..., 2, 1]) + np.log(samples[..., 1, 2])


def circular_block_indices(n: int, length: int, draws: int, seed: int) -> np.ndarray:
    blocks_per_draw = math.ceil(n / length)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(draws, blocks_per_draw))
    offsets = np.arange(length, dtype=np.int64)
    return ((starts[..., None] + offsets) % n).reshape(draws, -1)[:, :n]


def analyze(values: np.ndarray, *, seed_base: int = SEED_BASE):
    if isinstance(seed_base, bool) or not isinstance(seed_base, int) or seed_base < 0:
        raise ValueError("seed_base must be a nonnegative integer")
    means = values.mean(axis=0)
    required_means = np.asarray([means[2, 2], means[1, 1], means[2, 1], means[1, 2]])
    point_valid = bool(np.all(np.isfinite(required_means)) and np.all(required_means > 0))
    point = float(creutz(means)) if point_valid else None
    draws_by_length = {}
    rows = []
    intervals = []
    qlo, qhi = ALPHA_LANE / 2, 1 - ALPHA_LANE / 2
    for length in BLOCK_LENGTHS:
        seed = seed_base + length
        indices = circular_block_indices(len(values), length, N_DRAWS, seed)
        sampled = values[indices]
        draw_means = sampled.mean(axis=1)
        required_draw_means = np.stack([draw_means[:, 2, 2], draw_means[:, 1, 1],
                                        draw_means[:, 2, 1], draw_means[:, 1, 2]], axis=-1)
        valid = np.all(np.isfinite(required_draw_means) & (required_draw_means > 0), axis=1)
        estimates = np.full(N_DRAWS, np.nan, dtype=np.float64)
        if np.any(valid):
            estimates[valid] = creutz(draw_means[valid])
        valid &= np.isfinite(estimates)
        estimates[~valid] = np.nan
        valid_count = int(np.count_nonzero(valid))
        invalid_count = N_DRAWS - valid_count
        invalid_fraction = invalid_count / N_DRAWS
        if valid_count:
            lower, upper = (float(x) for x in np.quantile(estimates[valid], [qlo, qhi], method="linear"))
            halfwidth = (upper - lower) / 2
            interval_row = {"block_length": length, "lower": lower, "upper": upper,
                            "halfwidth": halfwidth, "seed": seed,
                            "planned_draws": N_DRAWS, "valid_draws": valid_count,
                            "invalid_draws": invalid_count, "invalid_fraction": invalid_fraction}
            intervals.append(interval_row)
        else:
            lower = upper = halfwidth = None
            interval_row = {"block_length": length, "lower": None, "upper": None,
                            "halfwidth": None, "seed": seed,
                            "planned_draws": N_DRAWS, "valid_draws": 0,
                            "invalid_draws": N_DRAWS, "invalid_fraction": 1.0}
        rows.append(interval_row)
        draws_by_length[f"L{length}"] = estimates

    if len(intervals) == len(BLOCK_LENGTHS):
        hull_low = min(row["lower"] for row in intervals)
        hull_high = max(row["upper"] for row in intervals)
        hull_halfwidth = (hull_high - hull_low) / 2
    else:
        hull_low = hull_high = hull_halfwidth = None
    invalid_pass = all(row["invalid_fraction"] <= INVALID_LIMIT for row in rows)
    precision_pass = hull_halfwidth is not None and hull_halfwidth <= HALFWIDTH_TARGET
    empirical_gate_pass = bool(point_valid and invalid_pass and precision_pass)
    for row in rows:
        row.update({"alpha_lane": ALPHA_LANE,
                    "interval_level": 1 - ALPHA_LANE,
                    "precision_target_halfwidth": HALFWIDTH_TARGET,
                    "hull_lower": hull_low,
                    "hull_upper": hull_high,
                    "hull_halfwidth": hull_halfwidth,
                    "invalid_gate_pass": row["invalid_fraction"] <= INVALID_LIMIT,
                    "precision_gate_pass": precision_pass,
                    "empirical_numeric_gate_pass": empirical_gate_pass})
    return means, point, point_valid, rows, draws_by_length, hull_low, hull_high, hull_halfwidth, empirical_gate_pass


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def make_plot(path: Path, point, point_valid, rows, hull_low, hull_high, hull_halfwidth):
    fig, (ax, ax_width, ax_invalid) = plt.subplots(
        3, 1, figsize=(9.2, 8.2), gridspec_kw={"height_ratios": [3, 1.2, 1]},
        constrained_layout=True)
    if point_valid:
        ax.axvline(point, color="#222222", linewidth=1.4, label=f"point = {point:.6f}")
    y_positions = [3, 2, 1]
    for y, row in zip(y_positions, rows):
        if row["lower"] is not None:
            ax.plot([row["lower"], row["upper"]], [y, y], color="#286d9b", linewidth=3)
            ax.plot([row["lower"], row["upper"]], [y, y], "|", color="#163e59", markersize=11)
            ax.scatter([(row["lower"] + row["upper"]) / 2], [y], s=28, color="#163e59", zorder=3)
        else:
            ax.text(0.02, y, "all draws invalid", transform=ax.get_yaxis_transform(), va="center")
    if hull_low is not None:
        ax.plot([hull_low, hull_high], [0, 0], color="#bd5b36", linewidth=5,
                label=f"interval hull, half-width={hull_halfwidth:.5f}")
        ax.plot([hull_low, hull_high], [0, 0], "|", color="#7b2f17", markersize=13)
    ax.set_yticks([0, 1, 2, 3], ["hull", "L=10", "L=5", "L=2"])
    ax.set_xlabel(r"finite-$T$ $\chi_{stout}(R=2,T=5)$ (dimensionless)")
    ax.set_title("Open D development sample: C-lane retrospective preflight")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    width_labels = ["L=2", "L=5", "L=10", "hull"]
    width_values = [row["halfwidth"] for row in rows] + [hull_halfwidth]
    width_positions = [3, 2, 1, 0]
    for y, value in zip(width_positions, width_values):
        if value is not None:
            ax_width.scatter([value], [y], color="#286d9b" if y else "#bd5b36", zorder=3)
            ax_width.plot([0, value], [y, y], color="#286d9b" if y else "#bd5b36", alpha=0.6)
    ax_width.axvline(HALFWIDTH_TARGET, color="#bd3b32", linestyle="--", linewidth=1.3,
                     label=f"half-width limit = {HALFWIDTH_TARGET:.3f}")
    ax_width.set_yticks(width_positions, width_labels)
    ax_width.set_xlabel("interval half-width (dimensionless)")
    ax_width.grid(axis="x", alpha=0.25)
    ax_width.legend(loc="best", fontsize=8)
    ax_invalid.axhline(100 * INVALID_LIMIT, color="#bd3b32", linestyle="--", linewidth=1.3,
                       label="1% invalid-draw limit")
    ax_invalid.bar([row["block_length"] for row in rows],
                   [100 * row["invalid_fraction"] for row in rows],
                   width=1.1, color="#d89255")
    ax_invalid.set_xticks([2, 5, 10], ["2", "5", "10"])
    ax_invalid.set_xlabel("circular block length (configurations)")
    ax_invalid.set_ylabel("invalid draws (%)")
    ax_invalid.set_ylim(bottom=0, top=max(1.2, max(100 * row["invalid_fraction"] for row in rows) * 1.25))
    ax_invalid.grid(axis="y", alpha=0.2)
    ax_invalid.legend(loc="upper right", fontsize=8)
    fig.text(0.5, -0.01,
             "Retrospective development only; intervals do not certify coverage or a physical score.",
             ha="center", fontsize=9, color="#8a2f20")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(input_dir: Path | None, output_dir: Path, *,
        source_snapshot: SourceByteSnapshot | None = None,
        source_names: dict[str, str] | None = None):
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    source_path = Path(__file__).resolve()
    source_bytes = source_path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    source_integrity_path = Path(source_integrity_module.__file__).resolve()
    source_integrity_bytes = source_integrity_path.read_bytes()
    source_integrity_sha = hashlib.sha256(source_integrity_bytes).hexdigest()
    npz_safety_path = Path(npz_safety_module.__file__).resolve()
    npz_safety_bytes = npz_safety_path.read_bytes()
    npz_safety_sha = hashlib.sha256(npz_safety_bytes).hexdigest()

    ids, loops, protocol_hash, trajectories, hashes = load_transaction(
        input_dir, source_snapshot=source_snapshot, source_names=source_names
    )
    means, point, point_valid, rows, draws, hull_low, hull_high, hull_halfwidth, gate_pass = analyze(loops)
    output_dir.mkdir(parents=True, exist_ok=False)

    point_means = {
        "W(3,6)": float(means[2, 2]),
        "W(2,5)": float(means[1, 1]),
        "W(3,5)": float(means[2, 1]),
        "W(2,6)": float(means[1, 2]),
    }
    protocol = {
        "schema": "wilson-c-lane-open-development-diagnostic-v1",
        "status": "RETROSPECTIVE_DEVELOPMENT_ONLY_NOT_CONFIRMATION",
        "input_split": "development IDs 0-99 only",
        "input_generation": GENERATION,
        "input_hashes": hashes,
        "input_protocol_hash": protocol_hash,
        "geometry": [16, 16, 16, 16],
        "beta": 6.0,
        "boundary_conditions": "periodic_all_directions",
        "temporal_extent_Nt": 16,
        "kernel_id": KERNEL_ID,
        "kernel_parameters": {"kind": "stout", "rho": 0.1, "steps": 5},
        "R": 2,
        "T": 5,
        "formula": "-[log Wbar(3,6)+log Wbar(2,5)-log Wbar(3,5)-log Wbar(2,6)]",
        "loop_mean_rule": "paired configuration/block means first; take logs only when all four means are finite and positive",
        "resampling": {
            "method": "circular moving-block bootstrap over saved-configuration order",
            "same_indices_for_all_four_loop_means": True,
            "block_lengths": list(BLOCK_LENGTHS),
            "draws_per_length": N_DRAWS,
            "seed_rule": f"seed = {SEED_BASE} + block_length",
            "interval": "equal-tail percentile interval at lane alpha=0.025; quantiles 0.0125 and 0.9875",
            "hull": "minimum lower endpoint to maximum upper endpoint across L=2,5,10",
            "invalid_draw": "any required loop mean nonfinite or nonpositive; retained in planned denominator",
        },
        "numeric_gates": {
            "invalid_fraction_max": INVALID_LIMIT,
            "interval_hull_halfwidth_max": HALFWIDTH_TARGET,
            "interpretation": "empirical preflight checks only; bootstrap assumptions/coverage are not established",
        },
        "source_script_sha256": source_sha,
        "analysis_source_sha256": {
            "tasks/wilson_loop/c_lane_open_development_diagnostic.py": source_sha,
            "core/npz_safety.py": npz_safety_sha,
            "core/source_integrity.py": source_integrity_sha,
        },
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
    }
    summary = {
        "status": "RETROSPECTIVE_DEVELOPMENT_ONLY_NOT_CONFIRMATION",
        "physical_score": None,
        "certified": False,
        "planned_configurations": int(len(ids)),
        "configuration_ids": [int(ids[0]), int(ids[-1])],
        "hmc_trajectory_range": [int(min(trajectories)), int(max(trajectories))],
        "hmc_trajectory_spacing_min_max": [int(min(np.diff(trajectories))), int(max(np.diff(trajectories)))],
        "kernel_id": KERNEL_ID,
        "observable": "absolute finite-T Creutz chi_stout(R=2,T=5), not ground-state force",
        "point_estimate": point,
        "point_estimate_valid": point_valid,
        "four_required_loop_means": point_means,
        "block_intervals": rows,
        "interval_hull": {"lower": hull_low, "upper": hull_high, "halfwidth": hull_halfwidth},
        "invalid_draws": {"planned": len(rows) * N_DRAWS,
                           "valid": sum(row["valid_draws"] for row in rows),
                           "invalid": sum(row["invalid_draws"] for row in rows),
                           "abstained": 0},
        "precision_gate_outcome": "PASS" if gate_pass else "FAIL_OR_ABSTAIN",
        "empirical_numeric_gates_pass": gate_pass,
        "independent_calibration_gate": "NOT_MET: D0-99 was already observed and used for development; retrospective re-analysis is not confirmation",
        "bootstrap_coverage_gate": "NOT_ESTABLISHED: block lengths and draw count do not prove dependence assumptions or interval coverage",
        "f_lane_status": "UNCHANGED_UNCERTIFIED: this package has Nt=16 loops only; it has no independent GEVP receipt or matched Nt=32 thermal data",
        "data_hashes": hashes,
        "analysis_source_sha256": protocol["analysis_source_sha256"],
    }

    write_json(output_dir / "protocol.json", protocol)
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "intervals.csv").open("w", newline="") as stream:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output_dir / "bootstrap_estimates.npz", **draws)
    (output_dir / "snapshot_c_lane_open_development_diagnostic.py").write_bytes(
        source_bytes
    )
    (output_dir / "snapshot_core_npz_safety.py").write_bytes(npz_safety_bytes)
    (output_dir / "snapshot_core_source_integrity.py").write_bytes(
        source_integrity_bytes
    )
    make_plot(output_dir / "c_lane_bootstrap_intervals.png", point, point_valid,
              rows, hull_low, hull_high, hull_halfwidth)

    readme = f"""# Wilson C-lane open-development preflight

**Status:** `RETROSPECTIVE_DEVELOPMENT_ONLY_NOT_CONFIRMATION`; `physical_score=null`; no physical gate is closed by this report.

## Data boundary and identity

This run reads only the pinned D-chain development transaction, IDs 0–99: `development.npz`, `development_provenance.json`, and `development.commit.json`. The NPZ, provenance, generation, loop-array fingerprint, and frozen source fingerprints are checked before analysis. Validation, gap, shared holdout, and confirmation files are never opened. Input hashes are recorded in `summary.json` and `protocol.json`; bootstrap estimates plus snapshots of the diagnostic script and exact `core/npz_safety.py` preflight source are covered by `manifest.json` and `analysis_source_sha256`.

The open package is 16^4, beta=6, periodic in all directions, with `N_t=16`. The selected spatial kernel is `{KERNEL_ID}` (stout rho=0.1, five steps). Four loop means use the same IDs and, in each bootstrap draw, the same sampled configuration indices:

```text
chi(2,5) = -[log Wbar(3,6) + log Wbar(2,5) - log Wbar(3,5) - log Wbar(2,6)]
```

## Procedure and denominator

The report source map binds the task reader, NPZ preflight, and shared source-snapshot helper; the exact helper bytes are retained and covered by `manifest.json`.

The script forms the four full-sample means, then uses 10,000 circular moving-block bootstrap draws at each block length `L=2,5,10`. The deterministic seed is `20,261,125 + L`. Every draw resamples exactly 100 saved-configuration positions and applies the formula only if all four required means are finite and positive. Invalid draws stay in the planned denominator. Each interval is the equal-tail 97.5% percentile interval (lane alpha=0.025); the reported uncertainty envelope is the hull over the three block-length intervals.

| Quantity | Result |
|---|---:|
| Configurations planned | {len(ids)} |
| Full-sample four-loop means positive | {point_valid} |
| Point estimate | {"not defined" if point is None else f"{point:.8f}"} |
| Bootstrap draws planned / valid / invalid | {len(rows)*N_DRAWS} / {sum(row['valid_draws'] for row in rows)} / {sum(row['invalid_draws'] for row in rows)} |
| Hull lower / upper | {"not defined" if hull_low is None else f"{hull_low:.8f} / {hull_high:.8f}"} |
| Hull half-width; target | {"not defined" if hull_halfwidth is None else f"{hull_halfwidth:.8f}"} ; {HALFWIDTH_TARGET:.3f} |
| Empirical numeric preflight | {"PASS" if gate_pass else "FAIL / ABSTAIN"} |

![C-lane intervals and invalid-draw counts](c_lane_bootstrap_intervals.png)

## Interpretation and limits

The interval hull checks the selected numeric precision rule on an already-observed development sample. Even if its width is below 0.02 and invalid draws are at most 1%, it does **not** establish bootstrap coverage, finite dependence, a plateau, a ground-state force, or an independent calibration result. D0–99 has already been used for development and possible selection, so it cannot be relabelled as confirmation. A separately authorized, previously unobserved calibration package is still required for an evidence-bearing C-lane measurement.

This is the absolute finite-`T` Creutz observable at `R=2,T=5`. It is not the F-lane ground-state discrete static force. The package contains only `N_t=16` loop data; it has no independent 3x3 GEVP/spectral receipt or matched `N_t=32` thermal data, so the F lane remains `GROUND_STATE_FORCE_UNCERTIFIED`.

Text data flow: `pinned open D development transaction -> verify hashes/split/IDs/kernel -> four paired loop means -> circular L=2/5/10 resampling -> keep invalid draws in denominator -> 97.5% intervals -> interval hull and numeric gates -> retrospective-only report; independent calibration and physical certification remain NOT_READY`.

Rerun from the repository root with the open package path supplied explicitly:

```bash
python -m tasks.wilson_loop.c_lane_open_development_diagnostic \\
  --input-dir /path/to/results/wilson_loop/s16t16d_development_20261120
```

The default output directory is next to this report and fails closed if it already exists. The script never copies the input NPZ into the repository.
"""
    (output_dir / "README.md").write_text(readme)

    manifest_files = [
        output_dir / "README.md", output_dir / "protocol.json", output_dir / "summary.json",
        output_dir / "intervals.csv", output_dir / "bootstrap_estimates.npz",
        output_dir / "c_lane_bootstrap_intervals.png",
        output_dir / "snapshot_c_lane_open_development_diagnostic.py",
        output_dir / "snapshot_core_npz_safety.py",
        output_dir / "snapshot_core_source_integrity.py",
    ]
    manifest = {
        "schema": "latticearena-diagnostic-manifest-v1",
        "status": summary["status"],
        "files": {str(path.relative_to(output_dir)): sha256_file(path)
                  for path in manifest_files},
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary,
                      "manifest": manifest}, indent=2, sort_keys=True, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="directory containing the pinned open D development transaction")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "c_lane_open_development_diagnostic_20261125_v1",
                        help="new output directory; existing directories are never overwritten")
    args = parser.parse_args()
    run(args.input_dir.expanduser().resolve(), args.output_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
