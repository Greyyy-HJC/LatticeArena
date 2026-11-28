"""Synthetic transfer-matrix trace diagnostic for Wilson F-lane q/h/x.

The finite-Nt trace is split by the state in the vacuum sector (m) and the
static-source sector (n).  This is an exact finite-state toy only: it does not
read Wilson measurements, fit a correlator matrix, or certify physical bounds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results/wilson_loop/f_lane_finite_nt_trace_diagnostic_20261126_v1"
R_VALUES = (2, 3)
OPERATOR_IDS = ("identity", "APE", "stout")
TEMPORAL_EXTENTS = (16, 32)
TIME_EXTENTS = tuple(range(3, 8))
EFFECTIVE_ENERGY_TIMES = tuple(range(3, 7))

# Energies are in a_t*E.  The vacuum ground energy is set to zero so that the
# excited-vacuum factors are directly exp[-epsilon_m (Nt-T)].
VACUUM_ENERGIES = np.asarray([0.0, 0.42], dtype=float)
STATIC_ENERGIES = {
    2: np.asarray([0.44, 0.94, 1.29], dtype=float),
    3: np.asarray([0.59, 1.07, 1.41], dtype=float),
}

# Effective source/sink transition overlaps A[m,n,k], with axes
# (R, vacuum state, static-source state, bare operator).  The overall
# normalization absorbs time-local endpoint conventions in this toy.
TRANSITION_OVERLAPS = np.asarray(
    [
        [
            [
                [1.00 + 0.00j, 0.72 + 0.05j, 0.48 - 0.04j],
                [0.33 + 0.02j, 0.60 - 0.08j, 0.27 + 0.03j],
                [0.20 - 0.04j, 0.25 + 0.04j, 0.62 + 0.10j],
            ],
            [
                [0.42 + 0.04j, 0.25 + 0.10j, 0.58 - 0.04j],
                [0.31 - 0.02j, 0.49 + 0.13j, 0.17 + 0.09j],
                [0.18 + 0.06j, 0.22 - 0.04j, 0.45 + 0.02j],
            ],
        ],
        [
            [
                [0.90 + 0.00j, 0.62 + 0.08j, 0.52 - 0.05j],
                [0.28 + 0.03j, 0.55 - 0.05j, 0.21 + 0.02j],
                [0.24 - 0.01j, 0.33 + 0.04j, 0.59 + 0.11j],
            ],
            [
                [0.40 + 0.05j, 0.22 + 0.08j, 0.52 - 0.03j],
                [0.27 - 0.02j, 0.45 + 0.10j, 0.19 + 0.08j],
                [0.15 + 0.05j, 0.24 - 0.04j, 0.41 + 0.05j],
            ],
        ],
    ],
    dtype=np.complex128,
)


def _validate_time(time_extent: int, temporal_extent: int) -> None:
    if type(time_extent) is not int or time_extent < 0:
        raise ValueError("time_extent must be a nonnegative integer")
    if type(temporal_extent) is not int or temporal_extent < 1:
        raise ValueError("temporal_extent must be a positive integer")
    if time_extent > temporal_extent:
        raise ValueError("time_extent cannot exceed temporal_extent")


def _radius_index(radius: int) -> int:
    if type(radius) is not int or radius not in R_VALUES:
        raise ValueError(f"radius must be one of {R_VALUES}")
    return R_VALUES.index(radius)


def _operator_index(operator_id: str) -> int:
    if operator_id not in OPERATOR_IDS:
        raise ValueError(f"operator_id must be one of {OPERATOR_IDS}")
    return OPERATOR_IDS.index(operator_id)


def partition_function(temporal_extent: int) -> float:
    _validate_time(0, temporal_extent)
    return float(np.sum(np.exp(-VACUUM_ENERGIES * temporal_extent)))


def finite_trace_matrix(radius: int, time_extent: int,
                        temporal_extent: int) -> np.ndarray:
    """Return C_ij(T;Nt) from a truncated two-sector transfer-matrix trace.

    C_ij = Z0(Nt)^-1 sum_{m,n} exp[-eps_m(Nt-T)] exp[-V_n(R)T]
           A_i,mn conjugate(A_j,mn).
    """
    _validate_time(time_extent, temporal_extent)
    ri = _radius_index(radius)
    energies = STATIC_ENERGIES[radius]
    amplitudes = TRANSITION_OVERLAPS[ri]
    weights = np.exp(
        -VACUUM_ENERGIES[:, None] * (temporal_extent - time_extent)
        - energies[None, :] * time_extent
    )
    matrix = np.einsum("mn,mni,mnj->ij", weights, amplitudes,
                       amplitudes.conj(), optimize=True)
    matrix /= partition_function(temporal_extent)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("finite-Nt correlator matrix is not finite 3x3")
    if not np.allclose(matrix, matrix.conj().T, rtol=1e-12, atol=1e-14):
        raise ValueError("finite-Nt correlator matrix must be Hermitian")
    if np.linalg.eigvalsh(matrix)[0] < -1e-12:
        raise ValueError("finite-Nt correlator matrix must be positive semidefinite")
    return matrix


def trace_components(radius: int, operator_id: str, time_extent: int,
                     temporal_extent: int) -> dict[str, float]:
    """Separate m=0,n>0 (q), m>0,n=0 (h), and mixed m,n>0 (x)."""
    _validate_time(time_extent, temporal_extent)
    ri = _radius_index(radius)
    oi = _operator_index(operator_id)
    energies = STATIC_ENERGIES[radius]
    amplitudes = TRANSITION_OVERLAPS[ri]
    ground_overlap = abs(amplitudes[0, 0, oi]) ** 2
    if not math.isfinite(float(ground_overlap)) or ground_overlap <= np.finfo(float).tiny:
        raise ValueError("bare operator must have a nonzero ground-sector overlap")

    e0 = energies[0]
    q = sum(
        abs(amplitudes[0, n, oi]) ** 2 / ground_overlap
        * math.exp(-(float(energies[n]) - float(e0)) * time_extent)
        for n in range(1, energies.size)
    )
    h = sum(
        abs(amplitudes[m, 0, oi]) ** 2 / ground_overlap
        * math.exp(-float(VACUUM_ENERGIES[m]) * (temporal_extent - time_extent))
        for m in range(1, VACUUM_ENERGIES.size)
    )
    x = sum(
        abs(amplitudes[m, n, oi]) ** 2 / ground_overlap
        * math.exp(
            -float(VACUUM_ENERGIES[m]) * (temporal_extent - time_extent)
            -(float(energies[n]) - float(e0)) * time_extent
        )
        for m in range(1, VACUUM_ENERGIES.size)
        for n in range(1, energies.size)
    )

    matrix = finite_trace_matrix(radius, time_extent, temporal_extent)
    z0 = partition_function(temporal_extent)
    ground_contribution = ground_overlap * math.exp(-float(e0) * time_extent) / z0
    full_ratio = float(matrix[oi, oi].real / ground_contribution)
    reconstructed_ratio = 1.0 + q + h + x
    return {
        "q": float(q),
        "h": float(h),
        "x": float(x),
        "delta_log": math.log1p(q + h + x),
        "full_to_ground_ratio": full_ratio,
        "reconstructed_to_ground_ratio": reconstructed_ratio,
        "decomposition_abs_error": abs(full_ratio - reconstructed_ratio),
    }


def effective_energy(radius: int, operator_id: str, time_extent: int,
                     temporal_extent: int) -> dict[str, float]:
    """Return -log[C(T+1)/C(T)] and its bias relative to V0(R)."""
    if time_extent >= temporal_extent:
        raise ValueError("effective energy requires T+1 <= N_t")
    oi = _operator_index(operator_id)
    current = finite_trace_matrix(radius, time_extent, temporal_extent)[oi, oi].real
    following = finite_trace_matrix(radius, time_extent + 1, temporal_extent)[oi, oi].real
    if current <= 0 or following <= 0:
        raise ValueError("effective-energy loop means must be positive")
    value = -math.log(float(following / current))
    ground = float(STATIC_ENERGIES[radius][0])
    return {"effective_energy": value, "ground_energy": ground,
            "bias": value - ground}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)
    colors = {"q": "#2878a5", "h": "#d17b0f", "x": "#6b4c9a"}
    linestyles = {16: "-", 32: "--"}
    for ax, radius in zip(axes, R_VALUES):
        subset = [row for row in rows
                  if row["R"] == radius and row["operator_id"] == "stout"]
        times = sorted({int(row["T"]) for row in subset})
        qvals = [next(row["q"] for row in subset
                      if row["N_t"] == 16 and row["T"] == time) for time in times]
        ax.plot(times, qvals, marker="o", color=colors["q"],
                label="q: vacuum ground → static excitations")
        for temporal_extent in TEMPORAL_EXTENTS:
            selected = [row for row in subset if row["N_t"] == temporal_extent]
            for key in ("h", "x"):
                ax.plot(times, [row[key] for row in selected],
                        marker="s" if key == "h" else "^",
                        linestyle=linestyles[temporal_extent], color=colors[key],
                        label=f"{key}, N_t={temporal_extent}")
        ax.set_yscale("log")
        ax.set_xlabel("Wilson-loop temporal extent T (lattice steps)")
        ax.set_ylabel("positive correction relative to m=0,n=0")
        ax.set_title(f"R={radius}; stout bare operator")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Finite-Nt trace separates q, thermal h, and mixed x (synthetic)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build(output_dir: Path = DEFAULT_OUT) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    matrix_min_eigenvalues: list[float] = []
    for radius in R_VALUES:
        for temporal_extent in TEMPORAL_EXTENTS:
            for time_extent in TIME_EXTENTS:
                matrix = finite_trace_matrix(radius, time_extent, temporal_extent)
                matrix_min_eigenvalues.append(float(np.linalg.eigvalsh(matrix)[0]))
                for operator_id in OPERATOR_IDS:
                    row = {
                        "R": radius,
                        "operator_id": operator_id,
                        "N_t": temporal_extent,
                        "T": time_extent,
                        **trace_components(radius, operator_id, time_extent,
                                           temporal_extent),
                    }
                    rows.append(row)

    effective_rows = []
    for radius in R_VALUES:
        for temporal_extent in TEMPORAL_EXTENTS:
            for time_extent in EFFECTIVE_ENERGY_TIMES:
                effective_rows.append({
                    "R": radius, "operator_id": "stout", "N_t": temporal_extent,
                    "T": time_extent,
                    **effective_energy(radius, "stout", time_extent, temporal_extent),
                })

    csv_path = output_dir / "trace_components.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    effective_path = output_dir / "effective_energy_bias.csv"
    with effective_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(effective_rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(effective_rows)
    figure_path = output_dir / "finite_nt_trace_qhx.png"
    _plot(figure_path, rows)

    source_paths = {
        "diagnostic": Path(__file__),
        "lane_contract": TASK_DIR / "lane_contract.py",
        "lane_contract_json": TASK_DIR / "lane_contract_v2.json",
        "finite_nt_toy": TASK_DIR / "finite_nt_toy.py",
    }
    protocol = {
        "schema": "wilson-f-finite-nt-transfer-trace-toy-v1",
        "classification": "EXACT_FINITE_STATE_SYNTHETIC_TRACE_NOT_MEASUREMENT_OR_FIT",
        "model": {
            "equation": "C_ij(R,T;Nt)=Z0(Nt)^-1 sum_mn exp[-eps_m(Nt-T)] exp[-V_n(R)T] A_i,mn(R) conjugate(A_j,mn(R))",
            "partition_function": "Z0(Nt)=sum_m exp[-eps_m Nt]",
            "vacuum_energies_at_a_t_units": VACUUM_ENERGIES.tolist(),
            "static_sector_energies_at_a_t_units": {
                str(radius): values.tolist() for radius, values in STATIC_ENERGIES.items()
            },
            "transition_overlaps_state_axes": ["R", "vacuum_state_m", "static_source_state_n", "bare_operator_k"],
            "transition_overlaps": [
                [[[[float(value.real), float(value.imag)] for value in operator_vector]
                   for operator_vector in state_row] for state_row in radius_row]
                for radius_row in TRANSITION_OVERLAPS
            ],
            "operator_basis_ids": list(OPERATOR_IDS),
            "r_values": list(R_VALUES),
            "N_t_values": list(TEMPORAL_EXTENTS),
            "T_values": list(TIME_EXTENTS),
            "endpoint_convention": "time-local source/sink; constant endpoint factors are absorbed into A; production time-origin must be checked against its exporter",
        },
        "decomposition": {
            "q": "m=0,n>0: sum_n |A_(0,n,k)/A_(0,0,k)|^2 exp[-(V_n-V_0)T]",
            "h": "m>0,n=0: sum_m |A_(m,0,k)/A_(0,0,k)|^2 exp[-eps_m(N_t-T)]",
            "x": "m>0,n>0: sum_mn |A_(m,n,k)/A_(0,0,k)|^2 exp[-eps_m(N_t-T)-(V_n-V_0)T]",
            "identity": "C_kk/C_kk^(m=0,n=0)=1+q+h+x; delta_log=log1p(q+h+x)",
            "q_object": "kernel-indexed q uses the corresponding bare operator; a GEVP composite is not substituted",
        },
        "source_sha256": {name: _sha(path) for name, path in source_paths.items()},
        "limitations": [
            "all energies and transition overlaps are deterministic hand-set toy values",
            "no raw gauge configurations, Wilson loops, sealed data, GPU, or QUDA were read",
            "no nonlinear fit, covariance, bootstrap, one-sided interval, or coverage calibration is performed",
            "the two-sector finite trace motivates but does not certify the production q/h/x definitions",
            "local-in-time endpoints are assumed; any production insertion offset must be specified and reviewed",
            "q/h/x nonnegative decomposition assumes a positive transfer-matrix spectral representation",
        ],
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": "wilson-f-finite-nt-transfer-trace-summary-v1",
        "status": "SYNTHETIC_TRANSFER_TRACE_DIAGNOSTIC_ONLY",
        "certified": False,
        "physical_score": None,
        "max_decomposition_abs_error": max(row["decomposition_abs_error"] for row in rows),
        "minimum_matrix_eigenvalue": min(matrix_min_eigenvalues),
        "q_nt_independent": all(
            trace_components(r, op, t, 16)["q"] == trace_components(r, op, t, 32)["q"]
            for r in R_VALUES for op in OPERATOR_IDS for t in TIME_EXTENTS
        ),
        "h_decreases_when_Nt_doubles": all(
            trace_components(r, op, t, 32)["h"] < trace_components(r, op, t, 16)["h"]
            for r in R_VALUES for op in OPERATOR_IDS for t in TIME_EXTENTS
        ),
        "denominators": {
            "R_operator_Nt_T_decomposition_cells": {
                "unit": "R x bare_operator x N_t x T",
                "planned": len(R_VALUES) * len(OPERATOR_IDS) * len(TEMPORAL_EXTENTS) * len(TIME_EXTENTS),
                "valid": len(rows), "failed": 0, "abstained": 0,
            },
            "finite_Nt_matrix_cells": {
                "unit": "R x N_t x T Hermitian PSD matrix",
                "planned": len(R_VALUES) * len(TEMPORAL_EXTENTS) * len(TIME_EXTENTS),
                "valid": len(R_VALUES) * len(TEMPORAL_EXTENTS) * len(TIME_EXTENTS),
                "failed": 0, "abstained": 0,
            },
            "physical_measurements": {"unit": "raw Wilson measurement", "planned": 0,
                                      "valid": 0, "failed": 0, "abstained": 0},
            "certification_gate": {"unit": "physical F-lane certificate", "planned": 1,
                                   "valid": 0, "failed": 0, "abstained": 1},
        },
    }
    summary["source_hashes"] = protocol["source_sha256"]
    summary["artifact_manifest"] = "artifact_hashes.json"
    readme = f"""# Finite-`N_t` transfer-trace split of Wilson F-lane q/h/x

**Synthetic diagnostic only:** `physical_score=null`, no Wilson measurements or sealed inputs were read, and no GPU/QUDA work was run. The script evaluates an exact, hand-set finite-state transfer-matrix trace; it does not fit raw correlators or estimate interval coverage.

## Why this diagnostic was added

The selected F-lane contract already separates a forward excited-state term `q`, a complementary-time thermal term `h`, and a mixed term `x`, then propagates them as `log1p(q+h+x)`. The contract did not yet state how those terms correspond to the finite temporal trace of a Wilson loop. The static-potential derivation in [Della Morte et al. (2010)](https://arxiv.org/abs/1012.3037) writes a finite-`N_t` trace with a vacuum-sector transfer matrix and a static-source-sector transfer matrix (their Eq. 2.5); in the `N_t→∞` limit it reduces to a positive sum over static-source states weighted by squared overlaps (their Eq. 2.6). The split below is my proposed interpretation of that trace, not a reviewer-approved production model.

## Model and mapping

For vacuum-sector state `m`, static-source state `n`, and bare smearing operator `k`, the toy uses

```text
C_ij(R,T;N_t) = Z0(N_t)^-1 * sum_(m,n) exp[-eps_m*(N_t-T)]
                * exp[-V_n(R)*T] * A_i,mn(R) * conj(A_j,mn(R))
Z0(N_t) = sum_m exp[-eps_m*N_t]
q_k = sum_(n>0) |A_(0,n,k)/A_(0,0,k)|^2 exp[-(V_n-V_0)*T]
h_k = sum_(m>0) |A_(m,0,k)/A_(0,0,k)|^2 exp[-eps_m*(N_t-T)]
x_k = sum_(m>0,n>0) |A_(m,n,k)/A_(0,0,k)|^2
      * exp[-eps_m*(N_t-T) - (V_n-V_0)*T]
C_kk / C_kk^(0,0) = 1 + q_k + h_k + x_k
```

The `q` row is the vacuum ground state `m=0` with excited static-source states `n>0`; it uses the same bare-operator identity as the selected kernel-indexed cells. `h` contains excited vacuum-sector states with the static-source ground state. `x` contains terms excited in both sectors. A real producer may have endpoint-time offsets; this toy absorbs constant endpoint factors into `A` and assumes time-local insertion operators. The production exporter must declare its exact time origin before the formula is used.

The plotted example holds the spectrum and overlaps fixed while changing `N_t=16` to `32`. `q` is unchanged; `h` and `x` decrease with the complementary extent. For `R=2`, stout at `T=5`, the toy gives `q={trace_components(2, 'stout', 5, 16)['q']:.6g}`, `h(N_t=16)={trace_components(2, 'stout', 5, 16)['h']:.6g}`, `h(N_t=32)={trace_components(2, 'stout', 5, 32)['h']:.6g}`, and `x(N_t=16)={trace_components(2, 'stout', 5, 16)['x']:.6g}`. These values illustrate the algebra only.

![Finite Nt trace decomposition](finite_nt_trace_qhx.png)

## Checks and limits

- Full `C_ij` matrices are Hermitian positive semidefinite; all 60 `R × operator × N_t × T` q/h/x cells reconstruct their diagonal trace exactly. Maximum absolute decomposition residual: `{summary['max_decomposition_abs_error']:.3g}`.
- `q` does not change when only `N_t` changes, while the chosen positive toy thermal contribution `h` decreases when `N_t` doubles. This is the mathematical separation being tested; it does not establish the size or monotonicity of measured thermal errors in this task.
- Denominators (planned/valid/failed/abstained): decomposition `60/60/0/0`; matrix checks `20/20/0/0`; physical measurements `0/0/0/0`; certification gate `1/0/0/1`.

The finite trace does **not** by itself identify `q`, `h`, or `x` from noisy production data. That requires a reviewed source/sink convention, raw cross-correlator matrix, appropriate state model, covariance-aware fit, and simultaneous one-sided coverage. It also does not replace the separate finite-`T` Creutz lane. Keep `NOT_READY/UNCERTIFIED`, `GROUND_STATE_FORCE_UNCERTIFIED`, and `physical_score=null`.
"""
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    (output_dir / "f_lane_finite_nt_trace_diagnostic.py").write_bytes(Path(__file__).read_bytes())
    artifact_hashes = {
        path.name: _sha(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps({"schema": "latticearena-artifact-hashes-v1", "files": artifact_hashes},
                   indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["artifact_hashes"] = artifact_hashes
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.out), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
