"""Synthetic diagnostic for the operator object used by Wilson F-lane q(T).

The report proposes a precise meaning for the currently underspecified
``td=2``: solve the principal GEVP vector at (td,t0)=(2,1), normalize it in
the C(t0) metric, and compute q(T) from the overlaps of that fixed projected
operator.  This is a method proposal backed only by an exact toy spectrum; it
does not alter a production contract or certify a physical contamination
bound.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigh

from .gevp_t0_truncation_diagnostic import (
    CONTRACT_PATH,
    OVERLAPS_4X3,
    TOY_ENERGIES_4,
    correlation_matrix,
    sha256,
)


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = TASK_DIR / "gevp_overlap_object_diagnostic_20261126_v1"
T0 = 1
TD_VALUES = (2, 3, 4, 5)
CONTAMINATION_TIMES = (5, 6)


def _positive_hermitian_matrix(value: object, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=complex)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if not np.allclose(matrix, matrix.conj().T, rtol=1e-12, atol=1e-13):
        raise ValueError(f"{name} must be Hermitian; no silent symmetrization")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] <= 0:
        raise ValueError(f"{name} must be positive definite")
    return matrix


def principal_gevp_vector(matrix_td: object, matrix_t0: object) -> tuple[np.ndarray, float]:
    """Return the largest-eigenvalue vector, C(t0)-normalized and phase-fixed."""
    c_td = _positive_hermitian_matrix(matrix_td, "C(td)")
    c_t0 = _positive_hermitian_matrix(matrix_t0, "C(t0)")
    values, vectors = eigh(c_td, c_t0, check_finite=True)
    if not np.all(np.isfinite(values)) or values[-1] <= 0:
        raise ValueError("principal generalized eigenvalue must be finite and positive")
    vector = np.asarray(vectors[:, -1], dtype=complex)
    metric_norm = float(np.real(vector.conj() @ c_t0 @ vector))
    if not math.isfinite(metric_norm) or metric_norm <= 0:
        raise ValueError("principal GEVP vector has invalid C(t0) norm")
    vector /= math.sqrt(metric_norm)
    pivot = int(np.argmax(np.abs(vector)))
    if abs(vector[pivot]) == 0:
        raise ValueError("principal GEVP vector is numerically zero")
    vector *= np.exp(-1j * np.angle(vector[pivot]))
    return vector, float(values[-1])


def projected_state_overlaps(overlaps: object, vector: object) -> np.ndarray:
    """Compute Z_n(v)=sum_i conj(v_i) Z_ni for C_ij=sum_n Z_ni Z*_nj e^-E_nt."""
    z = np.asarray(overlaps, dtype=complex)
    v = np.asarray(vector, dtype=complex)
    if z.ndim != 2 or z.shape[1] != 3 or v.shape != (3,):
        raise ValueError("overlaps and GEVP vector must have shapes (states,3) and (3,)")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(v)):
        raise ValueError("overlaps and GEVP vector must be finite")
    return z @ v.conj()


def spectral_q(energies: object, state_overlaps: object, time: int) -> float:
    """Return the positive spectral excited/ground ratio q(T), not an energy bias."""
    energy = np.asarray(energies, dtype=float)
    amplitude = np.asarray(state_overlaps, dtype=complex)
    if (energy.ndim != 1 or amplitude.shape != energy.shape or energy.size < 2
            or not np.all(np.isfinite(energy)) or not np.all(np.isfinite(amplitude))
            or np.any(energy <= 0) or np.any(np.diff(energy) <= 0)):
        raise ValueError("energies and overlaps must be finite, ordered, and state-matched")
    if isinstance(time, bool) or not isinstance(time, (int, np.integer)) or time < 0:
        raise ValueError("time must be a nonnegative integer")
    if abs(amplitude[0]) <= np.finfo(float).tiny:
        raise ValueError("ground-state overlap must be nonzero")
    terms = np.square(np.abs(amplitude[1:] / amplitude[0])) * np.exp(
        -(energy[1:] - energy[0]) * int(time)
    )
    result = float(np.sum(terms))
    if not math.isfinite(result) or result < 0:
        raise ValueError("q(T) must be finite and nonnegative")
    return result


def _calculate_rows(
    energies: object = TOY_ENERGIES_4,
    overlaps: object = OVERLAPS_4X3,
) -> list[dict[str, Any]]:
    energy = np.asarray(energies, dtype=float)
    z = np.asarray(overlaps, dtype=complex)
    if z.ndim != 2 or z.shape != (energy.size, 3) or np.linalg.matrix_rank(z) < 3:
        raise ValueError("the toy spectrum requires a full-rank three-operator basis")
    matrices = {
        time: correlation_matrix(energy, z, time)
        for time in range(T0, max(TD_VALUES) + 1)
    }
    rows: list[dict[str, Any]] = []
    for operator_index in range(3):
        bare = z[:, operator_index]
        for time in CONTAMINATION_TIMES:
            rows.append({
                "operator_object": f"bare_operator_{operator_index}",
                "kind": "original_basis_operator",
                "t0": T0,
                "td": None,
                "T": time,
                "q_T": spectral_q(energy, bare, time),
                "metric_condition_number": None,
                "diagonalization_matrix_condition_number": None,
                "largest_gevp_eigenvalue": None,
            })
    for td in TD_VALUES:
        vector, eigenvalue = principal_gevp_vector(matrices[td], matrices[T0])
        projected = projected_state_overlaps(z, vector)
        metric_condition = float(np.linalg.cond(matrices[T0]))
        diagonalization_condition = float(np.linalg.cond(matrices[td]))
        for time in CONTAMINATION_TIMES:
            rows.append({
                "operator_object": f"principal_gevp_projection_td_{td}",
                "kind": "fixed_gevp_projected_operator",
                "t0": T0,
                "td": td,
                "T": time,
                "q_T": spectral_q(energy, projected, time),
                "metric_condition_number": metric_condition,
                "diagonalization_matrix_condition_number": diagonalization_condition,
                "largest_gevp_eigenvalue": eigenvalue,
            })
    return rows


def _plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["bare O0", "bare O1", "bare O2", "GEVP td=2", "GEVP td=3", "GEVP td=4", "GEVP td=5"]
    objects = [
        "bare_operator_0", "bare_operator_1", "bare_operator_2",
        *(f"principal_gevp_projection_td_{td}" for td in TD_VALUES),
    ]
    values = {
        (row["operator_object"], int(row["T"])): float(row["q_T"])
        for row in rows
    }
    q5 = [values[(name, 5)] for name in objects]
    q6 = [values[(name, 6)] for name in objects]
    x = np.arange(len(objects))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), constrained_layout=True)
    axes[0].bar(x - width / 2, q5, width, label="T=5", color="#2878a5")
    axes[0].bar(x + width / 2, q6, width, label="T=6", color="#e08b31")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, labels, rotation=28, ha="right")
    axes[0].set_ylabel(r"$q(T)=\sum_{n>0}|Z_n/Z_0|^2 e^{-\Delta E_n T}$")
    axes[0].set_title("The operator object changes q(T)")
    axes[0].grid(True, axis="y", which="both", alpha=0.25)
    axes[0].legend()

    gevp_objects = objects[3:]
    td = np.asarray(TD_VALUES, dtype=float)
    axes[1].plot(td, [values[(name, 5)] for name in gevp_objects], "o-",
                 label="q(T=5)", color="#2878a5")
    axes[1].plot(td, [values[(name, 6)] for name in gevp_objects], "s--",
                 label="q(T=6)", color="#e08b31")
    conditions = [
        float(next(row["diagonalization_matrix_condition_number"] for row in rows
                   if row["operator_object"] == name))
        for name in gevp_objects
    ]
    axes[1].set_xlabel(r"fixed eigenvector time $t_d$ (with $t_0=1$)")
    axes[1].set_ylabel("projected q(T)", color="#245b76")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(loc="upper right")
    twin = axes[1].twinx()
    twin.plot(td, conditions, "D:", color="#6a4c93", label="cond C(td)")
    twin.set_ylabel(r"$\mathrm{cond}[C(t_d)]$", color="#6a4c93")
    twin.legend(loc="center right")
    axes[1].set_title("td sensitivity in the same truncated toy")
    fig.suptitle("Wilson F-lane overlap-object proposal (synthetic only)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build(output_dir: Path = DEFAULT_OUT) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    rows = _calculate_rows()
    with (output_dir / "overlap_object_q.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _plot(output_dir / "overlap_object_q.png", rows)

    value_map = {(row["operator_object"], int(row["T"])): row for row in rows}
    bare_max = {
        str(time): max(float(value_map[(f"bare_operator_{j}", time)]["q_T"])
                       for j in range(3))
        for time in CONTAMINATION_TIMES
    }
    selected = {
        str(time): float(value_map[("principal_gevp_projection_td_2", time)]["q_T"])
        for time in CONTAMINATION_TIMES
    }
    condition_rows = [row for row in rows if row["kind"] == "fixed_gevp_projected_operator"]
    protocol = {
        "schema": "wilson-gevp-overlap-object-protocol-v1",
        "classification": "SYNTHETIC_FORWARD_SPECTRUM_METHOD_PROPOSAL_NOT_CALIBRATION",
        "model": {
            "correlator": "C_ij(t)=sum_n Z_ni*conj(Z_nj)*exp(-E_n*t)",
            "energies_at_t_equals_a_t_units": np.asarray(TOY_ENERGIES_4, dtype=float).tolist(),
            "overlaps_state_by_operator": np.asarray(OVERLAPS_4X3, dtype=float).tolist(),
            "states": int(np.asarray(TOY_ENERGIES_4).size),
            "operators": 3,
            "time_extent": "infinite forward spectrum; no backward or thermal term",
            "noise": False,
            "bootstrap": False,
        },
        "method": {
            "t0": T0,
            "t0_role": "metric-reference time in C(t0)",
            "td_values": list(TD_VALUES),
            "td_role": "fixed time used once to solve the principal GEVP and define a time-independent projected operator",
            "gevp": "C(td) v0 = lambda0 C(t0) v0; select the largest positive generalized eigenvalue",
            "normalization": "v0^dagger C(t0) v0 = 1",
            "phase_convention": "largest-magnitude vector component is real and nonnegative; q is phase invariant",
            "projected_overlap": "Z_n(v0)=sum_i conjugate(v0_i) Z_ni",
            "q": "q_v(T)=sum_(n>0) |Z_n(v0)/Z_0(v0)|^2 exp[-(E_n-E_0) T]",
            "q_times": list(CONTAMINATION_TIMES),
        },
        "source_hashes": {
            "contract_v2": sha256(CONTRACT_PATH),
            "parent_gevp_diagnostic": sha256(Path(__file__).with_name("gevp_t0_truncation_diagnostic.py")),
            "diagnostic_source": sha256(Path(__file__)),
        },
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "schema": "wilson-gevp-overlap-object-diagnostic-v1",
        "classification": "SYNTHETIC_FORWARD_SPECTRUM_METHOD_PROPOSAL_NOT_CALIBRATION",
        "status": "DIAGNOSTIC_ONLY",
        "physical_score": None,
        "certified": False,
        "lane_status": "GROUND_STATE_FORCE_UNCERTIFIED",
        "proposed_contract_candidate": {
            "t0_role": "metric_reference_time_in_C(t0)",
            "t0": T0,
            "td_role": "fixed time used once to solve the principal GEVP and define a time-independent projected operator",
            "td_for_first_candidate": 2,
            "eigenproblem": "C(td) v0 = lambda0 C(t0) v0; choose largest lambda0",
            "normalization": "v0^dagger C(t0) v0 = 1; phase fixed only for deterministic serialization",
            "overlap_object": "Z_n(v0)=sum_i conjugate(v0_i) Z_ni in C_ij=sum_n Z_ni conjugate(Z_nj) exp(-E_n t)",
            "q_definition": "sum_{n>0}|Z_n(v0)/Z_0(v0)|^2 exp[-(E_n-E_0) T]",
            "q_times": list(CONTAMINATION_TIMES),
            "status": "review_candidate_only_not_written_into_production_contract",
        },
        "results": {
            "largest_bare_operator_q_by_T": bare_max,
            "selected_td2_projected_q_by_T": selected,
            "selected_to_largest_bare_ratio_by_T": {
                str(time): selected[str(time)] / bare_max[str(time)]
                for time in CONTAMINATION_TIMES
            },
            "gevp_projected_q_by_td_and_T": {
                str(td): {
                    str(time): float(value_map[(f"principal_gevp_projection_td_{td}", time)]["q_T"])
                    for time in CONTAMINATION_TIMES
                } for td in TD_VALUES
            },
            "C_t0_condition_number": float(condition_rows[0]["metric_condition_number"]),
            "C_td_condition_number_by_td": {
                str(td): float(next(
                    row["diagonalization_matrix_condition_number"]
                    for row in condition_rows if row["td"] == td
                )) for td in TD_VALUES
            },
        },
        "denominators": {
            "bare_operator_time_cells": {
                "unit": "operator,T pair", "planned": 6, "valid": 6,
                "failed": 0, "abstained": 0,
            },
            "gevp_projected_time_cells": {
                "unit": "td,T pair", "planned": len(condition_rows) * len(CONTAMINATION_TIMES),
                "valid": len(condition_rows) * len(CONTAMINATION_TIMES),
                "failed": 0, "abstained": 0,
            },
            "physical_inputs": {
                "unit": "raw Wilson correlator or independent spectrum receipt",
                "planned": 0, "valid": 0, "failed": 0, "abstained": 0,
            },
            "physical_certification_gate": {
                "unit": "F-lane certification", "planned": 1, "valid": 0,
                "failed": 0, "abstained": 1,
            },
        },
        "limitations": [
            "the four-state, three-operator spectrum is deterministic and has infinite temporal extent",
            "q(T) is a positive spectral contamination diagnostic, not the effective-energy bias",
            "there is no noise, bootstrap, fitted interval, finite-Nt backward term, thermal h, mixed x, or force propagation",
            "the proposed td and projected-overlap semantics require independent physics review before contract adoption",
        ],
        "source_hashes": {
            "contract": sha256(CONTRACT_PATH),
            "parent_gevp_diagnostic": sha256(Path(__file__).with_name("gevp_t0_truncation_diagnostic.py")),
            "diagnostic_source": sha256(Path(__file__)),
            "protocol": sha256(output_dir / "protocol.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copyfile(Path(__file__), output_dir / Path(__file__).name)
    shutil.copyfile(CONTRACT_PATH, output_dir / CONTRACT_PATH.name)
    parent = Path(__file__).with_name("gevp_t0_truncation_diagnostic.py")
    shutil.copyfile(parent, output_dir / parent.name)
    readme = [
        "# Wilson F-lane GEVP overlap-object diagnostic",
        "",
        "This is a deterministic synthetic forward-spectrum method proposal. It does not read Wilson measurements, open sealed data, or certify a physical force.",
        "",
        "## Candidate definition",
        "",
        "Interpret `t0=1` as the metric-reference matrix and `td=2` as the fixed time at which the principal GEVP eigenvector is solved once. Normalize `v0` by `v0† C(t0) v0 = 1`. For the matrix convention `C_ij(t)=Σ_n Z_ni Z*_nj exp(-E_n t)`, use the projected overlap `Z_n(v0)=Σ_i conj(v0_i) Z_ni`, and define `q_v(T)=Σ_{n>0}|Z_n(v0)/Z_0(v0)|² exp[-(E_n-E_0)T]`. The eigenvector is held fixed for T=5 and 6.",
        "",
        "This resolves the object ambiguity as a review candidate. It does not change `lane_contract_v2.json` or connect the candidate to the F gate.",
        "",
        "## What the toy shows",
        "",
        f"In the four-state/three-operator toy, selected `td=2` gives q(T=5)={selected['5']:.6g} and q(T=6)={selected['6']:.6g}; the largest of the three original basis operators gives {bare_max['5']:.6g} and {bare_max['6']:.6g}. The projected q is therefore {selected['5']/bare_max['5']:.3g}x and {selected['6']/bare_max['6']:.3g}x those maxima. This is evidence that the operator object cannot be left implicit in this toy, not a claim that GEVP worsens physical contamination.",
        "",
        "The right panel varies the fixed diagonalization time. q(T) changes strongly with td while the same exact four-state spectrum and three-operator basis are held fixed. It also shows `cond[C(td)]`; this is only the exact toy matrix condition number, not the condition of a noisy covariance-weighted fit.",
        "",
        "![q(T) for original and GEVP-projected operator objects](overlap_object_q.png)",
        "",
        "## Denominators and boundaries",
        "",
        f"Bare operator×T cells: {summary['denominators']['bare_operator_time_cells']['planned']}/{summary['denominators']['bare_operator_time_cells']['valid']}/0/0. GEVP td×T cells: {summary['denominators']['gevp_projected_time_cells']['planned']}/{summary['denominators']['gevp_projected_time_cells']['valid']}/0/0. Physical raw inputs: 0/0/0/0. F certification gate: 1/0/0/1. Cell counts are planned/valid/failed/abstained.",
        "",
        "The toy has no measurement noise, bootstrap interval, finite-Nt backward propagation, matched-Nt=32 thermal term, mixed-sector bound, reference-force interval, or model-truncation budget. `q(T)` is not itself the finite-T effective-energy bias. Keep `GROUND_STATE_FORCE_UNCERTIFIED`, `NOT_READY/UNCERTIFIED`, and `physical_score=null`.",
        "",
        "## Reproducibility",
        "",
        "`overlap_object_q.csv` contains every operator×T and td×T cell. `protocol.json` records the exact synthetic spectrum and the candidate method definitions. `summary.json` gives the results and denominators. `artifact_hashes.json` binds the report, CSV, image, protocol, summary, source snapshots and selected v2 contract.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    artifact_names = sorted(
        path.name for path in output_dir.iterdir()
        if path.is_file() and path.name != "artifact_hashes.json"
    )
    hashes = {name: sha256(output_dir / name) for name in artifact_names}
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        summary = build(args.out)
    except (FileExistsError, OSError, ValueError, ArithmeticError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps({
        "status": summary["status"],
        "output": str(args.out),
        "results": summary["results"],
        "denominators": summary["denominators"],
        "physical_score": None,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
