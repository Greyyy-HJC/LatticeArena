"""Data-free GEVP truncation diagnostic for the Wilson F-lane time choices.

The synthetic correlator uses an exact positive forward spectral sum with four
states and a three-operator basis.  It demonstrates finite-basis contamination
in the standard moving GEVP.  It is not a fit to Wilson-loop measurements and
does not create overlap/gap confidence intervals or a physical score.
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
from scipy.linalg import eigh


ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = TASK_DIR / "lane_contract_v2.json"
DEFAULT_OUT = TASK_DIR / "gevp_t0_truncation_diagnostic_20261126_v1"
FIT_TIMES = tuple(range(3, 8))
FIXED_T0_VALUES = (1, 2, 3)
TOY_ENERGIES_4 = np.array([0.18, 0.32, 0.49, 0.70], dtype=float)
OVERLAPS_4X3 = np.array([
    [1.00, 0.80, 0.55],
    [0.72, -0.35, 0.60],
    [0.45, 0.62, -0.50],
    [1.20, 0.75, 0.95],
], dtype=float)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def correlation_matrix(energies: object, overlaps: object, time: int) -> np.ndarray:
    """Return C_ij(t)=sum_n Z_ni Z*_nj exp(-E_n t) for a forward spectrum."""
    energy = np.asarray(energies, dtype=float)
    vectors = np.asarray(overlaps, dtype=complex)
    if energy.ndim != 1 or energy.size < 1 or not np.all(np.isfinite(energy)):
        raise ValueError("energies must be a nonempty finite vector")
    if np.any(energy <= 0) or np.any(np.diff(energy) <= 0):
        raise ValueError("energies must be strictly increasing and positive")
    if vectors.ndim != 2 or vectors.shape[0] != energy.size or vectors.shape[1] != 3:
        raise ValueError("overlaps must have shape (n_states, 3 operators)")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("overlaps must be finite")
    if isinstance(time, bool) or not isinstance(time, (int, np.integer)) or time < 0:
        raise ValueError("time must be a nonnegative integer")
    weights = np.exp(-energy * int(time))
    matrix = np.einsum("n,ni,nj->ij", weights, vectors, vectors.conj())
    return np.asarray(matrix, dtype=complex)


def _check_hermitian_positive(matrix: object, label: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=complex)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    if not np.allclose(value, value.conj().T, rtol=1e-12, atol=1e-13):
        raise ValueError(f"{label} must be Hermitian; no silent symmetrization")
    eigenvalues = np.linalg.eigvalsh(value)
    if eigenvalues[0] <= 0:
        raise ValueError(f"{label} must be positive definite for this GEVP")
    return value


def principal_eigenvalues(matrix_t: object, matrix_t0: object) -> np.ndarray:
    """Solve C(t)v=lambda C(t0)v; return positive lambda values, largest first."""
    c_t = _check_hermitian_positive(matrix_t, "C(t)")
    c_t0 = _check_hermitian_positive(matrix_t0, "C(t0)")
    values = eigh(c_t, c_t0, eigvals_only=True, check_finite=True)
    if not np.all(np.isfinite(values)) or values[0] <= 0:
        raise ValueError("GEVP principal eigenvalues must be finite and positive")
    return values[::-1].copy()


def effective_ground_energy(
    matrix_t: object,
    matrix_t_plus_1: object,
    matrix_t0: object,
) -> tuple[float, float, float]:
    """Compute log(lambda_0(t,t0)/lambda_0(t+1,t0)) and a conditioning receipt."""
    c_t0 = _check_hermitian_positive(matrix_t0, "C(t0)")
    lam_t = principal_eigenvalues(matrix_t, c_t0)[0]
    lam_next = principal_eigenvalues(matrix_t_plus_1, c_t0)[0]
    if lam_t <= 0 or lam_next <= 0:
        raise ValueError("ground principal eigenvalue must stay positive")
    return math.log(lam_t / lam_next), float(lam_t), float(lam_next)


def _calculate_rows(
    energies: np.ndarray = TOY_ENERGIES_4,
    overlaps: np.ndarray = OVERLAPS_4X3,
) -> list[dict[str, Any]]:
    if np.linalg.matrix_rank(np.asarray(overlaps, dtype=float)) < 3:
        raise ValueError("the three-operator overlap matrix must span rank 3")
    matrices = {t: correlation_matrix(energies, overlaps, t) for t in range(0, 9)}
    rows: list[dict[str, Any]] = []
    modes: list[tuple[str, int, int]] = []
    for t0 in FIXED_T0_VALUES:
        for time in FIT_TIMES:
            if t0 < time:
                modes.append((f"fixed_t0_{t0}", time, t0))
    for time in FIT_TIMES:
        # The stronger t0 >= (t+1)/2 condition is used so the paired
        # lambda(t), lambda(t+1) energy difference satisfies it at both times.
        t0 = (time + 2) // 2
        modes.append(("paired_time_condition_t0_ge_half", time, t0))

    for mode, time, t0 in modes:
        energy_eff, lam_t, lam_next = effective_ground_energy(
            matrices[time], matrices[time + 1], matrices[t0]
        )
        condition = float(np.linalg.cond(matrices[t0]))
        rows.append({
            "mode": mode,
            "t": time,
            "t0": t0,
            "effective_ground_energy": energy_eff,
            "exact_ground_energy": float(energies[0]),
            "bias": energy_eff - float(energies[0]),
            "abs_bias": abs(energy_eff - float(energies[0])),
            "lambda_ground_t": lam_t,
            "lambda_ground_t_plus_1": lam_next,
            "C_t0_condition_number": condition,
        })
    return rows


def _plot(path: Path, rows: list[dict[str, Any]], ground_energy: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "fixed_t0_1": ("#c62828", "o", r"fixed $t_0=1$ (selected)"),
        "fixed_t0_2": ("#1565c0", "s", r"fixed $t_0=2$"),
        "fixed_t0_3": ("#2e7d32", "^", r"fixed $t_0=3$"),
        "paired_time_condition_t0_ge_half": ("#6a1b9a", "D", r"$t_0\geq(t+1)/2$"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    for mode, (color, marker, label) in styles.items():
        subset = [row for row in rows if row["mode"] == mode]
        times = [row["t"] for row in subset]
        energies = [row["effective_ground_energy"] for row in subset]
        biases = [row["abs_bias"] for row in subset]
        axes[0].plot(times, energies, marker=marker, color=color,
                     linewidth=1.8, label=label)
        axes[1].plot(times, biases, marker=marker, color=color,
                     linewidth=1.8, label=label)

    axes[0].axhline(ground_energy, color="black", linestyle="--",
                    label=rf"exact $E_0={ground_energy:.2f}$")
    axes[0].axvspan(2.5, 7.5, color="#ffcc80", alpha=0.16,
                    label="selected fit window t=3..7")
    axes[0].set_xlabel("effective-energy time t")
    axes[0].set_ylabel(r"$\log[\lambda_0(t,t_0)/\lambda_0(t+1,t_0)]$")
    axes[0].set_title("Three-operator basis with a fourth omitted state")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].set_yscale("log")
    axes[1].set_xlabel("effective-energy time t")
    axes[1].set_ylabel(r"absolute bias $|E_0^{eff}-E_0|$")
    axes[1].set_title("Finite-basis contamination in the exact toy")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Wilson F-lane GEVP time-choice sensitivity (synthetic only)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build(output_dir: Path = DEFAULT_OUT) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    rows = _calculate_rows()
    with (output_dir / "effective_energies.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _plot(output_dir / "gevp_t0_sensitivity.png", rows, float(TOY_ENERGIES_4[0]))

    summaries = {}
    for mode in sorted({row["mode"] for row in rows}):
        subset = [row for row in rows if row["mode"] == mode]
        summaries[mode] = {
            "planned": len(subset), "valid": len(subset), "failed": 0, "abstained": 0,
            "max_abs_bias_in_fit_window": max(row["abs_bias"] for row in subset),
            "mean_abs_bias_in_fit_window": float(np.mean([row["abs_bias"] for row in subset])),
            "unit": "dimensionless_energy_at_a_t_units_per_t_point",
        }
    protocol = {
        "schema": "wilson-gevp-t0-truncation-diagnostic-v1",
        "classification": "SYNTHETIC_FORWARD_SPECTRUM_DEVELOPMENT_ONLY_NOT_CALIBRATION",
        "model": {
            "formula": "C_ij(t)=sum_n Z_ni*conj(Z_nj)*exp(-E_n*t)",
            "temporal_extent": "infinite_forward_model; no finite-Nt or thermal term",
            "states": int(TOY_ENERGIES_4.size),
            "operators": 3,
            "energies": TOY_ENERGIES_4.tolist(),
            "overlaps_state_by_operator": OVERLAPS_4X3.tolist(),
            "measurement_noise": False,
            "bootstrap": False,
        },
        "method": {
            "gevp": "C(t)v=lambda(t,t0)C(t0)v",
            "effective_energy": "log(lambda_0(t,t0)/lambda_0(t+1,t0))",
            "fit_times": list(FIT_TIMES),
            "fixed_t0_sensitivity_values": list(FIXED_T0_VALUES),
            "paired_time_condition": "t0=ceil((t+1)/2), so t0>=t/2 for both lambda times",
            "selected_contract_t0": 1,
            "contract_td": 2,
            "td_semantics": "not defined by this diagnostic; current contract must say whether td freezes eigenvectors or denotes another time",
        },
        "denominators": {
            "deterministic_gevp_points": {
                "planned": len(rows), "valid": len(rows), "failed": 0, "abstained": 0,
                "unit": "one fixed-time pair (mode,t,t0)",
            },
            "synthetic_model": {"planned": 1, "valid": 1, "failed": 0, "abstained": 0},
            "physical_inputs": {
                "planned": 0, "valid": 0, "failed": 0, "abstained": 0,
                "unit": "raw correlator matrix or physical fit receipt",
            },
            "physical_certification_gate": {
                "planned": 1, "valid": 0, "failed": 0, "abstained": 1,
            },
        },
        "mode_summaries": summaries,
        "status": "DIAGNOSTIC_ONLY",
        "physical_score": None,
        "certified": False,
        "lane_status": "GROUND_STATE_FORCE_UNCERTIFIED",
        "contract_sha256": sha256(CONTRACT_PATH),
        "source_sha256": sha256(Path(__file__)),
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "schema": protocol["schema"],
        "status": protocol["status"],
        "mode_summaries": summaries,
        "denominators": protocol["denominators"],
        "physical_score": None,
        "certified": False,
        "lane_status": protocol["lane_status"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    readme = f"""# Wilson F-lane GEVP truncation diagnostic

**Synthetic, data-free method diagnostic.** It uses a positive forward spectral sum with four known energy levels and only three operators. It does not read Wilson data, produce q/h/x receipts, calibrate intervals, or change the F-lane certification status. `physical_score=null`.

## Formula and result

The standard moving GEVP is

```text
C(t) v_n(t,t0) = lambda_n(t,t0) C(t0) v_n(t,t0),  t > t0
E_eff,n(t,t0) = log[lambda_n(t,t0) / lambda_n(t+1,t0)].
```

The toy matrix is exactly `C_ij(t)=sum_n Z_ni Z*_nj exp(-E_n t)`. A three-operator basis resolves only three of its four states. The omitted fourth state produces nonzero effective-energy bias even though every matrix and energy is known exactly. The selected contract uses `t0=1`; its largest absolute bias over `t=3..7` is **{summaries['fixed_t0_1']['max_abs_bias_in_fit_window']:.6f}** in this toy. The purple curve changes `t0` with `t` so the more restrictive condition holds for both time points used by the finite difference; its largest bias is **{summaries['paired_time_condition_t0_ge_half']['max_abs_bias_in_fit_window']:.6f}**. This is a sensitivity comparison, not a universal bound or a proposed threshold.

![Effective-energy and truncation-bias sensitivity to t0](gevp_t0_sensitivity.png)

The test with exactly three states checks the finite-basis algebra: for a full-rank three-operator overlap matrix, the principal correlator gives the exact ground energy up to floating-point error. Adding one omitted state supplies a controlled negative fixture. `effective_energies.csv` retains every planned `(mode,t,t0)` point and the `C(t0)` condition number.

## Applicability boundary and protocol issue

The primary GEVP analysis by Blossier et al. defines the moving problem and effective energy, then analyzes omitted-state corrections. It assumes finite-time contributions can be neglected and states that the temporal extent must be large; its improved `t0 >= t/2` asymptotic behavior is a sufficient regime for a faster correction, not a universal finite-time certification. The project has `N_t=16` and a separate matched-`N_t=32` thermal lane, so this forward-only toy cannot validate the physical model. Source: [Blossier et al., *On the generalized eigenvalue method for energies and matrix elements in lattice field theory*](https://arxiv.org/abs/0902.1265), especially §§2.1–2.2.

There is also a concrete metadata ambiguity: the selected contract records both `t0=1` and `td=2`, but neither the raw reader nor the q-support receipt defines whether `td` is a fixed eigenvector diagonalization time, a time in the moving GEVP, or another fit parameter. This diagnostic uses the standard moving GEVP and does not assign semantics to `td`. Before a raw-matrix producer is wired to the q reader, the contract should freeze one definition and state whether the overlap ratios are for original smearing operators or GEVP-projected operators. Those ratios are not interchangeable without a derivation.

Text flow: `exact 4-state forward spectrum -> 3x3 C(t) matrices -> moving GEVP at fixed t0 -> effective ground energy over t=3..7 -> compare t0=1,2,3 and t0>=ceil((t+1)/2) -> bias/conditioning plot`. The diagnostic denominator is **{len(rows)}/{len(rows)}/0/0** deterministic GEVP points; the physical-input denominator is `0/0/0/0`, and the physical certification gate is `1/0/0/1`.

## Next step

Freeze the meaning of `td` and the operator whose overlap ratio enters `q`. Then implement the raw-matrix same-byte reader and fit producer against that reviewed definition, including finite-`N_t` treatment and paired resampling. Coverage, positive-spectrum assumptions, matched thermal `h`, mixed `x`, raw force/reference bootstrap, independent review, and owner authorization remain open. Keep the lane `GROUND_STATE_FORCE_UNCERTIFIED` until those receipts exist.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    files = sorted(path for path in output_dir.iterdir() if path.is_file())
    hashes = {path.name: sha256(path) for path in files}
    hashes["gevp_t0_truncation_diagnostic.py"] = sha256(Path(__file__))
    hashes["lane_contract_v2.json"] = sha256(CONTRACT_PATH)
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
