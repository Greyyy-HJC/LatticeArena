"""Transfer-matrix excited-state bound and parameter-space sensitivity map.

The functions cover only a positive, zero-temperature spectral sum. They do
not estimate overlap ratios or gaps from production data and do not include
finite-Nt winding/thermal contributions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_OUT = (Path(__file__).resolve().parents[2]
               / "results/wilson_loop/ground_state_bound_sensitivity_20261123_v1")
T_VALUES = (4, 5, 6, 7)  # Creutz uses W(T) and W(T+1), with measured T<=8.
Q_VALUES = np.geomspace(1.0e-3, 1.0e2, 61)  # illustrative total overlap ratio
GAP_VALUES = np.linspace(0.01, 1.0, 61)  # illustrative lattice-unit gap


def _positive_finite(value: float, label: str, *, allow_zero: bool = False) -> float:
    if (not math.isfinite(value)
            or value < 0
            or (value == 0 and not allow_zero)):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{label} must be finite and {qualifier}")
    return float(value)


def excited_fraction_bound(total_overlap_ratio: float, minimum_gap: float,
                           time_extent: int) -> float:
    """Bound rho_R(T)=sum_n q_n exp(-DeltaE_n*T) by Q exp(-Delta_min*T)."""
    q = _positive_finite(total_overlap_ratio, "total_overlap_ratio", allow_zero=True)
    gap = _positive_finite(minimum_gap, "minimum_gap", allow_zero=True)
    if (isinstance(time_extent, bool) or not isinstance(time_extent, int)
            or time_extent < 1):
        raise ValueError("time_extent must be an integer >= 1")
    return q * math.exp(-gap * time_extent)


def force_bias_bound(total_overlap_ratio: float, minimum_gap: float,
                     time_extent: int) -> float:
    """Dimensionless bound log(1+rho_max(T)) for finite-T force/Creutz bias."""
    rho = excited_fraction_bound(total_overlap_ratio, minimum_gap, time_extent)
    return math.log1p(rho)


def minimum_gap_for_tolerance(total_overlap_ratio: float, tolerance: float,
                              time_extent: int) -> float:
    """Minimum gap sufficient for the excited-state bound to be <= tolerance."""
    q = _positive_finite(total_overlap_ratio, "total_overlap_ratio", allow_zero=True)
    tau = _positive_finite(tolerance, "tolerance")
    if (isinstance(time_extent, bool) or not isinstance(time_extent, int)
            or time_extent < 1):
        raise ValueError("time_extent must be an integer >= 1")
    if q == 0 or q <= math.expm1(tau):
        return 0.0
    return math.log(q / math.expm1(tau)) / time_extent


def effective_energy_correction(rho_t: float, rho_t_plus_one: float) -> float:
    """Return log[(1+rho(T))/(1+rho(T+1))] for a decaying positive spectrum."""
    current = _positive_finite(rho_t, "rho_t", allow_zero=True)
    following = _positive_finite(rho_t_plus_one, "rho_t_plus_one", allow_zero=True)
    if following > current:
        raise ValueError("positive spectral contamination must not grow with T")
    return math.log1p(current) - math.log1p(following)


def creutz_excited_correction(rho_r_t: float, rho_r_t_plus_one: float,
                              rho_r1_t: float, rho_r1_t_plus_one: float) -> float:
    """Return chi(R,T)-f0(R+1/2) under the positive-spectrum model."""
    g_r = effective_energy_correction(rho_r_t, rho_r_t_plus_one)
    g_r1 = effective_energy_correction(rho_r1_t, rho_r1_t_plus_one)
    return g_r1 - g_r


def _plot(path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_grid, gap_grid = np.meshgrid(Q_VALUES, GAP_VALUES)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    image = None
    for ax, t in zip(axes.flat, T_VALUES):
        bias = np.log1p(q_grid * np.exp(-gap_grid * t))
        image = ax.pcolormesh(Q_VALUES, GAP_VALUES, bias, shading="auto", cmap="magma")
        ax.set_xscale("log")
        ax.set_title(f"available finite T={t} (uses loops through T={t+1})")
        ax.set_xlabel(r"$Q=\sum_{n>0} c_n/c_0$ (illustrative)")
        ax.set_ylabel(r"$\Delta E_{min}$ (lattice units; illustrative)")
        ax.grid(alpha=0.15)
    fig.colorbar(image, ax=axes, label=r"$\ln(1+Qe^{-\Delta E_{min}T})$ bound", shrink=0.92)
    fig.suptitle("Ground-state force / finite-T Creutz: excited-state sensitivity only\n"
                 "Q and gap are unknown inputs, not inferred from D development data")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(output: str | Path = DEFAULT_OUT,
        protocol_path: str | Path | None = None) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for t in T_VALUES:
        for q in Q_VALUES:
            for gap in GAP_VALUES:
                rows.append({
                    "T": int(t),
                    "Q_illustrative": float(q),
                    "gap_illustrative": float(gap),
                    "bias_bound_lattice_units": force_bias_bound(float(q), float(gap), int(t)),
                })
    csv_path = out / "sensitivity_grid.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    protocol_sha256 = None
    if protocol_path is not None:
        protocol_path = Path(protocol_path)
        # Bind metadata only when the caller explicitly supplies a public design
        # artifact. The analytic default does not inspect any task protocol/data.
        json.loads(protocol_path.read_text())
        protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    spec = {
        "classification": "analytic transfer-matrix sensitivity; no physical contamination bound",
        "positive_spectral_model": "W_k,R(T)=A0_k,R exp(-E0_R T) [1+rho_k,R(T)]",
        "rho_bound": "rho_k,R(T) <= Q_k,R exp(-DeltaE_min,R*T), Q=sum excited/ground overlap ratios",
        "creutz_identity": "chi_k(R,T)=f0(R+1/2)+g_k,R+1(T)-g_k,R(T); g=log[(1+rho(T))/(1+rho(T+1))]",
        "bias_bound": "|chi-f0| <= max_R log(1+rho_R(T)) <= log(1+Qmax exp(-DeltaE_min*T))",
        "difference_candidate_identity_bound": "sum of the two lane-specific bias bounds",
        "time_values": list(T_VALUES),
        "Q_axis_illustrative_only": [float(Q_VALUES[0]), float(Q_VALUES[-1])],
        "gap_axis_illustrative_only": [float(GAP_VALUES[0]), float(GAP_VALUES[-1])],
        "finite_extent_warning": "This derivation assumes a positive forward spectral sum. Finite-Nt winding/thermal terms must be separately bounded or modeled.",
        "data_use": "No Wilson loop values or restricted split were read by this analytic script.",
        "protocol_sha256": protocol_sha256,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "summary.json").write_text(json.dumps(spec, indent=2) + "\n")
    _plot(out / "contamination_sensitivity.png")
    report = [
        "# Ground-state static-force contamination bound",
        "",
        "**Analytic sensitivity report; no new fit or measurement.** The plotted `Q` and gap",
        "axes are illustrative parameter ranges, not estimates from the D ensemble.",
        "No Wilson-loop measurements or restricted splits were read.",
        "",
        "## Derivation",
        "",
        r"Assume a positive forward transfer-matrix expansion $W_{k,R}(T)=A_{0,k,R}e^{-E_0(R)T}[1+\rho_{k,R}(T)]$, with $\rho=\sum_{n>0}(A_n/A_0)e^{-\Delta E_nT}\ge0$.",
        r"The effective energy is $V_{eff,k}(R,T)=-\log[W(R,T+1)/W(R,T)]=E_0(R)+g_{k,R}(T)$, where $g=\log[(1+\rho(T))/(1+\rho(T+1))]$.",
        r"For decaying positive contamination, $0\le g_{k,R}(T)\le\log[1+\rho_{k,R}(T)]$. The finite-$T$ Creutz ratio is the difference of neighboring effective energies, so $\chi_k(R,T)=f_0(R+1/2)+g_{k,R+1}(T)-g_{k,R}(T)$.",
        r"Thus $|\chi_k-f_0|\le\max_{R,R+1}\log[1+\rho_{k,R}(T)]$. If a defensible total-overlap bound $Q=\sum_{n>0}A_n/A_0$ and gap lower bound $\Delta E_{min}$ are available, then $|\chi_k-f_0|\le\log[1+Qe^{-\Delta E_{min}T}]$.",
        "",
        "![Excited-state contamination sensitivity](contamination_sensitivity.png)",
        "",
        "A candidate-minus-identity comparison needs both kernels' contamination controls; a",
        "conservative absolute bias bound is their sum. Statistical paired block-bootstrap",
        "uncertainty must be reported separately and combined with this systematic bound",
        "under a rule frozen before confirmation.",
        "",
        "## Physical limits and next contract",
        "",
        "The formula assumes a positive zero-temperature forward spectrum. Thermal/winding",
        "contributions and finite-extent effects need a separate model or upper bound.",
        "APE/stout changes the",
        "overlap ratios `Q` but should not change `E0` when the operators share the same",
        "static-source action and have nonzero ground-state overlap. No such overlap bound",
        "or lower gap bound is measured here, so the plot is a design aid only.",
        "",
        "For the selected finite-T Creutz lane, report `chi_k(R,T)` as an observable with",
        "its paired interval; do not apply this correction to relabel it as ground state.",
        "For the selected ground-state-force lane, require a per-kernel/per-R `Q` and gap",
        "bound (or an identifiable multi-T spectrum with validated truncation), add a",
        "separate thermal bound, and budget `statistical + excited-state + thermal +",
        "finite-a/volume` errors against the researcher-set tolerance. If the bounds cannot",
        "be sourced, preserve `GROUND_STATE_FORCE_UNCERTIFIED` and do not open validation",
        "or shared holdout data.",
    ]
    (out / "README.md").write_text("\n".join(report) + "\n")
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (out / "README.md", out / "summary.json", out / "sensitivity_grid.csv",
                  out / "contamination_sensitivity.png")
    }
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return {"rows": len(rows), **spec}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=None,
                        help="optional public protocol JSON to hash; never reads data files")
    args = parser.parse_args()
    result = run(args.output, args.protocol)
    print(json.dumps({"output": str(args.output), "rows": result["rows"],
                      "classification": result["classification"]}))


if __name__ == "__main__":
    main()
