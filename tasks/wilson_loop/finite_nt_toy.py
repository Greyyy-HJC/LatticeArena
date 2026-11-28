"""Synthetic finite-Nt Wilson-loop algebra check; never reads loop data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from core.numerics import as_finite_real


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/wilson_loop/finite_nt_toy_validation_20261123_v2"
TOY_R = {
    "A0": 1.2,
    "V0": 0.52,
    "Q": 0.9,
    "H": 0.3,
    "X": 0.2,
    "gap_q": 0.35,
    "gap_0": 0.25,
}
TOY_R_PLUS_1 = {
    "A0": 0.9,
    "V0": 0.65,
    "Q": 1.3,
    "H": 0.45,
    "X": 0.15,
    "gap_q": 0.3,
    "gap_0": 0.25,
}
TOY_NT = 16
TOY_TIMES = (4, 5, 6, 7, 8)


def _finite_nonnegative(value: object, name: str) -> float:
    result = as_finite_real(value, name=name)
    if result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _log1p_nonnegative_sum(values: tuple[float, ...]) -> float:
    """Evaluate log(1 + sum(values)) without overflowing the sum."""
    largest = max(values, default=0.0)
    if largest <= 1.0:
        return math.log1p(math.fsum(values))
    log_sum = math.log(largest) + math.log(math.fsum(value / largest for value in values))
    return log_sum + math.log1p(math.exp(-log_sum))


def _time_inputs(time_extent: int, temporal_extent: int) -> None:
    if (isinstance(time_extent, bool) or not isinstance(time_extent, int)
            or time_extent < 0):
        raise ValueError("time_extent must be a nonnegative integer")
    if (isinstance(temporal_extent, bool)
            or not isinstance(temporal_extent, int)
            or temporal_extent < 1):
        raise ValueError("temporal_extent must be a positive integer")
    if time_extent > temporal_extent:
        raise ValueError("time_extent cannot exceed temporal_extent")


def contamination_terms(parameters: dict, time_extent: int,
                        temporal_extent: int) -> dict:
    """Evaluate positive toy q/h/x terms at one loop extent.

    Energies and gaps are dimensionless lattice quantities (``a_t E``).
    ``H`` propagates over the complementary temporal extent ``N_t-t``.
    """
    _time_inputs(time_extent, temporal_extent)
    required = ("Q", "H", "X", "gap_q", "gap_0")
    missing = [name for name in required if name not in parameters]
    if missing:
        raise ValueError(f"missing contamination parameters: {missing}")
    q0, h0, x0, gap_q, gap_0 = (
        _finite_nonnegative(parameters[name], name) for name in required
    )
    complement = temporal_extent - time_extent
    q = q0 * math.exp(-gap_q * time_extent)
    h = h0 * math.exp(-gap_0 * complement)
    x = x0 * math.exp(-gap_q * time_extent - gap_0 * complement)
    delta = _log1p_nonnegative_sum((q, h, x))
    return {"q": q, "h": h, "x": x,
            "complement_extent": complement,
            "delta": delta,
            "loop_envelope": delta}


def log_loop(parameters: dict, time_extent: int,
             temporal_extent: int) -> tuple[float, dict]:
    """Return toy ``log W`` and its exact positive contamination terms."""
    _time_inputs(time_extent, temporal_extent)
    if "A0" not in parameters or "V0" not in parameters:
        raise ValueError("parameters must contain A0 and V0")
    a0 = as_finite_real(parameters["A0"], name="A0")
    v0 = as_finite_real(parameters["V0"], name="V0")
    if a0 <= 0:
        raise ValueError("A0 must be finite and positive")
    terms = contamination_terms(parameters, time_extent, temporal_extent)
    value = math.log(a0) - v0 * time_extent + terms["delta"]
    if not math.isfinite(value):
        raise ValueError("log Wilson loop is outside finite floating-point range")
    return value, terms


def finite_nt_creutz(parameters_r: dict, parameters_r_plus_1: dict,
                     time_extent: int, temporal_extent: int) -> dict:
    """Evaluate one finite-Nt Creutz four-point stencil and its force bias."""
    _time_inputs(time_extent, temporal_extent)
    if time_extent < 1:
        raise ValueError("Creutz time_extent must be at least one")
    _time_inputs(time_extent + 1, temporal_extent)
    log_r_t, terms_r_t = log_loop(parameters_r, time_extent, temporal_extent)
    log_r_tp1, terms_r_tp1 = log_loop(
        parameters_r, time_extent + 1, temporal_extent)
    log_r1_t, terms_r1_t = log_loop(
        parameters_r_plus_1, time_extent, temporal_extent)
    log_r1_tp1, terms_r1_tp1 = log_loop(
        parameters_r_plus_1, time_extent + 1, temporal_extent)

    # Same sign convention as the Wilson task's finite-time Creutz definition.
    chi = -(log_r1_tp1 + log_r_t - log_r1_t - log_r_tp1)
    target_force = float(parameters_r_plus_1["V0"] - parameters_r["V0"])
    delta_cross = (terms_r1_tp1["delta"] + terms_r_t["delta"]
                   - terms_r1_t["delta"] - terms_r_tp1["delta"])
    envelope = sum(terms["loop_envelope"] for terms in
                   (terms_r_t, terms_r_tp1, terms_r1_t, terms_r1_tp1))
    return {
        "time_extent": time_extent,
        "temporal_extent": temporal_extent,
        "chi_lattice": chi,
        "target_force_lattice": target_force,
        "bias_lattice": chi - target_force,
        "delta_R_delta_t": delta_cross,
        "identity_bias_lattice": -delta_cross,
        "four_loop_envelope_lattice": envelope,
        "envelope_covers_bias": abs(chi - target_force) <= envelope + 1e-14,
        "logW_R_t": log_r_t,
        "logW_R_t_plus_1": log_r_tp1,
        "logW_R_plus_1_t": log_r1_t,
        "logW_R_plus_1_t_plus_1": log_r1_tp1,
        "terms": {
            "R_t": terms_r_t,
            "R_t_plus_1": terms_r_tp1,
            "R_plus_1_t": terms_r1_t,
            "R_plus_1_t_plus_1": terms_r1_tp1,
        },
    }


def _write_artifacts(out: Path, rows: list[dict]) -> dict:
    csv_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key != "terms"}
        for position, terms in row["terms"].items():
            for key, value in terms.items():
                flat[f"{position}_{key}"] = value
        csv_rows.append(flat)
    with (out / "toy_validation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)

    denominators = [
        {
            "unit": "hypothetical toy Creutz time point",
            "planned": len(rows),
            "valid": sum(row["envelope_covers_bias"] for row in rows),
            "failed": sum(not row["envelope_covers_bias"] for row in rows),
            "abstained": 0,
            "interpretation": "Deterministic synthetic identities, not measured samples.",
        },
        {
            "unit": "four-loop evaluations in toy stencils",
            "planned": 4 * len(rows),
            "valid": 4 * len(rows),
            "failed": 0,
            "abstained": 0,
            "interpretation": "Four symbolic loop points per planned toy time point.",
        },
    ]
    with (out / "denominator_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(denominators[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(denominators)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = [row["time_extent"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), layout="constrained")
    for label, parameters, color in (
        ("R", TOY_R, "#0072B2"), ("R+1", TOY_R_PLUS_1, "#D55E00"),
    ):
        terms = [contamination_terms(parameters, t, TOY_NT) for t in times]
        for key, style in (("q", "-"), ("h", "--"), ("x", ":")):
            axes[0].plot(times, [term[key] for term in terms],
                         linestyle=style, marker="o", color=color,
                         label=f"{label}: {key}")
    axes[0].set(xlabel="loop extent t", ylabel="relative contamination term",
                title="Illustrative q/h/x terms at N_t=16")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)

    biases = [row["bias_lattice"] for row in rows]
    envelopes = [row["four_loop_envelope_lattice"] for row in rows]
    axes[1].plot(times, biases, marker="o", color="#0072B2",
                 label="signed Creutz minus target force")
    axes[1].plot(times, envelopes, marker="s", color="#D55E00",
                 label="conservative sum of four log envelopes")
    axes[1].plot(times, [-value for value in envelopes], marker="s",
                 color="#D55E00", alpha=0.45, label="negative envelope")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(xlabel="Creutz time extent t", ylabel="bias / envelope (lattice units)",
                title="Exact four-point correction stays inside envelope")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Synthetic finite-N_t transfer-matrix algebra check; not measured data")
    fig.savefig(out / "finite_nt_toy_validation.png", dpi=180)
    plt.close(fig)

    return denominators


def build(output: str | Path = DEFAULT_OUT) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows = [finite_nt_creutz(TOY_R, TOY_R_PLUS_1, t, TOY_NT)
            for t in TOY_TIMES]
    denominators = _write_artifacts(out, rows)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / "snapshot_finite_nt_toy.py").write_bytes(Path(__file__).read_bytes())
    numerics_path = ROOT / "core/numerics.py"
    numerics_hash = hashlib.sha256(numerics_path.read_bytes()).hexdigest()
    (out / "snapshot_core_numerics.py").write_bytes(numerics_path.read_bytes())
    summary = {
        "schema": "wilson-finite-nt-toy-validation-v1",
        "status": "SYNTHETIC_ALGEBRA_VALIDATION_ONLY",
        "physical_score": None,
        "data_access": "No Wilson-loop measurements, sealed split, GPU, or QUDA data were read.",
        "unit": "dimensionless lattice units; V0 is a_t*V and gaps are a_t*Delta",
        "temporal_extent": TOY_NT,
        "time_extents": list(TOY_TIMES),
        "parameters": {"R": TOY_R, "R_plus_1": TOY_R_PLUS_1},
        "formulae": {
            "q": "Q*exp(-gap_q*t)",
            "h": "H*exp(-gap_0*(N_t-t))",
            "x": "X*exp(-gap_q*t-gap_0*(N_t-t))",
            "logW": "log(A0)-V0*t+log1p(q+h+x)",
            "chi": "-[logW(R+1,t+1)+logW(R,t)-logW(R+1,t)-logW(R,t+1)]",
            "force_bias": "chi-(V0(R+1)-V0(R))=-Delta_R_Delta_t_delta",
            "envelope": "sum of log1p(q+h+x) over the four loop points",
        },
        "identity_max_abs_error": max(
            abs(row["bias_lattice"] - row["identity_bias_lattice"])
            for row in rows),
        "all_envelopes_cover": all(row["envelope_covers_bias"] for row in rows),
        "denominator_summary": denominators,
        "planned_time_points": len(rows),
        "source_sha256": source_hash,
        "analysis_source_sha256": {
            "finite_nt_toy.py": source_hash,
            "core/numerics.py": numerics_hash,
        },
        "files": ["README.md", "toy_validation.csv", "denominator_summary.csv",
                  "finite_nt_toy_validation.png", "snapshot_finite_nt_toy.py",
                  "snapshot_core_numerics.py",
                  "summary.json", "artifact_hashes.json"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Finite-N_t Creutz toy validation",
        "",
        "**SYNTHETIC ALGEBRA VALIDATION ONLY. `physical_score=null`.** The inputs below are hand-set dimensionless toy values. No Wilson-loop measurement, sealed split, GPU, or QUDA input was read.",
        "",
        "## Explicit toy model",
        "",
        "```text",
        "q(R,t) = Q(R) exp[-gap_q(R) t]",
        "h(R,N_t-t) = H(R) exp[-gap_0(R) (N_t-t)]",
        "x(R,t,N_t-t) = X(R) exp[-gap_q(R) t-gap_0(R) (N_t-t)]",
        "log W(R,t;N_t) = log A0(R) - V0(R) t + log1p(q+h+x)",
        "```",
        "",
        f"The toy uses `N_t={TOY_NT}` and loop times `{list(TOY_TIMES)}`. Parameters for `R` are `{TOY_R}`; parameters for `R+1` are `{TOY_R_PLUS_1}`. `V0` and gaps are in dimensionless lattice units (`a_t V0`, `a_t Delta`). The toy values are not fitted, measured, or proposed physical bounds.",
        "",
        "## Four-point identity and bound",
        "",
        "The code computes all four `log W` values and then applies the task sign convention:",
        "",
        "```text",
        "chi = -[logW(R+1,t+1)+logW(R,t)-logW(R+1,t)-logW(R,t+1)]",
        "chi - [V0(R+1)-V0(R)] = -Delta_R Delta_t delta",
        "|chi - Delta_R V0| <= sum_four_points log1p(q+h+x)",
        "```",
        "",
        f"Maximum direct-vs-identity bias discrepancy: `{summary['identity_max_abs_error']:.3g}`. Four-point envelope covers every one of {len(rows)} toy time stencils: `{summary['all_envelopes_cover']}`. The bound is deliberately loose; this check validates indexing/sign and error propagation, not usefulness of a physical envelope.",
        "",
        "![Finite-N_t toy contamination and four-point bound](finite_nt_toy_validation.png)",
        "",
        "The left panel shows the finite-extent tradeoff: `q` decreases with `t`, while the complementary-time `h` term can increase. The right panel overlays the signed exact bias and the conservative four-loop envelope.",
        "",
        "## Denominators and scope",
        "",
        "`denominator_summary.csv` reports planned/valid/failed/abstained separately for toy time points and for their four-loop evaluations. These are deterministic algebra checks, not statistical samples, acceptance tests, or physical certification. The code rejects negative contamination amplitudes/gaps, nonpositive `A0`, and a stencil that would require a loop beyond `N_t`.",
        "The source map hashes both this toy implementation and the shared strict numeric-conversion helper; their bytes are retained as `snapshot_finite_nt_toy.py` and `snapshot_core_numerics.py`.",
        "Finite-`N_t` `q/h/x` terms can only be promoted to a ground-state force bound when independent physical envelopes and a prospective tolerance exist. Until then, retain `GROUND_STATE_FORCE_UNCERTIFIED`; finite-T Creutz remains its own observable lane.",
        "",
        "Rebuild into a new directory: `python -B -m tasks.wilson_loop.finite_nt_toy --output NEW_DIRECTORY`.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (out / "README.md", out / "toy_validation.csv",
                     out / "denominator_summary.csv",
                     out / "finite_nt_toy_validation.png",
                     out / "snapshot_finite_nt_toy.py",
                     out / "snapshot_core_numerics.py", out / "summary.json")
    }
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = build(args.output)
    print(json.dumps({
        "status": result["status"],
        "planned_time_points": result["planned_time_points"],
        "identity_max_abs_error": result["identity_max_abs_error"],
        "all_envelopes_cover": result["all_envelopes_cover"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
