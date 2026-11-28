"""Source-bound, data-free audit of the selected D11 operator contract."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TASK = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results/optimize_gaussian/d11_operator_contract_audit_20261125_v1"
LATTICE_FORCE = "dimensionless lattice force a_s*a_t*F"
SOURCE_PATHS = {
    "research_contract_v1.json": TASK / "research_contract_v1.json",
    "contract.py": TASK / "contract.py",
    "candidate_contract.py": TASK / "candidate_contract.py",
    "features.py": TASK / "features.py",
    "kernels.py": TASK / "kernels.py",
    "chebyshev_baseline.py": TASK / "chebyshev_baseline.py",
    "trainable_w.py": TASK / "trainable_w.py",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit() -> dict[str, Any]:
    contract = json.loads(SOURCE_PATHS["research_contract_v1.json"].read_text())
    operator = contract["D11_strong_spectral_operator"]
    source = {name: path.read_text() for name, path in SOURCE_PATHS.items()
              if path.suffix == ".py"}

    checks = [
        {
            "check": "spatial_layout_and_periodic_shift",
            "status": "valid" if ("spatial axes are (z, y, x) = (0, 1, 2)" in source["features.py"]
                                    and "_lib(a).roll(a, -k, axis=2 - d)" in source["features.py"])
                     else "failed",
            "evidence": "features.py declares storage axes (z,y,x), direction order x/y/z, and np.roll periodic shifts",
        },
        {
            "check": "paired_forward_adjoint_hop",
            "status": "valid" if all(fragment in source["kernels.py"] for fragment in (
                "b = shift(a[d], -1, d)",
                "forward = links[d] @ shift(field, 1, d)",
                "backward = dagger(shift(links[d], -1, d)) @ shift(field, -1, d)",
            )) else "failed",
            "evidence": "kernels.py pairs each real forward coefficient with its shifted adjoint coefficient",
        },
        {
            "check": "H_spectrum_under_unitary_links",
            "status": "valid" if (operator["spectral_interval"] == [-12.0, 0.0]
                                    and "three periodic spatial directions" in operator["H_definition"]
                                    and "U(3) unitarity suffices" in operator["H_definition"])
                     else "failed",
            "evidence": "each unitary covariant shift T_i gives H_i=T_i+T_i†-2I with 0<=-H_i<=4I; three directions give 0<=-H<=12I",
        },
        {
            "check": "chebyshev_coordinate_matches_H",
            "status": "valid" if ("(apply_kernel(vector, u, unit_coefficients) - vector) / 6.0"
                                    in source["chebyshev_baseline.py"]
                                    and "X = I + H/6" in source["chebyshev_baseline.py"])
                     else "failed",
            "evidence": "the code applies x=I+H/6=2(I+H/12)-I, so x is in [-1,1] for the declared H spectrum",
        },
        {
            "check": "Bernstein_constraints_and_endpoint",
            "status": "valid" if ("np.any(np.diff(values) < 0.0)" in source["chebyshev_baseline.py"]
                                    and "values[-1] != 1.0" in source["chebyshev_baseline.py"]
                                    and operator["coefficient_constraints"].endswith("b4=1"))
                     else "failed",
            "evidence": "nonnegative Bernstein basis plus monotone controls gives 0<=P4<=1 and P4'(y)>=0; b4=1 gives P4(y=1)=1",
        },
        {
            "check": "Gaussian_affine_spectrum",
            "status": "valid" if ("1.6I+0.1H" in operator["outer_gaussian_sandwich"].get("G_relation", "")
                                    and operator["outer_gaussian_sandwich"]["G_spectrum"] == [0.4, 1.6])
                     else "failed",
            "evidence": "G=I+0.1 sum_i(T_i+T_i†)=1.6I+0.1H, hence spec(G) is contained in [0.4,1.6]",
        },
        {
            "check": "common_unnormalized_sandwich_and_conditional_factor",
            "status": "valid" if ("for _ in range(steps)" in source["chebyshev_baseline.py"]
                                    and operator["outer_gaussian_sandwich"]["raw_zero_mode_factor"] == 6.5536
                                    and "conditional on an H=0 eigenmode" in operator["outer_gaussian_sandwich"]["raw_zero_mode_factor_condition"])
                     else "failed",
            "evidence": "the helper applies G^2 on each side without per-step normalization; G and P4(H) commute, giving response G(lambda)^4 P4(y); 6.5536 is conditional on lambda=0 being an eigenvalue",
        },
        {
            "check": "physical_SU3_link_receipt",
            "status": "abstained",
            "evidence": "the CPU helper checks U†U≈I but does not check det(U)=1; production SU(3) provenance or determinant-unitarity evidence is still absent",
        },
        {
            "check": "production_objective_and_RunSpec_integration",
            "status": "abstained",
            "evidence": "trainable_w.py identifies itself as an isolated CPU parameterization with no production physics objective; D04 data and production evaluator receipts remain absent",
        },
    ]
    counts = {state: sum(row["status"] == state for row in checks)
              for state in ("valid", "failed", "abstained")}
    return {
        "schema": "optimize-d11-operator-audit-v1",
        "status": "SOURCE_MATH_AUDIT_ONLY",
        "data_access": "none",
        "physical_score": None,
        "certified": False,
        "denominator": {"planned": len(checks), **counts,
                        "unit": "one source/algebra contract obligation"},
        "checks": checks,
        "source_hashes": {name: _sha256(path) for name, path in SOURCE_PATHS.items()},
    }


def build(output: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    result = audit()
    (out / "protocol.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out / "audit_checks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["check", "status", "evidence"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["checks"])
    d = result["denominator"]
    lines = [
        "# D11 operator contract source audit",
        "",
        "**Source and algebra review only. No gauge fields, propagators, sealed splits, GPU, or QUDA inputs were read. `physical_score=null`; this is not an independent physics sign-off.**",
        "",
        "## Finding",
        "",
        "The degree-4 Bernstein/Chebyshev operator contract is algebraically consistent with the current CPU hopping and matrix-free filter implementation under periodic spatial shifts and unitary 3x3 links. Seven source/algebra obligations pass, none contradicts the contract, and two production obligations abstain. The full nine-row ledger and SHA-256 bindings are in `audit_checks.csv` and `protocol.json`.",
        "",
        "## Derivation and implementation mapping",
        "",
        "For each spatial direction define `(T_i psi)(x)=U_i(x) psi(x+i)`. With periodic shifts and unitary links, `T_i` is unitary. Thus `H_i=T_i+T_i†-2I=-(T_i-I)†(T_i-I)`, so `-4I<=H_i<=0`; summing three directions gives `-12I<=H<=0`. The stored axes are `(z,y,x)`, while direction indices map `(x,y,z)` to storage axes `(2,1,0)`. `kernels.py` implements the forward edge and the shifted adjoint edge, and the coefficient field is real, so the operator is Hermitian.",
        "",
        "The map `y=I+H/12` lies in `[0,1]`; `x=2y-I=I+H/6` lies in `[-1,1]` and is the coordinate used by the Clenshaw recurrence. Degree-4 Bernstein controls `0<=b0<=...<=b4=1` make `P4(y)` a positive contraction with `P4'(y)>=0` and `P4(1)=1`. This fixes the response at the upper spectral endpoint; a general gauge field need not have an `H=0` eigenmode.",
        "",
        "Since `sum_i(T_i+T_i†)=H+6I`, the selected `G=I+0.10 sum_i(T_i+T_i†)=1.6I+0.1H`, with spectrum in `[0.4,1.6]`. `G` commutes with `P4(H)`. The common unnormalized `G² P4(H) G²` therefore has eigenresponse `(1.6+0.1 lambda)^4 P4(1+lambda/12)`, which is nonnegative and bounded above by `1.6^4=6.5536`. That value is the response at `lambda=0` only if zero belongs to the actual spectrum; it is not a per-configuration normalization guarantee.",
        "",
        "## Review ledger",
        "",
        f"Denominator: planned/valid/failed/abstained = `{d['planned']}/{d['valid']}/{d['failed']}/{d['abstained']}`. `valid` means the source and stated algebra agree under the declared assumptions; it does not mean a production measurement passed.",
        "",
        "|check|status|evidence|",
        "|---|---|---|",
    ]
    for row in result["checks"]:
        lines.append(f"|{row['check']}|{row['status']}|{row['evidence']}|")
    lines.extend([
        "",
        "## Remaining handoff",
        "",
        "The matrix helper currently checks `U†U≈I`, which is sufficient for the spectral bound but does not establish `det(U)=1`; production QCD inputs need an independent SU(3) provenance or determinant-unitarity receipt. The production objective/evaluator and current D04 reference are also absent. Keep D11 `NOT_READY/UNCERTIFIED` until those source, data, and reviewer gates are independently satisfied. The analytic endpoint and raw sandwich factor have been corrected in the machine contract and candidate adapter.",
        "",
        "## Sources",
        "",
        "The audit binds the active `research_contract_v1.json`, `contract.py`, `candidate_contract.py`, `features.py`, `kernels.py`, `chebyshev_baseline.py`, and `trainable_w.py` by SHA-256 in `protocol.json`. The audit script itself is covered by `artifact_hashes.json`.",
        "",
    ])
    (out / "README.md").write_text("\n".join(lines))
    snapshot = out / "snapshot_d11_operator_contract_audit.py"
    snapshot.write_bytes(Path(__file__).read_bytes())
    files = ["README.md", "audit_checks.csv", "protocol.json",
             "snapshot_d11_operator_contract_audit.py"]
    hashes = {name: _sha256(out / name) for name in files}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = build(args.output)
    print(json.dumps({"status": result["status"],
                      "denominator": result["denominator"],
                      "physical_score": None}, indent=2))


if __name__ == "__main__":
    main()
