"""Diagnostic-only adapter for Optimize's selected Chebyshev candidate.

This registers an executable CPU candidate shape with LatticeArena's generic
candidate contract. Selected method defaults live in the research contract;
this adapter does not supply D04 data, read task measurements, or certify a
physical production method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from core.io import digest
from core.receipts import run_receipt, scenario_spec
from core.specs import candidate_spec, operator_contract

from .chebyshev_baseline import validate_control_points


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/optimize_gaussian/framework_candidate_contract_20261125_v4"
SCHEMA = "optimize-framework-candidate-diagnostic-v1"
BUILDER = "tasks.optimize_gaussian.chebyshev_baseline:apply_hopping_filter_cpu"
EXAMPLE_CONTROLS = [0.0, 0.25, 0.5, 0.75, 1.0]


def operator_contract_for() -> dict:
    """Return the versioned operator shape without claiming D11 approval."""
    baseline = Path(__file__).with_name("chebyshev_baseline.py")
    kernels = Path(__file__).with_name("kernels.py")
    core_specs = ROOT / "core/specs.py"
    core_receipts = ROOT / "core/receipts.py"
    research_contract = Path(__file__).with_name("research_contract_v1.json")
    return operator_contract(
        name="optimize-chebyshev-diagnostic-v1",
        axes="(z,y,x,3,nrhs)",
        dtype="complex128",
        transformation="gauge_covariant",
        support="time_slice",
        parameters={
            "role": "diagnostic candidate schema for selected D11 method defaults; production review remains open",
            "builder_parameters": {
                "control_points": "five finite real Bernstein controls, 0<=b0<=...<=b4=1",
                "links": "three periodic spatial directions of unitary 3x3 links",
                "field": "(z,y,x,3,nrhs)",
            },
            "spectral_map": "P4(I+H/12), proposed H spectrum [-12,0]",
        },
        dependencies={
            "candidate_contract.py": digest(Path(__file__)),
            "chebyshev_baseline.py": digest(baseline),
            "research_contract_v1.json": digest(research_contract),
            "core/receipts.py": digest(core_receipts),
            "core/specs.py": digest(core_specs),
            "kernels.py": digest(kernels),
        },
        constraints={
            "physical_contract_frozen": False,
            "selected_method_defaults_recorded": True,
            "production_evaluator_connected": False,
            "spatial_directions": 3,
            "boundary": "periodic spatial shifts",
            "link_condition": "unitary SU(3)-like 3x3 matrices checked by CPU helper",
            "control_constraints": "0<=b0<=b1<=b2<=b3<=b4=1",
            "normalization": "P4(1)=1 for the proposed low-mode endpoint",
            "unresolved": [
                "D04 independent E-chain reference and receipt",
                "D06 empirical paired-risk and full-cost receipts",
                "D11 independent production-link/operator review",
                "data-owner and GPU/QUDA authorization",
            ],
        },
    )


def candidate_spec_for(control_points=EXAMPLE_CONTROLS) -> dict:
    """Build one validated candidate descriptor; defaults are illustrative only."""
    controls = validate_control_points(control_points).tolist()
    contract = operator_contract_for()
    return candidate_spec(
        BUILDER,
        {"control_points": controls},
        contract,
        contract["dependencies"],
    )


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_report(output: str | Path = DEFAULT_OUT) -> dict:
    """Write a no-clobber registration fixture with text flow and file hashes."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    contract = operator_contract_for()
    candidate = candidate_spec_for()
    source_paths = {
        "snapshot_candidate_contract.py": Path(__file__),
        "snapshot_chebyshev_baseline.py": Path(__file__).with_name("chebyshev_baseline.py"),
        "snapshot_research_contract_v1.json": Path(__file__).with_name("research_contract_v1.json"),
        "snapshot_kernels.py": Path(__file__).with_name("kernels.py"),
        "snapshot_core_receipts.py": ROOT / "core/receipts.py",
        "snapshot_core_specs.py": ROOT / "core/specs.py",
    }
    for target, source in source_paths.items():
        (out / target).write_bytes(source.read_bytes())

    report = {
        "schema": SCHEMA,
        "status": "DIAGNOSTIC_SCHEMA_ONLY",
        "physical_score": None,
        "data_access": "none",
        "authorization": {"production_data": False, "gpu_quda": False},
        "operator_contract": contract,
        "example_candidate": candidate,
        "example_only": True,
        "denominators": {
            "schema_fixture": {
                "planned": 1, "valid": 1, "failed": 0, "abstained": 0,
                "unit": "one illustrative candidate descriptor; no fit or measurement",
            },
            "coefficient_fit": {
                "planned": 0, "valid": 0, "failed": 0, "abstained": 0,
                "unit": "no D04 objective supplied",
            },
        },
        "open_gates": [
            "D04 independent reference and data receipts",
            "D06 empirical paired-risk and full-cost receipts",
            "D11 independent production-link/operator review",
            "measurement evaluator and production authorization",
        ],
    }
    _json(out / "contract.json", report)
    readme = """# Optimize candidate contract: diagnostic registration

**Status: `DIAGNOSTIC_SCHEMA_ONLY`; selected D11 method defaults are recorded, while production review and evidence remain open; `physical_score=null`; no data or GPU access.**

This package registers one degree-4 Bernstein/Chebyshev CPU candidate with the generic LatticeArena operator/candidate schema. The example controls `(0, .25, .5, .75, 1)` are illustrative; they were not fitted to a reference or measurement. The descriptor requires a spatial field with layout `(z,y,x,3,nrhs)`, periodic shifts, and checked unitary 3x3 links. Its proposed response is `P4(I+H/12)` with `0<=b0<=...<=b4=1`.

The text flow is:

```text
selected research-contract defaults -> validate five Bernstein controls -> emit candidate descriptor -> CPU operator helper
                                      |                                      |
                                      +-- invalid controls reject            +-- D04/D06 receipts, D11 review and evaluator/auth gates stay open
```

The schema fixture denominator is `planned/valid/failed/abstained = 1/1/0/0`; it counts one illustrative descriptor only. Coefficient-fit denominator is `0/0/0/0`: no D04 callback was supplied. `run_receipt.json` records the scenario assumptions, source hashes (including the selected research contract), denominator units, text flow and payload hashes. Any failed, abstained or not-run planned units require reason rows whose counts match exactly. Receipt status `complete` means all planned units are accounted for, not that a method succeeded. This adapter does not create an Optimize `TaskSpec`, connect a measurement evaluator, authorize production runs, or establish that the proposed periodic three-dimensional operator matches production conventions. The receipt itself is covered by `artifact_hashes.json` to avoid a recursive self-hash.
"""
    (out / "README.md").write_text(readme)
    scenario = scenario_spec(
        "optimize-chebyshev-schema-example",
        "one illustrative candidate descriptor",
        "Schema validation only; no fitted coefficients or measurements.",
        assumptions=["D11 production review and receipts remain open", "the controls are illustrative"],
        source_hashes=contract["dependencies"],
    )
    text_flow = (
        "selected research-contract defaults -> validate five Bernstein controls -> "
        "emit candidate descriptor -> CPU operator helper; D04/D06 empirical receipts, "
        "D11 production review and measurement authorization remain open"
    )
    receipt_payloads = sorted(path for path in out.iterdir() if path.is_file())
    receipt_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in receipt_payloads
    }
    receipt = run_receipt(
        scenario, report["denominators"], artifact_hashes=receipt_hashes,
        text_flow=text_flow,
    )
    _json(out / "run_receipt.json", receipt)
    payloads = sorted(path for path in out.iterdir() if path.is_file())
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in payloads}
    _json(out / "artifact_hashes.json", hashes)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    result = build_report(args.output)
    print(json.dumps({
        "status": result["status"],
        "physical_score": result["physical_score"],
        "data_access": result["data_access"],
        "output": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
