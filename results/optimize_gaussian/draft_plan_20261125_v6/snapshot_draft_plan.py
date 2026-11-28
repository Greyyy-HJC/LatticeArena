"""Build a no-launch TaskSpec/RunSpec draft for the selected Optimize contract.

This joins the generic LatticeArena schemas to the selected D04/D06/D11
defaults. It intentionally has no measurement inputs, training entrypoint, or
production authorization and must never be treated as a production RunSpec.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from core.specs import run_spec, task_spec

from .candidate_contract import (
    EXAMPLE_CONTROLS,
    candidate_spec_for,
    open_gate_run_state,
    operator_contract_for,
)


ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results/optimize_gaussian/draft_plan_20261125_v6"
CONTRACT_PATH = TASK_DIR / "research_contract_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_plan() -> dict[str, Any]:
    """Bind the selected method draft to the generic specs without launching."""
    source_paths = {
        "draft_plan.py": Path(__file__),
        "research_contract_v1.json": CONTRACT_PATH,
        "contract.py": TASK_DIR / "contract.py",
        "candidate_contract.py": TASK_DIR / "candidate_contract.py",
        "chebyshev_baseline.py": TASK_DIR / "chebyshev_baseline.py",
        "risk_statistics.py": TASK_DIR / "risk_statistics.py",
        "kernels.py": TASK_DIR / "kernels.py",
        "core/specs.py": ROOT / "core/specs.py",
        "core/state.py": ROOT / "core/state.py",
    }
    dependencies = {name: sha256(path) for name, path in source_paths.items()}
    contract_hash = dependencies["research_contract_v1.json"]
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    operator = operator_contract_for()
    candidate = candidate_spec_for(EXAMPLE_CONTROLS)

    # These are already-open development IDs from the project ledger. They are
    # represented only as ranges: this builder reads no measurements or files.
    # The generic schema's `validation` key is explicitly scoped to development
    # selection here and must not be mistaken for the sealed final confirmation.
    task = task_spec(
        name="optimize_gaussian",
        observable="fixed-protocol dimensionless Pion Gaussian-smearing risk",
        geometry=[16, 16, 16, 16],
        boundary={
            "gauge_links": {"spatial": "periodic", "temporal": "periodic"},
            "valence_quarks": {"spatial": "periodic", "temporal": "antiperiodic"},
        },
        source={"kind": "point", "position": [0, 0, 0, 0]},
        splits={"development": [0, 59], "validation": [60, 99]},
        reference={
            "observable": contract["D04_reference"]["observable"],
            "units": contract["D04_reference"]["units"],
            "method": contract["D04_reference"]["reference_method"],
            "source": "independent E-chain; receipt missing",
            "receipt_sha256": contract["D04_reference"]["data_receipt_sha256"],
        },
        metric={
            "objective_id": contract["fair_budget"]["objective_id"],
            "risk": contract["D04_reference"]["risk_objective"],
            "comparison_family": contract["comparison_family"],
            "D06": contract["D06_effect_threshold"],
        },
        budget={
            "objective_gradient_call_cap_per_arm": contract["fair_budget"]["objective_gradient_call_cap_per_arm"],
            "starts_per_arm": 1,
            "methods": ["trainable_W", "same_budget_SLSQP", "fixed_degree4_Chebyshev"],
            "walltime_rule": contract["fair_budget"]["walltime_rule"],
            "measured_costs": False,
        },
        dependencies=dependencies,
    )
    spec = run_spec(
        task,
        contract_hash,
        data={},
        status="DRAFT_NO_LAUNCH",
        data_access="none",
        authorization={"production_data": False, "gpu_quda": False},
        launch_allowed=False,
        physical_score=None,
        split_semantics={
            "development": "D0-D59 fit-only role; IDs represented as metadata, no bytes read",
            "validation": "D60-D99 development-selection role; not final confirmation",
            "sealed_roles_included": False,
        },
        missing_input_hashes=[
            "independent E-chain correlator/reference receipt",
            "authorized D-chain input manifest for this task",
            "fit/development input file hashes",
            "frozen seed/initialization plan receipt",
            "production evaluator and training entrypoint receipt",
        ],
        methods={
            "selected_comparisons": contract["comparison_family"],
            "D04_reference": contract["D04_reference"],
            "D06_effect_threshold": contract["D06_effect_threshold"],
            "D11_operator": contract["D11_strong_spectral_operator"],
        },
        operator_contract=operator,
        illustrative_candidate=candidate,
        run_state=open_gate_run_state(),
    )
    return {
        "schema": "optimize-gaussian-draft-run-plan-v1",
        "status": "DRAFT_NO_LAUNCH",
        "data_access": "none",
        "authorization": {"production_data": False, "gpu_quda": False},
        "launch_allowed": False,
        "physical_score": None,
        "contract_sha256": contract_hash,
        "source_hashes": dependencies,
        "task_spec": task,
        "run_spec": spec,
        "operator_contract": operator,
        "illustrative_candidate": candidate,
        "run_state": open_gate_run_state(),
        "denominators": {
            "schema_plan": {
                "planned": 1, "valid": 1, "failed": 0, "abstained": 0,
                "unit": "one generic TaskSpec/RunSpec draft validated in memory",
            },
            "physical_gates": {
                "planned": 5, "valid": 0, "failed": 0, "abstained": 5,
                "unit": "operator/reference/numerical/statistical/comparison certification stages",
            },
            "data_inputs": {
                "planned": 0, "valid": 0, "failed": 0, "abstained": 0,
                "unit": "no input manifest or measurement read",
            },
        },
        "text_flow": [
            "research_contract_v1.json + source files -> SHA-256 dependency map",
            "selected D04/D06/D11 defaults -> generic TaskSpec + draft RunSpec",
            "risk_statistics.py -> fail-closed D04/D06 array reducers; no draw generation or fitting",
            "no data hashes / no runner entrypoint / no authorization -> launch_allowed=false",
            "five physical certification stages remain abstained; physical_score=null",
        ],
        "not_established": [
            "production TaskSpec/RunSpec acceptance by a task runner",
            "D04 independent reference receipt or successful reference gate",
            "D11 independent production-operator review",
            "D06 paired outcomes, measured costs or complete failure denominator",
            "a train.main/training entrypoint in this checkout",
            "an evaluator that generates paired outer block-bootstrap and inner D04 draws",
            "coverage calibration for the selected percentile-quantile convention",
            "GPU/QUDA permission or sealed-data permission",
        ],
    }


def build_report(output: str | Path = DEFAULT_OUT) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    plan = build_plan()
    (out / "run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    shutil_snapshot = {
        "draft_plan.py": Path(__file__),
        "research_contract_v1.json": CONTRACT_PATH,
        "core_specs.py": ROOT / "core/specs.py",
        "core_state.py": ROOT / "core/state.py",
        "candidate_contract.py": TASK_DIR / "candidate_contract.py",
        "risk_statistics.py": TASK_DIR / "risk_statistics.py",
    }
    for name, source in shutil_snapshot.items():
        (out / f"snapshot_{name}").write_bytes(source.read_bytes())
    readme = """# Optimize generic TaskSpec/RunSpec draft

**Status: `DRAFT_NO_LAUNCH`.** This is a data-free schema integration of the selected D04/D06/D11 method defaults. It contains no measurement inputs and cannot launch training; `physical_score=null`.

## What this closes

The selected research contract is now represented in LatticeArena's generic `core.specs.task_spec` and `core.specs.run_spec` objects, with a SHA-256 map for the contract and source dependencies. The task spec keeps the two agreed comparisons: trainable-W against same-budget SLSQP, and trainable-W against the fixed degree-4 Chebyshev baseline. The example Chebyshev candidate is illustrative only.

The generic `validation` range is explicitly limited to the already-open D60-D99 **development-selection** role. It is not final confirmation. Sealed roles are not included in the plan. The ranges are metadata only; no data bytes, manifests, anchor chain, gap, validation or holdout files were opened.

The source hash map also binds `risk_statistics.py`, which implements pure-array D04/D06 reducers with fixed selected constants, paired draw-ID checks, full planned denominators, no epsilon repair, and abstention on incomplete or invalid draws. These helpers consume prepared arrays only; they do not generate the paired outer block-bootstrap or inner D04 bootstrap draws, fit a reference, or establish coverage for the percentile-quantile convention.

## Text flow

```text
selected research contract + source files -> SHA-256 dependency map
  -> generic TaskSpec (D0-D59 fit; D60-D99 development selection)
  -> D04/D06 array reducers are source-bound, but no production draw generator is wired
  -> generic RunSpec with empty data-hash map
  -> missing reference/input/runner receipts => launch_allowed=false
  -> five physical RunState stages abstained => physical_score=null
```

The schema-plan denominator is planned/valid/failed/abstained=`1/1/0/0`: one in-memory spec object. It is not a run or measurement. The physical gate denominator is `5/0/0/5`. Data-input denominator is `0/0/0/0` because this diagnostic reads no data.

## Remaining implementation and evidence

This checkout does not contain `tasks/optimize_gaussian/runspec_builder.py`, a task-specific worker, or `train.main`; the plan therefore does not point to a child command and cannot become a production RunSpec. This corrects broad historical TODO wording that described those components as present. The next software change requires restoring or implementing the actual Optimize entrypoint and binding an authorized, unsealed input manifest to its hash. The open physical gates separately require the E-chain reference, D11 production-link review, D06 paired outcomes/cost ledger, and explicit data/GPU authorization. No gate is inferred from schema validity.
"""
    (out / "README.md").write_text(readme)
    payloads = sorted(path for path in out.iterdir() if path.is_file())
    hashes = {path.name: sha256(path) for path in payloads}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    return {"status": plan["status"], "physical_score": None,
            "denominators": plan["denominators"], "output": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build_report(args.output), indent=2))


if __name__ == "__main__":
    main()
