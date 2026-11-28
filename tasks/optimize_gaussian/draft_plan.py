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
DEFAULT_OUT = ROOT / "results/optimize_gaussian/draft_plan_20261128_v9"
CONTRACT_PATH = TASK_DIR / "research_contract_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_paths() -> dict[str, Path]:
    """List each source that is both hashed and retained in the report."""
    return {
        "draft_plan.py": Path(__file__),
        "research_contract_v1.json": CONTRACT_PATH,
        "contract.py": TASK_DIR / "contract.py",
        "candidate_contract.py": TASK_DIR / "candidate_contract.py",
        "chebyshev_baseline.py": TASK_DIR / "chebyshev_baseline.py",
        "trainable_w.py": TASK_DIR / "trainable_w.py",
        "features.py": TASK_DIR / "features.py",
        "risk_statistics.py": TASK_DIR / "risk_statistics.py",
        "d04_raw_reference.py": TASK_DIR / "d04_raw_reference.py",
        "evidence_adapter.py": TASK_DIR / "evidence_adapter.py",
        "source_reader.py": TASK_DIR / "source_reader.py",
        "bounded_block_risk.py": TASK_DIR / "bounded_block_risk.py",
        "bounded_block_risk_markov.py": TASK_DIR / "bounded_block_risk_markov.py",
        "paired_cost_accounting.py": TASK_DIR / "paired_cost_accounting.py",
        "kernels.py": TASK_DIR / "kernels.py",
        "core/io.py": ROOT / "core/io.py",
        "core/evidence.py": ROOT / "core/evidence.py",
        "core/receipts.py": ROOT / "core/receipts.py",
        "core/specs.py": ROOT / "core/specs.py",
        "core/source_integrity.py": ROOT / "core/source_integrity.py",
        "core/numerics.py": ROOT / "core/numerics.py",
        "core/npz_safety.py": ROOT / "core/npz_safety.py",
        "core/state.py": ROOT / "core/state.py",
    }


def build_plan() -> dict[str, Any]:
    """Bind the selected method draft to the generic specs without launching."""
    source_paths = _source_paths()
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
            "core/numerics.py -> strict real/exact-integer conversion for the D06 iid/Markov readers",
            "selected D04/D06/D11 defaults -> generic TaskSpec + draft RunSpec",
            "D04 raw E-chain snapshot reader -> PP/A4P folding + six joint-fit variants -> fit diagnostic only",
            "D04/D11 summary JSON -> strict same-snapshot parser/content digest; no physical fit or production-operator evaluator",
            "source_reader.py -> immutable D06 q-matrix bytes -> paired block scores with assigned-failure penalty",
            "paired_cost_accounting.py -> same input/seed + environment checks -> shared setup/training and per-block cost totals; no measured ledger yet",
            "risk_statistics.py -> fail-closed D04/D06 array reducers; no draw generation or fitting",
            "no data hashes / no runner entrypoint / no authorization -> launch_allowed=false",
            "five physical certification stages remain abstained; physical_score=null",
        ],
        "not_established": [
            "production TaskSpec/RunSpec acceptance by a task runner",
            "D04 independent reference receipt or successful reference gate",
            "D11 independent production-operator review",
            "D06 paired outcomes, measured costs or complete failure denominator",
            "a complete comparable paired cost ledger with setup/training charged to the selected one-task batch",
            "a train.main/training entrypoint in this checkout",
            "an evaluator that generates paired outer block-bootstrap and inner D04 draws",
            "production D04/D11 summary generation from raw correlator/link inputs",
            "coverage calibration for the selected percentile-quantile convention",
            "GPU/QUDA permission or sealed-data permission",
        ],
    }


def build_report(output: str | Path = DEFAULT_OUT) -> dict:
    out = Path(output)
    plan = build_plan()
    source_bytes = {
        name: source_path.read_bytes()
        for name, source_path in _source_paths().items()
    }
    stale = [
        name for name, payload in source_bytes.items()
        if hashlib.sha256(payload).hexdigest() != plan["source_hashes"][name]
    ]
    if stale:
        raise ValueError(f"source changed while building draft report: {stale}")
    out.mkdir(parents=True, exist_ok=False)
    (out / "run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    for name, payload in source_bytes.items():
        snapshot_name = name.replace("/", "_")
        (out / f"snapshot_{snapshot_name}").write_bytes(payload)
    readme = """# Optimize generic TaskSpec/RunSpec draft

**Status: `DRAFT_NO_LAUNCH`.** This is a data-free schema integration of the selected D04/D06/D11 method defaults. It contains no measurement inputs and cannot launch training; `physical_score=null`.

## What this closes

The selected research contract is now represented in LatticeArena's generic `core.specs.task_spec` and `core.specs.run_spec` objects, with a SHA-256 map for the contract and source dependencies. The task spec keeps the two agreed comparisons: trainable-W against same-budget SLSQP, and trainable-W against the fixed degree-4 Chebyshev baseline. The example Chebyshev candidate is illustrative only.

The generic `validation` range is explicitly limited to the already-open D60-D99 **development-selection** role. It is not final confirmation. Sealed roles are not included in the plan. The ranges are metadata only; no data bytes, manifests, anchor chain, gap, validation or holdout files were opened.

Every dependency in the source hash map is copied into the report snapshot using the same source-path map, so the diagnostic has a one-to-one code/contract fingerprint and retained-source inventory. Snapshot bytes are checked against the plan hashes before the output directory is created. The map binds the selected trainable-W CPU parameterization and directional features as well as the D04 raw E-chain diagnostic reader, D04/D06 reducers, bounded-risk and Markov formulas, paired-cost ledger, evidence adapter, snapshot reader, shared evidence gate, receipt/state models, source-integrity code, and `core/npz_safety.py`. The trainable-W implementation is still an isolated coefficient/operator prototype; its coefficient-MSE target is not the selected physical risk and it is not wired as the production training arm. The NPZ safety helper checks ZIP/NPY structure and the decompressed-byte budget before NumPy allocates arrays. The reducers use fixed selected constants, paired draw-ID checks, full planned denominators, no epsilon repair, and abstention on incomplete or invalid draws. The D06 reader consumes one explicitly captured calibration NPZ, checks its schema, split, ordered arms/times, and macroblock IDs, then derives the three-arm bounded score matrix; a declared optimizer failure receives the preselected loss 1 while remaining in the assigned-arm denominator. The paired-cost reader separately retains shared setup/training and per-block solve/inference/I/O/failure-recovery time, checks matched device/precision/thread/software/clock metadata, and withholds cost ratios if rows are missing or environments differ; it is only a schema-checked reducer and receives no measurement records in this draft. This CPU reader has no path fallback and rejects closed split names. The D04 raw reader consumes captured manifest/protocol/correlator bytes, applies fixed PP/A4P folding and six joint-fit variants, and emits fit diagnostics only; it does not generate dependence-aware uncertainty or a gate-ready reference summary. D04/D11 metadata adapters can parse their summary JSON directly from the same captured source snapshot, rejecting duplicate keys, non-finite JSON constants, mismatched caller summaries and stale canonical content digests. This only binds a summary object to captured bytes; it does not calculate a D11 production operator summary. None of these modules launches solvers or establishes macroblock independence.

## Text flow

```text
selected contract + trainable-W prototype/features + reducer/reader/integrity/NPZ parser sources -> SHA-256 map and source snapshots
  -> generic TaskSpec (D0-D59 fit; D60-D99 development selection)
  -> D06 snapshot NPZ -> fixed arm/time/ID checks -> bounded paired block scores
  -> paired cost receipt -> shared setup/training + per-block solve/inference/I/O/recovery -> compare only with a complete same-environment ledger
  -> D04 raw correlator snapshot -> PP/A4P fold + six fit variants -> diagnostic only
  -> D04/D11 captured JSON summary -> strict decode/content digest -> metadata gate only
  -> D04/D06 array reducers remain source-bound; production draw generation is not wired
  -> generic RunSpec with empty data-hash map
  -> missing reference/input/runner receipts => launch_allowed=false
  -> five physical RunState stages abstained => physical_score=null
```

The schema-plan denominator is planned/valid/failed/abstained=`1/1/0/0`: one in-memory spec object. It is not a run or measurement. The physical gate denominator is `5/0/0/5`. Data-input denominator is `0/0/0/0` because this diagnostic reads no data.

## Remaining implementation and evidence

This checkout does not contain `tasks/optimize_gaussian/runspec_builder.py`, a task-specific worker, or `train.main`; the plan therefore does not point to a child command and cannot become a production RunSpec. The new D06 reader is a bounded-array reader, not a solver/evaluator and not the missing task runner. The D04 raw reader is now hash-bound in this draft and copied into its report, but remains a synthetic-capable diagnostic that lacks dependence-aware mass errors, paired shifts, and a gate-ready summary. The D04/D11 metadata parser does not substitute for production reference fitting or a production-link operator evaluator. The next software change requires integrating these arrays with a reviewed paired-results producer and RunSpec while preserving the closed-split preflight. The open physical gates separately require the E-chain reference, D11 production-link review, D06 paired solver/cost receipts, and explicit data/GPU authorization. No gate is inferred from schema validity.
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
