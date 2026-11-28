"""Create a text-ledger diagnostic for the optional vector-anchor route.

Only a deterministic arithmetic fixture is generated. No anchor chain,
correlator, physical reference, validation, holdout, GPU, or QUDA data is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from core.receipts import run_receipt, scenario_spec
from .bounded_block_risk_anchor_vector import markov_anchor_mean_box


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/pion_projection/bounded_block_risk_anchor_vector_20261125_v1"


def _fixture_samples(count: int = 4000) -> np.ndarray:
    index = np.arange(count, dtype=float)
    return np.column_stack((
        0.5 + 0.15 * np.sin(2.0 * np.pi * index / 16.0),
        0.1 + 0.2 * np.cos(2.0 * np.pi * index / 13.0),
        0.7 + 0.1 * np.sin(2.0 * np.pi * index / 7.0),
    ))


def build_report(output: str | Path = DEFAULT_OUT) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    samples = _fixture_samples()
    support = [[0.0, 1.0], [-1.0, 1.0], [0.0, 1.0]]
    protocol = {
        "schema": "pion-vector-anchor-box-diagnostic-v1",
        "status": "FORMULA_FIXTURE_ONLY",
        "fixture_source": "deterministic trigonometric arithmetic fixture; not a Markov path or measurement",
        "sample_count": int(samples.shape[0]),
        "feature_count": int(samples.shape[1]),
        "support_bounds": support,
        "gamma_ps_lower_assumed": 0.20,
        "alpha_anchor": 0.01,
        "initial_law_tv_bound_assumed": 0.001,
        "burn_in_units_assumed": 8,
        "stationary_start": False,
        "physical_score": None,
        "data_access": "none",
    }
    result = markov_anchor_mean_box(
        samples,
        support_bounds=support,
        gamma_ps_lower=protocol["gamma_ps_lower_assumed"],
        alpha_anchor=protocol["alpha_anchor"],
        initial_law_tv_bound=protocol["initial_law_tv_bound_assumed"],
        burn_in_units=protocol["burn_in_units_assumed"],
        stationary_start=False,
    )
    report = {
        "schema": "pion-vector-anchor-box-report-v1",
        "status": "CONDITIONAL_VECTOR_MEAN_BOX_ONLY",
        "physical_score": None,
        "certified": False,
        "data_access": "none",
        "authorization": {"physical_data": False, "gpu_quda": False},
        "denominators": result["denominators"],
        "result": result,
        "target_fit_interval": None,
        "common_bias_bound_C": None,
        "not_established": [
            "the assumed anchor-chain pseudo-gap or TV bound",
            "the feature support bounds for actual correlator contributions",
            "a certified interval image of the fixed PP/A4P nonlinear fit over the mean box",
            "the correspondence of that fit image to the selected m_star",
            "the common-bias bridge C and independence from the score-chain outcomes",
        ],
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    module_path = Path(__file__).with_name("bounded_block_risk_anchor_vector.py")
    gap_path = Path(__file__).with_name("bounded_block_risk_markov_gap.py")
    scenario = scenario_spec(
        "pion-vector-anchor-box-math-fixture",
        "one deterministic vector-mean arithmetic fixture",
        "Conditional componentwise Markov concentration plus a union bound; no fit propagation.",
        assumptions=[
            "feature support, gamma_ps and TV values are illustrative assumptions",
            "the fixture is not a Markov chain or a physical measurement",
        ],
        source_hashes={
            "bounded_block_risk_anchor_vector_diagnostic.py": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
            "bounded_block_risk_anchor_vector.py": hashlib.sha256(
                module_path.read_bytes()).hexdigest(),
            "bounded_block_risk_markov_gap.py": hashlib.sha256(
                gap_path.read_bytes()).hexdigest(),
        },
    )
    rows = result["feature_intervals"]
    table = "\n".join(
        f"| {row['feature_index']} | {row['support'][0]:.3g}, {row['support'][1]:.3g} | "
        f"{row['sample_mean']:.6f} | {row['confidence_interval'][0]:.6f}, "
        f"{row['confidence_interval'][1]:.6f} | {row['capped_radius']:.6f} |"
        for row in rows
    )
    readme = f"""# Pion optional Markov vector-anchor box

**Status: conditional formula fixture only. `physical_score=null`; no physical/anchor data, sealed inputs, GPU or QUDA were accessed.**

This diagnostic applies the existing corrected Markov Bernstein radius to each fixed feature of a vector from the same hypothetical anchor chain. A union bound with family size `d=3` creates a simultaneous box. The single path-level initial-law TV budget `0.001` is subtracted once from `alpha_anchor=0.01`, leaving stationary alpha `0.009`; per-coordinate tails use family size three. The assumed pseudo-gap lower bound is `0.20`, and the assumed post-burn-in count is 8. These are illustrative inputs only.

The 4,000 rows are deterministic sine/cosine arithmetic, not configurations, a Markov path, or Monte Carlo. Their sample means only position the example intervals; they do not test coverage. The feature support ranges are also assumed.

| feature | assumed support | fixture mean | conditional interval | capped radius |
| ---: | ---: | ---: | ---: | ---: |
{table}

**Text flow:** `fixed bounded per-state feature vector -> Markov Bernstein interval for each coordinate -> Bonferroni joint mean box -> certified interval propagation through the frozen PP/A4P fit -> m_star interval and bias bridge C`. The implementation ends at the vector mean box. It deliberately returns `target_fit_interval=null` and `common_bias_bound_C=null` because no verified range solver is supplied.

The vector route avoids asserting that the nonlinear ensemble fit itself is a per-configuration scalar. It still needs proven feature supports, anchor-chain gap/TV receipts, an interval-certified image of the selected multichannel fit over the joint box, correspondence to the same `m_star`, and a common-bias/score-independence receipt. Any missing assumption leaves the Pion task `NOT_READY/UNCERTIFIED`.

Receipt denominators: vector box `1/1/0/0`; feature coordinates `3/3/0/0` (planned/valid/failed/abstained). `run_receipt.json` records source hashes, assumptions, denominators, artifact hashes and the text flow. Receipt `complete` accounts for this synthetic fixture only; it is not a measurement or coverage result.
"""
    (out / "README.md").write_text(readme)
    payload_paths = sorted(path for path in out.iterdir() if path.is_file())
    receipt = run_receipt(
        scenario,
        {
            "vector_box": {**result["denominators"]["vector_box"],
                           "unit": "one conditional arithmetic fixture"},
            "feature_coordinates": {**result["denominators"]["feature_coordinates"],
                                    "unit": "one coordinate in the synthetic feature vector"},
        },
        artifact_hashes={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in payload_paths},
        text_flow=(
            "fixed bounded feature vector on one anchor chain -> componentwise Markov "
            "Bernstein intervals with family=d -> joint mean box -> certified fit-range "
            "propagation still required -> otherwise ABSTAIN and keep physical_score=null"
        ),
    )
    (out / "run_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    tracked = sorted(path for path in out.iterdir() if path.name != "artifact_hashes.json")
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    report = build_report(args.output)
    print(json.dumps({
        "status": report["status"],
        "physical_score": report["physical_score"],
        "data_access": report["data_access"],
        "output": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
