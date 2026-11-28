"""Write a small, deterministic text diagnostic for the D04/D06 reducers.

Only constructed numeric fixtures are used. This is not a data runner,
bootstrap generator, coverage study, or physical comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .risk_statistics import (
    D06_BLOCK_LENGTHS,
    d04_bootstrap_risk,
    d06_paired_ratio_hull,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/optimize_gaussian/d04_d06_risk_statistics_cpu_20261125_v4"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ratio_fixture(values: dict[int, list[float]]) -> dict:
    risks = {length: np.asarray(values[length], dtype=float) for length in D06_BLOCK_LENGTHS}
    trainable = {length: np.ones(len(risks[length]), dtype=float) for length in D06_BLOCK_LENGTHS}
    ids = {length: np.arange(len(risks[length]), dtype=np.int64) for length in D06_BLOCK_LENGTHS}
    return d06_paired_ratio_hull(
        risks, trainable, ids, {length: ids[length].copy() for length in D06_BLOCK_LENGTHS},
        planned_draws_per_length=4,
    )


def build_report(output: str | Path = DEFAULT_OUT) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)

    d04 = d04_bootstrap_risk(np.asarray([
        [1.0, 0.5, 1.0, 1.0],
        [2.0, 0.5, 1.0, 0.0],
        [3.0, 0.5, 2.0, 1.0],
    ]), planned_draws=3)
    d06_cleared = _ratio_fixture({
        2: [1.40, 1.45, 1.50, 1.55],
        5: [1.35, 1.40, 1.45, 1.50],
        10: [1.30, 1.35, 1.40, 1.45],
    })
    d06_not_cleared = _ratio_fixture({
        2: [1.40, 1.45, 1.50, 1.55],
        5: [1.35, 1.40, 1.45, 1.50],
        10: [1.05, 1.10, 1.15, 1.20],
    })
    d06_invalid = _ratio_fixture({
        2: [1.40, 1.45, 1.50, 1.55],
        5: [1.35, 1.40, 1.45, 1.50],
        10: [1.05, 0.0, 1.15, 1.20],
    })

    scenarios = {
        "d04_hand_calculation": d04,
        "d06_synthetic_hull_clears": d06_cleared,
        "d06_synthetic_hull_does_not_clear": d06_not_cleared,
        "d06_synthetic_zero_risk_abstains": d06_invalid,
    }
    payload = {
        "schema": "optimize-d04-d06-risk-arithmetic-diagnostic-v1",
        "evidence_class": "synthetic_arithmetic_fixture_only",
        "source_hashes": {
            "risk_statistics.py": _sha256(Path(__file__).with_name("risk_statistics.py")),
            "research_contract_v1.json": _sha256(Path(__file__).with_name("research_contract_v1.json")),
        },
        "scenarios": scenarios,
    }
    (out / "diagnostic.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )

    readme = f"""# D04/D06 pure-array reducer diagnostic

**Evidence class:** deterministic CPU arithmetic fixture only. No gauge or correlator data, reference fit, bootstrap resampling, solver, GPU, sealed split, or physical score was used. The selected production contract remains `NOT_READY/UNCERTIFIED`.

## Text flow

```text
hand-entered q(theta,t) draws at t=(2,3,4,5)
  -> sample variance (ddof=1) + squared bias from one
  -> D04 scalar risk
hand-entered paired comparator/trainable-W risk arrays + matching draw IDs
  -> per-length comparator/candidate ratios
  -> linear 0.025 quantile for L=2,5,10
  -> minimum across lengths compared with r_star=10/9
invalid or incomplete inputs -> planned denominator retained -> ABSTAIN, no epsilon
```

The D04 arithmetic fixture used planned/valid/failed/abstained draws `3/3/0/0` and returned risk `{d04['risk']:.9f}`. The illustrative D06 fixtures used four paired draws per block length (`12` planned total, rather than the contract's `30,000`). The all-above-threshold fixture returned hull lower quantile `{d06_cleared['hull_lower']:.6f}` and array-only status `{d06_cleared['status']}`. Lowering the L=10 ratios returned `{d06_not_cleared['hull_lower']:.6f}` and `{d06_not_cleared['status']}`. A zero risk retained denominator `12/11/1/0`, returned no hull, and abstained.

These examples verify arithmetic and fail-closed behavior. They do **not** show that the chosen percentile quantile has calibrated coverage, generate the required paired outer block-bootstrap/inner D04 draws, establish independent configuration units, compare methods, or certify the D04 reference. Those remain open implementation and evidence gates.

## Denominators

| Fixture | Draw planned / valid / failed / abstained | Result |
| --- | ---: | --- |
| D04 hand calculation | 3 / 3 / 0 / 0 | one reducer result, uncertified |
| D06 synthetic clear | 12 / 12 / 0 / 0 | threshold clears on constructed numbers only |
| D06 synthetic below threshold | 12 / 12 / 0 / 0 | no clearance |
| D06 zero-risk case | 12 / 11 / 1 / 0 | abstain, ratio undefined |
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    artifacts = sorted(path for path in out.iterdir() if path.is_file())
    (out / "artifact_hashes.json").write_text(
        json.dumps({path.name: _sha256(path) for path in artifacts}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "output": str(out),
        "d04_risk": d04["risk"],
        "d06_statuses": {
            name: scenarios[name]["status"]
            for name in ("d06_synthetic_hull_clears", "d06_synthetic_hull_does_not_clear", "d06_synthetic_zero_risk_abstains")
        },
        "all_physical_scores_null": all(value["physical_score"] is None for value in scenarios.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build_report(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
