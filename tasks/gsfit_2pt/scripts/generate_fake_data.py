"""Generate deterministic fake data for the gsfit_2pt task."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.gsfit_2pt.dataset.synthetic import make_synthetic_cases, save_synthetic_cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fake pion 2pt data for gsfit_2pt"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tasks/gsfit_2pt/dataset/fake_data.npz"),
        help="Output .npz archive path",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=24,
        help="Number of bootstrap/jackknife-like samples",
    )
    parser.add_argument(
        "--lt", type=int, default=48, help="Temporal extent of the fake correlators"
    )
    parser.add_argument(
        "--noise-multiplier",
        type=float,
        default=1.0,
        help="Global multiplier applied to the built-in synthetic noise levels",
    )
    args = parser.parse_args()

    cases = make_synthetic_cases(
        num_samples=args.num_samples,
        noise_multiplier=args.noise_multiplier,
        lt=args.lt,
    )
    path = save_synthetic_cases(cases, args.output)

    summary = {
        "output": str(path),
        "num_cases": len(cases),
        "num_samples": args.num_samples,
        "lt": args.lt,
        "cases": [
            {
                "name": case.name,
                "true_e0": case.true_e0,
                "num_states": len(case.true_energies),
            }
            for case in cases
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
