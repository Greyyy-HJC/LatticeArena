"""Measure Wilson-loop correlators for a spatial-operator submission."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.wilson_loop.interface import SpatialOperator
from tasks.wilson_loop.scripts.gauge_io import load_task_gauge_npy


def load_submission(submission_name: str) -> SpatialOperator:
    """Import and instantiate a submission from submissions/<name>.py."""

    module = importlib.import_module(f"tasks.wilson_loop.submissions.{submission_name}")
    for value in module.__dict__.values():
        if isinstance(value, type) and issubclass(value, SpatialOperator) and value is not SpatialOperator:
            return value()
    raise ValueError(f"No SpatialOperator submission found in module '{submission_name}'.")


def dagger(field: np.ndarray) -> np.ndarray:
    """Hermitian conjugate on the color indices of a lattice matrix field."""

    return np.swapaxes(field.conj(), -1, -2)


def identity_field(lx: int, ly: int, lz: int) -> np.ndarray:
    """Identity matrix at every spatial site."""

    return np.broadcast_to(np.eye(3, dtype=np.complex128), (lx, ly, lz, 3, 3)).copy()


def temporal_line(gauge_field: np.ndarray, t_source: int, tau: int) -> np.ndarray:
    """Build temporal Wilson lines for all spatial sites from t_source to t_source + tau."""

    _, lx, ly, lz, lt, _, _ = gauge_field.shape
    line = identity_field(lx, ly, lz)
    if tau == 0:
        return line

    time_links = gauge_field[3]
    for step in range(tau):
        t = (t_source + step) % lt
        line = line @ time_links[:, :, :, t]
    return line


def measure_single_config(
    gauge_field: np.ndarray,
    operator: SpatialOperator,
    r_values: list[int],
    t_values: list[int],
) -> np.ndarray:
    """Measure C(r, t) on one gauge configuration."""

    _, lx, ly, lz, lt, _, _ = gauge_field.shape
    latt_size = (lx, ly, lz, lt)
    operator.setup(gauge_field, latt_size)

    correlator = np.zeros((len(r_values), len(t_values)), dtype=np.complex128)

    for r_idx, r in enumerate(r_values):
        direction_values = []
        for direction in range(3):
            time_values = []
            for tau_idx, tau in enumerate(t_values):
                source_values = []
                for t_source in range(lt):
                    spatial_line_source = operator.compute(gauge_field, r, direction, t_source)
                    temporal_forward_shifted = np.roll(
                        temporal_line(gauge_field, t_source, tau),
                        shift=-r,
                        axis=direction,
                    )
                    spatial_line_sink = operator.compute(gauge_field, r, direction, (t_source + tau) % lt)
                    temporal_backward = dagger(temporal_line(gauge_field, t_source, tau))

                    wilson_loop = (
                        spatial_line_source
                        @ temporal_forward_shifted
                        @ dagger(spatial_line_sink)
                        @ temporal_backward
                    )
                    loop_trace = np.trace(wilson_loop, axis1=-2, axis2=-1)
                    source_values.append(np.mean(loop_trace))

                time_values.append(np.mean(source_values))
            direction_values.append(time_values)

        correlator[r_idx] = np.mean(np.asarray(direction_values), axis=0)

    return correlator


def list_gauge_files(dataset_path: Path) -> list[Path]:
    """List task-format gauge files from a file or directory path."""

    if dataset_path.is_file():
        return [dataset_path]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    return sorted(dataset_path.glob("cfg_*.npy"))


def measure_dataset(
    dataset_path: Path,
    operator: SpatialOperator,
    r_values: list[int],
    t_values: list[int],
    max_configs: int | None = None,
) -> dict[str, np.ndarray | list[str] | list[int]]:
    """Measure Wilson-loop correlators for every config in a dataset."""

    gauge_files = list_gauge_files(dataset_path)
    if not gauge_files:
        raise FileNotFoundError(f"No gauge configs found under {dataset_path}")
    if max_configs is not None:
        if max_configs <= 0:
            raise ValueError("max_configs must be positive when provided.")
        gauge_files = gauge_files[:max_configs]

    per_config = []
    for gauge_file in gauge_files:
        gauge_field = load_task_gauge_npy(gauge_file)
        per_config.append(measure_single_config(gauge_field, operator, r_values, t_values))

    per_config_array = np.asarray(per_config, dtype=np.complex128)
    return {
        "files": [str(path) for path in gauge_files],
        "r_values": r_values,
        "t_values": t_values,
        "per_config": per_config_array,
        "mean": np.mean(per_config_array, axis=0),
    }


def parse_value_list(arg: str) -> list[int]:
    values = [int(part) for part in arg.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    if any(value < 0 for value in values):
        raise ValueError("All values must be non-negative.")
    return values


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Wilson-loop correlators for one submission.")
    parser.add_argument("--submission", type=str, required=True, help="Submission module name under submissions/")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("tasks/wilson_loop/dataset/test_small"),
        help="Gauge config directory or one cfg_XXXX.npy file in task ordering.",
    )
    parser.add_argument("--r-values", type=str, default="1,2,3", help="Comma-separated spatial separations.")
    parser.add_argument("--t-values", type=str, default="0,1,2,3,4", help="Comma-separated temporal extents.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional .npz output path for measured correlators.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Optional cap on the number of configs loaded from a dataset directory.",
    )
    args = parser.parse_args()

    submission = load_submission(args.submission)
    r_values = parse_value_list(args.r_values)
    t_values = parse_value_list(args.t_values)
    result = measure_dataset(args.dataset_path, submission, r_values, t_values, max_configs=args.max_configs)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output,
            files=np.asarray(result["files"], dtype=object),
            r_values=np.asarray(result["r_values"], dtype=np.int64),
            t_values=np.asarray(result["t_values"], dtype=np.int64),
            per_config=result["per_config"],
            mean=result["mean"],
        )

    print(
        json.dumps(
            {
                "submission": submission.meta.name,
                "dataset_path": str(args.dataset_path),
                "n_configs": len(result["files"]),
                "max_configs": args.max_configs,
                "r_values": result["r_values"],
                "t_values": result["t_values"],
                "mean": result["mean"],
                "output": str(args.output) if args.output is not None else None,
            },
            indent=2,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
