"""Measure boosted pion two-point correlators for one submission."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.pion_2pt.interface import PionInterpolatingOperator


_PYQUDA_INITIALIZED = False


@dataclass(frozen=True)
class EnsembleConfig:
    """Metadata required to run the fixed pion 2pt workflow."""

    name: str
    latt_size: tuple[int, int, int, int]
    xi_0: float
    nu: float
    mass: float
    csw_r: float
    csw_t: float
    t_boundary: int
    source_times: tuple[int, ...]
    benchmark_momentum: tuple[int, int, int]
    gauge_glob: str
    gauge_format: str
    lattice_spacing_fm: float | None
    resource_path: str | None
    invert_tolerance: float
    invert_maxiter: int


def load_submission(submission_name: str) -> PionInterpolatingOperator:
    """Import and instantiate a submission from submissions/<name>.py."""

    module = importlib.import_module(f"tasks.pion_2pt.submissions.{submission_name}")
    for value in module.__dict__.values():
        if (
            isinstance(value, type)
            and issubclass(value, PionInterpolatingOperator)
            and value is not PionInterpolatingOperator
        ):
            return value()
    raise ValueError(
        f"No PionInterpolatingOperator submission found in module '{submission_name}'."
    )


def parse_momentum_list(arg: str) -> list[tuple[int, int, int]]:
    """Parse ``a,b,c;d,e,f`` into lattice momentum triplets."""

    momenta: list[tuple[int, int, int]] = []
    for chunk in arg.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [int(value.strip()) for value in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Expected three integers per momentum, got '{chunk}'.")
        momenta.append((parts[0], parts[1], parts[2]))
    if not momenta:
        raise ValueError("Expected at least one momentum triplet.")
    return momenta


def parse_time_list(arg: str) -> list[int]:
    """Parse a comma-separated list of source times."""

    values = [int(part.strip()) for part in arg.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one source time.")
    if any(value < 0 for value in values):
        raise ValueError("Source times must be non-negative.")
    return values


def load_ensemble_config(dataset_path: Path) -> tuple[EnsembleConfig, Path]:
    """Load task-local dataset metadata from ``ensemble.json``."""

    metadata_path = dataset_path
    if dataset_path.is_dir():
        metadata_path = dataset_path / "ensemble.json"
    if metadata_path.name != "ensemble.json":
        raise FileNotFoundError(
            "dataset_path must be a dataset directory or an ensemble.json file."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing ensemble metadata: {metadata_path}")

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    anisotropy = data.get("anisotropy", {})
    clover = data.get("clover", {})
    ensemble = EnsembleConfig(
        name=str(data.get("name", metadata_path.parent.name)),
        latt_size=tuple(int(v) for v in data["latt_size"]),
        xi_0=float(anisotropy.get("xi_0", 1.0)),
        nu=float(anisotropy.get("nu", 1.0)),
        mass=float(clover["mass"]),
        csw_r=float(clover["csw_r"]),
        csw_t=float(clover["csw_t"]),
        t_boundary=int(clover.get("t_boundary", -1)),
        source_times=tuple(int(v) for v in data.get("source_times", [0])),
        benchmark_momentum=tuple(
            int(v) for v in data.get("benchmark_momentum", [3, 3, 3])
        ),
        gauge_glob=str(data.get("gauge_glob", "*.nersc")),
        gauge_format=str(data.get("format", "nersc")),
        lattice_spacing_fm=(
            None
            if data.get("lattice_spacing_fm") is None
            else float(data["lattice_spacing_fm"])
        ),
        resource_path=data.get("resource_path"),
        invert_tolerance=float(data.get("invert_tolerance", 1e-8)),
        invert_maxiter=int(data.get("invert_maxiter", 10000)),
    )
    if len(ensemble.latt_size) != 4:
        raise ValueError("latt_size must have four entries.")
    return ensemble, metadata_path.parent


def list_gauge_files(dataset_path: Path, gauge_glob: str | None = None) -> list[Path]:
    """List gauge files for a dataset directory or metadata file."""

    ensemble, dataset_dir = load_ensemble_config(dataset_path)
    pattern = gauge_glob or ensemble.gauge_glob
    gauge_files = sorted(path for path in dataset_dir.glob(pattern) if path.is_file())
    if not gauge_files:
        raise FileNotFoundError(
            f"No gauge files matching '{pattern}' were found under {dataset_dir}."
        )
    return gauge_files


def _load_pyquda() -> dict[str, Any]:
    try:
        import cupy as cp
        from pyquda_utils import core, io, source
    except (ImportError, ModuleNotFoundError) as exc:
        message = "\n".join(
            [
                "PyQUDA is required to run pion_2pt measurements.",
                "Install QUDA first, then install the Python packages:",
                "  export QUDA_PATH=/abs/path/to/quda",
                '  python3 -m pip install -e ".[gpu-cuda12]"',
                f"Original import error: {exc}",
            ]
        )
        raise SystemExit(message) from None
    return {"cp": cp, "core": core, "io": io, "source": source}


def _ensure_pyquda_runtime(core: Any, resource_path: Path) -> None:
    global _PYQUDA_INITIALIZED
    if _PYQUDA_INITIALIZED:
        return
    resource_path.mkdir(parents=True, exist_ok=True)
    core.init(resource_path=str(resource_path))
    _PYQUDA_INITIALIZED = True


def _gamma5() -> np.ndarray:
    return np.diag([1.0, 1.0, -1.0, -1.0]).astype(np.complex128)


def _profile_to_tzyx(profile: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(profile, dtype=np.complex128), (2, 1, 0))


def _build_source_propagator(
    cp: Any,
    core: Any,
    source_utils: Any,
    latt_info: Any,
    source_profile: np.ndarray,
    t_source: int,
) -> Any:
    lx, ly, lz, lt = latt_info.global_size
    if source_profile.shape != (lx, ly, lz):
        raise ValueError(
            "source_profile has incompatible shape. "
            f"Expected {(lx, ly, lz)}, got {source_profile.shape}."
        )

    source_field = np.zeros((lt, lz, ly, lx), dtype=np.complex128)
    source_field[t_source] = _profile_to_tzyx(source_profile)
    base_phase = cp.asarray(latt_info.evenodd(source_field, False))

    propagator = core.LatticePropagator(latt_info)
    for spin in range(core.Ns):
        for color in range(core.Nc):
            source_phase = cp.zeros(
                (*base_phase.shape, core.Nc), dtype=base_phase.dtype
            )
            source_phase[..., color] = base_phase
            fermion = source_utils.source(
                latt_info, "colorvector", t_source, spin, color, source_phase
            )
            propagator.setFermion(fermion, spin, color)
    return propagator


def _build_sink_weights(cp: Any, latt_info: Any, sink_profile: np.ndarray) -> Any:
    lx, ly, lz, lt = latt_info.global_size
    if sink_profile.shape != (lx, ly, lz):
        raise ValueError(
            "sink_profile has incompatible shape. "
            f"Expected {(lx, ly, lz)}, got {sink_profile.shape}."
        )

    sink_field = np.zeros((lt, lz, ly, lx), dtype=np.complex128)
    sink_field[:] = _profile_to_tzyx(sink_profile)[None, ...]
    sink_weights = latt_info.evenodd(sink_field, False)
    return cp.asarray(sink_weights)


def _contract_correlator(
    cp: Any,
    propagator: Any,
    sink_weights: Any,
    gamma_matrix: np.ndarray,
) -> np.ndarray:
    gamma5 = cp.asarray(_gamma5())
    gamma_sink = gamma5 @ cp.asarray(np.asarray(gamma_matrix, dtype=np.complex128))
    gamma_source = gamma5 @ cp.asarray(
        np.asarray(gamma_matrix, dtype=np.complex128).conj().T
    )
    correlator = cp.einsum(
        "wtzyx,wtzyxjiba,jk,wtzyxklba,li->t",
        sink_weights.conj(),
        propagator.data.conj(),
        gamma_sink,
        propagator.data,
        gamma_source,
    )
    return cp.asnumpy(correlator)


def measure_single_config(
    gauge_file: Path,
    operator: PionInterpolatingOperator,
    ensemble: EnsembleConfig,
    *,
    momentum_modes: list[tuple[int, int, int]],
    source_times: list[int],
    resource_path: Path,
) -> np.ndarray:
    """Measure correlators for one gauge configuration."""

    pyquda = _load_pyquda()
    cp = pyquda["cp"]
    core = pyquda["core"]
    io = pyquda["io"]
    source_utils = pyquda["source"]

    _ensure_pyquda_runtime(core, resource_path)
    latt_info = core.LatticeInfo(
        list(ensemble.latt_size), ensemble.t_boundary, ensemble.xi_0 / ensemble.nu
    )
    dirac = core.getClover(
        latt_info,
        ensemble.mass,
        ensemble.invert_tolerance,
        ensemble.invert_maxiter,
        ensemble.xi_0,
        ensemble.csw_r,
        ensemble.csw_t,
        None,
    )

    if ensemble.gauge_format.lower() != "nersc":
        raise ValueError(
            f"Unsupported gauge format '{ensemble.gauge_format}'. Only 'nersc' is implemented."
        )
    gauge = io.readNERSCGauge(str(gauge_file))
    dirac.loadGauge(gauge)
    operator.setup(gauge, ensemble.latt_size, ensemble.lattice_spacing_fm)

    per_source = np.zeros(
        (len(source_times), len(momentum_modes), ensemble.latt_size[3]),
        dtype=np.complex128,
    )
    for t_index, t_source in enumerate(source_times):
        for p_index, momentum_mode in enumerate(momentum_modes):
            components = operator.build(
                gauge, momentum_mode=momentum_mode, t_source=t_source
            )
            source_propagator = _build_source_propagator(
                cp, core, source_utils, latt_info, components.source_profile, t_source
            )
            solved = core.invertPropagator(dirac, source_propagator)
            sink_weights = _build_sink_weights(cp, latt_info, components.sink_profile)
            correlator = _contract_correlator(
                cp, solved, sink_weights, components.gamma_matrix
            )
            per_source[t_index, p_index] = np.roll(correlator, -t_source)

    return np.mean(per_source, axis=0)


def measure_dataset(
    dataset_path: Path,
    operator: PionInterpolatingOperator,
    *,
    momentum_modes: list[tuple[int, int, int]] | None = None,
    source_times: list[int] | None = None,
    max_configs: int | None = None,
    resource_path: Path | None = None,
) -> dict[str, Any]:
    """Measure pion correlators for every config in a dataset."""

    ensemble, dataset_dir = load_ensemble_config(dataset_path)
    selected_momenta = momentum_modes or [ensemble.benchmark_momentum]
    selected_sources = source_times or list(ensemble.source_times)
    gauge_files = list_gauge_files(dataset_path)
    if max_configs is not None:
        if max_configs <= 0:
            raise ValueError("max_configs must be positive when provided.")
        gauge_files = gauge_files[:max_configs]

    resolved_resource_path = resource_path
    if resolved_resource_path is None:
        default_dir = dataset_dir / ".cache"
        if ensemble.resource_path is not None:
            default_dir = (dataset_dir / ensemble.resource_path).resolve()
        resolved_resource_path = default_dir

    per_config = []
    for gauge_file in gauge_files:
        per_config.append(
            measure_single_config(
                gauge_file,
                operator,
                ensemble,
                momentum_modes=selected_momenta,
                source_times=selected_sources,
                resource_path=resolved_resource_path,
            )
        )

    per_config_array = np.asarray(per_config, dtype=np.complex128)
    return {
        "ensemble_name": ensemble.name,
        "dataset_path": str(dataset_dir),
        "files": [str(path) for path in gauge_files],
        "latt_size": list(ensemble.latt_size),
        "source_times": list(selected_sources),
        "momentum_modes": [list(momentum) for momentum in selected_momenta],
        "per_config": per_config_array,
        "mean": np.mean(per_config_array, axis=0),
    }


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return value.tolist()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure pion two-point correlators for one submission."
    )
    parser.add_argument(
        "--submission",
        type=str,
        required=True,
        help="Submission module name under submissions/",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("tasks/pion_2pt/dataset/test_small"),
        help="Dataset directory or ensemble.json path.",
    )
    parser.add_argument(
        "--momenta",
        type=str,
        default=None,
        help="Semicolon-separated lattice momenta, for example '3,3,3;0,0,0'.",
    )
    parser.add_argument(
        "--source-times",
        type=str,
        default=None,
        help="Comma-separated source times. Defaults to ensemble metadata.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Optional cap on the number of gauge configurations to load.",
    )
    parser.add_argument(
        "--resource-path",
        type=Path,
        default=None,
        help="Optional PyQUDA resource cache path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional .npz output path for measured correlators.",
    )
    args = parser.parse_args()

    submission = load_submission(args.submission)
    result = measure_dataset(
        dataset_path=args.dataset_path,
        operator=submission,
        momentum_modes=parse_momentum_list(args.momenta) if args.momenta else None,
        source_times=parse_time_list(args.source_times) if args.source_times else None,
        max_configs=args.max_configs,
        resource_path=args.resource_path,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output,
            files=np.asarray(result["files"], dtype=object),
            latt_size=np.asarray(result["latt_size"], dtype=np.int64),
            source_times=np.asarray(result["source_times"], dtype=np.int64),
            momentum_modes=np.asarray(result["momentum_modes"], dtype=np.int64),
            per_config=result["per_config"],
            mean=result["mean"],
        )

    print(
        json.dumps(
            {
                "submission": submission.meta.name,
                "dataset_path": str(args.dataset_path),
                "n_configs": len(result["files"]),
                "latt_size": result["latt_size"],
                "source_times": result["source_times"],
                "momentum_modes": result["momentum_modes"],
                "output": str(args.output) if args.output is not None else None,
                "mean": result["mean"],
            },
            indent=2,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
