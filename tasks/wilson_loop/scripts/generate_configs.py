"""Generate pure-gauge Wilson-loop datasets with PyQUDA."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "tasks" / "wilson_loop" / "dataset" / "test_small"
DEFAULT_RESOURCE_PATH = PROJECT_ROOT / ".cache" / "quda"
NULLPTR = np.empty((0, 0), "<c16")


@dataclass(frozen=True)
class GenerationConfig:
    latt: tuple[int, int, int, int]
    beta: float
    n_configs: int
    warmup: int
    save_every: int
    traj_length: float
    n_steps: int
    seed: int
    output: Path
    resource_path: Path


def parse_args(argv: list[str] | None = None) -> GenerationConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latt", default="8,8,8,16", help="Lattice size as Lx,Ly,Lz,Lt.")
    parser.add_argument("--beta", type=float, default=5.8, help="Wilson gauge coupling beta.")
    parser.add_argument("--n-configs", type=int, default=100, help="Number of saved configurations.")
    parser.add_argument("--warmup", type=int, default=500, help="Number of warmup trajectories.")
    parser.add_argument("--save-every", type=int, default=5, help="Save one config every N post-warmup trajectories.")
    parser.add_argument("--traj-length", type=float, default=1.0, help="Molecular dynamics trajectory length.")
    parser.add_argument("--n-steps", type=int, default=100, help="Integrator steps per trajectory.")
    parser.add_argument("--seed", type=int, default=10086, help="Random seed for HMC.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output dataset directory.")
    parser.add_argument(
        "--resource-path",
        type=Path,
        default=DEFAULT_RESOURCE_PATH,
        help="QUDA resource directory passed to pyquda_utils.core.init().",
    )
    args = parser.parse_args(argv)

    try:
        latt = tuple(int(part) for part in args.latt.split(","))
    except ValueError as exc:
        raise SystemExit("--latt must be a comma-separated list of integers like 8,8,8,16.") from exc

    if len(latt) != 4:
        raise SystemExit("--latt must contain exactly four integers: Lx,Ly,Lz,Lt.")
    if any(length <= 0 for length in latt):
        raise SystemExit("All lattice extents must be positive.")
    if args.n_configs <= 0:
        raise SystemExit("--n-configs must be positive.")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative.")
    if args.save_every <= 0:
        raise SystemExit("--save-every must be positive.")
    if args.traj_length <= 0:
        raise SystemExit("--traj-length must be positive.")
    if args.n_steps <= 0:
        raise SystemExit("--n-steps must be positive.")

    return GenerationConfig(
        latt=latt,
        beta=args.beta,
        n_configs=args.n_configs,
        warmup=args.warmup,
        save_every=args.save_every,
        traj_length=args.traj_length,
        n_steps=args.n_steps,
        seed=args.seed,
        output=args.output.resolve(),
        resource_path=args.resource_path.resolve(),
    )


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _load_pyquda() -> dict[str, Any]:
    if os.environ.get("LATTICEARENA_FORCE_MISSING_PYQUDA") == "1":
        exc = "forced missing dependency for test"
        message = "\n".join(
            [
                "PyQUDA is required to generate Wilson-loop gauge configurations.",
                "Install QUDA first, then install the Python packages:",
                "  export QUDA_PATH=/abs/path/to/quda",
                "  python3 -m pip install pyquda pyquda-utils gmpy2",
                f"Original import error: {exc}",
            ]
        )
        raise SystemExit(message)

    try:
        from pyquda.action import GaugeAction
        from pyquda.hmc import HMC as BaseHMC, O2Nf1Ng0V
        from pyquda.quda import loadGaugeQuda, updateGaugeFieldQuda
        from pyquda_utils import core
        from pyquda_utils.hmc_param import wilsonGaugeLoopParam
    except (ImportError, ModuleNotFoundError) as exc:
        message = "\n".join(
            [
                "PyQUDA is required to generate Wilson-loop gauge configurations.",
                "Install QUDA first, then install the Python packages:",
                "  export QUDA_PATH=/abs/path/to/quda",
                "  python3 -m pip install pyquda pyquda-utils gmpy2",
                f"Original import error: {exc}",
            ]
        )
        raise SystemExit(message) from None

    class PureGaugeHMC(BaseHMC):
        """Work around PyQUDA resident-field issues for pure-gauge HMC updates."""

        def gaugeForce(self, dt: float):
            for monomial in self.gauge_monomials:
                monomial.force(dt, None)

        def updateGauge(self, dt: float):
            updateGaugeFieldQuda(NULLPTR, NULLPTR, dt, False, True, self.gauge_param)
            loadGaugeQuda(NULLPTR, self.gauge_param)

    return {
        "GaugeAction": GaugeAction,
        "HMC": PureGaugeHMC,
        "O2Nf1Ng0V": O2Nf1Ng0V,
        "core": core,
        "wilsonGaugeLoopParam": wilsonGaugeLoopParam,
    }


def _ensure_pyquda_runtime(core: Any, resource_path: Path) -> None:
    try:
        core.init(resource_path=str(resource_path))
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        if missing == "cupy":
            raise SystemExit(
                "\n".join(
                    [
                        "PyQUDA found the Python packages but the GPU runtime is incomplete.",
                        "Missing dependency: cupy",
                        "Install a CuPy wheel that matches your CUDA runtime.",
                        "For CUDA 12 systems in this repository:",
                        '  python3 -m pip install -e ".[gpu-cuda12]"',
                        f"Original import error: {exc}",
                    ]
                )
            ) from None
        raise


def _write_metadata(config: GenerationConfig, output_dir: Path) -> None:
    metadata = {
        "lattice": list(config.latt),
        "beta": config.beta,
        "action": "wilson",
        "integrator": "O2Nf1Ng0V",
        "warmup": config.warmup,
        "save_every": config.save_every,
        "traj_length": config.traj_length,
        "n_steps": config.n_steps,
        "n_configs": config.n_configs,
        "seed": config.seed,
        "output_format": {
            "shape": ["Nd", "Lx", "Ly", "Lz", "Lt", "Nc", "Nc"],
            "dtype": "complex128",
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "resource_path": str(config.resource_path),
        "pyquda_version": _package_version("pyquda"),
        "pyquda_utils_version": _package_version("pyquda-utils"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _update_metadata(output_dir: Path, updates: dict[str, Any]) -> None:
    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(updates)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _manifest_writer(output_dir: Path) -> tuple[Any, csv.DictWriter]:
    manifest_fp = (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        manifest_fp,
        fieldnames=[
            "config_id",
            "trajectory",
            "plaquette",
            "plaquette_spatial",
            "plaquette_temporal",
            "max_link_deviation_from_identity",
            "accepted",
            "hmc_seconds",
        ],
    )
    writer.writeheader()
    return manifest_fp, writer


def _normalize_plaquette(plaquette: Any) -> tuple[float, float, float]:
    if isinstance(plaquette, (list, tuple)) and len(plaquette) >= 3:
        return float(plaquette[0]), float(plaquette[1]), float(plaquette[2])
    value = float(plaquette)
    return value, value, value


def _max_link_deviation_from_identity(gauge_lexico: Any) -> float:
    eye = np.eye(3, dtype=np.complex128)
    return float(np.max(np.abs(gauge_lexico - eye)))


def generate(config: GenerationConfig) -> None:
    pyquda = _load_pyquda()
    try:
        from .gauge_io import save_task_gauge_npy
    except ImportError:
        from gauge_io import save_task_gauge_npy

    core = pyquda["core"]
    GaugeAction = pyquda["GaugeAction"]
    HMC = pyquda["HMC"]
    O2Nf1Ng0V = pyquda["O2Nf1Ng0V"]
    wilsonGaugeLoopParam = pyquda["wilsonGaugeLoopParam"]

    config.output.mkdir(parents=True, exist_ok=True)
    config.resource_path.mkdir(parents=True, exist_ok=True)
    _write_metadata(config, config.output)
    manifest_fp, manifest_writer = _manifest_writer(config.output)
    try:
        _ensure_pyquda_runtime(core, config.resource_path)
        latt_info = core.LatticeInfo(list(config.latt))
        hmc = HMC(
            latt_info,
            [GaugeAction(latt_info, wilsonGaugeLoopParam(), config.beta)],
            O2Nf1Ng0V(config.n_steps),
        )
        gauge = core.LatticeGauge(latt_info)
        hmc.initialize(config.seed, gauge)

        width = max(4, len(str(config.n_configs)))
        saved = 0
        trajectory = 0
        warmup_accepted = 0
        post_warmup_accepted = 0

        initial_plaq = hmc.plaquette()
        print(f"Trajectory 0: plaquette={initial_plaq}")
        initial_total, _, _ = _normalize_plaquette(initial_plaq)
        last_total_plaquette = initial_total
        max_saved_deviation = 0.0
        saved_plaquette_keys: set[str] = set()
        saved_configs_accepted = 0

        while saved < config.n_configs:
            trajectory += 1
            started = perf_counter()

            hmc.gaussMom()
            kinetic_old = hmc.momAction()
            potential_old = hmc.gaugeAction()
            energy_old = kinetic_old + potential_old

            hmc.integrate(config.traj_length, 2e-14)

            kinetic = hmc.momAction()
            potential = hmc.gaugeAction()
            energy = kinetic + potential

            accepted = hmc.accept(energy - energy_old)
            warmup = trajectory <= config.warmup
            if accepted:
                if warmup:
                    warmup_accepted += 1
                else:
                    post_warmup_accepted += 1
            if accepted or warmup:
                hmc.saveGauge(gauge)
            else:
                hmc.loadGauge(gauge)

            plaquette = hmc.plaquette()
            plaq_total, plaq_spatial, plaq_temporal = _normalize_plaquette(plaquette)
            last_total_plaquette = plaq_total
            elapsed = perf_counter() - started

            print(
                " ".join(
                    [
                        f"Trajectory {trajectory}:",
                        f"plaquette={plaquette}",
                        f"accepted={accepted}",
                        f"warmup={warmup}",
                        f"state_updated={accepted or warmup}",
                        f"hmc_seconds={elapsed:.3f}",
                        f"acceptance_rate={exp(min(energy_old - energy, 0)) * 100:.2f}%",
                    ]
                )
            )

            post_warmup = trajectory - config.warmup
            if post_warmup > 0 and post_warmup % config.save_every == 0:
                saved += 1
                saved_configs_accepted += int(accepted)
                filename = f"cfg_{saved:0{width}d}.npy"
                gauge_lexico = gauge.lexico()
                max_link_deviation = _max_link_deviation_from_identity(gauge_lexico)
                max_saved_deviation = max(max_saved_deviation, max_link_deviation)
                save_task_gauge_npy(config.output / filename, gauge_lexico)
                plaquette_key = f"{plaq_total:.16e}|{plaq_spatial:.16e}|{plaq_temporal:.16e}"
                saved_plaquette_keys.add(plaquette_key)
                manifest_writer.writerow(
                    {
                        "config_id": saved,
                        "trajectory": trajectory,
                        "plaquette": f"{plaq_total:.16e}",
                        "plaquette_spatial": f"{plaq_spatial:.16e}",
                        "plaquette_temporal": f"{plaq_temporal:.16e}",
                        "max_link_deviation_from_identity": f"{max_link_deviation:.16e}",
                        "accepted": int(accepted),
                        "hmc_seconds": f"{elapsed:.6f}",
                    }
                )
                manifest_fp.flush()
                print(f"Saved {saved}/{config.n_configs}: {filename}")

        evolution_detected = (abs(last_total_plaquette - initial_total) > 1e-12) or (max_saved_deviation > 1e-12)
        _update_metadata(
            config.output,
            {
                "initial_plaquette": initial_total,
                "final_plaquette": last_total_plaquette,
                "max_saved_link_deviation_from_identity": max_saved_deviation,
                "evolution_detected": evolution_detected,
                "total_trajectories": trajectory,
                "warmup_accepted_trajectories": warmup_accepted,
                "post_warmup_accepted_trajectories": post_warmup_accepted,
                "post_warmup_attempted_trajectories": max(trajectory - config.warmup, 0),
                "post_warmup_acceptance_rate": (
                    post_warmup_accepted / max(trajectory - config.warmup, 1)
                    if trajectory > config.warmup
                    else 0.0
                ),
                "saved_configs_accepted": saved_configs_accepted,
                "saved_configs_total": saved,
                "unique_saved_plaquette_count": len(saved_plaquette_keys),
            },
        )
        if not evolution_detected:
            print(
                "WARNING: No gauge evolution was detected in the saved configurations. "
                "Plaquette stayed at the cold-start value and links remained equal to identity."
            )
    finally:
        manifest_fp.close()


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    generate(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
