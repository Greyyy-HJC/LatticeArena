"""CLI tests for Wilson-loop config generation."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "tasks" / "wilson_loop" / "scripts" / "generate_configs.py"


def test_generate_configs_reports_missing_pyquda_cleanly() -> None:
    env = dict(**os.environ, LATTICEARENA_FORCE_MISSING_PYQUDA="1")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--n-configs", "1", "--warmup", "0", "--save-every", "1"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    stderr = result.stderr or result.stdout
    assert "PyQUDA is required" in stderr
    assert "python3 -m pip install pyquda pyquda-utils gmpy2" in stderr
    assert "Traceback" not in stderr


def test_generate_configs_small_smoke(tmp_path: Path) -> None:
    pytest.importorskip("pyquda")
    pytest.importorskip("pyquda_utils")

    output_dir = tmp_path / "dataset"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--latt",
            "4,4,4,8",
            "--beta",
            "5.8",
            "--n-configs",
            "1",
            "--warmup",
            "0",
            "--save-every",
            "1",
            "--traj-length",
            "0.25",
            "--n-steps",
            "2",
            "--output",
            str(output_dir),
            "--resource-path",
            str(tmp_path / ".cache" / "quda"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        dependency_markers = [
            "PyQUDA is required",
            "GPU runtime is incomplete",
            "Original import error:",
            "No module named 'gmpy2'",
            "No module named 'cupy'",
            "No module named 'pyquda'",
            "No module named 'pyquda_utils'",
        ]
        stderr = result.stderr or result.stdout
        if any(marker in stderr for marker in dependency_markers):
            pytest.skip(f"PyQUDA runtime is not fully available for integration smoke test:\n{stderr}")
        raise AssertionError(f"generate_configs.py failed unexpectedly:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{stderr}")

    assert "Saved 1/1" in result.stdout

    cfg_path = output_dir / "cfg_0001.npy"
    metadata_path = output_dir / "metadata.json"
    manifest_path = output_dir / "manifest.csv"

    assert cfg_path.exists()
    assert metadata_path.exists()
    assert manifest_path.exists()

    gauge = np.load(cfg_path, allow_pickle=False)
    assert gauge.shape == (4, 4, 4, 4, 8, 3, 3)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["action"] == "wilson"
    assert metadata["lattice"] == [4, 4, 4, 8]
    assert "post_warmup_acceptance_rate" in metadata
    assert "saved_configs_accepted" in metadata
    assert "unique_saved_plaquette_count" in metadata

    with manifest_path.open("r", newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    assert len(rows) == 1
    assert rows[0]["config_id"] == "1"
