"""Benchmark smoke tests for wilson_loop."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tasks.wilson_loop.benchmark.metrics import benchmark_submission
from tasks.wilson_loop.submissions.plain import PlainWilsonLine
from tasks.wilson_loop.scripts.gauge_io import save_task_gauge_npy
from tasks.wilson_loop.tests.validation import identity_gauge_field


REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_identity_dataset(dataset_dir: Path) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    gauge_task = identity_gauge_field((4, 4, 4, 8))
    gauge_lexico = np.transpose(gauge_task, (0, 4, 3, 2, 1, 5, 6))
    save_task_gauge_npy(dataset_dir / "cfg_0001.npy", gauge_lexico)
    return dataset_dir


def test_benchmark_submission_on_identity_dataset(tmp_path: Path) -> None:
    dataset_dir = _write_identity_dataset(tmp_path / "dataset")
    summary = benchmark_submission(
        PlainWilsonLine(),
        dataset_path=str(dataset_dir),
        r_values=[1, 2],
        t_values=[0, 1, 2],
    )

    assert summary["n_configs"] == 1
    assert summary["score"] > 0.0
    assert summary["mean_correlator_real"] == [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]


def test_benchmark_submission_respects_max_configs(tmp_path: Path) -> None:
    dataset_dir = _write_identity_dataset(tmp_path / "dataset")
    gauge_task = identity_gauge_field((4, 4, 4, 8))
    gauge_lexico = np.transpose(gauge_task, (0, 4, 3, 2, 1, 5, 6))
    save_task_gauge_npy(dataset_dir / "cfg_0002.npy", gauge_lexico)

    summary = benchmark_submission(
        PlainWilsonLine(),
        dataset_path=str(dataset_dir),
        r_values=[1],
        t_values=[0, 1],
        max_configs=1,
    )

    assert summary["n_configs"] == 1
    assert len(summary["files"]) == 1


def test_benchmark_cli_smoke(tmp_path: Path) -> None:
    dataset_dir = _write_identity_dataset(tmp_path / "dataset")
    output_dir = tmp_path / "results"

    result = subprocess.run(
        [
            sys.executable,
            "tasks/wilson_loop/benchmark/run.py",
            "--submission",
            "plain",
            "--dataset-path",
            str(dataset_dir),
            "--r-values",
            "1,2",
            "--t-values",
            "0,1,2",
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["task"] == "wilson_loop"
    assert payload["submission"] == "plain"
    assert (output_dir / "plain.json").exists()
