"""Validation tests for pion_2pt operators."""

from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

from core.testing import identity_gauge_field
from tasks.pion_2pt.submissions.axial_smeared import AxialSmearedBoostedPion
from tasks.pion_2pt.submissions.plain import PlainBoostedPion
from tasks.pion_2pt.tests.validation import validate_submission


def test_plain_submission_passes_validation_helper() -> None:
    assert validate_submission(PlainBoostedPion())


def test_plain_operator_shapes() -> None:
    latt_size = (8, 8, 8, 32)
    op = PlainBoostedPion()
    gauge = identity_gauge_field(latt_size)

    op.setup(gauge, latt_size=latt_size, lattice_spacing_fm=0.09)
    components = op.build(gauge, momentum_mode=(1, 0, 0), t_source=0)

    assert components.source_profile.shape == (8, 8, 8)
    assert components.sink_profile.shape == (8, 8, 8)
    assert components.gamma_matrix.shape == (4, 4)
    assert np.iscomplexobj(components.source_profile)


def test_plain_operator_normalization() -> None:
    latt_size = (8, 8, 8, 32)
    op = PlainBoostedPion()
    gauge = identity_gauge_field(latt_size)

    op.setup(gauge, latt_size=latt_size, lattice_spacing_fm=0.09)
    components = op.build(gauge, momentum_mode=(1, 0, 0), t_source=0)

    src_norm = np.linalg.norm(components.source_profile)
    sink_norm = np.linalg.norm(components.sink_profile)
    assert np.isclose(src_norm, 1.0, atol=1e-12)
    assert np.isclose(sink_norm, 1.0, atol=1e-12)


def test_plain_operator_requires_setup() -> None:
    op = PlainBoostedPion()
    gauge = identity_gauge_field((4, 4, 4, 8))

    try:
        op.build(gauge, momentum_mode=(1, 0, 0), t_source=0)
    except RuntimeError:
        return
    raise AssertionError("Expected RuntimeError when build() is called before setup().")


def test_axial_smeared_submission_passes_validation_helper() -> None:
    assert validate_submission(AxialSmearedBoostedPion())


def test_plain_baseline_is_point_source() -> None:
    latt_size = (8, 8, 8, 16)
    op = PlainBoostedPion()
    gauge = identity_gauge_field(latt_size)

    op.setup(gauge, latt_size=latt_size, lattice_spacing_fm=None)
    components = op.build(gauge, momentum_mode=(3, 3, 3), t_source=4)

    assert np.count_nonzero(np.abs(components.source_profile) > 0) == 1
    assert components.source_profile[0, 0, 0] == 1.0


def test_axial_smeared_profile_is_normalized_and_delocalized() -> None:
    latt_size = (8, 8, 8, 16)
    op = AxialSmearedBoostedPion(sigma=1.5)
    gauge = identity_gauge_field(latt_size)

    op.setup(gauge, latt_size=latt_size, lattice_spacing_fm=None)
    components = op.build(gauge, momentum_mode=(3, 3, 3), t_source=0)

    assert np.isclose(np.linalg.norm(components.source_profile), 1.0, atol=1e-12)
    assert np.count_nonzero(np.abs(components.source_profile) > 1e-12) > 1


def test_benchmark_cli_runs_pytest_gate_by_default(monkeypatch) -> None:
    from tasks.pion_2pt.benchmark import run as benchmark_run

    class DummyTask:
        tests_path = Path("tasks/pion_2pt/tests")

        @staticmethod
        def validate(_submission: object) -> bool:
            return True

    calls: list[list[str]] = []

    def fake_subprocess_run(command: list[str], cwd: Path, check: bool):
        calls.append(command)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(benchmark_run, "_default_dataset_path", lambda: Path("."))
    monkeypatch.setattr(benchmark_run, "Pion2PtTask", DummyTask)
    monkeypatch.setattr(
        benchmark_run, "load_submission", lambda _submission_name: PlainBoostedPion()
    )
    monkeypatch.setattr(
        benchmark_run, "benchmark_submission", lambda *_args, **_kwargs: {"score": 1.0}
    )
    monkeypatch.setattr(
        benchmark_run, "save_result", lambda *_args, **_kwargs: Path("result.json")
    )
    monkeypatch.setattr(benchmark_run.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--submission", "plain", "--dataset-path", "."],
    )

    benchmark_run.main()

    assert calls
    assert calls[0][:4] == [
        sys.executable,
        "-m",
        "pytest",
        "tasks/pion_2pt/tests/test_validation.py",
    ]


def test_benchmark_cli_skip_tests_bypasses_pytest_gate(monkeypatch) -> None:
    from tasks.pion_2pt.benchmark import run as benchmark_run

    class DummyTask:
        tests_path = Path("tasks/pion_2pt/tests")

        @staticmethod
        def validate(_submission: object) -> bool:
            return True

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("pytest validation gate should be skipped")

    monkeypatch.setattr(benchmark_run, "_default_dataset_path", lambda: Path("."))
    monkeypatch.setattr(benchmark_run, "Pion2PtTask", DummyTask)
    monkeypatch.setattr(
        benchmark_run, "load_submission", lambda _submission_name: PlainBoostedPion()
    )
    monkeypatch.setattr(
        benchmark_run, "benchmark_submission", lambda *_args, **_kwargs: {"score": 1.0}
    )
    monkeypatch.setattr(
        benchmark_run, "save_result", lambda *_args, **_kwargs: Path("result.json")
    )
    monkeypatch.setattr(benchmark_run.subprocess, "run", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--submission", "plain", "--dataset-path", ".", "--skip-tests"],
    )

    benchmark_run.main()
