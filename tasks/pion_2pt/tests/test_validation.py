"""Validation tests for pion_2pt operators."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from core.testing import identity_gauge_field
from tasks.pion_2pt.benchmark.metrics import effective_mass_periodic
from tasks.pion_2pt.dirac import gamma5_matrix, gamma_t_gamma5_matrix
from tasks.pion_2pt.interface import PlaneWaveSinkSpec, PointSourceSpec
from tasks.pion_2pt.scripts.measure import (
    load_submission,
    source_contraction_gamma,
)
from tasks.pion_2pt.submissions.plain import PlainBoostedPion
from tasks.pion_2pt.submissions.temporal_axial import TemporalAxialBoostedPion
from tasks.pion_2pt.tests.validation import validate_submission


def test_plain_submission_passes_validation_helper() -> None:
    assert validate_submission(PlainBoostedPion())


def test_plain_operator_design_matches_pion_disp_setup() -> None:
    op = PlainBoostedPion()
    gauge = identity_gauge_field((8, 8, 8, 32))

    op.setup(gauge, latt_size=(8, 8, 8, 32), lattice_spacing_fm=0.09)
    source_spec = op.design_source(gauge, momentum_mode=(1, 0, 0), t_source=0)
    sink_spec = op.design_sink(gauge, momentum_mode=(1, 0, 0), t_source=0)
    gamma_matrix = op.gamma_matrix(gauge, momentum_mode=(1, 0, 0), t_source=0)

    assert source_spec == PointSourceSpec(position=(0, 0, 0))
    assert sink_spec == PlaneWaveSinkSpec(momentum_mode=(1, 0, 0))
    assert gamma_matrix.shape == (4, 4)
    assert np.allclose(gamma_matrix, gamma5_matrix())


def test_plain_operator_requires_setup() -> None:
    op = PlainBoostedPion()
    gauge = identity_gauge_field((4, 4, 4, 8))

    try:
        op.design_source(gauge, momentum_mode=(1, 0, 0), t_source=0)
    except RuntimeError:
        return
    raise AssertionError(
        "Expected RuntimeError when design_source() is called before setup()."
    )


def test_temporal_axial_submission_passes_validation_helper() -> None:
    assert validate_submission(TemporalAxialBoostedPion())


def test_load_submission_returns_module_local_class() -> None:
    submission = load_submission("temporal_axial")

    assert isinstance(submission, TemporalAxialBoostedPion)
    assert submission.meta.name == "temporal_axial"


def test_plain_baseline_is_point_source() -> None:
    op = PlainBoostedPion()
    gauge = identity_gauge_field((8, 8, 8, 16))

    op.setup(gauge, latt_size=(8, 8, 8, 16), lattice_spacing_fm=None)
    source_spec = op.design_source(gauge, momentum_mode=(3, 3, 3), t_source=4)

    assert source_spec == PointSourceSpec(position=(0, 0, 0))


def test_temporal_axial_matches_plain_source_sink_and_gamma() -> None:
    plain = PlainBoostedPion()
    temporal_axial = TemporalAxialBoostedPion()
    gauge = identity_gauge_field((8, 8, 8, 16))

    plain.setup(gauge, latt_size=(8, 8, 8, 16), lattice_spacing_fm=None)
    temporal_axial.setup(gauge, latt_size=(8, 8, 8, 16), lattice_spacing_fm=None)

    plain_source = plain.design_source(gauge, momentum_mode=(3, 3, 3), t_source=0)
    plain_sink = plain.design_sink(gauge, momentum_mode=(3, 3, 3), t_source=0)
    temporal_source = temporal_axial.design_source(
        gauge, momentum_mode=(3, 3, 3), t_source=0
    )
    temporal_sink = temporal_axial.design_sink(
        gauge, momentum_mode=(3, 3, 3), t_source=0
    )

    assert temporal_source == plain_source
    assert temporal_sink == plain_sink
    assert np.allclose(
        temporal_axial.gamma_matrix(gauge, momentum_mode=(3, 3, 3), t_source=0),
        gamma_t_gamma5_matrix(),
    )


def test_source_contraction_gamma_uses_gamma_dagger_then_gamma5() -> None:
    gamma = gamma_t_gamma5_matrix()

    expected = gamma.conj().T @ gamma5_matrix()
    old_order = gamma5_matrix() @ gamma.conj().T

    assert np.allclose(source_contraction_gamma(gamma), expected)
    assert not np.allclose(expected, old_order)


def test_effective_mass_periodic_reproduces_known_cosh_plateau() -> None:
    lt = 32
    mass = 0.37
    times = np.arange(lt, dtype=np.float64)
    correlator = np.cosh(mass * (times - lt / 2.0))

    meff = effective_mass_periodic(correlator)

    assert meff.shape == (lt - 2,)
    assert np.allclose(meff, mass, atol=1e-12)


def test_effective_mass_periodic_marks_invalid_domain_as_nan() -> None:
    correlator = np.asarray([1.0, -0.25, 1.0, 0.0, 1.0], dtype=np.float64)

    meff = effective_mass_periodic(correlator)

    assert meff.shape == (3,)
    assert np.all(np.isnan(meff))


def test_benchmark_cli_runs_pytest_gate_by_default(monkeypatch, tmp_path: Path) -> None:
    from tasks.pion_2pt.benchmark import run as benchmark_run

    calls: list[list[str]] = []

    def fake_subprocess_run(command: list[str], cwd: Path, check: bool):
        calls.append(command)

        class _Result:
            returncode = 0

        return _Result()

    def fake_benchmark_submission(*_args, artifact_dir: Path, **_kwargs):
        plot_path = artifact_dir / "plain_meff.pdf"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path.write_bytes(b"%PDF-1.4\n% fake meff plot\n")
        return {
            "score": 1.0,
            "artifacts": {"effective_mass_plot": str(plot_path)},
        }

    monkeypatch.setattr(benchmark_run, "_default_dataset_path", lambda: Path("."))
    monkeypatch.setattr(
        benchmark_run, "load_submission", lambda _submission_name: PlainBoostedPion()
    )
    monkeypatch.setattr(
        benchmark_run, "benchmark_submission", fake_benchmark_submission
    )
    monkeypatch.setattr(benchmark_run.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--submission",
            "plain",
            "--dataset-path",
            ".",
            "--output-dir",
            str(tmp_path),
        ],
    )

    benchmark_run.main()

    assert calls
    assert calls[0][:4] == [
        sys.executable,
        "-m",
        "pytest",
        "tasks/pion_2pt/tests/test_validation.py",
    ]
    assert (tmp_path / "plain.json").exists()
    assert (tmp_path / "plain_meff.pdf").exists()


def test_benchmark_cli_skip_tests_bypasses_pytest_gate(
    monkeypatch, tmp_path: Path
) -> None:
    from tasks.pion_2pt.benchmark import run as benchmark_run

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("pytest validation gate should be skipped")

    def fake_benchmark_submission(*_args, artifact_dir: Path, **_kwargs):
        plot_path = artifact_dir / "plain_meff.pdf"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path.write_bytes(b"%PDF-1.4\n% fake meff plot\n")
        return {
            "score": 1.0,
            "artifacts": {"effective_mass_plot": str(plot_path)},
        }

    monkeypatch.setattr(benchmark_run, "_default_dataset_path", lambda: Path("."))
    monkeypatch.setattr(
        benchmark_run, "load_submission", lambda _submission_name: PlainBoostedPion()
    )
    monkeypatch.setattr(
        benchmark_run, "benchmark_submission", fake_benchmark_submission
    )
    monkeypatch.setattr(benchmark_run.subprocess, "run", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--submission",
            "plain",
            "--dataset-path",
            ".",
            "--skip-tests",
            "--output-dir",
            str(tmp_path),
        ],
    )

    benchmark_run.main()
    assert (tmp_path / "plain.json").exists()
    assert (tmp_path / "plain_meff.pdf").exists()


def test_task_benchmark_uses_quenched_default_dataset_path(monkeypatch) -> None:
    from tasks.pion_2pt import task as task_module

    captured: dict[str, object] = {}

    def fake_benchmark_submission(
        operator: object, dataset_path: str, *, artifact_dir: Path, **_kwargs
    ) -> dict[str, float]:
        captured["operator"] = operator
        captured["dataset_path"] = dataset_path
        captured["artifact_dir"] = artifact_dir
        return {"score": 1.0}

    task = task_module.Pion2PtTask()
    monkeypatch.setattr(task_module, "benchmark_submission", fake_benchmark_submission)
    monkeypatch.setattr(task, "validate", lambda _submission: True)

    result = task.benchmark(PlainBoostedPion())

    assert result.score == 1.0
    assert captured["dataset_path"] == str(
        task.dataset_path / "quenched_wilson_b6_8x32"
    )
    assert captured["artifact_dir"] == task.root / "benchmark" / "results"
