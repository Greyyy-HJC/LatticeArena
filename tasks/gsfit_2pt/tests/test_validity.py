"""Validation and benchmark tests for gsfit_2pt."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from tasks.gsfit_2pt.benchmark.core import (
    benchmark_config,
    benchmark_submission,
    load_synthetic_cases,
    make_synthetic_cases,
    save_synthetic_cases,
)
from tasks.gsfit_2pt.interface import (
    FitSubmissionMeta,
    GroundStateFitConfig,
    Pion2PtGroundStateFit,
    validate_config,
)
from tasks.gsfit_2pt.operators.plain import PlainGroundStateFit
from tasks.gsfit_2pt.operators.nn import NNTunedGroundStateFit


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_plain_submission_is_valid() -> None:
    submission = PlainGroundStateFit()
    validate_config(submission.config)
    assert submission.meta.name == "plain"
    assert submission.config.n_states == 2


def test_nn_submission_is_valid() -> None:
    submission = NNTunedGroundStateFit()
    validate_config(submission.config)
    assert submission.meta.name == "nn"


def test_invalid_prior_lengths_are_rejected() -> None:
    config = GroundStateFitConfig(
        t_min=4,
        t_max=12,
        n_states=2,
        e0_prior=(0.3, 0.1),
        delta_e_priors=[],
        amplitude_priors=[(0.8, 0.2), (0.2, 0.1)],
    )

    try:
        validate_config(config)
    except ValueError as exc:
        assert "delta_e_priors" in str(exc)
        return
    raise AssertionError("Expected invalid delta_e_priors length to raise ValueError.")


def test_non_positive_prior_width_is_rejected() -> None:
    config = GroundStateFitConfig(
        t_min=4,
        t_max=12,
        n_states=1,
        e0_prior=(0.3, 0.0),
        delta_e_priors=[],
        amplitude_priors=[(0.8, 0.2)],
    )

    try:
        validate_config(config)
    except ValueError as exc:
        assert "width" in str(exc)
        return
    raise AssertionError("Expected non-positive prior width to raise ValueError.")


def test_invalid_fit_window_is_rejected() -> None:
    config = GroundStateFitConfig(
        t_min=16,
        t_max=16,
        n_states=1,
        e0_prior=(0.3, 0.1),
        delta_e_priors=[],
        amplitude_priors=[(0.8, 0.2)],
    )

    try:
        validate_config(config)
    except ValueError as exc:
        assert "t_min < t_max" in str(exc)
        return
    raise AssertionError("Expected invalid fit window to raise ValueError.")


def test_benchmark_smoke_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tasks/gsfit_2pt/benchmark/run.py",
            "--operator",
            "plain",
            "--num-samples",
            "8",
            "--max-resamples",
            "4",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"operator": "plain"' in result.stdout
    assert '"score":' in result.stdout


def test_gsfit_script_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tasks/gsfit_2pt/scripts/gsfit.py",
            "--operator",
            "plain",
            "--dataset-file",
            "tasks/gsfit_2pt/dataset/fake_data.npz",
            "--case",
            "boosted_clean",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"operator": "plain"' in result.stdout
    assert '"case": "boosted_clean"' in result.stdout
    assert '"energies":' in result.stdout


def test_nn_optimizer_cli(tmp_path: Path) -> None:
    config_path = tmp_path / "nn_config.json"
    result = subprocess.run(
        [
            sys.executable,
            "tasks/gsfit_2pt/scripts/optimize_nn.py",
            "--train-evals",
            "8",
            "--proposal-samples",
            "32",
            "--top-k",
            "6",
            "--max-resamples",
            "4",
            "--output-config",
            str(config_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert config_path.exists()
    assert '"best_score":' in result.stdout


def test_fake_data_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "fake_data.npz"
    original_cases = make_synthetic_cases(num_samples=10, noise_multiplier=0.8)
    saved_path = save_synthetic_cases(original_cases, output)
    loaded_cases = load_synthetic_cases(saved_path)

    assert len(loaded_cases) == len(original_cases)
    assert loaded_cases[0].name == original_cases[0].name
    assert loaded_cases[0].description == original_cases[0].description
    assert loaded_cases[0].lt == original_cases[0].lt
    assert np.allclose(loaded_cases[0].samples, original_cases[0].samples)
    assert loaded_cases[0].true_energies == original_cases[0].true_energies
    assert loaded_cases[0].amplitudes == original_cases[0].amplitudes


def test_plain_recovers_e0_on_low_noise_data() -> None:
    submission = PlainGroundStateFit()
    summary = benchmark_submission(
        submission,
        cases=make_synthetic_cases(num_samples=18, noise_multiplier=0.35),
        max_resample_fits=8,
    )

    clean_case = next(case for case in summary["cases"] if case["case_name"] == "boosted_clean")
    assert clean_case["fit_success"]
    assert clean_case["relative_bias"] < 0.08


class UnderfitOneState(Pion2PtGroundStateFit):
    """Intentionally underfit one-state submission for regression testing."""

    @property
    def meta(self) -> FitSubmissionMeta:
        return FitSubmissionMeta(
            name="underfit_one_state",
            description="Intentionally underfit 1-state submission for tests",
            authors=["LatticeArena"],
        )

    @property
    def config(self) -> GroundStateFitConfig:
        return GroundStateFitConfig(
            t_min=4,
            t_max=16,
            n_states=1,
            e0_prior=(0.34, 0.18),
            delta_e_priors=[],
            amplitude_priors=[(0.90, 0.50)],
        )


def test_underfit_single_state_scores_worse_than_baseline() -> None:
    cases = make_synthetic_cases(num_samples=18, noise_multiplier=1.0)
    baseline_score = benchmark_submission(PlainGroundStateFit(), cases=cases, max_resample_fits=8)["score"]
    underfit_score = benchmark_submission(UnderfitOneState(), cases=cases, max_resample_fits=8)["score"]

    assert baseline_score > underfit_score


def test_nn_submission_matches_or_beats_plain() -> None:
    cases = make_synthetic_cases(num_samples=24, noise_multiplier=1.0)
    baseline_score = benchmark_config(PlainGroundStateFit().config, cases=cases, max_resample_fits=8)["score"]
    nn_score = benchmark_config(NNTunedGroundStateFit().config, cases=cases, max_resample_fits=8)["score"]

    assert nn_score >= baseline_score


def test_build_leaderboard_page_cli(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "tasks/gsfit_2pt/benchmark/run.py",
            "--operator",
            "plain",
            "--dataset-file",
            "tasks/gsfit_2pt/dataset/fake_data.npz",
            "--max-resamples",
            "4",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    output_html = tmp_path / "leaderboard.html"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_leaderboard_page.py",
            "--output",
            str(output_html),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    html = output_html.read_text()
    assert "LatticeArena Leaderboard" in html
    assert "gsfit_2pt" in html
