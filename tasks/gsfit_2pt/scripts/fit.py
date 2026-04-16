"""Ground-state fitting pipeline for gsfit_2pt.

This module is the single source of truth for the fitting logic used by both
the interactive analysis workflow and the benchmark scoring pipeline.

Key functions:
- ``fit_correlator``: Low-level fit of a single correlator with given covariance.
- ``fit_case``: High-level convenience wrapper that computes the mean and
  covariance from an ensemble of samples before fitting.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import gvar as gv
import lsqfit as lsf
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.gsfit_2pt.dataset.synthetic import load_synthetic_cases
from tasks.gsfit_2pt.interface import GroundStateFitConfig, Pion2PtGroundStateFit, validate_config


def load_submission(submission_name: str) -> Pion2PtGroundStateFit:
    """Import and instantiate a submission from submissions/<name>.py."""

    module = importlib.import_module(f"tasks.gsfit_2pt.submissions.{submission_name}")
    for value in module.__dict__.values():
        if (
            isinstance(value, type)
            and issubclass(value, Pion2PtGroundStateFit)
            and value is not Pion2PtGroundStateFit
        ):
            return value()
    raise ValueError(f"No Pion2PtGroundStateFit submission found in module '{submission_name}'.")


def samples_to_gvar(samples: np.ndarray) -> gv.GVar:
    """Convert bootstrap/jackknife-like samples into correlated gvar data."""

    mean = np.mean(samples, axis=0)
    covariance = np.cov(samples, rowvar=False, ddof=1)
    return gv.gvar(mean, covariance)


def effective_mass(correlator: gv.GVar) -> gv.GVar:
    """Simple log-ratio effective mass."""

    return gv.log(correlator[:-1] / correlator[1:])


def build_prior(config: GroundStateFitConfig, normalize_factor: float = 1.0) -> gv.BufferDict:
    """Build lsqfit priors from the task configuration."""

    validate_config(config)
    priors = gv.BufferDict()
    priors["E0"] = gv.gvar(config.e0_prior[0], config.e0_prior[1])

    for idx, prior in enumerate(config.delta_e_priors, start=1):
        delta_mean = max(prior[0], 1e-8)
        delta_width = max(prior[1], 1e-8)
        log_mean = float(np.log(delta_mean))
        log_width = float(max(delta_width / delta_mean, 1e-6))
        priors[f"log(dE{idx})"] = gv.gvar(log_mean, log_width)

    for idx, prior in enumerate(config.amplitude_priors):
        priors[f"A{idx}"] = gv.gvar(prior[0] / normalize_factor, prior[1] / normalize_factor)

    return priors


def pt2_multi_state_fcn(times: np.ndarray, params: gv.BufferDict, lt: int, n_states: int) -> gv.GVar:
    """Periodic multi-state 2pt function in a notebook-friendly lsqfit form."""

    values = 0
    energy = params["E0"]
    for idx in range(n_states):
        if idx > 0:
            energy = energy + gv.exp(params[f"log(dE{idx})"])
        amplitude = params[f"A{idx}"]
        values = values + amplitude * (gv.exp(-energy * times) + gv.exp(-energy * (lt - times)))
    return values


def _do_fit(
    correlator_gv: gv.GVar,
    config: GroundStateFitConfig,
    lt: int,
    *,
    normalize: bool = True,
) -> tuple[lsf.nonlinear_fit, float]:
    """Internal: run lsqfit and return the raw fit result + normalization factor."""
    normalization_factor = abs(gv.mean(correlator_gv[0]))
    if normalization_factor == 0:
        normalization_factor = 1.0

    fit_data = correlator_gv / normalization_factor if normalize else correlator_gv
    priors = build_prior(config, normalize_factor=normalization_factor if normalize else 1.0)
    t_range = np.arange(config.t_min, config.t_max + 1)

    def fcn(t: np.ndarray, p: gv.BufferDict) -> gv.GVar:
        return pt2_multi_state_fcn(t, p, lt=lt, n_states=config.n_states)

    fit_res = lsf.nonlinear_fit(
        data=(t_range, fit_data[config.t_min : config.t_max + 1]),
        prior=priors,
        fcn=fcn,
        maxit=10000,
    )
    return fit_res, normalization_factor


def _extract_results(
    fit_res: lsf.nonlinear_fit,
    config: GroundStateFitConfig,
    normalization_factor: float,
) -> dict[str, Any]:
    """Extract structured results from an lsqfit result object."""
    energies = [fit_res.p["E0"]]
    for idx in range(1, config.n_states):
        energies.append(energies[-1] + gv.exp(fit_res.p[f"log(dE{idx})"]))

    amplitudes = []
    for idx in range(config.n_states):
        amplitudes.append(fit_res.p[f"A{idx}"] * normalization_factor)

    return {
        "fit_success": bool(fit_res.Q > 0),
        "normalization_factor": float(normalization_factor),
        "chi2": float(fit_res.chi2),
        "dof": int(fit_res.dof),
        "chi2_dof": float(fit_res.chi2 / fit_res.dof),
        "Q": float(fit_res.Q),
        "logGBF": float(fit_res.logGBF),
        "energies": [
            {"mean": float(gv.mean(energy)), "sdev": float(gv.sdev(energy))}
            for energy in energies
        ],
        "amplitudes": [
            {"mean": float(gv.mean(amplitude)), "sdev": float(gv.sdev(amplitude))}
            for amplitude in amplitudes
        ],
        "format": fit_res.format(maxline=True),
    }


def fit_correlator(
    correlator: np.ndarray,
    covariance: np.ndarray,
    config: GroundStateFitConfig,
    lt: int,
    *,
    normalize: bool = True,
) -> dict[str, Any]:
    """Fit a single correlator with a given covariance matrix.

    This is the low-level entry point used by the benchmark to fit
    individual resamples against a pre-computed covariance.

    Args:
        correlator: 1D array of correlator values, shape ``(Lt,)``.
        covariance: Covariance matrix, shape ``(Lt, Lt)``.
        config: Fit configuration.
        lt: Temporal extent.
        normalize: Whether to normalize before fitting.

    Returns:
        Dict with keys: fit_success, chi2, dof, chi2_dof, Q, logGBF,
        energies, amplitudes, normalization_factor, format.
    """
    validate_config(config, lt)
    correlator_gv = gv.gvar(correlator, covariance)
    fit_res, norm = _do_fit(correlator_gv, config, lt, normalize=normalize)
    return _extract_results(fit_res, config, norm)


def fit_case(
    samples: np.ndarray,
    config: GroundStateFitConfig,
    lt: int,
    *,
    normalize: bool = True,
    label: str | None = None,
) -> dict[str, Any]:
    """Run a correlated ``gvar``/``lsqfit`` fit on a correlator ensemble.

    Computes the mean and covariance from the sample array, then fits.
    This is the high-level entry point for interactive analysis.
    """
    validate_config(config, lt)
    pt2_gv = samples_to_gvar(samples)
    fit_res, norm = _do_fit(pt2_gv, config, lt, normalize=normalize)
    result = _extract_results(fit_res, config, norm)

    # Add meff preview for interactive use / plotting
    meff = effective_mass(pt2_gv)
    fit_curve = pt2_multi_state_fcn(
        np.arange(lt), fit_res.p, lt=lt, n_states=config.n_states,
    )
    if normalize:
        fit_curve = fit_curve * norm
    fit_meff = effective_mass(fit_curve)

    result["label"] = label
    result["meff_preview"] = {
        "t": list(range(min(8, len(meff)))),
        "data_mean": [float(x) for x in gv.mean(meff[:8])],
        "data_sdev": [float(x) for x in gv.sdev(meff[:8])],
        "fit_mean": [float(x) for x in gv.mean(fit_meff[:8])],
        "fit_sdev": [float(x) for x in gv.sdev(fit_meff[:8])],
    }

    return result


def maybe_make_plot(
    samples: np.ndarray,
    config: GroundStateFitConfig,
    lt: int,
    output_path: Path,
    *,
    normalize: bool = True,
) -> Path:
    """Save a simple meff plot for one case."""

    import matplotlib.pyplot as plt

    pt2_gv = samples_to_gvar(samples)
    fit_summary = fit_case(samples, config, lt, normalize=normalize)
    meff = effective_mass(pt2_gv)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    t_vals = np.arange(len(meff))
    ax.errorbar(t_vals, gv.mean(meff), yerr=gv.sdev(meff), fmt="o", ms=4, capsize=3, label="data")

    fit_t = np.asarray(fit_summary["meff_preview"]["t"], dtype=int)
    fit_mean = np.asarray(fit_summary["meff_preview"]["fit_mean"], dtype=float)
    fit_sdev = np.asarray(fit_summary["meff_preview"]["fit_sdev"], dtype=float)
    ax.fill_between(fit_t, fit_mean - fit_sdev, fit_mean + fit_sdev, alpha=0.3, label="fit")

    ax.set_xlabel("t")
    ax.set_ylabel("m_eff")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed gsfit_2pt analysis pipeline on synthetic data")
    parser.add_argument("--submission", type=str, required=True, help="Submission module name under submissions/")
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=Path("tasks/gsfit_2pt/dataset/fake_data.npz"),
        help="Saved synthetic dataset archive",
    )
    parser.add_argument("--case", type=str, default="boosted_clean", help="Case name inside the dataset archive")
    parser.add_argument("--no-normalize", action="store_true", help="Disable the notebook-style 2pt normalization")
    parser.add_argument("--plot-output", type=Path, default=None, help="Optional path to save a meff plot")
    args = parser.parse_args()

    submission = load_submission(args.submission)
    cases = {case.name: case for case in load_synthetic_cases(args.dataset_file)}
    if args.case not in cases:
        available = ", ".join(sorted(cases))
        raise KeyError(f"Unknown case '{args.case}'. Available cases: {available}")

    case = cases[args.case]
    summary = fit_case(
        case.samples,
        submission.config,
        case.lt,
        normalize=not args.no_normalize,
        label=args.case,
    )
    summary["submission"] = submission.meta.name
    summary["case"] = args.case
    summary["true_e0"] = case.true_e0
    summary["dataset_file"] = str(args.dataset_file)

    if args.plot_output is not None:
        plot_path = maybe_make_plot(
            case.samples,
            submission.config,
            case.lt,
            output_path=args.plot_output,
            normalize=not args.no_normalize,
        )
        summary["plot_output"] = str(plot_path)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
