"""Example ground-state fit script for gsfit_2pt."""

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

from tasks.gsfit_2pt.benchmark.core import load_synthetic_cases
from tasks.gsfit_2pt.interface import GroundStateFitConfig, Pion2PtGroundStateFit, validate_config


def load_submission(operator_name: str) -> Pion2PtGroundStateFit:
    """Import and instantiate a submission from operators/<name>.py."""

    module = importlib.import_module(f"tasks.gsfit_2pt.operators.{operator_name}")
    for value in module.__dict__.values():
        if (
            isinstance(value, type)
            and issubclass(value, Pion2PtGroundStateFit)
            and value is not Pion2PtGroundStateFit
        ):
            return value()
    raise ValueError(f"No Pion2PtGroundStateFit implementation found in operator '{operator_name}'.")


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


def fit_case(
    samples: np.ndarray,
    config: GroundStateFitConfig,
    lt: int,
    *,
    normalize: bool = True,
    label: str | None = None,
) -> dict[str, Any]:
    """Run a correlated `gvar`/`lsqfit` fit on one correlator ensemble."""

    validate_config(config, lt)
    pt2_gv = samples_to_gvar(samples)
    normalization_factor = abs(gv.mean(pt2_gv[0]))
    if normalization_factor == 0:
        normalization_factor = 1.0

    fit_data = pt2_gv / normalization_factor if normalize else pt2_gv
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

    energies = [fit_res.p["E0"]]
    for idx in range(1, config.n_states):
        energies.append(energies[-1] + gv.exp(fit_res.p[f"log(dE{idx})"]))

    amplitudes = []
    amp_scale = normalization_factor if normalize else 1.0
    for idx in range(config.n_states):
        amplitudes.append(fit_res.p[f"A{idx}"] * amp_scale)

    meff = effective_mass(pt2_gv)
    fit_curve = pt2_multi_state_fcn(np.arange(lt), fit_res.p, lt=lt, n_states=config.n_states)
    if normalize:
        fit_curve = fit_curve * normalization_factor
    fit_meff = effective_mass(fit_curve)

    summary = {
        "label": label,
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
        "meff_preview": {
            "t": list(range(min(8, len(meff)))),
            "data_mean": [float(x) for x in gv.mean(meff[:8])],
            "data_sdev": [float(x) for x in gv.sdev(meff[:8])],
            "fit_mean": [float(x) for x in gv.mean(fit_meff[:8])],
            "fit_sdev": [float(x) for x in gv.sdev(fit_meff[:8])],
        },
        "format": fit_res.format(maxline=True),
    }
    return summary


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
    parser = argparse.ArgumentParser(description="Run an example 2pt gsfit on fake gsfit_2pt data")
    parser.add_argument("--operator", type=str, required=True, help="Operator module name under operators/")
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

    submission = load_submission(args.operator)
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
    summary["operator"] = submission.meta.name
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
