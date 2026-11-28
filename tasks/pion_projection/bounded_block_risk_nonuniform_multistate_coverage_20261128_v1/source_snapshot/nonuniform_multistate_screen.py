"""Joint anchor/setwise coverage screen for a nonuniform multistate chain.

The score chain is a reversible four-state refresh chain with known stationary
law, exact pseudo-gap, and an exact post-burn-in TV bound. This is synthetic
method evidence only; it does not estimate physical HMC dependence.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import beta

from tasks.pion_projection.bounded_block_risk_independent_anchor_coverage_screen import _wilson
from tasks.pion_projection.bounded_block_risk_markov_varying import (
    time_varying_markov_radius_from_gap_bound,
)
from tasks.pion_projection.bounded_block_risk_setwise_markov import (
    _target_grid,
    setwise_markov_score_lcb_from_gap_bound,
)


DEFAULT_OUT = TASK_DIR / "bounded_block_risk_nonuniform_multistate_coverage_20261128_v1"
TARGET_PROBABILITY = 0.4
ANCHOR_SAMPLE_SIZE = 256
STATIONARY_LAW = (Fraction(1, 20), Fraction(3, 20), Fraction(3, 10), Fraction(1, 2))
REFERENCE_BY_STATE = (0.0, 0.25, 0.75, 1.0)
CANDIDATE_PREDICTION = 0.5
REFRESH_EIGENVALUE = Fraction(7, 10)
PSEUDO_GAP = Fraction(51, 100)
BURN_IN = 24
PLANNED_BLOCKS = 16_384
REPLICATES = 1_000
ALPHA_ANCHOR = 0.025
ALPHA_SCORE = 0.025
ALPHA_TOTAL = 0.05
PREDICTION_DOMAIN = (0.0, 1.0)
SCORE_SCALE_R = 1.0
MESH_STEP_R = 0.02
ANCHOR_SEED = 202611282901
SCORE_SEED = 202611292901
GAMMA_PS_LOWER = math.nextafter(float(PSEUDO_GAP), 0.0)
TV_EXACT = (1 - STATIONARY_LAW[0]) * REFRESH_EIGENVALUE**BURN_IN
TV_UPPER = math.nextafter(float(TV_EXACT), math.inf)


def _cp_interval(successes: int, trials: int, alpha: float) -> tuple[float, float]:
    if type(successes) is not int or type(trials) is not int:
        raise ValueError("successes and trials must be integers")
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("require trials >= 1 and 0 <= successes <= trials")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    lower = 0.0 if successes == 0 else float(
        beta.ppf(alpha / 2.0, successes, trials - successes + 1)
    )
    upper = 1.0 if successes == trials else float(
        beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes)
    )
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError("failed to construct a finite Clopper-Pearson interval")
    return lower, upper


def _transition_matrix() -> np.ndarray:
    """P = lambda I + (1-lambda) 1 pi: reversible with eigenvalues 1, lambda x3."""
    pi = np.asarray([float(value) for value in STATIONARY_LAW], dtype=float)
    transition = (float(REFRESH_EIGENVALUE) * np.eye(len(pi))
                  + (1.0 - float(REFRESH_EIGENVALUE)) * np.tile(pi, (len(pi), 1)))
    return transition


def _model_diagnostics() -> dict[str, Any]:
    transition = _transition_matrix()
    pi = np.asarray([float(value) for value in STATIONARY_LAW], dtype=float)
    detailed_balance = pi[:, None] * transition
    eigenvalues = np.linalg.eigvals(transition)
    expected_eigenvalues = np.asarray([float(REFRESH_EIGENVALUE)] * 3 + [1.0])
    eigenvalue_error = float(np.max(np.abs(np.sort(eigenvalues.real) - expected_eigenvalues)))
    stationary_error = float(np.max(np.abs(pi @ transition - pi)))
    detailed_balance_error = float(np.max(np.abs(detailed_balance - detailed_balance.T)))
    if (not np.allclose(transition.sum(axis=1), 1.0, rtol=0.0, atol=2e-15)
            or stationary_error > 2e-15 or detailed_balance_error > 2e-15
            or eigenvalue_error > 2e-14):
        raise AssertionError("finite-state transition matrix failed its analytic model checks")
    return {
        "transition_matrix": transition.tolist(),
        "stationary_law": pi.tolist(),
        "stationarity_max_abs_error": stationary_error,
        "detailed_balance_max_abs_error": detailed_balance_error,
        "eigenvalues_sorted": np.sort(eigenvalues.real).tolist(),
        "eigenvalue_max_abs_error": eigenvalue_error,
    }


def _draw_anchor_samples(*, seed: int, replicates: int = REPLICATES,
                         trials: int = ANCHOR_SAMPLE_SIZE) -> np.ndarray:
    if type(replicates) is not int or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if type(trials) is not int or trials < 1:
        raise ValueError("trials must be a positive integer")
    return np.random.default_rng(seed).random((replicates, trials)) < TARGET_PROBABILITY


def _draw_score_chain(*, seed: int, replicates: int = REPLICATES,
                      blocks: int = PLANNED_BLOCKS,
                      burn_in: int = BURN_IN) -> tuple[np.ndarray, np.ndarray]:
    if type(replicates) is not int or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if type(blocks) is not int or blocks < 2:
        raise ValueError("blocks must be an integer >= 2")
    if type(burn_in) is not int or burn_in < 0:
        raise ValueError("burn_in must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    pi = np.asarray([float(value) for value in STATIONARY_LAW], dtype=float)
    eigenvalue = float(REFRESH_EIGENVALUE)
    trajectory = np.empty((replicates, burn_in + blocks), dtype=np.uint8)
    trajectory[:, 0] = 0
    current = np.zeros(replicates, dtype=np.uint8)
    for time in range(1, burn_in + blocks):
        refresh = rng.random(replicates) < (1.0 - eigenvalue)
        count = int(np.count_nonzero(refresh))
        if count:
            current[refresh] = rng.choice(len(pi), size=count, p=pi).astype(np.uint8)
        trajectory[:, time] = current
    return trajectory, trajectory[:, burn_in:].copy()


def _score_values(target: float) -> np.ndarray:
    references = np.asarray(REFERENCE_BY_STATE, dtype=float)
    scores = ((target - references) ** 2
              - (target - CANDIDATE_PREDICTION) ** 2) / SCORE_SCALE_R**2
    return np.clip(scores, -1.0, 1.0)


def _expected_score(target: float) -> float:
    pi = np.asarray([float(value) for value in STATIONARY_LAW], dtype=float)
    return float(pi @ _score_values(target))


def _true_interval_minimum(lower: float, upper: float) -> float:
    # The clipping is inactive for target, reference and candidate in [0, 1],
    # so the expected squared-risk difference is affine in target.
    slope = 2.0 * (
        CANDIDATE_PREDICTION
        - sum(float(pi) * ref for pi, ref in zip(STATIONARY_LAW, REFERENCE_BY_STATE))
    ) / SCORE_SCALE_R**2
    endpoint = upper if slope < 0.0 else lower
    return _expected_score(endpoint)


def _uniform_lcb_from_state_counts(
    state_counts: np.ndarray,
    intervals: np.ndarray,
    *,
    blocks: int = PLANNED_BLOCKS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if state_counts.ndim != 2 or state_counts.shape[1] != len(STATIONARY_LAW):
        raise ValueError("state_counts must have shape (replicates, four states)")
    if intervals.shape != (state_counts.shape[0], 2):
        raise ValueError("state counts and anchor intervals have incompatible shapes")
    fractions = state_counts / blocks
    lower_bounds = np.empty(state_counts.shape[0], dtype=float)
    true_minima = np.empty(state_counts.shape[0], dtype=float)
    point_counts = np.empty(state_counts.shape[0], dtype=np.int64)
    radius_cache: dict[int, float] = {}
    for index, (lower, upper) in enumerate(intervals):
        grid, mesh_penalty, _ = _target_grid(
            float(lower), float(upper), score_scale_R=SCORE_SCALE_R,
            mesh_step_R=MESH_STEP_R, max_grid_points=4096,
        )
        family_size = len(grid)
        point_counts[index] = family_size
        if family_size not in radius_cache:
            alpha_effective = ALPHA_SCORE - family_size * TV_UPPER
            radius_cache[family_size] = time_varying_markov_radius_from_gap_bound(
                blocks=blocks, gamma_ps_lower=GAMMA_PS_LOWER,
                family=family_size, alpha_stationary=alpha_effective,
            )["radius"]
        grid_means = np.asarray([_score_values(target) for target in grid]) @ fractions[index]
        raw_lcb = (
            float(np.min(grid_means)) - radius_cache[family_size]
            - 2.0 * TV_UPPER - mesh_penalty
        )
        lower_bounds[index] = max(-1.0, math.nextafter(raw_lcb, -math.inf))
        true_minima[index] = _true_interval_minimum(float(lower), float(upper))
    return lower_bounds, true_minima, point_counts


def _validate_against_production_helper(states: np.ndarray, intervals: np.ndarray,
                                        expected_lcbs: np.ndarray) -> dict[str, Any]:
    checked = min(8, states.shape[0])
    hashes = {
        "gamma_ps_lower": hashlib.sha256(b"known-reversible-refresh-chain-gap-v1").hexdigest(),
        "dependence": hashlib.sha256(b"four-state-refresh-transition-and-stationary-law-v1").hexdigest(),
        "block_chain": hashlib.sha256(b"refresh-chain-block-state-schedule-v1").hexdigest(),
        "initial_law": hashlib.sha256(b"state-zero-then-exact-24-step-refresh-tv-v1").hexdigest(),
        "anchor_shift": hashlib.sha256(b"synthetic-zero-anchor-bridge-shift-v1").hexdigest(),
        "score_schedule": hashlib.sha256(
            json.dumps({"reference_by_state": REFERENCE_BY_STATE,
                        "candidate": CANDIDATE_PREDICTION,
                        "score_scale_R": SCORE_SCALE_R}, sort_keys=True).encode()
        ).hexdigest(),
        "target_set": hashlib.sha256(b"independent-iid-bernoulli-clopper-pearson-target-set-v1").hexdigest(),
        "prediction_domain": hashlib.sha256(b"synthetic-unit-interval-domain-v1").hexdigest(),
    }
    errors = []
    for index in range(checked):
        reference = np.asarray(REFERENCE_BY_STATE, dtype=float)[states[index]]
        helper = setwise_markov_score_lcb_from_gap_bound(
            reference, np.full(reference.size, CANDIDATE_PREDICTION),
            valid_blocks=np.ones(reference.size, dtype=bool),
            target_interval=intervals[index].tolist(),
            prediction_domain=list(PREDICTION_DOMAIN), score_scale_R=SCORE_SCALE_R,
            mesh_step_R=MESH_STEP_R, gamma_ps_lower=GAMMA_PS_LOWER, family=1,
            alpha_score=ALPHA_SCORE, alpha_anchor=ALPHA_ANCHOR,
            alpha_total=ALPHA_TOTAL, initial_law_mode="post_burn_in_tv",
            burn_in_blocks=BURN_IN, initial_law_tv_bound=TV_UPPER,
            anchor_score_shift_bound=0.0,
            gamma_ps_lower_receipt_sha256=hashes["gamma_ps_lower"],
            dependence_evidence_receipt_sha256=hashes["dependence"],
            block_chain_receipt_sha256=hashes["block_chain"],
            initial_law_receipt_sha256=hashes["initial_law"],
            anchor_shift_receipt_sha256=hashes["anchor_shift"],
            score_schedule_receipt_sha256=hashes["score_schedule"],
            target_set_coverage_receipt_sha256=hashes["target_set"],
            prediction_domain_receipt_sha256=hashes["prediction_domain"],
        )
        error = abs(float(expected_lcbs[index])
                    - float(helper["uniform_target_set_lower_bound"]))
        errors.append(error)
    return {"planned": checked, "valid": checked,
            "max_abs_difference": max(errors, default=0.0),
            "tolerance": 2e-14,
            "all_match": all(error <= 2e-14 for error in errors),
            "receipt_hashes_are_synthetic_identifiers": True}


def _cell(*, replicates: int = REPLICATES, blocks: int = PLANNED_BLOCKS,
          anchor_seed: int = ANCHOR_SEED,
          score_seed: int = SCORE_SEED) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    anchors = _draw_anchor_samples(seed=anchor_seed, replicates=replicates)
    trajectory, scored_states = _draw_score_chain(
        seed=score_seed, replicates=replicates, blocks=blocks,
    )
    anchor_counts = anchors.sum(axis=1, dtype=np.int64)
    intervals = np.asarray([
        _cp_interval(int(count), ANCHOR_SAMPLE_SIZE, ALPHA_ANCHOR)
        for count in anchor_counts
    ], dtype=float)
    state_counts = np.stack([
        np.count_nonzero(scored_states == state, axis=1) for state in range(4)
    ], axis=1).astype(np.int64)
    lcbs, true_minima, grid_points = _uniform_lcb_from_state_counts(
        state_counts, intervals, blocks=blocks,
    )
    anchor_covered = ((intervals[:, 0] <= TARGET_PROBABILITY)
                      & (TARGET_PROBABILITY <= intervals[:, 1]))
    score_covered = lcbs <= true_minima
    joint_covered = anchor_covered & score_covered
    widths = intervals[:, 1] - intervals[:, 0]
    slacks = true_minima - lcbs
    helper_check = _validate_against_production_helper(scored_states, intervals, lcbs)
    if not helper_check["all_match"]:
        raise AssertionError("vectorized multistate score LCB differs from production helper")

    rows = [{
        "replicate": index,
        "anchor_seed": anchor_seed,
        "score_chain_seed": score_seed,
        "planned_anchor_observations": ANCHOR_SAMPLE_SIZE,
        "valid_anchor_observations": ANCHOR_SAMPLE_SIZE,
        "failed_anchor_observations": 0,
        "anchor_successes": int(anchor_counts[index]),
        "target_interval_lower": float(intervals[index, 0]),
        "target_interval_upper": float(intervals[index, 1]),
        "target_interval_width": float(widths[index]),
        "planned_score_blocks": blocks,
        "valid_score_blocks": blocks,
        "failed_score_blocks": 0,
        "state_counts": ";".join(str(int(value)) for value in state_counts[index]),
        "target_grid_points": int(grid_points[index]),
        "uniform_lcb": float(lcbs[index]),
        "exact_interval_minimum_score": float(true_minima[index]),
        "lcb_slack": float(slacks[index]),
        "anchor_covered": bool(anchor_covered[index]),
        "score_uniformly_covered": bool(score_covered[index]),
        "jointly_covered": bool(joint_covered[index]),
        "positive_lcb": bool(lcbs[index] > 0.0),
    } for index in range(replicates)]

    def stats(mask: np.ndarray) -> dict[str, Any]:
        successes = int(mask.sum())
        low, high = _wilson(successes, replicates)
        return {"successes": successes, "trials": replicates,
                "rate": successes / replicates,
                "wilson95_low": low, "wilson95_high": high}

    pi = np.asarray([float(value) for value in STATIONARY_LAW])
    observed_occupancy = state_counts.sum(axis=0) / (replicates * blocks)
    summary = {
        "planned_replicates": replicates,
        "valid_replicates": replicates,
        "failed_replicates": 0,
        "abstained_replicates": 0,
        "planned_anchor_observations": replicates * ANCHOR_SAMPLE_SIZE,
        "valid_anchor_observations": replicates * ANCHOR_SAMPLE_SIZE,
        "failed_anchor_observations": 0,
        "planned_burn_in_transitions": replicates * BURN_IN,
        "planned_score_blocks": replicates * blocks,
        "valid_score_blocks": replicates * blocks,
        "failed_score_blocks": 0,
        "anchor_coverage": stats(anchor_covered),
        "score_uniform_coverage": stats(score_covered),
        "joint_coverage": stats(joint_covered),
        "nominal_anchor_coverage": 1.0 - ALPHA_ANCHOR,
        "nominal_score_coverage": 1.0 - ALPHA_SCORE,
        "union_bound_joint_coverage_lower": 1.0 - ALPHA_TOTAL,
        "positive_lcb_replicates": int(np.count_nonzero(lcbs > 0.0)),
        "positive_lcb_rate": float(np.mean(lcbs > 0.0)),
        "median_target_interval_width": float(np.median(widths)),
        "median_uniform_lcb": float(np.median(lcbs)),
        "median_exact_interval_minimum_score": float(np.median(true_minima)),
        "median_lcb_slack": float(np.median(slacks)),
        "median_grid_points": float(np.median(grid_points)),
        "stationary_law": pi.tolist(),
        "empirical_score_chain_occupancy": observed_occupancy.tolist(),
        "reference_by_state": list(REFERENCE_BY_STATE),
        "candidate_prediction": CANDIDATE_PREDICTION,
        "pseudo_spectral_gap_exact": str(PSEUDO_GAP),
        "pseudo_spectral_gap_lower": GAMMA_PS_LOWER,
        "refresh_eigenvalue_exact": str(REFRESH_EIGENVALUE),
        "burn_in_transitions": BURN_IN,
        "initial_law_tv_exact": str(TV_EXACT),
        "initial_law_tv_upper": TV_UPPER,
        "helper_equivalence": helper_check,
        "anchor_seed": anchor_seed,
        "score_chain_seed": score_seed,
    }
    raw = {
        "anchor_samples": anchors,
        "score_chain_trajectory_including_burnin": trajectory,
        "score_chain_scored_states": scored_states,
        "anchor_counts": anchor_counts,
        "target_intervals": intervals,
        "state_counts": state_counts,
        "uniform_lcbs": lcbs,
        "exact_interval_minima": true_minima,
    }
    return raw, rows, summary


def _plot(rows: list[dict], summary: dict, path: Path) -> None:
    events = ["anchor_covered", "score_uniformly_covered", "jointly_covered"]
    labels = ["Anchor interval", "Score-set LCB", "Joint event"]
    colors = ["#0072b2", "#009e73", "#cc79a7"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    rates, lower, upper = [], [], []
    for event in events:
        values = np.asarray([bool(row[event]) for row in rows])
        successes = int(values.sum())
        lo, hi = _wilson(successes, len(values))
        rates.append(successes / len(values)); lower.append(lo); upper.append(hi)
    errors = [[rate - lo for rate, lo in zip(rates, lower)],
              [hi - rate for rate, hi in zip(rates, upper)]]
    axes[0].bar(range(3), rates, color=colors, width=0.68)
    axes[0].errorbar(range(3), rates, yerr=errors, color="#222222", fmt="none",
                     capsize=4, linewidth=1)
    axes[0].axhline(0.975, color="#333333", linestyle=":", linewidth=1,
                    label="individual nominal coverage 97.5%")
    axes[0].axhline(0.95, color="#555555", linestyle="-.", linewidth=1,
                    label="joint union-bound floor 95%")
    axes[0].set_xticks(range(3), labels, rotation=10)
    axes[0].set_ylim(0.90, 1.005)
    axes[0].set_ylabel("empirical event rate with Wilson 95% MC interval")
    axes[0].set_title("Coverage events over independent replicates")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(fontsize=7)

    true_minima = np.asarray([row["exact_interval_minimum_score"] for row in rows])
    lcbs = np.asarray([row["uniform_lcb"] for row in rows])
    axes[1].scatter(true_minima, lcbs, s=14, alpha=0.42, color="#6a4c93")
    low = min(float(true_minima.min()), float(lcbs.min()))
    high = max(float(true_minima.max()), float(lcbs.max()))
    axes[1].plot([low, high], [low, high], color="#555555", linestyle="--",
                 label="LCB = exact interval minimum")
    axes[1].axhline(0.0, color="#cc3311", linestyle=":", label="positive LCB threshold")
    axes[1].set_xlabel("exact stationary minimum over anchor interval")
    axes[1].set_ylabel("uniform Markov lower bound")
    axes[1].set_title(f"Sharpness; positive rate {summary['positive_lcb_rate']:.1%}")
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=8)
    fig.suptitle("Nonuniform four-state reversible chain; synthetic only")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def build_report(output: str | Path = DEFAULT_OUT) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    model = _model_diagnostics()
    raw, rows, summary = _cell()
    np.savez_compressed(out / "synthetic_inputs.npz", **raw)
    with (out / "replicate_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    _plot(rows, summary, out / "coverage_and_sharpness.png")

    pi = np.asarray(model["stationary_law"], dtype=float)
    source_map = {
        "nonuniform_multistate_screen.py": Path(__file__),
        "independent_anchor_coverage_screen.py": (
            TASK_DIR / "bounded_block_risk_independent_anchor_coverage_screen.py"
        ),
        "setwise_markov.py": TASK_DIR / "bounded_block_risk_setwise_markov.py",
        "markov_varying.py": TASK_DIR / "bounded_block_risk_markov_varying.py",
        "source_integrity.py": REPO_ROOT / "core/source_integrity.py",
        "numerics.py": REPO_ROOT / "core/numerics.py",
    }
    source_dir = out / "source_snapshot"
    source_dir.mkdir()
    source_hashes = {}
    for name, source in source_map.items():
        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        source_hashes[name] = digest
        (source_dir / name).write_bytes(payload)

    summary_payload = {
        "schema": "pion-nonuniform-multistate-markov-coverage-summary-v1",
        "classification": "SYNTHETIC_NONUNIFORM_MULTISTATE_DEPENDENCE_SCREEN_NOT_CALIBRATION",
        "certified": False,
        "physical_score": None,
        "method_summary": summary,
        "model_checks": model,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "rng": "numpy.random.default_rng (PCG64)",
        },
        "source_hashes": source_hashes,
        "scope_limits": [
            "this known four-state refresh chain is synthetic and does not estimate any HMC pseudo-gap",
            "the exact Clopper-Pearson interval is valid only for the independent Bernoulli anchor draw used here",
            "the exact initial TV and pseudo-gap follow from the declared matrix and fixed initial state",
            "the synthetic anchor target equals the score target by construction; no physical common-bias bridge is tested",
            "no production inputs, sealed data, GPU or QUDA are read or used",
        ],
        "physical_status": "NOT_READY/UNCERTIFIED; physical_score=null; sealed data closed",
    }
    (out / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    protocol = {
        "schema": "pion-nonuniform-multistate-markov-coverage-protocol-v1",
        "classification": summary_payload["classification"],
        "stationary_law_exact": [str(value) for value in STATIONARY_LAW],
        "transition": "P=lambda*I+(1-lambda)*1*pi",
        "refresh_eigenvalue_exact": str(REFRESH_EIGENVALUE),
        "pseudo_spectral_gap_exact": str(PSEUDO_GAP),
        "initial_state": 0,
        "burn_in_transitions": BURN_IN,
        "initial_tv_exact": str(TV_EXACT),
        "initial_tv_upper": TV_UPPER,
        "reference_prediction_by_state": list(REFERENCE_BY_STATE),
        "candidate_prediction": CANDIDATE_PREDICTION,
        "anchor_target_probability": TARGET_PROBABILITY,
        "anchor_sample_size": ANCHOR_SAMPLE_SIZE,
        "anchor_alpha": ALPHA_ANCHOR,
        "score_alpha": ALPHA_SCORE,
        "joint_alpha": ALPHA_TOTAL,
        "score_chain_blocks": PLANNED_BLOCKS,
        "replicates": REPLICATES,
        "score_chain_seed": SCORE_SEED,
        "anchor_seed": ANCHOR_SEED,
        "score_chain_generator": "independent refresh-or-stay transition draws; initial state fixed at 0",
        "rng": "numpy.random.default_rng (PCG64); independent anchor and score seeds",
        "reproduction_command": "python tasks/pion_projection/bounded_block_risk_nonuniform_multistate_coverage_screen.py --output <new-empty-directory>",
        "tv_adjustment": "family*TV removed from score alpha; 2*TV lower-mean shift penalty per grid point",
        "target_mesh_step_R": MESH_STEP_R,
        "helper_equivalence": summary["helper_equivalence"],
        "sealed_data_access": False,
        "gpu_quda": False,
        "physical_score": None,
        "no_clobber": True,
    }
    (out / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    readme = [
        "# Pion bounded risk: nonuniform four-state Markov coverage screen", "",
        "**Classification:** synthetic dependence diagnostic; not a physical calibration or an HMC receipt.", "",
        "Reproduce from the repository root with `python tasks/pion_projection/bounded_block_risk_nonuniform_multistate_coverage_screen.py --output <new-empty-directory>`. The default output is the versioned report folder; existing output is never overwritten.", "",
        "![Coverage and LCB sharpness](coverage_and_sharpness.png)", "",
        f"The score stream uses a four-state reversible refresh chain with stationary law `{pi.tolist()}` and transition `P=0.7 I + 0.3 1 pi`. Its numerically checked eigenvalues are `{model['eigenvalues_sorted']}`, consistent with the exact spectrum `1,0.7,0.7,0.7`; hence the exact normal-kernel pseudo-gap is `1-0.7^2=0.51`. Every chain starts at state 0, burns 24 transitions, and then contributes {PLANNED_BLOCKS:,} scored states. The exact post-burn-in TV distance is `{TV_EXACT}`; the reader receives its upward-rounded float bound `{TV_UPPER:.9g}`.",
        "",
        f"An independent anchor stream draws {ANCHOR_SAMPLE_SIZE} Bernoulli observations at target `p=0.4` and forms a two-sided exact Clopper-Pearson interval with `alpha_anchor=0.025`. A separate score chain maps its four states to reference predictions `{list(REFERENCE_BY_STATE)}` and compares them with candidate prediction `{CANDIDATE_PREDICTION}` over that random target interval. The score interval is uniformized with the production exact target grid, support-gap Markov bound, TV-adjusted alpha, `2*TV` mean-shift penalty and Lipschitz mesh cover cost. The expected stationary score is affine in the target, so its exact minimum over each sampled interval is evaluated at the appropriate endpoint.",
        "",
        f"The run planned {REPLICATES:,} independent anchor/score pairs, {REPLICATES * ANCHOR_SAMPLE_SIZE:,} anchor observations, {REPLICATES * PLANNED_BLOCKS:,} scored Markov states and {REPLICATES * BURN_IN:,} burn-in transitions. Failures/abstentions are `0/0` by construction. The 8-replicate vectorized implementation check matches the production helper within `{summary['helper_equivalence']['max_abs_difference']:.3g}` (tolerance `2e-14`).",
        "",
        "| Event | Successes / planned | Rate | Wilson 95% Monte Carlo interval |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in (("Anchor interval covers p", "anchor_coverage"),
                       ("Score LCB covers interval minimum", "score_uniform_coverage"),
                       ("Joint event", "joint_coverage")):
        item = summary[key]
        readme.append(
            f"| {label} | {item['successes']}/{item['trials']} | {item['rate']:.4f} | "
            f"[{item['wilson95_low']:.4f}, {item['wilson95_high']:.4f}] |"
        )
    readme += [
        "",
        f"Median target interval width is `{summary['median_target_interval_width']:.4f}`; median exact interval minimum is `{summary['median_exact_interval_minimum_score']:.4f}`; median uniform LCB is `{summary['median_uniform_lcb']:.4f}`; positive-LCB rate is `{summary['positive_lcb_rate']:.1%}`. Coverage counts describe this finite simulation and its exact toy DGP, not physical coverage.",
        "",
        "The key new dependency feature is a nonuniform stationary law with state-dependent scores and a nonstationary fixed start. The chain is nevertheless analytically tractable: refresh-or-stay gives a known reversible kernel, exact spectrum and exact TV after burn-in. The 95% joint lower guarantee follows from the declared independent anchor event plus the score-chain family bound; receipt hashes are identifiers for synthetic fixtures only.",
        "",
        "Data flow: `independent Bernoulli anchors -> exact random interval` plus `state-0 initialization -> 24 refresh transitions -> four-state Markov score chain -> all-state score map -> production helper-equivalent simultaneous uniform LCB -> anchor / score / joint coverage and sharpness figure`.",
        "",
        "This closes one additional software/method stress category only. It does not validate HMC independence, gap, TV or burn-in; does not establish anchor-to-E common-bias coverage; and does not provide production map/support, reference, review or authorization receipts. No production or sealed input, GPU or QUDA was accessed. Pion remains `NOT_READY/UNCERTIFIED`, `physical_score=null`, sealed inputs closed.",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    artifact_hashes = {
        str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (out / "artifact_hashes.json").write_text(
        json.dumps({"schema": "pion-nonuniform-multistate-artifact-hashes-v1",
                    "artifacts": artifact_hashes}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = build_report(args.output)
    method = result["method_summary"]
    print(json.dumps({
        "planned_replicates": method["planned_replicates"],
        "anchor_coverage": method["anchor_coverage"],
        "score_uniform_coverage": method["score_uniform_coverage"],
        "joint_coverage": method["joint_coverage"],
        "positive_lcb_rate": method["positive_lcb_rate"],
        "helper_equivalence": method["helper_equivalence"],
    }, indent=2, sort_keys=True))
