"""Strict CPU reader and joint spectral-fit diagnostics for D04 E-chain bytes.

This module deliberately stops before the D04 evidence gate.  It makes the
raw-array layout, folding, common covariance and six candidate-blind fit
variants executable, but does not establish E-chain provenance, dependence,
review independence, a valid physical reference, or bootstrap coverage.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution

from core.npz_safety import validate_npz_payload
from core.source_integrity import SourceByteSnapshot


MANIFEST_SCHEMA = "optimize-d04-e-chain-manifest-v1"
PROTOCOL_SCHEMA = "optimize-d04-e-chain-source-protocol-v1"
CORRELATOR_SCHEMA = "optimize-d04-raw-correlators-v1"
OUTPUT_SCHEMA = "optimize-d04-joint-fit-diagnostic-v1"
TEMPORAL_EXTENT = 16
PP_ORDER = ("PP_n0", "PP_n8", "PP_n16")
TIMES = (2, 3, 4, 5, 6, 7, 8)
SHRINKAGE = 0.25
EIGENVALUE_FLOOR = 1e-10
_REQUIRED_MANIFEST = {
    "schema", "split_id", "geometry", "action", "kappa",
    "temporal_correlator_boundary", "source_protocol_sha256",
    "configuration_ids_sha256", "trajectory_ids_sha256",
    "correlator_data_sha256", "pp_order", "dimension_order",
}
_FORBIDDEN_SPLIT_MARKERS = ("gap", "validation", "holdout", "sealed", "confirmation")
_ALL_FIT_CHANNELS = ("PP0", "PP1", "PP2", "A4P")
_PP_CHANNELS = ("PP0", "PP1", "PP2")
_PRIMARY_VARIANT = "joint_tmin3_nstates2"


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(snapshot: SourceByteSnapshot, source_name: str) -> dict[str, Any]:
    payload = snapshot.read_bytes(source_name)
    if sha256(payload).hexdigest() != snapshot.expected_hashes.get(source_name):
        raise ValueError(f"{source_name}: captured bytes do not match expected hash")
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {token}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source_name}: invalid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{source_name}: root must be a JSON object")
    return value


def _scalar_text(value: object, label: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "US":
        raise ValueError(f"{label} must be a scalar string")
    return str(array.item())


def _id_vector(value: object, label: str) -> np.ndarray:
    ids = np.asarray(value)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or ids.dtype.kind == "b":
        raise ValueError(f"{label} must be a one-dimensional integer array")
    # Hash ID bytes in a portable, explicit little-endian representation.
    ids = ids.astype("<i8", copy=False)
    if len(ids) < 12 or np.any(np.diff(ids) <= 0):
        raise ValueError(f"{label} must contain at least 12 strictly increasing IDs")
    return ids


def fold_pp_even(pp: object, temporal_extent: int = TEMPORAL_EXTENT) -> np.ndarray:
    """Fold real PP channels evenly; keep t=0 and T/2 once each."""
    values = np.asarray(pp)
    if (values.ndim < 1 or values.shape[-1] != temporal_extent
            or temporal_extent % 2 or values.dtype.kind != "f"
            or not np.all(np.isfinite(values))):
        raise ValueError("PP must be finite real data with an even temporal extent")
    half = temporal_extent // 2
    return np.stack(
        [values[..., 0]]
        + [0.5 * (values[..., t] + values[..., temporal_extent - t])
           for t in range(1, half)]
        + [values[..., half]], axis=-1,
    )


def fold_a4p_odd(a4p: object, temporal_extent: int = TEMPORAL_EXTENT) -> np.ndarray:
    """Fold real A4P antisymmetrically; the midpoint is exactly zero."""
    values = np.asarray(a4p)
    if (values.ndim < 1 or values.shape[-1] != temporal_extent
            or temporal_extent % 2 or values.dtype.kind != "f"
            or not np.all(np.isfinite(values))):
        raise ValueError("A4P must be finite real data with an even temporal extent")
    half = temporal_extent // 2
    return np.stack(
        [np.zeros_like(values[..., 0])]
        + [0.5 * (values[..., t] - values[..., temporal_extent - t])
           for t in range(1, half)]
        + [np.zeros_like(values[..., half])], axis=-1,
    )


def _raw_reflection_diagnostics(pp: np.ndarray, a4p: np.ndarray) -> dict[str, Any]:
    """Summarize ensemble-mean parity defects without assuming per-config symmetry."""
    half = TEMPORAL_EXTENT // 2
    pp_mean = pp.mean(axis=0)
    a_mean = a4p.mean(axis=0)
    pp_num = np.array([pp_mean[:, t] - pp_mean[:, TEMPORAL_EXTENT - t]
                       for t in range(1, half)])
    a_num = np.array([a_mean[t] + a_mean[TEMPORAL_EXTENT - t]
                      for t in range(1, half)])
    pp_scale = max(float(np.sqrt(np.mean(pp_mean[:, 1:half] ** 2))), 1e-300)
    a_scale = max(float(np.sqrt(np.mean(a_mean[1:half] ** 2))), 1e-300)
    return {
        "pp_even_relative_rms_defect": float(np.sqrt(np.mean(pp_num ** 2)) / pp_scale),
        "a4p_odd_relative_rms_defect": float(np.sqrt(np.mean(a_num ** 2)) / a_scale),
        "interpretation": "descriptive ensemble-mean checks; no threshold or per-configuration symmetry imposed",
    }


def _covariance_of_mean(folded_pp: np.ndarray, folded_a4p: np.ndarray) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Build one regularized covariance over the shared t=2..8 fit support."""
    columns: list[tuple[str, int]] = []
    values = []
    for channel in range(3):
        for t in TIMES:
            columns.append((f"PP{channel}", t))
            values.append(folded_pp[:, channel, t])
    for t in TIMES[:-1]:  # A4P is identically zero at T/2 and carries no fit datum there.
        columns.append(("A4P", t))
        values.append(folded_a4p[:, t])
    matrix = np.column_stack(values)
    covariance = np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1)) / len(matrix)
    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - SHRINKAGE) * covariance + SHRINKAGE * diagonal
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = float(np.max(eigenvalues))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("D04 correlator covariance is unidentifiable")
    floor = scale * EIGENVALUE_FLOOR
    eigenvalues = np.maximum(eigenvalues, floor)
    return (eigenvectors * eigenvalues) @ eigenvectors.T, columns


def _softplus(value: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, value)


def _inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


def _masses(parameters: np.ndarray, nstates: int) -> np.ndarray:
    increments = _softplus(parameters[:nstates]) + 1e-5
    return np.cumsum(increments)


def _selected_indices(
    columns: list[tuple[str, int]], tmin: int,
    channels: tuple[str, ...] = _ALL_FIT_CHANNELS,
) -> list[int]:
    return [i for i, (channel, t) in enumerate(columns)
            if channel in channels and t >= tmin and (channel != "A4P" or t < 8)]


def _design(
    masses: np.ndarray, nstates: int, tmin: int,
    channels: tuple[str, ...] = _ALL_FIT_CHANNELS,
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    times_pp = np.arange(tmin, 9, dtype=float)
    times_a4p = np.arange(tmin, 8, dtype=float)
    pp_basis = np.exp(-masses[:, None] * times_pp) + np.exp(
        -masses[:, None] * (TEMPORAL_EXTENT - times_pp))
    a_basis = np.exp(-masses[:, None] * times_a4p) - np.exp(
        -masses[:, None] * (TEMPORAL_EXTENT - times_a4p))
    pp_channels = tuple(channel for channel in channels if channel.startswith("PP"))
    include_a4p = "A4P" in channels
    nrows = len(pp_channels) * len(times_pp) + (len(times_a4p) if include_a4p else 0)
    namplitudes = len(pp_channels) * nstates + (nstates if include_a4p else 0)
    matrix = np.zeros((nrows, namplitudes))
    row = 0
    amplitude_column = 0
    for channel in pp_channels:
        width = len(times_pp)
        matrix[row:row + width,
               amplitude_column:amplitude_column + nstates] = pp_basis.T
        row += width
        amplitude_column += nstates
    labels = [(channel, int(t)) for channel in pp_channels for t in times_pp]
    if include_a4p:
        matrix[row:, amplitude_column:amplitude_column + nstates] = a_basis.T
        labels.extend(("A4P", int(t)) for t in times_a4p)
    return matrix, labels


def _fit_variant(
    mean_vector: np.ndarray,
    covariance: np.ndarray,
    columns: list[tuple[str, int]],
    *, tmin: int, nstates: int,
    channels: tuple[str, ...] = _ALL_FIT_CHANNELS,
) -> dict[str, Any]:
    selected = _selected_indices(columns, tmin, channels)
    y = mean_vector[selected]
    cov = covariance[np.ix_(selected, selected)]
    evals, evecs = np.linalg.eigh(cov)
    scale = float(np.max(evals))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("fit-window covariance is unidentifiable")
    evals = np.maximum(evals, scale * EIGENVALUE_FLOOR)
    lower_mass_coordinate, upper_mass_coordinate = -8.0, 3.0
    namplitudes = (sum(channel.startswith("PP") for channel in channels)
                   + int("A4P" in channels)) * nstates
    lower = np.r_[np.full(nstates, lower_mass_coordinate), np.full(namplitudes, -np.inf)]
    upper = np.r_[np.full(nstates, upper_mass_coordinate), np.full(namplitudes, np.inf)]
    starts = []
    for base, gap in ((0.25, 0.35), (0.40, 0.65), (0.18, 0.80)):
        starts.append(np.array([_inverse_softplus(base)] +
                               [_inverse_softplus(gap)] * (nstates - 1)))
    attempts = []
    fits = []
    for start_id, mass_coordinates in enumerate(starts):
        m = _masses(mass_coordinates, nstates)
        matrix, model_labels = _design(m, nstates, tmin, channels)
        # Use the same fixed support/order for covariance-selected observations.
        if model_labels != [columns[index] for index in selected]:
            raise ValueError("internal D04 fit design and covariance order disagree")
        amplitudes, *_ = np.linalg.lstsq(matrix, y, rcond=None)
        x0 = np.r_[mass_coordinates, amplitudes]

        def residual(parameters):
            masses = _masses(parameters, nstates)
            design, _ = _design(masses, nstates, tmin, channels)
            raw = design @ parameters[nstates:] - y
            return (evecs.T @ raw) / np.sqrt(evals)

        try:
            result = least_squares(
                residual, x0, bounds=(lower, upper), method="trf",
                x_scale="jac", max_nfev=5000, ftol=1e-11, xtol=1e-11, gtol=1e-11,
            )
            value = float(np.dot(result.fun, result.fun))
            converged = bool(result.success and np.isfinite(value)
                             and np.all(np.isfinite(result.x)))
            attempts.append({
                "start_id": start_id, "converged": converged,
                "status": int(result.status), "nfev": int(result.nfev),
                "cost": value if np.isfinite(value) else None,
                "message": str(result.message),
            })
            if converged:
                fits.append((value, result, evecs, evals))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            attempts.append({"start_id": start_id, "converged": False,
                             "status": "exception", "nfev": 0,
                             "cost": None, "message": type(exc).__name__})
    if not fits:
        return {"status": "failed", "tmin": tmin, "nstates": nstates,
                "fit_channels": list(channels),
                "planned_multistarts": len(starts), "attempts": attempts,
                "reference_mass": None, "chi2": None, "dof": None,
                "gof_p_approx": None, "boundary_hit_count": None}
    value, best, _, _ = min(fits, key=lambda item: item[0])
    boundary = np.logical_or(best.x[:nstates] - lower_mass_coordinate < 1e-5,
                             upper_mass_coordinate - best.x[:nstates] < 1e-5)
    dof = int(len(y) - len(best.x))
    p_value = float(chi2_distribution.sf(value, dof)) if dof > 0 else None
    masses = _masses(best.x, nstates)
    minimum_gap = float(np.min(np.diff(masses))) if nstates > 1 else None
    return {
        "status": "valid", "tmin": tmin, "tmax": 8, "nstates": nstates,
        "fit_channels": list(channels),
        "reference_mass": float(masses[0]), "state_masses": masses.tolist(),
        "minimum_fitted_state_gap": minimum_gap,
        "near_degenerate_state_gap_below_0p05": bool(minimum_gap is not None and minimum_gap < 0.05),
        "chi2": value, "dof": dof, "gof_p_approx": p_value,
        "boundary_hit_count": int(np.sum(boundary)),
        "successful_multistarts": len(fits), "planned_multistarts": len(starts),
        "attempts": attempts,
        "interpretation": "chi-square p is an approximate GLS diagnostic; dependence and finite-sample calibration are not established",
    }


def _fit_inputs(folded_pp: np.ndarray, folded_a4p: np.ndarray):
    covariance, columns = _covariance_of_mean(folded_pp, folded_a4p)
    matrix = np.column_stack(
        [folded_pp[:, channel, t] for channel in range(3) for t in TIMES]
        + [folded_a4p[:, t] for t in TIMES[:-1]]
    )
    return matrix.mean(axis=0), covariance, columns


def _d04_fit_specs() -> list[dict[str, Any]]:
    specs = [
        {"variant_id": f"joint_tmin{tmin}_nstates{nstates}",
         "tmin": tmin, "nstates": nstates, "channels": _ALL_FIT_CHANNELS}
        for tmin in (2, 3, 4) for nstates in (2, 3)
    ]
    specs.extend((
        {"variant_id": "pp_only_tmin3_nstates2", "tmin": 3,
         "nstates": 2, "channels": _PP_CHANNELS},
        {"variant_id": "a4p_only_tmin3_nstates2", "tmin": 3,
         "nstates": 2, "channels": ("A4P",)},
    ))
    return specs


def _circular_moving_block_indices(
    n_observations: int, block_length: int, rng: np.random.Generator,
) -> np.ndarray:
    if (not isinstance(block_length, int) or isinstance(block_length, bool)
            or block_length < 1 or block_length > n_observations):
        raise ValueError("block length must be an integer in [1, number of configurations]")
    n_blocks = int(np.ceil(n_observations / block_length))
    starts = rng.integers(0, n_observations, size=n_blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % n_observations).ravel()[:n_observations]


def _bootstrap_interval(values: list[float], planned: int) -> dict[str, Any]:
    valid = len(values)
    failed = planned - valid
    if valid < max(2, int(np.ceil(0.90 * planned))):
        return {
            "status": "abstained", "denominator": {
                "planned": 1, "valid": 0, "failed": 0, "abstained": 1,
            },
            "draw_denominator": {"planned": planned, "valid": valid,
                                 "failed": failed, "abstained": 0},
            "lower_95_percentile": None, "upper_95_percentile": None,
            "bootstrap_se": None,
            "reason": "fewer than 90% of paired bootstrap draws are valid",
        }
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("bootstrap interval input contains non-finite values")
    return {
        "status": "diagnostic_only", "denominator": {
            "planned": 1, "valid": 1, "failed": 0, "abstained": 0,
        },
        "draw_denominator": {"planned": planned, "valid": valid,
                             "failed": failed, "abstained": 0},
        "lower_95_percentile": float(np.quantile(array, 0.025)),
        "upper_95_percentile": float(np.quantile(array, 0.975)),
        "bootstrap_se": float(np.std(array, ddof=1)),
        "reason": "empirical percentile summary; no finite-sample coverage guarantee",
    }


def _block_bootstrap_diagnostic(
    pp: np.ndarray, a4p: np.ndarray, *, block_lengths: tuple[int, ...],
    replicates: int, seed: int,
) -> dict[str, Any]:
    if (not isinstance(replicates, int) or isinstance(replicates, bool)
            or replicates < 2):
        raise ValueError("block-bootstrap replicates must be an integer >= 2")
    if (not block_lengths or any(not isinstance(length, int) or isinstance(length, bool)
                                 for length in block_lengths)
            or len(set(block_lengths)) != len(block_lengths)):
        raise ValueError("block lengths must be a nonempty tuple of unique integers")
    if any(length < 1 or length > len(pp) for length in block_lengths):
        raise ValueError("block length must be in [1, number of configurations]")

    specs = _d04_fit_specs()
    results_by_length: dict[str, Any] = {}
    primary_point = None
    point_folded_pp = fold_pp_even(pp)
    point_folded_a4p = fold_a4p_odd(a4p)
    point_mean, point_cov, point_columns = _fit_inputs(point_folded_pp, point_folded_a4p)
    point_estimates = {}
    for spec in specs:
        fit = _fit_variant(
            point_mean, point_cov, point_columns,
            tmin=spec["tmin"], nstates=spec["nstates"], channels=spec["channels"],
        )
        point_estimates[spec["variant_id"]] = fit["reference_mass"]
        if spec["variant_id"] == _PRIMARY_VARIANT and fit["status"] == "valid":
            primary_point = fit["reference_mass"]
    if primary_point is None:
        abstained = {"planned": replicates, "valid": 0, "failed": 0,
                     "abstained": replicates}
        abstained_interval = {
            "status": "abstained",
            "denominator": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
            "draw_denominator": dict(abstained),
            "lower_95_percentile": None, "upper_95_percentile": None,
            "bootstrap_se": None,
            "reason": "primary point-data fit failed before resampling",
        }
        no_reference = {}
        for block_length in block_lengths:
            no_reference[str(block_length)] = {
                "block_length": block_length,
                "seed": seed + block_length,
                "resample_covariance_denominator": dict(abstained),
                "fit_denominators": {
                    spec["variant_id"]: dict(abstained) for spec in specs
                },
                "primary_mass": dict(abstained_interval),
                "contrasts": {
                    spec["variant_id"]: {
                        "point_shift_from_primary": None,
                        "bootstrap_paired_shift": dict(abstained_interval),
                        "paired_shift_over_se": None,
                    }
                    for spec in specs if spec["variant_id"] != _PRIMARY_VARIANT
                },
            }
        return {
            "schema": "optimize-d04-circular-mbb-diagnostic-v1",
            "status": "diagnostic_incomplete", "certified": False,
            "estimator": "primary joint PP-even/A4P-odd two-state GLS, t=3..8",
            "reason": "point-data primary fit failed; all bootstrap work abstained",
            "resampling": {
                "method": "circular moving-block bootstrap over the supplied configuration order",
                "block_lengths": list(block_lengths), "replicates_per_length": replicates,
                "seed_base": seed,
            },
            "point_estimates": point_estimates,
            "by_block_length": no_reference,
            "open_gates": [
                "the primary point-data fit failed, so bootstrap intervals and paired shifts are undefined",
                "no dependence, coverage, provenance, review, or authorization receipt is established",
            ],
        }

    for block_length in block_lengths:
        rng = np.random.default_rng(seed + block_length)
        draws: dict[str, list[float | None]] = {spec["variant_id"]: [] for spec in specs}
        attempts = {spec["variant_id"]: {"planned": replicates, "valid": 0,
                                         "failed": 0, "abstained": 0}
                    for spec in specs}
        covariance_denominator = {"planned": replicates, "valid": 0,
                                  "failed": 0, "abstained": 0}
        for _ in range(replicates):
            indices = _circular_moving_block_indices(len(pp), block_length, rng)
            folded_pp = fold_pp_even(pp[indices])
            folded_a4p = fold_a4p_odd(a4p[indices])
            try:
                mean_vector, covariance, columns = _fit_inputs(folded_pp, folded_a4p)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                covariance_denominator["failed"] += 1
                for denominator in attempts.values():
                    denominator["failed"] += 1
                for variant_id in draws:
                    draws[variant_id].append(None)
                continue
            covariance_denominator["valid"] += 1
            for spec in specs:
                variant_id = spec["variant_id"]
                try:
                    fit = _fit_variant(
                        mean_vector, covariance, columns,
                        tmin=spec["tmin"], nstates=spec["nstates"],
                        channels=spec["channels"],
                    )
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    fit = {"status": "failed", "reference_mass": None}
                if fit["status"] == "valid" and fit["reference_mass"] is not None:
                    draws[variant_id].append(float(fit["reference_mass"]))
                    attempts[variant_id]["valid"] += 1
                else:
                    draws[variant_id].append(None)
                    attempts[variant_id]["failed"] += 1

        primary_values = [value for value in draws[_PRIMARY_VARIANT] if value is not None]
        mass_summary = _bootstrap_interval(primary_values, replicates)
        contrasts = {}
        for spec in specs:
            variant_id = spec["variant_id"]
            if variant_id == _PRIMARY_VARIANT:
                continue
            # All fits share the same resample indices. Pair by draw ordinal, not by
            # sorting estimates; if either fit failed, that paired draw is missing.
            paired_deltas = [float(target - primary)
                             for target, primary in zip(draws[variant_id],
                                                        draws[_PRIMARY_VARIANT])
                             if target is not None and primary is not None]
            contrasts[variant_id] = {
                "point_shift_from_primary": (
                    None if point_estimates[variant_id] is None else
                    float(point_estimates[variant_id] - primary_point)
                ),
                "bootstrap_paired_shift": _bootstrap_interval(paired_deltas, replicates),
                "paired_shift_over_se": (
                    None if len(paired_deltas) < max(2, int(np.ceil(0.90 * replicates)))
                    or np.std(paired_deltas, ddof=1) == 0 else
                    float(abs(point_estimates[variant_id] - primary_point)
                          / np.std(paired_deltas, ddof=1))
                ) if point_estimates[variant_id] is not None else None,
            }
        results_by_length[str(block_length)] = {
            "block_length": block_length,
            "seed": seed + block_length,
            "resample_covariance_denominator": covariance_denominator,
            "fit_denominators": attempts,
            "primary_mass": mass_summary,
            "contrasts": contrasts,
        }
    overall_valid = all(
        item["primary_mass"]["status"] == "diagnostic_only"
        for item in results_by_length.values()
    )
    return {
        "schema": "optimize-d04-circular-mbb-diagnostic-v1",
        "status": "diagnostic_only" if overall_valid else "diagnostic_incomplete",
        "certified": False,
        "estimator": "primary joint PP-even/A4P-odd two-state GLS, t=3..8",
        "resampling": {
            "method": "circular moving-block bootstrap over the supplied configuration order",
            "block_lengths": list(block_lengths), "replicates_per_length": replicates,
            "seed_base": seed,
            "interpretation": "L=1 is the iid bootstrap baseline; L=4,8 are user-selected sensitivity values, not estimated dependence scales",
            "assumptions_not_verified": [
                "configuration order represents a stationary sequence",
                "circular wraparound is acceptable for this finite sequence",
                "the chosen block lengths cover relevant dependence",
            ],
        },
        "point_estimates": point_estimates,
        "by_block_length": results_by_length,
        "open_gates": [
            "block lengths are sensitivity settings, not a dependence receipt",
            "bootstrap percentiles are not coverage calibrated for this chain",
            "the source reader has not authenticated provenance, independence, review, or authorization",
            "the result cannot be converted into a gate-ready reference_summary",
        ],
    }


def evaluate_d04_raw_snapshot(
    snapshot: SourceByteSnapshot,
    *, manifest_name: str = "manifest",
    correlator_name: str = "correlator_data",
    protocol_name: str = "source_protocol",
    block_bootstrap_replicates: int = 0,
    block_lengths: tuple[int, ...] = (1, 4, 8),
    block_bootstrap_seed: int = 20261127,
) -> dict[str, Any]:
    """Parse captured D04 bytes and run frozen fits plus an optional MBB pilot.

    This function emits a diagnostic object, never an evidence envelope or a
    gate-ready summary. The original snapshot is the only data source. If
    ``block_bootstrap_replicates`` is positive, the requested moving-block
    sensitivity is conditional on the supplied configuration order and block
    lengths; it is not a chain-dependence or coverage receipt.
    """
    if not isinstance(snapshot, SourceByteSnapshot):
        raise ValueError("snapshot must be a SourceByteSnapshot")
    if (not isinstance(block_bootstrap_replicates, int)
            or isinstance(block_bootstrap_replicates, bool)
            or block_bootstrap_replicates < 0):
        raise ValueError("block-bootstrap replicates must be a nonnegative integer")
    manifest = _strict_json(snapshot, manifest_name)
    if set(manifest) != _REQUIRED_MANIFEST:
        raise ValueError("D04 manifest keys do not match the frozen reader schema")
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise ValueError("D04 manifest schema mismatch")
    split_id = manifest["split_id"]
    if not isinstance(split_id, str) or not split_id.upper().startswith("E"):
        raise ValueError("D04 source must identify an independent E-chain")
    if any(marker in split_id.lower() for marker in _FORBIDDEN_SPLIT_MARKERS):
        raise ValueError("D04 refuses sealed, gap, validation, holdout, or confirmation splits")
    if (manifest["geometry"] != "S16T16" or manifest["action"] != "quenched_wilson"
            or manifest["kappa"] != 0.14
            or manifest["temporal_correlator_boundary"] != "periodic"):
        raise ValueError("D04 manifest geometry/action/kappa/boundary mismatch")
    if manifest["pp_order"] != list(PP_ORDER):
        raise ValueError("D04 PP channel order must be PP_n0, PP_n8, PP_n16")
    if manifest["dimension_order"] != "configuration,channel,time":
        raise ValueError("D04 dimension order mismatch")

    data_bytes = snapshot.read_bytes(correlator_name)
    observed_data_hash = sha256(data_bytes).hexdigest()
    if observed_data_hash != snapshot.expected_hashes.get(correlator_name):
        raise ValueError("D04 correlator bytes do not match their captured hash")
    if manifest["correlator_data_sha256"] != observed_data_hash:
        raise ValueError("D04 manifest correlator hash does not match captured bytes")
    validate_npz_payload(data_bytes, max_uncompressed_bytes=64 * 1024 * 1024)
    try:
        archive = np.load(BytesIO(data_bytes), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("D04 correlator NPZ is invalid") from exc
    with archive:
        if set(archive.files) != {
            "schema", "configuration_ids", "trajectory_ids", "pp", "a4p"
        }:
            raise ValueError("D04 correlator NPZ keys do not match the frozen schema")
        if _scalar_text(archive["schema"], "schema") != CORRELATOR_SCHEMA:
            raise ValueError("D04 correlator schema mismatch")
        config_ids = _id_vector(archive["configuration_ids"], "configuration_ids")
        trajectory_ids = _id_vector(archive["trajectory_ids"], "trajectory_ids")
        pp = np.asarray(archive["pp"])
        a4p = np.asarray(archive["a4p"])
    if len(config_ids) != len(trajectory_ids) or len(config_ids) != len(pp):
        raise ValueError("D04 configuration, trajectory and PP row counts differ")
    if pp.shape != (len(config_ids), 3, TEMPORAL_EXTENT):
        raise ValueError("D04 PP axes must be (configuration, PP channel, t=0..15)")
    if a4p.shape != (len(config_ids), TEMPORAL_EXTENT):
        raise ValueError("D04 A4P axes must be (configuration, t=0..15)")
    if pp.dtype.kind != "f" or a4p.dtype.kind != "f":
        raise ValueError("D04 raw correlators must be explicitly real; complex data are never projected")
    if not np.all(np.isfinite(pp)) or not np.all(np.isfinite(a4p)):
        raise ValueError("D04 raw correlators contain non-finite values")
    for key, values in (("configuration_ids_sha256", config_ids),
                        ("trajectory_ids_sha256", trajectory_ids)):
        observed = sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        if manifest[key] != observed:
            raise ValueError(f"D04 manifest {key} does not match the raw ID array")
    protocol_hash = manifest["source_protocol_sha256"]
    if (not isinstance(protocol_hash, str) or len(protocol_hash) != 64
            or any(ch not in "0123456789abcdef" for ch in protocol_hash)):
        raise ValueError("D04 source protocol hash must be lowercase SHA-256")
    protocol = _strict_json(snapshot, protocol_name)
    protocol_payload = snapshot.read_bytes(protocol_name)
    if sha256(protocol_payload).hexdigest() != protocol_hash:
        raise ValueError("D04 source protocol bytes do not match the manifest hash")
    protocol_keys = {
        "schema", "geometry", "action", "kappa", "temporal_correlator_boundary",
        "pp_order", "dimension_order", "temporal_extent",
    }
    if set(protocol) != protocol_keys or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("D04 source protocol keys or schema do not match the reader contract")
    if (protocol.get("geometry") != "S16T16"
            or protocol.get("action") != "quenched_wilson"
            or protocol.get("kappa") != 0.14
            or protocol.get("temporal_correlator_boundary") != "periodic"
            or protocol.get("pp_order") != list(PP_ORDER)
            or protocol.get("dimension_order") != "configuration,channel,time"
            or protocol.get("temporal_extent") != TEMPORAL_EXTENT):
        raise ValueError("D04 source protocol geometry/action/kappa/channel convention mismatch")

    folded_pp = fold_pp_even(pp)
    folded_a4p = fold_a4p_odd(a4p)
    mean_vector, covariance, columns = _fit_inputs(folded_pp, folded_a4p)
    fits = [_fit_variant(mean_vector, covariance, columns, tmin=tmin, nstates=nstates)
            for tmin in (2, 3, 4) for nstates in (2, 3)]
    valid = sum(fit["status"] == "valid" for fit in fits)
    failed = len(fits) - valid
    result = {
        "schema": OUTPUT_SCHEMA,
        "status": "diagnostic_only" if failed == 0 else "diagnostic_incomplete",
        "certified": False, "physical_score": None,
        "data_access": "captured_snapshot_bytes_only",
        "source_hashes": {
            manifest_name: snapshot.expected_hashes[manifest_name],
            correlator_name: observed_data_hash,
            protocol_name: protocol_hash,
        },
        "split_id": split_id,
        "configuration_ids": {"first": int(config_ids[0]), "last": int(config_ids[-1]),
                               "count": len(config_ids)},
        "trajectory_ids": {"first": int(trajectory_ids[0]), "last": int(trajectory_ids[-1]),
                           "count": len(trajectory_ids)},
        "folding": {
            "pp": "0.5*(C(t)+C(T-t)); t=0,T/2 kept once",
            "a4p": "0.5*(C(t)-C(T-t)); t=T/2 fixed to zero and excluded from fit",
            "reflection_diagnostics": _raw_reflection_diagnostics(pp, a4p),
        },
        "fit_protocol": {
            "windows": [[tmin, 8] for tmin in (2, 3, 4)],
            "state_counts": [2, 3], "common_covariance_for_all_variants": True,
            "diagonal_shrinkage": SHRINKAGE, "relative_eigenvalue_floor": EIGENVALUE_FLOOR,
            "multistarts_per_variant": 3,
        },
        "fit_denominator": {"unit": "joint_fit_variant", "planned": 6,
                            "valid": valid, "failed": failed, "abstained": 0},
        "fit_variants": fits,
        "open_gates": [
            "no dependence-aware block bootstrap or paired fit-shift uncertainty is computed",
            "approximate GLS goodness-of-fit is not finite-sample calibrated",
            "the reader does not authenticate chain provenance, review independence, or authorization",
            "a valid synthetic fit does not produce a gate-ready D04 reference summary",
        ],
    }
    if block_bootstrap_replicates:
        result["block_bootstrap"] = _block_bootstrap_diagnostic(
            pp, a4p, block_lengths=block_lengths,
            replicates=block_bootstrap_replicates, seed=block_bootstrap_seed,
        )
        result["open_gates"] = [
            "block lengths are sensitivity settings, not a dependence receipt",
            "bootstrap percentiles are not coverage calibrated for this chain",
            "the reader does not authenticate provenance, review independence, or authorization",
            "the result cannot be converted into a gate-ready reference_summary",
        ]
    return result
