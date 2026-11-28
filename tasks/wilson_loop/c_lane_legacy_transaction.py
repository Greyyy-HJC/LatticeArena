"""Read-only bridge from archived Wilson measurement transactions to C schema.

The bridge consumes only caller-captured bytes. It validates the old NPZ,
provenance, commit marker, and HMC input manifest before deriving the current
finite-T Creutz reader inputs. Hash agreement is an internal consistency check;
it does not authenticate the producer or certify the physics.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from io import BytesIO
from types import MappingProxyType
from typing import Any

import numpy as np

from core.io import fingerprint
from core.npz_safety import validate_npz_payload
from core.numerics import as_real_float_array
from core.source_integrity import SourceByteSnapshot

from . import c_lane_snapshot_reader as c_reader
from .lane_contract import selected_protocol


_ROLES = frozenset({
    "producer_npz", "producer_provenance", "producer_commit", "input_manifest",
})
_DEPENDENCIES = frozenset({
    "kernels.py", "statistics.py", "measure.py", "common.py", "identity.py",
})
_PINNED_ARCHIVE_SOURCES = {
    "measure.py": "5fdee931db8a0b9bc74debb6f8c53e659ec6ccef7a42b0443788db4593e84ef6",
    "kernels.py": "a6b54eb606f1b7c790de596185d8c0190601fde4495d9f26eba89ba16abdd7dc",
    "stout.py": "7d9f5f8ea0c0e006b09f58abc087e5825fd74c5efecf085db176c7c23ed69f35",
    "statistics.py": "27c691dad427deb5e46447e8a9d6de6eea9027b65d5c99f0290f8ec2590f1c86",
    "common.py": "74de1e8cfaafbae2744855e85992561681057832f52bf537255f9d35230dd29e",
    "identity.py": "d15841a933c22502a616a9e3d56d12dff13dc11227d26343a3da325f03b65fbe",
}
# SHA-256 values are from the retained implementation at this commit.
_PINNED_ARCHIVE_COMMIT = "54f0e97ba85b7f8f724c31d9f9d3512f040583a9"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OLD_NPZ_KEYS = frozenset({"ids", "loops", "protocol_hash", "generation"})
_OLD_COMMIT_KEYS = frozenset({
    "generation", "split", "npz_sha256", "provenance_sha256", "loops_sha256",
})
_PRIMARY_KERNEL_ALIASES = frozenset({
    "stout_rho0.1_n5", "spatial_stout_rho0.10_n5",
})


class _Refusal(ValueError):
    def __init__(self, status: str, reason: str, denominator: dict[str, Any] | None = None):
        super().__init__(reason)
        self.status = status
        self.denominator = denominator


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} has a duplicate JSON key")
            value[key] = item
        return value

    def reject_constant(_value):
        raise ValueError(f"{label} contains a non-finite JSON constant")

    try:
        result = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{label} must have an object root")
    return result


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _axis_matches(value: object, expected: tuple[int, ...]) -> bool:
    return (
        isinstance(value, list)
        and all(type(item) is int for item in value)
        and value == list(expected)
    )


def _npz_text(archive: Any, name: str) -> str:
    value = np.asarray(archive[name])
    if value.ndim != 0 or value.dtype.kind not in "US":
        raise ValueError(f"legacy NPZ field {name} must be a scalar string")
    item = value.item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _source_payloads(
    snapshot: SourceByteSnapshot,
    source_names: object,
    denominator: dict[str, Any],
) -> tuple[dict[str, bytes], dict[str, str]]:
    if not isinstance(snapshot, SourceByteSnapshot):
        raise _Refusal("ABSTAIN_C_LEGACY_SNAPSHOT", "input is not a SourceByteSnapshot")
    if not isinstance(source_names, Mapping) or set(source_names) != _ROLES:
        raise _Refusal("ABSTAIN_C_LEGACY_SOURCE_MAP", "source_names must map exactly the four legacy inputs")
    if any(not isinstance(name, str) or not name for name in source_names.values()):
        raise _Refusal("ABSTAIN_C_LEGACY_SOURCE_MAP", "captured source keys must be nonempty strings")
    source_keys = list(source_names.values())
    if len(set(source_keys)) != len(source_keys) or set(snapshot.expected_hashes) != set(source_keys):
        raise _Refusal("ABSTAIN_C_LEGACY_SOURCE_MAP", "snapshot must contain exactly the four mapped legacy inputs")

    payloads: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    counts = {"valid": 0, "failed": 0, "abstained": 0}
    for role in sorted(_ROLES):
        source_key = source_names[role]
        try:
            payload = snapshot.read_bytes(source_key)
        except (KeyError, TypeError, ValueError) as exc:
            entry = snapshot.capture_entries.get(source_key, {})
            status = entry.get("status")
            counts["failed" if status == "failed" else "abstained"] += 1
            continue
        digest = _sha256(payload)
        if snapshot.expected_hashes.get(source_key) != digest:
            counts["failed"] += 1
            continue
        payloads[role] = payload
        hashes[role] = digest
        counts["valid"] += 1
    denominator.update(counts)
    if counts["valid"] != len(_ROLES):
        raise _Refusal(
            "ABSTAIN_C_LEGACY_SOURCE_BYTES",
            "one or more legacy transaction files are missing, failed capture, or have a digest mismatch",
            denominator.copy(),
        )
    return payloads, hashes


def _canonical_boundary_conditions(value: object) -> dict[str, str]:
    selected = selected_protocol()["ensemble_design"]["boundary_conditions"]
    if value == "periodic_all_directions":
        return dict(selected)
    if value == selected:
        return dict(selected)
    raise _Refusal("ABSTAIN_C_LEGACY_PROTOCOL", "producer boundary conditions are not periodic in space and time")


def _canonical_kernel_descriptors(
    protocol: Mapping[str, Any],
    primary_kernel_id: str,
) -> tuple[list[str], dict[str, dict[str, Any]], int]:
    descriptors = protocol.get("kernels")
    if not isinstance(descriptors, list) or not descriptors:
        raise _Refusal("ABSTAIN_C_LEGACY_KERNEL", "producer protocol has no kernel list")
    ids: list[str] = []
    parameters: dict[str, dict[str, Any]] = {}
    primary_indices: list[int] = []
    for index, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, dict):
            raise _Refusal("ABSTAIN_C_LEGACY_KERNEL", "kernel descriptors must be objects")
        name = descriptor.get("name")
        if not isinstance(name, str) or not name:
            raise _Refusal("ABSTAIN_C_LEGACY_KERNEL", "kernel descriptor has no string name")
        if name in _PRIMARY_KERNEL_ALIASES:
            if set(descriptor) != {"name", "kind", "rho", "steps"}:
                raise _Refusal("ABSTAIN_C_LEGACY_KERNEL", "stout descriptor has unreviewed extra parameters")
            if (descriptor.get("kind"), descriptor.get("rho"), descriptor.get("steps")) != (
                "stout", 0.1, 5,
            ):
                raise _Refusal("ABSTAIN_C_LEGACY_KERNEL", "stout descriptor differs from rho=0.10, five steps")
            canonical = primary_kernel_id
            primary_indices.append(index)
            parameters[canonical] = {"kind": "stout", "rho": 0.1, "steps": 5}
        else:
            canonical = name
        ids.append(canonical)
    if len(set(ids)) != len(ids):
        raise _Refusal("ABSTAIN_C_LEGACY_KERNEL", "kernel aliases collide after canonical mapping")
    if len(primary_indices) != 1:
        raise _Refusal("ABSTAIN_C_LEGACY_KERNEL", "producer transaction needs exactly one selected stout kernel")
    return ids, parameters, primary_indices[0]


def _memory_snapshot(payloads: Mapping[str, bytes]) -> SourceByteSnapshot:
    hashes = {name: _sha256(data) for name, data in payloads.items()}
    entries = {
        name: {
            "status": "valid", "path": f"derived:{name}",
            "expected_sha256": hashes[name], "observed_sha256": hashes[name],
            "reason": None,
        }
        for name in payloads
    }
    return SourceByteSnapshot(
        root="derived-from-captured-legacy-transaction",
        expected_hashes=hashes,
        source_paths={name: f"derived:{name}" for name in payloads},
        source_bytes=payloads,
        capture_entries=entries,
        max_total_bytes=sum(len(payload) for payload in payloads.values()),
    )


def _abort(
    status: str,
    reason: str,
    source_denominator: dict[str, Any],
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "schema": "wilson-c-legacy-transaction-adapter-v1",
        "status": status,
        "reason": reason,
        "source_denominator": source_denominator,
        "certified": False,
        "physical_score": None,
        "canonical_snapshot": None,
    }
    if details:
        result.update(details)
    return result


def adapt_legacy_calibration_snapshot(
    snapshot: SourceByteSnapshot,
    *,
    source_names: object,
    calibration_opt_in: bool = False,
) -> dict[str, Any]:
    """Validate a legacy producer transaction and derive current C reader bytes.

    Calibration use is explicit and disabled by default. The function has no
    filesystem path fallback and never reads raw gauge files. Returned
    canonical bytes are in-memory derivatives of the captured transaction.
    """
    denominator = {
        "unit": "captured legacy input file",
        "planned": len(_ROLES), "valid": 0, "failed": 0, "abstained": len(_ROLES),
    }
    configuration_denominator = None
    reader_result = None
    if calibration_opt_in is not True:
        return _abort(
            "ABSTAIN_C_LEGACY_OPT_IN",
            "calibration_opt_in must be explicitly set to True",
            denominator,
        )
    try:
        payloads, source_hashes = _source_payloads(snapshot, source_names, denominator)
        provenance = _strict_json(payloads["producer_provenance"], "producer provenance")
        commit = _strict_json(payloads["producer_commit"], "producer commit marker")
        hmc_manifest = _strict_json(payloads["input_manifest"], "HMC input manifest")
        if set(commit) != _OLD_COMMIT_KEYS:
            raise _Refusal("ABSTAIN_C_LEGACY_COMMIT", "legacy commit marker schema is not exact")
        required_provenance = {
            "protocol", "kernel_sha256", "input_manifest_sha256",
            "generation", "dependencies", "records", "git",
        }
        if not required_provenance.issubset(provenance):
            raise _Refusal("ABSTAIN_C_LEGACY_PROVENANCE", "producer provenance is missing required transaction fields")
        protocol = provenance.get("protocol")
        if not isinstance(protocol, dict):
            raise _Refusal("ABSTAIN_C_LEGACY_PROVENANCE", "producer protocol must be an object")

        split = "calibration"
        allowed = protocol.get("measurement_splits", ["development", "gap", "validation"])
        if not isinstance(allowed, list) or split not in allowed:
            raise _Refusal("ABSTAIN_C_LEGACY_OPT_IN", "producer protocol does not explicitly allow calibration measurement")
        if split not in protocol or not isinstance(protocol[split], list) or len(protocol[split]) != 2:
            raise _Refusal("ABSTAIN_C_LEGACY_SPLIT", "producer protocol has no inclusive calibration ID range")
        split_range = protocol[split]
        if (any(type(value) is not int for value in split_range)
                or split_range[0] < 0 or split_range[0] > split_range[1]):
            raise _Refusal("ABSTAIN_C_LEGACY_SPLIT", "calibration ID range is malformed")
        if commit.get("split") != split or provenance.get("generation") != commit.get("generation"):
            raise _Refusal("ABSTAIN_C_LEGACY_IDENTITY", "commit, provenance, and selected split/generation disagree")

        npz_sha = source_hashes["producer_npz"]
        provenance_sha = source_hashes["producer_provenance"]
        if commit.get("npz_sha256") != npz_sha or commit.get("provenance_sha256") != provenance_sha:
            raise _Refusal("ABSTAIN_C_LEGACY_COMMIT", "commit marker does not bind the captured NPZ and provenance")
        _require_sha(commit.get("generation"), "legacy transaction generation")
        if provenance.get("input_manifest_sha256") != source_hashes["input_manifest"]:
            raise _Refusal("ABSTAIN_C_LEGACY_INPUT_MANIFEST", "provenance does not bind the captured HMC input manifest")

        input_spec = protocol.get("input_manifest")
        if not isinstance(input_spec, dict):
            raise _Refusal("ABSTAIN_C_LEGACY_INPUT_MANIFEST", "producer protocol lacks an input-manifest specification")
        manifest_sha = _require_sha(input_spec.get("sha256"), "protocol input manifest SHA")
        if manifest_sha != source_hashes["input_manifest"]:
            raise _Refusal("ABSTAIN_C_LEGACY_INPUT_MANIFEST", "captured HMC manifest differs from the protocol hash")
        if hmc_manifest.get("schema") != input_spec.get("schema"):
            raise _Refusal("ABSTAIN_C_LEGACY_INPUT_MANIFEST", "HMC manifest schema differs from the producer protocol")
        if hmc_manifest.get("ensemble") != input_spec.get("ensemble"):
            raise _Refusal("ABSTAIN_C_LEGACY_INPUT_MANIFEST", "HMC manifest ensemble differs from the producer protocol")
        geometry = protocol.get("geometry")
        if (not isinstance(geometry, list) or any(type(value) is not int for value in geometry)
                or hmc_manifest.get("lattice") != geometry):
            raise _Refusal("ABSTAIN_C_LEGACY_PROTOCOL", "HMC lattice geometry differs from the producer protocol")
        if "beta" in input_spec and hmc_manifest.get("beta") != input_spec["beta"]:
            raise _Refusal("ABSTAIN_C_LEGACY_INPUT_MANIFEST", "HMC beta differs from the producer input-manifest specification")
        if "hmc_seed" in input_spec and hmc_manifest.get("seed") != input_spec["hmc_seed"]:
            raise _Refusal("ABSTAIN_C_LEGACY_INPUT_MANIFEST", "HMC seed differs from the producer input-manifest specification")

        dependencies = provenance.get("dependencies")
        if not isinstance(dependencies, dict) or set(dependencies) != _DEPENDENCIES:
            raise _Refusal("ABSTAIN_C_LEGACY_DEPENDENCIES", "producer dependency map is incomplete or has unknown entries")
        for name, digest in dependencies.items():
            _require_sha(digest, f"producer dependency {name}")
        kernel_sha = _require_sha(provenance.get("kernel_sha256"), "producer kernel SHA")
        if kernel_sha != dependencies["kernels.py"]:
            raise _Refusal("ABSTAIN_C_LEGACY_DEPENDENCIES", "kernel SHA differs from the dependency map")
        protocol_sources = protocol.get("source_hashes")
        if not isinstance(protocol_sources, dict) or set(protocol_sources) != set(_PINNED_ARCHIVE_SOURCES):
            raise _Refusal("ABSTAIN_C_LEGACY_DEPENDENCIES", "producer source map differs from the pinned archived implementation")
        if protocol_sources != _PINNED_ARCHIVE_SOURCES:
            raise _Refusal("ABSTAIN_C_LEGACY_DEPENDENCIES", "producer code hashes do not match the pinned archived implementation")
        for name in _DEPENDENCIES:
            if protocol_sources.get(name) != dependencies[name]:
                raise _Refusal("ABSTAIN_C_LEGACY_DEPENDENCIES", f"protocol and provenance source hashes differ for {name}")
        git = provenance.get("git")
        if (not isinstance(git, dict) or not isinstance(git.get("dirty"), bool)
                or not isinstance(git.get("commit"), str)
                or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", git["commit"]) is None):
            raise _Refusal("ABSTAIN_C_LEGACY_PROVENANCE", "producer git state is absent or malformed")
        if git.get("dirty") is not False:
            raise _Refusal("ABSTAIN_C_LEGACY_PROVENANCE", "producer working tree was dirty")

        if geometry != [16, 16, 16, 16] or protocol.get("beta") != 6.0:
            raise _Refusal("ABSTAIN_C_LEGACY_PROTOCOL", "geometry or beta differs from the selected Wilson ensemble")
        selected = selected_protocol()
        common = selected["ensemble_design"]
        c_lane = selected["finite_T_Creutz"]
        if protocol.get("action") != common["action"] or hmc_manifest.get("action") != common["action"]:
            raise _Refusal("ABSTAIN_C_LEGACY_PROTOCOL", "producer and HMC manifest must explicitly identify the selected Wilson action")
        boundary_conditions = _canonical_boundary_conditions(protocol.get("boundary_conditions"))
        if not _axis_matches(protocol.get("r_values"), c_reader.R_VALUES):
            raise _Refusal("ABSTAIN_C_LEGACY_STENCIL", "producer R axis does not contain the selected C stencil")
        if not _axis_matches(protocol.get("t_values"), c_reader.T_VALUES):
            raise _Refusal("ABSTAIN_C_LEGACY_STENCIL", "producer T axis does not contain the selected C stencil")

        primary_kernel_id = c_lane["kernel_id"]
        kernel_ids, kernel_parameters, kernel_index = _canonical_kernel_descriptors(
            protocol, primary_kernel_id,
        )

        validate_npz_payload(payloads["producer_npz"])
        try:
            with np.load(BytesIO(payloads["producer_npz"]), allow_pickle=False) as archive:
                if set(archive.files) != _OLD_NPZ_KEYS:
                    raise _Refusal("ABSTAIN_C_LEGACY_NPZ", "legacy NPZ keys differ from ids/loops/protocol_hash/generation")
                ids_raw = np.array(archive["ids"], copy=True)
                loops_raw = np.array(archive["loops"], copy=True)
                npz_protocol_hash = _npz_text(archive, "protocol_hash")
                npz_generation = _npz_text(archive, "generation")
        except _Refusal:
            raise
        except (OSError, ValueError, KeyError, EOFError) as exc:
            raise _Refusal("ABSTAIN_C_LEGACY_NPZ", "legacy NPZ payload cannot be parsed safely") from exc

        if ids_raw.ndim != 1 or ids_raw.dtype.kind not in "iu":
            raise _Refusal("ABSTAIN_C_LEGACY_IDS", "legacy configuration IDs must be a one-dimensional integer array")
        ids = [int(value) for value in ids_raw.tolist()]
        first, last = split_range
        if len(ids) < 2 or ids != list(range(first, first + len(ids))) or ids[-1] > last:
            raise _Refusal("ABSTAIN_C_LEGACY_IDS", "legacy IDs must be an increasing prefix of the authorized calibration range")
        configuration_denominator = {
            "unit": "producer row / paired saved gauge configuration",
            "planned": len(ids), "valid": 0, "failed": 0, "abstained": len(ids),
        }
        if npz_protocol_hash != fingerprint(protocol):
            raise _Refusal("ABSTAIN_C_LEGACY_IDENTITY", "NPZ protocol hash differs from captured provenance")
        if npz_generation != commit["generation"]:
            raise _Refusal("ABSTAIN_C_LEGACY_IDENTITY", "NPZ generation differs from the transaction marker")
        generation_spec = {
            "split": split,
            "ids": ids,
            "protocol_hash": fingerprint(protocol),
            "kernel_sha256": kernel_sha,
            "dependencies": dependencies,
        }
        if fingerprint(generation_spec) != commit["generation"]:
            raise _Refusal("ABSTAIN_C_LEGACY_IDENTITY", "generation does not match the archived measurement identity rule")

        loops = as_real_float_array(loops_raw, name="legacy loops")
        expected_shape = (len(ids), len(kernel_ids), len(c_reader.R_VALUES), len(c_reader.T_VALUES))
        if loops.shape != expected_shape or not np.isfinite(loops).all():
            raise _Refusal("ABSTAIN_C_LEGACY_LOOPS", "legacy loop tensor has the wrong shape or non-finite values")
        loops_digest = _sha256(np.ascontiguousarray(loops_raw).tobytes())
        if commit.get("loops_sha256") != loops_digest:
            raise _Refusal("ABSTAIN_C_LEGACY_COMMIT", "commit marker does not bind the legacy loop tensor")

        records = provenance.get("records")
        manifest_rows = hmc_manifest.get("configs")
        if not isinstance(records, list) or len(records) != len(ids):
            raise _Refusal("ABSTAIN_C_LEGACY_PROVENANCE", "producer records do not close the configuration denominator")
        if not isinstance(manifest_rows, list):
            raise _Refusal("ABSTAIN_C_LEGACY_INPUT_MANIFEST", "HMC manifest has no configuration table")
        manifest_by_id: dict[int, dict[str, Any]] = {}
        for row in manifest_rows:
            if not isinstance(row, dict) or type(row.get("id")) is not int or row["id"] in manifest_by_id:
                raise _Refusal("ABSTAIN_C_LEGACY_INPUT_MANIFEST", "HMC manifest has invalid or duplicate IDs")
            manifest_by_id[row["id"]] = row
        for index, (cfg, record) in enumerate(zip(ids, records, strict=True)):
            try:
                if not isinstance(record, dict) or record.get("id") != cfg:
                    raise _Refusal("ABSTAIN_C_LEGACY_PROVENANCE", "producer record order differs from the NPZ IDs")
                manifest_row = manifest_by_id.get(cfg)
                if manifest_row is None:
                    raise _Refusal("ABSTAIN_C_LEGACY_INPUT_MANIFEST", f"HMC manifest has no row for configuration {cfg}")
                manifest_gauge_sha = _require_sha(
                    manifest_row.get("sha256"), f"HMC manifest gauge SHA for {cfg}",
                )
                if (type(manifest_row.get("trajectory")) is not int
                        or type(record.get("hmc_trajectory")) is not int
                        or record["hmc_trajectory"] != manifest_row["trajectory"]
                        or record.get("gauge_sha256") != manifest_gauge_sha):
                    raise _Refusal("ABSTAIN_C_LEGACY_PROVENANCE", f"trajectory/raw-gauge claims disagree for configuration {cfg}")
                row_hash = _require_sha(record.get("loops_sha256"), f"producer loop hash for {cfg}")
                if row_hash != _sha256(np.ascontiguousarray(loops_raw[index:index + 1]).tobytes()):
                    raise _Refusal("ABSTAIN_C_LEGACY_PROVENANCE", f"producer loop hash disagrees for configuration {cfg}")
            except (ValueError, TypeError, KeyError) as exc:
                configuration_denominator["failed"] += 1
                configuration_denominator["abstained"] -= 1
                if isinstance(exc, _Refusal):
                    raise
                raise _Refusal(
                    "ABSTAIN_C_LEGACY_PROVENANCE",
                    f"configuration {cfg} failed the HMC/provenance join",
                ) from exc
            configuration_denominator["valid"] += 1
            configuration_denominator["abstained"] -= 1

        manifest = {
            "schema": c_reader.MANIFEST_SCHEMA,
            "split_id": split,
            "config_ids": ids,
            "geometry": common["geometry"],
            "action": common["action"],
            "boundary_conditions": boundary_conditions,
            "r_values": list(c_reader.R_VALUES),
            "t_values": list(c_reader.T_VALUES),
            "kernel_ids": kernel_ids,
            "primary_kernel_id": primary_kernel_id,
            "kernel_parameters": kernel_parameters,
        }
        manifest_bytes = json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        measurement_stream = BytesIO()
        np.savez_compressed(
            measurement_stream,
            schema=np.asarray(c_reader.MEASUREMENT_SCHEMA),
            split_id=np.asarray(split),
            config_ids=np.asarray(ids, dtype=np.int64),
            kernel_ids=np.asarray(kernel_ids, dtype=np.str_),
            r_values=np.asarray(c_reader.R_VALUES, dtype=np.int64),
            t_values=np.asarray(c_reader.T_VALUES, dtype=np.int64),
            manifest_sha256=np.asarray(_sha256(manifest_bytes)),
            loops=np.asarray(loops, dtype=np.float64),
        )
        measurement_bytes = measurement_stream.getvalue()
        canonical_hashes = {
            "manifest": _sha256(manifest_bytes),
            "loop_measurements": _sha256(measurement_bytes),
        }
        # Reuse the exact selected Creutz implementation to create the saved
        # summary that the same-byte current reader will independently recompute.
        from .c_lane_open_development_diagnostic import analyze as analyze_creutz

        seed_base = c_lane["bootstrap_seed_base"]
        analyzed = analyze_creutz(loops[:, kernel_index, :, :], seed_base=seed_base)
        _means, _point, _point_valid, rows, _draws, hull_low, hull_high, halfwidth, _empirical_pass = analyzed
        planned_draws = len(rows) * c_reader.N_DRAWS
        invalid_draws = sum(row["invalid_draws"] for row in rows)
        summary = {
            "schema": c_reader.SUMMARY_SCHEMA,
            "units": "dimensionless",
            "hull_lower": None if hull_low is None else float(hull_low),
            "hull_upper": None if hull_high is None else float(hull_high),
            "halfwidth": None if halfwidth is None else float(halfwidth),
            "invalid_draws": invalid_draws,
            "planned_draws": planned_draws,
            "resampling_protocol": {
                "method": "paired circular moving-block bootstrap",
                "block_lengths": list(c_reader.BLOCK_LENGTHS),
                "draws_per_length": c_reader.N_DRAWS,
                "alpha_lane": c_reader.ALPHA_LANE,
                "invalid_fraction_max": c_reader.INVALID_LIMIT,
                "precision_halfwidth_target": c_reader.HALFWIDTH_TARGET,
                "seed_base": seed_base,
                "seed_rule": "seed = seed_base + block_length",
                "interval_level": 0.975,
            },
        }
        summary_bytes = json.dumps(
            summary, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        canonical_payloads = {
            "manifest": manifest_bytes,
            "loop_measurements": measurement_bytes,
            "interval_summary": summary_bytes,
        }
        canonical_snapshot = _memory_snapshot(canonical_payloads)
        canonical_hashes["interval_summary"] = _sha256(summary_bytes)
        reader_result = c_reader.read_creutz_calibration_snapshot(
            canonical_snapshot,
            split_id=split,
            required_source_hashes=canonical_hashes,
        )
        if reader_result.get("status") != "ARRAY_REDUCED_NOT_CERTIFIED":
            raise _Refusal(
                "ABSTAIN_C_LEGACY_READER",
                f"derived bytes did not pass the current C reader: {reader_result.get('status')}",
            )

        return {
            "schema": "wilson-c-legacy-transaction-adapter-v1",
            "status": "LEGACY_TRANSACTION_ADAPTED_DIAGNOSTIC_ONLY",
            "split_id": split,
            "source_hashes": source_hashes,
            "canonical_source_hashes": canonical_hashes,
            "source_denominator": {
                "unit": "captured legacy input file", "planned": len(_ROLES),
                "valid": len(_ROLES), "failed": 0, "abstained": 0,
            },
            "configuration_denominator": reader_result["configuration_denominator"],
            "draw_denominator": reader_result["draw_denominator"],
            "current_reader": reader_result,
            "producer_hash_claims": {
                "dependency_hashes_internally_consistent": True,
                "source_bytes_for_producer_code_captured": False,
                "pinned_archive_commit": _PINNED_ARCHIVE_COMMIT,
                "pinned_archive_source_hashes": dict(_PINNED_ARCHIVE_SOURCES),
                "producer_git_commit": git.get("commit"),
                "producer_worktree_dirty": git["dirty"],
            },
            "text_flow": [
                "four captured bytes: legacy NPZ + provenance + commit marker + HMC input manifest",
                "explicit calibration opt-in + transaction/generation/dependency/hash checks",
                "join config IDs, trajectories, raw-gauge hashes, and per-config loop hashes",
                "reject wrong geometry, boundary, kernel, R/T stencil, or split before conversion",
                "derive canonical manifest and loop NPZ from the immutable bytes",
                "recompute the selected C interval and pass derived bytes to the current reader",
                "return diagnostic-only; the evidence gate, receipts, review, and authorization remain separate",
            ],
            "limitations": [
                "a caller-supplied SHA map proves byte consistency, not hash authority or producer identity",
                "producer dependency hashes are cross-checked as claims; producer code bytes are not part of this transaction",
                "does not read raw gauge files or validate loop contraction from gauge links",
                "does not establish block-bootstrap coverage, independence, or a physical Creutz result",
                "does not supply dependence, coverage, review, or data-authorization receipts",
                "does not estimate the separate F-lane ground-state force",
            ],
            "certified": False,
            "physical_score": None,
            "canonical_snapshot": canonical_snapshot,
        }
    except _Refusal as exc:
        details = {}
        if configuration_denominator is not None:
            details["configuration_denominator"] = configuration_denominator
        if reader_result is not None:
            details["current_reader"] = reader_result
            if "draw_denominator" in reader_result:
                details["draw_denominator"] = reader_result["draw_denominator"]
        return _abort(exc.status, str(exc), exc.denominator or denominator, details)
    except (KeyError, TypeError, ValueError, OverflowError, FloatingPointError) as exc:
        details = {}
        if configuration_denominator is not None:
            details["configuration_denominator"] = configuration_denominator
        if reader_result is not None:
            details["current_reader"] = reader_result
            if "draw_denominator" in reader_result:
                details["draw_denominator"] = reader_result["draw_denominator"]
        return _abort(
            "ABSTAIN_C_LEGACY_INVALID",
            f"legacy transaction failed closed: {type(exc).__name__}",
            denominator,
            details,
        )
