"""Verify explicitly declared source-file bytes against SHA-256 expectations.

Use ``capture_source_snapshot`` when a reader and its later evidence gate need
to consume the exact same byte sequence.

This proves only that the bytes read from the supplied paths match the supplied
digests. It does not establish who produced the files, whether the hashes came
from a trusted authority, whether samples are independent, or whether the
scientific claims are valid.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


SOURCE_VERIFICATION_SCHEMA = "latticearena-source-byte-verification-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _validate_expected_hashes(expected_hashes: object) -> dict[str, str]:
    if not isinstance(expected_hashes, dict) or not expected_hashes:
        raise ValueError("expected_hashes must be a nonempty mapping")
    normalized: dict[str, str] = {}
    for name, digest in expected_hashes.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source names must be nonempty strings")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"expected_hashes[{name!r}] must be a lowercase SHA-256 hash")
        normalized[name] = digest
    return normalized


def _safe_relative_path(value: object) -> tuple[Path | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, "source_path_missing_or_invalid"
    if "\\" in value:
        return None, "backslash_path_not_allowed"
    if "\x00" in value:
        return None, "nul_byte_in_path"
    relative = Path(value)
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        return None, "path_must_remain_relative_to_root"
    if not relative.parts or value.strip() in (".", "./"):
        return None, "path_must_name_a_file"
    return relative, None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceByteSnapshot:
    """Immutable bytes captured for a reader and a later evidence gate.

    Use ``read_bytes`` as the reader input, then pass this same snapshot to the
    evidence gate. It is intentionally limited to a caller-selected byte
    budget so large measurement arrays can use a future streaming snapshot
    implementation instead of being copied into process memory.
    """

    root: str
    expected_hashes: Mapping[str, str]
    source_paths: Mapping[str, str | None]
    source_bytes: Mapping[str, bytes]
    capture_entries: Mapping[str, Mapping[str, Any]]
    max_total_bytes: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected_hashes", MappingProxyType(dict(self.expected_hashes)))
        object.__setattr__(self, "source_paths", MappingProxyType(dict(self.source_paths)))
        object.__setattr__(
            self, "source_bytes",
            MappingProxyType({name: bytes(data) for name, data in self.source_bytes.items()}),
        )
        object.__setattr__(
            self, "capture_entries",
            MappingProxyType({
                name: MappingProxyType(dict(entry))
                for name, entry in self.capture_entries.items()
            }),
        )

    def read_bytes(self, source_name: str) -> bytes:
        """Return the exact captured bytes, and refuse incomplete sources."""
        entry = self.capture_entries.get(source_name)
        data = self.source_bytes.get(source_name)
        if entry is None or entry.get("status") != "valid" or data is None:
            reason = "source was not captured as valid"
            if entry is not None and entry.get("reason"):
                reason = str(entry["reason"])
            raise ValueError(f"{source_name!r}: {reason}")
        return data


def _source_verification(entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    valid = sum(entry["status"] == "valid" for entry in entries.values())
    failed = sum(entry["status"] == "failed" for entry in entries.values())
    abstained = sum(entry["status"] == "abstained" for entry in entries.values())
    planned = len(entries)
    return {
        "schema": SOURCE_VERIFICATION_SCHEMA,
        "status": "verified" if valid == planned else "incomplete",
        "read_mode": "immutable_snapshot",
        "denominator": {
            "unit": "source_file", "planned": planned, "valid": valid,
            "failed": failed, "abstained": abstained, "not_run": 0,
        },
        "entries": entries,
    }


def capture_source_snapshot(
    root: str | Path,
    expected_hashes: dict[str, str],
    source_paths: dict[str, str],
    *,
    max_total_bytes: int | None = 64 * 1024 * 1024,
) -> SourceByteSnapshot:
    """Read explicitly mapped files once and retain only hash-matching bytes.

    The returned snapshot is a suitable input source for a task reader and its
    later evidence gate. Missing files are abstentions; malformed paths, read
    errors, size-limit violations and digest mismatches are failures. The
    expected digests still need an independently trusted authority.
    """
    expected = _validate_expected_hashes(expected_hashes)
    if not isinstance(source_paths, dict):
        raise ValueError("source_paths must map expected source names to relative paths")
    unexpected = set(source_paths) - set(expected)
    if unexpected:
        names = ", ".join(sorted(str(item) for item in unexpected))
        raise ValueError(f"source_paths contains unexpected source names: {names}")
    if (max_total_bytes is not None
            and (isinstance(max_total_bytes, bool) or not isinstance(max_total_bytes, int)
                 or max_total_bytes < 0)):
        raise ValueError("max_total_bytes must be a nonnegative integer or None")

    root_path = Path(root)
    try:
        root_path = root_path.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise ValueError("root must name an existing directory") from exc
    if not root_path.is_dir():
        raise ValueError("root must name an existing directory")

    paths = {name: source_paths.get(name) for name in expected}
    entries: dict[str, dict[str, Any]] = {}
    captured: dict[str, bytes] = {}
    total_bytes = 0
    for name, expected_digest in expected.items():
        raw_path = paths[name]
        relative, path_error = _safe_relative_path(raw_path)
        if path_error is not None:
            status = "abstained" if raw_path is None else "failed"
            entries[name] = {
                "status": status, "path": raw_path,
                "expected_sha256": expected_digest, "observed_sha256": None,
                "reason": path_error,
            }
            continue

        candidate = root_path / relative
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            entries[name] = {
                "status": "abstained", "path": raw_path,
                "expected_sha256": expected_digest, "observed_sha256": None,
                "reason": "source_file_not_found",
            }
            continue
        except (OSError, ValueError) as exc:
            entries[name] = {
                "status": "failed", "path": raw_path,
                "expected_sha256": expected_digest, "observed_sha256": None,
                "reason": f"source_path_resolution_failed:{type(exc).__name__}",
            }
            continue

        if root_path not in resolved.parents:
            entries[name] = {
                "status": "failed", "path": raw_path,
                "expected_sha256": expected_digest, "observed_sha256": None,
                "reason": "resolved_source_escapes_root",
            }
            continue
        if not resolved.is_file():
            entries[name] = {
                "status": "failed", "path": raw_path,
                "expected_sha256": expected_digest, "observed_sha256": None,
                "reason": "source_is_not_a_regular_file",
            }
            continue

        try:
            with resolved.open("rb") as stream:
                chunks = []
                file_size = 0
                while True:
                    chunk = stream.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    file_size += len(chunk)
                    if (max_total_bytes is not None
                            and total_bytes + file_size > max_total_bytes):
                        chunks = []
                        break
                    chunks.append(chunk)
        except OSError as exc:
            entries[name] = {
                "status": "failed", "path": raw_path,
                "expected_sha256": expected_digest, "observed_sha256": None,
                "reason": f"source_read_failed:{type(exc).__name__}",
            }
            continue

        if max_total_bytes is not None and total_bytes + file_size > max_total_bytes:
            entries[name] = {
                "status": "failed", "path": raw_path,
                "expected_sha256": expected_digest, "observed_sha256": None,
                "reason": "source_snapshot_size_limit_exceeded",
            }
            continue
        data = b"".join(chunks)
        observed_digest = hashlib.sha256(data).hexdigest()
        if observed_digest != expected_digest:
            entries[name] = {
                "status": "failed", "path": raw_path,
                "expected_sha256": expected_digest, "observed_sha256": observed_digest,
                "reason": "sha256_mismatch",
            }
            continue
        total_bytes += len(data)
        captured[name] = data
        entries[name] = {
            "status": "valid", "path": raw_path,
            "expected_sha256": expected_digest, "observed_sha256": observed_digest,
            "reason": None,
        }

    frozen_entries = MappingProxyType({
        name: MappingProxyType(dict(entry)) for name, entry in entries.items()
    })
    return SourceByteSnapshot(
        root=str(root_path),
        expected_hashes=MappingProxyType(dict(expected)),
        source_paths=MappingProxyType(dict(paths)),
        source_bytes=MappingProxyType(dict(captured)),
        capture_entries=frozen_entries,
        max_total_bytes=max_total_bytes,
    )


def verify_source_snapshot(
    snapshot: SourceByteSnapshot,
    expected_hashes: dict[str, str],
) -> dict[str, Any]:
    """Recheck immutable captured bytes against the gate's required digests."""
    if not isinstance(snapshot, SourceByteSnapshot):
        raise ValueError("snapshot must be a SourceByteSnapshot")
    expected = _validate_expected_hashes(expected_hashes)
    entries: dict[str, dict[str, Any]] = {}
    for name, expected_digest in expected.items():
        captured = snapshot.capture_entries.get(name)
        data = snapshot.source_bytes.get(name)
        path = snapshot.source_paths.get(name)
        if captured is None:
            entries[name] = {
                "status": "abstained", "path": None,
                "expected_sha256": expected_digest, "observed_sha256": None,
                "reason": "source_not_in_snapshot",
            }
            continue
        if captured.get("status") != "valid" or data is None:
            entry = dict(captured)
            entry["expected_sha256"] = expected_digest
            entries[name] = entry
            continue
        observed_digest = hashlib.sha256(data).hexdigest()
        matched = observed_digest == expected_digest
        entries[name] = {
            "status": "valid" if matched else "failed", "path": path,
            "expected_sha256": expected_digest, "observed_sha256": observed_digest,
            "reason": None if matched else "sha256_mismatch",
        }
    return _source_verification(entries)


def verify_source_bytes(
    root: str | Path,
    expected_hashes: dict[str, str],
    source_paths: dict[str, str],
) -> dict[str, Any]:
    """Hash only the explicitly mapped files contained under ``root``.

    Each expected hash is one planned source. Missing mappings/files are
    abstentions; malformed paths, read errors and hash mismatches are failures.
    Unexpected mapping keys raise before any source file is opened so an
    unregistered path can never trigger an implicit read.
    """
    expected = _validate_expected_hashes(expected_hashes)
    if not isinstance(source_paths, dict):
        raise ValueError("source_paths must be a mapping of expected names to relative paths")
    unexpected = set(source_paths) - set(expected)
    if unexpected:
        names = ", ".join(sorted(str(item) for item in unexpected))
        raise ValueError(f"source_paths contains unexpected source names: {names}")

    root_path = Path(root)
    try:
        root_path = root_path.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise ValueError("root must name an existing directory") from exc
    if not root_path.is_dir():
        raise ValueError("root must name an existing directory")

    entries: dict[str, dict[str, Any]] = {}
    valid = failed = abstained = 0
    for name, expected_digest in expected.items():
        raw_path = source_paths.get(name)
        relative, path_error = _safe_relative_path(raw_path)
        if path_error is not None:
            status = "abstained" if raw_path is None else "failed"
            if status == "abstained":
                abstained += 1
            else:
                failed += 1
            entries[name] = {
                "status": status,
                "path": raw_path if isinstance(raw_path, str) else None,
                "expected_sha256": expected_digest,
                "observed_sha256": None,
                "reason": path_error,
            }
            continue

        candidate = root_path / relative
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            abstained += 1
            entries[name] = {
                "status": "abstained",
                "path": raw_path,
                "expected_sha256": expected_digest,
                "observed_sha256": None,
                "reason": "source_file_not_found",
            }
            continue
        except (OSError, ValueError) as exc:
            failed += 1
            entries[name] = {
                "status": "failed",
                "path": raw_path,
                "expected_sha256": expected_digest,
                "observed_sha256": None,
                "reason": f"source_path_resolution_failed:{type(exc).__name__}",
            }
            continue

        if root_path not in resolved.parents:
            failed += 1
            entries[name] = {
                "status": "failed",
                "path": raw_path,
                "expected_sha256": expected_digest,
                "observed_sha256": None,
                "reason": "resolved_source_escapes_root",
            }
            continue
        if not resolved.is_file():
            failed += 1
            entries[name] = {
                "status": "failed",
                "path": raw_path,
                "expected_sha256": expected_digest,
                "observed_sha256": None,
                "reason": "source_is_not_a_regular_file",
            }
            continue

        try:
            observed_digest = _sha256_file(resolved)
        except OSError as exc:
            failed += 1
            entries[name] = {
                "status": "failed",
                "path": raw_path,
                "expected_sha256": expected_digest,
                "observed_sha256": None,
                "reason": f"source_read_failed:{type(exc).__name__}",
            }
            continue

        if observed_digest == expected_digest:
            valid += 1
            entries[name] = {
                "status": "valid",
                "path": raw_path,
                "expected_sha256": expected_digest,
                "observed_sha256": observed_digest,
                "reason": None,
            }
        else:
            failed += 1
            entries[name] = {
                "status": "failed",
                "path": raw_path,
                "expected_sha256": expected_digest,
                "observed_sha256": observed_digest,
                "reason": "sha256_mismatch",
            }

    planned = len(expected)
    return {
        "schema": SOURCE_VERIFICATION_SCHEMA,
        "status": "verified" if valid == planned else "incomplete",
        "denominator": {
            "unit": "source_file",
            "planned": planned,
            "valid": valid,
            "failed": failed,
            "abstained": abstained,
            "not_run": 0,
        },
        "entries": entries,
    }
