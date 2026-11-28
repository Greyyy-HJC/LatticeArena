import hashlib
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from core.source_integrity import (
    capture_source_snapshot,
    verify_source_bytes,
    verify_source_snapshot,
)
from core.npz_safety import validate_npz_payload


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_verify_source_bytes_hashes_only_named_files_under_root(tmp_path):
    (tmp_path / "manifest.json").write_bytes(b'{"id": 1}')
    (tmp_path / "correlator.bin").write_bytes(b"synthetic-correlator")

    receipt = verify_source_bytes(
        tmp_path,
        {"manifest": _sha(b'{"id": 1}'), "correlator": _sha(b"synthetic-correlator")},
        {"manifest": "manifest.json", "correlator": "correlator.bin"},
    )

    assert receipt["status"] == "verified"
    assert receipt["denominator"] == {
        "unit": "source_file",
        "planned": 2,
        "valid": 2,
        "failed": 0,
        "abstained": 0,
        "not_run": 0,
    }
    assert receipt["entries"]["manifest"]["observed_sha256"] == _sha(b'{"id": 1}')


def test_hash_mismatch_is_failed_and_does_not_claim_verified(tmp_path):
    (tmp_path / "data.bin").write_bytes(b"observed bytes")

    receipt = verify_source_bytes(
        tmp_path,
        {"data": _sha(b"expected bytes")},
        {"data": "data.bin"},
    )

    assert receipt["status"] == "incomplete"
    assert receipt["denominator"]["failed"] == 1
    assert receipt["entries"]["data"]["reason"] == "sha256_mismatch"
    assert receipt["entries"]["data"]["observed_sha256"] == _sha(b"observed bytes")


def test_missing_path_mapping_and_missing_file_are_abstentions(tmp_path):
    receipt = verify_source_bytes(
        tmp_path,
        {"unmapped": _sha(b"a"), "missing": _sha(b"b")},
        {"missing": "not-present.bin"},
    )

    assert receipt["status"] == "incomplete"
    assert receipt["denominator"]["planned"] == 2
    assert receipt["denominator"]["abstained"] == 2
    assert receipt["denominator"]["not_run"] == 0
    assert receipt["entries"]["unmapped"]["reason"] == "source_path_missing_or_invalid"
    assert receipt["entries"]["missing"]["reason"] == "source_file_not_found"


def test_path_traversal_is_failed_without_hashing_outside_file(tmp_path, monkeypatch):
    outside = tmp_path.parent / "outside-source.bin"
    outside.write_bytes(b"must not be opened")
    called = False

    def forbidden_hash(path):
        nonlocal called
        called = True
        raise AssertionError(f"unexpected hash read: {path}")

    monkeypatch.setattr("core.source_integrity._sha256_file", forbidden_hash)
    receipt = verify_source_bytes(
        tmp_path,
        {"data": _sha(b"must not be opened")},
        {"data": "../outside-source.bin"},
    )

    assert not called
    assert receipt["denominator"]["failed"] == 1
    assert receipt["entries"]["data"]["observed_sha256"] is None
    assert receipt["entries"]["data"]["reason"] == "path_must_remain_relative_to_root"


def test_symlink_escape_is_failed_before_reading_target(tmp_path, monkeypatch):
    outside = tmp_path.parent / "outside-link-target.bin"
    outside.write_bytes(b"must not be opened")
    link = tmp_path / "external.bin"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    called = False

    def forbidden_hash(path):
        nonlocal called
        called = True
        raise AssertionError(f"unexpected hash read: {path}")

    monkeypatch.setattr("core.source_integrity._sha256_file", forbidden_hash)
    receipt = verify_source_bytes(
        tmp_path,
        {"data": _sha(b"must not be opened")},
        {"data": "external.bin"},
    )

    assert not called
    assert receipt["denominator"]["failed"] == 1
    assert receipt["entries"]["data"]["reason"] == "resolved_source_escapes_root"


def test_unexpected_map_key_and_invalid_digest_fail_before_any_read(tmp_path, monkeypatch):
    called = False

    def forbidden_hash(path):
        nonlocal called
        called = True
        raise AssertionError(f"unexpected hash read: {path}")

    monkeypatch.setattr("core.source_integrity._sha256_file", forbidden_hash)
    with pytest.raises(ValueError, match="unexpected source names"):
        verify_source_bytes(
            tmp_path,
            {"expected": _sha(b"bytes")},
            {"unexpected": "outside.bin"},
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        verify_source_bytes(tmp_path, {"data": "bad"}, {})
    assert not called


def test_source_snapshot_reuses_the_exact_bytes_after_path_changes(tmp_path):
    original = b'{"measurement": [1, 2, 3]}'
    (tmp_path / "measurement.json").write_bytes(original)
    expected = {"measurement": _sha(original)}
    snapshot = capture_source_snapshot(
        tmp_path, expected, {"measurement": "measurement.json"},
    )

    # A task reader consumes the retained bytes; a later path edit cannot
    # change either the value it parsed or the bytes presented to the gate.
    read_by_task = snapshot.read_bytes("measurement")
    (tmp_path / "measurement.json").write_bytes(b"replaced after capture")
    verification = verify_source_snapshot(snapshot, expected)

    assert read_by_task == original
    assert snapshot.read_bytes("measurement") == original
    assert verification["status"] == "verified"
    assert verification["read_mode"] == "immutable_snapshot"
    assert verification["denominator"]["valid"] == 1
    with pytest.raises(TypeError):
        snapshot.source_bytes["measurement"] = b"changed"


def test_source_snapshot_fails_closed_on_size_limit_and_digest_mismatch(tmp_path):
    payload = b"too much for this snapshot"
    (tmp_path / "large.bin").write_bytes(payload)
    oversized = capture_source_snapshot(
        tmp_path, {"large": _sha(payload)}, {"large": "large.bin"},
        max_total_bytes=len(payload) - 1,
    )
    assert verify_source_snapshot(oversized, {"large": _sha(payload)})[
        "denominator"]["failed"] == 1
    with pytest.raises(ValueError, match="size_limit"):
        oversized.read_bytes("large")

    mismatched = capture_source_snapshot(
        tmp_path, {"large": _sha(b"different")}, {"large": "large.bin"},
    )
    result = verify_source_snapshot(mismatched, {"large": _sha(b"different")})
    assert result["denominator"]["failed"] == 1
    assert result["entries"]["large"]["reason"] == "sha256_mismatch"


def test_source_snapshot_stops_one_byte_past_aggregate_budget(tmp_path, monkeypatch):
    from pathlib import Path
    payload = b"oversized payload"
    source = tmp_path / "large.bin"
    source.write_bytes(payload)
    read_sizes = []
    original_open = Path.open

    class ReadSpy:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, size=-1):
            read_sizes.append(size)
            return self.stream.read(size)

    def tracked_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return ReadSpy(stream) if path == source else stream

    monkeypatch.setattr(Path, "open", tracked_open)
    snapshot = capture_source_snapshot(
        tmp_path, {"large": _sha(payload)}, {"large": "large.bin"},
        max_total_bytes=7,
    )

    assert read_sizes == [8]
    assert snapshot.capture_entries["large"]["reason"] == "source_snapshot_size_limit_exceeded"


def test_source_snapshot_rejects_unmapped_paths_and_root_escapes(tmp_path):
    outside = tmp_path.parent / "snapshot-outside.bin"
    outside.write_bytes(b"outside")
    traversal = capture_source_snapshot(
        tmp_path, {"outside": _sha(b"outside")}, {"outside": "../snapshot-outside.bin"},
    )
    assert verify_source_snapshot(traversal, {"outside": _sha(b"outside")})[
        "entries"]["outside"]["reason"] == "path_must_remain_relative_to_root"

    (tmp_path / "expected.bin").write_bytes(b"inside")
    try:
        (tmp_path / "escape.bin").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    escaped = capture_source_snapshot(
        tmp_path, {"escape": _sha(b"outside")}, {"escape": "escape.bin"},
    )
    assert verify_source_snapshot(escaped, {"escape": _sha(b"outside")})[
        "entries"]["escape"]["reason"] == "resolved_source_escapes_root"

    with pytest.raises(ValueError, match="unexpected source names"):
        capture_source_snapshot(
            tmp_path, {"expected": _sha(b"inside")},
            {"unexpected": "expected.bin"},
        )


def test_npz_preflight_accepts_small_numeric_archive_and_reports_expansion():
    stream = BytesIO()
    import numpy as np
    np.savez_compressed(stream, ids=np.arange(4096), values=np.zeros((4096, 3)))

    result = validate_npz_payload(stream.getvalue())

    assert result["member_count"] == 2
    assert result["uncompressed_bytes"] > len(stream.getvalue())


def test_npz_preflight_rejects_compressed_expansion_before_numpy_load():
    stream = BytesIO()
    import numpy as np
    np.savez_compressed(stream, values=np.zeros(4096, dtype=np.float64))

    with pytest.raises(ValueError, match="uncompressed size limit"):
        validate_npz_payload(stream.getvalue(), max_uncompressed_bytes=1024)


def test_npz_preflight_rejects_header_claiming_more_data_than_archive_contains():
    header = BytesIO()
    import numpy as np
    np.lib.format.write_array_header_1_0(
        header,
        {"descr": np.dtype("<f8").str, "fortran_order": False,
         "shape": (10**8,)},
    )
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("oversized.npy", header.getvalue() + b"\\x00")

    with pytest.raises(ValueError, match="size mismatch"):
        validate_npz_payload(stream.getvalue())


def test_npz_preflight_rejects_object_arrays_and_duplicate_members():
    import numpy as np
    objects = BytesIO()
    np.savez(objects, values=np.asarray([{"not": "allowed"}], dtype=object))
    with pytest.raises(ValueError, match="object arrays"):
        validate_npz_payload(objects.getvalue())

    duplicate = BytesIO()
    with pytest.warns(UserWarning, match="Duplicate name"):
        with ZipFile(duplicate, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("same.npy", b"first")
            archive.writestr("same.npy", b"second")
    with pytest.raises(ValueError, match="duplicate member"):
        validate_npz_payload(duplicate.getvalue())


def _mutate_npz_member(payload: bytes, *, compression: int | None = None,
                       encrypted: bool = False) -> bytes:
    import struct
    data = bytearray(payload)
    local = data.find(bytes.fromhex("504b0304"))
    central = data.find(bytes.fromhex("504b0102"))
    assert local >= 0 and central >= 0
    if compression is not None:
        struct.pack_into("<H", data, local + 8, compression)
        struct.pack_into("<H", data, central + 10, compression)
    if encrypted:
        for offset in (local + 6, central + 8):
            flags = struct.unpack_from("<H", data, offset)[0]
            struct.pack_into("<H", data, offset, flags | 1)
    return bytes(data)


@pytest.mark.parametrize("mode", ["unsupported_compression", "encrypted"])
def test_npz_preflight_normalizes_zip_member_errors_to_value_error(mode):
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("values.npy", b"not actually an NPY member")
    payload = stream.getvalue()
    if mode == "unsupported_compression":
        payload = _mutate_npz_member(payload, compression=99)
    else:
        payload = _mutate_npz_member(payload, encrypted=True)

    with pytest.raises(ValueError, match="invalid NPY member 'values.npy'"):
        validate_npz_payload(payload)


def test_npz_preflight_normalizes_corrupt_deflate_stream_to_value_error():
    import struct
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("values.npy", b"0" * 10000)
    payload = bytearray(stream.getvalue())
    local = payload.find(bytes.fromhex("504b0304"))
    name_size = struct.unpack_from("<H", payload, local + 26)[0]
    extra_size = struct.unpack_from("<H", payload, local + 28)[0]
    compressed_data = local + 30 + name_size + extra_size
    payload[compressed_data] ^= 0xFF

    with pytest.raises(ValueError, match="invalid NPY member 'values.npy'"):
        validate_npz_payload(bytes(payload))
