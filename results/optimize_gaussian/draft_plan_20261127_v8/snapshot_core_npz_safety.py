"""Bounded preflight for NPZ payloads before NumPy allocates their arrays.

Source snapshots cap the compressed file bytes.  NPZ is a ZIP container, so a
small captured file can otherwise expand into much larger arrays.  This helper
checks the central-directory sizes and each NPY header before a task reader
calls ``numpy.load``.  It does not establish data provenance or scientific
validity.
"""
from __future__ import annotations

from io import BytesIO
import math
from zipfile import BadZipFile, ZipFile

import numpy as np


DEFAULT_MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_MAX_MEMBERS = 128
_MAX_HEADER_BYTES = 10_000


def validate_npz_payload(
    payload: bytes,
    *,
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
) -> dict[str, int]:
    """Reject malformed or over-budget NPZ bytes without loading array data.

    The limit covers total declared NPY member sizes, including their headers.
    Duplicate or nested member names, unsupported NPY versions, object arrays,
    inconsistent NPY headers, and oversized archives fail closed.
    """
    if not isinstance(payload, bytes):
        raise ValueError("NPZ payload must be immutable bytes")
    if (isinstance(max_uncompressed_bytes, bool)
            or not isinstance(max_uncompressed_bytes, int)
            or max_uncompressed_bytes < 0):
        raise ValueError("max_uncompressed_bytes must be a nonnegative integer")

    try:
        archive = ZipFile(BytesIO(payload), "r")
    except (BadZipFile, OSError, ValueError) as exc:
        raise ValueError("NPZ payload is not a readable ZIP archive") from exc

    with archive:
        members = archive.infolist()
        if not members or len(members) > _MAX_MEMBERS:
            raise ValueError(f"NPZ member count must be between 1 and {_MAX_MEMBERS}")
        names = [member.filename for member in members]
        if len(set(names)) != len(names):
            raise ValueError("NPZ archive contains duplicate member names")

        total_size = 0
        for member in members:
            name = member.filename
            if (member.is_dir() or "/" in name or "\\" in name
                    or not name.endswith(".npy") or name == ".npy"):
                raise ValueError("NPZ members must be flat .npy files")
            total_size += member.file_size
            if total_size > max_uncompressed_bytes:
                raise ValueError("NPZ uncompressed size limit exceeded")

            try:
                with archive.open(member, "r") as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(
                            stream, max_header_size=_MAX_HEADER_BYTES
                        )
                    elif version in ((2, 0), (3, 0)):
                        shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(
                            stream, max_header_size=_MAX_HEADER_BYTES
                        )
                    else:
                        raise ValueError(f"unsupported NPY version {version!r}")
                    if dtype.hasobject:
                        raise ValueError("object arrays are not allowed in NPZ inputs")
                    element_count = math.prod(shape)
                    expected_size = stream.tell() + element_count * dtype.itemsize
                    if expected_size != member.file_size:
                        raise ValueError(f"NPY header size mismatch in {name!r}")
            except (BadZipFile, EOFError, OSError, TypeError, OverflowError) as exc:
                raise ValueError(f"invalid NPY member {name!r}") from exc

    return {"member_count": len(members), "uncompressed_bytes": total_size}
