"""Deterministic checksum helpers for files, JSON, and parquet slices."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def parquet_content_checksum(path: Path) -> str:
    """Hash parquet bytes plus schema metadata.

    The file bytes make normal corruption/mutation checks cheap. The explicit schema
    component keeps the contract tied to the logical table shape even when callers use the
    helper on in-memory test files with minimal parquet metadata.
    """

    parquet_path = Path(path)
    digest = hashlib.sha256()
    digest.update(sha256_file(parquet_path).encode("ascii"))
    try:
        schema = pq.read_schema(parquet_path)
        digest.update(str(schema).encode("utf-8"))
        metadata = schema.metadata or {}
        for key in sorted(metadata):
            digest.update(key)
            digest.update(metadata[key])
    except Exception:
        digest.update(b"<schema-unavailable>")
    return digest.hexdigest()


def checksum_paths(paths: Iterable[Path], *, root: Path | None = None) -> list[dict[str, Any]]:
    root = Path(root).resolve() if root else None
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(item) for item in paths):
        resolved = path.resolve()
        display = str(resolved.relative_to(root)) if root and resolved.is_relative_to(root) else str(path)
        rows.append(
            {
                "path": display.replace("\\", "/"),
                "bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return rows


def stable_row_hashes(frame: pd.DataFrame) -> pd.Series:
    """Return row hashes that are stable across platforms and parquet engines."""

    if frame.empty:
        return pd.Series([], dtype="object", index=frame.index)
    normalized = frame.copy()
    normalized.columns = [str(column) for column in normalized.columns]
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    row_hash = pd.util.hash_pandas_object(normalized.fillna("<NA>"), index=False)
    return row_hash.map(lambda value: hashlib.sha256(str(int(value)).encode("ascii")).hexdigest())
