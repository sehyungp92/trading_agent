"""Small shared helpers for JSONL append stores."""

from __future__ import annotations

import json
import threading
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from filelock import FileLock


_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[Path, threading.Lock] = {}


def _lock_for(path: Path) -> threading.Lock:
    key = path.expanduser().resolve(strict=False)
    with _LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PATH_LOCKS[key] = lock
        return lock


def append_jsonl(path: Path, records: Iterable[dict]) -> None:
    """Append JSON-serializable records under a process-wide per-path lock."""
    path = Path(path)
    records = list(records)
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_file_lock(path), _lock_for(path), open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")


@contextmanager
def jsonl_file_lock(path: Path) -> Iterator[None]:
    """Cross-instance/process lock for JSONL lifecycle stores."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        yield


def read_jsonl_tail(
    path: Path,
    *,
    max_records: int = 500,
    chunk_size: int = 64 * 1024,
) -> list[dict]:
    """Read at most the last ``max_records`` JSON objects without scanning the file."""
    path = Path(path)
    if not path.exists() or max_records <= 0:
        return []
    chunks: deque[bytes] = deque()
    newline_count = 0
    with jsonl_file_lock(path), path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        while position > 0 and newline_count <= max_records:
            read_size = min(chunk_size, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            chunks.appendleft(chunk)
            newline_count += chunk.count(b"\n")

    raw_lines = b"".join(chunks).splitlines()
    lines = [
        line.decode("utf-8")
        for line in raw_lines[-max_records:]
        if line.strip()
    ]
    records: list[dict] = []
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            records.append(entry)
    return records


def write_json_projection(path: Path, *, key_field: str, records: Iterable[dict]) -> None:
    """Persist a compact keyed projection next to a JSONL lifecycle file."""
    projection_path = Path(str(path) + ".index.json")
    projection: dict[str, dict] = {}
    for record in records:
        key = record.get(key_field)
        if key:
            projection[str(key)] = record
    projection_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = projection_path.with_suffix(projection_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(projection, indent=2, default=str), encoding="utf-8")
    tmp_path.replace(projection_path)


def read_json_projection(path: Path) -> dict[str, dict]:
    projection_path = Path(str(path) + ".index.json")
    if not projection_path.exists():
        return {}
    try:
        data = json.loads(projection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): value for key, value in data.items() if isinstance(value, dict)}
