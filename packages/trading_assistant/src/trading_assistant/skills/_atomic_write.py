# skills/_atomic_write.py
"""Atomic JSONL rewrite — write-to-temp then os.replace() for crash safety."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def atomic_rewrite_jsonl(path: Path, records: list) -> None:
    """Atomically rewrite a JSONL file with the given records.

    Writes to a temporary file in the same directory, then uses os.replace()
    which is atomic on both POSIX and Windows NTFS.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = None  # os.fdopen takes ownership
            for rec in records:
                if hasattr(rec, "model_dump"):
                    data = rec.model_dump(mode="json")
                elif isinstance(rec, dict):
                    data = rec
                else:
                    data = rec
                f.write(json.dumps(data, default=str) + "\n")
        os.replace(tmp_path, path)
        tmp_path = None  # replaced successfully
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
