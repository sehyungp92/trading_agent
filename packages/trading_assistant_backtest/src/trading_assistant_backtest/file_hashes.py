"""Stable file hashing for text artifacts across checkout policies."""

from __future__ import annotations

import hashlib
from pathlib import Path

TEXT_ARTIFACT_SUFFIXES = frozenset(
    {
        ".json",
        ".md",
        ".py",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)


def sha256_file(
    path: str | Path,
    *,
    missing_ok: bool = False,
    normalize_text: bool = True,
) -> str:
    """Hash a file while normalizing text artifact line endings to LF."""

    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        if missing_ok:
            return ""
        raise FileNotFoundError(resolved)

    data = resolved.read_bytes()
    if normalize_text and _is_text_artifact(resolved, data):
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _is_text_artifact(path: Path, data: bytes) -> bool:
    if path.suffix.lower() in TEXT_ARTIFACT_SUFFIXES:
        return True
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True
