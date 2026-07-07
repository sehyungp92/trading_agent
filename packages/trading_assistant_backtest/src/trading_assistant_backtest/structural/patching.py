"""Patch artifact helpers."""

from __future__ import annotations

from pathlib import Path


def write_empty_patch(path: Path, label: str) -> Path:
    path.write_text(
        f"# No {label} patch generated for deterministic no-adoption run.\n", encoding="utf-8"
    )
    return path
