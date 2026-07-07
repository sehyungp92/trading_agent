"""Tiny .env loader for local source-refresh credentials."""

from __future__ import annotations

import os
from pathlib import Path


def load_local_env(repo_root: Path) -> list[Path]:
    """Load repo-local .env files without overriding the process environment."""

    repo_root = Path(repo_root).resolve()
    loaded: list[Path] = []
    for path in (repo_root.parent / ".env", repo_root / ".env"):
        if not path.exists():
            continue
        _load_env_file(path)
        loaded.append(path)
    return loaded


def _load_env_file(path: Path) -> None:
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _clean_value(value.strip())


def _clean_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
