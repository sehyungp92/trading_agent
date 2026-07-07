"""Path helpers for the control-plane workspace."""

from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    package_dir = Path(__file__).resolve().parent
    if package_dir.parent.name == "src":
        return package_dir.parent.parent
    return package_dir


def monorepo_root() -> Path:
    root = package_root()
    return root.parent.parent if root.parent.name == "packages" else root.parent


def data_root() -> Path:
    return package_root() / "data"


def memory_root() -> Path:
    return package_root() / "memory"


def docs_root() -> Path:
    return package_root() / "docs"
