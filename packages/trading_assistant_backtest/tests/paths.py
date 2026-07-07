from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_ROOT = PACKAGE_ROOT.parent.parent if PACKAGE_ROOT.parent.name == "packages" else PACKAGE_ROOT.parent


def package_workspace(name: str) -> Path:
    final_path = MONOREPO_ROOT / "packages" / name
    if final_path.exists():
        return final_path
    return MONOREPO_ROOT / name
