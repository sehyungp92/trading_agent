"""Workspace entrypoint wrapper for the OLR/KALCB runtime-session operator CLI."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Sequence


def _load_legacy_module() -> ModuleType:
    script_name = Path("scripts") / "run_olr_kalcb_runtime_session.py"
    candidates = [
        Path(os.environ["K_STOCK_TRADER_ROOT"]) / script_name
        for key in ("K_STOCK_TRADER_ROOT",)
        if key in os.environ
    ]
    candidates.extend(
        [
            Path.cwd() / script_name,
            Path(__file__).resolve().parents[2] / script_name,
        ]
    )
    script = next((path for path in candidates if path.is_file()), candidates[0])
    spec = importlib.util.spec_from_file_location("_k_stock_legacy_runtime_session", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runtime-session script: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: Sequence[str] | None = None) -> int:
    return int(_load_legacy_module().main(argv))


__all__ = ["main"]
