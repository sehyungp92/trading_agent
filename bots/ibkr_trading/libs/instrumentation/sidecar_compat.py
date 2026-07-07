"""Compatibility helpers for legacy sidecar patch/import paths."""
from __future__ import annotations

import sys
import types
from typing import Any


def install_legacy_sidecar_alias(module_name: str) -> None:
    """Expose the current family sidecar at ``instrumentation.src.sidecar``.

    Older copied family tests and local scripts patched that path directly.
    Runtime imports should keep using the family-specific sidecar modules.
    """
    module = sys.modules[module_name]
    src_module = sys.modules.setdefault("instrumentation.src", types.ModuleType("instrumentation.src"))
    src_module.sidecar = module
    sys.modules["instrumentation.src.sidecar"] = module
    try:
        import instrumentation as instrumentation_pkg

        setattr(instrumentation_pkg, "src", src_module)
    except Exception:
        pass


def requests_client(default_requests: Any) -> Any:
    alias = sys.modules.get("instrumentation.src.sidecar")
    if alias is not None and hasattr(alias, "requests"):
        return getattr(alias, "requests")
    return default_requests
