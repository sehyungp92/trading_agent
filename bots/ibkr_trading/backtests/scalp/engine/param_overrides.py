from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import Any


@contextmanager
def temporary_param_overrides(
    overrides: dict[str, Any],
    modules: Iterable[ModuleType],
) -> Iterator[None]:
    """Temporarily patch module-level strategy constants for optimizer runs."""
    patches: list[tuple[ModuleType, str, Any]] = []
    created: list[tuple[ModuleType, str]] = []
    if not overrides:
        yield
        return

    module_list = tuple(modules)
    for raw_key, value in overrides.items():
        key = raw_key.split(".", 1)[1] if raw_key.startswith("param_overrides.") else raw_key
        for module in module_list:
            if hasattr(module, key):
                patches.append((module, key, getattr(module, key)))
                setattr(module, key, value)
            elif module.__name__.endswith(".config"):
                created.append((module, key))
                setattr(module, key, value)
    try:
        yield
    finally:
        for module, key, original in reversed(patches):
            setattr(module, key, original)
        for module, key in reversed(created):
            try:
                delattr(module, key)
            except AttributeError:
                pass
