"""Pytest root conftest -- redirect ``research.backtests.*`` -> ``backtests.*``.

Several test files import from ``research.backtests.*`` which was the old
package path before the repo consolidation.  The actual code now lives at
``backtests/``.  A custom meta-path finder transparently rewrites every
``research.backtests.X`` import to ``backtests.X`` so both ``from`` and
bare ``import`` forms work without touching the test files.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types

_PREFIX = "research.backtests"


class _ResearchFinder(importlib.abc.MetaPathFinder):
    """Intercept ``research`` and ``research.backtests.*`` imports."""

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname == "research":
            spec = importlib.machinery.ModuleSpec(
                "research", loader=_ResearchLoader(), is_package=True,
            )
            spec.submodule_search_locations = []
            return spec

        if fullname.startswith(_PREFIX):
            real_name = "backtests" + fullname[len(_PREFIX):]
            real_spec = importlib.util.find_spec(real_name)
            if real_spec is None:
                return None
            # Build a spec that re-uses the real module's loader
            spec = importlib.machinery.ModuleSpec(
                fullname,
                loader=_AliasLoader(real_name),
                is_package=real_spec.submodule_search_locations is not None,
            )
            if real_spec.submodule_search_locations is not None:
                spec.submodule_search_locations = list(
                    real_spec.submodule_search_locations,
                )
            return spec

        return None


class _ResearchLoader(importlib.abc.Loader):
    """Loader for the synthetic ``research`` top-level package."""

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType:
        mod = types.ModuleType(spec.name)
        mod.__path__ = []  # type: ignore[attr-defined]
        mod.__package__ = spec.name
        mod.__spec__ = spec
        return mod

    def exec_module(self, module: types.ModuleType) -> None:
        pass


class _AliasLoader(importlib.abc.Loader):
    """Loader that imports the real ``backtests.*`` module and aliases it."""

    def __init__(self, real_name: str) -> None:
        self._real_name = real_name

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType | None:
        return None  # let exec_module handle it

    def exec_module(self, module: types.ModuleType) -> None:
        real_mod = importlib.import_module(self._real_name)
        # Copy all attributes from the real module
        module.__dict__.update(real_mod.__dict__)
        if hasattr(real_mod, "__path__"):
            module.__path__ = real_mod.__path__  # type: ignore[attr-defined]
        # Also register alias so subsequent lookups hit cache
        sys.modules[module.__name__] = real_mod
        # Wire parent attribute for bare ``import a.b.c`` traversal
        parent_name, _, child = module.__name__.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, child, real_mod)


sys.meta_path.insert(0, _ResearchFinder())

# ---------------------------------------------------------------------------
# Tests whose target modules were renamed/removed (can't even import)
# ---------------------------------------------------------------------------
import pathlib as _pathlib

collect_ignore = [
    str(_pathlib.Path(__file__).parent / "unit" / "test_alcb_p10plus_phase_plugin.py"),
]
