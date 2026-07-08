"""Workspace entrypoint wrappers for the legacy IBKR runtime CLI."""

from __future__ import annotations

from apps.runtime.cli import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
