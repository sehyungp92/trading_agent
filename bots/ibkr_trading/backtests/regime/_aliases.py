"""Regime family import aliases for backtest compatibility."""
from __future__ import annotations

ALIASES: dict[str, str] = {
    "regime": "regime",
    "backtest": "backtests.regime",
}


def install() -> None:
    """Install the regime alias redirector."""
    import sys
    from pathlib import Path

    # Add project root to sys.path if not already present
    root = Path(__file__).resolve().parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
