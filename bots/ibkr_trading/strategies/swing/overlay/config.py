"""Configuration for the idle-capital EMA crossover overlay."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OverlayConfig:
    """Live overlay config; structural params mirror backtest defaults."""

    enabled: bool = False
    symbols: list[str] = field(default_factory=lambda: ["QQQ", "GLD"])
    max_equity_pct: float = 0.85
    ema_fast: int = 13
    ema_slow: int = 48
    ema_overrides: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"QQQ": (10, 21), "GLD": (13, 21)},
    )
    weights: dict[str, float] | None = None  # None = equal-weight
    state_file: str = "overlay_state.json"
