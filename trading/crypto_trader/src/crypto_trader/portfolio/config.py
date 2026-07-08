"""Portfolio configuration — frozen dataclasses for multi-strategy coordination."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StrategyAllocation:
    """Per-strategy allocation and limits within a portfolio."""

    strategy_id: str
    enabled: bool = True
    base_risk_pct: float = 0.01
    max_concurrent: int = 3
    daily_stop_R: float = 3.0
    priority: int = 0  # lower = higher priority


@dataclass(frozen=True)
class PortfolioConfig:
    """Portfolio-level configuration for multi-strategy coordination.

    Frozen dataclass — replace atomically for regime changes.
    """

    initial_equity: float = 10_000.0
    strategies: tuple[StrategyAllocation, ...] = ()

    # Portfolio-level risk rules
    heat_cap_R: float = 6.0
    directional_cap_R: float = 4.0
    portfolio_daily_stop_R: float = 5.0
    max_total_positions: int = 9

    # Drawdown tiers: (dd_pct_threshold, size_multiplier)
    # Applied in order — first tier where dd >= threshold wins
    dd_tiers: tuple[tuple[float, float], ...] = (
        (0.08, 1.00),
        (0.12, 0.50),
        (0.15, 0.25),
        (1.00, 0.00),
    )

    # Symbol collision — critical since all 3 strategies trade BTC/ETH/SOL
    symbol_collision: str = "cap"  # "allow" | "block" | "cap"
    symbol_exposure_cap_R: float = 3.0
    terminal_accounting_mode: str = "terminal_mark"  # "terminal_mark" | "force_close"

    # Priority reservation
    priority_headroom_R: float = 0.0
    priority_reserve_threshold: int = 1  # priority >= this gets blocked when headroom binds

    def get_strategy(self, strategy_id: str) -> StrategyAllocation | None:
        """Look up allocation by strategy_id."""
        for s in self.strategies:
            if s.strategy_id == strategy_id:
                return s
        return None

    def priority_order(self) -> list[StrategyAllocation]:
        """Return strategies sorted by priority (lower = higher priority)."""
        return sorted(self.strategies, key=lambda s: s.priority)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "initial_equity": self.initial_equity,
            "strategies": [
                {
                    "strategy_id": s.strategy_id,
                    "enabled": s.enabled,
                    "base_risk_pct": s.base_risk_pct,
                    "max_concurrent": s.max_concurrent,
                    "daily_stop_R": s.daily_stop_R,
                    "priority": s.priority,
                }
                for s in self.strategies
            ],
            "heat_cap_R": self.heat_cap_R,
            "directional_cap_R": self.directional_cap_R,
            "portfolio_daily_stop_R": self.portfolio_daily_stop_R,
            "max_total_positions": self.max_total_positions,
            "dd_tiers": [list(t) for t in self.dd_tiers],
            "symbol_collision": self.symbol_collision,
            "symbol_exposure_cap_R": self.symbol_exposure_cap_R,
            "terminal_accounting_mode": self.terminal_accounting_mode,
            "priority_headroom_R": self.priority_headroom_R,
            "priority_reserve_threshold": self.priority_reserve_threshold,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PortfolioConfig:
        """Deserialize from dict."""
        strategies = tuple(
            StrategyAllocation(**s) for s in d.get("strategies", [])
        )
        dd_tiers = tuple(
            tuple(t) for t in d.get("dd_tiers", cls.dd_tiers)
        )
        return cls(
            initial_equity=d.get("initial_equity", cls.initial_equity),
            strategies=strategies,
            heat_cap_R=d.get("heat_cap_R", cls.heat_cap_R),
            directional_cap_R=d.get("directional_cap_R", cls.directional_cap_R),
            portfolio_daily_stop_R=d.get("portfolio_daily_stop_R", cls.portfolio_daily_stop_R),
            max_total_positions=d.get("max_total_positions", cls.max_total_positions),
            dd_tiers=dd_tiers,
            symbol_collision=d.get("symbol_collision", cls.symbol_collision),
            symbol_exposure_cap_R=d.get("symbol_exposure_cap_R", cls.symbol_exposure_cap_R),
            terminal_accounting_mode=d.get("terminal_accounting_mode", cls.terminal_accounting_mode),
            priority_headroom_R=d.get("priority_headroom_R", cls.priority_headroom_R),
            priority_reserve_threshold=d.get("priority_reserve_threshold", cls.priority_reserve_threshold),
        )
