"""Portfolio configuration for the legacy momentum overlay.

The dedicated Momentum Helix engine has been retired.  These presets now
cover the remaining NQ momentum engines that still feed the old overlay:
NQDTC and Vdubus.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AccountTier(Enum):
    SMALL = "10K"
    STANDARD = "100K"


@dataclass(frozen=True)
class StrategyAllocation:
    """Per-strategy risk allocation within a portfolio."""

    strategy_id: str
    enabled: bool
    base_risk_pct: float
    daily_stop_R: float
    max_concurrent: int = 1
    priority: int = 0
    continuation_half_size: bool = False
    continuation_size_mult: float = 0.50
    reversal_only: bool = False


@dataclass(frozen=True)
class PortfolioConfig:
    """Portfolio-level configuration for the combined backtest."""

    tier: AccountTier
    initial_equity: float
    heat_cap_R: float
    portfolio_daily_stop_R: float
    max_total_positions: int
    directional_cap_R: float
    strategies: tuple[StrategyAllocation, ...]
    point_value: float = 2.0
    tick_size: float = 0.25
    tick_value: float = 0.50
    commission_per_side: float = 0.62
    dd_tiers: tuple[tuple[float, float], ...] = (
        (0.08, 1.00),
        (0.12, 0.50),
        (0.15, 0.25),
        (1.00, 0.00),
    )
    nqdtc_direction_filter_enabled: bool = True
    nqdtc_agree_size_mult: float = 1.50
    nqdtc_oppose_size_mult: float = 0.0

    def get_strategy(self, strategy_id: str) -> StrategyAllocation | None:
        for strategy in self.strategies:
            if strategy.strategy_id == strategy_id:
                return strategy
        return None

    def priority_order(self) -> list[StrategyAllocation]:
        """Return strategies sorted by priority, lowest number first."""

        return sorted(self.strategies, key=lambda strategy: strategy.priority)


def _momentum_pair_config(
    *,
    tier: AccountTier,
    initial_equity: float,
    nqdtc_risk: float,
    vdubus_risk: float,
    heat_cap_R: float,
    daily_stop_R: float,
    directional_cap_R: float,
    nqdtc_daily_stop_R: float,
    vdubus_daily_stop_R: float,
    continuation_size_mult: float = 0.70,
) -> PortfolioConfig:
    return PortfolioConfig(
        tier=tier,
        initial_equity=initial_equity,
        heat_cap_R=heat_cap_R,
        portfolio_daily_stop_R=daily_stop_R,
        max_total_positions=2,
        directional_cap_R=directional_cap_R,
        strategies=(
            StrategyAllocation(
                strategy_id="Vdubus",
                enabled=True,
                base_risk_pct=vdubus_risk,
                daily_stop_R=vdubus_daily_stop_R,
                max_concurrent=1,
                priority=0,
            ),
            StrategyAllocation(
                strategy_id="NQDTC",
                enabled=True,
                base_risk_pct=nqdtc_risk,
                daily_stop_R=nqdtc_daily_stop_R,
                max_concurrent=1,
                priority=1,
                continuation_half_size=True,
                continuation_size_mult=continuation_size_mult,
            ),
        ),
    )


def make_10k_config() -> PortfolioConfig:
    """Conservative $10K NQDTC + Vdubus configuration."""

    return _momentum_pair_config(
        tier=AccountTier.SMALL,
        initial_equity=10_000.0,
        nqdtc_risk=0.006,
        vdubus_risk=0.006,
        heat_cap_R=2.0,
        daily_stop_R=2.0,
        directional_cap_R=2.0,
        nqdtc_daily_stop_R=2.0,
        vdubus_daily_stop_R=2.0,
        continuation_size_mult=0.50,
    )


def make_100k_config() -> PortfolioConfig:
    """Conservative $100K NQDTC + Vdubus configuration."""

    return _momentum_pair_config(
        tier=AccountTier.STANDARD,
        initial_equity=100_000.0,
        nqdtc_risk=0.003,
        vdubus_risk=0.003,
        heat_cap_R=2.5,
        daily_stop_R=3.0,
        directional_cap_R=2.5,
        nqdtc_daily_stop_R=2.5,
        vdubus_daily_stop_R=2.5,
        continuation_size_mult=0.50,
    )


def make_10k_optimized_config() -> PortfolioConfig:
    """Balanced $10K NQDTC + Vdubus configuration."""

    return _momentum_pair_config(
        tier=AccountTier.SMALL,
        initial_equity=10_000.0,
        nqdtc_risk=0.008,
        vdubus_risk=0.008,
        heat_cap_R=2.5,
        daily_stop_R=2.5,
        directional_cap_R=2.5,
        nqdtc_daily_stop_R=2.5,
        vdubus_daily_stop_R=2.5,
    )


def make_100k_optimized_config() -> PortfolioConfig:
    """Balanced $100K NQDTC + Vdubus configuration."""

    return _momentum_pair_config(
        tier=AccountTier.STANDARD,
        initial_equity=100_000.0,
        nqdtc_risk=0.0035,
        vdubus_risk=0.003,
        heat_cap_R=3.0,
        daily_stop_R=3.5,
        directional_cap_R=3.0,
        nqdtc_daily_stop_R=3.0,
        vdubus_daily_stop_R=2.5,
    )


def make_10k_v3_config() -> PortfolioConfig:
    return make_10k_optimized_config()


def make_100k_v3_config() -> PortfolioConfig:
    return make_100k_optimized_config()


def make_100k_v3_max_config() -> PortfolioConfig:
    return _momentum_pair_config(
        tier=AccountTier.STANDARD,
        initial_equity=100_000.0,
        nqdtc_risk=0.005,
        vdubus_risk=0.003,
        heat_cap_R=3.0,
        daily_stop_R=3.5,
        directional_cap_R=3.0,
        nqdtc_daily_stop_R=3.0,
        vdubus_daily_stop_R=2.5,
    )


def make_10k_v4_config() -> PortfolioConfig:
    return _momentum_pair_config(
        tier=AccountTier.SMALL,
        initial_equity=10_000.0,
        nqdtc_risk=0.008,
        vdubus_risk=0.008,
        heat_cap_R=3.0,
        daily_stop_R=2.5,
        directional_cap_R=3.0,
        nqdtc_daily_stop_R=2.5,
        vdubus_daily_stop_R=2.5,
    )


def make_10k_v5_config() -> PortfolioConfig:
    return _momentum_pair_config(
        tier=AccountTier.SMALL,
        initial_equity=10_000.0,
        nqdtc_risk=0.008,
        vdubus_risk=0.009,
        heat_cap_R=3.0,
        daily_stop_R=2.0,
        directional_cap_R=3.0,
        nqdtc_daily_stop_R=2.5,
        vdubus_daily_stop_R=2.5,
    )


def make_10k_v6_config() -> PortfolioConfig:
    return _momentum_pair_config(
        tier=AccountTier.SMALL,
        initial_equity=10_000.0,
        nqdtc_risk=0.008,
        vdubus_risk=0.010,
        heat_cap_R=3.0,
        daily_stop_R=1.5,
        directional_cap_R=3.0,
        nqdtc_daily_stop_R=2.5,
        vdubus_daily_stop_R=2.5,
    )


def make_10k_v7_config() -> PortfolioConfig:
    return make_10k_v6_config()


def make_100k_halfsized_config() -> PortfolioConfig:
    return make_100k_optimized_config()
