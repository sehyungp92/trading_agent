"""Thin bootstrap wrapper for config-driven capital allocation.

Usage:
    allocs = bootstrap_capital(account_nav, config_dir)
    my_nav = allocs["ATRSS"].allocated_nav
"""
from __future__ import annotations

from pathlib import Path

from .capital_allocation import StrategyCapitalAllocation, resolve_strategy_capital_allocation
from .loader import load_portfolio_config, load_strategy_registry


def bootstrap_capital(
    account_nav: float,
    config_dir: str | Path,
    *,
    live: bool = False,
) -> dict[str, StrategyCapitalAllocation]:
    """Load portfolio + registry configs and resolve per-strategy NAV.

    Args:
        account_nav: Total account equity (live or paper).
        config_dir: Directory containing strategies.yaml and portfolio.yaml.
        live: If True, filter paper-only strategies exactly like runtime startup.

    Returns:
        Dict mapping strategy_id → StrategyCapitalAllocation with allocated_nav.
    """
    config_dir = Path(config_dir)
    registry = load_strategy_registry(config_dir)
    portfolio = load_portfolio_config(config_dir)
    return {
        s.strategy_id: resolve_strategy_capital_allocation(
            s.strategy_id, account_nav, registry, portfolio
        )
        for s in registry.enabled_strategies(live=live)
    }
