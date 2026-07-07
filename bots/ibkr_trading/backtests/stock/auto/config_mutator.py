"""Config mutation for automated experiments.

Generates mutated ALCB/IARIC/Portfolio configs from a base config
and a mutation dictionary using dot-notation keys.
"""
from __future__ import annotations

from dataclasses import fields, replace

from backtests.stock.config import SlippageConfig
from backtests.stock.config_alcb import ALCBAblationFlags, ALCBBacktestConfig
from backtests.stock.config_iaric import IARICAblationFlags, IARICBacktestConfig
from backtests.stock.config_portfolio import PortfolioBacktestConfig


_IARIC_LEGACY_TOP_LEVEL_PARAM_ALIASES = {
    "max_per_sector": "max_positions_per_sector",
    "max_positions_tier_a": "max_positions_tier_a",
    "max_positions_tier_b": "max_positions_tier_b",
    "sector_risk_cap_pct": "sector_risk_cap_pct",
}


def mutate_alcb_config(
    base: ALCBBacktestConfig,
    mutations: dict,
) -> ALCBBacktestConfig:
    """Apply dot-notation mutations to an ALCB config.

    Mutation key patterns:
      - "ablation.use_stale_exit": False  → builds new ALCBAblationFlags
      - "param_overrides.stale_exit_days": 12  → merges into param_overrides
      - "slippage.commission_per_share": 0.01  → builds new SlippageConfig
      - "max_positions": 3  → direct replace on top-level field
    """
    config = base
    ablation_changes: dict = {}
    param_changes: dict = {}
    slippage_changes: dict = {}
    top_level: dict = {}

    for key, value in mutations.items():
        if key.startswith("ablation."):
            ablation_changes[key.split(".", 1)[1]] = value
        elif key.startswith("param_overrides."):
            param_changes[key.split(".", 1)[1]] = value
        elif key.startswith("slippage."):
            slippage_changes[key.split(".", 1)[1]] = value
        else:
            top_level[key] = value

    if ablation_changes:
        new_ablation = replace(config.ablation, **ablation_changes)
        config = replace(config, ablation=new_ablation)

    if param_changes:
        merged = {**config.param_overrides, **param_changes}
        config = replace(config, param_overrides=merged)

    if slippage_changes:
        new_slippage = _mutate_slippage(config.slippage, slippage_changes)
        config = replace(config, slippage=new_slippage)

    if top_level:
        config = replace(config, **top_level)

    return config


def mutate_iaric_config(
    base: IARICBacktestConfig,
    mutations: dict,
) -> IARICBacktestConfig:
    """Apply dot-notation mutations to an IARIC config."""
    config = base
    ablation_changes: dict = {}
    param_changes: dict = {}
    slippage_changes: dict = {}
    top_level: dict = {}

    for key, value in mutations.items():
        if key.startswith("ablation."):
            ablation_changes[key.split(".", 1)[1]] = value
        elif key.startswith("param_overrides."):
            param_changes[key.split(".", 1)[1]] = value
        elif key.startswith("slippage."):
            slippage_changes[key.split(".", 1)[1]] = value
        elif key in _IARIC_LEGACY_TOP_LEVEL_PARAM_ALIASES:
            param_changes[_IARIC_LEGACY_TOP_LEVEL_PARAM_ALIASES[key]] = value
        else:
            top_level[key] = value

    if ablation_changes:
        new_ablation = replace(config.ablation, **ablation_changes)
        config = replace(config, ablation=new_ablation)

    if param_changes:
        merged = {**config.param_overrides, **param_changes}
        config = replace(config, param_overrides=merged)

    if slippage_changes:
        new_slippage = _mutate_slippage(config.slippage, slippage_changes)
        config = replace(config, slippage=new_slippage)

    if top_level:
        config = replace(config, **top_level)

    return config


def mutate_portfolio_config(
    base: PortfolioBacktestConfig,
    mutations: dict,
) -> PortfolioBacktestConfig:
    """Apply mutations to a portfolio config."""
    config = base
    slippage_changes: dict = {}
    top_level: dict = {}

    for key, value in mutations.items():
        if key.startswith("slippage."):
            slippage_changes[key.split(".", 1)[1]] = value
        else:
            top_level[key] = value

    if slippage_changes:
        new_slippage = _mutate_slippage(config.slippage, slippage_changes)
        config = replace(config, slippage=new_slippage)

    if top_level:
        config = replace(config, **top_level)

    return config


def _mutate_slippage(base: SlippageConfig, changes: dict) -> SlippageConfig:
    """Build a new SlippageConfig with the given changes."""
    current = {f.name: getattr(base, f.name) for f in fields(base)}
    current.update(changes)
    return SlippageConfig(**current)
