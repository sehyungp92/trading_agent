"""Dot-notation config mutation for momentum auto-backtesting.

Three mutator functions, one per config type. Each parses mutation keys
by prefix, collects changes by category, and applies via dataclasses.replace().

Key routing:
  - flags.<field>           -> replace config.flags
  - param_overrides.<KEY>   -> merge into config.param_overrides
  - slippage.<field>        -> replace config.slippage
  - <top-level>             -> replace on config directly

Portfolio config additionally routes:
  - portfolio.<field>       -> replace config.portfolio
  - portfolio.strategies[N].<field> -> rebuild strategies tuple
  - run_nqdtc / run_vdubus -> direct replace
  - nqdtc_flags.<field> / vdubus_flags.<field> -> pass-through
  - nqdtc_param.<KEY> / vdubus_param.<KEY> -> pass-through
"""
from __future__ import annotations

import re
from dataclasses import fields, replace

from backtests.momentum.config import SlippageConfig
from backtests.momentum.config_nqdtc import NQDTCAblationFlags, NQDTCBacktestConfig
from backtests.momentum.config_vdubus import VdubusAblationFlags, VdubusBacktestConfig
from backtests.momentum.config_portfolio import PortfolioBacktestConfig
from libs.oms.config.portfolio_config import PortfolioConfig, StrategyAllocation


# ---------------------------------------------------------------------------
# Strategy-level mutators
# ---------------------------------------------------------------------------

def mutate_nqdtc_config(base: NQDTCBacktestConfig, mutations: dict) -> NQDTCBacktestConfig:
    """Apply dot-notation mutations to a NQDTCBacktestConfig."""
    return _mutate_strategy_config(base, mutations, NQDTCAblationFlags)


def mutate_vdubus_config(base: VdubusBacktestConfig, mutations: dict) -> VdubusBacktestConfig:
    """Apply dot-notation mutations to a VdubusBacktestConfig."""
    return _mutate_strategy_config(base, mutations, VdubusAblationFlags)


def _mutate_strategy_config(base, mutations: dict, flags_cls):
    """Generic strategy config mutation routing."""
    if not mutations:
        return base

    flag_changes: dict[str, object] = {}
    param_changes: dict[str, object] = {}
    slippage_changes: dict[str, object] = {}
    top_level: dict[str, object] = {}

    for key, value in mutations.items():
        if key.startswith("flags."):
            field_name = key[len("flags."):]
            flag_changes[field_name] = value
        elif key.startswith("param_overrides."):
            param_key = key[len("param_overrides."):]
            param_changes[param_key] = value
        elif key.startswith("slippage."):
            field_name = key[len("slippage."):]
            slippage_changes[field_name] = value
        else:
            top_level[key] = value

    config = base

    if flag_changes:
        new_flags = replace(config.flags, **flag_changes)
        config = replace(config, flags=new_flags)

    if param_changes:
        merged = {**config.param_overrides, **param_changes}
        config = replace(config, param_overrides=merged)

    if slippage_changes:
        config = replace(config, slippage=_mutate_slippage(config.slippage, slippage_changes))

    if top_level:
        config = replace(config, **top_level)

    return config


# ---------------------------------------------------------------------------
# Portfolio-level mutator
# ---------------------------------------------------------------------------

_STRATEGY_PREFIX_RE = re.compile(r"^portfolio\.strategies\[(\d+)\]\.(.+)$")


def mutate_portfolio_config(
    base: PortfolioBacktestConfig,
    mutations: dict,
) -> PortfolioBacktestConfig:
    """Apply dot-notation mutations to a PortfolioBacktestConfig.

    Routes:
      portfolio.<field>                    -> PortfolioConfig field
      portfolio.strategies[N].<field>      -> StrategyAllocation at index N
      portfolio.dd_tiers                   -> tuple of (dd_pct, mult) tuples
      run_nqdtc / run_vdubus  -> direct on PortfolioBacktestConfig
      nqdtc_flags.<f> / vdubus_flags.<f>  -> stored for pass-through
      nqdtc_param.<K> / vdubus_param.<K>  -> stored for pass-through
      <top-level>                          -> direct on PortfolioBacktestConfig
    """
    if not mutations:
        return base

    portfolio_changes: dict[str, object] = {}
    strategy_changes: dict[int, dict[str, object]] = {}
    top_level: dict[str, object] = {}

    # These are pass-through keys that the harness reads when building
    # per-strategy configs for portfolio experiments
    _passthrough_prefixes = (
        "nqdtc_flags.", "vdubus_flags.",
        "nqdtc_param.", "vdubus_param.",
    )

    for key, value in mutations.items():
        # Strategy allocation mutations: portfolio.strategies[N].field
        m = _STRATEGY_PREFIX_RE.match(key)
        if m:
            idx = int(m.group(1))
            field_name = m.group(2)
            strategy_changes.setdefault(idx, {})[field_name] = value
            continue

        if key.startswith("portfolio."):
            field_name = key[len("portfolio."):]
            portfolio_changes[field_name] = value
        elif any(key.startswith(p) for p in _passthrough_prefixes):
            # Passthrough keys (nqdtc_flags.*, nqdtc_param.*, etc.) are read
            # directly from the mutations dict by the harness, not from the config.
            continue
        else:
            top_level[key] = value

    config = base

    # Apply portfolio-level changes
    if portfolio_changes or strategy_changes:
        pc = config.portfolio

        # Rebuild strategies tuple if any allocation changes
        if strategy_changes:
            strategies_list = list(pc.strategies)
            for idx, changes in strategy_changes.items():
                if 0 <= idx < len(strategies_list):
                    strategies_list[idx] = replace(strategies_list[idx], **changes)
            portfolio_changes["strategies"] = tuple(strategies_list)

        # Handle dd_tiers — expect tuple of tuples
        if "dd_tiers" in portfolio_changes:
            val = portfolio_changes["dd_tiers"]
            if isinstance(val, list):
                portfolio_changes["dd_tiers"] = tuple(tuple(t) for t in val)

        if portfolio_changes:
            pc = replace(pc, **portfolio_changes)
            config = replace(config, portfolio=pc)

    # Apply top-level changes (run_nqdtc, run_vdubus, etc.)
    # Filter to only valid PortfolioBacktestConfig fields
    bt_fields = {f.name for f in fields(PortfolioBacktestConfig)}
    bt_level = {k: v for k, v in top_level.items() if k in bt_fields}
    if bt_level:
        config = replace(config, **bt_level)

    return config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mutate_slippage(base: SlippageConfig, changes: dict) -> SlippageConfig:
    """Build a new SlippageConfig from base + changes."""
    kw = {}
    for f in fields(base):
        kw[f.name] = changes.get(f.name, getattr(base, f.name))
    return SlippageConfig(**kw)


def extract_passthrough_mutations(
    mutations: dict,
    strategy: str,
) -> dict:
    """Extract strategy-specific flag/param mutations from portfolio experiment mutations.

    Used by harness to build per-strategy configs when running portfolio experiments.
    E.g. nqdtc_flags.use_trailing -> flags.use_trailing (for NQDTC config)
         nqdtc_param.TRAIL_MULT  -> param_overrides.TRAIL_MULT
    """
    prefix_flags = f"{strategy}_flags."
    prefix_param = f"{strategy}_param."
    result: dict[str, object] = {}

    for key, value in mutations.items():
        if key.startswith(prefix_flags):
            field_name = key[len(prefix_flags):]
            result[f"flags.{field_name}"] = value
        elif key.startswith(prefix_param):
            param_key = key[len(prefix_param):]
            result[f"param_overrides.{param_key}"] = value

    return result
