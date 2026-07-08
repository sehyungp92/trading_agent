"""Config mutation for automated swing experiments.

Generates mutated ATRSS/Helix/Unified configs from a base config
and a mutation dictionary using dot-notation keys.
"""
from __future__ import annotations

from dataclasses import fields, replace

from backtests.swing.config import AblationFlags, BacktestConfig, SlippageConfig
from backtests.swing.config_helix import HelixAblationFlags, HelixBacktestConfig
from backtests.swing.config_unified import StrategySlot, UnifiedBacktestConfig


def mutate_atrss_config(
    base: BacktestConfig,
    mutations: dict,
) -> BacktestConfig:
    """Apply dot-notation mutations to an ATRSS config.

    Mutation key patterns:
      - "flags.stall_exit": False  → builds new AblationFlags
      - "param_overrides.cooldown_strong": 4  → merges into param_overrides
      - "slippage.commission_per_share_etf": 0.01  → builds new SlippageConfig
      - "initial_equity": 50000  → direct replace on top-level field
    """
    config = base
    flag_changes: dict = {}
    param_changes: dict = {}
    slippage_changes: dict = {}
    top_level: dict = {}

    for key, value in mutations.items():
        if key.startswith("flags."):
            flag_changes[key.split(".", 1)[1]] = value
        elif key.startswith("param_overrides."):
            param_changes[key.split(".", 1)[1]] = value
        elif key.startswith("slippage."):
            slippage_changes[key.split(".", 1)[1]] = value
        else:
            top_level[key] = value

    if flag_changes:
        new_flags = replace(config.flags, **flag_changes)
        config = replace(config, flags=new_flags)

    if param_changes:
        merged = {**config.param_overrides, **param_changes}
        config = replace(config, param_overrides=merged)

    if slippage_changes:
        new_slippage = _mutate_slippage(config.slippage, slippage_changes)
        config = replace(config, slippage=new_slippage)

    if top_level:
        config = replace(config, **top_level)

    return config


def mutate_helix_config(
    base: HelixBacktestConfig,
    mutations: dict,
) -> HelixBacktestConfig:
    """Apply dot-notation mutations to a Helix config.

    Mutation key patterns:
      - "flags.disable_class_c": True  → builds new HelixAblationFlags
      - "param_overrides.CHANDELIER_MULT": 1.5  → merges into param_overrides (NEW)
      - "slippage.*"  → SlippageConfig
      - top-level  → direct replace
    """
    config = base
    flag_changes: dict = {}
    param_changes: dict = {}
    slippage_changes: dict = {}
    top_level: dict = {}

    for key, value in mutations.items():
        if key.startswith("flags."):
            flag_changes[key.split(".", 1)[1]] = value
        elif key.startswith("param_overrides."):
            param_changes[key.split(".", 1)[1]] = value
        elif key.startswith("slippage."):
            slippage_changes[key.split(".", 1)[1]] = value
        else:
            top_level[key] = value

    if flag_changes:
        new_flags = replace(config.flags, **flag_changes)
        config = replace(config, flags=new_flags)

    if param_changes:
        merged = {**config.param_overrides, **param_changes}
        config = replace(config, param_overrides=merged)

    if slippage_changes:
        new_slippage = _mutate_slippage(config.slippage, slippage_changes)
        config = replace(config, slippage=new_slippage)

    if top_level:
        config = replace(config, **top_level)

    return config


def mutate_unified_config(
    base: UnifiedBacktestConfig,
    mutations: dict,
) -> UnifiedBacktestConfig:
    """Apply mutations to a UnifiedBacktestConfig.

    Mutation key patterns:
      - "heat_cap_R": 3.5  → top-level field
      - "atrss.unit_risk_pct": 0.020  → strategy slot mutation
      - "helix.max_heat_R": 1.5  → strategy slot mutation
      - "overlay_enabled": False  → top-level overlay toggle
      - "enable_atrss_helix_tighten": False  → coordination toggle
      - "helix_param.STALE_4H_BARS": 4  → per-strategy engine param
      - "helix_flags.disable_class_c": True  → per-strategy flag
    """
    config = base
    slot_changes: dict[str, dict] = {}
    param_override_changes: dict[str, dict] = {}
    flag_changes: dict[str, dict] = {}
    top_level: dict = {}

    strategy_slot_names = {"atrss", "helix", "tpc"}
    param_override_prefixes = {
        "atrss_param", "helix_param", "tpc_param",
    }
    flag_prefixes = {"atrss_flags", "helix_flags"}

    for key, value in mutations.items():
        parts = key.split(".", 1)
        if len(parts) == 2 and parts[0] in strategy_slot_names:
            slot_name, field_name = parts
            slot_changes.setdefault(slot_name, {})[field_name] = value
        elif len(parts) == 2 and parts[0] in param_override_prefixes:
            prefix, param_name = parts
            param_override_changes.setdefault(prefix, {})[param_name] = value
        elif len(parts) == 2 and parts[0] in flag_prefixes:
            prefix, flag_name = parts
            flag_changes.setdefault(prefix, {})[flag_name] = value
        else:
            top_level[key] = value

    # Apply slot mutations
    for slot_name, changes in slot_changes.items():
        current_slot: StrategySlot = getattr(config, slot_name)
        new_slot = replace(current_slot, **changes)
        config = replace(config, **{slot_name: new_slot})

    # Apply per-strategy param overrides
    for prefix, changes in param_override_changes.items():
        field_name = prefix + "_overrides"
        current = getattr(config, field_name, {})
        merged = {**current, **changes}
        config = replace(config, **{field_name: merged})

    # Apply per-strategy flag mutations
    for prefix, changes in flag_changes.items():
        current_flags = getattr(config, prefix)
        new_flags = replace(current_flags, **changes)
        config = replace(config, **{prefix: new_flags})

    if top_level:
        config = replace(config, **top_level)

    return config


def _mutate_slippage(base: SlippageConfig, changes: dict) -> SlippageConfig:
    """Build a new SlippageConfig from base + changes."""
    kw = {}
    for f in fields(base):
        kw[f.name] = changes.get(f.name, getattr(base, f.name))
    return SlippageConfig(**kw)
