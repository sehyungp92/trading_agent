"""Downturn config mutation — dot-notation for DownturnBacktestConfig.

Mutation routing:
  - "flags.<field>"          → replace config.flags
  - "param_overrides.<KEY>"  → merge into config.param_overrides
  - "slippage.<field>"       → replace config.slippage
  - "<top-level>"            → replace on config directly
"""
from __future__ import annotations

from dataclasses import fields, replace

from backtests.momentum.config_downturn import (
    DownturnAblationFlags,
    DownturnBacktestConfig,
)


def mutate_downturn_config(
    base: DownturnBacktestConfig,
    mutations: dict,
) -> DownturnBacktestConfig:
    """Apply dot-notation mutations to a DownturnBacktestConfig."""
    if not mutations:
        return base

    flag_updates: dict = {}
    param_updates: dict = {}
    slippage_updates: dict = {}
    top_updates: dict = {}

    flag_fields = {f.name: f for f in fields(DownturnAblationFlags)}
    # Determine expected type from default value (bool check first since bool is subclass of int)
    flag_types: dict[str, type] = {}
    for fname, fobj in flag_fields.items():
        default = fobj.default
        if isinstance(default, bool):
            flag_types[fname] = bool
        elif isinstance(default, int):
            flag_types[fname] = int
        elif isinstance(default, float):
            flag_types[fname] = float
        else:
            flag_types[fname] = type(default) if default is not None else bool

    for key, value in mutations.items():
        if key.startswith("flags."):
            field_name = key[len("flags."):]
            if field_name in flag_fields:
                expected = flag_types.get(field_name, bool)
                if expected is bool:
                    flag_updates[field_name] = bool(value)
                elif expected is int:
                    flag_updates[field_name] = int(value)
                elif expected is float:
                    flag_updates[field_name] = float(value)
                else:
                    flag_updates[field_name] = value
            else:
                raise ValueError(f"Unknown flag field: {field_name}")

        elif key.startswith("param_overrides."):
            param_key = key[len("param_overrides."):]
            param_updates[param_key] = float(value)

        elif key.startswith("slippage."):
            field_name = key[len("slippage."):]
            slippage_updates[field_name] = value

        else:
            top_updates[key] = value

    config = base

    if flag_updates:
        config = replace(config, flags=replace(config.flags, **flag_updates))

    if param_updates:
        merged = dict(config.param_overrides)
        merged.update(param_updates)
        config = replace(config, param_overrides=merged)

    if slippage_updates:
        config = replace(config, slippage=replace(config.slippage, **slippage_updates))

    if top_updates:
        config = replace(config, **top_updates)

    return config
