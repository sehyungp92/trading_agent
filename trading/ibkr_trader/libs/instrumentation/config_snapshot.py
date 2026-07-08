"""Shared effective-config snapshot helpers."""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from .event_contract import write_config_snapshot, write_deployment_event
from .lineage import LineageContext, lineage_from_runtime, redact_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {
            str(k): _plain(v)
            for k, v in value.items()
            if not str(k).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(item) for item in value)
    return value


def snapshot_yaml_file(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        data = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return redact_config(_plain(data)) if isinstance(data, Mapping) else {}


def flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, raw in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(raw, Mapping):
            flattened.update(flatten_mapping(raw, name))
        else:
            flattened[name] = raw
    return flattened


def default_yaml_watch_paths(config_dir: str | Path | None = None) -> list[Path]:
    cfg_dir = Path(config_dir) if config_dir is not None else _repo_root() / "config"
    return [
        cfg_dir / "strategies.yaml",
        cfg_dir / "portfolio.yaml",
        cfg_dir / "sector_map.yaml",
        cfg_dir / "routing.yaml",
        cfg_dir / "contracts.yaml",
        cfg_dir / "event_calendar.yaml",
    ]


def build_effective_strategy_config(
    strategy_id: str,
    *,
    config_dir: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    runtime_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    env = env or os.environ
    cfg_dir = Path(config_dir) if config_dir is not None else _repo_root() / "config"
    payload: dict[str, Any] = {
        "strategy_id": strategy_id,
        "runtime_config": redact_config(dict(runtime_config or {})),
        "env_overrides": _selected_env(env),
    }
    try:
        from libs.config.loader import load_strategy_registry

        registry = load_strategy_registry(cfg_dir)
        manifest = registry.strategies.get(strategy_id)
        if manifest is not None:
            payload["strategy_manifest"] = _plain(manifest)
            group = registry.connection_groups.get(manifest.connection_group)
            if group is not None:
                payload["connection_group"] = _plain(group)
    except Exception as exc:
        payload["load_error"] = str(exc)
    return redact_config(payload)


def build_effective_portfolio_config(
    config_dir: str | Path | None = None,
    *,
    family_id: str = "",
    env: Mapping[str, str] | None = None,
    portfolio_rules_config: Any = None,
) -> dict[str, Any]:
    env = env or os.environ
    cfg_dir = Path(config_dir) if config_dir is not None else _repo_root() / "config"
    payload: dict[str, Any] = {"family_id": family_id, "env_overrides": _selected_env(env)}
    try:
        from libs.config.loader import load_portfolio_config, load_strategy_registry

        portfolio = load_portfolio_config(cfg_dir)
        registry = load_strategy_registry(cfg_dir)
        enabled = [
            strategy_id
            for strategy_id, manifest in registry.strategies.items()
            if manifest.enabled and (not family_id or manifest.family == family_id)
        ]
        payload.update(
            {
                "portfolio": _plain(portfolio),
                "enabled_strategy_ids": sorted(enabled),
                "family_allocation": _plain(portfolio.capital.family_allocations).get(family_id, 0.0),
                "strategy_allocations": _plain(portfolio.capital.strategy_allocations),
            }
        )
    except Exception as exc:
        payload["load_error"] = str(exc)
    if portfolio_rules_config is not None:
        payload["portfolio_rules_config"] = _plain(portfolio_rules_config)
    return redact_config(payload)


def build_effective_risk_config(
    family_id: str,
    portfolio_rules_config: Any = None,
    *,
    config_dir: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = env or os.environ
    cfg_dir = Path(config_dir) if config_dir is not None else _repo_root() / "config"
    payload: dict[str, Any] = {"family_id": family_id, "env_overrides": _selected_env(env)}
    try:
        from libs.config.loader import load_portfolio_config, load_strategy_registry

        portfolio = load_portfolio_config(cfg_dir)
        registry = load_strategy_registry(cfg_dir)
        payload.update(
            {
                "portfolio_risk": _plain(portfolio.risk),
                "drawdown_tiers": _plain(portfolio.drawdown_tiers),
                "coordination": _plain(portfolio.coordination),
                "strategy_risk": {
                    strategy_id: _plain(manifest.risk)
                    for strategy_id, manifest in registry.strategies.items()
                    if not family_id or manifest.family == family_id
                },
            }
        )
    except Exception as exc:
        payload["load_error"] = str(exc)
    if portfolio_rules_config is not None:
        payload["portfolio_rules_config"] = _plain(portfolio_rules_config)
    return redact_config(payload)


def build_lineage_context(
    *,
    bot_id: str,
    strategy_id: str = "",
    family_id: str = "",
    config_dir: str | Path | None = None,
    runtime_config: Mapping[str, Any] | None = None,
    portfolio_rules_config: Any = None,
    env: Mapping[str, str] | None = None,
) -> LineageContext:
    effective_strategy_config = build_effective_strategy_config(
        strategy_id,
        config_dir=config_dir,
        env=env,
        runtime_config=runtime_config,
    )
    return lineage_from_runtime(
        bot_id=bot_id,
        strategy_id=strategy_id,
        family_id=family_id,
        config_dir=config_dir,
        effective_strategy_config=effective_strategy_config,
        portfolio_rules_config=portfolio_rules_config,
        env=env,
    )


def _selected_env(env: Mapping[str, str]) -> dict[str, str]:
    keys = (
        "INSTRUMENTATION_CONTRACT_VERSION",
        "TRADING_MODE",
        "TRADING_ENV",
        "PORTFOLIO_ID",
        "TRADING_PORTFOLIO_ID",
        "ACCOUNT_ALIAS",
        "TRADING_ACCOUNT_ALIAS",
        "PARAMETER_SET_ID",
        "DEPLOYMENT_ID",
    )
    return {key: str(env[key]) for key in keys if env.get(key)}


__all__ = [
    "build_effective_portfolio_config",
    "build_effective_risk_config",
    "build_effective_strategy_config",
    "build_lineage_context",
    "default_yaml_watch_paths",
    "flatten_mapping",
    "snapshot_yaml_file",
    "write_config_snapshot",
    "write_deployment_event",
]
