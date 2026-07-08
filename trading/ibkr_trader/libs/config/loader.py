"""Config loading helpers for the unified runtime scaffold."""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from .event_calendar import EventCalendar, EventWindow
from .models import (
    ContractTemplate,
    ExchangeRoute,
    PortfolioConfig,
    StrategyRegistryConfig,
)

T = TypeVar("T", bound=BaseModel)

_ENV_VAR_RE = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def _resolve_env_vars(obj):
    """Recursively resolve ${VAR} and ${VAR:default} in string values."""
    if isinstance(obj, str):
        def _replace(m):
            var_name, default = m.group(1), m.group(2)
            return os.environ.get(var_name, default if default is not None else m.group(0))
        return _ENV_VAR_RE.sub(_replace, obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    return obj


def _load_yaml_file(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except (yaml.YAMLError, OSError) as exc:
        raise ValueError(f"Failed to load config from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return _resolve_env_vars(data)


def _load_model(path: Path, model_cls: type[T]) -> T:
    return model_cls.model_validate(_load_yaml_file(path))


def load_strategy_registry(config_dir: str | Path) -> StrategyRegistryConfig:
    return _load_model(Path(config_dir) / "strategies.yaml", StrategyRegistryConfig)


def load_portfolio_config(config_dir: str | Path) -> PortfolioConfig:
    return _load_model(Path(config_dir) / "portfolio.yaml", PortfolioConfig)


def load_contracts(config_dir: str | Path) -> dict[str, ContractTemplate]:
    raw = _load_yaml_file(Path(config_dir) / "contracts.yaml")
    return {symbol: ContractTemplate.model_validate(payload) for symbol, payload in raw.items()}


def load_routes(config_dir: str | Path) -> dict[str, ExchangeRoute]:
    raw = _load_yaml_file(Path(config_dir) / "routing.yaml")
    return {symbol: ExchangeRoute.model_validate(payload) for symbol, payload in raw.items()}


def load_event_calendar(config_dir: str | Path) -> EventCalendar:
    raw = _load_yaml_file(Path(config_dir) / "event_calendar.yaml")
    windows = [
        EventWindow(
            name=window["name"],
            start_utc=_parse_datetime(window["start_utc"]),
            end_utc=_parse_datetime(window["end_utc"]),
            cooldown_bars=int(window.get("cooldown_bars", 3)),
            max_extension_minutes=int(window.get("max_extension_minutes", 60)),
        )
        for window in raw.get("windows", [])
    ]
    return EventCalendar(windows)


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Datetime {value!r} must be timezone-aware")
    return parsed

