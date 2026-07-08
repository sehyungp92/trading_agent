"""Config loader + market-hours awareness for the watchdog."""
from __future__ import annotations

import os
from datetime import datetime, time, timedelta
from pathlib import Path

from libs.config.loader import _load_yaml_file, load_strategy_registry
from libs.config.market_calendar import AssetClass, MarketCalendar

_ASSET_CLASS_MAP = {
    "equity": AssetClass.EQUITY,
    "cme_futures": AssetClass.CME_FUTURES,
}

_calendar = MarketCalendar()


def load_watchdog_config(config_dir: str | Path | None = None) -> dict:
    """Load watchdog.yaml with ${ENV_VAR} substitution."""
    if config_dir is None:
        config_dir = os.environ.get("CONFIG_DIR", "config")
    return _load_yaml_file(Path(config_dir) / "watchdog.yaml")


def build_strategy_family_map(config_dir: str | Path | None = None) -> dict[str, str]:
    """Return {strategy_id: family} from strategies.yaml."""
    if config_dir is None:
        config_dir = os.environ.get("CONFIG_DIR", "config")
    registry = load_strategy_registry(config_dir)
    return {
        sid: m.family
        for sid, m in registry.strategies.items()
        if m.enabled
    }


def is_family_active(family: str, now_et: datetime, schedules: dict) -> bool:
    """Check if a family should be monitored right now.

    Considers: (a) trading day for this asset class, (b) within active window,
    (c) on half-days, clamp active_end to market close + 5 min.
    """
    sched = schedules.get(family)
    if sched is None:
        return False

    asset_class = _ASSET_CLASS_MAP.get(sched["asset_class"], AssetClass.EQUITY)
    today = now_et.date()

    if not _calendar.is_trading_day(today, asset_class):
        return False

    active_start = _parse_time(sched["active_start"])
    active_end = _parse_time(sched["active_end"])

    # On half-days, clamp active_end to market close + 5 min
    if _calendar.is_half_day(today, asset_class):
        close = _calendar.market_close_time_et(today, asset_class)
        close_dt = datetime.combine(today, close) + timedelta(minutes=5)
        clamped = close_dt.time()
        if clamped < active_end:
            active_end = clamped

    now_time = now_et.time()
    return active_start <= now_time <= active_end


def _parse_time(s: str) -> time:
    parts = s.split(":")
    return time(int(parts[0]), int(parts[1]))
