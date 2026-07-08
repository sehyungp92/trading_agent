from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any


_RUNTIME_BARS_ATTR = "_idle_market_runtime_bars"
_RUNTIME_SYMBOL_ATTR = "_idle_market_runtime_symbol"
_RUNTIME_TIMEFRAME_ATTR = "_idle_market_runtime_timeframe"


IDLE_MARKET_OBSERVATION_TRIGGER_CODES = {
    "AWAITING_DATA",
    "CIRCUIT_BREAKER",
    "NO_SIGNAL",
    "OUTSIDE_RTH",
    "SIGNAL_FILTERED",
}


def remember_idle_market_bars(
    engine: Any,
    bars: Sequence[Any],
    *,
    symbol: str = "",
    timeframe: str = "",
) -> None:
    """Remember the latest bars consumed by an engine-owned live path."""

    if not bool(getattr(engine, "_idle_market_observation_enabled", False)):
        return
    consumed = list(bars or [])
    if not consumed:
        return
    setattr(engine, _RUNTIME_BARS_ATTR, consumed)
    setattr(engine, _RUNTIME_SYMBOL_ATTR, str(symbol or ""))
    setattr(engine, _RUNTIME_TIMEFRAME_ATTR, str(timeframe or ""))


def maybe_record_idle_market_observation(
    engine: Any,
    code: str,
    *,
    strategy_id: str,
    build_core_state: Callable[[], Any],
    apply_core_state: Callable[[Any], None],
    on_bar: Callable[..., tuple[Any, Sequence[Any], Sequence[Any]]],
    default_symbol: str = "",
    default_timeframe: str = "",
) -> bool:
    """Record an idle observation from bars consumed by the live path."""

    if str(code).upper() not in IDLE_MARKET_OBSERVATION_TRIGGER_CODES:
        return False
    if not bool(getattr(engine, "_idle_market_observation_enabled", False)):
        return False
    bars = list(getattr(engine, _RUNTIME_BARS_ATTR, []) or [])
    if not bars:
        return False
    timestamp = _latest_bar_timestamp(bars)
    if timestamp is None:
        return False
    symbol = str(getattr(engine, _RUNTIME_SYMBOL_ATTR, "") or default_symbol)
    timeframe = str(getattr(engine, _RUNTIME_TIMEFRAME_ATTR, "") or default_timeframe)
    next_state, actions, _events = on_bar(
        build_core_state(),
        bar_ts=timestamp,
        idle_market_bars=bars,
        idle_market_symbol=symbol,
        idle_market_timeframe=timeframe,
    )
    if actions:
        raise RuntimeError(
            f"{strategy_id} idle market observation generated actions: {actions}"
        )
    apply_core_state(next_state)
    _stamp_engine_bar_timestamp(engine, strategy_id, symbol, timestamp)
    return True


def _latest_bar_timestamp(bars: Sequence[Any]) -> datetime | None:
    if not bars:
        return None
    row = _bar_row(bars[-1])
    value = row.get("timestamp") or row.get("time")
    if value in (None, ""):
        return None
    return _coerce_datetime(value)


def idle_market_details(
    bars: Sequence[Any],
    *,
    symbol: str = "",
    timeframe: str = "",
) -> dict[str, Any]:
    """Return a deterministic, strategy-core observation from consumed bars."""

    rows = [_bar_row(bar) for bar in bars]
    if not rows:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "bar_count": 0,
            "first_bar_ts": "",
            "last_bar_ts": "",
            "last_ohlcv": {},
            "close_delta": 0,
            "range_sum": 0,
            "volume_sum": 0,
        }
    first = rows[0]
    last = rows[-1]
    return {
        "symbol": symbol or str(last.get("symbol", "")),
        "timeframe": timeframe or str(last.get("timeframe", "")),
        "bar_count": len(rows),
        "first_bar_ts": _dt_iso(first.get("timestamp") or first.get("time")),
        "last_bar_ts": _dt_iso(last.get("timestamp") or last.get("time")),
        "last_ohlcv": {
            "open": _number(last.get("open")),
            "high": _number(last.get("high")),
            "low": _number(last.get("low")),
            "close": _number(last.get("close")),
            "volume": _number(last.get("volume")),
        },
        "close_delta": _number(_number(last.get("close")) - _number(first.get("open"))),
        "range_sum": _number(sum(_number(row.get("high")) - _number(row.get("low")) for row in rows)),
        "volume_sum": _number(sum(_number(row.get("volume")) for row in rows)),
    }


def _bar_row(bar: Any) -> dict[str, Any]:
    if isinstance(bar, Mapping):
        return dict(bar)
    return {
        "symbol": getattr(bar, "symbol", ""),
        "timeframe": getattr(bar, "timeframe", ""),
        "timestamp": (
            getattr(bar, "timestamp", None)
            or getattr(bar, "time", None)
            or getattr(bar, "end_time", None)
            or getattr(bar, "date", None)
        ),
        "open": getattr(bar, "open", 0.0),
        "high": getattr(bar, "high", 0.0),
        "low": getattr(bar, "low", 0.0),
        "close": getattr(bar, "close", 0.0),
        "volume": getattr(bar, "volume", 0.0),
    }


def _dt_iso(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, datetime):
        ts = value
    else:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        ts = value
    else:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _stamp_engine_bar_timestamp(
    engine: Any,
    strategy_id: str,
    symbol: str,
    timestamp: datetime,
) -> None:
    if hasattr(engine, "_last_bar_ts"):
        engine._last_bar_ts = timestamp
    symbol_ts = getattr(engine, "_symbol_last_bar_ts", None)
    if isinstance(symbol_ts, dict):
        symbol_ts[strategy_id] = timestamp
        if symbol:
            symbol_ts[symbol] = timestamp
    bar_ts_by_symbol = getattr(engine, "_bar_ts_by_symbol", None)
    if isinstance(bar_ts_by_symbol, dict) and symbol:
        bar_ts_by_symbol[symbol] = timestamp


def _number(value: Any) -> int | float:
    number = round(float(value or 0.0), 6)
    return int(number) if number.is_integer() else number
