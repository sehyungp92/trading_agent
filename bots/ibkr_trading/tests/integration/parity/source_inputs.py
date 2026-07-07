from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import numpy as np

from strategies.core.actions import SubmitEntry, SubmitProfitTarget, SubmitProtectiveStop
from strategies.swing._shared.etf_core import BarWindow


IDLE_MARKET_INPUT_ARTIFACT_KEYS = {
    "NQDTC_v2.1": "nqdtc",
    "VdubusNQ_v4": "vdub",
    "DownturnDominator_v1": "downturn",
    "ALCB_v1": "alcb",
    "ATRSS": "atrss",
    "AKC_HELIX": "akc_helix",
}


def parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        ts = value
    else:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def strategy_ids(fixture: Mapping[str, Any]) -> list[str]:
    ids = [
        str(item["id"])
        for item in (fixture.get("family_config", {}) or {}).get("strategies", [])
        if item.get("id")
    ]
    configured = (fixture.get("strategy_config", {}) or {}).get("strategy_id")
    if configured and str(configured) not in ids:
        ids.append(str(configured))
    return ids


def idle_market_input(fixture: Mapping[str, Any], strategy_id: str) -> dict[str, Any]:
    key = IDLE_MARKET_INPUT_ARTIFACT_KEYS.get(strategy_id, strategy_id)
    raw = ((fixture.get("artifacts", {}) or {}).get(key, {}) or {}).get("idle_market_input", {})
    if not isinstance(raw, Mapping):
        raw = {}
    symbol = str(raw.get("symbol", _default_idle_symbol(fixture)))
    timeframe = str(raw.get("timeframe", "5m"))
    bars = _sort_bar_rows(raw.get("bars", []) or [])
    if not bars:
        bars = source_bars(fixture, symbol, timeframe)
    timestamp = raw.get("timestamp")
    if not timestamp and bars:
        timestamp = bars[-1].get("timestamp") or bars[-1].get("time")
    timestamp = timestamp or fixture.get("clock_start")
    return {
        "strategy_id": strategy_id,
        "artifact_key": key,
        "symbol": symbol,
        "timeframe": timeframe,
        "bars": bars,
        "bars_by_timeframe": _idle_bars_by_timeframe(fixture, raw, symbol, timeframe, bars),
        "timestamp": parse_time(timestamp),
        "reason": str(raw.get("reason", "fixture_idle_market_input")),
    }


def _idle_bars_by_timeframe(
    fixture: Mapping[str, Any],
    raw: Mapping[str, Any],
    symbol: str,
    timeframe: str,
    primary_bars: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    by_tf = raw.get("bars_by_timeframe", {}) or {}
    if isinstance(by_tf, Mapping):
        for tf, rows in by_tf.items():
            grouped[str(tf).lower()] = _sort_bar_rows(rows or [])
    grouped.setdefault(timeframe.lower(), _sort_bar_rows(primary_bars))
    for tf in ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "daily"):
        if tf not in grouped:
            rows = source_bars(fixture, symbol, "1d" if tf == "daily" else tf)
            if rows:
                grouped[tf] = [dict(row) for row in rows]
    return grouped


def _sort_bar_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows or []),
        key=_bar_sort_key,
    )


def _bar_sort_key(row: Mapping[str, Any]) -> str:
    value = row.get("timestamp") or row.get("time")
    return parse_time(value).isoformat() if value else ""


def bar_objects(rows: Sequence[Mapping[str, Any]]) -> list[Any]:
    objects = []
    for row in rows:
        ts = parse_time(row.get("timestamp") or row.get("time"))
        objects.append(
            SimpleNamespace(
                date=ts,
                time=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0)),
                barCount=int(row.get("barCount", row.get("bar_count", 0)) or 0),
                average=float(row.get("average", row.get("close", 0.0)) or 0.0),
            )
        )
    return objects


def _default_idle_symbol(fixture: Mapping[str, Any]) -> str:
    for row in fixture.get("bars", []) or []:
        symbol = str(row.get("symbol", ""))
        if symbol:
            return symbol
    instruments = fixture.get("instruments", []) or []
    if instruments:
        return str(instruments[0].get("symbol", ""))
    return ""


def family_resolver(fixture: Mapping[str, Any]):
    family_cfg = fixture.get("family_config", {}) or {}
    default_family = str(family_cfg.get("family", fixture.get("family", "")))
    mapping = {
        str(item.get("id")): str(item.get("family", default_family))
        for item in family_cfg.get("strategies", [])
        if item.get("id")
    }
    configured = (fixture.get("strategy_config", {}) or {}).get("strategy_id")
    if configured:
        mapping.setdefault(str(configured), str((fixture.get("strategy_config", {}) or {}).get("family", default_family)))
    return lambda strategy_id: mapping.get(strategy_id, default_family)


def instrument_ticks(fixture: Mapping[str, Any]) -> dict[str, float]:
    ticks: dict[str, float] = {}
    for item in fixture.get("instruments", []):
        tick = float(item.get("tick_size", 0.01))
        ticks[str(item.get("symbol"))] = tick
        ticks[str(item.get("trade_symbol", item.get("symbol")))] = tick
    return ticks


def point_value(fixture: Mapping[str, Any], symbol: str) -> float:
    for item in fixture.get("instruments", []):
        if symbol in {str(item.get("symbol")), str(item.get("trade_symbol", item.get("symbol")))}:
            return float(item.get("point_value", item.get("multiplier", 1.0)))
    return 1.0


def source_bars(fixture: Mapping[str, Any], symbol: str, timeframe: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in fixture.get("bars", []):
        if str(row.get("symbol", "")).upper() == symbol.upper() and str(row.get("timeframe", "")).lower() == timeframe.lower():
            rows.append(dict(row))
    htf = fixture.get("higher_timeframe_bars", {}) or {}
    for row in htf.get(symbol, []) if isinstance(htf, Mapping) else []:
        if str(row.get("timeframe", "")).lower() == timeframe.lower():
            item = dict(row)
            item.setdefault("symbol", symbol)
            rows.append(item)
    return sorted(rows, key=lambda item: str(item.get("timestamp") or item.get("time") or ""))


def tpc_symbol_config(fixture: Mapping[str, Any], symbol: str):
    from strategies.swing.tpc.config import TPCSymbolConfig

    cfg = TPCSymbolConfig(symbol=symbol)
    overrides = dict((fixture.get("strategy_config", {}) or {}).get("config_overrides", {}) or {})
    merged: dict[str, Any] = {}
    merged.update(overrides.get("all", {}) or {})
    merged.update(overrides.get(symbol, {}) or {})
    if not merged:
        return cfg
    field_names = {field.name for field in fields(TPCSymbolConfig)}
    cleaned = {
        key: tuple(value) if key.endswith("_et") and isinstance(value, list) else value
        for key, value in merged.items()
        if key in field_names
    }
    return replace(cfg, **cleaned)


def tpc_bar_input(fixture: Mapping[str, Any], symbol: str):
    from strategies.swing.tpc.core.state import TPCBarInput

    bars_15m = _bar_window(source_bars(fixture, symbol, "15m"))
    bar_15m = bars_15m.last if bars_15m is not None else None
    indicators = dict(((fixture.get("artifacts", {}) or {}).get("tpc", {}) or {}).get("indicators", {}) or {})
    return TPCBarInput(
        symbol=symbol,
        bar_15m=bar_15m,
        bars_15m=bars_15m,
        bars_30m=_bar_window(source_bars(fixture, symbol, "30m")),
        bars_1h=_bar_window(source_bars(fixture, symbol, "1h")),
        bars_4h=_bar_window(source_bars(fixture, symbol, "4h")),
        bars_daily=_bar_window(source_bars(fixture, symbol, "1d")),
        indicators=indicators,
        equity=float((fixture.get("account_state", {}) or {}).get("equity", 100_000.0)),
        timestamp=bar_15m.timestamp if bar_15m is not None else parse_time(fixture["clock_start"]),
    )


def nq_daily_context(fixture: Mapping[str, Any]):
    from strategies.momentum.nq_regime.core.levels import KeyLevels

    artifacts = fixture.get("artifacts", {}) or {}
    payload = dict((artifacts.get("nq_regime", {}) or {}).get("daily_context", {}) or artifacts.get("daily_levels", {}) or {})
    return KeyLevels(
        pdh=float(payload.get("pdh", 0.0)),
        pdl=float(payload.get("pdl", 0.0)),
        pdm=float(payload.get("pdm", 0.0)),
        weekly_high=float(payload.get("weekly_high", 0.0)),
        weekly_low=float(payload.get("weekly_low", 0.0)),
    )


def nq_bar_data(row: Mapping[str, Any]):
    from strategies.momentum.nq_regime.core.state import BarData

    return BarData(
        ts=parse_time(row.get("timestamp") or row.get("time")),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row.get("volume", 0.0)),
        vwap=float(row["vwap"]) if row.get("vwap") is not None else None,
    )


def nq_live_context(fixture: Mapping[str, Any]) -> dict[str, Any]:
    return dict(((fixture.get("artifacts", {}) or {}).get("nq_regime", {}) or {}).get("live_context", {}) or {})


def iaric_artifact(fixture: Mapping[str, Any]):
    from strategies.stock.iaric.models import RegimeSnapshot, WatchlistArtifact, WatchlistItem

    clock = parse_time(fixture["clock_start"])
    artifact_payload = fixture.get("artifacts", {}) or {}
    iaric_payload = artifact_payload.get("iaric", {}) or artifact_payload
    items: list[WatchlistItem] = []
    for raw in iaric_payload.get("watchlist", []) or []:
        symbol = str(raw["symbol"]).upper()
        price = float(raw.get("price", _last_close(fixture, symbol, 0.0)))
        items.append(
            WatchlistItem(
                symbol=symbol,
                exchange=str(raw.get("exchange", "SMART")),
                primary_exchange=str(raw.get("primary_exchange", "NASDAQ")),
                currency=str(raw.get("currency", "USD")),
                tick_size=float(raw.get("tick_size", 0.01)),
                point_value=float(raw.get("point_value", point_value(fixture, symbol))),
                sector=str(raw.get("sector", "Technology")),
                regime_score=float(raw.get("regime_score", 0.75)),
                regime_tier=str(raw.get("regime_tier", "B")),
                regime_risk_multiplier=float(raw.get("regime_risk_multiplier", 1.0)),
                sector_score=float(raw.get("sector_score", 0.7)),
                sector_rank_weight=float(raw.get("sector_rank_weight", 1.0)),
                sponsorship_score=float(raw.get("sponsorship_score", 0.7)),
                sponsorship_state=str(raw.get("sponsorship_state", "ACCUMULATION")),
                persistence=float(raw.get("persistence", 0.7)),
                intensity_z=float(raw.get("intensity_z", 1.0)),
                accel_z=float(raw.get("accel_z", 0.5)),
                rs_percentile=float(raw.get("rs_percentile", 80.0)),
                leader_pass=bool(raw.get("leader_pass", True)),
                trend_pass=bool(raw.get("trend_pass", True)),
                trend_strength=float(raw.get("trend_strength", 0.7)),
                earnings_risk_flag=bool(raw.get("earnings_risk_flag", False)),
                blacklist_flag=bool(raw.get("blacklist_flag", False)),
                anchor_date=clock.date(),
                anchor_type=str(raw.get("anchor_type", "AVWAP")),
                acceptance_pass=bool(raw.get("acceptance_pass", True)),
                avwap_ref=float(raw.get("avwap_ref", price)),
                avwap_band_lower=float(raw.get("avwap_band_lower", price * 0.99)),
                avwap_band_upper=float(raw.get("avwap_band_upper", price * 1.01)),
                daily_atr_estimate=float(raw.get("daily_atr", raw.get("daily_atr_estimate", 1.0))),
                intraday_atr_seed=float(raw.get("intraday_atr_seed", raw.get("daily_atr", 1.0))),
                daily_rank=float(raw.get("daily_rank", 1.0)),
                tradable_flag=bool(raw.get("tradable_flag", True)),
                conviction_bucket=str(raw.get("conviction_bucket", "GREEN")),
                conviction_multiplier=float(raw.get("conviction_multiplier", 1.0)),
                recommended_risk_r=float(raw.get("recommended_risk_r", 1.0)),
                average_30m_volume=float(raw.get("average_30m_volume", 600_000.0)),
                expected_5m_volume=float(raw.get("expected_5m_volume", 100_000.0)),
                entry_gap_pct=float(raw.get("entry_gap_pct", 0.0)),
                daily_signal_score=float(raw.get("daily_signal_score", 82.0)),
                trigger_types=list(raw.get("trigger_types", ["OPENING_RECLAIM"])),
                trigger_tier=str(raw.get("trigger_tier", "STANDARD")),
                trend_tier=str(raw.get("trend_tier", "STRONG")),
                rescue_flow_candidate=bool(raw.get("rescue_flow_candidate", False)),
                sizing_mult=float(raw.get("sizing_mult", 1.0)),
                cdd_value=int(raw.get("cdd_value", 0)),
                ema10_daily=float(raw.get("ema10_daily", 0.0)),
                rsi14_daily=float(raw.get("rsi14_daily", 0.0)),
            )
        )
    regime_payload = dict(iaric_payload.get("regime", {}) or {})
    regime = RegimeSnapshot(
        score=float(regime_payload.get("score", 0.75)),
        tier=str(regime_payload.get("tier", "B")),
        risk_multiplier=float(regime_payload.get("risk_multiplier", 1.0)),
        price_ok=bool(regime_payload.get("price_ok", True)),
        breadth_ok=bool(regime_payload.get("breadth_ok", True)),
        vol_ok=bool(regime_payload.get("vol_ok", True)),
        credit_ok=bool(regime_payload.get("credit_ok", True)),
    )
    return WatchlistArtifact(trade_date=clock.date(), generated_at=clock, regime=regime, items=items, tradable=items, overflow=[])


def iaric_state_snapshot(fixture: Mapping[str, Any], strategy_id: str = "IARIC_v1"):
    from strategies.stock.iaric.artifact_store import coerce_intraday_state_snapshot
    from strategies.stock.iaric.models import IntradayStateSnapshot

    payload = (fixture.get("initial_strategy_state", {}) or {}).get(strategy_id)
    if payload:
        return coerce_intraday_state_snapshot(payload)
    clock = parse_time(fixture["clock_start"])
    return IntradayStateSnapshot(
        trade_date=clock.date(),
        saved_at=clock,
        symbols=[],
        last_decision_code="IDLE",
        meta={"active_symbols": []},
    )


def iaric_quote(fixture: Mapping[str, Any], symbol: str):
    from strategies.stock.iaric.models import QuoteSnapshot

    quote = dict(((fixture.get("artifacts", {}) or {}).get("iaric", {}) or {}).get("quotes", {}).get(symbol, {}) or {})
    last = float(quote.get("last", _last_close(fixture, symbol, 0.0)))
    bid = float(quote.get("bid", last - 0.01))
    ask = float(quote.get("ask", last + 0.01))
    return QuoteSnapshot(
        ts=parse_time(quote.get("timestamp", fixture["clock_start"])),
        bid=bid,
        ask=ask,
        last=last,
        bid_size=float(quote.get("bid_size", 100.0)),
        ask_size=float(quote.get("ask_size", 100.0)),
        cumulative_volume=float(quote.get("cumulative_volume", 1_000_000.0)),
        cumulative_value=float(quote.get("cumulative_value", last * 1_000_000.0)),
        spread_pct=float(quote.get("spread_pct", (ask - bid) / max(last, 1e-9))),
    )


def iaric_minute_bars(fixture: Mapping[str, Any], symbol: str):
    from strategies.stock.iaric.models import Bar

    direct = source_bars(fixture, symbol, "1m")
    if direct:
        return [_stock_bar(row, minutes=1) for row in direct]
    bars_5m = source_bars(fixture, symbol, "5m")
    output = []
    for row in bars_5m:
        start = parse_time(row.get("timestamp") or row.get("time"))
        volume = float(row.get("volume", 0.0)) / 5.0
        prices = [
            (float(row["open"]), max(float(row["open"]), float(row["high"])), min(float(row["open"]), float(row["low"])), float(row["open"])),
            (float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])),
            (float(row["close"]), float(row["high"]), float(row["low"]), float(row["close"])),
            (float(row["close"]), float(row["high"]), float(row["low"]), float(row["close"])),
            (float(row["close"]), float(row["high"]), float(row["low"]), float(row["close"])),
        ]
        for index, (open_, high, low, close) in enumerate(prices):
            minute_start = start + timedelta(minutes=index)
            output.append(
                Bar(
                    symbol=symbol,
                    start_time=minute_start,
                    end_time=minute_start + timedelta(minutes=1),
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                )
            )
    return output


def overlay_rebalance_payload(fixture: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(((fixture.get("artifacts", {}) or {}).get("overlay_rebalance", {}) or {}))
    target_weights = dict(raw.get("target_weights", {}) or {})
    symbols = list(raw.get("symbols", []) or target_weights.keys())
    initial_overlay = ((fixture.get("initial_family_state", {}) or {}).get("overlay", {}) or {})
    if not symbols:
        symbols = [str(symbol) for symbol in (initial_overlay.get("starting_holdings", {}) or {})]
    daily_bars = {
        str(symbol): source_bars(fixture, str(symbol), "1d")
        for symbol in symbols
    }
    prices = dict(raw.get("prices", {}) or {})
    for symbol in symbols:
        if symbol not in prices:
            closes = [float(row["close"]) for row in daily_bars.get(str(symbol), [])]
            if not closes:
                closes = [
                    float(row["close"])
                    for row in fixture.get("bars", [])
                    if str(row.get("symbol", "")).upper() == str(symbol).upper()
                ]
            if closes:
                prices[str(symbol)] = closes[-1]
    return {
        "symbols": symbols,
        "target_weights": target_weights,
        "ema_overrides": {
            str(symbol): list(periods)
            for symbol, periods in (raw.get("ema_overrides", {}) or {}).items()
        },
        "signals": dict(raw.get("signals", {}) or {}),
        "daily_bars": daily_bars,
        "prices": prices,
        "starting_holdings": dict(initial_overlay.get("starting_holdings", {}) or {}),
        "rebalance_due": bool(initial_overlay.get("rebalance_due", bool(raw))),
        "rebalance_reason": str(raw.get("rebalance_reason", "fixture")),
        "timestamp": str(raw.get("timestamp", fixture.get("clock_start"))),
        "equity": float((fixture.get("account_state", {}) or {}).get("equity", 100_000.0)),
        "max_equity_pct": float(raw.get("max_equity_pct", 0.85)),
    }


def action_order_row(action: Any, strategy_id: str, family: str = "") -> dict[str, Any]:
    role = str(getattr(action, "role", "") or "")
    if isinstance(action, SubmitProtectiveStop):
        role = "STOP"
    elif isinstance(action, SubmitProfitTarget):
        role = "TP"
    elif isinstance(action, SubmitEntry):
        role = role or "ENTRY"
    order_type = str(
        getattr(action, "order_type", "")
        or ("LIMIT" if isinstance(action, SubmitProfitTarget) else "STOP" if isinstance(action, SubmitProtectiveStop) else "")
    )
    return {
        "strategy_id": strategy_id,
        "family": family,
        "symbol": str(getattr(action, "symbol", "")),
        "side": str(getattr(action, "side", "")).upper(),
        "qty": int(getattr(action, "qty", 0) or 0),
        "order_type": order_type.upper(),
        "tif": str(getattr(action, "tif", "DAY") or "DAY"),
        "limit_price": getattr(action, "limit_price", None),
        "stop_price": getattr(action, "stop_price", None),
        "role": role.upper(),
        "client_order_id": str(getattr(action, "client_order_id", "")),
    }


def plain(value: Any) -> Any:
    if is_dataclass(value):
        return {key: plain(val) for key, val in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): plain(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(item) for item in value]
    return value


def _bar_window(rows: Sequence[Mapping[str, Any]]) -> BarWindow | None:
    if not rows:
        return None
    opens = np.array([float(row["open"]) for row in rows], dtype=float)
    highs = np.array([float(row["high"]) for row in rows], dtype=float)
    lows = np.array([float(row["low"]) for row in rows], dtype=float)
    closes = np.array([float(row["close"]) for row in rows], dtype=float)
    volumes = np.array([float(row.get("volume", 0.0)) for row in rows], dtype=float)
    times = tuple(parse_time(row.get("timestamp") or row.get("time")) for row in rows)
    return BarWindow(opens=opens, highs=highs, lows=lows, closes=closes, volumes=volumes, times=times)


def _stock_bar(row: Mapping[str, Any], *, minutes: int):
    from strategies.stock.iaric.models import Bar

    start = parse_time(row.get("timestamp") or row.get("time"))
    return Bar(
        symbol=str(row["symbol"]).upper(),
        start_time=start,
        end_time=start + timedelta(minutes=minutes),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row.get("volume", 0.0)),
    )


def _last_close(fixture: Mapping[str, Any], symbol: str, default: float) -> float:
    rows = [row for row in fixture.get("bars", []) if str(row.get("symbol", "")).upper() == symbol.upper()]
    if not rows:
        return default
    return float(rows[-1].get("close", default))
