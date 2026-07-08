from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Any

from strategies.core.idle_market import remember_idle_market_bars
from tests.integration.parity.live_layer2 import compact_engine_state as _compact_engine_state
from tests.integration.parity.live_oms import settle_callbacks as _settle_callbacks
from tests.integration.parity.source_inputs import (
    IDLE_MARKET_INPUT_ARTIFACT_KEYS,
    bar_objects,
    iaric_quote,
    idle_market_input,
    parse_time,
    strategy_ids,
)


_MISSING_IDLE_MARKET_OBSERVATION = object()


async def _drive_idle_market_children(
    fixture: Mapping[str, Any],
    engines: Mapping[str, Any],
) -> None:
    for strategy_id in strategy_ids(fixture):
        if strategy_id not in IDLE_MARKET_INPUT_ARTIFACT_KEYS:
            continue
        engine = engines.get(strategy_id)
        if engine is None:
            raise AssertionError(f"configured idle child has no live engine: {strategy_id}")
        await _apply_idle_market_input(engine, strategy_id, idle_market_input(fixture, strategy_id))


async def _apply_idle_market_input(engine: Any, strategy_id: str, market_input: Mapping[str, Any]) -> None:
    before = _engine_order_count(engine, strategy_id)
    if strategy_id == "NQDTC_v2.1":
        await _run_live_cycle_with_market_input(
            engine,
            strategy_id,
            market_input,
            cycle="_on_5m_close",
            seed=_seed_nqdtc_market_input,
        )
    elif strategy_id == "VdubusNQ_v4":
        await _run_live_cycle_with_market_input(
            engine,
            strategy_id,
            market_input,
            cycle="_on_15m_close",
            seed=_seed_vdub_market_input,
        )
    elif strategy_id == "DownturnDominator_v1":
        await _run_live_cycle_with_market_input(
            engine,
            strategy_id,
            market_input,
            cycle="_on_5m_close",
            seed=_seed_downturn_market_input,
        )
    elif strategy_id == "ALCB_v1":
        restore_idle = _enable_idle_market_observation(engine)
        try:
            for row in market_input.get("bars", []) or []:
                symbol = str(row.get("symbol", market_input.get("symbol", ""))).upper()
                engine.on_quote(symbol, iaric_quote({"artifacts": {"iaric": {"quotes": {}}}, "bars": [row], "clock_start": row.get("timestamp") or row.get("time")}, symbol))
                for bar in _minute_bars_from_rows([row], symbol):
                    engine.on_bar(symbol, bar)
                    await _settle_callbacks()
        finally:
            restore_idle()
    elif strategy_id == "ATRSS":
        await _run_live_cycle_with_market_input(
            engine,
            strategy_id,
            market_input,
            cycle="_hourly_cycle",
            seed=_seed_swing_fetchers,
        )
    elif strategy_id == "AKC_HELIX":
        await _run_live_cycle_with_market_input(
            engine,
            strategy_id,
            market_input,
            cycle="_hourly_cycle",
            seed=_seed_swing_fetchers,
        )
    after = _engine_order_count(engine, strategy_id)
    if after > before:
        raise AssertionError(f"{strategy_id} idle market input generated {after - before} live order(s)")
    _assert_idle_market_observed(engine, strategy_id)


def _assert_idle_market_observed(engine: Any, strategy_id: str) -> None:
    state = _compact_engine_state(engine, strategy_id)
    details = state.get("last_decision_details", {}) or {}
    if state.get("last_decision_code") != "IDLE_MARKET_OBSERVED" or not details.get("bar_count"):
        raise AssertionError(
            f"{strategy_id} live idle cycle did not record IDLE_MARKET_OBSERVED from consumed bars"
        )


def _enable_idle_market_observation(engine: Any):
    managed_attrs = (
        "_idle_market_observation_enabled",
        "_idle_market_runtime_bars",
        "_idle_market_runtime_symbol",
        "_idle_market_runtime_timeframe",
    )
    originals = {
        attr: getattr(engine, attr, _MISSING_IDLE_MARKET_OBSERVATION)
        for attr in managed_attrs
    }
    engine._idle_market_observation_enabled = True
    for attr in managed_attrs[1:]:
        try:
            delattr(engine, attr)
        except AttributeError:
            pass

    def _restore() -> None:
        for attr, value in originals.items():
            if value is _MISSING_IDLE_MARKET_OBSERVATION:
                try:
                    delattr(engine, attr)
                except AttributeError:
                    pass
            else:
                setattr(engine, attr, value)

    return _restore


async def _run_live_cycle_with_market_input(
    engine: Any,
    strategy_id: str,
    market_input: Mapping[str, Any],
    *,
    cycle: str,
    seed,
) -> None:
    restore = seed(engine, market_input)
    restore_idle = _enable_idle_market_observation(engine)
    original_refresh = getattr(engine, "_refresh_equity", None)

    async def _refresh_equity_noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    if original_refresh is not None:
        engine._refresh_equity = _refresh_equity_noop
    try:
        runner = getattr(engine, cycle)
        await runner()
    finally:
        if original_refresh is not None:
            engine._refresh_equity = original_refresh
        restore_idle()
        restore()


def _seed_nqdtc_market_input(engine: Any, market_input: Mapping[str, Any]):
    from strategies.momentum.nqdtc.engine import Session, _filter_bars_by_session

    original_fetch = getattr(engine, "_fetch_bars")

    async def _fetch_bars(*_args: Any, **_kwargs: Any) -> None:
        grouped = market_input.get("bars_by_timeframe", {}) or {}
        bars_5m = bar_objects(grouped.get("5m", market_input.get("bars", [])))
        bars_15m = bar_objects(grouped.get("15m", grouped.get("5m", market_input.get("bars", []))))
        bars_30m = bar_objects(grouped.get("30m", grouped.get("5m", market_input.get("bars", []))))
        bars_1h = bar_objects(grouped.get("1h", grouped.get("30m", market_input.get("bars", []))))
        bars_4h = bar_objects(grouped.get("4h", grouped.get("1h", market_input.get("bars", []))))
        bars_d = bar_objects(grouped.get("1d", grouped.get("daily", market_input.get("bars", []))))
        if bars_5m:
            remember_idle_market_bars(
                engine,
                bars_5m,
                symbol=str(market_input.get("symbol", "")),
                timeframe="5m",
            )
            engine._bars_5m = engine._bars_to_arrays(bars_5m)
        if bars_15m:
            engine._bars_15m = engine._bars_to_arrays(bars_15m)
        if bars_30m:
            engine._raw_bars_30m = bars_30m
            engine._bars_30m = engine._bars_to_arrays(bars_30m)
            for session in (Session.ETH, Session.RTH):
                filtered = _filter_bars_by_session(bars_30m, session) or bars_30m
                engine._bars_30m_session[session] = engine._bars_to_arrays(filtered)
        if bars_1h:
            engine._bars_1h = engine._bars_to_arrays(bars_1h)
        if bars_4h:
            engine._bars_4h = engine._bars_to_arrays(bars_4h)
        if bars_d:
            engine._bars_daily = engine._bars_to_arrays(bars_d)

    engine._fetch_bars = _fetch_bars
    return lambda: setattr(engine, "_fetch_bars", original_fetch)


def _seed_downturn_market_input(engine: Any, market_input: Mapping[str, Any]):
    original_fetch = getattr(engine, "_fetch_bars")

    async def _fetch_bars(*_args: Any, **_kwargs: Any) -> None:
        grouped = market_input.get("bars_by_timeframe", {}) or {}
        for tf, attr in (
            ("5m", "_bars_5m"),
            ("15m", "_bars_15m"),
            ("30m", "_bars_30m"),
            ("1h", "_bars_1h"),
            ("4h", "_bars_4h"),
            ("1d", "_bars_daily"),
        ):
            rows = grouped.get(tf) or grouped.get("daily" if tf == "1d" else "") or market_input.get("bars", [])
            bars = bar_objects(rows)
            if bars:
                if tf == "5m":
                    remember_idle_market_bars(
                        engine,
                        bars,
                        symbol=str(market_input.get("symbol", "")),
                        timeframe="5m",
                    )
                setattr(engine, attr, engine._bars_to_arrays(bars))

    engine._fetch_bars = _fetch_bars
    return lambda: setattr(engine, "_fetch_bars", original_fetch)


def _seed_vdub_market_input(engine: Any, market_input: Mapping[str, Any]):
    import numpy as np

    original_fetch = getattr(engine, "_fetch_bars")

    def _assign_arrays(
        rows: Sequence[Mapping[str, Any]],
        close_attr: str,
        high_attr: str,
        low_attr: str,
        vol_attr: str,
        time_attr: str | None = None,
        *,
        timeframe: str = "",
    ) -> None:
        bars = bar_objects(rows)
        if not bars:
            return
        if timeframe == "15m":
            remember_idle_market_bars(
                engine,
                bars,
                symbol=str(market_input.get("symbol", "")),
                timeframe="15m",
            )
        setattr(engine, close_attr, np.array([bar.close for bar in bars], dtype=float))
        setattr(engine, high_attr, np.array([bar.high for bar in bars], dtype=float))
        setattr(engine, low_attr, np.array([bar.low for bar in bars], dtype=float))
        setattr(engine, vol_attr, np.array([bar.volume for bar in bars], dtype=float))
        if time_attr:
            setattr(engine, time_attr, [bar.date for bar in bars])

    async def _fetch_bars(*_args: Any, **_kwargs: Any) -> None:
        grouped = market_input.get("bars_by_timeframe", {}) or {}
        _assign_arrays(grouped.get("15m", market_input.get("bars", [])), "_c15", "_h15", "_l15", "_v15", "_t15", timeframe="15m")
        _assign_arrays(grouped.get("1h", market_input.get("bars", [])), "_c1h", "_h1h", "_l1h", "_v1h", "_t1h", timeframe="1h")
        daily = grouped.get("1d", grouped.get("daily", market_input.get("bars", [])))
        bars = bar_objects(daily)
        if bars:
            engine._es_c = np.array([bar.close for bar in bars], dtype=float)
            engine._es_h = np.array([bar.high for bar in bars], dtype=float)
            engine._es_l = np.array([bar.low for bar in bars], dtype=float)
            engine._t_nq_daily = [bar.date for bar in bars]

    engine._fetch_bars = _fetch_bars
    return lambda: setattr(engine, "_fetch_bars", original_fetch)


def _seed_swing_fetchers(engine: Any, market_input: Mapping[str, Any]):
    import numpy as np

    grouped = market_input.get("bars_by_timeframe", {}) or {}
    target_symbol = str(market_input.get("symbol", "")).upper()
    if hasattr(engine, "_fetch_bars") and not hasattr(engine, "_fetch_daily_bars"):
        original_fetch = getattr(engine, "_fetch_bars")

        async def _fetch_bars(sym: str, _cfg: Any, bar_size: str, duration: str, request_kind: str = "recurring"):
            tf = _bar_size_to_timeframe(bar_size)
            rows = grouped.get(tf, market_input.get("bars", []))
            filtered = [row for row in rows if str(row.get("symbol", sym)).upper() == str(sym).upper()] or rows
            bars = bar_objects(filtered)
            if tf == "1h" and bars and (not target_symbol or str(sym).upper() == target_symbol):
                remember_idle_market_bars(engine, bars, symbol=sym, timeframe="1h")
            return bars

        engine._fetch_bars = _fetch_bars
        return lambda: setattr(engine, "_fetch_bars", original_fetch)

    original_daily = getattr(engine, "_fetch_daily_bars")
    original_hourly = getattr(engine, "_fetch_hourly_bars")

    async def _fetch_daily_bars(sym: str, _cfg: Any, request_kind: str = "recurring"):
        rows = grouped.get("1d", grouped.get("daily", market_input.get("bars", [])))
        bars = bar_objects([row for row in rows if str(row.get("symbol", sym)).upper() == str(sym).upper()] or rows)
        if not bars:
            return None, None, None, None
        return (
            np.array([bar.close for bar in bars], dtype=float),
            np.array([bar.high for bar in bars], dtype=float),
            np.array([bar.low for bar in bars], dtype=float),
            str(bars[-1].date.date()),
        )

    async def _fetch_hourly_bars(sym: str, _cfg: Any, request_kind: str = "recurring"):
        rows = grouped.get("1h", market_input.get("bars", []))
        bars = bar_objects([row for row in rows if str(row.get("symbol", sym)).upper() == str(sym).upper()] or rows)
        if not bars:
            return None, None, None, None
        if not target_symbol or str(sym).upper() == target_symbol:
            remember_idle_market_bars(engine, bars, symbol=sym, timeframe="1h")
        return (
            np.array([bar.close for bar in bars], dtype=float),
            np.array([bar.high for bar in bars], dtype=float),
            np.array([bar.low for bar in bars], dtype=float),
            np.array([bar.open for bar in bars], dtype=float),
        )

    engine._fetch_daily_bars = _fetch_daily_bars
    engine._fetch_hourly_bars = _fetch_hourly_bars

    def _restore() -> None:
        engine._fetch_daily_bars = original_daily
        engine._fetch_hourly_bars = original_hourly

    return _restore


def _bar_size_to_timeframe(bar_size: str) -> str:
    text = str(bar_size).lower()
    if "day" in text:
        return "1d"
    if "hour" in text:
        return "4h" if text.startswith("4") else "1h"
    if "30" in text:
        return "30m"
    if "15" in text:
        return "15m"
    if "5" in text:
        return "5m"
    return "1h"


def _minute_bars_from_rows(rows: Sequence[Mapping[str, Any]], symbol: str) -> list[Any]:
    from strategies.stock.alcb.models import Bar

    output = []
    for row in rows:
        end = parse_time(row.get("timestamp") or row.get("time"))
        start = end - timedelta(minutes=5)
        volume = float(row.get("volume", 0.0)) / 5.0
        for index in range(5):
            minute_start = start + timedelta(minutes=index)
            output.append(
                Bar(
                    symbol=symbol,
                    start_time=minute_start,
                    end_time=minute_start + timedelta(minutes=1),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=volume,
                )
            )
    return output


def _engine_order_count(engine: Any, strategy_id: str) -> int:
    repo = getattr(getattr(getattr(engine, "_oms", None), "_handler", None), "_repo", None)
    orders = getattr(repo, "_orders", {}) if repo is not None else {}
    return sum(1 for order in orders.values() if getattr(order, "strategy_id", "") == strategy_id)


drive_idle_market_children = _drive_idle_market_children
