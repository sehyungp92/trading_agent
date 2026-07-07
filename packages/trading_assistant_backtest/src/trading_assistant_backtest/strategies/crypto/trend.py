"""Crypto trend strategy bridge backed by the production ``crypto_trader`` code.

This adapter is intentionally read-only. It imports a pinned local checkout of
the live repo and turns deterministic parity fixtures into normalized decision
trace events. In the first bridge, the backtest adapter delegates to the same
production decision kernel, which is the desired shared-kernel architecture.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from trading_assistant_backtest.contract_models import DecisionParityReport, MonthlyRunManifest
from trading_assistant_backtest.replay.decision_trace import DecisionTraceEvent
from trading_assistant_backtest.replay.parity import decision_parity_report_from_traces
from trading_assistant_backtest.strategies.live_clone import validate_pinned_head

PLUGIN_ID = "crypto-trend-v1"
DECISION_API_VERSION = "crypto_trader_trend_decision_api_v1"


def build_crypto_trend_decision_parity_report(
    manifest: MonthlyRunManifest,
    *,
    candidate_id: str,
    fixture_paths: Iterable[str | Path],
    live_repo_path: str | Path,
    live_repo_commit_sha: str = "",
    backtest_adapter_commit_sha: str = "",
) -> DecisionParityReport:
    """Build a decision parity report from deterministic crypto trend fixtures."""

    repo_path = Path(live_repo_path)
    if live_repo_commit_sha:
        checkout_errors = validate_pinned_head(repo_path, live_repo_commit_sha)
        if checkout_errors:
            raise ValueError("; ".join(checkout_errors))

    live_events: list[DecisionTraceEvent] = []
    adapter_events: list[DecisionTraceEvent] = []
    evidence_paths: list[str] = []
    for fixture_path in fixture_paths:
        path = Path(fixture_path)
        fixture = json.loads(path.read_text(encoding="utf-8"))
        evidence_paths.append(str(path))
        live_events.extend(live_decision_trace_from_fixture(fixture, live_repo_path=repo_path))
        adapter_events.extend(
            backtest_adapter_decision_trace_from_fixture(fixture, live_repo_path=repo_path)
        )

    return decision_parity_report_from_traces(
        manifest,
        candidate_id=candidate_id,
        live_events=live_events,
        adapter_events=adapter_events,
        evidence_paths=evidence_paths,
        live_repo_commit_sha=live_repo_commit_sha,
        backtest_adapter_commit_sha=backtest_adapter_commit_sha,
    )


def live_decision_trace_from_fixture(
    fixture: Mapping[str, Any],
    *,
    live_repo_path: str | Path,
) -> list[DecisionTraceEvent]:
    """Run the fixture through the live repo decision API."""

    return decision_trace_from_fixture(fixture, live_repo_path=live_repo_path)


def backtest_adapter_decision_trace_from_fixture(
    fixture: Mapping[str, Any],
    *,
    live_repo_path: str | Path,
) -> list[DecisionTraceEvent]:
    """Run the fixture through the backtest adapter's shared production kernel."""

    return decision_trace_from_fixture(fixture, live_repo_path=live_repo_path)


def decision_trace_from_fixture(
    fixture: Mapping[str, Any],
    *,
    live_repo_path: str | Path,
) -> list[DecisionTraceEvent]:
    """Run one fixture through the pinned production trend decision components."""

    with _crypto_trader_import_path(Path(live_repo_path)):
        from crypto_trader.core.models import Bar, Position, SetupGrade, Side, TimeFrame
        from crypto_trader.core.runtime_types import DecisionContext, OrderIntent
        from crypto_trader.strategy.trend.config import TrendConfig
        from crypto_trader.strategy.trend.confirmation import TriggerResult
        from crypto_trader.strategy.trend.entry import EntryGenerator
        from crypto_trader.strategy.trend.exits import ExitManager
        from crypto_trader.strategy.trend.setup import TrendSetupResult
        from crypto_trader.strategy.trend.sizing import SizingResult
        from crypto_trader.strategy.trend.stops import StopPlacer

        cfg = (
            TrendConfig.from_dict(dict(fixture.get("config") or {}))
            if fixture.get("config")
            else TrendConfig()
        )
        fixture_id = str(fixture.get("fixture_id") or "crypto-trend-fixture")
        symbol = str(fixture.get("symbol") or "BTC").upper()
        timeframe = TimeFrame.from_interval(str(fixture.get("timeframe") or "1h"))
        ts = _parse_ts(fixture.get("timestamp"))
        bar = _bar(Bar, TimeFrame, fixture, symbol=symbol, timeframe=timeframe, ts=ts)
        direction = Side(str(fixture.get("direction") or "LONG").upper())
        setup = _setup(TrendSetupResult, SetupGrade, Side, fixture, direction)
        sizing = _sizing(SizingResult, fixture)
        trigger = _trigger(TriggerResult, fixture)
        order_id = str(fixture.get("order_id") or f"{fixture_id}:entry")

        entry_order = EntryGenerator(cfg.entry).generate(
            bar,
            direction,
            sizing.qty,
            sizing,
            setup,
            trigger,
            symbol,
            order_id,
            is_reentry=bool(fixture.get("is_reentry", False)),
        )
        stop_series = [
            _bar(Bar, TimeFrame, item, symbol=symbol, timeframe=timeframe, ts=ts)
            for item in _stop_bars(fixture)
        ]
        stop_price = StopPlacer(cfg.stops).compute(
            stop_series,
            direction,
            float(fixture.get("atr", 250.0)),
            float(fixture.get("entry_price", bar.close)),
        )
        exit_orders = _exit_orders(
            ExitManager,
            Position,
            Bar,
            TimeFrame,
            cfg,
            fixture,
            symbol=symbol,
            direction=direction,
            ts=ts,
            timeframe=timeframe,
        )
        decision_context = DecisionContext(
            decision_id=fixture_id,
            strategy_id="trend",
            symbol=symbol,
            timeframe=timeframe,
            decision_time=ts,
            decision_key=f"trend|{symbol}|{timeframe.value}|{ts.isoformat()}",
        )
        if entry_order is not None:
            decision_context.record_order()

        key = f"{symbol}:{timeframe.value}:{fixture_id}"
        return [
            _event(ts, "signals", key, _signal_payload(setup, trigger)),
            _event(ts, "filters", key, _filter_payload(fixture, entry_order is not None)),
            _event(ts, "entries", key, _entry_payload(entry_order)),
            _event(ts, "exits", key, _orders_payload(exit_orders)),
            _event(
                ts,
                "stops",
                key,
                {"stop_price": stop_price, "setup_stop_level": setup.stop_level},
            ),
            _event(ts, "sizing", key, _sizing_payload(sizing)),
            _event(ts, "risk_caps", key, _risk_payload(cfg, sizing)),
            _event(
                ts,
                "order_intent",
                key,
                (
                    OrderIntent.from_order(entry_order, decision_context).to_dict()
                    if entry_order is not None
                    else {"action": "no_order", "strategy_id": "trend", "symbol": symbol}
                ),
            ),
        ]


@contextmanager
def _crypto_trader_import_path(repo_path: Path) -> Iterator[None]:
    src_path = (repo_path / "src").resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"crypto_trader src path is missing: {src_path}")
    text = str(src_path)
    already_present = text in sys.path
    if not already_present:
        sys.path.insert(0, text)
    prior_modules = _pop_crypto_trader_modules()
    try:
        yield
    finally:
        _pop_crypto_trader_modules()
        sys.modules.update(prior_modules)
        if not already_present:
            try:
                sys.path.remove(text)
            except ValueError:
                pass


def _pop_crypto_trader_modules() -> dict[str, Any]:
    modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "crypto_trader" or name.startswith("crypto_trader.")
    }
    for name in modules:
        sys.modules.pop(name, None)
    return modules


def _parse_ts(raw: object) -> datetime:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
    text = str(raw or "2026-03-15T10:00:00+00:00")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _bar(
    Bar: type,
    TimeFrame: type,
    data: Mapping[str, Any],
    *,
    symbol: str,
    timeframe: Any,
    ts: datetime,
) -> Any:
    raw = data.get("bar") if isinstance(data.get("bar"), Mapping) else data
    return Bar(
        timestamp=_parse_ts(raw.get("timestamp") if isinstance(raw, Mapping) else ts),
        symbol=str(raw.get("symbol") or symbol).upper(),
        open=float(raw.get("open", 50000.0)),
        high=float(raw.get("high", 50100.0)),
        low=float(raw.get("low", 49900.0)),
        close=float(raw.get("close", 50050.0)),
        volume=float(raw.get("volume", 100.0)),
        timeframe=TimeFrame.from_interval(str(raw.get("timeframe") or timeframe.value)),
    )


def _setup(
    TrendSetupResult: type,
    SetupGrade: type,
    Side: type,
    fixture: Mapping[str, Any],
    direction: Any,
) -> Any:
    raw = dict(fixture.get("setup") or {})
    confluences = tuple(raw.get("confluences") or ("h1_ema_zone", "rsi_pullback"))
    return TrendSetupResult(
        grade=SetupGrade(str(raw.get("grade") or "B").upper()),
        direction=Side(str(raw.get("direction") or direction.value).upper()),
        impulse_start=float(raw.get("impulse_start", 49000.0)),
        impulse_end=float(raw.get("impulse_end", 50500.0)),
        impulse_atr_move=float(raw.get("impulse_atr_move", 2.0)),
        pullback_depth=float(raw.get("pullback_depth", 0.3)),
        confluences=confluences,
        zone_price=float(raw.get("zone_price", fixture.get("entry_price", 50050.0))),
        room_r=float(raw.get("room_r", 2.5)),
        stop_level=float(raw.get("stop_level", 49500.0)),
        setup_score=float(raw.get("setup_score", 2.0)),
    )


def _sizing(SizingResult: type, fixture: Mapping[str, Any]) -> Any:
    raw = dict(fixture.get("sizing") or {})
    return SizingResult(
        qty=float(raw.get("qty", 0.1)),
        leverage=float(raw.get("leverage", 5.0)),
        liquidation_price=float(raw.get("liquidation_price", 40000.0)),
        risk_pct_actual=float(raw.get("risk_pct_actual", 0.005)),
        notional=float(raw.get("notional", 5000.0)),
        was_reduced=bool(raw.get("was_reduced", False)),
        reduction_reason=raw.get("reduction_reason"),
    )


def _trigger(TriggerResult: type, fixture: Mapping[str, Any]) -> Any | None:
    raw = fixture.get("trigger")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("trigger must be an object or null")
    return TriggerResult(
        str(raw.get("pattern") or "engulfing"),
        float(raw.get("trigger_price", fixture.get("entry_price", 50050.0))),
        int(raw.get("bar_offset", 0)),
        bool(raw.get("valid", True)),
    )


def _stop_bars(fixture: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = fixture.get("stop_bars")
    if isinstance(raw, list) and raw:
        return [item for item in raw if isinstance(item, Mapping)]
    return [fixture]


def _exit_orders(
    ExitManager: type,
    Position: type,
    Bar: type,
    TimeFrame: type,
    cfg: Any,
    fixture: Mapping[str, Any],
    *,
    symbol: str,
    direction: Any,
    ts: datetime,
    timeframe: Any,
) -> list[Any]:
    raw = dict(fixture.get("exit_state") or {})
    if fixture.get("position_open") is False:
        return []
    manager = ExitManager(cfg.exits)
    entry_price = float(raw.get("entry_price", fixture.get("entry_price", 50050.0)))
    stop_distance = float(raw.get("stop_distance", 500.0))
    sizing = fixture.get("sizing")
    default_qty = sizing.get("qty", 0.1) if isinstance(sizing, Mapping) else 0.1
    qty = float(raw.get("qty", default_qty))
    manager.init_position(symbol, entry_price, stop_distance, qty, direction)
    state = manager.get_state(symbol)
    if state is not None:
        for name in (
            "bars_since_entry",
            "mfe_r",
            "mae_r",
            "peak_r",
            "tp1_hit",
            "tp2_hit",
            "be_moved",
        ):
            if name in raw:
                setattr(state, name, raw[name])
    position = Position(
        symbol=symbol,
        direction=direction,
        qty=qty,
        avg_entry=entry_price,
        open_time=ts,
        leverage=float(raw.get("leverage", 5.0)),
    )
    exit_bar = fixture.get("exit_bar")
    bar = _bar(
        Bar,
        TimeFrame,
        exit_bar if isinstance(exit_bar, Mapping) else fixture,
        symbol=symbol,
        timeframe=timeframe,
        ts=ts,
    )
    indicator = SimpleNamespace(ema_fast=float(raw.get("ema_fast", bar.close)))
    return manager.manage(position, bar, [bar], indicator, broker=SimpleNamespace())


def _signal_payload(setup: Any, trigger: Any | None) -> dict[str, Any]:
    return {
        "grade": setup.grade.value,
        "direction": setup.direction.value,
        "confluences": list(setup.confluences),
        "room_r": setup.room_r,
        "setup_score": getattr(setup, "setup_score", 0.0),
        "confirmation": trigger.pattern if trigger is not None else "none",
    }


def _filter_payload(fixture: Mapping[str, Any], has_order: bool) -> dict[str, Any]:
    return {
        "warmup": bool(fixture.get("warmup", True)),
        "entry_window": bool(fixture.get("entry_window", True)),
        "risk_check": bool(fixture.get("risk_check", True)),
        "setup": bool(fixture.get("setup_passed", True)),
        "confirmation": fixture.get("trigger") is not None,
        "entry_order": has_order,
    }


def _entry_payload(order: Any | None) -> dict[str, Any]:
    if order is None:
        return {"action": "no_order"}
    return {
        "action": "enter",
        "side": order.side.value,
        "order_type": order.order_type.value,
        "qty": order.qty,
        "entry_method": order.metadata.get("entry_method", ""),
    }


def _orders_payload(orders: list[Any]) -> dict[str, Any]:
    return {
        "orders": [
            {
                "tag": order.tag,
                "side": order.side.value,
                "order_type": order.order_type.value,
                "qty": order.qty,
                "stop_price": order.stop_price,
                "limit_price": order.limit_price,
            }
            for order in orders
        ]
    }


def _sizing_payload(sizing: Any) -> dict[str, Any]:
    return {
        "qty": sizing.qty,
        "leverage": sizing.leverage,
        "liquidation_price": sizing.liquidation_price,
        "risk_pct_actual": sizing.risk_pct_actual,
        "notional": sizing.notional,
        "was_reduced": sizing.was_reduced,
        "reduction_reason": sizing.reduction_reason,
    }


def _risk_payload(cfg: Any, sizing: Any) -> dict[str, Any]:
    return {
        "risk_pct_actual": sizing.risk_pct_actual,
        "max_risk_pct": cfg.risk.max_risk_pct,
        "max_leverage_major": cfg.risk.max_leverage_major,
        "max_leverage_alt": cfg.risk.max_leverage_alt,
        "was_reduced": sizing.was_reduced,
        "reduction_reason": sizing.reduction_reason or "",
    }


def _event(
    ts: datetime,
    dimension: str,
    key: str,
    payload: Mapping[str, Any],
) -> DecisionTraceEvent:
    return DecisionTraceEvent(ts=ts, dimension=dimension, key=key, payload=_clean(payload))


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items() if item is not None}
    if isinstance(value, list | tuple):
        return [_clean(item) for item in value]
    return value
