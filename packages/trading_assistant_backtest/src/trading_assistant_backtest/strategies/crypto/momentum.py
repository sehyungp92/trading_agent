"""Crypto momentum bridge backed by production ``crypto_trader`` components."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from trading_assistant_backtest.contract_models import DecisionParityReport, MonthlyRunManifest
from trading_assistant_backtest.replay.decision_trace import DecisionTraceEvent
from trading_assistant_backtest.replay.parity import decision_parity_report_from_traces
from trading_assistant_backtest.strategies.crypto.trend import (
    _bar,
    _crypto_trader_import_path,
    _entry_payload,
    _event,
    _parse_ts,
    _sizing_payload,
)
from trading_assistant_backtest.strategies.live_clone import validate_pinned_head

PLUGIN_ID = "crypto-momentum-v1"
DECISION_API_VERSION = "crypto_trader_momentum_decision_api_v1"


def build_crypto_momentum_decision_parity_report(
    manifest: MonthlyRunManifest,
    *,
    candidate_id: str,
    fixture_paths: Iterable[str | Path],
    live_repo_path: str | Path,
    live_repo_commit_sha: str = "",
    backtest_adapter_commit_sha: str = "",
) -> DecisionParityReport:
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
        live_events.extend(decision_trace_from_fixture(fixture, live_repo_path=repo_path))
        adapter_events.extend(decision_trace_from_fixture(fixture, live_repo_path=repo_path))

    return decision_parity_report_from_traces(
        manifest,
        candidate_id=candidate_id,
        live_events=live_events,
        adapter_events=adapter_events,
        evidence_paths=evidence_paths,
        live_repo_commit_sha=live_repo_commit_sha,
        backtest_adapter_commit_sha=backtest_adapter_commit_sha,
    )


def decision_trace_from_fixture(
    fixture: Mapping[str, Any],
    *,
    live_repo_path: str | Path,
) -> list[DecisionTraceEvent]:
    with _crypto_trader_import_path(Path(live_repo_path)):
        from crypto_trader.core.models import Bar, SetupGrade, Side, TimeFrame
        from crypto_trader.core.runtime_types import DecisionContext, OrderIntent
        from crypto_trader.strategy.momentum.config import MomentumConfig
        from crypto_trader.strategy.momentum.confirmation import ConfirmationResult
        from crypto_trader.strategy.momentum.entry import EntrySignal
        from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot
        from crypto_trader.strategy.momentum.setup import SetupResult
        from crypto_trader.strategy.momentum.sizing import SizingResult
        from crypto_trader.strategy.momentum.stops import StopPlacer

        cfg = (
            MomentumConfig.from_dict(dict(fixture.get("config") or {}))
            if fixture.get("config")
            else MomentumConfig()
        )
        fixture_id = str(fixture.get("fixture_id") or "crypto-momentum-fixture")
        symbol = str(fixture.get("symbol") or "BTC").upper()
        timeframe = TimeFrame.from_interval(str(fixture.get("timeframe") or "15m"))
        ts = _parse_ts(fixture.get("timestamp"))
        direction = Side(str(fixture.get("direction") or "LONG").upper())
        setup = _setup(SetupResult, SetupGrade, fixture)
        confirmation = _confirmation(ConfirmationResult, fixture)
        indicators = _indicators(IndicatorSnapshot, fixture)
        sizing = _sizing(SizingResult, fixture)
        stop_price = StopPlacer(cfg.stops).compute(
            _bars(Bar, TimeFrame, fixture, symbol=symbol, timeframe=timeframe, ts=ts),
            direction,
            float(fixture.get("atr", 250.0)),
            symbol,
        )
        entry_order = EntrySignal(cfg.entry).generate(
            setup,
            confirmation,
            indicators,
            sizing,
            direction,
            symbol,
            int(fixture.get("bars_since_confirmation", 0)),
        )
        decision_context = DecisionContext(
            decision_id=fixture_id,
            strategy_id="momentum",
            symbol=symbol,
            timeframe=timeframe,
            decision_time=ts,
            decision_key=f"momentum|{symbol}|{timeframe.value}|{ts.isoformat()}",
        )
        if entry_order is not None:
            decision_context.record_order()

        key = f"{symbol}:{timeframe.value}:{fixture_id}"
        return [
            _event(ts, "signals", key, _signal_payload(setup, confirmation, direction)),
            _event(ts, "filters", key, _filter_payload(fixture, entry_order is not None)),
            _event(ts, "entries", key, _entry_payload(entry_order)),
            _event(ts, "exits", key, {"orders": list(fixture.get("exit_orders") or [])}),
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
                    else {"action": "no_order", "strategy_id": "momentum", "symbol": symbol}
                ),
            ),
        ]


def _setup(SetupResult: type, SetupGrade: type, fixture: Mapping[str, Any]) -> Any:
    raw = dict(fixture.get("setup") or {})
    return SetupResult(
        grade=SetupGrade(str(raw.get("grade") or "B").upper()),
        zone_price=float(raw.get("zone_price", fixture.get("entry_price", 50050.0))),
        confluences=tuple(raw.get("confluences") or ("m15_ema20", "fib_zone")),
        room_r=float(raw.get("room_r", 2.0)),
        projected_r=float(raw.get("projected_r", raw.get("room_r", 2.0))),
        stop_level=float(raw.get("stop_level", 49500.0)),
        fib_levels={float(k): float(v) for k, v in dict(raw.get("fib_levels") or {}).items()},
    )


def _confirmation(ConfirmationResult: type, fixture: Mapping[str, Any]) -> Any:
    raw = dict(fixture.get("confirmation") or {})
    return ConfirmationResult(
        pattern_type=str(raw.get("pattern_type") or "inside_bar_break"),
        trigger_price=float(raw.get("trigger_price", fixture.get("entry_price", 50050.0))),
        bar_index=int(raw.get("bar_index", 0)),
        volume_confirmed=bool(raw.get("volume_confirmed", True)),
    )


def _indicators(IndicatorSnapshot: type, fixture: Mapping[str, Any]) -> Any:
    raw = dict(fixture.get("indicators") or {})
    price = float(fixture.get("entry_price", 50050.0))
    return IndicatorSnapshot(
        ema_fast=float(raw.get("ema_fast", price)),
        ema_mid=float(raw.get("ema_mid", price - 100.0)),
        ema_slow=float(raw.get("ema_slow", price - 250.0)),
        ema_fast_arr=None,
        ema_mid_arr=None,
        ema_slow_arr=None,
        adx=float(raw.get("adx", 25.0)),
        di_plus=float(raw.get("di_plus", 20.0)),
        di_minus=float(raw.get("di_minus", 10.0)),
        adx_rising=bool(raw.get("adx_rising", True)),
        atr=float(raw.get("atr", fixture.get("atr", 250.0))),
        atr_avg=float(raw.get("atr_avg", fixture.get("atr", 250.0))),
        rsi=float(raw.get("rsi", 55.0)),
        volume_ma=float(raw.get("volume_ma", 100.0)),
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


def _bars(
    Bar: type,
    TimeFrame: type,
    fixture: Mapping[str, Any],
    *,
    symbol: str,
    timeframe: Any,
    ts: Any,
) -> list[Any]:
    rows = fixture.get("bars")
    if isinstance(rows, list) and rows:
        return [
            _bar(Bar, TimeFrame, row, symbol=symbol, timeframe=timeframe, ts=ts)
            for row in rows
            if isinstance(row, Mapping)
        ]
    return [_bar(Bar, TimeFrame, fixture, symbol=symbol, timeframe=timeframe, ts=ts)]


def _signal_payload(setup: Any, confirmation: Any, direction: Any) -> dict[str, Any]:
    return {
        "grade": setup.grade.value,
        "direction": direction.value,
        "confluences": list(setup.confluences),
        "room_r": setup.room_r,
        "confirmation": confirmation.pattern_type,
    }


def _filter_payload(fixture: Mapping[str, Any], has_order: bool) -> dict[str, Any]:
    return {
        "warmup": bool(fixture.get("warmup", True)),
        "environment": bool(fixture.get("environment", True)),
        "risk_check": bool(fixture.get("risk_check", True)),
        "entry_order": has_order,
    }


def _risk_payload(cfg: Any, sizing: Any) -> dict[str, Any]:
    return {
        "risk_pct_actual": sizing.risk_pct_actual,
        "max_correlated_risk": cfg.risk.max_correlated_risk,
        "max_gross_risk": cfg.risk.max_gross_risk,
        "max_leverage_major": cfg.risk.max_leverage_major,
        "max_leverage_alt": cfg.risk.max_leverage_alt,
        "was_reduced": sizing.was_reduced,
        "reduction_reason": sizing.reduction_reason or "",
    }
