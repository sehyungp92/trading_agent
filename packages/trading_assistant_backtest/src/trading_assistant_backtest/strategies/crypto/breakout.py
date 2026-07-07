"""Crypto breakout bridge backed by production ``crypto_trader`` components."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from trading_assistant_backtest.contract_models import DecisionParityReport, MonthlyRunManifest
from trading_assistant_backtest.replay.decision_trace import DecisionTraceEvent
from trading_assistant_backtest.replay.parity import decision_parity_report_from_traces
from trading_assistant_backtest.strategies.crypto.momentum import _sizing
from trading_assistant_backtest.strategies.crypto.trend import (
    _bar,
    _crypto_trader_import_path,
    _entry_payload,
    _event,
    _orders_payload,
    _parse_ts,
    _risk_payload,
    _sizing_payload,
)
from trading_assistant_backtest.strategies.live_clone import validate_pinned_head

PLUGIN_ID = "crypto-breakout-v1"
DECISION_API_VERSION = "crypto_trader_breakout_decision_api_v1"


def build_crypto_breakout_decision_parity_report(
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
        from crypto_trader.strategy.breakout.balance import BalanceZone
        from crypto_trader.strategy.breakout.config import BreakoutConfig
        from crypto_trader.strategy.breakout.confirmation import BreakoutConfirmation
        from crypto_trader.strategy.breakout.entry import EntryGenerator
        from crypto_trader.strategy.breakout.setup import BreakoutSetupResult
        from crypto_trader.strategy.breakout.sizing import SizingResult
        from crypto_trader.strategy.breakout.stops import StopPlacer

        cfg = (
            BreakoutConfig.from_dict(dict(fixture.get("config") or {}))
            if fixture.get("config")
            else BreakoutConfig()
        )
        fixture_id = str(fixture.get("fixture_id") or "crypto-breakout-fixture")
        symbol = str(fixture.get("symbol") or "BTC").upper()
        timeframe = TimeFrame.from_interval(str(fixture.get("timeframe") or "30m"))
        ts = _parse_ts(fixture.get("timestamp"))
        bar = _bar(Bar, TimeFrame, fixture, symbol=symbol, timeframe=timeframe, ts=ts)
        direction = Side(str(fixture.get("direction") or "LONG").upper())
        setup = _setup(BreakoutSetupResult, BalanceZone, SetupGrade, Side, fixture, direction)
        confirmation = _confirmation(BreakoutConfirmation, fixture)
        sizing = _sizing(SizingResult, fixture)
        stop_price = StopPlacer(cfg.stops).compute(
            setup,
            None,
            float(fixture.get("entry_price", bar.close)),
            float(fixture.get("atr", 250.0)),
            direction,
        )
        entry_order = EntryGenerator(cfg.entry).generate(
            bar,
            direction,
            sizing.qty,
            sizing,
            setup,
            confirmation,
            symbol,
            f"{fixture_id}:entry",
        )
        decision_context = DecisionContext(
            decision_id=fixture_id,
            strategy_id="breakout",
            symbol=symbol,
            timeframe=timeframe,
            decision_time=ts,
            decision_key=f"breakout|{symbol}|{timeframe.value}|{ts.isoformat()}",
        )
        if entry_order is not None:
            decision_context.record_order()

        key = f"{symbol}:{timeframe.value}:{fixture_id}"
        return [
            _event(ts, "signals", key, _signal_payload(setup, confirmation)),
            _event(ts, "filters", key, _filter_payload(fixture, entry_order is not None)),
            _event(ts, "entries", key, _entry_payload(entry_order)),
            _event(ts, "exits", key, _orders_payload([])),
            _event(ts, "stops", key, {"stop_price": stop_price}),
            _event(ts, "sizing", key, _sizing_payload(sizing)),
            _event(ts, "risk_caps", key, _risk_payload(cfg, sizing)),
            _event(
                ts,
                "order_intent",
                key,
                (
                    OrderIntent.from_order(entry_order, decision_context).to_dict()
                    if entry_order is not None
                    else {"action": "no_order", "strategy_id": "breakout", "symbol": symbol}
                ),
            ),
        ]


def _setup(
    BreakoutSetupResult: type,
    BalanceZone: type,
    SetupGrade: type,
    Side: type,
    fixture: Mapping[str, Any],
    direction: Any,
) -> Any:
    raw = dict(fixture.get("setup") or {})
    zone_raw = dict(raw.get("balance_zone") or {})
    zone = BalanceZone(
        center=float(zone_raw.get("center", 50000.0)),
        upper=float(zone_raw.get("upper", 50100.0)),
        lower=float(zone_raw.get("lower", 49900.0)),
        bars_in_zone=int(zone_raw.get("bars_in_zone", 20)),
        touches=int(zone_raw.get("touches", 3)),
        formation_bar_idx=int(zone_raw.get("formation_bar_idx", 10)),
        volume_contracting=bool(zone_raw.get("volume_contracting", True)),
        width_atr=float(zone_raw.get("width_atr", 1.2)),
    )
    return BreakoutSetupResult(
        grade=SetupGrade(str(raw.get("grade") or "B").upper()),
        is_a_plus=bool(raw.get("is_a_plus", False)),
        direction=Side(str(raw.get("direction") or direction.value).upper()),
        balance_zone=zone,
        breakout_price=float(raw.get("breakout_price", fixture.get("entry_price", 50150.0))),
        lvn_runway_atr=float(raw.get("lvn_runway_atr", 1.5)),
        confluences=tuple(raw.get("confluences") or ("h4_alignment", "volume_surge")),
        room_r=float(raw.get("room_r", 1.8)),
        volume_mult=float(raw.get("volume_mult", 1.2)),
        body_ratio=float(raw.get("body_ratio", 0.6)),
        signal_variant=str(raw.get("signal_variant") or "core"),
        risk_scale=float(raw.get("risk_scale", 1.0)),
    )


def _confirmation(BreakoutConfirmation: type, fixture: Mapping[str, Any]) -> Any:
    raw = dict(fixture.get("confirmation") or {})
    return BreakoutConfirmation(
        model=str(raw.get("model") or "model1_close"),
        trigger_price=float(raw.get("trigger_price", fixture.get("entry_price", 50150.0))),
        bar_index=int(raw.get("bar_index", 0)),
        volume_confirmed=bool(raw.get("volume_confirmed", True)),
    )


def _signal_payload(setup: Any, confirmation: Any) -> dict[str, Any]:
    return {
        "grade": setup.grade.value,
        "direction": setup.direction.value,
        "is_a_plus": setup.is_a_plus,
        "signal_variant": setup.signal_variant,
        "confluences": list(setup.confluences),
        "room_r": setup.room_r,
        "confirmation": confirmation.model,
    }


def _filter_payload(fixture: Mapping[str, Any], has_order: bool) -> dict[str, Any]:
    return {
        "warmup": bool(fixture.get("warmup", True)),
        "context": bool(fixture.get("context", True)),
        "risk_check": bool(fixture.get("risk_check", True)),
        "entry_order": has_order,
    }
