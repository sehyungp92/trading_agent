from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from backtests.shared.parity.replay_driver import ReplayStep, run_replay
from tests.integration.parity.replay_candidates import (
    ReplayDecisionTimeline,
    broker_event_key as _broker_event_key,
)
from tests.integration.parity.source_inputs import (
    iaric_artifact,
    iaric_minute_bars,
    iaric_quote,
    iaric_state_snapshot,
    nq_bar_data,
    nq_daily_context,
    nq_live_context,
    parse_time,
    source_bars,
    tpc_bar_input,
    tpc_symbol_config,
)


def _replay_tpc(fixture: Mapping[str, Any], out: ReplayDecisionTimeline) -> None:
    from strategies.swing.tpc.core import logic
    from strategies.swing.tpc.core.serializers import restore_state, snapshot_state

    state = restore_state((fixture.get("initial_strategy_state", {}) or {}).get("TPC", {}))
    symbols = {str(row["symbol"]) for row in fixture.get("bars", []) if str(row.get("timeframe", "")).lower() == "15m"}
    for symbol in sorted(symbols):
        cfg = tpc_symbol_config(fixture, symbol)
        replay = run_replay(
            state,
            steps=[ReplayStep(bar_input=tpc_bar_input(fixture, symbol))],
            on_bar=lambda current, bar_input, cfg=cfg: logic.on_bar(current, bar_input, cfg),
            on_order_update=logic.on_order_update,
            on_fill=logic.on_fill,
        )
        state = replay.state
        out.record_actions("TPC", replay.actions)
    out.strategy_state["TPC"] = {
        "setups": sorted(snapshot_state(state).get("setups", {}).keys()),
        "positions": sorted(snapshot_state(state).get("positions", {}).keys()),
        "pending_count": len(snapshot_state(state).get("pending_orders", {}) or {}),
    }


def _replay_nq_regime(fixture: Mapping[str, Any], out: ReplayDecisionTimeline) -> None:
    from strategies.momentum.nq_regime.config import StrategyRuntimeSettings
    from strategies.momentum.nq_regime.core.data_policy import CompletedBarPolicy
    from strategies.momentum.nq_regime.core.logic import on_bar, on_fill
    from strategies.momentum.nq_regime.core.serializers import hydrate_state, snapshot_state
    from strategies.momentum.nq_regime.core.state import FillEvent

    settings = StrategyRuntimeSettings(
        initial_equity=float((fixture.get("account_state", {}) or {}).get("equity", 100_000.0)),
        max_contracts=int(((fixture.get("strategy_config", {}) or {}).get("config_overrides", {}) or {}).get("max_contracts", 5)),
        enable_liquidity_reversion=False,
        enable_second_wind=False,
    )
    state = hydrate_state((fixture.get("initial_strategy_state", {}) or {}).get("NQ_REGIME", {}))
    policy = CompletedBarPolicy()
    for row in source_bars(fixture, "NQ", "5m") or source_bars(fixture, "MNQ", "5m"):
        bar = nq_bar_data(row)
        step = ReplayStep(
            bar_input=policy.build_event(
                bar_5m=bar,
                recent_5m=[*state.bars_5m, bar],
                daily_context=nq_daily_context(fixture),
                live_context=nq_live_context(fixture),
            )
        )
        replay = run_replay(
            state,
            steps=[step],
            on_bar=lambda current, event: on_bar(current, event, scheduled_news=[], settings=settings),
            on_order_update=lambda current, update: (current, [], []),
            on_fill=lambda current, fill: on_fill(current, fill, settings=settings),
        )
        state = replay.state
        out.record_actions("NQ_REGIME", replay.actions)
    for event in fixture.get("broker_event_script", []):
        if str((event.get("order_match", {}) or {}).get("strategy_id")) != "NQ_REGIME":
            continue
        key = _broker_event_key(event)
        if key in out._applied:
            continue
        order = out._match_order(event.get("order_match", {}))
        if order is None:
            continue
        out.note_broker_event(order, event)
        out._applied.add(key)
        fill = FillEvent(
            oms_order_id=str(order["client_tag"]),
            fill_price=float(event.get("price", order.get("limit_price") or 0.0)),
            fill_qty=int(float(event.get("qty", order["qty"]))),
            fill_time=parse_time(event.get("timestamp")),
            symbol=str(order["symbol"]),
            commission=float(event.get("commission", 0.0)),
            order_role="entry",
        )
        state, actions, _events = on_fill(state, fill, settings=settings)
        out.record_actions("NQ_REGIME", actions)
    snap = snapshot_state(state)
    out.strategy_state["NQ_REGIME"] = {
        "position_side": str(snap.get("position_side", "")),
        "entry_price": snap.get("entry_price", 0.0),
        "stop_price": snap.get("stop_price", 0.0),
        "qty_open": snap.get("qty_open", 0),
        "daily_trades": snap.get("daily_trades", 0),
        "last_decision_code": snap.get("last_decision_code", ""),
    }


def _replay_iaric(fixture: Mapping[str, Any], out: ReplayDecisionTimeline) -> None:
    from strategies.stock.iaric.config import StrategySettings
    from strategies.stock.iaric.core import logic as iaric_logic
    from strategies.stock.iaric.core.state import IARICFill
    from strategies.stock.iaric.entry_request import build_ready_entry_request
    from strategies.stock.iaric.models import PortfolioState

    settings = StrategySettings(
        base_risk_fraction=float(((fixture.get("strategy_config", {}) or {}).get("config_overrides", {}) or {}).get("base_risk_fraction", StrategySettings().base_risk_fraction))
    )
    artifact = iaric_artifact(fixture)
    state = iaric_state_snapshot(fixture, "IARIC_v1")
    symbols = [symbol_state.symbol for symbol_state in state.symbols]
    for symbol in symbols:
        bars = iaric_minute_bars(fixture, symbol)
        if not bars:
            continue
        bar_5m = _aggregate_5m(bars[-5:])
        symbol_state = next(item for item in state.symbols if item.symbol == symbol)
        item = artifact.by_symbol[symbol]
        quote = iaric_quote(fixture, symbol)
        market = type("ReplayMarket", (), {})()
        market.bars_5m = [bar_5m]
        market.last_5m_bar = bar_5m
        market.last_30m_bar = bar_5m
        market.session_vwap = bar_5m.close
        market.session_low = min(symbol_state.session_low or bar_5m.low, bar_5m.low)
        market.session_high = max(symbol_state.session_high or bar_5m.high, bar_5m.high)
        market.last_price = bar_5m.close
        market.ask = quote.ask
        market.bid = quote.bid
        market.spread_pct = quote.spread_pct
        step = iaric_logic.evaluate_ready_entry(
            settings,
            symbol_state,
            item,
            bar_5m,
            market,
            max(int(symbol_state.bars_seen_today), 0),
            max(float(symbol_state.daily_atr), 0.01),
            bars=[bar_5m],
        )
        if step is None or step.acceptance is None:
            continue
        iaric_logic.apply_entry_acceptance(symbol_state, step.acceptance)
        portfolio = PortfolioState(
            account_equity=float((fixture.get("account_state", {}) or {}).get("equity", 100_000.0)),
            base_risk_fraction=settings.base_risk_fraction,
        )
        request_build = build_ready_entry_request(
            symbol=symbol,
            state=symbol_state,
            item=item,
            market=market,
            portfolio=portfolio,
            symbol_to_sector={symbol: item.sector},
            settings=settings,
            now=bar_5m.end_time,
            route=str(symbol_state.route_family or "OPENING_RECLAIM"),
        )
        if request_build.entry_request is None:
            continue
        symbol_state.risk_per_share = max(request_build.entry_price - float(symbol_state.stop_level), 0.01)
        entry_request = request_build.entry_request
        state, actions, _events = iaric_logic.on_bar(state, bar_ts=bar_5m.end_time, entry_request=entry_request)
        out.record_actions("IARIC_v1", actions)
    for event in fixture.get("broker_event_script", []):
        if str((event.get("order_match", {}) or {}).get("strategy_id")) != "IARIC_v1":
            continue
        key = _broker_event_key(event)
        if key in out._applied:
            continue
        order = out._match_order(event.get("order_match", {}))
        if order is None:
            continue
        out.note_broker_event(order, event)
        out._applied.add(key)
        state, actions, _events = iaric_logic.on_fill(
            state,
            IARICFill(
                oms_order_id=str(order["client_tag"]),
                fill_price=float(event.get("price", order.get("limit_price") or 0.0)),
                fill_qty=int(float(event.get("qty", order["qty"]))),
                fill_time=parse_time(event.get("timestamp")),
                commission=float(event.get("commission", 0.0)),
                symbol=str(order["symbol"]),
                order_role="ENTRY",
            ),
        )
        out.record_actions("IARIC_v1", actions)
    out.strategy_state["IARIC_v1"] = _compact_iaric_state(state)


def _aggregate_5m(bars: list[Any]) -> Any:
    from strategies.stock.iaric.models import Bar

    return Bar(
        symbol=bars[0].symbol,
        start_time=bars[0].start_time,
        end_time=bars[-1].end_time,
        open=bars[0].open,
        high=max(bar.high for bar in bars),
        low=min(bar.low for bar in bars),
        close=bars[-1].close,
        volume=sum(bar.volume for bar in bars),
    )


def _compact_iaric_state(state: Any) -> dict[str, Any]:
    symbols = {}
    for symbol_state in sorted(state.symbols, key=lambda item: item.symbol):
        position = getattr(symbol_state, "position", None)
        symbols[symbol_state.symbol] = {
            "stage": getattr(symbol_state, "stage", ""),
            "route_family": getattr(symbol_state, "route_family", ""),
            "in_position": bool(getattr(symbol_state, "in_position", False)),
            "risk_per_share": getattr(symbol_state, "risk_per_share", 0.0),
            "position": (
                {
                    "qty_open": getattr(position, "qty_open", 0),
                    "entry_price": getattr(position, "entry_price", 0.0),
                    "current_stop": getattr(position, "current_stop", 0.0),
                }
                if position is not None
                else None
            ),
        }
    return {"symbols": symbols, "last_decision_code": getattr(state, "last_decision_code", "")}


replay_tpc = _replay_tpc
replay_nq_regime = _replay_nq_regime
replay_iaric = _replay_iaric
