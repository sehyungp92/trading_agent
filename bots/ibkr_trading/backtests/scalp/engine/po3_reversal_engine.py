from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace

import numpy as np

from backtests.scalp.analysis.metrics import extract_metrics
from backtests.scalp.config_po3_reversal import Po3ReversalBacktestConfig
from backtests.scalp.data.multi_instrument import ScalpMultiInstrumentData, load_po3_data
from backtests.scalp.data.preprocessing import NumpyBars
from backtests.scalp.engine.param_overrides import temporary_param_overrides
from backtests.scalp.engine.sim_broker import FillStatus, OrderSide, OrderType, SimBroker, SimOrder
from backtests.shared.parity.legacy_result_outputs import (
    decision_stream_from_records,
    decision_stream_from_trades,
    merge_decision_streams,
    trade_outcomes_from_records,
)
from strategies.core.actions import FlattenPosition, ReplaceProtectiveStop, SubmitEntry, SubmitProfitTarget, SubmitProtectiveStop
from strategies.core.events import DecisionEvent
from strategies.scalp._shared.nq_contract import spec_for
from strategies.scalp._shared.session import must_flatten
from strategies.scalp._shared.time_utils import session_date
from strategies.scalp.po3_reversal import config as po3_config
from strategies.scalp.po3_reversal import gates as po3_gates
from strategies.scalp.po3_reversal import htf_context as po3_htf_context
from strategies.scalp.po3_reversal import signals as po3_signals
from strategies.scalp.po3_reversal import smt as po3_smt
from strategies.scalp.po3_reversal import stops as po3_stops
from strategies.scalp.po3_reversal.config import SetupTier, TradeDirection
from strategies.scalp.po3_reversal.core import logic as core_logic
from strategies.scalp.po3_reversal.core.state import Po3BarInput, Po3Fill, Po3FlattenRequest, Po3ReversalCoreState
from strategies.scalp.po3_reversal.fvg import FvgStateMachine
from strategies.scalp.po3_reversal.gates import direction_gate, session_gate
from strategies.scalp.po3_reversal.htf_context import build_context, determine_trade_tier
from strategies.scalp.po3_reversal.indicators import atr
from strategies.scalp.po3_reversal.liquidity import detect_liquidity_pools, detect_sweep
from strategies.scalp.po3_reversal.models import PriceBar, TradeRecord
from strategies.scalp.po3_reversal.signals import score_components, threshold_for_tier
from strategies.scalp.po3_reversal.smt import detect_smt_divergence
from strategies.scalp.po3_reversal.stops import (
    compute_entry_price,
    compute_position_size,
    compute_stop_price,
    reward_to_risk,
    target_passes_rr,
)


_PO3_OVERRIDE_MODULES = (
    po3_config,
    po3_gates,
    po3_htf_context,
    po3_signals,
    po3_smt,
    po3_stops,
    sys.modules[__name__],
)


@dataclass
class Po3SymbolResult:
    symbol: str
    trades: list[TradeRecord] = field(default_factory=list)
    signal_events: list[DecisionEvent] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    decision_stream: list[dict] = field(default_factory=list)
    trade_outcomes: list[dict] = field(default_factory=list)


@dataclass
class Po3BacktestResult:
    symbol_results: dict[str, Po3SymbolResult] = field(default_factory=dict)
    combined_equity: np.ndarray = field(default_factory=lambda: np.array([]))
    combined_timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    metrics: dict[str, float] = field(default_factory=dict)
    decision_stream: list[dict] = field(default_factory=list)
    trade_outcomes: list[dict] = field(default_factory=list)


def load_po3_reversal_data(config: Po3ReversalBacktestConfig) -> ScalpMultiInstrumentData:
    return load_po3_data(
        config.data_dir,
        analysis_symbol=config.analysis_symbol,
        confirmation_symbol=config.confirmation_symbol,
    )


def run_po3_reversal_backtest(
    data: ScalpMultiInstrumentData,
    config: Po3ReversalBacktestConfig,
) -> Po3BacktestResult:
    if config.param_overrides:
        with temporary_param_overrides(config.param_overrides, _PO3_OVERRIDE_MODULES):
            return _run_po3_reversal_backtest(data, config)
    return _run_po3_reversal_backtest(data, config)


def _run_po3_reversal_backtest(
    data: ScalpMultiInstrumentData,
    config: Po3ReversalBacktestConfig,
) -> Po3BacktestResult:
    analysis_symbol = config.analysis_symbol
    trade_symbol = config.trade_symbol
    point_value = spec_for(trade_symbol).point_value
    m1 = data.analysis.get("1m", NumpyBars())
    if len(m1) == 0:
        return Po3BacktestResult(combined_equity=np.array([config.initial_equity]), metrics={})

    state = Po3ReversalCoreState()
    broker = SimBroker()
    fvg = FvgStateMachine(symbol=analysis_symbol, max_age_bars=int(config.param_overrides.get("retest_wait_bars", 3)))
    ledger: dict[str, SimpleNamespace] = {}
    trades: list[TradeRecord] = []
    events: list[DecisionEvent] = []
    equity = config.initial_equity
    equity_curve: list[float] = []
    timestamps: list = []
    active_day = None

    for idx in range(len(m1)):
        ts = _dt(m1.times[idx])
        if config.start_date and ts < config.start_date:
            continue
        if config.end_date and ts > config.end_date:
            continue
        day_key = session_date(ts)
        if active_day is not None and day_key != active_day:
            state = Po3ReversalCoreState()
            broker = SimBroker()
            fvg = FvgStateMachine(symbol=analysis_symbol, max_age_bars=int(config.param_overrides.get("retest_wait_bars", 3)))
            ledger.clear()
        active_day = day_key

        fills = broker.process_bar(trade_symbol, ts, float(m1.opens[idx]), float(m1.highs[idx]), float(m1.lows[idx]), float(m1.closes[idx]))
        for result in fills:
            if result.status is not FillStatus.FILLED:
                continue
            fill = Po3Fill(
                oms_order_id=result.order.order_id,
                fill_price=result.fill_price,
                fill_qty=result.order.qty,
                symbol=trade_symbol,
                fill_time=result.fill_time,
                commission=result.commission,
                order_role=result.order.tag if result.order.tag in {"entry", "stop", "target"} else "unknown",
            )
            state, child_actions, fill_events = core_logic.on_fill(state, fill)
            events.extend(fill_events)
            for action in child_actions:
                _submit_action(broker, action, ts, state)
            trade = _update_ledger(ledger, result, point_value=point_value)
            if trade is not None:
                trades.append(trade)
                equity += trade.pnl_dollars

        current = _bar_at(m1, idx)
        fvg.update(current)
        payload = _build_payload(data, config, idx, current, fvg, equity)
        state, actions, bar_events = core_logic.on_bar(state, payload)
        events.extend(bar_events)
        if must_flatten(ts) and state.position is not None:
            state, flatten_actions, flatten_events = _request_flatten(state, payload, trade_symbol)
            actions.extend(flatten_actions)
            events.extend(flatten_events)
        for action in actions:
            _submit_action(broker, action, ts, state)

        mark = equity
        if state.position is not None:
            pos = state.position
            direction = 1 if pos.direction is TradeDirection.LONG else -1
            mark += (float(m1.closes[idx]) - pos.avg_entry) * direction * pos.qty * point_value
        equity_curve.append(mark)
        timestamps.append(ts)

    result = Po3SymbolResult(
        symbol=trade_symbol,
        trades=trades,
        signal_events=events,
        equity_curve=np.asarray(equity_curve, dtype=float),
        timestamps=np.asarray(timestamps),
    )
    result.decision_stream = merge_decision_streams(
        decision_stream_from_records(events, timeframe="1m"),
        decision_stream_from_trades(trades, timeframe="1m"),
    )
    result.trade_outcomes = trade_outcomes_from_records(trades)
    metrics = extract_metrics(trades, result.equity_curve, result.timestamps, config.initial_equity)
    metrics_dict = metrics.__dict__.copy()
    return Po3BacktestResult(
        symbol_results={trade_symbol: result},
        combined_equity=result.equity_curve,
        combined_timestamps=result.timestamps,
        metrics=metrics_dict,
        decision_stream=result.decision_stream,
        trade_outcomes=result.trade_outcomes,
    )


def _build_payload(
    data: ScalpMultiInstrumentData,
    config: Po3ReversalBacktestConfig,
    idx: int,
    current: PriceBar,
    fvg: FvgStateMachine,
    equity: float,
) -> Po3BarInput:
    analysis_symbol = config.analysis_symbol
    trade_symbol = config.trade_symbol
    analysis = data.analysis
    confirmation = data.confirmation
    h1 = _slice_completed(analysis.get("1h", NumpyBars()), data.analysis_idx_maps.get("1h"), idx, current.ts, "1h")
    h4 = _slice_completed(analysis.get("4h", NumpyBars()), data.analysis_idx_maps.get("4h"), idx, current.ts, "4h")
    daily = _slice_completed(analysis.get("daily", NumpyBars()), data.analysis_idx_maps.get("daily"), idx, current.ts, "daily")
    context = build_context(daily, h4, h1)

    recent = [_bar_at(analysis["1m"], i) for i in range(max(0, idx - 60), idx + 1)]
    pools = detect_liquidity_pools([bar.high for bar in recent], [bar.low for bar in recent], symbol=analysis_symbol)
    sweep = detect_sweep(current.low, pools, min_ticks=int(po3_config.MIN_NQ_SWEEP_TICKS), side="sell_side")
    if not sweep.swept:
        sweep = detect_sweep(current.high, pools, min_ticks=int(po3_config.MIN_NQ_SWEEP_TICKS), side="buy_side")
    direction = sweep.direction
    nq_m1 = recent
    es_m1 = _slice_recent_completed_by_time(confirmation.get("1m", NumpyBars()), current.ts, 60)
    smt = detect_smt_divergence(nq_m1, es_m1, direction) if es_m1 else None
    ifvg = fvg.get_ifvg_entry(direction) if direction is not TradeDirection.FLAT else None
    tier_hint = determine_trade_tier(
        context.daily_bias,
        context.h4_bias,
        {
            "prime_window": session_gate(current.ts).passed,
            "sweep": sweep.swept,
            "smt": bool(smt and smt.present),
            "ifvg": ifvg is not None,
            "target": context.h1_target > 0,
        },
    )
    direction_ok = direction_gate(context.daily_bias, context.h4_bias, direction).passed
    score = score_components(
        h4_location=direction_ok,
        daily_h4_alignment=context.daily_bias in {TradeDirection.FLAT, context.h4_bias},
        liquidity_sweep=sweep.swept and not config.flags.disable_liquidity_sweep,
        smt_present=bool(smt and smt.present) and not config.flags.disable_smt,
        smt_strength=smt.strength if smt else 0.0,
        ifvg_close_through=ifvg is not None and not config.flags.disable_ifvg,
        ifvg_retest_rejection=ifvg is not None and current.body_percent >= 0.5,
        displacement_body_pct=current.body_percent,
        spread_clean=True,
        atr_normal=True,
        h1_target_clean=context.h1_target > 0,
        tier_hint=tier_hint,
    )
    tier = score.tier
    if config.flags.disable_b_tier and tier is SetupTier.B:
        tier = SetupTier.NONE
    atr_1m = atr(recent[-20:])
    entry = compute_entry_price(current, direction) if direction is not TradeDirection.FLAT else 0.0
    smt_extreme = smt.nq_extreme if smt and smt.nq_extreme else (current.low if direction is TradeDirection.LONG else current.high)
    stop = compute_stop_price(direction=direction, smt_extreme=smt_extreme, atr_1m=atr_1m) if entry else 0.0
    target = context.h1_target
    rr = reward_to_risk(entry, stop, target, direction) if target else 0.0
    qty = config.fixed_qty or compute_position_size(equity=equity, tier=tier, entry=entry, stop=stop, symbol=trade_symbol)
    threshold = threshold_for_tier(tier)
    body_ok = current.body_percent >= po3_config.MIN_BODY_PERCENT
    risk_ok = session_gate(current.ts).passed and target_passes_rr(rr, tier) and body_ok
    return Po3BarInput(
        symbol=trade_symbol,
        bar_ts=current.ts,
        bar_ohlcv=(current.open, current.high, current.low, current.close, current.volume),
        context=context,
        sweep=sweep,
        smt=smt,
        ifvg=ifvg,
        signal_score=score.total,
        signal_threshold=threshold,
        tier=tier,
        direction=direction,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        qty=qty,
        rr=rr,
        risk_approved=risk_ok,
        decision_code="SIGNAL_EVALUATED",
        decision_details={
            "score": score.total,
            "tier": tier.value,
            "rr": rr,
            "direction": int(direction),
            "body_ok": body_ok,
            "analysis_symbol": analysis_symbol,
            "trade_symbol": trade_symbol,
            "confirmation_symbol": config.confirmation_symbol,
        },
    )


def _request_flatten(
    state: Po3ReversalCoreState,
    payload: Po3BarInput,
    symbol: str,
) -> tuple[Po3ReversalCoreState, list, list[DecisionEvent]]:
    if state.position is None:
        return state, [], []
    return core_logic.on_bar(
        state,
        payload,
        flatten_request=Po3FlattenRequest(setup_id=state.position.setup_id, symbol=symbol, reason="EOD_FLATTEN"),
    )


def _submit_action(broker: SimBroker, action, ts: datetime, state: Po3ReversalCoreState | None = None) -> None:
    if isinstance(action, SubmitEntry):
        side = OrderSide.BUY if action.side == "BUY" else OrderSide.SELL
        order_type = OrderType.STOP if action.order_type == "STOP" else OrderType.LIMIT if action.order_type == "LIMIT" else OrderType.MARKET
        broker.submit_order(
            SimOrder(
                order_id=action.client_order_id,
                symbol=action.symbol,
                side=side,
                order_type=order_type,
                qty=action.qty,
                stop_price=action.stop_price or 0.0,
                limit_price=action.limit_price or action.price or 0.0,
                submit_time=ts,
                earliest_fill_time=ts,
                tag="entry",
                metadata={**action.metadata, **action.risk_context},
            )
        )
    elif isinstance(action, ReplaceProtectiveStop):
        broker.cancel_order_id(action.target_order_id)
        broker.submit_order(
            SimOrder(
                order_id=action.target_order_id,
                symbol=action.symbol,
                side=OrderSide.BUY if action.side == "BUY" else OrderSide.SELL,
                order_type=OrderType.STOP,
                qty=action.qty,
                stop_price=action.stop_price,
                submit_time=ts,
                earliest_fill_time=ts,
                tag="stop",
                metadata={"reason": action.reason, **action.metadata},
            )
        )
    elif isinstance(action, FlattenPosition):
        setup_id = str(action.metadata.get("setup_id", ""))
        order_id = f"{action.symbol}-po3-flatten-{setup_id or ts.strftime('%Y%m%d%H%M%S')}"
        broker.submit_order(
            SimOrder(
                order_id=order_id,
                symbol=action.symbol,
                side=OrderSide.BUY if action.side == "BUY" else OrderSide.SELL,
                order_type=OrderType.MARKET,
                qty=action.qty,
                submit_time=ts,
                earliest_fill_time=ts,
                tag="flatten",
                metadata={"setup_id": setup_id, "reason": action.reason},
            )
        )
        if state is not None and setup_id:
            state.order_to_setup[order_id] = setup_id
            state.order_kind[order_id] = "flatten"
    elif isinstance(action, SubmitProtectiveStop):
        broker.submit_order(
            SimOrder(
                order_id=action.client_order_id,
                symbol=action.symbol,
                side=OrderSide.BUY if action.side == "BUY" else OrderSide.SELL,
                order_type=OrderType.STOP,
                qty=action.qty,
                stop_price=action.stop_price,
                submit_time=ts,
                earliest_fill_time=ts,
                tag="stop",
                oca_group=action.oca_group,
                metadata=action.metadata,
            )
        )
    elif isinstance(action, SubmitProfitTarget):
        broker.submit_order(
            SimOrder(
                order_id=action.client_order_id,
                symbol=action.symbol,
                side=OrderSide.BUY if action.side == "BUY" else OrderSide.SELL,
                order_type=OrderType.LIMIT,
                qty=action.qty,
                limit_price=action.limit_price,
                submit_time=ts,
                earliest_fill_time=ts,
                tag="target",
                oca_group=action.oca_group,
                metadata=action.metadata,
            )
        )


def _update_ledger(ledger: dict[str, SimpleNamespace], result, *, point_value: float) -> TradeRecord | None:
    order = result.order
    setup_id = order.metadata.get("setup_id", "")
    if order.tag == "entry":
        ledger[setup_id] = SimpleNamespace(
            setup_id=setup_id,
            entry_time=result.fill_time,
            entry_price=result.fill_price,
            qty=order.qty,
            side="BUY" if order.side is OrderSide.BUY else "SELL",
            commission=result.commission,
            tier=order.metadata.get("tier", ""),
            entry_type=order.metadata.get("entry_type", ""),
        )
        return None
    open_trade = ledger.pop(setup_id, None)
    if open_trade is None:
        return None
    direction = 1 if open_trade.side == "BUY" else -1
    gross = (result.fill_price - open_trade.entry_price) * direction * open_trade.qty * point_value
    commission = open_trade.commission + result.commission
    net = gross - commission
    risk = abs(open_trade.entry_price - float(order.metadata.get("stop_for_risk", open_trade.entry_price))) * open_trade.qty * point_value
    return TradeRecord(
        symbol=order.symbol,
        side=open_trade.side,
        qty=open_trade.qty,
        entry_time=open_trade.entry_time,
        exit_time=result.fill_time,
        entry_price=open_trade.entry_price,
        exit_price=result.fill_price,
        gross_pnl=gross,
        commission=commission,
        pnl_dollars=net,
        r_multiple=net / risk if risk > 0 else 0.0,
        exit_reason=order.tag,
        setup_id=setup_id,
        tier=open_trade.tier,
        entry_type=open_trade.entry_type,
    )


def _slice_completed(
    bars: NumpyBars,
    idx_map: np.ndarray | None,
    primary_idx: int,
    primary_time: datetime,
    timeframe: str,
) -> list[PriceBar]:
    if idx_map is None or len(bars) == 0:
        return []
    end = int(idx_map[min(primary_idx, len(idx_map) - 1)])
    end = min(end, len(bars) - 1)
    while end >= 0:
        bar_time = _dt(bars.times[end])
        if timeframe == "daily":
            if bar_time.date() < primary_time.date():
                break
        elif bar_time <= primary_time:
            break
        end -= 1
    if end < 0:
        return []
    return [_bar_at(bars, i) for i in range(0, end + 1)]


def _slice_recent_completed_by_time(bars: NumpyBars, as_of: datetime, max_count: int) -> list[PriceBar]:
    if len(bars) == 0:
        return []
    as_of64 = np.datetime64(as_of.replace(tzinfo=None), "ns")
    end = int(np.searchsorted(bars.times, as_of64, side="right")) - 1
    if end < 0:
        return []
    start = max(0, end - max_count + 1)
    return [_bar_at(bars, i) for i in range(start, end + 1)]


def _bar_at(bars: NumpyBars, idx: int) -> PriceBar:
    idx = max(0, min(idx, len(bars) - 1))
    return PriceBar(
        ts=_dt(bars.times[idx]),
        open=float(bars.opens[idx]),
        high=float(bars.highs[idx]),
        low=float(bars.lows[idx]),
        close=float(bars.closes[idx]),
        volume=float(bars.volumes[idx]),
    )


def _dt(value) -> datetime:
    import pandas as pd

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()
