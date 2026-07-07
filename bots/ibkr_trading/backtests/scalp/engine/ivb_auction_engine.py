from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace

import numpy as np

from backtests.scalp.analysis.metrics import extract_metrics
from backtests.scalp.config_ivb_auction import IvbAuctionBacktestConfig
from backtests.scalp.data.preprocessing import NumpyBars, load_bar_data, load_tick_data
from backtests.scalp.data.tick_replay import build_footprint_bars
from backtests.scalp.data.volume_profile_precompute import precompute_ivb_levels
from backtests.scalp.engine.param_overrides import temporary_param_overrides
from backtests.scalp.engine.sim_broker import FillStatus, OrderSide, OrderType, SimBroker, SimOrder
from backtests.shared.parity.legacy_result_outputs import (
    decision_stream_from_records,
    decision_stream_from_trades,
    merge_decision_streams,
    trade_outcomes_from_records,
)
from strategies.core.actions import FlattenPosition, ReplaceProtectiveStop, SubmitEntry, SubmitPartialExit, SubmitProfitTarget, SubmitProtectiveStop
from strategies.core.events import DecisionEvent
from strategies.scalp._shared.nq_contract import spec_for
from strategies.scalp._shared.session import get_session_block, must_flatten
from strategies.scalp._shared.time_utils import session_date, to_et
from strategies.scalp.ivb_auction import config as ivb_config
from strategies.scalp.ivb_auction.config import A1_MIN_SCORE, A2_MIN_SCORE, EntryTrigger, IvbModule, TradeDirection
from strategies.scalp.ivb_auction.core import logic as core_logic
from strategies.scalp.ivb_auction.core.state import IvbAuctionCoreState, IvbBarInput, IvbFill, IvbFlattenRequest
from strategies.scalp.ivb_auction import gates as ivb_gates
from strategies.scalp.ivb_auction import signals as ivb_signals
from strategies.scalp.ivb_auction import stops as ivb_stops
from strategies.scalp.ivb_auction import target_model as ivb_target_model
from strategies.scalp.ivb_auction.gates import breakout_acceptance, ivb_range_gate
from strategies.scalp.ivb_auction.models import FootprintBarData, TradeRecord
from strategies.scalp.ivb_auction.signals import score_signal
from strategies.scalp.ivb_auction.stops import (
    compute_position_size,
    continuation_stop,
    reclaim_stop,
    reward_to_risk,
    stop_within_cap,
)
from strategies.scalp.ivb_auction.target_model import fallback_targets, reclaim_targets


_IVB_OVERRIDE_MODULES = (
    ivb_config,
    ivb_gates,
    ivb_signals,
    ivb_stops,
    ivb_target_model,
    sys.modules[__name__],
)


@dataclass
class IvbAuctionData:
    analysis_symbol: str = "NQ"
    bars: dict[str, NumpyBars] = field(default_factory=dict)
    footprint_by_minute: dict[datetime, FootprintBarData] = field(default_factory=dict)


@dataclass
class IvbSymbolResult:
    symbol: str
    trades: list[TradeRecord] = field(default_factory=list)
    signal_events: list[DecisionEvent] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    decision_stream: list[dict] = field(default_factory=list)
    trade_outcomes: list[dict] = field(default_factory=list)


@dataclass
class IvbBacktestResult:
    symbol_results: dict[str, IvbSymbolResult] = field(default_factory=dict)
    combined_equity: np.ndarray = field(default_factory=lambda: np.array([]))
    combined_timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    metrics: dict[str, float] = field(default_factory=dict)
    decision_stream: list[dict] = field(default_factory=list)
    trade_outcomes: list[dict] = field(default_factory=list)


def load_ivb_auction_data(config: IvbAuctionBacktestConfig) -> IvbAuctionData:
    analysis_symbol = config.analysis_symbol
    bars = load_bar_data(config.data_dir, analysis_symbol)
    ticks = load_tick_data(config.data_dir, analysis_symbol)
    footprints = build_footprint_bars(ticks).bars if config.replay_mode != "bar_only" else []
    footprint_by_minute = {_minute_key(item.end_ts): item for item in footprints}
    return IvbAuctionData(analysis_symbol=analysis_symbol, bars=bars, footprint_by_minute=footprint_by_minute)


def run_ivb_auction_backtest(
    data: IvbAuctionData,
    config: IvbAuctionBacktestConfig,
) -> IvbBacktestResult:
    if config.param_overrides:
        with temporary_param_overrides(config.param_overrides, _IVB_OVERRIDE_MODULES):
            return _run_ivb_auction_backtest(data, config)
    return _run_ivb_auction_backtest(data, config)


def _run_ivb_auction_backtest(
    data: IvbAuctionData,
    config: IvbAuctionBacktestConfig,
) -> IvbBacktestResult:
    trade_symbol = config.trade_symbol
    point_value = spec_for(trade_symbol).point_value
    bars = data.bars.get("1m", NumpyBars())
    if len(bars) == 0:
        return IvbBacktestResult(combined_equity=np.array([config.initial_equity]), metrics={})

    levels_cache = precompute_ivb_levels(bars)
    state = IvbAuctionCoreState()
    broker = SimBroker()
    ledger: dict[str, SimpleNamespace] = {}
    trades: list[TradeRecord] = []
    events: list[DecisionEvent] = []
    equity = config.initial_equity
    equity_curve: list[float] = []
    timestamps: list = []
    accepted_breaks: dict[str, TradeDirection] = {}
    break_hold_seconds: dict[str, tuple[TradeDirection, float]] = {}
    active_day = None

    for idx in range(len(bars)):
        ts = _dt(bars.times[idx])
        if config.start_date and ts < config.start_date:
            continue
        if config.end_date and ts > config.end_date:
            continue
        day_key = session_date(ts)
        if active_day is not None and day_key != active_day:
            state = IvbAuctionCoreState()
            broker = SimBroker()
            ledger.clear()
        active_day = day_key

        fills = broker.process_bar(trade_symbol, ts, float(bars.opens[idx]), float(bars.highs[idx]), float(bars.lows[idx]), float(bars.closes[idx]))
        for result in fills:
            if result.status is not FillStatus.FILLED:
                continue
            fill = IvbFill(
                oms_order_id=result.order.order_id,
                fill_price=result.fill_price,
                fill_qty=result.order.qty,
                symbol=trade_symbol,
                fill_time=result.fill_time,
                commission=result.commission,
                order_role=_ivb_order_role(result.order.tag),
            )
            state, child_actions, fill_events = core_logic.on_fill(state, fill)
            events.extend(fill_events)
            for action in child_actions:
                _submit_action(broker, action, ts, state)
            trade = _update_ledger(ledger, result, point_value=point_value)
            if trade is not None:
                trades.append(trade)
                equity += trade.pnl_dollars

        payload = _build_payload(data, config, bars, idx, equity, accepted_breaks, break_hold_seconds, levels_cache)
        state, actions, bar_events = core_logic.on_bar(state, payload)
        events.extend(bar_events)
        if must_flatten(ts) and state.positions.get(trade_symbol) is not None:
            state, flatten_actions, flatten_events = _request_flatten(state, payload, trade_symbol)
            actions.extend(flatten_actions)
            events.extend(flatten_events)
        for action in actions:
            _submit_action(broker, action, ts, state)

        mark = equity
        position = state.positions.get(trade_symbol)
        if position is not None:
            direction = 1 if position.direction is TradeDirection.LONG else -1
            mark += (float(bars.closes[idx]) - position.avg_entry) * direction * position.qty * point_value
        equity_curve.append(mark)
        timestamps.append(ts)

    result = IvbSymbolResult(
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
    return IvbBacktestResult(
        symbol_results={trade_symbol: result},
        combined_equity=result.equity_curve,
        combined_timestamps=result.timestamps,
        metrics=metrics_dict,
        decision_stream=result.decision_stream,
        trade_outcomes=result.trade_outcomes,
    )


def _build_payload(
    data: IvbAuctionData,
    config: IvbAuctionBacktestConfig,
    bars: NumpyBars,
    idx: int,
    equity: float,
    accepted_breaks: dict[str, TradeDirection],
    break_hold_seconds: dict[str, tuple[TradeDirection, float]],
    levels_cache,
) -> IvbBarInput:
    analysis_symbol = config.analysis_symbol
    trade_symbol = config.trade_symbol
    ts = _dt(bars.times[idx])
    et = to_et(ts)
    day = session_date(ts).isoformat()
    ivb = levels_cache.levels_by_date.get(day)
    block = get_session_block(et)
    ohlcv = (
        float(bars.opens[idx]),
        float(bars.highs[idx]),
        float(bars.lows[idx]),
        float(bars.closes[idx]),
        float(bars.volumes[idx]),
    )
    if ivb is None or et.time().hour < 10:
        return IvbBarInput(
            symbol=trade_symbol,
            bar_ts=ts,
            bar_ohlcv=ohlcv,
            session_block=block,
            ivb_levels=ivb,
            decision_code="IVB_FORMING" if ivb is None else "IVB_LOCKED",
            decision_details={"analysis_symbol": analysis_symbol, "trade_symbol": trade_symbol},
        )

    if not config.flags.disable_ivb_range_filter and not ivb_range_gate(ivb).passed:
        return IvbBarInput(
            symbol=trade_symbol,
            bar_ts=ts,
            bar_ohlcv=ohlcv,
            session_block=block,
            ivb_levels=ivb,
            decision_code="SESSION_FILTERED",
            decision_details={"range": ivb.range_pts, "analysis_symbol": analysis_symbol, "trade_symbol": trade_symbol},
        )

    previous_break = accepted_breaks.get(day, TradeDirection.FLAT)
    direction = TradeDirection.FLAT
    accepted = False
    if previous_break is TradeDirection.FLAT:
        if bars.closes[idx] > ivb.high:
            direction = TradeDirection.LONG
        elif bars.closes[idx] < ivb.low:
            direction = TradeDirection.SHORT
        if direction is not TradeDirection.FLAT:
            previous_hold_direction, previous_hold = break_hold_seconds.get(day, (TradeDirection.FLAT, 0.0))
            held_seconds = previous_hold + 60.0 if previous_hold_direction is direction else 60.0
            break_hold_seconds[day] = (direction, held_seconds)
            accepted, conditions = breakout_acceptance(
                direction=int(direction),
                close=float(bars.closes[idx]),
                high=float(bars.highs[idx]),
                low=float(bars.lows[idx]),
                ivb=ivb,
                held_seconds=held_seconds,
                breakout_volume=float(bars.volumes[idx]),
                rolling_volume_median=float(np.median(bars.volumes[max(0, idx - 10):idx + 1])),
                delta_60s=None,
                rolling_delta_median=None,
            )
            if accepted:
                accepted_breaks[day] = direction
        else:
            break_hold_seconds.pop(day, None)
    else:
        direction = previous_break
        accepted = True

    module = None
    trigger = None
    trade_direction = direction
    if accepted and direction is not TradeDirection.FLAT:
        module = IvbModule.A1_CONTINUATION
        trigger = EntryTrigger.PROFILE_RELOAD
    if config.flags.disable_a2_module is False and previous_break is TradeDirection.LONG and float(bars.closes[idx]) < ivb.high:
        module = IvbModule.A2_RECLAIM
        trigger = EntryTrigger.RECLAIM_RETEST
        trade_direction = TradeDirection.SHORT
    elif config.flags.disable_a2_module is False and previous_break is TradeDirection.SHORT and float(bars.closes[idx]) > ivb.low:
        module = IvbModule.A2_RECLAIM
        trigger = EntryTrigger.RECLAIM_RETEST
        trade_direction = TradeDirection.LONG

    footprint = data.footprint_by_minute.get(_minute_key(ts))
    if module is None or trigger is None:
        return IvbBarInput(
            symbol=trade_symbol,
            bar_ts=ts,
            bar_ohlcv=ohlcv,
            session_block=block,
            ivb_levels=ivb,
            breakout_direction=direction,
            breakout_accepted=accepted,
            footprint_state=footprint,
            decision_code="AWAITING_BREAK",
            decision_details={"analysis_symbol": analysis_symbol, "trade_symbol": trade_symbol},
        )

    regime_quality = 1.0 if accepted else 0.65
    retest_quality = _retest_quality(ohlcv, ivb, trade_direction)
    target_quality = 0.8
    volatility_quality = 1.0 if ivb_range_gate(ivb).passed else 0.0
    time_quality = 1.0 if config.flags.disable_time_filter or 10 <= et.hour < 12 else 0.65
    absorption_quality = None if config.flags.disable_absorption_gate or footprint is None else min(1.0, footprint.absorption_score / 1000.0)
    delta_quality = None if config.flags.disable_delta_gate or footprint is None else (1.0 if footprint.delta * int(trade_direction) > 0 else 0.0)
    score = score_signal(
        regime_quality=regime_quality,
        retest_quality=retest_quality,
        target_quality=target_quality,
        volatility_quality=volatility_quality,
        time_quality=time_quality,
        absorption_quality=absorption_quality,
        delta_confirmation=delta_quality,
    )
    entry = _entry_price(ohlcv, ivb, trade_direction, module)
    if module is IvbModule.A2_RECLAIM:
        stop = reclaim_stop(direction=trade_direction, ivb=ivb, failed_break_extreme=ohlcv[1] if trade_direction is TradeDirection.SHORT else ohlcv[2])
    else:
        stop = continuation_stop(direction=trade_direction, ivb=ivb)
    targets = (
        reclaim_targets(entry_price=entry, direction=trade_direction, ivb=ivb)
        if module is IvbModule.A2_RECLAIM
        else fallback_targets(entry_price=entry, direction=trade_direction, ivb=ivb)
    )
    rr = reward_to_risk(entry, stop, targets.tp1, trade_direction)
    min_score = A2_MIN_SCORE if module is IvbModule.A2_RECLAIM else A1_MIN_SCORE
    qty = config.fixed_qty or compute_position_size(
        equity=equity,
        module=module,
        size_multiplier=score.size_multiplier,
        entry=entry,
        stop=stop,
        symbol=trade_symbol,
    )
    chase_rejected = (
        module is IvbModule.A1_CONTINUATION
        and not config.flags.disable_chase_rejection
        and (
            (trade_direction is TradeDirection.LONG and float(bars.highs[idx]) > ivb.high + ivb_config.MAX_CHASE_EXTENSION_RANGE_FRACTION * ivb.range_pts)
            or (trade_direction is TradeDirection.SHORT and float(bars.lows[idx]) < ivb.low - ivb_config.MAX_CHASE_EXTENSION_RANGE_FRACTION * ivb.range_pts)
        )
        and retest_quality < 0.5
    )
    min_rr = ivb_config.A2_MIN_R_TO_TP1 if module is IvbModule.A2_RECLAIM else ivb_config.MIN_R_TO_TP1
    rr_rejected = (not config.flags.disable_target_gate) and rr < min_rr
    if not stop_within_cap(entry, stop, ivb) or rr_rejected or score.total < min_score or chase_rejected:
        qty = 0
    return IvbBarInput(
        symbol=trade_symbol,
        bar_ts=ts,
        bar_ohlcv=ohlcv,
        session_block=block,
        ivb_levels=ivb,
        breakout_direction=trade_direction,
        breakout_accepted=accepted,
        module=module,
        trigger=trigger,
        entry_price=entry,
        stop_price=stop,
        tp1_price=targets.tp1,
        tp2_price=targets.tp2,
        qty=qty,
        rr_to_tp1=rr,
        signal_score=score.total,
        size_multiplier=score.size_multiplier,
        footprint_state=footprint,
        decision_code="SIGNAL_EVALUATED",
        decision_details={
            "module": module.value,
            "score": score.total,
            "rr": rr,
            "footprint": score.footprint_available,
            "analysis_symbol": analysis_symbol,
            "trade_symbol": trade_symbol,
        },
    )


def _request_flatten(
    state: IvbAuctionCoreState,
    payload: IvbBarInput,
    symbol: str,
) -> tuple[IvbAuctionCoreState, list, list[DecisionEvent]]:
    actions: list = []
    events: list[DecisionEvent] = []
    next_state = state
    for setup in list(next_state.active_setups.values()):
        if setup.symbol == symbol and setup.qty_open > 0:
            next_state, new_actions, new_events = core_logic.on_bar(
                next_state,
                payload,
                flatten_request=IvbFlattenRequest(setup_id=setup.setup_id, symbol=symbol, reason="EOD_FLATTEN"),
            )
            actions.extend(new_actions)
            events.extend(new_events)
    return next_state, actions, events


def _submit_action(broker: SimBroker, action, ts: datetime, state: IvbAuctionCoreState | None = None) -> None:
    if isinstance(action, SubmitEntry):
        broker.submit_order(
            SimOrder(
                order_id=action.client_order_id,
                symbol=action.symbol,
                side=OrderSide.BUY if action.side == "BUY" else OrderSide.SELL,
                order_type=OrderType.LIMIT if action.order_type == "LIMIT" else OrderType.STOP if action.order_type == "STOP" else OrderType.MARKET,
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
    elif isinstance(action, SubmitPartialExit):
        order_type = OrderType.LIMIT if action.order_type == "LIMIT" else OrderType.MARKET
        broker.submit_order(
            SimOrder(
                order_id=action.client_order_id,
                symbol=action.symbol,
                side=OrderSide.BUY if action.side == "BUY" else OrderSide.SELL,
                order_type=order_type,
                qty=action.qty,
                limit_price=action.limit_price or action.price or 0.0,
                submit_time=ts,
                earliest_fill_time=ts,
                tag="partial",
                metadata=action.metadata,
            )
        )
    elif isinstance(action, FlattenPosition):
        setup_id = str(action.metadata.get("setup_id", ""))
        order_id = f"{action.symbol}-ivb-flatten-{setup_id or ts.strftime('%Y%m%d%H%M%S')}"
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
            module=order.metadata.get("module", ""),
            trigger=order.metadata.get("trigger", ""),
            stop_for_risk=order.metadata.get("stop_for_risk", result.fill_price),
        )
        return None
    open_trade = ledger.pop(setup_id, None)
    if open_trade is None:
        return None
    direction = 1 if open_trade.side == "BUY" else -1
    gross = (result.fill_price - open_trade.entry_price) * direction * open_trade.qty * point_value
    commission = open_trade.commission + result.commission
    net = gross - commission
    risk = abs(open_trade.entry_price - float(open_trade.stop_for_risk)) * open_trade.qty * point_value
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
        module=open_trade.module,
        trigger=open_trade.trigger,
    )


def _ivb_order_role(tag: str) -> str:
    if tag == "target":
        return "tp1"
    return tag if tag in {"entry", "stop", "partial", "flatten"} else "unknown"


def _entry_price(ohlcv: tuple[float, float, float, float, float], ivb, direction: TradeDirection, module: IvbModule) -> float:
    if module is IvbModule.A2_RECLAIM:
        return ivb.vah if direction is TradeDirection.SHORT else ivb.val
    if direction is TradeDirection.LONG:
        return ivb.vah or ivb.high
    return ivb.val or ivb.low


def _retest_quality(ohlcv: tuple[float, float, float, float, float], ivb, direction: TradeDirection) -> float:
    _, high, low, close, _ = ohlcv
    if direction is TradeDirection.LONG:
        distance = abs(low - (ivb.vah or ivb.high))
    else:
        distance = abs(high - (ivb.val or ivb.low))
    return max(0.0, min(1.0, 1.0 - distance / max(ivb.range_pts * 0.25, 1.0)))


def _minute_key(ts: datetime) -> datetime:
    return ts.replace(second=0, microsecond=0)


def _dt(value) -> datetime:
    import pandas as pd

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()
