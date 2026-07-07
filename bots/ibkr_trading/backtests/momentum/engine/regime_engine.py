from __future__ import annotations

import contextlib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from backtests.momentum.config_regime import NqRegimeBacktestConfig
from backtests.momentum.data.cache import bar_path, load_bars
from backtests.momentum.data.preprocessing import NumpyBars, build_numpy_arrays, filter_rth, normalize_timezone
from backtests.momentum.engine.sim_broker import FillStatus, OrderSide, OrderType, SimBroker, SimOrder
from strategies.core.actions import CancelAction, FlattenPosition, ReplaceProtectiveStop, SubmitEntry, SubmitProfitTarget, SubmitProtectiveStop
from strategies.core.events import DecisionEvent
from strategies.momentum.nq_regime import config as nq_config
from strategies.momentum.nq_regime.core.data_policy import CompletedBarPolicy
from strategies.momentum.nq_regime.core.levels import KeyLevels
from strategies.momentum.nq_regime.core.logic import on_bar as core_on_bar
from strategies.momentum.nq_regime.core.logic import on_fill as core_on_fill
from strategies.momentum.nq_regime.core.logic import on_order_update as core_on_order_update
from strategies.momentum.nq_regime.core.state import BarData, FillEvent, OrderUpdateEvent, RegimeCoreState
from strategies.momentum.nq_regime.modules import liquidity_reversion, second_wind, structural_expansion
from strategies.scalp._shared.nq_contract import spec_for
from strategies.scalp._shared.time_utils import session_date, to_et


@dataclass
class NqRegimeData:
    analysis_symbol: str = "NQ"
    bars_5m: NumpyBars | None = None
    daily_context: dict[str, KeyLevels] = field(default_factory=dict)


@dataclass
class NqRegimeTradeRecord:
    symbol: str
    side: str
    qty: int
    entry_time: datetime
    exit_time: datetime | None = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    initial_stop: float = 0.0
    module: str = ""
    candidate_id: str = ""
    setup_type: str = ""
    entry_model: str = ""
    signal_time: datetime | None = None
    regime: str = ""
    grade: str = ""
    setup_score: int = 0
    target_room_r: float = 0.0
    stop_distance_points: float = 0.0
    initial_target: float = 0.0
    level: float = 0.0
    ib_type: str = ""
    penetration: float = 0.0
    value_factors: int = 0
    vwap_room_r: float = 0.0
    squeeze_duration: int = 0
    squeeze_range: float = 0.0
    volume_multiple: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    pnl_dollars: float = 0.0
    commission: float = 0.0
    r_multiple: float = 0.0
    exit_reason: str = ""
    mfe_r: float = 0.0
    mae_r: float = 0.0


@dataclass
class NqRegimeResult:
    symbol: str
    trades: list[NqRegimeTradeRecord] = field(default_factory=list)
    signal_events: list[DecisionEvent] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    metrics: dict[str, float] = field(default_factory=dict)
    decision_stream: list[dict[str, Any]] = field(default_factory=list)
    trade_outcomes: list[dict[str, Any]] = field(default_factory=list)


def load_nq_regime_data(config: NqRegimeBacktestConfig) -> NqRegimeData:
    path = bar_path(config.data_dir, config.analysis_symbol, "5m")
    if not path.exists():
        fallback = bar_path(config.data_dir, config.trade_symbol, "5m")
        path = fallback if fallback.exists() else path
    if not path.exists():
        return NqRegimeData(analysis_symbol=config.analysis_symbol, bars_5m=NumpyBars(np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])))
    raw_df = normalize_timezone(load_bars(path))
    daily_context = _build_daily_context(raw_df, timestamp_mode=config.bar_timestamp_mode)
    df = _replay_5m_frame(raw_df, timestamp_mode=config.bar_timestamp_mode)
    return NqRegimeData(
        analysis_symbol=config.analysis_symbol,
        bars_5m=build_numpy_arrays(df),
        daily_context=daily_context,
    )


def run_nq_regime_backtest(data: NqRegimeData, config: NqRegimeBacktestConfig) -> NqRegimeResult:
    if config.param_overrides:
        with _temporary_param_overrides(config.param_overrides):
            return _run_nq_regime_backtest(data, config)
    return _run_nq_regime_backtest(data, config)


def _run_nq_regime_backtest(data: NqRegimeData, config: NqRegimeBacktestConfig) -> NqRegimeResult:
    bars = data.bars_5m
    if bars is None or len(bars) == 0:
        return NqRegimeResult(symbol=config.trade_symbol, metrics=_extract_metrics([], [config.initial_equity], [], config.initial_equity, []))

    settings = nq_config.StrategyRuntimeSettings(
        analysis_symbol=config.analysis_symbol,
        trade_symbol=config.trade_symbol,
        initial_equity=config.initial_equity,
        max_contracts=config.fixed_qty or config.max_contracts,
        enable_structural_expansion=config.flags.enable_structural_expansion,
        enable_liquidity_reversion=config.flags.enable_liquidity_reversion,
        enable_second_wind=config.flags.enable_second_wind,
    )
    state = RegimeCoreState()
    policy = CompletedBarPolicy()
    broker = SimBroker(slippage_config=config.slippage)
    ledger = _TradeLedger(point_value=spec_for(config.trade_symbol).point_value)
    events: list[DecisionEvent] = []
    trades: list[NqRegimeTradeRecord] = []
    equity = config.initial_equity
    equity_curve: list[float] = []
    timestamps: list[datetime] = []

    for idx in range(len(bars)):
        ts = _dt(bars.times[idx])
        if config.start_date and ts < config.start_date:
            continue
        if config.end_date and ts > config.end_date:
            continue
        bar = BarData(
            ts=ts,
            open=float(bars.opens[idx]),
            high=float(bars.highs[idx]),
            low=float(bars.lows[idx]),
            close=float(bars.closes[idx]),
            volume=float(bars.volumes[idx]),
        )

        fills = broker.process_bar(
            config.trade_symbol,
            ts,
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            spec_for(config.trade_symbol).tick,
        )
        for fill_result in fills:
            state, equity, child_actions = _process_fill_result(
                fill_result,
                broker=broker,
                state=state,
                ledger=ledger,
                events=events,
                trades=trades,
                equity=equity,
                settings=settings,
                trade_symbol=config.trade_symbol,
                ts=ts,
            )
            if _same_bar_child_exit_allowed(fill_result, child_actions):
                same_bar_fills = broker.process_bar(
                    config.trade_symbol,
                    ts,
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    spec_for(config.trade_symbol).tick,
                )
                for child_fill_result in same_bar_fills:
                    state, equity, _ = _process_fill_result(
                        child_fill_result,
                        broker=broker,
                        state=state,
                        ledger=ledger,
                        events=events,
                        trades=trades,
                        equity=equity,
                        settings=settings,
                        trade_symbol=config.trade_symbol,
                        ts=ts,
                    )

        recent = [
            BarData(
                ts=_dt(bars.times[j]),
                open=float(bars.opens[j]),
                high=float(bars.highs[j]),
                low=float(bars.lows[j]),
                close=float(bars.closes[j]),
                volume=float(bars.volumes[j]),
            )
            for j in range(max(0, idx - 2), idx + 1)
        ]
        day_context = data.daily_context.get(session_date(ts).isoformat())
        event = policy.build_event(bar_5m=bar, recent_5m=recent, daily_context=day_context)
        state, actions, bar_events = core_on_bar(state, event, settings=settings)
        events.extend(bar_events)
        for action in actions:
            if isinstance(action, SubmitEntry) and config.fixed_qty:
                action = replace(action, qty=config.fixed_qty)
            state, update_events = _submit_action_and_updates(broker, state, action, ts)
            events.extend(update_events)

        ledger.mark_bar(ts, bar.high, bar.low)
        mark = equity + _open_pnl(state, bar.close, spec_for(config.trade_symbol).point_value)
        equity_curve.append(mark)
        timestamps.append(ts)

    result = NqRegimeResult(
        symbol=config.trade_symbol,
        trades=trades,
        signal_events=events,
        equity_curve=np.asarray(equity_curve, dtype=float),
        timestamps=np.asarray(timestamps),
    )
    result.metrics = _extract_metrics(trades, equity_curve, timestamps, config.initial_equity, events)
    result.decision_stream = _decision_stream(events, trades)
    result.trade_outcomes = _trade_outcomes(trades)
    return result


def _replay_5m_frame(raw_df: pd.DataFrame, *, timestamp_mode: str) -> pd.DataFrame:
    """Return the RTH replay frame with timestamps at bar-close availability."""
    if timestamp_mode == "close":
        return _filter_rth_close_labeled(raw_df)
    if timestamp_mode != "start":
        raise ValueError("timestamp_mode must be 'start' or 'close'")
    df = filter_rth(raw_df)
    shifted = df.copy()
    shifted.index = shifted.index + pd.Timedelta(minutes=5)
    return shifted


def _filter_rth_close_labeled(df: pd.DataFrame) -> pd.DataFrame:
    idx_et = df.index.tz_convert("America/New_York")
    minutes = idx_et.hour * 60 + idx_et.minute
    mask = (minutes > 540) & (minutes <= 960) & (idx_et.weekday < 5)
    return df.loc[mask]


def _process_fill_result(
    fill_result,
    *,
    broker: SimBroker,
    state: RegimeCoreState,
    ledger: "_TradeLedger",
    events: list[DecisionEvent],
    trades: list[NqRegimeTradeRecord],
    equity: float,
    settings: nq_config.StrategyRuntimeSettings,
    trade_symbol: str,
    ts: datetime,
) -> tuple[RegimeCoreState, float, list]:
    if fill_result.status is not FillStatus.FILLED:
        state, _, update_events = core_on_order_update(
            state,
            OrderUpdateEvent(
                oms_order_id=fill_result.order.order_id,
                status=fill_result.status.name.lower(),
                timestamp=fill_result.fill_time or ts,
                symbol=fill_result.order.symbol,
                order_role=_order_role(fill_result.order.tag),
                reason=fill_result.status.name.lower(),
            ),
        )
        events.extend(update_events)
        return state, equity, []
    fill = FillEvent(
        oms_order_id=fill_result.order.order_id,
        fill_price=fill_result.fill_price,
        fill_qty=fill_result.order.qty,
        fill_time=fill_result.fill_time or ts,
        symbol=trade_symbol,
        commission=fill_result.commission,
        order_role=_order_role(fill_result.order.tag),
    )
    state, child_actions, fill_events = core_on_fill(state, fill, settings=settings)
    events.extend(fill_events)
    completed = ledger.on_fill(fill_result, state)
    if completed is not None:
        trades.append(completed)
        equity += completed.pnl_dollars
    for action in child_actions:
        state, update_events = _submit_action_and_updates(broker, state, action, ts)
        events.extend(update_events)
    return state, equity, child_actions


def _same_bar_child_exit_allowed(fill_result, child_actions: list) -> bool:
    if fill_result.status is not FillStatus.FILLED or fill_result.order.tag != "entry":
        return False
    if not bool(getattr(fill_result, "filled_at_open", False)):
        return False
    return any(isinstance(action, (SubmitProtectiveStop, SubmitProfitTarget)) for action in child_actions)


class _TradeLedger:
    def __init__(self, *, point_value: float) -> None:
        self.point_value = point_value
        self.active: NqRegimeTradeRecord | None = None
        self.remaining_qty = 0
        self.realized_pnl = 0.0
        self.commission = 0.0
        self.mfe_price = 0.0
        self.mae_price = 0.0
        self.entry_filled_at_open = False

    def on_fill(self, fill, state: RegimeCoreState) -> NqRegimeTradeRecord | None:
        tag = fill.order.tag
        metadata = dict(getattr(fill.order, "metadata", {}) or {})
        side = "BUY" if fill.order.side is OrderSide.BUY else "SELL"
        if tag == "entry":
            details = dict(metadata.get("candidate_details", {}) or {})
            targets = tuple(metadata.get("targets", ()) or ())
            signal_ts = _parse_dt(metadata.get("signal_ts"))
            initial_stop = float(metadata.get("stop_for_risk", state.stop_price or fill.fill_price))
            self.active = NqRegimeTradeRecord(
                symbol=fill.order.symbol,
                side=side,
                qty=fill.order.qty,
                entry_time=fill.fill_time,
                entry_price=fill.fill_price,
                initial_stop=initial_stop,
                module=str(metadata.get("module", state.entry_module.value)),
                candidate_id=str(metadata.get("candidate_id", "")),
                setup_type=str(metadata.get("setup_type", "")),
                entry_model=str(metadata.get("entry_model", "")),
                signal_time=signal_ts,
                regime=str(details.get("regime", "")),
                grade=str(metadata.get("grade", state.setup_grade.value)),
                setup_score=int(metadata.get("score", state.setup_score) or 0),
                target_room_r=float(metadata.get("target_room_r", 0.0) or 0.0),
                stop_distance_points=abs(fill.fill_price - initial_stop),
                initial_target=float(targets[0]) if targets else 0.0,
                level=float(metadata.get("level", 0.0) or 0.0),
                ib_type=str(details.get("ib_type", "")),
                penetration=float(details.get("penetration", 0.0) or 0.0),
                value_factors=int(details.get("value_factors", 0) or 0),
                vwap_room_r=float(details.get("vwap_room_r", 0.0) or 0.0),
                squeeze_duration=int(details.get("squeeze_duration", 0) or 0),
                squeeze_range=float(details.get("squeeze_high", 0.0) or 0.0) - float(details.get("squeeze_low", 0.0) or 0.0),
                volume_multiple=float(details.get("volume_multiple", 0.0) or 0.0),
                details=details,
                commission=fill.commission,
            )
            self.remaining_qty = fill.order.qty
            self.realized_pnl = -fill.commission
            self.commission = fill.commission
            self.mfe_price = fill.fill_price
            self.mae_price = fill.fill_price
            self.entry_filled_at_open = bool(getattr(fill, "filled_at_open", False))
            return None
        if self.active is None:
            return None
        self._mark_price(fill.fill_price)
        direction = 1 if self.active.side == "BUY" else -1
        qty = min(fill.order.qty, self.remaining_qty)
        pnl = (fill.fill_price - self.active.entry_price) * direction * qty * self.point_value - fill.commission
        self.realized_pnl += pnl
        self.commission += fill.commission
        self.remaining_qty -= qty
        if self.remaining_qty > 0:
            return None
        completed = self.active
        completed.exit_time = fill.fill_time
        completed.exit_price = fill.fill_price
        completed.pnl_dollars = self.realized_pnl
        completed.commission = self.commission
        completed.exit_reason = str(metadata.get("exit_reason") or metadata.get("flatten_reason") or tag)
        risk = abs(completed.entry_price - completed.initial_stop) * completed.qty * self.point_value
        completed.r_multiple = completed.pnl_dollars / risk if risk > 0 else 0.0
        point_risk = abs(completed.entry_price - completed.initial_stop)
        if point_risk > 0:
            if completed.side == "BUY":
                completed.mfe_r = max(0.0, (self.mfe_price - completed.entry_price) / point_risk)
                completed.mae_r = min(0.0, (self.mae_price - completed.entry_price) / point_risk)
            else:
                completed.mfe_r = max(0.0, (completed.entry_price - self.mae_price) / point_risk)
                completed.mae_r = min(0.0, (completed.entry_price - self.mfe_price) / point_risk)
        self.active = None
        self.remaining_qty = 0
        self.entry_filled_at_open = False
        return completed

    def mark_bar(self, bar_time: datetime, high: float, low: float) -> None:
        if self.active is None:
            return
        if self.active.entry_time == bar_time and not self.entry_filled_at_open:
            return
        self.mfe_price = max(self.mfe_price, high)
        self.mae_price = min(self.mae_price, low)

    def _mark_price(self, price: float) -> None:
        self.mfe_price = max(self.mfe_price, price)
        self.mae_price = min(self.mae_price, price)


def _submit_action_and_updates(
    broker: SimBroker,
    state: RegimeCoreState,
    action: Any,
    ts: datetime,
) -> tuple[RegimeCoreState, list[DecisionEvent]]:
    updates = _submit_action(broker, action, ts)
    events: list[DecisionEvent] = []
    for update in updates:
        state, _, update_events = core_on_order_update(state, update)
        events.extend(update_events)
    return state, events


def _submit_action(broker: SimBroker, action: Any, ts: datetime) -> list[OrderUpdateEvent]:
    if isinstance(action, SubmitEntry):
        order_type = OrderType.MARKET if action.order_type == "MARKET" else OrderType.LIMIT if action.order_type == "LIMIT" else OrderType.STOP
        broker.submit_order(_with_metadata(
            SimOrder(
                    order_id=action.client_order_id,
                    symbol=action.symbol,
                    side=OrderSide.BUY if action.side == "BUY" else OrderSide.SELL,
                    order_type=order_type,
                    qty=action.qty,
                    limit_price=action.limit_price or action.price or 0.0,
                    stop_price=action.stop_price or action.limit_price or action.price or 0.0,
                    tick_size=spec_for(action.symbol).tick,
                    submit_time=ts,
                    ttl_minutes=int(action.metadata.get("ttl_minutes", 60) or 0),
                    tag="entry",
                    invalidation_price=float(action.risk_context.get("invalidation_price", 0.0) or 0.0),
                ),
            {**action.metadata, **action.risk_context},
        )
        )
    elif isinstance(action, SubmitProtectiveStop):
        broker.submit_order(_with_metadata(
            SimOrder(
                order_id=action.client_order_id,
                symbol=action.symbol,
                side=OrderSide.BUY if action.side == "BUY" else OrderSide.SELL,
                order_type=OrderType.STOP,
                qty=action.qty,
                stop_price=action.stop_price,
                tick_size=spec_for(action.symbol).tick,
                submit_time=ts,
                ttl_hours=0,
                tag="stop",
                oca_group=action.oca_group,
            ),
            action.metadata,
        )
        )
    elif isinstance(action, SubmitProfitTarget):
        broker.submit_order(_with_metadata(
            SimOrder(
                order_id=action.client_order_id,
                symbol=action.symbol,
                side=OrderSide.BUY if action.side == "BUY" else OrderSide.SELL,
                order_type=OrderType.LIMIT,
                qty=action.qty,
                limit_price=action.limit_price,
                tick_size=spec_for(action.symbol).tick,
                submit_time=ts,
                ttl_hours=0,
                tag=action.role or "target_1",
                oca_group=action.oca_group,
            ),
            action.metadata,
        )
        )
    elif isinstance(action, ReplaceProtectiveStop):
        broker.cancel_orders(action.symbol, tag="stop")
        broker.submit_order(_with_metadata(
            SimOrder(
                order_id=action.target_order_id,
                symbol=action.symbol,
                side=OrderSide.BUY if action.side == "BUY" else OrderSide.SELL,
                order_type=OrderType.STOP,
                qty=action.qty,
                stop_price=action.stop_price,
                tick_size=spec_for(action.symbol).tick,
                submit_time=ts,
                ttl_hours=0,
                tag="stop",
                oca_group=action.oca_group,
            ),
            action.metadata,
        )
        )
    elif isinstance(action, FlattenPosition):
        broker.cancel_orders(action.symbol)
        broker.submit_order(_with_metadata(
            SimOrder(
                order_id=f"{action.symbol}-nqreg-flatten-{ts.strftime('%Y%m%d%H%M%S')}",
                symbol=action.symbol,
                side=OrderSide.BUY if action.side == "BUY" else OrderSide.SELL,
                order_type=OrderType.MARKET,
                qty=action.qty,
                tick_size=spec_for(action.symbol).tick,
                submit_time=ts,
                tag="flatten",
            ),
            {
                **action.metadata,
                "exit_reason": action.reason or action.metadata.get("exit_reason", "flatten"),
                "flatten_reason": action.reason or action.metadata.get("flatten_reason", "flatten"),
            },
        )
        )
    elif isinstance(action, CancelAction):
        cancelled = _cancel_order_id(broker, action.target_order_id)
        if cancelled is not None:
            return [
                OrderUpdateEvent(
                    oms_order_id=cancelled.order_id,
                    status="cancelled",
                    timestamp=ts,
                    symbol=cancelled.symbol,
                    order_role=_order_role(cancelled.tag),
                    reason=action.reason,
                )
            ]
    return []


def _with_metadata(order: SimOrder, metadata: dict[str, Any]) -> SimOrder:
    order.metadata = dict(metadata)
    return order


def _cancel_order_id(broker: SimBroker, order_id: str) -> SimOrder | None:
    remaining: list[SimOrder] = []
    cancelled: SimOrder | None = None
    for order in broker.pending_orders:
        if order.order_id == order_id and cancelled is None:
            cancelled = order
        else:
            remaining.append(order)
    broker.pending_orders = remaining
    return cancelled


def _build_daily_context(df: pd.DataFrame, *, timestamp_mode: str = "start") -> dict[str, KeyLevels]:
    contexts: dict[str, KeyLevels] = {}
    if df.empty:
        return contexts
    df_et = df.copy()
    df_et.index = df_et.index.tz_convert("America/New_York")
    prior_rth: dict[Any, tuple[float, float, float]] = {}
    last_high = last_low = last_mid = 0.0
    last_vah = last_val = 0.0
    prior_value: dict[Any, tuple[float, float]] = {}
    for day, frame in df_et.groupby(df_et.index.date):
        prior_rth[day] = (last_high, last_low, last_mid)
        prior_value[day] = (last_vah, last_val)
        rth = _session_slice(frame, 9 * 60 + 30, 16 * 60, timestamp_mode=timestamp_mode)
        if not rth.empty:
            last_high = float(rth["high"].max())
            last_low = float(rth["low"].min())
            last_mid = (last_high + last_low) / 2.0
            last_vah, last_val = _value_area_from_frame(rth)

    trailing_week: list[tuple[float, float]] = []
    for day, frame in df_et.groupby(df_et.index.date):
        day_key = day.isoformat()
        pdh, pdl, pdm = prior_rth.get(day, (0.0, 0.0, 0.0))
        vah, val = prior_value.get(day, (0.0, 0.0))
        premarket = _session_slice(frame, 8 * 60 + 30, 9 * 60 + 30, timestamp_mode=timestamp_mode)
        overnight = _before_session(frame, 9 * 60 + 30, timestamp_mode=timestamp_mode)
        weekly_high = max((item[0] for item in trailing_week), default=0.0)
        weekly_low = min((item[1] for item in trailing_week), default=0.0)
        contexts[day_key] = KeyLevels(
            pdh=pdh,
            pdl=pdl,
            pdm=pdm,
            onh=float(overnight["high"].max()) if not overnight.empty else 0.0,
            onl=float(overnight["low"].min()) if not overnight.empty else 0.0,
            onm=_midpoint(overnight),
            pmh=float(premarket["high"].max()) if not premarket.empty else 0.0,
            pml=float(premarket["low"].min()) if not premarket.empty else 0.0,
            pmm=_midpoint(premarket),
            vah=vah,
            val=val,
            weekly_high=weekly_high,
            weekly_low=weekly_low,
        )
        rth = _session_slice(frame, 9 * 60 + 30, 16 * 60, timestamp_mode=timestamp_mode)
        if not rth.empty:
            trailing_week.append((float(rth["high"].max()), float(rth["low"].min())))
            trailing_week = trailing_week[-5:]
    return contexts


def _session_slice(frame: pd.DataFrame, start_minute: int, end_minute: int, *, timestamp_mode: str = "start") -> pd.DataFrame:
    minutes = frame.index.hour * 60 + frame.index.minute
    if timestamp_mode == "close":
        return frame[(minutes > start_minute) & (minutes <= end_minute)]
    return frame[(minutes >= start_minute) & (minutes < end_minute)]


def _before_session(frame: pd.DataFrame, end_minute: int, *, timestamp_mode: str = "start") -> pd.DataFrame:
    minutes = frame.index.hour * 60 + frame.index.minute
    if timestamp_mode == "close":
        return frame[minutes <= end_minute]
    return frame[minutes < end_minute]


def _midpoint(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return (float(frame["high"].max()) + float(frame["low"].min())) / 2.0


def _value_area_from_frame(frame: pd.DataFrame) -> tuple[float, float]:
    if frame.empty or "volume" not in frame.columns:
        return 0.0, 0.0
    prices = frame["close"].dropna()
    volumes = frame.loc[prices.index, "volume"].fillna(0.0)
    if prices.empty or float(volumes.sum()) <= 0:
        return 0.0, 0.0
    bins = (prices / 5.0).round() * 5.0
    profile = volumes.groupby(bins).sum().sort_values(ascending=False)
    target_volume = float(profile.sum()) * 0.70
    selected: list[float] = []
    running = 0.0
    for price, volume in profile.items():
        selected.append(float(price))
        running += float(volume)
        if running >= target_volume:
            break
    return (max(selected), min(selected)) if selected else (0.0, 0.0)


@contextlib.contextmanager
def _temporary_param_overrides(overrides: dict[str, Any]) -> Iterator[None]:
    modules = (nq_config, structural_expansion, liquidity_reversion, second_wind)
    originals: list[tuple[Any, str, Any]] = []
    try:
        for key, value in overrides.items():
            for module in modules:
                if hasattr(module, key):
                    originals.append((module, key, getattr(module, key)))
                    setattr(module, key, value)
        yield
    finally:
        for module, key, value in reversed(originals):
            setattr(module, key, value)


def _extract_metrics(
    trades: list[NqRegimeTradeRecord],
    equity_curve: list[float] | np.ndarray,
    timestamps: list[datetime] | np.ndarray,
    initial_equity: float,
    signal_events: list[DecisionEvent] | None = None,
) -> dict[str, float]:
    total = len(trades)
    pnl = sum(trade.pnl_dollars for trade in trades)
    wins = [trade for trade in trades if trade.pnl_dollars > 0]
    losses = [trade for trade in trades if trade.pnl_dollars < 0]
    gross_win = sum(trade.pnl_dollars for trade in wins)
    gross_loss = abs(sum(trade.pnl_dollars for trade in losses))
    equity_arr = np.asarray(equity_curve, dtype=float)
    max_dd = 0.0
    if len(equity_arr):
        peaks = np.maximum.accumulate(equity_arr)
        dd = (peaks - equity_arr) / np.maximum(peaks, 1.0)
        max_dd = float(np.max(dd))
    days = max(1.0, (timestamps[-1] - timestamps[0]).days + 1 if len(timestamps) >= 2 else 1.0)
    total_r = float(sum(trade.r_multiple for trade in trades))
    avg_r = float(np.mean([trade.r_multiple for trade in trades])) if trades else 0.0
    mfe_capture = _mfe_capture(trades)
    positive_mfe_loser_rate = _positive_mfe_loser_rate(trades)
    metrics = {
        "total_trades": float(total),
        "net_profit": float(pnl),
        "net_return_pct": float(pnl / initial_equity * 100.0) if initial_equity > 0 else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else (10.0 if gross_win > 0 else 0.0),
        "win_rate": float(len(wins) / total) if total else 0.0,
        "avg_r": avg_r,
        "total_r": total_r,
        "total_r_per_month": float(total_r / days * 21.0),
        "max_drawdown_pct": max_dd,
        "trades_per_month": float(total / days * 21.0),
        "expectancy_dollar": float(pnl / total) if total else 0.0,
        "mfe_capture": mfe_capture,
        "positive_mfe_loser_rate": positive_mfe_loser_rate,
    }
    metrics.update(_module_metrics(trades))
    metrics.update(_second_wind_setup_type_metrics(trades, days))
    metrics.update(_routing_metrics(signal_events or []))
    metrics.update(_liquidity_reversion_health_metrics(trades, signal_events or [], days))
    metrics.update(_second_wind_health_metrics(trades, signal_events or [], days))
    selected_total = (
        metrics.get("routing_structural_expansion_selected", 0.0)
        + metrics.get("routing_liquidity_reversion_selected", 0.0)
        + metrics.get("routing_second_wind_selected", 0.0)
    )
    metrics["execution_conversion"] = float(total / selected_total) if selected_total else 0.0
    return metrics


def _mfe_capture(trades: list[NqRegimeTradeRecord]) -> float:
    eligible = [trade for trade in trades if trade.mfe_r > 0]
    if not eligible:
        return 0.0
    captures = [max(0.0, trade.r_multiple) / max(trade.mfe_r, 1e-9) for trade in eligible]
    return float(np.mean([min(1.0, value) for value in captures]))


def _positive_mfe_loser_rate(trades: list[NqRegimeTradeRecord]) -> float:
    losers = [trade for trade in trades if trade.r_multiple <= 0]
    if not losers:
        return 0.0
    leaked = [trade for trade in losers if trade.mfe_r >= 0.5]
    return float(len(leaked) / len(losers))


def _module_metrics(trades: list[NqRegimeTradeRecord]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    modules = ("structural_expansion", "liquidity_reversion", "second_wind")
    active_modules = 0
    min_trades = None
    for module in modules:
        group = [trade for trade in trades if trade.module == module]
        count = len(group)
        if count:
            active_modules += 1
        min_trades = count if min_trades is None else min(min_trades, count)
        wins = [trade for trade in group if trade.pnl_dollars > 0]
        gross_win = sum(trade.pnl_dollars for trade in group if trade.pnl_dollars > 0)
        gross_loss = abs(sum(trade.pnl_dollars for trade in group if trade.pnl_dollars < 0))
        prefix = f"module_{module}"
        metrics[f"{prefix}_trades"] = float(count)
        metrics[f"{prefix}_avg_r"] = float(np.mean([trade.r_multiple for trade in group])) if group else 0.0
        metrics[f"{prefix}_total_r"] = float(sum(trade.r_multiple for trade in group))
        metrics[f"{prefix}_win_rate"] = float(len(wins) / count) if count else 0.0
        metrics[f"{prefix}_profit_factor"] = float(gross_win / gross_loss) if gross_loss > 0 else (10.0 if gross_win > 0 else 0.0)
        metrics[f"{prefix}_mfe_capture"] = _mfe_capture(group)
        metrics[f"{prefix}_positive_mfe_loser_rate"] = _positive_mfe_loser_rate(group)
        metrics[f"{prefix}_avg_mfe_r"] = float(np.mean([trade.mfe_r for trade in group])) if group else 0.0
        metrics[f"{prefix}_avg_mae_r"] = float(np.mean([trade.mae_r for trade in group])) if group else 0.0
        abs_total_r = sum(abs(trade.r_multiple) for trade in group)
        metrics[f"{prefix}_top_trade_share"] = (
            float(max((abs(trade.r_multiple) for trade in group), default=0.0) / abs_total_r)
            if abs_total_r > 0
            else 0.0
        )
    metrics["module_coverage"] = active_modules / len(modules)
    metrics["min_module_trades"] = float(min_trades or 0)
    return metrics


def _second_wind_setup_type_metrics(trades: list[NqRegimeTradeRecord], days: float) -> dict[str, float]:
    metrics: dict[str, float] = {}
    setup_types = (
        "pm_squeeze_fire",
        "pm_vwap_reclaim",
        "pm_micro_compression_break",
        "pm_range_acceptance",
        "pm_second_leg",
    )
    for setup_type in setup_types:
        group = [trade for trade in trades if trade.module == "second_wind" and trade.setup_type == setup_type]
        count = len(group)
        gross_win = sum(trade.pnl_dollars for trade in group if trade.pnl_dollars > 0)
        gross_loss = abs(sum(trade.pnl_dollars for trade in group if trade.pnl_dollars < 0))
        wins = [trade for trade in group if trade.pnl_dollars > 0]
        prefix = f"module_second_wind_{setup_type}"
        metrics[f"{prefix}_trades"] = float(count)
        metrics[f"{prefix}_avg_r"] = float(np.mean([trade.r_multiple for trade in group])) if group else 0.0
        metrics[f"{prefix}_total_r"] = float(sum(trade.r_multiple for trade in group))
        metrics[f"{prefix}_total_r_per_month"] = float(sum(trade.r_multiple for trade in group) / days * 21.0)
        metrics[f"{prefix}_trades_per_month"] = float(count / days * 21.0)
        metrics[f"{prefix}_win_rate"] = float(len(wins) / count) if count else 0.0
        metrics[f"{prefix}_profit_factor"] = float(gross_win / gross_loss) if gross_loss > 0 else (10.0 if gross_win > 0 else 0.0)
        metrics[f"{prefix}_mfe_capture"] = _mfe_capture(group)
        metrics[f"{prefix}_positive_mfe_loser_rate"] = _positive_mfe_loser_rate(group)
    return metrics


def _liquidity_reversion_health_metrics(
    trades: list[NqRegimeTradeRecord],
    signal_events: list[DecisionEvent],
    days: float,
) -> dict[str, float]:
    group = [trade for trade in trades if trade.module == "liquidity_reversion"]
    requested = [
        event for event in signal_events
        if event.code == "ENTRY_REQUESTED" and event.details.get("module") == "liquidity_reversion"
    ]
    selected = [
        event for event in signal_events
        if event.code == "ROUTING_DECISION" and event.details.get("selected_module") == "liquidity_reversion"
    ]
    return {
        "module_liquidity_reversion_total_r_per_month": float(sum(trade.r_multiple for trade in group) / days * 21.0),
        "module_liquidity_reversion_trades_per_month": float(len(group) / days * 21.0),
        "module_liquidity_reversion_mfe_capture": _mfe_capture(group),
        "module_liquidity_reversion_positive_mfe_loser_rate": _positive_mfe_loser_rate(group),
        "routing_liquidity_reversion_entry_requested": float(len(requested)),
        "routing_liquidity_reversion_request_to_fill_rate": float(len(group) / len(requested)) if requested else 0.0,
        "routing_liquidity_reversion_selected_to_fill_rate": float(len(group) / len(selected)) if selected else 0.0,
    }


def _second_wind_health_metrics(
    trades: list[NqRegimeTradeRecord],
    signal_events: list[DecisionEvent],
    days: float,
) -> dict[str, float]:
    group = [trade for trade in trades if trade.module == "second_wind"]
    requested = [
        event for event in signal_events
        if event.code == "ENTRY_REQUESTED" and event.details.get("module") == "second_wind"
    ]
    selected = [
        event for event in signal_events
        if event.code == "ROUTING_DECISION" and event.details.get("selected_module") == "second_wind"
    ]
    return {
        "module_second_wind_total_r_per_month": float(sum(trade.r_multiple for trade in group) / days * 21.0),
        "module_second_wind_trades_per_month": float(len(group) / days * 21.0),
        "module_second_wind_mfe_capture": _mfe_capture(group),
        "module_second_wind_positive_mfe_loser_rate": _positive_mfe_loser_rate(group),
        "routing_second_wind_entry_requested": float(len(requested)),
        "routing_second_wind_request_to_fill_rate": float(len(group) / len(requested)) if requested else 0.0,
        "routing_second_wind_selected_to_fill_rate": float(len(group) / len(selected)) if selected else 0.0,
    }


def _routing_metrics(signal_events: list[DecisionEvent]) -> dict[str, float]:
    routing = [event for event in signal_events if event.code == "ROUTING_DECISION"]
    selected = [event for event in routing if event.details.get("selected")]
    blocked = sum(float(event.details.get("blocked", 0) or 0) for event in routing)
    total = len(routing)
    metrics = {
        "routing_decisions": float(total),
        "routing_selected_rate": float(len(selected) / total) if total else 0.0,
        "routing_avg_blocked": float(blocked / total) if total else 0.0,
        "entry_blocked_by_session": float(sum(1 for event in signal_events if event.code == "ENTRY_BLOCKED_BY_SESSION")),
        "entry_blocked_by_size": float(sum(1 for event in signal_events if event.code == "ENTRY_BLOCKED_BY_SIZE")),
        "news_veto_events": float(sum(1 for event in signal_events if event.code == "NEWS_VETO")),
        "daily_lockout_events": float(sum(1 for event in signal_events if event.code == "DAILY_LOCKOUT")),
    }
    candidate_totals: dict[str, int] = {module: 0 for module in ("structural_expansion", "liquidity_reversion", "second_wind")}
    valid_candidate_totals: dict[str, int] = {module: 0 for module in candidate_totals}
    selected_totals: dict[str, int] = {module: 0 for module in candidate_totals}
    blocked_totals: dict[str, int] = {module: 0 for module in candidate_totals}
    valid_blocked_totals: dict[str, int] = {module: 0 for module in candidate_totals}
    valid_regime_mismatch_totals: dict[str, int] = {module: 0 for module in candidate_totals}
    candidate_count = 0
    for event in routing:
        details = event.details
        inventory = list(details.get("candidate_inventory") or [])
        candidate_count += len(inventory)
        for item in inventory:
            module = str(item.get("module", ""))
            if module in candidate_totals:
                candidate_totals[module] += 1
                if bool(item.get("valid")):
                    valid_candidate_totals[module] += 1
        selected_module = str(details.get("selected_module", ""))
        if selected_module in selected_totals and details.get("selected"):
            selected_totals[selected_module] += 1
        for item in details.get("blocked_candidates") or []:
            module = str(item.get("module", ""))
            if module in blocked_totals:
                blocked_totals[module] += 1
                if bool(item.get("valid")):
                    valid_blocked_totals[module] += 1
                    if str(item.get("block_reason", "")) == "regime_mismatch":
                        valid_regime_mismatch_totals[module] += 1
    metrics["routing_candidate_events"] = float(candidate_count)
    metrics["routing_candidates_per_decision"] = float(candidate_count / total) if total else 0.0
    for module in candidate_totals:
        prefix = f"routing_{module}"
        metrics[f"{prefix}_candidates"] = float(candidate_totals[module])
        metrics[f"{prefix}_valid_candidates"] = float(valid_candidate_totals[module])
        metrics[f"{prefix}_selected"] = float(selected_totals[module])
        metrics[f"{prefix}_blocked"] = float(blocked_totals[module])
        metrics[f"{prefix}_valid_blocked"] = float(valid_blocked_totals[module])
        metrics[f"{prefix}_valid_regime_mismatch"] = float(valid_regime_mismatch_totals[module])
        metrics[f"{prefix}_select_rate_when_candidate"] = (
            float(selected_totals[module] / candidate_totals[module]) if candidate_totals[module] else 0.0
        )
        metrics[f"{prefix}_select_rate_when_valid"] = (
            float(selected_totals[module] / valid_candidate_totals[module]) if valid_candidate_totals[module] else 0.0
        )
    return metrics


def _decision_stream(events: list[DecisionEvent], trades: list[NqRegimeTradeRecord]) -> list[dict[str, Any]]:
    rows = [
        {"ts": event.ts.isoformat(), "symbol": event.symbol, "code": event.code, "timeframe": event.timeframe, "details": event.details}
        for event in events
    ]
    for trade in trades:
        rows.append(
            {
                "ts": trade.exit_time.isoformat() if trade.exit_time else trade.entry_time.isoformat(),
                "symbol": trade.symbol,
                "code": "TRADE_CLOSED",
                "timeframe": "5m",
                "details": {"pnl": trade.pnl_dollars, "r": trade.r_multiple, "module": trade.module},
            }
        )
    return sorted(rows, key=lambda row: row["ts"])


def _trade_outcomes(trades: list[NqRegimeTradeRecord]) -> list[dict[str, Any]]:
    return [
        {
            "symbol": trade.symbol,
            "entry_time": trade.entry_time.isoformat(),
            "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
            "pnl_dollars": trade.pnl_dollars,
            "r_multiple": trade.r_multiple,
            "module": trade.module,
            "grade": trade.grade,
            "setup_type": trade.setup_type,
            "entry_model": trade.entry_model,
            "setup_score": trade.setup_score,
            "mfe_r": trade.mfe_r,
            "mae_r": trade.mae_r,
            "exit_reason": trade.exit_reason,
        }
        for trade in trades
    ]


def _open_pnl(state: RegimeCoreState, close: float, point_value: float) -> float:
    if state.position_side is nq_config.TradeSide.FLAT or state.qty_open <= 0:
        return 0.0
    return (close - state.entry_price) * state.position_side.sign * state.qty_open * point_value


def _order_role(tag: str) -> str:
    if tag in {"entry", "stop", "target_1", "target_2", "partial", "flatten"}:
        return tag
    if tag.startswith("target"):
        return "target_1"
    return "unknown"


def _dt(value: Any) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return _dt(value)
    except Exception:
        return None
