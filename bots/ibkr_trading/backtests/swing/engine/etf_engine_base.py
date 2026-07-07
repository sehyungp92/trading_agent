"""Shared 15m ETF replay engine for TPC."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import inspect
from typing import Any

import numpy as np
import pandas as pd

from backtests.shared.parity.legacy_result_outputs import trade_outcomes_from_records
from backtests.swing.config_etf_base import ETFSlippageConfig
from backtests.swing.data.preprocessing import NumpyBars
from backtests.swing.engine.backtest_engine import SymbolResult, TradeRecord
from backtests.swing.engine.sim_broker import FillStatus, OrderSide, OrderType, SimBroker, SimOrder
from strategies.core.actions import (
    FlattenPosition,
    ReplaceProtectiveStop,
    SubmitAddOnEntry,
    SubmitEntry,
    SubmitPartialExit,
    SubmitProfitTarget,
    SubmitProtectiveStop,
)
from strategies.swing._shared.etf_core import BarData, BarWindow
from strategies.swing._shared.models import Direction


@dataclass
class ETFBacktestResult:
    strategy_id: str
    symbol_results: dict[str, SymbolResult] = field(default_factory=dict)
    combined_equity: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    combined_timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    decision_stream: list[dict[str, Any]] = field(default_factory=list)
    trade_outcomes: list[dict[str, Any]] = field(default_factory=list)
    total_commission: float = 0.0

    @property
    def trades(self) -> list[TradeRecord]:
        out: list[TradeRecord] = []
        for result in self.symbol_results.values():
            out.extend(result.trades)
        return out


@dataclass
class _OpenTrade:
    record: TradeRecord
    qty_open: int
    qty_initial: int
    direction: Direction
    setup_id: str
    risk_per_share: float
    risk_dollars: float = 0.0
    realised_pnl: float = 0.0
    commission: float = 0.0


class ETFStrategyBacktestEngine:
    def __init__(
        self,
        *,
        strategy_id: str,
        configs: dict[str, Any],
        core_logic: Any,
        state_factory: Any,
        bar_input_factory: Any,
        fill_factory: Any,
        order_update_factory: Any,
        indicator_module: Any,
        slippage: ETFSlippageConfig | None = None,
        initial_equity: float = 100_000.0,
        warmup_15m: int = 2_000,
        indicator_cache: dict | None = None,
    ) -> None:
        self.strategy_id = strategy_id
        self.configs = configs
        self.core_logic = core_logic
        self.state_factory = state_factory
        self.bar_input_factory = bar_input_factory
        self.fill_factory = fill_factory
        self.order_update_factory = order_update_factory
        self.indicator_module = indicator_module
        self.slippage = slippage or ETFSlippageConfig()
        self.initial_equity = initial_equity
        self.warmup_15m = warmup_15m
        self._indicator_cache = indicator_cache

    def run(self, replay_data: dict[str, dict[str, Any]]) -> ETFBacktestResult:
        broker = SimBroker(slippage_config=self.slippage)  # type: ignore[arg-type]
        state = self.state_factory()
        cash = self.initial_equity
        positions: dict[str, tuple[int, float]] = {}
        open_trades: dict[str, _OpenTrade] = {}
        order_context: dict[str, dict[str, Any]] = {}
        symbol_results = {sym: SymbolResult(symbol=sym) for sym in replay_data}
        equity_curve: list[float] = []
        timestamps: list[datetime] = []
        decision_stream: list[dict[str, Any]] = []

        # Phase 1: Pre-populate timestamp cache to avoid repeated pd.Timestamp conversions
        _dt_cache: dict[Any, datetime] = {}
        for sym_payload in replay_data.values():
            for key in ("bars_15m", "bars_30m", "bars_1h", "bars_4h", "bars_daily"):
                bars = sym_payload.get(key)
                if bars is not None and hasattr(bars, "times"):
                    for t in bars.times:
                        if t not in _dt_cache:
                            _dt_cache[t] = pd.Timestamp(t).to_pydatetime()

        def _cached_to_py_datetime(value: Any, _c: dict = _dt_cache) -> datetime:
            cached = _c.get(value)
            if cached is not None:
                return cached
            result = pd.Timestamp(value).to_pydatetime()
            _c[value] = result
            return result

        self._to_py_datetime = _cached_to_py_datetime  # type: ignore[assignment]

        prepared = {sym: self._prepare_symbol(sym, payload) for sym, payload in replay_data.items() if sym in self.configs}
        if not prepared:
            return ETFBacktestResult(strategy_id=self.strategy_id)
        primary_symbol = max(prepared, key=lambda s: len(prepared[s]["bars_15m"].closes))
        n = len(prepared[primary_symbol]["bars_15m"])
        start = min(max(self.warmup_15m, 1), max(n - 1, 1))

        for i in range(start, n):
            ts = self._time_at(prepared[primary_symbol]["bars_15m"], i)

            for symbol, payload in prepared.items():
                if i >= len(payload["bars_15m"]):
                    continue
                bar = payload["bars_15m"]
                self._mark_open_trade_excursion(
                    open_trades.get(symbol),
                    high=float(bar.highs[i]),
                    low=float(bar.lows[i]),
                )
                fills = broker.process_bar(
                    symbol,
                    self._time_at(bar, i),
                    float(bar.opens[i]),
                    float(bar.highs[i]),
                    float(bar.lows[i]),
                    float(bar.closes[i]),
                    self.configs[symbol].tick_size,
                )
                for result in fills:
                    cash, state, fill_events = self._handle_fill(
                        state=state,
                        result=result,
                        positions=positions,
                        open_trades=open_trades,
                        order_context=order_context,
                        symbol_results=symbol_results,
                        cash=cash,
                        broker=broker,
                        bar_index=i,
                    )
                    decision_stream.extend(fill_events)

            mtm_equity = self._mark_to_market(cash, positions, prepared, i)

            for symbol, payload in prepared.items():
                if i >= len(payload["bars_15m"]):
                    continue
                cfg = self.configs[symbol]
                bar_input = self._bar_input(symbol, payload, i, mtm_equity)
                if bar_input is None:
                    continue
                state, actions, events = self.core_logic.on_bar(state, bar_input, cfg)
                decision_stream.extend(self._events_to_dicts(events))
                action_ts = (
                    bar_input.bar_15m.timestamp
                    if bar_input.bar_15m is not None
                    else bar_input.timestamp or self._time_at(payload["bars_15m"], i)
                )
                self._submit_actions(actions, broker, order_context, cfg, timestamp=action_ts, bar_index=i)

            equity_curve.append(self._mark_to_market(cash, positions, prepared, i))
            timestamps.append(ts)

        # Restore class-level staticmethod
        del self._to_py_datetime  # type: ignore[misc]

        result = ETFBacktestResult(
            strategy_id=self.strategy_id,
            symbol_results=symbol_results,
            combined_equity=np.asarray(equity_curve, dtype=float),
            combined_timestamps=np.asarray(timestamps, dtype=object),
            decision_stream=decision_stream,
        )
        for sym_result in symbol_results.values():
            sym_result.decision_stream = [d for d in decision_stream if d.get("symbol") == sym_result.symbol]
            sym_result.trade_outcomes = trade_outcomes_from_records(sym_result.trades)
            result.total_commission += sym_result.total_commission
            result.trade_outcomes.extend(sym_result.trade_outcomes)
        return result

    def _prepare_symbol(self, symbol: str, payload: dict[str, Any]) -> dict[str, Any]:
        bars_15m = payload["bars_15m"]
        bars_30m = payload.get("bars_30m")
        bars_1h = payload["bars_1h"]
        bars_4h = payload["bars_4h"]
        bars_daily = payload.get("bars_daily")
        cfg = self.configs[symbol]

        # Phase 3: Indicator cache — check BEFORE building full BarWindows to avoid
        # expensive timestamp-tuple construction on cache hits during auto-opt.
        config_keys = getattr(self.indicator_module, "INDICATOR_CONFIG_KEYS", None)
        cache_key = None
        if config_keys and self._indicator_cache is not None:
            cache_key = (symbol, tuple(getattr(cfg, k, None) for k in config_keys))
            cached = self._indicator_cache.get(cache_key)
            if cached is not None:
                return {**payload, "indicators": cached}

        # Full BarWindows are only needed for compute_indicators (not used in _bar_input)
        full_15m = self._window(bars_15m, 0, len(bars_15m) - 1, include_times=True)
        full_30m = self._window(bars_30m, 0, len(bars_30m) - 1, include_times=True) if bars_30m is not None and len(bars_30m) else None
        full_1h = self._window(bars_1h, 0, len(bars_1h) - 1, include_times=True)
        full_4h = self._window(bars_4h, 0, len(bars_4h) - 1, include_times=True)
        full_daily = self._window(bars_daily, 0, len(bars_daily) - 1, include_times=True) if bars_daily is not None and len(bars_daily) else None

        param_count = len(inspect.signature(self.indicator_module.compute_indicators).parameters)
        if param_count == 6:
            indicators = self.indicator_module.compute_indicators(full_15m, full_30m, full_1h, full_4h, full_daily, cfg)
        elif param_count == 5:
            indicators = self.indicator_module.compute_indicators(full_15m, full_30m, full_1h, full_4h, cfg)
        else:
            indicators = self.indicator_module.compute_indicators(full_15m, full_1h, full_4h, cfg)

        if cache_key is not None:
            self._indicator_cache[cache_key] = indicators

        return {**payload, "indicators": indicators}

    def _bar_input(self, symbol: str, payload: dict[str, Any], i: int, equity: float):
        idx_30m = payload.get("idx_30m")
        idx_1h = payload.get("idx_1h")
        idx_4h = payload.get("idx_4h")
        idx_daily = payload.get("idx_daily")
        j30 = int(idx_30m[i]) if idx_30m is not None and i < len(idx_30m) and idx_30m[i] >= 0 else None
        j1 = int(idx_1h[i]) if idx_1h is not None and i < len(idx_1h) and idx_1h[i] >= 0 else -1
        j4 = int(idx_4h[i]) if idx_4h is not None and i < len(idx_4h) and idx_4h[i] >= 0 else -1
        jd = int(idx_daily[i]) if idx_daily is not None and i < len(idx_daily) and idx_daily[i] >= 0 else None
        if j1 < 0 or j4 < 0:
            return self.bar_input_factory(symbol=symbol, timestamp=self._time_at(payload["bars_15m"], i), equity=equity)
        indicators = self._indicator_snapshot(payload["indicators"], i, j30, j1, j4, jd)
        context_indicators = payload.get("context_indicators")
        if context_indicators:
            indicators = dict(indicators)
            indicators.update(self._context_indicator_snapshot(context_indicators, i))
        bars_15m = self._window(payload["bars_15m"], max(0, i - 3000), i)
        bars_30m = self._window(payload["bars_30m"], max(0, (j30 or 0) - 1500), j30) if j30 is not None and payload.get("bars_30m") is not None else None
        bars_1h = self._window(payload["bars_1h"], max(0, j1 - 1000), j1)
        bars_4h = self._window(payload["bars_4h"], max(0, j4 - 400), j4)
        bars_daily = self._window(payload["bars_daily"], max(0, jd - 260), jd) if jd is not None and payload.get("bars_daily") is not None else None
        bar = BarData(
            timestamp=self._time_at(payload["bars_15m"], i),
            open=float(payload["bars_15m"].opens[i]),
            high=float(payload["bars_15m"].highs[i]),
            low=float(payload["bars_15m"].lows[i]),
            close=float(payload["bars_15m"].closes[i]),
            volume=float(payload["bars_15m"].volumes[i]),
        )
        return self.bar_input_factory(
            symbol=symbol,
            bar_15m=bar,
            bars_15m=bars_15m,
            bars_30m=bars_30m,
            bars_1h=bars_1h,
            bars_4h=bars_4h,
            bars_daily=bars_daily,
            indicators=indicators,
            equity=equity,
            timestamp=bar.timestamp,
        )

    def _indicator_snapshot(self, arrays: Any, i15: int, i30: int | None, i1h: int, i4h: int, idaily: int | None) -> dict[str, float]:
        try:
            return self.indicator_module.snapshot(arrays, i15, i30, i1h, i4h, idaily)
        except TypeError:
            try:
                return self.indicator_module.snapshot(arrays, i15, i30, i1h, i4h)
            except TypeError:
                return self.indicator_module.snapshot(arrays, i15, i1h, i4h)

    @staticmethod
    def _context_indicator_snapshot(arrays: dict[str, np.ndarray], i15: int) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, values in arrays.items():
            if i15 < 0 or i15 >= len(values):
                out[key] = float("nan")
            else:
                out[key] = float(values[i15])
        return out

    def _handle_fill(
        self,
        *,
        state,
        result,
        positions: dict[str, tuple[int, float]],
        open_trades: dict[str, _OpenTrade],
        order_context: dict[str, dict[str, Any]],
        symbol_results: dict[str, SymbolResult],
        cash: float,
        broker: SimBroker,
        bar_index: int,
    ) -> tuple[float, Any, list[dict[str, Any]]]:
        order = result.order
        ctx = order_context.pop(order.order_id, {})
        if result.status != FillStatus.FILLED:
            update = self.order_update_factory(
                oms_order_id=order.order_id,
                status=result.status.name.lower(),
                symbol=order.symbol,
                timestamp=result.fill_time,
                order_role=str(ctx.get("role", order.tag) or "unknown"),
                decision_details={"fill_status": result.status.name.lower()},
            )
            state, actions, events = self.core_logic.on_order_update(state, update)
            self._submit_actions(actions, broker, order_context, self.configs[order.symbol], timestamp=result.fill_time, bar_index=bar_index)
            event_dicts = self._events_to_dicts(events)
            symbol_results[order.symbol].decision_stream.extend(event_dicts)
            return cash, state, event_dicts
        symbol = order.symbol
        role = ctx.get("role", order.tag)
        exit_type = ctx.get("exit_type", "")
        role_str = str(role)
        order_role = (
            "partial"
            if role_str.startswith("partial") or role in {"profit_target", "target", "t1"}
            else (
                "stop"
                if role == "stop"
                else (
                    "flatten"
                    if role == "flatten"
                    else ("add_on_entry" if role_str in {"add_on_entry", "addon", "addon_entry"} else "entry")
                )
            )
        )
        if order_role != "entry" and symbol not in open_trades:
            return cash, state, []
        effective_qty = int(order.qty)
        effective_commission = float(result.commission)
        if order_role not in {"entry", "add_on_entry"}:
            open_trade = open_trades.get(symbol)
            if open_trade is None:
                return cash, state, []
            effective_qty = min(effective_qty, max(open_trade.qty_open, 0))
            if effective_qty <= 0:
                return cash, state, []
            if order.qty > 0 and effective_qty != order.qty:
                effective_commission *= effective_qty / order.qty
        side_mult = 1 if order.side == OrderSide.BUY else -1
        signed_delta = side_mult * effective_qty
        cash -= signed_delta * result.fill_price
        cash -= effective_commission
        fill = self.fill_factory(
            oms_order_id=order.order_id,
            fill_price=result.fill_price,
            fill_qty=effective_qty,
            symbol=symbol,
            fill_time=result.fill_time,
            commission=effective_commission,
            order_role=order_role,
            exit_type=exit_type,
            decision_details={"reason": exit_type},
        )
        state, actions, events = self.core_logic.on_fill(state, fill)
        self._submit_actions(actions, broker, order_context, self.configs[symbol], timestamp=result.fill_time, bar_index=bar_index)
        event_dicts = self._events_to_dicts(events)
        symbol_results[symbol].decision_stream.extend(event_dicts)

        if order_role == "add_on_entry":
            open_trade = open_trades.get(symbol)
            if open_trade is None:
                return cash, state, event_dicts
            current_signed, current_avg = positions.get(symbol, (0, result.fill_price))
            old_abs = abs(current_signed)
            new_signed = current_signed + signed_delta
            new_abs = old_abs + effective_qty
            avg_price = ((current_avg * old_abs) + (result.fill_price * effective_qty)) / max(new_abs, 1)
            positions[symbol] = (new_signed, avg_price)

            old_open_qty = max(open_trade.qty_open, 0)
            open_trade.record.entry_price = (
                (open_trade.record.entry_price * old_open_qty) + (result.fill_price * effective_qty)
            ) / max(old_open_qty + effective_qty, 1)
            open_trade.qty_open += effective_qty
            open_trade.qty_initial += effective_qty
            open_trade.record.qty = open_trade.qty_initial
            open_trade.record.addon_a_qty += effective_qty

            stop_for_risk = float(ctx.get("stop_price", 0.0) or ctx.get("stop_for_risk", 0.0) or 0.0)
            addon_risk = abs(result.fill_price - stop_for_risk) if stop_for_risk > 0 else open_trade.risk_per_share
            open_trade.risk_dollars += max(addon_risk, 1e-9) * effective_qty
            open_trade.realised_pnl -= effective_commission
            open_trade.commission += effective_commission
            symbol_results[symbol].total_commission += effective_commission
            return cash, state, event_dicts

        if role == "entry":
            positions[symbol] = (signed_delta, result.fill_price)
            setup = state.setups.get(order.order_id)
            initial_stop = float(ctx.get("stop_price", 0.0) or (setup.stop_price if setup else 0.0))
            risk = max(abs(result.fill_price - initial_stop), 1e-9)
            direction = Direction.LONG if signed_delta > 0 else Direction.SHORT
            record = TradeRecord(
                symbol=symbol,
                direction=int(direction),
                entry_type=str(ctx.get("setup_type", "")),
                entry_time=result.fill_time,
                entry_price=result.fill_price,
                qty=effective_qty,
                initial_stop=initial_stop,
                signal_time=ctx.get("signal_time") or result.fill_time,
                fill_time=result.fill_time,
                signal_bar_index=int(ctx.get("signal_bar_index", -1)),
                fill_bar_index=bar_index,
                campaign_id=str(ctx.get("setup_id", order.order_id)),
                leg_type=str(ctx.get("entry_model", "")),
                regime_entry=str(ctx.get("grade", "")),
                score_entry=float(ctx.get("score", 0.0) or 0.0),
                quality_score=float(ctx.get("score", 0.0) or 0.0),
            )
            open_trades[symbol] = _OpenTrade(
                record,
                effective_qty,
                effective_qty,
                direction,
                order.order_id,
                risk,
                risk_dollars=risk * effective_qty,
                realised_pnl=-effective_commission,
                commission=effective_commission,
            )
            symbol_results[symbol].total_commission += effective_commission
            return cash, state, event_dicts

        open_trade = open_trades.get(symbol)
        if open_trade is not None:
            qty = min(effective_qty, open_trade.qty_open)
            if open_trade.direction == Direction.LONG:
                pnl = (result.fill_price - open_trade.record.entry_price) * qty
                new_signed = positions.get(symbol, (0, 0.0))[0] - qty
            else:
                pnl = (open_trade.record.entry_price - result.fill_price) * qty
                new_signed = positions.get(symbol, (0, 0.0))[0] + qty
            open_trade.realised_pnl += pnl - effective_commission
            open_trade.commission += effective_commission
            open_trade.qty_open -= qty
            symbol_results[symbol].total_commission += effective_commission
            if new_signed == 0 or open_trade.qty_open <= 0:
                positions.pop(symbol, None)
                rec = open_trade.record
                rec.exit_time = result.fill_time
                rec.exit_price = result.fill_price
                rec.exit_reason = exit_type or role.upper()
                rec.pnl_dollars = open_trade.realised_pnl
                rec.net_pnl = open_trade.realised_pnl
                rec.gross_pnl = open_trade.realised_pnl + open_trade.commission
                rec.pnl_points = open_trade.realised_pnl / max(open_trade.qty_initial, 1)
                rec.r_multiple = open_trade.realised_pnl / max(open_trade.risk_dollars, 1e-9)
                rec.commission = open_trade.commission
                if rec.entry_time and rec.exit_time:
                    rec.bars_held = max(1, int((rec.exit_time - rec.entry_time).total_seconds() // 900))
                symbol_results[symbol].trades.append(rec)
                open_trades.pop(symbol, None)
                self._cancel_symbol_pending_orders(broker, order_context, symbol)
            else:
                positions[symbol] = (new_signed, positions.get(symbol, (new_signed, result.fill_price))[1])
        return cash, state, event_dicts

    @staticmethod
    def _mark_open_trade_excursion(open_trade: _OpenTrade | None, *, high: float, low: float) -> None:
        if open_trade is None or open_trade.risk_per_share <= 0:
            return
        entry = open_trade.record.entry_price
        risk = open_trade.risk_per_share
        if open_trade.direction == Direction.LONG:
            mfe = (high - entry) / risk
            mae = (entry - low) / risk
        else:
            mfe = (entry - low) / risk
            mae = (high - entry) / risk
        open_trade.record.mfe_r = max(open_trade.record.mfe_r, float(mfe))
        open_trade.record.mae_r = max(open_trade.record.mae_r, float(mae))

    def _submit_actions(
        self,
        actions: list[Any],
        broker: SimBroker,
        order_context: dict[str, dict[str, Any]],
        cfg: Any,
        *,
        timestamp: datetime | None,
        bar_index: int | None = None,
    ) -> None:
        for action in actions:
            if isinstance(action, SubmitEntry):
                side = OrderSide.BUY if action.side == "BUY" else OrderSide.SELL
                order_type_name = str(action.order_type or "MARKET").upper()
                try:
                    order_type = OrderType[order_type_name]
                except KeyError:
                    order_type = OrderType.MARKET
                ttl_hours = float(action.metadata.get("ttl_hours", 0.0) or 0.0)
                order = SimOrder(
                    action.client_order_id,
                    action.symbol,
                    side,
                    order_type,
                    action.qty,
                    limit_price=action.limit_price,
                    stop_price=action.stop_price,
                    tick_size=cfg.tick_size,
                    submit_time=timestamp,
                    ttl_hours=ttl_hours,
                    tag="entry",
                )
                broker.submit_order(order)
                order_context[order.order_id] = {
                    "role": "entry",
                    "setup_id": action.metadata.get("setup_id", action.client_order_id),
                    "stop_price": action.risk_context.get("stop_for_risk", 0.0),
                    "setup_type": action.metadata.get("setup_type", ""),
                    "entry_model": action.metadata.get("entry_model", ""),
                    "grade": action.metadata.get("grade", ""),
                    "score": action.metadata.get("score", 0.0),
                    "signal_time": timestamp,
                    "signal_bar_index": bar_index if bar_index is not None else -1,
                }
            elif isinstance(action, SubmitAddOnEntry):
                side = OrderSide.BUY if action.side == "BUY" else OrderSide.SELL
                order_type_name = str(action.order_type or "MARKET").upper()
                try:
                    order_type = OrderType[order_type_name]
                except KeyError:
                    order_type = OrderType.MARKET
                order = SimOrder(
                    action.client_order_id,
                    action.symbol,
                    side,
                    order_type,
                    action.qty,
                    limit_price=action.limit_price,
                    stop_price=action.stop_price,
                    tick_size=cfg.tick_size,
                    submit_time=timestamp,
                    ttl_hours=0,
                    tag="add_on_entry",
                )
                broker.submit_order(order)
                order_context[order.order_id] = {
                    "role": "add_on_entry",
                    "setup_id": action.metadata.get("setup_id", action.client_order_id),
                    "stop_price": action.risk_context.get("stop_for_risk", 0.0),
                    "score": action.metadata.get("score", 0.0),
                    "signal_time": timestamp,
                    "signal_bar_index": bar_index if bar_index is not None else -1,
                }
            elif isinstance(action, SubmitProtectiveStop):
                side = OrderSide.SELL if action.side == "SELL" else OrderSide.BUY
                order = SimOrder(
                    action.client_order_id,
                    action.symbol,
                    side,
                    OrderType.STOP,
                    action.qty,
                    stop_price=action.stop_price,
                    tick_size=cfg.tick_size,
                    submit_time=timestamp,
                    ttl_hours=0,
                    tag="stop",
                )
                broker.submit_order(order)
                order_context[order.order_id] = {"role": "stop", "exit_type": "STOP"}
            elif isinstance(action, ReplaceProtectiveStop):
                for order in broker.pending_orders:
                    if order.order_id == action.target_order_id:
                        order.stop_price = action.stop_price
                        order.qty = action.qty
                        break
            elif isinstance(action, SubmitPartialExit):
                side = OrderSide.SELL if action.side == "SELL" else OrderSide.BUY
                reason = str(action.metadata.get("reason", "PARTIAL"))
                order = SimOrder(action.client_order_id, action.symbol, side, OrderType.MARKET, action.qty, tick_size=cfg.tick_size, submit_time=timestamp, tag=f"partial:{reason}")
                broker.submit_order(order)
                order_context[order.order_id] = {"role": "partial", "exit_type": reason}
            elif isinstance(action, SubmitProfitTarget):
                side = OrderSide.SELL if action.side == "SELL" else OrderSide.BUY
                reason = str(action.metadata.get("reason", "T1"))
                order = SimOrder(
                    action.client_order_id,
                    action.symbol,
                    side,
                    OrderType.LIMIT,
                    action.qty,
                    limit_price=action.limit_price,
                    tick_size=cfg.tick_size,
                    submit_time=timestamp,
                    ttl_hours=0,
                    tag="profit_target",
                )
                broker.submit_order(order)
                order_context[order.order_id] = {"role": "profit_target", "exit_type": reason}
            elif isinstance(action, FlattenPosition):
                if not action.side or action.qty <= 0:
                    continue
                side = OrderSide.SELL if action.side == "SELL" else OrderSide.BUY
                order_id = f"{action.symbol}-flatten-{timestamp.timestamp() if timestamp else broker.next_order_id()}"
                order = SimOrder(order_id, action.symbol, side, OrderType.MARKET, action.qty, tick_size=cfg.tick_size, submit_time=timestamp, tag="flatten")
                broker.submit_order(order)
                order_context[order.order_id] = {"role": "flatten", "exit_type": action.reason}

    @staticmethod
    def _cancel_symbol_pending_orders(
        broker: SimBroker,
        order_context: dict[str, dict[str, Any]],
        symbol: str,
    ) -> None:
        cancelled = broker.cancel_all(symbol)
        for order in cancelled:
            order_context.pop(order.order_id, None)

    def _mark_to_market(self, cash: float, positions: dict[str, tuple[int, float]], prepared: dict[str, dict[str, Any]], i: int) -> float:
        equity = cash
        for symbol, (signed_qty, _avg) in positions.items():
            bars = prepared[symbol]["bars_15m"]
            idx = min(i, len(bars.closes) - 1)
            equity += signed_qty * float(bars.closes[idx])
        return equity

    def _window(self, bars: NumpyBars, start: int, end: int | None, *, include_times: bool = False) -> BarWindow:
        if end is None or end < start:
            return BarWindow(np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), ())
        end = min(end, len(bars) - 1)
        sl = slice(start, end + 1)
        if include_times:
            times = tuple(self._to_py_datetime(x) for x in bars.times[sl])
        else:
            times = (self._to_py_datetime(bars.times[end]),)
        return BarWindow(
            opens=bars.opens[sl],
            highs=bars.highs[sl],
            lows=bars.lows[sl],
            closes=bars.closes[sl],
            volumes=bars.volumes[sl],
            times=times,
        )

    def _time_at(self, bars: NumpyBars, idx: int) -> datetime:
        return self._to_py_datetime(bars.times[idx])

    @staticmethod
    def _to_py_datetime(value: Any) -> datetime:
        return pd.Timestamp(value).to_pydatetime()

    @staticmethod
    def _events_to_dicts(events: list[Any]) -> list[dict[str, Any]]:
        out = []
        for event in events:
            out.append(
                {
                    "strategy_id": getattr(event, "strategy_id", ""),
                    "code": getattr(event, "code", ""),
                    "symbol": getattr(event, "symbol", ""),
                    "timeframe": getattr(event, "timeframe", ""),
                    "ts": getattr(event, "ts", None),
                    "details": dict(getattr(event, "details", {}) or {}),
                }
            )
        return out
