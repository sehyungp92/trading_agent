"""Trend exit management: TP1/TP2, smart BE, time stop, scratch exit, EMA fail-safe."""

from __future__ import annotations

from dataclasses import dataclass, field

from crypto_trader.core.models import Bar, Order, OrderType, Position, Side

from .config import TrendExitParams


@dataclass
class TrendExitState:
    entry_price: float
    stop_distance: float
    original_qty: float
    remaining_qty: float
    direction: Side
    current_stop_order_id: str | None = None
    mfe_r: float = 0.0
    mae_r: float = 0.0
    peak_r: float = 0.0
    current_r: float = 0.0
    bars_since_entry: int = 0
    bars_above_1r: int = 0
    tp1_hit: bool = False
    tp2_hit: bool = False
    be_moved: bool = False
    partial_exits: list[float] = field(default_factory=list)


class ExitManager:
    """Manage exits: TP1/TP2, breakeven, scratch exit, time stop, EMA fail-safe."""

    def __init__(self, cfg: TrendExitParams) -> None:
        self._cfg = cfg
        self._states: dict[str, TrendExitState] = {}
        self._next_order_id = 0

    def init_position(
        self,
        sym: str,
        entry_price: float,
        stop_distance: float,
        qty: float,
        direction: Side,
        stop_order_id: str | None = None,
    ) -> None:
        self._states[sym] = TrendExitState(
            entry_price=entry_price,
            stop_distance=stop_distance,
            original_qty=qty,
            remaining_qty=qty,
            direction=direction,
            current_stop_order_id=stop_order_id,
        )

    def get_state(self, sym: str) -> TrendExitState | None:
        return self._states.get(sym)

    def remove_position(self, sym: str) -> TrendExitState | None:
        return self._states.pop(sym, None)

    def manage(
        self,
        position: Position,
        bar: Bar,
        h1_bars: list[Bar],
        h1_ind,  # IndicatorSnapshot
        broker,
    ) -> list[Order]:
        sym = position.symbol
        state = self._states.get(sym)
        if state is None:
            return []

        cfg = self._cfg
        orders: list[Order] = []
        direction = state.direction
        full_close_requested = False

        state.bars_since_entry += 1
        state.remaining_qty = position.qty

        if state.stop_distance > 0:
            if direction == Side.LONG:
                state.current_r = (bar.close - state.entry_price) / state.stop_distance
                peak_r = (bar.high - state.entry_price) / state.stop_distance
                trough_r = (bar.low - state.entry_price) / state.stop_distance
            else:
                state.current_r = (state.entry_price - bar.close) / state.stop_distance
                peak_r = (state.entry_price - bar.low) / state.stop_distance
                trough_r = (state.entry_price - bar.high) / state.stop_distance

            state.peak_r = max(state.peak_r, peak_r)
            state.mfe_r = max(state.mfe_r, peak_r)
            state.mae_r = min(state.mae_r, trough_r)

        if state.current_r >= 1.0:
            state.bars_above_1r += 1
        else:
            state.bars_above_1r = 0

        if not state.tp1_hit and state.peak_r >= cfg.tp1_r:
            tp1_qty = state.original_qty * cfg.tp1_frac
            if tp1_qty > 0 and state.remaining_qty > tp1_qty:
                reverse_side = Side.SHORT if direction == Side.LONG else Side.LONG
                orders.append(Order(
                    order_id=self._gen_id(sym, "tp1"),
                    symbol=sym,
                    side=reverse_side,
                    order_type=OrderType.MARKET,
                    qty=tp1_qty,
                    tag="tp1",
                ))
                state.tp1_hit = True
                state.partial_exits.append(tp1_qty)

        if (cfg.be_after_tp1 and state.tp1_hit and not state.be_moved
                and state.bars_above_1r >= cfg.be_min_bars_above):
            state.be_moved = True

        if state.tp1_hit and not state.tp2_hit and state.peak_r >= cfg.tp2_r:
            tp2_qty = state.original_qty * cfg.tp2_frac
            if tp2_qty > 0 and state.remaining_qty > tp2_qty:
                reverse_side = Side.SHORT if direction == Side.LONG else Side.LONG
                orders.append(Order(
                    order_id=self._gen_id(sym, "tp2"),
                    symbol=sym,
                    side=reverse_side,
                    order_type=OrderType.MARKET,
                    qty=tp2_qty,
                    tag="tp2",
                ))
                state.tp2_hit = True
                state.partial_exits.append(tp2_qty)

        if (cfg.scratch_exit_enabled
                and not state.tp1_hit
                and state.bars_since_entry >= cfg.scratch_min_bars
                and state.peak_r >= cfg.scratch_peak_r
                and state.current_r <= cfg.scratch_floor_r
                and state.remaining_qty > 0):
            reverse_side = Side.SHORT if direction == Side.LONG else Side.LONG
            orders.append(Order(
                order_id=self._gen_id(sym, "scratch_exit"),
                symbol=sym,
                side=reverse_side,
                order_type=OrderType.MARKET,
                qty=state.remaining_qty,
                tag="scratch_exit",
            ))
            full_close_requested = True

        if (not full_close_requested
                and cfg.mfe_lock_exit_enabled
                and state.bars_since_entry >= cfg.mfe_lock_min_bars
                and state.peak_r >= cfg.mfe_lock_trigger_r
                and state.current_r <= cfg.mfe_lock_floor_r
                and state.remaining_qty > 0):
            reverse_side = Side.SHORT if direction == Side.LONG else Side.LONG
            orders.append(Order(
                order_id=self._gen_id(sym, "mfe_lock_exit"),
                symbol=sym,
                side=reverse_side,
                order_type=OrderType.MARKET,
                qty=state.remaining_qty,
                tag="mfe_lock_exit",
            ))
            full_close_requested = True

        if (not full_close_requested
                and state.bars_since_entry >= cfg.time_stop_bars
                and state.current_r < cfg.time_stop_min_progress_r
                and not state.tp1_hit):
            reverse_side = Side.SHORT if direction == Side.LONG else Side.LONG
            exit_qty = state.remaining_qty if cfg.time_stop_action == "exit" else state.remaining_qty * 0.5
            if exit_qty > 0:
                orders.append(Order(
                    order_id=self._gen_id(sym, "time_stop"),
                    symbol=sym,
                    side=reverse_side,
                    order_type=OrderType.MARKET,
                    qty=exit_qty,
                    tag="time_stop",
                ))
                if exit_qty >= state.remaining_qty:
                    full_close_requested = True

        if (cfg.quick_exit_enabled
                and not full_close_requested
                and not any(o.tag == "time_stop" for o in orders)
                and state.bars_since_entry >= cfg.quick_exit_bars
                and state.mfe_r < cfg.quick_exit_max_mfe_r
                and state.current_r <= cfg.quick_exit_max_r
                and state.remaining_qty > 0):
            reverse_side = Side.SHORT if direction == Side.LONG else Side.LONG
            orders.append(Order(
                order_id=self._gen_id(sym, "quick_exit"),
                symbol=sym,
                side=reverse_side,
                order_type=OrderType.MARKET,
                qty=state.remaining_qty,
                tag="quick_exit",
            ))
            full_close_requested = True

        if (not full_close_requested
                and cfg.ema_failsafe_enabled and state.tp1_hit
                and state.mfe_r >= cfg.ema_failsafe_min_expansion_r):
            ema = h1_ind.ema_fast if h1_ind else None
            if ema is not None:
                should_exit = False
                if direction == Side.LONG and bar.close < ema:
                    should_exit = True
                elif direction == Side.SHORT and bar.close > ema:
                    should_exit = True

                if should_exit and state.remaining_qty > 0:
                    reverse_side = Side.SHORT if direction == Side.LONG else Side.LONG
                    orders.append(Order(
                        order_id=self._gen_id(sym, "ema_failsafe"),
                        symbol=sym,
                        side=reverse_side,
                        order_type=OrderType.MARKET,
                        qty=state.remaining_qty,
                        tag="ema_failsafe",
                    ))

        return orders

    def get_be_price(self, sym: str) -> float | None:
        """Get breakeven stop price if BE move is triggered."""
        state = self._states.get(sym)
        if state is None or not state.be_moved:
            return None
        cfg = self._cfg
        if state.direction == Side.LONG:
            return state.entry_price + cfg.be_buffer_r * state.stop_distance
        return state.entry_price - cfg.be_buffer_r * state.stop_distance

    def _gen_id(self, sym: str, tag: str) -> str:
        self._next_order_id += 1
        return f"trend_exit_{sym}_{tag}_{self._next_order_id}"
