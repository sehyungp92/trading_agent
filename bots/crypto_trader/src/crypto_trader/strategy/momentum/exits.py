"""Scaled exit management — TP1/TP2/BE/time stops/early exits."""

from __future__ import annotations

from dataclasses import dataclass, field

from crypto_trader.core.models import Order, OrderType, Position, Side
from crypto_trader.strategy.momentum.config import ExitParams
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot


def _confirmation_family(confirmation_type: str | None) -> str:
    pattern = (confirmation_type or "").lower()
    if pattern in {"inside_bar_break", "base_break"}:
        return "continuation"
    if pattern in {"micro_structure_shift", "hammer", "engulfing"}:
        return "inflection"
    return "other"


def _scope_matches(scope: str, family: str) -> bool:
    normalized = (scope or "all").lower()
    if normalized in {"all", "any"}:
        return True
    if normalized in {"none", "off"}:
        return False
    return normalized == family


@dataclass
class PositionExitState:
    """Per-position exit tracking state."""
    entry_price: float = 0.0
    stop_distance: float = 0.0
    original_qty: float = 0.0
    remaining_qty: float = 0.0
    tp1_hit: bool = False
    tp2_hit: bool = False
    be_moved: bool = False
    bars_since_entry: int = 0
    bars_above_1r: int = 0
    mae_r: float = 0.0
    mfe_r: float = 0.0
    peak_r: float = 0.0
    current_stop_order_id: str | None = None
    current_stop_price: float = 0.0
    current_stop_tag: str = ""
    proof_lock_moved: bool = False
    partial_exits: list[dict] = field(default_factory=list)


class ExitManager:
    def __init__(self, params: ExitParams) -> None:
        self._p = params
        self._states: dict[str, PositionExitState] = {}

    def init_position(
        self,
        symbol: str,
        entry_price: float,
        stop_distance: float,
        qty: float,
        stop_order_id: str | None = None,
        stop_price: float = 0.0,
        stop_tag: str = "protective_stop",
    ) -> None:
        self._states[symbol] = PositionExitState(
            entry_price=entry_price,
            stop_distance=stop_distance,
            original_qty=qty,
            remaining_qty=qty,
            current_stop_order_id=stop_order_id,
            current_stop_price=stop_price,
            current_stop_tag=stop_tag,
        )

    def get_state(self, symbol: str) -> PositionExitState | None:
        return self._states.get(symbol)

    def remove_position(self, symbol: str) -> PositionExitState | None:
        return self._states.pop(symbol, None)

    def manage(
        self,
        position: Position,
        current_bar: object,  # Bar
        m15_bars: list,
        indicators: IndicatorSnapshot | None,
        broker: object,  # BrokerAdapter
        confirmation_type: str | None = None,
    ) -> list[Order]:
        sym = position.symbol
        state = self._states.get(sym)
        if state is None or state.stop_distance <= 0:
            return []

        orders: list[Order] = []
        state.bars_since_entry += 1

        # Update MAE/MFE using intra-bar extremes
        bar = current_bar  # type: ignore
        price = bar.close  # type: ignore
        if position.direction == Side.LONG:
            current_r = (price - state.entry_price) / state.stop_distance
            peak_r = (bar.high - state.entry_price) / state.stop_distance  # type: ignore
            bar_mae = (bar.low - state.entry_price) / state.stop_distance  # type: ignore
        else:
            current_r = (state.entry_price - price) / state.stop_distance
            peak_r = (state.entry_price - bar.low) / state.stop_distance  # type: ignore
            bar_mae = (state.entry_price - bar.high) / state.stop_distance  # type: ignore

        state.mfe_r = max(state.mfe_r, peak_r)
        state.peak_r = peak_r
        state.mae_r = min(state.mae_r, bar_mae)
        family = _confirmation_family(confirmation_type)

        # TP1 at 1R: close 30% (using intra-bar peak for detection)
        if not state.tp1_hit and peak_r >= self._p.tp1_r:
            tp1_qty = state.original_qty * self._p.tp1_frac
            if tp1_qty > 0 and state.remaining_qty > tp1_qty:
                close_side = Side.SHORT if position.direction == Side.LONG else Side.LONG
                orders.append(Order(
                    order_id="",
                    symbol=sym,
                    side=close_side,
                    order_type=OrderType.MARKET,
                    qty=tp1_qty,
                    tag="tp1",
                ))
                state.tp1_hit = True
                state.remaining_qty -= tp1_qty
                state.partial_exits.append({"level": "tp1", "qty": tp1_qty, "price": price})

        # BE acceptance: 2 consecutive closes above 1R
        if state.tp1_hit and not state.be_moved:
            if current_r >= self._p.tp1_r:
                state.bars_above_1r += 1
            else:
                state.bars_above_1r = 0

            if state.bars_above_1r >= self._p.be_acceptance_bars:
                state.be_moved = True
                # Cancel old stop, submit new at entry + buffer
                if position.direction == Side.LONG:
                    be_price = state.entry_price + self._p.be_buffer_r * state.stop_distance
                else:
                    be_price = state.entry_price - self._p.be_buffer_r * state.stop_distance
                if self._stop_improves(position.direction, be_price, state.current_stop_price):
                    orders.extend(self._resubmit_stop(
                        sym, position.direction, be_price,
                        state.remaining_qty, state, broker,
                    ))

        # TP2 at 2R: close 30% (using intra-bar peak for detection)
        if state.tp1_hit and not state.tp2_hit and peak_r >= self._p.tp2_r:
            tp2_qty = state.original_qty * self._p.tp2_frac
            if tp2_qty > 0 and state.remaining_qty > tp2_qty:
                close_side = Side.SHORT if position.direction == Side.LONG else Side.LONG
                orders.append(Order(
                    order_id="",
                    symbol=sym,
                    side=close_side,
                    order_type=OrderType.MARKET,
                    qty=tp2_qty,
                    tag="tp2",
                ))
                state.tp2_hit = True
                state.remaining_qty -= tp2_qty
                state.partial_exits.append({"level": "tp2", "qty": tp2_qty, "price": price})

        # Proof-lock: once the trade proves itself, stop waiting for a full loss.
        if (
            state.remaining_qty > 0
            and self._p.proof_lock_enabled
            and not state.proof_lock_moved
            and state.bars_since_entry >= self._p.proof_lock_min_bars
            and state.mfe_r >= self._p.proof_lock_trigger_r
            and current_r >= self._p.proof_lock_stop_r
        ):
            proof_lock_price = self._r_to_stop_price(
                position.direction,
                state.entry_price,
                state.stop_distance,
                self._p.proof_lock_stop_r,
            )
            if self._stop_improves(position.direction, proof_lock_price, state.current_stop_price):
                orders.extend(self._resubmit_stop(
                    sym,
                    position.direction,
                    proof_lock_price,
                    state.remaining_qty,
                    state,
                    broker,
                    stop_tag="proof_lock_stop",
                ))
                state.proof_lock_moved = True

        # Time stops (only if we haven't already closed everything above)
        if state.remaining_qty > 0:
            if state.bars_since_entry >= self._p.hard_time_stop_bars:
                if current_r < self._p.hard_time_stop_min_r:
                    orders.append(self._close_remaining(sym, position.direction, state, "hard_time_stop"))
            elif state.bars_since_entry >= self._p.soft_time_stop_bars:
                if current_r < self._p.soft_time_stop_min_r:
                    orders.append(self._close_remaining(sym, position.direction, state, "soft_time_stop"))

        # Quick exit for stagnant trades (no momentum development)
        if (state.remaining_qty > 0
                and self._p.quick_exit_enabled
                and state.bars_since_entry >= self._p.quick_exit_bars
                and state.mfe_r < self._p.quick_exit_max_mfe_r
                and current_r <= self._p.quick_exit_max_r):
            orders.append(self._close_remaining(sym, position.direction, state, "quick_exit"))

        # Failure-to-follow-through: trades that prove a little, then roll over quickly.
        if (
            state.remaining_qty > 0
            and self._p.followthrough_exit_enabled
            and _scope_matches(self._p.followthrough_scope, family)
            and state.bars_since_entry <= self._p.followthrough_bars
            and state.mfe_r >= self._p.followthrough_peak_r
            and current_r <= self._p.followthrough_floor_r
        ):
            orders.append(self._close_remaining(
                sym,
                position.direction,
                state,
                "followthrough_exit",
            ))

        # Peak-MFE retrace exit: cap giveback once a runner has materially proved itself.
        if (
            state.remaining_qty > 0
            and self._p.mfe_retrace_exit_enabled
            and _scope_matches(self._p.mfe_retrace_scope, family)
            and state.bars_since_entry >= self._p.mfe_retrace_min_bars
            and state.mfe_r >= self._p.mfe_retrace_trigger_r
        ):
            retained_floor_r = max(
                self._p.mfe_retrace_min_r,
                state.mfe_r - self._p.mfe_retrace_giveback_r,
            )
            if current_r <= retained_floor_r:
                orders.append(self._close_remaining(
                    sym,
                    position.direction,
                    state,
                    "mfe_retrace_exit",
                ))

        # Early exit triggers (only if position still open)
        if state.remaining_qty > 0:
            early = self._check_early_exits(position, m15_bars, indicators, current_r)
            if early:
                orders.append(self._close_remaining(sym, position.direction, state, early))

        return [o for o in orders if o is not None]

    def _close_remaining(
        self, symbol: str, direction: Side, state: PositionExitState, reason: str
    ) -> Order | None:
        if state.remaining_qty <= 0:
            return None
        close_side = Side.SHORT if direction == Side.LONG else Side.LONG
        qty = state.remaining_qty
        state.remaining_qty = 0
        state.partial_exits.append({"level": reason, "qty": qty, "price": 0})
        return Order(
            order_id="",
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            qty=qty,
            tag=reason,
        )

    def _resubmit_stop(
        self,
        symbol: str,
        direction: Side,
        stop_price: float,
        qty: float,
        state: PositionExitState,
        broker: object,
        stop_tag: str = "breakeven_stop",
    ) -> list[Order]:
        orders: list[Order] = []
        # Cancel existing stop
        if state.current_stop_order_id:
            cancel = getattr(broker, "cancel_order", None)
            if cancel:
                cancel(state.current_stop_order_id)
            state.current_stop_order_id = None

        # New stop order
        close_side = Side.SHORT if direction == Side.LONG else Side.LONG
        orders.append(Order(
            order_id="",
            symbol=symbol,
            side=close_side,
            order_type=OrderType.STOP,
            qty=qty,
            stop_price=stop_price,
            tag=stop_tag,
        ))
        state.current_stop_price = stop_price
        state.current_stop_tag = stop_tag
        return orders

    @staticmethod
    def _r_to_stop_price(
        direction: Side,
        entry_price: float,
        stop_distance: float,
        r_multiple: float,
    ) -> float:
        if direction == Side.LONG:
            return entry_price + r_multiple * stop_distance
        return entry_price - r_multiple * stop_distance

    @staticmethod
    def _stop_improves(direction: Side, new_stop: float, current_stop: float) -> bool:
        if current_stop <= 0:
            return True
        if direction == Side.LONG:
            return new_stop > current_stop
        return new_stop < current_stop

    def _check_early_exits(
        self,
        position: Position,
        m15_bars: list,
        indicators: IndicatorSnapshot | None,
        current_r: float,
    ) -> str | None:
        if not m15_bars or indicators is None:
            return None

        # Structure break: significant move against position
        if self._p.enable_structure_break_exit and len(m15_bars) >= 3:
            last = m15_bars[-1]
            body = last.close - last.open  # type: ignore
            threshold = indicators.atr * self._p.structure_break_body_atr_mult
            if position.direction == Side.LONG and body < -threshold:
                return "structure_break"
            if position.direction == Side.SHORT and body > threshold:
                return "structure_break"

        # Reversal candle with volume
        if self._p.enable_reversal_candle_exit and len(m15_bars) >= 2:
            cur = m15_bars[-1]
            body = abs(cur.close - cur.open)  # type: ignore
            body_threshold = indicators.atr * self._p.reversal_body_atr_mult
            volume_threshold = indicators.volume_ma * self._p.reversal_volume_mult
            if body > body_threshold and cur.volume > volume_threshold:  # type: ignore
                if position.direction == Side.LONG and cur.close < cur.open:  # type: ignore
                    return "reversal_candle"
                if position.direction == Side.SHORT and cur.close > cur.open:  # type: ignore
                    return "reversal_candle"

        return None
