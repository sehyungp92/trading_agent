"""Breakout exit management — TP1/TP2, smart BE, time stop, invalidation, quick exit."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_trader.core.models import Bar, Order, OrderType, Side

from .balance import BalanceZone
from .config import BreakoutExitParams


@dataclass
class BreakoutExitState:
    """Per-position exit tracking state."""

    entry_price: float
    stop_distance: float
    original_qty: float
    remaining_qty: float
    direction: Side
    balance_upper: float  # For invalidation check
    balance_lower: float  # For invalidation check
    atr: float = 0.0      # For invalidation depth calculation
    mfe_r: float = 0.0
    mae_r: float = 0.0
    peak_r: float = 0.0
    current_r: float = 0.0
    bars_since_entry: int = 0
    early_lock_applied: bool = False
    tp1_hit: bool = False
    tp2_hit: bool = False
    be_moved: bool = False


class ExitManager:
    """Manage breakout exits: TP1/TP2, smart BE, time stop, invalidation, quick exit."""

    def __init__(self, cfg: BreakoutExitParams) -> None:
        self._p = cfg
        self._states: dict[str, BreakoutExitState] = {}

    def init_position(
        self,
        sym: str,
        entry_price: float,
        stop_distance: float,
        qty: float,
        direction: Side,
        balance_zone: BalanceZone,
        atr: float = 0.0,
    ) -> None:
        """Register a new position for exit management."""
        self._states[sym] = BreakoutExitState(
            entry_price=entry_price,
            stop_distance=stop_distance,
            original_qty=qty,
            remaining_qty=qty,
            direction=direction,
            balance_upper=balance_zone.upper,
            balance_lower=balance_zone.lower,
            atr=atr,
        )

    def get_state(self, sym: str) -> BreakoutExitState | None:
        """Return the exit state for *sym*, or ``None``."""
        return self._states.get(sym)

    def process_bar(self, bar: Bar, sym: str) -> list[Order]:
        """Evaluate all exit conditions for *sym* on the current bar.

        Returns a (possibly empty) list of exit orders.
        """
        state = self._states.get(sym)
        if state is None:
            return []

        orders: list[Order] = []
        state.bars_since_entry += 1

        # Update R-multiples using intra-bar peaks
        if state.stop_distance > 0:
            if state.direction == Side.LONG:
                peak_r = (bar.high - state.entry_price) / state.stop_distance
                trough_r = (bar.low - state.entry_price) / state.stop_distance
                state.current_r = (bar.close - state.entry_price) / state.stop_distance
            else:
                peak_r = (state.entry_price - bar.low) / state.stop_distance
                trough_r = (state.entry_price - bar.high) / state.stop_distance
                state.current_r = (state.entry_price - bar.close) / state.stop_distance

            state.peak_r = max(state.peak_r, peak_r)
            state.mfe_r = max(state.mfe_r, peak_r)
            state.mae_r = min(state.mae_r, trough_r)

        # 1. Invalidation exit: price re-entered balance zone
        if (self._p.invalidation_exit and not state.tp1_hit
                and state.bars_since_entry >= self._p.invalidation_min_bars):
            inv_order = self._check_invalidation(bar, state, sym)
            if inv_order is not None:
                orders.append(inv_order)
                return orders  # Full close

        # 2. Pre-TP1 profit lock
        if (
            self._p.early_lock_enabled
            and not state.early_lock_applied
            and not state.tp1_hit
            and state.mfe_r >= self._p.early_lock_mfe_r
        ):
            state.early_lock_applied = True

        # 3. TP1
        if not state.tp1_hit and state.peak_r >= self._p.tp1_r:
            tp1_qty = state.original_qty * self._p.tp1_frac
            tp1_qty = min(tp1_qty, state.remaining_qty)
            if tp1_qty > 0:
                orders.append(Order(
                    order_id="",
                    symbol=bar.symbol,
                    side=_opposite(state.direction),
                    order_type=OrderType.MARKET,
                    qty=tp1_qty,
                    tag="tp1",
                    metadata={"r_at_tp": state.peak_r},
                ))
                state.tp1_hit = True
                state.remaining_qty -= tp1_qty

        # 4. TP2 (guard: skip when tp2_r <= tp1_r to prevent same-bar cascade)
        if (state.tp1_hit and not state.tp2_hit
                and self._p.tp2_r > self._p.tp1_r
                and state.peak_r >= self._p.tp2_r):
            tp2_qty = state.original_qty * self._p.tp2_frac
            tp2_qty = min(tp2_qty, state.remaining_qty)
            if tp2_qty > 0:
                orders.append(Order(
                    order_id="",
                    symbol=bar.symbol,
                    side=_opposite(state.direction),
                    order_type=OrderType.MARKET,
                    qty=tp2_qty,
                    tag="tp2",
                    metadata={"r_at_tp": state.peak_r},
                ))
                state.tp2_hit = True
                state.remaining_qty -= tp2_qty

        # 5. Smart BE after TP1
        if state.tp1_hit and not state.be_moved and self._p.be_after_tp1:
            if state.bars_since_entry >= self._p.be_min_bars_above:
                state.be_moved = True
                # BE level = entry + be_buffer_r * stop_distance
                # Actual stop resubmit is handled by strategy via trail/stop logic

        # 6. Time stop
        if state.bars_since_entry >= self._p.time_stop_bars:
            if state.current_r < self._p.time_stop_min_progress_r:
                close_qty = state.remaining_qty
                if self._p.time_stop_action == "reduce":
                    close_qty = state.remaining_qty * 0.5
                if close_qty > 0:
                    orders.append(Order(
                        order_id="",
                        symbol=bar.symbol,
                        side=_opposite(state.direction),
                        order_type=OrderType.MARKET,
                        qty=close_qty,
                        tag="time_stop",
                        metadata={"bars": state.bars_since_entry},
                    ))
                    state.remaining_qty -= close_qty

        # 7. Quick exit
        if self._p.quick_exit_enabled:
            if (
                state.bars_since_entry >= self._p.quick_exit_bars
                and state.mfe_r < self._p.quick_exit_max_mfe_r
                and state.current_r <= self._p.quick_exit_max_r
            ):
                if state.remaining_qty > 0:
                    orders.append(Order(
                        order_id="",
                        symbol=bar.symbol,
                        side=_opposite(state.direction),
                        order_type=OrderType.MARKET,
                        qty=state.remaining_qty,
                        tag="quick_exit",
                        metadata={"mfe_r": state.mfe_r},
                    ))
                    state.remaining_qty = 0.0

        return orders

    def remove(self, sym: str) -> BreakoutExitState | None:
        """Remove and return the exit state for *sym*."""
        return self._states.pop(sym, None)

    # ------------------------------------------------------------------
    # private
    # ------------------------------------------------------------------

    def _check_invalidation(
        self, bar: Bar, state: BreakoutExitState, sym: str,
    ) -> Order | None:
        """Return a full-close order if price re-entered the balance zone by depth threshold."""
        depth_required = self._p.invalidation_depth_atr * state.atr if state.atr > 0 else 0.0

        if state.direction == Side.LONG:
            penetration = state.balance_upper - bar.close
            if penetration >= depth_required and bar.close < state.balance_upper:
                if state.remaining_qty > 0:
                    return Order(
                        order_id="",
                        symbol=bar.symbol,
                        side=Side.SHORT,
                        order_type=OrderType.MARKET,
                        qty=state.remaining_qty,
                        tag="invalidation",
                    )
        else:
            penetration = bar.close - state.balance_lower
            if penetration >= depth_required and bar.close > state.balance_lower:
                if state.remaining_qty > 0:
                    return Order(
                        order_id="",
                        symbol=bar.symbol,
                        side=Side.LONG,
                        order_type=OrderType.MARKET,
                        qty=state.remaining_qty,
                        tag="invalidation",
                    )
        return None


def _opposite(side: Side) -> Side:
    """Return the opposite trade direction."""
    return Side.SHORT if side == Side.LONG else Side.LONG
