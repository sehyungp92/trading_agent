"""Post-hoc portfolio simulation engine.

Runs each strategy independently, then merges trade lists and simulates
portfolio rules (heat cap, directional cap, daily/weekly stops, drawdown
tiers, NQDTC continuation sizing, etc.) on the unified
timeline.  Position sizes are computed from equity * base_risk_pct, not
fixed_qty — this reflects realistic $10K account sizing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from backtests.shared.parity.legacy_result_outputs import (
    decision_stream_from_trades,
    trade_outcomes_from_records,
)
from backtests.momentum.config_portfolio import PortfolioBacktestConfig
from libs.oms.config.portfolio_config import PortfolioConfig

# ---------------------------------------------------------------------------
# ET timezone offset helpers (US Eastern approximation for futures)
# ---------------------------------------------------------------------------

_ET_OFFSET = timedelta(hours=-5)  # EST; DST not modeled (futures use ET)
_ET_TZ = timezone(_ET_OFFSET)


def _to_et(dt: datetime) -> datetime:
    """Convert a datetime to US Eastern (naive -> assume UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_ET_TZ)


def _trading_day(dt: datetime) -> tuple[int, int, int]:
    """Return (year, month, day) of the futures trading day.

    Futures trading day starts at 18:00 ET the prior calendar day.
    A bar at 2025-03-04 19:00 ET belongs to trading day 2025-03-05.
    """
    et = _to_et(dt)
    if et.hour >= 18:
        et += timedelta(days=1)
    return (et.year, et.month, et.day)


def _trading_week_key(dt: datetime) -> tuple[int, int]:
    """Return (year, iso_week) for weekly P&L resets.

    Week resets on Monday 18:00 ET (Sunday evening session start).
    """
    et = _to_et(dt)
    if et.hour >= 18:
        et += timedelta(days=1)
    iso = et.isocalendar()
    return (iso[0], iso[1])


# ---------------------------------------------------------------------------
# Normalized trade record
# ---------------------------------------------------------------------------

@dataclass
class PortfolioTrade:
    """Normalized trade from any strategy, with portfolio overlay fields."""

    strategy_id: str = ""
    direction: int = 0
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    initial_stop: float = 0.0

    # Raw results (from independent engine run, using engine's fixed_qty)
    raw_pnl_dollars: float = 0.0
    raw_qty: int = 0
    r_multiple: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    commission_per_contract: float = 0.62

    # NQDTC-specific
    is_continuation: bool = False

    # Portfolio overlay
    portfolio_approved: bool = True
    denial_reason: str = ""
    size_multiplier: float = 1.0  # aggregate multiplier from all rules
    portfolio_qty: int = 0        # computed from equity * risk_pct
    adjusted_pnl: float = 0.0    # pnl at portfolio-sized qty
    adjusted_commission: float = 0.0
    dd_tier_mult: float = 1.0
    portfolio_equity_at_entry: float = 0.0
    portfolio_heat_at_entry: float = 0.0

    # Context
    exit_reason: str = ""
    session: str = ""


# ---------------------------------------------------------------------------
# Portfolio result
# ---------------------------------------------------------------------------

@dataclass
class PortfolioResult:
    """Combined portfolio backtest output."""

    trades: list[PortfolioTrade] = field(default_factory=list)
    blocked_trades: list[PortfolioTrade] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    equity_timestamps: list[datetime] = field(default_factory=list)
    initial_equity: float = 0.0

    # Per-strategy trade counts
    strategy_trade_counts: dict[str, int] = field(default_factory=dict)
    strategy_blocked_counts: dict[str, int] = field(default_factory=dict)

    # Rule impact counters: rule_name -> count
    rule_blocks: dict[str, int] = field(default_factory=dict)
    # Rule impact PnL: rule_name -> sum of raw_pnl of blocked trades
    rule_blocked_pnl: dict[str, float] = field(default_factory=dict)

    # Concurrent position tracking
    max_concurrent: int = 0
    concurrent_distribution: dict[int, int] = field(default_factory=dict)
    decision_stream: list[dict] = field(default_factory=list)
    trade_outcomes: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

@dataclass
class _OpenPosition:
    """Position currently open in portfolio state."""
    trade_idx: int
    strategy_id: str
    direction: int
    risk_R: float  # risk in R-units (after size mult)
    entry_time: datetime | None = None


@dataclass
class _PortfolioState:
    """Mutable state walked across the event timeline."""
    equity: float = 0.0
    peak_equity: float = 0.0
    open_positions: list[_OpenPosition] = field(default_factory=list)

    # Daily P&L per strategy (R-multiples)
    daily_pnl: dict[str, float] = field(default_factory=dict)
    daily_pnl_total: float = 0.0
    current_day: tuple[int, int, int] = (0, 0, 0)

    # Weekly P&L (total R)
    weekly_pnl_R: float = 0.0
    current_week: tuple[int, int] = (0, 0)


    # Last NQDTC direction (for Vdubus direction filter)
    last_nqdtc_direction: int = 0
    last_nqdtc_exit_time: datetime | None = None


# ---------------------------------------------------------------------------
# Trade conversion from strategy-specific records
# ---------------------------------------------------------------------------

def _from_nqdtc(trade) -> PortfolioTrade:
    """Convert NQDTCTradeRecord -> PortfolioTrade."""
    return PortfolioTrade(
        strategy_id="NQDTC",
        direction=trade.direction,
        entry_time=trade.entry_time,
        exit_time=trade.exit_time,
        entry_price=trade.entry_price,
        exit_price=trade.exit_price,
        initial_stop=trade.initial_stop,
        raw_pnl_dollars=trade.pnl_dollars,
        raw_qty=trade.qty,
        r_multiple=trade.r_multiple,
        mfe_r=trade.mfe_r,
        mae_r=trade.mae_r,
        commission_per_contract=trade.commission / max(trade.qty * 2, 1),
        is_continuation=trade.continuation,
        exit_reason=trade.exit_reason,
        session=trade.session,
    )


def _from_vdubus(trade) -> PortfolioTrade:
    """Convert VdubusTradeRecord -> PortfolioTrade."""
    return PortfolioTrade(
        strategy_id="Vdubus",
        direction=trade.direction,
        entry_time=trade.entry_time,
        exit_time=trade.exit_time,
        entry_price=trade.entry_price,
        exit_price=trade.exit_price,
        initial_stop=trade.initial_stop,
        raw_pnl_dollars=trade.pnl_dollars,
        raw_qty=trade.qty,
        r_multiple=trade.r_multiple,
        mfe_r=trade.mfe_r,
        mae_r=trade.mae_r,
        commission_per_contract=trade.commission / max(trade.qty * 2, 1),
        exit_reason=trade.exit_reason,
        session=trade.session,
    )


# ---------------------------------------------------------------------------
# Portfolio backtester
# ---------------------------------------------------------------------------

class PortfolioBacktester:
    """Post-hoc portfolio simulation.

    Takes trade lists from independent engine runs, merges them, and applies
    portfolio rules to determine which trades would be approved and at what
    sizing.  Builds a combined equity curve and computes portfolio metrics.
    """

    def __init__(self, config: PortfolioBacktestConfig):
        self.config = config
        self.pc: PortfolioConfig = config.portfolio

    def run(
        self,
        nqdtc_trades: list | None = None,
        vdubus_trades: list | None = None,
    ) -> PortfolioResult:
        """Run the portfolio simulation.

        Args:
            nqdtc_trades: List of NQDTCTradeRecord from independent run.
            vdubus_trades: List of VdubusTradeRecord from independent run.
        """
        # 1. Convert all trades to normalized PortfolioTrade
        all_trades: list[PortfolioTrade] = []


        if nqdtc_trades and self.config.run_nqdtc:
            for t in nqdtc_trades:
                pt = _from_nqdtc(t)
                if self._in_date_range(pt):
                    all_trades.append(pt)

        if vdubus_trades and self.config.run_vdubus:
            for t in vdubus_trades:
                pt = _from_vdubus(t)
                if self._in_date_range(pt):
                    all_trades.append(pt)

        if not all_trades:
            return PortfolioResult(initial_equity=self.pc.initial_equity)

        # 2. Build event timeline: (time, event_type, trade_index)
        #    event_type: 0 = exit, 1 = entry  (exits first at same timestamp)
        events: list[tuple[datetime, int, int]] = []
        for i, t in enumerate(all_trades):
            if t.entry_time:
                events.append((t.entry_time, 1, i))
            if t.exit_time:
                events.append((t.exit_time, 0, i))

        # Sort by time, then exits before entries at the same timestamp
        events.sort(key=lambda e: (e[0], e[1]))

        # 3. Walk the timeline
        state = _PortfolioState(
            equity=self.pc.initial_equity,
            peak_equity=self.pc.initial_equity,
        )

        result = PortfolioResult(initial_equity=self.pc.initial_equity)
        equity_points: list[tuple[datetime, float]] = [
            (events[0][0], self.pc.initial_equity),
        ]

        for evt_time, evt_type, trade_idx in events:
            trade = all_trades[trade_idx]

            # Check date/week boundary resets
            self._check_boundaries(state, evt_time)

            if evt_type == 0:
                # EXIT event
                self._process_exit(state, trade, trade_idx, equity_points)
            else:
                # ENTRY event
                self._process_entry(state, trade, trade_idx, result)

        # 4. Separate approved vs blocked
        approved = [t for t in all_trades if t.portfolio_approved]
        blocked = [t for t in all_trades if not t.portfolio_approved]

        result.trades = approved
        result.blocked_trades = blocked
        result.decision_stream = decision_stream_from_trades(approved, timeframe="portfolio")
        result.trade_outcomes = trade_outcomes_from_records(approved)

        # 5. Build daily equity curve (one point per trading day)
        if equity_points:
            equity_points.sort(key=lambda x: x[0])
            result.equity_timestamps, result.equity_curve = (
                self._build_daily_equity(equity_points)
            )
        else:
            result.equity_curve = np.array([self.pc.initial_equity])
            result.equity_timestamps = []

        # 6. Per-strategy counts
        for t in approved:
            result.strategy_trade_counts[t.strategy_id] = (
                result.strategy_trade_counts.get(t.strategy_id, 0) + 1
            )
        for t in blocked:
            result.strategy_blocked_counts[t.strategy_id] = (
                result.strategy_blocked_counts.get(t.strategy_id, 0) + 1
            )

        # 7. Concurrent position tracking
        self._compute_concurrent_stats(approved, result)

        return result

    # ------------------------------------------------------------------
    # Daily equity curve builder
    # ------------------------------------------------------------------

    def _build_daily_equity(
        self,
        equity_points: list[tuple[datetime, float]],
    ) -> tuple[list[datetime], np.ndarray]:
        """Resample trade-exit equity points to one value per calendar day.

        Uses last-value-wins within each day to create a daily equity series.
        Forward-fills days with no exits. This gives a proper daily series
        for Sharpe/Sortino computation.
        """
        from datetime import date as dt_date

        if not equity_points:
            return [], np.array([self.pc.initial_equity])

        # Group by calendar date, take last value per day
        daily: dict[dt_date, float] = {}
        for ts, eq in equity_points:
            d = ts.date() if hasattr(ts, 'date') else ts
            daily[d] = eq  # last value wins

        # Build continuous daily series (forward-fill)
        first_date = min(daily.keys())
        last_date = max(daily.keys())
        current = first_date
        one_day = timedelta(days=1)

        timestamps: list[datetime] = []
        values: list[float] = []
        last_eq = self.pc.initial_equity

        while current <= last_date:
            if current in daily:
                last_eq = daily[current]
            timestamps.append(datetime(current.year, current.month, current.day))
            values.append(last_eq)
            current += one_day

        return timestamps, np.array(values)

    # ------------------------------------------------------------------
    # Date range filter
    # ------------------------------------------------------------------

    def _in_date_range(self, t: PortfolioTrade) -> bool:
        if self.config.start_date and t.entry_time:
            if t.entry_time < self.config.start_date.replace(tzinfo=t.entry_time.tzinfo):
                return False
        if self.config.end_date and t.entry_time:
            if t.entry_time > self.config.end_date.replace(tzinfo=t.entry_time.tzinfo):
                return False
        return True

    # ------------------------------------------------------------------
    # Boundary resets
    # ------------------------------------------------------------------

    def _check_boundaries(self, state: _PortfolioState, now: datetime):
        """Reset daily/weekly P&L on boundary crossings."""
        day = _trading_day(now)
        if day != state.current_day:
            # New trading day
            state.daily_pnl.clear()
            state.daily_pnl_total = 0.0
            state.current_day = day

        week = _trading_week_key(now)
        if week != state.current_week:
            # New trading week (Monday 18:00 ET boundary)
            state.weekly_pnl_R = 0.0
            state.current_week = week

    # ------------------------------------------------------------------
    # Entry processing with portfolio rules
    # ------------------------------------------------------------------

    def _process_entry(
        self,
        state: _PortfolioState,
        trade: PortfolioTrade,
        trade_idx: int,
        result: PortfolioResult,
    ):
        """Apply portfolio rules and decide whether to approve the trade."""
        pc = self.pc
        alloc = pc.get_strategy(trade.strategy_id)

        if alloc is None or not alloc.enabled:
            self._deny(trade, result, "strategy_disabled")
            return

        # --- Rule checks (in priority order) ---

        # 1. Max total positions
        if len(state.open_positions) >= pc.max_total_positions:
            self._deny(trade, result, "max_total_positions")
            return

        # 2. Per-strategy max concurrent
        strat_open = sum(
            1 for p in state.open_positions if p.strategy_id == trade.strategy_id
        )
        if strat_open >= alloc.max_concurrent:
            self._deny(trade, result, "max_concurrent")
            return

        # 3. Per-strategy daily stop
        strat_daily = state.daily_pnl.get(trade.strategy_id, 0.0)
        if strat_daily <= -alloc.daily_stop_R:
            self._deny(trade, result, "strategy_daily_stop")
            return

        # 4. Portfolio daily stop
        if state.daily_pnl_total <= -pc.portfolio_daily_stop_R:
            self._deny(trade, result, "portfolio_daily_stop")
            return

        # 5. Portfolio weekly stop
        if (
            self.config.portfolio_weekly_stop_R > 0
            and state.weekly_pnl_R <= -self.config.portfolio_weekly_stop_R
        ):
            self._deny(trade, result, "portfolio_weekly_stop")
            return

        # 6. Heat cap
        open_heat = sum(p.risk_R for p in state.open_positions)
        if open_heat + 1.0 > pc.heat_cap_R:
            self._deny(trade, result, "heat_cap")
            return

        # 7. Directional cap
        same_dir_risk = sum(
            p.risk_R for p in state.open_positions if p.direction == trade.direction
        )
        if same_dir_risk + 1.0 > pc.directional_cap_R:
            self._deny(trade, result, "directional_cap")
            return

        # --- Size multiplier accumulation ---
        size_mult = 1.0

        # 10. NQDTC continuation sizing
        if trade.strategy_id == "NQDTC" and trade.is_continuation:
            if alloc.reversal_only:
                self._deny(trade, result, "reversal_only")
                return
            if alloc.continuation_half_size:
                size_mult *= alloc.continuation_size_mult

        # 11. NQDTC direction filter for Vdubus
        if (
            trade.strategy_id == "Vdubus"
            and pc.nqdtc_direction_filter_enabled
            and state.last_nqdtc_direction != 0
        ):
            if trade.direction == state.last_nqdtc_direction:
                size_mult *= pc.nqdtc_agree_size_mult
            else:
                if pc.nqdtc_oppose_size_mult == 0.0:
                    self._deny(trade, result, "nqdtc_direction_oppose")
                    return
                size_mult *= pc.nqdtc_oppose_size_mult

        # 12. Drawdown tiers
        drawdown_pct = (
            (state.peak_equity - state.equity) / state.peak_equity
            if state.peak_equity > 0
            else 0.0
        )
        dd_mult = 1.0
        for threshold, mult in pc.dd_tiers:
            if drawdown_pct < threshold:
                dd_mult = mult
                break
        if dd_mult == 0.0:
            self._deny(trade, result, "drawdown_halt")
            return
        size_mult *= dd_mult

        # --- Compute portfolio-sized qty ---
        risk_per_trade = state.equity * alloc.base_risk_pct
        stop_distance = abs(trade.entry_price - trade.initial_stop)
        if stop_distance <= 0:
            self._deny(trade, result, "zero_stop_distance")
            return

        risk_per_contract = stop_distance * self.config.point_value
        raw_qty = risk_per_trade * size_mult / risk_per_contract
        portfolio_qty = max(1, int(round(raw_qty)))

        # Compute adjusted PnL: scale from raw engine qty to portfolio qty
        if trade.raw_qty > 0:
            pnl_per_contract = trade.raw_pnl_dollars / trade.raw_qty
        else:
            # Fallback: compute from price move
            move = (trade.exit_price - trade.entry_price) * trade.direction
            pnl_per_contract = move * self.config.point_value

        adjusted_pnl = pnl_per_contract * portfolio_qty
        adjusted_comm = trade.commission_per_contract * portfolio_qty * 2  # round-trip

        # Fill trade fields
        trade.portfolio_approved = True
        trade.size_multiplier = size_mult
        trade.portfolio_qty = portfolio_qty
        trade.adjusted_pnl = adjusted_pnl - adjusted_comm
        trade.adjusted_commission = adjusted_comm
        trade.dd_tier_mult = dd_mult
        trade.portfolio_equity_at_entry = state.equity
        trade.portfolio_heat_at_entry = open_heat

        # Add to open positions
        position_risk_R = size_mult  # 1R scaled by size_mult
        state.open_positions.append(_OpenPosition(
            trade_idx=trade_idx,
            strategy_id=trade.strategy_id,
            direction=trade.direction,
            risk_R=position_risk_R,
            entry_time=trade.entry_time,
        ))


    # ------------------------------------------------------------------
    # Exit processing
    # ------------------------------------------------------------------

    def _process_exit(
        self,
        state: _PortfolioState,
        trade: PortfolioTrade,
        trade_idx: int,
        equity_points: list[tuple[datetime, float]],
    ):
        """Process a trade exit: remove from open positions, update equity."""
        if not trade.portfolio_approved:
            return

        # Remove from open positions
        state.open_positions = [
            p for p in state.open_positions if p.trade_idx != trade_idx
        ]

        # Update equity
        state.equity += trade.adjusted_pnl
        if state.equity > state.peak_equity:
            state.peak_equity = state.equity

        # Update daily/weekly P&L (in R)
        r = trade.r_multiple * trade.size_multiplier
        state.daily_pnl[trade.strategy_id] = (
            state.daily_pnl.get(trade.strategy_id, 0.0) + r
        )
        state.daily_pnl_total += r
        state.weekly_pnl_R += r

        # Track NQDTC direction for Vdubus filter
        if trade.strategy_id == "NQDTC":
            state.last_nqdtc_direction = trade.direction
            state.last_nqdtc_exit_time = trade.exit_time

        # Record equity point
        if trade.exit_time:
            equity_points.append((trade.exit_time, state.equity))

    # ------------------------------------------------------------------
    # Denial helper
    # ------------------------------------------------------------------

    def _deny(
        self,
        trade: PortfolioTrade,
        result: PortfolioResult,
        reason: str,
    ):
        trade.portfolio_approved = False
        trade.denial_reason = reason
        trade.adjusted_pnl = 0.0
        trade.portfolio_qty = 0

        result.rule_blocks[reason] = result.rule_blocks.get(reason, 0) + 1
        result.rule_blocked_pnl[reason] = (
            result.rule_blocked_pnl.get(reason, 0.0) + trade.raw_pnl_dollars
        )

    # ------------------------------------------------------------------
    # Concurrent position analysis
    # ------------------------------------------------------------------

    def _compute_concurrent_stats(
        self,
        approved: list[PortfolioTrade],
        result: PortfolioResult,
    ):
        """Count distribution of simultaneous open positions."""
        if not approved:
            return

        # Build intervals
        intervals: list[tuple[datetime, int]] = []  # (time, +1 or -1)
        for t in approved:
            if t.entry_time and t.exit_time:
                intervals.append((t.entry_time, 1))
                intervals.append((t.exit_time, -1))

        intervals.sort(key=lambda x: (x[0], x[1]))

        concurrent = 0
        max_concurrent = 0
        counts: dict[int, int] = {}

        for _, delta in intervals:
            if delta == -1:
                # Record time spent at this concurrency level
                counts[concurrent] = counts.get(concurrent, 0) + 1
                concurrent -= 1
            else:
                concurrent += 1
                if concurrent > max_concurrent:
                    max_concurrent = concurrent
                counts[concurrent] = counts.get(concurrent, 0) + 1

        result.max_concurrent = max_concurrent
        result.concurrent_distribution = counts
