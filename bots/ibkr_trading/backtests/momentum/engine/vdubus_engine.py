"""VdubusNQ v4.0 backtest engine — 15-minute primary loop.

Replicates strategy_3/engine.py orchestration in synchronous, bar-by-bar mode.
Imports pure functions from strategy_3 (signals, regime, exits, risk, indicators).

Primary feed: 15m NQ bars. Higher TFs: 1H NQ, daily ES via idx maps.
Optional: 5m NQ bars for micro-trigger refinement.

Dual-instrument: NQ (trade instrument) + ES (regime via SMA200).
"""
from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from backtests.momentum.analysis.vdubus_shadow_tracker import VdubusShadowTracker
from backtests.momentum.config import SlippageConfig, round_to_tick
from backtests.momentum.config_vdubus import VdubusAblationFlags, VdubusBacktestConfig
from backtests.momentum.data.preprocessing import NumpyBars
from backtests.shared.parity.decision_capture import normalize_decision_stream
from backtests.shared.parity.replay_driver import ReplayStep, run_replay
from backtests.shared.parity.trade_outcomes import normalize_trade_outcome_stream
from backtests.momentum.engine.sim_broker import (
    FillResult,
    FillStatus,
    OrderSide,
    OrderType,
    SimBroker,
    SimOrder,
)

from strategies.momentum.vdub import config as C
from strategies.momentum.vdub import exits
from strategies.momentum.vdub import indicators as ind
from strategies.momentum.vdub import regime as reg
from strategies.momentum.vdub import risk
from strategies.momentum.vdub import signals as sig
from strategies.momentum.vdub.models import (
    DayCounters,
    Direction,
    EntryType,
    EventBlockState,
    PivotPoint,
    PositionStage,
    PositionState,
    RegimeState,
    SessionWindow,
    SubWindow,
    VolState,
    WorkingEntry,
)
from strategies.core.events import DecisionEvent
from strategies.momentum.vdub.core import logic as vdub_core_logic
from strategies.momentum.vdub.core.state import (
    VdubCoreState,
    VdubEntryFillContext,
    VdubEntrySubmitted,
    VdubFill,
    VdubFlattenRequest,
    VdubOrderUpdate,
    VdubPartialExitDone,
    VdubStopUpdateRequest,
)

logger = logging.getLogger(__name__)

_ET = None


def _get_et():
    global _ET
    if _ET is None:
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/New_York")
    return _ET


def _to_et(dt_utc: datetime) -> datetime:
    return dt_utc.astimezone(_get_et())


def _minutes(h: int, m: int) -> int:
    return h * 60 + m


# ---------------------------------------------------------------------------
# Session / time classification (reuse from strategies.momentum.vdub.engine)
# ---------------------------------------------------------------------------

def classify_session(dt_utc: datetime) -> tuple[SessionWindow, SubWindow]:
    et = _to_et(dt_utc)
    m = _minutes(et.hour, et.minute)
    if et.weekday() >= 5:
        return SessionWindow.BLOCKED, SubWindow.CORE

    # Hard blocks
    if _minutes(9, 30) <= m < _minutes(9, 40):
        return SessionWindow.BLOCKED, SubWindow.OPEN
    if _minutes(15, 50) <= m < _minutes(16, 0):
        return SessionWindow.BLOCKED, SubWindow.CORE
    if m >= _minutes(*C.EVENING_END) or m < _minutes(9, 40):
        return SessionWindow.BLOCKED, SubWindow.CORE

    # Midday dead-zone block (with late shoulder exception)
    if _minutes(*C.MIDDAY_DEAD_START) <= m < _minutes(*C.MIDDAY_DEAD_END):
        if C.USE_LATE_SHOULDER:
            ls = C.LATE_SHOULDER_RANGE
            if _minutes(*ls[0]) <= m < _minutes(*ls[1]):
                return SessionWindow.RTH, SubWindow.CORE
        return SessionWindow.BLOCKED, SubWindow.CORE

    # RTH sub-windows
    close_start = _minutes(*C.CLOSE_RANGE[0])
    if _minutes(9, 40) <= m < _minutes(10, 30):
        return SessionWindow.RTH, SubWindow.OPEN
    if _minutes(10, 30) <= m < close_start:
        return SessionWindow.RTH, SubWindow.CORE
    if close_start <= m < _minutes(15, 50):
        return SessionWindow.RTH, SubWindow.CLOSE

    # Evening
    if _minutes(*C.EVENING_START) <= m < _minutes(*C.EVENING_END):
        return SessionWindow.EVENING, SubWindow.EVENING

    return SessionWindow.BLOCKED, SubWindow.CORE


def _is_shoulder_period(dt_utc: datetime) -> bool:
    """Return True if time falls in midday shoulder periods (conditional entry)."""
    et = _to_et(dt_utc)
    m = _minutes(et.hour, et.minute)
    early = C.MIDDAY_SHOULDER_EARLY
    late = C.MIDDAY_SHOULDER_LATE
    return (_minutes(*early[0]) <= m < _minutes(*early[1]) or
            _minutes(*late[0]) <= m < _minutes(*late[1]))


def _is_late_shoulder(dt_utc: datetime) -> bool:
    """Return True if time falls in the late shoulder period."""
    if not C.USE_LATE_SHOULDER:
        return False
    et = _to_et(dt_utc)
    m = _minutes(et.hour, et.minute)
    ls = C.LATE_SHOULDER_RANGE
    return _minutes(*ls[0]) <= m < _minutes(*ls[1])


def _is_1550(dt_utc: datetime) -> bool:
    et = _to_et(dt_utc)
    return et.hour == 15 and et.minute == 50


def _is_friday(dt_utc: datetime) -> bool:
    return _to_et(dt_utc).weekday() == 4


def _is_overnight(dt_utc: datetime) -> bool:
    m = _minutes(_to_et(dt_utc).hour, _to_et(dt_utc).minute)
    return m >= _minutes(16, 0) or m < _minutes(9, 40)


def _is_rth_open(dt_utc: datetime) -> bool:
    et = _to_et(dt_utc)
    return et.hour == 9 and 30 <= et.minute < 45


def _is_rth_close(dt_utc: datetime) -> bool:
    et = _to_et(dt_utc)
    return et.hour == 16 and et.minute == 0


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class VdubusTradeRecord:
    """Completed VdubusNQ trade."""

    symbol: str = "NQ"
    direction: int = 0
    entry_type: str = ""  # TYPE_A / TYPE_B
    is_flip: bool = False
    is_addon: bool = False
    # Session context
    session: str = ""
    sub_window: str = ""
    # Timing
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    bars_held_15m: int = 0
    overnight_sessions: int = 1
    # Prices
    entry_price: float = 0.0
    exit_price: float = 0.0
    initial_stop: float = 0.0
    signal_entry_price: float = 0.0  # computed stop_entry before fill slippage
    qty: int = 0
    # PnL
    pnl_dollars: float = 0.0
    r_multiple: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    # Exit
    exit_reason: str = ""
    # Context at entry
    daily_trend: int = 0
    vol_state: str = ""
    trend_1h: int = 0
    class_mult: float = 1.0
    vwap_used_at_entry: float = 0.0
    # Lifecycle
    partial_done: bool = False
    decision_gate_action: str = ""
    stage_at_exit: str = ""
    commission: float = 0.0


# ---------------------------------------------------------------------------
# Signal event for gating attribution
# ---------------------------------------------------------------------------

@dataclass
class VdubusSignalEvent:
    """Logged for every 15m entry evaluation (pass or reject)."""

    timestamp: datetime | None = None
    direction: int = 0
    session: str = ""
    sub_window: str = ""
    # Per-gate pass/fail
    regime_pass: bool = True
    alignment_pass: bool = True
    direction_cap_pass: bool = True
    slope_pass: bool = True
    signal_pass: bool = True
    predator_pass: bool = True
    viability_pass: bool = True
    risk_gate_pass: bool = True
    event_block_pass: bool = True
    shock_block_pass: bool = True
    # Outcome
    passed_all: bool = False
    first_block_reason: str = ""
    entry_type: str = ""
    would_be_entry: float = 0.0
    would_be_stop: float = 0.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class VdubusResult:
    """VdubusNQ backtest output."""

    trades: list[VdubusTradeRecord] = field(default_factory=list)
    signal_events: list[VdubusSignalEvent] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    time_series: np.ndarray = field(default_factory=lambda: np.array([]))
    # Funnel stats
    evaluations: int = 0
    regime_passed: int = 0
    signals_found: int = 0
    entries_placed: int = 0
    entries_filled: int = 0
    total_commission: float = 0.0
    shadow_summary: str = ""
    shadow_tracker: VdubusShadowTracker | None = None
    decision_stream: list[dict] = field(default_factory=list)
    trade_outcomes: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal active position wrapper
# ---------------------------------------------------------------------------

@dataclass
class _ActivePosition:
    """Internal mutable position tracking for backtest."""

    pos: PositionState
    record: VdubusTradeRecord
    mfe_price: float = 0.0
    mae_price: float = 0.0
    stop_order_id: str = ""
    commission_at_start: float = 0.0


# ---------------------------------------------------------------------------
# Ablation patch context manager
# ---------------------------------------------------------------------------

@contextmanager
def _ablation_patch(overrides: dict[str, float]):
    """Temporarily override strategy_3.config module-level constants."""
    originals = {}
    for key, value in overrides.items():
        upper = key.upper()
        if hasattr(C, upper):
            originals[upper] = getattr(C, upper)
            setattr(C, upper, value)
    try:
        yield
    finally:
        for key, value in originals.items():
            setattr(C, key, value)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class VdubusEngine:
    """Single-symbol bar-by-bar VdubusNQ v4.0 backtest on 15-minute bars."""

    def __init__(
        self,
        symbol: str,
        bt_config: VdubusBacktestConfig,
    ) -> None:
        self.symbol = symbol
        self.cfg = bt_config
        self.flags = bt_config.flags
        self.tick = bt_config.tick_size
        self.pv = bt_config.point_value

        # Sync global NQ_SPEC with backtest point_value so shared modules
        # (e.g. risk.compute_qty) use the same value as self.pv.
        C.NQ_SPEC["point_value"] = bt_config.point_value
        C.NQ_SPEC["tick_value"] = bt_config.tick_size * bt_config.point_value

        # Simulated broker
        self.broker = SimBroker(slippage_config=bt_config.slippage)

        # Account
        self.equity = bt_config.initial_equity

        # Shared state
        self.regime = RegimeState()
        self.counters = DayCounters()
        self.event_state = EventBlockState()

        # Drawdown throttle
        from libs.risk.drawdown_throttle import DrawdownThrottle
        self._throttle = DrawdownThrottle(bt_config.initial_equity)

        # Shadow tracker
        self.shadow_tracker: VdubusShadowTracker | None = (
            VdubusShadowTracker() if bt_config.track_shadows else None
        )

        # Active position (one at a time for NQ)
        self._active: _ActivePosition | None = None
        self._working: dict[str, WorkingEntry] = {}

        # Indicator caches
        self._atr15: np.ndarray = np.array([])
        self._atr1h: np.ndarray = np.array([])
        self._mom15: np.ndarray = np.array([])
        self._pivots_1h: list[PivotPoint] = []
        self._pivots_daily: list[PivotPoint] = []
        self._svwap: np.ndarray = np.array([])
        self._vwap_a_arr: np.ndarray = np.array([])
        self._vwap_a_val: float = np.nan

        # Incremental indicator state (initialized in run())
        self._inc_atr15: ind.IncrementalATR | None = None
        self._inc_macd15: ind.IncrementalMACD | None = None
        self._atr1h_precomputed: np.ndarray = np.array([])
        self._all_pivots_1h: list[PivotPoint] = []
        self._h_to_15m_map: np.ndarray = np.array([], dtype=np.intp)
        # Incremental VWAP state
        self._svwap_full: np.ndarray = np.array([])
        self._svwap_cum_tpv: float = 0.0
        self._svwap_cum_vol: float = 0.0
        self._svwap_sess_idx: int = -1
        self._vwap_a_full: np.ndarray = np.array([])
        self._vwap_a_cum_tpv: float = 0.0
        self._vwap_a_cum_vol: float = 0.0
        self._vwap_a_anchor_idx: int = -1
        self._vwap_a_pivot_key: tuple | None = None

        # TF boundary tracking
        self._last_1h_idx = -1
        self._last_d_idx = -1
        self._last_reset_date = ""
        self._bar_idx = 0
        self._total_commission = 0.0

        # Session start tracking (for VWAP no look-ahead)
        self._session_starts: dict[str, int] = {}

        # Result accumulators
        self._trades: list[VdubusTradeRecord] = []
        self._signal_events: list[VdubusSignalEvent] = []
        self._equity_history: list[float] = []
        self._time_history: list = []

        # Rolling win-rate tracking for adaptive sizing
        self._recent_wins: list[bool] = []

        # Funnel counters
        self._evaluations = 0
        self._regime_passed = 0
        self._signals_found = 0
        self._entries_placed = 0
        self._entries_filled = 0

        # Event calendar (optional)
        self._event_calendar: list[dict] | None = None
        if bt_config.news_calendar_path and bt_config.news_calendar_path.exists():
            self._event_calendar = self._load_event_calendar(bt_config.news_calendar_path)

        # Core logic state (thin-driver parity layer)
        self._core_state = VdubCoreState()
        self._decision_events: list[DecisionEvent] = []

    # ------------------------------------------------------------------
    # Core replay delegation
    # ------------------------------------------------------------------

    def _replay_core_step(self, *, bar_input=None, order_updates=None, fills=None):
        result = run_replay(
            self._core_state,
            steps=[ReplayStep(
                bar_input=bar_input,
                order_updates=order_updates or [],
                fills=fills or [],
            )],
            on_bar=lambda state, payload: vdub_core_logic.on_bar(state, **payload),
            on_order_update=vdub_core_logic.on_order_update,
            on_fill=vdub_core_logic.on_fill,
        )
        self._core_state = result.state
        self._decision_events.extend(result.events)
        return result

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        bars_15m: NumpyBars,
        bars_5m: NumpyBars | None,
        hourly: NumpyBars,
        daily_es: NumpyBars,
        hourly_idx_map: np.ndarray,
        daily_es_idx_map: np.ndarray,
        five_to_15_idx_map: np.ndarray | None,
    ) -> VdubusResult:
        """Run the full backtest. Returns VdubusResult."""
        n = len(bars_15m)
        wu_d = self.cfg.warmup_daily_es
        wu_1h = self.cfg.warmup_1h
        wu_15m = self.cfg.warmup_15m

        # Pre-compute session start indices for VWAP (no look-ahead)
        self._build_session_starts(bars_15m.times)

        # Precompute epoch arrays for accurate 1H→15m pivot anchor mapping
        self._hourly_epochs = np.array(
            [np.datetime64(t, "ns").astype("int64") for t in hourly.times]
        )
        self._15m_epochs = np.array(
            [np.datetime64(t, "ns").astype("int64") for t in bars_15m.times]
        )

        # Initialize incremental indicators (O(1) per bar instead of O(n))
        self._inc_atr15 = ind.IncrementalATR(n, C.VOL_ATR_PERIOD)
        self._inc_macd15 = ind.IncrementalMACD(n)
        self._svwap_full = np.full(n, np.nan)
        self._vwap_a_full = np.full(n, np.nan)
        self._svwap_sess_idx = -1
        self._vwap_a_pivot_key = None
        self._vwap_a_anchor_idx = -1

        # Pre-compute 1H ATR once (pure function, no look-ahead)
        n_1h = len(hourly)
        if n_1h > 0:
            self._atr1h_precomputed = ind.atr(
                hourly.highs, hourly.lows, hourly.closes)
        else:
            self._atr1h_precomputed = np.array([])

        # Pre-compute all 1H pivots (filter by confirmed_at at query time)
        if n_1h > 0:
            self._all_pivots_1h = ind.confirmed_pivots(
                hourly.highs, hourly.lows, C.NCONFIRM_1H)
        else:
            self._all_pivots_1h = []

        # Pre-compute hourly→15m nearest-index mapping
        if n_1h > 0 and n > 0:
            h_to_15m = np.searchsorted(self._15m_epochs, self._hourly_epochs)
            # Clamp and adjust to nearest match
            h_to_15m = np.clip(h_to_15m, 0, n - 1)
            prev_ok = h_to_15m > 0
            closer = prev_ok & (
                np.abs(self._15m_epochs[np.clip(h_to_15m - 1, 0, n - 1)]
                       - self._hourly_epochs)
                < np.abs(self._15m_epochs[h_to_15m] - self._hourly_epochs))
            h_to_15m[closer] -= 1
            self._h_to_15m_map = h_to_15m
        else:
            self._h_to_15m_map = np.array([], dtype=np.intp)

        # Merge ablation flag overrides into config patches
        _overrides = dict(self.cfg.param_overrides)
        if not self.flags.vwap_cap_gate:
            _overrides.setdefault("vwap_cap_core", 1e6)
            _overrides.setdefault("vwap_cap_open_eve", 1e6)
        if not self.flags.extension_sanity:
            _overrides.setdefault("extension_skip_atr", 1e6)
        if not self.flags.touch_lookback_gate:
            _overrides.setdefault("touch_lookback_15m", 100_000)
        if not self.flags.momentum_floor:
            _overrides.setdefault("floor_pct", 0.0)
        if not self.flags.min_max_stop:
            _overrides.setdefault("min_stop_points", 0)
            _overrides.setdefault("max_stop_points", 100_000)

        with _ablation_patch(_overrides):
            for t in range(n):
                bar_time = self._bar_time(bars_15m.times[t])
                O = bars_15m.opens[t]
                H = bars_15m.highs[t]
                L = bars_15m.lows[t]
                Cl = bars_15m.closes[t]
                V = bars_15m.volumes[t]

                if np.isnan(O) or np.isnan(Cl):
                    continue

                h_idx = int(hourly_idx_map[t])
                d_idx = int(daily_es_idx_map[t])

                self._step_15m(
                    t, bar_time, O, H, L, Cl, V,
                    bars_15m, hourly, daily_es,
                    h_idx, d_idx,
                    wu_d, wu_1h, wu_15m,
                    bars_5m, five_to_15_idx_map,
                )

        # Close any remaining position at last bar
        if self._active and self._active.pos.qty_open > 0:
            last_close = float(bars_15m.closes[-1])
            last_time = self._bar_time(bars_15m.times[-1])
            self._close_position(last_close, last_time, "END_OF_DATA", market_exit=True)

        # Shadow simulation
        shadow_summary = ""
        if self.shadow_tracker is not None and self.shadow_tracker.rejections:
            self.shadow_tracker.simulate_shadows(
                bars_15m.opens, bars_15m.highs,
                bars_15m.lows, bars_15m.closes,
                bars_15m.times,
            )
            shadow_summary = self.shadow_tracker.format_summary()

        return VdubusResult(
            trades=self._trades,
            signal_events=self._signal_events,
            equity_curve=np.array(self._equity_history),
            time_series=np.array(self._time_history),
            evaluations=self._evaluations,
            regime_passed=self._regime_passed,
            signals_found=self._signals_found,
            entries_placed=self._entries_placed,
            entries_filled=self._entries_filled,
            total_commission=self._total_commission,
            shadow_summary=shadow_summary,
            shadow_tracker=self.shadow_tracker,
            decision_stream=normalize_decision_stream(self._decision_events),
            trade_outcomes=normalize_trade_outcome_stream(self._trades),
        )

    # ------------------------------------------------------------------
    # Per-15m-bar step
    # ------------------------------------------------------------------

    def _step_15m(
        self,
        t: int,
        bar_time: datetime,
        O: float, H: float, L: float, Cl: float, V: float,
        bars_15m: NumpyBars,
        hourly: NumpyBars,
        daily_es: NumpyBars,
        h_idx: int, d_idx: int,
        wu_d: int, wu_1h: int, wu_15m: int,
        bars_5m: NumpyBars | None,
        five_to_15_idx_map: np.ndarray | None,
    ) -> None:
        self._bar_idx = t

        # 1. Classify session
        session, sub_window = classify_session(bar_time)

        # 2. Daily reset at 09:30 ET
        self._check_daily_reset(bar_time)

        # 3. Process broker fills
        fills = self.broker.process_bar(self.symbol, bar_time, O, H, L, Cl, self.tick)
        for fill in fills:
            self._handle_fill(fill, bar_time, Cl, session, sub_window)

        # 4. MFE/MAE update
        if self._active and self._active.pos.qty_open > 0:
            self._update_mfe_mae(H, L)

        # 5. Update indicators (check TF boundaries)
        new_1h = h_idx != self._last_1h_idx and h_idx >= wu_1h
        new_d = d_idx != self._last_d_idx and d_idx >= wu_d

        if new_d:
            self._on_daily_es_boundary(daily_es, d_idx)
            self._last_d_idx = d_idx

        if new_1h:
            self._on_1h_boundary(hourly, h_idx)
            self._last_1h_idx = h_idx

        # 15m indicators — O(1) incremental update (must run every bar for state)
        self._inc_atr15.update(t, H, L, Cl)
        self._inc_macd15.update(t, Cl)
        if t >= wu_15m:
            self._atr15 = self._inc_atr15.values[:t + 1]
            self._mom15 = self._inc_macd15.values[:t + 1]

        # 6. Compute VWAP (session + VWAP-A) — O(1) incremental
        if t >= wu_15m:
            self._update_vwaps(bars_15m, t, bar_time, H, L, Cl, V)

        # 7. Manage working entries (TTL, teleport, fallback)
        self._manage_working_entries(bar_time, Cl)

        # 8. Manage open position
        if self._active and self._active.pos.qty_open > 0:
            self._manage_position(bar_time, bars_15m, hourly, h_idx, t)

        # 9. Decision gate at 15:50 ET
        if _is_1550(bar_time) and self._active and self._active.pos.qty_open > 0:
            self._run_decision_gate(bar_time, bars_15m, t)

        # 10. Overnight trail
        if self._active and self._active.pos.qty_open > 0:
            if self._active.pos.stage == PositionStage.SWING_HOLD and _is_overnight(bar_time):
                self._apply_overnight_trail(hourly, h_idx, bar_time)

        # 11. VWAP-A failure check
        if self._active and self._active.pos.qty_open > 0:
            self._check_vwap_a_failure(bar_time, hourly, h_idx, bars_15m, t)

        # 12. Entry evaluation (skip if in warmup, blocked, or shock)
        if t >= wu_15m and session != SessionWindow.BLOCKED:
            # Event gate
            if self.flags.event_blocking:
                self._update_event_state(bar_time)
                if not self.event_state.rearmed:
                    self._equity_snapshot(bar_time, Cl)
                    return

            # Shock block
            if self.flags.shock_block and self.regime.vol_state == VolState.SHOCK:
                self._shock_tighten_all()
                self._equity_snapshot(bar_time, Cl)
                return

            for direction in (Direction.LONG, Direction.SHORT):
                self._evaluate_direction(
                    direction, session, sub_window, bar_time,
                    bars_15m, hourly, h_idx, t,
                    bars_5m, five_to_15_idx_map,
                )

        # 13. Equity snapshot
        self._equity_snapshot(bar_time, Cl)

    # ------------------------------------------------------------------
    # Higher-TF boundary handlers
    # ------------------------------------------------------------------

    def _on_daily_es_boundary(self, daily_es: NumpyBars, d_idx: int) -> None:
        """Update regime from ES daily bars."""
        closes = daily_es.closes[:d_idx + 1]
        highs = daily_es.highs[:d_idx + 1]
        lows = daily_es.lows[:d_idx + 1]

        if len(closes) > C.DAILY_SMA_PERIOD:
            reg.compute_daily_trend(closes, self.regime)
        if len(closes) > C.VOL_ATR_PERIOD:
            self.regime.vol_state = reg.compute_vol_state(highs, lows, closes)

    def _on_1h_boundary(self, hourly: NumpyBars, h_idx: int) -> None:
        """Update 1H indicators and trend."""
        closes = hourly.closes[:h_idx + 1]
        highs = hourly.highs[:h_idx + 1]
        lows = hourly.lows[:h_idx + 1]

        if len(closes) > C.HOURLY_EMA_PERIOD:
            reg.compute_1h_trend(closes, self.regime)
        if len(closes) > C.CHOP_PERIOD + 1:
            self.regime.choppiness = reg.compute_choppiness(
                highs, lows, closes, C.CHOP_PERIOD)
        # Use pre-computed 1H ATR (O(1) view instead of O(n) recompute)
        self._atr1h = self._atr1h_precomputed[:h_idx + 1]
        # Use pre-computed pivots filtered by confirmation bar (no look-ahead)
        self._pivots_1h = [
            p for p in self._all_pivots_1h if p.confirmed_at <= h_idx]

    # ------------------------------------------------------------------
    # VWAP computation (no look-ahead)
    # ------------------------------------------------------------------

    def _build_session_starts(self, times: np.ndarray) -> None:
        """Pre-scan times to build session_starts: {date_str: bar_idx}."""
        self._session_starts = {}
        for i in range(len(times)):
            bt = self._bar_time(times[i])
            et = _to_et(bt)
            m = _minutes(et.hour, et.minute)
            date_str = et.strftime("%Y-%m-%d")
            # Session VWAP starts at 09:30 (first RTH bar)
            if _minutes(9, 30) <= m < _minutes(9, 45) and date_str not in self._session_starts:
                self._session_starts[date_str] = i

    def _update_vwaps(self, bars_15m: NumpyBars, t: int,
                      bar_time: datetime,
                      H: float, L: float, Cl: float, V: float) -> None:
        """Incremental session VWAP and VWAP-A. O(1) per bar."""
        et = _to_et(bar_time)
        date_str = et.strftime("%Y-%m-%d")
        sess_start = self._session_starts.get(date_str, 0)

        # Session VWAP — reset on new session, O(1) accumulate
        if sess_start != self._svwap_sess_idx:
            # Clear stale values from the previous session
            if self._svwap_sess_idx >= 0:
                clear_end = min(sess_start, t + 1)
                if clear_end > self._svwap_sess_idx:
                    self._svwap_full[self._svwap_sess_idx:clear_end] = np.nan
            self._svwap_sess_idx = sess_start
            self._svwap_cum_tpv = 0.0
            self._svwap_cum_vol = 0.0

        if t >= sess_start:
            tp = (H + L + Cl) / 3.0
            v = max(float(V), 1.0)
            self._svwap_cum_tpv += tp * v
            self._svwap_cum_vol += v
            self._svwap_full[t] = self._svwap_cum_tpv / self._svwap_cum_vol

        self._svwap = self._svwap_full[:t + 1]

        # VWAP-A — O(1) per bar, O(n) on anchor change
        self._update_vwap_a(bars_15m, t, H, L, Cl, V)

    def _update_vwap_a(self, bars_15m: NumpyBars, t: int,
                        H: float, L: float, Cl: float, V: float) -> None:
        """Incremental VWAP-A. O(1) per bar, O(t-anchor) on anchor change."""
        self._vwap_a_val = np.nan

        if not self._pivots_1h:
            self._vwap_a_arr = self._vwap_a_full[:t + 1]
            return

        # Find anchor: most recent pivot matching dominant trend direction
        target = "low" if self.regime.daily_trend >= 0 else "high"
        candidates = [p for p in self._pivots_1h if p.ptype == target]
        if not candidates:
            self._vwap_a_arr = self._vwap_a_full[:t + 1]
            return

        pivot = candidates[-1]
        pivot_key = (pivot.idx, target)

        if pivot_key != self._vwap_a_pivot_key:
            # Anchor changed — recompute from new anchor to current bar
            self._vwap_a_pivot_key = pivot_key
            anchor = min(int(self._h_to_15m_map[pivot.idx]), t)
            self._vwap_a_anchor_idx = anchor
            self._vwap_a_full[:] = np.nan
            self._vwap_a_cum_tpv = 0.0
            self._vwap_a_cum_vol = 0.0
            for i in range(anchor, t + 1):
                tp_i = (float(bars_15m.highs[i]) + float(bars_15m.lows[i])
                        + float(bars_15m.closes[i])) / 3.0
                v_i = max(float(bars_15m.volumes[i]), 1.0)
                self._vwap_a_cum_tpv += tp_i * v_i
                self._vwap_a_cum_vol += v_i
                self._vwap_a_full[i] = self._vwap_a_cum_tpv / self._vwap_a_cum_vol
        else:
            # Anchor unchanged — O(1) increment
            anchor = self._vwap_a_anchor_idx
            if anchor >= 0 and t >= anchor:
                tp = (H + L + Cl) / 3.0
                v = max(float(V), 1.0)
                self._vwap_a_cum_tpv += tp * v
                self._vwap_a_cum_vol += v
                self._vwap_a_full[t] = self._vwap_a_cum_tpv / self._vwap_a_cum_vol

        self._vwap_a_arr = self._vwap_a_full[:t + 1]
        if not np.isnan(self._vwap_a_full[t]):
            self._vwap_a_val = float(self._vwap_a_full[t])

    # ------------------------------------------------------------------
    # Working entry management (Section 15)
    # ------------------------------------------------------------------

    def _manage_working_entries(self, bar_time: datetime, close: float) -> None:
        """TTL cancel, teleport skip, fallback market."""
        to_remove: list[str] = []

        for oid, we in list(self._working.items()):
            bars_since = self._bar_idx - we.submitted_bar_idx

            # TTL cancel
            if self.flags.ttl_cancel and bars_since >= we.ttl_bars:
                self.broker.cancel_orders(self.symbol, tag=None)
                to_remove.append(oid)
                continue

            tick = self.tick

            # Teleport skip
            if self.flags.teleport_skip:
                if we.direction == Direction.LONG and close > we.limit_entry + C.TELEPORT_TICKS * tick:
                    self.broker.cancel_orders(self.symbol, tag=None)
                    to_remove.append(oid)
                    continue
                if we.direction == Direction.SHORT and close < we.limit_entry - C.TELEPORT_TICKS * tick:
                    self.broker.cancel_orders(self.symbol, tag=None)
                    to_remove.append(oid)
                    continue

            # Detect trigger
            if not we.triggered:
                if we.direction == Direction.LONG and close >= we.stop_entry:
                    we.triggered = True
                    we.triggered_bar_idx = self._bar_idx
                elif we.direction == Direction.SHORT and close <= we.stop_entry:
                    we.triggered = True
                    we.triggered_bar_idx = self._bar_idx

            # Fallback market
            if self.flags.fallback_market and we.triggered and we.fallback_allowed:
                if self._bar_idx - we.triggered_bar_idx >= C.FALLBACK_WAIT_BARS:
                    atr15_val = self._safe_atr15()
                    atr_ticks = atr15_val / tick if atr15_val > 0 else 999
                    if atr_ticks > C.FALLBACK_ATR_TICKS_CAP:
                        continue
                    slip_cost = C.FALLBACK_SLIP_MAX_TICKS * C.NQ_SPEC["tick_value"] * we.qty
                    r_usd = abs(we.stop_entry - we.initial_stop) * self.pv * we.qty
                    if r_usd > 0 and slip_cost / r_usd > C.COST_RISK_MAX:
                        continue
                    # Cancel the stop-limit and submit market
                    self.broker.cancel_orders(self.symbol, tag=None)
                    to_remove.append(oid)
                    self._submit_fallback_market(we, bar_time)
                    we.fallback_allowed = False

        for oid in to_remove:
            self._working.pop(oid, None)

    def _submit_fallback_market(self, we: WorkingEntry, bar_time: datetime) -> None:
        """Submit fallback market order from a working entry."""
        side = OrderSide.BUY if we.direction == Direction.LONG else OrderSide.SELL
        order_id = self.broker.next_order_id()

        order = SimOrder(
            order_id=order_id, symbol=self.symbol, side=side,
            order_type=OrderType.MARKET, qty=we.qty,
            tick_size=self.tick, submit_time=bar_time,
            tag=f"fallback_{we.entry_type.value}",
        )
        self.broker.submit_order(order)

        fb = WorkingEntry(
            oms_order_id=order_id,
            entry_type=we.entry_type, direction=we.direction,
            stop_entry=we.stop_entry, limit_entry=we.stop_entry,
            qty=we.qty, submitted_bar_idx=self._bar_idx,
            initial_stop=we.initial_stop, vwap_used=we.vwap_used,
            class_mult=we.class_mult, fallback_allowed=False,
            session=we.session,
            is_flip=we.is_flip, is_addon=we.is_addon,
        )
        self._working[order_id] = fb

    # ------------------------------------------------------------------
    # Position management (Section 16)
    # ------------------------------------------------------------------

    def _manage_position(
        self, bar_time: datetime,
        bars_15m: NumpyBars, hourly: NumpyBars,
        h_idx: int, t: int,
    ) -> None:
        """Per-bar position management: partials, trailing, VWAP fail, stale."""
        if self._active is None:
            return
        pos = self._active.pos
        if pos.qty_open <= 0:
            return

        pos.bars_since_entry += 1

        # Update highest/lowest since entry (mirrors live engine)
        pos.highest_since_entry = max(
            pos.highest_since_entry, float(bars_15m.highs[t]))
        pos.lowest_since_entry = min(
            pos.lowest_since_entry, float(bars_15m.lows[t]))

        price = float(bars_15m.closes[t])
        atr15 = self._safe_atr15()

        # Update peak MFE R for early kill tracking
        unreal_r = self._unrealized_r(pos, price)
        pos.peak_mfe_r = max(pos.peak_mfe_r, unreal_r)

        # v4.2: MFE ratchet floor — lock a minimum stop as trade advances
        if self.flags.mfe_ratchet:
            ratchet_floor = exits.compute_mfe_ratchet_floor(pos)
            if ratchet_floor > 0.0:
                if pos.direction == Direction.LONG:
                    new_floor = max(pos.stop_price, ratchet_floor)
                else:
                    new_floor = min(pos.stop_price, ratchet_floor)
                if new_floor != pos.stop_price:
                    pos.stop_price = new_floor
                    self._update_stop_price(new_floor, bar_time)

        # v4.5: protect trades that showed useful MFE and then stalled.
        if self.flags.mfe_rescue_stop:
            rescue_stop = exits.compute_mfe_rescue_stop(pos, price)
            if rescue_stop != pos.stop_price:
                pos.stop_price = rescue_stop
                self._update_stop_price(rescue_stop, bar_time)

        # Early kill: fast-dying trades
        if self.flags.early_kill and exits.check_early_kill(pos, price):
            self._close_position(price, bar_time, "EARLY_KILL", market_exit=True)
            return

        # Max duration hard stop
        if self.flags.max_duration and exits.check_max_duration(pos):
            self._close_position(price, bar_time, "MAX_DURATION", market_exit=True)
            return

        # +1R free-ride (Section 16.1)
        if self.flags.plus_1r_partial and not pos.partial_done:
            # v4.2: CLOSE entries skip partial — move to BE at +1R, keep full position
            _is_close_entry = (
                self.flags.close_skip_partial
                and pos.entry_time is not None
                and classify_session(pos.entry_time)[1] == SubWindow.CLOSE
            )
            if _is_close_entry:
                if self._unrealized_r(pos, price) >= 1.0:
                    pos.stop_price = pos.entry_price
                    self._update_stop_price(pos.entry_price, bar_time)
                    pos.partial_done = True
                    pos.stage = PositionStage.ACTIVE_FREE
            else:
                qty_close = exits.check_partial(pos, price)
                if qty_close > 0:
                    self._execute_partial(pos, qty_close, price, bar_time)
                elif self._unrealized_r(pos, price) >= 1.0:
                    # 1-lot: just move stop to BE
                    pos.stop_price = pos.entry_price
                    self._update_stop_price(pos.entry_price, bar_time)
                    pos.partial_done = True
                    pos.stage = PositionStage.ACTIVE_FREE

        # ACTIVE_FREE tracking and exits
        if pos.stage == PositionStage.ACTIVE_FREE:
            pos.bars_since_partial += 1
            unreal_r = self._unrealized_r(pos, price)
            pos.peak_r_since_free = max(pos.peak_r_since_free, unreal_r)

            # Profit lock: tighten stop to lock +0.25R once peak >= 0.50R
            if self.flags.free_profit_lock:
                lock_stop = exits.compute_free_profit_lock(pos, price)
                if lock_stop != pos.stop_price:
                    pos.stop_price = lock_stop
                    self._update_stop_price(lock_stop, bar_time)

            # v4.2: CLOSE-specific MFE ratchet (applied in ACTIVE_FREE, after BE move)
            if self.flags.close_skip_partial and pos.entry_time is not None:
                if classify_session(pos.entry_time)[1] == SubWindow.CLOSE:
                    ratchet = exits.compute_close_mfe_ratchet(pos)
                    if ratchet > 0.0:
                        if pos.direction == Direction.LONG:
                            new_floor = max(pos.stop_price, ratchet)
                        else:
                            new_floor = min(pos.stop_price, ratchet)
                        if new_floor != pos.stop_price:
                            pos.stop_price = new_floor
                            self._update_stop_price(new_floor, bar_time)

            # Free-ride stale exit
            if self.flags.free_ride_stale and exits.check_free_ride_stale(pos, price):
                self._close_position(price, bar_time, "FREE_STALE", market_exit=True)
                return

        # Late Trail (v4.4) -- independent trail, no partial, late activation
        if self.flags.late_trail and not pos.partial_done:
            # BE move -- once peak_mfe_r crosses BE_R
            if (not pos.late_trail_be_done
                    and C.LATE_TRAIL_BE_R > 0
                    and pos.peak_mfe_r >= C.LATE_TRAIL_BE_R):
                if pos.direction == Direction.LONG:
                    new_be = max(pos.stop_price, pos.entry_price)
                else:
                    new_be = min(pos.stop_price, pos.entry_price)
                if new_be != pos.stop_price:
                    pos.stop_price = new_be
                    self._update_stop_price(new_be, bar_time)
                pos.late_trail_be_done = True

            # Activate trailing once MFE crosses activation threshold
            if not pos.late_trail_active and pos.peak_mfe_r >= C.LATE_TRAIL_ACTIVATE_R:
                pos.late_trail_active = True

            # Trail computation (only when active, not overnight)
            if pos.late_trail_active and not _is_overnight(bar_time) and t > 0:
                _, sub_window = classify_session(bar_time)
                h_slice = bars_15m.highs[:t + 1]
                l_slice = bars_15m.lows[:t + 1]
                new_stop = exits.compute_late_trail_stop(
                    pos, h_slice, l_slice, atr15, price,
                    sub_window=sub_window.value)
                if new_stop != pos.stop_price:
                    pos.stop_price = new_stop
                    self._update_stop_price(new_stop, bar_time)

        # Intraday trailing (Section 16.2) — post +1R, not overnight
        if pos.partial_done and not _is_overnight(bar_time) and t > 0:
            _, sub_window = classify_session(bar_time)
            # Window-specific trail tightening
            tf = C.TRAIL_WINDOW_MULT.get(sub_window.value, 1.0)
            # Additional tightening for OPEN entries transitioned to CORE
            entered_open = (pos.entry_time is not None and
                            classify_session(pos.entry_time)[1] == SubWindow.OPEN)
            if entered_open and sub_window == SubWindow.CORE:
                tf *= C.TRAIL_CORE_TRANSITION_REDUCTION
            h_slice = bars_15m.highs[:t + 1]
            l_slice = bars_15m.lows[:t + 1]
            trail_stage = pos.stage if self.flags.post_partial_trail_tighten else PositionStage.ACTIVE_RISK
            new_stop = exits.compute_intraday_trail(
                pos, h_slice, l_slice, atr15, price, tighten_factor=tf,
                stage=trail_stage)
            if new_stop != pos.stop_price:
                pos.stop_price = new_stop
                self._update_stop_price(new_stop, bar_time)

        # VWAP failure exit (Section 16.3) — pre +1R, skip evening (stale VWAP)
        vwap_fail_ok = self.flags.vwap_fail_evening or pos.entry_session != SessionWindow.EVENING
        if vwap_fail_ok and self.flags.vwap_failure_exit and not pos.partial_done and pos.vwap_used_at_entry != 0.0:
            c_slice = bars_15m.closes[:t + 1]
            if exits.check_vwap_failure(pos, c_slice, pos.vwap_used_at_entry):
                self._close_position(price, bar_time, "VWAP_FAIL", market_exit=True)
                return

        # Stale exit (Section 16.4) — pre +1R
        if self.flags.stale_exit and not pos.partial_done:
            # v4.3: exempt trades showing significant directional MFE
            if self.flags.stale_mfe_exempt and pos.peak_mfe_r >= C.STALE_MFE_EXEMPT_R:
                pass  # let trail/VWAP_A/gate handle exit instead
            else:
                _cur_sub_win = classify_session(bar_time)[1].value if self.flags.adaptive_stale else "CORE"
                if exits.check_stale_exit(pos, price, sub_window=_cur_sub_win):
                    self._close_position(price, bar_time, "STALE", market_exit=True)
                    return

    def _execute_partial(
        self, pos: PositionState, qty_close: int,
        price: float, bar_time: datetime,
    ) -> None:
        """Execute +1R partial close."""
        # Market-exit slippage on partial
        slip = self.cfg.slippage.slip_ticks_normal * self.tick
        if pos.direction == Direction.LONG:
            price -= slip   # selling partial: adverse = lower fill
        else:
            price += slip   # covering partial: adverse = higher fill

        # Partial PnL
        if pos.direction == Direction.LONG:
            partial_pnl = (price - pos.entry_price) * self.pv * qty_close
        else:
            partial_pnl = (pos.entry_price - price) * self.pv * qty_close

        commission = self.cfg.slippage.commission_per_contract * qty_close
        self.equity += partial_pnl - commission
        self._total_commission += commission
        pos.qty_open -= qty_close
        pos.stop_price = pos.entry_price
        self._update_stop_price(pos.entry_price, bar_time)
        pos.partial_done = True
        pos.stage = PositionStage.ACTIVE_FREE

        self._replay_core_step(bar_input=dict(
            bar_ts=bar_time,
            partial_exit_done=VdubPartialExitDone(
                pos_id=pos.trade_id,
                qty_closed=qty_close,
                new_qty=pos.qty_open,
            ),
        ))

    def _apply_overnight_trail(self, hourly: NumpyBars, h_idx: int, bar_time: datetime | None = None) -> None:
        """Overnight trail using 1H data."""
        if self._active is None:
            return
        pos = self._active.pos
        atr1h = self._safe_atr1h()
        if atr1h == 0:
            return

        h_slice = hourly.highs[:h_idx + 1]
        l_slice = hourly.lows[:h_idx + 1]
        new_stop = exits.compute_overnight_trail(pos, h_slice, l_slice, atr1h)
        if new_stop != pos.stop_price:
            if self.flags.overnight_widening or (
                (pos.direction == Direction.LONG and new_stop > pos.stop_price) or
                (pos.direction == Direction.SHORT and new_stop < pos.stop_price)
            ):
                old_stop = pos.stop_price
                pos.stop_price = new_stop
                # Update broker stop
                for o in self.broker.pending_orders:
                    if o.order_id == self._active.stop_order_id:
                        o.stop_price = new_stop
                        break
                # Core state tracking: overnight trail
                if bar_time is not None and new_stop != old_stop:
                    self._replay_core_step(bar_input=dict(
                        bar_ts=bar_time,
                        stop_updates=[VdubStopUpdateRequest(
                            pos_id=pos.trade_id,
                            new_stop=new_stop,
                            reason="overnight_trail",
                        )],
                    ))

    def _check_vwap_a_failure(
        self, bar_time: datetime,
        hourly: NumpyBars, h_idx: int,
        bars_15m: NumpyBars, t: int,
    ) -> None:
        """VWAP-A failure check (multi-session, profitable)."""
        if not self.flags.vwap_a_failure:
            return
        if self._active is None:
            return
        pos = self._active.pos
        if pos.session_count < C.VWAP_A_FAIL_MIN_SESSIONS or np.isnan(self._vwap_a_val):
            return
        if h_idx < 1:
            return

        close_1h = float(hourly.closes[h_idx])
        price = float(bars_15m.closes[t])
        atr1h = self._safe_atr1h()

        if exits.check_vwap_a_failure(pos, close_1h, self._vwap_a_val, price, atr1h=atr1h):
            self._close_position(price, bar_time, "VWAP_A_FAIL", market_exit=True)

    # ------------------------------------------------------------------
    # Decision gate (Section 17)
    # ------------------------------------------------------------------

    def _run_decision_gate(
        self, bar_time: datetime, bars_15m: NumpyBars, t: int,
    ) -> None:
        """Execute 15:50 decision gate."""
        if not self.flags.decision_gate:
            return
        if self._active is None:
            return
        pos = self._active.pos
        if pos.qty_open <= 0:
            return

        price = float(bars_15m.closes[t])
        friday = _is_friday(bar_time)

        if self.flags.friday_override and friday:
            pass  # Use Friday thresholds

        long_ok, short_ok = sig.slope_ok(self._mom15) if len(self._mom15) > 0 else (False, False)
        slope_ok_dir = long_ok if pos.direction == Direction.LONG else short_ok
        trend_ok = (pos.direction == Direction.LONG and self.regime.trend_1h == 1) or \
                   (pos.direction == Direction.SHORT and self.regime.trend_1h == -1)

        action, new_stop = exits.decision_gate(
            pos, friday, price, slope_ok_dir, trend_ok)

        self._active.record.decision_gate_action = action

        if action == "HOLD":
            if new_stop != pos.stop_price:
                pos.stop_price = new_stop
                self._update_stop_price(new_stop, bar_time)
            pos.stage = PositionStage.SWING_HOLD
        else:
            self._close_position(price, bar_time, "GATE_FLATTEN", market_exit=True)

    # ------------------------------------------------------------------
    # Shock mid-position
    # ------------------------------------------------------------------

    def _shock_tighten_all(self) -> None:
        if self._active and self._active.pos.qty_open > 0:
            new_stop = exits.shock_stop_tighten(self._active.pos)
            self._active.pos.stop_price = new_stop
            for o in self.broker.pending_orders:
                if o.order_id == self._active.stop_order_id:
                    o.stop_price = new_stop
                    break

    # ------------------------------------------------------------------
    # Event safety (Section 6)
    # ------------------------------------------------------------------

    def _update_event_state(self, bar_time: datetime) -> None:
        """Check event calendar and update blocking state."""
        if self._event_calendar is None:
            return

        es = self.event_state
        if es.block_end_ts and bar_time < es.block_end_ts:
            es.rearmed = False
            return
        if es.cooldown_remaining > 0:
            es.cooldown_remaining -= 1
            es.rearmed = False
            return
        if not es.rearmed and es.block_end_ts and bar_time >= es.block_end_ts:
            if self._check_rearm():
                es.rearmed = True
            else:
                max_ext = es.block_end_ts + timedelta(minutes=C.MAX_POST_EVENT_MINUTES)
                if bar_time >= max_ext:
                    es.rearmed = True

        # Check upcoming events
        et = _to_et(bar_time)
        for event in self._event_calendar:
            event_dt = event.get("datetime")
            if event_dt is None:
                continue
            pre_block = event_dt - timedelta(minutes=C.EVENT_PRE_MINUTES)
            if pre_block <= bar_time < event_dt and es.rearmed:
                es.blocked = True
                es.block_end_ts = event_dt + timedelta(minutes=C.EVENT_POST_MINUTES)
                es.event_type = event.get("event_type", "UNKNOWN")
                es.rearmed = False
                es.cooldown_remaining = C.COOLDOWN_BARS.get(es.event_type, 3)
                if len(self._atr15) >= 12:
                    es.pre_event_atr15 = float(np.nanmean(self._atr15[-12:]))
                break

    def _check_rearm(self) -> bool:
        if len(self._atr15) < 2 or np.isnan(self._atr15[-1]):
            return True
        current = float(self._atr15[-1])
        pre = self.event_state.pre_event_atr15
        if pre > 0 and current < C.ATR_NORM_MULT * pre:
            return True
        return False

    @staticmethod
    def _load_event_calendar(path) -> list[dict]:
        """Load news calendar CSV."""
        import csv
        events = []
        try:
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    date_str = row.get("date", "")
                    time_str = row.get("time_et", "")
                    event_type = row.get("event_type", "")
                    try:
                        dt = datetime.strptime(
                            f"{date_str} {time_str}", "%Y-%m-%d %H:%M"
                        )
                        from zoneinfo import ZoneInfo
                        dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
                        dt = dt.astimezone(timezone.utc)
                        events.append({"datetime": dt, "event_type": event_type})
                    except (ValueError, TypeError):
                        continue
        except Exception:
            logger.warning("Failed to load event calendar: %s", path)
        return events

    # ------------------------------------------------------------------
    # Direction evaluation (entry signal pipeline)
    # ------------------------------------------------------------------

    def _evaluate_direction(
        self,
        direction: Direction,
        session: SessionWindow,
        sub_window: SubWindow,
        bar_time: datetime,
        bars_15m: NumpyBars,
        hourly: NumpyBars,
        h_idx: int,
        t: int,
        bars_5m: NumpyBars | None,
        five_to_15_idx_map: np.ndarray | None,
    ) -> None:
        """Full entry evaluation for one direction."""
        self._evaluations += 1

        # Drawdown throttle: daily loss cap halt (matches live engine strategy_3/engine.py:322)
        if self._throttle.daily_halted:
            return

        # Build signal event for tracking
        evt = VdubusSignalEvent(
            timestamp=bar_time,
            direction=int(direction),
            session=session.value,
            sub_window=sub_window.value,
        )
        entry_size_mult = 1.0

        # Pre-compute approximate entry/stop for shadow tracking
        _atr15 = self._safe_atr15()
        _atr1h = self._safe_atr1h()
        if _atr15 > 0:
            _atr15_ticks = _atr15 / self.tick
            _se, _ = risk.compute_entry_prices(
                float(bars_15m.highs[t]), float(bars_15m.lows[t]),
                _atr15_ticks, direction,
            )
            _is = risk.compute_initial_stop(_se, direction, self._pivots_1h, _atr1h, _atr15)
            evt.would_be_entry = _se
            evt.would_be_stop = _is

        # --- Gate 0: v4.2 Evening 20:00 ET hour block ---
        if self.flags.block_20h_hour and session == SessionWindow.EVENING:
            et_hour = _to_et(bar_time).hour
            if et_hour == 20:
                evt.signal_pass = False
                evt.first_block_reason = "evening_20h_block"
                self._record_signal_event(evt)
                return

        # --- Gate 1: Regime permission (Section 4) ---
        is_flip = False
        if self.flags.daily_trend_gate:
            if not reg.direction_allowed(self.regime, direction):
                if reg.flip_entry_eligible(self.regime, self.counters, direction):
                    is_flip = True
                else:
                    evt.regime_pass = False
                    evt.first_block_reason = "daily_trend"
                    self._record_signal_event(evt)
                    return

        # --- Late shoulder conditional gates ---
        in_late_shoulder = _is_late_shoulder(bar_time)
        if in_late_shoulder:
            if C.LATE_SHOULDER_REQUIRE_1H_ALIGN:
                aligned_1h = (direction == Direction.LONG and self.regime.trend_1h == 1) or \
                             (direction == Direction.SHORT and self.regime.trend_1h == -1)
                if not aligned_1h:
                    evt.signal_pass = False
                    evt.first_block_reason = "shoulder_1h"
                    self._record_signal_event(evt)
                    return
            if C.LATE_SHOULDER_REQUIRE_LOW_CHOP and self.regime.choppiness > C.CHOP_THRESHOLD:
                evt.signal_pass = False
                evt.first_block_reason = "shoulder_chop"
                self._record_signal_event(evt)
                return

        # --- Gate 2: 1H alignment -> hard gate (unless flip) ---
        hourly_mult = C.HOURLY_ALIGNED_MULT
        if self.flags.hourly_alignment and not is_flip:
            aligned = (direction == Direction.LONG and self.regime.trend_1h == 1) or \
                      (direction == Direction.SHORT and self.regime.trend_1h == -1)
            if not aligned:
                if self.flags.hourly_bypass_quality and self._quality_bypass_ok(
                    direction, sub_window, bars_15m, hourly, h_idx, t,
                    min_eqs=C.HOURLY_BYPASS_EQS_MIN,
                    max_chop=C.HOURLY_BYPASS_MAX_CHOP,
                    allowed_windows=C.HOURLY_BYPASS_ALLOWED_WINDOWS,
                ):
                    hourly_mult = min(hourly_mult, C.HOURLY_BYPASS_SIZE_MULT)
                else:
                    evt.alignment_pass = False
                    evt.first_block_reason = "hourly_align"
                    self._record_signal_event(evt)
                    return

        self._regime_passed += 1

        # --- Gate 3: Direction caps (reduced in choppy regimes) ---
        if self.flags.direction_caps:
            if self.flags.choppiness_gate and self.regime.choppiness > C.CHOP_THRESHOLD:
                max_l = C.CHOP_MAX_LONGS
                max_s = C.CHOP_MAX_SHORTS
            else:
                max_l = C.MAX_LONGS_PER_DAY
                max_s = C.MAX_SHORTS_PER_DAY
            if direction == Direction.LONG and self.counters.long_fills >= max_l:
                evt.direction_cap_pass = False
                evt.first_block_reason = "long_cap"
                self._record_signal_event(evt)
                return
            if direction == Direction.SHORT and self.counters.short_fills >= max_s:
                evt.direction_cap_pass = False
                evt.first_block_reason = "short_cap"
                self._record_signal_event(evt)
                return

        # --- Gate 4: Momentum slope (Section 7) ---
        if self.flags.slope_gate and len(self._mom15) > 0:
            long_ok, short_ok = sig.slope_ok(self._mom15)
            if direction == Direction.LONG and not long_ok:
                if self.flags.slope_bypass_quality and self._quality_bypass_ok(
                    direction, sub_window, bars_15m, hourly, h_idx, t,
                    min_eqs=C.SLOPE_BYPASS_EQS_MIN,
                    max_chop=C.SLOPE_BYPASS_MAX_CHOP,
                    allowed_windows=C.SLOPE_BYPASS_ALLOWED_WINDOWS,
                    mom_abs_min=C.SLOPE_BYPASS_MOM_ABS_MIN,
                ):
                    entry_size_mult = min(entry_size_mult, C.SLOPE_BYPASS_SIZE_MULT)
                else:
                    evt.slope_pass = False
                    evt.first_block_reason = "slope"
                    self._record_signal_event(evt)
                    return
            if direction == Direction.SHORT and not short_ok:
                if self.flags.slope_bypass_quality and self._quality_bypass_ok(
                    direction, sub_window, bars_15m, hourly, h_idx, t,
                    min_eqs=C.SLOPE_BYPASS_EQS_MIN,
                    max_chop=C.SLOPE_BYPASS_MAX_CHOP,
                    allowed_windows=C.SLOPE_BYPASS_ALLOWED_WINDOWS,
                    mom_abs_min=C.SLOPE_BYPASS_MOM_ABS_MIN,
                ):
                    entry_size_mult = min(entry_size_mult, C.SLOPE_BYPASS_SIZE_MULT)
                else:
                    evt.slope_pass = False
                    evt.first_block_reason = "slope"
                    self._record_signal_event(evt)
                    return

        # --- Gate 5: Signal detection (Type A / Type B) ---
        atr15_val = self._safe_atr15()
        atr1h_val = self._safe_atr1h()
        if atr15_val == 0:
            return

        closes_15m = bars_15m.closes[:t + 1]
        lows_15m = bars_15m.lows[:t + 1]
        highs_15m = bars_15m.highs[:t + 1]

        svwap = self._svwap if len(self._svwap) > 0 else np.full(t + 1, np.nan)
        vwap_a = self._vwap_a_arr if len(self._vwap_a_arr) > 0 else np.full(t + 1, np.nan)

        signal = None
        signal_type = EntryType.TYPE_A
        vwap_used = 0.0

        if self.flags.type_a_enabled:
            signal = sig.type_a_check(
                closes_15m, lows_15m, highs_15m,
                svwap, vwap_a,
                atr15_val, direction, sub_window,
            )
            if signal:
                signal_type = EntryType.TYPE_A
                vwap_used = signal.get("vwap_used", 0.0) or 0.0

        if signal is None and C.USE_TYPE_B and self.flags.type_b_enabled and sub_window.value in C.TYPE_B_ALLOWED_WINDOWS:
            # Type B: require 1H alignment if configured
            type_b_ok = True
            if C.TYPE_B_REQUIRE_1H_ALIGN:
                aligned_1h = (direction == Direction.LONG and self.regime.trend_1h == 1) or \
                             (direction == Direction.SHORT and self.regime.trend_1h == -1)
                if not aligned_1h:
                    type_b_ok = False
            if type_b_ok:
                signal = sig.type_b_check(
                    closes_15m, lows_15m, highs_15m,
                    self._pivots_1h, len(hourly.closes[:h_idx + 1]) if h_idx >= 0 else 0,
                    atr15_val, direction,
                )
                if signal:
                    signal_type = EntryType.TYPE_B

        if signal is None and C.USE_TYPE_C and self.flags.type_c_enabled:
            signal = sig.type_c_continuation_check(
                closes_15m, lows_15m, highs_15m,
                svwap, vwap_a, atr15_val, direction, sub_window,
            )
            if signal:
                signal_type = EntryType.TYPE_C
                vwap_used = signal.get("vwap_used", 0.0) or 0.0

        if signal is None:
            evt.signal_pass = False
            evt.first_block_reason = "no_signal"
            self._record_signal_event(evt)
            return

        self._signals_found += 1

        # v4.2: Bar quality gate — filter spike bars, weak closes, single-bar reclaims
        if self.flags.bar_quality_gate and signal_type == EntryType.TYPE_A:
            _high = float(highs_15m[-1])
            _low = float(lows_15m[-1])
            _close = float(closes_15m[-1])
            _bar_range = _high - _low
            # Filter 1: Spike bar (range > N × ATR)
            if atr15_val > 0 and _bar_range > C.BAR_QUALITY_SPIKE_ATR * atr15_val:
                evt.signal_pass = False
                evt.first_block_reason = "bq_spike"
                self._record_signal_event(evt)
                return
            # Filter 2: Weak close (long close in bottom N% of bar range)
            if _bar_range > 0:
                if direction == Direction.LONG:
                    close_frac = (_close - _low) / _bar_range
                    if close_frac < C.BAR_QUALITY_CLOSE_FRAC:
                        evt.signal_pass = False
                        evt.first_block_reason = "bq_weak_close"
                        self._record_signal_event(evt)
                        return
                else:
                    close_frac = (_high - _close) / _bar_range
                    if close_frac < C.BAR_QUALITY_CLOSE_FRAC:
                        evt.signal_pass = False
                        evt.first_block_reason = "bq_weak_close"
                        self._record_signal_event(evt)
                        return
            # Filter 3: Persistence (previous bar also must have closed above/below VWAP)
            if C.BAR_QUALITY_PERSIST and len(closes_15m) >= 2:
                _vwap = signal.get("vwap_used", 0.0) or 0.0
                if _vwap > 0:
                    _prev_close = float(closes_15m[-2])
                    if direction == Direction.LONG and _prev_close <= _vwap:
                        evt.signal_pass = False
                        evt.first_block_reason = "bq_no_persist"
                        self._record_signal_event(evt)
                        return
                    if direction == Direction.SHORT and _prev_close >= _vwap:
                        evt.signal_pass = False
                        evt.first_block_reason = "bq_no_persist"
                        self._record_signal_event(evt)
                        return

        # v4.1: Entry Quality Score gate
        if self.flags.entry_quality_gate and C.EQS_MIN_RTH > 0:
            eqs_min = C.EQS_MIN_EVENING if session == SessionWindow.EVENING else C.EQS_MIN_RTH
            if eqs_min > 0:
                eqs = self._compute_eqs(
                    direction, closes_15m, highs_15m, lows_15m,
                    atr15_val, h_idx, hourly,
                )
                if eqs < eqs_min:
                    evt.signal_pass = False
                    evt.first_block_reason = f"eqs_{eqs}"
                    self._record_signal_event(evt)
                    return

        # v4.1: Evening-specific VWAP cap
        if self.flags.evening_vwap_cap and session == SessionWindow.EVENING and signal_type == EntryType.TYPE_A:
            close = float(closes_15m[-1])
            vw = signal.get("vwap_used", 0.0)
            if vw and atr15_val > 0:
                dist = abs(close - vw)
                if dist > C.VWAP_CAP_EVENING * atr15_val:
                    evt.signal_pass = False
                    evt.first_block_reason = "evening_vwap_cap"
                    self._record_signal_event(evt)
                    return

        # Late shoulder: tighter VWAP cap override
        if in_late_shoulder and signal.get("type") == "A":
            close = float(closes_15m[-1])
            vw = signal.get("vwap_used", 0.0)
            if vw and atr15_val > 0:
                dist = abs(close - vw)
                if dist > C.LATE_SHOULDER_VWAP_CAP * atr15_val:
                    evt.signal_pass = False
                    evt.first_block_reason = "shoulder_vwap_cap"
                    self._record_signal_event(evt)
                    return

        # --- Gate 6: Predator overlay -> class_mult (Section 8) ---
        if is_flip:
            class_mult = C.CLASS_MULT_FLIP
        elif self.flags.predator_overlay and sig.predator_present(
            self._pivots_1h,
            hourly.highs[:h_idx + 1] if h_idx >= 0 else np.array([]),
            hourly.lows[:h_idx + 1] if h_idx >= 0 else np.array([]),
            self._mom15, 4, direction,
        ):
            class_mult = C.CLASS_MULT_PREDATOR
        else:
            class_mult = C.CLASS_MULT_NOPRED

        # Late shoulder: cap class_mult
        if in_late_shoulder:
            class_mult = min(class_mult, C.LATE_SHOULDER_CLASS_MULT)

        # Type B: cap class_mult
        if signal_type == EntryType.TYPE_B:
            class_mult = min(class_mult, C.TYPE_B_CLASS_MULT)
        if signal_type == EntryType.TYPE_C:
            class_mult = min(class_mult, C.TYPE_C_CLASS_MULT)

        session_key = "RTH" if session == SessionWindow.RTH else "EVENING"
        session_mult = C.SESSION_MULT[session_key]

        # Check pyramiding
        is_pyramid = False
        if self._active and self._active.pos.qty_open > 0:
            close_price = float(closes_15m[-1])
            if risk.pyramid_eligible(self._active.pos, direction, close_price, self.counters):
                is_pyramid = True
            else:
                return  # Already have position, can't pyramid

        # --- Compute entry/stop prices (Section 15) ---
        atr15_ticks = atr15_val / self.tick
        stop_entry, limit_entry = risk.compute_entry_prices(
            float(highs_15m[-1]), float(lows_15m[-1]),
            atr15_ticks, direction,
        )
        entry_est = stop_entry

        # Initial stop (Section 13)
        initial_stop = risk.compute_initial_stop(
            entry_est, direction, self._pivots_1h, atr1h_val, atr15_val)
        r_points = abs(entry_est - initial_stop)
        if r_points == 0:
            return

        # --- Gate 7: Sizing + viability (Section 12, 14) ---
        unit_risk = risk.compute_unit_risk(self.equity, self.regime.vol_state)
        eff_risk = risk.compute_effective_risk(unit_risk, class_mult, session_mult * hourly_mult)
        if is_pyramid:
            eff_risk = risk.compute_addon_risk(eff_risk)

        if self.cfg.fixed_qty is not None:
            qty = max(1, int(self.cfg.fixed_qty * hourly_mult * entry_size_mult))
        else:
            qty = risk.compute_qty(eff_risk * entry_size_mult, r_points)

        # v4.1: Day-of-week sizing reduction
        if self.flags.dow_sizing and C.DOW_SIZE_MULT:
            from strategies.momentum.vdub.config import DOW_SIZE_MULT
            weekday = _to_et(bar_time).weekday()
            dow_mult = DOW_SIZE_MULT.get(weekday, DOW_SIZE_MULT.get(str(weekday), 1.0))
            if dow_mult < 1.0:
                qty = max(1, int(qty * dow_mult))

        # Drawdown throttle: reduce sizing during drawdowns (matches live engine strategy_3/engine.py:606-608)
        dd_mult = max(0.75, self._throttle.dd_size_mult)
        if dd_mult < 1.0:
            qty = max(1, int(qty * dd_mult))

        if qty < 1:
            return

        if self.flags.viability_filter:
            ok, reason = risk.pass_viability(qty, r_points, sub_window)
            if not ok:
                evt.viability_pass = False
                evt.first_block_reason = f"viability_{reason}"
                self._record_signal_event(evt)
                return

        # --- Gate 8: Risk gates (heat cap, breaker, direction caps) ---
        if self.flags.heat_cap:
            open_risk = self._compute_open_risk()
            new_risk = r_points * self.pv * qty
            ok, reason = risk.pass_risk_gates(
                self.counters, direction, open_risk, new_risk, unit_risk)
            if not ok:
                evt.risk_gate_pass = False
                evt.first_block_reason = f"risk_{reason}"
                self._record_signal_event(evt)
                return

        # --- ALL GATES PASSED ---
        evt.passed_all = True
        evt.entry_type = signal_type.value
        evt.would_be_entry = entry_est
        evt.would_be_stop = initial_stop
        self._record_signal_event(evt)

        # --- Optional 5m micro-trigger ---
        if (C.USE_MICRO_TRIGGER and bars_5m is not None
                and five_to_15_idx_map is not None):
            micro_result = self._check_micro_trigger(
                bars_5m, five_to_15_idx_map, t,
                direction, atr15_val, sub_window,
            )
            if micro_result is not None:
                stop_entry, limit_entry = micro_result

        # Submit entry order
        self._submit_entry(
            direction=direction, qty=qty,
            stop_entry=stop_entry, limit_entry=limit_entry,
            initial_stop=initial_stop,
            signal_type=signal_type, is_flip=is_flip, is_pyramid=is_pyramid,
            class_mult=class_mult, vwap_used=vwap_used,
            session=session, sub_window=sub_window,
            bar_time=bar_time,
        )

    # ------------------------------------------------------------------
    # 5m micro-trigger
    # ------------------------------------------------------------------

    def _check_micro_trigger(
        self,
        bars_5m: NumpyBars,
        five_to_15_idx_map: np.ndarray,
        t_15m: int,
        direction: Direction,
        atr15_val: float,
        sub_window: SubWindow,
    ) -> tuple[float, float] | None:
        """Scan 5m bars within current 15m bar for micro-confirmation.

        Returns (stop_entry, limit_entry) with tighter entry if confirmed,
        or None to use standard 15m entry.
        """
        # Find 5m bars that map to this 15m bar
        mask = five_to_15_idx_map == t_15m
        indices_5m = np.where(mask)[0]

        if len(indices_5m) == 0:
            return None

        window = min(C.MICRO_WINDOW_BARS, len(indices_5m))
        for i in indices_5m[:window]:
            # Simple micro-confirmation: 5m bar closes in trade direction
            close_5m = float(bars_5m.closes[i])
            if len(self._svwap) > 0 and not np.isnan(self._svwap[-1]):
                vwap_val = float(self._svwap[-1])
                if direction == Direction.LONG and close_5m > vwap_val:
                    # Tighter entry: use 5m bar high instead of 15m bar high
                    h5 = float(bars_5m.highs[i])
                    l5 = float(bars_5m.lows[i])
                    atr_ticks = atr15_val / self.tick
                    return risk.compute_entry_prices(h5, l5, atr_ticks, direction)
                elif direction == Direction.SHORT and close_5m < vwap_val:
                    h5 = float(bars_5m.highs[i])
                    l5 = float(bars_5m.lows[i])
                    atr_ticks = atr15_val / self.tick
                    return risk.compute_entry_prices(h5, l5, atr_ticks, direction)

        return None

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def _submit_entry(
        self, *,
        direction: Direction, qty: int,
        stop_entry: float, limit_entry: float,
        initial_stop: float,
        signal_type: EntryType, is_flip: bool, is_pyramid: bool,
        class_mult: float, vwap_used: float,
        session: SessionWindow, sub_window: SubWindow,
        bar_time: datetime,
    ) -> None:
        """Submit stop-limit entry order via SimBroker."""
        # Don't submit if we already have working orders
        if self._working:
            return

        side = OrderSide.BUY if direction == Direction.LONG else OrderSide.SELL
        order_id = self.broker.next_order_id()

        ttl_minutes = C.TTL_BARS * 15  # 15m bars

        order = SimOrder(
            order_id=order_id, symbol=self.symbol, side=side,
            order_type=OrderType.STOP_LIMIT, qty=qty,
            stop_price=stop_entry, limit_price=limit_entry,
            tick_size=self.tick, submit_time=bar_time,
            ttl_minutes=ttl_minutes,
            tag=signal_type.value,
        )
        self.broker.submit_order(order)

        we = WorkingEntry(
            oms_order_id=order_id,
            entry_type=signal_type, direction=direction,
            stop_entry=stop_entry, limit_entry=limit_entry,
            qty=qty, submitted_bar_idx=self._bar_idx,
            ttl_bars=C.TTL_BARS,
            initial_stop=initial_stop, vwap_used=vwap_used,
            class_mult=class_mult,
            session=session,
            is_flip=is_flip, is_addon=is_pyramid,
        )
        self._working[order_id] = we
        self._entries_placed += 1

        self._replay_core_step(bar_input=dict(
            bar_ts=bar_time,
            entry_submitted=VdubEntrySubmitted(
                working_entry=we, oms_order_id=order_id, bar_idx=self._bar_idx,
            ),
        ))

        if is_flip:
            if direction == Direction.LONG:
                self.counters.flip_entry_used_long = True
            else:
                self.counters.flip_entry_used_short = True
        if is_pyramid:
            if direction == Direction.LONG:
                self.counters.addon_used_long = True
            else:
                self.counters.addon_used_short = True

    # ------------------------------------------------------------------
    # Fill handling
    # ------------------------------------------------------------------

    def _handle_fill(
        self, fill: FillResult, bar_time: datetime, current_close: float,
        session: SessionWindow, sub_window: SubWindow,
    ) -> None:
        order = fill.order

        if fill.status == FillStatus.FILLED:
            we = self._working.pop(order.order_id, None)
            if we is not None:
                self._on_entry_fill(we, fill, bar_time, session, sub_window)
            elif self._active and order.order_id == self._active.stop_order_id:
                self._on_stop_fill(fill, bar_time)

        elif fill.status == FillStatus.EXPIRED:
            self._working.pop(order.order_id, None)
            self._replay_core_step(order_updates=[VdubOrderUpdate(
                oms_order_id=order.order_id, status="EXPIRED",
                timestamp=bar_time, order_role="entry",
            )])

        elif fill.status == FillStatus.CANCELLED:
            self._working.pop(order.order_id, None)
            self._replay_core_step(order_updates=[VdubOrderUpdate(
                oms_order_id=order.order_id, status="CANCELLED",
                timestamp=bar_time, order_role="entry",
            )])

    def _on_entry_fill(
        self, we: WorkingEntry, fill: FillResult, bar_time: datetime,
        session: SessionWindow, sub_window: SubWindow,
    ) -> None:
        fill_price = fill.fill_price
        commission = fill.commission
        commission_before_trade = self._total_commission
        self._total_commission += commission
        self.equity -= commission
        self._throttle.update_equity(self.equity)
        self._entries_filled += 1

        r_points = abs(fill_price - we.initial_stop)
        if r_points <= 0:
            r_points = 1.0

        pos = PositionState(
            trade_id=uuid.uuid4().hex[:12],
            direction=we.direction,
            entry_price=fill_price,
            stop_price=we.initial_stop,
            qty_entry=we.qty,
            qty_open=we.qty,
            r_points=r_points,
            entry_time=bar_time,
            entry_type=we.entry_type,
            vwap_used_at_entry=we.vwap_used,
            is_addon=we.is_addon,
            class_mult=we.class_mult,
            is_flip_entry=we.is_flip,
            highest_since_entry=fill_price,
            lowest_since_entry=fill_price,
            entry_session=we.session,
        )

        record = VdubusTradeRecord(
            symbol=self.symbol,
            direction=int(we.direction),
            entry_type=we.entry_type.value,
            is_flip=we.is_flip,
            is_addon=we.is_addon,
            session=session.value,
            sub_window=sub_window.value,
            entry_time=bar_time,
            entry_price=fill_price,
            initial_stop=we.initial_stop,
            qty=we.qty,
            daily_trend=self.regime.daily_trend,
            vol_state=self.regime.vol_state.value,
            trend_1h=self.regime.trend_1h,
            class_mult=we.class_mult,
            vwap_used_at_entry=we.vwap_used,
            signal_entry_price=we.stop_entry,
        )

        active = _ActivePosition(
            pos=pos, record=record,
            mfe_price=fill_price, mae_price=fill_price,
            commission_at_start=commission_before_trade,
        )
        self._active = active

        self._replay_core_step(fills=[VdubFill(
            oms_order_id=we.oms_order_id,
            fill_price=fill_price,
            fill_qty=we.qty,
            fill_time=bar_time,
            point_value=self.pv,
            commission=commission,
            entry_context=VdubEntryFillContext(working_entry=we),
        )])

        # Place protective stop
        stop_side = OrderSide.SELL if we.direction == Direction.LONG else OrderSide.BUY
        stop_id = self.broker.next_order_id()
        stop_order = SimOrder(
            order_id=stop_id, symbol=self.symbol, side=stop_side,
            order_type=OrderType.STOP, qty=we.qty,
            stop_price=we.initial_stop, tick_size=self.tick,
            submit_time=bar_time, tag="protective_stop",
        )
        self.broker.submit_order(stop_order)
        active.stop_order_id = stop_id

        # Update counters
        if we.direction == Direction.LONG:
            self.counters.long_fills += 1
        else:
            self.counters.short_fills += 1

    def _on_stop_fill(self, fill: FillResult, bar_time: datetime) -> None:
        if self._active is None:
            return
        self._replay_core_step(fills=[VdubFill(
            oms_order_id=fill.order.order_id,
            fill_price=fill.fill_price,
            fill_qty=self._active.pos.qty_open,
            fill_time=bar_time,
            point_value=self.pv,
            commission=fill.commission,
        )])
        # Commission handled in _close_position (single authority for exit commission)
        self._close_position(fill.fill_price, bar_time, "STOP")

    # ------------------------------------------------------------------
    # Close position
    # ------------------------------------------------------------------

    def _close_position(
        self, exit_price: float, bar_time: datetime, reason: str,
        market_exit: bool = False,
    ) -> None:
        if self._active is None:
            return
        pos = self._active.pos
        if pos.qty_open <= 0:
            return

        # Flatten core step for discretionary exits
        if market_exit:
            self._replay_core_step(bar_input=dict(
                bar_ts=bar_time,
                flatten_requests=[VdubFlattenRequest(
                    pos_id=pos.trade_id, reason=reason,
                )],
            ))

        # Market-exit slippage (discretionary exits at bar close)
        if market_exit:
            slip = self.cfg.slippage.slip_ticks_normal * self.tick
            if pos.direction == Direction.LONG:
                exit_price -= slip   # selling: adverse = lower fill
            else:
                exit_price += slip   # covering: adverse = higher fill

        # PnL on remaining qty
        if pos.direction == Direction.LONG:
            pnl = (exit_price - pos.entry_price) * self.pv * pos.qty_open
        else:
            pnl = (pos.entry_price - exit_price) * self.pv * pos.qty_open

        commission = self.cfg.slippage.commission_per_contract * pos.qty_open
        self._total_commission += commission
        self.equity += pnl - commission

        # R-multiple
        r_points = pos.r_points
        if r_points > 0:
            if pos.direction == Direction.LONG:
                r_mult = (exit_price - pos.entry_price) / r_points
            else:
                r_mult = (pos.entry_price - exit_price) / r_points
        else:
            r_mult = 0.0

        # MFE/MAE in R
        if r_points > 0:
            if pos.direction == Direction.LONG:
                mfe_r = (self._active.mfe_price - pos.entry_price) / r_points
                mae_r = (pos.entry_price - self._active.mae_price) / r_points
            else:
                mfe_r = (pos.entry_price - self._active.mfe_price) / r_points
                mae_r = (self._active.mae_price - pos.entry_price) / r_points
        else:
            mfe_r = mae_r = 0.0

        # Update daily realized PnL
        self.counters.daily_realized_pnl += pnl

        # Drawdown throttle
        self._throttle.update_equity(self.equity)
        self._throttle.record_trade_close(r_mult)

        # Finalize record
        record = self._active.record
        record.exit_time = bar_time
        record.exit_price = exit_price
        record.pnl_dollars = pnl - commission
        record.r_multiple = r_mult
        record.mfe_r = mfe_r
        record.mae_r = mae_r
        record.exit_reason = reason
        record.bars_held_15m = pos.bars_since_entry
        record.overnight_sessions = pos.session_count
        record.partial_done = pos.partial_done
        record.stage_at_exit = pos.stage.value
        record.commission = self._total_commission - self._active.commission_at_start

        self._trades.append(record)
        self._recent_wins.append(pnl > 0)

        # Cancel all remaining orders
        self.broker.cancel_all(self.symbol)
        self._working.clear()

        pos.qty_open = 0
        self._active = None

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def _update_mfe_mae(self, H: float, L: float) -> None:
        if self._active is None:
            return
        if self._active.pos.direction == Direction.LONG:
            self._active.mfe_price = max(self._active.mfe_price, H)
            self._active.mae_price = min(self._active.mae_price, L)
        else:
            self._active.mfe_price = min(self._active.mfe_price, L)
            self._active.mae_price = max(self._active.mae_price, H)

    def _update_stop_price(self, new_stop: float, bar_time: datetime) -> None:
        if self._active is None:
            return
        new_stop = round_to_tick(new_stop, self.tick)
        old_stop = self._active.pos.stop_price
        self._active.pos.stop_price = new_stop
        for o in self.broker.pending_orders:
            if o.order_id == self._active.stop_order_id:
                o.stop_price = new_stop
                break
        if new_stop != old_stop:
            self._replay_core_step(bar_input=dict(
                bar_ts=bar_time,
                stop_updates=[VdubStopUpdateRequest(
                    pos_id=self._active.pos.trade_id,
                    new_stop=new_stop,
                    reason="trail",
                )],
            ))

    def _unrealized_r(self, pos: PositionState, price: float) -> float:
        if pos.r_points <= 0:
            return 0.0
        if pos.direction == Direction.LONG:
            return (price - pos.entry_price) / pos.r_points
        return (pos.entry_price - price) / pos.r_points

    def _quality_bypass_ok(
        self,
        direction: Direction,
        sub_window: SubWindow,
        bars_15m: NumpyBars,
        hourly: NumpyBars,
        h_idx: int,
        t: int,
        *,
        min_eqs: int,
        max_chop: float,
        allowed_windows,
        mom_abs_min: float = 0.0,
    ) -> bool:
        """Shared quality guard for conditional gate-bypass experiments."""
        if sub_window.value not in allowed_windows:
            return False
        if self.regime.choppiness > max_chop:
            return False
        atr15 = self._safe_atr15()
        if atr15 <= 0:
            return False
        if mom_abs_min > 0:
            if len(self._mom15) == 0 or np.isnan(self._mom15[-1]):
                return False
            if abs(float(self._mom15[-1])) < mom_abs_min:
                return False

        closes_15m = bars_15m.closes[:t + 1]
        highs_15m = bars_15m.highs[:t + 1]
        lows_15m = bars_15m.lows[:t + 1]
        return self._compute_eqs(
            direction, closes_15m, highs_15m, lows_15m, atr15, h_idx, hourly,
        ) >= min_eqs

    def _compute_eqs(
        self, direction: Direction,
        closes_15m: np.ndarray, highs_15m: np.ndarray, lows_15m: np.ndarray,
        atr15: float, h_idx: int, hourly,
    ) -> int:
        """Compute Entry Quality Score (0-4) for v4.1 gate."""
        score = 0
        n = len(closes_15m)
        if n < 4 or atr15 <= 0:
            return 0

        # 1. Structural confirmation: pivot support/resistance exists
        if self._pivots_1h:
            last_pivot = self._pivots_1h[-1]
            if direction == Direction.LONG and last_pivot.price < closes_15m[-1]:
                score += 1
            elif direction == Direction.SHORT and last_pivot.price > closes_15m[-1]:
                score += 1

        # 2. VWAP approach quality: >=2 of last 3 bars moving toward VWAP
        lb = min(C.EQS_APPROACH_BARS, n - 1)
        if lb >= 2 and len(self._svwap) > 0:
            vwap_val = float(self._svwap[-1]) if not np.isnan(self._svwap[-1]) else 0
            if vwap_val > 0:
                approach_count = 0
                for i in range(-lb, 0):
                    prev_dist = abs(float(closes_15m[i - 1]) - vwap_val)
                    curr_dist = abs(float(closes_15m[i]) - vwap_val)
                    if curr_dist < prev_dist:
                        approach_count += 1
                if approach_count >= 2:
                    score += 1

        # 3. Bar quality: trigger bar range < 1.5x ATR15
        bar_range = float(highs_15m[-1]) - float(lows_15m[-1])
        if bar_range < C.EQS_BAR_RANGE_ATR_MAX * atr15:
            score += 1

        # 4. MACD histogram alignment
        if len(self._mom15) > 0 and not np.isnan(self._mom15[-1]):
            hist_val = float(self._mom15[-1])
            if (direction == Direction.LONG and hist_val > 0) or \
               (direction == Direction.SHORT and hist_val < 0):
                score += 1

        return score

    def _compute_open_risk(self) -> float:
        if self._active is None or self._active.pos.qty_open <= 0:
            return 0.0
        pos = self._active.pos
        return pos.r_points * self.pv * pos.qty_open

    def _safe_atr15(self) -> float:
        if len(self._atr15) == 0 or np.isnan(self._atr15[-1]):
            return 0.0
        return float(self._atr15[-1])

    def _safe_atr1h(self) -> float:
        if len(self._atr1h) == 0 or np.isnan(self._atr1h[-1]):
            return 0.0
        return float(self._atr1h[-1])

    def _record_signal_event(self, evt: VdubusSignalEvent) -> None:
        if not evt.passed_all and self.shadow_tracker is not None and evt.would_be_entry > 0:
            self.shadow_tracker.record_rejection(
                direction=evt.direction,
                filter_name=evt.first_block_reason,
                time=evt.timestamp,
                entry_price=evt.would_be_entry,
                stop_price=evt.would_be_stop,
                session=evt.session,
                sub_window=evt.sub_window,
                entry_type=evt.entry_type,
            )
        if self.cfg.track_signals:
            self._signal_events.append(evt)

    def _equity_snapshot(self, bar_time: datetime, current_price: float = 0.0) -> None:
        mtm = self.equity
        if current_price and self._active and self._active.pos.qty_open > 0:
            p = self._active.pos
            mtm += (current_price - p.entry_price) * p.direction * self.pv * p.qty_open
        self._equity_history.append(mtm)
        self._time_history.append(bar_time)

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def _check_daily_reset(self, bar_time: datetime) -> None:
        et = _to_et(bar_time)
        today = et.strftime("%Y-%m-%d")
        if today != self._last_reset_date and et.hour >= 9 and et.minute >= 30:
            self.counters.reset()
            self.counters.trade_date = today
            self._last_reset_date = today
            self._throttle.daily_reset()
            self._throttle.update_equity(self.equity)
            # Increment session count for held positions
            if self._active and self._active.pos.qty_open > 0:
                self._active.pos.session_count += 1

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _bar_time(ts) -> datetime:
        """Convert numpy datetime64 or pd.Timestamp to timezone-aware datetime."""
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                return ts.replace(tzinfo=timezone.utc)
            return ts
        import pandas as pd
        if isinstance(ts, (pd.Timestamp, np.datetime64)):
            t = pd.Timestamp(ts)
            if t.tzinfo is None:
                t = t.tz_localize("UTC")
            return t.to_pydatetime()
        return datetime.now(timezone.utc)
