"""NQDTC v2.0 backtest engine — 5-minute primary loop.

Replicates the live NQDTC orchestration in synchronous, bar-by-bar mode.
Imports pure functions from strategies.momentum.nqdtc.

Primary feed: 5m bars. Higher TFs: 30m, 1H, 4H, Daily via idx maps.
Dual session state: independent ETH + RTH box/breakout/VWAP/chop state.
Shared state: RegimeState (4H + Daily), PositionState, DailyRiskState.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from backtests.momentum.analysis.nqdtc_shadow_tracker import NQDTCShadowTracker
from backtests.momentum.config import SlippageConfig, round_to_tick
from backtests.momentum.config_nqdtc import NQDTCAblationFlags, NQDTCBacktestConfig
from backtests.momentum.data.preprocessing import NumpyBars
from backtests.momentum.engine.sim_broker import (
    FillResult,
    FillStatus,
    OrderSide,
    OrderType,
    SimBroker,
    SimOrder,
)
from backtests.shared.parity.decision_capture import normalize_decision_stream
from backtests.shared.parity.execution_adapters import ParitySimOrder, neutral_action_to_sim_order
from backtests.shared.parity.replay_driver import ReplayStep, run_replay
from backtests.shared.parity.trade_outcomes import normalize_trade_outcome_stream

import copy as _copy

from strategies.core.actions import SubmitEntry
from strategies.momentum.nqdtc.core import logic as nqdtc_core_logic
from strategies.momentum.nqdtc.core.state import (
    NQDTCCoreState,
    NQDTCEntryFillContext,
    NQDTCEntryRequest,
    NQDTCFill,
    NQDTCOrderUpdate,
    NQDTCSimpleRequest,
)
from strategies.momentum.nqdtc import box as box_mod
from strategies.momentum.nqdtc import config as C
from strategies.momentum.nqdtc import indicators as ind
from strategies.momentum.nqdtc import signals as sig
from strategies.momentum.nqdtc import sizing
from strategies.momentum.nqdtc import stops
from strategies.momentum.nqdtc.models import (
    BoxEngineState,
    BoxState,
    BreakoutEngineState,
    ChopMode,
    CompositeRegime,
    DailyRiskState,
    Direction,
    EntrySubtype,
    ExitTier,
    PositionState,
    RegimeState,
    Regime4H,
    RollingBuffer,
    Session,
    SessionEngineState,
    TPLevel,
    VWAPAccumulator,
    WorkingOrder,
)

logger = logging.getLogger(__name__)


def _sim_order_from_parity(order: ParitySimOrder) -> SimOrder:
    return SimOrder(
        order_id=order.order_id,
        symbol=order.symbol,
        side=OrderSide[order.side.name],
        order_type=OrderType[order.order_type.name],
        qty=order.qty,
        stop_price=order.stop_price,
        limit_price=order.limit_price,
        tick_size=order.tick_size,
        submit_time=order.submit_time,
        ttl_hours=order.ttl_hours,
        ttl_minutes=order.ttl_minutes,
        tag=order.tag,
        oca_group=order.oca_group,
        invalidation_price=order.invalidation_price,
        triggered_ts=order.triggered_ts,
    )

# ---------------------------------------------------------------------------
# Snapshot of patchable C module values -- captured lazily on first use.
# _apply_param_overrides resets C to these originals before each patch to
# prevent state leakage between sequential engine instantiations.
# ---------------------------------------------------------------------------
_C_ORIGINALS: dict | None = None

_PATCHABLE_SCALAR_KEYS = [
    'SCORE_NORMAL', 'SCORE_DEGRADED', 'Q_DISP', 'Q_DISP_TIGHT_BOX',
    'Q_DISP_ALIGNED', 'STALE_BARS_NORMAL', 'STALE_BARS_DEGRADED',
    'STALE_R_THRESHOLD', 'DAILY_STOP_R', 'WEEKLY_STOP_R', 'MONTHLY_STOP_R',
    'BASE_RISK_PCT', 'RISK_PCT', 'CHOP_SIZE_MULT', 'FRICTION_CAP',
    'A_ENTRY_ENABLED', 'C_CONT_ENTRY_ENABLED',
    'WEAK_SCORE_BAND_FILTER_ENABLED', 'WEAK_SCORE_BAND_LOW',
    'WEAK_SCORE_BAND_HIGH', 'WEAK_SCORE_BAND_MAX_BOX_WIDTH',
    'WEAK_SCORE_BAND_MIN_RVOL', 'WIDE_BOX_SCORE_FILTER_ENABLED',
    'WIDE_BOX_MIN_WIDTH', 'WIDE_BOX_MIN_SCORE', 'WIDE_BOX_MIN_RVOL',
    'B_ALLOW_ALIGNED', 'B_ALLOW_RANGE', 'B_ALLOW_NEUTRAL',
    'B_ALLOW_CAUTION', 'B_MIN_DISP_Q',
    'A_STOP_ATR_MULT', 'C_CONT_MFE_GATE_R', 'C_ENTRY_OFFSET_ATR_STANDARD',
    'C_ENTRY_OFFSET_ATR_CONTINUATION', 'C_ENTRY_OFFSET_ATR', 'C_HOLD_BARS',
    'A_TTL_5M_BARS', 'A2_BUFFER_TICKS', 'A_CANCEL_DEPTH_ATR',
    'B_SWEEP_DEPTH_ATR', 'RESCUE_MAX_SLIP_ATR', 'C_CONT_PAUSE_ATR_MULT',
    'A_MAX_BOX_WIDTH', 'A_MIN_SCORE', 'A_BLOCK_WEAK_SCORE_BAND',
    'A_WEAK_SCORE_BAND_LOW', 'A_WEAK_SCORE_BAND_HIGH',
    'LOSS_STREAK_THRESHOLD', 'LOSS_STREAK_SKIP_BARS',
    'PROFIT_BE_R', 'MIN_INTER_TRADE_GAP_MINUTES',
    'ETH_SHORT_SIZE_MULT', 'MIN_BOX_WIDTH', 'MAX_BOX_WIDTH',
    'EARLY_BE_MFE_R', 'CONT_SIZE_MULT', 'REVERSAL_SIZE_MULT',
    'CONTINUATION_BREAKOUT_SIZE_MULT', 'MAX_STOP_ATR_MULT',
    'MAX_STOP_WIDTH_PTS', 'MAX_LOSS_CAP_R',
    'REENTRY_MIN_LOSS_R', 'REENTRY_COOLDOWN_MIN',
    'BLOCK_CONT_ALIGNED', 'BLOCK_STD_NEUTRAL_LOW_DISP',
    'BLOCK_NEUTRAL_REGIME', 'BLOCK_ALIGNED_REGIME', 'BLOCK_CAUTION_REGIME',
    'SCORE_NON_RANGE_MULT',
    'RVOL_SCORE_THRESH',
    'CHANDELIER_TIER0_MULT', 'CHANDELIER_TIER1_MULT', 'CHANDELIER_TIER2_MULT',
    'CHANDELIER_TIER3_MULT', 'CHANDELIER_TIER4_MULT',
    'CHANDELIER_GRACE_BARS_30M',
    'CHANDELIER_POST_TP1_MULT_DECAY', 'CHANDELIER_POST_TP1_FLOOR_MULT',
    'RATCHET_LOCK_PCT', 'RATCHET_THRESHOLD_R',
    'TP1_ONLY_CAP_MODE', 'MFE_RATCHET_TIERS_ENABLED',
    'MFE_RATCHET_T1_R', 'MFE_RATCHET_T1_LOCK_R',
    'MFE_RATCHET_T2_R', 'MFE_RATCHET_T2_LOCK_R',
    'MFE_RATCHET_T3_R', 'MFE_RATCHET_T3_LOCK_R',
    'BE_BUFFER_ATR_5M',
]


def _ensure_c_snapshot() -> dict:
    """Lazily capture original C values for all patchable keys."""
    global _C_ORIGINALS
    if _C_ORIGINALS is not None:
        return _C_ORIGINALS
    snap: dict = {}
    for key in _PATCHABLE_SCALAR_KEYS:
        if hasattr(C, key):
            snap[key] = getattr(C, key)
    snap['REGIME_MULT'] = dict(C.REGIME_MULT)
    snap['EXIT_TIERS'] = _copy.deepcopy(C.EXIT_TIERS)
    snap['CHANDELIER_TIERS'] = list(C.CHANDELIER_TIERS)
    _C_ORIGINALS = snap
    return _C_ORIGINALS


def _reset_c_to_originals() -> None:
    """Reset all patchable C module attributes to their original values."""
    snap = _ensure_c_snapshot()
    for key in _PATCHABLE_SCALAR_KEYS:
        if key in snap:
            setattr(C, key, snap[key])
    C.REGIME_MULT = dict(snap['REGIME_MULT'])
    C.EXIT_TIERS = _copy.deepcopy(snap['EXIT_TIERS'])
    C.CHANDELIER_TIERS = list(snap['CHANDELIER_TIERS'])


_ET = None


def _get_et():
    global _ET
    if _ET is None:
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/New_York")
    return _ET


def _to_ny(dt_utc: datetime) -> datetime:
    return dt_utc.astimezone(_get_et())


def _to_utc_aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _minutes(h: int, m: int) -> int:
    return h * 60 + m


# ---------------------------------------------------------------------------
# Session classification
# ---------------------------------------------------------------------------

def _classify_session(dt_utc: datetime) -> Session:
    """Classify bar into ETH or RTH by NY time."""
    ny = _to_ny(dt_utc)
    t = _minutes(ny.hour, ny.minute)
    if _minutes(C.RTH_START_H, C.RTH_START_M) <= t < _minutes(C.RTH_END_H, C.RTH_END_M):
        return Session.RTH
    return Session.ETH


def _in_entry_window(dt_utc: datetime, session: Session) -> bool:
    """Check if current time is within entry window for session."""
    ny = _to_ny(dt_utc)
    t = _minutes(ny.hour, ny.minute)
    if session == Session.ETH:
        return _minutes(C.ETH_ENTRY_START_H, C.ETH_ENTRY_START_M) <= t < _minutes(C.ETH_ENTRY_END_H, C.ETH_ENTRY_END_M)
    return _minutes(C.RTH_ENTRY_START_H, C.RTH_ENTRY_START_M) <= t < _minutes(C.RTH_ENTRY_END_H, C.RTH_ENTRY_END_M)


def _is_rth_close(dt_utc: datetime) -> bool:
    """Check if bar is at RTH close (16:15 ET)."""
    ny = _to_ny(dt_utc)
    return ny.hour == C.RTH_END_H and ny.minute == C.RTH_END_M


def _is_rth_open(dt_utc: datetime) -> bool:
    """Check if bar is at RTH open (09:30 ET)."""
    ny = _to_ny(dt_utc)
    return ny.hour == C.RTH_START_H and ny.minute == C.RTH_START_M


def _30m_bar_session(bar_close_utc: datetime) -> Session:
    """Assign 30m bar to session by its close time.

    Uses exclusive start: 09:30 bar (09:00-09:30) → ETH because its data is pre-RTH.
    First RTH bar is 10:00 (09:30-10:00).
    """
    ny = _to_ny(bar_close_utc)
    t = _minutes(ny.hour, ny.minute)
    if _minutes(C.RTH_START_H, C.RTH_START_M) < t <= _minutes(C.RTH_END_H, C.RTH_END_M):
        return Session.RTH
    return Session.ETH


# ---------------------------------------------------------------------------
# News calendar helpers
# ---------------------------------------------------------------------------

# Use config values for news blackout windows (matches live engine)
_NEWS_BLACKOUT_BEFORE = C.NEWS_BLACKOUT_WINDOW_BEFORE_MIN
_NEWS_BLACKOUT_AFTER = C.NEWS_BLACKOUT_WINDOW_AFTER_MIN


def _load_news_events(
    calendar_path: Path | None,
) -> tuple[list[tuple[datetime, datetime]], list[datetime]]:
    """Load news calendar.

    Returns:
        (blackout_windows, event_times_utc) where blackout_windows are
        (block_start, block_end) pairs and event_times_utc are the raw event
        timestamps used for the 15-min flatten check.
    """
    if calendar_path is None or not calendar_path.exists():
        return [], []
    try:
        raw = json.loads(calendar_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to load news calendar from %s", calendar_path)
        return [], []
    from zoneinfo import ZoneInfo
    et_tz = ZoneInfo("America/New_York")
    windows: list[tuple[datetime, datetime]] = []
    event_times: list[datetime] = []
    for entry in raw:
        time_str = entry.get("time_et", "")
        date_str = entry.get("date", "")
        if not time_str or not date_str:
            continue
        try:
            from datetime import date as _date, time as _time
            y, mo, d = (int(x) for x in date_str.split("-"))
            h, m = (int(x) for x in time_str.split(":"))
            event_utc = datetime.combine(
                _date(y, mo, d), _time(h, m), tzinfo=et_tz,
            ).astimezone(timezone.utc)
        except Exception:
            continue
        windows.append((
            event_utc - timedelta(minutes=_NEWS_BLACKOUT_BEFORE),
            event_utc + timedelta(minutes=_NEWS_BLACKOUT_AFTER),
        ))
        event_times.append(event_utc)
    if windows:
        logger.info("Loaded %d news blackout windows from %s", len(windows), calendar_path)
    return windows, event_times


def _in_news_blackout(bar_time: datetime, windows: list[tuple[datetime, datetime]]) -> bool:
    """Check if bar_time falls within any news blackout window."""
    for start, end in windows:
        if start <= bar_time < end:
            return True
    return False


def _news_flatten_imminent(bar_time: datetime, event_times: list[datetime]) -> bool:
    """Check if a news event is within the flatten lead time (15 min)."""
    lead_seconds = C.NEWS_FLATTEN_LEAD_MIN * 60
    for evt in event_times:
        delta = (evt - bar_time).total_seconds()
        if 0 <= delta <= lead_seconds:
            return True
    return False


# ---------------------------------------------------------------------------
# Signal event for gating attribution
# ---------------------------------------------------------------------------

@dataclass
class NQDTCSignalEvent:
    """Logged for every 30m breakout evaluation (pass or reject)."""
    timestamp: datetime
    session: str
    direction: int
    box_high: float
    box_low: float
    box_width: float
    close_30m: float
    displacement: float
    disp_threshold: float
    score: float
    score_threshold: float
    rvol: float
    chop_mode: str
    composite_regime: str
    # Gate results (True = passed, False = blocked)
    regime_pass: bool = True
    chop_pass: bool = True
    displacement_pass: bool = True
    quality_reject_pass: bool = True
    score_pass: bool = True
    news_pass: bool = True
    daily_stop_pass: bool = True
    weekly_stop_pass: bool = True
    monthly_stop_pass: bool = True
    friction_pass: bool = True
    micro_guard_pass: bool = True
    reentry_pass: bool = True
    # Outcome
    passed_all: bool = False
    first_block_reason: str = ""
    # Shadow trade reference
    would_be_entry: float = 0.0
    would_be_stop: float = 0.0


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class NQDTCTradeRecord:
    """Completed NQDTC trade."""
    symbol: str = "NQ"
    direction: int = 0
    entry_subtype: str = ""
    session: str = ""
    # Timing
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    bars_held_30m: int = 0
    # Prices
    entry_price: float = 0.0
    exit_price: float = 0.0
    initial_stop: float = 0.0
    qty: int = 0
    # PnL
    pnl_dollars: float = 0.0
    r_multiple: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    # Exit
    exit_reason: str = ""
    exit_tier: str = ""
    # Context at entry
    composite_regime: str = ""
    chop_mode: str = ""
    score_at_entry: float = 0.0
    displacement_at_entry: float = 0.0
    rvol_at_entry: float = 0.0
    quality_mult: float = 0.0
    expiry_mult: float = 0.0
    disp_norm_at_entry: float = 0.0
    # TP tracking
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    continuation: bool = False
    commission: float = 0.0
    # Box context
    box_width: float = 0.0
    adaptive_L: int = 0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class NQDTCSymbolResult:
    """Per-symbol NQDTC backtest output."""
    symbol: str = "NQ"
    trades: list[NQDTCTradeRecord] = field(default_factory=list)
    signal_events: list[NQDTCSignalEvent] = field(default_factory=list)
    decision_stream: list[dict[str, Any]] = field(default_factory=list)
    trade_outcomes: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    total_commission: float = 0.0
    # Counters
    breakouts_evaluated: int = 0
    breakouts_qualified: int = 0
    entries_placed: int = 0
    entries_filled: int = 0
    gates_blocked: int = 0
    shadow_summary: str = ""


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

@dataclass
class _ActivePosition:
    """Internal mutable position state tracking for the backtest."""
    pos: PositionState
    record: NQDTCTradeRecord
    mfe_price: float = 0.0
    mae_price: float = 0.0
    stop_order_id: str = ""
    tp_order_ids: list[str] = field(default_factory=list)
    session_state: SessionEngineState | None = None
    commission_at_start: float = 0.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class NQDTCEngine:
    """Single-symbol bar-by-bar NQDTC v2.0 backtest on 5-minute bars."""

    def __init__(
        self,
        symbol: str,
        bt_config: NQDTCBacktestConfig,
    ) -> None:
        self.symbol = symbol
        self.cfg = bt_config
        self.flags = bt_config.flags
        self.tick = bt_config.tick_size
        self.pv = bt_config.point_value

        # Simulated broker
        self.broker = SimBroker(slippage_config=bt_config.slippage)
        self._core_state = NQDTCCoreState(symbol=symbol)
        self._decision_events: list[Any] = []
        self._bar_count_5m = 0

        # News blackout windows + event times (for flatten/tighten)
        self._news_windows, self._news_event_times = _load_news_events(bt_config.news_calendar_path)

        # Account
        self.equity = bt_config.initial_equity

        # Dual session state (independent ETH + RTH)
        self.eth = SessionEngineState(session=Session.ETH)
        self.rth = SessionEngineState(session=Session.RTH)
        self._last_session: Optional[Session] = None

        # Shared state
        self.regime = RegimeState()
        self.daily_risk = DailyRiskState()

        # Drawdown throttle (daily cap disabled — NQDTC has its own via DailyRiskState)
        from libs.risk.drawdown_throttle import DrawdownThrottle, DrawdownThrottleConfig
        self._throttle = DrawdownThrottle(
            bt_config.initial_equity,
            DrawdownThrottleConfig(daily_loss_cap_r=None),
        )

        # Shadow tracker
        self.shadow_tracker: NQDTCShadowTracker | None = (
            NQDTCShadowTracker() if bt_config.track_shadows else None
        )

        # Active position (one at a time for NQ)
        self._active: _ActivePosition | None = None
        self._working: dict[str, WorkingOrder] = {}

        # Consecutive-loss cooldown state
        self._consec_losses: int = 0
        self._cooldown_bars: int = 0  # 5m bars remaining in cooldown (6 bars = 30 min)

        # Indicator caches (higher TF)
        self._ema50_4h: np.ndarray = np.array([])
        self._atr14_4h: np.ndarray = np.array([])
        self._adx14_4h: np.ndarray = np.array([])
        self._ema50_d: np.ndarray = np.array([])
        self._atr14_d: np.ndarray = np.array([])
        self._atr14_1h: np.ndarray = np.array([])
        self._highs_1h_cache: np.ndarray = np.array([])
        self._lows_1h_cache: np.ndarray = np.array([])

        # 5m indicator cache
        self._atr14_5m_val: float = 0.0

        # Session-filtered 30m bar indices (matches live engine's _bars_30m_session)
        self._30m_session_indices: dict[Session, list[int]] = {
            Session.ETH: [],
            Session.RTH: [],
        }

        # TF boundary tracking
        self._last_30m_idx = -1
        self._last_1h_idx = -1
        self._last_4h_idx = -1
        self._last_d_idx = -1
        self._last_reset_date = ""
        self._bar_idx_30m = 0
        self._total_commission = 0.0

        # 5m bar accumulators for entry evaluation
        self._5m_closes: list[float] = []
        self._5m_highs: list[float] = []
        self._5m_lows: list[float] = []
        self._5m_bar_count_since_breakout = 0
        self._a_fallback_eligible = False

        # Phase 1.1: 15m bar resampling from 5m for slope filter
        self._closes_15m: list[float] | None = [] if C.SLOPE_FILTER_ENABLED else None
        self._15m_bar_counter: int = 0
        self._15m_accum_close: float = 0.0

        # Result accumulators
        self._trades: list[NQDTCTradeRecord] = []
        self._signal_events: list[NQDTCSignalEvent] = []
        self._equity_history: list[float] = []
        self._time_history: list = []

        # Inter-trade cooldown (Prereq 1)
        self._last_fill_time: datetime | None = None

        # Early termination on drawdown (scoring mode)
        self._peak_equity = bt_config.initial_equity
        self._max_dd_abort = bt_config.max_dd_abort
        self._abort = False

        # Counters
        self._breakouts_evaluated = 0
        self._breakouts_qualified = 0
        self._entries_placed = 0
        self._entries_filled = 0
        self._gates_blocked = 0

        # Prereq 0: patch C module constants from param_overrides
        self._apply_param_overrides()

    # ------------------------------------------------------------------
    # Prereq 0: param_overrides -> C module patching
    # ------------------------------------------------------------------

    def _apply_param_overrides(self) -> None:
        """Patch C module constants from config.param_overrides.

        Always resets C to original defaults first to prevent state leakage
        between sequential engine instantiations in the same process.
        """
        _reset_c_to_originals()
        po = self.cfg.param_overrides
        if not po:
            return
        # Scalar constants
        for key in _PATCHABLE_SCALAR_KEYS:
            if key in po:
                setattr(C, key, po[key])
        # REGIME_MULT dict entries
        for regime in ['Aligned', 'Neutral', 'Caution', 'Range', 'Counter']:
            k = f'regime_mult_{regime.lower()}'
            if k in po:
                C.REGIME_MULT[regime] = po[k]
        # EXIT_TIERS (TP structure) -- rebuild schedule from individual params
        if {'TP1_R', 'TP1_PARTIAL_PCT', 'TP2_R', 'TP2_PARTIAL_PCT'} & set(po):
            schedule = [(po.get('TP1_R', C.TP1_R), po.get('TP1_PARTIAL_PCT', 0.55))]
            if 'TP2_R' in po:
                schedule.append((po['TP2_R'], po.get('TP2_PARTIAL_PCT', 0.25)))
            C.EXIT_TIERS = {tier: list(schedule) for tier in C.EXIT_TIERS}
        # CHANDELIER_TIERS -- per-tier or flat override of mult and lookback
        _tier_keys = ['CHANDELIER_TIER0_MULT', 'CHANDELIER_TIER1_MULT',
                      'CHANDELIER_TIER2_MULT', 'CHANDELIER_TIER3_MULT',
                      'CHANDELIER_TIER4_MULT']
        has_per_tier = any(po.get(k, 0.0) > 0 for k in _tier_keys)
        has_flat = 'CHANDELIER_ATR_MULT' in po
        has_lb = 'CHANDELIER_LOOKBACK' in po
        if has_per_tier or has_flat or has_lb:
            per_mults = [po.get(k, 0.0) for k in _tier_keys]
            new_tiers = []
            for i, (min_r, max_r, mm_req, lb, mult) in enumerate(C.CHANDELIER_TIERS):
                new_lb = int(po.get('CHANDELIER_LOOKBACK', lb))
                if i < len(per_mults) and per_mults[i] > 0:
                    new_mult = per_mults[i]
                elif has_flat:
                    new_mult = po['CHANDELIER_ATR_MULT']
                else:
                    new_mult = mult
                new_tiers.append((min_r, max_r, mm_req, new_lb, new_mult))
            C.CHANDELIER_TIERS = new_tiers

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        five_min_bars: NumpyBars,
        thirty_min: NumpyBars,
        hourly: NumpyBars,
        four_hour: NumpyBars,
        daily: NumpyBars,
        thirty_min_idx_map: np.ndarray,
        hourly_idx_map: np.ndarray,
        four_hour_idx_map: np.ndarray,
        daily_idx_map: np.ndarray,
        daily_es: Optional[NumpyBars] = None,
        daily_es_idx_map: Optional[np.ndarray] = None,
    ) -> NQDTCSymbolResult:
        """Run the full backtest. Returns NQDTCSymbolResult."""
        n = len(five_min_bars)
        wu_d = self.cfg.warmup_daily
        wu_30m = self.cfg.warmup_30m
        wu_1h = self.cfg.warmup_1h
        wu_4h = self.cfg.warmup_4h

        # ES daily data for cross-strategy regime
        self._daily_es = daily_es
        self._daily_es_idx_map = daily_es_idx_map
        self._last_es_d_idx = -1
        if daily_es is not None and len(daily_es) > C.ES_DAILY_SMA_PERIOD:
            from strategies.momentum.vdub.indicators import sma
            self._es_sma200_pre = sma(daily_es.closes[:len(daily_es)], C.ES_DAILY_SMA_PERIOD)
        else:
            self._es_sma200_pre = np.array([])

        # -- Pre-compute higher-TF indicators once (O(n) total, not per-boundary) --
        n_d = len(daily)
        self._ema50_d_pre = ind.ema(daily.closes[:n_d], C.EMA50_PERIOD) if n_d > C.EMA50_PERIOD else np.array([])
        self._atr14_d_pre = ind.atr(daily.highs[:n_d], daily.lows[:n_d], daily.closes[:n_d], C.ATR14_PERIOD) if n_d > C.ATR14_PERIOD else np.array([])

        n_4h = len(four_hour)
        self._ema50_4h_pre = ind.ema(four_hour.closes[:n_4h], C.EMA50_PERIOD) if n_4h > C.EMA50_PERIOD else np.array([])
        self._atr14_4h_pre = ind.atr(four_hour.highs[:n_4h], four_hour.lows[:n_4h], four_hour.closes[:n_4h], C.ATR14_PERIOD) if n_4h > C.ATR14_PERIOD else np.array([])
        if n_4h > C.ADX_PERIOD * 2:
            self._adx14_4h_pre, _, _ = ind.adx(four_hour.highs[:n_4h], four_hour.lows[:n_4h], four_hour.closes[:n_4h], C.ADX_PERIOD)
        else:
            self._adx14_4h_pre = np.array([])

        n_1h = len(hourly)
        self._atr14_1h_pre = ind.atr(hourly.highs[:n_1h], hourly.lows[:n_1h], hourly.closes[:n_1h], C.ATR14_1H_PERIOD) if n_1h > C.ATR14_1H_PERIOD else np.array([])

        # -- Incremental 5m ATR --
        self._inc_atr_5m = ind.IncrementalATR(n, C.ATR14_5M_PERIOD)

        # -- Per-session incremental 30m ATR (fed at each 30m boundary) --
        n_30m = len(thirty_min)
        self._inc_atr14_30m = {
            Session.ETH: ind.IncrementalATR(n_30m, C.ATR14_PERIOD),
            Session.RTH: ind.IncrementalATR(n_30m, C.ATR14_PERIOD),
        }
        self._inc_atr50_30m = {
            Session.ETH: ind.IncrementalATR(n_30m, C.ATR50_PERIOD),
            Session.RTH: ind.IncrementalATR(n_30m, C.ATR50_PERIOD),
        }
        self._inc_30m_count: dict[Session, int] = {Session.ETH: 0, Session.RTH: 0}

        # -- Pre-compute bar times vectorized (avoids 93K+ per-bar conversions) --
        _bar_times_idx = pd.DatetimeIndex(five_min_bars.times)
        if _bar_times_idx.tz is None:
            _bar_times_idx = _bar_times_idx.tz_localize("UTC")
        self._bar_times = _bar_times_idx.to_pydatetime()
        end_dt = _to_utc_aware(self.cfg.end_date)
        last_processed_t: int | None = None

        # -- Pre-compute NY times, sessions, daily dates, entry windows --
        _et = _get_et()
        _ny_times = _bar_times_idx.tz_convert(_et)
        _ny_total_minutes = (_ny_times.hour * 60 + _ny_times.minute).values
        _ny_hours = _ny_times.hour.values
        _ny_minutes = _ny_times.minute.values

        _rth_start_m = _minutes(C.RTH_START_H, C.RTH_START_M)
        _rth_end_m = _minutes(C.RTH_END_H, C.RTH_END_M)
        self._sessions_rth_arr = (_ny_total_minutes >= _rth_start_m) & (_ny_total_minutes < _rth_end_m)

        self._daily_dates_arr = _ny_times.strftime("%Y-%m-%d").values
        self._ny_hours_arr = _ny_hours
        self._ny_weekdays_arr = _ny_times.weekday.values
        self._ny_days_arr = _ny_times.day.values

        self._is_rth_open_arr = (_ny_hours == C.RTH_START_H) & (_ny_minutes == C.RTH_START_M)
        self._is_eth_start_arr = (_ny_hours == C.ETH_START_H) & (_ny_minutes == C.ETH_START_M)

        self._in_eth_entry_arr = (
            (_ny_total_minutes >= _minutes(C.ETH_ENTRY_START_H, C.ETH_ENTRY_START_M))
            & (_ny_total_minutes < _minutes(C.ETH_ENTRY_END_H, C.ETH_ENTRY_END_M))
        )
        self._in_rth_entry_arr = (
            (_ny_total_minutes >= _minutes(C.RTH_ENTRY_START_H, C.RTH_ENTRY_START_M))
            & (_ny_total_minutes < _minutes(C.RTH_ENTRY_END_H, C.RTH_ENTRY_END_M))
        )

        for t in range(n):
            bar_time = self._bar_times[t]
            if end_dt is not None and bar_time >= end_dt:
                break
            O = five_min_bars.opens[t]
            H = five_min_bars.highs[t]
            L = five_min_bars.lows[t]
            Cl = five_min_bars.closes[t]
            V = five_min_bars.volumes[t]

            if np.isnan(O) or np.isnan(Cl):
                continue

            m30_idx = int(thirty_min_idx_map[t])
            h_idx = int(hourly_idx_map[t])
            fh_idx = int(four_hour_idx_map[t])
            d_idx = int(daily_idx_map[t])
            es_d_idx = int(daily_es_idx_map[t]) if daily_es_idx_map is not None else -1

            self._step_5m(
                t, bar_time, O, H, L, Cl, V,
                thirty_min, hourly, four_hour, daily,
                m30_idx, h_idx, fh_idx, d_idx, es_d_idx,
                wu_d, wu_30m, wu_1h, wu_4h,
            )
            last_processed_t = t
            if self._abort:
                break

        # Close any remaining position at last bar
        if self._active and self._active.pos.open:
            final_t = last_processed_t if last_processed_t is not None else n - 1
            last_close = float(five_min_bars.closes[final_t])
            last_time = self._bar_times[final_t]
            self._close_position_market(last_close, last_time, "END_OF_DATA")

        # Shadow simulation
        shadow_summary = ""
        if self.shadow_tracker is not None and self.shadow_tracker.rejections:
            self.shadow_tracker.simulate_shadows(
                five_min_bars.opens, five_min_bars.highs,
                five_min_bars.lows, five_min_bars.closes,
                five_min_bars.times,
            )
            shadow_summary = self.shadow_tracker.format_summary()

        return NQDTCSymbolResult(
            symbol=self.symbol,
            trades=self._trades,
            signal_events=self._signal_events,
            decision_stream=normalize_decision_stream(self._decision_events) if not self.cfg.scoring_mode else [],
            trade_outcomes=normalize_trade_outcome_stream(self._trades) if not self.cfg.scoring_mode else [],
            equity_curve=np.array(self._equity_history),
            timestamps=np.array(self._time_history),
            total_commission=self._total_commission,
            breakouts_evaluated=self._breakouts_evaluated,
            breakouts_qualified=self._breakouts_qualified,
            entries_placed=self._entries_placed,
            entries_filled=self._entries_filled,
            gates_blocked=self._gates_blocked,
            shadow_summary=shadow_summary,
        )

    def _replay_core_step(
        self,
        *,
        bar_input: dict[str, Any] | None = None,
        order_updates: list[NQDTCOrderUpdate] | None = None,
        fills: list[NQDTCFill] | None = None,
    ):
        result = run_replay(
            self._core_state,
            steps=[
                ReplayStep(
                    bar_input=bar_input,
                    order_updates=order_updates or [],
                    fills=fills or [],
                )
            ],
            on_bar=lambda state, payload: nqdtc_core_logic.on_bar(state, **payload),
            on_order_update=nqdtc_core_logic.on_order_update,
            on_fill=nqdtc_core_logic.on_fill,
        )
        self._core_state = result.state
        self._decision_events.extend(result.events)
        return result

    # ------------------------------------------------------------------
    # Per-5m-bar step
    # ------------------------------------------------------------------

    def _step_5m(
        self,
        t: int,
        bar_time: datetime,
        O: float, H: float, L: float, Cl: float, V: float,
        thirty_min: NumpyBars, hourly: NumpyBars,
        four_hour: NumpyBars, daily: NumpyBars,
        m30_idx: int, h_idx: int, fh_idx: int, d_idx: int, es_d_idx: int,
        wu_d: int, wu_30m: int, wu_1h: int, wu_4h: int,
    ) -> None:
        self._bar_count_5m += 1

        # 1. Classify session (pre-computed array lookup)
        session = Session.RTH if self._sessions_rth_arr[t] else Session.ETH
        sess_state = self.eth if session == Session.ETH else self.rth

        # 2. Daily reset (pre-computed array lookup)
        self._check_daily_reset(t)

        # 2a. Cooldown decrement on every 5m bar (matches live engine)
        if self.flags.loss_streak_cooldown and self._cooldown_bars > 0:
            self._cooldown_bars -= 1

        # 2b. Session boundary reset (Section 1.4: fresh state each session)
        if self._last_session is not None and self._last_session != session:
            opposite = self.eth if session == Session.RTH else self.rth
            opposite.box = BoxEngineState()
            opposite.breakout = BreakoutEngineState()
            opposite.vwap_session.reset()
            opposite.vwap_box.reset()
            opposite.chop_score = 0
            opposite.mode = ChopMode.NORMAL
            opposite.atr14_30m = 0.0
            opposite.atr50_30m = 0.0
            opposite.last_score = 0.0
            opposite.last_disp_metric = 0.0
            opposite.last_disp_threshold = 0.0
            opposite.last_rvol = 0.0
            opposite.reentry_allowed = True
            opposite.reentry_used = False
            opposite.last_stopout_r = 0.0
            opposite.last_stopout_ts = None
            opposite.last_30m_bar_count = 0
            opposite.last_profitable_exit_dir = Direction.FLAT
        self._last_session = session

        # 3. Session VWAP reset at session open (pre-computed array lookup)
        if self._is_rth_open_arr[t]:
            self.rth.vwap_session.reset(bar_time)
        # ETH resets at 18:00 NY (start of next trading day)
        if self._is_eth_start_arr[t]:
            self.eth.vwap_session.reset(bar_time)

        # 4. Update session VWAP
        sess_state.vwap_session.update(H, L, Cl, V)

        # 5. Process broker fills
        fills = self.broker.process_bar(self.symbol, bar_time, O, H, L, Cl, self.tick)
        for fill in fills:
            self._handle_fill(fill, bar_time, Cl)

        # 6. MFE/MAE update
        if self._active and self._active.pos.open:
            self._update_mfe_mae(H, L)

        # 6b. News flatten/tighten moved to step 10c (after position management, matches live)

        # 7. Update 5m running caches
        self._5m_closes.append(Cl)
        self._5m_highs.append(H)
        self._5m_lows.append(L)
        # Keep bounded
        if len(self._5m_closes) > 500:
            self._5m_closes = self._5m_closes[-500:]
            self._5m_highs = self._5m_highs[-500:]
            self._5m_lows = self._5m_lows[-500:]

        # 7b. Phase 1.1: resample 5m close → 15m close for slope filter
        if self._closes_15m is not None:
            self._15m_bar_counter += 1
            self._15m_accum_close = Cl  # use last 5m close as 15m close
            if self._15m_bar_counter >= 3:
                self._closes_15m.append(self._15m_accum_close)
                self._15m_bar_counter = 0
                if len(self._closes_15m) > 200:
                    self._closes_15m = self._closes_15m[-200:]

        # 8. Compute 5m ATR — O(1) incremental update
        self._inc_atr_5m.update(t, H, L, Cl)
        val = self._inc_atr_5m.values[t]
        if not np.isnan(val):
            self._atr14_5m_val = float(val)

        # 9. Higher-TF boundary processing
        new_d = d_idx != self._last_d_idx and d_idx >= wu_d
        new_4h = fh_idx != self._last_4h_idx and fh_idx >= wu_4h
        new_1h = h_idx != self._last_1h_idx and h_idx >= wu_1h
        new_30m = m30_idx != self._last_30m_idx and m30_idx >= wu_30m

        if new_d:
            self._on_daily_boundary(daily, d_idx)
            self._last_d_idx = d_idx

        # ES daily boundary (cross-strategy regime)
        if (es_d_idx >= 0 and es_d_idx != self._last_es_d_idx
                and self.flags.es_daily_trend
                and len(self._es_sma200_pre) > 0):
            self._on_es_daily_boundary(es_d_idx)
            self._last_es_d_idx = es_d_idx

        if new_4h:
            self._on_4h_boundary(four_hour, fh_idx, bar_time)
            self._last_4h_idx = fh_idx

        if new_1h:
            self._on_1h_boundary(hourly, h_idx)
            self._last_1h_idx = h_idx

        if new_30m:
            self._on_30m_boundary(thirty_min, m30_idx, bar_time, sess_state)
            self._last_30m_idx = m30_idx
            self._bar_idx_30m += 1

        # 10. Position management on 30m boundary
        if new_30m and self._active and self._active.pos.open:
            self._manage_position_30m(bar_time, Cl)

        # 10a. Increment bars_since_entry AFTER position management (matches live engine)
        if new_30m and self._active and self._active.pos.open:
            self._active.pos.bars_since_entry_30m += 1
            if self._active.pos.bars_since_tp1 >= 0:
                self._active.pos.bars_since_tp1 += 1

        # 10b. Profit-funded BE stop update (every 5m bar, matching live engine)
        if (self._active and self._active.pos.open
                and self._active.pos.profit_funded
                and not self._active.pos.runner_active
                and self._atr14_5m_val > 0):
            pos = self._active.pos
            be_stop = stops.compute_be_stop(
                pos.direction, pos.entry_price, self._atr14_5m_val, self.tick,
            )
            if pos.direction == Direction.LONG and be_stop > pos.stop_price:
                self._update_stop_price(be_stop, bar_time)
            elif pos.direction == Direction.SHORT and be_stop < pos.stop_price:
                self._update_stop_price(be_stop, bar_time)

        # 10c. News flatten/tighten (Section 4 — 15 min before event, after position management)
        if (self.flags.news_blackout
                and self._active and self._active.pos.open
                and _news_flatten_imminent(bar_time, self._news_event_times)):
            pos = self._active.pos
            if not pos.profit_funded:
                self._close_position_market(Cl, bar_time, "NEWS_FLATTEN")
            else:
                if pos.direction == Direction.LONG:
                    be_tick = pos.entry_price + self.tick
                else:
                    be_tick = pos.entry_price - self.tick
                if ((pos.direction == Direction.LONG and be_tick > pos.stop_price)
                        or (pos.direction == Direction.SHORT and be_tick < pos.stop_price)):
                    self._update_stop_price(be_tick, bar_time)

        # 11. 5m entry evaluation (current session only, matches live engine)
        if not (self._active and self._active.pos.open):
            if not self.flags.rth_entries and session == Session.RTH:
                pass  # RTH entries disabled
            elif sess_state.breakout.active and (self._in_rth_entry_arr[t] if session == Session.RTH else self._in_eth_entry_arr[t]):
                self._evaluate_5m_entry(bar_time, O, H, L, Cl, sess_state)

        # 12. Check working order cancellation (A cancel depth)
        self._check_working_order_cancellation(bar_time, Cl)

        # 13. Equity snapshot (every 30m, mark-to-market)
        if new_30m or len(self._equity_history) == 0:
            mtm = self.equity
            if self._active and self._active.pos.open:
                p = self._active.pos
                unrealized = (Cl - p.entry_price) * p.direction * self.pv * p.qty_open
                mtm += unrealized
            if mtm > self._peak_equity:
                self._peak_equity = mtm
            self._equity_history.append(mtm)
            self._time_history.append(bar_time)
            # Early termination on excessive drawdown
            if self._max_dd_abort > 0 and self._peak_equity > 0:
                dd = (self._peak_equity - mtm) / self._peak_equity
                if dd > self._max_dd_abort:
                    self._abort = True

    # ------------------------------------------------------------------
    # Higher-TF boundary handlers
    # ------------------------------------------------------------------

    def _on_daily_boundary(self, daily: NumpyBars, d_idx: int) -> None:
        if len(self._ema50_d_pre) > 0:
            self._ema50_d = self._ema50_d_pre[:d_idx + 1]
        if len(self._atr14_d_pre) > 0:
            self._atr14_d = self._atr14_d_pre[:d_idx + 1]

    def _on_es_daily_boundary(self, es_d_idx: int) -> None:
        """Update ES SMA200 daily trend with 2-bar persistence (same as Vdubus)."""
        if es_d_idx < 0 or es_d_idx >= len(self._es_sma200_pre):
            return
        sma_val = self._es_sma200_pre[es_d_idx]
        if np.isnan(sma_val):
            return
        es_close = float(self._daily_es.closes[es_d_idx])
        raw = 1 if es_close > sma_val else -1

        # 2-bar persistence logic (matches strategy_3/regime.py:update_persisted_trend)
        r = self.regime
        if raw == r.last_es_daily_raw:
            r.es_daily_raw_streak += 1
        else:
            r.es_daily_raw_streak = 1
        r.last_es_daily_raw = raw
        if r.es_daily_raw_streak >= C.ES_TREND_PERSIST_BARS and raw != r.es_daily_trend:
            r.es_daily_trend = raw

    def _es_opposes_direction(self, direction: Direction) -> bool:
        """Check if trade direction opposes ES SMA200 daily trend."""
        if not self.flags.es_daily_trend or self.regime.es_daily_trend == 0:
            return False
        if direction == Direction.LONG and self.regime.es_daily_trend == -1:
            return True
        if direction == Direction.SHORT and self.regime.es_daily_trend == 1:
            return True
        return False

    def _on_4h_boundary(self, four_hour: NumpyBars, fh_idx: int, bar_time: datetime) -> None:
        if len(self._ema50_4h_pre) > 0:
            self._ema50_4h = self._ema50_4h_pre[:fh_idx + 1]
        if len(self._atr14_4h_pre) > 0:
            self._atr14_4h = self._atr14_4h_pre[:fh_idx + 1]
        if len(self._adx14_4h_pre) > 0:
            self._adx14_4h = self._adx14_4h_pre[:fh_idx + 1]

        # Classify 4H regime
        if len(self._ema50_4h) > 3 and len(self._atr14_4h) > 0 and len(self._adx14_4h) > 0:
            regime_str, trend_dir, slope, adx_val = sig.classify_4h(
                self._ema50_4h, self._atr14_4h, self._adx14_4h,
            )
            self.regime.regime_4h = Regime4H(regime_str)
            self.regime.trend_dir_4h = trend_dir
            self.regime.slope_4h = slope
            self.regime.adx_4h = adx_val

    def _on_1h_boundary(self, hourly: NumpyBars, h_idx: int) -> None:
        if len(self._atr14_1h_pre) > 0:
            self._atr14_1h = self._atr14_1h_pre[:h_idx + 1]
        self._highs_1h_cache = hourly.highs[:h_idx + 1]
        self._lows_1h_cache = hourly.lows[:h_idx + 1]

    def _on_30m_boundary(
        self,
        thirty_min: NumpyBars,
        m30_idx: int,
        bar_time: datetime,
        active_sess: SessionEngineState,
    ) -> None:
        """Process new 30m bar: box update, breakout detection, qualification gates."""
        if m30_idx < 0:
            return

        # Determine which session this 30m bar belongs to
        bar_close_time = self._bar_time(thirty_min.times[m30_idx])
        bar_session = _30m_bar_session(bar_close_time)
        sess = self.eth if bar_session == Session.ETH else self.rth

        # Track session membership and build session-filtered arrays
        # (matches live engine's _bars_30m_session — fix #1)
        self._30m_session_indices[bar_session].append(m30_idx)
        sess_idx = np.array(self._30m_session_indices[bar_session], dtype=np.intp)
        closes = thirty_min.closes[sess_idx]
        highs = thirty_min.highs[sess_idx]
        lows = thirty_min.lows[sess_idx]
        opens = thirty_min.opens[sess_idx]
        volumes = thirty_min.volumes[sess_idx]

        # Incremental 30m ATR — O(1) update (must run every bar for state continuity)
        sc = self._inc_30m_count[bar_session]
        bar_H = float(thirty_min.highs[m30_idx])
        bar_L = float(thirty_min.lows[m30_idx])
        bar_C = float(thirty_min.closes[m30_idx])
        self._inc_atr14_30m[bar_session].update(sc, bar_H, bar_L, bar_C)
        self._inc_atr50_30m[bar_session].update(sc, bar_H, bar_L, bar_C)
        self._inc_30m_count[bar_session] = sc + 1

        if len(closes) < 20:
            return

        # Read latest ATR values from incremental instances
        val14 = self._inc_atr14_30m[bar_session].values[sc]
        if not np.isnan(val14):
            sess.atr14_30m = float(val14)
        val50 = self._inc_atr50_30m[bar_session].values[sc]
        if not np.isnan(val50):
            sess.atr50_30m = float(val50)

        # Update box-anchored VWAP
        if sess.box.state != BoxState.INACTIVE:
            sess.vwap_box.update(
                float(thirty_min.highs[m30_idx]), float(thirty_min.lows[m30_idx]),
                float(thirty_min.closes[m30_idx]), float(thirty_min.volumes[m30_idx]),
            )

        # Update box state (session-filtered bars)
        box_mod.update_box_state(
            sess.box, sess.breakout,
            highs, lows, closes,
            sess.atr14_30m, sess.atr50_30m,
            bar_time, sess.vwap_box,
            sess.squeeze_hist.data,
            volumes_30m=volumes,
        )
        # Ablation: suppress DIRTY state when flag disabled
        if not self.flags.dirty_mechanism and sess.box.state == BoxState.DIRTY:
            sess.box.state = BoxState.ACTIVE

        # Squeeze metric (past-only: append AFTER use in box update)
        if sess.box.state == BoxState.ACTIVE and sess.atr14_30m > 0:
            sq = ind.squeeze_metric(sess.box.box_width, sess.atr14_30m)
            sess.squeeze_hist.append(sq)

        # Update breakout state
        if sess.breakout.active:
            # Check regime hard block
            hard_blocked = sig.regime_hard_block(
                self.regime.regime_4h.value,
                self.regime.trend_dir_4h,
                sess.breakout.direction,
                self.regime.daily_opposes,
            )
            sig.update_breakout_state(
                sess.breakout,
                float(closes[-1]),
                sess.atr14_30m,
                sess.box.box_high, sess.box.box_low,
                regime_hard_blocked=hard_blocked,
            )

        # CHOP score (min 60 bars to match live engine)
        if len(closes) >= 60:
            cnt = self._inc_30m_count[bar_session]
            atr_pctl = ind.percentile_rank(
                sess.atr14_30m,
                self._inc_atr14_30m[bar_session].values[:cnt],
            )
            vwap_arr = ind.session_vwap(highs, lows, closes, volumes, max(0, len(closes) - 100))
            cross_cnt = ind.vwap_cross_count(closes, vwap_arr, C.CHOP_VWAP_CROSS_LB)
            sess.chop_score = sig.compute_chop_score(atr_pctl, cross_cnt)
            sess.mode = sig.chop_mode(sess.chop_score)

        # NOTE: bars_since_entry_30m increment moved to _step_5m after _manage_position_30m

        # Try breakout detection on this 30m bar (session-filtered bars)
        if sess.box.state == BoxState.ACTIVE and not sess.breakout.active:
            self._try_breakout(sess, closes, highs, lows, opens, volumes, bar_time, m30_idx)

    # ------------------------------------------------------------------
    # Breakout detection and qualification
    # ------------------------------------------------------------------

    def _try_breakout(
        self,
        sess: SessionEngineState,
        closes_30m: np.ndarray,
        highs_30m: np.ndarray,
        lows_30m: np.ndarray,
        opens_30m: np.ndarray,
        volumes_30m: np.ndarray,
        bar_time: datetime,
        m30_idx: int,
    ) -> None:
        """Check for structural breakout and run qualification gates."""
        close = float(closes_30m[-1])
        direction = sig.breakout_structural(close, sess.box.box_high, sess.box.box_low)
        if direction is None:
            return

        self._breakouts_evaluated += 1

        # Build signal event for gating attribution
        evt = NQDTCSignalEvent(
            timestamp=bar_time,
            session=sess.session.value,
            direction=int(direction),
            box_high=sess.box.box_high,
            box_low=sess.box.box_low,
            box_width=sess.box.box_width,
            close_30m=close,
            displacement=0.0,
            disp_threshold=0.0,
            score=0.0,
            score_threshold=0.0,
            rvol=0.0,
            chop_mode=sess.mode.value,
            composite_regime=self.regime.composite.value,
        )

        # Pre-compute approximate entry/stop for shadow tracking
        _box_mid = sess.box.box_mid
        if direction == Direction.LONG:
            evt.would_be_entry = sess.box.box_high
            evt.would_be_stop = _box_mid - 0.10 * sess.atr14_30m
        else:
            evt.would_be_entry = sess.box.box_low
            evt.would_be_stop = _box_mid + 0.10 * sess.atr14_30m

        # --- Gate evaluation (deterministic order) ---

        # 0. Pre-compute daily support + composite regime (needed for gate 1b)
        if len(self._ema50_d) > 3 and len(self._atr14_d) > 0:
            _ds, _do = sig.classify_daily_support(
                self._ema50_d, self._atr14_d, direction,
            )
            self.regime.daily_supports = _ds
            self.regime.daily_opposes = _do
        self.regime.composite = sig.compute_composite_regime(
            self.regime.regime_4h.value, self.regime.trend_dir_4h,
            direction, self.regime.daily_supports, self.regime.daily_opposes,
        )
        evt.composite_regime = self.regime.composite.value

        # 1. Regime hard block
        hard_blocked = sig.regime_hard_block(
            self.regime.regime_4h.value,
            self.regime.trend_dir_4h,
            direction,
            self.regime.daily_opposes,
        )
        if hard_blocked:
            evt.regime_pass = False
            evt.first_block_reason = "regime_hard_block"
            self._record_signal_event(evt, sess, direction, bar_time)
            return

        # 1b. Composite regime block (post-audit regime filtering)
        if C.BLOCK_NEUTRAL_REGIME or C.BLOCK_ALIGNED_REGIME or C.BLOCK_CAUTION_REGIME:
            _comp = self.regime.composite
            if (C.BLOCK_NEUTRAL_REGIME and _comp == CompositeRegime.NEUTRAL) or \
               (C.BLOCK_ALIGNED_REGIME and _comp == CompositeRegime.ALIGNED) or \
               (C.BLOCK_CAUTION_REGIME and _comp == CompositeRegime.CAUTION):
                evt.regime_pass = False
                evt.first_block_reason = f"regime_{_comp.value.lower()}_block"
                self._record_signal_event(evt, sess, direction, bar_time)
                return

        # 2. CHOP halt
        if self.flags.chop_halt and sess.mode == ChopMode.HALT:
            evt.chop_pass = False
            evt.first_block_reason = "chop_halt"
            self._record_signal_event(evt, sess, direction, bar_time)
            return

        # 3. Displacement (past-only, context-adaptive)
        vwap_box_val = sess.vwap_box.value
        atr_expanding = sess.atr14_30m > sess.atr50_30m if sess.atr50_30m > 0 else False
        # Pre-compute squeeze context for adaptive displacement threshold
        _sq_arr = np.array(sess.squeeze_hist.data) if sess.squeeze_hist.data else np.array([1.0])
        _current_sq = ind.squeeze_metric(sess.box.box_width, sess.atr14_30m) if sess.atr14_30m > 0 else 1.0
        _squeeze_good_disp = _current_sq <= float(np.quantile(_sq_arr, C.SQUEEZE_GOOD_QUANTILE)) if len(_sq_arr) > 5 else False
        _regime_aligned = (self.regime.composite == CompositeRegime.ALIGNED)
        disp_metric, disp_threshold, disp_passed = sig.displacement_pass(
            close, vwap_box_val, sess.atr14_30m,
            sess.disp_hist.data,
            atr_expanding=atr_expanding,
            squeeze_good=_squeeze_good_disp,
            regime_aligned=_regime_aligned,
        )
        # Past-only: append observation AFTER comparison
        sess.disp_hist.append(disp_metric)
        evt.displacement = disp_metric
        evt.disp_threshold = disp_threshold
        sess.last_disp_metric = disp_metric
        sess.last_disp_threshold = disp_threshold

        if self.flags.displacement_threshold and not disp_passed:
            evt.displacement_pass = False
            evt.first_block_reason = "displacement"
            self._record_signal_event(evt, sess, direction, bar_time)
            return

        # 4. Breakout quality reject
        bar_h = float(highs_30m[-1])
        bar_l = float(lows_30m[-1])
        bar_o = float(opens_30m[-1])
        median_vol = float(np.median(volumes_30m[-60:])) if len(volumes_30m) >= 60 else float(np.median(volumes_30m))
        rvol = ind.compute_rvol(float(volumes_30m[-1]), median_vol)
        evt.rvol = rvol
        sess.last_rvol = rvol

        rejected, body_decisive = sig.breakout_quality_reject(
            bar_h, bar_l, bar_o, close, sess.atr14_30m, rvol, direction,
        )
        if self.flags.breakout_quality_reject and rejected:
            evt.quality_reject_pass = False
            evt.first_block_reason = "breakout_quality_reject"
            self._record_signal_event(evt, sess, direction, bar_time)
            return

        # 5. Evidence score
        # Build score inputs
        two_outside = False
        if len(closes_30m) >= 2:
            prev = float(closes_30m[-2])
            two_outside = (
                (close > sess.box.box_high and prev > sess.box.box_high) or
                (close < sess.box.box_low and prev < sess.box.box_low)
            )
        atr_rising = sess.atr14_30m > sess.atr50_30m if sess.atr50_30m > 0 else False

        # Squeeze check
        sq_arr = np.array(sess.squeeze_hist.data) if sess.squeeze_hist.data else np.array([1.0])
        current_sq = ind.squeeze_metric(sess.box.box_width, sess.atr14_30m) if sess.atr14_30m > 0 else 1.0
        squeeze_good = current_sq <= float(np.quantile(sq_arr, C.SQUEEZE_GOOD_QUANTILE)) if len(sq_arr) > 5 else False
        squeeze_loose = current_sq >= float(np.quantile(sq_arr, C.SQUEEZE_LOOSE_QUANTILE)) if len(sq_arr) > 5 else False

        # Daily support + composite regime (already computed at gate entry, step 0)
        daily_supports = self.regime.daily_supports
        daily_opposes = self.regime.daily_opposes
        composite = self.regime.composite

        score = sig.compute_score(
            rvol, two_outside, atr_rising, squeeze_good, squeeze_loose,
            self.regime.regime_4h.value, self.regime.trend_dir_4h,
            direction, daily_supports, body_decisive,
        )
        evt.score = score
        s_threshold = sig.score_threshold(sess.mode)
        # Post-audit: raise score threshold for non-Range regimes
        if composite != CompositeRegime.RANGE and C.SCORE_NON_RANGE_MULT != 1.0:
            s_threshold *= C.SCORE_NON_RANGE_MULT
        evt.score_threshold = s_threshold
        sess.last_score = score

        if self.flags.score_threshold and score < s_threshold:
            evt.score_pass = False
            evt.first_block_reason = "score"
            self._record_signal_event(evt, sess, direction, bar_time)
            return

        context_ok, context_reason = sig.contextual_score_filter_pass(
            score=score,
            box_width=sess.box.box_width,
            rvol=rvol,
        )
        if self.flags.score_threshold and not context_ok:
            evt.score_pass = False
            evt.first_block_reason = context_reason
            self._record_signal_event(evt, sess, direction, bar_time)
            return

        # 5b. Box width filter (Prereq 3)
        min_bw = getattr(C, 'MIN_BOX_WIDTH', 0)
        max_bw = getattr(C, 'MAX_BOX_WIDTH', 99999)
        if sess.box.box_width < min_bw or sess.box.box_width > max_bw:
            evt.first_block_reason = "box_width_filter"
            self._record_signal_event(evt, sess, direction, bar_time)
            return

        # 6. News blackout
        if self.flags.news_blackout and _in_news_blackout(bar_time, self._news_windows):
            evt.news_pass = False
            evt.first_block_reason = "news_blackout"
            self._record_signal_event(evt, sess, direction, bar_time)
            return

        # 7. Daily stop
        if self.flags.daily_stop and self.daily_risk.halted:
            evt.daily_stop_pass = False
            evt.first_block_reason = "daily_stop"
            self._record_signal_event(evt, sess, direction, bar_time)
            return

        # 8. Weekly stop
        if self.flags.weekly_stop and self.daily_risk.weekly_halted:
            evt.weekly_stop_pass = False
            evt.first_block_reason = "weekly_stop"
            self._record_signal_event(evt, sess, direction, bar_time)
            return

        # 9. Monthly stop
        if self.flags.monthly_stop and self.daily_risk.monthly_halted:
            evt.monthly_stop_pass = False
            evt.first_block_reason = "monthly_stop"
            self._record_signal_event(evt, sess, direction, bar_time)
            return

        # 10. Friction gate — moved to _evaluate_5m_entry (matches live engine timing)

        # 11. Micro-guard
        disp_norm = sizing.compute_disp_norm(
            disp_metric,
            disp_threshold,
            disp_threshold * 1.3 if disp_threshold > 0 else 1.0,
        )
        es_opp = self._es_opposes_direction(direction)
        quality_mult = sizing.compute_quality_mult(composite, sess.mode, disp_norm, es_opposing=es_opp)
        exit_tier = stops.determine_exit_tier(composite.value, quality_mult)

        # 12. Re-entry check (matches live engine's _evaluate_reentry)
        if not sess.reentry_allowed and not sess.reentry_used:
            if sess.last_stopout_ts:
                elapsed = (bar_time - sess.last_stopout_ts).total_seconds()
                # Block re-entry if stopout was too severe (matches live engine)
                if sess.last_stopout_r < C.REENTRY_MIN_LOSS_R:
                    sess.reentry_used = True  # permanently block for this box
                    evt.reentry_pass = False
                    evt.first_block_reason = "reentry_loss_too_severe"
                    self._record_signal_event(evt, sess, direction, bar_time)
                    return
                # Check cooldown
                if elapsed < C.REENTRY_COOLDOWN_MIN * 60:
                    evt.reentry_pass = False
                    evt.first_block_reason = "reentry_cooldown"
                    self._record_signal_event(evt, sess, direction, bar_time)
                    return
                else:
                    sess.reentry_allowed = True

        # --- ALL GATES PASSED ---
        self._breakouts_qualified += 1
        evt.passed_all = True

        # Compute ATR percentile for expiry (use cached incremental ATR array)
        cnt = self._inc_30m_count[sess.session]
        atr14_for_pctl = self._inc_atr14_30m[sess.session].values[:cnt]
        valid_atr = atr14_for_pctl[~np.isnan(atr14_for_pctl)]
        if len(valid_atr) > 10:
            atr_pctl = ind.percentile_rank(sess.atr14_30m, valid_atr)
        else:
            atr_pctl = 50.0

        # Activate breakout
        sig.activate_breakout(
            sess.breakout, direction, atr_pctl, bar_time,
            sess.box.box_high, sess.box.box_low, sess.box.box_width,
            sess.box.box_bars_active,
            breakout_bar_high=float(highs_30m[-1]),
            breakout_bar_low=float(lows_30m[-1]),
        )

        # Store quality context for entry fill
        sess.last_score = score
        sess.last_disp_metric = disp_metric
        sess.last_rvol = rvol

        # Record signal event (would_be_entry/stop already set at top)
        self._record_signal_event(evt, sess, direction, bar_time)

        # Reset 5m bar counter and fallback flag
        self._5m_bar_count_since_breakout = 0
        self._a_fallback_eligible = False

    def _record_signal_event(
        self,
        evt: NQDTCSignalEvent,
        sess: SessionEngineState,
        direction: Direction,
        bar_time: datetime,
    ) -> None:
        """Record signal event and update counters."""
        if not evt.passed_all:
            self._gates_blocked += 1
            if self.shadow_tracker is not None and evt.would_be_entry > 0:
                self.shadow_tracker.record_rejection(
                    direction=int(direction),
                    filter_name=evt.first_block_reason,
                    time=bar_time,
                    entry_price=evt.would_be_entry,
                    stop_price=evt.would_be_stop,
                    session=evt.session,
                    score=evt.score,
                    displacement=evt.displacement,
                    composite_regime=evt.composite_regime,
                )
        if self.cfg.track_signals:
            self._signal_events.append(evt)

    # ------------------------------------------------------------------
    # 5m entry evaluation
    # ------------------------------------------------------------------

    def _evaluate_5m_entry(
        self,
        bar_time: datetime,
        O: float, H: float, L: float, Cl: float,
        sess: SessionEngineState,
    ) -> None:
        """Evaluate 5m bar for entry triggers within an active breakout.

        All entry types (A, B, C) are evaluated independently each bar.
        A shared OCA group ensures the first fill cancels all other pending
        entry orders for this breakout.
        """
        if not sess.breakout.active:
            return
        if self._active and self._active.pos.open:
            return

        # Consecutive-loss cooldown: block entries (decrement moved to _step_5m)
        if self.flags.loss_streak_cooldown and self._cooldown_bars > 0:
            return

        # Prereq 1: inter-trade cooldown timer
        gap_min = getattr(C, 'MIN_INTER_TRADE_GAP_MINUTES', 0)
        if gap_min > 0 and self._last_fill_time is not None:
            elapsed = (bar_time - self._last_fill_time).total_seconds() / 60
            if elapsed < gap_min:
                return

        # Block entries during 05:00 ET hour
        if self.flags.block_05_et and _to_ny(bar_time).hour == 5 and _to_ny(bar_time).minute < 30:
            return

        if self.flags.block_04_et and _to_ny(bar_time).hour == 4:
            return

        if self.flags.block_06_et and _to_ny(bar_time).hour == 6:
            return

        if self.flags.block_09_et and _to_ny(bar_time).hour == 9:
            return

        if self.flags.block_12_et and _to_ny(bar_time).hour == 12:
            return

        # Block entries on Thursday (DOW=3)
        if self.flags.block_thursday and _to_ny(bar_time).weekday() == 3:
            return

        # Prereq 2: block ETH shorts (session-direction filter)
        if self.flags.block_eth_shorts and sess.session == Session.ETH and sess.breakout.direction == Direction.SHORT:
            return

        # Block DEGRADED-mode entries during RTH
        if C.BLOCK_RTH_DEGRADED and sess.session == Session.RTH and sess.mode == ChopMode.DEGRADED:
            return

        direction = sess.breakout.direction
        es_opp = self._es_opposes_direction(direction)
        self._5m_bar_count_since_breakout += 1
        vwap_val = sess.vwap_session.value
        tick = self.tick

        # Re-check regime hard block at entry time (matches live engine)
        if len(self._ema50_d) > 3 and len(self._atr14_d) > 0:
            _, daily_opposes = sig.classify_daily_support(
                self._ema50_d, self._atr14_d, direction,
            )
            if sig.regime_hard_block(
                self.regime.regime_4h.value, self.regime.trend_dir_4h, direction, daily_opposes,
            ):
                return

        # Re-check composite regime block at entry time
        if C.BLOCK_NEUTRAL_REGIME or C.BLOCK_ALIGNED_REGIME or C.BLOCK_CAUTION_REGIME:
            _comp = self.regime.composite
            if (C.BLOCK_NEUTRAL_REGIME and _comp == CompositeRegime.NEUTRAL) or \
               (C.BLOCK_ALIGNED_REGIME and _comp == CompositeRegime.ALIGNED) or \
               (C.BLOCK_CAUTION_REGIME and _comp == CompositeRegime.CAUTION):
                return

        # Friction gate (moved from _try_breakout to match live engine timing)
        if self.flags.friction_gate:
            r_dollars = self.equity * C.RISK_PCT
            if not sizing.friction_ok(self.symbol, r_dollars):
                return

        # Check if we already have A orders working (to avoid duplicate A submissions)
        has_a_working = any(
            w.subtype in (EntrySubtype.A_RETEST, EntrySubtype.A_LATCH)
            for w in self._working.values()
        )

        # Shared OCA group: reuse existing group if A orders are pending,
        # otherwise create a new one so first fill cancels all siblings.
        existing_oca = ""
        for w in self._working.values():
            if w.oca_group:
                existing_oca = w.oca_group
                break
        oca_group = existing_oca or f"ENTRY_{uuid.uuid4().hex[:8]}"

        # Phase 1.1: slope filter — compute once, apply to each entry's final_risk_pct
        slope_mult = 1.0
        if C.SLOPE_FILTER_ENABLED and self._closes_15m is not None and len(self._closes_15m) >= C.MACD_SLOW + C.MACD_SIGNAL + C.SLOPE_LOOKBACK:
            is_continuation = sig.slope_supports_breakout(np.array(self._closes_15m), direction)
            slope_mult = C.CONT_SIZE_MULT if is_continuation else C.REVERSAL_SIZE_MULT

        # Prereq 2: ETH short size reduction
        eth_short_mult = 1.0
        _eth_sm = getattr(C, 'ETH_SHORT_SIZE_MULT', 1.0)
        if _eth_sm < 1.0 and sess.session == Session.ETH and direction == Direction.SHORT:
            eth_short_mult = _eth_sm

        # --- Entry A: limit (A1 retest) + stop_limit (A2 latch) ---
        # Phase 1.2: gated by A_ENTRY_ENABLED
        if (
            C.A_ENTRY_ENABLED
            and not has_a_working
            and (self.flags.entry_a_retest or self.flags.entry_a_latch)
            and sig.a_entry_context_allowed(score=sess.last_score, box_width=sess.box.box_width)[0]
        ):
            a1_price, a2_price = sig.entry_a_trigger(
                Cl, L, H, vwap_val,
                sess.breakout.breakout_bar_high, sess.breakout.breakout_bar_low,
                sess.box.box_high, sess.atr14_30m, direction,
            )
            a1_price = round_to_tick(a1_price, tick)
            a2_price = round_to_tick(a2_price, tick)

            # Compute separate stops for A1 and A2 (matches live engine)
            stop_a1 = stops.compute_initial_stop(
                EntrySubtype.A_RETEST, direction,
                a1_price, sess.box.box_high, sess.box.box_low,
                sess.box.box_mid, sess.atr14_30m,
                tick_size=tick,
            )
            stop_a2 = stops.compute_initial_stop(
                EntrySubtype.A_LATCH, direction,
                a2_price, sess.box.box_high, sess.box.box_low,
                sess.box.box_mid, sess.atr14_30m,
                tick_size=tick,
            )

            # Quality + sizing
            disp_norm = sizing.compute_disp_norm(
                sess.last_disp_metric, sess.last_disp_threshold,
                sess.last_disp_threshold * 1.3 if sess.last_disp_threshold > 0 else 1.0,
            )
            quality_mult = sizing.compute_quality_mult(self.regime.composite, sess.mode, disp_norm, es_opposing=es_opp)
            final_risk_pct, _ = sizing.compute_final_risk_pct(quality_mult)
            final_risk_pct *= slope_mult

            qty_a1 = sizing.compute_contracts(
                self.symbol, a1_price, stop_a1, self.equity, final_risk_pct,
            )
            qty_a2 = sizing.compute_contracts(
                self.symbol, a2_price, stop_a2, self.equity, final_risk_pct,
            )
            if self.cfg.fixed_qty is not None:
                qty_a1 = self.cfg.fixed_qty
                qty_a2 = self.cfg.fixed_qty
            qty_a1 = self._apply_dd_throttle(qty_a1) if qty_a1 >= 1 else qty_a1
            qty_a2 = self._apply_dd_throttle(qty_a2) if qty_a2 >= 1 else qty_a2
            if (qty_a1 is not None and qty_a1 >= 1) or (qty_a2 is not None and qty_a2 >= 1):
                exit_tier = stops.determine_exit_tier(self.regime.composite.value, quality_mult)

                # A1 — LIMIT order (retest)
                if self.flags.entry_a_retest and qty_a1 is not None and qty_a1 >= 1:
                    self._submit_entry(
                        direction=direction, qty=qty_a1,
                        order_type=OrderType.LIMIT, limit_price=a1_price,
                        subtype=EntrySubtype.A_RETEST,
                        stop_for_risk=stop_a1,
                        quality_mult=quality_mult,
                        disp_norm=disp_norm,
                        ttl_bars=C.A_TTL_5M_BARS,
                        oca_group=oca_group,
                        bar_time=bar_time, sess=sess,
                    )

                # A2 — STOP_LIMIT order (latch)
                if self.flags.entry_a_latch and qty_a2 is not None and qty_a2 >= 1:
                    if direction == Direction.LONG:
                        a2_stop = a2_price
                        a2_limit = a2_price + C.A2_BUFFER_TICKS * tick
                    else:
                        a2_stop = a2_price
                        a2_limit = a2_price - C.A2_BUFFER_TICKS * tick
                    a2_limit = round_to_tick(a2_limit, tick)

                    self._submit_entry(
                        direction=direction, qty=qty_a2,
                        order_type=OrderType.STOP_LIMIT,
                        stop_price=a2_stop, limit_price=a2_limit,
                        subtype=EntrySubtype.A_LATCH,
                        stop_for_risk=stop_a2,
                        quality_mult=quality_mult,
                        disp_norm=disp_norm,
                        ttl_bars=C.A_TTL_5M_BARS,
                        oca_group=oca_group,
                        bar_time=bar_time, sess=sess,
                    )

        # --- Entry B: sweep + reclaim as live-style marketable IOC LIMIT ---
        # Permission gates are shared config, plus high displacement and no continuation.
        if self.flags.entry_b_sweep and sess.atr14_30m > 0:
            b_permitted = (
                sig.b_entry_regime_allowed(self.regime.composite)
                and not sess.breakout.continuation_mode
                and len(sess.disp_hist.data) > 10
                and sess.last_disp_metric >= ind.rolling_quantile_past_only(sess.disp_hist.data, C.B_MIN_DISP_Q)
            )
            b_triggered = b_permitted and sig.entry_b_trigger(
                L, H, Cl, vwap_val, sess.atr14_30m, direction,
            )
            if b_triggered:
                stop_price = stops.compute_initial_stop(
                    EntrySubtype.B_SWEEP, direction,
                    Cl, sess.box.box_high, sess.box.box_low,
                    sess.box.box_mid, sess.atr14_30m,
                    tick_size=tick,
                )
                disp_norm = sizing.compute_disp_norm(
                    sess.last_disp_metric, sess.last_disp_threshold,
                    sess.last_disp_threshold * 1.3 if sess.last_disp_threshold > 0 else 1.0,
                )
                quality_mult = sizing.compute_quality_mult(self.regime.composite, sess.mode, disp_norm, es_opposing=es_opp)
                final_risk_pct, _ = sizing.compute_final_risk_pct(quality_mult)
                final_risk_pct *= slope_mult * eth_short_mult
                qty = sizing.compute_contracts(self.symbol, Cl, stop_price, self.equity, final_risk_pct)
                if self.cfg.fixed_qty is not None:
                    qty = self.cfg.fixed_qty
                qty = self._apply_dd_throttle(qty) if qty >= 1 else qty
                if qty is not None and qty >= 1:
                    slip_cap = C.RESCUE_MAX_SLIP_ATR * sess.atr14_30m
                    if direction == Direction.LONG:
                        limit_price = round_to_tick(Cl + slip_cap, tick, "up")
                    else:
                        limit_price = round_to_tick(Cl - slip_cap, tick, "down")
                    self._submit_entry(
                        direction=direction, qty=qty,
                        order_type=OrderType.LIMIT,
                        limit_price=limit_price,
                        subtype=EntrySubtype.B_SWEEP,
                        stop_for_risk=stop_price,
                        quality_mult=quality_mult,
                        disp_norm=disp_norm,
                        oca_group=oca_group,
                        bar_time=bar_time, sess=sess,
                        ioc_bar=(O, H, L, Cl),
                    )
                    return  # B is IOC/immediate, skip C eval

        # --- Entry C: hold check ---
        if (self.flags.entry_c_standard or self.flags.entry_c_continuation) and len(self._5m_closes) >= C.C_HOLD_BARS:
            arr_c = np.array(self._5m_closes[-10:], dtype=np.float64)
            arr_l = np.array(self._5m_lows[-10:], dtype=np.float64)
            arr_h = np.array(self._5m_highs[-10:], dtype=np.float64)

            c_triggered, hold_ref = sig.entry_c_hold_check(
                arr_c, arr_l, arr_h, vwap_val, direction,
                atr14_30m=sess.atr14_30m,
            )
            if c_triggered:
                is_cont = sess.breakout.continuation_mode and self.flags.continuation_mode
                c_cont_enabled = self.flags.entry_c_continuation and C.C_CONT_ENTRY_ENABLED
                subtype = EntrySubtype.C_CONTINUATION if (is_cont and c_cont_enabled) else EntrySubtype.C_STANDARD
                if (subtype == EntrySubtype.C_STANDARD and not self.flags.entry_c_standard):
                    pass
                elif (subtype == EntrySubtype.C_CONTINUATION and not c_cont_enabled):
                    pass
                elif is_cont and not c_cont_enabled:
                    pass  # block continuation entries even when reclassified as C_STANDARD
                else:
                    # Phase 4: regime x subtype blocks
                    disp_norm = sizing.compute_disp_norm(
                        sess.last_disp_metric, sess.last_disp_threshold,
                        sess.last_disp_threshold * 1.3 if sess.last_disp_threshold > 0 else 1.0,
                    )
                    blocked = False
                    # Cap at 1 continuation per breakout
                    if (subtype == EntrySubtype.C_CONTINUATION
                            and sess.breakout.continuation_fills >= 1):
                        blocked = True
                    # MFE gate: require prior trade to have proven the breakout
                    elif (subtype == EntrySubtype.C_CONTINUATION
                            and sess.breakout.last_trade_peak_r < C.C_CONT_MFE_GATE_R):
                        blocked = True
                    elif (C.BLOCK_CONT_ALIGNED
                            and subtype == EntrySubtype.C_CONTINUATION
                            and self.regime.composite == CompositeRegime.ALIGNED):
                        blocked = True
                    elif (C.BLOCK_STD_NEUTRAL_LOW_DISP
                            and subtype == EntrySubtype.C_STANDARD
                            and self.regime.composite == CompositeRegime.NEUTRAL
                            and disp_norm < 0.5):
                        blocked = True

                    if not blocked:
                        stop_price = stops.compute_initial_stop(
                            subtype, direction,
                            Cl, sess.box.box_high, sess.box.box_low,
                            sess.box.box_mid, sess.atr14_30m,
                            hold_ref=hold_ref, tick_size=tick,
                        )
                        quality_mult = sizing.compute_quality_mult(self.regime.composite, sess.mode, disp_norm, es_opposing=es_opp)
                        final_risk_pct, _ = sizing.compute_final_risk_pct(quality_mult)
                        final_risk_pct *= slope_mult * eth_short_mult

                        # C entry price: limit at hold reference + offset (differentiated by subtype)
                        if subtype == EntrySubtype.C_STANDARD:
                            c_offset = C.C_ENTRY_OFFSET_ATR_STANDARD * sess.atr14_30m if sess.atr14_30m > 0 else tick
                        elif subtype == EntrySubtype.C_CONTINUATION:
                            c_offset = C.C_ENTRY_OFFSET_ATR_CONTINUATION * sess.atr14_30m if sess.atr14_30m > 0 else tick
                        else:
                            c_offset = C.C_ENTRY_OFFSET_ATR * sess.atr14_30m if sess.atr14_30m > 0 else tick
                        if direction == Direction.LONG:
                            c_price = round_to_tick(hold_ref + c_offset, tick)
                        else:
                            c_price = round_to_tick(hold_ref - c_offset, tick)

                        qty = sizing.compute_contracts(self.symbol, c_price, stop_price, self.equity, final_risk_pct)
                        if self.cfg.fixed_qty is not None:
                            qty = self.cfg.fixed_qty
                        qty = self._apply_dd_throttle(qty) if qty >= 1 else qty
                        if qty is not None and qty >= 1:
                            self._submit_entry(
                                direction=direction, qty=qty,
                                order_type=OrderType.LIMIT, limit_price=c_price,
                                subtype=subtype,
                                stop_for_risk=stop_price,
                                quality_mult=quality_mult,
                                disp_norm=disp_norm,
                                ttl_bars=C.A_TTL_5M_BARS,
                                oca_group=oca_group,
                                bar_time=bar_time, sess=sess,
                            )

        # --- Market fallback after A orders expire (Phase 1.2: gated by A_ENTRY_ENABLED) ---
        if C.A_ENTRY_ENABLED and self._a_fallback_eligible and not self._working:
            on_breakout_side = (
                (direction == Direction.LONG and Cl > sess.box.box_high) or
                (direction == Direction.SHORT and Cl < sess.box.box_low)
            )
            if on_breakout_side:
                stop_price = stops.compute_initial_stop(
                    EntrySubtype.MARKET_FALLBACK, direction,
                    Cl, sess.box.box_high, sess.box.box_low,
                    sess.box.box_mid, sess.atr14_30m,
                    tick_size=tick,
                )
                disp_norm = sizing.compute_disp_norm(
                    sess.last_disp_metric, sess.last_disp_threshold,
                    sess.last_disp_threshold * 1.3 if sess.last_disp_threshold > 0 else 1.0,
                )
                quality_mult = sizing.compute_quality_mult(self.regime.composite, sess.mode, disp_norm, es_opposing=es_opp)
                final_risk_pct, _ = sizing.compute_final_risk_pct(quality_mult)
                final_risk_pct *= slope_mult * eth_short_mult
                qty = sizing.compute_contracts(self.symbol, Cl, stop_price, self.equity, final_risk_pct)
                if self.cfg.fixed_qty is not None:
                    qty = self.cfg.fixed_qty
                qty = self._apply_dd_throttle(qty) if qty >= 1 else qty
                if qty is not None and qty >= 1:
                    self._submit_entry(
                        direction=direction, qty=qty,
                        order_type=OrderType.MARKET,
                        subtype=EntrySubtype.MARKET_FALLBACK,
                        stop_for_risk=stop_price,
                        quality_mult=quality_mult,
                        disp_norm=disp_norm,
                        bar_time=bar_time, sess=sess,
                    )
                self._a_fallback_eligible = False


    # ------------------------------------------------------------------
    # Drawdown throttle helper
    # ------------------------------------------------------------------

    def _apply_dd_throttle(self, qty: int) -> int | None:
        """Apply drawdown sizing multiplier. Returns None if entry blocked."""
        if not self.flags.drawdown_throttle:
            return qty
        dd_mult = self._throttle.dd_size_mult
        if dd_mult <= 0.0:
            self._throttle.entries_blocked_dd += 1
            return None
        if dd_mult < 1.0:
            return max(1, int(qty * dd_mult))
        return qty

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def _submit_entry(
        self, *,
        direction: Direction, qty: int,
        order_type: OrderType,
        subtype: EntrySubtype,
        stop_for_risk: float,
        quality_mult: float,
        disp_norm: float = 0.0,
        limit_price: float = 0.0,
        stop_price: float = 0.0,
        ttl_bars: int = 3,
        oca_group: str = "",
        bar_time: datetime,
        sess: SessionEngineState,
        ioc_bar: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Submit entry order via SimBroker."""
        side = OrderSide.BUY if direction == Direction.LONG else OrderSide.SELL
        order_id = self.broker.next_order_id()

        # Convert ttl_bars to minutes (5m bars)
        ttl_minutes = ttl_bars * 5 if ttl_bars > 0 else 0

        entry_request = NQDTCEntryRequest(
            client_order_id=order_id,
            symbol=self.symbol,
            subtype=subtype,
            direction=direction,
            qty=qty,
            stop_for_risk=stop_for_risk,
            tif="IOC" if ioc_bar is not None else "DAY",
            order_type=order_type.name,
            price=limit_price or stop_price or None,
            limit_price=limit_price or None,
            stop_price=stop_price or None,
            oca_group=oca_group,
            is_limit=(order_type == OrderType.LIMIT),
            quality_mult=quality_mult,
            submitted_bar_idx=self._bar_count_5m,
            ttl_bars=ttl_bars,
        )
        replay = self._replay_core_step(
            bar_input={
                "bar_count_5m": self._bar_count_5m,
                "bar_ts": bar_time,
                "entry_request": entry_request,
            }
        )
        submit_action = next((action for action in replay.actions if isinstance(action, SubmitEntry)), None)
        if submit_action is None:
            return
        order = _sim_order_from_parity(
            neutral_action_to_sim_order(
                submit_action,
                tick_size=self.tick,
                submit_time=bar_time,
            )
        )
        order.tag = subtype.value
        order.oca_group = oca_group
        order.ttl_minutes = ttl_minutes
        if ioc_bar is None:
            self.broker.submit_order(order)
        self._replay_core_step(
            order_updates=[
                NQDTCOrderUpdate(
                    oms_order_id=order.order_id,
                    status="accepted",
                    timestamp=bar_time,
                    order_role="entry",
                    accepted_entry=entry_request,
                )
            ]
        )

        wo = WorkingOrder(
            oms_order_id=order_id,
            subtype=subtype,
            direction=direction,
            price=limit_price if limit_price else stop_price,
            qty=qty,
            submitted_bar_idx=self._bar_idx_30m,
            ttl_bars=ttl_bars,
            oca_group=oca_group,
            is_limit=(order_type == OrderType.LIMIT),
            quality_mult=quality_mult,
            stop_for_risk=stop_for_risk,
            disp_norm=disp_norm,
        )
        self._working[order_id] = wo
        self._entries_placed += 1
        if ioc_bar is not None:
            O, H, L, Cl = ioc_bar
            fill = self.broker.fill_marketable_ioc_limit(
                order,
                bar_time,
                O,
                H,
                L,
                Cl,
                self.tick,
            )
            self._handle_fill(fill, bar_time, Cl)

    # ------------------------------------------------------------------
    # Fill handling
    # ------------------------------------------------------------------

    def _handle_fill(self, fill: FillResult, bar_time: datetime, current_close: float) -> None:
        order = fill.order

        if fill.status == FillStatus.FILLED:
            we = self._working.pop(order.order_id, None)
            if we is not None:
                self._on_entry_fill(we, fill, bar_time)
            elif self._active and order.order_id == self._active.stop_order_id:
                self._on_stop_fill(fill, bar_time)
            elif self._active:
                # Could be a TP fill
                self._on_tp_fill(fill, bar_time)

        elif fill.status == FillStatus.EXPIRED:
            self._replay_core_step(
                order_updates=[
                    NQDTCOrderUpdate(
                        oms_order_id=order.order_id,
                        status="expired",
                        timestamp=bar_time,
                    )
                ]
            )
            wo = self._working.pop(order.order_id, None)
            # If an A order expired, enable market fallback
            if wo is not None and wo.subtype in (EntrySubtype.A_RETEST, EntrySubtype.A_LATCH):
                # Check if all A orders are now gone
                has_remaining_a = any(
                    w.subtype in (EntrySubtype.A_RETEST, EntrySubtype.A_LATCH)
                    for w in self._working.values()
                )
                if not has_remaining_a:
                    self._a_fallback_eligible = True

        elif fill.status == FillStatus.CANCELLED:
            self._replay_core_step(
                order_updates=[
                    NQDTCOrderUpdate(
                        oms_order_id=order.order_id,
                        status="cancelled",
                        timestamp=bar_time,
                    )
                ]
            )
            self._working.pop(order.order_id, None)

        elif fill.status == FillStatus.REJECTED:
            self._replay_core_step(
                order_updates=[
                    NQDTCOrderUpdate(
                        oms_order_id=order.order_id,
                        status="rejected",
                        timestamp=bar_time,
                    )
                ]
            )
            if order.tag == EntrySubtype.B_SWEEP.value:
                self._working.pop(order.order_id, None)
            # Keep other rejected stop-limit orders working for re-eval.

    def _on_entry_fill(self, wo: WorkingOrder, fill: FillResult, bar_time: datetime) -> None:
        # Block fills during 05:00 ET hour
        if self.flags.block_05_et and _to_ny(bar_time).hour == 5 and _to_ny(bar_time).minute < 30:
            return

        # Block fills during 06:00 ET hour
        if self.flags.block_06_et and _to_ny(bar_time).hour == 6:
            return

        # Block fills during 12:00 ET hour
        if self.flags.block_12_et and _to_ny(bar_time).hour == 12:
            return

        # Block fills on Thursday
        if self.flags.block_thursday and _to_ny(bar_time).weekday() == 3:
            return

        fill_price = fill.fill_price
        commission = fill.commission
        commission_before_trade = self._total_commission
        self._total_commission += commission
        self.equity -= commission
        self._throttle.update_equity(self.equity)
        self._entries_filled += 1
        self._last_fill_time = bar_time  # Prereq 1: track for cooldown

        # Cancel OCA siblings
        if wo.oca_group:
            self.broker.cancel_oca_group(wo.oca_group)
            # Remove from working
            to_remove = [oid for oid, w in self._working.items() if w.oca_group == wo.oca_group]
            for oid in to_remove:
                self._working.pop(oid, None)

        # Determine session state
        session = _classify_session(bar_time)
        sess = self.eth if session == Session.ETH else self.rth

        # Compute position state
        stop_price = wo.stop_for_risk
        r_points = abs(fill_price - stop_price)

        # Max stop width gate: reject entries with outsized stop distance
        if self.flags.max_stop_width and r_points > C.MAX_STOP_WIDTH_PTS:
            return

        # Min stop distance gate: reject entries with pathologically tight stops
        if self.flags.min_stop_distance > 0 and r_points < self.flags.min_stop_distance:
            return

        r_dollars = r_points * self.pv * wo.qty
        if r_dollars <= 0:
            r_dollars = 1.0

        # Exit tier
        exit_tier = stops.determine_exit_tier(self.regime.composite.value, wo.quality_mult)

        # TP1-only cap for DEGRADED/RANGE (skip if chop_degraded ablation off)
        mode_for_exits = sess.mode.value
        if not self.flags.chop_degraded and sess.mode == ChopMode.DEGRADED:
            mode_for_exits = ChopMode.CAUTION.value
        tp1_cap = stops.should_cap_tp1_only(mode_for_exits, self.regime.regime_4h.value)

        # Compute TP levels (skip if tiered_exits disabled)
        if self.flags.tiered_exits:
            tp_levels = stops.compute_tp_levels(
                wo.direction, fill_price, r_points, exit_tier, wo.qty, self.tick,
            )
            if tp1_cap and len(tp_levels) > 1:
                tp_levels = tp_levels[:1]
        else:
            tp_levels = []

        self._replay_core_step(
            fills=[
                NQDTCFill(
                    oms_order_id=fill.order.order_id,
                    fill_price=fill_price,
                    fill_qty=wo.qty,
                    fill_time=bar_time,
                    entry_context=NQDTCEntryFillContext(
                        exit_tier=exit_tier,
                        tp_levels=list(tp_levels),
                        mm_level=sess.breakout.mm_level if sess.breakout.active else 0.0,
                        mm_reached=sess.breakout.mm_reached if sess.breakout.active else False,
                        box_high_at_entry=sess.box.box_high,
                        box_low_at_entry=sess.box.box_low,
                        box_mid_at_entry=sess.box.box_mid,
                        entry_session=sess.session,
                        tp1_only_cap=tp1_cap,
                    ),
                )
            ]
        )

        # Build position state
        pos = PositionState(
            open=True,
            symbol=self.symbol,
            direction=wo.direction,
            entry_subtype=wo.subtype,
            entry_price=fill_price,
            stop_price=stop_price,
            initial_stop_price=stop_price,
            qty=wo.qty,
            qty_open=wo.qty,
            R_dollars=r_dollars,
            risk_pct=0.0,
            quality_mult=wo.quality_mult,
            exit_tier=exit_tier,
            profit_funded=False,
            tp_levels=tp_levels,
            chandelier_trail=0.0,
            mm_level=sess.breakout.mm_level if sess.breakout.active else 0.0,
            mm_reached=sess.breakout.mm_reached if sess.breakout.active else False,
            bars_since_entry_30m=0,
            highest_since_entry=fill_price,
            lowest_since_entry=fill_price,
            hold_ref=0.0,
            box_high_at_entry=sess.box.box_high,
            box_low_at_entry=sess.box.box_low,
            box_mid_at_entry=sess.box.box_mid,
            entry_session=sess.session,
            tp1_only_cap=tp1_cap,
        )

        record = NQDTCTradeRecord(
            symbol=self.symbol,
            direction=int(wo.direction),
            entry_subtype=wo.subtype.value,
            session=sess.session.value,
            entry_time=bar_time,
            entry_price=fill_price,
            initial_stop=stop_price,
            qty=wo.qty,
            composite_regime=self.regime.composite.value,
            chop_mode=sess.mode.value,
            score_at_entry=sess.last_score,
            displacement_at_entry=sess.last_disp_metric,
            rvol_at_entry=sess.last_rvol,
            quality_mult=wo.quality_mult,
            expiry_mult=1.0,
            disp_norm_at_entry=wo.disp_norm,
            exit_tier=exit_tier.value,
            continuation=sess.breakout.continuation_mode if sess.breakout.active else False,
            box_width=sess.box.box_width,
            adaptive_L=sess.box.L_used,
        )

        active = _ActivePosition(
            pos=pos, record=record,
            mfe_price=fill_price, mae_price=fill_price,
            session_state=sess,
            commission_at_start=commission_before_trade,
        )
        self._active = active

        # Track continuation fills per breakout
        if wo.subtype == EntrySubtype.C_CONTINUATION and sess.breakout.active:
            sess.breakout.continuation_fills += 1

        # Place protective stop
        stop_side = OrderSide.SELL if wo.direction == Direction.LONG else OrderSide.BUY
        stop_id = self.broker.next_order_id()
        stop_order = SimOrder(
            order_id=stop_id, symbol=self.symbol, side=stop_side,
            order_type=OrderType.STOP, qty=wo.qty,
            stop_price=stop_price, tick_size=self.tick,
            submit_time=bar_time, ttl_hours=0, tag="protective_stop",
        )
        self.broker.submit_order(stop_order)
        active.stop_order_id = stop_id
        self._replay_core_step(
            order_updates=[
                NQDTCOrderUpdate(
                    oms_order_id=stop_id,
                    status="accepted",
                    timestamp=bar_time,
                    order_role="stop",
                )
            ]
        )

        # Place TP orders
        for i, tp in enumerate(tp_levels):
            tp_side = OrderSide.SELL if wo.direction == Direction.LONG else OrderSide.BUY
            tp_id = self.broker.next_order_id()
            if wo.direction == Direction.LONG:
                tp_price = round_to_tick(fill_price + tp.r_target * r_points, self.tick)
            else:
                tp_price = round_to_tick(fill_price - tp.r_target * r_points, self.tick)

            tp_order = SimOrder(
                order_id=tp_id, symbol=self.symbol, side=tp_side,
                order_type=OrderType.LIMIT, qty=tp.qty,
                limit_price=tp_price, tick_size=self.tick,
                submit_time=bar_time, ttl_hours=0, tag=f"TP{i+1}",
            )
            self.broker.submit_order(tp_order)
            tp.oms_order_id = tp_id
            active.tp_order_ids.append(tp_id)

    def _on_stop_fill(self, fill: FillResult, bar_time: datetime) -> None:
        if self._active is None:
            return
        self._replay_core_step(
            fills=[
                NQDTCFill(
                    oms_order_id=fill.order.order_id,
                    fill_price=fill.fill_price,
                    fill_qty=fill.order.qty,
                    fill_time=bar_time,
                    exit_type="STOP",
                )
            ]
        )
        commission = fill.commission
        self._total_commission += commission
        self.equity -= commission
        self._close_position(fill.fill_price, bar_time, "STOP")

    def _on_tp_fill(self, fill: FillResult, bar_time: datetime) -> None:
        """Handle TP order fill — partial exit."""
        if self._active is None:
            return
        pos = self._active.pos
        order = fill.order

        tp_idx = -1
        for i, tp in enumerate(pos.tp_levels):
            if tp.oms_order_id == order.order_id:
                tp_idx = i
                break
        if tp_idx < 0:
            return

        tp = pos.tp_levels[tp_idx]
        tp.filled = True
        commission = fill.commission
        self._total_commission += commission
        self.equity -= commission

        # Partial PnL
        if pos.direction == Direction.LONG:
            partial_pnl = (fill.fill_price - pos.entry_price) * self.pv * tp.qty
        else:
            partial_pnl = (pos.entry_price - fill.fill_price) * self.pv * tp.qty
        self.equity += partial_pnl
        pos.qty_open -= tp.qty

        # Update trade record TP hit flags
        if tp_idx == 0:
            self._active.record.tp1_hit = True
            pos.bars_since_tp1 = 0
        elif tp_idx == 1:
            self._active.record.tp2_hit = True
        elif tp_idx == 2:
            self._active.record.tp3_hit = True

        # Profit-funded BE after TP1
        if self.flags.profit_funded_be and tp_idx == 0 and not pos.profit_funded:
            pos.profit_funded = True
            be_stop = stops.compute_be_stop(
                pos.direction, pos.entry_price, self._atr14_5m_val, self.tick,
            )
            self._update_stop_price(be_stop, bar_time)

        # If all qty closed, finalize
        if pos.qty_open <= 0:
            self._replay_core_step(
                fills=[
                    NQDTCFill(
                        oms_order_id=order.order_id,
                        fill_price=fill.fill_price,
                        fill_qty=tp.qty,
                        fill_time=bar_time,
                        exit_type=f"TP{tp_idx+1}_FULL",
                    )
                ]
            )
            self._close_position(fill.fill_price, bar_time, f"TP{tp_idx+1}_FULL")

    # ------------------------------------------------------------------
    # Position management (30m boundaries)
    # ------------------------------------------------------------------

    def _manage_position_30m(self, bar_time: datetime, price: float) -> None:
        if self._active is None:
            return
        pos = self._active.pos
        sess = self._active.session_state
        if sess is None:
            return

        # Update position tracking
        if pos.direction == Direction.LONG:
            pos.highest_since_entry = max(pos.highest_since_entry, price)
            pos.lowest_since_entry = min(pos.lowest_since_entry, price)
        else:
            pos.lowest_since_entry = min(pos.lowest_since_entry, price)
            pos.highest_since_entry = max(pos.highest_since_entry, price)

        # Open R calculation (current stop basis — for stale exit etc.)
        r_points = abs(pos.entry_price - pos.stop_price)
        if r_points > 0:
            if pos.direction == Direction.LONG:
                open_r = (price - pos.entry_price) / r_points
            else:
                open_r = (pos.entry_price - price) / r_points
        else:
            open_r = 0.0

        # Open R on initial stop basis (for chandelier tier + ratchet)
        init_r_points = abs(pos.entry_price - pos.initial_stop_price)
        if init_r_points > 0:
            if pos.direction == Direction.LONG:
                open_r_initial = (price - pos.entry_price) / init_r_points
            else:
                open_r_initial = (pos.entry_price - price) / init_r_points
            pos.peak_r_initial = max(pos.peak_r_initial, open_r_initial)
        else:
            open_r_initial = 0.0

        # Early breakeven: move stop to entry when MFE threshold reached (pre-TP1)
        if (self.flags.early_be
                and not pos.profit_funded
                and not pos.early_be_triggered
                and pos.peak_r_initial >= C.EARLY_BE_MFE_R):
            pos.early_be_triggered = True
            be_stop = stops.compute_be_stop(
                pos.direction, pos.entry_price, self._atr14_5m_val, self.tick,
            )
            if pos.direction == Direction.LONG and be_stop > pos.stop_price:
                self._update_stop_price(be_stop, bar_time)
            elif pos.direction == Direction.SHORT and be_stop < pos.stop_price:
                self._update_stop_price(be_stop, bar_time)

        # Post-TP1 ratchet floor: lock fraction of peak R
        if (pos.profit_funded
                and init_r_points > 0
                and pos.peak_r_initial >= C.RATCHET_THRESHOLD_R):
            if pos.direction == Direction.LONG:
                ratchet_stop = pos.entry_price + C.RATCHET_LOCK_PCT * pos.peak_r_initial * init_r_points
            else:
                ratchet_stop = pos.entry_price - C.RATCHET_LOCK_PCT * pos.peak_r_initial * init_r_points
            ratchet_stop = round_to_tick(ratchet_stop, self.tick)
            if pos.direction == Direction.LONG and ratchet_stop > pos.stop_price:
                pos.stop_source = "RATCHET"
                self._update_stop_price(ratchet_stop, bar_time)
            elif pos.direction == Direction.SHORT and ratchet_stop < pos.stop_price:
                pos.stop_source = "RATCHET"
                self._update_stop_price(ratchet_stop, bar_time)

        mfe_ratchet_stop = stops.compute_mfe_ratcheted_stop(
            pos.direction,
            pos.entry_price,
            init_r_points,
            pos.peak_r_initial,
            self.tick,
        )
        if mfe_ratchet_stop is not None:
            if pos.direction == Direction.LONG and mfe_ratchet_stop > pos.stop_price:
                pos.stop_source = "MFE_RATCHET"
                self._update_stop_price(mfe_ratchet_stop, bar_time)
            elif pos.direction == Direction.SHORT and mfe_ratchet_stop < pos.stop_price:
                pos.stop_source = "MFE_RATCHET"
                self._update_stop_price(mfe_ratchet_stop, bar_time)

        # Chandelier trailing stop (use initial-stop R for tier selection)
        _grace_ok = pos.bars_since_entry_30m >= getattr(C, 'CHANDELIER_GRACE_BARS_30M', 0)
        if self.flags.chandelier_trailing and pos.runner_active and _grace_ok and len(self._atr14_1h) > 0:
            lookback, mult = stops.chandelier_params(open_r_initial, pos.mm_reached)
            # Post-TP1 mult decay: progressively tighten trail after TP1
            if pos.profit_funded and C.CHANDELIER_POST_TP1_MULT_DECAY > 0 and pos.bars_since_tp1 >= 0:
                decay = C.CHANDELIER_POST_TP1_MULT_DECAY * pos.bars_since_tp1
                mult = max(mult - decay, C.CHANDELIER_POST_TP1_FLOOR_MULT)
            if pos.direction == Direction.LONG:
                trail = ind.chandelier_long(self._highs_1h_cache, self._atr14_1h, lookback, mult)
                if trail > pos.chandelier_trail:
                    pos.chandelier_trail = trail
                    new_stop = round_to_tick(trail, self.tick)
                    if new_stop > pos.stop_price:
                        self._update_stop_price(new_stop, bar_time)
            else:
                trail = ind.chandelier_short(self._lows_1h_cache, self._atr14_1h, lookback, mult)
                if trail < pos.chandelier_trail or pos.chandelier_trail == 0:
                    pos.chandelier_trail = trail
                    new_stop = round_to_tick(trail, self.tick)
                    if new_stop < pos.stop_price:
                        self._update_stop_price(new_stop, bar_time)

        # Activate runner after all TPs filled
        if not pos.runner_active and all(tp.filled for tp in pos.tp_levels):
            pos.runner_active = True

        # Early chandelier: activate runner immediately after TP1
        if (
            self.flags.early_chandelier
            and not pos.runner_active
            and pos.profit_funded
        ):
            pos.runner_active = True

        # Stale exit
        if self.flags.stale_exit:
            bridge_extra = pos.stale_bridge_extra_bars if pos.stale_bridge_extended else 0
            stale_mode = sess.mode.value
            if not self.flags.chop_degraded and sess.mode == ChopMode.DEGRADED:
                stale_mode = ChopMode.CAUTION.value
            if stops.stale_exit_check(
                pos.bars_since_entry_30m, open_r, stale_mode, bridge_extra,
                tp1_filled=pos.profit_funded,
            ):
                self._close_position_market(price, bar_time, "STALE")
                return

        # Max loss cap: force exit if unrealized loss exceeds -3R (initial risk basis)
        if self.flags.max_loss_cap:
            init_r_points = abs(pos.entry_price - pos.initial_stop_price)
            if init_r_points > 0:
                if pos.direction == Direction.LONG:
                    init_open_r = (price - pos.entry_price) / init_r_points
                else:
                    init_open_r = (pos.entry_price - price) / init_r_points
                if init_open_r <= -3.0:
                    self._close_position_market(price, bar_time, "MAX_LOSS_CAP")
                    return

        # Overnight bridge
        if self.flags.overnight_bridge and _is_rth_close(bar_time):
            if stops.overnight_bridge_eligible(
                price, pos.box_high_at_entry, pos.box_low_at_entry,
                pos.direction, self.regime.regime_4h.value, self.regime.trend_dir_4h,
            ):
                pos.stale_bridge_extended = True
                pos.stale_bridge_extra_bars = C.OVERNIGHT_BRIDGE_EXTRA_BARS

        # Measured move check
        if not pos.mm_reached:
            if pos.direction == Direction.LONG and price >= pos.mm_level:
                pos.mm_reached = True
            elif pos.direction == Direction.SHORT and price <= pos.mm_level:
                pos.mm_reached = True

    # ------------------------------------------------------------------
    # Close position
    # ------------------------------------------------------------------

    def _close_position(self, exit_price: float, bar_time: datetime, reason: str) -> None:
        if self._active is None:
            return
        pos = self._active.pos
        if not pos.open:
            return

        # PnL on remaining qty
        if pos.direction == Direction.LONG:
            pnl = (exit_price - pos.entry_price) * self.pv * pos.qty_open
        else:
            pnl = (pos.entry_price - exit_price) * self.pv * pos.qty_open
        self.equity += pnl

        # R-multiple (use initial stop for true risk, not migrated stop)
        r_points = abs(pos.entry_price - pos.initial_stop_price)
        if r_points <= 0:
            r_points = abs(pos.entry_price - pos.stop_price)
        if r_points > 0 and pos.R_dollars > 0:
            # Total trade PnL including TP partials
            if pos.direction == Direction.LONG:
                total_pnl = (exit_price - pos.entry_price) * self.pv * pos.qty
            else:
                total_pnl = (pos.entry_price - exit_price) * self.pv * pos.qty
            # Add TP partial profits already realized
            for tp in pos.tp_levels:
                if tp.filled:
                    if pos.direction == Direction.LONG:
                        tp_price = pos.entry_price + tp.r_target * r_points
                    else:
                        tp_price = pos.entry_price - tp.r_target * r_points
                    if pos.direction == Direction.LONG:
                        total_pnl += (tp_price - pos.entry_price) * self.pv * tp.qty
                    else:
                        total_pnl += (pos.entry_price - tp_price) * self.pv * tp.qty
                    # Subtract close PnL already counted for TP qty
                    if pos.direction == Direction.LONG:
                        total_pnl -= (exit_price - pos.entry_price) * self.pv * tp.qty
                    else:
                        total_pnl -= (pos.entry_price - exit_price) * self.pv * tp.qty
            r_mult = total_pnl / pos.R_dollars
        else:
            r_mult = 0.0
            total_pnl = pnl

        # MFE/MAE in R
        if r_points > 0 and self.pv > 0:
            if pos.direction == Direction.LONG:
                mfe_r = (self._active.mfe_price - pos.entry_price) / r_points
                mae_r = (pos.entry_price - self._active.mae_price) / r_points
            else:
                mfe_r = (pos.entry_price - self._active.mfe_price) / r_points
                mae_r = (self._active.mae_price - pos.entry_price) / r_points
        else:
            mfe_r = mae_r = 0.0

        # Update daily risk tracking
        r_pnl = total_pnl / pos.R_dollars if pos.R_dollars > 0 else 0.0
        self.daily_risk.realized_pnl_R += r_pnl
        self.daily_risk.weekly_realized_R += r_pnl
        self.daily_risk.monthly_realized_R += r_pnl

        if self.daily_risk.realized_pnl_R <= C.DAILY_STOP_R:
            self.daily_risk.halted = True
        if self.daily_risk.weekly_realized_R <= C.WEEKLY_STOP_R:
            self.daily_risk.weekly_halted = True
        if self.daily_risk.monthly_realized_R <= C.MONTHLY_STOP_R:
            self.daily_risk.monthly_halted = True

        # Consecutive-loss cooldown tracking (parameterized via param_overrides)
        _streak_threshold = getattr(C, 'LOSS_STREAK_THRESHOLD', 3)
        _streak_skip = getattr(C, 'LOSS_STREAK_SKIP_BARS', 6)
        if r_mult <= 0:
            self._consec_losses += 1
            if self._consec_losses >= _streak_threshold:
                self._cooldown_bars = _streak_skip
        else:
            self._consec_losses = 0

        # Drawdown throttle equity update
        self._throttle.update_equity(self.equity)
        self._throttle.record_trade_close(r_pnl)

        # Record peak MFE R on breakout state for C_continuation gating
        sess = self._active.session_state
        if sess and sess.breakout.active:
            sess.breakout.last_trade_peak_r = max(sess.breakout.last_trade_peak_r, mfe_r)

        # Update session re-entry state
        if sess and r_pnl < C.REENTRY_MIN_LOSS_R:
            sess.reentry_allowed = False
            sess.reentry_used = False
            sess.last_stopout_r = r_pnl
            sess.last_stopout_ts = bar_time

        # Finalize record
        record = self._active.record
        record.exit_time = bar_time
        record.exit_price = exit_price
        record.pnl_dollars = total_pnl
        record.r_multiple = r_mult
        record.mfe_r = mfe_r
        record.mae_r = mae_r
        record.exit_reason = reason
        record.bars_held_30m = pos.bars_since_entry_30m
        record.commission = self._total_commission - self._active.commission_at_start

        self._trades.append(record)

        # Cancel all remaining orders
        self.broker.cancel_all(self.symbol)
        self._working.clear()

        pos.open = False
        self._active = None

    def _close_position_market(
        self, price: float, bar_time: datetime, reason: str,
    ) -> None:
        """Close via market exit -- adds slippage + exit commission."""
        if self._active is None:
            return
        pos = self._active.pos
        if not pos.open:
            return
        self._replay_core_step(
            bar_input={
                "bar_count_5m": self._bar_count_5m,
                "bar_ts": bar_time,
                "flatten_request": NQDTCSimpleRequest(reason=reason, qty=pos.qty_open),
            }
        )
        # Market-exit slippage
        slip = self.cfg.slippage.slip_ticks_normal * self.tick
        if pos.direction == Direction.LONG:
            exit_price = price - slip   # selling: adverse = lower
        else:
            exit_price = price + slip   # covering: adverse = higher
        self._replay_core_step(
            fills=[
                NQDTCFill(
                    oms_order_id=self._active.stop_order_id or reason,
                    fill_price=exit_price,
                    fill_qty=pos.qty_open,
                    fill_time=bar_time,
                    exit_type=reason,
                )
            ]
        )
        # Exit commission (matches _on_stop_fill pattern)
        commission = self.cfg.slippage.commission_per_contract * pos.qty_open
        self._total_commission += commission
        self.equity -= commission
        self._close_position(exit_price, bar_time, reason)

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
        self._replay_core_step(
            bar_input={
                "bar_count_5m": self._bar_count_5m,
                "bar_ts": bar_time,
                "stop_update": NQDTCSimpleRequest(
                    reason=self._active.pos.stop_source,
                    price=new_stop,
                    qty=self._active.pos.qty_open,
                ),
            }
        )
        self._active.pos.stop_price = new_stop
        # Update broker's stop order
        for o in self.broker.pending_orders:
            if o.order_id == self._active.stop_order_id:
                o.stop_price = new_stop
                break

    def _check_working_order_cancellation(self, bar_time: datetime, close: float) -> None:
        """Cancel A orders if price retraces too deep into box."""
        to_cancel: list[str] = []
        for oid, wo in self._working.items():
            if wo.subtype in (EntrySubtype.A_RETEST, EntrySubtype.A_LATCH):
                sess = self.eth if _classify_session(bar_time) == Session.ETH else self.rth
                if sess.atr14_30m > 0 and sig.entry_a_cancel_check(
                    close, sess.box.box_high, sess.box.box_low,
                    sess.atr14_30m, wo.direction,
                ):
                    to_cancel.append(oid)

        for oid in to_cancel:
            self._replay_core_step(
                bar_input={
                    "bar_count_5m": self._bar_count_5m,
                    "bar_ts": bar_time,
                    "cancel_order_ids": [oid],
                }
            )
            self._replay_core_step(
                order_updates=[
                    NQDTCOrderUpdate(
                        oms_order_id=oid,
                        status="cancelled",
                        timestamp=bar_time,
                    )
                ]
            )
            self.broker.cancel_orders(self.symbol, tag=None)
            self._working.pop(oid, None)

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def _check_daily_reset(self, t: int) -> None:
        today = self._daily_dates_arr[t]
        if today != self._last_reset_date and self._ny_hours_arr[t] >= 4:
            # Track daily PnL to ledger before reset
            if self.daily_risk.trade_date:
                self.daily_risk.daily_pnl_ledger.append(
                    (self.daily_risk.trade_date, self.daily_risk.realized_pnl_R)
                )
                if len(self.daily_risk.daily_pnl_ledger) > 20:
                    self.daily_risk.daily_pnl_ledger = self.daily_risk.daily_pnl_ledger[-20:]

            self.daily_risk.realized_pnl_R = 0.0
            self.daily_risk.halted = False
            self.daily_risk.trade_date = today
            self._last_reset_date = today
            self._throttle.daily_reset()
            self._throttle.update_equity(self.equity)

            # Weekly reset on Monday
            if self._ny_weekdays_arr[t] == 0:
                self.daily_risk.weekly_realized_R = 0.0
                self.daily_risk.weekly_halted = False

            # Monthly reset on 1st
            if self._ny_days_arr[t] == 1:
                self.daily_risk.monthly_realized_R = 0.0
                self.daily_risk.monthly_halted = False

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
