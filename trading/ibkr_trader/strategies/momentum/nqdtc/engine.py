"""NQ Dominant Trend Capture v2.0 — main async strategy engine (5m evaluation loop)."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from copy import deepcopy
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from libs.broker_ibkr.risk_support.tick_rules import round_to_tick
from libs.market_data.futures_roll import roll_blackout_reason, roll_force_flatten_reason
from libs.market_data.live_futures import req_panama_adjusted_historical_data
from libs.oms.models.events import OMSEventType
from libs.oms.models.intent import Intent, IntentType
from libs.oms.models.order import (
    EntryPolicy, OMSOrder, OrderRole, OrderSide, OrderType, RiskContext,
)
from libs.oms.instrumentation.runtime_refs import fill_runtime_refs
from libs.oms.risk.calculator import RiskCalculator
from libs.services.trade_recorder import TradeRecorder
from strategies.core.actions import CancelAction, SubmitEntry, SubmitExit
from strategies.core.idle_market import (
    maybe_record_idle_market_observation,
    remember_idle_market_bars,
)

from libs.risk.drawdown_throttle import DrawdownThrottle, DrawdownThrottleConfig
from . import box as box_mod
from . import config as C
from . import indicators as ind
from . import signals as sig
from . import sizing
from . import stops
from .models import (
    BoxState, BreakoutEngineState, ChopMode, CompositeRegime,
    DailyRiskState, Direction, EntrySubtype, ExitTier,
    BoxEngineState, NewsEvent,
    PositionState, RegimeState, Regime4H, RollingBuffer, Session,
    SessionEngineState, TPLevel, VWAPAccumulator, WorkingOrder,
)
from .core import logic as nqdtc_core_logic
from .core.serializers import restore_state as restore_core_state
from .core.serializers import snapshot_state as snapshot_core_state
from .core.state import (
    NQDTCCoreState,
    NQDTCEntryFillContext,
    NQDTCEntryRequest,
    NQDTCFill,
    NQDTCOrderUpdate,
)
from strategies.momentum.instrumentation.src.config_snapshot import snapshot_config_module
from strategies.momentum.nqdtc import config as strategy_config

logger = logging.getLogger(__name__)

_ET = None


def _get_et():
    global _ET
    if _ET is None:
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/New_York")
    return _ET


def _to_ny(dt_utc: datetime) -> datetime:
    return dt_utc.astimezone(_get_et())


def _minutes(h: int, m: int) -> int:
    return h * 60 + m


# ---------------------------------------------------------------------------
# Session classification (Section 1.3)
# ---------------------------------------------------------------------------

def session_type(ts_ny: datetime) -> Session:
    """Classify current real-time timestamp into ETH or RTH (inclusive start)."""
    t = ts_ny.hour * 60 + ts_ny.minute
    if _minutes(C.RTH_START_H, C.RTH_START_M) <= t < _minutes(C.RTH_END_H, C.RTH_END_M):
        return Session.RTH
    return Session.ETH


def bar_data_session(ts_ny: datetime) -> Session:
    """Classify a bar by its close time using exclusive start.

    The 09:30 bar (09:00-09:30) contains pre-RTH data → ETH.
    First RTH bar is 10:00 (09:30-10:00).
    Matches backtest _30m_bar_session() convention.
    """
    t = ts_ny.hour * 60 + ts_ny.minute
    if _minutes(C.RTH_START_H, C.RTH_START_M) < t <= _minutes(C.RTH_END_H, C.RTH_END_M):
        return Session.RTH
    return Session.ETH


def entry_window_ok(ts_ny: datetime, session: Session) -> bool:
    """Check if current time is within the entry window for the session."""
    t = ts_ny.hour * 60 + ts_ny.minute
    if session == Session.ETH:
        return _minutes(C.ETH_ENTRY_START_H, C.ETH_ENTRY_START_M) <= t < _minutes(C.ETH_ENTRY_END_H, C.ETH_ENTRY_END_M)
    return _minutes(C.RTH_ENTRY_START_H, C.RTH_ENTRY_START_M) <= t < _minutes(C.RTH_ENTRY_END_H, C.RTH_ENTRY_END_M)


def _bar_in_session(bar_dt: datetime, session: Session) -> bool:
    """Check if a bar's data belongs to the given session (exclusive start)."""
    ny = bar_dt.astimezone(_get_et()) if bar_dt.tzinfo else bar_dt
    return bar_data_session(ny) == session


# ---------------------------------------------------------------------------
# Session bar filtering (fix #1)
# ---------------------------------------------------------------------------

def _filter_bars_by_session(
    bars: list, session: Session,
) -> list:
    """Filter raw IB bars to only those belonging to the given session."""
    filtered = []
    for b in bars:
        dt = getattr(b, "date", None)
        if dt is None:
            continue
        if not isinstance(dt, datetime):
            continue
        if _bar_in_session(dt, session):
            filtered.append(b)
    return filtered


# ---------------------------------------------------------------------------
# Telemetry helper (fix #16)
# ---------------------------------------------------------------------------

def _telemetry_entry(
    event: str, session: str, mode: str, regime: str, direction: str,
    **kwargs: Any,
) -> dict:
    """Build structured telemetry dict for signal/trade events."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "session": session,
        "mode": mode,
        "regime": regime,
        "direction": direction,
    }
    record.update(kwargs)
    return record


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class NQDTCEngine:
    """Dual-session engine: ETH + RTH with independent box/breakout state."""

    def __init__(
        self,
        ib_session: Any,
        oms_service: Any,
        instruments: dict[str, Any],
        trade_recorder: TradeRecorder | None = None,
        equity: float = 100_000.0,
        symbol: str = C.DEFAULT_SYMBOL,
        state_dir: Path | None = None,
        instrumentation=None,
        equity_alloc_pct: float = 1.0,
        disable_background_tasks: bool = False,
    ) -> None:
        self._ib = ib_session
        self._oms = oms_service
        self._instruments = instruments
        self._recorder = trade_recorder
        self._equity = equity
        self._equity_alloc_pct = equity_alloc_pct
        self._symbol = symbol
        self._state_dir = state_dir or Path(".")
        self._instr = instrumentation
        self._disable_background_tasks = bool(disable_background_tasks)
        self._instr_trade_id: str = ""  # current trade ID for instrumentation

        from strategies.momentum.instrumentation.src.facade import InstrumentationKit
        self._kit = InstrumentationKit(self._instr, strategy_type="nqdtc")

        # Dual session engines
        self._engines: dict[Session, SessionEngineState] = {
            Session.ETH: SessionEngineState(session=Session.ETH),
            Session.RTH: SessionEngineState(session=Session.RTH),
        }

        # Global state
        self._regime = RegimeState()
        self._position = PositionState()
        self._daily_risk = DailyRiskState()
        self._working_orders: list[WorkingOrder] = []
        self._a_fallback_eligible = False
        self._news_blackout = False
        self._news_events: list[NewsEvent] = []

        # Consecutive-loss cooldown state
        self._consec_losses: int = 0
        self._cooldown_bars: int = 0  # 5m bars remaining in cooldown (6 = 30 min)
        self._last_fill_time: datetime | None = None

        # Drawdown throttle (daily cap disabled — NQDTC has own DailyRiskState)
        self._throttle = DrawdownThrottle(
            equity, DrawdownThrottleConfig(daily_loss_cap_r=None))


        # Bar caches — raw (all sessions)
        self._bars_5m: dict[str, np.ndarray] = {}
        self._bars_15m: dict[str, np.ndarray] = {}  # Phase 1.1: 15m bars for slope filter
        self._bars_30m: dict[str, np.ndarray] = {}
        self._bars_1h: dict[str, np.ndarray] = {}
        self._bars_4h: dict[str, np.ndarray] = {}
        self._bars_daily: dict[str, np.ndarray] = {}
        # Session-filtered 30m bars (fix #1)
        self._bars_30m_session: dict[Session, dict[str, np.ndarray]] = {
            Session.ETH: {},
            Session.RTH: {},
        }
        self._bar_count_5m = 0
        # Raw 30m bar objects for session filtering
        self._raw_bars_30m: list = []

        # Volume slot medians for RVOL (rolling)
        self._vol_slot_medians: dict[str, list[float]] = {}

        # Session boundary tracking (Section 1.4)
        self._last_session: Optional[Session] = None

        # Telemetry log (fix #16)
        self._telemetry_log: list[dict] = []


        # Signal evolution ring buffer (M2)
        from collections import deque as _deque
        self._signal_ring: _deque = _deque(maxlen=10)

        # Execution cascade timestamps (#16)
        self._cascade_ts: dict[str, datetime] = {}

        # Session transition tracking (#17)
        self._session_transitions: list[dict] = []

        # Flatten-order tracking (Rec 1/3: fill-authoritative flatten)
        self._last_flatten_oms_id: str | None = None
        self._pending_flatten_instrumentation: dict[str, dict] = {}

        # Async tasks
        self._event_task: Optional[asyncio.Task] = None
        self._cycle_task: Optional[asyncio.Task] = None
        self._event_queue: Optional[asyncio.Queue] = None
        self._running = False

        # Diagnostic pulse state
        self._last_decision_code: str = "IDLE"
        self._last_decision_details: dict = {}
        self._last_bar_ts: datetime | None = None
        self._symbol_last_bar_ts: dict[str, datetime] = {}

    def _record_decision(self, code: str, details: dict | None = None) -> None:
        if maybe_record_idle_market_observation(
            self,
            code,
            strategy_id=C.STRATEGY_ID,
            build_core_state=self._build_core_state,
            apply_core_state=self._apply_core_state,
            on_bar=nqdtc_core_logic.on_bar,
            default_symbol=self._symbol,
            default_timeframe="5m",
        ):
            return
        self._last_decision_code = code
        self._last_decision_details = details or {}

    def health_status(self) -> dict:
        return {
            "strategy_id": C.STRATEGY_ID,
            "running": self._running,
            "last_decision_code": self._last_decision_code,
            "last_decision_details": self._last_decision_details,
            "last_bar_ts": self._last_bar_ts.isoformat() if self._last_bar_ts else None,
        }

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "strategy_id": C.STRATEGY_ID,
            "symbol": self._symbol,
            "equity": self._equity,
            "core": snapshot_core_state(self._build_core_state()),
        }

    async def hydrate(self, snapshot: dict[str, Any]) -> None:
        core_snapshot = snapshot.get("core", snapshot)
        self._apply_core_state(restore_core_state(core_snapshot))
        if "equity" in snapshot:
            self._equity = float(snapshot["equity"])

    def _build_core_state(self) -> NQDTCCoreState:
        return NQDTCCoreState(
            symbol=self._symbol,
            position=deepcopy(self._position),
            working_orders=deepcopy(self._working_orders),
            bar_count_5m=self._bar_count_5m,
            last_decision_code=self._last_decision_code,
            last_decision_details=dict(self._last_decision_details),
            last_bar_ts=self._last_bar_ts,
        )

    def _apply_core_state(self, state: NQDTCCoreState) -> None:
        if state.symbol:
            self._symbol = state.symbol
        self._position = deepcopy(state.position)
        self._working_orders = deepcopy(state.working_orders)
        self._bar_count_5m = state.bar_count_5m
        self._last_decision_code = state.last_decision_code
        self._last_decision_details = dict(state.last_decision_details)
        self._last_bar_ts = state.last_bar_ts

    def _apply_core_events(self, events: list[Any]) -> None:
        for event in events:
            self._record_decision(event.code, dict(event.details))
            self._last_bar_ts = event.ts
            if event.ts is not None:
                self._symbol_last_bar_ts[C.STRATEGY_ID] = event.ts

    def liveness_payload(self) -> dict:
        return {
            "bars_processed": self._bar_count_5m,
            "symbol_freshness": {
                sym: ts.isoformat() for sym, ts in self._symbol_last_bar_ts.items()
            },
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        logger.info("NQDTC engine starting …")
        self._running = True
        self._restore_state()

        self._event_queue = self._oms.stream_events(C.STRATEGY_ID)
        self._event_task = asyncio.create_task(self._process_events())

        if not self._disable_background_tasks:
            await self._fetch_bars(request_kind="startup")
            self._update_regime()

            self._cycle_task = asyncio.create_task(self._5m_scheduler())
        logger.info("NQDTC engine started (symbol=%s)", self._symbol)

    def get_position_snapshot(self) -> list[dict]:
        """Return current position state for heartbeat emission."""
        if not self._position.open:
            return []
        r_pts = abs(self._position.entry_price - self._position.initial_stop_price)
        if self._position.direction == Direction.LONG:
            last = self._bars_5m.get("close", np.array([0]))[-1]
            ur = (last - self._position.entry_price) / r_pts if r_pts > 0 else 0
        else:
            last = self._bars_5m.get("close", np.array([0]))[-1]
            ur = (self._position.entry_price - last) / r_pts if r_pts > 0 else 0
        return [{
            "strategy_type": "nqdtc",
            "direction": "LONG" if self._position.direction == Direction.LONG else "SHORT",
            "entry_price": self._position.entry_price,
            "qty": self._position.qty_open,
            "unrealized_pnl_r": round(ur, 3),
        }]

    async def stop(self) -> None:
        logger.info("NQDTC engine stopping …")
        self._running = False
        for task in [self._cycle_task, self._event_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        # Cancel working orders
        for wo in list(self._working_orders):
            await self._cancel_order(wo.oms_order_id)
        self._working_orders.clear()
        self._persist_state()
        logger.info("NQDTC engine stopped")

    def _reset_engine(self, engine: SessionEngineState) -> None:
        """Reset session engine state (Section 1.4: fresh state each session)."""
        engine.box = BoxEngineState()
        engine.breakout = BreakoutEngineState()
        engine.vwap_session.reset()
        engine.vwap_box.reset()
        engine.chop_score = 0
        engine.mode = ChopMode.NORMAL
        engine.atr14_30m = 0.0
        engine.atr50_30m = 0.0
        engine.last_score = 0.0
        engine.last_disp_metric = 0.0
        engine.last_disp_threshold = 0.0
        engine.last_rvol = 0.0
        engine.reentry_allowed = True
        engine.reentry_used = False
        engine.last_stopout_r = 0.0
        engine.last_stopout_ts = None
        engine.last_30m_bar_count = 0
        engine.last_profitable_exit_dir = Direction.FLAT
        # Preserve disp_hist and squeeze_hist (session-scoped rolling buffers)

    def set_news_events(self, events: list[NewsEvent]) -> None:
        """Set upcoming news events for blackout checks (fix #11)."""
        self._news_events = events


    # ------------------------------------------------------------------
    # 5-minute scheduler
    # ------------------------------------------------------------------

    async def _5m_scheduler(self) -> None:
        """Sleep until next 5m boundary, then run cycle."""
        while self._running:
            now = datetime.now(timezone.utc)
            # Next 5m boundary
            minute = now.minute
            next_5 = ((minute // 5) + 1) * 5
            if next_5 >= 60:
                next_bar = (now + timedelta(hours=1)).replace(minute=next_5 - 60, second=10, microsecond=0)
            else:
                next_bar = now.replace(minute=next_5, second=10, microsecond=0)
            wait = (next_bar - now).total_seconds()
            if wait < 0:
                wait = 0
            await asyncio.sleep(wait)
            if not self._running:
                break
            try:
                await self._on_5m_close()
            except Exception:
                logger.exception("Error in 5m cycle")

    # ------------------------------------------------------------------
    # Core 5m cycle (Section E / N)
    # ------------------------------------------------------------------

    async def _on_5m_close(self) -> None:
        now = datetime.now(timezone.utc)
        self._last_bar_ts = now
        ts_ny = _to_ny(now)
        session = session_type(ts_ny)
        self._bar_count_5m += 1

        # Decrement cooldown timer on each 5m bar
        if self._cooldown_bars > 0:
            self._cooldown_bars -= 1

        # Daily risk reset at 00:00 UTC (fix #10)
        self._check_daily_risk_reset(now)

        # Session boundary reset (Section 1.4)
        if self._last_session is not None and self._last_session != session:
            # Track session transition for open position (#17)
            if self._position.open:
                r_pts = abs(self._position.entry_price - self._position.initial_stop_price)
                if self._position.direction == Direction.LONG:
                    ur = (self._bars_5m.get("close", np.array([0]))[-1] - self._position.entry_price) / r_pts if r_pts > 0 else 0
                else:
                    ur = (self._position.entry_price - self._bars_5m.get("close", np.array([0]))[-1]) / r_pts if r_pts > 0 else 0
                self._session_transitions.append({
                    "from_session": self._last_session.value,
                    "to_session": session.value,
                    "transition_time": now.isoformat(),
                    "unrealized_pnl_r": round(ur, 4),
                    "bars_held": self._position.bars_since_entry,
                    "price_at_transition": float(self._bars_5m.get("close", np.array([0]))[-1]),
                })
            opposite = Session.ETH if session == Session.RTH else Session.RTH
            self._reset_engine(self._engines[opposite])
            # Reset session VWAP for new session
            self._engines[session].vwap_session.reset(now)
            logger.info("Session switch %s → %s, reset %s engine", self._last_session.value, session.value, opposite.value)
        self._last_session = session

        engine = self._engines[session]

        logger.debug("=== 5m close %s | session=%s ===", now.isoformat(), session.value)

        # 1) Refresh equity
        await self._refresh_equity()

        # 2) Fetch latest bars
        await self._fetch_bars()

        # 3) Update session VWAP
        self._update_session_vwap(engine, session)

        # 4) Detect new 30m bar via bar count change (fix #14)
        new_30m = self._detect_new_30m_bar(engine, session)

        # 5) Update news blackout (fix #11)
        self._update_news_blackout(now)

        # 6) Update regime + chop
        self._update_regime()
        if new_30m:
            self._update_chop(engine, session)

        # 7) Compute hard gates
        gates_ok = self._hard_gates_pass(engine, ts_ny, session)

        # 8) Manage working orders on every 5m cycle (fix #17)
        await self._manage_working_orders(engine)

        if self._position.open and await self._force_flatten_for_roll(engine, now):
            self._persist_state()
            return

        # 9) If position open → manage
        if self._position.open:
            self._record_decision("MANAGING_POSITION", {"session": session.value})
            await self._manage_position(engine, now, ts_ny, session)
            if new_30m:
                self._position.bars_since_entry_30m += 1
            self._persist_state()
            return

        # 10) Not in entry window → just update state
        if not entry_window_ok(ts_ny, session):
            self._record_decision("OUTSIDE_RTH", {"reason": "entry_window_closed", "session": session.value})
            if new_30m:
                self._update_box_and_breakout(engine, session, now)
            self._persist_state()
            return

        # 11) Update box state on 30m close
        if new_30m:
            self._update_box_and_breakout(engine, session, now)

        # 11b) Block 05:00-05:29 ET: cancel pending entries, skip entry evaluation
        if ts_ny.hour == 5 and ts_ny.minute < 30:
            if self._working_orders:
                for wo in list(self._working_orders):
                    await self._cancel_order(wo.oms_order_id)
                self._working_orders.clear()
            self._persist_state()
            return

        # 12) Block RTH entries (matches backtest rth_entries=False baseline)
        if not C.RTH_ENTRIES_ENABLED and session == Session.RTH:
            self._record_decision("OUTSIDE_RTH", {"reason": "rth_entries_disabled"})
            self._persist_state()
            return

        # 13) If breakout active + gates pass → evaluate entries
        if engine.breakout.active and gates_ok and engine.mode != ChopMode.HALT:
            await self._evaluate_entries(engine, session, now)
        elif not engine.breakout.active:
            self._record_decision("AWAITING_DATA", {"reason": "no_breakout_active", "session": session.value})
        elif not gates_ok:
            self._record_decision("SIGNAL_FILTERED", {"reason": "hard_gates_failed", "session": session.value})
        elif engine.mode == ChopMode.HALT:
            self._record_decision("SIGNAL_FILTERED", {"reason": "chop_halt", "session": session.value})

        # 14) Re-entry evaluation (fix #2)
        if (
            not self._position.open
            and not engine.breakout.active
            and engine.reentry_allowed
            and not engine.reentry_used
            and engine.last_stopout_ts is not None
        ):
            await self._evaluate_reentry(engine, session, now, gates_ok)

        self._persist_state()

    # ------------------------------------------------------------------
    # 30m bar detection via bar count (fix #14)
    # ------------------------------------------------------------------

    def _detect_new_30m_bar(self, engine: SessionEngineState, session: Session) -> bool:
        """Detect new 30m bar by checking if session-filtered bar count changed."""
        bars = self._bars_30m_session.get(session, {})
        c30 = bars.get("close")
        if c30 is None:
            return False
        current_count = len(c30)
        if current_count > engine.last_30m_bar_count:
            engine.last_30m_bar_count = current_count
            return True
        return False

    # ------------------------------------------------------------------
    # News blackout (fix #11)
    # ------------------------------------------------------------------

    def _update_news_blackout(self, now: datetime) -> None:
        """Check if current time falls within any news blackout window."""
        self._news_blackout = False
        before_td = timedelta(minutes=C.NEWS_BLACKOUT_WINDOW_BEFORE_MIN)
        after_td = timedelta(minutes=C.NEWS_BLACKOUT_WINDOW_AFTER_MIN)
        for evt in self._news_events:
            if evt.event_time_utc is None:
                continue
            if (evt.event_time_utc - before_td) <= now <= (evt.event_time_utc + after_td):
                self._news_blackout = True
                return

    def _news_flatten_imminent(self, now: datetime) -> bool:
        """Check if a news event is within the flatten lead time (15 min)."""
        lead_td = timedelta(minutes=C.NEWS_FLATTEN_LEAD_MIN)
        for evt in self._news_events:
            if evt.event_time_utc is None:
                continue
            if 0 <= (evt.event_time_utc - now).total_seconds() <= lead_td.total_seconds():
                return True
        return False

    # ------------------------------------------------------------------
    # Daily risk reset (fix #10)
    # ------------------------------------------------------------------

    def _check_daily_risk_reset(self, now: datetime) -> None:
        """Reset daily PnL at 4AM ET (matches backtest). Update rolling weekly/monthly sums."""
        ny = _to_ny(now)
        date_str = ny.strftime("%Y-%m-%d")
        if self._daily_risk.trade_date != date_str and ny.hour >= 4:
            # Save previous day's PnL to ledger before resetting
            if self._daily_risk.trade_date is not None:
                self._daily_risk.daily_pnl_ledger.append(
                    (self._daily_risk.trade_date, self._daily_risk.realized_pnl_R))
                # Trim to last 20 entries
                if len(self._daily_risk.daily_pnl_ledger) > 20:
                    self._daily_risk.daily_pnl_ledger = self._daily_risk.daily_pnl_ledger[-20:]

            self._daily_risk.realized_pnl_R = 0.0
            self._daily_risk.halted = False
            self._daily_risk.trade_date = date_str
            self._throttle.daily_reset()
            self._throttle.update_equity(self._equity)
            logger.info("Daily risk reset for %s", date_str)

            # Weekly reset on Monday (matches backtest)
            if ny.weekday() == 0:
                self._daily_risk.weekly_realized_R = 0.0
                self._daily_risk.weekly_halted = False

            # Monthly reset on 1st (matches backtest)
            if ny.day == 1:
                self._daily_risk.monthly_realized_R = 0.0
                self._daily_risk.monthly_halted = False

        # Check daily halt
        if self._daily_risk.realized_pnl_R <= C.DAILY_STOP_R:
            self._daily_risk.halted = True
        # Weekly/monthly halts (no +realized_pnl_R: accumulators already include today's PnL)
        if self._daily_risk.weekly_realized_R <= C.WEEKLY_STOP_R:
            self._daily_risk.weekly_halted = True
        if self._daily_risk.monthly_realized_R <= C.MONTHLY_STOP_R:
            self._daily_risk.monthly_halted = True

    # ------------------------------------------------------------------
    # Session VWAP update
    # ------------------------------------------------------------------

    def _update_session_vwap(self, engine: SessionEngineState, session: Session) -> None:
        """Update session VWAP from latest 5m bar."""
        h = self._bars_5m.get("high")
        l = self._bars_5m.get("low")
        c = self._bars_5m.get("close")
        v = self._bars_5m.get("volume")
        if h is None or len(h) == 0:
            return
        engine.vwap_session.update(float(h[-1]), float(l[-1]), float(c[-1]), float(v[-1]))

    # ------------------------------------------------------------------
    # Box + breakout update (30m)
    # ------------------------------------------------------------------

    def _update_box_and_breakout(self, engine: SessionEngineState, session: Session, now: datetime) -> None:
        """Update box state machine and breakout qualification on 30m close."""
        # Use session-filtered bars (fix #1)
        bars = self._bars_30m_session.get(session, {})
        h30 = bars.get("high")
        l30 = bars.get("low")
        c30 = bars.get("close")
        o30 = bars.get("open")
        v30 = bars.get("volume")
        if h30 is None or len(h30) < 20:
            return

        # Compute 30m ATR
        atr14 = ind.atr(h30, l30, c30, C.ATR14_PERIOD)
        atr50 = ind.atr(h30, l30, c30, C.ATR50_PERIOD)
        engine.atr14_30m = float(atr14[-1]) if not np.isnan(atr14[-1]) else 0.0
        engine.atr50_30m = float(atr50[-1]) if not np.isnan(atr50[-1]) else 0.0

        # Update box-anchored VWAP
        if engine.box.state != BoxState.INACTIVE and v30 is not None and len(v30) > 0:
            engine.vwap_box.update(float(h30[-1]), float(l30[-1]), float(c30[-1]), float(v30[-1]))

        # Update box state machine (pass volumes for VWAP backfill, fix #21)
        box_mod.update_box_state(
            engine.box, engine.breakout,
            h30, l30, c30,
            engine.atr14_30m, engine.atr50_30m,
            now, engine.vwap_box, engine.squeeze_hist.data,
            volumes_30m=v30,
        )

        # Squeeze metric (past-only: append BEFORE breakout attempt, matches backtest)
        if engine.box.state == BoxState.ACTIVE and engine.atr14_30m > 0:
            sq_val = ind.squeeze_metric(engine.box.box_width, engine.atr14_30m)
            engine.squeeze_hist.append(sq_val)

        # Update existing breakout state (with regime hard block check)
        if engine.breakout.active:
            daily_supports, daily_opposes = sig.classify_daily_support(
                self._bars_daily.get("ema50", np.array([])),
                self._bars_daily.get("atr14", np.array([])),
                engine.breakout.direction,
            )
            regime_blocked = sig.regime_hard_block(
                self._regime.regime_4h.value, self._regime.trend_dir_4h,
                engine.breakout.direction, daily_opposes,
            )
            sig.update_breakout_state(
                engine.breakout, float(c30[-1]), engine.atr14_30m,
                engine.box.box_high, engine.box.box_low,
                regime_hard_blocked=regime_blocked,
            )

        # Breakout qualification (only from ACTIVE state, matches backtest)
        if engine.box.state == BoxState.ACTIVE and not engine.breakout.active:
            self._try_breakout(engine, h30, l30, c30, o30, v30, now)

        # Signal evolution: snapshot 30m state after all updates (M2)
        self._signal_ring.append(self._snapshot_signal_state(engine))

    def _try_breakout(
        self, engine: SessionEngineState,
        h30: np.ndarray, l30: np.ndarray, c30: np.ndarray,
        o30: Optional[np.ndarray], v30: Optional[np.ndarray],
        now: datetime,
    ) -> None:
        """Attempt breakout qualification on 30m close."""
        close = float(c30[-1])
        direction = sig.breakout_structural(close, engine.box.box_high, engine.box.box_low)
        if direction is None:
            return

        # 1. Regime hard block (first gate, matches backtest)
        daily_supports, daily_opposes = sig.classify_daily_support(
            self._bars_daily.get("ema50", np.array([])),
            self._bars_daily.get("atr14", np.array([])),
            direction,
        )
        if sig.regime_hard_block(
            self._regime.regime_4h.value, self._regime.trend_dir_4h, direction, daily_opposes,
        ):
            self._log_telemetry("breakout_blocked", engine, direction, reason="regime_hard_block")
            return

        # 2. CHOP halt
        if engine.mode == ChopMode.HALT:
            self._log_telemetry("breakout_blocked", engine, direction, reason="chop_halt")
            return

        # 3. Displacement check (past-only, context-adaptive)
        vwap_box_val = engine.vwap_box.value
        atr_expanding = engine.atr14_30m > engine.atr50_30m if engine.atr50_30m > 0 else False
        _sq_good_disp, _ = self._squeeze_flags(engine)
        _regime_aligned = (self._regime.composite == CompositeRegime.ALIGNED)
        disp, threshold, passed = sig.displacement_pass(
            close, vwap_box_val, engine.atr14_30m,
            engine.disp_hist.data, atr_expanding=atr_expanding,
            squeeze_good=_sq_good_disp, regime_aligned=_regime_aligned,
        )
        # Past-only: append observation AFTER comparison
        engine.disp_hist.append(disp)
        engine.last_disp_metric = disp
        engine.last_disp_threshold = threshold
        if C.DISPLACEMENT_THRESHOLD_ENABLED and not passed:
            self._log_telemetry("breakout_blocked", engine, direction, reason="displacement_fail",
                                disp=disp, threshold=threshold)
            return

        # 4. Quality reject
        bar_h = float(h30[-1])
        bar_l = float(l30[-1])
        bar_o = float(o30[-1]) if o30 is not None else close
        rvol = 1.0
        if v30 is not None and len(v30) > 0:
            slot_key = str(now.hour)
            medians = self._vol_slot_medians.get(slot_key)
            median_vol = float(np.median(medians)) if medians and len(medians) > 0 else float(v30[-1])
            rvol = ind.compute_rvol(float(v30[-1]), median_vol)
        engine.last_rvol = rvol

        rejected, body_decisive = sig.breakout_quality_reject(
            bar_h, bar_l, bar_o, close, engine.atr14_30m, rvol, direction,
        )
        if rejected:
            self._log_telemetry("breakout_rejected", engine, direction, rvol=rvol)
            return

        # 5. Score
        squeeze_good, squeeze_loose = self._squeeze_flags(engine)
        atr_rising = engine.atr14_30m > engine.atr50_30m if engine.atr50_30m > 0 else False
        two_outside = False
        if len(c30) >= 2:
            prev = float(c30[-2])
            two_outside = (
                (close > engine.box.box_high and prev > engine.box.box_high) or
                (close < engine.box.box_low and prev < engine.box.box_low)
            )

        score = sig.compute_score(
            rvol, two_outside, atr_rising,
            squeeze_good, squeeze_loose,
            self._regime.regime_4h.value, self._regime.trend_dir_4h, direction,
            daily_supports, body_decisive,
        )
        engine.last_score = score

        min_score = sig.score_threshold(engine.mode)
        if self._regime.composite != CompositeRegime.RANGE and C.SCORE_NON_RANGE_MULT != 1.0:
            min_score *= C.SCORE_NON_RANGE_MULT
        if score < min_score:
            self._log_telemetry("breakout_score_fail", engine, direction, score=score, threshold=min_score)
            return

        context_ok, context_reason = sig.contextual_score_filter_pass(
            score=score,
            box_width=engine.box.box_width,
            rvol=rvol,
        )
        if not context_ok:
            self._log_telemetry(
                "breakout_score_fail",
                engine,
                direction,
                reason=context_reason,
                score=score,
                rvol=rvol,
                box_width=engine.box.box_width,
            )
            return

        # Exit-opt: reject narrow/wide boxes
        if engine.box.box_width < C.MIN_BOX_WIDTH or engine.box.box_width > C.MAX_BOX_WIDTH:
            self._log_telemetry("breakout_blocked", engine, direction,
                               reason="box_width_filter", box_width=engine.box.box_width)
            return

        # ATR percentile for expiry
        atr14 = ind.atr(h30, l30, c30, C.ATR14_PERIOD)
        atr_pctl = ind.percentile_rank(engine.atr14_30m, atr14[~np.isnan(atr14)])

        # Activate breakout (store breakout bar extremes for A2 placement)
        sig.activate_breakout(
            engine.breakout, direction, atr_pctl, now,
            engine.box.box_high, engine.box.box_low, engine.box.box_width,
            engine.box.box_bars_active,
            breakout_bar_high=bar_h, breakout_bar_low=bar_l,
        )
        self._a_fallback_eligible = False

        self._log_telemetry("breakout_activated", engine, direction,
                            score=score, disp=disp, threshold=threshold, rvol=rvol,
                            expiry_bars=engine.breakout.expiry_bars)
        logger.info(
            "BREAKOUT %s: score=%.1f disp=%.3f(th=%.3f) rvol=%.2f expiry=%d",
            "LONG" if direction == Direction.LONG else "SHORT",
            score, disp, threshold, rvol, engine.breakout.expiry_bars,
        )

    def _squeeze_flags(self, engine: SessionEngineState) -> tuple[bool, bool]:
        """Compute squeeze_good and squeeze_loose from history."""
        if not engine.squeeze_hist.data:
            return False, False
        arr = np.array(engine.squeeze_hist.data)
        good_th = float(np.quantile(arr, C.SQUEEZE_GOOD_QUANTILE))
        loose_th = float(np.quantile(arr, C.SQUEEZE_LOOSE_QUANTILE))
        if engine.atr14_30m <= 0 or engine.box.box_width <= 0:
            return False, False
        current = ind.squeeze_metric(engine.box.box_width, engine.atr14_30m)
        return current <= good_th, current >= loose_th

    # ------------------------------------------------------------------
    # Regime update (4H + Daily)
    # ------------------------------------------------------------------

    def _update_regime(self) -> None:
        """Update 4H and daily regime classification."""
        # 4H
        h4_close = self._bars_4h.get("close")
        h4_high = self._bars_4h.get("high")
        h4_low = self._bars_4h.get("low")
        if h4_close is not None and len(h4_close) > C.EMA50_PERIOD:
            ema50_4h = ind.ema(h4_close, C.EMA50_PERIOD)
            atr14_4h = ind.atr(h4_high, h4_low, h4_close, C.ATR14_PERIOD)
            adx14_4h, _, _ = ind.adx(h4_high, h4_low, h4_close, C.ADX_PERIOD)

            regime_str, trend_dir, slope, adx_val = sig.classify_4h(ema50_4h, atr14_4h, adx14_4h)
            self._regime.regime_4h = Regime4H(regime_str)
            self._regime.trend_dir_4h = trend_dir
            self._regime.slope_4h = slope
            self._regime.adx_4h = adx_val

            # Store for daily support lookups
            self._bars_4h["ema50"] = ema50_4h
            self._bars_4h["atr14"] = atr14_4h

        # Daily
        d_close = self._bars_daily.get("close")
        d_high = self._bars_daily.get("high")
        d_low = self._bars_daily.get("low")
        if d_close is not None and len(d_close) > C.EMA50_PERIOD:
            ema50_d = ind.ema(d_close, C.EMA50_PERIOD)
            atr14_d = ind.atr(d_high, d_low, d_close, C.ATR14_PERIOD)
            self._bars_daily["ema50"] = ema50_d
            self._bars_daily["atr14"] = atr14_d

    # ------------------------------------------------------------------
    # Chop update (30m, session-scoped)
    # ------------------------------------------------------------------

    def _update_chop(self, engine: SessionEngineState, session: Session) -> None:
        """Update chop score and mode."""
        # Use session-filtered bars (fix #1) and session VWAP (fix #13)
        bars = self._bars_30m_session.get(session, {})
        h30 = bars.get("high")
        l30 = bars.get("low")
        c30 = bars.get("close")
        v30 = bars.get("volume")
        if c30 is None or len(c30) < 60:
            return

        atr14 = ind.atr(h30, l30, c30, C.ATR14_PERIOD)
        valid = atr14[~np.isnan(atr14)]
        atr_pctl = ind.percentile_rank(engine.atr14_30m, valid) if len(valid) > 0 else 50.0

        # Build VWAP array from session VWAP accumulator values (fix #13)
        # Use session VWAP instead of a from-scratch full-period VWAP
        vwap_arr = ind.session_vwap(
            h30, l30, c30,
            v30 if v30 is not None else np.ones(len(c30)),
            max(0, len(c30) - C.CHOP_VWAP_CROSS_LB - 5),
        )
        cross_cnt = ind.vwap_cross_count(c30, vwap_arr, C.CHOP_VWAP_CROSS_LB)

        engine.chop_score = sig.compute_chop_score(atr_pctl, cross_cnt)
        engine.mode = sig.chop_mode(engine.chop_score)

    # ------------------------------------------------------------------
    # Hard gates (Section 6)
    # ------------------------------------------------------------------

    def _hard_gates_pass(self, engine: SessionEngineState, ts_ny: datetime, session: Session) -> bool:
        """Check all hard safety gates. Returns True if entry is allowed."""
        if self._news_blackout:
            return False
        if self._daily_risk.halted:
            return False
        if self._daily_risk.weekly_halted or self._daily_risk.monthly_halted:
            return False
        if engine.mode == ChopMode.HALT:
            return False
        if self._position.open:
            return False
        # Friction gate checked at entry time
        return True

    # ------------------------------------------------------------------
    # Entry evaluation (5m)
    # ------------------------------------------------------------------

    async def _evaluate_entries(self, engine: SessionEngineState, session: Session, now: datetime) -> None:
        """Evaluate and place entries when breakout is active."""
        # Record signal detection timestamp (#16)
        self._cascade_ts["_last_eval"] = now

        # Consecutive-loss cooldown: block entries during cooldown period
        if self._cooldown_bars > 0:
            return

        gap_min = getattr(C, "MIN_INTER_TRADE_GAP_MINUTES", 0)
        if gap_min > 0 and self._last_fill_time is not None:
            elapsed = (now - self._last_fill_time).total_seconds() / 60.0
            if elapsed < gap_min:
                return

        # Block entries during 04:00 ET hour (thin pre-dawn liquidity)
        if C.BLOCK_04_ET and _to_ny(now).hour == 4:
            return

        # Block entries during 05:00-05:29 ET (backtest shows 05:30-05:59 is +$7.7k/10 trades)
        if C.BLOCK_05_ET and _to_ny(now).hour == 5 and _to_ny(now).minute < 30:
            return

        # Block entries during 06:00 ET hour (pre-European-open, WR=39%)
        if C.BLOCK_06_ET and _to_ny(now).hour == 6:
            return

        # Block entries during 09:00 ET hour (RTH open whipsaw)
        if C.BLOCK_09_ET and _to_ny(now).hour == 9:
            return

        # Block entries during 12:00 ET hour (17% WR, outlier-dependent)
        if C.BLOCK_12_ET and _to_ny(now).hour == 12:
            return

        # Block DEGRADED-mode entries during RTH
        if C.BLOCK_RTH_DEGRADED and engine.session == Session.RTH and engine.mode == ChopMode.DEGRADED:
            return

        # Exit-opt: block ETH short entries (negative expectancy)
        if C.BLOCK_ETH_SHORTS and engine.session == Session.ETH and engine.breakout.direction == Direction.SHORT:
            return

        direction = engine.breakout.direction
        trade_dir = direction

        # Phase 2B: emit indicator snapshot at entry evaluation
        if self._kit.active:
            try:
                self._kit.on_indicator_snapshot(
                    pair=self._symbol,
                    indicators={
                        "atr14_30m": engine.atr14_30m,
                        "displacement": engine.last_disp_metric,
                        "disp_threshold": engine.last_disp_threshold,
                        "chop_score": engine.chop_score,
                        "rvol": engine.last_rvol,
                        "score": engine.last_score,
                        "vwap_session": float(engine.vwap_session.value) if engine.vwap_session.value else 0.0,
                    },
                    signal_name=f"nqdtc_breakout_{direction.name}",
                    signal_strength=0.5,
                    decision="eval",
                    strategy_type="nqdtc",
                    exchange_timestamp=now,
                    context={
                        "session": engine.session.value,
                        "mode": engine.mode.value,
                        "regime_4h": self._regime.regime_4h.value,
                        "composite": self._regime.composite.value if self._regime.composite else "",
                        "concurrent_positions": 1 if self._position.open else 0,
                        "drawdown_tier": self._dd_tier_name(),
                    },
                )
            except Exception:
                pass

        # Compute composite regime for this trade direction
        daily_supports, daily_opposes = sig.classify_daily_support(
            self._bars_daily.get("ema50", np.array([])),
            self._bars_daily.get("atr14", np.array([])),
            trade_dir,
        )
        composite = sig.compute_composite_regime(
            self._regime.regime_4h.value, self._regime.trend_dir_4h, trade_dir,
            daily_supports, daily_opposes,
        )
        self._regime.composite = composite

        # Hard block
        if sig.regime_hard_block(
            self._regime.regime_4h.value, self._regime.trend_dir_4h, trade_dir, daily_opposes,
        ):
            self._log_missed(direction, "BREAKOUT", "", "regime_hard_block", "4H regime opposes direction")
            return

        if (
            (C.BLOCK_NEUTRAL_REGIME and composite == CompositeRegime.NEUTRAL)
            or (C.BLOCK_ALIGNED_REGIME and composite == CompositeRegime.ALIGNED)
            or (C.BLOCK_CAUTION_REGIME and composite == CompositeRegime.CAUTION)
        ):
            self._log_missed(direction, "BREAKOUT", "", "regime_composite_block", composite.value)
            return

        # Sizing
        disp_norm = sizing.compute_disp_norm(
            engine.last_disp_metric,
            engine.last_disp_threshold,
            engine.last_disp_threshold * 1.3 if engine.last_disp_threshold > 0 else 1.0,
        )
        # ES SMA200 directional sizing — backtest-validated, live requires ES data feed
        es_opp = (self._regime.es_daily_trend != 0
                  and ((trade_dir == Direction.LONG and self._regime.es_daily_trend == -1)
                       or (trade_dir == Direction.SHORT and self._regime.es_daily_trend == 1)))
        quality_mult = sizing.compute_quality_mult(composite, engine.mode, disp_norm, es_opposing=es_opp)
        final_risk_pct, floored = sizing.compute_final_risk_pct(quality_mult)

        # Phase 1.1: slope filter — half-size continuation breakouts
        if C.SLOPE_FILTER_ENABLED:
            c15 = self._bars_15m.get("close")
            if c15 is not None and len(c15) >= C.MACD_SLOW + C.MACD_SIGNAL + C.SLOPE_LOOKBACK:
                is_continuation = sig.slope_supports_breakout(c15, trade_dir)
                if is_continuation:
                    final_risk_pct *= C.CONT_SIZE_MULT
                else:
                    final_risk_pct *= C.REVERSAL_SIZE_MULT

        # v7: portfolio-level continuation sizing — reduce size on box continuation breakouts
        if engine.breakout.continuation_mode and C.CONTINUATION_BREAKOUT_SIZE_MULT < 1.0:
            final_risk_pct *= C.CONTINUATION_BREAKOUT_SIZE_MULT

        exit_tier = stops.determine_exit_tier(composite.value, quality_mult)

        # Friction gate + TP1 viability
        R_dollars = self._equity * C.RISK_PCT
        fee_r = sizing.fee_R_estimate(self._symbol, R_dollars)
        if not sizing.friction_ok(self._symbol, R_dollars):
            self._log_telemetry("entry_skipped", engine, direction, reason="friction_gate", fee_R=fee_r)
            return
        if not sizing.tp1_viable(self._symbol, R_dollars):
            self._log_telemetry("entry_skipped", engine, direction, reason="tp1_fee_viability", fee_R=fee_r)
            return

        # Drawdown throttle: block entries entirely when sizing paused
        dd_mult = self._throttle.dd_size_mult
        if dd_mult <= 0.0:
            self._log_telemetry("entry_skipped", engine, direction, reason="drawdown_pause")
            return

        # Get 5m bars
        c5 = self._bars_5m.get("close")
        h5 = self._bars_5m.get("high")
        l5 = self._bars_5m.get("low")
        if c5 is None or len(c5) < 3:
            return

        close_5m = float(c5[-1])
        vwap_val = engine.vwap_session.value

        # --- Normal breakout phase ---
        # Shared OCA group: reuse existing group from A orders if pending,
        # so first fill (A, B, or C) cancels all siblings.
        existing_oca = ""
        for wo in self._working_orders:
            if wo.oca_group:
                existing_oca = wo.oca_group
                break
        oca_group = existing_oca or f"ENTRY_{uuid.uuid4().hex[:8]}"

        # Entry A: place OCO if not already placed (Phase 1.2: gated by A_ENTRY_ENABLED)
        if (
            C.A_ENTRY_ENABLED
            and (C.A_ENTRY_RETEST_ENABLED or C.A_ENTRY_LATCH_ENABLED)
            and not self._has_active_a_orders()
        ):
            a_allowed, a_reason = sig.a_entry_context_allowed(
                score=engine.last_score,
                box_width=engine.box.box_width,
            )
            if a_allowed:
                await self._place_entry_a(engine, direction, vwap_val, quality_mult, exit_tier, final_risk_pct, now)
            else:
                self._log_missed(
                    direction,
                    EntrySubtype.A_RETEST.value,
                    f"A_{direction.name}",
                    a_reason,
                    f"score={engine.last_score:.2f}, box={engine.box.box_width:.1f}",
                )

        # Entry B: sweep specialist
        # Permission gates are shared config, plus high displacement and no continuation.
        if engine.atr14_30m > 0:
            b_permitted = (
                sig.b_entry_regime_allowed(composite)
                and not engine.breakout.continuation_mode
                and len(engine.disp_hist.data) > 10
                and engine.last_disp_metric >= ind.rolling_quantile_past_only(engine.disp_hist.data, C.B_MIN_DISP_Q)
            )
            if b_permitted and sig.entry_b_trigger(float(l5[-1]), float(h5[-1]), close_5m, vwap_val, engine.atr14_30m, direction):
                await self._place_entry_b(engine, direction, close_5m, quality_mult, exit_tier, final_risk_pct, now, oca_group=oca_group)

        # Entry C: evaluate independently with shared OCA group
        if len(c5) >= C.C_HOLD_BARS:
            ok, hold_ref = sig.entry_c_hold_check(c5, l5, h5, vwap_val, direction, atr14_30m=engine.atr14_30m)
            if ok:
                subtype = EntrySubtype.C_CONTINUATION if engine.breakout.continuation_mode else EntrySubtype.C_STANDARD
                _c_sig_id = f"C_{direction.name}"
                # Phase 4: regime x subtype blocks
                blocked = False
                # nqdtc_v4 step 1: disable C_continuation entirely
                if subtype == EntrySubtype.C_CONTINUATION and not C.C_CONT_ENTRY_ENABLED:
                    blocked = True
                    self._log_missed(direction, subtype.value, _c_sig_id, "C_CONT_DISABLED", "C_continuation disabled")
                # Cap at 1 continuation per breakout
                elif (subtype == EntrySubtype.C_CONTINUATION
                        and engine.breakout.continuation_fills >= 1):
                    blocked = True
                    self._log_missed(direction, subtype.value, _c_sig_id, "C_CONT_MAX_FILLS", f"fills={engine.breakout.continuation_fills}")
                # MFE gate: require prior trade to have proven the breakout
                elif (subtype == EntrySubtype.C_CONTINUATION
                        and engine.breakout.last_trade_peak_r < C.C_CONT_MFE_GATE_R):
                    blocked = True
                    self._log_missed(direction, subtype.value, _c_sig_id, "C_CONT_MFE_GATE", f"peak_r={engine.breakout.last_trade_peak_r:.2f}",
                                    filter_decisions=[{"filter_name": "C_CONT_MFE_GATE", "threshold": C.C_CONT_MFE_GATE_R, "actual_value": engine.breakout.last_trade_peak_r, "passed": False}])
                elif (C.BLOCK_CONT_ALIGNED
                        and subtype == EntrySubtype.C_CONTINUATION
                        and self._regime.composite == CompositeRegime.ALIGNED):
                    blocked = True
                    self._log_missed(direction, subtype.value, _c_sig_id, "C_CONT_ALIGNED_BLOCK", "continuation in ALIGNED")
                elif (C.BLOCK_STD_NEUTRAL_LOW_DISP
                        and subtype == EntrySubtype.C_STANDARD
                        and self._regime.composite == CompositeRegime.NEUTRAL
                        and disp_norm < 0.5):
                    blocked = True
                    self._log_missed(direction, subtype.value, _c_sig_id, "C_STD_NEUTRAL_LOW_DISP", f"disp_norm={disp_norm:.2f}",
                                    filter_decisions=[{"filter_name": "C_STD_NEUTRAL_LOW_DISP", "threshold": 0.5, "actual_value": disp_norm, "passed": False}])
                if not blocked:
                    await self._place_entry_c(
                        engine, direction, hold_ref, vwap_val,
                        subtype, quality_mult, exit_tier, final_risk_pct, now,
                        oca_group=oca_group,
                    )

        # --- Market fallback after A orders expire (Phase 1.2: gated by A_ENTRY_ENABLED) ---
        if C.A_ENTRY_ENABLED and self._a_fallback_eligible and not self._working_orders:
            on_breakout_side = (
                (direction == Direction.LONG and close_5m > engine.box.box_high) or
                (direction == Direction.SHORT and close_5m < engine.box.box_low)
            )
            if on_breakout_side:
                await self._place_market_fallback(
                    engine, direction, close_5m, quality_mult, exit_tier, final_risk_pct, now,
                )
            self._a_fallback_eligible = False

    # ------------------------------------------------------------------
    # Entry placement
    # ------------------------------------------------------------------

    def _apply_dd_throttle(self, qty: int) -> int | None:
        """Apply drawdown-based live sizing throttle."""
        dd_mult = self._throttle.dd_size_mult
        if dd_mult <= 0.0:
            self._throttle.entries_blocked_dd += 1
            return None
        if dd_mult < 1.0:
            return max(1, int(qty * dd_mult))
        return qty

    async def _place_entry_a(
        self, engine: SessionEngineState, direction: Direction,
        vwap_session: float, quality_mult: float, exit_tier: ExitTier,
        final_risk_pct: float, now: datetime,
    ) -> None:
        """Place A1 (limit) + A2 (stop) as OCO pair."""
        inst = self._instruments.get(self._symbol)
        if inst is None:
            return

        tick = inst.tick_size
        # Use frozen breakout bar high/low for A2 placement
        bo_high = engine.breakout.breakout_bar_high
        bo_low = engine.breakout.breakout_bar_low
        if bo_high == 0.0 and bo_low == 0.0:
            return

        a1_price, a2_price = sig.entry_a_trigger(
            0, 0, 0, vwap_session, bo_high, bo_low,
            engine.box.box_high, engine.atr14_30m, direction,
        )

        a1_price = round_to_tick(a1_price, tick)
        a2_price = round_to_tick(a2_price, tick)

        # Compute stops
        stop_a1 = stops.compute_initial_stop(
            EntrySubtype.A_RETEST, direction, a1_price,
            engine.box.box_high, engine.box.box_low, engine.box.box_mid,
            engine.atr14_30m, tick_size=tick,
        )
        stop_a2 = stops.compute_initial_stop(
            EntrySubtype.A_LATCH, direction, a2_price,
            engine.box.box_high, engine.box.box_low, engine.box.box_mid,
            engine.atr14_30m, tick_size=tick,
        )

        # Sizing (with drawdown throttle)
        qty_a1 = sizing.compute_contracts(self._symbol, a1_price, stop_a1, self._equity, final_risk_pct)
        qty_a2 = sizing.compute_contracts(self._symbol, a2_price, stop_a2, self._equity, final_risk_pct)
        qty_a1 = self._apply_dd_throttle(qty_a1) if qty_a1 >= 1 else qty_a1
        qty_a2 = self._apply_dd_throttle(qty_a2) if qty_a2 >= 1 else qty_a2
        qty_a1 = qty_a1 or 0
        qty_a2 = qty_a2 or 0
        if (
            (not C.A_ENTRY_RETEST_ENABLED or qty_a1 < 1)
            and (not C.A_ENTRY_LATCH_ENABLED or qty_a2 < 1)
        ):
            return

        oca_group = f"A_OCO_{uuid.uuid4().hex[:8]}"

        # A1 limit
        if C.A_ENTRY_RETEST_ENABLED and qty_a1 >= 1:
            await self._submit_order(
                subtype=EntrySubtype.A_RETEST,
                direction=direction,
                order_type=OrderType.LIMIT,
                price=a1_price,
                stop_price=None,
                qty=qty_a1,
                stop_for_risk=stop_a1,
                oca_group=oca_group,
                is_limit=True,
                quality_mult=quality_mult,
            )

        # A2 stop-limit (trigger at a2_price, limit buffer above/below)
        if C.A_ENTRY_LATCH_ENABLED and qty_a2 >= 1:
            if direction == Direction.LONG:
                a2_limit = round_to_tick(a2_price + C.A2_BUFFER_TICKS * tick, tick)
            else:
                a2_limit = round_to_tick(a2_price - C.A2_BUFFER_TICKS * tick, tick)
            await self._submit_order(
                subtype=EntrySubtype.A_LATCH,
                direction=direction,
                order_type=OrderType.STOP_LIMIT,
                price=a2_limit,
                stop_price=a2_price,
                qty=qty_a2,
                stop_for_risk=stop_a2,
                oca_group=oca_group,
                is_limit=False,
                quality_mult=quality_mult,
            )

        logger.info("Placed A OCO: A1(limit)=%.2f qty=%d | A2(stop)=%.2f qty=%d | oca=%s",
                     a1_price, qty_a1, a2_price, qty_a2, oca_group)

    async def _place_entry_b(
        self, engine: SessionEngineState, direction: Direction,
        close_5m: float, quality_mult: float, exit_tier: ExitTier,
        final_risk_pct: float, now: datetime,
        oca_group: str = "",
    ) -> None:
        """Place B sweep entry (marketable limit/IOC)."""
        tick = C.NQ_SPECS[self._symbol]["tick"]
        stop_b = stops.compute_initial_stop(
            EntrySubtype.B_SWEEP, direction, close_5m,
            engine.box.box_high, engine.box.box_low, engine.box.box_mid,
            engine.atr14_30m, tick_size=tick,
        )
        qty = sizing.compute_contracts(self._symbol, close_5m, stop_b, self._equity, final_risk_pct)
        qty = self._apply_dd_throttle(qty) if qty >= 1 else qty
        if qty is None:
            return
        if qty < 1:
            return

        # Slippage cap
        slip_cap = C.RESCUE_MAX_SLIP_ATR * engine.atr14_30m
        if direction == Direction.LONG:
            limit_price = round_to_tick(close_5m + slip_cap, tick, "up")
        else:
            limit_price = round_to_tick(close_5m - slip_cap, tick, "down")

        await self._submit_order(
            subtype=EntrySubtype.B_SWEEP,
            direction=direction,
            order_type=OrderType.LIMIT,
            price=limit_price,
            stop_price=None,
            qty=qty,
            stop_for_risk=stop_b,
            tif="IOC",
            quality_mult=quality_mult,
            oca_group=oca_group,
        )
        logger.info("Placed B_sweep: limit=%.2f qty=%d stop=%.2f", limit_price, qty, stop_b)

    async def _place_entry_c(
        self, engine: SessionEngineState, direction: Direction,
        hold_ref: float, vwap_session: float,
        subtype: EntrySubtype, quality_mult: float, exit_tier: ExitTier,
        final_risk_pct: float, now: datetime,
        oca_group: str = "",
    ) -> None:
        """Place C_standard or C_continuation entry."""
        tick = C.NQ_SPECS[self._symbol]["tick"]
        # C entry price: limit at hold reference + offset (differentiated by subtype)
        if subtype == EntrySubtype.C_STANDARD:
            c_offset = C.C_ENTRY_OFFSET_ATR_STANDARD * engine.atr14_30m if engine.atr14_30m > 0 else tick
        elif subtype == EntrySubtype.C_CONTINUATION:
            c_offset = C.C_ENTRY_OFFSET_ATR_CONTINUATION * engine.atr14_30m if engine.atr14_30m > 0 else tick
        else:
            c_offset = C.C_ENTRY_OFFSET_ATR * engine.atr14_30m if engine.atr14_30m > 0 else tick
        if direction == Direction.LONG:
            entry_price = round_to_tick(hold_ref + c_offset, tick)
        else:
            entry_price = round_to_tick(hold_ref - c_offset, tick)

        stop_c = stops.compute_initial_stop(
            subtype, direction, entry_price,
            engine.box.box_high, engine.box.box_low, engine.box.box_mid,
            engine.atr14_30m, hold_ref=hold_ref, tick_size=tick,
        )
        qty = sizing.compute_contracts(self._symbol, entry_price, stop_c, self._equity, final_risk_pct)
        qty = self._apply_dd_throttle(qty) if qty >= 1 else qty
        if qty is None:
            return
        if qty < 1:
            return

        await self._submit_order(
            subtype=subtype,
            direction=direction,
            order_type=OrderType.LIMIT,
            price=entry_price,
            stop_price=None,
            qty=qty,
            stop_for_risk=stop_c,
            is_limit=True,
            quality_mult=quality_mult,
            oca_group=oca_group,
        )
        logger.info("Placed %s: price=%.2f qty=%d stop=%.2f", subtype.value, entry_price, qty, stop_c)

    async def _place_market_fallback(
        self, engine: SessionEngineState, direction: Direction,
        close_5m: float, quality_mult: float, exit_tier: ExitTier,
        final_risk_pct: float, now: datetime,
    ) -> None:
        """Place market fallback entry after A orders expire (Section 16)."""
        tick = C.NQ_SPECS[self._symbol]["tick"]
        stop_price = stops.compute_initial_stop(
            EntrySubtype.MARKET_FALLBACK, direction, close_5m,
            engine.box.box_high, engine.box.box_low, engine.box.box_mid,
            engine.atr14_30m, tick_size=tick,
        )
        qty = sizing.compute_contracts(self._symbol, close_5m, stop_price, self._equity, final_risk_pct)
        qty = self._apply_dd_throttle(qty) if qty >= 1 else qty
        if qty is None:
            return
        if qty < 1:
            return

        # Slippage cap
        slip_cap = C.RESCUE_MAX_SLIP_ATR * engine.atr14_30m
        if direction == Direction.LONG:
            limit_price = round_to_tick(close_5m + slip_cap, tick, "up")
        else:
            limit_price = round_to_tick(close_5m - slip_cap, tick, "down")

        await self._submit_order(
            subtype=EntrySubtype.MARKET_FALLBACK,
            direction=direction,
            order_type=OrderType.LIMIT,
            price=limit_price,
            stop_price=None,
            qty=qty,
            stop_for_risk=stop_price,
            tif="IOC",
            quality_mult=quality_mult,
        )
        logger.info("Placed MARKET_FALLBACK: limit=%.2f qty=%d stop=%.2f", limit_price, qty, stop_price)

    # ------------------------------------------------------------------
    # Re-entry evaluation (fix #2)
    # ------------------------------------------------------------------

    async def _evaluate_reentry(
        self, engine: SessionEngineState, session: Session,
        now: datetime, gates_ok: bool,
    ) -> None:
        """Evaluate re-entry after stop-out (Section 18.1)."""
        if not gates_ok:
            return
        if engine.last_stopout_ts is None:
            return

        # Cooldown check (30 minutes)
        elapsed = (now - engine.last_stopout_ts).total_seconds() / 60.0
        if elapsed < C.REENTRY_COOLDOWN_MIN:
            return

        # Stop-out must have been >= -0.5R
        if engine.last_stopout_r < C.REENTRY_MIN_LOSS_R:
            return

        # Breakout must still be valid — re-check qualification
        # If box is still ACTIVE, try to re-activate breakout
        bars = self._bars_30m_session.get(session, {})
        h30 = bars.get("high")
        l30 = bars.get("low")
        c30 = bars.get("close")
        o30 = bars.get("open")
        v30 = bars.get("volume")
        if c30 is None or len(c30) < 20:
            return

        close = float(c30[-1])
        direction = sig.breakout_structural(close, engine.box.box_high, engine.box.box_low)
        if direction is None:
            return

        # Mark re-entry as used (one per box)
        engine.reentry_used = True
        logger.info("Re-entry attempt: direction=%s, stopout_R=%.2f, cooldown=%.1fmin",
                     "LONG" if direction == Direction.LONG else "SHORT",
                     engine.last_stopout_r, elapsed)

        # Attempt full breakout qualification
        self._try_breakout(engine, h30, l30, c30, o30, v30, now)

        # If breakout activated, evaluate entries
        if engine.breakout.active:
            await self._evaluate_entries(engine, session, now)

    # ------------------------------------------------------------------
    # Order submission helper
    # ------------------------------------------------------------------

    async def _submit_order(
        self,
        subtype: EntrySubtype,
        direction: Direction,
        order_type: OrderType,
        price: Optional[float],
        stop_price: Optional[float],
        qty: int,
        stop_for_risk: float,
        oca_group: str = "",
        tif: str = "DAY",
        is_limit: bool = False,
        quality_mult: float = 1.0,
    ) -> Optional[str]:
        """Build and submit an OMS entry order."""
        inst = self._instruments.get(self._symbol)
        if inst is None:
            return None

        side = OrderSide.BUY if direction == Direction.LONG else OrderSide.SELL
        planned_entry = price or stop_price or 0.0
        signal_context = self._entry_signal_context(subtype=subtype, direction=direction)

        entry_request = NQDTCEntryRequest(
            client_order_id=f"{C.STRATEGY_ID}:{subtype.value}:{self._bar_count_5m}:{len(self._working_orders)}",
            symbol=self._symbol,
            subtype=subtype,
            direction=direction,
            qty=qty,
            stop_for_risk=stop_for_risk,
            tif=tif,
            order_type=order_type.name,
            price=price,
            limit_price=price if is_limit else price,
            stop_price=stop_price,
            oca_group=oca_group,
            is_limit=is_limit,
            quality_mult=quality_mult,
            submitted_bar_idx=self._bar_count_5m,
            ttl_bars=C.A_TTL_5M_BARS if subtype in (EntrySubtype.A_RETEST, EntrySubtype.A_LATCH) else 6,
        )
        core_state, actions, events = nqdtc_core_logic.on_bar(
            self._build_core_state(),
            bar_count_5m=self._bar_count_5m,
            bar_ts=self._last_bar_ts or datetime.now(timezone.utc),
            entry_request=entry_request,
        )
        self._apply_core_state(core_state)
        self._apply_core_events(events)
        submit_action = next((action for action in actions if isinstance(action, SubmitEntry)), None)
        if submit_action is None:
            return None

        risk_ctx = RiskContext(
            stop_for_risk=stop_for_risk,
            planned_entry_price=planned_entry,
            risk_dollars=RiskCalculator.compute_order_risk_dollars(
                planned_entry, stop_for_risk, qty, inst.point_value,
            ),
            **signal_context,
        )

        order = OMSOrder(
            strategy_id=C.STRATEGY_ID,
            instrument=inst,
            side=side,
            qty=submit_action.qty,
            order_type=order_type,
            limit_price=submit_action.limit_price,
            stop_price=submit_action.stop_price,
            tif=submit_action.tif,
            role=OrderRole.ENTRY,
            entry_policy=EntryPolicy(ttl_bars=C.A_TTL_5M_BARS if subtype in (EntrySubtype.A_RETEST, EntrySubtype.A_LATCH) else None),
            risk_context=risk_ctx,
            oca_group=oca_group,
            oca_type=1 if oca_group else 0,
        )

        receipt = await self._oms.submit_intent(Intent(
            intent_type=IntentType.NEW_ORDER,
            strategy_id=C.STRATEGY_ID,
            order=order,
        ))

        if receipt.oms_order_id:
            core_state, _, events = nqdtc_core_logic.on_order_update(
                self._build_core_state(),
                NQDTCOrderUpdate(
                    oms_order_id=receipt.oms_order_id,
                    status="accepted",
                    timestamp=datetime.now(timezone.utc),
                    order_role="entry",
                    accepted_entry=entry_request,
                ),
            )
            self._apply_core_state(core_state)
            self._apply_core_events(events)
            return receipt.oms_order_id
        return None

    def _entry_signal_context(
        self,
        *,
        subtype: EntrySubtype,
        direction: Direction,
        bar_ts: datetime | None = None,
    ) -> dict[str, Any]:
        ts = bar_ts or self._last_bar_ts or datetime.now(timezone.utc)
        ts_text = ts.isoformat()
        return {
            "signal_id": f"{self._symbol}:{subtype.value}:{direction.name}:{ts_text}",
            "bar_id": f"{self._symbol}:5m:{ts_text}",
            "exchange_timestamp": ts,
        }

    # ------------------------------------------------------------------
    # Position management (Section 17)
    # ------------------------------------------------------------------

    async def _manage_position(self, engine: SessionEngineState, now: datetime, ts_ny: datetime, session: Session) -> None:
        """Manage open position: TP targets, trailing, stale exit."""
        pos = self._position
        if not pos.open:
            return

        c5 = self._bars_5m.get("close")
        h5 = self._bars_5m.get("high")
        l5 = self._bars_5m.get("low")
        if c5 is None or len(c5) == 0:
            return

        close = float(c5[-1])
        high = float(h5[-1])
        low = float(l5[-1])

        # Update extremes
        if high > pos.highest_since_entry:
            pos.highest_since_entry = high
        if low < pos.lowest_since_entry:
            pos.lowest_since_entry = low
        # Update MFE/MAE in R-multiples
        _risk_per_unit = abs(pos.entry_price - pos.stop_price) if pos.stop_price else 1.0
        if _risk_per_unit > 0:
            if pos.direction == Direction.LONG:
                _current_r = (close - pos.entry_price) / _risk_per_unit
            else:
                _current_r = (pos.entry_price - close) / _risk_per_unit
            pos.peak_mfe_r = max(pos.peak_mfe_r, max(0.0, _current_r))
            pos.peak_mae_r = max(pos.peak_mae_r, max(0.0, -_current_r))

        # Compute open R
        r_points = abs(pos.entry_price - pos.stop_price)
        if r_points <= 0:
            r_points = 1.0
        if pos.direction == Direction.LONG:
            open_r = (close - pos.entry_price) / r_points
        else:
            open_r = (pos.entry_price - close) / r_points

        # Open R on initial stop basis (for chandelier tier + ratchet)
        init_r_points = abs(pos.entry_price - pos.initial_stop_price)
        if init_r_points > 0:
            if pos.direction == Direction.LONG:
                open_r_initial = (close - pos.entry_price) / init_r_points
            else:
                open_r_initial = (pos.entry_price - close) / init_r_points
            pos.peak_r_initial = max(pos.peak_r_initial, open_r_initial)
        else:
            open_r_initial = 0.0

        # Post-TP1 ratchet floor: lock fraction of peak R
        tick = C.NQ_SPECS[self._symbol]["tick"]
        if (pos.profit_funded
                and init_r_points > 0
                and pos.peak_r_initial >= C.RATCHET_THRESHOLD_R):
            if pos.direction == Direction.LONG:
                ratchet_stop = pos.entry_price + C.RATCHET_LOCK_PCT * pos.peak_r_initial * init_r_points
            else:
                ratchet_stop = pos.entry_price - C.RATCHET_LOCK_PCT * pos.peak_r_initial * init_r_points
            ratchet_stop = round_to_tick(ratchet_stop, tick)
            if pos.direction == Direction.LONG and ratchet_stop > pos.stop_price:
                _old = pos.stop_price
                pos.stop_price = ratchet_stop
                pos.stop_source = "RATCHET"
                await self._update_stop(ratchet_stop, old_stop=_old, source="RATCHET")
            elif pos.direction == Direction.SHORT and ratchet_stop < pos.stop_price:
                _old = pos.stop_price
                pos.stop_price = ratchet_stop
                pos.stop_source = "RATCHET"
                await self._update_stop(ratchet_stop, old_stop=_old, source="RATCHET")

        mfe_ratchet_stop = stops.compute_mfe_ratcheted_stop(
            pos.direction,
            pos.entry_price,
            init_r_points,
            pos.peak_r_initial,
            tick,
        )
        if mfe_ratchet_stop is not None:
            if pos.direction == Direction.LONG and mfe_ratchet_stop > pos.stop_price:
                _old = pos.stop_price
                pos.stop_price = mfe_ratchet_stop
                pos.stop_source = "MFE_RATCHET"
                await self._update_stop(mfe_ratchet_stop, old_stop=_old, source="MFE_RATCHET")
            elif pos.direction == Direction.SHORT and mfe_ratchet_stop < pos.stop_price:
                _old = pos.stop_price
                pos.stop_price = mfe_ratchet_stop
                pos.stop_source = "MFE_RATCHET"
                await self._update_stop(mfe_ratchet_stop, old_stop=_old, source="MFE_RATCHET")

        # TP targets (use initial stop distance, not migrated stop)
        tp_r_points = init_r_points if init_r_points > 0 else r_points
        for i, tp in enumerate(pos.tp_levels):
            if tp.filled:
                continue
            # If TP1-only cap is active, skip TP2+ (fix #3)
            if pos.tp1_only_cap and i > 0:
                break
            if pos.direction == Direction.LONG:
                tp_price = pos.entry_price + tp.r_target * tp_r_points
                if close >= tp_price:
                    await self._exit_partial(pos, tp, close, engine)
            else:
                tp_price = pos.entry_price - tp.r_target * tp_r_points
                if close <= tp_price:
                    await self._exit_partial(pos, tp, close, engine)

        # Profit-funded BE move (after TP1)
        if pos.profit_funded and not pos.runner_active:
            atr14_5m = ind.atr(h5, l5, c5, C.ATR14_5M_PERIOD)
            atr5 = float(atr14_5m[-1]) if not np.isnan(atr14_5m[-1]) else 0.0
            be_stop = stops.compute_be_stop(pos.direction, pos.entry_price, atr5, C.NQ_SPECS[self._symbol]["tick"])
            if pos.direction == Direction.LONG and be_stop > pos.stop_price:
                _old = pos.stop_price
                pos.stop_price = be_stop
                pos.stop_source = "BE"
                await self._update_stop(pos.stop_price, old_stop=_old, source="BE")
            elif pos.direction == Direction.SHORT and be_stop < pos.stop_price:
                _old = pos.stop_price
                pos.stop_price = be_stop
                pos.stop_source = "BE"
                await self._update_stop(pos.stop_price, old_stop=_old, source="BE")

        # Measured move check
        if not pos.mm_reached:
            if pos.direction == Direction.LONG and close >= pos.mm_level:
                pos.mm_reached = True
            elif pos.direction == Direction.SHORT and close <= pos.mm_level:
                pos.mm_reached = True

        # Chandelier runner trail (1H, Section 17.5)
        if pos.runner_active:
            h1 = self._bars_1h.get("high")
            l1 = self._bars_1h.get("low")
            c1 = self._bars_1h.get("close")
            if h1 is not None and len(h1) > 14:
                atr14_1h = ind.atr(h1, l1, c1, C.ATR14_1H_PERIOD)
                lookback, mult = stops.chandelier_params(open_r_initial, pos.mm_reached)
                if pos.direction == Direction.LONG:
                    trail = ind.chandelier_long(h1, atr14_1h, lookback, mult)
                    trail = round_to_tick(trail, C.NQ_SPECS[self._symbol]["tick"], "down")
                    if trail > pos.chandelier_trail:
                        pos.chandelier_trail = trail
                        if trail > pos.stop_price:
                            _old = pos.stop_price
                            pos.stop_price = trail
                            pos.stop_source = "CHANDELIER"
                            await self._update_stop(trail, old_stop=_old, source="CHANDELIER")
                else:
                    trail = ind.chandelier_short(l1, atr14_1h, lookback, mult)
                    trail = round_to_tick(trail, C.NQ_SPECS[self._symbol]["tick"], "up")
                    if trail < pos.chandelier_trail or pos.chandelier_trail == 0:
                        pos.chandelier_trail = trail
                        if trail < pos.stop_price:
                            _old = pos.stop_price
                            pos.stop_price = trail
                            pos.stop_source = "CHANDELIER"
                            await self._update_stop(trail, old_stop=_old, source="CHANDELIER")

        # Overnight bridge check (fix #9)
        t_ny = ts_ny.hour * 60 + ts_ny.minute
        rth_close_min = _minutes(C.RTH_END_H, C.RTH_END_M)
        if t_ny == rth_close_min and not pos.stale_bridge_extended:
            if stops.overnight_bridge_eligible(
                close, pos.box_high_at_entry, pos.box_low_at_entry,
                pos.direction, self._regime.regime_4h.value, self._regime.trend_dir_4h,
            ):
                pos.stale_bridge_extended = True
                pos.stale_bridge_extra_bars = C.OVERNIGHT_BRIDGE_EXTRA_BARS
                logger.info("Overnight bridge: extending stale timer by %d bars", C.OVERNIGHT_BRIDGE_EXTRA_BARS)

        # Stale exit (Section 17.6)
        mode_str = engine.mode.value
        if stops.stale_exit_check(pos.bars_since_entry_30m, open_r, mode_str, pos.stale_bridge_extra_bars, tp1_filled=pos.profit_funded):
            logger.info("Stale exit: bars=%d R=%.2f", pos.bars_since_entry_30m, open_r)
            await self._flatten(engine, open_r, reason="STALE")

        # News blackout management (Section 4)
        if pos.open and self._news_flatten_imminent(now):
            if not pos.profit_funded:
                logger.info("News flatten: not profit-funded")
                await self._flatten(engine, open_r, reason="NEWS_BLACKOUT")
            else:
                tick = C.NQ_SPECS[self._symbol]["tick"]
                if pos.direction == Direction.LONG:
                    be_tick = pos.entry_price + tick
                else:
                    be_tick = pos.entry_price - tick
                if (pos.direction == Direction.LONG and be_tick > pos.stop_price) or \
                   (pos.direction == Direction.SHORT and be_tick < pos.stop_price):
                    _old = pos.stop_price
                    pos.stop_price = round_to_tick(be_tick, tick)
                    pos.stop_source = "NEWS_BE"
                    await self._update_stop(pos.stop_price, old_stop=_old, source="NEWS_BE")
                    logger.info("News blackout: tightened stop to BE±1tick=%.2f", pos.stop_price)

        # v7: Max loss cap — force exit if unrealized loss exceeds threshold (complements min_stop_distance)
        if pos.open:
            init_r_points = abs(pos.entry_price - pos.initial_stop_price)
            if init_r_points > 0:
                if pos.direction == Direction.LONG:
                    init_open_r = (close - pos.entry_price) / init_r_points
                else:
                    init_open_r = (pos.entry_price - close) / init_r_points
                if init_open_r <= C.MAX_LOSS_CAP_R:
                    logger.info("Max loss cap: R=%.2f <= %.1f, force exit", init_open_r, C.MAX_LOSS_CAP_R)
                    await self._flatten(engine, open_r, reason="DAILY_LOSS_CAP")
                    return

    async def _exit_partial(self, pos: PositionState, tp: TPLevel, close: float, engine: SessionEngineState) -> None:
        """Exit partial qty at TP level."""
        exit_qty = min(tp.qty, pos.qty_open)
        if exit_qty <= 0:
            return
        tp.filled = True

        inst = self._instruments.get(self._symbol)
        if inst is None:
            return

        exit_side = OrderSide.SELL if pos.direction == Direction.LONG else OrderSide.BUY
        order = OMSOrder(
            strategy_id=C.STRATEGY_ID,
            instrument=inst,
            side=exit_side,
            qty=exit_qty,
            order_type=OrderType.MARKET,
            role=OrderRole.TP,
        )
        receipt = await self._oms.submit_intent(Intent(
            intent_type=IntentType.NEW_ORDER,
            strategy_id=C.STRATEGY_ID,
            order=order,
        ))

        pos.qty_open -= exit_qty
        logger.info("TP%.1fR: exit %d contracts (remaining=%d)", tp.r_target, exit_qty, pos.qty_open)

        # First TP → profit-funded
        if not pos.profit_funded:
            pos.profit_funded = True
            logger.info("Position profit-funded after TP1")

        # Activate runner when all TPs filled
        if not pos.runner_active and all(tp.filled for tp in pos.tp_levels) and pos.qty_open > 0:
            pos.runner_active = True
            logger.info("Runner activated: %d contracts remaining", pos.qty_open)

        # Early chandelier: activate runner immediately after TP1
        if not pos.runner_active and pos.profit_funded:
            pos.runner_active = True
            logger.info("Early chandelier: runner activated after TP1 (%d contracts)", pos.qty_open)

        # Check if fully exited
        if pos.qty_open <= 0:
            # Track for trend cycling (fix #18)
            engine.last_profitable_exit_dir = pos.direction
            self._consec_losses = 0  # reset loss streak on profitable exit
            self._accumulate_realized_pnl(0.0)  # qty_open=0, all profit from TP R-targets
            self._clear_position()

    # ------------------------------------------------------------------
    # Order management helpers
    # ------------------------------------------------------------------

    def _has_active_a_orders(self) -> bool:
        return any(
            wo.subtype in (EntrySubtype.A_RETEST, EntrySubtype.A_LATCH)
            for wo in self._working_orders
        )

    def _a_orders_expired(self) -> bool:
        """Check if A orders have expired (TTL elapsed)."""
        for wo in self._working_orders:
            if wo.subtype in (EntrySubtype.A_RETEST, EntrySubtype.A_LATCH):
                return False  # still active
        return True  # no A orders → they expired or were never placed

    async def _force_flatten_for_roll(self, engine: SessionEngineState, now: datetime) -> bool:
        reason = roll_force_flatten_reason(
            self._instruments.get(self._symbol) or self._symbol,
            as_of=now,
        )
        if not reason:
            return False
        close = self._latest_close_5m(default=self._position.entry_price)
        open_r = self._position_open_r(close)
        self._record_decision("ROLL_FORCE_FLATTEN", {"reason": reason, "open_r": open_r})
        logger.critical("NQDTC forcing flatten for roll safety: %s", reason)
        await self._flatten(engine, open_r, reason="ROLL_SAFETY")
        return True

    def _latest_close_5m(self, *, default: float = 0.0) -> float:
        close = self._bars_5m.get("close")
        if close is None or len(close) == 0:
            return float(default)
        return float(close[-1])

    def _position_open_r(self, close: float) -> float:
        pos = self._position
        r_pts = abs(pos.entry_price - pos.initial_stop_price)
        if r_pts <= 0:
            return 0.0
        if pos.direction == Direction.LONG:
            return (close - pos.entry_price) / r_pts
        if pos.direction == Direction.SHORT:
            return (pos.entry_price - close) / r_pts
        return 0.0

    async def _manage_working_orders(self, engine: SessionEngineState) -> None:
        """Cancel expired or invalidated working orders."""
        reason = roll_blackout_reason(
            self._instruments.get(self._symbol) or self._symbol,
            as_of=self._last_bar_ts or datetime.now(timezone.utc),
        )
        if reason and self._working_orders:
            for wo in list(self._working_orders):
                await self._cancel_order(wo.oms_order_id)
            cancelled = len(self._working_orders)
            self._working_orders.clear()
            self._a_fallback_eligible = False
            self._record_decision("ENTRY_CANCELLED_BY_ROLL_BLACKOUT", {"reason": reason, "count": cancelled})
            logger.warning("Cancelled %d NQDTC working entries during roll blackout: %s", cancelled, reason)
            return

        c5 = self._bars_5m.get("close")
        if c5 is None or len(c5) == 0:
            return
        close = float(c5[-1])

        to_remove = []
        for wo in self._working_orders:
            # TTL expiry
            bars_elapsed = self._bar_count_5m - wo.submitted_bar_idx
            if bars_elapsed >= wo.ttl_bars:
                await self._cancel_order(wo.oms_order_id)
                to_remove.append(wo)
                continue

            # Regime re-check: cancel if direction now hard-blocked
            d_sup, d_opp = sig.classify_daily_support(
                self._bars_daily.get("ema50", np.array([])),
                self._bars_daily.get("atr14", np.array([])),
                wo.direction,
            )
            if sig.regime_hard_block(
                self._regime.regime_4h.value, self._regime.trend_dir_4h,
                wo.direction, d_opp,
            ):
                await self._cancel_order(wo.oms_order_id)
                to_remove.append(wo)
                continue

            # A cancel check
            if wo.subtype in (EntrySubtype.A_RETEST, EntrySubtype.A_LATCH):
                if sig.entry_a_cancel_check(
                    close, engine.box.box_high, engine.box.box_low,
                    engine.atr14_30m, wo.direction,
                ):
                    await self._cancel_order(wo.oms_order_id)
                    # Cancel sibling too
                    for sibling in self._working_orders:
                        if sibling.oca_group == wo.oca_group and sibling.oms_order_id != wo.oms_order_id:
                            await self._cancel_order(sibling.oms_order_id)
                            to_remove.append(sibling)
                    to_remove.append(wo)

        # Check if removed orders include A types
        had_a_removed = any(
            wo.subtype in (EntrySubtype.A_RETEST, EntrySubtype.A_LATCH)
            for wo in to_remove
        )

        for wo in to_remove:
            if wo in self._working_orders:
                self._working_orders.remove(wo)

        # Enable market fallback if all A orders are now gone
        if had_a_removed:
            has_remaining_a = any(
                wo.subtype in (EntrySubtype.A_RETEST, EntrySubtype.A_LATCH)
                for wo in self._working_orders
            )
            if not has_remaining_a:
                self._a_fallback_eligible = True

    async def _cancel_order(self, oms_order_id: str) -> None:
        try:
            await self._oms.submit_intent(Intent(
                intent_type=IntentType.CANCEL_ORDER,
                strategy_id=C.STRATEGY_ID,
                target_oms_order_id=oms_order_id,
            ))
        except Exception as e:
            logger.warning("Cancel failed for %s: %s", oms_order_id, e)

    async def _reject_filled_entry(
        self,
        filled_order_id: str,
        sibling_order_ids: list[str],
        *,
        reason: str,
        details: dict[str, Any],
    ) -> None:
        for sibling_order_id in sibling_order_ids:
            await self._cancel_order(sibling_order_id)

        remove_ids = {filled_order_id, *sibling_order_ids}
        self._working_orders = [
            order for order in self._working_orders if order.oms_order_id not in remove_ids
        ]
        self._record_decision("ENTRY_FILL_REJECTED", {"reason": reason, **details})

        try:
            receipt = await self._oms.submit_intent(Intent(
                intent_type=IntentType.FLATTEN,
                strategy_id=C.STRATEGY_ID,
                instrument_symbol=self._symbol,
            ))
            if receipt and receipt.oms_order_id:
                self._last_flatten_oms_id = receipt.oms_order_id
        except Exception:
            logger.exception("Failed emergency flatten after rejected filled entry %s", filled_order_id)

    async def _update_stop(self, new_stop: float, old_stop: float = 0.0, source: str = "") -> None:
        """Update protective stop order."""
        if self._position.stop_oms_order_id:
            await self._oms.submit_intent(Intent(
                intent_type=IntentType.REPLACE_ORDER,
                strategy_id=C.STRATEGY_ID,
                target_oms_order_id=self._position.stop_oms_order_id,
                new_stop_price=new_stop,
            ))
        if self._kit and old_stop > 0 and old_stop != new_stop:
            adj_type = {"RATCHET": "trailing", "BE": "breakeven", "CHANDELIER": "trailing",
                        "NEWS_BE": "breakeven"}.get(source, "trailing")
            self._kit.log_stop_adjustment(
                trade_id=self._position.trade_id or f"NQDTC-{self._symbol}",
                symbol=self._symbol, old_stop=old_stop, new_stop=new_stop,
                adjustment_type=adj_type, trigger=source.lower() or "nqdtc_trail",
            )

    async def _flatten(self, engine: Optional[SessionEngineState] = None, open_r: float = 0.0, reason: str = "FLATTEN") -> None:
        """Flatten entire position."""
        pos = self._position
        direction = pos.direction

        # Capture order ID for terminal-event detection (Rec 1/3)
        receipt = await self._oms.submit_intent(Intent(
            intent_type=IntentType.FLATTEN,
            strategy_id=C.STRATEGY_ID,
            instrument_symbol=self._symbol,
        ))
        self._last_flatten_oms_id = receipt.oms_order_id if receipt and receipt.oms_order_id else None

        # Record peak MFE R on breakout state for C_continuation gating
        if engine is not None and engine.breakout.active:
            engine.breakout.last_trade_peak_r = max(
                engine.breakout.last_trade_peak_r, pos.peak_r_initial,
            )

        # Track stop-out for re-entry (fix #2)
        if engine is not None and pos.open and open_r < 0:
            engine.last_stopout_r = open_r
            engine.last_stopout_ts = datetime.now(timezone.utc)
            engine.reentry_allowed = True
            logger.info("Stop-out tracked: R=%.2f", open_r)

        # Track profitable exit direction for trend cycling (fix #18)
        if engine is not None and open_r > 0:
            engine.last_profitable_exit_dir = direction

        # Consecutive-loss cooldown tracking
        if open_r <= 0:
            self._consec_losses += 1
            if self._consec_losses >= C.LOSS_STREAK_THRESHOLD:
                self._cooldown_bars = C.LOSS_STREAK_SKIP_BARS
                logger.info("Cooldown activated: %d consecutive losses", self._consec_losses)
        else:
            self._consec_losses = 0

        if self._last_flatten_oms_id and self._instr_trade_id:
            r_pts = abs(pos.entry_price - pos.initial_stop_price)
            if pos.direction == Direction.LONG:
                expected_exit = pos.entry_price + open_r * r_pts if r_pts > 0 else pos.entry_price
            else:
                expected_exit = pos.entry_price - open_r * r_pts if r_pts > 0 else pos.entry_price
            self._pending_flatten_instrumentation[self._last_flatten_oms_id] = {
                "trade_id": self._instr_trade_id,
                "reason": reason,
                "expected_exit_price": expected_exit,
                "mfe_r": pos.peak_mfe_r,
                "mae_r": pos.peak_mae_r,
                "mfe_price": pos.highest_since_entry if pos.direction == Direction.LONG else pos.lowest_since_entry,
                "mae_price": pos.lowest_since_entry if pos.direction == Direction.LONG else pos.highest_since_entry,
                "session_transitions": self._session_transitions or None,
            }

        self._accumulate_realized_pnl(open_r)
        self._clear_position()

    def _dd_tier_name(self) -> str:
        mult = getattr(self._throttle, 'dd_size_mult', 1.0)
        if mult >= 1.0:
            return "full"
        elif mult >= 0.5:
            return "half"
        elif mult >= 0.25:
            return "quarter"
        return "halt"

    def _build_gate_filter_decisions(self, now_ny, r_points: float) -> list[dict]:
        """Build structured filter decisions from current gate state."""
        decisions = []
        MIN_STOP_DISTANCE = 3.0

        # Daily risk halt
        realized_r = getattr(self._daily_risk, 'realized_pnl_R', 0.0)
        decisions.append({
            "filter_name": "daily_risk_halted",
            "threshold": C.DAILY_STOP_R,
            "actual_value": round(realized_r, 3),
            "passed": not self._daily_risk.halted,
            "margin_pct": round((realized_r - C.DAILY_STOP_R) / abs(C.DAILY_STOP_R) * 100, 1)
                if C.DAILY_STOP_R != 0 else None,
        })

        # Min stop distance
        decisions.append({
            "filter_name": "MIN_STOP_DISTANCE",
            "threshold": MIN_STOP_DISTANCE,
            "actual_value": round(r_points, 2),
            "passed": r_points >= MIN_STOP_DISTANCE,
            "margin_pct": round((r_points - MIN_STOP_DISTANCE) / abs(MIN_STOP_DISTANCE) * 100, 1),
        })

        # v7: Max stop width
        decisions.append({
            "filter_name": "MAX_STOP_WIDTH",
            "threshold": C.MAX_STOP_WIDTH_PTS,
            "actual_value": round(r_points, 2),
            "passed": r_points <= C.MAX_STOP_WIDTH_PTS,
            "margin_pct": round((C.MAX_STOP_WIDTH_PTS - r_points) / C.MAX_STOP_WIDTH_PTS * 100, 1),
        })

        # Hour blocks
        hour = now_ny.hour
        decisions.append({
            "filter_name": "BLOCK_06_ET",
            "threshold": 0.0,
            "actual_value": hour,
            "passed": not (C.BLOCK_06_ET and hour == 6),
            "margin_pct": None,
        })
        decisions.append({
            "filter_name": "BLOCK_12_ET",
            "threshold": 0.0,
            "actual_value": hour,
            "passed": not (C.BLOCK_12_ET and hour == 12),
            "margin_pct": None,
        })

        return decisions

    def _get_bid_ask(self) -> tuple[float, float] | None:
        """Return (bid, ask) from IB tickers, or None if unavailable."""
        try:
            for t in self._ib.ib.tickers():
                if t.contract and t.contract.symbol == "NQ":
                    if t.bid > 0 and t.ask > 0:
                        return (t.bid, t.ask)
        except Exception:
            pass
        return None

    def _clear_position(self) -> None:
        """Reset position state."""
        self._position = PositionState()

    def _accumulate_realized_pnl(self, exit_r: float) -> None:
        """Accumulate realized trade PnL (in R) into daily/weekly/monthly trackers.

        Computes a blended R-PnL across TP partials (at their R-targets) and
        the final exit (at *exit_r*), weighted by contract counts.
        """
        pos = self._position
        if not pos.open or pos.qty <= 0:
            return
        # Weighted R: remaining contracts at exit_r + filled TPs at their targets
        total_r_qty = exit_r * pos.qty_open
        for tp in pos.tp_levels:
            if tp.filled:
                total_r_qty += tp.r_target * tp.qty
        blended_r = total_r_qty / pos.qty

        self._daily_risk.realized_pnl_R += blended_r
        self._daily_risk.weekly_realized_R += blended_r
        self._daily_risk.monthly_realized_R += blended_r
        self._throttle.update_equity(self._equity)
        self._throttle.record_trade_close(blended_r)
        logger.info("Risk tracking: trade R=%.2f (daily=%.2f weekly=%.2f monthly=%.2f)",
                     blended_r, self._daily_risk.realized_pnl_R,
                     self._daily_risk.weekly_realized_R, self._daily_risk.monthly_realized_R)

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    async def _process_events(self) -> None:
        """Process OMS events (fills, cancels, rejects)."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=5.0)
                await self._handle_event(event)
            except asyncio.TimeoutError:
                pass  # Working orders now managed in main loop (fix #17)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error processing event")

    async def _handle_event(self, event: Any) -> None:
        """Route OMS event to handler."""
        etype = getattr(event, "event_type", None)
        if etype == OMSEventType.FILL:
            await self._on_fill(event)
        elif etype in (OMSEventType.ORDER_CANCELLED, OMSEventType.ORDER_EXPIRED,
                       OMSEventType.ORDER_REJECTED):
            await self._on_order_update(event)

    async def _on_fill(self, event: Any) -> None:
        """Handle fill event."""
        oms_id = getattr(event, "oms_order_id", "")
        payload = event.payload or {}
        price = payload.get("price", 0.0)
        qty = payload.get("qty", 0)
        fill_time = getattr(event, "timestamp", None) or datetime.now(timezone.utc)
        if fill_time.tzinfo is None:
            fill_time = fill_time.replace(tzinfo=timezone.utc)

        # Flatten fill confirmation -- broker executed the pre-booked exit
        if self._last_flatten_oms_id and oms_id == self._last_flatten_oms_id:
            pending = self._pending_flatten_instrumentation.pop(oms_id, {})
            trade_id = str(pending.get("trade_id") or self._instr_trade_id or "")
            if self._kit.active and trade_id:
                try:
                    self._kit.log_exit(
                        trade_id=trade_id,
                        exit_price=price,
                        exit_reason=str(pending.get("reason") or "FLATTEN"),
                        expected_exit_price=pending.get("expected_exit_price") or price,
                        mfe_r=pending.get("mfe_r"),
                        mae_r=pending.get("mae_r"),
                        mfe_price=pending.get("mfe_price"),
                        mae_price=pending.get("mae_price"),
                        session_transitions=pending.get("session_transitions"),
                        **fill_runtime_refs(oms_id, payload, fill_qty=qty, is_exit=True),
                    )
                    _ba = self._get_bid_ask()
                    self._kit.on_orderbook_context(
                        pair=self._symbol,
                        best_bid=_ba[0] if _ba else price,
                        best_ask=_ba[1] if _ba else price,
                        trade_context="exit",
                        related_trade_id=trade_id,
                    )
                    if self._instr_trade_id == trade_id:
                        self._instr_trade_id = ""
                except Exception:
                    pass
            self._last_flatten_oms_id = None
            return

        # Find matching working order
        wo = None
        for w in self._working_orders:
            if w.oms_order_id == oms_id:
                wo = w
                break

        if wo is None:
            # Check if this is a stop fill for our position
            if self._position.open and self._position.stop_oms_order_id == oms_id:
                init_r_points = abs(self._position.entry_price - self._position.initial_stop_price)
                if init_r_points > 0:
                    if self._position.direction == Direction.LONG:
                        stop_r = (price - self._position.entry_price) / init_r_points
                    else:
                        stop_r = (self._position.entry_price - price) / init_r_points
                else:
                    stop_r = -1.0
                if stop_r <= 0:
                    self._consec_losses += 1
                    if self._consec_losses >= C.LOSS_STREAK_THRESHOLD:
                        self._cooldown_bars = C.LOSS_STREAK_SKIP_BARS
                        logger.info("Cooldown activated: %d consecutive losses (stop fill)", self._consec_losses)
                else:
                    self._consec_losses = 0
                logger.info("Stop fill: price=%.2f R=%.2f", price, stop_r)
                if self._kit.active and self._instr_trade_id:
                    try:
                        stop_reason = f"STOP_{self._position.stop_source}"
                        self._kit.log_exit(
                            trade_id=self._instr_trade_id,
                            exit_price=price,
                            exit_reason=stop_reason,
                            expected_exit_price=self._position.stop_price,
                            mfe_r=self._position.peak_mfe_r,
                            mae_r=self._position.peak_mae_r,
                            mfe_price=self._position.highest_since_entry if self._position.direction == Direction.LONG else self._position.lowest_since_entry,
                            mae_price=self._position.lowest_since_entry if self._position.direction == Direction.LONG else self._position.highest_since_entry,
                            session_transitions=self._session_transitions or None,
                            **fill_runtime_refs(oms_id, payload, fill_qty=qty, is_exit=True),
                        )
                        _ba = self._get_bid_ask()
                        self._kit.on_orderbook_context(
                            pair=self._symbol,
                            best_bid=_ba[0] if _ba else price,
                            best_ask=_ba[1] if _ba else price,
                            trade_context="exit",
                            related_trade_id=self._instr_trade_id,
                        )
                        self._instr_trade_id = ""
                    except Exception:
                        pass
                self._accumulate_realized_pnl(stop_r)
                core_state, _, events = nqdtc_core_logic.on_fill(
                    self._build_core_state(),
                    NQDTCFill(
                        oms_order_id=oms_id,
                        fill_price=price,
                        fill_qty=qty,
                        fill_time=fill_time,
                        exit_type="stop",
                    ),
                )
                self._apply_core_state(core_state)
                self._apply_core_events(events)
                self._last_flatten_oms_id = None
            return

        sibling_order_ids = [
            sibling.oms_order_id
            for sibling in self._working_orders
            if wo.oca_group and sibling.oca_group == wo.oca_group and sibling.oms_order_id != oms_id
        ]

        # Open position
        inst = self._instruments.get(self._symbol)
        tick = inst.tick_size if inst else 0.25

        # Get active session engine for box state
        now_ny = _to_ny(fill_time)
        session = session_type(now_ny)
        engine = self._engines[session]

        # Use pre-computed stop from working order (matches backtest)
        stop_price = wo.stop_for_risk
        r_points = abs(price - stop_price)

        # Min stop distance gate: reject pathologically tight stops
        MIN_STOP_DISTANCE = 3.0
        if r_points < MIN_STOP_DISTANCE:
            logger.info("Min stop gate: stop_dist=%.2f < %.1f, rejecting fill", r_points, MIN_STOP_DISTANCE)
            self._log_missed(wo.direction, wo.subtype.value, oms_id, "MIN_STOP_DISTANCE", f"r={r_points:.2f}", signal_strength=wo.quality_mult,
                            filter_decisions=[{"filter_name": "MIN_STOP_DISTANCE", "threshold": MIN_STOP_DISTANCE, "actual_value": r_points, "passed": False}])
            await self._reject_filled_entry(
                oms_id,
                sibling_order_ids,
                reason="MIN_STOP_DISTANCE",
                details={"actual_value": r_points, "threshold": MIN_STOP_DISTANCE, "subtype": wo.subtype.value},
            )
            return

        # v7: Max stop width gate: reject excessively wide stops
        if r_points > C.MAX_STOP_WIDTH_PTS:
            logger.info("Max stop width: stop_dist=%.2f > %.1f, rejecting", r_points, C.MAX_STOP_WIDTH_PTS)
            self._log_missed(wo.direction, wo.subtype.value, oms_id, "MAX_STOP_WIDTH", f"r={r_points:.2f}", signal_strength=wo.quality_mult,
                            filter_decisions=[{"filter_name": "MAX_STOP_WIDTH", "threshold": C.MAX_STOP_WIDTH_PTS, "actual_value": r_points, "passed": False}])
            await self._reject_filled_entry(
                oms_id,
                sibling_order_ids,
                reason="MAX_STOP_WIDTH",
                details={"actual_value": r_points, "threshold": C.MAX_STOP_WIDTH_PTS, "subtype": wo.subtype.value},
            )
            return

        # Block fills during 06:00 ET hour (pre-European-open, WR=39%)
        if C.BLOCK_06_ET and now_ny.hour == 6:
            logger.info("06:00 ET fill block: rejecting entry fill at %s", now_ny.strftime("%H:%M"))
            self._log_missed(wo.direction, wo.subtype.value, oms_id, "BLOCK_06_ET", "fill at 06:xx ET", signal_strength=wo.quality_mult,
                            filter_decisions=[{"filter_name": "BLOCK_06_ET", "threshold": 6, "actual_value": now_ny.hour, "passed": False}])
            await self._reject_filled_entry(
                oms_id,
                sibling_order_ids,
                reason="BLOCK_06_ET",
                details={"actual_value": now_ny.hour, "threshold": 6, "subtype": wo.subtype.value},
            )
            return

        # Block fills during 12:00 ET hour (17% WR, outlier-dependent)
        if C.BLOCK_12_ET and now_ny.hour == 12:
            logger.info("12:00 ET fill block: rejecting entry fill at %s", now_ny.strftime("%H:%M"))
            self._log_missed(wo.direction, wo.subtype.value, oms_id, "BLOCK_12_ET", "fill at 12:xx ET", signal_strength=wo.quality_mult,
                            filter_decisions=[{"filter_name": "BLOCK_12_ET", "threshold": 12, "actual_value": now_ny.hour, "passed": False}])
            await self._reject_filled_entry(
                oms_id,
                sibling_order_ids,
                reason="BLOCK_12_ET",
                details={"actual_value": now_ny.hour, "threshold": 12, "subtype": wo.subtype.value},
            )
            return

        # Determine exit tier using actual quality_mult from working order (fix #5)
        daily_supports, daily_opposes = sig.classify_daily_support(
            self._bars_daily.get("ema50", np.array([])),
            self._bars_daily.get("atr14", np.array([])),
            wo.direction,
        )
        composite = sig.compute_composite_regime(
            self._regime.regime_4h.value, self._regime.trend_dir_4h, wo.direction,
            daily_supports, daily_opposes,
        )
        exit_tier = stops.determine_exit_tier(composite.value, wo.quality_mult)

        if (
            (C.BLOCK_NEUTRAL_REGIME and composite == CompositeRegime.NEUTRAL)
            or (C.BLOCK_ALIGNED_REGIME and composite == CompositeRegime.ALIGNED)
            or (C.BLOCK_CAUTION_REGIME and composite == CompositeRegime.CAUTION)
        ):
            logger.info("Composite regime fill block: %s", composite.value)
            await self._reject_filled_entry(
                oms_id,
                sibling_order_ids,
                reason="REGIME_COMPOSITE_BLOCK",
                details={"composite": composite.value, "subtype": wo.subtype.value},
            )
            return

        # Check TP1-only cap at entry time (fix #3)
        tp1_cap = stops.should_cap_tp1_only(engine.mode.value, self._regime.regime_4h.value)

        # Build TP levels
        tp_levels = stops.compute_tp_levels(
            wo.direction, price, r_points, exit_tier, qty, tick,
        )
        if tp1_cap and len(tp_levels) > 1:
            tp_levels = tp_levels[:1]

        core_state, actions, events = nqdtc_core_logic.on_fill(
            self._build_core_state(),
            NQDTCFill(
                oms_order_id=oms_id,
                fill_price=price,
                fill_qty=qty,
                fill_time=fill_time,
                entry_context=NQDTCEntryFillContext(
                    exit_tier=exit_tier,
                    tp_levels=tp_levels,
                    mm_level=engine.breakout.mm_level,
                    mm_reached=engine.breakout.mm_reached if engine.breakout.active else False,
                    box_high_at_entry=engine.box.box_high,
                    box_low_at_entry=engine.box.box_low,
                    box_mid_at_entry=engine.box.box_mid,
                    entry_session=session,
                    tp1_only_cap=tp1_cap,
                    r_dollars=r_points * (inst.point_value if inst else 20.0) * qty,
                ),
            ),
        )
        self._apply_core_state(core_state)
        self._apply_core_events(events)
        if not self._position.open:
            logger.error("Shared core did not open NQDTC position for filled entry %s", oms_id)
            await self._reject_filled_entry(
                oms_id,
                sibling_order_ids,
                reason="CORE_ENTRY_FILL_ERROR",
                details={"oms_order_id": oms_id, "fill_price": price, "qty": qty, "subtype": wo.subtype.value},
            )
            return
        self._last_fill_time = fill_time
        self._position.symbol = self._symbol
        self._position.R_dollars = r_points * (inst.point_value if inst else 20.0) * qty

        for sibling_order_id in sibling_order_ids:
            await self._cancel_order(sibling_order_id)
            self._working_orders = [
                order for order in self._working_orders if order.oms_order_id != sibling_order_id
            ]

        # Track continuation fills per breakout
        if wo.subtype == EntrySubtype.C_CONTINUATION and engine.breakout.active:
            engine.breakout.continuation_fills += 1

        for action in actions:
            if isinstance(action, SubmitExit):
                await self._place_protective_stop(action.stop_price or stop_price, action.qty, wo.direction)

        # Telemetry (fix #16)
        fill_R_dollars = r_points * (inst.point_value if inst else 20.0) * qty
        self._log_telemetry("fill", engine, wo.direction,
                            subtype=wo.subtype.value, price=price, qty=qty,
                            stop=stop_price, tier=exit_tier.value,
                            quality_mult=wo.quality_mult, tp1_cap=tp1_cap,
                            fee_R=sizing.fee_R_estimate(self._symbol, fill_R_dollars))

        logger.info(
            "FILL %s %s: price=%.2f qty=%d stop=%.2f tier=%s qm=%.2f",
            wo.subtype.value, "LONG" if wo.direction == Direction.LONG else "SHORT",
            price, qty, stop_price, exit_tier.value, wo.quality_mult,
        )

        if self._kit.active:
            try:
                self._instr_trade_id = oms_id
                inst_obj = self._instruments.get(self._symbol)
                pv = inst_obj.point_value if inst_obj else 20.0
                config_snapshot = snapshot_config_module(strategy_config)

                # Execution cascade timestamps (#16)
                signal_detected_at = self._cascade_ts.pop("_last_eval", now_ny)
                fill_received_at = now_ny
                exec_ts = {
                    "signal_detected_at": signal_detected_at.isoformat(),
                    "fill_received_at": fill_received_at.isoformat(),
                    "cascade_duration_ms": round(
                        (fill_received_at - signal_detected_at).total_seconds() * 1000
                    ),
                }

                # Clear session transitions for new position (#17)
                self._session_transitions.clear()

                # Capture portfolio state at entry (G4)
                portfolio_state = None
                try:
                    risk_state = await self._oms.get_portfolio_risk()
                    portfolio_state = {
                        "total_exposure_r": risk_state.open_risk_R,
                        "daily_realized_pnl": risk_state.daily_realized_pnl,
                        "daily_realized_r": risk_state.daily_realized_R,
                        "weekly_realized_pnl": risk_state.weekly_realized_pnl,
                        "weekly_realized_r": risk_state.weekly_realized_R,
                        "open_risk_r": risk_state.open_risk_R,
                        "pending_entry_risk_r": risk_state.pending_entry_risk_R,
                        "halted": risk_state.halted,
                    }
                except Exception:
                    portfolio_state = None

                self._kit.log_entry(
                    trade_id=oms_id,
                    pair=self._symbol,
                    side="LONG" if wo.direction == Direction.LONG else "SHORT",
                    entry_price=price,
                    position_size=qty,
                    position_size_quote=qty * price * pv,
                    entry_signal=wo.subtype.value,
                    entry_signal_id=oms_id,
                    entry_signal_strength=wo.quality_mult,
                    expected_entry_price=wo.expected_fill_price or wo.price,
                    strategy_params={
                        "stop": stop_price,
                        "subtype": wo.subtype.value,
                        "exit_tier": exit_tier.value,
                        "quality_mult": wo.quality_mult,
                        **config_snapshot,
                    },
                    signal_factors=[
                        {"factor_name": "quality_mult", "factor_value": wo.quality_mult,
                         "threshold": 0.0, "contribution": wo.quality_mult},
                        {"factor_name": "subtype", "factor_value": wo.subtype.value,
                         "threshold": "A", "contribution": "entry_type_quality"},
                        {"factor_name": "session", "factor_value": session.value,
                         "threshold": "RTH", "contribution": "session_quality"},
                        {"factor_name": "box_width", "factor_value": engine.box.box_width,
                         "threshold": C.MIN_BOX_WIDTH, "contribution": "setup_quality"},
                        {"factor_name": "dd_mult", "factor_value": getattr(self._throttle, 'dd_size_mult', 1.0),
                         "threshold": 1.0, "contribution": "drawdown_regime"},
                    ],
                    filter_decisions=self._build_gate_filter_decisions(now_ny, r_points),
                    sizing_inputs={"quality_mult": wo.quality_mult, "contracts": qty,
                                   "dd_mult": getattr(self._throttle, 'dd_size_mult', 1.0),
                                   "equity": getattr(self, '_equity', None),
                                   "final_risk_pct": round(r_points * qty / self._equity * 100, 2) if getattr(self, '_equity', 0) else None},
                    concurrent_positions=1 if self._position.open else 0,
                    drawdown_pct=getattr(self._throttle, 'dd_pct', None),
                    drawdown_tier=self._dd_tier_name(),
                    drawdown_size_mult=getattr(self._throttle, 'dd_size_mult', None),
                    portfolio_state=portfolio_state,
                    signal_evolution=self._build_signal_evolution(),
                    execution_timestamps=exec_ts,
                    **fill_runtime_refs(oms_id, payload, fill_qty=qty),
                )

                # Phase 2B: emit orderbook context at entry
                _ba = self._get_bid_ask()
                self._kit.on_orderbook_context(
                    pair=self._symbol,
                    best_bid=_ba[0] if _ba else price,
                    best_ask=_ba[1] if _ba else price,
                    trade_context="entry",
                    related_trade_id=oms_id,
                    exchange_timestamp=now_ny,
                )
            except Exception:
                pass

    async def _place_protective_stop(self, stop_price: float, qty: int, direction: Direction) -> None:
        """Place protective stop order."""
        inst = self._instruments.get(self._symbol)
        if inst is None:
            return
        exit_side = OrderSide.SELL if direction == Direction.LONG else OrderSide.BUY
        order = OMSOrder(
            strategy_id=C.STRATEGY_ID,
            instrument=inst,
            side=exit_side,
            qty=qty,
            order_type=OrderType.STOP,
            stop_price=stop_price,
            tif="GTC",
            role=OrderRole.STOP,
        )
        receipt = await self._oms.submit_intent(Intent(
            intent_type=IntentType.NEW_ORDER,
            strategy_id=C.STRATEGY_ID,
            order=order,
        ))
        if receipt.oms_order_id:
            core_state, _, events = nqdtc_core_logic.on_order_update(
                self._build_core_state(),
                NQDTCOrderUpdate(
                    oms_order_id=receipt.oms_order_id,
                    status="accepted",
                    timestamp=datetime.now(timezone.utc),
                    order_role="stop",
                ),
            )
            self._apply_core_state(core_state)
            self._apply_core_events(events)

    async def _on_order_update(self, event: Any) -> None:
        """Handle terminal order events (cancel, reject, expire)."""
        oms_id = event.oms_order_id or ""
        etype = event.event_type

        core_state, _, events = nqdtc_core_logic.on_order_update(
            self._build_core_state(),
            NQDTCOrderUpdate(
                oms_order_id=oms_id,
                status=str(getattr(etype, "value", "terminal")).lower(),
                timestamp=datetime.now(timezone.utc),
            ),
        )
        self._apply_core_state(core_state)
        self._apply_core_events(events)

        # Flatten order failed — resubmit emergency flatten (Rec 1/3)
        if self._last_flatten_oms_id and oms_id == self._last_flatten_oms_id:
            self._last_flatten_oms_id = None
            self._pending_flatten_instrumentation.pop(oms_id, None)
            if not self._position.open:
                return  # Position already closed (e.g. stop filled first)
            logger.critical(
                "FLATTEN ORDER %s CANCELLED/REJECTED -- resubmitting emergency flatten",
                oms_id,
            )
            receipt = await self._oms.submit_intent(Intent(
                intent_type=IntentType.FLATTEN,
                strategy_id=C.STRATEGY_ID, instrument_symbol=self._symbol,
            ))
            self._last_flatten_oms_id = receipt.oms_order_id if receipt and receipt.oms_order_id else None
            return

        # If stop order was cancelled/rejected for open position
        if self._position.stop_oms_order_id == oms_id:
            logger.warning("Protective stop %s -> %s!", oms_id, etype.value)

    # ------------------------------------------------------------------
    # Bar fetching (fix #1: session filtering, fix #12: 60D for 30m)
    # ------------------------------------------------------------------

    async def _req_completed_bars(
        self,
        contract: Any,
        duration: str,
        bar_size: str,
        *,
        request_kind: str,
        use_rth: bool = False,
    ) -> list | None:
        bars = await req_panama_adjusted_historical_data(
            self._ib,
            contract,
            symbol=getattr(self, "_symbol", C.DEFAULT_SYMBOL),
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=use_rth,
            formatDate=1,
            request_kind=request_kind,
            completed_only=True,
        )
        return bars if bars else None

    async def _fetch_bars(self, request_kind: str = "recurring") -> None:
        """Fetch all timeframes from IB."""
        if not self._ib.ib.isConnected():
            if not getattr(self, '_fetch_disconn_logged', False):
                logger.warning("Skipping bar fetch — IB not connected")
                self._fetch_disconn_logged = True
            return
        self._fetch_disconn_logged = False
        contract = self._get_contract()
        if contract is None:
            return
        contracts = await self._ib.ib.qualifyContractsAsync(contract)
        if contracts:
            contract = contracts[0]

        try:
            bars_5m = await self._req_completed_bars(contract, "2 D", "5 mins", request_kind=request_kind)
            if bars_5m:
                remember_idle_market_bars(self, bars_5m, symbol=self._symbol, timeframe="5m")
                self._bars_5m = self._bars_to_arrays(bars_5m)
        except Exception:
            logger.exception("Error fetching 5m bars")

        # Phase 1.1: 15m bars for slope filter
        if C.SLOPE_FILTER_ENABLED:
            try:
                bars_15m = await self._req_completed_bars(contract, "5 D", "15 mins", request_kind=request_kind)
                if bars_15m:
                    self._bars_15m = self._bars_to_arrays(bars_15m)
            except Exception:
                logger.exception("Error fetching 15m bars")

        try:
            # Fix #12: fetch 60D instead of 30D for proper ~60d ATR percentile
            bars_30m = await self._req_completed_bars(contract, "60 D", "30 mins", request_kind=request_kind)
            if bars_30m:
                self._raw_bars_30m = bars_30m
                self._bars_30m = self._bars_to_arrays(bars_30m)
                # Session-filter 30m bars (fix #1)
                for sess in (Session.ETH, Session.RTH):
                    filtered = _filter_bars_by_session(bars_30m, sess)
                    if filtered:
                        self._bars_30m_session[sess] = self._bars_to_arrays(filtered)
        except Exception:
            logger.exception("Error fetching 30m bars")

        try:
            bars_1h = await self._req_completed_bars(contract, "60 D", "1 hour", request_kind=request_kind)
            if bars_1h:
                self._bars_1h = self._bars_to_arrays(bars_1h)
        except Exception:
            logger.exception("Error fetching 1H bars")

        try:
            bars_4h = await self._req_completed_bars(contract, "1 Y", "4 hours", request_kind=request_kind)
            if bars_4h:
                self._bars_4h = self._bars_to_arrays(bars_4h)
        except Exception:
            logger.exception("Error fetching 4H bars")

        try:
            bars_d = await self._req_completed_bars(
                contract,
                "2 Y",
                "1 day",
                request_kind=request_kind,
                use_rth=True,
            )
            if bars_d:
                self._bars_daily = self._bars_to_arrays(bars_d)
        except Exception:
            logger.exception("Error fetching daily bars")

    @staticmethod
    def _bars_to_arrays(bars: list) -> dict[str, np.ndarray]:
        return {
            "open": np.array([b.open for b in bars], dtype=float),
            "high": np.array([b.high for b in bars], dtype=float),
            "low": np.array([b.low for b in bars], dtype=float),
            "close": np.array([b.close for b in bars], dtype=float),
            "volume": np.array([getattr(b, "volume", 0) for b in bars], dtype=float),
        }

    def _get_contract(self) -> Any:
        """Build continuous NQ futures contract for historical data."""
        try:
            from ib_async import ContFuture
            return ContFuture(symbol=self._symbol, exchange="CME", currency="USD")
        except Exception:
            logger.warning("Cannot build contract for %s", self._symbol)
            return None

    async def _refresh_equity(self) -> None:
        try:
            accounts = self._ib.ib.managedAccounts()
            if accounts:
                for item in self._ib.ib.accountValues():
                    if item.tag == "NetLiquidation" and item.currency == "USD" and item.account == accounts[0]:
                        self._equity = float(item.value) * self._equity_alloc_pct
                        self._throttle.update_equity(self._equity)
                        return
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Signal evolution (M2)
    # ------------------------------------------------------------------

    def _snapshot_signal_state(self, engine: SessionEngineState) -> dict:
        """Capture current 30m signal state for evolution tracking."""
        bars = self._bars_30m_session.get(engine.session, {})
        c30 = bars.get("close")
        close = float(c30[-1]) if c30 is not None and len(c30) > 0 else 0.0
        return {
            "close": close,
            "atr_14": engine.atr14_30m,
            "displacement": engine.last_disp_metric,
            "score": engine.last_score,
            "chop_score": engine.chop_score,
            "rvol": engine.last_rvol,
            "squeeze": float(engine.squeeze_hist.data[-1]) if len(engine.squeeze_hist.data) > 0 else 0.0,
            "vwap_session": float(engine.vwap_session.value) if engine.vwap_session.value else 0.0,
            "regime": self._regime.composite.value if self._regime.composite else "",
        }

    def _build_signal_evolution(self, n: int = 5) -> list[dict]:
        """Return last n signal snapshots with bars_ago labels."""
        items = list(self._signal_ring)[-n:]
        return [{"bars_ago": n - 1 - i, **s} for i, s in enumerate(items)]

    # ------------------------------------------------------------------
    # Missed opportunity logging
    # ------------------------------------------------------------------

    def _log_missed(self, direction, signal: str, signal_id: str, blocked_by: str,
                    block_reason: str, signal_strength: float = 0.5,
                    filter_decisions: list[dict] | None = None, **extra):
        if not self._kit.active:
            return
        try:
            self._kit.log_missed(
                pair=self._symbol,
                side="LONG" if direction == Direction.LONG else "SHORT",
                signal=signal, signal_id=signal_id,
                signal_strength=signal_strength,
                blocked_by=blocked_by, block_reason=block_reason,
                filter_decisions=filter_decisions,
                strategy_params={"composite": self._regime.composite.value
                                 if self._regime.composite else "", **extra},
                concurrent_positions=1 if self._position.open else 0,
                drawdown_pct=getattr(self._throttle, 'dd_pct', None),
                drawdown_tier=self._dd_tier_name(),
                signal_evolution=self._build_signal_evolution(),
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Telemetry (fix #16)
    # ------------------------------------------------------------------

    def _log_telemetry(self, event: str, engine: SessionEngineState, direction: Direction, **kwargs: Any) -> None:
        """Write structured telemetry record."""
        extra = {
            "box_L": engine.box.L_used,
            "box_height": engine.box.box_width,
            "atr14_30m": engine.atr14_30m,
            "disp_metric": engine.last_disp_metric,
            "disp_threshold": engine.last_disp_threshold,
            "score": engine.last_score,
            "rvol": engine.last_rvol,
            "chop_score": engine.chop_score,
            "breakout_active": engine.breakout.active,
            "continuation_mode": engine.breakout.continuation_mode,
            "composite_regime": self._regime.composite.value,
            "box_state": engine.box.state.value,
            "equity": self._equity,
            "vwap_session": engine.vwap_session.value,
            "vwap_box": engine.vwap_box.value,
            "daily_risk_R": self._daily_risk.realized_pnl_R,
        }
        extra.update(kwargs)  # caller overrides defaults
        record = _telemetry_entry(
            event=event,
            session=engine.session.value,
            mode=engine.mode.value,
            regime=self._regime.regime_4h.value,
            direction="LONG" if direction == Direction.LONG else "SHORT",
            **extra,
        )
        self._telemetry_log.append(record)
        logger.debug("TELEMETRY: %s", json.dumps(record, default=str))

        if len(self._telemetry_log) >= 20:
            self._flush_telemetry()

    def _flush_telemetry(self) -> None:
        """Flush telemetry log to file."""
        if not self._telemetry_log:
            return
        try:
            path = self._state_dir / "nqdtc_telemetry.jsonl"
            with open(path, "a") as f:
                for record in self._telemetry_log:
                    f.write(json.dumps(record, default=str) + "\n")
            self._telemetry_log.clear()
        except Exception:
            logger.exception("Error flushing telemetry")

    # ------------------------------------------------------------------
    # State persistence (fix #15)
    # ------------------------------------------------------------------

    def _persist_state(self) -> None:
        """Save engine state to disk on every 5m close and on fills."""
        try:
            state = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "equity": self._equity,
                "bar_count_5m": self._bar_count_5m,
                "position_open": self._position.open,
                "daily_risk": {
                    "realized_pnl_R": self._daily_risk.realized_pnl_R,
                    "halted": self._daily_risk.halted,
                    "trade_date": self._daily_risk.trade_date,
                    "weekly_realized_R": self._daily_risk.weekly_realized_R,
                    "monthly_realized_R": self._daily_risk.monthly_realized_R,
                    "daily_pnl_ledger": self._daily_risk.daily_pnl_ledger,
                },
                "engines": {},
            }
            for sess, eng in self._engines.items():
                state["engines"][sess.value] = {
                    "box_state": eng.box.state.value,
                    "box_high": eng.box.box_high,
                    "box_low": eng.box.box_low,
                    "box_width": eng.box.box_width,
                    "box_mid": eng.box.box_mid,
                    "box_bars_active": eng.box.box_bars_active,
                    "L": eng.box.L,
                    "L_used": eng.box.L_used,
                    "breakout_active": eng.breakout.active,
                    "breakout_dir": eng.breakout.direction.value,
                    "bars_since_breakout": eng.breakout.bars_since_breakout,
                    "continuation_mode": eng.breakout.continuation_mode,
                    "mm_reached": eng.breakout.mm_reached,
                    "chop_score": eng.chop_score,
                    "mode": eng.mode.value,
                    "atr14_30m": eng.atr14_30m,
                    "reentry_allowed": eng.reentry_allowed,
                    "reentry_used": eng.reentry_used,
                    "last_stopout_r": eng.last_stopout_r,
                    "disp_hist_data": eng.disp_hist.data,
                    "squeeze_hist_data": eng.squeeze_hist.data,
                    "vwap_session": {"cum_tpv": eng.vwap_session.cum_tpv, "cum_vol": eng.vwap_session.cum_vol},
                    "vwap_box": {"cum_tpv": eng.vwap_box.cum_tpv, "cum_vol": eng.vwap_box.cum_vol},
                }
            if self._position.open:
                state["position"] = {
                    "symbol": self._position.symbol,
                    "direction": self._position.direction.value,
                    "entry_subtype": self._position.entry_subtype.value,
                    "entry_price": self._position.entry_price,
                    "stop_price": self._position.stop_price,
                    "qty": self._position.qty,
                    "qty_open": self._position.qty_open,
                    "profit_funded": self._position.profit_funded,
                    "runner_active": self._position.runner_active,
                    "exit_tier": self._position.exit_tier.value,
                    "bars_since_entry_30m": self._position.bars_since_entry_30m,
                    "mm_reached": self._position.mm_reached,
                    "tp1_only_cap": self._position.tp1_only_cap,
                }

            state["working_orders"] = [
                {
                    "oms_order_id": wo.oms_order_id, "subtype": wo.subtype.value,
                    "direction": wo.direction.value, "price": wo.price, "qty": wo.qty,
                    "submitted_bar_idx": wo.submitted_bar_idx, "ttl_bars": wo.ttl_bars,
                    "oca_group": wo.oca_group, "is_limit": wo.is_limit,
                    "rescue_attempted": wo.rescue_attempted, "quality_mult": wo.quality_mult,
                    "stop_for_risk": wo.stop_for_risk, "expected_fill_price": wo.expected_fill_price,
                }
                for wo in self._working_orders
            ]

            path = self._state_dir / C.STATE_FILE
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception:
            logger.exception("Error persisting state")

    def _restore_state(self) -> None:
        """Restore engine state from disk on startup."""
        path = self._state_dir / C.STATE_FILE
        if not path.exists():
            return
        try:
            state = json.loads(path.read_text())
        except Exception:
            logger.exception("Failed to load state from %s", path)
            return

        self._equity = state.get("equity", self._equity)
        self._bar_count_5m = state.get("bar_count_5m", 0)

        dr = state.get("daily_risk", {})
        self._daily_risk.realized_pnl_R = dr.get("realized_pnl_R", 0.0)
        self._daily_risk.halted = dr.get("halted", False)
        self._daily_risk.trade_date = dr.get("trade_date")
        self._daily_risk.weekly_realized_R = dr.get("weekly_realized_R", 0.0)
        self._daily_risk.monthly_realized_R = dr.get("monthly_realized_R", 0.0)
        self._daily_risk.daily_pnl_ledger = dr.get("daily_pnl_ledger", [])

        for sess_str, eng_data in state.get("engines", {}).items():
            try:
                sess = Session(sess_str)
            except ValueError:
                continue
            eng = self._engines.get(sess)
            if eng is None:
                continue
            # Box state
            try:
                eng.box.state = BoxState(eng_data.get("box_state", "INACTIVE"))
            except ValueError:
                pass
            eng.box.box_high = eng_data.get("box_high", 0.0)
            eng.box.box_low = eng_data.get("box_low", 0.0)
            eng.box.box_width = eng_data.get("box_width", 0.0)
            eng.box.box_mid = eng_data.get("box_mid", 0.0)
            eng.box.box_bars_active = eng_data.get("box_bars_active", 0)
            eng.box.L = eng_data.get("L", 32)
            eng.box.L_used = eng_data.get("L_used", 32)

            # Breakout state
            eng.breakout.active = eng_data.get("breakout_active", False)
            try:
                eng.breakout.direction = Direction(eng_data.get("breakout_dir", 0))
            except ValueError:
                pass
            eng.breakout.bars_since_breakout = eng_data.get("bars_since_breakout", 0)
            eng.breakout.continuation_mode = eng_data.get("continuation_mode", False)
            eng.breakout.mm_reached = eng_data.get("mm_reached", False)

            # Chop / mode / re-entry
            eng.chop_score = eng_data.get("chop_score", 0)
            try:
                eng.mode = ChopMode(eng_data.get("mode", "NORMAL"))
            except ValueError:
                pass
            eng.atr14_30m = eng_data.get("atr14_30m", 0.0)
            eng.reentry_allowed = eng_data.get("reentry_allowed", True)
            eng.reentry_used = eng_data.get("reentry_used", False)
            eng.last_stopout_r = eng_data.get("last_stopout_r", 0.0)

            # Rolling buffers
            disp_data = eng_data.get("disp_hist_data", [])
            if disp_data:
                eng.disp_hist.data = list(disp_data)
            squeeze_data = eng_data.get("squeeze_hist_data", [])
            if squeeze_data:
                eng.squeeze_hist.data = list(squeeze_data)
            vs = eng_data.get("vwap_session", {})
            eng.vwap_session.cum_tpv = vs.get("cum_tpv", 0.0)
            eng.vwap_session.cum_vol = vs.get("cum_vol", 0.0)
            vb = eng_data.get("vwap_box", {})
            eng.vwap_box.cum_tpv = vb.get("cum_tpv", 0.0)
            eng.vwap_box.cum_vol = vb.get("cum_vol", 0.0)

        for wo_data in state.get("working_orders", []):
            try:
                self._working_orders.append(WorkingOrder(
                    oms_order_id=wo_data["oms_order_id"],
                    subtype=EntrySubtype(wo_data["subtype"]),
                    direction=Direction(wo_data["direction"]),
                    price=wo_data["price"],
                    qty=wo_data["qty"],
                    submitted_bar_idx=wo_data["submitted_bar_idx"],
                    ttl_bars=wo_data["ttl_bars"],
                    oca_group=wo_data.get("oca_group", ""),
                    is_limit=wo_data.get("is_limit", False),
                    rescue_attempted=wo_data.get("rescue_attempted", False),
                    quality_mult=wo_data.get("quality_mult", 1.0),
                    stop_for_risk=wo_data.get("stop_for_risk", 0.0),
                    expected_fill_price=wo_data.get("expected_fill_price", wo_data.get("price", 0.0)),
                ))
            except Exception:
                logger.warning("Skipped invalid working order in state file")

        # Position
        pos_data = state.get("position")
        if pos_data and state.get("position_open"):
            try:
                self._position = PositionState(
                    open=True,
                    symbol=pos_data.get("symbol", "NQ"),
                    direction=Direction(pos_data["direction"]),
                    entry_subtype=EntrySubtype(pos_data["entry_subtype"]),
                    entry_price=pos_data["entry_price"],
                    stop_price=pos_data["stop_price"],
                    qty=pos_data["qty"],
                    qty_open=pos_data.get("qty_open", pos_data["qty"]),
                    profit_funded=pos_data.get("profit_funded", False),
                    runner_active=pos_data.get("runner_active", False),
                    exit_tier=ExitTier(pos_data.get("exit_tier", "Neutral")),
                    bars_since_entry_30m=pos_data.get("bars_since_entry_30m", 0),
                    mm_reached=pos_data.get("mm_reached", False),
                    tp1_only_cap=pos_data.get("tp1_only_cap", False),
                )
            except Exception:
                logger.warning("Failed to restore position from state file")

        logger.info("State restored: equity=%.2f, %d working orders, pos_open=%s",
                    self._equity, len(self._working_orders), self._position.open)
