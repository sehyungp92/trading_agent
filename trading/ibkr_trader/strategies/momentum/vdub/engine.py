"""Vdubus NQ v4.0 — main async strategy engine (15m evaluation loop).

Integrates: indicators, regime, signals, risk, exits with OMS/IBKR shared infra.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
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

from libs.risk.drawdown_throttle import DrawdownThrottle

from strategies.core.idle_market import (
    maybe_record_idle_market_observation,
    remember_idle_market_bars,
)
from . import config as C
from .core import logic as vdub_core_logic
from .core.logic import apply_core_state as apply_core_runtime_state
from .core.logic import build_core_state as build_core_runtime_state
from .core.serializers import restore_state as restore_core_state
from .core.serializers import snapshot_state as snapshot_core_state
from .core.state import (
    VdubEntryFillContext,
    VdubEntrySubmitted,
    VdubFill,
    VdubFlattenRequest,
    VdubOrderUpdate,
    VdubPartialExitDone,
    VdubStopUpdateRequest,
)
from strategies.core.actions import FlattenPosition, ReplaceProtectiveStop, SubmitExit
from . import indicators as ind
from . import signals as sig
from . import exits
from . import regime as reg
from . import risk
from .models import (
    DayCounters, Direction, EntryType, EventBlockState, PivotPoint,
    PositionStage, PositionState, RegimeState, SessionWindow, SubWindow,
    VolState, WorkingEntry,
)
from strategies.momentum.instrumentation.src.config_snapshot import snapshot_config_module
from strategies.momentum.vdub import config as strategy_config

logger = logging.getLogger(__name__)

_ET = None


def _get_et():
    global _ET
    if _ET is None:
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/New_York")
    return _ET


def _to_et(dt: datetime) -> datetime:
    return dt.astimezone(_get_et())


def _minutes(h: int, m: int) -> int:
    return h * 60 + m


# ---------------------------------------------------------------------------
# Session / time classification
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


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class VdubNQv4Engine:
    """Core 15m-cycle engine for the Vdubus NQ Dominant-Trend Swing Protocol."""

    def __init__(
        self,
        ib_session: Any,
        oms_service: Any,
        instruments: list[Any],
        trade_recorder: TradeRecorder | None = None,
        equity: float = 100_000.0,
        instrumentation=None,
        equity_alloc_pct: float = 1.0,
        disable_background_tasks: bool = False,
    ) -> None:
        self._ib = ib_session
        self._oms = oms_service
        self._instruments = {i.symbol: i for i in instruments}
        self._recorder = trade_recorder
        self._equity = equity
        self._equity_alloc_pct = equity_alloc_pct
        self._symbol = C.DEFAULT_SYMBOL
        self._instr = instrumentation
        self._disable_background_tasks = bool(disable_background_tasks)

        # Sync NQ_SPEC for MNQ trading (matches backtest engine L318-321).
        # TRADING_SYMBOL defaults to NQ (price data) but we trade MNQ contracts.
        # All shared modules (risk, exits) read C.NQ_SPEC for point_value.
        if C.DEFAULT_SYMBOL == "NQ":
            C.NQ_SPEC["point_value"] = 2.0    # MNQ
            C.NQ_SPEC["tick_value"] = 0.50     # MNQ

        from strategies.momentum.instrumentation.src.facade import InstrumentationKit
        self._kit = InstrumentationKit(self._instr, strategy_type="vdubus")

        # State
        self.regime = RegimeState()
        self.counters = DayCounters()
        self.positions: list[PositionState] = []
        self.working_entries: dict[str, WorkingEntry] = {}
        self.event_state = EventBlockState()
        self._bar_idx = 0
        self._last_reset_date = ""

        # Drawdown throttle (DD-based sizing reduction + daily loss cap)
        self._throttle = DrawdownThrottle(equity)

        # Rolling win-rate tracking for adaptive sizing
        self._recent_wins: list[bool] = []

        # Indicator caches
        self._mom15 = np.array([])
        self._atr15 = np.array([])
        self._atr1h = np.array([])
        self._svwap = np.array([])
        self._vwap_a_val = np.nan
        self._vwap_a_arr = np.array([])
        self._pivots_1h: list[PivotPoint] = []
        self._pivots_daily: list[PivotPoint] = []

        # Bar arrays — 15m NQ
        self._c15 = np.array([])
        self._h15 = np.array([])
        self._l15 = np.array([])
        self._v15 = np.array([])
        self._t15: list[datetime] = []

        # Bar arrays — 1H NQ
        self._c1h = np.array([])
        self._h1h = np.array([])
        self._l1h = np.array([])
        self._v1h = np.array([])
        self._t1h: list[datetime] = []

        # Bar arrays — daily NQ (timestamps for VWAP-A anchor)
        self._t_nq_daily: list[datetime] = []

        # Bar arrays — ES daily
        self._es_c = np.array([])
        self._es_h = np.array([])
        self._es_l = np.array([])

        # Signal evolution ring buffer (M2)
        from collections import deque as _deque
        self._signal_ring: _deque = _deque(maxlen=10)

        # Execution cascade timestamps (#16)
        self._cascade_ts: dict[str, datetime] = {}

        # Session transition tracking (#17)
        self._last_session_window: str = ""

        # Flatten-order tracking (Rec 1/3: fill-authoritative flatten)
        self._last_flatten_oms_id: str | None = None
        self._pending_flatten_instrumentation: dict[str, dict] = {}

        # Async
        self._event_task: asyncio.Task | None = None
        self._cycle_task: asyncio.Task | None = None
        self._event_queue: asyncio.Queue | None = None
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
            build_core_state=lambda: build_core_runtime_state(self),
            apply_core_state=lambda state: apply_core_runtime_state(self, state),
            on_bar=vdub_core_logic.on_bar,
            default_symbol=self._symbol,
            default_timeframe="15m",
        ):
            return
        self._last_decision_code = code
        self._last_decision_details = details or {}

    def liveness_payload(self) -> dict:
        return {
            "bars_processed": self._bar_idx,
            "symbol_freshness": {
                sym: ts.isoformat() for sym, ts in self._symbol_last_bar_ts.items()
            },
        }

    # ------------------------------------------------------------------
    # Core on_bar routing helpers
    # ------------------------------------------------------------------

    async def _bar_route_stop(
        self, pos_id: str, new_stop: float, reason: str,
        adjustment_type: str = "trailing",
    ) -> None:
        """Route a stop update through core on_bar, then dispatch to OMS."""
        req = VdubStopUpdateRequest(pos_id=pos_id, new_stop=new_stop, reason=reason)
        core_state = build_core_runtime_state(self)
        core_state, actions, events = vdub_core_logic.on_bar(
            core_state, bar_ts=self._last_bar_ts, stop_updates=[req])
        apply_core_runtime_state(self, core_state)
        for action in actions:
            if isinstance(action, ReplaceProtectiveStop):
                p = next((p for p in self.positions if p.trade_id == pos_id), None)
                if p and p.stop_oms_order_id:
                    old_stop = (action.metadata or {}).get("old_stop", 0.0)
                    await self._update_stop(
                        p, action.stop_price,
                        adjustment_type=adjustment_type,
                        trigger=reason, old_stop=old_stop,
                    )
        for event in events:
            self._record_decision(event.code, event.details)

    async def _bar_route_flatten(self, pos_id: str, reason: str) -> None:
        """Route a flatten request through core on_bar, then dispatch to OMS."""
        req = VdubFlattenRequest(pos_id=pos_id, reason=reason)
        core_state = build_core_runtime_state(self)
        core_state, actions, events = vdub_core_logic.on_bar(
            core_state, bar_ts=self._last_bar_ts, flatten_requests=[req])
        apply_core_runtime_state(self, core_state)
        for action in actions:
            if isinstance(action, FlattenPosition):
                p_id = (action.metadata or {}).get("pos_id")
                p = next((p for p in self.positions
                          if p.trade_id == p_id and p.qty_open > 0), None)
                if p:
                    await self._flatten_position(p, action.reason or reason)
        for event in events:
            self._record_decision(event.code, event.details)

    def _bar_route_decision(self, code: str, details: dict | None = None) -> None:
        """Route a decision record through core on_bar."""
        core_state = build_core_runtime_state(self)
        core_state, _actions, events = vdub_core_logic.on_bar(
            core_state, bar_ts=self._last_bar_ts,
            decision_code=code, decision_details=details)
        apply_core_runtime_state(self, core_state)
        for event in events:
            self._record_decision(event.code, event.details)

    def health_status(self) -> dict:
        return {
            "strategy_id": C.STRATEGY_ID,
            "running": self._running,
            "last_decision_code": self._last_decision_code,
            "last_decision_details": self._last_decision_details,
            "last_bar_ts": self._last_bar_ts.isoformat() if self._last_bar_ts else None,
        }

    def snapshot_state(self) -> dict[str, Any]:
        return snapshot_core_state(build_core_runtime_state(self))

    async def hydrate(self, snapshot: dict[str, Any]) -> None:
        if not snapshot:
            return
        apply_core_runtime_state(self, restore_core_state(snapshot))

    # ------------------------------------------------------------------
    # Signal evolution (M2)
    # ------------------------------------------------------------------

    def _snapshot_signal_state(self) -> dict:
        """Capture current 15m signal state for evolution tracking."""
        c = float(self._c15[-1]) if len(self._c15) > 0 else 0.0
        svwap = float(self._svwap[-1]) if len(self._svwap) > 0 else 0.0
        vwap_a = float(self._vwap_a_val) if not np.isnan(self._vwap_a_val) else None
        atr = float(self._atr15[-1]) if len(self._atr15) > 0 else 0.0
        mom = float(self._mom15[-1]) if len(self._mom15) > 0 else 0.0
        return {
            "close": c,
            "momentum": mom,
            "atr_15m": atr,
            "session_vwap": svwap,
            "vwap_a": vwap_a,
            "vwap_distance_pct": round((c - svwap) / c * 100, 4) if c > 0 and svwap > 0 else None,
        }

    def _build_signal_evolution(self, n: int = 5) -> list[dict]:
        """Return last n signal snapshots with bars_ago labels."""
        items = list(self._signal_ring)[-n:]
        return [{"bars_ago": n - 1 - i, **s} for i, s in enumerate(items)]

    # ------------------------------------------------------------------
    # Missed opportunity logging
    # ------------------------------------------------------------------

    def _log_missed(self, direction, signal_type, signal_id: str, blocked_by: str,
                    block_reason: str, signal_strength: float = 0.5,
                    filter_decisions: list[dict] | None = None, **extra):
        if not self._kit.active:
            return
        try:
            self._kit.log_missed(
                pair=self._symbol,
                side="LONG" if direction == Direction.LONG else "SHORT",
                signal=signal_type.value if hasattr(signal_type, 'value') else str(signal_type),
                signal_id=signal_id, signal_strength=signal_strength,
                blocked_by=blocked_by, block_reason=block_reason,
                filter_decisions=filter_decisions,
                strategy_params={"daily_trend": self.regime.daily_trend, **extra},
                concurrent_positions=len(self.positions),
                drawdown_pct=getattr(self._throttle, 'dd_pct', None),
                drawdown_tier=self._dd_tier_name(),
                signal_evolution=self._build_signal_evolution(),
            )
        except Exception:
            pass

    def _build_gate_filter_decisions(self, direction, qty: int, r_points: float,
                                     unit_risk: float, sub_window) -> list[dict]:
        """Build structured filter decisions from current gate state."""
        decisions = []

        # Heat cap
        open_risk = self._compute_open_risk()
        new_risk = r_points * C.NQ_SPEC["point_value"] * qty
        cap = C.HEAT_CAP_MULT * unit_risk
        decisions.append({
            "filter_name": "heat_cap",
            "threshold": round(cap, 2),
            "actual_value": round(open_risk + new_risk, 2),
            "passed": open_risk + new_risk <= cap,
            "margin_pct": round(((open_risk + new_risk) - cap) / abs(cap) * 100, 1)
                if cap > 0 else None,
        })

        # Daily breaker
        breaker_thresh = C.DAILY_BREAKER_MULT * unit_risk
        realized = self.counters.daily_realized_pnl
        decisions.append({
            "filter_name": "daily_breaker",
            "threshold": round(breaker_thresh, 2),
            "actual_value": round(realized, 2),
            "passed": not self.counters.breaker_hit and realized > breaker_thresh,
            "margin_pct": round((realized - breaker_thresh) / abs(breaker_thresh) * 100, 1)
                if breaker_thresh != 0 else None,
        })

        # Direction cap
        if direction == Direction.LONG:
            decisions.append({
                "filter_name": "long_cap",
                "threshold": C.MAX_LONGS_PER_DAY,
                "actual_value": self.counters.long_fills,
                "passed": self.counters.long_fills < C.MAX_LONGS_PER_DAY,
                "margin_pct": round((self.counters.long_fills - C.MAX_LONGS_PER_DAY)
                                    / abs(C.MAX_LONGS_PER_DAY) * 100, 1),
            })
        else:
            decisions.append({
                "filter_name": "short_cap",
                "threshold": C.MAX_SHORTS_PER_DAY,
                "actual_value": self.counters.short_fills,
                "passed": self.counters.short_fills < C.MAX_SHORTS_PER_DAY,
                "margin_pct": round((self.counters.short_fills - C.MAX_SHORTS_PER_DAY)
                                    / abs(C.MAX_SHORTS_PER_DAY) * 100, 1),
            })

        # Viability (cost/risk ratio)
        slip_ticks = C.SLIP_TICKS_BY_WINDOW.get(sub_window.value, 1)
        tick_val = C.NQ_SPEC["tick_value"]
        slip_cost = slip_ticks * tick_val * qty
        fees_cost = C.RT_COMM_FEES * qty
        total_cost = slip_cost + fees_cost
        risk_usd = r_points * C.NQ_SPEC["point_value"] * qty
        cost_ratio = total_cost / risk_usd if risk_usd > 0 else float('inf')
        decisions.append({
            "filter_name": "viability",
            "threshold": C.COST_RISK_MAX,
            "actual_value": round(cost_ratio, 4) if risk_usd > 0 else 0.0,
            "passed": risk_usd > 0 and cost_ratio <= C.COST_RISK_MAX,
            "margin_pct": round((cost_ratio - C.COST_RISK_MAX) / abs(C.COST_RISK_MAX) * 100, 1)
                if C.COST_RISK_MAX > 0 else None,
        })

        return decisions

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        logger.info("Engine starting")
        self._running = True
        self._event_queue = self._oms.stream_events(C.STRATEGY_ID)
        self._event_task = asyncio.create_task(self._process_events())
        if not self._disable_background_tasks:
            await self._load_initial_bars()
            self._cycle_task = asyncio.create_task(self._15m_scheduler())
        logger.info("Engine started")

    def get_position_snapshot(self) -> list[dict]:
        """Return current position state for heartbeat emission."""
        result = []
        for pos in self.positions:
            if pos.qty_open <= 0:
                continue
            d = 1 if pos.direction == Direction.LONG else -1
            ur = 0.0
            if pos.r_points > 0:
                last = self._bars_15m.get("close", np.array([0]))[-1] if hasattr(self, '_bars_15m') else 0
                ur = (last - pos.entry_price) * d / pos.r_points if pos.r_points > 0 else 0
            result.append({
                "strategy_type": "vdubus",
                "direction": "LONG" if pos.direction == Direction.LONG else "SHORT",
                "entry_price": pos.entry_price,
                "qty": pos.qty_open,
                "unrealized_pnl_r": round(ur, 3),
            })
        return result

    async def stop(self) -> None:
        logger.info("Engine stopping")
        self._running = False
        for task in (self._cycle_task, self._event_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        for oms_id in list(self.working_entries):
            await self._cancel_order(oms_id)
        logger.info("Engine stopped")

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    async def _15m_scheduler(self) -> None:
        while self._running:
            now = datetime.now(timezone.utc)
            minute = now.minute
            next_q = ((minute // 15) + 1) * 15
            if next_q >= 60:
                nxt = (now + timedelta(hours=1)).replace(
                    minute=0, second=10, microsecond=0)
            else:
                nxt = now.replace(minute=next_q, second=10, microsecond=0)
            await asyncio.sleep(max(0, (nxt - now).total_seconds()))
            if not self._running:
                break
            try:
                await self._on_15m_close()
            except Exception:
                logger.exception("Error in 15m cycle")

    # ------------------------------------------------------------------
    # Core 15m cycle (Section 19 gate order)
    # ------------------------------------------------------------------

    async def _on_15m_close(self) -> None:
        now = datetime.now(timezone.utc)
        self._last_bar_ts = now
        self._bar_idx += 1
        self._symbol_last_bar_ts[self._symbol] = now
        logger.info("=== 15m bar %d  %s ===", self._bar_idx, now.isoformat())

        self._check_daily_reset(now)
        await self._refresh_equity()
        self._throttle.update_equity(self._equity)
        await self._fetch_bars(request_kind="startup")
        self._update_indicators()
        self._update_regime()
        self._signal_ring.append(self._snapshot_signal_state())

        # Phase 2B: emit indicator snapshot at each 15m evaluation
        if self._kit.active:
            try:
                c = float(self._c15[-1]) if len(self._c15) > 0 else 0.0
                svwap = float(self._svwap[-1]) if len(self._svwap) > 0 else 0.0
                self._kit.on_indicator_snapshot(
                    pair=self._symbol,
                    indicators={
                        "close": c,
                        "momentum": float(self._mom15[-1]) if len(self._mom15) > 0 else 0.0,
                        "atr_15m": float(self._atr15[-1]) if len(self._atr15) > 0 else 0.0,
                        "atr_1h": float(self._atr1h[-1]) if len(self._atr1h) > 0 else 0.0,
                        "session_vwap": svwap,
                        "vwap_a": float(self._vwap_a_val) if not np.isnan(self._vwap_a_val) else 0.0,
                        "vwap_distance_pct": round((c - svwap) / c * 100, 4) if c > 0 and svwap > 0 else 0.0,
                        "choppiness": self.regime.choppiness,
                    },
                    signal_name="vdubus_eval",
                    signal_strength=0.0,
                    decision="eval",
                    strategy_type="vdubus",
                    exchange_timestamp=now,
                    context={
                        "daily_trend": self.regime.daily_trend,
                        "vol_state": self.regime.vol_state.value,
                        "trend_1h": self.regime.trend_1h,
                        "concurrent_positions": len(self.positions),
                        "drawdown_tier": self._dd_tier_name(),
                    },
                )
            except Exception:
                pass

        # 1) Manage working entries (TTL, teleport, fallback)
        await self._manage_working_entries()

        if await self._force_flatten_for_roll(now):
            return

        # 2) Manage open positions
        await self._manage_positions(now)

        # 3) Decision gate at 15:50
        if _is_1550(now):
            await self._decision_gate(now)

        # 4) Overnight trail (Section 18.1) — after decision gate
        for pos in list(self.positions):
            if pos.qty_open > 0 and pos.stage == PositionStage.SWING_HOLD and _is_overnight(now):
                atr1h = self._safe_atr1h()
                if atr1h == 0:
                    continue
                new_stop = exits.compute_overnight_trail(
                    pos, self._h1h, self._l1h, atr1h)
                if new_stop != pos.stop_price:
                    await self._bar_route_stop(pos.trade_id, new_stop, "overnight_trail")

        # 5) VWAP-A failure (Section 18.3) — after decision gate
        for pos in list(self.positions):
            if pos.qty_open > 0 and pos.session_count >= C.VWAP_A_FAIL_MIN_SESSIONS and len(self._c1h) > 0 and not np.isnan(self._vwap_a_val):
                price = float(self._c15[-1]) if len(self._c15) > 0 else pos.entry_price
                atr1h = self._safe_atr1h()
                if exits.check_vwap_a_failure(
                    pos, float(self._c1h[-1]), self._vwap_a_val, price, atr1h=atr1h,
                ):
                    logger.info("VWAP-A failure: %s", pos.trade_id)
                    await self._bar_route_flatten(pos.trade_id, "VWAP_A_FAIL")

        # 6) Entry evaluation
        session, sub_window = classify_session(now)

        # Session transition tracking (#17)
        session_name = session.value if hasattr(session, 'value') else str(session)
        if self._last_session_window and self._last_session_window != session_name:
            last_price = float(self._c15[-1]) if len(self._c15) > 0 else 0.0
            for pos in self.positions:
                r_pts = abs(pos.entry_price - pos.stop_price)
                if r_pts > 0:
                    if pos.direction == Direction.LONG:
                        ur = (last_price - pos.entry_price) / r_pts
                    else:
                        ur = (pos.entry_price - last_price) / r_pts
                else:
                    ur = 0.0
                pos.session_transitions_log.append({
                    "from_session": self._last_session_window,
                    "to_session": session_name,
                    "transition_time": now.isoformat(),
                    "unrealized_pnl_r": round(ur, 4),
                    "bars_held": pos.bars_since_entry,
                    "price_at_transition": last_price,
                })
        self._last_session_window = session_name

        if session == SessionWindow.BLOCKED:
            self._bar_route_decision("OUTSIDE_RTH", {"session": "BLOCKED"})
            return

        # Event gate (Section 6)
        self._update_event_state(now)
        if not self.event_state.rearmed:
            self._bar_route_decision("SIGNAL_FILTERED", {"reason": "event_block"})
            return

        if self.regime.vol_state == VolState.SHOCK:
            self._bar_route_decision("SIGNAL_FILTERED", {"reason": "vol_shock"})
            await self._shock_tighten_all()
            return

        # Drawdown throttle: daily loss cap halt
        if self._throttle.daily_halted:
            self._bar_route_decision("CIRCUIT_BREAKER", {"reason": "daily_loss_cap_halt"})
            return

        decision_before = self._last_decision_code
        for direction in (Direction.LONG, Direction.SHORT):
            await self._evaluate_direction(direction, session, sub_window, now)
        if self._last_decision_code == decision_before and not self.positions and not self.working_entries:
            self._bar_route_decision(
                "NO_SIGNAL",
                {"session": session.value, "reason": "no_direction_eligible"},
            )

    # ------------------------------------------------------------------
    # Daily reset (09:30 ET)
    # ------------------------------------------------------------------

    def _check_daily_reset(self, now: datetime) -> None:
        et = _to_et(now)
        today = et.strftime("%Y-%m-%d")
        if today != self._last_reset_date and et.hour >= 9 and et.minute >= 30:
            self.counters.reset()
            self.counters.trade_date = today
            self._last_reset_date = today
            self._throttle.daily_reset()
            self._throttle.update_equity(self._equity)
            # Increment session count for held positions
            for pos in self.positions:
                if pos.qty_open > 0:
                    pos.session_count += 1
            logger.info("Daily reset: %s", today)

    # ------------------------------------------------------------------
    # Regime update
    # ------------------------------------------------------------------

    def _update_regime(self) -> None:
        old_dt = self.regime.daily_trend
        old_vs = self.regime.vol_state
        old_1h = self.regime.trend_1h

        if len(self._es_c) > C.DAILY_SMA_PERIOD:
            reg.compute_daily_trend(self._es_c, self.regime)
        if len(self._es_c) > C.VOL_ATR_PERIOD:
            self.regime.vol_state = reg.compute_vol_state(
                self._es_h, self._es_l, self._es_c)
        if len(self._c1h) > C.HOURLY_EMA_PERIOD:
            reg.compute_1h_trend(self._c1h, self.regime)
        if len(self._c1h) > C.CHOP_PERIOD + 1:
            self.regime.choppiness = reg.compute_choppiness(
                self._h1h, self._l1h, self._c1h, C.CHOP_PERIOD)

        if self.regime.daily_trend != old_dt:
            logger.info("DailyTrend: %+d -> %+d", old_dt, self.regime.daily_trend)
        if self.regime.vol_state != old_vs:
            logger.info("VolState: %s -> %s", old_vs.value, self.regime.vol_state.value)
        if self.regime.trend_1h != old_1h:
            logger.info("1H Trend: %+d -> %+d", old_1h, self.regime.trend_1h)

    # ------------------------------------------------------------------
    # Indicator refresh
    # ------------------------------------------------------------------

    def _update_indicators(self) -> None:
        if len(self._c15) > C.VOL_ATR_PERIOD:
            self._atr15 = ind.atr(self._h15, self._l15, self._c15)
        if len(self._c15) > C.MOM_N + C.SLOPE_LB + 1:
            self._mom15 = ind.macd_hist(self._c15)
        if len(self._c1h) > C.VOL_ATR_PERIOD:
            self._atr1h = ind.atr(self._h1h, self._l1h, self._c1h)
            self._pivots_1h = ind.confirmed_pivots(
                self._h1h, self._l1h, C.NCONFIRM_1H)
        self._update_vwaps()

    def _update_vwaps(self) -> None:
        if len(self._c15) < 2:
            return
        # Find session start (most recent bar at/near 09:30 ET)
        sess_idx = 0
        for i in range(len(self._t15) - 1, -1, -1):
            et = _to_et(self._t15[i])
            m = _minutes(et.hour, et.minute)
            if _minutes(9, 30) <= m < _minutes(9, 45):
                sess_idx = i
                break
        self._svwap = ind.session_vwap(
            self._h15, self._l15, self._c15, self._v15, sess_idx)
        self._update_vwap_a()

    def _update_vwap_a(self) -> None:
        self._vwap_a_arr = np.full(len(self._c15), np.nan) if len(self._c15) > 0 else np.array([])
        anchor_idx = self._find_vwap_a_anchor()
        if anchor_idx is not None:
            self._vwap_a_arr = ind.anchored_vwap_series(
                self._h15, self._l15, self._c15, self._v15, anchor_idx)
        self._vwap_a_val = float(self._vwap_a_arr[-1]) if len(self._vwap_a_arr) > 0 and not np.isnan(self._vwap_a_arr[-1]) else np.nan

    def _find_vwap_a_anchor(self) -> Optional[int]:
        """Map swing-origin pivot to 15m bar index via timestamp (1H pivots only)."""
        if not self._t15 or not self._pivots_1h:
            return None
        target = "low" if self.regime.daily_trend >= 0 else "high"
        candidates = [p for p in self._pivots_1h if p.ptype == target]
        if not candidates or not self._t1h:
            return None
        pivot = candidates[-1]
        if pivot.idx >= len(self._t1h):
            return None
        idx_15m = self._ts_to_15m_idx(self._t1h[pivot.idx])
        if idx_15m is not None and 0 <= idx_15m < len(self._c15):
            return idx_15m
        return None

    def _ts_to_15m_idx(self, ts: datetime) -> Optional[int]:
        """Find the 15m bar index closest to the given timestamp."""
        if not self._t15:
            return None
        try:
            ts_epoch = ts.timestamp()
        except Exception:
            ts_epoch = ts.replace(tzinfo=timezone.utc).timestamp()
        best_idx, best_diff = 0, float('inf')
        for i, t in enumerate(self._t15):
            try:
                diff = abs(t.timestamp() - ts_epoch)
            except Exception:
                continue
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        return best_idx

    # ------------------------------------------------------------------
    # Direction evaluation (entry signal pipeline)
    # ------------------------------------------------------------------

    async def _evaluate_direction(
        self, direction: Direction, session: SessionWindow,
        sub_window: SubWindow, now: datetime,
    ) -> None:
        # Record signal evaluation timestamp (#16)
        self._cascade_ts["_last_eval"] = now

        # v4.2: Block entries during the 20:00 ET hour (40% WR, avgR=-0.141)
        if session == SessionWindow.EVENING and _to_et(now).hour == 20:
            return

        # Permission gate (Section 4)
        is_flip = False
        if not reg.direction_allowed(self.regime, direction):
            if reg.flip_entry_eligible(self.regime, self.counters, direction):
                is_flip = True
            else:
                return

        # Late shoulder conditional gates
        in_late_shoulder = _is_late_shoulder(now)
        if in_late_shoulder:
            if C.LATE_SHOULDER_REQUIRE_1H_ALIGN:
                aligned_1h = (direction == Direction.LONG and self.regime.trend_1h == 1) or \
                             (direction == Direction.SHORT and self.regime.trend_1h == -1)
                if not aligned_1h:
                    return
            if C.LATE_SHOULDER_REQUIRE_LOW_CHOP and self.regime.choppiness > C.CHOP_THRESHOLD:
                return

        # 1H alignment -> hard gate (unless flip)
        hourly_mult = C.HOURLY_ALIGNED_MULT
        if not is_flip:
            aligned = (direction == Direction.LONG and self.regime.trend_1h == 1) or \
                      (direction == Direction.SHORT and self.regime.trend_1h == -1)
            if not aligned:
                return

        # Direction caps — enforce MAX per day (reduced in choppy regimes)
        max_l = C.CHOP_MAX_LONGS if self.regime.choppiness > C.CHOP_THRESHOLD else C.MAX_LONGS_PER_DAY
        max_s = C.CHOP_MAX_SHORTS if self.regime.choppiness > C.CHOP_THRESHOLD else C.MAX_SHORTS_PER_DAY
        if direction == Direction.LONG and self.counters.long_fills >= max_l:
            return
        if direction == Direction.SHORT and self.counters.short_fills >= max_s:
            return

        # Momentum confirmation (Section 7)
        if len(self._mom15) == 0:
            self._bar_route_decision(
                "NO_SIGNAL",
                {
                    "direction": direction.name,
                    "session": session.value,
                    "reason": "insufficient_momentum_history",
                },
            )
            return
        long_ok, short_ok = sig.slope_ok(self._mom15)
        if direction == Direction.LONG and not long_ok:
            return
        if direction == Direction.SHORT and not short_ok:
            return

        atr15_val = self._safe_atr15()
        atr1h_val = self._safe_atr1h()
        if atr15_val == 0:
            self._bar_route_decision(
                "NO_SIGNAL",
                {
                    "direction": direction.name,
                    "session": session.value,
                    "reason": "insufficient_atr_history",
                },
            )
            return

        # Signal: Type A priority, then Type B (Section 9.3)
        signal = sig.type_a_check(
            self._c15, self._l15, self._h15, self._svwap,
            self._vwap_a_arr,
            atr15_val, direction, sub_window,
        )
        signal_type = EntryType.TYPE_A
        vwap_used = signal["vwap_used"] if signal else 0.0

        if signal is None and C.USE_TYPE_B and sub_window.value in C.TYPE_B_ALLOWED_WINDOWS:
            # Type B: require 1H alignment if configured
            type_b_ok = True
            if C.TYPE_B_REQUIRE_1H_ALIGN:
                aligned_1h = (direction == Direction.LONG and self.regime.trend_1h == 1) or \
                             (direction == Direction.SHORT and self.regime.trend_1h == -1)
                if not aligned_1h:
                    type_b_ok = False
            if type_b_ok:
                signal = sig.type_b_check(
                    self._c15, self._l15, self._h15,
                    self._pivots_1h, len(self._c1h),
                    atr15_val, direction,
                )
                signal_type = EntryType.TYPE_B
        if signal is None:
            self._bar_route_decision("NO_SIGNAL", {"direction": direction.name, "session": session.value})
            return

        _sig_id = f"{signal_type.value}_{direction.name}_{self._bar_idx}"

        # Late shoulder: tighter VWAP cap override
        if in_late_shoulder and signal.get("type") == "A":
            close = float(self._c15[-1])
            vw = signal.get("vwap_used", 0.0)
            if vw and atr15_val > 0:
                dist = abs(close - vw)
                if dist > C.LATE_SHOULDER_VWAP_CAP * atr15_val:
                    return

        # Predator overlay -> class_mult (Section 8)
        if is_flip:
            class_mult = C.CLASS_MULT_FLIP
        elif sig.predator_present(
            self._pivots_1h, self._h1h, self._l1h,
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

        session_key = "RTH" if session == SessionWindow.RTH else "EVENING"
        session_mult = C.SESSION_MULT[session_key]

        # Block entry if any position is open in opposite direction (single-position model)
        any_open = [p for p in self.positions if p.qty_open > 0]
        if any_open:
            opposite = [p for p in any_open if p.direction != direction]
            if opposite:
                self._log_missed(direction, signal_type, _sig_id, "OPPOSITE_POSITION", "position open in opposite direction")
                return  # Cannot enter opposite direction while position is open

        # Check pyramiding
        is_pyramid = False
        existing = self._get_position(direction)
        if existing:
            close_price = float(self._c15[-1])
            if risk.pyramid_eligible(existing, direction, close_price, self.counters):
                is_pyramid = True
            else:
                self._log_missed(direction, signal_type, _sig_id, "PYRAMID_NOT_ELIGIBLE", "pyramid conditions not met")
                return

        # Compute entry/stop prices (Section 15)
        atr15_ticks = atr15_val / C.NQ_SPEC["tick"]
        stop_entry, limit_entry = risk.compute_entry_prices(
            float(self._h15[-1]), float(self._l15[-1]),
            atr15_ticks, direction,
        )
        entry_est = stop_entry

        # Initial stop (Section 13)
        initial_stop = risk.compute_initial_stop(
            entry_est, direction, self._pivots_1h, atr1h_val, atr15_val)
        r_points = abs(entry_est - initial_stop)
        if r_points == 0:
            return

        # Sizing (Section 12)
        unit_risk = risk.compute_unit_risk(self._equity, self.regime.vol_state)
        eff_risk = risk.compute_effective_risk(unit_risk, class_mult, session_mult * hourly_mult)
        if is_pyramid:
            eff_risk = risk.compute_addon_risk(eff_risk)
        qty = risk.compute_qty(eff_risk, r_points)

        # Drawdown throttle: reduce sizing during drawdowns (floor at 0.75x)
        dd_mult = max(0.75, self._throttle.dd_size_mult)
        if dd_mult < 1.0:
            qty = max(1, int(qty * dd_mult))

        if qty < 1:
            return

        # Build filter decisions for instrumentation
        gate_fds = self._build_gate_filter_decisions(direction, qty, r_points, unit_risk, sub_window)

        # Viability (Section 14)
        ok, reason = risk.pass_viability(qty, r_points, sub_window)
        if not ok:
            self._log_missed(direction, signal_type, _sig_id, f"viability_{reason}", reason,
                             filter_decisions=gate_fds)
            return

        # Risk gates: heat cap, breaker, direction caps (Section 12.5-12.7)
        open_risk = self._compute_open_risk()
        new_risk = r_points * C.NQ_SPEC["point_value"] * qty
        ok, reason = risk.pass_risk_gates(
            self.counters, direction, open_risk, new_risk, unit_risk)
        if not ok:
            self._log_missed(direction, signal_type, _sig_id, f"risk_gate_{reason}", reason,
                             filter_decisions=gate_fds)
            return

        # Submit entry order
        self._bar_route_decision("ENTRY_SUBMITTED", {
            "direction": direction.name, "signal_type": signal_type.value,
            "qty": qty, "is_pyramid": is_pyramid,
        })
        await self._submit_entry(
            direction, qty, stop_entry, limit_entry, initial_stop,
            signal_type, is_flip, is_pyramid, class_mult, vwap_used,
            session, filter_decisions=gate_fds, signal_id=_sig_id,
        )

    # ------------------------------------------------------------------
    # Working entry management (Section 15)
    # ------------------------------------------------------------------

    async def _force_flatten_for_roll(self, now: datetime) -> bool:
        reason = roll_force_flatten_reason(
            self._instruments.get(self._symbol) or self._symbol,
            as_of=now,
        )
        if not reason:
            return False
        open_positions = [pos for pos in list(self.positions) if pos.qty_open > 0]
        if not open_positions:
            return False
        self._record_decision("ROLL_FORCE_FLATTEN", {"reason": reason, "count": len(open_positions)})
        logger.critical("Vdub forcing %d positions flat for roll safety: %s", len(open_positions), reason)
        for pos in open_positions:
            await self._flatten_position(pos, "ROLL_SAFETY")
        return True

    async def _manage_working_entries(self) -> None:
        reason = roll_blackout_reason(
            self._instruments.get(self._symbol) or self._symbol,
            as_of=self._last_bar_ts or datetime.now(timezone.utc),
        )
        if reason and self.working_entries:
            to_cancel = list(self.working_entries)
            for oms_id in to_cancel:
                await self._cancel_order(oms_id)
            self.working_entries.clear()
            self._record_decision(
                "ENTRY_CANCELLED_BY_ROLL_BLACKOUT",
                {"reason": reason, "count": len(to_cancel)},
            )
            logger.warning("Cancelled %d Vdub working entries during roll blackout: %s", len(to_cancel), reason)
            return

        for oms_id, we in list(self.working_entries.items()):
            bars_since = self._bar_idx - we.submitted_bar_idx

            # TTL cancel (Section 15.2)
            if bars_since >= we.ttl_bars:
                await self._cancel_order(oms_id)
                self.working_entries.pop(oms_id, None)
                logger.info("TTL cancel: %s", oms_id)
                continue

            if len(self._c15) == 0:
                continue
            price = float(self._c15[-1])
            tick = C.NQ_SPEC["tick"]

            # Detect trigger
            if not we.triggered:
                if we.direction == Direction.LONG and price >= we.stop_entry:
                    we.triggered = True
                    we.triggered_bar_idx = self._bar_idx
                elif we.direction == Direction.SHORT and price <= we.stop_entry:
                    we.triggered = True
                    we.triggered_bar_idx = self._bar_idx

            # Teleport skip (Section 15.3)
            if we.direction == Direction.LONG and price > we.limit_entry + C.TELEPORT_TICKS * tick:
                await self._cancel_order(oms_id)
                self.working_entries.pop(oms_id, None)
                logger.info("Teleport skip: %s", oms_id)
                continue
            if we.direction == Direction.SHORT and price < we.limit_entry - C.TELEPORT_TICKS * tick:
                await self._cancel_order(oms_id)
                self.working_entries.pop(oms_id, None)
                logger.info("Teleport skip: %s", oms_id)
                continue

            # Fallback (Section 15.4)
            if we.triggered and we.fallback_allowed:
                if self._bar_idx - we.triggered_bar_idx >= C.FALLBACK_WAIT_BARS:
                    atr_ticks = self._safe_atr15() / tick if self._safe_atr15() > 0 else 999
                    if atr_ticks > C.FALLBACK_ATR_TICKS_CAP:
                        continue
                    slip_cost = C.FALLBACK_SLIP_MAX_TICKS * C.NQ_SPEC["tick_value"] * we.qty
                    r_usd = abs(we.stop_entry - we.initial_stop) * C.NQ_SPEC["point_value"] * we.qty
                    if r_usd > 0 and slip_cost / r_usd > C.COST_RISK_MAX:
                        logger.info("Fallback skip (slip/risk): %s", oms_id)
                        continue
                    await self._cancel_order(oms_id)
                    await self._submit_fallback_market(we)
                    we.fallback_allowed = False
                    self.working_entries.pop(oms_id, None)
                    logger.info("Fallback market: %s", oms_id)

    # ------------------------------------------------------------------
    # Position management (Section 16)
    # ------------------------------------------------------------------

    async def _manage_positions(self, now: datetime) -> None:
        for _pos_ref in list(self.positions):
            _tid = _pos_ref.trade_id
            # Re-fetch: prior iteration's core routing may have replaced self.positions
            pos = next((p for p in self.positions if p.trade_id == _tid), None)
            if pos is None or pos.qty_open <= 0:
                continue
            pos.bars_since_entry += 1

            if len(self._h15) > 0:
                pos.highest_since_entry = max(
                    pos.highest_since_entry, float(self._h15[-1]))
                pos.lowest_since_entry = min(
                    pos.lowest_since_entry, float(self._l15[-1]))

            price = float(self._c15[-1]) if len(self._c15) > 0 else pos.entry_price
            atr15 = self._safe_atr15()

            # Update peak MFE R for early kill tracking
            unreal_r = self._unrealized_r(pos, price)
            pos.peak_mfe_r = max(pos.peak_mfe_r, unreal_r)
            pos.peak_mae_r = max(pos.peak_mae_r, max(0.0, -unreal_r))

            # Early kill: fast-dying trades
            if exits.check_early_kill(pos, price):
                logger.info("Early kill: %s (%.2fR, MFE %.2fR)",
                            pos.trade_id, unreal_r, pos.peak_mfe_r)
                await self._bar_route_flatten(pos.trade_id, "EARLY_KILL")
                continue

            # Max duration hard stop
            if exits.check_max_duration(pos):
                logger.info("Max duration exit: %s (%d bars)", pos.trade_id, pos.bars_since_entry)
                await self._bar_route_flatten(pos.trade_id, "MAX_DURATION")
                continue

            # +1R free-ride (Section 16.1)
            # v7: PLUS_1R_PARTIAL_ENABLED=False skips entire +1R block (no BE, no ACTIVE_FREE)
            # — positions stay in ACTIVE_RISK, managed by protective stop + VWAP fail + stale exit
            if C.PLUS_1R_PARTIAL_ENABLED and not pos.partial_done:
                # v4.2: CLOSE entries skip partial — move to BE, keep full position running
                _is_close_entry = (
                    C.CLOSE_SKIP_PARTIAL
                    and pos.entry_time is not None
                    and classify_session(pos.entry_time)[1] == SubWindow.CLOSE
                )
                if _is_close_entry:
                    if self._unrealized_r(pos, price) >= 1.0:
                        await self._bar_route_stop(pos.trade_id, pos.entry_price, "plus1r_be_skip_partial", adjustment_type="breakeven")
                        pos = next((p for p in self.positions if p.trade_id == _tid), pos)
                        pos.partial_done = True
                        pos.stage = PositionStage.ACTIVE_FREE
                        logger.info("+1R BE (skip-partial): %s", pos.trade_id)
                else:
                    qty_close = exits.check_partial(pos, price)
                    if qty_close > 0:
                        await self._submit_partial_exit(pos, qty_close)
                        pos.qty_open -= qty_close
                        await self._bar_route_stop(pos.trade_id, pos.entry_price, "plus1r_partial_be", adjustment_type="breakeven")
                        pos = next((p for p in self.positions if p.trade_id == _tid), pos)
                        pos.partial_done = True
                        pos.stage = PositionStage.ACTIVE_FREE
                        logger.info("+1R partial: %s closed=%d", pos.trade_id, qty_close)
                    elif self._unrealized_r(pos, price) >= 1.0:
                        # 1-lot: just move stop to BE
                        await self._bar_route_stop(pos.trade_id, pos.entry_price, "plus1r_be", adjustment_type="breakeven")
                        pos = next((p for p in self.positions if p.trade_id == _tid), pos)
                        pos.partial_done = True
                        pos.stage = PositionStage.ACTIVE_FREE
                        logger.info("+1R BE: %s", pos.trade_id)

            # ACTIVE_FREE tracking and exits
            if pos.stage == PositionStage.ACTIVE_FREE:
                pos.bars_since_partial += 1
                unreal_r = self._unrealized_r(pos, price)
                pos.peak_r_since_free = max(pos.peak_r_since_free, unreal_r)

                # Profit lock: tighten stop to lock +0.25R once peak >= 0.50R
                lock_stop = exits.compute_free_profit_lock(pos, price)
                if lock_stop != pos.stop_price:
                    await self._bar_route_stop(pos.trade_id, lock_stop, "free_profit_lock")
                    pos = next((p for p in self.positions if p.trade_id == _tid), pos)

                # v4.2: CLOSE-specific MFE ratchet (applied after BE move)
                if C.CLOSE_SKIP_PARTIAL and pos.entry_time is not None:
                    if classify_session(pos.entry_time)[1] == SubWindow.CLOSE:
                        ratchet = exits.compute_close_mfe_ratchet(pos)
                        if ratchet > 0.0:
                            if pos.direction == Direction.LONG:
                                new_floor = max(pos.stop_price, ratchet)
                            else:
                                new_floor = min(pos.stop_price, ratchet)
                            if new_floor != pos.stop_price:
                                await self._bar_route_stop(pos.trade_id, new_floor, "close_mfe_ratchet")
                                pos = next((p for p in self.positions if p.trade_id == _tid), pos)
                                logger.debug("CLOSE ratchet: %s stop=%.2f (MFE=%.2fR)",
                                             pos.trade_id, new_floor, pos.peak_mfe_r)

                # Free-ride stale exit
                if exits.check_free_ride_stale(pos, price):
                    logger.info("Free-ride stale exit: %s (%d bars since partial)",
                                pos.trade_id, pos.bars_since_partial)
                    await self._bar_route_flatten(pos.trade_id, "FREE_STALE")
                    continue

            # Intraday trailing (Section 16.2) — post +1R, not overnight
            if pos.partial_done and not _is_overnight(now):
                _, sub_window = classify_session(now)
                # Window-specific trail tightening
                tf = C.TRAIL_WINDOW_MULT.get(sub_window.value, 1.0)
                # Additional tightening for OPEN entries transitioned to CORE
                entered_open = (pos.entry_time is not None and
                                classify_session(pos.entry_time)[1] == SubWindow.OPEN)
                if entered_open and sub_window == SubWindow.CORE:
                    tf *= C.TRAIL_CORE_TRANSITION_REDUCTION
                new_stop = exits.compute_intraday_trail(
                    pos, self._h15, self._l15, atr15, price, tighten_factor=tf,
                    stage=pos.stage)
                if new_stop != pos.stop_price:
                    await self._bar_route_stop(pos.trade_id, new_stop, "intraday_trail")
                    pos = next((p for p in self.positions if p.trade_id == _tid), pos)

            # VWAP failure exit (Section 16.3) — pre +1R, skip evening (stale VWAP)
            vwap_fail_ok = C.VWAP_FAIL_EVENING or pos.entry_session != SessionWindow.EVENING
            if vwap_fail_ok and not pos.partial_done and pos.vwap_used_at_entry != 0.0:
                if exits.check_vwap_failure(pos, self._c15, pos.vwap_used_at_entry):
                    logger.info("VWAP failure exit: %s", pos.trade_id)
                    await self._bar_route_flatten(pos.trade_id, "VWAP_FAIL")
                    continue

            # Stale exit (Section 16.4) — pre +1R
            if not pos.partial_done and exits.check_stale_exit(pos, price, sub_window="CORE"):
                logger.info("Stale exit: %s (%d bars)", pos.trade_id, pos.bars_since_entry)
                await self._bar_route_flatten(pos.trade_id, "STALE")
                continue

    # ------------------------------------------------------------------
    # Decision gate (Section 17)
    # ------------------------------------------------------------------

    async def _decision_gate(self, now: datetime) -> None:
        friday = _is_friday(now)
        for pos in list(self.positions):
            if pos.qty_open <= 0:
                continue
            price = float(self._c15[-1]) if len(self._c15) > 0 else pos.entry_price
            long_ok, short_ok = sig.slope_ok(self._mom15) if len(self._mom15) > 0 else (False, False)
            slope_ok_dir = long_ok if pos.direction == Direction.LONG else short_ok
            trend_ok = (pos.direction == Direction.LONG and self.regime.trend_1h == 1) or \
                       (pos.direction == Direction.SHORT and self.regime.trend_1h == -1)

            action, new_stop = exits.decision_gate(
                pos, friday, price, slope_ok_dir, trend_ok)

            if action == "HOLD":
                if new_stop != pos.stop_price:
                    await self._bar_route_stop(pos.trade_id, new_stop, "gate_hold")
                    pos = next((p for p in self.positions if p.trade_id == pos.trade_id), pos)
                pos.stage = PositionStage.SWING_HOLD
                logger.info("Gate HOLD: %s (%.2fR)",
                            pos.trade_id, self._unrealized_r(pos, price))
            else:
                logger.info("Gate FLATTEN: %s (%.2fR)",
                            pos.trade_id, self._unrealized_r(pos, price))
                await self._bar_route_flatten(pos.trade_id, "GATE_FLATTEN")

    # ------------------------------------------------------------------
    # Shock mid-position (Section 4.3)
    # ------------------------------------------------------------------

    async def _shock_tighten_all(self) -> None:
        for pos in list(self.positions):
            if pos.qty_open > 0:
                new_stop = exits.shock_stop_tighten(pos)
                if new_stop != pos.stop_price:
                    await self._bar_route_stop(pos.trade_id, new_stop, "shock_tighten")

    # ------------------------------------------------------------------
    # Event safety (Section 6)
    # ------------------------------------------------------------------

    def _update_event_state(self, now: datetime) -> None:
        es = self.event_state
        if es.block_end_ts and now < es.block_end_ts:
            es.rearmed = False
            return
        if es.cooldown_remaining > 0:
            es.cooldown_remaining -= 1
            es.rearmed = False
            return
        if not es.rearmed and es.block_end_ts and now >= es.block_end_ts:
            if self._check_rearm():
                es.rearmed = True
                logger.info("Event re-armed")
            else:
                max_ext = es.block_end_ts + timedelta(minutes=C.MAX_POST_EVENT_MINUTES)
                if now >= max_ext:
                    es.rearmed = True
                    logger.info("Event force re-armed at max extension")

    def _check_rearm(self) -> bool:
        if len(self._atr15) < 2 or np.isnan(self._atr15[-1]):
            return True
        current = float(self._atr15[-1])
        pre = self.event_state.pre_event_atr15
        if pre > 0 and current < C.ATR_NORM_MULT * pre:
            return True
        return False

    def set_event_block(self, event_type: str, block_end: datetime) -> None:
        es = self.event_state
        es.blocked = True
        es.block_end_ts = block_end
        es.event_type = event_type
        es.rearmed = False
        es.cooldown_remaining = C.COOLDOWN_BARS.get(event_type, 3)
        if len(self._atr15) >= 12:
            es.pre_event_atr15 = float(np.nanmean(self._atr15[-12:]))
        logger.info("Event block: %s until %s", event_type, block_end.isoformat())

    # ------------------------------------------------------------------
    # Order submission (shared OMS integration)
    # ------------------------------------------------------------------

    async def _submit_entry(
        self, direction: Direction, qty: int,
        stop_entry: float, limit_entry: float, initial_stop: float,
        signal_type: EntryType, is_flip: bool, is_pyramid: bool,
        class_mult: float, vwap_used: float,
        session: SessionWindow = SessionWindow.RTH,
        filter_decisions: list[dict] | None = None,
        signal_id: str = "",
    ) -> None:
        # Don't submit if we already have working orders
        if self.working_entries:
            self._log_missed(direction, signal_type, f"{signal_type.value}_{direction.name}_{self._bar_idx}", "WORKING_ORDER_EXISTS", "working entry already pending")
            return
        inst = self._instruments.get(self._symbol)
        if inst is None:
            return
        side = OrderSide.BUY if direction == Direction.LONG else OrderSide.SELL
        signal_context = self._entry_signal_context(
            signal_type=signal_type,
            direction=direction,
            signal_id=signal_id,
        )
        risk_ctx = RiskContext(
            stop_for_risk=initial_stop,
            planned_entry_price=stop_entry,
            risk_dollars=RiskCalculator.compute_order_risk_dollars(
                stop_entry, initial_stop, qty, C.NQ_SPEC["point_value"]),
            **signal_context,
        )
        order = OMSOrder(
            strategy_id=C.STRATEGY_ID, instrument=inst, side=side, qty=qty,
            order_type=OrderType.STOP_LIMIT,
            stop_price=stop_entry, limit_price=limit_entry,
            tif="GTC", role=OrderRole.ENTRY,
            entry_policy=EntryPolicy(
                ttl_bars=C.TTL_BARS, teleport_ticks=C.TELEPORT_TICKS),
            risk_context=risk_ctx,
        )
        intent = Intent(
            intent_type=IntentType.NEW_ORDER,
            strategy_id=C.STRATEGY_ID, order=order,
        )
        receipt = await self._oms.submit_intent(intent)
        if not receipt.oms_order_id:
            return

        we = WorkingEntry(
            oms_order_id=receipt.oms_order_id,
            entry_type=signal_type, direction=direction,
            stop_entry=stop_entry, limit_entry=limit_entry,
            qty=qty, submitted_bar_idx=self._bar_idx,
            ttl_bars=C.TTL_BARS,
            initial_stop=initial_stop, vwap_used=vwap_used,
            class_mult=class_mult,
            session=session,
            is_flip=is_flip, is_addon=is_pyramid,
            filter_decisions=filter_decisions,
            signal_id=signal_context["signal_id"],
            bar_id=signal_context["bar_id"],
            exchange_timestamp=signal_context["exchange_timestamp"],
        )
        self.working_entries[receipt.oms_order_id] = we

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

        logger.info(
            "Entry: %s %s %s qty=%d stop=%.2f lim=%.2f R=%.1f -> %s",
            signal_type.value, direction.name,
            "FLIP" if is_flip else ("ADD" if is_pyramid else "NEW"),
            qty, stop_entry, limit_entry,
            abs(stop_entry - initial_stop), receipt.oms_order_id,
        )

    async def _submit_fallback_market(self, we: WorkingEntry) -> None:
        inst = self._instruments.get(self._symbol)
        if inst is None:
            return
        side = OrderSide.BUY if we.direction == Direction.LONG else OrderSide.SELL
        signal_context = self._entry_signal_context(
            signal_type=we.entry_type,
            direction=we.direction,
            signal_id=we.signal_id,
            bar_id=we.bar_id,
            bar_ts=we.exchange_timestamp,
            bar_idx=we.submitted_bar_idx,
        )
        risk_ctx = RiskContext(
            stop_for_risk=we.initial_stop,
            planned_entry_price=we.stop_entry,
            risk_dollars=RiskCalculator.compute_order_risk_dollars(
                we.stop_entry, we.initial_stop, we.qty, C.NQ_SPEC["point_value"]),
            **signal_context,
        )
        order = OMSOrder(
            strategy_id=C.STRATEGY_ID, instrument=inst, side=side, qty=we.qty,
            order_type=OrderType.MARKET, tif="GTC", role=OrderRole.ENTRY,
            risk_context=risk_ctx,
        )
        intent = Intent(
            intent_type=IntentType.NEW_ORDER,
            strategy_id=C.STRATEGY_ID, order=order,
        )
        receipt = await self._oms.submit_intent(intent)
        if receipt.oms_order_id:
            fb = WorkingEntry(
                oms_order_id=receipt.oms_order_id,
                entry_type=we.entry_type, direction=we.direction,
                stop_entry=we.stop_entry, limit_entry=we.stop_entry,
                qty=we.qty, submitted_bar_idx=self._bar_idx,
                initial_stop=we.initial_stop, vwap_used=we.vwap_used,
                class_mult=we.class_mult, fallback_allowed=False,
                session=we.session,
                is_flip=we.is_flip, is_addon=we.is_addon,
                signal_id=signal_context["signal_id"],
                bar_id=signal_context["bar_id"],
                exchange_timestamp=signal_context["exchange_timestamp"],
            )
            self.working_entries[receipt.oms_order_id] = fb

    def _entry_signal_context(
        self,
        *,
        signal_type: EntryType,
        direction: Direction,
        signal_id: str = "",
        bar_id: str = "",
        bar_ts: datetime | None = None,
        bar_idx: int | None = None,
    ) -> dict[str, Any]:
        ts = bar_ts or self._last_bar_ts or datetime.now(timezone.utc)
        ts_text = ts.isoformat()
        idx = self._bar_idx if bar_idx is None else bar_idx
        resolved_signal_id = signal_id or f"{signal_type.value}_{direction.name}_{idx}"
        return {
            "signal_id": resolved_signal_id,
            "bar_id": bar_id or f"{self._symbol}:15m:{ts_text}",
            "exchange_timestamp": ts,
        }

    async def _submit_partial_exit(self, pos: PositionState, qty: int) -> None:
        inst = self._instruments.get(self._symbol)
        if inst is None:
            return
        side = OrderSide.SELL if pos.direction == Direction.LONG else OrderSide.BUY
        order = OMSOrder(
            strategy_id=C.STRATEGY_ID, instrument=inst, side=side, qty=qty,
            order_type=OrderType.MARKET, tif="GTC", role=OrderRole.EXIT,
        )
        intent = Intent(
            intent_type=IntentType.NEW_ORDER,
            strategy_id=C.STRATEGY_ID, order=order,
        )
        await self._oms.submit_intent(intent)

    async def _place_stop(self, pos: PositionState) -> None:
        inst = self._instruments.get(self._symbol)
        if inst is None:
            return
        side = OrderSide.SELL if pos.direction == Direction.LONG else OrderSide.BUY
        order = OMSOrder(
            strategy_id=C.STRATEGY_ID, instrument=inst, side=side,
            qty=pos.qty_open, order_type=OrderType.STOP,
            stop_price=pos.stop_price, tif="GTC", role=OrderRole.STOP,
        )
        intent = Intent(
            intent_type=IntentType.NEW_ORDER,
            strategy_id=C.STRATEGY_ID, order=order,
        )
        receipt = await self._oms.submit_intent(intent)
        if receipt.oms_order_id:
            pos.stop_oms_order_id = receipt.oms_order_id

    async def _update_stop(self, pos: PositionState, new_stop: float,
                           adjustment_type: str = "trailing", trigger: str = "vdub_trail",
                           old_stop: float = 0.0) -> None:
        if not pos.stop_oms_order_id:
            return
        tick = C.NQ_SPEC["tick"]
        new_stop = round_to_tick(new_stop, tick)
        await self._oms.submit_intent(Intent(
            intent_type=IntentType.REPLACE_ORDER,
            strategy_id=C.STRATEGY_ID,
            target_oms_order_id=pos.stop_oms_order_id,
            new_stop_price=new_stop,
        ))
        if self._kit and old_stop != new_stop:
            self._kit.log_stop_adjustment(
                trade_id=pos.trade_id or f"VDUB-{self._symbol}",
                symbol=self._symbol, old_stop=old_stop, new_stop=new_stop,
                adjustment_type=adjustment_type, trigger=trigger,
            )

    def _dd_tier_name(self) -> str:
        mult = getattr(self._throttle, 'dd_size_mult', 1.0)
        if mult >= 1.0:
            return "full"
        elif mult >= 0.5:
            return "half"
        elif mult >= 0.25:
            return "quarter"
        return "halt"

    async def _flatten_position(
        self, pos: PositionState, reason: str = "FLATTEN",
    ) -> None:
        # Cancel protective stop
        if pos.stop_oms_order_id:
            await self._cancel_order(pos.stop_oms_order_id)

        # Flatten via OMS — capture order ID for terminal-event detection
        receipt = await self._oms.submit_intent(Intent(
            intent_type=IntentType.FLATTEN,
            strategy_id=C.STRATEGY_ID, instrument_symbol=self._symbol,
        ))
        self._last_flatten_oms_id = receipt.oms_order_id if receipt and receipt.oms_order_id else None

        # Record exit
        price = float(self._c15[-1]) if len(self._c15) > 0 else pos.entry_price
        pnl_pts = (price - pos.entry_price) if pos.direction == Direction.LONG else (pos.entry_price - price)
        realized_usd = pnl_pts * C.NQ_SPEC["point_value"] * pos.qty_open
        self.counters.daily_realized_pnl += realized_usd
        self._recent_wins.append(pnl_pts > 0)

        r = pnl_pts / pos.r_points if pos.r_points > 0 else 0
        self._throttle.record_trade_close(r)

        logger.info("FLATTEN %s reason=%s pnl=$%.2f", pos.trade_id, reason, realized_usd)
        if self._last_flatten_oms_id and pos.trade_id:
            self._pending_flatten_instrumentation[self._last_flatten_oms_id] = {
                "trade_id": pos.trade_id,
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "qty_open": pos.qty_open,
                "r_points": pos.r_points,
                "reason": reason,
                "expected_exit_price": pos.stop_price if reason == "STOP" else price,
                "mfe_r": pos.peak_mfe_r,
                "mae_r": pos.peak_mae_r,
                "mfe_price": pos.highest_since_entry if pos.direction == Direction.LONG else pos.lowest_since_entry,
                "mae_price": pos.lowest_since_entry if pos.direction == Direction.LONG else pos.highest_since_entry,
                "duration_bars": pos.bars_since_entry,
                "session_transitions": getattr(pos, "session_transitions_log", None) or None,
            }
        pos.qty_open = 0
        pos.direction = Direction.FLAT
        self.positions = [p for p in self.positions if p.qty_open > 0]

    async def _cancel_order(self, oms_order_id: str) -> None:
        try:
            await self._oms.submit_intent(Intent(
                intent_type=IntentType.CANCEL_ORDER,
                strategy_id=C.STRATEGY_ID,
                target_oms_order_id=oms_order_id,
            ))
        except Exception as e:
            logger.warning("Cancel error %s: %s", oms_order_id, e)

    # ------------------------------------------------------------------
    # Event processing (OMS fill/cancel callbacks)
    # ------------------------------------------------------------------

    async def _process_events(self) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            try:
                await self._handle_event(event)
            except Exception:
                logger.exception("Event error")

    async def _handle_event(self, event: Any) -> None:
        etype = event.event_type
        oms_id = event.oms_order_id

        if etype == OMSEventType.FILL:
            await self._on_fill(oms_id, event.payload or {})
        elif etype in (OMSEventType.ORDER_CANCELLED, OMSEventType.ORDER_EXPIRED,
                       OMSEventType.ORDER_REJECTED):
            await self._on_terminal(oms_id)

    async def _on_fill(self, oms_id: str | None, payload: dict) -> None:
        if not oms_id:
            return
        from copy import deepcopy

        fill_price = payload.get("price", 0.0)
        fill_qty = int(payload.get("qty", 0))
        fill_time = datetime.now(timezone.utc)

        if self._last_flatten_oms_id and oms_id == self._last_flatten_oms_id:
            pending = self._pending_flatten_instrumentation.pop(oms_id, {})
            self._last_flatten_oms_id = None
            if pending:
                qty_open = int(pending.get("qty_open") or fill_qty or 0)
                direction = pending.get("direction")
                entry_price = float(pending.get("entry_price") or fill_price)
                pnl_pts = (
                    (fill_price - entry_price)
                    if direction == Direction.LONG else
                    (entry_price - fill_price)
                )
                realized_usd = pnl_pts * C.NQ_SPEC["point_value"] * qty_open
                r_points = float(pending.get("r_points") or 0.0)
                realized_r = pnl_pts / r_points if r_points > 0 else 0.0
                trade_id = str(pending.get("trade_id") or "")
                if self._recorder and trade_id:
                    try:
                        await self._recorder.record_exit(
                            trade_id=trade_id,
                            exit_price=Decimal(str(round(fill_price, 2))),
                            exit_ts=fill_time,
                            exit_reason=str(pending.get("reason") or "FLATTEN"),
                            realized_r=Decimal(str(round(realized_r, 4))),
                            realized_usd=Decimal(str(round(realized_usd, 2))),
                            duration_bars=int(pending.get("duration_bars") or 0),
                        )
                    except Exception:
                        logger.exception("Error recording flatten exit")
                if self._kit.active and trade_id:
                    try:
                        self._kit.log_exit(
                            trade_id=trade_id,
                            exit_price=fill_price,
                            exit_reason=str(pending.get("reason") or "FLATTEN"),
                            expected_exit_price=pending.get("expected_exit_price") or fill_price,
                            mfe_r=pending.get("mfe_r"),
                            mae_r=pending.get("mae_r"),
                            mfe_price=pending.get("mfe_price"),
                            mae_price=pending.get("mae_price"),
                            session_transitions=pending.get("session_transitions"),
                            **fill_runtime_refs(oms_id, payload, fill_qty=fill_qty, is_exit=True),
                        )
                        _ba = self._get_bid_ask()
                        self._kit.on_orderbook_context(
                            pair=self._symbol,
                            best_bid=_ba[0] if _ba else fill_price,
                            best_ask=_ba[1] if _ba else fill_price,
                            trade_context="exit",
                            related_trade_id=trade_id,
                        )
                    except Exception:
                        pass
            return

        # Build fill object with entry context if applicable
        fill = VdubFill(
            oms_order_id=oms_id,
            fill_price=fill_price,
            fill_qty=fill_qty,
            fill_time=fill_time,
            point_value=C.NQ_SPEC["point_value"],
        )
        matched_we = None
        if oms_id in self.working_entries:
            matched_we = self.working_entries[oms_id]
            fill.fill_price = fill_price or matched_we.stop_entry
            fill.fill_qty = fill_qty or matched_we.qty
            fill.entry_context = VdubEntryFillContext(working_entry=matched_we)

        # Snapshot pre-fill state for instrumentation
        pre_positions = deepcopy(self.positions)

        # Route through core
        core_state = build_core_runtime_state(self)
        core_state, actions, events = vdub_core_logic.on_fill(core_state, fill)
        apply_core_runtime_state(self, core_state)

        # Dispatch actions
        for action in actions:
            if isinstance(action, SubmitExit):
                pos = next(
                    (p for p in self.positions
                     if p.trade_id == action.metadata.get("pos_id")),
                    None,
                )
                if pos:
                    await self._place_stop(pos)
            elif isinstance(action, FlattenPosition):
                reason = action.reason
                logger.warning("Core requested flatten: %s", reason)
                await self._oms.submit_intent(Intent(
                    intent_type=IntentType.FLATTEN,
                    strategy_id=C.STRATEGY_ID,
                    instrument_symbol=self._symbol,
                ))

        # Record trade_id from recorder AFTER core creates position
        # (core uses working_entry.oms_order_id as trade_id; we overwrite with recorder ID)
        trade_id = ""
        if matched_we is not None:
            if self._recorder:
                try:
                    trade_id = await self._recorder.record_entry(
                        strategy_id=C.STRATEGY_ID, instrument=self._symbol,
                        direction="LONG" if matched_we.direction == Direction.LONG else "SHORT",
                        quantity=fill.fill_qty, entry_price=Decimal(str(fill.fill_price)),
                        entry_ts=fill_time, setup_tag=matched_we.entry_type.value,
                        entry_type=matched_we.entry_type.value,
                    )
                except Exception:
                    logger.exception("Error recording entry")
            # Overwrite core's trade_id with recorder trade_id
            if trade_id:
                for pos in self.positions:
                    if pos.trade_id == matched_we.oms_order_id:
                        pos.trade_id = trade_id
                        break

        # Post-core instrumentation based on events
        for event in events:
            if event.code == "ENTRY_FILLED":
                we = matched_we
                pos = next(
                    (p for p in self.positions
                     if p.entry_price == fill.fill_price and p.entry_time == fill_time),
                    None,
                )
                if we and pos:
                    logger.info("FILL %s %s %d @ %.2f (R=%.1f)",
                                we.entry_type.value, we.direction.name,
                                fill.fill_qty, fill.fill_price, pos.r_points)
                    await self._on_entry_fill_instrumentation(
                        we, pos, fill.fill_price, fill.fill_qty, fill_time, payload)
            elif event.code == "STOP_FILLED":
                pre_pos = next(
                    (p for p in pre_positions
                     if p.trade_id == event.details.get("trade_id")), None)
                if pre_pos:
                    stop_fill_price = event.details.get("fill_price", fill_price)
                    self._throttle.record_trade_close(event.details.get("r", 0))
                    await self._on_stop_fill_instrumentation(
                        pre_pos, stop_fill_price, payload)
            elif event.code == "ENTRY_FILL_REJECTED":
                logger.warning("Fill rejected: %s", event.details.get("reason", "unknown"))

    async def _on_entry_fill_instrumentation(
        self, we, pos, fill_price: float, fill_qty: int, fill_time: datetime, payload: dict | None = None,
    ) -> None:
        """Instrumentation-only: log entry after core has created the position."""
        trade_id = pos.trade_id
        if not self._kit.active or not trade_id:
            return
        try:
            pv = C.NQ_SPEC["point_value"]
            config_snapshot = snapshot_config_module(strategy_config)

            signal_detected_at = self._cascade_ts.pop("_last_eval", fill_time)
            exec_ts = {
                "signal_detected_at": signal_detected_at.isoformat(),
                "fill_received_at": fill_time.isoformat(),
                "cascade_duration_ms": round(
                    (fill_time - signal_detected_at).total_seconds() * 1000
                ),
            }

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
                trade_id=trade_id,
                pair=self._symbol,
                side="LONG" if we.direction == Direction.LONG else "SHORT",
                entry_price=fill_price,
                position_size=fill_qty,
                position_size_quote=fill_qty * fill_price * pv,
                entry_signal=we.entry_type.value,
                entry_signal_id=trade_id,
                entry_signal_strength=we.class_mult,
                expected_entry_price=we.stop_entry,
                strategy_params={
                    "entry_type": we.entry_type.value,
                    "initial_stop": we.initial_stop,
                    "session": we.session.value if hasattr(we.session, 'value') else str(we.session),
                    "class_mult": we.class_mult,
                    **config_snapshot,
                },
                filter_decisions=we.filter_decisions,
                signal_factors=[
                    {"factor_name": "class_mult", "factor_value": we.class_mult,
                     "threshold": 0.0, "contribution": we.class_mult},
                    {"factor_name": "entry_type", "factor_value": we.entry_type.value,
                     "threshold": "TYPE_A", "contribution": "entry_quality"},
                    {"factor_name": "session", "factor_value": we.session.value if hasattr(we.session, 'value') else str(we.session),
                     "threshold": "CORE", "contribution": "session_quality"},
                    {"factor_name": "daily_trend", "factor_value": self.regime.daily_trend,
                     "threshold": 0, "contribution": "trend_alignment"},
                    {"factor_name": "vol_state", "factor_value": self.regime.vol_state.value if hasattr(self.regime.vol_state, 'value') else str(self.regime.vol_state),
                     "threshold": "NORMAL", "contribution": "volatility_regime"},
                    {"factor_name": "chop_value", "factor_value": self.regime.choppiness,
                     "threshold": C.CHOP_THRESHOLD, "contribution": "trend_clarity"},
                ],
                sizing_inputs={
                    "unit_risk": risk.compute_unit_risk(self._equity, self.regime.vol_state),
                    "class_mult": we.class_mult,
                    "session_mult": C.SESSION_MULT.get(
                        we.session.value if hasattr(we.session, 'value') else "RTH", 1.0),
                    "dd_mult": max(0.75, self._throttle.dd_size_mult),
                    "contracts": fill_qty,
                    "equity": self._equity,
                },
                session_type=we.session.value if hasattr(we.session, 'value') else str(we.session),
                concurrent_positions=len(self.positions),
                drawdown_pct=getattr(self._throttle, 'dd_pct', None),
                drawdown_tier=self._dd_tier_name(),
                drawdown_size_mult=getattr(self._throttle, 'dd_size_mult', None),
                portfolio_state=portfolio_state,
                signal_evolution=self._build_signal_evolution(),
                execution_timestamps=exec_ts,
                **fill_runtime_refs(getattr(we, "oms_order_id", ""), payload, fill_qty=fill_qty),
            )

            _ba = self._get_bid_ask()
            self._kit.on_orderbook_context(
                pair=self._symbol,
                best_bid=_ba[0] if _ba else fill_price,
                best_ask=_ba[1] if _ba else fill_price,
                trade_context="entry",
                related_trade_id=trade_id,
                exchange_timestamp=fill_time,
            )
        except Exception:
            pass

    async def _on_stop_fill_instrumentation(
        self, pre_pos: PositionState, fill_price: float, payload: dict,
    ) -> None:
        """Instrumentation-only: log stop exit after core has closed the position."""
        pv = C.NQ_SPEC["point_value"]
        pnl_pts = (fill_price - pre_pos.entry_price) if pre_pos.direction == Direction.LONG else \
                   (pre_pos.entry_price - fill_price)
        realized_usd = pnl_pts * pv * pre_pos.qty_open
        r = pnl_pts / pre_pos.r_points if pre_pos.r_points > 0 else 0

        if self._recorder and pre_pos.trade_id:
            try:
                await self._recorder.record_exit(
                    trade_id=pre_pos.trade_id,
                    exit_price=Decimal(str(fill_price)),
                    exit_ts=datetime.now(timezone.utc),
                    exit_reason="STOP",
                    realized_r=Decimal(str(round(r, 4))),
                    realized_usd=Decimal(str(round(realized_usd, 2))),
                    duration_bars=pre_pos.bars_since_entry,
                )
            except Exception:
                logger.exception("Error recording stop exit")

        logger.info("STOPPED %s @ %.2f ($%.2f)",
                     pre_pos.trade_id, fill_price, realized_usd)
        if self._kit.active and pre_pos.trade_id:
            try:
                self._kit.log_exit(
                    trade_id=pre_pos.trade_id,
                    exit_price=fill_price,
                    exit_reason="STOP",
                    expected_exit_price=pre_pos.stop_price,
                    mfe_r=pre_pos.peak_mfe_r,
                    mae_r=pre_pos.peak_mae_r,
                    mfe_price=pre_pos.highest_since_entry if pre_pos.direction == Direction.LONG else pre_pos.lowest_since_entry,
                    mae_price=pre_pos.lowest_since_entry if pre_pos.direction == Direction.LONG else pre_pos.highest_since_entry,
                    session_transitions=getattr(pre_pos, 'session_transitions_log', None) or None,
                    **fill_runtime_refs(payload.get("oms_order_id", ""), payload, fill_qty=payload.get("qty", pre_pos.qty_open), is_exit=True),
                )
                _ba = self._get_bid_ask()
                self._kit.on_orderbook_context(
                    pair=self._symbol,
                    best_bid=_ba[0] if _ba else fill_price,
                    best_ask=_ba[1] if _ba else fill_price,
                    trade_context="exit",
                    related_trade_id=pre_pos.trade_id,
                )
            except Exception:
                pass

    async def _on_terminal(self, oms_id: str | None) -> None:
        if not oms_id:
            return

        update = VdubOrderUpdate(
            oms_order_id=oms_id,
            status="cancelled",
            timestamp=datetime.now(timezone.utc),
        )
        core_state = build_core_runtime_state(self)
        core_state, actions, events = vdub_core_logic.on_order_update(core_state, update)
        apply_core_runtime_state(self, core_state)

        # Dispatch actions
        for action in actions:
            if isinstance(action, FlattenPosition):
                reason = action.reason
                if reason == "FLATTEN_RESUBMIT":
                    if not self.positions:
                        continue
                    self._pending_flatten_instrumentation.pop(oms_id, None)
                    logger.critical(
                        "FLATTEN ORDER %s CANCELLED/REJECTED -- resubmitting emergency flatten",
                        oms_id,
                    )
                    receipt = await self._oms.submit_intent(Intent(
                        intent_type=IntentType.FLATTEN,
                        strategy_id=C.STRATEGY_ID, instrument_symbol=self._symbol,
                    ))
                    self._last_flatten_oms_id = (
                        receipt.oms_order_id if receipt and receipt.oms_order_id else None
                    )
                elif reason == "STOP_LOST":
                    pos_id = action.metadata.get("pos_id", "")
                    pos = next((p for p in self.positions if p.trade_id == pos_id), None)
                    if pos:
                        logger.error("STOP ORDER LOST for %s -- flattening position", pos.trade_id)
                        await self._flatten_position(pos, "STOP_LOST")

        for event in events:
            if event.code == "ENTRY_CANCELLED":
                logger.info("Order terminal: %s", oms_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_position(self, direction: Direction) -> Optional[PositionState]:
        for p in self.positions:
            if p.direction == direction and p.qty_open > 0:
                return p
        return None

    def _unrealized_r(self, pos: PositionState, price: float) -> float:
        if pos.r_points <= 0:
            return 0.0
        if pos.direction == Direction.LONG:
            return (price - pos.entry_price) / pos.r_points
        return (pos.entry_price - price) / pos.r_points

    def _compute_open_risk(self) -> float:
        pv = C.NQ_SPEC["point_value"]
        return sum(p.r_points * pv * p.qty_open for p in self.positions if p.qty_open > 0)

    def _safe_atr15(self) -> float:
        if len(self._atr15) == 0 or np.isnan(self._atr15[-1]):
            return 0.0
        return float(self._atr15[-1])

    def _get_spread_ticks(self) -> Optional[float]:
        """Current bid-ask spread in ticks, or None if unavailable."""
        try:
            for t in self._ib.ib.tickers():
                if t.contract and t.contract.symbol == self._symbol:
                    if t.bid > 0 and t.ask > 0:
                        return (t.ask - t.bid) / C.NQ_SPEC["tick"]
        except Exception:
            pass
        return None

    def _get_bid_ask(self) -> tuple[float, float] | None:
        """Return (bid, ask) from IB tickers, or None if unavailable."""
        try:
            for t in self._ib.ib.tickers():
                if t.contract and t.contract.symbol == self._symbol:
                    if t.bid > 0 and t.ask > 0:
                        return (t.bid, t.ask)
        except Exception:
            pass
        return None

    def _safe_atr1h(self) -> float:
        if len(self._atr1h) == 0 or np.isnan(self._atr1h[-1]):
            return 0.0
        return float(self._atr1h[-1])

    # ------------------------------------------------------------------
    # Historical data (IB integration)
    # ------------------------------------------------------------------

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

    async def _load_initial_bars(self) -> None:
        logger.info("Loading initial bars")
        await self._fetch_bars()
        self._update_indicators()
        self._update_regime()
        logger.info(
            "Init: DT=%+d VS=%s 1H=%+d ATR15=%.2f ATR1H=%.2f pivots=%d",
            self.regime.daily_trend, self.regime.vol_state.value,
            self.regime.trend_1h, self._safe_atr15(), self._safe_atr1h(),
            len(self._pivots_1h),
        )

    async def _fetch_bars(self, request_kind: str = "recurring") -> None:
        if not self._ib.ib.isConnected():
            if not getattr(self, '_fetch_disconn_logged', False):
                logger.warning("Skipping bar fetch — IB not connected")
                self._fetch_disconn_logged = True
            return
        self._fetch_disconn_logged = False
        nq = self._get_contract(self._symbol)
        es = self._get_contract("ES")
        for c in [nq, es]:
            if c is not None:
                await self._ib.ib.qualifyContractsAsync(c)

        # 15m NQ
        bars = await self._req_bars(nq, "30 D", "15 mins", request_kind=request_kind)
        if bars:
            remember_idle_market_bars(self, bars, symbol=self._symbol, timeframe="15m")
            self._c15 = np.array([b.close for b in bars], dtype=float)
            self._h15 = np.array([b.high for b in bars], dtype=float)
            self._l15 = np.array([b.low for b in bars], dtype=float)
            self._v15 = np.array([b.volume for b in bars], dtype=float)
            self._t15 = [getattr(b, 'date', datetime.now(timezone.utc)) for b in bars]

        # 1H NQ
        bars = await self._req_bars(nq, "120 D", "1 hour", request_kind=request_kind)
        if bars:
            self._c1h = np.array([b.close for b in bars], dtype=float)
            self._h1h = np.array([b.high for b in bars], dtype=float)
            self._l1h = np.array([b.low for b in bars], dtype=float)
            self._v1h = np.array([b.volume for b in bars], dtype=float)
            self._t1h = [getattr(b, 'date', datetime.now(timezone.utc)) for b in bars]

        # Daily ES (regime)
        bars = await self._req_bars(es, "2 Y", "1 day", use_rth=True, request_kind=request_kind)
        if bars:
            self._es_c = np.array([b.close for b in bars], dtype=float)
            self._es_h = np.array([b.high for b in bars], dtype=float)
            self._es_l = np.array([b.low for b in bars], dtype=float)

        # Daily NQ (VWAP-A anchor pivots)
        bars = await self._req_bars(nq, "200 D", "1 day", use_rth=True, request_kind=request_kind)
        if bars:
            nq_d_h = np.array([b.high for b in bars], dtype=float)
            nq_d_l = np.array([b.low for b in bars], dtype=float)
            self._t_nq_daily = [getattr(b, 'date', datetime.now(timezone.utc)) for b in bars]
            self._pivots_daily = ind.confirmed_pivots(nq_d_h, nq_d_l, C.NCONFIRM_D)

    async def _req_bars(
        self, contract: Any, duration: str, bar_size: str, use_rth: bool = False,
        request_kind: str = "recurring",
    ) -> list | None:
        if contract is None:
            return None
        try:
            bars = await req_panama_adjusted_historical_data(
                self._ib, contract,
                symbol=getattr(self, "_symbol", C.DEFAULT_SYMBOL),
                endDateTime="", durationStr=duration,
                barSizeSetting=bar_size, whatToShow="TRADES",
                useRTH=use_rth, formatDate=1, request_kind=request_kind,
                completed_only=True,
            )
            return bars if bars else None
        except Exception:
            logger.exception("Error fetching %s bars", bar_size)
            return None

    def _get_contract(self, sym: str) -> Any | None:
        try:
            from ib_async import ContFuture
            return ContFuture(symbol=sym, exchange="CME", currency="USD")
        except Exception:
            logger.warning("Cannot build contract for %s", sym)
            return None
