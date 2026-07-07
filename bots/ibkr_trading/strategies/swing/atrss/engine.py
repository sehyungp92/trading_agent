"""ETRS vFinal — main async strategy engine."""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import math
from collections import deque
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import numpy as np

from libs.broker_ibkr.risk_support.tick_rules import round_to_tick
from libs.oms.models.events import OMSEventType
from libs.oms.models.intent import Intent, IntentType
from libs.oms.models.order import (
    EntryPolicy,
    OMSOrder,
    OrderRole,
    OrderSide,
    OrderType,
    RiskContext,
)
from libs.oms.instrumentation.runtime_refs import fill_runtime_refs
from libs.oms.risk.calculator import RiskCalculator
from libs.services.trade_recorder import TradeRecorder
from strategies.core.actions import FlattenPosition, ReplaceProtectiveStop, SubmitAddOnEntry, SubmitEntry, SubmitPartialExit, SubmitProtectiveStop
from strategies.core.idle_market import (
    maybe_record_idle_market_observation,
    remember_idle_market_bars,
)

from . import allocator, signals, stops
from .core import logic as atrss_core_logic
from .core.logic import apply_core_state as apply_core_runtime_state
from .core.logic import build_core_state as build_core_runtime_state
from .core.serializers import restore_state as restore_core_state
from .core.serializers import snapshot_state as snapshot_core_state
from .core.state import (
    ATRSSAddOnARequest,
    ATRSSEntryRequest,
    ATRSSFill,
    ATRSSFlattenRequest,
    ATRSSOrderUpdate,
    ATRSSPartialExitRequest,
    ATRSSStopUpdateRequest,
)
from .config import (
    ADDON_A_SIZE_MULT,
    ADDON_B_ENABLED,
    ADDON_B_SIZE_MULT,
    ARM_WINDOW_HOURS,
    BE_TRIGGER_R,
    CHANDELIER_TRIGGER_R,
    DYNAMIC_RISK_STRONG_TREND_MULT,
    DYNAMIC_RISK_WEAK_TREND_MULT,
    EARLY_STALL_ENABLED,
    EARLY_STALL_CHECK_HOURS,
    EARLY_STALL_MFE_THRESHOLD,
    EARLY_STALL_PARTIAL_FRAC,
    MAX_ENTRY_SLIP_ATR,
    MAX_HOLD_HOURS,
    MOMENTUM_TOLERANCE_ATR,
    ORDER_EXPIRY_HOURS,
    QUALITY_GATE_THRESHOLD,
    STALL_CHECK_HOURS,
    STALL_EXIT_ENABLED,
    STALL_MFE_THRESHOLD,
    STRATEGY_ID,
    SYMBOL_CONFIGS,
    SymbolConfig,
    TP1_FRAC,
    TP1_R,
    TP2_FRAC,
    TP2_R,
    TREND_STOP_TIGHTENING,
)
from .indicators import compute_daily_state, compute_hourly_state
from .models import (
    BreakoutArmState,
    Candidate,
    CandidateType,
    DailyState,
    Direction,
    HaltState,
    HourlyState,
    LegType,
    PositionBook,
    PositionLeg,
    ReentryState,
    Regime,
)

logger = logging.getLogger(__name__)
halt_audit_logger = logging.getLogger(f"{__name__}.halt_audit")


# ---------------------------------------------------------------------------
# RTH helpers
# ---------------------------------------------------------------------------

def _is_rth(dt_utc: datetime, market_calendar=None) -> bool:
    """Return True if *dt_utc* falls within NYSE RTH (09:30-16:00 ET)."""
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    local = dt_utc.astimezone(et)
    if local.weekday() >= 5:  # Sat/Sun
        return False
    if market_calendar and not market_calendar.is_trading_day(local.date()):
        return False
    t = local.hour * 60 + local.minute
    return 9 * 60 + 30 <= t < 16 * 60


def _is_entry_restricted(dt_utc: datetime) -> bool:
    """Return True if within first 5 min after open or last 5 min before close.

    Entries allowed 09:35-15:55 ET.
    """
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    local = dt_utc.astimezone(et)
    if local.weekday() >= 5:
        return True  # no entries on weekends
    t = local.hour * 60 + local.minute
    # First 5 min: 09:30 <= t < 09:35
    if 9 * 60 + 30 <= t < 9 * 60 + 35:
        return True
    # Last 5 min: 15:55 <= t < 16:00
    if 15 * 60 + 55 <= t < 16 * 60:
        return True
    return False


class ATRSSEngine:
    """Core ETRS vFinal event-driven engine."""

    def __init__(
        self,
        ib_session: Any,
        oms_service: Any,
        instruments: dict[str, Any],
        config: dict[str, SymbolConfig],
        trade_recorder: TradeRecorder | None = None,
        equity: float = 100_000.0,
        market_calendar: Any | None = None,
        kit: Any | None = None,
        equity_offset: float = 0.0,
        equity_alloc_pct: float = 1.0,
        disable_background_tasks: bool = False,
    ) -> None:
        self._ib = ib_session
        self._oms = oms_service
        self._instruments = instruments
        self._config = config
        self._recorder = trade_recorder
        self._equity = equity
        self._equity_offset = equity_offset
        self._equity_alloc_pct = equity_alloc_pct
        self._market_cal = market_calendar
        self._kit = kit
        self._disable_background_tasks = bool(disable_background_tasks)

        # Wire drawdown tracker with initial equity
        if self._kit and self._kit.ctx and self._kit.ctx.drawdown_tracker:
            self._kit.ctx.drawdown_tracker.update_equity(self._equity)

        # Per-symbol state
        self.daily_states: dict[str, DailyState] = {}
        self.hourly_states: dict[str, HourlyState] = {}
        self.positions: dict[str, PositionBook] = {}
        self.reentry_states: dict[str, ReentryState] = {}
        self.pending_orders: dict[str, dict] = {}  # oms_order_id → metadata
        self.contracts: dict[str, Any] = {}        # symbol → (Contract, spec)
        # Previous daily trend_dir per symbol (stored before update for reverse detection)
        self._prev_trend_dirs: dict[str, Direction] = {}
        # Halt/reopen state (spec Section 12)
        self.halt_states: dict[str, HaltState] = {}
        # Pending reverse candidates from bias flip exits (spec Section 7.4)
        self._pending_reverses: list[Candidate] = []
        # Track flatten order IDs for fill cleanup
        self._pending_flattens: dict[str, dict] = {}
        # Track reopen timestamps for one-bar delay (spec Section 10.3)
        self._reopen_at: dict[str, datetime] = {}
        # Breakout arm state per symbol (spec Section 7.2)
        self.breakout_arm_states: dict[str, BreakoutArmState] = {}
        self._risk_halted = False
        self._risk_halt_reason = ""
        # Signal evolution ring buffer for TA alpha decay detector
        self._signal_ring: dict[str, deque] = {}  # sym → deque of snapshots

        # Async tasks
        self._event_task: asyncio.Task | None = None
        self._cycle_task: asyncio.Task | None = None
        self._event_queue: asyncio.Queue | None = None
        self._running = False

        # Diagnostic pulse state
        self._last_decision_code: str = "IDLE"
        self._last_decision_details: dict = {}
        self._last_bar_ts: datetime | None = None
        self._cycles_completed: int = 0
        self._symbol_last_bar_ts: dict[str, datetime] = {}

    def _record_decision(self, code: str, details: dict | None = None) -> None:
        if maybe_record_idle_market_observation(
            self,
            code,
            strategy_id=STRATEGY_ID,
            build_core_state=lambda: build_core_runtime_state(self),
            apply_core_state=lambda state: apply_core_runtime_state(self, state),
            on_bar=atrss_core_logic.on_bar,
            default_symbol="",
            default_timeframe="1h",
        ):
            return
        self._last_decision_code = code
        self._last_decision_details = details or {}

    def health_status(self) -> dict:
        return {
            "strategy_id": STRATEGY_ID,
            "running": self._running,
            "last_decision_code": self._last_decision_code,
            "last_decision_details": self._last_decision_details,
            "last_bar_ts": self._last_bar_ts.isoformat() if self._last_bar_ts else None,
        }

    def liveness_payload(self) -> dict:
        return {
            "bars_processed": self._cycles_completed,
            "symbol_freshness": {
                sym: ts.isoformat() for sym, ts in self._symbol_last_bar_ts.items()
            },
        }

    def snapshot_state(self) -> dict[str, Any]:
        return snapshot_core_state(build_core_runtime_state(self))

    async def hydrate(self, snapshot: dict[str, Any]) -> None:
        if not snapshot:
            return
        apply_core_runtime_state(self, restore_core_state(snapshot))

    def _apply_core_bar_transition(
        self,
        *,
        bar_ts: datetime | None,
        **payload: Any,
    ):
        core_state = build_core_runtime_state(self)
        new_state, actions, events = atrss_core_logic.on_bar(
            core_state,
            bar_ts=bar_ts,
            **payload,
        )
        apply_core_runtime_state(self, new_state)
        return actions, events

    # ------------------------------------------------------------------
    # Signal evolution tracking (for TA alpha decay detector)
    # ------------------------------------------------------------------

    def _snapshot_signal_state(self, sym: str, quality_score: float,
                               daily: Any, hourly: Any, regime: str) -> None:
        """Capture signal components for evolution tracking."""
        ring = self._signal_ring.setdefault(sym, deque(maxlen=10))
        ring.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "quality_score": quality_score,
            "daily_adx": getattr(daily, "adx", None),
            "daily_ema_sep": getattr(daily, "ema_sep_pct", None),
            "regime": regime,
            "hourly_ema_mom": getattr(hourly, "ema_mom", None),
        })

    def _build_signal_evolution(self, sym: str, n: int = 5) -> list[dict]:
        """Return last N signal snapshots with bars_ago index."""
        ring = self._signal_ring.get(sym)
        if not ring:
            return []
        items = list(ring)[-n:]
        return [{"bars_ago": n - 1 - i, **s} for i, s in enumerate(items)]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to events, load initial bar history, start hourly scheduler."""
        logger.info("ATRSS engine starting …")
        self._running = True

        # Subscribe to OMS events
        self._event_queue = self._oms.stream_events(STRATEGY_ID)
        self._event_task = asyncio.create_task(self._process_events())

        # Resolve contracts for each symbol
        cf = getattr(self._ib, "_contract_factory", None)
        for sym, cfg in self._config.items():
            if cf is None:
                break
            try:
                contract, spec = await cf.resolve(
                    sym,
                    cfg.contract_expiry,
                    instrument=self._instruments.get(sym),
                )
                self.contracts[sym] = (contract, spec)
            except Exception as e:
                logger.warning("Could not resolve contract for %s: %s", sym, e)

        # Initialize re-entry states and breakout arm states
        for sym in self._config:
            self.reentry_states.setdefault(sym, ReentryState())
            self.breakout_arm_states.setdefault(sym, BreakoutArmState())

        if not self._disable_background_tasks:
            # Load initial bar history and compute initial daily states
            await self._load_initial_bars()

            # Start hourly cycle scheduler
            self._cycle_task = asyncio.create_task(self._hourly_scheduler())
        logger.info("ATRSS engine started for %s", list(self._config.keys()))

    async def stop(self) -> None:
        """Cancel all pending, cleanup."""
        logger.info("ATRSS engine stopping …")
        self._running = False

        if self._cycle_task:
            self._cycle_task.cancel()
            try:
                await self._cycle_task
            except asyncio.CancelledError:
                pass

        if self._event_task:
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass

        # Cancel pending orders
        for oms_id, meta in list(self.pending_orders.items()):
            try:
                receipt = await self._oms.submit_intent(
                    Intent(
                        intent_type=IntentType.CANCEL_ORDER,
                        strategy_id=STRATEGY_ID,
                        target_oms_order_id=oms_id,
                    )
                )
                logger.info("Cancelled pending order %s: %s", oms_id, receipt.result)
            except Exception as e:
                logger.warning("Error cancelling order %s: %s", oms_id, e)

        logger.info("ATRSS engine stopped")

    # ------------------------------------------------------------------
    # Hourly scheduler
    # ------------------------------------------------------------------

    async def _hourly_scheduler(self) -> None:
        """Sleep until the next hour boundary, then run the hourly cycle."""
        while self._running:
            now = datetime.now(timezone.utc)
            # Next whole hour
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=10, microsecond=0)
            wait_secs = (next_hour - now).total_seconds()
            logger.debug("Next cycle in %.0fs at %s", wait_secs, next_hour)
            await asyncio.sleep(wait_secs)
            if not self._running:
                break
            try:
                await self._hourly_cycle()
            except Exception:
                logger.exception("Error in hourly cycle")

    # ------------------------------------------------------------------
    # Core hourly cycle
    # ------------------------------------------------------------------

    async def _refresh_equity(self) -> None:
        """Fetch current account equity from IB (spec S9, audit H5)."""
        try:
            accounts = self._ib.ib.managedAccounts()
            if accounts:
                account = accounts[0]
                for item in self._ib.ib.accountValues():
                    if item.tag == "NetLiquidation" and item.currency == "USD" and item.account == account:
                        raw = float(item.value)
                        if raw > 0:
                            new_equity = raw * self._equity_alloc_pct + self._equity_offset
                            self._equity = new_equity
                            if self._kit and self._kit.ctx and self._kit.ctx.drawdown_tracker:
                                self._kit.ctx.drawdown_tracker.update_equity(new_equity)
                            logger.debug("Equity updated to $%.2f", new_equity)
                        return
        except Exception:
            logger.warning("Could not refresh equity from IB, using $%.2f", self._equity)

    async def _hourly_cycle(self) -> None:
        """Execute the 10-step hourly cycle per spec."""
        now = datetime.now(timezone.utc)
        logger.info("=== Hourly cycle %s ===", now.isoformat())
        self._last_bar_ts = datetime.now(timezone.utc)
        self._cycles_completed += 1

        # Refresh equity from broker before allocation
        await self._refresh_equity()

        for sym in self._config:
            try:
                await self._cycle_symbol(sym, now)
            except Exception:
                logger.exception("Error in cycle for %s", sym)

        # 8 – Portfolio allocator across all symbols
        all_candidates = self._collect_candidates(now)
        # Merge reverse candidates generated during position management (spec Section 7.4)
        all_candidates.extend(self._pending_reverses)
        self._pending_reverses.clear()
        if all_candidates:
            # Exclude positions with pending flatten orders from heat calculation
            # so reverse candidates aren't blocked by the exiting position's heat.
            active_positions = {
                sym: pos for sym, pos in self.positions.items()
                if sym not in self._pending_flattens
            }
            accepted = allocator.allocate(
                all_candidates,
                active_positions,
                self.daily_states,
                self._equity,
                self._instruments,
                self.hourly_states,
            )

            # Hook 3: Log allocator rejections as missed opportunities
            if self._kit:
                accepted_syms = {c.symbol for c in accepted}
                for cand in all_candidates:
                    if cand.symbol not in accepted_syms:
                        self._kit.log_missed(
                            pair=cand.symbol,
                            side="LONG" if cand.direction == Direction.LONG else "SHORT",
                            signal=cand.type.value,
                            signal_id=f"{cand.symbol}_{cand.type.value}_{now.isoformat()}",
                            signal_strength=cand.quality_score if hasattr(cand, 'quality_score') else 0.5,
                            blocked_by="allocator",
                            block_reason="rejected by portfolio allocator",
                        )

            for cand in accepted:
                await self._submit_entry(cand)

        has_positions = any(
            pos.direction != Direction.FLAT for pos in self.positions.values()
        )
        if not all_candidates and not has_positions:
            self._record_decision("NO_SIGNAL", {"symbols": list(self._config.keys())})
        elif has_positions and not all_candidates:
            self._record_decision("MANAGING_POSITION", {
                "position_count": sum(1 for p in self.positions.values() if p.direction != Direction.FLAT),
            })

        # Hook 1: Market snapshot + regime classification (post-decision, never affects trading)
        if self._kit:
            for sym in self._config:
                self._kit.classify_regime(sym)
                self._kit.capture_snapshot(sym)

    async def _cycle_symbol(self, sym: str, now: datetime) -> None:
        """Per-symbol steps 1-7 of the hourly cycle."""
        self._symbol_last_bar_ts[sym] = datetime.now(timezone.utc)
        cfg = self._config[sym]

        # 1 – Fetch latest bars
        closes_d, highs_d, lows_d, last_daily_date = await self._fetch_daily_bars(sym, cfg)
        closes_h, highs_h, lows_h, opens_h = await self._fetch_hourly_bars(sym, cfg)

        # Halt detection (spec Section 12.1)
        was_halted = self.halt_states.get(sym, HaltState()).is_halted
        if closes_h is None:
            await self._mark_halted(sym, now)
            if closes_d is None:
                return
            # Still update daily state even if hourly is unavailable
        elif was_halted:
            # Transition from halted → tradable
            await self._on_reopen(sym, now)

        if closes_d is None or closes_h is None:
            logger.warning("No bar data for %s, skipping", sym)
            return

        # 2 – Update daily state (store previous trend_dir for reverse detection)
        prev_daily = self.daily_states.get(sym)
        self._prev_trend_dirs[sym] = prev_daily.trend_dir if prev_daily else Direction.FLAT
        daily = compute_daily_state(closes_d, highs_d, lows_d, prev_daily, cfg, last_daily_date)
        self.daily_states[sym] = daily

        if self._kit and closes_d is not None and len(closes_d) > 0:
            self._kit.record_close(sym, float(closes_d[-1]))

        # 3 – Compute hourly indicators
        hourly = compute_hourly_state(closes_h, highs_h, lows_h, daily, cfg, now, opens_h)
        self.hourly_states[sym] = hourly

        # 4 – Update reset flags
        reentry = self.reentry_states[sym]
        if hourly.close < hourly.ema_pull:
            reentry.reset_seen_long = True
        if hourly.close > hourly.ema_pull:
            reentry.reset_seen_short = True

        # 5 – Manage open positions
        if sym in self.positions and self.positions[sym].direction != Direction.FLAT:
            await self._manage_position(sym, hourly, daily, now)

        # 6 – Cancel expired orders
        await self._cancel_expired_orders(sym, now)

        # 7 – Check breakout arm events (spec Section 7.2)
        arm_dir = signals.check_breakout_arm(hourly, daily)
        if arm_dir != Direction.FLAT:
            arm_state = self.breakout_arm_states.get(sym, BreakoutArmState())
            arm_state.breakout_armed_dir = arm_dir
            arm_state.breakout_armed_until = now + timedelta(hours=ARM_WINDOW_HOURS)
            arm_state.breakout_arm_high = hourly.high
            arm_state.breakout_arm_low = hourly.low
            self.breakout_arm_states[sym] = arm_state
            logger.info("%s breakout armed %s until %s", sym, arm_dir.name,
                        arm_state.breakout_armed_until.isoformat())

        # Expire stale arm
        arm_state = self.breakout_arm_states.get(sym)
        if arm_state and arm_state.breakout_armed_until and now > arm_state.breakout_armed_until:
            arm_state.breakout_armed_dir = Direction.FLAT
            arm_state.breakout_armed_until = None

    def _collect_candidates(self, now: datetime) -> list[Candidate]:
        """Generate candidates across all symbols (steps 7a-c)."""
        all_candidates: list[Candidate] = []

        # Entry time restrictions (spec Section 14.1)
        if _is_entry_restricted(now):
            return all_candidates

        for sym, cfg in self._config.items():
            daily = self.daily_states.get(sym)
            hourly = self.hourly_states.get(sym)
            if daily is None or hourly is None:
                continue

            # Skip if halted (spec Section 12.2)
            halt = self.halt_states.get(sym)
            if halt and halt.is_halted:
                continue

            # Skip SHORT generation if disabled for this symbol (D8)
            if daily.trend_dir == Direction.SHORT and not cfg.shorts_enabled:
                if self._kit:
                    self._kit.log_missed(
                        pair=sym, side="SHORT", signal="short_disabled",
                        signal_id=f"{sym}_short_disabled_{now.isoformat()}",
                        signal_strength=0.0, blocked_by="short_gate",
                        block_reason="shorts disabled for symbol",
                    )
                continue

            # Per-symbol short gate (R1)
            if daily.trend_dir == Direction.SHORT:
                _short_safety_passed = signals.short_safety_ok(daily)
                if self._kit:
                    self._kit.on_filter_decision(
                        pair=sym, filter_name="short_safety_ok",
                        passed=_short_safety_passed,
                        threshold=0.0,
                        actual_value=daily.ema_fast_slope_5,
                        signal_name="atrss_entry", strategy_id=STRATEGY_ID,
                    )
                if not _short_safety_passed:
                    if self._kit:
                        self._kit.log_missed(
                            pair=sym, side="SHORT", signal="short_safety_fail",
                            signal_id=f"{sym}_short_safety_{now.isoformat()}",
                            signal_strength=0.0, blocked_by="short_safety_ok",
                            block_reason=f"EMA fast slope {daily.ema_fast_slope_5:.4f} > 0",
                        )
                    continue

                _short_gate_passed = signals.short_symbol_gate(sym, daily, hourly)
                if self._kit:
                    _adx_thresholds = {"GLD": 22.0, "USO": 22.0}
                    self._kit.on_filter_decision(
                        pair=sym, filter_name="short_symbol_gate",
                        passed=_short_gate_passed,
                        threshold=_adx_thresholds.get(sym, 0.0),
                        actual_value=daily.adx,
                        signal_name="atrss_entry", strategy_id=STRATEGY_ID,
                    )
                if not _short_gate_passed:
                    if self._kit:
                        self._kit.log_missed(
                            pair=sym, side="SHORT", signal="short_gate_fail",
                            signal_id=f"{sym}_short_gate_{now.isoformat()}",
                            signal_strength=0.0, blocked_by="short_gate",
                            block_reason="short symbol gate failed",
                        )
                    continue

            # Per-symbol time/day blocking (R2)
            if cfg.blocked_hours_et or cfg.blocked_weekdays:
                from zoneinfo import ZoneInfo
                et = ZoneInfo("America/New_York")
                dt_et = now.astimezone(et)
                if dt_et.hour in cfg.blocked_hours_et:
                    continue
                if dt_et.weekday() in cfg.blocked_weekdays:
                    continue

            # One-bar delay after reopen (spec Section 10.3)
            reopen_time = self._reopen_at.get(sym)
            if reopen_time and (now - reopen_time) < timedelta(hours=1):
                continue

            pos = self.positions.get(sym)
            reentry = self.reentry_states.get(sym, ReentryState())

            # --- Base entries (only if no open position for this symbol) ---
            has_position = pos is not None and pos.direction != Direction.FLAT

            if not has_position:
                # 7a – Pullback (exempt from momentum_ok per backtest)
                pb_dir = signals.pullback_signal(hourly, daily)

                # Emit indicator snapshot at pullback evaluation
                if self._kit:
                    self._kit.on_indicator_snapshot(
                        pair=sym,
                        indicators={
                            "ema_pull": hourly.ema_pull,
                            "ema_mom": hourly.ema_mom,
                            "atr_hourly": hourly.atrh,
                            "atr_daily": daily.atr20,
                            "adx": daily.adx,
                            "plus_di": daily.plus_di,
                            "minus_di": daily.minus_di,
                            "donchian_high": hourly.donchian_high,
                            "donchian_low": hourly.donchian_low,
                            "ema_sep_pct": daily.ema_sep_pct,
                            "regime_score": daily.score,
                        },
                        signal_name="atrss_pullback",
                        signal_strength=0.0,
                        decision="enter" if pb_dir != Direction.FLAT else "skip",
                        strategy_id=STRATEGY_ID,
                        exchange_timestamp=now,
                    )
                    if pb_dir != Direction.FLAT:
                        _snap = self._kit.capture_snapshot(sym)
                        if _snap:
                            self._kit.on_orderbook_context(
                                pair=sym,
                                best_bid=_snap.get("bid", 0),
                                best_ask=_snap.get("ask", 0),
                                trade_context="signal_eval",
                                exchange_timestamp=now,
                            )

                if pb_dir != Direction.FLAT:
                    quality_score = signals.compute_entry_quality(hourly, daily, pb_dir)
                    self._snapshot_signal_state(sym, quality_score, daily, hourly, getattr(daily, 'regime', 'unknown'))

                    # Emit quality gate filter decision
                    if self._kit:
                        self._kit.on_filter_decision(
                            pair=sym, filter_name="entry_quality_gate",
                            passed=quality_score >= QUALITY_GATE_THRESHOLD,
                            threshold=QUALITY_GATE_THRESHOLD,
                            actual_value=quality_score,
                            signal_name="atrss_pullback",
                            signal_strength=quality_score / 7.0,
                            strategy_id=STRATEGY_ID,
                        )

                    if quality_score < QUALITY_GATE_THRESHOLD:
                        # Hook 2: Missed opportunity — quality gate
                        if self._kit:
                            self._kit.log_missed(
                                pair=sym,
                                side="LONG" if pb_dir == Direction.LONG else "SHORT",
                                signal="pullback",
                                signal_id=f"{sym}_pb_{now.isoformat()}",
                                signal_strength=quality_score,
                                blocked_by="quality_gate",
                                block_reason=f"quality {quality_score:.2f} < {QUALITY_GATE_THRESHOLD}",
                            )
                        continue
                    if signals.same_direction_reentry_allowed(reentry, pb_dir, now, daily.regime, daily.trend_dir):
                        cand = self._build_candidate(
                            sym, CandidateType.PULLBACK, pb_dir, hourly, daily, cfg,
                        )
                        if cand:
                            all_candidates.append(cand)

                # 7b – Breakout pullback (arm-then-pullback, spec S7.2-7.3)
                arm_state = self.breakout_arm_states.get(sym)
                if arm_state and arm_state.breakout_armed_dir != Direction.FLAT:
                    bo_dir = signals.breakout_pullback_signal(
                        hourly, daily, arm_state.breakout_armed_dir,
                        arm_high=arm_state.breakout_arm_high,
                        arm_low=arm_state.breakout_arm_low,
                    )

                    # Emit indicator snapshot at breakout evaluation
                    if self._kit and bo_dir != Direction.FLAT:
                        self._kit.on_indicator_snapshot(
                            pair=sym,
                            indicators={
                                "ema_pull": hourly.ema_pull,
                                "ema_mom": hourly.ema_mom,
                                "atr_hourly": hourly.atrh,
                                "atr_daily": daily.atr20,
                                "adx": daily.adx,
                                "plus_di": daily.plus_di,
                                "minus_di": daily.minus_di,
                                "donchian_high": hourly.donchian_high,
                                "donchian_low": hourly.donchian_low,
                                "arm_high": arm_state.breakout_arm_high,
                                "arm_low": arm_state.breakout_arm_low,
                            },
                            signal_name="atrss_breakout",
                            signal_strength=0.0,
                            decision="enter",
                            strategy_id=STRATEGY_ID,
                            exchange_timestamp=now,
                        )
                        _snap = self._kit.capture_snapshot(sym)
                        if _snap:
                            self._kit.on_orderbook_context(
                                pair=sym,
                                best_bid=_snap.get("bid", 0),
                                best_ask=_snap.get("ask", 0),
                                trade_context="signal_eval",
                                exchange_timestamp=now,
                            )

                    if bo_dir != Direction.FLAT:
                        quality_score = signals.compute_entry_quality(hourly, daily, bo_dir)

                        # Emit quality gate filter decision (breakout)
                        if self._kit:
                            self._kit.on_filter_decision(
                                pair=sym, filter_name="entry_quality_gate",
                                passed=quality_score >= QUALITY_GATE_THRESHOLD,
                                threshold=QUALITY_GATE_THRESHOLD,
                                actual_value=quality_score,
                                signal_name="atrss_breakout",
                                signal_strength=quality_score / 7.0,
                                strategy_id=STRATEGY_ID,
                            )

                        if quality_score < QUALITY_GATE_THRESHOLD:
                            # Hook 2: Missed opportunity — quality gate (breakout)
                            if self._kit:
                                self._kit.log_missed(
                                    pair=sym,
                                    side="LONG" if bo_dir == Direction.LONG else "SHORT",
                                    signal="breakout_pullback",
                                    signal_id=f"{sym}_bo_{now.isoformat()}",
                                    signal_strength=quality_score,
                                    blocked_by="quality_gate",
                                    block_reason=f"quality {quality_score:.2f} < {QUALITY_GATE_THRESHOLD}",
                                )
                            continue

                        _mom_ok = signals.momentum_ok(hourly, bo_dir)
                        _reentry_ok = signals.same_direction_reentry_allowed(
                            reentry, bo_dir, now, daily.regime, daily.trend_dir,
                        )

                        # Emit momentum filter decision
                        if self._kit:
                            _mom_tol = MOMENTUM_TOLERANCE_ATR * hourly.atrh
                            _mom_actual = (hourly.close - hourly.ema_mom if bo_dir == Direction.LONG
                                           else hourly.ema_mom - hourly.close)
                            self._kit.on_filter_decision(
                                pair=sym, filter_name="momentum_ok",
                                passed=_mom_ok, threshold=_mom_tol,
                                actual_value=_mom_actual,
                                signal_name="atrss_breakout",
                                strategy_id=STRATEGY_ID,
                            )

                        if _mom_ok and _reentry_ok:
                            cand = self._build_candidate(
                                sym, CandidateType.BREAKOUT, bo_dir, hourly, daily, cfg,
                            )
                            if cand:
                                all_candidates.append(cand)
                                # Consume arm on trigger
                                arm_state.breakout_armed_dir = Direction.FLAT
                                arm_state.breakout_armed_until = None

            # --- Add-on B entries (only if base position open) ---
            # Note: Add-on A is event-driven inside _manage_position, not collected here
            if has_position and pos is not None and ADDON_B_ENABLED:
                # Add-on B
                if signals.addon_b_eligible(pos, hourly, daily):
                    cand = self._build_addon_b_candidate(
                        sym, pos, hourly, daily, cfg,
                    )
                    if cand:
                        all_candidates.append(cand)

        return all_candidates

    # ------------------------------------------------------------------
    # Candidate builders
    # ------------------------------------------------------------------

    def _build_candidate(
        self,
        sym: str,
        ctype: CandidateType,
        direction: Direction,
        h: HourlyState,
        d: DailyState,
        cfg: SymbolConfig,
    ) -> Candidate | None:
        """Create a base-entry candidate with risk-sized qty."""
        # Trigger = signal_high + 1 tick (LONG) or signal_low - 1 tick (SHORT)
        if direction == Direction.LONG:
            trigger = h.high + cfg.tick_size
        else:
            trigger = h.low - cfg.tick_size
        trigger = round_to_tick(trigger, cfg.tick_size)

        d_mult = cfg.daily_mult
        h_mult = cfg.hourly_mult
        if d.regime == Regime.TREND:
            d_mult *= TREND_STOP_TIGHTENING
            h_mult *= TREND_STOP_TIGHTENING
        initial_stop = stops.compute_initial_stop(
            direction, trigger, h, d.atr20, h.atrh,
            d_mult, h_mult, cfg.tick_size,
        )
        # Risk sizing per symbol — regime-adaptive (spec Section 9)
        risk_pct = cfg.base_risk_pct
        if d.regime == Regime.STRONG_TREND and d.score >= 60:
            risk_pct *= DYNAMIC_RISK_STRONG_TREND_MULT
        elif d.regime == Regime.TREND and d.score < 45:
            risk_pct *= DYNAMIC_RISK_WEAK_TREND_MULT
        inst = self._instruments.get(sym)
        if inst is None:
            return None
        qty = allocator.compute_position_size(
            trigger, initial_stop, self._equity, risk_pct, inst.point_value,
        )
        if qty <= 0:
            return None

        # Month-based size reduction (e.g. QQQ half size in December)
        if cfg.size_reduction_months and h.time is not None:
            month = h.time.month
            for m, frac in cfg.size_reduction_months:
                if month == m:
                    qty = max(1, int(qty * frac))
                    break

        return Candidate(
            symbol=sym,
            type=ctype,
            direction=direction,
            trigger_price=trigger,
            initial_stop=initial_stop,
            qty=qty,
            signal_bar=h,
            time=h.time,
            rank_score=d.score,
            atrh=h.atrh,
            tick_size=cfg.tick_size,
        )

    def _build_addon_b_candidate(
        self,
        sym: str,
        pos: PositionBook,
        h: HourlyState,
        d: DailyState,
        cfg: SymbolConfig,
    ) -> Candidate | None:
        """Create an Add-on B candidate (stop order, submitted via allocator)."""
        direction = pos.direction
        # Add-on B uses signal bar trigger like base entries
        if direction == Direction.LONG:
            trigger = h.high + cfg.tick_size
        else:
            trigger = h.low - cfg.tick_size
        trigger = round_to_tick(trigger, cfg.tick_size)

        # ADDON_B: full hybrid stop from signal candle
        d_mult = cfg.daily_mult
        h_mult = cfg.hourly_mult
        if d.regime == Regime.TREND:
            d_mult *= TREND_STOP_TIGHTENING
            h_mult *= TREND_STOP_TIGHTENING
        initial_stop = stops.compute_initial_stop(
            direction, trigger, h, d.atr20, h.atrh,
            d_mult, h_mult, cfg.tick_size,
        )
        inst = self._instruments.get(sym)
        if inst is None:
            return None
        risk_pct = cfg.base_risk_pct
        qty = allocator.compute_position_size(
            trigger, initial_stop, self._equity, risk_pct, inst.point_value,
        )
        # Cap at 0.5 × base qty
        base = pos.base_leg
        if base:
            qty = min(qty, max(1, int(base.qty * ADDON_B_SIZE_MULT)))
        if qty <= 0:
            return None

        return Candidate(
            symbol=sym,
            type=CandidateType.ADDON_B,
            direction=direction,
            trigger_price=trigger,
            initial_stop=initial_stop,
            qty=qty,
            signal_bar=h,
            time=h.time,
            rank_score=d.score,
            atrh=h.atrh,
            tick_size=cfg.tick_size,
        )

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    async def _manage_position(
        self, sym: str, h: HourlyState, d: DailyState, now: datetime,
    ) -> None:
        """Update MFE, apply stop logic (BE → chandelier → profit floor), time decay, Add-on A."""
        pos = self.positions[sym]
        cfg = self._config[sym]
        inst = self._instruments.get(sym)
        if inst is None:
            return

        # C3: Skip management while protective stop is still being placed
        if pos.stop_pending:
            logger.debug("%s skipping management — stop_pending=True", sym)
            return

        # bars_held: only count RTH hours (spec Section 12)
        if _is_rth(now, self._market_cal):
            pos.bars_held += 1

        # M3: Update MFE — mfe_price is initialized to fill_price at position
        # creation, so we only need the directional comparison (no == 0 fallback)
        if pos.direction == Direction.LONG:
            if h.high > pos.mfe_price:
                pos.mfe_price = h.high
        else:
            if h.low < pos.mfe_price:
                pos.mfe_price = h.low

        # Update MAE — mae_price is initialized to fill_price at position creation
        if pos.direction == Direction.LONG:
            if h.low < pos.mae_price:
                pos.mae_price = h.low
        else:
            if h.high > pos.mae_price:
                pos.mae_price = h.high

        base = pos.base_leg
        if base is None:
            return

        risk_per_unit = pos.base_risk_per_unit
        if risk_per_unit > 0:
            if pos.direction == Direction.LONG:
                pos.mfe = (pos.mfe_price - base.entry_price) / risk_per_unit
                pos.mae = (base.entry_price - pos.mae_price) / risk_per_unit
            else:
                pos.mfe = (base.entry_price - pos.mfe_price) / risk_per_unit
                pos.mae = (pos.mae_price - base.entry_price) / risk_per_unit

        # Compute current R for TP / stall / addon decisions
        if risk_per_unit > 0:
            if pos.direction == Direction.LONG:
                cur_r = (h.close - base.entry_price) / risk_per_unit
            else:
                cur_r = (base.entry_price - h.close) / risk_per_unit
        else:
            cur_r = 0.0

        # --- Catastrophic loss cap (spec §13.1a) ---
        if cur_r < -2.0:
            logger.warning(
                "%s catastrophic cap: cur_r=%.2fR, flattening immediately",
                sym, cur_r,
            )
            await self._flatten_position(sym, "FLATTEN_CATASTROPHIC_CAP")
            return

        # --- Partial profit-taking (TP1 + TP2) ---
        base_qty = base.qty
        if base_qty > 1 and not pos.tp1_done and pos.mfe >= TP1_R and cur_r >= TP1_R * 0.8:
            await self._partial_close_position(sym, TP1_FRAC, "TP1")
            pos.tp1_done = True

        if base_qty > 1 and not pos.tp2_done and pos.tp1_done and pos.mfe >= TP2_R and cur_r >= TP2_R * 0.8:
            await self._partial_close_position(sym, TP2_FRAC, "TP2")
            pos.tp2_done = True

        # --- Early stall partial exit (non-developing trade) ---
        if (
            EARLY_STALL_ENABLED
            and not pos.early_partial_done
            and pos.bars_held >= EARLY_STALL_CHECK_HOURS
            and pos.mfe < EARLY_STALL_MFE_THRESHOLD
            and cur_r <= 0.2
        ):
            await self._partial_close_position(sym, EARLY_STALL_PARTIAL_FRAC, "EARLY_STALL_PARTIAL")
            pos.early_partial_done = True

        # --- Stall exit (full flatten for non-developing trade) ---
        if STALL_EXIT_ENABLED and pos.bars_held >= STALL_CHECK_HOURS:
            if pos.mfe < STALL_MFE_THRESHOLD and cur_r <= 0.2:
                logger.info(
                    "%s stall exit: %d bars held, MFE %.2fR, cur_r %.2fR",
                    sym, pos.bars_held, pos.mfe, cur_r,
                )
                await self._flatten_position(sym, "FLATTEN_STALL")
                return

        # --- Time decay exit (M3) ---
        if pos.bars_held >= MAX_HOLD_HOURS:
            if cur_r < 1.0:
                logger.info(
                    "%s time decay exit: %d bars held, profit %.2fR < 1R",
                    sym, pos.bars_held, cur_r,
                )
                await self._flatten_position(sym, "FLATTEN_TIME_DECAY")
                return

        new_stop = pos.current_stop

        # --- BE trigger at +1.5R ---
        if not pos.be_triggered and pos.mfe >= BE_TRIGGER_R:
            be_stop = stops.compute_be_stop(
                pos.direction, base.entry_price, d.atr20, cfg.tick_size,
            )
            # Only move stop in favorable direction
            if pos.direction == Direction.LONG and be_stop > pos.current_stop:
                new_stop = be_stop
                pos.be_triggered = True
                logger.info("%s BE triggered → stop %.4f", sym, be_stop)
            elif pos.direction == Direction.SHORT and be_stop < pos.current_stop:
                new_stop = be_stop
                pos.be_triggered = True
                logger.info("%s BE triggered → stop %.4f", sym, be_stop)

        # --- Add-on A: event-driven at MFE crossing 1.5R (M5/M6/M7) ---
        # C2: check both addon_a_done (fill confirmed) and addon_a_pending_id (submitted, awaiting fill)
        if pos.be_triggered and not pos.addon_a_done and not pos.addon_a_pending_id and pos.mfe >= BE_TRIGGER_R:
            if signals.addon_a_eligible(pos, h, d, current_r=cur_r):
                await self._submit_addon_a(sym, pos, h, d, cfg)

        # --- Chandelier trailing (regime-adaptive multiplier) ---
        if pos.be_triggered and pos.mfe >= CHANDELIER_TRIGGER_R:
            effective_chand_mult = cfg.chand_mult
            if d.regime == Regime.STRONG_TREND:
                effective_chand_mult *= 1.15  # wider trail, let winners run
            elif d.regime == Regime.TREND:
                effective_chand_mult *= 0.85  # tighter trail, protect gains
            chandelier = stops.compute_chandelier_stop(
                pos.direction, d, effective_chand_mult, cfg.tick_size,
            )
            if pos.direction == Direction.LONG and chandelier > new_stop:
                new_stop = chandelier
            elif pos.direction == Direction.SHORT and chandelier < new_stop:
                new_stop = chandelier

        # --- Profit floor (spec Section 10.4) ---
        if risk_per_unit > 0:
            new_stop = stops.apply_profit_floor(
                pos.direction, base.entry_price, risk_per_unit,
                pos.mfe, new_stop, cfg.tick_size,
            )

        # --- Submit stop update if changed ---
        if new_stop != pos.current_stop:
            old_stop = pos.current_stop
            pos.current_stop = new_stop
            await self._update_stop(sym, new_stop)
            if self._kit:
                self._kit.log_stop_adjustment(
                    trade_id=pos.trade_id or f"ATRSS-{sym}",
                    symbol=sym, old_stop=old_stop, new_stop=new_stop,
                    adjustment_type="trailing", trigger="atr_chandelier",
                )

        # --- Bias flip exit + stop-and-reverse (spec Section 7.4) ---
        prev_dir = self._prev_trend_dirs.get(sym, Direction.FLAT)
        if d.trend_dir != Direction.FLAT and d.trend_dir != pos.direction and d.trend_dir != prev_dir:
            logger.info(
                "%s bias flip detected: %s → %s, flattening",
                sym, pos.direction.name, d.trend_dir.name,
            )
            # Generate reverse candidate BEFORE flatten (same-hour per spec)
            # Reverse fires based only on reverse_entry_ok() — no extra signal requirement (spec S9)
            if signals.reverse_entry_ok(h, d):
                rev_cand = self._build_candidate(
                    sym, CandidateType.REVERSE, d.trend_dir, h, d, cfg,
                )
                if rev_cand:
                    self._pending_reverses.append(rev_cand)
            await self._flatten_position(sym, "FLATTEN_BIAS_FLIP")

    # ------------------------------------------------------------------
    # Halt / Limit / Reopen (spec Section 12)
    # ------------------------------------------------------------------

    async def _mark_halted(self, sym: str, now: datetime) -> None:
        """Transition symbol to halted state (spec Section 12.1/12.2)."""
        halt = self.halt_states.get(sym)
        if halt and halt.is_halted:
            return  # already halted
        halt = HaltState(
            is_halted=True,
            halt_detected_at=now,
            pre_halt_order_ids=[
                oid for oid, m in self.pending_orders.items()
                if m.get("symbol") == sym
            ],
        )
        self.halt_states[sym] = halt
        logger.warning("%s HALT detected at %s — suppressing new entries", sym, now)
        halt_audit_logger.info(
            "HALT_DETECTED symbol=%s time=%s pre_halt_orders=%s",
            sym, now.isoformat(), halt.pre_halt_order_ids,
        )

    async def _on_reopen(self, sym: str, now: datetime) -> None:
        """Handle transition from halted → tradable (spec Section 12.3/12.4)."""
        halt = self.halt_states.get(sym)
        if halt is None:
            return

        logger.info("%s REOPEN detected at %s", sym, now)
        halt_audit_logger.info(
            "REOPEN symbol=%s time=%s halt_start=%s queued_stops=%d was_unprotected=%s",
            sym, now.isoformat(),
            halt.halt_detected_at.isoformat() if halt.halt_detected_at else "unknown",
            len(halt.queued_stop_updates),
            halt.unprotected,
        )

        # 1. Replay queued stop updates
        for queued_sym, queued_stop in halt.queued_stop_updates:
            try:
                await self._update_stop(queued_sym, queued_stop)
            except Exception:
                logger.exception("Error replaying queued stop for %s", queued_sym)

        # 2. Gap-through check: if open price beyond stop, flatten immediately (spec S14.3)
        pos = self.positions.get(sym)
        hourly = self.hourly_states.get(sym)
        if pos and pos.direction != Direction.FLAT and hourly:
            gap_through = False
            if pos.direction == Direction.LONG and hourly.open <= pos.current_stop:
                gap_through = True
            elif pos.direction == Direction.SHORT and hourly.open >= pos.current_stop:
                gap_through = True
            if gap_through:
                logger.warning(
                    "%s gap-through on reopen: open=%.4f stop=%.4f, flattening",
                    sym, hourly.open, pos.current_stop,
                )
                halt_audit_logger.info(
                    "GAP_THROUGH symbol=%s open=%.4f stop=%.4f direction=%s",
                    sym, hourly.open, pos.current_stop, pos.direction.name,
                )
                await self._flatten_position(sym, "FLATTEN_GAP_THROUGH")

        # 3. Cancel stale pre-halt entry orders (>1h old or pre-halt)
        for oid in list(halt.pre_halt_order_ids):
            if oid in self.pending_orders:
                try:
                    intent = Intent(
                        intent_type=IntentType.CANCEL_ORDER,
                        strategy_id=STRATEGY_ID,
                        target_oms_order_id=oid,
                    )
                    await self._oms.submit_intent(intent)
                    self.pending_orders.pop(oid, None)
                    logger.info("Cancelled pre-halt order %s on reopen", oid)
                except Exception as e:
                    logger.warning("Error cancelling pre-halt order %s: %s", oid, e)

        # Also cancel any orders >1h old
        for oid, meta in list(self.pending_orders.items()):
            if meta.get("symbol") != sym:
                continue
            submitted = meta.get("submitted_at")
            if submitted and (now - submitted) > timedelta(hours=1):
                try:
                    intent = Intent(
                        intent_type=IntentType.CANCEL_ORDER,
                        strategy_id=STRATEGY_ID,
                        target_oms_order_id=oid,
                    )
                    await self._oms.submit_intent(intent)
                    self.pending_orders.pop(oid, None)
                    logger.info("Cancelled stale order %s (>1h) on reopen", oid)
                except Exception as e:
                    logger.warning("Error cancelling stale order %s: %s", oid, e)

        # Record reopen time for one-bar delay (spec Section 10.3)
        self._reopen_at[sym] = now
        # Clear halt state
        self.halt_states[sym] = HaltState()

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    async def _submit_entry(self, candidate: Candidate) -> None:
        """Build OMS intent for an entry or add-on order (stop-limit)."""
        if self._risk_halted:
            logger.warning(
                "ATRSS entry suppressed while OMS risk halt is active: %s",
                self._risk_halt_reason or "unspecified",
            )
            return
        inst = self._instruments.get(candidate.symbol)
        if inst is None:
            return

        side = OrderSide.BUY if candidate.direction == Direction.LONG else OrderSide.SELL

        # Compute limit band per spec Section 6
        cfg = self._config.get(candidate.symbol)
        tick = candidate.tick_size or inst.tick_size
        limit_ticks = cfg.limit_ticks if cfg else 2
        limit_pct = cfg.limit_pct if cfg else 0.0010
        limit_band = max(limit_ticks * tick, limit_pct * candidate.trigger_price)
        if candidate.direction == Direction.LONG:
            limit_price = candidate.trigger_price + limit_band
        else:
            limit_price = candidate.trigger_price - limit_band

        entry_request = ATRSSEntryRequest(
            client_order_id=f"{candidate.symbol}-entry-{int(datetime.now(timezone.utc).timestamp())}",
            symbol=candidate.symbol,
            candidate=candidate,
            limit_price=limit_price,
        )
        actions, _events = self._apply_core_bar_transition(
            bar_ts=self._last_bar_ts,
            entry_request=entry_request,
        )
        submit_action = next(
            (
                action
                for action in actions
                if isinstance(action, (SubmitEntry, SubmitAddOnEntry))
            ),
            None,
        )
        if submit_action is None:
            return

        signal_context = self._candidate_signal_context(candidate)
        risk_ctx = RiskContext(
            stop_for_risk=candidate.initial_stop,
            planned_entry_price=candidate.trigger_price,
            risk_dollars=RiskCalculator.compute_order_risk_dollars(
                candidate.trigger_price,
                candidate.initial_stop,
                candidate.qty,
                inst.point_value,
            ),
            **signal_context,
        )

        order = OMSOrder(
            strategy_id=STRATEGY_ID,
            instrument=inst,
            side=side,
            qty=submit_action.qty,
            order_type=OrderType.STOP_LIMIT,
            stop_price=submit_action.stop_price or candidate.trigger_price,
            limit_price=submit_action.limit_price or limit_price,
            tif="GTC",
            role=OrderRole.ENTRY,
            entry_policy=EntryPolicy(ttl_seconds=ORDER_EXPIRY_HOURS * 3600),
            risk_context=risk_ctx,
        )

        intent = Intent(
            intent_type=IntentType.NEW_ORDER,
            strategy_id=STRATEGY_ID,
            order=order,
        )

        receipt = await self._oms.submit_intent(intent)
        if receipt.oms_order_id:
            self._record_decision("ENTRY_SUBMITTED", {
                "symbol": candidate.symbol,
                "type": candidate.type.value,
                "qty": submit_action.qty,
                "oms_order_id": receipt.oms_order_id,
            })
            daily = self.daily_states.get(candidate.symbol)
            hourly = self.hourly_states.get(candidate.symbol)
            cfg_snap = self._config.get(candidate.symbol)
            self.pending_orders[receipt.oms_order_id] = {
                "symbol": candidate.symbol,
                "type": candidate.type,
                "direction": candidate.direction,
                "trigger_price": candidate.trigger_price,
                "initial_stop": candidate.initial_stop,
                "qty": submit_action.qty,
                "submitted_at": datetime.now(timezone.utc),
                # Carry-forward for enriched telemetry
                "quality_score": candidate.rank_score,
                "daily_regime": daily.regime.value if daily else "unknown",
                "daily_adx": daily.adx if daily else 0,
                "daily_trend_dir": daily.trend_dir.name if daily else "FLAT",
                "daily_ema_sep_pct": daily.ema_sep_pct if daily else 0,
                "daily_plus_di": daily.plus_di if daily else 0,
                "daily_minus_di": daily.minus_di if daily else 0,
                "daily_mult": cfg_snap.daily_mult if cfg_snap else 0,
                "hourly_mult": cfg_snap.hourly_mult if cfg_snap else 0,
                "chand_mult": cfg_snap.chand_mult if cfg_snap else 0,
                "base_risk_pct": cfg_snap.base_risk_pct if cfg_snap else 0.01,
                "adx_on": cfg_snap.adx_on if cfg_snap else 20,
                "adx_off": cfg_snap.adx_off if cfg_snap else 18,
                "hourly_close": hourly.close if hourly else 0,
                "hourly_ema_mom": hourly.ema_mom if hourly else 0,
                "hourly_atrh": hourly.atrh if hourly else 0,
            }
            # Consume voucher if this is a same-direction re-entry (spec Section 4.2)
            reentry = self.reentry_states.get(candidate.symbol)
            if reentry and candidate.type not in (CandidateType.ADDON_A, CandidateType.REVERSE):
                daily = self.daily_states.get(candidate.symbol)
                td = daily.trend_dir if daily else Direction.FLAT
                if signals._has_valid_voucher(reentry, candidate.direction, datetime.now(timezone.utc), td):
                    signals.consume_voucher(reentry, candidate.direction)
                    logger.info("%s voucher consumed for %s re-entry",
                                candidate.symbol, candidate.direction.name)

            logger.info(
                "Submitted %s %s %s qty=%d trigger=%.4f stop=%.4f → %s",
                candidate.symbol, candidate.type.value,
                "LONG" if candidate.direction == Direction.LONG else "SHORT",
                submit_action.qty, candidate.trigger_price, candidate.initial_stop,
                receipt.oms_order_id,
            )

            if self._kit:
                self._kit.on_order_event(
                    order_id=receipt.oms_order_id,
                    pair=candidate.symbol,
                    side="LONG" if candidate.direction == Direction.LONG else "SHORT",
                    order_type="STOP_LIMIT",
                    status="SUBMITTED",
                    requested_qty=float(submit_action.qty),
                    requested_price=candidate.trigger_price,
                    strategy_id=STRATEGY_ID,
                )
        else:
            self._record_decision("ENTRY_DENIED", {
                "symbol": candidate.symbol,
                "type": candidate.type.value,
                "reason": "oms_rejected",
            })

    async def _submit_addon_a(
        self,
        sym: str,
        pos: PositionBook,
        h: HourlyState,
        d: DailyState,
        cfg: SymbolConfig,
    ) -> None:
        """Event-driven Add-on A: market order with IBKR sequencing (M5/M6/M7/M8).

        Sequencing per spec Section 8.1:
        1. Update stop to BE+ and wait for ack
        2. Submit Add-on A as market order
        3. Update stop qty (handled in _on_fill)
        """
        if self._risk_halted:
            logger.warning(
                "%s Add-on A suppressed while OMS risk halt is active: %s",
                sym, self._risk_halt_reason or "unspecified",
            )
            return
        inst = self._instruments.get(sym)
        if inst is None:
            return
        base = pos.base_leg
        if base is None:
            return

        # Compute qty: ceil(0.5 * base_qty), minimum 1 with risk cap guard (M8)
        # spec Section 11.1: ceil() not int()
        desired = math.ceil(base.qty * ADDON_A_SIZE_MULT)
        if desired <= 0:
            # Allow qty=1 only if risk is capped
            addon_risk = abs(h.close - pos.current_stop) * inst.point_value * 1
            base_risk = abs(base.entry_price - base.initial_stop) * inst.point_value * base.qty
            max_risk = min(0.25 * base_risk, 0.0015 * self._equity)
            if base.qty >= 1 and addon_risk <= max_risk:
                desired = 1
            else:
                logger.debug("%s Add-on A skipped: risk cap exceeded", sym)
                return
        qty = desired

        # Step 1: Ensure stop is at BE+ and wait for OMS acknowledgment
        # Per spec Section 8.1: must confirm stop replacement before market add
        if pos.stop_oms_order_id:
            await self._update_stop(sym, pos.current_stop)
            # Brief pause to allow IB to process the stop replacement
            await asyncio.sleep(0.5)

        # Step 2: Submit Add-on A as market order
        add_on_request = ATRSSAddOnARequest(
            client_order_id=f"{sym}-addon-a-{int(datetime.now(timezone.utc).timestamp())}",
            symbol=sym,
            direction=pos.direction,
            qty=qty,
            entry_price=h.close,
            stop_price=pos.current_stop,
        )
        actions, _events = self._apply_core_bar_transition(
            bar_ts=self._last_bar_ts,
            add_on_a_request=add_on_request,
        )
        submit_action = next((action for action in actions if isinstance(action, SubmitAddOnEntry)), None)
        if submit_action is None:
            return

        side = OrderSide.BUY if pos.direction == Direction.LONG else OrderSide.SELL
        signal_context = self._addon_signal_context(sym=sym, pos=pos, hourly=h)
        risk_ctx = RiskContext(
            stop_for_risk=pos.current_stop,
            planned_entry_price=h.close,
            risk_dollars=RiskCalculator.compute_order_risk_dollars(
                h.close, pos.current_stop, qty, inst.point_value,
            ),
            **signal_context,
        )
        order = OMSOrder(
            strategy_id=STRATEGY_ID,
            instrument=inst,
            side=side,
            qty=submit_action.qty,
            order_type=OrderType.MARKET,
            tif="GTC",
            role=OrderRole.ENTRY,
            risk_context=risk_ctx,
        )
        intent = Intent(
            intent_type=IntentType.NEW_ORDER,
            strategy_id=STRATEGY_ID,
            order=order,
        )
        receipt = await self._oms.submit_intent(intent)
        if receipt.oms_order_id:
            self.pending_orders[receipt.oms_order_id] = {
                "symbol": sym,
                "type": CandidateType.ADDON_A,
                "direction": pos.direction,
                "trigger_price": h.close,
                "initial_stop": pos.current_stop,
                "qty": submit_action.qty,
                "submitted_at": datetime.now(timezone.utc),
                "signal_id": signal_context["signal_id"],
                "bar_id": signal_context["bar_id"],
                "exchange_timestamp": signal_context["exchange_timestamp"].isoformat(),
            }
            # C2: Track pending order ID to prevent re-triggering during async window.
            # Only set addon_a_done on fill, not submission.
            pos.addon_a_pending_id = receipt.oms_order_id
            logger.info(
                "Submitted Add-on A MARKET %s %s qty=%d → %s",
                sym, "LONG" if pos.direction == Direction.LONG else "SHORT",
                qty, receipt.oms_order_id,
            )

    def _candidate_signal_context(self, candidate: Candidate) -> dict[str, Any]:
        ts = (
            candidate.time
            or getattr(candidate.signal_bar, "time", None)
            or self._symbol_last_bar_ts.get(candidate.symbol)
            or self._last_bar_ts
            or datetime.now(timezone.utc)
        )
        ts_text = ts.isoformat()
        return {
            "signal_id": (
                f"{candidate.symbol}:{candidate.type.value}:"
                f"{candidate.direction.name}:{ts_text}"
            ),
            "bar_id": f"{candidate.symbol}:1h:{ts_text}",
            "exchange_timestamp": ts,
        }

    def _addon_signal_context(
        self,
        *,
        sym: str,
        pos: PositionBook,
        hourly: HourlyState,
    ) -> dict[str, Any]:
        ts = (
            hourly.time
            or self._symbol_last_bar_ts.get(sym)
            or self._last_bar_ts
            or datetime.now(timezone.utc)
        )
        ts_text = ts.isoformat()
        return {
            "signal_id": f"{sym}:{CandidateType.ADDON_A.value}:{pos.direction.name}:{ts_text}",
            "bar_id": f"{sym}:1h:{ts_text}",
            "exchange_timestamp": ts,
        }

    async def _cancel_symbol_orders(self, sym: str) -> None:
        """Cancel all pending entry orders for a symbol (M4)."""
        to_cancel = [
            oms_id for oms_id, meta in self.pending_orders.items()
            if meta.get("symbol") == sym
        ]
        for oms_id in to_cancel:
            try:
                intent = Intent(
                    intent_type=IntentType.CANCEL_ORDER,
                    strategy_id=STRATEGY_ID,
                    target_oms_order_id=oms_id,
                )
                await self._oms.submit_intent(intent)
                self.pending_orders.pop(oms_id, None)
                logger.info("Cancelled pending order %s for %s on exit", oms_id, sym)
            except Exception as e:
                logger.warning("Error cancelling order %s for %s: %s", oms_id, sym, e)

    async def _place_stop(
        self, sym: str, stop_price: float, qty: int
    ) -> str | None:
        """Place a protective stop order via OMS."""
        inst = self._instruments.get(sym)
        if inst is None:
            return None

        pos = self.positions.get(sym)
        if pos is None:
            return None

        side = OrderSide.SELL if pos.direction == Direction.LONG else OrderSide.BUY

        order = OMSOrder(
            strategy_id=STRATEGY_ID,
            instrument=inst,
            side=side,
            qty=qty,
            order_type=OrderType.STOP,
            stop_price=stop_price,
            tif="GTC",
            role=OrderRole.STOP,
        )

        intent = Intent(
            intent_type=IntentType.NEW_ORDER,
            strategy_id=STRATEGY_ID,
            order=order,
        )

        receipt = await self._oms.submit_intent(intent)
        if receipt.oms_order_id:
            logger.info("Placed stop for %s at %.4f qty=%d → %s", sym, stop_price, qty, receipt.oms_order_id)
        return receipt.oms_order_id

    async def _update_stop(self, sym: str, new_stop: float) -> None:
        """Replace the current protective stop via OMS.

        If the symbol is halted and the update is rejected, queue it for
        replay on reopen (spec Section 12.2).
        """
        pos = self.positions.get(sym)
        if pos is None or not pos.stop_oms_order_id:
            return

        stop_request = ATRSSStopUpdateRequest(
            symbol=sym,
            stop_price=new_stop,
            qty=pos.total_qty,
            reason="trail",
        )
        actions, _events = self._apply_core_bar_transition(
            bar_ts=self._last_bar_ts,
            stop_update=stop_request,
        )
        pos = self.positions.get(sym)
        replace_action = next((action for action in actions if isinstance(action, ReplaceProtectiveStop)), None)
        if pos is None or replace_action is None:
            return
        new_stop = replace_action.stop_price

        intent = Intent(
            intent_type=IntentType.REPLACE_ORDER,
            strategy_id=STRATEGY_ID,
            target_oms_order_id=pos.stop_oms_order_id,
            new_stop_price=new_stop,
            new_qty=replace_action.qty,
        )

        try:
            receipt = await self._oms.submit_intent(intent)
            logger.debug("Updated stop for %s to %.4f → %s", sym, new_stop, receipt.result)
        except Exception as e:
            halt = self.halt_states.get(sym)
            if halt and halt.is_halted:
                halt.queued_stop_updates.append((sym, new_stop))
                halt.unprotected = True
                logger.warning("UNPROTECTED %s: stop rejected during halt, queued %.4f", sym, new_stop)
                halt_audit_logger.info(
                    "UNPROTECTED symbol=%s stop=%.4f reason=stop_rejected_during_halt",
                    sym, new_stop,
                )
            else:
                raise

    async def _partial_close_position(self, sym: str, frac: float, reason: str) -> None:
        """Close a fraction of the base leg via a market order.

        If only 1 contract remains, falls back to full flatten.
        """
        pos = self.positions.get(sym)
        if pos is None or pos.direction == Direction.FLAT:
            return
        base = pos.base_leg
        if base is None or base.qty < 1:
            return
        if base.qty == 1:
            await self._flatten_position(sym, reason)
            return
        partial_qty = max(1, int(base.qty * frac))
        if partial_qty >= base.qty:
            partial_qty = base.qty - 1
        inst = self._instruments.get(sym)
        if inst is None:
            return
        partial_request = ATRSSPartialExitRequest(
            client_order_id=f"{sym}-partial-{int(datetime.now(timezone.utc).timestamp())}",
            symbol=sym,
            qty=partial_qty,
            reason=reason,
        )
        actions, _events = self._apply_core_bar_transition(
            bar_ts=self._last_bar_ts,
            partial_exit_request=partial_request,
        )
        pos = self.positions.get(sym)
        partial_action = next((action for action in actions if isinstance(action, SubmitPartialExit)), None)
        if pos is None or partial_action is None:
            return
        side = OrderSide.SELL if pos.direction == Direction.LONG else OrderSide.BUY
        order = OMSOrder(
            strategy_id=STRATEGY_ID,
            instrument=inst,
            side=side,
            qty=partial_action.qty,
            order_type=OrderType.MARKET,
            tif="GTC",
            role=OrderRole.ENTRY,
        )
        intent = Intent(
            intent_type=IntentType.NEW_ORDER,
            strategy_id=STRATEGY_ID,
            order=order,
        )
        receipt = await self._oms.submit_intent(intent)
        if receipt.oms_order_id:
            self.pending_orders[receipt.oms_order_id] = {
                "symbol": sym,
                "type": "PARTIAL_CLOSE",
                "direction": pos.direction,
                "partial_qty": partial_qty,
                "reason": reason,
                "submitted_at": datetime.now(timezone.utc),
            }
            logger.info(
                "Submitted PARTIAL_CLOSE %s %s qty=%d reason=%s → %s",
                sym, pos.direction.name, partial_qty, reason,
                receipt.oms_order_id,
            )

    async def _flatten_position(self, sym: str, reason: str = "FLATTEN") -> None:
        """Flatten all positions for a symbol via OMS FLATTEN intent.

        Also cancels all pending orders for the symbol (M4, spec Section 12.2).
        """
        inst = self._instruments.get(sym)
        if inst is None:
            return

        flatten_request = ATRSSFlattenRequest(symbol=sym, reason=reason)
        actions, _events = self._apply_core_bar_transition(
            bar_ts=self._last_bar_ts,
            flatten_request=flatten_request,
        )
        flatten_action = next((action for action in actions if isinstance(action, FlattenPosition)), None)
        if flatten_action is None:
            return
        reason = flatten_action.reason

        # Cancel all pending entry orders for this symbol
        await self._cancel_symbol_orders(sym)

        # Cancel protective stop order explicitly (M4)
        pos = self.positions.get(sym)
        if pos and pos.stop_oms_order_id:
            try:
                await self._oms.submit_intent(Intent(
                    intent_type=IntentType.CANCEL_ORDER,
                    strategy_id=STRATEGY_ID,
                    target_oms_order_id=pos.stop_oms_order_id,
                ))
            except Exception as e:
                logger.warning("Error cancelling protective stop for %s: %s", sym, e)

        intent = Intent(
            intent_type=IntentType.FLATTEN,
            strategy_id=STRATEGY_ID,
            instrument_symbol=inst.symbol,
        )
        receipt = await self._oms.submit_intent(intent)
        if receipt.oms_order_id:
            self._pending_flattens[sym] = {
                "oms_order_id": receipt.oms_order_id,
                "reason": reason,
            }
        logger.info("Flatten %s reason=%s → %s", sym, reason, receipt.result)

    async def _cancel_expired_orders(self, sym: str, now: datetime) -> None:
        """Cancel pending entry orders older than ORDER_EXPIRY_HOURS."""
        expired = []
        for oms_id, meta in list(self.pending_orders.items()):
            if meta.get("symbol") != sym:
                continue
            submitted = meta.get("submitted_at")
            if submitted and (now - submitted) > timedelta(hours=ORDER_EXPIRY_HOURS):
                expired.append(oms_id)

        for oms_id in expired:
            try:
                intent = Intent(
                    intent_type=IntentType.CANCEL_ORDER,
                    strategy_id=STRATEGY_ID,
                    target_oms_order_id=oms_id,
                )
                await self._oms.submit_intent(intent)
                self.pending_orders.pop(oms_id, None)
                logger.info("Expired order cancelled: %s", oms_id)
            except Exception as e:
                logger.warning("Error cancelling expired order %s: %s", oms_id, e)

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    async def _process_events(self) -> None:
        """Listen on OMS event queue and route to handlers."""
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
                logger.exception("Error handling event %s", event)

    async def _handle_event(self, event: Any) -> None:
        """Route an OMS event to the appropriate handler."""
        etype = event.event_type
        oms_id = event.oms_order_id

        if etype == OMSEventType.FILL:
            await self._on_fill(oms_id, event.payload or {})
        elif etype == OMSEventType.RISK_HALT:
            await self._on_risk_halt((event.payload or {}).get("reason", ""))

        elif etype == OMSEventType.ORDER_REJECTED:
            # Check if rejection indicates halt/limit state (spec Section 10.1)
            reason = str((event.payload or {}).get("reason", "")).lower()
            if any(kw in reason for kw in ("halt", "limit", "auction", "frozen")):
                meta = self.pending_orders.get(oms_id, {})
                rej_sym = meta.get("symbol")
                if rej_sym:
                    await self._mark_halted(rej_sym, datetime.now(timezone.utc))
            await self._on_terminal(oms_id, etype)

        elif etype in (
            OMSEventType.ORDER_CANCELLED,
            OMSEventType.ORDER_EXPIRED,
        ):
            await self._on_terminal(oms_id, etype)

    async def _on_risk_halt(self, reason: str) -> None:
        """Pause new entries and cancel outstanding entry intents."""
        if self._risk_halted:
            return

        self._risk_halted = True
        self._risk_halt_reason = reason or "OMS risk halt"
        logger.error("ATRSS risk halt engaged: %s", self._risk_halt_reason)

        for oms_id, meta in list(self.pending_orders.items()):
            if meta.get("type") == "PARTIAL_CLOSE":
                continue
            try:
                await self._oms.submit_intent(
                    Intent(
                        intent_type=IntentType.CANCEL_ORDER,
                        strategy_id=STRATEGY_ID,
                        target_oms_order_id=oms_id,
                    )
                )
            except Exception:
                logger.warning("Failed to cancel ATRSS pending order %s during risk halt", oms_id)

    async def _on_fill(self, oms_order_id: str | None, payload: dict) -> None:
        """Handle a fill event -- route through core, dispatch actions, record/instrument."""
        if not oms_order_id:
            return

        # 1. Capture pre-fill context (core will pop pending_orders / positions)
        meta = self.pending_orders.get(oms_order_id)
        pre_positions = {s: deepcopy(p) for s, p in self.positions.items()}

        # Resolve fill details
        fill_price = payload.get("price", 0.0)
        fill_qty = int(payload.get("qty", 0))
        fill_time = datetime.now(timezone.utc)
        symbol = ""
        if meta:
            symbol = meta["symbol"]
            fill_price = fill_price or meta.get("trigger_price", 0.0)
            fill_qty = fill_qty or int(meta.get("qty", 0))

        # Detect bad-fill BEFORE core routing (need meta + config)
        is_bad_fill = False
        if meta and meta.get("type") not in ("PARTIAL_CLOSE",):
            cfg = self._config.get(symbol)
            if cfg:
                hourly = self.hourly_states.get(symbol)
                atrh = hourly.atrh if hourly else 0.0
                max_slip_pct = cfg.max_entry_slip_pct * meta.get("trigger_price", 0)
                max_slip_atr = MAX_ENTRY_SLIP_ATR * atrh
                max_slip = min(max_slip_pct, max_slip_atr)
                actual_slip = abs(fill_price - meta.get("trigger_price", fill_price))
                if max_slip > 0 and actual_slip > max_slip:
                    logger.warning(
                        "%s BAD FILL: slip=%.4f > max=%.4f (trigger=%.4f fill=%.4f), "
                        "will flatten after recording",
                        symbol, actual_slip, max_slip, meta["trigger_price"], fill_price,
                    )
                    is_bad_fill = True

        # 2. Route through core for state transitions
        fill = ATRSSFill(
            oms_order_id=oms_order_id,
            fill_price=fill_price,
            fill_qty=fill_qty,
            symbol=symbol,
            fill_time=fill_time,
            commission=payload.get("commission", 0.0),
            fill_id=str(payload.get("fill_id") or payload.get("exec_id") or ""),
            intent_id=str(payload.get("intent_id") or ""),
            risk_decision_ref=str(payload.get("risk_decision_ref") or ""),
            portfolio_decision_ref=str(payload.get("portfolio_decision_ref") or ""),
            runtime_payload={**payload, "oms_order_id": oms_order_id},
        )
        core_state = build_core_runtime_state(self)
        new_state, actions, events = atrss_core_logic.on_fill(core_state, fill)
        apply_core_runtime_state(self, new_state)

        # 3. Dispatch OMS actions from core
        for action in actions:
            if isinstance(action, SubmitProtectiveStop):
                stop_id = await self._place_stop(action.symbol, action.stop_price, action.qty)
                pos = self.positions.get(action.symbol)
                if pos:
                    pos.stop_oms_order_id = stop_id or ""
                    pos.stop_pending = False
            elif isinstance(action, ReplaceProtectiveStop):
                await self._update_stop_qty(action.symbol)
            elif isinstance(action, FlattenPosition):
                await self._flatten_position(action.symbol, action.reason)

        # 4. Instrumentation per event
        for event in events:
            self._record_decision(event.code, event.details)
            ev_sym = event.details.get("symbol", symbol) if event.details else symbol

            if event.code == "ENTRY_FILLED" and meta:
                await self._record_entry_instrumentation(
                    ev_sym, meta, fill_price, fill_qty, fill_time, oms_order_id, payload,
                )
            elif event.code == "ADD_ON_FILLED":
                logger.info("ADD_ON FILL %s @ %.4f", ev_sym, fill_price)
            elif event.code == "ORPHANED_ADDON_FILLED":
                logger.warning("ORPHANED ADDON %s @ %.4f -- flattening", ev_sym, fill_price)
            elif event.code in ("STOP_FILLED", "EXIT_FILLED"):
                pre_pos = pre_positions.get(ev_sym)
                if pre_pos:
                    exit_reason = event.details.get("reason", "STOP") if event.details else "STOP"
                    await self._record_exit_instrumentation(
                        ev_sym, pre_pos, fill_price, fill_time, exit_reason, oms_order_id, payload,
                    )
                    # Cancel all pending orders for exited symbol (M4)
                    await self._cancel_symbol_orders(ev_sym)
            elif event.code == "PARTIAL_EXIT_FILLED" and meta:
                pre_pos = pre_positions.get(ev_sym)
                await self._record_partial_instrumentation(
                    ev_sym, meta, pre_pos, fill_price, fill_time,
                )

        # 5. Bad-fill flatten AFTER instrumentation
        if is_bad_fill and symbol:
            await self._flatten_position(symbol, "FLATTEN_BAD_FILL")

    async def _record_entry_instrumentation(
        self, sym: str, meta: dict,
        fill_price: float, fill_qty: int, fill_time: datetime, oms_order_id: str, payload: dict,
    ) -> None:
        """Record trade entry + kit logging for a filled entry."""
        direction = meta["direction"]
        ctype = meta["type"]
        cfg = self._config.get(sym)
        hourly = self.hourly_states.get(sym)
        atrh = hourly.atrh if hourly else 0.0

        # Record trade entry via recorder
        trade_id = ""
        if self._recorder:
            try:
                trade_id = await self._recorder.record_entry(
                    strategy_id=STRATEGY_ID,
                    instrument=sym,
                    direction="LONG" if direction == Direction.LONG else "SHORT",
                    quantity=fill_qty,
                    entry_price=Decimal(str(fill_price)),
                    entry_ts=fill_time,
                    setup_tag=ctype.value if hasattr(ctype, 'value') else str(ctype),
                    entry_type=ctype.value if hasattr(ctype, 'value') else str(ctype),
                )
            except Exception:
                logger.exception("Error recording entry for %s", sym)

        # Patch trade_id onto the leg that core created
        pos = self.positions.get(sym)
        if pos and trade_id:
            for leg in pos.legs:
                if leg.oms_order_id == oms_order_id and not leg.trade_id:
                    leg.trade_id = trade_id
                    break

        if self._kit and cfg:
            side_str = "LONG" if direction == Direction.LONG else "SHORT"
            active_pos = {s: p for s, p in self.positions.items() if p.direction != Direction.FLAT}
            quality_score = meta.get("quality_score", 0)
            quality_margin = (quality_score - QUALITY_GATE_THRESHOLD) / max(QUALITY_GATE_THRESHOLD, 0.01) * 100 if QUALITY_GATE_THRESHOLD > 0 else 0
            _cfg_dict = dataclasses.asdict(cfg)
            _param_set_id = hashlib.md5(
                json.dumps(_cfg_dict, sort_keys=True, default=str).encode()
            ).hexdigest()[:8]
            _ctype_val = ctype.value if hasattr(ctype, 'value') else str(ctype)
            self._kit.log_entry(
                trade_id=trade_id or f"{sym}_{fill_time.isoformat()}",
                pair=sym,
                side=side_str,
                entry_price=fill_price,
                position_size=float(fill_qty),
                position_size_quote=fill_price * fill_qty,
                entry_signal=_ctype_val,
                entry_signal_id=f"{sym}_{_ctype_val}_{fill_time.isoformat()}",
                entry_signal_strength=meta.get("quality_score", 0.5),
                active_filters=["quality_gate", "momentum", "portfolio_heat"],
                passed_filters=["quality_gate", "momentum", "portfolio_heat"],
                filter_decisions=[
                    {"filter_name": "quality_gate", "threshold": QUALITY_GATE_THRESHOLD,
                     "actual_value": quality_score, "passed": True,
                     "margin_pct": round(quality_margin, 1)},
                    {"filter_name": "momentum", "threshold": 0,
                     "actual_value": meta.get("hourly_ema_mom", 0), "passed": True,
                     "margin_pct": 0},
                ],
                strategy_params={
                    "param_set_id": _param_set_id,
                    "config": _cfg_dict,
                    "atrh": atrh,
                    "initial_stop": meta["initial_stop"],
                    "daily_mult": meta.get("daily_mult"),
                    "hourly_mult": meta.get("hourly_mult"),
                    "chand_mult": meta.get("chand_mult"),
                    "base_risk_pct": meta.get("base_risk_pct"),
                    "quality_gate_threshold": QUALITY_GATE_THRESHOLD,
                    "adx_on": meta.get("adx_on"),
                    "adx_off": meta.get("adx_off"),
                    "regime": meta.get("daily_regime"),
                },
                expected_entry_price=meta["trigger_price"],
                signal_factors=[
                    {"factor_name": "quality_score", "factor_value": meta.get("quality_score", 0),
                     "threshold": QUALITY_GATE_THRESHOLD, "contribution": "entry_quality"},
                    {"factor_name": "adx", "factor_value": meta.get("daily_adx", 0),
                     "threshold": meta.get("adx_on", 20), "contribution": "trend_strength"},
                    {"factor_name": "ema_separation", "factor_value": meta.get("daily_ema_sep_pct", 0),
                     "threshold": 0.15, "contribution": "trend_separation"},
                    {"factor_name": "regime", "factor_value": meta.get("daily_regime", "unknown"),
                     "threshold": "TREND", "contribution": "regime_context"},
                    {"factor_name": "signal_type", "factor_value": _ctype_val,
                     "threshold": "pullback", "contribution": "entry_type"},
                ],
                sizing_inputs={
                    "target_risk_pct": cfg.base_risk_pct,
                    "account_equity": self._equity,
                    "volatility_basis": atrh,
                    "sizing_model": "atr_risk",
                },
                portfolio_state_at_entry={
                    "num_positions": len(active_pos),
                    "long_positions": sum(1 for p in active_pos.values() if p.direction == Direction.LONG),
                    "short_positions": sum(1 for p in active_pos.values() if p.direction == Direction.SHORT),
                    "symbols_held": list(active_pos.keys()),
                },
                signal_evolution=self._build_signal_evolution(sym),
                concurrent_positions_strategy=len(self.positions),
                fill_time_ms=int(fill_time.timestamp() * 1000),
                **fill_runtime_refs(oms_order_id, payload, fill_qty=float(fill_qty)),
            )

            self._kit.on_order_event(
                order_id=oms_order_id,
                pair=sym,
                side="LONG" if direction == Direction.LONG else "SHORT",
                order_type="STOP_LIMIT",
                status="FILLED",
                requested_qty=float(meta["qty"]),
                filled_qty=float(fill_qty),
                requested_price=meta["trigger_price"],
                fill_price=fill_price,
                related_trade_id=trade_id,
                strategy_id=STRATEGY_ID,
            )

        logger.info(
            "FILL %s %s %s %d @ %.4f (stop=%.4f)",
            sym, ctype.value if hasattr(ctype, 'value') else str(ctype),
            "LONG" if direction == Direction.LONG else "SHORT",
            fill_qty, fill_price, meta["initial_stop"],
        )

    async def _record_exit_instrumentation(
        self, sym: str, pre_pos: Any,
        fill_price: float, fill_time: datetime, reason: str, oms_order_id: str, payload: dict,
    ) -> None:
        """Record trade exits + kit logging for a stop or flatten fill."""
        # Record exits for all legs
        if self._recorder:
            for leg in pre_pos.legs:
                if leg.trade_id:
                    try:
                        risk_per_unit = pre_pos.base_risk_per_unit
                        if pre_pos.direction == Direction.LONG:
                            pnl = fill_price - leg.entry_price
                        else:
                            pnl = leg.entry_price - fill_price
                        realized_r = pnl / risk_per_unit if risk_per_unit > 0 else 0

                        inst = self._instruments.get(sym)
                        pv = inst.point_value if inst else 1.0
                        realized_usd = pnl * pv * leg.qty

                        await self._recorder.record_exit(
                            trade_id=leg.trade_id,
                            exit_price=Decimal(str(fill_price)),
                            exit_ts=fill_time,
                            exit_reason=reason,
                            realized_r=Decimal(str(round(realized_r, 4))),
                            realized_usd=Decimal(str(round(realized_usd, 2))),
                            mfe_r=Decimal(str(round(pre_pos.mfe, 4))),
                            duration_bars=pre_pos.bars_held,
                        )
                    except Exception:
                        logger.exception("Error recording exit for %s leg", sym)

        # Kit instrumentation
        if self._kit:
            _entry = pre_pos.base_leg.entry_price if pre_pos.base_leg else fill_price
            if pre_pos.direction == Direction.LONG:
                _mfe_pct = (pre_pos.mfe_price - _entry) / _entry if _entry > 0 else None
                _mae_pct = (_entry - pre_pos.mae_price) / _entry if _entry > 0 else None
                _pnl_pct = (fill_price - _entry) / _entry if _entry > 0 else None
            else:
                _mfe_pct = (_entry - pre_pos.mfe_price) / _entry if _entry > 0 else None
                _mae_pct = (pre_pos.mae_price - _entry) / _entry if _entry > 0 else None
                _pnl_pct = (_entry - fill_price) / _entry if _entry > 0 else None
            for leg in pre_pos.legs:
                tid = leg.trade_id or f"{sym}_{leg.fill_time.isoformat()}"
                self._kit.log_exit(
                    trade_id=tid,
                    exit_price=fill_price,
                    exit_reason=reason,
                    expected_exit_price=pre_pos.current_stop,
                    mfe_price=pre_pos.mfe_price,
                    mae_price=pre_pos.mae_price,
                    mfe_r=pre_pos.mfe,
                    mae_r=pre_pos.mae,
                    mfe_pct=_mfe_pct,
                    mae_pct=_mae_pct,
                    pnl_pct=_pnl_pct,
                    **fill_runtime_refs(
                        oms_order_id,
                        payload,
                        fill_qty=float(pre_pos.total_qty),
                        is_exit=True,
                    ),
                )

            self._kit.on_order_event(
                order_id=oms_order_id,
                pair=sym,
                side="SELL" if pre_pos.direction == Direction.LONG else "BUY",
                order_type="STOP",
                status="FILLED",
                requested_qty=float(pre_pos.total_qty),
                filled_qty=float(pre_pos.total_qty),
                requested_price=pre_pos.current_stop,
                fill_price=fill_price,
                related_trade_id=pre_pos.base_leg.trade_id if pre_pos.base_leg else "",
                strategy_id=STRATEGY_ID,
            )

        logger.info(
            "%s %s @ %.4f after %d bars",
            "STOPPED OUT" if reason == "STOP" else "FLATTEN FILL",
            sym, fill_price, pre_pos.bars_held,
        )

    async def _record_partial_instrumentation(
        self, sym: str, meta: dict, pre_pos: Any,
        fill_price: float, fill_time: datetime,
    ) -> None:
        """Record partial exit via recorder."""
        partial_qty = meta.get("partial_qty", 0)
        reason = meta.get("reason", "PARTIAL")
        if self._recorder and pre_pos and pre_pos.base_leg and pre_pos.base_leg.trade_id:
            try:
                risk_per_unit = pre_pos.base_risk_per_unit
                if pre_pos.direction == Direction.LONG:
                    pnl = fill_price - pre_pos.base_leg.entry_price
                else:
                    pnl = pre_pos.base_leg.entry_price - fill_price
                realized_r = pnl / risk_per_unit if risk_per_unit > 0 else 0
                inst = self._instruments.get(sym)
                pv = inst.point_value if inst else 1.0
                realized_usd = pnl * pv * partial_qty
                await self._recorder.record_exit(
                    trade_id=pre_pos.base_leg.trade_id,
                    exit_price=Decimal(str(fill_price)),
                    exit_ts=fill_time,
                    exit_reason=reason,
                    realized_r=Decimal(str(round(realized_r, 4))),
                    realized_usd=Decimal(str(round(realized_usd, 2))),
                    mfe_r=Decimal(str(round(pre_pos.mfe, 4))),
                    duration_bars=pre_pos.bars_held,
                )
            except Exception:
                logger.exception("Error recording partial exit for %s", sym)
        logger.info(
            "PARTIAL_CLOSE FILL %s qty=%d reason=%s @ %.4f",
            sym, partial_qty, reason, fill_price,
        )

    async def _on_terminal(self, oms_order_id: str | None, etype: Any) -> None:
        """Clean up pending orders on cancel/reject/expire -- routed through core."""
        if oms_order_id:
            # Capture meta for instrumentation before core pops it
            meta = self.pending_orders.get(oms_order_id)

            status_map = {
                OMSEventType.ORDER_REJECTED: "rejected",
                OMSEventType.ORDER_CANCELLED: "cancelled",
                OMSEventType.ORDER_EXPIRED: "expired",
            }
            update = ATRSSOrderUpdate(
                oms_order_id=oms_order_id,
                status=status_map.get(etype, "cancelled"),
                symbol=meta.get("symbol", "") if meta else "",
                timestamp=datetime.now(timezone.utc),
                order_role="entry",
            )

            # Route through core for state cleanup
            core_state = build_core_runtime_state(self)
            new_state, _actions, events = atrss_core_logic.on_order_update(core_state, update)
            apply_core_runtime_state(self, new_state)

            for event in events:
                self._record_decision(event.code, event.details)

            # Kit instrumentation (preserved from original)
            if meta:
                sym = meta.get("symbol")
                logger.info(
                    "Order %s for %s %s: %s",
                    oms_order_id, sym, meta.get("type"), etype,
                )

                if self._kit:
                    kit_status_map = {
                        OMSEventType.ORDER_REJECTED: "REJECTED",
                        OMSEventType.ORDER_CANCELLED: "CANCELLED",
                        OMSEventType.ORDER_EXPIRED: "EXPIRED",
                    }
                    self._kit.on_order_event(
                        order_id=oms_order_id,
                        pair=meta.get("symbol", ""),
                        side="LONG" if meta.get("direction") == Direction.LONG else "SHORT",
                        order_type="STOP_LIMIT",
                        status=kit_status_map.get(etype, "CANCELLED"),
                        requested_qty=float(meta.get("qty", 0)),
                        requested_price=meta.get("trigger_price", 0),
                        strategy_id=STRATEGY_ID,
                    )
                # C2 addon_a clearing now handled by core on_order_update
                # Verify: check that core cleared addon_a_pending_id
                if meta.get("type") == CandidateType.ADDON_A and sym:
                    pos = self.positions.get(sym)
                    if pos and pos.addon_a_pending_id == oms_order_id:
                        pos.addon_a_pending_id = ""
                        logger.info(
                            "%s Add-on A %s — cleared pending, will retry next cycle",
                            sym, etype,
                        )

    async def _update_stop_qty(self, sym: str) -> None:
        """Update the stop order quantity to cover all legs."""
        pos = self.positions.get(sym)
        if pos is None or not pos.stop_oms_order_id:
            return

        total_qty = pos.total_qty
        intent = Intent(
            intent_type=IntentType.REPLACE_ORDER,
            strategy_id=STRATEGY_ID,
            target_oms_order_id=pos.stop_oms_order_id,
            new_qty=total_qty,
            new_stop_price=pos.current_stop,
        )
        receipt = await self._oms.submit_intent(intent)
        logger.debug("Updated stop qty for %s to %d → %s", sym, total_qty, receipt.result)

    # ------------------------------------------------------------------
    # Historical data loading
    # ------------------------------------------------------------------

    async def _load_initial_bars(self) -> None:
        """Load enough historical bars to seed all indicators."""
        for sym in self._config:
            try:
                cfg = self._config[sym]
                closes_d, highs_d, lows_d, last_daily_date = await self._fetch_daily_bars(
                    sym, cfg, request_kind="startup",
                )
                if closes_d is not None:
                    daily = compute_daily_state(closes_d, highs_d, lows_d, None, cfg, last_daily_date)
                    self.daily_states[sym] = daily
                    logger.info(
                        "%s daily: regime=%s trend=%s adx=%.1f score=%.0f",
                        sym, daily.regime.value, daily.trend_dir.name,
                        daily.adx, daily.score,
                    )
            except Exception:
                logger.exception("Error loading initial bars for %s", sym)

    async def _fetch_daily_bars(
        self, sym: str, cfg: SymbolConfig, request_kind: str = "recurring",
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, str | None]:
        """Fetch daily historical bars from IB.

        Returns (closes, highs, lows, last_bar_date).
        """
        try:
            contract = self._get_contract(sym)
            if contract is None:
                return None, None, None, None

            bars = await self._ib.req_historical_data(
                contract,
                endDateTime="",
                durationStr="200 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
                request_kind=request_kind,
                completed_only=True,
            )
            if not bars:
                return None, None, None, None

            remember_idle_market_bars(self, bars, symbol=sym, timeframe="1h")
            closes = np.array([b.close for b in bars], dtype=float)
            highs = np.array([b.high for b in bars], dtype=float)
            lows = np.array([b.low for b in bars], dtype=float)
            last_bar_date = str(bars[-1].date) if bars else None
            return closes, highs, lows, last_bar_date
        except Exception:
            logger.exception("Error fetching daily bars for %s", sym)
            return None, None, None, None

    async def _fetch_hourly_bars(
        self, sym: str, cfg: SymbolConfig, request_kind: str = "recurring",
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Fetch hourly historical bars from IB. Returns (closes, highs, lows, opens)."""
        try:
            contract = self._get_contract(sym)
            if contract is None:
                return None, None, None, None

            bars = await self._ib.req_historical_data(
                contract,
                endDateTime="",
                durationStr="10 D",
                barSizeSetting="1 hour",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
                request_kind=request_kind,
                completed_only=True,
            )
            if not bars:
                return None, None, None, None

            closes = np.array([b.close for b in bars], dtype=float)
            highs = np.array([b.high for b in bars], dtype=float)
            lows = np.array([b.low for b in bars], dtype=float)
            opens = np.array([b.open for b in bars], dtype=float)
            return closes, highs, lows, opens
        except Exception:
            logger.exception("Error fetching hourly bars for %s", sym)
            return None, None, None, None

    def _get_contract(self, sym: str) -> Any | None:
        """Get the IB contract for a symbol from cache or build a generic one."""
        if sym in self.contracts:
            return self.contracts[sym][0]

        # Fallback: build a minimal contract from config
        try:
            cfg = self._config[sym]
            cf = getattr(self._ib, "_contract_factory", None)
            if cf is not None:
                return cf.build_contract(
                    sym,
                    cfg.contract_expiry,
                    instrument=self._instruments.get(sym),
                )
            if cfg.sec_type == "STK":
                from ib_async import Stock
                c = Stock(
                    symbol=sym,
                    exchange=cfg.exchange,
                    currency="USD",
                )
                if cfg.primary_exchange:
                    c.primaryExchange = cfg.primary_exchange
            else:
                from ib_async import Future
                c = Future(
                    symbol=sym,
                    exchange=cfg.exchange,
                    currency="USD",
                )
                if cfg.trading_class:
                    c.tradingClass = cfg.trading_class
            return c
        except Exception:
            logger.warning("Cannot build contract for %s", sym)
            return None
