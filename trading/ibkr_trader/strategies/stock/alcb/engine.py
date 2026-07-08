"""ALCB T2 momentum continuation live engine.

Implements the P14 final-optimized momentum breakout strategy on 5m bars
using ib_async real-time data and the unified OMS infrastructure.

Strategy logic ported from: research/backtests/stock/engine/alcb_engine.py
Infrastructure adapted from: strategies/stock/alcb/engine.py (compression-breakout)
"""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import suppress
from copy import deepcopy
import logging
import uuid
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Any

from libs.oms.models.events import OMSEventType
from libs.oms.models.intent import Intent, IntentType
from libs.oms.instrumentation.runtime_refs import fill_runtime_refs
from .config import ET, STRATEGY_ID, StrategySettings
from .data import CanonicalBarBuilder
from .diagnostics import JsonlDiagnostics
from .artifact_store import persist_intraday_state_t2, load_intraday_state_t2
from .core import logic as alcb_core_logic
from .core.logic import apply_core_state as apply_core_runtime_state
from .core.logic import build_core_state as build_core_runtime_state
from .core.serializers import restore_state as restore_core_state
from .core.serializers import snapshot_state as snapshot_core_state
from .core.state import (
    ALCBEntryFillContext,
    ALCBEntryRequest,
    ALCBFill,
    ALCBFlattenRequest,
    ALCBOrderUpdate,
    ALCBPartialExitRequest,
    ALCBStopUpdateRequest,
)
from strategies.core.idle_market import (
    maybe_record_idle_market_observation,
    remember_idle_market_bars,
)
from strategies.core.actions import (
    FlattenPosition,
    ReplaceProtectiveStop,
    SubmitEntry,
    SubmitExit,
    SubmitPartialExit,
)
from .execution import build_entry_order, build_market_exit, build_stock_instrument, build_stop_order
from .exits import (
    carry_eligible_momentum,
    classify_momentum_trade,
    should_fr_exit,
    should_quick_exit,
    should_quick_exit_stage1,
    should_take_partial,
    update_fr_trailing_stop,
)
from .models import (
    Bar,
    CandidateArtifact,
    CandidateItem,
    Direction,
    EntryType,
    MarketSnapshot,
    PositionPlan,
    T2PositionState,
)
from .risk import (
    momentum_regime_mult,
    momentum_size_mult,
    momentum_stop_price,
    sector_sizing_mult,
)
from .signals import (
    adx_from_bars,
    atr_from_bars,
    close_location_value,
    compute_bar_rvol,
    compute_momentum_score,
    compute_opening_range,
    compute_session_avwap,
    is_momentum_breakout,
)

logger = logging.getLogger(__name__)


class ALCBT2Engine:
    """Live T2 momentum engine for ALCB (P14 final config).

    Per trading day:
    1. Initialize from CandidateArtifact (nightly research)
    2. Receive 1m bars via on_bar(), aggregate to 5m via CanonicalBarBuilder
    3. Build opening range from first N 5m bars
    4. Scan for momentum breakouts with quality gates
    5. Manage positions with exit cascade matching backtest exactly
    6. EOD flatten or carry qualified positions
    """

    def __init__(
        self,
        oms_service,
        artifact: CandidateArtifact,
        account_id: str,
        nav: float,
        settings: StrategySettings | None = None,
        trade_recorder=None,
        diagnostics: JsonlDiagnostics | None = None,
        instrumentation=None,
    ) -> None:
        self._oms = oms_service
        self._artifact = artifact
        self._items: dict[str, CandidateItem] = artifact.by_symbol
        self._account_id = account_id
        self._settings = settings or StrategySettings()
        self._trade_recorder = trade_recorder
        self._diagnostics = diagnostics or JsonlDiagnostics(
            self._settings.diagnostics_dir, enabled=False
        )
        self._instrumentation = instrumentation
        self._kit_cache = None
        self._signal_evolution: dict[str, deque] = {}

        # Position tracking (T2-specific)
        self._positions: dict[str, T2PositionState] = {}

        # Market state
        self._markets: dict[str, MarketSnapshot] = {}
        self._bar_builder = CanonicalBarBuilder()

        # Per-day session state
        self._or_built: dict[str, bool] = {}
        self._or_data: dict[str, tuple[float, float, float]] = {}  # (high, low, vol)
        self._session_bars_5m: dict[str, list[Bar]] = {}
        self._prior_day: dict[str, tuple[float, float, float]] = {}  # (pdh, pdl, pdc)

        # Order tracking
        self._order_index: dict[str, tuple[str, str]] = {}  # oms_id -> (symbol, role)
        self._pending_entries: dict[str, str] = {}  # symbol -> oms_order_id
        self._pending_exits: dict[str, str] = {}  # symbol -> oms_order_id
        self._pending_plans: dict[str, PositionPlan] = {}  # oms_id -> plan
        self._entry_meta: dict[str, dict[str, Any]] = {}  # oms_id -> entry metadata
        self._exit_reasons: dict[str, str] = {}  # oms_id -> reason

        # Portfolio
        self._equity = nav

        # Safety tracking
        self._expected_stop_cancels: set[str] = set()
        self._last_save_ts: datetime | None = None
        self._bar_ts_by_symbol: dict[str, datetime] = {}  # symbol -> last bar time (staleness)

        # Async infrastructure
        self._event_queue: asyncio.Queue | None = None
        self._event_task: asyncio.Task | None = None
        self._pulse_task: asyncio.Task | None = None
        self._running = False

        # Diagnostic pulse state
        self._last_decision_code: str = "IDLE"
        self._last_decision_details: dict = {}
        self._last_bar_ts: datetime | None = None
        self._bars_processed: int = 0

        self._initialize_from_artifact()

    def _record_decision(self, code: str, details: dict | None = None) -> None:
        """Record the latest decision for diagnostic pulse reporting."""
        if maybe_record_idle_market_observation(
            self,
            code,
            strategy_id=STRATEGY_ID,
            build_core_state=lambda: build_core_runtime_state(self),
            apply_core_state=lambda state: apply_core_runtime_state(self, state),
            on_bar=alcb_core_logic.on_bar,
            default_symbol="",
            default_timeframe="5m",
        ):
            return
        self._last_decision_code = code
        self._last_decision_details = details or {}

    def liveness_payload(self) -> dict:
        return {
            "bars_processed": self._bars_processed,
            "symbol_freshness": {
                sym: ts.isoformat() for sym, ts in self._bar_ts_by_symbol.items()
            },
        }

    @staticmethod
    def _signal_time(bar: Bar) -> datetime:
        return bar.end_time or (bar.start_time + timedelta(minutes=5))

    @staticmethod
    def _entry_bar_index(signal_time: datetime) -> int:
        signal_time_et = signal_time.astimezone(ET)
        minutes_after_open = (signal_time_et.hour * 60 + signal_time_et.minute) - 570
        return max(1, int(minutes_after_open // 5) + 1)

    @staticmethod
    def _targeted_entry_size_mult(
        entry_type: str,
        entry_bar_index: int,
        signal_time: datetime | time,
        settings: StrategySettings,
    ) -> float:
        if isinstance(signal_time, datetime):
            signal_time_et = signal_time.astimezone(ET).time()
        else:
            signal_time_et = signal_time

        mult = 1.0
        if entry_type == "PDH_BREAKOUT":
            mult *= max(0.0, settings.pdh_size_mult)
        if entry_bar_index == 9:
            mult *= max(0.0, settings.bar9_size_mult)
        if signal_time_et >= settings.late_entry_cutoff:
            mult *= max(0.0, settings.late_entry_size_mult)
        return mult

    @property
    def _instr_kit(self):
        """Lazy InstrumentationKit for direct facade calls."""
        if self._kit_cache is None and self._instrumentation is not None:
            try:
                from strategies.stock.instrumentation.src.facade import InstrumentationKit
                self._kit_cache = InstrumentationKit(self._instrumentation, strategy_type="strategy_alcb")
            except Exception:
                pass
        return self._kit_cache

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_from_artifact(self) -> None:
        for item in self._artifact.tradable:
            sym = item.symbol
            self._markets[sym] = MarketSnapshot(
                symbol=sym,
                last_price=item.price,
                daily_bars=item.daily_bars[:],
            )
            self._or_built[sym] = False
            self._session_bars_5m[sym] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running or self._oms is None:
            return
        self._running = True
        self._event_queue = self._oms.stream_events(STRATEGY_ID)
        self._event_task = asyncio.create_task(self._event_loop())
        self._pulse_task = asyncio.create_task(self._pulse_loop())
        logger.info("ALCBT2Engine started (%d symbols)", len(self._items))

    async def stop(self) -> None:
        self._running = False
        await self._save_state("stop")
        for task in (self._pulse_task, self._event_task):
            if task is None:
                continue
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        logger.info("ALCBT2Engine stopped")

    async def _reconcile_after_reconnect(self) -> None:
        """Re-sync OMS state after an IB Gateway reconnection."""
        if self._oms is None:
            return
        logger.warning("IB reconnected — triggering OMS reconciliation")
        try:
            await self._oms.request_reconciliation()
            logger.info("Post-reconnect OMS reconciliation complete")
        except Exception as exc:
            logger.error("Post-reconnect reconciliation failed: %s", exc, exc_info=exc)

    def subscription_instruments(self) -> list:
        """Return instruments needing real-time streaming subscriptions."""
        instruments = []
        seen: set[str] = set()
        for symbol in sorted(set(self._items) | set(self._positions)):
            item = self._items.get(symbol)
            if item is None or symbol in seen:
                continue
            instruments.append(build_stock_instrument(item))
            seen.add(symbol)
        return instruments

    def polling_instruments(self) -> list[tuple]:
        """Return instruments for periodic historical bar polling (cold symbols)."""
        return []  # T2 uses only hot-subscribed symbols from artifact

    async def _save_state(self, trigger: str) -> None:
        """Persist position state for crash recovery."""
        try:
            snapshot = self._build_state_snapshot()
            persist_intraday_state_t2(snapshot, settings=self._settings)
            self._last_save_ts = datetime.now(timezone.utc)
            self._diagnostics.log_order("_system_", "STATE_SAVE", {"trigger": trigger})
        except Exception as exc:
            logger.error("T2 state save failed: %s", exc, exc_info=exc)

    def _build_state_snapshot(self) -> dict:
        """Build serializable snapshot of current engine state."""
        snapshot = snapshot_core_state(build_core_runtime_state(self))
        snapshot["trade_date"] = self._artifact.trade_date.isoformat()
        snapshot["saved_at"] = datetime.now(timezone.utc).isoformat()
        return snapshot

    def hydrate_state(self, snapshot: dict) -> None:
        """Restore engine state from a persisted snapshot."""
        if not snapshot:
            return
        apply_core_runtime_state(self, restore_core_state(snapshot))
        logger.info("T2 state hydrated: %d positions", len(self._positions))

    @staticmethod
    def try_load_state(trade_date, settings=None):
        """Load persisted T2 state for the given date, or None."""
        return load_intraday_state_t2(trade_date, settings=settings)

    def health_status(self) -> dict:
        return self.snapshot_state()

    @staticmethod
    def _log_task_exception(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Unhandled exception in background task: %s", exc, exc_info=exc)

    # ------------------------------------------------------------------
    # Market data ingestion
    # ------------------------------------------------------------------

    def on_quote(self, symbol: str, quote) -> None:
        normalized = symbol.upper()
        market = self._markets.get(normalized)
        if market is None:
            return
        market.bid = quote.bid
        market.ask = quote.ask
        market.spread_pct = quote.spread_pct
        if hasattr(quote, "last") and quote.last > 0:
            market.last_price = quote.last

    def on_bar(self, symbol: str, bar: Bar) -> None:
        """Receive a completed 1m bar and aggregate to 5m."""
        normalized = symbol.upper()
        market = self._markets.get(normalized)
        if market is None:
            return
        if bar.start_time.astimezone(ET).date() != self._artifact.trade_date:
            return
        if market.minute_bars and market.minute_bars[-1].start_time >= bar.start_time:
            return

        market.minute_bars.append(bar)
        market.last_1m_bar = bar
        market.last_price = bar.close
        self._bar_ts_by_symbol[normalized] = datetime.now(timezone.utc)
        self._last_bar_ts = datetime.now(timezone.utc)
        self._bars_processed += 1
        self._bar_builder.ingest_bar(bar)

        for bar_5m in self._bar_builder.aggregate_new_bars(normalized, 5):
            self._on_5m_bar(normalized, bar_5m)

    # ------------------------------------------------------------------
    # 5m bar processing — main decision point
    # ------------------------------------------------------------------

    def _on_5m_bar(self, symbol: str, bar: Bar) -> None:
        settings = self._settings

        # Track session bars
        self._session_bars_5m.setdefault(symbol, []).append(bar)
        remember_idle_market_bars(self, [bar], symbol=symbol, timeframe="5m")

        # --- Position management ---
        if symbol in self._positions:
            self._record_decision("MANAGING_POSITION", {"symbol": symbol})
            pos = self._positions[symbol]
            pos.hold_bars += 1
            pos.update_mfe_mae(bar.high, bar.low)
            prev_stop = pos.current_stop

            # Update MFE-activated trailing stop (ratchets up only)
            pos.current_stop = update_fr_trailing_stop(
                pos.current_stop,
                pos.entry_price,
                pos.risk_per_share,
                pos.mfe_r,
                pos.direction.value,
                settings,
            )

            # Breakeven stop after reaching threshold
            if settings.close_stop_be_after_r > 0 and pos.risk_per_share > 0:
                if pos.mfe_r >= settings.close_stop_be_after_r:
                    if pos.direction == Direction.LONG:
                        be_price = pos.entry_price + 0.01
                        if be_price > pos.current_stop:
                            pos.current_stop = be_price
                    else:
                        be_price = pos.entry_price - 0.01
                        if be_price < pos.current_stop:
                            pos.current_stop = be_price

            # Adaptive trailing stop (time-phased: late-only per P14)
            if settings.adaptive_trail_start_bars > 0 and pos.hold_bars >= settings.adaptive_trail_start_bars:
                if pos.hold_bars >= settings.adaptive_trail_tighten_bars:
                    at_activate = settings.adaptive_trail_late_activate_r
                    at_distance = settings.adaptive_trail_late_distance_r
                else:
                    at_activate = settings.adaptive_trail_mid_activate_r
                    at_distance = settings.adaptive_trail_mid_distance_r
                if pos.mfe_r >= at_activate and pos.risk_per_share > 0:
                    at_trail_r = pos.mfe_r - at_distance
                    if at_trail_r > 0:
                        if pos.direction == Direction.LONG:
                            at_price = pos.entry_price + at_trail_r * pos.risk_per_share
                            if at_price > pos.current_stop:
                                pos.current_stop = at_price
                        else:
                            at_price = pos.entry_price - at_trail_r * pos.risk_per_share
                            if at_price < pos.current_stop:
                                pos.current_stop = at_price

            # Track whether any trailing mechanism ratcheted the stop
            if pos.current_stop != prev_stop:
                pos.fr_trailing_active = True
                kit = self._instr_kit
                if kit:
                    kit.log_stop_adjustment(
                        trade_id=pos.trade_id or f"ALCB-{symbol}",
                        symbol=symbol, old_stop=prev_stop, new_stop=pos.current_stop,
                        adjustment_type="trailing", trigger="alcb_composite_trail",
                    )

            # Periodic indicator snapshot while in position (every 6th bar = ~30 min)
            if pos.hold_bars % 6 == 0:
                kit = self._instr_kit
                if kit:
                    try:
                        sb = self._session_bars_5m.get(symbol, [])
                        avwap = compute_session_avwap(sb, len(sb) - 1) if sb else 0.0
                        kit.on_indicator_snapshot(
                            pair=symbol,
                            indicators={
                                "hold_bars": float(pos.hold_bars),
                                "unrealized_r": float(pos.unrealized_r(bar.close)),
                                "mfe_r": float(pos.mfe_r),
                                "current_stop": float(pos.current_stop),
                                "avwap": float(avwap),
                                "bar_close": float(bar.close),
                                "mae_r": float((pos.max_adverse - pos.entry_price) / max(pos.risk_per_share, 1e-9)),
                                "partial_taken": pos.partial_taken,
                                "fr_trailing_active": pos.fr_trailing_active,
                                "trade_class": pos.trade_class or "",
                            },
                            signal_name="alcb_position_monitor",
                            signal_strength=float(pos.unrealized_r(bar.close)),
                            decision="in_position",
                            strategy_type="strategy_alcb",
                            exchange_timestamp=bar.start_time,
                            bar_id=bar.start_time.isoformat() if bar.start_time else None,
                        )
                    except Exception:
                        pass

            # Sync updated stop to broker if it changed
            if pos.current_stop != prev_stop and pos.stop_order_id:
                task = asyncio.create_task(self._replace_stop(symbol))
                task.add_done_callback(self._log_task_exception)

            self._check_exits(symbol, bar)
            return

        # --- Opening Range Build ---
        if not self._or_built.get(symbol, False):
            self._record_decision("AWAITING_DATA", {"symbol": symbol, "reason": "building_opening_range"})
            sb = self._session_bars_5m.get(symbol, [])
            if len(sb) >= settings.opening_range_bars:
                oh, ol, ov = compute_opening_range(sb, settings.opening_range_bars)
                self._or_data[symbol] = (oh, ol, ov)
                self._or_built[symbol] = True
                self._diagnostics.log_order(
                    symbol, "or_built",
                    {"or_high": oh, "or_low": ol, "or_volume": ov, "n_bars": settings.opening_range_bars},
                )
            return

        # --- Entry Logic ---
        self._check_entry(symbol, bar)

    # ------------------------------------------------------------------
    # Exit cascade — matches backtest _exit_cascade() exactly
    # ------------------------------------------------------------------

    def _check_exits(self, symbol: str, bar: Bar) -> None:
        pos = self._positions.get(symbol)
        if pos is None:
            return
        if symbol in self._pending_exits:
            return  # exit already in flight

        settings = self._settings
        sb = self._session_bars_5m.get(symbol, [])

        # 1. CLOSE_STOP — only as fallback when no broker stop is active.
        #    The broker stop order handles intra-bar stop execution. Checking
        #    bar.low here when a broker stop is live creates a double-exit race
        #    (broker fills stop AND engine submits market exit → phantom short).
        if not pos.stop_order_id:
            stop_hit = (
                (pos.direction == Direction.LONG and bar.low <= pos.current_stop)
                or (pos.direction == Direction.SHORT and bar.high >= pos.current_stop)
            )
            if stop_hit:
                self._fire_exit(symbol, "CLOSE_STOP_FALLBACK")
                return

        # 2. QUICK_EXIT STAGE 1 (cut deeply underwater trades early)
        ur = pos.unrealized_r(bar.close)
        if should_quick_exit_stage1(pos.hold_bars, ur, settings):
            self._fire_exit(symbol, "QUICK_EXIT_S1")
            return

        # 3. QUICK_EXIT standard (cut short-hold losers)
        if should_quick_exit(pos.hold_bars, ur, settings):
            self._fire_exit(symbol, "QUICK_EXIT")
            return

        # 4. MFE conviction check (accepted P15 exit mutation)
        if settings.mfe_conviction_check_bars > 0 and pos.hold_bars == settings.mfe_conviction_check_bars:
            mfe_r = pos.unrealized_r(pos.max_favorable) if pos.risk_per_share > 0 else 0.0
            if mfe_r < settings.mfe_conviction_min_r:
                if settings.mfe_conviction_floor_r != 0.0 and ur >= settings.mfe_conviction_floor_r:
                    pass
                else:
                    self._fire_exit(symbol, "MFE_CONVICTION")
                    return

        # 5. FLOW_REVERSAL (with MFE grace + max_hold gating)
        if should_fr_exit(bar, sb, pos.entry_price, pos.hold_bars, pos.mfe_r, settings):
            self._fire_exit(symbol, "FLOW_REVERSAL")
            return

        # 6. PARTIAL_TAKE
        if settings.use_partial_takes:
            take, frac = should_take_partial(ur, pos.partial_taken, settings)
            if take:
                partial_qty = max(1, int(pos.quantity * frac))
                if partial_qty < pos.quantity:
                    self._fire_partial(symbol, partial_qty)

        # 7. EOD check
        bar_time_et = bar.start_time.astimezone(ET).time()
        if bar_time_et >= settings.eod_flatten_time:
            avwap = compute_session_avwap(sb, len(sb) - 1) if sb else 0.0
            trade_class = classify_momentum_trade(sb[-8:], pos.entry_price, avwap)
            pos.trade_class = trade_class.value if hasattr(trade_class, 'value') else str(trade_class)
            eod_cpr = close_location_value(bar)
            regime_tier = pos.regime_tier
            eligible, _ = carry_eligible_momentum(
                trade_class, ur, eod_cpr, regime_tier, settings,
            )
            if eligible:
                logger.info("T2 carry approved: %s R=%.2f CPR=%.2f", symbol, ur, eod_cpr)
                return  # hold overnight
            self._fire_exit(symbol, "EOD_FLATTEN")

    def _fire_exit(self, symbol: str, reason: str) -> None:
        """Initiate a full exit via async task, routed through core."""
        # Route through core for decision tracking
        flatten_req = ALCBFlattenRequest(symbol=symbol, reason=reason)
        core_state = build_core_runtime_state(self)
        new_state, actions, events = alcb_core_logic.on_bar(
            core_state, bar_ts=self._last_bar_ts, flatten_request=flatten_req,
        )
        apply_core_runtime_state(self, new_state)
        # Dispatch via OMS (core emits FlattenPosition action)
        for action in actions:
            if isinstance(action, FlattenPosition):
                task = asyncio.create_task(self._request_full_exit(action.symbol, action.reason))
                task.add_done_callback(self._log_task_exception)

    def _fire_partial(self, symbol: str, qty: int) -> None:
        """Initiate partial exit + stop-to-breakeven, routed through core."""
        pos = self._positions.get(symbol)
        if pos is None:
            return
        # Engine-specific: mark partial taken and move stop to BE
        pos.partial_taken = True
        if self._settings.move_stop_to_be:
            old_stop = pos.current_stop
            pos.current_stop = pos.entry_price
            kit = self._instr_kit
            if kit and old_stop != pos.entry_price:
                kit.log_stop_adjustment(
                    trade_id=pos.trade_id or f"ALCB-{symbol}",
                    symbol=symbol, old_stop=old_stop, new_stop=pos.entry_price,
                    adjustment_type="breakeven", trigger="partial_be",
                )
        # Route through core for decision tracking
        partial_req = ALCBPartialExitRequest(
            client_order_id=f"T2-{symbol}-partial-{uuid.uuid4().hex[:6]}",
            symbol=symbol, qty=qty,
        )
        core_state = build_core_runtime_state(self)
        new_state, actions, events = alcb_core_logic.on_bar(
            core_state, bar_ts=self._last_bar_ts, partial_exit_request=partial_req,
        )
        apply_core_runtime_state(self, new_state)
        # Dispatch via OMS (core emits SubmitPartialExit action)
        for action in actions:
            if isinstance(action, SubmitPartialExit):
                task = asyncio.create_task(self._submit_partial_exit(action.symbol, action.qty))
                task.add_done_callback(self._log_task_exception)

    # ------------------------------------------------------------------
    # Entry logic — matches backtest _try_entry() exactly
    # ------------------------------------------------------------------

    def _check_entry(self, symbol: str, bar: Bar) -> None:
        settings = self._settings
        if symbol in self._positions:
            return
        if symbol in self._pending_entries:
            return
        if symbol not in self._or_data:
            return

        item = self._items.get(symbol)
        if item is None:
            return

        # Gate collector for full filter_decisions breakdown
        _gates: list[dict] = []
        signal_time = self._signal_time(bar)

        # Entry window uses the closed signal bar timestamp, matching backtest semantics.
        bar_time_et = signal_time.astimezone(ET).time()
        _ew_passed = settings.entry_window_start <= bar_time_et <= settings.entry_window_end
        _gates.append({"filter_name": "entry_window", "threshold": f"{settings.entry_window_start}-{settings.entry_window_end}", "actual_value": str(bar_time_et), "passed": _ew_passed})
        if not _ew_passed:
            self._log_missed(
                symbol=symbol, blocked_by="entry_window", block_reason="outside_entry_window",
                signal_strength=0.0, exchange_timestamp=signal_time,
                filter_decisions=_gates,
            )
            return

        # Prior day data
        pdh, pdl, pdc = self._prior_day.get(symbol, (0.0, 0.0, 0.0))
        if pdh == 0.0 and item.daily_bars:
            last_daily = item.daily_bars[-1]
            pdh, pdl, pdc = last_daily.high, last_daily.low, last_daily.close

        or_high, or_low, _or_vol = self._or_data[symbol]
        expected_vol = (item.average_30m_volume / 6.0) if item.average_30m_volume > 0 else 1.0
        bar_rvol = compute_bar_rvol(bar.volume, expected_vol)
        cpr = close_location_value(bar)

        sb = self._session_bars_5m.get(symbol, [])
        avwap = compute_session_avwap(sb, len(sb) - 1) if sb else 0.0

        daily_bars = item.daily_bars
        daily_atr = atr_from_bars(daily_bars, 14) if len(daily_bars) >= 2 else 0.0
        adx_val = adx_from_bars(daily_bars, 14) if len(daily_bars) >= 16 else 0.0
        sector_flow = 0.0  # not available in CandidateItem

        entry_price = bar.close
        stop_price = momentum_stop_price(entry_price, or_low, bar.low, daily_atr, settings)
        risk_per_share = abs(entry_price - stop_price)

        # --- Breakout detection ---
        is_breakout, entry_type = is_momentum_breakout(
            bar, pdh, or_high, bar_rvol, cpr, settings,
        )
        _gates.append({"filter_name": "breakout_detection", "threshold": "true", "actual_value": str(is_breakout), "passed": is_breakout})
        if not is_breakout:
            self._record_decision("NO_SIGNAL", {"symbol": symbol, "reason": "no_breakout"})
            self._log_missed(
                symbol=symbol, blocked_by="breakout_detection", block_reason="no_breakout",
                signal_strength=0.0, exchange_timestamp=signal_time,
                filter_decisions=_gates,
            )
            return

        # --- Momentum score ---
        m_score, score_detail = compute_momentum_score(
            bar, sb, pdh, pdc, or_high, avwap, adx_val, sector_flow, settings,
        )
        entry_type_str = entry_type.value if entry_type else "OR_BREAKOUT"

        # Track signal evolution for signal_decay hypothesis
        if symbol not in self._signal_evolution:
            self._signal_evolution[symbol] = deque(maxlen=15)
        self._signal_evolution[symbol].append({
            "bar_time": signal_time.isoformat(),
            "momentum_score": float(m_score),
            "bar_rvol": float(bar_rvol),
            "avwap": float(avwap),
            "adx": float(adx_val),
        })

        self._emit_indicator_snapshot(
            symbol, m_score, bar_rvol, avwap, adx_val, daily_atr,
            or_high, or_low, entry_type_str, score_detail, signal_time,
        )

        # --- Gate checks (matching backtest exactly) ---

        # AVWAP filter
        _avwap_passed = not (avwap > 0 and bar.close < avwap)
        _gates.append({"filter_name": "avwap_filter", "threshold": float(avwap), "actual_value": float(bar.close), "passed": _avwap_passed})
        if not _avwap_passed:
            self._log_missed(
                symbol=symbol, blocked_by="avwap_filter", block_reason="below_avwap",
                signal_strength=float(m_score), exchange_timestamp=signal_time,
                filter_decisions=_gates,
            )
            return

        # RVOL cap
        _rvol_passed = not (settings.rvol_max < 999 and bar_rvol > settings.rvol_max)
        _gates.append({"filter_name": "rvol_cap", "threshold": float(settings.rvol_max), "actual_value": float(bar_rvol), "passed": _rvol_passed})
        if not _rvol_passed:
            self._log_missed(
                symbol=symbol, blocked_by="rvol_cap", block_reason="rvol_exceeded",
                signal_strength=float(m_score), exchange_timestamp=signal_time,
                filter_decisions=_gates,
            )
            return

        # Momentum score gate (with late-entry escalation)
        effective_score_min = settings.momentum_score_min
        if settings.late_entry_score_min > 0 and bar_time_et >= settings.late_entry_cutoff:
            effective_score_min = max(effective_score_min, settings.late_entry_score_min)
        _mscore_passed = m_score >= effective_score_min
        _gates.append({"filter_name": "momentum_score_gate", "threshold": float(effective_score_min), "actual_value": float(m_score), "passed": _mscore_passed})
        if not _mscore_passed:
            self._log_missed(
                symbol=symbol, blocked_by="momentum_score_gate", block_reason="below_minimum",
                signal_strength=float(m_score), exchange_timestamp=signal_time,
                filter_decisions=_gates,
            )
            return

        entry_bar_index = self._entry_bar_index(signal_time)
        avwap_dist_pct = (bar.close - avwap) / avwap if avwap > 0 else 0.0
        is_pdh_entry = entry_type_str == "PDH_BREAKOUT"
        is_bar9_entry = entry_bar_index == 9
        is_late_entry = bar_time_et >= settings.late_entry_cutoff

        if is_late_entry and settings.late_avwap_cap_pct > 0:
            _la_passed = avwap_dist_pct <= settings.late_avwap_cap_pct
            _gates.append({"filter_name": "late_avwap_cap", "threshold": float(settings.late_avwap_cap_pct), "actual_value": float(avwap_dist_pct), "passed": _la_passed, "applicable": True})
            if not _la_passed:
                self._log_missed(
                    symbol=symbol, blocked_by="late_entry_quality", block_reason="late_avwap_cap",
                    signal_strength=float(m_score), exchange_timestamp=signal_time,
                    filter_decisions=_gates,
                )
                return

        if is_bar9_entry:
            if settings.bar9_score_min > 0:
                _b9s_passed = m_score >= settings.bar9_score_min
                _gates.append({"filter_name": "bar9_score_gate", "threshold": float(settings.bar9_score_min), "actual_value": float(m_score), "passed": _b9s_passed, "applicable": True})
                if not _b9s_passed:
                    self._log_missed(
                        symbol=symbol, blocked_by="bar9_quality", block_reason="score_too_low",
                        signal_strength=float(m_score), exchange_timestamp=signal_time,
                        filter_decisions=_gates,
                    )
                    return
            if settings.bar9_rvol_min > 0:
                _b9r_passed = bar_rvol >= settings.bar9_rvol_min
                _gates.append({"filter_name": "bar9_rvol_gate", "threshold": float(settings.bar9_rvol_min), "actual_value": float(bar_rvol), "passed": _b9r_passed, "applicable": True})
                if not _b9r_passed:
                    self._log_missed(
                        symbol=symbol, blocked_by="bar9_quality", block_reason="rvol_too_low",
                        signal_strength=float(m_score), exchange_timestamp=signal_time,
                        filter_decisions=_gates,
                    )
                    return
            if settings.bar9_avwap_cap_pct > 0:
                _b9a_passed = avwap_dist_pct <= settings.bar9_avwap_cap_pct
                _gates.append({"filter_name": "bar9_avwap_cap", "threshold": float(settings.bar9_avwap_cap_pct), "actual_value": float(avwap_dist_pct), "passed": _b9a_passed, "applicable": True})
                if not _b9a_passed:
                    self._log_missed(
                        symbol=symbol, blocked_by="bar9_quality", block_reason="avwap_distance_exceeded",
                        signal_strength=float(m_score), exchange_timestamp=signal_time,
                        filter_decisions=_gates,
                    )
                    return

        if is_pdh_entry:
            _pew_passed = bar_time_et <= settings.pdh_entry_window_end
            _gates.append({"filter_name": "pdh_entry_window", "threshold": str(settings.pdh_entry_window_end), "actual_value": str(bar_time_et), "passed": _pew_passed, "applicable": True})
            if not _pew_passed:
                self._log_missed(
                    symbol=symbol, blocked_by="pdh_quality", block_reason="outside_pdh_entry_window",
                    signal_strength=float(m_score), exchange_timestamp=signal_time,
                    filter_decisions=_gates,
                )
                return
            if settings.pdh_breakout_score_min > 0:
                _pds_passed = m_score >= settings.pdh_breakout_score_min
                _gates.append({"filter_name": "pdh_quality_score", "threshold": float(settings.pdh_breakout_score_min), "actual_value": float(m_score), "passed": _pds_passed, "applicable": True})
                if not _pds_passed:
                    self._log_missed(
                        symbol=symbol, blocked_by="pdh_quality", block_reason="score_too_low",
                        signal_strength=float(m_score), exchange_timestamp=signal_time,
                        filter_decisions=_gates,
                    )
                    return
            if settings.pdh_breakout_min_rvol > 0:
                _pdr_passed = bar_rvol >= settings.pdh_breakout_min_rvol
                _gates.append({"filter_name": "pdh_quality_rvol", "threshold": float(settings.pdh_breakout_min_rvol), "actual_value": float(bar_rvol), "passed": _pdr_passed, "applicable": True})
                if not _pdr_passed:
                    self._log_missed(
                        symbol=symbol, blocked_by="pdh_quality", block_reason="rvol_too_low",
                        signal_strength=float(m_score), exchange_timestamp=signal_time,
                        filter_decisions=_gates,
                    )
                    return
            if settings.pdh_avwap_cap_pct > 0:
                _pda_passed = avwap_dist_pct <= settings.pdh_avwap_cap_pct
                _gates.append({"filter_name": "pdh_avwap_cap", "threshold": float(settings.pdh_avwap_cap_pct), "actual_value": float(avwap_dist_pct), "passed": _pda_passed, "applicable": True})
                if not _pda_passed:
                    self._log_missed(
                        symbol=symbol, blocked_by="pdh_quality", block_reason="avwap_distance_exceeded",
                        signal_strength=float(m_score), exchange_timestamp=signal_time,
                        filter_decisions=_gates,
                    )
                    return

        # Block COMBINED_BREAKOUT in Tier B
        regime_tier_check = self._artifact.regime.tier if self._artifact.regime else "A"
        if entry_type_str == "COMBINED_BREAKOUT":
            _crb_passed = not (settings.block_combined_regime_b and regime_tier_check == "B")
            _gates.append({"filter_name": "combined_regime_block", "threshold": "not_B", "actual_value": regime_tier_check, "passed": _crb_passed, "applicable": True})
            if not _crb_passed:
                self._log_missed(
                    symbol=symbol, blocked_by="block_combined_regime_b",
                    block_reason="combined_blocked_in_tier_b",
                    signal_strength=float(m_score), exchange_timestamp=signal_time,
                    filter_decisions=_gates,
                )
                return

        # COMBINED_BREAKOUT quality gate
        if entry_type_str == "COMBINED_BREAKOUT":
            _cs_passed = not (settings.combined_breakout_score_min > 0 and m_score < settings.combined_breakout_score_min)
            _gates.append({"filter_name": "combined_score", "threshold": float(settings.combined_breakout_score_min), "actual_value": float(m_score), "passed": _cs_passed, "applicable": True})
            if not _cs_passed:
                self._log_missed(
                    symbol=symbol, blocked_by="combined_quality", block_reason="score_too_low",
                    signal_strength=float(m_score), exchange_timestamp=signal_time,
                    filter_decisions=_gates,
                )
                return
            _cr_passed = not (settings.combined_breakout_min_rvol > 0 and bar_rvol < settings.combined_breakout_min_rvol)
            _gates.append({"filter_name": "combined_rvol", "threshold": float(settings.combined_breakout_min_rvol), "actual_value": float(bar_rvol), "passed": _cr_passed, "applicable": True})
            if not _cr_passed:
                self._log_missed(
                    symbol=symbol, blocked_by="combined_quality", block_reason="rvol_too_low",
                    signal_strength=float(m_score), exchange_timestamp=signal_time,
                    filter_decisions=_gates,
                )
                return
            # COMBINED-specific AVWAP distance cap
            if settings.combined_avwap_cap_pct > 0 and avwap > 0:
                _ca_passed = avwap_dist_pct <= settings.combined_avwap_cap_pct
                _gates.append({"filter_name": "combined_avwap_cap", "threshold": float(settings.combined_avwap_cap_pct), "actual_value": float(avwap_dist_pct), "passed": _ca_passed, "applicable": True})
                if not _ca_passed:
                    self._log_missed(
                        symbol=symbol, blocked_by="combined_quality",
                        block_reason="avwap_distance_exceeded",
                        signal_strength=float(m_score), exchange_timestamp=signal_time,
                        filter_decisions=_gates,
                    )
                    return
            # COMBINED-specific breakout distance cap
            if settings.combined_breakout_cap_r > 0 and risk_per_share > 0:
                breakout_dist_r = (bar.close - or_high) / risk_per_share
                _cb_passed = breakout_dist_r <= settings.combined_breakout_cap_r
                _gates.append({"filter_name": "combined_breakout_cap", "threshold": float(settings.combined_breakout_cap_r), "actual_value": float(breakout_dist_r), "passed": _cb_passed, "applicable": True})
                if not _cb_passed:
                    self._log_missed(
                        symbol=symbol, blocked_by="combined_quality",
                        block_reason="breakout_distance_exceeded",
                        signal_strength=float(m_score), exchange_timestamp=signal_time,
                        filter_decisions=_gates,
                    )
                    return

        # OR_BREAKOUT quality gate
        if entry_type_str == "OR_BREAKOUT":
            _os_passed = not (settings.or_breakout_score_min > 0 and m_score < settings.or_breakout_score_min)
            _gates.append({"filter_name": "or_score", "threshold": float(settings.or_breakout_score_min), "actual_value": float(m_score), "passed": _os_passed, "applicable": True})
            if not _os_passed:
                self._log_missed(
                    symbol=symbol, blocked_by="or_quality", block_reason="score_too_low",
                    signal_strength=float(m_score), exchange_timestamp=signal_time,
                    filter_decisions=_gates,
                )
                return
            _or_passed = not (settings.or_breakout_min_rvol > 0 and bar_rvol < settings.or_breakout_min_rvol)
            _gates.append({"filter_name": "or_rvol", "threshold": float(settings.or_breakout_min_rvol), "actual_value": float(bar_rvol), "passed": _or_passed, "applicable": True})
            if not _or_passed:
                self._log_missed(
                    symbol=symbol, blocked_by="or_quality", block_reason="rvol_too_low",
                    signal_strength=float(m_score), exchange_timestamp=signal_time,
                    filter_decisions=_gates,
                )
                return

        # AVWAP distance cap
        if settings.avwap_distance_cap_pct > 0 and avwap > 0:
            _ad_passed = avwap_dist_pct <= settings.avwap_distance_cap_pct
            _gates.append({"filter_name": "avwap_distance_cap", "threshold": float(settings.avwap_distance_cap_pct), "actual_value": float(avwap_dist_pct), "passed": _ad_passed})
            if not _ad_passed:
                self._log_missed(
                    symbol=symbol, blocked_by="avwap_distance", block_reason="exceeded_cap",
                    signal_strength=float(m_score), exchange_timestamp=signal_time,
                    filter_decisions=_gates,
                )
                return

        # OR width minimum
        if settings.or_width_min_pct > 0 and or_high > 0:
            or_width_pct = (or_high - or_low) / or_high
            _ow_passed = or_width_pct >= settings.or_width_min_pct
            _gates.append({"filter_name": "or_width_min", "threshold": float(settings.or_width_min_pct), "actual_value": float(or_width_pct), "passed": _ow_passed})
            if not _ow_passed:
                self._log_missed(
                    symbol=symbol, blocked_by="or_width", block_reason="too_narrow",
                    signal_strength=float(m_score), exchange_timestamp=signal_time,
                    filter_decisions=_gates,
                )
                return

        # Breakout distance cap
        if settings.breakout_distance_cap_r > 0 and risk_per_share > 0:
            breakout_dist_r = (bar.close - or_high) / risk_per_share
            _bd_passed = breakout_dist_r <= settings.breakout_distance_cap_r
            _gates.append({"filter_name": "breakout_distance_cap", "threshold": float(settings.breakout_distance_cap_r), "actual_value": float(breakout_dist_r), "passed": _bd_passed})
            if not _bd_passed:
                self._log_missed(
                    symbol=symbol, blocked_by="breakout_distance", block_reason="exceeded_cap",
                    signal_strength=float(m_score), exchange_timestamp=signal_time,
                    filter_decisions=_gates,
                )
                return

        # Portfolio limits
        n_open = len(self._positions) + len(self._pending_entries)
        _mp_passed = n_open < settings.max_positions
        _gates.append({"filter_name": "max_positions", "threshold": float(settings.max_positions), "actual_value": float(n_open), "passed": _mp_passed})
        if not _mp_passed:
            self._log_missed(
                symbol=symbol, blocked_by="portfolio_constraint", block_reason="max_positions",
                signal_strength=float(m_score), exchange_timestamp=signal_time,
                filter_decisions=_gates,
            )
            return

        # Sector limit
        sector_count = sum(1 for p in self._positions.values() if p.sector == item.sector)
        _sl_passed = sector_count < settings.max_positions_per_sector
        _gates.append({"filter_name": "sector_limit", "threshold": float(settings.max_positions_per_sector), "actual_value": float(sector_count), "passed": _sl_passed})
        if not _sl_passed:
            self._log_missed(
                symbol=symbol, blocked_by="portfolio_constraint", block_reason="sector_limit",
                signal_strength=float(m_score), exchange_timestamp=signal_time,
                filter_decisions=_gates,
            )
            return

        # Heat cap
        open_risk = sum(p.risk_per_share * p.quantity for p in self._positions.values())
        _risk_budget = max(self._equity * settings.base_risk_fraction, 1e-9)
        _hc_passed = open_risk < settings.heat_cap_r * _risk_budget
        _open_risk_ratio = float(open_risk / _risk_budget)
        _gates.append({"filter_name": "heat_cap", "threshold": float(settings.heat_cap_r), "actual_value": _open_risk_ratio, "passed": _hc_passed})
        if not _hc_passed:
            self._log_missed(
                symbol=symbol, blocked_by="portfolio_constraint", block_reason="heat_cap_exceeded",
                signal_strength=float(m_score), exchange_timestamp=signal_time,
                filter_decisions=_gates,
            )
            return

        # Regime gate
        regime_tier = self._artifact.regime.tier if self._artifact.regime else "A"
        reg_mult = momentum_regime_mult(regime_tier, settings)
        _rg_passed = reg_mult > 0
        _gates.append({"filter_name": "regime_gate", "threshold": 0.0, "actual_value": float(reg_mult), "passed": _rg_passed})
        if not _rg_passed:
            self._log_missed(
                symbol=symbol, blocked_by="regime_gate", block_reason="regime_blocked",
                signal_strength=float(m_score), exchange_timestamp=signal_time,
                filter_decisions=_gates,
            )
            return

        # --- Sizing (matches backtest _try_entry exactly) ---
        _rps_passed = risk_per_share > 0
        _gates.append({"filter_name": "risk_per_share", "threshold": 0.0, "actual_value": float(risk_per_share), "passed": _rps_passed})
        if not _rps_passed:
            self._log_missed(
                symbol=symbol, blocked_by="sizing", block_reason="zero_risk_per_share",
                signal_strength=float(m_score), exchange_timestamp=signal_time,
                filter_decisions=_gates,
            )
            return

        size_mult = momentum_size_mult(m_score, settings)
        targeted_size_mult = self._targeted_entry_size_mult(entry_type_str, entry_bar_index, signal_time, settings)
        sec_mult = sector_sizing_mult(item.sector, settings)
        weekday = signal_time.astimezone(ET).weekday()
        dow_mult = settings.thursday_sizing_mult if weekday == 3 else settings.tuesday_sizing_mult if weekday == 1 else 1.0
        risk_budget = self._equity * settings.base_risk_fraction * reg_mult * size_mult * sec_mult * dow_mult * targeted_size_mult
        qty = int(risk_budget / risk_per_share)
        _qty_passed = qty > 0
        _gates.append({"filter_name": "qty_sizing", "threshold": 1.0, "actual_value": float(qty), "passed": _qty_passed})
        if not _qty_passed:
            self._log_missed(
                symbol=symbol, blocked_by="sizing", block_reason="qty_zero",
                signal_strength=float(m_score), exchange_timestamp=signal_time,
                filter_decisions=_gates,
            )
            return

        # Buying power constraint
        if settings.intraday_leverage > 0:
            total_notional = sum(p.entry_price * p.quantity for p in self._positions.values())
            available_bp = self._equity * settings.intraday_leverage - total_notional
            max_qty_bp = int(available_bp / entry_price)
            qty = min(qty, max_qty_bp)
            _bp_passed = qty > 0
            _gates.append({"filter_name": "buying_power", "threshold": float(available_bp), "actual_value": float(entry_price * qty if qty > 0 else entry_price), "passed": _bp_passed})
            if not _bp_passed:
                self._log_missed(
                    symbol=symbol, blocked_by="sizing", block_reason="buying_power",
                    signal_strength=float(m_score), exchange_timestamp=signal_time,
                    filter_decisions=_gates,
                )
                return

        # Participation limit
        if item.average_30m_volume > 0:
            max_qty = int(item.average_30m_volume * settings.max_participation_30m)
            _pl_clamped = qty > max_qty
            _gates.append({"filter_name": "participation_limit", "threshold": float(max_qty), "actual_value": float(qty), "passed": True, "clamped": _pl_clamped})
            qty = min(qty, max(1, max_qty))

        # --- Build plan and submit ---
        plan = PositionPlan(
            symbol=symbol,
            direction=Direction.LONG,
            entry_type=entry_type or EntryType.OR_BREAKOUT,
            entry_price=entry_price,
            stop_price=stop_price,
            tp1_price=0.0,
            tp2_price=0.0,
            quantity=qty,
            risk_per_share=risk_per_share,
            risk_dollars=qty * risk_per_share,
            quality_mult=1.0,
            regime_mult=reg_mult,
            corr_mult=1.0,
        )
        meta = {
            "entry_type": entry_type_str,
            "momentum_score": m_score,
            "score_detail": score_detail,
            "avwap": avwap,
            "or_high": or_high,
            "or_low": or_low,
            "regime_tier": regime_tier,
            "sector": item.sector,
            "bar_rvol": bar_rvol,
            "adx_val": adx_val,
            "daily_atr": daily_atr,
            "cpr": cpr,
            "selection_score": item.selection_score,
            "rs_percentile": item.relative_strength_percentile,
            "accumulation_score": item.accumulation_score,
            "stock_regime": item.stock_regime,
            "sector_regime": item.sector_regime,
            "daily_trend_sign": item.daily_trend_sign,
            "signal_ts": signal_time,
            "entry_bar_index": entry_bar_index,
            "signal_factors": self._entry_signal_factors(m_score, bar_rvol, avwap, adx_val, bar.close),
            "signal_evolution": list(self._signal_evolution.get(symbol, [])),
            "filter_decisions": _gates,
            "gate_decisions": {g["filter_name"]: g["passed"] for g in _gates},
        }
        # Route through core for decision event tracking
        entry_req = ALCBEntryRequest(
            client_order_id=f"T2-{symbol}-{uuid.uuid4().hex[:8]}",
            symbol=symbol, plan=plan, meta=meta,
        )
        core_state = build_core_runtime_state(self)
        new_state, _, _ = alcb_core_logic.on_bar(
            core_state, bar_ts=self._last_bar_ts, entry_request=entry_req,
        )
        apply_core_runtime_state(self, new_state)
        # Register synchronously BEFORE creating async task to prevent
        # concurrent entries from bypassing max_positions check (H1 fix).
        self._pending_entries[symbol] = "SUBMITTING"
        task = asyncio.create_task(self._submit_entry(symbol, plan, meta))
        task.add_done_callback(self._log_task_exception)

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    async def _submit_entry(self, symbol: str, plan: PositionPlan, meta: dict) -> None:
        if self._oms is None:
            self._pending_entries.pop(symbol, None)
            return
        item = self._items.get(symbol)
        if item is None:
            self._pending_entries.pop(symbol, None)
            return
        try:
            signal_ts = meta.get("signal_ts")
            signal_ts_text = signal_ts.isoformat() if hasattr(signal_ts, "isoformat") else str(signal_ts or "")
            entry_type = str(meta.get("entry_type") or plan.entry_type.value)
            order = build_entry_order(
                item,
                self._account_id,
                plan,
                signal_id=f"{symbol}:{entry_type}:{signal_ts_text or meta.get('entry_bar_index', '')}",
                bar_id=f"{symbol}:{signal_ts_text or meta.get('entry_bar_index', '')}",
                exchange_timestamp=signal_ts if hasattr(signal_ts, "isoformat") else None,
            )
            receipt = await self._oms.submit_intent(
                Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=order)
            )
            now = datetime.now(timezone.utc)
            if receipt.oms_order_id:
                meta["submitted_at"] = now
                update = ALCBOrderUpdate(
                    oms_order_id=receipt.oms_order_id,
                    status="accepted",
                    timestamp=now,
                    accepted_entry=ALCBEntryRequest(
                        client_order_id="",
                        symbol=symbol,
                        plan=plan,
                        meta=meta,
                    ),
                )
                state = build_core_runtime_state(self)
                new_state, _, _ = alcb_core_logic.on_order_update(state, update)
                apply_core_runtime_state(self, new_state)
                self._record_decision("ENTRY_SUBMITTED", {"symbol": symbol, "qty": plan.quantity, "price": plan.entry_price})
                self._diagnostics.log_order(symbol, "t2_submit_entry", {
                    "entry_type": meta["entry_type"],
                    "qty": plan.quantity,
                    "price": plan.entry_price,
                    "stop": plan.stop_price,
                    "momentum_score": meta["momentum_score"],
                    "bar_rvol": round(meta.get("bar_rvol", 0), 2),
                })
                logger.info(
                    "T2 entry submitted: %s %s qty=%d price=%.2f stop=%.2f score=%d",
                    symbol, meta["entry_type"], plan.quantity,
                    plan.entry_price, plan.stop_price, meta["momentum_score"],
                )
            else:
                self._pending_entries.pop(symbol, None)
                self._record_decision("ENTRY_DENIED", {"symbol": symbol, "denial_reason": receipt.denial_reason or "unknown"})
                logger.info(
                    "T2 entry denied for %s: %s", symbol,
                    receipt.denial_reason or "unknown",
                )
                self._log_missed(
                    symbol=symbol, blocked_by="oms_submit",
                    block_reason=receipt.denial_reason or "entry_denied",
                    signal_strength=float(meta.get("momentum_score", 0)),
                    exchange_timestamp=now,
                    strategy_params={
                        "entry_type": meta.get("entry_type"),
                        "entry_price": plan.entry_price,
                        "stop_price": plan.stop_price,
                        "quantity": plan.quantity,
                        "momentum_score": meta.get("momentum_score"),
                        "sector": meta.get("sector"),
                        "regime_tier": meta.get("regime_tier"),
                    },
                    filter_decisions=meta.get("filter_decisions"),
                )
        except Exception as exc:
            self._pending_entries.pop(symbol, None)
            logger.error("T2 submit_entry failed for %s: %s", symbol, exc, exc_info=exc)
            if self._instrumentation:
                try:
                    self._instrumentation.log_error(
                        error_type="submit_entry_failed", message=str(exc),
                        severity="high", category="engine",
                        context={"symbol": symbol}, exc=exc)
                except Exception:
                    pass

    async def _submit_stop(self, symbol: str, _retries: int = 2) -> None:
        pos = self._positions.get(symbol)
        if pos is None or self._oms is None:
            return
        item = self._items.get(symbol)
        if item is None or pos.stop_order_id:
            return

        last_exc: Exception | None = None
        for attempt in range(_retries + 1):
            try:
                order = build_stop_order(
                    item, self._account_id,
                    pos.quantity, pos.current_stop, pos.direction,
                )
                receipt = await self._oms.submit_intent(
                    Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=order)
                )
                if receipt.oms_order_id:
                    update = ALCBOrderUpdate(
                        oms_order_id=receipt.oms_order_id,
                        status="accepted",
                        symbol=symbol,
                        order_role="stop",
                    )
                    core_st = build_core_runtime_state(self)
                    new_st, _, _ = alcb_core_logic.on_order_update(core_st, update)
                    apply_core_runtime_state(self, new_st)
                    self._diagnostics.log_order(symbol, "t2_submit_stop", {
                        "qty": pos.quantity, "stop_price": pos.current_stop,
                    })
                    return  # success
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "T2 submit_stop attempt %d/%d failed for %s: %s",
                    attempt + 1, _retries + 1, symbol, exc,
                )
                if attempt < _retries:
                    await asyncio.sleep(0.5 * (attempt + 1))

        # All retries exhausted — position has no protective stop.
        # Fire emergency market exit to prevent unprotected exposure.
        logger.critical(
            "T2 submit_stop FAILED after %d attempts for %s — firing emergency exit: %s",
            _retries + 1, symbol, last_exc,
        )
        if self._instrumentation:
            try:
                self._instrumentation.log_error(
                    error_type="submit_stop_failed", message=str(last_exc),
                    severity="critical", category="engine",
                    context={"symbol": symbol, "retries": _retries + 1}, exc=last_exc)
            except Exception:
                pass
        self._fire_exit(symbol, "STOP_SUBMIT_FAILED")

    async def _replace_stop(self, symbol: str) -> None:
        pos = self._positions.get(symbol)
        if pos is None or not pos.stop_order_id or self._oms is None:
            return
        try:
            await self._oms.submit_intent(
                Intent(
                    intent_type=IntentType.REPLACE_ORDER,
                    strategy_id=STRATEGY_ID,
                    target_oms_order_id=pos.stop_order_id,
                    new_qty=pos.quantity,
                    new_stop_price=pos.current_stop,
                )
            )
            self._diagnostics.log_order(symbol, "t2_replace_stop", {
                "qty": pos.quantity, "stop_price": pos.current_stop,
            })
        except Exception as exc:
            logger.error("T2 replace_stop failed for %s: %s", symbol, exc, exc_info=exc)
            if self._instrumentation:
                try:
                    self._instrumentation.log_error(
                        error_type="replace_stop_failed", message=str(exc),
                        severity="high", category="engine",
                        context={"symbol": symbol}, exc=exc)
                except Exception:
                    pass

    async def _submit_partial_exit(self, symbol: str, qty: int) -> None:
        pos = self._positions.get(symbol)
        if pos is None or self._oms is None or symbol in self._pending_exits:
            return
        item = self._items.get(symbol)
        if item is None:
            return
        try:
            order = build_market_exit(item, self._account_id, qty, pos.direction)
            receipt = await self._oms.submit_intent(
                Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=order)
            )
            if receipt.oms_order_id:
                update = ALCBOrderUpdate(
                    oms_order_id=receipt.oms_order_id,
                    status="accepted",
                    symbol=symbol,
                    order_role="partial",
                )
                core_st = build_core_runtime_state(self)
                new_st, _, _ = alcb_core_logic.on_order_update(core_st, update)
                apply_core_runtime_state(self, new_st)
                self._diagnostics.log_order(symbol, "t2_submit_partial", {"qty": qty})
        except Exception as exc:
            logger.error("T2 submit_partial failed for %s: %s", symbol, exc, exc_info=exc)

    async def _request_full_exit(self, symbol: str, reason: str) -> None:
        pos = self._positions.get(symbol)
        if pos is None or self._oms is None:
            return
        if symbol in self._pending_exits:
            return
        item = self._items.get(symbol)
        if item is None:
            return

        # Cancel existing stop order first — track as expected so terminal
        # handler doesn't fire a redundant emergency exit.
        stop_cancelled = False
        if pos.stop_order_id:
            self._expected_stop_cancels.add(pos.stop_order_id)
            try:
                await self._oms.submit_intent(
                    Intent(
                        intent_type=IntentType.CANCEL_ORDER,
                        strategy_id=STRATEGY_ID,
                        target_oms_order_id=pos.stop_order_id,
                    )
                )
                stop_cancelled = True
            except Exception as exc:
                logger.warning(
                    "T2 cancel_stop failed for %s (proceeding with exit): %s",
                    symbol, exc,
                )
                # Stop may still be live at broker — _cleanup_orphaned_orders
                # on position close will attempt to clean it up.

        try:
            order = build_market_exit(item, self._account_id, pos.quantity, pos.direction)
            receipt = await self._oms.submit_intent(
                Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=order)
            )
            if receipt.oms_order_id:
                update = ALCBOrderUpdate(
                    oms_order_id=receipt.oms_order_id,
                    status="accepted",
                    symbol=symbol,
                    order_role="exit",
                    reason=reason,
                )
                core_st = build_core_runtime_state(self)
                new_st, _, _ = alcb_core_logic.on_order_update(core_st, update)
                apply_core_runtime_state(self, new_st)
                self._diagnostics.log_order(symbol, "t2_submit_exit", {
                    "reason": reason, "qty": pos.quantity,
                    "stop_cancelled": stop_cancelled,
                })
                logger.info("T2 exit submitted: %s reason=%s qty=%d", symbol, reason, pos.quantity)
        except Exception as exc:
            logger.error("T2 request_full_exit failed for %s: %s", symbol, exc, exc_info=exc)
            if self._instrumentation:
                try:
                    self._instrumentation.log_error(
                        error_type="submit_exit_failed", message=str(exc),
                        severity="critical", category="engine",
                        context={"symbol": symbol, "reason": reason}, exc=exc)
                except Exception:
                    pass

    async def _flatten_all(self, reason: str) -> None:
        for symbol in list(self._positions):
            await self._request_full_exit(symbol, reason)
        # Cancel pending entries (skip "SUBMITTING" sentinels from H1 guard)
        for symbol, oms_id in list(self._pending_entries.items()):
            if oms_id == "SUBMITTING":
                self._pending_entries.pop(symbol, None)
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
                pass

    async def _cleanup_orphaned_orders(self, symbol: str, pos: T2PositionState) -> None:
        """Cancel any remaining orders for a symbol after position close."""
        # Cancel orphaned stop order if still tracked
        if pos.stop_order_id:
            self._expected_stop_cancels.add(pos.stop_order_id)
            try:
                await self._oms.submit_intent(
                    Intent(
                        intent_type=IntentType.CANCEL_ORDER,
                        strategy_id=STRATEGY_ID,
                        target_oms_order_id=pos.stop_order_id,
                    )
                )
                logger.info("T2 cleaned up orphaned stop for %s: %s", symbol, pos.stop_order_id)
            except Exception as exc:
                logger.warning("T2 orphan stop cancel failed for %s: %s", symbol, exc)

        # Persist state after position close
        await self._save_state("position_closed")

    # ------------------------------------------------------------------
    # Async loops
    # ------------------------------------------------------------------

    async def _event_loop(self) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            try:
                await self._handle_event(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "T2 event error: %s/%s: %s",
                    getattr(event, "event_type", "UNKNOWN"),
                    getattr(event, "oms_order_id", ""),
                    exc, exc_info=exc,
                )

    async def _pulse_loop(self) -> None:
        _STALE_THRESHOLD_S = 150.0  # >2× the 1m bar interval
        _SAVE_INTERVAL_S = 60.0

        while self._running:
            now = datetime.now(timezone.utc)
            now_et = now.astimezone(ET)

            # Forced flatten near close
            if now_et.time() >= self._settings.forced_flatten:
                await self._flatten_all("forced_flatten")

            # Data staleness watchdog — warn if bars stop arriving for open positions
            if now_et.time() >= self._settings.entry_window_start:
                for sym in list(self._positions):
                    last_ts = self._bar_ts_by_symbol.get(sym)
                    if last_ts is not None:
                        gap = (now - last_ts).total_seconds()
                        if gap > _STALE_THRESHOLD_S:
                            logger.warning(
                                "T2 STALE DATA: %s — no bar for %.0fs (last: %s)",
                                sym, gap, last_ts.isoformat(),
                            )

            # Periodic state save
            if self._positions and (
                self._last_save_ts is None
                or (now - self._last_save_ts).total_seconds() > _SAVE_INTERVAL_S
            ):
                await self._save_state("periodic")

            await asyncio.sleep(5.0)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    async def _handle_event(self, event) -> None:
        if event.event_type == OMSEventType.FILL:
            await self._handle_fill(event)
        elif event.event_type in (
            OMSEventType.ORDER_CANCELLED,
            OMSEventType.ORDER_EXPIRED,
            OMSEventType.ORDER_REJECTED,
        ):
            await self._handle_terminal(event)
        elif event.event_type == OMSEventType.RISK_HALT:
            logger.warning("T2 risk halt received")
            await self._flatten_all("risk_halt")

    async def _handle_fill(self, event) -> None:
        """Route fill event through shared core decision machine."""
        payload = event.payload or {}
        oms_order_id = getattr(event, "oms_order_id", None)
        if oms_order_id is None:
            return

        lookup = self._order_index.get(oms_order_id)
        if lookup is None:
            return
        symbol, role = lookup

        fill_qty = int(payload.get("filled_qty") or payload.get("qty") or 0)
        fill_price = float(payload.get("avg_price") or payload.get("price") or 0.0)
        fill_time = datetime.now(timezone.utc)
        fill_commission = float(payload.get("commission", 0.0) or 0.0)

        # Snapshot state before core call (needed for instrumentation)
        pre_positions = {sym: pos for sym, pos in self._positions.items()}
        pre_meta = dict(self._entry_meta.get(oms_order_id, {}))
        pre_plan = self._pending_plans.get(oms_order_id)

        # Build entry context if this is an entry fill
        entry_context = None
        if role == "ENTRY":
            entry_context = ALCBEntryFillContext(
                trade_id=f"T2-{symbol}-{uuid.uuid4().hex[:8]}",
            )

        # Build fill and route through core
        fill = ALCBFill(
            oms_order_id=oms_order_id,
            fill_price=fill_price,
            fill_qty=fill_qty,
            fill_time=fill_time,
            commission=fill_commission,
            exit_type=role if role != "ENTRY" else None,
            entry_context=entry_context,
        )
        core_state = build_core_runtime_state(self)
        new_state, actions, events = alcb_core_logic.on_fill(core_state, fill)
        apply_core_runtime_state(self, new_state)

        # Post-processing: submit protective stop for new entry positions
        if role == "ENTRY":
            pos = self._positions.get(symbol)
            if pos is not None:
                pos.entry_commission = fill_commission
                await self._submit_stop(symbol)

        # Dispatch core actions (stop resize after partial fill)
        for action in actions:
            if isinstance(action, SubmitExit):
                # Protective stop — already handled by _submit_stop above
                pass
            elif isinstance(action, ReplaceProtectiveStop):
                pos = self._positions.get(action.symbol)
                if pos is not None and pos.stop_order_id:
                    task = asyncio.create_task(self._replace_stop(action.symbol))
                    task.add_done_callback(self._log_task_exception)

        # Process events for instrumentation (use pre-captured meta/plan
        # since core's on_fill pops them from state)
        pending_meta = pre_meta
        pending_plan = pre_plan
        for ev in events:
            if ev.code == "ENTRY_FILLED":
                await self._handle_entry_fill_instrumentation(
                    symbol, oms_order_id, fill_qty, fill_price, fill_time,
                    pending_meta, pending_plan, payload,
                )
            elif ev.code == "EXIT_FILLED":
                pre_pos = pre_positions.get(ev.symbol)
                reason = ev.details.get("reason", role)
                if pre_pos is not None:
                    await self._handle_exit_fill_instrumentation(
                        ev.symbol, pre_pos, fill_qty, fill_price, reason, payload,
                    )
            elif ev.code == "PARTIAL_EXIT_FILLED":
                pre_pos = pre_positions.get(ev.symbol)
                reason = ev.details.get("reason", "PARTIAL")
                if pre_pos is not None:
                    self._handle_partial_fill_instrumentation(
                        ev.symbol, pre_pos, fill_qty, fill_price, reason,
                    )
            elif ev.code == "EXIT_PARTIALLY_FILLED":
                # Non-final partial of a full exit — just log
                logger.info("T2 exit partially filled: %s qty=%d", ev.symbol, fill_qty)

        if not events:
            logger.debug("T2 unmatched fill event: %s", oms_order_id)

    async def _handle_entry_fill_instrumentation(
        self, symbol: str, oms_order_id: str,
        fill_qty: int, fill_price: float, fill_time: datetime,
        meta: dict, plan: PositionPlan | None, payload: dict | None = None,
    ) -> None:
        """Instrumentation-only entry fill handler.

        State mutations (position creation, pending cleanup, stop submission)
        are handled by core logic in ``_handle_fill``.  This method only
        records the trade via trade_recorder, instrumentation kit, and
        diagnostics.
        """
        # Position was already created by the core -- read it back
        pos = self._positions.get(symbol)
        if pos is None or fill_qty <= 0:
            return

        logger.info(
            "T2 entry filled: %s %s qty=%d price=%.2f stop=%.2f",
            symbol, pos.entry_type, fill_qty, fill_price, pos.current_stop,
        )
        self._diagnostics.log_order(symbol, "t2_entry_filled", {
            "qty": fill_qty, "price": fill_price,
            "entry_type": pos.entry_type, "risk_per_share": pos.risk_per_share,
        })

        # Record trade event for instrumentation
        if self._trade_recorder is not None:
            try:
                pos.trade_id = await self._trade_recorder.record_entry(
                    strategy_id=STRATEGY_ID,
                    instrument=symbol,
                    direction="LONG",
                    quantity=fill_qty,
                    entry_price=Decimal(str(fill_price)),
                    entry_ts=fill_time,
                    setup_tag=pos.setup_tag,
                    entry_type="stop_limit",
                    meta={
                        "entry_signal": "alcb_momentum_breakout",
                        "entry_signal_id": oms_order_id or symbol,
                        "entry_signal_strength": float(meta.get("momentum_score", 0)) / 8.0,
                        "strategy_params": {
                            "momentum_score": meta.get("momentum_score", 0),
                            "entry_type": meta.get("entry_type", ""),
                            "avwap": meta.get("avwap", 0.0),
                            "or_high": meta.get("or_high", 0.0),
                            "or_low": meta.get("or_low", 0.0),
                            "regime_tier": meta.get("regime_tier", "A"),
                            "bar_rvol": meta.get("bar_rvol", 0.0),
                            "adx_val": meta.get("adx_val", 0.0),
                            "daily_atr": meta.get("daily_atr", 0.0),
                            "cpr": meta.get("cpr", 0.0),
                            "sector": meta.get("sector", ""),
                            "selection_score": meta.get("selection_score", 0),
                            "rs_percentile": meta.get("rs_percentile", 0.0),
                            "accumulation_score": meta.get("accumulation_score", 0.0),
                            "stock_regime": meta.get("stock_regime", ""),
                            "sector_regime": meta.get("sector_regime", ""),
                            "daily_trend_sign": meta.get("daily_trend_sign", 0),
                        },
                        "signal_factors": meta.get("signal_factors", []),
                        "filter_decisions": meta.get("filter_decisions", []),
                        "signal_evolution": meta.get("signal_evolution", []),
                        "sizing_inputs": {
                            "qty": fill_qty,
                            "entry_price": float(plan.entry_price) if plan else fill_price,
                            "stop_price": float(plan.stop_price) if plan else pos.current_stop,
                            "risk_per_share": pos.risk_per_share,
                            "risk_dollars": pos.risk_per_share * fill_qty,
                            "base_risk_fraction": self._settings.base_risk_fraction,
                            "account_equity": self._equity,
                            "regime_mult": float(plan.regime_mult) if plan else 1.0,
                        },
                        "portfolio_state": self._portfolio_state_snapshot(),
                        "session_type": self._session_type(fill_time),
                        "exchange_timestamp": fill_time.isoformat(),
                        "expected_entry_price": float(plan.entry_price) if plan else fill_price,
                        "concurrent_positions": len(self._positions),
                        "drawdown_pct": round(1.0 - self._equity / max(getattr(self._settings, 'starting_equity', self._equity) or self._equity, 1e-9), 4),
                        "bar_id": f"{symbol}:{fill_time.strftime('%Y%m%dT%H%M%S')}",
                        "entry_latency_ms": (
                            int((fill_time - meta["submitted_at"]).total_seconds() * 1000)
                            if meta.get("submitted_at") else None
                        ),
                        "execution_timestamps": {
                            "signal_ts": meta["signal_ts"].isoformat() if meta.get("signal_ts") else None,
                            "submitted_ts": meta["submitted_at"].isoformat() if meta.get("submitted_at") else None,
                            "filled_ts": fill_time.isoformat(),
                        },
                    },
                    account_id=self._account_id,
                )
            except Exception as exc:
                logger.debug("T2 trade_recorder.record_entry failed: %s", exc)
        # Wire JSONL/sidecar emission for TA pipeline
        kit = self._instr_kit
        if kit:
            try:
                kit.log_entry(
                    trade_id=pos.trade_id or f"ALCB-{symbol}",
                    pair=symbol,
                    side="LONG",
                    entry_price=fill_price,
                    position_size=float(fill_qty),
                    position_size_quote=float(fill_price * fill_qty),
                    entry_signal="alcb_momentum_breakout",
                    entry_signal_id=oms_order_id or symbol,
                    entry_signal_strength=float(meta.get("momentum_score", 0)) / 8.0,
                    signal_factors=meta.get("signal_factors", []),
                    filter_decisions=meta.get("filter_decisions", []),
                    sizing_inputs={
                        "qty": fill_qty,
                        "entry_price": float(plan.entry_price) if plan else fill_price,
                        "stop_price": float(plan.stop_price) if plan else pos.current_stop,
                        "risk_per_share": pos.risk_per_share,
                        "risk_dollars": pos.risk_per_share * fill_qty,
                        "base_risk_fraction": self._settings.base_risk_fraction,
                        "account_equity": self._equity,
                        "regime_mult": float(plan.regime_mult) if plan else 1.0,
                    },
                    exchange_timestamp=fill_time,
                    strategy_params={
                        "momentum_score": meta.get("momentum_score", 0),
                        "entry_type": meta.get("entry_type", ""),
                        "regime_tier": meta.get("regime_tier", "A"),
                        "sector": meta.get("sector", ""),
                    },
                    portfolio_state=self._portfolio_state_snapshot(),
                    concurrent_positions=len(self._positions),
                    session_type=self._session_type(fill_time),
                    **fill_runtime_refs(oms_order_id or "", payload, fill_qty=fill_qty),
                )
            except Exception:
                pass
        self._log_orderbook_context(
            symbol=symbol, trade_context="entry",
            related_trade_id=pos.trade_id,
            exchange_timestamp=fill_time,
        )

    async def _handle_exit_fill_instrumentation(
        self, symbol: str, pre_pos: T2PositionState,
        fill_qty: int, fill_price: float, reason: str,
        payload: dict | None = None,
    ) -> None:
        """Instrumentation-only exit fill handler.

        State mutations (position deletion, pending cleanup, qty decrement)
        are handled by core logic in ``_handle_fill``.  This method only
        records the trade exit via trade_recorder, instrumentation kit, and
        diagnostics.  ``pre_pos`` is the position snapshot taken *before*
        the core call (the core may have already deleted it).
        """
        pos = pre_pos
        exit_comm = float((payload or {}).get("commission", 0.0) or 0.0)
        pos.exit_commission += exit_comm

        # Cancel any orphaned orders for this symbol
        await self._cleanup_orphaned_orders(symbol, pos)

        logger.info(
            "T2 position closed: %s reason=%s exit_price=%.2f",
            symbol, reason, fill_price,
        )
        self._diagnostics.log_order(symbol, "t2_position_closed", {
            "reason": reason, "exit_price": fill_price,
            "entry_price": pos.entry_price, "hold_bars": pos.hold_bars,
            "mfe_r": round(pos.mfe_r, 4),
        })

        # Compute trade_class if not already set (EOD path sets it in _check_exits)
        if not pos.trade_class:
            sb = self._session_bars_5m.get(symbol, [])
            if sb:
                avwap = compute_session_avwap(sb, len(sb) - 1)
                tc = classify_momentum_trade(sb[-8:], pos.entry_price, avwap)
                pos.trade_class = tc.value if hasattr(tc, 'value') else str(tc)

        # Record trade exit for instrumentation
        if self._trade_recorder is not None:
            try:
                exit_time = datetime.now(timezone.utc)
                total_fees = pos.entry_commission + pos.exit_commission
                net_pnl = (fill_price - pos.entry_price) * fill_qty + pos.realized_partial_pnl - total_fees
                realized_r = net_pnl / max(pos.risk_per_share * pos.qty_original, 1e-9)
                mfe_r = (pos.max_favorable - pos.entry_price) / max(pos.risk_per_share, 1e-9)
                mae_r = (pos.max_adverse - pos.entry_price) / max(pos.risk_per_share, 1e-9)
                await self._trade_recorder.record_exit(
                    trade_id=pos.trade_id,
                    exit_price=Decimal(str(fill_price)),
                    exit_ts=exit_time,
                    exit_reason=reason,
                    realized_r=Decimal(str(round(realized_r, 4))),
                    realized_usd=Decimal(str(round(net_pnl, 2))),
                    mfe_r=Decimal(str(round(mfe_r, 4))),
                    mae_r=Decimal(str(round(mae_r, 4))),
                    max_adverse_price=Decimal(str(pos.max_adverse)),
                    max_favorable_price=Decimal(str(pos.max_favorable)),
                    duration_bars=pos.hold_bars,
                    meta={
                        "exchange_timestamp": exit_time.isoformat(),
                        "expected_exit_price": fill_price,
                        "fees_paid": pos.entry_commission + pos.exit_commission,
                        "session_transitions": [],
                        "exit_latency_ms": None,
                        "hold_bars": pos.hold_bars,
                        "hold_days": max(0, (exit_time.date() - pos.entry_time.date()).days) if pos.entry_time else 0,
                        "mfe_r": round(mfe_r, 4),
                        "mae_r": round(mae_r, 4),
                        "exit_efficiency": round(realized_r / max(mfe_r, 0.01), 4) if mfe_r > 0 else 0.0,
                        "partial_taken": pos.partial_taken,
                        "partial_qty_exited": pos.partial_qty_exited,
                        "fr_trailing_active": pos.fr_trailing_active,
                        "trade_class": pos.trade_class,
                        "carry_days": pos.carry_days,
                        "entry_type": pos.entry_type,
                        "regime_tier": pos.regime_tier,
                        "momentum_score_at_entry": pos.momentum_score,
                        "sector": pos.sector,
                        "or_high": pos.or_high,
                        "or_low": pos.or_low,
                    },
                )
            except Exception as exc:
                logger.debug("T2 trade_recorder.record_exit failed: %s", exc)
        # Wire JSONL/sidecar exit emission for TA pipeline
        kit = self._instr_kit
        if kit and pos.trade_id:
            try:
                mfe_r = (pos.max_favorable - pos.entry_price) / max(pos.risk_per_share, 1e-9)
                mae_r = (pos.max_adverse - pos.entry_price) / max(pos.risk_per_share, 1e-9)
                kit.log_exit(
                    trade_id=pos.trade_id,
                    exit_price=fill_price,
                    exit_reason=reason,
                    exchange_timestamp=datetime.now(timezone.utc),
                    mfe_r=round(mfe_r, 4),
                    mae_r=round(mae_r, 4),
                    mfe_price=pos.max_favorable,
                    mae_price=pos.max_adverse,
                    **fill_runtime_refs((payload or {}).get("oms_order_id", ""), payload, fill_qty=fill_qty, is_exit=True),
                )
            except Exception:
                pass
        self._log_orderbook_context(
            symbol=symbol, trade_context="exit",
            related_trade_id=pos.trade_id,
            exchange_timestamp=datetime.now(timezone.utc),
        )

    def _handle_partial_fill_instrumentation(
        self, symbol: str, pre_pos: T2PositionState,
        fill_qty: int, fill_price: float, reason: str,
    ) -> None:
        """Instrumentation-only partial fill handler.

        State mutations (qty decrement, realized_partial_pnl, stop resize)
        are handled by core logic in ``_handle_fill``.  This method only
        records diagnostics.
        """
        pos = self._positions.get(symbol) or pre_pos
        self._diagnostics.log_order(symbol, "t2_partial_filled", {
            "qty_exited": fill_qty, "qty_remaining": pos.quantity,
            "reason": reason,
        })

    async def _handle_terminal(self, event) -> None:
        oms_order_id = getattr(event, "oms_order_id", None)
        if oms_order_id is None:
            return
        # Capture pre-state for instrumentation before core modifies it
        lookup = self._order_index.get(oms_order_id)
        if lookup is None:
            return
        symbol, role = lookup
        entry_meta = self._entry_meta.get(oms_order_id)

        event_type = getattr(event, "event_type", None)
        status_str = event_type.value if hasattr(event_type, "value") else str(event_type)

        # Expected stop cancel — skip core to avoid clearing stop_order_id
        if role == "STOP" and oms_order_id in self._expected_stop_cancels:
            self._expected_stop_cancels.discard(oms_order_id)
            self._order_index.pop(oms_order_id, None)
            logger.info("T2 order terminal (expected): %s %s %s", symbol, role, event_type)
            return

        # Route through core — pops order_index, pending_*, clears stop_order_id
        update = ALCBOrderUpdate(
            oms_order_id=oms_order_id,
            status=status_str,
            symbol=symbol,
            order_role=role.lower(),
        )
        state = build_core_runtime_state(self)
        new_state, _, events = alcb_core_logic.on_order_update(state, update)
        apply_core_runtime_state(self, new_state)

        logger.info("T2 order terminal: %s %s %s", symbol, role, event_type)

        # Post-core instrumentation and safety logic
        if role == "ENTRY":
            self._log_missed(
                symbol=symbol, blocked_by="entry_terminal",
                block_reason=str(event_type),
                signal_strength=float((entry_meta or {}).get("momentum_score", 0)),
                exchange_timestamp=datetime.now(timezone.utc),
                strategy_params=entry_meta,
            )
        elif role == "STOP":
            # Unexpected stop termination — position is now unprotected.
            # Fire emergency market exit if no exit is already pending.
            if symbol not in self._pending_exits and symbol in self._positions:
                logger.critical(
                    "T2 UNEXPECTED stop termination for %s (%s) — firing emergency exit",
                    symbol, event_type,
                )
                self._fire_exit(symbol, "EMERGENCY_STOP_LOST")

    # ------------------------------------------------------------------
    # Day reset (called by coordinator at session open)
    # ------------------------------------------------------------------

    def reset_session(self, artifact: CandidateArtifact, nav: float) -> None:
        """Reset per-day state for a new trading session."""
        # Update prior day from yesterday's bars
        for sym, bars_list in self._session_bars_5m.items():
            if bars_list:
                self._prior_day[sym] = (
                    max(b.high for b in bars_list),
                    min(b.low for b in bars_list),
                    bars_list[-1].close,
                )

        self._artifact = artifact
        self._items = artifact.by_symbol
        self._equity = nav
        self._or_built.clear()
        self._or_data.clear()
        self._session_bars_5m.clear()
        self._signal_evolution.clear()
        self._bar_builder = CanonicalBarBuilder()

        for item in artifact.tradable:
            sym = item.symbol
            if sym not in self._markets:
                self._markets[sym] = MarketSnapshot(
                    symbol=sym, last_price=item.price, daily_bars=item.daily_bars[:],
                )
            else:
                self._markets[sym].daily_bars = item.daily_bars[:]
                self._markets[sym].minute_bars.clear()
            self._or_built[sym] = False
            self._session_bars_5m[sym] = []

        # Handle overnight carry positions — check gap stops
        for sym in list(self._positions):
            if sym not in self._items:
                # Symbol no longer tradable
                self._fire_exit(sym, "NOT_IN_UNIVERSE")
                continue
            pos = self._positions[sym]
            pos.carry_days += 1
            if pos.carry_days > self._settings.max_carry_days:
                self._fire_exit(sym, "CARRY_TIMEOUT")

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    def snapshot_state(self) -> dict[str, Any]:
        snapshot = snapshot_core_state(build_core_runtime_state(self))
        # Ensure diagnostic keys are always present at top level
        snapshot["last_decision_code"] = self._last_decision_code
        snapshot["last_decision_details"] = self._last_decision_details
        snapshot["last_bar_ts"] = self._last_bar_ts
        snapshot["engine"] = "ALCBT2"
        snapshot["running"] = self._running
        snapshot["or_built_count"] = sum(1 for v in self._or_built.values() if v)
        snapshot["total_symbols"] = len(self._items)
        # Position summary with computed display fields
        pos_summary = {}
        for sym, pos in self._positions.items():
            market = self._markets.get(sym)
            last_price = (market.last_price or pos.entry_price) if market else pos.entry_price
            pos_summary[sym] = {
                "entry_price": pos.entry_price,
                "current_stop": pos.current_stop,
                "quantity": pos.quantity,
                "hold_bars": pos.hold_bars,
                "mfe_r": round(pos.mfe_r, 4),
                "entry_type": pos.entry_type,
                "partial_taken": pos.partial_taken,
                "unrealized_r": round(pos.unrealized_r(last_price), 4),
            }
        snapshot["position_summary"] = pos_summary
        return snapshot

    def get_position_snapshot(self) -> list[dict[str, Any]]:
        result = []
        for sym, pos in self._positions.items():
            market = self._markets.get(sym)
            last_price = (market.last_price or pos.entry_price) if market else pos.entry_price
            result.append({
                "symbol": sym,
                "direction": pos.direction.value,
                "entry_price": pos.entry_price,
                "current_stop": pos.current_stop,
                "quantity": pos.quantity,
                "hold_bars": pos.hold_bars,
                "mfe_r": round(pos.mfe_r, 4),
                "unrealized_r": round(pos.unrealized_r(last_price), 4),
                "entry_type": pos.entry_type,
                "setup_tag": pos.setup_tag,
                "partial_taken": pos.partial_taken,
            })
        return result

    def open_order_count(self) -> int:
        return len(self._order_index)

    # ------------------------------------------------------------------
    # Instrumentation helpers
    # ------------------------------------------------------------------

    def _session_type(self, now: datetime) -> str:
        et_now = now.astimezone(ET).time()
        if et_now < self._settings.entry_window_start:
            return "OPENING_RANGE"
        if et_now >= self._settings.forced_flatten:
            return "LATE_DAY"
        return "RTH"

    def _portfolio_state_snapshot(self) -> dict[str, Any]:
        return {
            "open_positions": len(self._positions),
            "pending_entries": len(self._pending_entries),
            "nav": self._equity,
            "sectors_in_use": sorted({p.sector for p in self._positions.values() if p.sector}),
            "regime_tier": self._artifact.regime.tier if self._artifact.regime else "A",
            "correlated_pairs_detail": [
                {"symbol": sym, "sector": p.sector, "direction": p.direction.value,
                 "unrealized_r": round(p.unrealized_r(
                     self._markets[sym].last_price
                     if self._markets[sym].last_price is not None
                     else p.entry_price), 3)}
                if sym in self._markets
                else {"symbol": sym, "sector": p.sector, "direction": p.direction.value,
                      "unrealized_r": 0.0}
                for sym, p in self._positions.items()
            ],
        }

    def _entry_signal_factors(self, m_score, bar_rvol, avwap, adx_val, bar_close) -> list[dict[str, Any]]:
        return [
            {"factor_name": "momentum_score", "factor_value": float(m_score),
             "threshold": float(self._settings.momentum_score_min), "contribution": float(m_score) / 10.0},
            {"factor_name": "bar_rvol", "factor_value": float(bar_rvol),
             "threshold": 1.0, "contribution": min(float(bar_rvol), 5.0)},
            {"factor_name": "avwap_location", "factor_value": float(bar_close),
             "threshold": float(avwap) if avwap > 0 else 0.0,
             "contribution": 1.0 if avwap > 0 and bar_close >= avwap else 0.0},
            {"factor_name": "adx", "factor_value": float(adx_val),
             "threshold": 0.0, "contribution": float(adx_val) / 100.0},
        ]

    def _entry_filter_decisions(self, symbol: str) -> list[dict]:
        """Build filter_decisions snapshot for missed opportunity analysis."""
        settings = self._settings
        n_open = len(self._positions) + len(self._pending_entries)
        item = self._items.get(symbol)
        sector = item.sector if item else ""
        sector_count = sum(1 for p in self._positions.values() if p.sector == sector) if sector else 0
        open_risk = sum(p.risk_per_share * p.quantity for p in self._positions.values())
        heat_ratio = open_risk / max(self._equity * settings.base_risk_fraction, 1e-9)
        regime_tier = self._artifact.regime.tier if self._artifact.regime else "A"
        reg_mult = momentum_regime_mult(regime_tier, settings)
        return [
            {"filter_name": "max_positions", "threshold": float(settings.max_positions),
             "actual_value": float(n_open), "passed": n_open < settings.max_positions},
            {"filter_name": "sector_limit", "threshold": float(settings.max_positions_per_sector),
             "actual_value": float(sector_count), "passed": sector_count < settings.max_positions_per_sector},
            {"filter_name": "heat_cap", "threshold": float(settings.heat_cap_r),
             "actual_value": round(heat_ratio, 4), "passed": heat_ratio < settings.heat_cap_r},
            {"filter_name": "regime_gate", "threshold": 0.0,
             "actual_value": float(reg_mult), "passed": reg_mult > 0},
        ]

    def _log_missed(self, *, symbol, blocked_by, block_reason, signal_strength=0.0,
                    exchange_timestamp, strategy_params=None, filter_decisions=None) -> None:
        kit = self._instr_kit
        if kit is None:
            return
        try:
            item = self._items.get(symbol)
            kit.log_missed(
                pair=symbol, side="LONG", signal="alcb_momentum_breakout",
                signal_id=f"{symbol}:{blocked_by}:{int(exchange_timestamp.timestamp())}",
                signal_strength=float(signal_strength),
                blocked_by=blocked_by, block_reason=block_reason,
                strategy_params=strategy_params or {
                    "regime_tier": self._artifact.regime.tier if self._artifact.regime else "A",
                    "sector": item.sector if item else "",
                },
                filter_decisions=filter_decisions or self._entry_filter_decisions(symbol),
                session_type=self._session_type(exchange_timestamp),
                concurrent_positions=len(self._positions),
                exchange_timestamp=exchange_timestamp,
                signal_evolution=list(self._signal_evolution.get(symbol, [])),
            )
        except Exception:
            pass

    def _log_orderbook_context(self, *, symbol, trade_context, related_trade_id="",
                               exchange_timestamp=None) -> None:
        kit = self._instr_kit
        if kit is None:
            return
        market = self._markets.get(symbol)
        if market is None:
            return
        best_bid = market.bid or (market.last_price or 0.0)
        best_ask = market.ask or (market.last_price or 0.0)
        if best_bid <= 0 or best_ask <= 0:
            return
        bid_depth = float(getattr(market.last_quote, "bid_size", 0.0) or 0.0) if market.last_quote else 0.0
        ask_depth = float(getattr(market.last_quote, "ask_size", 0.0) or 0.0) if market.last_quote else 0.0
        bid_levels = [{"price": best_bid, "size": bid_depth}] if bid_depth > 0 else None
        ask_levels = [{"price": best_ask, "size": ask_depth}] if ask_depth > 0 else None
        try:
            kit.on_orderbook_context(
                pair=symbol, best_bid=best_bid, best_ask=best_ask,
                trade_context=trade_context, related_trade_id=related_trade_id or None,
                bid_depth_10bps=bid_depth, ask_depth_10bps=ask_depth,
                bid_levels=bid_levels, ask_levels=ask_levels,
                exchange_timestamp=exchange_timestamp,
            )
        except Exception:
            pass

    def _emit_indicator_snapshot(self, symbol, m_score, bar_rvol, avwap, adx_val,
                                daily_atr, or_high, or_low, entry_type_str,
                                score_detail, bar_time) -> None:
        kit = self._instr_kit
        if kit is None:
            return
        market = self._markets.get(symbol)
        try:
            kit.on_indicator_snapshot(
                pair=symbol,
                indicators={
                    "momentum_score": float(m_score),
                    "bar_rvol": float(bar_rvol),
                    "avwap": float(avwap),
                    "adx": float(adx_val),
                    "daily_atr": float(daily_atr),
                    "or_high": float(or_high),
                    "or_low": float(or_low),
                    "spread_pct": float(market.spread_pct) if market else 0.0,
                    "entry_type": entry_type_str,
                },
                signal_name="alcb_momentum_decision",
                signal_strength=float(m_score),
                decision="candidate",
                strategy_type="strategy_alcb",
                exchange_timestamp=bar_time,
                bar_id=bar_time.isoformat() if bar_time else None,
                context={"score_detail": score_detail},
            )
        except Exception:
            pass
