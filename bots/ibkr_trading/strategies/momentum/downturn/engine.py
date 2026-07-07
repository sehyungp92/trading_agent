"""Downturn Dominator v1 -- live engine for short-only regime-adaptive MNQ scalping.

Follows NQDTC engine patterns for bar fetching, OMS integration, and state
persistence.  Reuses all backtest pure-function modules for signals, stops,
regime classification, and indicators.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from copy import deepcopy
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

# OMS models
from libs.market_data.futures_roll import roll_blackout_reason, roll_force_flatten_reason
from libs.market_data.live_futures import req_panama_adjusted_historical_data
from libs.oms.models.events import OMSEventType
from libs.oms.models.intent import Intent, IntentType
from libs.oms.models.order import (
    OMSOrder, OrderRole, OrderSide, OrderType, RiskContext,
)
from libs.oms.instrumentation.runtime_refs import fill_runtime_refs
from libs.oms.risk.calculator import RiskCalculator
from strategies.core.actions import CancelAction, SubmitEntry, SubmitExit
from strategies.core.idle_market import (
    maybe_record_idle_market_observation,
    remember_idle_market_bars,
)

# Pure-function modules (shared with backtests)
from .indicators import (
    IncrementalATR,
    IncrementalEMA,
    compute_atr,
    compute_adx_suite,
    compute_box_adaptive_length,
    compute_ema,
    compute_ema_array,
    compute_extension,
    compute_macd_hist,
    compute_momentum_slope_ok,
    compute_session_vwap,
    compute_sma,
    compute_trend_strength,
    percentile_rank,
)
from .regime import (
    check_bear_structure_override,
    check_drawdown_override,
    check_fast_crash_override,
    classify_4h_regime,
    classify_daily_trend,
    compute_bear_conviction,
    compute_composite_regime,
    compute_regime_on,
    compute_strong_bear,
    compute_vol_factor,
    compute_vol_state,
    regime_sizing_mult,
)
from .signals import (
    compute_entry_subtype_stop,
    detect_fade_short,
    detect_momentum_impulse,
    detect_reversal_short,
    update_box_state,
)
from .stops import (
    check_catastrophic_exit,
    check_climax_exit,
    check_stale_exit,
    check_vwap_failure_exit,
    compute_adaptive_lock_pct,
    compute_breakeven_stop,
    compute_chandelier_regime_mult,
    compute_multi_tier_profit_floor,
    compute_profit_floor_stop,
    compute_tiered_tp_schedule,
    update_chandelier_trail,
)
from .bt_models import BreakdownBoxState

# Local config & models
from . import config as C
from .models import (
    ActivePosition,
    CompositeRegime,
    DownturnRegimeCtx,
    EngineCounters,
    EngineTag,
    FadeSignal,
    FadeState,
    ReversalState,
    VolState,
    WorkingEntry,
)
from .core import logic as downturn_core_logic
from .core.serializers import restore_state as restore_core_state
from .core.serializers import snapshot_state as snapshot_core_state
from .core.state import (
    DownturnCoreState,
    DownturnEntryRequest,
    DownturnFill,
    DownturnOrderUpdate,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timezone for session classification
# ---------------------------------------------------------------------------
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

RTH_START = time(9, 30)
RTH_END = time(16, 0)


class DownturnEngine:
    """Live engine for Downturn Dominator v1.

    Short-only regime-adaptive MNQ scalper.  Three sub-engines:
      Reversal  -- 4H pivot bearish divergence
      Breakdown -- 30m box breach (disabled in R7c)
      Fade      -- 15m VWAP rejection
    Plus momentum impulse as an alternative fade entry.
    """

    # ── Constructor ───────────────────────────────────────────────────

    def __init__(
        self,
        ib_session: Any,
        oms_service: Any,
        instruments: dict[str, Any],
        trade_recorder: Any = None,
        equity: float = 10_000.0,
        symbol: str = C.DEFAULT_SYMBOL,
        state_dir: Path | None = None,
        instrumentation: Any = None,
        equity_alloc_pct: float = 1.0,
        disable_background_tasks: bool = False,
    ) -> None:
        self._ib = ib_session
        self._oms = oms_service
        self._instruments = instruments
        self._trade_recorder = trade_recorder
        self._equity = equity
        self._equity_alloc_pct = equity_alloc_pct
        self._symbol = symbol
        self._state_dir = state_dir or Path(".")
        self._instr = instrumentation
        self._disable_background_tasks = bool(disable_background_tasks)
        self._running = False

        # R7c configuration
        self._flags = C.R7C_FLAGS
        self._po = C.R7C_PARAM_OVERRIDES

        # Instrumentation kit
        self._kit: Any = None
        try:
            from strategies.momentum.instrumentation.src.facade import InstrumentationKit
            self._kit = InstrumentationKit(self._instr, strategy_type="downturn")
        except Exception:
            pass

        # Contract specs
        # Note: backtest uses tick_size=0.50 (MNQ tick_value) for stop/entry
        # rounding in all pure-function modules.  Use tick_value here to match.
        spec = C.NQ_SPECS.get(symbol, C.NQ_SPECS["MNQ"])
        self._tick_size: float = spec["tick_value"]
        self._point_value: float = spec["point_value"]

        # ── Regime & state objects ────────────────────────────────────
        self._regime = DownturnRegimeCtx()
        self._reversal = ReversalState()
        self._box = BreakdownBoxState()
        self._fade = FadeState()

        # ── Incremental indicators (O(1) per boundary) ───────────────
        self._inc_atr_15m = IncrementalATR(14)
        self._inc_ema_15m_fast = IncrementalEMA(5)
        self._inc_ema_15m_slow = IncrementalEMA(13)
        self._inc_atr_30m = IncrementalATR(14)
        self._inc_atr_30m_fast = IncrementalATR(5)
        self._inc_atr_1h = IncrementalATR(14)
        self._inc_ema_1h_20 = IncrementalEMA(20)

        # ── Float indicator caches ────────────────────────────────────
        self._atr_d: float = 0.0
        self._atr_d_baseline: float = 0.0
        self._atr_d_pctl: float = 0.5
        self._atr_30m: float = 0.0
        self._atr_15m: float = 0.0
        self._atr_1h: float = 0.0
        self._atr_4h: float = 0.0
        self._atr_4h_fast: float = 0.0
        self._atr_4h_slow: float = 0.0
        self._ema_fast_d: float = 0.0
        self._ema_slow_d: float = 0.0
        self._sma200_d: float = 0.0
        self._short_sma_d: float = 0.0
        self._ema20_1h: float = 0.0
        self._prev_ema_fast_d: float = 0.0

        # ── Rolling lists ─────────────────────────────────────────────
        self._atr_d_history: list[float] = []
        self._trend_strength_3d: list[float] = []
        self._mom15: list[float] = []
        self._4h_highs: list[float] = []
        self._4h_lows: list[float] = []
        self._4h_macd_hist: list[float] = []

        # ── State flags ───────────────────────────────────────────────
        self._fast_crash_active: bool = False
        self._bear_conviction: float = 0.0
        self._bear_structure_active: bool = False
        self._drawdown_override_active: bool = False
        self._regime_on: bool = False
        self._mom15_slope_ok: bool = False
        self._vwap_cross_count: int = 0
        self._in_correction_now: bool = False
        self._momentum_impulse_pending: bool = False
        self._daily_adx: float = 0.0
        self._daily_plus_di: float = 0.0
        self._daily_minus_di: float = 0.0
        self._session_start_15m_idx: int = 0

        # ── Daily risk ────────────────────────────────────────────────
        self._daily_loss: float = 0.0
        self._daily_trades: int = 0
        self._circuit_breaker_tripped: bool = False
        self._last_risk_reset_date: str = ""

        # ── Position & order tracking ─────────────────────────────────
        self._position: Optional[ActivePosition] = None
        self._working_entries: list[WorkingEntry] = []
        self._bars_since_last_entry: int = 999
        self._roll_flatten_pending: bool = False
        self._roll_flatten_oms_id: str = ""

        # ── Engine counters ───────────────────────────────────────────
        self._reversal_ctr = EngineCounters()
        self._fade_ctr = EngineCounters()

        # ── Bar caches (dict-of-arrays, NQDTC pattern) ────────────────
        self._bars_5m: dict[str, Any] = {}
        self._bars_15m: dict[str, Any] = {}
        self._bars_30m: dict[str, Any] = {}
        self._bars_1h: dict[str, Any] = {}
        self._bars_4h: dict[str, Any] = {}
        self._bars_daily: dict[str, Any] = {}

        # Bar counts for boundary detection
        self._n_bars: dict[str, int] = {
            "daily": 0, "4h": 0, "1h": 0, "30m": 0, "15m": 0, "5m": 0,
        }

        # Tasks
        self._cycle_task: Optional[asyncio.Task] = None
        self._event_task: Optional[asyncio.Task] = None
        self._event_queue: Any = None
        self._bar_count_5m: int = 0
        self._signal_ring: deque[dict] = deque(maxlen=20)

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
            on_bar=downturn_core_logic.on_bar,
            default_symbol=self._symbol,
            default_timeframe="5m",
        ):
            return
        self._last_decision_code = code
        self._last_decision_details = details or {}

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        logger.info("Downturn engine starting (symbol=%s) ...", self._symbol)
        self._running = True
        self._restore_state()

        # Stream OMS events
        self._event_queue = self._oms.stream_events(C.STRATEGY_ID)
        self._event_task = asyncio.create_task(self._process_events())

        if not self._disable_background_tasks:
            # Initial bar fetch + state initialization
            await self._fetch_bars(request_kind="startup")
            self._initialize_boundaries()

            self._cycle_task = asyncio.create_task(self._5m_scheduler())
        logger.info("Downturn engine started (symbol=%s)", self._symbol)

    async def stop(self) -> None:
        logger.info("Downturn engine stopping ...")
        self._running = False
        for task in [self._cycle_task, self._event_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        # Cancel working entry orders
        for we in list(self._working_entries):
            await self._cancel_order(we.oms_order_id)
        self._working_entries.clear()
        self._persist_state()
        logger.info("Downturn engine stopped")

    def health_status(self) -> dict[str, Any]:
        return {
            "strategy_id": C.STRATEGY_ID,
            "running": self._running,
            "symbol": self._symbol,
            "equity": self._equity,
            "position_open": self._position is not None,
            "regime": self._regime.composite_regime.value if self._regime else "unknown",
            "daily_loss": self._daily_loss,
            "circuit_breaker": self._circuit_breaker_tripped,
            "bar_count_5m": self._bar_count_5m,
            "last_decision_code": self._last_decision_code,
            "last_decision_details": self._last_decision_details,
            "last_bar_ts": self._last_bar_ts.isoformat() if self._last_bar_ts else None,
        }

    def liveness_payload(self) -> dict:
        return {
            "bars_processed": self._bar_count_5m,
            "symbol_freshness": {
                sym: ts.isoformat() for sym, ts in self._symbol_last_bar_ts.items()
            },
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

    def _build_core_state(self) -> DownturnCoreState:
        return DownturnCoreState(
            symbol=self._symbol,
            position=deepcopy(self._position),
            working_entries=deepcopy(self._working_entries),
            bar_count_5m=self._bar_count_5m,
            bars_since_last_entry=self._bars_since_last_entry,
            last_decision_code=self._last_decision_code,
            last_decision_details=dict(self._last_decision_details),
            last_bar_ts=self._last_bar_ts,
        )

    def _apply_core_state(self, state: DownturnCoreState) -> None:
        if state.symbol:
            self._symbol = state.symbol
        self._position = deepcopy(state.position)
        self._working_entries = deepcopy(state.working_entries)
        self._bar_count_5m = state.bar_count_5m
        self._bars_since_last_entry = state.bars_since_last_entry
        self._last_decision_code = state.last_decision_code
        self._last_decision_details = dict(state.last_decision_details)
        self._last_bar_ts = state.last_bar_ts

    def _apply_core_events(self, events: list[Any]) -> None:
        for event in events:
            self._record_decision(event.code, dict(event.details))
            self._last_bar_ts = event.ts

    # ── Instrumentation helpers ──────────────────────────────────────

    def _dd_tier_name(self) -> str:
        """Drawdown tier based on daily P&L relative to circuit breaker threshold."""
        cb = self._po.get("circuit_breaker_threshold", -3000.0)
        if cb >= 0:
            return "full"
        ratio = self._daily_loss / cb  # 0.0 = fresh, 1.0 = at threshold
        if ratio < 0.33:
            return "full"
        elif ratio < 0.66:
            return "half"
        elif ratio < 1.0:
            return "quarter"
        return "halt"

    def _build_signal_evolution(self, n: int = 5) -> list[dict]:
        """Ring buffer of recent signal evaluations."""
        items = list(self._signal_ring)[-n:]
        return [{"bars_ago": n - 1 - i, **s} for i, s in enumerate(items)]

    def _get_bid_ask(self) -> tuple[float, float] | None:
        """Get current bid/ask from IB tickers (NQDTC pattern)."""
        try:
            for t in self._ib.ib.tickers():
                if t.contract and t.contract.symbol in ("MNQ", "NQ"):
                    if t.bid > 0 and t.ask > 0:
                        return (t.bid, t.ask)
        except Exception:
            pass
        return None

    def _build_indicator_dict(self, effective_regime) -> dict[str, float]:
        """Shared indicator snapshot dict."""
        return {
            "composite_regime": self._regime.composite_regime.value,
            "effective_regime": effective_regime.value if hasattr(effective_regime, "value") else str(effective_regime),
            "bear_conviction": round(self._bear_conviction, 2),
            "strong_bear": int(self._regime.strong_bear),
            "in_correction": int(self._in_correction_now),
            "atr_1h": round(self._atr_1h, 4),
            "atr_d": round(self._atr_d, 4),
            "ema_fast_d": round(self._ema_fast_d, 2),
            "vol_state": self._regime.vol_state.value,
            "trend_strength": round(self._regime.trend_strength, 4),
            "fast_crash_active": int(self._fast_crash_active),
            "daily_trend": self._regime.daily_trend,
        }

    def _emit_indicator_snapshot(self, effective_regime, signal_name: str = "downturn_eval",
                                 decision: str = "skip", signal_strength: float = 0.0) -> None:
        """Fire-and-forget indicator snapshot emission."""
        if not self._kit or not self._kit.active:
            return
        try:
            self._kit.on_indicator_snapshot(
                pair=self._symbol,
                indicators=self._build_indicator_dict(effective_regime),
                signal_name=signal_name,
                signal_strength=signal_strength,
                decision=decision,
                strategy_type="downturn",
                exchange_timestamp=datetime.now(timezone.utc),
                context={
                    "session": self._classify_session(),
                    "concurrent_positions": 1 if self._position else 0,
                    "drawdown_tier": self._dd_tier_name(),
                    "daily_trades": self._daily_trades,
                },
            )
        except Exception:
            pass

    def _log_missed(self, signal_name: str, signal_id: str,
                    signal_strength: float, blocked_by: str,
                    block_reason: str, *,
                    filter_decisions: list[dict] | None = None,
                    **extra) -> None:
        """Log missed opportunity with the shared momentum instrumentation shape."""
        if not self._kit or not self._kit.active:
            return
        try:
            self._kit.log_missed(
                pair=self._symbol,
                side="SHORT",
                signal=signal_name,
                signal_id=signal_id,
                signal_strength=signal_strength,
                blocked_by=blocked_by,
                block_reason=block_reason,
                filter_decisions=filter_decisions,
                strategy_params={
                    "composite_regime": self._regime.composite_regime.value,
                    "bear_conviction": round(self._bear_conviction, 2),
                    "strong_bear": self._regime.strong_bear,
                    "in_correction": self._in_correction_now,
                    "vol_state": self._regime.vol_state.value,
                    "daily_loss": round(self._daily_loss, 2),
                    "daily_trades": self._daily_trades,
                    **extra,
                },
                session_type=self._classify_session(),
                concurrent_positions=1 if self._position else 0,
                drawdown_pct=round(abs(self._daily_loss / self._equity) * 100, 2) if self._equity > 0 else 0,
                drawdown_tier=self._dd_tier_name(),
                signal_evolution=self._build_signal_evolution(),
            )
        except Exception:
            pass

    # ── 5m Scheduler ──────────────────────────────────────────────────

    async def _5m_scheduler(self) -> None:
        while self._running:
            now = datetime.now(timezone.utc)
            minute = now.minute
            next_5 = ((minute // 5) + 1) * 5
            if next_5 >= 60:
                target = (now + timedelta(hours=1)).replace(
                    minute=next_5 - 60, second=10, microsecond=0,
                )
            else:
                target = now.replace(minute=next_5, second=10, microsecond=0)
            wait = max(0, (target - now).total_seconds())
            await asyncio.sleep(wait)
            if not self._running:
                break
            try:
                await self._on_5m_close()
            except Exception:
                logger.exception("Error in 5m cycle")

    async def _on_5m_close(self) -> None:
        """Core 5m cycle: fetch bars, detect boundaries, manage/signal."""
        await self._refresh_equity()
        await self._fetch_bars()
        self._last_bar_ts = datetime.now(timezone.utc)
        self._bar_count_5m += 1
        self._symbol_last_bar_ts[self._symbol] = self._last_bar_ts

        # Detect new boundaries by comparing bar counts
        new_d = self._detect_boundary("daily")
        new_4h = self._detect_boundary("4h")
        new_1h = self._detect_boundary("1h")
        new_30m = self._detect_boundary("30m")
        new_15m = self._detect_boundary("15m")

        # Check daily risk reset
        self._check_daily_risk_reset()

        # Run boundary handlers (order matters: daily first)
        if new_d:
            self._on_daily_boundary()
        if new_4h:
            self._on_4h_boundary()
        if new_1h:
            self._on_1h_boundary()
        if new_30m:
            self._on_30m_boundary()
        if new_15m:
            self._on_15m_boundary()

        # Position tracking
        if self._position is not None:
            self._position.hold_bars_5m += 1

        # Increment entry cooldown
        self._bars_since_last_entry += 1

        # TTL expiry for working entries
        self._expire_working_entries()

        if await self._force_flatten_for_roll(self._last_bar_ts):
            self._persist_state()
            return

        # Manage existing position
        if self._position is not None:
            self._record_decision("MANAGING_POSITION")
            await self._manage_position()

        # Evaluate new entries
        if self._position is None and not self._working_entries:
            gate_block = self._check_entry_gates()
            if gate_block is None:
                result = self._evaluate_signals()
                if result is not None:
                    signal, tag = result
                    self._record_decision("ENTRY_SUBMITTED", {"tag": tag.value if hasattr(tag, 'value') else str(tag)})
                    await self._submit_entry(signal, tag)
                else:
                    self._record_decision("NO_SIGNAL")
            elif gate_block in ("session_window", "dead_zone"):
                self._record_decision("OUTSIDE_RTH", {"gate": gate_block})
            elif gate_block == "circuit_breaker":
                self._record_decision("CIRCUIT_BREAKER", {"gate": gate_block})
            else:
                self._record_decision("SIGNAL_FILTERED", {"gate": gate_block})
                # Log actionable rejections; skip high-frequency time-window noise
                self._log_missed(
                    signal_name="gate_block",
                    signal_id=f"gate_{self._bar_count_5m}",
                    signal_strength=0.0,
                    blocked_by=gate_block,
                    block_reason=f"Entry gate: {gate_block}",
                    filter_decisions=self._entry_gate_decisions(),
                )

        self._persist_state()

    # ── Bar Management ────────────────────────────────────────────────

    async def _req_completed_bars(
        self,
        contract: Any,
        duration: str,
        bar_size: str,
        *,
        use_rth: bool,
        request_kind: str,
    ) -> list[Any] | None:
        bars = await req_panama_adjusted_historical_data(
            self._ib,
            contract,
            symbol=self._symbol,
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
        if not getattr(self._ib, "is_connected", True):
            if not getattr(self, "_fetch_disconn_logged", False):
                logger.warning("Skipping bar fetch -- IB not connected")
                self._fetch_disconn_logged = True
            return
        self._fetch_disconn_logged = False

        contract = self._get_contract()
        if contract is None:
            return

        ib = self._ib.ib
        try:
            [contract] = await ib.qualifyContractsAsync(contract)
        except Exception:
            logger.warning("Contract qualification failed for %s", self._symbol)
            return

        # Fetch each TF independently so one failure doesn't block others
        fetch_specs = [
            ("5m",    "2 D",  "5 mins",  False, "_bars_5m"),
            ("15m",   "5 D",  "15 mins", False, "_bars_15m"),
            ("30m",   "60 D", "30 mins", False, "_bars_30m"),
            ("1h",    "60 D", "1 hour",  False, "_bars_1h"),
            ("4h",    "1 Y",  "4 hours", False, "_bars_4h"),
            ("daily", "2 Y",  "1 day",   True,  "_bars_daily"),
        ]
        for tf_label, duration, bar_size, use_rth, attr in fetch_specs:
            try:
                bars = await self._req_completed_bars(
                    contract,
                    duration,
                    bar_size,
                    use_rth=use_rth,
                    request_kind=request_kind,
                )
                if bars:
                    if tf_label == "5m":
                        remember_idle_market_bars(self, bars, symbol=self._symbol, timeframe="5m")
                    setattr(self, attr, self._bars_to_arrays(bars))
            except Exception:
                logger.exception("Error fetching %s bars for %s", tf_label, self._symbol)

    @staticmethod
    def _bars_to_arrays(bars: list) -> dict[str, Any]:
        result: dict[str, Any] = {
            "open": np.array([b.open for b in bars], dtype=float),
            "high": np.array([b.high for b in bars], dtype=float),
            "low": np.array([b.low for b in bars], dtype=float),
            "close": np.array([b.close for b in bars], dtype=float),
            "volume": np.array([getattr(b, "volume", 0) for b in bars], dtype=float),
        }
        # Include timestamps for session boundary detection
        dates = [getattr(b, "date", None) for b in bars]
        if dates and dates[0] is not None:
            result["time"] = dates
        return result

    def _get_contract(self) -> Any:
        try:
            from ib_async import ContFuture
            return ContFuture(symbol=self._symbol, exchange="CME", currency="USD")
        except Exception:
            logger.warning("Cannot build contract for %s", self._symbol)
            return None

    def _detect_boundary(self, tf: str) -> bool:
        """Return True if a new bar appeared in the given timeframe."""
        bars_map = {
            "daily": self._bars_daily, "4h": self._bars_4h,
            "1h": self._bars_1h, "30m": self._bars_30m,
            "15m": self._bars_15m, "5m": self._bars_5m,
        }
        bars = bars_map.get(tf, {})
        n = len(bars.get("close", []))
        prev = self._n_bars.get(tf, 0)
        self._n_bars[tf] = n
        return n > prev and prev > 0

    def _initialize_boundaries(self) -> None:
        """Run all boundary handlers once to initialize indicator state."""
        # Store initial bar counts
        for tf, bars in [
            ("daily", self._bars_daily), ("4h", self._bars_4h),
            ("1h", self._bars_1h), ("30m", self._bars_30m),
            ("15m", self._bars_15m), ("5m", self._bars_5m),
        ]:
            self._n_bars[tf] = len(bars.get("close", []))

        # Warm up incremental indicators from bar history
        self._warmup_incremental_indicators()

        # Pre-populate rolling history that boundary handlers only add
        # one entry to per call (without this, vol_state/baseline/reversal
        # pivots would be invalid for days after startup)
        self._warmup_daily_history()
        self._warmup_4h_history()

        # Run non-incremental boundary handlers to initialize state.
        # 1h/30m/15m use incremental indicators already warmed above;
        # calling their boundary handlers again would double-feed the last bar.
        if self._bars_daily and len(self._bars_daily["close"]) > 0:
            self._on_daily_boundary()
        if self._bars_4h and len(self._bars_4h["close"]) > 0:
            self._on_4h_boundary()

        # Initialize session VWAP and momentum slope for fade signals
        self._update_session_start_15m()
        bars_15m = self._bars_15m
        if bars_15m and len(bars_15m.get("close", [])) > 0:
            s = self._session_start_15m_idx
            n15 = len(bars_15m["close"])
            if s < n15:
                self._fade.vwap_session = compute_session_vwap(
                    bars_15m["high"][s:], bars_15m["low"][s:],
                    bars_15m["close"][s:], bars_15m["volume"][s:], 0,
                )
                self._fade.vwap_used = self._fade.vwap_session
        if len(self._mom15) > 3:
            self._mom15_slope_ok = compute_momentum_slope_ok(
                np.array(self._mom15[-10:]), len(self._mom15[-10:]) - 1, 3,
            )

        logger.info(
            "Initialized boundaries: regime=%s, atr_d=%.2f, atr_d_history=%d",
            self._regime.composite_regime.value, self._atr_d,
            len(self._atr_d_history),
        )

    def _warmup_incremental_indicators(self) -> None:
        """Feed historical bars into incremental ATR/EMA indicators."""
        # 1H indicators
        if self._bars_1h and len(self._bars_1h["close"]) > 0:
            h, l, c = self._bars_1h["high"], self._bars_1h["low"], self._bars_1h["close"]
            for i in range(len(c)):
                self._inc_atr_1h.update(h[i], l[i], c[i])
                self._inc_ema_1h_20.update(c[i])
            self._atr_1h = self._inc_atr_1h.value
            self._ema20_1h = self._inc_ema_1h_20.value

        # 30m indicators
        if self._bars_30m and len(self._bars_30m["close"]) > 0:
            h, l, c = self._bars_30m["high"], self._bars_30m["low"], self._bars_30m["close"]
            for i in range(len(c)):
                self._inc_atr_30m.update(h[i], l[i], c[i])
                self._inc_atr_30m_fast.update(h[i], l[i], c[i])
            self._atr_30m = self._inc_atr_30m.value

        # 15m indicators
        if self._bars_15m and len(self._bars_15m["close"]) > 0:
            h, l, c = self._bars_15m["high"], self._bars_15m["low"], self._bars_15m["close"]
            for i in range(len(c)):
                self._inc_atr_15m.update(h[i], l[i], c[i])
                ema_f = self._inc_ema_15m_fast.update(c[i])
                ema_s = self._inc_ema_15m_slow.update(c[i])
                self._mom15.append(ema_f - ema_s)
            self._atr_15m = self._inc_atr_15m.value
            # Trim mom15 to last 50 entries
            if len(self._mom15) > 50:
                self._mom15 = self._mom15[-50:]

    def _warmup_daily_history(self) -> None:
        """Pre-populate daily ATR history and trend strength from bar data.

        Without this, vol_state/vol_factor/baseline stay at defaults until 60+
        days of live data accumulate, causing incorrect position sizing.
        """
        bars = self._bars_daily
        closes = bars.get("close", np.array([]))
        highs = bars.get("high", np.array([]))
        lows = bars.get("low", np.array([]))
        n = len(closes)
        if n < 60:
            return
        po = self._po
        ema_fast_p = int(po.get("ema_fast_period", 20))
        ema_slow_p = int(po.get("ema_slow_period", 50))

        # ATR history: compute for last 60 windows (exclude last --
        # _on_daily_boundary will add it, reaching 61 total and triggering
        # baseline computation at > 60)
        for end in range(n - 60, n):
            atr = compute_atr(highs[:end], lows[:end], closes[:end], 14)
            self._atr_d_history.append(atr)

        # Trend strength history: last 10 windows
        if n >= ema_slow_p:
            for end in range(max(ema_slow_p, n - 10), n):
                ef = compute_ema(closes[:end], ema_fast_p)
                es = compute_ema(closes[:end], ema_slow_p)
                atr_val = compute_atr(highs[:end], lows[:end], closes[:end], 14)
                if atr_val > 0:
                    self._trend_strength_3d.append(
                        compute_trend_strength(ef, es, atr_val)
                    )

    def _warmup_4h_history(self) -> None:
        """Pre-populate 4H rolling lists for reversal pivot tracking.

        Without this, reversal signals can't fire until 3+ 4H bars pass
        after startup (no swing highs for divergence detection).
        """
        bars = self._bars_4h
        highs = bars.get("high", np.array([]))
        lows = bars.get("low", np.array([]))
        closes = bars.get("close", np.array([]))
        n = len(highs)
        if n < 50:
            return

        # Take last 100 bars (exclude last -- _on_4h_boundary will add it)
        lookback = min(100, n - 1)
        start = n - 1 - lookback
        for i in range(start, n - 1):
            self._4h_highs.append(float(highs[i]))
            self._4h_lows.append(float(lows[i]))

        # MACD hist for each 4H bar
        for end in range(start + 1, n):
            _, _, hist = compute_macd_hist(closes[:end])
            self._4h_macd_hist.append(float(hist))

    # ── Boundary Handlers ─────────────────────────────────────────────

    def _on_daily_boundary(self) -> None:
        bars = self._bars_daily
        closes = bars["close"]
        highs = bars["high"]
        lows = bars["low"]
        n = len(closes)
        if n < 14:
            return
        po = self._po
        close_d = closes[-1]

        # ATR daily (14-period)
        self._atr_d = compute_atr(highs, lows, closes, 14)
        self._atr_d_history.append(self._atr_d)
        if len(self._atr_d_history) > 60:
            self._atr_d_history = self._atr_d_history[-60:]
        if len(self._atr_d_history) > 60 and self._atr_d_baseline == 0.0:
            self._atr_d_baseline = float(np.median(self._atr_d_history[-60:]))

        # Vol state + percentile (require 60 bars for reliable pctl)
        if len(self._atr_d_history) >= 60:
            hist_arr = np.array(self._atr_d_history[-60:])
            atr_med = float(np.median(hist_arr))
            atr_pct = percentile_rank(self._atr_d, hist_arr, 60)
            self._atr_d_pctl = atr_pct
            self._regime.vol_state = compute_vol_state(atr_pct, self._atr_d, atr_med)
            self._regime.vol_factor = compute_vol_factor(
                self._atr_d_baseline, self._atr_d, atr_pct,
            )

        # EMAs & SMA200
        ema_fast_p = int(po.get("ema_fast_period", 20))
        ema_slow_p = int(po.get("ema_slow_period", 50))
        sma200_p = int(po.get("sma200_period", 200))
        self._ema_fast_d = compute_ema(closes, ema_fast_p)
        self._ema_slow_d = compute_ema(closes, ema_slow_p)
        if n >= sma200_p:
            self._sma200_d = compute_sma(closes, sma200_p)
        elif self._flags.progressive_sma:
            progressive_min = int(po.get("progressive_sma_min", 50))
            if n >= progressive_min:
                self._sma200_d = float(np.mean(closes))

        # Short SMA
        if self._flags.short_sma_trend:
            short_sma_p = int(po.get("short_sma_period", 50))
            if n >= short_sma_p:
                self._short_sma_d = compute_sma(closes, short_sma_p)
                self._regime.short_trend = -1 if close_d < self._short_sma_d else 0

        # Daily trend classification (2-bar persistence)
        if self._sma200_d > 0:
            trend, consec = classify_daily_trend(
                close_d, self._sma200_d,
                self._regime.daily_trend, self._regime.daily_trend_consec,
            )
            self._regime.daily_trend = trend
            self._regime.daily_trend_consec = consec

        # Extension
        self._regime.extension_short, self._regime.extension_long = compute_extension(
            close_d, self._ema_fast_d, self._atr_d,
        )

        # Trend strength
        ts = compute_trend_strength(self._ema_fast_d, self._ema_slow_d, self._atr_d)
        self._regime.trend_strength = ts
        self._trend_strength_3d.append(ts)
        if len(self._trend_strength_3d) > 10:
            self._trend_strength_3d = self._trend_strength_3d[-10:]

        # ADX suite (conviction scoring + bear structure)
        if self._flags.conviction_scoring or self._flags.bear_structure_override:
            self._daily_adx, self._daily_plus_di, self._daily_minus_di = compute_adx_suite(
                highs, lows, closes, 14,
            )

        # Fast crash override
        if self._flags.fast_crash_override:
            self._fast_crash_active = check_fast_crash_override(
                closes, self._ema_fast_d, self._atr_d, self._atr_d_baseline, po,
            )

        # Bear conviction
        if self._flags.conviction_scoring:
            self._bear_conviction = compute_bear_conviction(
                self._daily_adx, self._daily_plus_di, self._daily_minus_di,
                self._ema_fast_d, self._ema_slow_d, close_d,
                prev_ema_fast=self._prev_ema_fast_d,
            )

        # Bear structure override
        if self._flags.bear_structure_override:
            adx_on = po.get("bear_structure_adx_on", 25.0)
            adx_off = po.get("bear_structure_adx_off", 15.0)
            self._regime_on = compute_regime_on(
                self._daily_adx, self._regime_on, adx_on, adx_off,
            )
            self._bear_structure_active = check_bear_structure_override(
                self._daily_adx, self._daily_plus_di, self._daily_minus_di,
                close_d, self._ema_fast_d, self._ema_slow_d,
                self._regime_on, self._bear_conviction, po,
            )

        self._prev_ema_fast_d = self._ema_fast_d

        # Drawdown override
        if self._flags.drawdown_regime_override:
            dd_lookback = int(po.get("drawdown_lookback", 20))
            dd_threshold = po.get("drawdown_threshold", 0.03)
            self._drawdown_override_active = check_drawdown_override(
                closes, dd_lookback, dd_threshold,
            )

        # Correction window check (>3% drawdown from 20-day high)
        if n >= 20:
            rolling_high = float(np.max(closes[-20:]))
            dd = (close_d - rolling_high) / rolling_high if rolling_high > 0 else 0
            self._in_correction_now = dd <= -0.03
        else:
            self._in_correction_now = False

        logger.debug(
            "Daily boundary: atr_d=%.2f, trend=%d, regime=%s, correction=%s",
            self._atr_d, self._regime.daily_trend,
            self._regime.composite_regime.value, self._in_correction_now,
        )

    def _on_4h_boundary(self) -> None:
        bars = self._bars_4h
        closes = bars["close"]
        highs = bars["high"]
        lows = bars["low"]
        n = len(closes)
        if n < 50:
            return
        po = self._po

        # ADX 4H
        adx_val, _, _ = compute_adx_suite(highs, lows, closes, 14)

        # EMA50 slope
        ema50_arr = compute_ema_array(closes, 50)
        slope_4h = ema50_arr[-1] - ema50_arr[-2] if len(ema50_arr) >= 2 else 0.0
        slope_dir = 1 if slope_4h > 0 else -1

        # 4H regime classification
        self._regime.regime_4h = classify_4h_regime(
            adx_val, slope_4h,
            adx_trending_threshold=po.get("adx_trending_threshold", 25.0),
            adx_range_threshold=po.get("adx_range_threshold", 15.0),
        )

        # Composite regime
        self._regime.composite_regime = compute_composite_regime(
            self._regime.regime_4h, self._regime.daily_trend, slope_dir,
            short_trend=self._regime.short_trend if self._flags.short_sma_trend else 0,
        )

        # ATR 4H (multiple periods)
        self._atr_4h = compute_atr(highs, lows, closes, 14)
        self._atr_4h_fast = compute_atr(highs, lows, closes, 5)
        self._atr_4h_slow = compute_atr(highs, lows, closes, 20)

        # MACD histogram for divergence tracking
        _, _, hist = compute_macd_hist(closes)
        self._4h_highs.append(float(highs[-1]))
        self._4h_lows.append(float(lows[-1]))
        self._4h_macd_hist.append(float(hist))
        # Trim rolling lists
        for lst in [self._4h_highs, self._4h_lows, self._4h_macd_hist]:
            if len(lst) > 200:
                del lst[:100]

        # Reversal pivot tracking
        self._update_reversal_pivots()

        # Strong bear detection
        alignment = 1.0 if self._regime.composite_regime == CompositeRegime.ALIGNED_BEAR else 0.0
        self._regime.strong_bear = compute_strong_bear(
            self._regime.trend_strength, alignment,
        )

        # Disable reversal in strong bear or shock
        self._reversal.disabled = (
            (self._regime.strong_bear and not self._flags.allow_reversal_strong_bear)
            or self._regime.vol_state == VolState.SHOCK
        )

        # Position tracking
        if self._position is not None:
            self._position.hold_bars_4h += 1

        logger.debug(
            "4H boundary: regime_4h=%s, composite=%s, strong_bear=%s",
            self._regime.regime_4h.value, self._regime.composite_regime.value,
            self._regime.strong_bear,
        )

    def _on_1h_boundary(self) -> None:
        bars = self._bars_1h
        n = len(bars.get("close", []))
        if n == 0:
            return
        idx = n - 1
        self._atr_1h = self._inc_atr_1h.update(
            bars["high"][idx], bars["low"][idx], bars["close"][idx],
        )
        self._ema20_1h = self._inc_ema_1h_20.update(bars["close"][idx])
        if self._position is not None:
            self._position.hold_bars_1h += 1

    def _on_30m_boundary(self) -> None:
        bars = self._bars_30m
        n = len(bars.get("close", []))
        if n == 0:
            return
        idx = n - 1

        self._atr_30m = self._inc_atr_30m.update(
            bars["high"][idx], bars["low"][idx], bars["close"][idx],
        )

        # Adaptive box length
        atr_fast = self._inc_atr_30m_fast.update(
            bars["high"][idx], bars["low"][idx], bars["close"][idx],
        )
        if self._atr_30m > 0:
            atr_ratio = atr_fast / self._atr_30m
            adaptive_L = compute_box_adaptive_length(atr_ratio)
        else:
            adaptive_L = 32

        # Update box state (tracks range even with breakdown disabled)
        self._box = update_box_state(
            self._box, bars["high"][idx], bars["low"][idx], bars["close"][idx],
            self._atr_30m, adaptive_L, self._po,
        )

        # Box VWAP (from box start to current)
        if self._box.active and self._box.age > 0:
            start = max(0, n - self._box.age)
            self._box.vwap_box = compute_session_vwap(
                bars["high"][start:], bars["low"][start:],
                bars["close"][start:], bars["volume"][start:], 0,
            )

        if self._position is not None:
            self._position.hold_bars_30m += 1

    def _on_15m_boundary(self) -> None:
        bars = self._bars_15m
        n = len(bars.get("close", []))
        if n == 0:
            return
        idx = n - 1
        close_15m = bars["close"][idx]

        # ATR 15m
        self._atr_15m = self._inc_atr_15m.update(
            bars["high"][idx], bars["low"][idx], close_15m,
        )

        # Momentum EMAs
        ema_fast = self._inc_ema_15m_fast.update(close_15m)
        ema_slow = self._inc_ema_15m_slow.update(close_15m)
        self._mom15.append(ema_fast - ema_slow)
        if len(self._mom15) > 50:
            self._mom15 = self._mom15[-50:]

        # Momentum slope (3-bar)
        if len(self._mom15) > 3:
            self._mom15_slope_ok = compute_momentum_slope_ok(
                np.array(self._mom15[-10:]), len(self._mom15[-10:]) - 1, 3,
            )
        else:
            self._mom15_slope_ok = False

        # VWAP cross counting
        if len(self._mom15) >= 2 and self._mom15[-1] * self._mom15[-2] < 0:
            self._vwap_cross_count += 1

        # Session start detection (day boundary in 15m bars)
        self._update_session_start_15m()

        # Session VWAP
        s = self._session_start_15m_idx
        if s < n and n - s > 0:
            self._fade.vwap_session = compute_session_vwap(
                bars["high"][s:], bars["low"][s:],
                bars["close"][s:], bars["volume"][s:], 0,
            )
        self._fade.vwap_used = self._fade.vwap_session

        # Touch tracking
        vwap = self._fade.vwap_used
        touched = bars["high"][idx] >= vwap if vwap > 0 else False
        self._fade.touch_bars.append(touched)
        if len(self._fade.touch_bars) > 8:
            self._fade.touch_bars = self._fade.touch_bars[-8:]

        # Consecutive above VWAP
        if vwap > 0 and close_15m > vwap:
            self._fade.consecutive_above_vwap += 1
        else:
            self._fade.consecutive_above_vwap = 0

    # ── Reversal Pivot Tracking ───────────────────────────────────────

    def _update_reversal_pivots(self) -> None:
        """Track 4H swing highs for reversal bearish divergence."""
        n = len(self._4h_highs)
        if n < 3:
            return

        # Check if bar at n-2 is a local swing high
        if self._4h_highs[-2] > self._4h_highs[-3] and self._4h_highs[-2] > self._4h_highs[-1]:
            swing_high = self._4h_highs[-2]
            macd_val = self._4h_macd_hist[-2] if len(self._4h_macd_hist) >= 2 else 0.0

            if not self._reversal.divergence_arm_active:
                # First pivot (h1)
                self._reversal.h1_price = swing_high
                self._reversal.h1_idx = n - 2
                self._reversal.macd_at_h1 = macd_val
                self._reversal.l_between = float(min(self._4h_lows[-2:]))
                self._reversal.divergence_arm_active = True
            else:
                # Second pivot (h2)
                self._reversal.h2_price = swing_high
                self._reversal.h2_idx = n - 2
                self._reversal.macd_at_h2 = macd_val
                # Update l_between from h1 to now
                h1_idx = self._reversal.h1_idx
                if 0 <= h1_idx < n:
                    self._reversal.l_between = float(min(self._4h_lows[h1_idx:]))

        # Keep tracking low between pivots
        if self._reversal.divergence_arm_active and self._4h_lows:
            self._reversal.l_between = min(
                self._reversal.l_between, self._4h_lows[-1],
            )

    def _update_session_start_15m(self) -> None:
        """Find start of current trading session in 15m bars."""
        times = self._bars_15m.get("time", [])
        if not times:
            return
        today = datetime.now(ET).date()
        # Walk backward to find first bar of today
        for i in range(len(times) - 1, -1, -1):
            try:
                bd = times[i].date() if hasattr(times[i], "date") else None
                if bd is not None and bd < today:
                    self._session_start_15m_idx = i + 1
                    return
            except Exception:
                continue
        self._session_start_15m_idx = 0

    # ── Entry Guards ──────────────────────────────────────────────────

    def _check_entry_gates(self) -> str | None:
        """Check entry gates, returning rejection reason string or None if allowed."""
        flags = self._flags
        po = self._po

        if flags.daily_circuit_breaker and self._circuit_breaker_tripped:
            return "circuit_breaker"
        if flags.use_shock_block and self._regime.vol_state == VolState.SHOCK:
            return "vol_shock"

        now_ny = datetime.now(ET)
        now_et = now_ny.time()
        mins = now_et.hour * 60 + now_et.minute

        if flags.use_dead_zones and (565 <= mins < 575 or 950 <= mins < 960):
            return "dead_zone"
        if flags.use_entry_windows:
            can_enter = (575 <= mins < 950) or (240 <= mins < 565) or (1080 <= mins < 1200)
            if not can_enter:
                return "session_window"
        if flags.directional_entry_caps:
            max_daily = int(po.get("max_daily_entries", 3))
            if self._daily_trades >= max_daily:
                return "daily_cap"
        if flags.use_news_blackout and (570 <= mins < 575 or 955 <= mins < 960):
            return "news_blackout"
        if flags.friction_gate and self._atr_d > 0:
            if self._atr_d_pctl < po.get("friction_min_atr_pctl", 0.10):
                return "friction_gate"
        if flags.block_counter_regime:
            if self._regime.composite_regime == CompositeRegime.COUNTER:
                if not (flags.allow_reversal_in_correction and self._in_correction_now):
                    return "counter_regime"
        return None

    def _entry_gate_decisions(self) -> list[dict]:
        """Snapshot all entry gate states for filter_decisions."""
        flags = self._flags
        po = self._po
        now_ny = datetime.now(ET)
        mins = now_ny.hour * 60 + now_ny.minute
        max_daily = int(po.get("max_daily_entries", 3))
        min_pctl = po.get("friction_min_atr_pctl", 0.10)
        is_counter = self._regime.composite_regime == CompositeRegime.COUNTER
        allow_rev = flags.allow_reversal_in_correction and self._in_correction_now
        return [
            {"filter_name": "circuit_breaker",
             "passed": not (flags.daily_circuit_breaker and self._circuit_breaker_tripped),
             "threshold": 1.0, "actual_value": float(self._circuit_breaker_tripped)},
            {"filter_name": "vol_shock",
             "passed": not (flags.use_shock_block and self._regime.vol_state == VolState.SHOCK),
             "threshold": 1.0, "actual_value": float(self._regime.vol_state == VolState.SHOCK)},
            {"filter_name": "dead_zone",
             "passed": not (flags.use_dead_zones and (565 <= mins < 575 or 950 <= mins < 960)),
             "threshold": 1.0, "actual_value": float(565 <= mins < 575 or 950 <= mins < 960)},
            {"filter_name": "session_window",
             "passed": not flags.use_entry_windows or (575 <= mins < 950) or (240 <= mins < 565) or (1080 <= mins < 1200),
             "threshold": 1.0, "actual_value": float(mins)},
            {"filter_name": "daily_cap",
             "passed": not flags.directional_entry_caps or self._daily_trades < max_daily,
             "threshold": float(max_daily), "actual_value": float(self._daily_trades)},
            {"filter_name": "news_blackout",
             "passed": not (flags.use_news_blackout and (570 <= mins < 575 or 955 <= mins < 960)),
             "threshold": 1.0, "actual_value": float(570 <= mins < 575 or 955 <= mins < 960)},
            {"filter_name": "friction_gate",
             "passed": not flags.friction_gate or self._atr_d <= 0 or self._atr_d_pctl >= min_pctl,
             "threshold": float(min_pctl), "actual_value": round(self._atr_d_pctl, 4)},
            {"filter_name": "counter_regime",
             "passed": not (flags.block_counter_regime and is_counter and not allow_rev),
             "threshold": 1.0, "actual_value": float(is_counter)},
        ]

    def _can_enter(self) -> bool:
        """Boolean wrapper for backward compat."""
        return self._check_entry_gates() is None

    # ── Signal Evaluation ─────────────────────────────────────────────

    def _evaluate_signals(self) -> Optional[tuple[Any, EngineTag]]:
        """Evaluate signals with regime overrides.  Returns (signal, tag) or None."""
        flags = self._flags
        po = self._po

        # ── Regime overrides (expand bearish regime envelope) ─────────
        effective_regime = self._regime.composite_regime

        # Correction regime override
        if flags.correction_regime_override and self._in_correction_now:
            if effective_regime in (CompositeRegime.NEUTRAL, CompositeRegime.RANGE):
                effective_regime = CompositeRegime.EMERGING_BEAR

        # Fast crash override (conviction gate only when conviction_scoring enabled)
        if (flags.fast_crash_override and self._fast_crash_active
                and effective_regime in (CompositeRegime.NEUTRAL, CompositeRegime.RANGE)):
            if flags.conviction_scoring:
                threshold = po.get("conviction_threshold", 50)
                if self._bear_conviction >= threshold:
                    effective_regime = CompositeRegime.EMERGING_BEAR
            else:
                effective_regime = CompositeRegime.EMERGING_BEAR

        # Bear structure override
        if flags.bear_structure_override and self._bear_structure_active:
            if effective_regime in (CompositeRegime.NEUTRAL, CompositeRegime.RANGE):
                effective_regime = CompositeRegime.EMERGING_BEAR

        # Drawdown override
        if flags.drawdown_regime_override and self._drawdown_override_active:
            if effective_regime in (
                CompositeRegime.NEUTRAL, CompositeRegime.RANGE, CompositeRegime.COUNTER,
            ):
                effective_regime = CompositeRegime.EMERGING_BEAR

        # ── Signal ring (recent evaluation state for instrumentation) ──
        self._signal_ring.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "composite_regime": self._regime.composite_regime.value,
            "effective_regime": effective_regime.value,
            "bear_conviction": round(self._bear_conviction, 2),
            "strong_bear": self._regime.strong_bear,
            "in_correction": self._in_correction_now,
            "atr_1h": round(self._atr_1h, 4),
            "vol_state": self._regime.vol_state.value,
        })

        # ── 1. Reversal (4H pivot divergence) ─────────────────────────
        if flags.reversal_engine and not self._reversal.disabled:
            ts_3d_ago = (
                self._trend_strength_3d[-4]
                if len(self._trend_strength_3d) >= 4
                else 0.0
            )
            sig = detect_reversal_short(
                self._reversal,
                self._regime.trend_strength,
                ts_3d_ago,
                float(self._bars_daily["close"][-1]) if self._bars_daily else 0.0,
                self._ema_fast_d,
                self._atr_d,
                self._atr_4h_fast,
                self._atr_4h_slow,
                flags,
                po,
            )
            if sig is not None:
                self._reversal_ctr.signals_detected += 1
                self._emit_indicator_snapshot(effective_regime, "reversal", "enter", 0.7)
                return sig, EngineTag.REVERSAL

        # ── 2. Fade (15m VWAP rejection) ──────────────────────────────
        if flags.fade_engine and self._bars_15m and len(self._bars_15m["close"]) > 0:
            bars_15m = self._bars_15m
            n15 = len(bars_15m["close"])
            close_15m = bars_15m["close"][-1]
            lookback = int(po.get("rejection_lookback_bars", 8))
            start_lb = max(0, n15 - lookback)
            high_recent = bars_15m["high"][start_lb:]

            session_type = self._classify_session()

            sig = detect_fade_short(
                self._fade,
                close_15m,
                high_recent,
                effective_regime,
                self._mom15_slope_ok,
                self._regime.extension_short,
                self._atr_15m,
                session_type,
                flags,
                po,
            )
            if sig is not None:
                self._fade_ctr.signals_detected += 1
                self._momentum_impulse_pending = False
                self._emit_indicator_snapshot(effective_regime, "fade", "enter", sig.class_mult)
                return sig, EngineTag.FADE

        # ── 3. Momentum impulse (alternative fade entry) ──────────────
        if flags.momentum_signal and self._bars_15m and len(self._bars_15m["close"]) > 5:
            cooldown = int(po.get("momentum_cooldown_bars", 36))
            if self._bars_since_last_entry >= cooldown:
                bars_15m = self._bars_15m
                close_15m = bars_15m["close"][-1]
                close_5ago = bars_15m["close"][-6]
                roc_5bar = (close_15m - close_5ago) / close_5ago if close_5ago > 0 else 0.0
                ema_fast_15m = self._inc_ema_15m_fast.value

                if detect_momentum_impulse(
                    close_15m, ema_fast_15m, roc_5bar, effective_regime, po,
                ):
                    sig = FadeSignal(
                        vwap_used=self._fade.vwap_used,
                        rejection_close=close_15m,
                        class_mult=0.70,
                        predator_present=False,
                    )
                    self._momentum_impulse_pending = True
                    self._fade_ctr.signals_detected += 1
                    self._emit_indicator_snapshot(effective_regime, "momentum_impulse", "enter", 0.5)
                    return sig, EngineTag.FADE

        self._emit_indicator_snapshot(effective_regime)
        return None

    # ── Entry Submission ──────────────────────────────────────────────

    async def _submit_entry(self, signal: Any, tag: EngineTag) -> None:
        """Build and submit a short entry order via OMS."""
        inst = self._instruments.get(self._symbol)
        if inst is None:
            logger.warning("No instrument for %s", self._symbol)
            return

        po = self._po
        close = float(self._bars_5m["close"][-1]) if self._bars_5m else 0.0
        if close <= 0:
            return

        # Determine signal class (matches backtest naming)
        if self._momentum_impulse_pending:
            sig_class = "momentum_impulse"
        elif tag == EngineTag.REVERSAL:
            sig_class = "classic_divergence"
        else:
            sig_class = "vwap_rejection"

        # Entry price, initial stop, entry type
        atr = self._atr_1h if tag != EngineTag.BREAKDOWN else self._atr_30m
        low_recent = close - 2 * self._tick_size
        entry_price, stop0, entry_type = compute_entry_subtype_stop(
            tag, signal, close, atr, low_recent, self._tick_size, po,
        )
        if entry_price <= 0 or stop0 <= 0:
            return

        # Position sizing
        risk_per_unit = abs(stop0 - entry_price) * self._point_value
        if risk_per_unit <= 0:
            return

        base_risk_pct = po.get("base_risk_pct", 0.01)
        r_mult = regime_sizing_mult(self._regime.composite_regime, po)
        vol_factor = self._regime.vol_factor if self._flags.use_volatility_states else 1.0
        strong_bonus = (
            1.25 if self._regime.strong_bear and self._flags.use_strong_bear_bonus else 1.0
        )

        risk_dollars = self._equity * base_risk_pct * r_mult * vol_factor * strong_bonus

        # Correction sizing bonus
        if self._in_correction_now and self._flags.correction_sizing_bonus:
            corr_mult = po.get("correction_sizing_mult", 1.30)
            risk_dollars *= corr_mult

        qty = max(1, int(risk_dollars / risk_per_unit))

        # Leverage cap (configurable, default was 20x)
        max_leverage = C.MAX_LEVERAGE_MULT
        notional_per = entry_price * self._point_value
        if notional_per > 0:
            max_qty = max(1, int(self._equity * max_leverage / notional_per))
            qty = min(qty, max_qty)

        # TP schedule
        tp_sched = compute_tiered_tp_schedule(tag, self._regime.composite_regime, po)

        if entry_type == "stop_market":
            neutral_order_type = "STOP"
            oms_order_type = OrderType.STOP
            limit_price = None
        else:
            neutral_order_type = "STOP_LIMIT"
            oms_order_type = OrderType.STOP_LIMIT
            limit_price = entry_price - 4 * self._tick_size
        signal_context = self._entry_signal_context(
            tag=tag,
            signal_class=sig_class,
            signal=signal,
        )

        entry_request = DownturnEntryRequest(
            client_order_id=f"{C.STRATEGY_ID}:{tag.value}:{self._bar_count_5m}:{len(self._working_entries)}",
            symbol=self._symbol,
            engine_tag=tag,
            signal_class=sig_class,
            qty=qty,
            entry_price=entry_price,
            stop0=stop0,
            order_type=neutral_order_type,
            price=entry_price if neutral_order_type == "STOP_LIMIT" else None,
            limit_price=limit_price,
            stop_price=entry_price,
            submitted_bar_idx=self._bar_count_5m,
            ttl_bars=72,
            composite_regime=self._regime.composite_regime,
            vol_state=self._regime.vol_state,
            in_correction=self._in_correction_now,
            predator=getattr(signal, "predator_present", False),
            tp_schedule=tp_sched,
            signal_strength=getattr(signal, "class_mult", 0.5),
        )
        core_state, actions, events = downturn_core_logic.on_bar(
            self._build_core_state(),
            bar_count_5m=self._bar_count_5m,
            bar_ts=self._last_bar_ts or datetime.now(timezone.utc),
            entry_request=entry_request,
        )
        self._apply_core_state(core_state)
        self._apply_core_events(events)
        submit_action = next((action for action in actions if isinstance(action, SubmitEntry)), None)
        if submit_action is None:
            return

        risk_ctx = RiskContext(
            stop_for_risk=stop0,
            planned_entry_price=entry_price,
            risk_dollars=RiskCalculator.compute_order_risk_dollars(
                entry_price, stop0, qty, self._point_value,
            ),
            **signal_context,
        )

        order = OMSOrder(
            strategy_id=C.STRATEGY_ID,
            instrument=inst,
            side=OrderSide.SELL,
            qty=submit_action.qty,
            order_type=oms_order_type,
            limit_price=submit_action.limit_price,
            stop_price=submit_action.stop_price,
            tif=submit_action.tif,
            role=OrderRole.ENTRY,
            risk_context=risk_ctx,
        )

        receipt = await self._oms.submit_intent(Intent(
            intent_type=IntentType.NEW_ORDER,
            strategy_id=C.STRATEGY_ID,
            order=order,
        ))

        if receipt and receipt.oms_order_id:
            core_state, _, events = downturn_core_logic.on_order_update(
                self._build_core_state(),
                DownturnOrderUpdate(
                    oms_order_id=receipt.oms_order_id,
                    status="accepted",
                    timestamp=datetime.now(timezone.utc),
                    order_role="entry",
                    accepted_entry=entry_request,
                ),
            )
            self._apply_core_state(core_state)
            self._apply_core_events(events)
            logger.info(
                "Entry submitted: %s/%s @ %.2f stop=%.2f qty=%d",
                tag.value, sig_class, entry_price, stop0, qty,
            )

    # ── Order Helpers ─────────────────────────────────────────────────

    def _entry_signal_context(
        self,
        *,
        tag: EngineTag,
        signal_class: str,
        signal: Any = None,
        bar_ts: datetime | None = None,
    ) -> dict[str, Any]:
        ts = (
            bar_ts
            or getattr(signal, "timestamp", None)
            or self._last_bar_ts
            or datetime.now(timezone.utc)
        )
        ts_text = ts.isoformat()
        return {
            "signal_id": f"{self._symbol}:{tag.value}:{signal_class}:{ts_text}",
            "bar_id": f"{self._symbol}:5m:{ts_text}",
            "exchange_timestamp": ts,
        }

    async def _place_protective_stop(self, stop_price: float, qty: int) -> None:
        """Place protective BUY stop order (covers short position)."""
        inst = self._instruments.get(self._symbol)
        if inst is None:
            return
        order = OMSOrder(
            strategy_id=C.STRATEGY_ID,
            instrument=inst,
            side=OrderSide.BUY,
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
        if receipt and receipt.oms_order_id and self._position:
            core_state, _, events = downturn_core_logic.on_order_update(
                self._build_core_state(),
                DownturnOrderUpdate(
                    oms_order_id=receipt.oms_order_id,
                    status="accepted",
                    timestamp=datetime.now(timezone.utc),
                    order_role="stop",
                ),
            )
            self._apply_core_state(core_state)
            self._apply_core_events(events)

    async def _update_stop(self, new_stop: float, *, trigger: str = "trailing") -> None:
        """Update protective stop via REPLACE_ORDER intent."""
        pos = self._position
        if pos is None:
            return
        old_stop = pos.chandelier_stop
        # Instrumentation: log stop adjustment for TA pipeline
        if self._kit and self._kit.active and old_stop != new_stop:
            try:
                self._kit.log_stop_adjustment(
                    trade_id=pos.trade_id or f"DT_{self._symbol}",
                    symbol=self._symbol,
                    old_stop=old_stop,
                    new_stop=new_stop,
                    adjustment_type="trailing",
                    trigger=trigger,
                    metadata={
                        "r_at_peak": round(pos.r_at_peak, 3),
                        "engine_tag": pos.engine_tag.value,
                        "be_triggered": pos.be_triggered,
                        "hold_bars_5m": pos.hold_bars_5m,
                    },
                )
            except Exception:
                pass
        if pos.stop_oms_order_id:
            await self._oms.submit_intent(Intent(
                intent_type=IntentType.REPLACE_ORDER,
                strategy_id=C.STRATEGY_ID,
                target_oms_order_id=pos.stop_oms_order_id,
                new_stop_price=new_stop,
            ))

    async def _flatten(self, reason: str = "market_exit") -> None:
        """Flatten entire position via market order."""
        receipt = await self._oms.submit_intent(Intent(
            intent_type=IntentType.FLATTEN,
            strategy_id=C.STRATEGY_ID,
            instrument_symbol=self._symbol,
        ))
        if reason == "ROLL_SAFETY":
            self._roll_flatten_oms_id = str(getattr(receipt, "oms_order_id", "") or "")
            self._roll_flatten_pending = bool(self._roll_flatten_oms_id)
        logger.info("Flatten submitted: reason=%s", reason)

    async def _cancel_order(self, oms_order_id: str) -> None:
        """Cancel a working order."""
        try:
            await self._oms.submit_intent(Intent(
                intent_type=IntentType.CANCEL_ORDER,
                strategy_id=C.STRATEGY_ID,
                target_oms_order_id=oms_order_id,
            ))
        except Exception:
            logger.debug("Cancel failed for %s", oms_order_id)

    # ── Position Management ───────────────────────────────────────────

    async def _manage_position(self) -> None:  # noqa: C901
        """Stop management cascade for open short position."""
        pos = self._position
        if pos is None:
            return

        flags = self._flags
        po = self._po

        # Current price (latest 5m close)
        close = float(self._bars_5m["close"][-1]) if self._bars_5m else 0.0
        if close <= 0:
            return

        # Update MFE/MAE (short: lower price = more profit)
        if close < pos.mfe_price or pos.mfe_price == 0.0:
            pos.mfe_price = close
        if close > pos.mae_price:
            pos.mae_price = close

        r = pos.r_state(close)
        pos.r_at_peak = max(pos.r_at_peak, r)

        # ── 1. Min hold period ────────────────────────────────────────
        if flags.min_hold_period:
            min_bars = int(po.get("min_hold_bars", 6))
            if pos.hold_bars_5m < min_bars:
                if check_catastrophic_exit(r):
                    await self._flatten("catastrophic")
                return

        # ── 2. Multi-tier profit floor ────────────────────────────────
        if flags.multi_tier_profit_floor:
            mt_stop = compute_multi_tier_profit_floor(
                pos.entry_price, pos.r_at_peak, pos.risk_per_unit,
                self._tick_size, po,
            )
            if mt_stop is not None and mt_stop < pos.chandelier_stop:
                pos.chandelier_stop = mt_stop
                pos.exit_trigger = "profit_floor_multi"
                await self._update_stop(mt_stop, trigger="profit_floor_multi")

        # ── 3. Single-tier profit floor with adaptive lock ────────────
        elif flags.profit_floor_trail:
            if flags.adaptive_profit_floor:
                base_lock = po.get("profit_floor_lock_pct", 0.40)
                adapted_lock = compute_adaptive_lock_pct(pos.r_at_peak, base_lock, po)
                po_adapted = {**po, "profit_floor_lock_pct": adapted_lock}
            else:
                po_adapted = po
            pf_stop = compute_profit_floor_stop(
                pos.entry_price, r, pos.risk_per_unit, self._tick_size, po_adapted,
            )
            if pf_stop is not None and pf_stop < pos.chandelier_stop:
                pos.chandelier_stop = pf_stop
                pos.exit_trigger = "profit_floor"
                await self._update_stop(pf_stop, trigger="profit_floor")

        # ── 4. Breakeven stop ─────────────────────────────────────────
        be_trigger_r = po.get("be_trigger_r", 1.0)
        if not pos.be_triggered and r >= be_trigger_r:
            be_stop = compute_breakeven_stop(
                pos.entry_price, self._atr_1h, self._tick_size, po,
            )
            if be_stop < pos.chandelier_stop:
                pos.chandelier_stop = be_stop
                await self._update_stop(be_stop, trigger="breakeven")
            pos.be_triggered = True

        # ── 5. Chandelier trailing stop ───────────────────────────────
        if flags.chandelier_trailing and self._atr_1h > 0:
            bars_1h = self._bars_1h
            n1h = len(bars_1h.get("low", []))
            lookback = int(po.get("chandelier_lookback", 14))
            if n1h > lookback:
                start = max(0, n1h - lookback)
                ll = float(np.min(bars_1h["low"][start:]))

                regime_mult = None
                if flags.regime_adaptive_chandelier:
                    regime_mult = compute_chandelier_regime_mult(
                        self._regime.composite_regime, po,
                    )

                new_stop = update_chandelier_trail(
                    ll, self._atr_1h, r, self._regime.strong_bear,
                    pos.chandelier_stop, self._tick_size, po,
                    tp1_hit=(pos.tp_idx > 0),
                    regime_mult=regime_mult,
                )
                if new_stop < pos.chandelier_stop:
                    pos.chandelier_stop = new_stop
                    pos.exit_trigger = "chandelier"
                    await self._update_stop(new_stop, trigger="chandelier")

        # ── 6. Stale exit ─────────────────────────────────────────────
        if flags.stale_exit:
            if pos.engine_tag == EngineTag.REVERSAL:
                bars_held = pos.hold_bars_4h
            elif pos.engine_tag == EngineTag.BREAKDOWN:
                bars_held = pos.hold_bars_30m
            else:
                bars_held = pos.hold_bars_1h
            if check_stale_exit(pos.engine_tag, bars_held, r, po):
                await self._flatten("stale")
                return

        # ── 7. Climax exit ────────────────────────────────────────────
        if flags.climax_exit:
            if check_climax_exit(close, self._ema20_1h, self._atr_1h, r, po):
                await self._flatten("climax")
                return

        # ── 8. VWAP failure exit (Fade only) ──────────────────────────
        if flags.vwap_failure_exit and pos.engine_tag == EngineTag.FADE:
            if check_vwap_failure_exit(self._fade.consecutive_above_vwap, r):
                await self._flatten("vwap_failure")
                return

        # ── 9. Catastrophic exit ──────────────────────────────────────
        if check_catastrophic_exit(r):
            await self._flatten("catastrophic")

    # ── Event Processing ──────────────────────────────────────────────

    async def _process_events(self) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=5.0)
                await self._handle_event(event)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error processing event")

    async def _handle_event(self, event: Any) -> None:
        etype = getattr(event, "event_type", None)
        if etype == OMSEventType.FILL:
            await self._on_fill(event)
        elif etype in (
            OMSEventType.ORDER_CANCELLED,
            OMSEventType.ORDER_EXPIRED,
            OMSEventType.ORDER_REJECTED,
        ):
            self._on_order_terminal(event)

    async def _on_fill(self, event: Any) -> None:
        oms_order_id = getattr(event, "oms_order_id", "")
        payload = getattr(event, "payload", {})
        fill_price = payload.get("price", 0.0)
        fill_qty = payload.get("qty", 0)
        fill_commission = float(payload.get("commission", 0.0) or 0.0)
        fill_time = getattr(event, "timestamp", None) or datetime.now(timezone.utc)

        # Check if this is an entry fill
        for we in list(self._working_entries):
            if we.oms_order_id == oms_order_id:
                await self._on_entry_fill(we, fill_price, fill_qty, fill_commission, fill_time, payload)
                return

        # Check if this is an exit fill (protective stop or flatten)
        if self._position is not None:
            if oms_order_id == self._position.stop_oms_order_id:
                await self._on_exit_fill(fill_price, fill_qty, "stop", fill_commission, fill_time, oms_order_id, payload)
            else:
                # Flatten or other exit — treat as market exit
                await self._on_exit_fill(fill_price, fill_qty, "market_exit", fill_commission, fill_time, oms_order_id, payload)

    async def _on_entry_fill(
        self, we: WorkingEntry, fill_price: float, fill_qty: int,
        fill_commission: float = 0.0, fill_time: datetime | None = None,
        payload: dict | None = None,
    ) -> None:
        """Create ActivePosition from filled entry order."""
        core_state, actions, events = downturn_core_logic.on_fill(
            self._build_core_state(),
            DownturnFill(
                oms_order_id=we.oms_order_id,
                fill_price=fill_price,
                fill_qty=fill_qty or we.qty,
                commission=fill_commission,
                fill_time=fill_time or datetime.now(timezone.utc),
            ),
        )
        self._apply_core_state(core_state)
        self._apply_core_events(events)
        if self._position is None:
            logger.error("Shared core did not open downturn position for filled entry %s", we.oms_order_id)
            self._record_decision(
                "CORE_ENTRY_FILL_ERROR",
                {"oms_order_id": we.oms_order_id, "fill_price": fill_price, "qty": fill_qty or we.qty},
            )
            try:
                await self._flatten("CORE_ENTRY_FILL_ERROR")
            except Exception:
                logger.exception("Failed emergency flatten after downturn core entry-fill mismatch")
            return
        qty = self._position.qty if self._position is not None else (fill_qty or we.qty)

        for action in actions:
            if isinstance(action, SubmitExit):
                await self._place_protective_stop(action.stop_price or we.stop0, action.qty)

        logger.info(
            "Entry filled: %s/%s @ %.2f qty=%d stop0=%.2f",
            we.engine_tag.value, we.signal_class, fill_price, qty, we.stop0,
        )
        self._persist_state()

        # Instrumentation: log_entry + orderbook context
        if self._kit and self._kit.active:
            try:
                pos = self._position
                self._kit.log_entry(
                    trade_id=pos.trade_id,
                    pair=self._symbol,
                    side="SHORT",
                    entry_price=fill_price,
                    position_size=qty,
                    position_size_quote=qty * fill_price * self._point_value,
                    entry_signal=f"{we.engine_tag.value}_{we.signal_class}",
                    entry_signal_id=pos.trade_id,
                    entry_signal_strength=we.signal_strength,
                    expected_entry_price=we.entry_price,
                    strategy_params={
                        "engine_tag": we.engine_tag.value,
                        "signal_class": we.signal_class,
                        "stop0": we.stop0,
                        "composite_regime": we.composite_regime.value,
                        "vol_state": we.vol_state.value,
                        "in_correction": we.in_correction,
                        "predator": we.predator,
                    },
                    signal_factors=[
                        {"factor_name": "regime", "factor_value": we.composite_regime.value,
                         "threshold": "emerging_bear", "contribution": 0.4},
                        {"factor_name": "engine_tag", "factor_value": we.engine_tag.value,
                         "threshold": "any", "contribution": 0.3},
                        {"factor_name": "in_correction", "factor_value": int(we.in_correction),
                         "threshold": 1, "contribution": 0.15 if we.in_correction else 0.0},
                        {"factor_name": "predator", "factor_value": int(we.predator),
                         "threshold": 1, "contribution": 0.15 if we.predator else 0.0},
                    ],
                    sizing_inputs={
                        "equity": self._equity,
                        "base_risk_pct": self._po.get("base_risk_pct", 0.01),
                        "regime_mult": regime_sizing_mult(we.composite_regime, self._po),
                        "qty": qty,
                        "risk_per_unit": round(abs(we.stop0 - fill_price) * self._point_value, 2),
                    },
                    portfolio_state={
                        "equity": self._equity,
                        "daily_loss": self._daily_loss,
                        "daily_trades": self._daily_trades,
                        "circuit_breaker_tripped": self._circuit_breaker_tripped,
                        "has_open_position": self._position is not None,
                        "composite_regime": self._regime.composite_regime.value if self._regime else None,
                        "vol_state": self._regime.vol_state.value if self._regime else None,
                    },
                    filter_decisions=self._entry_gate_decisions(),
                    session_type=self._classify_session(),
                    concurrent_positions=1 if self._position else 0,
                    drawdown_pct=round(abs(self._daily_loss / self._equity) * 100, 2) if self._equity > 0 else 0,
                    drawdown_tier=self._dd_tier_name(),
                    signal_evolution=self._build_signal_evolution(),
                    execution_timestamps={"fill_time": datetime.now(timezone.utc).isoformat()},
                    **fill_runtime_refs(we.oms_order_id, payload, fill_qty=qty),
                )

                ba = self._get_bid_ask()
                if ba:
                    self._kit.on_orderbook_context(
                        pair=self._symbol,
                        best_bid=ba[0], best_ask=ba[1],
                        trade_context="entry",
                        related_trade_id=pos.trade_id,
                        exchange_timestamp=datetime.now(timezone.utc),
                    )
            except Exception:
                pass

    async def _on_exit_fill(
        self, fill_price: float, fill_qty: int, exit_type: str,
        fill_commission: float = 0.0, fill_time: datetime | None = None,
        oms_order_id: str = "",
        payload: dict | None = None,
    ) -> None:
        """Process exit fill: record trade, update risk, clear position."""
        pos = self._position
        if pos is None:
            return

        pos.commission += fill_commission

        # Substitute exit_trigger (profit_floor, chandelier, etc.) for generic "stop"
        if exit_type == "stop" and pos.exit_trigger:
            exit_type = pos.exit_trigger

        # PnL calculation (short: entry - exit = profit), net of fees
        close_qty = fill_qty if fill_qty > 0 else pos.qty
        pnl = (pos.entry_price - fill_price) * close_qty * self._point_value - pos.commission
        r_mult = pos.r_state(fill_price)  # gross R (matches backtest pattern)

        # Daily risk tracking
        self._daily_loss += min(0, pnl)
        self._daily_trades += 1

        # Circuit breaker
        if self._flags.daily_circuit_breaker:
            cb_threshold = self._po.get("circuit_breaker_threshold", -3000.0)
            if self._daily_loss <= cb_threshold:
                self._circuit_breaker_tripped = True
                logger.warning("Circuit breaker tripped: daily_loss=%.2f", self._daily_loss)

        logger.info(
            "Exit filled: %s @ %.2f, PnL=%.2f (%.2fR), exit=%s",
            pos.engine_tag.value, fill_price, pnl, r_mult, exit_type,
        )

        # Instrumentation: log_exit + orderbook context
        if self._kit and self._kit.active:
            try:
                rpu = pos.risk_per_unit
                mfe_r = (pos.entry_price - pos.mfe_price) / rpu if rpu > 0 else 0.0
                mae_r = -(pos.mae_price - pos.entry_price) / rpu if rpu > 0 else 0.0

                self._kit.log_exit(
                    trade_id=pos.trade_id,
                    exit_price=fill_price,
                    exit_reason=exit_type,
                    mfe_r=round(mfe_r, 3),
                    mae_r=round(mae_r, 3),
                    mfe_price=pos.mfe_price,
                    mae_price=pos.mae_price,
                    session_transitions=None,
                    **fill_runtime_refs(oms_order_id, payload, fill_qty=close_qty, is_exit=True),
                )

                ba = self._get_bid_ask()
                if ba:
                    self._kit.on_orderbook_context(
                        pair=self._symbol,
                        best_bid=ba[0], best_ask=ba[1],
                        trade_context="exit",
                        related_trade_id=pos.trade_id,
                        exchange_timestamp=datetime.now(timezone.utc),
                    )
            except Exception:
                pass

        # Trade recorder
        if self._trade_recorder:
            try:
                await self._trade_recorder.record({
                    "strategy_id": C.STRATEGY_ID,
                    "symbol": self._symbol,
                    "direction": "SHORT",
                    "entry_price": pos.entry_price,
                    "exit_price": fill_price,
                    "shares": pos.qty,
                    "realized_pnl": pnl,
                    "r_multiple": r_mult,
                    "entry_type": pos.signal_class,
                    "exit_reason": exit_type,
                    "entry_ts": pos.entry_time,
                    "exit_ts": datetime.now(timezone.utc),
                    "trade_id": pos.trade_id,
                    "hold_bars_5m": pos.hold_bars_5m,
                    "engine_tag": pos.engine_tag.value,
                    "regime": pos.composite_regime.value,
                    "commission": pos.commission,
                })
            except Exception:
                logger.warning("Trade recorder failed", exc_info=True)

        core_state, _, events = downturn_core_logic.on_fill(
            self._build_core_state(),
            DownturnFill(
                oms_order_id=pos.stop_oms_order_id or pos.entry_oms_order_id,
                fill_price=fill_price,
                fill_qty=close_qty,
                commission=fill_commission,
                fill_time=fill_time or datetime.now(timezone.utc),
                exit_type=exit_type,
            ),
        )
        self._apply_core_state(core_state)
        self._apply_core_events(events)
        self._roll_flatten_pending = False
        self._roll_flatten_oms_id = ""
        self._momentum_impulse_pending = False
        self._persist_state()

    def _on_order_terminal(self, event: Any) -> None:
        """Handle cancelled/expired/rejected orders."""
        oms_order_id = getattr(event, "oms_order_id", "")
        is_roll_flatten_terminal = bool(
            self._roll_flatten_oms_id and oms_order_id == self._roll_flatten_oms_id
        )
        core_state, _, events = downturn_core_logic.on_order_update(
            self._build_core_state(),
            DownturnOrderUpdate(
                oms_order_id=oms_order_id,
                status=str(getattr(getattr(event, "event_type", None), "value", "terminal")).lower(),
                timestamp=datetime.now(timezone.utc),
            ),
        )
        self._apply_core_state(core_state)
        self._apply_core_events(events)
        if is_roll_flatten_terminal:
            self._roll_flatten_pending = False
            self._roll_flatten_oms_id = ""
            if self._position is not None:
                self._record_decision(
                    "ROLL_FORCE_FLATTEN_RETRY",
                    {"oms_order_id": oms_order_id},
                )

    async def _force_flatten_for_roll(self, now: datetime | None) -> bool:
        if self._position is None:
            self._roll_flatten_pending = False
            self._roll_flatten_oms_id = ""
            return False
        reason = roll_force_flatten_reason(
            self._instruments.get(self._symbol) or self._symbol,
            as_of=now or datetime.now(timezone.utc),
        )
        if not reason:
            return False
        if self._roll_flatten_pending:
            self._record_decision(
                "ROLL_FORCE_FLATTEN_PENDING",
                {"reason": reason, "oms_order_id": self._roll_flatten_oms_id},
            )
            return True
        self._roll_flatten_pending = True
        self._record_decision("ROLL_FORCE_FLATTEN", {"reason": reason})
        logger.critical("Downturn forcing flatten for roll safety: %s", reason)
        try:
            await self._flatten("ROLL_SAFETY")
        except Exception:
            self._roll_flatten_pending = False
            self._roll_flatten_oms_id = ""
            self._record_decision("ROLL_FORCE_FLATTEN_FAILED", {"reason": reason})
            logger.exception("Downturn roll-safety flatten submission failed")
        return True

    def _expire_working_entries(self) -> None:
        """Cancel entries that exceeded their TTL."""
        reason = roll_blackout_reason(
            self._instruments.get(self._symbol) or self._symbol,
            as_of=self._last_bar_ts or datetime.now(timezone.utc),
        )
        if reason and self._working_entries:
            entries = list(self._working_entries)
            for we in entries:
                asyncio.create_task(self._cancel_order(we.oms_order_id))
                core_state, _, events = downturn_core_logic.on_order_update(
                    self._build_core_state(),
                    DownturnOrderUpdate(
                        oms_order_id=we.oms_order_id,
                        status="cancelled",
                        timestamp=datetime.now(timezone.utc),
                        order_role="entry",
                    ),
                )
                self._apply_core_state(core_state)
                self._apply_core_events(events)
                self._log_missed(
                    signal_name=f"{we.engine_tag.value}_{we.signal_class}",
                    signal_id=we.oms_order_id,
                    signal_strength=we.signal_strength,
                    blocked_by="roll_blackout",
                    block_reason=reason,
                    filter_decisions=self._entry_gate_decisions(),
                    entry_price=we.entry_price,
                    stop0=we.stop0,
                )
            self._record_decision("ENTRY_CANCELLED_BY_ROLL_BLACKOUT", {"reason": reason, "count": len(entries)})
            logger.warning("Cancelled %d Downturn working entries during roll blackout: %s", len(entries), reason)
            return

        entries_by_id = {we.oms_order_id: we for we in self._working_entries}
        core_state, actions, events = downturn_core_logic.on_bar(
            self._build_core_state(),
            bar_count_5m=self._bar_count_5m,
            bar_ts=self._last_bar_ts or datetime.now(timezone.utc),
            expire_entries=True,
        )
        self._apply_core_state(core_state)
        self._apply_core_events(events)
        expired_ids = [
            action.target_order_id
            for action in actions
            if isinstance(action, CancelAction) and action.reason == "ttl_expiry"
        ]
        for oms_order_id in expired_ids:
            we = entries_by_id.get(oms_order_id)
            if we is None:
                continue
            asyncio.create_task(self._cancel_order(oms_order_id))
            self._log_missed(
                signal_name=f"{we.engine_tag.value}_{we.signal_class}",
                signal_id=oms_order_id,
                signal_strength=we.signal_strength,
                blocked_by="ttl_expiry",
                block_reason=f"Working entry expired after {we.ttl_bars} bars without fill",
                filter_decisions=self._entry_gate_decisions(),
                entry_price=we.entry_price,
                stop0=we.stop0,
            )

    # ── State Persistence ─────────────────────────────────────────────

    def _persist_state(self) -> None:
        path = self._state_dir / "downturn_state.json"
        try:
            state: dict[str, Any] = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "equity": self._equity,
                "bar_count_5m": self._bar_count_5m,
                "bars_since_last_entry": self._bars_since_last_entry,
                "daily_risk": {
                    "daily_loss": self._daily_loss,
                    "daily_trades": self._daily_trades,
                    "circuit_breaker": self._circuit_breaker_tripped,
                    "last_reset_date": self._last_risk_reset_date,
                },
                "regime": {
                    "composite": self._regime.composite_regime.value,
                    "regime_4h": self._regime.regime_4h.value,
                    "daily_trend": self._regime.daily_trend,
                    "daily_trend_consec": self._regime.daily_trend_consec,
                    "strong_bear": self._regime.strong_bear,
                    "short_trend": self._regime.short_trend,
                    "vol_state": self._regime.vol_state.value,
                    "vol_factor": self._regime.vol_factor,
                    "trend_strength": self._regime.trend_strength,
                },
                "indicators": {
                    "atr_d": self._atr_d,
                    "atr_d_baseline": self._atr_d_baseline,
                    "atr_30m": self._atr_30m,
                    "atr_15m": self._atr_15m,
                    "atr_1h": self._atr_1h,
                    "atr_4h": self._atr_4h,
                    "ema_fast_d": self._ema_fast_d,
                    "ema_slow_d": self._ema_slow_d,
                    "sma200_d": self._sma200_d,
                    "ema20_1h": self._ema20_1h,
                    "atr_d_history": self._atr_d_history[-60:],
                },
                "state_flags": {
                    "fast_crash_active": self._fast_crash_active,
                    "bear_structure_active": self._bear_structure_active,
                    "drawdown_override_active": self._drawdown_override_active,
                    "in_correction_now": self._in_correction_now,
                    "regime_on": self._regime_on,
                    "bear_conviction": self._bear_conviction,
                },
                "position_open": self._position is not None,
            }

            if self._position is not None:
                pos = self._position
                state["position"] = {
                    "engine_tag": pos.engine_tag.value,
                    "signal_class": pos.signal_class,
                    "trade_id": pos.trade_id,
                    "entry_price": pos.entry_price,
                    "stop0": pos.stop0,
                    "qty": pos.qty,
                    "remaining_qty": pos.remaining_qty,
                    "entry_oms_order_id": pos.entry_oms_order_id,
                    "stop_oms_order_id": pos.stop_oms_order_id,
                    "entry_time": pos.entry_time.isoformat() if pos.entry_time else None,
                    "hold_bars_5m": pos.hold_bars_5m,
                    "hold_bars_1h": pos.hold_bars_1h,
                    "hold_bars_30m": pos.hold_bars_30m,
                    "hold_bars_4h": pos.hold_bars_4h,
                    "mfe_price": pos.mfe_price,
                    "mae_price": pos.mae_price,
                    "r_at_peak": pos.r_at_peak,
                    "chandelier_stop": pos.chandelier_stop,
                    "be_triggered": pos.be_triggered,
                    "tp_idx": pos.tp_idx,
                    "composite_regime": pos.composite_regime.value,
                    "vol_state": pos.vol_state.value,
                    "in_correction": pos.in_correction,
                    "commission": pos.commission,
                }

            if self._working_entries:
                state["working_entries"] = [
                    {
                        "oms_order_id": we.oms_order_id,
                        "engine_tag": we.engine_tag.value,
                        "signal_class": we.signal_class,
                        "entry_price": we.entry_price,
                        "stop0": we.stop0,
                        "qty": we.qty,
                        "submitted_bar_idx": we.submitted_bar_idx,
                        "ttl_bars": we.ttl_bars,
                    }
                    for we in self._working_entries
                ]

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state, indent=2))
        except Exception:
            logger.debug("State persist failed", exc_info=True)

    def _restore_state(self) -> None:
        path = self._state_dir / "downturn_state.json"
        if not path.exists():
            return
        try:
            state = json.loads(path.read_text())
            self._equity = state.get("equity", self._equity)
            self._bar_count_5m = state.get("bar_count_5m", 0)
            self._bars_since_last_entry = state.get("bars_since_last_entry", 999)

            dr = state.get("daily_risk", {})
            self._daily_loss = dr.get("daily_loss", 0.0)
            self._daily_trades = dr.get("daily_trades", 0)
            self._circuit_breaker_tripped = dr.get("circuit_breaker", False)
            self._last_risk_reset_date = dr.get("last_reset_date", "")

            reg = state.get("regime", {})
            try:
                self._regime.composite_regime = CompositeRegime(reg.get("composite", "neutral"))
            except ValueError:
                pass
            self._regime.daily_trend = reg.get("daily_trend", 0)
            self._regime.daily_trend_consec = reg.get("daily_trend_consec", 0)
            self._regime.strong_bear = reg.get("strong_bear", False)
            self._regime.short_trend = reg.get("short_trend", 0)
            self._regime.trend_strength = reg.get("trend_strength", 0.0)

            ind = state.get("indicators", {})
            self._atr_d = ind.get("atr_d", 0.0)
            self._atr_d_baseline = ind.get("atr_d_baseline", 0.0)
            self._atr_1h = ind.get("atr_1h", 0.0)
            self._ema_fast_d = ind.get("ema_fast_d", 0.0)
            self._ema_slow_d = ind.get("ema_slow_d", 0.0)
            self._sma200_d = ind.get("sma200_d", 0.0)
            self._ema20_1h = ind.get("ema20_1h", 0.0)
            self._atr_d_history = ind.get("atr_d_history", [])

            sf = state.get("state_flags", {})
            self._fast_crash_active = sf.get("fast_crash_active", False)
            self._bear_structure_active = sf.get("bear_structure_active", False)
            self._drawdown_override_active = sf.get("drawdown_override_active", False)
            self._in_correction_now = sf.get("in_correction_now", False)
            self._regime_on = sf.get("regime_on", False)
            self._bear_conviction = sf.get("bear_conviction", 0.0)

            # Restore position
            if state.get("position_open") and "position" in state:
                p = state["position"]
                entry_time = None
                if p.get("entry_time"):
                    try:
                        entry_time = datetime.fromisoformat(p["entry_time"])
                    except Exception:
                        pass
                self._position = ActivePosition(
                    engine_tag=EngineTag(p.get("engine_tag", "fade")),
                    signal_class=p.get("signal_class", ""),
                    trade_id=p.get("trade_id", ""),
                    entry_price=p.get("entry_price", 0.0),
                    stop0=p.get("stop0", 0.0),
                    qty=p.get("qty", 0),
                    remaining_qty=p.get("remaining_qty", 0),
                    entry_oms_order_id=p.get("entry_oms_order_id", ""),
                    stop_oms_order_id=p.get("stop_oms_order_id", ""),
                    entry_time=entry_time,
                    hold_bars_5m=p.get("hold_bars_5m", 0),
                    hold_bars_1h=p.get("hold_bars_1h", 0),
                    hold_bars_30m=p.get("hold_bars_30m", 0),
                    hold_bars_4h=p.get("hold_bars_4h", 0),
                    mfe_price=p.get("mfe_price", 0.0),
                    mae_price=p.get("mae_price", 0.0),
                    r_at_peak=p.get("r_at_peak", 0.0),
                    chandelier_stop=p.get("chandelier_stop", 0.0),
                    be_triggered=p.get("be_triggered", False),
                    tp_idx=p.get("tp_idx", 0),
                    composite_regime=CompositeRegime(p.get("composite_regime", "neutral")),
                    vol_state=VolState(p.get("vol_state", "normal")),
                    in_correction=p.get("in_correction", False),
                    commission=p.get("commission", 0.0),
                )

            # Restore working entries
            for we_data in state.get("working_entries", []):
                self._working_entries.append(WorkingEntry(
                    oms_order_id=we_data.get("oms_order_id", ""),
                    engine_tag=EngineTag(we_data.get("engine_tag", "fade")),
                    signal_class=we_data.get("signal_class", ""),
                    entry_price=we_data.get("entry_price", 0.0),
                    stop0=we_data.get("stop0", 0.0),
                    qty=we_data.get("qty", 0),
                    submitted_bar_idx=we_data.get("submitted_bar_idx", 0),
                    ttl_bars=we_data.get("ttl_bars", 72),
                ))

            logger.info("State restored: bar_count=%d, position=%s",
                        self._bar_count_5m, self._position is not None)
        except Exception:
            logger.warning("State restore failed", exc_info=True)

    # ── Helpers ───────────────────────────────────────────────────────

    def _classify_session(self) -> str:
        """Return 'core' (RTH) or 'extended' (ETH) based on current NY time.

        Matches backtest _classify_session: core = 09:35-15:50 ET (mult 1.0),
        extended = ETH morning/evening (mult < 1.0).
        """
        now_ny = datetime.now(ET)
        mins = now_ny.hour * 60 + now_ny.minute
        if 575 <= mins < 950:
            return "core"
        return "extended"

    def _check_daily_risk_reset(self) -> None:
        """Reset daily risk counters at start of new trading day (4 AM ET)."""
        now_ny = datetime.now(ET)
        date_str = now_ny.strftime("%Y-%m-%d")
        if date_str != self._last_risk_reset_date and now_ny.hour >= 4:
            self._daily_loss = 0.0
            self._daily_trades = 0
            self._circuit_breaker_tripped = False
            self._last_risk_reset_date = date_str
            self._vwap_cross_count = 0

    async def _refresh_equity(self) -> None:
        try:
            accounts = self._ib.ib.managedAccounts()
            if accounts:
                for item in self._ib.ib.accountValues():
                    if item.tag == "NetLiquidation" and item.currency == "USD" and item.account == accounts[0]:
                        self._equity = float(item.value) * self._equity_alloc_pct
                        return
        except Exception:
            pass
