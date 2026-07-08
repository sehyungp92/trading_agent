"""BreakoutStrategy — Volume Profile Breakout perps unified strategy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone

import structlog

from crypto_trader.core.engine import StrategyContext
from crypto_trader.core.events import PositionClosedEvent
from crypto_trader.core.models import (
    Bar,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    SetupGrade,
    Side,
    TerminalMark,
    TimeFrame,
    Trade,
)
from crypto_trader.instrumentation.lineage import stable_hash
from crypto_trader.strategy.momentum.indicators import (
    IncrementalIndicators,
    IndicatorSnapshot,
)
from crypto_trader.strategy.momentum.journal import TradeJournal

from .balance import BalanceDetector, BalanceZone
from .config import BreakoutConfig
from .confirmation import ConfirmationDetector
from .context import ContextAnalyzer
from .entry import EntryGenerator
from .exits import BreakoutExitState, ExitManager
from .profile import VolumeProfiler, VolumeProfileResult
from .risk import RiskManager
from .setup import BreakoutDetector
from .sizing import PositionSizer
from .stops import StopPlacer
from .trail import TrailManager
from crypto_trader.instrumentation.collector import InstrumentationCollector
from crypto_trader.instrumentation.quality import ProcessQualityScorer
from crypto_trader.strategy.snapshot import dataclass_from_plain, to_plain

log = structlog.get_logger()

WARMUP_BARS = 101  # ema_slow=100 + 1


@dataclass
class _PositionMeta:
    """Per-position metadata tracked by the strategy."""
    setup_grade: SetupGrade | None = None
    is_a_plus: bool = False
    confluences: tuple[str, ...] = ()
    confirmation_type: str = ""
    entry_method: str = ""
    signal_variant: str = "core"
    entry_price: float = 0.0
    stop_level: float = 0.0
    stop_distance: float = 0.0
    leverage: float = 0.0
    liquidation_price: float = 0.0
    risk_pct: float = 0.0
    original_qty: float = 0.0
    entry_bar_index: int = 0
    stop_order_id: str | None = None
    balance_zone: BalanceZone | None = None
    h4_context_notes: str = ""


class BreakoutStrategy:
    """Volume Profile Breakout — M30 primary / H4 context.

    Detects consolidation around high-volume nodes, then trades the
    directional expansion when price breaks into low-volume space.
    """

    def __init__(self, config: BreakoutConfig | None = None, bot_id: str = "") -> None:
        self._cfg = config or BreakoutConfig()

        # Module instances
        self._profiler = VolumeProfiler(self._cfg.profile)
        self._balance_detector = BalanceDetector(self._cfg.balance)
        self._context_analyzer = ContextAnalyzer(self._cfg.context)
        self._breakout_detector = BreakoutDetector(self._cfg.setup, self._cfg.symbol_filter)
        self._confirmation_detector = ConfirmationDetector(self._cfg.confirmation)
        self._entry_generator = EntryGenerator(self._cfg.entry)
        self._stop_placer = StopPlacer(self._cfg.stops)
        self._sizer = PositionSizer(self._cfg.risk, self._cfg.limits)
        self._exit_manager = ExitManager(self._cfg.exits)
        self._trail_manager = TrailManager(self._cfg.trail)
        self._risk_manager = RiskManager(self._cfg.limits)
        self._journal = TradeJournal()

        # Instrumentation
        self._collector = InstrumentationCollector(strategy_id="breakout", bot_id=bot_id)
        self._quality_scorer = ProcessQualityScorer()

        # Per-symbol state
        self._position_meta: dict[str, _PositionMeta] = {}
        self._m30_bar_count: dict[str, int] = {}
        self._m30_indicators: dict[str, IndicatorSnapshot | None] = {}
        self._h4_indicators: dict[str, IndicatorSnapshot | None] = {}

        # Per-symbol incremental indicator instances
        self._m30_inc: dict[str, IncrementalIndicators] = {}
        self._h4_inc: dict[str, IncrementalIndicators] = {}

        # Volume profile state
        self._profile_bar_count: dict[str, int] = {}
        self._current_profile: dict[str, VolumeProfileResult | None] = {}

        # Re-entry tracking
        self._recent_exits: dict[str, dict] = {}
        self._reentry_count: dict[str, int] = {}
        self._blocked_relaxed_body_signals: list[dict[str, object]] = []

        self._ctx: StrategyContext | None = None

    @property
    def name(self) -> str:
        return "volume_profile_breakout"

    @property
    def symbols(self) -> list[str]:
        return self._cfg.symbols

    @property
    def timeframes(self) -> list[TimeFrame]:
        return [TimeFrame.M30, TimeFrame.H4]

    @property
    def journal(self) -> TradeJournal:
        return self._journal

    @staticmethod
    def _management_order_id(purpose: str, symbol: str, seed: dict[str, object]) -> str:
        order_seed = {
            "strategy": "breakout",
            "purpose": purpose,
            "symbol": symbol,
            **seed,
        }
        return f"brk_{purpose}_{symbol}_{stable_hash(order_seed, length=8)}"

    def _ensure_exit_order_id(
        self,
        order: Order,
        *,
        bar: Bar,
        state: BreakoutExitState | None,
        order_index: int,
    ) -> None:
        if order.order_id:
            return
        purpose = str(order.tag or order.order_type.value or "exit").lower()
        seed: dict[str, object] = {
            "bar_timestamp": bar.timestamp.isoformat(),
            "timeframe": bar.timeframe.value,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "qty": order.qty,
            "limit_price": order.limit_price,
            "stop_price": order.stop_price,
            "tag": order.tag,
            "order_index": order_index,
        }
        if state is not None:
            seed.update({
                "direction": state.direction.value,
                "entry_price": state.entry_price,
                "stop_distance": state.stop_distance,
                "remaining_qty": state.remaining_qty,
                "bars_since_entry": state.bars_since_entry,
                "current_r": state.current_r,
                "mfe_r": state.mfe_r,
                "mae_r": state.mae_r,
                "peak_r": state.peak_r,
                "tp1_hit": state.tp1_hit,
                "tp2_hit": state.tp2_hit,
                "be_moved": state.be_moved,
                "early_lock_applied": state.early_lock_applied,
            })
        order.order_id = self._management_order_id(purpose, order.symbol, seed)
        order.metadata.setdefault("client_order_id", order.order_id)

    def snapshot_state(self) -> dict:
        return {
            "position_meta": to_plain(self._position_meta),
            "exit_states": to_plain(getattr(self._exit_manager, "_states", {})),
            "trail_stops": to_plain(getattr(self._trail_manager, "_last_stops", {})),
            "recent_exits": to_plain(self._recent_exits),
            "reentry_count": to_plain(self._reentry_count),
            "m30_bar_count": deepcopy(self._m30_bar_count),
            "m30_indicators": deepcopy(self._m30_indicators),
            "h4_indicators": deepcopy(self._h4_indicators),
            "m30_incremental": deepcopy(self._m30_inc),
            "h4_incremental": deepcopy(self._h4_inc),
            "profile_bar_count": deepcopy(self._profile_bar_count),
            "current_profile": deepcopy(self._current_profile),
            "balance_zones": deepcopy(getattr(self._balance_detector, "_zones", {})),
            "pending_retests": deepcopy(getattr(self._confirmation_detector, "_pending", {})),
            "risk_manager": self._risk_manager.snapshot_state(),
            "blocked_relaxed_body_signals": deepcopy(self._blocked_relaxed_body_signals),
            "detector_blocked_relaxed_body_signals": deepcopy(
                getattr(self._breakout_detector, "_blocked_relaxed_body_signals", [])
            ),
            "journal": deepcopy(self._journal),
        }

    def restore_state(self, snapshot: dict) -> None:
        self._position_meta = {
            sym: dataclass_from_plain(_PositionMeta, data)
            for sym, data in snapshot.get("position_meta", {}).items()
        }
        self._exit_manager._states = {
            sym: dataclass_from_plain(BreakoutExitState, data)
            for sym, data in snapshot.get("exit_states", {}).items()
        }
        self._trail_manager._last_stops = dict(snapshot.get("trail_stops", {}))
        self._recent_exits = dict(snapshot.get("recent_exits", {}))
        self._reentry_count = {
            sym: int(count) for sym, count in snapshot.get("reentry_count", {}).items()
        }
        if "m30_bar_count" in snapshot:
            self._m30_bar_count = deepcopy(snapshot["m30_bar_count"])
        if "m30_indicators" in snapshot:
            self._m30_indicators = deepcopy(snapshot["m30_indicators"])
        if "h4_indicators" in snapshot:
            self._h4_indicators = deepcopy(snapshot["h4_indicators"])
        if "m30_incremental" in snapshot:
            self._m30_inc = deepcopy(snapshot["m30_incremental"])
        if "h4_incremental" in snapshot:
            self._h4_inc = deepcopy(snapshot["h4_incremental"])
        if "profile_bar_count" in snapshot:
            self._profile_bar_count = deepcopy(snapshot["profile_bar_count"])
        if "current_profile" in snapshot:
            self._current_profile = deepcopy(snapshot["current_profile"])
        if "balance_zones" in snapshot:
            self._balance_detector._zones = deepcopy(snapshot["balance_zones"])
        if "pending_retests" in snapshot:
            self._confirmation_detector._pending = deepcopy(snapshot["pending_retests"])
        if "risk_manager" in snapshot:
            self._risk_manager.restore_state(snapshot["risk_manager"])
        if "blocked_relaxed_body_signals" in snapshot:
            self._blocked_relaxed_body_signals = deepcopy(
                snapshot["blocked_relaxed_body_signals"]
            )
        if "detector_blocked_relaxed_body_signals" in snapshot:
            self._breakout_detector._blocked_relaxed_body_signals = deepcopy(
                snapshot["detector_blocked_relaxed_body_signals"]
            )
        if "journal" in snapshot:
            self._journal = deepcopy(snapshot["journal"])

    def on_init(self, ctx: StrategyContext) -> None:
        self._ctx = ctx

        for sym in self._cfg.symbols:
            self._m30_bar_count[sym] = 0
            self._m30_indicators[sym] = None
            self._h4_indicators[sym] = None
            self._profile_bar_count[sym] = 0
            self._current_profile[sym] = None
            self._recent_exits[sym] = {}
            self._reentry_count[sym] = 0

            # Incremental indicators per timeframe
            self._m30_inc[sym] = IncrementalIndicators(self._cfg.m30_indicators)
            self._h4_inc[sym] = IncrementalIndicators(self._cfg.h4_indicators)

        # Subscribe to position closed events
        ctx.events.subscribe(PositionClosedEvent, self._on_position_closed)

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        sym = bar.symbol
        if sym not in self._m30_bar_count:
            return

        if bar.timeframe == TimeFrame.H4:
            self._handle_h4(bar, sym)
        elif bar.timeframe == TimeFrame.M30:
            self._handle_m30(bar, sym, ctx)

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None:
        sym = fill.symbol

        if fill.tag == "entry":
            self._on_entry_fill(fill, ctx)
        elif fill.tag in ("tp1", "tp2"):
            self._on_tp_fill(fill, ctx)
        elif fill.tag in ("time_stop", "invalidation", "quick_exit"):
            pass  # Exit fills — position closed event handles bookkeeping

    def on_shutdown(self, ctx: StrategyContext) -> None:
        self._journal.save()
        log.info("strategy.shutdown", trades=len(self._journal.entries))

    def export_diagnostic_context(self) -> dict[str, object]:
        """Expose strategy-side diagnostic context for report generation."""
        return {
            "blocked_relaxed_body_signals": list(self._blocked_relaxed_body_signals),
        }

    def _measurement_start(self, ctx: StrategyContext) -> datetime | None:
        start_date = getattr(getattr(ctx, "config", None), "start_date", None)
        if start_date is None:
            return None
        if isinstance(start_date, datetime):
            return (
                start_date.astimezone(timezone.utc)
                if start_date.tzinfo is not None
                else start_date.replace(tzinfo=timezone.utc)
            )
        if not isinstance(start_date, date):
            return None
        return datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)

    def _entry_window_open(self, timestamp: datetime, ctx: StrategyContext) -> bool:
        measurement_start = self._measurement_start(ctx)
        return measurement_start is None or timestamp >= measurement_start

    @staticmethod
    def _scaled_risk_units(actual_risk_pct: float, baseline_risk_pct: float) -> float:
        if baseline_risk_pct <= 0:
            return 1.0
        return actual_risk_pct / baseline_risk_pct

    @staticmethod
    def _portfolio_snapshot(
        ctx: StrategyContext,
        sym: str,
        direction: Side,
    ) -> dict | None:
        snapshot_fn = getattr(ctx.broker, "get_portfolio_snapshot", None)
        if not callable(snapshot_fn):
            return None
        return snapshot_fn(sym, direction)

    def enrich_terminal_marks(self, terminal_marks: list[TerminalMark]) -> None:
        for mark in terminal_marks:
            meta = self._position_meta.get(mark.symbol)
            if meta is None:
                continue

            mark.setup_grade = meta.setup_grade
            mark.confluences_used = list(meta.confluences)
            mark.confirmation_type = meta.confirmation_type or None
            mark.entry_method = meta.entry_method or None
            mark.leverage = meta.leverage or mark.leverage
            mark.liquidation_price = meta.liquidation_price or mark.liquidation_price

            if meta.stop_distance > 0:
                if mark.direction == Side.LONG:
                    mark.unrealized_r_at_mark = (
                        (mark.mark_price_net_liquidation - mark.entry_price) / meta.stop_distance
                    )
                else:
                    mark.unrealized_r_at_mark = (
                        (mark.entry_price - mark.mark_price_net_liquidation) / meta.stop_distance
                    )

            mark.metadata.update({
                "risk_pct": meta.risk_pct,
                "stop_price": meta.stop_level,
                "stop_distance": meta.stop_distance,
                "original_qty": meta.original_qty,
                "h4_context": meta.h4_context_notes,
                "is_a_plus": meta.is_a_plus,
                "signal_variant": meta.signal_variant,
            })

    # ─── H4 handler ─────────────────────────────────────────────────────

    def _handle_h4(self, bar: Bar, sym: str) -> None:
        """Update H4 indicators for context bias."""
        snap = self._h4_inc[sym].update(bar)
        if snap is not None:
            self._h4_indicators[sym] = snap

    # ─── M30 handler (main flow) ────────────────────────────────────────

    def _handle_m30(self, bar: Bar, sym: str, ctx: StrategyContext) -> None:
        """Main M30 processing — volume profile, balance zones, breakout detection."""
        # Update bar count and indicators
        self._m30_bar_count[sym] += 1
        snap = self._m30_inc[sym].update(bar)
        if snap is not None:
            self._m30_indicators[sym] = snap

        # Begin instrumentation bar cycle
        self._collector.begin_bar(sym, bar.close)

        # Warmup
        bar_count = self._m30_bar_count[sym]
        warmup_met = bar_count >= WARMUP_BARS
        self._collector.record_gate(sym, "warmup", warmup_met,
            "insufficient_bars" if not warmup_met else "",
            threshold=WARMUP_BARS, actual_value=bar_count)
        if not warmup_met:
            self._collector.end_bar(sym)
            return

        m30_ind = self._m30_indicators[sym]
        self._collector.record_gate(sym, "indicators", m30_ind is not None,
            "no_snapshot" if m30_ind is None else "")
        if m30_ind is None:
            self._collector.end_bar(sym)
            return

        # --- Fetch M30 history ---
        lookback = self._cfg.profile.lookback_bars + 50
        m30_bars = ctx.bars.get(sym, TimeFrame.M30, count=lookback)
        has_history = bool(m30_bars)
        self._collector.record_gate(sym, "m30_history", has_history,
            "no_m30_bars" if not has_history else "")
        if not has_history:
            self._collector.end_bar(sym)
            return

        atr = m30_ind.atr
        atr_valid = atr > 0
        self._collector.record_gate(sym, "atr_validity", atr_valid,
            "atr_zero_or_negative" if not atr_valid else "")
        if not atr_valid:
            self._collector.end_bar(sym)
            return

        # --- Update volume profile (periodic recalc) ---
        self._profile_bar_count[sym] += 1
        if (self._current_profile[sym] is None
                or self._profile_bar_count[sym] >= self._cfg.profile.recalc_interval_bars):
            profile_bars = m30_bars[-self._cfg.profile.lookback_bars:]
            profile = self._profiler.build(profile_bars)
            if profile is not None:
                self._current_profile[sym] = profile
            else:
                log.warning("breakout.profile_stale", symbol=sym,
                    using_cached=self._current_profile.get(sym) is not None)
            self._profile_bar_count[sym] = 0

        profile = self._current_profile[sym]

        # --- Update balance zones ---
        if profile is not None:
            self._balance_detector.update(
                sym=sym,
                bars=m30_bars[-self._cfg.profile.lookback_bars:],
                profile=profile,
                atr=atr,
                bar_index=self._m30_bar_count[sym],
            )

        # --- Manage existing positions ---
        self._manage_positions(bar, sym, ctx, m30_bars, m30_ind)

        entry_window_open = self._entry_window_open(bar.timestamp, ctx)

        # --- Check pending retest (Model 2) ---
        if self._confirmation_detector.has_pending(sym):
            pending_setup = self._confirmation_detector.get_pending_setup(sym)
            confirmation = self._confirmation_detector.check_retest(
                sym=sym,
                bar=bar,
                bars=m30_bars,
                atr=atr,
                bar_index=self._m30_bar_count[sym],
            )
            if confirmation is not None and pending_setup is not None:
                # Record instrumentation for the retest entry
                self._collector.record_gate(sym, "setup", True, "",
                    context={"confluences": list(pending_setup.confluences),
                             "grade": pending_setup.grade.value,
                             "room_r": pending_setup.room_r})
                self._collector.record_gate(sym, "model2_retest", True, "",
                    context={"model": confirmation.model})
                h4_ctx = self._context_analyzer.evaluate(self._h4_indicators.get(sym))
                self._collector.snapshot_context(sym, m30_ind,
                    h4_context_direction=h4_ctx.direction.value if h4_ctx.direction else None,
                    setup_grade=pending_setup.grade.value,
                    setup_confluences=list(pending_setup.confluences),
                    setup_room_r=pending_setup.room_r,
                    funding_rate=0.0)
                self._try_execute_confirmed_entry(
                    bar, sym, ctx, pending_setup, confirmation, m30_ind,
                    entry_window_open=entry_window_open,
                    retest_bar=bar,
                )
                self._collector.end_bar(sym)
                return

        # H4 context bias
        h4_ind = self._h4_indicators.get(sym)
        context = self._context_analyzer.evaluate(h4_ind)

        # Symbol direction filter (pre-detection)
        sf = self._cfg.symbol_filter
        rule = getattr(sf, f"{sym.lower()}_direction", "both")
        pre_dir_ok = rule != "disabled"
        self._collector.record_gate(sym, "symbol_direction_pre", pre_dir_ok,
            "symbol_disabled" if not pre_dir_ok else "",
            context={"rule": rule})
        if not pre_dir_ok:
            self._collector.end_bar(sym)
            return

        # Get active balance zones
        zones = self._balance_detector.get_active_zones(sym)
        has_zones = bool(zones)
        self._collector.record_gate(sym, "zones_available", has_zones,
            "no_active_zones" if not has_zones else "",
            context={"zone_count": len(zones)} if zones else {})
        if not has_zones:
            self._collector.end_bar(sym)
            return

        # Breakout detection
        setup = self._breakout_detector.detect(
            bar=bar,
            zones=zones,
            profile=profile,
            profiler=self._profiler,
            context=context,
            m30_ind=m30_ind,
            atr=atr,
            sym=sym,
        )
        blocked_relaxed = self._breakout_detector.consume_blocked_relaxed_body_signals()
        for candidate in blocked_relaxed:
            self._blocked_relaxed_body_signals.append({
                "signal_time": bar.timestamp,
                "symbol": sym,
                "direction": candidate.setup.direction.value,
                "blocked_rule": candidate.blocked_rule,
                "grade": candidate.setup.grade.value,
                "signal_variant": candidate.setup.signal_variant,
                "breakout_price": candidate.setup.breakout_price,
                "body_ratio": candidate.setup.body_ratio,
                "room_r": candidate.setup.room_r,
                "volume_mult": candidate.setup.volume_mult,
                "confluences": list(candidate.setup.confluences),
                "confluence_count": len(candidate.setup.confluences),
            })
        self._collector.record_gate(sym, "breakout_detection", setup is not None,
            "no_breakout_signal" if setup is None else "",
            context={"grade": setup.grade.value, "direction": setup.direction.value,
                     "room_r": setup.room_r, "confluences": list(setup.confluences)} if setup else {})
        if setup is None:
            self._collector.end_bar(sym)
            return

        # Update context with setup info
        self._collector.snapshot_context(sym, m30_ind,
            h4_context_direction=context.direction.value if context.direction else None,
            h4_context_strength=context.strength if hasattr(context, 'strength') else None,
            setup_grade=setup.grade.value, setup_confluences=list(setup.confluences),
            setup_room_r=setup.room_r, funding_rate=0.0)

        # Apply symbol direction filter on detected direction
        post_dir_ok = not ((rule == "long_only" and setup.direction == Side.SHORT) or
                           (rule == "short_only" and setup.direction == Side.LONG))
        self._collector.record_gate(sym, "symbol_direction_post", post_dir_ok,
            f"{rule}_blocks_{setup.direction.value}" if not post_dir_ok else "",
            context={"rule": rule, "direction": setup.direction.value})
        if not post_dir_ok:
            self._collector.end_bar(sym)
            return

        # Countertrend blocking
        countertrend_ok = True
        if not self._cfg.context.allow_countertrend and context.direction is not None:
            if context.direction != setup.direction:
                countertrend_ok = False
        self._collector.record_gate(sym, "countertrend_context", countertrend_ok,
            "h4_context_opposes_breakout" if not countertrend_ok else "")
        if not countertrend_ok:
            self._collector.end_bar(sym)
            return

        # Model 1: immediate entry signal. Zone lifecycle remains market-derived;
        # execution state must not mutate the active signal inventory.
        if self._cfg.confirmation.enable_model1:
            confirmation = self._confirmation_detector.check_breakout_close(
                bar=bar, setup=setup, m30_ind=m30_ind,
            )
            model1_ok = confirmation is not None
            self._collector.record_gate(sym, "model1_confirmation", model1_ok,
                "model1_not_confirmed" if not model1_ok else "")
            if model1_ok:
                self._collector.record_signal_factor(sym, "setup_room_r", setup.room_r or 0.0)
                self._collector.record_signal_factor(sym, "confluences", len(setup.confluences) / 8.0)

                self._try_execute_confirmed_entry(
                    bar, sym, ctx, setup, confirmation, m30_ind,
                    entry_window_open=entry_window_open,
                    retest_bar=None,
                )
                self._collector.end_bar(sym)
                return

        # Model 2: register for retest monitoring without retiring the source zone.
        if self._cfg.confirmation.enable_model2:
            self._confirmation_detector.register_breakout(
                sym=sym,
                setup=setup,
                bar_idx=self._m30_bar_count[sym],
            )

        self._collector.end_bar(sym)

    def _try_execute_confirmed_entry(
        self,
        bar: Bar,
        sym: str,
        ctx: StrategyContext,
        setup,
        confirmation,
        m30_ind: IndicatorSnapshot,
        entry_window_open: bool,
        retest_bar: Bar | None,
    ) -> bool:
        """Submit a confirmed signal only if execution-state gates allow it."""
        self._collector.record_gate(sym, "entry_window", entry_window_open,
            "before_measurement_start" if not entry_window_open else "")
        if not entry_window_open:
            return False

        pos = ctx.broker.get_position(sym)
        has_pos = pos is not None
        self._collector.record_gate(sym, "position_check", not has_pos,
            "position_exists" if has_pos else "")
        if has_pos:
            return False

        is_reentry, reentry_block_reason = self._evaluate_reentry_for_execution(
            sym,
            setup.direction,
        )
        self._collector.record_gate(sym, "reentry_eval", reentry_block_reason == "",
            reentry_block_reason)
        if reentry_block_reason:
            return False

        equity = ctx.broker.get_equity()
        stopped, stop_reason = self._risk_manager.is_session_stopped(equity, bar.timestamp)
        self._collector.record_gate(sym, "risk_check", not stopped, stop_reason)
        if stopped:
            return False

        execution_setup = setup
        if is_reentry and self._cfg.reentry.risk_scale != 1.0:
            execution_setup = replace(
                setup,
                risk_scale=setup.risk_scale * self._cfg.reentry.risk_scale,
            )

        entered = self._execute_entry(
            bar, sym, ctx, execution_setup, confirmation, m30_ind,
            retest_bar=retest_bar,
        )
        if is_reentry and entered:
            self._reentry_count[sym] = self._reentry_count.get(sym, 0) + 1
        return entered

    def _evaluate_reentry_for_execution(
        self,
        sym: str,
        setup_direction: Side,
    ) -> tuple[bool, str]:
        """Return (is_reentry, block_reason) for execution-only reentry gates."""
        recent = self._recent_exits.get(sym) or {}
        if not recent:
            return False, ""

        bars_since = self._m30_bar_count[sym] - recent.get("bar_idx", 0)
        max_wait = self._cfg.reentry.max_wait_bars
        if max_wait > 0 and bars_since > max_wait:
            self._clear_reentry_state(sym)
            return False, ""

        recent_side = recent.get("side")
        recent_side_value = getattr(recent_side, "value", recent_side)
        if recent_side_value != setup_direction.value:
            self._clear_reentry_state(sym)
            return False, ""

        if not self._cfg.reentry.enabled:
            self._clear_reentry_state(sym)
            return False, ""

        loss_r = abs(recent.get("loss_r", 0))
        if loss_r > self._cfg.reentry.max_loss_r:
            self._clear_reentry_state(sym)
            return False, ""

        if bars_since < self._cfg.reentry.cooldown_bars:
            return False, "reentry_cooldown"

        count = self._reentry_count.get(sym, 0)
        if count >= self._cfg.reentry.max_reentries:
            self._clear_reentry_state(sym)
            return False, ""

        return True, ""

    def _clear_reentry_state(self, sym: str) -> None:
        self._recent_exits[sym] = {}
        self._reentry_count[sym] = 0

    # ─── Entry execution ────────────────────────────────────────────────

    def _execute_entry(
        self,
        bar: Bar,
        sym: str,
        ctx: StrategyContext,
        setup,
        confirmation,
        m30_ind: IndicatorSnapshot,
        retest_bar: Bar | None,
    ) -> bool:
        """Shared entry path for Model 1 and Model 2.

        Returns True if the entry order was submitted, False otherwise.
        """
        equity = ctx.broker.get_equity()

        # Stop placement
        stop_level = self._stop_placer.compute(
            setup=setup,
            retest_bar=retest_bar,
            entry_price=bar.close,
            atr=m30_ind.atr,
            direction=setup.direction,
        )
        stop_distance = abs(bar.close - stop_level)
        if stop_distance <= 0:
            self._collector.record_gate(sym, "execute_stop", False, "zero_stop_distance")
            log.warning("breakout.entry_aborted", symbol=sym, reason="zero_stop_distance")
            return False

        # Position sizing
        open_positions = [
            ctx.broker.get_position(s)
            for s in self._cfg.symbols
            if ctx.broker.get_position(s) is not None
        ]
        sizing, sizing_reason = self._sizer.compute(
            equity=equity,
            entry_price=bar.close,
            stop_distance=stop_distance,
            grade=setup.grade,
            is_a_plus=setup.is_a_plus,
            symbol=sym,
            open_positions=open_positions,
            direction=setup.direction,
            risk_scale=setup.risk_scale,
        )
        self._collector.record_gate(sym, "execute_sizing", sizing is not None, sizing_reason)
        if sizing is None:
            return False

        # Entry order
        order_seed = {
            "symbol": sym,
            "timeframe": bar.timeframe.value,
            "bar_timestamp": bar.timestamp.isoformat(),
            "direction": setup.direction.value,
            "grade": setup.grade.value,
            "is_a_plus": setup.is_a_plus,
        }
        order_id = f"brk_entry_{sym}_{stable_hash(order_seed, length=8)}"
        entry_order = self._entry_generator.generate(
            bar=bar,
            direction=setup.direction,
            qty=sizing.qty,
            sizing_result=sizing,
            setup=setup,
            confirmation=confirmation,
            symbol=sym,
            order_id=order_id,
        )
        if entry_order is None:
            self._collector.record_gate(sym, "execute_entry_order", False, "entry_generation_failed")
            log.warning("breakout.entry_aborted", symbol=sym, reason="entry_generation_failed")
            return False

        if setup.is_a_plus:
            baseline_risk_pct = self._cfg.risk.risk_pct_a_plus
        elif setup.grade == SetupGrade.A:
            baseline_risk_pct = self._cfg.risk.risk_pct_a
        else:
            baseline_risk_pct = self._cfg.risk.risk_pct_b
        entry_order.metadata["risk_R"] = self._scaled_risk_units(
            sizing.risk_pct_actual,
            baseline_risk_pct,
        )

        # Record entry instrumentation just before submission
        portfolio_state = self._portfolio_snapshot(ctx, sym, setup.direction)
        self._collector.record_entry(sym, self._cfg.to_dict(), {
            "equity": equity,
            "stop_distance": stop_distance,
            "risk_scale": setup.risk_scale,
        }, portfolio_state=portfolio_state)

        # Submit and store meta
        ctx.broker.submit_order(entry_order)
        if entry_order.status == OrderStatus.REJECTED:
            return False

        self._position_meta[sym] = _PositionMeta(
            setup_grade=setup.grade,
            is_a_plus=setup.is_a_plus,
            confluences=setup.confluences,
            confirmation_type=confirmation.model,
            entry_method=confirmation.model,
            signal_variant=setup.signal_variant,
            entry_price=bar.close,
            stop_level=stop_level,
            stop_distance=stop_distance,
            leverage=sizing.leverage,
            liquidation_price=sizing.liquidation_price,
            risk_pct=sizing.risk_pct_actual,
            original_qty=sizing.qty,
            entry_bar_index=self._m30_bar_count[sym],
            balance_zone=setup.balance_zone,
            h4_context_notes=str(self._context_analyzer.evaluate(
                self._h4_indicators.get(sym)
            ).reasons),
        )

        log.info(
            "breakout.entry_submitted",
            symbol=sym,
            direction=setup.direction.value,
            grade=setup.grade.value,
            a_plus=setup.is_a_plus,
            model=confirmation.model,
            signal_variant=setup.signal_variant,
            confluences=list(setup.confluences),
            room_r=f"{setup.room_r:.2f}",
            stop_dist=f"{stop_distance:.2f}",
        )
        return True

    # ─── Position management ────────────────────────────────────────────

    def _manage_positions(
        self, bar: Bar, sym: str, ctx: StrategyContext,
        m30_bars: list[Bar], m30_ind: IndicatorSnapshot,
    ) -> None:
        """Manage exits and trail for existing positions."""
        pos = ctx.broker.get_position(sym)
        if pos is None:
            return

        meta = self._position_meta.get(sym)
        if meta is None:
            return

        # Exit management — returns orders to submit
        orders = self._exit_manager.process_bar(bar, sym)
        exit_state = self._exit_manager.get_state(sym)
        for order_index, order in enumerate(orders):
            self._ensure_exit_order_id(
                order,
                bar=bar,
                state=exit_state,
                order_index=order_index,
            )
            ctx.broker.submit_order(order)

        # Check if remaining quantity is 0 after partial exits
        exit_state = self._exit_manager.get_state(sym)
        if exit_state is None:
            return

        remaining_qty = pos.qty  # Current broker position qty

        # Smart BE — apply if be_moved and current stop is worse than BE price
        if (
            exit_state.early_lock_applied
            and self._cfg.exits.early_lock_enabled
            and not exit_state.tp1_hit
        ):
            lock_price = meta.entry_price
            if meta.stop_distance > 0:
                if exit_state.direction == Side.LONG:
                    lock_price = (
                        meta.entry_price
                        + self._cfg.exits.early_lock_stop_r * meta.stop_distance
                    )
                else:
                    lock_price = (
                        meta.entry_price
                        - self._cfg.exits.early_lock_stop_r * meta.stop_distance
                    )

            if meta.stop_order_id is not None:
                current_stop = self._get_current_stop_price(sym, ctx)
                should_apply_lock = current_stop is None
                if current_stop is not None:
                    if exit_state.direction == Side.LONG:
                        should_apply_lock = current_stop < lock_price
                    else:
                        should_apply_lock = current_stop > lock_price
                if should_apply_lock:
                    cancelled = ctx.broker.cancel_order(meta.stop_order_id)
                    if not cancelled:
                        log.warning("breakout.cancel_failed", symbol=sym,
                            order_id=meta.stop_order_id, context="early_lock_be")
                    new_stop_id = self._management_order_id("lock", sym, {
                        "bar_timestamp": bar.timestamp.isoformat(),
                        "timeframe": bar.timeframe.value,
                        "direction": exit_state.direction.value,
                        "qty": remaining_qty,
                        "stop_price": lock_price,
                        "previous_stop_order_id": meta.stop_order_id,
                    })
                    reverse_side = Side.SHORT if exit_state.direction == Side.LONG else Side.LONG
                    lock_order = Order(
                        order_id=new_stop_id,
                        symbol=sym,
                        side=reverse_side,
                        order_type=OrderType.STOP,
                        qty=remaining_qty,
                        stop_price=lock_price,
                        tag="protective_stop",
                    )
                    ctx.broker.submit_order(lock_order)
                    meta.stop_order_id = new_stop_id

        if exit_state.be_moved and self._cfg.exits.be_after_tp1:
            be_price = meta.entry_price
            if meta.stop_distance > 0:
                if exit_state.direction == Side.LONG:
                    be_price = meta.entry_price + self._cfg.exits.be_buffer_r * meta.stop_distance
                else:
                    be_price = meta.entry_price - self._cfg.exits.be_buffer_r * meta.stop_distance

            if meta.stop_order_id is not None:
                current_stop = self._get_current_stop_price(sym, ctx)
                should_apply_be = current_stop is None
                if current_stop is not None:
                    if exit_state.direction == Side.LONG:
                        should_apply_be = current_stop < be_price
                    else:
                        should_apply_be = current_stop > be_price
                if should_apply_be:
                    cancelled = ctx.broker.cancel_order(meta.stop_order_id)
                    if not cancelled:
                        log.warning("breakout.cancel_failed", symbol=sym,
                            order_id=meta.stop_order_id, context="smart_be_after_tp1")
                    new_stop_id = self._management_order_id("be", sym, {
                        "bar_timestamp": bar.timestamp.isoformat(),
                        "timeframe": bar.timeframe.value,
                        "direction": exit_state.direction.value,
                        "qty": remaining_qty,
                        "stop_price": be_price,
                        "previous_stop_order_id": meta.stop_order_id,
                    })
                    reverse_side = Side.SHORT if exit_state.direction == Side.LONG else Side.LONG
                    be_order = Order(
                        order_id=new_stop_id,
                        symbol=sym,
                        side=reverse_side,
                        order_type=OrderType.STOP,
                        qty=remaining_qty,
                        stop_price=be_price,
                        tag="protective_stop",
                    )
                    ctx.broker.submit_order(be_order)
                    meta.stop_order_id = new_stop_id

        # Trail management
        if remaining_qty > 0:
            current_stop = self._get_current_stop_price(sym, ctx)
            bars_since = self._m30_bar_count[sym] - meta.entry_bar_index

            new_stop = self._trail_manager.update(
                sym=sym,
                direction=exit_state.direction,
                bars=m30_bars,
                m30_ind=m30_ind,
                current_stop=current_stop,
                bars_since_entry=bars_since,
                current_r=exit_state.current_r,
                mfe_r=exit_state.mfe_r,
            )

            if new_stop is not None and meta.stop_order_id is not None:
                cancelled = ctx.broker.cancel_order(meta.stop_order_id)
                if not cancelled:
                    log.warning("breakout.cancel_failed", symbol=sym,
                        order_id=meta.stop_order_id, context="trail_resubmit")
                new_stop_id = self._management_order_id("trail", sym, {
                    "bar_timestamp": bar.timestamp.isoformat(),
                    "timeframe": bar.timeframe.value,
                    "direction": exit_state.direction.value,
                    "qty": remaining_qty,
                    "stop_price": new_stop,
                    "previous_stop_order_id": meta.stop_order_id,
                    "bars_since_entry": bars_since,
                })
                reverse_side = Side.SHORT if exit_state.direction == Side.LONG else Side.LONG
                trail_order = Order(
                    order_id=new_stop_id,
                    symbol=sym,
                    side=reverse_side,
                    order_type=OrderType.STOP,
                    qty=remaining_qty,
                    stop_price=new_stop,
                    tag="protective_stop",
                )
                ctx.broker.submit_order(trail_order)
                meta.stop_order_id = new_stop_id

    def _get_current_stop_price(self, sym: str, ctx: StrategyContext) -> float | None:
        """Get the current protective stop price from open orders."""
        meta = self._position_meta.get(sym)
        if meta is None or meta.stop_order_id is None:
            return None

        for order in ctx.broker.get_open_orders(sym):
            if order.order_id == meta.stop_order_id and order.stop_price is not None:
                return order.stop_price

        return None

    # ─── Fill handlers ──────────────────────────────────────────────────

    def _on_entry_fill(self, fill: Fill, ctx: StrategyContext) -> None:
        """Handle entry fill: update meta, place protective stop."""
        sym = fill.symbol
        meta = self._position_meta.get(sym)
        if meta is None:
            return

        # Update with actual fill price
        meta.entry_price = fill.fill_price
        meta.original_qty = fill.qty
        meta.stop_distance = abs(fill.fill_price - meta.stop_level)
        if meta.stop_distance <= 0:
            meta.stop_distance = 0.001  # Safety

        # Submit protective stop
        reverse_side = Side.SHORT if fill.side == Side.LONG else Side.LONG
        stop_id = self._management_order_id("stop", sym, {
            "fill_id": fill.exchange_fill_id or fill.order_id,
            "order_id": fill.order_id,
            "timestamp": fill.timestamp.isoformat(),
            "side": fill.side.value,
            "qty": meta.original_qty,
            "stop_price": meta.stop_level,
        })
        stop_order = Order(
            order_id=stop_id,
            symbol=sym,
            side=reverse_side,
            order_type=OrderType.STOP,
            qty=meta.original_qty,
            stop_price=meta.stop_level,
            tag="protective_stop",
        )
        ctx.broker.submit_order(stop_order)
        meta.stop_order_id = stop_id

        # Initialize exit manager
        m30_ind = self._m30_indicators.get(sym)
        atr = m30_ind.atr if m30_ind is not None else 0.0
        self._exit_manager.init_position(
            sym=sym,
            entry_price=fill.fill_price,
            stop_distance=meta.stop_distance,
            qty=meta.original_qty,
            direction=fill.side,
            balance_zone=meta.balance_zone,
            atr=atr,
        )

        log.info(
            "breakout.entry_filled",
            symbol=sym,
            price=fill.fill_price,
            qty=fill.qty,
            stop=meta.stop_level,
        )

    def _on_tp_fill(self, fill: Fill, ctx: StrategyContext) -> None:
        """Handle TP1/TP2 fill: update stop with remaining qty."""
        sym = fill.symbol
        meta = self._position_meta.get(sym)
        if meta is None:
            return

        pos = ctx.broker.get_position(sym)
        if pos is None:
            return

        remaining_qty = pos.qty
        if remaining_qty <= 0:
            return

        # Get current stop price BEFORE cancelling to prevent regression
        current_stop = self._get_current_stop_price(sym, ctx)

        # Cancel old stop and resubmit with reduced qty
        if meta.stop_order_id:
            cancelled = ctx.broker.cancel_order(meta.stop_order_id)
            if not cancelled:
                log.warning("breakout.cancel_failed", symbol=sym,
                    order_id=meta.stop_order_id, context="tp_fill_stop_resubmit")

        # Determine stop price — use BE if moved, else original stop
        exit_state = self._exit_manager.get_state(sym)
        stop_price = meta.stop_level
        if exit_state and exit_state.be_moved:
            if exit_state.direction == Side.LONG:
                be = meta.entry_price + self._cfg.exits.be_buffer_r * meta.stop_distance
            else:
                be = meta.entry_price - self._cfg.exits.be_buffer_r * meta.stop_distance
            stop_price = be

        # Never regress stop — keep the better of current trail vs BE/original
        if current_stop is not None and exit_state is not None:
            if exit_state.direction == Side.LONG:
                stop_price = max(stop_price, current_stop)
            else:
                stop_price = min(stop_price, current_stop)

        new_stop_id = self._management_order_id("stop", sym, {
            "fill_id": fill.exchange_fill_id or fill.order_id,
            "order_id": fill.order_id,
            "timestamp": fill.timestamp.isoformat(),
            "side": fill.side.value,
            "qty": remaining_qty,
            "stop_price": stop_price,
            "previous_stop_order_id": meta.stop_order_id,
        })
        stop_order = Order(
            order_id=new_stop_id,
            symbol=sym,
            side=fill.side,  # Same side as exit (opposite of position)
            order_type=OrderType.STOP,
            qty=remaining_qty,
            stop_price=stop_price,
            tag="protective_stop",
        )
        ctx.broker.submit_order(stop_order)
        meta.stop_order_id = new_stop_id

    # ─── Position closed event ──────────────────────────────────────────

    def _on_position_closed(self, event: PositionClosedEvent) -> None:
        """Enrich trade with strategy metadata on position close."""
        trade = event.trade
        sym = trade.symbol
        meta = self._position_meta.pop(sym, None)

        # Cancel ALL open orders for this symbol to prevent orphaned stops
        # from accidentally opening new unwanted positions
        if self._ctx is not None:
            for order in self._ctx.broker.get_open_orders(sym):
                cancelled = self._ctx.broker.cancel_order(order.order_id)
                if not cancelled:
                    log.warning("breakout.cancel_failed", symbol=sym,
                                order_id=order.order_id, context="position_closed_cleanup")

        # Clean up exit/trail state
        exit_state = self._exit_manager.remove(sym)
        self._trail_manager.remove(sym)

        # Enrich trade
        if meta is not None:
            trade.setup_grade = meta.setup_grade
            trade.confluences_used = list(meta.confluences)
            trade.confirmation_type = meta.confirmation_type
            trade.entry_method = meta.entry_method
            trade.signal_variant = meta.signal_variant

        # R-multiple is derived from price/stop geometry, independent of exit_state.
            if meta.stop_distance > 0:
                if trade.direction == Side.LONG:
                    trade.r_multiple = (trade.exit_price - trade.entry_price) / meta.stop_distance
                else:
                    trade.r_multiple = (trade.entry_price - trade.exit_price) / meta.stop_distance
                initial_risk = trade.qty * meta.stop_distance
                if initial_risk > 0:
                    trade.realized_r_multiple = trade.net_pnl / initial_risk

            # MAE/MFE from exit state (only available for normal exits)
            if exit_state is not None:
                trade.mae_r = exit_state.mae_r
                trade.mfe_r = exit_state.mfe_r

            # Instrumentation: score quality and build instrumented event
            entry_ctx = self._collector._entry_context.get(sym)
            entry_decisions = self._collector._entry_decisions.get(sym, [])
            entry_sizing = self._collector._entry_sizing_inputs.get(sym, {})
            quality_score, root_causes = self._quality_scorer.score(
                trade, entry_ctx, entry_decisions, entry_sizing)
            instrumented = self._collector.on_trade_closed(
                sym, trade, quality_score, root_causes)
            if self._collector.emitter:
                self._collector.emitter.emit_trade(instrumented)

        # Track for re-entry only after R fields have been computed.
        loss_r = trade.economic_r_multiple
        if trade.net_pnl < 0 and loss_r is not None:
            self._recent_exits[sym] = {
                "bar_idx": self._m30_bar_count.get(sym, 0),
                "side": trade.direction,
                "loss_r": abs(loss_r),
            }
        else:
            self._recent_exits[sym] = {}
            self._reentry_count[sym] = 0

        # Record in risk manager
        self._risk_manager.record_trade_exit(
            trade.net_pnl,
            trade.exit_time if trade.exit_time else trade.entry_time,
        )

        # Record in journal
        context = {}
        if meta:
            context = {
                "leverage": meta.leverage,
                "risk_pct": meta.risk_pct,
                "h4_context": meta.h4_context_notes,
                "is_a_plus": meta.is_a_plus,
                "signal_variant": meta.signal_variant,
            }
        self._journal.record(trade, context)

        log.info(
            "breakout.trade_closed",
            symbol=sym,
            pnl=f"{trade.net_pnl:.2f}",
            r=f"{trade.r_multiple:.2f}" if trade.r_multiple is not None else "n/a",
            exit_reason=trade.exit_reason,
        )
