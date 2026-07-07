"""TrendStrategy — Institutional Anchor Pro trend-following continuation."""

from __future__ import annotations

from dataclasses import dataclass
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

from .config import TrendConfig
from .confirmation import TriggerDetector
from .entry import EntryGenerator
from .exits import ExitManager, TrendExitState
from .indicators import WeeklyTracker
from .regime import RegimeClassifier, RegimeResult, StructureTracker
from .risk import RiskManager
from .setup import SetupDetector, TrendSetupResult
from .sizing import PositionSizer
from .stops import StopPlacer
from .trail import TrailManager
from crypto_trader.instrumentation.collector import InstrumentationCollector
from crypto_trader.instrumentation.quality import ProcessQualityScorer
from crypto_trader.strategy.snapshot import dataclass_from_plain, to_plain

log = structlog.get_logger()

WARMUP_BARS = 50  # ~2 days of H1 data


@dataclass
class _PositionMeta:
    """Per-position metadata tracked by the strategy."""
    setup_grade: SetupGrade | None = None
    confluences: tuple[str, ...] = ()
    confirmation_type: str = ""
    entry_method: str = ""
    entry_price: float = 0.0
    stop_level: float = 0.0
    stop_distance: float = 0.0
    leverage: float = 0.0
    liquidation_price: float = 0.0
    risk_pct: float = 0.0
    original_qty: float = 0.0
    entry_bar_index: int = 0
    entry_m15_bar_index: int = 0
    stop_order_id: str | None = None
    d1_regime_notes: str = ""
    is_reentry: bool = False


@dataclass
class _PendingTrendSetup:
    """Setup waiting for a later H1 confirmation bar."""
    setup: TrendSetupResult
    created_h1_bar_index: int
    regime_tier: str
    regime_reasons: tuple[str, ...] = ()
    is_reentry: bool = False
    min_confluences_override: int | None = None


class TrendStrategy:
    """Institutional Anchor Pro — trend-following continuation strategy.

    D1 → regime | H1 → setup + entry | M15 → position management (primary).
    """

    def __init__(self, config: TrendConfig | None = None, bot_id: str = "") -> None:
        self._cfg = config or TrendConfig()

        # Module instances
        self._regime_classifier = RegimeClassifier(self._cfg.regime)
        self._setup_detector = SetupDetector(self._cfg.setup)
        self._trigger_detector = TriggerDetector(self._cfg.confirmation)
        self._entry_generator = EntryGenerator(self._cfg.entry)
        self._stop_placer = StopPlacer(self._cfg.stops)
        self._sizer = PositionSizer(self._cfg.risk, self._cfg.limits)
        self._exit_manager = ExitManager(self._cfg.exits)
        self._trail_manager = TrailManager(self._cfg.trail)
        self._risk_manager = RiskManager(self._cfg.limits)
        self._journal = TradeJournal()

        # Instrumentation
        self._collector = InstrumentationCollector(strategy_id="trend", bot_id=bot_id)
        self._quality_scorer = ProcessQualityScorer()

        # Per-symbol state
        self._position_meta: dict[str, _PositionMeta] = {}
        self._h1_bar_count: dict[str, int] = {}
        self._m15_bar_count: dict[str, int] = {}
        self._current_regime: dict[str, RegimeResult | None] = {}
        self._h1_indicators: dict[str, IndicatorSnapshot | None] = {}
        self._d1_indicators: dict[str, IndicatorSnapshot | None] = {}
        self._m15_indicators: dict[str, IndicatorSnapshot | None] = {}
        self._pending_setups: dict[str, _PendingTrendSetup] = {}

        # Per-symbol incremental indicator instances
        self._h1_inc: dict[str, IncrementalIndicators] = {}
        self._d1_inc: dict[str, IncrementalIndicators] = {}
        self._m15_inc: dict[str, IncrementalIndicators] = {}

        # Structure and weekly trackers (per symbol)
        self._structure_trackers: dict[str, StructureTracker] = {}
        self._weekly_trackers: dict[str, WeeklyTracker] = {}

        # Re-entry tracking
        self._recent_exits: dict[str, dict] = {}
        self._reentry_count: dict[str, int] = {}

        self._ctx: StrategyContext | None = None

    @property
    def name(self) -> str:
        return "trend_anchor"

    @property
    def symbols(self) -> list[str]:
        return self._cfg.symbols

    @property
    def timeframes(self) -> list[TimeFrame]:
        return [TimeFrame.M15, TimeFrame.H1, TimeFrame.D1]

    @property
    def journal(self) -> TradeJournal:
        return self._journal

    @staticmethod
    def _management_order_id(purpose: str, symbol: str, seed: dict[str, object]) -> str:
        order_seed = {
            "strategy": "trend",
            "purpose": purpose,
            "symbol": symbol,
            **seed,
        }
        return f"trend_{purpose}_{symbol}_{stable_hash(order_seed, length=8)}"

    def snapshot_state(self) -> dict:
        return {
            "position_meta": to_plain(self._position_meta),
            "exit_states": to_plain(getattr(self._exit_manager, "_states", {})),
            "pending_setups": to_plain(self._pending_setups),
            "recent_exits": to_plain(self._recent_exits),
            "reentry_count": to_plain(self._reentry_count),
        }

    def restore_state(self, snapshot: dict) -> None:
        self._position_meta = {
            sym: dataclass_from_plain(_PositionMeta, data)
            for sym, data in snapshot.get("position_meta", {}).items()
        }
        self._exit_manager._states = {
            sym: dataclass_from_plain(TrendExitState, data)
            for sym, data in snapshot.get("exit_states", {}).items()
        }
        self._pending_setups = {
            sym: dataclass_from_plain(_PendingTrendSetup, data)
            for sym, data in snapshot.get("pending_setups", {}).items()
        }
        self._recent_exits = dict(snapshot.get("recent_exits", {}))
        self._reentry_count = {
            sym: int(count) for sym, count in snapshot.get("reentry_count", {}).items()
        }

    def on_init(self, ctx: StrategyContext) -> None:
        self._ctx = ctx

        for sym in self._cfg.symbols:
            self._h1_bar_count[sym] = 0
            self._m15_bar_count[sym] = 0
            self._current_regime[sym] = None
            self._h1_indicators[sym] = None
            self._d1_indicators[sym] = None
            self._m15_indicators[sym] = None
            self._recent_exits[sym] = {}
            self._reentry_count[sym] = 0

            # Incremental indicators per timeframe
            self._h1_inc[sym] = IncrementalIndicators(self._cfg.h1_indicators)
            self._d1_inc[sym] = IncrementalIndicators(self._cfg.d1_indicators)
            self._m15_inc[sym] = IncrementalIndicators(self._cfg.m15_indicators)

            # Structure & weekly
            self._structure_trackers[sym] = StructureTracker(
                lookback=self._cfg.regime.structure_lookback
            )
            self._weekly_trackers[sym] = WeeklyTracker()

        # Subscribe to position closed events
        ctx.events.subscribe(PositionClosedEvent, self._on_position_closed)

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        sym = bar.symbol
        if sym not in self._h1_bar_count:
            return

        if bar.timeframe == TimeFrame.D1:
            self._handle_d1(bar, sym)
        elif bar.timeframe == TimeFrame.H1:
            self._handle_h1(bar, sym, ctx)
        elif bar.timeframe == TimeFrame.M15:
            self._handle_m15(bar, sym, ctx)

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None:
        sym = fill.symbol

        if fill.tag == "entry":
            self._on_entry_fill(fill, ctx)
        elif fill.tag in ("tp1", "tp2"):
            self._on_tp_fill(fill, ctx)
        elif fill.tag in ("time_stop", "ema_failsafe", "quick_exit", "scratch_exit", "mfe_lock_exit"):
            pass  # Exit fills — position closed event handles bookkeeping

    def on_shutdown(self, ctx: StrategyContext) -> None:
        self._journal.save()
        log.info("strategy.shutdown", trades=len(self._journal.entries))

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

    def _pending_setup_is_valid(
        self,
        sym: str,
        pending: _PendingTrendSetup,
        direction: Side,
    ) -> bool:
        bars_since = self._h1_bar_count.get(sym, 0) - pending.created_h1_bar_index
        max_bars = max(int(self._cfg.confirmation.max_bars_after_setup), 0)
        return (
            pending.setup.direction == direction
            and bars_since > 0
            and bars_since <= max_bars
        )

    def _clear_pending_setup(self, sym: str) -> None:
        self._pending_setups.pop(sym, None)

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
                "d1_regime": meta.d1_regime_notes,
            })

    # ─── D1 handler ───────────────────────────────────────────────────────

    def _handle_d1(self, bar: Bar, sym: str) -> None:
        """Update D1 indicators, regime, structure, weekly tracker."""
        # Update D1 indicators
        snap = self._d1_inc[sym].update(bar)
        if snap is not None:
            self._d1_indicators[sym] = snap

        # Update structure tracker
        self._structure_trackers[sym].update(bar)

        # Update weekly tracker
        self._weekly_trackers[sym].update(bar)

        # Classify regime
        d1_ind = self._d1_indicators[sym]
        if d1_ind is None:
            log.debug("trend.d1_not_ready", symbol=sym)
            return
        structure = self._structure_trackers[sym].state
        regime = self._regime_classifier.evaluate(bar, d1_ind, structure)
        self._current_regime[sym] = regime
        log.debug("trend.d1_regime", symbol=sym,
                  tier=regime.tier if regime else "none",
                  direction=regime.direction.value if regime and regime.direction else "none")

    # ─── H1 handler (main flow) ───────────────────────────────────────────

    def _handle_h1(self, bar: Bar, sym: str, ctx: StrategyContext) -> None:
        """Main H1 processing — mirrors momentum's _handle_m15."""
        # Update bar count and indicators
        self._h1_bar_count[sym] += 1
        snap = self._h1_inc[sym].update(bar)
        if snap is not None:
            self._h1_indicators[sym] = snap

        # Begin instrumentation bar cycle
        self._collector.begin_bar(sym, bar.close)

        # Warmup
        bar_count = self._h1_bar_count[sym]
        warmup_met = bar_count >= WARMUP_BARS
        self._collector.record_gate(sym, "warmup", warmup_met,
            "insufficient_bars" if not warmup_met else "",
            threshold=WARMUP_BARS, actual_value=bar_count)
        if not warmup_met:
            self._collector.end_bar(sym)
            return

        h1_ind = self._h1_indicators[sym]
        self._collector.record_gate(sym, "indicators", h1_ind is not None,
            "no_snapshot" if h1_ind is None else "")
        if h1_ind is None:
            self._collector.end_bar(sym)
            return

        window_open = self._entry_window_open(bar.timestamp, ctx)
        self._collector.record_gate(sym, "entry_window", window_open,
            "before_measurement_start" if not window_open else "")
        if not window_open:
            self._collector.end_bar(sym)
            return

        # --- Entry gate: skip if position exists ---
        has_pos = ctx.broker.get_position(sym) is not None
        self._collector.record_gate(sym, "position_check", not has_pos,
            "position_exists" if has_pos else "")
        if has_pos:
            self._clear_pending_setup(sym)
            self._collector.end_bar(sym)
            return

        # --- Fetch H1 history ---
        h1_bars = ctx.bars.get(sym, TimeFrame.H1, count=50)
        has_history = bool(h1_bars)
        self._collector.record_gate(sym, "h1_history", has_history,
            "no_h1_bars" if not has_history else "")
        if not has_history:
            self._collector.end_bar(sym)
            return

        # --- Re-entry evaluation ---
        is_reentry = False
        min_conf_override = None
        recent = self._recent_exits.get(sym, {})
        if recent and self._cfg.reentry.enabled:
            bars_since = self._h1_bar_count[sym] - recent.get("bar_idx", 0)
            loss_r = abs(recent.get("loss_r", 0))
            count = self._reentry_count.get(sym, 0)
            reentry_cfg = self._cfg.reentry
            max_wait = max(int(reentry_cfg.max_wait_bars), 0)

            if max_wait > 0 and bars_since > max_wait:
                self._clear_recent_reentry(sym)
                recent = {}
            elif (reentry_cfg.only_after_scratch_exit
                    and recent.get("exit_reason") != "scratch_exit"):
                self._clear_recent_reentry(sym)
                recent = {}
            elif (bars_since >= reentry_cfg.cooldown_bars
                    and loss_r <= reentry_cfg.max_loss_r
                    and count < reentry_cfg.max_reentries):
                is_reentry = True
                min_conf_override = reentry_cfg.min_confluences_override
            else:
                self._collector.record_gate(sym, "reentry_eval", False, "cooldown_or_max_reached")
                self._collector.end_bar(sym)
                return  # Still in cooldown or max reentries reached
        elif recent:
            self._collector.record_gate(sym, "reentry_eval", False, "reentry_disabled")
            self._collector.end_bar(sym)
            return  # Re-entry disabled, skip if recently stopped

        # --- Risk check ---
        equity = ctx.broker.get_equity()
        stopped, stop_reason = self._risk_manager.is_session_stopped(equity, bar.timestamp)
        self._collector.record_gate(sym, "risk_check", not stopped, stop_reason)
        if stopped:
            self._collector.end_bar(sym)
            return

        # --- Regime check ---
        regime = self._current_regime.get(sym)
        if regime is None or regime.tier == "none" or regime.direction is None:
            regime = self._regime_classifier.evaluate_h1(bar.close, h1_ind)
        has_regime = regime is not None and regime.direction is not None
        self._collector.record_gate(sym, "regime", has_regime,
            "no_regime" if not has_regime else "",
            context={"tier": regime.tier, "direction": regime.direction.value} if has_regime else {})
        if not has_regime:
            self._collector.end_bar(sym)
            return

        direction = regime.direction

        if (is_reentry
                and self._cfg.reentry.require_same_direction
                and recent.get("side") is not None
                and recent.get("side") != direction):
            self._collector.record_gate(sym, "reentry_direction", False, "direction_mismatch")
            self._clear_recent_reentry(sym)
            is_reentry = False
            min_conf_override = None

        # --- Symbol direction filter ---
        sf = self._cfg.symbol_filter
        rule = getattr(sf, f"{sym.lower()}_direction", "both")
        dir_ok = not (rule == "disabled" or
                      (rule == "long_only" and direction == Side.SHORT) or
                      (rule == "short_only" and direction == Side.LONG))
        self._collector.record_gate(sym, "symbol_direction", dir_ok,
            f"{rule}_blocks_{direction.value}" if not dir_ok else "",
            context={"rule": rule, "direction": direction.value})
        if not dir_ok:
            self._collector.end_bar(sym)
            return

        # --- Perp / relative strength filters ---
        funding_ok = self._passes_funding_filter(sym, direction, bar.timestamp, ctx)
        self._collector.record_gate(sym, "funding_filter", funding_ok,
            "funding_opposes_direction" if not funding_ok else "")
        if not funding_ok:
            self._collector.end_bar(sym)
            return

        rs_ok = self._passes_relative_strength_filter(sym, direction, h1_bars, ctx)
        self._collector.record_gate(sym, "relative_strength", rs_ok,
            "relative_strength_too_weak" if not rs_ok else "")
        if not rs_ok:
            self._collector.end_bar(sym)
            return

        # --- Setup detection ---
        d1_ind = self._d1_indicators.get(sym)
        weekly_tracker = self._weekly_trackers.get(sym)
        weekly_high = weekly_tracker.prior_week_high if weekly_tracker else None
        weekly_low = weekly_tracker.prior_week_low if weekly_tracker else None

        setup = self._setup_detector.detect(
            h1_bars=h1_bars,
            h1_ind=h1_ind,
            d1_ind=d1_ind,
            regime=regime,
            weekly_high=weekly_high,
            weekly_low=weekly_low,
            min_confluences_override=min_conf_override,
        )
        trigger = None
        setup_source = "fresh"
        if setup is None:
            pending = self._pending_setups.get(sym)
            if pending is not None and self._pending_setup_is_valid(sym, pending, direction):
                setup = pending.setup
                setup_source = "pending_confirmation"
                is_reentry = pending.is_reentry
                min_conf_override = pending.min_confluences_override
                trigger = self._trigger_detector.check(h1_bars, setup.direction, h1_ind)
            elif pending is not None:
                self._clear_pending_setup(sym)

        self._collector.record_gate(sym, "setup", setup is not None,
            "no_setup_detected" if setup is None else "",
            context={"confluences": list(setup.confluences), "grade": setup.grade.value,
                     "room_r": setup.room_r, "source": setup_source} if setup else {})
        if setup is None:
            self._collector.end_bar(sym)
            return

        # Update context with setup info
        self._collector.snapshot_context(sym, h1_ind,
            regime_tier=regime.tier, regime_direction=direction.value,
            setup_grade=setup.grade.value, setup_confluences=list(setup.confluences),
            setup_room_r=setup.room_r, funding_rate=0.0)

        # --- Confirmation ---
        if trigger is None:
            trigger = self._trigger_detector.check(h1_bars, setup.direction, h1_ind)
        confirmation_required = (
            self._cfg.confirmation.require_confirmation
            or (self._cfg.confirmation.require_confirmation_for_b and setup.grade == SetupGrade.B)
        )
        confirm_ok = trigger is not None or not confirmation_required
        self._collector.record_gate(sym, "confirmation", confirm_ok,
            "no_confirmation_pattern" if not confirm_ok else "",
            context={"pattern": trigger.pattern, "source": setup_source} if trigger else {
                "source": setup_source,
                "max_bars_after_setup": self._cfg.confirmation.max_bars_after_setup,
            })
        if not confirm_ok:
            if setup_source == "fresh":
                self._pending_setups[sym] = _PendingTrendSetup(
                    setup=setup,
                    created_h1_bar_index=self._h1_bar_count[sym],
                    regime_tier=regime.tier,
                    regime_reasons=tuple(regime.reasons) if regime else (),
                    is_reentry=is_reentry,
                    min_confluences_override=min_conf_override,
                )
            self._collector.end_bar(sym)
            return
        self._clear_pending_setup(sym)

        # --- Stop placement ---
        stop_level = self._stop_placer.compute(
            h1_bars=h1_bars,
            direction=setup.direction,
            atr=h1_ind.atr,
            entry_price=bar.close,
        )
        stop_distance = abs(bar.close - stop_level)
        stop_valid = stop_distance > 0
        self._collector.record_gate(sym, "stop_validity", stop_valid,
            "zero_stop_distance" if not stop_valid else "")
        if not stop_valid:
            self._collector.end_bar(sym)
            return

        # --- Position sizing ---
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
            symbol=sym,
            open_positions=open_positions,
            direction=setup.direction,
            risk_scale=self._cfg.reentry.risk_scale if is_reentry else 1.0,
        )
        self._collector.record_gate(sym, "sizing", sizing is not None,
            sizing_reason if sizing is None else "",
            context={"risk_pct": sizing.risk_pct_actual, "leverage": sizing.leverage} if sizing else {})
        if sizing is None:
            self._collector.end_bar(sym)
            return

        # --- Entry order ---
        order_seed = {
            "symbol": sym,
            "timeframe": bar.timeframe.value,
            "bar_timestamp": bar.timestamp.isoformat(),
            "direction": setup.direction.value,
            "grade": setup.grade.value,
            "is_reentry": is_reentry,
        }
        order_id = f"trend_entry_{sym}_{stable_hash(order_seed, length=8)}"
        entry_order = self._entry_generator.generate(
            bar=bar,
            direction=setup.direction,
            qty=sizing.qty,
            sizing_result=sizing,
            setup=setup,
            trigger=trigger,
            symbol=sym,
            order_id=order_id,
            is_reentry=is_reentry,
        )
        self._collector.record_gate(sym, "entry_order", entry_order is not None,
            "entry_generation_failed" if entry_order is None else "")
        if entry_order is None:
            self._collector.end_bar(sym)
            return

        baseline_risk_pct = (
            self._cfg.risk.risk_pct_a
            if setup.grade == SetupGrade.A
            else self._cfg.risk.risk_pct_b
        )
        entry_order.metadata["risk_R"] = self._scaled_risk_units(
            sizing.risk_pct_actual,
            baseline_risk_pct,
        )

        # Record entry with full context
        signal_strength = (setup.room_r or 1.0) * (1.0 if setup.grade == SetupGrade.A else 0.7)
        self._collector.record_signal_factor(sym, "setup_room_r", setup.room_r or 0.0)
        self._collector.record_signal_factor(sym, "confluences", len(setup.confluences) / 6.0)
        portfolio_state = self._portfolio_snapshot(ctx, sym, setup.direction)
        self._collector.record_entry(sym, self._cfg.to_dict(),
            sizing_inputs={"risk_pct": sizing.risk_pct_actual, "leverage": sizing.leverage,
                           "stop_distance": stop_distance, "atr": h1_ind.atr,
                           "equity": equity},
            portfolio_state=portfolio_state,
            signal_strength=signal_strength)

        # --- Submit and store meta ---
        ctx.broker.submit_order(entry_order)
        if entry_order.status == OrderStatus.REJECTED:
            self._collector.end_bar(sym)
            return

        self._position_meta[sym] = _PositionMeta(
            setup_grade=setup.grade,
            confluences=setup.confluences,
            confirmation_type=trigger.pattern if trigger else "none",
            entry_method=str(entry_order.metadata.get("entry_method", "aggressive")),
            entry_price=bar.close,
            stop_level=stop_level,
            stop_distance=stop_distance,
            leverage=sizing.leverage,
            liquidation_price=sizing.liquidation_price,
            risk_pct=sizing.risk_pct_actual,
            original_qty=sizing.qty,
            entry_bar_index=self._h1_bar_count[sym],
            entry_m15_bar_index=self._m15_bar_count.get(sym, 0),
            d1_regime_notes=str(regime.reasons) if regime else "",
            is_reentry=is_reentry,
        )

        if is_reentry:
            self._reentry_count[sym] = self._reentry_count.get(sym, 0) + 1

        log.info(
            "trend.entry_submitted",
            symbol=sym,
            direction=setup.direction.value,
            grade=setup.grade.value,
            confluences=list(setup.confluences),
            room_r=f"{setup.room_r:.2f}",
            stop_dist=f"{stop_distance:.2f}",
        )
        self._collector.end_bar(sym)

    def _clear_recent_reentry(self, sym: str) -> None:
        self._recent_exits[sym] = {}
        self._reentry_count[sym] = 0

    def _passes_funding_filter(
        self,
        sym: str,
        direction: Side,
        timestamp: datetime,
        ctx: StrategyContext,
    ) -> bool:
        if not self._cfg.filters.funding_filter_enabled:
            return True

        rate_fn = getattr(ctx.broker, "get_funding_rate", None)
        if not callable(rate_fn):
            return True

        rate = float(rate_fn(sym, int(timestamp.timestamp() * 1000)))
        threshold = float(self._cfg.filters.funding_extreme_threshold)
        if direction == Side.LONG:
            return rate < threshold
        return rate > -threshold

    def _passes_relative_strength_filter(
        self,
        sym: str,
        direction: Side,
        h1_bars: list[Bar],
        ctx: StrategyContext,
    ) -> bool:
        filters = self._cfg.filters
        if not filters.relative_strength_filter_enabled or sym == "BTC":
            return True

        lookback = max(int(filters.relative_strength_lookback), 1)
        if len(h1_bars) < lookback + 1:
            return True

        btc_bars = ctx.bars.get("BTC", TimeFrame.H1, count=lookback + 1)
        if len(btc_bars) < lookback + 1:
            return True

        asset_start = h1_bars[-lookback - 1].close
        btc_start = btc_bars[-lookback - 1].close
        if asset_start <= 0 or btc_start <= 0:
            return True

        asset_ret = (h1_bars[-1].close / asset_start) - 1.0
        btc_ret = (btc_bars[-1].close / btc_start) - 1.0
        delta = asset_ret - btc_ret
        threshold = float(filters.relative_strength_min_delta)

        if direction == Side.LONG:
            return delta >= threshold
        return delta <= -threshold

    # ─── M15 handler (position management) ─────────────────────────────────

    _M15_WARMUP = 4  # 1 hour of M15 bars

    def _handle_m15(self, bar: Bar, sym: str, ctx: StrategyContext) -> None:
        """M15 position management — trail, TP, BE, exits every 15 min."""
        self._m15_bar_count[sym] = self._m15_bar_count.get(sym, 0) + 1
        snap = self._m15_inc[sym].update(bar)
        if snap is not None:
            self._m15_indicators[sym] = snap
        if self._m15_bar_count[sym] < self._M15_WARMUP:
            return
        pos = ctx.broker.get_position(sym)
        if pos is None:
            return
        meta = self._position_meta.get(sym)
        if meta is None:
            return
        h1_bars = ctx.bars.get(sym, TimeFrame.H1, count=50)
        self._manage_positions(bar, sym, ctx, h1_bars)

    # ─── Position management ──────────────────────────────────────────────

    def _manage_positions(
        self, bar: Bar, sym: str, ctx: StrategyContext, h1_bars: list[Bar],
    ) -> None:
        """Manage exits and trail for existing positions."""
        pos = ctx.broker.get_position(sym)
        if pos is None:
            return

        meta = self._position_meta.get(sym)
        if meta is None:
            return

        h1_ind = self._h1_indicators.get(sym)
        if h1_ind is None:
            return

        # Exit management — returns orders to submit
        orders = self._exit_manager.manage(pos, bar, h1_bars, h1_ind, ctx.broker)
        for order in orders:
            ctx.broker.submit_order(order)

        # Check if remaining quantity is 0 after partial exits
        exit_state = self._exit_manager.get_state(sym)
        if exit_state is None:
            return

        remaining_qty = pos.qty  # Current broker position qty

        # Smart BE — only apply if current stop is worse than BE price
        if exit_state.be_moved:
            be_price = self._exit_manager.get_be_price(sym)
            if be_price is not None and meta.stop_order_id is not None:
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
                        log.warning("strategy.cancel_failed", symbol=sym, order_id=meta.stop_order_id, context="smart_be")
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
            bars_since = self._m15_bar_count.get(sym, 0) - meta.entry_m15_bar_index

            new_stop = self._trail_manager.update(
                sym=sym,
                direction=exit_state.direction,
                h1_bars=h1_bars,
                h1_ind=h1_ind,
                current_stop=current_stop,
                bars_since_entry=bars_since,
                current_r=exit_state.current_r,
                mfe_r=exit_state.mfe_r,
            )

            if new_stop is not None and meta.stop_order_id is not None:
                cancelled = ctx.broker.cancel_order(meta.stop_order_id)
                if not cancelled:
                    log.warning("strategy.cancel_failed", symbol=sym, order_id=meta.stop_order_id, context="trail_resubmit")
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

    # ─── Fill handlers ────────────────────────────────────────────────────

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
        self._exit_manager.init_position(
            sym=sym,
            entry_price=fill.fill_price,
            stop_distance=meta.stop_distance,
            qty=meta.original_qty,
            direction=fill.side,
            stop_order_id=stop_id,
        )

        log.info(
            "trend.entry_filled",
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

        # Cancel old stop and resubmit with reduced qty
        if meta.stop_order_id:
            cancelled = ctx.broker.cancel_order(meta.stop_order_id)
            if not cancelled:
                log.warning("strategy.cancel_failed", symbol=sym, order_id=meta.stop_order_id, context="tp_fill_stop_resubmit")

        # Determine stop price — use BE if moved, else current stop
        exit_state = self._exit_manager.get_state(sym)
        stop_price = meta.stop_level
        if exit_state and exit_state.be_moved:
            be = self._exit_manager.get_be_price(sym)
            if be is not None:
                stop_price = be

        new_stop_id = self._management_order_id("stop", sym, {
            "fill_id": fill.exchange_fill_id or fill.order_id,
            "order_id": fill.order_id,
            "timestamp": fill.timestamp.isoformat(),
            "side": fill.side.value,
            "qty": remaining_qty,
            "stop_price": stop_price,
            "previous_stop_order_id": meta.stop_order_id,
        })
        # fill.side is the exit side (opposite of position direction) —
        # the protective stop uses the same side to close the remainder
        stop_order = Order(
            order_id=new_stop_id,
            symbol=sym,
            side=fill.side,
            order_type=OrderType.STOP,
            qty=remaining_qty,
            stop_price=stop_price,
            tag="protective_stop",
        )
        ctx.broker.submit_order(stop_order)
        meta.stop_order_id = new_stop_id

    # ─── Position closed event ────────────────────────────────────────────

    def _on_position_closed(self, event: PositionClosedEvent) -> None:
        """Enrich trade with strategy metadata on position close."""
        trade = event.trade
        sym = trade.symbol
        meta = self._position_meta.pop(sym, None)

        # Clean up exit/trail state
        exit_state = self._exit_manager.remove_position(sym)
        self._trail_manager.remove(sym)

        # Enrich trade
        if meta is not None:
            trade.setup_grade = meta.setup_grade
            trade.confluences_used = list(meta.confluences)
            trade.confirmation_type = meta.confirmation_type
            trade.entry_method = meta.entry_method

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
                "bar_idx": self._h1_bar_count.get(sym, 0),
                "side": trade.direction,
                "loss_r": abs(loss_r),
                "exit_reason": trade.exit_reason,
            }
        else:
            self._clear_recent_reentry(sym)

        # Record in risk manager
        self._risk_manager.record_trade(
            trade.net_pnl,
            trade.exit_time if trade.exit_time else trade.entry_time,
        )

        # Record in journal
        context = {}
        if meta:
            context = {
                "leverage": meta.leverage,
                "risk_pct": meta.risk_pct,
                "d1_regime": meta.d1_regime_notes,
            }
        self._journal.record(trade, context)

        log.info(
            "trend.trade_closed",
            symbol=sym,
            pnl=f"{trade.net_pnl:.2f}",
            r=f"{trade.r_multiple:.2f}" if trade.r_multiple is not None else "n/a",
            exit_reason=trade.exit_reason,
        )
