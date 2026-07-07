"""MomentumStrategy orchestrator — implements the Strategy protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    Position,
    SetupGrade,
    Side,
    TerminalMark,
    TimeFrame,
    Trade,
)
from crypto_trader.strategy.momentum.bias import BiasDetector, BiasResult
from crypto_trader.strategy.momentum.config import MomentumConfig
from crypto_trader.strategy.momentum.confirmation import ConfirmationDetector
from crypto_trader.strategy.momentum.entry import EntrySignal
from crypto_trader.strategy.momentum.exits import ExitManager, PositionExitState
from crypto_trader.strategy.momentum.filters import EnvironmentFilter
from crypto_trader.strategy.momentum.indicators import (
    IncrementalIndicators,
    IndicatorSnapshot,
    compute_indicators,
)
from crypto_trader.strategy.momentum.journal import TradeJournal
from crypto_trader.strategy.momentum.risk import SessionRiskManager
from crypto_trader.strategy.momentum.setup import SetupDetector
from crypto_trader.strategy.momentum.sizing import PositionSizer
from crypto_trader.strategy.momentum.stops import StopPlacer
from crypto_trader.strategy.momentum.trail import TrailManager
from crypto_trader.strategy.snapshot import dataclass_from_plain, to_plain
from crypto_trader.instrumentation.collector import InstrumentationCollector
from crypto_trader.instrumentation.lineage import stable_hash
from crypto_trader.instrumentation.quality import ProcessQualityScorer

log = structlog.get_logger()

WARMUP_BARS = 200
EXIT_STOP_TAGS = frozenset({"protective_stop", "breakeven_stop", "proof_lock_stop", "trailing_stop"})


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
    stop_order_id: str | None = None
    h4_bias_notes: str = ""
    h1_trend_notes: str = ""


class MomentumStrategy:
    """Momentum pullback strategy — implements Strategy protocol."""

    def __init__(self, config: MomentumConfig | None = None, bot_id: str = "") -> None:
        self._cfg = config or MomentumConfig()

        # Module instances
        self._bias_detector = BiasDetector(self._cfg.bias)
        self._setup_detector = SetupDetector(self._cfg.setup)
        self._confirmation_detector = ConfirmationDetector(self._cfg.confirmation)
        self._entry_signal = EntrySignal(self._cfg.entry)
        self._sizer = PositionSizer(self._cfg.risk)
        self._stop_placer = StopPlacer(self._cfg.stops)
        self._exit_manager = ExitManager(self._cfg.exits)
        self._trail_manager = TrailManager(self._cfg.trail)
        self._env_filter = EnvironmentFilter(self._cfg.filters, self._cfg.session)
        self._risk_manager = SessionRiskManager(self._cfg.daily_limits)
        self._journal = TradeJournal()

        # Instrumentation
        self._collector = InstrumentationCollector(strategy_id="momentum", bot_id=bot_id)
        self._quality_scorer = ProcessQualityScorer()

        # Per-symbol state
        self._position_meta: dict[str, _PositionMeta] = {}
        self._m15_bar_count: dict[str, int] = {}
        self._current_bias: dict[str, BiasResult | None] = {}
        self._m15_indicators: dict[str, IndicatorSnapshot | None] = {}
        self._h1_indicators: dict[str, IndicatorSnapshot | None] = {}
        self._h4_indicators: dict[str, IndicatorSnapshot | None] = {}

        # Re-entry tracking
        self._recent_exits: dict[str, dict] = {}   # sym -> {bar_idx, side, loss_r}
        self._reentry_count: dict[str, int] = {}   # sym -> count this trend

        self._ctx: StrategyContext | None = None

    @property
    def name(self) -> str:
        return "momentum_pullback"

    @property
    def symbols(self) -> list[str]:
        return self._cfg.symbols

    @property
    def timeframes(self) -> list[TimeFrame]:
        return [TimeFrame.M15, TimeFrame.H1, TimeFrame.H4]

    @property
    def journal(self) -> TradeJournal:
        return self._journal

    @staticmethod
    def _management_order_id(purpose: str, symbol: str, seed: dict[str, object]) -> str:
        order_seed = {
            "strategy": "momentum",
            "purpose": purpose,
            "symbol": symbol,
            **seed,
        }
        return f"mom_{purpose}_{symbol}_{stable_hash(order_seed, length=8)}"

    def _ensure_exit_order_id(
        self,
        order: Order,
        *,
        bar: Bar,
        position: Position,
        state: PositionExitState | None,
        order_index: int,
        confirmation_type: str | None,
    ) -> None:
        if order.order_id:
            return
        purpose = str(order.tag or order.order_type.value or "exit").lower()
        seed: dict[str, object] = {
            "bar_timestamp": bar.timestamp.isoformat(),
            "timeframe": bar.timeframe.value,
            "position_direction": position.direction.value,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "qty": order.qty,
            "limit_price": order.limit_price,
            "stop_price": order.stop_price,
            "tag": order.tag,
            "order_index": order_index,
            "confirmation_type": confirmation_type or "",
        }
        if state is not None:
            seed.update({
                "entry_price": state.entry_price,
                "stop_distance": state.stop_distance,
                "remaining_qty": state.remaining_qty,
                "bars_since_entry": state.bars_since_entry,
                "mfe_r": state.mfe_r,
                "mae_r": state.mae_r,
                "current_stop_order_id": state.current_stop_order_id or "",
                "current_stop_price": state.current_stop_price,
                "current_stop_tag": state.current_stop_tag,
                "tp1_hit": state.tp1_hit,
                "tp2_hit": state.tp2_hit,
                "be_moved": state.be_moved,
                "proof_lock_moved": state.proof_lock_moved,
            })
        order.order_id = self._management_order_id(purpose, order.symbol, seed)
        order.metadata.setdefault("client_order_id", order.order_id)

    def snapshot_state(self) -> dict:
        return {
            "position_meta": to_plain(self._position_meta),
            "exit_states": to_plain(getattr(self._exit_manager, "_states", {})),
            "trail_stops": to_plain(getattr(self._trail_manager, "_current_stops", {})),
            "recent_exits": to_plain(self._recent_exits),
            "reentry_count": to_plain(self._reentry_count),
        }

    def restore_state(self, snapshot: dict) -> None:
        self._position_meta = {
            sym: dataclass_from_plain(_PositionMeta, data)
            for sym, data in snapshot.get("position_meta", {}).items()
        }
        self._exit_manager._states = {
            sym: dataclass_from_plain(PositionExitState, data)
            for sym, data in snapshot.get("exit_states", {}).items()
        }
        self._trail_manager._current_stops = dict(snapshot.get("trail_stops", {}))
        self._recent_exits = dict(snapshot.get("recent_exits", {}))
        self._reentry_count = {
            sym: int(count) for sym, count in snapshot.get("reentry_count", {}).items()
        }

    def on_init(self, ctx: StrategyContext) -> None:
        self._ctx = ctx
        for sym in self._cfg.symbols:
            self._m15_bar_count[sym] = 0
            self._current_bias[sym] = None
            self._m15_indicators[sym] = None
            self._h1_indicators[sym] = None
            self._h4_indicators[sym] = None

        # Incremental indicator state — O(1) per bar instead of O(window)
        p = self._cfg.indicators
        self._inc: dict[str, dict[TimeFrame, IncrementalIndicators]] = {
            sym: {
                TimeFrame.M15: IncrementalIndicators(p),
                TimeFrame.H1: IncrementalIndicators(p),
                TimeFrame.H4: IncrementalIndicators(p),
            }
            for sym in self._cfg.symbols
        }

        # Subscribe to PositionClosedEvent for trade enrichment
        ctx.events.subscribe(PositionClosedEvent, self._on_position_closed)
        log.info("strategy.init", name=self.name, symbols=self._cfg.symbols)

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        sym = bar.symbol
        if sym not in self._cfg.symbols:
            return

        if bar.timeframe == TimeFrame.H4:
            self._handle_h4(bar, ctx)
        elif bar.timeframe == TimeFrame.H1:
            self._handle_h1(bar, ctx)
        elif bar.timeframe == TimeFrame.M15:
            self._handle_m15(bar, ctx)

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None:
        sym = fill.symbol
        if sym not in self._cfg.symbols:
            return

        if fill.tag == "entry":
            meta = self._position_meta.get(sym)
            if meta:
                meta.entry_price = fill.fill_price
                meta.original_qty = fill.qty
                # Submit protective stop
                stop_price = meta.stop_level
                stop_dist = abs(fill.fill_price - stop_price) if stop_price else 0
                if stop_dist > 0:
                    meta.stop_distance = stop_dist
                    close_side = Side.SHORT if fill.side == Side.LONG else Side.LONG
                    stop_id = self._management_order_id("stop", sym, {
                        "fill_id": fill.exchange_fill_id or fill.order_id,
                        "order_id": fill.order_id,
                        "timestamp": fill.timestamp.isoformat(),
                        "side": fill.side.value,
                        "qty": fill.qty,
                        "stop_price": stop_price,
                    })
                    stop_order = Order(
                        order_id=stop_id,
                        symbol=sym,
                        side=close_side,
                        order_type=OrderType.STOP,
                        qty=fill.qty,
                        stop_price=stop_price,
                        tag="protective_stop",
                    )
                    oid = ctx.broker.submit_order(stop_order)
                    meta.stop_order_id = oid
                    # Initialize exit manager state
                    self._exit_manager.init_position(
                        sym,
                        fill.fill_price,
                        stop_dist,
                        fill.qty,
                        oid,
                        stop_price=stop_price,
                        stop_tag="protective_stop",
                    )
                    log.info(
                        "strategy.entry_filled",
                        symbol=sym, price=fill.fill_price,
                        stop=stop_price, qty=fill.qty,
                    )

        elif fill.tag in EXIT_STOP_TAGS:
            # Stop was hit — position closing handled by SimBroker
            pass

        elif fill.tag in ("tp1", "tp2"):
            # Partial exit filled — update stop order with reduced qty
            meta = self._position_meta.get(sym)
            state = self._exit_manager.get_state(sym)
            if meta and state and state.current_stop_order_id and state.remaining_qty > 0:
                # Cancel and resubmit with new qty
                cancelled = ctx.broker.cancel_order(state.current_stop_order_id)
                if not cancelled:
                    log.warning("strategy.cancel_failed", symbol=sym, order_id=state.current_stop_order_id, context="tp_fill_stop_resubmit")
                stop_price = state.current_stop_price
                stop_tag = state.current_stop_tag or (
                    "breakeven_stop" if state.be_moved else "protective_stop"
                )
                if stop_price <= 0:
                    if state.be_moved:
                        if fill.side == Side.SHORT:
                            # closing side SHORT means position is LONG
                            stop_price = (
                                meta.entry_price
                                + self._cfg.exits.be_buffer_r * meta.stop_distance
                            )
                        else:
                            stop_price = (
                                meta.entry_price
                                - self._cfg.exits.be_buffer_r * meta.stop_distance
                            )
                        stop_tag = "breakeven_stop"
                    else:
                        stop_price = meta.stop_level
                        stop_tag = "protective_stop"
                previous_stop_order_id = state.current_stop_order_id
                stop_purpose = (
                    "be" if stop_tag == "breakeven_stop"
                    else "trail" if stop_tag == "trailing_stop"
                    else "stop"
                )
                new_stop_id = self._management_order_id(stop_purpose, sym, {
                    "fill_id": fill.exchange_fill_id or fill.order_id,
                    "order_id": fill.order_id,
                    "timestamp": fill.timestamp.isoformat(),
                    "side": fill.side.value,
                    "qty": state.remaining_qty,
                    "stop_price": stop_price,
                    "previous_stop_order_id": previous_stop_order_id,
                })
                new_stop = Order(
                    order_id=new_stop_id,
                    symbol=sym,
                    side=fill.side,  # closing side (same direction as the TP fill)
                    order_type=OrderType.STOP,
                    qty=state.remaining_qty,
                    stop_price=stop_price,
                    tag=stop_tag,
                )
                oid = ctx.broker.submit_order(new_stop)
                state.current_stop_order_id = oid
                state.current_stop_price = stop_price
                state.current_stop_tag = stop_tag

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

    def _entry_window_open(self, bar: Bar, ctx: StrategyContext) -> bool:
        measurement_start = self._measurement_start(ctx)
        return measurement_start is None or bar.timestamp >= measurement_start

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
                "h4_bias_notes": meta.h4_bias_notes,
                "h1_trend_notes": meta.h1_trend_notes,
            })

    # ------------------------------------------------------------------
    # Timeframe handlers
    # ------------------------------------------------------------------

    def _handle_h4(self, bar: Bar, ctx: StrategyContext) -> None:
        sym = bar.symbol
        self._h4_indicators[sym] = self._inc[sym][TimeFrame.H4].update(bar)
        self._update_bias(sym, ctx)

    def _handle_h1(self, bar: Bar, ctx: StrategyContext) -> None:
        sym = bar.symbol
        self._h1_indicators[sym] = self._inc[sym][TimeFrame.H1].update(bar)
        self._update_bias(sym, ctx)

    def _handle_m15(self, bar: Bar, ctx: StrategyContext) -> None:
        sym = bar.symbol
        self._m15_bar_count[sym] = self._m15_bar_count.get(sym, 0) + 1

        # Current bar is already in ctx.bars (engine appends before on_bar)
        m15_bars = ctx.bars.get(sym, TimeFrame.M15)

        # O(1) incremental indicator update
        m15_ind = self._inc[sym][TimeFrame.M15].update(bar)
        self._m15_indicators[sym] = m15_ind

        # Begin instrumentation bar cycle
        self._collector.begin_bar(sym, bar.close)

        if m15_ind is None:
            self._collector.record_gate(sym, "indicators", False, "no_snapshot")
            self._collector.end_bar(sym)
            return
        self._collector.record_gate(sym, "indicators", True)

        # Warmup check
        bar_count = self._m15_bar_count[sym]
        warmup_met = bar_count >= WARMUP_BARS
        self._collector.record_gate(sym, "warmup", warmup_met,
            "insufficient_bars" if not warmup_met else "",
            threshold=WARMUP_BARS, actual_value=bar_count)
        if not warmup_met:
            self._collector.end_bar(sym)
            return

        # 1. Manage existing positions
        self._manage_positions(bar, ctx, m15_bars, m15_ind)

        # 2. Check if we already have a position in this symbol
        has_pos = ctx.broker.get_position(sym) is not None
        self._collector.record_gate(sym, "position_check", not has_pos,
            "position_exists" if has_pos else "")
        if has_pos:
            self._collector.end_bar(sym)
            return

        # Warmup bars may build state, but entries start at the measurement boundary.
        window_open = self._entry_window_open(bar, ctx)
        self._collector.record_gate(sym, "entry_window", window_open,
            "before_measurement_start" if not window_open else "")
        if not window_open:
            self._collector.end_bar(sym)
            return

        # 2.5 Re-entry evaluation
        is_reentry = False
        if self._cfg.reentry.enabled and sym in self._recent_exits:
            re = self._recent_exits[sym]
            bars_since_exit = self._m15_bar_count[sym] - re["bar_idx"]
            bias = self._current_bias.get(sym)
            # Clear re-entry tracking if bias flipped (new trend)
            if bias and bias.direction is not None and bias.direction != re["side"]:
                del self._recent_exits[sym]
                self._reentry_count.pop(sym, None)
            elif (bars_since_exit >= self._cfg.reentry.cooldown_bars
                  and re["loss_r"] <= self._cfg.reentry.max_loss_r
                  and self._reentry_count.get(sym, 0) < self._cfg.reentry.max_reentries):
                is_reentry = True

        # 3. Risk manager check
        stopped, stop_reason = self._risk_manager.is_session_stopped(ctx.broker.get_equity(), ctx.clock.now())
        self._collector.record_gate(sym, "risk_check", not stopped, stop_reason)
        if stopped:
            self._collector.end_bar(sym)
            return

        # 4. Environment filter
        funding_rate = 0.0  # Would come from FundingHelper in live
        filter_result = self._env_filter.check(m15_ind, funding_rate, ctx.clock.now())
        self._collector.record_gate(sym, "env_filter", filter_result.allowed,
            "; ".join(filter_result.reasons) if not filter_result.allowed else "")
        if not filter_result.allowed:
            self._collector.end_bar(sym)
            return

        # 5. Bias check
        bias = self._current_bias.get(sym)
        has_bias = bias is not None and bias.direction is not None
        self._collector.record_gate(sym, "bias", has_bias,
            "no_bias" if not has_bias else "")
        if not has_bias:
            self._collector.end_bar(sym)
            return

        # 5.5 Symbol direction filter
        direction = bias.direction
        sf = self._cfg.symbol_filter
        rule = getattr(sf, f"{sym.lower().replace('usdt', '')}_direction", "both")
        dir_ok = not (rule == "disabled" or
                      (rule == "long_only" and direction == Side.SHORT) or
                      (rule == "short_only" and direction == Side.LONG))
        self._collector.record_gate(sym, "symbol_direction", dir_ok,
            f"{rule}_blocks_{direction.value}" if not dir_ok else "",
            context={"rule": rule, "direction": direction.value})
        if not dir_ok:
            self._collector.end_bar(sym)
            return

        # 6. Setup detection (relaxed confluence for re-entries)
        h1_bars = ctx.bars.get(sym, TimeFrame.H1)
        min_conf = self._cfg.reentry.min_confluences_override if is_reentry else None
        setup = self._setup_detector.detect(m15_bars, h1_bars, m15_ind, bias, min_confluences_override=min_conf)
        self._collector.record_gate(sym, "setup", setup is not None,
            "no_setup_detected" if setup is None else "",
            context={"confluences": list(setup.confluences), "grade": setup.grade.value,
                     "room_r": setup.room_r} if setup else {})
        if setup is None:
            self._collector.end_bar(sym)
            return

        # Update context with setup info
        self._collector.snapshot_context(sym, m15_ind,
            bias_direction=bias.direction.value, bias_strength=bias.confidence,
            setup_grade=setup.grade.value, setup_confluences=list(setup.confluences),
            setup_room_r=setup.room_r, funding_rate=funding_rate)

        # 7. A-grade filter
        if filter_result.require_a_grade:
            is_a = setup.grade == SetupGrade.A
            self._collector.record_gate(sym, "a_grade_filter", is_a,
                "b_grade_in_adverse_env" if not is_a else "")
            if not is_a:
                self._collector.end_bar(sym)
                return

        # 8. Confirmation
        confirmation = self._confirmation_detector.check(
            m15_bars,
            setup.zone_price,
            bias.direction,
            m15_ind.volume_ma,
            m15_ind.atr,
        )
        self._collector.record_gate(sym, "confirmation", confirmation is not None,
            "no_confirmation_pattern" if confirmation is None else "",
            context={"pattern": confirmation.pattern_type} if confirmation else {})
        if confirmation is None:
            self._collector.end_bar(sym)
            return

        # 8.5 Confluence gate for weak confirmations (skip for re-entries)
        if not is_reentry and confirmation.pattern_type in self._cfg.confirmation.weak_confirmations:
            n_conf = len(setup.confluences)
            gate_ok = n_conf >= self._cfg.confirmation.min_confluences_for_weak
            self._collector.record_gate(sym, "confluence_gate", gate_ok,
                "weak_confirmation_insufficient_confluences" if not gate_ok else "",
                threshold=self._cfg.confirmation.min_confluences_for_weak,
                actual_value=n_conf)
            if not gate_ok:
                self._collector.end_bar(sym)
                return

        if self._cfg.confirmation.enforce_volume_on_trigger and not confirmation.volume_confirmed:
            self._collector.record_gate(sym, "volume_on_trigger", False, "no_volume_confirmation")
            self._collector.end_bar(sym)
            return
        if (
            self._cfg.confirmation.enforce_volume_on_weak_confirmations
            and confirmation.pattern_type in self._cfg.confirmation.weak_confirmations
            and not confirmation.volume_confirmed
        ):
            self._collector.record_gate(sym, "volume_on_weak", False, "weak_no_volume")
            self._collector.end_bar(sym)
            return

        # 9. Stop placement
        stop_price = self._stop_placer.compute(
            m15_bars, bias.direction, m15_ind.atr, sym,
        )

        # 10. Position sizing
        entry_price_est = self._entry_signal.estimate_entry_price(confirmation, bar.close)
        stop_dist = abs(entry_price_est - stop_price)
        stop_valid = stop_dist > 0
        self._collector.record_gate(sym, "stop_validity", stop_valid,
            "zero_stop_distance" if not stop_valid else "")
        if not stop_valid:
            self._collector.end_bar(sym)
            return

        sizing, sizing_reason = self._sizer.compute(
            equity=ctx.broker.get_equity(),
            entry_price=entry_price_est,
            stop_distance=stop_dist,
            setup_grade=setup.grade,
            symbol=sym,
            open_positions=ctx.broker.get_positions(),
            direction=bias.direction,
        )
        self._collector.record_gate(sym, "sizing", sizing is not None,
            sizing_reason,
            context={"risk_pct": sizing.risk_pct_actual, "leverage": sizing.leverage} if sizing else {})
        if sizing is None:
            self._collector.end_bar(sym)
            return

        # 11. Generate entry order
        entry_order = self._entry_signal.generate(
            setup=setup,
            confirmation=confirmation,
            indicators=m15_ind,
            sizing=sizing,
            direction=bias.direction,
            symbol=sym,
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
        self._collector.record_signal_factor(sym, "confirmation_volume",
            1.0 if confirmation.volume_confirmed else 0.5)
        portfolio_state = self._portfolio_snapshot(ctx, sym, bias.direction)
        self._collector.record_entry(sym, self._cfg.to_dict(),
            sizing_inputs={"risk_pct": sizing.risk_pct_actual, "leverage": sizing.leverage,
                           "stop_distance": stop_dist, "atr": m15_ind.atr,
                           "equity": ctx.broker.get_equity()},
            portfolio_state=portfolio_state,
            signal_strength=signal_strength)

        # 12. Submit and store metadata
        oid = ctx.broker.submit_order(entry_order)
        if entry_order.status == OrderStatus.REJECTED:
            self._collector.end_bar(sym)
            return
        self._position_meta[sym] = _PositionMeta(
            setup_grade=setup.grade,
            confluences=setup.confluences,
            confirmation_type=confirmation.pattern_type,
            entry_method=str(entry_order.metadata.get("entry_method", "")),
            stop_level=stop_price,
            stop_distance=stop_dist,
            leverage=sizing.leverage,
            liquidation_price=sizing.liquidation_price,
            risk_pct=sizing.risk_pct_actual,
            original_qty=sizing.qty,
            entry_bar_index=self._m15_bar_count[sym],
            h4_bias_notes="; ".join(bias.reasons),
        )
        if is_reentry:
            self._reentry_count[sym] = self._reentry_count.get(sym, 0) + 1
        log.info(
            "strategy.entry_submitted",
            symbol=sym, direction=bias.direction.value,
            grade=setup.grade.value, confluences=setup.confluences,
            confirmation=confirmation.pattern_type,
            reentry=is_reentry,
        )
        self._collector.end_bar(sym)

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _manage_positions(
        self,
        bar: Bar,
        ctx: StrategyContext,
        m15_bars: list[Bar],
        m15_ind: IndicatorSnapshot,
    ) -> None:
        for pos in ctx.broker.get_positions():
            if pos.symbol != bar.symbol:
                continue

            meta = self._position_meta.get(pos.symbol)
            confirmation_type = meta.confirmation_type if meta else None

            # Exit manager
            exit_orders = self._exit_manager.manage(
                pos,
                bar,
                m15_bars,
                m15_ind,
                ctx.broker,
                confirmation_type=confirmation_type,
            )
            state = self._exit_manager.get_state(pos.symbol)
            for order_index, order in enumerate(exit_orders):
                self._ensure_exit_order_id(
                    order,
                    bar=bar,
                    position=pos,
                    state=state,
                    order_index=order_index,
                    confirmation_type=confirmation_type,
                )
                oid = ctx.broker.submit_order(order)
                # Track new stop order ID
                if state and order.tag in EXIT_STOP_TAGS:
                    state.current_stop_order_id = oid

            # Trail manager
            state = self._exit_manager.get_state(pos.symbol)
            current_stop = None
            if state and state.current_stop_order_id:
                # Find current stop price from open orders
                for o in ctx.broker.get_open_orders(pos.symbol):
                    if o.order_id == state.current_stop_order_id:
                        current_stop = o.stop_price
                        break

            # Compute activation data for trail manager
            bars_since_entry = 0
            current_r = 0.0
            if meta:
                bars_since_entry = self._m15_bar_count.get(pos.symbol, 0) - meta.entry_bar_index
                stop_dist = abs(meta.entry_price - meta.stop_level)
                if stop_dist > 0 and meta.entry_price > 0:
                    if pos.direction == Side.LONG:
                        current_r = (bar.close - meta.entry_price) / stop_dist
                    else:
                        current_r = (meta.entry_price - bar.close) / stop_dist

            mfe_r = state.mfe_r if state else 0.0
            new_trail = self._trail_manager.update(
                pos, m15_bars, m15_ind, current_stop,
                bars_since_entry=bars_since_entry,
                current_r=current_r,
                mfe_r=mfe_r,
                confirmation_type=confirmation_type,
            )
            if new_trail is not None and state and state.remaining_qty > 0:
                # Cancel and resubmit stop at trail level
                if state.current_stop_order_id:
                    cancelled = ctx.broker.cancel_order(state.current_stop_order_id)
                    if not cancelled:
                        log.warning("strategy.cancel_failed", symbol=pos.symbol, order_id=state.current_stop_order_id, context="trail_resubmit")
                close_side = Side.SHORT if pos.direction == Side.LONG else Side.LONG
                trail_stop_id = self._management_order_id("trail", pos.symbol, {
                    "bar_timestamp": bar.timestamp.isoformat(),
                    "timeframe": bar.timeframe.value,
                    "direction": pos.direction.value,
                    "qty": state.remaining_qty,
                    "stop_price": new_trail,
                    "previous_stop_order_id": state.current_stop_order_id,
                    "bars_since_entry": bars_since_entry,
                    "confirmation_type": confirmation_type,
                })
                trail_stop = Order(
                    order_id=trail_stop_id,
                    symbol=pos.symbol,
                    side=close_side,
                    order_type=OrderType.STOP,
                    qty=state.remaining_qty,
                    stop_price=new_trail,
                    tag="trailing_stop",
                )
                oid = ctx.broker.submit_order(trail_stop)
                state.current_stop_order_id = oid
                state.current_stop_price = new_trail
                state.current_stop_tag = "trailing_stop"

    # ------------------------------------------------------------------
    # Bias
    # ------------------------------------------------------------------

    def _update_bias(self, sym: str, ctx: StrategyContext) -> None:
        h4_bars = ctx.bars.get(sym, TimeFrame.H4)
        h1_bars = ctx.bars.get(sym, TimeFrame.H1)

        self._current_bias[sym] = self._bias_detector.compute(
            h4_bars, h1_bars,
            self._h4_indicators.get(sym),
            self._h1_indicators.get(sym),
        )

    # ------------------------------------------------------------------
    # Trade enrichment
    # ------------------------------------------------------------------

    def _on_position_closed(self, event: PositionClosedEvent) -> None:
        trade = event.trade
        sym = trade.symbol
        meta = self._position_meta.pop(sym, None)
        exit_state = self._exit_manager.remove_position(sym)
        self._trail_manager.remove(sym)

        if meta:
            # Enrich trade
            trade.setup_grade = meta.setup_grade
            trade.confluences_used = list(meta.confluences)
            trade.confirmation_type = meta.confirmation_type
            trade.entry_method = meta.entry_method

            # R-multiple
            if meta.stop_distance > 0:
                if trade.direction == Side.LONG:
                    trade.r_multiple = (trade.exit_price - trade.entry_price) / meta.stop_distance
                else:
                    trade.r_multiple = (trade.entry_price - trade.exit_price) / meta.stop_distance
                initial_risk = trade.qty * meta.stop_distance
                if initial_risk > 0:
                    trade.realized_r_multiple = trade.net_pnl / initial_risk

            # MAE/MFE from exit state
            if exit_state:
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

            # Record exit for re-entry tracking after R is known.
            if self._cfg.reentry.enabled:
                self._recent_exits[sym] = {
                    "bar_idx": self._m15_bar_count.get(sym, 0),
                    "side": trade.direction,
                    "loss_r": abs(min(0.0, trade.r_multiple or 0.0)),
                }

            # Record trade PnL for risk manager
            self._risk_manager.record_trade(
                trade.net_pnl,
                trade.exit_time if trade.exit_time else trade.entry_time,
            )

            # Journal
            context = {
                "stop_price": meta.stop_level,
                "liquidation_price": meta.liquidation_price,
                "leverage": meta.leverage,
                "risk_pct": meta.risk_pct,
                "confluences": meta.confluences,
                "exit_distribution": exit_state.partial_exits if exit_state else [],
                "h4_bias_notes": meta.h4_bias_notes,
                "h1_trend_notes": meta.h1_trend_notes,
            }
            self._journal.record(trade, context)
            log.info(
                "strategy.trade_closed",
                symbol=sym, pnl=f"{trade.net_pnl:.2f}",
                r=f"{trade.r_multiple:.2f}" if trade.r_multiple is not None else "N/A",
                grade=meta.setup_grade.value if meta.setup_grade else "N/A",
            )
