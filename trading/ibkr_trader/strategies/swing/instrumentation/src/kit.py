"""InstrumentationKit — facade for clean instrumentation API.

Wraps InstrumentationContext with a simple 3-method interface:
- log_entry(...) — capture entry with enriched fields
- log_exit(...) — capture exit and auto-score
- log_missed(...) — capture blocked signal
- classify_regime(...) — get market regime
- capture_snapshot(...) — capture market data

All methods swallow exceptions using safe_instrument pattern.
Never crashes trading.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
from pathlib import Path

from .context import InstrumentationContext
from .hooks import safe_instrument

logger = logging.getLogger("instrumentation.kit")


class InstrumentationKit:
    """Central facade for all instrumentation operations.

    Usage::

        ctx = InstrumentationContext(...)
        kit = InstrumentationKit(ctx, strategy_id="ATRSS")

        trade = kit.log_entry(
            trade_id="t1",
            pair="BTC/USDT",
            side="LONG",
            entry_price=50000,
            position_size=1.0,
            position_size_quote=50000,
            entry_signal="EMA cross",
            entry_signal_id="ema_123",
            entry_signal_strength=0.8,
            active_filters=["volume"],
            passed_filters=["volume"],
            strategy_params={"ema_fast": 12},
            signal_factors=[{"factor": "momentum", "value": 0.75}],
            filter_decisions=[{"filter": "volume", "current": 1000000, "threshold": 500000}],
            sizing_inputs={"risk_pct": 1.0, "atr": 500},
            portfolio_state_at_entry={"total_exposure": 0.5, "positions": 3},
        )

        kit.log_exit(
            trade_id="t1",
            exit_price=51000,
            exit_reason="TAKE_PROFIT",
            fees_paid=50,
        )
    """

    def __init__(self, ctx: InstrumentationContext, strategy_id: str):
        """Initialize the kit with context and strategy ID.

        Args:
            ctx: InstrumentationContext with all services
            strategy_id: Strategy identifier (e.g. "ATRSS", "AKC_HELIX")
        """
        self.ctx = ctx
        self.strategy_id = strategy_id

        # ConfigWatcher — initialized lazily on first check_config_changes()
        self._config_watcher = None
        try:
            from .config_watcher import ConfigWatcher
            experiments_path = Path(ctx.data_dir).parent / "config" / "experiments.yaml"
            yaml_paths = [experiments_path] if experiments_path.exists() else []
            self._config_watcher = ConfigWatcher(
                config={"bot_id": ctx.bot_id or self.strategy_id,
                         "data_dir": ctx.data_dir,
                         "data_source_id": "ibkr_execution"},
                config_modules=["strategy.config"],
                yaml_paths=yaml_paths,
            )
            self._config_watcher.take_baseline()
        except Exception:
            pass  # config watching is optional

    def _record_strategy_decision(
        self,
        code: str,
        details: dict,
        exchange_timestamp: Optional[datetime] = None,
    ) -> None:
        if self.ctx is None or not code:
            return
        pg_store = getattr(self.ctx, "pg_store", None)
        try:
            from libs.instrumentation.event_contract import write_strategy_decision_event

            data_dir = getattr(self.ctx, "data_dir", "")
            if data_dir:
                write_strategy_decision_event(
                    data_dir,
                    code=code,
                    strategy_id=self.strategy_id,
                    details=details or {},
                    exchange_timestamp=exchange_timestamp,
                    lineage=getattr(self.ctx, "lineage", None),
                )
        except Exception:
            logger.debug("Failed to write strategy decision event", exc_info=True)
        if pg_store is None:
            return
        async def _persist() -> None:
            try:
                await pg_store.record_strategy_decision(
                    self.strategy_id,
                    code,
                    details=details,
                    last_seen_bar_ts=exchange_timestamp,
                )
            except Exception:
                logger.debug("Failed to persist strategy decision", exc_info=True)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_persist())
        except RuntimeError:
            logger.debug("No running loop for strategy decision persistence")

    def log_entry(
        self,
        trade_id: str,
        pair: str,
        side: str,
        entry_price: float,
        position_size: float,
        position_size_quote: float,
        entry_signal: str,
        entry_signal_id: str,
        entry_signal_strength: float,
        active_filters: List[str],
        passed_filters: List[str],
        strategy_params: dict,
        signal_factors: Optional[List[dict]] = None,
        filter_decisions: Optional[List[dict]] = None,
        sizing_inputs: Optional[dict] = None,
        portfolio_state_at_entry: Optional[dict] = None,
        exchange_timestamp: Optional[datetime] = None,
        expected_entry_price: Optional[float] = None,
        entry_latency_ms: Optional[int] = None,
        bar_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        concurrent_positions_strategy: Optional[int] = None,
        correlated_pairs_detail: Optional[list] = None,
        execution_timeline: Optional[dict] = None,
        experiment_variant: Optional[str] = None,
        **runtime_refs,
    ) -> Any:
        """Log a trade entry with full instrumentation and enriched data.

        Automatically:
        - Calls regime_classifier.current_regime(pair) to tag market condition
        - Captures market snapshot
        - Stores enriched fields (signal_factors, filter_decisions, sizing_inputs, portfolio_state)
        - Writes to JSONL

        Never raises. Returns TradeEvent on success, empty dict on failure.

        Args:
            trade_id: Unique trade identifier
            pair: Trading pair (e.g. "BTC/USDT")
            side: "LONG" or "SHORT"
            entry_price: Actual fill price
            position_size: Size in base asset
            position_size_quote: Size in quote asset
            entry_signal: Signal name (e.g. "EMA cross")
            entry_signal_id: Signal instance ID
            entry_signal_strength: 0.0-1.0 signal confidence
            active_filters: List of filters that ran
            passed_filters: List of filters that passed
            strategy_params: Strategy config snapshot
            signal_factors: List of dicts describing what drove the signal
            filter_decisions: List of dicts with filter decision details
            sizing_inputs: Dict with inputs to position sizing (risk, ATR, etc)
            portfolio_state_at_entry: Dict with portfolio exposure and positions
            exchange_timestamp: Exchange order timestamp
            expected_entry_price: Expected vs actual slippage
            entry_latency_ms: Time from signal to fill
            bar_id: Bar/candle identifier for reproducibility

        Returns:
            TradeEvent dict on success, empty dict on failure (never raises)
        """

        def _log_entry_impl():
            if self.ctx is None or self.ctx.trade_logger is None:
                return {}

            # Auto-assign experiment variant if not provided
            nonlocal experiment_id, experiment_variant
            if not experiment_id and self.experiment_registry is not None:
                try:
                    for exp in self.experiment_registry.active_experiments():
                        if exp.strategy_type and exp.strategy_type not in ("", self.strategy_id, "coordinator"):
                            continue
                        experiment_id = exp.experiment_id
                        experiment_variant = self.experiment_registry.assign_variant(
                            exp.experiment_id, trade_id,
                        )
                        break
                except Exception:
                    pass

            # Get current market regime
            regime = "unknown"
            if self.ctx.regime_classifier is not None:
                regime = self.ctx.regime_classifier.current_regime(pair) or "unknown"

            # Auto-capture drawdown context
            drawdown_ctx = {}
            if self.ctx.drawdown_tracker is not None:
                drawdown_ctx = self.ctx.drawdown_tracker.get_entry_context()

            # Auto-capture session context
            from .session_classifier import SessionClassifier
            session_ctx = SessionClassifier.classify(datetime.now(ET))

            # Auto-capture overnight gap
            gap_ctx = {}
            if self.ctx.overnight_gap_tracker is not None:
                gap_ctx = self.ctx.overnight_gap_tracker.compute_gap(pair, entry_price)

            # Auto-capture overlay state
            overlay_ctx = {}
            if self.ctx.overlay_state_provider is not None:
                try:
                    signals = self.ctx.overlay_state_provider()
                    overlay_ctx = {"overlay_state": {
                        "qqq_ema_bullish": signals.get("QQQ", False),
                        "gld_ema_bullish": signals.get("GLD", False),
                    }}
                except Exception:
                    pass

            # Macro regime from RegimeService (global, distinct from technical market_regime)
            _macro = ""
            _stress = 0.0
            if self.ctx.get_regime_ctx is not None:
                try:
                    _rctx = self.ctx.get_regime_ctx()
                    if _rctx is not None:
                        _macro = _rctx.regime
                        _stress = _rctx.stress_level
                except Exception:
                    pass

            # Log the entry with all parameters including enriched ones
            trade_event = self.ctx.trade_logger.log_entry(
                trade_id=trade_id,
                pair=pair,
                side=side,
                entry_price=entry_price,
                position_size=position_size,
                position_size_quote=position_size_quote,
                entry_signal=entry_signal,
                entry_signal_id=entry_signal_id,
                entry_signal_strength=entry_signal_strength,
                active_filters=active_filters,
                passed_filters=passed_filters,
                strategy_params=strategy_params,
                strategy_id=self.strategy_id,
                exchange_timestamp=exchange_timestamp,
                expected_entry_price=expected_entry_price,
                entry_latency_ms=entry_latency_ms,
                market_regime=regime,
                macro_regime=_macro,
                stress_level_at_entry=_stress,
                bar_id=bar_id,
                # Enriched fields passed as kwargs (Task 3 will add them to signature)
                signal_factors=signal_factors or [],
                filter_decisions=filter_decisions or [],
                sizing_inputs=sizing_inputs,
                portfolio_state_at_entry=portfolio_state_at_entry,
                experiment_id=experiment_id,
                concurrent_positions_strategy=concurrent_positions_strategy,
                correlated_pairs_detail=correlated_pairs_detail,
                execution_timeline=execution_timeline,
                experiment_variant=experiment_variant,
                **runtime_refs,
                **drawdown_ctx,
                **session_ctx,
                **gap_ctx,
                **overlay_ctx,
            )

            # Auto-emit OrderBookContext at entry
            try:
                if self.ctx.orderbook_logger is not None:
                    entry_snap = trade_event.entry_snapshot if isinstance(trade_event.entry_snapshot, dict) else {}
                    bid = entry_snap.get("bid", 0)
                    ask = entry_snap.get("ask", 0)
                    if bid > 0 and ask > 0:
                        self.ctx.orderbook_logger.log_context(
                            pair=pair,
                            best_bid=bid,
                            best_ask=ask,
                            trade_context="entry",
                            related_trade_id=trade_id,
                            exchange_timestamp=exchange_timestamp,
                        )
            except Exception:
                pass

            self._record_strategy_decision(
                f"ENTRY:{entry_signal}" if entry_signal else "ENTRY",
                {
                    "pair": pair,
                    "side": side,
                    "trade_id": trade_id,
                    "signal_id": entry_signal_id,
                    "signal_strength": entry_signal_strength,
                    "bar_id": bar_id,
                    "filter_decisions": filter_decisions or [],
                },
                exchange_timestamp,
            )
            return trade_event.to_dict() if hasattr(trade_event, 'to_dict') else {}

        return safe_instrument(_log_entry_impl) or {}

    def log_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        fees_paid: float = 0.0,
        exchange_timestamp: Optional[datetime] = None,
        expected_exit_price: Optional[float] = None,
        exit_latency_ms: Optional[int] = None,
        mfe_price: Optional[float] = None,
        mae_price: Optional[float] = None,
        mfe_pct: Optional[float] = None,
        mae_pct: Optional[float] = None,
        mfe_r: Optional[float] = None,
        mae_r: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        **runtime_refs,
    ) -> Any:
        """Log a trade exit and auto-score the process quality.

        Automatically:
        - Logs the exit to TradeLogger
        - Calls ProcessScorer.score_and_write to tag root causes
        - Writes both trade and score to JSONL

        Never raises. Returns the scored TradeEvent on success, empty dict on failure.

        Args:
            trade_id: Unique trade identifier (must match entry)
            exit_price: Actual exit fill price
            exit_reason: Exit category (SIGNAL, STOP_LOSS, TAKE_PROFIT, TRAILING, TIMEOUT, MANUAL, etc)
            fees_paid: Fees charged for the exit
            exchange_timestamp: Exchange order timestamp
            expected_exit_price: Expected vs actual slippage
            exit_latency_ms: Time from exit signal to fill

        Returns:
            TradeEvent dict with process_score on success, empty dict on failure (never raises)
        """

        def _log_exit_impl():
            if self.ctx is None or self.ctx.trade_logger is None:
                return {}

            # Compute exit efficiency
            exit_efficiency = None
            if mfe_pct and mfe_pct > 0 and pnl_pct is not None:
                exit_efficiency = round(pnl_pct / mfe_pct, 4)

            # Log the exit
            trade_event = self.ctx.trade_logger.log_exit(
                trade_id=trade_id,
                exit_price=exit_price,
                exit_reason=exit_reason,
                fees_paid=fees_paid,
                exchange_timestamp=exchange_timestamp,
                expected_exit_price=expected_exit_price,
                exit_latency_ms=exit_latency_ms,
                mfe_price=mfe_price,
                mae_price=mae_price,
                mfe_pct=mfe_pct,
                mae_pct=mae_pct,
                mfe_r=mfe_r,
                mae_r=mae_r,
                exit_efficiency=exit_efficiency,
                **runtime_refs,
            )

            if trade_event is None:
                return {}

            # Auto-score the trade and merge score onto the exit record
            if self.ctx.process_scorer is not None:
                trade_dict = trade_event.to_dict() if hasattr(trade_event, 'to_dict') else trade_event
                score_result = self.ctx.process_scorer.score_and_write(
                    trade=trade_dict,
                    strategy_type=self.strategy_id,
                    data_dir=self.ctx.data_dir,
                )
                # Amend the JSONL exit record so sidecar forwards it with score
                if score_result is not None:
                    enrichments = {
                        "process_quality_score": score_result.process_quality_score,
                        "root_causes": score_result.root_causes,
                        "evidence_refs": score_result.evidence_refs,
                    }
                    try:
                        self.ctx.trade_logger.amend_last_event(trade_id, enrichments)
                    except Exception:
                        pass
                    # Also update the in-memory object for the return value
                    if hasattr(trade_event, "process_quality_score"):
                        trade_event.process_quality_score = score_result.process_quality_score
                        trade_event.root_causes = score_result.root_causes
                        trade_event.evidence_refs = score_result.evidence_refs

            # Auto-emit OrderBookContext at exit
            try:
                if self.ctx.orderbook_logger is not None:
                    exit_snap = trade_event.exit_snapshot if isinstance(trade_event.exit_snapshot, dict) else {}
                    bid = exit_snap.get("bid", 0)
                    ask = exit_snap.get("ask", 0)
                    pair = trade_event.pair
                    if bid > 0 and ask > 0:
                        self.ctx.orderbook_logger.log_context(
                            pair=pair,
                            best_bid=bid,
                            best_ask=ask,
                            trade_context="exit",
                            related_trade_id=trade_id,
                            exchange_timestamp=exchange_timestamp,
                        )
            except Exception:
                pass

            return trade_event.to_dict() if hasattr(trade_event, 'to_dict') else {}

        return safe_instrument(_log_exit_impl) or {}

    def log_missed(
        self,
        pair: str,
        side: str,
        signal: str,
        signal_id: str,
        signal_strength: float,
        blocked_by: str,
        block_reason: str = "",
        strategy_params: Optional[dict] = None,
        market_regime: str = "",
        filter_decisions: Optional[List[dict]] = None,
        exchange_timestamp: Optional[datetime] = None,
        bar_id: Optional[str] = None,
    ) -> Any:
        """Log a signal that fired but was blocked by a filter or risk limit.

        Automatically:
        - Captures market snapshot
        - Computes hypothetical entry price
        - Schedules outcome backfill
        - Writes to missed opportunity JSONL

        Never raises. Returns MissedOpportunityEvent on success, empty dict on failure.

        Args:
            pair: Trading pair
            side: "LONG" or "SHORT"
            signal: Signal name
            signal_id: Signal instance ID
            signal_strength: 0.0-1.0 signal confidence
            blocked_by: What blocked the signal (e.g. "max_open_trades")
            block_reason: More detailed reason
            strategy_params: Strategy config at time of signal
            market_regime: Market regime classification
            filter_decisions: List of filter gate snapshots
            exchange_timestamp: Signal timestamp
            bar_id: Bar/candle identifier

        Returns:
            MissedOpportunityEvent dict on success, empty dict on failure (never raises)
        """

        def _log_missed_impl():
            if self.ctx is None or self.ctx.missed_logger is None:
                return {}

            event = self.ctx.missed_logger.log_missed(
                pair=pair,
                side=side,
                signal=signal,
                signal_id=signal_id,
                signal_strength=signal_strength,
                blocked_by=blocked_by,
                block_reason=block_reason,
                strategy_params=strategy_params,
                strategy_type=self.strategy_id,
                strategy_id=self.strategy_id,
                market_regime=market_regime,
                filter_decisions=filter_decisions,
                exchange_timestamp=exchange_timestamp,
                bar_id=bar_id,
            )

            self._record_strategy_decision(
                f"BLOCKED:{blocked_by}" if blocked_by else "BLOCKED",
                {
                    "pair": pair,
                    "side": side,
                    "signal": signal,
                    "signal_id": signal_id,
                    "signal_strength": signal_strength,
                    "block_reason": block_reason,
                    "bar_id": bar_id,
                    "filter_decisions": filter_decisions or [],
                },
                exchange_timestamp,
            )
            return event.to_dict() if hasattr(event, 'to_dict') else {}

        return safe_instrument(_log_missed_impl) or {}

    def classify_regime(self, symbol: str) -> str:
        """Get the current market regime for a symbol.

        Returns one of:
        - "trending_up"
        - "trending_down"
        - "ranging"
        - "volatile"
        - "unknown" (on error or no classifier)

        Never raises.

        Args:
            symbol: Trading symbol

        Returns:
            Regime string, always valid (never raises)
        """

        def _classify_impl():
            if self.ctx is None or self.ctx.regime_classifier is None:
                return "unknown"

            regime = self.ctx.regime_classifier.classify(symbol)
            return regime if regime in {"trending_up", "trending_down", "ranging", "volatile", "unknown"} else "unknown"

        return safe_instrument(_classify_impl) or "unknown"

    def record_close(self, symbol: str, close_price: float) -> None:
        """Record previous day's closing price for overnight gap tracking."""
        def _impl():
            if self.ctx and self.ctx.overnight_gap_tracker:
                self.ctx.overnight_gap_tracker.record_close(symbol, close_price)
        safe_instrument(_impl)

    def on_indicator_snapshot(
        self,
        pair: str,
        indicators: dict[str, float],
        signal_name: str,
        signal_strength: float,
        decision: str,
        strategy_id: str = "",
        overlay_state: dict | None = None,
        drawdown_tier: str = "",
        market_session: str = "",
        exchange_timestamp=None,
        bar_id: str | None = None,
    ) -> None:
        """Fire-and-forget indicator snapshot at signal evaluation."""
        try:
            if self.ctx is None or self.ctx.indicator_logger is None:
                return

            # Auto-capture context
            context: Dict[str, Any] = {}
            if overlay_state:
                context["overlay_state"] = overlay_state
            elif self.ctx.overlay_state_provider is not None:
                try:
                    signals = self.ctx.overlay_state_provider()
                    context["overlay_state"] = {
                        "qqq_ema_bullish": signals.get("QQQ", False),
                        "gld_ema_bullish": signals.get("GLD", False),
                    }
                except Exception:
                    pass

            if drawdown_tier:
                context["drawdown_tier"] = drawdown_tier
            elif self.ctx.drawdown_tracker is not None:
                dd_ctx = self.ctx.drawdown_tracker.get_entry_context()
                context["drawdown_tier"] = dd_ctx.get("drawdown_tier_at_entry", "NORMAL")

            if market_session:
                context["market_session"] = market_session
            else:
                from .session_classifier import SessionClassifier
                session_ctx = SessionClassifier.classify(datetime.now(ET))
                context["market_session"] = session_ctx.get("market_session", "")

            self.ctx.indicator_logger.log_snapshot(
                pair=pair,
                indicators=indicators,
                signal_name=signal_name,
                signal_strength=signal_strength,
                decision=decision,
                strategy_type=strategy_id or self.strategy_id,
                exchange_timestamp=exchange_timestamp,
                bar_id=bar_id,
                context=context,
            )
            if decision:
                self._record_strategy_decision(
                    f"{signal_name}:{decision}" if signal_name else decision,
                    {
                        "pair": pair,
                        "signal_name": signal_name,
                        "signal_strength": signal_strength,
                        "strategy_id": strategy_id or self.strategy_id,
                        "bar_id": bar_id,
                        "context": context,
                    },
                    exchange_timestamp,
                )
        except Exception:
            pass  # instrumentation must never affect trading

    def on_filter_decision(
        self,
        pair: str,
        filter_name: str,
        passed: bool,
        threshold: float,
        actual_value: float,
        signal_name: str = "",
        signal_strength: float = 0.0,
        strategy_id: str = "",
        coordinator_triggered: bool = False,
        exchange_timestamp=None,
        bar_id: str | None = None,
    ) -> None:
        """Fire-and-forget filter decision event."""
        try:
            if self.ctx is None or self.ctx.filter_logger is None:
                return
            self.ctx.filter_logger.log_decision(
                pair=pair,
                filter_name=filter_name,
                passed=passed,
                threshold=threshold,
                actual_value=actual_value,
                signal_name=signal_name,
                signal_strength=signal_strength,
                strategy_type=strategy_id or self.strategy_id,
                coordinator_triggered=coordinator_triggered,
                exchange_timestamp=exchange_timestamp,
                bar_id=bar_id,
            )
            self._record_strategy_decision(
                f"FILTER_BLOCKED:{filter_name}" if not passed else "FILTERS_PASSED",
                {
                    "pair": pair,
                    "filter_name": filter_name,
                    "passed": passed,
                    "threshold": threshold,
                    "actual_value": actual_value,
                    "signal_name": signal_name,
                    "signal_strength": signal_strength,
                    "strategy_id": strategy_id or self.strategy_id,
                    "bar_id": bar_id,
                },
                exchange_timestamp,
            )
        except Exception:
            pass

    def on_orderbook_context(
        self,
        pair: str,
        best_bid: float,
        best_ask: float,
        trade_context: str | None = None,
        related_trade_id: str | None = None,
        bid_depth_10bps: float = 0.0,
        ask_depth_10bps: float = 0.0,
        bid_levels: list[dict] | None = None,
        ask_levels: list[dict] | None = None,
        exchange_timestamp=None,
    ) -> None:
        """Fire-and-forget order book context capture."""
        try:
            if self.ctx is None or self.ctx.orderbook_logger is None:
                return
            self.ctx.orderbook_logger.log_context(
                pair=pair,
                best_bid=best_bid,
                best_ask=best_ask,
                trade_context=trade_context,
                related_trade_id=related_trade_id,
                bid_depth_10bps=bid_depth_10bps,
                ask_depth_10bps=ask_depth_10bps,
                bid_levels=bid_levels,
                ask_levels=ask_levels,
                exchange_timestamp=exchange_timestamp,
            )
        except Exception:
            pass

    def check_config_changes(self) -> None:
        """Call periodically from main loop. Fire-and-forget."""
        try:
            if self._config_watcher is not None:
                self._config_watcher.check()
        except Exception:
            pass

    @property
    def experiment_registry(self):
        """Access to experiment registry for variant assignment."""
        if self.ctx is None:
            return None
        return getattr(self.ctx, "experiment_registry", None)

    def capture_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Capture a market snapshot for a symbol.

        Never raises. Returns dict on success, None on failure.

        Args:
            symbol: Trading symbol

        Returns:
            Market snapshot dict with bid/ask/mid/atr/volume/etc, or None on error
        """

        def _capture_impl():
            if self.ctx is None or self.ctx.snapshot_service is None:
                return None

            snapshot = self.ctx.snapshot_service.capture_now(symbol)
            return snapshot.to_dict() if hasattr(snapshot, 'to_dict') else None

        return safe_instrument(_capture_impl)

    def on_order_event(
        self,
        order_id: str,
        pair: str,
        side: str,
        order_type: str,
        status: str,
        requested_qty: float,
        filled_qty: float = 0.0,
        requested_price: float | None = None,
        fill_price: float | None = None,
        reject_reason: str = "",
        latency_ms: float | None = None,
        related_trade_id: str = "",
        strategy_id: str = "",
        order_action: str = "NEW",
        coordinator_triggered: bool = False,
        coordinator_rule: str = "",
        modification_details: dict | None = None,
        exchange_timestamp=None,
        bar_id: str | None = None,
    ) -> None:
        """Record an order lifecycle event. Auto-captures context. Fire-and-forget."""
        try:
            if self.ctx is None or self.ctx.order_logger is None:
                return

            # Auto-capture context from existing trackers
            dd_tier = ""
            if self.ctx.drawdown_tracker is not None:
                dd_ctx = self.ctx.drawdown_tracker.get_entry_context()
                dd_tier = dd_ctx.get("drawdown_tier_at_entry", "")

            market_session = ""
            from .session_classifier import SessionClassifier
            session_ctx = SessionClassifier.classify(datetime.now(ET))
            market_session = session_ctx.get("market_session", "")

            overlay = None
            if self.ctx.overlay_state_provider is not None:
                try:
                    overlay = self.ctx.overlay_state_provider()
                except Exception:
                    pass

            self.ctx.order_logger.log_order(
                order_id=order_id,
                pair=pair,
                side=side,
                order_type=order_type,
                status=status,
                requested_qty=requested_qty,
                filled_qty=filled_qty,
                requested_price=requested_price,
                fill_price=fill_price,
                reject_reason=reject_reason,
                latency_ms=latency_ms,
                related_trade_id=related_trade_id,
                strategy_id=strategy_id,
                order_action=order_action,
                coordinator_triggered=coordinator_triggered,
                coordinator_rule=coordinator_rule,
                modification_details=modification_details,
                overlay_state=overlay,
                drawdown_tier=dd_tier,
                market_session=market_session,
                exchange_timestamp=exchange_timestamp,
                bar_id=bar_id,
            )
        except Exception:
            pass  # instrumentation must never affect trading

    def emit_heartbeat(
        self,
        active_positions: int,
        open_orders: int,
        uptime_s: float,
        error_count_1h: int,
        positions: list[dict] | None = None,
        portfolio_exposure: dict | None = None,
    ) -> None:
        """Emit a heartbeat event with optional position and exposure data.

        If positions/portfolio_exposure are not provided, attempts to auto-build
        from internal state. Falls back to basic heartbeat on any error.
        Never raises.
        """
        try:
            if self.ctx is None:
                return

            if positions is None and hasattr(self.ctx, "portfolio_tracker") and self.ctx.portfolio_tracker:
                positions, portfolio_exposure = self._build_position_snapshot()

            heartbeat_data = {
                "bot_id": self.ctx.bot_id or self.strategy_id,
                "active_positions": active_positions,
                "open_orders": open_orders,
                "uptime_s": uptime_s,
                "error_count_1h": error_count_1h,
            }

            if positions is not None:
                heartbeat_data["positions"] = positions
            if portfolio_exposure is not None:
                heartbeat_data["portfolio_exposure"] = portfolio_exposure
            try:
                from libs.instrumentation.event_contract import enrich_payload
                heartbeat_data = enrich_payload(
                    heartbeat_data,
                    lineage=getattr(self.ctx, "lineage", None),
                    event_type="heartbeat",
                    scope="strategy",
                )
            except Exception:
                pass

            # Write to heartbeat JSONL
            hb_dir = Path(self.ctx.data_dir) / "heartbeat"
            hb_dir.mkdir(parents=True, exist_ok=True)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filepath = hb_dir / f"heartbeat_{today}.jsonl"
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(heartbeat_data, default=str) + "\n")

        except Exception:
            # Fallback: try to emit basic heartbeat without position data
            try:
                hb_dir = Path(self.ctx.data_dir) / "heartbeat"
                hb_dir.mkdir(parents=True, exist_ok=True)
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                filepath = hb_dir / f"heartbeat_{today}.jsonl"
                basic = {
                    "active_positions": active_positions,
                    "open_orders": open_orders,
                    "uptime_s": uptime_s,
                    "error_count_1h": error_count_1h,
                }
                try:
                    from libs.instrumentation.event_contract import enrich_payload
                    basic = enrich_payload(
                        basic,
                        lineage=getattr(self.ctx, "lineage", None),
                        event_type="heartbeat",
                        scope="strategy",
                    )
                except Exception:
                    pass
                with open(filepath, "a", encoding="utf-8") as f:
                    f.write(json.dumps(basic, default=str) + "\n")
            except Exception:
                pass

    def log_stop_adjustment(
        self,
        trade_id: str,
        symbol: str,
        old_stop: float,
        new_stop: float,
        adjustment_type: str,
        trigger: str,
        metadata: dict | None = None,
    ) -> None:
        """Log a stop-loss adjustment event to JSONL for TA analysis.

        Fire-and-forget — never raises.

        Args:
            trade_id: Trade being adjusted
            symbol: Instrument symbol
            old_stop: Previous stop price
            new_stop: New stop price
            adjustment_type: trailing | breakeven | coordination_tighten | partial_trail | time_decay
            trigger: What caused it (atr_trail, mfe_threshold, coord_rule, etc.)
            metadata: Optional extra context
        """
        try:
            if old_stop == new_stop:
                return
            data_dir = self.ctx.data_dir if self.ctx else None
            if not data_dir:
                return
            now = datetime.now(timezone.utc)
            record = {
                "timestamp": now.isoformat(),
                "strategy_id": self.strategy_id,
                "trade_id": trade_id,
                "symbol": symbol,
                "old_stop": old_stop,
                "new_stop": new_stop,
                "adjustment_type": adjustment_type,
                "trigger": trigger,
                "tightening_distance": round(abs(new_stop - old_stop), 6),
                "metadata": metadata or {},
            }
            try:
                from libs.instrumentation.event_contract import enrich_payload
                record = enrich_payload(
                    record,
                    lineage=getattr(self.ctx, "lineage", None),
                    event_type="stop_adjustment",
                    scope="strategy",
                )
            except Exception:
                pass
            date_str = now.strftime("%Y-%m-%d")
            out_dir = Path(data_dir) / "stop_adjustments"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"{date_str}.jsonl", "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass

    def _build_position_snapshot(self) -> tuple[list[dict], dict]:
        """Build position list and portfolio exposure from current state."""
        raw_positions = self.ctx.portfolio_tracker.get_open_positions()

        positions = []
        for pos in raw_positions:
            current_price = pos.entry_price
            if self.ctx.snapshot_service:
                current = self.ctx.snapshot_service.get_latest(pos.symbol)
                if current:
                    current_price = current.last_trade_price

            is_long = pos.side == "LONG"
            unrealized = (current_price - pos.entry_price) * pos.qty if is_long else \
                         (pos.entry_price - current_price) * pos.qty

            duration_minutes = 0
            if hasattr(pos, "entry_time") and pos.entry_time:
                duration_minutes = int(
                    (datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 60
                )

            dd_pct = 0.0
            size_mult = 1.0
            if self.ctx.drawdown_tracker:
                dd_pct = getattr(self.ctx.drawdown_tracker, "drawdown_pct", 0.0)
                size_mult = getattr(self.ctx.drawdown_tracker, "position_size_multiplier", 1.0)

            positions.append({
                "pair": pos.symbol,
                "side": pos.side,
                "qty": pos.qty,
                "entry_price": pos.entry_price,
                "current_price": current_price,
                "unrealized_pnl": round(unrealized, 4),
                "unrealized_pnl_pct": round(
                    unrealized / (pos.entry_price * pos.qty) * 100, 4
                ) if pos.entry_price * pos.qty > 0 else 0.0,
                "duration_minutes": duration_minutes,
                "strategy_id": getattr(pos, "strategy_id", ""),
                "is_overlay": getattr(pos, "strategy_id", "") == "OVERLAY",
                "drawdown_pct_current": dd_pct,
                "position_size_multiplier": size_mult,
            })

        # Build portfolio exposure
        dd_ctx = {}
        if self.ctx.drawdown_tracker:
            dd_ctx = self.ctx.drawdown_tracker.get_entry_context()

        from .session_classifier import SessionClassifier
        session_ctx = SessionClassifier.classify(datetime.now(ET))

        overlay = {}
        if self.ctx.overlay_state_provider:
            try:
                overlay = self.ctx.overlay_state_provider()
            except Exception:
                pass

        main_positions = [p for p in positions if not p["is_overlay"]]
        overlay_positions = [p for p in positions if p["is_overlay"]]

        coordinator_rules: list = []
        if hasattr(self.ctx, "coordinator") and self.ctx.coordinator:
            try:
                coordinator_rules = self.ctx.coordinator.get_active_rules()
            except Exception:
                pass

        # Group by strategy
        by_strategy: dict[str, dict] = {}
        for p in positions:
            sid = p["strategy_id"]
            if sid not in by_strategy:
                by_strategy[sid] = {
                    "positions": 0,
                    "unrealized_pnl": 0.0,
                    "symbols": [],
                }
            by_strategy[sid]["positions"] += 1
            by_strategy[sid]["unrealized_pnl"] += p["unrealized_pnl"]
            by_strategy[sid]["symbols"].append(p["pair"])

        account_equity = getattr(self.ctx.drawdown_tracker, "current_equity", None) or 1.0
        total_exposure = sum(p["qty"] * p["entry_price"] for p in positions)

        exposure = {
            "total_positions": len(positions),
            "main_strategy_positions": len(main_positions),
            "overlay_positions": len(overlay_positions),
            "total_exposure_pct": round(total_exposure / account_equity * 100, 2),
            "main_exposure_pct": round(
                sum(p["qty"] * p["entry_price"] for p in main_positions)
                / account_equity * 100, 2
            ),
            "overlay_exposure_pct": round(
                sum(p["qty"] * p["entry_price"] for p in overlay_positions)
                / account_equity * 100, 2
            ),
            "total_unrealized_pnl": round(
                sum(p["unrealized_pnl"] for p in positions), 4
            ),
            "daily_realized_pnl": getattr(
                getattr(self.ctx, "daily_pnl_tracker", None), "realized_pnl", 0.0
            ),
            "drawdown_tier": dd_ctx.get("drawdown_tier_at_entry", "NORMAL"),
            "market_session": session_ctx.get("market_session", ""),
            "overlay_state": overlay,
            "coordinator_active_rules": coordinator_rules,
            "by_strategy": by_strategy,
        }

        return positions, exposure
