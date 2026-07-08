"""InstrumentationKit facade — single clean API for strategy engines.

Usage in strategy engine:
    kit = InstrumentationKit(instr_manager, strategy_type="helix")
    kit.log_entry(trade_id=..., signal_factors=[...], filter_decisions=[...], ...)
    kit.log_exit(trade_id=..., exit_price=..., exit_reason=...)
    kit.log_missed(pair=..., blocked_by=..., filter_decisions=[...], ...)
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Any

if TYPE_CHECKING:
    from .bootstrap import InstrumentationManager

logger = logging.getLogger("instrumentation.facade")


class InstrumentationKit:
    """Thin facade over InstrumentationManager for strategy-engine callers.

    All methods are fire-and-forget: exceptions are caught and logged,
    never propagated to strategy code.
    """

    def __init__(self, manager: Optional["InstrumentationManager"], strategy_type: str = ""):
        self._mgr = manager
        self._strategy_type = strategy_type
        # Get experiment tracking from manager's config
        self._experiment_id = None
        self._experiment_variant = None
        # Lazy-init loggers for Phase 2B event types
        self._indicator_logger = None
        self._filter_event_logger = None
        self._orderbook_logger = None
        self._data_dir: Path | None = None
        if manager:
            try:
                config = getattr(manager, '_config', None)
                if not isinstance(config, dict):
                    config = {}
                self._experiment_id = config.get("experiment_id")
                self._experiment_variant = config.get("experiment_variant")
                data_dir = config.get("data_dir")
                bot_id = config.get("bot_id", "")
                if data_dir and isinstance(data_dir, (str, Path)):
                    self._data_dir = Path(data_dir)
                    from .indicator_logger import IndicatorLogger
                    from .filter_event_logger import FilterEventLogger
                    from .orderbook_logger import OrderBookLogger
                    lineage = getattr(manager, "lineage", None)
                    self._indicator_logger = IndicatorLogger(data_dir=data_dir, bot_id=bot_id, lineage=lineage)
                    self._filter_event_logger = FilterEventLogger(data_dir=data_dir, bot_id=bot_id, lineage=lineage)
                    self._orderbook_logger = OrderBookLogger(data_dir=data_dir, bot_id=bot_id, lineage=lineage)
                register = getattr(manager, "_register_instrumentation_kit", None)
                if callable(register):
                    register(self)
            except Exception:
                pass

    @property
    def active(self) -> bool:
        return self._mgr is not None

    def refresh_lineage(self, lineage=None) -> None:
        """Refresh lazy facade-owned loggers after runtime rule changes."""
        lineage = lineage or getattr(self._mgr, "lineage", None)
        if lineage is None:
            return
        for component in (
            self._indicator_logger,
            self._filter_event_logger,
            self._orderbook_logger,
        ):
            if hasattr(component, "_lineage"):
                component._lineage = lineage

    def _sync_lineage(self) -> None:
        self.refresh_lineage(getattr(self._mgr, "lineage", None))

    def _record_strategy_decision(
        self,
        code: str,
        details: dict,
        exchange_timestamp: Optional[datetime] = None,
    ) -> None:
        if not self._mgr or not code:
            return
        pg_store = getattr(self._mgr, "_pg_store", None)
        strategy_id = getattr(self._mgr, "_strategy_id", "")
        if not strategy_id:
            return
        try:
            from libs.instrumentation.event_contract import write_strategy_decision_event

            config = getattr(self._mgr, "_config", {}) or {}
            data_dir = config.get("data_dir") or self._data_dir
            if data_dir:
                write_strategy_decision_event(
                    data_dir,
                    code=code,
                    strategy_id=strategy_id,
                    details=details or {},
                    exchange_timestamp=exchange_timestamp,
                    lineage=getattr(self._mgr, "lineage", None),
                )
        except Exception:
            logger.debug("Failed to write strategy decision event", exc_info=True)
        if pg_store is None:
            return
        async def _persist() -> None:
            try:
                await pg_store.record_strategy_decision(
                    strategy_id,
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

    def health_check(self) -> dict:
        """Return instrumentation health status so strategies can detect silent failures.

        Returns dict with 'healthy' bool and component-level detail.
        """
        result: dict = {"healthy": False, "components": {}}
        if not self._mgr:
            return result
        try:
            sidecar_diag = self._mgr.get_sidecar_diagnostics()
            result["components"]["sidecar"] = sidecar_diag or {}
            result["components"]["error_count_1h"] = self._mgr.recent_error_count_1h()
            sidecar_ok = sidecar_diag.get("relay_reachable", False) if sidecar_diag else False
            result["components"]["sidecar_reachable"] = sidecar_ok
            result["healthy"] = True  # facade itself is functional
        except Exception:
            pass
        return result

    def log_entry(
        self,
        *,
        trade_id: str,
        pair: str,
        side: str,
        entry_price: float,
        position_size: float,
        position_size_quote: float,
        entry_signal: str,
        entry_signal_id: str,
        entry_signal_strength: float,
        expected_entry_price: Optional[float] = None,
        strategy_params: Optional[dict] = None,
        # Enriched fields
        signal_factors: Optional[List[dict]] = None,
        filter_decisions: Optional[List[dict]] = None,
        conviction_factors: Optional[dict] = None,
        sizing_inputs: Optional[dict] = None,
        portfolio_state: Optional[dict] = None,
        session_type: str = "",
        contract_month: str = "",
        margin_used_pct: Optional[float] = None,
        concurrent_positions: Optional[int] = None,
        drawdown_pct: Optional[float] = None,
        drawdown_tier: str = "",
        drawdown_size_mult: Optional[float] = None,
        bar_id: Optional[str] = None,
        exchange_timestamp: Optional[datetime] = None,
        entry_latency_ms: Optional[int] = None,
        signal_evolution: Optional[list[dict]] = None,
        execution_timestamps: Optional[dict] = None,
        **runtime_refs,
    ) -> None:
        if not self._mgr:
            return
        try:
            regime = self._mgr.regime_classifier.current_regime(pair)

            _macro = ""
            _stress = 0.0
            _getter = getattr(self._mgr, '_get_regime_ctx', None)
            if _getter is not None:
                try:
                    _rctx = _getter()
                    if _rctx is not None:
                        _macro = _rctx.regime
                        _stress = _rctx.stress_level
                except Exception:
                    pass

            exp_id = self._experiment_id or ""
            exp_var = self._experiment_variant or ""
            if self._mgr and hasattr(self._mgr, 'experiment_registry') and self._mgr.experiment_registry:
                try:
                    registry = self._mgr.experiment_registry
                    for exp in registry.active_experiments():
                        if exp.strategy_type and exp.strategy_type != self._strategy_type:
                            continue
                        exp_id = exp.experiment_id
                        exp_var = registry.assign_variant(exp.experiment_id, trade_id)
                        break
                except Exception:
                    pass

            self._mgr.trade_logger.log_entry(
                trade_id=trade_id,
                pair=pair,
                side=side,
                entry_price=entry_price,
                position_size=position_size,
                position_size_quote=position_size_quote,
                entry_signal=entry_signal,
                entry_signal_id=entry_signal_id,
                entry_signal_strength=entry_signal_strength,
                active_filters=[d["filter_name"] for d in (filter_decisions or [])],
                passed_filters=[d["filter_name"] for d in (filter_decisions or []) if d.get("passed")],
                strategy_params=strategy_params or {},
                expected_entry_price=expected_entry_price,
                market_regime=regime,
                macro_regime=_macro,
                stress_level_at_entry=_stress,
                bar_id=bar_id,
                exchange_timestamp=exchange_timestamp,
                entry_latency_ms=entry_latency_ms,
                portfolio_state=portfolio_state,
                signal_factors=signal_factors,
                filter_decisions=filter_decisions,
                conviction_factors=conviction_factors,
                sizing_inputs=sizing_inputs,
                session_type=session_type,
                contract_month=contract_month,
                margin_used_pct=margin_used_pct,
                concurrent_positions=concurrent_positions,
                drawdown_pct=drawdown_pct,
                drawdown_tier=drawdown_tier,
                drawdown_size_mult=drawdown_size_mult,
                signal_evolution=signal_evolution,
                execution_timestamps=execution_timestamps,
                experiment_id=exp_id,
                experiment_variant=exp_var,
                **runtime_refs,
            )

            # Phase 2B: emit standalone filter decision events
            if filter_decisions:
                self.on_filter_decisions(
                    filter_decisions=filter_decisions,
                    pair=pair,
                    signal_name=entry_signal,
                    signal_strength=entry_signal_strength,
                    strategy_type=self._strategy_type,
                    exchange_timestamp=exchange_timestamp,
                    bar_id=bar_id,
                )

            self._record_strategy_decision(
                f"ENTRY:{entry_signal}" if entry_signal else "ENTRY",
                {
                    "pair": pair,
                    "side": side,
                    "trade_id": trade_id,
                    "signal_id": entry_signal_id,
                    "signal_strength": entry_signal_strength,
                    "strategy_type": self._strategy_type,
                    "bar_id": bar_id,
                    "filter_decisions": filter_decisions or [],
                },
                exchange_timestamp,
            )

        except Exception as e:
            logger.warning("InstrumentationKit.log_entry failed: %s", e)

    def log_exit(
        self,
        *,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        fees_paid: float = 0.0,
        exchange_timestamp: Optional[datetime] = None,
        expected_exit_price: Optional[float] = None,
        exit_latency_ms: Optional[int] = None,
        mfe_r: Optional[float] = None,
        mae_r: Optional[float] = None,
        mfe_price: Optional[float] = None,
        mae_price: Optional[float] = None,
        session_transitions: Optional[list] = None,
        **runtime_refs,
    ) -> None:
        if not self._mgr:
            return
        try:
            self._mgr.trade_logger.log_exit(
                trade_id=trade_id,
                exit_price=exit_price,
                exit_reason=exit_reason,
                fees_paid=fees_paid,
                exchange_timestamp=exchange_timestamp,
                expected_exit_price=expected_exit_price,
                exit_latency_ms=exit_latency_ms,
                mfe_r=mfe_r,
                mae_r=mae_r,
                mfe_price=mfe_price,
                mae_price=mae_price,
                session_transitions=session_transitions,
                **runtime_refs,
            )
        except Exception as e:
            logger.warning("InstrumentationKit.log_exit failed: %s", e)

    def log_missed(
        self,
        *,
        pair: str,
        side: str,
        signal: str,
        signal_id: str,
        signal_strength: float,
        blocked_by: str,
        block_reason: str = "",
        strategy_params: Optional[dict] = None,
        filter_decisions: Optional[List[dict]] = None,
        coordination_context: Optional[dict] = None,
        session_type: str = "",
        concurrent_positions: Optional[int] = None,
        drawdown_pct: Optional[float] = None,
        drawdown_tier: str = "",
        exchange_timestamp: Optional[datetime] = None,
        bar_id: Optional[str] = None,
        signal_evolution: Optional[list[dict]] = None,
    ) -> None:
        if not self._mgr:
            return
        try:
            regime = self._mgr.regime_classifier.current_regime(pair)

            # Enrich strategy_params with context that doesn't have dedicated fields yet
            enriched_params = dict(strategy_params or {})
            if concurrent_positions is not None:
                enriched_params["_concurrent_positions"] = concurrent_positions
            if session_type:
                enriched_params["_session_type"] = session_type
            if drawdown_pct is not None:
                enriched_params["_drawdown_pct"] = drawdown_pct
                enriched_params["_drawdown_tier"] = drawdown_tier
            if filter_decisions:
                enriched_params["_filter_decisions"] = filter_decisions
                # Phase 2B: emit standalone filter decision events
                self.on_filter_decisions(
                    filter_decisions=filter_decisions,
                    pair=pair,
                    signal_name=signal,
                    signal_strength=signal_strength,
                    strategy_type=self._strategy_type,
                    exchange_timestamp=exchange_timestamp,
                    bar_id=bar_id,
                )
            if coordination_context:
                enriched_params["_coordination_context"] = coordination_context

            self._mgr.missed_logger.log_missed(
                pair=pair,
                side=side,
                signal=signal,
                signal_id=signal_id,
                signal_strength=signal_strength,
                blocked_by=blocked_by,
                block_reason=block_reason,
                strategy_params=enriched_params,
                strategy_type=self._strategy_type,
                market_regime=regime,
                exchange_timestamp=exchange_timestamp,
                bar_id=bar_id,
                signal_evolution=signal_evolution,
                filter_decisions=filter_decisions,
                coordination_context=coordination_context,
                concurrent_positions=concurrent_positions,
                session_type=session_type,
                drawdown_pct=drawdown_pct,
                drawdown_tier=drawdown_tier,
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
                    "strategy_type": self._strategy_type,
                    "bar_id": bar_id,
                    "filter_decisions": filter_decisions or [],
                },
                exchange_timestamp,
            )
        except Exception as e:
            logger.warning("InstrumentationKit.log_missed failed: %s", e)

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
        strategy_type: str = "",
        session: str = "",
        contract_month: str = "",
        order_book_depth: dict | None = None,
        exchange_timestamp=None,
        bar_id: str | None = None,
    ) -> None:
        """Record an order lifecycle event. Fire-and-forget."""
        if not self._mgr:
            return
        try:
            self._mgr.order_logger.log_order(
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
                strategy_type=strategy_type or self._strategy_type,
                session=session,
                contract_month=contract_month,
                order_book_depth=order_book_depth,
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
        """Emit a heartbeat event with optional position state."""
        if not self._mgr:
            return
        try:
            heartbeat_data = {
                "bot_id": self._mgr.bot_id if hasattr(self._mgr, 'bot_id') else
                    getattr(self._mgr, '_strategy_id', ''),
                "strategy_type": self._strategy_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "active_positions": active_positions,
                "open_orders": open_orders,
                "uptime_s": uptime_s,
                "error_count_1h": error_count_1h,
                "positions": positions or [],
                "portfolio_exposure": portfolio_exposure or {},
            }

            # Include sidecar diagnostics if available
            diag = self._mgr.get_sidecar_diagnostics()
            if diag:
                heartbeat_data["sidecar"] = diag
            try:
                from libs.instrumentation.event_contract import enrich_payload
                heartbeat_data = enrich_payload(
                    heartbeat_data,
                    lineage=getattr(self._mgr, "lineage", None),
                    event_type="heartbeat",
                    scope="strategy",
                )
            except Exception:
                pass

            if not self._data_dir:
                return
            hb_dir = self._data_dir / "heartbeats"
            hb_dir.mkdir(parents=True, exist_ok=True)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filepath = hb_dir / f"heartbeat_{today}.jsonl"
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(heartbeat_data, default=str) + "\n")
        except Exception:
            pass  # instrumentation must never affect trading

    def log_error(
        self,
        *,
        error_type: str,
        message: str,
        severity: str = "medium",
        category: str = "unknown",
        context: dict[str, Any] | None = None,
        exc: BaseException | None = None,
        exchange_timestamp=None,
    ) -> None:
        """Record a structured runtime/instrumentation error event."""
        if not self._mgr:
            return
        try:
            self._mgr.record_error(
                error_type=error_type,
                message=message,
                severity=severity,
                category=category,
                context=context,
                exc=exc,
                exchange_timestamp=exchange_timestamp,
            )
        except Exception:
            pass

    def on_indicator_snapshot(
        self,
        pair: str,
        indicators: dict[str, float],
        signal_name: str,
        signal_strength: float,
        decision: str,
        strategy_type: str,
        exchange_timestamp=None,
        bar_id: str | None = None,
        context: dict | None = None,
    ) -> None:
        """Fire-and-forget indicator snapshot."""
        try:
            self._sync_lineage()
            if self._indicator_logger:
                self._indicator_logger.log_snapshot(
                    pair=pair, indicators=indicators,
                    signal_name=signal_name, signal_strength=signal_strength,
                    decision=decision, strategy_type=strategy_type,
                    exchange_timestamp=exchange_timestamp, bar_id=bar_id,
                    context=context,
                )
            if decision:
                self._record_strategy_decision(
                    f"{signal_name}:{decision}" if signal_name else decision,
                    {
                        "pair": pair,
                        "signal_name": signal_name,
                        "signal_strength": signal_strength,
                        "strategy_type": strategy_type,
                        "bar_id": bar_id,
                        "context": context or {},
                    },
                    exchange_timestamp,
                )
        except Exception:
            pass

    def on_filter_decisions(
        self,
        filter_decisions: list,
        pair: str,
        signal_name: str = "",
        signal_strength: float = 0.0,
        strategy_type: str = "",
        exchange_timestamp=None,
        bar_id: str | None = None,
    ) -> None:
        """Emit all filter decisions from a signal evaluation as standalone events."""
        try:
            self._sync_lineage()
            if self._filter_event_logger:
                from .filter_decision import FilterDecision
                fds = []
                for fd in filter_decisions:
                    if isinstance(fd, FilterDecision):
                        fds.append(fd)
                    elif isinstance(fd, dict):
                        fds.append(FilterDecision(
                            filter_name=fd.get("filter_name", ""),
                            threshold=fd.get("threshold", 0.0),
                            actual_value=fd.get("actual_value", 0.0),
                            passed=fd.get("passed", True),
                        ))
                self._filter_event_logger.log_decisions(
                    fds, pair=pair, signal_name=signal_name,
                    signal_strength=signal_strength, strategy_type=strategy_type,
                    exchange_timestamp=exchange_timestamp, bar_id=bar_id,
                )
                if fds:
                    failed = next((fd for fd in fds if not getattr(fd, "passed", True)), None)
                    code = (
                        f"FILTER_BLOCKED:{failed.filter_name}"
                        if failed is not None
                        else "FILTERS_PASSED"
                    )
                    self._record_strategy_decision(
                        code,
                        {
                            "pair": pair,
                            "signal_name": signal_name,
                            "signal_strength": signal_strength,
                            "strategy_type": strategy_type,
                            "bar_id": bar_id,
                            "filter_decisions": [fd.to_dict() for fd in fds],
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
        """Fire-and-forget order book context."""
        try:
            self._sync_lineage()
            if self._orderbook_logger:
                self._orderbook_logger.log_context(
                    pair=pair, best_bid=best_bid, best_ask=best_ask,
                    trade_context=trade_context, related_trade_id=related_trade_id,
                    bid_depth_10bps=bid_depth_10bps, ask_depth_10bps=ask_depth_10bps,
                    bid_levels=bid_levels, ask_levels=ask_levels,
                    exchange_timestamp=exchange_timestamp,
                )
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
        """Log a stop-loss adjustment event to JSONL for TA analysis. Fire-and-forget."""
        try:
            if old_stop == new_stop:
                return
            data_dir = self._data_dir
            if not data_dir:
                return
            now = datetime.now(timezone.utc)
            record = {
                "timestamp": now.isoformat(),
                "strategy_id": self._strategy_type,
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
                    lineage=getattr(self._mgr, "lineage", None),
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
