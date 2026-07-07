"""InstrumentationCollector — accumulates gate decisions and market context per bar cycle."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from crypto_trader.core.runtime_types import TradeOutcome
from crypto_trader.instrumentation.lineage import LineageContext, stable_hash
from crypto_trader.instrumentation.pipeline_tracker import PipelineTracker
from crypto_trader.instrumentation.types import (
    EventMetadata,
    FilterDecision,
    GenericInstrumentationEvent,
    InstrumentedTradeEvent,
    MarketContext,
    MissedOpportunityEvent,
    SignalFactor,
)

if TYPE_CHECKING:
    from crypto_trader.core.models import Trade
    from crypto_trader.instrumentation.emitter import EventEmitter


class InstrumentationCollector:
    """Accumulates gate decisions and market context per bar cycle.

    Zero I/O in hot path — everything stored in-memory lists.
    """

    def __init__(
        self,
        strategy_id: str,
        bot_id: str = "",
        lineage: LineageContext | dict | None = None,
    ) -> None:
        self._strategy_id = strategy_id
        self._bot_id = bot_id
        self._lineage_context = lineage
        self._lineage = self._lineage_for_strategy()

        # Pipeline funnel tracker
        self._pipeline = PipelineTracker(strategy_id)

        # Per-bar-cycle accumulator (reset each primary TF bar)
        self._current_decisions: dict[str, list[FilterDecision]] = {}
        self._current_context: dict[str, MarketContext] = {}
        self._current_signal_factors: dict[str, list[SignalFactor]] = {}
        self._current_bar_close: dict[str, float] = {}
        self._current_bar_id: dict[str, str] = {}
        self._current_decision_id: dict[str, str] = {}
        self._current_signal_id: dict[str, str] = {}
        self._current_timeframe: dict[str, str] = {}
        self._current_exchange_ts: dict[str, datetime] = {}
        self._active_decision_context: dict[str, object] = {}
        self._last_regime_state: dict[str, dict[str, object]] = {}

        # Per-position accumulator (persists across bars until trade closes)
        self._entry_decisions: dict[str, list[FilterDecision]] = {}
        self._entry_context: dict[str, MarketContext] = {}
        self._entry_signal_factors: dict[str, list[SignalFactor]] = {}
        self._entry_config: dict[str, dict] = {}
        self._entry_sizing_inputs: dict[str, dict] = {}
        self._entry_portfolio_state: dict[str, dict | None] = {}
        self._entry_signal_strength: dict[str, float] = {}
        self._entry_bar_id: dict[str, str] = {}
        self._entry_decision_id: dict[str, str] = {}
        self._entry_signal_id: dict[str, str] = {}
        self._entry_lineage: dict[str, dict] = {}

        # Missed opportunities buffer (flushed by emitter)
        self._missed_buffer: list[MissedOpportunityEvent] = []

        # Emitter reference (set by engine/runner)
        self._emitter: EventEmitter | None = None

    @property
    def emitter(self) -> EventEmitter | None:
        return self._emitter

    @emitter.setter
    def emitter(self, value: EventEmitter) -> None:
        self._emitter = value

    @property
    def pipeline(self) -> PipelineTracker:
        """Read-only access to the pipeline tracker."""
        return self._pipeline

    def set_lineage(self, lineage: LineageContext | dict | None) -> None:
        """Set runtime lineage once the live engine has loaded config versions."""
        self._lineage_context = lineage
        self._lineage = self._lineage_for_strategy()

    def begin_decision_context(self, context: object) -> None:
        """Receive deterministic decision context from StrategySlotRuntime."""
        symbol = str(getattr(context, "symbol", ""))
        if symbol:
            self._active_decision_context[symbol] = context

    def end_decision_context(self, context: object) -> None:
        """End the active runtime decision context for this callback."""
        symbol = str(getattr(context, "symbol", ""))
        if symbol and self._active_decision_context.get(symbol) is context:
            self._active_decision_context.pop(symbol, None)

    def begin_bar(self, sym: str, bar_close: float = 0.0) -> None:
        """Reset per-bar accumulator for this symbol."""
        self._current_decisions[sym] = []
        self._current_signal_factors[sym] = []
        self._current_bar_close[sym] = bar_close
        context = self._active_decision_context.get(sym)
        decision_id = str(getattr(context, "decision_id", "")) if context is not None else ""
        decision_time = getattr(context, "decision_time", None) if context is not None else None
        timeframe = getattr(getattr(context, "timeframe", None), "value", "") if context is not None else ""
        metadata = getattr(context, "metadata", {}) if context is not None else {}
        bar_id = ""
        if isinstance(metadata, dict):
            bar_id = str(metadata.get("bar_id") or "")
        if not bar_id and decision_id:
            bar_id = stable_hash({"decision_id": decision_id, "symbol": sym})
        if not bar_id:
            bar_id = stable_hash({
                "strategy_id": self._strategy_id,
                "symbol": sym,
                "bar_close": bar_close,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        signal_id = stable_hash({
            "strategy_id": self._strategy_id,
            "symbol": sym,
            "decision_id": decision_id,
            "bar_id": bar_id,
        })
        self._current_bar_id[sym] = bar_id
        self._current_decision_id[sym] = decision_id
        self._current_signal_id[sym] = signal_id
        self._current_timeframe[sym] = timeframe
        self._current_exchange_ts[sym] = (
            decision_time if isinstance(decision_time, datetime) else datetime.now(timezone.utc)
        )
        self._pipeline.record_bar(sym)

    def record_gate(
        self,
        sym: str,
        gate_name: str,
        passed: bool,
        reason: str = "",
        threshold: float | None = None,
        actual_value: float | None = None,
        context: dict | None = None,
    ) -> None:
        """Record a gate evaluation with filter-level detail."""
        margin = None
        if threshold is not None and actual_value is not None and threshold != 0:
            margin = (actual_value - threshold) / abs(threshold) * 100

        self._current_decisions.setdefault(sym, []).append(
            FilterDecision(
                filter_name=gate_name,
                passed=passed,
                threshold=threshold,
                actual_value=actual_value,
                margin_pct=margin,
                reason=reason,
                context=context or {},
            )
        )
        self._pipeline.record_gate(sym, gate_name, passed)
        self._emit_filter_decision(sym, self._current_decisions[sym][-1], len(self._current_decisions[sym]) - 1)

    def snapshot_context(self, sym: str, indicators: object, **kwargs) -> None:
        """Capture market state from indicator snapshot + strategy-specific data."""
        self._current_context[sym] = MarketContext(
            atr=getattr(indicators, "atr", 0.0),
            adx=getattr(indicators, "adx", 0.0),
            rsi=getattr(indicators, "rsi", None),
            ema_fast=getattr(indicators, "ema_fast", 0.0),
            ema_mid=getattr(indicators, "ema_mid", 0.0),
            ema_slow=getattr(indicators, "ema_slow", 0.0),
            volume_ma=getattr(indicators, "volume_ma", 0.0),
            funding_rate=kwargs.get("funding_rate", 0.0),
            bias_direction=kwargs.get("bias_direction"),
            bias_strength=kwargs.get("bias_strength"),
            regime_tier=kwargs.get("regime_tier"),
            regime_direction=kwargs.get("regime_direction"),
            h4_context_direction=kwargs.get("h4_context_direction"),
            h4_context_strength=kwargs.get("h4_context_strength"),
            setup_grade=kwargs.get("setup_grade"),
            setup_confluences=kwargs.get("setup_confluences", []),
            setup_room_r=kwargs.get("setup_room_r"),
        )
        self._emit_indicator_snapshot(sym, self._current_context[sym])
        self._emit_regime_transition(sym, self._current_context[sym])

    def record_signal_factor(self, sym: str, factor: str, value: float) -> None:
        """Record a signal factor that drove the entry decision."""
        self._current_signal_factors.setdefault(sym, []).append(
            SignalFactor(factor=factor, value=value)
        )

    def record_entry(
        self,
        sym: str,
        config_dict: dict,
        sizing_inputs: dict,
        portfolio_state: dict | None = None,
        signal_strength: float = 0.0,
    ) -> None:
        """Freeze current gate decisions + context at entry submission time."""
        self._entry_decisions[sym] = list(self._current_decisions.get(sym, []))
        self._entry_context[sym] = self._current_context.get(sym)  # type: ignore[assignment]
        self._entry_signal_factors[sym] = list(
            self._current_signal_factors.get(sym, [])
        )
        self._entry_config[sym] = config_dict
        self._entry_sizing_inputs[sym] = sizing_inputs
        self._entry_portfolio_state[sym] = portfolio_state
        self._entry_signal_strength[sym] = signal_strength
        self._entry_bar_id[sym] = self._current_bar_id.get(sym, "")
        self._entry_decision_id[sym] = self._current_decision_id.get(sym, "")
        self._entry_signal_id[sym] = self._current_signal_id.get(sym, "")
        self._entry_lineage[sym] = dict(self._lineage)

    def on_trade_closed(
        self,
        sym: str,
        trade: Trade,
        process_score: int = 100,
        root_causes: list[str] | None = None,
    ) -> InstrumentedTradeEvent:
        """Build full instrumented event from accumulated data + trade outcome."""
        self._pipeline.record_trade_closed(sym)
        decisions = self._entry_decisions.pop(sym, [])
        ctx = self._entry_context.pop(sym, None)
        factors = self._entry_signal_factors.pop(sym, [])
        config = self._entry_config.pop(sym, {})
        sizing = self._entry_sizing_inputs.pop(sym, {})
        portfolio = self._entry_portfolio_state.pop(sym, None)
        strength = self._entry_signal_strength.pop(sym, 0.0)
        entry_bar_id = self._entry_bar_id.pop(sym, "")
        entry_decision_id = self._entry_decision_id.pop(sym, "")
        entry_signal_id = self._entry_signal_id.pop(sym, "")
        lineage = self._entry_lineage.pop(sym, dict(self._lineage))
        completion = getattr(trade, "instrumentation_context", None)
        completion = completion if isinstance(completion, dict) else {}

        passed = [d.filter_name for d in decisions if d.passed]
        active = [d.filter_name for d in decisions]

        outcome = TradeOutcome.from_trade(trade)
        reporting_r = (
            outcome.realized_r_net
            if outcome.realized_r_net is not None
            else outcome.geometric_r
        )
        exit_eff = None
        if (
            reporting_r is not None
            and trade.mfe_r is not None
            and reporting_r > 0
            and trade.mfe_r > 0
        ):
            exit_eff = reporting_r / trade.mfe_r

        pnl_pct = 0.0
        if trade.entry_price > 0 and trade.qty > 0:
            notional = trade.entry_price * trade.qty
            if notional > 0:
                pnl_pct = outcome.realized_pnl_net / notional * 100

        metadata = EventMetadata.create(
            bot_id=self._bot_id,
            strategy_id=self._strategy_id,
            exchange_ts=trade.exit_time,
            event_type="trade",
            payload_key=trade.trade_id,
            bar_id=entry_bar_id,
            lineage=lineage,
            **self._metadata_defaults(),
        )

        return InstrumentedTradeEvent(
            metadata=metadata,
            lineage=lineage,
            logical_event_id=trade.trade_id,
            trade_id=trade.trade_id,
            pair=trade.symbol,
            side=trade.direction.value,
            entry_decision_id=entry_decision_id or str(completion.get("entry_decision_id") or ""),
            exit_decision_id=str(completion.get("exit_decision_id") or ""),
            entry_signal_id=entry_signal_id,
            entry_bar_id=entry_bar_id,
            exit_bar_id=str(completion.get("exit_bar_id") or ""),
            entry_order_ids=list(completion.get("entry_order_ids") or []),
            exit_order_ids=list(completion.get("exit_order_ids") or []),
            entry_fill_ids=list(completion.get("entry_fill_ids") or []),
            exit_fill_ids=list(completion.get("exit_fill_ids") or []),
            client_order_ids=list(completion.get("client_order_ids") or []),
            exchange_order_ids=list(completion.get("exchange_order_ids") or []),
            intent_id=str(completion.get("intent_id") or ""),
            decision_ref=dict(completion.get("decision_ref") or {}),
            action_ref=dict(completion.get("action_ref") or {}),
            portfolio_decision_ref=dict(completion.get("portfolio_decision_ref") or {}),
            artifact_hash=str(completion.get("artifact_hash") or ""),
            resource_plan_hash=str(completion.get("resource_plan_hash") or ""),
            runtime_join=dict(completion.get("runtime_join") or {}),
            entry_time=trade.entry_time,
            exit_time=trade.exit_time,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            position_size=trade.qty,
            pnl=outcome.realized_pnl_net,
            price_pnl_gross=outcome.price_pnl_gross,
            total_fees=outcome.total_fees,
            price_pnl_after_funding=outcome.price_pnl_after_funding,
            realized_pnl_net=outcome.realized_pnl_net,
            pnl_pct=pnl_pct,
            r_multiple=reporting_r,
            realized_r_net=outcome.realized_r_net,
            geometric_r=outcome.geometric_r,
            commission=outcome.total_fees,
            funding_paid=outcome.funding_paid,
            entry_signal=trade.confirmation_type or "",
            entry_signal_strength=strength,
            setup_grade=trade.setup_grade.value if trade.setup_grade else "",
            exit_reason=trade.exit_reason,
            confluences=list(trade.confluences_used or []),
            entry_method=trade.entry_method or "",
            signal_factors=factors,
            filter_decisions=decisions,
            passed_filters=passed,
            active_filters=active,
            market_context=ctx,
            mfe_r=trade.mfe_r,
            mae_r=trade.mae_r,
            exit_efficiency=exit_eff,
            process_quality_score=process_score,
            root_causes=root_causes or [],
            strategy_params_at_entry=config,
            sizing_inputs=sizing,
            portfolio_state_at_entry=portfolio,
            portfolio_rule_event_id=str(completion.get("portfolio_rule_event_id") or ""),
            risk_decision_id=str(completion.get("risk_decision_id") or ""),
        )

    def end_bar(self, sym: str) -> None:
        """Finalize bar cycle. Detect missed opportunities from gate failures.

        A signal is "missed" when the pipeline progressed past the setup gate
        but was blocked by a downstream gate. If setup itself failed, there
        was no actionable signal to miss.
        """
        decisions = self._current_decisions.get(sym, [])
        if not decisions:
            return

        # Find setup gate
        setup_idx = None
        for i, d in enumerate(decisions):
            if d.filter_name == "setup":
                setup_idx = i
                break

        if setup_idx is None:
            return  # Pipeline didn't reach setup evaluation

        if not decisions[setup_idx].passed:
            return  # No setup found — not a missed signal

        # Find first failing gate AFTER setup
        blocker = None
        for d in decisions[setup_idx + 1 :]:
            if not d.passed:
                blocker = d
                break

        if blocker is None:
            return  # All gates passed — entry was submitted, not a miss

        # Build MissedOpportunityEvent
        ctx = self._current_context.get(sym)
        grade_str = ctx.setup_grade if ctx else ""
        room_r = ctx.setup_room_r if ctx else 0.0

        blocker_idx = decisions.index(blocker)
        metadata = EventMetadata.create(
            bot_id=self._bot_id,
            strategy_id=self._strategy_id,
            exchange_ts=self._current_exchange_ts.get(sym, datetime.now(timezone.utc)),
            event_type="missed_opportunity",
            payload_key=f"{self._current_signal_id.get(sym, sym)}:{blocker.filter_name}:revision:0",
            bar_id=self._current_bar_id.get(sym, ""),
            lineage=self._lineage,
            **self._metadata_defaults(),
        )
        logical_id = stable_hash({
            "strategy_id": self._strategy_id,
            "symbol": sym,
            "decision_id": self._current_decision_id.get(sym, ""),
            "bar_id": self._current_bar_id.get(sym, ""),
            "blocker": blocker.filter_name,
        })

        missed = MissedOpportunityEvent(
            metadata=metadata,
            lineage=dict(self._lineage),
            opportunity_id=logical_id,
            logical_event_id=logical_id,
            pair=sym,
            symbol=sym,
            timeframe=self._current_timeframe.get(sym, ""),
            bar_id=self._current_bar_id.get(sym, ""),
            decision_id=self._current_decision_id.get(sym, ""),
            signal_id=self._current_signal_id.get(sym, ""),
            signal=f"{self._strategy_id}_{grade_str}" if grade_str else self._strategy_id,
            signal_strength=room_r or 0.0,
            blocked_by=blocker.filter_name,
            block_reason=blocker.reason,
            blocking_rule_type="strategy_filter",
            margin_pct=blocker.margin_pct,
            hypothetical_entry=self._current_bar_close.get(sym, 0.0),
            simulation_policy={
                "entry_price_source": "bar_close",
                "outcome_windows": ["1h", "4h", "24h"],
            },
            market_context=ctx,
            filter_decisions=list(decisions[: blocker_idx + 1]),
            backfill_status="pending",
        )
        self._missed_buffer.append(missed)

        # Auto-emit if emitter is wired
        if self._emitter is not None:
            self._emitter.emit_missed(missed)

    def flush_missed(self) -> list[MissedOpportunityEvent]:
        """Return and clear the missed opportunity buffer."""
        buf = self._missed_buffer
        self._missed_buffer = []
        return buf

    def _lineage_for_strategy(self) -> dict[str, object]:
        if isinstance(self._lineage_context, LineageContext):
            return self._lineage_context.for_strategy(self._strategy_id)
        if isinstance(self._lineage_context, dict):
            lineage = dict(self._lineage_context)
            lineage.setdefault("strategy_id", self._strategy_id)
            return lineage
        return {"strategy_id": self._strategy_id}

    def _metadata_defaults(self) -> dict:
        if isinstance(self._lineage_context, LineageContext):
            defaults = self._lineage_context.metadata_defaults(self._strategy_id)
            defaults.pop("lineage", None)
            return defaults
        return {
            "family_id": str(self._lineage.get("family_id", "crypto_perps")),
            "portfolio_id": str(self._lineage.get("portfolio_id", "default")),
            "account_alias": str(self._lineage.get("account_alias", "default")),
            "config_version": str(self._lineage.get("config_version", "")),
            "deployment_id": str(self._lineage.get("deployment_id", "")),
            "code_sha": str(self._lineage.get("code_sha", "")),
        }

    def _emit_filter_decision(self, sym: str, decision: FilterDecision, index: int) -> None:
        if self._emitter is None:
            return
        decision_id = self._current_decision_id.get(sym, "")
        bar_id = self._current_bar_id.get(sym, "")
        payload = {
            "filter_event_id": stable_hash({
                "decision_id": decision_id,
                "bar_id": bar_id,
                "filter_name": decision.filter_name,
                "filter_index": index,
            }),
            "decision_id": decision_id,
            "bar_id": bar_id,
            "signal_id": self._current_signal_id.get(sym, ""),
            "strategy_id": self._strategy_id,
            "symbol": sym,
            "timeframe": self._current_timeframe.get(sym, ""),
            "filter_name": decision.filter_name,
            "filter_index": index,
            "passed": decision.passed,
            "threshold": decision.threshold,
            "actual_value": decision.actual_value,
            "margin_pct": decision.margin_pct,
            "reason": decision.reason,
            "context": dict(decision.context),
        }
        self._emit_generic("filter_decision", sym, payload, payload["filter_event_id"])

    def _emit_indicator_snapshot(self, sym: str, context: MarketContext) -> None:
        if self._emitter is None:
            return
        event_id = stable_hash({
            "event_type": "indicator_snapshot",
            "strategy_id": self._strategy_id,
            "symbol": sym,
            "bar_id": self._current_bar_id.get(sym, ""),
            "decision_id": self._current_decision_id.get(sym, ""),
        })
        payload = {
            "indicator_snapshot_id": event_id,
            "decision_id": self._current_decision_id.get(sym, ""),
            "bar_id": self._current_bar_id.get(sym, ""),
            "signal_id": self._current_signal_id.get(sym, ""),
            "strategy_id": self._strategy_id,
            "symbol": sym,
            "timeframe": self._current_timeframe.get(sym, ""),
            "market_context": context.to_dict(),
        }
        self._emit_generic("indicator_snapshot", sym, payload, event_id)

    def _emit_regime_transition(self, sym: str, context: MarketContext) -> None:
        if self._emitter is None:
            return
        new_state = self._regime_state(context)
        if not new_state:
            return
        previous_state = self._last_regime_state.get(sym)
        if previous_state == new_state:
            return
        transition_kind = "initial" if previous_state is None else "change"
        changed_fields = sorted(
            key
            for key in set(new_state) | set(previous_state or {})
            if new_state.get(key) != (previous_state or {}).get(key)
        )
        event_id = stable_hash({
            "event_type": "regime_transition",
            "strategy_id": self._strategy_id,
            "symbol": sym,
            "bar_id": self._current_bar_id.get(sym, ""),
            "decision_id": self._current_decision_id.get(sym, ""),
            "new_state": new_state,
            "previous_state": previous_state or {},
        })
        payload = {
            "regime_transition_id": event_id,
            "transition_kind": transition_kind,
            "decision_id": self._current_decision_id.get(sym, ""),
            "bar_id": self._current_bar_id.get(sym, ""),
            "signal_id": self._current_signal_id.get(sym, ""),
            "strategy_id": self._strategy_id,
            "symbol": sym,
            "timeframe": self._current_timeframe.get(sym, ""),
            "previous_state": dict(previous_state or {}),
            "new_state": dict(new_state),
            "changed_fields": changed_fields,
            "market_context": context.to_dict(),
        }
        self._last_regime_state[sym] = dict(new_state)
        self._emit_generic("regime_transition", sym, payload, event_id)

    @staticmethod
    def _regime_state(context: MarketContext) -> dict[str, object]:
        state: dict[str, object] = {}
        for key in (
            "bias_direction",
            "regime_tier",
            "regime_direction",
            "h4_context_direction",
            "h4_context_strength",
        ):
            value = getattr(context, key)
            if value not in (None, ""):
                state[key] = value
        if context.bias_strength is not None:
            state["bias_strength_bucket"] = round(float(context.bias_strength), 2)
        return state

    def _emit_generic(self, event_type: str, sym: str, payload: dict, payload_key: str) -> None:
        if self._emitter is None:
            return
        metadata = EventMetadata.create(
            bot_id=self._bot_id,
            strategy_id=self._strategy_id,
            exchange_ts=self._current_exchange_ts.get(sym, datetime.now(timezone.utc)),
            event_type=event_type,
            payload_key=payload_key,
            bar_id=self._current_bar_id.get(sym, ""),
            lineage=self._lineage,
            **self._metadata_defaults(),
        )
        self._emitter.emit(
            event_type,
            GenericInstrumentationEvent(
                metadata=metadata,
                payload=payload,
                lineage=dict(self._lineage),
                logical_event_id=payload_key,
            ),
        )
