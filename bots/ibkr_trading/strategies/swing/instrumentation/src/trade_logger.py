"""Trade Event Logger — structured trade events with full context.

Wraps the bot's existing entry/exit logic to capture WHY a trade was taken
and what the market looked like.  The wrapper is transparent: same inputs,
same outputs, same side effects.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from .event_metadata import EventMetadata, create_event_metadata
from .market_snapshot import MarketSnapshot, MarketSnapshotService
from libs.instrumentation.event_contract import enrich_payload, write_error_event
from libs.instrumentation.trade_completion import enrich_trade_completion
from libs.instrumentation.lineage import lineage_from_config
from libs.oms.instrumentation.correlation_snapshot import (
    capture_concurrent_positions_from_coordinator,
)

logger = logging.getLogger("instrumentation.trade_logger")


@dataclass
class TradeEvent:
    """Complete record of a single trade from entry to exit.

    Created at entry time with exit fields as None.
    Updated at exit time to fill in exit data.
    Written to JSONL at both entry and exit (as separate events).
    """
    # Identity + timing
    trade_id: str
    bot_id: str = ""
    event_metadata: dict = field(default_factory=dict)
    entry_snapshot: dict = field(default_factory=dict)
    exit_snapshot: Optional[dict] = None

    # Trade data
    pair: str = ""
    side: str = ""                          # "LONG" or "SHORT"
    strategy_id: str = ""
    entry_time: str = ""
    exit_time: Optional[str] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    position_size: float = 0.0
    position_size_quote: float = 0.0
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    fees_paid: Optional[float] = None

    # WHY — critical instrumentation
    entry_signal: str = ""
    entry_signal_id: str = ""
    entry_signal_strength: float = 0.0      # 0.0-1.0
    exit_reason: str = ""                   # SIGNAL | STOP_LOSS | TAKE_PROFIT | TRAILING | TIMEOUT | MANUAL | STALL | CATASTROPHIC | BIAS_FLIP
    market_regime: str = ""
    macro_regime: str = ""                    # G/R/S/D from RegimeService
    stress_level_at_entry: float = 0.0        # 0-1 from RegimeService

    # Filters
    active_filters: List[str] = field(default_factory=list)
    passed_filters: List[str] = field(default_factory=list)
    blocked_by: Optional[str] = None

    # Context at entry
    atr_at_entry: Optional[float] = None
    spread_at_entry_bps: Optional[float] = None
    volume_24h_at_entry: Optional[float] = None
    funding_rate_at_entry: Optional[float] = None
    open_interest_at_entry: Optional[float] = None

    # Strategy config snapshot
    strategy_params_at_entry: Optional[dict] = None

    # Enriched instrumentation
    signal_factors: List[dict] = field(default_factory=list)       # what factors contributed to entry
    filter_decisions: List[dict] = field(default_factory=list)     # how close each filter was to blocking
    sizing_inputs: Optional[dict] = None                           # what drove the size decision
    portfolio_state_at_entry: Optional[dict] = None                # exposure, direction, correlated positions

    # Post-exit price tracking (backfilled by PostExitTracker)
    post_exit_1h_pct: Optional[float] = None
    post_exit_4h_pct: Optional[float] = None
    post_exit_1h_price: Optional[float] = None
    post_exit_4h_price: Optional[float] = None

    # Execution quality
    expected_entry_price: Optional[float] = None
    entry_slippage_bps: Optional[float] = None
    expected_exit_price: Optional[float] = None
    exit_slippage_bps: Optional[float] = None
    entry_latency_ms: Optional[int] = None
    exit_latency_ms: Optional[int] = None
    execution_timeline: Optional[dict] = None  # {signal_generated_at, oms_received_at, order_submitted_at, fill_confirmed_at}
    execution_timestamps: Optional[dict] = None
    runtime_join_refs: Optional[dict] = None
    decision_ref: str = ""
    action_ref: str = ""
    portfolio_decision_ref: str = ""
    intent_id: str = ""
    order_ids: List[str] = field(default_factory=list)
    fill_ids: List[str] = field(default_factory=list)
    artifact_hash: str = ""
    resource_plan_hash: str = ""

    # Excursion tracking (B5, B9)
    mfe_price: Optional[float] = None
    mae_price: Optional[float] = None
    mfe_pct: Optional[float] = None
    mae_pct: Optional[float] = None
    mfe_r: Optional[float] = None
    mae_r: Optional[float] = None
    exit_efficiency: Optional[float] = None  # actual_pnl_pct / mfe_pct

    # Drawdown context (B8)
    drawdown_pct_at_entry: Optional[float] = None
    drawdown_tier_at_entry: Optional[str] = None  # NORMAL/CAUTION/DANGER/HALT
    position_size_multiplier: Optional[float] = None

    # Session context (S4)
    market_session: Optional[str] = None  # PRE/RTH/ETH_POST/WEEKEND
    minutes_into_session: Optional[int] = None

    # Overnight gap (B10)
    overnight_gap_pct: Optional[float] = None
    prev_close_price: Optional[float] = None

    # Overlay macro regime
    overlay_state: Optional[dict] = None  # {"qqq_ema_bullish": bool, "gld_ema_bullish": bool}

    # Metadata (B11, S4)
    experiment_id: Optional[str] = None
    experiment_variant: Optional[str] = None
    concurrent_positions_strategy: Optional[int] = None
    correlated_pairs_detail: Optional[list] = None

    # Signal evolution and fill details (for SignalHealthAnalyzer / FillQualityAnalyzer)
    signal_evolution: Optional[List[dict]] = None
    entry_fill_details: Optional[dict] = None
    exit_fill_details: Optional[dict] = None

    # Process quality (merged from scorer for TA compatibility)
    process_quality_score: Optional[int] = None
    root_causes: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)

    # Event stage
    stage: str = "entry"                    # "entry" or "exit"

    def to_dict(self) -> dict:
        d = asdict(self)
        # Add trading_assistant-compatible alias fields.
        # TA's TradeEvent Pydantic model expects these exact names; without them
        # the event silently fails validation and is dropped from analysis.
        d["market_snapshot"] = d.get("entry_snapshot")
        d["spread_at_entry"] = d.get("spread_at_entry_bps") or 0.0
        d["volume_24h"] = d.get("volume_24h_at_entry") or 0.0
        d["funding_rate"] = d.get("funding_rate_at_entry") or 0.0
        d["open_interest_delta"] = d.get("open_interest_at_entry") or 0.0
        d["signal_id"] = d.get("entry_signal_id", "")

        # TA expects process_quality_score as int (default 100 when absent).
        # Emit 100 when None to prevent Pydantic validation failures.
        if d.get("process_quality_score") is None:
            d["process_quality_score"] = 100
        return d


class TradeLogger:
    """Captures trade events by wrapping the bot's entry/exit functions.

    Usage::

        logger = TradeLogger(config, snapshot_service)
        trade = logger.log_entry(trade_id="abc", pair="QQQ", ...)
        logger.log_exit(trade_id="abc", exit_price=510.0, ...)
    """

    def __init__(self, config: dict, snapshot_service: MarketSnapshotService,
                 coordinator=None):
        self.bot_id = config["bot_id"]
        self.strategy_id = config.get("strategy_id", "")
        self.data_dir = Path(config["data_dir"]) / "trades"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_service = snapshot_service
        self.data_source_id = config.get("data_source_id", "ibkr_execution")
        self._coordinator = coordinator
        self._lineage = lineage_from_config(
            config,
            family_id="swing",
            strategy_id=self.strategy_id,
        )
        self._open_trades: Dict[str, TradeEvent] = {}

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
        strategy_id: str = "",
        exchange_timestamp: Optional[datetime] = None,
        expected_entry_price: Optional[float] = None,
        entry_latency_ms: Optional[int] = None,
        market_regime: str = "",
        bar_id: Optional[str] = None,
        signal_factors: Optional[List[dict]] = None,
        filter_decisions: Optional[List[dict]] = None,
        sizing_inputs: Optional[dict] = None,
        portfolio_state_at_entry: Optional[dict] = None,
        signal_evolution: Optional[List[dict]] = None,
        **kwargs,
    ) -> TradeEvent:
        """Call immediately after a trade entry is confirmed (fill received)."""
        try:
            now = datetime.now(timezone.utc)
            exch_ts = exchange_timestamp or now

            entry_snapshot = self.snapshot_service.capture_now(pair)

            entry_slippage_bps = None
            if expected_entry_price and expected_entry_price > 0:
                entry_slippage_bps = abs(entry_price - expected_entry_price) / expected_entry_price * 10000

            metadata = create_event_metadata(
                bot_id=self.bot_id,
                event_type="trade",
                payload_key=f"{trade_id}_entry",
                exchange_timestamp=exch_ts,
                data_source_id=self.data_source_id,
                bar_id=bar_id,
                lineage=self._lineage,
            )

            trade = TradeEvent(
                trade_id=trade_id,
                bot_id=self.bot_id,
                event_metadata=metadata.to_dict(),
                entry_snapshot=entry_snapshot.to_dict(),
                pair=pair,
                side=side,
                strategy_id=strategy_id,
                entry_time=exch_ts.isoformat(),
                entry_price=entry_price,
                position_size=position_size,
                position_size_quote=position_size_quote,
                entry_signal=entry_signal,
                entry_signal_id=entry_signal_id,
                entry_signal_strength=entry_signal_strength,
                market_regime=market_regime,
                active_filters=active_filters,
                passed_filters=passed_filters,
                atr_at_entry=entry_snapshot.atr_14,
                spread_at_entry_bps=entry_snapshot.spread_bps,
                volume_24h_at_entry=entry_snapshot.volume_24h,
                funding_rate_at_entry=entry_snapshot.funding_rate,
                open_interest_at_entry=entry_snapshot.open_interest,
                strategy_params_at_entry=strategy_params,
                signal_factors=signal_factors or [],
                filter_decisions=filter_decisions or [],
                sizing_inputs=sizing_inputs,
                portfolio_state_at_entry=portfolio_state_at_entry,
                expected_entry_price=expected_entry_price,
                entry_slippage_bps=round(entry_slippage_bps, 2) if entry_slippage_bps else None,
                entry_latency_ms=entry_latency_ms,
                stage="entry",
                **{k: v for k, v in kwargs.items()
                   if k in {f.name for f in fields(TradeEvent)}},
            )

            # Assemble entry fill details for FillQualityAnalyzer
            trade.entry_fill_details = {
                "order_id": kwargs.get("fill_order_id"),
                "fill_id": kwargs.get("fill_id") or kwargs.get("entry_fill_id") or kwargs.get("exec_id"),
                "fill_qty": kwargs.get("fill_qty") or position_size,
                "fill_price": entry_price,
                "fill_time_ms": kwargs.get("fill_time_ms"),
                "slippage_bps": round(entry_slippage_bps, 2) if entry_slippage_bps is not None else None,
                "fill_latency_ms": entry_latency_ms,
                "fill_type": kwargs.get("fill_type", "limit"),
            }
            if signal_evolution is not None:
                trade.signal_evolution = signal_evolution

            # Populate correlated_pairs_detail from coordinator (shared OMS)
            try:
                corr = capture_concurrent_positions_from_coordinator(
                    self._coordinator,
                    current_strategy_id=strategy_id or self.strategy_id,
                    current_symbol=pair,
                )
                if corr:
                    trade.correlated_pairs_detail = corr
            except Exception as e:
                logger.warning("Failed to capture concurrent positions: %s", e)

            self._open_trades[trade_id] = trade
            self._write_event(trade)
            return trade

        except Exception as e:
            self._write_error("log_entry", trade_id, e)
            return TradeEvent(trade_id=trade_id, bot_id=self.bot_id)

    def log_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        fees_paid: float = 0.0,
        exchange_timestamp: Optional[datetime] = None,
        expected_exit_price: Optional[float] = None,
        exit_latency_ms: Optional[int] = None,
        **kwargs,
    ) -> Optional[TradeEvent]:
        """Call immediately after a trade exit is confirmed."""
        try:
            trade = self._open_trades.pop(trade_id, None)
            if trade is None:
                self._write_error("log_exit", trade_id,
                    Exception(f"No open trade found for trade_id={trade_id}"))
                return None

            now = datetime.now(timezone.utc)
            exch_ts = exchange_timestamp or now

            exit_snapshot = self.snapshot_service.capture_now(trade.pair)

            if trade.side == "LONG":
                pnl = (exit_price - trade.entry_price) * trade.position_size - fees_paid
                pnl_pct = (exit_price - trade.entry_price) / trade.entry_price * 100 if trade.entry_price else 0
            else:
                pnl = (trade.entry_price - exit_price) * trade.position_size - fees_paid
                pnl_pct = (trade.entry_price - exit_price) / trade.entry_price * 100 if trade.entry_price else 0

            exit_slippage_bps = None
            if expected_exit_price and expected_exit_price > 0:
                exit_slippage_bps = abs(exit_price - expected_exit_price) / expected_exit_price * 10000

            trade.exit_snapshot = exit_snapshot.to_dict()
            trade.exit_time = exch_ts.isoformat()
            trade.exit_price = exit_price
            trade.exit_reason = exit_reason
            trade.pnl = round(pnl, 4)
            trade.pnl_pct = round(pnl_pct, 4)
            trade.fees_paid = fees_paid
            trade.expected_exit_price = expected_exit_price
            trade.exit_slippage_bps = round(exit_slippage_bps, 2) if exit_slippage_bps else None
            trade.exit_latency_ms = exit_latency_ms

            # Assemble exit fill details for FillQualityAnalyzer
            trade.exit_fill_details = {
                "order_id": kwargs.get("fill_order_id"),
                "fill_id": kwargs.get("fill_id") or kwargs.get("exit_fill_id") or kwargs.get("exec_id"),
                "fill_qty": kwargs.get("fill_qty") or trade.position_size,
                "fill_price": exit_price,
                "fill_time_ms": kwargs.get("fill_time_ms"),
                "slippage_bps": round(exit_slippage_bps, 2) if exit_slippage_bps is not None else None,
                "fill_latency_ms": exit_latency_ms,
                "fill_type": "stop" if exit_reason.startswith("STOP") else "market",
            }

            trade.stage = "exit"

            trade.event_metadata = create_event_metadata(
                bot_id=self.bot_id,
                event_type="trade",
                payload_key=f"{trade_id}_exit",
                exchange_timestamp=exch_ts,
                data_source_id=self.data_source_id,
                lineage=self._lineage,
            ).to_dict()

            # Ensure bot_id is on the exit record
            trade.bot_id = self.bot_id

            # Apply enriched fields from kwargs
            for k, v in kwargs.items():
                if hasattr(trade, k):
                    setattr(trade, k, v)

            self._write_event(trade)
            return trade

        except Exception as e:
            self._write_error("log_exit", trade_id, e)
            return None

    def get_open_trades(self) -> Dict[str, TradeEvent]:
        return dict(self._open_trades)

    def amend_last_event(self, trade_id: str, updates: dict) -> None:
        """Amend the last written event if it matches trade_id.

        Used to merge process quality score onto the exit record after scoring.
        """
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filepath = self.data_dir / f"trades_{today}.jsonl"
            if not filepath.exists():
                return
            text = filepath.read_text(encoding="utf-8").rstrip("\n")
            if not text:
                return
            lines = text.split("\n")
            event = json.loads(lines[-1])
            if event.get("trade_id") == trade_id:
                event.update(updates)
                event = enrich_payload(
                    event,
                    lineage=self._lineage,
                    event_type="trade",
                    scope="strategy",
                )
                event = enrich_trade_completion(event)
                lines[-1] = json.dumps(event, default=str)
                filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to amend trade event: %s", e)

    def _write_event(self, trade: TradeEvent) -> None:
        """Append trade event to daily JSONL file."""
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filepath = self.data_dir / f"trades_{today}.jsonl"
            event_type = "trade_entry" if str(trade.stage).lower() == "entry" else "trade"
            payload = enrich_payload(
                trade.to_dict(),
                lineage=self._lineage,
                event_type=event_type,
                scope="strategy",
            )
            if event_type == "trade":
                payload = enrich_trade_completion(payload)
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        except Exception as e:
            logger.warning("Failed to write trade event: %s", e)

    def _write_error(self, method: str, trade_id: str, error: Exception) -> None:
        """Log instrumentation errors without crashing."""
        try:
            write_error_event(
                Path(self.data_dir).parent,
                self._lineage,
                component="trade_logger",
                method=method,
                message=str(error),
                error_type=type(error).__name__,
                context={"trade_id": trade_id},
                exc=error,
            )
        except Exception:
            pass
