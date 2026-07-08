import hashlib
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import yaml

from .event_metadata import EventMetadata, create_event_metadata
from .market_snapshot import MarketSnapshot, MarketSnapshotService
from libs.instrumentation.event_contract import enrich_payload, write_error_event
from libs.instrumentation.trade_completion import enrich_trade_completion
from libs.instrumentation.lineage import lineage_from_config
from libs.oms.instrumentation.correlation_snapshot import (
    capture_concurrent_positions,
    run_async_safely,
)

logger = logging.getLogger("instrumentation.trade_logger")

_sector_map_cache: dict | None = None


def _load_sector_map() -> dict:
    """Load and cache the sector map from config/sector_map.yaml."""
    global _sector_map_cache
    if _sector_map_cache is not None:
        return _sector_map_cache
    try:
        path = Path(__file__).resolve().parents[4] / "config" / "sector_map.yaml"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                _sector_map_cache = yaml.safe_load(f) or {}
        else:
            _sector_map_cache = {}
    except Exception as e:
        logger.warning("Failed to load sector_map.yaml: %s", e)
        _sector_map_cache = {}
    return _sector_map_cache


@dataclass
class TradeEvent:
    """
    Complete record of a single trade from entry to exit.
    Written to JSONL at both entry and exit (as separate events).
    """
    trade_id: str
    event_metadata: dict
    entry_snapshot: dict
    market_snapshot: Optional[dict] = None
    exit_snapshot: Optional[dict] = None
    bot_id: str = ""

    pair: str = ""
    side: str = ""                          # "LONG" or "SHORT"
    entry_time: str = ""
    exit_time: Optional[str] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    position_size: float = 0.0
    position_size_quote: float = 0.0
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    fees_paid: Optional[float] = None

    # WHY
    entry_signal: str = ""
    entry_signal_id: str = ""
    entry_signal_strength: float = 0.0
    exit_reason: str = ""
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
    spread_at_entry: Optional[float] = None
    volume_24h_at_entry: Optional[float] = None
    volume_24h: Optional[float] = None
    funding_rate_at_entry: Optional[float] = None
    funding_rate: Optional[float] = None
    open_interest_at_entry: Optional[float] = None
    open_interest_delta: Optional[float] = None

    # Process quality scoring
    process_quality_score: int = 100
    root_causes: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)

    # Strategy config snapshot
    strategy_params_at_entry: Optional[dict] = None

    # Signal evolution: last N bars of signal component values before entry (M2)
    # Each dict: {bars_ago: int, close: float, ...strategy-specific signal components}
    signal_evolution: Optional[List[dict]] = None

    # Signal confluence factors (highest-impact #1)
    # Each dict: {factor_name: str, factor_value: float, threshold: float, contribution: float}
    signal_factors: List[dict] = field(default_factory=list)

    # IARIC conviction factor decomposition (raw component scores)
    conviction_factors: Optional[dict] = None

    # Filter threshold context (highest-impact #2)
    # Each dict: {filter_name: str, threshold: float, actual_value: float, passed: bool, margin_pct: float}
    filter_decisions: List[dict] = field(default_factory=list)

    # Position sizing inputs (highest-impact #3)
    # Dict: {target_risk_pct: float, account_equity: float, volatility_basis: float,
    #         sizing_model: str, unit_risk_usd: float, setup_size_mult: float,
    #         session_size_mult: float, hour_mult: float, dow_mult: float, dd_mult: float}
    sizing_inputs: Optional[dict] = None

    # Portfolio state at entry (G4)
    portfolio_state_at_entry: Optional[dict] = None

    # Futures-specific context (critical gap #5)
    session_type: str = ""           # "RTH" / "ETH" / specific block name
    contract_month: str = ""         # e.g. "2026-03" or "MARCH_2026"
    margin_used_pct: Optional[float] = None

    # Concurrent position tracking (critical gap #4)
    concurrent_positions_at_entry: Optional[int] = None
    correlated_pairs_detail: Optional[list] = None

    # Sector metadata (for TA portfolio-level sector exposure analysis)
    sector: str = ""
    industry: str = ""

    # Drawdown state at entry (critical gap #3)
    drawdown_pct: Optional[float] = None
    drawdown_tier: str = ""          # "full" / "half" / "quarter" / "halt"
    drawdown_size_mult: Optional[float] = None

    # Post-exit price tracking (highest-impact #5)
    post_exit_1h_price: Optional[float] = None
    post_exit_4h_price: Optional[float] = None
    post_exit_1h_move_pct: Optional[float] = None
    post_exit_4h_move_pct: Optional[float] = None
    post_exit_backfill_status: str = "pending"

    # MFE/MAE (Gap G1, B5)
    mfe_r: Optional[float] = None           # peak favorable excursion in R-multiples
    mae_r: Optional[float] = None           # peak adverse excursion in R-multiples
    mfe_price: Optional[float] = None       # price at peak MFE
    mae_price: Optional[float] = None       # price at peak MAE
    mfe_pct: Optional[float] = None
    mae_pct: Optional[float] = None

    # Exit efficiency (B9) - actual_pnl_r / mfe_r
    exit_efficiency: Optional[float] = None

    # Per-order fill detail (G6)
    entry_fill_details: Optional[dict] = None
    exit_fill_details: Optional[dict] = None

    # Order book depth at entry (G7)
    order_book_depth_at_entry: Optional[dict] = None

    # Market conditions at entry (G8)
    market_conditions_at_entry: Optional[dict] = None

    # Execution quality
    expected_entry_price: Optional[float] = None
    entry_slippage_bps: Optional[float] = None
    expected_exit_price: Optional[float] = None
    exit_slippage_bps: Optional[float] = None
    entry_latency_ms: Optional[int] = None
    exit_latency_ms: Optional[int] = None

    # Experiment tracking (G5, B11)
    experiment_id: Optional[str] = None
    experiment_variant: Optional[str] = None

    # Execution cascade timestamps (#16)
    # {signal_detected_at, intent_created_at, risk_checked_at, order_submitted_at, fill_received_at}
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

    # Session transition tracking (#17)
    # Each: {from_session, to_session, transition_time, unrealized_pnl_r, bars_held, price_at_transition}
    session_transitions: Optional[List[dict]] = None

    # Strategy identification
    strategy_id: str = ""                # "IARIC_v1" / "ALCB_v1"
    strategy_type: str = ""
    param_set_id: Optional[str] = None   # sha256[:16] of strategy_params for grouping

    # Event stage
    stage: str = "entry"

    def to_dict(self) -> dict:
        d = asdict(self)
        # Add TA-compatible signal_id alias (TA expects signal_id, we emit entry_signal_id)
        d["signal_id"] = d.get("entry_signal_id", "")
        return d


class TradeLogger:
    """
    Captures trade events by wrapping the bot's entry/exit functions.

    Usage:
        logger = TradeLogger(config, snapshot_service)
        trade = logger.log_entry(trade_id="abc", pair="NQ", ...)
        logger.log_exit(trade_id="abc", exit_price=21000.0, ...)
    """

    def __init__(self, config: dict, snapshot_service: MarketSnapshotService,
                 process_scorer=None, strategy_type: str = "", error_logger=None,
                 pg_store=None, family_strategy_ids: list[str] | None = None):
        self.bot_id = config["bot_id"]
        self.strategy_id = config.get("strategy_id", "")
        self.data_dir = Path(config["data_dir"]) / "trades"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_service = snapshot_service
        self.data_source_id = config.get("data_source_id", "ibkr_cme_nq")
        self.process_scorer = process_scorer
        self.strategy_type = strategy_type or config.get("strategy_type", "unknown")
        self.experiment_id = config.get("experiment_id")
        self.experiment_variant = config.get("experiment_variant")
        self._error_logger = error_logger
        self._pg_store = pg_store
        self._family_strategy_ids = family_strategy_ids or []
        self._sector_map = _load_sector_map()
        self._lineage = lineage_from_config(
            config,
            family_id="stock",
            strategy_id=self.strategy_id,
        )
        self._open_trades: Dict[str, TradeEvent] = {}
        self._pending_exit_backfills: list[dict] = []

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
        exchange_timestamp: Optional[datetime] = None,
        expected_entry_price: Optional[float] = None,
        entry_latency_ms: Optional[int] = None,
        market_regime: str = "",
        macro_regime: str = "",
        stress_level_at_entry: float = 0.0,
        bar_id: Optional[str] = None,
        portfolio_state: Optional[dict] = None,
        signal_factors: Optional[List[dict]] = None,
        filter_decisions: Optional[List[dict]] = None,
        conviction_factors: Optional[dict] = None,
        sizing_inputs: Optional[dict] = None,
        session_type: str = "",
        contract_month: str = "",
        margin_used_pct: Optional[float] = None,
        concurrent_positions: Optional[int] = None,
        drawdown_pct: Optional[float] = None,
        drawdown_tier: str = "",
        drawdown_size_mult: Optional[float] = None,
        signal_evolution: Optional[List[dict]] = None,
        execution_timestamps: Optional[dict] = None,
        experiment_id: Optional[str] = None,
        experiment_variant: Optional[str] = None,
        **runtime_refs,
    ) -> TradeEvent:
        """Call this immediately after a trade entry is confirmed."""
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
                market_snapshot=entry_snapshot.to_dict(),
                pair=pair,
                side=side,
                entry_time=exch_ts.isoformat(),
                entry_price=entry_price,
                position_size=position_size,
                position_size_quote=position_size_quote,
                entry_signal=entry_signal,
                entry_signal_id=entry_signal_id,
                entry_signal_strength=entry_signal_strength,
                market_regime=market_regime,
                macro_regime=macro_regime,
                stress_level_at_entry=stress_level_at_entry,
                active_filters=active_filters,
                passed_filters=passed_filters,
                atr_at_entry=entry_snapshot.atr_14,
                spread_at_entry_bps=entry_snapshot.spread_bps,
                spread_at_entry=entry_snapshot.spread_bps,
                volume_24h_at_entry=entry_snapshot.volume_24h,
                volume_24h=entry_snapshot.volume_24h,
                funding_rate_at_entry=entry_snapshot.funding_rate,
                funding_rate=entry_snapshot.funding_rate,
                open_interest_at_entry=entry_snapshot.open_interest,
                open_interest_delta=entry_snapshot.open_interest,
                process_quality_score=100,
                root_causes=[],
                evidence_refs=[],
                strategy_params_at_entry=strategy_params,
                signal_factors=signal_factors or [],
                conviction_factors=conviction_factors,
                filter_decisions=filter_decisions or [],
                sizing_inputs=sizing_inputs,
                expected_entry_price=expected_entry_price,
                entry_slippage_bps=round(entry_slippage_bps, 2) if entry_slippage_bps else None,
                entry_latency_ms=entry_latency_ms,
                portfolio_state_at_entry=portfolio_state,
                session_type=session_type,
                contract_month=contract_month,
                margin_used_pct=margin_used_pct,
                concurrent_positions_at_entry=concurrent_positions,
                drawdown_pct=drawdown_pct,
                drawdown_tier=drawdown_tier,
                drawdown_size_mult=drawdown_size_mult,
                signal_evolution=signal_evolution,
                execution_timestamps=execution_timestamps,
                experiment_id=experiment_id if experiment_id is not None else self.experiment_id,
                experiment_variant=experiment_variant if experiment_variant is not None else self.experiment_variant,
                strategy_id=self.strategy_id,
                strategy_type=self.strategy_type,
                stage="entry",
            )

            # Assemble entry fill details for FillQualityAnalyzer
            trade.entry_fill_details = {
                "order_id": (
                    runtime_refs.get("entry_order_id")
                    or runtime_refs.get("fill_order_id")
                    or runtime_refs.get("order_id")
                    or runtime_refs.get("oms_order_id")
                ),
                "fill_id": runtime_refs.get("entry_fill_id") or runtime_refs.get("fill_id") or runtime_refs.get("exec_id"),
                "fill_price": entry_price,
                "fill_qty": runtime_refs.get("fill_qty") or position_size,
                "slippage_bps": round(entry_slippage_bps, 2) if entry_slippage_bps is not None else None,
                "fill_latency_ms": entry_latency_ms,
                "fill_type": "limit",
            }
            for key, value in runtime_refs.items():
                if hasattr(trade, key):
                    setattr(trade, key, value)

            # Compute param_set_id hash for efficient grouping
            if strategy_params:
                params_str = json.dumps(strategy_params, sort_keys=True, default=str)
                trade.param_set_id = hashlib.sha256(params_str.encode()).hexdigest()[:16]

            # Populate sector metadata from static map
            try:
                sector_info = self._sector_map.get(pair, {})
                if sector_info:
                    trade.sector = sector_info.get("sector", "")
                    trade.industry = sector_info.get("industry", "")
            except Exception as e:
                logger.warning("Failed to look up sector for %s: %s", pair, e)

            # Populate correlated_pairs_detail via DB query (separate OMSs)
            try:
                if self._pg_store and self._family_strategy_ids:
                    corr = run_async_safely(
                        capture_concurrent_positions(
                            self._pg_store, "stock",
                            self.strategy_id,
                            pair, self._family_strategy_ids,
                        )
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
            return TradeEvent(trade_id=trade_id, event_metadata={}, entry_snapshot={})

    def log_exit(
        self,
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
        session_transitions: Optional[List[dict]] = None,
        **runtime_refs,
    ) -> Optional[TradeEvent]:
        """Call this immediately after a trade exit is confirmed."""
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
                "order_id": (
                    runtime_refs.get("exit_order_id")
                    or runtime_refs.get("fill_order_id")
                    or runtime_refs.get("order_id")
                    or runtime_refs.get("oms_order_id")
                ),
                "fill_id": runtime_refs.get("exit_fill_id") or runtime_refs.get("fill_id") or runtime_refs.get("exec_id"),
                "fill_price": exit_price,
                "fill_qty": runtime_refs.get("fill_qty") or trade.position_size,
                "slippage_bps": round(exit_slippage_bps, 2) if exit_slippage_bps is not None else None,
                "fill_latency_ms": exit_latency_ms,
                "fill_type": "stop" if exit_reason in ("STOP_LOSS", "STOP") else "market",
            }

            # MFE/MAE fields (Gap G1, B5)
            trade.mfe_r = round(mfe_r, 4) if mfe_r is not None else None
            trade.mae_r = round(mae_r, 4) if mae_r is not None else None
            trade.mfe_price = mfe_price
            trade.mae_price = mae_price
            if mfe_price is not None and trade.entry_price:
                if trade.side == "LONG":
                    trade.mfe_pct = round(
                        ((mfe_price - trade.entry_price) / trade.entry_price) * 100,
                        4,
                    )
                else:
                    trade.mfe_pct = round(
                        ((trade.entry_price - mfe_price) / trade.entry_price) * 100,
                        4,
                    )
            if mae_price is not None and trade.entry_price:
                if trade.side == "LONG":
                    trade.mae_pct = round(
                        ((mae_price - trade.entry_price) / trade.entry_price) * 100,
                        4,
                    )
                else:
                    trade.mae_pct = round(
                        ((trade.entry_price - mae_price) / trade.entry_price) * 100,
                        4,
                    )

            # Compute exit efficiency (B9): actual_pnl_r / mfe_r
            if mfe_r and mfe_r > 0 and trade.entry_price and trade.entry_price > 0:
                stop0 = (trade.strategy_params_at_entry or {}).get("stop0", trade.entry_price)
                risk_per_unit = abs(trade.entry_price - stop0)
                if risk_per_unit > 0:
                    if trade.side == "LONG":
                        actual_r = (exit_price - trade.entry_price) / risk_per_unit
                    else:
                        actual_r = (trade.entry_price - exit_price) / risk_per_unit
                    trade.exit_efficiency = round(actual_r / mfe_r, 4)

            if session_transitions:
                trade.session_transitions = session_transitions
            for key, value in runtime_refs.items():
                if hasattr(trade, key):
                    setattr(trade, key, value)

            trade.stage = "exit"

            # Queue post-exit price backfill
            self._pending_exit_backfills.append({
                "trade_id": trade_id,
                "pair": trade.pair,
                "side": trade.side,
                "exit_price": exit_price,
                "exit_time": exch_ts,
                "file_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            })

            # Score process quality if scorer is available
            if self.process_scorer:
                try:
                    score = self.process_scorer.score_trade(
                        trade.to_dict(), strategy_type=self.strategy_type
                    )
                    trade.process_quality_score = score.process_quality_score
                    trade.root_causes = list(score.root_causes)
                    trade.evidence_refs = list(score.evidence_refs)
                    self._write_score(score)
                except Exception as e:
                    logger.warning("Process scoring failed for %s: %s", trade_id, e)

            trade.event_metadata = create_event_metadata(
                bot_id=self.bot_id,
                event_type="trade",
                payload_key=f"{trade_id}_exit",
                exchange_timestamp=exch_ts,
                data_source_id=self.data_source_id,
                lineage=self._lineage,
            ).to_dict()

            self._write_event(trade)

            return trade

        except Exception as e:
            self._write_error("log_exit", trade_id, e)
            return None

    def get_open_trades(self) -> Dict[str, TradeEvent]:
        return dict(self._open_trades)

    def _write_event(self, trade: TradeEvent):
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

    def _write_score(self, score):
        try:
            score_dir = Path(self.data_dir).parent / "scores"
            score_dir.mkdir(parents=True, exist_ok=True)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filepath = score_dir / f"scores_{today}.jsonl"
            payload = enrich_payload(
                score.to_dict(),
                lineage=self._lineage,
                event_type="process_quality",
                scope="strategy",
            )
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        except Exception as e:
            logger.warning("Failed to write score: %s", e)

    def _write_error(self, method: str, trade_id: str, error: Exception):
        if self._error_logger is not None:
            self._error_logger.log_error(
                error_type=f"trade_logger_{method}",
                message=str(error),
                severity="medium",
                category="warning",
                context={
                    "component": "trade_logger",
                    "method": method,
                    "trade_id": trade_id,
                },
                exc=error,
                source_file=__file__,
            )
            return
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

    def run_post_exit_backfill(self, data_provider) -> None:
        """Backfill 1h/4h post-exit prices. Call periodically."""
        now = datetime.now(timezone.utc)
        completed = []

        for item in list(self._pending_exit_backfills):
            elapsed = now - item["exit_time"]
            if elapsed < timedelta(hours=4):
                continue

            try:
                candles = data_provider.get_ohlcv(
                    item["pair"], timeframe="5m",
                    since=int(item["exit_time"].timestamp() * 1000),
                    limit=60,
                )
                if not candles or len(candles) < 12:
                    continue

                price_1h = None
                price_4h = None
                for candle in candles:
                    candle_time = datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc)
                    candle_elapsed = candle_time - item["exit_time"]
                    if candle_elapsed >= timedelta(hours=1) and price_1h is None:
                        price_1h = candle[4]
                    if candle_elapsed >= timedelta(hours=4) and price_4h is None:
                        price_4h = candle[4]

                exit_price = item["exit_price"]
                side = item["side"]

                def move_pct(post_price):
                    if post_price is None or exit_price == 0:
                        return None
                    if side == "LONG":
                        return round((post_price - exit_price) / exit_price * 100, 4)
                    else:
                        return round((exit_price - post_price) / exit_price * 100, 4)

                outcomes = {
                    "post_exit_1h_price": price_1h,
                    "post_exit_4h_price": price_4h,
                    "post_exit_1h_move_pct": move_pct(price_1h),
                    "post_exit_4h_move_pct": move_pct(price_4h),
                    "post_exit_backfill_status": "complete",
                }

                self._update_trade_event(item["trade_id"], item["file_date"], outcomes)
                self._emit_post_exit_event(item, outcomes)
                completed.append(item)

            except Exception as e:
                logger.warning("Post-exit backfill failed for %s: %s", item["trade_id"], e)

        for c in completed:
            if c in self._pending_exit_backfills:
                self._pending_exit_backfills.remove(c)

    def _emit_post_exit_event(self, item: dict, outcomes: dict) -> None:
        """Write a separate post_exit event so the sidecar forwards backfill data to the relay."""
        try:
            post_exit_dir = self.data_dir.parent / "post_exit"
            post_exit_dir.mkdir(parents=True, exist_ok=True)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filepath = post_exit_dir / f"post_exit_{today}.jsonl"

            ts = datetime.now(timezone.utc).isoformat()
            raw = f"{self.bot_id}|{ts}|post_exit|{item['trade_id']}"
            event_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

            event = {
                "event_id": event_id,
                "trade_id": item["trade_id"],
                "pair": item.get("pair", ""),
                "side": item.get("side", ""),
                "exit_price": item.get("exit_price"),
                "exit_time": item.get("exit_time").isoformat() if item.get("exit_time") else None,
                "timestamp": ts,
                **outcomes,
            }
            event = enrich_payload(
                event,
                lineage=self._lineage,
                event_type="post_exit",
                scope="strategy",
            )
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.debug("Failed to emit post_exit event for %s: %s", item.get("trade_id"), e)

    def _update_trade_event(self, trade_id: str, file_date: str, updates: dict) -> None:
        """Update a completed trade event in the JSONL file."""
        filepath = self.data_dir / f"trades_{file_date}.jsonl"
        if not filepath.exists():
            return
        try:
            lines = filepath.read_text(encoding="utf-8").strip().split("\n")
            new_lines = []
            for line in lines:
                try:
                    event = json.loads(line)
                    if event.get("trade_id") == trade_id and event.get("stage") == "exit":
                        event.update(updates)
                    new_lines.append(json.dumps(event, default=str))
                except json.JSONDecodeError:
                    new_lines.append(line)
            filepath.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to update trade %s: %s", trade_id, e)
