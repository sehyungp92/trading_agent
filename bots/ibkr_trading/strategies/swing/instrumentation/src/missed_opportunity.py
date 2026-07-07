"""Missed Opportunity Logger — logs blocked signals and backfills outcomes.

Every signal that fires but is blocked by a filter or risk limit is
recorded here, along with hypothetical outcome simulation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict

from .event_metadata import create_event_metadata
from .market_snapshot import MarketSnapshot, MarketSnapshotService
from libs.instrumentation.event_contract import enrich_payload, write_error_event
from libs.instrumentation.lineage import lineage_from_config

logger = logging.getLogger("instrumentation.missed_opportunity")


@dataclass
class SimulationPolicy:
    """Defines assumptions for hypothetical outcome calculation."""
    entry_fill_model: str = "mid"
    slippage_model: str = "fixed_bps"
    slippage_bps: float = 5.0
    fees_included: bool = True
    fee_bps: float = 7.0
    tp_sl_logic: str = "atr_based"
    tp_value: float = 2.0
    sl_value: float = 1.0
    max_hold_bars: int = 100

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MissedOpportunityEvent:
    """A signal that fired but was not executed."""
    event_metadata: dict
    market_snapshot: dict

    bot_id: str = ""
    pair: str = ""
    side: str = ""
    signal: str = ""
    signal_id: str = ""
    signal_strength: float = 0.0
    signal_time: str = ""
    blocked_by: str = ""
    block_reason: str = ""
    strategy_id: str = ""

    hypothetical_entry_price: float = 0.0

    outcome_1h: Optional[float] = None
    outcome_4h: Optional[float] = None
    outcome_24h: Optional[float] = None
    outcome_pnl_1h: Optional[float] = None
    outcome_pnl_4h: Optional[float] = None
    outcome_pnl_24h: Optional[float] = None
    would_have_hit_tp: Optional[bool] = None
    would_have_hit_sl: Optional[bool] = None
    bars_to_tp: Optional[int] = None
    bars_to_sl: Optional[int] = None
    first_hit: Optional[str] = None

    simulation_policy: Optional[dict] = None
    simulation_confidence: float = 0.0
    assumption_tags: List[str] = field(default_factory=list)
    backfill_status: str = "pending"

    strategy_params_at_signal: Optional[dict] = None
    market_regime: str = ""

    # Structured filter context
    filter_decisions: List[dict] = field(default_factory=list)

    # How close the blocking filter was to passing (percentage margin)
    margin_pct: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Add trading_assistant-compatible alias fields.
        # TA's MissedOpportunityEvent expects these exact names.
        d["hypothetical_entry"] = d.get("hypothetical_entry_price", 0.0)
        d["confidence"] = d.get("simulation_confidence", 0.0)
        return d


class MissedOpportunityLogger:
    """Logs missed opportunities and manages outcome backfill."""

    def __init__(self, config: dict, snapshot_service: MarketSnapshotService):
        self.bot_id = config["bot_id"]
        self.strategy_id = config.get("strategy_id", "")
        self.data_dir = Path(config["data_dir"]) / "missed"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_service = snapshot_service
        self.data_source_id = config.get("data_source_id", "ibkr_historical")
        self._lineage = lineage_from_config(
            config,
            family_id="swing",
            strategy_id=self.strategy_id,
        )

        self.simulation_policies = self._load_simulation_policies(config)
        self._pending_backfills: List[Dict] = []
        self._backfill_lock = threading.Lock()

    def _load_simulation_policies(self, config: dict) -> Dict[str, SimulationPolicy]:
        policies: Dict[str, SimulationPolicy] = {}
        policy_file = Path("instrumentation/config/simulation_policies.yaml")
        if policy_file.exists():
            try:
                import yaml
                with open(policy_file) as f:
                    raw = yaml.safe_load(f)
                for name, params in raw.get("simulation_policies", {}).items():
                    policies[name] = SimulationPolicy(**params)
            except Exception as e:
                logger.warning("Failed to load simulation policies: %s", e)
        if not policies:
            policies["default"] = SimulationPolicy()
        return policies

    def _get_policy(self, strategy_type: str = None) -> SimulationPolicy:
        if strategy_type and strategy_type in self.simulation_policies:
            return self.simulation_policies[strategy_type]
        return self.simulation_policies.get("default", SimulationPolicy())

    def _compute_hypothetical_entry(
        self, snapshot: MarketSnapshot, side: str, policy: SimulationPolicy,
    ) -> float:
        if policy.entry_fill_model == "mid":
            base_price = snapshot.mid if snapshot.mid else snapshot.last_trade_price
        elif policy.entry_fill_model == "bid_ask":
            base_price = snapshot.ask if side == "LONG" else snapshot.bid
            if not base_price:
                base_price = snapshot.last_trade_price
        elif policy.entry_fill_model == "next_trade":
            base_price = snapshot.last_trade_price
        else:
            base_price = snapshot.mid if snapshot.mid else snapshot.last_trade_price

        if policy.slippage_model == "fixed_bps":
            slippage = base_price * policy.slippage_bps / 10000
        elif policy.slippage_model == "spread_proportional":
            slippage = (snapshot.ask - snapshot.bid) * 0.5 if snapshot.ask and snapshot.bid else 0
        else:
            slippage = base_price * policy.slippage_bps / 10000

        if side == "LONG":
            return base_price + slippage
        else:
            return base_price - slippage

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
        strategy_type: Optional[str] = None,
        strategy_id: str = "",
        market_regime: str = "",
        exchange_timestamp: Optional[datetime] = None,
        bar_id: Optional[str] = None,
        filter_decisions: Optional[List[dict]] = None,
    ) -> MissedOpportunityEvent:
        """Call when a signal fires but is blocked."""
        try:
            now = datetime.now(timezone.utc)
            exch_ts = exchange_timestamp or now

            snapshot = self.snapshot_service.capture_now(pair)
            policy = self._get_policy(strategy_type)
            hyp_entry = self._compute_hypothetical_entry(snapshot, side, policy)

            assumption_tags = [
                f"{policy.entry_fill_model}_fill",
                f"{policy.slippage_bps}bps_slippage" if policy.slippage_model == "fixed_bps"
                    else f"{policy.slippage_model}_slippage",
            ]
            if policy.fees_included:
                assumption_tags.append(f"{policy.fee_bps}bps_fees")
            else:
                assumption_tags.append("no_fees")
            assumption_tags.append(f"{policy.tp_sl_logic}_tp_sl")

            signal_hash = hashlib.sha256(
                f"{pair}|{side}|{signal_id}|{exch_ts.isoformat()}".encode()
            ).hexdigest()[:12]

            metadata = create_event_metadata(
                bot_id=self.bot_id,
                event_type="missed_opportunity",
                payload_key=signal_hash,
                exchange_timestamp=exch_ts,
                data_source_id=self.data_source_id,
                bar_id=bar_id,
                lineage=self._lineage,
            )

            event = MissedOpportunityEvent(
                event_metadata=metadata.to_dict(),
                market_snapshot=snapshot.to_dict(),
                bot_id=self.bot_id,
                pair=pair,
                side=side,
                signal=signal,
                signal_id=signal_id,
                signal_strength=signal_strength,
                signal_time=exch_ts.isoformat(),
                blocked_by=blocked_by,
                block_reason=block_reason,
                strategy_id=strategy_id,
                hypothetical_entry_price=hyp_entry,
                simulation_policy=policy.to_dict(),
                assumption_tags=assumption_tags,
                strategy_params_at_signal=strategy_params,
                market_regime=market_regime,
                backfill_status="pending",
            )

            if filter_decisions:
                event.filter_decisions = filter_decisions
                # Compute margin_pct from the blocking filter's threshold vs actual
                for fd in filter_decisions:
                    if fd.get("filter_name") == blocked_by and not fd.get("passed", True):
                        threshold = fd.get("threshold")
                        actual = fd.get("actual_value")
                        if threshold and actual and threshold != 0:
                            event.margin_pct = round(
                                ((actual - threshold) / abs(threshold)) * 100, 2
                            )
                        break

            self._write_event(event)

            with self._backfill_lock:
                self._pending_backfills.append({
                    "event_id": metadata.event_id,
                    "pair": pair,
                    "side": side,
                    "entry_price": hyp_entry,
                    "signal_time": exch_ts,
                    "policy": policy,
                    "snapshot": snapshot,
                    "file_date": now.strftime("%Y-%m-%d"),
                })

            return event

        except Exception as e:
            self._write_error("log_missed", f"{pair}_{signal_id}", e)
            return MissedOpportunityEvent(event_metadata={}, market_snapshot={})

    def run_backfill(self, data_provider) -> None:
        """Process pending backfills.  Call periodically (e.g. every 5 min)."""
        now = datetime.now(timezone.utc)
        completed = []

        with self._backfill_lock:
            pending = list(self._pending_backfills)

        for item in pending:
            elapsed = now - item["signal_time"]

            if elapsed < timedelta(hours=24):
                outcomes = self._compute_outcomes(item, data_provider, partial=True, elapsed=elapsed)
                if outcomes:
                    self._update_event(item["event_id"], item["file_date"], outcomes, status="partial")
                continue

            outcomes = self._compute_outcomes(item, data_provider, partial=False, elapsed=elapsed)
            if outcomes:
                self._update_event(item["event_id"], item["file_date"], outcomes, status="complete")
                completed.append(item)

        with self._backfill_lock:
            for c in completed:
                if c in self._pending_backfills:
                    self._pending_backfills.remove(c)

    def _compute_outcomes(
        self, item: dict, data_provider, partial: bool, elapsed: timedelta,
    ) -> Optional[dict]:
        try:
            pair = item["pair"]
            side = item["side"]
            entry_price = item["entry_price"]
            signal_time = item["signal_time"]
            policy = item["policy"]
            snapshot = item["snapshot"]

            candles = None
            if hasattr(data_provider, "get_ohlcv"):
                candles = data_provider.get_ohlcv(
                    pair, timeframe="5m",
                    since=int(signal_time.timestamp() * 1000),
                    limit=300,
                )

            if not candles or len(candles) < 2:
                return None

            # Compute TP/SL prices
            if policy.tp_sl_logic == "atr_based":
                atr = snapshot.atr_14 or (entry_price * 0.01)
                if side == "LONG":
                    tp_price = entry_price + (atr * policy.tp_value)
                    sl_price = entry_price - (atr * policy.sl_value)
                else:
                    tp_price = entry_price - (atr * policy.tp_value)
                    sl_price = entry_price + (atr * policy.sl_value)
            elif policy.tp_sl_logic == "fixed_pct":
                if side == "LONG":
                    tp_price = entry_price * (1 + policy.tp_value / 100)
                    sl_price = entry_price * (1 - policy.sl_value / 100)
                else:
                    tp_price = entry_price * (1 - policy.tp_value / 100)
                    sl_price = entry_price * (1 + policy.sl_value / 100)
            else:
                atr = snapshot.atr_14 or (entry_price * 0.01)
                if side == "LONG":
                    tp_price = entry_price + (atr * 2)
                    sl_price = entry_price - atr
                else:
                    tp_price = entry_price - (atr * 2)
                    sl_price = entry_price + atr

            would_have_hit_tp = False
            would_have_hit_sl = False
            bars_to_tp = None
            bars_to_sl = None
            first_hit = "TIMEOUT"
            price_1h = None
            price_4h = None
            price_24h = None

            for i, candle in enumerate(candles):
                ts_val = candle[0] if isinstance(candle, (list, tuple)) else getattr(candle, "date", 0)
                if isinstance(ts_val, (int, float)):
                    candle_time = datetime.fromtimestamp(ts_val / 1000, tz=timezone.utc)
                else:
                    candle_time = ts_val if hasattr(ts_val, "tzinfo") else signal_time
                candle_elapsed = candle_time - signal_time

                if isinstance(candle, (list, tuple)):
                    high, low, close = candle[2], candle[3], candle[4]
                else:
                    high = getattr(candle, "high", 0)
                    low = getattr(candle, "low", 0)
                    close = getattr(candle, "close", 0)

                if candle_elapsed >= timedelta(hours=1) and price_1h is None:
                    price_1h = close
                if candle_elapsed >= timedelta(hours=4) and price_4h is None:
                    price_4h = close
                if candle_elapsed >= timedelta(hours=24) and price_24h is None:
                    price_24h = close

                if not would_have_hit_tp and not would_have_hit_sl:
                    if side == "LONG":
                        if high >= tp_price:
                            would_have_hit_tp = True
                            bars_to_tp = i
                            if first_hit == "TIMEOUT":
                                first_hit = "TP"
                        if low <= sl_price:
                            would_have_hit_sl = True
                            bars_to_sl = i
                            if first_hit == "TIMEOUT" or (first_hit == "TP" and bars_to_sl <= (bars_to_tp or i)):
                                first_hit = "SL"
                    else:
                        if low <= tp_price:
                            would_have_hit_tp = True
                            bars_to_tp = i
                            if first_hit == "TIMEOUT":
                                first_hit = "TP"
                        if high >= sl_price:
                            would_have_hit_sl = True
                            bars_to_sl = i
                            if first_hit == "TIMEOUT" or (first_hit == "TP" and bars_to_sl <= (bars_to_tp or i)):
                                first_hit = "SL"

            if bars_to_tp is not None and bars_to_sl is not None:
                if bars_to_tp < bars_to_sl:
                    first_hit = "TP"
                elif bars_to_sl < bars_to_tp:
                    first_hit = "SL"
                else:
                    first_hit = "SL"  # conservative

            fee_factor = policy.fee_bps / 10000 if policy.fees_included else 0

            def compute_pnl(exit_price):
                if exit_price is None:
                    return None
                if side == "LONG":
                    gross = (exit_price - entry_price) / entry_price
                else:
                    gross = (entry_price - exit_price) / entry_price
                return round((gross - 2 * fee_factor) * 100, 4)

            confidence = 0.3
            if price_1h is not None:
                confidence += 0.2
            if price_4h is not None:
                confidence += 0.2
            if price_24h is not None:
                confidence += 0.2
            if would_have_hit_tp or would_have_hit_sl:
                confidence += 0.1

            return {
                "outcome_1h": price_1h,
                "outcome_4h": price_4h,
                "outcome_24h": price_24h,
                "outcome_pnl_1h": compute_pnl(price_1h),
                "outcome_pnl_4h": compute_pnl(price_4h),
                "outcome_pnl_24h": compute_pnl(price_24h),
                "would_have_hit_tp": would_have_hit_tp,
                "would_have_hit_sl": would_have_hit_sl,
                "bars_to_tp": bars_to_tp,
                "bars_to_sl": bars_to_sl,
                "first_hit": first_hit,
                "simulation_confidence": round(confidence, 2),
            }

        except Exception as e:
            self._write_error("compute_outcomes", item.get("event_id", "unknown"), e)
            return None

    def _update_event(self, event_id: str, file_date: str, outcomes: dict, status: str) -> None:
        filepath = self.data_dir / f"missed_{file_date}.jsonl"
        if not filepath.exists():
            return
        try:
            lines = filepath.read_text().strip().split("\n")
            updated = False
            new_lines = []
            for line in lines:
                try:
                    event = json.loads(line)
                    if event.get("event_metadata", {}).get("event_id") == event_id:
                        event.update(outcomes)
                        event["backfill_status"] = status
                        event = enrich_payload(
                            event,
                            lineage=self._lineage,
                            event_type="missed_opportunity",
                            scope="strategy",
                        )
                        updated = True
                    new_lines.append(json.dumps(event, default=str))
                except json.JSONDecodeError:
                    new_lines.append(line)
            if updated:
                filepath.write_text("\n".join(new_lines) + "\n")
        except Exception as e:
            logger.warning("Failed to update missed event %s: %s", event_id, e)

    def _write_event(self, event: MissedOpportunityEvent) -> None:
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filepath = self.data_dir / f"missed_{today}.jsonl"
            payload = enrich_payload(
                event.to_dict(),
                lineage=self._lineage,
                event_type="missed_opportunity",
                scope="strategy",
            )
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        except Exception as e:
            logger.warning("Failed to write missed event: %s", e)

    def _write_error(self, method: str, context: str, error: Exception) -> None:
        try:
            write_error_event(
                Path(self.data_dir).parent,
                self._lineage,
                component="missed_opportunity",
                method=method,
                message=str(error),
                error_type=type(error).__name__,
                context={"context": context},
                exc=error,
            )
        except Exception:
            pass
