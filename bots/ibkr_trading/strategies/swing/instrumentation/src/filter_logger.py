"""Filter decision event logging — captures each filter gate evaluation.

Writes FilterDecisionEvent records to JSONL so we can analyze which filters
block the most trades and how close near-misses come to the threshold.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .event_metadata import create_event_metadata
from libs.instrumentation.event_contract import enrich_payload
from libs.instrumentation.lineage import lineage_from_config

logger = logging.getLogger("instrumentation.filter_logger")


@dataclass
class FilterDecisionEvent:
    """A single filter gate evaluation."""

    bot_id: str
    pair: str
    timestamp: str
    filter_name: str
    passed: bool
    threshold: float
    actual_value: float
    signal_name: str = ""
    signal_strength: float = 0.0
    strategy_type: str = ""
    coordinator_triggered: bool = False  # True if coordinator rule, not strategy filter
    event_id: str = ""
    bar_id: Optional[str] = None
    event_metadata: dict = field(default_factory=dict)

    @property
    def margin_pct(self) -> Optional[float]:
        if self.threshold == 0.0:
            return None
        return round((self.actual_value - self.threshold) / abs(self.threshold) * 100, 2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["margin_pct"] = self.margin_pct
        return d


class FilterLogger:
    """Writes FilterDecisionEvent records to daily JSONL files."""

    def __init__(self, config: dict):
        self.bot_id = config["bot_id"]
        self.data_dir = Path(config["data_dir"]) / "filter_decisions"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_source_id = config.get("data_source_id", "ibkr_execution")
        self._lineage = lineage_from_config(
            config,
            family_id="swing",
            strategy_id=config.get("strategy_id", ""),
        )

    def log_decision(
        self,
        pair: str,
        filter_name: str,
        passed: bool,
        threshold: float,
        actual_value: float,
        signal_name: str = "",
        signal_strength: float = 0.0,
        strategy_type: str = "",
        coordinator_triggered: bool = False,
        exchange_timestamp: Optional[datetime] = None,
        bar_id: Optional[str] = None,
    ) -> FilterDecisionEvent:
        """Record a filter gate evaluation."""
        now = exchange_timestamp or datetime.now(timezone.utc)
        ts_str = now.isoformat() if isinstance(now, datetime) else str(now)

        meta = create_event_metadata(
            bot_id=self.bot_id,
            event_type="filter_decision",
            payload_key=f"{pair}:{filter_name}:{ts_str}",
            exchange_timestamp=now if isinstance(now, datetime) else datetime.now(timezone.utc),
            data_source_id=self.data_source_id,
            bar_id=bar_id,
            lineage=self._lineage,
        )

        event = FilterDecisionEvent(
            bot_id=self.bot_id,
            pair=pair,
            timestamp=ts_str,
            filter_name=filter_name,
            passed=passed,
            threshold=threshold,
            actual_value=actual_value,
            signal_name=signal_name,
            signal_strength=signal_strength,
            strategy_type=strategy_type,
            coordinator_triggered=coordinator_triggered,
            event_id=meta.event_id,
            bar_id=bar_id,
            event_metadata=meta.to_dict(),
        )

        self._write_event(event)
        return event

    def _write_event(self, event: FilterDecisionEvent) -> None:
        try:
            date_str = event.timestamp[:10] if event.timestamp else (
                datetime.now(timezone.utc).strftime("%Y-%m-%d")
            )
            filepath = self.data_dir / f"filter_decisions_{date_str}.jsonl"
            payload = enrich_payload(
                event.to_dict(),
                lineage=self._lineage,
                event_type="filter_decision",
                scope="strategy",
            )
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        except Exception:
            logger.exception("Failed to write FilterDecisionEvent")
