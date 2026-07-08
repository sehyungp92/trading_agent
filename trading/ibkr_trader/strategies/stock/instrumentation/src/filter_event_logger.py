"""Filter event logger — emits standalone FilterDecisionEvent JSONL."""
from __future__ import annotations

import json
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .filter_decision import FilterDecision
from libs.instrumentation.event_contract import enrich_payload
from libs.instrumentation.lineage import LineageContext

logger = logging.getLogger("instrumentation.filter_event_logger")


class FilterEventLogger:
    """Writes FilterDecision events as standalone JSONL entries."""

    def __init__(self, data_dir: str | Path, bot_id: str, lineage: LineageContext | dict | None = None) -> None:
        self._data_dir = Path(data_dir) / "filter_decisions"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._bot_id = bot_id
        self._lineage = lineage or {"bot_id": bot_id}

    def log_decision(
        self,
        filter_decision: FilterDecision,
        pair: str,
        signal_name: str = "",
        signal_strength: float = 0.0,
        strategy_type: str = "",
        exchange_timestamp: Optional[datetime] = None,
        bar_id: Optional[str] = None,
    ) -> None:
        """Write a single FilterDecision as a standalone event."""
        ts = exchange_timestamp or datetime.now(timezone.utc)
        ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)

        raw = f"{self._bot_id}|{ts_str}|filter_decision|{pair}:{filter_decision.filter_name}"
        event_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

        record = {
            "bot_id": self._bot_id,
            "pair": pair,
            "timestamp": ts_str,
            "event_id": event_id,
            **filter_decision.to_dict(),  # filter_name, threshold, actual_value, passed, margin_pct
            "signal_name": signal_name,
            "signal_strength": signal_strength,
            "strategy_type": strategy_type,
            "bar_id": bar_id,
        }
        record = enrich_payload(
            record,
            lineage=self._lineage,
            event_type="filter_decision",
            scope="strategy",
        )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = self._data_dir / f"filter_decisions_{today}.jsonl"
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.debug("Failed to write filter event: %s", e)

    def log_decisions(
        self,
        filter_decisions: list[FilterDecision],
        pair: str,
        signal_name: str = "",
        signal_strength: float = 0.0,
        strategy_type: str = "",
        exchange_timestamp: Optional[datetime] = None,
        bar_id: Optional[str] = None,
    ) -> None:
        """Write multiple filter decisions from a single signal evaluation."""
        for fd in filter_decisions:
            self.log_decision(
                fd, pair=pair, signal_name=signal_name,
                signal_strength=signal_strength, strategy_type=strategy_type,
                exchange_timestamp=exchange_timestamp, bar_id=bar_id,
            )
