"""Indicator snapshot logger — emits IndicatorSnapshot JSONL events."""
from __future__ import annotations

import json
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from libs.instrumentation.event_contract import enrich_payload
from libs.instrumentation.lineage import LineageContext

logger = logging.getLogger("instrumentation.indicator_logger")


@dataclass
class IndicatorSnapshot:
    """Point-in-time capture of indicator values at signal evaluation."""

    bot_id: str
    pair: str
    timestamp: str
    indicators: dict[str, float]
    signal_name: str
    signal_strength: float
    decision: str  # "enter", "skip", "exit"
    strategy_type: str  # "helix", "nqdtc", "vdubus"
    event_id: str = ""
    bar_id: Optional[str] = None
    context: dict = field(default_factory=dict)
    # momentum_trader-specific context keys:
    #   "session": "RTH" | "ETH"
    #   "contract_month": "2026-06"
    #   "signal_class": "M" | "F" | "T"  (Helix signal class)
    #   "concurrent_positions": 2
    #   "drawdown_tier": "NORMAL" | "CAUTION" | "DEFENSIVE"

    def to_dict(self) -> dict:
        return asdict(self)


class IndicatorLogger:
    """Writes IndicatorSnapshot events to daily JSONL files."""

    def __init__(self, data_dir: str | Path, bot_id: str, lineage: LineageContext | dict | None = None) -> None:
        self._data_dir = Path(data_dir) / "indicators"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._bot_id = bot_id
        self._lineage = lineage or {"bot_id": bot_id}

    def log_snapshot(
        self,
        pair: str,
        indicators: dict[str, float],
        signal_name: str,
        signal_strength: float,
        decision: str,
        strategy_type: str,
        exchange_timestamp: Optional[datetime] = None,
        bar_id: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> IndicatorSnapshot:
        """Record an indicator snapshot at a signal evaluation point."""
        ts = exchange_timestamp or datetime.now(timezone.utc)
        ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)

        raw = f"{self._bot_id}|{ts_str}|indicator_snapshot|{pair}:{signal_name}"
        event_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

        snapshot = IndicatorSnapshot(
            bot_id=self._bot_id,
            pair=pair,
            timestamp=ts_str,
            indicators=indicators,
            signal_name=signal_name,
            signal_strength=signal_strength,
            decision=decision,
            strategy_type=strategy_type,
            event_id=event_id,
            bar_id=bar_id,
            context=context or {},
        )

        self._write(snapshot)
        return snapshot

    def _write(self, snapshot: IndicatorSnapshot) -> None:
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filepath = self._data_dir / f"indicators_{today}.jsonl"
            payload = enrich_payload(
                snapshot.to_dict(),
                lineage=self._lineage,
                event_type="indicator_snapshot",
                scope="strategy",
            )
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        except Exception as e:
            logger.debug("Failed to write indicator snapshot: %s", e)
