"""Indicator snapshot logging — captures indicator state at signal evaluation.

Writes IndicatorSnapshot events to JSONL for post-hoc analysis of what
the strategy saw when it decided to enter, skip, or exit.
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

logger = logging.getLogger("instrumentation.indicator_logger")


@dataclass
class IndicatorSnapshot:
    """Point-in-time indicator state at signal evaluation."""

    bot_id: str
    pair: str
    timestamp: str
    indicators: dict[str, float]
    signal_name: str
    signal_strength: float
    decision: str                         # "enter", "skip", "exit"
    strategy_type: str                    # "ATRSS", "AKC_HELIX", etc.
    event_id: str = ""
    bar_id: Optional[str] = None
    context: dict = field(default_factory=dict)
    # swing_trader-specific context keys:
    #   "overlay_state": {"qqq_bullish": True, "gld_bullish": False}
    #   "drawdown_tier": "NORMAL" | "CAUTION" | "DEFENSIVE" | "HALT"
    #   "market_session": "PRE" | "RTH" | "ETH_POST" | "WEEKEND"
    #   "coordinator_active_rules": ["tight_stop_rule", ...]

    event_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class IndicatorLogger:
    """Writes IndicatorSnapshot events to daily JSONL files."""

    def __init__(self, config: dict):
        self.bot_id = config["bot_id"]
        self.data_dir = Path(config["data_dir"]) / "indicators"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_source_id = config.get("data_source_id", "ibkr_execution")
        self._lineage = lineage_from_config(
            config,
            family_id="swing",
            strategy_id=config.get("strategy_id", ""),
        )

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
        """Record an indicator snapshot at signal evaluation time."""
        now = exchange_timestamp or datetime.now(timezone.utc)
        ts_str = now.isoformat() if isinstance(now, datetime) else str(now)

        meta = create_event_metadata(
            bot_id=self.bot_id,
            event_type="indicator_snapshot",
            payload_key=f"{pair}:{signal_name}:{ts_str}",
            exchange_timestamp=now if isinstance(now, datetime) else datetime.now(timezone.utc),
            data_source_id=self.data_source_id,
            bar_id=bar_id,
            lineage=self._lineage,
        )

        snapshot = IndicatorSnapshot(
            bot_id=self.bot_id,
            pair=pair,
            timestamp=ts_str,
            indicators=indicators,
            signal_name=signal_name,
            signal_strength=signal_strength,
            decision=decision,
            strategy_type=strategy_type,
            event_id=meta.event_id,
            bar_id=bar_id,
            context=context or {},
            event_metadata=meta.to_dict(),
        )

        self._write_event(snapshot)
        return snapshot

    def _write_event(self, snapshot: IndicatorSnapshot) -> None:
        try:
            date_str = snapshot.timestamp[:10] if snapshot.timestamp else (
                datetime.now(timezone.utc).strftime("%Y-%m-%d")
            )
            filepath = self.data_dir / f"indicators_{date_str}.jsonl"
            payload = enrich_payload(
                snapshot.to_dict(),
                lineage=self._lineage,
                event_type="indicator_snapshot",
                scope="strategy",
            )
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        except Exception:
            logger.exception("Failed to write IndicatorSnapshot")
