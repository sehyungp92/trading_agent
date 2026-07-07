"""Weekly trade and coordinator event loading support."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class WeeklyEventLoadingSupport:
    """Weekly trade and coordinator event loading support."""

    def _load_trades_for_week(
        self, bot_id: str, week_start: str, week_end: str,
    ) -> tuple:
        """Load trade and missed opportunity events for a bot within a date range."""
        from trading_assistant.schemas.events import TradeEvent, MissedOpportunityEvent

        trades: list[TradeEvent] = []
        missed: list[MissedOpportunityEvent] = []
        start = datetime.strptime(week_start, "%Y-%m-%d")
        end = datetime.strptime(week_end, "%Y-%m-%d")

        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            bot_dir = self._curated_dir / date_str / bot_id

            if bot_dir.is_dir():
                trades_file = bot_dir / "trades.jsonl"
                if trades_file.exists():
                    total = 0
                    dropped = 0
                    for line in trades_file.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        total += 1
                        try:
                            trades.append(TradeEvent(**json.loads(line)))
                        except Exception:
                            dropped += 1
                    if dropped:
                        logger.warning(
                            "Dropped %d/%d malformed trade records from %s",
                            dropped, total, trades_file,
                        )

                missed_file = bot_dir / "missed.jsonl"
                if missed_file.exists():
                    total = 0
                    dropped = 0
                    for line in missed_file.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        total += 1
                        try:
                            missed.append(MissedOpportunityEvent(**json.loads(line)))
                        except Exception:
                            dropped += 1
                    if dropped:
                        logger.warning(
                            "Dropped %d/%d malformed missed-opportunity records from %s",
                            dropped, total, missed_file,
                        )

            current += timedelta(days=1)

        return (trades, missed)

    def _load_coordinator_events(
        self, week_start: str, week_end: str,
    ) -> list:
        """Load coordinator action events for swing_multi_01 within a date range."""
        from trading_assistant.schemas.interaction_analysis import CoordinatorAction

        events: list[CoordinatorAction] = []
        start = datetime.strptime(week_start, "%Y-%m-%d")
        end = datetime.strptime(week_end, "%Y-%m-%d")

        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            coord_file = self._curated_dir / date_str / "swing_multi_01" / "coordinator_impact.json"
            if not coord_file.exists():
                current += timedelta(days=1)
                continue

            try:
                data = json.loads(coord_file.read_text())
                for evt in data.get("events", []):
                    try:
                        events.append(CoordinatorAction(**evt))
                    except Exception:
                        logger.warning("Skipping malformed coordinator event in %s", coord_file)
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not read coordinator file %s", coord_file)

            current += timedelta(days=1)

        return events
