"""HourlyAnalyzer — computes time-of-day performance buckets.

Groups trades by entry hour (UTC) and computes PnL, win rate, and
process quality per hour bucket.
"""
from __future__ import annotations

from collections import defaultdict

from trading_assistant.schemas.events import TradeEvent
from trading_assistant.schemas.hourly_performance import HourlyBucket, HourlyPerformance


class HourlyAnalyzer:
    def __init__(self, bot_id: str, date: str, bot_timezone: str = "UTC") -> None:
        self._bot_id = bot_id
        self._date = date
        self._bot_timezone = bot_timezone

    def compute(self, trades: list[TradeEvent]) -> HourlyPerformance:
        from trading_assistant.orchestrator.tz_utils import to_local_hour

        by_hour: dict[int, list[TradeEvent]] = defaultdict(list)
        for t in trades:
            hour = to_local_hour(t.entry_time, self._bot_timezone)
            by_hour[hour].append(t)

        buckets: list[HourlyBucket] = []
        for hour in sorted(by_hour):
            hour_trades = by_hour[hour]
            wins = sum(1 for t in hour_trades if t.pnl > 0)
            count = len(hour_trades)
            buckets.append(HourlyBucket(
                hour=hour,
                trade_count=count,
                pnl=sum(t.pnl for t in hour_trades),
                win_rate=wins / count if count > 0 else 0.0,
                avg_process_quality=(
                    sum(t.process_quality_score for t in hour_trades) / count
                    if count > 0 else 0.0
                ),
            ))

        return HourlyPerformance(
            bot_id=self._bot_id,
            date=self._date,
            buckets=buckets,
            timezone=self._bot_timezone,
        )
