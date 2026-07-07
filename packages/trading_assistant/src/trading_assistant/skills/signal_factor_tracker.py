# skills/signal_factor_tracker.py
"""Signal factor tracker — JSONL persistence + rolling analysis for factor performance.

Records daily factor attribution stats to a persistent JSONL file, enabling
30-day rolling trend analysis. When a factor's win_rate degrades over the
window, it's flagged for review.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from trading_assistant.schemas.signal_factor_history import (
    DailyFactorSnapshot,
    FactorRollingResult,
    SignalFactorRollingReport,
)


class SignalFactorTracker:
    """JSONL-backed signal factor history with rolling analysis."""

    def __init__(self, findings_dir: Path) -> None:
        self._findings_dir = Path(findings_dir)
        self._history_path = self._findings_dir / "signal_factor_history.jsonl"

    def record_daily(self, snapshot: DailyFactorSnapshot) -> None:
        """Append one day's factor snapshot to the history file."""
        self._findings_dir.mkdir(parents=True, exist_ok=True)
        with open(self._history_path, "a") as f:
            f.write(json.dumps(snapshot.model_dump(mode="json"), default=str) + "\n")

    def load_history(
        self,
        bot_id: str,
        days: int = 30,
        as_of: str = "",
    ) -> list[DailyFactorSnapshot]:
        """Load and filter history by bot and date window."""
        if not self._history_path.exists():
            return []

        if as_of:
            end_date = datetime.strptime(as_of, "%Y-%m-%d")
        else:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        records: list[DailyFactorSnapshot] = []
        with open(self._history_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("bot_id") != bot_id:
                    continue
                try:
                    rec_date = datetime.strptime(data["date"], "%Y-%m-%d")
                except (KeyError, ValueError):
                    continue
                if start_date <= rec_date <= end_date:
                    records.append(DailyFactorSnapshot(**data))

        return records

    def compute_rolling(
        self,
        bot_id: str,
        current_date: str,
        win_rate_threshold: float = 0.35,
        min_days: int = 10,
    ) -> SignalFactorRollingReport:
        """Compute rolling 30d analysis for one bot.

        Compares current day to 30d baseline. Computes trend by comparing
        first-half vs second-half average win rates.
        """
        history = self.load_history(bot_id, days=30, as_of=current_date)

        if len(history) < min_days:
            return SignalFactorRollingReport(
                bot_id=bot_id,
                date=current_date,
                window_days=30,
            )

        # Get current day data
        current_day = [h for h in history if h.date == current_date]
        current_factors: dict[str, dict] = {}
        if current_day:
            for f in current_day[0].factors:
                current_factors[f.factor_name] = {
                    "win_rate": f.win_rate,
                    "avg_pnl": f.avg_pnl,
                }

        # Aggregate rolling stats per factor
        factor_history: dict[str, list[tuple[str, float, float, int]]] = {}
        for snap in history:
            for f in snap.factors:
                factor_history.setdefault(f.factor_name, []).append(
                    (snap.date, f.win_rate, f.avg_pnl, f.trade_count)
                )

        results: list[FactorRollingResult] = []
        alerts: list[str] = []

        for factor_name, daily_stats in factor_history.items():
            # Sort by date
            daily_stats.sort(key=lambda x: x[0])

            total_wins = 0
            total_trades = 0
            total_pnl = 0.0
            for _, wr, avg_pnl, tc in daily_stats:
                total_wins += round(wr * tc)
                total_trades += tc
                total_pnl += avg_pnl * tc

            rolling_win_rate = total_wins / total_trades if total_trades > 0 else 0.0
            rolling_avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0

            # Trend detection: first half vs second half
            mid = len(daily_stats) // 2
            first_half = daily_stats[:mid] if mid > 0 else []
            second_half = daily_stats[mid:] if mid > 0 else daily_stats

            first_wr = self._weighted_avg_win_rate(first_half) if first_half else rolling_win_rate
            second_wr = self._weighted_avg_win_rate(second_half)

            diff = second_wr - first_wr
            if diff > 0.10:
                trend = "improving"
            elif diff < -0.10:
                trend = "degrading"
            else:
                trend = "stable"

            current = current_factors.get(factor_name, {})
            below = rolling_win_rate < win_rate_threshold

            result = FactorRollingResult(
                factor_name=factor_name,
                bot_id=bot_id,
                current_win_rate=current.get("win_rate", 0.0),
                rolling_30d_win_rate=round(rolling_win_rate, 4),
                win_rate_trend=trend,
                current_avg_pnl=current.get("avg_pnl", 0.0),
                rolling_30d_avg_pnl=round(rolling_avg_pnl, 4),
                days_of_data=len(daily_stats),
                below_threshold=below,
            )
            results.append(result)

            if below:
                alerts.append(
                    f"Factor '{factor_name}' rolling win rate {rolling_win_rate:.0%} "
                    f"is below threshold ({win_rate_threshold:.0%})"
                )
            if trend == "degrading":
                alerts.append(
                    f"Factor '{factor_name}' win rate is degrading: "
                    f"first-half {first_wr:.0%} -> second-half {second_wr:.0%}"
                )

        return SignalFactorRollingReport(
            bot_id=bot_id,
            date=current_date,
            window_days=30,
            factors=results,
            alerts=alerts,
        )

    @staticmethod
    def _weighted_avg_win_rate(
        stats: list[tuple[str, float, float, int]],
    ) -> float:
        """Compute trade-weighted average win rate from daily stats."""
        total_wins = 0
        total_trades = 0
        for _, wr, _, tc in stats:
            total_wins += round(wr * tc)
            total_trades += tc
        return total_wins / total_trades if total_trades > 0 else 0.0
