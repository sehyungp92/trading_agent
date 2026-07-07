"""Weekly aggregate calculation support."""

from __future__ import annotations

import logging
import math
import statistics
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class WeeklyAggregateSupport:
    """Weekly aggregate calculation support."""

    def _aggregate_weekly_filter_summaries(
        self,
        week_start: str,
        week_end: str,
        bot_ids: list[str],
        missed_by_bot: dict[str, list],
    ) -> dict[str, list]:
        """Aggregate daily filter analysis and missed events into weekly summaries."""
        from trading_assistant.schemas.weekly_metrics import FilterWeeklySummary

        aggregated: dict[str, dict[str, dict[str, float]]] = {}
        for date_str in self._iter_date_range(week_start, week_end):
            for bot_id in bot_ids:
                data = self._load_json_file(
                    self._curated_dir / date_str / bot_id / "filter_analysis.json"
                )
                if not isinstance(data, dict):
                    continue
                counts = data.get("filter_block_counts", {})
                saved = data.get("filter_saved_pnl", {})
                missed_pnl = data.get("filter_missed_pnl", {})
                if not isinstance(counts, dict):
                    counts = {}
                if not isinstance(saved, dict):
                    saved = {}
                if not isinstance(missed_pnl, dict):
                    missed_pnl = {}

                for filter_name in set(counts) | set(saved) | set(missed_pnl):
                    record = aggregated.setdefault(bot_id, {}).setdefault(
                        filter_name,
                        {
                            "total_blocks": 0.0,
                            "blocks_that_would_have_won": 0.0,
                            "blocks_that_would_have_lost": 0.0,
                            "net_impact_pnl": 0.0,
                        },
                    )
                    record["total_blocks"] += float(counts.get(filter_name, 0) or 0)
                    record["net_impact_pnl"] += (
                        float(saved.get(filter_name, 0.0) or 0.0)
                        - float(missed_pnl.get(filter_name, 0.0) or 0.0)
                    )

        for bot_id, missed_events in missed_by_bot.items():
            for event in missed_events:
                filter_name = getattr(event, "blocked_by", "") or ""
                if not filter_name:
                    continue
                record = aggregated.setdefault(bot_id, {}).setdefault(
                    filter_name,
                    {
                        "total_blocks": 0.0,
                        "blocks_that_would_have_won": 0.0,
                        "blocks_that_would_have_lost": 0.0,
                        "net_impact_pnl": 0.0,
                    },
                )
                if getattr(event, "outcome_24h", 0.0) > 0:
                    record["blocks_that_would_have_won"] += 1.0
                else:
                    record["blocks_that_would_have_lost"] += 1.0

        results: dict[str, list] = {}
        for bot_id, filters in aggregated.items():
            summaries: list[FilterWeeklySummary] = []
            for filter_name, record in sorted(filters.items()):
                total_blocks = int(record["total_blocks"])
                classified = (
                    int(record["blocks_that_would_have_won"])
                    + int(record["blocks_that_would_have_lost"])
                )
                confidence = min(1.0, classified / total_blocks) if total_blocks > 0 else 0.0
                summaries.append(
                    FilterWeeklySummary(
                        bot_id=bot_id,
                        filter_name=filter_name,
                        total_blocks=total_blocks,
                        blocks_that_would_have_won=int(record["blocks_that_would_have_won"]),
                        blocks_that_would_have_lost=int(record["blocks_that_would_have_lost"]),
                        net_impact_pnl=round(record["net_impact_pnl"], 4),
                        confidence=round(confidence, 4),
                    )
                )
            if summaries:
                results[bot_id] = summaries
        return results

    def _aggregate_weekly_regime_trends(
        self, week_end: str, bot_ids: list[str], lookback_weeks: int = 4,
    ) -> dict[str, list]:
        """Build regime trends from the last few weeks of daily regime analysis."""
        from trading_assistant.schemas.weekly_metrics import RegimePerformanceTrend

        end_dt = datetime.strptime(week_end, "%Y-%m-%d")
        history: dict[str, dict[str, dict[str, list[float] | list[int]]]] = {}

        for week_offset in range(lookback_weeks - 1, -1, -1):
            window_end = end_dt - timedelta(days=week_offset * 7)
            window_start = window_end - timedelta(days=6)
            weekly_data: dict[str, dict[str, dict[str, float]]] = {}

            for date_str in self._iter_date_range(
                window_start.strftime("%Y-%m-%d"),
                window_end.strftime("%Y-%m-%d"),
            ):
                for bot_id in bot_ids:
                    data = self._load_json_file(
                        self._curated_dir / date_str / bot_id / "regime_analysis.json"
                    )
                    if not isinstance(data, dict):
                        continue
                    regime_pnl = data.get("regime_pnl", {})
                    regime_trade_count = data.get("regime_trade_count", {})
                    regime_win_rate = data.get("regime_win_rate", {})
                    if not isinstance(regime_pnl, dict):
                        regime_pnl = {}
                    if not isinstance(regime_trade_count, dict):
                        regime_trade_count = {}
                    if not isinstance(regime_win_rate, dict):
                        regime_win_rate = {}

                    for regime_name in (
                        set(regime_pnl) | set(regime_trade_count) | set(regime_win_rate)
                    ):
                        record = weekly_data.setdefault(bot_id, {}).setdefault(
                            regime_name,
                            {"pnl": 0.0, "trade_count": 0.0, "weighted_wins": 0.0},
                        )
                        trade_count = float(regime_trade_count.get(regime_name, 0) or 0)
                        record["pnl"] += float(regime_pnl.get(regime_name, 0.0) or 0.0)
                        record["trade_count"] += trade_count
                        record["weighted_wins"] += (
                            float(regime_win_rate.get(regime_name, 0.0) or 0.0) * trade_count
                        )

            for bot_id in bot_ids:
                known_regimes = set(history.get(bot_id, {})) | set(weekly_data.get(bot_id, {}))
                for regime_name in known_regimes:
                    payload = weekly_data.get(bot_id, {}).get(
                        regime_name,
                        {"pnl": 0.0, "trade_count": 0.0, "weighted_wins": 0.0},
                    )
                    trend = history.setdefault(bot_id, {}).setdefault(
                        regime_name,
                        {
                            "weekly_pnl": [],
                            "weekly_trade_count": [],
                            "weekly_win_rate": [],
                        },
                    )
                    trade_count = int(payload["trade_count"])
                    win_rate = (
                        float(payload["weighted_wins"]) / float(payload["trade_count"])
                        if payload["trade_count"] > 0 else 0.0
                    )
                    trend["weekly_pnl"].append(round(float(payload["pnl"]), 4))
                    trend["weekly_trade_count"].append(trade_count)
                    trend["weekly_win_rate"].append(round(win_rate, 4))

        results: dict[str, list] = {}
        for bot_id, regimes in history.items():
            entries: list[RegimePerformanceTrend] = []
            for regime_name, payload in sorted(regimes.items()):
                trade_counts = payload["weekly_trade_count"]
                if not any(trade_counts):
                    continue
                entries.append(
                    RegimePerformanceTrend(
                        bot_id=bot_id,
                        regime=regime_name,
                        weekly_pnl=list(payload["weekly_pnl"]),
                        weekly_win_rate=list(payload["weekly_win_rate"]),
                        weekly_trade_count=list(trade_counts),
                    )
                )
            if entries:
                results[bot_id] = entries
        return results

    def _aggregate_rolling_sharpe(
        self, week_end: str, bot_ids: list[str],
    ) -> dict[str, dict[str, float]]:
        """Compute 30d/60d/90d rolling Sharpe from daily summaries."""
        results: dict[str, dict[str, float]] = {}
        for bot_id in bot_ids:
            pnls = self._load_recent_daily_metric_series(bot_id, week_end, "net_pnl", 90)
            if not pnls:
                continue
            results[bot_id] = {
                "30d": round(self._compute_sharpe(pnls[-30:]), 4),
                "60d": round(self._compute_sharpe(pnls[-60:]), 4),
                "90d": round(self._compute_sharpe(pnls[-90:]), 4),
            }
        return results

    def _aggregate_signal_correlations(
        self, week_end: str, bot_ids: list[str],
    ) -> dict[str, dict[str, float]]:
        """Compute short/long signal-to-outcome correlation baselines."""
        results: dict[str, dict[str, float]] = {}
        for bot_id in bot_ids:
            corr_30d = self._load_recent_signal_correlation(bot_id, week_end, 30)
            corr_90d = self._load_recent_signal_correlation(bot_id, week_end, 90)
            if corr_30d is None and corr_90d is None:
                continue
            results[bot_id] = {
                "30d": round(corr_30d or 0.0, 4),
                "90d": round(corr_90d if corr_90d is not None else (corr_30d or 0.0), 4),
            }
        return results

    def _aggregate_hourly_buckets(
        self, week_start: str, week_end: str, bot_ids: list[str],
    ) -> dict[str, list]:
        """Aggregate hourly buckets across the current week."""
        from trading_assistant.schemas.hourly_performance import HourlyBucket

        aggregated: dict[str, dict[int, dict[str, float]]] = {}
        for date_str in self._iter_date_range(week_start, week_end):
            for bot_id in bot_ids:
                data = self._load_json_file(
                    self._curated_dir / date_str / bot_id / "hourly_performance.json"
                )
                if not isinstance(data, dict):
                    continue
                buckets = data.get("buckets", [])
                if not isinstance(buckets, list):
                    continue
                for bucket in buckets:
                    if not isinstance(bucket, dict):
                        continue
                    hour = int(bucket.get("hour", 0) or 0)
                    trade_count = int(bucket.get("trade_count", 0) or 0)
                    record = aggregated.setdefault(bot_id, {}).setdefault(
                        hour,
                        {
                            "trade_count": 0.0,
                            "pnl": 0.0,
                            "win_count": 0.0,
                            "process_quality_sum": 0.0,
                        },
                    )
                    record["trade_count"] += trade_count
                    record["pnl"] += float(bucket.get("pnl", 0.0) or 0.0)
                    record["win_count"] += (
                        float(bucket.get("win_rate", 0.0) or 0.0) * trade_count
                    )
                    record["process_quality_sum"] += (
                        float(bucket.get("avg_process_quality", 0.0) or 0.0) * trade_count
                    )

        results: dict[str, list] = {}
        for bot_id, hours in aggregated.items():
            buckets: list[HourlyBucket] = []
            for hour, record in sorted(hours.items()):
                trade_count = int(record["trade_count"])
                if trade_count <= 0:
                    continue
                buckets.append(
                    HourlyBucket(
                        hour=hour,
                        trade_count=trade_count,
                        pnl=round(record["pnl"], 4),
                        win_rate=round(record["win_count"] / trade_count, 4),
                        avg_process_quality=round(
                            record["process_quality_sum"] / trade_count, 4,
                        ),
                    )
                )
            if buckets:
                results[bot_id] = buckets
        return results

    def _build_bot_correlation_summaries(self, bot_summaries: dict[str, object]) -> list:
        """Compute bot-level weekly correlation summaries from daily PnL series."""
        from trading_assistant.schemas.weekly_metrics import CorrelationSummary

        bot_ids = sorted(bot_summaries)
        all_dates: set[str] = set()
        daily_pnl_by_bot: dict[str, dict[str, float]] = {}
        for bot_id, summary in bot_summaries.items():
            daily = getattr(summary, "daily_pnl", {}) or {}
            if not isinstance(daily, dict):
                daily = {}
            daily_pnl_by_bot[bot_id] = {str(k): float(v) for k, v in daily.items()}
            all_dates.update(daily.keys())

        if not all_dates:
            return []

        ordered_dates = sorted(all_dates)
        summaries: list[CorrelationSummary] = []
        for idx, bot_a in enumerate(bot_ids):
            for bot_b in bot_ids[idx + 1:]:
                series_a = [daily_pnl_by_bot.get(bot_a, {}).get(date, 0.0) for date in ordered_dates]
                series_b = [daily_pnl_by_bot.get(bot_b, {}).get(date, 0.0) for date in ordered_dates]
                active_pairs = [
                    (left, right)
                    for left, right in zip(series_a, series_b, strict=True)
                    if left != 0.0 and right != 0.0
                ]
                same_direction = 0.0
                if active_pairs:
                    same_direction = (
                        sum(1 for left, right in active_pairs if left * right > 0)
                        / len(active_pairs)
                    )
                corr = self._pearson(series_a, series_b)
                summaries.append(
                    CorrelationSummary(
                        bot_a=bot_a,
                        bot_b=bot_b,
                        rolling_30d_correlation=round(corr, 4),
                        weekly_pnl_correlation=round(corr, 4),
                        same_direction_pct=round(same_direction, 4),
                    )
                )
        return summaries

    def _aggregate_drawdown_data(
        self, week_end: str, trades_by_bot: dict[str, list],
    ) -> dict[str, dict]:
        """Compute drawdown concentration inputs for the strategy engine."""
        from trading_assistant.skills.drawdown_analyzer import DrawdownAnalyzer

        results: dict[str, dict] = {}
        for bot_id, trades in trades_by_bot.items():
            if not trades:
                continue
            report = DrawdownAnalyzer(bot_id=bot_id, date=week_end).compute(trades)
            loss_pcts = [abs(float(t.pnl_pct)) for t in trades if getattr(t, "pnl", 0.0) < 0]
            results[bot_id] = {
                "largest_single_loss_pct": round(report.largest_single_loss_pct, 4),
                "max_drawdown_pct": round(report.max_drawdown_pct, 4),
                "avg_loss_pct": round(
                    (sum(loss_pcts) / len(loss_pcts)) if loss_pcts else 0.0,
                    4,
                ),
            }
        return results

    def _aggregate_orderbook_stats(
        self, week_start: str, week_end: str, bot_ids: list[str],
    ) -> dict[str, dict]:
        """Aggregate orderbook context stats across the current week."""
        aggregated: dict[str, dict[str, dict[str, float]]] = {}
        for date_str in self._iter_date_range(week_start, week_end):
            for bot_id in bot_ids:
                data = self._load_json_file(
                    self._curated_dir / date_str / bot_id / "orderbook_stats.json"
                )
                if not isinstance(data, dict):
                    continue
                by_context = data.get("by_context", {})
                if not isinstance(by_context, dict):
                    continue
                for context_name, context_data in by_context.items():
                    if not isinstance(context_data, dict):
                        continue
                    count = int(context_data.get("count", 0) or 0)
                    if count <= 0:
                        continue
                    record = aggregated.setdefault(bot_id, {}).setdefault(
                        context_name,
                        {
                            "count": 0.0,
                            "spread_sum": 0.0,
                            "imbalance_sum": 0.0,
                        },
                    )
                    record["count"] += count
                    record["spread_sum"] += (
                        float(context_data.get("spread_stats", {}).get("mean", 0.0) or 0.0)
                        * count
                    )
                    record["imbalance_sum"] += (
                        float(context_data.get("imbalance_stats", {}).get("mean", 0.0) or 0.0)
                        * count
                    )

        results: dict[str, dict] = {}
        for bot_id, contexts in aggregated.items():
            by_context: dict[str, dict] = {}
            for context_name, record in contexts.items():
                count = int(record["count"])
                if count <= 0:
                    continue
                by_context[context_name] = {
                    "count": count,
                    "spread_stats": {"mean": round(record["spread_sum"] / count, 4)},
                    "imbalance_stats": {"mean": round(record["imbalance_sum"] / count, 4)},
                }
            if by_context:
                results[bot_id] = {"by_context": by_context}
        return results

    @staticmethod
    def _compute_sharpe(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        try:
            mean = statistics.mean(values)
            std = statistics.stdev(values)
        except statistics.StatisticsError:
            return 0.0
        if std == 0:
            return 0.0
        return (mean / std) * math.sqrt(252)

    @staticmethod
    def _pearson(left: list[float], right: list[float]) -> float:
        if len(left) != len(right) or len(left) < 2:
            return 0.0
        try:
            mean_left = statistics.mean(left)
            mean_right = statistics.mean(right)
        except statistics.StatisticsError:
            return 0.0
        numerator = sum(
            (a - mean_left) * (b - mean_right)
            for a, b in zip(left, right, strict=True)
        )
        denom_left = math.sqrt(sum((a - mean_left) ** 2 for a in left))
        denom_right = math.sqrt(sum((b - mean_right) ** 2 for b in right))
        if denom_left == 0 or denom_right == 0:
            return 0.0
        return numerator / (denom_left * denom_right)
