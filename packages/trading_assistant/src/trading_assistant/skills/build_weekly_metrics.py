# skills/build_weekly_metrics.py
"""Weekly metrics aggregation — 7 daily summaries → weekly portfolio metrics.

Deterministic pipeline. No LLM calls. Produces weekly curated data that Claude
interprets for the weekly summary report.

Output directory: data/curated/weekly/<week_start>/
"""
from __future__ import annotations

import json
from pathlib import Path

from trading_assistant.schemas.daily_metrics import BotDailySummary, FilterAnalysis
from trading_assistant.schemas.weekly_metrics import (
    BotWeeklySummary,
    StrategyWeeklySummary,
    WeeklySummary,
    WeekOverWeekComparison,
    ProcessQualityTrend,
    FilterWeeklySummary,
)


class WeeklyMetricsBuilder:
    """Aggregates daily curated data into weekly metrics."""

    def __init__(self, week_start: str, week_end: str, bots: list[str]) -> None:
        self.week_start = week_start
        self.week_end = week_end
        self.bots = bots

    def build_bot_summary(
        self, bot_id: str, dailies: list[BotDailySummary]
    ) -> BotWeeklySummary:
        """Aggregate daily summaries into a single weekly summary for one bot."""
        if not dailies:
            return BotWeeklySummary(
                week_start=self.week_start, week_end=self.week_end, bot_id=bot_id
            )

        total_trades = sum(d.total_trades for d in dailies)
        win_count = sum(d.win_count for d in dailies)
        loss_count = sum(d.loss_count for d in dailies)
        gross_pnl = sum(d.gross_pnl for d in dailies)
        net_pnl = sum(d.net_pnl for d in dailies)
        daily_pnl = {d.date: d.net_pnl for d in dailies}

        # Weighted averages
        wins_with_avg = [(d.avg_win, d.win_count) for d in dailies if d.win_count > 0]
        losses_with_avg = [(d.avg_loss, d.loss_count) for d in dailies if d.loss_count > 0]
        total_win_weight = sum(w for _, w in wins_with_avg)
        total_loss_weight = sum(w for _, w in losses_with_avg)
        avg_win = (
            sum(a * w for a, w in wins_with_avg) / total_win_weight
            if total_win_weight > 0
            else 0.0
        )
        avg_loss = (
            sum(a * w for a, w in losses_with_avg) / total_loss_weight
            if total_loss_weight > 0
            else 0.0
        )

        avg_pq = sum(d.avg_process_quality for d in dailies) / len(dailies)
        max_dd = max(d.max_drawdown_pct for d in dailies)
        missed_count = sum(d.missed_count for d in dailies)
        missed_would_have_won = sum(d.missed_would_have_won for d in dailies)
        error_count = sum(d.error_count for d in dailies)
        avg_uptime = sum(d.uptime_pct for d in dailies) / len(dailies)

        per_strategy = self.build_strategy_summaries(bot_id, dailies)

        return BotWeeklySummary(
            week_start=self.week_start,
            week_end=self.week_end,
            bot_id=bot_id,
            total_trades=total_trades,
            win_count=win_count,
            loss_count=loss_count,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            max_drawdown_pct=max_dd,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_process_quality=avg_pq,
            missed_count=missed_count,
            missed_would_have_won=missed_would_have_won,
            error_count=error_count,
            avg_uptime_pct=avg_uptime,
            daily_pnl=daily_pnl,
            per_strategy_summary=per_strategy,
        )

    def build_strategy_summaries(
        self, bot_id: str, dailies: list[BotDailySummary]
    ) -> dict[str, StrategyWeeklySummary]:
        """Aggregate per-strategy daily data into weekly per-strategy summaries."""
        accum: dict[str, dict] = {}

        for day in dailies:
            for strat_id, strat in day.per_strategy_summary.items():
                if strat_id not in accum:
                    accum[strat_id] = {
                        "trades": 0,
                        "win_count": 0,
                        "loss_count": 0,
                        "gross_pnl": 0.0,
                        "net_pnl": 0.0,
                        "win_amounts": [],
                        "loss_amounts": [],
                        "daily_pnl": {},
                    }
                a = accum[strat_id]
                a["trades"] += strat.trades
                a["win_count"] += strat.win_count
                a["loss_count"] += strat.loss_count
                a["gross_pnl"] += strat.gross_pnl
                a["net_pnl"] += strat.net_pnl
                a["daily_pnl"][day.date] = strat.net_pnl
                if strat.win_count > 0:
                    a["win_amounts"].append((strat.avg_win, strat.win_count))
                if strat.loss_count > 0:
                    a["loss_amounts"].append((strat.avg_loss, strat.loss_count))

        result: dict[str, StrategyWeeklySummary] = {}
        for strat_id, a in accum.items():
            total_win_w = sum(w for _, w in a["win_amounts"])
            total_loss_w = sum(w for _, w in a["loss_amounts"])
            avg_win = (
                sum(v * w for v, w in a["win_amounts"]) / total_win_w
                if total_win_w > 0
                else 0.0
            )
            avg_loss = (
                sum(v * w for v, w in a["loss_amounts"]) / total_loss_w
                if total_loss_w > 0
                else 0.0
            )
            total = a["trades"]
            win_rate = a["win_count"] / total if total > 0 else 0.0

            result[strat_id] = StrategyWeeklySummary(
                strategy_id=strat_id,
                bot_id=bot_id,
                total_trades=total,
                win_count=a["win_count"],
                loss_count=a["loss_count"],
                gross_pnl=a["gross_pnl"],
                net_pnl=a["net_pnl"],
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                daily_pnl=a["daily_pnl"],
            )

        return result

    def build_portfolio_summary(
        self, dailies_by_bot: dict[str, list[BotDailySummary]]
    ) -> WeeklySummary:
        """Aggregate all bots into a portfolio-level weekly summary."""
        bot_summaries = {}
        for bot_id, dailies in dailies_by_bot.items():
            bot_summaries[bot_id] = self.build_bot_summary(bot_id, dailies)

        total_net = sum(s.net_pnl for s in bot_summaries.values())
        total_gross = sum(s.gross_pnl for s in bot_summaries.values())
        total_trades = sum(s.total_trades for s in bot_summaries.values())

        return WeeklySummary(
            week_start=self.week_start,
            week_end=self.week_end,
            bot_summaries=bot_summaries,
            total_net_pnl=total_net,
            total_gross_pnl=total_gross,
            total_trades=total_trades,
        )

    def compare_weeks(
        self, current: WeeklySummary, previous: WeeklySummary
    ) -> WeekOverWeekComparison:
        """Compute week-over-week deltas."""
        pnl_delta = current.total_net_pnl - previous.total_net_pnl
        if previous.total_net_pnl != 0:
            pnl_delta_pct = (pnl_delta / abs(previous.total_net_pnl)) * 100.0
        else:
            pnl_delta_pct = 0.0

        trade_delta = current.total_trades - previous.total_trades

        return WeekOverWeekComparison(
            current_week=current.week_start,
            previous_week=previous.week_start,
            pnl_delta=pnl_delta,
            pnl_delta_pct=pnl_delta_pct,
            trade_count_delta=trade_delta,
        )

    def compute_process_quality_trend(
        self,
        bot_id: str,
        weekly_avg_scores: list[float],
        root_cause_distribution: dict[str, int],
    ) -> ProcessQualityTrend:
        """Determine if process quality is improving, degrading, or stable."""
        current = weekly_avg_scores[-1] if weekly_avg_scores else 0.0

        if len(weekly_avg_scores) >= 2:
            first_half = weekly_avg_scores[: len(weekly_avg_scores) // 2]
            second_half = weekly_avg_scores[len(weekly_avg_scores) // 2 :]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            delta = second_avg - first_avg
            if delta > 2.0:
                direction = "improving"
            elif delta < -2.0:
                direction = "degrading"
            else:
                direction = "stable"
        else:
            direction = "stable"

        return ProcessQualityTrend(
            bot_id=bot_id,
            weekly_avg_scores=weekly_avg_scores,
            current_avg=current,
            trend_direction=direction,
            most_frequent_root_causes=root_cause_distribution,
        )

    def build_filter_weekly_summary(
        self, bot_id: str, daily_filters: list[FilterAnalysis]
    ) -> list[FilterWeeklySummary]:
        """Aggregate 7 days of filter analysis into weekly filter summaries."""
        all_filters: dict[str, dict] = {}

        for day in daily_filters:
            for filter_name, count in day.filter_block_counts.items():
                if filter_name not in all_filters:
                    all_filters[filter_name] = {
                        "total_blocks": 0,
                        "saved_pnl": 0.0,
                        "missed_pnl": 0.0,
                    }
                all_filters[filter_name]["total_blocks"] += count
                all_filters[filter_name]["saved_pnl"] += day.filter_saved_pnl.get(
                    filter_name, 0.0
                )
                all_filters[filter_name]["missed_pnl"] += day.filter_missed_pnl.get(
                    filter_name, 0.0
                )

        return [
            FilterWeeklySummary(
                bot_id=bot_id,
                filter_name=name,
                total_blocks=data["total_blocks"],
                net_impact_pnl=data["saved_pnl"] - data["missed_pnl"],
            )
            for name, data in all_filters.items()
        ]

    def write_weekly_curated(
        self, summary: WeeklySummary, base_dir: Path
    ) -> Path:
        """Write weekly curated data to base_dir/weekly/<week_start>/."""
        output_dir = base_dir / "weekly" / self.week_start
        output_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(
            output_dir / "weekly_summary.json",
            summary.model_dump(mode="json"),
        )

        return output_dir

    def _write_json(self, path: Path, data: dict | list) -> None:
        path.write_text(json.dumps(data, indent=2, default=str))
