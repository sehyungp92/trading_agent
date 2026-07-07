"""Weekly evidence loading support."""

from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class WeeklyEvidenceLoadingSupport:
    """Weekly evidence loading support."""

    def _load_bot_dailies(self, bot_id: str, week_start: str, week_end: str) -> list:
        """Load BotDailySummary objects from curated dir for a date range."""
        from datetime import timedelta
        from trading_assistant.schemas.daily_metrics import BotDailySummary

        start = datetime.strptime(week_start, "%Y-%m-%d")
        dailies = []
        for i in range(7):
            date_str = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            summary_path = self._curated_dir / date_str / bot_id / "summary.json"
            if summary_path.exists():
                try:
                    data = json.loads(summary_path.read_text())
                    dailies.append(BotDailySummary(**data))
                except (json.JSONDecodeError, Exception):
                    logger.warning("Could not load daily summary for %s on %s", bot_id, date_str)
        return dailies

    def _load_weekly_strategy_evidence(
        self,
        week_start: str,
        week_end: str,
        bot_summaries: dict[str, object],
        signal_health_data: dict[str, dict] | None = None,
        factor_rolling_data: dict[str, list[dict]] | None = None,
        simulation_results: dict | None = None,
    ) -> dict:
        """Aggregate curated weekly evidence into strategy-engine inputs."""
        bot_ids = list(bot_summaries.keys())
        trades_by_bot: dict[str, list] = {}
        missed_by_bot: dict[str, list] = {}
        for bot_id in bot_ids:
            trades, missed = self._load_trades_for_week(bot_id, week_start, week_end)
            trades_by_bot[bot_id] = trades
            missed_by_bot[bot_id] = missed

        evidence = {
            "filter_summaries": self._aggregate_weekly_filter_summaries(
                week_start, week_end, bot_ids, missed_by_bot,
            ),
            "regime_trends": self._aggregate_weekly_regime_trends(week_end, bot_ids),
            "rolling_sharpe": self._aggregate_rolling_sharpe(week_end, bot_ids),
            "signal_correlations": self._aggregate_signal_correlations(week_end, bot_ids),
            "hourly_buckets": self._aggregate_hourly_buckets(week_start, week_end, bot_ids),
            "correlation_summaries": self._build_bot_correlation_summaries(bot_summaries),
            "drawdown_data": self._aggregate_drawdown_data(week_end, trades_by_bot),
            "signal_health": signal_health_data or None,
            "factor_rolling": factor_rolling_data or None,
            "filter_interactions": self._load_filter_interactions_from_simulations(
                simulation_results or {},
            ),
            "orderbook_stats": self._aggregate_orderbook_stats(week_start, week_end, bot_ids),
            "exit_sweep": self._extract_per_bot_sim(
                simulation_results or {}, "exit_sweep_", bot_ids,
            ),
            "filter_sensitivity": self._extract_per_bot_sim(
                simulation_results or {}, "filter_sensitivity_", bot_ids,
            ),
            "counterfactual": self._extract_per_bot_sim(
                simulation_results or {}, "counterfactual_", bot_ids,
            ),
        }

        # Load macro regime data from most recent portfolio curated file
        macro_regime_data = None
        end_dt = datetime.strptime(week_end, "%Y-%m-%d")
        for i in range(7):
            date_str = (end_dt - timedelta(days=i)).strftime("%Y-%m-%d")
            mr_path = self._curated_dir / date_str / "portfolio" / "macro_regime_analysis.json"
            if mr_path.exists():
                try:
                    macro_regime_data = json.loads(mr_path.read_text(encoding="utf-8"))
                    if macro_regime_data:
                        break
                except (json.JSONDecodeError, OSError):
                    pass
        evidence["macro_regime_data"] = macro_regime_data

        # Aggregate exit efficiency data for detect_exit_timing_issues
        exit_efficiency_data: dict[str, dict] = {}
        for bot_id in bot_ids:
            daily_effs: list[dict] = []
            for date_str in self._iter_date_range(week_start, week_end):
                data = self._load_json_file(
                    self._curated_dir / date_str / bot_id / "exit_efficiency.json"
                )
                if isinstance(data, dict) and data.get("total_trades_with_data", 0) > 0:
                    daily_effs.append(data)
            if daily_effs:
                avg_eff = statistics.mean(d["avg_efficiency"] for d in daily_effs)
                avg_premature = statistics.mean(d["premature_exit_pct"] for d in daily_effs)
                exit_efficiency_data[bot_id] = {
                    "avg_exit_efficiency": avg_eff,
                    "premature_exit_pct": avg_premature,
                }
        evidence["exit_efficiency_data"] = exit_efficiency_data or None

        # Aggregate enriched instrumentation curated files
        evidence["execution_latency"] = self._aggregate_curated_file(
            "execution_latency.json", week_start, week_end, bot_ids,
        )
        evidence["sizing_data"] = self._aggregate_curated_file(
            "sizing_analysis.json", week_start, week_end, bot_ids,
        )
        evidence["param_correlations"] = self._aggregate_curated_file(
            "param_outcome_correlation.json", week_start, week_end, bot_ids,
        )
        evidence["portfolio_context"] = self._aggregate_curated_file(
            "portfolio_context.json", week_start, week_end, bot_ids,
        )

        # Crypto perpetual curated files (only present for crypto bots)
        evidence["funding_data"] = self._aggregate_curated_file(
            "funding_analysis.json", week_start, week_end, bot_ids,
        )
        evidence["grade_data"] = self._aggregate_curated_file(
            "grade_analysis.json", week_start, week_end, bot_ids,
        )
        evidence["confluence_data"] = self._aggregate_curated_file(
            "confluence_analysis.json", week_start, week_end, bot_ids,
        )
        evidence["leverage_data"] = self._aggregate_curated_file(
            "leverage_analysis.json", week_start, week_end, bot_ids,
        )
        evidence["crypto_trade_data"] = {
            bot_id: trades for bot_id, trades in trades_by_bot.items()
            if any(
                getattr(trade, "funding_paid", 0.0)
                or getattr(trade, "setup_grade", "")
                or getattr(trade, "bias_direction", "")
                or getattr(trade, "sizing_inputs", None)
                for trade in trades
            )
        }

        return {key: value for key, value in evidence.items() if value}

    def _load_filter_interactions_from_simulations(self, simulation_results: dict) -> dict[str, list]:
        """Pull filter interaction evidence from simulator outputs when present."""
        interactions: dict[str, list] = {}
        for key, value in simulation_results.items():
            if not key.startswith("filter_interaction_") or not isinstance(value, dict):
                continue
            bot_id = str(value.get("bot_id", "") or key.removeprefix("filter_interaction_"))
            pairs = value.get("pairs", [])
            if bot_id and isinstance(pairs, list) and pairs:
                interactions[bot_id] = pairs
        return interactions

    @staticmethod
    def _extract_per_bot_sim(
        simulation_results: dict, prefix: str, bot_ids: list[str],
    ) -> dict[str, dict]:
        """Extract per-bot simulator outputs from the weekly results dict.

        The weekly handler stores sim outputs as ``{f"{prefix}{bot_id}": dict}``;
        this helper rebuilds the per-bot mapping that ``StrategyEngine.build_report``
        expects.
        """
        out: dict[str, dict] = {}
        if not simulation_results or not prefix:
            return out
        wanted = set(bot_ids)
        for key, value in simulation_results.items():
            if not isinstance(key, str) or not key.startswith(prefix):
                continue
            if not isinstance(value, dict):
                continue
            bot_id = key[len(prefix):]
            if wanted and bot_id not in wanted:
                continue
            out[bot_id] = value
        return out

    def _aggregate_curated_file(
        self,
        filename: str,
        week_start: str,
        week_end: str,
        bot_ids: list[str],
    ) -> dict[str, dict] | None:
        """Load a named curated JSON file per bot, using most recent day's data."""
        result: dict[str, dict] = {}
        for bot_id in bot_ids:
            # Walk backwards through the week to find most recent data
            for date_str in reversed(self._iter_date_range(week_start, week_end)):
                data = self._load_json_file(
                    self._curated_dir / date_str / bot_id / filename
                )
                if isinstance(data, dict) and data.get("coverage", 0) > 0:
                    result[bot_id] = data
                    break
        return result or None

    def _load_recent_daily_metric_series(
        self, bot_id: str, end_date: str, metric_key: str, max_points: int,
    ) -> list[float]:
        """Load recent metric values from daily summaries for rolling windows."""
        values: list[float] = []
        for date_str in self._list_available_dates(end=end_date):
            summary = self._load_json_file(self._curated_dir / date_str / bot_id / "summary.json")
            if not isinstance(summary, dict):
                continue
            value = summary.get(metric_key)
            if not isinstance(value, (int, float)):
                continue
            values.append(float(value))
        return values[-max_points:]

    def _load_recent_signal_correlation(
        self, bot_id: str, end_date: str, lookback_days: int,
    ) -> float | None:
        """Load weighted average signal correlation over a rolling lookback."""
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=lookback_days - 1)
        weighted_total = 0.0
        total_weight = 0.0

        for date_str in self._iter_date_range(
            start_dt.strftime("%Y-%m-%d"), end_date,
        ):
            data = self._load_json_file(
                self._curated_dir / date_str / bot_id / "signal_health.json"
            )
            if not isinstance(data, dict):
                continue
            components = data.get("components", [])
            if not isinstance(components, list):
                continue
            for component in components:
                if not isinstance(component, dict):
                    continue
                correlation = float(component.get("win_correlation", 0.0) or 0.0)
                weight = max(int(component.get("trade_count", 0) or 0), 1)
                weighted_total += correlation * weight
                total_weight += weight

        if total_weight <= 0:
            return None
        return weighted_total / total_weight

    def _list_available_dates(self, start: str = "", end: str = "") -> list[str]:
        """List curated date directories in lexical date order."""
        if not self._curated_dir.exists():
            return []
        dates: list[str] = []
        for entry in sorted(self._curated_dir.iterdir()):
            if not entry.is_dir():
                continue
            name = entry.name
            if len(name) != 10 or name[4] != "-" or name[7] != "-":
                continue
            if start and name < start:
                continue
            if end and name > end:
                continue
            dates.append(name)
        return dates

    def _iter_date_range(self, start: str, end: str) -> list[str]:
        """Return inclusive YYYY-MM-DD date strings for a range."""
        current = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        dates: list[str] = []
        while current <= end_dt:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        return dates
