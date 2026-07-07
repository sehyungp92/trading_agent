# analysis/weekly_triage.py
"""Deterministic weekly triage — pre-processes a week of curated data to
identify structural patterns, compute trajectory, and generate focused
analytical questions for Claude.

No LLM calls. Produces a WeeklyTriageReport with computed summaries and
2-3 retrospective + 2-3 discovery questions.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class WeeklyAnomaly:
    """A pattern spanning multiple days that deserves analysis."""
    anomaly_type: str  # trajectory_change, cross_bot_divergence, suggestion_retrospective, persistent_pattern
    bot_id: str  # empty for cross-bot
    severity: str  # high, medium
    description: str
    relevant_data_keys: list[str] = field(default_factory=list)


@dataclass
class BotWeekSummary:
    """Computed weekly summary for a single bot."""
    bot_id: str
    total_pnl: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    daily_pnls: list[float] = field(default_factory=list)
    dominant_regime: str = ""
    regime_changes: int = 0
    trend: str = ""  # improving, declining, stable


@dataclass
class WeeklyTriageReport:
    """Output of weekly triage: computed summaries + anomalies + questions."""
    week_start: str
    week_end: str
    bot_summaries: list[BotWeekSummary] = field(default_factory=list)
    computed_summary: str = ""
    anomalies: list[WeeklyAnomaly] = field(default_factory=list)
    retrospective_questions: list[str] = field(default_factory=list)
    discovery_questions: list[str] = field(default_factory=list)
    relevant_data_keys: list[str] = field(default_factory=list)


class WeeklyTriage:
    """Deterministic pre-processor for weekly analysis."""

    def __init__(
        self,
        curated_dir: Path,
        end_date: str,
        bots: list[str],
        active_suggestions: list[dict] | None = None,
        outcome_measurements: list[dict] | None = None,
        prediction_accuracy: dict | None = None,
    ) -> None:
        self._curated_dir = curated_dir
        self._end_date = end_date
        self._bots = bots
        self._active_suggestions = active_suggestions or []
        self._outcome_measurements = outcome_measurements or []
        self._prediction_accuracy = prediction_accuracy or {}
        end = datetime.strptime(end_date, "%Y-%m-%d")
        self._start_date = (end - timedelta(days=6)).strftime("%Y-%m-%d")

    def run(self) -> WeeklyTriageReport:
        """Run weekly triage and produce a focused report."""
        bot_summaries: list[BotWeekSummary] = []
        anomalies: list[WeeklyAnomaly] = []
        relevant_keys: set[str] = set()

        for bot_id in self._bots:
            summary = self._compute_bot_week(bot_id)
            bot_summaries.append(summary)

        # Cross-bot divergence detection
        divergence = self._check_cross_bot_divergence(bot_summaries)
        if divergence:
            anomalies.append(divergence)
            relevant_keys.update(divergence.relevant_data_keys)

        # Suggestion retrospective
        retro_anomalies = self._check_suggestion_retrospective()
        anomalies.extend(retro_anomalies)
        for a in retro_anomalies:
            relevant_keys.update(a.relevant_data_keys)

        # Persistent pattern detection
        for summary in bot_summaries:
            if summary.regime_changes >= 3:
                anomalies.append(WeeklyAnomaly(
                    anomaly_type="persistent_pattern",
                    bot_id=summary.bot_id,
                    severity="medium",
                    description=f"High regime instability: {summary.regime_changes} regime changes this week",
                    relevant_data_keys=["regime_analysis.json"],
                ))

            # Trajectory change
            if summary.trend in ("improving", "declining") and len(summary.daily_pnls) >= 4:
                anomalies.append(WeeklyAnomaly(
                    anomaly_type="trajectory_change",
                    bot_id=summary.bot_id,
                    severity="medium",
                    description=f"Performance trajectory: {summary.trend} (daily PnLs: {[round(p, 1) for p in summary.daily_pnls]})",
                    relevant_data_keys=["summary.json"],
                ))

        # Build computed summary
        computed = self._build_computed_summary(bot_summaries)

        # Generate questions
        retro_questions = self._generate_retrospective_questions(anomalies)
        discovery_questions = self._generate_discovery_questions(bot_summaries, anomalies)

        relevant_keys.add("summary.json")

        return WeeklyTriageReport(
            week_start=self._start_date,
            week_end=self._end_date,
            bot_summaries=bot_summaries,
            computed_summary=computed,
            anomalies=anomalies,
            retrospective_questions=retro_questions,
            discovery_questions=discovery_questions,
            relevant_data_keys=sorted(relevant_keys),
        )

    def _compute_bot_week(self, bot_id: str) -> BotWeekSummary:
        """Compute weekly summary for a single bot from daily summaries."""
        dates = self._week_dates()
        summaries = []
        for d in dates:
            s = self._load_json(bot_id, d, "summary.json")
            if s and isinstance(s, dict):
                summaries.append(s)

        if not summaries:
            return BotWeekSummary(bot_id=bot_id)

        daily_pnls = [self._get_pnl(s) for s in summaries]
        total_wins = sum(s.get("win_count", 0) for s in summaries)
        total_trades = sum(s.get("total_trades", 0) for s in summaries)

        # Regime analysis
        regimes = []
        for d in dates:
            r = self._load_json(bot_id, d, "regime_analysis.json")
            if r and isinstance(r, dict):
                regime = r.get("dominant_regime", "") or r.get("regime", "")
                if regime:
                    regimes.append(regime)

        regime_changes = 0
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i - 1]:
                regime_changes += 1

        dominant = ""
        if regimes:
            counts: dict[str, int] = {}
            for r in regimes:
                counts[r] = counts.get(r, 0) + 1
            dominant = max(counts, key=counts.get)  # type: ignore[arg-type]

        # Trend detection (uses difference-based comparison to handle negative PnL)
        trend = "stable"
        if len(daily_pnls) >= 4:
            first_half = sum(daily_pnls[:len(daily_pnls) // 2])
            second_half = sum(daily_pnls[len(daily_pnls) // 2:])
            diff = second_half - first_half
            # Use absolute scale to avoid sign issues with multiplicative comparison
            scale = max(abs(first_half), abs(second_half), 1.0)
            if diff > scale * 0.3:
                trend = "improving"
            elif diff < -scale * 0.3:
                trend = "declining"

        return BotWeekSummary(
            bot_id=bot_id,
            total_pnl=sum(daily_pnls),
            total_trades=total_trades,
            win_rate=total_wins / total_trades if total_trades > 0 else 0.0,
            max_drawdown=max((abs(s.get("max_drawdown_pct", 0)) for s in summaries), default=0),
            daily_pnls=daily_pnls,
            dominant_regime=dominant,
            regime_changes=regime_changes,
            trend=trend,
        )

    def _check_cross_bot_divergence(
        self, summaries: list[BotWeekSummary]
    ) -> WeeklyAnomaly | None:
        """Detect when bots significantly diverge in performance."""
        pnls = [(s.bot_id, s.total_pnl) for s in summaries if s.total_trades > 0]
        if len(pnls) < 2:
            return None
        values = [p for _, p in pnls]
        if max(values) - min(values) == 0:
            return None
        mean = statistics.mean(values)
        if len(values) >= 3:
            stdev = statistics.stdev(values)
            spread = max(abs(mean) * 0.5, 1.0)  # minimum threshold to avoid false positives near zero
            if stdev > 0 and stdev > spread:
                best = max(pnls, key=lambda x: x[1])
                worst = min(pnls, key=lambda x: x[1])
                return WeeklyAnomaly(
                    anomaly_type="cross_bot_divergence",
                    bot_id="",
                    severity="high",
                    description=f"Large performance divergence: {best[0]}={best[1]:+.2f} vs {worst[0]}={worst[1]:+.2f}",
                    relevant_data_keys=["summary.json", "regime_analysis.json"],
                )
        return None

    def _check_suggestion_retrospective(self) -> list[WeeklyAnomaly]:
        """Check outcome measurements for surprising results."""
        anomalies = []
        for outcome in self._outcome_measurements:
            verdict = outcome.get("verdict", "")
            quality = outcome.get("measurement_quality", "high")
            sid = outcome.get("suggestion_id", "?")[:8]
            if verdict == "negative" and quality in ("high", "medium"):
                anomalies.append(WeeklyAnomaly(
                    anomaly_type="suggestion_retrospective",
                    bot_id=outcome.get("bot_id", ""),
                    severity="high",
                    description=f"Suggestion #{sid} has NEGATIVE measured outcome (quality: {quality})",
                    relevant_data_keys=["summary.json"],
                ))
            elif verdict == "inconclusive":
                anomalies.append(WeeklyAnomaly(
                    anomaly_type="suggestion_retrospective",
                    bot_id=outcome.get("bot_id", ""),
                    severity="medium",
                    description=f"Suggestion #{sid} outcome is INCONCLUSIVE — regime mismatch or confounders",
                    relevant_data_keys=["regime_analysis.json"],
                ))
        return anomalies[:3]

    def _generate_retrospective_questions(
        self, anomalies: list[WeeklyAnomaly]
    ) -> list[str]:
        """Generate 2-3 retrospective questions about past decisions."""
        questions: list[str] = []

        retro = [a for a in anomalies if a.anomaly_type == "suggestion_retrospective"]
        for a in retro[:2]:
            questions.append(
                f"{a.description}. What went wrong with the original analysis? "
                f"Was the evidence insufficient, or did market conditions change?"
            )

        # Prediction accuracy retrospective
        acc = self._prediction_accuracy
        if acc.get("has_predictions") and acc.get("accuracy_by_metric"):
            worst_metrics = sorted(
                acc["accuracy_by_metric"].items(),
                key=lambda x: x[1].get("accuracy", 1.0) if isinstance(x[1], dict) else 1.0,
            )
            if worst_metrics:
                metric, data = worst_metrics[0]
                acc_val = data.get("accuracy", 0) if isinstance(data, dict) else 0
                if acc_val < 0.5:
                    questions.append(
                        f"Prediction accuracy for '{metric}' is {acc_val:.0%}. "
                        f"Why are predictions for this metric systematically off? "
                        f"Should the prediction approach change?"
                    )

        if not questions:
            questions.append(
                "Review this week's suggestions and predictions. Which were well-calibrated "
                "and which were overconfident? What patterns distinguish good predictions from bad ones?"
            )

        return questions[:3]

    def _generate_discovery_questions(
        self,
        summaries: list[BotWeekSummary],
        anomalies: list[WeeklyAnomaly],
    ) -> list[str]:
        """Generate 2-3 discovery questions about novel patterns."""
        questions: list[str] = []

        divergence = [a for a in anomalies if a.anomaly_type == "cross_bot_divergence"]
        for a in divergence[:1]:
            questions.append(
                f"{a.description}. What market factor explains this divergence? "
                f"Is there a common characteristic of the winning bot's trades that "
                f"the losing bot could learn from?"
            )

        trajectory = [a for a in anomalies if a.anomaly_type == "trajectory_change"]
        for a in trajectory[:1]:
            questions.append(
                f"[{a.bot_id}] {a.description}. Is this the beginning of a structural change, "
                f"or is it explainable by recent regime conditions?"
            )

        if not questions:
            questions.append(
                "What patterns in this week's data are NOT covered by the automated detectors? "
                "Look for correlations between time-of-day, regime, and specific trade setups "
                "that might suggest a new hypothesis."
            )

        questions.append(
            "Based on this week's data, what is one concrete testable hypothesis "
            "about the bots' strategies that could be validated with next week's data?"
        )

        return questions[:3]

    def _build_computed_summary(self, bot_summaries: list[BotWeekSummary]) -> str:
        """Build a human-readable computed summary from bot week summaries."""
        if not bot_summaries:
            return "No bot data available for this week."
        lines = []
        total_pnl = 0.0
        total_trades = 0
        for s in bot_summaries:
            trend_tag = f" [{s.trend}]" if s.trend and s.trend != "stable" else ""
            lines.append(
                f"  {s.bot_id}: PnL={s.total_pnl:+.2f}, "
                f"{s.total_trades} trades, WR={s.win_rate:.0%}, "
                f"maxDD={s.max_drawdown:.1f}%{trend_tag}"
            )
            total_pnl += s.total_pnl
            total_trades += s.total_trades
        header = f"Week total: PnL={total_pnl:+.2f}, {total_trades} trades"
        return header + "\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _week_dates(self) -> list[str]:
        end = datetime.strptime(self._end_date, "%Y-%m-%d")
        return [(end - timedelta(days=d)).strftime("%Y-%m-%d") for d in range(6, -1, -1)]

    def _load_json(
        self, bot_id: str, date: str, filename: str
    ) -> dict | list | None:
        path = self._curated_dir / date / bot_id / filename
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _get_pnl(summary: dict) -> float:
        val = summary.get("net_pnl")
        if val is None:
            val = summary.get("gross_pnl", 0)
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0
