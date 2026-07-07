"""Weekly retrospective that compares prior predictions to realized outcomes."""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel
from trading_assistant.schemas.learning_ledger import RetrospectiveSynthesis

logger = logging.getLogger(__name__)


class PredictionOutcome(BaseModel):
    """A single prior prediction matched against realized outcomes."""

    prediction_source: str
    prediction_text: str
    prediction_type: str
    actual_outcome: str = ""
    accuracy: str = ""
    bot_id: str = ""
    metric: str = ""
    predicted_direction: str = ""
    correct: bool | None = None


class WeeklyRetrospective(BaseModel):
    """Summary of retrospective accuracy for a weekly review window."""

    week_start: str
    week_end: str
    predictions_reviewed: int = 0
    correct: int = 0
    partially_correct: int = 0
    incorrect: int = 0
    unverifiable: int = 0
    accuracy_pct: float = 0.0
    predictions: list[PredictionOutcome] = []
    summary: str = ""


_SUGGESTION_MATCHERS: dict[str, dict[str, object]] = {
    "stop": {
        "metrics": ["avg_mae_pct", "exit_efficiency"],
        "positive_direction": "decrease_mae_or_increase_efficiency",
    },
    "filter": {
        "metrics": ["missed_would_have_won"],
        "positive_direction": "decrease_missed_winners",
    },
    "regime": {
        "metrics": ["regime_pnl"],
        "positive_direction": "improvement_in_flagged_regime",
    },
    "hour": {
        "metrics": ["hourly_performance"],
        "positive_direction": "improvement_in_flagged_hours",
    },
    "position_siz": {
        "metrics": ["avg_win", "avg_loss"],
        "positive_direction": "loss_win_ratio_decrease",
    },
    "drawdown": {
        "metrics": ["max_drawdown_pct"],
        "positive_direction": "decrease",
    },
    "allocation": {
        "metrics": ["net_pnl"],
        "positive_direction": "increase",
    },
}

_METRIC_ALIASES: dict[str, list[str]] = {
    "pnl": ["net_pnl", "total_pnl", "pnl"],
    "net_pnl": ["net_pnl", "total_pnl", "pnl"],
    "win_rate": ["win_rate", "win_pct"],
    "drawdown": ["max_drawdown_pct", "max_drawdown", "drawdown"],
    "max_drawdown_pct": ["max_drawdown_pct", "max_drawdown", "drawdown"],
    "sharpe": ["sharpe_rolling_30d", "sharpe", "sharpe_ratio"],
    "sharpe_rolling_30d": ["sharpe_rolling_30d", "sharpe", "sharpe_ratio"],
}

_DECREASING_METRICS = {"drawdown", "max_drawdown_pct"}


class RetrospectiveBuilder:
    """Builds retrospective accuracy from prior runs and curated outcomes."""

    def __init__(self, runs_dir: Path, curated_dir: Path, memory_dir: Path) -> None:
        self._runs_dir = runs_dir
        self._curated_dir = curated_dir
        self._memory_dir = memory_dir

    def build(self, week_start: str, week_end: str) -> WeeklyRetrospective:
        """Build a retrospective for the requested weekly window."""
        start = datetime.strptime(week_start, "%Y-%m-%d")
        end = datetime.strptime(week_end, "%Y-%m-%d")

        predictions = self._extract_predictions(start, end)
        current_outcomes = self._load_actual_outcomes(start, end)

        prior_start = start - timedelta(days=7)
        prior_end = start - timedelta(days=1)
        prior_outcomes = self._load_actual_outcomes(prior_start, prior_end)

        matched: list[PredictionOutcome] = []
        for prediction in predictions:
            if prediction.metric:
                prediction.actual_outcome = self._describe_metric_outcome(
                    prediction, prior_outcomes, current_outcomes,
                )
            else:
                prediction.actual_outcome = self._find_matching_outcome(
                    prediction, current_outcomes,
                )
            prediction.accuracy = self._assess_accuracy(
                prediction, prior_outcomes, current_outcomes,
            )
            prediction.correct = prediction.accuracy == "correct"
            matched.append(prediction)

        correct = sum(1 for p in matched if p.accuracy == "correct")
        partial = sum(1 for p in matched if p.accuracy == "partially_correct")
        incorrect = sum(1 for p in matched if p.accuracy == "incorrect")
        unverifiable = sum(1 for p in matched if p.accuracy == "unverifiable")
        total = len(matched)

        accuracy_pct = (correct + 0.5 * partial) / total * 100.0 if total else 0.0

        return WeeklyRetrospective(
            week_start=week_start,
            week_end=week_end,
            predictions_reviewed=total,
            correct=correct,
            partially_correct=partial,
            incorrect=incorrect,
            unverifiable=unverifiable,
            accuracy_pct=round(accuracy_pct, 1),
            predictions=matched,
            summary=self._build_summary(correct, partial, incorrect, unverifiable, total),
        )

    def build_synthesis(self, week_start: str, week_end: str) -> RetrospectiveSynthesis:
        """Build a keep/discard synthesis from outcomes and ground truth.

        1. Load outcomes.jsonl for the week period
        2. Load ground truth deltas from learning_ledger.jsonl (graceful if empty)
        3. Match outcomes to ground truth: positive + GT improvement → what_worked
        4. Identify discard categories: (bot_id, category) with 3+ failures, 0 successes
        5. Extract lessons from outcome reasonings
        6. Persist to retrospective_synthesis.jsonl
        """
        from trading_assistant.schemas.learning_ledger import (
            DiscardItem,
            RetrospectiveSynthesis,
            SynthesisItem,
        )

        # Load outcomes for the week
        outcomes = self._load_week_outcomes(week_start, week_end)
        # Load ground truth deltas
        gt_deltas = self._load_gt_deltas(week_start)
        # Load outcome reasonings for mechanism details
        reasonings = self._load_week_reasonings(week_start, week_end)

        # Build reasoning lookup
        reasoning_by_sid: dict[str, dict] = {}
        for r in reasonings:
            sid = r.get("suggestion_id", "")
            if sid:
                reasoning_by_sid[sid] = r

        what_worked: list[SynthesisItem] = []
        what_failed: list[SynthesisItem] = []

        # Track per-(bot_id, category) counts for discard detection
        category_results: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"success": 0, "failure": 0}
        )

        for outcome in outcomes:
            sid = outcome.get("suggestion_id", "")
            bot_id = outcome.get("bot_id", "")
            category = outcome.get("category", "")
            title = outcome.get("title", "")
            verdict = outcome.get("verdict", "")

            # Get ground truth delta for this bot
            bot_gt_delta = gt_deltas.get(bot_id, 0.0)
            # Get mechanism from reasoning
            reasoning = reasoning_by_sid.get(sid, {})
            mechanism = reasoning.get("mechanism", "")

            item = SynthesisItem(
                suggestion_id=sid,
                bot_id=bot_id,
                category=category,
                title=title,
                outcome_verdict=verdict,
                ground_truth_delta=bot_gt_delta,
                mechanism=mechanism,
            )

            # GT-aware classification: positive verdict + declining bot → failed
            if verdict == "positive" and bot_gt_delta >= 0:
                what_worked.append(item)
                category_results[(bot_id, category)]["success"] += 1
            elif verdict in ("negative", "neutral") or (
                verdict == "positive" and bot_gt_delta < 0
            ):
                what_failed.append(item)
                category_results[(bot_id, category)]["failure"] += 1

        # Identify discard categories: 3+ failures, 0 successes
        discard: list[DiscardItem] = []
        for (bot_id, category), counts in category_results.items():
            if counts["failure"] >= 3 and counts["success"] == 0:
                discard.append(DiscardItem(
                    bot_id=bot_id,
                    category=category,
                    failure_count=counts["failure"],
                    reason=f"{counts['failure']} attempts, 0 improvements",
                ))

        # Extract unique lessons from outcome reasonings
        lessons: list[str] = []
        seen_lessons: set[str] = set()
        for r in reasonings:
            raw = r.get("lessons_learned", [])
            # Handle both string and list formats
            items = [raw] if isinstance(raw, str) else (raw or [])
            for lesson in items:
                if not isinstance(lesson, str):
                    continue
                normalized = lesson.strip().lower()
                if normalized and normalized not in seen_lessons:
                    seen_lessons.add(normalized)
                    lessons.append(lesson)

        synthesis = RetrospectiveSynthesis(
            week_start=week_start,
            week_end=week_end,
            what_worked=what_worked,
            what_failed=what_failed,
            discard=discard,
            lessons=lessons,
            ground_truth_deltas=gt_deltas,
        )

        # Persist
        synth_path = self._memory_dir / "findings" / "retrospective_synthesis.jsonl"
        synth_path.parent.mkdir(parents=True, exist_ok=True)
        with open(synth_path, "a", encoding="utf-8") as f:
            f.write(synthesis.model_dump_json() + "\n")

        return synthesis

    def _load_week_outcomes(self, week_start: str, week_end: str) -> list[dict]:
        """Load outcomes from the given week period."""
        path = self._memory_dir / "findings" / "outcomes.jsonl"
        if not path.exists():
            return []
        outcomes: list[dict] = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            try:
                o = json.loads(line)
                measured = o.get("measurement_date", "") or o.get("measured_at", "") or o.get("timestamp", "")
                if measured and week_start <= measured[:10] <= week_end:
                    outcomes.append(o)
            except json.JSONDecodeError:
                pass
        return outcomes

    def _load_gt_deltas(self, week_start: str) -> dict[str, float]:
        """Load ground truth composite deltas from learning_ledger.jsonl."""
        path = self._memory_dir / "findings" / "learning_ledger.jsonl"
        if not path.exists():
            return {}
        try:
            for line in reversed(path.read_text(encoding="utf-8").strip().splitlines()):
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("week_start") == week_start:
                    return entry.get("composite_delta", {})
        except (json.JSONDecodeError, OSError):
            pass
        return {}

    def _load_week_reasonings(self, week_start: str, week_end: str) -> list[dict]:
        """Load outcome reasonings from the given week period."""
        path = self._memory_dir / "findings" / "outcome_reasonings.jsonl"
        if not path.exists():
            return []
        reasonings: list[dict] = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                ts = r.get("reasoned_at", "") or r.get("timestamp", "")
                if ts and week_start <= ts[:10] <= week_end:
                    reasonings.append(r)
            except json.JSONDecodeError:
                pass
        return reasonings

    def _extract_predictions(self, start: datetime, end: datetime) -> list[PredictionOutcome]:
        """Extract predictions from daily and weekly run outputs."""
        predictions: list[PredictionOutcome] = []
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            for run_dir in self._find_run_dirs("daily", date_str):
                predictions.extend(self._parse_run_predictions(run_dir, run_dir.name))
            current += timedelta(days=1)

        prev_week = (start - timedelta(days=7)).strftime("%Y-%m-%d")
        for run_dir in self._find_run_dirs("weekly", prev_week):
            predictions.extend(self._parse_run_predictions(run_dir, run_dir.name))

        return predictions

    def _find_run_dirs(self, run_type: str, date_str: str) -> list[Path]:
        """Return real and legacy run directories for a given date."""
        seen: dict[Path, Path] = {}
        for run_dir in sorted(self._runs_dir.glob(f"{run_type}-{date_str}*")):
            if run_dir.is_dir():
                seen[run_dir] = run_dir

        legacy_daily = self._runs_dir / date_str / "daily-report"
        legacy_weekly = self._runs_dir / date_str / "weekly-report"
        if run_type == "daily" and legacy_daily.is_dir():
            seen[legacy_daily] = legacy_daily
        if run_type == "weekly" and legacy_weekly.is_dir():
            seen[legacy_weekly] = legacy_weekly

        return list(seen.values())

    def _parse_run_predictions(self, run_dir: Path, source: str) -> list[PredictionOutcome]:
        """Parse structured predictions first, then legacy suggestion/warning output."""
        predictions: list[PredictionOutcome] = []
        parsed_path = run_dir / "parsed_analysis.json"
        if parsed_path.exists():
            try:
                data = json.loads(parsed_path.read_text(encoding="utf-8"))
                predictions.extend(self._parse_structured_predictions(data, source))
                predictions.extend(self._parse_legacy_prediction_lists(data, source))
            except (json.JSONDecodeError, OSError):
                pass

        for output_file in sorted(run_dir.glob("*.json")):
            if output_file.name == "parsed_analysis.json":
                continue
            try:
                data = json.loads(output_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            predictions.extend(self._parse_structured_predictions(data, source))
            predictions.extend(self._parse_legacy_prediction_lists(data, source))

        return predictions

    def _parse_structured_predictions(
        self, data: dict, source: str,
    ) -> list[PredictionOutcome]:
        """Parse structured metric predictions from parsed agent output."""
        items = data.get("predictions", [])
        if not isinstance(items, list):
            return []

        parsed: list[PredictionOutcome] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            bot_id = str(item.get("bot_id", "") or "")
            metric = str(item.get("metric", "") or "")
            direction = str(item.get("direction", "") or "")
            reasoning = str(item.get("reasoning", "") or "").strip()
            if not metric:
                continue
            description = f"{bot_id} {metric} {direction}".strip()
            if reasoning:
                description = f"{description}: {reasoning}".strip(": ")
            parsed.append(
                PredictionOutcome(
                    prediction_source=source,
                    prediction_text=description[:500],
                    prediction_type="metric_prediction",
                    bot_id=bot_id,
                    metric=metric,
                    predicted_direction=direction,
                )
            )
        return parsed

    def _parse_legacy_prediction_lists(
        self, data: dict, source: str,
    ) -> list[PredictionOutcome]:
        """Parse older suggestion/warning list-style outputs."""
        parsed: list[PredictionOutcome] = []
        if not isinstance(data, dict):
            return parsed

        for suggestion in data.get("suggestions", []):
            text = self._stringify_prediction_text(suggestion)
            if text:
                parsed.append(
                    PredictionOutcome(
                        prediction_source=source,
                        prediction_text=text[:500],
                        prediction_type="suggestion",
                    )
                )

        for warning in data.get("warnings", data.get("risk_warnings", [])):
            text = self._stringify_prediction_text(warning)
            if text:
                parsed.append(
                    PredictionOutcome(
                        prediction_source=source,
                        prediction_text=text[:500],
                        prediction_type="risk_warning",
                    )
                )

        return parsed

    @staticmethod
    def _stringify_prediction_text(item: object) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            title = str(item.get("title", "") or "").strip()
            description = str(item.get("description", "") or "").strip()
            combined = " ".join(part for part in [title, description] if part)
            return combined or json.dumps(item, sort_keys=True)
        return str(item)

    def _load_actual_outcomes(self, start: datetime, end: datetime) -> dict[str, list[dict]]:
        """Load daily curated summaries and supporting analysis files."""
        outcomes: dict[str, list[dict]] = {}
        extra_files = [
            "exit_efficiency.json",
            "filter_analysis.json",
            "regime_analysis.json",
            "hourly_performance.json",
            "orderbook_stats.json",
            "signal_health.json",
        ]

        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            date_dir = self._curated_dir / date_str
            if date_dir.is_dir():
                for bot_dir in date_dir.iterdir():
                    if not bot_dir.is_dir():
                        continue
                    summary_file = bot_dir / "summary.json"
                    if not summary_file.exists():
                        continue
                    try:
                        data = json.loads(summary_file.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        continue
                    if not isinstance(data, dict):
                        continue
                    data.setdefault("bot_id", bot_dir.name)
                    for extra_name in extra_files:
                        extra_path = bot_dir / extra_name
                        if not extra_path.exists():
                            continue
                        try:
                            extra_data = json.loads(extra_path.read_text(encoding="utf-8"))
                        except (json.JSONDecodeError, OSError):
                            continue
                        data[extra_name.removesuffix(".json")] = extra_data
                    outcomes.setdefault(date_str, []).append(data)
            current += timedelta(days=1)

        return outcomes

    def _describe_metric_outcome(
        self,
        prediction: PredictionOutcome,
        prior_outcomes: dict[str, list[dict]],
        current_outcomes: dict[str, list[dict]],
    ) -> str:
        """Describe the realized metric values for a structured prediction."""
        current_value = self._extract_metric_average(
            current_outcomes, prediction.metric, prediction.bot_id,
        )
        if current_value is None:
            return "No outcome data available for comparison"

        current_metric = self._canonical_metric_name(prediction.metric)
        prior_value = self._extract_metric_average(
            prior_outcomes, prediction.metric, prediction.bot_id,
        )
        if prior_value is None:
            return (
                f"{prediction.bot_id or 'portfolio'} {current_metric}="
                f"{current_value:.4f} (no prior baseline)"
            )
        return (
            f"{prediction.bot_id or 'portfolio'} {current_metric}: "
            f"{prior_value:.4f} -> {current_value:.4f}"
        )

    def _find_matching_outcome(
        self, prediction: PredictionOutcome, outcomes: dict[str, list[dict]],
    ) -> str:
        """Find matching realized evidence for a legacy suggestion or warning."""
        if not outcomes:
            return "No outcome data available for comparison"

        text_lower = prediction.prediction_text.lower()
        for keyword, matcher in _SUGGESTION_MATCHERS.items():
            if keyword not in text_lower:
                continue
            metric_keys = matcher["metrics"]
            if not isinstance(metric_keys, list):
                continue
            metric_values = self._extract_metric_values(outcomes, metric_keys)
            if metric_values:
                parts = [f"Matched metrics for '{keyword}':"]
                for metric_name, value in metric_values.items():
                    parts.append(f"{metric_name}={value:.4f}")
                return " ".join(parts)

        total_pnl = 0.0
        total_trades = 0
        for date_outcomes in outcomes.values():
            for outcome in date_outcomes:
                total_pnl += float(outcome.get("net_pnl", 0.0) or 0.0)
                total_trades += int(outcome.get("total_trades", 0) or 0)
        return f"Week total: {total_trades} trades, ${total_pnl:.2f} net PnL"

    def _extract_metric_values(
        self,
        outcomes: dict[str, list[dict]],
        metric_keys: list[str],
        bot_id: str = "",
    ) -> dict[str, float]:
        """Extract average values for the requested metrics."""
        values: dict[str, float] = {}
        for metric_key in metric_keys:
            value = self._extract_metric_average(outcomes, metric_key, bot_id)
            if value is not None:
                values[self._canonical_metric_name(metric_key)] = value
        return values

    def _extract_metric_average(
        self,
        outcomes: dict[str, list[dict]],
        metric: str,
        bot_id: str = "",
    ) -> float | None:
        """Compute an average metric value across matching outcome entries."""
        samples: list[float] = []
        for date_outcomes in outcomes.values():
            for outcome in date_outcomes:
                if bot_id and outcome.get("bot_id") != bot_id:
                    continue
                value = self._extract_metric_from_entry(outcome, metric)
                if value is not None:
                    samples.append(value)
        if not samples:
            return None
        return sum(samples) / len(samples)

    def _extract_metric_from_entry(self, outcome: dict, metric: str) -> float | None:
        """Extract a metric from a summary entry or nested analysis object."""
        aliases = _METRIC_ALIASES.get(metric, [metric])
        for alias in aliases:
            value = outcome.get(alias)
            if isinstance(value, (int, float)):
                return float(value)

        for nested in outcome.values():
            if not isinstance(nested, dict):
                continue
            for alias in aliases:
                value = nested.get(alias)
                if isinstance(value, (int, float)):
                    return float(value)

        if metric == "hourly_performance":
            hourly = outcome.get("hourly_performance")
            if isinstance(hourly, dict):
                buckets = hourly.get("buckets", [])
                pnl_values = [
                    float(bucket.get("pnl", 0.0))
                    for bucket in buckets
                    if isinstance(bucket, dict)
                ]
                if pnl_values:
                    return sum(pnl_values) / len(pnl_values)

        if metric == "regime_pnl":
            regime = outcome.get("regime_analysis")
            if isinstance(regime, dict):
                pnl_by_regime = regime.get("regime_pnl", {})
                if isinstance(pnl_by_regime, dict) and pnl_by_regime:
                    numeric = [
                        float(value) for value in pnl_by_regime.values()
                        if isinstance(value, (int, float))
                    ]
                    if numeric:
                        return sum(numeric) / len(numeric)

        return None

    def _assess_accuracy(
        self,
        prediction: PredictionOutcome,
        prior_outcomes: dict[str, list[dict]] | None = None,
        current_outcomes: dict[str, list[dict]] | None = None,
    ) -> str:
        """Assess whether a prediction aligned with realized outcomes."""
        if not prediction.actual_outcome or "No outcome data" in prediction.actual_outcome:
            return "unverifiable"

        if prediction.metric:
            return self._assess_metric_prediction(
                prediction, prior_outcomes or {}, current_outcomes or {},
            )

        if not prediction.actual_outcome.startswith("Matched metrics"):
            return "unverifiable"

        text_lower = prediction.prediction_text.lower()
        matched_keyword = next(
            (keyword for keyword in _SUGGESTION_MATCHERS if keyword in text_lower),
            None,
        )
        if matched_keyword is None:
            return "unverifiable"

        matcher = _SUGGESTION_MATCHERS[matched_keyword]
        metric_keys = matcher["metrics"]
        direction = matcher["positive_direction"]
        if not isinstance(metric_keys, list) or not isinstance(direction, str):
            return "unverifiable"

        if prior_outcomes and current_outcomes:
            prior_values = self._extract_metric_values(prior_outcomes, metric_keys)
            current_values = self._extract_metric_values(current_outcomes, metric_keys)
            if prior_values and current_values:
                return self._classify_by_delta(direction, prior_values, current_values)

        metric_values = self._parse_outcome_metrics(prediction.actual_outcome)
        if not metric_values:
            return "unverifiable"
        return self._classify_by_direction(direction, metric_values)

    def _assess_metric_prediction(
        self,
        prediction: PredictionOutcome,
        prior_outcomes: dict[str, list[dict]],
        current_outcomes: dict[str, list[dict]],
    ) -> str:
        """Assess a structured metric prediction against prior/current outcomes."""
        if not prediction.metric or not prediction.predicted_direction:
            return "unverifiable"

        current_value = self._extract_metric_average(
            current_outcomes, prediction.metric, prediction.bot_id,
        )
        if current_value is None:
            return "unverifiable"

        prior_value = self._extract_metric_average(
            prior_outcomes, prediction.metric, prediction.bot_id,
        )
        if prior_value is None:
            return self._classify_metric_absolute(
                prediction.metric, prediction.predicted_direction, current_value,
            )

        actual_direction = self._infer_metric_direction(
            prediction.metric, prior_value, current_value,
        )
        if actual_direction == "stable":
            return "correct" if prediction.predicted_direction == "stable" else "partially_correct"
        if prediction.predicted_direction == actual_direction:
            return "correct"
        if prediction.predicted_direction == "stable":
            return "incorrect"
        return "incorrect"

    @staticmethod
    def _parse_outcome_metrics(outcome_str: str) -> dict[str, float]:
        """Parse metric name=value pairs from an outcome description."""
        import re

        metrics: dict[str, float] = {}
        for match in re.finditer(r"(\w+)=([-+]?\d*\.?\d+)", outcome_str):
            metrics[match.group(1)] = float(match.group(2))
        return metrics

    @staticmethod
    def _classify_by_direction(direction: str, metrics: dict[str, float]) -> str:
        """Fallback heuristic for legacy suggestion accuracy."""
        values = list(metrics.values())
        if not values:
            return "unverifiable"

        primary = values[0]
        if direction == "decrease_mae_or_increase_efficiency":
            if primary > 0.55:
                return "partially_correct"
            if primary < 0.3:
                return "incorrect"
            return "unverifiable"

        if direction == "decrease_missed_winners":
            if primary <= 2:
                return "partially_correct"
            if primary >= 10:
                return "incorrect"
            return "unverifiable"

        if direction in {"decrease", "loss_win_ratio_decrease"}:
            if primary < 0.95:
                return "partially_correct"
            if primary > 1.05:
                return "incorrect"
            return "unverifiable"

        if direction == "increase":
            if primary > 0:
                return "partially_correct"
            if primary < 0:
                return "incorrect"
            return "unverifiable"

        if direction in {"improvement_in_flagged_regime", "improvement_in_flagged_hours"}:
            if primary > 0:
                return "partially_correct"
            if primary < 0:
                return "incorrect"
            return "unverifiable"

        return "unverifiable"

    @staticmethod
    def _classify_by_delta(
        direction: str,
        prior: dict[str, float],
        current: dict[str, float],
    ) -> str:
        """Classify legacy keyword matches using prior/current metric deltas."""
        common_keys = sorted(set(prior) & set(current))
        if not common_keys:
            return "unverifiable"

        key = common_keys[0]
        old_value = prior[key]
        new_value = current[key]

        if old_value == 0.0:
            if new_value == 0.0:
                return "unverifiable"
            delta_pct = 100.0 if new_value > 0 else -100.0
        else:
            delta_pct = (new_value - old_value) / abs(old_value) * 100.0

        if direction in {"decrease", "decrease_missed_winners", "loss_win_ratio_decrease"}:
            delta_pct = -delta_pct
        elif direction not in {
            "increase",
            "improvement_in_flagged_regime",
            "improvement_in_flagged_hours",
            "decrease_mae_or_increase_efficiency",
        }:
            return "unverifiable"

        if delta_pct >= 5.0:
            return "correct"
        if delta_pct > 0.0:
            return "partially_correct"
        if delta_pct < -5.0:
            return "incorrect"
        return "unverifiable"

    def _infer_metric_direction(
        self, metric: str, prior_value: float, current_value: float,
    ) -> str:
        """Infer improve/decline/stable direction for a structured prediction metric."""
        epsilon = self._metric_stable_threshold(metric, prior_value, current_value)
        delta = current_value - prior_value
        if abs(delta) <= epsilon:
            return "stable"

        normalized_metric = self._normalize_metric(metric)
        if normalized_metric in _DECREASING_METRICS:
            return "improve" if delta < 0 else "decline"
        return "improve" if delta > 0 else "decline"

    @staticmethod
    def _metric_stable_threshold(metric: str, prior_value: float, current_value: float) -> float:
        baseline = max(abs(prior_value), abs(current_value), 1.0)
        normalized_metric = RetrospectiveBuilder._normalize_metric(metric)
        if normalized_metric == "win_rate":
            return 0.02
        if normalized_metric in {"sharpe", "sharpe_rolling_30d"}:
            return 0.1
        return baseline * 0.05

    def _classify_metric_absolute(
        self, metric: str, predicted_direction: str, current_value: float,
    ) -> str:
        """Compatibility fallback when no prior baseline exists."""
        normalized_metric = self._normalize_metric(metric)

        if normalized_metric in _DECREASING_METRICS:
            if predicted_direction == "improve":
                return "partially_correct" if current_value <= 2.0 else "incorrect"
            if predicted_direction == "decline":
                return "partially_correct" if current_value >= 5.0 else "incorrect"
            return "unverifiable"

        if normalized_metric in {"pnl", "net_pnl", "sharpe", "sharpe_rolling_30d"}:
            if predicted_direction == "improve":
                return "partially_correct" if current_value > 0 else "incorrect"
            if predicted_direction == "decline":
                return "partially_correct" if current_value < 0 else "incorrect"
            return "unverifiable"

        if normalized_metric == "win_rate":
            if predicted_direction == "improve":
                return "partially_correct" if current_value >= 0.5 else "incorrect"
            if predicted_direction == "decline":
                return "partially_correct" if current_value < 0.5 else "incorrect"
            return "unverifiable"

        return "unverifiable"

    @staticmethod
    def _normalize_metric(metric: str) -> str:
        for canonical, aliases in _METRIC_ALIASES.items():
            if metric == canonical or metric in aliases:
                return canonical
        return metric

    @staticmethod
    def _canonical_metric_name(metric: str) -> str:
        aliases = _METRIC_ALIASES.get(metric)
        if aliases:
            return aliases[0]
        normalized = RetrospectiveBuilder._normalize_metric(metric)
        aliases = _METRIC_ALIASES.get(normalized)
        if aliases:
            return aliases[0]
        return metric

    @staticmethod
    def _build_summary(
        correct: int,
        partial: int,
        incorrect: int,
        unverifiable: int,
        total: int,
    ) -> str:
        if total == 0:
            return "No predictions to review this week."
        return (
            f"Reviewed {total} predictions: "
            f"{correct} correct, {partial} partially correct, "
            f"{incorrect} incorrect, {unverifiable} unverifiable. "
            f"Ask Claude to assess accuracy of unverifiable predictions "
            f"given the actual outcome data."
        )
