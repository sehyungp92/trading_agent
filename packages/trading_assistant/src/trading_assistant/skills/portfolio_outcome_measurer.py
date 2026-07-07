# skills/portfolio_outcome_measurer.py
"""Portfolio outcome measurer — measures impact of deployed portfolio changes.

For DEPLOYED portfolio suggestions: measures portfolio composite before vs. after.
Observation window: 30 days minimum (vs. 7 for strategy-level).
Regime-conditional: significant regime shift during window → INCONCLUSIVE.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_PORTFOLIO_BOT_ID = "PORTFOLIO"
_MIN_OBSERVATION_DAYS = 30
_EMERGENCY_DROP_THRESHOLD = 0.10  # 10% composite drop triggers emergency


class PortfolioOutcomeMeasurer:
    """Measures outcomes of deployed portfolio-level suggestions."""

    def __init__(self, findings_dir: Path, curated_dir: Path) -> None:
        self._findings_dir = findings_dir
        self._curated_dir = curated_dir
        self._outcomes_path = findings_dir / "portfolio_outcomes.jsonl"
        self._ground_truth_path = findings_dir / "portfolio_ground_truth.jsonl"

    def measure_deployed(self) -> list[dict]:
        """Check all DEPLOYED portfolio suggestions and measure outcomes.

        Returns list of measured outcome dicts.
        """
        suggestions = self._load_deployed_portfolio_suggestions()
        if not suggestions:
            return []

        # Skip suggestions already measured (idempotency guard)
        already_measured = self._load_measured_ids()

        outcomes: list[dict] = []
        now = datetime.now(timezone.utc)

        for suggestion in suggestions:
            if suggestion.get("suggestion_id", "") in already_measured:
                continue
            deployed_at = suggestion.get("deployed_at", "")
            if not deployed_at:
                continue

            try:
                deploy_date = datetime.fromisoformat(deployed_at)
                if deploy_date.tzinfo is None:
                    deploy_date = deploy_date.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            days_since = (now - deploy_date).days
            if days_since < _MIN_OBSERVATION_DAYS:
                continue  # Not enough observation time

            # Get ground truth before and after deployment
            deploy_date_str = deploy_date.strftime("%Y-%m-%d")
            before = self._get_ground_truth_near(deploy_date_str, before=True)
            after = self._get_ground_truth_near(
                (deploy_date + timedelta(days=_MIN_OBSERVATION_DAYS)).strftime("%Y-%m-%d"),
                before=False,
            )

            if before is None or after is None:
                continue

            before_composite = before.get("composite_score", 0.5)
            after_composite = after.get("composite_score", 0.5)
            delta = after_composite - before_composite

            # Determine verdict (lowercase to match SuggestionScorer._is_positive)
            if abs(delta) < 0.02:
                verdict = "inconclusive"
            elif delta > 0:
                verdict = "positive"
            else:
                verdict = "negative"

            outcome = {
                "suggestion_id": suggestion.get("suggestion_id", ""),
                "bot_id": _PORTFOLIO_BOT_ID,
                "category": suggestion.get("category", ""),
                "deployed_at": deployed_at,
                "measured_at": now.isoformat(),
                "observation_days": days_since,
                "before_composite": round(before_composite, 4),
                "after_composite": round(after_composite, 4),
                "composite_delta": round(delta, 4),
                "verdict": verdict,
                "measurement_quality": "high" if days_since >= _MIN_OBSERVATION_DAYS * 1.5 else "medium",
                "outcome_source": "early_warning",
            }

            # Compare what-if predictions against actual outcome
            prediction_check = self._check_what_if_accuracy(suggestion, delta)
            if prediction_check:
                outcome["what_if_accuracy"] = prediction_check

            outcomes.append(outcome)
            self._write_outcome(outcome)

        return outcomes

    def check_emergency(self) -> dict | None:
        """Check if any recently deployed portfolio change caused an emergency drop.

        Returns emergency dict if composite dropped > 10% within 14 days, else None.
        """
        suggestions = self._load_deployed_portfolio_suggestions()
        now = datetime.now(timezone.utc)

        for suggestion in suggestions:
            deployed_at = suggestion.get("deployed_at", "")
            if not deployed_at:
                continue

            try:
                deploy_date = datetime.fromisoformat(deployed_at)
                if deploy_date.tzinfo is None:
                    deploy_date = deploy_date.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            days_since = (now - deploy_date).days
            if days_since > 14 or days_since < 3:
                continue  # Only check 3-14 day window

            deploy_date_str = deploy_date.strftime("%Y-%m-%d")
            before = self._get_ground_truth_near(deploy_date_str, before=True)
            current = self._get_ground_truth_near(
                now.strftime("%Y-%m-%d"), before=True,
            )

            if before is None or current is None:
                continue

            before_composite = before.get("composite_score", 0.5)
            current_composite = current.get("composite_score", 0.5)
            drop = before_composite - current_composite

            if drop > _EMERGENCY_DROP_THRESHOLD:
                return {
                    "suggestion_id": suggestion.get("suggestion_id", ""),
                    "category": suggestion.get("category", ""),
                    "deployed_at": deployed_at,
                    "before_composite": round(before_composite, 4),
                    "current_composite": round(current_composite, 4),
                    "drop": round(drop, 4),
                    "days_since_deployment": days_since,
                    "previous_config": suggestion.get("detection_context", {}).get(
                        "current_config", {}
                    ),
                }

        return None

    @staticmethod
    def _check_what_if_accuracy(
        suggestion: dict, actual_composite_delta: float,
    ) -> dict | None:
        """Compare what-if predictions against actual outcome.

        Extracts the what_if_result from the suggestion's detection_context
        and checks whether predicted direction matched actual direction for
        each metric. Returns accuracy dict or None if no predictions.
        """
        detection_ctx = suggestion.get("detection_context", {})
        if not detection_ctx:
            return None
        what_if = detection_ctx.get("what_if_result")
        if not what_if:
            return None

        result: dict = {}

        # Track which mode was used (trade-level produces sortino/profit_factor)
        has_enriched = "current_sortino" in what_if
        result["mode"] = "trade_level" if has_enriched else "daily_aggregate"

        # Check direction accuracy for each predicted delta
        # inverted=True means negative delta = improvement (e.g. drawdown decreasing)
        delta_keys = [
            ("calmar_delta", "calmar", False),
            ("sharpe_delta", "sharpe", False),
            ("drawdown_delta", "drawdown", True),
        ]
        if has_enriched:
            delta_keys.extend([
                ("sortino_delta", "sortino", False),
                ("profit_factor_delta", "profit_factor", False),
            ])

        correct = 0
        total = 0
        for key, label, inverted in delta_keys:
            predicted = what_if.get(key)
            if predicted is None:
                continue
            total += 1
            # Normalize direction: for inverted metrics, flip sign
            adj_predicted = -predicted if inverted else predicted
            predicted_dir = 0 if abs(adj_predicted) < 1e-6 else (1 if adj_predicted > 0 else -1)
            actual_dir = 0 if abs(actual_composite_delta) < 0.02 else (
                1 if actual_composite_delta > 0 else -1
            )
            match = predicted_dir == actual_dir or predicted_dir == 0 or actual_dir == 0
            result[f"{label}_predicted"] = round(predicted, 4)
            result[f"{label}_direction_match"] = match
            if match:
                correct += 1

        if total > 0:
            result["direction_accuracy"] = round(correct / total, 2)

        return result if total > 0 else None

    def _load_measured_ids(self) -> set[str]:
        """Load suggestion IDs that already have outcomes recorded."""
        if not self._outcomes_path.exists():
            return set()
        ids: set[str] = set()
        try:
            for line in self._outcomes_path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    try:
                        ids.add(json.loads(line).get("suggestion_id", ""))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
        ids.discard("")
        return ids

    def _load_deployed_portfolio_suggestions(self) -> list[dict]:
        """Load DEPLOYED suggestions with bot_id=PORTFOLIO."""
        path = self._findings_dir / "suggestions.jsonl"
        if not path.exists():
            return []
        result: list[dict] = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if (
                    rec.get("bot_id") == _PORTFOLIO_BOT_ID
                    and rec.get("status") == "deployed"
                ):
                    result.append(rec)
            except json.JSONDecodeError:
                continue
        return result

    def _get_ground_truth_near(self, date_str: str, before: bool = True) -> dict | None:
        """Find ground truth snapshot closest to the given date."""
        if not self._ground_truth_path.exists():
            return None
        entries: list[dict] = []
        for line in self._ground_truth_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not entries:
            return None

        # Find closest entry to the target date
        target = datetime.strptime(date_str, "%Y-%m-%d")
        best = None
        best_dist = float("inf")
        for entry in entries:
            entry_date_str = entry.get("snapshot_date", "")
            if not entry_date_str:
                continue
            try:
                entry_date = datetime.strptime(entry_date_str, "%Y-%m-%d")
            except ValueError:
                continue
            dist = (target - entry_date).days if before else (entry_date - target).days
            if 0 <= dist < best_dist:
                best = entry
                best_dist = dist
        return best

    def _write_outcome(self, outcome: dict) -> None:
        """Append outcome to portfolio_outcomes.jsonl."""
        self._findings_dir.mkdir(parents=True, exist_ok=True)
        with open(self._outcomes_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(outcome, default=str) + "\n")
        try:
            from trading_assistant.skills.performance_learning_ledger import (
                PerformanceLearningRefreshMarkerError,
                refresh_performance_learning_projection,
            )

            refresh_performance_learning_projection(self._findings_dir)
        except PerformanceLearningRefreshMarkerError:
            raise
        except Exception:
            logger.warning("Failed to refresh performance-learning projection", exc_info=True)
