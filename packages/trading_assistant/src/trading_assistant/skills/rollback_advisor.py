"""Approval-gated rollback recommendations from outcome evidence."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from trading_assistant.schemas.monthly_outcome import MonthlyOutcomeRecord, MonthlyOutcomeVerdict
from trading_assistant.schemas.rollback_thresholds import (
    RollbackRecommendation,
    RollbackRecommendationAction,
    RollbackThresholds,
)


class RollbackAdvisor:
    """Emits auditable recommendations; it never performs rollback actions."""

    def __init__(self, thresholds: RollbackThresholds | None = None) -> None:
        self.thresholds = thresholds or RollbackThresholds()

    def recommend(self, outcome: MonthlyOutcomeRecord) -> RollbackRecommendation | None:
        reasons: list[str] = []
        action: RollbackRecommendationAction | None = None

        if outcome.confidence < self.thresholds.min_confidence:
            if outcome.verdict == MonthlyOutcomeVerdict.WATCH:
                action = RollbackRecommendationAction.WATCH
                reasons.append("monthly verdict is watch")
            else:
                return None

        if outcome.verdict == MonthlyOutcomeVerdict.QUARANTINE:
            action = RollbackRecommendationAction.QUARANTINE
            reasons.append("monthly verdict is quarantine")
        elif outcome.verdict == MonthlyOutcomeVerdict.ROLLBACK:
            action = RollbackRecommendationAction.ROLLBACK
            reasons.append("monthly verdict is rollback")
        elif outcome.verdict == MonthlyOutcomeVerdict.REPAIR:
            action = RollbackRecommendationAction.REPAIR
            reasons.append("monthly verdict is repair")

        if outcome.live_vs_expected_objective_delta <= self.thresholds.rollback_objective_delta:
            action = RollbackRecommendationAction.ROLLBACK
            reasons.append("objective degradation crossed rollback threshold")
        elif outcome.live_vs_expected_objective_delta <= self.thresholds.repair_objective_delta:
            action = action or RollbackRecommendationAction.REPAIR
            reasons.append("objective degradation crossed repair threshold")

        if outcome.drawdown_delta >= self.thresholds.rollback_drawdown_delta:
            action = RollbackRecommendationAction.ROLLBACK
            reasons.append("drawdown degradation crossed rollback threshold")
        elif outcome.drawdown_delta >= self.thresholds.repair_drawdown_delta:
            action = action or RollbackRecommendationAction.REPAIR
            reasons.append("drawdown degradation crossed repair threshold")

        if outcome.execution_slippage_delta >= self.thresholds.rollback_execution_slippage_delta:
            action = RollbackRecommendationAction.ROLLBACK
            reasons.append("execution slippage crossed rollback threshold")
        elif outcome.execution_slippage_delta >= self.thresholds.repair_execution_slippage_delta:
            action = action or RollbackRecommendationAction.REPAIR
            reasons.append("execution slippage crossed repair threshold")

        if action is None:
            return None

        recommendation = RollbackRecommendation(
            bot_id=outcome.bot_id,
            strategy_id=outcome.strategy_id,
            strategy_change_record_id=outcome.strategy_change_record_id,
            deployment_id=outcome.deployment_id,
            action=action,
            reason="; ".join(dict.fromkeys(reasons)),
            confidence=outcome.confidence,
            evidence_paths=outcome.evidence_paths,
            source_outcome_id=outcome.outcome_id,
            policy_version=self.thresholds.policy_version,
        )
        recommendation.recommendation_id = hashlib.sha256(
            f"{outcome.outcome_id}:{action.value}:{recommendation.reason}".encode("utf-8")
        ).hexdigest()[:16]
        return recommendation

    @classmethod
    def from_policy_file(cls, path: Path) -> "RollbackAdvisor":
        """Load the simple YAML policy committed under memory/policies."""
        data = _read_simple_yaml(path)
        return cls(RollbackThresholds.model_validate(data))


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        text = value.strip().strip('"').strip("'")
        if text.lower() in {"true", "false"}:
            data[key.strip()] = text.lower() == "true"
            continue
        try:
            data[key.strip()] = int(text)
            continue
        except ValueError:
            pass
        try:
            data[key.strip()] = float(text)
            continue
        except ValueError:
            pass
        data[key.strip()] = text
    return data
