"""Translate monthly validation results into deployed-change verdicts."""
from __future__ import annotations

from typing import Any

from trading_assistant.schemas.monthly_outcome import (
    MonthlyOutcomeRecord,
    MonthlyOutcomeVerdict,
    OutcomeDataSufficiency,
    OutcomeSource,
)
from trading_assistant.schemas.monthly_validation import MonthlyValidationResult, MonthlyValidationStatus


_STATUS_TO_VERDICT = {
    MonthlyValidationStatus.KEEP: MonthlyOutcomeVerdict.KEEP,
    MonthlyValidationStatus.NO_CHANGE: MonthlyOutcomeVerdict.KEEP,
    MonthlyValidationStatus.WATCH: MonthlyOutcomeVerdict.WATCH,
    MonthlyValidationStatus.REPAIR: MonthlyOutcomeVerdict.REPAIR,
    MonthlyValidationStatus.ROLLBACK: MonthlyOutcomeVerdict.ROLLBACK,
    MonthlyValidationStatus.QUARANTINE: MonthlyOutcomeVerdict.QUARANTINE,
}


class MonthlyOutcomeScorer:
    """Builds source-aware outcome records from monthly validation artifacts."""

    def score_validation_result(
        self,
        result: MonthlyValidationResult,
        *,
        strategy_change_record_id: str = "",
        deployment_id: str = "",
        config_version: str = "",
        strategy_version: str = "",
        commit_sha: str = "",
        proposal_ids: list[str] | None = None,
        suggestion_ids: list[str] | None = None,
        mutation_family: str = "",
        category: str = "",
        objective_deltas: dict[str, float] | None = None,
        minimum_trade_count_met: bool = False,
        source: OutcomeSource = OutcomeSource.MONTHLY,
        workflow: str = "monthly_validation",
        source_provider: str = "",
        source_model: str = "",
    ) -> MonthlyOutcomeRecord:
        deltas = objective_deltas or {}
        verdict = _STATUS_TO_VERDICT.get(result.status, MonthlyOutcomeVerdict.INCONCLUSIVE)
        data_sufficiency = self._data_sufficiency(result)
        confidence = self._confidence(result, data_sufficiency)
        return MonthlyOutcomeRecord(
            source=source,
            bot_id=result.bot_id,
            strategy_id=result.strategy_id,
            run_id=result.run_id,
            run_month=result.run_month,
            workflow=workflow,
            source_provider=source_provider or result.model_review_provider,
            source_model=source_model or result.model_review_model,
            strategy_change_record_id=strategy_change_record_id or result.strategy_change_record_id,
            deployment_id=deployment_id,
            config_version=config_version,
            strategy_version=strategy_version,
            commit_sha=commit_sha,
            proposal_ids=proposal_ids or [],
            suggestion_ids=suggestion_ids or [],
            mutation_family=mutation_family,
            category=category,
            verdict=verdict,
            live_vs_expected_objective_delta=float(deltas.get("live_vs_expected", deltas.get("objective", 0.0))),
            trade_frequency_delta=float(deltas.get("trade_frequency", 0.0)),
            drawdown_delta=float(deltas.get("drawdown", 0.0)),
            execution_slippage_delta=float(deltas.get("execution_slippage", 0.0)),
            objective_deltas={str(k): float(v) for k, v in deltas.items()},
            gap_attribution=self._gap_payload(result),
            confidence=confidence,
            data_sufficiency=data_sufficiency,
            recommended_next_action=self._next_action(verdict),
            evidence_paths=list(result.evidence_paths),
            minimum_trade_count_met=minimum_trade_count_met,
            persistence_confirmed=source == OutcomeSource.FOLLOW_UP and verdict == MonthlyOutcomeVerdict.KEEP,
            objective_version=result.objective_version,
        )

    @staticmethod
    def _data_sufficiency(result: MonthlyValidationResult) -> OutcomeDataSufficiency:
        if result.status in {
            MonthlyValidationStatus.INSUFFICIENT_DATA,
            MonthlyValidationStatus.INSUFFICIENT_LINEAGE,
            MonthlyValidationStatus.UNSUPPORTED_NO_REPLAY_PLUGIN,
        }:
            return OutcomeDataSufficiency.INSUFFICIENT
        if result.blocking_reasons:
            return OutcomeDataSufficiency.SPARSE
        return OutcomeDataSufficiency.SUFFICIENT

    @staticmethod
    def _confidence(
        result: MonthlyValidationResult,
        data_sufficiency: OutcomeDataSufficiency,
    ) -> float:
        if data_sufficiency == OutcomeDataSufficiency.INSUFFICIENT:
            return 0.0
        gap_confidence = float(getattr(result.gap_attribution, "confidence", 0.0) or 0.0)
        if gap_confidence > 0:
            return min(1.0, max(0.0, gap_confidence))
        if result.status in {MonthlyValidationStatus.KEEP, MonthlyValidationStatus.NO_CHANGE}:
            return 0.7
        if result.status in {
            MonthlyValidationStatus.REPAIR,
            MonthlyValidationStatus.ROLLBACK,
            MonthlyValidationStatus.QUARANTINE,
        }:
            return 0.75
        return 0.5

    @staticmethod
    def _gap_payload(result: MonthlyValidationResult) -> dict[str, Any]:
        try:
            return result.gap_attribution.model_dump(mode="json")
        except Exception:
            return {}

    @staticmethod
    def _next_action(verdict: MonthlyOutcomeVerdict) -> str:
        return {
            MonthlyOutcomeVerdict.KEEP: "keep deployed change; wait for follow-up confirmation before strengthening priors",
            MonthlyOutcomeVerdict.WATCH: "watch next monthly window and avoid compounding similar changes",
            MonthlyOutcomeVerdict.REPAIR: "route targeted repair through monthly candidate generation",
            MonthlyOutcomeVerdict.ROLLBACK: "recommend approval-gated rollback",
            MonthlyOutcomeVerdict.QUARANTINE: "quarantine interacting mutation family pending review",
            MonthlyOutcomeVerdict.INCONCLUSIVE: "do not update positive priors; collect more evidence",
        }[verdict]
