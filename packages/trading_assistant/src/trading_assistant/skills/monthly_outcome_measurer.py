"""Authoritative monthly outcome persistence and ledger fan-out."""
from __future__ import annotations

import json
from pathlib import Path

from trading_assistant.schemas.monthly_outcome import (
    FollowUpOutcomeSchedule,
    FollowUpTrigger,
    MonthlyOutcomeRecord,
    OutcomeSource,
)
from trading_assistant.schemas.proposal_ledger import ProposalOutcome
from trading_assistant.schemas.suggestion_tracking import SuggestionOutcome
from trading_assistant.skills.monthly_outcome_scorer import MonthlyOutcomeScorer
from trading_assistant.skills.outcome_prior_store import OutcomePriorStore
from trading_assistant.skills.proposal_ledger import ProposalLedger
from trading_assistant.skills.rollback_advisor import RollbackAdvisor
from trading_assistant.skills.strategy_change_ledger import StrategyChangeLedger
from trading_assistant.skills.suggestion_tracker import SuggestionTracker


class MonthlyOutcomeMeasurer:
    """Writes monthly/follow-up verdicts to all learning-loop ledgers."""

    def __init__(
        self,
        findings_dir: Path,
        *,
        suggestion_tracker: SuggestionTracker | None = None,
        proposal_ledger: ProposalLedger | None = None,
        strategy_change_ledger: StrategyChangeLedger | None = None,
        prior_store: OutcomePriorStore | None = None,
        rollback_advisor: RollbackAdvisor | None = None,
    ) -> None:
        self.findings_dir = Path(findings_dir)
        self.path = self.findings_dir / "monthly_outcomes.jsonl"
        self.followups_path = self.findings_dir / "monthly_outcome_followups.jsonl"
        self.suggestion_tracker = suggestion_tracker or SuggestionTracker(self.findings_dir)
        self.proposal_ledger = proposal_ledger or ProposalLedger(self.findings_dir)
        self.strategy_change_ledger = strategy_change_ledger or StrategyChangeLedger(self.findings_dir)
        self.prior_store = prior_store or OutcomePriorStore(self.findings_dir)
        self.rollback_advisor = rollback_advisor or self._default_rollback_advisor()
        self.scorer = MonthlyOutcomeScorer()

    def record_monthly_verdict(self, outcome: MonthlyOutcomeRecord) -> MonthlyOutcomeRecord:
        existing_ids = {row.get("outcome_id") for row in self._read_jsonl(self.path)}
        if outcome.outcome_id not in existing_ids:
            self.findings_dir.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(outcome.model_dump_json() + "\n")

        self._update_strategy_change_ledger(outcome)
        self._update_suggestion_tracker(outcome)
        self._update_proposal_ledger(outcome)
        self.prior_store.record_outcome(outcome)
        self._record_rollback_recommendation(outcome)
        if outcome.source == OutcomeSource.MONTHLY:
            self.schedule_follow_up(outcome)
        return outcome

    def record_from_monthly_validation(self, result, **kwargs) -> MonthlyOutcomeRecord:
        outcome = self.scorer.score_validation_result(result, **kwargs)
        return self.record_monthly_verdict(outcome)

    def schedule_follow_up(
        self,
        outcome: MonthlyOutcomeRecord,
        *,
        trigger: FollowUpTrigger = FollowUpTrigger.THREE_MONTHS,
        min_trade_count: int = 30,
    ) -> FollowUpOutcomeSchedule:
        due_month = _add_months(outcome.run_month, 3) if outcome.run_month else ""
        schedule = FollowUpOutcomeSchedule(
            outcome_id=outcome.outcome_id,
            bot_id=outcome.bot_id,
            strategy_id=outcome.strategy_id,
            strategy_change_record_id=outcome.strategy_change_record_id,
            deployment_id=outcome.deployment_id,
            trigger=trigger,
            due_run_month=due_month,
            min_trade_count=min_trade_count,
        )
        existing = self._read_jsonl(self.followups_path)
        if not any(row.get("outcome_id") == schedule.outcome_id for row in existing):
            self.findings_dir.mkdir(parents=True, exist_ok=True)
            with self.followups_path.open("a", encoding="utf-8") as f:
                f.write(schedule.model_dump_json() + "\n")
        return schedule

    def load_outcomes(self) -> list[MonthlyOutcomeRecord]:
        out: list[MonthlyOutcomeRecord] = []
        for row in self._read_jsonl(self.path):
            try:
                out.append(MonthlyOutcomeRecord.model_validate(row))
            except Exception:
                continue
        return out

    def _update_strategy_change_ledger(self, outcome: MonthlyOutcomeRecord) -> None:
        if not outcome.strategy_change_record_id:
            return
        existing = self.strategy_change_ledger.get_by_id(outcome.strategy_change_record_id)
        if existing is not None:
            current = (
                existing.follow_up_verdict
                if outcome.source == OutcomeSource.FOLLOW_UP
                else existing.monthly_verdict
            )
            if isinstance(current, dict) and current.get("outcome_id") == outcome.outcome_id:
                return
        payload = outcome.model_dump(mode="json")
        if outcome.source == OutcomeSource.FOLLOW_UP:
            self.strategy_change_ledger.record_follow_up_verdict(
                outcome.strategy_change_record_id,
                payload,
            )
        else:
            self.strategy_change_ledger.record_one_month_verdict(
                outcome.strategy_change_record_id,
                payload,
            )

    def _update_suggestion_tracker(self, outcome: MonthlyOutcomeRecord) -> None:
        existing_outcomes = self.suggestion_tracker.load_outcomes()
        for suggestion_id in outcome.suggestion_ids:
            if not any(
                row.get("suggestion_id") == suggestion_id
                and row.get("monthly_outcome_id") == outcome.outcome_id
                for row in existing_outcomes
            ):
                self.suggestion_tracker.record_outcome(SuggestionOutcome(
                    suggestion_id=suggestion_id,
                    strategy_id=outcome.strategy_id or None,
                    implemented_date=outcome.run_month + "-01" if outcome.run_month else "",
                    pnl_delta_30d=outcome.live_vs_expected_objective_delta,
                    drawdown_delta_30d=outcome.drawdown_delta,
                    outcome_source=outcome.source.value,
                    monthly_outcome_id=outcome.outcome_id,
                    strategy_change_record_id=outcome.strategy_change_record_id,
                    verdict=_proposal_verdict(outcome),
                ))
            self.suggestion_tracker.mark_measured(
                suggestion_id,
                source=outcome.source.value,
                outcome_id=outcome.outcome_id,
                strategy_change_record_id=outcome.strategy_change_record_id,
                final=True,
            )

    def _update_proposal_ledger(self, outcome: MonthlyOutcomeRecord) -> None:
        verdict = _proposal_verdict(outcome)
        for proposal_id in outcome.proposal_ids:
            existing = self.proposal_ledger.get_by_id(proposal_id)
            if existing and any(o.monthly_outcome_id == outcome.outcome_id for o in existing.outcomes):
                continue
            self.proposal_ledger.record_outcome(
                proposal_id,
                ProposalOutcome(
                    proposal_id=proposal_id,
                    deployment_id=outcome.deployment_id,
                    objective_delta=outcome.live_vs_expected_objective_delta,
                    verdict=verdict,
                    measurement_path=str(self.path),
                    outcome_source=outcome.source.value,
                    monthly_outcome_id=outcome.outcome_id,
                    strategy_change_record_id=outcome.strategy_change_record_id,
                ),
            )

    def _record_rollback_recommendation(self, outcome: MonthlyOutcomeRecord) -> None:
        recommendation = self.rollback_advisor.recommend(outcome)
        if recommendation is None:
            return
        path = self.findings_dir / "rollback_recommendations.jsonl"
        existing = self._read_jsonl(path)
        if any(row.get("recommendation_id") == recommendation.recommendation_id for row in existing):
            return
        self.findings_dir.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(recommendation.model_dump_json() + "\n")

    def _default_rollback_advisor(self) -> RollbackAdvisor:
        policy_path = self.findings_dir.parent / "policies" / "v1" / "rollback_thresholds.yaml"
        if policy_path.exists():
            return RollbackAdvisor.from_policy_file(policy_path)
        return RollbackAdvisor()

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        rows: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows


def _proposal_verdict(outcome: MonthlyOutcomeRecord) -> str:
    if outcome.verdict.value == "keep":
        return "positive"
    if outcome.verdict.value in {"repair", "rollback", "quarantine"}:
        return "negative"
    if outcome.verdict.value == "watch":
        return "neutral"
    return "inconclusive"


def _add_months(run_month: str, months: int) -> str:
    year_s, month_s = run_month.split("-", 1)
    year = int(year_s)
    month = int(month_s) + months
    while month > 12:
        year += 1
        month -= 12
    return f"{year:04d}-{month:02d}"
