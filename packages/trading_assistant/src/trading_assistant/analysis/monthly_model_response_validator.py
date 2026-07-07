"""Validation for monthly model-review output."""
from __future__ import annotations

from pathlib import Path

from trading_assistant.schemas.monthly_candidates import MonthlyRiskClassification
from trading_assistant.schemas.monthly_model_review import (
    MonthlyModelReview,
    MonthlyModelRouting,
    MonthlyModelValidationIssue,
    MonthlyModelValidationResult,
)

_ACTIONABLE_ROUTES = {
    MonthlyModelRouting.SMOKE_REPAIR.value,
    MonthlyModelRouting.PHASED_AUTO.value,
    MonthlyModelRouting.EXPERIMENT.value,
    MonthlyModelRouting.MANUAL_DESIGN_REVIEW.value,
}


class MonthlyModelResponseValidator:
    """Fail closed on model proposals that lack deterministic provenance."""

    def validate(
        self,
        review: MonthlyModelReview,
        *,
        allow_actionable: bool = True,
        allowed_evidence_paths: list[str] | None = None,
        expected_run_id: str = "",
        expected_bot_id: str = "",
        expected_strategy_id: str = "",
    ) -> MonthlyModelValidationResult:
        issues: list[MonthlyModelValidationIssue] = []
        approval_tiers: dict[str, str] = {}
        actionable: list[str] = []
        hypothesis_only: list[str] = []
        allowed = {_evidence_key(path) for path in allowed_evidence_paths or [] if str(path)}

        if not review.parse_success:
            issues.append(MonthlyModelValidationIssue(
                item_type="review",
                message="model review could not be parsed",
            ))
        if expected_run_id and review.run_id and review.run_id != expected_run_id:
            issues.append(MonthlyModelValidationIssue(
                item_type="review",
                message=f"run_id mismatch: {review.run_id} != {expected_run_id}",
            ))
        if expected_bot_id and review.bot_id and review.bot_id != expected_bot_id:
            issues.append(MonthlyModelValidationIssue(
                item_type="review",
                message=f"bot_id mismatch: {review.bot_id} != {expected_bot_id}",
            ))
        if expected_strategy_id and review.strategy_id and review.strategy_id != expected_strategy_id:
            issues.append(MonthlyModelValidationIssue(
                item_type="review",
                message=f"strategy_id mismatch: {review.strategy_id} != {expected_strategy_id}",
            ))

        for candidate in review.candidate_reviews:
            item_id = candidate.candidate_id
            route = candidate.routing.value
            if route in _ACTIONABLE_ROUTES:
                actionable.append(item_id)
                if expected_bot_id and not review.bot_id:
                    issues.append(MonthlyModelValidationIssue(
                        item_type="candidate_review",
                        item_id=item_id,
                        message="bot_id scope is required for actionable candidate review",
                    ))
                if expected_strategy_id and not review.strategy_id:
                    issues.append(MonthlyModelValidationIssue(
                        item_type="candidate_review",
                        item_id=item_id,
                        message="strategy_id scope is required for actionable candidate review",
                    ))
                _require(candidate.evidence_paths, issues, "candidate_review", item_id, "evidence_paths are required")
                _require_known_evidence(candidate.evidence_paths, allowed, issues, "candidate_review", item_id)
                _require(candidate.capability_labels, issues, "candidate_review", item_id, "capability_labels are required")
                _require_authoritative_evidence(candidate.evidence_authority, issues, "candidate_review", item_id)
                _require(candidate.expected_objective_impact, issues, "candidate_review", item_id, "expected_objective_impact is required")
                _require(candidate.replay_or_experiment_plan, issues, "candidate_review", item_id, "replay_or_experiment_plan is required")
                _require(candidate.acceptance_criteria, issues, "candidate_review", item_id, "acceptance_criteria are required")
                _require(candidate.rollback_plan, issues, "candidate_review", item_id, "rollback_plan is required")
                if not allow_actionable:
                    issues.append(MonthlyModelValidationIssue(
                        item_type="candidate_review",
                        item_id=item_id,
                        message="actionable routing is not allowed for unsupported deterministic evidence",
                    ))
                approval_tiers[item_id] = _approval_tier(candidate.risk_classification)
            else:
                hypothesis_only.append(item_id)

        for proposal in review.structural_proposals:
            item_id = proposal.hypothesis_id or proposal.linked_suggestion_id or proposal.title
            route = (proposal.routing or MonthlyModelRouting.HYPOTHESIS_ONLY.value).strip().lower()
            if route in _ACTIONABLE_ROUTES:
                actionable.append(item_id)
                if expected_strategy_id and not (proposal.affected_strategy_id or review.strategy_id):
                    issues.append(MonthlyModelValidationIssue(
                        item_type="structural_proposal",
                        item_id=item_id,
                        message="affected_strategy_id or review strategy_id is required for actionable structural proposal",
                    ))
                _require(proposal.evidence_paths, issues, "structural_proposal", item_id, "evidence_paths are required")
                _require_known_evidence(proposal.evidence_paths, allowed, issues, "structural_proposal", item_id)
                _require(proposal.capability_labels, issues, "structural_proposal", item_id, "capability_labels are required")
                _require_authoritative_evidence(proposal.evidence_authority, issues, "structural_proposal", item_id)
                _require(proposal.objective_impact, issues, "structural_proposal", item_id, "objective_impact is required")
                _require(proposal.replay_or_experiment_plan, issues, "structural_proposal", item_id, "replay_or_experiment_plan is required")
                _require(proposal.acceptance_criteria, issues, "structural_proposal", item_id, "acceptance_criteria are required")
                _require(proposal.rollback_plan, issues, "structural_proposal", item_id, "rollback_plan is required")
                if not allow_actionable:
                    issues.append(MonthlyModelValidationIssue(
                        item_type="structural_proposal",
                        item_id=item_id,
                        message="unsupported structural proposal must remain hypothesis_only",
                    ))
                approval_tiers[item_id] = _approval_tier(_risk(proposal.risk_classification))
            else:
                hypothesis_only.append(item_id)

        return MonthlyModelValidationResult(
            valid=not issues,
            issues=issues,
            approval_tiers=approval_tiers,
            actionable_candidate_ids=actionable,
            hypothesis_only_ids=hypothesis_only,
        )


def _require(value, issues: list[MonthlyModelValidationIssue], item_type: str, item_id: str, message: str) -> None:
    if value:
        return
    issues.append(MonthlyModelValidationIssue(
        item_type=item_type,
        item_id=item_id,
        message=message,
    ))


def _require_known_evidence(
    paths: list[str],
    allowed: set[str],
    issues: list[MonthlyModelValidationIssue],
    item_type: str,
    item_id: str,
) -> None:
    if not paths or not allowed:
        return
    unknown = [
        path for path in paths
        if _evidence_key(path) not in allowed
    ]
    if unknown:
        issues.append(MonthlyModelValidationIssue(
            item_type=item_type,
            item_id=item_id,
            message=f"evidence_paths are not in deterministic evidence set: {', '.join(unknown)}",
        ))


def _require_authoritative_evidence(
    value: str,
    issues: list[MonthlyModelValidationIssue],
    item_type: str,
    item_id: str,
) -> None:
    normalized = str(value or "").strip().lower()
    if normalized == "learning_authoritative":
        return
    if normalized in {"diagnostics_only", "diagnostic", "blocked", "insufficient_lineage", "insufficient_denominators", "insufficient_after_costs"}:
        message = "diagnostics-only evidence cannot be presented as approval-grade"
    else:
        message = "evidence_authority must be learning_authoritative for actionable output"
    issues.append(MonthlyModelValidationIssue(
        item_type=item_type,
        item_id=item_id,
        message=message,
    ))


def _evidence_key(path: str) -> str:
    return str(Path(path)).replace("\\", "/")


def _risk(value: str) -> MonthlyRiskClassification:
    try:
        return MonthlyRiskClassification((value or "").lower())
    except ValueError:
        return MonthlyRiskClassification.MEDIUM


def _approval_tier(risk: MonthlyRiskClassification) -> str:
    if risk in {MonthlyRiskClassification.HIGH, MonthlyRiskClassification.CRITICAL}:
        return "requires_double_approval"
    return "requires_approval"
