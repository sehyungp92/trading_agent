"""Deterministic greedy selection helper."""

from __future__ import annotations

from trading_assistant_backtest.auto.types import CandidateEvaluation


def ranked_candidates(evaluations: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
    return sorted(
        evaluations,
        key=lambda item: (not item.passed, -item.objective_score, item.candidate.candidate_id),
    )


def best_passing_candidate(evaluations: list[CandidateEvaluation]) -> CandidateEvaluation | None:
    passing = [item for item in ranked_candidates(evaluations) if item.passed]
    if not passing:
        return None
    return passing[0]


def no_adoption_reason(evaluations: list[CandidateEvaluation], default: str) -> str:
    if not evaluations:
        return default
    reasons = [
        reason
        for item in evaluations
        for reason in item.reasons
        if reason
    ]
    if not reasons:
        return "no candidate passed deterministic replay gates"
    preview = "; ".join(dict.fromkeys(reasons))
    return preview[:500]
