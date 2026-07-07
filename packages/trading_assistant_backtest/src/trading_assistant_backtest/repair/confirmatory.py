"""Repair-centered confirmatory follow-up helpers."""

from __future__ import annotations

from collections.abc import Iterable

from trading_assistant_backtest.auto.types import Candidate, CandidateEvaluation


def build_confirmatory_variants(
    primary: Candidate | None,
    alternatives: Iterable[Candidate],
) -> list[Candidate]:
    if primary is None:
        return []
    variants = [
        Candidate(
            candidate_id=f"{primary.candidate_id}-confirm",
            family=primary.family,
            payload={**primary.payload, "variant_type": "primary_confirmatory"},
        )
    ]
    for candidate in alternatives:
        variants.append(
            Candidate(
                candidate_id=f"{candidate.candidate_id}-confirm",
                family=candidate.family,
                payload={**candidate.payload, "variant_type": "alternative_confirmatory"},
            )
        )
    return variants


def variant_payload(evaluation: CandidateEvaluation) -> dict:
    return {
        "candidate_id": evaluation.candidate.candidate_id,
        "source_candidate_id": str(evaluation.candidate.payload.get("source_candidate_id") or ""),
        "variant_type": str(evaluation.candidate.payload.get("variant_type") or ""),
        "objective_score": evaluation.objective_score,
        "baseline_score": 0.0,
        "in_sample_delta": 0.0,
        "selection_oos_delta": 0.0,
        "fold_support_passed": evaluation.passed,
        "deterministic_replay_passed": evaluation.passed,
        "materially_degrades_in_sample": False,
        "evidence_paths": [],
    }
