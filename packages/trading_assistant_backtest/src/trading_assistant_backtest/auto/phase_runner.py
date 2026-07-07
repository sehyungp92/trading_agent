"""Shared deterministic phase runner.

The runner owns candidate enumeration and failure retention. Strategy plugins
can provide the evaluator; without one, candidates are evaluated as
diagnostic-only and fail closed instead of disappearing as an empty phase.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from trading_assistant_backtest.auto.types import Candidate, CandidateEvaluation, PhaseSpec

CandidateEvaluator = Callable[[Candidate], CandidateEvaluation]


def enumerate_phase_candidates(phase: PhaseSpec) -> list[Candidate]:
    families = _dedupe([family for family in phase.candidate_families if family])
    limit = phase.max_candidates if phase.max_candidates > 0 else len(families)
    candidates: list[Candidate] = []
    for index, family in enumerate(families[:limit], start=1):
        family_payload = {}
        raw_families = phase.metadata.get("candidate_payloads_by_family", {})
        if isinstance(raw_families, dict):
            family_payload = raw_families.get(family, {}) or {}
            if not isinstance(family_payload, dict):
                family_payload = {}
        candidates.append(
            Candidate(
                candidate_id=f"{_safe_id(phase.phase_id)}-{_safe_id(family)}-{index}",
                family=family,
                payload={
                    "phase_id": phase.phase_id,
                    "candidate_family": family,
                    "ordinal": index,
                    "mutation_type": _mutation_type(family),
                    "target_scope": phase.metadata.get("target_scope", {}),
                    "parameter_patch": family_payload.get(
                        "parameter_patch",
                        {"family": family, "mode": "deterministic_monthly_candidate"},
                    ),
                    "structural_patch_ref": family_payload.get("structural_patch_ref", ""),
                    "expected_mechanism": family_payload.get(
                        "expected_mechanism", _expected_mechanism(family)
                    ),
                    "source_evidence_paths": family_payload.get(
                        "source_evidence_paths",
                        phase.metadata.get("source_evidence_paths", []),
                    ),
                    "weekly_signal_attribution": family_payload.get(
                        "weekly_signal_attribution",
                        phase.metadata.get("weekly_signal_attribution", []),
                    ),
                    "rollback_plan_ref": family_payload.get(
                        "rollback_plan_ref",
                        f"rollback:{_safe_id(family)}:restore_round_n_config",
                    ),
                    **family_payload,
                },
            )
        )
    return candidates


def run_phase(
    phase: PhaseSpec,
    *,
    evaluator: CandidateEvaluator | None = None,
) -> list[CandidateEvaluation]:
    evaluate = evaluator or _diagnostic_only_evaluator
    return [evaluate(candidate) for candidate in enumerate_phase_candidates(phase)]


def run_empty_phase(phase: PhaseSpec) -> list[CandidateEvaluation]:
    """Compatibility wrapper; phases now emit retained diagnostic rejections."""

    return run_phase(phase)


def _diagnostic_only_evaluator(candidate: Candidate) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate=candidate,
        objective_score=0.0,
        passed=False,
        reasons=[
            "strategy plugin has not provided a replay-backed evaluator for this candidate family"
        ],
    )


def _safe_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return text[:64] or "candidate"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _mutation_type(family: str) -> str:
    normalized = family.lower()
    if "rollback" in normalized:
        return "accepted_mutation_rollback"
    if "structural" in normalized:
        return "structural_patch"
    if "risk" in normalized or "size" in normalized:
        return "sizing_risk_cap_patch"
    if "exit" in normalized or "stop" in normalized or "take" in normalized:
        return "exit_management_patch"
    if "filter" in normalized or "regime" in normalized or "session" in normalized:
        return "filter_threshold_patch"
    if "entry" in normalized or "signal" in normalized:
        return "entry_signal_patch"
    return "parameter_patch"


def _expected_mechanism(family: str) -> str:
    normalized = family.lower().replace("_", " ")
    if "filter" in normalized:
        return "adjust filter strictness to recover persistent fold-supported signal quality"
    if "exit" in normalized:
        return "alter exits or stops to retain expectancy without drawdown regression"
    if "risk" in normalized or "size" in normalized:
        return "resize exposure or caps to improve risk-adjusted objective stability"
    if "session" in normalized:
        return "gate weak session windows while preserving in-sample fold coverage"
    if "regime" in normalized:
        return "condition entries on regime evidence to reduce adverse environments"
    if "rollback" in normalized:
        return "remove a prior accepted mutation implicated by selection-OOS degradation"
    return f"test {normalized} as a bounded monthly optimizer mutation"
