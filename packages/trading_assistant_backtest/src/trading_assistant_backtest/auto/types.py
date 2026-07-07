"""Shared optimization types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: Candidate
    objective_score: float
    passed: bool
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PhaseSpec:
    phase_id: str
    candidate_families: list[str]
    max_candidates: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
