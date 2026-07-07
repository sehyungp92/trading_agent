"""Phase checkpoint state."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PhaseState:
    completed_phase_ids: list[str] = field(default_factory=list)
    rejected_candidate_ids: list[str] = field(default_factory=list)
