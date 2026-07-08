"""Compatibility shim for shared phase state."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from backtests.shared.auto.phase_state import (
    PhaseState as _SharedPhaseState,
    _NumpySafeEncoder,
    _atomic_write_json,
    load_phase_state,
    save_phase_state,
)


class PhaseState(_SharedPhaseState):
    @classmethod
    def load(cls, path: Path) -> "PhaseState":
        state = load_phase_state(path)
        return cls(**asdict(state))

    def save(self, path: Path) -> None:
        save_phase_state(self, path)

    def advance_phase(self) -> None:  # type: ignore[override]
        if self.current_phase and self.current_phase not in self.completed_phases:
            self.completed_phases.append(self.current_phase)
        self.current_phase = self.current_phase + 1 if self.current_phase else 1
        self.retry_count.pop(self.current_phase, None)
        self.scoring_retries.pop(self.current_phase, None)
        self.diagnostic_retries.pop(self.current_phase, None)
        self.phase_timestamps.setdefault(self.current_phase, {})

    def increment_retry(self) -> None:  # type: ignore[override]
        super().increment_retry(self.current_phase or 1)

    def increment_scoring_retry(self) -> None:  # type: ignore[override]
        super().increment_scoring_retry(self.current_phase or 1)

    def increment_diagnostic_retry(self) -> None:  # type: ignore[override]
        super().increment_diagnostic_retry(self.current_phase or 1)


__all__ = [
    "PhaseState",
    "_NumpySafeEncoder",
    "_atomic_write_json",
    "load_phase_state",
    "save_phase_state",
]
