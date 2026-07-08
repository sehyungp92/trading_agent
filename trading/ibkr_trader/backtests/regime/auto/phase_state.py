"""Phase state persistence for multi-phase regime optimization."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class _NumpySafeEncoder(json.JSONEncoder):
    """Handle numpy types that aren't JSON serializable."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class PhaseState:
    """Tracks progress across optimization phases."""
    current_phase: int = 0
    completed_phases: list[int] = field(default_factory=list)
    cumulative_mutations: dict[str, Any] = field(default_factory=dict)
    phase_results: dict[int, dict] = field(default_factory=dict)
    phase_gate_results: dict[int, dict] = field(default_factory=dict)
    retry_count: dict[int, int] = field(default_factory=dict)

    def advance_phase(self, phase: int, mutations: dict, result: dict) -> None:
        """Record a completed phase and accumulate mutations."""
        if phase not in self.completed_phases:
            self.completed_phases.append(phase)
        self.cumulative_mutations.update(mutations)
        self.phase_results[phase] = result
        self.current_phase = phase

    def record_gate(self, phase: int, gate_result: dict) -> None:
        self.phase_gate_results[phase] = gate_result

    def increment_retry(self, phase: int) -> int:
        self.retry_count[phase] = self.retry_count.get(phase, 0) + 1
        return self.retry_count[phase]


def save_phase_state(state: PhaseState, path: Path) -> None:
    """Serialize phase state to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert int keys to strings for JSON
    data = {
        "current_phase": state.current_phase,
        "completed_phases": state.completed_phases,
        "cumulative_mutations": state.cumulative_mutations,
        "phase_results": {str(k): v for k, v in state.phase_results.items()},
        "phase_gate_results": {str(k): v for k, v in state.phase_gate_results.items()},
        "retry_count": {str(k): v for k, v in state.retry_count.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=_NumpySafeEncoder)


def load_phase_state(path: Path) -> PhaseState:
    """Deserialize phase state from JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return PhaseState(
        current_phase=data.get("current_phase", 0),
        completed_phases=data.get("completed_phases", []),
        cumulative_mutations=data.get("cumulative_mutations", {}),
        phase_results={int(k): v for k, v in data.get("phase_results", {}).items()},
        phase_gate_results={int(k): v for k, v in data.get("phase_gate_results", {}).items()},
        retry_count={int(k): v for k, v in data.get("retry_count", {}).items()},
    )
