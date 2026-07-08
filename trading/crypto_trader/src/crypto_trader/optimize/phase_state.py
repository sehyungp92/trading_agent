"""Atomic JSON persistence for optimization state across phases."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class NumpySafeEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _atomic_write_json(data: Any, path: Path) -> None:
    """Atomically write JSON data using os.replace (safe on Windows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, cls=NumpySafeEncoder)
        os.replace(str(tmp_path), str(path))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _int_key_dict(d: dict) -> dict[int, Any]:
    """Convert string-keyed dict to int-keyed dict."""
    return {int(k): v for k, v in d.items()}


@dataclass
class PhaseState:
    """Tracks optimization progress across phases."""

    current_phase: int = 0
    completed_phases: list[int] = field(default_factory=list)
    cumulative_mutations: dict[str, Any] = field(default_factory=dict)
    phase_metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    round_name: str = ""

    # Per-phase retry tracking
    scoring_retries: dict[int, int] = field(default_factory=dict)
    diagnostic_retries: dict[int, int] = field(default_factory=dict)
    retry_count: dict[int, int] = field(default_factory=dict)

    # Per-phase result/gate/timestamp storage
    phase_results: dict[int, dict] = field(default_factory=dict)
    phase_gate_results: dict[int, dict] = field(default_factory=dict)
    phase_timestamps: dict[int, dict[str, str]] = field(default_factory=dict)
    contract_hash: str = ""
    contract: dict[str, Any] = field(default_factory=dict)
    invalid_phases: dict[int, dict[str, Any]] = field(default_factory=dict)

    _path: Path | None = field(default=None, repr=False)

    def set_contract(self, contract: dict[str, Any]) -> None:
        """Attach the optimization contract used by this state file."""
        contract_hash = str(contract.get("contract_hash") or "")
        self.contract_hash = contract_hash
        self.contract = dict(contract)

    def ensure_contract(self, contract: dict[str, Any], *, strict: bool = True) -> None:
        """Validate that persisted state belongs to the current run contract."""
        expected = str(contract.get("contract_hash") or "")
        if not expected:
            return

        is_fresh = (
            not self.contract_hash
            and not self.contract
            and self.current_phase == 0
            and not self.completed_phases
            and not self.phase_results
            and not self.phase_metrics
        )
        if is_fresh:
            self.set_contract(contract)
            return

        if self.contract_hash == expected:
            if not self.contract:
                self.contract = dict(contract)
            return

        if strict:
            found = self.contract_hash or "<missing>"
            raise RuntimeError(
                f"Phase state contract mismatch: found {found}, expected {expected}"
            )

        self.set_contract(contract)

    def mark_phase_invalid(
        self,
        phase: int,
        *,
        reason: str,
        error: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record that a phase produced untrusted final evidence."""
        self.invalid_phases[phase] = {
            "reason": reason,
            "error": error,
            "metadata": metadata or {},
            "timestamp": _utc_now_iso(),
        }

    def start_phase(self, phase: int) -> None:
        """Mark a phase as started."""
        self.current_phase = phase
        self.phase_timestamps.setdefault(phase, {})
        self.phase_timestamps[phase]["started"] = _utc_now_iso()

    def advance_phase(
        self,
        phase: int,
        mutations: dict[str, Any],
        metrics_or_result: dict[str, float] | dict[str, Any],
    ) -> None:
        """Record phase completion, merge mutations, store metrics."""
        self.cumulative_mutations.update(mutations)

        # Extract metrics dict — if it has "final_metrics" key, use that
        if isinstance(metrics_or_result, dict) and "final_metrics" in metrics_or_result:
            metrics = metrics_or_result["final_metrics"]
            self.phase_results[phase] = metrics_or_result
        else:
            metrics = metrics_or_result

        self.phase_metrics[phase] = metrics

        if phase not in self.completed_phases:
            self.completed_phases.append(phase)
        self.current_phase = phase + 1
        self.invalid_phases.pop(phase, None)

    def complete_phase(self, phase: int) -> None:
        """Record phase completion timestamp."""
        self.phase_timestamps.setdefault(phase, {})
        self.phase_timestamps[phase]["completed"] = _utc_now_iso()

    def increment_scoring_retry(self, phase: int) -> int:
        """Increment and return scoring retry count for a phase."""
        self.scoring_retries[phase] = self.scoring_retries.get(phase, 0) + 1
        return self.scoring_retries[phase]

    def increment_diagnostic_retry(self, phase: int) -> int:
        """Increment and return diagnostic retry count for a phase."""
        self.diagnostic_retries[phase] = self.diagnostic_retries.get(phase, 0) + 1
        return self.diagnostic_retries[phase]

    def increment_retry(self, phase: int) -> int:
        """Increment and return general retry count for a phase."""
        self.retry_count[phase] = self.retry_count.get(phase, 0) + 1
        return self.retry_count[phase]

    def record_gate(self, phase: int, gate_dict: dict) -> None:
        """Store gate result dict for a phase."""
        self.phase_gate_results[phase] = gate_dict

    def record_result(self, phase: int, result_dict: dict) -> None:
        """Store result dict for a phase."""
        self.phase_results[phase] = result_dict

    def get_phase_metrics(self, phase: int) -> dict | None:
        """Get metrics for a completed phase."""
        return self.phase_metrics.get(phase)

    def rollback_to_phase(self, target_phase: int) -> list[int]:
        """Remove data for phases >= target_phase and re-derive cumulative mutations.

        Returns list of rolled-back phase numbers.
        """
        stale = [p for p in self.completed_phases if p >= target_phase]
        if not stale:
            return []

        # Remove stale phases
        for p in stale:
            self.completed_phases.remove(p)
            self.phase_metrics.pop(p, None)
            self.phase_results.pop(p, None)
            self.phase_gate_results.pop(p, None)
            self.phase_timestamps.pop(p, None)
            self.scoring_retries.pop(p, None)
            self.diagnostic_retries.pop(p, None)
            self.retry_count.pop(p, None)
            self.invalid_phases.pop(p, None)

        # Re-derive cumulative mutations from remaining phase results
        self.cumulative_mutations = {}
        for p in sorted(self.completed_phases):
            result = self.phase_results.get(p)
            if result and "final_mutations" in result:
                self.cumulative_mutations.update(result["final_mutations"])

        self.current_phase = target_phase
        return stale

    def save(self, path: Path | None = None) -> None:
        """Atomically save state to JSON."""
        path = path or self._path
        if path is None:
            raise ValueError("No save path specified")
        self._path = path

        data = {
            "current_phase": self.current_phase,
            "completed_phases": self.completed_phases,
            "cumulative_mutations": self.cumulative_mutations,
            "phase_metrics": {str(k): v for k, v in self.phase_metrics.items()},
            "round_name": self.round_name,
            "scoring_retries": {str(k): v for k, v in self.scoring_retries.items()},
            "diagnostic_retries": {str(k): v for k, v in self.diagnostic_retries.items()},
            "retry_count": {str(k): v for k, v in self.retry_count.items()},
            "phase_results": {str(k): v for k, v in self.phase_results.items()},
            "phase_gate_results": {str(k): v for k, v in self.phase_gate_results.items()},
            "phase_timestamps": {str(k): v for k, v in self.phase_timestamps.items()},
            "contract_hash": self.contract_hash,
            "contract": self.contract,
            "invalid_phases": {str(k): v for k, v in self.invalid_phases.items()},
        }

        _atomic_write_json(data, path)

    @classmethod
    def load(cls, path: Path) -> PhaseState:
        """Load state from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        state = cls(
            current_phase=data.get("current_phase", 0),
            completed_phases=data.get("completed_phases", []),
            cumulative_mutations=data.get("cumulative_mutations", {}),
            phase_metrics=_int_key_dict(data.get("phase_metrics", {})),
            round_name=data.get("round_name", ""),
            scoring_retries=_int_key_dict(data.get("scoring_retries", {})),
            diagnostic_retries=_int_key_dict(data.get("diagnostic_retries", {})),
            retry_count=_int_key_dict(data.get("retry_count", {})),
            phase_results=_int_key_dict(data.get("phase_results", {})),
            phase_gate_results=_int_key_dict(data.get("phase_gate_results", {})),
            phase_timestamps=_int_key_dict(data.get("phase_timestamps", {})),
            contract_hash=data.get("contract_hash", ""),
            contract=data.get("contract", {}),
            invalid_phases=_int_key_dict(data.get("invalid_phases", {})),
            _path=path,
        )
        return state

    @classmethod
    def load_or_create(cls, path: Path) -> PhaseState:
        """Load existing state or create fresh."""
        if path.exists():
            return cls.load(path)
        state = cls(_path=path)
        return state
