"""Compatibility shim for shared phase state."""
from backtests.shared.auto.phase_state import (
    PhaseState,
    _NumpySafeEncoder,
    _atomic_write_json,
    load_phase_state,
    save_phase_state,
)

__all__ = [
    "PhaseState",
    "_NumpySafeEncoder",
    "_atomic_write_json",
    "load_phase_state",
    "save_phase_state",
]
