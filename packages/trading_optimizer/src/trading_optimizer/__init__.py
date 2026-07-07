"""Shared optimizer adapter interfaces."""

from trading_optimizer.archived_smoke import dimension_payloads, stable_payload_hash
from trading_optimizer.phase_runner_adapters import (
    ArchivedPhaseRunnerAdapter,
    PhaseRunnerSpec,
    runner_specs_for_records,
)

__all__ = [
    "ArchivedPhaseRunnerAdapter",
    "PhaseRunnerSpec",
    "dimension_payloads",
    "runner_specs_for_records",
    "stable_payload_hash",
]
