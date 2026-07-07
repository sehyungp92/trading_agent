"""Runner observability payloads."""

from __future__ import annotations

from datetime import UTC, datetime

from trading_assistant_backtest.contract_models import MonthlyRunManifest


def runner_event(manifest: MonthlyRunManifest, *, phase: str, status: str, **fields) -> dict:
    return {
        "run_id": manifest.run_id,
        "manifest_id": manifest.manifest_id if manifest.optimizer_mode else "",
        "phase": phase,
        "attempt_state": status,
        "generated_at": datetime.now(UTC).isoformat(),
        **fields,
    }
