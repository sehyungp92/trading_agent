from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from strategies.stock.iaric.artifact_store import coerce_intraday_state_snapshot


def snapshot_state(state) -> dict[str, Any]:
    return asdict(state) if is_dataclass(state) else dict(state)


def restore_state(snapshot) -> Any:
    return coerce_intraday_state_snapshot(snapshot)
