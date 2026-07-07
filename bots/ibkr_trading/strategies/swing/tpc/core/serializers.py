from __future__ import annotations

from typing import Any, Mapping

from strategies.core.serialization import restore_dataclass, snapshot_dataclass

from .state import TPCCoreState


def snapshot_state(state: TPCCoreState) -> dict[str, Any]:
    return snapshot_dataclass(state)


def restore_state(snapshot: Mapping[str, Any] | None) -> TPCCoreState:
    if snapshot is None:
        return TPCCoreState()
    return restore_dataclass(TPCCoreState, snapshot)
