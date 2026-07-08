from __future__ import annotations

from typing import Any, Mapping

from strategies.core.serialization import restore_dataclass, snapshot_dataclass

from .state import ATRSSCoreState


def snapshot_state(state: ATRSSCoreState) -> dict[str, Any]:
    return snapshot_dataclass(state)


def restore_state(snapshot: Mapping[str, Any]) -> ATRSSCoreState:
    return restore_dataclass(ATRSSCoreState, snapshot)
