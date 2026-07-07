from __future__ import annotations

from typing import Any, Mapping

from strategies.core.serialization import restore_dataclass, snapshot_dataclass

from .state import VdubCoreState


def snapshot_state(state: VdubCoreState) -> dict[str, Any]:
    return snapshot_dataclass(state)


def restore_state(snapshot: Mapping[str, Any]) -> VdubCoreState:
    return restore_dataclass(VdubCoreState, snapshot)
