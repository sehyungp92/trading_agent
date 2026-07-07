from __future__ import annotations

from typing import Any, Mapping

from strategies.core.serialization import restore_dataclass, snapshot_dataclass

from .state import ALCBCoreState


def snapshot_state(state: ALCBCoreState) -> dict[str, Any]:
    return snapshot_dataclass(state)


def restore_state(snapshot: Mapping[str, Any]) -> ALCBCoreState:
    return restore_dataclass(ALCBCoreState, snapshot)
