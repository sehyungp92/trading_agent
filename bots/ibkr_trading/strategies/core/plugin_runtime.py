from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Mapping
from typing import Any


async def delegate_hydrate(engine: Any, snapshot: dict[str, Any]) -> None:
    hydrate = getattr(engine, "hydrate", None)
    if callable(hydrate):
        result = hydrate(snapshot)
        if inspect.isawaitable(result):
            await result
        return

    hydrate_state = getattr(engine, "hydrate_state", None)
    if callable(hydrate_state):
        hydrate_state(snapshot)


def delegate_snapshot_state(engine: Any, *, strategy_id: str) -> dict[str, Any]:
    snapshot_state = getattr(engine, "snapshot_state", None)
    if not callable(snapshot_state):
        return {"strategy_id": strategy_id}

    state = snapshot_state()
    if dataclasses.is_dataclass(state):
        return dataclasses.asdict(state)
    if isinstance(state, Mapping):
        return dict(state)
    return {"strategy_id": strategy_id}
