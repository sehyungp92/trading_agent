from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from tests.integration.parity.fixtures import load_parity_fixture
from tests.integration.parity.live_runners import (
    run_layer2_live_trace,
    run_layer3_family_live_trace,
)
from tests.integration.parity.live_shadow_contract import FamilyShadowContract, LiveShadowContract, ParityTrace
from tests.integration.parity.replay_runners import (
    run_layer2_replay_trace,
    run_layer3_family_replay_trace,
)


async def run_layer2_contract(surface: str, fixture_path: Path) -> LiveShadowContract:
    fixture = load_parity_fixture(fixture_path)
    _assert_surface(surface, fixture)
    live = await run_layer2_live_trace(fixture)
    replay = await asyncio.to_thread(run_layer2_replay_trace, fixture)
    return LiveShadowContract(surface=surface, live=live, replay=replay)


async def run_layer3_family_contract(family: str, fixture_path: Path) -> FamilyShadowContract:
    fixture = load_parity_fixture(fixture_path)
    live = await run_layer3_family_live_trace(fixture)
    replay = await asyncio.to_thread(run_layer3_family_replay_trace, fixture)
    children = [
        LiveShadowContract(
            surface=f"{family}:{strategy_id}",
            live=_filter_trace(live, strategy_id),
            replay=_filter_trace(replay, strategy_id),
        )
        for strategy_id in _strategy_ids(fixture)
    ]
    return FamilyShadowContract(
        family=family,
        children=children,
        live_family_state=_family_state(live),
        replay_family_state=_family_state(replay),
    )


async def run_layer3_family_contracts(family: str, fixture_path: Path) -> list[LiveShadowContract]:
    return (await run_layer3_family_contract(family, fixture_path)).children


def _assert_surface(surface: str, fixture: dict[str, Any]) -> None:
    actual = str(fixture.get("surface", ""))
    assert actual == surface, f"fixture surface mismatch: expected {surface}, got {actual}"


def _strategy_ids(fixture: dict[str, Any]) -> list[str]:
    return [
        str(item["id"])
        for item in (fixture.get("family_config", {}) or {}).get("strategies", [])
        if item.get("id")
    ]


def _filter_trace(trace: ParityTrace, strategy_id: str) -> ParityTrace:
    state = trace.state_snapshot or {}
    strategy_state = state.get("strategy_state", {}) if isinstance(state, dict) else {}
    orders = state.get("orders", []) if isinstance(state, dict) else []
    positions = state.get("positions", []) if isinstance(state, dict) else []
    return ParityTrace(
        producer=trace.producer,
        source_fingerprint=trace.source_fingerprint,
        order_intents=[row for row in trace.order_intents if row.get("strategy_id") == strategy_id],
        terminal_events=[row for row in trace.terminal_events if row.get("strategy_id") == strategy_id],
        trade_ledger=[row for row in trace.trade_ledger if row.get("strategy_id") == strategy_id],
        state_snapshot={
            "strategy_id": strategy_id,
            "orders": [row for row in orders if row.get("strategy_id") == strategy_id],
            "positions": [
                _child_position(row)
                for row in positions
                if row.get("strategy_id") == strategy_id
            ],
            "strategy_state": strategy_state.get(strategy_id, {}),
        },
    )


def _family_state(trace: ParityTrace) -> dict[str, Any] | None:
    state = trace.state_snapshot or {}
    if not isinstance(state, dict):
        return None
    family_state = state.get("family_state")
    return family_state if isinstance(family_state, dict) else None


def _child_position(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy_id": row.get("strategy_id", ""),
        "symbol": row.get("symbol") or row.get("instrument_symbol") or "",
        "net_qty": row.get("net_qty", row.get("qty", 0)),
        "avg_price": row.get("avg_price", row.get("entry_price", 0)),
        "realized_pnl": row.get("realized_pnl", 0),
    }
