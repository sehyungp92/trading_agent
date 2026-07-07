from __future__ import annotations

from types import SimpleNamespace

import pytest

from strategies.momentum.vdub.plugin import VdubNQv4Plugin
from strategies.swing.akc_helix.plugin import AKCHelixPlugin
from strategies.swing.atrss.plugin import ATRSSPlugin
from strategies.swing.tpc.plugin import TPCPlugin


@pytest.mark.parametrize(
    ("plugin_cls", "strategy_id"),
    [
        (ATRSSPlugin, "ATRSS"),
        (AKCHelixPlugin, "AKC_HELIX"),
        (TPCPlugin, "TPC"),
        (VdubNQv4Plugin, "VdubusNQ_v4"),
    ],
)
def test_plugin_health_status_delegates_full_engine_payload(plugin_cls, strategy_id) -> None:
    plugin = plugin_cls.__new__(plugin_cls)
    plugin._engine = SimpleNamespace(
        health_status=lambda: {
            "strategy_id": strategy_id,
            "running": True,
            "last_decision_code": "ENTRY_SUBMITTED",
        }
    )

    assert plugin.health_status() == {
        "strategy_id": strategy_id,
        "running": True,
        "last_decision_code": "ENTRY_SUBMITTED",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plugin_cls",
    [
        ATRSSPlugin,
        AKCHelixPlugin,
        TPCPlugin,
        VdubNQv4Plugin,
    ],
)
async def test_plugin_snapshot_and_hydrate_delegate_when_engine_supports_contract(plugin_cls) -> None:
    hydrated: list[dict[str, object]] = []

    class _FakeEngine:
        async def hydrate(self, snapshot: dict[str, object]) -> None:
            hydrated.append(snapshot)

        def snapshot_state(self) -> dict[str, object]:
            return {"restored": True}

    plugin = plugin_cls.__new__(plugin_cls)
    plugin._engine = _FakeEngine()

    await plugin.hydrate({"sequence": 1})

    assert hydrated == [{"sequence": 1}]
    assert plugin.snapshot_state() == {"restored": True}


@pytest.mark.asyncio
async def test_plugin_hydrate_falls_back_to_sync_hydrate_state() -> None:
    hydrated: list[dict[str, object]] = []

    class _FakeEngine:
        def hydrate_state(self, snapshot: dict[str, object]) -> None:
            hydrated.append(snapshot)

        def snapshot_state(self) -> dict[str, object]:
            return {"restored": "sync"}

    plugin = ATRSSPlugin.__new__(ATRSSPlugin)
    plugin._engine = _FakeEngine()

    await plugin.hydrate({"sequence": 2})

    assert hydrated == [{"sequence": 2}]
    assert plugin.snapshot_state() == {"restored": "sync"}
