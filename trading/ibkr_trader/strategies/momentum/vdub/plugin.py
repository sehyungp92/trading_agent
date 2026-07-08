"""Plugin adapter for VdubusNQ v4 momentum strategy."""
from __future__ import annotations

from typing import Any

from strategies.contracts import RuntimeContext
from strategies.core.capital import resolve_plugin_nav
from strategies.core.plugin_runtime import delegate_hydrate, delegate_snapshot_state
from .engine import VdubNQv4Engine


class VdubNQv4Plugin:
    strategy_id = "VdubusNQ_v4"

    def __init__(self, ctx: RuntimeContext) -> None:
        self._ctx = ctx
        self._engine = VdubNQv4Engine(
            ib_session=ctx.session,
            oms_service=ctx.oms,
            instruments=list(ctx.contracts.values()),
            trade_recorder=getattr(ctx.instrumentation, "trade_recorder", None),
            equity=resolve_plugin_nav(ctx, self.strategy_id),
            instrumentation=ctx.instrumentation,
        )

    async def start(self) -> None:
        await self._engine.start()

    async def stop(self) -> None:
        await self._engine.stop()

    def health_status(self) -> dict[str, Any]:
        return self._engine.health_status()

    async def hydrate(self, snapshot: dict[str, Any]) -> None:
        await delegate_hydrate(self._engine, snapshot)

    def snapshot_state(self) -> dict[str, Any]:
        return delegate_snapshot_state(self._engine, strategy_id=self.strategy_id)

    async def on_market_data(self, event: Any) -> None:
        pass  # Engine subscribes directly via IB session

    async def on_order_event(self, event: Any) -> None:
        pass  # Engine listens via OMS event bus

    async def on_fill_event(self, event: Any) -> None:
        pass  # Engine listens via OMS event bus
