"""Plugin adapter for Downturn Dominator v1 momentum strategy."""
from __future__ import annotations

from typing import Any

from strategies.contracts import RuntimeContext
from .engine import DownturnEngine


class DownturnPlugin:
    strategy_id = "DownturnDominator_v1"

    def __init__(self, ctx: RuntimeContext) -> None:
        self._ctx = ctx
        cfg = getattr(ctx.manifest, "config", {})
        self._engine = DownturnEngine(
            ib_session=ctx.session,
            oms_service=ctx.oms,
            instruments=dict(ctx.contracts),
            trade_recorder=getattr(ctx.instrumentation, "trade_recorder", None),
            equity=getattr(ctx.portfolio, "allocation", 10_000.0),
            symbol=cfg.get("symbol", "MNQ"),
            state_dir=getattr(ctx.state_store, "state_dir", None),
            instrumentation=ctx.instrumentation,
        )

    async def start(self) -> None:
        await self._engine.start()

    async def stop(self) -> None:
        await self._engine.stop()

    def health_status(self) -> dict[str, Any]:
        return self._engine.health_status()

    async def hydrate(self, snapshot: dict[str, Any]) -> None:
        await self._engine.hydrate(snapshot)

    def snapshot_state(self) -> dict[str, Any]:
        return self._engine.snapshot_state()

    async def on_market_data(self, event: Any) -> None:
        pass  # Engine subscribes directly via IB session

    async def on_order_event(self, event: Any) -> None:
        pass  # Engine listens via OMS event bus

    async def on_fill_event(self, event: Any) -> None:
        pass  # Engine listens via OMS event bus
