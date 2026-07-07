"""Plugin adapter for ATRSS (ETRS vFinal) swing strategy."""
from __future__ import annotations

from typing import Any

from strategies.contracts import RuntimeContext
from strategies.core.capital import resolve_plugin_nav
from strategies.core.plugin_runtime import delegate_hydrate, delegate_snapshot_state
from .config import SYMBOL_CONFIGS, SymbolConfig
from .engine import ATRSSEngine


class ATRSSPlugin:
    strategy_id = "ATRSS"

    def __init__(self, ctx: RuntimeContext) -> None:
        self._ctx = ctx
        self._engine = ATRSSEngine(
            ib_session=ctx.session,
            oms_service=ctx.oms,
            instruments=ctx.contracts,
            config=SYMBOL_CONFIGS,
            trade_recorder=getattr(ctx.instrumentation, "trade_recorder", None),
            equity=resolve_plugin_nav(ctx, self.strategy_id),
            market_calendar=getattr(ctx.market_data, "market_calendar", None),
            kit=ctx.instrumentation,
            equity_offset=getattr(
                ctx.manifest.allocation, "equity_offset", 0.0
            ),
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
