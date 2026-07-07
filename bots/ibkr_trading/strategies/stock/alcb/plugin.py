"""Plugin adapter for ALCB campaign stock strategy."""
from __future__ import annotations

import dataclasses
import logging
from typing import Any

from strategies.contracts import RuntimeContext
from strategies.core.capital import resolve_plugin_nav
from .config import StrategySettings
from .diagnostics import JsonlDiagnostics
from .engine import ALCBT2Engine

logger = logging.getLogger(__name__)


class ALCBPlugin:
    strategy_id = "ALCB_v1"

    def __init__(self, ctx: RuntimeContext) -> None:
        self._ctx = ctx
        manifest = ctx.manifest
        settings = StrategySettings()

        # account_id from the connection group tied to this strategy
        conn_group = ctx.registry.connection_groups[manifest.connection_group]
        account_id = conn_group.account_id or ""

        nav = resolve_plugin_nav(ctx, self.strategy_id)

        # Artifact will be supplied by the family coordinator before start().
        # Store a sentinel so the coordinator can inject it.
        self._artifact: Any = None

        trade_recorder = getattr(ctx.instrumentation, "trade_recorder", None)
        diagnostics = JsonlDiagnostics(settings.diagnostics_dir, enabled=True)

        self._settings = settings
        self._account_id = account_id
        self._nav = nav
        self._trade_recorder = trade_recorder
        self._diagnostics = diagnostics
        self._instrumentation = ctx.instrumentation
        self._engine: ALCBT2Engine | None = None
        self._pending_snapshot: dict[str, Any] | None = None

    # -- lifecycle --------------------------------------------------------

    def _build_engine(self) -> ALCBT2Engine:
        if self._artifact is None:
            raise RuntimeError(
                f"{self.strategy_id}: artifact must be set before start(). "
                "The family coordinator should call plugin._artifact = artifact."
            )
        return ALCBT2Engine(
            oms_service=self._ctx.oms,
            artifact=self._artifact,
            account_id=self._account_id,
            nav=self._nav,
            settings=self._settings,
            trade_recorder=self._trade_recorder,
            diagnostics=self._diagnostics,
            instrumentation=self._instrumentation,
        )

    async def start(self) -> None:
        self._engine = self._build_engine()
        if self._pending_snapshot is not None:
            self._engine.hydrate_state(self._pending_snapshot)
        await self._engine.start()

    async def stop(self) -> None:
        if self._engine is not None:
            await self._engine.stop()

    def health_status(self) -> dict[str, Any]:
        if self._engine is not None:
            return self._engine.health_status()
        return {
            "strategy_id": self.strategy_id,
            "running": False,
            "has_artifact": self._artifact is not None,
        }

    async def hydrate(self, snapshot: dict[str, Any]) -> None:
        self._pending_snapshot = snapshot
        if self._engine is not None:
            self._engine.hydrate_state(snapshot)

    def snapshot_state(self) -> dict[str, Any]:
        if self._engine is not None and hasattr(self._engine, "snapshot_state"):
            state = self._engine.snapshot_state()
            if dataclasses.is_dataclass(state):
                return dataclasses.asdict(state)
            return state
        if self._pending_snapshot is not None:
            return self._pending_snapshot
        return {"strategy_id": self.strategy_id}

    async def on_market_data(self, event: Any) -> None:
        pass

    async def on_order_event(self, event: Any) -> None:
        pass

    async def on_fill_event(self, event: Any) -> None:
        pass
