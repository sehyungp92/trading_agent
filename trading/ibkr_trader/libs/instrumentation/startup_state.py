"""Best-effort startup state collection for instrumentation snapshots."""
from __future__ import annotations

from typing import Any


async def collect_startup_snapshot_state(
    oms: Any,
    *,
    strategy_ids: list[str] | None = None,
    default_strategy_id: str = "",
    source: str = "runtime_startup",
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Collect hydrated OMS state for startup allocation, portfolio, and positions.

    Collection is fail-open: callers still emit startup events with explicit
    source markers even when OMS state is unavailable.
    """
    allocation_state: dict[str, Any] = {"source": source}
    portfolio_state: dict[str, Any] = {"source": source}
    positions: list[dict[str, Any]] = []
    if oms is None:
        return allocation_state, portfolio_state, positions

    try:
        from libs.oms.instrumentation.daily_state import collect_family_daily_state

        collected_portfolio, _, collected_allocation = await collect_family_daily_state(
            [oms],
            strategy_ids=list(strategy_ids or []),
            default_strategy_id=default_strategy_id,
        )
        if isinstance(collected_portfolio, dict):
            portfolio_state.update(collected_portfolio)
            raw_positions = collected_portfolio.get("positions") or []
            if isinstance(raw_positions, list):
                positions = [dict(item) for item in raw_positions if isinstance(item, dict)]
        if isinstance(collected_allocation, dict):
            allocation_state.update(collected_allocation)
        allocation_state["source"] = source
        portfolio_state["source"] = source
        return allocation_state, portfolio_state, positions
    except Exception as exc:
        allocation_state["collection_error"] = str(exc)
        portfolio_state["collection_error"] = str(exc)
        return allocation_state, portfolio_state, positions
