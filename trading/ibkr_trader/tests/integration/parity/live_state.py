from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tests.integration.parity.family_state import build_family_state
from tests.integration.parity.live_family import family_surface_adapter_name
from tests.integration.parity.live_layer2 import compact_engine_state, compact_overlay_state
from tests.integration.parity.live_oms import (
    blocked_reasons_from_repo_events,
    plain_dataclass as _plain_dataclass,
    portfolio_rules_state,
)
from tests.integration.parity.source_inputs import strategy_ids


async def _state_from_repos(
    repos: list[Any],
    oms_services: list[Any],
    fixture: Mapping[str, Any],
    engines: Mapping[str, Any],
    coordinator: Any,
) -> dict[str, Any]:
    orders = []
    positions = []
    for repo in repos:
        for order in repo._orders.values():
            orders.append(
                {
                    "oms_order_id": order.oms_order_id,
                    "strategy_id": order.strategy_id,
                    "symbol": order.instrument.symbol if order.instrument else "",
                    "side": order.side.value,
                    "qty": order.qty,
                    "order_type": order.order_type.value,
                    "limit_price": order.limit_price,
                    "stop_price": order.stop_price,
                    "role": order.role.value,
                    "status": order.status.value,
                    "filled_qty": order.filled_qty,
                    "remaining_qty": order.remaining_qty,
                    "avg_fill_price": order.avg_fill_price,
                    "client_tag": order.client_order_id,
                    "reject_reason": order.reject_reason,
                }
            )
        for pos in repo._positions.values():
            positions.append(
                {
                    "strategy_id": pos.strategy_id,
                    "symbol": pos.instrument_symbol,
                    "net_qty": pos.net_qty,
                    "avg_price": pos.avg_price,
                    "realized_pnl": pos.realized_pnl,
                    "open_risk_dollars": pos.open_risk_dollars,
                    "open_risk_R": pos.open_risk_R,
                }
            )
    strategy_risk = {}
    portfolio_risk = []
    configured_strategy_ids = strategy_ids(fixture)
    for service in oms_services:
        for sid in configured_strategy_ids:
            get_strategy_risk = getattr(service, "get_strategy_risk", None)
            if get_strategy_risk is not None:
                await get_strategy_risk(sid)
        get_portfolio_risk = getattr(service, "get_portfolio_risk", None)
        if get_portfolio_risk is not None:
            await get_portfolio_risk()
        for sid, state in getattr(service, "_strategy_risk_states", {}).items():
            strategy_risk[sid] = _plain_dataclass(state)
        prs = getattr(service, "_portfolio_risk_state", None)
        if prs is not None:
            portfolio_risk.append(_plain_dataclass(prs))
    strategy_state = {
        strategy_id: compact_engine_state(engine, strategy_id)
        for strategy_id, engine in sorted(engines.items())
        if strategy_id in set(strategy_ids(fixture))
    }
    overlay_state = compact_overlay_state(engines.get("OVERLAY"))
    coordinator_class = type(coordinator).__name__ if coordinator is not None else ""
    blocked_reasons = blocked_reasons_from_repo_events(repos, orders)
    return {
        "orders": orders,
        "positions": positions,
        "strategy_risk": strategy_risk,
        "portfolio_risk": portfolio_risk,
        "portfolio_rules": portfolio_rules_state(oms_services),
        "blocked_reasons": blocked_reasons,
        "strategy_state": strategy_state,
        "family_state": build_family_state(
            fixture,
            coordinator_class=coordinator_class,
            orders=orders,
            positions=positions,
            strategy_risk=strategy_risk,
            portfolio_risk=portfolio_risk,
            portfolio_rules=portfolio_rules_state(oms_services),
            strategy_state=strategy_state,
            overlay_state=overlay_state,
            surface_adapter=family_surface_adapter_name(fixture),
            blocked_reasons=blocked_reasons,
        ),
    }


state_from_repos = _state_from_repos
