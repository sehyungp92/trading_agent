from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from libs.oms.models.order import OrderSide
from tests.integration.parity.normalizers import normalize_reason


def drain_queue(queue: Any) -> list[Any]:
    events = []
    if queue is None:
        return events
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def plain_dataclass(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def portfolio_rules_state(oms_or_services: Any) -> list[dict[str, Any]]:
    rules = []
    for service in _services(oms_or_services):
        checker = getattr(service, "_portfolio_checker", None)
        config = getattr(checker, "_cfg", None)
        if config is not None:
            rules.append(plain_dataclass(config))
    return rules


def blocked_reasons_from_repo_events(
    repos_or_repo: Any,
    orders: list[Mapping[str, Any]],
) -> dict[str, list[str]]:
    order_to_strategy = {
        str(order.get("oms_order_id", "")): str(order.get("strategy_id", ""))
        for order in orders
    }
    reasons: dict[str, list[str]] = {}
    for repo in _repos(repos_or_repo):
        for event in getattr(repo, "_events", []) or []:
            if str(event.get("event_type", "")).upper() != "RISK_DENIED":
                continue
            sid = order_to_strategy.get(str(event.get("oms_order_id", "")), "")
            reason = normalize_reason(str((event.get("payload", {}) or {}).get("reason", "")))
            if sid and reason:
                reasons.setdefault(sid, []).append(reason)
    return {sid: sorted(values) for sid, values in sorted(reasons.items())}


async def ledger_from_repo(repo: Any, family_for_strategy) -> list[dict[str, Any]]:
    rows = []
    for fill in sorted(repo._fills.values(), key=lambda item: item.timestamp):
        order = await repo.get_order(fill.oms_order_id)
        if order is None:
            continue
        rows.append(
            {
                "strategy_id": order.strategy_id,
                "family": family_for_strategy(order.strategy_id),
                "symbol": order.instrument.symbol if order.instrument else "",
                "direction": "LONG" if order.side is OrderSide.BUY else "SHORT",
                "qty": fill.qty,
                "entry_time": fill.timestamp,
                "entry_price": fill.price,
                "exit_time": None,
                "exit_price": None,
                "gross_pnl": 0.0,
                "commission": fill.fees,
                "net_pnl": -float(fill.fees or 0.0),
                "exit_reason": "",
                "r_multiple": None,
            }
        )
    return rows


def _services(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return value
    return (value,)


def _repos(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return value
    return (value,)
