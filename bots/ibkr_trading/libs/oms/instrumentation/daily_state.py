"""Helpers for family-level daily reconciliation state."""
from __future__ import annotations

from typing import Any

from ._shared import as_float, plain


def _non_empty_dict(value: Any) -> dict:
    data = plain(value)
    return data if isinstance(data, dict) else {}


def _sum_or_single(values: list[float]) -> float:
    numbers = [value for value in values if value != 0.0]
    if not numbers:
        return 0.0
    first = numbers[0]
    if all(abs(value - first) < 1e-9 for value in numbers[1:]):
        return first
    return sum(numbers)


def _merge_targets(current: dict, incoming: dict) -> dict:
    if not incoming:
        return current
    if not current:
        return dict(incoming)

    merged = {**current}
    families = {
        **dict(current.get("families", current.get("family_target_weights", {})) or {}),
        **dict(incoming.get("families", incoming.get("family_target_weights", {})) or {}),
    }
    strategies = {
        **dict(current.get("strategies", current.get("strategy_target_weights", {})) or {}),
        **dict(incoming.get("strategies", incoming.get("strategy_target_weights", {})) or {}),
    }
    if families:
        merged["families"] = families
    if strategies:
        merged["strategies"] = strategies
    active = list(dict.fromkeys(
        list(current.get("active_strategy_ids") or [])
        + list(incoming.get("active_strategy_ids") or [])
    ))
    if active:
        merged["active_strategy_ids"] = active
    for key, value in incoming.items():
        if key not in merged or merged[key] in (None, "", [], {}):
            merged[key] = value
    return merged


def _merge_risk_states(states: list[dict], positions: list[dict]) -> dict:
    if not states:
        return {}

    richest = max(
        states,
        key=lambda state: len(dict(state.get("strategy_daily_pnl") or {})),
    )
    merged = dict(richest)

    strategy_daily_pnl: dict[str, float] = {}
    for state in states:
        for key, value in dict(state.get("strategy_daily_pnl") or {}).items():
            strategy_daily_pnl[str(key)] = as_float(value, 0.0)

    if strategy_daily_pnl:
        merged["strategy_daily_pnl"] = dict(sorted(strategy_daily_pnl.items()))
        merged["daily_realized_pnl"] = sum(strategy_daily_pnl.values())
        cover_counts = [
            len(set(dict(state.get("strategy_daily_pnl") or {})) & set(strategy_daily_pnl))
            for state in states
        ]
        best_idx = max(range(len(states)), key=lambda idx: cover_counts[idx])
        if cover_counts[best_idx] == len(strategy_daily_pnl):
            merged["daily_realized_R"] = as_float(states[best_idx].get("daily_realized_R"), 0.0)
        else:
            merged["daily_realized_R"] = _sum_or_single([
                as_float(state.get("daily_realized_R"), 0.0) for state in states
            ])
    else:
        merged["daily_realized_pnl"] = _sum_or_single([
            as_float(state.get("daily_realized_pnl"), 0.0) for state in states
        ])
        merged["daily_realized_R"] = _sum_or_single([
            as_float(state.get("daily_realized_R"), 0.0) for state in states
        ])

    merged["weekly_realized_pnl"] = _sum_or_single([
        as_float(state.get("weekly_realized_pnl"), 0.0) for state in states
    ])
    merged["weekly_realized_R"] = _sum_or_single([
        as_float(state.get("weekly_realized_R"), 0.0) for state in states
    ])

    open_risk_dollars = sum(as_float(pos.get("open_risk_dollars"), 0.0) for pos in positions)
    position_open_risk_R = sum(as_float(pos.get("open_risk_R"), 0.0) for pos in positions)
    portfolio_open_risk_values = [
        as_float(state.get("open_risk_R"), 0.0)
        for state in states
        if "open_risk_R" in state
    ]
    if open_risk_dollars or positions:
        merged["open_risk_dollars"] = open_risk_dollars
    if portfolio_open_risk_values:
        merged["open_risk_R"] = _sum_or_single(portfolio_open_risk_values)
    elif position_open_risk_R or positions:
        merged["open_risk_R"] = position_open_risk_R

    merged["pending_entry_risk_R"] = _sum_or_single([
        as_float(state.get("pending_entry_risk_R"), 0.0) for state in states
    ])
    merged["halted"] = any(bool(state.get("halted")) for state in states)
    reasons = [
        str(state.get("halt_reason") or "")
        for state in states
        if str(state.get("halt_reason") or "")
    ]
    if reasons:
        merged["halt_reason"] = "; ".join(dict.fromkeys(reasons))
    merged["risk_state_source_count"] = len(states)
    merged["risk_state_aggregation"] = "family_merged"
    return merged


async def collect_family_daily_state(
    services: list,
    *,
    strategy_ids: list[str] | None = None,
    default_strategy_id: str = "",
) -> tuple[dict, dict, dict]:
    """Collect authoritative family positions, risk state, targets, and NAV."""
    requested_strategy_ids = list(strategy_ids or [])
    positions: list[dict] = []
    seen_positions: set[tuple[str, str, str]] = set()
    risk_states: list[dict] = []
    allocation_targets: dict = {}
    account_state: dict = {}

    for idx, oms in enumerate(list(services or [])):
        if oms is None:
            continue

        risk_state = None
        get_portfolio_risk = getattr(oms, "get_portfolio_risk", None)
        if callable(get_portfolio_risk):
            try:
                risk_state = await get_portfolio_risk()
            except Exception:
                risk_state = getattr(oms, "_portfolio_risk_state", {})
        else:
            risk_state = getattr(oms, "_portfolio_risk_state", {})
        risk_payload = _non_empty_dict(risk_state)
        if risk_payload:
            risk_states.append(risk_payload)

        allocation_targets = _merge_targets(
            allocation_targets,
            _non_empty_dict(getattr(oms, "_allocation_targets", {})),
        )

        provider = getattr(oms, "_account_state_provider", None)
        if callable(provider) and not account_state:
            try:
                state = _non_empty_dict(provider())
                if state.get("equity") or state.get("net_liquidation") or state.get("raw_nav"):
                    account_state = state
            except Exception:
                account_state = {}

        repo = getattr(oms, "_oms_repo", None)
        if repo is None:
            continue
        service_strategy_ids = (
            [requested_strategy_ids[idx]]
            if idx < len(requested_strategy_ids)
            else list(getattr(oms, "_family_strategy_ids", []) or [default_strategy_id])
        )
        try:
            repo_positions = await repo.get_positions_for_strategies(service_strategy_ids)
        except Exception:
            repo_positions = []
        for pos in plain(repo_positions):
            if not isinstance(pos, dict):
                continue
            key = (
                str(pos.get("account_id", "")),
                str(pos.get("strategy_id", "")),
                str(pos.get("instrument_symbol", pos.get("symbol", ""))),
            )
            if key in seen_positions:
                continue
            seen_positions.add(key)
            positions.append(pos)

    portfolio_state = _merge_risk_states(risk_states, positions)
    portfolio_state.update(account_state)
    portfolio_state["positions"] = positions
    raw_nav = account_state.get("raw_nav") or account_state.get("equity") or account_state.get("net_liquidation")
    allocated_nav = account_state.get("allocated_nav") or raw_nav
    allocation_state = {
        "source": "daily_closeout",
        "targets": allocation_targets,
        "raw_nav": raw_nav,
        "allocated_nav": allocated_nav,
        "account_state": account_state,
    }
    return portfolio_state, allocation_targets, allocation_state
