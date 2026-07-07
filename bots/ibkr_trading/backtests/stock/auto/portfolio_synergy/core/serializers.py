from __future__ import annotations

from collections import deque
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any

from backtests.stock.models import Direction

from .state import BlockedCandidate, PortfolioCoreState, PortfolioPosition

_STATE_VERSION = 1


def snapshot_portfolio_state(state: PortfolioCoreState) -> dict[str, Any]:
    return {
        "schema": "stock_portfolio_core_state",
        "version": _STATE_VERSION,
        "state": _to_payload(state),
    }


def hydrate_portfolio_state(snapshot: dict[str, Any]) -> PortfolioCoreState:
    payload = snapshot.get("state", snapshot)
    state = PortfolioCoreState(
        equity=float(payload.get("equity", 0.0)),
        peak_equity=float(payload.get("peak_equity", payload.get("equity", 0.0))),
        reference_risk_pct=float(payload.get("reference_risk_pct", 0.0)),
        active_positions=[
            _hydrate_position(item) for item in payload.get("active_positions", [])
        ],
        accepted_positions=[
            _hydrate_position(item) for item in payload.get("accepted_positions", [])
        ],
        blocked_candidates=[
            _hydrate_blocked_candidate(item)
            for item in payload.get("blocked_candidates", [])
        ],
        equity_points=[float(value) for value in payload.get("equity_points", [])],
        equity_times=[
            _hydrate_datetime(value) for value in payload.get("equity_times", [])
        ],
        daily_realized_r={
            str(key): float(value)
            for key, value in payload.get("daily_realized_r", {}).items()
        },
        weekly_realized_r={
            str(key): float(value)
            for key, value in payload.get("weekly_realized_r", {}).items()
        },
        strategy_recent={
            str(key): _hydrate_deque(value)
            for key, value in payload.get("strategy_recent", {}).items()
        },
        risk_by_strategy={
            str(key): float(value)
            for key, value in payload.get("risk_by_strategy", {}).items()
        },
        candidate_count=int(payload.get("candidate_count", 0)),
        decision_seq=int(payload.get("decision_seq", 0)),
    )
    return state


def _to_payload(value: Any) -> Any:
    if isinstance(value, datetime):
        return {"__type__": "datetime", "value": value.isoformat()}
    if isinstance(value, Direction):
        return {"__type__": "Direction", "value": int(value)}
    if isinstance(value, IntEnum):
        return {"__type__": value.__class__.__name__, "value": int(value)}
    if isinstance(value, Enum):
        return {"__type__": value.__class__.__name__, "value": value.value}
    if isinstance(value, deque):
        return {
            "__type__": "deque",
            "maxlen": value.maxlen,
            "values": [_to_payload(item) for item in value],
        }
    if is_dataclass(value):
        return {
            field.name: _to_payload(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _to_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_payload(item) for item in value]
    return value


def _hydrate_position(payload: dict[str, Any]) -> PortfolioPosition:
    return PortfolioPosition(
        strategy=str(payload["strategy"]),
        symbol=str(payload["symbol"]),
        sector=str(payload.get("sector", "")),
        direction=_hydrate_direction(payload.get("direction", Direction.FLAT)),
        entry_time=_hydrate_datetime(payload["entry_time"]),
        decision_time=_hydrate_datetime(payload.get("decision_time", payload["entry_time"])),
        fill_time=_hydrate_datetime(payload.get("fill_time", payload["entry_time"])),
        exit_time=_hydrate_datetime(payload["exit_time"]),
        risk_dollars=float(payload.get("risk_dollars", 0.0)),
        pnl=float(payload.get("pnl", 0.0)),
        r_multiple=float(payload.get("r_multiple", 0.0)),
        quality=float(payload.get("quality", 0.0)),
        entry_price=float(payload.get("entry_price", 0.0)),
        exit_price=float(payload.get("exit_price", 0.0)),
        quantity=float(payload.get("quantity", 0.0)),
        price_scale=float(payload.get("price_scale", 0.0)),
        commission=float(payload.get("commission", 0.0)),
        exit_reason=str(payload.get("exit_reason", "")),
        entry_type=str(payload.get("entry_type", "")),
        metadata=_hydrate_plain(payload.get("metadata", {})),
    )


def _hydrate_blocked_candidate(payload: dict[str, Any]) -> BlockedCandidate:
    return BlockedCandidate(
        strategy=str(payload["strategy"]),
        symbol=str(payload["symbol"]),
        sector=str(payload.get("sector", "")),
        entry_time=_hydrate_datetime(payload["entry_time"]),
        r_multiple=float(payload.get("r_multiple", 0.0)),
        reason=str(payload.get("reason", "")),
        quality=float(payload.get("quality", 0.0)),
        heat_r=float(payload.get("heat_r", 0.0)),
    )


def _hydrate_plain(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__type__") == "datetime":
            return _hydrate_datetime(value)
        if value.get("__type__") == "Direction":
            return _hydrate_direction(value)
        if value.get("__type__") == "deque":
            return _hydrate_deque(value)
        return {key: _hydrate_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_hydrate_plain(item) for item in value]
    return value


def _hydrate_deque(payload: Any) -> deque[float]:
    if isinstance(payload, dict):
        values = [_hydrate_plain(item) for item in payload.get("values", [])]
        return deque((float(item) for item in values), maxlen=payload.get("maxlen"))
    return deque(float(item) for item in payload)


def _hydrate_direction(payload: Any) -> Direction:
    if isinstance(payload, dict):
        payload = payload.get("value", 0)
    try:
        return Direction(int(payload))
    except (TypeError, ValueError):
        return Direction.FLAT


def _hydrate_datetime(payload: Any) -> datetime:
    if isinstance(payload, datetime):
        return payload
    if isinstance(payload, dict):
        payload = payload.get("value")
    if not isinstance(payload, str):
        raise TypeError(f"Cannot hydrate datetime from {type(payload).__name__}")
    return datetime.fromisoformat(payload)
