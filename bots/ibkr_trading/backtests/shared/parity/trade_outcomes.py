from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Iterable


@dataclass(slots=True, frozen=True)
class TradeOutcome:
    symbol: str
    side: str
    qty: int
    decision_ts: datetime | None
    entry_ts: datetime | None
    fill_ts: datetime | None
    exit_ts: datetime | None
    gross_pnl: float
    commission: float
    net_pnl: float
    exit_reason: str
    metadata: dict[str, Any]


def normalize_trade_outcome(record: Any, *, symbol: str | None = None, side: str | None = None) -> TradeOutcome:
    commission = float(_first_attr(record, "commission", "total_commission", default=0.0) or 0.0)
    net_pnl = _net_pnl_for(record, commission=commission)
    gross_pnl = _gross_pnl_for(record, net_pnl=net_pnl, commission=commission)
    quantity = int(
        _first_attr(
            record,
            "qty",
            "quantity",
            "qty_entry",
            "size",
            "entry_contracts",
            "contracts",
            "raw_qty",
            "portfolio_qty",
            default=0,
        )
        or 0
    )
    resolved_symbol = str(symbol or _first_attr(record, "symbol", default=""))
    resolved_side = str(side or _side_for(record))
    decision_ts = _coerce_datetime(
        _first_attr(record, "decision_time", "decision_ts", "signal_time", "setup_time", default=None)
    )
    entry_ts = _coerce_datetime(_first_attr(record, "entry_time", "entry_ts", default=None))
    fill_ts = _coerce_datetime(_first_attr(record, "fill_time", "filled_at", default=entry_ts))
    exit_ts = _coerce_datetime(_first_attr(record, "exit_time", "exit_ts", default=None))
    exit_reason = str(_first_attr(record, "exit_reason", "exit_type", "denial_reason", default=""))

    return TradeOutcome(
        symbol=resolved_symbol,
        side=resolved_side,
        qty=quantity,
        decision_ts=decision_ts,
        entry_ts=entry_ts,
        fill_ts=fill_ts,
        exit_ts=exit_ts,
        gross_pnl=gross_pnl,
        commission=commission,
        net_pnl=net_pnl,
        exit_reason=exit_reason,
        metadata=_normalize_value(_metadata_for(record)),
    )


def normalize_trade_outcome_stream(records: Iterable[Any]) -> list[dict[str, Any]]:
    return [normalize_trade_outcome_dict(record) for record in records]


def normalize_trade_outcome_dict(record: Any, *, symbol: str | None = None, side: str | None = None) -> dict[str, Any]:
    outcome = normalize_trade_outcome(record, symbol=symbol, side=side)
    payload = asdict(outcome)
    return _normalize_value(payload)


def _metadata_for(record: Any) -> dict[str, Any]:
    if is_dataclass(record):
        payload = asdict(record)
    elif isinstance(record, dict):
        payload = dict(record)
    else:
        payload = {
            key: value
            for key, value in vars(record).items()
            if not key.startswith("_")
        } if hasattr(record, "__dict__") else {}
    for key in (
        "symbol",
        "side",
        "direction",
        "qty",
        "quantity",
        "qty_entry",
        "size",
        "pnl",
        "pnl_dollars",
        "net_pnl",
        "gross_pnl",
        "adjusted_pnl",
        "raw_pnl_dollars",
        "commission",
        "adjusted_commission",
        "entry_time",
        "entry_ts",
        "exit_time",
        "exit_ts",
        "fill_time",
        "fill_ts",
        "filled_at",
        "decision_time",
        "decision_ts",
        "signal_time",
        "setup_time",
        "avg_entry",
        "entry_contracts",
        "exit_contracts",
        "raw_qty",
        "portfolio_qty",
        "exit_reason",
        "exit_type",
        "denial_reason",
    ):
        payload.pop(key, None)
    return payload


def _side_for(record: Any) -> str:
    direction = _first_attr(record, "side", "direction", default="")
    if isinstance(direction, str):
        upper = direction.upper()
        if upper in {"BUY", "SELL"}:
            return upper
        if upper in {"LONG", "SHORT"}:
            return "BUY" if upper == "LONG" else "SELL"
    if hasattr(direction, "value"):
        return _side_for({"direction": direction.value})
    if isinstance(direction, (int, float)):
        return "BUY" if direction >= 0 else "SELL"
    return ""


def _net_pnl_for(record: Any, *, commission: float) -> float:
    if hasattr(record, "net_pnl"):
        return float(record.net_pnl)
    if hasattr(record, "pnl_dollars"):
        return float(record.pnl_dollars)
    if hasattr(record, "adjusted_pnl"):
        return float(record.adjusted_pnl)
    if hasattr(record, "raw_pnl_dollars"):
        return float(record.raw_pnl_dollars)
    if hasattr(record, "pnl"):
        return float(record.pnl)
    if isinstance(record, dict):
        for key in ("net_pnl", "pnl_dollars", "adjusted_pnl", "raw_pnl_dollars", "pnl"):
            if key in record:
                return float(record[key])
    return 0.0


def _gross_pnl_for(record: Any, *, net_pnl: float, commission: float) -> float:
    if hasattr(record, "gross_pnl"):
        return float(record.gross_pnl)
    if isinstance(record, dict) and "gross_pnl" in record:
        return float(record["gross_pnl"])
    return net_pnl + commission


def _first_attr(record: Any, *names: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        for name in names:
            if name in record:
                return record[name]
        return default
    for name in names:
        if hasattr(record, name):
            return getattr(record, name)
    return default


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        return datetime.fromisoformat(value)
    return None


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _normalize_value(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _normalize_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value
