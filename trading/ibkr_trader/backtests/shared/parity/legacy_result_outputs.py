from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Iterable

from backtests.shared.parity.trade_outcomes import normalize_trade_outcome_stream


def decision_stream_from_records(
    records: Iterable[Any],
    *,
    timeframe: str = "",
) -> list[dict[str, Any]]:
    stream: list[dict[str, Any]] = []
    for record in records:
        payload = _to_payload(record)
        ts = _coerce_datetime(
            payload.get("ts")
            or payload.get("timestamp")
            or payload.get("time")
            or payload.get("signal_time")
            or payload.get("entry_time")
            or payload.get("entry_ts")
            or payload.get("exit_time")
            or payload.get("exit_ts")
        )
        if ts is None:
            continue
        code = _decision_code_for(payload)
        if not code:
            continue
        symbol = str(payload.get("symbol") or payload.get("instrument") or "")
        details = {
            key: value
            for key, value in payload.items()
            if key not in {"ts", "timestamp", "time", "symbol", "instrument"}
        }
        stream.append(
            {
                "code": code,
                "ts": ts.isoformat(),
                "symbol": symbol,
                "timeframe": timeframe,
                "details": _normalize_value(details),
            }
        )
    return stream


def decision_stream_from_trades(
    trades: Iterable[Any],
    *,
    timeframe: str = "",
    entry_code: str = "ENTRY_FILLED",
    exit_code: str = "EXIT_FILLED",
) -> list[dict[str, Any]]:
    stream: list[dict[str, Any]] = []
    for trade in trades:
        payload = _to_payload(trade)
        symbol = str(payload.get("symbol") or "")
        qty = int(
            payload.get("qty")
            or payload.get("quantity")
            or payload.get("entry_contracts")
            or payload.get("contracts")
            or payload.get("raw_qty")
            or payload.get("portfolio_qty")
            or 0
        )
        entry_price = float(payload.get("entry_price") or payload.get("avg_entry") or 0.0)
        exit_price = float(payload.get("exit_price") or 0.0)
        side = _trade_side(payload)

        decision_ts = _coerce_datetime(
            payload.get("signal_time")
            or payload.get("decision_time")
            or payload.get("setup_time")
        )
        fill_ts = _coerce_datetime(payload.get("fill_time") or payload.get("entry_time") or payload.get("entry_ts"))
        exit_ts = _coerce_datetime(payload.get("exit_time") or payload.get("exit_ts"))

        if fill_ts is not None:
            stream.append(
                {
                    "code": entry_code,
                    "ts": (decision_ts or fill_ts).isoformat(),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "details": _normalize_value(
                        {
                            "entry_price": entry_price,
                            "fill_time": fill_ts,
                            "qty": qty,
                            "side": side,
                            "source": "trade_record",
                        }
                    ),
                }
            )
        if exit_ts is not None:
            stream.append(
                {
                    "code": exit_code,
                    "ts": exit_ts.isoformat(),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "details": _normalize_value(
                        {
                            "exit_price": exit_price,
                            "exit_reason": payload.get("exit_reason") or payload.get("exit_type") or "",
                            "qty": qty,
                            "side": side,
                            "source": "trade_record",
                        }
                    ),
                }
            )
    return stream


def merge_decision_streams(*streams: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = [event for stream in streams for event in stream]
    return sorted(merged, key=lambda event: (event.get("ts", ""), event.get("code", ""), event.get("symbol", "")))


def trade_outcomes_from_records(records: Iterable[Any]) -> list[dict[str, Any]]:
    return normalize_trade_outcome_stream(records)


def _decision_code_for(payload: dict[str, Any]) -> str:
    for key in ("code", "decision_code", "event_type", "event", "state"):
        value = payload.get(key)
        if value:
            return str(value)
    decision = payload.get("decision")
    if decision:
        normalized = str(decision).strip().upper()
        if normalized == "PLACED":
            return "ENTRY_REQUESTED"
        if normalized == "BLOCKED":
            return "SIGNAL_FILTERED"
        return normalized
    if "allowed" in payload:
        return "SIGNAL_ALLOWED" if bool(payload.get("allowed")) else "SIGNAL_FILTERED"
    if "passed_all" in payload:
        return "SIGNAL_ALLOWED" if bool(payload.get("passed_all")) else "SIGNAL_FILTERED"
    blocked_reason = payload.get("blocked_reason") or payload.get("block_reason")
    if blocked_reason:
        return "SIGNAL_FILTERED"
    if payload.get("first_block_reason"):
        return "SIGNAL_FILTERED"
    if payload.get("to_state"):
        return f"FSM_{str(payload['to_state']).upper()}"
    if payload.get("setup_class") and payload.get("entry_stop") is not None:
        return "SETUP_DETECTED"
    if payload.get("filled"):
        return "ENTRY_FILLED"
    if payload.get("expired"):
        return "ORDER_EXPIRED"
    if payload.get("entry_type_selected") or payload.get("campaign_state"):
        return "SIGNAL_EVALUATED"
    return ""


def _trade_side(payload: dict[str, Any]) -> str:
    direction = payload.get("direction") or payload.get("side")
    if isinstance(direction, str):
        upper = direction.upper()
        if upper in {"LONG", "BUY"}:
            return "BUY"
        if upper in {"SHORT", "SELL"}:
            return "SELL"
    if isinstance(direction, (int, float)):
        return "BUY" if direction >= 0 else "SELL"
    return ""


def _to_payload(record: Any) -> dict[str, Any]:
    if is_dataclass(record):
        return asdict(record)
    if isinstance(record, dict):
        return dict(record)
    if hasattr(record, "__dict__"):
        return {
            key: value
            for key, value in vars(record).items()
            if not key.startswith("_")
        }
    return {}


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
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value
