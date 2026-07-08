from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class ParityTrace:
    producer: str
    source_fingerprint: str
    order_intents: list[dict[str, Any]] = field(default_factory=list)
    terminal_events: list[dict[str, Any]] = field(default_factory=list)
    trade_ledger: list[dict[str, Any]] = field(default_factory=list)
    state_snapshot: dict[str, Any] | None = None


@dataclass(frozen=True)
class LiveShadowContract:
    surface: str
    live: ParityTrace
    replay: ParityTrace


@dataclass(frozen=True)
class FamilyShadowContract:
    family: str
    children: list[LiveShadowContract]
    live_family_state: dict[str, Any] | None = None
    replay_family_state: dict[str, Any] | None = None


def assert_shadow_contract(contract: LiveShadowContract) -> None:
    """Assert equality between independently produced live and replay traces."""
    _assert_equal(contract, "source_fingerprint", contract.live.source_fingerprint, contract.replay.source_fingerprint)
    _assert_equal(contract, "order_intents", contract.live.order_intents, contract.replay.order_intents)
    _assert_equal(contract, "terminal_events", contract.live.terminal_events, contract.replay.terminal_events)
    _assert_equal(contract, "trade_ledger", contract.live.trade_ledger, contract.replay.trade_ledger)
    if contract.live.state_snapshot is not None or contract.replay.state_snapshot is not None:
        _assert_equal(contract, "state_snapshot", contract.live.state_snapshot, contract.replay.state_snapshot)


def assert_family_shadow_contract(
    contract: FamilyShadowContract,
    *,
    expected_trades: int,
    expected_surfaces: set[str] | None = None,
) -> None:
    if expected_surfaces is not None:
        actual_surfaces = [child.surface for child in contract.children]
        assert sorted(actual_surfaces) == sorted(expected_surfaces), (
            "family child contract mismatch: expected "
            f"{sorted(expected_surfaces)}, got {sorted(actual_surfaces)}"
        )
    for child in contract.children:
        assert_shadow_contract(child)
    live = [row for child in contract.children for row in child.live.trade_ledger]
    replay = [row for child in contract.children for row in child.replay.trade_ledger]
    assert len(live) == expected_trades, (
        f"family merged ledger trade-count mismatch: expected {expected_trades}, got {len(live)}"
    )
    assert live == replay, "family merged ledger mismatch between live and replay traces"
    if contract.live_family_state is not None or contract.replay_family_state is not None:
        assert contract.live_family_state == contract.replay_family_state, (
            f"{contract.family} family state mismatch between live and replay traces"
        )


def assert_merged_family_ledger(
    contracts: list[LiveShadowContract],
    *,
    expected_trades: int,
    expected_surfaces: set[str] | None = None,
) -> None:
    assert_family_shadow_contract(
        FamilyShadowContract(family="family", children=contracts),
        expected_trades=expected_trades,
        expected_surfaces=expected_surfaces,
    )


def source_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        normalize_fingerprint_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_fingerprint_payload(value: Any) -> Any:
    if is_dataclass(value):
        return normalize_fingerprint_payload(asdict(value))
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise TypeError(f"unsupported non-finite source fingerprint decimal: {value!r}")
        quantized = value.quantize(Decimal("0.0000000001")).normalize()
        return int(quantized) if quantized == quantized.to_integral() else float(quantized)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): normalize_fingerprint_payload(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple):
        return [normalize_fingerprint_payload(item) for item in value]
    if isinstance(value, list):
        return [normalize_fingerprint_payload(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (normalize_fingerprint_payload(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"unsupported non-finite source fingerprint float: {value!r}")
        return value
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    raise TypeError(f"unsupported source fingerprint value: {type(value).__name__}")


def _assert_equal(contract: LiveShadowContract, field_name: str, live_value: Any, replay_value: Any) -> None:
    assert live_value == replay_value, (
        f"{contract.surface} {field_name} mismatch between "
        f"{contract.live.producer} and {contract.replay.producer}"
    )
