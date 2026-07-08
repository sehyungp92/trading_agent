from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
import math
from pathlib import Path
from typing import Any

from tests.integration.parity.live_shadow_contract import source_fingerprint
from tests.integration.parity.source_inputs import IDLE_MARKET_INPUT_ARTIFACT_KEYS
from tests.integration.parity.source_contract import (
    SCRIPTED_DECISION_KEYS,
    SOURCE_TOP_LEVEL_KEYS,
    SourceContractError,
    validate_contract_table,
    validate_family_alias,
    validate_top_level_source_fields,
)


FIXTURE_SCHEMA_VERSION = 2

_SOURCE_KEYS = SOURCE_TOP_LEVEL_KEYS


class ParityFixtureError(ValueError):
    pass


def _reject_json_constant(value: str) -> None:
    raise ParityFixtureError(f"invalid non-finite JSON value in parity fixture: {value}")


def load_parity_fixture(path: Path | str) -> dict[str, Any]:
    fixture_path = Path(path)
    try:
        payload = json.loads(
            fixture_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ParityFixtureError(f"invalid parity fixture JSON: {fixture_path}: {exc}") from exc

    _validate_fixture(payload, fixture_path)
    return payload


def normalized_source_payload(fixture: Mapping[str, Any]) -> dict[str, Any]:
    try:
        validate_contract_table()
        validate_top_level_source_fields(fixture, Path("<payload>"))
        validate_family_alias(fixture, Path("<payload>"))
    except SourceContractError as exc:
        raise ParityFixtureError(str(exc)) from exc
    _validate_consumed_initial_family_state(fixture, Path("<payload>"))
    payload = {
        key: fixture.get(key)
        for key in _SOURCE_KEYS
        if key in fixture
    }
    return _normalize_value(payload, path="")


def fixture_source_fingerprint(fixture: Mapping[str, Any]) -> str:
    return source_fingerprint(normalized_source_payload(fixture))


def _validate_fixture(payload: Mapping[str, Any], path: Path) -> None:
    if not isinstance(payload, Mapping):
        raise ParityFixtureError(f"{path} fixture root must be an object")
    validate_contract_table()
    try:
        validate_top_level_source_fields(payload, path)
        validate_family_alias(payload, path)
    except SourceContractError as exc:
        raise ParityFixtureError(str(exc)) from exc

    required = {
        "schema_version",
        "surface",
        "family",
        "clock_start",
        "instruments",
        "bars",
        "higher_timeframe_bars",
        "artifacts",
        "strategy_config",
        "family_config",
        "initial_repository_state",
        "initial_strategy_state",
        "initial_family_state",
        "account_state",
        "broker_event_script",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ParityFixtureError(f"{path} missing required field(s): {', '.join(missing)}")

    if payload.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ParityFixtureError(
            f"{path} schema_version={payload.get('schema_version')!r}; "
            f"expected {FIXTURE_SCHEMA_VERSION}"
        )
    if not isinstance(payload.get("instruments"), list) or not payload["instruments"]:
        raise ParityFixtureError(f"{path} must define at least one instrument")
    if not isinstance(payload.get("bars"), list):
        raise ParityFixtureError(f"{path} bars must be a list")
    if not isinstance(payload.get("artifacts"), Mapping):
        raise ParityFixtureError(f"{path} artifacts must be an object")
    for field in (
        "higher_timeframe_bars",
        "strategy_config",
        "family_config",
        "initial_repository_state",
        "initial_strategy_state",
        "initial_family_state",
        "account_state",
    ):
        if field in payload and not isinstance(payload.get(field), Mapping):
            raise ParityFixtureError(f"{path} {field} must be an object")
    if not isinstance(payload.get("broker_event_script"), list):
        raise ParityFixtureError(f"{path} broker_event_script must be a list")
    scripted_path = _find_scripted_decision_field(payload, path="")
    if scripted_path:
        raise ParityFixtureError(f"{path} uses removed scripted decision field: {scripted_path}")
    _validate_no_naive_timestamp_strings(payload, path)
    _validate_source_values_are_normalizable(payload, path)
    _validate_consumed_initial_family_state(payload, path)
    _validate_idle_market_inputs(payload, path)
    for event in payload.get("broker_event_script", []):
        if not isinstance(event, Mapping):
            raise ParityFixtureError(f"{path} broker events must be objects")
        match = event.get("order_match")
        if match is None or not isinstance(match, Mapping):
            raise ParityFixtureError(f"{path} broker events must target orders with order_match")


def _validate_consumed_initial_family_state(payload: Mapping[str, Any], path: Path) -> None:
    family = str(payload.get("family") or (payload.get("family_config", {}) or {}).get("family") or "")
    surface = str(payload.get("surface") or "")
    if family not in {"momentum", "stock", "swing"} or not surface.endswith("_family"):
        return

    state = payload.get("initial_family_state")
    if not isinstance(state, Mapping):
        raise ParityFixtureError(f"{path} initial_family_state must be an object")
    if family in {"momentum", "stock"}:
        if state:
            keys = ", ".join(sorted(str(key) for key in state))
            raise ParityFixtureError(
                f"{path} initial_family_state contains unconsumed Layer-3 source field(s): {keys}"
            )
        return
    unsupported = sorted(str(key) for key in state if str(key) != "overlay")
    if unsupported:
        raise ParityFixtureError(
            f"{path} initial_family_state contains unconsumed Layer-3 source field(s): "
            f"{', '.join(unsupported)}"
        )


def _validate_idle_market_inputs(payload: Mapping[str, Any], path: Path) -> None:
    artifacts = payload.get("artifacts", {}) or {}
    family_cfg = payload.get("family_config", {}) or {}
    for item in family_cfg.get("strategies", []) or []:
        strategy_id = str(item.get("id", ""))
        artifact_key = IDLE_MARKET_INPUT_ARTIFACT_KEYS.get(strategy_id)
        if not artifact_key:
            continue
        artifact = artifacts.get(artifact_key)
        if not isinstance(artifact, Mapping):
            raise ParityFixtureError(
                f"{path} configured {strategy_id} but missing artifacts.{artifact_key}.idle_market_input"
            )
        if "no_order_probe" in artifact:
            raise ParityFixtureError(
                f"{path} uses removed timestamp-only field: artifacts.{artifact_key}.no_order_probe"
            )
        market_input = artifact.get("idle_market_input")
        if not isinstance(market_input, Mapping):
            raise ParityFixtureError(
                f"{path} configured {strategy_id} but missing artifacts.{artifact_key}.idle_market_input"
            )
        missing = {"symbol", "timeframe", "bars"} - set(market_input)
        if missing:
            raise ParityFixtureError(
                f"{path} artifacts.{artifact_key}.idle_market_input missing: {', '.join(sorted(missing))}"
            )
        bars = market_input.get("bars")
        if not isinstance(bars, list) or not bars:
            raise ParityFixtureError(
                f"{path} artifacts.{artifact_key}.idle_market_input.bars must be a non-empty list"
            )


def _normalize_value(value: Any, *, path: str) -> Any:
    if isinstance(value, datetime):
        return _normalize_datetime(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return _normalize_decimal(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ParityFixtureError(f"non-finite fixture source float at {path or '<root>'}: {value!r}")
        return _normalize_decimal(Decimal(str(value)))
    if isinstance(value, str):
        return _normalize_string(value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_value(
                val,
                path=f"{path}.{key}" if path else str(key),
            )
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
            if str(key) not in {"exec_id", "broker_order_id", "perm_id"}
        }
    if isinstance(value, (list, tuple)):
        normalized = [
            _normalize_value(item, path=f"{path}[]")
            for item in value
        ]
        if all(isinstance(item, Mapping) for item in normalized) and not _preserve_list_order(path):
            return sorted(normalized, key=_stable_sort_key)
        return normalized
    raise ParityFixtureError(
        f"unsupported fixture source value at {path or '<root>'}: {type(value).__name__}"
    )


def _preserve_list_order(path: str) -> bool:
    return path == "broker_event_script" or path.endswith(".broker_event_script")


def _normalize_datetime(value: datetime) -> str:
    if _is_naive_datetime(value):
        raise ParityFixtureError("fixture source timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def _normalize_string(value: str) -> str:
    if "T" not in value:
        return value
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return _normalize_datetime(parsed)


def _normalize_decimal(value: Decimal) -> int | float:
    if not value.is_finite():
        raise ParityFixtureError(f"non-finite fixture source decimal: {value!r}")
    quantized = value.quantize(Decimal("0.0000000001")).normalize()
    if quantized == quantized.to_integral():
        return int(quantized)
    return float(quantized)


def _stable_sort_key(value: Mapping[str, Any]) -> str:
    preferred = (
        value.get("surface"),
        value.get("strategy_id"),
        value.get("symbol"),
        value.get("timeframe"),
        value.get("timestamp") or value.get("ts") or value.get("time"),
        value.get("client_order_id"),
        value.get("exec_id"),
    )
    if any(part is not None for part in preferred):
        return "|".join("" if part is None else str(part) for part in preferred)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _find_scripted_decision_field(value: Any, *, path: str) -> str:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_s = str(key)
            current = f"{path}.{key_s}" if path else key_s
            if key_s == "expected_normalized_outputs":
                continue
            if key_s in SCRIPTED_DECISION_KEYS:
                return current
            nested = _find_scripted_decision_field(item, path=current)
            if nested:
                return nested
    if isinstance(value, list):
        for index, item in enumerate(value):
            nested = _find_scripted_decision_field(item, path=f"{path}[{index}]")
            if nested:
                return nested
    return ""


def _validate_source_values_are_normalizable(payload: Mapping[str, Any], path: Path) -> None:
    try:
        normalized_source_payload(payload)
    except ParityFixtureError as exc:
        raise ParityFixtureError(f"{path} invalid source payload: {exc}") from exc


def _validate_no_naive_timestamp_strings(value: Any, path: Path, *, current_path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_s = str(key)
            nested = f"{current_path}.{key_s}" if current_path else key_s
            _validate_no_naive_timestamp_strings(item, path, current_path=nested)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_no_naive_timestamp_strings(item, path, current_path=f"{current_path}[{index}]")
        return
    if not isinstance(value, str) or "T" not in value:
        return
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return
    if _is_naive_datetime(parsed):
        raise ParityFixtureError(
            f"{path} timestamp at {current_path or '<root>'} must be timezone-aware"
        )


def _is_naive_datetime(value: datetime) -> bool:
    return value.tzinfo is None or value.utcoffset() is None
