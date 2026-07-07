from __future__ import annotations

import json
import re
from collections.abc import Mapping

import pytest

from libs.instrumentation.event_contract import REQUIRED_SECTION_6_2_FIELDS
from tests.unit.test_synthetic_day_instrumentation import run_family_synthetic_instrumentation_chain


_RAW_ACCOUNT_PATTERNS = (
    re.compile(r"\bU\d{5,}\b"),
    re.compile(r"\bDU\d{5,}\b"),
    re.compile(r"\bACCT-[A-Z0-9_-]+\b"),
)
_SECRET_KEY_FRAGMENTS = ("secret", "password", "token", "api_key", "apikey", "private_key", "hmac")
_MAX_PAYLOAD_BYTES = 50_000


def _walk(value):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key), item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _contains_raw_account(value) -> bool:
    if isinstance(value, str):
        return any(pattern.search(value) for pattern in _RAW_ACCOUNT_PATTERNS)
    if isinstance(value, Mapping):
        return any(_contains_raw_account(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_raw_account(item) for item in value)
    return False


def _allows_empty(payload: Mapping, field: str, outer: Mapping | None = None) -> bool:
    if field not in {"strategy_id", "strategy_version"}:
        return False
    outer = outer or payload
    return (
        payload.get("scope") in {"portfolio", "family"}
        or (outer.get("event_type") == "risk_halt" and outer.get("halt_scope") == "portfolio")
    )


def _present(payload: Mapping, field: str, outer: Mapping | None = None) -> bool:
    if field not in payload:
        return False
    if _allows_empty(payload, field, outer):
        return payload.get(field) is not None
    return payload.get(field) not in ("", None, [], {})


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["stock", "momentum", "swing"])
async def test_synthetic_day_contract_scan(tmp_path, monkeypatch, family: str) -> None:
    sent = await run_family_synthetic_instrumentation_chain(
        tmp_path / family,
        monkeypatch,
        family,
    )
    if family == "stock":
        from strategies.stock.instrumentation.src import sidecar as sidecar_module
    elif family == "momentum":
        from strategies.momentum.instrumentation.src import sidecar as sidecar_module
    else:
        from strategies.swing.instrumentation.src import sidecar as sidecar_module
    mapped_event_types = (
        set(sidecar_module._DIR_TO_EVENT_TYPE.values())
        | set(getattr(sidecar_module, "_EVENT_PRIORITY", {}))
        | set(getattr(sidecar_module, "_PRIORITY_MAP", {}))
    )

    assert sent
    for envelope in sent:
        assert envelope["event_type"] in mapped_event_types
        assert envelope["event_id"]
        assert envelope["bot_id"]
        assert envelope["exchange_timestamp"]
        assert "payload" in envelope
        assert len(envelope["payload"].encode("utf-8")) <= _MAX_PAYLOAD_BYTES

        payload = json.loads(envelope["payload"])
        assert not _contains_raw_account(payload)
        for key, value in _walk(payload):
            key_lower = key.lower()
            if any(fragment in key_lower for fragment in _SECRET_KEY_FRAGMENTS):
                assert value in ("", None, "<redacted>", [], {})
        assert "lineage_gaps" not in payload
        assert isinstance(payload.get("lineage"), Mapping)
        for field in REQUIRED_SECTION_6_2_FIELDS:
            assert _present(payload, field), (envelope["event_type"], field, "top-level")
            assert _present(payload["lineage"], field, payload), (envelope["event_type"], field, "lineage")
            assert payload.get(field) == payload["lineage"].get(field), (
                envelope["event_type"],
                field,
                payload.get(field),
                payload["lineage"].get(field),
            )
        assert payload["event_type"] == envelope["event_type"]
