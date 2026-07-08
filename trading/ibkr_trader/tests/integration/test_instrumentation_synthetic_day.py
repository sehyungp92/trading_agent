from __future__ import annotations

import pytest

from tests.unit.test_synthetic_day_instrumentation import run_family_synthetic_instrumentation_chain


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["stock", "momentum", "swing"])
async def test_instrumentation_synthetic_day_end_to_end(tmp_path, monkeypatch, family: str) -> None:
    sent = await run_family_synthetic_instrumentation_chain(
        tmp_path / family,
        monkeypatch,
        family,
    )
    event_types = {event["event_type"] for event in sent}

    assert {
        "deployment",
        "config_snapshot",
        "portfolio_rule_check",
        "risk_decision",
        "risk_halt",
        "missed_opportunity",
        "filter_decision",
        "coordinator_action",
        "heartbeat",
        "error",
    }.issubset(event_types)
    assert len(sent) >= (15 if family == "stock" else 10)
