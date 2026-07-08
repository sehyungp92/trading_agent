from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tests.integration.parity.fake_ibkr import FakeIBKRExecutionAdapter


@pytest.mark.asyncio
async def test_fake_ibkr_adapter_drives_registered_callbacks() -> None:
    adapter = FakeIBKRExecutionAdapter(auto_ack=True)
    events: list[tuple[str, str]] = []
    adapter.on_ack = lambda oms_order_id, _ref: events.append(("ack", oms_order_id))
    adapter.on_status = lambda oms_order_id, status, _remaining: events.append(("status", f"{oms_order_id}:{status}"))
    adapter.on_fill = lambda oms_order_id, exec_id, *_args: events.append(("fill", f"{oms_order_id}:{exec_id}"))
    adapter.on_reject = lambda oms_order_id, reason, *_args: events.append(("reject", f"{oms_order_id}:{reason}"))

    ref = await adapter.submit_order(
        oms_order_id="OMS-1",
        contract_symbol="QQQ",
        action="BUY",
        order_type="LIMIT",
        qty=1,
        limit_price=101.0,
        tif="DAY",
    )
    adapter.emit_status(ref.broker_order_id, "Submitted", remaining=1)
    adapter.emit_fill(
        ref.broker_order_id,
        exec_id="EXEC-1",
        price=101.0,
        qty=1,
        fill_time=datetime(2026, 5, 20, 14, 30, tzinfo=timezone.utc),
    )
    adapter.emit_reject(ref.broker_order_id, "late reject")

    assert events == [
        ("ack", "OMS-1"),
        ("status", "OMS-1:Submitted"),
        ("fill", "OMS-1:EXEC-1"),
        ("reject", "OMS-1:late reject"),
    ]
