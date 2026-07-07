"""Tests for async message bus (M1)."""
from __future__ import annotations
import asyncio
from trading_assistant.comms.message_bus import MessageBus
from trading_assistant.comms.bus_events import InboundMessage, OutboundMessage
from trading_assistant.schemas.notifications import NotificationPayload

class TestMessageBus:
    async def test_put_and_get_inbound(self):
        bus = MessageBus()
        msg = InboundMessage(source_channel="telegram", text="hello")
        await bus.put_inbound(msg)
        assert bus.inbound_size() == 1
        got = await asyncio.wait_for(bus.inbound.get(), timeout=1.0)
        assert got.text == "hello"

    async def test_put_and_get_outbound(self):
        bus = MessageBus()
        payload = NotificationPayload(notification_type="test", title="Test")
        msg = OutboundMessage(payload=payload)
        await bus.put_outbound(msg)
        assert bus.outbound_size() == 1
        got = await asyncio.wait_for(bus.outbound.get(), timeout=1.0)
        assert got.payload.title == "Test"

    async def test_start_stop_lifecycle(self):
        bus = MessageBus()
        await bus.start()
        assert bus.is_running
        await bus.stop()
        assert not bus.is_running

    async def test_outbound_consumer(self):
        bus = MessageBus()
        received = []
        async def handler(msg):
            received.append(msg)

        await bus.start(outbound_handler=handler)
        payload = NotificationPayload(notification_type="test", title="Consumed")
        await bus.put_outbound(OutboundMessage(payload=payload))
        await asyncio.sleep(0)  # Let consumer process
        await bus.stop()
        assert len(received) == 1
        assert received[0].payload.title == "Consumed"

    async def test_outbound_consumer_handles_errors(self):
        bus = MessageBus()
        call_count = 0
        async def failing_handler(msg):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("handler error")

        await bus.start(outbound_handler=failing_handler)
        payload = NotificationPayload(notification_type="test")
        await bus.put_outbound(OutboundMessage(payload=payload))
        await asyncio.sleep(0)
        await bus.stop()
        assert call_count == 1  # Handler was called despite error

    async def test_multiple_messages_processed_in_order(self):
        bus = MessageBus()
        received = []
        async def handler(msg):
            received.append(msg.payload.title)

        await bus.start(outbound_handler=handler)
        for i in range(5):
            payload = NotificationPayload(notification_type="test", title=f"msg-{i}")
            await bus.put_outbound(OutboundMessage(payload=payload))
        await asyncio.sleep(0)
        await bus.stop()
        assert received == [f"msg-{i}" for i in range(5)]

    async def test_inbound_priority_message(self):
        bus = MessageBus()
        msg = InboundMessage(source_channel="discord", callback_data="cmd_daily")
        await bus.put_inbound(msg)
        got = await asyncio.wait_for(bus.inbound.get(), timeout=1.0)
        assert got.callback_data == "cmd_daily"
        assert got.source_channel == "discord"

    async def test_stop_without_start(self):
        bus = MessageBus()
        await bus.stop()  # Should not raise
        assert not bus.is_running
