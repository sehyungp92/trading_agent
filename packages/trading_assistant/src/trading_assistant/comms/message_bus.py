from __future__ import annotations
import asyncio
import logging
from trading_assistant.comms.bus_events import InboundMessage, OutboundMessage

logger = logging.getLogger(__name__)

class MessageBus:
    """Async message bus decoupling channels from processing."""

    def __init__(self, maxsize: int = 100) -> None:
        self._inbound: asyncio.Queue[InboundMessage] = asyncio.Queue(maxsize=maxsize)
        self._outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue(maxsize=maxsize)
        self._running = False
        self._consumer_task: asyncio.Task | None = None

    @property
    def inbound(self) -> asyncio.Queue[InboundMessage]:
        return self._inbound

    @property
    def outbound(self) -> asyncio.Queue[OutboundMessage]:
        return self._outbound

    @property
    def is_running(self) -> bool:
        return self._running

    async def put_inbound(self, msg: InboundMessage) -> None:
        await self._inbound.put(msg)

    async def put_outbound(self, msg: OutboundMessage) -> None:
        await self._outbound.put(msg)

    async def start(self, outbound_handler=None) -> None:
        """Start the bus. If outbound_handler is provided, start a consumer loop."""
        self._running = True
        if outbound_handler:
            self._consumer_task = asyncio.create_task(
                self._consume_outbound(outbound_handler)
            )

    async def stop(self) -> None:
        self._running = False
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None

    async def _consume_outbound(self, handler) -> None:
        """Drain outbound queue and call handler for each message."""
        while self._running:
            try:
                msg = await asyncio.wait_for(self._outbound.get(), timeout=1.0)
                try:
                    await handler(msg)
                except Exception:
                    logger.exception("Outbound handler failed for message")
            except asyncio.TimeoutError:
                continue

    def inbound_size(self) -> int:
        return self._inbound.qsize()

    def outbound_size(self) -> int:
        return self._outbound.qsize()
