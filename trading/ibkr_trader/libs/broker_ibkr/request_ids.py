"""Request and order ID allocation."""
import asyncio


class RequestIdAllocator:
    """Async-safe allocator for IB request and order IDs.

    M5: Uses asyncio.Lock instead of threading.Lock since all callers
    are in an async context.
    """

    def __init__(self, initial_order_id: int = 0):
        self._next_order_id = initial_order_id
        self._next_req_id = 1
        self._lock = asyncio.Lock()

    async def set_next_valid_id(self, order_id: int) -> None:
        """Called when IB sends nextValidId."""
        async with self._lock:
            self._next_order_id = max(self._next_order_id, order_id)

    async def next_order_id(self) -> int:
        async with self._lock:
            oid = self._next_order_id
            self._next_order_id += 1
            return oid

    async def next_request_id(self) -> int:
        async with self._lock:
            rid = self._next_req_id
            self._next_req_id += 1
            return rid
