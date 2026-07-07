"""IB Gateway/TWS connection lifecycle management."""
import asyncio
import logging
import random
from typing import Callable, Optional

from ib_async import IB

from libs.config.models import ConnectionGroupConfig

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages IB Gateway/TWS connection with exponential backoff reconnect."""

    RECONNECT_COOLDOWN_S = 300  # 5-min mega-backoff between retry cycles

    def __init__(self, profile: ConnectionGroupConfig):
        self._profile = profile
        self._ib = IB()
        self._connected = asyncio.Event()
        self._shutting_down = False
        self._retry_count = 0
        self._reconnect_task: Optional[asyncio.Task] = None
        # CONN-1: list of post-reconnect callbacks. The previous single-slot
        # design caused later coordinators to silently overwrite earlier
        # registrations — swing's OMS reconciler was lost when stock
        # registered its engine-only callback, and momentum registered
        # nothing at all. Now every family appends and all run on reconnect.
        self._on_reconnect_callbacks: list[Callable] = []

    @property
    def ib(self) -> IB:
        return self._ib

    @property
    def is_connected(self) -> bool:
        return self._ib.isConnected()

    async def connect(self) -> None:
        """Connect with exponential backoff. Sets _connected event when ready."""
        self._ib.disconnectedEvent += self._on_disconnected
        await self._attempt_connect()

    async def _attempt_connect(self) -> None:
        while not self._shutting_down:
            try:
                await self._ib.connectAsync(
                    host=self._profile.host,
                    port=self._profile.port,
                    clientId=self._profile.client_id,
                    readonly=self._profile.readonly,
                )
                self._retry_count = 0
                self._connected.set()
                logger.info(
                    f"Connected to IB Gateway at {self._profile.host}:{self._profile.port}"
                )
                return
            except Exception as e:
                self._retry_count += 1
                if self._retry_count > self._profile.reconnect_max_retries:
                    logger.error(f"Max reconnect attempts reached: {e}")
                    raise
                delay = self._backoff_delay()
                logger.warning(f"Connection failed: {e}. Retrying in {delay:.1f}s")
                await asyncio.sleep(delay)

    def _backoff_delay(self) -> float:
        delay = self._profile.reconnect_base_delay_s * (2 ** (self._retry_count - 1))
        delay = min(delay, self._profile.reconnect_max_delay_s)
        # M5 fix: add random jitter to prevent thundering herd
        delay = delay * (0.5 + random.random())
        return delay

    def add_reconnect_callback(self, callback: Callable) -> None:
        """CONN-1: append a post-reconnect callback. Idempotent — duplicate
        registrations of the same callable are dropped to keep the list
        deterministic across coordinator restarts.
        """
        if callback not in self._on_reconnect_callbacks:
            self._on_reconnect_callbacks.append(callback)

    def set_reconnect_callback(self, callback: Callable) -> None:
        """Deprecated alias for add_reconnect_callback. Kept so existing call
        sites that haven't migrated still work; once they're all migrated
        this can be deleted.
        """
        self.add_reconnect_callback(callback)

    async def disconnect(self) -> None:
        """Graceful disconnect. Cancels any in-flight reconnect loop."""
        self._shutting_down = True
        # Cancel reconnect task so we don't have a zombie sleep running
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
        # Unregister handler to prevent double-registration on re-connect
        self._ib.disconnectedEvent -= self._on_disconnected
        if self._ib.isConnected():
            self._ib.disconnect()
        self._connected.clear()

    async def wait_until_ready(self) -> None:
        """Block until connected."""
        await self._connected.wait()

    def _on_disconnected(self) -> None:
        """Callback: IB fires this on socket drop."""
        self._connected.clear()
        if not self._shutting_down:
            # Guard: don't spawn a second reconnect loop if one is already running
            if self._reconnect_task and not self._reconnect_task.done():
                logger.debug("Reconnect already in progress, ignoring duplicate disconnect")
                return
            logger.warning("Disconnected from IB Gateway, initiating reconnect")
            self._reconnect_task = asyncio.ensure_future(self._reconnect_loop())
            self._reconnect_task.add_done_callback(self._reconnect_done_callback)

    @staticmethod
    def _reconnect_done_callback(task: asyncio.Task) -> None:
        """M7 fix: Handle unhandled exceptions from reconnect loop."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                f"CRITICAL: Reconnect loop failed permanently: {exc}. "
                f"System has NO broker connection. Manual intervention required.",
                exc_info=exc,
            )

    async def _reconnect_loop(self) -> None:
        """Reconnect with exponential backoff. Never gives up unless shutting down."""
        while not self._shutting_down:
            try:
                await self._attempt_connect()
                # CONN-1: invoke every registered callback. One failure must
                # not skip the others — each gets its own try/except so a
                # broken family-level callback can't suppress reconciliation
                # in another family.
                for cb in list(self._on_reconnect_callbacks):
                    try:
                        result = cb()
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(f"Post-reconnect callback failed: {e}")
                if self._on_reconnect_callbacks:
                    logger.info(
                        "Post-reconnect callbacks completed (%d)",
                        len(self._on_reconnect_callbacks),
                    )
                return  # connected successfully
            except Exception as e:
                if self._shutting_down:
                    return
                logger.error(
                    "Reconnect attempts exhausted (%d retries). "
                    "Waiting %ds before next cycle. Error: %s",
                    self._profile.reconnect_max_retries,
                    self.RECONNECT_COOLDOWN_S,
                    e,
                )
                self._retry_count = 0  # reset for next cycle
                await asyncio.sleep(self.RECONNECT_COOLDOWN_S)
