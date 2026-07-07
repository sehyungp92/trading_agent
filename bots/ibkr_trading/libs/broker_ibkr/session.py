"""Unified multi-group IBKR session management."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from ib_async import IB

from libs.config.models import ConnectionGroupConfig
from libs.config.completed_bar_policy import filter_completed_live_bars

from .connection import ConnectionManager
from .farm_monitor import FarmMonitor
from .heartbeat import HeartbeatMonitor
from .request_ids import RequestIdAllocator
from .throttler import GlobalThrottler, PacingChannel

logger = logging.getLogger(__name__)

_HISTORICAL_TIMEOUT_WINDOW_S = 180.0
_HISTORICAL_BREAKER_OPEN_S = 120.0
_HISTORICAL_SKIP_LOG_INTERVAL_S = 30.0


@dataclass
class ConnectionGroup:
    """One IBKR client connection for a group of strategies."""

    group_id: str
    config: ConnectionGroupConfig
    conn: ConnectionManager
    ids: RequestIdAllocator = field(default_factory=RequestIdAllocator)
    heartbeat: HeartbeatMonitor | None = None
    farm_monitor: FarmMonitor | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    farm_recovery_callbacks: list[Callable[[str], None]] = field(default_factory=list)


class UnifiedIBSession:
    """Owns the runtime's set of IBKR connection groups."""

    def __init__(
        self,
        connection_groups: dict[str, ConnectionGroupConfig],
        strategy_group_map: dict[str, str],
    ):
        self._groups = {
            group_id: ConnectionGroup(
                group_id=group_id,
                config=group_config,
                conn=ConnectionManager(group_config),
            )
            for group_id, group_config in connection_groups.items()
        }
        self._strategy_group_map = dict(strategy_group_map)
        first_group = next(iter(connection_groups.values()), None)
        global_rate = first_group.pacing_messages_per_sec if first_group else 45.0
        order_rate = first_group.pacing_orders_per_sec if first_group else 5.0
        self._throttler = GlobalThrottler(
            global_msg_per_sec=global_rate,
            orders_per_sec=order_rate,
        )
        self._historical_recent: deque[float] = deque()
        self._historical_timeouts: deque[float] = deque()
        self._historical_timeout_count = 0
        self._historical_skipped_count = 0
        self._historical_breaker_open_until = 0.0
        self._historical_breaker_open_until_wall: datetime | None = None
        self._historical_last_skip_log_at = 0.0

    @property
    def groups(self) -> dict[str, ConnectionGroup]:
        return self._groups

    @property
    def ib(self) -> IB:
        """First group's IB instance — all coordinators assume single group."""
        first = next(iter(self._groups.values()), None)
        if first is None:
            raise RuntimeError("No connection groups configured")
        return first.conn.ib

    @property
    def is_ready(self) -> bool:
        return all(g.ready.is_set() for g in self._groups.values())

    @property
    def is_congested(self) -> bool:
        return self._throttler.is_congested

    async def start(self) -> None:
        """Connect all groups with staggered startup. Cleans up on partial failure."""
        try:
            for group in self._groups.values():
                logger.info(
                    "Connecting group %s to %s:%s client_id=%s",
                    group.group_id,
                    group.config.host,
                    group.config.port,
                    group.config.client_id,
                )
                # Capture 321 (Read-Only API) warnings during connect/sync
                readonly_warnings: list[str] = []

                def _capture_readonly(reqId, errorCode, errorString, contract):
                    if errorCode == 321:
                        readonly_warnings.append(errorString)

                group.conn.ib.errorEvent += _capture_readonly
                await group.conn.connect()
                await group.conn.wait_until_ready()
                group.conn.ib.errorEvent -= _capture_readonly

                if readonly_warnings:
                    logger.error(
                        "IB Gateway API is in READ-ONLY mode -- order placement will fail. "
                        "Fix: set ReadOnlyApi=no and ReadOnlyLogin=no in "
                        "/opt/ibc/config/config.ini and restart IB Gateway."
                    )

                mdt = group.config.market_data_type
                group.conn.ib.reqMarketDataType(mdt)
                if mdt != 1:
                    logger.info("Market data type set to %d for group %s", mdt, group.group_id)
                await group.ids.set_next_valid_id(group.conn.ib.client.getReqId())
                group.heartbeat = HeartbeatMonitor(group.conn.ib)
                await group.heartbeat.start()
                group.farm_monitor = FarmMonitor(group.conn.ib)
                group.farm_monitor.on_farm_recovered = lambda farm, gid=group.group_id: (
                    self._dispatch_farm_recovery(gid, farm)
                )
                group.farm_monitor.start()
                group.ready.set()
                await asyncio.sleep(1.0)
        except Exception:
            logger.error("Partial startup failure -- cleaning up already-started groups")
            await self.stop()
            raise

    async def verify_streaming_data(self, test_symbol: str = "SPY") -> None:
        """Preflight: subscribe to a test symbol and verify market data flows.

        - Tries the configured market_data_type first (default: real-time).
        - If real-time fails with 10089/10189, falls back to delayed (type 3).
        - If delayed works, logs a WARNING but allows startup to proceed.
        - If both fail, raises RuntimeError with troubleshooting steps.
        - Set env SKIP_STREAMING_CHECK=1 to bypass entirely.
        """
        if os.environ.get("SKIP_STREAMING_CHECK", "").lower() in ("1", "true", "yes"):
            logger.warning("SKIP_STREAMING_CHECK set -- skipping market data verification")
            return

        from ib_async import Stock

        ib = self.ib
        contract = Stock(test_symbol, "SMART", "USD")
        qualified = await ib.qualifyContractsAsync(contract)
        if not qualified or qualified[0] is None:
            raise RuntimeError(
                f"Could not qualify test symbol {test_symbol} -- "
                "check IB Gateway connectivity"
            )

        test_contract = qualified[0]
        errors: list[tuple[int, str]] = []

        def _capture(reqId, errorCode, errorString, contract):
            if errorCode in (10089, 10189):
                errors.append((errorCode, errorString))

        ib.errorEvent += _capture
        try:
            ib.reqMktData(test_contract, "", False, False)
            await asyncio.sleep(3.0)  # IBKR returns 10089 within ~1s
            ib.cancelMktData(test_contract)
        finally:
            ib.errorEvent -= _capture

        if not errors:
            logger.info(
                "Streaming market data verified OK (test symbol: %s)", test_symbol,
            )
            return

        # Real-time failed -- try delayed data as fallback
        rt_code, rt_msg = errors[0]
        logger.warning(
            "Real-time streaming unavailable for %s (error %d), "
            "trying delayed data fallback...",
            test_symbol, rt_code,
        )
        errors.clear()
        ib.reqMarketDataType(3)  # delayed
        ib.errorEvent += _capture
        try:
            ib.reqMktData(test_contract, "", False, False)
            await asyncio.sleep(3.0)
            ib.cancelMktData(test_contract)
        finally:
            ib.errorEvent -= _capture
            # Restore configured market data type
            first_group = next(iter(self._groups.values()), None)
            if first_group:
                ib.reqMarketDataType(first_group.config.market_data_type)

        if errors:
            raise RuntimeError(
                f"No market data available for {test_symbol} "
                f"(IBKR error {rt_code}). "
                "Troubleshooting:\n"
                "  1. IBKR Account Management -> Settings -> Market Data Connections\n"
                "     Ensure 'Market data for use with the API' is enabled\n"
                "  2. Verify US exchange subscriptions (NASDAQ Network C, NYSE Arca)\n"
                "     or 'US Securities Snapshot and Futures Value Bundle'\n"
                "  3. Paper account: confirm 'Share real-time market data "
                "subscriptions with paper trading account' is Yes\n"
                "  4. Restart IB Gateway after any subscription changes\n"
                f"  Raw error: {rt_msg}"
            )
        logger.warning(
            "DEGRADED: Only delayed market data available for %s. "
            "Real-time streaming requires API market data subscription. "
            "Strategies will use delayed data until subscription is configured.",
            test_symbol,
        )

    async def stop(self) -> None:
        """Disconnect all groups in reverse order. Continues on per-group errors."""
        first_error: Exception | None = None
        for group in reversed(list(self._groups.values())):
            try:
                if group.farm_monitor:
                    group.farm_monitor.stop()
                if group.heartbeat:
                    await group.heartbeat.stop()
                await group.conn.disconnect()
                group.ready.clear()
            except Exception as exc:
                logger.error("Error stopping group %s: %s", group.group_id, exc, exc_info=True)
                group.ready.clear()
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _resolve_group(self, strategy_id: str) -> ConnectionGroup:
        """Map strategy_id → ConnectionGroup, raising ValueError on unknown IDs."""
        try:
            group_id = self._strategy_group_map[strategy_id]
        except KeyError:
            raise ValueError(
                f"Unknown strategy_id {strategy_id!r}. "
                f"Registered: {sorted(self._strategy_group_map)}"
            ) from None
        return self._groups[group_id]

    def _get_group(self, group_id: str) -> ConnectionGroup:
        """Resolve group_id → ConnectionGroup, raising ValueError on unknown IDs."""
        try:
            return self._groups[group_id]
        except KeyError:
            raise ValueError(
                f"Unknown group_id {group_id!r}. "
                f"Available: {sorted(self._groups)}"
            ) from None

    async def wait_ready(self, group_id: str | None = None) -> None:
        if group_id is None:
            for group in self._groups.values():
                await group.ready.wait()
            return
        await self._get_group(group_id).ready.wait()

    def get_ib(self, strategy_id: str) -> IB:
        return self._resolve_group(strategy_id).conn.ib

    def strategy_group(self, strategy_id: str) -> str:
        return self._resolve_group(strategy_id).group_id

    async def next_order_id(self, strategy_id: str) -> int:
        return await self._resolve_group(strategy_id).ids.next_order_id()

    async def next_request_id(self, strategy_id: str) -> int:
        return await self._resolve_group(strategy_id).ids.next_request_id()

    async def throttled(self, channel: PacingChannel) -> None:
        """Acquire a pacing token. Raises CongestionError if queue is overloaded."""
        await self._throttler.acquire(channel)

    async def req_historical_data(
        self,
        contract: Any,
        endDateTime: Any = "",
        durationStr: str = "",
        barSizeSetting: str = "",
        whatToShow: str = "TRADES",
        useRTH: bool = False,
        formatDate: int = 1,
        keepUpToDate: bool = False,
        chartOptions: list[Any] | None = None,
        timeout: float | None = None,
        request_kind: str = "recurring",
        completed_only: bool = False,
        as_of: datetime | None = None,
    ) -> Any:
        """Paced, bounded historical-data request with a short farm-hiccup breaker."""
        timeout = self._historical_timeout(timeout, request_kind, keepUpToDate)
        now = time.monotonic()
        self._prune_historical_window(self._historical_recent, now)
        self._historical_recent.append(now)

        if not keepUpToDate and now < self._historical_breaker_open_until:
            self._historical_skipped_count += 1
            self._log_historical_skip(now, contract, durationStr, barSizeSetting)
            return []

        await self.throttled(PacingChannel.HISTORICAL)
        start = time.monotonic()
        try:
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime=endDateTime,
                durationStr=durationStr,
                barSizeSetting=barSizeSetting,
                whatToShow=whatToShow,
                useRTH=useRTH,
                formatDate=formatDate,
                keepUpToDate=keepUpToDate,
                chartOptions=chartOptions or [],
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            self._record_historical_timeout(time.monotonic())
            logger.warning(
                "Historical data request timed out: %s %s %s",
                getattr(contract, "symbol", contract),
                durationStr,
                barSizeSetting,
            )
            return []
        elapsed = time.monotonic() - start
        if (
            timeout > 0
            and not bars
            and elapsed >= max(timeout * 0.9, timeout - 1.0)
        ):
            self._record_historical_timeout(time.monotonic())
        if bars and completed_only and not keepUpToDate:
            bars = self.filter_completed_historical_bars(
                bars,
                bar_size_setting=barSizeSetting,
                useRTH=useRTH,
                endDateTime=endDateTime,
                as_of=as_of,
            )
        return bars

    @staticmethod
    def filter_completed_historical_bars(
        bars: Any,
        *,
        bar_size_setting: str,
        useRTH: bool,
        endDateTime: Any = "",
        as_of: datetime | None = None,
    ) -> list[Any]:
        return filter_completed_live_bars(
            bars,
            bar_size_setting=bar_size_setting,
            use_rth=useRTH,
            end_datetime=endDateTime,
            as_of=as_of,
        )

    def historical_health(self) -> dict[str, Any]:
        now = time.monotonic()
        self._prune_historical_window(self._historical_recent, now)
        self._prune_historical_window(self._historical_timeouts, now)
        return {
            "recent_count": len(self._historical_recent),
            "timeout_count": len(self._historical_timeouts),
            "total_timeout_count": self._historical_timeout_count,
            "breaker_open_until": (
                self._historical_breaker_open_until_wall.isoformat()
                if self._historical_breaker_open_until > now
                and self._historical_breaker_open_until_wall is not None
                else None
            ),
            "skipped_count": self._historical_skipped_count,
        }

    @staticmethod
    def _historical_timeout(
        timeout: float | None, request_kind: str, keep_up_to_date: bool
    ) -> float:
        if timeout is not None:
            return timeout
        if keep_up_to_date or request_kind in {"startup", "backfill", "subscription"}:
            return 45.0
        if request_kind == "quick":
            return 15.0
        return 20.0

    @staticmethod
    def _prune_historical_window(events: deque[float], now: float) -> None:
        cutoff = now - _HISTORICAL_TIMEOUT_WINDOW_S
        while events and events[0] < cutoff:
            events.popleft()

    def _log_historical_skip(
        self, now: float, contract: Any, duration: str, bar_size: str
    ) -> None:
        if now - self._historical_last_skip_log_at < _HISTORICAL_SKIP_LOG_INTERVAL_S:
            return
        self._historical_last_skip_log_at = now
        remaining = max(0, int(self._historical_breaker_open_until - now))
        logger.warning(
            "Skipping historical data request while breaker is open (%ss remaining): %s %s %s",
            remaining,
            getattr(contract, "symbol", contract),
            duration,
            bar_size,
        )

    def _record_historical_timeout(self, now: float) -> None:
        self._prune_historical_window(self._historical_timeouts, now)
        self._historical_timeouts.append(now)
        self._historical_timeout_count += 1
        if len(self._historical_timeouts) >= 3:
            was_open = now < self._historical_breaker_open_until
            self._historical_breaker_open_until = max(
                self._historical_breaker_open_until,
                now + _HISTORICAL_BREAKER_OPEN_S,
            )
            self._historical_breaker_open_until_wall = (
                datetime.now(timezone.utc)
                + timedelta(seconds=self._historical_breaker_open_until - now)
            )
            if not was_open:
                logger.warning(
                    "Historical data breaker opened for %ds after %d timeouts in %ds",
                    int(_HISTORICAL_BREAKER_OPEN_S),
                    len(self._historical_timeouts),
                    int(_HISTORICAL_TIMEOUT_WINDOW_S),
                )

    def register_farm_recovery_callback(self, group_id: str, cb: Callable[[str], None]) -> None:
        self._get_group(group_id).farm_recovery_callbacks.append(cb)

    def add_reconnect_callback(self, group_id_or_callback, callback: Callable | None = None) -> None:
        """CONN-1: append a post-reconnect callback. Multiple families can
        each add their own callback and all will fire after a successful
        reconnect.

        Supports two calling conventions:
          - add_reconnect_callback(group_id, callback)  — target specific group
          - add_reconnect_callback(callback)             — broadcast to all groups
        """
        if callback is None and callable(group_id_or_callback):
            for group in self._groups.values():
                group.conn.add_reconnect_callback(group_id_or_callback)
        else:
            self._get_group(group_id_or_callback).conn.add_reconnect_callback(callback)

    def set_reconnect_callback(self, group_id_or_callback, callback: Callable | None = None) -> None:
        """Deprecated alias — append, don't overwrite. Existing callers
        won't break, and now multiple families can each register their own
        callback (the previous behaviour was destructive: the last setter won).
        """
        self.add_reconnect_callback(group_id_or_callback, callback)

    def _dispatch_farm_recovery(self, group_id: str, farm_name: str) -> None:
        for callback in self._groups[group_id].farm_recovery_callbacks:
            try:
                callback(farm_name)
            except Exception:
                logger.exception("Farm recovery callback failed for %s", group_id)


IBSession = UnifiedIBSession

