"""Conservative IBKR historical-data pacing helpers."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Hashable


SleepFn = Callable[[float], Awaitable[None]]
NowFn = Callable[[], float]


def request_weight(what_to_show: str) -> int:
    return 2 if what_to_show.upper() == "BID_ASK" else 1


@dataclass
class RequestPacer:
    """Throttle historical requests under the published IBKR pacing rules."""

    min_interval_seconds: float = 12.0
    identical_cooldown_seconds: float = 15.0
    max_weight_per_window: int = 60
    window_seconds: float = 600.0
    sleep_fn: SleepFn = asyncio.sleep
    now_fn: NowFn = time.monotonic
    _last_request_at: float | None = None
    _last_by_signature: dict[Hashable, float] = field(default_factory=dict)
    _window: deque[tuple[float, int]] = field(default_factory=deque)

    async def wait(self, signature: Hashable, *, weight: int = 1) -> None:
        while True:
            now = self.now_fn()
            sleep_for = self._required_delay(now, signature, weight)
            if sleep_for <= 0:
                self._record(now, signature, weight)
                return
            await self.sleep_fn(sleep_for)

    def _required_delay(self, now: float, signature: Hashable, weight: int) -> float:
        delays: list[float] = []
        if self._last_request_at is not None:
            delays.append(self.min_interval_seconds - (now - self._last_request_at))
        if signature in self._last_by_signature:
            delays.append(self.identical_cooldown_seconds - (now - self._last_by_signature[signature]))

        self._drop_expired(now)
        window_weight = sum(item_weight for _ts, item_weight in self._window)
        if window_weight + weight > self.max_weight_per_window and self._window:
            oldest_ts, _oldest_weight = self._window[0]
            delays.append(self.window_seconds - (now - oldest_ts))

        return max([delay for delay in delays if delay > 0], default=0.0)

    def _record(self, now: float, signature: Hashable, weight: int) -> None:
        self._last_request_at = now
        self._last_by_signature[signature] = now
        self._window.append((now, weight))
        self._drop_expired(now)

    def _drop_expired(self, now: float) -> None:
        while self._window and now - self._window[0][0] >= self.window_seconds:
            self._window.popleft()

