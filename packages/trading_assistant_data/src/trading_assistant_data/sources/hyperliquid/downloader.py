"""Read-only Hyperliquid candle and funding downloader."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests


API_URL = "https://api.hyperliquid.xyz/info"
INTERVALS = ("1m", "5m", "15m", "30m", "1h", "4h", "1d")


@dataclass(frozen=True)
class HyperliquidDownloader:
    api_url: str = API_URL
    timeout_seconds: int = 20

    def candles(
        self,
        symbol: str,
        interval: str,
        *,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        if interval not in INTERVALS:
            raise ValueError(f"unsupported Hyperliquid interval: {interval}")
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol.upper(),
                "interval": interval,
                "startTime": _ms(start),
                "endTime": _ms(end),
            },
        }
        return self._post(payload)

    def funding(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        payload = {
            "type": "fundingHistory",
            "coin": symbol.upper(),
            "startTime": _ms(start),
            "endTime": _ms(end),
        }
        return self._post(payload)

    def _post(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        response = requests.post(self.api_url, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise ValueError("Hyperliquid response was not a list")
        return [dict(item) for item in data if isinstance(item, dict)]


def _ms(value: datetime) -> int:
    ts = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return int(ts.astimezone(timezone.utc).timestamp() * 1000)

