"""Telegram sender + message formatter."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import aiohttp

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")


class TelegramAlerter:
    """Send messages to Telegram via Bot API with retry."""

    def __init__(self, bot_token: str, chat_id: str, session: aiohttp.ClientSession):
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._chat_id = chat_id
        self._session = session

    async def send(self, text: str) -> bool:
        """Send a message. Returns True on success. Never raises."""
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        timeout = aiohttp.ClientTimeout(total=15)
        for attempt in range(3):
            try:
                async with self._session.post(self._url, json=payload, timeout=timeout) as resp:
                    if resp.status == 200:
                        return True
                    if resp.status == 429:
                        body = await resp.json()
                        retry_after = body.get("parameters", {}).get("retry_after", 5)
                        logger.warning("Telegram rate-limited, retry after %ds", retry_after)
                        await asyncio.sleep(retry_after)
                        continue
                    if resp.status >= 500:
                        logger.warning("Telegram %d on attempt %d", resp.status, attempt + 1)
                        await asyncio.sleep(2 ** attempt)
                        continue
                    # 4xx (not 429) -- bad request, don't retry
                    body = await resp.text()
                    logger.error("Telegram error %d: %s", resp.status, body[:200])
                    return False
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning("Telegram network error attempt %d: %s", attempt + 1, exc)
                await asyncio.sleep(2 ** attempt)
        logger.error("Telegram send failed after 3 attempts")
        return False


def format_alert(check: str, details: str, is_recovery: bool = False) -> str:
    """Format an alert or recovery message."""
    now_et = datetime.now(_ET).strftime("%H:%M:%S ET")
    if is_recovery:
        return f"[ok] <b>RECOVERED:</b> {check}\n{details}\n<i>{now_et}</i>"
    return f"[!] <b>ALERT:</b> {check}\n{details}\n<i>{now_et}</i>"


def format_startup_summary(
    strategy_count: int, family_count: int, interval: int
) -> str:
    """Format the boot notification."""
    now_et = datetime.now(_ET).strftime("%Y-%m-%d %H:%M ET")
    return (
        f"<b>Watchdog started</b>\n"
        f"Monitoring {strategy_count} strategies across {family_count} families\n"
        f"Polling every {interval}s\n"
        f"<i>{now_et}</i>"
    )
