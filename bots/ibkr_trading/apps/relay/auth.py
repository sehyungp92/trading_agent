"""HMAC-SHA256 authentication for the relay service."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging

logger = logging.getLogger(__name__)


class HMACAuth:
    """Verifies HMAC-SHA256 signatures on incoming event batches."""

    def __init__(self, shared_secrets: dict[str, str] | None = None):
        """
        Args:
            shared_secrets: mapping of bot_id -> hex-encoded HMAC secret.
                            If empty/None, auth is disabled (dev mode).
        """
        self.secrets = shared_secrets or {}
        if not self.secrets:
            logger.warning("No shared secrets configured — HMAC auth disabled")

    @property
    def enabled(self) -> bool:
        return len(self.secrets) > 0

    def verify(self, body: bytes, signature: str, bot_id: str) -> bool:
        """Verify the HMAC-SHA256 signature of a request body.

        Args:
            body: raw request body bytes (must be canonical JSON with sort_keys=True)
            signature: hex-encoded HMAC-SHA256 signature from X-Signature header
            bot_id: the bot_id claiming to send this request

        Returns:
            True if signature is valid or auth is disabled
        """
        if not self.enabled:
            return True

        secret = self.secrets.get(bot_id)
        if not secret:
            logger.warning("Unknown bot_id: %s", bot_id)
            return False

        expected = hmac.new(
            secret.encode(), body, hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(expected, signature)
