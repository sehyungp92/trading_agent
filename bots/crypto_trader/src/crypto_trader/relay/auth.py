"""HMAC-SHA256 authentication for relay service."""

from __future__ import annotations

import hashlib
import hmac
import json


def sign_payload(payload: dict, secret: str) -> str:
    """Create HMAC-SHA256 signature of canonical JSON payload."""
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hmac.new(secret.encode(), canonical.encode(), hashlib.sha256).hexdigest()


def verify_signature(payload: dict, signature: str, secret: str) -> bool:
    """Verify HMAC-SHA256 signature against payload."""
    expected = sign_payload(payload, secret)
    return hmac.compare_digest(expected, signature)


class RelayAuth:
    """Manages per-bot shared secrets for relay authentication."""

    def __init__(self, bot_secrets: dict[str, str] | None = None) -> None:
        self._secrets: dict[str, str] = bot_secrets or {}

    def add_bot(self, bot_id: str, secret: str) -> None:
        self._secrets[bot_id] = secret

    def verify(self, bot_id: str, payload: dict, signature: str) -> bool:
        """Verify a request from a specific bot."""
        secret = self._secrets.get(bot_id)
        if secret is None:
            return False
        return verify_signature(payload, signature, secret)
