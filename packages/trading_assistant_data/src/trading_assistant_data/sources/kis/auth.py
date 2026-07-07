"""KIS auth value objects.

Token creation is intentionally separate from the read-only client. No order or account
mutation endpoints are exposed in this repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class KisCredentials:
    base_url: str
    app_key: str
    app_secret: str
    access_token: str


@dataclass(frozen=True)
class KisCredentialSettings:
    base_url: str
    app_key: str
    app_secret: str
    account_mode: str
    timeout_seconds: int = 20

    @classmethod
    def from_env(cls) -> "KisCredentialSettings":
        ack = os.getenv("KIS_READ_ONLY_ACK", "").strip().lower()
        if ack not in {"1", "true", "yes", "read_only", "read-only"}:
            raise RuntimeError("KIS_READ_ONLY_ACK must confirm read-only market-data use")
        app_key = os.getenv("KIS_APP_KEY", "").strip()
        app_secret = os.getenv("KIS_APP_SECRET", "").strip()
        if not app_key or not app_secret:
            raise RuntimeError("KIS_APP_KEY and KIS_APP_SECRET are required")
        return cls(
            base_url=os.getenv("KIS_BASE_URL", "https://openapi.koreainvestment.com:9443").strip(),
            app_key=app_key,
            app_secret=app_secret,
            account_mode=os.getenv("KIS_ACCOUNT_MODE", "paper").strip(),
            timeout_seconds=int(os.getenv("KIS_TIMEOUT_SECONDS", "20")),
        )


def issue_access_token(settings: KisCredentialSettings) -> KisCredentials:
    """Issue a KIS OAuth token for quotation/history endpoints only."""

    response = requests.post(
        f"{settings.base_url.rstrip('/')}/oauth2/tokenP",
        json={
            "grant_type": "client_credentials",
            "appkey": settings.app_key,
            "appsecret": settings.app_secret,
        },
        timeout=settings.timeout_seconds,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    token = str(payload.get("access_token", "")).strip()
    if not token:
        raise RuntimeError("KIS token response did not include access_token")
    return KisCredentials(
        base_url=settings.base_url,
        app_key=settings.app_key,
        app_secret=settings.app_secret,
        access_token=token,
    )
