"""Cache key helpers."""

from __future__ import annotations

from trading_assistant_backtest.auto.provenance import stable_fingerprint


def replay_cache_key(*parts: object) -> str:
    return stable_fingerprint({"parts": [str(part) for part in parts]})[:24]
