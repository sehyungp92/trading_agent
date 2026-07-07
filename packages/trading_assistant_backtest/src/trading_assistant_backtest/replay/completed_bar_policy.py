"""Completed-bar and no-lookahead policy primitives."""

from __future__ import annotations

from datetime import datetime


def bar_visible_at(*, bar_close_ts: datetime, decision_ts: datetime) -> bool:
    """A bar is visible only after its close timestamp."""
    return bar_close_ts <= decision_ts
