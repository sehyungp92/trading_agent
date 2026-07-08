"""Utility for round-tripping datetime.time values through JSON state files.

When phase state is saved, datetime.time objects are serialized as ISO strings
(e.g., "10:30:00"). On reload, these must be converted back to time objects
before passing to the config mutator / engine.
"""
from __future__ import annotations

import re
from datetime import time
from typing import Any

_TIME_PARAM_KEYS = frozenset({
    "param_overrides.entry_window_start",
    "param_overrides.entry_window_end",
    "param_overrides.eod_flatten_time",
    "param_overrides.late_entry_cutoff",
    "param_overrides.orb_time_decay_start",
    "param_overrides.pdh_entry_window_end",
})

_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")


def hydrate_time_mutations(mutations: dict[str, Any]) -> dict[str, Any]:
    """Convert ISO time strings back to datetime.time for known keys."""
    result = dict(mutations)
    for key in _TIME_PARAM_KEYS:
        val = result.get(key)
        if isinstance(val, str):
            match = _TIME_RE.match(val)
            if match:
                result[key] = time(int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))
    return result
