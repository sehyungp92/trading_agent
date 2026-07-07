"""Candidate provenance fingerprints."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
