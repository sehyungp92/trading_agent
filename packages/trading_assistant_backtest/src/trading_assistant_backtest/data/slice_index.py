"""Optional slice index reader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_slice_index(bundle_path: Path) -> dict[str, Any]:
    path = bundle_path.parent / "slice_index.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"slices": payload}
