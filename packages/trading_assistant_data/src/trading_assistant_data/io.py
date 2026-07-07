"""Small IO helpers."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from .checksums import json_default


def write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, default=json_default)
    last_error: OSError | None = None
    for attempt in range(5):
        tmp = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")
        try:
            tmp.write_text(rendered, encoding="utf-8")
            tmp.replace(path)
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                path.write_text(rendered, encoding="utf-8")
            return path
        except OSError as exc:
            last_error = exc
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            time.sleep(0.1 * (attempt + 1))
    if last_error is not None:
        raise last_error
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
