"""Artifact emission for the monthly runner."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from trading_assistant_backtest.artifact_index import build_artifact_index
from trading_assistant_backtest.contract_models import (
    BacktestArtifactIndex,
    BacktestExitStatus,
    MonthlyRunManifest,
)


class ArtifactWriter:
    def __init__(self, manifest: MonthlyRunManifest, artifact_root: Path) -> None:
        self.manifest = manifest
        self.root = artifact_root
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / name

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(payload, "model_dump"):
            raw = payload.model_dump(mode="json")
        else:
            raw = payload
        path.write_text(json.dumps(raw, indent=2, default=_json_default) + "\n", encoding="utf-8")
        return path

    def write_jsonl(self, name: str, rows: list[Any]) -> Path:
        path = self.path(name)
        lines = []
        for row in rows:
            raw = row.model_dump(mode="json") if hasattr(row, "model_dump") else row
            lines.append(json.dumps(raw, default=_json_default))
        path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def write_exit_status(
        self,
        *,
        started_at: datetime,
        exit_code: int = 0,
        error: str = "",
    ) -> Path:
        return self.write_json(
            "exit_status.json",
            BacktestExitStatus(
                exit_code=exit_code,
                timed_out=False,
                error=error,
                started_at=started_at,
                finished_at=datetime.now(UTC),
            ),
        )

    def write_index(self) -> BacktestArtifactIndex:
        index = build_artifact_index(self.manifest, self.root)
        self.write_json("artifact_index.json", index)
        return index


def _json_default(value: Any) -> str:
    if isinstance(value, datetime | date):
        return value.isoformat()
    return str(value)
