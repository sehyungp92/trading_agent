"""Structured diagnostics for the IARIC runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class JsonlDiagnostics:
    root: Path
    enabled: bool = True
    _cache: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def emit(self, stream: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._cache.setdefault(stream, self.root / f"{stream}.jsonl")
        record = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def log_filter(self, symbol: str, stage: str, accepted: bool, reason: str, details: dict[str, Any] | None = None) -> None:
        self.emit("filters", {"symbol": symbol, "stage": stage, "accepted": accepted, "reason": reason, "details": details or {}})

    def log_regime(self, payload: dict[str, Any]) -> None:
        self.emit("regime", payload)

    def log_artifact_diff(self, trade_date: str, added: list[str], removed: list[str], changed: list[str]) -> None:
        self.emit("artifact_diff", {"trade_date": trade_date, "added": added, "removed": removed, "changed": changed})

    def log_tier_change(self, symbol: str, from_tier: str, to_tier: str, reason: str) -> None:
        self.emit("tiers", {"symbol": symbol, "from_tier": from_tier, "to_tier": to_tier, "reason": reason})

    def log_setup(self, symbol: str, setup_type: str, location_grade: str, reclaim_level: float, stop_level: float) -> None:
        self.emit(
            "setups",
            {
                "symbol": symbol,
                "setup_type": setup_type,
                "location_grade": location_grade,
                "reclaim_level": reclaim_level,
                "stop_level": stop_level,
            },
        )

    def log_acceptance(self, symbol: str, count: int, required: int, adders: list[str], confidence: str | None) -> None:
        self.emit(
            "acceptance",
            {"symbol": symbol, "count": count, "required": required, "adders": adders, "confidence": confidence},
        )

    def log_decision(self, code: str, payload: dict[str, Any]) -> None:
        self.emit("decisions", {"code": code, **payload})

    def log_order(self, symbol: str, action: str, payload: dict[str, Any]) -> None:
        self.emit("orders", {"symbol": symbol, "action": action, **payload})

    def log_exit(self, symbol: str, reason: str, payload: dict[str, Any]) -> None:
        self.emit("exits", {"symbol": symbol, "reason": reason, **payload})

    def log_carry(self, symbol: str, eligible: bool, reason: str, payload: dict[str, Any] | None = None) -> None:
        self.emit("carry", {"symbol": symbol, "eligible": eligible, "reason": reason, "details": payload or {}})

    def log_ledger(self, symbol: str, session_vwap: float | None, avwap_live: float | None) -> None:
        self.emit("ledgers", {"symbol": symbol, "session_vwap": session_vwap, "avwap_live": avwap_live})

    def log_degraded(self, component: str, reason: str, payload: dict[str, Any] | None = None) -> None:
        self.emit("degraded", {"component": component, "reason": reason, "details": payload or {}})
