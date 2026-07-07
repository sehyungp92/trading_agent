"""Structured diagnostics for the ALCB runtime."""

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

    def log_decision(self, code: str, payload: dict[str, Any]) -> None:
        self.emit("decisions", {"code": code, **payload})

    def log_order(self, symbol: str, action: str, payload: dict[str, Any]) -> None:
        self.emit("orders", {"symbol": symbol, "action": action, **payload})

    def log_exit(self, symbol: str, reason: str, payload: dict[str, Any]) -> None:
        self.emit("exits", {"symbol": symbol, "reason": reason, **payload})

    def log_campaign(self, symbol: str, state: str, payload: dict[str, Any] | None = None) -> None:
        self.emit("campaigns", {"symbol": symbol, "state": state, "details": payload or {}})

    def log_ledger(self, symbol: str, session_vwap: float | None, avwap_live: float | None, weekly_vwap: float | None) -> None:
        self.emit("ledgers", {"symbol": symbol, "session_vwap": session_vwap, "avwap_live": avwap_live, "weekly_vwap": weekly_vwap})

    def log_missed_fill(self, symbol: str, payload: dict[str, Any]) -> None:
        self.emit("missed", {"symbol": symbol, **payload})

    def log_signal_evaluation(
        self, symbol: str, entry_type: str, triggered: bool,
        conditions: dict[str, Any], limit_price: float | None = None,
    ) -> None:
        self.emit("signals", {
            "symbol": symbol, "entry_type": entry_type,
            "triggered": triggered, "conditions": conditions,
            "limit_price": limit_price,
        })

    def log_exit_decision(
        self, symbol: str, exit_reason: str, context: dict[str, Any],
    ) -> None:
        self.emit("exit_decisions", {
            "symbol": symbol, "exit_reason": exit_reason, **context,
        })

    def log_setup_detected(self, symbol: str, payload: dict[str, Any]) -> None:
        self.emit("setups", {"symbol": symbol, **payload})

    def log_continuation_check(self, symbol: str, payload: dict[str, Any]) -> None:
        self.emit("continuation", {"symbol": symbol, **payload})
