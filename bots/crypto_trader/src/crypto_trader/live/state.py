"""Persistent state for live trading — crash recovery via JSON files."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import structlog

log = structlog.get_logger()


class PersistentState:
    """Manages persistent state files for crash recovery.

    Files in state_dir/:
        portfolio_state.json    — current PortfolioState snapshot
        trades.jsonl            — append-only trade journal
        equity_snapshots.jsonl  — periodic equity records
        rule_events.jsonl       — portfolio rule check log
    """

    def __init__(self, state_dir: Path) -> None:
        self._dir = state_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def portfolio_state_path(self) -> Path:
        return self._dir / "portfolio_state.json"

    @property
    def trades_path(self) -> Path:
        return self._dir / "trades.jsonl"

    @property
    def equity_path(self) -> Path:
        return self._dir / "equity_snapshots.jsonl"

    @property
    def rule_events_path(self) -> Path:
        return self._dir / "rule_events.jsonl"

    @property
    def instrumented_trades_path(self) -> Path:
        return self._dir / "instrumented_trades.jsonl"

    @property
    def missed_opportunities_path(self) -> Path:
        return self._dir / "missed_opportunities.jsonl"

    @property
    def daily_snapshots_path(self) -> Path:
        return self._dir / "daily_snapshots.jsonl"

    @property
    def errors_path(self) -> Path:
        return self._dir / "errors.jsonl"

    def save_portfolio_state(self, state_dict: dict) -> None:
        """Atomically save portfolio state."""
        self._atomic_write_json(self.portfolio_state_path, state_dict)

    def load_portfolio_state(self) -> dict | None:
        """Load portfolio state from disk. Returns None if not found."""
        return self._read_json(self.portfolio_state_path)

    def append_trade(self, trade_dict: dict) -> None:
        """Append a trade record to the journal."""
        self._append_jsonl(self.trades_path, trade_dict)

    def load_trades(self) -> list[dict]:
        """Load all trade records."""
        return self._read_jsonl(self.trades_path)

    def append_equity_snapshot(self, equity: float, timestamp: datetime | None = None) -> None:
        """Record an equity snapshot."""
        ts = timestamp or datetime.now(timezone.utc)
        self._append_jsonl(self.equity_path, {
            "timestamp": ts.isoformat(),
            "equity": equity,
        })

    def load_equity_snapshots(self) -> list[dict]:
        """Load all equity snapshots."""
        return self._read_jsonl(self.equity_path)

    def append_rule_event(self, event_dict: dict) -> None:
        """Append a portfolio rule event."""
        self._append_jsonl(self.rule_events_path, event_dict)

    def _atomic_write_json(self, path: Path, data: dict) -> None:
        """Write JSON atomically (tmp + rename)."""
        tmp_path = path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp_path, path)
        except Exception:
            log.exception("state.write_failed", path=str(path))
            if tmp_path.exists():
                tmp_path.unlink()

    def _read_json(self, path: Path) -> dict | None:
        """Read a JSON file. Returns None if not found."""
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            log.exception("state.read_failed", path=str(path))
            return None

    def _append_jsonl(self, path: Path, record: dict) -> None:
        """Append a JSON line to a JSONL file."""
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            log.exception("state.append_failed", path=str(path))

    def _read_jsonl(self, path: Path) -> list[dict]:
        """Read all records from a JSONL file."""
        if not path.exists():
            return []
        records = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception:
            log.exception("state.read_jsonl_failed", path=str(path))
        return records
