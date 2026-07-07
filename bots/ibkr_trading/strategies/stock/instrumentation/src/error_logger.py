from __future__ import annotations

import json
import logging
import traceback
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .event_metadata import create_event_metadata
from libs.instrumentation.event_contract import enrich_payload
from libs.instrumentation.lineage import lineage_from_config

logger = logging.getLogger("instrumentation.error_logger")


@dataclass
class ErrorRecord:
    """Structured bot/runtime error payload compatible with trading_assistant triage."""

    bot_id: str
    error_type: str
    message: str
    stack_trace: str = ""
    source_file: str = ""
    source_line: int = 0
    severity: str = "medium"
    category: str = "unknown"
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    event_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ErrorLogger:
    """Write structured error events and track recent error volume for heartbeats."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.bot_id = config["bot_id"]
        self.data_dir = Path(config["data_dir"]) / "errors"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_source_id = config.get("data_source_id", "ibkr_us_equities")
        self._lineage = lineage_from_config(
            config,
            family_id="stock",
            strategy_id=config.get("strategy_id", ""),
        )
        self._recent: deque[datetime] = deque()

    def log_error(
        self,
        *,
        error_type: str,
        message: str,
        severity: str = "medium",
        category: str = "unknown",
        context: dict[str, Any] | None = None,
        exc: BaseException | None = None,
        source_file: str = "",
        source_line: int = 0,
        exchange_timestamp: datetime | None = None,
        stack_trace: str = "",
    ) -> ErrorRecord:
        ts = exchange_timestamp or datetime.now(timezone.utc)
        sev = str(severity or "medium").lower()
        cat = str(category or "unknown").lower()

        if exc is not None and not stack_trace:
            stack_trace = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            if not source_file or not source_line:
                source_file, source_line = self._extract_source(exc)

        metadata = create_event_metadata(
            bot_id=self.bot_id,
            event_type="error",
            payload_key=f"{error_type}:{source_file}:{source_line}:{ts.isoformat()}",
            exchange_timestamp=ts,
            data_source_id=self.data_source_id,
            lineage=self._lineage,
        )

        record = ErrorRecord(
            bot_id=self.bot_id,
            error_type=error_type,
            message=message,
            stack_trace=stack_trace,
            source_file=source_file,
            source_line=source_line,
            severity=sev,
            category=cat,
            context=dict(context or {}),
            timestamp=ts.isoformat(),
            event_metadata=metadata.to_dict(),
        )
        self._write(record)
        self._record_recent(ts)
        return record

    def log_exception(
        self,
        error_type: str,
        exc: BaseException,
        *,
        severity: str = "medium",
        category: str = "unknown",
        context: dict[str, Any] | None = None,
        exchange_timestamp: datetime | None = None,
    ) -> ErrorRecord:
        return self.log_error(
            error_type=error_type,
            message=str(exc),
            severity=severity,
            category=category,
            context=context,
            exc=exc,
            exchange_timestamp=exchange_timestamp,
        )

    def count_recent(self, window_seconds: int = 3600) -> int:
        cutoff = datetime.now(timezone.utc).timestamp() - window_seconds
        while self._recent and self._recent[0].timestamp() < cutoff:
            self._recent.popleft()
        return max(len(self._recent), self._count_recent_from_file(cutoff))

    def _record_recent(self, ts: datetime) -> None:
        self._recent.append(ts)
        self.count_recent()

    @staticmethod
    def _extract_source(exc: BaseException) -> tuple[str, int]:
        tb = exc.__traceback__
        last_tb = None
        while tb is not None:
            last_tb = tb
            tb = tb.tb_next
        if last_tb is None:
            return "", 0
        frame = last_tb.tb_frame
        return str(frame.f_code.co_filename), int(last_tb.tb_lineno)

    def _write(self, record: ErrorRecord) -> None:
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filepath = self.data_dir / f"instrumentation_errors_{today}.jsonl"
            payload = enrich_payload(
                record.to_dict(),
                lineage=self._lineage,
                event_type="error",
                scope="strategy",
            )
            with open(filepath, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to write error record %s: %s", record.error_type, exc)

    def _count_recent_from_file(self, cutoff_ts: float) -> int:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = self.data_dir / f"instrumentation_errors_{today}.jsonl"
        if not filepath.exists():
            return 0

        count = 0
        try:
            for line in filepath.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                ts_raw = payload.get("timestamp")
                if not isinstance(ts_raw, str):
                    continue
                try:
                    ts = datetime.fromisoformat(ts_raw)
                except ValueError:
                    continue
                if ts.timestamp() >= cutoff_ts:
                    count += 1
        except Exception:
            return 0
        return count
