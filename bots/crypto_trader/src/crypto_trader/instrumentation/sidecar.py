"""SidecarForwarder — background thread that polls JSONL files and forwards to relay."""

from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import structlog

from crypto_trader.instrumentation.types import canonical_event_envelope

log = structlog.get_logger()

# JSONL file types to poll
_EVENT_FILES = (
    "instrumented_trades",
    "missed_opportunities",
    "daily_snapshots",
    "errors",
    "pipeline_funnels",
    "health_reports",
)

_EVENT_FILE_MAP = {
    "instrumented_trades": "trade",
    "missed_opportunities": "missed_opportunity",
    "daily_snapshots": "daily_snapshot",
    "errors": "error",
    "pipeline_funnels": "pipeline_funnel",
    "health_reports": "heartbeat",
    "orders": "order",
    "fills": "fill",
    "portfolio_rules": "portfolio_rule",
    "risk_decisions": "risk_decision",
    "positions": "position_snapshot",
    "portfolio": "portfolio_snapshot",
    "allocations": "allocation_snapshot",
    "config_snapshots": "config_snapshot",
    "deployments": "deployment",
    "decisions": "decision_event",
    "markets": "market_snapshot",
    "indicators": "indicator_snapshot",
    "filters": "filter_decision",
    "regime": "regime_transition",
}


EVENT_VALUE_CLASSES = {
    "trade": "learning_authority",
    "missed_opportunity": "learning_authority",
    "pipeline_funnel": "learning_authority",
    "order": "learning_authority",
    "fill": "learning_authority",
    "portfolio_rule": "learning_authority",
    "risk_decision": "learning_authority",
    "filter_decision": "learning_authority",
    "deployment": "learning_authority",
    "indicator_snapshot": "learning_gap_diagnostic",
    "market_snapshot": "learning_gap_diagnostic",
    "decision_event": "learning_gap_diagnostic",
    "regime_transition": "learning_gap_diagnostic",
    "daily_snapshot": "operational_health",
    "heartbeat": "operational_health",
    "health_report": "operational_health",
    "error": "operational_health",
    "position_snapshot": "operational_health",
    "portfolio_snapshot": "operational_health",
    "allocation_snapshot": "operational_health",
    "config_snapshot": "operational_health",
}


class SidecarForwarder:
    """Background thread that polls JSONL files and forwards events to relay.

    Matches the reference architecture's sidecar pattern:
    - Reads new lines since last watermark
    - Signs payload with HMAC-SHA256
    - Gzip compresses if > 1KB
    - POSTs to relay /events endpoint
    - Persists watermarks for crash recovery
    """

    def __init__(
        self,
        state_dir: Path,
        relay_url: str,
        bot_id: str,
        shared_secret: str,
        poll_interval: float = 5.0,
        batch_size: int = 50,
        error_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._state_dir = state_dir
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._relay_url = relay_url.rstrip("/")
        self._bot_id = bot_id
        self._secret = shared_secret.encode()
        self._poll_interval = poll_interval
        self._batch_size = batch_size

        self._watermarks: dict[str, int] = {}
        self._watermark_file = state_dir / ".sidecar_watermarks.json"
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_successful_send_at: str | None = None
        self._consecutive_send_failures = 0
        self._last_send_error: str | None = None
        self._error_callback = error_callback

    def start(self) -> None:
        """Start the sidecar polling thread."""
        self._load_watermarks()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="sidecar")
        self._thread.start()
        log.info("sidecar.started", relay_url=self._relay_url)

    def stop(self) -> None:
        """Signal the polling thread to stop and wait."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        log.info("sidecar.stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        """Return sidecar state suitable for health report enrichment."""
        return {
            "enabled": True,
            "running": self.is_running,
            "event_files": list(_EVENT_FILES),
            "canonical_event_files": [
                str(path.relative_to(self._state_dir))
                for path in self._canonical_event_paths()
            ],
            "event_file_map": dict(_EVENT_FILE_MAP),
            "watermarks": dict(self._watermarks),
            "watermark_file": str(self._watermark_file),
            "last_successful_send_at": self._last_successful_send_at,
            "consecutive_send_failures": self._consecutive_send_failures,
            "last_error": self._last_send_error,
        }

    def _poll_loop(self) -> None:
        """Main loop: read new JSONL lines, batch, sign, POST to relay."""
        while not self._stop_event.is_set():
            self._poll_once()
            self._stop_event.wait(self._poll_interval)

    def _poll_once(self) -> None:
        """Poll each configured JSONL event file once."""
        for event_type, path, watermark_key in self._event_sources():
            try:
                if not path.exists():
                    continue
                new_events, new_offset = self._read_since_watermark(path, watermark_key)
                if new_events:
                    if self._send_batch(new_events, event_type):
                        # Only advance watermark after successful send
                        self._watermarks[watermark_key] = new_offset
                        self._save_watermarks()
            except Exception:
                log.exception("sidecar.poll_error", event_type=event_type)
                self._emit_error(
                    error_type="RuntimeError",
                    message=f"sidecar poll failed for {event_type}",
                    severity="low",
                    recovery_action="retry_next_poll",
                    event_type=event_type,
                )

    def _event_sources(self) -> list[tuple[str, Path, str]]:
        canonical_paths = self._canonical_event_paths()
        canonical_types = {path.parent.name for path in canonical_paths}
        sources = []
        for legacy_type in _EVENT_FILES:
            canonical_type = _EVENT_FILE_MAP.get(legacy_type, legacy_type)
            legacy_path = self._state_dir / f"{legacy_type}.jsonl"
            if canonical_type not in canonical_types or self._has_unread_bytes(legacy_path, legacy_type):
                sources.append((
                    legacy_type,
                    legacy_path,
                    legacy_type,
                ))
        for path in canonical_paths:
            try:
                rel = path.relative_to(self._state_dir).as_posix()
                event_type = path.parent.name
                sources.append((event_type, path, rel))
            except ValueError:
                continue
        return sources

    def _has_unread_bytes(self, path: Path, key: str) -> bool:
        if not path.exists():
            return False
        try:
            size = path.stat().st_size
        except OSError:
            return False
        offset = self._watermarks.get(key, 0)
        return offset < size or offset > size

    def _canonical_event_paths(self) -> list[Path]:
        root = self._state_dir / "instrumentation" / "events"
        if not root.exists():
            return []
        return sorted(root.glob("*/*.jsonl"))

    def _read_since_watermark(self, path: Path, key: str) -> tuple[list[dict], int]:
        """Read new lines since last watermark offset.

        Returns (events, new_offset). Caller is responsible for advancing
        the watermark only after successful delivery.
        """
        offset = self._watermarks.get(key, 0)
        file_size = path.stat().st_size
        if offset > file_size:
            log.warning(
                "sidecar.watermark_beyond_eof",
                file=key,
                watermark=offset,
                file_size=file_size,
            )
            offset = 0
        events: list[dict] = []
        new_offset = offset

        try:
            with open(path, "r", encoding="utf-8") as f:
                f.seek(offset)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            log.warning("sidecar.bad_json_line", file=key)
                    if len(events) >= self._batch_size:
                        break
                new_offset = f.tell()

        except Exception:
            log.exception("sidecar.read_error", file=key)

        return events, new_offset

    def _send_batch(self, events: list[dict], event_type: str) -> bool:
        """Sign with HMAC-SHA256, gzip if needed, POST to relay /events.

        Returns True if the batch was delivered successfully.
        """
        last_error: str | None = None
        try:
            import urllib.request
            canonical_type = _EVENT_FILE_MAP.get(event_type, event_type)
            canonical_events = [
                canonical_event_envelope(
                    canonical_type,
                    event,
                    bot_id=self._bot_id,
                    source={"file_event_type": event_type},
                )
                for event in events
            ]

            payload = {
                "bot_id": self._bot_id,
                "event_type": canonical_type,
                "events": canonical_events,
            }

            # Canonical JSON for HMAC
            canonical = json.dumps(payload, sort_keys=True, default=str)
            signature = hmac.new(self._secret, canonical.encode(), hashlib.sha256).hexdigest()

            body = canonical.encode()
            headers = {
                "Content-Type": "application/json",
                "X-Bot-Id": self._bot_id,
                "X-Signature": signature,
            }

            # Gzip if > 1KB
            if len(body) > 1024:
                body = gzip.compress(body)
                headers["Content-Encoding"] = "gzip"

            url = f"{self._relay_url}/events"
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        if resp.status < 300:
                            self._last_successful_send_at = datetime.now(timezone.utc).isoformat()
                            self._consecutive_send_failures = 0
                            self._last_send_error = None
                            log.debug("sidecar.batch_sent",
                                     event_type=canonical_type, count=len(events))
                            return True
                except Exception as e:
                    last_error = str(e)
                    if attempt < max_retries - 1:
                        wait = min(2 ** attempt, 60)
                        log.warning("sidecar.retry", attempt=attempt + 1, wait=wait, error=str(e))
                        if self._stop_event.wait(wait):
                            break
                    else:
                        log.error("sidecar.send_failed", event_type=event_type, error=str(e))
                        self._emit_error(
                            error_type=type(e).__name__,
                            message=str(e),
                            severity="medium",
                            recovery_action="retry_next_poll",
                            event_type=event_type,
                        )

        except Exception as exc:
            last_error = str(exc)
            log.exception("sidecar.batch_error", event_type=event_type)
            self._emit_error(
                error_type=type(exc).__name__,
                message=str(exc),
                severity="medium",
                recovery_action="retry_next_poll",
                event_type=event_type,
            )

        self._consecutive_send_failures += 1
        self._last_send_error = last_error
        return False

    def _emit_error(
        self,
        *,
        error_type: str,
        message: str,
        severity: str,
        recovery_action: str,
        event_type: str,
    ) -> None:
        if self._error_callback is None or _EVENT_FILE_MAP.get(event_type, event_type) == "error":
            return
        try:
            self._error_callback({
                "component": "sidecar",
                "error_type": error_type,
                "message": message,
                "severity": severity,
                "recovery_action": recovery_action,
                "event_type": event_type,
            })
        except Exception:
            log.exception("sidecar.error_callback_failed", event_type=event_type)

    def _load_watermarks(self) -> None:
        """Load watermarks from disk."""
        if self._watermark_file.exists():
            try:
                with open(self._watermark_file, "r", encoding="utf-8") as f:
                    self._watermarks = json.load(f)
            except Exception:
                log.warning("sidecar.watermark_load_failed")
                self._watermarks = {}

    def _save_watermarks(self) -> None:
        """Persist watermarks atomically for crash recovery."""
        tmp = self._watermark_file.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._watermarks, f)
            os.replace(tmp, self._watermark_file)
        except Exception:
            log.exception("sidecar.watermark_save_failed")
            if tmp.exists():
                tmp.unlink()
