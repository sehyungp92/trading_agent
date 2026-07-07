import gzip
import hashlib
import hmac
import json
import os
import time
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from libs.instrumentation.event_contract import enrich_envelope
from libs.instrumentation.sidecar_compat import install_legacy_sidecar_alias, requests_client

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger("instrumentation.sidecar")


install_legacy_sidecar_alias(__name__)

_DIR_TO_EVENT_TYPE = {
    "trades": "trade",
    "missed": "missed_opportunity",
    "errors": "error",
    "scores": "process_quality",
    "daily": "daily_snapshot",
    "orders": "order",
    "heartbeats": "heartbeat",
    "portfolio_rules": "portfolio_rule_check",
    "risk_denials": "risk_denial",
    "risk_decisions": "risk_decision",
    "risk_halts": "risk_halt",
    # Phase 2B event types
    "indicators": "indicator_snapshot",
    "filter_decisions": "filter_decision",
    "orderbook": "orderbook_context",
    "config_changes": "parameter_change",
    "snapshots": "market_snapshot",
    "post_exit": "post_exit",
    "coordination_events": "coordinator_action",
    "stop_adjustments": "stop_adjustment",
    "positions": "position_snapshot",
    "portfolio": "portfolio_snapshot",
    "family": "family_daily_snapshot",
    "allocations": "allocation_snapshot",
    "reconciliation": "reconciliation_alert",
    "allocation_drift": "allocation_drift",
    "admin_corrections": "admin_correction",
    "inferred_fills": "inferred_fill",
    "exposure": "sector_exposure",
    "correlation": "correlation_snapshot",
    "deployments": "deployment",
    "config_snapshots": "config_snapshot",
    "decisions": "decision_event",
    "regime_transitions": "regime_transition",
    "pipeline_funnel": "pipeline_funnel",
}

_MUTABLE_JSONL_EVENT_TYPES = {"missed_opportunity"}

# Event priority for sorting (#25): lower number = higher priority
_EVENT_PRIORITY = {
    "error": 0,
    "daily_snapshot": 1,
    "trade": 2,
    "missed_opportunity": 3,
    "order": 3,
    "portfolio_rule_check": 3,
    "parameter_change": 1,
    "process_quality": 4,
    "indicator_snapshot": 4,
    "filter_decision": 3,
    "orderbook_context": 4,
    "market_snapshot": 4,
    "post_exit": 4,
    "stop_adjustment": 4,
    "risk_denial": 3,
    "risk_decision": 3,
    "risk_halt": 0,
    "bot_error": 0,
    "deployment": 1,
    "config_snapshot": 1,
    "allocation_snapshot": 4,
    "position_snapshot": 4,
    "portfolio_snapshot": 4,
    "family_daily_snapshot": 1,
    "reconciliation_alert": 1,
    "allocation_freeze": 1,
    "allocation_unfreeze": 1,
    "allocation_drift": 3,
    "drift_assignment": 3,
    "admin_correction": 3,
    "inferred_fill": 3,
    "sector_exposure": 4,
    "correlation_snapshot": 4,
    "decision_event": 4,
    "regime_transition": 4,
    "pipeline_funnel": 4,
    "heartbeat": 5,
    "coordinator_action": 3,
    "trade_entry": 2,
}

_EVENT_VALUE_CLASSES = {
    "trade": "learning_authority",
    "trade_entry": "learning_authority",
    "missed_opportunity": "learning_authority",
    "order": "learning_authority",
    "inferred_fill": "learning_authority",
    "filter_decision": "learning_authority",
    "orderbook_context": "learning_authority",
    "portfolio_rule_check": "learning_authority",
    "risk_decision": "learning_authority",
    "risk_denial": "learning_authority",
    "pipeline_funnel": "learning_authority",
    "deployment": "learning_authority",
    "process_quality": "learning_gap_diagnostic",
    "indicator_snapshot": "learning_gap_diagnostic",
    "market_snapshot": "learning_gap_diagnostic",
    "post_exit": "learning_gap_diagnostic",
    "stop_adjustment": "learning_gap_diagnostic",
    "decision_event": "learning_gap_diagnostic",
    "regime_transition": "learning_gap_diagnostic",
    "coordinator_action": "learning_gap_diagnostic",
    "sector_exposure": "learning_gap_diagnostic",
    "correlation_snapshot": "learning_gap_diagnostic",
    "parameter_change": "learning_gap_diagnostic",
    "allocation_drift": "learning_gap_diagnostic",
    "drift_assignment": "learning_gap_diagnostic",
    "admin_correction": "learning_gap_diagnostic",
    "error": "operational_health",
    "bot_error": "operational_health",
    "risk_halt": "operational_health",
    "daily_snapshot": "operational_health",
    "config_snapshot": "operational_health",
    "allocation_snapshot": "operational_health",
    "position_snapshot": "operational_health",
    "portfolio_snapshot": "operational_health",
    "family_daily_snapshot": "operational_health",
    "reconciliation_alert": "operational_health",
    "allocation_freeze": "operational_health",
    "allocation_unfreeze": "operational_health",
    "heartbeat": "operational_health",
}


class Sidecar:
    """
    Forwards events from local JSONL files to the central relay.
    Runs as a background thread or standalone process.

    Usage:
        sidecar = Sidecar(config)
        sidecar.start()  # background thread
        # or
        sidecar.run_once()  # one-shot forward
    """

    def __init__(self, config: dict):
        self.bot_id = config["bot_id"]
        self.data_dir = Path(config["data_dir"])

        sidecar_config = config.get("sidecar", {})
        rotation_config = config.get("rotation", {})
        raw_url = os.environ.get("INSTRUMENTATION_RELAY_URL") or sidecar_config.get("relay_url", "")
        self.relay_url = raw_url.rstrip("/") + "/events" if raw_url and not raw_url.rstrip("/").endswith("/events") else raw_url
        self.batch_size = sidecar_config.get("batch_size", 50)
        self.retry_max = sidecar_config.get("retry_max", 5)
        self.retry_backoff_base = sidecar_config.get("retry_backoff_base_seconds", 10)
        self.buffer_dir = Path(sidecar_config.get("buffer_dir", str(self.data_dir / ".sidecar_buffer")))
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_age_days = int(rotation_config.get("max_file_age_days", 0) or 0)
        self.max_disk_bytes = int(float(rotation_config.get("max_disk_mb", 0) or 0) * 1024 * 1024)

        hmac_env = sidecar_config.get("hmac_secret_env", "INSTRUMENTATION_HMAC_SECRET")
        self.hmac_secret = os.environ.get(hmac_env, "").encode()
        if not self.hmac_secret:
            logger.warning("HMAC secret not set in %s — events will be unsigned", hmac_env)

        self.watermark_file = self.buffer_dir / "watermark.json"
        self.watermarks = self._load_watermarks()
        self._observed_mutable_jsonl_hashes: Dict[str, List[str]] = {}

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.poll_interval = sidecar_config.get("poll_interval_seconds", 60)

        # gzip compression (#26)
        self.use_gzip = sidecar_config.get("use_gzip", False)

        # Diagnostics state (#24)
        self._buffer_depth = 0
        self._relay_reachable = False
        self._last_successful_forward_at: Optional[str] = None
        self._total_forwarded = 0
        self._last_error: Optional[str] = None

    def get_diagnostics(self) -> dict:
        """Return snapshot of sidecar health state (#24)."""
        return {
            "sidecar_buffer_depth": self._buffer_depth,
            "relay_reachable": self._relay_reachable,
            "last_successful_forward_at": self._last_successful_forward_at,
            "total_forwarded": self._total_forwarded,
            "last_error": self._last_error,
        }

    def validate_configuration(self, strict: bool = False) -> None:
        missing: list[str] = []
        if not self.relay_url:
            missing.append("relay_url")
        if not self.hmac_secret:
            missing.append("HMAC secret")
        if not missing:
            return
        message = "Sidecar relay configuration incomplete: missing " + ", ".join(missing)
        if strict:
            raise RuntimeError(message)
        logger.warning(message)

    def _load_watermarks(self) -> Dict[str, int]:
        if self.watermark_file.exists():
            try:
                return json.loads(self.watermark_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_watermarks(self):
        """Persist watermarks via atomic temp-file + rename to prevent corruption."""
        try:
            tmp_path = self.watermark_file.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(self.watermarks, indent=2))
            tmp_path.replace(self.watermark_file)
        except OSError as e:
            logger.warning("Failed to save watermarks: %s", e)

    def _get_event_files(self) -> List[Tuple[Path, str]]:
        files: List[Tuple[Path, str]] = []
        for subdir, event_type in _DIR_TO_EVENT_TYPE.items():
            dir_path = self.data_dir / subdir
            if not dir_path.exists():
                continue

            if subdir == "daily":
                for f in sorted(dir_path.glob("daily_*.json")):
                    files.append((f, event_type))
            else:
                for f in sorted(dir_path.glob("*.jsonl")):
                    files.append((f, event_type))

        return files

    def _read_unsent_events(self, filepath: Path, event_type: str) -> List[dict]:
        key = str(filepath)
        watermark = self.watermarks.get(key, 0)
        is_mutable_jsonl = filepath.suffix == ".jsonl" and event_type in _MUTABLE_JSONL_EVENT_TYPES

        events = []
        try:
            if filepath.suffix == ".jsonl":
                last_sent = watermark if isinstance(watermark, int) else 0
                known_hashes = []
                if isinstance(watermark, dict) and watermark.get("kind") == "jsonl":
                    stored_hashes = watermark.get("line_hashes", [])
                    if isinstance(stored_hashes, list):
                        known_hashes = [str(value) for value in stored_hashes]

                with open(filepath, "r", encoding="utf-8") as handle:
                    if is_mutable_jsonl:
                        observed_hashes: List[str] = []
                    for i, line in enumerate(handle):
                        line = line.strip()
                        if not line:
                            continue
                        line_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()
                        if is_mutable_jsonl:
                            observed_hashes.append(line_hash)
                            if known_hashes:
                                if i < len(known_hashes) and known_hashes[i] == line_hash:
                                    continue
                            elif i < last_sent:
                                continue
                        elif i < last_sent:
                            continue
                        try:
                            raw = json.loads(line)
                            revision_salt = None
                            if is_mutable_jsonl and known_hashes and i < len(known_hashes):
                                revision_salt = line_hash
                            wrapped = self._wrap_event(raw, event_type, revision_salt=revision_salt)
                            wrapped["_source_file"] = key
                            wrapped["_line_number"] = i
                            wrapped["_line_hash"] = line_hash
                            events.append(wrapped)
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning("Skipping bad line %d in %s: %s", i, filepath, e)
                    if is_mutable_jsonl:
                        self._observed_mutable_jsonl_hashes[key] = observed_hashes
            elif filepath.suffix == ".json":
                stat = filepath.stat()
                current_mtime_ns = int(stat.st_mtime_ns)
                last_mtime_ns = self._json_watermark_mtime(watermark)
                if current_mtime_ns != last_mtime_ns:
                    raw = json.loads(filepath.read_text())
                    wrapped = self._wrap_event(raw, event_type)
                    wrapped["_source_file"] = key
                    wrapped["_line_number"] = 1
                    wrapped["_json_mtime_ns"] = current_mtime_ns
                    events.append(wrapped)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to read %s: %s", filepath, e)

        return events

    @staticmethod
    def _json_watermark_mtime(watermark: object) -> int:
        if isinstance(watermark, dict):
            try:
                return int(watermark.get("mtime_ns", 0) or 0)
            except (TypeError, ValueError):
                return 0
        return 0

    def _wrap_event(
        self,
        raw_event: dict,
        event_type: str,
        revision_salt: Optional[str] = None,
    ) -> dict:
        forwarded_event_type = self._forward_event_type(raw_event, event_type)
        metadata = raw_event.get("event_metadata", {})
        event_id = metadata.get("event_id", "")

        exchange_ts = (
            metadata.get("exchange_timestamp", "")
            or raw_event.get("entry_time", "")
            or raw_event.get("timestamp", "")
            or raw_event.get("date", "")
            or datetime.now(timezone.utc).isoformat()
        )

        if revision_salt:
            logical_event_id = event_id or raw_event.get("trade_id", raw_event.get("date", raw_event.get("snapshot_id", "")))
            raw_str = f"{self.bot_id}|{forwarded_event_type}|{logical_event_id}|{revision_salt}"
            event_id = hashlib.sha256(raw_str.encode()).hexdigest()[:16]
        elif not event_id:
            key = raw_event.get("trade_id", raw_event.get("date", raw_event.get("snapshot_id", "")))
            raw_str = f"{self.bot_id}|{exchange_ts}|{forwarded_event_type}|{key}"
            event_id = hashlib.sha256(raw_str.encode()).hexdigest()[:16]

        wrapped = {
            "event_id": event_id,
            "bot_id": self.bot_id,
            "event_type": forwarded_event_type,
            "priority": _EVENT_PRIORITY.get(forwarded_event_type, _EVENT_PRIORITY.get(event_type, 99)),
            "payload": json.dumps(raw_event, default=str),
            "exchange_timestamp": exchange_ts,
        }
        return enrich_envelope(wrapped, raw_event)

    @staticmethod
    def _forward_event_type(raw_event: dict, event_type: str) -> str:
        """Keep the assistant's canonical trade feed limited to completed trades."""
        if event_type == "trade":
            stage = str(raw_event.get("stage", "")).strip().lower()
            if stage and stage != "exit":
                return f"trade_{stage}"
        payload_event_type = str(raw_event.get("event_type", "")).strip()
        if payload_event_type and event_type in {
            "reconciliation_alert",
            "allocation_drift",
            "admin_correction",
            "inferred_fill",
        }:
            return payload_event_type
        return event_type

    def _sign_payload(self, canonical_json: str) -> str:
        """HMAC-SHA256 signature of the canonicalized JSON payload.

        CRITICAL: The relay verifies against json.dumps(data, sort_keys=True).
        The input MUST be the sort_keys=True serialization.
        """
        if not self.hmac_secret:
            return ""
        return hmac.new(self.hmac_secret, canonical_json.encode(), hashlib.sha256).hexdigest()

    def _send_batch(self, events: List[dict]) -> bool:
        if not self.relay_url:
            logger.warning("No relay_url configured — skipping send")
            return False

        client = requests_client(requests)
        if client is None:
            logger.error("requests library not installed — cannot forward events")
            return False

        clean_events = []
        for e in events:
            clean = {k: v for k, v in e.items() if not k.startswith("_")}
            clean_events.append(clean)

        envelope = {
            "bot_id": self.bot_id,
            "events": clean_events,
        }

        canonical = json.dumps(envelope, sort_keys=True)
        signature = self._sign_payload(canonical)

        headers = {
            "Content-Type": "application/json",
            "X-Bot-ID": self.bot_id,
            "X-Signature": signature,
        }

        # gzip compression (#26): compress payload, HMAC on uncompressed
        body = canonical.encode()
        if self.use_gzip:
            body = gzip.compress(body)
            headers["Content-Encoding"] = "gzip"

        for attempt in range(self.retry_max):
            try:
                response = client.post(
                    self.relay_url,
                    data=body,
                    headers=headers,
                    timeout=30,
                )
                if response.status_code in (200, 409):
                    self._relay_reachable = True
                    self._last_successful_forward_at = datetime.now(timezone.utc).isoformat()
                    self._total_forwarded += len(events)
                    self._last_error = None
                    return True
                elif response.status_code == 401:
                    logger.error("Authentication failed — check HMAC secret")
                    self._last_error = "auth_failed"
                    return False
                elif response.status_code == 415 and self.use_gzip:
                    # Relay doesn't support gzip — auto-disable and retry uncompressed
                    logger.warning("Relay returned 415 — disabling gzip compression")
                    self.use_gzip = False
                    body = canonical.encode()
                    headers.pop("Content-Encoding", None)
                    continue
                elif response.status_code == 429:
                    logger.warning("Rate limited by relay — backing off")
                else:
                    logger.warning(
                        "Relay returned %d (attempt %d/%d)",
                        response.status_code, attempt + 1, self.retry_max,
                    )
            except Exception as e:
                logger.warning("Send failed (attempt %d/%d): %s", attempt + 1, self.retry_max, e)

            backoff = self.retry_backoff_base * (2 ** attempt)
            time.sleep(min(backoff, 300))

        self._relay_reachable = False
        self._last_error = "all_retries_failed"
        return False

    def run_once(self):
        """Collect and forward all unsent events, sorted by priority (#25)."""
        self.cleanup_old_watermarks()
        self.cleanup_local_files()
        self._observed_mutable_jsonl_hashes = {}
        all_files = self._get_event_files()

        # Collect ALL unsent events across all files first (#25)
        all_unsent: List[dict] = []
        remaining_by_source: Dict[str, int] = {}
        for filepath, event_type in all_files:
            unsent = self._read_unsent_events(filepath, event_type)
            all_unsent.extend(unsent)
            if unsent:
                remaining_by_source[str(filepath)] = len(unsent)

        # Update buffer depth diagnostic (#24)
        self._buffer_depth = len(all_unsent)

        if not all_unsent:
            self._persist_observed_mutable_jsonl_hashes(
                set(self._observed_mutable_jsonl_hashes.keys())
            )
            return

        # Sort by priority: errors first, then snapshots, trades, missed, scores (#25)
        all_unsent.sort(key=lambda e: e.get("priority", 99))

        total_sent = 0
        for i in range(0, len(all_unsent), self.batch_size):
            batch = all_unsent[i:i + self.batch_size]
            if self._send_batch(batch):
                # Update watermarks per-event using _source_file/_line_number
                for evt in batch:
                    src = evt["_source_file"]
                    line = evt["_line_number"]
                    if src in remaining_by_source:
                        remaining_by_source[src] = max(0, remaining_by_source[src] - 1)
                    if "_json_mtime_ns" in evt:
                        self.watermarks[src] = {
                            "kind": "json",
                            "mtime_ns": int(evt["_json_mtime_ns"]),
                        }
                        continue
                    if src in self._observed_mutable_jsonl_hashes:
                        continue
                    current = self.watermarks.get(src, 0)
                    current_line = current if isinstance(current, int) else 0
                    if line + 1 > current_line:
                        self.watermarks[src] = line + 1
                self._save_watermarks()
                total_sent += len(batch)
            else:
                logger.warning("Failed to send batch, will retry")
                break

        fully_sent_sources = {
            src for src, remaining in remaining_by_source.items() if remaining == 0
        }
        fully_sent_sources.update(
            src
            for src in self._observed_mutable_jsonl_hashes
            if src not in remaining_by_source
        )
        if fully_sent_sources:
            self._persist_observed_mutable_jsonl_hashes(fully_sent_sources)

        # Update buffer depth after sending
        self._buffer_depth = max(0, self._buffer_depth - total_sent)

        if total_sent > 0:
            logger.info("Forwarded %d events to relay", total_sent)

    def _persist_observed_mutable_jsonl_hashes(self, sources: set[str]) -> None:
        updated = False
        for src in sources:
            line_hashes = self._observed_mutable_jsonl_hashes.get(src)
            if not line_hashes:
                continue
            self.watermarks[src] = {
                "kind": "jsonl",
                "line_hashes": line_hashes,
            }
            updated = True
        if updated:
            self._save_watermarks()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Sidecar started (poll every %ds)", self.poll_interval)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def _run_loop(self):
        while self._running:
            try:
                self.run_once()
            except Exception as e:
                logger.error("Sidecar run_once failed: %s", e)
            time.sleep(self.poll_interval)

    def cleanup_old_watermarks(self):
        to_remove = [key for key in self.watermarks if not Path(key).exists()]
        for key in to_remove:
            del self.watermarks[key]
        if to_remove:
            self._save_watermarks()

    def cleanup_local_files(self) -> None:
        managed_files = self._managed_local_files()
        if not managed_files:
            return

        now = time.time()
        removed = False

        if self.max_file_age_days > 0:
            max_age_seconds = self.max_file_age_days * 86400
            for path in list(managed_files):
                try:
                    if now - path.stat().st_mtime <= max_age_seconds:
                        continue
                except OSError:
                    continue
                if self._delete_managed_file(path):
                    removed = True

        if self.max_disk_bytes > 0:
            managed_files = self._managed_local_files()
            total_size = 0
            file_sizes: List[Tuple[Path, int, float]] = []
            for path in managed_files:
                try:
                    stat = path.stat()
                except OSError:
                    continue
                total_size += stat.st_size
                file_sizes.append((path, stat.st_size, stat.st_mtime))
            file_sizes.sort(key=lambda item: item[2])
            for path, size_bytes, _ in file_sizes:
                if total_size <= self.max_disk_bytes:
                    break
                if self._delete_managed_file(path):
                    total_size = max(0, total_size - size_bytes)
                    removed = True

        if removed:
            self.cleanup_old_watermarks()

    def _managed_local_files(self) -> List[Path]:
        if not self.data_dir.exists():
            return []
        files: List[Path] = []
        for path in self.data_dir.rglob("*"):
            if not path.is_file():
                continue
            if path == self.watermark_file or self.buffer_dir in path.parents:
                continue
            if path.suffix not in {".json", ".jsonl", ".log"}:
                continue
            files.append(path)
        return files

    def _delete_managed_file(self, path: Path) -> bool:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to prune %s: %s", path, exc)
            return False
        self.watermarks.pop(str(path), None)
        return True
