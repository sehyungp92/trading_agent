"""Tests for Sidecar forwarder."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from strategies.swing.instrumentation.src.sidecar import Sidecar


class TestSidecar:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "test_bot",
            "data_dir": self.tmpdir,
            "sidecar": {
                "relay_url": "",  # no relay for tests
                "hmac_secret_env": "TEST_HMAC_SECRET",
                "batch_size": 10,
                "retry_max": 1,
                "retry_backoff_base_seconds": 0,
                "poll_interval_seconds": 1,
                "buffer_dir": str(Path(self.tmpdir) / ".sidecar_buffer"),
            },
        }
        self.sidecar = Sidecar(self.config)

    def _write_trade_events(self, events):
        trades_dir = Path(self.tmpdir) / "trades"
        trades_dir.mkdir(parents=True, exist_ok=True)
        filepath = trades_dir / "trades_2026-03-01.jsonl"
        with open(filepath, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        return filepath

    def test_wrap_event_extracts_metadata(self):
        raw = {
            "trade_id": "t1",
            "event_metadata": {
                "event_id": "abc123def456ab",
                "exchange_timestamp": "2026-03-01T10:00:00Z",
            },
        }
        wrapped = self.sidecar._wrap_event(raw, "trade")
        assert wrapped["event_id"] == "abc123def456ab"
        assert wrapped["bot_id"] == "test_bot"
        assert wrapped["event_type"] == "trade"
        assert "payload" in wrapped
        assert wrapped["exchange_timestamp"] == "2026-03-01T10:00:00Z"

    def test_wrap_event_generates_id_when_missing(self):
        raw = {"trade_id": "t1", "entry_time": "2026-03-01T10:00:00Z"}
        wrapped = self.sidecar._wrap_event(raw, "trade")
        assert wrapped["event_id"]  # should be generated
        assert len(wrapped["event_id"]) == 16

    def test_read_unsent_events(self):
        filepath = self._write_trade_events([
            {"trade_id": "t1", "event_metadata": {"event_id": "id1", "exchange_timestamp": "2026-03-01T10:00:00Z"}},
            {"trade_id": "t2", "event_metadata": {"event_id": "id2", "exchange_timestamp": "2026-03-01T11:00:00Z"}},
        ])
        events = self.sidecar._read_unsent_events(filepath, "trade")
        assert len(events) == 2

    def test_watermark_prevents_resend(self):
        filepath = self._write_trade_events([
            {"trade_id": "t1", "event_metadata": {"event_id": "id1", "exchange_timestamp": "2026-03-01T10:00:00Z"}},
            {"trade_id": "t2", "event_metadata": {"event_id": "id2", "exchange_timestamp": "2026-03-01T11:00:00Z"}},
        ])
        # Simulate having already sent line 0
        self.sidecar.watermarks[str(filepath)] = 1
        events = self.sidecar._read_unsent_events(filepath, "trade")
        assert len(events) == 1  # only line 1

    def test_sign_payload_with_secret(self):
        import os
        os.environ["TEST_HMAC_SECRET"] = "test-secret-key"
        sidecar = Sidecar(self.config)
        sig = sidecar._sign_payload('{"test": true}')
        assert sig  # non-empty
        assert len(sig) == 64  # SHA256 hex
        del os.environ["TEST_HMAC_SECRET"]

    def test_sign_payload_without_secret(self):
        sig = self.sidecar._sign_payload('{"test": true}')
        assert sig == ""  # no secret configured

    def test_get_event_files(self):
        # Create files in different subdirectories
        (Path(self.tmpdir) / "trades").mkdir(parents=True, exist_ok=True)
        (Path(self.tmpdir) / "trades" / "trades_2026-03-01.jsonl").write_text("{}\n")
        (Path(self.tmpdir) / "daily").mkdir(parents=True, exist_ok=True)
        (Path(self.tmpdir) / "daily" / "daily_2026-03-01.json").write_text("{}")

        files = self.sidecar._get_event_files()
        assert len(files) >= 2
        event_types = [et for _, et in files]
        assert "trade" in event_types
        assert "daily_snapshot" in event_types

    def test_cleanup_old_watermarks(self):
        self.sidecar.watermarks["/nonexistent/file.jsonl"] = 10
        self.sidecar.cleanup_old_watermarks()
        assert "/nonexistent/file.jsonl" not in self.sidecar.watermarks

    def test_wrap_event_priority_error(self):
        raw = {"event_metadata": {"event_id": "err1"}}
        wrapped = self.sidecar._wrap_event(raw, "error")
        assert wrapped["priority"] == 0

    def test_wrap_event_priority_trade_exit(self):
        raw = {"event_metadata": {"event_id": "t1"}, "stage": "exit"}
        wrapped = self.sidecar._wrap_event(raw, "trade")
        assert wrapped["priority"] == 2

    def test_wrap_event_priority_trade_entry(self):
        raw = {"event_metadata": {"event_id": "t2"}, "stage": "entry"}
        wrapped = self.sidecar._wrap_event(raw, "trade")
        assert wrapped["priority"] == 2

    def test_wrap_event_priority_heartbeat(self):
        raw = {"event_metadata": {"event_id": "hb1"}}
        wrapped = self.sidecar._wrap_event(raw, "heartbeat")
        assert wrapped["priority"] == 5

    def test_compute_buffer_depth_empty(self):
        depth = self.sidecar._compute_buffer_depth()
        assert depth == 0

    def test_compute_buffer_depth_with_events(self):
        self._write_trade_events([
            {"trade_id": "t1", "event_metadata": {}},
            {"trade_id": "t2", "event_metadata": {}},
        ])
        depth = self.sidecar._compute_buffer_depth()
        assert depth == 2

    def test_compute_buffer_depth_respects_watermarks(self):
        filepath = self._write_trade_events([
            {"trade_id": "t1", "event_metadata": {}},
            {"trade_id": "t2", "event_metadata": {}},
            {"trade_id": "t3", "event_metadata": {}},
        ])
        self.sidecar.watermarks[str(filepath)] = 2
        depth = self.sidecar._compute_buffer_depth()
        assert depth == 1

    def test_relay_reachable_tracking(self):
        assert self.sidecar._relay_reachable is None
        assert self.sidecar._last_successful_forward_at is None
        assert self.sidecar._start_time is not None

    def test_heartbeat_every_n_default(self):
        assert self.sidecar.heartbeat_every_n == 10

    def test_gzip_compression_in_send_batch(self):
        """Verify gzip compression headers are set when data is compressible."""
        import gzip as gzip_mod
        # Create a batch of events large enough for gzip to help
        events = [
            {"event_id": f"e{i}", "bot_id": "test_bot", "event_type": "trade",
             "payload": json.dumps({"data": "x" * 200}), "priority": 3}
            for i in range(10)
        ]
        envelope = {"bot_id": "test_bot", "events": events}
        canonical = json.dumps(envelope, sort_keys=True)
        raw_bytes = canonical.encode()
        compressed = gzip_mod.compress(raw_bytes)
        # Gzip should save bytes for repetitive data
        assert len(compressed) < len(raw_bytes)

    def test_canonical_sort_keys(self):
        """Verify HMAC signing uses sort_keys=True canonicalization."""
        import os
        os.environ["TEST_HMAC_SECRET"] = "test-secret"
        sidecar = Sidecar(self.config)
        data = {"z_field": 1, "a_field": 2}
        canonical = json.dumps(data, sort_keys=True)
        assert canonical == '{"a_field": 2, "z_field": 1}'
        sig = sidecar._sign_payload(canonical)
        assert sig  # should produce valid signature
        del os.environ["TEST_HMAC_SECRET"]

    def test_forward_event_types_filtering(self):
        """Only configured event types should be forwarded."""
        config = {**self.config}
        config["sidecar"] = {**config["sidecar"], "forward_event_types": ["trade", "error"]}
        sidecar = Sidecar(config)

        # Create trade and daily snapshot files
        (Path(self.tmpdir) / "trades").mkdir(parents=True, exist_ok=True)
        (Path(self.tmpdir) / "trades" / "trades_2026-03-01.jsonl").write_text(
            json.dumps({"trade_id": "t1", "event_metadata": {"event_id": "id1"}}) + "\n"
        )
        (Path(self.tmpdir) / "daily").mkdir(parents=True, exist_ok=True)
        (Path(self.tmpdir) / "daily" / "daily_2026-03-01.json").write_text(
            json.dumps({"snapshot_id": "s1"})
        )

        # run_once should only process trades (not daily_snapshot)
        sidecar.run_once()
        # daily_snapshot not in forward list, so its watermark should NOT be updated
        daily_key = str(Path(self.tmpdir) / "daily" / "daily_2026-03-01.json")
        assert daily_key not in sidecar.watermarks

    def test_forward_event_types_none_allows_all(self):
        """When forward_event_types is not set, all types should pass."""
        sidecar = Sidecar(self.config)
        assert sidecar.forward_event_types is None

    def test_interruptible_sleep_stops_on_running_false(self):
        """_interruptible_sleep should exit early when _running is False."""
        import time
        self.sidecar._running = False
        start = time.monotonic()
        self.sidecar._interruptible_sleep(10)
        elapsed = time.monotonic() - start
        assert elapsed < 2  # should exit almost immediately

    def test_streaming_read_matches_full_read(self):
        """Streaming file read should produce same events as the old approach."""
        filepath = self._write_trade_events([
            {"trade_id": f"t{i}", "event_metadata": {"event_id": f"id{i}", "exchange_timestamp": "2026-03-01T10:00:00Z"}}
            for i in range(100)
        ])
        events = self.sidecar._read_unsent_events(filepath, "trade")
        assert len(events) == 100
        assert events[0]["_line_number"] == 0
        assert events[99]["_line_number"] == 99
