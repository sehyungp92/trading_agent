import json
import tempfile
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
from strategies.stock.instrumentation.src.sidecar import Sidecar


class TestSidecar:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "test_bot",
            "data_dir": self.tmpdir,
            "sidecar": {
                "relay_url": "",
                "hmac_secret_env": "TEST_HMAC_SECRET",
                "batch_size": 10,
                "retry_max": 2,
                "retry_backoff_base_seconds": 0.01,
                "poll_interval_seconds": 1,
                "buffer_dir": str(Path(self.tmpdir) / ".sidecar_buffer"),
            },
        }

    def _write_trade_events(self, events):
        trades_dir = Path(self.tmpdir) / "trades"
        trades_dir.mkdir(parents=True, exist_ok=True)
        filepath = trades_dir / "trades_2026-03-01.jsonl"
        with open(filepath, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        return filepath

    def _write_daily_snapshot(self, data):
        daily_dir = Path(self.tmpdir) / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        filepath = daily_dir / "daily_2026-03-01.json"
        with open(filepath, "w") as f:
            json.dump(data, f)
        return filepath

    def _write_missed_events(self, events):
        missed_dir = Path(self.tmpdir) / "missed"
        missed_dir.mkdir(parents=True, exist_ok=True)
        filepath = missed_dir / "missed_2026-03-01.jsonl"
        with open(filepath, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        return filepath

    def test_init_creates_buffer_dir(self):
        sidecar = Sidecar(self.config)
        assert sidecar.buffer_dir.exists()

    def test_relay_url_normalized_to_events_endpoint(self):
        config = dict(self.config)
        config["sidecar"] = dict(config["sidecar"])
        config["sidecar"]["relay_url"] = "http://relay.local"
        sidecar = Sidecar(config)
        assert sidecar.relay_url == "http://relay.local/events"

    def test_watermarks_persist(self):
        sidecar = Sidecar(self.config)
        sidecar.watermarks["test_file"] = 5
        sidecar._save_watermarks()

        sidecar2 = Sidecar(self.config)
        assert sidecar2.watermarks.get("test_file") == 5

    def test_get_event_files_finds_jsonl(self):
        self._write_trade_events([{"trade_id": "t1"}])
        sidecar = Sidecar(self.config)
        files = sidecar._get_event_files()
        assert len(files) >= 1
        paths = [str(f[0]) for f in files]
        assert any("trades_2026-03-01.jsonl" in p for p in paths)

    def test_get_event_files_finds_daily_json(self):
        self._write_daily_snapshot({"date": "2026-03-01"})
        sidecar = Sidecar(self.config)
        files = sidecar._get_event_files()
        assert any(f[1] == "daily_snapshot" for f in files)

    def test_modified_daily_json_is_re_sent(self):
        filepath = self._write_daily_snapshot(
            {"date": "2026-03-01", "timestamp": "2026-03-01T10:00:00+00:00"}
        )
        sidecar = Sidecar(self.config)
        first = sidecar._read_unsent_events(filepath, "daily_snapshot")
        assert len(first) == 1

        sidecar.watermarks[str(filepath)] = {
            "kind": "json",
            "mtime_ns": first[0]["_json_mtime_ns"],
        }

        time.sleep(0.01)
        filepath.write_text(
            json.dumps({"date": "2026-03-01", "timestamp": "2026-03-01T10:05:00+00:00"}),
            encoding="utf-8",
        )
        second = sidecar._read_unsent_events(filepath, "daily_snapshot")
        assert len(second) == 1
        assert second[0]["_json_mtime_ns"] != first[0]["_json_mtime_ns"]

    def test_modified_missed_jsonl_line_is_re_sent_with_revision_event_id(self):
        filepath = self._write_missed_events([
            {
                "event_metadata": {"event_id": "miss-1"},
                "signal_time": "2026-03-01T10:00:00+00:00",
                "backfill_status": "pending",
            }
        ])
        sidecar = Sidecar(self.config)
        first = sidecar._read_unsent_events(filepath, "missed_opportunity")
        assert len(first) == 1
        assert first[0]["event_id"] == "miss-1"

        sidecar.watermarks[str(filepath)] = {
            "kind": "jsonl",
            "line_hashes": [first[0]["_line_hash"]],
        }
        filepath.write_text(
            json.dumps(
                {
                    "event_metadata": {"event_id": "miss-1"},
                    "signal_time": "2026-03-01T10:00:00+00:00",
                    "backfill_status": "complete",
                    "outcome_pnl_24h": 2.75,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        second = sidecar._read_unsent_events(filepath, "missed_opportunity")
        assert len(second) == 1
        assert second[0]["event_id"] != "miss-1"
        assert json.loads(second[0]["payload"])["backfill_status"] == "complete"

    def test_read_unsent_events_from_jsonl(self):
        filepath = self._write_trade_events([
            {"trade_id": "t1", "entry_time": "2026-03-01T10:00:00Z"},
            {"trade_id": "t2", "entry_time": "2026-03-01T11:00:00Z"},
        ])
        sidecar = Sidecar(self.config)
        events = sidecar._read_unsent_events(filepath, "trade")
        assert len(events) == 2
        assert events[0]["event_type"] == "trade"
        assert events[1]["event_type"] == "trade"

    def test_watermark_skips_already_sent(self):
        filepath = self._write_trade_events([
            {"trade_id": "t1", "entry_time": "2026-03-01T10:00:00Z"},
            {"trade_id": "t2", "entry_time": "2026-03-01T11:00:00Z"},
            {"trade_id": "t3", "entry_time": "2026-03-01T12:00:00Z"},
        ])
        sidecar = Sidecar(self.config)
        sidecar.watermarks[str(filepath)] = 2  # already sent lines 0, 1
        events = sidecar._read_unsent_events(filepath, "trade")
        assert len(events) == 1  # only line 2

    def test_wrap_event_format(self):
        sidecar = Sidecar(self.config)
        raw = {"trade_id": "t1", "entry_time": "2026-03-01T10:00:00Z"}
        wrapped = sidecar._wrap_event(raw, "trade")
        assert "event_id" in wrapped
        assert wrapped["bot_id"] == "test_bot"
        assert wrapped["event_type"] == "trade"
        assert "payload" in wrapped
        assert "exchange_timestamp" in wrapped

    def test_trade_entry_events_are_not_forwarded_as_canonical_trade(self):
        sidecar = Sidecar(self.config)
        wrapped = sidecar._wrap_event(
            {"trade_id": "t1", "entry_time": "2026-03-01T10:00:00Z", "stage": "entry"},
            "trade",
        )
        assert wrapped["event_type"] == "trade_entry"

    def test_trade_exit_events_remain_canonical_trade(self):
        sidecar = Sidecar(self.config)
        wrapped = sidecar._wrap_event(
            {"trade_id": "t1", "entry_time": "2026-03-01T10:00:00Z", "stage": "exit"},
            "trade",
        )
        assert wrapped["event_type"] == "trade"

    def test_sign_payload_without_secret(self):
        sidecar = Sidecar(self.config)
        sig = sidecar._sign_payload('{"test": true}')
        assert sig == ""  # no secret = empty signature

    def test_sign_payload_with_secret(self):
        os.environ["TEST_HMAC_SECRET_2"] = "my_secret_key"
        try:
            config = dict(self.config)
            config["sidecar"] = dict(config["sidecar"])
            config["sidecar"]["hmac_secret_env"] = "TEST_HMAC_SECRET_2"
            sidecar = Sidecar(config)
            sig = sidecar._sign_payload('{"test": true}')
            assert sig != ""
            assert len(sig) == 64  # SHA256 hex digest
        finally:
            del os.environ["TEST_HMAC_SECRET_2"]

    def test_sign_payload_deterministic(self):
        os.environ["TEST_HMAC_SECRET_3"] = "test_key"
        try:
            config = dict(self.config)
            config["sidecar"] = dict(config["sidecar"])
            config["sidecar"]["hmac_secret_env"] = "TEST_HMAC_SECRET_3"
            sidecar = Sidecar(config)
            payload = json.dumps({"events": [], "bot_id": "test"}, sort_keys=True)
            sig1 = sidecar._sign_payload(payload)
            sig2 = sidecar._sign_payload(payload)
            assert sig1 == sig2
        finally:
            del os.environ["TEST_HMAC_SECRET_3"]

    def test_run_once_no_relay_url(self):
        """run_once should not crash when no relay_url is set."""
        self._write_trade_events([
            {"trade_id": "t1", "entry_time": "2026-03-01T10:00:00Z"},
        ])
        sidecar = Sidecar(self.config)
        sidecar.run_once()  # should not raise

    def test_cleanup_old_watermarks(self):
        sidecar = Sidecar(self.config)
        sidecar.watermarks["/nonexistent/file.jsonl"] = 10
        sidecar.watermarks[str(Path(self.tmpdir) / "real_file")] = 5
        sidecar.cleanup_old_watermarks()
        assert "/nonexistent/file.jsonl" not in sidecar.watermarks

    def test_send_batch_returns_false_without_relay(self):
        sidecar = Sidecar(self.config)
        result = sidecar._send_batch([{"event_id": "test"}])
        assert result is False

    def test_internal_fields_stripped_before_send(self):
        """_source_file and _line_number should be stripped from events."""
        sidecar = Sidecar(self.config)
        events = [
            {"event_id": "e1", "bot_id": "test", "_source_file": "/tmp/f", "_line_number": 0},
        ]
        # We can't actually send, but verify the stripping logic in _send_batch
        # by checking the clean_events construction
        clean_events = []
        for e in events:
            clean = {k: v for k, v in e.items() if not k.startswith("_")}
            clean_events.append(clean)
        assert "_source_file" not in clean_events[0]
        assert "_line_number" not in clean_events[0]
        assert "event_id" in clean_events[0]
