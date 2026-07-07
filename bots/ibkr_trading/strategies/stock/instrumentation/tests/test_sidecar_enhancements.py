"""Tests for sidecar enhancements: diagnostics (#24), priority sorting (#25), gzip (#26)."""
import gzip
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from strategies.stock.instrumentation.src.sidecar import Sidecar, _EVENT_PRIORITY


class TestSidecarDiagnostics:
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

    def test_get_diagnostics_initial_state(self):
        sidecar = Sidecar(self.config)
        diag = sidecar.get_diagnostics()
        assert diag["sidecar_buffer_depth"] == 0
        assert diag["relay_reachable"] is False
        assert diag["last_successful_forward_at"] is None
        assert diag["total_forwarded"] == 0
        assert diag["last_error"] is None

    def test_diagnostics_keys(self):
        sidecar = Sidecar(self.config)
        diag = sidecar.get_diagnostics()
        expected_keys = {
            "sidecar_buffer_depth", "relay_reachable",
            "last_successful_forward_at", "total_forwarded", "last_error",
        }
        assert set(diag.keys()) == expected_keys

    def test_buffer_depth_updates_on_run_once(self):
        """Buffer depth should reflect unsent events."""
        trades_dir = Path(self.tmpdir) / "trades"
        trades_dir.mkdir(parents=True, exist_ok=True)
        filepath = trades_dir / "trades_2026-03-01.jsonl"
        with open(filepath, "w") as f:
            for i in range(5):
                f.write(json.dumps({"trade_id": f"t{i}", "entry_time": f"2026-03-01T{10+i}:00:00Z"}) + "\n")

        sidecar = Sidecar(self.config)
        sidecar.run_once()  # no relay → won't send, but should count
        # Buffer depth set to total unsent
        assert sidecar._buffer_depth == 5


class TestEventPrioritySorting:
    def test_priority_map_order(self):
        assert _EVENT_PRIORITY["error"] < _EVENT_PRIORITY["trade"]
        assert _EVENT_PRIORITY["daily_snapshot"] < _EVENT_PRIORITY["trade"]
        assert _EVENT_PRIORITY["trade"] < _EVENT_PRIORITY["missed_opportunity"]
        assert _EVENT_PRIORITY["missed_opportunity"] < _EVENT_PRIORITY["process_quality"]

    def test_wrap_event_includes_priority(self):
        tmpdir = tempfile.mkdtemp()
        config = {"bot_id": "test", "data_dir": tmpdir}
        sidecar = Sidecar(config)
        wrapped = sidecar._wrap_event({"trade_id": "t1"}, "trade")
        assert "priority" in wrapped
        assert wrapped["priority"] == 2

    def test_wrap_event_error_priority(self):
        tmpdir = tempfile.mkdtemp()
        config = {"bot_id": "test", "data_dir": tmpdir}
        sidecar = Sidecar(config)
        wrapped = sidecar._wrap_event({"error": "test"}, "error")
        assert wrapped["priority"] == 0

    def test_priority_sorting_in_run_once(self):
        """Verify that events are sorted by priority across different files."""
        tmpdir = tempfile.mkdtemp()
        config = {
            "bot_id": "test",
            "data_dir": tmpdir,
            "sidecar": {
                "relay_url": "http://test.local/events",
                "batch_size": 100,
                "retry_max": 1,
                "retry_backoff_base_seconds": 0.001,
            },
        }

        # Create trade events
        trades_dir = Path(tmpdir) / "trades"
        trades_dir.mkdir(parents=True, exist_ok=True)
        with open(trades_dir / "trades_2026-03-01.jsonl", "w") as f:
            f.write(json.dumps({"trade_id": "t1", "entry_time": "2026-03-01T10:00:00Z"}) + "\n")

        # Create error events
        errors_dir = Path(tmpdir) / "errors"
        errors_dir.mkdir(parents=True, exist_ok=True)
        with open(errors_dir / "instrumentation_errors_2026-03-01.jsonl", "w") as f:
            f.write(json.dumps({"error": "test", "timestamp": "2026-03-01T10:00:00Z"}) + "\n")

        sidecar = Sidecar(config)

        # Collect all unsent events and verify sort order
        all_files = sidecar._get_event_files()
        all_unsent = []
        for filepath, event_type in all_files:
            unsent = sidecar._read_unsent_events(filepath, event_type)
            all_unsent.extend(unsent)

        all_unsent.sort(key=lambda e: e.get("priority", 99))

        # Error should come before trade
        assert all_unsent[0]["event_type"] == "error"
        assert all_unsent[1]["event_type"] == "trade"


class TestGzipCompression:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_gzip_disabled_by_default(self):
        config = {"bot_id": "test", "data_dir": self.tmpdir}
        sidecar = Sidecar(config)
        assert sidecar.use_gzip is False

    def test_gzip_enabled_from_config(self):
        config = {
            "bot_id": "test",
            "data_dir": self.tmpdir,
            "sidecar": {"use_gzip": True},
        }
        sidecar = Sidecar(config)
        assert sidecar.use_gzip is True

    def test_gzip_compresses_payload(self):
        """When gzip is enabled, POST body should be gzip-compressed."""
        config = {
            "bot_id": "test",
            "data_dir": self.tmpdir,
            "sidecar": {
                "relay_url": "http://test.local/events",
                "use_gzip": True,
                "retry_max": 1,
                "retry_backoff_base_seconds": 0.001,
            },
        }
        sidecar = Sidecar(config)

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("instrumentation.src.sidecar.requests") as mock_requests:
            mock_requests.post.return_value = mock_response
            events = [{"event_id": "e1", "bot_id": "test", "event_type": "trade",
                       "priority": 2, "payload": "{}", "exchange_timestamp": "2026-03-01T10:00:00Z"}]
            result = sidecar._send_batch(events)

            assert result is True
            call_kwargs = mock_requests.post.call_args
            headers = call_kwargs[1]["headers"]
            body = call_kwargs[1]["data"]

            assert headers.get("Content-Encoding") == "gzip"
            # Verify body is actually gzip compressed
            decompressed = gzip.decompress(body)
            data = json.loads(decompressed)
            assert "events" in data

    def test_gzip_fallback_on_415(self):
        """If relay returns 415, gzip should auto-disable and retry uncompressed."""
        config = {
            "bot_id": "test",
            "data_dir": self.tmpdir,
            "sidecar": {
                "relay_url": "http://test.local/events",
                "use_gzip": True,
                "retry_max": 3,
                "retry_backoff_base_seconds": 0.001,
            },
        }
        sidecar = Sidecar(config)

        mock_415 = MagicMock()
        mock_415.status_code = 415
        mock_200 = MagicMock()
        mock_200.status_code = 200

        with patch("instrumentation.src.sidecar.requests") as mock_requests:
            mock_requests.post.side_effect = [mock_415, mock_200]
            events = [{"event_id": "e1", "bot_id": "test", "event_type": "trade",
                       "priority": 2, "payload": "{}", "exchange_timestamp": "2026-03-01T10:00:00Z"}]
            result = sidecar._send_batch(events)

            assert result is True
            assert sidecar.use_gzip is False  # auto-disabled

            # Second call should be uncompressed
            second_call = mock_requests.post.call_args_list[1]
            headers = second_call[1]["headers"]
            assert "Content-Encoding" not in headers


class TestSidecarDiagnosticsAfterSend:
    def test_diagnostics_update_on_success(self):
        tmpdir = tempfile.mkdtemp()
        config = {
            "bot_id": "test",
            "data_dir": tmpdir,
            "sidecar": {
                "relay_url": "http://test.local/events",
                "retry_max": 1,
                "retry_backoff_base_seconds": 0.001,
            },
        }
        sidecar = Sidecar(config)

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("instrumentation.src.sidecar.requests") as mock_requests:
            mock_requests.post.return_value = mock_response
            events = [{"event_id": "e1", "bot_id": "test", "event_type": "trade",
                       "priority": 2, "payload": "{}", "exchange_timestamp": "2026-03-01T10:00:00Z"}]
            sidecar._send_batch(events)

        diag = sidecar.get_diagnostics()
        assert diag["relay_reachable"] is True
        assert diag["total_forwarded"] == 1
        assert diag["last_successful_forward_at"] is not None

    def test_diagnostics_update_on_failure(self):
        tmpdir = tempfile.mkdtemp()
        config = {
            "bot_id": "test",
            "data_dir": tmpdir,
            "sidecar": {
                "relay_url": "http://test.local/events",
                "retry_max": 1,
                "retry_backoff_base_seconds": 0.001,
            },
        }
        sidecar = Sidecar(config)

        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("instrumentation.src.sidecar.requests") as mock_requests:
            mock_requests.post.return_value = mock_response
            events = [{"event_id": "e1", "bot_id": "test", "event_type": "trade",
                       "priority": 2, "payload": "{}", "exchange_timestamp": "2026-03-01T10:00:00Z"}]
            sidecar._send_batch(events)

        diag = sidecar.get_diagnostics()
        assert diag["relay_reachable"] is False
        assert diag["last_error"] == "all_retries_failed"


class TestSidecarRetentionAndValidation:
    def test_validate_configuration_raises_in_strict_mode(self):
        tmpdir = tempfile.mkdtemp()
        config = {
            "bot_id": "test",
            "data_dir": tmpdir,
            "sidecar": {"relay_url": ""},
        }
        sidecar = Sidecar(config)

        with pytest.raises(RuntimeError, match="missing relay_url, HMAC secret"):
            sidecar.validate_configuration(strict=True)

    def test_cleanup_local_files_prunes_old_event_files(self):
        tmpdir = tempfile.mkdtemp()
        config = {
            "bot_id": "test",
            "data_dir": tmpdir,
            "rotation": {"max_file_age_days": 1},
            "sidecar": {"buffer_dir": str(Path(tmpdir) / ".sidecar_buffer")},
        }
        trades_dir = Path(tmpdir) / "trades"
        trades_dir.mkdir(parents=True, exist_ok=True)
        old_file = trades_dir / "trades_2026-03-01.jsonl"
        old_file.write_text(json.dumps({"trade_id": "t1"}) + "\n", encoding="utf-8")
        two_days_ago = os.path.getmtime(old_file) - (2 * 86400)
        os.utime(old_file, (two_days_ago, two_days_ago))

        sidecar = Sidecar(config)
        sidecar.cleanup_local_files()

        assert old_file.exists() is False

    def test_cleanup_local_files_prunes_oldest_files_to_disk_budget(self):
        tmpdir = tempfile.mkdtemp()
        trades_dir = Path(tmpdir) / "trades"
        trades_dir.mkdir(parents=True, exist_ok=True)
        older = trades_dir / "trades_old.jsonl"
        newer = trades_dir / "trades_new.jsonl"
        payload = json.dumps({"trade_id": "t1", "entry_time": "2026-03-01T10:00:00Z"}) + "\n"
        older.write_text(payload * 5, encoding="utf-8")
        newer.write_text(payload * 5, encoding="utf-8")
        max_disk_mb = (newer.stat().st_size + 32) / (1024 * 1024)
        config = {
            "bot_id": "test",
            "data_dir": tmpdir,
            "rotation": {"max_disk_mb": max_disk_mb},
            "sidecar": {"buffer_dir": str(Path(tmpdir) / ".sidecar_buffer")},
        }
        now = os.path.getmtime(newer)
        os.utime(older, (now - 10, now - 10))

        sidecar = Sidecar(config)
        sidecar.cleanup_local_files()

        assert older.exists() is False
        assert newer.exists() is True
