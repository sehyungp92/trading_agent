"""Tests for CoordinationLogger."""
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from strategies.swing.instrumentation.src.coordination_logger import CoordinationLogger, CoordinationEvent


class TestCoordinationLogger:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "test_bot",
            "data_dir": self.tmpdir,
            "data_source_id": "test",
        }
        self.logger = CoordinationLogger(self.config)
        self.today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def test_log_action_writes_jsonl(self):
        event = self.logger.log_action(
            action="tighten_stop_be",
            trigger_strategy="ATRSS",
            target_strategy="AKC_HELIX",
            symbol="QQQ",
            rule="rule_1",
            details={"old_stop": 480.0, "new_stop": 485.0},
            outcome="applied",
        )
        assert event is not None
        filepath = Path(self.tmpdir) / "coordination" / f"coordination_{self.today}.jsonl"
        assert filepath.exists()
        data = json.loads(filepath.read_text().strip())
        assert data["action"] == "tighten_stop_be"
        assert data["symbol"] == "QQQ"

    def test_log_action_all_fields_present(self):
        event = self.logger.log_action(
            action="size_boost",
            trigger_strategy="ATRSS",
            target_strategy="AKC_HELIX",
            symbol="SPY",
            rule="rule_2",
            details={"boost_factor": 1.25, "original_size_mult": 1.0},
            outcome="applied",
        )
        d = event.to_dict()
        assert "timestamp" in d
        assert "event_metadata" in d
        assert d["action"] == "size_boost"
        assert d["trigger_strategy"] == "ATRSS"
        assert d["target_strategy"] == "AKC_HELIX"
        assert d["symbol"] == "SPY"
        assert d["rule"] == "rule_2"
        assert d["details"]["boost_factor"] == 1.25
        assert d["outcome"] == "applied"

    def test_log_action_never_crashes(self):
        """Logger must not raise even with bad config."""
        bad_logger = CoordinationLogger.__new__(CoordinationLogger)
        bad_logger.bot_id = "test"
        bad_logger.data_dir = Path("/nonexistent/path/xyz")
        bad_logger.data_source_id = "test"
        # _write_event will fail but log_action should not raise
        result = bad_logger.log_action(
            action="test",
            trigger_strategy="A",
            target_strategy="B",
            symbol="QQQ",
            rule="test",
        )
        # Should still return the event (write failure is swallowed)
        assert result is not None
        assert result.action == "test"

    def test_multiple_actions_same_day(self):
        for i in range(3):
            self.logger.log_action(
                action=f"action_{i}",
                trigger_strategy="ATRSS",
                target_strategy="AKC_HELIX",
                symbol="QQQ",
                rule="rule_1",
                outcome="applied",
            )
        filepath = Path(self.tmpdir) / "coordination" / f"coordination_{self.today}.jsonl"
        lines = filepath.read_text().strip().split("\n")
        assert len(lines) == 3
        for i, line in enumerate(lines):
            data = json.loads(line)
            assert data["action"] == f"action_{i}"

    def test_default_outcome(self):
        event = self.logger.log_action(
            action="test",
            trigger_strategy="A",
            target_strategy="B",
            symbol="QQQ",
            rule="rule_1",
        )
        assert event.outcome == "applied"

    def test_default_details_empty_dict(self):
        event = self.logger.log_action(
            action="test",
            trigger_strategy="A",
            target_strategy="B",
            symbol="QQQ",
            rule="rule_1",
        )
        assert event.details == {}

    def test_event_metadata_has_event_id(self):
        event = self.logger.log_action(
            action="test",
            trigger_strategy="A",
            target_strategy="B",
            symbol="QQQ",
            rule="rule_1",
        )
        assert "event_id" in event.event_metadata
        assert len(event.event_metadata["event_id"]) == 16
