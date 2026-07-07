"""Tests for provider attribution in validation logs and suggestion records."""
from __future__ import annotations

import json
from pathlib import Path


from trading_assistant.schemas.outcome_measurement import OutcomeMeasurement


class TestOutcomeMeasurementProviderFields:
    """OutcomeMeasurement schema accepts and serializes provider fields."""

    def test_default_empty_strings(self):
        m = OutcomeMeasurement(
            suggestion_id="x",
            implemented_date="2026-01-01",
            measurement_date="2026-01-08",
            window_days=7,
        )
        assert m.source_provider == ""
        assert m.source_model == ""

    def test_accepts_provider_fields(self):
        m = OutcomeMeasurement(
            suggestion_id="x",
            implemented_date="2026-01-01",
            measurement_date="2026-01-08",
            window_days=7,
            bot_id="bot1",
            category="parameter",
            source_provider="claude_max",
            source_model="claude-opus-4-6",
            source_run_id="daily-2026-01-01",
        )
        assert m.bot_id == "bot1"
        assert m.category == "parameter"
        assert m.source_provider == "claude_max"
        assert m.source_model == "claude-opus-4-6"
        assert m.source_run_id == "daily-2026-01-01"

    def test_round_trip_json(self):
        m = OutcomeMeasurement(
            suggestion_id="x",
            implemented_date="2026-01-01",
            measurement_date="2026-01-08",
            window_days=7,
            source_provider="openrouter",
            source_model="deepseek-v3",
        )
        raw = m.model_dump_json()
        restored = OutcomeMeasurement.model_validate_json(raw)
        assert restored.source_provider == "openrouter"
        assert restored.source_model == "deepseek-v3"

    def test_backward_compat_no_provider(self):
        """Existing JSONL entries without provider fields still deserialize."""
        raw = json.dumps({
            "suggestion_id": "x",
            "implemented_date": "2026-01-01",
            "measurement_date": "2026-01-08",
            "window_days": 7,
        })
        m = OutcomeMeasurement.model_validate_json(raw)
        assert m.source_provider == ""
        assert m.source_model == ""


class TestValidationLogProviderAttribution:
    """_validate_and_annotate includes provider/model in log entries."""

    def _make_handlers(self, tmp_path: Path):
        from unittest.mock import MagicMock
        from trading_assistant.orchestrator.handlers import Handlers

        runner = MagicMock()
        runner.session_store = MagicMock()
        return Handlers(
            agent_runner=runner,
            event_stream=MagicMock(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=["bot1"],
        )

    def _make_parsed(self, suggestions=None):
        from unittest.mock import MagicMock
        parsed = MagicMock()
        parsed.raw_report = "test report"
        parsed.suggestions = suggestions or []
        parsed.predictions = []
        parsed.structural_proposals = []
        parsed.portfolio_proposals = []
        return parsed

    def test_log_entry_includes_provider_model(self, tmp_path):
        h = self._make_handlers(tmp_path)
        findings = tmp_path / "memory" / "findings"
        findings.mkdir(parents=True)

        # Create minimal parsed with no suggestions so validation returns quickly
        parsed = self._make_parsed()

        h._validate_and_annotate(
            parsed, "2026-01-01",
            provider="claude_max", model="opus-4",
            run_id="daily-2026-01-01", agent_type="daily_analysis", bot_ids="bot1",
        )

        log_path = findings / "validation_log.jsonl"
        assert log_path.exists(), "validation_log.jsonl should be written"
        entry = json.loads(log_path.read_text().strip().split("\n")[-1])
        assert entry["provider"] == "claude_max"
        assert entry["model"] == "opus-4"
        assert entry["run_id"] == "daily-2026-01-01"
        assert entry["agent_type"] == "daily_analysis"
        assert entry["bot_ids"] == "bot1"

    def test_log_entry_empty_when_not_passed(self, tmp_path):
        h = self._make_handlers(tmp_path)
        findings = tmp_path / "memory" / "findings"
        findings.mkdir(parents=True)

        parsed = self._make_parsed()
        h._validate_and_annotate(parsed, "2026-01-01")

        log_path = findings / "validation_log.jsonl"
        assert log_path.exists(), "validation_log.jsonl should be written"
        entry = json.loads(log_path.read_text().strip().split("\n")[-1])
        assert entry["provider"] == ""
        assert entry["model"] == ""


class TestDetectionContextProviderAttribution:
    """_record_agent_suggestions includes source_provider/source_model in detection_context."""

    def test_detection_context_includes_provider(self, tmp_path):
        from unittest.mock import MagicMock
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.schemas.suggestion_tracking import SuggestionRecord

        runner = MagicMock()
        runner.session_store = MagicMock()

        tracker = MagicMock()
        tracker.record = MagicMock(return_value=True)

        h = Handlers(
            agent_runner=runner,
            event_stream=MagicMock(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=["bot1"],
            suggestion_tracker=tracker,
        )

        # Make a mock validation result with one approved suggestion
        suggestion = MagicMock()
        suggestion.title = "test suggestion"
        suggestion.bot_id = "bot1"
        suggestion.category = "parameter"
        suggestion.evidence_summary = "test"
        suggestion.confidence = 0.8
        suggestion.suggestion_id = ""
        suggestion.target_param = None
        suggestion.proposed_value = None
        suggestion.expected_impact = ""

        validation = MagicMock()
        validation.approved_suggestions = [suggestion]
        validation.blocked_suggestions = []

        parsed = MagicMock()
        parsed.structural_proposals = []

        h._record_agent_suggestions(
            validation, "run1", parsed,
            provider="codex_pro", model="gpt-4.1",
        )

        # Check the SuggestionRecord passed to tracker.record
        assert tracker.record.called
        recorded: SuggestionRecord = tracker.record.call_args[0][0]
        assert recorded.detection_context["source_provider"] == "codex_pro"
        assert recorded.detection_context["source_model"] == "gpt-4.1"

    def test_detection_context_empty_when_no_provider(self, tmp_path):
        from unittest.mock import MagicMock
        from trading_assistant.orchestrator.handlers import Handlers

        runner = MagicMock()
        runner.session_store = MagicMock()

        tracker = MagicMock()
        tracker.record = MagicMock(return_value=True)

        h = Handlers(
            agent_runner=runner,
            event_stream=MagicMock(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=["bot1"],
            suggestion_tracker=tracker,
        )

        suggestion = MagicMock()
        suggestion.title = "test"
        suggestion.bot_id = "bot1"
        suggestion.category = "parameter"
        suggestion.evidence_summary = ""
        suggestion.confidence = 0.5
        suggestion.suggestion_id = ""
        suggestion.target_param = None
        suggestion.proposed_value = None
        suggestion.expected_impact = ""

        validation = MagicMock()
        validation.approved_suggestions = [suggestion]
        validation.blocked_suggestions = []

        parsed = MagicMock()
        parsed.structural_proposals = []

        h._record_agent_suggestions(validation, "run2", parsed)

        recorded = tracker.record.call_args[0][0]
        ctx = recorded.detection_context or {}
        # No provider keys when not passed
        assert "source_provider" not in ctx
        assert "source_model" not in ctx
