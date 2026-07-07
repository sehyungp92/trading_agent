# tests/test_reliability_tracker.py
"""Tests for reliability tracking (Section B)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trading_assistant.schemas.reliability_learning import (
    BugClass,
    InterventionStatus,
    ReliabilityIntervention,
    ReliabilityScorecard,
    ReliabilitySummary,
)
from trading_assistant.skills.reliability_tracker import ReliabilityTracker


class TestReliabilityIntervention:
    def test_is_within_observation_true(self):
        i = ReliabilityIntervention(
            intervention_id="test1", bot_id="bot1",
            bug_class=BugClass.CONNECTION,
            observation_window_days=14,
        )
        assert i.is_within_observation is True

    def test_is_within_observation_false(self):
        i = ReliabilityIntervention(
            intervention_id="test1", bot_id="bot1",
            bug_class=BugClass.CONNECTION,
            observation_window_days=14,
            opened_at=datetime.now(timezone.utc) - timedelta(days=20),
        )
        assert i.is_within_observation is False

    def test_default_status(self):
        i = ReliabilityIntervention(
            intervention_id="test1", bot_id="bot1",
            bug_class=BugClass.LOGIC,
        )
        assert i.status == InterventionStatus.OPEN
        assert i.recurrence_count == 0


class TestReliabilityTracker:
    @pytest.fixture
    def tracker(self, tmp_path):
        return ReliabilityTracker(store_dir=tmp_path)

    def test_generate_id_deterministic(self):
        id1 = ReliabilityTracker.generate_id("bot1", "connection", "2026-03-14")
        id2 = ReliabilityTracker.generate_id("bot1", "connection", "2026-03-14")
        assert id1 == id2
        assert id1.startswith("rel_")

    def test_generate_id_differs(self):
        id1 = ReliabilityTracker.generate_id("bot1", "connection", "2026-03-14")
        id2 = ReliabilityTracker.generate_id("bot2", "connection", "2026-03-14")
        assert id1 != id2

    def test_record_intervention(self, tracker):
        intervention = ReliabilityIntervention(
            intervention_id="rel_001", bot_id="bot1",
            bug_class=BugClass.CONNECTION,
            fix_description="Fixed reconnect logic",
        )
        assert tracker.record_intervention(intervention) is True

    def test_record_intervention_dedup(self, tracker):
        intervention = ReliabilityIntervention(
            intervention_id="rel_001", bot_id="bot1",
            bug_class=BugClass.CONNECTION,
        )
        tracker.record_intervention(intervention)
        assert tracker.record_intervention(intervention) is False

    def test_record_recurrence_matches(self, tracker):
        intervention = ReliabilityIntervention(
            intervention_id="rel_001", bot_id="bot1",
            bug_class=BugClass.CONNECTION,
            error_category="timeout",
        )
        tracker.record_intervention(intervention)
        matched = tracker.record_recurrence("bot1", BugClass.CONNECTION, "timeout")
        assert matched == "rel_001"

    def test_record_recurrence_no_match(self, tracker):
        intervention = ReliabilityIntervention(
            intervention_id="rel_001", bot_id="bot1",
            bug_class=BugClass.CONNECTION,
            error_category="timeout",
        )
        tracker.record_intervention(intervention)
        matched = tracker.record_recurrence("bot2", BugClass.CONNECTION, "timeout")
        assert matched is None

    def test_record_recurrence_updates_status(self, tracker):
        intervention = ReliabilityIntervention(
            intervention_id="rel_001", bot_id="bot1",
            bug_class=BugClass.LOGIC,
            error_category="assertion",
        )
        tracker.record_intervention(intervention)
        tracker.record_recurrence("bot1", BugClass.LOGIC, "assertion")

        records = tracker._load_all()
        assert records[0].status == InterventionStatus.RECURRED
        assert records[0].recurrence_count == 1

    def test_verify_completed_auto_verifies(self, tracker):
        intervention = ReliabilityIntervention(
            intervention_id="rel_001", bot_id="bot1",
            bug_class=BugClass.CONFIG,
            observation_window_days=0,  # Immediately past window
            opened_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        tracker.record_intervention(intervention)
        verified = tracker.verify_completed()
        assert "rel_001" in verified

        records = tracker._load_all()
        assert records[0].status == InterventionStatus.VERIFIED
        assert records[0].verified_at is not None

    def test_verify_completed_skips_within_window(self, tracker):
        intervention = ReliabilityIntervention(
            intervention_id="rel_001", bot_id="bot1",
            bug_class=BugClass.CONFIG,
            observation_window_days=30,
        )
        tracker.record_intervention(intervention)
        verified = tracker.verify_completed()
        assert verified == []

    def test_verify_completed_marks_recurred(self, tracker):
        intervention = ReliabilityIntervention(
            intervention_id="rel_001", bot_id="bot1",
            bug_class=BugClass.TIMING,
            observation_window_days=0,
            opened_at=datetime.now(timezone.utc) - timedelta(days=1),
            recurrence_count=2,
        )
        tracker.record_intervention(intervention)
        verified = tracker.verify_completed()
        assert verified == []

        records = tracker._load_all()
        assert records[0].status == InterventionStatus.RECURRED

    def test_compute_summary_empty(self, tracker):
        summary = tracker.compute_summary()
        assert summary.total_open == 0
        assert summary.chronic_bug_classes == []
        assert summary.scorecards_by_class == {}

    def test_compute_summary_with_data(self, tracker):
        for i in range(3):
            tracker.record_intervention(ReliabilityIntervention(
                intervention_id=f"rel_{i:03d}", bot_id="bot1",
                bug_class=BugClass.CONNECTION,
                error_category="timeout",
            ))
        # Record 3 recurrences
        for _ in range(3):
            # Reset status to OPEN for each test
            records = tracker._load_all()
            for r in records:
                if r.status == InterventionStatus.RECURRED:
                    r.status = InterventionStatus.OPEN
            tracker._save_all(records)
            tracker.record_recurrence("bot1", BugClass.CONNECTION, "timeout")

        summary = tracker.compute_summary()
        assert "connection" in summary.scorecards_by_class
        sc = summary.scorecards_by_class["connection"]
        assert sc.intervention_count == 3
        assert "connection" in summary.chronic_bug_classes


class TestHypothesisLibraryReliability:
    def test_create_from_reliability(self, tmp_path):
        from trading_assistant.skills.hypothesis_library import HypothesisLibrary

        lib = HypothesisLibrary(tmp_path)
        summary = ReliabilitySummary(
            scorecards_by_class={
                "connection": ReliabilityScorecard(
                    bug_class=BugClass.CONNECTION,
                    intervention_count=5,
                    verified_count=1,
                    recurrence_rate=0.6,
                ),
            },
            chronic_bug_classes=["connection"],
            total_open=3,
        )
        created = lib.create_from_reliability(summary)
        assert len(created) == 1
        # Check hypothesis was actually created
        records = lib.get_all_records()
        reliability_records = [r for r in records if "reliability" in r.category]
        assert len(reliability_records) == 1
        assert "connection" in reliability_records[0].title.lower()

    def test_create_from_reliability_empty(self, tmp_path):
        from trading_assistant.skills.hypothesis_library import HypothesisLibrary

        lib = HypothesisLibrary(tmp_path)
        summary = ReliabilitySummary()
        created = lib.create_from_reliability(summary)
        assert created == []


class TestHandlersMapErrorToBugClass:
    def test_maps_known_categories(self):
        from trading_assistant.orchestrator.handlers import Handlers

        assert Handlers._map_error_to_bug_class("connection_timeout") == BugClass.CONNECTION
        assert Handlers._map_error_to_bug_class("data_integrity_check") == BugClass.DATA_INTEGRITY
        assert Handlers._map_error_to_bug_class("config_missing") == BugClass.CONFIG
        assert Handlers._map_error_to_bug_class("logic_error") == BugClass.LOGIC

    def test_maps_unknown(self):
        from trading_assistant.orchestrator.handlers import Handlers

        assert Handlers._map_error_to_bug_class("something_else") == BugClass.UNKNOWN
        assert Handlers._map_error_to_bug_class("") == BugClass.UNKNOWN
