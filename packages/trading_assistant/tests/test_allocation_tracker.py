"""Tests for AllocationTracker (#19)."""
from pathlib import Path

import pytest

from trading_assistant.schemas.allocation_history import AllocationRecord, AllocationSource
from trading_assistant.skills.allocation_tracker import AllocationTracker


@pytest.fixture
def tracker(tmp_path: Path) -> AllocationTracker:
    return AllocationTracker(findings_dir=tmp_path / "findings")


class TestAllocationTracker:
    def test_record_appends_to_jsonl(self, tracker: AllocationTracker):
        entry = AllocationRecord(
            date="2026-03-01",
            bot_id="bot-a",
            allocation_pct=25.0,
            source=AllocationSource.MANUAL,
            reason="Initial allocation",
        )
        tracker.record(entry)
        records = tracker.load_all()
        assert len(records) == 1
        assert records[0]["bot_id"] == "bot-a"
        assert records[0]["allocation_pct"] == 25.0

    def test_load_all_returns_multiple_records(self, tracker: AllocationTracker):
        for i in range(3):
            tracker.record(AllocationRecord(
                date=f"2026-03-0{i + 1}",
                bot_id="bot-a",
                allocation_pct=20.0 + i,
            ))
        records = tracker.load_all()
        assert len(records) == 3

    def test_creates_parent_dirs(self, tmp_path: Path):
        tracker = AllocationTracker(findings_dir=tmp_path / "deep" / "nested" / "findings")
        tracker.record(AllocationRecord(
            date="2026-03-01",
            bot_id="bot-a",
            allocation_pct=50.0,
        ))
        assert tracker.load_all()[0]["allocation_pct"] == 50.0

    def test_empty_returns_empty_list(self, tracker: AllocationTracker):
        assert tracker.load_all() == []
