"""Tests for LearningWriteCoordinator outcome reasoning wiring (Phase D)."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from trading_assistant.skills.learning_write_coordinator import LearningWriteCoordinator, WriteGroup


@pytest.fixture
def findings_dir(tmp_path: Path) -> Path:
    d = tmp_path / "findings"
    d.mkdir()
    return d


@pytest.fixture
def coordinator(findings_dir: Path) -> LearningWriteCoordinator:
    return LearningWriteCoordinator(findings_dir=findings_dir)


class TestCoordinatorReasoningWrites:
    def _build_reasonings_group(
        self,
        coordinator: LearningWriteCoordinator,
        reasonings: list[dict],
        spurious: list[dict] | None = None,
        recalibs: list[dict] | None = None,
    ) -> WriteGroup:
        """Simulate what _measure_outcomes does after parsing."""
        group = coordinator.begin(
            source_workflow="outcome_reasoning",
            source_run_id="reasoning-2026-04-17",
        )
        coordinator.add_jsonl_append(
            group, "record_reasonings",
            "outcome_reasonings.jsonl", reasonings,
        )
        if spurious:
            coordinator.add_jsonl_append(
                group, "record_spurious",
                "spurious_outcomes.jsonl", spurious,
            )
        if recalibs:
            coordinator.add_jsonl_append(
                group, "record_recalibrations",
                "recalibrations.jsonl", recalibs,
            )
        return group

    def test_all_records_land_in_one_write_group(self, coordinator, findings_dir):
        """Reasoning + spurious + recalibration should share one group_id."""
        reasonings = [{"suggestion_id": "s1", "reasoned_at": "2026-04-17T00:00:00Z"}]
        spurious = [{"suggestion_id": "s2", "mechanism": "random"}]
        recalibs = [{"suggestion_id": "s3", "revised_confidence": 0.4}]

        group = self._build_reasonings_group(coordinator, reasonings, spurious, recalibs)
        result = coordinator.execute(group)

        assert len(result.operations) == 3
        assert result.all_succeeded
        # All 3 files should exist
        assert (findings_dir / "outcome_reasonings.jsonl").exists()
        assert (findings_dir / "spurious_outcomes.jsonl").exists()
        assert (findings_dir / "recalibrations.jsonl").exists()

    def test_shared_group_id_across_write_types(self, coordinator, findings_dir):
        """All records written in one group should share the same _write_group_id."""
        reasonings = [{"suggestion_id": "s1"}]
        spurious = [{"suggestion_id": "s2"}]

        group = self._build_reasonings_group(coordinator, reasonings, spurious)
        result = coordinator.execute(group)
        gid = result.group_id

        r_line = json.loads((findings_dir / "outcome_reasonings.jsonl").read_text().strip())
        s_line = json.loads((findings_dir / "spurious_outcomes.jsonl").read_text().strip())
        assert r_line["_write_group_id"] == gid
        assert s_line["_write_group_id"] == gid

    def test_empty_spurious_recalib_produce_no_write_ops(self, coordinator, findings_dir):
        """When no spurious/recalib records, only the reasonings op should exist."""
        group = self._build_reasonings_group(coordinator, [{"suggestion_id": "s1"}])
        result = coordinator.execute(group)

        assert len(result.operations) == 1
        assert result.operations[0].name == "record_reasonings"
        assert not (findings_dir / "spurious_outcomes.jsonl").exists()
        assert not (findings_dir / "recalibrations.jsonl").exists()

    def test_records_collected_from_loop_before_batch_write(self, coordinator, findings_dir):
        """Simulate the collect-then-write pattern from _measure_outcomes."""
        # Simulate what the refactored code does: loop to collect, then batch write
        reasonings = [
            {"suggestion_id": "s1", "genuine_effect": False, "revised_confidence": 0.3},
            {"suggestion_id": "s2", "genuine_effect": True, "revised_confidence": None},
            {"suggestion_id": "s3", "genuine_effect": False, "revised_confidence": 0.6},
        ]
        spurious_records = []
        recalib_records = []

        for r in reasonings:
            sid = r["suggestion_id"]
            if r.get("genuine_effect") is False:
                spurious_records.append({"suggestion_id": sid, "mechanism": "test"})
            revised = r.get("revised_confidence")
            if revised is not None:
                recalib_records.append({"suggestion_id": sid, "revised_confidence": revised})

        group = self._build_reasonings_group(
            coordinator, reasonings, spurious_records, recalib_records,
        )
        result = coordinator.execute(group)

        assert result.all_succeeded
        assert len(spurious_records) == 2  # s1 and s3
        assert len(recalib_records) == 2  # s1 and s3
        # Verify file contents
        spur_lines = (findings_dir / "spurious_outcomes.jsonl").read_text().strip().splitlines()
        assert len(spur_lines) == 2
        recal_lines = (findings_dir / "recalibrations.jsonl").read_text().strip().splitlines()
        assert len(recal_lines) == 2

    def test_spurious_records_include_bot_id(self, coordinator, findings_dir):
        """Spurious records should carry bot_id from the suggestion lookup."""
        # Simulate the enrichment pattern from _measure_outcomes:
        # suggestion_lookup maps suggestion_id → original suggestion record
        suggestion_lookup = {
            "s1": {"suggestion_id": "s1", "bot_id": "bot_a", "category": "exit_timing"},
        }
        spurious_records = [{
            "suggestion_id": "s1",
            "bot_id": suggestion_lookup.get("s1", {}).get("bot_id", ""),
            "mechanism": "Coincidental regime shift",
            "confounders": ["vol_spike"],
        }]

        group = self._build_reasonings_group(
            coordinator,
            [{"suggestion_id": "s1", "genuine_effect": False}],
            spurious=spurious_records,
        )
        result = coordinator.execute(group)
        assert result.all_succeeded

        line = json.loads(
            (findings_dir / "spurious_outcomes.jsonl").read_text().strip()
        )
        assert line["bot_id"] == "bot_a"
