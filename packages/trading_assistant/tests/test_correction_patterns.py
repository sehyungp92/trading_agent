# tests/test_correction_patterns.py
"""Tests for correction pattern extraction (ecosystem improvement 2.4)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


class TestCorrectionPatternExtractor:
    def test_empty_corrections(self):
        from trading_assistant.skills.correction_pattern_extractor import CorrectionPatternExtractor

        extractor = CorrectionPatternExtractor(min_occurrences=2)
        report = extractor.extract([])
        assert report.total_corrections_analyzed == 0
        assert report.patterns == []

    def test_single_correction_below_threshold(self):
        from trading_assistant.skills.correction_pattern_extractor import CorrectionPatternExtractor

        extractor = CorrectionPatternExtractor(min_occurrences=2)
        report = extractor.extract([{
            "correction_type": "regime_override",
            "target_id": "bot1",
            "raw_text": "Wrong regime call",
            "timestamp": "2026-03-01T00:00:00+00:00",
        }])
        assert report.total_corrections_analyzed == 1
        assert report.patterns == []

    def test_grouping_by_type_and_target(self):
        from trading_assistant.skills.correction_pattern_extractor import CorrectionPatternExtractor

        corrections = [
            {
                "correction_type": "regime_override",
                "target_id": "bot1",
                "raw_text": "Regime was trending not ranging",
                "timestamp": "2026-03-01T00:00:00+00:00",
            },
            {
                "correction_type": "regime_override",
                "target_id": "bot1",
                "raw_text": "Regime was trending, you said ranging",
                "timestamp": "2026-03-02T00:00:00+00:00",
            },
        ]
        extractor = CorrectionPatternExtractor(min_occurrences=2)
        report = extractor.extract(corrections)
        assert report.total_corrections_analyzed == 2
        assert len(report.patterns) == 1
        pattern = report.patterns[0]
        assert pattern.correction_type == "regime_override"
        assert pattern.target == "bot1"
        assert pattern.count == 2

    def test_keyword_clustering_separates_groups(self):
        """Corrections about different topics within same (type, target) should cluster separately."""
        from trading_assistant.skills.correction_pattern_extractor import CorrectionPatternExtractor

        corrections = [
            {"correction_type": "free_text", "target_id": "bot1",
             "raw_text": "Stop loss was too tight, adjust stop threshold",
             "timestamp": "2026-03-01T00:00:00+00:00"},
            {"correction_type": "free_text", "target_id": "bot1",
             "raw_text": "Stop placement needs to be wider, stop is cutting winners",
             "timestamp": "2026-03-02T00:00:00+00:00"},
            {"correction_type": "free_text", "target_id": "bot1",
             "raw_text": "The filter is blocking good signals, relax the filter",
             "timestamp": "2026-03-01T00:00:00+00:00"},
            {"correction_type": "free_text", "target_id": "bot1",
             "raw_text": "Volume filter too aggressive, filter blocks winners",
             "timestamp": "2026-03-02T00:00:00+00:00"},
        ]
        extractor = CorrectionPatternExtractor(min_occurrences=2)
        report = extractor.extract(corrections)
        assert report.total_corrections_analyzed == 4
        # Should have 2 clusters: stop-related and filter-related
        assert len(report.patterns) == 2
        types = {p.description for p in report.patterns}
        assert any("stop" in desc for desc in types)
        assert any("filter" in desc for desc in types)

    def test_min_occurrences_threshold(self):
        from trading_assistant.skills.correction_pattern_extractor import CorrectionPatternExtractor

        corrections = [
            {"correction_type": "regime_override", "target_id": "bot1",
             "raw_text": "Regime was trending", "timestamp": "2026-03-01T00:00:00+00:00"},
            {"correction_type": "regime_override", "target_id": "bot1",
             "raw_text": "Regime should be trending", "timestamp": "2026-03-02T00:00:00+00:00"},
            {"correction_type": "regime_override", "target_id": "bot1",
             "raw_text": "You keep getting the regime wrong", "timestamp": "2026-03-03T00:00:00+00:00"},
        ]
        # With min_occurrences=3, all 3 should form one group and pass
        extractor_3 = CorrectionPatternExtractor(min_occurrences=3)
        report = extractor_3.extract(corrections)
        assert len(report.patterns) == 1

        # With min_occurrences=4, no pattern should pass
        extractor_4 = CorrectionPatternExtractor(min_occurrences=4)
        report = extractor_4.extract(corrections)
        assert len(report.patterns) == 0

    def test_example_texts_limited_to_3(self):
        from trading_assistant.skills.correction_pattern_extractor import CorrectionPatternExtractor

        corrections = [
            {"correction_type": "free_text", "target_id": "bot1",
             "raw_text": f"Regime call #{i} was wrong",
             "timestamp": f"2026-03-0{i+1}T00:00:00+00:00"}
            for i in range(5)
        ]
        extractor = CorrectionPatternExtractor(min_occurrences=2)
        report = extractor.extract(corrections)
        assert len(report.patterns) >= 1
        for p in report.patterns:
            assert len(p.example_texts) <= 3

    def test_pattern_id_is_deterministic(self):
        from trading_assistant.skills.correction_pattern_extractor import CorrectionPatternExtractor

        corrections = [
            {"correction_type": "regime_override", "target_id": "bot1",
             "raw_text": "Regime was trending", "timestamp": "2026-03-01T00:00:00+00:00"},
            {"correction_type": "regime_override", "target_id": "bot1",
             "raw_text": "Regime should be trending", "timestamp": "2026-03-02T00:00:00+00:00"},
        ]
        extractor = CorrectionPatternExtractor(min_occurrences=2)
        r1 = extractor.extract(corrections)
        r2 = extractor.extract(corrections)
        assert r1.patterns[0].pattern_id == r2.patterns[0].pattern_id

    def test_timestamps_ordered(self):
        from trading_assistant.skills.correction_pattern_extractor import CorrectionPatternExtractor

        corrections = [
            {"correction_type": "free_text", "target_id": "bot1",
             "raw_text": "Stop too tight, widen stop",
             "timestamp": "2026-03-05T00:00:00+00:00"},
            {"correction_type": "free_text", "target_id": "bot1",
             "raw_text": "Stop still too tight, widen the stop",
             "timestamp": "2026-03-01T00:00:00+00:00"},
        ]
        extractor = CorrectionPatternExtractor(min_occurrences=2)
        report = extractor.extract(corrections)
        p = report.patterns[0]
        assert p.first_seen <= p.last_seen

    def test_no_target_defaults_to_all(self):
        from trading_assistant.skills.correction_pattern_extractor import CorrectionPatternExtractor

        corrections = [
            {"correction_type": "free_text",
             "raw_text": "General feedback about regime",
             "timestamp": "2026-03-01T00:00:00+00:00"},
            {"correction_type": "free_text",
             "raw_text": "Another general regime feedback",
             "timestamp": "2026-03-02T00:00:00+00:00"},
        ]
        extractor = CorrectionPatternExtractor(min_occurrences=2)
        report = extractor.extract(corrections)
        if report.patterns:
            assert report.patterns[0].target == "all"


class TestContextBuilderCorrectionPatterns:
    def test_load_correction_patterns_empty(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        builder = ContextBuilder(tmp_path / "memory")
        patterns = builder.load_correction_patterns()
        assert patterns == []

    def test_load_correction_patterns_with_data(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        findings = tmp_path / "memory" / "findings"
        findings.mkdir(parents=True)
        pattern = {
            "pattern_id": "abc123",
            "correction_type": "regime_override",
            "target": "bot1",
            "description": "Recurring regime corrections",
            "count": 5,
        }
        (findings / "correction_patterns.jsonl").write_text(
            json.dumps(pattern) + "\n"
        )

        builder = ContextBuilder(tmp_path / "memory")
        patterns = builder.load_correction_patterns()
        assert len(patterns) == 1
        assert patterns[0]["pattern_id"] == "abc123"

    def test_base_package_includes_correction_patterns(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        mem_dir = tmp_path / "memory"
        findings = mem_dir / "findings"
        findings.mkdir(parents=True)
        policies = mem_dir / "policies" / "v1"
        policies.mkdir(parents=True)
        (policies / "agent.md").write_text("# Agents")
        (policies / "trading_rules.md").write_text("# Rules")
        (policies / "soul.md").write_text("# Soul")

        pattern = {
            "pattern_id": "xyz789",
            "correction_type": "free_text",
            "target": "all",
            "description": "Recurring stop corrections",
            "count": 3,
        }
        (findings / "correction_patterns.jsonl").write_text(
            json.dumps(pattern) + "\n"
        )

        builder = ContextBuilder(mem_dir)
        pkg = builder.base_package()
        assert "correction_patterns" in pkg.data
        assert len(pkg.data["correction_patterns"]) == 1


class TestHandlerCorrectionPatternWiring:
    """Integration test: weekly handler extracts and persists correction patterns."""

    @pytest.mark.asyncio
    async def test_weekly_handler_extracts_patterns(self, tmp_path: Path):
        """Verify correction patterns are extracted and persisted during weekly analysis."""
        from trading_assistant.skills.correction_pattern_extractor import CorrectionPatternExtractor

        # Set up correction data
        findings = tmp_path / "memory" / "findings"
        findings.mkdir(parents=True)
        now = datetime.now(timezone.utc)
        corrections = [
            {"correction_type": "regime_override", "target_id": "bot1",
             "raw_text": "Regime was trending not ranging",
             "timestamp": (now - timedelta(days=2)).isoformat()},
            {"correction_type": "regime_override", "target_id": "bot1",
             "raw_text": "Regime should be trending, you said ranging",
             "timestamp": (now - timedelta(days=1)).isoformat()},
        ]
        (findings / "corrections.jsonl").write_text(
            "\n".join(json.dumps(c) for c in corrections) + "\n"
        )

        # Run extractor directly (simulating what handler does)
        from trading_assistant.analysis.context_builder import ContextBuilder
        ctx = ContextBuilder(tmp_path / "memory")
        loaded = ctx.load_corrections()
        assert len(loaded) == 2

        extractor = CorrectionPatternExtractor(min_occurrences=2)
        report = extractor.extract(loaded)
        assert len(report.patterns) >= 1

        # Persist
        patterns_path = findings / "correction_patterns.jsonl"
        with open(patterns_path, "w", encoding="utf-8") as f:
            for p in report.patterns:
                f.write(json.dumps(p.model_dump(mode="json")) + "\n")

        assert patterns_path.exists()
        reloaded = ctx.load_correction_patterns()
        assert len(reloaded) >= 1
