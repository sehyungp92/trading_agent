# tests/test_retrospective_synthesis.py
"""Tests for Phase 3 — Retrospective Synthesis + Dynamic Prompt Evolution."""
from __future__ import annotations

import json
from pathlib import Path


from trading_assistant.schemas.learning_ledger import (
    DiscardItem,
    RetrospectiveSynthesis,
    SynthesisItem,
)


# ── Synthesis Schema ──

class TestSynthesisSchemas:
    def test_synthesis_item_schema(self):
        item = SynthesisItem(
            suggestion_id="s1",
            bot_id="bot_a",
            category="exit_timing",
            title="Tighter trailing stop",
            outcome_verdict="positive",
            ground_truth_delta=0.05,
            mechanism="Reduced tail losses by 30%",
        )
        assert item.suggestion_id == "s1"
        assert item.ground_truth_delta == 0.05

    def test_discard_item_schema(self):
        item = DiscardItem(
            bot_id="bot_a",
            category="filter_threshold",
            failure_count=4,
            reason="4 consecutive failures with no successes",
        )
        assert item.failure_count == 4
        assert item.bot_id == "bot_a"

    def test_retrospective_synthesis_defaults(self):
        synth = RetrospectiveSynthesis(
            week_start="2026-03-01",
            week_end="2026-03-07",
        )
        assert synth.what_worked == []
        assert synth.what_failed == []
        assert synth.discard == []
        assert synth.lessons == []
        assert synth.ground_truth_deltas == {}


# ── Build Synthesis ──

class TestBuildSynthesis:
    def _setup_memory(self, tmp_path: Path):
        """Create minimal memory directory structure."""
        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()
        return memory_dir, runs_dir, curated_dir

    def _write_outcomes(self, memory_dir: Path, outcomes: list[dict]):
        path = memory_dir / "findings" / "outcomes.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for o in outcomes:
                f.write(json.dumps(o) + "\n")

    def _write_suggestions(self, memory_dir: Path, suggestions: list[dict]):
        path = memory_dir / "findings" / "suggestions.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in suggestions:
                f.write(json.dumps(s) + "\n")

    def _write_reasonings(self, memory_dir: Path, reasonings: list[dict]):
        path = memory_dir / "findings" / "outcome_reasonings.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in reasonings:
                f.write(json.dumps(r) + "\n")

    def test_cold_start_empty_synthesis(self, tmp_path: Path):
        """No data → empty synthesis with all empty lists."""
        memory_dir, runs_dir, curated_dir = self._setup_memory(tmp_path)
        from trading_assistant.skills.retrospective_builder import RetrospectiveBuilder

        builder = RetrospectiveBuilder(
            runs_dir=runs_dir,
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )
        synth = builder.build_synthesis("2026-03-01", "2026-03-07")
        assert isinstance(synth, RetrospectiveSynthesis)
        assert synth.what_worked == []
        assert synth.what_failed == []
        assert synth.discard == []
        assert synth.lessons == []

    def test_positive_outcome_in_what_worked(self, tmp_path: Path):
        memory_dir, runs_dir, curated_dir = self._setup_memory(tmp_path)
        self._write_outcomes(memory_dir, [
            {
                "suggestion_id": "s1",
                "bot_id": "bot_a",
                "category": "exit_timing",
                "verdict": "positive",
                "pnl_delta": 0.05,
                "measured_at": "2026-03-03T10:00:00+00:00",
                "title": "Tighter trailing stop",
            },
        ])
        self._write_suggestions(memory_dir, [
            {"suggestion_id": "s1", "bot_id": "bot_a", "category": "exit_timing"},
        ])

        from trading_assistant.skills.retrospective_builder import RetrospectiveBuilder

        builder = RetrospectiveBuilder(
            runs_dir=runs_dir, curated_dir=curated_dir, memory_dir=memory_dir,
        )
        synth = builder.build_synthesis("2026-03-01", "2026-03-07")
        assert len(synth.what_worked) == 1
        assert synth.what_worked[0].suggestion_id == "s1"
        assert synth.what_worked[0].outcome_verdict == "positive"

    def test_negative_outcome_in_what_failed(self, tmp_path: Path):
        memory_dir, runs_dir, curated_dir = self._setup_memory(tmp_path)
        self._write_outcomes(memory_dir, [
            {
                "suggestion_id": "s2",
                "bot_id": "bot_a",
                "category": "signal",
                "verdict": "negative",
                "pnl_delta": -0.03,
                "measured_at": "2026-03-04T10:00:00+00:00",
                "title": "Bad signal tweak",
            },
        ])

        from trading_assistant.skills.retrospective_builder import RetrospectiveBuilder

        builder = RetrospectiveBuilder(
            runs_dir=runs_dir, curated_dir=curated_dir, memory_dir=memory_dir,
        )
        synth = builder.build_synthesis("2026-03-01", "2026-03-07")
        assert len(synth.what_failed) == 1
        assert synth.what_failed[0].outcome_verdict == "negative"

    def test_discard_category_threshold(self, tmp_path: Path):
        """Category with 3+ failures and 0 successes → discard."""
        memory_dir, runs_dir, curated_dir = self._setup_memory(tmp_path)
        outcomes = [
            {
                "suggestion_id": f"s{i}",
                "bot_id": "bot_a",
                "category": "filter_threshold",
                "verdict": "negative",
                "pnl_delta": -0.01,
                "measured_at": f"2026-03-0{i+1}T10:00:00+00:00",
                "title": f"Bad filter {i}",
            }
            for i in range(3)
        ]
        self._write_outcomes(memory_dir, outcomes)

        from trading_assistant.skills.retrospective_builder import RetrospectiveBuilder

        builder = RetrospectiveBuilder(
            runs_dir=runs_dir, curated_dir=curated_dir, memory_dir=memory_dir,
        )
        synth = builder.build_synthesis("2026-03-01", "2026-03-07")
        assert len(synth.discard) >= 1
        discard_cats = [(d.bot_id, d.category) for d in synth.discard]
        assert ("bot_a", "filter_threshold") in discard_cats

    def test_no_discard_with_successes(self, tmp_path: Path):
        """Category with failures AND successes → not discarded."""
        memory_dir, runs_dir, curated_dir = self._setup_memory(tmp_path)
        outcomes = [
            {
                "suggestion_id": "s1", "bot_id": "bot_a", "category": "signal",
                "verdict": "negative", "pnl_delta": -0.01,
                "measured_at": "2026-03-02T10:00:00+00:00",
            },
            {
                "suggestion_id": "s2", "bot_id": "bot_a", "category": "signal",
                "verdict": "negative", "pnl_delta": -0.02,
                "measured_at": "2026-03-03T10:00:00+00:00",
            },
            {
                "suggestion_id": "s3", "bot_id": "bot_a", "category": "signal",
                "verdict": "negative", "pnl_delta": -0.01,
                "measured_at": "2026-03-04T10:00:00+00:00",
            },
            {
                "suggestion_id": "s4", "bot_id": "bot_a", "category": "signal",
                "verdict": "positive", "pnl_delta": 0.05,
                "measured_at": "2026-03-05T10:00:00+00:00",
            },
        ]
        self._write_outcomes(memory_dir, outcomes)

        from trading_assistant.skills.retrospective_builder import RetrospectiveBuilder

        builder = RetrospectiveBuilder(
            runs_dir=runs_dir, curated_dir=curated_dir, memory_dir=memory_dir,
        )
        synth = builder.build_synthesis("2026-03-01", "2026-03-07")
        discard_cats = [(d.bot_id, d.category) for d in synth.discard]
        assert ("bot_a", "signal") not in discard_cats

    def test_lessons_extracted_from_reasonings(self, tmp_path: Path):
        memory_dir, runs_dir, curated_dir = self._setup_memory(tmp_path)
        self._write_reasonings(memory_dir, [
            {
                "suggestion_id": "s1",
                "lessons_learned": "Exit timing matters more in volatile regimes",
                "reasoned_at": "2026-03-03T12:00:00+00:00",
            },
            {
                "suggestion_id": "s2",
                "lessons_learned": "Filter changes are regime-dependent",
                "reasoned_at": "2026-03-04T12:00:00+00:00",
            },
        ])

        from trading_assistant.skills.retrospective_builder import RetrospectiveBuilder

        builder = RetrospectiveBuilder(
            runs_dir=runs_dir, curated_dir=curated_dir, memory_dir=memory_dir,
        )
        synth = builder.build_synthesis("2026-03-01", "2026-03-07")
        assert len(synth.lessons) == 2
        assert any("Exit timing" in l for l in synth.lessons)

    def test_lessons_deduplication(self, tmp_path: Path):
        memory_dir, runs_dir, curated_dir = self._setup_memory(tmp_path)
        self._write_reasonings(memory_dir, [
            {
                "suggestion_id": "s1",
                "lessons_learned": "Same lesson",
                "reasoned_at": "2026-03-03T12:00:00+00:00",
            },
            {
                "suggestion_id": "s2",
                "lessons_learned": "Same lesson",
                "reasoned_at": "2026-03-04T12:00:00+00:00",
            },
        ])

        from trading_assistant.skills.retrospective_builder import RetrospectiveBuilder

        builder = RetrospectiveBuilder(
            runs_dir=runs_dir, curated_dir=curated_dir, memory_dir=memory_dir,
        )
        synth = builder.build_synthesis("2026-03-01", "2026-03-07")
        assert len(synth.lessons) == 1

    def test_ground_truth_deltas_loaded(self, tmp_path: Path):
        memory_dir, runs_dir, curated_dir = self._setup_memory(tmp_path)
        ledger_path = memory_dir / "findings" / "learning_ledger.jsonl"
        ledger_path.write_text(json.dumps({
            "week_start": "2026-03-01",
            "week_end": "2026-03-07",
            "composite_delta": {"bot_a": 0.15, "bot_b": -0.03},
        }) + "\n")

        from trading_assistant.skills.retrospective_builder import RetrospectiveBuilder

        builder = RetrospectiveBuilder(
            runs_dir=runs_dir, curated_dir=curated_dir, memory_dir=memory_dir,
        )
        synth = builder.build_synthesis("2026-03-01", "2026-03-07")
        assert synth.ground_truth_deltas.get("bot_a") == 0.15
        assert synth.ground_truth_deltas.get("bot_b") == -0.03

    def test_synthesis_persistence(self, tmp_path: Path):
        memory_dir, runs_dir, curated_dir = self._setup_memory(tmp_path)
        self._write_outcomes(memory_dir, [
            {
                "suggestion_id": "s1", "bot_id": "bot_a", "category": "exit_timing",
                "verdict": "positive", "pnl_delta": 0.05,
                "measured_at": "2026-03-03T10:00:00+00:00",
                "title": "Good change",
            },
        ])

        from trading_assistant.skills.retrospective_builder import RetrospectiveBuilder

        builder = RetrospectiveBuilder(
            runs_dir=runs_dir, curated_dir=curated_dir, memory_dir=memory_dir,
        )
        builder.build_synthesis("2026-03-01", "2026-03-07")

        synth_path = memory_dir / "findings" / "retrospective_synthesis.jsonl"
        assert synth_path.exists()
        lines = synth_path.read_text().strip().splitlines()
        assert len(lines) >= 1
        loaded = json.loads(lines[-1])
        assert loaded["week_start"] == "2026-03-01"

    def test_outcomes_outside_week_excluded(self, tmp_path: Path):
        memory_dir, runs_dir, curated_dir = self._setup_memory(tmp_path)
        self._write_outcomes(memory_dir, [
            {
                "suggestion_id": "s1", "bot_id": "bot_a", "category": "exit_timing",
                "verdict": "positive", "pnl_delta": 0.05,
                "measured_at": "2026-02-25T10:00:00+00:00",  # before week
            },
            {
                "suggestion_id": "s2", "bot_id": "bot_a", "category": "signal",
                "verdict": "negative", "pnl_delta": -0.02,
                "measured_at": "2026-03-10T10:00:00+00:00",  # after week
            },
        ])

        from trading_assistant.skills.retrospective_builder import RetrospectiveBuilder

        builder = RetrospectiveBuilder(
            runs_dir=runs_dir, curated_dir=curated_dir, memory_dir=memory_dir,
        )
        synth = builder.build_synthesis("2026-03-01", "2026-03-07")
        assert synth.what_worked == []
        assert synth.what_failed == []


# ── Category Recalibration ──

class TestCategoryRecalibration:
    def test_apply_recalibration_writes_overrides(self, tmp_path: Path):
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        scorer = SuggestionScorer(findings_dir)
        discard_items = [
            DiscardItem(
                bot_id="bot_a",
                category="filter_threshold",
                failure_count=4,
                reason="4 consecutive failures",
            ),
        ]
        scorer.apply_recalibration(discard_items)

        overrides_path = findings_dir / "category_overrides.jsonl"
        assert overrides_path.exists()
        lines = overrides_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["bot_id"] == "bot_a"
        assert entry["category"] == "filter_threshold"
        assert entry["confidence_multiplier"] == 0.3

    def test_scorecard_uses_overrides(self, tmp_path: Path):
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        # Write suggestions and outcomes
        suggestions = [
            {"suggestion_id": "s1", "bot_id": "bot_a", "category": "exit_timing"},
        ]
        outcomes = [
            {"suggestion_id": "s1", "verdict": "positive", "pnl_delta": 0.05},
        ]
        (findings_dir / "suggestions.jsonl").write_text(
            "\n".join(json.dumps(s) for s in suggestions) + "\n"
        )
        (findings_dir / "outcomes.jsonl").write_text(
            "\n".join(json.dumps(o) for o in outcomes) + "\n"
        )

        # Write override that reduces multiplier
        (findings_dir / "category_overrides.jsonl").write_text(
            json.dumps({
                "bot_id": "bot_a",
                "category": "exit_timing",
                "confidence_multiplier": 0.3,
            }) + "\n"
        )

        scorer = SuggestionScorer(findings_dir)
        scorecard = scorer.compute_scorecard()
        assert len(scorecard.scores) == 1
        assert scorecard.scores[0].confidence_multiplier == 0.3

    def test_scorecard_default_without_overrides(self, tmp_path: Path):
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        suggestions = [
            {"suggestion_id": "s1", "bot_id": "bot_a", "category": "signal"},
        ]
        outcomes = [
            {"suggestion_id": "s1", "verdict": "positive", "pnl_delta": 0.02},
        ]
        (findings_dir / "suggestions.jsonl").write_text(
            "\n".join(json.dumps(s) for s in suggestions) + "\n"
        )
        (findings_dir / "outcomes.jsonl").write_text(
            "\n".join(json.dumps(o) for o in outcomes) + "\n"
        )

        scorer = SuggestionScorer(findings_dir)
        scorecard = scorer.compute_scorecard()
        assert len(scorecard.scores) == 1
        # With only 1 sample (< 5), default multiplier is 1.0
        assert scorecard.scores[0].confidence_multiplier == 1.0

    def test_recalibration_updates_existing_overrides(self, tmp_path: Path):
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        # Pre-existing override
        (findings_dir / "category_overrides.jsonl").write_text(
            json.dumps({
                "bot_id": "bot_a", "category": "signal",
                "confidence_multiplier": 0.5,
            }) + "\n"
        )

        scorer = SuggestionScorer(findings_dir)
        scorer.apply_recalibration([
            DiscardItem(
                bot_id="bot_a", category="signal",
                failure_count=3, reason="test",
            ),
        ])

        lines = (findings_dir / "category_overrides.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1  # updated, not duplicated
        entry = json.loads(lines[0])
        assert entry["confidence_multiplier"] == 0.3  # updated to discard value

    def test_recalibration_preserves_other_overrides(self, tmp_path: Path):
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        # Pre-existing override for different category
        (findings_dir / "category_overrides.jsonl").write_text(
            json.dumps({
                "bot_id": "bot_b", "category": "stop_loss",
                "confidence_multiplier": 0.7,
            }) + "\n"
        )

        scorer = SuggestionScorer(findings_dir)
        scorer.apply_recalibration([
            DiscardItem(
                bot_id="bot_a", category="signal",
                failure_count=3, reason="test",
            ),
        ])

        lines = (findings_dir / "category_overrides.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2  # both preserved
        entries = [json.loads(l) for l in lines]
        bots = {(e["bot_id"], e["category"]) for e in entries}
        assert ("bot_b", "stop_loss") in bots
        assert ("bot_a", "signal") in bots


# ── Prompt Injection ──

class TestPromptInjection:
    def test_daily_prompt_contains_feedback_loop_markers(self):
        from trading_assistant.analysis.prompt_assembler import _FOCUSED_INSTRUCTIONS

        for marker in [
            "PREDICTION TRACK RECORD",
            "accuracy < 50%",
            "accuracy > 70%",
            "GROUND TRUTH PERFORMANCE",
            "BLOCKED APPROACHES",
        ]:
            assert marker in _FOCUSED_INSTRUCTIONS

    def test_weekly_prompt_contains_feedback_loop_markers(self):
        from trading_assistant.analysis.weekly_prompt_assembler import _FOCUSED_WEEKLY_INSTRUCTIONS

        for marker in [
            "PREDICTION TRACK RECORD",
            "systematic biases",
            "GROUND TRUTH PERFORMANCE",
            "LEARNING SYNTHESIS",
            "what_worked",
            "discard",
            "STRATEGY IDEAS",
        ]:
            assert marker in _FOCUSED_WEEKLY_INSTRUCTIONS


# ── Context Builder Integration ──

class TestContextBuilderSynthesis:
    def test_synthesis_loaded_into_base_package(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True)

        synth = {
            "week_start": "2026-03-01",
            "week_end": "2026-03-07",
            "what_worked": [{"suggestion_id": "s1", "bot_id": "bot_a",
                            "category": "exit_timing", "title": "Good",
                            "outcome_verdict": "positive",
                            "ground_truth_delta": 0.05, "mechanism": "test"}],
            "what_failed": [],
            "discard": [],
            "lessons": ["Exit timing matters"],
            "ground_truth_deltas": {"bot_a": 0.05},
        }
        synth_path = memory_dir / "findings" / "retrospective_synthesis.jsonl"
        synth_path.write_text(json.dumps(synth) + "\n")

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()
        assert "last_week_synthesis" in pkg.data

    def test_spurious_outcomes_loaded(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True)

        spurious_path = memory_dir / "findings" / "spurious_outcomes.jsonl"
        spurious_path.write_text(json.dumps({
            "suggestion_id": "s1",
            "mechanism": "Coincidental regime shift",
        }) + "\n")

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()
        assert "spurious_outcomes" in pkg.data
        assert len(pkg.data["spurious_outcomes"]) == 1

    def test_empty_synthesis_not_injected(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True)

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()
        # No synthesis file → key not present (not an empty dict)
        assert "last_week_synthesis" not in pkg.data


# ── Dynamic Evolution ──

class TestDynamicEvolution:
    def test_different_synthesis_produces_different_context(self, tmp_path: Path):
        """Same base_package call, different synthesis files → different data."""
        from trading_assistant.analysis.context_builder import ContextBuilder

        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True)

        synth_path = memory_dir / "findings" / "retrospective_synthesis.jsonl"

        # Week 1 synthesis
        synth1 = {
            "week_start": "2026-03-01", "week_end": "2026-03-07",
            "lessons": ["Lesson A"], "what_worked": [], "what_failed": [],
            "discard": [], "ground_truth_deltas": {},
        }
        synth_path.write_text(json.dumps(synth1) + "\n")
        ctx = ContextBuilder(memory_dir)
        pkg1 = ctx.base_package()
        lessons1 = pkg1.data.get("last_week_synthesis", {}).get("lessons", [])

        # Week 2 synthesis (overwrite)
        synth2 = {
            "week_start": "2026-03-08", "week_end": "2026-03-14",
            "lessons": ["Lesson B", "Lesson C"], "what_worked": [], "what_failed": [],
            "discard": [], "ground_truth_deltas": {},
        }
        synth_path.write_text(json.dumps(synth1) + "\n" + json.dumps(synth2) + "\n")
        ctx2 = ContextBuilder(memory_dir)
        pkg2 = ctx2.base_package()
        lessons2 = pkg2.data.get("last_week_synthesis", {}).get("lessons", [])

        assert lessons1 != lessons2
        assert lessons1 == ["Lesson A"]
        assert lessons2 == ["Lesson B", "Lesson C"]
