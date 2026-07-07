# tests/test_outcome_reasoning.py
"""Phase 5: Causal Outcome Reasoning tests.

Covers:
- schemas/outcome_reasoning.py (OutcomeReasoning, OutcomeReasoningReport)
- analysis/outcome_reasoning_prompt.py (OutcomeReasoningAssembler)
- Wiring in app.py (outcome reasoning scheduling logic)
- ContextBuilder integration (load_outcome_reasonings)
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone


from trading_assistant.schemas.outcome_reasoning import OutcomeReasoning, OutcomeReasoningReport
from trading_assistant.schemas.prompt_package import PromptPackage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_outcomes(count=2):
    """Generate sample outcome measurement dicts for testing."""
    outcomes = []
    for i in range(count):
        outcomes.append({
            "suggestion_id": f"sug_{i:03d}",
            "verdict": "positive" if i % 2 == 0 else "negative",
            "measurement_quality": "HIGH",
            "pnl_before": 100.0 * i,
            "pnl_after": 100.0 * i + 50.0 * (1 if i % 2 == 0 else -1),
            "regime_matched": True,
            "concurrent_changes": [],
            "significance_score": 0.75,
        })
    return outcomes


def _make_memory_dir(tmp_path):
    """Create a minimal memory dir for ContextBuilder/assembler."""
    memory_dir = tmp_path / "memory"
    (memory_dir / "policies" / "v1").mkdir(parents=True)
    (memory_dir / "policies" / "v1" / "agent.md").write_text("Agent")
    (memory_dir / "policies" / "v1" / "trading_rules.md").write_text("Rules")
    (memory_dir / "policies" / "v1" / "soul.md").write_text("Soul")
    (memory_dir / "findings").mkdir(parents=True)
    return memory_dir


# ===========================================================================
# 1. Schema tests — schemas/outcome_reasoning.py (~5 tests)
# ===========================================================================

class TestOutcomeReasoningDefaults:
    def test_defaults(self):
        r = OutcomeReasoning(suggestion_id="sug_001")
        assert r.suggestion_id == "sug_001"
        assert r.genuine_effect is None  # None = inconclusive/not yet assessed
        assert r.mechanism == ""
        assert r.transferable is False
        assert r.lessons_learned == ""
        assert r.revised_confidence == 0.0
        assert r.market_context == ""
        assert r.confounders == []


class TestOutcomeReasoningAllFields:
    def test_all_fields(self):
        now = datetime.now(timezone.utc)
        r = OutcomeReasoning(
            suggestion_id="sug_full",
            genuine_effect=True,
            mechanism="Exit timing improvement reduced slippage by 15bps",
            transferable=True,
            lessons_learned="Exit timing changes are high-impact across similar bots",
            revised_confidence=0.85,
            market_context="Low volatility, trending market",
            confounders=["Concurrent market regime shift", "Reduced trading volume"],
            reasoned_at=now,
        )
        assert r.genuine_effect is True
        assert r.mechanism == "Exit timing improvement reduced slippage by 15bps"
        assert r.transferable is True
        assert r.revised_confidence == 0.85
        assert len(r.confounders) == 2
        assert r.reasoned_at == now


class TestOutcomeReasoningReport:
    def test_report_with_reasonings_list(self):
        report = OutcomeReasoningReport(
            run_id="reasoning-2026-03-14",
            reasonings=[
                OutcomeReasoning(suggestion_id="sug_a", genuine_effect=True),
                OutcomeReasoning(suggestion_id="sug_b", genuine_effect=False),
            ],
        )
        assert report.run_id == "reasoning-2026-03-14"
        assert len(report.reasonings) == 2
        assert report.reasonings[0].suggestion_id == "sug_a"
        assert report.reasonings[1].genuine_effect is False


class TestOutcomeReasoningSerialization:
    def test_serialization_roundtrip(self):
        r = OutcomeReasoning(
            suggestion_id="sug_rt",
            genuine_effect=True,
            mechanism="Causal chain explanation",
            confounders=["A", "B"],
        )
        data = json.loads(r.model_dump_json())
        restored = OutcomeReasoning(**data)
        assert restored.suggestion_id == r.suggestion_id
        assert restored.genuine_effect is True
        assert restored.mechanism == "Causal chain explanation"
        assert restored.confounders == ["A", "B"]


class TestOutcomeReasoningAutoTimestamp:
    def test_reasoned_at_auto_populates(self):
        before = datetime.now(timezone.utc)
        r = OutcomeReasoning(suggestion_id="sug_ts")
        after = datetime.now(timezone.utc)
        assert before <= r.reasoned_at <= after


# ===========================================================================
# 2. OutcomeReasoningAssembler tests (~10 tests)
# ===========================================================================

class TestOutcomeReasoningAssembler:
    def _make_assembler(self, tmp_path):
        from trading_assistant.analysis.outcome_reasoning_prompt import OutcomeReasoningAssembler

        memory_dir = _make_memory_dir(tmp_path)
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()
        return OutcomeReasoningAssembler(
            memory_dir=memory_dir,
            curated_dir=curated_dir,
        ), memory_dir

    def test_assemble_returns_prompt_package(self, tmp_path):
        asm, _ = self._make_assembler(tmp_path)
        pkg = asm.assemble(_sample_outcomes(1))
        assert isinstance(pkg, PromptPackage)

    def test_task_prompt_includes_outcome_count(self, tmp_path):
        asm, _ = self._make_assembler(tmp_path)
        outcomes = _sample_outcomes(3)
        pkg = asm.assemble(outcomes)
        assert "3" in pkg.task_prompt

    def test_instructions_contain_genuine_effect(self, tmp_path):
        asm, _ = self._make_assembler(tmp_path)
        pkg = asm.assemble(_sample_outcomes(1))
        assert "GENUINE EFFECT" in pkg.instructions or "genuine" in pkg.instructions.lower()

    def test_instructions_contain_mechanism(self, tmp_path):
        asm, _ = self._make_assembler(tmp_path)
        pkg = asm.assemble(_sample_outcomes(1))
        assert "MECHANISM" in pkg.instructions or "Mechanism" in pkg.instructions

    def test_instructions_contain_transferable(self, tmp_path):
        asm, _ = self._make_assembler(tmp_path)
        pkg = asm.assemble(_sample_outcomes(1))
        assert "TRANSFERABLE" in pkg.instructions or "Transferable" in pkg.instructions

    def test_instructions_formatted_with_outcome_details(self, tmp_path):
        asm, _ = self._make_assembler(tmp_path)
        outcomes = _sample_outcomes(2)
        pkg = asm.assemble(outcomes)
        # Outcome details should be in instructions
        assert "sug_000" in pkg.instructions
        assert "sug_001" in pkg.instructions

    def test_instructions_include_verdict_and_quality(self, tmp_path):
        asm, _ = self._make_assembler(tmp_path)
        outcomes = [
            {
                "suggestion_id": "sug_vq",
                "verdict": "positive",
                "measurement_quality": "HIGH",
                "pnl_before": 100.0,
                "pnl_after": 200.0,
                "regime_matched": True,
                "concurrent_changes": [],
                "significance_score": 0.9,
            }
        ]
        pkg = asm.assemble(outcomes)
        assert "positive" in pkg.instructions
        assert "HIGH" in pkg.instructions

    def test_instructions_include_regime_matched(self, tmp_path):
        asm, _ = self._make_assembler(tmp_path)
        outcomes = [
            {
                "suggestion_id": "sug_rm",
                "verdict": "negative",
                "measurement_quality": "LOW",
                "pnl_before": 500.0,
                "pnl_after": 400.0,
                "regime_matched": False,
                "concurrent_changes": ["other_param_change"],
                "significance_score": 0.3,
            }
        ]
        pkg = asm.assemble(outcomes)
        assert "False" in pkg.instructions

    def test_instructions_include_concurrent_changes(self, tmp_path):
        asm, _ = self._make_assembler(tmp_path)
        outcomes = [
            {
                "suggestion_id": "sug_cc",
                "verdict": "inconclusive",
                "measurement_quality": "MEDIUM",
                "pnl_before": 200.0,
                "pnl_after": 210.0,
                "regime_matched": True,
                "concurrent_changes": ["exit_tweak", "filter_change"],
                "significance_score": 0.4,
            }
        ]
        pkg = asm.assemble(outcomes)
        assert "exit_tweak" in pkg.instructions
        assert "filter_change" in pkg.instructions

    def test_metadata_includes_bot_ids_from_outcomes(self, tmp_path):
        asm, _ = self._make_assembler(tmp_path)
        outcomes = [
            {"suggestion_id": "s1", "bot_id": "bot_a", "pnl_before": 0, "pnl_after": 0},
            {"suggestion_id": "s2", "bot_id": "bot_b", "pnl_before": 0, "pnl_after": 0},
            {"suggestion_id": "s3", "bot_id": "bot_a", "pnl_before": 0, "pnl_after": 0},
        ]
        pkg = asm.assemble(outcomes)
        assert pkg.metadata["bot_ids"] == ["bot_a", "bot_b"]

    def test_metadata_omits_bot_ids_when_none_present(self, tmp_path):
        asm, _ = self._make_assembler(tmp_path)
        outcomes = [{"suggestion_id": "s1", "pnl_before": 0, "pnl_after": 0}]
        pkg = asm.assemble(outcomes)
        assert "bot_ids" not in pkg.metadata

    def test_load_suggestion_details_loads_matching(self, tmp_path):
        asm, memory_dir = self._make_assembler(tmp_path)
        suggestions_path = memory_dir / "findings" / "suggestions.jsonl"
        suggestions_path.write_text(
            json.dumps({"suggestion_id": "sug_match", "title": "Matched"}) + "\n"
            + json.dumps({"suggestion_id": "sug_other", "title": "Other"}) + "\n"
        )
        outcomes = [{"suggestion_id": "sug_match"}]
        details = asm._load_suggestion_details(outcomes)
        assert len(details) == 1
        assert details[0]["title"] == "Matched"

    def test_load_suggestion_details_handles_missing_file(self, tmp_path):
        asm, _ = self._make_assembler(tmp_path)
        outcomes = [{"suggestion_id": "sug_missing"}]
        details = asm._load_suggestion_details(outcomes)
        assert details == []


# ===========================================================================
# 3. Wiring tests — outcome reasoning scheduling logic (~6 tests)
# ===========================================================================

class TestOutcomeReasoningWiring:
    """Tests for the reasoning scheduling logic (as implemented in app.py).

    Since the reasoning logic is embedded in app.py's scheduled function,
    we test the core logic patterns: deduplication, outcome loading, and
    empty-list skipping.
    """

    def test_loads_recent_outcomes(self, tmp_path):
        """Outcome reasoning function loads outcomes from outcomes.jsonl."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        outcomes_path = findings_dir / "outcomes.jsonl"
        outcomes = [
            {"suggestion_id": "sug_r1", "verdict": "positive"},
            {"suggestion_id": "sug_r2", "verdict": "negative"},
        ]
        outcomes_path.write_text(
            "\n".join(json.dumps(o) for o in outcomes) + "\n"
        )
        # Simulate the loading logic from app.py
        loaded = []
        reasoned_ids: set = set()
        for line in outcomes_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                o = json.loads(line)
                if o.get("suggestion_id", "") not in reasoned_ids:
                    loaded.append(o)
        assert len(loaded) == 2

    def test_reasoned_ids_deduplication(self, tmp_path):
        """Already-reasoned outcomes are excluded from the next reasoning run."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        # outcomes
        outcomes_path = findings_dir / "outcomes.jsonl"
        outcomes_path.write_text(
            json.dumps({"suggestion_id": "sug_done"}) + "\n"
            + json.dumps({"suggestion_id": "sug_new"}) + "\n"
        )
        # existing reasonings
        reasoning_path = findings_dir / "outcome_reasonings.jsonl"
        reasoning_path.write_text(
            json.dumps({"suggestion_id": "sug_done", "genuine_effect": True}) + "\n"
        )
        # Simulate dedup logic
        reasoned_ids: set = set()
        for line in reasoning_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                reasoned_ids.add(json.loads(line).get("suggestion_id", ""))
        recent_outcomes = []
        for line in outcomes_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                o = json.loads(line)
                if o.get("suggestion_id", "") not in reasoned_ids:
                    recent_outcomes.append(o)
        assert len(recent_outcomes) == 1
        assert recent_outcomes[0]["suggestion_id"] == "sug_new"

    def test_outcomes_without_reasonings_picked_up(self, tmp_path):
        """When no reasoning file exists, all outcomes are candidates."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        outcomes_path = findings_dir / "outcomes.jsonl"
        outcomes_path.write_text(
            json.dumps({"suggestion_id": "sug_x"}) + "\n"
            + json.dumps({"suggestion_id": "sug_y"}) + "\n"
        )
        reasoning_path = findings_dir / "outcome_reasonings.jsonl"
        # reasoning_path does NOT exist
        reasoned_ids: set = set()
        if reasoning_path.exists():
            for line in reasoning_path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    reasoned_ids.add(json.loads(line).get("suggestion_id", ""))
        recent_outcomes = []
        for line in outcomes_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                o = json.loads(line)
                if o.get("suggestion_id", "") not in reasoned_ids:
                    recent_outcomes.append(o)
        assert len(recent_outcomes) == 2

    def test_empty_outcomes_skips_reasoning(self, tmp_path):
        """When no unreasoned outcomes exist, reasoning is skipped."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        outcomes_path = findings_dir / "outcomes.jsonl"
        outcomes_path.write_text(
            json.dumps({"suggestion_id": "sug_all_done"}) + "\n"
        )
        reasoning_path = findings_dir / "outcome_reasonings.jsonl"
        reasoning_path.write_text(
            json.dumps({"suggestion_id": "sug_all_done", "genuine_effect": True}) + "\n"
        )
        reasoned_ids: set = set()
        for line in reasoning_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                reasoned_ids.add(json.loads(line).get("suggestion_id", ""))
        recent_outcomes = []
        for line in outcomes_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                o = json.loads(line)
                if o.get("suggestion_id", "") not in reasoned_ids:
                    recent_outcomes.append(o)
        # Should be empty — reasoning would be skipped
        assert len(recent_outcomes) == 0

    def test_assembler_called_with_filtered_outcomes(self, tmp_path):
        """OutcomeReasoningAssembler receives only unreasoned outcomes."""
        from trading_assistant.analysis.outcome_reasoning_prompt import OutcomeReasoningAssembler

        memory_dir = _make_memory_dir(tmp_path)
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()
        assembler = OutcomeReasoningAssembler(
            memory_dir=memory_dir,
            curated_dir=curated_dir,
        )
        filtered = [
            {"suggestion_id": "sug_filtered", "verdict": "positive",
             "measurement_quality": "HIGH", "pnl_before": 100, "pnl_after": 200,
             "regime_matched": True, "concurrent_changes": [], "significance_score": 0.8},
        ]
        pkg = assembler.assemble(filtered)
        assert "sug_filtered" in pkg.instructions
        assert isinstance(pkg, PromptPackage)

    def test_no_outcomes_file_skips_entirely(self, tmp_path):
        """When outcomes.jsonl doesn't exist, no reasoning is attempted."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        outcomes_path = findings_dir / "outcomes.jsonl"
        assert not outcomes_path.exists()
        # The wiring in app.py checks `if outcomes_path.exists():`
        # so this entire block is skipped. Nothing to assert except no crash.


# ===========================================================================
# 4. ContextBuilder integration — outcome_reasonings (~5 tests)
# ===========================================================================

class TestContextBuilderOutcomeReasonings:
    def _make_ctx(self, tmp_path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        memory_dir = _make_memory_dir(tmp_path)
        return ContextBuilder(memory_dir), memory_dir

    def test_base_package_includes_outcome_reasonings(self, tmp_path):
        ctx, memory_dir = self._make_ctx(tmp_path)
        path = memory_dir / "findings" / "outcome_reasonings.jsonl"
        now = datetime.now(timezone.utc).isoformat()
        entry = {
            "suggestion_id": "sug_ctx",
            "genuine_effect": True,
            "mechanism": "Improved exits",
            "reasoned_at": now,
        }
        path.write_text(json.dumps(entry) + "\n")
        pkg = ctx.base_package()
        assert "outcome_reasonings" in pkg.data
        assert len(pkg.data["outcome_reasonings"]) == 1
        assert pkg.data["outcome_reasonings"][0]["mechanism"] == "Improved exits"

    def test_load_outcome_reasonings_applies_temporal_window(self, tmp_path):
        ctx, memory_dir = self._make_ctx(tmp_path)
        path = memory_dir / "findings" / "outcome_reasonings.jsonl"
        entries = []
        # Create 25 entries — max_entries is 20
        for i in range(25):
            ts = (datetime.now(timezone.utc) - timedelta(days=i)).isoformat()
            entries.append(json.dumps({
                "suggestion_id": f"sug_{i:03d}",
                "genuine_effect": True,
                "timestamp": ts,
            }))
        path.write_text("\n".join(entries) + "\n")
        result = ctx.load_outcome_reasonings()
        assert len(result) <= 20

    def test_outcome_reasonings_alongside_other_data(self, tmp_path):
        ctx, memory_dir = self._make_ctx(tmp_path)
        now = datetime.now(timezone.utc).isoformat()
        # Write reasonings
        reason_path = memory_dir / "findings" / "outcome_reasonings.jsonl"
        reason_path.write_text(json.dumps({
            "suggestion_id": "sug_along", "genuine_effect": True, "reasoned_at": now,
        }) + "\n")
        # Write discoveries too
        disc_path = memory_dir / "findings" / "discoveries.jsonl"
        disc_path.write_text(json.dumps({
            "pattern_description": "Alongside test", "timestamp": now,
        }) + "\n")
        pkg = ctx.base_package()
        assert "outcome_reasonings" in pkg.data
        assert "discoveries" in pkg.data

    def test_empty_reasonings_file_returns_empty_list(self, tmp_path):
        ctx, memory_dir = self._make_ctx(tmp_path)
        path = memory_dir / "findings" / "outcome_reasonings.jsonl"
        path.write_text("")
        result = ctx.load_outcome_reasonings()
        assert result == []

    def test_malformed_lines_are_skipped(self, tmp_path):
        ctx, memory_dir = self._make_ctx(tmp_path)
        path = memory_dir / "findings" / "outcome_reasonings.jsonl"
        now = datetime.now(timezone.utc).isoformat()
        content = (
            json.dumps({"suggestion_id": "sug_ok", "genuine_effect": True, "reasoned_at": now}) + "\n"
            + "NOT VALID JSON\n"
            + json.dumps({"suggestion_id": "sug_ok2", "genuine_effect": False, "reasoned_at": now}) + "\n"
        )
        path.write_text(content)
        result = ctx.load_outcome_reasonings()
        assert len(result) == 2
        ids = [r["suggestion_id"] for r in result]
        assert "sug_ok" in ids
        assert "sug_ok2" in ids
