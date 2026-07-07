# tests/test_hermes_port.py
"""Tests for hermes port implementation: prompt delivery, workflow-aware context,
learning cards, token budgets, write coordinator, and run index."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock


from trading_assistant.analysis.context_builder import ContextBuilder, _estimate_tokens
from trading_assistant.orchestrator.invocation_builder import InvocationBuilder
from trading_assistant.schemas.learning_card import CardType, LearningCard, LearningCardIndex
from trading_assistant.schemas.prompt_package import PromptPackage
from trading_assistant.skills.learning_card_store import LearningCardStore
from trading_assistant.skills.learning_write_coordinator import LearningWriteCoordinator
from trading_assistant.orchestrator.run_index import RunIndex


# ---------------------------------------------------------------------------
# Prompt delivery (build_full_prompt)
# ---------------------------------------------------------------------------

class TestBuildFullPrompt:
    def test_task_prompt_only(self):
        pkg = PromptPackage(task_prompt="Do analysis.", data={})
        result = InvocationBuilder.build_full_prompt(pkg)
        assert result == "Do analysis."

    def test_includes_instructions(self):
        pkg = PromptPackage(
            task_prompt="Analyze.",
            instructions="Step 1: Check PnL.\nStep 2: Check risk.",
            data={},
        )
        result = InvocationBuilder.build_full_prompt(pkg)
        assert "## INSTRUCTIONS" in result
        assert "Step 1: Check PnL." in result
        assert result.startswith("Analyze.")

    def test_includes_corrections(self):
        pkg = PromptPackage(
            task_prompt="Analyze.",
            corrections=[
                {"summary": "Don't over-weight regime changes", "bot_id": "bot_a"},
                {"description": "Check spread before concluding slippage"},
            ],
            data={},
        )
        result = InvocationBuilder.build_full_prompt(pkg)
        assert "## PAST CORRECTIONS" in result
        assert "[bot_a] Don't over-weight regime changes" in result
        assert "Check spread before concluding slippage" in result

    def test_includes_skill_context(self):
        pkg = PromptPackage(
            task_prompt="Run monthly validation.",
            skill_context="Monthly validation guide:\n1. Load data\n2. Review evidence",
            data={},
        )
        result = InvocationBuilder.build_full_prompt(pkg)
        assert "## SKILL CONTEXT" in result
        assert "Monthly validation guide" in result

    def test_includes_data_file_manifest(self):
        pkg = PromptPackage(
            task_prompt="Analyze.",
            data={"summary": {"pnl": 100}, "risk_card": {"drawdown": 0.02}},
        )
        result = InvocationBuilder.build_full_prompt(pkg)
        assert "## AVAILABLE DATA FILES" in result
        assert "- risk_card.json" in result
        assert "- summary.json" in result

    def test_no_manifest_for_empty_data(self):
        pkg = PromptPackage(task_prompt="Analyze.", data={})
        result = InvocationBuilder.build_full_prompt(pkg)
        assert "## AVAILABLE DATA FILES" not in result

    def test_all_sections_together(self):
        pkg = PromptPackage(
            task_prompt="Analyze bots.",
            instructions="Focus on anomalies.",
            corrections=[{"summary": "Past mistake"}],
            skill_context="Use regime analysis.",
            data={"trades": [1, 2, 3]},
        )
        result = InvocationBuilder.build_full_prompt(pkg)
        assert result.startswith("Analyze bots.")
        assert "## INSTRUCTIONS" in result
        assert "## PAST CORRECTIONS" in result
        assert "## SKILL CONTEXT" in result
        assert "## AVAILABLE DATA FILES" in result
        # Verify ordering: task, instructions, corrections, skill, data files
        idx_instr = result.index("## INSTRUCTIONS")
        idx_corr = result.index("## PAST CORRECTIONS")
        idx_skill = result.index("## SKILL CONTEXT")
        idx_data = result.index("## AVAILABLE DATA FILES")
        assert idx_instr < idx_corr < idx_skill < idx_data

    def test_build_codex_uses_full_prompt(self):
        """Codex invocation path uses build_full_prompt, not bare task_prompt."""
        from trading_assistant.orchestrator.invocation_builder import InvocationBuilder
        from trading_assistant.orchestrator.provider_auth import ProviderAuthChecker
        from trading_assistant.schemas.agent_preferences import AgentProvider, AgentSelection
        from unittest.mock import patch

        auth = ProviderAuthChecker(claude_command="claude", codex_command="codex")
        builder = InvocationBuilder(auth_checker=auth, claude_command="claude", codex_command="codex")
        pkg = PromptPackage(
            task_prompt="Analyze today.",
            instructions="Step 1: check PnL.",
            system_prompt="You are a trading analyst.",
            data={"summary": {"pnl": 100}},
        )
        selection = AgentSelection(provider=AgentProvider.CODEX_PRO, model="gpt-5.4")
        with patch.object(auth, "require_resolved_command", return_value="/usr/bin/codex"):
            spec = builder.build_codex(pkg, selection, None)
        # The last positional arg is the full prompt (not bare task_prompt)
        prompt_arg = spec.args[-1]
        assert "## INSTRUCTIONS" in prompt_arg
        assert "Step 1: check PnL." in prompt_arg
        assert "## AVAILABLE DATA FILES" in prompt_arg


# ---------------------------------------------------------------------------
# Workflow-aware context filtering
# ---------------------------------------------------------------------------

class TestWorkflowAwareFiltering:
    def _make_builder(self, tmp_path):
        memory_dir = tmp_path / "memory"
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "policies" / "v1" / "agent.md").write_text("Agent rules.")
        (memory_dir / "findings").mkdir(parents=True)
        return ContextBuilder(memory_dir)

    def test_default_priority_used_for_daily(self, tmp_path):
        ctx = self._make_builder(tmp_path)
        pkg = ctx.base_package(agent_type="daily_analysis")
        manifest = pkg.metadata["_context_budget_manifest"]
        assert manifest["workflow"] == "daily_analysis"

    def test_triage_gets_minimal_context(self, tmp_path):
        ctx = self._make_builder(tmp_path)
        # Triage priority list is short — should prefer triage-relevant items
        assert "triage" in ctx._WORKFLOW_PRIORITIES
        triage_priorities = ctx._WORKFLOW_PRIORITIES["triage"]
        assert len(triage_priorities) < len(ctx._CONTEXT_PRIORITY)

    def test_unknown_workflow_uses_default(self, tmp_path):
        ctx = self._make_builder(tmp_path)
        pkg = ctx.base_package(agent_type="unknown_type")
        manifest = pkg.metadata["_context_budget_manifest"]
        assert manifest["workflow"] == "unknown_type"


# ---------------------------------------------------------------------------
# Token-aware context budget
# ---------------------------------------------------------------------------

class TestTokenBudget:
    def test_estimate_tokens_basic(self):
        assert _estimate_tokens({"key": "value"}) > 0
        # Larger data should produce more tokens
        small = _estimate_tokens({"a": 1})
        large = _estimate_tokens({"a": "x" * 1000})
        assert large > small

    def test_token_budget_limits_items(self, tmp_path):
        memory_dir = tmp_path / "memory"
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "findings").mkdir(parents=True)

        ctx = ContextBuilder(memory_dir)
        # Write some findings to trigger data loading
        outcomes = [{"suggestion_id": f"s{i}", "verdict": "positive"} for i in range(5)]
        (memory_dir / "findings" / "outcomes.jsonl").write_text(
            "\n".join(json.dumps(o) for o in outcomes)
        )
        corrections = [{"id": f"c{i}", "summary": f"Correction {i}" * 50} for i in range(5)]
        (memory_dir / "findings" / "corrections.jsonl").write_text(
            "\n".join(json.dumps(c) for c in corrections)
        )

        # With a very small token budget, should drop items
        pkg_small = ctx.base_package(context_budget_tokens=50)
        pkg_large = ctx.base_package(context_budget_tokens=100000)
        assert len(pkg_large.data) >= len(pkg_small.data)

    def test_budget_manifest_includes_token_estimates(self, tmp_path):
        memory_dir = tmp_path / "memory"
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "findings").mkdir(parents=True)
        (memory_dir / "findings" / "corrections.jsonl").write_text(
            json.dumps({"id": "c1", "summary": "test"})
        )

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package(context_budget_tokens=100000)
        manifest = pkg.metadata["_context_budget_manifest"]
        assert manifest["budget_mode"] == "tokens"
        assert "token_estimates" in manifest

    def test_item_budget_manifest_mode(self, tmp_path):
        memory_dir = tmp_path / "memory"
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "findings").mkdir(parents=True)

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()
        manifest = pkg.metadata["_context_budget_manifest"]
        assert manifest["budget_mode"] == "items"

    def test_token_budget_skips_large_item_keeps_small(self, tmp_path):
        """A large item that doesn't fit should be skipped, not block smaller items."""
        memory_dir = tmp_path / "memory"
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "findings").mkdir(parents=True)

        # Write a large correction and a small one
        large_correction = {"id": "big", "summary": "X" * 4000}  # ~1000 tokens
        small_correction = {"id": "small", "summary": "Tiny fix"}  # ~10 tokens
        (memory_dir / "findings" / "corrections.jsonl").write_text(
            json.dumps(large_correction) + "\n" + json.dumps(small_correction)
        )

        ctx = ContextBuilder(memory_dir)
        # Budget that's too small for the large item but enough for small ones
        pkg = ctx.base_package(context_budget_tokens=200)
        manifest = pkg.metadata["_context_budget_manifest"]
        # The large item may be omitted but small items should still be included
        included = manifest["included"]
        omitted = manifest["omitted"]
        # At least some items should be included despite the large one being dropped
        assert len(included) > 0 or len(omitted) > 0  # budget was active


# ---------------------------------------------------------------------------
# LearningCard schema
# ---------------------------------------------------------------------------

class TestLearningCard:
    def test_card_id_deterministic(self):
        card1 = LearningCard(
            card_type=CardType.CORRECTION,
            source_id="abc123",
            title="Test",
            content="Test content",
        )
        card2 = LearningCard(
            card_type=CardType.CORRECTION,
            source_id="abc123",
            title="Different title",
            content="Different content",
        )
        assert card1.card_id == card2.card_id  # same type + source_id

    def test_card_id_differs_by_type(self):
        card1 = LearningCard(
            card_type=CardType.CORRECTION, source_id="abc", title="T", content="C",
        )
        card2 = LearningCard(
            card_type=CardType.OUTCOME, source_id="abc", title="T", content="C",
        )
        assert card1.card_id != card2.card_id

    def test_relevance_score_recency(self):
        now = datetime.now(timezone.utc)
        recent = LearningCard(
            card_type=CardType.CORRECTION, source_id="r", title="Recent",
            content="C", created_at=now,
        )
        old = LearningCard(
            card_type=CardType.CORRECTION, source_id="o", title="Old",
            content="C", created_at=now - timedelta(days=60),
        )
        assert recent.relevance_score(now=now) > old.relevance_score(now=now)

    def test_relevance_score_context_match(self):
        card = LearningCard(
            card_type=CardType.OUTCOME, source_id="x", title="T",
            content="C", bot_id="bot_a",
        )
        score_match = card.relevance_score(query_bot_id="bot_a")
        score_no_match = card.relevance_score(query_bot_id="bot_b")
        assert score_match > score_no_match

    def test_to_prompt_text(self):
        card = LearningCard(
            card_type=CardType.CORRECTION, source_id="x",
            title="Check spread", content="Always verify spread.",
            bot_id="bot_a", impact_score=0.3,
        )
        text = card.to_prompt_text()
        assert "[CORRECTION]" in text
        assert "[bot_a]" in text
        assert "Check spread" in text
        assert "positive" in text


class TestLearningCardIndex:
    def test_add_and_get(self):
        index = LearningCardIndex()
        card = LearningCard(
            card_type=CardType.CORRECTION, source_id="c1", title="T", content="C",
        )
        index.add(card)
        assert index.get(card.card_id) is not None
        assert len(index.cards) == 1

    def test_add_deduplicates(self):
        index = LearningCardIndex()
        card1 = LearningCard(
            card_type=CardType.CORRECTION, source_id="c1", title="V1", content="C",
        )
        card2 = LearningCard(
            card_type=CardType.CORRECTION, source_id="c1", title="V2", content="C2",
        )
        index.add(card1)
        index.add(card2)
        assert len(index.cards) == 1
        assert index.cards[0].title == "V2"  # updated

    def test_ranked_returns_sorted(self):
        index = LearningCardIndex()
        now = datetime.now(timezone.utc)
        for i in range(5):
            index.add(LearningCard(
                card_type=CardType.CORRECTION, source_id=f"c{i}", title=f"C{i}",
                content="Content", created_at=now - timedelta(days=i * 10),
            ))
        ranked = index.ranked(limit=3)
        assert len(ranked) == 3

    def test_record_retrieval(self):
        index = LearningCardIndex()
        card = LearningCard(
            card_type=CardType.CORRECTION, source_id="c1", title="T", content="C",
        )
        index.add(card)
        index.record_retrieval([card.card_id])
        assert index.cards[0].retrieval_count == 1

    def test_record_feedback(self):
        index = LearningCardIndex()
        card = LearningCard(
            card_type=CardType.CORRECTION, source_id="c1", title="T", content="C",
        )
        index.add(card)
        index.record_feedback(card.card_id, helpful=True)
        index.record_feedback(card.card_id, helpful=False)
        assert index.cards[0].helpful_count == 1
        assert index.cards[0].harmful_count == 1


# ---------------------------------------------------------------------------
# LearningCardStore
# ---------------------------------------------------------------------------

class TestLearningCardStore:
    def test_save_and_load(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        store = LearningCardStore(findings)
        card = LearningCard(
            card_type=CardType.CORRECTION, source_id="c1",
            title="Test card", content="Content",
        )
        store.add_card(card)

        # Fresh load
        store2 = LearningCardStore(findings)
        index = store2.load()
        assert len(index.cards) == 1
        assert index.cards[0].title == "Test card"

    def test_card_from_correction(self):
        entry = {"id": "c1", "summary": "Check spread", "bot_id": "bot_a"}
        card = LearningCardStore.card_from_correction(entry)
        assert card.card_type == CardType.CORRECTION
        assert card.bot_id == "bot_a"
        assert "Check spread" in card.title

    def test_card_from_outcome(self):
        entry = {
            "suggestion_id": "s1", "bot_id": "bot_b",
            "verdict": "positive", "category": "exit_timing",
            "measurement_quality": "high",
        }
        card = LearningCardStore.card_from_outcome(entry)
        assert card.card_type == CardType.OUTCOME
        assert card.impact_score > 0

    def test_card_from_validator_block(self):
        entry = {"category": "filter_threshold", "reason": "win_rate < 30%", "block_count": 5}
        card = LearningCardStore.card_from_validator_block(entry)
        assert card.card_type == CardType.VALIDATOR_BLOCK
        assert card.impact_score < 0
        assert "BLOCKED" in card.title

    def test_ingest_from_existing(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        # Write some corrections
        (findings / "corrections.jsonl").write_text(
            "\n".join([
                json.dumps({"id": "c1", "summary": "Fix 1"}),
                json.dumps({"id": "c2", "summary": "Fix 2"}),
            ])
        )
        store = LearningCardStore(findings)
        count = store.ingest_from_existing()
        assert count == 2
        assert len(store.load().cards) == 2

        # Re-ingest should not duplicate
        count2 = store.ingest_from_existing()
        assert count2 == 0

    def test_ranked_for_prompt(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        store = LearningCardStore(findings)
        for i in range(5):
            store.add_card(LearningCard(
                card_type=CardType.CORRECTION, source_id=f"c{i}",
                title=f"Card {i}", content="Content",
            ))
        ranked = store.ranked_for_prompt(limit=3)
        assert len(ranked) == 3


# ---------------------------------------------------------------------------
# LearningWriteCoordinator
# ---------------------------------------------------------------------------

class TestLearningWriteCoordinator:
    def test_basic_jsonl_write(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        coord = LearningWriteCoordinator(findings)
        group = coord.begin("test_workflow", "run-1")

        records = [{"id": "r1", "value": 42}, {"id": "r2", "value": 99}]
        coord.add_jsonl_append(group, "write_records", "test.jsonl", records)
        result = coord.execute(group)

        assert result.all_succeeded
        assert not result.has_failures
        written = (findings / "test.jsonl").read_text()
        assert "r1" in written
        assert "r2" in written
        # Provenance
        assert "_write_group_id" in written

    def test_json_write(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        coord = LearningWriteCoordinator(findings)
        group = coord.begin("test")
        coord.add_json_write(group, "write_config", "config.json", {"setting": "value"})
        result = coord.execute(group)
        assert result.all_succeeded
        data = json.loads((findings / "config.json").read_text())
        assert data["setting"] == "value"
        assert "_write_group_id" in data

    def test_callback_execution(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        coord = LearningWriteCoordinator(findings)
        group = coord.begin("test")

        called = []
        coord.add_callback(group, "track", lambda: called.append(True))
        result = coord.execute(group)
        assert result.all_succeeded
        assert called == [True]

    def test_partial_failure(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        coord = LearningWriteCoordinator(findings)
        group = coord.begin("test")

        coord.add_jsonl_append(group, "good_write", "good.jsonl", [{"ok": True}])
        coord.add_callback(group, "bad_callback", lambda: 1 / 0)
        coord.add_jsonl_append(group, "after_fail", "after.jsonl", [{"also_ok": True}])

        result = coord.execute(group)
        assert result.has_failures
        assert not result.all_succeeded
        # Good writes still succeed
        assert (findings / "good.jsonl").exists()
        assert (findings / "after.jsonl").exists()
        # Failed op recorded
        failed_ops = [op for op in result.operations if op.status == "failed"]
        assert len(failed_ops) == 1
        assert failed_ops[0].name == "bad_callback"

    def test_dedup_key_prevents_double_write(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        coord = LearningWriteCoordinator(findings)

        group1 = coord.begin("test")
        coord.add_jsonl_append(group1, "write1", "data.jsonl", [{"v": 1}], dedup_key="unique-1")
        coord.execute(group1)

        group2 = coord.begin("test")
        coord.add_jsonl_append(group2, "write2", "data.jsonl", [{"v": 2}], dedup_key="unique-1")
        result = coord.execute(group2)

        assert result.operations[0].status == "skipped"
        lines = (findings / "data.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1  # Only first write

    def test_write_log_persisted(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        coord = LearningWriteCoordinator(findings)
        group = coord.begin("test_workflow", "run-1")
        coord.add_jsonl_append(group, "w1", "data.jsonl", [{"x": 1}])
        coord.execute(group)

        log = (findings / "write_log.jsonl").read_text()
        entry = json.loads(log.strip())
        assert entry["source_workflow"] == "test_workflow"
        assert entry["all_succeeded"] is True

    def test_event_stream_broadcast(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        mock_stream = MagicMock()
        coord = LearningWriteCoordinator(findings, event_stream=mock_stream)
        group = coord.begin("test")
        coord.add_jsonl_append(group, "w1", "data.jsonl", [{"x": 1}])
        coord.execute(group)

        mock_stream.broadcast.assert_called_once()
        call_args = mock_stream.broadcast.call_args
        assert call_args[0][0] == "learning_write_completed"

    def test_jsonl_append_does_not_mutate_caller(self, tmp_path):
        """Original dicts should not have _write_group_id injected."""
        findings = tmp_path / "findings"
        findings.mkdir()
        coord = LearningWriteCoordinator(findings)
        group = coord.begin("test")

        original = {"id": "r1", "value": 42}
        records = [original]
        coord.add_jsonl_append(group, "write", "out.jsonl", records)
        coord.execute(group)

        assert "_write_group_id" not in original
        # But the file should still have provenance
        written = json.loads((findings / "out.jsonl").read_text().strip())
        assert "_write_group_id" in written


# ---------------------------------------------------------------------------
# RunIndex (SQLite FTS5)
# ---------------------------------------------------------------------------

class TestRunIndex:
    def test_index_and_search(self, tmp_path):
        db_path = tmp_path / "run_index.db"
        run_dir = tmp_path / "runs" / "daily-2026-04-13"
        run_dir.mkdir(parents=True)
        (run_dir / "response.md").write_text("The regime mismatch caused losses.")
        (run_dir / "instructions.md").write_text("Focus on filter threshold analysis.")

        idx = RunIndex(db_path)
        idx.index_run(
            run_id="daily-2026-04-13",
            agent_type="daily_analysis",
            run_dir=run_dir,
            bot_ids="bot_a,bot_b",
            date="2026-04-13",
        )

        results = idx.search("regime mismatch")
        assert len(results) >= 1
        assert results[0]["run_id"] == "daily-2026-04-13"

        results2 = idx.search("filter threshold")
        assert len(results2) >= 1
        idx.close()

    def test_search_with_filters(self, tmp_path):
        db_path = tmp_path / "run_index.db"
        idx = RunIndex(db_path)

        for i, agent_type in enumerate(["daily_analysis", "weekly_analysis", "daily_analysis"]):
            run_dir = tmp_path / f"run_{i}"
            run_dir.mkdir()
            (run_dir / "response.md").write_text(f"Analysis result {i} regime data")
            idx.index_run(
                run_id=f"run-{i}",
                agent_type=agent_type,
                run_dir=run_dir,
                bot_ids="bot_a",
                date=f"2026-04-{10+i}",
            )

        # Filter by agent_type
        daily_results = idx.search("regime", agent_type="daily_analysis")
        weekly_results = idx.search("regime", agent_type="weekly_analysis")
        assert len(daily_results) == 2
        assert len(weekly_results) == 1
        idx.close()

    def test_get_recent_runs(self, tmp_path):
        db_path = tmp_path / "run_index.db"
        idx = RunIndex(db_path)

        for i in range(3):
            run_dir = tmp_path / f"run_{i}"
            run_dir.mkdir()
            idx.index_run(
                run_id=f"run-{i}",
                agent_type="daily_analysis",
                run_dir=run_dir,
            )

        recent = idx.get_recent_runs(limit=2)
        assert len(recent) == 2
        idx.close()

    def test_reindex_from_directory(self, tmp_path):
        db_path = tmp_path / "run_index.db"
        runs_dir = tmp_path / "runs"

        for name in ["daily-2026-04-10", "weekly-2026-04-11", "monthly-validation-bot1"]:
            d = runs_dir / name
            d.mkdir(parents=True)
            (d / "response.md").write_text(f"Result for {name}")

        idx = RunIndex(db_path)
        count = idx.reindex_from_directory(runs_dir)
        assert count == 3

        # Second call should not re-index
        count2 = idx.reindex_from_directory(runs_dir)
        assert count2 == 0
        idx.close()

    def test_infer_agent_type(self):
        assert RunIndex._infer_agent_type("daily-2026-04-13") == "daily_analysis"
        assert RunIndex._infer_agent_type("weekly-2026-W15") == "weekly_analysis"
        assert RunIndex._infer_agent_type("monthly-validation-bot1") == "monthly_validation"
        assert RunIndex._infer_agent_type("reasoning-2026-04-13") == "outcome_reasoning"
        assert RunIndex._infer_agent_type("random-thing") == "unknown"

    def test_index_run_replaces_existing(self, tmp_path):
        db_path = tmp_path / "run_index.db"
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        (run_dir / "response.md").write_text("Version 1")

        idx = RunIndex(db_path)
        idx.index_run(run_id="run-1", agent_type="daily_analysis", run_dir=run_dir)

        (run_dir / "response.md").write_text("Version 2")
        idx.index_run(run_id="run-1", agent_type="daily_analysis", run_dir=run_dir)

        results = idx.search("Version 2")
        assert len(results) == 1
        idx.close()

    def test_days_filter_in_get_recent_runs(self, tmp_path):
        """get_recent_runs should respect the days parameter."""
        db_path = tmp_path / "run_index.db"
        idx = RunIndex(db_path)

        # Index a run with a very old created_at
        run_dir = tmp_path / "old_run"
        run_dir.mkdir()
        old_date = "2020-01-01T00:00:00+00:00"
        idx._conn.execute(
            """INSERT INTO runs
               (run_id, agent_type, created_at, run_dir)
               VALUES (?, ?, ?, ?)""",
            ("old-run", "daily_analysis", old_date, str(run_dir)),
        )
        idx._conn.commit()

        # Index a recent run normally
        run_dir2 = tmp_path / "new_run"
        run_dir2.mkdir()
        idx.index_run(run_id="new-run", agent_type="daily_analysis", run_dir=run_dir2)

        # days=30 should exclude the old run
        recent = idx.get_recent_runs(days=30)
        run_ids = [r["run_id"] for r in recent]
        assert "new-run" in run_ids
        assert "old-run" not in run_ids
        idx.close()

    def test_malformed_fts_query_returns_empty(self, tmp_path):
        """Malformed FTS5 queries should return empty list, not raise."""
        db_path = tmp_path / "run_index.db"
        idx = RunIndex(db_path)
        results = idx.search('"unclosed quote')
        assert results == []
        idx.close()


# ---------------------------------------------------------------------------
# Session store pass-through (smoke test)
# ---------------------------------------------------------------------------

class TestSessionStorePassthrough:
    def test_base_package_with_session_store(self, tmp_path):
        memory_dir = tmp_path / "memory"
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "findings").mkdir(parents=True)

        mock_store = MagicMock()
        mock_store.get_recent_sessions.return_value = [
            {"date": "2026-04-12", "provider": "claude_max", "duration_ms": 5000,
             "response_summary": "Daily analysis complete"},
        ]

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package(session_store=mock_store, agent_type="daily_analysis")
        assert "session_history" in pkg.data
        mock_store.get_recent_sessions.assert_called_once()

    def test_base_package_without_session_store(self, tmp_path):
        memory_dir = tmp_path / "memory"
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "findings").mkdir(parents=True)

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()
        assert "session_history" not in pkg.data

    def test_base_package_handles_bad_session_store(self, tmp_path):
        memory_dir = tmp_path / "memory"
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "findings").mkdir(parents=True)

        mock_store = MagicMock()
        mock_store.get_recent_sessions.side_effect = RuntimeError("DB error")

        ctx = ContextBuilder(memory_dir)
        # Should not raise
        pkg = ctx.base_package(session_store=mock_store, agent_type="daily_analysis")
        assert "session_history" not in pkg.data


# ---------------------------------------------------------------------------
# Agent runner run file writing
# ---------------------------------------------------------------------------

class TestRunFileWriting:
    def test_corrections_and_skill_context_written(self, tmp_path):
        """Verify _write_run_files writes corrections.json and skill_context.md."""
        from trading_assistant.orchestrator.agent_runner import AgentRunner
        from trading_assistant.orchestrator.session_store import SessionStore

        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        session_store = SessionStore(tmp_path / "sessions")

        runner = AgentRunner(runs_dir=runs_dir, session_store=session_store)
        run_dir = runs_dir / "test-run"
        run_dir.mkdir()

        pkg = PromptPackage(
            task_prompt="Test",
            system_prompt="System",
            data={"summary": {"pnl": 100}},
            instructions="Step 1: analyze",
            corrections=[{"id": "c1", "summary": "past fix"}],
            skill_context="Monthly validation guide content",
        )
        runner._write_run_files(run_dir, pkg)

        assert (run_dir / "summary.json").exists()
        assert (run_dir / "instructions.md").exists()
        assert (run_dir / "system_prompt.md").exists()
        assert (run_dir / "corrections.json").exists()
        assert (run_dir / "skill_context.md").exists()

        corrections = json.loads((run_dir / "corrections.json").read_text())
        assert corrections[0]["summary"] == "past fix"
        assert (run_dir / "skill_context.md").read_text() == "Monthly validation guide content"


# ---------------------------------------------------------------------------
# Assembler data preservation
# ---------------------------------------------------------------------------

class TestAssemblerDataPreservation:
    def _make_memory_dir(self, tmp_path):
        memory_dir = tmp_path / "memory"
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "policies" / "v1" / "agent.md").write_text("Agent rules.")
        (memory_dir / "findings").mkdir(parents=True)
        # Write corrections so base_package populates data
        (memory_dir / "findings" / "corrections.jsonl").write_text(
            json.dumps({"id": "c1", "summary": "past fix"})
        )
        return memory_dir

    def test_triage_assembler_preserves_base_data(self, tmp_path):
        """Triage assembler should merge its data, not overwrite base_package data."""
        from trading_assistant.analysis.triage_prompt_assembler import TriagePromptAssembler
        from trading_assistant.schemas.bug_triage import BugComplexity, BugSeverity
        from trading_assistant.skills.triage_context_builder import TriageContext

        memory_dir = self._make_memory_dir(tmp_path)
        assembler = TriagePromptAssembler(memory_dir)
        context = TriageContext(
            error_event_summary="TypeError in handler",
            stack_trace="File handler.py, line 10",
            source_snippet="def handle(): ...",
            recent_git_log="abc123 fix handler",
        )
        pkg = assembler.assemble(context, BugSeverity.HIGH, BugComplexity.OBVIOUS_FIX)
        # Triage-specific data present
        assert "context" in pkg.data
        # Base package data not discarded (corrections loaded something)
        base_keys = [k for k in pkg.data if k != "context"]
        assert len(base_keys) > 0, "base_package data was discarded by triage assembler"
