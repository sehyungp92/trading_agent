"""Tests for RunIndex production wiring (Phase C)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock


from trading_assistant.orchestrator.run_index import RunIndex
from trading_assistant.schemas.prompt_package import PromptPackage


class TestRunIndexInCreateApp:
    def test_run_index_passed_to_agent_runner(self, tmp_path: Path):
        """create_app() should instantiate RunIndex and pass to AgentRunner."""
        import inspect
        from trading_assistant.orchestrator import app as app_module

        source = inspect.getsource(app_module.create_app)
        # Verify RunIndex is instantiated and passed to AgentRunner
        assert "RunIndex(" in source
        assert "run_index=run_index" in source


class TestIndexRunWithPromptPackage:
    def test_passes_bot_ids_and_date_from_prompt_package(self, tmp_path: Path):
        """_index_run should extract bot_ids and date from prompt_package.metadata."""
        from trading_assistant.orchestrator.agent_runner import AgentRunner, AgentResult

        mock_run_index = MagicMock()
        runner = AgentRunner(
            runs_dir=tmp_path / "runs",
            session_store=MagicMock(),
            run_index=mock_run_index,
        )

        pkg = PromptPackage(system_prompt="", metadata={"bot_ids": "bot_a,bot_b", "date": "2026-04-17"})
        result = AgentResult(response="ok", run_dir=tmp_path, duration_ms=100, session_id="s1")
        invocation = MagicMock()
        invocation.provider.value = "claude_max"
        invocation.effective_model = "sonnet"

        runner._index_run(
            run_id="r1",
            agent_type="daily_analysis",
            run_dir=tmp_path,
            result=result,
            invocation=invocation,
            prompt_package=pkg,
        )
        mock_run_index.index_run.assert_called_once()
        call_kwargs = mock_run_index.index_run.call_args.kwargs
        assert call_kwargs["bot_ids"] == "bot_a,bot_b"
        assert call_kwargs["date"] == "2026-04-17"

    def test_handles_none_prompt_package_gracefully(self, tmp_path: Path):
        """_index_run should work when prompt_package is None."""
        from trading_assistant.orchestrator.agent_runner import AgentRunner, AgentResult

        mock_run_index = MagicMock()
        runner = AgentRunner(
            runs_dir=tmp_path / "runs",
            session_store=MagicMock(),
            run_index=mock_run_index,
        )

        result = AgentResult(response="ok", run_dir=tmp_path, duration_ms=100, session_id="s1")
        invocation = MagicMock()
        invocation.provider.value = "claude_max"
        invocation.effective_model = "sonnet"

        runner._index_run(
            run_id="r2",
            agent_type="daily_analysis",
            run_dir=tmp_path,
            result=result,
            invocation=invocation,
            prompt_package=None,
        )
        call_kwargs = mock_run_index.index_run.call_args.kwargs
        assert call_kwargs["bot_ids"] == ""
        assert call_kwargs["date"] == ""

    def test_index_metadata_excludes_injected_prompt_blocks(self, tmp_path: Path):
        """RunIndex metadata should stay compact and omit injected text blobs."""
        from trading_assistant.orchestrator.agent_runner import AgentRunner, AgentResult

        mock_run_index = MagicMock()
        runner = AgentRunner(
            runs_dir=tmp_path / "runs",
            session_store=MagicMock(),
            run_index=mock_run_index,
        )

        pkg = PromptPackage(system_prompt="", metadata={
            "bot_ids": "bot_a",
            "date": "2026-04-17",
            "_learning_cards_text": "x" * 1000,
            "_generated_playbooks_text": "y" * 1000,
            "_retrieval_profile": {"tags": ["category:parameter"]},
        })
        result = AgentResult(response="ok", run_dir=tmp_path, duration_ms=100, session_id="s1")
        invocation = MagicMock()
        invocation.provider.value = "claude_max"
        invocation.effective_model = "sonnet"

        runner._index_run(
            run_id="r4",
            agent_type="daily_analysis",
            run_dir=tmp_path,
            result=result,
            invocation=invocation,
            prompt_package=pkg,
        )

        metadata = mock_run_index.index_run.call_args.kwargs["metadata"]
        assert "_learning_cards_text" not in metadata
        assert "_generated_playbooks_text" not in metadata
        assert metadata["_retrieval_profile"]["tags"] == ["category:parameter"]

    def test_index_run_errors_dont_propagate(self, tmp_path: Path):
        """_index_run should swallow exceptions (best-effort)."""
        from trading_assistant.orchestrator.agent_runner import AgentRunner, AgentResult

        mock_run_index = MagicMock()
        mock_run_index.index_run.side_effect = RuntimeError("DB error")
        runner = AgentRunner(
            runs_dir=tmp_path / "runs",
            session_store=MagicMock(),
            run_index=mock_run_index,
        )

        result = AgentResult(response="ok", run_dir=tmp_path, duration_ms=100, session_id="s1")
        invocation = MagicMock()
        invocation.provider.value = "claude_max"
        invocation.effective_model = "sonnet"

        # Should not raise
        runner._index_run(
            run_id="r3",
            agent_type="daily_analysis",
            run_dir=tmp_path,
            result=result,
            invocation=invocation,
            prompt_package=PromptPackage(system_prompt=""),
        )


class TestMetadataJson:
    def test_metadata_json_written_with_correct_keys(self, tmp_path: Path):
        """_write_run_files should create metadata.json with effective_model key."""
        from trading_assistant.orchestrator.agent_runner import AgentRunner

        runner = AgentRunner(
            runs_dir=tmp_path / "runs",
            session_store=MagicMock(),
        )
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        pkg = PromptPackage(
            system_prompt="sys",
            metadata={"bot_ids": "bot_x", "date": "2026-04-17"},
        )
        runner._write_run_files(run_dir, pkg, agent_type="monthly_validation")

        meta_path = run_dir / "metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["agent_type"] == "monthly_validation"
        assert meta["bot_ids"] == "bot_x"
        assert meta["date"] == "2026-04-17"
        assert "effective_model" in meta  # Key expected by reindex_from_directory
        assert "created_at" in meta

    def test_reindex_reads_written_metadata(self, tmp_path: Path):
        """reindex_from_directory should successfully read our metadata.json format."""
        from trading_assistant.orchestrator.agent_runner import AgentRunner

        runner = AgentRunner(
            runs_dir=tmp_path / "runs",
            session_store=MagicMock(),
        )
        run_dir = tmp_path / "runs" / "test-run-1"
        run_dir.mkdir(parents=True)
        pkg = PromptPackage(
            system_prompt="sys",
            metadata={"bot_ids": "bot_y", "date": "2026-04-17"},
        )
        runner._write_run_files(run_dir, pkg, agent_type="daily_analysis")

        # Create a real RunIndex and reindex
        ri = RunIndex(tmp_path / "test_index.db")
        ri.reindex_from_directory(tmp_path / "runs")
        results = ri.get_recent_runs(limit=10)
        ri.close()
        assert len(results) == 1
        assert results[0]["run_id"] == "test-run-1"
        assert results[0]["bot_ids"] == "bot_y"

    def test_metadata_json_backfill_updates_provider_and_model(self, tmp_path: Path):
        """Post-invocation backfill should populate provider/effective_model in metadata.json."""
        from trading_assistant.orchestrator.agent_runner import AgentRunner

        runner = AgentRunner(
            runs_dir=tmp_path / "runs",
            session_store=MagicMock(),
        )
        run_dir = tmp_path / "run_backfill"
        run_dir.mkdir()
        pkg = PromptPackage(
            system_prompt="sys",
            metadata={"bot_ids": "bot_z", "date": "2026-04-17"},
        )
        runner._write_run_files(run_dir, pkg, agent_type="monthly_validation")

        # Verify initially empty
        meta = json.loads((run_dir / "metadata.json").read_text())
        assert meta["provider"] == ""
        assert meta["effective_model"] == ""

        # Simulate post-invocation backfill (same logic as _invoke_with_selection_inner)
        meta["provider"] = "claude_max"
        meta["effective_model"] = "sonnet"
        (run_dir / "metadata.json").write_text(json.dumps(meta, default=str))

        # Verify reindex picks up updated values
        ri = RunIndex(tmp_path / "test_index.db")
        ri.reindex_from_directory(tmp_path)
        results = ri.get_recent_runs(limit=10)
        ri.close()
        assert len(results) == 1
        assert results[0]["provider"] == "claude_max"
        assert results[0]["model"] == "sonnet"

    def test_search_snippet_can_come_from_validator_notes(self, tmp_path: Path):
        """FTS snippets should surface hits from validator notes, not just response.md."""
        run_dir = tmp_path / "runs" / "daily-2026-04-17"
        run_dir.mkdir(parents=True)
        (run_dir / "validator_notes.md").write_text(
            "Regime mismatch blocked this suggestion.",
            encoding="utf-8",
        )
        (run_dir / "metadata.json").write_text(json.dumps({
            "provider": "claude_max",
            "effective_model": "sonnet",
            "bot_ids": "bot_y",
            "date": "2026-04-17",
            "success": True,
            "duration_ms": 10,
            "cost_usd": 0.0,
        }), encoding="utf-8")

        ri = RunIndex(tmp_path / "test_index.db")
        ri.index_run(
            run_id="daily-2026-04-17",
            agent_type="daily_analysis",
            run_dir=run_dir,
            provider="claude_max",
            model="sonnet",
            bot_ids="bot_y",
            date="2026-04-17",
        )
        results = ri.search("mismatch", agent_type="daily_analysis")
        ri.close()

        assert len(results) == 1
        assert "mismatch" in (results[0]["snippet"] or "").lower()


class TestRunIndexClose:
    def test_close_called_on_shutdown(self, tmp_path: Path):
        """run_index.close() should be in the lifespan shutdown path."""
        # Verify by reading the source code for the close call
        import inspect
        from trading_assistant.orchestrator import app as app_module

        source = inspect.getsource(app_module.create_app)
        assert "run_index.close()" in source


class TestAssemblerMetadata:
    def test_daily_assembler_sets_bot_ids_and_date(self, tmp_path: Path):
        from trading_assistant.analysis.prompt_assembler import DailyPromptAssembler

        memory_dir = tmp_path / "memory"
        policy_dir = memory_dir / "policies" / "v1"
        policy_dir.mkdir(parents=True)
        (policy_dir / "agent.md").write_text("agent")
        (policy_dir / "soul.md").write_text("soul")
        (policy_dir / "trading_rules.md").write_text("rules")
        (memory_dir / "findings").mkdir()
        curated = tmp_path / "curated"
        curated.mkdir()

        assembler = DailyPromptAssembler(
            date="2026-04-17", bots=["bot_a", "bot_b"],
            curated_dir=curated, memory_dir=memory_dir,
        )
        pkg = assembler.assemble()
        assert pkg.metadata["bot_ids"] == "bot_a,bot_b"
        assert pkg.metadata["date"] == "2026-04-17"
