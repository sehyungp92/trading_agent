from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from trading_assistant.analysis.context_builder import ContextBuilder


def _make_memory_dir(tmp_path: Path) -> Path:
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "policies" / "v1").mkdir(parents=True)
    (mem / "findings").mkdir(parents=True)
    return mem


class TestLoadSimilarRuns:
    def test_prefers_query_search_before_recent_fallback(self, tmp_path):
        mem = _make_memory_dir(tmp_path)
        run_index = MagicMock()
        run_index.search.return_value = [{
            "run_id": "daily-2026-04-01",
            "date": "2026-04-01",
            "agent_type": "daily_analysis",
            "provider": "claude_max",
            "snippet": "Matched on regime shift",
        }]

        ctx = ContextBuilder(mem, run_index=run_index)
        results = ctx.load_similar_runs(agent_type="daily_analysis", bot_id="bot1")

        assert len(results) == 1
        assert results[0]["run_id"] == "daily-2026-04-01"
        assert "Matched on regime shift" in results[0]["snippet"]
        run_index.search.assert_called_once()
        run_index.get_recent_runs.assert_not_called()

    def test_falls_back_to_recent_runs_when_search_empty(self, tmp_path):
        mem = _make_memory_dir(tmp_path)
        run_index = MagicMock()
        run_index.search.return_value = []
        run_index.get_recent_runs.return_value = [{
            "run_id": "daily-2026-04-02",
            "date": "2026-04-02",
            "agent_type": "daily_analysis",
            "provider": "claude_max",
            "response_preview": "Recent fallback result",
        }]

        ctx = ContextBuilder(mem, run_index=run_index)
        results = ctx.load_similar_runs(agent_type="daily_analysis", bot_id="bot1")

        assert len(results) == 1
        assert results[0]["snippet"] == "Recent fallback result"
        run_index.get_recent_runs.assert_called_once_with(
            agent_type="daily_analysis", bot_id="bot1", limit=5, days=60,
        )

    def test_run_index_none_returns_empty(self, tmp_path):
        mem = _make_memory_dir(tmp_path)
        ctx = ContextBuilder(mem, run_index=None)
        assert ctx.load_similar_runs(agent_type="daily_analysis") == []

    def test_snippets_truncated_to_200_chars(self, tmp_path):
        mem = _make_memory_dir(tmp_path)
        run_index = MagicMock()
        run_index.search.return_value = [{
            "run_id": "r1",
            "date": "2026-04-01",
            "agent_type": "daily_analysis",
            "provider": "p",
            "snippet": "x" * 500,
        }]

        ctx = ContextBuilder(mem, run_index=run_index)
        results = ctx.load_similar_runs(agent_type="daily_analysis")
        assert len(results[0]["snippet"]) == 200

    def test_exception_returns_empty(self, tmp_path):
        mem = _make_memory_dir(tmp_path)
        run_index = MagicMock()
        run_index.search.side_effect = RuntimeError("DB error")

        ctx = ContextBuilder(mem, run_index=run_index)
        assert ctx.load_similar_runs(agent_type="daily_analysis") == []


class TestBasePackageIntegration:
    def test_similar_past_runs_in_data(self, tmp_path):
        mem = _make_memory_dir(tmp_path)
        run_index = MagicMock()
        run_index.search.return_value = [{
            "run_id": "daily-2026-04-01",
            "date": "2026-04-01",
            "agent_type": "daily_analysis",
            "provider": "claude_max",
            "snippet": "Regime shift detected.",
        }]

        ctx = ContextBuilder(mem, curated_dir=tmp_path / "curated", run_index=run_index)
        pkg = ctx.base_package(agent_type="daily_analysis")
        assert "similar_past_runs" in pkg.data
        assert len(pkg.data["similar_past_runs"]) == 1

    def test_no_run_index_no_key(self, tmp_path):
        mem = _make_memory_dir(tmp_path)
        ctx = ContextBuilder(mem, curated_dir=tmp_path / "curated")
        pkg = ctx.base_package(agent_type="daily_analysis")
        assert "similar_past_runs" not in pkg.data

    def test_exception_doesnt_crash_base_package(self, tmp_path):
        mem = _make_memory_dir(tmp_path)
        run_index = MagicMock()
        run_index.search.side_effect = Exception("boom")

        ctx = ContextBuilder(mem, curated_dir=tmp_path / "curated", run_index=run_index)
        assert ctx.base_package(agent_type="daily_analysis") is not None


class TestRetrievalProfileFiltering:
    def test_validation_patterns_can_be_filtered_by_bot(self, tmp_path):
        mem = _make_memory_dir(tmp_path)
        findings = mem / "findings"
        (findings / "validation_log.jsonl").write_text(
            "\n".join([
                '{"timestamp":"2026-04-24T00:00:00+00:00","blocked_details":[{"category":"parameter","reason":"duplicate idea","bot_id":"bot1"}]}',
                '{"timestamp":"2026-04-24T00:00:00+00:00","blocked_details":[{"category":"stop_loss","reason":"too risky","bot_id":"bot2"}]}',
            ]),
            encoding="utf-8",
        )

        ctx = ContextBuilder(mem)
        patterns = ctx.load_validation_patterns(bot_id="bot1")

        assert "parameter" in patterns
        assert "stop_loss" not in patterns
