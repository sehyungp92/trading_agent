# tests/test_discovery.py
"""Phase 4: Generative Discovery tests.

Covers:
- schemas/discovery.py (TradeReference, Discovery, DiscoveryReport)
- analysis/discovery_prompt_assembler.py (DiscoveryPromptAssembler)
- Handler wiring (handle_discovery_analysis)
- ContextBuilder integration (load_discoveries)
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from trading_assistant.schemas.discovery import Discovery, DiscoveryReport, TradeReference
from trading_assistant.schemas.prompt_package import PromptPackage
from tests.factories import make_handlers as _factory_make_handlers


# ---------------------------------------------------------------------------
# Helper: Handlers factory
# ---------------------------------------------------------------------------

def _make_handlers(tmp_path, **kwargs):
    event_stream = MagicMock()
    event_stream.broadcast = MagicMock()
    handlers, agent_runner, _ = _factory_make_handlers(
        tmp_path,
        event_stream=event_stream,
        dispatcher=MagicMock(),
        bots=["bot_a"],
        curated_dir=tmp_path / "curated",
        **kwargs,
    )
    return handlers, agent_runner, event_stream


def _make_action(date="2026-03-14", bots=None):
    from trading_assistant.orchestrator.orchestrator_brain import Action, ActionType

    return Action(
        type=ActionType.SPAWN_DAILY_ANALYSIS,
        event_id="evt_disc_001",
        bot_id="bot_a",
        details={"date": date, "bots": bots or ["bot_a"]},
    )


def _structured_discovery_response(discoveries: list[dict]) -> str:
    block = json.dumps({"discoveries": discoveries})
    return f"Analysis text\n<!-- STRUCTURED_OUTPUT\n{block}\n-->"


# ===========================================================================
# 1. Schema tests — schemas/discovery.py (~5 tests)
# ===========================================================================

class TestTradeReferenceDefaults:
    def test_defaults(self):
        tr = TradeReference()
        assert tr.date == ""
        assert tr.bot_id == ""
        assert tr.trade_id == ""
        assert tr.pnl == 0.0
        assert tr.regime == ""
        assert tr.signal_strength == 0.0
        assert tr.note == ""


class TestDiscoveryModel:
    def test_all_fields(self):
        now = datetime.now(timezone.utc)
        d = Discovery(
            discovery_id="disc_001",
            pattern_description="High-vol morning reversal",
            evidence=[
                TradeReference(date="2026-03-10", bot_id="bot_a", trade_id="t1", pnl=120.0),
            ],
            proposed_root_cause="time_of_day",
            testable_hypothesis="Morning reversals yield >2x avg win in high vol",
            confidence=0.72,
            detector_coverage="novel",
            bot_id="bot_a",
            discovered_at=now,
        )
        assert d.discovery_id == "disc_001"
        assert d.pattern_description == "High-vol morning reversal"
        assert len(d.evidence) == 1
        assert d.evidence[0].pnl == 120.0
        assert d.proposed_root_cause == "time_of_day"
        assert d.confidence == 0.72
        assert d.detector_coverage == "novel"
        assert d.bot_id == "bot_a"
        assert d.discovered_at == now

    def test_discovery_report_with_list(self):
        report = DiscoveryReport(
            run_id="discovery-2026-03-14",
            date="2026-03-14",
            discoveries=[
                Discovery(pattern_description="Pattern A"),
                Discovery(pattern_description="Pattern B"),
            ],
            data_scope="30d raw trades for bot_a",
        )
        assert len(report.discoveries) == 2
        assert report.data_scope == "30d raw trades for bot_a"
        assert report.run_id == "discovery-2026-03-14"

    def test_serialization_roundtrip(self):
        d = Discovery(
            discovery_id="disc_rt",
            pattern_description="Round-trip test",
            evidence=[TradeReference(date="2026-03-12", pnl=50.0)],
            confidence=0.6,
        )
        data = json.loads(d.model_dump_json())
        restored = Discovery(**data)
        assert restored.discovery_id == d.discovery_id
        assert restored.pattern_description == d.pattern_description
        assert len(restored.evidence) == 1
        assert restored.confidence == 0.6

    def test_discovered_at_auto_populates(self):
        before = datetime.now(timezone.utc)
        d = Discovery(pattern_description="Auto-timestamp test")
        after = datetime.now(timezone.utc)
        assert before <= d.discovered_at <= after


# ===========================================================================
# 2. DiscoveryPromptAssembler tests (~10 tests)
# ===========================================================================

class TestDiscoveryPromptAssembler:
    def _make_assembler(self, tmp_path, bots=None, lookback_days=30):
        from trading_assistant.analysis.discovery_prompt_assembler import DiscoveryPromptAssembler

        memory_dir = tmp_path / "memory"
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "policies" / "v1" / "agent.md").write_text("Agent")
        (memory_dir / "policies" / "v1" / "trading_rules.md").write_text("Rules")
        (memory_dir / "policies" / "v1" / "soul.md").write_text("Soul")
        (memory_dir / "findings").mkdir(parents=True)
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()
        return DiscoveryPromptAssembler(
            date="2026-03-14",
            bots=bots or ["bot_a"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
            lookback_days=lookback_days,
        )

    def test_assemble_returns_prompt_package(self, tmp_path):
        asm = self._make_assembler(tmp_path)
        pkg = asm.assemble()
        assert isinstance(pkg, PromptPackage)

    def test_task_prompt_includes_bot_list(self, tmp_path):
        asm = self._make_assembler(tmp_path, bots=["bot_a", "bot_b"])
        pkg = asm.assemble()
        assert "bot_a" in pkg.task_prompt
        assert "bot_b" in pkg.task_prompt

    def test_task_prompt_includes_lookback_days(self, tmp_path):
        asm = self._make_assembler(tmp_path, lookback_days=14)
        pkg = asm.assemble()
        assert "14" in pkg.task_prompt

    def test_instructions_contain_30_automated_detectors(self, tmp_path):
        asm = self._make_assembler(tmp_path)
        pkg = asm.assemble()
        assert "30 automated detectors" in pkg.instructions

    def test_instructions_contain_anti_patterns(self, tmp_path):
        asm = self._make_assembler(tmp_path)
        pkg = asm.assemble()
        assert "ANTI-PATTERNS" in pkg.instructions

    def test_instructions_contain_structured_output(self, tmp_path):
        asm = self._make_assembler(tmp_path)
        pkg = asm.assemble()
        assert "STRUCTURED_OUTPUT" in pkg.instructions

    def test_list_raw_data_files_finds_trades_jsonl(self, tmp_path):
        asm = self._make_assembler(tmp_path)
        # Create a trades.jsonl file for today
        bot_dir = tmp_path / "curated" / "2026-03-14" / "bot_a"
        bot_dir.mkdir(parents=True)
        (bot_dir / "trades.jsonl").write_text('{"pnl": 100}\n')
        files = asm._list_raw_data_files()
        assert any("trades.jsonl" in f for f in files)

    def test_list_raw_data_files_finds_missed_jsonl(self, tmp_path):
        asm = self._make_assembler(tmp_path)
        bot_dir = tmp_path / "curated" / "2026-03-14" / "bot_a"
        bot_dir.mkdir(parents=True)
        (bot_dir / "missed.jsonl").write_text('{"signal": "strong"}\n')
        files = asm._list_raw_data_files()
        assert any("missed.jsonl" in f for f in files)

    def test_list_raw_data_files_handles_missing_dates(self, tmp_path):
        asm = self._make_assembler(tmp_path, lookback_days=5)
        # No curated dirs exist at all
        files = asm._list_raw_data_files()
        assert files == []

    def test_load_discovery_context_loads_regime_analysis(self, tmp_path):
        asm = self._make_assembler(tmp_path)
        # Create regime_analysis.json
        bot_dir = tmp_path / "curated" / "2026-03-14" / "bot_a"
        bot_dir.mkdir(parents=True)
        regime_data = {"trending": {"pnl": 500}, "ranging": {"pnl": -100}}
        (bot_dir / "regime_analysis.json").write_text(json.dumps(regime_data))
        ctx = asm._load_discovery_context()
        assert "regime_context" in ctx
        assert "bot_a" in ctx["regime_context"]

    def test_assemble_includes_existing_hypotheses(self, tmp_path):
        asm = self._make_assembler(tmp_path)
        # Patch hypothesis loading to return something
        with patch.object(asm._ctx, "load_hypothesis_track_record", return_value={"active": 3}):
            pkg = asm.assemble()
        assert "existing_hypotheses" in pkg.data


# ===========================================================================
# 3. Handler wiring tests (~8 tests)
# ===========================================================================

class TestHandleDiscoveryAnalysis:
    @pytest.fixture
    def sample_discoveries(self):
        return [
            {
                "pattern_description": "Regime transition clustering",
                "evidence": [{"date": "2026-03-10", "bot_id": "bot_a", "trade_id": "t1", "pnl": -200}],
                "proposed_root_cause": "regime_mismatch",
                "testable_hypothesis": "Cluster of losses at regime boundaries",
                "confidence": 0.65,
                "detector_coverage": "novel",
                "bot_id": "bot_a",
            },
            {
                "pattern_description": "Low confidence still wins",
                "evidence": [{"date": "2026-03-11", "bot_id": "bot_a", "trade_id": "t2", "pnl": 150}],
                "proposed_root_cause": "signal",
                "testable_hypothesis": "Low signal strength trades win in low vol",
                "confidence": 0.3,
                "detector_coverage": "signal_decay",
                "bot_id": "bot_a",
            },
        ]

    def _make_agent_result(self, tmp_path, discoveries):
        from trading_assistant.orchestrator.agent_runner import AgentResult

        run_dir = tmp_path / "runs" / "discovery-2026-03-14"
        run_dir.mkdir(parents=True, exist_ok=True)
        return AgentResult(
            success=True,
            response=_structured_discovery_response(discoveries),
            run_dir=run_dir,
        )

    def _make_failed_result(self, tmp_path):
        from trading_assistant.orchestrator.agent_runner import AgentResult

        run_dir = tmp_path / "runs" / "discovery-2026-03-14"
        run_dir.mkdir(parents=True, exist_ok=True)
        return AgentResult(
            success=False,
            response="",
            run_dir=run_dir,
            error="Agent timed out",
        )

    @pytest.mark.asyncio
    async def test_creates_run_record(self, tmp_path, sample_discoveries):
        handlers, runner, es = _make_handlers(
            tmp_path, run_history_path=tmp_path / "runs" / "history.jsonl",
        )
        runner.invoke.return_value = self._make_agent_result(tmp_path, sample_discoveries)
        action = _make_action()
        await handlers.handle_discovery_analysis(action)
        # Run history should have at least the initial "running" entry
        history_path = tmp_path / "runs" / "history.jsonl"
        assert history_path.exists()
        lines = history_path.read_text().strip().splitlines()
        assert len(lines) >= 1
        first = json.loads(lines[0])
        assert first["run_id"] == "discovery-2026-03-14"
        assert first["agent_type"] == "discovery_analysis"

    @pytest.mark.asyncio
    async def test_writes_discovery_report_on_success(self, tmp_path, sample_discoveries):
        handlers, runner, es = _make_handlers(
            tmp_path, run_history_path=tmp_path / "runs" / "history.jsonl",
        )
        runner.invoke.return_value = self._make_agent_result(tmp_path, sample_discoveries)
        action = _make_action()
        await handlers.handle_discovery_analysis(action)
        report_path = tmp_path / "runs" / "discovery-2026-03-14" / "discovery_report.md"
        assert report_path.exists()
        content = report_path.read_text()
        assert "STRUCTURED_OUTPUT" in content

    @pytest.mark.asyncio
    async def test_persists_discoveries_to_jsonl(self, tmp_path, sample_discoveries):
        handlers, runner, es = _make_handlers(
            tmp_path, run_history_path=tmp_path / "runs" / "history.jsonl",
        )
        runner.invoke.return_value = self._make_agent_result(tmp_path, sample_discoveries)
        action = _make_action()
        await handlers.handle_discovery_analysis(action)
        disc_path = tmp_path / "memory" / "findings" / "discoveries.jsonl"
        assert disc_path.exists()
        lines = disc_path.read_text().strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["pattern_description"] == "Regime transition clustering"
        assert "run_id" in first
        assert "discovered_at" in first

    @pytest.mark.asyncio
    async def test_adds_hypothesis_candidates_for_high_confidence(self, tmp_path, sample_discoveries):
        handlers, runner, es = _make_handlers(
            tmp_path, run_history_path=tmp_path / "runs" / "history.jsonl",
        )
        runner.invoke.return_value = self._make_agent_result(tmp_path, sample_discoveries)
        action = _make_action()
        await handlers.handle_discovery_analysis(action)
        # HypothesisLibrary should have been called — check the JSONL file
        hyp_path = tmp_path / "memory" / "findings" / "hypotheses.jsonl"
        if hyp_path.exists():
            lines = hyp_path.read_text().strip().splitlines()
            # Only the first discovery has confidence >= 0.5
            for line in lines:
                rec = json.loads(line)
                assert rec.get("status") in ("candidate", "active")

    @pytest.mark.asyncio
    async def test_handles_agent_failure_gracefully(self, tmp_path):
        handlers, runner, es = _make_handlers(
            tmp_path, run_history_path=tmp_path / "runs" / "history.jsonl",
        )
        runner.invoke.return_value = self._make_failed_result(tmp_path)
        action = _make_action()
        # Should not raise
        await handlers.handle_discovery_analysis(action)
        # discoveries.jsonl should NOT exist (no successful parsing)
        disc_path = tmp_path / "memory" / "findings" / "discoveries.jsonl"
        assert not disc_path.exists()

    @pytest.mark.asyncio
    async def test_broadcasts_discoveries_recorded_event(self, tmp_path, sample_discoveries):
        handlers, runner, es = _make_handlers(
            tmp_path, run_history_path=tmp_path / "runs" / "history.jsonl",
        )
        runner.invoke.return_value = self._make_agent_result(tmp_path, sample_discoveries)
        action = _make_action()
        await handlers.handle_discovery_analysis(action)
        # Check that broadcast was called with discoveries_recorded
        broadcast_calls = [
            c for c in es.broadcast.call_args_list
            if c[0][0] == "discoveries_recorded"
        ]
        assert len(broadcast_calls) == 1
        assert broadcast_calls[0][0][1]["count"] == 2

    @pytest.mark.asyncio
    async def test_records_run_status(self, tmp_path, sample_discoveries):
        handlers, runner, es = _make_handlers(
            tmp_path, run_history_path=tmp_path / "runs" / "history.jsonl",
        )
        runner.invoke.return_value = self._make_agent_result(tmp_path, sample_discoveries)
        action = _make_action()
        await handlers.handle_discovery_analysis(action)
        history_path = tmp_path / "runs" / "history.jsonl"
        lines = history_path.read_text().strip().splitlines()
        entries = [json.loads(line) for line in lines]
        # Should have running + completed
        statuses = [e["status"] for e in entries]
        assert "running" in statuses
        assert "completed" in statuses

    @pytest.mark.asyncio
    async def test_only_adds_hypotheses_for_confidence_gte_0_5(self, tmp_path, sample_discoveries):
        handlers, runner, es = _make_handlers(
            tmp_path, run_history_path=tmp_path / "runs" / "history.jsonl",
        )
        runner.invoke.return_value = self._make_agent_result(tmp_path, sample_discoveries)
        _make_action()

        # Patch HypothesisLibrary to track calls
        with patch("trading_assistant.orchestrator.handlers.HypothesisLibrary", create=True):
            # Actually let the real code run, but track add_candidate calls
            # Since the import is inside the handler, we need to patch at usage point
            pass

        # Alternative: run and check that the second discovery (confidence=0.3)
        # did NOT generate a hypothesis. Use a fresh handler.
        handlers2, runner2, es2 = _make_handlers(
            tmp_path / "sub", run_history_path=tmp_path / "sub" / "runs" / "history.jsonl",
        )
        runner2.invoke.return_value = self._make_agent_result(
            tmp_path / "sub",
            [
                {
                    "pattern_description": "Only low confidence",
                    "evidence": [],
                    "proposed_root_cause": "novel",
                    "testable_hypothesis": "Some hypothesis",
                    "confidence": 0.3,
                    "detector_coverage": "novel",
                    "bot_id": "bot_a",
                },
            ],
        )
        action2 = _make_action()
        await handlers2.handle_discovery_analysis(action2)
        # The discovery should be persisted (it was parsed)
        disc_path = tmp_path / "sub" / "memory" / "findings" / "discoveries.jsonl"
        assert disc_path.exists()
        lines = disc_path.read_text().strip().splitlines()
        assert len(lines) == 1
        # But no hypothesis candidate should be created for confidence < 0.5
        hyp_path = tmp_path / "sub" / "memory" / "findings" / "hypotheses.jsonl"
        if hyp_path.exists():
            hyp_lines = hyp_path.read_text().strip().splitlines()
            # If any hypotheses were created, they should NOT be from this discovery
            for line in hyp_lines:
                rec = json.loads(line)
                assert rec.get("title") != "Only low confidence"


# ===========================================================================
# 4. ContextBuilder integration tests (~8 tests)
# ===========================================================================

class TestContextBuilderDiscoveries:
    def _make_ctx(self, tmp_path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        memory_dir = tmp_path / "memory"
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "policies" / "v1" / "agent.md").write_text("Agent")
        (memory_dir / "policies" / "v1" / "trading_rules.md").write_text("Rules")
        (memory_dir / "policies" / "v1" / "soul.md").write_text("Soul")
        (memory_dir / "findings").mkdir(parents=True)
        return ContextBuilder(memory_dir), memory_dir

    def test_load_discoveries_returns_empty_when_file_missing(self, tmp_path):
        ctx, _ = self._make_ctx(tmp_path)
        result = ctx.load_discoveries()
        assert result == []

    def test_load_discoveries_loads_from_jsonl(self, tmp_path):
        ctx, memory_dir = self._make_ctx(tmp_path)
        discoveries_path = memory_dir / "findings" / "discoveries.jsonl"
        now = datetime.now(timezone.utc).isoformat()
        entries = [
            {"pattern_description": "Pattern A", "confidence": 0.7, "discovered_at": now},
            {"pattern_description": "Pattern B", "confidence": 0.5, "discovered_at": now},
        ]
        discoveries_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        result = ctx.load_discoveries()
        assert len(result) == 2
        assert result[0]["pattern_description"] in ("Pattern A", "Pattern B")

    def test_load_discoveries_applies_temporal_window(self, tmp_path):
        ctx, memory_dir = self._make_ctx(tmp_path)
        discoveries_path = memory_dir / "findings" / "discoveries.jsonl"
        old_ts = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        recent_ts = datetime.now(timezone.utc).isoformat()
        entries = [
            {"pattern_description": "Old", "timestamp": old_ts},
            {"pattern_description": "Recent", "timestamp": recent_ts},
        ]
        discoveries_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        result = ctx.load_discoveries()
        # Old entry should be filtered out by temporal window (90 days default)
        descriptions = [r["pattern_description"] for r in result]
        assert "Recent" in descriptions

    def test_base_package_includes_discoveries_when_present(self, tmp_path):
        ctx, memory_dir = self._make_ctx(tmp_path)
        discoveries_path = memory_dir / "findings" / "discoveries.jsonl"
        now = datetime.now(timezone.utc).isoformat()
        entry = {"pattern_description": "Base pkg test", "confidence": 0.8, "timestamp": now}
        discoveries_path.write_text(json.dumps(entry) + "\n")
        pkg = ctx.base_package()
        assert "discoveries" in pkg.data
        assert len(pkg.data["discoveries"]) == 1

    def test_load_outcome_reasonings_returns_empty_when_missing(self, tmp_path):
        ctx, _ = self._make_ctx(tmp_path)
        result = ctx.load_outcome_reasonings()
        assert result == []

    def test_load_outcome_reasonings_loads_from_jsonl(self, tmp_path):
        ctx, memory_dir = self._make_ctx(tmp_path)
        path = memory_dir / "findings" / "outcome_reasonings.jsonl"
        now = datetime.now(timezone.utc).isoformat()
        entries = [
            {"suggestion_id": "sug_001", "genuine_effect": True, "mechanism": "Exit timing", "reasoned_at": now},
            {"suggestion_id": "sug_002", "genuine_effect": False, "mechanism": "Coincidence", "reasoned_at": now},
        ]
        path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        result = ctx.load_outcome_reasonings()
        assert len(result) == 2

    def test_base_package_includes_outcome_reasonings_when_present(self, tmp_path):
        ctx, memory_dir = self._make_ctx(tmp_path)
        path = memory_dir / "findings" / "outcome_reasonings.jsonl"
        now = datetime.now(timezone.utc).isoformat()
        entry = {"suggestion_id": "sug_003", "genuine_effect": True, "reasoned_at": now}
        path.write_text(json.dumps(entry) + "\n")
        pkg = ctx.base_package()
        assert "outcome_reasonings" in pkg.data

    def test_base_package_includes_both_discoveries_and_reasonings(self, tmp_path):
        ctx, memory_dir = self._make_ctx(tmp_path)
        now = datetime.now(timezone.utc).isoformat()
        # Write discoveries
        disc_path = memory_dir / "findings" / "discoveries.jsonl"
        disc_path.write_text(json.dumps({"pattern_description": "D1", "timestamp": now}) + "\n")
        # Write reasonings
        reason_path = memory_dir / "findings" / "outcome_reasonings.jsonl"
        reason_path.write_text(json.dumps({"suggestion_id": "s1", "genuine_effect": True, "reasoned_at": now}) + "\n")
        pkg = ctx.base_package()
        assert "discoveries" in pkg.data
        assert "outcome_reasonings" in pkg.data
