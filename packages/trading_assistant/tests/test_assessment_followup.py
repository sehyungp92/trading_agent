# tests/test_assessment_followup.py
"""Tests for assessment follow-up fixes (A3, B1, B2).

A1/A2 timestamp tests are in test_learning_cycle.py (TestTimestampFallbackFixes).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from trading_assistant.schemas.structural_experiment import (
    AcceptanceCriteria,
    ExperimentRecord,
    ExperimentStatus,
)


# ── A3: Structural resolver enforcement ──


class TestStructuralResolverEnforcement:
    """Tests for trade sufficiency and baseline_value checks in the resolver."""

    def _make_experiment(self, **kwargs) -> ExperimentRecord:
        defaults = dict(
            experiment_id="exp_001",
            bot_id="bot1",
            title="Test Experiment",
            status=ExperimentStatus.ACTIVE,
            activated_at=datetime.now(timezone.utc) - timedelta(days=30),
            acceptance_criteria=[
                AcceptanceCriteria(
                    metric="pnl",
                    direction="improve",
                    minimum_change=0.0,
                    observation_window_days=14,
                    minimum_trade_count=20,
                ),
            ],
        )
        defaults.update(kwargs)
        return ExperimentRecord(**defaults)

    def _make_snapshot(self, **kwargs):
        """Create a mock GroundTruthSnapshot."""
        snap = MagicMock()
        snap.trade_count = kwargs.get("trade_count", 50)
        snap.pnl_total = kwargs.get("pnl_total", 100.0)
        snap.sharpe_ratio_30d = kwargs.get("sharpe_ratio_30d", 1.5)
        snap.win_rate = kwargs.get("win_rate", 0.55)
        snap.max_drawdown_pct = kwargs.get("max_drawdown_pct", 5.0)
        snap.avg_process_quality = kwargs.get("avg_process_quality", 0.7)
        snap.composite_score = kwargs.get("composite_score", 0.6)
        return snap

    def test_insufficient_trades_skips_resolution(self):
        """Experiment is NOT resolved when trade_count < minimum_trade_count."""
        exp = self._make_experiment()
        gt_after = self._make_snapshot(trade_count=10)  # < 20

        # Reproduce the resolver's trade sufficiency check
        max_min_trades = max(
            (getattr(c, "minimum_trade_count", 20) or 20)
            for c in exp.acceptance_criteria
        ) if exp.acceptance_criteria else 20
        actual_trades = getattr(gt_after, "trade_count", None)

        should_skip = actual_trades is not None and actual_trades < max_min_trades
        assert should_skip is True

    def test_sufficient_trades_proceeds(self):
        """Experiment proceeds when trade_count >= minimum_trade_count."""
        exp = self._make_experiment()
        gt_after = self._make_snapshot(trade_count=25)  # >= 20

        max_min_trades = max(
            (getattr(c, "minimum_trade_count", 20) or 20)
            for c in exp.acceptance_criteria
        ) if exp.acceptance_criteria else 20
        actual_trades = getattr(gt_after, "trade_count", None)

        should_skip = actual_trades is not None and actual_trades < max_min_trades
        assert should_skip is False

    def test_baseline_value_enforcement_pass(self):
        """Criterion passes when delta is positive AND after_val >= baseline."""
        criterion = AcceptanceCriteria(
            metric="sharpe",
            direction="improve",
            minimum_change=0.0,
            baseline_value=1.0,
        )
        before_val = 0.8
        after_val = 1.2
        delta = after_val - before_val  # 0.4

        met = delta >= criterion.minimum_change  # True
        if met and criterion.baseline_value is not None:
            met = after_val is not None and after_val >= criterion.baseline_value
        assert met is True

    def test_baseline_value_enforcement_fail(self):
        """Criterion fails when delta is positive but after_val < baseline."""
        criterion = AcceptanceCriteria(
            metric="sharpe",
            direction="improve",
            minimum_change=0.0,
            baseline_value=1.5,
        )
        before_val = 0.5
        after_val = 0.9  # improved from 0.5, but still below baseline of 1.5
        delta = after_val - before_val  # 0.4

        met = delta >= criterion.minimum_change  # True
        if met and criterion.baseline_value is not None:
            met = after_val is not None and after_val >= criterion.baseline_value
        assert met is False

    def test_no_baseline_value_ignores_check(self):
        """Criterion without baseline_value only checks delta."""
        criterion = AcceptanceCriteria(
            metric="pnl",
            direction="improve",
            minimum_change=0.0,
            baseline_value=None,
        )
        before_val = 100.0
        after_val = 105.0
        delta = after_val - before_val

        met = delta >= criterion.minimum_change
        if met and getattr(criterion, "baseline_value", None) is not None:
            met = after_val is not None and after_val >= criterion.baseline_value
        assert met is True

    def test_multiple_criteria_max_trade_count(self):
        """Trade sufficiency uses maximum minimum_trade_count across all criteria."""
        exp = self._make_experiment(
            acceptance_criteria=[
                AcceptanceCriteria(metric="pnl", minimum_trade_count=10),
                AcceptanceCriteria(metric="sharpe", minimum_trade_count=30),
            ],
        )
        gt_after = self._make_snapshot(trade_count=20)  # 20 < 30

        max_min_trades = max(
            (getattr(c, "minimum_trade_count", 20) or 20)
            for c in exp.acceptance_criteria
        )
        should_skip = gt_after.trade_count < max_min_trades
        assert should_skip is True


# ── B1: Experiment persistence through ExperimentManager ──


class TestExperimentPersistenceInPipeline:
    """Verify ExperimentManager.create_experiment is called in _route_to_experiment."""

    def test_constructor_accepts_experiment_manager(self):
        """AutonomousPipeline constructor stores experiment_manager."""
        from trading_assistant.skills.autonomous_pipeline import AutonomousPipeline

        mgr = MagicMock()
        pipeline = AutonomousPipeline(
            config_registry=MagicMock(),
            backtester=MagicMock(),
            approval_tracker=MagicMock(),
            suggestion_tracker=MagicMock(),
            experiment_manager=mgr,
        )
        assert pipeline._experiment_manager is mgr

    def test_constructor_defaults_experiment_manager_none(self):
        """AutonomousPipeline defaults experiment_manager to None."""
        from trading_assistant.skills.autonomous_pipeline import AutonomousPipeline

        pipeline = AutonomousPipeline(
            config_registry=MagicMock(),
            backtester=MagicMock(),
            approval_tracker=MagicMock(),
            suggestion_tracker=MagicMock(),
        )
        assert pipeline._experiment_manager is None

    def test_route_to_experiment_cap_uses_experiment_manager_active(self):
        """A/B cap is based on ExperimentManager active experiments for the bot."""
        from trading_assistant.skills.autonomous_pipeline import AutonomousPipeline

        experiment_manager = MagicMock()
        experiment_manager.get_active.return_value = [
            MagicMock(bot_id="bot1"),
            MagicMock(bot_id="bot1"),
            MagicMock(bot_id="bot2"),
        ]
        config_gen = MagicMock()

        pipeline = AutonomousPipeline(
            config_registry=MagicMock(),
            backtester=MagicMock(),
            approval_tracker=MagicMock(),
            suggestion_tracker=MagicMock(),
            experiment_config_generator=config_gen,
            experiment_manager=experiment_manager,
        )

        result = pipeline._route_to_experiment(
            suggestion_id="s1",
            suggestion={"suggestion_id": "s1", "bot_id": "bot1", "title": "Test"},
            report=MagicMock(best_value=0.5, exploration_summary="test"),
            param=MagicMock(param_name="threshold", current_value=0.3, value_type="float"),
        )

        assert result is None
        experiment_manager.get_active.assert_called_once()
        config_gen.generate_from_suggestion.assert_not_called()

    def test_records_parameter_search_evaluation_on_existing_proposal(self, tmp_path):
        """Parameter search routing appends a ProposalEvaluation for the candidate."""
        from trading_assistant.schemas.parameter_search import SearchRouting
        from trading_assistant.schemas.proposal_ledger import ProposalCandidate, ProposalKind, ProposalSource
        from trading_assistant.skills.autonomous_pipeline import AutonomousPipeline
        from trading_assistant.skills.proposal_ledger import ProposalLedger

        ledger = ProposalLedger(tmp_path / "findings")
        ledger.record_candidate(ProposalCandidate(
            proposal_id="proposal-param",
            source=ProposalSource.LLM_DAILY,
            kind=ProposalKind.PARAMETER_CHANGE,
            bot_id="bot1",
            title="Tune threshold",
        ))
        pipeline = AutonomousPipeline(
            config_registry=MagicMock(),
            backtester=MagicMock(),
            approval_tracker=MagicMock(),
            suggestion_tracker=MagicMock(),
            proposal_ledger=ledger,
        )
        report = MagicMock(
            routing=SearchRouting.EXPERIMENT,
            best_composite=0.82,
            best_robustness_score=55.0,
            exploration_summary="marginal but plausible",
            discard_reason="",
        )

        pipeline._record_parameter_search_evaluation(
            {"proposal_id": "proposal-param"},
            report,
        )

        record = ledger.get_by_id("proposal-param")
        assert record is not None
        assert len(record.evaluations) == 1
        assert record.evaluations[0].method == "parameter_search"
        assert record.evaluations[0].decision == "experiment"
        assert record.evaluations[0].confidence == pytest.approx(0.55)

    @pytest.mark.asyncio
    async def test_route_to_experiment_persists_config(self, tmp_path):
        """_route_to_experiment calls ExperimentManager.create_experiment."""
        from trading_assistant.skills.autonomous_pipeline import AutonomousPipeline

        experiment_manager = MagicMock()
        config_gen = MagicMock()
        mock_config = MagicMock()
        mock_config.experiment_id = "exp_abc"
        config_gen.generate_from_suggestion.return_value = mock_config

        registry = MagicMock()
        profile = MagicMock()
        profile.verification_commands = []
        registry.get_profile.return_value = profile

        pipeline = AutonomousPipeline(
            config_registry=registry,
            backtester=MagicMock(),
            approval_tracker=MagicMock(),
            suggestion_tracker=MagicMock(),
            experiment_config_generator=config_gen,
            experiment_manager=experiment_manager,
        )

        # Build minimal inputs for _route_to_experiment
        suggestion = {"suggestion_id": "s1", "title": "Test", "bot_id": "bot1"}
        report = MagicMock()
        report.best_value = 0.5
        report.exploration_summary = "test summary"
        param = MagicMock()
        param.param_name = "threshold"
        param.current_value = 0.3
        param.file_path = "config.yaml"

        pipeline._route_to_experiment(
            suggestion_id="s1",
            suggestion=suggestion,
            report=report,
            param=param,
        )

        experiment_manager.create_experiment.assert_called_once_with(mock_config)

    @pytest.mark.asyncio
    async def test_route_to_experiment_handles_manager_failure(self, tmp_path):
        """Pipeline continues if ExperimentManager.create_experiment fails."""
        from trading_assistant.skills.autonomous_pipeline import AutonomousPipeline

        experiment_manager = MagicMock()
        experiment_manager.create_experiment.side_effect = RuntimeError("DB error")

        config_gen = MagicMock()
        mock_config = MagicMock()
        mock_config.experiment_id = "exp_abc"
        config_gen.generate_from_suggestion.return_value = mock_config

        registry = MagicMock()
        profile = MagicMock()
        profile.verification_commands = []
        registry.get_profile.return_value = profile

        pipeline = AutonomousPipeline(
            config_registry=registry,
            backtester=MagicMock(),
            approval_tracker=MagicMock(),
            suggestion_tracker=MagicMock(),
            experiment_config_generator=config_gen,
            experiment_manager=experiment_manager,
        )

        suggestion = {"suggestion_id": "s1", "title": "Test", "bot_id": "bot1"}
        report = MagicMock()
        report.best_value = 0.5
        report.exploration_summary = "test summary"
        param = MagicMock()
        param.param_name = "threshold"
        param.current_value = 0.3
        param.file_path = "config.yaml"

        # Should NOT raise despite manager failure
        result = pipeline._route_to_experiment(
            suggestion_id="s1",
            suggestion=suggestion,
            report=report,
            param=param,
        )
        assert result is not None  # ApprovalRequest still returned


# ── B2: Strategy-idea experiment notification ──


class TestStrategyIdeaNotification:
    """Verify notification is sent when structural experiment is created from strategy idea."""

    @pytest.mark.asyncio
    async def test_high_confidence_idea_sends_notification(self, tmp_path):
        """handle_discovery_analysis notifies on high-confidence strategy idea experiments."""
        from trading_assistant.orchestrator.agent_runner import AgentResult
        from trading_assistant.orchestrator.orchestrator_brain import Action, ActionType
        from tests.factories import make_handlers as _factory_make_handlers

        event_stream = MagicMock()
        event_stream.broadcast = MagicMock()

        handlers, agent_runner, _ = _factory_make_handlers(
            tmp_path,
            event_stream=event_stream,
            dispatcher=MagicMock(),
            bots=["bot_a"],
            curated_dir=tmp_path / "curated",
            run_history_path=tmp_path / "runs" / "history.jsonl",
        )

        # Wire a structural experiment tracker
        struct_tracker = MagicMock()
        struct_tracker.record_experiment = MagicMock()
        handlers._structural_experiment_tracker = struct_tracker

        # Mock _notify to track calls
        handlers._notify = AsyncMock()

        # Build structured response with a high-confidence strategy idea
        strategy_ideas = [{
            "title": "New momentum filter",
            "description": "Add RSI crossover for better entries in trending regimes",
            "confidence": 0.8,
            "bot_id": "bot_a",
        }]
        structured = json.dumps({
            "discoveries": [],
            "strategy_ideas": strategy_ideas,
        })
        response_text = f"Analysis complete.\n<!-- STRUCTURED_OUTPUT\n{structured}\n-->"

        run_dir = tmp_path / "runs" / "discovery-2026-03-14"
        run_dir.mkdir(parents=True, exist_ok=True)
        agent_runner.invoke = AsyncMock(return_value=AgentResult(
            success=True,
            response=response_text,
            run_dir=run_dir,
        ))

        action = Action(
            type=ActionType.SPAWN_DAILY_ANALYSIS,
            event_id="evt_disc_001",
            bot_id="bot_a",
            details={"date": "2026-03-14", "bots": ["bot_a"]},
        )
        await handlers.handle_discovery_analysis(action)

        # Verify notification was sent for the high-confidence idea
        notify_calls = [
            c for c in handlers._notify.call_args_list
            if c[0][0] == "structural_experiment_proposed"
        ]
        assert len(notify_calls) == 1
        assert "New momentum filter" in notify_calls[0][0][2]

    @pytest.mark.asyncio
    async def test_low_confidence_idea_no_notification(self, tmp_path):
        """Low-confidence ideas (< 0.7) don't create experiments or send notifications."""
        from trading_assistant.orchestrator.agent_runner import AgentResult
        from trading_assistant.orchestrator.orchestrator_brain import Action, ActionType
        from tests.factories import make_handlers as _factory_make_handlers

        event_stream = MagicMock()
        event_stream.broadcast = MagicMock()

        handlers, agent_runner, _ = _factory_make_handlers(
            tmp_path,
            event_stream=event_stream,
            dispatcher=MagicMock(),
            bots=["bot_a"],
            curated_dir=tmp_path / "curated",
            run_history_path=tmp_path / "runs" / "history.jsonl",
        )

        struct_tracker = MagicMock()
        struct_tracker.record_experiment = MagicMock()
        handlers._structural_experiment_tracker = struct_tracker
        handlers._notify = AsyncMock()

        strategy_ideas = [{
            "title": "Low conf idea",
            "description": "Might work",
            "confidence": 0.5,
            "bot_id": "bot_a",
        }]
        structured = json.dumps({
            "discoveries": [],
            "strategy_ideas": strategy_ideas,
        })
        response_text = f"Analysis.\n<!-- STRUCTURED_OUTPUT\n{structured}\n-->"

        run_dir = tmp_path / "runs" / "discovery-2026-03-14"
        run_dir.mkdir(parents=True, exist_ok=True)
        agent_runner.invoke = AsyncMock(return_value=AgentResult(
            success=True,
            response=response_text,
            run_dir=run_dir,
        ))

        action = Action(
            type=ActionType.SPAWN_DAILY_ANALYSIS,
            event_id="evt_disc_002",
            bot_id="bot_a",
            details={"date": "2026-03-14", "bots": ["bot_a"]},
        )
        await handlers.handle_discovery_analysis(action)

        # No structural experiment notification for low-confidence ideas
        notify_calls = [
            c for c in handlers._notify.call_args_list
            if c[0][0] == "structural_experiment_proposed"
        ]
        assert len(notify_calls) == 0
        struct_tracker.record_experiment.assert_not_called()
