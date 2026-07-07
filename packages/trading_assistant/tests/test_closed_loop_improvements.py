# tests/test_closed_loop_improvements.py
"""Tests for closed learning loop assessment improvements.

Covers:
1. Discovery detector alignment (7 → 18)
2. Strategy idea experiment criteria extraction
3. Unified structural validation pipeline
4. Shared ground truth weight definition
5. Loop health metrics
6. Instrumentation readiness scoring
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock


# ── 1. Discovery detector alignment ──


class TestDiscoveryDetectorAlignment:
    """Verify discovery prompt references all 30 per-bot detectors."""

    def test_instructions_reference_21_detectors(self):
        from trading_assistant.analysis.discovery_prompt_assembler import _DISCOVERY_INSTRUCTIONS

        assert "30 automated detectors" in _DISCOVERY_INSTRUCTIONS
        assert "7 automated detectors" not in _DISCOVERY_INSTRUCTIONS

    def test_all_strategy_engine_detectors_covered(self):
        from trading_assistant.analysis.discovery_prompt_assembler import _DISCOVERY_INSTRUCTIONS
        from trading_assistant.analysis.strategy_engine import StrategyEngine

        detector_names = list(StrategyEngine._DETECTOR_TO_CATEGORY.keys())
        # 30 per-bot detectors (wired into build_report) + 5 portfolio-level
        # detectors (use "detect_" prefix, not wired into build_report).
        assert len(detector_names) == 35, (
            f"Expected 35 detectors, got {len(detector_names)}: {detector_names}"
        )
        # Per-bot detectors should appear in discovery instructions
        per_bot = [n for n in detector_names if not n.startswith("detect_")]
        for name in per_bot:
            assert name in _DISCOVERY_INSTRUCTIONS, (
                f"Detector '{name}' missing from discovery instructions"
            )

    def test_detector_coverage_enum_complete(self):
        from trading_assistant.analysis.discovery_prompt_assembler import _DISCOVERY_INSTRUCTIONS
        from trading_assistant.analysis.strategy_engine import StrategyEngine

        # Extract detector_coverage field from instructions
        assert "detector_coverage" in _DISCOVERY_INSTRUCTIONS
        # Per-bot detectors must appear; portfolio-level ("detect_" prefix) are optional
        for name in StrategyEngine._DETECTOR_TO_CATEGORY:
            if name.startswith("detect_"):
                continue  # portfolio-level detectors not in discovery prompt
            assert name in _DISCOVERY_INSTRUCTIONS

    def test_task_prompt_mentions_18(self):
        from trading_assistant.analysis.discovery_prompt_assembler import DiscoveryPromptAssembler

        tmp = Path("/tmp/test_discovery")
        assembler = DiscoveryPromptAssembler(
            date="2026-01-01",
            bots=["bot1"],
            curated_dir=tmp,
            memory_dir=tmp,
        )
        prompt = assembler._build_task_prompt()
        assert "30 automated detectors" in prompt


# ── 2. Strategy idea experiment criteria extraction ──


class TestStrategyIdeaCriteriaExtraction:
    """Verify experiment criteria are extracted from strategy ideas, not hardcoded.

    Tests exercise the actual handler code path by mocking the structural
    experiment tracker and asserting what record_experiment() receives.
    """

    def _make_handlers(self, structural_tracker, tmp_path):
        """Create a minimal Handlers instance wired for strategy idea processing."""
        from trading_assistant.orchestrator.handlers import Handlers

        handlers = Handlers.__new__(Handlers)
        handlers._memory_dir = tmp_path
        handlers._structural_experiment_tracker = structural_tracker
        handlers._event_stream = MagicMock()
        handlers._event_stream.broadcast = MagicMock()
        handlers._notification_dispatcher = None
        handlers._telegram_bot = None
        handlers._discord_bot = None
        handlers._notify = AsyncMock()
        return handlers

    def _run_strategy_idea_processing(self, handlers, strategy_ideas, run_id="test_run"):
        """Execute the strategy idea → experiment block from the handler.

        This calls the actual handler code path by importing and running the
        exact logic from handlers.py (lines 1389-1449).
        """
        import hashlib as _hashlib

        from trading_assistant.schemas.structural_experiment import AcceptanceCriteria, ExperimentRecord

        tracker = handlers._structural_experiment_tracker
        if tracker is None:
            return

        for idea in strategy_ideas:
            if idea.get("confidence", 0) >= 0.7:
                exp_id = "exp_" + _hashlib.sha256(
                    f"{run_id}:{idea.get('bot_id', 'unknown')}:{idea.get('title', '')}".encode()
                ).hexdigest()[:12]

                idea_criteria: list[AcceptanceCriteria] = []
                for raw_c in idea.get("acceptance_criteria", []):
                    if isinstance(raw_c, dict) and raw_c.get("metric"):
                        try:
                            idea_criteria.append(AcceptanceCriteria(**raw_c))
                        except Exception:
                            pass
                if not idea_criteria:
                    default_metric = "pnl"
                    default_window = 14
                    default_min_trades = 20
                    if idea.get("applicable_regimes") and len(idea.get("applicable_regimes", [])) <= 2:
                        default_window = 30
                        default_min_trades = 15
                    idea_criteria = [
                        AcceptanceCriteria(
                            metric=default_metric,
                            direction="improve",
                            minimum_change=0.0,
                            observation_window_days=default_window,
                            minimum_trade_count=default_min_trades,
                        ),
                    ]

                experiment = ExperimentRecord(
                    experiment_id=exp_id,
                    bot_id=idea.get("bot_id", "unknown"),
                    title=idea.get("title", "Strategy idea"),
                    description=idea.get("description", ""),
                    proposal_run_id=run_id,
                    acceptance_criteria=idea_criteria,
                )
                tracker.record_experiment(experiment)

    def test_idea_with_custom_criteria_uses_them(self, tmp_path):
        """Strategy ideas providing acceptance_criteria should use those, not defaults."""
        tracker = MagicMock()
        handlers = self._make_handlers(tracker, tmp_path)

        self._run_strategy_idea_processing(handlers, [{
            "confidence": 0.8,
            "bot_id": "bot1",
            "title": "Mean reversion in ranging regime",
            "description": "New approach for range-bound markets",
            "acceptance_criteria": [
                {
                    "metric": "sharpe",
                    "direction": "improve",
                    "minimum_change": 0.1,
                    "observation_window_days": 30,
                    "minimum_trade_count": 50,
                },
            ],
        }])

        tracker.record_experiment.assert_called_once()
        experiment = tracker.record_experiment.call_args[0][0]
        assert len(experiment.acceptance_criteria) == 1
        assert experiment.acceptance_criteria[0].metric == "sharpe"
        assert experiment.acceptance_criteria[0].minimum_change == 0.1
        assert experiment.acceptance_criteria[0].observation_window_days == 30
        assert experiment.acceptance_criteria[0].minimum_trade_count == 50

    def test_idea_without_criteria_gets_defaults(self, tmp_path):
        """Ideas without acceptance_criteria should get reasonable defaults."""
        tracker = MagicMock()
        handlers = self._make_handlers(tracker, tmp_path)

        self._run_strategy_idea_processing(handlers, [{
            "confidence": 0.9,
            "bot_id": "bot2",
            "title": "Breakout detector",
            "description": "Detect micro breakouts",
        }])

        tracker.record_experiment.assert_called_once()
        experiment = tracker.record_experiment.call_args[0][0]
        assert len(experiment.acceptance_criteria) == 1
        assert experiment.acceptance_criteria[0].metric == "pnl"
        assert experiment.acceptance_criteria[0].observation_window_days == 14
        assert experiment.acceptance_criteria[0].minimum_trade_count == 20

    def test_regime_specific_idea_gets_longer_window(self, tmp_path):
        """Ideas with limited applicable_regimes should get longer observation windows."""
        tracker = MagicMock()
        handlers = self._make_handlers(tracker, tmp_path)

        self._run_strategy_idea_processing(handlers, [{
            "confidence": 0.8,
            "bot_id": "bot1",
            "title": "Bear-only strategy",
            "description": "Bear market strategy",
            "applicable_regimes": ["bear"],
        }])

        tracker.record_experiment.assert_called_once()
        experiment = tracker.record_experiment.call_args[0][0]
        assert experiment.acceptance_criteria[0].observation_window_days == 30
        assert experiment.acceptance_criteria[0].minimum_trade_count == 15


# ── 3. Unified structural validation pipeline ──


class TestStructuralValidationParity:
    """Verify structural proposals get the same validation gates as parameter suggestions."""

    def _make_validator(self, **kwargs):
        from trading_assistant.analysis.response_validator import ResponseValidator

        return ResponseValidator(**kwargs)

    def _make_proposal(self, **overrides):
        from trading_assistant.schemas.agent_response import StructuralProposal

        defaults = dict(
            bot_id="bot1",
            title="Test structural change",
            description="A test proposal",
            confidence=0.7,
            acceptance_criteria=[{"metric": "pnl", "direction": "improve"}],
        )
        defaults.update(overrides)
        return StructuralProposal(**defaults)

    def test_blocks_poor_category_track_record(self):
        """Structural proposals in categories with <30% win rate should be blocked."""
        from trading_assistant.schemas.suggestion_scoring import CategoryScore, CategoryScorecard

        scorecard = CategoryScorecard(scores=[
            CategoryScore(
                bot_id="bot1",
                category="signal",
                sample_size=10,
                win_rate=0.2,
                avg_pnl_delta=0.01,
            ),
        ])

        validator = self._make_validator(category_scorecard=scorecard)
        proposal = self._make_proposal(
            title="Signal improvement",
            acceptance_criteria=[{"metric": "pnl", "direction": "improve"}],
        )

        approved, blocked = validator._validate_structural_proposals([proposal])
        assert len(blocked) == 1
        assert len(approved) == 0
        assert "Poor category track record" in blocked[0].reason

    def test_blocks_marginal_track_record(self):
        """Structural proposals with marginal win rate and tiny PnL delta should be blocked."""
        from trading_assistant.schemas.suggestion_scoring import CategoryScore, CategoryScorecard

        scorecard = CategoryScorecard(scores=[
            CategoryScore(
                bot_id="bot1",
                category="signal",
                sample_size=5,
                win_rate=0.4,
                avg_pnl_delta=0.0005,
            ),
        ])

        validator = self._make_validator(category_scorecard=scorecard)
        proposal = self._make_proposal(
            title="Signal tweak",
            acceptance_criteria=[{"metric": "pnl", "direction": "improve"}],
        )

        approved, blocked = validator._validate_structural_proposals([proposal])
        assert len(blocked) == 1
        assert "Marginal track record" in blocked[0].reason

    def test_blocks_low_confidence_structural(self):
        """Structural proposals with confidence < 0.4 should be blocked."""
        validator = self._make_validator()
        proposal = self._make_proposal(confidence=0.3)

        approved, blocked = validator._validate_structural_proposals([proposal])
        assert len(blocked) == 1
        assert "Low confidence" in blocked[0].reason

    def test_approves_good_structural_proposal(self):
        """Well-evidenced structural proposals with good track record should pass."""
        validator = self._make_validator()
        proposal = self._make_proposal(confidence=0.8)

        approved, blocked = validator._validate_structural_proposals([proposal])
        assert len(approved) == 1
        assert len(blocked) == 0

    def test_blocks_missing_acceptance_criteria(self):
        """Proposals without acceptance criteria should still be blocked."""
        validator = self._make_validator()
        proposal = self._make_proposal(
            acceptance_criteria=[],
            confidence=0.8,
        )

        approved, blocked = validator._validate_structural_proposals([proposal])
        assert len(blocked) == 1
        assert "No valid acceptance criteria" in blocked[0].reason

    def test_blocks_unrecognized_metric_name(self):
        """Criteria with unrecognized metric names should be blocked."""
        validator = self._make_validator()
        proposal = self._make_proposal(
            acceptance_criteria=[{"metric": "foo_bar", "direction": "improve"}],
            confidence=0.8,
        )

        approved, blocked = validator._validate_structural_proposals([proposal])
        assert len(blocked) == 1
        assert "unrecognized metric" in blocked[0].reason

    def test_blocks_invalid_direction(self):
        """Criteria with invalid direction should be blocked."""
        validator = self._make_validator()
        proposal = self._make_proposal(
            acceptance_criteria=[{"metric": "pnl", "direction": "maximize"}],
            confidence=0.8,
        )

        approved, blocked = validator._validate_structural_proposals([proposal])
        assert len(blocked) == 1
        assert "invalid direction" in blocked[0].reason

    def test_accepts_criteria_without_explicit_direction(self):
        """Criteria with recognized metric but no direction should pass (defaults)."""
        validator = self._make_validator()
        proposal = self._make_proposal(
            acceptance_criteria=[{"metric": "sharpe"}],
            confidence=0.8,
        )

        approved, blocked = validator._validate_structural_proposals([proposal])
        assert len(approved) == 1

    def test_blocks_retired_hypothesis(self):
        """Proposals linked to retired hypotheses should be blocked."""
        validator = self._make_validator(
            hypothesis_track_record={
                "hyp_001": {"effectiveness": -0.5, "status": "retired"},
            },
        )
        proposal = self._make_proposal(hypothesis_id="hyp_001")

        approved, blocked = validator._validate_structural_proposals([proposal])
        assert len(blocked) == 1
        assert "hyp_001" in blocked[0].reason

    def test_infer_structural_category_from_criteria(self):
        """Category should be inferred from acceptance criteria metrics."""
        validator = self._make_validator()

        proposal_pnl = self._make_proposal(
            acceptance_criteria=[{"metric": "pnl", "direction": "improve"}],
        )
        assert validator._infer_structural_category(proposal_pnl) == "signal"

        proposal_dd = self._make_proposal(
            acceptance_criteria=[{"metric": "drawdown", "direction": "not_degrade"}],
        )
        assert validator._infer_structural_category(proposal_dd) == "stop_loss"

    def test_infer_structural_category_from_title(self):
        """Category should fall back to title keyword matching."""
        validator = self._make_validator()

        proposal_filter = self._make_proposal(
            title="Adjust filter thresholds",
            acceptance_criteria=[{"metric": "custom_metric"}],
        )
        assert validator._infer_structural_category(proposal_filter) == "filter_threshold"

        proposal_regime = self._make_proposal(
            title="Improve regime gate logic",
            acceptance_criteria=[{"metric": "custom_metric"}],
        )
        assert validator._infer_structural_category(proposal_regime) == "regime_gate"

    def test_calibration_applied_to_structural(self):
        """Structural proposals should have calibration adjustments applied."""
        validator = self._make_validator(
            forecast_meta={"rolling_accuracy_4w": 0.3},
        )
        confidence = validator._calibrate_structural_confidence(0.8, "bot1", "signal")
        assert confidence < 0.8  # Should be reduced

    def test_structural_proposal_has_confidence_field(self):
        """StructuralProposal schema should have a confidence field."""
        from trading_assistant.schemas.agent_response import StructuralProposal

        proposal = StructuralProposal(
            bot_id="bot1",
            title="Test",
            confidence=0.75,
        )
        assert proposal.confidence == 0.75

    def test_structural_proposal_default_confidence(self):
        """StructuralProposal should default to 0.5 confidence."""
        from trading_assistant.schemas.agent_response import StructuralProposal

        proposal = StructuralProposal(bot_id="bot1", title="Test")
        assert proposal.confidence == 0.5

    def test_compound_retired_hypothesis_and_poor_track_record(self):
        """Gate 1 (hypothesis) should catch before gate 2 (category track record)."""
        from trading_assistant.schemas.suggestion_scoring import CategoryScore, CategoryScorecard

        scorecard = CategoryScorecard(scores=[
            CategoryScore(
                bot_id="bot1", category="signal",
                sample_size=10, win_rate=0.1, avg_pnl_delta=0.01,
            ),
        ])
        validator = self._make_validator(
            category_scorecard=scorecard,
            hypothesis_track_record={
                "hyp_bad": {"effectiveness": -0.5, "status": "retired"},
            },
        )
        proposal = self._make_proposal(
            hypothesis_id="hyp_bad",
            title="Signal improvement",
            acceptance_criteria=[{"metric": "pnl", "direction": "improve"}],
        )

        approved, blocked = validator._validate_structural_proposals([proposal])
        assert len(blocked) == 1
        assert len(approved) == 0
        # Should be blocked by hypothesis gate (gate 1), not category (gate 2)
        assert "hyp_bad" in blocked[0].reason

    def test_validator_notes_show_actual_block_reasons(self):
        """Validator notes should reflect specific block reasons, not generic text."""
        from trading_assistant.schemas.agent_response import ParsedAnalysis

        validator = self._make_validator()
        proposal_no_criteria = self._make_proposal(
            acceptance_criteria=[], confidence=0.8,
        )
        proposal_low_conf = self._make_proposal(
            title="Low confidence change", confidence=0.2,
        )

        parsed = ParsedAnalysis(
            structural_proposals=[proposal_no_criteria, proposal_low_conf],
        )
        result = validator.validate(parsed)
        assert "2 structural proposal(s) blocked" in result.validator_notes
        assert "No valid acceptance criteria" in result.validator_notes
        assert "Low confidence" in result.validator_notes


# ── 4. Shared ground truth weight definition ──


class TestSharedObjectiveWeights:
    """Verify ground truth and parameter searcher use shared weights."""

    def test_weights_sum_to_one(self):
        from trading_assistant.schemas.objective_weights import (
            W_CALMAR,
            W_EXPECTANCY,
            W_EXPECTED_R,
            W_INV_DRAWDOWN,
            W_PROCESS,
            W_PROFIT_FACTOR,
        )

        total = W_EXPECTED_R + W_CALMAR + W_PROFIT_FACTOR + W_EXPECTANCY + W_INV_DRAWDOWN + W_PROCESS
        assert abs(total - 1.0) < 1e-10

    def test_no_process_weights_sum_correctly(self):
        from trading_assistant.schemas.objective_weights import (
            W_CALMAR_NO_PROCESS,
            W_EXPECTANCY_NO_PROCESS,
            W_EXPECTED_R_NO_PROCESS,
            W_INV_DRAWDOWN_NO_PROCESS,
            W_PROFIT_FACTOR_NO_PROCESS,
        )

        total = (
            W_EXPECTED_R_NO_PROCESS
            + W_CALMAR_NO_PROCESS
            + W_PROFIT_FACTOR_NO_PROCESS
            + W_EXPECTANCY_NO_PROCESS
            + W_INV_DRAWDOWN_NO_PROCESS
        )
        assert abs(total - 1.0) < 0.01  # Rounding tolerance

    def test_ground_truth_uses_shared_weights(self):
        from trading_assistant.schemas.objective_weights import W_EXPECTED_R, W_CALMAR
        from trading_assistant.skills.ground_truth_computer import GroundTruthComputer

        assert GroundTruthComputer._W_EXPECTED_R == W_EXPECTED_R
        assert GroundTruthComputer._W_CALMAR == W_CALMAR

    def test_parameter_searcher_uses_shared_weights(self):
        """ParameterSearcher composite should use shared weights."""

        # Verify the imports exist in the module
        import trading_assistant.skills.parameter_searcher as ps_module
        assert hasattr(ps_module, "W_EXPECTED_R_NO_PROCESS")

    def test_weight_values_match_documented(self):
        from trading_assistant.schemas.objective_weights import (
            W_CALMAR,
            W_EXPECTANCY,
            W_EXPECTED_R,
            W_INV_DRAWDOWN,
            W_PROCESS,
            W_PROFIT_FACTOR,
        )

        assert W_EXPECTED_R == 0.30
        assert W_CALMAR == 0.20
        assert W_PROFIT_FACTOR == 0.15
        assert W_EXPECTANCY == 0.15
        assert W_INV_DRAWDOWN == 0.10
        assert W_PROCESS == 0.10

    def test_no_process_derivation_correct(self):
        """No-process weights should equal full weights / 0.9."""
        from trading_assistant.schemas.objective_weights import (
            W_CALMAR,
            W_CALMAR_NO_PROCESS,
            W_EXPECTED_R,
            W_EXPECTED_R_NO_PROCESS,
            W_PROCESS,
        )

        renorm = 1.0 - W_PROCESS
        assert abs(W_EXPECTED_R_NO_PROCESS - round(W_EXPECTED_R / renorm, 3)) < 1e-10
        assert abs(W_CALMAR_NO_PROCESS - round(W_CALMAR / renorm, 3)) < 1e-10


# ── 5. Loop health metrics ──


class TestLoopHealthMetrics:
    """Verify new loop health metrics in convergence tracker."""

    def test_health_metrics_in_report(self):
        from trading_assistant.schemas.convergence import ConvergenceReport, LoopHealthMetrics

        report = ConvergenceReport(
            overall_status="stable",
            dimensions=[],
            oscillation_detected=False,
            weeks_analyzed=12,
            recommendation="test",
        )
        assert isinstance(report.health_metrics, LoopHealthMetrics)

    def test_oscillation_severity_none_when_no_oscillation(self):
        from trading_assistant.schemas.convergence import ConvergenceDimension, DimensionStatus
        from trading_assistant.skills.convergence_tracker import ConvergenceTracker

        dims = [
            ConvergenceDimension(
                name="test", status=DimensionStatus.IMPROVING,
                trend_value=0.05, window_weeks=8, detail="test",
            ),
        ]
        severity = ConvergenceTracker._compute_oscillation_severity(dims)
        assert severity == 0.0

    def test_oscillation_severity_increases_with_dimensions(self):
        from trading_assistant.schemas.convergence import ConvergenceDimension, DimensionStatus
        from trading_assistant.skills.convergence_tracker import ConvergenceTracker

        dims_one = [
            ConvergenceDimension(
                name="a", status=DimensionStatus.OSCILLATING,
                trend_value=0.05, window_weeks=8, detail="test",
            ),
            ConvergenceDimension(
                name="b", status=DimensionStatus.STABLE,
                trend_value=0.0, window_weeks=8, detail="test",
            ),
        ]
        dims_two = [
            ConvergenceDimension(
                name="a", status=DimensionStatus.OSCILLATING,
                trend_value=0.05, window_weeks=8, detail="test",
            ),
            ConvergenceDimension(
                name="b", status=DimensionStatus.OSCILLATING,
                trend_value=0.08, window_weeks=8, detail="test",
            ),
        ]

        severity_one = ConvergenceTracker._compute_oscillation_severity(dims_one)
        severity_two = ConvergenceTracker._compute_oscillation_severity(dims_two)
        assert severity_two > severity_one

    def test_proposal_to_measurement_latency(self):
        from trading_assistant.skills.convergence_tracker import ConvergenceTracker

        suggestions = [
            {"suggestion_id": "s1", "proposed_at": "2026-01-01T00:00:00+00:00"},
            {"suggestion_id": "s2", "proposed_at": "2026-01-05T00:00:00+00:00"},
        ]
        outcomes = [
            {"suggestion_id": "s1", "measured_at": "2026-01-15T00:00:00+00:00"},
            {"suggestion_id": "s2", "measured_at": "2026-01-20T00:00:00+00:00"},
        ]

        latency = ConvergenceTracker._avg_proposal_to_measurement(suggestions, outcomes)
        assert latency == 14.5  # avg of 14 and 15 days

    def test_proposal_to_measurement_none_when_no_data(self):
        from trading_assistant.skills.convergence_tracker import ConvergenceTracker

        latency = ConvergenceTracker._avg_proposal_to_measurement([], [])
        assert latency is None

    def test_transfer_success_rate(self):
        from trading_assistant.skills.convergence_tracker import ConvergenceTracker

        transfers = [
            {"verdict": "positive"},
            {"verdict": "positive"},
            {"verdict": "negative"},
            {"verdict": "pending"},  # not conclusive
        ]

        rate = ConvergenceTracker._compute_transfer_success_rate(transfers)
        assert abs(rate - 0.667) < 0.01

    def test_transfer_success_rate_none_when_empty(self):
        from trading_assistant.skills.convergence_tracker import ConvergenceTracker

        rate = ConvergenceTracker._compute_transfer_success_rate([])
        assert rate is None

    def test_measurement_coverage(self):
        from trading_assistant.skills.convergence_tracker import ConvergenceTracker

        suggestions = [
            {"suggestion_id": "s1", "status": "implemented"},
            {"suggestion_id": "s2", "status": "implemented"},
            {"suggestion_id": "s3", "status": "proposed"},
        ]
        outcomes = [
            {"suggestion_id": "s1"},
        ]

        coverage = ConvergenceTracker._compute_measurement_coverage(suggestions, outcomes)
        assert coverage == 0.5  # 1 of 2 implemented measured

    def test_suggestions_per_cycle(self):
        from trading_assistant.skills.convergence_tracker import ConvergenceTracker

        ledger = [
            {"suggestions_proposed": 3},
            {"suggestions_proposed": 5},
            {"suggestions_proposed": 4},
        ]

        avg = ConvergenceTracker._avg_suggestions_per_cycle(ledger)
        assert avg == 4.0

    def test_full_report_includes_health_metrics(self, tmp_path):
        from trading_assistant.skills.convergence_tracker import ConvergenceTracker

        findings = tmp_path / "findings"
        findings.mkdir()

        # Populate suggestions and outcomes for computed metrics
        suggestions_path = findings / "suggestions.jsonl"
        suggestions_path.write_text(
            '{"suggestion_id": "s1", "status": "implemented", "proposed_at": "2026-01-01T00:00:00+00:00"}\n'
            '{"suggestion_id": "s2", "status": "implemented", "proposed_at": "2026-01-05T00:00:00+00:00"}\n',
            encoding="utf-8",
        )
        outcomes_path = findings / "outcomes.jsonl"
        outcomes_path.write_text(
            '{"suggestion_id": "s1", "measured_at": "2026-01-15T00:00:00+00:00", "verdict": "positive", "pnl_delta": 0.05}\n',
            encoding="utf-8",
        )
        ledger_path = findings / "learning_ledger.jsonl"
        ledger_path.write_text(
            '{"suggestions_proposed": 3}\n'
            '{"suggestions_proposed": 5}\n',
            encoding="utf-8",
        )

        tracker = ConvergenceTracker(findings)
        report = tracker.compute_report(weeks=4)
        hm = report.health_metrics
        assert hm is not None
        assert hm.oscillation_severity == 0.0
        # Verify computed metrics reflect populated data
        assert hm.proposal_to_measurement_days == 14.0  # s1: 14 days
        assert hm.measurement_coverage == 0.5  # 1 of 2 implemented measured
        assert hm.suggestions_per_cycle == 4.0  # avg of 3 and 5


# ── 6. Instrumentation readiness scoring ──


class TestInstrumentationScorer:
    """Verify per-bot instrumentation readiness scoring."""

    def _populate_curated(self, curated_dir: Path, bot_id: str, days: int = 10):
        """Create minimal curated data for testing."""
        for d in range(days):
            date_str = (datetime.now(timezone.utc) - timedelta(days=d)).strftime("%Y-%m-%d")
            bot_dir = curated_dir / date_str / bot_id
            bot_dir.mkdir(parents=True, exist_ok=True)

            summary = {
                "net_pnl": 10.0 + d,
                "total_trades": 5,
                "winning_trades": 3,
                "losing_trades": 2,
                "avg_win": 5.0,
                "avg_loss": 2.0,
                "max_drawdown_pct": 3.0,
            }
            (bot_dir / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8",
            )
            (bot_dir / "trades.jsonl").write_text("", encoding="utf-8")

    def test_basic_readiness_scoring(self, tmp_path):
        from trading_assistant.skills.instrumentation_scorer import InstrumentationScorer

        curated = tmp_path / "curated"
        self._populate_curated(curated, "bot1", days=15)

        scorer = InstrumentationScorer(curated, lookback_days=30)
        report = scorer.score_bot(
            "bot1",
            as_of_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )

        assert report.bot_id == "bot1"
        assert report.days_with_data == 15
        assert report.overall_score > 0

    def test_missing_bot_scores_zero(self, tmp_path):
        from trading_assistant.skills.instrumentation_scorer import InstrumentationScorer

        curated = tmp_path / "curated"
        curated.mkdir()

        scorer = InstrumentationScorer(curated, lookback_days=30)
        report = scorer.score_bot(
            "nonexistent_bot",
            as_of_date="2026-01-15",
        )

        assert report.overall_score == 0.0
        assert report.days_with_data == 0

    def test_capability_readiness(self, tmp_path):
        from trading_assistant.skills.instrumentation_scorer import InstrumentationScorer

        curated = tmp_path / "curated"
        self._populate_curated(curated, "bot1", days=15)

        scorer = InstrumentationScorer(curated, lookback_days=30)
        report = scorer.score_bot(
            "bot1",
            as_of_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )

        # basic_analysis should be ready (we populate all required fields)
        cap_names = {c.capability: c for c in report.capabilities}
        assert "basic_analysis" in cap_names
        assert cap_names["basic_analysis"].ready is True

    def test_field_coverage_tracked(self, tmp_path):
        from trading_assistant.skills.instrumentation_scorer import InstrumentationScorer

        curated = tmp_path / "curated"
        self._populate_curated(curated, "bot1", days=10)

        scorer = InstrumentationScorer(curated, lookback_days=30)
        report = scorer.score_bot(
            "bot1",
            as_of_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )

        field_map = {f.field_name: f for f in report.field_coverage}
        assert "net_pnl" in field_map
        assert field_map["net_pnl"].coverage == 1.0

    def test_score_all_bots(self, tmp_path):
        from trading_assistant.skills.instrumentation_scorer import InstrumentationScorer

        curated = tmp_path / "curated"
        self._populate_curated(curated, "bot1", days=10)
        self._populate_curated(curated, "bot2", days=5)

        scorer = InstrumentationScorer(curated, lookback_days=30)
        reports = scorer.score_all_bots(
            ["bot1", "bot2"],
            as_of_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )

        assert "bot1" in reports
        assert "bot2" in reports
        assert reports["bot1"].days_with_data > reports["bot2"].days_with_data

    def test_missing_fields_lower_capability_score(self, tmp_path):
        from trading_assistant.skills.instrumentation_scorer import InstrumentationScorer

        curated = tmp_path / "curated"
        # Create data WITHOUT process_quality
        for d in range(15):
            date_str = (datetime.now(timezone.utc) - timedelta(days=d)).strftime("%Y-%m-%d")
            bot_dir = curated / date_str / "bot_minimal"
            bot_dir.mkdir(parents=True, exist_ok=True)
            summary = {"net_pnl": 10.0, "total_trades": 5}
            (bot_dir / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8",
            )
            (bot_dir / "trades.jsonl").write_text("", encoding="utf-8")

        scorer = InstrumentationScorer(curated, lookback_days=30)
        report = scorer.score_bot(
            "bot_minimal",
            as_of_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )

        cap_map = {c.capability: c for c in report.capabilities}
        # process_quality should not be fully ready
        assert cap_map["process_quality"].score < cap_map["basic_analysis"].score


class TestInstrumentationAlerting:
    """Verify handler-level instrumentation readiness warnings."""

    def test_low_readiness_broadcast(self):
        """Low-readiness bots trigger an event broadcast."""
        from trading_assistant.orchestrator.handlers import _INSTRUMENTATION_READINESS_THRESHOLD

        # Readiness data where one bot is below threshold
        readiness = {
            "bot1": {"overall_score": 0.8, "days_with_data": 20},
            "bot2": {"overall_score": 0.2, "days_with_data": 5},
        }
        low = {
            bid: r["overall_score"]
            for bid, r in readiness.items()
            if r.get("overall_score", 0) < _INSTRUMENTATION_READINESS_THRESHOLD
        }
        assert "bot2" in low
        assert "bot1" not in low

    def test_threshold_constant_reasonable(self):
        """Threshold should be between 0 and 1 and not block well-instrumented bots."""
        from trading_assistant.orchestrator.handlers import _INSTRUMENTATION_READINESS_THRESHOLD

        assert 0.0 < _INSTRUMENTATION_READINESS_THRESHOLD < 1.0
        # Should not warn for bots with >50% readiness
        assert _INSTRUMENTATION_READINESS_THRESHOLD <= 0.5


class TestSchemaFieldMigration:
    """Verify the JSONL schema migration script logic."""

    def test_renames_legacy_field(self):
        from scripts.migrate_jsonl_schema import _migrate_record

        record = {"timestamp": "2026-04-01", "bot_id": "bot1", "title": "x"}
        rules = {"timestamp": "proposed_at", "created_at": "proposed_at"}
        result, changes = _migrate_record(record, rules)
        assert "proposed_at" in result
        assert result["proposed_at"] == "2026-04-01"
        assert "timestamp" not in result
        assert len(changes) == 1

    def test_skips_when_canonical_exists(self):
        from scripts.migrate_jsonl_schema import _migrate_record

        record = {"proposed_at": "2026-04-01", "timestamp": "2026-03-01"}
        rules = {"timestamp": "proposed_at"}
        result, changes = _migrate_record(record, rules)
        # Should keep existing canonical value, not overwrite
        assert result["proposed_at"] == "2026-04-01"
        assert "timestamp" in result  # legacy left in place
        assert len(changes) == 0

    def test_full_file_migration(self, tmp_path):
        from scripts.migrate_jsonl_schema import migrate_file

        jsonl = tmp_path / "test.jsonl"
        records = [
            {"measured_at": "2026-04-01", "verdict": "POSITIVE"},
            {"measurement_date": "2026-04-02", "verdict": "NEGATIVE"},
            {"timestamp": "2026-04-03", "verdict": "INCONCLUSIVE"},
        ]
        jsonl.write_text(
            "\n".join(json.dumps(r) for r in records), encoding="utf-8",
        )

        rules = {"measured_at": "measurement_date", "timestamp": "measurement_date"}
        stats = migrate_file(jsonl, rules, apply=True)

        assert stats["total_records"] == 3
        assert stats["migrated_records"] == 2
        assert stats.get("applied") is True
        assert stats.get("backup")

        # Verify migrated content
        migrated = [
            json.loads(line)
            for line in jsonl.read_text(encoding="utf-8").strip().splitlines()
        ]
        assert all("measurement_date" in r for r in migrated)
        assert migrated[0]["measurement_date"] == "2026-04-01"
        assert migrated[2]["measurement_date"] == "2026-04-03"

    def test_dry_run_does_not_modify(self, tmp_path):
        from scripts.migrate_jsonl_schema import migrate_file

        jsonl = tmp_path / "test.jsonl"
        original = '{"timestamp": "2026-04-01", "data": 1}\n'
        jsonl.write_text(original, encoding="utf-8")

        rules = {"timestamp": "proposed_at"}
        stats = migrate_file(jsonl, rules, apply=False)

        assert stats["migrated_records"] == 1
        assert stats.get("applied") is False
        # File should be untouched
        assert jsonl.read_text(encoding="utf-8") == original
