"""Integration tests for ecosystem evaluation gap implementations."""


from trading_assistant.schemas.events import MissedOpportunityEvent
from tests.factories import make_trade


def _make_trade(trade_id, pnl, regime="trending", factors=None, post_1h=None, post_4h=None):
    return make_trade(
        trade_id=trade_id, pnl=pnl, market_regime=regime,
        spread_at_entry=3.0, signal_factors=factors,
        post_exit_1h_price=post_1h, post_exit_4h_price=post_4h,
        process_quality_score=80,
        root_causes=["normal_win" if pnl > 0 else "normal_loss"],
    )


class TestDataPipelineIntegration:
    """Verify all new curated files are written by DailyMetricsBuilder."""

    def test_write_curated_produces_all_files(self, tmp_path):
        from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder

        trades = [
            _make_trade("t1", 10.0, "trending",
                        factors=[{"factor_name": "rsi", "contribution": 0.6}],
                        post_1h=107.0),
            _make_trade("t2", -5.0, "ranging",
                        factors=[{"factor_name": "macd", "contribution": 0.4}],
                        post_1h=96.0, post_4h=94.0),
        ]
        missed = [
            MissedOpportunityEvent(
                bot_id="bot1", pair="BTC/USDT", signal="momentum",
                blocked_by="volume_filter", outcome_24h=20.0,
                confidence=0.8, assumption_tags=[],
            ),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, missed, tmp_path)

        expected = [
            "summary.json", "winners.json", "losers.json",
            "process_failures.json", "notable_missed.json",
            "regime_analysis.json", "filter_analysis.json",
            "root_cause_summary.json", "hourly_performance.json",
            "slippage_stats.json", "factor_attribution.json",
            "exit_efficiency.json", "regime_bps.json",
        ]
        for f in expected:
            assert (output_dir / f).exists(), f"Missing: {f}"


class TestQualityGateIntegration:
    """Verify quality gate works with new file list."""

    def test_passes_with_all_new_files(self, tmp_path):
        from trading_assistant.analysis.quality_gate import QualityGate

        bot_dir = tmp_path / "2026-03-01" / "bot1"
        bot_dir.mkdir(parents=True)
        for f in ["summary.json", "winners.json", "losers.json",
                   "process_failures.json", "notable_missed.json",
                   "regime_analysis.json", "filter_analysis.json",
                   "root_cause_summary.json", "hourly_performance.json",
                   "slippage_stats.json", "factor_attribution.json",
                   "exit_efficiency.json", "trades.jsonl", "missed.jsonl"]:
            (bot_dir / f).write_text("" if f.endswith(".jsonl") else "{}")
        (tmp_path / "2026-03-01" / "portfolio_risk_card.json").write_text("{}")

        gate = QualityGate("r1", "2026-03-01", ["bot1"], tmp_path)
        checklist = gate.run()
        assert checklist.can_proceed is True
        assert checklist.data_completeness == 1.0


class TestCounterfactualIntegration:
    """Verify counterfactual simulator works with strategy engine data."""

    def test_regime_gate_counterfactual_matches_engine_detection(self):
        from trading_assistant.analysis.strategy_engine import StrategyEngine
        from trading_assistant.skills.counterfactual_simulator import CounterfactualSimulator

        trades = [
            _make_trade("t1", 10.0, "trending"),
            _make_trade("t2", -15.0, "ranging"),
            _make_trade("t3", -12.0, "ranging"),
            _make_trade("t4", 8.0, "trending"),
        ]

        # Strategy engine detects regime mismatch
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        impact = engine.compute_regime_exclusion_impact("bot1", trades, "ranging")
        assert impact["delta_pnl"] > 0

        # Counterfactual simulator gives same result
        sim = CounterfactualSimulator()
        result = sim.simulate_regime_gate(trades, [], "ranging")
        assert result.delta_pnl == impact["delta_pnl"]


class TestBrainErrorTracking:
    """Verify error tracking prevents storm of triages."""

    def test_ten_errors_suppress_duplicates_escalate_storms(self):
        import json as jsonmod
        from trading_assistant.orchestrator.orchestrator_brain import OrchestratorBrain, ActionType

        brain = OrchestratorBrain()
        triage_count = 0
        suppressed_count = 0
        for i in range(10):
            actions = brain.decide({
                "event_type": "error",
                "event_id": f"e{i}",
                "bot_id": "bot1",
                "payload": jsonmod.dumps({"severity": "HIGH", "error_type": "ConnErr"}),
            })
            if actions[0].type == ActionType.SPAWN_TRIAGE:
                triage_count += 1
            elif actions[0].type == ActionType.QUEUE_FOR_DAILY:
                suppressed_count += 1

        # First error triggers triage, errors before storm threshold are suppressed,
        # storm threshold (>=3) onwards trigger triage escalations.
        # With 10 errors and storm_threshold=3:
        #   e0 -> first occurrence -> SPAWN_TRIAGE
        #   e1 -> already triaging, not yet storm -> suppressed (QUEUE_FOR_DAILY)
        #   e2..e9 -> storm detected -> SPAWN_TRIAGE (8 more)
        assert triage_count == 9
        assert suppressed_count == 1

    def test_not_all_errors_produce_triages(self):
        """Verify the tracker actually suppresses at least some duplicates."""
        import json as jsonmod
        from trading_assistant.orchestrator.orchestrator_brain import OrchestratorBrain, ActionType

        brain = OrchestratorBrain()
        triage_count = 0
        for i in range(10):
            actions = brain.decide({
                "event_type": "error",
                "event_id": f"e{i}",
                "bot_id": "bot1",
                "payload": jsonmod.dumps({"severity": "HIGH", "error_type": "ConnErr"}),
            })
            if actions[0].type == ActionType.SPAWN_TRIAGE:
                triage_count += 1

        # Not every error triggers a triage — some are suppressed
        assert triage_count < 10


class TestFilterSensitivityIntegration:
    """Verify filter sensitivity integrates with missed opportunity pipeline."""

    def test_filter_analysis_and_sensitivity_agree(self):
        from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder
        from trading_assistant.skills.filter_sensitivity_analyzer import FilterSensitivityAnalyzer

        missed = [
            MissedOpportunityEvent(
                bot_id="bot1", pair="BTC/USDT", signal="momentum",
                blocked_by="volume_filter", outcome_24h=500.0,
                confidence=0.8, assumption_tags=[],
            ),
            MissedOpportunityEvent(
                bot_id="bot1", pair="ETH/USDT", signal="momentum",
                blocked_by="volume_filter", outcome_24h=-200.0,
                confidence=0.7, assumption_tags=[],
            ),
        ]

        # DailyMetricsBuilder filter analysis
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        filter_analysis = builder.filter_analysis(missed)
        assert filter_analysis.filter_missed_pnl.get("volume_filter", 0) == 500.0

        # FilterSensitivityAnalyzer
        analyzer = FilterSensitivityAnalyzer(bot_id="bot1", date="2026-03-01")
        report = analyzer.analyze(missed)
        vol = next(c for c in report.curves if c.filter_name == "volume_filter")
        assert vol.current_net_impact == 300.0  # 500 - 200


class TestExitSimulationIntegration:
    """Verify exit simulation integrates with exit efficiency pipeline."""

    def test_exit_efficiency_and_simulator_consistent(self):
        from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder
        from trading_assistant.skills.exit_strategy_simulator import ExitStrategySimulator
        from trading_assistant.schemas.exit_simulation import ExitStrategyConfig, ExitStrategyType

        trades = [
            _make_trade("t1", 5.0, post_1h=110.0),  # left money on table
        ]

        # Exit efficiency detects premature exit
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        eff = builder.exit_efficiency(trades)
        assert eff["premature_exit_pct"] > 0

        # Exit simulator confirms time-based would be better
        sim = ExitStrategySimulator()
        config = ExitStrategyConfig(
            strategy_type=ExitStrategyType.TIME_BASED,
            params={"hold_hours": 1},
        )
        result = sim.simulate(trades, config)
        assert result.improvement > 0
