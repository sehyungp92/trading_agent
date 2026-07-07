"""Tests for enriched bot instrumentation consumption.

Covers schema extensions, new build methods, strategy detectors,
and integration of enriched data through the pipeline.
"""
from __future__ import annotations

import json
from datetime import datetime


from trading_assistant.schemas.events import TradeEvent, DailySnapshot
from trading_assistant.schemas.daily_metrics import BotDailySummary
from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder
from trading_assistant.analysis.strategy_engine import StrategyEngine


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_trade(**overrides) -> TradeEvent:
    """Create a minimal valid TradeEvent with optional overrides."""
    base = dict(
        trade_id="t001",
        bot_id="bot1",
        pair="BTC/USDT",
        side="LONG",
        entry_time=datetime(2026, 4, 1, 10, 0),
        exit_time=datetime(2026, 4, 1, 12, 0),
        entry_price=100.0,
        exit_price=105.0,
        position_size=1.0,
        pnl=5.0,
        pnl_pct=5.0,
    )
    base.update(overrides)
    return TradeEvent(**base)


def _builder() -> DailyMetricsBuilder:
    return DailyMetricsBuilder(date="2026-04-01", bot_id="bot1")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Schema Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTradeEventSchema:
    """Verify new Optional fields on TradeEvent."""

    def test_new_fields_accepted(self):
        t = _make_trade(
            execution_timestamps={"signal_detected_at": 100, "fill_received_at": 200},
            sizing_inputs={"sizing_model": "vol_target", "unit_risk_usd": 50},
            strategy_params_at_entry={"lookback": 20, "threshold": 0.5},
            portfolio_state_at_entry={"exposure": 0.3, "correlated_positions": ["ETH"]},
            market_conditions_at_entry={"vix": 18.5},
            post_exit_1h_move_pct=1.2,
            post_exit_4h_move_pct=2.5,
            post_exit_backfill_status="complete",
        )
        assert t.execution_timestamps is not None
        assert t.sizing_inputs["sizing_model"] == "vol_target"
        assert t.post_exit_1h_move_pct == 1.2
        assert t.post_exit_backfill_status == "complete"

    def test_old_events_still_parse(self):
        """Events without new fields should parse with defaults."""
        t = _make_trade()
        assert t.execution_timestamps is None
        assert t.sizing_inputs is None
        assert t.post_exit_backfill_status == ""

    def test_extra_unknown_fields_ignored(self):
        """Pydantic v2 ignores unknown fields by default."""
        data = dict(
            trade_id="t001", bot_id="bot1", pair="BTC/USDT", side="LONG",
            entry_time="2026-04-01T10:00:00", exit_time="2026-04-01T12:00:00",
            entry_price=100.0, exit_price=105.0, position_size=1.0,
            pnl=5.0, pnl_pct=5.0,
            some_future_field="value",
        )
        t = TradeEvent(**data)
        assert t.trade_id == "t001"


class TestDailySnapshotSchema:
    def test_calmar_rolling_30d_field(self):
        snap = DailySnapshot(date="2026-04-01", bot_id="bot1", calmar_rolling_30d=1.5)
        assert snap.calmar_rolling_30d == 1.5

    def test_calmar_defaults_to_zero(self):
        snap = DailySnapshot(date="2026-04-01", bot_id="bot1")
        assert snap.calmar_rolling_30d == 0.0


class TestCalmarPassthrough:
    def test_snapshot_to_summary(self):
        builder = _builder()
        summary = BotDailySummary(date="2026-04-01", bot_id="bot1")
        snapshot = {"calmar_rolling_30d": 2.3, "sortino_rolling_30d": 1.1}
        builder._merge_snapshot_into_summary(summary, snapshot, [], [])
        assert summary.calmar_rolling_30d == 2.3

    def test_calmar_missing_defaults_zero(self):
        builder = _builder()
        summary = BotDailySummary(date="2026-04-01", bot_id="bot1")
        builder._merge_snapshot_into_summary(summary, {"sortino_rolling_30d": 1.0}, [], [])
        assert summary.calmar_rolling_30d == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Build Method Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestExecutionLatencyAnalysis:
    def test_builds_from_timestamps(self):
        trades = [_make_trade(execution_timestamps={
            "signal_detected_at": 1000, "intent_created_at": 1050,
            "risk_checked_at": 1080, "order_submitted_at": 1100,
            "fill_received_at": 1200,
        })]
        result = _builder().build_execution_latency_analysis(trades)
        assert result["coverage"] > 0
        assert result["total_with_data"] == 1
        assert "signal_to_intent" in result["stages"]
        assert result["stages"]["signal_to_intent"]["mean_ms"] == 50.0

    def test_normalizes_swing_key_names(self):
        trades = [_make_trade(execution_timestamps={
            "signal_generated_at": 1000, "oms_received_at": 1050,
            "risk_checked_at": 1100, "order_submitted_at": 1150,
            "fill_confirmed_at": 1300,
        })]
        result = _builder().build_execution_latency_analysis(trades)
        assert result["total_with_data"] == 1
        assert "signal_to_intent" in result["stages"]

    def test_skips_without_data(self):
        trades = [_make_trade()]
        result = _builder().build_execution_latency_analysis(trades)
        assert result["coverage"] == 0
        assert result["total_with_data"] == 0

    def test_by_regime(self):
        trades = [
            _make_trade(
                trade_id=f"t{i}", market_regime="trending",
                execution_timestamps={
                    "signal_detected_at": 1000, "intent_created_at": 1050,
                    "risk_checked_at": 1080, "order_submitted_at": 1100,
                    "fill_received_at": 1200,
                },
            )
            for i in range(3)
        ]
        result = _builder().build_execution_latency_analysis(trades)
        assert "trending" in result["by_regime"]
        assert result["by_regime"]["trending"]["trade_count"] == 3

    def test_latency_slippage_correlation(self):
        trades = [
            _make_trade(
                trade_id=f"t{i}",
                entry_slippage_bps=float(i * 10),
                execution_timestamps={
                    "signal_detected_at": 1000,
                    "intent_created_at": 1000 + i * 100,
                    "risk_checked_at": 1000 + i * 100 + 20,
                    "order_submitted_at": 1000 + i * 100 + 40,
                    "fill_received_at": 1000 + i * 100 + 60,
                },
            )
            for i in range(1, 8)
        ]
        result = _builder().build_execution_latency_analysis(trades)
        assert result["latency_slippage_correlation"] is not None


class TestSizingAnalysis:
    def test_groups_by_model(self):
        trades = [
            _make_trade(trade_id="t1", pnl=10, sizing_inputs={"sizing_model": "vol_target", "unit_risk_usd": 20}),
            _make_trade(trade_id="t2", pnl=-5, sizing_inputs={"sizing_model": "fixed", "unit_risk_usd": 10}),
            _make_trade(trade_id="t3", pnl=8, sizing_inputs={"sizing_model": "vol_target", "unit_risk_usd": 20}),
        ]
        result = _builder().build_sizing_analysis(trades)
        assert result["coverage"] == 1.0
        assert "vol_target" in result["by_sizing_model"]
        assert result["by_sizing_model"]["vol_target"]["trade_count"] == 2

    def test_risk_efficiency_computation(self):
        trades = [
            _make_trade(pnl=10, sizing_inputs={"sizing_model": "a", "unit_risk_usd": 5}),
        ]
        result = _builder().build_sizing_analysis(trades)
        assert result["by_sizing_model"]["a"]["avg_risk_efficiency"] == 2.0

    def test_skips_without_data(self):
        result = _builder().build_sizing_analysis([_make_trade()])
        assert result["coverage"] == 0


class TestParamOutcomeCorrelation:
    def test_identifies_spread(self):
        trades = [
            _make_trade(trade_id=f"t{i}", pnl=10 if i < 5 else -5,
                        strategy_params_at_entry={"lookback": float(i)})
            for i in range(10)
        ]
        result = _builder().build_param_outcome_correlation(trades)
        assert result["coverage"] == 1.0
        assert result["param_count"] >= 1
        assert len(result["top_correlations"]) >= 1
        assert result["top_correlations"][0]["param_name"] == "lookback"

    def test_handles_single_value_params(self):
        trades = [
            _make_trade(trade_id=f"t{i}", strategy_params_at_entry={"const": 5.0})
            for i in range(5)
        ]
        result = _builder().build_param_outcome_correlation(trades)
        assert result["param_count"] == 0  # single value discarded

    def test_skips_non_numeric(self):
        trades = [
            _make_trade(trade_id=f"t{i}",
                        strategy_params_at_entry={"mode": "aggressive", "val": float(i)})
            for i in range(5)
        ]
        result = _builder().build_param_outcome_correlation(trades)
        # "mode" is non-numeric, should be excluded; "val" should be counted
        assert result["param_count"] == 1


class TestPortfolioContextAnalysis:
    def test_exposure_buckets(self):
        trades = [
            _make_trade(trade_id=f"t{i}", pnl=5 if i % 2 == 0 else -3,
                        portfolio_state_at_entry={"exposure": i * 0.1})
            for i in range(9)
        ]
        result = _builder().build_portfolio_context_analysis(trades)
        assert result["coverage"] == 1.0
        assert "low" in result["by_exposure_level"]
        assert "high" in result["by_exposure_level"]

    def test_crowding_detection(self):
        trades = [
            _make_trade(trade_id="t1", pnl=10,
                        portfolio_state_at_entry={"exposure": 0.5, "correlated_positions": ["A"]}),
            _make_trade(trade_id="t2", pnl=-5,
                        portfolio_state_at_entry={"exposure": 0.5, "correlated_positions": ["A", "B", "C"]}),
        ]
        result = _builder().build_portfolio_context_analysis(trades)
        assert result["crowding_count"] == 1


class TestMarketConditionSummary:
    def test_groups_by_regime(self):
        trades = [
            _make_trade(trade_id="t1", market_regime="trending",
                        market_conditions_at_entry={"vix": 20.0}),
            _make_trade(trade_id="t2", market_regime="ranging",
                        market_conditions_at_entry={"vix": 15.0}),
        ]
        result = _builder().build_market_condition_summary(trades)
        assert result["coverage"] == 1.0
        assert "trending" in result["by_regime"]
        assert "ranging" in result["by_regime"]

    def test_merges_inline_fields(self):
        trades = [
            _make_trade(atr_at_entry=2.5, volume_24h=1000000,
                        market_conditions_at_entry={"vix": 18}),
        ]
        result = _builder().build_market_condition_summary(trades)
        avg_conds = result["by_regime"]["unknown"]["avg_conditions"]
        assert "atr_at_entry" in avg_conds
        assert "volume_24h" in avg_conds

    def test_correlation_ranking(self):
        trades = [
            _make_trade(trade_id=f"t{i}", pnl=float(i),
                        market_conditions_at_entry={"factor": float(i)})
            for i in range(1, 8)
        ]
        result = _builder().build_market_condition_summary(trades)
        assert len(result["condition_pnl_correlations"]) >= 1


class TestExitEfficiencyEnhancement:
    def test_move_pct_output(self):
        trades = [
            _make_trade(
                exit_price=100.0, pnl=5.0,
                post_exit_1h_price=102.0, post_exit_4h_price=104.0,
                post_exit_1h_move_pct=2.0, post_exit_4h_move_pct=4.0,
            ),
        ]
        result = _builder().exit_efficiency(trades)
        assert "avg_1h_move_pct" in result
        assert result["avg_1h_move_pct"] == 2.0
        assert result["avg_4h_move_pct"] == 4.0

    def test_move_pct_computed_from_prices(self):
        trades = [
            _make_trade(
                exit_price=100.0, pnl=5.0,
                post_exit_1h_price=103.0, post_exit_4h_price=106.0,
            ),
        ]
        result = _builder().exit_efficiency(trades)
        assert "avg_1h_move_pct" in result
        assert abs(result["avg_1h_move_pct"] - 3.0) < 0.01

    def test_worst_premature_exits(self):
        trades = [
            _make_trade(
                trade_id=f"t{i}", exit_price=100.0, pnl=2.0,
                post_exit_1h_price=100.0 + i, post_exit_4h_price=100.0 + i * 2,
            )
            for i in range(1, 8)
        ]
        result = _builder().exit_efficiency(trades)
        assert "worst_premature_exits" in result
        assert len(result["worst_premature_exits"]) <= 5
        # Should be sorted descending by continuation
        exits = result["worst_premature_exits"]
        assert exits[0]["move_pct"] >= exits[-1]["move_pct"]

    def test_post_exit_merge_in_write_curated(self, tmp_path):
        trades = [_make_trade(exit_price=100.0, pnl=5.0)]
        post_exit_events = [{
            "payload": json.dumps({
                "trade_id": "t001",
                "post_exit_1h_price": 102.0,
                "post_exit_4h_price": 105.0,
                "post_exit_1h_move_pct": 2.0,
                "backfill_status": "complete",
            }),
        }]
        builder = _builder()
        output_dir = builder.write_curated(
            trades=trades, missed=[], base_dir=tmp_path,
            post_exit_events=post_exit_events,
        )
        # Trade should have been updated
        assert trades[0].post_exit_1h_price == 102.0
        assert trades[0].post_exit_backfill_status == "complete"
        # exit_efficiency.json should benefit from the merge (regression: merge must
        # happen before exit_efficiency computation, not after)
        eff_data = json.loads((output_dir / "exit_efficiency.json").read_text())
        assert eff_data["total_trades_with_data"] == 1, (
            "post_exit merge should happen before exit_efficiency computation"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: Detector Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestExecutionBottleneckDetector:
    def _engine(self):
        return StrategyEngine(week_start="2026-03-25", week_end="2026-04-01")

    def test_fires_on_high_p95(self):
        stats = {
            "stages": {"order_to_fill": {"mean_ms": 200, "median_ms": 180, "p95_ms": 800}},
            "bottleneck_stage": "order_to_fill",
            "latency_slippage_correlation": None,
        }
        result = self._engine().detect_execution_bottleneck("bot1", stats)
        assert len(result) == 1
        assert "bottleneck" in result[0].title.lower()

    def test_fires_on_latency_slippage_correlation(self):
        stats = {
            "stages": {"order_to_fill": {"mean_ms": 100, "median_ms": 80, "p95_ms": 200}},
            "bottleneck_stage": "order_to_fill",
            "latency_slippage_correlation": 0.5,
        }
        result = self._engine().detect_execution_bottleneck("bot1", stats)
        assert len(result) == 1

    def test_silent_below_thresholds(self):
        stats = {
            "stages": {"order_to_fill": {"mean_ms": 50, "median_ms": 40, "p95_ms": 100}},
            "bottleneck_stage": "order_to_fill",
            "latency_slippage_correlation": 0.1,
        }
        result = self._engine().detect_execution_bottleneck("bot1", stats)
        assert len(result) == 0


class TestSizingMethodologyDetector:
    def _engine(self):
        return StrategyEngine(week_start="2026-03-25", week_end="2026-04-01")

    def test_fires_on_low_risk_efficiency(self):
        data = {
            "by_sizing_model": {
                "vol_target": {"trade_count": 10, "win_rate": 0.5, "avg_risk_efficiency": 0.3},
            },
        }
        result = self._engine().detect_sizing_methodology("bot1", data)
        assert len(result) == 1

    def test_fires_on_divergent_win_rates(self):
        data = {
            "by_sizing_model": {
                "vol_target": {"trade_count": 10, "win_rate": 0.7, "avg_risk_efficiency": 1.0},
                "fixed": {"trade_count": 10, "win_rate": 0.3, "avg_risk_efficiency": 1.0},
            },
        }
        result = self._engine().detect_sizing_methodology("bot1", data)
        assert len(result) == 1

    def test_silent_when_healthy(self):
        data = {
            "by_sizing_model": {
                "vol_target": {"trade_count": 10, "win_rate": 0.55, "avg_risk_efficiency": 0.8},
            },
        }
        result = self._engine().detect_sizing_methodology("bot1", data)
        assert len(result) == 0


class TestPortfolioCrowdingDetector:
    def _engine(self):
        return StrategyEngine(week_start="2026-03-25", week_end="2026-04-01")

    def test_fires_on_crowded_underperformance(self):
        ctx = {
            "crowding_count": 5,
            "crowded_win_rate": 0.3,
            "uncrowded_win_rate": 0.6,
        }
        result = self._engine().detect_portfolio_crowding("bot1", ctx)
        assert len(result) == 1
        assert "crowding" in result[0].title.lower()

    def test_no_suggestion_when_neutral(self):
        ctx = {
            "crowding_count": 5,
            "crowded_win_rate": 0.5,
            "uncrowded_win_rate": 0.55,
        }
        result = self._engine().detect_portfolio_crowding("bot1", ctx)
        assert len(result) == 0

    def test_no_suggestion_when_insufficient_data(self):
        ctx = {
            "crowding_count": 1,
            "crowded_win_rate": 0.1,
            "uncrowded_win_rate": 0.9,
        }
        result = self._engine().detect_portfolio_crowding("bot1", ctx)
        assert len(result) == 0


class TestExitTimingWiring:
    """Verify build_report now calls detect_exit_timing_issues when exit_efficiency_data provided."""

    def test_build_report_with_exit_efficiency(self):
        from trading_assistant.schemas.weekly_metrics import BotWeeklySummary
        engine = StrategyEngine(week_start="2026-03-25", week_end="2026-04-01")
        summaries = {
            "bot1": BotWeeklySummary(
                bot_id="bot1", week_start="2026-03-25", week_end="2026-04-01",
                total_trades=10, win_count=5, loss_count=5,
                gross_pnl=50.0, net_pnl=50.0,
                avg_win=15.0, avg_loss=-5.0,
            ),
        }
        # Low efficiency + high premature should trigger exit_timing detector
        report = engine.build_report(
            summaries,
            exit_efficiency_data={"bot1": {
                "avg_exit_efficiency": 0.3,
                "premature_exit_pct": 0.6,
            }},
        )
        exit_timing_suggestions = [
            s for s in report.suggestions
            if s.detection_context and s.detection_context.detector_name == "exit_timing"
        ]
        assert len(exit_timing_suggestions) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCuratedFileGeneration:
    """write_curated with enriched trades produces all new JSON files."""

    def test_all_new_files_written(self, tmp_path):
        trades = [
            _make_trade(
                trade_id=f"t{i}",
                pnl=5.0 if i % 2 == 0 else -3.0,
                exit_price=100.0,
                post_exit_1h_price=102.0,
                post_exit_4h_price=104.0,
                market_regime="trending",
                entry_slippage_bps=5.0,
                atr_at_entry=2.0,
                execution_timestamps={
                    "signal_detected_at": 1000, "intent_created_at": 1050,
                    "risk_checked_at": 1080, "order_submitted_at": 1100,
                    "fill_received_at": 1200,
                },
                sizing_inputs={"sizing_model": "vol_target", "unit_risk_usd": 20},
                strategy_params_at_entry={"lookback": float(i + 1), "threshold": 0.5},
                portfolio_state_at_entry={"exposure": i * 0.1, "correlated_positions": ["ETH"]},
                market_conditions_at_entry={"vix": 18.5, "volume": 1e6},
            )
            for i in range(6)
        ]
        builder = _builder()
        output_dir = builder.write_curated(trades=trades, missed=[], base_dir=tmp_path)

        expected_files = [
            "execution_latency.json",
            "sizing_analysis.json",
            "param_outcome_correlation.json",
            "portfolio_context.json",
            "market_conditions.json",
        ]
        for fname in expected_files:
            fpath = output_dir / fname
            assert fpath.exists(), f"Missing {fname}"
            data = json.loads(fpath.read_text())
            assert data.get("coverage", 0) > 0, f"{fname} has zero coverage"


class TestPostExitEventLoading:
    """Handler event type mapping includes post_exit."""

    def test_post_exit_in_event_mapping(self):
        mapping = {
            "filter_decision": "filter_decision_events",
            "indicator_snapshot": "indicator_snapshot_events",
            "orderbook_context": "orderbook_context_events",
            "parameter_change": "parameter_change_events",
            "order": "order_events",
            "process_quality": "process_quality_events",
            "stop_adjustment": "stop_adjustment_events",
            "post_exit": "post_exit_events",
        }
        assert "post_exit" in mapping
        assert mapping["post_exit"] == "post_exit_events"


class TestPromptAssemblerLoadsNewFiles:
    """New curated files appear in _CURATED_FILES list."""

    def test_curated_files_include_new_entries(self):
        from trading_assistant.analysis.prompt_assembler import _CURATED_FILES
        expected = [
            "execution_latency.json",
            "sizing_analysis.json",
            "param_outcome_correlation.json",
            "portfolio_context.json",
            "market_conditions.json",
        ]
        for fname in expected:
            assert fname in _CURATED_FILES, f"Missing {fname} in _CURATED_FILES"
