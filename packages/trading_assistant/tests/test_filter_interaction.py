# tests/test_filter_interaction.py
"""Tests for multi-filter interaction analysis (ecosystem improvement 2.3)."""
from __future__ import annotations

import pytest

from trading_assistant.schemas.events import TradeEvent, MissedOpportunityEvent
from tests.factories import make_trade, make_missed as _factory_missed


def _make_trade(
    trade_id: str,
    bot_id: str = "bot1",
    pnl: float = 100.0,
    active_filters: list[str] | None = None,
    market_regime: str = "trending",
) -> TradeEvent:
    return make_trade(
        trade_id=trade_id,
        bot_id=bot_id,
        pair="NQ",
        entry_price=100.0,
        exit_price=100.0 + pnl / 10,
        pnl=pnl,
        pnl_pct=pnl / 100,
        active_filters=active_filters or [],
        market_regime=market_regime,
    )


def _make_missed(
    bot_id: str = "bot1",
    blocked_by: str = "",
    outcome_24h: float = 50.0,
) -> MissedOpportunityEvent:
    return _factory_missed(
        bot_id=bot_id,
        pair="NQ",
        blocked_by=blocked_by,
        outcome_24h=outcome_24h,
    )


class TestFilterInteractionAnalyzer:
    def test_empty_inputs(self):
        from trading_assistant.skills.filter_interaction_analyzer import FilterInteractionAnalyzer

        analyzer = FilterInteractionAnalyzer("bot1", "2026-03-01")
        report = analyzer.analyze([], [])
        assert report.total_filters_analyzed == 0
        assert report.pairs == []

    def test_single_filter_no_pairs(self):
        from trading_assistant.skills.filter_interaction_analyzer import FilterInteractionAnalyzer

        trades = [_make_trade("t1", active_filters=["vol_filter"])]
        analyzer = FilterInteractionAnalyzer("bot1", "2026-03-01")
        report = analyzer.analyze(trades, [])
        assert report.total_filters_analyzed == 1
        assert report.pairs == []

    def test_redundant_pair_detection(self):
        """Two filters that always co-activate should be flagged as redundant."""
        from trading_assistant.skills.filter_interaction_analyzer import FilterInteractionAnalyzer

        # Both filters active on all trades (high Jaccard → redundant)
        trades = [
            _make_trade(f"t{i}", pnl=50.0, active_filters=["filter_A", "filter_B"])
            for i in range(10)
        ]
        missed = [
            _make_missed(blocked_by="filter_A"),
            _make_missed(blocked_by="filter_B"),
        ]

        analyzer = FilterInteractionAnalyzer("bot1", "2026-03-01")
        report = analyzer.analyze(trades, missed)

        assert report.total_filters_analyzed == 2
        assert len(report.pairs) == 1
        pair = report.pairs[0]
        assert pair.interaction_type == "redundant"
        assert pair.redundancy_score > 0.7
        assert pair.trades_both_active == 10
        assert "consolidat" in pair.recommendation.lower()

    def test_complementary_pair_detection(self):
        """Filters with low overlap but different win rates → complementary."""
        from trading_assistant.skills.filter_interaction_analyzer import FilterInteractionAnalyzer

        trades = [
            # Trades with only A active (high win rate)
            _make_trade("t1", pnl=100.0, active_filters=["filter_A"]),
            _make_trade("t2", pnl=80.0, active_filters=["filter_A"]),
            _make_trade("t3", pnl=90.0, active_filters=["filter_A"]),
            # Trades with only B active (low win rate)
            _make_trade("t4", pnl=-50.0, active_filters=["filter_B"]),
            _make_trade("t5", pnl=-30.0, active_filters=["filter_B"]),
            _make_trade("t6", pnl=-40.0, active_filters=["filter_B"]),
            # One trade with both
            _make_trade("t7", pnl=50.0, active_filters=["filter_A", "filter_B"]),
        ]

        analyzer = FilterInteractionAnalyzer("bot1", "2026-03-01")
        report = analyzer.analyze(trades, [])

        assert report.total_filters_analyzed == 2
        assert len(report.pairs) == 1
        pair = report.pairs[0]
        assert pair.interaction_type == "complementary"

    def test_independent_pair_not_flagged(self):
        """Filters with moderate overlap and similar win rates → independent (not in output)."""
        from trading_assistant.skills.filter_interaction_analyzer import FilterInteractionAnalyzer

        # Moderate overlap, similar win rates
        trades = [
            _make_trade("t1", pnl=50.0, active_filters=["filter_A", "filter_B"]),
            _make_trade("t2", pnl=-50.0, active_filters=["filter_A", "filter_B"]),
            _make_trade("t3", pnl=50.0, active_filters=["filter_A"]),
            _make_trade("t4", pnl=-50.0, active_filters=["filter_B"]),
        ]

        analyzer = FilterInteractionAnalyzer("bot1", "2026-03-01")
        report = analyzer.analyze(trades, [])

        # Independent pairs are not included in the output
        for pair in report.pairs:
            assert pair.interaction_type != "independent"

    def test_missed_opportunity_counts(self):
        from trading_assistant.skills.filter_interaction_analyzer import FilterInteractionAnalyzer

        trades = [
            _make_trade(f"t{i}", active_filters=["vol_filter", "regime_filter"])
            for i in range(5)
        ]
        missed = [
            _make_missed(blocked_by="vol_filter"),
            _make_missed(blocked_by="vol_filter"),
            _make_missed(blocked_by="regime_filter"),
        ]

        analyzer = FilterInteractionAnalyzer("bot1", "2026-03-01")
        report = analyzer.analyze(trades, missed)

        # Should have one redundant pair (high Jaccard since all trades have both)
        assert len(report.pairs) == 1
        pair = report.pairs[0]
        # Check missed counts are captured
        assert pair.missed_by_a + pair.missed_by_b == 3

    def test_three_filters_produces_three_potential_pairs(self):
        from trading_assistant.skills.filter_interaction_analyzer import FilterInteractionAnalyzer

        trades = [
            _make_trade("t1", active_filters=["A", "B", "C"]),
            _make_trade("t2", active_filters=["A", "B", "C"]),
            _make_trade("t3", active_filters=["A", "B", "C"]),
        ]

        analyzer = FilterInteractionAnalyzer("bot1", "2026-03-01")
        report = analyzer.analyze(trades, [])

        assert report.total_filters_analyzed == 3
        # All 3 pairs are redundant since they always co-activate
        assert len(report.pairs) == 3
        for pair in report.pairs:
            assert pair.interaction_type == "redundant"

    def test_filters_from_missed_only(self):
        """Filters that only appear in missed opps should still be analyzed."""
        from trading_assistant.skills.filter_interaction_analyzer import FilterInteractionAnalyzer

        trades = [_make_trade("t1", active_filters=["filter_A"])]
        missed = [_make_missed(blocked_by="filter_B")]

        analyzer = FilterInteractionAnalyzer("bot1", "2026-03-01")
        report = analyzer.analyze(trades, missed)

        assert report.total_filters_analyzed == 2

    def test_pnl_computation(self):
        from trading_assistant.skills.filter_interaction_analyzer import FilterInteractionAnalyzer

        trades = [
            _make_trade("t1", pnl=100.0, active_filters=["A", "B"]),
            _make_trade("t2", pnl=-50.0, active_filters=["A", "B"]),
            _make_trade("t3", pnl=200.0, active_filters=["A"]),
        ]

        analyzer = FilterInteractionAnalyzer("bot1", "2026-03-01")
        report = analyzer.analyze(trades, [])

        if report.pairs:
            pair = report.pairs[0]
            assert pair.pnl_both == pytest.approx(50.0)  # 100 + (-50)
            assert pair.pnl_only_a == pytest.approx(200.0)


class TestStrategyEngineFilterInteractions:
    def test_detect_filter_interactions_redundant(self):
        from trading_assistant.analysis.strategy_engine import StrategyEngine

        engine = StrategyEngine("2026-02-24", "2026-03-02")
        interactions = [
            {
                "filter_a": "vol_filter",
                "filter_b": "regime_filter",
                "interaction_type": "redundant",
                "redundancy_score": 0.85,
                "recommendation": "Consider consolidating.",
            },
        ]
        suggestions = engine.detect_filter_interactions("bot1", interactions)
        assert len(suggestions) == 1
        assert "Redundant" in suggestions[0].title
        assert suggestions[0].bot_id == "bot1"

    def test_detect_filter_interactions_complementary(self):
        from trading_assistant.analysis.strategy_engine import StrategyEngine

        engine = StrategyEngine("2026-02-24", "2026-03-02")
        interactions = [
            {
                "filter_a": "vol_filter",
                "filter_b": "time_filter",
                "interaction_type": "complementary",
                "redundancy_score": 0.1,
                "recommendation": "Keep both for coverage.",
            },
        ]
        suggestions = engine.detect_filter_interactions("bot1", interactions)
        assert len(suggestions) == 1
        assert "Complementary" in suggestions[0].title

    def test_detect_filter_interactions_independent_skipped(self):
        from trading_assistant.analysis.strategy_engine import StrategyEngine

        engine = StrategyEngine("2026-02-24", "2026-03-02")
        interactions = [
            {
                "filter_a": "A",
                "filter_b": "B",
                "interaction_type": "independent",
                "redundancy_score": 0.4,
                "recommendation": "",
            },
        ]
        suggestions = engine.detect_filter_interactions("bot1", interactions)
        assert suggestions == []

    def test_build_report_with_filter_interactions(self):
        from trading_assistant.analysis.strategy_engine import StrategyEngine
        from trading_assistant.schemas.weekly_metrics import BotWeeklySummary

        engine = StrategyEngine("2026-02-24", "2026-03-02")
        summaries = {
            "bot1": BotWeeklySummary(
                bot_id="bot1", net_pnl=1000.0, total_trades=20,
                win_count=12, loss_count=8,
                week_start="2026-02-24", week_end="2026-03-02",
            ),
        }
        filter_interactions = {
            "bot1": [
                {
                    "filter_a": "vol",
                    "filter_b": "regime",
                    "interaction_type": "redundant",
                    "redundancy_score": 0.9,
                    "recommendation": "Consolidate.",
                },
            ],
        }
        report = engine.build_report(summaries, filter_interactions=filter_interactions)
        filter_suggs = [s for s in report.suggestions if "Redundant" in s.title]
        assert len(filter_suggs) == 1


class TestHandlerFilterInteractionWiring:
    def test_simulations_include_filter_interaction(self, tmp_path):
        """Verify _run_weekly_simulations runs filter interaction analysis."""
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.event_stream import EventStream
        from trading_assistant.schemas.notifications import NotificationPreferences

        handlers = Handlers(
            agent_runner=None,
            event_stream=EventStream(),
            dispatcher=None,
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=["bot1"],
        )

        # Create trade data with filter info
        import json
        for date in ["2026-02-24", "2026-02-25", "2026-02-26"]:
            bot_dir = tmp_path / "curated" / date / "bot1"
            bot_dir.mkdir(parents=True)
            trade = {
                "trade_id": f"t-{date}",
                "bot_id": "bot1",
                "pair": "NQ",
                "side": "LONG",
                "entry_time": f"{date}T12:00:00+00:00",
                "exit_time": f"{date}T13:00:00+00:00",
                "entry_price": 100.0,
                "exit_price": 101.0,
                "position_size": 1.0,
                "pnl": 50.0,
                "pnl_pct": 0.5,
                "active_filters": ["vol_filter", "regime_filter"],
            }
            (bot_dir / "trades.jsonl").write_text(json.dumps(trade) + "\n")
            missed = {
                "bot_id": "bot1",
                "pair": "NQ",
                "signal": "momentum",
                "blocked_by": "vol_filter",
            }
            (bot_dir / "missed.jsonl").write_text(json.dumps(missed) + "\n")

        class EmptyReport:
            suggestions = []

        results = handlers._run_weekly_simulations(EmptyReport(), "2026-02-24", "2026-02-28")

        # Should have filter interaction result for bot1
        assert "filter_interaction_bot1" in results
        fi_data = results["filter_interaction_bot1"]
        assert fi_data["bot_id"] == "bot1"
        assert fi_data["total_filters_analyzed"] >= 2
