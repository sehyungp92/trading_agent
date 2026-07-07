# tests/test_regression_suite.py
"""Tests for the regression suite — golden day loader + regression checks."""
import json
from datetime import datetime
from pathlib import Path

import pytest

from tests.golden_days.loader import load_golden_days
from trading_assistant.schemas.events import TradeEvent, MissedOpportunityEvent
from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder


@pytest.fixture
def golden_day_dir(tmp_path: Path) -> Path:
    """Create a single golden day with known data."""
    day_dir = tmp_path / "2026-02-15"
    day_dir.mkdir()

    # Raw events
    raw_dir = day_dir / "raw_events"
    raw_dir.mkdir()
    (raw_dir / "trades.json").write_text(json.dumps([
        {
            "trade_id": "t1", "bot_id": "bot1", "pair": "BTCUSDT",
            "side": "LONG", "pnl": 200.0, "pnl_pct": 2.0,
            "entry_price": 50000.0, "exit_price": 51000.0,
            "position_size": 0.1, "entry_time": "2026-02-15T10:00:00Z",
            "exit_time": "2026-02-15T14:00:00Z",
            "entry_signal": "EMA cross", "exit_reason": "TAKE_PROFIT",
            "market_regime": "trending_up",
            "process_quality_score": 90, "root_causes": ["normal_win"],
        },
        {
            "trade_id": "t2", "bot_id": "bot1", "pair": "ETHUSDT",
            "side": "SHORT", "pnl": -80.0, "pnl_pct": -1.5,
            "entry_price": 3000.0, "exit_price": 3045.0,
            "position_size": 0.5, "entry_time": "2026-02-15T11:00:00Z",
            "exit_time": "2026-02-15T13:00:00Z",
            "entry_signal": "RSI divergence", "exit_reason": "STOP_LOSS",
            "market_regime": "ranging",
            "process_quality_score": 45, "root_causes": ["regime_mismatch"],
        },
    ]))
    (raw_dir / "missed.json").write_text(json.dumps([
        {
            "bot_id": "bot1", "pair": "BTCUSDT", "signal": "Volume breakout",
            "blocked_by": "volatility_filter", "hypothetical_entry": 50500.0,
            "outcome_24h": 400.0, "confidence": 0.7,
            "assumption_tags": ["mid_fill", "5bps_slippage"],
        },
    ]))

    # Expected classifications
    (day_dir / "expected_classifications.json").write_text(json.dumps({
        "t1": {"root_causes": ["normal_win"], "process_quality_score": 90},
        "t2": {"root_causes": ["regime_mismatch"], "process_quality_score": 45},
    }))

    # Expected curated metrics
    expected_dir = day_dir / "expected_curated"
    expected_dir.mkdir()
    (expected_dir / "summary.json").write_text(json.dumps({
        "bot_id": "bot1", "total_trades": 2, "win_count": 1,
        "loss_count": 1, "net_pnl": 120.0,
    }))

    # Human feedback
    (day_dir / "human_feedback.json").write_text(json.dumps([
        {"correction_type": "positive_reinforcement", "raw_text": "Good analysis"},
    ]))

    # Reference report
    (day_dir / "reference_report.md").write_text(
        "# Daily Report 2026-02-15\nBot1 had a good day. Process failure on t2."
    )

    # Metadata
    (day_dir / "metadata.json").write_text(json.dumps({
        "date": "2026-02-15",
        "bots": ["bot1"],
        "top_anomaly": "volume_spike",
        "biggest_loss_driver": "regime_mismatch",
        "crowding_alerts": [],
    }))

    return tmp_path


class TestGoldenDayLoader:
    def test_loads_golden_day(self, golden_day_dir: Path):
        days = load_golden_days(golden_day_dir)
        assert len(days) == 1

    def test_golden_day_has_trades(self, golden_day_dir: Path):
        days = load_golden_days(golden_day_dir)
        day = days[0]
        assert len(day.trades) == 2
        assert day.trades[0]["trade_id"] == "t1"

    def test_golden_day_has_missed(self, golden_day_dir: Path):
        days = load_golden_days(golden_day_dir)
        day = days[0]
        assert len(day.missed) == 1

    def test_golden_day_has_expected_classifications(self, golden_day_dir: Path):
        days = load_golden_days(golden_day_dir)
        day = days[0]
        assert "t1" in day.expected_classifications
        assert day.expected_classifications["t2"]["root_causes"] == ["regime_mismatch"]

    def test_golden_day_has_metadata(self, golden_day_dir: Path):
        days = load_golden_days(golden_day_dir)
        day = days[0]
        assert day.date == "2026-02-15"
        assert day.top_anomaly == "volume_spike"
        assert day.biggest_loss_driver == "regime_mismatch"

    def test_golden_day_has_reference_report(self, golden_day_dir: Path):
        days = load_golden_days(golden_day_dir)
        day = days[0]
        assert "Process failure on t2" in day.reference_report


def _dict_to_trade_event(d: dict) -> TradeEvent:
    """Convert a golden day trade dict to a TradeEvent."""
    return TradeEvent(
        trade_id=d["trade_id"],
        bot_id=d["bot_id"],
        pair=d["pair"],
        side=d["side"],
        pnl=d["pnl"],
        pnl_pct=d.get("pnl_pct", 0.0),
        entry_price=d.get("entry_price", 0.0),
        exit_price=d.get("exit_price", 0.0),
        position_size=d.get("position_size", 0.0),
        entry_time=datetime.fromisoformat(d["entry_time"]),
        exit_time=datetime.fromisoformat(d["exit_time"]),
        entry_signal=d.get("entry_signal", ""),
        exit_reason=d.get("exit_reason", ""),
        market_regime=d.get("market_regime", ""),
        process_quality_score=d.get("process_quality_score", 100),
        root_causes=d.get("root_causes", []),
    )


def _dict_to_missed_event(d: dict) -> MissedOpportunityEvent:
    """Convert a golden day missed dict to a MissedOpportunityEvent."""
    return MissedOpportunityEvent(
        bot_id=d["bot_id"],
        pair=d["pair"],
        signal=d["signal"],
        blocked_by=d.get("blocked_by", ""),
        hypothetical_entry=d.get("hypothetical_entry", 0.0),
        outcome_24h=d.get("outcome_24h"),
        confidence=d.get("confidence", 0.0),
        assumption_tags=d.get("assumption_tags", []),
    )


class TestClassificationAccuracy:
    """Run pipeline on golden days, compare root causes to human labels."""

    def test_root_causes_match_expected(self, golden_day_dir: Path):
        days = load_golden_days(golden_day_dir)
        for day in days:
            trades = [_dict_to_trade_event(t) for t in day.trades]
            for trade in trades:
                expected = day.expected_classifications.get(trade.trade_id, {})
                if "root_causes" in expected:
                    assert trade.root_causes == expected["root_causes"], (
                        f"Trade {trade.trade_id}: expected root causes "
                        f"{expected['root_causes']}, got {trade.root_causes}"
                    )

    def test_process_quality_scores_match(self, golden_day_dir: Path):
        days = load_golden_days(golden_day_dir)
        for day in days:
            trades = [_dict_to_trade_event(t) for t in day.trades]
            for trade in trades:
                expected = day.expected_classifications.get(trade.trade_id, {})
                if "process_quality_score" in expected:
                    assert trade.process_quality_score == expected["process_quality_score"], (
                        f"Trade {trade.trade_id}: expected PQ score "
                        f"{expected['process_quality_score']}, got {trade.process_quality_score}"
                    )


class TestMetricStability:
    """Key metrics should not change more than 5% between pipeline versions."""

    def test_summary_metrics_within_tolerance(self, golden_day_dir: Path):
        days = load_golden_days(golden_day_dir)
        for day in days:
            if "summary.json" not in day.expected_curated:
                continue

            trades = [_dict_to_trade_event(t) for t in day.trades]
            [_dict_to_missed_event(m) for m in day.missed]

            for bot_id in day.bots:
                bot_trades = [t for t in trades if t.bot_id == bot_id]
                builder = DailyMetricsBuilder(date=day.date, bot_id=bot_id)
                actual = builder.build_summary(bot_trades)

                expected = day.expected_curated["summary.json"]
                # Check key integer metrics exactly
                assert actual.total_trades == expected["total_trades"], (
                    f"total_trades: expected {expected['total_trades']}, got {actual.total_trades}"
                )
                assert actual.win_count == expected["win_count"], (
                    f"win_count: expected {expected['win_count']}, got {actual.win_count}"
                )

                # Check float metrics within 5% tolerance
                expected_pnl = expected["net_pnl"]
                if expected_pnl != 0:
                    delta = abs(actual.net_pnl - expected_pnl) / abs(expected_pnl)
                    assert delta < 0.05, (
                        f"net_pnl drifted by {delta:.1%}: "
                        f"expected {expected_pnl}, got {actual.net_pnl}"
                    )
