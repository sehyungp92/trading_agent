"""Tests for Handlers._merge_daily_snapshots."""

import pytest

from trading_assistant.orchestrator.handlers import Handlers


class TestMergeDailySnapshots:
    def test_merge_single_snapshot(self):
        snapshot = {"total_trades": 5, "win_count": 3, "gross_pnl": 200.0}
        result = Handlers._merge_daily_snapshots([snapshot])
        assert result is snapshot

    def test_merge_empty_list(self):
        result = Handlers._merge_daily_snapshots([])
        assert result == {}

    def test_merge_two_strategies_additive_fields(self):
        a = {"total_trades": 5, "win_count": 3, "loss_count": 2, "gross_pnl": 200.0, "net_pnl": 180.0,
             "missed_count": 1, "missed_would_have_won": 0, "error_count": 0,
             "avg_win": 100.0, "avg_loss": -50.0, "avg_process_quality": 80.0}
        b = {"total_trades": 3, "win_count": 1, "loss_count": 2, "gross_pnl": 50.0, "net_pnl": 40.0,
             "missed_count": 2, "missed_would_have_won": 1, "error_count": 1,
             "avg_win": 50.0, "avg_loss": -30.0, "avg_process_quality": 60.0}
        result = Handlers._merge_daily_snapshots([a, b])

        assert result["total_trades"] == 8
        assert result["win_count"] == 4
        assert result["loss_count"] == 4
        assert result["gross_pnl"] == 250.0
        assert result["net_pnl"] == 220.0
        assert result["missed_count"] == 3
        assert result["missed_would_have_won"] == 1
        assert result["error_count"] == 1

    def test_merge_count_weighted_avg_win_avg_loss(self):
        a = {"total_trades": 5, "win_count": 2, "loss_count": 3, "avg_win": 100.0, "avg_loss": -40.0,
             "avg_process_quality": 70.0}
        b = {"total_trades": 5, "win_count": 3, "loss_count": 2, "avg_win": 50.0, "avg_loss": -80.0,
             "avg_process_quality": 90.0}
        result = Handlers._merge_daily_snapshots([a, b])

        # avg_win: (2*100 + 3*50) / 5 = 350/5 = 70
        assert result["avg_win"] == pytest.approx(70.0)
        # avg_loss: (3*-40 + 2*-80) / 5 = (-120 + -160) / 5 = -56
        assert result["avg_loss"] == pytest.approx(-56.0)

    def test_merge_count_weighted_avg_process_quality(self):
        a = {"total_trades": 5, "win_count": 3, "loss_count": 2, "avg_process_quality": 80.0}
        b = {"total_trades": 3, "win_count": 1, "loss_count": 2, "avg_process_quality": 60.0}
        result = Handlers._merge_daily_snapshots([a, b])

        # avg_pq: (5*80 + 3*60) / 8 = (400+180)/8 = 72.5
        assert result["avg_process_quality"] == pytest.approx(72.5)

    def test_merge_win_rate_recomputed(self):
        a = {"total_trades": 5, "win_count": 3, "loss_count": 2, "win_rate": 60.0}
        b = {"total_trades": 3, "win_count": 1, "loss_count": 2, "win_rate": 33.3}
        result = Handlers._merge_daily_snapshots([a, b])

        # win_rate: 4/8 * 100 = 50.0
        assert result["win_rate"] == pytest.approx(50.0)

    def test_merge_non_additive_from_last(self):
        a = {"total_trades": 0, "win_count": 0, "loss_count": 0,
             "sharpe_rolling_30d": 1.5, "max_drawdown_pct": 5.0}
        b = {"total_trades": 0, "win_count": 0, "loss_count": 0,
             "sharpe_rolling_30d": 2.0, "max_drawdown_pct": 3.0}
        result = Handlers._merge_daily_snapshots([a, b])

        # Non-additive fields come from last snapshot (b)
        assert result["sharpe_rolling_30d"] == 2.0
        assert result["max_drawdown_pct"] == 3.0

    def test_merge_per_strategy_summary_union(self):
        a = {"total_trades": 2, "win_count": 1, "loss_count": 1,
             "per_strategy_summary": {"ALPHA": {"trades": 2, "pnl": 100}}}
        b = {"total_trades": 3, "win_count": 2, "loss_count": 1,
             "per_strategy_summary": {"BETA": {"trades": 3, "pnl": 200}}}
        result = Handlers._merge_daily_snapshots([a, b])

        assert "ALPHA" in result["per_strategy_summary"]
        assert "BETA" in result["per_strategy_summary"]
        assert result["per_strategy_summary"]["ALPHA"]["trades"] == 2
        assert result["per_strategy_summary"]["BETA"]["trades"] == 3

    def test_merge_root_cause_distribution_summed(self):
        a = {"total_trades": 2, "win_count": 1, "loss_count": 1,
             "root_cause_distribution": {"weak_signal": 3, "slippage_spike": 1}}
        b = {"total_trades": 3, "win_count": 2, "loss_count": 1,
             "root_cause_distribution": {"weak_signal": 2, "late_entry": 4}}
        result = Handlers._merge_daily_snapshots([a, b])

        assert result["root_cause_distribution"]["weak_signal"] == 5
        assert result["root_cause_distribution"]["slippage_spike"] == 1
        assert result["root_cause_distribution"]["late_entry"] == 4

    def test_merge_missing_optional_fields(self):
        a = {"total_trades": 2, "win_count": 1, "loss_count": 1}
        b = {"total_trades": 3}
        result = Handlers._merge_daily_snapshots([a, b])

        assert result["total_trades"] == 5
        assert result["win_count"] == 1
        assert result["loss_count"] == 1
        assert result["gross_pnl"] == 0.0
        assert result["missed_count"] == 0
