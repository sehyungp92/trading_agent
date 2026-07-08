"""Tests for DailySnapshotBuilder."""
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from strategies.swing.instrumentation.src.daily_snapshot import DailySnapshotBuilder, DailySnapshot


class TestDailySnapshot:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "test_bot",
            "strategy_type": "test_strategy",
            "data_dir": self.tmpdir,
        }
        self.builder = DailySnapshotBuilder(self.config)
        self.today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _write_trades(self, trades):
        trades_dir = Path(self.tmpdir) / "trades"
        trades_dir.mkdir(parents=True, exist_ok=True)
        filepath = trades_dir / f"trades_{self.today}.jsonl"
        with open(filepath, "w") as f:
            for t in trades:
                f.write(json.dumps(t) + "\n")

    def _write_missed(self, events):
        missed_dir = Path(self.tmpdir) / "missed"
        missed_dir.mkdir(parents=True, exist_ok=True)
        filepath = missed_dir / f"missed_{self.today}.jsonl"
        with open(filepath, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

    def _write_scores(self, scores):
        scores_dir = Path(self.tmpdir) / "scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
        filepath = scores_dir / f"scores_{self.today}.jsonl"
        with open(filepath, "w") as f:
            for s in scores:
                f.write(json.dumps(s) + "\n")

    def test_build_with_trades(self):
        self._write_trades([
            {"stage": "entry", "trade_id": "t1"},
            {"stage": "exit", "trade_id": "t1", "pnl": 100, "fees_paid": 5,
             "market_regime": "trending_up"},
            {"stage": "entry", "trade_id": "t2"},
            {"stage": "exit", "trade_id": "t2", "pnl": -50, "fees_paid": 5,
             "market_regime": "trending_up"},
        ])
        snap = self.builder.build(self.today)
        assert snap.total_trades == 2
        assert snap.win_count == 1
        assert snap.loss_count == 1
        assert snap.net_pnl == 50.0  # 100 + (-50)
        assert snap.win_rate == 0.5

    def test_build_with_no_data(self):
        snap = self.builder.build(self.today)
        assert snap.total_trades == 0
        assert snap.net_pnl == 0
        assert snap.missed_count == 0

    def test_build_with_missed(self):
        self._write_missed([
            {"blocked_by": "volume_filter", "first_hit": "TP"},
            {"blocked_by": "quality_gate", "first_hit": "SL"},
            {"blocked_by": "volume_filter", "first_hit": "TP"},
        ])
        snap = self.builder.build(self.today)
        assert snap.missed_count == 3
        assert snap.missed_would_have_won == 2
        assert snap.top_missed_filter == "volume_filter"

    def test_build_with_scores(self):
        self._write_scores([
            {"process_quality_score": 90, "classification": "good_process",
             "root_causes": ["regime_aligned", "normal_win"]},
            {"process_quality_score": 30, "classification": "bad_process",
             "root_causes": ["regime_mismatch", "weak_signal"]},
        ])
        snap = self.builder.build(self.today)
        assert snap.avg_process_quality == 60.0
        assert snap.process_scores_distribution["good_process"] == 1
        assert snap.process_scores_distribution["bad_process"] == 1

    def test_save_creates_json(self):
        snap = self.builder.build(self.today)
        self.builder.save(snap)
        filepath = Path(self.tmpdir) / "daily" / f"daily_{self.today}.json"
        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert data["bot_id"] == "test_bot"
        assert data["date"] == self.today

    def test_regime_breakdown(self):
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 100, "market_regime": "trending_up"},
            {"stage": "exit", "trade_id": "t2", "pnl": -50, "market_regime": "trending_up"},
            {"stage": "exit", "trade_id": "t3", "pnl": 200, "market_regime": "volatile"},
        ])
        snap = self.builder.build(self.today)
        assert "trending_up" in snap.regime_breakdown
        assert snap.regime_breakdown["trending_up"]["trades"] == 2
        assert snap.regime_breakdown["volatile"]["trades"] == 1

    def test_profit_factor(self):
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 300, "fees_paid": 0},
            {"stage": "exit", "trade_id": "t2", "pnl": -100, "fees_paid": 0},
        ])
        snap = self.builder.build(self.today)
        assert snap.profit_factor == 3.0

    def test_excursion_and_session_aggregates(self):
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 100,
             "mfe_pct": 0.05, "mae_pct": 0.01, "exit_efficiency": 0.8,
             "market_session": "RTH"},
            {"stage": "exit", "trade_id": "t2", "pnl": -50,
             "mfe_pct": 0.02, "mae_pct": 0.03, "exit_efficiency": -1.0,
             "market_session": "RTH"},
            {"stage": "exit", "trade_id": "t3", "pnl": 200,
             "mfe_pct": 0.08, "mae_pct": 0.005, "exit_efficiency": 0.9,
             "market_session": "ETH_POST"},
        ])
        snap = self.builder.build(self.today)
        assert snap.avg_mfe_pct is not None
        assert snap.avg_mae_pct is not None
        assert snap.avg_exit_efficiency is not None
        assert abs(snap.avg_mfe_pct - 0.05) < 0.001
        assert abs(snap.avg_mae_pct - 0.015) < 0.001
        assert "RTH" in snap.session_breakdown
        assert snap.session_breakdown["RTH"]["trades"] == 2
        assert snap.session_breakdown["ETH_POST"]["trades"] == 1

    def test_snapshot_new_fields_default_none(self):
        snap = self.builder.build(self.today)
        assert snap.avg_mfe_pct is None
        assert snap.avg_mae_pct is None
        assert snap.avg_exit_efficiency is None
        assert snap.session_breakdown == {}

    def test_per_strategy_summary(self):
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 100, "fees_paid": 5,
             "strategy_id": "ATRSS", "pair": "AAPL",
             "mfe_pct": 0.04, "mae_pct": 0.01, "exit_efficiency": 0.8},
            {"stage": "exit", "trade_id": "t2", "pnl": -50, "fees_paid": 5,
             "strategy_id": "ATRSS", "pair": "MSFT",
             "mfe_pct": 0.02, "mae_pct": 0.02, "exit_efficiency": -0.5},
            {"stage": "exit", "trade_id": "t3", "pnl": 200, "fees_paid": 3,
             "strategy_id": "AKC_HELIX", "pair": "SPY",
             "mfe_pct": 0.06, "mae_pct": 0.005, "exit_efficiency": 0.9},
            {"stage": "exit", "trade_id": "t4", "pnl": 80, "fees_paid": 3,
             "strategy_id": "AKC_HELIX", "pair": "QQQ",
             "mfe_pct": 0.03, "mae_pct": 0.01, "exit_efficiency": 0.7},
        ])
        snap = self.builder.build(self.today)
        assert snap.per_strategy_summary is not None
        assert "ATRSS" in snap.per_strategy_summary
        assert "AKC_HELIX" in snap.per_strategy_summary
        atrss = snap.per_strategy_summary["ATRSS"]
        assert atrss["trades"] == 2
        assert atrss["win_count"] == 1
        assert atrss["loss_count"] == 1
        assert atrss["win_rate"] == 0.5
        assert atrss["gross_pnl"] == 50.0
        assert atrss["net_pnl"] == 40.0  # (100-5) + (-50-5) = 40
        assert sorted(atrss["symbols_traded"]) == ["AAPL", "MSFT"]
        helix = snap.per_strategy_summary["AKC_HELIX"]
        assert helix["trades"] == 2
        assert helix["win_count"] == 2
        assert helix["win_rate"] == 1.0
        assert sorted(helix["symbols_traded"]) == ["QQQ", "SPY"]

    def test_overlay_state_summary(self):
        self._write_trades([
            {"stage": "entry", "trade_id": "ov1", "strategy_id": "OVERLAY", "pair": "QQQ"},
        ])
        snap = self.builder.build(self.today)
        assert snap.overlay_state_summary is not None
        assert snap.overlay_state_summary["qqq_bullish"] is True
        assert snap.overlay_state_summary["gld_bullish"] is False
        assert snap.overlay_state_summary["entry_count_today"] == 1
        assert snap.overlay_state_summary["exit_count_today"] == 0
        assert snap.overlay_state_summary["active_symbols"] == ["QQQ"]

    def test_coordinator_actions_in_overlay_summary(self):
        coord_dir = Path(self.tmpdir) / "coordination"
        coord_dir.mkdir(parents=True, exist_ok=True)
        filepath = coord_dir / f"coordination_{self.today}.jsonl"
        actions = [
            {"action": "tighten_stop_be", "symbol": "QQQ", "outcome": "applied"},
            {"action": "tighten_stop_be", "symbol": "SPY", "outcome": "skipped_already_tighter"},
            {"action": "size_boost", "symbol": "QQQ", "outcome": "applied"},
        ]
        with open(filepath, "w") as f:
            for a in actions:
                f.write(json.dumps(a) + "\n")

        snap = self.builder.build(self.today)
        assert snap.overlay_state_summary is not None
        assert snap.overlay_state_summary["coordinator_actions_today"] == 3
        types = snap.overlay_state_summary["coordinator_action_types"]
        assert types["tighten_stop_be"] == 2
        assert types["size_boost"] == 1

    def test_avg_signal_to_fill_ms(self):
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 100,
             "execution_timeline": {
                 "signal_generated_at": "2026-03-01T10:00:00.000000",
                 "fill_confirmed_at": "2026-03-01T10:00:00.250000",
             }},
            {"stage": "exit", "trade_id": "t2", "pnl": -50,
             "execution_timeline": {
                 "signal_generated_at": "2026-03-01T11:00:00.000000",
                 "fill_confirmed_at": "2026-03-01T11:00:00.750000",
             }},
        ])
        snap = self.builder.build(self.today)
        assert snap.avg_signal_to_fill_ms is not None
        assert abs(snap.avg_signal_to_fill_ms - 500.0) < 1.0  # avg of 250ms + 750ms

    def test_avg_signal_to_fill_ms_none_when_no_timelines(self):
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 100},
        ])
        snap = self.builder.build(self.today)
        assert snap.avg_signal_to_fill_ms is None

    def test_experiment_breakdown(self):
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 100,
             "experiment_id": "exp1", "experiment_variant": "A"},
            {"stage": "exit", "trade_id": "t2", "pnl": -50,
             "experiment_id": "exp1", "experiment_variant": "A"},
            {"stage": "exit", "trade_id": "t3", "pnl": 200,
             "experiment_id": "exp1", "experiment_variant": "B"},
        ])
        snap = self.builder.build(self.today)
        assert snap.experiment_breakdown is not None
        assert "exp1:A" in snap.experiment_breakdown
        assert snap.experiment_breakdown["exp1:A"]["trades"] == 2
        assert snap.experiment_breakdown["exp1:A"]["wins"] == 1
        assert "exp1:B" in snap.experiment_breakdown
        assert snap.experiment_breakdown["exp1:B"]["trades"] == 1

    def test_experiment_breakdown_none_when_no_experiments(self):
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 100},
        ])
        snap = self.builder.build(self.today)
        assert snap.experiment_breakdown is None

    def test_overlay_excluded_from_per_strategy(self):
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 100, "fees_paid": 0,
             "strategy_id": "ATRSS", "pair": "AAPL"},
            {"stage": "exit", "trade_id": "ov1", "pnl": 50, "fees_paid": 0,
             "strategy_id": "OVERLAY", "pair": "QQQ"},
        ])
        snap = self.builder.build(self.today)
        assert snap.per_strategy_summary is not None
        assert "OVERLAY" not in snap.per_strategy_summary
        assert "ATRSS" in snap.per_strategy_summary
