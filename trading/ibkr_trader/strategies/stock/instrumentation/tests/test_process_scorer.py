import tempfile
import yaml
from pathlib import Path
from strategies.stock.instrumentation.src.process_scorer import ProcessScorer, ROOT_CAUSES


def _create_scorer():
    tmpdir = tempfile.mkdtemp()
    rules = {
        "global": {
            "max_entry_latency_ms": 5000,
            "max_slippage_multiplier": 2.0,
            "min_signal_strength": 0.3,
            "strong_signal_threshold": 0.7,
            "expected_slippage_bps": 3,
        },
        "strategies": {
            "helix": {
                "preferred_regimes": ["trending_up", "trending_down"],
                "adverse_regimes": ["ranging"],
                "expected_slippage_bps": 2,
            },
        },
    }
    rules_path = Path(tmpdir) / "rules.yaml"
    with open(rules_path, "w") as f:
        yaml.dump(rules, f)
    return ProcessScorer(str(rules_path))


class TestProcessScorer:
    def setup_method(self):
        self.scorer = _create_scorer()

    def test_perfect_trade_scores_high(self):
        trade = {
            "trade_id": "t1", "market_regime": "trending_up",
            "entry_signal_strength": 0.8, "entry_latency_ms": 100,
            "entry_slippage_bps": 1.0, "exit_slippage_bps": 1.0,
            "exit_reason": "TAKE_PROFIT", "pnl": 500, "pnl_pct": 2.0,
        }
        score = self.scorer.score_trade(trade, "helix")
        assert score.process_quality_score >= 80
        assert "regime_aligned" in score.root_causes
        assert score.classification == "good_process"

    def test_bad_trade_scores_low(self):
        trade = {
            "trade_id": "t2", "market_regime": "ranging",
            "entry_signal_strength": 0.1, "entry_latency_ms": 10000,
            "entry_slippage_bps": 20.0, "exit_slippage_bps": 15.0,
            "exit_reason": "STOP_LOSS", "pnl": -200, "pnl_pct": -1.5,
        }
        score = self.scorer.score_trade(trade, "helix")
        assert score.process_quality_score < 50
        assert "regime_mismatch" in score.root_causes
        assert "weak_signal" in score.root_causes
        assert score.classification == "bad_process"

    def test_normal_loss_tagged_correctly(self):
        """Good process but negative PnL = normal_loss, not bad_process."""
        trade = {
            "trade_id": "t3", "market_regime": "trending_up",
            "entry_signal_strength": 0.8, "entry_latency_ms": 200,
            "entry_slippage_bps": 1.0, "exit_reason": "STOP_LOSS",
            "pnl": -100, "pnl_pct": -0.5,
        }
        score = self.scorer.score_trade(trade, "helix")
        assert score.process_quality_score >= 80
        assert "normal_loss" in score.root_causes
        assert score.classification == "good_process"

    def test_all_root_causes_from_taxonomy(self):
        """No root cause should exist outside the controlled taxonomy."""
        trade = {"trade_id": "t4", "pnl": 0}
        score = self.scorer.score_trade(trade, "helix")
        for cause in score.root_causes:
            assert cause in ROOT_CAUSES, f"'{cause}' not in ROOT_CAUSES taxonomy"

    def test_strong_signal_tagged(self):
        trade = {
            "trade_id": "t5", "market_regime": "trending_up",
            "entry_signal_strength": 0.9, "pnl": 100, "pnl_pct": 1.0,
        }
        score = self.scorer.score_trade(trade, "helix")
        assert "strong_signal" in score.root_causes

    def test_late_entry_tagged(self):
        trade = {
            "trade_id": "t6", "entry_latency_ms": 8000,
            "pnl": 0,
        }
        score = self.scorer.score_trade(trade, "helix")
        assert "late_entry" in score.root_causes

    def test_slippage_spike_tagged(self):
        trade = {
            "trade_id": "t7", "entry_slippage_bps": 15.0,
            "pnl": 0,
        }
        score = self.scorer.score_trade(trade, "helix")
        assert "slippage_spike" in score.root_causes

    def test_good_execution_tagged(self):
        trade = {
            "trade_id": "t8", "entry_slippage_bps": 0.5,
            "pnl": 100, "pnl_pct": 1.0,
        }
        score = self.scorer.score_trade(trade, "helix")
        assert "good_execution" in score.root_causes

    def test_scorer_never_crashes(self):
        """ProcessScorer must never crash even with garbage input."""
        score = self.scorer.score_trade({}, "unknown_strategy")
        assert score.process_quality_score >= 0
        assert score.process_quality_score <= 100

    def test_scorer_handles_missing_rules_path(self):
        scorer = ProcessScorer("/nonexistent/path.yaml")
        score = scorer.score_trade({"trade_id": "t9", "pnl": 0}, "default")
        assert score.process_quality_score >= 0

    def test_score_clamped_0_to_100(self):
        # Many penalties stacked
        trade = {
            "trade_id": "t10", "market_regime": "ranging",
            "entry_signal_strength": 0.01, "entry_latency_ms": 99999,
            "entry_slippage_bps": 999, "exit_slippage_bps": 999,
            "exit_reason": "MANUAL", "pnl": -5000, "pnl_pct": -50,
        }
        score = self.scorer.score_trade(trade, "helix")
        assert score.process_quality_score >= 0
        assert score.process_quality_score <= 100

    def test_to_dict(self):
        trade = {"trade_id": "t11", "pnl": 100, "pnl_pct": 1.0}
        score = self.scorer.score_trade(trade, "helix")
        d = score.to_dict()
        assert isinstance(d, dict)
        assert "process_quality_score" in d
        assert "root_causes" in d
        assert "classification" in d
