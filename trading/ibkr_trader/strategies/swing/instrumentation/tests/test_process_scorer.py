"""Tests for ProcessScorer."""
import tempfile
from pathlib import Path

import yaml

from strategies.swing.instrumentation.src.process_scorer import ProcessScorer, ROOT_CAUSES


class TestProcessScorer:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        rules = {
            "global": {
                "max_entry_latency_ms": 5000,
                "max_slippage_multiplier": 2.0,
                "min_signal_strength": 0.3,
                "strong_signal_threshold": 0.7,
                "expected_slippage_bps": 5,
            },
            "strategies": {
                "trend_follow": {
                    "preferred_regimes": ["trending_up", "trending_down"],
                    "adverse_regimes": ["ranging"],
                    "expected_slippage_bps": 5,
                },
            },
        }
        self.rules_path = Path(self.tmpdir) / "rules.yaml"
        with open(self.rules_path, "w") as f:
            yaml.dump(rules, f)
        self.scorer = ProcessScorer(str(self.rules_path))

    def test_perfect_trade_scores_high(self):
        trade = {
            "trade_id": "t1", "market_regime": "trending_up",
            "entry_signal_strength": 0.8, "entry_latency_ms": 100,
            "entry_slippage_bps": 2.0, "exit_slippage_bps": 2.0,
            "exit_reason": "TAKE_PROFIT", "pnl": 500, "pnl_pct": 2.0,
        }
        score = self.scorer.score_trade(trade, "trend_follow")
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
        score = self.scorer.score_trade(trade, "trend_follow")
        assert score.process_quality_score < 50
        assert "regime_mismatch" in score.root_causes
        assert "weak_signal" in score.root_causes
        assert score.classification == "bad_process"

    def test_normal_loss_tagged_correctly(self):
        trade = {
            "trade_id": "t3", "market_regime": "trending_up",
            "entry_signal_strength": 0.8, "entry_latency_ms": 200,
            "entry_slippage_bps": 3.0, "exit_reason": "STOP_LOSS",
            "pnl": -100, "pnl_pct": -0.5,
        }
        score = self.scorer.score_trade(trade, "trend_follow")
        assert score.process_quality_score >= 80
        assert "normal_loss" in score.root_causes
        assert score.classification == "good_process"

    def test_exceptional_win_tagged(self):
        trade = {
            "trade_id": "t4", "market_regime": "trending_up",
            "entry_signal_strength": 0.9, "entry_slippage_bps": 1.0,
            "exit_reason": "TAKE_PROFIT", "pnl": 5000, "pnl_pct": 5.0,
        }
        score = self.scorer.score_trade(trade, "trend_follow")
        assert "exceptional_win" in score.root_causes

    def test_all_root_causes_from_taxonomy(self):
        trade = {"trade_id": "t5", "pnl": 0}
        score = self.scorer.score_trade(trade, "trend_follow")
        for cause in score.root_causes:
            assert cause in ROOT_CAUSES, f"'{cause}' not in ROOT_CAUSES taxonomy"

    def test_score_bounded_0_100(self):
        # Worst case trade — all penalties
        trade = {
            "trade_id": "t6", "market_regime": "ranging",
            "entry_signal_strength": 0.01, "entry_latency_ms": 50000,
            "entry_slippage_bps": 100.0, "exit_slippage_bps": 100.0,
            "exit_reason": "MANUAL", "pnl": -10000, "pnl_pct": -50,
            "funding_rate_at_entry": 0.05, "side": "LONG",
            "strategy_params_at_entry": {"sl_atr_mult": 0.3},
        }
        score = self.scorer.score_trade(trade, "trend_follow")
        assert 0 <= score.process_quality_score <= 100

    def test_scorer_handles_missing_fields(self):
        trade = {"trade_id": "t7"}
        score = self.scorer.score_trade(trade, "trend_follow")
        assert score.process_quality_score >= 0
        assert score.classification in ("good_process", "neutral", "bad_process")

    def test_to_dict(self):
        trade = {"trade_id": "t8", "pnl": 100, "pnl_pct": 1.0}
        score = self.scorer.score_trade(trade, "trend_follow")
        d = score.to_dict()
        assert isinstance(d, dict)
        assert "process_quality_score" in d
