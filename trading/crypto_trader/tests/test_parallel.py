"""Tests for parallel candidate evaluation and CachedStore."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from crypto_trader.backtest.metrics import PerformanceMetrics, metrics_to_dict
import crypto_trader.optimize.parallel as parallel_module
from crypto_trader.optimize.parallel import _CachedStore, evaluate_parallel
from crypto_trader.optimize.types import Experiment, ScoredCandidate
from crypto_trader.strategy.momentum.config import MomentumConfig


# ── metrics_to_dict ──────────────────────────────────────────────────────


class TestMetricsToDict:
    def test_basic_fields(self):
        m = PerformanceMetrics(
            net_profit=1000.0,
            total_trades=20,
            win_rate=55.0,
            sharpe_ratio=1.5,
            calmar_ratio=2.0,
            max_drawdown_pct=15.0,
        )
        d = metrics_to_dict(m)
        assert d["net_profit"] == 1000.0
        assert d["total_trades"] == 20.0  # float
        assert d["win_rate"] == 55.0
        assert d["sharpe_ratio"] == 1.5
        assert d["calmar_ratio"] == 2.0
        assert d["max_drawdown_pct"] == 15.0

    def test_all_keys_present(self):
        m = PerformanceMetrics()
        d = metrics_to_dict(m)
        expected_keys = {
            "net_profit", "net_return_pct", "realized_pnl_net",
            "terminal_mark_pnl_net", "terminal_mark_count", "total_trades", "win_rate",
            "avg_winner_r", "avg_loser_r", "expectancy_r", "profit_factor",
            "max_drawdown_pct", "max_drawdown_duration", "sharpe_ratio",
            "sortino_ratio", "calmar_ratio", "avg_bars_held", "avg_mae_r",
            "avg_mfe_r", "exit_efficiency", "a_setup_win_rate", "b_setup_win_rate",
            "long_win_rate", "short_win_rate", "total_fees", "funding_cost_total",
            "edge_ratio", "payoff_ratio", "recovery_factor",
            "max_consecutive_losses",
        }
        assert set(d.keys()) == expected_keys

    def test_int_fields_become_float(self):
        m = PerformanceMetrics(total_trades=42, max_drawdown_duration=100, max_consecutive_losses=5)
        d = metrics_to_dict(m)
        assert isinstance(d["total_trades"], float)
        assert isinstance(d["max_drawdown_duration"], float)
        assert isinstance(d["max_consecutive_losses"], float)


# ── CachedStore ──────────────────────────────────────────────────────────


class TestCachedStore:
    def test_returns_cached_data(self):
        candle_df = pd.DataFrame({"ts": [1, 2], "close": [100.0, 101.0]})
        funding_df = pd.DataFrame({"ts": [1], "rate": [0.001]})

        mock_store = MagicMock()
        mock_store.load_candles.return_value = candle_df
        mock_store.load_funding.return_value = funding_df

        cached = _CachedStore(mock_store, ["BTC"], ["15m", "1h"])

        result = cached.load_candles("BTC", "15m")
        assert result is candle_df

        result = cached.load_funding("BTC")
        assert result is funding_df

    def test_missing_returns_none(self):
        mock_store = MagicMock()
        mock_store.load_candles.return_value = None
        mock_store.load_funding.return_value = None

        cached = _CachedStore(mock_store, ["BTC"], ["15m"])

        assert cached.load_candles("ETH", "15m") is None
        assert cached.load_funding("ETH") is None

    def test_worker_cache_key_separates_timeframe_sets(self, tmp_path, monkeypatch):
        """Sequential strategy switches should not reuse the wrong cached store."""
        created = []

        class FakeCachedStore:
            def __init__(self, store, symbols, timeframes):
                self.store = store
                self.symbols = tuple(symbols)
                self.timeframes = tuple(timeframes)
                created.append(self)

        monkeypatch.setattr(parallel_module, "_CachedStore", FakeCachedStore)
        parallel_module._worker_stores.clear()
        parallel_module._worker_store = None

        parallel_module._init_worker(str(tmp_path), ["BTC"], ["15m", "1h", "4h"])
        momentum_store = parallel_module._worker_store
        parallel_module._init_worker(str(tmp_path), ["BTC"], ["30m", "4h"])
        breakout_store = parallel_module._worker_store
        parallel_module._init_worker(str(tmp_path), ["BTC"], ["15m", "1h", "4h"])

        assert momentum_store is not breakout_store
        assert parallel_module._worker_store is momentum_store
        assert len(created) == 2


# ── evaluate_parallel ────────────────────────────────────────────────────


class TestEvaluateParallel:
    def _make_bt_config(self):
        from datetime import date
        from crypto_trader.backtest.config import BacktestConfig
        return BacktestConfig(
            symbols=["BTC"],
            start_date=date(2026, 1, 1),
            end_date=date(2026, 3, 1),
        )

    def test_empty_candidates(self):
        result = evaluate_parallel(
            candidates=[],
            current_mutations={},
            cumulative_mutations={},
            base_config=MomentumConfig(),
            backtest_config=self._make_bt_config(),
            data_dir="data",
            scoring_weights={"coverage": 1.0},
            hard_rejects={},
            phase=1,
        )
        assert result == []

    @patch("crypto_trader.optimize.parallel._init_worker")
    @patch("crypto_trader.optimize.parallel._evaluate_single")
    def test_sequential_fallback(self, mock_eval, mock_init):
        """max_workers=1 runs sequentially without pool."""
        mock_eval.return_value = (0, ScoredCandidate(
            experiment=Experiment(name="TEST", mutations={"x": 1}),
            score=0.5,
            metrics={"total_trades": 10.0},
            rejected=False,
            reject_reason="",
        ))

        results = evaluate_parallel(
            candidates=[Experiment("TEST", {"x": 1})],
            current_mutations={},
            cumulative_mutations={},
            base_config=MomentumConfig(),
            backtest_config=self._make_bt_config(),
            data_dir="data",
            scoring_weights={"coverage": 1.0},
            hard_rejects={},
            phase=1,
            max_workers=1,
        )

        assert len(results) == 1
        assert results[0].experiment.name == "TEST"
        assert results[0].score == 0.5

    @patch("crypto_trader.optimize.parallel._init_worker")
    @patch("crypto_trader.optimize.parallel._evaluate_single")
    def test_result_ordering(self, mock_eval, mock_init):
        """Results are returned in input order regardless of completion order."""
        def side_effect(args):
            idx = args[0]
            name = args[1]
            return idx, ScoredCandidate(
                experiment=Experiment(name=name, mutations={}),
                score=float(idx),
                metrics={},
                rejected=False,
                reject_reason="",
            )

        mock_eval.side_effect = side_effect

        candidates = [
            Experiment("A", {}),
            Experiment("B", {}),
            Experiment("C", {}),
        ]

        results = evaluate_parallel(
            candidates=candidates,
            current_mutations={},
            cumulative_mutations={},
            base_config=MomentumConfig(),
            backtest_config=self._make_bt_config(),
            data_dir="data",
            scoring_weights={"coverage": 1.0},
            hard_rejects={},
            phase=1,
            max_workers=1,
        )

        assert [r.experiment.name for r in results] == ["A", "B", "C"]

    @patch("crypto_trader.optimize.parallel._init_worker")
    @patch("crypto_trader.optimize.parallel._evaluate_single")
    def test_exception_handling(self, mock_eval, mock_init):
        """Worker exceptions produce rejected ScoredCandidates."""
        mock_eval.side_effect = Exception("boom")

        # For sequential path, exceptions in _evaluate_single are caught
        # inside the function itself, so we need to simulate an error
        # that _evaluate_single returns
        mock_eval.return_value = (0, ScoredCandidate(
            experiment=Experiment(name="FAIL", mutations={}),
            score=0.0,
            metrics={},
            rejected=True,
            reject_reason="Exception: boom",
        ))
        mock_eval.side_effect = None

        results = evaluate_parallel(
            candidates=[Experiment("FAIL", {})],
            current_mutations={},
            cumulative_mutations={},
            base_config=MomentumConfig(),
            backtest_config=self._make_bt_config(),
            data_dir="data",
            scoring_weights={"coverage": 1.0},
            hard_rejects={},
            phase=1,
            max_workers=1,
        )

        assert len(results) == 1
        assert results[0].rejected is True

    def test_parallel_worker_error_is_retried_in_isolated_worker(self, monkeypatch):
        """A broken batch worker should not become a false strategy reject."""

        class FakeProcessPoolExecutor:
            instances = []

            def __init__(self, *args, **kwargs):
                self.call_index = len(self.instances)
                self.instances.append(self)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, item):
                future = Future()
                name = item[1]
                idx = item[0]
                if self.call_index == 0 and name == "RETRY":
                    future.set_exception(RuntimeError("process pool broke"))
                    return future

                score = 0.8 if name == "RETRY" else 0.4
                future.set_result((
                    idx,
                    ScoredCandidate(
                        experiment=Experiment(name=name, mutations=item[2]),
                        score=score,
                        metrics={"total_trades": 20.0},
                        rejected=False,
                        reject_reason="",
                    ),
                ))
                return future

        monkeypatch.setattr(parallel_module, "ProcessPoolExecutor", FakeProcessPoolExecutor)

        results = evaluate_parallel(
            candidates=[Experiment("OK", {}), Experiment("RETRY", {"x": 1})],
            current_mutations={},
            cumulative_mutations={},
            base_config=MomentumConfig(),
            backtest_config=self._make_bt_config(),
            data_dir="data",
            scoring_weights={"coverage": 1.0},
            hard_rejects={},
            phase=1,
            max_workers=2,
        )

        assert [result.experiment.name for result in results] == ["OK", "RETRY"]
        assert results[1].rejected is False
        assert results[1].score == 0.8
        assert len(FakeProcessPoolExecutor.instances) == 2


# ── MomentumConfig round-trip ────────────────────────────────────────────


class TestMomentumConfigRoundTrip:
    def test_to_dict_from_dict(self):
        original = MomentumConfig()
        d = original.to_dict()
        restored = MomentumConfig.from_dict(d)

        assert restored.indicators.ema_fast == original.indicators.ema_fast
        assert restored.indicators.ema_mid == original.indicators.ema_mid
        assert restored.bias.min_4h_conditions == original.bias.min_4h_conditions
        assert restored.exits.tp1_r == original.exits.tp1_r
        assert restored.risk.risk_pct_a == original.risk.risk_pct_a
        assert restored.symbols == original.symbols

    def test_to_dict_with_mutations(self):
        from crypto_trader.optimize.config_mutator import apply_mutations

        original = MomentumConfig()
        mutated = apply_mutations(original, {"indicators.ema_fast": 15, "exits.tp1_r": 0.8})
        d = mutated.to_dict()
        restored = MomentumConfig.from_dict(d)

        assert restored.indicators.ema_fast == 15
        assert restored.exits.tp1_r == 0.8
