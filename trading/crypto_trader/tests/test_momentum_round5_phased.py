"""Tests for the momentum round-5 phased optimization module."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.optimize.momentum_round5_phased import (
    IMMUTABLE_HARD_REJECTS,
    IMMUTABLE_SCORING_WEIGHTS,
    PHASE_GATE_CRITERIA,
    MomentumRound5PhasedPlugin,
    _phase1_candidates,
    _phase2_candidates,
    _phase3_candidates,
    _phase4_candidates,
    _phase5_candidates,
    load_momentum_strategy,
)
from crypto_trader.strategy.momentum.config import MomentumConfig


def _make_plugin() -> MomentumRound5PhasedPlugin:
    bc = BacktestConfig(
        start_date=date(2026, 2, 25),
        end_date=date(2026, 4, 20),
        symbols=["BTC", "ETH", "SOL"],
        initial_equity=10_000.0,
    )
    return MomentumRound5PhasedPlugin(
        backtest_config=bc,
        base_config=MomentumConfig(),
        data_dir=Path("data"),
        max_workers=2,
    )


def test_weights_sum_to_one():
    assert sum(IMMUTABLE_SCORING_WEIGHTS.values()) == pytest.approx(1.0)


def test_hard_rejects_have_required_metrics():
    required = {"total_trades", "profit_factor", "max_drawdown_pct"}
    assert required.issubset(IMMUTABLE_HARD_REJECTS.keys())
    for metric, (op, _val) in IMMUTABLE_HARD_REJECTS.items():
        assert op in (">=", "<="), f"{metric} has invalid operator {op}"


def test_each_phase_produces_candidates():
    for fn in (_phase1_candidates, _phase2_candidates, _phase3_candidates,
               _phase4_candidates, _phase5_candidates):
        candidates = fn()
        assert len(candidates) >= 3, f"{fn.__name__} returned too few candidates"
        names = [c.name for c in candidates]
        assert len(names) == len(set(names)), f"{fn.__name__} has duplicate names"


def test_phase_gate_criteria_tighten_into_later_phases():
    assert any(g.metric == "avg_mae_r" for g in PHASE_GATE_CRITERIA[2])
    assert any(g.metric == "exit_efficiency" for g in PHASE_GATE_CRITERIA[3])
    assert any(g.metric == "sharpe_ratio" for g in PHASE_GATE_CRITERIA[5])


def test_plugin_uses_immutable_weights_and_named_phases():
    plugin = _make_plugin()
    spec = plugin.get_phase_spec(1, state=None)
    assert spec.name == "Signal Discrimination"
    assert spec.scoring_weights == IMMUTABLE_SCORING_WEIGHTS
    assert spec.max_rounds == 4


def test_load_momentum_strategy_round_trip(tmp_path: Path):
    path = tmp_path / "optimized_config.json"
    payload = {
        "strategy": MomentumConfig().to_dict(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    cfg = load_momentum_strategy(path)
    assert isinstance(cfg, MomentumConfig)
    assert cfg.symbols == ["BTC", "ETH", "SOL"]
