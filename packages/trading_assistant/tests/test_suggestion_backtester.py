# tests/test_suggestion_backtester.py
"""Tests for SuggestionBacktester."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from trading_assistant.skills.suggestion_backtester import SuggestionBacktester
from trading_assistant.skills.config_registry import ConfigRegistry


def _make_registry(tmp_path: Path) -> ConfigRegistry:
    d = tmp_path / "bot_configs"
    d.mkdir()
    (d / "test_bot.yaml").write_text(yaml.dump({
        "bot_id": "test_bot",
        "strategies": ["alpha"],
        "parameters": [
            {
                "param_name": "quality_min",
                "param_type": "YAML_FIELD",
                "file_path": "config.yaml",
                "yaml_key": "alpha.quality_min",
                "current_value": 0.6,
                "valid_range": [0.0, 1.0],
                "value_type": "float",
                "category": "entry_signal",
                "is_safety_critical": False,
            },
            {
                "param_name": "base_risk_pct",
                "param_type": "PYTHON_CONSTANT",
                "file_path": "config/risk.py",
                "python_path": "BASE_RISK_PCT",
                "current_value": 0.02,
                "valid_range": [0.005, 0.05],
                "value_type": "float",
                "category": "risk_management",
                "is_safety_critical": True,
            },
        ],
    }), encoding="utf-8")
    return ConfigRegistry(d)


def _write_trades(tmp_path: Path, bot_id: str, trades: list[dict]) -> None:
    curated = tmp_path / "data" / "curated" / "2026-03-06" / bot_id
    curated.mkdir(parents=True)
    with open(curated / "trades.jsonl", "w") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")


def _sample_trades(n: int = 20) -> list[dict]:
    """Generate n sample trades with PnLs."""
    trades = []
    for i in range(n):
        pnl = 100.0 if i % 3 != 0 else -50.0  # ~67% win rate
        trades.append({
            "pnl": pnl,
            "date": f"2026-03-{(i % 28) + 1:02d}",
            "symbol": "AAPL",
        })
    return trades


@pytest.fixture
def setup(tmp_path: Path):
    registry = _make_registry(tmp_path)
    backtester = SuggestionBacktester(registry, tmp_path)
    return registry, backtester, tmp_path


class TestSuggestionBacktester:
    @pytest.mark.asyncio
    async def test_comparison_with_trades(self, setup):
        _, backtester, tmp_path = setup
        _write_trades(tmp_path, "test_bot", _sample_trades(20))

        result = await backtester.backtest_suggestion(
            suggestion_id="s1", bot_id="test_bot",
            param_name="quality_min", current_value=0.6, proposed_value=0.7,
        )
        assert result.context.suggestion_id == "s1"
        assert result.baseline.total_trades > 0
        assert result.proposed.total_trades > 0
        # Baseline and proposed are identical (honest about limitation)
        assert result.baseline.total_trades == result.proposed.total_trades

    @pytest.mark.asyncio
    async def test_empty_trades_fails_safety(self, setup):
        _, backtester, _ = setup
        result = await backtester.backtest_suggestion(
            suggestion_id="s1", bot_id="test_bot",
            param_name="quality_min", current_value=0.6, proposed_value=0.7,
        )
        assert result.passes_safety is False
        assert any("No trade data" in n for n in result.safety_notes)

    @pytest.mark.asyncio
    async def test_safety_sharpe_negative_fails(self, setup):
        _, backtester, tmp_path = setup
        # All losses → negative sharpe
        trades = [{"pnl": -100.0, "date": "2026-03-01"} for _ in range(15)]
        _write_trades(tmp_path, "test_bot", trades)

        result = await backtester.backtest_suggestion(
            suggestion_id="s1", bot_id="test_bot",
            param_name="quality_min", current_value=0.6, proposed_value=0.7,
        )
        assert result.passes_safety is False
        assert any("Negative Sharpe" in n or "Profit factor" in n for n in result.safety_notes)

    @pytest.mark.asyncio
    async def test_safety_profit_factor_below_one(self, setup):
        _, backtester, tmp_path = setup
        # More losses than wins
        trades = [{"pnl": -100.0, "date": "2026-03-01"} for _ in range(12)]
        trades.append({"pnl": 50.0, "date": "2026-03-02"})
        _write_trades(tmp_path, "test_bot", trades)

        result = await backtester.backtest_suggestion(
            suggestion_id="s1", bot_id="test_bot",
            param_name="quality_min", current_value=0.6, proposed_value=0.7,
        )
        assert result.passes_safety is False

    @pytest.mark.asyncio
    async def test_safety_insufficient_trades(self, setup):
        _, backtester, tmp_path = setup
        _write_trades(tmp_path, "test_bot", _sample_trades(5))

        result = await backtester.backtest_suggestion(
            suggestion_id="s1", bot_id="test_bot",
            param_name="quality_min", current_value=0.6, proposed_value=0.7,
        )
        assert result.passes_safety is False
        assert any("Insufficient trades" in n for n in result.safety_notes)

    @pytest.mark.asyncio
    async def test_safety_critical_requires_more_trades(self, setup):
        _, backtester, tmp_path = setup
        # 15 trades — enough for normal, not for safety-critical
        _write_trades(tmp_path, "test_bot", _sample_trades(15))

        result = await backtester.backtest_suggestion(
            suggestion_id="s1", bot_id="test_bot",
            param_name="base_risk_pct", current_value=0.02, proposed_value=0.03,
        )
        assert result.passes_safety is False
        assert any("Safety-critical" in n for n in result.safety_notes)

    @pytest.mark.asyncio
    async def test_trade_count_recorded_in_context(self, setup):
        _, backtester, tmp_path = setup
        _write_trades(tmp_path, "test_bot", _sample_trades(15))

        result = await backtester.backtest_suggestion(
            suggestion_id="s1", bot_id="test_bot",
            param_name="quality_min", current_value=0.6, proposed_value=0.7,
        )
        assert result.context.trade_count == 15

    @pytest.mark.asyncio
    async def test_pct_changes_computed(self, setup):
        _, backtester, tmp_path = setup
        _write_trades(tmp_path, "test_bot", _sample_trades(20))

        result = await backtester.backtest_suggestion(
            suggestion_id="s1", bot_id="test_bot",
            param_name="quality_min", current_value=0.6, proposed_value=0.7,
        )
        # Same data → 0% change (honest)
        assert result.sharpe_change_pct == pytest.approx(0.0)

    def test_compute_trade_metrics_produces_metrics(self, setup):
        _, backtester, _ = setup
        trades = _sample_trades(20)
        metrics = backtester._compute_trade_metrics(trades)
        assert metrics.total_trades == 20
        assert metrics.win_count > 0
        assert metrics.loss_count > 0
        assert metrics.profit_factor > 0

    @pytest.mark.asyncio
    async def test_passing_backtest(self, setup):
        _, backtester, tmp_path = setup
        _write_trades(tmp_path, "test_bot", _sample_trades(25))

        result = await backtester.backtest_suggestion(
            suggestion_id="s1", bot_id="test_bot",
            param_name="quality_min", current_value=0.6, proposed_value=0.7,
        )
        # With 25 trades and 67% win rate, should pass safety
        assert result.passes_safety is True

    @pytest.mark.asyncio
    async def test_safety_note_about_limitation(self, setup):
        _, backtester, tmp_path = setup
        _write_trades(tmp_path, "test_bot", _sample_trades(25))

        result = await backtester.backtest_suggestion(
            suggestion_id="s1", bot_id="test_bot",
            param_name="quality_min", current_value=0.6, proposed_value=0.7,
        )
        assert any("data quality" in n for n in result.safety_notes)
