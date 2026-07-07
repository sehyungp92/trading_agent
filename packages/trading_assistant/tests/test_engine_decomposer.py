"""Tests for EngineDecomposer — per-engine decomposition of bot performance."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_assistant.schemas.engine_metrics import (
    EngineDecomposition,
    EngineMetrics,
    RegimeEngineStats,
)
from trading_assistant.schemas.events import TradeEvent
from trading_assistant.schemas.strategy_profile import StrategyProfile, StrategyRegistry
from trading_assistant.skills.engine_decomposer import EngineDecomposer
from tests.factories import make_trade


# ─── Helpers ─────────────────────────────────────────────────────────


def _make_registry(
    strategy_id: str = "DownturnDominator_v1",
    bot_id: str = "bot1",
    sub_engines: list[str] | None = None,
) -> StrategyRegistry:
    """Build a StrategyRegistry with one strategy containing sub_engines."""
    if sub_engines is None:
        sub_engines = ["reversal", "breakdown", "fade"]
    return StrategyRegistry(
        strategies={
            strategy_id: StrategyProfile(
                display_name=strategy_id,
                bot_id=bot_id,
                sub_engines=sub_engines,
            ),
        },
    )


def _make_registry_no_sub_engines(
    strategy_id: str = "NQDTC_v2.1",
    bot_id: str = "bot2",
) -> StrategyRegistry:
    """Build a registry for a strategy without sub_engines."""
    return StrategyRegistry(
        strategies={
            strategy_id: StrategyProfile(
                display_name=strategy_id,
                bot_id=bot_id,
                sub_engines=[],
            ),
        },
    )


def _trade(
    trade_id: str = "t1",
    entry_signal: str = "reversal_short",
    strategy_id: str = "DownturnDominator_v1",
    market_regime: str = "trending",
    pnl: float = 100.0,
    exit_efficiency: float | None = None,
    entry_signal_strength: float = 0.0,
    bot_id: str = "bot1",
) -> "TradeEvent":
    return make_trade(
        trade_id=trade_id,
        bot_id=bot_id,
        entry_signal=entry_signal,
        strategy_id=strategy_id,
        market_regime=market_regime,
        pnl=pnl,
        exit_efficiency=exit_efficiency,
        entry_signal_strength=entry_signal_strength,
    )


# ─── Schema Tests ────────────────────────────────────────────────────


class TestSchemaDefaults:
    """EngineMetrics, RegimeEngineStats, EngineDecomposition instantiation."""

    def test_regime_engine_stats_defaults(self):
        stats = RegimeEngineStats(regime="trending")
        assert stats.regime == "trending"
        assert stats.trade_count == 0
        assert stats.win_rate == 0.0
        assert stats.avg_pnl == 0.0
        assert stats.profit_factor == 0.0

    def test_engine_metrics_defaults(self):
        m = EngineMetrics(engine="REVERSAL")
        assert m.engine == "REVERSAL"
        assert m.strategy_id == ""
        assert m.bot_id == ""
        assert m.trade_count == 0
        assert m.win_rate == 0.0
        assert m.avg_pnl == 0.0
        assert m.profit_factor == 0.0
        assert m.avg_exit_efficiency == 0.0
        assert m.avg_signal_strength == 0.0
        assert m.regime_breakdown == []

    def test_engine_decomposition_defaults(self):
        d = EngineDecomposition(bot_id="bot1")
        assert d.bot_id == "bot1"
        assert d.period == ""
        assert d.engines == []
        assert d.unmapped_trades == 0

    def test_engine_decomposition_with_period(self):
        d = EngineDecomposition(bot_id="bot1", period="2026-05-01")
        assert d.period == "2026-05-01"

    def test_engine_metrics_serialization(self):
        m = EngineMetrics(
            engine="FADE",
            trade_count=10,
            win_rate=0.6,
            avg_pnl=50.0,
            regime_breakdown=[
                RegimeEngineStats(regime="trending", trade_count=5, win_rate=0.8),
            ],
        )
        data = m.model_dump(mode="json")
        assert data["engine"] == "FADE"
        assert data["regime_breakdown"][0]["regime"] == "trending"


# ─── Engine Tag Parsing ──────────────────────────────────────────────


class TestParseEngine:
    """parse_engine() extracts engine tag from entry_signal."""

    def test_reversal_prefix(self):
        reg = _make_registry()
        d = EngineDecomposer(reg)
        assert d.parse_engine("reversal_short", "DownturnDominator_v1") == "REVERSAL"

    def test_breakdown_prefix(self):
        reg = _make_registry()
        d = EngineDecomposer(reg)
        assert d.parse_engine("breakdown_long_aggressive", "DownturnDominator_v1") == "BREAKDOWN"

    def test_fade_prefix(self):
        reg = _make_registry()
        d = EngineDecomposer(reg)
        assert d.parse_engine("fade_short", "DownturnDominator_v1") == "FADE"

    def test_case_insensitivity(self):
        reg = _make_registry()
        d = EngineDecomposer(reg)
        assert d.parse_engine("REVERSAL_SHORT", "DownturnDominator_v1") == "REVERSAL"

    def test_case_insensitivity_mixed(self):
        reg = _make_registry()
        d = EngineDecomposer(reg)
        assert d.parse_engine("Reversal_Long", "DownturnDominator_v1") == "REVERSAL"

    def test_fallback_unknown_signal(self):
        """Signal that doesn't match any engine prefix returns full string."""
        reg = _make_registry()
        d = EngineDecomposer(reg)
        result = d.parse_engine("unknown_signal", "DownturnDominator_v1")
        assert result == "unknown_signal"

    def test_strategy_without_sub_engines(self):
        """Strategy without sub_engines returns full signal string."""
        reg = _make_registry_no_sub_engines()
        d = EngineDecomposer(reg)
        result = d.parse_engine("some_signal", "NQDTC_v2.1")
        assert result == "some_signal"

    def test_no_registry(self):
        """EngineDecomposer(None) returns full signal string."""
        d = EngineDecomposer(None)
        result = d.parse_engine("reversal_short", "DownturnDominator_v1")
        assert result == "reversal_short"

    def test_empty_entry_signal(self):
        """Empty entry_signal returns 'unknown'."""
        reg = _make_registry()
        d = EngineDecomposer(reg)
        assert d.parse_engine("", "DownturnDominator_v1") == "unknown"

    def test_strategy_not_in_registry(self):
        """Unknown strategy_id falls back to full signal string."""
        reg = _make_registry()
        d = EngineDecomposer(reg)
        result = d.parse_engine("reversal_short", "NonExistent_v1")
        assert result == "reversal_short"

    def test_hyphen_separator(self):
        """Engine patterns match hyphen separators too."""
        reg = _make_registry()
        d = EngineDecomposer(reg)
        assert d.parse_engine("reversal-short", "DownturnDominator_v1") == "REVERSAL"

    def test_exact_engine_name_no_suffix(self):
        """Bare engine name with optional separator matches."""
        reg = _make_registry()
        d = EngineDecomposer(reg)
        # Pattern is ^reversal[_\-]? so "reversal" alone should match
        assert d.parse_engine("reversal", "DownturnDominator_v1") == "REVERSAL"


# ─── Decomposition ──────────────────────────────────────────────────


class TestDecompose:
    """decompose() groups trades by engine and computes metrics."""

    def test_empty_trades(self):
        reg = _make_registry()
        d = EngineDecomposer(reg)
        result = d.decompose([], "bot1", period="2026-05-01")
        assert result.bot_id == "bot1"
        assert result.period == "2026-05-01"
        assert result.engines == []
        assert result.unmapped_trades == 0

    def test_groups_by_engine(self):
        reg = _make_registry()
        d = EngineDecomposer(reg)
        trades = [
            _trade(trade_id="t1", entry_signal="reversal_short", pnl=100),
            _trade(trade_id="t2", entry_signal="reversal_long", pnl=50),
            _trade(trade_id="t3", entry_signal="fade_short", pnl=-30),
            _trade(trade_id="t4", entry_signal="breakdown_long_aggressive", pnl=200),
        ]
        result = d.decompose(trades, "bot1", period="2026-05-01")
        engine_names = [e.engine for e in result.engines]
        assert "REVERSAL" in engine_names
        assert "FADE" in engine_names
        assert "BREAKDOWN" in engine_names

    def test_reversal_engine_count(self):
        reg = _make_registry()
        d = EngineDecomposer(reg)
        trades = [
            _trade(trade_id="t1", entry_signal="reversal_short", pnl=100),
            _trade(trade_id="t2", entry_signal="reversal_long", pnl=50),
            _trade(trade_id="t3", entry_signal="fade_short", pnl=-30),
        ]
        result = d.decompose(trades, "bot1")
        rev = next(e for e in result.engines if e.engine == "REVERSAL")
        assert rev.trade_count == 2

    def test_win_rate(self):
        reg = _make_registry()
        d = EngineDecomposer(reg)
        trades = [
            _trade(trade_id="t1", entry_signal="reversal_short", pnl=100),
            _trade(trade_id="t2", entry_signal="reversal_long", pnl=-50),
            _trade(trade_id="t3", entry_signal="reversal_mid", pnl=200),
        ]
        result = d.decompose(trades, "bot1")
        rev = next(e for e in result.engines if e.engine == "REVERSAL")
        # 2 wins out of 3 = 0.6667
        assert rev.win_rate == pytest.approx(0.6667, abs=0.001)

    def test_avg_pnl(self):
        reg = _make_registry()
        d = EngineDecomposer(reg)
        trades = [
            _trade(trade_id="t1", entry_signal="reversal_short", pnl=100),
            _trade(trade_id="t2", entry_signal="reversal_long", pnl=-50),
            _trade(trade_id="t3", entry_signal="reversal_mid", pnl=200),
        ]
        result = d.decompose(trades, "bot1")
        rev = next(e for e in result.engines if e.engine == "REVERSAL")
        expected_avg = (100 + (-50) + 200) / 3
        assert rev.avg_pnl == pytest.approx(expected_avg, abs=0.01)

    def test_profit_factor(self):
        reg = _make_registry()
        d = EngineDecomposer(reg)
        trades = [
            _trade(trade_id="t1", entry_signal="fade_short", pnl=300),
            _trade(trade_id="t2", entry_signal="fade_swing", pnl=-100),
            _trade(trade_id="t3", entry_signal="fade_dip", pnl=-50),
        ]
        result = d.decompose(trades, "bot1")
        fade = next(e for e in result.engines if e.engine == "FADE")
        # profit_factor = 300 / (100 + 50) = 2.0
        assert fade.profit_factor == pytest.approx(2.0, abs=0.01)

    def test_profit_factor_no_losses(self):
        """All wins: profit_factor capped at 99.99."""
        reg = _make_registry()
        d = EngineDecomposer(reg)
        trades = [
            _trade(trade_id="t1", entry_signal="reversal_a", pnl=100),
            _trade(trade_id="t2", entry_signal="reversal_b", pnl=200),
        ]
        result = d.decompose(trades, "bot1")
        rev = next(e for e in result.engines if e.engine == "REVERSAL")
        # inf capped to 99.99
        assert rev.profit_factor == 99.99

    def test_profit_factor_no_wins_no_losses(self):
        """All zero PnL: profit_factor = 0.0."""
        reg = _make_registry()
        d = EngineDecomposer(reg)
        trades = [
            _trade(trade_id="t1", entry_signal="reversal_a", pnl=0),
            _trade(trade_id="t2", entry_signal="reversal_b", pnl=0),
        ]
        result = d.decompose(trades, "bot1")
        rev = next(e for e in result.engines if e.engine == "REVERSAL")
        assert rev.profit_factor == 0.0

    def test_unmapped_trades_counted(self):
        """Trades with empty entry_signal are counted as unmapped."""
        reg = _make_registry()
        d = EngineDecomposer(reg)
        trades = [
            _trade(trade_id="t1", entry_signal="reversal_short", pnl=100),
            _trade(trade_id="t2", entry_signal="", pnl=-50),
            _trade(trade_id="t3", entry_signal="", pnl=30),
        ]
        result = d.decompose(trades, "bot1")
        assert result.unmapped_trades == 2
        # Only REVERSAL engine in output — "unknown" trades excluded from engines list
        assert len(result.engines) == 1
        assert result.engines[0].engine == "REVERSAL"

    def test_avg_exit_efficiency(self):
        reg = _make_registry()
        d = EngineDecomposer(reg)
        trades = [
            _trade(trade_id="t1", entry_signal="reversal_a", pnl=100, exit_efficiency=0.8),
            _trade(trade_id="t2", entry_signal="reversal_b", pnl=50, exit_efficiency=0.6),
        ]
        result = d.decompose(trades, "bot1")
        rev = next(e for e in result.engines if e.engine == "REVERSAL")
        assert rev.avg_exit_efficiency == pytest.approx(0.7, abs=0.01)

    def test_avg_exit_efficiency_none_values(self):
        """None exit_efficiency values are skipped (result is 0 if all None)."""
        reg = _make_registry()
        d = EngineDecomposer(reg)
        trades = [
            _trade(trade_id="t1", entry_signal="reversal_a", pnl=100, exit_efficiency=None),
            _trade(trade_id="t2", entry_signal="reversal_b", pnl=50, exit_efficiency=None),
        ]
        result = d.decompose(trades, "bot1")
        rev = next(e for e in result.engines if e.engine == "REVERSAL")
        assert rev.avg_exit_efficiency == 0.0

    def test_avg_signal_strength(self):
        reg = _make_registry()
        d = EngineDecomposer(reg)
        trades = [
            _trade(trade_id="t1", entry_signal="fade_a", pnl=100, entry_signal_strength=0.9),
            _trade(trade_id="t2", entry_signal="fade_b", pnl=50, entry_signal_strength=0.5),
        ]
        result = d.decompose(trades, "bot1")
        fade = next(e for e in result.engines if e.engine == "FADE")
        assert fade.avg_signal_strength == pytest.approx(0.7, abs=0.01)

    def test_avg_signal_strength_zero_skipped(self):
        """Zero signal strengths are skipped."""
        reg = _make_registry()
        d = EngineDecomposer(reg)
        trades = [
            _trade(trade_id="t1", entry_signal="fade_a", pnl=100, entry_signal_strength=0.0),
            _trade(trade_id="t2", entry_signal="fade_b", pnl=50, entry_signal_strength=0.0),
        ]
        result = d.decompose(trades, "bot1")
        fade = next(e for e in result.engines if e.engine == "FADE")
        assert fade.avg_signal_strength == 0.0


# ─── Regime Breakdown ────────────────────────────────────────────────


class TestRegimeBreakdown:
    """Per-regime breakdown within an engine."""

    def test_regime_groups(self):
        reg = _make_registry()
        d = EngineDecomposer(reg)
        trades = [
            _trade(trade_id="t1", entry_signal="reversal_short", pnl=100, market_regime="trending"),
            _trade(trade_id="t2", entry_signal="reversal_long", pnl=-50, market_regime="trending"),
            _trade(trade_id="t3", entry_signal="reversal_dip", pnl=200, market_regime="ranging"),
        ]
        result = d.decompose(trades, "bot1")
        rev = next(e for e in result.engines if e.engine == "REVERSAL")
        regimes = {r.regime: r for r in rev.regime_breakdown}
        assert "trending" in regimes
        assert "ranging" in regimes

    def test_regime_win_rate(self):
        reg = _make_registry()
        d = EngineDecomposer(reg)
        trades = [
            _trade(trade_id="t1", entry_signal="reversal_a", pnl=100, market_regime="trending"),
            _trade(trade_id="t2", entry_signal="reversal_b", pnl=-50, market_regime="trending"),
        ]
        result = d.decompose(trades, "bot1")
        rev = next(e for e in result.engines if e.engine == "REVERSAL")
        trending = next(r for r in rev.regime_breakdown if r.regime == "trending")
        assert trending.win_rate == pytest.approx(0.5, abs=0.01)
        assert trending.trade_count == 2

    def test_regime_profit_factor(self):
        reg = _make_registry()
        d = EngineDecomposer(reg)
        trades = [
            _trade(trade_id="t1", entry_signal="reversal_a", pnl=300, market_regime="ranging"),
            _trade(trade_id="t2", entry_signal="reversal_b", pnl=-100, market_regime="ranging"),
        ]
        result = d.decompose(trades, "bot1")
        rev = next(e for e in result.engines if e.engine == "REVERSAL")
        ranging = next(r for r in rev.regime_breakdown if r.regime == "ranging")
        assert ranging.profit_factor == pytest.approx(3.0, abs=0.01)

    def test_empty_regime_defaults_to_unknown(self):
        reg = _make_registry()
        d = EngineDecomposer(reg)
        trades = [
            _trade(trade_id="t1", entry_signal="reversal_a", pnl=100, market_regime=""),
        ]
        result = d.decompose(trades, "bot1")
        rev = next(e for e in result.engines if e.engine == "REVERSAL")
        regimes = {r.regime for r in rev.regime_breakdown}
        assert "unknown" in regimes


# ─── Strategy Resolution ─────────────────────────────────────────────


class TestStrategyResolution:
    """_resolve_strategy_id correctly identifies the strategy for a bot."""

    def test_single_strategy_for_bot(self):
        reg = _make_registry(strategy_id="DD_v1", bot_id="bot1")
        d = EngineDecomposer(reg)
        trades = [_trade(trade_id="t1", strategy_id="", bot_id="bot1")]
        result = d.decompose(trades, "bot1")
        # With no sub_engines matched, engine = full signal
        # The point is it resolved strategy_id correctly
        assert result.bot_id == "bot1"

    def test_strategy_from_trade_field(self):
        """When multiple strategies for a bot, infer from trade.strategy_id."""
        reg = StrategyRegistry(
            strategies={
                "DD_v1": StrategyProfile(bot_id="bot1", sub_engines=["reversal"]),
                "DD_v2": StrategyProfile(bot_id="bot1", sub_engines=["breakout"]),
            },
        )
        d = EngineDecomposer(reg)
        trades = [
            _trade(trade_id="t1", strategy_id="DD_v1", entry_signal="reversal_short"),
        ]
        result = d.decompose(trades, "bot1")
        rev = next((e for e in result.engines if e.engine == "REVERSAL"), None)
        assert rev is not None
        assert rev.strategy_id == "DD_v1"

    def test_no_registry_empty_strategy(self):
        d = EngineDecomposer(None)
        trades = [_trade(trade_id="t1", entry_signal="reversal_short")]
        result = d.decompose(trades, "bot1")
        # No registry → full signal string used as engine name
        assert result.engines[0].engine == "reversal_short"
        assert result.engines[0].strategy_id == ""


# ─── Integration with build_daily_metrics ─────────────────────────────


class TestBuildDailyMetricsIntegration:
    """write_curated produces engine_decomposition.json when registry is provided."""

    def test_produces_engine_decomposition_json(self, tmp_path: Path):
        from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder

        builder = DailyMetricsBuilder("2026-05-01", "bot1")
        reg = _make_registry()
        trades = [
            _trade(trade_id="t1", entry_signal="reversal_short", pnl=100),
            _trade(trade_id="t2", entry_signal="fade_short", pnl=-30),
        ]
        builder.write_curated(
            trades=trades,
            missed=[],
            base_dir=tmp_path,
            strategy_registry=reg,
        )
        output = tmp_path / "2026-05-01" / "bot1" / "engine_decomposition.json"
        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["bot_id"] == "bot1"
        assert len(data["engines"]) >= 1
        engine_names = [e["engine"] for e in data["engines"]]
        assert "REVERSAL" in engine_names

    def test_no_engine_decomposition_without_registry(self, tmp_path: Path):
        from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder

        builder = DailyMetricsBuilder("2026-05-01", "bot1")
        trades = [
            _trade(trade_id="t1", entry_signal="reversal_short", pnl=100),
        ]
        builder.write_curated(
            trades=trades,
            missed=[],
            base_dir=tmp_path,
            strategy_registry=None,
        )
        output = tmp_path / "2026-05-01" / "bot1" / "engine_decomposition.json"
        assert not output.exists()

    def test_no_engine_decomposition_for_empty_engines(self, tmp_path: Path):
        """If no trades match any engine, the file is not written (engines list empty)."""
        from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder

        builder = DailyMetricsBuilder("2026-05-01", "bot1")
        reg = _make_registry()
        # All trades with empty entry_signal → unmapped → no engines in output
        trades = [
            _trade(trade_id="t1", entry_signal="", pnl=100),
            _trade(trade_id="t2", entry_signal="", pnl=-50),
        ]
        builder.write_curated(
            trades=trades,
            missed=[],
            base_dir=tmp_path,
            strategy_registry=reg,
        )
        output = tmp_path / "2026-05-01" / "bot1" / "engine_decomposition.json"
        assert not output.exists()
