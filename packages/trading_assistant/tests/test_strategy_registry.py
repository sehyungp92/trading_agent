"""Tests for strategy registry loading and querying."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trading_assistant.orchestrator.strategy_registry_loader import load_strategy_registry
from trading_assistant.paths import package_root
from trading_assistant.schemas.strategy_profile import (
    StrategyArchetype,
    StrategyRegistry,
)


@pytest.fixture()
def yaml_path(tmp_path: Path) -> Path:
    """Write a minimal strategy_profiles.yaml for testing."""
    data = {
        "strategies": {
            "ATRSS": {
                "display_name": "ATR Swing Strategy",
                "bot_id": "swing_multi_01",
                "family": "swing",
                "archetype": "trend_follow",
                "asset_class": "mixed",
                "symbols": ["QQQ", "GLD"],
                "preferred_regimes": ["trending_up", "trending_down"],
                "adverse_regimes": ["ranging"],
                "holding_period": "multi_day",
                "risk": {"unit_risk_dollars": 200.0, "daily_stop_R": 2.0, "max_heat_R": 1.5, "tp_ratio": 2.0},
                "allocation": {"base_risk_pct": 0.018},
            },
            "AKC_HELIX": {
                "display_name": "AKC Helix",
                "bot_id": "swing_multi_01",
                "family": "swing",
                "archetype": "divergence_swing",
                "asset_class": "mixed",
                "symbols": ["QQQ"],
                "preferred_regimes": ["trending_up"],
                "adverse_regimes": ["ranging"],
                "holding_period": "multi_day",
            },
            "IARIC_v1": {
                "display_name": "IARIC",
                "bot_id": "stock_trader",
                "family": "stock",
                "archetype": "intraday_momentum",
                "asset_class": "equity",
                "symbols": [],
                "preferred_regimes": ["trending_up", "volatile"],
                "adverse_regimes": ["ranging"],
                "holding_period": "intraday",
            },
            "AKC_Helix_v40": {
                "display_name": "AKC Helix Momentum",
                "bot_id": "momentum_nq_01",
                "family": "momentum",
                "archetype": "multi_tf_momentum",
                "asset_class": "futures",
                "symbols": ["NQ"],
                "preferred_regimes": ["trending_up", "trending_down"],
                "adverse_regimes": ["ranging"],
                "holding_period": "intraday",
            },
        },
        "coordination": {
            "signals": [
                {
                    "trigger": {"strategy": "ATRSS", "event": "ENTRY_FILL"},
                    "target": {"strategy": "AKC_HELIX", "action": "TIGHTEN_STOP_BE"},
                    "condition": "SAME_SYMBOL",
                },
            ],
            "cooldown_pairs": [
                {"strategies": ["AKC_Helix_v40", "NQDTC_v2.1"], "minutes": 120},
            ],
            "direction_filter": {
                "observer": "VdubusNQ_v4",
                "reference": "NQDTC_v2.1",
                "agree_mult": 1.5,
                "oppose_mult": 0.0,
            },
            "stock_coordination": {
                "directional_cap_R": 8.0,
                "symbol_collision_action": "half_size",
                "strategies": ["IARIC_v1"],
            },
        },
        "portfolio": {
            "heat_cap_R": 2.5,
            "portfolio_daily_stop_R": 3.0,
            "portfolio_weekly_stop_R": 5.0,
            "family_allocations": {"swing": 0.334, "stock": 0.333, "momentum": 0.333},
            "drawdown_tiers": [[0.08, 1.0], [0.12, 0.5]],
        },
        "archetype_expectations": {
            "trend_follow": {
                "expected_win_rate": [0.35, 0.50],
                "expected_payoff_ratio": [1.8, 3.0],
                "regime_sensitivity": "high",
                "typical_holding_bars": [20, 200],
            },
            "intraday_momentum": {
                "expected_win_rate": [0.40, 0.55],
                "expected_payoff_ratio": [1.2, 2.0],
                "regime_sensitivity": "medium",
                "typical_holding_bars": [1, 20],
            },
        },
    }
    path = tmp_path / "strategy_profiles.yaml"
    path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    return path


@pytest.fixture()
def registry(yaml_path: Path) -> StrategyRegistry:
    return load_strategy_registry(yaml_path)


class TestStrategyRegistryLoading:
    def test_loads_all_strategies(self, registry: StrategyRegistry) -> None:
        assert len(registry.strategies) == 4

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        reg = load_strategy_registry(tmp_path / "nope.yaml")
        assert reg.strategies == {}
        assert reg.coordination.signals == []

    def test_malformed_yaml_returns_empty(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("{{{{not yaml", encoding="utf-8")
        reg = load_strategy_registry(bad)
        assert reg.strategies == {}

    def test_invalid_schema_returns_empty(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.dump({"strategies": "not_a_dict"}), encoding="utf-8")
        reg = load_strategy_registry(bad)
        assert reg.strategies == {}

    def test_strategy_profile_fields(self, registry: StrategyRegistry) -> None:
        atrss = registry.strategies["ATRSS"]
        assert atrss.bot_id == "swing_multi_01"
        assert atrss.family == "swing"
        assert atrss.archetype == StrategyArchetype.TREND_FOLLOW
        assert atrss.risk.tp_ratio == 2.0
        assert atrss.allocation.base_risk_pct == 0.018


class TestStrategyRegistryQueries:
    def test_strategies_for_bot_swing(self, registry: StrategyRegistry) -> None:
        swing = registry.strategies_for_bot("swing_multi_01")
        assert len(swing) == 2
        assert "ATRSS" in swing
        assert "AKC_HELIX" in swing

    def test_strategies_for_bot_stock(self, registry: StrategyRegistry) -> None:
        stock = registry.strategies_for_bot("stock_trader")
        assert len(stock) == 1
        assert "IARIC_v1" in stock

    def test_strategies_for_bot_momentum(self, registry: StrategyRegistry) -> None:
        momentum = registry.strategies_for_bot("momentum_nq_01")
        assert len(momentum) == 1
        assert "AKC_Helix_v40" in momentum

    def test_strategies_for_unknown_bot(self, registry: StrategyRegistry) -> None:
        assert registry.strategies_for_bot("nonexistent") == {}

    def test_strategies_in_family(self, registry: StrategyRegistry) -> None:
        swing = registry.strategies_in_family("swing")
        assert len(swing) == 2

    def test_archetype_for_strategy(self, registry: StrategyRegistry) -> None:
        assert registry.archetype_for_strategy("ATRSS") == StrategyArchetype.TREND_FOLLOW
        assert registry.archetype_for_strategy("IARIC_v1") == StrategyArchetype.INTRADAY_MOMENTUM

    def test_archetype_for_unknown_strategy(self, registry: StrategyRegistry) -> None:
        assert registry.archetype_for_strategy("UNKNOWN") is None

    def test_expectations_for_archetype(self, registry: StrategyRegistry) -> None:
        exp = registry.expectations_for_archetype("trend_follow")
        assert exp is not None
        assert exp.expected_win_rate == (0.35, 0.50)
        assert exp.regime_sensitivity == "high"

    def test_expectations_for_unknown_archetype(self, registry: StrategyRegistry) -> None:
        assert registry.expectations_for_archetype("unknown") is None


class TestCoordinationParsing:
    def test_coordination_signals(self, registry: StrategyRegistry) -> None:
        assert len(registry.coordination.signals) == 1
        sig = registry.coordination.signals[0]
        assert sig.trigger["strategy"] == "ATRSS"
        assert sig.condition == "SAME_SYMBOL"

    def test_cooldown_pairs(self, registry: StrategyRegistry) -> None:
        assert len(registry.coordination.cooldown_pairs) == 1
        pair = registry.coordination.cooldown_pairs[0]
        assert pair.minutes == 120

    def test_direction_filter(self, registry: StrategyRegistry) -> None:
        df = registry.coordination.direction_filter
        assert df is not None
        assert df.observer == "VdubusNQ_v4"
        assert df.agree_mult == 1.5

    def test_stock_coordination(self, registry: StrategyRegistry) -> None:
        sc = registry.coordination.stock_coordination
        assert sc is not None
        assert sc.directional_cap_R == 8.0


class TestPortfolioParsing:
    def test_portfolio_config(self, registry: StrategyRegistry) -> None:
        p = registry.portfolio
        assert p.heat_cap_R == 2.5
        assert p.portfolio_daily_stop_R == 3.0
        assert len(p.drawdown_tiers) == 2

    def test_family_allocations(self, registry: StrategyRegistry) -> None:
        alloc = registry.portfolio.family_allocations
        assert alloc["swing"] == pytest.approx(0.334)


_REAL_YAML = package_root() / "data" / "strategy_profiles.yaml"


class TestFullYAMLFile:
    """Test loading the actual data/strategy_profiles.yaml file."""

    @pytest.fixture()
    def full_registry(self) -> StrategyRegistry:
        if not _REAL_YAML.exists():
            pytest.skip("strategy_profiles.yaml not present")
        return load_strategy_registry(_REAL_YAML)

    def test_loads_all_strategies(self, full_registry: StrategyRegistry) -> None:
        # Reference monorepo deletions (breakout, keltner, helix_v40, etc.)
        # and additions (tpc, nq_regime) reduced the count from 20 to 16
        # in the optimisation commit.
        assert len(full_registry.strategies) == 16

    def test_swing_bot_has_3_strategies(self, full_registry: StrategyRegistry) -> None:
        # ATRSS, AKC_HELIX, TPC. Earlier 6-strategy roster lost
        # SWING_BREAKOUT_V3, S5_PB, S5_DUAL and gained TPC.
        assert len(full_registry.strategies_for_bot("swing_multi_01")) == 3

    def test_momentum_bot_has_4_strategies(self, full_registry: StrategyRegistry) -> None:
        assert len(full_registry.strategies_for_bot("momentum_nq_01")) == 4

    def test_stock_bot_has_2_strategies(self, full_registry: StrategyRegistry) -> None:
        # IARIC_v1, ALCB_v1.
        assert len(full_registry.strategies_for_bot("stock_trader")) == 2

    def test_k_stock_bot_has_4_strategies(self, full_registry: StrategyRegistry) -> None:
        assert len(full_registry.strategies_for_bot("k_stock_trader")) == 4

    def test_tpc_archetype_resolves(self, full_registry: StrategyRegistry) -> None:
        # TPC replaced the deleted breakout/S5 swing entries.
        assert full_registry.archetype_for_strategy("TPC") == StrategyArchetype.PULLBACK

    def test_downturn_archetype_resolves(self, full_registry: StrategyRegistry) -> None:
        assert full_registry.archetype_for_strategy("DownturnDominator_v1") == StrategyArchetype.MULTI_ENGINE_BEAR

    def test_iaric_archetype_updated(self, full_registry: StrategyRegistry) -> None:
        assert full_registry.archetype_for_strategy("IARIC_v1") == StrategyArchetype.MEAN_REVERSION_PULLBACK

    def test_new_archetype_expectations(self, full_registry: StrategyRegistry) -> None:
        for arch in ("bear_regime_swing", "multi_engine_bear", "mean_reversion_pullback"):
            exp = full_registry.expectations_for_archetype(arch)
            assert exp is not None, f"Missing archetype_expectations for {arch}"

    def test_profile_metadata_fields(self, full_registry: StrategyRegistry) -> None:
        # TPC is the new pullback strategy that took over from the deleted
        # breakout/S5 entries; verify its registered entry_types and exit
        # profile loaded cleanly.
        tpc = full_registry.strategies["TPC"]
        assert len(tpc.entry_types) >= 5
        assert "TYPE_A" in tpc.entry_types
        assert tpc.exit_profile is not None
        assert any(t.tier_name == "T1" for t in tpc.exit_profile.tiers)

        downturn = full_registry.strategies["DownturnDominator_v1"]
        assert downturn.sub_engines == ["REVERSAL", "BREAKDOWN", "FADE"]
        assert downturn.regime_model == "composite_with_vol_state"
        assert len(downturn.key_metadata_fields) == 6

    def test_no_signals_reference_retired_strategies(
        self, full_registry: StrategyRegistry
    ) -> None:
        """Coordination signals must not reference strategies that have been
        deleted upstream (e.g. AKC_Helix_v40)."""
        active_ids = set(full_registry.strategies)
        for sig in full_registry.coordination.signals:
            for side in (sig.trigger, sig.target):
                sid = side.get("strategy")
                assert not sid or sid in active_ids, (
                    f"signal references retired strategy {sid!r}"
                )

    def test_no_cooldown_pairs_reference_retired_strategies(
        self, full_registry: StrategyRegistry
    ) -> None:
        """Cooldown pairs must only contain currently-registered strategies."""
        active_ids = set(full_registry.strategies)
        for pair in full_registry.coordination.cooldown_pairs:
            for sid in pair.strategies:
                assert sid in active_ids, (
                    f"cooldown_pair references retired strategy {sid!r}"
                )
