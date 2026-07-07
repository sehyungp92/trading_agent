"""Tests for regime integration: persistence, mapping tables, portfolio rules, and coordinators."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from regime.context import RegimeContext
from datetime import datetime, timezone

from regime.persistence import (
    RECOVERY_DEFAULT,
    load_regime_context,
    save_regime_context,
)
from regime.integration import (
    STOCK_RULES, MOMENTUM_RULES, SWING_RULES, DD_TIERS,
    OVERLAY_WEIGHTS, STOCK_PROFILES, MOMENTUM_PROFILES,
    build_stock_rules, build_momentum_rules, build_swing_rules,
)
from libs.oms.risk.portfolio_rules import PortfolioRulesConfig, PortfolioRuleChecker, PortfolioRuleResult


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_ctx(regime: str = "G", confidence: float = 0.9) -> RegimeContext:
    return RegimeContext(
        regime=regime,
        regime_confidence=confidence,
        stress_level=0.0,
        stress_onset=False,
        shift_velocity=0.0,
        suggested_leverage_mult=1.0,
        regime_allocations={"SPY": 0.3, "TLT": 0.3, "GLD": 0.2, "CASH": 0.2},
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def _base_stock_rules() -> PortfolioRulesConfig:
    return PortfolioRulesConfig(
        directional_cap_R=12.0,
        initial_equity=50_000.0,
        family_strategy_ids=("IARIC_v1", "ALCB_v1"),
        symbol_collision_action="half_size",
        priority_headroom_R=4.0,
        priority_reserve_threshold=1,
    )


def _base_momentum_rules() -> PortfolioRulesConfig:
    return PortfolioRulesConfig(
        directional_cap_R=4.25,
        directional_cap_long_R=10.0,
        directional_cap_short_R=10.5,
        initial_equity=10_000.0,
        max_family_contracts_mnq_eq=10,
        nqdtc_direction_filter_enabled=True,
        nqdtc_agree_size_mult=1.25,
        nqdtc_oppose_size_mult=0.50,
    )


def _base_swing_rules() -> PortfolioRulesConfig:
    return PortfolioRulesConfig(
        directional_cap_R=6.0,
        initial_equity=100_000.0,
        symbol_collision_action="half_size",
    )


# ── Persistence tests ────────────────────────────────────────────────

class TestPersistence:
    def test_round_trip(self, tmp_path: Path):
        ctx = _make_ctx("R", 0.85)
        path = tmp_path / "ctx.json"
        save_regime_context(ctx, path)
        loaded = load_regime_context(path)
        assert loaded.regime == "R"
        assert loaded.regime_confidence == 0.85
        assert loaded.computed_at == ctx.computed_at
        assert loaded.regime_allocations == ctx.regime_allocations

    def test_missing_file_returns_default(self, tmp_path: Path):
        path = tmp_path / "nonexistent.json"
        loaded = load_regime_context(path)
        assert loaded.regime == RECOVERY_DEFAULT.regime
        assert loaded.regime_confidence == RECOVERY_DEFAULT.regime_confidence

    def test_corrupt_file_returns_default(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("not json {{{", encoding="utf-8")
        loaded = load_regime_context(path)
        assert loaded.regime == RECOVERY_DEFAULT.regime

    def test_missing_key_returns_default(self, tmp_path: Path):
        path = tmp_path / "partial.json"
        path.write_text(json.dumps({"regime": "S"}), encoding="utf-8")
        loaded = load_regime_context(path)
        assert loaded.regime == RECOVERY_DEFAULT.regime

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "deep" / "nested" / "ctx.json"
        save_regime_context(_make_ctx(), path)
        assert path.exists()

    def test_computed_at_field_preserved(self):
        ctx = _make_ctx()
        assert ctx.computed_at is not None
        assert "T" in ctx.computed_at  # ISO format with time separator


# ── Integration mapping tests ────────────────────────────────────────

class TestMappingTables:
    @pytest.mark.parametrize("regime", ["G", "R", "S", "D"])
    def test_stock_rules_valid(self, regime: str):
        ctx = _make_ctx(regime)
        result = build_stock_rules(ctx, _base_stock_rules())
        assert result.directional_cap_R > 0
        assert 0 < result.regime_unit_risk_mult <= 1.0
        assert result.priority_headroom_R > 0
        assert result.symbol_collision_action in ("half_size", "block")
        assert result.dd_tiers == DD_TIERS[regime]

    @pytest.mark.parametrize("regime", ["G", "R", "S", "D"])
    def test_momentum_rules_valid(self, regime: str):
        ctx = _make_ctx(regime)
        result = build_momentum_rules(ctx, _base_momentum_rules(), 10)
        assert result.directional_cap_R > 0
        assert 0 < result.regime_unit_risk_mult <= 1.0
        assert result.max_family_contracts_mnq_eq > 0
        assert result.nqdtc_direction_filter_enabled is True
        assert result.dd_tiers == DD_TIERS[regime]

    @pytest.mark.parametrize("regime", ["G", "R", "S", "D"])
    def test_swing_rules_valid(self, regime: str):
        ctx = _make_ctx(regime)
        result = build_swing_rules(ctx, _base_swing_rules())
        assert result.directional_cap_R > 0
        assert 0 < result.regime_unit_risk_mult <= 1.0
        assert result.dd_tiers == DD_TIERS[regime]

    def test_stock_directional_cap_monotonically_decreases(self):
        caps = [STOCK_RULES[r]["directional_cap_R"] for r in ["G", "R", "S", "D"]]
        assert caps == sorted(caps, reverse=True)

    def test_momentum_directional_cap_monotonically_decreases(self):
        caps = [MOMENTUM_RULES[r]["directional_cap_R"] for r in ["G", "R", "S", "D"]]
        assert caps == sorted(caps, reverse=True)

    def test_swing_directional_cap_monotonically_decreases(self):
        caps = [SWING_RULES[r]["directional_cap_R"] for r in ["G", "R", "S", "D"]]
        assert caps == sorted(caps, reverse=True)

    def test_dd_tiers_stress_earlier_for_stress(self):
        g_first = DD_TIERS["G"][0][0]
        d_first = DD_TIERS["D"][0][0]
        assert d_first < g_first  # Defensive triggers earlier

    def test_overlay_weights_sum_to_one(self):
        for regime, weights in OVERLAY_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-9, f"Regime {regime}: weights sum to {total}"

    def test_overlay_defensive_no_qqq(self):
        assert OVERLAY_WEIGHTS["D"]["QQQ"] == 0.0
        assert OVERLAY_WEIGHTS["D"]["GLD"] == 1.0

    def test_stock_profiles_leave_live_stock_strategies_enabled_in_stress(self):
        assert len(STOCK_PROFILES["S"]["disabled"]) == 0
        assert len(STOCK_PROFILES["D"]["disabled"]) == 0
        assert len(STOCK_PROFILES["G"]["disabled"]) == 0

    def test_momentum_profiles_disable_downturn_in_growth(self):
        assert "DownturnDominator_v1" in MOMENTUM_PROFILES["G"]["disabled"]
        assert "DownturnDominator_v1" in MOMENTUM_PROFILES["R"]["disabled"]
        assert len(MOMENTUM_PROFILES["S"]["disabled"]) == 0

    def test_momentum_contracts_scale(self):
        ctx_d = _make_ctx("D")
        result = build_momentum_rules(ctx_d, _base_momentum_rules(), 10)
        assert result.max_family_contracts_mnq_eq == 5  # 10 * 0.5

    def test_initial_equity_preserved(self):
        """Verify build functions preserve initial_equity from base config."""
        base = _base_stock_rules()
        result = build_stock_rules(_make_ctx("D"), base)
        assert result.initial_equity == base.initial_equity

    def test_regime_scales_supplied_base_instead_of_replacing_with_table(self):
        base = dataclasses.replace(
            _base_stock_rules(),
            directional_cap_R=20.0,
            directional_cap_long_R=18.0,
            priority_headroom_R=2.0,
            dd_tiers=((0.10, 1.0), (0.15, 0.6), (0.20, 0.3), (1.0, 0.0)),
        )

        result = build_stock_rules(_make_ctx("D"), base)

        assert result.directional_cap_R == pytest.approx(11.0)
        assert result.directional_cap_long_R == pytest.approx(9.9)
        assert result.priority_headroom_R == pytest.approx(1.1)
        assert result.dd_tiers == ((0.07, 1.0), (0.105, 0.6), (0.14, 0.3), (1.0, 0.0))

    def test_context_uses_final_leverage_without_second_stress_cut(self):
        ctx = _make_ctx("S")
        ctx = dataclasses.replace(ctx, stress_level=0.50, stress_onset=True, suggested_leverage_mult=1.0)

        result = build_swing_rules(ctx, _base_swing_rules())

        assert result.regime_unit_risk_mult == pytest.approx(0.80)

    def test_context_suggested_leverage_still_dampens_regime_unit_risk(self):
        ctx = dataclasses.replace(_make_ctx("S"), suggested_leverage_mult=0.95)

        result = build_swing_rules(ctx, _base_swing_rules())

        assert result.regime_unit_risk_mult == pytest.approx(0.76)


# ── Portfolio rules regime checks ─────────────────────────────────────

class TestPortfolioRulesRegime:
    @pytest.fixture
    def checker(self):
        cfg = PortfolioRulesConfig(
            directional_cap_R=8.0,
            initial_equity=10_000.0,
            regime_unit_risk_mult=1.0,
            disabled_strategies=frozenset(),
            nqdtc_direction_filter_enabled=False,
        )
        return PortfolioRuleChecker(
            config=cfg,
            get_strategy_signal=AsyncMock(return_value=None),
            get_directional_risk_R=AsyncMock(return_value=0.0),
            get_current_equity=lambda: 10_000.0,
        )

    def test_disabled_strategy_denied(self, checker: PortfolioRuleChecker):
        new_cfg = dataclasses.replace(
            checker._cfg,
            disabled_strategies=frozenset({"ALCB_v1"}),
        )
        checker.update_config(new_cfg)
        result = asyncio.run(checker.check_entry("ALCB_v1", "LONG", 1.0))
        assert not result.approved
        assert "regime_disabled" in result.denial_reason

    def test_non_disabled_strategy_allowed(self, checker: PortfolioRuleChecker):
        new_cfg = dataclasses.replace(
            checker._cfg,
            disabled_strategies=frozenset({"ALCB_v1"}),
        )
        checker.update_config(new_cfg)
        result = asyncio.run(checker.check_entry("IARIC_v1", "LONG", 1.0))
        assert result.approved

    def test_regime_unit_risk_mult_applied(self, checker: PortfolioRuleChecker):
        new_cfg = dataclasses.replace(checker._cfg, regime_unit_risk_mult=0.5)
        checker.update_config(new_cfg)
        result = asyncio.run(checker.check_entry("ALCB_v1", "LONG", 1.0))
        assert result.approved
        assert result.size_multiplier == pytest.approx(0.5)

    def test_directional_unit_risk_mult_applied(self, checker: PortfolioRuleChecker):
        new_cfg = dataclasses.replace(
            checker._cfg,
            regime_unit_risk_mult=0.8,
            regime_unit_risk_long_mult=0.5,
            regime_unit_risk_short_mult=1.0,
        )
        checker.update_config(new_cfg)

        long_result = asyncio.run(checker.check_entry("ALCB_v1", "LONG", 1.0))
        short_result = asyncio.run(checker.check_entry("ALCB_v1", "SHORT", 1.0))

        assert long_result.approved
        assert short_result.approved
        assert long_result.size_multiplier == pytest.approx(0.4)
        assert short_result.size_multiplier == pytest.approx(0.8)

    def test_regime_mult_compounds_with_dd(self, checker: PortfolioRuleChecker):
        """Defensive (0.5x) at 10% DD (0.5x) = 0.25x final multiplier."""
        new_cfg = dataclasses.replace(
            checker._cfg,
            regime_unit_risk_mult=0.5,
            dd_tiers=((0.08, 1.0), (0.12, 0.50), (0.15, 0.25), (1.00, 0.00)),
        )
        checker_dd = PortfolioRuleChecker(
            config=new_cfg,
            get_strategy_signal=AsyncMock(return_value=None),
            get_directional_risk_R=AsyncMock(return_value=0.0),
            get_current_equity=lambda: 9_000.0,  # 10% DD
        )
        result = asyncio.run(checker_dd.check_entry("ALCB_v1", "LONG", 1.0))
        assert result.approved
        assert result.size_multiplier == pytest.approx(0.25)  # 0.50 * 0.50

    def test_update_config_atomic(self, checker: PortfolioRuleChecker):
        """Verify update_config replaces the entire config object."""
        old_cap = checker._cfg.directional_cap_R
        new_cfg = dataclasses.replace(checker._cfg, directional_cap_R=4.0)
        checker.update_config(new_cfg)
        assert checker._cfg.directional_cap_R == 4.0
        assert checker._cfg.directional_cap_R != old_cap
