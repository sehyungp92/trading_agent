"""Tests for stock family portfolio rules integration.

Validates:
  - Drawdown tiers reduce stock sizing
  - Directional cap with family-scoped query
  - Symbol collision blocks/reduces when sibling holds same symbol
  - Momentum-specific checks are no-ops for stock strategy IDs
  - check_entry symbol param is backward-compatible (Optional, defaults to None)
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock

import pytest

from libs.oms.config.risk_config import RiskConfig, StrategyRiskConfig
from libs.oms.models.instrument import Instrument
from libs.oms.models.order import (
    OMSOrder,
    OrderRole,
    OrderSide,
    OrderType,
    RiskContext,
)
from libs.oms.models.risk_state import PortfolioRiskState, StrategyRiskState
from libs.oms.risk.calendar import EventCalendar
from libs.oms.risk.gateway import RiskGateway
from libs.oms.risk.portfolio_rules import (
    PortfolioRuleChecker,
    PortfolioRuleResult,
    PortfolioRulesConfig,
)

STOCK_IDS = ("IARIC_v1", "ALCB_v1")


def _make_checker(
    *,
    equity: float = 10_000.0,
    initial_equity: float = 10_000.0,
    directional_cap_R: float = 8.0,
    family_strategy_ids: tuple[str, ...] = STOCK_IDS,
    symbol_collision_action: str = "none",
    dir_risk_global: float = 0.0,
    dir_risk_family: float = 0.0,
    sibling_holds_symbol: bool = False,
) -> PortfolioRuleChecker:
    config = PortfolioRulesConfig(
        directional_cap_R=directional_cap_R,
        initial_equity=initial_equity,
        family_strategy_ids=family_strategy_ids,
        symbol_collision_action=symbol_collision_action,
    )
    return PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=dir_risk_global),
        get_current_equity=lambda: equity,
        get_directional_risk_R_for_strategies=AsyncMock(return_value=dir_risk_family),
        get_sibling_positions_for_symbol=AsyncMock(return_value=sibling_holds_symbol),
    )


# ── Drawdown tiers ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_drawdown_tier_full_size_when_no_drawdown():
    checker = _make_checker(equity=10_000.0, initial_equity=10_000.0)
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0)
    assert result.approved
    assert result.size_multiplier == 1.0


@pytest.mark.asyncio
async def test_drawdown_tier_half_size_at_10pct_dd():
    # 10% drawdown → falls in [0.08, 0.12) tier → 0.50 multiplier
    checker = _make_checker(equity=9_000.0, initial_equity=10_000.0)
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0)
    assert result.approved
    assert result.size_multiplier == 0.5


@pytest.mark.asyncio
async def test_drawdown_tier_quarter_size_at_13pct_dd():
    # 13% drawdown → falls in [0.12, 0.15) tier → 0.25 multiplier
    checker = _make_checker(equity=8_700.0, initial_equity=10_000.0)
    result = await checker.check_entry("ALCB_v1", "LONG", 1.0)
    assert result.approved
    assert result.size_multiplier == 0.25


@pytest.mark.asyncio
async def test_drawdown_tier_halt_at_16pct_dd():
    # 16% drawdown → beyond 0.15 tier → halt (0.0)
    checker = _make_checker(equity=8_400.0, initial_equity=10_000.0)
    result = await checker.check_entry("ALCB_v1", "LONG", 1.0)
    assert not result.approved
    assert "drawdown_halt" in result.denial_reason


# ── Family-scoped directional cap ───────────────────────────────────


@pytest.mark.asyncio
async def test_directional_cap_uses_family_scoped_query():
    """When family_strategy_ids is set, directional cap uses the family-scoped callback."""
    checker = _make_checker(dir_risk_family=7.0, dir_risk_global=20.0)
    # 7R family + 1R new = 8R ≤ 8R cap → approved
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0)
    assert result.approved


@pytest.mark.asyncio
async def test_directional_cap_blocks_when_family_exceeds():
    checker = _make_checker(dir_risk_family=7.5)
    # 7.5R family + 1R new = 8.5R > 8R cap → blocked
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0)
    assert not result.approved
    assert "directional_cap" in result.denial_reason


@pytest.mark.asyncio
async def test_directional_cap_disabled_when_zero():
    checker = _make_checker(directional_cap_R=0, dir_risk_family=100.0)
    result = await checker.check_entry("ALCB_v1", "LONG", 1.0)
    assert result.approved


@pytest.mark.asyncio
async def test_directional_cap_falls_back_to_global_without_family_ids():
    """Without family_strategy_ids, falls back to global directional risk query."""
    checker = _make_checker(
        family_strategy_ids=(),
        dir_risk_global=9.0,
    )
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0)
    assert not result.approved
    assert "directional_cap" in result.denial_reason


@pytest.mark.asyncio
async def test_max_total_active_positions_blocks_family_capacity():
    config = PortfolioRulesConfig(
        directional_cap_R=0,
        family_strategy_ids=STOCK_IDS,
        max_total_active_positions=2,
    )
    checker = PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=0.0),
        get_current_equity=lambda: 10_000.0,
        get_open_position_count_for_strategies=AsyncMock(return_value=2),
    )

    result = await checker.check_entry("IARIC_v1", "LONG", 1.0, symbol="AAPL")

    assert not result.approved
    assert "max_total_active_positions" in result.denial_reason


@pytest.mark.asyncio
async def test_symbol_heat_cap_uses_reference_risk_dollars():
    config = PortfolioRulesConfig(
        directional_cap_R=0,
        family_strategy_ids=STOCK_IDS,
        reference_unit_risk_dollars=100.0,
        max_symbol_heat_R=2.2,
    )
    checker = PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=0.0),
        get_current_equity=lambda: 10_000.0,
        get_symbol_open_risk_dollars_for_strategies=AsyncMock(return_value=150.0),
    )

    result = await checker.check_entry(
        "IARIC_v1", "LONG", 0.8, symbol="AAPL", new_risk_dollars=80.0,
    )

    assert not result.approved
    assert "symbol_heat_cap" in result.denial_reason


@pytest.mark.asyncio
async def test_reference_risk_pct_tracks_current_equity_for_heat_caps():
    equity = [10_000.0]
    config = PortfolioRulesConfig(
        directional_cap_R=0,
        family_strategy_ids=STOCK_IDS,
        reference_unit_risk_dollars=100.0,
        reference_unit_risk_pct=0.01,
        portfolio_heat_cap_R=2.0,
    )
    checker = PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=0.0),
        get_current_equity=lambda: equity[0],
        get_active_risk_dollars_for_strategies=AsyncMock(return_value=150.0),
    )

    blocked = await checker.check_entry(
        "IARIC_v1", "LONG", 0.8, symbol="AAPL", new_risk_dollars=80.0,
    )
    equity[0] = 20_000.0
    approved = await checker.check_entry(
        "IARIC_v1", "LONG", 0.8, symbol="AAPL", new_risk_dollars=80.0,
    )

    assert not blocked.approved
    assert "portfolio_heat_cap" in blocked.denial_reason
    assert approved.approved


@pytest.mark.asyncio
async def test_sector_heat_cap_uses_symbol_sector_map():
    sector_risk = AsyncMock(return_value=300.0)
    config = PortfolioRulesConfig(
        directional_cap_R=0,
        family_strategy_ids=STOCK_IDS,
        reference_unit_risk_dollars=100.0,
        same_sector_heat_cap_R=3.8,
        symbol_sector_map=(("AAPL", "Technology"), ("MSFT", "Technology")),
    )
    checker = PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=0.0),
        get_current_equity=lambda: 10_000.0,
        get_symbols_open_risk_dollars_for_strategies=sector_risk,
    )

    result = await checker.check_entry(
        "ALCB_v1", "LONG", 0.9, symbol="AAPL", new_risk_dollars=90.0,
    )

    assert not result.approved
    assert "sector_heat_cap" in result.denial_reason
    assert set(sector_risk.await_args.args[1]) == {"AAPL", "MSFT"}


@pytest.mark.asyncio
async def test_size_multipliers_apply_before_directional_cap():
    config = PortfolioRulesConfig(
        directional_cap_R=1.0,
        family_strategy_ids=STOCK_IDS,
        reference_unit_risk_dollars=100.0,
        initial_equity=10_000.0,
        dd_tiers=((0.10, 0.50), (1.00, 0.00)),
    )
    checker = PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=0.0),
        get_current_equity=lambda: 9_500.0,
        get_directional_risk_dollars_for_strategies=AsyncMock(return_value=60.0),
    )

    result = await checker.check_entry(
        "ALCB_v1", "LONG", 0.8, symbol="AAPL", new_risk_dollars=80.0,
    )

    assert result.approved
    assert result.size_multiplier == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_dynamic_allocation_boosts_recent_positive_expectancy():
    config = PortfolioRulesConfig(
        directional_cap_R=0,
        family_strategy_ids=STOCK_IDS,
        dynamic_allocation_enabled=True,
        dynamic_lookback_trades=60,
        dynamic_positive_expectancy_boost=0.10,
        dynamic_max_mult=1.22,
    )
    checker = PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=0.0),
        get_current_equity=lambda: 10_000.0,
        get_recent_strategy_r_multiples=AsyncMock(return_value=[0.4] * 40 + [-0.1] * 10),
    )

    result = await checker.check_entry("IARIC_v1", "LONG", 1.0, symbol="AAPL")

    assert result.approved
    assert result.size_multiplier == pytest.approx(1.10)


@pytest.mark.asyncio
async def test_strategy_active_positions_blocks_per_strategy_capacity():
    async def open_count(ids: list[str]) -> int:
        return 3 if ids == ["NQ_REGIME"] else 0

    config = PortfolioRulesConfig(
        directional_cap_R=0,
        family_strategy_ids=("NQ_REGIME", "VdubusNQ_v4"),
        max_strategy_active_positions=(("NQ_REGIME", 3), ("VdubusNQ_v4", 2)),
    )
    checker = PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=0.0),
        get_current_equity=lambda: 10_000.0,
        get_open_position_count_for_strategies=open_count,
    )

    result = await checker.check_entry("NQ_REGIME", "LONG", 1.0, symbol="MNQ")

    assert not result.approved
    assert "max_strategy_active_positions" in result.denial_reason


@pytest.mark.asyncio
async def test_momentum_dynamic_sizing_fits_remaining_capacity_before_caps():
    config = PortfolioRulesConfig(
        directional_cap_R=2.0,
        family_strategy_ids=("NQ_REGIME", "VdubusNQ_v4"),
        reference_unit_risk_dollars=100.0,
        portfolio_heat_cap_R=2.0,
        strategy_size_multipliers=(("NQ_REGIME", 0.75),),
        max_trade_risk_R=2.0,
        fit_to_remaining_heat=True,
        fit_to_remaining_directional_cap=True,
    )
    checker = PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=0.0),
        get_current_equity=lambda: 10_000.0,
        get_directional_risk_dollars_for_strategies=AsyncMock(return_value=100.0),
        get_active_risk_dollars_for_strategies=AsyncMock(return_value=50.0),
    )

    result = await checker.check_entry(
        "NQ_REGIME", "LONG", 4.0, symbol="MNQ", new_qty=4, new_risk_dollars=400.0,
    )

    assert result.approved
    assert result.size_multiplier == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_momentum_dynamic_sizing_blocks_when_capacity_floor_has_no_room():
    config = PortfolioRulesConfig(
        directional_cap_R=0,
        family_strategy_ids=("NQ_REGIME", "VdubusNQ_v4"),
        reference_unit_risk_dollars=100.0,
        portfolio_heat_cap_R=2.0,
        fit_to_remaining_heat=True,
    )
    checker = PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=0.0),
        get_current_equity=lambda: 10_000.0,
        get_active_risk_dollars_for_strategies=AsyncMock(return_value=200.0),
    )

    result = await checker.check_entry(
        "NQ_REGIME", "LONG", 1.0, symbol="MNQ", new_qty=1, new_risk_dollars=100.0,
    )

    assert not result.approved
    assert result.denial_reason == "dynamic_capacity_floor"


@pytest.mark.asyncio
async def test_risk_gateway_applies_portfolio_multiplier_before_heat_cap():
    class HalfSizeChecker:
        async def check_entry(self, **kwargs):
            return PortfolioRuleResult(approved=True, size_multiplier=0.5)

    gateway = RiskGateway(
        config=RiskConfig(
            heat_cap_R=2.0,
            portfolio_daily_stop_R=3.0,
            portfolio_weekly_stop_R=9.0,
            portfolio_urd=100.0,
            strategy_configs={
                "NQ_REGIME": StrategyRiskConfig(
                    strategy_id="NQ_REGIME",
                    unit_risk_dollars=100.0,
                    daily_stop_R=3.0,
                )
            },
        ),
        calendar=EventCalendar(),
        get_strategy_risk=AsyncMock(
            return_value=StrategyRiskState("NQ_REGIME", date.today())
        ),
        get_portfolio_risk=AsyncMock(
            return_value=PortfolioRiskState(
                date.today(),
                open_risk_R=1.0,
            )
        ),
        portfolio_checker=HalfSizeChecker(),
    )
    order = OMSOrder(
        strategy_id="NQ_REGIME",
        instrument=Instrument(
            symbol="MNQ",
            root="MNQ",
            venue="CME",
            tick_size=0.25,
            tick_value=0.5,
            multiplier=2.0,
            point_value=2.0,
        ),
        side=OrderSide.BUY,
        qty=2,
        order_type=OrderType.LIMIT,
        role=OrderRole.ENTRY,
        risk_context=RiskContext(planned_entry_price=100.0, stop_for_risk=50.0),
    )

    denial = await gateway.check_entry(order)

    assert denial is None
    assert order.risk_context.portfolio_size_mult == pytest.approx(0.5)
    assert order.risk_context.risk_dollars == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_risk_gateway_threads_live_correlation_context_to_portfolio_checker():
    class CapturingChecker:
        def __init__(self):
            self.kwargs = {}

        async def check_entry(self, **kwargs):
            self.kwargs = kwargs
            return PortfolioRuleResult(approved=True)

    checker = CapturingChecker()
    gateway = RiskGateway(
        config=RiskConfig(
            heat_cap_R=5.0,
            portfolio_urd=100.0,
            strategy_configs={
                "IARIC_v1": StrategyRiskConfig(
                    strategy_id="IARIC_v1",
                    unit_risk_dollars=100.0,
                    daily_stop_R=3.0,
                )
            },
        ),
        calendar=EventCalendar(),
        get_strategy_risk=AsyncMock(return_value=StrategyRiskState("IARIC_v1", date.today())),
        get_portfolio_risk=AsyncMock(return_value=PortfolioRiskState(date.today())),
        portfolio_checker=checker,
    )
    exchange_ts = datetime(2026, 6, 3, 14, 30, tzinfo=timezone.utc)
    risk_context = RiskContext(planned_entry_price=100.0, stop_for_risk=95.0)
    risk_context.trace_id = "trace_live"
    risk_context.signal_id = "sig_live"
    risk_context.bar_id = "bar_live"
    risk_context.exchange_timestamp = exchange_ts
    risk_context.lineage_context = {"deployment_id": "dep_live"}
    order = OMSOrder(
        strategy_id="IARIC_v1",
        instrument=Instrument(
            symbol="AAPL",
            root="AAPL",
            venue="SMART",
            tick_size=0.01,
            tick_value=0.01,
            multiplier=1.0,
            point_value=1.0,
        ),
        side=OrderSide.BUY,
        qty=10,
        order_type=OrderType.LIMIT,
        role=OrderRole.ENTRY,
        risk_context=risk_context,
    )

    denial = await gateway.check_entry(order)

    assert denial is None
    assert checker.kwargs["trace_id"] == "trace_live"
    assert checker.kwargs["signal_id"] == "sig_live"
    assert checker.kwargs["bar_id"] == "bar_live"
    assert checker.kwargs["exchange_timestamp"] == exchange_ts
    assert checker.kwargs["lineage_context"] == {"deployment_id": "dep_live"}


@pytest.mark.asyncio
async def test_strategy_trade_share_cap_blocks_after_min_sample():
    config = PortfolioRulesConfig(
        directional_cap_R=0,
        family_strategy_ids=STOCK_IDS,
        max_single_strategy_trade_share=0.85,
        strategy_trade_share_min_total=50,
    )
    checker = PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=0.0),
        get_current_equity=lambda: 10_000.0,
        get_completed_trade_counts_for_strategies=AsyncMock(
            return_value={"IARIC_v1": 50, "ALCB_v1": 8}
        ),
    )

    result = await checker.check_entry("IARIC_v1", "LONG", 1.0, symbol="AAPL")

    assert not result.approved
    assert "strategy_trade_share_cap" in result.denial_reason


# ── Symbol collision ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_symbol_collision_block_mode():
    checker = _make_checker(
        symbol_collision_action="block",
        sibling_holds_symbol=True,
    )
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0, symbol="AAPL")
    assert not result.approved
    assert "symbol_collision" in result.denial_reason


@pytest.mark.asyncio
async def test_symbol_collision_half_size_mode():
    checker = _make_checker(
        symbol_collision_action="half_size",
        sibling_holds_symbol=True,
    )
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0, symbol="AAPL")
    assert result.approved
    assert result.size_multiplier == 0.5


@pytest.mark.asyncio
async def test_symbol_collision_no_collision():
    checker = _make_checker(
        symbol_collision_action="block",
        sibling_holds_symbol=False,
    )
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0, symbol="AAPL")
    assert result.approved
    assert result.size_multiplier == 1.0


@pytest.mark.asyncio
async def test_symbol_collision_ignored_when_action_none():
    checker = _make_checker(
        symbol_collision_action="none",
        sibling_holds_symbol=True,
    )
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0, symbol="AAPL")
    assert result.approved


@pytest.mark.asyncio
async def test_symbol_collision_ignored_without_symbol():
    checker = _make_checker(
        symbol_collision_action="block",
        sibling_holds_symbol=True,
    )
    # No symbol passed → skip collision check
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0)
    assert result.approved


# ── Momentum checks are no-ops for stock IDs ───────────────────────




@pytest.mark.asyncio
async def test_direction_filter_noop_for_stock():
    """Direction filter only fires for VdubusNQ — stock IDs pass through."""
    checker = _make_checker()
    result = await checker.check_entry("ALCB_v1", "SHORT", 1.0)
    assert result.approved




# ── Backward compatibility ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_check_entry_without_symbol_param():
    """check_entry works without symbol param (backward compat with momentum gateway)."""
    checker = _make_checker()
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0)
    assert result.approved


@pytest.mark.asyncio
async def test_combined_drawdown_and_collision_multipliers():
    """Drawdown tier and symbol collision multipliers compound."""
    checker = _make_checker(
        equity=9_000.0,
        initial_equity=10_000.0,  # 10% DD → 0.5x
        symbol_collision_action="half_size",
        sibling_holds_symbol=True,  # → 0.5x
    )
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0, symbol="AAPL")
    assert result.approved
    assert result.size_multiplier == pytest.approx(0.25)  # 0.5 * 0.5


# ── Robustness / edge cases ───────────────────────────────────────


@pytest.mark.asyncio
async def test_symbol_collision_excludes_requesting_strategy():
    """Sibling query must exclude the requesting strategy from the lookup."""
    sibling_mock = AsyncMock(return_value=True)
    config = PortfolioRulesConfig(
        directional_cap_R=0,
        family_strategy_ids=STOCK_IDS,
        symbol_collision_action="block",
    )
    checker = PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=0.0),
        get_current_equity=lambda: 10_000.0,
        get_sibling_positions_for_symbol=sibling_mock,
    )
    await checker.check_entry("ALCB_v1", "LONG", 1.0, symbol="TSLA")
    # Must be called with siblings only — ALCB_v1 excluded
    sibling_mock.assert_awaited_once()
    called_ids = sibling_mock.call_args[0][0]
    assert "ALCB_v1" not in called_ids
    assert set(called_ids) == {"IARIC_v1"}


def test_invalid_collision_action_raises():
    """Invalid symbol_collision_action raises ValueError at config init."""
    with pytest.raises(ValueError, match="Invalid symbol_collision_action"):
        PortfolioRulesConfig(symbol_collision_action="quarter_size")


def test_invalid_pair_collision_action_raises():
    """Invalid action in symbol_collision_pairs raises ValueError at config init."""
    with pytest.raises(ValueError, match="Invalid pair action"):
        PortfolioRulesConfig(
            symbol_collision_pairs=(("ALCB_v1", "IARIC_v1", "quarter_size"),),
        )


# ── Priority-based directional cap reservation ────────────────────


PRIORITY_KWARGS = dict(
    strategy_priorities=(("IARIC_v1", 0), ("ALCB_v1", 1)),
    priority_headroom_R=5.0,
    priority_reserve_threshold=0,
)


def _make_priority_checker(
    *,
    dir_risk_family: float = 0.0,
    **extra,
) -> PortfolioRuleChecker:
    kw = {**PRIORITY_KWARGS, **extra}
    config = PortfolioRulesConfig(
        directional_cap_R=8.0,
        initial_equity=10_000.0,
        family_strategy_ids=STOCK_IDS,
        symbol_collision_action="none",
        **kw,
    )
    return PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=0.0),
        get_current_equity=lambda: 10_000.0,
        get_directional_risk_R_for_strategies=AsyncMock(return_value=dir_risk_family),
        get_sibling_positions_for_symbol=AsyncMock(return_value=False),
    )


@pytest.mark.asyncio
async def test_priority_disabled_when_headroom_zero():
    """headroom_R=0 → backward compatible, no priority enforcement."""
    checker = _make_priority_checker(
        dir_risk_family=6.0,
        priority_headroom_R=0.0,
    )
    # 6R + 1R = 7R < 8R cap, remaining=2R < 3R headroom, but headroom disabled
    result = await checker.check_entry("ALCB_v1", "LONG", 1.0)
    assert result.approved


@pytest.mark.asyncio
async def test_priority_iaric_passes_in_reserved_headroom():
    """Priority 0 (IARIC) passes even when remaining <= headroom_R."""
    checker = _make_priority_checker(dir_risk_family=4.0)
    # remaining = 8-4 = 4R <= 5R headroom, but IARIC priority 0 <= threshold 0
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0)
    assert result.approved


@pytest.mark.asyncio
async def test_priority_alcb_blocked_in_reserved_headroom():
    """Priority 1 (ALCB) is blocked when remaining <= headroom_R."""
    checker = _make_priority_checker(dir_risk_family=4.0)
    # remaining = 4R <= 5R headroom, ALCB priority 1 > threshold 0 → blocked
    result = await checker.check_entry("ALCB_v1", "LONG", 1.0)
    assert not result.approved
    assert "directional_cap_reserved" in result.denial_reason
    assert "priority 1" in result.denial_reason


@pytest.mark.asyncio
async def test_priority_hard_cap_blocks_all_regardless_of_priority():
    """All strategies blocked when total > hard cap, priority irrelevant."""
    checker = _make_priority_checker(dir_risk_family=7.5)
    # 7.5R + 1R = 8.5R > 8R → hard cap blocks even IARIC
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0)
    assert not result.approved
    assert "directional_cap:" in result.denial_reason
    assert "directional_cap_reserved" not in result.denial_reason


@pytest.mark.asyncio
async def test_priority_unknown_strategy_gets_default_99():
    """Strategy not in strategy_priorities gets default priority 99 (blocked when headroom active)."""
    checker = _make_priority_checker(dir_risk_family=4.0)
    result = await checker.check_entry("UNKNOWN_v1", "LONG", 1.0)
    assert not result.approved
    assert "priority 99" in result.denial_reason


@pytest.mark.asyncio
async def test_priority_alcb_passes_when_headroom_sufficient():
    """ALCB passes when remaining > headroom_R (plenty of capacity)."""
    checker = _make_priority_checker(dir_risk_family=2.0)
    # remaining = 8-2 = 6R > 5R headroom → reservation not triggered
    result = await checker.check_entry("ALCB_v1", "LONG", 1.0)
    assert result.approved


# ── Per-pair symbol collision overrides ────────────────────────────


COLLISION_PAIR = (("ALCB_v1", "IARIC_v1", "block"),)


def _make_pair_checker(
    *,
    pairs: tuple[tuple[str, str, str], ...] = COLLISION_PAIR,
    holder_has_symbol: bool = False,
    generic_sibling_has: bool = False,
) -> PortfolioRuleChecker:
    """Build checker with per-pair collision overrides.

    Uses a sibling callback that returns holder_has_symbol when queried for a
    single holder ID, and generic_sibling_has for broader sibling queries.
    """
    async def _sibling_cb(ids: list[str], symbol: str) -> bool:
        # Per-pair queries pass a single holder ID
        if len(ids) == 1 and ids[0] in {h for h, _, _ in pairs}:
            return holder_has_symbol
        return generic_sibling_has

    config = PortfolioRulesConfig(
        directional_cap_R=0,  # disable cap to isolate collision tests
        family_strategy_ids=STOCK_IDS,
        symbol_collision_action="half_size",  # generic fallback
        symbol_collision_pairs=pairs,
    )
    return PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=0.0),
        get_current_equity=lambda: 10_000.0,
        get_sibling_positions_for_symbol=_sibling_cb,
    )


@pytest.mark.asyncio
async def test_pair_alcb_holds_blocks_iaric():
    """ALCB holds AAPL → IARIC fully blocked (not half_size)."""
    checker = _make_pair_checker(holder_has_symbol=True)
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0, symbol="AAPL")
    assert not result.approved
    assert "symbol_collision" in result.denial_reason


@pytest.mark.asyncio
async def test_pair_alcb_not_holding_falls_through_to_generic():
    """ALCB does NOT hold AAPL → pair override does not block the request."""
    checker = _make_pair_checker(holder_has_symbol=False, generic_sibling_has=True)
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0, symbol="AAPL")
    # With only one sibling in the family, the override holder and generic sibling
    # are the same strategy, so "holder missing" means no collision at all.
    assert result.approved
    assert result.size_multiplier == 1.0


@pytest.mark.asyncio
async def test_pair_iaric_holds_alcb_gets_half_size():
    """IARIC holds AAPL + ALCB requests → generic half_size (no pair override for IARIC→ALCB)."""
    checker = _make_pair_checker(holder_has_symbol=False, generic_sibling_has=True)
    result = await checker.check_entry("ALCB_v1", "LONG", 1.0, symbol="AAPL")
    assert result.approved
    assert result.size_multiplier == 0.5


@pytest.mark.asyncio
async def test_pair_alcb_holds_alcb_request_gets_half_size():
    """ALCB holds AAPL + ALCB requests → generic half_size (pair only targets IARIC)."""
    checker = _make_pair_checker(holder_has_symbol=True, generic_sibling_has=True)
    result = await checker.check_entry("ALCB_v1", "LONG", 1.0, symbol="AAPL")
    # Pair override doesn't match ALCB as requester → generic half_size
    assert result.approved
    assert result.size_multiplier == 0.5


@pytest.mark.asyncio
async def test_pair_no_pairs_backward_compatible():
    """Empty symbol_collision_pairs = backward compatible, generic action only."""
    checker = _make_pair_checker(pairs=(), generic_sibling_has=True)
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0, symbol="AAPL")
    assert result.approved
    assert result.size_multiplier == 0.5


@pytest.mark.asyncio
async def test_pair_no_collision_at_all():
    """No holder has symbol, no generic collision → approved at full size."""
    checker = _make_pair_checker(holder_has_symbol=False, generic_sibling_has=False)
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0, symbol="AAPL")
    assert result.approved
    assert result.size_multiplier == 1.0
