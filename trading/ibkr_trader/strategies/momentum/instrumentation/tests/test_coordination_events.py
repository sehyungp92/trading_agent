import asyncio
import pytest
from libs.oms.risk.portfolio_rules import (
    PortfolioRuleChecker, PortfolioRulesConfig, PortfolioRuleResult,
)


@pytest.fixture
def captured_events():
    return []


@pytest.fixture
def config():
    return PortfolioRulesConfig(
        directional_cap_R=3.5,
        initial_equity=10_000.0,
    )


def _make_checker(config, captured_events, equity=10_000.0, dir_risk=0.0, signal=None):
    async def get_signal(sid):
        return signal

    async def get_dir_risk(direction):
        return dir_risk

    checker = PortfolioRuleChecker(
        config=config,
        get_strategy_signal=get_signal,
        get_directional_risk_R=get_dir_risk,
        get_current_equity=lambda: equity,
        on_rule_event=lambda evt: captured_events.append(evt),
    )
    return checker


def test_directional_cap_denial_emits_event(config, captured_events):
    checker = _make_checker(config, captured_events, dir_risk=3.0)
    result = asyncio.run(
        checker.check_entry("NQDTC_v2.1", "LONG", new_risk_R=1.0)
    )
    assert result.approved is False
    assert len(captured_events) == 1
    evt = captured_events[0]
    assert evt["rule_name"] == "directional_cap"
    assert evt["details"]["strategy_id"] == "NQDTC_v2.1"
    assert evt["result"] == "block"


def test_drawdown_tier_emits_event(config, captured_events):
    # 10% DD = tier 2 (50% sizing)
    checker = _make_checker(config, captured_events, equity=9_000.0)
    result = asyncio.run(
        checker.check_entry("NQDTC_v2.1", "LONG", new_risk_R=0.5)
    )
    assert result.approved is True
    # Should emit a drawdown_tier event with size_mult < 1
    dd_events = [e for e in captured_events if e["rule_name"] == "drawdown_tier"]
    assert len(dd_events) == 1
    assert dd_events[0]["details"]["size_multiplier"] == 0.5
    assert dd_events[0]["details"]["drawdown_pct"] == pytest.approx(0.10, abs=0.001)


def test_no_event_when_all_pass_at_full_size(config, captured_events):
    checker = _make_checker(config, captured_events)
    result = asyncio.run(
        checker.check_entry("NQDTC_v2.1", "LONG", new_risk_R=0.5)
    )
    assert result.approved is True
    # No coordination events when everything passes at full size
    assert len(captured_events) == 0
