from __future__ import annotations

from types import SimpleNamespace

import pytest

from libs.oms.instrumentation.daily_state import collect_family_daily_state


class FakeRepo:
    def __init__(self, positions):
        self._positions = list(positions)

    async def get_positions_for_strategies(self, strategy_ids):
        wanted = set(strategy_ids)
        return [pos for pos in self._positions if pos["strategy_id"] in wanted]


@pytest.mark.asyncio
async def test_collect_family_daily_state_merges_per_strategy_oms_state() -> None:
    async def risk_a():
        return {
            "daily_realized_pnl": 100.0,
            "daily_realized_R": 1.0,
            "weekly_realized_pnl": 100.0,
            "weekly_realized_R": 1.0,
            "strategy_daily_pnl": {"A": 100.0},
            "open_risk_R": 0.1,
            "pending_entry_risk_R": 0.2,
        }

    async def risk_b():
        return {
            "daily_realized_pnl": 50.0,
            "daily_realized_R": 0.5,
            "weekly_realized_pnl": 50.0,
            "weekly_realized_R": 0.5,
            "strategy_daily_pnl": {"B": 50.0},
            "open_risk_R": 0.1,
            "pending_entry_risk_R": 0.1,
            "halted": True,
            "halt_reason": "daily_stop",
        }

    services = [
        SimpleNamespace(
            get_portfolio_risk=risk_a,
            _allocation_targets={"families": {"stock": 1.0}, "strategies": {"A": 0.6}},
            _account_state_provider=lambda: {"equity": 100_000.0, "raw_nav": 100_000.0},
            _oms_repo=FakeRepo([{
                "account_id": "paper",
                "strategy_id": "A",
                "instrument_symbol": "QQQ",
                "open_risk_dollars": 75.0,
                "open_risk_R": 0.15,
            }]),
        ),
        SimpleNamespace(
            get_portfolio_risk=risk_b,
            _allocation_targets={"strategies": {"B": 0.4}},
            _account_state_provider=lambda: {"equity": 99_000.0, "raw_nav": 99_000.0},
            _oms_repo=FakeRepo([{
                "account_id": "paper",
                "strategy_id": "B",
                "instrument_symbol": "SPY",
                "open_risk_dollars": 25.0,
                "open_risk_R": 0.05,
            }]),
        ),
    ]

    portfolio_state, targets, allocation_state = await collect_family_daily_state(
        services,
        strategy_ids=["A", "B"],
        default_strategy_id="A",
    )

    assert portfolio_state["daily_realized_pnl"] == 150.0
    assert portfolio_state["daily_realized_R"] == 1.5
    assert portfolio_state["strategy_daily_pnl"] == {"A": 100.0, "B": 50.0}
    assert portfolio_state["weekly_realized_pnl"] == 150.0
    assert portfolio_state["open_risk_dollars"] == 100.0
    assert portfolio_state["open_risk_R"] == 0.1
    assert portfolio_state["pending_entry_risk_R"] == pytest.approx(0.3)
    assert portfolio_state["halted"] is True
    assert portfolio_state["halt_reason"] == "daily_stop"
    assert len(portfolio_state["positions"]) == 2
    assert targets["families"] == {"stock": 1.0}
    assert targets["strategies"] == {"A": 0.6, "B": 0.4}
    assert allocation_state["raw_nav"] == 100_000.0
