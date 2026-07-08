"""Tests for portfolio backtest live-parity accounting helpers."""

from types import SimpleNamespace

import pytest

from crypto_trader.broker.sim_broker import SimBroker
from crypto_trader.portfolio.backtest_runner import _portfolio_equity_from_slots


def test_portfolio_equity_from_slots_uses_shared_capital_deltas():
    b1 = SimBroker(initial_equity=10_000.0)
    b2 = SimBroker(initial_equity=10_000.0)
    b1._equity = 10_500.0
    b2._equity = 9_750.0
    slots = [
        SimpleNamespace(broker=b1),
        SimpleNamespace(broker=b2),
    ]

    assert _portfolio_equity_from_slots(slots, 10_000.0) == pytest.approx(10_250.0)


def test_portfolio_equity_from_slots_deduplicates_shared_broker_instances():
    broker = SimBroker(initial_equity=10_000.0)
    broker._equity = 10_500.0
    slots = [
        SimpleNamespace(broker=broker),
        SimpleNamespace(broker=broker),
    ]

    assert _portfolio_equity_from_slots(slots, 10_000.0) == pytest.approx(10_500.0)
