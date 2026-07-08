from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from libs.broker_ibkr.config.schemas import ContractTemplate
from libs.broker_ibkr.mapping.contract_factory import ContractFactory
from libs.market_data.futures_roll import (
    active_contract_month,
    contract_month_for_order,
    is_roll_blackout,
    roll_blackout_reason,
    roll_force_flatten_reason,
    with_active_contract_expiry,
    with_contract_expiry_for_order,
)
from libs.market_data.live_futures import clear_historical_cache, req_panama_adjusted_historical_data
from libs.market_data.panama import stitch_panama
from libs.oms.intent.handler import IntentHandler
from libs.oms.models.instrument import Instrument
from libs.oms.models.intent import Intent, IntentResult, IntentType
from libs.oms.models.order import OMSOrder, OrderRole, OrderSide, OrderType


def _mnq_instrument(expiry: str = "") -> Instrument:
    return Instrument(
        symbol="MNQ",
        root="MNQ",
        venue="CME",
        tick_size=0.25,
        tick_value=0.50,
        multiplier=2.0,
        point_value=2.0,
        contract_expiry=expiry,
        sec_type="FUT",
        trading_class="MNQ",
    )


def test_active_contract_and_roll_blackout_match_backtest_policy():
    before_roll = datetime(2026, 3, 15, 12, tzinfo=timezone.utc)
    roll_day = datetime(2026, 3, 16, 12, tzinfo=timezone.utc)
    after_blackout = datetime(2026, 3, 21, 12, tzinfo=timezone.utc)

    assert active_contract_month("MNQ", as_of=before_roll) == "202603"
    assert active_contract_month("MNQ", as_of=roll_day) == "202606"
    assert is_roll_blackout("MNQ", as_of=before_roll)
    assert is_roll_blackout("MNQ", as_of=roll_day)
    assert not is_roll_blackout("MNQ", as_of=after_blackout)
    assert "202603 -> 202606" in (roll_blackout_reason("MNQ", as_of=roll_day) or "")
    assert "Futures roll safety exit" in (roll_force_flatten_reason("MNQ", as_of=roll_day) or "")


def test_non_entry_orders_keep_old_contract_during_roll_blackout():
    during_roll_week = datetime(2026, 3, 18, 12, tzinfo=timezone.utc)

    assert contract_month_for_order("MNQ", order_role="ENTRY", as_of=during_roll_week) == "202606"
    assert contract_month_for_order("MNQ", order_role="STOP", as_of=during_roll_week) == "202603"

    exit_inst = with_contract_expiry_for_order(
        _mnq_instrument(""),
        order_role="EXIT",
        as_of=during_roll_week,
    )
    assert exit_inst.contract_expiry == "202603"


def test_with_active_contract_expiry_populates_blank_supported_future():
    instrument = _mnq_instrument("")

    resolved = with_active_contract_expiry(
        instrument,
        as_of=datetime(2026, 3, 16, 12, tzinfo=timezone.utc),
    )

    assert resolved.contract_expiry == "202606"
    assert resolved.sec_type == "FUT"
    assert resolved.trading_class == "MNQ"
    assert instrument.contract_expiry == ""


def test_panama_stitch_fails_closed_on_implausible_roll_gap():
    old = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [10]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-13T21:00:00Z")]),
    )
    new = pd.DataFrame(
        {"open": [1000.0], "high": [1001.0], "low": [999.0], "close": [1000.5], "volume": [20]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-16T00:00:00Z")]),
    )

    stitched = stitch_panama(
        {"202603": old, "202606": new},
        [(date(2026, 3, 16), "202603", "202606")],
        tick_size=0.25,
    )

    assert stitched.empty


class _FakeIB:
    def __init__(self) -> None:
        self.contracts = []

    async def qualifyContractsAsync(self, contract):
        self.contracts.append(contract)
        contract.conId = 1001
        contract.secType = "FUT"
        contract.symbol = "MNQ"
        contract.exchange = "CME"
        contract.currency = "USD"
        contract.tradingClass = "MNQ"
        return [contract]


@pytest.mark.asyncio
async def test_contract_factory_resolves_blank_future_expiry_to_active_month(monkeypatch):
    fake_ib = _FakeIB()
    monkeypatch.setattr(
        "libs.broker_ibkr.mapping.contract_factory.active_contract_month",
        lambda symbol: "202606",
    )
    factory = ContractFactory(
        ib=fake_ib,
        templates={
            "MNQ": ContractTemplate(
                symbol="MNQ",
                sec_type="FUT",
                exchange="CME",
                currency="USD",
                multiplier=2.0,
                tick_size=0.25,
                tick_value=0.50,
                trading_class="MNQ",
            )
        },
        routes={},
    )

    contract, spec = await factory.resolve("MNQ")

    assert getattr(fake_ib.contracts[0], "lastTradeDateOrContractMonth", "") == "202606"
    assert getattr(contract, "lastTradeDateOrContractMonth", "") == "202606"
    assert spec.last_trade_date == "202606"


class _FakeSessionIB:
    def isConnected(self) -> bool:
        return True

    async def qualifyContractsAsync(self, contract):
        return [contract]


class _FakeHistoricalSession:
    def __init__(self) -> None:
        self.ib = _FakeSessionIB()
        self.requests: list[tuple[str, str]] = []

    async def req_historical_data(self, contract, **kwargs):
        month = getattr(contract, "lastTradeDateOrContractMonth", "")
        self.requests.append((month, kwargs.get("durationStr", "")))
        if month == "202603":
            return [
                SimpleNamespace(
                    date=datetime(2026, 3, 5, 21, tzinfo=timezone.utc),
                    open=95.0,
                    high=97.0,
                    low=94.0,
                    close=96.0,
                    volume=5,
                ),
                SimpleNamespace(
                    date=datetime(2026, 3, 13, 21, tzinfo=timezone.utc),
                    open=99.0,
                    high=101.0,
                    low=98.0,
                    close=100.0,
                    volume=10,
                )
            ]
        if month == "202606":
            return [
                SimpleNamespace(
                    date=datetime(2026, 3, 16, 0, tzinfo=timezone.utc),
                    open=110.0,
                    high=112.0,
                    low=109.0,
                    close=110.5,
                    volume=20,
                ),
                SimpleNamespace(
                    date=datetime(2026, 3, 17, 0, tzinfo=timezone.utc),
                    open=111.0,
                    high=113.0,
                    low=110.0,
                    close=112.0,
                    volume=30,
                ),
            ]
        return []


@pytest.mark.asyncio
async def test_live_historical_helper_fetches_physical_contracts_and_panama_stitches():
    clear_historical_cache()
    session = _FakeHistoricalSession()
    contract = SimpleNamespace(symbol="NQ")

    bars = await req_panama_adjusted_historical_data(
        session,
        contract,
        symbol="NQ",
        durationStr="10 D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=1,
        request_kind="test",
        completed_only=True,
        as_of=datetime(2026, 3, 20, 12, tzinfo=timezone.utc),
        cache_ttl_s=0,
    )

    assert [request[0] for request in session.requests] == ["202603", "202606"]
    assert len(bars) == 3
    assert bars[0].close == 90.0
    assert bars[1].close == 110.5


class _MissingNewSideSession:
    """Old-month bars only; new month returns nothing across the roll window."""

    def __init__(self) -> None:
        self.ib = _FakeSessionIB()
        self.requests: list[str] = []

    async def req_historical_data(self, contract, **kwargs):
        month = getattr(contract, "lastTradeDateOrContractMonth", "")
        self.requests.append(month)
        if month == "202603":
            return [
                SimpleNamespace(
                    date=datetime(2026, 3, 13, 21, tzinfo=timezone.utc),
                    open=99.0, high=101.0, low=98.0, close=100.0, volume=10,
                )
            ]
        return []


@pytest.mark.asyncio
async def test_live_historical_helper_fails_closed_when_roll_data_missing():
    clear_historical_cache()
    session = _MissingNewSideSession()

    bars = await req_panama_adjusted_historical_data(
        session,
        SimpleNamespace(symbol="NQ"),
        symbol="NQ",
        durationStr="10 D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=1,
        request_kind="test",
        completed_only=True,
        as_of=datetime(2026, 3, 20, 12, tzinfo=timezone.utc),
        cache_ttl_s=0,
    )

    assert bars == []


class _PassthroughSession:
    def __init__(self) -> None:
        self.ib = _FakeSessionIB()
        self.calls = 0

    async def req_historical_data(self, contract, **kwargs):
        self.calls += 1
        return [
            SimpleNamespace(
                date=datetime(2026, 3, 20, 12, tzinfo=timezone.utc),
                open=100.0, high=101.0, low=99.0, close=100.5, volume=1,
            )
        ]


@pytest.mark.asyncio
async def test_live_historical_helper_passes_through_for_unsupported_root():
    clear_historical_cache()
    session = _PassthroughSession()

    bars = await req_panama_adjusted_historical_data(
        session,
        SimpleNamespace(symbol="SPY"),
        symbol="SPY",
        durationStr="1 D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
        request_kind="test",
        completed_only=True,
        as_of=datetime(2026, 3, 20, 12, tzinfo=timezone.utc),
        cache_ttl_s=0,
    )

    assert session.calls == 1
    assert len(bars) == 1
    assert bars[0].close == 100.5


class _NoopRisk:
    called = False

    async def check_entry(self, *args, **kwargs):
        self.called = True
        return None


class _NoopRouter:
    called = False

    async def route(self, order):
        self.called = True


class _NoopRepo:
    async def get_positions(self, *args, **kwargs):
        return []


class _NoopBus:
    def emit_risk_denial(self, *args, **kwargs):
        pass

    def emit_order_event(self, *args, **kwargs):
        pass


@pytest.mark.asyncio
async def test_intent_handler_denies_new_entries_during_roll_blackout(monkeypatch):
    risk = _NoopRisk()
    router = _NoopRouter()
    handler = IntentHandler(
        risk=risk,
        router=router,
        repo=_NoopRepo(),
        bus=_NoopBus(),
    )
    monkeypatch.setattr(
        "libs.oms.intent.handler.roll_blackout_reason",
        lambda instrument, as_of=None: "roll blackout",
    )
    order = OMSOrder(
        strategy_id="test",
        instrument=_mnq_instrument(""),
        side=OrderSide.BUY,
        qty=1,
        order_type=OrderType.LIMIT,
        limit_price=100.0,
        role=OrderRole.ENTRY,
    )

    receipt = await handler.submit(Intent(IntentType.NEW_ORDER, strategy_id="test", order=order))

    assert receipt.result is IntentResult.DENIED
    assert receipt.denial_reason == "roll blackout"
    assert not risk.called
    assert not router.called
