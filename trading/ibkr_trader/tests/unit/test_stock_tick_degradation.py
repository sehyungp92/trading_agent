from __future__ import annotations

from types import SimpleNamespace

import pytest

from libs.oms.models.instrument import Instrument
from strategies.stock.alcb.data import IBMarketDataSource as ALCBMarketDataSource
from strategies.stock.iaric.data import IBMarketDataSource as IARICMarketDataSource


class _FakeEvent:
    def __init__(self) -> None:
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def __isub__(self, handler):
        self.handlers = [item for item in self.handlers if item != handler]
        return self


class _FakeIB:
    def __init__(self) -> None:
        self.pendingTickersEvent = _FakeEvent()
        self.errorEvent = _FakeEvent()
        self.req_mkt_data_calls = []
        self.req_tick_by_tick_calls = []
        self.cancel_tick_by_tick_calls = []
        self.cancel_mkt_data_calls = []

    def reqMktData(self, contract):
        self.req_mkt_data_calls.append(contract)
        return contract

    def reqTickByTickData(self, contract, tick_type):
        self.req_tick_by_tick_calls.append((contract, tick_type))

    def cancelTickByTickData(self, contract, tick_type):
        self.cancel_tick_by_tick_calls.append((contract, tick_type))

    def cancelMktData(self, contract):
        self.cancel_mkt_data_calls.append(contract)


class _FakeFactory:
    def __init__(self, broker_symbol: str = "NFLX") -> None:
        self._contracts = {}
        self._logical_by_conid = {}
        self._broker_symbol = broker_symbol

    async def resolve(self, symbol: str, expiry: str = "", instrument=None):
        contract = SimpleNamespace(
            conId=len(self._contracts) + 100,
            symbol=symbol,
            secType="STK",
            exchange="SMART",
            primaryExchange="NASDAQ",
            currency="USD",
        )
        self._contracts[symbol] = contract
        self._logical_by_conid[contract.conId] = symbol
        return contract, SimpleNamespace(con_id=contract.conId)

    def logical_symbol_for_contract(self, contract) -> str:
        con_id = int(getattr(contract, "conId", 0) or 0)
        return self._logical_by_conid.get(con_id, "")


def _instrument(symbol: str) -> Instrument:
    return Instrument(
        symbol=symbol,
        root=symbol,
        venue="SMART",
        tick_size=0.01,
        tick_value=0.01,
        multiplier=1.0,
        point_value=1.0,
        currency="USD",
        primary_exchange="NASDAQ",
        sec_type="STK",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_cls", "symbol"),
    [
        (IARICMarketDataSource, "NFLX"),
        (ALCBMarketDataSource, "TSLA"),
    ],
)
async def test_stock_data_sources_degrade_10190_without_blacklisting(source_cls, symbol: str) -> None:
    ib = _FakeIB()
    factory = _FakeFactory()
    source = source_cls(ib, factory, lambda *args: None, lambda *args: None)
    instrument = _instrument(symbol)

    await source.ensure_hot_symbols([instrument])
    source._on_ib_error(1, 10190, "tick cap", factory._contracts[symbol])

    assert symbol in source._tick_by_tick_disabled
    assert symbol not in source._blacklisted
    assert len(ib.cancel_tick_by_tick_calls) == 2
