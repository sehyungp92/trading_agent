from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from libs.oms.models.instrument import Instrument
from strategies.swing.akc_helix.config import SymbolConfig
from strategies.swing.akc_helix.engine import HelixEngine


class _FakeEvent:
    def __init__(self) -> None:
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def __isub__(self, handler):
        self.handlers = [item for item in self.handlers if item != handler]
        return self


class _FakeIBClient:
    def __init__(self) -> None:
        self.pendingTickersEvent = _FakeEvent()
        self.req_mkt_data_calls = []

    async def qualifyContractsAsync(self, contract):
        return [contract]

    def reqMktData(self, contract, *args):
        self.req_mkt_data_calls.append(contract)
        return SimpleNamespace(contract=contract)

    def cancelMktData(self, contract):
        return None


class _FakeSession:
    def __init__(self, factory) -> None:
        self.ib = _FakeIBClient()
        self._contract_factory = factory
        self.callbacks = []

    def register_farm_recovery_callback(self, group: str, callback) -> None:
        self.callbacks.append((group, callback))


class _FakeFactory:
    def __init__(self) -> None:
        self.contract = SimpleNamespace(
            conId=777,
            symbol="GLD",
            secType="STK",
            exchange="SMART",
            currency="USD",
            primaryExchange="ARCA",
        )

    async def resolve(self, symbol: str, expiry: str = "", instrument=None):
        return self.contract, SimpleNamespace(con_id=self.contract.conId)

    def build_contract(self, symbol: str, expiry: str = "", instrument=None):
        return self.contract

    def logical_symbol_for_contract(self, contract) -> str:
        if int(getattr(contract, "conId", 0) or 0) == self.contract.conId:
            return "GLD"
        return str(getattr(contract, "symbol", "") or "").upper()


class _FakeOMS:
    def stream_events(self, strategy_id: str):
        return asyncio.Queue()


@pytest.mark.asyncio
async def test_helix_start_uses_contract_factory_resolution_for_tradeable_symbol() -> None:
    factory = _FakeFactory()
    session = _FakeSession(factory)
    instruments = {
        "GLD": Instrument(
            symbol="GLD",
            root="GLD",
            venue="SMART",
            tick_size=0.01,
            tick_value=0.01,
            multiplier=1.0,
            point_value=1.0,
            currency="USD",
            primary_exchange="ARCA",
            sec_type="STK",
        )
    }
    config = {
        "GLD": SymbolConfig(
            symbol="GLD",
            is_etf=True,
            sec_type="STK",
            exchange="SMART",
            contract_expiry="",
        )
    }
    engine = HelixEngine(
        ib_session=session,
        oms_service=_FakeOMS(),
        instruments=instruments,
        config=config,
        equity=30_000.0,
    )
    engine._load_initial_bars = AsyncMock()
    engine._process_events = AsyncMock()
    engine._hourly_scheduler = AsyncMock()
    engine._trigger_monitor = AsyncMock()
    engine._window_close_scheduler = AsyncMock()

    await engine.start()

    assert engine.contracts["GLD"][0].symbol == "GLD"
    assert session.ib.req_mkt_data_calls
    assert session.ib.req_mkt_data_calls[0].symbol == "GLD"

    await engine.stop()
