from __future__ import annotations

import pytest

from libs.broker_ibkr.config.schemas import ContractTemplate, ExchangeRoute
from libs.broker_ibkr.mapping.contract_factory import ContractFactory
from libs.oms.models.instrument import Instrument


class _FakeIB:
    async def qualifyContractsAsync(self, contract):
        contract.conId = 123456
        contract.symbol = "UK1T"
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "GBP"
        contract.primaryExchange = "LSEETF"
        contract.tradingClass = "UK1T"
        return [contract]


@pytest.mark.asyncio
async def test_contract_factory_resolves_logical_symbol_to_broker_alias():
    factory = ContractFactory(
        ib=_FakeIB(),
        templates={
            "UKETF": ContractTemplate(
                symbol="UK1T",
                sec_type="STK",
                exchange="SMART",
                currency="GBP",
                multiplier=1.0,
                tick_size=0.01,
                tick_value=0.01,
                primary_exchange="LSEETF",
            )
        },
        routes={
            "UKETF": ExchangeRoute(
                root_symbol="UKETF",
                exchange="SMART",
                primary_exchange="LSEETF",
            )
        },
    )
    logical_instrument = Instrument(
        symbol="UKETF",
        root="UKETF",
        venue="SMART",
        tick_size=0.01,
        tick_value=0.01,
        multiplier=1.0,
        point_value=1.0,
        currency="USD",
        primary_exchange="NASDAQ",
        sec_type="STK",
    )

    contract, spec = await factory.resolve("UKETF", instrument=logical_instrument)

    assert contract.symbol == "UK1T"
    assert contract.currency == "GBP"
    assert getattr(contract, "primaryExchange", "") == "LSEETF"
    assert spec.symbol == "UK1T"
    assert spec.currency == "GBP"
    assert spec.primary_exchange == "LSEETF"
    assert factory.logical_symbol_for_contract(contract) == "UKETF"
