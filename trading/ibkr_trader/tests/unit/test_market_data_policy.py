from __future__ import annotations

from typing import Any

import pytest

import libs.broker_ibkr.session as session_module
from libs.broker_ibkr.session import UnifiedIBSession
from libs.config.models import ConnectionGroupConfig


class _FakeEvent:
    def __init__(self) -> None:
        self._handlers = []

    def __iadd__(self, handler):
        self._handlers.append(handler)
        return self

    def __isub__(self, handler):
        self._handlers = [item for item in self._handlers if item is not handler]
        return self

    def emit(self, *args: Any) -> None:
        for handler in list(self._handlers):
            handler(*args)


class _FakeIB:
    def __init__(self) -> None:
        self.errorEvent = _FakeEvent()
        self.market_data_type = 1
        self.market_data_type_calls: list[int] = []

    async def qualifyContractsAsync(self, contract):
        return [contract]

    def reqMarketDataType(self, market_data_type: int) -> None:
        self.market_data_type = market_data_type
        self.market_data_type_calls.append(market_data_type)

    def reqMktData(self, contract, *_args):
        if self.market_data_type != 3:
            self.errorEvent.emit(
                1,
                10089,
                "Requested market data requires additional subscription",
                contract,
            )

    def cancelMktData(self, _contract) -> None:
        return None


def _session(fake_ib: _FakeIB) -> UnifiedIBSession:
    session = UnifiedIBSession(
        connection_groups={"main": ConnectionGroupConfig(client_id=1, market_data_type=1)},
        strategy_group_map={"strategy": "main"},
    )
    session.groups["main"].conn._ib = fake_ib
    return session


@pytest.mark.asyncio
async def test_live_market_data_rejects_delayed_fallback(monkeypatch) -> None:
    monkeypatch.setattr(session_module.asyncio, "sleep", _noop_sleep)
    monkeypatch.setenv("ALLOW_DELAYED_MARKET_DATA", "1")
    session = _session(_FakeIB())

    with pytest.raises(RuntimeError, match="delayed fallback is forbidden"):
        await session.verify_streaming_data(runtime_env="live")


@pytest.mark.asyncio
async def test_live_market_data_rejects_skip_streaming_check(monkeypatch) -> None:
    monkeypatch.setenv("SKIP_STREAMING_CHECK", "1")
    session = _session(_FakeIB())

    with pytest.raises(RuntimeError, match="SKIP_STREAMING_CHECK is forbidden"):
        await session.verify_streaming_data(runtime_env="live")


@pytest.mark.asyncio
async def test_paper_market_data_requires_explicit_delayed_override(monkeypatch) -> None:
    monkeypatch.setattr(session_module.asyncio, "sleep", _noop_sleep)
    monkeypatch.delenv("ALLOW_DELAYED_MARKET_DATA", raising=False)
    session = _session(_FakeIB())

    with pytest.raises(RuntimeError, match="ALLOW_DELAYED_MARKET_DATA=1"):
        await session.verify_streaming_data(runtime_env="paper")


@pytest.mark.asyncio
async def test_paper_market_data_override_emits_delayed_metadata(monkeypatch) -> None:
    monkeypatch.setattr(session_module.asyncio, "sleep", _noop_sleep)
    monkeypatch.setenv("ALLOW_DELAYED_MARKET_DATA", "1")
    fake_ib = _FakeIB()
    session = _session(fake_ib)

    status = await session.verify_streaming_data(runtime_env="paper")

    assert status["mode"] == "delayed"
    assert status["delayed_data"] is True
    assert status["allow_delayed_market_data"] is True
    assert status["real_time_error_code"] == 10089
    assert fake_ib.market_data_type_calls == [3, 1]


async def _noop_sleep(_seconds: float) -> None:
    return None
