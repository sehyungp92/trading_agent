from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from strategies.momentum.downturn.engine import DownturnEngine
from strategies.momentum.nqdtc.engine import NQDTCEngine
from strategies.momentum.vdub.engine import VdubNQv4Engine
from strategies.swing.akc_helix.engine import HelixEngine
from strategies.swing.atrss.engine import ATRSSEngine

UTC = timezone.utc


def _bars() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            date=datetime(2024, 1, 5, 14, 0, tzinfo=UTC),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=10.0,
        ),
        SimpleNamespace(
            date=datetime(2024, 1, 5, 15, 0, tzinfo=UTC),
            open=100.5,
            high=101.5,
            low=100.0,
            close=101.0,
            volume=12.0,
        ),
    ]


class _FakeIBClient:
    def isConnected(self) -> bool:
        return True

    async def qualifyContractsAsync(self, contract: object) -> list[object]:
        return [contract]


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.ib = _FakeIBClient()
        self.is_connected = True

    async def req_historical_data(self, contract: object, **kwargs):
        self.calls.append(kwargs)
        return _bars()




@pytest.mark.asyncio
async def test_nqdtc_fetches_completed_only_bars() -> None:
    session = _FakeSession()
    engine = NQDTCEngine.__new__(NQDTCEngine)
    engine._ib = session

    await engine._req_completed_bars(object(), "2 D", "5 mins", request_kind="recurring")

    assert session.calls[-1]["completed_only"] is True


@pytest.mark.asyncio
async def test_downturn_fetches_completed_only_bars() -> None:
    session = _FakeSession()
    engine = DownturnEngine.__new__(DownturnEngine)
    engine._ib = session
    engine._symbol = "MNQ"
    engine._get_contract = lambda: object()

    await engine._fetch_bars()

    assert session.calls
    assert all(call["completed_only"] is True for call in session.calls)


@pytest.mark.asyncio
async def test_vdub_helper_fetches_completed_only_bars() -> None:
    session = _FakeSession()
    engine = VdubNQv4Engine.__new__(VdubNQv4Engine)
    engine._ib = session

    await engine._req_bars(object(), "30 D", "15 mins", request_kind="recurring")

    assert session.calls[-1]["completed_only"] is True


@pytest.mark.asyncio
async def test_atrss_daily_and_hourly_fetches_request_completed_only_bars() -> None:
    session = _FakeSession()
    engine = ATRSSEngine.__new__(ATRSSEngine)
    engine._ib = session
    engine._get_contract = lambda sym: object()

    await engine._fetch_daily_bars("SPY", None)
    await engine._fetch_hourly_bars("SPY", None)

    assert [call["barSizeSetting"] for call in session.calls] == ["1 day", "1 hour"]
    assert all(call["completed_only"] is True for call in session.calls)


@pytest.mark.asyncio
async def test_akc_helix_fetches_completed_only_bars() -> None:
    session = _FakeSession()
    engine = HelixEngine.__new__(HelixEngine)
    engine._ib = session
    engine._get_contract = lambda sym: object()

    await engine._fetch_bars("ES", None, "1 hour", "30 D")

    assert session.calls[-1]["completed_only"] is True
