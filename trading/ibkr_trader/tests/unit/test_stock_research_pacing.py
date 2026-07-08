from types import SimpleNamespace

import pytest

from strategies.stock.alcb import research_generator as alcb_research
from strategies.stock.iaric import research_generator as iaric_research


class _FakeIB:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def reqHistoricalDataAsync(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_alcb_fetch_bars_passes_timeout_and_retries_timeout_empty(monkeypatch) -> None:
    bar = object()

    class SlowFirstIB(_FakeIB):
        async def reqHistoricalDataAsync(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            if len(self.calls) == 1:
                return []
            return [bar]

    ib = SlowFirstIB([])
    contract = SimpleNamespace(symbol="AAPL")

    monkeypatch.setattr(alcb_research, "HISTORICAL_REQUEST_TIMEOUT", 0.001)
    monkeypatch.setattr(alcb_research, "HISTORICAL_TIMEOUT_RETRY_DELAYS", (0.0,))
    ticks = iter([0.0, 0.001, 0.001, 0.001])
    monkeypatch.setattr(
        alcb_research,
        "time",
        SimpleNamespace(monotonic=lambda: next(ticks)),
    )

    bars = await alcb_research._fetch_bars(ib, contract, "90 D", "30 mins")

    assert bars == [bar]
    assert len(ib.calls) == 2
    assert ib.calls[0][1]["timeout"] == 0.001


@pytest.mark.asyncio
async def test_iaric_request_historical_bars_does_not_retry_fast_empty(monkeypatch) -> None:
    ib = _FakeIB([[]])
    contract = SimpleNamespace(symbol="NOPE")

    monkeypatch.setattr(iaric_research, "HISTORICAL_REQUEST_TIMEOUT", 10.0)
    monkeypatch.setattr(iaric_research, "HISTORICAL_TIMEOUT_RETRY_DELAYS", (0.0,))

    bars = await iaric_research._request_historical_bars(
        ib,
        contract,
        duration="1 Y",
        bar_size="1 day",
    )

    assert bars == []
    assert len(ib.calls) == 1
    assert ib.calls[0][1]["timeout"] == 10.0
