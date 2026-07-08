from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from regime.live.provider import LiveDataProvider


def _seed_regime_cache(data_dir, dates: pd.DatetimeIndex) -> None:
    macro_df = pd.DataFrame(
        {
            "GROWTH": [-200_000.0] * len(dates),
            "INFLATION": [2.0] * len(dates),
        },
        index=dates,
    )
    market_df = pd.DataFrame(
        {
            "VIX": [18.0] * len(dates),
            "SPREAD": [2.5] * len(dates),
            "SLOPE_10Y2Y": [0.5] * len(dates),
            "REAL_RATE_10Y": [1.0] * len(dates),
            "DBC": [0.0] * len(dates),
        },
        index=dates,
    )
    strat_ret_df = pd.DataFrame(
        {
            "SPY": [0.001] * len(dates),
            "EFA": [0.001] * len(dates),
            "TLT": [0.0005] * len(dates),
            "GLD": [0.0002] * len(dates),
            "CASH": [0.0] * len(dates),
        },
        index=dates,
    )
    macro_df.to_parquet(data_dir / "macro_df.parquet")
    market_df.to_parquet(data_dir / "market_df.parquet")
    strat_ret_df.to_parquet(data_dir / "strat_ret_df.parquet")


@pytest.mark.asyncio
async def test_regime_provider_fetches_completed_daily_etf_bars(tmp_path) -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def req_historical_data(self, contract: object, **kwargs):
            self.calls.append(kwargs)
            start = datetime(2026, 4, 1, tzinfo=timezone.utc)
            return [
                SimpleNamespace(date=start + timedelta(days=i), close=100.0 + i)
                for i in range(5)
            ]

    session = FakeSession()
    provider = LiveDataProvider(
        session,
        tmp_path,
        now_provider=lambda: datetime(2026, 5, 1, 21, 10, tzinfo=timezone.utc),
    )
    provider._contracts = {"SPY": object(), "TLT": object()}

    prices = await provider._fetch_ibkr_bars()

    assert prices is not None
    assert {"SPY", "TLT"} <= set(prices.columns)
    assert all(call["completed_only"] is True for call in session.calls)
    assert all(call["barSizeSetting"] == "1 day" for call in session.calls)
    assert all(call["useRTH"] is True for call in session.calls)


@pytest.mark.asyncio
async def test_regime_provider_does_not_advance_hmm_data_on_fred_only_date(monkeypatch, tmp_path) -> None:
    dates = pd.date_range("2025-07-01", "2026-04-30", freq="D")
    _seed_regime_cache(tmp_path, dates)
    provider = LiveDataProvider(
        object(),
        tmp_path,
        now_provider=lambda: datetime(2026, 5, 1, 21, 10, tzinfo=timezone.utc),
    )

    async def no_fresh_ibkr():
        return None

    fred_next_day = pd.DataFrame(
        {
            "VIX": [25.0],
            "SPREAD": [3.0],
            "SLOPE_10Y2Y": [0.25],
            "REAL_RATE_10Y": [0.8],
            "INFLATION": [2.1],
        },
        index=[pd.Timestamp("2026-05-01")],
    )
    icsa = pd.Series([-210_000.0], index=[pd.Timestamp("2026-05-01")])

    monkeypatch.setattr(provider, "_fetch_ibkr_bars", no_fresh_ibkr)
    monkeypatch.setattr(provider, "_fetch_fred", lambda: (fred_next_day, icsa))

    macro_df, market_df, strat_ret_df = await provider.build_live_data()

    assert macro_df.index.max() == pd.Timestamp("2026-04-30")
    assert market_df.index.max() == pd.Timestamp("2026-04-30")
    assert strat_ret_df.index.max() == pd.Timestamp("2026-04-30")
    assert provider.last_data_as_of == "2026-04-30"
    assert "fred=fresh" in provider.last_data_status


@pytest.mark.asyncio
async def test_regime_provider_does_not_zero_fill_partial_live_etf_backlog(monkeypatch, tmp_path) -> None:
    dates = pd.date_range("2025-07-01", "2026-04-30", freq="D")
    _seed_regime_cache(tmp_path, dates)
    provider = LiveDataProvider(
        object(),
        tmp_path,
        now_provider=lambda: datetime(2026, 5, 1, 21, 10, tzinfo=timezone.utc),
    )

    async def partial_ibkr():
        return pd.DataFrame(
            {"SPY": [100.0, 101.0]},
            index=pd.DatetimeIndex(["2026-04-30", "2026-05-01"]),
        )

    monkeypatch.setattr(provider, "_fetch_ibkr_bars", partial_ibkr)
    monkeypatch.setattr(provider, "_fetch_fred", lambda: None)

    _, _, strat_ret_df = await provider.build_live_data()

    assert strat_ret_df.index.max() == pd.Timestamp("2026-04-30")
    assert provider.last_data_as_of == "2026-04-30"
    saved = pd.read_parquet(tmp_path / "strat_ret_df.parquet")
    assert saved.index.max() == pd.Timestamp("2026-04-30")


@pytest.mark.asyncio
async def test_regime_provider_fails_loudly_when_backfilled_hmm_backlog_is_too_short(monkeypatch, tmp_path) -> None:
    dates = pd.date_range("2026-04-01", periods=40, freq="D")
    _seed_regime_cache(tmp_path, dates)
    provider = LiveDataProvider(
        object(),
        tmp_path,
        now_provider=lambda: datetime(2026, 5, 10, 21, 10, tzinfo=timezone.utc),
    )

    async def no_fresh_ibkr():
        return None

    monkeypatch.setattr(provider, "_fetch_ibkr_bars", no_fresh_ibkr)
    monkeypatch.setattr(provider, "_fetch_fred", lambda: None)

    with pytest.raises(RuntimeError, match="Regime live backlog is insufficient"):
        await provider.build_live_data()
