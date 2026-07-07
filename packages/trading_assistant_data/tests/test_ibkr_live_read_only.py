from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from trading_assistant_data.sources.ibkr.cme_nq_read_only import CmeNqRefreshRequest
from trading_assistant_data.sources.ibkr.live_read_only import (
    ContractResolution,
    IBAsyncHistoricalBarProvider,
    IBKRReadOnlySettings,
    IBKRSourceDataUnavailable,
    _contract_specs_for_request,
)
from trading_assistant_data.sources.ibkr.us_equity_read_only import UsEquityRefreshRequest


def test_physical_future_resolution_falls_back_to_local_symbol_after_error_200() -> None:
    provider = _provider()
    request = _request(start=datetime(2026, 1, 5, tzinfo=timezone.utc))
    spec = _spec(request, "NQH6")
    ib = _FakeIB(
        fail_qualifiers={
            ("lastTradeDateOrContractMonth", "202603"),
            ("lastTradeDateOrContractMonth", "20260320"),
        }
    )

    resolution = provider._resolve_physical_future(_FakeModule, ib, request, spec)

    assert resolution is not None
    assert resolution.method == "localSymbol"
    assert resolution.local_symbol == "NQH6"
    assert resolution.con_id == "1001"
    assert [
        (call.get("lastTradeDateOrContractMonth"), call.get("localSymbol"))
        for call in ib.qualify_calls
    ][:3] == [("202603", None), ("20260320", None), (None, "NQH6")]


def test_physical_future_resolution_falls_back_to_contract_details() -> None:
    provider = _provider()
    request = _request(
        start=datetime(2026, 5, 5, tzinfo=timezone.utc),
        end=datetime(2026, 5, 6, tzinfo=timezone.utc),
    )
    spec = _spec(request, "NQM6")
    ib = _FakeIB(
        fail_qualifiers={
            ("lastTradeDateOrContractMonth", "202606"),
            ("lastTradeDateOrContractMonth", "20260619"),
            ("localSymbol", "NQM6"),
        }
    )

    resolution = provider._resolve_physical_future(_FakeModule, ib, request, spec)

    assert resolution is not None
    assert resolution.method == "contractDetails_filtered"
    assert resolution.con_id == "1002"
    assert ib.contract_details_call_count == 1


def test_noncritical_buffer_contract_error_200_does_not_block_physical_panama_slice() -> None:
    provider = _provider()
    request = _request(
        start=datetime(2026, 5, 5, tzinfo=timezone.utc),
        end=datetime(2026, 5, 6, tzinfo=timezone.utc),
    )
    ib = _FakeIB(fail_local_symbols={"NQH6", "NQU6"})

    frame = provider._contract_chain_futures_bars(_FakeModule, ib, request)

    assert not frame.empty
    assert set(frame["source_contract"]) == {"NQM6"}
    assert frame["source_conid"].astype(str).str.strip().ne("").all()


def test_critical_roll_contract_error_200_fails_closed() -> None:
    provider = _provider()
    request = _request(
        start=datetime(2026, 5, 5, tzinfo=timezone.utc),
        end=datetime(2026, 5, 6, tzinfo=timezone.utc),
    )
    ib = _FakeIB(fail_local_symbols={"NQM6"})

    with pytest.raises(IBKRSourceDataUnavailable, match="NQM6"):
        provider._contract_chain_futures_bars(_FakeModule, ib, request)


def test_timeout_splits_physical_historical_request_window() -> None:
    provider = _provider()
    request = _request(
        start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        end=datetime(2026, 5, 10, tzinfo=timezone.utc),
    )
    resolution = ContractResolution(
        contract=_FakeContract(localSymbol="NQM6", conId=1002),
        local_symbol="NQM6",
        yyyymm="202606",
        yyyymmdd="20260619",
        con_id="1002",
        ib_local_symbol="NQM6",
        last_trade_date_or_contract_month="20260619",
        method="localSymbol",
    )
    ib = _FakeIB(timeout_first_request=True)

    frame = provider._request_physical_contract_window(
        _FakeModule,
        ib,
        resolution,
        request=request,
        start=request.start,
        end=request.end,
        critical=True,
    )

    assert not frame.empty
    assert ib.historical_call_count == 3
    assert len({call["durationStr"] for call in ib.historical_calls}) > 1


def test_physical_timeout_marks_required_contract_unavailable_after_split_floor() -> None:
    provider = _provider()
    request = _request(
        start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        end=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )
    resolution = ContractResolution(
        contract=_FakeContract(localSymbol="NQM6", conId=1002),
        local_symbol="NQM6",
        yyyymm="202606",
        yyyymmdd="20260619",
        con_id="1002",
        ib_local_symbol="NQM6",
        last_trade_date_or_contract_month="20260619",
        method="localSymbol",
    )
    ib = _FakeIB(always_timeout=True)

    with pytest.raises(IBKRSourceDataUnavailable, match="timeout"):
        provider._request_physical_contract_window(
            _FakeModule,
            ib,
            resolution,
            request=request,
            start=request.start,
            end=request.end,
            critical=True,
        )


def test_stock_resolution_falls_back_to_primary_exchange_after_error_200() -> None:
    provider = _provider()
    request = UsEquityRefreshRequest(
        symbol="QQQ",
        timeframe="1h",
        start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        end=datetime(2026, 5, 2, tzinfo=timezone.utc),
    ).normalized()
    ib = _FakeIB(fail_stock_primaries={"NASDAQ"})

    resolution = provider._resolve_stock_contract(_FakeModule, ib, request)

    assert resolution.con_id == "5001"
    assert resolution.method == "stock_primary_ARCA"
    assert resolution.primary_exchange == "ARCA"


def test_daily_chunks_advance_by_whole_dates() -> None:
    from trading_assistant_data.sources.ibkr.live_read_only import _request_chunks

    chunks = _request_chunks(
        datetime(2010, 2, 8, tzinfo=timezone.utc),
        datetime(2025, 2, 10, tzinfo=timezone.utc),
        "1d",
    )

    assert chunks[1][0].time().isoformat() == "00:00:00"
    assert chunks[1][0].date() > chunks[0][1].date()


def test_daily_chunks_keep_multi_year_stock_windows_together() -> None:
    from trading_assistant_data.sources.ibkr.live_read_only import _request_chunks

    chunks = _request_chunks(
        datetime(2021, 2, 8, tzinfo=timezone.utc),
        datetime(2026, 5, 1, tzinfo=timezone.utc),
        "1d",
    )

    assert len(chunks) == 1


def test_daily_historical_request_end_includes_declared_end_date() -> None:
    from trading_assistant_data.sources.ibkr.live_read_only import _historical_request_end

    assert _historical_request_end(
        datetime(2026, 5, 1, tzinfo=timezone.utc),
        "1d",
    ) == datetime(2026, 5, 2, tzinfo=timezone.utc)


def _provider() -> IBAsyncHistoricalBarProvider:
    return IBAsyncHistoricalBarProvider(
        IBKRReadOnlySettings(
            host="127.0.0.1",
            port=4002,
            client_id=900,
            timeout_seconds=1,
            pacing_sleep_seconds=0,
        )
    )


def _request(
    *,
    start: datetime,
    end: datetime | None = None,
) -> CmeNqRefreshRequest:
    return CmeNqRefreshRequest(
        symbol="NQ",
        timeframe="5m",
        start=start,
        end=end or datetime(2026, 6, 1, tzinfo=timezone.utc),
        contract_chain=("NQH6", "NQM6", "NQU6"),
    ).normalized()


def _spec(request: CmeNqRefreshRequest, local_symbol: str) -> dict:
    return next(
        spec
        for spec in _contract_specs_for_request(request)
        if spec["local_symbol"] == local_symbol
    )


class _FakeContract:
    def __init__(self, **kwargs: object) -> None:
        self.secType = "FUT"
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.symbol = getattr(self, "symbol", None)
        self.exchange = getattr(self, "exchange", "CME")
        self.currency = getattr(self, "currency", "USD")
        self.tradingClass = getattr(self, "tradingClass", None)
        self.localSymbol = getattr(self, "localSymbol", "")
        self.lastTradeDateOrContractMonth = getattr(
            self,
            "lastTradeDateOrContractMonth",
            "",
        )
        self.conId = getattr(self, "conId", 0)


class _FakeUtil:
    @staticmethod
    def df(rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)


class _FakeModule:
    Future = _FakeContract
    util = _FakeUtil()

    @staticmethod
    def Stock(
        symbol: str,
        exchange: str,
        currency: str,
        *,
        primaryExchange: str | None = None,
    ) -> _FakeContract:
        return _FakeContract(
            secType="STK",
            symbol=symbol,
            exchange=exchange,
            currency=currency,
            primaryExchange=primaryExchange or "",
            localSymbol=symbol,
        )


class _FakeIB:
    def __init__(
        self,
        *,
        fail_qualifiers: set[tuple[str, str]] | None = None,
        fail_local_symbols: set[str] | None = None,
        fail_stock_primaries: set[str] | None = None,
        timeout_first_request: bool = False,
        always_timeout: bool = False,
    ) -> None:
        self.fail_qualifiers = fail_qualifiers or set()
        self.fail_local_symbols = fail_local_symbols or set()
        self.fail_stock_primaries = fail_stock_primaries or set()
        self.timeout_first_request = timeout_first_request
        self.always_timeout = always_timeout
        self.qualify_calls: list[dict] = []
        self.historical_calls: list[dict] = []
        self.historical_call_count = 0
        self.contract_details_call_count = 0

    def qualifyContracts(self, contract: _FakeContract) -> list[_FakeContract]:
        if getattr(contract, "secType", "") == "STK":
            primary = str(getattr(contract, "primaryExchange", "") or "")
            if primary in self.fail_stock_primaries:
                raise RuntimeError("Error 200, reqId 1: No security definition has been found")
            contract.conId = contract.conId or 5001
            contract.localSymbol = contract.localSymbol or contract.symbol
            contract.primaryExchange = primary or "SMART"
            return [contract]
        payload = {
            "lastTradeDateOrContractMonth": contract.lastTradeDateOrContractMonth or None,
            "localSymbol": contract.localSymbol or None,
        }
        self.qualify_calls.append(payload)
        qualifier = (
            "localSymbol",
            contract.localSymbol,
        ) if contract.localSymbol else (
            "lastTradeDateOrContractMonth",
            contract.lastTradeDateOrContractMonth,
        )
        if qualifier in self.fail_qualifiers or contract.localSymbol in self.fail_local_symbols:
            raise RuntimeError("Error 200, reqId 1: No security definition has been found")
        if (
            not contract.localSymbol
            and _local_symbol_for_month(contract.lastTradeDateOrContractMonth)
            in self.fail_local_symbols
        ):
            raise RuntimeError("Error 200, reqId 1: No security definition has been found")
        if not contract.localSymbol:
            contract.localSymbol = _local_symbol_for_month(contract.lastTradeDateOrContractMonth)
        if not contract.lastTradeDateOrContractMonth:
            contract.lastTradeDateOrContractMonth = _expiry_for_local(contract.localSymbol)
        contract.conId = contract.conId or _con_id_for_local(contract.localSymbol)
        return [contract]

    def reqContractDetails(self, _contract: _FakeContract) -> list[SimpleNamespace]:
        self.contract_details_call_count += 1
        if "NQM6" in self.fail_local_symbols:
            return []
        return [
            SimpleNamespace(
                contract=_FakeContract(
                    symbol="NQ",
                    exchange="CME",
                    currency="USD",
                    tradingClass="NQ",
                    localSymbol="NQM6",
                    lastTradeDateOrContractMonth="20260619",
                    conId=1002,
                )
            )
        ]

    def reqHistoricalData(self, contract: _FakeContract, **kwargs: object) -> list[dict]:
        self.historical_call_count += 1
        self.historical_calls.append(dict(kwargs))
        if self.always_timeout or (self.timeout_first_request and self.historical_call_count == 1):
            raise TimeoutError("historical request timed out")
        local = contract.localSymbol or "NQM6"
        return [
            _bar("2026-05-03T00:00:00Z", local),
            _bar("2026-05-05T12:00:00Z", local),
            _bar("2026-05-08T00:00:00Z", local),
        ]


def _bar(ts: str, local_symbol: str) -> dict:
    base = 18_000.0 if local_symbol == "NQM6" else 17_900.0
    return {
        "date": pd.Timestamp(ts),
        "open": base,
        "high": base + 10.0,
        "low": base - 10.0,
        "close": base + 1.0,
        "volume": 1000.0,
    }


def _local_symbol_for_month(month: str) -> str:
    return {"202603": "NQH6", "202606": "NQM6", "202609": "NQU6"}[month[:6]]


def _expiry_for_local(local_symbol: str) -> str:
    return {"NQH6": "20260320", "NQM6": "20260619", "NQU6": "20260918"}[local_symbol]


def _con_id_for_local(local_symbol: str) -> int:
    return {"NQH6": 1001, "NQM6": 1002, "NQU6": 1003}[local_symbol]
