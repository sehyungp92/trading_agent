"""Live read-only IBKR historical market-data provider.

The provider deliberately exposes only historical bar retrieval. It does not expose
orders, accounts, positions, or live trading state.
"""

from __future__ import annotations

import importlib
import math
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from trading_assistant_data.sources.ibkr.cme_nq_read_only import CmeNqRefreshRequest
from trading_assistant_data.sources.ibkr.us_equity_read_only import UsEquityRefreshRequest
from trading_assistant_data.transforms.panama import stitch_panama

MONTH_CODES = {3: "H", 6: "M", 9: "U", 12: "Z"}
CODE_MONTHS = {value: key for key, value in MONTH_CODES.items()}
EMPTY_BAR_COLUMNS = [
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source_contract",
    "source_conid",
    "source_local_symbol",
    "source_primary_exchange",
    "contract_resolution_method",
    "contract_last_trade_date",
]


class IBKRSourceDataUnavailable(RuntimeError):
    """Raised when an approval-grade source slice cannot be constructed."""


@dataclass(frozen=True)
class ContractResolution:
    contract: Any
    local_symbol: str
    yyyymm: str
    yyyymmdd: str
    con_id: str
    ib_local_symbol: str
    last_trade_date_or_contract_month: str
    method: str


@dataclass(frozen=True)
class StockContractResolution:
    contract: Any
    symbol: str
    con_id: str
    local_symbol: str
    primary_exchange: str
    method: str


@dataclass(frozen=True)
class HistoricalRequestOutcome:
    status: str
    bars: Any = None
    error: str = ""


@dataclass(frozen=True)
class IBKRReadOnlySettings:
    host: str
    port: int
    client_id: int
    timeout_seconds: int = 60
    pacing_sleep_seconds: float = 12.0

    @classmethod
    def from_env(cls) -> "IBKRReadOnlySettings":
        ack = os.getenv("IBKR_READ_ONLY_ACK", "").strip().lower()
        if ack not in {"1", "true", "yes", "read_only", "read-only"}:
            raise RuntimeError("IBKR_READ_ONLY_ACK must confirm read-only market-data use")
        return cls(
            host=os.getenv("IBKR_HOST", "127.0.0.1").strip(),
            port=int(os.getenv("IBKR_PORT", "4002")),
            client_id=int(os.getenv("IBKR_CLIENT_ID", "107")),
            timeout_seconds=int(os.getenv("IBKR_TIMEOUT_SECONDS", "60")),
            pacing_sleep_seconds=float(os.getenv("IBKR_PACING_SLEEP_SECONDS", "12.0")),
        )


class IBAsyncHistoricalBarProvider:
    """IBKR provider backed by ib_async's historical market-data API only."""

    def __init__(self, settings: IBKRReadOnlySettings | None = None) -> None:
        self.settings = settings or IBKRReadOnlySettings.from_env()
        self._last_request_monotonic: float | None = None
        self._physical_contract_cache: dict[tuple[str, str, str], ContractResolution] = {}
        self._stock_contract_cache: dict[tuple[str, str, str, str], StockContractResolution] = {}

    def historical_bars(self, request: CmeNqRefreshRequest | UsEquityRefreshRequest) -> pd.DataFrame:
        module = _load_ib_async()
        ib = module.IB()
        try:
            ib.connect(
                self.settings.host,
                self.settings.port,
                clientId=self.settings.client_id,
                timeout=self.settings.timeout_seconds,
            )
            if isinstance(request, CmeNqRefreshRequest):
                return self._futures_bars(module, ib, request.normalized())
            return self._stock_bars(module, ib, request.normalized())
        finally:
            if ib.isConnected():
                ib.disconnect()

    def _futures_bars(self, module: Any, ib: Any, request: CmeNqRefreshRequest) -> pd.DataFrame:
        return self._contract_chain_futures_bars(module, ib, request)

    def _contract_chain_futures_bars(
        self,
        module: Any,
        ib: Any,
        request: CmeNqRefreshRequest,
    ) -> pd.DataFrame:
        specs = _contract_specs_for_request(request)
        if not specs:
            raise IBKRSourceDataUnavailable(
                f"no physical contract chain specs for {request.symbol} {request.timeframe}"
            )
        contract_data: dict[str, pd.DataFrame] = {}
        failures: list[str] = []
        for index, spec in enumerate(specs):
            resolution = self._resolve_physical_future(module, ib, request, spec)
            if resolution is None:
                message = f"IBKR could not resolve physical futures contract {spec['local_symbol']}"
                if spec["critical"]:
                    raise IBKRSourceDataUnavailable(message)
                failures.append(message)
                continue
            window = _contract_request_window(specs, index, request)
            if window is None:
                continue
            try:
                frame = self._request_physical_contract_window(
                    module,
                    ib,
                    resolution,
                    request=request,
                    start=window[0],
                    end=window[1],
                    critical=spec["critical"],
                )
            except IBKRSourceDataUnavailable:
                if spec["critical"]:
                    raise
                continue
            if not frame.empty:
                contract_data[spec["yyyymm"]] = _frame_for_panama(frame)
        missing_critical = _missing_critical_specs(contract_data, specs)
        if missing_critical:
            raise IBKRSourceDataUnavailable(
                "missing required physical futures contracts: " + ", ".join(missing_critical)
            )
        if not contract_data:
            raise IBKRSourceDataUnavailable(
                "no physical futures contract data returned"
                + (f"; skipped non-critical contracts: {', '.join(failures)}" if failures else "")
            )
        rolls = _roll_pairs_for_specs(specs)
        critical_rolls = _missing_critical_roll_data(contract_data, rolls, request)
        if critical_rolls:
            raise IBKRSourceDataUnavailable(
                "missing critical futures roll data: " + ", ".join(critical_rolls)
            )
        stitched = stitch_panama(
            contract_data,
            rolls,
            fail_closed=True,
            min_gap_points=_panama_min_gap_points(request.symbol),
        )
        if stitched.empty:
            raise IBKRSourceDataUnavailable("Panama futures stitch returned no usable bars")
        stitched = stitched.reset_index().rename(columns={"index": "timestamp_utc"})
        if "timestamp_utc" not in stitched.columns:
            stitched = stitched.rename(columns={stitched.columns[0]: "timestamp_utc"})
        stitched["timestamp_utc"] = pd.to_datetime(stitched["timestamp_utc"], utc=True)
        start = pd.Timestamp(request.start)
        end = pd.Timestamp(request.end)
        stitched = stitched[
            (stitched["timestamp_utc"] >= start) & (stitched["timestamp_utc"] <= end)
        ]
        return _bar_frame(stitched)

    def _resolve_physical_future(
        self,
        module: Any,
        ib: Any,
        request: CmeNqRefreshRequest,
        spec: dict[str, Any],
    ) -> ContractResolution | None:
        cache_key = (request.exchange, request.symbol, spec["yyyymm"])
        cached = self._physical_contract_cache.get(cache_key)
        if cached is not None:
            return cached

        candidates = [
            (
                "lastTradeDateOrContractMonth_yyyymm",
                module.Future(
                    symbol=request.symbol,
                    exchange=request.exchange,
                    currency=request.currency,
                    tradingClass=request.symbol,
                    lastTradeDateOrContractMonth=spec["yyyymm"],
                    includeExpired=True,
                ),
            ),
            (
                "lastTradeDateOrContractMonth_yyyymmdd",
                module.Future(
                    symbol=request.symbol,
                    exchange=request.exchange,
                    currency=request.currency,
                    tradingClass=request.symbol,
                    lastTradeDateOrContractMonth=spec["yyyymmdd"],
                    includeExpired=True,
                ),
            ),
            (
                "localSymbol",
                module.Future(
                    localSymbol=spec["local_symbol"],
                    exchange=request.exchange,
                    currency=request.currency,
                    includeExpired=True,
                ),
            ),
        ]
        for method, contract in candidates:
            resolution = self._qualify_resolution(ib, contract, spec, method)
            if resolution is not None:
                self._physical_contract_cache[cache_key] = resolution
                return resolution

        details_resolution = self._resolve_from_contract_details(module, ib, request, spec)
        if details_resolution is not None:
            self._physical_contract_cache[cache_key] = details_resolution
            return details_resolution
        return None

    def _qualify_resolution(
        self,
        ib: Any,
        contract: Any,
        spec: dict[str, Any],
        method: str,
    ) -> ContractResolution | None:
        try:
            qualified = ib.qualifyContracts(contract)
            qualified_contract = _qualified_contract(qualified, contract)
        except Exception as exc:
            if _classify_ib_error(exc) == "unknown_contract":
                return None
            return None
        if not getattr(qualified_contract, "conId", 0):
            return None
        qualified_contract.includeExpired = True
        return _resolution_from_contract(qualified_contract, spec, method)

    def _resolve_from_contract_details(
        self,
        module: Any,
        ib: Any,
        request: CmeNqRefreshRequest,
        spec: dict[str, Any],
    ) -> ContractResolution | None:
        broad = module.Future(
            symbol=request.symbol,
            exchange=request.exchange,
            currency=request.currency,
            tradingClass=request.symbol,
            includeExpired=True,
        )
        try:
            details = ib.reqContractDetails(broad)
        except Exception:
            return None
        for detail in details or []:
            contract = getattr(detail, "contract", detail)
            if not _contract_matches_spec(contract, request, spec):
                continue
            try:
                qualified = ib.qualifyContracts(contract)
                qualified_contract = _qualified_contract(qualified, contract)
            except Exception:
                qualified_contract = contract
            if not getattr(qualified_contract, "conId", 0):
                continue
            qualified_contract.includeExpired = True
            return _resolution_from_contract(
                qualified_contract,
                spec,
                "contractDetails_filtered",
            )
        return None

    def _request_physical_contract_window(
        self,
        module: Any,
        ib: Any,
        resolution: ContractResolution,
        *,
        request: CmeNqRefreshRequest,
        start: datetime,
        end: datetime,
        critical: bool,
    ) -> pd.DataFrame:
        if end < start:
            return _empty_bar_frame()
        frame = self._request_physical_contract_window_split(
            module,
            ib,
            resolution,
            request=request,
            start=start,
            end=end,
            critical=critical,
            depth=0,
        )
        if frame.empty and critical:
            raise IBKRSourceDataUnavailable(
                f"no bars for required physical contract {resolution.local_symbol} "
                f"{start.isoformat()}..{end.isoformat()}"
            )
        return frame

    def _request_physical_contract_window_split(
        self,
        module: Any,
        ib: Any,
        resolution: ContractResolution,
        *,
        request: CmeNqRefreshRequest,
        start: datetime,
        end: datetime,
        critical: bool,
        depth: int,
    ) -> pd.DataFrame:
        outcome = self._request_with_status(
            ib,
            resolution.contract,
            end_datetime=_ib_end_datetime(end),
            duration=_ib_duration(start, end),
            timeframe=request.timeframe,
            what_to_show=request.what_to_show,
            use_rth=request.use_rth,
        )
        if outcome.status == "ok":
            return self._bars_to_frame(
                module,
                outcome.bars,
                start=start,
                end=end,
                resolution=resolution,
            )
        if outcome.status == "no_data":
            return _empty_bar_frame()
        if outcome.status == "timeout" and _can_split_request(start, end, request.timeframe):
            midpoint = start + (end - start) / 2
            left = self._request_physical_contract_window_split(
                module,
                ib,
                resolution,
                request=request,
                start=start,
                end=midpoint,
                critical=critical,
                depth=depth + 1,
            )
            right = self._request_physical_contract_window_split(
                module,
                ib,
                resolution,
                request=request,
                start=midpoint + timedelta(seconds=1),
                end=end,
                critical=critical,
                depth=depth + 1,
            )
            return _merge_frames([left, right])
        if critical:
            raise IBKRSourceDataUnavailable(
                f"IBKR historical request failed for required contract {resolution.local_symbol}: "
                f"{outcome.status}: {outcome.error}"
            )
        return _empty_bar_frame()

    def _request_with_status(
        self,
        ib: Any,
        contract: Any,
        *,
        end_datetime: str,
        duration: str,
        timeframe: str,
        what_to_show: str,
        use_rth: bool,
    ) -> HistoricalRequestOutcome:
        last_error: Exception | None = None
        for attempt in range(1, 5):
            try:
                self._pace_request()
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=end_datetime,
                    durationStr=duration,
                    barSizeSetting=_ib_bar_size(timeframe),
                    whatToShow=what_to_show,
                    useRTH=use_rth,
                    formatDate=2,
                    keepUpToDate=False,
                    timeout=max(float(self.settings.timeout_seconds), 120.0),
                )
                if bars:
                    return HistoricalRequestOutcome("ok", bars=bars)
                return HistoricalRequestOutcome("no_data")
            except Exception as exc:
                last_error = exc
                kind = _classify_ib_error(exc)
                if kind in {"no_data", "unknown_contract", "continuous_future_not_allowed"}:
                    return HistoricalRequestOutcome(kind, error=str(exc))
                if kind == "timeout":
                    return HistoricalRequestOutcome(kind, error=str(exc))
                if kind == "pacing":
                    time.sleep(65)
                elif attempt < 4:
                    time.sleep(5 * attempt)
        if last_error is None:
            return HistoricalRequestOutcome("no_data")
        return HistoricalRequestOutcome(_classify_ib_error(last_error), error=str(last_error))

    def _bars_to_frame(
        self,
        module: Any,
        bars: Any,
        *,
        start: datetime,
        end: datetime,
        resolution: ContractResolution,
    ) -> pd.DataFrame:
        frame = _bars_to_dataframe(module, bars)
        if frame.empty:
            return _empty_bar_frame()
        frame = frame.rename(columns={"date": "timestamp_utc"})
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
        frame = frame[
            (frame["timestamp_utc"] >= pd.Timestamp(start))
            & (frame["timestamp_utc"] <= pd.Timestamp(end))
        ]
        frame["source_contract"] = resolution.local_symbol
        frame["source_conid"] = resolution.con_id
        frame["contract_resolution_method"] = resolution.method
        frame["contract_last_trade_date"] = resolution.last_trade_date_or_contract_month
        return _bar_frame(frame)

    def _stock_bars(
        self,
        module: Any,
        ib: Any,
        request: UsEquityRefreshRequest,
    ) -> pd.DataFrame:
        resolution = self._resolve_stock_contract(module, ib, request)
        frames = [
            self._request_bars(
                module,
                ib,
                resolution.contract,
                start=start,
                end=end,
                timeframe=request.timeframe,
                what_to_show=request.what_to_show,
                use_rth=request.use_rth,
                source_contract=request.symbol,
                contract_resolution_method=resolution.method,
            )
            for start, end in _request_chunks(request.start, request.end, request.timeframe)
        ]
        return _merge_frames(frames)

    def _resolve_stock_contract(
        self,
        module: Any,
        ib: Any,
        request: UsEquityRefreshRequest,
    ) -> StockContractResolution:
        cache_key = (
            request.symbol,
            request.exchange,
            request.currency,
            request.primary_exchange,
        )
        cached = self._stock_contract_cache.get(cache_key)
        if cached is not None:
            return cached
        candidates = list(_stock_contract_candidates(module, request))
        for method, contract in candidates:
            resolution = self._qualify_stock_candidate(ib, contract, request, method)
            if resolution is not None:
                self._stock_contract_cache[cache_key] = resolution
                return resolution
        for method, contract in candidates:
            resolution = self._resolve_stock_from_contract_details(
                ib,
                contract,
                request,
                method,
            )
            if resolution is not None:
                self._stock_contract_cache[cache_key] = resolution
                return resolution
        raise RuntimeError(
            "IBKR could not qualify read-only stock historical-data contract "
            f"{request.symbol} {request.exchange} {request.currency}"
        )

    def _qualify_stock_candidate(
        self,
        ib: Any,
        contract: Any,
        request: UsEquityRefreshRequest,
        method: str,
    ) -> StockContractResolution | None:
        try:
            qualified = ib.qualifyContracts(contract)
            qualified_contract = _qualified_contract(qualified, contract)
        except Exception as exc:
            if _classify_ib_error(exc) == "unknown_contract":
                return None
            return None
        if not getattr(qualified_contract, "conId", 0):
            return None
        if not _stock_contract_matches_request(qualified_contract, request):
            return None
        return _stock_resolution_from_contract(qualified_contract, request, method)

    def _resolve_stock_from_contract_details(
        self,
        ib: Any,
        contract: Any,
        request: UsEquityRefreshRequest,
        method: str,
    ) -> StockContractResolution | None:
        try:
            details = ib.reqContractDetails(contract)
        except Exception:
            return None
        for detail in details or []:
            candidate = getattr(detail, "contract", detail)
            if not _stock_contract_matches_request(candidate, request):
                continue
            try:
                qualified = ib.qualifyContracts(candidate)
                qualified_contract = _qualified_contract(qualified, candidate)
            except Exception:
                qualified_contract = candidate
            if not getattr(qualified_contract, "conId", 0):
                continue
            return _stock_resolution_from_contract(
                qualified_contract,
                request,
                f"{method}_contractDetails",
            )
        return None

    def _request_bars(
        self,
        module: Any,
        ib: Any,
        contract: Any,
        *,
        start: datetime,
        end: datetime,
        timeframe: str,
        what_to_show: str,
        use_rth: bool,
        source_contract: str,
        contract_resolution_method: str = "qualifyContracts",
        end_datetime: str | None = None,
        duration: str | None = None,
    ) -> pd.DataFrame:
        if getattr(contract, "conId", 0):
            qualified_contract = contract
        else:
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                raise RuntimeError(f"IBKR could not qualify contract: {contract!r}")
            qualified_contract = _qualified_contract(qualified, contract)
            if getattr(contract, "includeExpired", False):
                qualified_contract.includeExpired = True
        bars = self._request_with_retry(
            ib,
            qualified_contract,
            end_datetime=end_datetime
            if end_datetime is not None
            else _ib_end_datetime(_historical_request_end(end, timeframe)),
            duration=duration or _ib_duration(start, end),
            timeframe=timeframe,
            what_to_show=what_to_show,
            use_rth=use_rth,
        )
        frame = module.util.df(bars)
        if frame is None or frame.empty:
            return _empty_bar_frame()
        frame = frame.rename(columns={"date": "timestamp_utc"})
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
        frame = frame[(frame["timestamp_utc"] >= pd.Timestamp(start)) & (frame["timestamp_utc"] <= pd.Timestamp(end))]
        frame["source_contract"] = source_contract
        frame["source_conid"] = str(getattr(qualified_contract, "conId", "") or "")
        frame["source_local_symbol"] = str(
            getattr(qualified_contract, "localSymbol", "") or source_contract
        )
        frame["source_primary_exchange"] = str(
            getattr(qualified_contract, "primaryExchange", "") or ""
        )
        frame["contract_resolution_method"] = contract_resolution_method
        return _bar_frame(frame)

    def _request_with_retry(
        self,
        ib: Any,
        contract: Any,
        *,
        end_datetime: str,
        duration: str,
        timeframe: str,
        what_to_show: str,
        use_rth: bool,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, 5):
            try:
                self._pace_request()
                return ib.reqHistoricalData(
                    contract,
                    endDateTime=end_datetime,
                    durationStr=duration,
                    barSizeSetting=_ib_bar_size(timeframe),
                    whatToShow=what_to_show,
                    useRTH=use_rth,
                    formatDate=2,
                    keepUpToDate=False,
                    timeout=max(float(self.settings.timeout_seconds), 120.0),
                )
            except Exception as exc:
                last_error = exc
                kind = _classify_ib_error(exc)
                if kind in {"no_data", "unknown_contract", "continuous_future_not_allowed"}:
                    return []
                if kind == "timeout" and getattr(contract, "includeExpired", False):
                    return []
                if kind == "pacing":
                    time.sleep(65)
                elif attempt < 4:
                    time.sleep(5 * attempt)
        if last_error is not None:
            raise last_error
        return []

    def _pace_request(self) -> None:
        if self._last_request_monotonic is not None:
            elapsed = time.monotonic() - self._last_request_monotonic
            remaining = self.settings.pacing_sleep_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_monotonic = time.monotonic()


def _resolution_from_contract(
    contract: Any,
    spec: dict[str, Any],
    method: str,
) -> ContractResolution:
    last_trade = str(getattr(contract, "lastTradeDateOrContractMonth", "") or "")
    local_symbol = str(getattr(contract, "localSymbol", "") or spec["local_symbol"])
    return ContractResolution(
        contract=contract,
        local_symbol=spec["local_symbol"],
        yyyymm=spec["yyyymm"],
        yyyymmdd=spec["yyyymmdd"],
        con_id=str(getattr(contract, "conId", "") or ""),
        ib_local_symbol=local_symbol,
        last_trade_date_or_contract_month=last_trade,
        method=method,
    )


def _contract_matches_spec(contract: Any, request: CmeNqRefreshRequest, spec: dict[str, Any]) -> bool:
    symbol = str(getattr(contract, "symbol", "") or "").upper()
    trading_class = str(getattr(contract, "tradingClass", "") or "").upper()
    local_symbol = str(getattr(contract, "localSymbol", "") or "").upper()
    last_trade = str(getattr(contract, "lastTradeDateOrContractMonth", "") or "")
    if symbol and symbol != request.symbol:
        return False
    if trading_class and trading_class != request.symbol:
        return False
    return (
        local_symbol == spec["local_symbol"]
        or last_trade.startswith(spec["yyyymm"])
        or last_trade.startswith(spec["yyyymmdd"])
    )


def _stock_contract_candidates(
    module: Any,
    request: UsEquityRefreshRequest,
) -> list[tuple[str, Any]]:
    primaries = _stock_primary_exchange_candidates(request.symbol, request.primary_exchange)
    candidates: list[tuple[str, Any]] = []
    for primary in primaries:
        candidates.append(
            (
                f"stock_primary_{primary}" if primary else "stock_no_primary",
                module.Stock(
                    request.symbol,
                    request.exchange,
                    request.currency,
                    primaryExchange=primary or None,
                ),
            )
        )
    return candidates


def _stock_primary_exchange_candidates(symbol: str, requested: str) -> list[str]:
    defaults = {
        "AAPL": ("NASDAQ",),
        "GLD": ("ARCA", "NYSE", "AMEX"),
        "MSFT": ("NASDAQ",),
        "QQQ": ("NASDAQ", "ARCA"),
        "SPY": ("ARCA", "NYSE"),
    }
    values: list[str] = []
    requested_values = (requested,) if requested.strip() else ()
    for value in (
        *requested_values,
        *defaults.get(symbol.upper(), ()),
        "NASDAQ",
        "ARCA",
        "NYSE",
        "AMEX",
        "ISLAND",
        "",
    ):
        cleaned = value.upper().strip() if value else ""
        if cleaned not in values:
            values.append(cleaned)
    return values


def _stock_contract_matches_request(contract: Any, request: UsEquityRefreshRequest) -> bool:
    symbol = str(getattr(contract, "symbol", "") or "").upper()
    local_symbol = str(getattr(contract, "localSymbol", "") or "").upper()
    sec_type = str(getattr(contract, "secType", "") or request.sec_type).upper()
    currency = str(getattr(contract, "currency", "") or request.currency).upper()
    if sec_type and sec_type != request.sec_type:
        return False
    if currency and currency != request.currency:
        return False
    return request.symbol in {symbol, local_symbol}


def _stock_resolution_from_contract(
    contract: Any,
    request: UsEquityRefreshRequest,
    method: str,
) -> StockContractResolution:
    return StockContractResolution(
        contract=contract,
        symbol=request.symbol,
        con_id=str(getattr(contract, "conId", "") or ""),
        local_symbol=str(getattr(contract, "localSymbol", "") or request.symbol),
        primary_exchange=str(getattr(contract, "primaryExchange", "") or ""),
        method=method,
    )


def _contract_specs_for_request(request: CmeNqRefreshRequest) -> list[dict[str, Any]]:
    symbols = list(request.normalized().contract_chain)
    specs = []
    base_decade = (request.start.year // 10) * 10
    for local_symbol in symbols:
        expiry = _expiry_for_local_symbol(local_symbol, base_decade)
        if expiry is None:
            continue
        yyyymm = f"{expiry.year}{expiry.month:02d}"
        specs.append(
            {
                "local_symbol": local_symbol,
                "yyyymm": yyyymm,
                "yyyymmdd": expiry.strftime("%Y%m%d"),
                "expiry": expiry,
                "roll_date": expiry - timedelta(days=4),
            }
        )
    specs.sort(key=lambda item: item["expiry"])
    critical_indices = [
        index
        for index, _spec in enumerate(specs)
        if _contract_active_overlaps_request(specs, index, request)
    ]
    if not critical_indices:
        return []
    selected_indices = set(critical_indices)
    selected_indices.add(max(0, min(critical_indices) - 1))
    selected_indices.add(min(len(specs) - 1, max(critical_indices) + 1))
    selected: list[dict[str, Any]] = []
    for index in sorted(selected_indices):
        spec = dict(specs[index])
        spec["critical"] = index in critical_indices
        selected.append(spec)
    return selected


def _contract_active_overlaps_request(
    specs: list[dict[str, Any]],
    index: int,
    request: CmeNqRefreshRequest,
) -> bool:
    previous_roll = specs[index - 1]["roll_date"] if index > 0 else date(2000, 1, 1)
    active_end = specs[index]["roll_date"]
    return active_end > request.start.date() and previous_roll <= request.end.date()


def _contract_request_window(
    specs: list[dict[str, Any]],
    index: int,
    request: CmeNqRefreshRequest,
) -> tuple[datetime, datetime] | None:
    buffer = _contract_window_buffer(request.timeframe)
    segment_start = request.start
    if index > 0:
        segment_start = datetime.combine(
            specs[index - 1]["roll_date"],
            datetime.min.time(),
            tzinfo=request.start.tzinfo,
        )
    segment_end = datetime.combine(
        specs[index]["roll_date"],
        datetime.min.time(),
        tzinfo=request.end.tzinfo,
    )
    window_start = max(request.start, segment_start) - buffer
    window_end = min(request.end, segment_end) + buffer
    if window_end < window_start:
        return None
    return window_start, window_end


def _contract_window_buffer(timeframe: str) -> timedelta:
    if timeframe.lower() in {"1d", "daily"}:
        return timedelta(days=10)
    return timedelta(days=5)


def _roll_pairs_for_specs(specs: list[dict[str, Any]]) -> list[tuple[date, str, str]]:
    rolls: list[tuple[date, str, str]] = []
    for old, new in zip(specs, specs[1:], strict=False):
        rolls.append((old["roll_date"], old["yyyymm"], new["yyyymm"]))
    return rolls


def _missing_critical_specs(
    contract_data: dict[str, pd.DataFrame],
    specs: list[dict[str, Any]],
) -> list[str]:
    return [
        spec["local_symbol"]
        for spec in specs
        if spec["critical"]
        and (spec["yyyymm"] not in contract_data or contract_data[spec["yyyymm"]].empty)
    ]


def _missing_critical_roll_data(
    contract_data: dict[str, pd.DataFrame],
    rolls: list[tuple[date, str, str]],
    request: CmeNqRefreshRequest,
) -> list[str]:
    missing: list[str] = []
    for roll_date, old_month, new_month in rolls:
        roll_ts = pd.Timestamp(datetime.combine(roll_date, datetime.min.time()), tz="UTC")
        if roll_ts < pd.Timestamp(request.start) or roll_ts > pd.Timestamp(request.end):
            continue
        old_frame = contract_data.get(old_month)
        new_frame = contract_data.get(new_month)
        old_before = (
            pd.DataFrame()
            if old_frame is None or old_frame.empty
            else old_frame[old_frame.index < roll_ts]
        )
        new_after = (
            pd.DataFrame()
            if new_frame is None or new_frame.empty
            else new_frame[new_frame.index >= roll_ts]
        )
        if old_before.empty or new_after.empty:
            missing.append(f"{old_month}->{new_month}@{roll_date.isoformat()}")
    return missing


def _frame_for_panama(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True)
    result = result.set_index("timestamp_utc").sort_index()
    return result


def _panama_min_gap_points(symbol: str) -> float:
    return {
        "ES": 500.0,
        "MES": 500.0,
        "NQ": 500.0,
        "MNQ": 500.0,
    }.get(symbol.upper(), 250.0)


def _expiry_for_local_symbol(local_symbol: str, base_decade: int) -> date | None:
    if len(local_symbol) < 3:
        return None
    code = local_symbol[-2]
    if code not in CODE_MONTHS:
        return None
    year_digit = int(local_symbol[-1])
    year = base_decade + year_digit
    if year < base_decade:
        year += 10
    return _third_friday(year, CODE_MONTHS[code])


def _third_friday(year: int, month: int) -> date:
    current = date(year, month, 1)
    first_friday_offset = (4 - current.weekday()) % 7
    return current + timedelta(days=first_friday_offset + 14)


def _load_ib_async() -> Any:
    try:
        return importlib.import_module("ib_async")
    except ImportError as exc:
        raise RuntimeError("install trading-assistant-data[ibkr] to enable live IBKR sync") from exc


def _qualified_contract(qualified: list[Any], requested: Any) -> Any:
    if not qualified or qualified[0] is None:
        raise RuntimeError(f"IBKR could not qualify historical-data contract: {requested!r}")
    return qualified[0]


def _classify_ib_error(exc: Exception) -> str:
    message = str(exc).lower()
    if "error 200" in message or "unknown contract" in message or "no security definition" in message:
        return "unknown_contract"
    if "no data" in message or "hmds query returned no data" in message:
        return "no_data"
    if "continuous future security type is not allowed" in message:
        return "continuous_future_not_allowed"
    if "timeout" in message or "timed out" in message:
        return "timeout"
    if "pacing" in message or "error 162" in message or " 162" in message:
        return "pacing"
    return "error"


def _bars_to_dataframe(module: Any, bars: Any) -> pd.DataFrame:
    util = getattr(module, "util", None)
    if util is not None and hasattr(util, "df"):
        frame = util.df(bars)
        if frame is not None:
            return frame
    rows: list[dict[str, Any]] = []
    for bar in bars or []:
        rows.append(
            {
                "date": _bar_value(bar, "date"),
                "open": _bar_value(bar, "open"),
                "high": _bar_value(bar, "high"),
                "low": _bar_value(bar, "low"),
                "close": _bar_value(bar, "close"),
                "volume": _bar_value(bar, "volume", 0.0),
            }
        )
    return pd.DataFrame(rows)


def _bar_value(bar: Any, field: str, default: Any = None) -> Any:
    if isinstance(bar, dict):
        return bar.get(field, default)
    return getattr(bar, field, default)


def _bar_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_bar_frame()
    result = frame.copy()
    for column in EMPTY_BAR_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True)
    return result.loc[:, EMPTY_BAR_COLUMNS].dropna(subset=["timestamp_utc", "open", "high", "low", "close"])


def _empty_bar_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EMPTY_BAR_COLUMNS)


def _can_split_request(start: datetime, end: datetime, timeframe: str) -> bool:
    min_days = {
        "1m": 1,
        "5m": 3,
        "15m": 5,
        "30m": 7,
        "1h": 14,
        "4h": 30,
        "1d": 90,
        "daily": 90,
    }.get(timeframe.lower(), 7)
    return (end - start) > timedelta(days=min_days)


def _request_chunks(start: datetime, end: datetime, timeframe: str) -> list[tuple[datetime, datetime]]:
    max_days = {
        "1m": 7,
        "5m": 30,
        "15m": 90,
        "30m": 120,
        "1h": 180,
        "4h": 365,
        "1d": 3650,
        "daily": 3650,
    }.get(timeframe.lower(), 30)
    chunks: list[tuple[datetime, datetime]] = []
    current = pd.Timestamp(start).to_pydatetime()
    final = pd.Timestamp(end).to_pydatetime()
    daily = timeframe.lower() in {"1d", "daily"}
    while current <= final:
        chunk_end = min(current + timedelta(days=max_days), final)
        chunks.append((current, chunk_end))
        current = chunk_end + (timedelta(days=1) if daily else timedelta(seconds=1))
    return chunks


def _ib_duration(start: datetime, end: datetime) -> str:
    seconds = max(1, int((pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()))
    days = max(1, math.ceil(seconds / 86_400))
    if days >= 365:
        return f"{math.ceil(days / 365)} Y"
    return f"{days} D"


def _ib_bar_size(timeframe: str) -> str:
    return {
        "1m": "1 min",
        "5m": "5 mins",
        "15m": "15 mins",
        "30m": "30 mins",
        "1h": "1 hour",
        "4h": "4 hours",
        "1d": "1 day",
    }[timeframe.lower()]


def _ib_end_datetime(value: datetime) -> str:
    return pd.Timestamp(value).tz_convert("UTC").strftime("%Y%m%d %H:%M:%S UTC")


def _historical_request_end(value: datetime, timeframe: str) -> datetime:
    if timeframe.lower() in {"1d", "daily"}:
        return pd.Timestamp(value).to_pydatetime() + timedelta(days=1)
    return value


def _merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    populated = [frame for frame in frames if frame is not None and not frame.empty]
    if not populated:
        return _empty_bar_frame()
    merged = pd.concat(populated, ignore_index=True)
    merged["timestamp_utc"] = pd.to_datetime(merged["timestamp_utc"], utc=True)
    return (
        merged.sort_values("timestamp_utc")
        .drop_duplicates(subset=["timestamp_utc"], keep="last")
        .reset_index(drop=True)
    )
