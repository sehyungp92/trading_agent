"""Contract resolution and caching."""
from __future__ import annotations

import dataclasses
import logging
import time
from typing import TYPE_CHECKING, Any

from ib_async import Contract, Future, IB, Stock

from libs.market_data.futures_roll import active_contract_month, is_supported_quarterly_future

from ..config.schemas import ContractTemplate, ExchangeRoute
from ..models.types import IBContractSpec

if TYPE_CHECKING:
    from libs.oms.models.instrument import Instrument

logger = logging.getLogger(__name__)


class ContractResolutionError(Exception):
    pass


class ContractFactory:
    """Builds and caches IB Contract objects from config templates.

    Supports two resolution paths:
    - Template-based: resolve(symbol, expiry) using contracts.yaml templates
    - Instrument-based: resolve(symbol, expiry, instrument=inst) using rich Instrument metadata
    """

    def __init__(
        self,
        ib: IB,
        templates: dict[str, ContractTemplate],
        routes: dict[str, ExchangeRoute],
    ):
        self._ib = ib
        self._templates = templates
        self._routes = routes
        self._cache: dict[tuple[str, str, str, str, str], tuple[Contract, IBContractSpec, float]] = {}
        self._cache_ttl_s = 86400  # 24 hours
        self._logical_symbols_by_conid: dict[int, str] = {}
        self._logical_symbols_by_signature: dict[tuple[str, str, str, str], str] = {}

    async def resolve(
        self,
        symbol: str,
        expiry: str = "",
        instrument: Instrument | None = None,
    ) -> tuple[Contract, IBContractSpec]:
        """Resolve an instrument or config symbol to a qualified IB Contract.

        Args:
            symbol: root symbol key from contracts.yaml (e.g. "MNQ", "QQQ")
            expiry: YYYYMM or YYYYMMDD, optional for stocks
            instrument: Rich instrument metadata for dynamic symbols

        Returns:
            (qualified Contract, IBContractSpec with conId and metadata)

        Raises:
            ContractResolutionError if not found or ambiguous.
        """
        resolved_instrument = self._resolve_instrument(
            symbol=symbol,
            expiry=expiry,
            instrument=instrument,
        )
        if resolved_instrument is None:
            raise ContractResolutionError(f"Unknown symbol: {symbol}")

        cache_key = (
            resolved_instrument.sec_type,
            resolved_instrument.symbol,
            resolved_instrument.contract_expiry or "",
            resolved_instrument.exchange,
            resolved_instrument.primary_exchange or "",
        )
        cached = self._cache.get(cache_key)
        if cached and (time.monotonic() - cached[2]) < self._cache_ttl_s:
            self._register_contract_symbol(symbol, cached[0])
            return cached[0], cached[1]

        contract = self._build_contract(resolved_instrument)

        qualified = await self._ib.qualifyContractsAsync(contract)
        if not qualified:
            expiry_msg = (
                f" {resolved_instrument.contract_expiry}"
                if resolved_instrument.contract_expiry
                else ""
            )
            raise ContractResolutionError(
                f"Failed to qualify {resolved_instrument.symbol}{expiry_msg}"
            )

        q = qualified[0]
        spec = IBContractSpec(
            con_id=q.conId,
            symbol=q.symbol,
            sec_type=q.secType,
            exchange=q.exchange,
            currency=q.currency,
            multiplier=float(q.multiplier) if q.multiplier else resolved_instrument.multiplier,
            tick_size=resolved_instrument.tick_size,
            trading_class=q.tradingClass or resolved_instrument.trading_class,
            primary_exchange=(
                getattr(q, "primaryExchange", "") or resolved_instrument.primary_exchange
            ),
            last_trade_date=(
                getattr(q, "lastTradeDateOrContractMonth", "")
                or resolved_instrument.contract_expiry
            ),
        )
        self._cache[cache_key] = (q, spec, time.monotonic())
        self._register_contract_symbol(symbol, q)
        logger.debug(
            "Resolved %s %s %s -> conId=%s",
            resolved_instrument.sec_type,
            resolved_instrument.symbol,
            resolved_instrument.contract_expiry,
            spec.con_id,
        )
        return q, spec

    def build_contract(
        self,
        symbol: str,
        expiry: str = "",
        instrument: Instrument | None = None,
    ) -> Contract:
        resolved_instrument = self._resolve_instrument(
            symbol=symbol,
            expiry=expiry,
            instrument=instrument,
        )
        if resolved_instrument is None:
            raise ContractResolutionError(f"Unknown symbol: {symbol}")
        return self._build_contract(resolved_instrument)

    def logical_symbol_for_contract(self, contract: Contract | None) -> str:
        if contract is None:
            return ""
        con_id = int(getattr(contract, "conId", 0) or 0)
        if con_id and con_id in self._logical_symbols_by_conid:
            return self._logical_symbols_by_conid[con_id]

        signature = self._contract_signature(contract)
        logical_symbol = self._logical_symbols_by_signature.get(signature)
        if logical_symbol:
            return logical_symbol

        broker_symbol = str(getattr(contract, "symbol", "") or "").upper()
        for logical, template in self._templates.items():
            if template.symbol.upper() == broker_symbol:
                return logical
        return broker_symbol

    def _instrument_from_template(
        self,
        symbol: str,
        template: ContractTemplate | None,
        route: ExchangeRoute | None,
        expiry: str,
    ) -> Instrument | None:
        """Build an Instrument from config templates (fallback when no Instrument provided)."""
        if template is None:
            return None
        from libs.oms.models.instrument import Instrument
        resolved_expiry = expiry
        if template.sec_type == "FUT" and not resolved_expiry and is_supported_quarterly_future(symbol):
            resolved_expiry = active_contract_month(symbol)
        return Instrument(
            symbol=template.symbol,
            root=symbol,
            venue=(route.exchange if route else template.exchange),
            tick_size=template.tick_size,
            tick_value=template.tick_value,
            multiplier=template.multiplier,
            currency=template.currency,
            contract_expiry=resolved_expiry if template.sec_type == "FUT" else "",
            sec_type=template.sec_type,
            primary_exchange=(route.primary_exchange if route and route.primary_exchange else template.primary_exchange or ""),
            trading_class=(route.trading_class if route and route.trading_class else template.trading_class or ""),
        )

    def _resolve_instrument(
        self,
        symbol: str,
        expiry: str,
        instrument: Instrument | None,
    ) -> Instrument | None:
        template = self._templates.get(symbol)
        route = self._routes.get(symbol)
        if instrument is None:
            return self._instrument_from_template(symbol, template, route, expiry)

        overrides: dict[str, Any] = {
            "root": instrument.root or symbol,
            "contract_expiry": (
                expiry if (template and template.sec_type == "FUT" and expiry) else instrument.contract_expiry
            ),
        }
        if template is not None:
            overrides.update(
                {
                    "symbol": template.symbol,
                    "sec_type": template.sec_type or instrument.sec_type,
                    "venue": template.exchange or instrument.exchange,
                    "currency": template.currency or instrument.currency,
                    "multiplier": template.multiplier or instrument.multiplier,
                    "tick_size": template.tick_size or instrument.tick_size,
                    "tick_value": template.tick_value or instrument.tick_value,
                    "trading_class": template.trading_class or instrument.trading_class,
                }
            )
            if template.primary_exchange:
                overrides["primary_exchange"] = template.primary_exchange
        if route is not None:
            overrides["venue"] = route.exchange or overrides.get("venue", instrument.exchange)
            if route.primary_exchange:
                overrides["primary_exchange"] = route.primary_exchange
            if route.trading_class:
                overrides["trading_class"] = route.trading_class
        resolved_root = str(overrides.get("root") or symbol or instrument.root or instrument.symbol)
        resolved_sec_type = str(overrides.get("sec_type") or instrument.sec_type or "").upper()
        if not resolved_sec_type and is_supported_quarterly_future(resolved_root):
            resolved_sec_type = "FUT"
            overrides["sec_type"] = "FUT"
        if (
            resolved_sec_type == "FUT"
            and not overrides.get("contract_expiry")
            and is_supported_quarterly_future(resolved_root)
        ):
            overrides["contract_expiry"] = active_contract_month(resolved_root)
            if not overrides.get("trading_class"):
                overrides["trading_class"] = resolved_root.upper()
        return dataclasses.replace(instrument, **overrides)

    @staticmethod
    def _contract_signature(contract: Contract) -> tuple[str, str, str, str]:
        return (
            str(getattr(contract, "symbol", "") or "").upper(),
            str(getattr(contract, "secType", "") or "").upper(),
            str(getattr(contract, "exchange", "") or "").upper(),
            str(getattr(contract, "primaryExchange", "") or "").upper(),
        )

    def _register_contract_symbol(self, logical_symbol: str, contract: Contract) -> None:
        con_id = int(getattr(contract, "conId", 0) or 0)
        if con_id:
            self._logical_symbols_by_conid[con_id] = logical_symbol
        self._logical_symbols_by_signature[self._contract_signature(contract)] = logical_symbol

    @staticmethod
    def _build_contract(instrument: Instrument) -> Contract:
        if instrument.sec_type == "STK":
            contract = Stock(
                symbol=instrument.symbol,
                exchange=instrument.exchange,
                currency=instrument.currency,
                primaryExchange=instrument.primary_exchange or "",
            )
        elif instrument.sec_type == "FUT":
            contract = Future(
                symbol=instrument.root or instrument.symbol,
                exchange=instrument.exchange,
                currency=instrument.currency,
                lastTradeDateOrContractMonth=instrument.contract_expiry,
            )
        else:
            contract = Contract(
                symbol=instrument.symbol,
                secType=instrument.sec_type,
                exchange=instrument.exchange,
                currency=instrument.currency,
            )
            if instrument.contract_expiry:
                contract.lastTradeDateOrContractMonth = instrument.contract_expiry
            if instrument.primary_exchange:
                contract.primaryExchange = instrument.primary_exchange

        if instrument.trading_class:
            contract.tradingClass = instrument.trading_class
        return contract

    def invalidate(
        self,
        symbol: str,
        expiry: str = "",
        sec_type: str = "FUT",
        exchange: str = "",
        primary_exchange: str = "",
    ) -> None:
        """Force cache eviction (e.g. on rollover)."""
        template = self._templates.get(symbol)
        exchange_name = exchange or (template.exchange if template else "")
        cache_key = (sec_type, symbol, expiry, exchange_name, primary_exchange)
        self._cache.pop(cache_key, None)

    def clear_cache(self) -> None:
        """Clear all cached contracts."""
        self._cache.clear()
        self._logical_symbols_by_conid.clear()
        self._logical_symbols_by_signature.clear()
