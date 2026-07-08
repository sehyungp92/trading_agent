"""Quarterly futures roll policy shared by live analysis and execution.

The momentum backtests use a conservative Panama convention: roll quarterly CME
index futures four calendar days before the third Friday expiry. Live code uses
the same policy here so analysis bars, routed contracts, and roll blackouts all
agree.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

QUARTER_MONTHS: tuple[tuple[int, str], ...] = (
    (3, "H"),
    (6, "M"),
    (9, "U"),
    (12, "Z"),
)

ROLL_DAYS_BEFORE_EXPIRY = 4
ROLL_BLACKOUT_DAYS_BEFORE = 1
ROLL_BLACKOUT_DAYS_AFTER = 4


@dataclass(frozen=True)
class FutureRootSpec:
    symbol: str
    exchange: str = "CME"
    trading_class: str | None = None
    currency: str = "USD"
    tick_size: float = 0.25
    point_value: float = 1.0

    @property
    def ib_trading_class(self) -> str:
        return self.trading_class or self.symbol


FUTURE_ROOTS: dict[str, FutureRootSpec] = {
    "NQ": FutureRootSpec("NQ", exchange="CME", trading_class="NQ", tick_size=0.25, point_value=20.0),
    "MNQ": FutureRootSpec("MNQ", exchange="CME", trading_class="MNQ", tick_size=0.25, point_value=2.0),
    "ES": FutureRootSpec("ES", exchange="CME", trading_class="ES", tick_size=0.25, point_value=50.0),
}


@dataclass(frozen=True)
class FuturesContractSpec:
    symbol: str
    yyyymm: str
    expiry: date
    roll_date: date
    code: str
    local_symbol: str
    exchange: str = "CME"
    trading_class: str | None = None
    currency: str = "USD"
    tick_size: float = 0.25

    @property
    def ib_trading_class(self) -> str:
        return self.trading_class or self.symbol


def normalize_root(symbol: str | Any) -> str:
    """Return the supported futures root from a string, contract, or Instrument."""
    if not isinstance(symbol, str):
        root = getattr(symbol, "root", "") or getattr(symbol, "symbol", "")
        symbol = str(root or "")
    normalized = symbol.upper().strip()
    for suffix in ("-FUT", ".FUT"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized


def root_spec(symbol: str | Any) -> FutureRootSpec:
    normalized = normalize_root(symbol)
    return FUTURE_ROOTS.get(normalized, FutureRootSpec(normalized, trading_class=normalized))


def is_supported_quarterly_future(symbol: str | Any) -> bool:
    return normalize_root(symbol) in FUTURE_ROOTS


def third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    days_to_friday = (4 - first.weekday()) % 7
    return first + timedelta(days=days_to_friday, weeks=2)


def make_contract_spec(
    symbol: str,
    year: int,
    month: int,
    *,
    roll_days_before_expiry: int = ROLL_DAYS_BEFORE_EXPIRY,
) -> FuturesContractSpec:
    root = root_spec(symbol)
    code = dict(QUARTER_MONTHS)[month]
    expiry = third_friday(year, month)
    roll_date = expiry - timedelta(days=roll_days_before_expiry)
    return FuturesContractSpec(
        symbol=root.symbol,
        yyyymm=f"{year}{month:02d}",
        expiry=expiry,
        roll_date=roll_date,
        code=code,
        local_symbol=f"{root.symbol}{code}{year % 10}",
        exchange=root.exchange,
        trading_class=root.ib_trading_class,
        currency=root.currency,
        tick_size=root.tick_size,
    )


def generate_quarterly_contracts(
    symbol: str,
    *,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
    years: int = 2,
    as_of: date | datetime | None = None,
    include_buffer_contracts: bool = True,
) -> list[FuturesContractSpec]:
    """Return contracts whose active windows overlap the requested span."""
    as_of_date = _coerce_date(as_of) or datetime.now(timezone.utc).date()
    end_date = _coerce_date(end) or as_of_date
    start_date = _coerce_date(start) or (end_date - timedelta(days=365 * years))

    all_contracts: list[FuturesContractSpec] = []
    for year in range(start_date.year - 1, end_date.year + 2):
        for month, _code in QUARTER_MONTHS:
            all_contracts.append(make_contract_spec(symbol, year, month))
    all_contracts.sort(key=lambda contract: contract.expiry)

    relevant: list[FuturesContractSpec] = []
    for idx, contract in enumerate(all_contracts):
        previous_roll = all_contracts[idx - 1].roll_date if idx > 0 else date(1900, 1, 1)
        active_start = previous_roll
        active_end = contract.roll_date
        if active_end >= start_date and active_start <= end_date:
            relevant.append(contract)

    if include_buffer_contracts and relevant:
        first_idx = all_contracts.index(relevant[0])
        last_idx = all_contracts.index(relevant[-1])
        relevant = all_contracts[max(0, first_idx - 1) : min(len(all_contracts), last_idx + 2)]

    return relevant


def roll_schedule(contracts: list[FuturesContractSpec]) -> list[tuple[date, str, str]]:
    ordered = sorted(contracts, key=lambda contract: contract.expiry)
    return [
        (old.roll_date, old.yyyymm, new.yyyymm)
        for old, new in zip(ordered, ordered[1:])
    ]


def active_contract(
    contracts: list[FuturesContractSpec],
    *,
    as_of: date | datetime | None = None,
) -> FuturesContractSpec | None:
    if not contracts:
        return None
    as_of_date = _coerce_date(as_of) or datetime.now(timezone.utc).date()
    ordered = sorted(contracts, key=lambda contract: contract.roll_date)
    for contract in ordered:
        if as_of_date < contract.roll_date:
            return contract
    return ordered[-1]


def active_contract_month(symbol: str | Any, *, as_of: date | datetime | None = None) -> str:
    root = normalize_root(symbol)
    contracts = generate_quarterly_contracts(root, years=1, as_of=as_of)
    contract = active_contract(contracts, as_of=as_of)
    return contract.yyyymm if contract else ""


def active_contract_spec(symbol: str | Any, *, as_of: date | datetime | None = None) -> FuturesContractSpec | None:
    root = normalize_root(symbol)
    contracts = generate_quarterly_contracts(root, years=1, as_of=as_of)
    return active_contract(contracts, as_of=as_of)


def is_roll_blackout(
    symbol: str | Any,
    *,
    as_of: date | datetime | None = None,
    days_before: int = ROLL_BLACKOUT_DAYS_BEFORE,
    days_after: int = ROLL_BLACKOUT_DAYS_AFTER,
) -> bool:
    return roll_blackout_context(
        symbol,
        as_of=as_of,
        days_before=days_before,
        days_after=days_after,
    ) is not None


def roll_blackout_context(
    symbol: str | Any,
    *,
    as_of: date | datetime | None = None,
    days_before: int = ROLL_BLACKOUT_DAYS_BEFORE,
    days_after: int = ROLL_BLACKOUT_DAYS_AFTER,
) -> dict[str, Any] | None:
    root = normalize_root(symbol)
    if root not in FUTURE_ROOTS:
        return None
    as_of_date = _coerce_date(as_of) or datetime.now(timezone.utc).date()
    contracts = generate_quarterly_contracts(
        root,
        start=as_of_date - timedelta(days=120),
        end=as_of_date + timedelta(days=120),
        include_buffer_contracts=True,
    )
    by_month = {contract.yyyymm: contract for contract in contracts}
    for roll_date, old_month, new_month in roll_schedule(contracts):
        start = roll_date - timedelta(days=days_before)
        end = roll_date + timedelta(days=days_after)
        if start <= as_of_date <= end:
            old_contract = by_month.get(old_month)
            new_contract = by_month.get(new_month)
            return {
                "root": root,
                "roll_date": roll_date,
                "blackout_start": start,
                "blackout_end": end,
                "old_month": old_month,
                "new_month": new_month,
                "old_local_symbol": old_contract.local_symbol if old_contract else old_month,
                "new_local_symbol": new_contract.local_symbol if new_contract else new_month,
            }
    return None


def roll_blackout_reason(symbol: str | Any, *, as_of: date | datetime | None = None) -> str | None:
    context = roll_blackout_context(symbol, as_of=as_of)
    if context is None:
        return None
    return (
        f"Futures roll blackout for {context['root']}: "
        f"{context['old_month']} -> {context['new_month']} on {context['roll_date'].isoformat()}; "
        f"new entries disabled from {context['blackout_start'].isoformat()} "
        f"through {context['blackout_end'].isoformat()}"
    )


def roll_force_flatten_reason(symbol: str | Any, *, as_of: date | datetime | None = None) -> str | None:
    """Return a reason to flatten open positions before the roll/expiry window.

    The same window used to block new entries is also the period where existing
    positions should be flattened rather than managed off the newly-rolled
    analysis series.
    """
    context = roll_blackout_context(symbol, as_of=as_of)
    if context is None:
        return None
    return (
        f"Futures roll safety exit for {context['root']}: "
        f"{context['old_month']} -> {context['new_month']} on {context['roll_date'].isoformat()}; "
        f"open positions must be flat during {context['blackout_start'].isoformat()} "
        f"through {context['blackout_end'].isoformat()}"
    )


def with_active_contract_expiry(instrument: Any, *, as_of: date | datetime | None = None) -> Any:
    """Populate blank supported futures instruments with the active contract month."""
    return with_contract_expiry_for_order(instrument, order_role="ENTRY", as_of=as_of)


def contract_month_for_order(
    symbol: str | Any,
    *,
    order_role: str = "ENTRY",
    as_of: date | datetime | None = None,
) -> str:
    """Return the month to use for a routed order.

    Entries always use the active month. During the roll blackout, non-entry
    orders keep using the old month so protective exits for any pre-roll
    position do not accidentally target the new contract. New entries are
    blocked for this whole period by the OMS.
    """
    root = normalize_root(symbol)
    role = str(order_role or "").upper()
    if role != "ENTRY":
        context = roll_blackout_context(root, as_of=as_of)
        if context is not None:
            return str(context["old_month"])
    return active_contract_month(root, as_of=as_of)


def with_contract_expiry_for_order(
    instrument: Any,
    *,
    order_role: str = "ENTRY",
    as_of: date | datetime | None = None,
) -> Any:
    """Populate blank supported futures instruments with a role-aware month."""
    if instrument is None:
        return None
    root = normalize_root(instrument)
    sec_type = str(getattr(instrument, "sec_type", "") or "").upper()
    expiry = str(getattr(instrument, "contract_expiry", "") or "")
    if root not in FUTURE_ROOTS or expiry:
        return instrument

    updates: dict[str, Any] = {
        "contract_expiry": contract_month_for_order(root, order_role=order_role, as_of=as_of),
        "trading_class": getattr(instrument, "trading_class", "") or root,
    }
    if not sec_type:
        updates["sec_type"] = "FUT"
    try:
        return dataclasses.replace(instrument, **updates)
    except TypeError:
        for key, value in updates.items():
            if hasattr(instrument, key):
                setattr(instrument, key, value)
        return instrument


def _coerce_date(value: date | datetime | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    return value
