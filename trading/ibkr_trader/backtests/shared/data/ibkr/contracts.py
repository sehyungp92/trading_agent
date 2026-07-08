"""Compatibility wrapper around the shared live/backtest futures roll policy."""
from __future__ import annotations

from libs.market_data.futures_roll import (
    FUTURE_ROOTS,
    QUARTER_MONTHS,
    FutureRootSpec,
    FuturesContractSpec,
    active_contract,
    generate_quarterly_contracts,
    make_contract_spec,
    roll_schedule,
    root_spec,
    third_friday,
)

__all__ = [
    "FUTURE_ROOTS",
    "QUARTER_MONTHS",
    "FutureRootSpec",
    "FuturesContractSpec",
    "active_contract",
    "generate_quarterly_contracts",
    "make_contract_spec",
    "roll_schedule",
    "root_spec",
    "third_friday",
]
