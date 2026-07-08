"""Instrument definition."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str  # e.g. "MNQ" or "MNQH6"
    root: str  # e.g. "MNQ", "MGC"
    venue: str  # e.g. "CME", "COMEX"
    tick_size: float
    tick_value: float  # tick_size * multiplier
    multiplier: float
    currency: str = "USD"
    point_value: float = 0.0  # computed: multiplier
    contract_expiry: str = ""  # YYYYMM or YYYYMMDD for futures
    primary_exchange: str = ""  # e.g. "ARCA", "NASDAQ" for stocks
    sec_type: str = ""  # e.g. "STK", "FUT"
    trading_class: str = ""  # e.g. "NQ", "MNQ" for futures; ticker for stocks

    @property
    def exchange(self) -> str:
        """Alias for venue — used by broker layer (contract_factory)."""
        return self.venue

    def __post_init__(self):
        if self.point_value == 0.0:
            object.__setattr__(self, "point_value", self.multiplier)
