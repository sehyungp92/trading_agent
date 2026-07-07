"""Small value objects for shared IBKR data downloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path


@dataclass(frozen=True)
class ConnectionSettings:
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 107
    timeout: int = 60


@dataclass(frozen=True)
class BarDownloadRequest:
    symbol: str
    timeframe: str
    sec_type: str = "FUT"
    exchange: str = "CME"
    trading_class: str | None = None
    currency: str = "USD"
    primary_exchange: str = ""
    what_to_show: str = "TRADES"
    use_rth: bool = False
    duration: str = "2 Y"
    start: datetime | None = None
    end: datetime | None = None
    output_dir: Path = Path("data/raw")
    family: str = ""
    allow_contfuture_legacy: bool = False

    @property
    def ib_trading_class(self) -> str:
        return self.trading_class or self.symbol


@dataclass(frozen=True)
class TickDownloadRequest:
    symbol: str
    start: datetime
    end: datetime
    tick_type: str = "TRADES"
    exchange: str = "CME"
    trading_class: str | None = None
    use_rth: bool = False
    ignore_size: bool = False
    session_start: time | None = None
    session_end: time | None = None
    output_dir: Path = Path("data/raw")

    @property
    def ib_trading_class(self) -> str:
        return self.trading_class or self.symbol


@dataclass
class DownloadResult:
    symbol: str
    timeframe: str = ""
    what_to_show: str = ""
    rows: int = 0
    start: datetime | None = None
    end: datetime | None = None
    paths: list[Path] = field(default_factory=list)
    dry_run: bool = False
    messages: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DownloadWindow:
    start: datetime
    end: datetime
    duration: str


@dataclass(frozen=True)
class Gap:
    start: datetime
    end: datetime
    expected_frequency: str

