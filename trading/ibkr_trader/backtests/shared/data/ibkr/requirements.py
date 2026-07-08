"""Family-level data requirements for shared IBKR sync."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BarRequirement:
    family: str
    symbol: str
    timeframe: str
    sec_type: str
    exchange: str
    trading_class: str | None = None
    what_to_show: str = "TRADES"
    use_rth: bool = False
    output_dir: Path = Path("data/raw")
    duration: str = "2 Y"
    primary_exchange: str = ""


def family_bar_requirements(family: str, *, years: int = 2) -> list[BarRequirement]:
    family = family.lower()
    duration = f"{years} Y"
    if family == "scalp":
        output = Path("data/raw")
        return [
            BarRequirement("scalp", symbol, timeframe, "FUT", "CME", symbol, "TRADES", timeframe == "1d", output, duration)
            for symbol in ("NQ", "ES")
            for timeframe in ("1m", "5m", "1h", "1d")
        ] + [
            BarRequirement("scalp", symbol, "1m", "FUT", "CME", symbol, "BID_ASK", False, output, duration)
            for symbol in ("NQ", "ES")
        ]
    if family == "momentum":
        output = Path("backtests/momentum/data/raw")
        return [
            BarRequirement("momentum", "NQ", "5m", "FUT", "CME", "NQ", "TRADES", False, output, duration),
            BarRequirement("momentum", "MNQ", "5m", "FUT", "CME", "MNQ", "TRADES", False, output, duration),
            BarRequirement("momentum", "ES", "1d", "FUT", "CME", "ES", "TRADES", True, output, duration),
        ]
    if family == "swing":
        output = Path("backtests/swing/data/raw")
        primary = {"QQQ": "NASDAQ", "GLD": "ARCA"}
        return [
            BarRequirement(
                "swing",
                symbol,
                timeframe,
                "STK",
                "SMART",
                None,
                "TRADES",
                timeframe == "1d",
                output,
                duration,
                primary_exchange=primary.get(symbol, ""),
            )
            for symbol in ("QQQ", "GLD")
            for timeframe in ("15m", "1h", "1d")
        ]
    if family == "stock":
        output = Path("backtests/stock/data/raw")
        symbols = [(symbol, primary) for symbol, primary in _stock_symbols_with_primary() if symbol != "VIX"]
        return [
            BarRequirement(
                "stock", symbol, "1d", "STK", "SMART", None, "TRADES", True, output, duration,
                primary_exchange=primary,
            )
            for symbol, primary in symbols
        ] + [
            BarRequirement("stock", "VIX", "1d", "IND", "CBOE", None, "TRADES", True, output, duration)
        ]
    raise ValueError(f"Unknown data family: {family}")


def _stock_symbols() -> list[str]:
    try:
        from backtests.stock.data.downloader import get_all_download_symbols

        return get_all_download_symbols()
    except Exception:
        return ["SPY", "QQQ", "VIX", "HYG"]


def _stock_symbols_with_primary() -> list[tuple[str, str]]:
    try:
        from backtests.stock.data.downloader import REFERENCE_SYMBOLS
        from strategies.stock.alcb.universe_constituents import SP500_CONSTITUENTS

        symbols: list[tuple[str, str]] = [(symbol, exchange) for symbol, _, exchange in SP500_CONSTITUENTS]
        existing = {symbol for symbol, _ in symbols}
        reference_primary = {
            "SPY": "ARCA",
            "HYG": "ARCA",
            "XLK": "ARCA",
            "XLV": "ARCA",
            "XLF": "ARCA",
            "XLY": "ARCA",
            "XLP": "ARCA",
            "XLE": "ARCA",
            "XLB": "ARCA",
            "XLI": "ARCA",
            "XLU": "ARCA",
            "XLRE": "ARCA",
            "XLC": "ARCA",
        }
        for symbol in REFERENCE_SYMBOLS:
            if symbol not in existing:
                symbols.append((symbol, reference_primary.get(symbol, "")))
                existing.add(symbol)
        return symbols
    except Exception:
        return [(symbol, "") for symbol in _stock_symbols()]
