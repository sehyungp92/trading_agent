"""Focused live stock universe shared by stock strategies.

The broad ``SP500_CONSTITUENTS`` list remains available for historical
backtest/research tooling. Live stock selection is intentionally limited to
the intraday backtest cohort while IB historical-data pacing is stabilized.
"""

from __future__ import annotations

from strategies.stock.alcb.universe_constituents import SP500_CONSTITUENTS

# Backtest cohort with local 5m and/or 30m bars in backtests/stock/data/raw.
BACKTESTED_INTRADAY_STOCK_SYMBOLS: tuple[str, ...] = (
    "A",
    "AAPL",
    "ABBV",
    "ABT",
    "ACN",
    "ADBE",
    "ADI",
    "AMAT",
    "AMD",
    "AMGN",
    "AMZN",
    "AVGO",
    "BAC",
    "BDX",
    "BIO",
    "BLK",
    "BRK B",
    "BSX",
    "CAT",
    "CDNS",
    "CDW",
    "CI",
    "COR",
    "CRM",
    "CRWD",
    "CSCO",
    "DHR",
    "DXCM",
    "ELV",
    "EPAM",
    "EW",
    "FSLR",
    "FTNT",
    "GEN",
    "GILD",
    "GOOG",
    "GOOGL",
    "GS",
    "HCA",
    "HD",
    "HPE",
    "HPQ",
    "HSIC",
    "IBM",
    "IDXX",
    "INTU",
    "IQV",
    "ISRG",
    "IT",
    "JNJ",
    "JPM",
    "KEYS",
    "KLAC",
    "LH",
    "LLY",
    "LRCX",
    "MA",
    "MCHP",
    "MCK",
    "MDT",
    "META",
    "MPWR",
    "MRK",
    "MSFT",
    "MSI",
    "MTD",
    "MU",
    "NFLX",
    "NOW",
    "NTAP",
    "NVDA",
    "NXPI",
    "ON",
    "ORCL",
    "PANW",
    "PFE",
    "PTC",
    "QCOM",
    "QRVO",
    "REGN",
    "RMD",
    "ROP",
    "SNPS",
    "SWKS",
    "SYK",
    "TDY",
    "TECH",
    "TER",
    "TMO",
    "TRMB",
    "TSLA",
    "TXN",
    "UNH",
    "V",
    "VRTX",
    "WMT",
    "ZBRA",
    "ZTS",
)

# Current component snapshots checked on 2026-05-03. These lists are kept for
# future expansion, but are not part of the active live universe for now.
NASDAQ_100_SYMBOLS: tuple[str, ...] = (
    "ADBE",
    "AMD",
    "ABNB",
    "ALNY",
    "GOOGL",
    "GOOG",
    "AMZN",
    "AEP",
    "AMGN",
    "ADI",
    "AAPL",
    "AMAT",
    "APP",
    "ARM",
    "ASML",
    "ADSK",
    "ADP",
    "AXON",
    "BKR",
    "BKNG",
    "AVGO",
    "CDNS",
    "CHTR",
    "CTAS",
    "CSCO",
    "CCEP",
    "CTSH",
    "CMCSA",
    "CEG",
    "CPRT",
    "CSGP",
    "COST",
    "CRWD",
    "CSX",
    "DDOG",
    "DXCM",
    "FANG",
    "DASH",
    "EA",
    "EXC",
    "FAST",
    "FER",
    "FTNT",
    "GEHC",
    "GILD",
    "HON",
    "IDXX",
    "INSM",
    "INTC",
    "INTU",
    "ISRG",
    "KDP",
    "KLAC",
    "KHC",
    "LRCX",
    "LIN",
    "MAR",
    "MRVL",
    "MELI",
    "META",
    "MCHP",
    "MU",
    "MSFT",
    "MSTR",
    "MDLZ",
    "MPWR",
    "MNST",
    "NFLX",
    "NVDA",
    "NXPI",
    "ORLY",
    "ODFL",
    "PCAR",
    "PLTR",
    "PANW",
    "PAYX",
    "PYPL",
    "PDD",
    "PEP",
    "QCOM",
    "REGN",
    "ROP",
    "ROST",
    "SNDK",
    "STX",
    "SHOP",
    "SBUX",
    "SNPS",
    "TMUS",
    "TTWO",
    "TSLA",
    "TXN",
    "TRI",
    "VRSK",
    "VRTX",
    "WMT",
    "WBD",
    "WDC",
    "WDAY",
    "XEL",
    "ZS",
)

DOW_JONES_SYMBOLS: tuple[str, ...] = (
    "MMM",
    "AXP",
    "AMGN",
    "AMZN",
    "AAPL",
    "BA",
    "CAT",
    "CVX",
    "CSCO",
    "KO",
    "DIS",
    "GS",
    "HD",
    "HON",
    "IBM",
    "JNJ",
    "JPM",
    "MCD",
    "MRK",
    "MSFT",
    "NKE",
    "NVDA",
    "PG",
    "CRM",
    "SHW",
    "TRV",
    "UNH",
    "VZ",
    "V",
    "WMT",
)

_INDEX_METADATA_OVERRIDES: dict[str, tuple[str, str]] = {
    "ABNB": ("Consumer Discretionary", "NASDAQ"),
    "ADSK": ("Technology", "NASDAQ"),
    "ALNY": ("Health Care", "NASDAQ"),
    "APP": ("Technology", "NASDAQ"),
    "ARM": ("Technology", "NASDAQ"),
    "ASML": ("Technology", "NASDAQ"),
    "AXON": ("Industrials", "NASDAQ"),
    "CCEP": ("Consumer Staples", "NASDAQ"),
    "CSGP": ("Real Estate", "NASDAQ"),
    "CTSH": ("Technology", "NASDAQ"),
    "DASH": ("Technology", "NASDAQ"),
    "DDOG": ("Technology", "NASDAQ"),
    "FER": ("Industrials", "NASDAQ"),
    "GEHC": ("Health Care", "NASDAQ"),
    "INSM": ("Health Care", "NASDAQ"),
    "INTC": ("Technology", "NASDAQ"),
    "KHC": ("Consumer Staples", "NASDAQ"),
    "MELI": ("Consumer Discretionary", "NASDAQ"),
    "MRVL": ("Technology", "NASDAQ"),
    "MSTR": ("Technology", "NASDAQ"),
    "ODFL": ("Industrials", "NASDAQ"),
    "PDD": ("Technology", "NASDAQ"),
    "PLTR": ("Technology", "NASDAQ"),
    "PYPL": ("Industrials", "NASDAQ"),
    "SHOP": ("Technology", "NASDAQ"),
    "SNDK": ("Technology", "NASDAQ"),
    "STX": ("Technology", "NASDAQ"),
    "TRI": ("Technology", "NASDAQ"),
    "WDAY": ("Technology", "NASDAQ"),
    "WDC": ("Technology", "NASDAQ"),
    "ZS": ("Technology", "NASDAQ"),
}

_DOW_JONES_PRIMARY_EXCHANGES: dict[str, str] = {
    "AAPL": "NASDAQ",
    "AMGN": "NASDAQ",
    "AMZN": "NASDAQ",
    "AXP": "NYSE",
    "BA": "NYSE",
    "CAT": "NYSE",
    "CRM": "NYSE",
    "CSCO": "NASDAQ",
    "CVX": "NYSE",
    "DIS": "NYSE",
    "GS": "NYSE",
    "HD": "NYSE",
    "HON": "NASDAQ",
    "IBM": "NYSE",
    "JNJ": "NYSE",
    "JPM": "NYSE",
    "KO": "NYSE",
    "MCD": "NYSE",
    "MMM": "NYSE",
    "MRK": "NYSE",
    "MSFT": "NASDAQ",
    "NKE": "NYSE",
    "NVDA": "NASDAQ",
    "PG": "NYSE",
    "SHW": "NYSE",
    "TRV": "NYSE",
    "UNH": "NYSE",
    "V": "NYSE",
    "VZ": "NYSE",
    "WMT": "NASDAQ",
}

_SP500_METADATA: dict[str, tuple[str, str]] = {
    symbol: (sector, primary_exchange)
    for symbol, sector, primary_exchange in SP500_CONSTITUENTS
}
_BACKTESTED_INTRADAY_SYMBOL_SET: frozenset[str] = frozenset(
    BACKTESTED_INTRADAY_STOCK_SYMBOLS
)
_NASDAQ_100_SYMBOL_SET: frozenset[str] = frozenset(NASDAQ_100_SYMBOLS)


def _unique_symbols(*groups: tuple[str, ...]) -> tuple[str, ...]:
    symbols: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for raw_symbol in group:
            symbol = raw_symbol.upper()
            if symbol in seen:
                continue
            seen.add(symbol)
            symbols.append(symbol)
    return tuple(symbols)


def _metadata_for_symbol(symbol: str) -> tuple[str, str]:
    if symbol in _SP500_METADATA:
        sector, primary_exchange = _SP500_METADATA[symbol]
    else:
        sector, primary_exchange = _INDEX_METADATA_OVERRIDES.get(symbol, ("Unknown", "NASDAQ"))

    if symbol in _NASDAQ_100_SYMBOL_SET:
        primary_exchange = "NASDAQ"
    else:
        primary_exchange = _DOW_JONES_PRIMARY_EXCHANGES.get(symbol, primary_exchange)
    return sector, primary_exchange


def _constituents_for(symbols: tuple[str, ...]) -> tuple[tuple[str, str, str], ...]:
    rows: list[tuple[str, str, str]] = []
    for symbol in symbols:
        sector, primary_exchange = _metadata_for_symbol(symbol)
        rows.append((symbol, sector, primary_exchange))
    return tuple(rows)


LIVE_STOCK_UNIVERSE_SYMBOLS: tuple[str, ...] = _unique_symbols(
    BACKTESTED_INTRADAY_STOCK_SYMBOLS,
)

LIVE_STOCK_UNIVERSE_ADDED_SYMBOLS: tuple[str, ...] = tuple(
    symbol
    for symbol in LIVE_STOCK_UNIVERSE_SYMBOLS
    if symbol not in _BACKTESTED_INTRADAY_SYMBOL_SET
)

LIVE_STOCK_UNIVERSE: tuple[tuple[str, str, str], ...] = _constituents_for(
    LIVE_STOCK_UNIVERSE_SYMBOLS
)
