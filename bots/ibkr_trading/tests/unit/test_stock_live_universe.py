from strategies.stock.alcb.universe_constituents import SP500_CONSTITUENTS
from strategies.stock.live_universe import (
    BACKTESTED_INTRADAY_STOCK_SYMBOLS,
    LIVE_STOCK_UNIVERSE,
    LIVE_STOCK_UNIVERSE_ADDED_SYMBOLS,
    LIVE_STOCK_UNIVERSE_SYMBOLS,
)


def test_live_stock_universe_is_backtested_intraday_cohort_only() -> None:
    symbols = list(LIVE_STOCK_UNIVERSE_SYMBOLS)

    assert len(BACKTESTED_INTRADAY_STOCK_SYMBOLS) == 98
    assert symbols == list(BACKTESTED_INTRADAY_STOCK_SYMBOLS)
    assert LIVE_STOCK_UNIVERSE_ADDED_SYMBOLS == ()


def test_live_stock_universe_is_not_the_broad_sp500_list() -> None:
    symbols = [symbol for symbol, _, _ in LIVE_STOCK_UNIVERSE]

    assert len(symbols) == len(set(symbols)) == 98
    assert len(symbols) < len(SP500_CONSTITUENTS)
    assert len(LIVE_STOCK_UNIVERSE_ADDED_SYMBOLS) == 0
    assert all(sector and primary_exchange for _, sector, primary_exchange in LIVE_STOCK_UNIVERSE)


def test_live_stock_universe_preserves_metadata_for_backtested_symbols() -> None:
    metadata = {
        symbol: (sector, primary_exchange)
        for symbol, sector, primary_exchange in LIVE_STOCK_UNIVERSE
    }

    assert metadata["AAPL"][1] == "NASDAQ"
    assert metadata["WMT"][1] == "NASDAQ"
    assert metadata["IBM"][1] == "NYSE"
