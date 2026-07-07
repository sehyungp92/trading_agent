from __future__ import annotations

import argparse
from pathlib import Path

from backtests.scalp.config_ivb_auction import IvbAuctionBacktestConfig
from backtests.scalp.config_po3_reversal import Po3ReversalBacktestConfig
from backtests.scalp.engine.ivb_auction_engine import load_ivb_auction_data, run_ivb_auction_backtest
from backtests.scalp.engine.po3_reversal_engine import load_po3_reversal_data, run_po3_reversal_backtest


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        import sys

        argv = sys.argv[1:]
    if argv and argv[0] == "download":
        from backtests.scalp.data.downloader import main as download_main

        return download_main(argv[1:])

    parser = argparse.ArgumentParser(description="Run scalp backtests.")
    parser.add_argument("strategy", choices=["ivb_auction", "po3_reversal"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--analysis-symbol", default="NQ")
    parser.add_argument("--trade-symbol", default="MNQ")
    parser.add_argument("--symbol", default=None, help="Deprecated alias for --analysis-symbol.")
    parser.add_argument("--confirmation-symbol", default="ES")
    args = parser.parse_args(argv)
    trade_symbol = args.trade_symbol.upper()
    analysis_symbol = (args.symbol or args.analysis_symbol).upper()
    if args.strategy == "ivb_auction":
        config = IvbAuctionBacktestConfig(
            analysis_symbol=analysis_symbol,
            trade_symbol=trade_symbol,
            data_dir=args.data_dir,
            initial_equity=args.initial_equity,
        )
        result = run_ivb_auction_backtest(load_ivb_auction_data(config), config)
    else:
        config = Po3ReversalBacktestConfig(
            analysis_symbol=analysis_symbol,
            trade_symbol=trade_symbol,
            confirmation_symbol=args.confirmation_symbol,
            data_dir=args.data_dir,
            initial_equity=args.initial_equity,
        )
        result = run_po3_reversal_backtest(load_po3_reversal_data(config), config)
    metrics = result.metrics
    print(
        f"{args.strategy}: trades={int(metrics.get('total_trades', 0))} "
        f"net={metrics.get('net_profit', 0.0):+.2f} "
        f"pf={metrics.get('profit_factor', 0.0):.2f} "
        f"dd={metrics.get('max_drawdown_pct', 0.0):.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
