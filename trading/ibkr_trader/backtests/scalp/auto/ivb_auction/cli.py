from __future__ import annotations

import argparse
from pathlib import Path

from backtests.shared.auto.phase_runner import PhaseRunner

from .plugin import IvbAuctionPlugin


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run IVB auction phased optimization.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("backtests/output/scalp/ivb_auction"))
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--analysis-symbol", default="NQ")
    parser.add_argument("--trade-symbol", default="MNQ")
    args = parser.parse_args(argv)
    runner = PhaseRunner(
        IvbAuctionPlugin(
            args.data_dir,
            args.initial_equity,
            analysis_symbol=args.analysis_symbol,
            trade_symbol=args.trade_symbol,
        ),
        args.output_dir,
    )
    runner.run_all_phases()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
