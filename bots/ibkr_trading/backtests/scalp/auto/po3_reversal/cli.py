from __future__ import annotations

import argparse
from pathlib import Path

from backtests.shared.auto.phase_runner import PhaseRunner

from .plugin import Po3ReversalPlugin


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PO3 reversal phased optimization.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("backtests/output/scalp/po3_reversal"))
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--analysis-symbol", default="NQ")
    parser.add_argument("--trade-symbol", default="MNQ")
    parser.add_argument("--confirmation-symbol", default="ES")
    args = parser.parse_args(argv)
    runner = PhaseRunner(
        Po3ReversalPlugin(
            args.data_dir,
            args.initial_equity,
            analysis_symbol=args.analysis_symbol,
            trade_symbol=args.trade_symbol,
            confirmation_symbol=args.confirmation_symbol,
        ),
        args.output_dir,
    )
    runner.run_all_phases()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
