from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from backtests.momentum.analysis.regime_diagnostics import generate_regime_diagnostics
from backtests.momentum.auto.nq_regime.phase_candidates import BASE_MUTATIONS
from backtests.momentum.auto.nq_regime.worker import mutate_config
from backtests.momentum.config_regime import NqRegimeBacktestConfig
from backtests.momentum.engine.regime_engine import load_nq_regime_data, run_nq_regime_backtest


DEFAULT_OUTPUT = Path("backtests/output/momentum/nq_regime/baseline_diagnostics.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate strict-baseline NQ_REGIME diagnostics.")
    parser.add_argument("--data-dir", type=Path, default=Path("backtests/momentum/data/raw"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--analysis-symbol", default="NQ")
    parser.add_argument("--trade-symbol", default="MNQ")
    parser.add_argument("--max-contracts", type=int, default=None)
    parser.add_argument("--start-date", default="", help="Optional ISO datetime, e.g. 2026-01-02T00:00:00+00:00")
    parser.add_argument("--end-date", default="", help="Optional ISO datetime, e.g. 2026-01-31T23:59:00+00:00")
    args = parser.parse_args()

    base = NqRegimeBacktestConfig(
        data_dir=args.data_dir,
        initial_equity=args.initial_equity,
        analysis_symbol=args.analysis_symbol,
        trade_symbol=args.trade_symbol,
        max_contracts=args.max_contracts,
        start_date=_parse_datetime(args.start_date),
        end_date=_parse_datetime(args.end_date),
    )
    config = mutate_config(base, BASE_MUTATIONS)
    data = load_nq_regime_data(config)
    result = run_nq_regime_backtest(data, config)
    report = generate_regime_diagnostics(
        result.trades,
        result.metrics,
        signal_events=result.signal_events,
        equity_curve=result.equity_curve,
        timestamps=result.timestamps,
        title="NQ_REGIME Strict Baseline Diagnostics",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(str(args.output))


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


if __name__ == "__main__":
    main()
