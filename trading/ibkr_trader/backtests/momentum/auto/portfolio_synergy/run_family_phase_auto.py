from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .family_phase_auto import (
    load_or_build_latest_strategy_trades,
    run_family_phase_auto,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run four-strategy momentum family portfolio phase auto optimization.",
    )
    parser.add_argument("--data-dir", default="backtests/momentum/data/raw")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--equity", type=float, default=50_000.0)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=0.00001)
    parser.add_argument("--force-rebuild-trades", action="store_true")
    args = parser.parse_args(argv)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"backtests/output/momentum/portfolio_synergy/family_phase_auto_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    trades = load_or_build_latest_strategy_trades(
        data_dir=Path(args.data_dir),
        output_dir=output_dir,
        initial_equity=args.equity,
        force=args.force_rebuild_trades,
    )
    summary = run_family_phase_auto(
        trades_by_strategy=trades,
        output_dir=output_dir,
        initial_equity=args.equity,
        max_workers=args.max_workers,
        min_delta=args.min_delta,
        data_dir=Path(args.data_dir),
    )
    print(f"Family portfolio phase auto complete: {output_dir}")
    print(f"Score components: {summary['score_component_count']}")
    print(f"Final score: {summary['final_score']:.4f}")
    print(f"Final metrics: {summary['final_metrics']}")


if __name__ == "__main__":
    main()
