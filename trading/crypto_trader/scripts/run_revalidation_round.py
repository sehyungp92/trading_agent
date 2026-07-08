"""Run manifest replay, ablation, perturbation, and cleaned-seed reruns."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_trader.optimize.revalidation import revalidate_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        choices=["all", "momentum", "trend", "breakout"],
        default="all",
        help="Strategy to revalidate.",
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "output" / "revalidated"),
        help="Root directory for revalidation outputs.",
    )
    parser.add_argument(
        "--skip-rerun",
        action="store_true",
        help="Stop after cleaned-seed generation and do not run the fresh phased rerun.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategies = (
        ["momentum", "trend", "breakout"]
        if args.strategy == "all"
        else [args.strategy]
    )

    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = Path(args.output_root) / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    for strategy in strategies:
        print(f"[{strategy}] starting revalidation")
        summary = revalidate_strategy(
            ROOT,
            strategy,
            run_root / strategy,
            include_rerun=not args.skip_rerun,
        )
        summaries[strategy] = summary
        print(
            json.dumps(
                {
                    "strategy": strategy,
                    "strategy_output_dir": summary["strategy_output_dir"],
                    "cleaned_seed_config": summary["cleaned_seed_config"],
                    "winner_score": summary["winner"]["score"],
                    "winner_return_pct": summary["winner"]["metrics"]["net_return_pct"],
                    "winner_total_trades": summary["winner"]["metrics"]["total_trades"],
                },
                indent=2,
                default=str,
            )
        )

    summary_path = run_root / "revalidation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_root": str(run_root),
                "strategies": summaries,
            },
            handle,
            indent=2,
            default=str,
        )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
