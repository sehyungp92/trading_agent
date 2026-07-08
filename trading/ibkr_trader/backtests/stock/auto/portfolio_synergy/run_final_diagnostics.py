from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .phase_candidates import BLOCKED_ALPHA_ROUND3_PROFILE, DEFAULT_PROFILE
from .plugin import StockPortfolioSynergyPlugin


def _diagnostic_round_label(run_dir: Path) -> str | None:
    match = re.fullmatch(r"round_(\d+)", run_dir.name)
    if not match:
        return None
    return f"round_{match.group(1)}_dynamic_stock_synergy"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate final stock portfolio synergy diagnostics from an optimized config.",
    )
    parser.add_argument("--run-dir", default="backtests/output/stock/portfolio_synergy/round_1")
    parser.add_argument("--data-dir", default="backtests/stock/data/raw")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-03-01")
    parser.add_argument("--equity", type=float, default=25_000.0)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument(
        "--profile",
        choices=(DEFAULT_PROFILE, BLOCKED_ALPHA_ROUND3_PROFILE),
        default=DEFAULT_PROFILE,
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / "round_final_diagnostics.txt"
    mutations = json.loads((run_dir / "optimized_config.json").read_text(encoding="utf-8"))

    plugin = StockPortfolioSynergyPlugin(
        data_dir=Path(args.data_dir),
        start_date=args.start,
        end_date=args.end,
        initial_equity=float(args.equity),
        max_workers=int(args.max_workers),
        round_profile=args.profile,
    )
    round_label = _diagnostic_round_label(run_dir)
    if round_label:
        plugin.diagnostic_round_label = round_label
    metrics = plugin.compute_final_metrics(mutations)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        plugin._format_diagnostics("FINAL STOCK PORTFOLIO SYNERGY DIAGNOSTICS", metrics, None),
        encoding="utf-8",
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
