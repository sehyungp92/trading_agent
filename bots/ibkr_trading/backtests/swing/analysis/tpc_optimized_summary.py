"""Generate a deterministic TPC optimized-config baseline summary."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.swing.auto.tpc.plugin import _extract_tpc_metrics
from backtests.swing.config_tpc import TPCBacktestConfig
from backtests.swing.data.replay_cache import load_tpc_replay_bundle
from backtests.swing.engine.tpc_engine import run_tpc_independent


def build_summary(
    *,
    config_path: Path,
    data_dir: Path,
    initial_equity: float,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    mutations = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = TPCBacktestConfig(initial_equity=initial_equity, data_dir=data_dir).with_overrides(mutations)
    bundle = load_tpc_replay_bundle(data_dir, start_date=start_date, end_date=end_date)
    result = run_tpc_independent(bundle.data, cfg, indicator_cache={})
    metrics = _extract_tpc_metrics(result, initial_equity)
    trades = list(result.trades)
    equity = np.asarray(result.combined_equity, dtype=float)
    closed_pnl = sum(float(getattr(trade, "pnl_dollars", 0.0) or 0.0) for trade in trades)
    final_equity = float(equity[-1]) if equity.size else initial_equity
    peak = np.maximum.accumulate(equity) if equity.size else np.asarray([], dtype=float)
    dd_dollars = float(np.max(peak - equity)) if equity.size else 0.0

    entry_requests = [event for event in result.decision_stream if event.get("code") == "ENTRY_REQUESTED"]
    requests_by_ts: dict[Any, set[str]] = defaultdict(set)
    for event in entry_requests:
        requests_by_ts[event.get("ts")].add(str(event.get("symbol", "")))

    fill_lags = [
        int(getattr(trade, "fill_bar_index", -999)) - int(getattr(trade, "signal_bar_index", -999))
        for trade in trades
    ]
    trade_outcomes = list(result.trade_outcomes)
    outcome_net = sum(float(outcome.get("net_pnl", 0.0) or 0.0) for outcome in trade_outcomes)
    outcome_gross = sum(float(outcome.get("gross_pnl", 0.0) or 0.0) for outcome in trade_outcomes)
    commission = sum(float(getattr(trade, "commission", 0.0) or 0.0) for trade in trades)

    return {
        "basis": "current-code TPC optimized-config replay",
        "replay_loader": "load_tpc_replay_bundle",
        "config_path": str(config_path.as_posix()),
        "data_dir": str(data_dir.as_posix()),
        "initial_equity": float(initial_equity),
        "start_date": start_date,
        "end_date": end_date,
        "source_fingerprint": bundle.cache_source_fingerprint,
        "mutation_count": len(mutations),
        "total_trades": float(len(trades)),
        "win_rate_pct": float(metrics.get("win_rate", 0.0) * 100.0),
        "profit_factor": float(metrics.get("dollar_profit_factor", 0.0)),
        "profit_factor_basis": "net_pnl_dollars",
        "r_profit_factor": float(metrics.get("profit_factor", 0.0)),
        "dollar_profit_factor": float(metrics.get("dollar_profit_factor", 0.0)),
        "total_pnl": float(closed_pnl),
        "closed_pnl": float(closed_pnl),
        "open_mtm_pnl": float(final_equity - initial_equity - closed_pnl),
        "total_r": float(metrics.get("total_r", 0.0)),
        "avg_r": float(metrics.get("avg_r", 0.0)),
        "net_return_pct": float(metrics.get("net_return_pct", 0.0)),
        "final_equity": float(final_equity),
        "max_drawdown_pct": float(metrics.get("max_dd_pct", 0.0)),
        "max_drawdown_dollars": dd_dollars,
        "commission": float(commission),
        "result_total_commission": float(result.total_commission),
        "trade_outcome_count": float(len(trade_outcomes)),
        "trade_outcome_net_pnl": float(outcome_net),
        "trade_outcome_gross_pnl": float(outcome_gross),
        "zero_net_trade_outcomes": float(
            sum(1 for outcome in trade_outcomes if abs(float(outcome.get("net_pnl", 0.0) or 0.0)) < 1e-9)
        ),
        "min_fill_minus_signal_bar": float(min(fill_lags)) if fill_lags else 0.0,
        "noncausal_entries": float(sum(1 for lag in fill_lags if lag <= 0)),
        "negative_holds": float(
            sum(
                1
                for trade in trades
                if getattr(trade, "entry_time", None)
                and getattr(trade, "exit_time", None)
                and trade.exit_time < trade.entry_time
            )
        ),
        "simultaneous_entry_request_bars": float(sum(1 for symbols in requests_by_ts.values() if len(symbols) > 1)),
        "entry_model_counts": dict(sorted(Counter(str(getattr(trade, "leg_type", "") or "") for trade in trades).items())),
        "symbol_counts": dict(sorted(Counter(str(getattr(trade, "symbol", "") or "") for trade in trades).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default="backtests/swing/data/raw")
    parser.add_argument("--equity", type=float, default=100_000.0)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary(
        config_path=Path(args.config),
        data_dir=Path(args.data_dir),
        initial_equity=args.equity,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
