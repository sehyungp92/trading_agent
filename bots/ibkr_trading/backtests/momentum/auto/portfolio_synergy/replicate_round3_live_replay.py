from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

from backtests.momentum.engine.family_portfolio_engine import (
    FamilyPortfolioBacktester,
    build_family_replay_bundle,
    family_config_from_dict,
)


DEFAULT_METRIC_KEYS: tuple[str, ...] = (
    "total_trades",
    "blocked_trades",
    "block_rate",
    "active_strategies",
    "min_strategy_trades",
    "net_profit",
    "net_return_pct",
    "profit_factor",
    "win_rate",
    "max_drawdown_pct",
    "sharpe",
    "sortino",
    "cagr",
    "calmar",
    "total_r",
    "total_r_per_month",
    "trades_per_month",
    "max_concurrent",
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Replicate the active momentum portfolio through the neutral-action live-rule replay.",
    )
    parser.add_argument(
        "--round-dir",
        default="backtests/output/momentum/portfolio_synergy/round_2",
    )
    parser.add_argument("--abs-tolerance", type=float, default=1e-6)
    parser.add_argument("--rel-tolerance", type=float, default=1e-9)
    args = parser.parse_args(argv)

    report = replicate_round3_live_replay(
        round_dir=Path(args.round_dir),
        abs_tolerance=args.abs_tolerance,
        rel_tolerance=args.rel_tolerance,
    )
    print(f"Momentum portfolio live-rule replay matched: {report['matched']}")
    print(f"Architecture: {report['replay_architecture']}")
    print(f"Action count: {report['action_count']}")
    print(f"Replay fingerprint: {report['replay_source_fingerprint']}")
    print(f"Wrote {report['output_path']}")


def replicate_round3_live_replay(
    *,
    round_dir: Path,
    abs_tolerance: float = 1e-6,
    rel_tolerance: float = 1e-9,
) -> dict[str, Any]:
    config = family_config_from_dict(
        json.loads((round_dir / "optimized_portfolio_config.json").read_text(encoding="utf-8"))
    )
    with (round_dir / "strategy_trades.pkl").open("rb") as fh:
        trades_by_strategy = pickle.load(fh)
    expected_summary = json.loads((round_dir / "run_summary.json").read_text(encoding="utf-8"))
    expected_metrics = expected_summary.get("final_metrics_realized", expected_summary["final_metrics"])
    expected_headline_metrics = expected_summary["final_metrics"]

    replay_bundle = build_family_replay_bundle(trades_by_strategy)
    result = FamilyPortfolioBacktester(config).run_bundle(replay_bundle)
    comparisons = {
        key: _compare_metric(
            expected_metrics.get(key),
            result.metrics.get(key),
            abs_tolerance=abs_tolerance,
            rel_tolerance=rel_tolerance,
        )
        for key in DEFAULT_METRIC_KEYS
    }
    matched = all(item["matched"] for item in comparisons.values())
    report = {
        "matched": matched,
        "round_dir": str(round_dir),
        "replay_architecture": result.replay_architecture,
        "replay_source_fingerprint": replay_bundle.source_fingerprint,
        "trade_outcome_count": len(replay_bundle.trade_outcomes),
        "decision_count": len(replay_bundle.decisions),
        "action_count": result.action_count,
        "accepted_trades": len(result.trades),
        "blocked_trades": len(result.blocked_trades),
        "strategy_trade_counts": result.strategy_trade_counts,
        "strategy_blocked_counts": result.strategy_blocked_counts,
        "rule_blocks": result.rule_blocks,
        "abs_tolerance": abs_tolerance,
        "rel_tolerance": rel_tolerance,
        "comparisons": comparisons,
        "metrics": result.metrics,
        "expected_headline_metrics": expected_headline_metrics,
        "comparison_basis": "realized_replay_metrics",
    }
    output_path = round_dir / "live_rule_replay_replication.json"
    output_path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")
    report["output_path"] = str(output_path)
    return report


def _compare_metric(
    expected: object,
    actual: object,
    *,
    abs_tolerance: float,
    rel_tolerance: float,
) -> dict[str, Any]:
    expected_float = float(expected or 0.0)
    actual_float = float(actual or 0.0)
    abs_diff = abs(actual_float - expected_float)
    rel_diff = abs_diff / max(abs(expected_float), abs(actual_float), 1.0)
    return {
        "expected": expected_float,
        "actual": actual_float,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "matched": abs_diff <= abs_tolerance or rel_diff <= rel_tolerance,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
