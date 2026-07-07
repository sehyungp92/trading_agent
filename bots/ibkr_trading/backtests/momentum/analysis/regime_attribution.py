from __future__ import annotations

from collections import defaultdict
from typing import Any


def module_attribution(trades: list[Any]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[Any]] = defaultdict(list)
    for trade in trades:
        buckets[str(getattr(trade, "module", "") or "unknown")].append(trade)
    result: dict[str, dict[str, float]] = {}
    for module, items in buckets.items():
        pnl = sum(float(getattr(trade, "pnl_dollars", 0.0)) for trade in items)
        r_total = sum(float(getattr(trade, "r_multiple", 0.0)) for trade in items)
        wins = sum(1 for trade in items if float(getattr(trade, "pnl_dollars", 0.0)) > 0)
        gross_win = sum(float(getattr(trade, "pnl_dollars", 0.0)) for trade in items if float(getattr(trade, "pnl_dollars", 0.0)) > 0)
        gross_loss = abs(sum(float(getattr(trade, "pnl_dollars", 0.0)) for trade in items if float(getattr(trade, "pnl_dollars", 0.0)) < 0))
        result[module] = {
            "trades": float(len(items)),
            "pnl_dollars": pnl,
            "total_r": r_total,
            "avg_r": r_total / len(items) if items else 0.0,
            "win_rate": wins / len(items) if items else 0.0,
            "profit_factor": gross_win / gross_loss if gross_loss > 0 else (10.0 if gross_win > 0 else 0.0),
            "avg_mfe_r": sum(float(getattr(trade, "mfe_r", 0.0)) for trade in items) / len(items) if items else 0.0,
            "avg_mae_r": sum(float(getattr(trade, "mae_r", 0.0)) for trade in items) / len(items) if items else 0.0,
            "avg_setup_score": sum(float(getattr(trade, "setup_score", 0.0)) for trade in items) / len(items) if items else 0.0,
        }
    return result
