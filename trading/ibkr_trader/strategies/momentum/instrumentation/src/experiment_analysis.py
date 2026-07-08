"""Lightweight experiment analysis helper (#20).

Reads JSONL trade files, groups by experiment variant, computes stats,
and runs a Welch's t-test using math.erfc (no scipy dependency).
"""
from __future__ import annotations

import json
import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("instrumentation.experiment_analysis")


@dataclass
class VariantStats:
    """Per-variant summary statistics."""
    variant: str
    trade_count: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    sharpe: float = 0.0
    total_pnl: float = 0.0


@dataclass
class ExperimentResult:
    """Result of analyzing an A/B experiment."""
    experiment_id: str
    variants: dict[str, VariantStats] = field(default_factory=dict)
    sufficient_sample: bool = False
    p_value: Optional[float] = None
    significant: bool = False


def _welch_t_test(mean1: float, mean2: float, var1: float, var2: float,
                  n1: int, n2: int) -> float:
    """Welch's t-test p-value approximation using math.erfc.

    Returns two-tailed p-value. Uses the erfc approximation for the
    t-distribution which is accurate for df > 5.
    """
    if n1 < 2 or n2 < 2:
        return 1.0

    se1 = var1 / n1
    se2 = var2 / n2
    denom = se1 + se2
    if denom <= 0:
        return 1.0

    t_stat = abs(mean1 - mean2) / math.sqrt(denom)

    # Welch-Satterthwaite degrees of freedom
    num = denom ** 2
    den = (se1 ** 2 / (n1 - 1)) + (se2 ** 2 / (n2 - 1))
    if den <= 0:
        return 1.0
    df = num / den

    # Approximate p-value using normal approximation for large df,
    # or a simple t-to-p conversion using erfc for moderate df
    if df > 30:
        # Normal approximation
        p = math.erfc(t_stat / math.sqrt(2))
    else:
        # Rough approximation: adjust t_stat for df
        adjusted_t = t_stat * (1 - 1 / (4 * df))
        p = math.erfc(adjusted_t / math.sqrt(2))

    return p


def _compute_variant_stats(variant: str, pnl_values: list[float]) -> VariantStats:
    """Compute summary statistics for a single variant."""
    n = len(pnl_values)
    if n == 0:
        return VariantStats(variant=variant)

    total = sum(pnl_values)
    mean = total / n
    wins = sum(1 for p in pnl_values if p > 0)

    # Sharpe: mean / std (annualize not applicable for trade-level)
    if n > 1:
        variance = sum((p - mean) ** 2 for p in pnl_values) / (n - 1)
        std = math.sqrt(variance) if variance > 0 else 0
        sharpe = mean / std if std > 0 else 0.0
    else:
        sharpe = 0.0

    return VariantStats(
        variant=variant,
        trade_count=n,
        win_rate=wins / n if n > 0 else 0.0,
        avg_pnl=mean,
        sharpe=round(sharpe, 4),
        total_pnl=round(total, 2),
    )


def analyze_experiment(
    experiment_id: str,
    trades_dir: str | Path,
    min_trades: int = 30,
) -> ExperimentResult:
    """Analyze an experiment by reading trade JSONL files.

    Groups trades by experiment_variant, computes per-variant stats,
    and runs significance test between the first two variants.
    """
    trades_path = Path(trades_dir)
    result = ExperimentResult(experiment_id=experiment_id)

    # Collect PnL per variant
    variant_pnls: dict[str, list[float]] = {}

    try:
        for jsonl_file in sorted(trades_path.glob("trades_*.jsonl")):
            for line in jsonl_file.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    trade = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if trade.get("experiment_id") != experiment_id:
                    continue
                if trade.get("stage") != "exit":
                    continue

                variant = trade.get("experiment_variant", "control")
                pnl = trade.get("pnl")
                if pnl is not None:
                    variant_pnls.setdefault(variant, []).append(float(pnl))
    except Exception as e:
        logger.warning("Error reading trades for experiment %s: %s", experiment_id, e)
        return result

    # Compute stats per variant
    for variant, pnls in variant_pnls.items():
        result.variants[variant] = _compute_variant_stats(variant, pnls)

    # Check sample sufficiency
    all_sufficient = all(
        vs.trade_count >= min_trades for vs in result.variants.values()
    )
    result.sufficient_sample = all_sufficient and len(result.variants) >= 2

    # Significance test between first two variants
    variants = list(variant_pnls.keys())
    if len(variants) >= 2 and result.sufficient_sample:
        pnl_a = variant_pnls[variants[0]]
        pnl_b = variant_pnls[variants[1]]
        n_a, n_b = len(pnl_a), len(pnl_b)
        mean_a = sum(pnl_a) / n_a
        mean_b = sum(pnl_b) / n_b
        var_a = sum((p - mean_a) ** 2 for p in pnl_a) / (n_a - 1) if n_a > 1 else 0
        var_b = sum((p - mean_b) ** 2 for p in pnl_b) / (n_b - 1) if n_b > 1 else 0

        p_value = _welch_t_test(mean_a, mean_b, var_a, var_b, n_a, n_b)
        result.p_value = round(p_value, 6)
        result.significant = p_value < 0.05

    return result
