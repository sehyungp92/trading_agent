from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path

from backtests.shared.auto.types import ScoredCandidate

from .phase_scoring import merge_alcb_metrics, score_alcb_phase
from .time_utils import hydrate_time_mutations

_worker_replay = None
_worker_config = None
_worker_equity: float = 0.0
_worker_phase: int = 0
_worker_hard_rejects: dict | None = None
_worker_scoring_weights: dict | None = None


def init_worker(
    data_dir_str: str,
    start_date: str,
    end_date: str,
    equity: float,
    phase: int = 0,
    hard_rejects: dict | None = None,
    scoring_weights: dict | None = None,
) -> None:
    global _worker_replay, _worker_config, _worker_equity, _worker_phase
    global _worker_hard_rejects, _worker_scoring_weights

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from backtests.stock.config_alcb import ALCBBacktestConfig
    from backtests.stock.data.replay_cache import load_research_replay_bundle

    data_dir = Path(data_dir_str)
    _worker_replay = load_research_replay_bundle(data_dir).data
    _worker_equity = equity
    _worker_phase = phase
    _worker_hard_rejects = hard_rejects or {}
    _worker_scoring_weights = scoring_weights or {}
    _worker_config = ALCBBacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_equity=equity,
        tier=2,
        data_dir=data_dir,
    )


def score_candidate(args: tuple) -> ScoredCandidate:
    """Score a single candidate configuration.

    Accepts either a 3-tuple (legacy, uses worker globals for phase config):
        (name, candidate_muts, base_muts)
    or a 6-tuple (pool-persistent, phase config passed per-call):
        (name, candidate_muts, base_muts, phase, hard_rejects, scoring_weights)
    """
    if len(args) >= 6:
        name, candidate_muts, base_muts, phase, hard_rejects, scoring_weights = args[:6]
    else:
        name, candidate_muts, base_muts = args[:3]
        phase = _worker_phase
        hard_rejects = _worker_hard_rejects
        scoring_weights = _worker_scoring_weights

    try:
        from backtests.stock.auto.config_mutator import mutate_alcb_config
        from backtests.stock.auto.scoring import extract_metrics
        from backtests.stock.engine.alcb_engine import ALCBIntradayEngine

        all_muts = hydrate_time_mutations(dict(base_muts))
        all_muts.update(candidate_muts)

        config = mutate_alcb_config(_worker_config, all_muts)
        result = ALCBIntradayEngine(config, _worker_replay).run()
        perf = extract_metrics(
            result.trades,
            result.equity_curve,
            result.timestamps,
            _worker_equity,
        )

        # Fast path: reject on cheap base-perf metrics before expensive
        # per-trade analysis in merge_alcb_metrics.
        early_reason = _early_reject_reason(perf, hard_rejects, phase=phase)
        if early_reason:
            return ScoredCandidate(
                name=name,
                score=0.0,
                rejected=True,
                reject_reason=early_reason,
            )

        merged_metrics = merge_alcb_metrics(perf, result.trades)
        reject_reason = phase_reject_reason(merged_metrics, hard_rejects, phase=phase)
        if reject_reason:
            return ScoredCandidate(
                name=name,
                score=0.0,
                rejected=True,
                reject_reason=reject_reason,
                metrics=merged_metrics,
            )

        score = score_alcb_phase(phase, merged_metrics, scoring_weights)
        return ScoredCandidate(name=name, score=score, metrics=merged_metrics)
    except Exception:
        return ScoredCandidate(name=name, score=0.0, rejected=True, reject_reason=traceback.format_exc())


_EARLY_REJECT_CHECKS: tuple[tuple[str, str, bool], ...] = (
    ("min_net_profit", "net_profit", True),
    ("min_expectancy_dollar", "expectancy_dollar", True),
    ("min_expected_total_r", "expected_total_r", False),
    ("min_trades_per_month", "trades_per_month", False),
    ("min_pf", "profit_factor", False),
    ("min_inv_dd", "inv_dd", False),
)


def _early_reject_reason(perf, hard_rejects: dict | None, *, phase: int = 0) -> str:
    """Reject on cheap base-perf metrics before per-trade analysis."""
    rejects = hard_rejects or {}
    if not rejects:
        return ""

    net_profit = float(getattr(perf, "net_profit", 0.0))
    expectancy_dollar = float(getattr(perf, "expectancy_dollar", 0.0))
    trades_per_month = float(getattr(perf, "trades_per_month", 0.0))
    profit_factor = float(getattr(perf, "profit_factor", 0.0))
    max_drawdown_pct = float(getattr(perf, "max_drawdown_pct", 0.0))
    total_trades = float(getattr(perf, "total_trades", 0.0))
    expectancy_r = float(getattr(perf, "expectancy", 0.0))

    values = {
        "net_profit": net_profit,
        "expectancy_dollar": expectancy_dollar,
        "expected_total_r": expectancy_r * total_trades,
        "trades_per_month": trades_per_month,
        "profit_factor": profit_factor,
        "inv_dd": min(1.0, max(0.0, 1.0 - max_drawdown_pct / 0.12)),
    }

    for reject_key, metric_key, strict_positive in _EARLY_REJECT_CHECKS:
        threshold = rejects.get(reject_key)
        if threshold is None:
            continue
        actual = values[metric_key]
        threshold_value = float(threshold)
        failed = actual <= threshold_value if strict_positive else actual < threshold_value
        if failed:
            return (
                f"phase{phase}_{metric_key} "
                f"({actual:.4f} {'<=' if strict_positive else '<'} {threshold_value:.4f})"
            )

    max_dd_threshold = rejects.get("max_dd_pct")
    if max_dd_threshold is not None and max_drawdown_pct > float(max_dd_threshold):
        return (
            f"phase{phase}_max_dd "
            f"({max_drawdown_pct:.2%} > {float(max_dd_threshold):.2%})"
        )

    return ""


def phase_reject_reason(
    metrics: dict[str, float],
    hard_rejects: dict | None,
    *,
    phase: int = 0,
) -> str:
    rejects = hard_rejects or {}

    min_checks = (
        ("min_net_profit", "net_profit", True),
        ("min_expectancy_dollar", "expectancy_dollar", True),
        ("min_expected_total_r", "expected_total_r", False),
        ("min_trades_per_month", "trades_per_month", False),
        ("min_pf", "profit_factor", False),
        ("min_entry_quality", "entry_quality", False),
        ("min_signal_quality", "signal_quality", False),
        ("min_timing_quality", "timing_quality", False),
        ("min_extended_avwap_inverse", "extended_avwap_inverse", False),
        ("min_bar9_inverse", "bar9_inverse", False),
        ("min_late_entry_quality", "late_entry_quality", False),
        ("min_score_monotonicity", "score_monotonicity", False),
        ("min_high_rvol_edge", "high_rvol_edge", False),
        ("min_profit_protection", "profit_protection", False),
        ("min_short_hold_drag_inverse", "short_hold_drag_inverse", False),
        ("min_short_hold_24_drag_inverse", "short_hold_24_drag_inverse", False),
        ("min_flow_reversal_short_inverse", "flow_reversal_short_inverse", False),
        ("min_flow_reversal_inverse", "flow_reversal_inverse", False),
        ("min_mfe_conviction_inverse", "mfe_conviction_inverse", False),
        ("min_flow_mfe_exit_inverse", "flow_mfe_exit_inverse", False),
        ("min_early_1000_drag_inverse", "early_1000_drag_inverse", False),
        ("min_long_hold_capture", "long_hold_capture", False),
        ("min_carry_capture", "carry_capture", False),
        ("min_inv_dd", "inv_dd", False),
        ("min_mfe_capture_efficiency", "mfe_capture_efficiency", False),
        ("min_sizing_alignment", "sizing_alignment", False),
    )
    for reject_key, metric_key, strict_positive in min_checks:
        threshold = rejects.get(reject_key)
        if threshold is None:
            continue
        actual = float(metrics.get(metric_key, 0.0))
        threshold_value = float(threshold)
        failed = actual <= threshold_value if strict_positive else actual < threshold_value
        if failed:
            return (
                f"phase{phase}_{metric_key} "
                f"({actual:.4f} {'<=' if strict_positive else '<'} {threshold_value:.4f})"
            )

    max_dd = rejects.get("max_dd_pct")
    if max_dd is not None and float(metrics.get("max_drawdown_pct", 0.0)) > float(max_dd):
        return (
            f"phase{phase}_max_dd "
            f"({metrics.get('max_drawdown_pct', 0.0):.2%} > {float(max_dd):.2%})"
        )

    return ""
