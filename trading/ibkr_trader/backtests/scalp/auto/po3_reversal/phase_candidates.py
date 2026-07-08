from __future__ import annotations

from typing import Any


BASE_MUTATIONS: dict[str, Any] = {}

PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("HTF CONTEXT", ["edge_velocity", "expectancy_dollar", "trades_per_month"]),
    2: ("LIQUIDITY + SWEEP", ["profit_factor", "expectancy_dollar", "total_trades"]),
    3: ("ENTRY PRECISION", ["profit_factor", "avg_r", "edge_velocity"]),
    4: ("RISK + EXITS", ["net_profit", "max_drawdown_pct", "edge_velocity"]),
}


def get_phase_candidates(
    phase: int,
    current_mutations: dict[str, Any] | None = None,
    suggested_experiments: list[tuple[str, dict]] | None = None,
) -> list[tuple[str, dict]]:
    del current_mutations
    candidates: list[tuple[str, dict]]
    if phase == 1:
        candidates = [
            ("daily_lookback_15", {"param_overrides.DAILY_LOOKBACK": 15}),
            ("daily_lookback_30", {"param_overrides.DAILY_LOOKBACK": 30}),
            ("tier_a_threshold_55", {"param_overrides.SCORE_THRESHOLD_A": 5.5}),
        ]
    elif phase == 2:
        candidates = [
            ("nq_sweep_4t", {"param_overrides.MIN_NQ_SWEEP_TICKS": 4}),
            ("nq_sweep_8t", {"param_overrides.MIN_NQ_SWEEP_TICKS": 8}),
            ("smt_strength_025", {"param_overrides.MIN_SMT_STRENGTH": 0.25}),
        ]
    elif phase == 3:
        candidates = [
            ("body_060", {"param_overrides.MIN_BODY_PERCENT": 0.60}),
            ("entry_offset_2t", {"param_overrides.ENTRY_OFFSET_TICKS": 2}),
            ("retest_wait_5", {"param_overrides.retest_wait_bars": 5}),
        ]
    elif phase == 4:
        candidates = [
            ("risk_a_25bp", {"param_overrides.A_RISK_PCT": 0.0025}),
            ("stop_buffer_6t", {"param_overrides.STOP_MIN_BUFFER_TICKS": 6}),
            ("b_tier_off", {"flags.disable_b_tier": True}),
        ]
    else:
        candidates = []
    existing = {name for name, _ in candidates}
    for name, mutations in suggested_experiments or []:
        if name not in existing:
            candidates.append((name, dict(mutations)))
    return candidates
