from __future__ import annotations

from typing import Any


BASE_MUTATIONS: dict[str, Any] = {}

PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("ENTRY QUALITY", ["edge_velocity", "expectancy_dollar", "trades_per_month"]),
    2: ("RISK + STOPS", ["max_drawdown_pct", "profit_factor", "avg_r"]),
    3: ("TARGET TUNING", ["net_profit", "profit_factor", "edge_velocity"]),
    4: ("SESSION + INTEGRATION", ["trades_per_month", "expectancy_dollar", "net_profit"]),
}


def get_phase_candidates(
    phase: int,
    current_mutations: dict[str, Any] | None = None,
    suggested_experiments: list[tuple[str, dict]] | None = None,
) -> list[tuple[str, dict]]:
    del current_mutations
    if phase == 1:
        candidates = [
            ("min_score_70", {"param_overrides.A1_MIN_SCORE": 70.0}),
            ("break_buffer_3pt", {"param_overrides.MIN_BUFFER_PTS": 3.0}),
            ("hold_120s", {"param_overrides.MIN_HOLD_SECONDS": 120}),
        ]
    elif phase == 2:
        candidates = [
            ("risk_35bp", {"param_overrides.BASE_RISK_PCT": 0.0035}),
            ("stop_cap_30", {"param_overrides.STOP_CAP_IVB_FRACTION": 0.30}),
            ("a2_half_off", {"flags.disable_a2_module": True}),
        ]
    elif phase == 3:
        candidates = [
            ("tp1_q70", {"param_overrides.TP1_QUANTILE": 0.70}),
            ("tp2_q95", {"param_overrides.TP2_QUANTILE": 0.95}),
            ("min_rr_2", {"param_overrides.MIN_R_TO_TP1": 2.0}),
        ]
    elif phase == 4:
        candidates = [
            ("ivb_range_tight", {"param_overrides.MIN_IVB_RANGE_POINTS": 30.0, "param_overrides.MAX_IVB_RANGE_POINTS": 150.0}),
            ("time_gate_off", {"flags.disable_time_filter": True}),
            ("no_chase_filter", {"flags.disable_chase_rejection": True}),
        ]
    else:
        candidates = []
    existing = {name for name, _ in candidates}
    for name, mutations in suggested_experiments or []:
        if name not in existing:
            candidates.append((name, dict(mutations)))
    return candidates

