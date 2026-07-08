from __future__ import annotations

import pytest

from backtests.shared.auto.plugin_utils import mutation_signature
from backtests.shared.auto.phase_state import PhaseState
from backtests.stock.auto.alcb_p10plus_phase.phase_candidates import (
    EXTRA_CANDIDATES,
    PHASE_CANDIDATES,
    PHASE_FOCUS,
    get_phase_candidates,
)
from backtests.stock.auto.alcb_p10plus_phase.plugin import ALCBP10PlusPlugin
from backtests.stock.auto.greedy_optimize import ALCB_T2_P10_CANDIDATES


def test_phase_candidates_cover_full_p10_union_and_all_extras_exactly_once():
    all_phase_names = [
        name
        for phase in sorted(PHASE_CANDIDATES)
        for name, _ in PHASE_CANDIDATES[phase]
    ]
    all_phase_set = set(all_phase_names)

    assert len(PHASE_FOCUS) == 5
    assert len(PHASE_CANDIDATES) == 5
    assert all(len(PHASE_CANDIDATES[phase]) > 0 for phase in PHASE_CANDIDATES)
    assert len(all_phase_names) == len(all_phase_set)
    assert {name for name, _ in ALCB_T2_P10_CANDIDATES}.issubset(all_phase_set)
    assert {name for name, _ in EXTRA_CANDIDATES}.issubset(all_phase_set)
    assert len(all_phase_names) == len(ALCB_T2_P10_CANDIDATES) + len(EXTRA_CANDIDATES)


def test_plugin_filters_subset_names_across_all_phases(tmp_path):
    subset = {
        "p10x_or_5bars",
        "p10x_rvol_min_250",
        "p10x_qe_2n02_6p00",
        "p10x_mfe_trail_02_02_be_02",
        "p10x_risk_0125",
    }
    plugin = ALCBP10PlusPlugin(tmp_path, max_workers=1, experiment_names=subset)

    observed = set()
    for phase in range(1, plugin.num_phases + 1):
        spec = plugin.get_phase_spec(phase, PhaseState())
        phase_names = {candidate.name for candidate in spec.candidates}
        assert phase_names == {name for name, _ in get_phase_candidates(phase) if name in subset}
        observed.update(phase_names)

    assert observed == subset


def test_plugin_dynamic_floor_resolution_keeps_phase_regressions_bounded(tmp_path):
    plugin = ALCBP10PlusPlugin(tmp_path, max_workers=1)
    baseline = {
        "expectancy_dollar": 0.82,
        "expected_total_r": 9.5,
        "trades_per_month": 44.0,
        "profit_factor": 1.08,
        "entry_quality": 0.51,
        "high_rvol_edge": 0.63,
        "profit_protection": 0.34,
        "short_hold_drag_inverse": 0.45,
        "flow_reversal_short_inverse": 0.29,
        "early_1000_drag_inverse": 0.31,
        "long_hold_capture": 0.68,
        "carry_capture": 0.0,
        "inv_dd": 0.76,
        "max_drawdown_pct": 0.062,
    }

    phase3 = plugin._resolve_phase_hard_rejects(3, baseline, {})
    phase5 = plugin._resolve_phase_hard_rejects(5, baseline, {})

    assert phase3["min_expectancy_dollar"] >= baseline["expectancy_dollar"] * 0.95
    assert phase3["min_profit_protection"] >= baseline["profit_protection"] * 0.98
    assert phase3["max_dd_pct"] <= 0.12
    assert phase5["min_expected_total_r"] >= baseline["expected_total_r"] * 0.98
    assert phase5["min_trades_per_month"] >= baseline["trades_per_month"] * 0.93
    assert phase5["min_long_hold_capture"] >= baseline["long_hold_capture"] * 0.95


def test_plugin_compute_final_metrics_uses_cached_evaluation_metrics(tmp_path, monkeypatch):
    plugin = ALCBP10PlusPlugin(tmp_path, max_workers=1)
    mutations = {"param_overrides.opening_range_bars": 5}
    plugin._metrics_cache[mutation_signature(mutations)] = {"avg_r": 0.25, "profit_factor": 1.12}
    monkeypatch.setattr(
        plugin,
        "_run_config",
        lambda *args, **kwargs: pytest.fail("_run_config should not be called when metrics are cached"),
    )

    assert plugin.compute_final_metrics(mutations) == {"avg_r": 0.25, "profit_factor": 1.12}
