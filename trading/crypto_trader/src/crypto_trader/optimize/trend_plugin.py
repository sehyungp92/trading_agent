"""TrendPlugin — 6-phase optimization plugin for the trend-following strategy.

Signal-first phase ordering designed for low-trade-count baselines.
"""

from __future__ import annotations

from numbers import Real
from pathlib import Path
from typing import Any

import structlog

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.backtest.diagnostics import generate_diagnostics
from crypto_trader.backtest.metrics import (
    PerformanceMetrics,
    metrics_to_dict,
)
from crypto_trader.backtest.runner import run
from crypto_trader.data.store import ParquetStore
from crypto_trader.optimize.config_mutator import apply_mutations
from crypto_trader.optimize.evaluation import (
    build_evaluation_report,
    format_dimension_text,
)
from crypto_trader.optimize.parallel import evaluate_parallel
from crypto_trader.optimize.types import (
    EndOfRoundArtifacts,
    EvaluateFn,
    Experiment,
    GateCriterion,
    GreedyResult,
    PhaseAnalysisPolicy,
    PhaseDecision,
    PhaseSpec,
    ScoredCandidate,
)
from crypto_trader.strategy.trend.config import TrendConfig

log = structlog.get_logger("optimize.trend")


def _diagnostic_initial_equity(config: Any, default: float = 10_000.0) -> float:
    value = getattr(config, "initial_equity", default)
    return float(value) if isinstance(value, Real) else default


def _result_trades(result: Any) -> list[Any]:
    trades = getattr(result, "trades", []) if result is not None else []
    return trades if isinstance(trades, list) else []


def _result_terminal_marks(result: Any) -> list[Any]:
    terminal_marks = getattr(result, "terminal_marks", []) if result is not None else []
    return terminal_marks if isinstance(terminal_marks, list) else []

# ── Scoring weights (balanced — warmup provides sufficient trades) ────────

SCORING_WEIGHTS: dict[str, float] = {
    "returns": 0.22,
    "coverage": 0.17,
    "edge": 0.16,
    "expectancy": 0.14,
    "capture": 0.13,
    "entry_quality": 0.10,
    "risk": 0.08,
}

PHASE_SCORING_EMPHASIS: dict[int, dict[str, float]] = {
    phase: dict(SCORING_WEIGHTS) for phase in range(1, 7)
}

# Hard rejects — baseline now has 35 trades, PF 3.10
HARD_REJECTS: dict[str, tuple[str, float]] = {
    "max_drawdown_pct": ("<=", 12.0),
    "total_trades": (">=", 30),
    "profit_factor": (">=", 1.5),
    "expectancy_r": (">=", 0.10),
    "net_return_pct": (">=", 20.0),
}

# Scoring ceilings leave headroom above the latest optimized baseline so the
# optimizer still sees improvements in return, coverage, capture, and entry quality.
SCORING_CEILINGS: dict[str, float] = {
    "returns": 85.0,
    "coverage": 85.0,
    "edge": 4.0,
    "expectancy": 0.60,
    "capture": 0.75,
    "entry_quality": 0.75,
    "risk": 12.0,
}

PHASE_GATE_CRITERIA: dict[int, list[GateCriterion]] = {
    1: [  # Signal — quality signal discovery
        GateCriterion(metric="total_trades", operator=">=", threshold=30),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=12),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.5),
    ],
    2: [  # Regime — coverage + regime tuning
        GateCriterion(metric="total_trades", operator=">=", threshold=30),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=12),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.5),
    ],
    3: [  # Trail — capture optimization
        GateCriterion(metric="total_trades", operator=">=", threshold=30),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=12),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.5),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.45),
    ],
    4: [  # Exit — exit efficiency
        GateCriterion(metric="total_trades", operator=">=", threshold=30),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=12),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.5),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.45),
    ],
    5: [  # Risk — risk-adjusted returns
        GateCriterion(metric="total_trades", operator=">=", threshold=30),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=12),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.5),
    ],
    6: [  # Finetune — balanced polish
        GateCriterion(metric="total_trades", operator=">=", threshold=30),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=12),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.5),
    ],
}

PHASE_NAMES: dict[int, str] = {
    1: "Signal & Setup",
    2: "Confirmation & Entry",
    3: "Early Trade Management",
    4: "Trail & Profit Capture",
    5: "Frequency Expansion",
    6: "Risk & Finetune",
}

# ── Phase diagnostic module mapping ──────────────────────────────────────

PHASE_DIAGNOSTIC_MODULES: dict[int, list[str]] = {
    1: ["D4", "D6"],                           # Signal → signal diagnostics
    2: ["D4", "D5", "D6"],                     # Regime → signal + environment
    3: ["D1", "D6"],                           # Trail → trail/stop diagnostics
    4: ["D1", "D2", "D6"],                     # Exit → trail + exit
    5: ["D3", "D5", "D6"],                     # Risk → risk/DD + environment
    6: ["D1", "D2", "D3", "D4", "D5", "D6"],  # Finetune → all
}

# Trend-specific confirmation type → config path mapping
CONFIRMATION_DISABLE_MAP: dict[str, str] = {
    "engulfing": "confirmation.enable_engulfing",
    "hammer": "confirmation.enable_hammer",
    "ema_reclaim": "confirmation.enable_ema_reclaim",
    "structure_break": "confirmation.enable_structure_break",
}


# ── Phase candidate generators ───────────────────────────────────────────

def _archived_phase1_candidates() -> list[Experiment]:
    """Signal & Setup — centered on baked impulse_min_atr_move=0.8."""
    experiments = []
    # Impulse parameters (0.8 is now default — explore below and above)
    for v in [0.5, 0.6, 1.0, 1.5, 2.0]:
        experiments.append(Experiment(f"impulse_atr_{v}", {"setup.impulse_min_atr_move": v}))
    experiments.append(Experiment("impulse_bars_2", {"setup.impulse_min_bars": 2}))
    for v in [20, 40, 50]:
        experiments.append(Experiment(f"impulse_lookback_{v}", {"setup.impulse_lookback": v}))
    # Pullback parameters
    experiments.append(Experiment("pullback_retrace_618", {"setup.pullback_max_retrace": 0.618}))
    experiments.append(Experiment("pullback_retrace_85", {"setup.pullback_max_retrace": 0.85}))
    # Room-to-target
    for v in [0.5, 0.8, 1.5, 2.0]:
        experiments.append(Experiment(f"room_{v}", {"setup.min_room_r": v}))
    for v in [1.5, 2.5]:
        experiments.append(Experiment(f"room_a_{v}", {"setup.min_room_r_a": v}))
    # Structural
    experiments.append(Experiment("impulse_incomplete", {"setup.require_completed_impulse": False}))
    experiments.append(Experiment("orderly_pullback", {"setup.require_orderly_pullback": True}))
    # Confluences
    for v in [1, 2]:
        experiments.append(Experiment(f"min_confluences_{v}", {"setup.min_confluences": v}))
    # Hammer re-enable (disabled by default — let optimizer decide)
    experiments.append(Experiment("enable_hammer", {"confirmation.enable_hammer": True}))
    # Compound experiments
    experiments.append(Experiment("combo_permissive", {
        "setup.impulse_min_atr_move": 1.0,
        "setup.min_room_r": 0.8,
    }))
    experiments.append(Experiment("combo_quality", {
        "setup.impulse_min_atr_move": 2.0,
        "setup.min_confluences": 2,
    }))
    experiments.append(Experiment("combo_max_coverage", {
        "setup.require_completed_impulse": False,
        "setup.impulse_min_atr_move": 0.6,
        "setup.min_room_r": 0.8,
    }))
    return experiments


def _archived_phase2_candidates() -> list[Experiment]:
    """Regime & Coverage — tune regime acceptance around baked h1_min_adx=22."""
    experiments = []
    # Regime ADX thresholds (12 is baked, explore around it)
    for v in [10.0, 14.0, 18.0]:
        experiments.append(Experiment(f"regime_a_adx_{v}", {"regime.a_min_adx": v}))
    for v in [8.0, 10.0]:
        experiments.append(Experiment(f"regime_b_adx_{v}", {"regime.b_min_adx": v}))
    for v in [7.0, 8.0]:
        experiments.append(Experiment(f"no_trade_adx_{v}", {"regime.no_trade_max_adx": v}))
    # Regime quality filters
    experiments.append(Experiment("require_structure", {"regime.require_structure": True}))
    experiments.append(Experiment("require_ema_cross", {"regime.require_ema_cross": True}))
    experiments.append(Experiment("b_adx_rising", {"regime.b_adx_rising_required": True}))
    for v in [15, 20]:
        experiments.append(Experiment(f"structure_lookback_{v}", {"regime.structure_lookback": v}))
    # H1 regime supplement (22.0 baked — explore below and above)
    experiments.append(Experiment("h1_regime_off", {"regime.h1_regime_enabled": False}))
    experiments.append(Experiment("h1_adx_15", {"regime.h1_min_adx": 15.0}))
    experiments.append(Experiment("h1_adx_18", {"regime.h1_min_adx": 18.0}))
    experiments.append(Experiment("h1_adx_20", {"regime.h1_min_adx": 20.0}))
    experiments.append(Experiment("h1_adx_25", {"regime.h1_min_adx": 25.0}))
    experiments.append(Experiment("h1_adx_30", {"regime.h1_min_adx": 30.0}))
    # Symbol direction filters
    experiments.append(Experiment("sol_disabled", {"symbol_filter.sol_direction": "disabled"}))
    experiments.append(Experiment("eth_long_only", {"symbol_filter.eth_direction": "long_only"}))
    # Re-entry
    experiments.append(Experiment("reentry_off", {"reentry.enabled": False}))
    experiments.append(Experiment("reentry_cool_2_max_2", {
        "reentry.cooldown_bars": 2, "reentry.max_reentries": 2,
    }))
    # Limits
    experiments.append(Experiment("max_daily_8", {"limits.max_trades_per_day": 8}))
    # Combo: max B-tier coverage
    experiments.append(Experiment("combo_b_coverage", {
        "regime.b_min_adx": 8.0, "regime.no_trade_max_adx": 7.0,
    }))
    return experiments


def _archived_phase3_candidates() -> list[Experiment]:
    """Trail & Stop — centered on baked atr_mult=2.0, trail_buffer_tight=0.1."""
    experiments = []
    # Trail ceiling (default 1.5, explore range)
    for v in [0.8, 1.0, 2.0, 2.5, 3.0]:
        experiments.append(Experiment(f"trail_ceiling_{v}", {"trail.trail_r_ceiling": v}))
    # Trail buffers (defaults: tight=0.1, wide=1.2)
    for v in [0.05, 0.15, 0.3, 0.4]:
        experiments.append(Experiment(f"trail_tight_{v}", {"trail.trail_buffer_tight": v}))
    for v in [0.8, 1.0, 1.5, 2.0]:
        experiments.append(Experiment(f"trail_wide_{v}", {"trail.trail_buffer_wide": v}))
    # Trail activation (M15 scale)
    for v in [0.2, 0.4, 0.5]:
        experiments.append(Experiment(f"trail_act_r_{v}", {"trail.trail_activation_r": v}))
    for v in [4, 6, 12, 16]:
        experiments.append(Experiment(f"trail_act_bars_{v}", {"trail.trail_activation_bars": v}))
    # Structure trail
    experiments.append(Experiment("struct_trail_on", {"trail.structure_trail_enabled": True}))
    # Stop ATR (2.0 is now default — explore around it)
    for v in [1.0, 1.5, 2.5, 3.0]:
        experiments.append(Experiment(f"stop_atr_{v}", {"stops.atr_mult": v}))
    for v in [0.8, 1.5]:
        experiments.append(Experiment(f"stop_min_atr_{v}", {"stops.min_stop_atr": v}))
    # Compound
    experiments.append(Experiment("combo_tight_trail", {
        "trail.trail_r_ceiling": 1.0, "trail.trail_buffer_tight": 0.15, "trail.trail_buffer_wide": 0.8,
    }))
    experiments.append(Experiment("combo_moderate_trail", {
        "trail.trail_r_ceiling": 1.5, "trail.trail_buffer_tight": 0.2, "trail.trail_buffer_wide": 1.0,
    }))
    experiments.append(Experiment("combo_wide_trail", {
        "trail.trail_r_ceiling": 3.0, "trail.trail_buffer_tight": 0.4, "trail.trail_buffer_wide": 2.0,
    }))
    return experiments


def _archived_phase4_candidates() -> list[Experiment]:
    """Profit Taking & Exit — centered on baked tp1_r=0.8, time_stop_bars=20."""
    experiments = []
    # TP1 (0.8 is now default — explore around it)
    for v in [0.5, 0.6, 1.0, 1.2]:
        experiments.append(Experiment(f"tp1_r_{v}", {"exits.tp1_r": v}))
    for v in [0.30, 0.40]:
        experiments.append(Experiment(f"tp1_frac_{v}", {"exits.tp1_frac": v}))
    # TP2
    for v in [1.5, 2.5, 3.0]:
        experiments.append(Experiment(f"tp2_r_{v}", {"exits.tp2_r": v}))
    for v in [0.40, 0.60]:
        experiments.append(Experiment(f"tp2_frac_{v}", {"exits.tp2_frac": v}))
    # Time stop (20 is now default — explore around it)
    for v in [8, 16, 24, 28, 32]:
        experiments.append(Experiment(f"time_stop_{v}", {"exits.time_stop_bars": v}))
    experiments.append(Experiment("time_stop_r_0_2", {"exits.time_stop_min_progress_r": 0.2}))
    experiments.append(Experiment("time_stop_exit", {"exits.time_stop_action": "exit"}))
    # BE
    for v in [0.1, 0.3]:
        experiments.append(Experiment(f"be_buffer_{v}", {"exits.be_buffer_r": v}))
    for v in [2, 6, 8]:
        experiments.append(Experiment(f"be_min_bars_{v}", {"exits.be_min_bars_above": v}))
    # EMA failsafe
    experiments.append(Experiment("ema_failsafe_15", {"exits.ema_failsafe_period": 15}))
    # Quick exit (M15 scale: 12 default = 3 hours; now enabled by default)
    experiments.append(Experiment("quick_exit_off", {"exits.quick_exit_enabled": False}))
    for v in [8, 16, 24]:
        experiments.append(Experiment(f"quick_exit_bars_{v}", {"exits.quick_exit_bars": v}))
    return experiments


def _archived_phase5_candidates() -> list[Experiment]:
    """Risk & Sizing — scale after alpha confirmed (risk_pct_b=0.01 baked)."""
    experiments = []
    for v in [0.01, 0.02, 0.025]:
        experiments.append(Experiment(f"risk_a_{v}", {"risk.risk_pct_a": v}))
    for v in [0.008, 0.012, 0.015]:
        experiments.append(Experiment(f"risk_b_{v}", {"risk.risk_pct_b": v}))
    for v in [3, 4, 6]:
        experiments.append(Experiment(f"max_pos_{v}", {"limits.max_concurrent_positions": v}))
    for v in [0.03, 0.05]:
        experiments.append(Experiment(f"daily_loss_{v}", {"limits.max_daily_loss_pct": v}))
    experiments.append(Experiment("funding_on", {"filters.funding_filter_enabled": True}))
    for v in [12, 20]:
        experiments.append(Experiment(f"lev_major_{v}", {"risk.max_leverage_major": float(v)}))
    for v in [10, 15]:
        experiments.append(Experiment(f"lev_alt_{v}", {"risk.max_leverage_alt": float(v)}))
    for v in [8, 12, 15]:
        experiments.append(Experiment(f"max_trades_{v}", {"limits.max_trades_per_day": v}))
    return experiments


def _archived_phase6_candidates(cumulative_mutations: dict[str, Any]) -> list[Experiment]:
    """Finetune — perturb accepted mutations + structural param variants."""
    experiments = []

    # EMA period sweeps
    for v in [15, 20, 25]:
        experiments.append(Experiment(f"ema_trail_{v}", {"trail.ema_trail_period": v}))
    for v in [15, 20, 30]:
        experiments.append(Experiment(f"h1_ema_fast_{v}", {"h1_indicators.ema_fast": v}))

    # Compound trail perturbation
    current_wide = cumulative_mutations.get("trail.trail_buffer_wide", 1.2)
    current_tight = cumulative_mutations.get("trail.trail_buffer_tight", 0.1)
    experiments.append(Experiment("compound_trail_tighter", {
        "trail.trail_buffer_wide": round(current_wide * 0.85, 2),
        "trail.trail_buffer_tight": round(current_tight * 0.85, 2),
    }))

    # Quick exit fine-tuning (enabled by default in M15 config)
    for bars in [6, 10, 16]:
        experiments.append(Experiment(f"qe_bars_{bars}", {"exits.quick_exit_bars": bars}))
    for mfe in [0.1, 0.2, 0.3]:
        experiments.append(Experiment(f"qe_mfe_{mfe}", {"exits.quick_exit_max_mfe_r": mfe}))

    # H1 regime ADX perturbation (22.0 is now default)
    for v in [18.0, 20.0, 24.0, 26.0]:
        experiments.append(Experiment(f"h1_adx_{v}", {"regime.h1_min_adx": v}))
    # Direction filter combos
    experiments.append(Experiment("btc_long_only", {"symbol_filter.btc_direction": "long_only"}))
    experiments.append(Experiment("sol_long_only", {"symbol_filter.sol_direction": "long_only"}))

    # Perturb accepted mutations by ×[0.8, 1.2]
    perturbable = {
        "setup.impulse_min_atr_move", "setup.min_room_r", "setup.pullback_max_retrace",
        "trail.trail_r_ceiling", "trail.trail_buffer_wide", "trail.trail_buffer_tight",
        "exits.tp1_r", "exits.tp2_r", "exits.be_buffer_r",
        "exits.time_stop_bars", "stops.atr_mult",
        "risk.risk_pct_a", "risk.risk_pct_b",
        "regime.a_min_adx", "regime.b_min_adx", "regime.h1_min_adx",
    }
    for key, val in cumulative_mutations.items():
        if key in perturbable and isinstance(val, (int, float)):
            for mult in [0.8, 1.2]:
                new_val = round(val * mult, 4)
                experiments.append(Experiment(
                    f"perturb_{key.split('.')[-1]}_{mult}",
                    {key: new_val},
                ))

    return experiments


def _phase1_candidates() -> list[Experiment]:
    """Signal & setup discrimination before adding new entry paths."""
    return [
        Experiment("score_b_1_20", {"setup.min_setup_score_b": 1.20}),
        Experiment("score_b_1_50", {"setup.min_setup_score_b": 1.50}),
        Experiment("score_a_2_30", {"setup.min_setup_score_a": 2.30}),
        Experiment("room_1_20", {"setup.min_room_r": 1.20}),
        Experiment("room_1_80", {"setup.min_room_r": 1.80}),
        Experiment("impulse_atr_0_70", {"setup.impulse_min_atr_move": 0.70}),
        Experiment("impulse_atr_1_00", {"setup.impulse_min_atr_move": 1.00}),
        Experiment("pullback_bars_14", {"setup.pullback_max_bars": 14}),
        Experiment("weekly_room_0_80", {
            "setup.weekly_room_filter_enabled": True,
            "setup.min_weekly_room_r": 0.80,
        }),
        Experiment("quality_bundle", {
            "setup.min_setup_score_b": 1.50,
            "setup.min_room_r": 1.80,
            "setup.pullback_max_bars": 14,
        }),
    ]


def _phase2_candidates() -> list[Experiment]:
    """Confirmation and entry timing, with pending H1 confirmation support."""
    return [
        Experiment("confirm_b_pending_2", {
            "confirmation.require_confirmation_for_b": True,
            "confirmation.max_bars_after_setup": 2,
        }),
        Experiment("confirm_b_pending_3", {
            "confirmation.require_confirmation_for_b": True,
            "confirmation.max_bars_after_setup": 3,
        }),
        Experiment("confirm_all_pending_2", {
            "confirmation.require_confirmation": True,
            "confirmation.max_bars_after_setup": 2,
        }),
        Experiment("trigger_structure_or_ema", {
            "confirmation.enable_engulfing": False,
            "confirmation.enable_hammer": False,
        }),
        Experiment("volume_trigger_1_00", {"confirmation.volume_threshold_mult": 1.00}),
        Experiment("volume_trigger_1_15", {"confirmation.volume_threshold_mult": 1.15}),
        Experiment("entry_hybrid_grade", {"entry.mode": "hybrid_grade"}),
        Experiment("entry_break", {"entry.mode": "break"}),
        Experiment("entry_confirm_ttl_1", {"entry.max_bars_after_confirmation": 1}),
        Experiment("confirm_entry_bundle", {
            "confirmation.require_confirmation_for_b": True,
            "confirmation.max_bars_after_setup": 2,
            "entry.mode": "confirm_preferred",
        }),
    ]


def _phase3_candidates() -> list[Experiment]:
    """Early trade management focused on failed follow-through."""
    return [
        Experiment("scratch_off", {"exits.scratch_exit_enabled": False}),
        Experiment("scratch_floor_0_00", {"exits.scratch_floor_r": 0.00}),
        Experiment("scratch_peak_0_50_floor_0_10", {
            "exits.scratch_peak_r": 0.50,
            "exits.scratch_floor_r": 0.10,
        }),
        Experiment("scratch_min_bars_4", {"exits.scratch_min_bars": 4}),
        Experiment("mfe_lock_0_75_0_00", {
            "exits.mfe_lock_exit_enabled": True,
            "exits.mfe_lock_trigger_r": 0.75,
            "exits.mfe_lock_floor_r": 0.00,
        }),
        Experiment("mfe_lock_1_00_0_20", {
            "exits.mfe_lock_exit_enabled": True,
            "exits.mfe_lock_trigger_r": 1.00,
            "exits.mfe_lock_floor_r": 0.20,
        }),
        Experiment("quick_exit_8", {"exits.quick_exit_bars": 8}),
        Experiment("quick_exit_16", {"exits.quick_exit_bars": 16}),
        Experiment("time_stop_exit_10", {
            "exits.time_stop_bars": 10,
            "exits.time_stop_action": "exit",
        }),
        Experiment("failed_followthrough_bundle", {
            "exits.scratch_peak_r": 0.50,
            "exits.scratch_floor_r": 0.10,
            "exits.mfe_lock_exit_enabled": True,
            "exits.mfe_lock_trigger_r": 1.00,
            "exits.mfe_lock_floor_r": 0.20,
        }),
    ]


def _phase4_candidates() -> list[Experiment]:
    """Profit capture after entry and early-management candidates settle."""
    return [
        Experiment("trail_use_mfe", {"trail.trail_use_mfe_for_adaptive": True}),
        Experiment("trail_use_mfe_tighter", {
            "trail.trail_use_mfe_for_adaptive": True,
            "trail.trail_buffer_tight": 0.10,
        }),
        Experiment("trail_activation_0_50_bars_4", {
            "trail.trail_activation_r": 0.50,
            "trail.trail_activation_bars": 4,
        }),
        Experiment("structure_trail_on", {"trail.structure_trail_enabled": True}),
        Experiment("tp1_1_00", {"exits.tp1_r": 1.00}),
        Experiment("tp2_1_50_frac_0_45", {
            "exits.tp2_r": 1.50,
            "exits.tp2_frac": 0.45,
        }),
        Experiment("be_min_bars_2", {"exits.be_min_bars_above": 2}),
        Experiment("ema_failsafe_0_75", {"exits.ema_failsafe_min_expansion_r": 0.75}),
        Experiment("capture_bundle", {
            "trail.trail_use_mfe_for_adaptive": True,
            "exits.mfe_lock_exit_enabled": True,
            "exits.mfe_lock_trigger_r": 1.00,
            "exits.mfe_lock_floor_r": 0.20,
        }),
    ]


def _phase5_candidates() -> list[Experiment]:
    """Frequency expansion, guarded by immutable score and hard rejects."""
    return [
        Experiment("h1_adx_18", {"regime.h1_min_adx": 18.0}),
        Experiment("b_adx_rising_off", {"regime.b_adx_rising_required": False}),
        Experiment("room_1_20_expansion", {"setup.min_room_r": 1.20}),
        Experiment("pullback_bars_24", {"setup.pullback_max_bars": 24}),
        Experiment("orderly_volume_1_10", {"setup.orderly_max_countertrend_volume_ratio": 1.10}),
        Experiment("btc_both", {"symbol_filter.btc_direction": "both"}),
        Experiment("sol_disabled", {"symbol_filter.sol_direction": "disabled"}),
        Experiment("relative_strength_24", {
            "filters.relative_strength_filter_enabled": True,
            "filters.relative_strength_lookback": 24,
            "filters.relative_strength_min_delta": 0.0,
        }),
        Experiment("coverage_bundle", {
            "regime.h1_min_adx": 18.0,
            "setup.min_room_r": 1.20,
            "setup.pullback_max_bars": 24,
        }),
    ]


def _phase6_candidates(cumulative_mutations: dict[str, Any]) -> list[Experiment]:
    """Risk scaling and small perturbations after structural alpha tests."""
    experiments = [
        Experiment("risk_b_0_018", {"risk.risk_pct_b": 0.018}),
        Experiment("risk_b_0_024", {"risk.risk_pct_b": 0.024}),
        Experiment("risk_a_0_010", {"risk.risk_pct_a": 0.010}),
        Experiment("risk_a_0_015", {"risk.risk_pct_a": 0.015}),
        Experiment("max_pos_4", {"limits.max_concurrent_positions": 4}),
        Experiment("max_pos_6", {"limits.max_concurrent_positions": 6}),
        Experiment("daily_loss_0_035", {"limits.max_daily_loss_pct": 0.035}),
        Experiment("daily_loss_0_050", {"limits.max_daily_loss_pct": 0.050}),
        Experiment("corr_risk_0_050", {"limits.max_correlated_risk_pct": 0.050}),
    ]

    perturbable = {
        "setup.min_setup_score_b",
        "setup.min_room_r",
        "setup.impulse_min_atr_move",
        "setup.pullback_max_bars",
        "confirmation.volume_threshold_mult",
        "trail.trail_buffer_tight",
        "exits.scratch_peak_r",
        "exits.scratch_floor_r",
        "exits.mfe_lock_trigger_r",
        "exits.mfe_lock_floor_r",
        "risk.risk_pct_a",
        "risk.risk_pct_b",
        "regime.h1_min_adx",
    }
    for key, val in cumulative_mutations.items():
        if key in perturbable and isinstance(val, (int, float)):
            for mult in (0.9, 1.1):
                experiments.append(Experiment(
                    f"perturb_{key.split('.')[-1]}_{mult}",
                    {key: round(val * mult, 4)},
                ))

    return experiments


PHASE_CANDIDATES = {
    1: _phase1_candidates,
    2: _phase2_candidates,
    3: _phase3_candidates,
    4: _phase4_candidates,
    5: _phase5_candidates,
}


class TrendPlugin:
    """StrategyPlugin implementation for trend-following strategy."""

    def __init__(
        self,
        backtest_config: BacktestConfig,
        base_config: TrendConfig,
        data_dir: Path = Path("data"),
        max_workers: int | None = None,
    ) -> None:
        # Enforce warmup for trend strategy — D1 EMA50 needs 51 bars
        backtest_config.warmup_days = max(backtest_config.warmup_days, 60)
        self.backtest_config = backtest_config
        self.base_config = base_config
        self.data_dir = data_dir
        self.max_workers = max_workers
        self._last_result: Any = None
        self._cached_store: Any = None

    @property
    def name(self) -> str:
        return "trend_anchor"

    @property
    def num_phases(self) -> int:
        return 6

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "total_trades": 30.0,
            "win_rate": 45.0,
            "profit_factor": 1.5,
            "max_drawdown_pct": 25.0,
            "sharpe_ratio": 1.5,
            "calmar_ratio": 2.0,
        }

    @property
    def initial_mutations(self) -> dict[str, Any]:
        return {}

    def get_phase_spec(self, phase: int, state: Any) -> PhaseSpec:
        if phase == 6:
            from crypto_trader.optimize.phase_state import PhaseState
            candidates = _phase6_candidates(
                state.cumulative_mutations if isinstance(state, PhaseState) else {}
            )
        else:
            gen = PHASE_CANDIDATES.get(phase)
            if gen is None:
                raise ValueError(f"Unknown phase: {phase}")
            candidates = gen()

        scoring = dict(PHASE_SCORING_EMPHASIS.get(phase, SCORING_WEIGHTS))
        gate_criteria = self._build_gate_criteria(phase)

        return PhaseSpec(
            phase_num=phase,
            name=PHASE_NAMES[phase],
            candidates=candidates,
            scoring_weights=scoring,
            hard_rejects=dict(HARD_REJECTS),
            min_delta=0.004,
            max_rounds=4,
            prune_threshold=0.0,
            gate_criteria=gate_criteria,
            gate_criteria_fn=lambda m, _p=phase: self._gate_criteria_fn(m, _p),
            analysis_policy=PhaseAnalysisPolicy(
                max_scoring_retries=1,
                max_diagnostic_retries=1,
                focus_metrics=[
                    "net_return_pct", "total_trades", "profit_factor",
                    "expectancy_r", "exit_efficiency", "avg_mae_r",
                    "max_drawdown_pct",
                ],
                diagnostic_gap_fn=lambda p, m: self._diagnostic_gap_fn(p, m),
                suggest_experiments_fn=lambda p, m, w, s: self._suggest_experiments_fn(p, m, w, s),
                decide_action_fn=lambda *args: self._decide_action_fn(*args),
                redesign_scoring_weights_fn=lambda *args: self._redesign_scoring_weights_fn(*args),
                build_extra_analysis_fn=lambda p, m, s, g: self._build_extra_analysis_fn(p, m, s, g),
                format_extra_analysis_fn=lambda d: self._format_extra_analysis_fn(d),
            ),
            focus=PHASE_NAMES[phase],
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        scoring_weights: dict[str, float],
        hard_rejects: dict[str, tuple[str, float]],
    ) -> EvaluateFn:
        ceilings = SCORING_CEILINGS

        def evaluate_fn(
            candidates: list[Experiment],
            current_mutations: dict[str, Any],
        ) -> list[ScoredCandidate]:
            return evaluate_parallel(
                candidates=candidates,
                current_mutations=current_mutations,
                cumulative_mutations=cumulative_mutations,
                base_config=self.base_config,
                backtest_config=self.backtest_config,
                data_dir=self.data_dir,
                scoring_weights=scoring_weights,
                hard_rejects=hard_rejects,
                phase=phase,
                max_workers=self.max_workers,
                strategy_type="trend",
                ceilings=ceilings,
            )

        return evaluate_fn

    def _get_store(self):
        """Lazily build and cache an in-memory store to avoid repeated disk I/O."""
        if self._cached_store is None:
            from crypto_trader.optimize.parallel import _CachedStore
            symbols = self.backtest_config.symbols or self.base_config.symbols
            raw = ParquetStore(base_dir=self.data_dir)
            self._cached_store = _CachedStore(raw, symbols, ["15m", "1h", "1d"])
        return self._cached_store

    def compute_final_metrics(
        self,
        mutations: dict[str, Any],
    ) -> dict[str, float]:
        """Run full-period backtest and return metrics for gate evaluation."""
        config = apply_mutations(self.base_config, mutations)
        result = run(
            config, self.backtest_config, self.data_dir,
            strategy_type="trend",
            store=self._get_store(),
        )
        self._last_result = result
        return metrics_to_dict(result.metrics)

    def run_phase_diagnostics(
        self,
        phase: int,
        state: Any,
        metrics: dict[str, float],
        greedy_result: GreedyResult,
    ) -> str:
        """Generate phase-targeted diagnostics with insights enrichment."""
        if self._last_result is None:
            mutations = state.cumulative_mutations if state else {}
            self.compute_final_metrics(mutations)

        result = self._last_result
        trades = _result_trades(result)
        terminal_marks = _result_terminal_marks(result)
        pm = getattr(result, "metrics", None) if result else None
        if not result or (not trades and not terminal_marks):
            return "No trades to diagnose."

        lines: list[str] = []

        # Phase summary header
        phase_name = PHASE_NAMES.get(phase, "Unknown") if phase else "Full"
        lines.append(f"=== Trend Diagnostics (Phase {phase}: {phase_name}) ===")
        lines.append(f"Trades: {len(trades)}, "
                      f"WR: {sum(1 for t in trades if t.net_pnl > 0) / max(len(trades), 1) * 100:.0f}%")
        if terminal_marks:
            lines.append(f"Terminal marks: {len(terminal_marks)}")
        lines.append("")

        # DiagnosticInsights enrichment
        try:
            from crypto_trader.backtest.diagnostics import (
                extract_diagnostic_insights,
                generate_phase_diagnostics,
            )
            if trades:
                insights = extract_diagnostic_insights(trades)

                # Per-confirmation breakdown
                if insights.per_confirmation:
                    lines.append("--- Per-Confirmation ---")
                    for ctype, stats in insights.per_confirmation.items():
                        n = stats.get("n", 0)
                        wr = stats.get("wr", 0)
                        avg_r = stats.get("avg_r", 0)
                        lines.append(f"  {ctype}: n={n}, WR={wr:.0f}%, avg_r={avg_r:+.3f}")
                    lines.append("")

                # Per-asset breakdown
                if insights.per_asset:
                    lines.append("--- Per-Asset ---")
                    for asset, stats in insights.per_asset.items():
                        n = stats.get("n", 0)
                        wr = stats.get("wr", 0)
                        avg_r = stats.get("avg_r", 0)
                        lines.append(f"  {asset}: n={n}, WR={wr:.0f}%, avg_r={avg_r:+.3f}")
                    lines.append("")

                # Exit attribution
                if insights.exit_attribution:
                    lines.append("--- Exit Attribution ---")
                    for reason, stats in insights.exit_attribution.items():
                        n = stats.get("n", 0)
                        avg_r = stats.get("avg_r", 0)
                        lines.append(f"  {reason}: n={n}, avg_r={avg_r:+.3f}")
                    lines.append("")

                # MFE capture
                if insights.mfe_capture:
                    cap = insights.mfe_capture
                    lines.append("--- MFE Capture ---")
                    lines.append(f"  Avg MFE: {cap.get('avg_mfe_r', 0):.3f}R")
                    lines.append(f"  Capture: {cap.get('avg_capture_pct', 0):.1%}")
                    lines.append(f"  Giveback: {cap.get('avg_giveback_pct', 0):.1%}")
                    lines.append("")

                # Direction
                if insights.direction:
                    lines.append("--- Direction ---")
                    for d, stats in insights.direction.items():
                        n = stats.get("n", 0)
                        avg_r = stats.get("avg_r", 0)
                        lines.append(f"  {d}: n={n}, avg_r={avg_r:+.3f}")
                    lines.append("")

            # Phase-targeted diagnostic sections
            modules = PHASE_DIAGNOSTIC_MODULES.get(phase, ["D6"]) if phase else ["D6"]
            lines.append(generate_phase_diagnostics(
                trades,
                modules,
                initial_equity=_diagnostic_initial_equity(self.backtest_config),
                title=f"Phase {phase} ({phase_name})" if phase else "Full",
                terminal_marks=terminal_marks,
                performance_metrics=pm,
            ))

        except Exception as e:
            log.warning("trend.diagnostics_error", error=str(e))
            lines.append(
                generate_diagnostics(
                    trades,
                    initial_equity=_diagnostic_initial_equity(self.backtest_config),
                    terminal_marks=terminal_marks,
                    performance_metrics=pm,
                )
            )

        return "\n".join(lines)

    def run_enhanced_diagnostics(
        self,
        phase: int,
        state: Any,
        metrics: dict[str, float],
        greedy_result: GreedyResult,
    ) -> str:
        """Generate full diagnostics using all modules."""
        if self._last_result is None:
            mutations = state.cumulative_mutations if state else {}
            self.compute_final_metrics(mutations)

        result = self._last_result
        trades = _result_trades(result)
        terminal_marks = _result_terminal_marks(result)
        pm = getattr(result, "metrics", None) if result else None
        if not result or (not trades and not terminal_marks):
            return "No trades to diagnose."

        try:
            from crypto_trader.backtest.diagnostics import generate_phase_diagnostics
            return generate_phase_diagnostics(
                trades,
                ["D1", "D2", "D3", "D4", "D5", "D6"],
                initial_equity=_diagnostic_initial_equity(self.backtest_config),
                title="Trend Enhanced Diagnostics",
                terminal_marks=terminal_marks,
                performance_metrics=pm,
            )
        except Exception:
            return generate_diagnostics(
                trades,
                initial_equity=_diagnostic_initial_equity(self.backtest_config),
                terminal_marks=terminal_marks,
                performance_metrics=pm,
            )

    def build_end_of_round_artifacts(
        self,
        state: Any,
    ) -> EndOfRoundArtifacts:
        """Build end-of-round evaluation with insights enrichment."""
        # Ensure we have a fresh result with cumulative mutations
        if self._last_result is None and state is not None:
            from crypto_trader.optimize.phase_state import PhaseState
            if isinstance(state, PhaseState):
                self.compute_final_metrics(state.cumulative_mutations)

        trades = _result_trades(self._last_result)
        terminal_marks = _result_terminal_marks(self._last_result)
        pm = getattr(self._last_result, "metrics", None) if self._last_result else None

        # Full diagnostics text
        diagnostics_text = generate_diagnostics(
            trades,
            initial_equity=_diagnostic_initial_equity(self.backtest_config),
            terminal_marks=terminal_marks,
            performance_metrics=pm,
        ) if (trades or terminal_marks) else "(no trades)"

        # Build dimension reports with insights
        dimension_reports = {}
        if pm:
            try:
                from crypto_trader.backtest.diagnostics import extract_diagnostic_insights
                insights = extract_diagnostic_insights(trades) if trades else None
                report = build_evaluation_report(pm, insights=insights)
            except Exception:
                report = build_evaluation_report(pm)

            for dim_name, dim_data in report.items():
                dimension_reports[dim_name] = format_dimension_text(dim_name, dim_data)

        verdict = self._build_verdict(pm) if pm else "No metrics available."

        return EndOfRoundArtifacts(
            final_diagnostics_text=diagnostics_text,
            dimension_reports=dimension_reports,
            overall_verdict=verdict,
        )

    # ─── Policy callbacks ─────────────────────────────────────────────────

    def _diagnostic_gap_fn(
        self, phase: int, metrics: dict[str, float],
    ) -> list[str]:
        """Identify diagnostic gaps using DiagnosticInsights."""
        gaps: list[str] = []
        trades_n = metrics.get("total_trades", 0)
        dd = metrics.get("max_drawdown_pct", 0)
        pf = metrics.get("profit_factor", 0)

        # Basic metric gaps
        if trades_n < 5:
            gaps.append(f"Low trade count ({trades_n:.0f}). Regime/setup filters may be too tight.")
        if dd > 35:
            gaps.append(f"High drawdown ({dd:.1f}%). [D1] Trail/stop calibration needed.")
        if pf < 1.2:
            gaps.append(f"Low profit factor ({pf:.2f}). [D4] Signal quality needs investigation.")

        # Insights-driven gaps
        if self._last_result and self._last_result.trades:
            try:
                from crypto_trader.backtest.diagnostics import extract_diagnostic_insights
                insights = extract_diagnostic_insights(self._last_result.trades)

                # MFE capture gaps (phases 3-4: trail & exit)
                if phase in (3, 4) and insights.mfe_capture:
                    cap = insights.mfe_capture.get("avg_capture_pct", 0)
                    giveback = insights.mfe_capture.get("avg_giveback_pct", 0)
                    if cap < 0.40:
                        gaps.append(f"Low MFE capture ({cap:.0%}). [D1] Trail too wide or exits too early.")
                    if giveback > 0.60:
                        gaps.append(f"High giveback ({giveback:.0%}). [D1] Trail too loose after peak R.")

                # Negative-R confirmations (phases 1-2: signal & regime)
                if phase in (1, 2) and insights.per_confirmation:
                    for ctype, stats in insights.per_confirmation.items():
                        n = stats.get("n", 0)
                        avg_r = stats.get("avg_r", 0)
                        if n >= 2 and avg_r < -0.3:
                            gaps.append(f"Confirmation '{ctype}' has avg_r={avg_r:+.2f} (n={n}). "
                                        f"[D4] Consider disabling.")

                # Concentration risk (phases 1-2, 5)
                if phase in (1, 2, 5) and insights.concentration:
                    top_pct = insights.concentration.get("top1_pct", 0)
                    if top_pct > 60:
                        gaps.append(f"Top trade is {top_pct:.0f}% of total P&L. [D3] Over-concentrated.")

                # Direction imbalance (phases 2, 5)
                if phase in (2, 5) and insights.direction:
                    long_stats = insights.direction.get("long", {})
                    short_stats = insights.direction.get("short", {})
                    long_n = long_stats.get("n", 0)
                    short_n = short_stats.get("n", 0)
                    if long_n > 0 and short_n > 0:
                        long_r = long_stats.get("avg_r", 0)
                        short_r = short_stats.get("avg_r", 0)
                        if abs(long_r - short_r) > 0.5:
                            weaker = "short" if long_r > short_r else "long"
                            gaps.append(f"Direction imbalance: {weaker} avg_r much lower. "
                                        f"[D5] Consider direction filtering.")

                # Duration outliers (phases 3-4: trail & exit)
                if phase in (3, 4) and insights.duration:
                    avg_bars = insights.duration.get("avg_bars", 0)
                    if avg_bars > 20:
                        gaps.append(f"Avg hold {avg_bars:.0f} bars. [D1] Trades may be stagnating.")

                # Stop-out dominance (phases 3-4)
                if phase in (3, 4) and insights.exit_attribution:
                    stop_stats = insights.exit_attribution.get("protective_stop", {})
                    stop_n = stop_stats.get("n", 0)
                    total = len(self._last_result.trades)
                    if total > 0 and stop_n / total > 0.7:
                        gaps.append(f"Stop-outs dominate ({stop_n}/{total}). "
                                    f"[D1] Trail/TP may be mis-calibrated.")

                # Trade count critically low (phase 1)
                if phase == 1 and trades_n < 6:
                    gaps.append("Trade count critically low. [D4] Consider relaxing "
                                "require_completed_impulse or lowering impulse_min_atr_move.")

            except Exception:
                pass  # Fall back to basic gaps only

        return gaps

    def _suggest_experiments_fn(
        self, phase: int, metrics: dict[str, float],
        weaknesses: list[str], state: Any,
    ) -> list[Experiment]:
        """Suggest targeted experiments based on trade insights."""
        suggestions: list[Experiment] = []

        if not self._last_result or not self._last_result.trades:
            return suggestions

        try:
            from crypto_trader.backtest.diagnostics import extract_diagnostic_insights
            insights = extract_diagnostic_insights(self._last_result.trades)
        except Exception:
            return suggestions

        trades_n = len(self._last_result.trades)

        # Phase 1: Signal — suggest structural relaxation for low trade count
        if phase == 1:
            if trades_n < 12:
                suggestions.append(Experiment(
                    "diag_impulse_atr_1", {"setup.impulse_min_atr_move": 1.0}
                ))
                suggestions.append(Experiment(
                    "diag_room_08", {"setup.min_room_r": 0.8}
                ))
            # Disable underperforming confirmations
            if insights.per_confirmation:
                for ctype, stats in insights.per_confirmation.items():
                    n = stats.get("n", 0)
                    avg_r = stats.get("avg_r", 0)
                    if n >= 2 and avg_r < -0.2:
                        config_path = CONFIRMATION_DISABLE_MAP.get(ctype)
                        if config_path:
                            suggestions.append(Experiment(
                                f"diag_disable_{ctype}", {config_path: False}
                            ))

        # Phase 2: Regime — suggest symbol direction filters
        if phase == 2:
            if insights.per_asset:
                sol_stats = insights.per_asset.get("SOL", {})
                if sol_stats.get("n", 0) == 0:
                    suggestions.append(Experiment(
                        "diag_sol_disabled", {"symbol_filter.sol_direction": "disabled"},
                    ))
                for asset, stats in insights.per_asset.items():
                    n = stats.get("n", 0)
                    avg_r = stats.get("avg_r", 0)
                    if n >= 2 and avg_r < -0.3:
                        suggestions.append(Experiment(
                            f"diag_{asset.lower()}_disabled",
                            {f"symbol_filter.{asset.lower()}_direction": "disabled"},
                        ))
            # ETH short WR check
            if insights.direction and insights.per_asset:
                eth_stats = insights.per_asset.get("ETH", {})
                if eth_stats.get("n", 0) >= 2 and eth_stats.get("wr", 0) < 40:
                    suggestions.append(Experiment(
                        "diag_eth_long_only", {"symbol_filter.eth_direction": "long_only"},
                    ))
            # Direction imbalance → long_only
            if insights.direction:
                for d_name in ("long", "short"):
                    d_stats = insights.direction.get(d_name, {})
                    if d_stats.get("n", 0) >= 2 and d_stats.get("avg_r", 0) < -0.3:
                        other = "long_only" if d_name == "short" else "short_only"
                        for asset in ["btc", "eth", "sol"]:
                            suggestions.append(Experiment(
                                f"diag_{asset}_{other}",
                                {f"symbol_filter.{asset}_direction": other},
                            ))
                        break  # One direction filter suggestion set is enough

        # Phase 3: Trail — capture-aware suggestions
        if phase == 3:
            if insights.mfe_capture:
                cap = insights.mfe_capture.get("avg_capture_pct", 0)
                if cap < 0.50:
                    # Low capture — try perturbing trail ceiling
                    suggestions.append(Experiment(
                        "diag_trail_ceiling_1_0", {"trail.trail_r_ceiling": 1.0}
                    ))
                    suggestions.append(Experiment(
                        "diag_trail_ceiling_0_6", {"trail.trail_r_ceiling": 0.6}
                    ))

        # Phase 3-4: Trail & Exit adjustments
        if phase in (3, 4):
            if insights.mfe_capture:
                giveback = insights.mfe_capture.get("avg_giveback_pct", 0)
                if giveback > 0.50:
                    suggestions.append(Experiment(
                        "diag_trail_tighter", {"trail.trail_buffer_tight": 0.2}
                    ))
                    suggestions.append(Experiment(
                        "diag_trail_ceiling_low", {"trail.trail_r_ceiling": 1.5}
                    ))

            # Stop-out share too high → try wider stop
            if insights.exit_attribution:
                stop_stats = insights.exit_attribution.get("protective_stop", {})
                stop_n = stop_stats.get("n", 0)
                if trades_n > 0 and stop_n / trades_n > 0.6:
                    suggestions.append(Experiment(
                        "diag_wider_stop", {"stops.atr_mult": 1.5}
                    ))

            # Duration-based: if avg hold > 8 bars with low R, try quick exit
            if insights.duration:
                avg_bars = insights.duration.get("avg_bars", 0)
                if avg_bars > 8:
                    suggestions.append(Experiment(
                        "diag_quick_exit", {
                            "exits.quick_exit_enabled": True,
                            "exits.quick_exit_bars": 8,
                            "exits.quick_exit_max_mfe_r": 0.2,
                            "exits.quick_exit_max_r": -0.2,
                        }
                    ))

        # Phase 5: Risk sizing
        if phase == 5:
            dd = metrics.get("max_drawdown_pct", 0)
            if dd > 30:
                suggestions.append(Experiment(
                    "diag_risk_reduce", {"risk.risk_pct_b": 0.003}
                ))
            if insights.concentration:
                top_pct = insights.concentration.get("top1_pct", 0)
                if top_pct > 50:
                    suggestions.append(Experiment(
                        "diag_max_pos_2", {"limits.max_concurrent_positions": 2}
                    ))

        # Phase 6: Fine-tuning
        if phase == 6:
            # Direction filter if one side is clearly worse
            if insights.direction:
                long_stats = insights.direction.get("long", {})
                short_stats = insights.direction.get("short", {})
                long_r = long_stats.get("avg_r", 0)
                short_r = short_stats.get("avg_r", 0)
                if long_stats.get("n", 0) >= 2 and short_stats.get("n", 0) >= 2:
                    if short_r < -0.3 and long_r > 0:
                        suggestions.append(Experiment(
                            "diag_regime_require_structure",
                            {"regime.require_structure": True}
                        ))

            # Quick exit if duration analysis shows stagnation
            if insights.duration:
                avg_bars = insights.duration.get("avg_bars", 0)
                if avg_bars > 15:
                    suggestions.append(Experiment(
                        "diag_quick_exit_ft", {
                            "exits.quick_exit_enabled": True,
                            "exits.quick_exit_bars": 6,
                        }
                    ))

        return suggestions

    def _decide_action_fn(
        self,
        phase: int,
        metrics: dict[str, float],
        state: Any,
        greedy_result: GreedyResult,
        gate_result: Any,
        current_weights: dict[str, float],
        goal_progress: dict[str, dict],
        max_scoring: int,
        max_diag: int,
    ) -> PhaseDecision | None:
        return None  # Use default fallback

    def _redesign_scoring_weights_fn(
        self, phase: int, current_weights: dict[str, float],
        metrics: dict[str, float], strengths: list[str], weaknesses: list[str],
    ) -> dict[str, float] | None:
        return None  # Keep immutable seven-component score

    def _build_extra_analysis_fn(
        self, phase: int, metrics: dict[str, float],
        state: Any, greedy_result: Any,
    ) -> dict[str, Any]:
        """Build phase-specific extra analysis data."""
        extra: dict[str, Any] = {}

        if not self._last_result or not self._last_result.trades:
            return extra

        try:
            from crypto_trader.backtest.diagnostics import extract_diagnostic_insights
            insights = extract_diagnostic_insights(self._last_result.trades)
        except Exception:
            return extra

        if phase in (1, 2):
            # Signal & regime quality
            if insights.per_confirmation:
                extra["confirmation_breakdown"] = {
                    k: {"n": v.get("n", 0), "avg_r": v.get("avg_r", 0)}
                    for k, v in insights.per_confirmation.items()
                }
            if insights.confluence:
                extra["confluence_ladder"] = {
                    str(k): {"n": v.get("n", 0), "avg_r": v.get("avg_r", 0)}
                    for k, v in insights.confluence.items()
                }
            if insights.per_asset:
                extra["per_asset_edge"] = {
                    k: {"n": v.get("n", 0), "avg_r": v.get("avg_r", 0)}
                    for k, v in insights.per_asset.items()
                }

        elif phase in (3, 4):
            # Trail & exit efficiency
            if insights.mfe_capture:
                extra["avg_mfe_r"] = insights.mfe_capture.get("avg_mfe_r", 0)
                extra["capture_pct"] = insights.mfe_capture.get("avg_capture_pct", 0)
                extra["giveback_pct"] = insights.mfe_capture.get("avg_giveback_pct", 0)
            if insights.exit_attribution:
                extra["exit_reasons"] = {k: v.get("n", 0) for k, v in insights.exit_attribution.items()}
            if insights.duration:
                extra["avg_bars_held"] = insights.duration.get("avg_bars", 0)

        elif phase in (5, 6):
            # Risk & overall
            if insights.concentration:
                extra["top_trade_pct"] = insights.concentration.get("top1_pct", 0)
            if insights.direction:
                extra["direction_breakdown"] = {
                    k: {"n": v.get("n", 0), "avg_r": v.get("avg_r", 0)}
                    for k, v in insights.direction.items()
                }
            if insights.r_stats:
                extra["r_skew"] = insights.r_stats.get("skew", 0)
                extra["r_mean"] = insights.r_stats.get("mean", 0)

        return extra

    def _format_extra_analysis_fn(
        self,
        extra_data: dict[str, Any],
    ) -> str:
        """Render extra analysis dict into text lines."""
        if not extra_data:
            return ""
        lines = []
        for k, v in extra_data.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            elif isinstance(v, dict):
                lines.append(f"  {k}:")
                for dk, dv in v.items():
                    lines.append(f"    {dk}: {dv}")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def _build_gate_criteria(self, phase: int = 0) -> list[GateCriterion]:
        """Build gate criteria — phase-specific when available, else from HARD_REJECTS."""
        if phase in PHASE_GATE_CRITERIA:
            return list(PHASE_GATE_CRITERIA[phase])
        criteria = []
        for metric, (op, threshold) in HARD_REJECTS.items():
            criteria.append(GateCriterion(
                metric=metric, operator=op, threshold=threshold
            ))
        return criteria

    def _gate_criteria_fn(
        self, metrics: dict[str, float], phase: int = 0,
    ) -> list[GateCriterion]:
        """Dynamic gate criteria based on phase."""
        return self._build_gate_criteria(phase)

    def _build_verdict(self, pm: PerformanceMetrics) -> str:
        """Build actionable verdict for end-of-round report."""
        lines = [
            f"Trades: {pm.total_trades}, Win rate: {pm.win_rate:.1f}%",
            f"PF: {pm.profit_factor:.2f}, Sharpe: {pm.sharpe_ratio:.2f}",
            f"Max DD: {pm.max_drawdown_pct:.1f}%, Net return: {pm.net_return_pct:.2f}%",
        ]

        # Actionable observations
        if pm.total_trades < 10:
            lines.append("Action: Loosen regime/setup filters for more trades.")
        if pm.max_drawdown_pct > 30:
            lines.append("Action: Tighten trail/stop for DD control.")
        if pm.profit_factor < 1.0:
            lines.append("Action: Improve signal quality or exit timing.")
        if pm.profit_factor >= 2.0 and pm.total_trades >= 10:
            lines.append("Strong edge detected. Focus on coverage expansion.")

        return "\n".join(lines)
