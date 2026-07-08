"""BreakoutPlugin — 6-phase optimization plugin for the volume profile breakout strategy.

Round 2 prep: structural parity with trend strategy — disabled Model 2
(25% WR, -1.80R), tighter stops (use_farther=False), raised TP1 to 0.8R/30%
(match trend pattern), enabled quick exit, time_stop_action=reduce, earlier
trail activation (0.3R/4 bars).  Baked 3 R1 mutations + 14 structural defaults.
Scoring rebalanced for profitable-baseline regime.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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
from crypto_trader.optimize.config_mutator import apply_mutations
from crypto_trader.optimize.evaluation import (
    build_evaluation_report,
    format_dimension_text,
)
from crypto_trader.optimize.parallel import evaluate_parallel
from crypto_trader.optimize.scoring import composite_score
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
from crypto_trader.strategy.breakout.config import BreakoutConfig

log = structlog.get_logger("optimize.breakout")


def _diagnostic_initial_equity(config: Any, default: float = 10_000.0) -> float:
    value = getattr(config, "initial_equity", default)
    return float(value) if isinstance(value, Real) else default


def _result_trades(result: Any) -> list[Any]:
    trades = getattr(result, "trades", []) if result is not None else []
    return trades if isinstance(trades, list) else []


def _result_terminal_marks(result: Any) -> list[Any]:
    terminal_marks = getattr(result, "terminal_marks", []) if result is not None else []
    return terminal_marks if isinstance(terminal_marks, list) else []


def _result_diagnostic_context(result: Any) -> dict[str, Any]:
    context = getattr(result, "diagnostic_context", {}) if result is not None else {}
    return dict(context) if isinstance(context, dict) else {}

# ── Scoring ceilings (calibrated for profitable-baseline regime) ──────────────

SCORING_CEILINGS: dict[str, float] = {
    "returns":  30.0,   # 30% = 1.0 (wider — expect higher returns)
    "edge":     10.0,   # PF 11 = 1.0 (wider — reduce saturation)
    "coverage": 15.0,   # 15 trades = 1.0 (tighter — model1 only)
}

# ── Scoring weights (returns-dominant for profitable baseline) ────────────────

SCORING_WEIGHTS: dict[str, float] = {
    "returns":  0.30,
    "coverage": 0.20,
    "edge":     0.20,
    "calmar":   0.15,
    "capture":  0.15,
}

HARD_REJECTS: dict[str, tuple[str, float]] = {
    "max_drawdown_pct": ("<=", 40.0),
    "total_trades": (">=", 5),
    "profit_factor": (">=", 0.8),   # Expect profitable baseline after structural fixes
}

PHASE_GATE_CRITERIA: dict[int, list[GateCriterion]] = {
    1: [
        GateCriterion(metric="total_trades", operator=">=", threshold=5),
    ],
    2: [
        GateCriterion(metric="total_trades", operator=">=", threshold=5),
        GateCriterion(metric="profit_factor", operator=">=", threshold=0.8),
    ],
    3: [
        GateCriterion(metric="total_trades", operator=">=", threshold=5),
        GateCriterion(metric="profit_factor", operator=">=", threshold=0.8),
    ],
    4: [
        GateCriterion(metric="total_trades", operator=">=", threshold=5),
        GateCriterion(metric="profit_factor", operator=">=", threshold=0.7),
    ],
    5: [
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=35.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=0.8),
    ],
    6: [
        GateCriterion(metric="total_trades", operator=">=", threshold=5),
        GateCriterion(metric="profit_factor", operator=">=", threshold=0.7),
    ],
}

PHASE_DIAGNOSTIC_MODULES: dict[int, list[str]] = {
    1: ["D4", "D5", "D6"],                       # Signal & Direction
    2: ["D2", "D3", "D6"],                        # Exit & Capture
    3: ["D1", "D6"],                              # Trail & Stop
    4: ["D4", "D5", "D6"],                        # Zone & Profile
    5: ["D3", "D6"],                              # Risk & Sizing
    6: ["D1", "D2", "D3", "D4", "D5", "D6"],     # Finetune
}

PHASE_SCORING_EMPHASIS: dict[int, dict[str, float]] = {
    1: {"returns": 0.20, "coverage": 0.30, "edge": 0.25, "calmar": 0.10, "capture": 0.15},
    2: {"returns": 0.25, "coverage": 0.15, "edge": 0.15, "calmar": 0.15, "capture": 0.30},
    3: {"returns": 0.20, "coverage": 0.15, "edge": 0.15, "calmar": 0.30, "capture": 0.20},
    4: {"returns": 0.20, "coverage": 0.35, "edge": 0.20, "calmar": 0.10, "capture": 0.15},
    5: {"returns": 0.35, "coverage": 0.15, "edge": 0.15, "calmar": 0.25, "capture": 0.10},
    6: {"returns": 0.30, "coverage": 0.20, "edge": 0.20, "calmar": 0.15, "capture": 0.15},
}

PHASE_NAMES: dict[int, str] = {
    1: "Signal & Direction",
    2: "Exit & Capture",
    3: "Trail & Stop",
    4: "Zone & Profile",
    5: "Risk & Sizing",
    6: "Finetune",
}


# ── Phase candidate generators ───────────────────────────────────────────

def _phase1_candidates() -> list[Experiment]:
    """Signal & Direction — re-centered on model1-only baseline.

    Model 2 disabled by default (25% WR, -1.80R).  Test re-enabling it.
    Direction close disabled by default (baked R1).  Experiments centered
    on new profitable-baseline defaults.
    """
    experiments = []

    # Direction filters (highest impact, most statistically robust)
    experiments.append(Experiment("eth_both", {"symbol_filter.eth_direction": "both"}))
    experiments.append(Experiment("eth_disabled", {"symbol_filter.eth_direction": "disabled"}))
    experiments.append(Experiment("sol_long_only", {"symbol_filter.sol_direction": "long_only"}))

    # Confluence quality
    experiments.append(Experiment("b_conf_1", {"setup.min_confluences_b": 1}))
    experiments.append(Experiment("b_conf_2", {"setup.min_confluences_b": 2}))
    experiments.append(Experiment("a_conf_3", {"setup.min_confluences_a": 3}))

    # Signal strength
    experiments.append(Experiment("require_vol_surge", {"setup.require_volume_surge": True}))
    experiments.append(Experiment("min_bo_atr_03", {"setup.min_breakout_atr": 0.3}))
    experiments.append(Experiment("min_bo_atr_05", {"setup.min_breakout_atr": 0.5}))
    experiments.append(Experiment("body_ratio_045", {"setup.body_ratio_min": 0.45}))
    experiments.append(Experiment("body_ratio_055", {"setup.body_ratio_min": 0.55}))

    # Room R (minimum reward potential)
    experiments.append(Experiment("room_r_b_15", {"setup.min_room_r_b": 1.5}))
    experiments.append(Experiment("room_r_a_22", {"setup.min_room_r_a": 2.2}))

    # Context alignment
    experiments.append(Experiment("no_countertrend", {"context.allow_countertrend": False}))
    experiments.append(Experiment("h4_adx_15", {"context.h4_adx_threshold": 15.0}))
    experiments.append(Experiment("h4_adx_20", {"context.h4_adx_threshold": 20.0}))

    # Entry model — test re-enabling Model 2 (disabled by default)
    experiments.append(Experiment("model2_on", {"confirmation.enable_model2": True}))
    experiments.append(Experiment("retest_bars_8", {"confirmation.retest_max_bars": 8}))
    experiments.append(Experiment("retest_bars_12", {"confirmation.retest_max_bars": 12}))
    experiments.append(Experiment("retest_zone_08", {"confirmation.retest_zone_atr": 0.8}))

    # Model 1 confirmation quality
    experiments.append(Experiment("no_vol_gate", {"confirmation.model1_require_volume": False}))
    experiments.append(Experiment("vol_mult_13", {"confirmation.model1_min_volume_mult": 1.3}))
    experiments.append(Experiment("dir_close_on", {"confirmation.model1_require_direction_close": True}))

    return experiments


def _phase2_candidates() -> list[Experiment]:
    """Exit & Capture — centered on new TP1=0.8R/30% default.

    TP structure aligned with trend: TP1 0.8R/30%, TP2 2.0R/40%, runner 30%.
    Quick exit enabled, time_stop_action=reduce.  BE buffer baked to 0.525.
    """
    experiments = []

    # TP1 (default now 0.8R) — test values around new default
    for v in [0.5, 0.6, 1.0, 1.2]:
        experiments.append(Experiment(f"tp1_r_{v}", {"exits.tp1_r": v}))

    # TP1 fraction (default now 0.3)
    for v in [0.2, 0.25, 0.4, 0.5]:
        experiments.append(Experiment(f"tp1_frac_{v}", {"exits.tp1_frac": v}))

    # TP2 (default now 2.0R)
    for v in [1.5, 2.5, 3.0]:
        experiments.append(Experiment(f"tp2_r_{v}", {"exits.tp2_r": v}))

    # TP2 fraction (default now 0.4)
    for v in [0.25, 0.30, 0.50]:
        experiments.append(Experiment(f"tp2_frac_{v}", {"exits.tp2_frac": v}))

    # Quick exit (now ON by default — test disabling)
    experiments.append(Experiment("quick_exit_off", {"exits.quick_exit_enabled": False}))
    experiments.append(Experiment("qe_bars_3", {"exits.quick_exit_bars": 3}))
    experiments.append(Experiment("qe_bars_6", {"exits.quick_exit_bars": 6}))
    experiments.append(Experiment("qe_mfe_015", {"exits.quick_exit_max_mfe_r": 0.15}))
    experiments.append(Experiment("qe_mfe_03", {"exits.quick_exit_max_mfe_r": 0.3}))
    experiments.append(Experiment("qe_r_neg01", {"exits.quick_exit_max_r": -0.1}))
    experiments.append(Experiment("qe_r_neg03", {"exits.quick_exit_max_r": -0.3}))

    # Time stop (default action now "reduce" — test reverting)
    experiments.append(Experiment("time_stop_action_close", {"exits.time_stop_action": "close"}))
    experiments.append(Experiment("time_stop_10", {"exits.time_stop_bars": 10}))
    experiments.append(Experiment("time_stop_24", {"exits.time_stop_bars": 24}))
    experiments.append(Experiment("time_progress_015", {"exits.time_stop_min_progress_r": 0.15}))

    # BE (default now 0.525)
    for v in [0.3, 0.4, 0.6, 0.7]:
        experiments.append(Experiment(f"be_buffer_{v}", {"exits.be_buffer_r": v}))

    # Invalidation tuning (default now 1.2 ATR depth)
    experiments.append(Experiment("no_invalidation", {"exits.invalidation_exit": False}))
    for v in [0.5, 0.8, 1.5, 2.0]:
        experiments.append(Experiment(f"invalidation_depth_{v}", {"exits.invalidation_depth_atr": v}))
    experiments.append(Experiment("invalidation_minbars_1", {"exits.invalidation_min_bars": 1}))
    experiments.append(Experiment("invalidation_minbars_5", {"exits.invalidation_min_bars": 5}))

    return experiments


def _phase3_candidates() -> list[Experiment]:
    """Trail & Stop — centered on earlier activation (0.3R/4 bars).

    Trail activation lowered from 0.5R/6 bars to 0.3R/4 bars.  use_farther
    disabled (tighter stops).  Test reverting both.
    """
    experiments = []

    # Trail activation (default now 0.3R / 4 bars)
    for v in [0.2, 0.4, 0.5]:
        experiments.append(Experiment(f"trail_act_r_{v}", {"trail.trail_activation_r": v}))
    for v in [3, 5, 6]:
        experiments.append(Experiment(f"trail_act_bars_{v}", {"trail.trail_activation_bars": v}))

    # Trail buffer (tight default 0.1575)
    for v in [1.0, 2.0]:
        experiments.append(Experiment(f"trail_wide_{v}", {"trail.trail_buffer_wide": v}))
    for v in [0.1, 0.2, 0.3]:
        experiments.append(Experiment(f"trail_tight_{v}", {"trail.trail_buffer_tight": v}))

    # Trail ceiling
    for v in [0.8, 1.0, 1.5]:
        experiments.append(Experiment(f"trail_ceiling_{v}", {"trail.trail_r_ceiling": v}))

    # Structure trail (test reverting baked-off default)
    experiments.append(Experiment("struct_trail_on", {"trail.structure_trail_enabled": True}))

    # Stops — test reverting use_farther
    experiments.append(Experiment("use_farther_on", {"stops.use_farther": True}))
    for v in [0.8, 1.5]:
        experiments.append(Experiment(f"stop_atr_{v}", {"stops.atr_mult": v}))
    experiments.append(Experiment("min_stop_05", {"stops.min_stop_atr": 0.5}))

    return experiments


def _phase4_candidates() -> list[Experiment]:
    """Zone & Profile — better zone detection = higher quality breakout signals.

    Profile lookback baked to 36 (18h), HVN threshold to 1.2 for wider pipeline.
    """
    experiments = []

    # Balance zone quality
    for v in [2, 8]:
        experiments.append(Experiment(f"min_bars_zone_{v}", {"balance.min_bars_in_zone": v}))
    experiments.append(Experiment("min_touches_3", {"balance.min_touches": 3}))
    for v in [16, 36]:
        experiments.append(Experiment(f"zone_age_{v}", {"balance.max_zone_age_bars": v}))
    for v in [1.0, 1.5]:
        experiments.append(Experiment(f"zone_width_{v}", {"balance.zone_width_atr": v}))

    # Profile construction (lookback default now 36, HVN threshold now 1.2)
    for v in [16, 24, 48]:
        experiments.append(Experiment(f"lookback_{v}", {"profile.lookback_bars": v}))
    for v in [1.0, 1.5, 1.8]:
        experiments.append(Experiment(f"hvn_thresh_{v}", {"profile.hvn_threshold_pct": v}))
    experiments.append(Experiment("lvn_thresh_03", {"profile.lvn_threshold_pct": 0.3}))

    # Setup quality
    for v in [0.5, 0.8]:
        experiments.append(Experiment(f"lvn_runway_{v}", {"setup.min_lvn_runway_atr": v}))
    experiments.append(Experiment("dedup_02", {"balance.dedup_atr_frac": 0.2}))

    return experiments


def _phase5_candidates() -> list[Experiment]:
    """Risk & Sizing — centered on perp-appropriate risk levels (1.5% B-grade).

    Risk defaults already calibrated from risk sweep (0.0225/0.01875/0.015).
    Test ranges around these optimal values.
    """
    experiments = []

    # Risk per trade (centered around 0.015 B-grade optimal)
    for v in [0.01, 0.012, 0.018, 0.02]:
        experiments.append(Experiment(f"risk_b_{v}", {"risk.risk_pct_b": v}))
    # A-grade (centered around 0.01875)
    for v in [0.015, 0.02, 0.025]:
        experiments.append(Experiment(f"risk_a_{v}", {"risk.risk_pct_a": v}))
    # A+ grade (centered around 0.0225)
    for v in [0.018, 0.02, 0.03]:
        experiments.append(Experiment(f"risk_a_plus_{v}", {"risk.risk_pct_a_plus": v}))

    # Limits
    for v in [5, 8]:
        experiments.append(Experiment(f"consec_loss_{v}", {"limits.max_consecutive_losses": v}))
    experiments.append(Experiment("daily_loss_004", {"limits.max_daily_loss_pct": 0.04}))
    experiments.append(Experiment("trades_per_day_8", {"limits.max_trades_per_day": 8}))
    experiments.append(Experiment("concurrent_5", {"limits.max_concurrent_positions": 5}))

    # Direction (if not already filtered in phase 1)
    experiments.append(Experiment("btc_long_only", {"symbol_filter.btc_direction": "long_only"}))
    experiments.append(Experiment("sol_long_only", {"symbol_filter.sol_direction": "long_only"}))

    # Re-entry
    experiments.append(Experiment("no_reentry", {"reentry.enabled": False}))
    experiments.append(Experiment("reentry_cooldown_6", {"reentry.cooldown_bars": 6}))
    experiments.append(Experiment("reentry_max_2", {"reentry.max_reentries": 2}))

    return experiments


def _phase6_candidates(cumulative: dict[str, Any]) -> list[Experiment]:
    """Finetune — re-sweep accepted params with tighter ranges."""
    experiments = []
    for key, val in cumulative.items():
        if isinstance(val, (int, float)):
            for mult in [0.85, 0.95, 1.05, 1.15]:
                new_val = round(val * mult, 6)
                if new_val != val:
                    experiments.append(Experiment(
                        f"finetune_{key.split('.')[-1]}_{mult}",
                        {key: new_val},
                    ))
    return experiments


_PHASE_CANDIDATES = {
    1: _phase1_candidates,
    2: _phase2_candidates,
    3: _phase3_candidates,
    4: _phase4_candidates,
    5: _phase5_candidates,
}


class BreakoutPlugin:
    """6-phase optimization plugin for Volume Profile Breakout strategy.

    Signal-first phase ordering: removes value-destroying segments and
    tightens signal quality before optimizing exits, trail, and risk.
    Immutable balanced scoring prevents quality-degrading mutations.
    """

    strategy_type = "breakout"

    def __init__(
        self,
        backtest_config: BacktestConfig,
        base_config: BreakoutConfig,
        data_dir: Path = Path("data"),
        max_workers: int | None = None,
    ) -> None:
        self.backtest_config = backtest_config
        self.base_config = base_config
        self.data_dir = data_dir
        self.max_workers = max_workers
        self._last_result: Any = None
        self._cached_store: Any = None

    @property
    def name(self) -> str:
        return "volume_profile_breakout"

    @property
    def num_phases(self) -> int:
        return 6

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "total_trades": 30.0,
            "win_rate": 45.0,
            "profit_factor": 1.5,
            "max_drawdown_pct": 30.0,
            "sharpe_ratio": 1.5,
        }

    @property
    def initial_mutations(self) -> dict[str, Any]:
        return {}

    def get_phase_spec(self, phase: int, state: Any) -> PhaseSpec:
        """Build PhaseSpec for *phase* with breakout-specific experiments."""
        if phase == 6:
            cumulative = state.cumulative_mutations if state else {}
            candidates = _phase6_candidates(cumulative)
        else:
            gen = _PHASE_CANDIDATES.get(phase)
            candidates = gen() if gen else []

        # Phase-specific scoring emphasis
        scoring = dict(PHASE_SCORING_EMPHASIS.get(phase, SCORING_WEIGHTS))
        gate_criteria = self._build_gate_criteria(phase)

        return PhaseSpec(
            phase_num=phase,
            name=PHASE_NAMES.get(phase, f"Phase {phase}"),
            candidates=candidates,
            scoring_weights=scoring,
            hard_rejects=dict(HARD_REJECTS),
            min_delta=0.005,
            max_rounds=3,
            gate_criteria=gate_criteria,
            gate_criteria_fn=lambda m, _p=phase: self._gate_criteria_fn(m, _p),
            analysis_policy=PhaseAnalysisPolicy(
                diagnostic_gap_fn=lambda p, m: self._diagnostic_gap_fn(p, m),
                suggest_experiments_fn=lambda p, m, w, s: self._suggest_experiments_fn(p, m, w, s),
                decide_action_fn=lambda *args: self._decide_action_fn(*args),
                redesign_scoring_weights_fn=lambda *args: self._redesign_scoring_weights_fn(*args),
                build_extra_analysis_fn=lambda p, m, s, g: self._build_extra_analysis_fn(p, m, s, g),
                format_extra_analysis_fn=lambda d: self._format_extra_analysis_fn(d),
            ),
            focus=PHASE_NAMES.get(phase, ""),
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
                strategy_type="breakout",
                ceilings=ceilings,
            )
        return evaluate_fn

    def _get_store(self):
        """Lazily build and cache an in-memory store to avoid repeated disk I/O."""
        if self._cached_store is None:
            from crypto_trader.optimize.parallel import _CachedStore
            from crypto_trader.data.store import ParquetStore

            symbols = self.backtest_config.symbols or self.base_config.symbols
            raw = ParquetStore(base_dir=self.data_dir)
            self._cached_store = _CachedStore(raw, symbols, ["30m", "4h"])
        return self._cached_store

    def _expected_symbols(self, config: BreakoutConfig | None = None) -> list[str]:
        symbols = self.backtest_config.symbols or (config.symbols if config is not None else self.base_config.symbols)
        return list(symbols)

    @staticmethod
    def _normalize_timestamp(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc) if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str) and value:
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
            return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
        return None

    def _build_blocked_relaxed_body_audit(
        self,
        config: BreakoutConfig,
        diagnostic_context: dict[str, Any],
    ) -> dict[str, Any] | None:
        raw_signals = diagnostic_context.get("blocked_relaxed_body_signals", [])
        if not isinstance(raw_signals, list) or not raw_signals:
            return None

        signals = [row for row in raw_signals if isinstance(row, dict)]
        if not signals:
            return None

        match_window_bars = max(
            2,
            int(config.confirmation.retest_max_bars) + int(config.entry.max_bars_after_signal) + 1,
        )
        match_window = timedelta(minutes=30 * match_window_bars)

        counterfactual_config = apply_mutations(
            config,
            {
                "symbol_filter.btc_relaxed_body_direction": "both",
                "symbol_filter.eth_relaxed_body_direction": "both",
                "symbol_filter.sol_relaxed_body_direction": "both",
            },
        )
        counterfactual_result = run(
            counterfactual_config,
            self.backtest_config,
            self.data_dir,
            strategy_type="breakout",
            store=self._get_store(),
        )

        relaxed_trades = [
            trade for trade in _result_trades(counterfactual_result)
            if getattr(trade, "signal_variant", "") == "relaxed_body"
        ]
        relaxed_trades.sort(
            key=lambda trade: self._normalize_timestamp(getattr(trade, "entry_time", None))
            or datetime.max.replace(tzinfo=timezone.utc)
        )

        used_trade_indexes: set[int] = set()
        enriched: list[dict[str, Any]] = []
        for signal in sorted(
            signals,
            key=lambda row: self._normalize_timestamp(row.get("signal_time")) or datetime.max.replace(tzinfo=timezone.utc),
        ):
            signal_row = dict(signal)
            signal_time = self._normalize_timestamp(signal_row.get("signal_time"))
            symbol = str(signal_row.get("symbol", "")).upper()
            direction = str(signal_row.get("direction", "")).upper()

            best_match: tuple[timedelta, int, Any] | None = None
            for idx, trade in enumerate(relaxed_trades):
                if idx in used_trade_indexes:
                    continue
                if str(getattr(trade, "symbol", "")).upper() != symbol:
                    continue
                trade_direction = getattr(getattr(trade, "direction", None), "value", "")
                if str(trade_direction).upper() != direction:
                    continue
                trade_entry_time = self._normalize_timestamp(getattr(trade, "entry_time", None))
                if signal_time is None or trade_entry_time is None or trade_entry_time < signal_time:
                    continue
                delta = trade_entry_time - signal_time
                if delta > match_window:
                    continue
                if best_match is None or delta < best_match[0]:
                    best_match = (delta, idx, trade)

            if best_match is None:
                signal_row["counterfactual_status"] = "no_counterfactual_trade"
                enriched.append(signal_row)
                continue

            _, trade_idx, trade = best_match
            used_trade_indexes.add(trade_idx)
            signal_row.update(
                {
                    "counterfactual_status": "matched_trade",
                    "counterfactual_entry_time": getattr(trade, "entry_time", None),
                    "counterfactual_confirmation_type": getattr(trade, "confirmation_type", None),
                    "counterfactual_entry_method": getattr(trade, "entry_method", None),
                    "counterfactual_signal_variant": getattr(trade, "signal_variant", None),
                    "counterfactual_r_multiple": getattr(trade, "economic_r_multiple", None),
                    "counterfactual_net_pnl": getattr(trade, "net_pnl", None),
                    "counterfactual_mfe_r": getattr(trade, "mfe_r", None),
                    "counterfactual_mae_r": getattr(trade, "mae_r", None),
                    "counterfactual_exit_reason": getattr(trade, "exit_reason", None),
                }
            )
            enriched.append(signal_row)

        return {
            "source": "relaxed_body_directions_forced_to_both",
            "match_window_bars": match_window_bars,
            "signals": enriched,
        }

    def compute_final_metrics(
        self,
        mutations: dict[str, Any],
    ) -> dict[str, float]:
        """Run full-period backtest and return metrics for gate evaluation."""
        config = apply_mutations(self.base_config, mutations)
        result = run(
            config, self.backtest_config, self.data_dir,
            strategy_type="breakout",
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
        diagnostic_context = _result_diagnostic_context(result)
        pm = getattr(result, "metrics", None) if result else None
        if not result or (not trades and not terminal_marks):
            return "No trades to diagnose."

        lines: list[str] = []

        phase_name = PHASE_NAMES.get(phase, "Unknown")
        lines.append(f"=== Breakout Diagnostics (Phase {phase}: {phase_name}) ===")
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
                    lines.append("--- Per-Confirmation Type ---")
                    for ctype, stats in insights.per_confirmation.items():
                        n = stats.get("n", 0)
                        wr = stats.get("wr", 0)
                        avg_r = stats.get("avg_r", 0)
                        if n > 0:
                            lines.append(f"  {ctype}: n={n}, WR={wr:.0f}%, avg_r={avg_r:+.3f}")
                    lines.append("")

                # Per-asset breakdown
                if insights.per_asset:
                    lines.append("--- Per-Asset ---")
                    for asset, stats in insights.per_asset.items():
                        n = stats.get("n", 0)
                        wr = stats.get("wr", 0)
                        avg_r = stats.get("avg_r", 0)
                        if n > 0:
                            lines.append(f"  {asset}: n={n}, WR={wr:.0f}%, avg_r={avg_r:+.3f}")
                    lines.append("")

                # Exit attribution
                if insights.exit_attribution:
                    lines.append("--- Exit Attribution ---")
                    for reason, stats in insights.exit_attribution.items():
                        n = stats.get("n", 0)
                        avg_r = stats.get("avg_r", 0)
                        if n > 0:
                            lines.append(f"  {reason}: n={n}, avg_r={avg_r:+.3f}")
                    lines.append("")

                # MFE capture
                if insights.mfe_capture:
                    cap = insights.mfe_capture
                    lines.append("--- MFE Capture ---")
                    lines.append(f"  Avg MFE: {cap.get('avg_mfe_r', 0):.2f}R")
                    lines.append(f"  Avg capture: {cap.get('avg_capture_pct', 0):.0%}")
                    lines.append(f"  Avg giveback: {cap.get('avg_giveback_pct', 0):.0%}")
                    lines.append("")

                # Direction analysis
                if insights.direction:
                    lines.append("--- Direction ---")
                    for d, stats in insights.direction.items():
                        n = stats.get("n", 0)
                        wr = stats.get("wr", 0)
                        avg_r = stats.get("avg_r", 0)
                        if n > 0:
                            lines.append(f"  {d}: n={n}, WR={wr:.0f}%, avg_r={avg_r:+.3f}")
                    lines.append("")

            # Phase-specific modular diagnostics
            modules = PHASE_DIAGNOSTIC_MODULES.get(phase, ["D6"])
            phase_diag = generate_phase_diagnostics(
                trades,
                modules,
                initial_equity=_diagnostic_initial_equity(self.backtest_config),
                title=f"Breakout Phase {phase} Diagnostics",
                terminal_marks=terminal_marks,
                performance_metrics=pm,
                expected_symbols=self._expected_symbols(),
                diagnostic_context=diagnostic_context,
            )
            lines.append(phase_diag)

        except Exception as e:
            lines.append(f"Diagnostics error: {e}")
            lines.append(
                generate_diagnostics(
                    trades,
                    initial_equity=_diagnostic_initial_equity(self.backtest_config),
                    terminal_marks=terminal_marks,
                    performance_metrics=pm,
                    expected_symbols=self._expected_symbols(),
                    diagnostic_context=diagnostic_context,
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
        """Run comprehensive diagnostics across all modules."""
        if self._last_result is None:
            mutations = state.cumulative_mutations if state else {}
            self.compute_final_metrics(mutations)

        result = self._last_result
        trades = _result_trades(result)
        terminal_marks = _result_terminal_marks(result)
        diagnostic_context = _result_diagnostic_context(result)
        pm = getattr(result, "metrics", None) if result else None
        if not result or (not trades and not terminal_marks):
            return "No trades for enhanced diagnostics."

        try:
            from crypto_trader.backtest.diagnostics import generate_phase_diagnostics
            return generate_phase_diagnostics(
                trades,
                ["D1", "D2", "D3", "D4", "D5", "D6"],
                initial_equity=_diagnostic_initial_equity(self.backtest_config),
                title="Breakout Enhanced Diagnostics",
                terminal_marks=terminal_marks,
                performance_metrics=pm,
                expected_symbols=self._expected_symbols(),
                diagnostic_context=diagnostic_context,
            )
        except Exception:
            return generate_diagnostics(
                trades,
                initial_equity=_diagnostic_initial_equity(self.backtest_config),
                terminal_marks=terminal_marks,
                performance_metrics=pm,
                expected_symbols=self._expected_symbols(),
                diagnostic_context=diagnostic_context,
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

        config = apply_mutations(
            self.base_config,
            state.cumulative_mutations if state is not None else {},
        )
        trades = _result_trades(self._last_result)
        terminal_marks = _result_terminal_marks(self._last_result)
        diagnostic_context = _result_diagnostic_context(self._last_result)
        pm = getattr(self._last_result, "metrics", None) if self._last_result else None

        try:
            blocked_audit = self._build_blocked_relaxed_body_audit(config, diagnostic_context)
        except Exception as exc:
            log.warning("breakout.blocked_relaxed_body_audit_failed", error=str(exc))
            blocked_audit = None
        if blocked_audit is not None:
            diagnostic_context["blocked_relaxed_body_audit"] = blocked_audit

        diagnostics_text = generate_diagnostics(
            trades,
            initial_equity=_diagnostic_initial_equity(self.backtest_config),
            terminal_marks=terminal_marks,
            performance_metrics=pm,
            expected_symbols=self._expected_symbols(config),
            diagnostic_context=diagnostic_context,
        ) if (trades or terminal_marks) else "(no trades)"

        # Build dimension reports
        dimension_reports: dict[str, str] = {}
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
        """Identify diagnostic gaps relevant to the current phase."""
        gaps = []

        if self._last_result and self._last_result.trades:
            try:
                from crypto_trader.backtest.diagnostics import extract_diagnostic_insights
                insights = extract_diagnostic_insights(self._last_result.trades)

                # Phase 1: Signal & Direction gaps
                if phase == 1:
                    if insights.direction:
                        for d, stats in insights.direction.items():
                            if stats.get("n", 0) >= 3 and stats.get("avg_r", 0) < -0.2:
                                gaps.append(f"[D4] {d} direction has negative avg_r ({stats['avg_r']:.2f}) — filter needed")
                    if insights.per_asset:
                        for asset, stats in insights.per_asset.items():
                            if stats.get("n", 0) >= 3 and stats.get("wr", 0) < 15:
                                gaps.append(f"[D5] {asset} has {stats['wr']:.0f}% WR — consider disabling")

                # Phase 2: Exit & Capture gaps
                elif phase == 2:
                    if insights.exit_attribution:
                        tp_hits = sum(
                            s.get("n", 0) for r, s in insights.exit_attribution.items()
                            if "tp" in r.lower()
                        )
                        if tp_hits == 0:
                            gaps.append("[D2] No TP hits — targets likely too high for MFE distribution")
                    if insights.mfe_capture:
                        capture = insights.mfe_capture.get("avg_capture_pct", 0)
                        if capture < 0:
                            gaps.append(f"[D2] Negative capture ({capture:.0%}) — exits systematically worse than entry")
                        elif capture < 0.30:
                            gaps.append(f"[D2] Low MFE capture ({capture:.0%}) — TP targets likely too high")

                # Phase 3: Trail & Stop gaps
                elif phase == 3:
                    if insights.exit_attribution:
                        stop_outs = sum(
                            s.get("n", 0) for r, s in insights.exit_attribution.items()
                            if "stop" in r.lower()
                        )
                        total = len(self._last_result.trades)
                        if total > 0 and stop_outs / total > 0.8:
                            gaps.append(f"[D1] Trail not activating — {stop_outs}/{total} exits via stop")
                    if insights.mfe_capture:
                        giveback = insights.mfe_capture.get("avg_giveback_pct", 0)
                        if giveback > 0.50:
                            gaps.append(f"[D1] High giveback ({giveback:.0%}) — trail not capturing momentum")

                # Phase 4: Zone & Profile gaps
                elif phase == 4:
                    if insights.per_asset:
                        for asset, stats in insights.per_asset.items():
                            if stats.get("n", 0) >= 3 and stats.get("avg_r", 0) < -0.5:
                                gaps.append(f"[D5] {asset} has negative avg_r ({stats['avg_r']:.2f}) — zone quality issue")
                    if metrics.get("total_trades", 0) < 8:
                        gaps.append(f"[D4] Low trade count ({metrics.get('total_trades', 0)}) — zone detection too restrictive")

                # Phase 5: Risk gaps
                elif phase == 5:
                    dd = metrics.get("max_drawdown_pct", 0)
                    if dd > 30:
                        gaps.append(f"[D3] High drawdown ({dd:.1f}%) — reduce risk sizing")
                    if insights.concentration:
                        top_pct = insights.concentration.get("top3_pnl_pct", 0)
                        if top_pct > 80:
                            gaps.append(f"[D3] High profit concentration ({top_pct:.0f}% in top 3)")

            except Exception:
                pass

        return gaps

    def _suggest_experiments_fn(
        self, phase: int, metrics: dict[str, float],
        weaknesses: list[str], state: Any,
    ) -> list[Experiment]:
        """Suggest additional experiments based on diagnostics."""
        suggestions = []

        if phase == 1:
            # If direction analysis shows value-destroying segments
            if self._last_result and self._last_result.trades:
                try:
                    from crypto_trader.backtest.diagnostics import extract_diagnostic_insights
                    insights = extract_diagnostic_insights(self._last_result.trades)
                    if insights.direction:
                        short_stats = insights.direction.get("short", {})
                        if short_stats.get("n", 0) >= 3 and short_stats.get("avg_r", 0) < -0.3:
                            suggestions.append(Experiment("combined_long_only",
                                {"symbol_filter.btc_direction": "long_only",
                                 "symbol_filter.eth_direction": "long_only",
                                 "symbol_filter.sol_direction": "long_only"}))
                except Exception:
                    pass

        elif phase == 2:
            # If no TP hits, suggest extremely low TP1
            if self._last_result and self._last_result.trades:
                try:
                    from crypto_trader.backtest.diagnostics import extract_diagnostic_insights
                    insights = extract_diagnostic_insights(self._last_result.trades)
                    if insights.exit_attribution:
                        tp_hits = sum(
                            s.get("n", 0) for r, s in insights.exit_attribution.items()
                            if "tp" in r.lower()
                        )
                        if tp_hits == 0:
                            suggestions.append(Experiment("extreme_low_tp",
                                {"exits.tp1_r": 0.2, "exits.tp1_frac": 0.5}))
                    if insights.mfe_capture:
                        capture = insights.mfe_capture.get("avg_capture_pct", 0)
                        if capture < 0:
                            suggestions.append(Experiment("capture_rescue",
                                {"exits.tp1_r": 0.3, "exits.quick_exit_enabled": True}))
                except Exception:
                    pass

        elif phase == 3:
            # Trail adjustments based on giveback
            if self._last_result and self._last_result.trades:
                try:
                    from crypto_trader.backtest.diagnostics import extract_diagnostic_insights
                    insights = extract_diagnostic_insights(self._last_result.trades)
                    if insights.mfe_capture:
                        giveback = insights.mfe_capture.get("avg_giveback_pct", 0)
                        if giveback > 0.50:
                            suggestions.append(Experiment("ultra_low_trail",
                                {"trail.trail_activation_r": 0.1, "trail.trail_r_ceiling": 0.5}))
                except Exception:
                    pass

        elif phase == 4:
            # If low trade count, suggest combined relaxation
            if metrics.get("total_trades", 0) < 5:
                suggestions.append(Experiment("aggressive_room",
                    {"setup.min_room_r_b": 0.3, "balance.max_zone_age_bars": 96,
                     "balance.min_touches": 1}))

        elif phase == 5:
            dd = metrics.get("max_drawdown_pct", 0)
            if dd > 30:
                suggestions.append(Experiment("reduce_risk",
                    {"risk.risk_pct_a_plus": 0.008, "risk.risk_pct_a": 0.005, "risk.risk_pct_b": 0.003}))

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
        """Decide phase action — return None to use default fallback."""
        return None

    def _redesign_scoring_weights_fn(
        self,
        phase: int,
        current_weights: dict[str, float],
        metrics: dict[str, float],
        strengths: list[str],
        weaknesses: list[str],
    ) -> dict[str, float] | None:
        """Optionally redesign scoring weights on retry.

        Uses phase-specific emphasis as base — amplify top-2 dimensions
        by 1.15x to increase sensitivity on retry.
        """
        base = dict(PHASE_SCORING_EMPHASIS.get(phase, SCORING_WEIGHTS))
        sorted_dims = sorted(base.items(), key=lambda x: x[1], reverse=True)
        for dim, _ in sorted_dims[:2]:
            base[dim] *= 1.15
        # Re-normalize
        total = sum(base.values())
        if total > 0:
            base = {k: v / total for k, v in base.items()}
        return base

    def _build_extra_analysis_fn(
        self, phase: int, metrics: dict[str, float],
        state: Any, greedy_result: GreedyResult,
    ) -> dict[str, Any]:
        """Build extra analysis data for phase report."""
        extra: dict[str, Any] = {}
        if self._last_result and self._last_result.trades:
            trades = self._last_result.trades
            extra["trade_count"] = len(trades)
            winners = [t for t in trades if t.net_pnl > 0]
            extra["winner_count"] = len(winners)
            if winners:
                winner_rs = [t.economic_r_multiple for t in winners if t.economic_r_multiple is not None]
                if winner_rs:
                    extra["avg_winner_r"] = sum(winner_rs) / len(winner_rs)
            losers = [t for t in trades if t.net_pnl <= 0]
            if losers:
                loser_rs = [t.economic_r_multiple for t in losers if t.economic_r_multiple is not None]
                if loser_rs:
                    extra["avg_loser_r"] = sum(loser_rs) / len(loser_rs)

            # Capture-specific analysis for exit and trail phases
            if phase in (2, 3):
                try:
                    from crypto_trader.backtest.diagnostics import extract_diagnostic_insights
                    insights = extract_diagnostic_insights(trades)
                    if insights.mfe_capture:
                        extra["avg_capture_pct"] = insights.mfe_capture.get("avg_capture_pct", 0)
                        extra["avg_giveback_pct"] = insights.mfe_capture.get("avg_giveback_pct", 0)
                    if insights.exit_attribution:
                        tp_exits = sum(
                            s.get("n", 0) for r, s in insights.exit_attribution.items()
                            if "tp" in r.lower()
                        )
                        extra["tp_exits"] = tp_exits
                except Exception:
                    pass

        return extra

    def _format_extra_analysis_fn(self, data: dict[str, Any]) -> str:
        """Format extra analysis dict to text."""
        lines = []
        for key, val in data.items():
            if isinstance(val, float):
                lines.append(f"  {key}: {val:.3f}")
            else:
                lines.append(f"  {key}: {val}")
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
            lines.append("ACTION: Trade count too low — prioritize signal generation")
        elif pm.profit_factor < 1.0:
            lines.append("ACTION: Negative expectancy — review entry quality and exit management")
        elif pm.max_drawdown_pct > 30:
            lines.append("ACTION: High drawdown — consider reducing risk sizing or adding filters")
        elif pm.sharpe_ratio < 1.0:
            lines.append("ACTION: Low risk-adjusted return — optimize exit timing")
        else:
            lines.append("STATUS: Baseline acceptable — continue optimization")

        return "\n".join(lines)
