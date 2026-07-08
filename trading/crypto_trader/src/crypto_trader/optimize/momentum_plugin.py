"""MomentumPlugin — 6-phase optimization plugin for the momentum pullback strategy."""

from __future__ import annotations

from numbers import Real
from pathlib import Path
from typing import Any

import structlog

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.backtest.metrics import (
    PerformanceMetrics,
    metrics_to_dict,
)
from crypto_trader.backtest.runner import run
from crypto_trader.data.store import ParquetStore
from crypto_trader.optimize.config_mutator import apply_mutations, merge_mutations
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
from crypto_trader.strategy.momentum.config import MomentumConfig

log = structlog.get_logger("optimize.momentum")


def _diagnostic_initial_equity(config: Any, default: float = 10_000.0) -> float:
    value = getattr(config, "initial_equity", default)
    return float(value) if isinstance(value, Real) else default


def _result_trades(result: Any) -> list[Any]:
    trades = getattr(result, "trades", []) if result is not None else []
    return trades if isinstance(trades, list) else []


def _result_terminal_marks(result: Any) -> list[Any]:
    terminal_marks = getattr(result, "terminal_marks", []) if result is not None else []
    return terminal_marks if isinstance(terminal_marks, list) else []

# ── Immutable scoring — same weights for ALL phases ──────────────────────

SCORING_WEIGHTS: dict[str, float] = {
    "returns": 0.30,
    "edge": 0.25,
    "coverage": 0.20,
    "calmar": 0.15,
    "capture": 0.10,
}

SCORING_CEILINGS: dict[str, float] = {
    "returns": 12.0,
    "edge": 2.5,
    "coverage": 36.0,
    "calmar": 2.5,
}

HARD_REJECTS: dict[str, tuple[str, float]] = {
    "max_drawdown_pct": ("<=", 50.0),   # Calmar scoring penalizes high DD
    "total_trades": (">=", 12),          # Preserve trade count (baseline=19); prevent artifact inflation
    "profit_factor": (">=", 0.8),        # Reject clearly unprofitable configs
}

# ── Phase-specific scoring emphasis (initial weights + retry amplification) ────

PHASE_SCORING_EMPHASIS: dict[int, dict[str, float]] = {
    phase: dict(SCORING_WEIGHTS) for phase in range(1, 7)
}

# ── Confirmation type → config param mapping ─────────────────────────────

CONFIRMATION_DISABLE_MAP: dict[str, str] = {
    "engulfing": "confirmation.enable_engulfing",
    "hammer": "confirmation.enable_hammer",
    "inside_bar": "confirmation.enable_inside_bar",
    "micro_shift": "confirmation.enable_micro_shift",
    "micro_structure_shift": "confirmation.enable_micro_shift",  # alias
    "base_break": "confirmation.enable_base_break",
}

# ── Phase diagnostic module mapping ───────────────────────────────────

PHASE_DIAGNOSTIC_MODULES: dict[int, list[str]] = {
    1: ["D1", "D6"],           # Trail & Stop → MFE capture, stop calibration, duration
    2: ["D1", "D2", "D6"],     # Exit → MFE capture + exit attribution, worst/best trades
    3: ["D4", "D6"],           # Signal → confirmation, confluence, entry method, per-asset
    4: ["D4", "D5", "D6"],     # Environment → signal quality + direction, timing, interactions
    5: ["D3", "D5", "D6"],     # Risk → drawdown, streaks, sizing + direction, timing
    6: ["D1", "D2", "D3", "D4", "D5", "D6"],  # Finetune → all modules
}

# ── Phase-specific gate criteria ──────────────────────────────────────

PHASE_GATE_CRITERIA: dict[int, list[GateCriterion]] = {
    1: [  # Trail — trail shouldn't kill many trades
        GateCriterion(metric="total_trades", operator=">=", threshold=10),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=40),
        GateCriterion(metric="profit_factor", operator=">=", threshold=0.8),
    ],
    3: [  # Signal — preserve trading frequency
        GateCriterion(metric="total_trades", operator=">=", threshold=12),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=50),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.0),
    ],
    4: [  # Coverage — must preserve trade count
        GateCriterion(metric="total_trades", operator=">=", threshold=10),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=50),
        GateCriterion(metric="profit_factor", operator=">=", threshold=0.7),
    ],
    5: [  # Risk — focus on DD control
        GateCriterion(metric="total_trades", operator=">=", threshold=10),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=40),
        GateCriterion(metric="profit_factor", operator=">=", threshold=0.7),
    ],
}

# ── Phase definitions ──────────────────────────────────────────────────

PHASE_NAMES = {
    1: "Trail & Stop Calibration",
    2: "Profit Taking & Exit Timing",
    3: "Signal & Entry Quality",
    4: "Environment & Filtering",
    5: "Risk & Position Sizing",
    6: "Fine-tuning",
}

PHASE_FOCUS_METRICS: dict[int, list[str]] = {
    1: ["exit_efficiency", "avg_bars_held", "avg_mfe_r"],
    2: ["exit_efficiency", "profit_factor", "win_rate"],
    3: ["total_trades", "win_rate", "profit_factor"],
    4: ["total_trades", "profit_factor", "net_return_pct"],
    5: ["sharpe_ratio", "calmar_ratio", "net_return_pct"],
    6: ["sharpe_ratio", "calmar_ratio", "net_return_pct"],
}


# ── Phase experiment generators ──────────────────────────────────────────


def _phase1_candidates() -> list[Experiment]:
    """Trail & Stop Calibration — 30 experiments targeting 0.731R winner giveback."""
    return [
        # R-adaptive buffer — single-parameter sweeps
        Experiment("TRAIL_CEILING_1_5", {"trail.trail_r_ceiling": 1.5}),
        Experiment("TRAIL_CEILING_1_2", {"trail.trail_r_ceiling": 1.2}),
        Experiment("TRAIL_CEILING_1_0", {"trail.trail_r_ceiling": 1.0}),
        Experiment("TRAIL_CEILING_0_8", {"trail.trail_r_ceiling": 0.8}),
        Experiment("TRAIL_WIDE_1_0", {"trail.trail_buffer_wide": 1.0}),
        Experiment("TRAIL_WIDE_2_0", {"trail.trail_buffer_wide": 2.0}),
        Experiment("TRAIL_TIGHT_0_1", {"trail.trail_buffer_tight": 0.1}),
        Experiment("TRAIL_TIGHT_0_15", {"trail.trail_buffer_tight": 0.15}),
        Experiment("TRAIL_TIGHT_0_2", {"trail.trail_buffer_tight": 0.2}),
        Experiment("TRAIL_TIGHT_0_5", {"trail.trail_buffer_tight": 0.5}),
        # Compound trail — reshape entire R-adaptive curve (parameters interact)
        # Tight capture: fast tightening, wider floor — locks profit early
        Experiment("TRAIL_COMBO_TIGHT_CAPTURE", {
            "trail.trail_r_ceiling": 1.0, "trail.trail_buffer_tight": 0.3,
            "trail.trail_buffer_wide": 1.0,
        }),
        # Runner: slow tightening, narrow floor — lets big winners run
        Experiment("TRAIL_COMBO_RUNNER", {
            "trail.trail_r_ceiling": 2.0, "trail.trail_buffer_tight": 0.2,
            "trail.trail_buffer_wide": 2.0,
        }),
        # Balanced: moderate tightening, moderate floor
        Experiment("TRAIL_COMBO_BALANCED", {
            "trail.trail_r_ceiling": 1.2, "trail.trail_buffer_tight": 0.25,
            "trail.trail_buffer_wide": 1.2,
        }),
        # Aggressive lock: very fast tightening — captures max from small winners
        Experiment("TRAIL_COMBO_AGGRESSIVE", {
            "trail.trail_r_ceiling": 0.8, "trail.trail_buffer_tight": 0.4,
            "trail.trail_buffer_wide": 1.0,
        }),
        # Trail activation and mechanics
        Experiment("TRAIL_ACTIVATION_5", {"trail.trail_activation_bars": 5}),
        Experiment("TRAIL_ACT_R_0_3", {"trail.trail_activation_r": 0.3}),
        Experiment("TRAIL_ACT_R_0_8", {"trail.trail_activation_r": 0.8}),
        Experiment("TRAIL_GENEROUS", {"trail.trail_use_tightest": False}),
        Experiment("TRAIL_EMA_20", {"trail.trail_ema_period": 20}),
        Experiment("TRAIL_EMA_15", {"trail.trail_ema_period": 15}),
        # MFE floor — recalibrated to actual trade data (biggest givebacks at MFE>1.5R)
        Experiment("TRAIL_MFE_FLOOR_HIGH", {
            "trail.trail_mfe_floor_enabled": True,
            "trail.trail_mfe_floor_threshold": 1.5, "trail.trail_mfe_floor_buffer": 0.6,
        }),
        Experiment("TRAIL_MFE_FLOOR_MID", {
            "trail.trail_mfe_floor_enabled": True,
            "trail.trail_mfe_floor_threshold": 1.0, "trail.trail_mfe_floor_buffer": 0.5,
        }),
        Experiment("TRAIL_MFE_FLOOR_LOW", {
            "trail.trail_mfe_floor_enabled": True,
            "trail.trail_mfe_floor_threshold": 0.5, "trail.trail_mfe_floor_buffer": 0.4,
        }),
        # Compound: MFE floor + tighter curve — lock in big winners AND tighten small ones
        Experiment("TRAIL_COMBO_MFE_LOCK", {
            "trail.trail_r_ceiling": 1.0, "trail.trail_buffer_tight": 0.3,
            "trail.trail_mfe_floor_enabled": True,
            "trail.trail_mfe_floor_threshold": 1.0, "trail.trail_mfe_floor_buffer": 0.5,
        }),
        # Stop sizing
        Experiment("STOP_ATR_1_5", {"stops.min_stop_atr_mult": 1.5}),
        Experiment("STOP_ATR_2_5", {"stops.min_stop_atr_mult": 2.5}),
        Experiment("STOP_BUF_0_4", {"stops.atr_buffer_mult": 0.4}),
    ]


def _phase2_candidates() -> list[Experiment]:
    """Profit Taking & Exit Timing — 19 experiments calibrating partial exits and early cuts."""
    return [
        # TP1 — lower target so more winners trigger partial exit
        Experiment("TP1_R_0_8", {"exits.tp1_r": 0.8}),
        Experiment("TP1_R_1_0", {"exits.tp1_r": 1.0}),
        # TP1 fraction — bank more when it triggers (currently 16%)
        Experiment("TP1_FRAC_0_25", {"exits.tp1_frac": 0.25}),
        Experiment("TP1_FRAC_0_35", {"exits.tp1_frac": 0.35}),
        Experiment("TP1_FRAC_0_50", {"exits.tp1_frac": 0.50}),
        # TP2 — lower target so achievable by more winners (currently 2.5R)
        Experiment("TP2_R_1_5", {"exits.tp2_r": 1.5}),
        Experiment("TP2_R_2_0", {"exits.tp2_r": 2.0}),
        # TP2 fraction
        Experiment("TP2_FRAC_0_30", {"exits.tp2_frac": 0.30}),
        Experiment("TP2_FRAC_0_40", {"exits.tp2_frac": 0.40}),
        Experiment("TP2_FRAC_0_50", {"exits.tp2_frac": 0.50}),
        # Breakeven — tune BE move after TP1
        Experiment("BE_BUFFER_0_3", {"exits.be_buffer_r": 0.3}),
        Experiment("BE_BUFFER_0", {"exits.be_buffer_r": 0.0}),
        Experiment("BE_ACCEPT_1", {"exits.be_acceptance_bars": 1}),
        # Quick exit — currently never fires (losers exit before bar 6 or MFE > 0.15)
        Experiment("QUICK_BARS_4", {"exits.quick_exit_bars": 4}),
        Experiment("QUICK_BARS_3", {"exits.quick_exit_bars": 3}),
        Experiment("QUICK_MFE_0_3", {"exits.quick_exit_max_mfe_r": 0.3}),
        Experiment("QUICK_R_0", {"exits.quick_exit_max_r": 0.0}),
        # Time stops — tighten so they actually fire (avg hold 8.4 bars)
        Experiment("TIME_SOFT_8", {"exits.soft_time_stop_bars": 8}),
        Experiment("TIME_SOFT_R_0_3", {"exits.soft_time_stop_min_r": 0.3}),
    ]


def _phase3_candidates() -> list[Experiment]:
    """Signal Quality & Entry — 16 experiments improving signal discrimination."""
    return [
        # micro_structure_shift — weakest signal (43% WR, 4/5 worst trades)
        Experiment("DISABLE_MICRO", {"confirmation.enable_micro_shift": False}),
        Experiment("MICRO_BARS_5", {"confirmation.micro_shift_min_bars": 5}),
        Experiment("MICRO_BARS_7", {"confirmation.micro_shift_min_bars": 7}),
        # Volume confirmation
        Experiment("VOLUME_STRICT", {"confirmation.volume_threshold_mult": 1.2}),
        Experiment("VOLUME_OFF", {"confirmation.require_volume_confirm": False}),
        # Quality gates — filter low-quality setups
        Experiment("WEAK_GATE_3", {"confirmation.min_confluences_for_weak": 3}),
        Experiment("WEAK_GATE_1", {"confirmation.min_confluences_for_weak": 1}),
        Experiment("CONFLUENCES_B_1", {"setup.min_confluences_b": 1}),
        Experiment("CONFLUENCES_B_2", {"setup.min_confluences_b": 2}),
        # Entry window and pullback zone
        Experiment("MAX_BARS_1", {"entry.max_bars_after_confirmation": 1}),
        Experiment("MAX_BARS_5", {"entry.max_bars_after_confirmation": 5}),
        Experiment("FIB_WIDER", {"setup.fib_high": 0.786}),
        Experiment("FIB_NARROW", {"setup.fib_high": 0.50}),
        Experiment("ROOM_B_1_0", {"setup.min_room_b": 1.0}),
        Experiment("RSI_PULLBACK_OFF", {"setup.use_rsi_pullback_filter": False}),
        # Entry mechanism — add break entries (currently only close entries)
        Experiment("ENTRY_BREAK", {"entry.entry_on_break": True}),
    ]


def _phase4_candidates() -> list[Experiment]:
    """Coverage & Environment — 14 experiments increasing trade frequency."""
    return [
        # Symbol direction — restrict direction but never disable instruments
        Experiment("ETH_BOTH", {"symbol_filter.eth_direction": "both"}),
        Experiment("ETH_LONG", {"symbol_filter.eth_direction": "long_only"}),
        Experiment("SOL_LONG", {"symbol_filter.sol_direction": "long_only"}),
        # Bias relaxation — unlock more setups
        Experiment("BIAS_H4_1", {"bias.min_4h_conditions": 1}),
        Experiment("BIAS_H1_1", {"bias.min_1h_conditions": 1}),
        # ADX/Chop thresholds
        Experiment("ADX_CHOP_5", {"filters.adx_chop_threshold": 5.0}),
        Experiment("ADX_CHOP_15", {"filters.adx_chop_threshold": 15.0}),
        Experiment("H1_ADX_10", {"bias.h1_adx_threshold": 10.0}),
        Experiment("H1_ADX_20", {"bias.h1_adx_threshold": 20.0}),
        # Re-entry tuning
        Experiment("REENTRY_COOL_2", {"reentry.cooldown_bars": 2}),
        Experiment("REENTRY_MAX_2", {"reentry.max_reentries": 2}),
        Experiment("REENTRY_OFF", {"reentry.enabled": False}),
        # Daily limits
        Experiment("MAX_DAILY_6", {"daily_limits.max_trades_per_day": 6}),
        Experiment("CONSEC_LOSS_3", {"daily_limits.max_consecutive_losses": 3}),
    ]


def _phase5_candidates() -> list[Experiment]:
    """Risk & Position Sizing — 12 experiments scaling optimized edge."""
    return [
        Experiment("RISK_A_0_025", {"risk.risk_pct_a": 0.025}),
        Experiment("RISK_A_0_030", {"risk.risk_pct_a": 0.030}),
        Experiment("RISK_B_0_012", {"risk.risk_pct_b": 0.012}),
        Experiment("RISK_B_0_014", {"risk.risk_pct_b": 0.014}),
        Experiment("RISK_B_0_015", {"risk.risk_pct_b": 0.015}),
        Experiment("RISK_B_0_020", {"risk.risk_pct_b": 0.020}),
        Experiment("RISK_B_0_025", {"risk.risk_pct_b": 0.025}),
        Experiment("LEV_MAJOR_12", {"risk.max_leverage_major": 12.0}),
        Experiment("LEV_MAJOR_15", {"risk.max_leverage_major": 15.0}),
        Experiment("LEV_ALT_10", {"risk.max_leverage_alt": 10.0}),
        Experiment("LEV_ALT_12", {"risk.max_leverage_alt": 12.0}),
        Experiment("MAX_POS_4", {"risk.max_concurrent_positions": 4}),
        Experiment("GROSS_RISK_0_05", {"risk.max_gross_risk": 0.05}),
    ]


def _legacy_phase6_candidates() -> list[Experiment]:
    """Replay the round-1 and round-2 fine-tune sweeps from the pre-round-1 seed."""
    legacy_values: tuple[tuple[str, float], ...] = (
        ("trail.trail_r_ceiling", 1.5),
        ("trail.trail_buffer_tight", 0.15),
        ("setup.min_room_b", 1.0),
        ("setup.fib_high", 0.5),
        ("exits.tp2_frac", 0.4),
        ("risk.risk_pct_b", 0.014),
    )
    experiments: list[Experiment] = []
    for key, value in legacy_values:
        for factor in (0.8, 0.9, 1.1, 1.2):
            experiments.append(
                Experiment(
                    f"FINETUNE_{key}_x{factor}",
                    {key: value * factor},
                )
            )
    return experiments


def _phase6_candidates(cumulative_mutations: dict[str, Any]) -> list[Experiment]:
    """Fine-tuning — historical replay plus dynamic experiments from accepted mutations."""
    experiments = _legacy_phase6_candidates()

    for key, value in cumulative_mutations.items():
        if isinstance(value, bool):
            continue  # Skip booleans

        if isinstance(value, (int, float)):
            factors = [0.8, 0.9, 1.1, 1.2]
            for factor in factors:
                new_val = value * factor
                if isinstance(value, int):
                    new_val = round(new_val)
                    if new_val == value:
                        continue
                name = f"FINETUNE_{key}_x{factor}"
                experiments.append(Experiment(name, {key: new_val}))

    return experiments


PHASE_CANDIDATES = {
    1: _phase1_candidates,
    2: _phase2_candidates,
    3: _phase3_candidates,
    4: _phase4_candidates,
    5: _phase5_candidates,
}


class MomentumPlugin:
    """StrategyPlugin implementation for momentum pullback strategy."""

    def __init__(
        self,
        backtest_config: BacktestConfig,
        base_config: MomentumConfig,
        data_dir: Path = Path("data"),
        max_workers: int | None = None,
    ) -> None:
        self.backtest_config = backtest_config
        self.base_config = base_config
        self.data_dir = data_dir
        self.max_workers = max_workers
        self._last_result: Any = None  # BacktestResult cached for diagnostics
        self._cached_store: Any = None

    @property
    def name(self) -> str:
        return "momentum_pullback"

    @property
    def num_phases(self) -> int:
        return 6

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "total_trades": 36.0,
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
            candidates = PHASE_CANDIDATES[phase]()

        gate_criteria = self._build_gate_criteria(phase)

        return PhaseSpec(
            phase_num=phase,
            name=PHASE_NAMES[phase],
            candidates=candidates,
            scoring_weights=dict(SCORING_WEIGHTS),
            hard_rejects=HARD_REJECTS,
            gate_criteria=gate_criteria,
            gate_criteria_fn=lambda m, _p=phase: self._gate_criteria_fn(m, _p),
            analysis_policy=PhaseAnalysisPolicy(
                max_scoring_retries=0,
                max_diagnostic_retries=1,
                focus_metrics=PHASE_FOCUS_METRICS.get(phase, []),
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
                ceilings=SCORING_CEILINGS,
            )

        return evaluate_fn

    def _get_store(self):
        """Lazily build and cache an in-memory store to avoid repeated disk I/O."""
        if self._cached_store is None:
            from crypto_trader.optimize.parallel import _CachedStore
            symbols = self.backtest_config.symbols or self.base_config.symbols
            raw = ParquetStore(base_dir=self.data_dir)
            self._cached_store = _CachedStore(raw, symbols, ["15m", "1h", "4h"])
        return self._cached_store

    def compute_final_metrics(
        self,
        mutations: dict[str, Any],
    ) -> dict[str, float]:
        """Run full-period backtest and return metrics for gate evaluation."""
        config = apply_mutations(self.base_config, mutations)
        result = run(config, self.backtest_config, self.data_dir,
                     store=self._get_store())
        self._last_result = result  # cache for diagnostics & callbacks
        return metrics_to_dict(result.metrics)

    def run_phase_diagnostics(
        self,
        phase: int,
        state: Any,
        metrics: dict[str, float],
        greedy_result: GreedyResult,
    ) -> str:
        """Run targeted diagnostics using trade-level insights."""
        lines = [f"Phase {phase} Diagnostics ({PHASE_NAMES.get(phase, '')})"]
        lines.append(f"  Accepted: {len(greedy_result.accepted_experiments)}")
        lines.append(f"  Rejected: {len(greedy_result.rejected_experiments)}")
        lines.append(f"  Score: {greedy_result.base_score:.4f} -> {greedy_result.final_score:.4f}")

        # Focus metrics for this phase
        focus = PHASE_FOCUS_METRICS.get(phase, [])
        if focus:
            lines.append(f"\n  Focus metrics:")
            for fm in focus:
                val = metrics.get(fm, 0.0)
                lines.append(f"    {fm}: {val:.2f}")

        trades = _result_trades(self._last_result)
        terminal_marks = _result_terminal_marks(self._last_result)

        # Trade-level insights if available
        if trades:
            from crypto_trader.backtest.diagnostics import (
                extract_diagnostic_insights,
                generate_phase_diagnostics,
            )
            insights = extract_diagnostic_insights(trades)

            # Per-confirmation breakdown (flag negative avg_r)
            if insights.per_confirmation:
                lines.append(f"\n  Confirmation breakdown:")
                for ctype, stats in sorted(
                    insights.per_confirmation.items(),
                    key=lambda x: -x[1]["n"],
                ):
                    flag = " *** NEGATIVE" if stats["avg_r"] < 0 else ""
                    lines.append(
                        f"    {ctype}: n={stats['n']:.0f}, "
                        f"WR={stats['wr']:.0f}%, "
                        f"avg_r={stats['avg_r']:+.3f}{flag}"
                    )

            # Per-asset summary (flag negative edge)
            if insights.per_asset:
                lines.append(f"\n  Per-asset edge:")
                for sym, stats in sorted(insights.per_asset.items()):
                    flag = " *** NEGATIVE EDGE" if stats["avg_r"] < 0 else ""
                    lines.append(
                        f"    {sym}: n={stats['n']:.0f}, "
                        f"WR={stats['wr']:.0f}%, "
                        f"avg_r={stats['avg_r']:+.3f}{flag}"
                    )

            # Exit attribution top 3 by P&L share
            if insights.exit_attribution:
                lines.append(f"\n  Exit attribution (top 3 by P&L share):")
                sorted_exits = sorted(
                    insights.exit_attribution.items(),
                    key=lambda x: abs(x[1].get("pnl_share", 0)),
                    reverse=True,
                )[:3]
                for reason, stats in sorted_exits:
                    lines.append(
                        f"    {reason}: n={stats['n']:.0f}, "
                        f"avg_r={stats['avg_r']:+.3f}, "
                        f"P&L share={stats['pnl_share']:.0%}"
                    )

            # MFE capture
            lines.append(f"\n  MFE capture:")
            lines.append(f"    avg_capture={insights.mfe_capture.get('avg_capture_pct', 0):.1%}")
            lines.append(f"    avg_giveback={insights.mfe_capture.get('avg_giveback_pct', 0):.1%}")

            # Phase-targeted diagnostic sections
            modules = PHASE_DIAGNOSTIC_MODULES.get(phase, ["D6"])
            lines.append("")
            lines.append(generate_phase_diagnostics(
                trades, modules,
                initial_equity=_diagnostic_initial_equity(self.backtest_config),
                title=f"Phase {phase} Targeted Sections",
                terminal_marks=terminal_marks,
            ))
        elif terminal_marks:
            from crypto_trader.backtest.diagnostics import generate_phase_diagnostics

            lines.append(f"\n  Terminal marks: {len(terminal_marks)}")
            modules = PHASE_DIAGNOSTIC_MODULES.get(phase, ["D6"])
            lines.append("")
            lines.append(generate_phase_diagnostics(
                [],
                modules,
                initial_equity=_diagnostic_initial_equity(self.backtest_config),
                title=f"Phase {phase} Targeted Sections",
                terminal_marks=terminal_marks,
            ))
        else:
            # Fallback to key metrics
            key_metrics = ["total_trades", "win_rate", "profit_factor",
                           "max_drawdown_pct", "sharpe_ratio", "calmar_ratio",
                           "exit_efficiency", "avg_bars_held"]
            lines.append(f"\n  Key metrics:")
            for m in key_metrics:
                val = metrics.get(m, 0.0)
                lines.append(f"    {m}: {val:.2f}")

        return "\n".join(lines)

    def run_enhanced_diagnostics(
        self,
        phase: int,
        state: Any,
        metrics: dict[str, float],
        greedy_result: GreedyResult,
    ) -> str:
        """Run enhanced diagnostics with full 22-section deep analysis."""
        lines = [f"Phase {phase} Enhanced Diagnostics ({PHASE_NAMES.get(phase, '')})"]
        lines.append(f"  Score: {greedy_result.base_score:.4f} -> {greedy_result.final_score:.4f}")

        if greedy_result.accepted_experiments:
            lines.append("  Accepted: " + ", ".join(
                sc.experiment.name for sc in greedy_result.accepted_experiments))

        trades = _result_trades(self._last_result)
        terminal_marks = _result_terminal_marks(self._last_result)
        if self._last_result and (trades or terminal_marks):
            from crypto_trader.backtest.diagnostics import generate_phase_diagnostics
            all_modules = ["D1", "D2", "D3", "D4", "D5", "D6"]
            lines.append("")
            lines.append(generate_phase_diagnostics(
                trades, all_modules,
                initial_equity=_diagnostic_initial_equity(self.backtest_config),
                title="Enhanced Full Diagnostics",
                terminal_marks=terminal_marks,
            ))
        else:
            # Fallback to metrics-only
            lines.append("\n  All metrics:")
            for m, v in sorted(metrics.items()):
                lines.append(f"    {m}: {v:.4f}")

        return "\n".join(lines)

    # ── Policy callbacks ──────────────────────────────────────────────

    def _diagnostic_gap_fn(self, phase: int, metrics: dict[str, float]) -> list[str]:
        """Identify what deeper analysis would illuminate using trade-level insights."""
        gaps: list[str] = []

        # Use trade-level insights when available
        if self._last_result and self._last_result.trades:
            from crypto_trader.backtest.diagnostics import extract_diagnostic_insights
            insights = extract_diagnostic_insights(self._last_result.trades)

            # MFE capture — relevant for phases 1-2 (trail/exit)
            if phase in (1, 2, 6):
                if insights.mfe_capture.get("avg_giveback_pct", 0) > 0.50:
                    gaps.append(
                        "High alpha giveback (D1) — trail/stop calibration needs review"
                    )

            # Negative-R confirmations — relevant for phases 3-4 (signal)
            if phase in (3, 4, 6):
                bad_confs = [c for c, s in insights.per_confirmation.items()
                             if s["n"] >= 2 and s["avg_r"] < 0]
                if bad_confs:
                    gaps.append(
                        f"Value-destroying signals (D4) — {', '.join(bad_confs)} "
                        f"have negative avg R"
                    )

            # Concentration risk — relevant for phases 3-5
            if phase in (3, 4, 5, 6):
                if insights.concentration.get("top1_pct", 0) > 0.50:
                    gaps.append(
                        "Single-trade concentration risk (D3) — risk sizing review needed"
                    )

            # Direction imbalance — relevant for phases 4-5
            if phase in (4, 5, 6):
                long_data = insights.direction.get("long", {})
                short_data = insights.direction.get("short", {})
                if (long_data.get("avg_r", 0) < 0 and long_data.get("n", 0) >= 2) or \
                   (short_data.get("avg_r", 0) < 0 and short_data.get("n", 0) >= 2):
                    gaps.append(
                        "Direction imbalance (D5) — one side has negative edge"
                    )

            # Duration outliers — relevant for phases 1-2
            if phase in (1, 2, 6):
                avg_bars = insights.duration.get("avg_bars", 0)
                if avg_bars > 20:
                    gaps.append(
                        "Duration outliers (D1) — avg hold > 20 bars, "
                        "quick exit or trail activation review"
                    )

            # Stop-out dominance — relevant for phases 1-2
            if phase in (1, 2, 6):
                stop_attr = insights.exit_attribution.get("protective_stop", {})
                if stop_attr.get("n", 0) > 0:
                    stop_share = stop_attr.get("pnl_share", 0)
                    if stop_share < -0.3:
                        gaps.append(
                            "High stop-out losses (D2) — stop calibration or entry quality issue"
                        )

            # Risk sizing — relevant for phase 5
            if phase in (5, 6):
                if metrics.get("max_drawdown_pct", 0) > 30:
                    gaps.append(
                        "Elevated drawdown (D3) — risk sizing needs tightening"
                    )

            return gaps

        # Fallback to metric-threshold checks when no _last_result
        if metrics.get("exit_efficiency", 1.0) < 0.35:
            gaps.append("MFE capture breakdown needed — exit efficiency below 35%")
        if metrics.get("win_rate", 100) < 45:
            gaps.append("Per-confirmation quality analysis needed — win rate below 45%")
        if metrics.get("total_trades", 0) < 8:
            gaps.append("Signal frequency analysis needed — fewer than 8 trades")
        if metrics.get("max_drawdown_pct", 0) > 35:
            gaps.append("Drawdown episode analysis needed — DD exceeds 35%")
        if metrics.get("profit_factor", 10) < 1.0:
            gaps.append("Loser profile analysis needed — negative expectancy")
        return gaps

    def _suggest_experiments_fn(
        self,
        phase: int,
        metrics: dict[str, float],
        weaknesses: list[str],
        state: Any,
    ) -> list[Experiment]:
        """Propose targeted experiments based on trade-level insights."""
        if not self._last_result or not self._last_result.trades:
            return []

        from crypto_trader.backtest.diagnostics import extract_diagnostic_insights
        insights = extract_diagnostic_insights(self._last_result.trades)
        experiments: list[Experiment] = []

        # Phase 1-2: If MFE capture is poor, suggest trail/TP adjustments
        if phase in (1, 2) and insights.mfe_capture.get("avg_capture_pct", 1) < 0.4:
            experiments.append(Experiment("SUGG_TRAIL_WIDE_3_5", {"trail.trail_buffer_wide": 3.5}))
            experiments.append(Experiment("SUGG_TRAIL_CEILING_5_0", {"trail.trail_r_ceiling": 5.0}))

        # Phase 3-4: Disable underperforming confirmations
        if phase in (3, 4):
            for conf_type, stats in insights.per_confirmation.items():
                if stats["n"] >= 2 and stats["avg_r"] < -0.3:
                    config_path = CONFIRMATION_DISABLE_MAP.get(conf_type)
                    if config_path:
                        experiments.append(Experiment(
                            f"SUGG_DISABLE_{conf_type.upper()}", {config_path: False}))

        # Phase 3: If confluence monotonicity fails, adjust gate
        if phase == 3 and len(insights.confluence) >= 2:
            sorted_confs = sorted(insights.confluence.items())
            low_r = sorted_confs[0][1].get("avg_r", 0)
            high_r = sorted_confs[-1][1].get("avg_r", 0)
            if low_r > high_r:  # More confluences = worse — monotonicity violation
                experiments.append(Experiment(
                    "SUGG_WEAK_GATE_0", {"confirmation.min_confluences_for_weak": 0}))

        # Phase 2: If winners give back >50% MFE, tighten trail
        if phase == 2 and insights.mfe_capture.get("avg_giveback_pct", 0) > 0.5:
            experiments.append(Experiment("SUGG_TRAIL_TIGHT_0_2", {"trail.trail_buffer_tight": 0.2}))
            experiments.append(Experiment("SUGG_TP1_R_0_9", {"exits.tp1_r": 0.9}))

        # Phase 2: High stop-out share → suggest stop buffer increase
        if phase == 2:
            stop_attr = insights.exit_attribution.get("protective_stop", {})
            total_exits = sum(a.get("n", 0) for a in insights.exit_attribution.values())
            if total_exits > 0 and stop_attr.get("n", 0) / total_exits > 0.5:
                experiments.append(Experiment(
                    "SUGG_STOP_BUF_0_5", {"stops.atr_buffer_mult": 0.5}))

        # Phase 1: Short duration → wider trail activation
        if phase == 1:
            avg_bars = insights.duration.get("avg_bars", 10)
            if avg_bars < 5:
                experiments.append(Experiment(
                    "SUGG_TRAIL_ACT_ATR_5", {"trail.trail_activation_bars": 5}))
            # Exits mostly by stop → suggest stop increase
            stop_attr = insights.exit_attribution.get("protective_stop", {})
            total_exits = sum(a.get("n", 0) for a in insights.exit_attribution.values())
            if total_exits > 0 and stop_attr.get("n", 0) / total_exits > 0.6:
                experiments.append(Experiment(
                    "SUGG_STOP_ATR_3_0", {"stops.min_stop_atr_mult": 3.0}))

        # Phase 5: Risk sizing experiments based on DD and concentration
        if phase == 5:
            dd = metrics.get("max_drawdown_pct", 0)
            pf = metrics.get("profit_factor", 0)
            if dd > 30:
                experiments.append(Experiment(
                    "SUGG_RISK_B_0_006", {"risk.risk_pct_b": 0.006}))
                experiments.append(Experiment(
                    "SUGG_RISK_B_0_008", {"risk.risk_pct_b": 0.008}))
            elif dd < 10 and pf > 2:
                experiments.append(Experiment(
                    "SUGG_RISK_B_0_014", {"risk.risk_pct_b": 0.014}))
            if insights.concentration.get("top1_pct", 0) > 0.5:
                experiments.append(Experiment(
                    "SUGG_CORR_RISK_0_02", {"risk.max_correlated_risk": 0.02}))

        # Phase 6: Fine-tune based on broad diagnostics
        if phase == 6:
            # Overall direction imbalance → direction filters per traded asset
            long_data = insights.direction.get("long", {})
            short_data = insights.direction.get("short", {})
            if long_data.get("avg_r", 0) < -0.3 and long_data.get("n", 0) >= 2:
                for sym in insights.per_asset:
                    sym_lower = sym.lower()
                    experiments.append(Experiment(
                        f"SUGG_{sym}_SHORT_ONLY",
                        {f"symbol_filter.{sym_lower}_direction": "short_only"}))
            if short_data.get("avg_r", 0) < -0.3 and short_data.get("n", 0) >= 2:
                for sym in insights.per_asset:
                    sym_lower = sym.lower()
                    experiments.append(Experiment(
                        f"SUGG_{sym}_LONG_ONLY",
                        {f"symbol_filter.{sym_lower}_direction": "long_only"}))
            # High avg duration → tighten quick exit
            if insights.duration.get("avg_bars", 0) > 20:
                experiments.append(Experiment(
                    "SUGG_QUICK_BARS_5", {"exits.quick_exit_bars": 5}))
            # Low MFE capture → trail adjustments
            if insights.mfe_capture.get("avg_capture_pct", 1) < 0.3:
                experiments.append(Experiment(
                    "SUGG_TRAIL_CEILING_1_0", {"trail.trail_r_ceiling": 1.0}))
                experiments.append(Experiment(
                    "SUGG_TRAIL_BUF_TIGHT_0_2", {"trail.trail_buffer_tight": 0.2}))

        return experiments

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
        """Structured decision logic for phase retry routing."""
        from crypto_trader.optimize.phase_state import PhaseState
        if not isinstance(state, PhaseState):
            return None

        diag_retries = state.diagnostic_retries.get(phase, 0)

        # Gate passed — always advance
        if gate_result.passed:
            accepted_msg = (f" with {greedy_result.accepted_count} mutations"
                            if greedy_result.accepted_count else "")
            return PhaseDecision(
                action="advance",
                reason=f"Gate passed{accepted_msg}",
            )

        # No experiments accepted — diagnostics first
        if greedy_result.accepted_count == 0:
            if diag_retries < max_diag:
                return PhaseDecision(
                    action="improve_diagnostics",
                    reason="No experiments improved the immutable score — deeper analysis needed",
                )
            return PhaseDecision(
                action="advance",
                reason="No experiments help, diagnostic budget exhausted",
            )

        return PhaseDecision(
            action="advance",
            reason=(
                "Gate failed after accepted mutations, but the score is intentionally "
                "immutable for this round"
            ),
        )

    def _redesign_scoring_weights_fn(
        self,
        phase: int,
        current_weights: dict[str, float],
        metrics: dict[str, float],
        strengths: list[str],
        weaknesses: list[str],
    ) -> dict[str, float] | None:
        """Round-3 replay keeps scoring fixed across all retries and phases."""
        return None

    def build_end_of_round_artifacts(
        self,
        state: Any,
    ) -> EndOfRoundArtifacts:
        """Build end-of-round evaluation artifacts with full diagnostics."""
        from crypto_trader.optimize.phase_state import PhaseState
        if not isinstance(state, PhaseState):
            return EndOfRoundArtifacts()

        # Run a final backtest with cumulative mutations
        config = apply_mutations(self.base_config, state.cumulative_mutations)
        bt_result = run(config, self.backtest_config, self.data_dir, store=self._get_store())
        pm = bt_result.metrics
        trades = bt_result.trades
        terminal_marks = bt_result.terminal_marks
        entries = bt_result.journal.entries if bt_result.journal else []

        # Full 22-section diagnostics
        from crypto_trader.backtest.diagnostics import (
            generate_diagnostics,
            extract_diagnostic_insights,
        )
        final_diagnostics_text = generate_diagnostics(
            trades,
            initial_equity=_diagnostic_initial_equity(self.backtest_config),
            terminal_marks=terminal_marks,
            performance_metrics=pm,
        ) if (trades or terminal_marks) else "(no trades)"
        insights = extract_diagnostic_insights(trades) if trades else None

        # Enhanced 5-dimension evaluation with trade-level insights
        dim_data = build_evaluation_report(pm, entries, insights=insights)
        dimension_reports = {
            name: format_dimension_text(name, data)
            for name, data in dim_data.items()
        }

        # Cross-phase progression + actionable verdict
        extra_sections: dict[str, str] = {}
        if state.phase_results:
            extra_sections["Phase Impact Analysis"] = self._build_phase_progression(state)

        verdict = self._build_verdict(pm, insights) if insights else self._build_basic_verdict(pm)

        return EndOfRoundArtifacts(
            final_diagnostics_text=final_diagnostics_text,
            dimension_reports=dimension_reports,
            overall_verdict=verdict,
            extra_sections=extra_sections,
        )

    def _build_phase_progression(self, state: Any) -> str:
        """Show how key metrics evolved across phases."""
        lines: list[str] = []
        tracked = ["total_trades", "win_rate", "profit_factor",
                    "max_drawdown_pct", "exit_efficiency", "sharpe_ratio"]
        for phase_num in sorted(state.phase_results.keys()):
            result = state.phase_results[phase_num]
            fm = result.get("final_metrics", {})
            accepted = result.get("accepted_count", 0)
            name = result.get("focus", f"Phase {phase_num}")
            lines.append(f"\nPhase {phase_num} ({name}) — {accepted} mutations:")
            for m in tracked:
                val = fm.get(m, 0)
                lines.append(f"  {m}: {val:.2f}")
        return "\n".join(lines)

    def _build_verdict(self, pm: PerformanceMetrics, insights: Any) -> str:
        """Build actionable verdict from metrics and trade-level insights."""
        parts: list[str] = []

        # Alpha capture
        if pm.profit_factor > 1.5 and pm.win_rate > 45:
            parts.append("ALPHA: Capturing meaningful alpha")
        elif pm.profit_factor > 1.0:
            parts.append("ALPHA: Marginal edge — signal quality needs improvement")
        else:
            parts.append("ALPHA: No edge — fundamental signal review needed")

        # MFE capture
        cap = insights.mfe_capture.get("avg_capture_pct", 0)
        if cap < 0.35:
            parts.append(f"CAPTURE: Poor ({cap:.0%}) — significant alpha left on table")
        elif cap < 0.55:
            parts.append(f"CAPTURE: Moderate ({cap:.0%}) — room for improvement")
        else:
            parts.append(f"CAPTURE: Good ({cap:.0%})")

        # Discrimination
        bad_confs = [c for c, s in insights.per_confirmation.items()
                     if s["avg_r"] < 0 and s["n"] >= 2]
        if bad_confs:
            parts.append(f"DISCRIMINATION: Value-destroying signals: {', '.join(bad_confs)}")
        else:
            parts.append("DISCRIMINATION: All signals positive expectancy")

        parts.append(
            f"\nTrades={pm.total_trades} | WR={pm.win_rate:.1f}% | "
            f"PF={pm.profit_factor:.2f} | DD={pm.max_drawdown_pct:.1f}% | "
            f"Sharpe={pm.sharpe_ratio:.2f} | Calmar={pm.calmar_ratio:.2f}"
        )
        return "\n".join(parts)

    def _build_basic_verdict(self, pm: PerformanceMetrics) -> str:
        """Fallback when no insights available."""
        return (
            f"Trades={pm.total_trades} | WR={pm.win_rate:.1f}% | "
            f"PF={pm.profit_factor:.2f} | DD={pm.max_drawdown_pct:.1f}% | "
            f"Sharpe={pm.sharpe_ratio:.2f} | Calmar={pm.calmar_ratio:.2f}"
        )

    def _build_extra_analysis_fn(
        self,
        phase: int,
        metrics: dict[str, float],
        state: Any,
        greedy_result: Any,
    ) -> dict[str, Any]:
        """Build extra analysis data for phase analysis report."""
        if not self._last_result or not self._last_result.trades:
            return {}

        from crypto_trader.backtest.diagnostics import extract_diagnostic_insights
        insights = extract_diagnostic_insights(self._last_result.trades)
        result: dict[str, Any] = {}

        if phase in (1, 2):
            # Trail efficiency summary
            trail_exits = sum(
                1 for t in self._last_result.trades
                if t.exit_reason == "trailing_stop"
            )
            result["trail_exits"] = trail_exits
            result["avg_capture_pct"] = insights.mfe_capture.get("avg_capture_pct", 0)
            result["avg_giveback_pct"] = insights.mfe_capture.get("avg_giveback_pct", 0)
            result["avg_bars_held"] = insights.duration.get("avg_bars", 0)

        elif phase in (3, 4):
            # Signal quality summary
            result["confirmation_types"] = len(insights.per_confirmation)
            result["positive_confirmations"] = sum(
                1 for s in insights.per_confirmation.values()
                if s.get("avg_r", 0) > 0
            )
            result["negative_confirmations"] = sum(
                1 for s in insights.per_confirmation.values()
                if s.get("avg_r", 0) < 0 and s.get("n", 0) >= 2
            )
            # Confluence count distribution
            result["confluence_levels"] = len(insights.confluence)

        elif phase in (5, 6):
            # Risk utilization summary
            result["max_drawdown_pct"] = metrics.get("max_drawdown_pct", 0)
            result["total_trades"] = metrics.get("total_trades", 0)
            result["concentration_top1"] = insights.concentration.get("top1_pct", 0)
            result["long_count"] = insights.direction.get("long", {}).get("n", 0)
            result["short_count"] = insights.direction.get("short", {}).get("n", 0)

        return result

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
