"""Dedicated breakout round-3 runner seeded from the pre-round-1 baseline.

This module rebuilds the earliest recoverable breakout baseline, then runs a
single phased optimization over the union of distinct round-1 and round-2
experiment settings on the maximum common BTC/ETH/SOL data span.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.backtest.profiles import LIVE_PARITY_PROFILE, build_backtest_config_from_profile
from crypto_trader.optimize.breakout_plugin import BreakoutPlugin, PHASE_NAMES
from crypto_trader.optimize.parallel import evaluate_parallel
from crypto_trader.optimize.types import (
    EvaluateFn,
    Experiment,
    GateCriterion,
    PhaseAnalysisPolicy,
    PhaseSpec,
    ScoredCandidate,
)
from crypto_trader.strategy.breakout.config import BreakoutConfig

SYMBOLS: list[str] = ["BTC", "ETH", "SOL"]
TIMEFRAMES: list[str] = ["30m", "4h"]

# Fixed score for this run: enough headroom above the historical round-2
# full-span result, but still sensitive near the weak pre-round-1 baseline.
IMMUTABLE_SCORING_WEIGHTS: dict[str, float] = {
    "returns": 0.22,
    "edge": 0.22,
    "sharpe": 0.16,
    "risk": 0.15,
    "coverage": 0.15,
    "capture": 0.10,
}

IMMUTABLE_SCORING_CEILINGS: dict[str, float] = {
    "returns": 15.0,
    "edge": 3.0,
    "sharpe": 2.5,
    "risk": 20.0,
    "coverage": 18.0,
}

# Keep gates permissive enough for early structural phases; edge is handled by
# the immutable score itself, while these reject only unusable paths.
IMMUTABLE_HARD_REJECTS: dict[str, tuple[str, float]] = {
    "total_trades": (">=", 6.0),
    "max_drawdown_pct": ("<=", 20.0),
}

IMMUTABLE_GATE_CRITERIA: list[GateCriterion] = [
    GateCriterion(metric="total_trades", operator=">=", threshold=6.0),
    GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=20.0),
]


def detect_common_window(
    data_dir: Path,
    *,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
) -> tuple[datetime, datetime]:
    """Return the common UTC window across candles and funding."""
    symbols = symbols or SYMBOLS
    timeframes = timeframes or TIMEFRAMES

    per_symbol_starts: list[int] = []
    per_symbol_ends: list[int] = []

    for symbol in symbols:
        starts: list[int] = []
        ends: list[int] = []

        for timeframe in timeframes:
            path = data_dir / "candles" / symbol / f"{timeframe}.parquet"
            frame = pd.read_parquet(path, columns=["ts"])
            starts.append(int(frame["ts"].min()))
            ends.append(int(frame["ts"].max()))

        funding_path = data_dir / "funding" / f"{symbol}.parquet"
        funding = pd.read_parquet(funding_path, columns=["ts"])
        starts.append(int(funding["ts"].min()))
        ends.append(int(funding["ts"].max()))

        per_symbol_starts.append(max(starts))
        per_symbol_ends.append(min(ends))

    start_ms = max(per_symbol_starts)
    end_ms = min(per_symbol_ends)
    return (
        datetime.fromtimestamp(start_ms / 1000, tz=UTC),
        datetime.fromtimestamp(end_ms / 1000, tz=UTC),
    )


def build_pre_round1_config() -> BreakoutConfig:
    """Reconstruct the earliest recoverable breakout baseline.

    These values come from a combination of current inline comments, the saved
    round-1/round-2 artifacts, and the implementation/spec docs. Anything not
    directly evidenced is left at the current strategy default.
    """
    cfg = BreakoutConfig()

    # Pre-round-1 / pre-round-2 values explicitly documented in code comments.
    cfg.profile.lookback_bars = 24
    cfg.profile.hvn_threshold_pct = 1.5
    cfg.balance.min_bars_in_zone = 6
    cfg.setup.body_ratio_min = 0.30
    cfg.confirmation.enable_model2 = True
    cfg.confirmation.model1_require_direction_close = True
    cfg.stops.atr_mult = 1.0
    cfg.stops.use_farther = True
    cfg.exits.tp1_r = 0.5
    cfg.exits.tp1_frac = 0.65
    cfg.exits.be_buffer_r = 0.3
    cfg.exits.time_stop_action = "close"
    cfg.exits.invalidation_depth_atr = 0.8
    cfg.exits.quick_exit_enabled = False
    cfg.exits.quick_exit_max_mfe_r = 0.15
    cfg.exits.quick_exit_max_r = -0.3
    cfg.trail.trail_activation_r = 0.5
    cfg.trail.trail_activation_bars = 6
    cfg.trail.structure_trail_enabled = True
    cfg.symbol_filter.eth_direction = "both"

    # Earliest documented risk/session limits from the implementation plan.
    cfg.risk.risk_pct_a_plus = 0.0075
    cfg.risk.risk_pct_a = 0.0075
    cfg.risk.risk_pct_b = 0.0040
    cfg.limits.max_consecutive_losses = 2
    cfg.limits.max_daily_loss_pct = 0.015
    cfg.limits.max_trades_per_day = 4

    cfg.symbols = list(SYMBOLS)
    return cfg


def _exp(name: str, mutations: dict[str, Any]) -> Experiment:
    return Experiment(name=name, mutations=mutations)


def _phase1_candidates() -> list[Experiment]:
    """Signal architecture union from rounds 1 and 2."""
    return [
        _exp("model2_off", {"confirmation.enable_model2": False}),
        _exp("no_dir_close", {"confirmation.model1_require_direction_close": False}),
        _exp("eth_long_only", {"symbol_filter.eth_direction": "long_only"}),
        _exp("eth_disabled", {"symbol_filter.eth_direction": "disabled"}),
        _exp("sol_long_only", {"symbol_filter.sol_direction": "long_only"}),
        _exp("b_conf_1", {"setup.min_confluences_b": 1}),
        _exp("b_conf_2", {"setup.min_confluences_b": 2}),
        _exp("a_conf_3", {"setup.min_confluences_a": 3}),
        _exp("require_vol_surge", {"setup.require_volume_surge": True}),
        _exp("min_bo_atr_03", {"setup.min_breakout_atr": 0.3}),
        _exp("min_bo_atr_05", {"setup.min_breakout_atr": 0.5}),
        _exp("body_ratio_045", {"setup.body_ratio_min": 0.45}),
        _exp("body_ratio_04675", {"setup.body_ratio_min": 0.4675}),
        _exp("body_ratio_055", {"setup.body_ratio_min": 0.55}),
        _exp("room_r_b_15", {"setup.min_room_r_b": 1.5}),
        _exp("room_r_a_22", {"setup.min_room_r_a": 2.2}),
        _exp("no_countertrend", {"context.allow_countertrend": False}),
        _exp("h4_adx_15", {"context.h4_adx_threshold": 15.0}),
        _exp("h4_adx_20", {"context.h4_adx_threshold": 20.0}),
        _exp("retest_bars_8", {"confirmation.retest_max_bars": 8}),
        _exp("retest_bars_10", {"confirmation.retest_max_bars": 10}),
        _exp("retest_bars_12", {"confirmation.retest_max_bars": 12}),
        _exp("retest_zone_08", {"confirmation.retest_zone_atr": 0.8}),
        _exp("no_vol_gate", {"confirmation.model1_require_volume": False}),
        _exp("vol_mult_13", {"confirmation.model1_min_volume_mult": 1.3}),
    ]


def _phase2_candidates() -> list[Experiment]:
    """Exit/capture union from rounds 1 and 2 plus round-2 bridge values."""
    return [
        _exp("tp1_r_0.6", {"exits.tp1_r": 0.6}),
        _exp("tp1_r_0.8", {"exits.tp1_r": 0.8}),
        _exp("tp1_r_1.0", {"exits.tp1_r": 1.0}),
        _exp("tp1_r_1.2", {"exits.tp1_r": 1.2}),
        _exp("tp1_frac_0.2", {"exits.tp1_frac": 0.2}),
        _exp("tp1_frac_0.25", {"exits.tp1_frac": 0.25}),
        _exp("tp1_frac_0.3", {"exits.tp1_frac": 0.3}),
        _exp("tp1_frac_0.4", {"exits.tp1_frac": 0.4}),
        _exp("tp1_frac_0.5", {"exits.tp1_frac": 0.5}),
        _exp("tp1_frac_0.8", {"exits.tp1_frac": 0.8}),
        _exp("tp2_r_1.5", {"exits.tp2_r": 1.5}),
        _exp("tp2_r_2.5", {"exits.tp2_r": 2.5}),
        _exp("tp2_r_3.0", {"exits.tp2_r": 3.0}),
        _exp("tp2_frac_0.2", {"exits.tp2_frac": 0.2}),
        _exp("tp2_frac_0.25", {"exits.tp2_frac": 0.25}),
        _exp("tp2_frac_0.3", {"exits.tp2_frac": 0.3}),
        _exp("tp2_frac_0.5", {"exits.tp2_frac": 0.5}),
        _exp("quick_exit_on", {"exits.quick_exit_enabled": True}),
        _exp(
            "qe_bars_3",
            {"exits.quick_exit_enabled": True, "exits.quick_exit_bars": 3},
        ),
        _exp(
            "qe_bars_6",
            {"exits.quick_exit_enabled": True, "exits.quick_exit_bars": 6},
        ),
        _exp(
            "qe_mfe_015",
            {"exits.quick_exit_enabled": True, "exits.quick_exit_max_mfe_r": 0.15},
        ),
        _exp(
            "qe_mfe_03",
            {"exits.quick_exit_enabled": True, "exits.quick_exit_max_mfe_r": 0.3},
        ),
        _exp(
            "qe_r_neg01",
            {"exits.quick_exit_enabled": True, "exits.quick_exit_max_r": -0.1},
        ),
        _exp(
            "qe_r_neg03",
            {"exits.quick_exit_enabled": True, "exits.quick_exit_max_r": -0.3},
        ),
        _exp("time_stop_action_reduce", {"exits.time_stop_action": "reduce"}),
        _exp("time_stop_10", {"exits.time_stop_bars": 10}),
        _exp("time_stop_24", {"exits.time_stop_bars": 24}),
        _exp("time_progress_015", {"exits.time_stop_min_progress_r": 0.15}),
        _exp("be_buffer_01", {"exits.be_buffer_r": 0.1}),
        _exp("be_buffer_0.4", {"exits.be_buffer_r": 0.4}),
        _exp("be_buffer_05", {"exits.be_buffer_r": 0.5}),
        _exp("be_buffer_0.6", {"exits.be_buffer_r": 0.6}),
        _exp("be_buffer_0.7", {"exits.be_buffer_r": 0.7}),
        _exp("invalidation_depth_1.2", {"exits.invalidation_depth_atr": 1.2}),
        _exp("no_invalidation", {"exits.invalidation_exit": False}),
        _exp("invalidation_depth_0.5", {"exits.invalidation_depth_atr": 0.5}),
        _exp("invalidation_depth_1.5", {"exits.invalidation_depth_atr": 1.5}),
        _exp("invalidation_depth_2.0", {"exits.invalidation_depth_atr": 2.0}),
        _exp("invalidation_minbars_1", {"exits.invalidation_min_bars": 1}),
        _exp("invalidation_minbars_5", {"exits.invalidation_min_bars": 5}),
    ]


def _phase3_candidates() -> list[Experiment]:
    """Trail/stop union from rounds 1 and 2 plus round-2 bridge values."""
    return [
        _exp("trail_act_r_0.2", {"trail.trail_activation_r": 0.2}),
        _exp("trail_act_r_0.3", {"trail.trail_activation_r": 0.3}),
        _exp("trail_act_r_0.4", {"trail.trail_activation_r": 0.4}),
        _exp("trail_act_r_0.7", {"trail.trail_activation_r": 0.7}),
        _exp("trail_act_bars_3", {"trail.trail_activation_bars": 3}),
        _exp("trail_act_bars_4", {"trail.trail_activation_bars": 4}),
        _exp("trail_act_bars_5", {"trail.trail_activation_bars": 5}),
        _exp("trail_wide_1.0", {"trail.trail_buffer_wide": 1.0}),
        _exp("trail_wide_2.0", {"trail.trail_buffer_wide": 2.0}),
        _exp("trail_tight_0.1", {"trail.trail_buffer_tight": 0.1}),
        _exp("trail_tight_0.2", {"trail.trail_buffer_tight": 0.2}),
        _exp("trail_tight_0.3", {"trail.trail_buffer_tight": 0.3}),
        _exp("trail_ceiling_0.8", {"trail.trail_r_ceiling": 0.8}),
        _exp("trail_ceiling_1.0", {"trail.trail_r_ceiling": 1.0}),
        _exp("trail_ceiling_1.5", {"trail.trail_r_ceiling": 1.5}),
        _exp("structure_trail_off", {"trail.structure_trail_enabled": False}),
        _exp("use_farther_off", {"stops.use_farther": False}),
        _exp("stop_atr_0.8", {"stops.atr_mult": 0.8}),
        _exp("stop_atr_1.5", {"stops.atr_mult": 1.5}),
        _exp("min_stop_05", {"stops.min_stop_atr": 0.5}),
    ]


def _phase4_candidates() -> list[Experiment]:
    """Zone/profile union from rounds 1 and 2 plus round-2 bridge values."""
    return [
        _exp("min_bars_zone_2", {"balance.min_bars_in_zone": 2}),
        _exp("min_bars_zone_8", {"balance.min_bars_in_zone": 8}),
        _exp("min_touches_3", {"balance.min_touches": 3}),
        _exp("zone_age_16", {"balance.max_zone_age_bars": 16}),
        _exp("zone_age_36", {"balance.max_zone_age_bars": 36}),
        _exp("zone_width_1.0", {"balance.zone_width_atr": 1.0}),
        _exp("zone_width_1.5", {"balance.zone_width_atr": 1.5}),
        _exp("lookback_16", {"profile.lookback_bars": 16}),
        _exp("lookback_36", {"profile.lookback_bars": 36}),
        _exp("lookback_48", {"profile.lookback_bars": 48}),
        _exp("hvn_thresh_1.0", {"profile.hvn_threshold_pct": 1.0}),
        _exp("hvn_thresh_1.2", {"profile.hvn_threshold_pct": 1.2}),
        _exp("hvn_thresh_1.8", {"profile.hvn_threshold_pct": 1.8}),
        _exp("lvn_thresh_03", {"profile.lvn_threshold_pct": 0.3}),
        _exp("lvn_runway_0.5", {"setup.min_lvn_runway_atr": 0.5}),
        _exp("lvn_runway_0.8", {"setup.min_lvn_runway_atr": 0.8}),
        _exp("dedup_02", {"balance.dedup_atr_frac": 0.2}),
    ]


def _phase5_candidates() -> list[Experiment]:
    """Risk/sizing union from rounds 1 and 2 plus bridge values."""
    return [
        _exp("risk_b_0.006", {"risk.risk_pct_b": 0.006}),
        _exp("risk_b_0.01", {"risk.risk_pct_b": 0.01}),
        _exp("risk_b_0.012", {"risk.risk_pct_b": 0.012}),
        _exp("risk_b_0.018", {"risk.risk_pct_b": 0.018}),
        _exp("risk_b_0.02", {"risk.risk_pct_b": 0.02}),
        _exp("risk_a_0.015", {"risk.risk_pct_a": 0.015}),
        _exp("risk_a_0.02", {"risk.risk_pct_a": 0.02}),
        _exp("risk_a_0.025", {"risk.risk_pct_a": 0.025}),
        _exp("risk_a_plus_0.018", {"risk.risk_pct_a_plus": 0.018}),
        _exp("risk_a_plus_0.02", {"risk.risk_pct_a_plus": 0.02}),
        _exp("risk_a_plus_0.03", {"risk.risk_pct_a_plus": 0.03}),
        _exp("consec_loss_3", {"limits.max_consecutive_losses": 3}),
        _exp("consec_loss_5", {"limits.max_consecutive_losses": 5}),
        _exp("consec_loss_8", {"limits.max_consecutive_losses": 8}),
        _exp("daily_loss_0025", {"limits.max_daily_loss_pct": 0.025}),
        _exp("daily_loss_004", {"limits.max_daily_loss_pct": 0.04}),
        _exp("trades_per_day_5", {"limits.max_trades_per_day": 5}),
        _exp("trades_per_day_8", {"limits.max_trades_per_day": 8}),
        _exp("concurrent_5", {"limits.max_concurrent_positions": 5}),
        _exp("btc_long_only", {"symbol_filter.btc_direction": "long_only"}),
        _exp("sol_long_only_risk", {"symbol_filter.sol_direction": "long_only"}),
        _exp("no_reentry", {"reentry.enabled": False}),
        _exp("reentry_cooldown_6", {"reentry.cooldown_bars": 6}),
        _exp("reentry_max_2", {"reentry.max_reentries": 2}),
    ]


def _phase6_candidates(cumulative: dict[str, Any]) -> list[Experiment]:
    """Integer-safe finetune sweep across accepted numeric mutations."""
    experiments: list[Experiment] = []
    for key, val in cumulative.items():
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue

        is_int = isinstance(val, int) and not isinstance(val, bool)
        for mult in (0.85, 0.95, 1.05, 1.15):
            raw = val * mult
            if is_int:
                new_val = int(round(raw))
                if new_val <= 0:
                    continue
            else:
                new_val = round(raw, 6)

            if new_val == val:
                continue

            suffix = (
                f"{new_val}"
                .replace(".", "_")
                .replace("-", "neg_")
            )
            experiments.append(
                _exp(
                    f"finetune_{key.split('.')[-1]}_{suffix}",
                    {key: new_val},
                )
            )
    return experiments


_PHASE_CANDIDATES = {
    1: _phase1_candidates,
    2: _phase2_candidates,
    3: _phase3_candidates,
    4: _phase4_candidates,
    5: _phase5_candidates,
}


class BreakoutRound3PreRound1Plugin(BreakoutPlugin):
    """Custom round-3 plugin using a fixed score and combined R1/R2 search."""

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "total_trades": 18.0,
            "win_rate": 45.0,
            "profit_factor": 1.8,
            "max_drawdown_pct": 12.0,
            "sharpe_ratio": 1.5,
        }

    def get_phase_spec(self, phase: int, state: Any) -> PhaseSpec:
        if phase == 6:
            cumulative = state.cumulative_mutations if state else {}
            candidates = _phase6_candidates(cumulative)
        else:
            gen = _PHASE_CANDIDATES.get(phase)
            candidates = gen() if gen else []

        policy = PhaseAnalysisPolicy(
            max_scoring_retries=0,
            max_diagnostic_retries=0,
            diagnostic_gap_fn=lambda p, m: self._diagnostic_gap_fn(p, m),
            suggest_experiments_fn=lambda p, m, w, s: self._suggest_experiments_fn(p, m, w, s),
            decide_action_fn=lambda *args: self._decide_action_fn(*args),
            redesign_scoring_weights_fn=lambda *args: None,
            build_extra_analysis_fn=lambda p, m, s, g: self._build_extra_analysis_fn(p, m, s, g),
            format_extra_analysis_fn=lambda d: self._format_extra_analysis_fn(d),
        )

        return PhaseSpec(
            phase_num=phase,
            name=PHASE_NAMES.get(phase, f"Phase {phase}"),
            candidates=candidates,
            scoring_weights=dict(IMMUTABLE_SCORING_WEIGHTS),
            hard_rejects=dict(IMMUTABLE_HARD_REJECTS),
            min_delta=0.005,
            max_rounds=4 if phase < 6 else 3,
            prune_threshold=0.08 if phase == 1 else 0.05,
            gate_criteria=list(IMMUTABLE_GATE_CRITERIA),
            gate_criteria_fn=lambda _m: list(IMMUTABLE_GATE_CRITERIA),
            analysis_policy=policy,
            focus=PHASE_NAMES.get(phase, ""),
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        scoring_weights: dict[str, float],
        hard_rejects: dict[str, tuple[str, float]],
    ) -> EvaluateFn:
        ceilings = IMMUTABLE_SCORING_CEILINGS

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


def build_backtest_config(data_dir: Path) -> tuple[BacktestConfig, dict[str, str]]:
    """Build a full-span backtest config for the shared breakout data window."""
    start_dt, end_dt = detect_common_window(data_dir)
    bt_cfg = build_backtest_config_from_profile(
        profile=LIVE_PARITY_PROFILE,
        symbols=list(SYMBOLS),
        start_date=start_dt.date(),
        end_date=end_dt.date(),
    )
    metadata = {
        "common_start_utc": start_dt.isoformat(),
        "common_end_utc": end_dt.isoformat(),
        "start_date": bt_cfg.start_date.isoformat(),
        "end_date": bt_cfg.end_date.isoformat(),
    }
    return bt_cfg, metadata
