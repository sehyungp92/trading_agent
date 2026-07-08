"""Run trend round 4 from the reconstructed pre-round-1 baseline.

This replay keeps the normal optimizer code untouched while restoring the
historical candidate values that were later baked into defaults. It also uses
fixed score ceilings sized for the full BTC/ETH/SOL common window so score
scaling stays immutable across the round.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog
import yaml

from crypto_trader.cli import _detect_next_round, _update_rounds_manifest
from crypto_trader.backtest.profiles import LIVE_PARITY_PROFILE, build_backtest_config_from_profile
from crypto_trader.data.store import ParquetStore
from crypto_trader.optimize.parallel import evaluate_parallel
from crypto_trader.optimize.phase_runner import PhaseRunner
from crypto_trader.optimize.phase_state import PhaseState, _atomic_write_json
from crypto_trader.optimize.trend_plugin import (
    HARD_REJECTS,
    PHASE_DIAGNOSTIC_MODULES,
    PHASE_GATE_CRITERIA,
    PHASE_NAMES,
    PHASE_SCORING_EMPHASIS,
    SCORING_WEIGHTS,
    TrendPlugin,
    _phase1_candidates,
    _phase2_candidates,
    _phase3_candidates,
    _phase4_candidates,
    _phase5_candidates,
)
from crypto_trader.optimize.types import (
    EndOfRoundArtifacts,
    EvaluateFn,
    Experiment,
    GateCriterion,
    PhaseAnalysisPolicy,
    PhaseSpec,
    ScoredCandidate,
)
from crypto_trader.strategy.trend.config import TrendConfig

ROOT = Path(__file__).resolve().parents[1]
SEED_CONFIG_PATH = ROOT / "config" / "trend_pre_round1.yaml"
DATA_DIR = ROOT / "data"
OUTPUT_BASE = ROOT / "output" / "trend"
SYMBOLS = ["BTC", "ETH", "SOL"]
TIMEFRAMES = ["15m", "1h", "1d"]
MAX_WORKERS = 2

# Fixed ceilings sized for the full 3-instrument replay window.
# The current production plugin ceilings are centered on an already-profitable
# baseline. For this replay, coverage and edge need more room to discriminate
# across the much larger max-window candidate set.
IMMUTABLE_SCORING_CEILINGS: dict[str, float] = {
    "returns": 25.0,
    "edge": 4.0,
    "coverage": 120.0,
    "calmar": 6.0,
}


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),
    )


log = structlog.get_logger("scripts.trend_round4_replay")


def _dedupe_candidates(candidates: list[Experiment]) -> list[Experiment]:
    seen: set[str] = set()
    ordered: list[Experiment] = []
    for candidate in candidates:
        if candidate.name not in seen:
            seen.add(candidate.name)
            ordered.append(candidate)
    return ordered


def _merge_candidates(
    base_candidates: list[Experiment],
    extras: list[Experiment],
    *,
    excluded_names: set[str] | None = None,
) -> list[Experiment]:
    excluded_names = excluded_names or set()
    filtered = [candidate for candidate in base_candidates if candidate.name not in excluded_names]
    return _dedupe_candidates([*filtered, *extras])


def _phase1_replay_candidates() -> list[Experiment]:
    extras = [
        Experiment("impulse_atr_0.8", {"setup.impulse_min_atr_move": 0.8}),
        Experiment("disable_hammer", {"confirmation.enable_hammer": False}),
    ]
    return _merge_candidates(
        _phase1_candidates(),
        extras,
        excluded_names={"enable_hammer"},
    )


def _phase2_replay_candidates() -> list[Experiment]:
    extras = [
        Experiment("h1_regime_on_15", {
            "regime.h1_regime_enabled": True,
            "regime.h1_min_adx": 15.0,
        }),
        Experiment("h1_regime_on_18", {
            "regime.h1_regime_enabled": True,
            "regime.h1_min_adx": 18.0,
        }),
        Experiment("h1_regime_on_20", {
            "regime.h1_regime_enabled": True,
            "regime.h1_min_adx": 20.0,
        }),
        Experiment("h1_regime_on_22", {
            "regime.h1_regime_enabled": True,
            "regime.h1_min_adx": 22.0,
        }),
        Experiment("h1_regime_on_25", {
            "regime.h1_regime_enabled": True,
            "regime.h1_min_adx": 25.0,
        }),
        Experiment("h1_regime_on_30", {
            "regime.h1_regime_enabled": True,
            "regime.h1_min_adx": 30.0,
        }),
    ]
    excluded = {"h1_regime_off"}
    excluded.update(
        candidate.name
        for candidate in _phase2_candidates()
        if candidate.name.startswith("h1_adx_")
    )
    return _merge_candidates(_phase2_candidates(), extras, excluded_names=excluded)


def _phase3_replay_candidates() -> list[Experiment]:
    extras = [
        Experiment("trail_tight_0.1", {"trail.trail_buffer_tight": 0.1}),
        Experiment("stop_atr_2.0", {"stops.atr_mult": 2.0}),
    ]
    return _merge_candidates(_phase3_candidates(), extras)


def _phase4_replay_candidates() -> list[Experiment]:
    extras = [
        Experiment("tp1_r_0.8", {"exits.tp1_r": 0.8}),
        Experiment("time_stop_20", {"exits.time_stop_bars": 20}),
    ]
    return _merge_candidates(_phase4_candidates(), extras)


def _phase6_replay_candidates(
    cumulative_mutations: dict[str, Any],
    base_config: TrendConfig,
) -> list[Experiment]:
    experiments: list[Experiment] = []

    for value in [15, 20, 25]:
        experiments.append(Experiment(f"ema_trail_{value}", {"trail.ema_trail_period": value}))
    for value in [15, 20, 30]:
        experiments.append(Experiment(f"h1_ema_fast_{value}", {"h1_indicators.ema_fast": value}))

    current_wide = float(
        cumulative_mutations.get("trail.trail_buffer_wide", base_config.trail.trail_buffer_wide)
    )
    current_tight = float(
        cumulative_mutations.get("trail.trail_buffer_tight", base_config.trail.trail_buffer_tight)
    )
    experiments.append(Experiment("compound_trail_tighter", {
        "trail.trail_buffer_wide": round(current_wide * 0.85, 2),
        "trail.trail_buffer_tight": round(current_tight * 0.85, 2),
    }))

    for bars in [6, 10, 16]:
        experiments.append(Experiment(f"qe_bars_{bars}", {"exits.quick_exit_bars": bars}))
    for mfe in [0.1, 0.2, 0.3]:
        experiments.append(Experiment(f"qe_mfe_{mfe}", {"exits.quick_exit_max_mfe_r": mfe}))

    for value in [18.0, 20.0, 22.0, 24.0, 26.0]:
        experiments.append(Experiment(f"h1_adx_{value}", {
            "regime.h1_regime_enabled": True,
            "regime.h1_min_adx": value,
        }))

    experiments.append(Experiment("btc_long_only", {"symbol_filter.btc_direction": "long_only"}))
    experiments.append(Experiment("sol_long_only", {"symbol_filter.sol_direction": "long_only"}))

    perturbable = {
        "setup.impulse_min_atr_move",
        "setup.min_room_r",
        "setup.pullback_max_retrace",
        "trail.trail_r_ceiling",
        "trail.trail_buffer_wide",
        "trail.trail_buffer_tight",
        "exits.tp1_r",
        "exits.tp2_r",
        "exits.be_buffer_r",
        "exits.time_stop_bars",
        "stops.atr_mult",
        "risk.risk_pct_a",
        "risk.risk_pct_b",
        "regime.a_min_adx",
        "regime.b_min_adx",
        "regime.h1_min_adx",
    }
    for key, value in cumulative_mutations.items():
        if key in perturbable and isinstance(value, (int, float)):
            for multiplier in [0.8, 1.2]:
                experiments.append(Experiment(
                    f"perturb_{key.split('.')[-1]}_{multiplier}",
                    {key: round(float(value) * multiplier, 4)},
                ))

    return _dedupe_candidates(experiments)


REPLAY_PHASE_CANDIDATES = {
    1: _phase1_replay_candidates,
    2: _phase2_replay_candidates,
    3: _phase3_replay_candidates,
    4: _phase4_replay_candidates,
    5: _phase5_candidates,
}


class ReplayTrendRound4Plugin(TrendPlugin):
    """Trend plugin variant for the pre-round-1 -> round-4 replay."""

    def get_phase_spec(self, phase: int, state: Any) -> PhaseSpec:
        if phase == 6:
            cumulative_mutations = getattr(state, "cumulative_mutations", {})
            candidates = _phase6_replay_candidates(cumulative_mutations, self.base_config)
        else:
            generator = REPLAY_PHASE_CANDIDATES.get(phase)
            if generator is None:
                raise ValueError(f"Unknown phase: {phase}")
            candidates = generator()

        return PhaseSpec(
            phase_num=phase,
            name=PHASE_NAMES[phase],
            candidates=candidates,
            scoring_weights=dict(PHASE_SCORING_EMPHASIS.get(phase, SCORING_WEIGHTS)),
            hard_rejects=dict(HARD_REJECTS),
            min_delta=0.005,
            max_rounds=3,
            gate_criteria=list(PHASE_GATE_CRITERIA[phase]),
            gate_criteria_fn=lambda metrics, _phase=phase: self._gate_criteria_fn(metrics, _phase),
            analysis_policy=PhaseAnalysisPolicy(
                diagnostic_gap_fn=lambda replay_phase, metrics: self._diagnostic_gap_fn(
                    replay_phase, metrics
                ),
                suggest_experiments_fn=lambda replay_phase, metrics, weaknesses, replay_state:
                    self._suggest_experiments_fn(
                        replay_phase,
                        metrics,
                        weaknesses,
                        replay_state,
                    ),
                decide_action_fn=lambda *args: self._decide_action_fn(*args),
                redesign_scoring_weights_fn=lambda *args: self._redesign_scoring_weights_fn(*args),
                build_extra_analysis_fn=lambda replay_phase, metrics, replay_state, greedy_result:
                    self._build_extra_analysis_fn(replay_phase, metrics, replay_state, greedy_result),
                format_extra_analysis_fn=lambda extra: self._format_extra_analysis_fn(extra),
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
        ceilings = dict(IMMUTABLE_SCORING_CEILINGS)

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


def _load_seed_config(path: Path) -> TrendConfig:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return TrendConfig.from_dict(raw.get("strategy", raw))


def _time_bounds_from_ts(min_ts: int, max_ts: int) -> tuple[datetime, datetime]:
    return (
        datetime.fromtimestamp(min_ts / 1000, tz=timezone.utc),
        datetime.fromtimestamp(max_ts / 1000, tz=timezone.utc),
    )


def _compute_common_window(
    data_dir: Path,
    symbols: list[str],
) -> tuple[datetime, datetime, dict[str, dict[str, str]]]:
    store = ParquetStore(base_dir=data_dir)
    common_start: datetime | None = None
    common_end: datetime | None = None
    detail: dict[str, dict[str, str]] = {}

    for symbol in symbols:
        for timeframe in TIMEFRAMES:
            df = store.load_candles(symbol, timeframe)
            if df is None or df.empty:
                raise RuntimeError(f"Missing candle data for {symbol} {timeframe}")
            start_dt, end_dt = _time_bounds_from_ts(int(df["ts"].min()), int(df["ts"].max()))
            label = f"{symbol}_{timeframe}"
            detail[label] = {"start": start_dt.isoformat(), "end": end_dt.isoformat()}
            common_start = start_dt if common_start is None else max(common_start, start_dt)
            common_end = end_dt if common_end is None else min(common_end, end_dt)

        funding_df = store.load_funding(symbol)
        if funding_df is None or funding_df.empty:
            raise RuntimeError(f"Missing funding data for {symbol}")
        start_dt, end_dt = _time_bounds_from_ts(
            int(funding_df["ts"].min()),
            int(funding_df["ts"].max()),
        )
        detail[f"{symbol}_funding"] = {"start": start_dt.isoformat(), "end": end_dt.isoformat()}
        common_start = start_dt if common_start is None else max(common_start, start_dt)
        common_end = end_dt if common_end is None else min(common_end, end_dt)

    if common_start is None or common_end is None or common_start > common_end:
        raise RuntimeError("Unable to derive a common BTC/ETH/SOL replay window.")

    return common_start, common_end, detail


def _build_replay_context(
    round_num: int,
    start_dt: datetime,
    end_dt: datetime,
    data_ranges: dict[str, dict[str, str]],
    plugin: ReplayTrendRound4Plugin,
    round_dir: Path,
) -> None:
    warmup_days = plugin.backtest_config.warmup_days
    warmup_safe_start = start_dt + timedelta(days=warmup_days)
    phase_counts: dict[str, int] = {}
    for phase in range(1, plugin.num_phases + 1):
        if phase == 6:
            candidates = _phase6_replay_candidates({}, plugin.base_config)
        else:
            candidates = REPLAY_PHASE_CANDIDATES[phase]()
        phase_counts[f"phase_{phase}"] = len(candidates)

    context = {
        "round": round_num,
        "baseline_config": str(SEED_CONFIG_PATH),
        "symbols": SYMBOLS,
        "max_workers": plugin.max_workers,
        "data_window_start": start_dt.isoformat(),
        "data_window_end": end_dt.isoformat(),
        "measurement_start": plugin.backtest_config.start_date.isoformat(),
        "measurement_end": plugin.backtest_config.end_date.isoformat(),
        "warmup_days": warmup_days,
        "full_warmup_available": warmup_safe_start.date() <= end_dt.date(),
        "warmup_safe_start_if_available": warmup_safe_start.date().isoformat(),
        "immutable_scoring_ceilings": IMMUTABLE_SCORING_CEILINGS,
        "phase_candidate_counts": phase_counts,
        "phase_scoring_emphasis": PHASE_SCORING_EMPHASIS,
        "phase_gate_criteria": {
            str(phase): [criterion.__dict__ for criterion in criteria]
            for phase, criteria in PHASE_GATE_CRITERIA.items()
        },
        "diagnostic_modules": PHASE_DIAGNOSTIC_MODULES,
        "data_ranges": data_ranges,
    }
    _atomic_write_json(context, round_dir / "replay_context.json")


def main() -> None:
    _configure_logging()

    round_num = _detect_next_round(OUTPUT_BASE)
    if round_num != 4:
        raise RuntimeError(
            f"Expected next trend round to be 4, but detected round_{round_num} in {OUTPUT_BASE}."
        )

    round_dir = OUTPUT_BASE / f"round_{round_num}"
    round_dir.mkdir(parents=True, exist_ok=True)

    seed_config = _load_seed_config(SEED_CONFIG_PATH)
    common_start, common_end, data_ranges = _compute_common_window(DATA_DIR, SYMBOLS)
    start_date = common_start.date()
    end_date = common_end.date()

    bt_config = build_backtest_config_from_profile(
        profile=LIVE_PARITY_PROFILE,
        symbols=list(SYMBOLS),
        start_date=start_date,
        end_date=end_date,
    )
    plugin = ReplayTrendRound4Plugin(
        bt_config,
        seed_config,
        data_dir=DATA_DIR,
        max_workers=MAX_WORKERS,
    )
    runner = PhaseRunner(plugin, round_dir)
    state_path = round_dir / "phase_state.json"
    state = PhaseState.load_or_create(state_path)

    _build_replay_context(round_num, common_start, common_end, data_ranges, plugin, round_dir)

    log.info(
        "trend.round4_replay.start",
        round=round_num,
        output_dir=str(round_dir),
        symbols=SYMBOLS,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        max_workers=MAX_WORKERS,
    )

    runner.run_all_phases(state)

    final_metrics = None
    if state.phase_metrics:
        last_phase = max(state.phase_metrics)
        final_metrics = state.phase_metrics[last_phase]
    _update_rounds_manifest(OUTPUT_BASE, round_num, state.cumulative_mutations, final_metrics)

    summary = {
        "round": round_num,
        "output_dir": str(round_dir),
        "completed_phases": state.completed_phases,
        "mutations": state.cumulative_mutations,
        "final_metrics": final_metrics,
        "measurement_start": start_date.isoformat(),
        "measurement_end": end_date.isoformat(),
    }
    _atomic_write_json(summary, round_dir / "replay_summary.json")

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
