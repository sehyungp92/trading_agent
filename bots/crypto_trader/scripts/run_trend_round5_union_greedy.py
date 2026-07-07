"""Run trend round 5 as a single-phase union-of-accepted-mutations greedy search.

Round 5 starts from the reconstructed pre-round-1 seed and only tests parameter
values that appear in either:
- output/trend/round_4_trend/optimized_config.json
- output/trend/round_4/optimized_config.json

The search is intentionally configured as one composite greedy phase:
- all union candidates live in the same pool
- pruning is disabled so synergistic candidates stay alive for later rounds
- score ceilings are immutable and sized off the current max common BTC/ETH/SOL
  replay window
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog
import yaml

from crypto_trader.backtest.profiles import LIVE_PARITY_PROFILE, build_backtest_config_from_profile
from crypto_trader.backtest.diagnostics import generate_diagnostics
from crypto_trader.backtest.metrics import metrics_to_dict
from crypto_trader.backtest.runner import run
from crypto_trader.cli import _detect_next_round, _update_rounds_manifest
from crypto_trader.data.store import ParquetStore
from crypto_trader.optimize.parallel import evaluate_parallel
from crypto_trader.optimize.phase_runner import PhaseRunner
from crypto_trader.optimize.phase_state import PhaseState, _atomic_write_json
from crypto_trader.optimize.scoring import composite_score
from crypto_trader.optimize.trend_plugin import HARD_REJECTS, TrendPlugin
from crypto_trader.optimize.types import (
    EvaluateFn,
    Experiment,
    GateCriterion,
    GreedyResult,
    PhaseAnalysisPolicy,
    PhaseSpec,
    ScoredCandidate,
)
from crypto_trader.strategy.trend.config import TrendConfig

ROOT = Path(__file__).resolve().parents[1]
SEED_CONFIG_PATH = ROOT / "config" / "trend_pre_round1.yaml"
PREVIOUS_ROUND4_CONFIG_PATH = ROOT / "output" / "trend" / "round_4_trend" / "optimized_config.json"
CURRENT_ROUND4_CONFIG_PATH = ROOT / "output" / "trend" / "round_4" / "optimized_config.json"
DATA_DIR = ROOT / "data"
OUTPUT_BASE = ROOT / "output" / "trend"
SYMBOLS = ["BTC", "ETH", "SOL"]
TIMEFRAMES = ["15m", "1h", "1d"]
MAX_WORKERS = 2

ROUND5_SCORING_WEIGHTS: dict[str, float] = {
    "returns": 0.28,
    "calmar": 0.18,
    "sharpe": 0.16,
    "coverage": 0.12,
    "risk": 0.12,
    "capture": 0.08,
    "edge": 0.06,
}

ROUND5_IMMUTABLE_SCORING_CEILINGS: dict[str, float] = {
    "returns": 25.0,
    "coverage": 100.0,
    "edge": 2.0,
    "calmar": 3.0,
    "sharpe": 4.0,
    "risk": 20.0,
}

ROUND5_GATE_CRITERIA: list[GateCriterion] = [
    GateCriterion(metric="total_trades", operator=">=", threshold=50.0),
    GateCriterion(metric="net_return_pct", operator=">=", threshold=10.0),
    GateCriterion(metric="profit_factor", operator=">=", threshold=1.2),
    GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=15.0),
    GateCriterion(metric="sharpe_ratio", operator=">=", threshold=2.5),
    GateCriterion(metric="calmar_ratio", operator=">=", threshold=1.2),
]

ROUND5_MIN_DELTA = 0.001


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


log = structlog.get_logger("scripts.trend_round5_union")


def _load_yaml_strategy(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    strategy = raw.get("strategy", raw)
    if not isinstance(strategy, dict):
        raise TypeError(f"Expected strategy mapping in {path}")
    return strategy


def _load_json_strategy(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    strategy = raw.get("strategy", raw)
    if not isinstance(strategy, dict):
        raise TypeError(f"Expected strategy mapping in {path}")
    return strategy


def _load_seed_config(path: Path) -> TrendConfig:
    return TrendConfig.from_dict(_load_yaml_strategy(path))


def _flatten_mapping(obj: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(obj, dict):
        result: dict[str, Any] = {}
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            result.update(_flatten_mapping(value, child_prefix))
        return result
    return {prefix: obj}


def _slugify_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value).replace("-", "neg_")
    if isinstance(value, float):
        text = f"{value:.10g}"
        return (
            text.replace("-", "neg_")
            .replace(".", "_")
            .replace("+", "")
        )
    return str(value).lower().replace("-", "_").replace(".", "_").replace(" ", "_")


def _candidate_name(path: str, value: Any) -> str:
    return f"set__{path.replace('.', '__')}__{_slugify_value(value)}"


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
            detail[f"{symbol}_{timeframe}"] = {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
            }
            common_start = start_dt if common_start is None else max(common_start, start_dt)
            common_end = end_dt if common_end is None else min(common_end, end_dt)

        funding_df = store.load_funding(symbol)
        if funding_df is None or funding_df.empty:
            raise RuntimeError(f"Missing funding data for {symbol}")
        start_dt, end_dt = _time_bounds_from_ts(
            int(funding_df["ts"].min()),
            int(funding_df["ts"].max()),
        )
        detail[f"{symbol}_funding"] = {
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
        }
        common_start = start_dt if common_start is None else max(common_start, start_dt)
        common_end = end_dt if common_end is None else min(common_end, end_dt)

    if common_start is None or common_end is None or common_start > common_end:
        raise RuntimeError("Unable to derive a common BTC/ETH/SOL replay window.")

    return common_start, common_end, detail


def _build_union_candidates(
    seed_strategy: dict[str, Any],
    previous_round4_strategy: dict[str, Any],
    current_round4_strategy: dict[str, Any],
) -> tuple[list[Experiment], list[dict[str, Any]]]:
    seed_flat = _flatten_mapping(seed_strategy)
    previous_flat = _flatten_mapping(previous_round4_strategy)
    current_flat = _flatten_mapping(current_round4_strategy)

    previous_final = {
        path: previous_flat[path]
        for path, seed_value in seed_flat.items()
        if previous_flat.get(path) != seed_value
    }
    current_final = {
        path: current_flat[path]
        for path, seed_value in seed_flat.items()
        if current_flat.get(path) != seed_value
    }

    shared = {
        path: previous_final[path]
        for path in previous_final
        if path in current_final and previous_final[path] == current_final[path]
    }
    previous_unique = {
        path: value
        for path, value in previous_final.items()
        if current_final.get(path) != value
    }
    current_unique = {
        path: value
        for path, value in current_final.items()
        if previous_final.get(path) != value
    }

    experiments: list[Experiment] = []
    metadata: list[dict[str, Any]] = []
    seen_payloads: set[str] = set()

    def add_candidate(
        name: str,
        mutations: dict[str, Any],
        *,
        kind: str,
        sources: list[str],
        rationale: str,
    ) -> None:
        if not mutations:
            return
        payload = json.dumps({"name": name, "mutations": mutations}, sort_keys=True, default=str)
        if payload in seen_payloads:
            return
        seen_payloads.add(payload)
        ordered_mutations = dict(sorted(mutations.items()))
        experiments.append(Experiment(name, ordered_mutations))
        metadata.append(
            {
                "name": name,
                "kind": kind,
                "sources": sources,
                "rationale": rationale,
                "mutations": ordered_mutations,
                "baseline_values": {
                    path: seed_flat[path] for path in ordered_mutations
                },
            }
        )

    add_candidate(
        "anchor_round_4_trend",
        previous_final,
        kind="anchor",
        sources=["round_4_trend"],
        rationale="Full previous round-4 cumulative endpoint from the pre-round-1 seed.",
    )
    add_candidate(
        "anchor_round_4",
        current_final,
        kind="anchor",
        sources=["round_4"],
        rationale="Full current round-4 cumulative endpoint from the pre-round-1 seed.",
    )
    add_candidate(
        "shared_h1_regime_profile",
        {
            path: shared[path]
            for path in ["regime.h1_regime_enabled", "regime.h1_min_adx"]
            if path in shared
        },
        kind="shared_bundle",
        sources=["round_4_trend", "round_4"],
        rationale="Keep the H1 regime toggle and its ADX threshold coupled.",
    )
    add_candidate(
        "previous_delta_from_round_4",
        previous_unique,
        kind="delta_bundle",
        sources=["round_4_trend"],
        rationale="All previous round-4 settings that differ from the current round-4 endpoint.",
    )
    add_candidate(
        "current_delta_from_round_4_trend",
        current_unique,
        kind="delta_bundle",
        sources=["round_4"],
        rationale="All current round-4 settings that differ from the previous round-4 endpoint.",
    )

    previous_profiles = {
        "previous_setup_profile": [
            "setup.impulse_min_atr_move",
            "setup.min_room_r",
            "setup.require_orderly_pullback",
        ],
        "previous_exit_profile": [
            "exits.time_stop_action",
            "exits.time_stop_bars",
            "exits.tp2_frac",
            "exits.tp2_r",
        ],
        "previous_regime_profile": [
            "regime.b_adx_rising_required",
            "regime.b_min_adx",
        ],
        "previous_risk_profile": [
            "risk.risk_pct_a",
            "risk.risk_pct_b",
        ],
        "previous_stop_profile": [
            "stops.atr_mult",
            "stops.min_stop_atr",
        ],
        "previous_h1_fast_profile": [
            "h1_indicators.ema_fast",
        ],
        "previous_trail_profile": [
            "trail.trail_buffer_tight",
        ],
    }
    current_profiles = {
        "current_signal_profile": [
            "setup.impulse_min_atr_move",
        ],
        "current_h1_fast_profile": [
            "h1_indicators.ema_fast",
        ],
        "current_trail_profile": [
            "trail.trail_buffer_tight",
            "trail.trail_buffer_wide",
            "trail.trail_r_ceiling",
        ],
    }

    for name, keys in previous_profiles.items():
        add_candidate(
            name,
            {path: previous_unique[path] for path in keys if path in previous_unique},
            kind="profile_bundle",
            sources=["round_4_trend"],
            rationale="Dependency-aware bundle from the previous round-4 endpoint.",
        )
    for name, keys in current_profiles.items():
        add_candidate(
            name,
            {path: current_unique[path] for path in keys if path in current_unique},
            kind="profile_bundle",
            sources=["round_4"],
            rationale="Dependency-aware bundle from the current round-4 endpoint.",
        )

    return experiments, metadata


class UnionTrendRound5Plugin(TrendPlugin):
    """Single-phase trend plugin for the round-5 union greedy search."""

    def __init__(
        self,
        backtest_config: BacktestConfig,
        base_config: TrendConfig,
        *,
        candidates: list[Experiment],
        data_dir: Path = DATA_DIR,
        max_workers: int | None = None,
    ) -> None:
        super().__init__(backtest_config, base_config, data_dir=data_dir, max_workers=max_workers)
        self._candidates = candidates

    @property
    def num_phases(self) -> int:
        return 1

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "net_return_pct": 18.0,
            "total_trades": 75.0,
            "profit_factor": 1.35,
            "max_drawdown_pct": 12.0,
            "sharpe_ratio": 3.0,
            "calmar_ratio": 1.8,
        }

    def get_phase_spec(self, phase: int, state: Any) -> PhaseSpec:
        if phase != 1:
            raise ValueError(f"Unknown phase: {phase}")
        return PhaseSpec(
            phase_num=1,
            name="Round 4 Union Greedy",
            candidates=list(self._candidates),
            scoring_weights=dict(ROUND5_SCORING_WEIGHTS),
            hard_rejects=dict(HARD_REJECTS),
            gate_criteria=list(ROUND5_GATE_CRITERIA),
            analysis_policy=PhaseAnalysisPolicy(
                max_scoring_retries=0,
                max_diagnostic_retries=0,
                focus_metrics=["net_return_pct", "calmar_ratio", "sharpe_ratio", "max_drawdown_pct"],
            ),
            min_delta=ROUND5_MIN_DELTA,
            focus="Round 4 Union Greedy",
            max_rounds=len(self._candidates),
            prune_threshold=0.0,
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        scoring_weights: dict[str, float],
        hard_rejects: dict[str, tuple[str, float]],
    ) -> EvaluateFn:
        ceilings = dict(ROUND5_IMMUTABLE_SCORING_CEILINGS)

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

    def run_phase_diagnostics(
        self,
        phase: int,
        state: Any,
        metrics: dict[str, float],
        greedy_result: GreedyResult,
    ) -> str:
        return self.run_enhanced_diagnostics(phase, state, metrics, greedy_result)

    def run_enhanced_diagnostics(
        self,
        phase: int,
        state: Any,
        metrics: dict[str, float],
        greedy_result: GreedyResult,
    ) -> str:
        if self._last_result is None:
            mutations = state.cumulative_mutations if state else {}
            self.compute_final_metrics(mutations)

        result = self._last_result
        trades = result.trades if result is not None else []
        terminal_marks = result.terminal_marks if result is not None else []
        if not result or (not trades and not terminal_marks):
            return "No trades to diagnose."

        try:
            from crypto_trader.backtest.diagnostics import generate_phase_diagnostics

            return generate_phase_diagnostics(
                trades,
                ["D1", "D2", "D3", "D4", "D5", "D6"],
                initial_equity=float(self.backtest_config.initial_equity),
                title="Round 5 Union Greedy",
                terminal_marks=terminal_marks,
            )
        except Exception:
            return generate_diagnostics(
                trades,
                initial_equity=float(self.backtest_config.initial_equity),
                terminal_marks=terminal_marks,
            )


def _score_metrics(metrics: dict[str, float]) -> tuple[float, bool, str]:
    return composite_score(
        metrics,
        weights=ROUND5_SCORING_WEIGHTS,
        hard_rejects=HARD_REJECTS,
        ceilings=ROUND5_IMMUTABLE_SCORING_CEILINGS,
    )


def _evaluate_anchor(
    label: str,
    config: TrendConfig,
    *,
    plugin: UnionTrendRound5Plugin,
) -> dict[str, Any]:
    result = run(
        config,
        plugin.backtest_config,
        plugin.data_dir,
        strategy_type="trend",
        store=plugin._get_store(),
    )
    metrics = metrics_to_dict(result.metrics)
    score, rejected, reject_reason = _score_metrics(metrics)
    return {
        "label": label,
        "score": score,
        "rejected": rejected,
        "reject_reason": reject_reason,
        "metrics": metrics,
    }


def _build_round5_context(
    *,
    round_num: int,
    round_dir: Path,
    start_dt: datetime,
    end_dt: datetime,
    data_ranges: dict[str, dict[str, str]],
    plugin: UnionTrendRound5Plugin,
    candidate_metadata: list[dict[str, Any]],
    anchor_summaries: list[dict[str, Any]],
) -> None:
    warmup_days = plugin.backtest_config.warmup_days
    warmup_safe_start = start_dt + timedelta(days=warmup_days)
    context = {
        "round": round_num,
        "round_type": "single_phase_union_greedy",
        "baseline_config": str(SEED_CONFIG_PATH),
        "source_configs": {
            "round_4_trend": str(PREVIOUS_ROUND4_CONFIG_PATH),
            "round_4": str(CURRENT_ROUND4_CONFIG_PATH),
        },
        "symbols": SYMBOLS,
        "max_workers": plugin.max_workers,
        "data_window_start": start_dt.isoformat(),
        "data_window_end": end_dt.isoformat(),
        "measurement_start": plugin.backtest_config.start_date.isoformat(),
        "measurement_end": plugin.backtest_config.end_date.isoformat(),
        "warmup_days": warmup_days,
        "full_warmup_available": warmup_safe_start.date() <= end_dt.date(),
        "warmup_safe_start_if_available": warmup_safe_start.date().isoformat(),
        "scoring_weights": ROUND5_SCORING_WEIGHTS,
        "immutable_scoring_ceilings": ROUND5_IMMUTABLE_SCORING_CEILINGS,
        "gate_criteria": [criterion.__dict__ for criterion in ROUND5_GATE_CRITERIA],
        "min_delta": ROUND5_MIN_DELTA,
        "prune_threshold": 0.0,
        "candidate_count": len(candidate_metadata),
        "candidates": candidate_metadata,
        "anchor_summaries": anchor_summaries,
        "data_ranges": data_ranges,
    }
    _atomic_write_json(context, round_dir / "round5_context.json")


def main() -> None:
    _configure_logging()

    round_num = _detect_next_round(OUTPUT_BASE)
    if round_num != 5:
        raise RuntimeError(
            f"Expected next trend round to be 5, but detected round_{round_num} in {OUTPUT_BASE}."
        )

    round_dir = OUTPUT_BASE / f"round_{round_num}"
    round_dir.mkdir(parents=True, exist_ok=True)

    seed_strategy = _load_yaml_strategy(SEED_CONFIG_PATH)
    previous_round4_strategy = _load_json_strategy(PREVIOUS_ROUND4_CONFIG_PATH)
    current_round4_strategy = _load_json_strategy(CURRENT_ROUND4_CONFIG_PATH)
    seed_config = TrendConfig.from_dict(seed_strategy)
    previous_round4_config = TrendConfig.from_dict(previous_round4_strategy)
    current_round4_config = TrendConfig.from_dict(current_round4_strategy)

    candidates, candidate_metadata = _build_union_candidates(
        seed_strategy,
        previous_round4_strategy,
        current_round4_strategy,
    )

    common_start, common_end, data_ranges = _compute_common_window(DATA_DIR, SYMBOLS)
    start_date = common_start.date()
    end_date = common_end.date()

    bt_config = build_backtest_config_from_profile(
        profile=LIVE_PARITY_PROFILE,
        symbols=list(SYMBOLS),
        start_date=start_date,
        end_date=end_date,
    )
    plugin = UnionTrendRound5Plugin(
        bt_config,
        seed_config,
        candidates=candidates,
        data_dir=DATA_DIR,
        max_workers=MAX_WORKERS,
    )
    runner = PhaseRunner(plugin, round_dir, round_name="trend_round5_union")
    state_path = round_dir / "phase_state.json"
    state = PhaseState.load_or_create(state_path)

    anchor_summaries = [
        _evaluate_anchor("pre_round1", seed_config, plugin=plugin),
        _evaluate_anchor("round_4_trend", previous_round4_config, plugin=plugin),
        _evaluate_anchor("round_4", current_round4_config, plugin=plugin),
    ]
    _build_round5_context(
        round_num=round_num,
        round_dir=round_dir,
        start_dt=common_start,
        end_dt=common_end,
        data_ranges=data_ranges,
        plugin=plugin,
        candidate_metadata=candidate_metadata,
        anchor_summaries=anchor_summaries,
    )

    log.info(
        "trend.round5_union.start",
        round=round_num,
        output_dir=str(round_dir),
        symbols=SYMBOLS,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        candidate_count=len(candidates),
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
        "candidate_count": len(candidates),
        "accepted_experiments": state.phase_results.get(1, {}).get("kept_features", []),
        "mutations": state.cumulative_mutations,
        "final_metrics": final_metrics,
        "measurement_start": start_date.isoformat(),
        "measurement_end": end_date.isoformat(),
        "anchor_scores": {
            item["label"]: {
                "score": item["score"],
                "rejected": item["rejected"],
                "metrics": item["metrics"],
            }
            for item in anchor_summaries
        },
    }
    _atomic_write_json(summary, round_dir / "round5_summary.json")

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
