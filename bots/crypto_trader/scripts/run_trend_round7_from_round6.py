"""Run canonical trend round 3 from the relabelled round-2 baseline."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog

from crypto_trader.backtest.profiles import LIVE_PARITY_PROFILE, build_backtest_config_from_profile
from crypto_trader.backtest.metrics import metrics_to_dict
from crypto_trader.backtest.runner import run
from crypto_trader.cli import _detect_next_round, _update_rounds_manifest
from crypto_trader.data.store import ParquetStore
from crypto_trader.optimize.phase_runner import PhaseRunner
from crypto_trader.optimize.phase_state import PhaseState, _atomic_write_json
from crypto_trader.optimize.scoring import composite_score
from crypto_trader.optimize.trend_round7_plugin import (
    ROUND7_HARD_REJECTS,
    ROUND7_IMMUTABLE_SCORING_CEILINGS,
    ROUND7_PHASE_CANDIDATES,
    ROUND7_PHASE_GATE_CRITERIA,
    ROUND7_PHASE_NAMES,
    ROUND7_PHASE_SCORING_EMPHASIS,
    ROUND7_SCORING_WEIGHTS,
    Round7TrendPlugin,
)
from crypto_trader.strategy.trend.config import TrendConfig

ROOT = Path(__file__).resolve().parents[1]
ROUND6_CONFIG_PATH = ROOT / "output" / "trend" / "round_2" / "optimized_config.json"
ROUND6_SUMMARY_PATH = ROOT / "output" / "trend" / "round_2" / "round2_summary.json"
DATA_DIR = ROOT / "data"
OUTPUT_BASE = ROOT / "output" / "trend"
SYMBOLS = ["BTC", "ETH", "SOL"]
TIMEFRAMES = ["15m", "1h", "1d"]
MAX_WORKERS = 2
BASELINE_TOLERANCE = 1e-6


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


log = structlog.get_logger("scripts.trend_round3")


def _load_json_strategy(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    strategy = raw.get("strategy", raw)
    if not isinstance(strategy, dict):
        raise TypeError(f"Expected strategy mapping in {path}")
    return strategy


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


def _score_metrics(metrics: dict[str, float]) -> tuple[float, bool, str]:
    return composite_score(
        metrics,
        weights=ROUND7_SCORING_WEIGHTS,
        hard_rejects=ROUND7_HARD_REJECTS,
        ceilings=ROUND7_IMMUTABLE_SCORING_CEILINGS,
    )


def _evaluate_anchor(
    label: str,
    config: TrendConfig,
    *,
    plugin: Round7TrendPlugin,
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


def _assert_baseline_matches(
    expected_metrics: dict[str, float],
    actual_metrics: dict[str, float],
) -> None:
    keys = [
        "net_profit",
        "net_return_pct",
        "total_trades",
        "win_rate",
        "expectancy_r",
        "profit_factor",
        "max_drawdown_pct",
        "sharpe_ratio",
        "calmar_ratio",
        "avg_bars_held",
        "avg_mae_r",
        "avg_mfe_r",
        "exit_efficiency",
    ]
    mismatches: list[str] = []
    for key in keys:
        expected = float(expected_metrics.get(key, 0.0))
        actual = float(actual_metrics.get(key, 0.0))
        if abs(expected - actual) > BASELINE_TOLERANCE:
            mismatches.append(f"{key}: expected {expected:.10f}, got {actual:.10f}")

    if mismatches:
        joined = "\n".join(mismatches)
        raise RuntimeError(
            "Round 3 baseline does not match round 2 final metrics under the new code.\n"
            f"{joined}"
        )


def _build_round7_context(
    *,
    round_num: int,
    round_dir: Path,
    start_dt: datetime,
    end_dt: datetime,
    data_ranges: dict[str, dict[str, str]],
    plugin: Round7TrendPlugin,
    anchor_summary: dict[str, Any],
    round6_summary: dict[str, Any],
) -> None:
    warmup_days = plugin.backtest_config.warmup_days
    warmup_safe_start = start_dt + timedelta(days=warmup_days)
    phase_counts = {
        f"phase_{phase}": len(generator())
        for phase, generator in ROUND7_PHASE_CANDIDATES.items()
    }
    context = {
        "round": round_num,
        "baseline_config": str(ROUND6_CONFIG_PATH),
        "baseline_round": 2,
        "symbols": SYMBOLS,
        "max_workers": plugin.max_workers,
        "data_window_start": start_dt.isoformat(),
        "data_window_end": end_dt.isoformat(),
        "measurement_start": plugin.backtest_config.start_date.isoformat(),
        "measurement_end": plugin.backtest_config.end_date.isoformat(),
        "warmup_days": warmup_days,
        "full_warmup_available": warmup_safe_start.date() <= end_dt.date(),
        "warmup_safe_start_if_available": warmup_safe_start.date().isoformat(),
        "scoring_weights": ROUND7_SCORING_WEIGHTS,
        "phase_scoring_emphasis": ROUND7_PHASE_SCORING_EMPHASIS,
        "immutable_scoring_ceilings": ROUND7_IMMUTABLE_SCORING_CEILINGS,
        "hard_rejects": ROUND7_HARD_REJECTS,
        "phase_gate_criteria": {
            str(phase): [criterion.__dict__ for criterion in criteria]
            for phase, criteria in ROUND7_PHASE_GATE_CRITERIA.items()
        },
        "phase_names": ROUND7_PHASE_NAMES,
        "phase_candidate_counts": phase_counts,
        "baseline_anchor": anchor_summary,
        "round2_summary": round6_summary,
        "data_ranges": data_ranges,
    }
    _atomic_write_json(context, round_dir / "round3_context.json")


def main() -> None:
    _configure_logging()

    if not ROUND6_CONFIG_PATH.exists() or not ROUND6_SUMMARY_PATH.exists():
        raise RuntimeError("Round 2 artifacts are required before running round 3.")

    round_num = _detect_next_round(OUTPUT_BASE)
    if round_num != 3:
        raise RuntimeError(
            f"Expected next trend round to be 3, but detected round_{round_num} in {OUTPUT_BASE}."
        )

    round_dir = OUTPUT_BASE / f"round_{round_num}"
    round_dir.mkdir(parents=True, exist_ok=True)

    baseline_config = TrendConfig.from_dict(_load_json_strategy(ROUND6_CONFIG_PATH))
    round6_summary = json.loads(ROUND6_SUMMARY_PATH.read_text(encoding="utf-8"))

    common_start, common_end, data_ranges = _compute_common_window(DATA_DIR, SYMBOLS)
    start_date = common_start.date()
    end_date = common_end.date()

    bt_config = build_backtest_config_from_profile(
        profile=LIVE_PARITY_PROFILE,
        symbols=list(SYMBOLS),
        start_date=start_date,
        end_date=end_date,
    )
    plugin = Round7TrendPlugin(
        bt_config,
        baseline_config,
        data_dir=DATA_DIR,
        max_workers=MAX_WORKERS,
    )
    runner = PhaseRunner(plugin, round_dir, round_name="trend_round3_trade_recovery")
    state_path = round_dir / "phase_state.json"
    state = PhaseState.load_or_create(state_path)

    baseline_anchor = _evaluate_anchor("round_2_baseline", baseline_config, plugin=plugin)
    _assert_baseline_matches(round6_summary["final_metrics"], baseline_anchor["metrics"])

    _build_round7_context(
        round_num=round_num,
        round_dir=round_dir,
        start_dt=common_start,
        end_dt=common_end,
        data_ranges=data_ranges,
        plugin=plugin,
        anchor_summary=baseline_anchor,
        round6_summary=round6_summary,
    )

    log.info(
        "trend.round3.start",
        round=round_num,
        output_dir=str(round_dir),
        symbols=SYMBOLS,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        max_workers=MAX_WORKERS,
        baseline_score=baseline_anchor["score"],
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
        "baseline_score": baseline_anchor["score"],
        "baseline_metrics": baseline_anchor["metrics"],
        "mutations": state.cumulative_mutations,
        "final_metrics": final_metrics,
        "measurement_start": start_date.isoformat(),
        "measurement_end": end_date.isoformat(),
    }
    _atomic_write_json(summary, round_dir / "round3_summary.json")

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
