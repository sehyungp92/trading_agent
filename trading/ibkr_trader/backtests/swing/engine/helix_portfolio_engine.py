"""Multi-symbol Helix portfolio backtesting engine.

Two modes:
- run_helix_independent: Each symbol runs its own HelixEngine (fast, for optimization)
- run_helix_synchronized: All symbols step together with cross-symbol allocation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd

from backtests.shared.parity.legacy_result_outputs import (
    decision_stream_from_trades,
    merge_decision_streams,
    trade_outcomes_from_records,
)
from strategies.swing.akc_helix.config import SYMBOL_CONFIGS, SymbolConfig
from strategies.swing.akc_helix.models import Direction

from backtests.swing.analysis.helix_shadow_tracker import (
    FilterStats,
    HelixShadowTracker,
)
from backtests.swing.config_helix import HelixBacktestConfig
from backtests.swing.data.preprocessing import (
    NumpyBars,
    align_4h_to_hourly,
    align_daily_to_hourly,
    build_numpy_arrays,
    resample_1h_to_4h,
)
from backtests.swing.engine.helix_engine import HelixEngine, HelixSymbolResult
from backtests.swing.engine.helix_engine import _AblationPatch

logger = logging.getLogger(__name__)


@dataclass
class HelixPortfolioData:
    """Pre-loaded data for all symbols including 4H bars."""

    daily: dict[str, NumpyBars] = field(default_factory=dict)
    hourly: dict[str, NumpyBars] = field(default_factory=dict)
    four_hour: dict[str, NumpyBars] = field(default_factory=dict)
    daily_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)
    four_hour_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class HelixHeatStats:
    """Portfolio heat utilization statistics."""

    avg_heat_pct: float = 0.0
    max_heat_pct: float = 0.0
    pct_time_at_limit: float = 0.0


@dataclass
class HelixPortfolioResult:
    """Combined results across all symbols."""

    symbol_results: dict[str, HelixSymbolResult] = field(default_factory=dict)
    combined_equity: np.ndarray = field(default_factory=lambda: np.array([]))
    combined_equity_mtm: np.ndarray = field(default_factory=lambda: np.array([]))
    combined_equity_realized: np.ndarray = field(default_factory=lambda: np.array([]))
    combined_timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    filter_summary: dict[str, FilterStats] = field(default_factory=dict)
    heat_stats: HelixHeatStats = field(default_factory=HelixHeatStats)
    decision_stream: list[dict] = field(default_factory=list)
    trade_outcomes: list[dict] = field(default_factory=list)


def _get_point_value(symbol: str) -> float:
    cfg = SYMBOL_CONFIGS.get(symbol)
    return cfg.multiplier if cfg else 1.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_helix_data(
    symbols: list[str],
    data_dir,
    *,
    start_date=None,
    end_date=None,
) -> HelixPortfolioData:
    """Load 1H + 1D parquets, resample 1H→4H, build NumpyBars + idx maps."""
    from pathlib import Path
    from backtests.swing.data.replay_cache import _coerce_utc_timestamp, _slice_timestamp_index
    data_dir = Path(data_dir)
    portfolio = HelixPortfolioData()
    start_ts = _coerce_utc_timestamp(start_date)
    end_ts = _coerce_utc_timestamp(end_date, end_of_day=True)

    for sym in symbols:
        hourly_path = data_dir / f"{sym}_1h.parquet"
        daily_path = data_dir / f"{sym}_1d.parquet"

        if not hourly_path.exists() or not daily_path.exists():
            logger.warning("Missing data for %s, skipping", sym)
            continue

        # Load DataFrames
        hourly_df = pd.read_parquet(hourly_path)
        daily_df = pd.read_parquet(daily_path)

        # Ensure DatetimeIndex
        if not isinstance(hourly_df.index, pd.DatetimeIndex):
            hourly_df.index = pd.DatetimeIndex(hourly_df.index)
        if not isinstance(daily_df.index, pd.DatetimeIndex):
            daily_df.index = pd.DatetimeIndex(daily_df.index)

        hourly_df = _slice_timestamp_index(hourly_df, start_ts, end_ts)
        daily_df = _slice_timestamp_index(daily_df, start_ts, end_ts)

        # Resample 1H → 4H
        four_hour_df = resample_1h_to_4h(hourly_df)

        # Build NumpyBars
        portfolio.hourly[sym] = build_numpy_arrays(hourly_df)
        portfolio.daily[sym] = build_numpy_arrays(daily_df)
        portfolio.four_hour[sym] = build_numpy_arrays(four_hour_df)

        # Alignment maps
        portfolio.daily_idx_maps[sym] = align_daily_to_hourly(hourly_df, daily_df)
        portfolio.four_hour_idx_maps[sym] = align_4h_to_hourly(hourly_df, four_hour_df)

    return portfolio


# ---------------------------------------------------------------------------
# Independent mode (fast, for optimization)
# ---------------------------------------------------------------------------

def run_helix_independent(
    data: HelixPortfolioData,
    bt_config: HelixBacktestConfig,
) -> HelixPortfolioResult:
    """Run each symbol independently (fast path for optimization)."""
    results: dict[str, HelixSymbolResult] = {}
    engines: dict[str, HelixEngine] = {}
    shadow = HelixShadowTracker() if bt_config.track_shadows else None
    configs: dict[str, SymbolConfig] = {}

    for sym in bt_config.symbols:
        if sym not in data.hourly or sym not in data.daily or sym not in data.four_hour:
            logger.warning("No data for %s, skipping", sym)
            continue

        cfg = SYMBOL_CONFIGS.get(sym)
        if cfg is None:
            continue
        cfg = _apply_overrides(cfg, bt_config.param_overrides)
        configs[sym] = cfg

        engine = HelixEngine(
            symbol=sym, cfg=cfg, bt_config=bt_config,
            point_value=_get_point_value(sym),
        )
        engines[sym] = engine
        results[sym] = engine.run(
            daily=data.daily[sym],
            hourly=data.hourly[sym],
            four_hour=data.four_hour[sym],
            daily_idx_map=data.daily_idx_maps[sym],
            four_hour_idx_map=data.four_hour_idx_maps[sym],
        )

    filter_summary = _run_shadow_sim(shadow, engines, configs, data, bt_config)

    combined_equity, combined_ts = _combine_equity_curves(results, bt_config.initial_equity)
    return HelixPortfolioResult(
        symbol_results=results,
        combined_equity=combined_equity,
        combined_equity_mtm=combined_equity,
        combined_equity_realized=combined_equity,
        combined_timestamps=combined_ts,
        filter_summary=filter_summary,
        decision_stream=merge_decision_streams(*(result.decision_stream for result in results.values())),
        trade_outcomes=[outcome for result in results.values() for outcome in result.trade_outcomes],
    )


# ---------------------------------------------------------------------------
# Synchronized mode (cross-symbol allocation)
# ---------------------------------------------------------------------------

def run_helix_synchronized(
    data: HelixPortfolioData,
    bt_config: HelixBacktestConfig,
) -> HelixPortfolioResult:
    """Run all symbols stepping through time with cross-symbol allocation.

    Steps through a unified hourly timestamp index.  On each bar:
    1. Each symbol updates state and detects setups
    2. Portfolio allocation ranks and filters across symbols
    """
    from strategies.swing.akc_helix import allocator
    from strategies.swing.akc_helix.models import SetupState

    engines: dict[str, HelixEngine] = {}
    configs: dict[str, SymbolConfig] = {}

    for sym in bt_config.symbols:
        if sym not in data.hourly or sym not in data.daily or sym not in data.four_hour:
            continue
        cfg = SYMBOL_CONFIGS.get(sym)
        if cfg is None:
            continue
        cfg = _apply_overrides(cfg, bt_config.param_overrides)
        configs[sym] = cfg
        engines[sym] = HelixEngine(
            symbol=sym, cfg=cfg, bt_config=bt_config,
            point_value=_get_point_value(sym),
        )

    if not engines:
        return HelixPortfolioResult()

    shadow = HelixShadowTracker() if bt_config.track_shadows else None

    # Mock instruments for allocator
    instruments = {
        sym: SimpleNamespace(point_value=_get_point_value(sym))
        for sym in engines
    }

    # Build unified timestamp index
    all_times_set: set = set()
    time_sets: dict[str, dict] = {}
    for sym in engines:
        times = data.hourly[sym].times
        mapping = {}
        for i in range(len(times)):
            key = times[i].item() if hasattr(times[i], 'item') else times[i]
            mapping[key] = i
        time_sets[sym] = mapping
        all_times_set.update(mapping.keys())

    unified_ts = sorted(all_times_set)
    init_eq = bt_config.initial_equity
    prev_sym_equity: dict[str, float] = {sym: init_eq for sym in engines}
    last_sym_mtm: dict[str, float] = {sym: init_eq for sym in engines}
    portfolio_equity = init_eq

    equity_curve_mtm: list[float] = []
    equity_curve_realized: list[float] = []
    timestamps: list = []
    heat_samples: list[float] = []

    from strategies.swing.akc_helix.config import PORTFOLIO_CAP_R

    with _AblationPatch(bt_config.flags, bt_config.param_overrides):
        for sym, engine in engines.items():
            engine._precompute_indicators(data.hourly[sym], data.four_hour[sym])

        for ts in unified_ts:
            for sym, engine in engines.items():
                bar_idx = time_sets[sym].get(ts)
                if bar_idx is None:
                    continue

                engine.sizing_equity = portfolio_equity
                engine._step_bar(
                    data.daily[sym], data.hourly[sym], data.four_hour[sym],
                    data.daily_idx_maps[sym], data.four_hour_idx_maps[sym],
                    bar_idx,
                    bt_config.warmup_daily, bt_config.warmup_hourly,
                )
                if engine.equity_curve:
                    last_sym_mtm[sym] = float(engine.equity_curve[-1])

            # Portfolio realized equity from per-symbol closed-PnL deltas.
            for sym, eng in engines.items():
                delta = eng.equity - prev_sym_equity[sym]
                portfolio_equity += delta
                prev_sym_equity[sym] = eng.equity

            # Add only the child open-PnL deltas to avoid double-counting realized PnL.
            open_mtm = sum(last_sym_mtm[sym] - eng.equity for sym, eng in engines.items())
            portfolio_mtm = portfolio_equity + open_mtm
            equity_curve_realized.append(portfolio_equity)
            equity_curve_mtm.append(portfolio_mtm)
            timestamps.append(ts)

            # Track heat
            total_heat = 0.0
            for sym, eng in engines.items():
                pos = eng.active_position
                if pos is not None and pos.qty_open > 0:
                    basis = getattr(pos, "avg_entry_price", 0.0) or pos.fill_price
                    if pos.setup.direction.name == "LONG":
                        risk_per_share = max(0.0, basis - pos.current_stop)
                    else:
                        risk_per_share = max(0.0, pos.current_stop - basis)
                    risk_dollars = risk_per_share * _get_point_value(sym) * pos.qty_open
                    total_heat += risk_dollars
            heat_base = portfolio_mtm if portfolio_mtm > 0 else portfolio_equity
            heat_pct = total_heat / heat_base if heat_base > 0 else 0.0
            heat_samples.append(heat_pct)

        for sym, engine in engines.items():
            if engine.active_position is None:
                continue
            hourly = data.hourly[sym]
            engine._flatten_at_end_of_data(
                hourly.closes[-1],
                engine._to_datetime(hourly.times[-1]),
            )
            delta = engine.equity - prev_sym_equity[sym]
            portfolio_equity += delta
            prev_sym_equity[sym] = engine.equity
            last_sym_mtm[sym] = engine.equity
        if equity_curve_mtm:
            equity_curve_mtm[-1] = portfolio_equity
            equity_curve_realized[-1] = portfolio_equity

    # Compute heat stats
    heat_arr = np.array(heat_samples) if heat_samples else np.array([0.0])
    heat = HelixHeatStats(
        avg_heat_pct=float(np.mean(heat_arr)),
        max_heat_pct=float(np.max(heat_arr)),
        pct_time_at_limit=float(np.mean(heat_arr >= PORTFOLIO_CAP_R)) * 100,
    )

    # Build per-symbol results
    results: dict[str, HelixSymbolResult] = {}
    for sym, engine in engines.items():
        results[sym] = HelixSymbolResult(
            symbol=sym,
            trades=engine.trades,
            equity_curve=np.array(engine.equity_curve),
            timestamps=np.array(engine.timestamps),
            total_commission=engine.total_commission,
            setups_detected=engine.setups_detected,
            setups_armed=engine.setups_armed,
            setups_filled=engine.setups_filled,
            setups_expired=engine.setups_expired,
            regime_days_bull=engine.regime_days_bull,
            regime_days_bear=engine.regime_days_bear,
            regime_days_chop=engine.regime_days_chop,
            decision_stream=decision_stream_from_trades(engine.trades, timeframe="1h"),
            trade_outcomes=trade_outcomes_from_records(engine.trades),
        )

    filter_summary = _run_shadow_sim(shadow, engines, configs, data, bt_config)

    return HelixPortfolioResult(
        symbol_results=results,
        combined_equity=np.array(equity_curve_mtm),
        combined_equity_mtm=np.array(equity_curve_mtm),
        combined_equity_realized=np.array(equity_curve_realized),
        combined_timestamps=np.array(timestamps),
        filter_summary=filter_summary,
        heat_stats=heat,
        decision_stream=merge_decision_streams(*(result.decision_stream for result in results.values())),
        trade_outcomes=[outcome for result in results.values() for outcome in result.trade_outcomes],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _combine_equity_curves(
    results: dict[str, HelixSymbolResult],
    initial_equity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine per-symbol equity curves into a portfolio curve."""
    if not results:
        return np.array([initial_equity]), np.array([])

    max_len = max(len(r.equity_curve) for r in results.values())
    combined = np.full(max_len, initial_equity, dtype=np.float64)

    for r in results.values():
        n = len(r.equity_curve)
        if n == 0:
            continue
        padded = np.full(max_len, r.equity_curve[-1] if n > 0 else initial_equity)
        padded[:n] = r.equity_curve
        combined += (padded - initial_equity)

    longest_sym = max(results, key=lambda s: len(results[s].timestamps))
    combined_ts = results[longest_sym].timestamps
    return combined, combined_ts


def _run_shadow_sim(
    shadow: HelixShadowTracker | None,
    engines: dict[str, HelixEngine],
    configs: dict[str, SymbolConfig],
    data: HelixPortfolioData,
    bt_config: HelixBacktestConfig,
) -> dict[str, FilterStats]:
    """Run shadow simulation on rejections."""
    if not shadow or not shadow.rejections:
        return {}
    syms = list(engines)
    shadow.simulate_shadows(
        hourly_data={
            s: (data.hourly[s].opens, data.hourly[s].highs,
                data.hourly[s].lows, data.hourly[s].closes,
                data.hourly[s].volumes)
            for s in syms
        },
        hourly_times={s: data.hourly[s].times for s in syms},
        configs=configs,
        point_values={s: _get_point_value(s) for s in syms},
        daily_states={s: engines[s]._daily_state_by_idx for s in syms},
        daily_idx_maps={s: data.daily_idx_maps[s] for s in syms},
    )
    return shadow.get_filter_summary()


def _apply_overrides(cfg: SymbolConfig, overrides: dict[str, float]) -> SymbolConfig:
    """Create a new SymbolConfig with parameter overrides applied."""
    if not overrides:
        return cfg

    changes: dict[str, object] = {}
    for key, value in overrides.items():
        suffix = f"_{cfg.symbol}"
        field_name = key[:-len(suffix)] if key.endswith(suffix) else key
        if hasattr(cfg, field_name):
            current = getattr(cfg, field_name)
            changes[field_name] = int(round(value)) if isinstance(current, int) else float(value)

    if not changes:
        return cfg

    from dataclasses import asdict
    d = asdict(cfg)
    d.update(changes)
    return SymbolConfig(**d)
