"""Multi-symbol portfolio backtesting engine.

Two modes:
- run_independent: Each symbol runs its own BacktestEngine (fast, for optimization)
- run_synchronized: All symbols step together with portfolio allocation (accurate)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from backtests.shared.parity.legacy_result_outputs import (
    decision_stream_from_trades,
    merge_decision_streams,
    trade_outcomes_from_records,
)
from strategies.swing.atrss import allocator
from strategies.swing.atrss.config import ALL_SYMBOL_CONFIGS, SYMBOL_CONFIGS, SymbolConfig
from strategies.swing.atrss.models import Candidate, CandidateType, Direction, PositionBook, Regime

from backtests.swing.analysis.shadow_tracker import FilterStats, ShadowTracker
from backtests.swing.config import BacktestConfig
from backtests.swing.data.preprocessing import NumpyBars
from backtests.swing.engine.backtest_engine import BacktestEngine, SymbolResult, _AblationPatch

logger = logging.getLogger(__name__)


def _timestamp_key(value):
    """Return a stable historical timestamp key without losing datetime units."""
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[ns]")
    return value.item() if hasattr(value, "item") else value


@dataclass
class PortfolioData:
    """Pre-loaded data for all symbols."""

    daily: dict[str, NumpyBars] = field(default_factory=dict)
    hourly: dict[str, NumpyBars] = field(default_factory=dict)
    daily_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class HeatStats:
    """Portfolio heat utilization statistics."""

    avg_heat_pct: float = 0.0
    max_heat_pct: float = 0.0
    pct_time_at_limit: float = 0.0  # % of bars where heat >= MAX_PORTFOLIO_HEAT


@dataclass
class PortfolioResult:
    """Combined results across all symbols."""

    symbol_results: dict[str, SymbolResult] = field(default_factory=dict)
    combined_equity: np.ndarray = field(default_factory=lambda: np.array([]))
    combined_timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    filter_summary: dict[str, FilterStats] = field(default_factory=dict)
    heat_stats: HeatStats = field(default_factory=HeatStats)
    decision_stream: list[dict] = field(default_factory=list)
    trade_outcomes: list[dict] = field(default_factory=list)


def _get_point_value(symbol: str) -> float:
    cfg = ALL_SYMBOL_CONFIGS.get(symbol) or SYMBOL_CONFIGS.get(symbol)
    return cfg.multiplier if cfg else 1.0


def _portfolio_mtm_equity(engines: dict[str, BacktestEngine], realized_equity: float) -> float:
    """Return shared realized equity plus each engine's latest open-position MTM."""
    open_unrealized = 0.0
    for engine in engines.values():
        if len(engine.equity_curve) == 0:
            continue
        open_unrealized += float(engine.equity_curve[-1]) - float(engine.equity)
    return realized_equity + open_unrealized


def _candidate_qty_for_equity(
    candidate: Candidate,
    daily_state,
    cfg: SymbolConfig,
    bt_config: BacktestConfig,
    equity: float,
    point_value: float,
    position: PositionBook | None = None,
) -> int:
    """Recompute base-entry quantity from the current shared equity ledger."""
    if candidate.type == CandidateType.ADDON_A:
        return int(candidate.qty)

    import strategies.swing.atrss.config as scfg

    if candidate.type == CandidateType.ADDON_B:
        base = position.base_leg if position is not None else None
        if base is None:
            return int(candidate.qty)
        if bt_config.fixed_qty is not None:
            qty = max(1, int(base.qty * scfg.ADDON_B_SIZE_MULT))
        else:
            qty = allocator.compute_position_size(
                candidate.trigger_price,
                candidate.initial_stop,
                equity,
                cfg.base_risk_pct,
                point_value,
            )
        return min(qty, max(1, int(base.qty * scfg.ADDON_B_SIZE_MULT)))

    if bt_config.fixed_qty is not None:
        qty = int(bt_config.fixed_qty)
        if scfg.FIXED_QTY_REGIME_SCALING_ENABLED:
            if daily_state.regime == Regime.STRONG_TREND and daily_state.score >= 60:
                qty = max(1, int(round(qty * scfg.FIXED_QTY_STRONG_TREND_MULT)))
            elif daily_state.regime == Regime.TREND and daily_state.score < 45:
                qty = max(1, int(round(qty * scfg.FIXED_QTY_WEAK_TREND_MULT)))
    else:
        risk_pct = cfg.base_risk_pct
        if daily_state.regime == Regime.STRONG_TREND and daily_state.score >= 60:
            risk_pct *= scfg.DYNAMIC_RISK_STRONG_TREND_MULT
        elif daily_state.regime == Regime.TREND and daily_state.score < 45:
            risk_pct *= scfg.DYNAMIC_RISK_WEAK_TREND_MULT
        qty = allocator.compute_position_size(
            candidate.trigger_price,
            candidate.initial_stop,
            equity,
            risk_pct,
            point_value,
        )

    if cfg.size_reduction_months and candidate.time is not None:
        for month, frac in cfg.size_reduction_months:
            if candidate.time.month == month:
                qty = max(1, int(qty * frac))
                break
    return int(qty)


def run_independent(
    data: PortfolioData,
    bt_config: BacktestConfig,
) -> PortfolioResult:
    """Run each symbol independently (fast path for optimization)."""
    results: dict[str, SymbolResult] = {}
    engines: dict[str, BacktestEngine] = {}
    shadow = ShadowTracker() if bt_config.track_shadows else None
    configs: dict[str, SymbolConfig] = {}

    for sym in bt_config.symbols:
        if sym not in data.hourly or sym not in data.daily:
            logger.warning("No data for %s, skipping", sym)
            continue

        cfg = ALL_SYMBOL_CONFIGS.get(sym) or SYMBOL_CONFIGS.get(sym)
        if cfg is None:
            continue
        cfg = _apply_overrides(cfg, bt_config.param_overrides)
        configs[sym] = cfg

        engine = BacktestEngine(
            symbol=sym, cfg=cfg, bt_config=bt_config,
            point_value=_get_point_value(sym),
        )
        if shadow:
            engine.on_rejection = shadow.record_rejection
        engines[sym] = engine
        results[sym] = engine.run(
            daily=data.daily[sym],
            hourly=data.hourly[sym],
            daily_idx_map=data.daily_idx_maps[sym],
        )

    filter_summary = _run_shadow_sim(shadow, engines, configs, data, bt_config)

    combined_equity, combined_ts = _combine_equity_curves(results, bt_config.initial_equity)
    return PortfolioResult(
        symbol_results=results,
        combined_equity=combined_equity,
        combined_timestamps=combined_ts,
        filter_summary=filter_summary,
        decision_stream=merge_decision_streams(*(result.decision_stream for result in results.values())),
        trade_outcomes=[outcome for result in results.values() for outcome in result.trade_outcomes],
    )


def run_synchronized(
    data: PortfolioData,
    bt_config: BacktestConfig,
    indicator_cache: dict | None = None,
) -> PortfolioResult:
    """Run all symbols stepping through time with cross-symbol allocation.

    Steps through a unified hourly timestamp index. On each bar:
    1. Each symbol updates state, processes fills, manages positions
    2. Candidates from all symbols are collected
    3. allocator.allocate() ranks and filters with portfolio heat caps
    4. Only accepted candidates are submitted

    Args:
        indicator_cache: Optional shared dict for caching indicator states
            across optimization runs. Safe only when indicator-affecting
            params (EMA/ATR periods, ADX thresholds) are identical across
            candidates. Caller must clear when those params change.
    """
    engines: dict[str, BacktestEngine] = {}
    configs: dict[str, SymbolConfig] = {}

    for sym in bt_config.symbols:
        if sym not in data.hourly or sym not in data.daily:
            continue
        cfg = ALL_SYMBOL_CONFIGS.get(sym) or SYMBOL_CONFIGS.get(sym)
        if cfg is None:
            continue
        cfg = _apply_overrides(cfg, bt_config.param_overrides)
        configs[sym] = cfg
        engines[sym] = BacktestEngine(
            symbol=sym, cfg=cfg, bt_config=bt_config,
            point_value=_get_point_value(sym),
            indicator_cache=indicator_cache,
        )

    if not engines:
        return PortfolioResult()

    shadow = ShadowTracker() if bt_config.track_shadows else None
    if shadow:
        for eng in engines.values():
            eng.on_rejection = shadow.record_rejection

    # Mock instruments for allocator (needs .point_value)
    point_values = {sym: _get_point_value(sym) for sym in engines}
    instruments = {sym: SimpleNamespace(point_value=pv) for sym, pv in point_values.items()}

    # Build unified timestamp index from all symbols
    time_sets: dict[str, dict] = {}
    all_times_set: set = set()
    for sym in engines:
        times = data.hourly[sym].times
        mapping = {}
        for i in range(len(times)):
            key = _timestamp_key(times[i])
            mapping[key] = i
        time_sets[sym] = mapping
        all_times_set.update(mapping.keys())

    unified_ts = sorted(all_times_set)
    warmup_d = bt_config.warmup_daily
    warmup_h = bt_config.warmup_hourly
    init_eq = bt_config.initial_equity
    prev_sym_equity: dict[str, float] = {sym: init_eq for sym in engines}
    portfolio_equity = init_eq

    equity_curve: list[float] = []
    timestamps: list = []
    heat_samples: list[float] = []

    import strategies.swing.atrss.config as scfg

    with _AblationPatch(bt_config.flags, bt_config.param_overrides):
        heat_limit = scfg.MAX_PORTFOLIO_HEAT
        for t in unified_ts:
            all_candidates = []

            for sym, engine in engines.items():
                bar_idx = time_sets[sym].get(t)
                if bar_idx is None:
                    continue

                engine.sizing_equity = portfolio_equity

                candidates = engine.step_bar(
                    data.daily[sym], data.hourly[sym],
                    data.daily_idx_maps[sym], bar_idx,
                    warmup_d, warmup_h,
                )
                all_candidates.extend(candidates)

            # Apply realized P&L from fills before allocating/submitting any
            # new deferred entries generated on this timestamp.
            for sym, eng in engines.items():
                delta = eng.equity - prev_sym_equity[sym]
                portfolio_equity += delta
                prev_sym_equity[sym] = eng.equity

            if all_candidates:
                positions = {
                    sym: eng.position for sym, eng in engines.items()
                    if eng.position.direction != Direction.FLAT
                }
                daily_states = {
                    sym: eng.daily_state for sym, eng in engines.items()
                    if eng.daily_state is not None
                }
                hourly_states = {
                    sym: eng.hourly_state for sym, eng in engines.items()
                    if eng.hourly_state is not None
                }

                resized_candidates = []
                for cand in all_candidates:
                    daily_state = daily_states.get(cand.symbol)
                    cfg = configs.get(cand.symbol)
                    point_value = point_values.get(cand.symbol)
                    if daily_state is None or cfg is None or point_value is None:
                        continue
                    cand.qty = _candidate_qty_for_equity(
                        cand,
                        daily_state,
                        cfg,
                        bt_config,
                        portfolio_equity,
                        point_value,
                        positions.get(cand.symbol),
                    )
                    if cand.qty > 0:
                        resized_candidates.append(cand)
                    else:
                        engines[cand.symbol]._funnel.rejected_sizing += 1

                accepted = allocator.allocate(
                    resized_candidates, positions, daily_states,
                    portfolio_equity, instruments, hourly_states,
                )

                bar_time = engines[next(iter(engines))]._to_datetime(
                    t if not hasattr(t, 'item') else t
                )
                for cand in accepted:
                    engines[cand.symbol].submit_candidate(cand, bar_time)

            equity_curve.append(_portfolio_mtm_equity(engines, portfolio_equity))
            timestamps.append(t)

            # Track portfolio heat utilization
            total_heat = 0.0
            for sym, eng in engines.items():
                pos = eng.position
                if pos.direction != Direction.FLAT and pos.base_leg is not None:
                    risk_dollars = abs(pos.base_leg.entry_price - pos.current_stop) * point_values[sym] * pos.total_qty
                    total_heat += risk_dollars
            heat_pct = total_heat / portfolio_equity if portfolio_equity > 0 else 0.0
            heat_samples.append(heat_pct)

    # Compute heat utilization stats
    heat_arr = np.array(heat_samples) if heat_samples else np.array([0.0])
    heat = HeatStats(
        avg_heat_pct=float(np.mean(heat_arr)),
        max_heat_pct=float(np.max(heat_arr)),
        pct_time_at_limit=float(np.mean(heat_arr >= heat_limit)) * 100,
    )

    # Build per-symbol results
    results: dict[str, SymbolResult] = {}
    for sym, engine in engines.items():
        results[sym] = SymbolResult(
            symbol=sym,
            trades=engine.trades,
            equity_curve=np.array(engine.equity_curve),
            timestamps=np.array(engine.timestamps),
            total_commission=engine.total_commission,
            bias_days_long=engine._bias_days_long,
            bias_days_short=engine._bias_days_short,
            bias_days_flat=engine._bias_days_flat,
            funnel=engine._funnel,
            order_metadata=engine._order_metadata,
            decision_stream=decision_stream_from_trades(engine.trades, timeframe="1h"),
            trade_outcomes=trade_outcomes_from_records(engine.trades),
        )

    filter_summary = _run_shadow_sim(shadow, engines, configs, data, bt_config)

    return PortfolioResult(
        symbol_results=results,
        combined_equity=np.array(equity_curve),
        combined_timestamps=np.array(timestamps),
        filter_summary=filter_summary,
        heat_stats=heat,
        decision_stream=merge_decision_streams(*(result.decision_stream for result in results.values())),
        trade_outcomes=[outcome for result in results.values() for outcome in result.trade_outcomes],
    )


def _combine_equity_curves(
    results: dict[str, SymbolResult],
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
    shadow: ShadowTracker | None,
    engines: dict[str, BacktestEngine],
    configs: dict[str, SymbolConfig],
    data: PortfolioData,
    bt_config: BacktestConfig,
) -> dict[str, FilterStats]:
    """Run shadow simulation on rejections collected during a backtest run."""
    if not shadow or not shadow.rejections:
        return {}
    syms = list(engines)
    # Run within patch context so shadow sim uses same overrides as main run
    with _AblationPatch(bt_config.flags, bt_config.param_overrides):
        shadow.simulate_shadows(
            hourly_data={s: (data.hourly[s].opens, data.hourly[s].highs, data.hourly[s].lows,
                             data.hourly[s].closes, data.hourly[s].volumes) for s in syms},
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
