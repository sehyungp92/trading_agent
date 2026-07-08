"""CLI entry points for the backtesting framework.

Usage:
    python -m backtest.cli run --strategy vdubus --diagnostics
    python -m backtest.cli run --strategy nqdtc
    python -m backtest.cli ablation --strategy vdubus --filter daily_trend_gate
    python -m backtest.cli optimize --strategy nqdtc --n-coarse 500 --n-refine 200
    python -m backtest.cli walk-forward --strategy vdubus --test-months 6
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _default_symbols(strategy: str) -> list[str]:
    """Resolve default symbols based on strategy."""
    return ["NQ"]


def _log_live_parity_caveats(strategy: str, fixed_qty: int | None) -> None:
    """Make non-live-equivalent assumptions explicit in single-strategy runs."""
    if fixed_qty not in (None, 0):
        logger.warning(
            "%s backtest is not live-equivalent: --fixed-qty=%s bypasses live risk-based sizing. "
            "Use --fixed-qty 0 to keep backtest sizing closer to live.",
            strategy,
            fixed_qty,
        )

    logger.warning(
        "%s single-strategy backtest bypasses live OMS entry gating "
        "(daily/weekly stops, heat cap, working-order limits, portfolio rules).",
        strategy,
    )

    if strategy == "NQDTC":
        logger.warning(
            "NQDTC backtest models B-sweep as a marketable IOC LIMIT, matching "
            "the live entry intent; single-strategy OMS gating caveats still apply."
        )
    elif strategy == "Vdubus":
        logger.warning(
            "Vdubus backtest uses a single active position model, while live can "
            "carry same-direction add-ons/pyramids."
        )


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_nqdtc_data(symbol: str, data_dir: Path) -> dict:
    """Load 5-min NQ parquet and resample to multi-TF arrays for NQDTCEngine."""
    from backtests.momentum.data.cache import load_bars
    from backtests.momentum.data.preprocessing import (
        align_daily_to_5m,
        align_higher_tf_to_5m,
        build_numpy_arrays,
        filter_eth,
        normalize_timezone,
        resample_5m_to_1h,
        resample_5m_to_30m,
        resample_5m_to_4h,
        resample_5m_to_daily,
    )

    five_min_path = data_dir / f"{symbol}_5m.parquet"
    es_daily_path = data_dir / "ES_1d.parquet"

    if not five_min_path.exists():
        raise FileNotFoundError(
            f"Missing 5-min data: {five_min_path}. "
            f"Run 'download --strategy nqdtc' first."
        )

    m_df = normalize_timezone(load_bars(five_min_path))
    m_df = filter_eth(m_df)

    thirty_min_df = resample_5m_to_30m(m_df)
    hourly_df = resample_5m_to_1h(m_df)
    four_hour_df = resample_5m_to_4h(m_df)

    daily_df = resample_5m_to_daily(m_df)

    five_min_bars = build_numpy_arrays(m_df)
    thirty_min = build_numpy_arrays(thirty_min_df)
    hourly = build_numpy_arrays(hourly_df)
    four_hour = build_numpy_arrays(four_hour_df)
    daily = build_numpy_arrays(daily_df)

    thirty_min_idx_map = align_higher_tf_to_5m(m_df, thirty_min_df)
    hourly_idx_map = align_higher_tf_to_5m(m_df, hourly_df)
    four_hour_idx_map = align_higher_tf_to_5m(m_df, four_hour_df)
    daily_idx_map = align_daily_to_5m(m_df, daily_df)

    result = {
        "five_min_bars": five_min_bars,
        "thirty_min": thirty_min,
        "hourly": hourly,
        "four_hour": four_hour,
        "daily": daily,
        "thirty_min_idx_map": thirty_min_idx_map,
        "hourly_idx_map": hourly_idx_map,
        "four_hour_idx_map": four_hour_idx_map,
        "daily_idx_map": daily_idx_map,
    }

    # ES daily data (optional ??for cross-strategy ES SMA200 regime)
    if es_daily_path.exists():
        es_daily_df = normalize_timezone(load_bars(es_daily_path))
        result["daily_es"] = build_numpy_arrays(es_daily_df)
        result["daily_es_idx_map"] = align_daily_to_5m(m_df, es_daily_df)
        logger.info(
            "Loaded %s: %d 5m bars, %d 30m, %d 1H, %d 4H, %d daily, %d ES daily",
            symbol, len(m_df), len(thirty_min_df), len(hourly_df),
            len(four_hour_df), len(daily_df), len(es_daily_df),
        )
    else:
        logger.info(
            "Loaded %s: %d 5m bars, %d 30m, %d 1H, %d 4H, %d daily (no ES daily)",
            symbol, len(m_df), len(thirty_min_df), len(hourly_df),
            len(four_hour_df), len(daily_df),
        )

    return result


def _load_downturn_data(symbol: str, data_dir: Path) -> dict:
    """Load 5m NQ and resample to 6 timeframes for DownturnEngine."""
    from backtests.momentum.data.cache import load_bars
    from backtests.momentum.data.preprocessing import (
        align_daily_to_5m,
        align_higher_tf_to_5m,
        build_numpy_arrays,
        filter_eth,
        normalize_timezone,
        resample_5m_to_15m,
        resample_5m_to_1h,
        resample_5m_to_30m,
        resample_5m_to_4h,
        resample_5m_to_daily,
    )

    five_min_path = data_dir / f"{symbol}_5m.parquet"
    es_daily_path = data_dir / "ES_1d.parquet"

    if not five_min_path.exists():
        raise FileNotFoundError(
            f"Missing 5-min data: {five_min_path}. "
            f"Run 'download --strategy nqdtc' first (shares same data)."
        )

    m_df = normalize_timezone(load_bars(five_min_path))
    m_df = filter_eth(m_df)

    m15_df = resample_5m_to_15m(m_df)
    m30_df = resample_5m_to_30m(m_df)
    h_df = resample_5m_to_1h(m_df)
    fh_df = resample_5m_to_4h(m_df)

    d_df = resample_5m_to_daily(m_df)

    five_min = build_numpy_arrays(m_df)
    m15 = build_numpy_arrays(m15_df)
    m30 = build_numpy_arrays(m30_df)
    h = build_numpy_arrays(h_df)
    fh = build_numpy_arrays(fh_df)
    d = build_numpy_arrays(d_df)

    result = {
        "five_min": five_min,
        "fifteen_min": m15,
        "thirty_min": m30,
        "hourly": h,
        "four_hour": fh,
        "daily": d,
        "fifteen_min_idx_map": align_higher_tf_to_5m(m_df, m15_df),
        "thirty_min_idx_map": align_higher_tf_to_5m(m_df, m30_df),
        "hourly_idx_map": align_higher_tf_to_5m(m_df, h_df),
        "four_hour_idx_map": align_higher_tf_to_5m(m_df, fh_df),
        "daily_idx_map": align_daily_to_5m(m_df, d_df),
    }

    if es_daily_path.exists():
        es_df = normalize_timezone(load_bars(es_daily_path))
        result["daily_es"] = build_numpy_arrays(es_df)
        result["daily_es_idx_map"] = align_daily_to_5m(m_df, es_df)
    else:
        result["daily_es"] = None
        result["daily_es_idx_map"] = None

    logger.info(
        "Loaded downturn %s: %d 5m, %d 15m, %d 30m, %d 1H, %d 4H, %d daily",
        symbol, len(m_df), len(m15_df), len(m30_df), len(h_df), len(fh_df), len(d_df),
    )
    return result


def _load_vdubus_data(symbol: str, data_dir: Path, include_5m: bool = False) -> dict:
    """Load NQ 15m + ES daily parquet and resample for VdubusEngine."""
    from backtests.momentum.data.cache import load_bars
    from backtests.momentum.data.preprocessing import (
        align_5m_to_15m,
        align_daily_to_15m,
        align_higher_tf_to_15m,
        build_numpy_arrays,
        filter_vdubus_session,
        normalize_timezone,
        resample_15m_to_1h,
        resample_5m_to_15m,
    )

    es_daily_path = data_dir / "ES_1d.parquet"
    five_min_path = data_dir / f"{symbol}_5m.parquet"
    fifteen_min_path = data_dir / f"{symbol}_15m.parquet"

    if not five_min_path.exists() and not fifteen_min_path.exists():
        raise FileNotFoundError(
            f"Missing 5m/15m data: {five_min_path} or {fifteen_min_path}. "
            f"Run 'download --strategy vdubus' first."
        )
    if not es_daily_path.exists():
        raise FileNotFoundError(
            f"Missing ES daily data: {es_daily_path}. "
            f"Run 'download --strategy vdubus' first."
        )

    if five_min_path.exists():
        five_min_source_df = normalize_timezone(load_bars(five_min_path))
        m_df = resample_5m_to_15m(five_min_source_df)
        source_label = "5m-derived"
    else:
        five_min_source_df = None
        m_df = normalize_timezone(load_bars(fifteen_min_path))
        source_label = "15m"
    m_df = filter_vdubus_session(m_df)

    hourly_df = resample_15m_to_1h(m_df)
    es_daily_df = normalize_timezone(load_bars(es_daily_path))

    bars_15m = build_numpy_arrays(m_df)
    hourly = build_numpy_arrays(hourly_df)
    daily_es = build_numpy_arrays(es_daily_df)

    hourly_idx_map = align_higher_tf_to_15m(m_df, hourly_df)
    daily_es_idx_map = align_daily_to_15m(m_df, es_daily_df)

    logger.info(
        "Loaded %s: %d %s 15m bars, %d 1H, %d ES daily",
        symbol, len(m_df), source_label, len(hourly_df), len(es_daily_df),
    )

    result = {
        "bars_15m": bars_15m,
        "hourly": hourly,
        "daily_es": daily_es,
        "hourly_idx_map": hourly_idx_map,
        "daily_es_idx_map": daily_es_idx_map,
    }

    # Optional 5m data for micro-trigger
    if include_5m:
        if five_min_source_df is not None:
            five_df = five_min_source_df
            fifteen_df_for_align = m_df  # reuse filtered 15m
            five_to_15_idx_map = align_5m_to_15m(five_df, fifteen_df_for_align)
            result["bars_5m"] = build_numpy_arrays(five_df)
            result["five_to_15_idx_map"] = five_to_15_idx_map
            logger.info("  + %d 5m bars for micro-trigger", len(five_df))
        else:
            logger.warning("5m data not found at %s, micro-trigger disabled", five_min_path)

    return result


def cmd_download(args):
    """Download historical data from IBKR."""
    data_dir = Path(args.data_dir)
    duration = args.duration

    if args.strategy == "nqdtc":
        from backtests.momentum.data.downloader import download_nqdtc_data

        symbol = args.symbols

        async def _run_nqdtc():
            result = await download_nqdtc_data(
                symbol=symbol,
                duration=duration,
                output_dir=data_dir,
            )
            for tf, path in result.items():
                logger.info("Downloaded %s %s -> %s", symbol, tf, path)

        asyncio.run(_run_nqdtc())
    elif args.strategy == "vdubus":
        from backtests.momentum.data.downloader import download_vdubus_data

        symbol = args.symbols

        async def _run_vdubus():
            result = await download_vdubus_data(
                symbol=symbol,
                duration=duration,
                output_dir=data_dir,
            )
            for key, path in result.items():
                logger.info("Downloaded %s -> %s", key, path)

        asyncio.run(_run_vdubus())
    else:
        logger.error("Unknown strategy: %s", args.strategy)


# ---------------------------------------------------------------------------
# NQDTC commands (strategy_2)
# ---------------------------------------------------------------------------

def _cmd_run_nqdtc(args):
    """Run a single NQDTC v2.0 backtest."""
    from backtests.momentum.analysis.metrics import compute_metrics
    from backtests.momentum.analysis.reports import (
        format_summary,
        nqdtc_behavior_report,
        nqdtc_diagnostic_report,
        nqdtc_performance_report,
    )
    from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
    from backtests.momentum.engine.nqdtc_engine import NQDTCEngine

    symbol = args.symbols  # "NQ" ??used for data file paths
    data_dir = Path(args.data_dir)
    nqdtc_data = _load_nqdtc_data(symbol, data_dir)  # loads NQ_5m.parquet

    _log_live_parity_caveats("NQDTC", args.fixed_qty)

    config = NQDTCBacktestConfig(
        symbols=["MNQ"],
        start_date=datetime.fromisoformat(args.start) if args.start else None,
        end_date=datetime.fromisoformat(args.end) if args.end else None,
        initial_equity=args.equity,
        data_dir=data_dir,
        fixed_qty=args.fixed_qty,
    )

    engine = NQDTCEngine("MNQ", config)  # engine uses MNQ cost specs
    result = engine.run(
        nqdtc_data["five_min_bars"],
        nqdtc_data["thirty_min"],
        nqdtc_data["hourly"],
        nqdtc_data["four_hour"],
        nqdtc_data["daily"],
        nqdtc_data["thirty_min_idx_map"],
        nqdtc_data["hourly_idx_map"],
        nqdtc_data["four_hour_idx_map"],
        nqdtc_data["daily_idx_map"],
        daily_es=nqdtc_data.get("daily_es"),
        daily_es_idx_map=nqdtc_data.get("daily_es_idx_map"),
    )

    report_sections: list[str] = []

    if not result.trades:
        logger.info("%s: No trades", symbol)
    else:
        pnls = np.array([t.pnl_dollars for t in result.trades])
        risks = np.array([
            abs(t.entry_price - t.initial_stop) * config.point_value * t.qty
            for t in result.trades
        ])
        holds = np.array([t.bars_held_30m for t in result.trades])
        comms = np.array([t.commission for t in result.trades])

        metrics = compute_metrics(
            trade_pnls=pnls,
            trade_risks=risks,
            trade_hold_hours=holds,
            trade_commissions=comms,
            equity_curve=result.equity_curve,
            timestamps=result.timestamps,
            initial_equity=config.initial_equity,
        )

        report_sections.append(nqdtc_performance_report(symbol, metrics))
        report_sections.append(nqdtc_behavior_report(result.trades))
        report_sections.append(nqdtc_diagnostic_report(result))
        report_sections.append(format_summary(metrics))

        # Extended diagnostics
        if getattr(args, 'diagnostics', False):
            from backtests.momentum.analysis.nqdtc_diagnostics import nqdtc_full_diagnostic
            report_sections.append(nqdtc_full_diagnostic(
                result.trades, signal_events=result.signal_events,
            ))

            # Gating attribution (critical deliverable)
            if result.signal_events:
                from backtests.momentum.analysis.nqdtc_filter_attribution import (
                    nqdtc_filter_attribution_report,
                )
                report_sections.append(
                    nqdtc_filter_attribution_report(
                        result.signal_events, result.trades,
                    )
                )

    # Shadow trade report (rejected-candidate simulation)
    if result.shadow_summary:
        report_sections.append(result.shadow_summary)

    for section in report_sections:
        try:
            print(f"\n{section}")
        except UnicodeEncodeError:
            print(section.encode("ascii", errors="replace").decode("ascii"))

    report_file = getattr(args, 'report_file', None)
    if report_file and report_sections:
        report_path = Path(report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n\n".join(report_sections) + "\n", encoding="utf-8")
        logger.info("Report saved: %s", report_path)


def _cmd_ablation_nqdtc(args):
    """Run NQDTC ablation test."""
    from backtests.momentum.analysis.metrics import compute_metrics
    from backtests.momentum.analysis.reports import print_summary
    from backtests.momentum.config_nqdtc import NQDTCAblationFlags, NQDTCBacktestConfig
    from backtests.momentum.engine.nqdtc_engine import NQDTCEngine

    symbol = args.symbols  # "NQ" ??used for data file paths
    data_dir = Path(args.data_dir)
    nqdtc_data = _load_nqdtc_data(symbol, data_dir)  # loads NQ_5m.parquet

    filter_names: list[str] = []
    if args.filters:
        filter_names = [f.strip() for f in args.filters.split(",") if f.strip()]
    elif args.filter:
        filter_names = [args.filter]
    else:
        logger.error("Must specify --filter or --filters")
        return

    flags_template = NQDTCAblationFlags()
    valid_flags = [f for f in vars(flags_template) if not f.startswith("_")]
    for fn in filter_names:
        if fn not in valid_flags:
            logger.error("Unknown NQDTC filter: %s", fn)
            logger.info("Valid filters: %s", valid_flags)
            return

    def _run_one(config):
        engine = NQDTCEngine("MNQ", config)  # engine uses MNQ cost specs
        return engine.run(
            nqdtc_data["five_min_bars"], nqdtc_data["thirty_min"],
            nqdtc_data["hourly"], nqdtc_data["four_hour"], nqdtc_data["daily"],
            nqdtc_data["thirty_min_idx_map"], nqdtc_data["hourly_idx_map"],
            nqdtc_data["four_hour_idx_map"], nqdtc_data["daily_idx_map"],
            daily_es=nqdtc_data.get("daily_es"),
            daily_es_idx_map=nqdtc_data.get("daily_es_idx_map"),
        )

    baseline_config = NQDTCBacktestConfig(
        symbols=["MNQ"], initial_equity=args.equity, data_dir=data_dir,
    )
    baseline_result = _run_one(baseline_config)

    flags = NQDTCAblationFlags()
    for fn in filter_names:
        if hasattr(flags, fn):
            setattr(flags, fn, False)

    ablation_config = NQDTCBacktestConfig(
        symbols=["MNQ"], initial_equity=args.equity, flags=flags, data_dir=data_dir,
    )
    ablation_result = _run_one(ablation_config)

    ablation_label = ",".join(filter_names)
    print(f"\n=== NQDTC Ablation: {ablation_label} = OFF ===\n")

    for label, result in [("Baseline", baseline_result), ("Ablated", ablation_result)]:
        if not result.trades:
            print(f"  {label}: No trades")
            continue
        pnls = np.array([t.pnl_dollars for t in result.trades])
        risks = np.array([
            abs(t.entry_price - t.initial_stop) * baseline_config.point_value * t.qty
            for t in result.trades
        ])
        holds = np.array([t.bars_held_30m for t in result.trades])
        comms = np.array([t.commission for t in result.trades])
        metrics = compute_metrics(
            pnls, risks, holds, comms,
            result.equity_curve, result.timestamps, args.equity,
        )
        print(f"  {label}: ", end="")
        print_summary(metrics)
    print()


def _cmd_optimize_nqdtc(args):
    """Run NQDTC parameter optimization."""
    from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
    from backtests.momentum.optimization.nqdtc_runner import NQDTCOptimizationRunner

    symbol = args.symbols
    data_dir = Path(args.data_dir)
    nqdtc_data = _load_nqdtc_data(symbol, data_dir)

    config = NQDTCBacktestConfig(
        symbols=[symbol],
        initial_equity=args.equity,
        data_dir=data_dir,
        track_signals=False,
        track_shadows=False,
    )

    runner = NQDTCOptimizationRunner(
        base_config=config,
        nqdtc_data=nqdtc_data,
        n_coarse=args.n_coarse,
        n_refine=args.n_refine,
    )
    result = runner.run()
    _print_optimization_results(result)


def _cmd_walk_forward_nqdtc(args):
    """Run NQDTC walk-forward validation."""
    from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
    from backtests.momentum.optimization.nqdtc_walk_forward import NQDTCWalkForwardValidator

    symbol = args.symbols
    data_dir = Path(args.data_dir)
    nqdtc_data = _load_nqdtc_data(symbol, data_dir)

    config = NQDTCBacktestConfig(
        symbols=[symbol],
        initial_equity=args.equity,
        data_dir=data_dir,
    )

    validator = NQDTCWalkForwardValidator(
        nqdtc_data=nqdtc_data,
        base_config=config,
        test_window_months=args.test_months,
    )
    result = validator.run()
    _print_walk_forward_results(result)


# ---------------------------------------------------------------------------
# VdubusNQ v4.0 commands (strategy_3)
# ---------------------------------------------------------------------------

def _cmd_run_vdubus(args):
    """Run a single VdubusNQ v4.0 backtest."""
    from backtests.momentum.analysis.metrics import compute_metrics
    from backtests.momentum.analysis.reports import format_summary
    from backtests.momentum.config_vdubus import VdubusAblationFlags, VdubusBacktestConfig
    from backtests.momentum.engine.vdubus_engine import VdubusEngine
    from strategies.momentum.vdub import config as C

    symbol = args.symbols
    data_dir = Path(args.data_dir)
    include_5m = getattr(args, 'micro_trigger', False)
    vdubus_data = _load_vdubus_data(symbol, data_dir, include_5m=include_5m)

    _log_live_parity_caveats("Vdubus", args.fixed_qty)

    # Patch strategy_3 config for MNQ (micro contract)
    orig_nq_spec = dict(C.NQ_SPEC)
    orig_rt_comm = C.RT_COMM_FEES
    C.NQ_SPEC["tick_value"] = 0.50
    C.NQ_SPEC["point_value"] = 2.0
    C.RT_COMM_FEES = 1.24

    # Disable gates N/A for fixed-qty
    flags = VdubusAblationFlags(
        heat_cap=False,
        viability_filter=False,
    )

    config = VdubusBacktestConfig(
        symbols=[symbol],
        start_date=datetime.fromisoformat(args.start) if args.start else None,
        end_date=datetime.fromisoformat(args.end) if args.end else None,
        initial_equity=args.equity,
        data_dir=data_dir,
        fixed_qty=args.fixed_qty,
        flags=flags,
    )

    try:
        engine = VdubusEngine(symbol, config)
        result = engine.run(
            vdubus_data["bars_15m"],
            vdubus_data.get("bars_5m"),
            vdubus_data["hourly"],
            vdubus_data["daily_es"],
            vdubus_data["hourly_idx_map"],
            vdubus_data["daily_es_idx_map"],
            vdubus_data.get("five_to_15_idx_map"),
        )
    finally:
        C.NQ_SPEC.update(orig_nq_spec)
        C.RT_COMM_FEES = orig_rt_comm

    report_sections: list[str] = []

    if not result.trades:
        logger.info("%s: No trades", symbol)
    else:
        pnls = np.array([t.pnl_dollars for t in result.trades])
        risks = np.array([
            abs(t.entry_price - t.initial_stop) * config.point_value * t.qty
            for t in result.trades
        ])
        holds = np.array([t.bars_held_15m for t in result.trades])
        comms = np.array([t.commission for t in result.trades])

        metrics = compute_metrics(
            trade_pnls=pnls,
            trade_risks=risks,
            trade_hold_hours=holds,
            trade_commissions=comms,
            equity_curve=result.equity_curve,
            timestamps=result.time_series,
            initial_equity=config.initial_equity,
        )

        # Performance summary
        lines = [
            f"=== VdubusNQ v4.0 Performance Report: {symbol} ===",
            f"Total trades:       {metrics.total_trades}",
            f"Win rate:           {metrics.win_rate:.1%}",
            f"Profit factor:      {metrics.profit_factor:.2f}",
            f"Expectancy (R):     {metrics.expectancy:+.3f}",
            f"Expectancy ($):     {metrics.expectancy_dollar:+,.2f}",
            f"Net profit:         ${metrics.net_profit:+,.2f}",
            f"CAGR:               {metrics.cagr:.1%}",
            f"Sharpe:             {metrics.sharpe:.2f}",
            f"Sortino:            {metrics.sortino:.2f}",
            f"Calmar:             {metrics.calmar:.2f}",
            f"Max drawdown:       {metrics.max_drawdown_pct:.1%} (${metrics.max_drawdown_dollar:,.2f})",
            f"Avg hold (15m bars):{metrics.avg_hold_hours:.1f}",
            f"Trades/month:       {metrics.trades_per_month:.1f}",
            f"Total commissions:  ${metrics.total_commissions:,.2f}",
        ]
        report_sections.append("\n".join(lines))

        # Signal funnel
        funnel = [
            f"=== VdubusNQ Signal Funnel ===",
            f"  15m evaluations:  {result.evaluations}",
            f"  Regime passed:    {result.regime_passed}",
            f"  Signals found:    {result.signals_found}",
            f"  Entries placed:   {result.entries_placed}",
            f"  Entries filled:   {result.entries_filled}",
            f"  Trades completed: {len(result.trades)}",
        ]
        report_sections.append("\n".join(funnel))

        report_sections.append(format_summary(metrics))

        # Extended diagnostics
        if getattr(args, 'diagnostics', False):
            from backtests.momentum.analysis.vdubus_diagnostics import vdubus_full_diagnostic
            report_sections.append(vdubus_full_diagnostic(
                result.trades, signal_events=result.signal_events,
                equity_curve=result.equity_curve,
                time_series=result.time_series,
            ))

            # Gating attribution
            if result.signal_events:
                from backtests.momentum.analysis.vdubus_filter_attribution import (
                    vdubus_filter_attribution_report,
                )
                report_sections.append(
                    vdubus_filter_attribution_report(
                        result.signal_events, result.trades,
                        shadow_tracker=result.shadow_tracker,
                    )
                )

    # Shadow trade report (rejected-candidate simulation)
    if result.shadow_summary:
        report_sections.append(result.shadow_summary)

    for section in report_sections:
        try:
            print(f"\n{section}")
        except UnicodeEncodeError:
            print(section.encode("ascii", errors="replace").decode("ascii"))

    report_file = getattr(args, 'report_file', None)
    if report_file and report_sections:
        report_path = Path(report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n\n".join(report_sections) + "\n", encoding="utf-8")
        logger.info("Report saved: %s", report_path)


def _cmd_ablation_vdubus(args):
    """Run VdubusNQ ablation test."""
    from backtests.momentum.analysis.metrics import compute_metrics
    from backtests.momentum.analysis.reports import print_summary
    from backtests.momentum.config_vdubus import VdubusAblationFlags, VdubusBacktestConfig
    from backtests.momentum.engine.vdubus_engine import VdubusEngine
    from strategies.momentum.vdub import config as C

    symbol = args.symbols
    data_dir = Path(args.data_dir)
    vdubus_data = _load_vdubus_data(symbol, data_dir)

    filter_names: list[str] = []
    if args.filters:
        filter_names = [f.strip() for f in args.filters.split(",") if f.strip()]
    elif args.filter:
        filter_names = [args.filter]
    else:
        logger.error("Must specify --filter or --filters")
        return

    flags_template = VdubusAblationFlags()
    valid_flags = [f for f in vars(flags_template) if not f.startswith("_")]
    for fn in filter_names:
        if fn not in valid_flags:
            logger.error("Unknown VdubusNQ filter: %s", fn)
            logger.info("Valid filters: %s", valid_flags)
            return

    # Patch strategy_3 config for MNQ (micro contract)
    orig_nq_spec = dict(C.NQ_SPEC)
    orig_rt_comm = C.RT_COMM_FEES
    C.NQ_SPEC["tick_value"] = 0.50
    C.NQ_SPEC["point_value"] = 2.0
    C.RT_COMM_FEES = 1.24

    def _run_one(config):
        engine = VdubusEngine(symbol, config)
        return engine.run(
            vdubus_data["bars_15m"], vdubus_data.get("bars_5m"),
            vdubus_data["hourly"], vdubus_data["daily_es"],
            vdubus_data["hourly_idx_map"], vdubus_data["daily_es_idx_map"],
            vdubus_data.get("five_to_15_idx_map"),
        )

    try:
        # Baseline uses same fixed-qty flags (heat_cap=False, viability_filter=False)
        baseline_flags = VdubusAblationFlags(
            heat_cap=False,
            viability_filter=False,
        )
        baseline_config = VdubusBacktestConfig(
            symbols=[symbol], initial_equity=args.equity, data_dir=data_dir,
            flags=baseline_flags,
        )
        baseline_result = _run_one(baseline_config)

        # Ablation toggle applies on top of baseline flags
        ablation_flags = VdubusAblationFlags(
            heat_cap=False,
            viability_filter=False,
        )
        for fn in filter_names:
            if hasattr(ablation_flags, fn):
                setattr(ablation_flags, fn, False)

        ablation_config = VdubusBacktestConfig(
            symbols=[symbol], initial_equity=args.equity, flags=ablation_flags,
            data_dir=data_dir,
        )
        ablation_result = _run_one(ablation_config)
    finally:
        C.NQ_SPEC.update(orig_nq_spec)
        C.RT_COMM_FEES = orig_rt_comm

    ablation_label = ",".join(filter_names)
    print(f"\n=== VdubusNQ Ablation: {ablation_label} = OFF ===\n")

    for label, result in [("Baseline", baseline_result), ("Ablated", ablation_result)]:
        if not result.trades:
            print(f"  {label}: No trades")
            continue
        pnls = np.array([t.pnl_dollars for t in result.trades])
        risks = np.array([
            abs(t.entry_price - t.initial_stop) * baseline_config.point_value * t.qty
            for t in result.trades
        ])
        holds = np.array([t.bars_held_15m for t in result.trades])
        comms = np.array([t.commission for t in result.trades])
        metrics = compute_metrics(
            pnls, risks, holds, comms,
            result.equity_curve, result.time_series, args.equity,
        )
        print(f"  {label}: ", end="")
        print_summary(metrics)
    print()


def _cmd_optimize_vdubus(args):
    """Run VdubusNQ parameter optimization."""
    from backtests.momentum.config_vdubus import VdubusBacktestConfig
    from backtests.momentum.optimization.vdubus_runner import VdubusOptimizationRunner

    symbol = args.symbols
    data_dir = Path(args.data_dir)
    vdubus_data = _load_vdubus_data(symbol, data_dir)

    config = VdubusBacktestConfig(
        symbols=[symbol],
        initial_equity=args.equity,
        data_dir=data_dir,
        track_signals=False,
        track_shadows=False,
    )

    runner = VdubusOptimizationRunner(
        base_config=config,
        vdubus_data=vdubus_data,
        n_coarse=args.n_coarse,
        n_refine=args.n_refine,
    )
    result = runner.run()
    _print_optimization_results(result)


def _cmd_walk_forward_vdubus(args):
    """Run VdubusNQ walk-forward validation."""
    from backtests.momentum.config_vdubus import VdubusBacktestConfig
    from backtests.momentum.optimization.vdubus_walk_forward import VdubusWalkForwardValidator

    symbol = args.symbols
    data_dir = Path(args.data_dir)
    vdubus_data = _load_vdubus_data(symbol, data_dir)

    config = VdubusBacktestConfig(
        symbols=[symbol],
        initial_equity=args.equity,
        data_dir=data_dir,
    )

    validator = VdubusWalkForwardValidator(
        vdubus_data=vdubus_data,
        base_config=config,
        test_window_months=args.test_months,
    )
    result = validator.run()
    _print_walk_forward_results(result)


# ---------------------------------------------------------------------------
# Portfolio command
# ---------------------------------------------------------------------------

def _cmd_run_portfolio(args):
    """Run combined portfolio backtest across all strategies."""
    from backtests.momentum.analysis.portfolio_reports import portfolio_full_report
    from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
    from backtests.momentum.config_portfolio import PRESETS, PortfolioBacktestConfig
    from backtests.momentum.config_vdubus import VdubusAblationFlags, VdubusBacktestConfig
    from backtests.momentum.engine.portfolio_engine import PortfolioBacktester

    data_dir = Path(args.data_dir)
    preset_name = getattr(args, 'preset', '10k_v6')

    if preset_name not in PRESETS:
        logger.error(
            "Unknown preset: %s. Available: %s",
            preset_name, list(PRESETS.keys()),
        )
        return

    pc = PRESETS[preset_name]()
    config = PortfolioBacktestConfig(
        portfolio=pc,
        data_dir=data_dir,
        start_date=datetime.fromisoformat(args.start) if args.start else None,
        end_date=datetime.fromisoformat(args.end) if args.end else None,
    )

    independent_pnl: dict[str, float] = {}  # sum of R-multiples per strategy
    independent_trades: dict[str, int] = {}  # trade count per strategy
    nqdtc_trades = None
    vdubus_trades = None

    # --- Run NQDTC ---
    nqdtc_alloc = pc.get_strategy("NQDTC")
    if config.run_nqdtc and nqdtc_alloc and nqdtc_alloc.enabled:
        from backtests.momentum.engine.nqdtc_engine import NQDTCEngine

        symbol = "NQ"
        nqdtc_data = _load_nqdtc_data(symbol, data_dir)
        nqdtc_cfg = NQDTCBacktestConfig(
            symbols=["MNQ"],
            start_date=config.start_date,
            end_date=config.end_date,
            initial_equity=pc.initial_equity,
            data_dir=data_dir,
            fixed_qty=10,  # engine uses fixed_qty; portfolio rescales
            track_signals=False,
            track_shadows=False,
        )
        engine = NQDTCEngine("MNQ", nqdtc_cfg)
        nqdtc_result = engine.run(
            nqdtc_data["five_min_bars"], nqdtc_data["thirty_min"],
            nqdtc_data["hourly"], nqdtc_data["four_hour"], nqdtc_data["daily"],
            nqdtc_data["thirty_min_idx_map"], nqdtc_data["hourly_idx_map"],
            nqdtc_data["four_hour_idx_map"], nqdtc_data["daily_idx_map"],
            daily_es=nqdtc_data.get("daily_es"),
            daily_es_idx_map=nqdtc_data.get("daily_es_idx_map"),
        )
        nqdtc_trades = nqdtc_result.trades
        independent_pnl["NQDTC"] = sum(t.r_multiple for t in nqdtc_trades)
        independent_trades["NQDTC"] = len(nqdtc_trades)
        logger.info("NQDTC: %d trades, %+.1fR", len(nqdtc_trades), independent_pnl["NQDTC"])

    # --- Run Vdubus ---
    vdubus_alloc = pc.get_strategy("Vdubus")
    if config.run_vdubus and vdubus_alloc and vdubus_alloc.enabled:
        from backtests.momentum.engine.vdubus_engine import VdubusEngine
        from strategies.momentum.vdub import config as C

        symbol = "NQ"
        vdubus_data = _load_vdubus_data(symbol, data_dir)

        orig_nq_spec = dict(C.NQ_SPEC)
        orig_rt_comm = C.RT_COMM_FEES
        C.NQ_SPEC["tick_value"] = 0.50
        C.NQ_SPEC["point_value"] = 2.0
        C.RT_COMM_FEES = 1.24

        vdubus_flags = VdubusAblationFlags(
            heat_cap=False,
            viability_filter=False,
        )
        vdubus_cfg = VdubusBacktestConfig(
            symbols=[symbol],
            start_date=config.start_date,
            end_date=config.end_date,
            initial_equity=pc.initial_equity,
            data_dir=data_dir,
            fixed_qty=10,  # engine uses fixed_qty; portfolio rescales
            flags=vdubus_flags,
            track_signals=False,
            track_shadows=False,
        )
        try:
            engine = VdubusEngine(symbol, vdubus_cfg)
            vdubus_result = engine.run(
                vdubus_data["bars_15m"], vdubus_data.get("bars_5m"),
                vdubus_data["hourly"], vdubus_data["daily_es"],
                vdubus_data["hourly_idx_map"], vdubus_data["daily_es_idx_map"],
                vdubus_data.get("five_to_15_idx_map"),
            )
        finally:
            C.NQ_SPEC.update(orig_nq_spec)
            C.RT_COMM_FEES = orig_rt_comm

        vdubus_trades = vdubus_result.trades
        independent_pnl["Vdubus"] = sum(t.r_multiple for t in vdubus_trades)
        independent_trades["Vdubus"] = len(vdubus_trades)
        logger.info("Vdubus: %d trades, %+.1fR", len(vdubus_trades), independent_pnl["Vdubus"])

    # --- Run portfolio simulation ---
    logger.info("Running portfolio simulation with preset '%s'...", preset_name)
    backtester = PortfolioBacktester(config)
    result = backtester.run(
        nqdtc_trades=nqdtc_trades,
        vdubus_trades=vdubus_trades,
    )

    # --- Generate report ---
    report = portfolio_full_report(result, independent_pnl=independent_pnl)

    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode("ascii", errors="replace").decode("ascii"))

    report_file = getattr(args, 'report_file', None)
    if report_file:
        report_path = Path(report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        logger.info("Report saved: %s", report_path)


def _cmd_sweep_portfolio(args):
    """Run portfolio parameter sweep across config variants."""
    from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
    from backtests.momentum.config_portfolio import PRESETS, PortfolioBacktestConfig
    from backtests.momentum.config_vdubus import VdubusAblationFlags, VdubusBacktestConfig
    from backtests.momentum.sweep_portfolio import (
        build_combined_variant,
        build_sweep_variants,
        find_winners,
        format_sweep_table,
        format_winners_summary,
        run_sweep,
    )

    data_dir = Path(args.data_dir)
    preset_name = getattr(args, 'preset', '10k_v6')

    if preset_name not in PRESETS:
        logger.error(
            "Unknown preset: %s. Available: %s",
            preset_name, list(PRESETS.keys()),
        )
        return

    pc = PRESETS[preset_name]()
    bt_config = PortfolioBacktestConfig(
        portfolio=pc,
        data_dir=data_dir,
        start_date=datetime.fromisoformat(args.start) if args.start else None,
        end_date=datetime.fromisoformat(args.end) if args.end else None,
    )

    independent_pnl: dict[str, float] = {}
    nqdtc_trades = None
    vdubus_trades = None

    # --- Run each engine once ---
    nqdtc_alloc = pc.get_strategy("NQDTC")
    if nqdtc_alloc and nqdtc_alloc.enabled:
        from backtests.momentum.engine.nqdtc_engine import NQDTCEngine

        symbol = "NQ"
        nqdtc_data = _load_nqdtc_data(symbol, data_dir)
        nqdtc_cfg = NQDTCBacktestConfig(
            symbols=["MNQ"],
            start_date=bt_config.start_date,
            end_date=bt_config.end_date,
            initial_equity=pc.initial_equity,
            data_dir=data_dir,
            fixed_qty=10,
            track_signals=False,
            track_shadows=False,
        )
        engine = NQDTCEngine("MNQ", nqdtc_cfg)
        nqdtc_result = engine.run(
            nqdtc_data["five_min_bars"], nqdtc_data["thirty_min"],
            nqdtc_data["hourly"], nqdtc_data["four_hour"], nqdtc_data["daily"],
            nqdtc_data["thirty_min_idx_map"], nqdtc_data["hourly_idx_map"],
            nqdtc_data["four_hour_idx_map"], nqdtc_data["daily_idx_map"],
            daily_es=nqdtc_data.get("daily_es"),
            daily_es_idx_map=nqdtc_data.get("daily_es_idx_map"),
        )
        nqdtc_trades = nqdtc_result.trades
        independent_pnl["NQDTC"] = sum(t.r_multiple for t in nqdtc_trades)
        logger.info("NQDTC: %d trades, %+.1fR", len(nqdtc_trades), independent_pnl["NQDTC"])

    vdubus_alloc = pc.get_strategy("Vdubus")
    if vdubus_alloc and vdubus_alloc.enabled:
        from backtests.momentum.engine.vdubus_engine import VdubusEngine
        from strategies.momentum.vdub import config as C

        symbol = "NQ"
        vdubus_data = _load_vdubus_data(symbol, data_dir)

        orig_nq_spec = dict(C.NQ_SPEC)
        orig_rt_comm = C.RT_COMM_FEES
        C.NQ_SPEC["tick_value"] = 0.50
        C.NQ_SPEC["point_value"] = 2.0
        C.RT_COMM_FEES = 1.24

        vdubus_flags = VdubusAblationFlags(
            heat_cap=False,
            viability_filter=False,
        )
        vdubus_cfg = VdubusBacktestConfig(
            symbols=[symbol],
            start_date=bt_config.start_date,
            end_date=bt_config.end_date,
            initial_equity=pc.initial_equity,
            data_dir=data_dir,
            fixed_qty=10,
            flags=vdubus_flags,
            track_signals=False,
            track_shadows=False,
        )
        try:
            engine = VdubusEngine(symbol, vdubus_cfg)
            vdubus_result = engine.run(
                vdubus_data["bars_15m"], vdubus_data.get("bars_5m"),
                vdubus_data["hourly"], vdubus_data["daily_es"],
                vdubus_data["hourly_idx_map"], vdubus_data["daily_es_idx_map"],
                vdubus_data.get("five_to_15_idx_map"),
            )
        finally:
            C.NQ_SPEC.update(orig_nq_spec)
            C.RT_COMM_FEES = orig_rt_comm

        vdubus_trades = vdubus_result.trades
        independent_pnl["Vdubus"] = sum(t.r_multiple for t in vdubus_trades)
        logger.info("Vdubus: %d trades, %+.1fR", len(vdubus_trades), independent_pnl["Vdubus"])

    iso_total_R = sum(independent_pnl.values())

    # --- Phase 1: Individual variants ---
    logger.info("Running parameter sweep (Phase 1: individual variants)...")
    variants = build_sweep_variants()
    results = run_sweep(
        baseline_config=pc,
        nqdtc_trades=nqdtc_trades,
        vdubus_trades=vdubus_trades,
        variants=variants,
        iso_total_R=iso_total_R,
        bt_config_template=bt_config,
    )

    table = format_sweep_table(results)
    try:
        print(table)
    except UnicodeEncodeError:
        print(table.encode("ascii", errors="replace").decode("ascii"))

    # --- Phase 2: Combined winners ---
    logger.info("Running parameter sweep (Phase 2: combined winners)...")
    winner_names = find_winners(results)
    combined_result = None

    if winner_names:
        combined_variant = build_combined_variant(winner_names, variants)
        if combined_variant:
            combined_config = combined_variant.config_factory(pc)
            combined_bt_cfg = PortfolioBacktestConfig(
                portfolio=combined_config,
                data_dir=data_dir,
                start_date=bt_config.start_date,
                end_date=bt_config.end_date,
            )
            from backtests.momentum.engine.portfolio_engine import PortfolioBacktester
            from backtests.momentum.sweep_portfolio import _extract_metrics

            bt = PortfolioBacktester(combined_bt_cfg)
            pr = bt.run(
                nqdtc_trades=nqdtc_trades,
                vdubus_trades=vdubus_trades,
            )
            combined_result = _extract_metrics(pr, iso_total_R, "COMBINED")
            baseline = results[0]
            combined_result.delta_R = combined_result.total_R - baseline.total_R
            combined_result.delta_sharpe = combined_result.sharpe - baseline.sharpe
            combined_result.delta_pnl = combined_result.net_pnl - baseline.net_pnl
            combined_result.delta_max_dd = combined_result.max_dd_pct - baseline.max_dd_pct

            # Print combined config for inspection
            logger.info(
                "Combined config: %s",
                {k: v for k, v in combined_config.__dict__.items()
                 if k != "strategies"},
            )

    summary = format_winners_summary(winner_names, combined_result)
    try:
        print(summary)
    except UnicodeEncodeError:
        print(summary.encode("ascii", errors="replace").decode("ascii"))

    # --- Save report ---
    report_file = getattr(args, 'report_file', None)
    if report_file:
        full_report = table + "\n" + summary
        report_path = Path(report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(full_report, encoding="utf-8")
        logger.info("Sweep report saved: %s", report_path)


# ---------------------------------------------------------------------------
# Command dispatchers
# ---------------------------------------------------------------------------

def _cmd_weakness_report(args):
    """Generate unified momentum weakness report by running active strategies."""
    from backtests.momentum.analysis.weakness_report import momentum_weakness_report
    from backtests.momentum.analysis.drawdown_attribution import drawdown_attribution_report

    logger.info("Running active momentum strategies for weakness report...")

    # Run each strategy and collect results
    results = {}
    all_trades = {}

    data_dir = Path(args.data_dir)

    # NQDTC
    try:
        from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
        from backtests.momentum.engine.nqdtc_engine import NQDTCEngine
        nqdtc_data = _load_nqdtc_data("NQ", data_dir)
        n_cfg = NQDTCBacktestConfig(
            symbols=["MNQ"], initial_equity=args.equity, data_dir=data_dir, fixed_qty=10,
        )
        n_eng = NQDTCEngine("MNQ", n_cfg)
        n_result = n_eng.run(
            nqdtc_data["five_min_bars"], nqdtc_data["thirty_min"],
            nqdtc_data["hourly"], nqdtc_data["four_hour"], nqdtc_data["daily"],
            nqdtc_data["thirty_min_idx_map"], nqdtc_data["hourly_idx_map"],
            nqdtc_data["four_hour_idx_map"], nqdtc_data["daily_idx_map"],
            daily_es=nqdtc_data.get("daily_es"),
            daily_es_idx_map=nqdtc_data.get("daily_es_idx_map"),
        )
        results["nqdtc"] = n_result
        all_trades["NQDTC"] = n_result.trades
    except Exception as e:
        logger.warning("NQDTC run failed: %s", e)

    # Vdubus
    try:
        from backtests.momentum.config_vdubus import VdubusAblationFlags, VdubusBacktestConfig
        from backtests.momentum.engine.vdubus_engine import VdubusEngine
        from strategies.momentum.vdub import config as C
        vdubus_data = _load_vdubus_data("NQ", data_dir)
        orig_nq_spec = dict(C.NQ_SPEC)
        orig_rt_comm = C.RT_COMM_FEES
        C.NQ_SPEC["tick_value"] = 0.50
        C.NQ_SPEC["point_value"] = 2.0
        C.RT_COMM_FEES = 1.24
        v_cfg = VdubusBacktestConfig(
            symbols=["NQ"], initial_equity=args.equity, data_dir=data_dir, fixed_qty=10,
            flags=VdubusAblationFlags(heat_cap=False, viability_filter=False),
        )
        try:
            v_eng = VdubusEngine("NQ", v_cfg)
            v_result = v_eng.run(
                vdubus_data["bars_15m"], vdubus_data.get("bars_5m"),
                vdubus_data["hourly"], vdubus_data["daily_es"],
                vdubus_data["hourly_idx_map"], vdubus_data["daily_es_idx_map"],
                vdubus_data.get("five_to_15_idx_map"),
            )
            results["vdubus"] = v_result
            all_trades["Vdubus"] = v_result.trades
        finally:
            C.NQ_SPEC.update(orig_nq_spec)
            C.RT_COMM_FEES = orig_rt_comm
    except Exception as e:
        logger.warning("Vdubus run failed: %s", e)

    report_sections = []

    # Weakness report
    report_sections.append(momentum_weakness_report(
        nqdtc_result=results.get("nqdtc"),
        vdubus_result=results.get("vdubus"),
    ))

    # Drawdown attribution (combined)
    combined_trades = [t for trades in all_trades.values() for t in trades]
    if combined_trades:
        # Build combined equity curve from individual results
        combined_eq = []
        for name, result in results.items():
            if hasattr(result, 'equity_curve'):
                eq = getattr(result, 'equity_curve', [])
                if isinstance(eq, np.ndarray):
                    combined_eq.append(eq)
                elif isinstance(eq, list) and eq:
                    combined_eq.append(np.array(eq))

        if combined_eq:
            # Use the longest equity curve as proxy
            longest = max(combined_eq, key=len)
            report_sections.append(drawdown_attribution_report(
                combined_trades, longest, list(range(len(longest))),
            ))

    output = "\n\n".join(s for s in report_sections if s)
    print(output)

    if getattr(args, 'report_file', None):
        Path(args.report_file).write_text(output)
        logger.info("Weakness report written to %s", args.report_file)


def _cmd_run_downturn(args):
    """Run a single Downturn Dominator backtest."""
    from backtests.momentum.config_downturn import DownturnBacktestConfig
    from backtests.momentum.engine.downturn_engine import DownturnEngine
    from backtests.momentum.analysis.downturn_diagnostics import (
        compute_downturn_metrics,
        generate_downturn_report,
    )

    symbol = args.symbols if isinstance(args.symbols, str) else "NQ"
    data_dir = Path(args.data_dir)
    data = _load_downturn_data(symbol, data_dir)

    config = DownturnBacktestConfig(
        symbols=["NQ"],
        start_date=datetime.fromisoformat(args.start) if args.start else None,
        end_date=datetime.fromisoformat(args.end) if args.end else None,
        initial_equity=args.equity,
        data_dir=data_dir,
    )

    engine = DownturnEngine("NQ", config)
    result = engine.run(**data)

    report_sections: list[str] = []

    if not result.trades:
        logger.info("Downturn: No trades generated")
    else:
        metrics = compute_downturn_metrics(result, data["daily"])
        report_sections.append(generate_downturn_report(metrics))

        # Per-engine signal counters
        lines = ["--- Engine Signal Counters ---"]
        for name, ctr in [
            ("Reversal", result.reversal_counters),
            ("Breakdown", result.breakdown_counters),
            ("Fade", result.fade_counters),
        ]:
            lines.append(
                f"  {name:12s}  signals={ctr.signals_detected:4d}  "
                f"placed={ctr.entries_placed:4d}  filled={ctr.entries_filled:4d}  "
                f"blocked={ctr.gates_blocked:4d}"
            )
        report_sections.append("\n".join(lines))

    output = "\n\n".join(s for s in report_sections if s)
    print(output)

    if getattr(args, 'report_file', None):
        Path(args.report_file).write_text(output)
        logger.info("Report written to %s", args.report_file)


def _cmd_ablation_downturn(args):
    """Run ablation test for Downturn Dominator."""
    from dataclasses import fields
    from backtests.momentum.config_downturn import DownturnAblationFlags, DownturnBacktestConfig
    from backtests.momentum.engine.downturn_engine import DownturnEngine
    from backtests.momentum.analysis.downturn_diagnostics import compute_downturn_metrics

    symbol = args.symbols if isinstance(args.symbols, str) else "NQ"
    data_dir = Path(args.data_dir)
    data = _load_downturn_data(symbol, data_dir)

    base_config = DownturnBacktestConfig(initial_equity=args.equity, data_dir=data_dir)

    # Baseline
    engine = DownturnEngine("NQ", base_config)
    base_result = engine.run(**data)
    base_metrics = compute_downturn_metrics(base_result, data["daily"])

    print(f"{'Flag':<35s} {'Trades':>7s} {'PF':>7s} {'DD%':>7s} {'CorrPnL':>10s} {'Delta PF':>9s}")
    print("-" * 80)
    print(f"{'BASELINE':<35s} {base_metrics.total_trades:>7d} {base_metrics.profit_factor:>7.2f} "
          f"{base_metrics.max_dd_pct:>7.2%} {base_metrics.correction_pnl_pct:>10.2f}")

    filter_names: list[str] = []
    if getattr(args, 'filter', None):
        filter_names = [args.filter]
    elif getattr(args, 'filters', None):
        filter_names = [f.strip() for f in args.filters.split(",") if f.strip()]
    else:
        filter_names = [f.name for f in fields(DownturnAblationFlags)
                        if f.default is True]  # only toggle enabled flags

    from dataclasses import replace
    for flag_name in filter_names:
        try:
            new_flags = replace(base_config.flags, **{flag_name: not getattr(base_config.flags, flag_name)})
        except TypeError:
            continue
        cfg = replace(base_config, flags=new_flags)
        eng = DownturnEngine("NQ", cfg)
        res = eng.run(**data)
        m = compute_downturn_metrics(res, data["daily"])
        delta_pf = m.profit_factor - base_metrics.profit_factor
        print(f"{flag_name:<35s} {m.total_trades:>7d} {m.profit_factor:>7.2f} "
              f"{m.max_dd_pct:>7.2%} {m.correction_pnl_pct:>10.2f} {delta_pf:>+9.2f}")


def _cmd_phase_run_downturn(args):
    """Run a single phase of downturn greedy optimization."""
    from backtests.momentum.auto.downturn.plugin import DownturnPlugin
    from backtests.shared.auto.phase_runner import PhaseRunner

    output_dir = Path(__file__).resolve().parent / "auto/downturn/output"
    plugin = DownturnPlugin(
        data_dir=Path(getattr(args, "data_dir", None) or "backtests/momentum/data/raw"),
        initial_equity=getattr(args, "equity", 100_000.0),
        max_workers=getattr(args, "max_workers", None),
    )
    runner = PhaseRunner(plugin=plugin, output_dir=output_dir, max_rounds=50)
    state = runner.run_phase(args.phase_num)
    result = state.phase_results.get(args.phase_num, {})
    logger.info(
        "Phase %d complete: %.4f -> %.4f (%d accepted, %.0fs)",
        args.phase_num,
        result.get("base_score", 0.0),
        result.get("final_score", 0.0),
        len(result.get("kept_features", [])),
        result.get("elapsed_seconds", 0.0),
    )


def _cmd_phase_auto_downturn(args):
    """Auto-run phases 1-5 with the shared analyzer loop."""
    from backtests.momentum.auto.downturn.plugin import DownturnPlugin
    from backtests.shared.auto.phase_runner import PhaseRunner

    output_dir = Path(__file__).resolve().parent / "auto/downturn/output"
    plugin = DownturnPlugin(
        data_dir=Path(getattr(args, "data_dir", None) or "backtests/momentum/data/raw"),
        initial_equity=getattr(args, "equity", 100_000.0),
        max_workers=getattr(args, "max_workers", None),
    )
    runner = PhaseRunner(
        plugin=plugin,
        output_dir=output_dir,
        max_rounds=50,
        max_retries=getattr(args, "max_retries", 2),
    )
    state = runner.run_all_phases()
    logger.info("All downturn phases complete. Final cumulative mutations: %s", state.cumulative_mutations)


def cmd_run(args):
    """Route to strategy-specific run."""
    if args.strategy == "nqdtc":
        _cmd_run_nqdtc(args)
    elif args.strategy == "vdubus":
        _cmd_run_vdubus(args)
    elif args.strategy == "portfolio":
        _cmd_run_portfolio(args)
    elif args.strategy == "downturn":
        _cmd_run_downturn(args)
    else:
        logger.error("Unknown strategy: %s", args.strategy)


def cmd_ablation(args):
    """Route to strategy-specific ablation."""
    if args.strategy == "nqdtc":
        _cmd_ablation_nqdtc(args)
    elif args.strategy == "vdubus":
        _cmd_ablation_vdubus(args)
    elif args.strategy == "downturn":
        _cmd_ablation_downturn(args)
    else:
        logger.error("Unknown strategy: %s", args.strategy)


def cmd_optimize(args):
    """Route to strategy-specific optimization."""
    if args.strategy == "nqdtc":
        _cmd_optimize_nqdtc(args)
    elif args.strategy == "vdubus":
        _cmd_optimize_vdubus(args)
    else:
        logger.error("Unknown strategy: %s", args.strategy)


def cmd_walk_forward(args):
    """Route to strategy-specific walk-forward."""
    if args.strategy == "nqdtc":
        _cmd_walk_forward_nqdtc(args)
    elif args.strategy == "vdubus":
        _cmd_walk_forward_vdubus(args)
    else:
        logger.error("Unknown strategy: %s", args.strategy)


def cmd_sweep(args):
    """Route to strategy-specific sweep."""
    if args.strategy == "portfolio":
        _cmd_sweep_portfolio(args)
    else:
        logger.error("Sweep only supported for --strategy portfolio")


# ---------------------------------------------------------------------------
# Shared output formatting
# ---------------------------------------------------------------------------

def _print_optimization_results(result):
    print(f"\n=== Optimization Results ===")
    print(f"Best score: {result.best_score:.4f}")
    print(f"Best params:")
    for k, v in sorted(result.best_params.items()):
        print(f"  {k}: {v}")

    if result.all_sorted:
        print(f"\nTop 10 trials:")
        for i, tr in enumerate(result.all_sorted[:10]):
            print(
                f"  #{i+1}: score={tr.score:.4f}  "
                f"CAGR={tr.cagr:.1%}  Sharpe={tr.sharpe:.2f}  "
                f"PF={tr.profit_factor:.2f}  MaxDD={tr.max_dd:.1%}  "
                f"trades/mo={tr.trades_per_month:.1f}"
            )


def _print_walk_forward_results(result):
    print(f"\n=== Walk-Forward Results ===")
    print(f"Folds: {len(result.folds)}")
    print(f"Avg test score: {result.avg_test_score:.4f}")
    print(f"Avg test Sharpe: {result.avg_test_sharpe:.2f}")
    print(f"Positive folds: {result.pct_positive_folds:.0f}%")
    print(f"Degradation ratio: {result.degradation_ratio:.2f}")
    print(f"Robustness: {'PASS' if result.passed else 'FAIL'}")
    if result.failure_reasons:
        for reason in result.failure_reasons:
            print(f"  - {reason}")

    print(f"\nPer-fold results:")
    for fold in result.folds:
        test_sharpe = fold.test_metrics.sharpe if fold.test_metrics else 0
        print(
            f"  Fold {fold.fold_id}: "
            f"train={fold.train_start.date()}-{fold.train_end.date()} "
            f"test={fold.test_start.date()}-{fold.test_end.date()} "
            f"train_score={fold.train_score:.4f} "
            f"test_score={fold.test_score:.4f} "
            f"test_sharpe={test_sharpe:.2f} "
            f"trades={fold.test_trades}"
        )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Auto research pipeline
# ---------------------------------------------------------------------------

def _cmd_auto(args):
    """Run the automated research pipeline."""
    from backtests.momentum.auto.runners.run_full_pipeline import main as pipeline_main

    pipeline_main(
        phase=args.phase,
        strategy_filter=args.auto_strategy,
        resume=not args.no_resume,
        max_workers=args.max_workers,
        equity=args.equity,
        data_dir=args.data_dir,
        experiment_ids=args.experiment_ids,
        skip_robustness=args.skip_robustness,
    )


# Main parser
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="backtest",
        description="Multi-Strategy Backtesting Framework (NQDTC v2.0 / VdubusNQ v4.0 / Downturn Dominator)",
    )
    parser.add_argument(
        "--strategy", "-s",
        choices=["nqdtc", "vdubus", "portfolio", "downturn"],
        default="vdubus",
        help="Strategy to backtest (default: vdubus)",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # download
    dl = sub.add_parser("download", help="Download historical data from IBKR")
    dl.add_argument("--symbols", default=None)
    dl.add_argument("--duration", default="5 Y")
    dl.add_argument("--data-dir", default="backtest/data/raw")

    # run
    run = sub.add_parser("run", help="Run a single backtest")
    run.add_argument("--symbols", default=None)
    run.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    run.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    run.add_argument("--equity", type=float, default=100_000)
    run.add_argument("--fixed-qty", type=int, default=10, help="Fixed position size in contracts (default: 10 MNQ, pass 0 for risk-based sizing)")
    run.add_argument("--data-dir", default="backtest/data/raw")
    run.add_argument("--diagnostics", action="store_true", default=False,
                     help="Print extended diagnostic reports")
    run.add_argument("--charts", action="store_true", default=False,
                     help="Generate candlestick chart PNGs")
    run.add_argument("--chart-dir", default="backtest/output/charts",
                     help="Directory for chart output (default: backtest/output/charts)")
    run.add_argument("--report-file", default=None,
                     help="Write all report output to this file")
    run.add_argument("--preset", default="10k_v6",
                     help="Portfolio preset name (default: 10k_v6). Use with --strategy portfolio.")

    # ablation
    ab = sub.add_parser("ablation", help="Run ablation test")
    ab.add_argument("--filter", default=None, help="Single filter name to ablate")
    ab.add_argument("--filters", default=None, help="Comma-separated filter names for multi-filter ablation")
    ab.add_argument("--symbols", default=None)
    ab.add_argument("--equity", type=float, default=100_000)
    ab.add_argument("--data-dir", default="backtest/data/raw")

    # optimize
    opt = sub.add_parser("optimize", help="Run parameter optimization")
    opt.add_argument("--symbols", default=None)
    opt.add_argument("--n-coarse", type=int, default=1000)
    opt.add_argument("--n-refine", type=int, default=300)
    opt.add_argument("--equity", type=float, default=100_000)
    opt.add_argument("--data-dir", default="backtest/data/raw")

    # walk-forward
    wf = sub.add_parser("walk-forward", help="Run walk-forward validation")
    wf.add_argument("--symbols", default=None)
    wf.add_argument("--test-months", type=int, default=12)
    wf.add_argument("--equity", type=float, default=100_000)
    wf.add_argument("--data-dir", default="backtest/data/raw")

    # sweep (portfolio parameter sweep)
    sw = sub.add_parser("sweep", help="Run portfolio parameter sweep")
    sw.add_argument("--preset", default="10k_v6",
                    help="Portfolio preset name (default: 10k_v6)")
    sw.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    sw.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    sw.add_argument("--data-dir", default="backtest/data/raw")
    sw.add_argument("--report-file", default=None,
                    help="Write sweep report to this file")

    # weakness-report subcommand
    wr = sub.add_parser("weakness-report", help="Generate unified weakness report across all strategies")
    wr.add_argument("--equity", type=float, default=100_000)
    wr.add_argument("--data-dir", default="backtest/data/raw")
    wr.add_argument("--report-file", default=None)
    wr.add_argument("--preset", default="10k_v6")

    # Auto research pipeline
    auto = sub.add_parser("auto", help="Automated research pipeline")
    auto.add_argument("--phase", choices=["experiments", "greedy", "diagnostics", "comparison", "full"],
                       default="full", help="Pipeline phase to run")
    auto.add_argument("--strategy", choices=["nqdtc", "vdubus", "portfolio", "all"],
                       default="all", dest="auto_strategy", help="Strategy filter")
    auto.add_argument("--experiment-ids", nargs="*", help="Specific experiment IDs to run")
    auto.add_argument("--skip-robustness", action="store_true", help="Skip robustness checks")
    auto.add_argument("--no-resume", action="store_true", help="Start fresh (ignore previous results)")
    auto.add_argument("--equity", type=float, default=10_000.0, help="Initial equity")
    auto.add_argument("--max-workers", type=int, default=None,
                       help="Parallel workers for experiments and greedy phases")
    auto.add_argument("--data-dir", default=None, help="Path to raw data dir (default: auto-detected)")

    # Phase commands (downturn phased auto-optimization)
    phase_run = sub.add_parser("phase-run", help="Run single phase of downturn greedy optimization")
    phase_run.add_argument("--phase", type=int, required=True, dest="phase_num",
                           choices=[1, 2, 3, 4, 5], help="Phase number (1-5)")
    phase_run.add_argument("--equity", type=float, default=100_000.0)
    phase_run.add_argument("--max-workers", type=int, default=4)
    phase_run.add_argument("--data-dir", default="backtests/momentum/data/raw")

    phase_auto = sub.add_parser("phase-auto", help="Auto-run phases 1-5 with shared analyzer loop")
    phase_auto.add_argument("--max-retries", type=int, default=2)
    phase_auto.add_argument("--equity", type=float, default=100_000.0)
    phase_auto.add_argument("--max-workers", type=int, default=4)
    phase_auto.add_argument("--data-dir", default="backtests/momentum/data/raw")

    args = parser.parse_args()

    # Resolve default symbols if not specified
    if hasattr(args, 'symbols') and args.symbols is None:
        args.symbols = ",".join(_default_symbols(args.strategy))

    if args.command == "download":
        cmd_download(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "ablation":
        cmd_ablation(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "walk-forward":
        cmd_walk_forward(args)
    elif args.command == "sweep":
        cmd_sweep(args)
    elif args.command == "weakness-report":
        _cmd_weakness_report(args)
    elif args.command == "auto":
        _cmd_auto(args)
    elif args.command == "phase-run":
        _cmd_phase_run_downturn(args)
    elif args.command == "phase-auto":
        _cmd_phase_auto_downturn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
