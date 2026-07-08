"""CLI entry points for the backtesting framework.

Usage:
    python -m backtest.cli download --duration "5 Y"
    python -m backtest.cli run --start 2020-01-01 --end 2024-12-31
    python -m backtest.cli run --strategy helix --symbols QQQ,GLD
    python -m backtest.cli ablation --filter momentum_filter
    python -m backtest.cli ablation --strategy helix --filter disable_class_a
    python -m backtest.cli optimize --n-coarse 1000 --n-refine 300
    python -m backtest.cli walk-forward --test-months 12

Symbol set is controlled via env vars:
    ATRSS: ATRSS_SYMBOL_SET=etf (default) -> QQQ, GLD
    Helix: AKCHELIX_SYMBOL_SET=etf (default) -> QQQ, GLD
Or override per-command: --symbols QQQ,GLD
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _default_symbols(strategy: str) -> list[str]:
    """Resolve default symbols based on strategy."""
    if strategy == "helix":
        try:
            from strategies.swing.akc_helix.config import SYMBOLS
            return list(SYMBOLS)
        except ImportError:
            return ["QQQ", "GLD"]
    else:  # atrss and regime both use ATRSS symbols
        try:
            from strategies.swing.atrss.config import SYMBOLS
            return list(SYMBOLS)
        except ImportError:
            return ["QQQ", "GLD"]


def _get_symbol_configs(strategy: str):
    """Get the SYMBOL_CONFIGS for the given strategy."""
    if strategy == "helix":
        from strategies.swing.akc_helix.config import SYMBOL_CONFIGS
        return SYMBOL_CONFIGS
    else:
        from strategies.swing.atrss.config import SYMBOL_CONFIGS
        return SYMBOL_CONFIGS


def _load_data(symbols: list[str], data_dir: Path):
    """Load cached parquet data into PortfolioData (ATRSS)."""
    from backtests.swing.data.cache import load_bars
    from backtests.swing.data.preprocessing import (
        align_daily_to_hourly,
        build_numpy_arrays,
        filter_rth,
        normalize_timezone,
    )
    from backtests.swing.engine.portfolio_engine import PortfolioData

    data = PortfolioData()

    for sym in symbols:
        hourly_path = data_dir / f"{sym}_1h.parquet"
        daily_path = data_dir / f"{sym}_1d.parquet"

        if not hourly_path.exists() or not daily_path.exists():
            logger.error("Missing data for %s. Run 'download' first.", sym)
            continue

        h_df = normalize_timezone(load_bars(hourly_path))
        h_df = filter_rth(h_df)
        d_df = normalize_timezone(load_bars(daily_path))

        data.hourly[sym] = build_numpy_arrays(h_df)
        data.daily[sym] = build_numpy_arrays(d_df)
        data.daily_idx_maps[sym] = align_daily_to_hourly(h_df, d_df)

        logger.info(
            "Loaded %s: %d hourly bars, %d daily bars",
            sym, len(h_df), len(d_df),
        )

    return data


def _load_helix_data(symbols: list[str], data_dir: Path):
    """Load cached parquet data into HelixPortfolioData (includes 4H)."""
    from backtests.swing.engine.helix_portfolio_engine import load_helix_data
    return load_helix_data(symbols, data_dir)


def cmd_download(args):
    """Download historical data from IBKR."""
    from backtests.swing.data.downloader import download_all_symbols

    symbols = args.symbols.split(",")
    data_dir = Path(args.data_dir)
    duration = args.duration
    SYMBOL_CONFIGS = _get_symbol_configs(args.strategy)

    async def _run():
        result = await download_all_symbols(
            symbols=symbols,
            configs=SYMBOL_CONFIGS,
            duration=duration,
            output_dir=data_dir,
        )
        for sym, paths in result.items():
            for tf, path in paths.items():
                logger.info("Downloaded %s %s -> %s", sym, tf, path)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# ATRSS commands
# ---------------------------------------------------------------------------

def _cmd_run_atrss(args):
    """Run a single ATRSS backtest."""
    from backtests.swing.analysis.metrics import compute_buy_and_hold, compute_metrics
    from backtests.swing.analysis.reports import (
        behavior_report,
        buy_and_hold_report,
        diagnostic_report,
        format_summary,
        performance_report,
    )
    from backtests.swing.config import BacktestConfig, SlippageConfig
    from backtests.swing.engine.portfolio_engine import run_synchronized
    from strategies.swing.atrss.config import SYMBOL_CONFIGS

    symbols = args.symbols.split(",")
    data_dir = Path(args.data_dir)
    data = _load_data(symbols, data_dir)

    all_stk = all(
        SYMBOL_CONFIGS.get(s, None) is not None and SYMBOL_CONFIGS[s].sec_type == "STK"
        for s in symbols
    )

    fixed_qty = args.fixed_qty
    if fixed_qty is None and all_stk:
        fixed_qty = 10
        logger.info("ETF mode detected: defaulting to fixed_qty=10, commission=$1.00")

    slippage = SlippageConfig()
    if all_stk and fixed_qty is not None:
        slippage = SlippageConfig(commission_per_contract=1.00)

    config = BacktestConfig(
        symbols=symbols,
        start_date=datetime.fromisoformat(args.start) if args.start else None,
        end_date=datetime.fromisoformat(args.end) if args.end else None,
        initial_equity=args.equity,
        data_dir=data_dir,
        slippage=slippage,
        fixed_qty=fixed_qty,
    )

    result = run_synchronized(data, config)
    report_sections: list[str] = []

    for sym, sr in result.symbol_results.items():
        if not sr.trades:
            logger.info("%s: No trades", sym)
            continue

        pnls = np.array([t.pnl_dollars for t in sr.trades])
        cfg = SYMBOL_CONFIGS[sym]
        risks = np.array([abs(t.entry_price - t.initial_stop) * cfg.multiplier * t.qty for t in sr.trades])
        holds = np.array([t.bars_held for t in sr.trades])
        comms = np.array([t.commission for t in sr.trades])

        metrics = compute_metrics(
            pnls, risks, holds, comms,
            sr.equity_curve, sr.timestamps, config.initial_equity,
        )

        report_sections.append(performance_report(sr, metrics))
        report_sections.append(behavior_report(sr.trades))
        report_sections.append(diagnostic_report(sr))
        report_sections.append(format_summary(metrics))

        if sym in data.daily:
            daily_closes = data.daily[sym].closes
            if len(sr.timestamps) >= 2:
                delta = sr.timestamps[-1] - sr.timestamps[0]
                if hasattr(delta, 'astype'):
                    span_s = float(delta / np.timedelta64(1, 's'))
                else:
                    span_s = delta.total_seconds()
                years = span_s / (365.25 * 24 * 3600)
            else:
                years = 1.0
            bh = compute_buy_and_hold(
                sym, daily_closes, years,
                qty=fixed_qty or 10,
                multiplier=cfg.multiplier,
                initial_equity=config.initial_equity,
            )
            report_sections.append(buy_and_hold_report(bh, metrics))

        # Extended ATRSS diagnostics
        if getattr(args, 'diagnostics', False):
            from backtests.swing.analysis.atrss_diagnostics import (
                atrss_addon_analysis,
                atrss_bias_alignment,
                atrss_breakout_arm_diagnostic,
                atrss_entry_type_drilldown,
                atrss_exit_analysis,
                atrss_filter_rejection_detail,
                atrss_losing_trade_detail,
                atrss_mfe_cohort_segmentation,
                atrss_order_fill_rate,
                atrss_position_occupancy,
                atrss_r_curve,
                atrss_regime_time_report,
                atrss_signal_funnel,
                atrss_stop_efficiency,
                atrss_streak_analysis,
                atrss_time_analysis,
            )
            # Signal funnel diagnostics (context first)
            shadow_rej = sum(s.rejected_count for s in result.filter_summary.values()) if result.filter_summary else 0
            report_sections.append(atrss_signal_funnel(sr.funnel, len(sr.trades), shadow_rej))
            report_sections.append(atrss_regime_time_report(sr.funnel, sr))
            report_sections.append(atrss_position_occupancy(sr.trades, sr.funnel))
            report_sections.append(atrss_filter_rejection_detail(result.filter_summary))
            # Existing diagnostics
            report_sections.append(atrss_entry_type_drilldown(sr.trades))
            report_sections.append(atrss_exit_analysis(sr.trades))
            report_sections.append(atrss_bias_alignment(sr.trades, sr))
            report_sections.append(atrss_stop_efficiency(sr.trades))
            report_sections.append(atrss_time_analysis(sr.trades))
            report_sections.append(atrss_losing_trade_detail(sr.trades))
            report_sections.append(atrss_r_curve(sr.trades))
            report_sections.append(atrss_streak_analysis(sr.trades))
            report_sections.append(atrss_addon_analysis(sr.trades))
            report_sections.append(atrss_mfe_cohort_segmentation(sr.trades))
            report_sections.append(atrss_breakout_arm_diagnostic(sr.funnel))
            report_sections.append(atrss_order_fill_rate(sr.order_metadata))

        # Candlestick charts
        if getattr(args, 'charts', False):
            from backtests.swing.analysis.charts import generate_backtest_charts
            chart_dir = Path(getattr(args, 'chart_dir', 'backtest/output/charts'))
            daily_bars = data.daily.get(sym)
            hourly_bars = data.hourly.get(sym)
            paths = generate_backtest_charts(
                sym, daily_bars, hourly_bars, sr.trades,
                chart_dir, strategy_label="atrss",
            )
            for p in paths:
                logger.info("Chart saved: %s", p)

    # Print all sections to stdout
    for section in report_sections:
        print(f"\n{section}")

    # Write to file if requested
    report_file = getattr(args, 'report_file', None)
    if report_file and report_sections:
        report_path = Path(report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n\n".join(report_sections) + "\n", encoding="utf-8")
        logger.info("Report saved: %s", report_path)


def _cmd_ablation_atrss(args):
    """Run ATRSS ablation test."""
    from backtests.swing.analysis.metrics import compute_metrics
    from backtests.swing.analysis.reports import print_summary
    from backtests.swing.config import AblationFlags, BacktestConfig, SlippageConfig
    from backtests.swing.engine.portfolio_engine import run_synchronized
    from strategies.swing.atrss.config import SYMBOL_CONFIGS

    symbols = args.symbols.split(",")
    data_dir = Path(args.data_dir)
    data = _load_data(symbols, data_dir)

    filter_names: list[str] = []
    if args.filters:
        filter_names = [f.strip() for f in args.filters.split(",") if f.strip()]
    elif args.filter:
        filter_names = [args.filter]
    else:
        logger.error("Must specify --filter or --filters")
        return

    flags_template = AblationFlags()
    valid_flags = [f for f in vars(flags_template) if not f.startswith("_")]
    special_variants = {"use_stop_market"}
    for fn in filter_names:
        if fn not in valid_flags and fn not in special_variants:
            logger.error("Unknown filter: %s", fn)
            logger.info("Valid filters: %s", valid_flags + list(special_variants))
            return

    baseline_config = BacktestConfig(
        symbols=symbols,
        initial_equity=args.equity,
        data_dir=data_dir,
    )
    baseline = run_synchronized(data, baseline_config)

    flags = AblationFlags()
    slippage = SlippageConfig()
    for fn in filter_names:
        if fn == "use_stop_market":
            slippage = SlippageConfig(use_stop_market=True)
        elif hasattr(flags, fn):
            setattr(flags, fn, False)

    ablation_config = BacktestConfig(
        symbols=symbols,
        initial_equity=args.equity,
        flags=flags,
        slippage=slippage,
        data_dir=data_dir,
    )
    ablation = run_synchronized(data, ablation_config)

    ablation_label = ",".join(filter_names)
    print(f"\n=== Ablation: {ablation_label} = OFF ===\n")

    for sym in symbols:
        print(f"--- {sym} ---")
        for label, res in [("Baseline", baseline), ("Ablated", ablation)]:
            sr = res.symbol_results.get(sym)
            if not sr or not sr.trades:
                print(f"  {label}: No trades")
                continue
            pnls = np.array([t.pnl_dollars for t in sr.trades])
            sym_cfg = SYMBOL_CONFIGS[sym]
            risks = np.array([abs(t.entry_price - t.initial_stop) * sym_cfg.multiplier * t.qty for t in sr.trades])
            holds = np.array([t.bars_held for t in sr.trades])
            comms = np.array([t.commission for t in sr.trades])
            metrics = compute_metrics(
                pnls, risks, holds, comms,
                sr.equity_curve, sr.timestamps, args.equity,
            )
            print(f"  {label}: ", end="")
            print_summary(metrics)
        print()


def _cmd_optimize_atrss(args):
    """Run ATRSS parameter optimization."""
    from backtests.swing.config import BacktestConfig
    from backtests.swing.optimization.runner import OptimizationRunner

    symbols = args.symbols.split(",")
    data_dir = Path(args.data_dir)
    data = _load_data(symbols, data_dir)

    config = BacktestConfig(
        symbols=symbols,
        initial_equity=args.equity,
        data_dir=data_dir,
        track_shadows=False,
    )

    runner = OptimizationRunner(
        base_config=config,
        data=data,
        n_coarse=args.n_coarse,
        n_refine=args.n_refine,
    )
    result = runner.run()
    _print_optimization_results(result)


def _cmd_walk_forward_atrss(args):
    """Run ATRSS walk-forward validation."""
    from backtests.swing.config import BacktestConfig
    from backtests.swing.optimization.walk_forward import WalkForwardValidator

    symbols = args.symbols.split(",")
    data_dir = Path(args.data_dir)
    data = _load_data(symbols, data_dir)

    config = BacktestConfig(
        symbols=symbols,
        initial_equity=args.equity,
        data_dir=data_dir,
    )

    validator = WalkForwardValidator(
        data=data,
        base_config=config,
        test_window_months=args.test_months,
    )
    result = validator.run()
    _print_walk_forward_results(result)


# ---------------------------------------------------------------------------
# Helix commands
# ---------------------------------------------------------------------------

def _cmd_run_helix(args):
    """Run a single Helix backtest."""
    from backtests.swing.analysis.metrics import compute_buy_and_hold, compute_metrics
    from backtests.swing.analysis.reports import (
        buy_and_hold_report,
        format_summary,
        helix_behavior_report,
        helix_diagnostic_report,
        helix_performance_report,
    )
    from backtests.swing.config import SlippageConfig
    from backtests.swing.config_helix import HelixBacktestConfig
    from backtests.swing.engine.helix_portfolio_engine import run_helix_synchronized
    from strategies.swing.akc_helix.config import SYMBOL_CONFIGS

    symbols = args.symbols.split(",")
    data_dir = Path(args.data_dir)
    data = _load_helix_data(symbols, data_dir)

    all_stk = all(
        SYMBOL_CONFIGS.get(s, None) is not None and SYMBOL_CONFIGS[s].is_etf
        for s in symbols
    )

    fixed_qty = args.fixed_qty
    slippage = SlippageConfig()
    if all_stk:
        slippage = SlippageConfig(commission_per_contract=1.00)
        if fixed_qty is None:
            fixed_qty = 10
            logger.info("ETF mode detected: defaulting to fixed_qty=10, commission=$1.00")

    config = HelixBacktestConfig(
        symbols=symbols,
        start_date=datetime.fromisoformat(args.start) if args.start else None,
        end_date=datetime.fromisoformat(args.end) if args.end else None,
        initial_equity=args.equity,
        data_dir=data_dir,
        slippage=slippage,
        fixed_qty=fixed_qty,
    )

    result = run_helix_synchronized(data, config)
    report_sections: list[str] = []

    for sym, sr in result.symbol_results.items():
        if not sr.trades:
            logger.info("%s: No trades", sym)
            continue

        pnls = np.array([t.pnl_dollars for t in sr.trades])
        cfg = SYMBOL_CONFIGS[sym]
        risks = np.array([abs(t.entry_price - t.initial_stop) * cfg.multiplier * t.qty for t in sr.trades])
        holds = np.array([t.bars_held for t in sr.trades])
        comms = np.array([t.commission for t in sr.trades])

        metrics = compute_metrics(
            pnls, risks, holds, comms,
            sr.equity_curve, sr.timestamps, config.initial_equity,
        )

        report_sections.append(helix_performance_report(sym, metrics))
        report_sections.append(helix_behavior_report(sr.trades))
        report_sections.append(helix_diagnostic_report(sr))
        report_sections.append(format_summary(metrics))

        if sym in data.daily:
            daily_closes = data.daily[sym].closes
            if len(sr.timestamps) >= 2:
                delta = sr.timestamps[-1] - sr.timestamps[0]
                if hasattr(delta, 'astype'):
                    span_s = float(delta / np.timedelta64(1, 's'))
                else:
                    span_s = delta.total_seconds()
                years = span_s / (365.25 * 24 * 3600)
            else:
                years = 1.0
            bh = compute_buy_and_hold(
                sym, daily_closes, years,
                qty=fixed_qty or 10,
                multiplier=cfg.multiplier,
                initial_equity=config.initial_equity,
            )
            report_sections.append(buy_and_hold_report(bh, metrics))

        # Extended Helix diagnostics
        if getattr(args, 'diagnostics', False):
            from backtests.swing.analysis.helix_diagnostics import (
                helix_class_drilldown,
                helix_divergence_quality,
                helix_losing_trade_detail,
                helix_r_curve,
                helix_regime_4h_alignment,
                helix_regime_alignment,
                helix_stop_efficiency,
                helix_streak_analysis,
                helix_time_analysis,
            )
            report_sections.append(helix_class_drilldown(sr.trades))
            report_sections.append(helix_divergence_quality(sr.trades))
            report_sections.append(helix_regime_4h_alignment(sr.trades))
            report_sections.append(helix_regime_alignment(sr.trades, sr))
            report_sections.append(helix_stop_efficiency(sr.trades))
            report_sections.append(helix_time_analysis(sr.trades))
            report_sections.append(helix_losing_trade_detail(sr.trades))
            report_sections.append(helix_r_curve(sr.trades))
            report_sections.append(helix_streak_analysis(sr.trades))

        # Candlestick charts
        if getattr(args, 'charts', False):
            from backtests.swing.analysis.charts import generate_backtest_charts
            chart_dir = Path(getattr(args, 'chart_dir', 'backtest/output/charts'))
            daily_bars = data.daily.get(sym)
            hourly_bars = data.hourly.get(sym)
            paths = generate_backtest_charts(
                sym, daily_bars, hourly_bars, sr.trades,
                chart_dir, strategy_label="helix",
            )
            for p in paths:
                logger.info("Chart saved: %s", p)

    # Print all sections to stdout
    for section in report_sections:
        print(f"\n{section}")

    # Write to file if requested
    report_file = getattr(args, 'report_file', None)
    if report_file and report_sections:
        report_path = Path(report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n\n".join(report_sections) + "\n", encoding="utf-8")
        logger.info("Report saved: %s", report_path)


def _cmd_ablation_helix(args):
    """Run Helix ablation test."""
    from backtests.swing.analysis.metrics import compute_metrics
    from backtests.swing.analysis.reports import print_summary
    from backtests.swing.config_helix import HelixAblationFlags, HelixBacktestConfig
    from backtests.swing.engine.helix_portfolio_engine import run_helix_synchronized
    from strategies.swing.akc_helix.config import SYMBOL_CONFIGS

    symbols = args.symbols.split(",")
    data_dir = Path(args.data_dir)
    data = _load_helix_data(symbols, data_dir)

    filter_names: list[str] = []
    if args.filters:
        filter_names = [f.strip() for f in args.filters.split(",") if f.strip()]
    elif args.filter:
        filter_names = [args.filter]
    else:
        logger.error("Must specify --filter or --filters")
        return

    flags_template = HelixAblationFlags()
    valid_flags = [f for f in vars(flags_template) if not f.startswith("_")]
    for fn in filter_names:
        if fn not in valid_flags:
            logger.error("Unknown Helix filter: %s", fn)
            logger.info("Valid filters: %s", valid_flags)
            return

    baseline_config = HelixBacktestConfig(
        symbols=symbols,
        initial_equity=args.equity,
        data_dir=data_dir,
    )
    baseline = run_helix_synchronized(data, baseline_config)

    flags = HelixAblationFlags()
    for fn in filter_names:
        if hasattr(flags, fn):
            setattr(flags, fn, True)  # True = disabled for Helix flags

    ablation_config = HelixBacktestConfig(
        symbols=symbols,
        initial_equity=args.equity,
        flags=flags,
        data_dir=data_dir,
    )
    ablation = run_helix_synchronized(data, ablation_config)

    ablation_label = ",".join(filter_names)
    print(f"\n=== Helix Ablation: {ablation_label} = ON (disabled) ===\n")

    for sym in symbols:
        print(f"--- {sym} ---")
        for label, res in [("Baseline", baseline), ("Ablated", ablation)]:
            sr = res.symbol_results.get(sym)
            if not sr or not sr.trades:
                print(f"  {label}: No trades")
                continue
            pnls = np.array([t.pnl_dollars for t in sr.trades])
            sym_cfg = SYMBOL_CONFIGS[sym]
            risks = np.array([abs(t.entry_price - t.initial_stop) * sym_cfg.multiplier * t.qty for t in sr.trades])
            holds = np.array([t.bars_held for t in sr.trades])
            comms = np.array([t.commission for t in sr.trades])
            metrics = compute_metrics(
                pnls, risks, holds, comms,
                sr.equity_curve, sr.timestamps, args.equity,
            )
            print(f"  {label}: ", end="")
            print_summary(metrics)
        print()


def _cmd_optimize_helix(args):
    """Run Helix parameter optimization."""
    from backtests.swing.config_helix import HelixBacktestConfig
    from backtests.swing.optimization.helix_runner import HelixOptimizationRunner

    symbols = args.symbols.split(",")
    data_dir = Path(args.data_dir)
    data = _load_helix_data(symbols, data_dir)

    config = HelixBacktestConfig(
        symbols=symbols,
        initial_equity=args.equity,
        data_dir=data_dir,
        track_shadows=False,
    )

    runner = HelixOptimizationRunner(
        base_config=config,
        data=data,
        n_coarse=args.n_coarse,
        n_refine=args.n_refine,
    )
    result = runner.run()
    _print_optimization_results(result)


def _cmd_walk_forward_helix(args):
    """Run Helix walk-forward validation."""
    from backtests.swing.config_helix import HelixBacktestConfig
    from backtests.swing.optimization.helix_walk_forward import HelixWalkForwardValidator

    symbols = args.symbols.split(",")
    data_dir = Path(args.data_dir)
    data = _load_helix_data(symbols, data_dir)

    config = HelixBacktestConfig(
        symbols=symbols,
        initial_equity=args.equity,
        data_dir=data_dir,
    )

    validator = HelixWalkForwardValidator(
        data=data,
        base_config=config,
        test_window_months=args.test_months,
    )
    result = validator.run()
    _print_walk_forward_results(result)


# ---------------------------------------------------------------------------
# Regime commands
# ---------------------------------------------------------------------------

def _cmd_run_regime(args):
    """Run a single regime-following backtest."""
    from backtests.swing.analysis.metrics import compute_buy_and_hold, compute_metrics
    from backtests.swing.analysis.reports import (
        behavior_report,
        buy_and_hold_report,
        format_summary,
        performance_report,
    )
    from backtests.swing.config import SlippageConfig
    from backtests.swing.config_regime import RegimeConfig
    from backtests.swing.engine.regime_engine import run_regime_independent
    from strategies.swing.atrss.config import SYMBOL_CONFIGS

    symbols = args.symbols.split(",")
    data_dir = Path(args.data_dir)
    data = _load_data(symbols, data_dir)

    all_stk = all(
        SYMBOL_CONFIGS.get(s, None) is not None and SYMBOL_CONFIGS[s].sec_type == "STK"
        for s in symbols
    )

    fixed_qty = args.fixed_qty
    if fixed_qty is None and all_stk:
        fixed_qty = 10
        logger.info("ETF mode detected: defaulting to fixed_qty=10, commission=$1.00")

    slippage = SlippageConfig()
    if all_stk and fixed_qty is not None:
        slippage = SlippageConfig(commission_per_contract=1.00)

    config = RegimeConfig(
        symbols=symbols,
        start_date=datetime.fromisoformat(args.start) if args.start else None,
        end_date=datetime.fromisoformat(args.end) if args.end else None,
        initial_equity=args.equity,
        data_dir=data_dir,
        slippage=slippage,
        fixed_qty=fixed_qty,
        chand_mult=args.chand_mult,
        regime_downgrade_exit=args.regime_downgrade_exit,
        time_exit_hours=args.time_exit_hours,
        shorts_enabled=args.shorts,
    )

    result = run_regime_independent(data, config)
    report_sections: list[str] = []

    for sym, sr in result.symbol_results.items():
        if not sr.trades:
            logger.info("%s: No trades", sym)
            continue

        pnls = np.array([t.pnl_dollars for t in sr.trades])
        cfg = SYMBOL_CONFIGS[sym]
        risks = np.array([abs(t.entry_price - t.initial_stop) * cfg.multiplier * t.qty for t in sr.trades])
        holds = np.array([t.bars_held for t in sr.trades])
        comms = np.array([t.commission for t in sr.trades])

        metrics = compute_metrics(
            pnls, risks, holds, comms,
            sr.equity_curve, sr.timestamps, config.initial_equity,
        )

        report_sections.append(performance_report(sr, metrics))
        report_sections.append(behavior_report(sr.trades))
        report_sections.append(format_summary(metrics))

        if sym in data.daily:
            daily_closes = data.daily[sym].closes
            if len(sr.timestamps) >= 2:
                delta = sr.timestamps[-1] - sr.timestamps[0]
                if hasattr(delta, 'astype'):
                    span_s = float(delta / np.timedelta64(1, 's'))
                else:
                    span_s = delta.total_seconds()
                years = span_s / (365.25 * 24 * 3600)
            else:
                years = 1.0
            bh = compute_buy_and_hold(
                sym, daily_closes, years,
                qty=fixed_qty or 10,
                multiplier=cfg.multiplier,
                initial_equity=config.initial_equity,
            )
            report_sections.append(buy_and_hold_report(bh, metrics))

    for section in report_sections:
        print(f"\n{section}")

    report_file = getattr(args, 'report_file', None)
    if report_file and report_sections:
        report_path = Path(report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n\n".join(report_sections) + "\n", encoding="utf-8")
        logger.info("Report saved: %s", report_path)


# ---------------------------------------------------------------------------
# Investigation commands (structural evaluation)
# ---------------------------------------------------------------------------

def _run_investigation_backtest(symbols, data_dir, equity, fixed_qty=None):
    """Shared: run ATRSS backtest and return (data, config, result, engines_info)."""
    from backtests.swing.config import BacktestConfig, SlippageConfig
    from backtests.swing.engine.portfolio_engine import PortfolioData, run_independent
    from strategies.swing.atrss.config import SYMBOL_CONFIGS

    data = _load_data(symbols, data_dir)

    all_stk = all(
        SYMBOL_CONFIGS.get(s) is not None and SYMBOL_CONFIGS[s].sec_type == "STK"
        for s in symbols
    )
    if fixed_qty is None and all_stk:
        fixed_qty = 10
    slippage = SlippageConfig()
    if all_stk and fixed_qty is not None:
        slippage = SlippageConfig(commission_per_contract=1.00)

    config = BacktestConfig(
        symbols=symbols,
        initial_equity=equity,
        data_dir=data_dir,
        slippage=slippage,
        fixed_qty=fixed_qty,
    )

    # We need access to engine internals (daily_state_by_idx), so run manually
    from backtests.swing.engine.backtest_engine import BacktestEngine, _AblationPatch
    from backtests.swing.analysis.shadow_tracker import ShadowTracker

    engines: dict[str, BacktestEngine] = {}
    results_by_sym = {}
    shadow = ShadowTracker() if config.track_shadows else None

    with _AblationPatch(config.flags, config.param_overrides):
        for sym in symbols:
            if sym not in data.hourly or sym not in data.daily:
                continue
            cfg = SYMBOL_CONFIGS.get(sym)
            if cfg is None:
                continue
            pv = cfg.multiplier
            engine = BacktestEngine(symbol=sym, cfg=cfg, bt_config=config, point_value=pv)
            if shadow:
                engine.on_rejection = shadow.record_rejection
            engines[sym] = engine
            results_by_sym[sym] = engine.run(
                daily=data.daily[sym],
                hourly=data.hourly[sym],
                daily_idx_map=data.daily_idx_maps[sym],
            )

    return data, config, results_by_sym, engines


def cmd_hold_time(args):
    """Investigation 5: Hold-time vs R analysis."""
    from backtests.swing.analysis.hold_time_analysis import (
        format_hold_time_report,
        hold_time_analysis,
    )

    symbols = args.symbols.split(",")
    data_dir = Path(args.data_dir)

    data, config, results_by_sym, engines = _run_investigation_backtest(
        symbols, data_dir, args.equity,
    )

    all_results = {}
    for sym, sr in results_by_sym.items():
        if sr.trades:
            all_results[sym] = hold_time_analysis(sr.trades)

    print(format_hold_time_report(all_results))


def cmd_exit_hyp(args):
    """Investigation 1: Exit hypotheticals."""
    from backtests.swing.analysis.exit_hypotheticals import (
        format_exit_hypotheticals_report,
        simulate_exit_hypotheticals,
    )
    from strategies.swing.atrss.config import SYMBOL_CONFIGS

    symbols = args.symbols.split(",")
    data_dir = Path(args.data_dir)

    data, config, results_by_sym, engines = _run_investigation_backtest(
        symbols, data_dir, args.equity,
    )

    all_stats = {}
    for sym, sr in results_by_sym.items():
        if not sr.trades:
            continue
        cfg = SYMBOL_CONFIGS[sym]
        daily_states = engines[sym]._daily_state_by_idx
        daily_idx_map = data.daily_idx_maps[sym]
        all_stats[sym] = simulate_exit_hypotheticals(
            sr.trades, data.hourly[sym], daily_states, daily_idx_map, cfg,
        )

    print(format_exit_hypotheticals_report(all_stats))


def cmd_regime_bh(args):
    """Investigation 3: Regime-filtered buy-and-hold benchmark."""
    from backtests.swing.analysis.regime_benchmark import (
        compute_regime_benchmark,
        format_regime_benchmark_report,
    )

    symbols = args.symbols.split(",")
    data_dir = Path(args.data_dir)

    data, config, results_by_sym, engines = _run_investigation_backtest(
        symbols, data_dir, args.equity,
    )

    results = {}
    for sym, sr in results_by_sym.items():
        daily_states = engines[sym]._daily_state_by_idx
        results[sym] = compute_regime_benchmark(
            sym, data.daily[sym], daily_states, sr.trades,
        )

    print(format_regime_benchmark_report(results))


def cmd_chand_sweep(args):
    """Investigation 4: Chandelier multiplier sweep."""
    from backtests.swing.analysis.metrics import compute_metrics
    from backtests.swing.config import BacktestConfig, SlippageConfig
    from backtests.swing.engine.portfolio_engine import run_independent
    from strategies.swing.atrss.config import SYMBOL_CONFIGS

    symbols = args.symbols.split(",")
    data_dir = Path(args.data_dir)
    data = _load_data(symbols, data_dir)

    mults = [float(x) for x in args.mults.split(",")]

    all_stk = all(
        SYMBOL_CONFIGS.get(s) is not None and SYMBOL_CONFIGS[s].sec_type == "STK"
        for s in symbols
    )
    fixed_qty = 10 if all_stk else None
    slippage = SlippageConfig()
    if all_stk:
        slippage = SlippageConfig(commission_per_contract=1.00)

    print("=" * 100)
    print("INVESTIGATION 4: CHANDELIER MULTIPLIER SWEEP")
    print("=" * 100)

    for sym in symbols:
        if sym not in data.hourly:
            continue
        cfg = SYMBOL_CONFIGS.get(sym)
        if cfg is None:
            continue

        print(f"\n--- {sym} (baseline chand_mult={cfg.chand_mult}) ---")
        print(
            f"{'chand_mult':>10} {'Trades':>7} {'WR%':>6} {'MeanR':>8} "
            f"{'TotalR':>8} {'PF':>6} {'NetPnL':>10} {'MFE':>8} "
            f"{'MFECap%':>8} {'AvgBars':>8}"
        )
        print("-" * 95)

        for mult in mults:
            overrides = {f"chand_mult_{sym}": mult, "chand_mult": mult}
            bt_config = BacktestConfig(
                symbols=[sym],
                initial_equity=args.equity,
                data_dir=data_dir,
                slippage=slippage,
                fixed_qty=fixed_qty,
                param_overrides=overrides,
            )
            result = run_independent(data, bt_config)
            sr = result.symbol_results.get(sym)
            if not sr or not sr.trades:
                print(f"{mult:>10.1f} {'no trades':>7}")
                continue

            trades = sr.trades
            rs = np.array([t.r_multiple for t in trades])
            mfes = np.array([t.mfe_r for t in trades])
            bars = np.array([t.bars_held for t in trades])
            pnls = np.array([t.pnl_dollars for t in trades])

            n = len(trades)
            wins = int(np.sum(rs > 0))
            wr = wins / n * 100
            mean_r = float(np.mean(rs))
            total_r = float(np.sum(rs))
            mean_mfe = float(np.mean(mfes))
            mfe_cap = mean_r / mean_mfe * 100 if mean_mfe > 0 else 0
            avg_bars = float(np.mean(bars))
            net_pnl = float(np.sum(pnls))

            gross_p = float(np.sum(rs[rs > 0]))
            gross_l = float(np.abs(np.sum(rs[rs < 0])))
            pf = gross_p / gross_l if gross_l > 0 else float('inf')
            pf_str = f"{pf:>6.2f}" if pf < 100 else "   inf"

            marker = " <-- baseline" if abs(mult - cfg.chand_mult) < 0.01 else ""
            print(
                f"{mult:>10.1f} {n:>7} {wr:>5.1f}% {mean_r:>+8.3f} "
                f"{total_r:>+8.2f} {pf_str} {net_pnl:>+10.2f} "
                f"{mean_mfe:>8.3f} {mfe_cap:>7.1f}% {avg_bars:>8.1f}{marker}"
            )


# ---------------------------------------------------------------------------
# Command dispatchers
# ---------------------------------------------------------------------------

def _cmd_weakness_report(args):
    """Generate unified swing weakness report by running the portfolio engine."""
    from backtests.swing.engine.unified_portfolio_engine import run_unified, load_unified_data
    from backtests.swing.config_unified import UnifiedBacktestConfig
    from backtests.swing.analysis.weakness_report import swing_weakness_report
    from backtests.swing.analysis.portfolio_diagnostics import portfolio_diagnostic_report
    from backtests.swing.analysis.drawdown_attribution import drawdown_attribution_report

    logger.info("Running unified portfolio for weakness report...")

    # Run the full portfolio
    config = UnifiedBacktestConfig(
        initial_equity=args.equity,
        data_dir=Path(args.data_dir),
        start_date=getattr(args, 'start', None),
        end_date=getattr(args, 'end', None),
    )
    data = load_unified_data(config)
    result = run_unified(data, config)

    # Build trade dicts for portfolio diagnostics
    all_trades = {
        "ATRSS": result.atrss_trades,
        "Helix": result.helix_trades,
    }

    report_sections = []

    # 1. Weakness report
    report_sections.append(swing_weakness_report(
        atrss_result=type('R', (), {'trades': result.atrss_trades})(),
        helix_result=type('R', (), {'trades': result.helix_trades})(),
        portfolio_result=result,
    ))

    # 2. Portfolio diagnostics
    report_sections.append(portfolio_diagnostic_report(
        result,
        all_trades=all_trades,
        heat_rejections=result.heat_rejections,
        coordination_events=result.coordination_events,
    ))

    # 3. Drawdown attribution
    report_sections.append(drawdown_attribution_report(
        [t for trades in all_trades.values() for t in trades],
        result.combined_equity,
        result.combined_timestamps,
        strategy_labels=list(all_trades.keys()),
    ))

    output = "\n\n".join(s for s in report_sections if s)
    print(output)

    if getattr(args, 'report_file', None):
        Path(args.report_file).write_text(output)
        logger.info("Weakness report written to %s", args.report_file)


def cmd_auto(args):
    """Run automated experiment harness."""
    from backtests.swing.auto.harness import SwingAutoHarness

    harness = SwingAutoHarness(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        initial_equity=args.equity,
    )
    harness.run_all(
        strategy_filter=args.strategy,
        experiment_ids=args.experiments,
        skip_robustness=args.skip_robustness,
        resume=args.resume,
    )


def cmd_run(args):
    """Route to ATRSS, Helix, or Regime run."""
    if args.strategy == "helix":
        _cmd_run_helix(args)
    elif args.strategy == "regime":
        _cmd_run_regime(args)
    else:
        _cmd_run_atrss(args)


def cmd_ablation(args):
    """Route to ATRSS or Helix ablation."""
    if args.strategy == "helix":
        _cmd_ablation_helix(args)
    else:
        _cmd_ablation_atrss(args)


def cmd_optimize(args):
    """Route to ATRSS or Helix optimization."""
    if args.strategy == "helix":
        _cmd_optimize_helix(args)
    else:
        _cmd_optimize_atrss(args)


def cmd_walk_forward(args):
    """Route to ATRSS or Helix walk-forward."""
    if args.strategy == "helix":
        _cmd_walk_forward_helix(args)
    else:
        _cmd_walk_forward_atrss(args)


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
# Main parser
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="backtest",
        description="Swing Backtesting Framework (ATRSS v4.5 / AKC-Helix v1.4)",
    )
    parser.add_argument(
        "--strategy", "-s",
        choices=["atrss", "helix", "regime"],
        default="atrss",
        help="Strategy to backtest (default: atrss)",
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
    run.add_argument("--fixed-qty", type=int, default=None, help="Fixed position size (overrides risk-based sizing)")
    run.add_argument("--data-dir", default="backtest/data/raw")
    run.add_argument("--diagnostics", action="store_true", default=False,
                     help="Print extended Helix diagnostic reports")
    run.add_argument("--charts", action="store_true", default=False,
                     help="Generate candlestick chart PNGs")
    run.add_argument("--chart-dir", default="backtest/output/charts",
                     help="Directory for chart output (default: backtest/output/charts)")
    run.add_argument("--report-file", default=None,
                     help="Write all report output to this file")
    # Regime strategy flags
    run.add_argument("--chand-mult", type=float, default=1.5,
                     help="Chandelier trailing multiplier (regime strategy, default: 1.5)")
    run.add_argument("--regime-downgrade-exit", action="store_true", default=True,
                     help="Exit on regime downgrade (default: True)")
    run.add_argument("--no-regime-downgrade-exit", dest="regime_downgrade_exit",
                     action="store_false",
                     help="Disable regime downgrade exit")
    run.add_argument("--time-exit-hours", type=int, default=100,
                     help="Forced exit if below min R after N hours (regime strategy, default: 100)")
    run.add_argument("--shorts", action="store_true", default=False,
                     help="Enable short trades (regime strategy, default: disabled)")
    run.add_argument("--weakness-report", action="store_true", default=False,
                     help="Generate unified weakness report across all strategies")

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

    # hold-time (Investigation 5)
    ht = sub.add_parser("hold-time", help="Investigation 5: Hold-time vs R analysis")
    ht.add_argument("--symbols", default=None)
    ht.add_argument("--equity", type=float, default=100_000)
    ht.add_argument("--data-dir", default="backtest/data/raw")

    # exit-hyp (Investigation 1)
    eh = sub.add_parser("exit-hyp", help="Investigation 1: Exit hypothetical simulator")
    eh.add_argument("--symbols", default=None)
    eh.add_argument("--equity", type=float, default=100_000)
    eh.add_argument("--data-dir", default="backtest/data/raw")

    # regime-bh (Investigation 3)
    rb = sub.add_parser("regime-bh", help="Investigation 3: Regime-filtered B&H benchmark")
    rb.add_argument("--symbols", default=None)
    rb.add_argument("--equity", type=float, default=100_000)
    rb.add_argument("--data-dir", default="backtest/data/raw")

    # chand-sweep (Investigation 4)
    cs = sub.add_parser("chand-sweep", help="Investigation 4: Chandelier multiplier sweep")
    cs.add_argument("--symbols", default=None)
    cs.add_argument("--equity", type=float, default=100_000)
    cs.add_argument("--data-dir", default="backtest/data/raw")
    cs.add_argument("--mults", default="1.0,1.2,1.5,1.8,2.2,2.5,3.0,3.2",
                    help="Comma-separated chandelier multipliers to test")

    # auto (automated experiment harness)
    auto = sub.add_parser("auto", help="Run automated experiment harness")
    auto.add_argument("--strategy", default="all",
                      choices=["all", "atrss", "helix", "portfolio"],
                      help="Strategy filter (default: all)")
    auto.add_argument("--experiments", nargs="*", default=None,
                      help="Specific experiment IDs to run")
    auto.add_argument("--skip-robustness", action="store_true", default=False,
                      help="Skip robustness checks for faster ablation scan")
    auto.add_argument("--resume", action="store_true", default=False,
                      help="Resume from previous run (skip completed experiments)")
    auto.add_argument("--equity", type=float, default=100_000,
                      help="Initial equity (default: 100000)")
    auto.add_argument("--data-dir", default="backtests/swing/data/raw",
                      help="Data directory")
    auto.add_argument("--output-dir", default="backtests/swing/auto/output",
                      help="Output directory for results and report")

    # weakness-report subcommand
    wr = sub.add_parser("weakness-report", help="Generate unified weakness report across all strategies")
    wr.add_argument("--symbols", default=None)
    wr.add_argument("--start", default=None, help="Start date (YYYY-MM-DD) to limit backtest range")
    wr.add_argument("--end", default=None, help="End date (YYYY-MM-DD) to limit backtest range")
    wr.add_argument("--equity", type=float, default=100_000)
    wr.add_argument("--data-dir", default="backtest/data/raw")
    wr.add_argument("--report-file", default=None)

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
    elif args.command == "hold-time":
        cmd_hold_time(args)
    elif args.command == "exit-hyp":
        cmd_exit_hyp(args)
    elif args.command == "regime-bh":
        cmd_regime_bh(args)
    elif args.command == "chand-sweep":
        cmd_chand_sweep(args)
    elif args.command == "auto":
        cmd_auto(args)
    elif args.command == "weakness-report":
        _cmd_weakness_report(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
