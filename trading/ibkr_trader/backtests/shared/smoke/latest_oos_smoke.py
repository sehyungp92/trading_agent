"""Latest-period out-of-sample smoke runner.

This module wraps the registered strategy runners in
``backtests.shared.validation.oos_validation`` and makes the split window
dynamic: the OOS slice is always the most recent requested period ending at
the latest available data date, unless explicitly overridden.

Examples:
    python -m backtests.shared.smoke.latest_oos_smoke --strategy tpc --period 1mo
    python -m backtests.shared.smoke.latest_oos_smoke --strategy swing --period 3mo
    python -m backtests.shared.smoke.latest_oos_smoke --strategy all --period 6mo --data-end-policy common
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from backtests.shared.validation import oos_validation as oos

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "backtests" / "output" / "oos_smoke"


@dataclass(frozen=True)
class Window:
    """Resolved latest-period OOS window."""

    label: str
    start: date
    end: date

    @property
    def last_seen(self) -> date:
        return self.start - timedelta(days=1)

    @property
    def months(self) -> float:
        return max((self.end - self.start).days / 30.44, 0.1)


@dataclass(frozen=True)
class AggregateMetrics:
    """Approximate multi-strategy aggregate from per-strategy window metrics.

    The existing registered runners return metrics, not a synchronized
    timestamped equity curve, so max drawdown is a conservative approximation
    rather than a true portfolio replay drawdown.
    """

    total_trades: int
    winning_trades: int
    win_rate: float
    profit_factor: float
    net_r: float
    avg_r: float
    max_drawdown_r_proxy: float
    trades_per_month: float
    strategy_count: int


STRATEGY_PROBES: dict[str, tuple[str, ...]] = {
    "iaric": ("backtests/stock/data/raw/*_5m.parquet",),
    "alcb": ("backtests/stock/data/raw/*_5m.parquet",),
    "stock_portfolio": ("backtests/stock/data/raw/*_5m.parquet",),
    "helix_swing": ("backtests/swing/data/raw/*_1h.parquet", "backtests/swing/data/raw/*_1d.parquet"),
    "atrss": ("backtests/swing/data/raw/*_1h.parquet", "backtests/swing/data/raw/*_1d.parquet"),
    "tpc": (
        "backtests/swing/data/raw/QQQ_15m.parquet",
        "backtests/swing/data/raw/QQQ_1h.parquet",
        "backtests/swing/data/raw/QQQ_1d.parquet",
        "backtests/swing/data/raw/GLD_15m.parquet",
        "backtests/swing/data/raw/GLD_1h.parquet",
        "backtests/swing/data/raw/GLD_1d.parquet",
        "backtests/swing/data/raw/NQ_1h.parquet",
        "backtests/swing/data/raw/NQ_1d.parquet",
        "backtests/swing/data/raw/GC_1h.parquet",
        "backtests/swing/data/raw/GC_1d.parquet",
    ),
    "swing_portfolio": (
        "backtests/swing/data/raw/QQQ_15m.parquet",
        "backtests/swing/data/raw/QQQ_1h.parquet",
        "backtests/swing/data/raw/QQQ_1d.parquet",
        "backtests/swing/data/raw/GLD_15m.parquet",
        "backtests/swing/data/raw/GLD_1h.parquet",
        "backtests/swing/data/raw/GLD_1d.parquet",
        "backtests/swing/data/raw/NQ_1h.parquet",
        "backtests/swing/data/raw/NQ_1d.parquet",
        "backtests/swing/data/raw/GC_1h.parquet",
        "backtests/swing/data/raw/GC_1d.parquet",
    ),
    "breakout": ("backtests/swing/data/raw/*_1h.parquet", "backtests/swing/data/raw/*_1d.parquet"),
    "brs": ("backtests/swing/data/raw/*_1h.parquet", "backtests/swing/data/raw/*_1d.parquet"),
    "helix_momentum": ("backtests/momentum/data/raw/NQ_5m.parquet",),
    "vdubus": ("backtests/momentum/data/raw/NQ_5m.parquet", "backtests/momentum/data/raw/ES_1d.parquet"),
    "nqdtc": ("backtests/momentum/data/raw/NQ_5m.parquet", "backtests/momentum/data/raw/ES_1d.parquet"),
    "downturn": ("backtests/momentum/data/raw/NQ_5m.parquet", "backtests/momentum/data/raw/ES_1d.parquet"),
    "nq_regime": ("backtests/momentum/data/raw/NQ_5m.parquet",),
    "momentum_portfolio": ("backtests/momentum/data/raw/NQ_5m.parquet", "backtests/momentum/data/raw/ES_1d.parquet"),
}


def parse_period(raw: str, data_end: date) -> Window:
    """Parse a period like 1mo, 3m, 4w, 30d, or 1y."""
    value = raw.strip().lower()
    if not value:
        raise ValueError("Period cannot be empty.")

    unit_aliases = {
        "d": "days",
        "day": "days",
        "days": "days",
        "w": "weeks",
        "wk": "weeks",
        "week": "weeks",
        "weeks": "weeks",
        "m": "months",
        "mo": "months",
        "mon": "months",
        "month": "months",
        "months": "months",
        "y": "years",
        "yr": "years",
        "year": "years",
        "years": "years",
    }

    digits = "".join(ch for ch in value if ch.isdigit())
    unit = value[len(digits):]
    if not digits or unit not in unit_aliases:
        raise ValueError(f"Unsupported period {raw!r}. Use forms like 1mo, 3m, 4w, 30d, or 1y.")

    amount = int(digits)
    if amount <= 0:
        raise ValueError("Period amount must be positive.")

    canonical = unit_aliases[unit]
    if canonical == "days":
        start = data_end - timedelta(days=amount)
    elif canonical == "weeks":
        start = data_end - timedelta(weeks=amount)
    elif canonical == "months":
        start = _subtract_months(data_end, amount)
    elif canonical == "years":
        start = _subtract_months(data_end, amount * 12)
    else:  # pragma: no cover - guarded by unit_aliases.
        raise ValueError(canonical)

    return Window(label=raw, start=start, end=data_end)


def _subtract_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 - months
    year = month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, _last_day_of_month(year, month))
    return date(year, month, day)


def _last_day_of_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - timedelta(days=1)).day


def detect_data_end(strategies: Iterable[str], *, policy: str = "common") -> tuple[date, list[str]]:
    """Detect the latest date available for selected strategy probes.

    ``common`` uses the minimum latest date across probe files, which is safer
    for portfolio-level checks. ``max`` uses the newest available probe date,
    which is useful for single-strategy diagnostics when some unrelated raw
    files are stale.
    """
    import pandas as pd

    paths = _probe_paths(strategies)
    if not paths:
        fallback = date.fromisoformat(oos._detect_data_end())
        return fallback, ["No probe files found; fell back to legacy OOS data-end detection."]

    latest_by_path: list[tuple[Path, date]] = []
    warnings: list[str] = []
    for path in paths:
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            warnings.append(f"Could not inspect {path}: {exc}")
            continue
        if df.empty:
            warnings.append(f"Empty data file: {path}")
            continue
        try:
            latest = oos._to_naive_utc(df.index[-1]).date()
        except Exception as exc:
            warnings.append(f"Could not read final timestamp from {path}: {exc}")
            continue
        latest_by_path.append((path, latest))

    if not latest_by_path:
        fallback = date.fromisoformat(oos._detect_data_end())
        warnings.append("No readable probe files; fell back to legacy OOS data-end detection.")
        return fallback, warnings

    dates = [item[1] for item in latest_by_path]
    if policy == "common":
        data_end = min(dates)
    elif policy == "max":
        data_end = max(dates)
    else:
        raise ValueError("--data-end-policy must be common or max.")

    stale = [(path, latest) for path, latest in latest_by_path if latest < data_end]
    if stale:
        examples = ", ".join(f"{path.name}={latest.isoformat()}" for path, latest in stale[:5])
        warnings.append(f"{len(stale)} probe files end before selected data_end {data_end.isoformat()}: {examples}.")
    return data_end, warnings


def _probe_paths(strategies: Iterable[str]) -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for strategy in strategies:
        for pattern in STRATEGY_PROBES.get(strategy, ()):
            matched = sorted(PROJECT_ROOT.glob(pattern))
            for path in matched:
                if path in seen or not path.is_file():
                    continue
                seen.add(path)
                paths.append(path)
    return paths


def parse_weights(items: list[str], strategies: list[str]) -> dict[str, float]:
    """Parse optional strategy=weight items and default missing weights to 1."""
    weights = {strategy: 1.0 for strategy in strategies}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--weight must be strategy=weight, got {item!r}")
        strategy, raw_weight = item.split("=", 1)
        strategy = strategy.strip()
        if strategy not in weights:
            raise ValueError(f"--weight references unselected strategy {strategy!r}")
        weight = float(raw_weight)
        if weight < 0:
            raise ValueError("Weights must be non-negative.")
        weights[strategy] = weight
    return weights


def aggregate_window_metrics(results: list[oos.OOSResult], weights: dict[str, float], *, window: str) -> AggregateMetrics:
    total_trades = 0
    winning_trades = 0
    weighted_net_r = 0.0
    weighted_dd_proxy = 0.0
    weighted_trades_per_month = 0.0
    gross_win = 0.0
    gross_loss = 0.0

    usable = [result for result in results if not result.error]
    for result in usable:
        metrics = result.oos_metrics if window == "oos" else result.is_metrics
        weight = weights.get(result.strategy, 1.0)
        total_trades += metrics.total_trades
        winning_trades += metrics.winning_trades
        weighted_net_r += weight * metrics.net_r
        weighted_dd_proxy += weight * metrics.max_drawdown_r
        weighted_trades_per_month += weight * metrics.trades_per_month
        win, loss = _gross_from_metrics(metrics)
        gross_win += weight * win
        gross_loss += weight * loss

    pf = gross_win / gross_loss if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0)
    return AggregateMetrics(
        total_trades=total_trades,
        winning_trades=winning_trades,
        win_rate=winning_trades / total_trades if total_trades else 0.0,
        profit_factor=pf,
        net_r=weighted_net_r,
        avg_r=weighted_net_r / total_trades if total_trades else 0.0,
        max_drawdown_r_proxy=weighted_dd_proxy,
        trades_per_month=weighted_trades_per_month,
        strategy_count=len(usable),
    )


def _gross_from_metrics(metrics: oos.WindowMetrics) -> tuple[float, float]:
    pf = metrics.profit_factor
    net = metrics.net_r
    if metrics.total_trades == 0:
        return 0.0, 0.0
    if math.isinf(pf):
        return max(net, 0.0), 0.0
    if pf <= 0:
        return 0.0, max(-net, 0.0)
    if abs(pf - 1.0) < 1e-9:
        gross = abs(net) / 2.0
        return gross, gross
    gross_loss = net / (pf - 1.0)
    gross_win = pf * gross_loss
    return max(gross_win, 0.0), max(gross_loss, 0.0)


def format_markdown_report(
    *,
    results: list[oos.OOSResult],
    window: Window,
    weights: dict[str, float],
    data_end_policy: str,
    data_warnings: list[str],
    config_resolution: str,
    max_workers: int,
) -> str:
    is_agg = aggregate_window_metrics(results, weights, window="is")
    oos_agg = aggregate_window_metrics(results, weights, window="oos")

    lines = [
        f"# Latest OOS Smoke - {window.label}",
        "",
        f"- Generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"- OOS window: `{window.start.isoformat()}` to `{window.end.isoformat()}`",
        f"- IS window starts: `{oos.BACKTEST_START_DATE.isoformat()}`",
        f"- Data end policy: `{data_end_policy}`",
        f"- Config resolution: `{config_resolution}`",
        f"- Max workers requested: `{max_workers}`",
        "- Note: portfolio aggregate uses per-strategy metric aggregation, not a synchronized equity replay.",
        "",
    ]
    if data_warnings:
        lines.extend(["## Data Warnings", ""])
        lines.extend(f"- {warning}" for warning in data_warnings)
        lines.append("")

    lines.extend([
        "## Strategy Summary",
        "",
        "| Strategy | Family | OOS Trades | OOS PF | OOS AvgR | OOS NetR | OOS DD R | OOS/mo | IS PF | Assessment | Warnings |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ])
    for result in results:
        if result.error:
            lines.append(f"| {result.strategy} | error |  |  |  |  |  |  |  | ERROR: {result.error} |  |")
            continue
        oos_m = result.oos_metrics
        is_m = result.is_metrics
        lines.append(
            f"| {result.strategy} | {result.family} | {oos_m.total_trades} | {_fmt_float(oos_m.profit_factor)} | "
            f"{oos_m.avg_r:.3f} | {oos_m.net_r:.2f} | {oos_m.max_drawdown_r:.2f} | "
            f"{oos_m.trades_per_month:.2f} | {_fmt_float(is_m.profit_factor)} | "
            f"{result.assessment} | {len(result.warnings)} |"
        )

    lines.extend([
        "",
        "## Aggregate",
        "",
        "| Window | Strategies | Trades | Win Rate | PF | AvgR | NetR | DD R Proxy | Trades/mo |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        _aggregate_row("IS", is_agg),
        _aggregate_row("OOS", oos_agg),
        "",
        "## Details",
        "",
    ])
    for result in results:
        lines.append(f"### {result.strategy}")
        if result.error:
            lines.extend(["", f"Error: `{result.error}`", ""])
            continue
        lines.extend([
            "",
            f"- Config: `{result.config_path}`",
            f"- Action: `{result.action}`",
            f"- IS: trades `{result.is_metrics.total_trades}`, PF `{_fmt_float(result.is_metrics.profit_factor)}`, "
            f"avgR `{result.is_metrics.avg_r:.3f}`, netR `{result.is_metrics.net_r:.2f}`",
            f"- OOS: trades `{result.oos_metrics.total_trades}`, PF `{_fmt_float(result.oos_metrics.profit_factor)}`, "
            f"avgR `{result.oos_metrics.avg_r:.3f}`, netR `{result.oos_metrics.net_r:.2f}`",
        ])
        if result.warnings:
            lines.append("- Warnings:")
            lines.extend(f"  - {warning}" for warning in result.warnings)
        lines.append("")
    return "\n".join(lines)


def _aggregate_row(label: str, metrics: AggregateMetrics) -> str:
    return (
        f"| {label} | {metrics.strategy_count} | {metrics.total_trades} | {metrics.win_rate:.1%} | "
        f"{_fmt_float(metrics.profit_factor)} | {metrics.avg_r:.3f} | {metrics.net_r:.2f} | "
        f"{metrics.max_drawdown_r_proxy:.2f} | {metrics.trades_per_month:.2f} |"
    )


def _fmt_float(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if np.isnan(value):
        return "nan"
    return f"{value:.2f}"


def write_outputs(
    *,
    output_dir: Path,
    report: str,
    results: list[oos.OOSResult],
    window: Window,
    weights: dict[str, float],
    metadata: dict[str, Any],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "latest_oos_smoke_report.md"
    json_path = output_dir / "latest_oos_smoke_results.json"
    csv_path = output_dir / "latest_oos_smoke_summary.csv"

    report_path.write_text(report, encoding="utf-8")

    payload = {
        "metadata": metadata,
        "window": {
            "label": window.label,
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
            "last_seen": window.last_seen.isoformat(),
            "months": window.months,
        },
        "weights": weights,
        "aggregate": {
            "is": asdict(aggregate_window_metrics(results, weights, window="is")),
            "oos": asdict(aggregate_window_metrics(results, weights, window="oos")),
        },
        "results": [
            {
                "strategy": result.strategy,
                "family": result.family,
                "assessment": result.assessment,
                "action": result.action,
                "config_path": result.config_path,
                "warnings": result.warnings,
                "error": result.error,
                "is_metrics": asdict(result.is_metrics) if not result.error else None,
                "oos_metrics": asdict(result.oos_metrics) if not result.error else None,
            }
            for result in results
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategy",
                "family",
                "config_path",
                "is_trades",
                "is_pf",
                "is_avg_r",
                "is_net_r",
                "oos_trades",
                "oos_pf",
                "oos_avg_r",
                "oos_net_r",
                "oos_dd_r",
                "oos_trades_per_month",
                "assessment",
                "action",
                "warnings",
                "error",
            ],
        )
        writer.writeheader()
        for result in results:
            row = {
                "strategy": result.strategy,
                "family": result.family,
                "config_path": result.config_path,
                "assessment": result.assessment,
                "action": result.action,
                "warnings": " | ".join(result.warnings),
                "error": result.error,
            }
            if not result.error:
                row.update({
                    "is_trades": result.is_metrics.total_trades,
                    "is_pf": result.is_metrics.profit_factor,
                    "is_avg_r": result.is_metrics.avg_r,
                    "is_net_r": result.is_metrics.net_r,
                    "oos_trades": result.oos_metrics.total_trades,
                    "oos_pf": result.oos_metrics.profit_factor,
                    "oos_avg_r": result.oos_metrics.avg_r,
                    "oos_net_r": result.oos_metrics.net_r,
                    "oos_dd_r": result.oos_metrics.max_drawdown_r,
                    "oos_trades_per_month": result.oos_metrics.trades_per_month,
                })
            writer.writerow(row)
    return report_path, json_path, csv_path


def timestamped_output_dir(root: Path, window: Window, strategies: list[str]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    strategy_label = "portfolio" if len(strategies) > 1 else strategies[0]
    safe_period = "".join(ch if ch.isalnum() else "_" for ch in window.label.lower())
    return root / f"{strategy_label}_{safe_period}_{stamp}"


def _run_strategy_child(payload: dict[str, Any]) -> oos.OOSResult:
    """Run one registered strategy in a worker process."""
    strategy = payload["strategy"]
    overrides = {key: Path(value) for key, value in payload["config_overrides"].items()}
    oos.configure_config_resolution(
        config_root=payload["config_root"],
        mode=payload["config_resolution"],
        overrides=overrides,
    )
    oos.configure_validation_window(
        backtest_start=payload["backtest_start"],
        last_seen_data_date=payload["last_seen_data_date"],
        oos_cutoff_date=payload["oos_cutoff_date"],
    )
    return oos.RUNNERS[strategy](payload["data_end"])


def run_strategies(
    strategies: list[str],
    *,
    window: Window,
    backtest_start: str,
    config_root: str,
    config_resolution: str,
    config_overrides: dict[str, Path],
    max_workers: int,
) -> list[oos.OOSResult]:
    """Run selected strategies, parallelizing across strategies when useful."""
    if max_workers <= 1 or len(strategies) <= 1:
        results: list[oos.OOSResult] = []
        for strategy in strategies:
            logger.info("Running %s...", strategy)
            try:
                result = oos.RUNNERS[strategy](window.end.isoformat())
                results.append(result)
                _log_result(result)
            except Exception as exc:
                logger.error("%s failed: %s", strategy, exc)
                traceback.print_exc()
                results.append(oos.OOSResult(strategy=strategy, family=oos.STRATEGY_FAMILIES.get(strategy, "unknown"), error=str(exc)))
        return results

    payload_base = {
        "backtest_start": backtest_start,
        "last_seen_data_date": window.last_seen.isoformat(),
        "oos_cutoff_date": window.start.isoformat(),
        "data_end": window.end.isoformat(),
        "config_root": config_root,
        "config_resolution": config_resolution,
        "config_overrides": {key: str(value) for key, value in config_overrides.items()},
    }
    results_by_strategy: dict[str, oos.OOSResult] = {}
    with ProcessPoolExecutor(max_workers=max(1, min(max_workers, len(strategies)))) as pool:
        futures = {
            pool.submit(_run_strategy_child, {**payload_base, "strategy": strategy}): strategy
            for strategy in strategies
        }
        for future in as_completed(futures):
            strategy = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.error("%s failed: %s", strategy, exc)
                traceback.print_exc()
                result = oos.OOSResult(strategy=strategy, family=oos.STRATEGY_FAMILIES.get(strategy, "unknown"), error=str(exc))
            results_by_strategy[strategy] = result
            _log_result(result)
    return [results_by_strategy[strategy] for strategy in strategies]


def _log_result(result: oos.OOSResult) -> None:
    if result.error:
        logger.error("%s returned error: %s", result.strategy, result.error)
        return
    logger.info(
        "%s: OOS trades=%s PF=%.2f avgR=%.3f assessment=%s",
        result.strategy,
        result.oos_metrics.total_trades,
        result.oos_metrics.profit_factor,
        result.oos_metrics.avg_r,
        result.assessment,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Reusable latest-period OOS smoke runner")
    parser.add_argument(
        "--strategy",
        nargs="+",
        default=["all"],
        help="Strategy names or groups: all, stock, swing, momentum, portfolios, or individual registered names.",
    )
    parser.add_argument("--period", default="1mo", help="Latest OOS period: e.g. 1mo, 3m, 4w, 30d, 1y.")
    parser.add_argument("--data-end", default=None, help="Override latest data end date as YYYY-MM-DD.")
    parser.add_argument(
        "--data-end-policy",
        choices=["common", "max"],
        default="common",
        help="common=min latest date across probes; max=newest available probe date.",
    )
    parser.add_argument("--backtest-start", default=oos.BACKTEST_START_DATE.isoformat())
    parser.add_argument("--config-root", default=str(oos.CONFIG_ROOT))
    parser.add_argument(
        "--config-resolution",
        choices=["auto", "manifest", "static"],
        default="auto",
        help="How optimized configs are resolved.",
    )
    parser.add_argument("--config", action="append", default=[], help="Override config path as strategy=path.")
    parser.add_argument("--weight", action="append", default=[], help="Portfolio weight as strategy=weight.")
    parser.add_argument("--max-workers", type=int, default=4, help="Recorded for smoke reproducibility.")
    parser.add_argument("--list-configs", action="store_true", help="List resolved configs and exit.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    strategies = oos.expand_strategy_selection(args.strategy)
    unknown = [strategy for strategy in strategies if strategy not in oos.RUNNERS]
    if unknown:
        raise SystemExit(f"Unknown strategy names: {', '.join(unknown)}")

    config_overrides = oos.parse_config_overrides(args.config)
    oos.configure_config_resolution(
        config_root=args.config_root,
        mode=args.config_resolution,
        overrides=config_overrides,
    )
    if args.list_configs:
        print(oos.format_config_plan(strategies))
        return 0

    data_warnings: list[str] = []
    if args.data_end:
        data_end = date.fromisoformat(args.data_end)
    else:
        data_end, data_warnings = detect_data_end(strategies, policy=args.data_end_policy)
    window = parse_period(args.period, data_end)

    oos.configure_validation_window(
        backtest_start=args.backtest_start,
        last_seen_data_date=window.last_seen.isoformat(),
        oos_cutoff_date=window.start.isoformat(),
    )

    weights = parse_weights(args.weight, strategies)
    logger.info("Latest OOS window: %s to %s", window.start.isoformat(), window.end.isoformat())
    logger.info("Strategies: %s", ", ".join(strategies))
    results = run_strategies(
        strategies,
        window=window,
        backtest_start=args.backtest_start,
        config_root=args.config_root,
        config_resolution=args.config_resolution,
        config_overrides=config_overrides,
        max_workers=args.max_workers,
    )

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategies": strategies,
        "data_end_policy": args.data_end_policy,
        "data_warnings": data_warnings,
        "config_resolution": args.config_resolution,
        "config_root": str(args.config_root),
        "max_workers": args.max_workers,
        "note": "OOS window is the latest requested period ending at selected data_end.",
    }
    report = format_markdown_report(
        results=results,
        window=window,
        weights=weights,
        data_end_policy=args.data_end_policy,
        data_warnings=data_warnings,
        config_resolution=args.config_resolution,
        max_workers=args.max_workers,
    )

    output_dir = args.output_dir or timestamped_output_dir(DEFAULT_OUTPUT_ROOT, window, strategies)
    report_path, json_path, csv_path = write_outputs(
        output_dir=output_dir,
        report=report,
        results=results,
        window=window,
        weights=weights,
        metadata=metadata,
    )
    print(report)
    logger.info("Report saved to: %s", report_path)
    logger.info("JSON saved to: %s", json_path)
    logger.info("CSV saved to: %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
