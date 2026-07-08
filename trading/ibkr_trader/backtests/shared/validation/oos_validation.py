"""Out-of-Sample Validation Runner.

Runs frozen-parameter backtests on extended data, splits trades at the OOS
boundary (first unseen data after 2026-03-20), and generates a comparison report per the OOS
Validation & Re-Optimization Guide.

Usage:
    python -m backtests.shared.validation.oos_validation --strategy all
    python -m backtests.shared.validation.oos_validation --strategy momentum
    python -m backtests.shared.validation.oos_validation --strategy iaric
    python -m backtests.shared.validation.oos_validation --strategy alcb helix_swing vdubus
    python -m backtests.shared.validation.oos_validation --strategy nqdtc --config nqdtc=path/to/optimized_config.json
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import traceback
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("strategies.momentum.nqdtc.box").setLevel(logging.WARNING)
logging.getLogger("backtests.momentum.engine.vdubus_engine").setLevel(logging.WARNING)
logging.getLogger("strategies.momentum.vdub").setLevel(logging.WARNING)
logging.getLogger("strategies.momentum.helix_v40.gates").setLevel(logging.WARNING)

BACKTEST_START = datetime(2024, 1, 1)
BACKTEST_START_DATE = date(2024, 1, 1)
LAST_SEEN_DATA_DATE = date(2026, 3, 20)
OOS_CUTOFF_DATE = LAST_SEEN_DATA_DATE + timedelta(days=1)
OOS_CUTOFF = datetime.combine(OOS_CUTOFF_DATE, datetime.min.time())
PROJECT_ROOT = Path(__file__).resolve().parents[3]

STATIC_OPTIMIZED_CONFIG_PATHS = {
    "iaric": "backtests/output/stock/iaric/round_2/optimized_config.json",
    "alcb": "backtests/output/stock/alcb/round_3/optimized_config.json",
    "stock_portfolio": "backtests/output/stock/portfolio_synergy/round_3/optimized_config.json",
    "helix_swing": "backtests/output/swing/helix/round_2/optimized_config.json",
    "atrss": "backtests/output/swing/atrss/round_3/optimized_config.json",
    "tpc": "backtests/output/swing/tpc/round_8/optimized_config.json",
    "swing_portfolio": "backtests/output/swing/portfolio_synergy/round_3/optimized_config.json",
    "breakout": "backtests/output/swing/breakout/round_5/optimized_config.json",
    "brs": "backtests/output/swing/brs/round_1/optimized_config.json",
    "helix_momentum": "backtests/output/momentum/helix/round_5/optimized_config.json",
    "vdubus": "backtests/output/momentum/vdubus/round_3/optimized_config.json",
    "nqdtc": "backtests/output/momentum/nqdtc/round_5/optimized_config.json",
    "downturn": "backtests/output/momentum/downturn/round_3/optimized_config.json",
    "nq_regime": "backtests/output/momentum/nq_regime/round_5/optimized_config.json",
    "momentum_portfolio": "backtests/output/momentum/portfolio_synergy/round_2/optimized_config.json",
}

STRATEGY_FAMILIES = {
    "iaric": "stock",
    "alcb": "stock",
    "stock_portfolio": "stock",
    "helix_swing": "swing",
    "atrss": "swing",
    "tpc": "swing",
    "swing_portfolio": "swing",
    "breakout": "swing",
    "brs": "swing",
    "helix_momentum": "momentum",
    "vdubus": "momentum",
    "nqdtc": "momentum",
    "downturn": "momentum",
    "nq_regime": "momentum",
    "momentum_portfolio": "momentum",
}

CONFIG_ROOT = PROJECT_ROOT / "backtests" / "output"
CONFIG_PATH_OVERRIDES: dict[str, Path] = {}
CONFIG_RESOLUTION_MODE = "auto"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class WindowMetrics:
    """Metrics for a single time window (IS or OOS)."""
    total_trades: int = 0
    winning_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    net_r: float = 0.0
    avg_r: float = 0.0
    max_drawdown_r: float = 0.0
    trades_per_month: float = 0.0
    months: float = 0.0


def compute_window_metrics(
    r_multiples: list[float],
    months: float,
) -> WindowMetrics:
    """Compute metrics from a list of R-multiples over a given period."""
    n = len(r_multiples)
    if n == 0:
        return WindowMetrics(months=months)

    wins = [r for r in r_multiples if r > 0]
    losses = [r for r in r_multiples if r < 0]

    gross_win = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0

    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    net_r = sum(r_multiples)
    avg_r = net_r / n

    # Drawdown in R
    cum = np.cumsum(r_multiples)
    running_max = np.maximum.accumulate(cum)
    drawdowns = running_max - cum
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    return WindowMetrics(
        total_trades=n,
        winning_trades=len(wins),
        win_rate=len(wins) / n if n > 0 else 0.0,
        profit_factor=pf,
        net_r=net_r,
        avg_r=avg_r,
        max_drawdown_r=max_dd,
        trades_per_month=n / months if months > 0 else 0.0,
        months=months,
    )


def _window_months(start: date, end: date) -> float:
    """Calendar-month approximation used for frequency normalization."""
    return max((end - start).days / 30.44, 0.1)


def _to_naive_utc(value: Any) -> datetime:
    """Normalize trade timestamps for deterministic cutoff comparisons."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, np.datetime64):
        value = value.astype("datetime64[us]").astype(datetime)
    if not isinstance(value, datetime):
        raise ValueError(f"Cannot interpret timestamp: {value!r}")
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def configure_validation_window(
    *,
    backtest_start: str | None = None,
    last_seen_data_date: str | None = None,
    oos_cutoff_date: str | None = None,
) -> None:
    """Configure the replay split dates used by all strategy runners."""
    global BACKTEST_START, BACKTEST_START_DATE, LAST_SEEN_DATA_DATE, OOS_CUTOFF_DATE, OOS_CUTOFF

    if backtest_start:
        BACKTEST_START_DATE = date.fromisoformat(backtest_start)
        BACKTEST_START = datetime.combine(BACKTEST_START_DATE, datetime.min.time())
    if last_seen_data_date:
        LAST_SEEN_DATA_DATE = date.fromisoformat(last_seen_data_date)
    if oos_cutoff_date:
        OOS_CUTOFF_DATE = date.fromisoformat(oos_cutoff_date)
    else:
        OOS_CUTOFF_DATE = LAST_SEEN_DATA_DATE + timedelta(days=1)
    OOS_CUTOFF = datetime.combine(OOS_CUTOFF_DATE, datetime.min.time())


def configure_config_resolution(
    *,
    config_root: str | Path | None = None,
    mode: str = "auto",
    overrides: dict[str, Path] | None = None,
) -> None:
    """Configure where optimized configs are resolved from.

    mode="auto" prefers the latest manifest/round directory and falls back to
    the static table. mode="manifest" requires latest-round discovery.
    mode="static" preserves the historic hard-coded table.
    """
    global CONFIG_ROOT, CONFIG_RESOLUTION_MODE, CONFIG_PATH_OVERRIDES

    if config_root is not None:
        root = Path(config_root)
        CONFIG_ROOT = root if root.is_absolute() else PROJECT_ROOT / root
    CONFIG_RESOLUTION_MODE = mode
    CONFIG_PATH_OVERRIDES = dict(overrides or {})


def resolve_optimized_config_path(strategy: str) -> Path:
    """Resolve the optimized config path for a strategy."""
    if strategy in CONFIG_PATH_OVERRIDES:
        return _existing_config_path(strategy, CONFIG_PATH_OVERRIDES[strategy], source="override")

    if CONFIG_RESOLUTION_MODE != "static":
        latest = _latest_round_config_path(strategy)
        if latest is not None:
            return latest
        if CONFIG_RESOLUTION_MODE == "manifest":
            raise FileNotFoundError(
                f"No latest round optimized config could be discovered for {strategy} under {CONFIG_ROOT}."
            )

    rel_path = STATIC_OPTIMIZED_CONFIG_PATHS.get(strategy)
    if not rel_path:
        raise KeyError(f"No static optimized config path registered for {strategy}.")
    return _existing_config_path(strategy, Path(rel_path), source="static")


def _existing_config_path(strategy: str, path: Path, *, source: str) -> Path:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(f"Missing {source} optimized config for {strategy}: {resolved}")
    return resolved


def _latest_round_config_path(strategy: str) -> Path | None:
    strategy_dir = _strategy_output_dir(strategy)
    if strategy_dir is None or not strategy_dir.exists():
        return None

    discovered = [
        item for item in (
            _latest_manifest_round(strategy_dir),
            _latest_numeric_round_dir(strategy_dir),
        )
        if item is not None
    ]
    latest_round = max(discovered) if discovered else None
    if latest_round is None:
        return None

    path = strategy_dir / f"round_{latest_round}" / "optimized_config.json"
    return path if path.exists() else None


def _strategy_output_dir(strategy: str) -> Path | None:
    rel_path = STATIC_OPTIMIZED_CONFIG_PATHS.get(strategy)
    if not rel_path:
        return None
    parts = Path(rel_path).parts
    try:
        idx = parts.index("output")
    except ValueError:
        return None
    if len(parts) <= idx + 2:
        return None
    family = parts[idx + 1]
    output_name = parts[idx + 2]
    return CONFIG_ROOT / family / output_name


def _latest_manifest_round(strategy_dir: Path) -> int | None:
    manifest_path = strategy_dir / "rounds_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        logger.warning("Could not read %s: %s", manifest_path, exc)
        return None
    rounds = []
    for entry in manifest.get("rounds", []):
        try:
            rounds.append(int(entry.get("round")))
        except (TypeError, ValueError):
            continue
    return max(rounds) if rounds else None


def _latest_numeric_round_dir(strategy_dir: Path) -> int | None:
    rounds = []
    for path in strategy_dir.glob("round_*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("round_")
        if not suffix.isdigit():
            continue
        if (path / "optimized_config.json").exists():
            rounds.append(int(suffix))
    return max(rounds) if rounds else None


def _load_mutations_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if isinstance(payload.get("mutations"), dict):
            return dict(payload["mutations"])
        if isinstance(payload.get("cumulative_mutations"), dict):
            return dict(payload["cumulative_mutations"])
        return dict(payload)
    raise TypeError(f"Unexpected optimized config payload in {path}")


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------

@dataclass
class OOSResult:
    """Result of a single strategy's OOS validation."""
    strategy: str
    family: str
    is_metrics: WindowMetrics = field(default_factory=WindowMetrics)
    oos_metrics: WindowMetrics = field(default_factory=WindowMetrics)
    assessment: str = "GREY"
    action: str = "Monitor"
    config_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


# IS baselines from the guide/run summaries. These are audit references only;
# assessments compare against the reproduced IS metrics from the same replay.
IS_BASELINES = {
    "iaric": {"pf": 1.78, "wr": 0.726, "trades_mo": 38.5},
    "alcb": {"pf": 2.20, "wr": 0.62, "trades_mo": 19.7},
    "helix_swing": {"pf": 4.01, "wr": 0.518, "trades_mo": 15.7},
    "atrss": {"pf": 5.93, "wr": 0.833, "trades_mo": 10.6},
    "tpc": {"pf": 2.34, "wr": 0.59, "trades_mo": 1.9},
    "breakout": {"pf": 6.50, "wr": 0.915, "trades_mo": 5.4},
    "brs": {"pf": 5.32, "wr": 0.70, "trades_mo": 3.9},
    "helix_momentum": {"pf": 3.27, "wr": 0.454, "trades_mo": 5.0},
    "vdubus": {"pf": 2.80, "wr": 0.524, "trades_mo": 6.5},
    "nqdtc": {"pf": 2.14, "wr": 0.58, "trades_mo": 3.8},
    "downturn": {"pf": 2.0, "wr": 0.518, "trades_mo": 4.4},
    "nq_regime": {"pf": 20.1, "wr": 0.782, "trades_mo": 3.0},
}


def _get_entry_time(trade) -> datetime:
    """Extract entry_time from a trade record (handles different types)."""
    if hasattr(trade, "entry_time"):
        return _to_naive_utc(trade.entry_time)
    if hasattr(trade, "entry_dt"):
        return _to_naive_utc(trade.entry_dt)
    if isinstance(trade, dict):
        return _to_naive_utc(trade.get("entry_time") or trade.get("entry_dt"))
    raise ValueError(f"Cannot extract entry_time from {type(trade)}")


def _get_r_multiple(trade) -> float:
    """Extract R-multiple from a trade record."""
    if hasattr(trade, "r_multiple"):
        return trade.r_multiple
    if hasattr(trade, "r_mult"):
        return trade.r_mult
    if hasattr(trade, "pnl_r"):
        return trade.pnl_r
    if isinstance(trade, dict):
        return trade.get("r_multiple", trade.get("r_mult", trade.get("pnl_r", 0.0)))
    return 0.0


def _split_and_analyze(trades, is_months: float, oos_months: float) -> tuple[WindowMetrics, WindowMetrics]:
    """Split trades at OOS boundary and compute metrics for each window."""
    is_rs = []
    oos_rs = []

    for t in trades:
        entry = _get_entry_time(t)
        if entry < BACKTEST_START:
            continue
        r = _get_r_multiple(t)
        if entry < OOS_CUTOFF:
            is_rs.append(r)
        else:
            oos_rs.append(r)

    is_metrics = compute_window_metrics(is_rs, is_months)
    oos_metrics = compute_window_metrics(oos_rs, oos_months)
    return is_metrics, oos_metrics


def _assess(_strategy: str, is_m: WindowMetrics, oos_m: WindowMetrics) -> tuple[str, str]:
    """Determine assessment color and action per the decision matrix."""
    if oos_m.total_trades < 8:
        return "GREY", "Inconclusive -- extend monitoring to 4+ months"

    if oos_m.total_trades < 15:
        if oos_m.avg_r > 0:
            return "GREY", "Positive sign only -- sample too small for validation"
        return "RED", "Negative expectancy with insufficient trades -- extend monitoring"

    is_pf = is_m.profit_factor
    has_comparable_pf = is_pf > 0 and np.isfinite(is_pf)

    if oos_m.total_trades < 30:
        if oos_m.avg_r <= 0:
            return "RED", "Negative edge but low sample -- extend monitoring"
        if has_comparable_pf and oos_m.profit_factor < is_pf * 0.5:
            return "YELLOW", "Positive but materially degraded -- monitor one more month"
        return "YELLOW", "Positive expectancy with marginal sample -- monitor"

    # Enough trades for comparison
    if oos_m.avg_r > 0 and not has_comparable_pf:
        return "YELLOW", "Positive expectancy; IS PF not comparable -- monitor"

    if oos_m.profit_factor >= is_pf * 0.5 and oos_m.avg_r > 0:
        # Check if within reasonable range
        if oos_m.profit_factor >= is_pf * 0.8:
            return "GREEN", "No action needed -- edge persists"
        return "YELLOW", "Monitor for 1 more month"

    if oos_m.avg_r <= 0:
        if oos_m.total_trades >= 30:
            return "ORANGE", "Investigate root cause -- negative OOS edge with sufficient data"
        return "RED", "Negative edge but low sample -- extend monitoring"

    if has_comparable_pf and oos_m.profit_factor < is_pf * 0.5:
        return "ORANGE", "PF degraded >50% -- investigate"

    return "YELLOW", "Moderate degradation -- monitor"


def _load_optimized_config(
    strategy: str,
    base_config: Any,
    mutator: Callable[[Any, dict[str, Any]], Any],
    *,
    data_end: str,
) -> tuple[Any, Path]:
    """Load the exact optimized mutation set and apply it to a base config."""
    path = resolve_optimized_config_path(strategy)
    mutations = _load_mutations_file(path)
    config = mutator(base_config, mutations)
    config = _coerce_json_config_values(config)
    config = _with_backtest_window(config, data_end)
    return config, path


def _coerce_json_config_values(config: Any) -> Any:
    """Restore non-JSON scalar types that optimized_config.json stores as strings."""
    param_overrides = getattr(config, "param_overrides", None)
    if not isinstance(param_overrides, dict):
        return config

    changed = False
    coerced: dict[str, Any] = {}
    for key, value in param_overrides.items():
        if isinstance(value, str) and _looks_like_time(value):
            coerced[key] = time.fromisoformat(value)
            changed = True
        else:
            coerced[key] = value
    if not changed:
        return config
    try:
        return replace(config, param_overrides=coerced)
    except TypeError:
        config.param_overrides = coerced
        return config


def _looks_like_time(value: str) -> bool:
    parts = value.split(":")
    if len(parts) not in {2, 3}:
        return False
    return all(part.isdigit() for part in parts)


def _with_backtest_window(config: Any, data_end: str) -> Any:
    """Set explicit validation dates where the config supports them."""
    changes: dict[str, Any] = {}
    if hasattr(config, "start_date"):
        current = getattr(config, "start_date")
        if isinstance(current, str):
            changes["start_date"] = BACKTEST_START_DATE.isoformat()
        else:
            start = BACKTEST_START
            if isinstance(current, datetime) and current.tzinfo is not None:
                start = start.replace(tzinfo=current.tzinfo)
            changes["start_date"] = start
    if hasattr(config, "end_date"):
        current = getattr(config, "end_date")
        end_date = date.fromisoformat(data_end)
        if isinstance(current, str):
            changes["end_date"] = end_date.isoformat()
        else:
            end = datetime.combine(end_date, datetime.min.time())
            if isinstance(current, datetime) and current.tzinfo is not None:
                end = end.replace(tzinfo=current.tzinfo)
            changes["end_date"] = end
    if not changes:
        return config
    try:
        return replace(config, **changes)
    except TypeError:
        for key, value in changes.items():
            setattr(config, key, value)
        return config


def _baseline_warnings(strategy: str, config_path: Path, is_metrics: WindowMetrics) -> list[str]:
    """Warn when frozen IS reproduction drifts from the stored round summary."""
    summary_path = config_path.with_name("run_summary.json")
    if not summary_path.exists() or is_metrics.total_trades == 0:
        return []
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Could not read run summary: {exc}"]

    source_label = "stored round summary"
    headline = summary.get("headline_metrics", {}) or {}
    final_metrics = summary.get("final_metrics", {}) or {}
    selection_repair = final_metrics.get("selection_oos_repair") or summary.get("selection_oos_repair")
    if isinstance(selection_repair, dict) and isinstance(selection_repair.get("is_metrics"), dict):
        headline = selection_repair["is_metrics"]
        source_label = "stored selection split"
    expected_trades = headline.get("total_trades")
    expected_pf = headline.get("profit_factor")
    warnings: list[str] = []

    if expected_trades is not None:
        diff = abs(int(expected_trades) - is_metrics.total_trades)
        if diff > max(3, int(expected_trades) * 0.03):
            if source_label == "stored round summary" and int(expected_trades) > is_metrics.total_trades:
                warnings.append(
                    f"Stored round summary appears to include post-cutoff trades "
                    f"({int(expected_trades)} vs reproduced IS {is_metrics.total_trades}); "
                    "assessment uses reproduced IS metrics."
                )
            else:
                warnings.append(
                    f"IS trade count does not reproduce {source_label} "
                    f"({is_metrics.total_trades} vs {int(expected_trades)})."
                )
    if expected_pf not in (None, 0):
        rel_diff = abs(is_metrics.profit_factor - float(expected_pf)) / max(abs(float(expected_pf)), 1e-9)
        if rel_diff > 0.10:
            warnings.append(
                f"IS profit factor does not reproduce {source_label} "
                f"({is_metrics.profit_factor:.2f} vs {float(expected_pf):.2f})."
            )
    return warnings


def _audit_paths(paths: list[Path], data_end: str) -> list[str]:
    """Check that required parquet files exist and reach the requested end date."""
    import pandas as pd

    target = date.fromisoformat(data_end)
    warnings: list[str] = []
    for path in paths:
        if not path.exists():
            warnings.append(f"Missing data file: {path}")
            continue
        try:
            df = pd.read_parquet(path)
            if df.empty:
                warnings.append(f"Empty data file: {path}")
                continue
            latest = _to_naive_utc(df.index[-1]).date()
        except Exception as exc:
            warnings.append(f"Could not inspect {path}: {exc}")
            continue
        if latest < target:
            warnings.append(f"{path.name} ends {latest}, before requested {target}.")
    return warnings


def _audit_stock_intraday(data_dir: Path, data_end: str) -> list[str]:
    """Summarize stock intraday coverage without dumping hundreds of paths."""
    import pandas as pd

    target = date.fromisoformat(data_end)
    files = sorted(data_dir.glob("*_5m.parquet"))
    if not files:
        return [f"No stock 5m parquet files found in {data_dir}."]

    stale: list[tuple[str, date]] = []
    unreadable: list[str] = []
    for path in files:
        try:
            df = pd.read_parquet(path)
            if df.empty:
                stale.append((path.name, date.min))
                continue
            latest = _to_naive_utc(df.index[-1]).date()
        except Exception as exc:
            unreadable.append(f"{path.name}: {exc}")
            continue
        if latest < target:
            stale.append((path.name, latest))

    warnings: list[str] = []
    if stale:
        examples = ", ".join(f"{name}={latest}" for name, latest in stale[:5])
        warnings.append(f"{len(stale)}/{len(files)} stock 5m files end before {target}; examples: {examples}.")
    if unreadable:
        warnings.append(f"{len(unreadable)} stock 5m files could not be inspected; first: {unreadable[0]}.")
    return warnings


def _collect_symbol_trades(result: Any) -> list[Any]:
    """Flatten strategy result objects that store trades per symbol."""
    if hasattr(result, "symbol_results"):
        trades: list[Any] = []
        for symbol, symbol_result in result.symbol_results.items():
            for trade in getattr(symbol_result, "trades", []):
                if not getattr(trade, "symbol", ""):
                    try:
                        trade.symbol = symbol
                    except Exception:
                        pass
                trades.append(trade)
        return trades
    return list(getattr(result, "trades", []))


# ---------------------------------------------------------------------------
# Individual strategy runners
# ---------------------------------------------------------------------------

def run_iaric(data_end: str) -> OOSResult:
    """Run IARIC frozen-parameter OOS validation."""
    from backtests.stock.config_iaric import IARICBacktestConfig
    from backtests.stock.auto.config_mutator import mutate_iaric_config
    from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine
    from backtests.stock.data.replay_cache import load_research_replay_bundle

    data_dir = PROJECT_ROOT / "backtests" / "stock" / "data" / "raw"
    base_config = IARICBacktestConfig(
        start_date="2024-01-01",
        end_date=data_end,
        initial_equity=10_000,
        tier=3,
        data_dir=data_dir,
    )
    config, config_path = _load_optimized_config("iaric", base_config, mutate_iaric_config, data_end=data_end)
    replay = load_research_replay_bundle(data_dir).data
    engine = IARICPullbackEngine(config, replay, collect_diagnostics=False)
    result = engine.run()

    is_months = _window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE)
    oos_months = _compute_oos_months(data_end)
    is_m, oos_m = _split_and_analyze(result.trades, is_months, oos_months)
    assessment, action = _assess("iaric", is_m, oos_m)
    warnings = _audit_stock_intraday(data_dir, data_end) + _baseline_warnings("iaric", config_path, is_m)

    return OOSResult(
        strategy="iaric", family="stock",
        is_metrics=is_m, oos_metrics=oos_m,
        assessment=assessment, action=action,
        config_path=str(config_path),
        warnings=warnings,
    )


def run_alcb(data_end: str) -> OOSResult:
    """Run ALCB frozen-parameter OOS validation."""
    from backtests.stock.config_alcb import ALCBBacktestConfig
    from backtests.stock.auto.config_mutator import mutate_alcb_config
    from backtests.stock.engine.alcb_engine import ALCBIntradayEngine
    from backtests.stock.data.replay_cache import load_research_replay_bundle

    data_dir = PROJECT_ROOT / "backtests" / "stock" / "data" / "raw"
    base_config = ALCBBacktestConfig(
        start_date="2024-01-01",
        end_date=data_end,
        initial_equity=10_000,
        tier=2,
        data_dir=data_dir,
    )
    config, config_path = _load_optimized_config("alcb", base_config, mutate_alcb_config, data_end=data_end)
    replay = load_research_replay_bundle(data_dir).data
    engine = ALCBIntradayEngine(config, replay)
    result = engine.run()

    is_months = _window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE)
    oos_months = _compute_oos_months(data_end)
    is_m, oos_m = _split_and_analyze(result.trades, is_months, oos_months)
    assessment, action = _assess("alcb", is_m, oos_m)
    warnings = _audit_stock_intraday(data_dir, data_end) + _baseline_warnings("alcb", config_path, is_m)

    return OOSResult(
        strategy="alcb", family="stock",
        is_metrics=is_m, oos_metrics=oos_m,
        assessment=assessment, action=action,
        config_path=str(config_path),
        warnings=warnings,
    )


def run_helix_swing(data_end: str) -> OOSResult:
    """Run Swing Helix frozen-parameter OOS validation."""
    from backtests.swing.config_helix import HelixBacktestConfig
    from backtests.swing.auto.config_mutator import mutate_helix_config
    from backtests.swing.engine.helix_portfolio_engine import run_helix_independent
    from backtests.swing.auto.helix.worker import load_helix_worker_data

    data_dir = PROJECT_ROOT / "backtests" / "swing" / "data" / "raw"
    base_config = HelixBacktestConfig(
        initial_equity=10_000,
        data_dir=data_dir,
    )
    config, config_path = _load_optimized_config("helix_swing", base_config, mutate_helix_config, data_end=data_end)
    data = load_helix_worker_data(config.symbols, data_dir)
    result = run_helix_independent(data, config)

    all_trades = _collect_symbol_trades(result)

    is_months = _window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE)
    oos_months = _compute_oos_months(data_end)
    is_m, oos_m = _split_and_analyze(all_trades, is_months, oos_months)
    assessment, action = _assess("helix_swing", is_m, oos_m)
    warnings = _audit_paths(
        [data_dir / f"{symbol}_{tf}.parquet" for symbol in config.symbols for tf in ("1h", "1d")],
        data_end,
    ) + _baseline_warnings("helix_swing", config_path, is_m)

    return OOSResult(
        strategy="helix_swing", family="swing",
        is_metrics=is_m, oos_metrics=oos_m,
        assessment=assessment, action=action,
        config_path=str(config_path),
        warnings=warnings,
    )


def run_atrss(data_end: str) -> OOSResult:
    """Run ATRSS frozen-parameter OOS validation."""
    from backtests.swing.config import AblationFlags, BacktestConfig, SlippageConfig
    from backtests.swing.auto.config_mutator import mutate_atrss_config
    from backtests.swing.engine.portfolio_engine import run_synchronized
    from backtests.swing.data.replay_cache import load_atrss_replay_bundle

    data_dir = PROJECT_ROOT / "backtests" / "swing" / "data" / "raw"
    base_config = BacktestConfig(
        initial_equity=10_000,
        fixed_qty=10,
        data_dir=data_dir,
        slippage=SlippageConfig(commission_per_contract=1.00),
        flags=AblationFlags(stall_exit=False),
    )
    config, config_path = _load_optimized_config("atrss", base_config, mutate_atrss_config, data_end=data_end)
    data = load_atrss_replay_bundle(data_dir, symbols=tuple(config.symbols)).data
    result = run_synchronized(data, config)

    all_trades = _collect_symbol_trades(result)
    is_months = _window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE)
    oos_months = _compute_oos_months(data_end)
    is_m, oos_m = _split_and_analyze(all_trades, is_months, oos_months)
    assessment, action = _assess("atrss", is_m, oos_m)
    warnings = _audit_paths(
        [data_dir / f"{symbol}_{tf}.parquet" for symbol in config.symbols for tf in ("1h", "1d")],
        data_end,
    ) + _baseline_warnings("atrss", config_path, is_m)

    return OOSResult(
        strategy="atrss", family="swing",
        is_metrics=is_m, oos_metrics=oos_m,
        assessment=assessment, action=action,
        config_path=str(config_path),
        warnings=warnings,
    )


def run_tpc(data_end: str) -> OOSResult:
    """Run TPC frozen-parameter OOS validation."""
    from backtests.swing.config_tpc import TPCBacktestConfig
    from backtests.swing.data.replay_cache import load_tpc_replay_bundle
    from backtests.swing.engine.tpc_engine import run_tpc_independent

    data_dir = PROJECT_ROOT / "backtests" / "swing" / "data" / "raw"
    base_config = TPCBacktestConfig(
        initial_equity=10_000,
        data_dir=data_dir,
    )
    config, config_path = _load_optimized_config(
        "tpc",
        base_config,
        lambda cfg, mutations: cfg.with_overrides(mutations),
        data_end=data_end,
    )
    bundle = load_tpc_replay_bundle(data_dir, symbols=tuple(config.symbols))
    result = run_tpc_independent(bundle.data, config, indicator_cache={})

    is_months = _window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE)
    oos_months = _compute_oos_months(data_end)
    is_m, oos_m = _split_and_analyze(result.trades, is_months, oos_months)
    assessment, action = _assess("tpc", is_m, oos_m)
    context_symbols = {"QQQ": "NQ", "GLD": "GC"}
    audit_paths = [
        data_dir / f"{symbol}_{tf}.parquet"
        for symbol in config.symbols
        for tf in ("15m", "1h", "1d")
    ]
    audit_paths.extend(
        data_dir / f"{context_symbols[symbol]}_{tf}.parquet"
        for symbol in config.symbols
        if symbol in context_symbols
        for tf in ("1h", "1d")
    )
    warnings = _audit_paths(audit_paths, data_end) + _baseline_warnings("tpc", config_path, is_m)

    return OOSResult(
        strategy="tpc", family="swing",
        is_metrics=is_m, oos_metrics=oos_m,
        assessment=assessment, action=action,
        config_path=str(config_path),
        warnings=warnings,
    )


def run_breakout(data_end: str) -> OOSResult:
    """Run Breakout frozen-parameter OOS validation."""
    from backtests.swing.config_breakout import BreakoutBacktestConfig
    from backtests.swing.auto.config_mutator import mutate_breakout_config
    from backtests.swing.engine.breakout_portfolio_engine import run_breakout_synchronized
    from backtests.swing.data.replay_cache import load_breakout_replay_bundle

    data_dir = PROJECT_ROOT / "backtests" / "swing" / "data" / "raw"
    base_config = BreakoutBacktestConfig(
        initial_equity=10_000,
        data_dir=data_dir,
        track_signals=False,
        track_shadows=False,
    )
    config, config_path = _load_optimized_config("breakout", base_config, mutate_breakout_config, data_end=data_end)
    data = load_breakout_replay_bundle(config.symbols, data_dir).data
    result = run_breakout_synchronized(data, config)

    all_trades = _collect_symbol_trades(result)

    is_months = _window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE)
    oos_months = _compute_oos_months(data_end)
    is_m, oos_m = _split_and_analyze(all_trades, is_months, oos_months)
    assessment, action = _assess("breakout", is_m, oos_m)
    warnings = _audit_paths(
        [data_dir / f"{symbol}_{tf}.parquet" for symbol in config.symbols for tf in ("1h", "1d")],
        data_end,
    ) + _baseline_warnings("breakout", config_path, is_m)

    return OOSResult(
        strategy="breakout", family="swing",
        is_metrics=is_m, oos_metrics=oos_m,
        assessment=assessment, action=action,
        config_path=str(config_path),
        warnings=warnings,
    )


def run_brs(data_end: str) -> OOSResult:
    """Run BRS frozen-parameter OOS validation."""
    from backtests.swing.config_brs import BRSConfig
    from backtests.swing.auto.brs.config_mutator import mutate_brs_config
    from backtests.swing.engine.brs_portfolio_engine import run_brs_synchronized
    from backtests.swing.data.replay_cache import load_brs_replay_bundle

    data_dir = PROJECT_ROOT / "backtests" / "swing" / "data" / "raw"
    base_config = BRSConfig(
        initial_equity=10_000,
        data_dir=data_dir,
    )
    config, config_path = _load_optimized_config("brs", base_config, mutate_brs_config, data_end=data_end)
    data = load_brs_replay_bundle(config).data
    result = run_brs_synchronized(data, config)

    all_trades = _collect_symbol_trades(result)
    is_months = _window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE)
    oos_months = _compute_oos_months(data_end)
    is_m, oos_m = _split_and_analyze(all_trades, is_months, oos_months)
    assessment, action = _assess("brs", is_m, oos_m)
    warnings = _audit_paths(
        [data_dir / f"{symbol}_{tf}.parquet" for symbol in config.symbols for tf in ("1h", "1d")],
        data_end,
    ) + _baseline_warnings("brs", config_path, is_m)

    return OOSResult(
        strategy="brs", family="swing",
        is_metrics=is_m, oos_metrics=oos_m,
        assessment=assessment, action=action,
        config_path=str(config_path),
        warnings=warnings,
    )


def run_helix_momentum(data_end: str) -> OOSResult:
    """Run Momentum AKC Helix frozen-parameter OOS validation."""
    from backtests.momentum.auto.config_mutator import mutate_helix_config
    from backtests.momentum.cli import _load_helix_data_cached
    from backtests.momentum.config_helix import Helix4BacktestConfig
    from backtests.momentum.engine.helix_engine import Helix4Engine

    data_dir = PROJECT_ROOT / "backtests" / "momentum" / "data" / "raw"
    base_config = Helix4BacktestConfig(
        initial_equity=10_000,
        data_dir=data_dir,
        fixed_qty=10,
        track_signals=False,
        track_shadows=False,
    )
    config, config_path = _load_optimized_config(
        "helix_momentum", base_config, mutate_helix_config, data_end=data_end
    )
    data = _load_helix_data_cached("NQ", data_dir)
    engine = Helix4Engine("NQ", config)
    result = engine.run(
        data["minute_bars"],
        data["hourly"],
        data["four_hour"],
        data["daily"],
        data["hourly_idx_map"],
        data["four_hour_idx_map"],
        data["daily_idx_map"],
    )

    is_months = _window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE)
    oos_months = _compute_oos_months(data_end)
    is_m, oos_m = _split_and_analyze(result.trades, is_months, oos_months)
    assessment, action = _assess("helix_momentum", is_m, oos_m)
    warnings = _audit_paths([data_dir / "NQ_5m.parquet"], data_end) + _baseline_warnings(
        "helix_momentum", config_path, is_m
    )

    return OOSResult(
        strategy="helix_momentum", family="momentum",
        is_metrics=is_m, oos_metrics=oos_m,
        assessment=assessment, action=action,
        config_path=str(config_path),
        warnings=warnings,
    )


def run_vdubus(data_end: str) -> OOSResult:
    """Run VdubusNQ frozen-parameter OOS validation."""
    from backtests.momentum.auto.config_mutator import mutate_vdubus_config
    from backtests.momentum.config_vdubus import VdubusAblationFlags, VdubusBacktestConfig
    from backtests.momentum.engine.vdubus_engine import VdubusEngine
    from backtests.momentum.data.replay_cache import (
        load_vdub_replay_bundle,
        replay_engine_kwargs,
    )

    data_dir = PROJECT_ROOT / "backtests" / "momentum" / "data" / "raw"
    base_config = VdubusBacktestConfig(
        initial_equity=10_000,
        data_dir=data_dir,
        fixed_qty=10,
        flags=VdubusAblationFlags(heat_cap=False, viability_filter=False),
        track_signals=False,
        track_shadows=False,
    )
    config, config_path = _load_optimized_config("vdubus", base_config, mutate_vdubus_config, data_end=data_end)
    bundle = load_vdub_replay_bundle("NQ", data_dir, include_5m=True)
    kwargs = replay_engine_kwargs(bundle)
    engine = VdubusEngine("NQ", config)
    result = engine.run(**kwargs)

    is_months = _window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE)
    oos_months = _compute_oos_months(data_end)
    is_m, oos_m = _split_and_analyze(result.trades, is_months, oos_months)
    assessment, action = _assess("vdubus", is_m, oos_m)
    warnings = _audit_paths([data_dir / "NQ_5m.parquet", data_dir / "ES_1d.parquet"], data_end) + _baseline_warnings(
        "vdubus", config_path, is_m
    )

    return OOSResult(
        strategy="vdubus", family="momentum",
        is_metrics=is_m, oos_metrics=oos_m,
        assessment=assessment, action=action,
        config_path=str(config_path),
        warnings=warnings,
    )


def run_nqdtc(data_end: str) -> OOSResult:
    """Run NQDTC frozen-parameter OOS validation."""
    from backtests.momentum.auto.config_mutator import mutate_nqdtc_config
    from backtests.momentum.auto.nqdtc.worker import load_worker_data
    from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
    from backtests.momentum.data.replay_cache import replay_engine_kwargs
    from backtests.momentum.engine.nqdtc_engine import NQDTCEngine

    data_dir = PROJECT_ROOT / "backtests" / "momentum" / "data" / "raw"
    base_config = NQDTCBacktestConfig(
        initial_equity=10_000,
        data_dir=data_dir,
        fixed_qty=10,
        track_signals=False,
        track_shadows=False,
        scoring_mode=False,
        max_dd_abort=0.0,
    )
    config, config_path = _load_optimized_config("nqdtc", base_config, mutate_nqdtc_config, data_end=data_end)
    bundle = load_worker_data("NQ", data_dir)
    kwargs = replay_engine_kwargs(bundle)
    engine = NQDTCEngine("MNQ", config)
    result = engine.run(**kwargs)

    is_months = _window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE)
    oos_months = _compute_oos_months(data_end)
    is_m, oos_m = _split_and_analyze(result.trades, is_months, oos_months)
    assessment, action = _assess("nqdtc", is_m, oos_m)
    warnings = _audit_paths([data_dir / "NQ_5m.parquet", data_dir / "ES_1d.parquet"], data_end) + _baseline_warnings(
        "nqdtc", config_path, is_m
    )

    return OOSResult(
        strategy="nqdtc", family="momentum",
        is_metrics=is_m, oos_metrics=oos_m,
        assessment=assessment, action=action,
        config_path=str(config_path),
        warnings=warnings,
    )


def run_downturn(data_end: str) -> OOSResult:
    """Run Downturn frozen-parameter OOS validation."""
    from backtests.momentum.auto.downturn.config_mutator import mutate_downturn_config
    from backtests.momentum.auto.downturn.worker import load_worker_data
    from backtests.momentum.config_downturn import DownturnBacktestConfig
    from backtests.momentum.data.replay_cache import replay_engine_kwargs
    from backtests.momentum.engine.downturn_engine import DownturnEngine

    data_dir = PROJECT_ROOT / "backtests" / "momentum" / "data" / "raw"
    base_config = DownturnBacktestConfig(
        initial_equity=10_000,
        data_dir=data_dir,
        track_signals=False,
        skip_parity_output=True,
        max_dd_abort=0.0,
    )
    config, config_path = _load_optimized_config("downturn", base_config, mutate_downturn_config, data_end=data_end)
    bundle = load_worker_data("NQ", data_dir)
    kwargs = replay_engine_kwargs(bundle)
    engine = DownturnEngine("NQ", config)
    result = engine.run(**kwargs)

    is_months = _window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE)
    oos_months = _compute_oos_months(data_end)
    is_m, oos_m = _split_and_analyze(result.trades, is_months, oos_months)
    assessment, action = _assess("downturn", is_m, oos_m)
    warnings = _audit_paths([data_dir / "NQ_5m.parquet", data_dir / "ES_1d.parquet"], data_end) + _baseline_warnings(
        "downturn", config_path, is_m
    )

    return OOSResult(
        strategy="downturn", family="momentum",
        is_metrics=is_m, oos_metrics=oos_m,
        assessment=assessment, action=action,
        config_path=str(config_path),
        warnings=warnings,
    )


def run_nq_regime(data_end: str) -> OOSResult:
    """Run NQ Regime frozen-parameter OOS validation."""
    from backtests.momentum.auto.nq_regime.worker import mutate_config
    from backtests.momentum.config_regime import NqRegimeBacktestConfig
    from backtests.momentum.engine.regime_engine import load_nq_regime_data, run_nq_regime_backtest

    data_dir = PROJECT_ROOT / "backtests" / "momentum" / "data" / "raw"
    base_config = NqRegimeBacktestConfig(
        start_date=BACKTEST_START.replace(tzinfo=timezone.utc),
        end_date=datetime.combine(date.fromisoformat(data_end), datetime.min.time(), tzinfo=timezone.utc),
        initial_equity=10_000,
        data_dir=data_dir,
        analysis_symbol="NQ",
        trade_symbol="MNQ",
        fixed_qty=10,
        track_decisions=False,
    )
    config, config_path = _load_optimized_config("nq_regime", base_config, mutate_config, data_end=data_end)
    data = load_nq_regime_data(config)
    result = run_nq_regime_backtest(data, config)

    is_months = _window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE)
    oos_months = _compute_oos_months(data_end)
    is_m, oos_m = _split_and_analyze(result.trades, is_months, oos_months)
    assessment, action = _assess("nq_regime", is_m, oos_m)
    warnings = _audit_paths([data_dir / "NQ_5m.parquet"], data_end) + _baseline_warnings(
        "nq_regime", config_path, is_m
    )

    return OOSResult(
        strategy="nq_regime", family="momentum",
        is_metrics=is_m, oos_metrics=oos_m,
        assessment=assessment, action=action,
        config_path=str(config_path),
        warnings=warnings,
    )


def run_stock_portfolio(data_end: str) -> OOSResult:
    """Run Stock portfolio-synergy frozen-config OOS validation."""
    from backtests.stock.auto.portfolio_synergy.core.logic import run_portfolio_replay
    from backtests.stock.auto.portfolio_synergy.evaluator import (
        build_effective_portfolio_config,
        load_evaluation_bundle,
    )
    from backtests.stock.auto.portfolio_synergy.phase_candidates import INITIAL_EQUITY

    data_dir = PROJECT_ROOT / "backtests" / "stock" / "data" / "raw"
    config_path = resolve_optimized_config_path("stock_portfolio")
    mutations = _load_mutations_file(config_path)
    initial_equity = float(mutations.get("initial_equity", INITIAL_EQUITY))
    bundle = load_evaluation_bundle(
        data_dir,
        initial_equity=initial_equity,
        start_date=BACKTEST_START_DATE.isoformat(),
        end_date=data_end,
    )
    effective = build_effective_portfolio_config(mutations, initial_equity=initial_equity)
    result = run_portfolio_replay(bundle.data.alcb_trades, bundle.data.iaric_trades, effective)
    trades = [
        {"entry_time": trade.entry_time, "r_multiple": float(trade.r_multiple)}
        for trade in result.trade_outcomes
    ]

    is_months = _window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE)
    oos_months = _compute_oos_months(data_end)
    is_m, oos_m = _split_and_analyze(trades, is_months, oos_months)
    assessment, action = _assess("stock_portfolio", is_m, oos_m)
    warnings = _audit_stock_intraday(data_dir, data_end) + _baseline_warnings("stock_portfolio", config_path, is_m)

    return OOSResult(
        strategy="stock_portfolio", family="stock",
        is_metrics=is_m, oos_metrics=oos_m,
        assessment=assessment, action=action,
        config_path=str(config_path),
        warnings=warnings,
    )


def run_swing_portfolio(data_end: str) -> OOSResult:
    """Run Swing portfolio-synergy frozen-config OOS validation."""
    from backtests.swing.auto.config_mutator import mutate_unified_config
    from backtests.swing.config_unified import UnifiedBacktestConfig
    from backtests.swing.engine.unified_portfolio_engine import load_unified_data, run_unified

    data_dir = PROJECT_ROOT / "backtests" / "swing" / "data" / "raw"
    config_path = resolve_optimized_config_path("swing_portfolio")
    mutations = _load_mutations_file(config_path)
    initial_equity = float(mutations.get("initial_equity", 50_000.0))
    base_config = UnifiedBacktestConfig(
        initial_equity=initial_equity,
        data_dir=data_dir,
        start_date=BACKTEST_START_DATE.isoformat(),
        end_date=data_end,
    )
    config = mutate_unified_config(base_config, mutations)
    data = load_unified_data(config)
    result = run_unified(data, config)
    trades: list[Any] = []
    for attr in ("atrss_trades", "helix_trades", "tpc_trades"):
        trades.extend(getattr(result, attr, []) or [])

    is_months = _window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE)
    oos_months = _compute_oos_months(data_end)
    is_m, oos_m = _split_and_analyze(trades, is_months, oos_months)
    assessment, action = _assess("swing_portfolio", is_m, oos_m)
    warnings = _audit_paths(
        [
            data_dir / f"{symbol}_{tf}.parquet"
            for symbol in ("QQQ", "GLD")
            for tf in ("15m", "1h", "1d")
        ]
        + [
            data_dir / f"{symbol}_{tf}.parquet"
            for symbol in ("NQ", "GC")
            for tf in ("1h", "1d")
        ],
        data_end,
    ) + _baseline_warnings("swing_portfolio", config_path, is_m)

    return OOSResult(
        strategy="swing_portfolio", family="swing",
        is_metrics=is_m, oos_metrics=oos_m,
        assessment=assessment, action=action,
        config_path=str(config_path),
        warnings=warnings,
    )


def run_momentum_portfolio(data_end: str) -> OOSResult:
    """Run Momentum portfolio-synergy frozen-config OOS validation."""
    from dataclasses import replace as dc_replace

    from backtests.momentum.auto.portfolio_synergy.family_phase_auto import (
        load_or_build_latest_strategy_trades,
    )
    from backtests.momentum.engine.family_portfolio_engine import (
        FamilyPortfolioBacktester,
        family_config_from_dict,
    )

    data_dir = PROJECT_ROOT / "backtests" / "momentum" / "data" / "raw"
    config_path = resolve_optimized_config_path("momentum_portfolio")
    config_payload = _load_mutations_file(config_path)
    config = family_config_from_dict(config_payload)
    config = dc_replace(
        config,
        start_date=BACKTEST_START,
        end_date=datetime.combine(date.fromisoformat(data_end), datetime.min.time()),
    )
    trades_by_strategy = load_or_build_latest_strategy_trades(
        data_dir=data_dir,
        output_dir=config_path.parent,
        initial_equity=float(config.initial_equity),
    )
    result = FamilyPortfolioBacktester(config).run(trades_by_strategy)
    reference_risk = max(float(config.reference_unit_risk_dollars), 1e-9)
    trades = [
        {
            "entry_time": trade.entry_time,
            "r_multiple": float(trade.adjusted_pnl) / reference_risk,
        }
        for trade in result.trades
    ]

    is_months = _window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE)
    oos_months = _compute_oos_months(data_end)
    is_m, oos_m = _split_and_analyze(trades, is_months, oos_months)
    assessment, action = _assess("momentum_portfolio", is_m, oos_m)
    warnings = _audit_paths([data_dir / "NQ_5m.parquet", data_dir / "ES_1d.parquet"], data_end) + _baseline_warnings(
        "momentum_portfolio", config_path, is_m
    )

    return OOSResult(
        strategy="momentum_portfolio", family="momentum",
        is_metrics=is_m, oos_metrics=oos_m,
        assessment=assessment, action=action,
        config_path=str(config_path),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_oos_months(data_end: str) -> float:
    """Compute OOS period length in months from cutoff to data end."""
    end = date.fromisoformat(data_end)
    delta = end - OOS_CUTOFF_DATE
    return max(delta.days / 30.44, 0.1)


def _detect_data_end() -> str:
    """Auto-detect the end date from available data."""
    import pandas as pd

    references = [
        PROJECT_ROOT / "backtests" / "swing" / "data" / "raw" / "QQQ_1h.parquet",
        PROJECT_ROOT / "backtests" / "momentum" / "data" / "raw" / "NQ_5m.parquet",
    ]
    stock_dir = PROJECT_ROOT / "backtests" / "stock" / "data" / "raw"
    references.extend(sorted(stock_dir.glob("*_5m.parquet")))

    latest_dates: list[date] = []
    for path in references:
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if not df.empty:
            latest_dates.append(_to_naive_utc(df.index[-1]).date())

    if latest_dates:
        return max(latest_dates).isoformat()

    return "2026-03-27"  # fallback


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

STRATEGY_ORDER = [
    "iaric", "alcb", "stock_portfolio",
    "helix_swing", "atrss", "tpc", "swing_portfolio",
    "vdubus", "nqdtc", "downturn", "nq_regime", "momentum_portfolio",
]
STRATEGY_GROUPS = {
    "all": STRATEGY_ORDER,
    "stock": [s for s in STRATEGY_ORDER if STRATEGY_FAMILIES.get(s) == "stock"],
    "swing": [s for s in STRATEGY_ORDER if STRATEGY_FAMILIES.get(s) == "swing"],
    "momentum": [s for s in STRATEGY_ORDER if STRATEGY_FAMILIES.get(s) == "momentum"],
    "portfolios": ["stock_portfolio", "swing_portfolio", "momentum_portfolio"],
}

RUNNERS = {
    "iaric": run_iaric,
    "alcb": run_alcb,
    "stock_portfolio": run_stock_portfolio,
    "helix_swing": run_helix_swing,
    "atrss": run_atrss,
    "tpc": run_tpc,
    "swing_portfolio": run_swing_portfolio,
    "breakout": run_breakout,
    "brs": run_brs,
    "helix_momentum": run_helix_momentum,
    "vdubus": run_vdubus,
    "nqdtc": run_nqdtc,
    "downturn": run_downturn,
    "nq_regime": run_nq_regime,
    "momentum_portfolio": run_momentum_portfolio,
}


def expand_strategy_selection(items: list[str]) -> list[str]:
    """Expand strategy groups such as all/momentum while preserving order."""
    selected: list[str] = []
    seen: set[str] = set()
    for item in items:
        expanded = STRATEGY_GROUPS.get(item, [item])
        for strategy in expanded:
            if strategy in seen:
                continue
            seen.add(strategy)
            selected.append(strategy)
    return selected


def parse_config_overrides(items: list[str]) -> dict[str, Path]:
    """Parse repeated --config strategy=path overrides."""
    overrides: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--config must be strategy=path, got {item!r}")
        strategy, raw_path = item.split("=", 1)
        strategy = strategy.strip()
        if strategy not in RUNNERS:
            raise ValueError(f"--config references unknown strategy {strategy!r}")
        overrides[strategy] = Path(raw_path.strip())
    return overrides


def format_config_plan(strategies: list[str]) -> str:
    """Show resolved config paths for a strategy selection."""
    lines = ["Resolved optimized configs:"]
    for strategy in strategies:
        try:
            path = resolve_optimized_config_path(strategy)
            lines.append(f"  {strategy:<18} {path}")
        except Exception as exc:
            lines.append(f"  {strategy:<18} ERROR: {exc}")
    return "\n".join(lines)


def format_report(results: list[OOSResult], data_end: str) -> str:
    """Format OOS validation results as a readable report."""
    lines = []
    lines.append("=" * 80)
    lines.append("OOS VALIDATION REPORT")
    lines.append(
        f"IS Period: {BACKTEST_START_DATE.isoformat()} to {LAST_SEEN_DATA_DATE.isoformat()} "
        f"(~{_window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE):.1f} months)"
    )
    lines.append(f"OOS Period: {OOS_CUTOFF_DATE.isoformat()} to {data_end}")
    lines.append("Assessment basis: reproduced IS metrics from the same frozen replay")
    lines.append("Ref PF: stored guide/run-summary baseline, shown for audit context only")
    lines.append("=" * 80)
    lines.append("")

    # Summary table
    header = (
        f"{'Strategy':<18} {'OOS#':>5} {'OOS WR':>7} {'OOS PF':>7} "
        f"{'OOS AvgR':>8} {'Repro IS PF':>11} {'Ref PF':>7} "
        f"{'Warn':>4} {'Assessment':<8} {'Action'}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        if r.error:
            lines.append(f"{r.strategy:<18} ERROR: {r.error}")
            continue
        oos = r.oos_metrics
        is_m = r.is_metrics
        reference_pf = IS_BASELINES.get(r.strategy, {}).get("pf", 0)
        lines.append(
            f"{r.strategy:<18} {oos.total_trades:>5} "
            f"{oos.win_rate:>6.1%} {oos.profit_factor:>7.2f} "
            f"{oos.avg_r:>8.3f} {is_m.profit_factor:>11.2f} "
            f"{reference_pf:>7.2f} "
            f"{len(r.warnings):>4} "
            f"{r.assessment:<8} {r.action}"
        )

    lines.append("")
    lines.append("=" * 80)
    lines.append("DETAILED BREAKDOWN")
    lines.append("=" * 80)

    for r in results:
        if r.error:
            continue
        lines.append("")
        lines.append(f"Strategy: {r.strategy} ({r.family})")
        lines.append(f"{'':->60}")
        lines.append(f"{'':20} {'IS (26 mo)':>12} {'OOS':>12} {'Delta':>10}")
        is_m = r.is_metrics
        oos_m = r.oos_metrics

        lines.append(f"{'Trades:':<20} {is_m.total_trades:>12} {oos_m.total_trades:>12}")
        lines.append(f"{'Win Rate:':<20} {is_m.win_rate:>11.1%} {oos_m.win_rate:>11.1%} {(oos_m.win_rate - is_m.win_rate):>+9.1%}")
        pf_delta = oos_m.profit_factor - is_m.profit_factor
        lines.append(f"{'Profit Factor:':<20} {is_m.profit_factor:>12.2f} {oos_m.profit_factor:>12.2f} {pf_delta:>+10.2f}")
        lines.append(f"{'Avg R/trade:':<20} {is_m.avg_r:>12.3f} {oos_m.avg_r:>12.3f} {(oos_m.avg_r - is_m.avg_r):>+10.3f}")
        lines.append(f"{'Net R:':<20} {is_m.net_r:>12.1f} {oos_m.net_r:>12.1f}")
        lines.append(f"{'Max DD (R):':<20} {is_m.max_drawdown_r:>12.2f} {oos_m.max_drawdown_r:>12.2f}")
        lines.append(f"{'Trades/month:':<20} {is_m.trades_per_month:>12.1f} {oos_m.trades_per_month:>12.1f}")
        lines.append(f"{'Config:':<20} {r.config_path or 'n/a'}")
        if r.warnings:
            lines.append(f"{'Warnings:':<20} {r.warnings[0]}")
            for warning in r.warnings[1:]:
                lines.append(f"{'':<20} {warning}")
        lines.append(f"{'Assessment:':<20} {r.assessment}")
        lines.append(f"{'Action:':<20} {r.action}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="OOS Validation Runner")
    parser.add_argument(
        "--strategy", nargs="+", default=["all"],
        help="Strategy names or groups to validate: all, stock, swing, momentum, portfolios, or individual names",
    )
    parser.add_argument(
        "--family", nargs="+", choices=["all", "stock", "swing", "momentum"], default=[],
        help="Optional family/group selector. Overrides --strategy when supplied.",
    )
    parser.add_argument(
        "--data-end", default=None,
        help="Data end date (auto-detected if not specified)",
    )
    parser.add_argument(
        "--backtest-start", default=BACKTEST_START_DATE.isoformat(),
        help="Backtest start date for the IS/OOS split window.",
    )
    parser.add_argument(
        "--last-seen-data-date", default=LAST_SEEN_DATA_DATE.isoformat(),
        help="Last date included in the original in-sample/development window.",
    )
    parser.add_argument(
        "--oos-cutoff-date", default=None,
        help="First OOS date. Defaults to --last-seen-data-date + 1 day.",
    )
    parser.add_argument(
        "--config-root", default=str(CONFIG_ROOT),
        help="Root containing family/strategy round directories.",
    )
    parser.add_argument(
        "--config-resolution", choices=["auto", "manifest", "static"], default="auto",
        help="auto=latest round if discoverable, else static table; manifest=latest round only; static=legacy table.",
    )
    parser.add_argument(
        "--config", action="append", default=[],
        help="Override one config path as strategy=path. Can be repeated.",
    )
    parser.add_argument(
        "--list-configs", action="store_true",
        help="Print resolved configs for the selected strategies and exit.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file for the report (default: stdout + saved to backtests/output/oos/)",
    )
    parser.add_argument(
        "--json-output", default=None,
        help="Output file for structured JSON results (default: report path with .json suffix when --output is set).",
    )
    args = parser.parse_args()

    configure_validation_window(
        backtest_start=args.backtest_start,
        last_seen_data_date=args.last_seen_data_date,
        oos_cutoff_date=args.oos_cutoff_date,
    )
    configure_config_resolution(
        config_root=args.config_root,
        mode=args.config_resolution,
        overrides=parse_config_overrides(args.config),
    )

    strategies = expand_strategy_selection(args.family or args.strategy)
    if args.list_configs:
        print(format_config_plan(strategies))
        return 0

    data_end = args.data_end or _detect_data_end()
    logger.info(f"Data end date: {data_end}")
    logger.info(
        f"OOS window: {OOS_CUTOFF_DATE.isoformat()} to {data_end} "
        f"({_compute_oos_months(data_end):.1f} months)"
    )

    results: list[OOSResult] = []
    for strat in strategies:
        if strat not in RUNNERS:
            logger.error(f"Unknown strategy: {strat}")
            continue
        logger.info(f"Running {strat}...")
        try:
            result = RUNNERS[strat](data_end)
            results.append(result)
            logger.info(
                f"  {strat}: {result.oos_metrics.total_trades} OOS trades, "
                f"PF={result.oos_metrics.profit_factor:.2f}, "
                f"Assessment={result.assessment}"
            )
        except Exception as e:
            logger.error(f"  {strat} FAILED: {e}")
            traceback.print_exc()
            results.append(OOSResult(strategy=strat, family="unknown", error=str(e)))

    # Generate report
    report = format_report(results, data_end)
    print(report)

    # Save report
    output_dir = PROJECT_ROOT / "backtests" / "output" / "oos"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(output_dir / f"oos_validation_{data_end}.txt")
    Path(output_path).write_text(report, encoding="utf-8")
    logger.info(f"Report saved to: {output_path}")

    # Save structured results as JSON
    if args.json_output:
        json_path = Path(args.json_output)
    elif args.output:
        json_path = Path(output_path).with_suffix(".json")
    else:
        json_path = output_dir / f"oos_results_{data_end}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_data = []
    for r in results:
        json_data.append({
            "strategy": r.strategy,
            "family": r.family,
            "assessment": r.assessment,
            "action": r.action,
            "config_path": r.config_path,
            "warnings": r.warnings,
            "error": r.error,
            "is_metrics": asdict(r.is_metrics) if not r.error else None,
            "oos_metrics": asdict(r.oos_metrics) if not r.error else None,
        })
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    logger.info(f"JSON results saved to: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
