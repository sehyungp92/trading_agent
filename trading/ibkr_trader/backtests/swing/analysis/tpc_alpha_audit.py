"""TPC round-6 alpha audit.

The audit is diagnostic rather than an optimiser. It replays the round-6 TPC
config, exports realized trade cohorts, builds a first-failure setup funnel,
and compares the six-month validation window with rolling six-month windows
inside the training sample.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.swing.auto.tpc.plugin import _extract_tpc_metrics
from backtests.swing.auto.tpc.round5_oos_repair import (
    DATA_DIR,
    DEFAULT_TRAIN_END,
    infer_holdout_warmup,
    metrics_with_cohorts,
)
from backtests.swing.config_etf_base import ETFSlippageConfig
from backtests.swing.config_tpc import TPCBacktestConfig
from backtests.swing.data.replay_cache import load_tpc_replay_bundle
from backtests.swing.engine.etf_engine_base import ETFStrategyBacktestEngine
from backtests.swing.engine.tpc_engine import run_tpc_independent
from strategies.swing.tpc import STRATEGY_ID, context, gates, indicators, signals, stops
from strategies.swing.tpc import allocator
from strategies.swing.tpc.config import SYMBOL_CONFIGS
from strategies.swing.tpc.core import logic
from strategies.swing.tpc.core.state import TPCBarInput, TPCCoreState, TPCFill, TPCOrderUpdate
from strategies.swing.tpc.models import Direction, RegimeGrade

ROUND6_ROOT = ROOT / "backtests" / "output" / "swing" / "tpc" / "round_6"
DEFAULT_CONFIG_PATH = ROUND6_ROOT / "optimized_config.json"
INITIAL_EQUITY = 100_000.0
ET_TZ = "America/New_York"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    parser.add_argument("--max-shadow-setups", type=int, default=25_000)
    args = parser.parse_args()

    started = time.time()
    output_dir = args.output_dir or (
        ROUND6_ROOT / f"alpha_audit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    mutations = read_json(args.config)
    train_bundle = load_tpc_replay_bundle(DATA_DIR, end_date=args.train_end)
    full_bundle = load_tpc_replay_bundle(DATA_DIR, end_date=None)
    oos_warmup = infer_holdout_warmup(full_bundle.data, args.train_end)

    train_cfg = TPCBacktestConfig(initial_equity=INITIAL_EQUITY, data_dir=DATA_DIR).with_overrides(mutations)
    oos_cfg = TPCBacktestConfig(initial_equity=INITIAL_EQUITY, data_dir=DATA_DIR).with_overrides(
        {**mutations, "warmup_15m": oos_warmup}
    )

    print("[audit] replaying round-6 train", flush=True)
    train_result = run_tpc_independent(train_bundle.data, train_cfg)
    print("[audit] replaying round-6 OOS", flush=True)
    oos_result = run_tpc_independent(full_bundle.data, oos_cfg)

    results = {"train": train_result, "oos": oos_result}
    metrics = {
        split: {
            "headline": _extract_tpc_metrics(result, INITIAL_EQUITY),
            "with_cohorts": metrics_with_cohorts(result),
        }
        for split, result in results.items()
    }
    trade_rows = []
    for split, result in results.items():
        trade_rows.extend(trade_to_row(split, trade) for trade in result.trades)
    write_csv(output_dir / "trades.csv", trade_rows)
    write_csv(output_dir / "cohort_summary.csv", cohort_summary(trade_rows))
    write_csv(output_dir / "event_funnel.csv", event_funnel(results))
    write_csv(output_dir / "rolling_6m_trade_windows.csv", rolling_trade_windows(trade_rows, args.train_end))
    write_csv(output_dir / "market_summary.csv", market_summary(full_bundle.data, args.train_end))
    write_csv(output_dir / "top_winners.csv", top_trades(trade_rows, reverse=True))
    write_csv(output_dir / "top_losers.csv", top_trades(trade_rows, reverse=False))

    print("[audit] building first-failure setup funnel", flush=True)
    shadow_funnel, shadow_setups = build_shadow_funnel(
        {
            "train": (train_bundle.data, train_cfg),
            "oos": (full_bundle.data, oos_cfg),
        },
        max_shadow_setups=args.max_shadow_setups,
    )
    write_csv(output_dir / "shadow_setup_funnel.csv", shadow_funnel)
    write_csv(output_dir / "shadow_entry_ready_setups.csv", shadow_setups)

    summary = build_summary(
        metrics=metrics,
        trade_rows=trade_rows,
        shadow_funnel=shadow_funnel,
        output_dir=output_dir,
        train_end=args.train_end,
        oos_warmup=oos_warmup,
        elapsed_seconds=time.time() - started,
    )
    write_json(output_dir / "summary.json", summary)
    report = format_report(summary)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (ROUND6_ROOT / "alpha_audit_report.md").write_text(report, encoding="utf-8")
    write_json(ROUND6_ROOT / "alpha_audit_summary.json", summary)
    print(report, flush=True)
    print(f"[audit] output: {output_dir.resolve()}", flush=True)


def trade_to_row(split: str, trade: Any) -> dict[str, Any]:
    entry_ts = as_utc_ts(getattr(trade, "entry_time", None))
    exit_ts = as_utc_ts(getattr(trade, "exit_time", None))
    entry_et = entry_ts.tz_convert(ET_TZ) if entry_ts is not None else None
    r = float(getattr(trade, "r_multiple", 0.0) or 0.0)
    mfe = float(getattr(trade, "mfe_r", 0.0) or 0.0)
    mae = float(getattr(trade, "mae_r", 0.0) or 0.0)
    pnl = float(getattr(trade, "pnl_dollars", 0.0) or 0.0)
    direction = int(getattr(trade, "direction", 0) or 0)
    return {
        "split": split,
        "symbol": str(getattr(trade, "symbol", "") or ""),
        "direction": "LONG" if direction > 0 else "SHORT" if direction < 0 else "FLAT",
        "entry_time_utc": iso_or_blank(entry_ts),
        "exit_time_utc": iso_or_blank(exit_ts),
        "entry_date_et": entry_et.date().isoformat() if entry_et is not None else "",
        "entry_hour_et": int(entry_et.hour) if entry_et is not None else "",
        "entry_weekday_et": int(entry_et.weekday()) if entry_et is not None else "",
        "month": entry_ts.strftime("%Y-%m") if entry_ts is not None else "",
        "entry_type": str(getattr(trade, "entry_type", "") or ""),
        "entry_model": str(getattr(trade, "leg_type", "") or ""),
        "grade": str(getattr(trade, "regime_entry", "") or ""),
        "score": float(getattr(trade, "score_entry", 0.0) or 0.0),
        "exit_reason": str(getattr(trade, "exit_reason", "") or ""),
        "qty": int(getattr(trade, "qty", 0) or 0),
        "addon_qty": int(getattr(trade, "addon_a_qty", 0) or 0) + int(getattr(trade, "addon_b_qty", 0) or 0),
        "entry_price": float(getattr(trade, "entry_price", 0.0) or 0.0),
        "exit_price": float(getattr(trade, "exit_price", 0.0) or 0.0),
        "initial_stop": float(getattr(trade, "initial_stop", 0.0) or 0.0),
        "bars_held_15m": int(getattr(trade, "bars_held", 0) or 0),
        "pnl_dollars": pnl,
        "r_multiple": r,
        "mfe_r": mfe,
        "mae_r": mae,
        "excellent": bool(r > 0.0 and mfe >= 1.0 and mae <= 1.10),
        "never_worked": bool(mfe < 0.5 and r <= 0.0),
        "low_mfe_loss": bool(mfe < 1.0 and r <= 0.0),
        "right_then_lost": bool(mfe >= 1.0 and r <= 0.0),
        "capture_ratio": r / mfe if mfe > 0 and r > 0 else 0.0,
    }


def cohort_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs: dict[str, list[str]] = {
        "split": ["split"],
        "split_symbol": ["split", "symbol"],
        "split_symbol_direction": ["split", "symbol", "direction"],
        "split_symbol_grade": ["split", "symbol", "grade"],
        "split_symbol_hour": ["split", "symbol", "entry_hour_et"],
        "split_symbol_entry_model": ["split", "symbol", "entry_model"],
        "split_symbol_exit_reason": ["split", "symbol", "exit_reason"],
        "split_month_symbol": ["split", "month", "symbol"],
    }
    out: list[dict[str, Any]] = []
    for name, keys in specs.items():
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[tuple(row.get(key, "") for key in keys)].append(row)
        for group_key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
            out.append({"cohort": name, **dict(zip(keys, group_key)), **summarise_trade_items(items)})
    return out


def summarise_trade_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    rs = np.asarray([float(row["r_multiple"]) for row in items], dtype=float)
    pnls = np.asarray([float(row["pnl_dollars"]) for row in items], dtype=float)
    mfes = np.asarray([float(row["mfe_r"]) for row in items], dtype=float)
    maes = np.asarray([float(row["mae_r"]) for row in items], dtype=float)
    wins = rs[rs > 0]
    losses = rs[rs < 0]
    return {
        "trades": len(items),
        "pnl_dollars": float(np.sum(pnls)) if pnls.size else 0.0,
        "total_r": float(np.sum(rs)) if rs.size else 0.0,
        "avg_r": float(np.mean(rs)) if rs.size else 0.0,
        "median_r": float(np.median(rs)) if rs.size else 0.0,
        "win_rate": float(np.mean(rs > 0.0)) if rs.size else 0.0,
        "profit_factor_r": float(np.sum(wins) / abs(np.sum(losses))) if losses.size and abs(np.sum(losses)) > 0 else float(np.sum(wins)) if wins.size else 0.0,
        "avg_mfe_r": float(np.mean(mfes)) if mfes.size else 0.0,
        "avg_mae_r": float(np.mean(maes)) if maes.size else 0.0,
        "excellent_trades": int(sum(bool(row["excellent"]) for row in items)),
        "excellent_rate": float(np.mean([bool(row["excellent"]) for row in items])) if items else 0.0,
        "never_worked_rate": float(np.mean([bool(row["never_worked"]) for row in items])) if items else 0.0,
        "low_mfe_loss_rate": float(np.mean([bool(row["low_mfe_loss"]) for row in items])) if items else 0.0,
        "right_then_lost_rate": float(np.mean([bool(row["right_then_lost"]) for row in items])) if items else 0.0,
        "top1_win_r_share": top_win_share(rs, 1),
        "top5_win_r_share": top_win_share(rs, 5),
    }


def event_funnel(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, result in results.items():
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for event in getattr(result, "decision_stream", []):
            grouped[(str(event.get("symbol", "")), str(event.get("code", "")))].append(event)
        symbols = sorted({key[0] for key in grouped} | {"QQQ", "GLD"})
        for symbol in symbols:
            req = len(grouped.get((symbol, "ENTRY_REQUESTED"), []))
            fill = len(grouped.get((symbol, "ENTRY_FILLED"), []))
            terminal = len(grouped.get((symbol, "ORDER_TERMINAL"), []))
            for code in sorted({key[1] for key in grouped if key[0] == symbol}):
                rows.append(
                    {
                        "split": split,
                        "symbol": symbol,
                        "code": code,
                        "count": len(grouped[(symbol, code)]),
                        "entry_requests": req,
                        "entry_fills": fill,
                        "order_terminals": terminal,
                        "fill_per_request": fill / max(req, 1),
                        "terminal_per_request": terminal / max(req, 1),
                    }
                )
    return rows


def rolling_trade_windows(rows: list[dict[str, Any]], train_end: str) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    frame["entry_ts"] = pd.to_datetime(frame["entry_time_utc"], utc=True, errors="coerce")
    train_cutoff = pd.Timestamp(train_end)
    train_cutoff = train_cutoff.tz_localize("UTC") if train_cutoff.tzinfo is None else train_cutoff.tz_convert("UTC")
    train = frame[(frame["split"] == "train") & (frame["entry_ts"] < train_cutoff)].copy()
    if train.empty:
        return []
    starts = pd.date_range(train["entry_ts"].min().floor("D"), train_cutoff - pd.Timedelta(days=182), freq="MS", tz="UTC")
    out: list[dict[str, Any]] = []
    for start in starts:
        end = start + pd.DateOffset(months=6)
        window = train[(train["entry_ts"] >= start) & (train["entry_ts"] < end)]
        row = {
            "window_start": start.date().isoformat(),
            "window_end": end.date().isoformat(),
            **summarise_trade_frame(window),
        }
        for symbol in ("QQQ", "GLD"):
            sub = window[window["symbol"] == symbol]
            prefix = symbol.lower()
            stats = summarise_trade_frame(sub)
            row[f"{prefix}_trades"] = stats["trades"]
            row[f"{prefix}_pnl_dollars"] = stats["pnl_dollars"]
            row[f"{prefix}_avg_r"] = stats["avg_r"]
            row[f"{prefix}_excellent_trades"] = stats["excellent_trades"]
        out.append(row)

    oos = frame[frame["split"] == "oos"].copy()
    if not oos.empty:
        row = {"window_start": "OOS", "window_end": "OOS", **summarise_trade_frame(oos)}
        for symbol in ("QQQ", "GLD"):
            sub = oos[oos["symbol"] == symbol]
            prefix = symbol.lower()
            stats = summarise_trade_frame(sub)
            row[f"{prefix}_trades"] = stats["trades"]
            row[f"{prefix}_pnl_dollars"] = stats["pnl_dollars"]
            row[f"{prefix}_avg_r"] = stats["avg_r"]
            row[f"{prefix}_excellent_trades"] = stats["excellent_trades"]
        out.append(row)
    return out


def summarise_trade_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "trades": 0,
            "pnl_dollars": 0.0,
            "net_return_pct_on_initial": 0.0,
            "avg_r": 0.0,
            "win_rate": 0.0,
            "excellent_trades": 0,
            "excellent_rate": 0.0,
            "low_mfe_loss_rate": 0.0,
            "right_then_lost_rate": 0.0,
        }
    rs = frame["r_multiple"].astype(float).to_numpy()
    pnls = frame["pnl_dollars"].astype(float).to_numpy()
    return {
        "trades": int(len(frame)),
        "pnl_dollars": float(np.sum(pnls)),
        "net_return_pct_on_initial": float(np.sum(pnls) / INITIAL_EQUITY * 100.0),
        "avg_r": float(np.mean(rs)),
        "win_rate": float(np.mean(rs > 0)),
        "excellent_trades": int(frame["excellent"].astype(bool).sum()),
        "excellent_rate": float(frame["excellent"].astype(bool).mean()),
        "low_mfe_loss_rate": float(frame["low_mfe_loss"].astype(bool).mean()),
        "right_then_lost_rate": float(frame["right_then_lost"].astype(bool).mean()),
    }


def market_summary(data: dict[str, dict[str, Any]], train_end: str) -> list[dict[str, Any]]:
    cutoff = pd.Timestamp(train_end)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    out: list[dict[str, Any]] = []
    for symbol, payload in data.items():
        daily = bars_to_frame(payload["bars_daily"])
        four_h = bars_to_frame(payload["bars_4h"])
        for split, mask in {
            "train": daily.index < cutoff,
            "oos": daily.index >= cutoff,
        }.items():
            sub = daily.loc[mask].dropna()
            sub4 = four_h.loc[four_h.index < cutoff] if split == "train" else four_h.loc[four_h.index >= cutoff]
            rets = sub["close"].pct_change().dropna()
            close = sub["close"]
            above20 = close > close.rolling(20).mean()
            above50 = close > close.rolling(50).mean()
            out.append(
                {
                    "split": split,
                    "symbol": symbol,
                    "start": sub.index.min().date().isoformat() if len(sub) else "",
                    "end": sub.index.max().date().isoformat() if len(sub) else "",
                    "daily_bars": int(len(sub)),
                    "four_h_bars": int(len(sub4)),
                    "buy_hold_return_pct": pct_return(close),
                    "daily_realized_vol_ann": float(rets.std() * math.sqrt(252)) if len(rets) > 1 else 0.0,
                    "daily_mean_abs_return": float(rets.abs().mean()) if len(rets) else 0.0,
                    "buy_hold_max_dd_pct": max_drawdown_pct(close.to_numpy()),
                    "positive_day_rate": float((rets > 0).mean()) if len(rets) else 0.0,
                    "above_daily_sma20_rate": float(above20.mean()) if len(above20) else 0.0,
                    "above_daily_sma50_rate": float(above50.mean()) if len(above50) else 0.0,
                    "daily_autocorr_1": float(rets.autocorr(lag=1)) if len(rets) > 3 else 0.0,
                }
            )
    return out


def build_shadow_funnel(
    split_data: dict[str, tuple[dict[str, dict[str, Any]], TPCBacktestConfig]],
    *,
    max_shadow_setups: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    setup_rows: list[dict[str, Any]] = []
    for split, (data, cfg) in split_data.items():
        engine = make_engine(cfg)
        prepared = {sym: engine._prepare_symbol(sym, payload) for sym, payload in data.items() if sym in cfg.symbol_configs}
        primary = max(prepared, key=lambda sym: len(prepared[sym]["bars_15m"].closes))
        start = min(max(cfg.warmup_15m, 1), max(len(prepared[primary]["bars_15m"]) - 1, 1))
        counters: dict[tuple[str, str, str, str], int] = Counter()
        meta_accum: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for i in range(start, len(prepared[primary]["bars_15m"])):
            for symbol, payload in prepared.items():
                if i >= len(payload["bars_15m"]):
                    continue
                cfg_sym = cfg.symbol_configs[symbol]
                bi = engine._bar_input(symbol, payload, i, INITIAL_EQUITY)
                reason, setup = shadow_first_failure(bi, cfg_sym)
                direction = setup.get("direction", "") if setup else ""
                grade = setup.get("grade", "") if setup else ""
                key = (symbol, direction, grade, reason)
                counters[key] += 1
                if setup:
                    meta_accum[key]["score_sum"] += float(setup.get("score", 0.0) or 0.0)
                    meta_accum[key]["depth_sum"] += float(setup.get("depth", 0.0) or 0.0)
                    meta_accum[key]["value_hits_sum"] += float(setup.get("value_hits", 0.0) or 0.0)
                if reason == "entry_ready" and len(setup_rows) < max_shadow_setups and setup:
                    setup_rows.append({"split": split, "symbol": symbol, **setup})
        for (symbol, direction, grade, reason), count in sorted(counters.items()):
            accum = meta_accum[(symbol, direction, grade, reason)]
            rows.append(
                {
                    "split": split,
                    "symbol": symbol,
                    "direction": direction,
                    "grade": grade,
                    "first_failure": reason,
                    "bars": count,
                    "avg_score": accum.get("score_sum", 0.0) / count if count else 0.0,
                    "avg_depth": accum.get("depth_sum", 0.0) / count if count else 0.0,
                    "avg_value_hits": accum.get("value_hits_sum", 0.0) / count if count else 0.0,
                }
            )
    return rows, setup_rows


def shadow_first_failure(bar_input: TPCBarInput, cfg: Any) -> tuple[str, dict[str, Any] | None]:
    bar = bar_input.bar_15m
    if bar is None:
        return "no_15m_bar", None
    if not gates.session_filter(bar.timestamp, cfg):
        return "session_filter", None
    if not gates.news_filter(bar.timestamp, cfg):
        return "news_filter", None
    direction, grade, regime_reason = gates.regime_direction(bar_input, cfg)
    if direction == Direction.FLAT:
        return f"regime_{regime_reason}", None
    direction_name = "LONG" if direction == Direction.LONG else "SHORT"
    base = {
        "timestamp_utc": iso_or_blank(as_utc_ts(bar.timestamp)),
        "direction": direction_name,
        "grade": grade.value if hasattr(grade, "value") else str(grade),
    }
    if direction == Direction.LONG and not cfg.longs_enabled:
        return "direction_longs_disabled", base
    if direction == Direction.SHORT:
        if not cfg.shorts_enabled:
            return "direction_shorts_disabled", base
        if cfg.shorts_require_a_plus and grade != RegimeGrade.A_PLUS:
            return "direction_short_not_a_plus", base
    pullback = signals.detect_pullback(bar_input, direction, grade, cfg)
    if pullback is None:
        return "pullback_none", base
    base.update(
        {
            "pullback_type": pullback.pullback_type.value,
            "depth": float(pullback.depth),
            "value_hits": int(pullback.value_hits),
            "orderly": bool(pullback.orderly),
        }
    )
    ok, confirmations = signals.check_confirmation(bar_input, direction, cfg)
    base["confirmations"] = "|".join(confirmations)
    base["confirmation_count"] = len(set(confirmations))
    if not ok:
        return "confirmation_count", base
    if not logic._confirmation_combo_allowed(confirmations, cfg):
        return "confirmation_combo", base
    if cfg.confirmation_max_count > 0 and len(set(confirmations)) > cfg.confirmation_max_count:
        return "confirmation_max_count", base
    atr4 = float(bar_input.indicators.get("atr_4h", np.nan))
    if not np.isfinite(atr4) or atr4 <= 0:
        return "atr_invalid", base
    entry_plan = logic._entry_plan(bar_input, direction, cfg)
    if entry_plan is None:
        return "entry_plan", base
    entry, _order_type, _limit, _stop_px, entry_model = entry_plan
    stop = logic._initial_stop(bar_input, pullback, direction, entry, atr4, cfg)
    base["entry_model"] = entry_model
    base["entry_price"] = float(entry)
    base["stop_price"] = float(stop)
    base["stop_atr_mult"] = abs(float(entry) - float(stop)) / max(atr4, 1e-9)
    if not stops.validate_stop(stop, entry, atr4, cfg):
        return "stop_invalid", base
    daily_levels = logic._daily_levels(bar_input)
    daily_has_room = gates.daily_room_filter(entry, stop, direction, daily_levels, cfg.daily_room_min_r)
    base["daily_has_room"] = bool(daily_has_room)
    if not daily_has_room:
        return "daily_room", base
    asset_context_score, asset_context_details = context.score_asset_context(bar_input, direction, cfg)
    base["asset_context_score"] = float(asset_context_score)
    base["asset_context_details"] = asset_context_details
    if asset_context_score < cfg.asset_context_min_score:
        return "asset_context", base
    score = allocator.score_setup(
        grade,
        pullback.pullback_type,
        confirmations,
        3.0,
        has_news_risk=False,
        asset_context_score=asset_context_score,
        daily_has_room=daily_has_room,
        orderly_pullback=pullback.orderly,
        score_model=cfg.score_model,
    )
    base["score"] = float(score)
    if direction == Direction.SHORT and cfg.min_short_score > 0 and score < cfg.min_short_score:
        return "short_score", base
    risk_pct = allocator.compute_risk_pct(score, pullback.pullback_type, cfg)
    if risk_pct is None:
        return "score_no_risk", base
    qty = allocator.compute_position_size(INITIAL_EQUITY, risk_pct, entry, stop, cfg)
    base["risk_pct"] = float(risk_pct)
    base["qty"] = int(qty)
    if qty <= 0:
        return "qty_zero", base
    return "entry_ready", base


def make_engine(cfg: TPCBacktestConfig) -> ETFStrategyBacktestEngine:
    return ETFStrategyBacktestEngine(
        strategy_id=STRATEGY_ID,
        configs=cfg.symbol_configs,
        core_logic=logic,
        state_factory=TPCCoreState,
        bar_input_factory=TPCBarInput,
        fill_factory=TPCFill,
        order_update_factory=TPCOrderUpdate,
        indicator_module=indicators,
        slippage=ETFSlippageConfig(),
        initial_equity=cfg.initial_equity,
        warmup_15m=cfg.warmup_15m,
    )


def top_trades(rows: list[dict[str, Any]], *, reverse: bool) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row["r_multiple"]), reverse=reverse)[:20]


def build_summary(
    *,
    metrics: dict[str, Any],
    trade_rows: list[dict[str, Any]],
    shadow_funnel: list[dict[str, Any]],
    output_dir: Path,
    train_end: str,
    oos_warmup: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    cohort_rows = cohort_summary(trade_rows)
    by = {(row.get("cohort"), row.get("split"), row.get("symbol", "")): row for row in cohort_rows}
    rolling = rolling_trade_windows(trade_rows, train_end)
    train_windows = [row for row in rolling if row.get("window_start") != "OOS"]
    oos_window = next((row for row in rolling if row.get("window_start") == "OOS"), {})
    shadow = summarise_shadow(shadow_funnel)
    qqq_train = by.get(("split_symbol", "train", "QQQ"), {})
    qqq_oos = by.get(("split_symbol", "oos", "QQQ"), {})
    gld_train = by.get(("split_symbol", "train", "GLD"), {})
    gld_oos = by.get(("split_symbol", "oos", "GLD"), {})
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir.resolve()),
        "train_end": train_end,
        "oos_warmup_15m": oos_warmup,
        "elapsed_minutes": elapsed_seconds / 60.0,
        "metrics": metrics,
        "symbol_trade_cohorts": {
            "train_QQQ": qqq_train,
            "oos_QQQ": qqq_oos,
            "train_GLD": gld_train,
            "oos_GLD": gld_oos,
        },
        "rolling_6m_context": {
            "train_window_count": len(train_windows),
            "oos_window": oos_window,
            "train_net_return_pct_percentiles": percentiles([row["net_return_pct_on_initial"] for row in train_windows]),
            "train_total_trades_percentiles": percentiles([row["trades"] for row in train_windows]),
            "train_qqq_trades_percentiles": percentiles([row["qqq_trades"] for row in train_windows]),
            "train_gld_trades_percentiles": percentiles([row["gld_trades"] for row in train_windows]),
            "oos_net_return_pct_rank_vs_train_6m": percentile_rank(
                [row["net_return_pct_on_initial"] for row in train_windows],
                float(oos_window.get("net_return_pct_on_initial", 0.0)),
            ),
            "oos_trade_count_rank_vs_train_6m": percentile_rank(
                [row["trades"] for row in train_windows],
                float(oos_window.get("trades", 0.0)),
            ),
        },
        "shadow_setup_context": shadow,
        "interpretation_flags": interpretation_flags(metrics, qqq_train, qqq_oos, gld_train, gld_oos, shadow),
    }


def summarise_shadow(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "oos"):
        for symbol in ("QQQ", "GLD"):
            sub = [row for row in rows if row["split"] == split and row["symbol"] == symbol]
            total = sum(int(row["bars"]) for row in sub)
            ready = sum(int(row["bars"]) for row in sub if row["first_failure"] == "entry_ready")
            pullback_none = sum(int(row["bars"]) for row in sub if row["first_failure"] == "pullback_none")
            confirmation = sum(int(row["bars"]) for row in sub if row["first_failure"].startswith("confirmation"))
            regime = sum(int(row["bars"]) for row in sub if row["first_failure"].startswith("regime_"))
            session = sum(int(row["bars"]) for row in sub if row["first_failure"] == "session_filter")
            out[f"{split}_{symbol}"] = {
                "shadow_bars": total,
                "entry_ready_bars": ready,
                "entry_ready_rate": ready / max(total, 1),
                "regime_reject_rate": regime / max(total, 1),
                "session_reject_rate": session / max(total, 1),
                "pullback_reject_rate": pullback_none / max(total, 1),
                "confirmation_reject_rate": confirmation / max(total, 1),
                "top_first_failures": sorted(
                    [{"reason": row["first_failure"], "bars": int(row["bars"])} for row in sub],
                    key=lambda row: row["bars"],
                    reverse=True,
                )[:8],
            }
    return out


def interpretation_flags(
    metrics: dict[str, Any],
    qqq_train: dict[str, Any],
    qqq_oos: dict[str, Any],
    gld_train: dict[str, Any],
    gld_oos: dict[str, Any],
    shadow: dict[str, Any],
) -> dict[str, Any]:
    train_head = metrics["train"]["headline"]
    oos_head = metrics["oos"]["headline"]
    train_top5 = float(train_head.get("top5_winner_share", 0.0))
    oos_trades = float(oos_head.get("total_trades", 0.0))
    qqq_oos_n = float(qqq_oos.get("trades", 0.0) or 0.0)
    qqq_train_ex_rate = float(qqq_train.get("excellent_rate", 0.0) or 0.0)
    prob_zero_qqq_ex = (1.0 - qqq_train_ex_rate) ** qqq_oos_n if qqq_oos_n >= 0 else 0.0
    return {
        "train_r_is_top_winner_concentrated": train_top5 >= 0.60,
        "oos_sample_is_small": oos_trades < 25,
        "qqq_oos_sample_is_too_small_for_decisive_rejection": qqq_oos_n < 10,
        "prob_zero_qqq_excellent_given_train_rate": prob_zero_qqq_ex,
        "qqq_oos_negative_avg_r": float(qqq_oos.get("avg_r", 0.0) or 0.0) < 0,
        "gld_oos_positive_avg_r": float(gld_oos.get("avg_r", 0.0) or 0.0) > 0,
        "asset_context_is_stubbed_in_code": False,
        "news_filter_is_stubbed_in_code": True,
        "score_gets_asset_context_credit_unconditionally": False,
        "shadow_qqq_oos_entry_ready_rate": shadow.get("oos_QQQ", {}).get("entry_ready_rate", 0.0),
        "shadow_gld_oos_entry_ready_rate": shadow.get("oos_GLD", {}).get("entry_ready_rate", 0.0),
    }


def format_report(summary: dict[str, Any]) -> str:
    cohorts = summary["symbol_trade_cohorts"]
    rolling = summary["rolling_6m_context"]
    flags = summary["interpretation_flags"]
    shadow = summary["shadow_setup_context"]
    train = summary["metrics"]["train"]["headline"]
    oos = summary["metrics"]["oos"]["headline"]

    def cohort_line(name: str, row: dict[str, Any]) -> str:
        return (
            f"{name}: trades {row.get('trades', 0):.0f}, pnl ${row.get('pnl_dollars', 0.0):+,.0f}, "
            f"avgR {row.get('avg_r', 0.0):+.3f}, win {row.get('win_rate', 0.0):.1%}, "
            f"excellent {row.get('excellent_trades', 0):.0f}/{row.get('trades', 0):.0f}."
        )

    context_line = (
        "- Asset context/news are currently stubbed in the executable code, so the implementation captures a narrower edge than the `swing_3.md` thesis."
        if flags.get("asset_context_is_stubbed_in_code", False)
        else "- Asset context is now executable via completed-bar proxy context; the news-risk input remains a stub, so that part of the `swing_3.md` thesis is still not captured."
    )
    lines = [
        "# TPC Round 6 Alpha Audit",
        "",
        f"Output directory: `{summary['output_dir']}`",
        f"Elapsed minutes: {summary['elapsed_minutes']:.1f}",
        "",
        "## Headline",
        f"Train: trades {train.get('total_trades', 0.0):.0f}, net {train.get('net_return_pct', 0.0):+.2f}%, avgR {train.get('avg_r', 0.0):+.3f}, win {train.get('win_rate', 0.0):.1%}, max DD {train.get('max_dd_pct', 0.0):.2f}%.",
        f"OOS: trades {oos.get('total_trades', 0.0):.0f}, net {oos.get('net_return_pct', 0.0):+.2f}%, avgR {oos.get('avg_r', 0.0):+.3f}, win {oos.get('win_rate', 0.0):.1%}, max DD {oos.get('max_dd_pct', 0.0):.2f}%.",
        "",
        "## Symbol Cohorts",
        "- " + cohort_line("Train QQQ", cohorts.get("train_QQQ", {})),
        "- " + cohort_line("OOS QQQ", cohorts.get("oos_QQQ", {})),
        "- " + cohort_line("Train GLD", cohorts.get("train_GLD", {})),
        "- " + cohort_line("OOS GLD", cohorts.get("oos_GLD", {})),
        "",
        "## Six-Month Context",
        f"OOS net-return rank versus rolling train six-month trade windows: {rolling.get('oos_net_return_pct_rank_vs_train_6m', 0.0):.1%}.",
        f"OOS trade-count rank versus rolling train six-month trade windows: {rolling.get('oos_trade_count_rank_vs_train_6m', 0.0):.1%}.",
        f"Rolling train six-month trade-count percentiles: {rolling.get('train_total_trades_percentiles', {})}.",
        "",
        "## Shadow Setup Supply",
        f"QQQ train entry-ready bar rate {shadow.get('train_QQQ', {}).get('entry_ready_rate', 0.0):.3%}; QQQ OOS {shadow.get('oos_QQQ', {}).get('entry_ready_rate', 0.0):.3%}.",
        f"GLD train entry-ready bar rate {shadow.get('train_GLD', {}).get('entry_ready_rate', 0.0):.3%}; GLD OOS {shadow.get('oos_GLD', {}).get('entry_ready_rate', 0.0):.3%}.",
        "",
        "## Interpretation Flags",
        f"- Train R is top-winner concentrated: {flags['train_r_is_top_winner_concentrated']} (top-5 R share {train.get('top5_winner_share', 0.0):.1%}).",
        f"- QQQ OOS sample too small for decisive rejection: {flags['qqq_oos_sample_is_too_small_for_decisive_rejection']} (n={cohorts.get('oos_QQQ', {}).get('trades', 0)}).",
        f"- Probability of zero QQQ excellent trades in OOS if train excellent rate persisted: {flags['prob_zero_qqq_excellent_given_train_rate']:.1%}.",
        context_line,
    ]
    return "\n".join(lines) + "\n"


def bars_to_frame(bars: Any) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(bars.times))
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return pd.DataFrame(
        {
            "open": np.asarray(bars.opens, dtype=float),
            "high": np.asarray(bars.highs, dtype=float),
            "low": np.asarray(bars.lows, dtype=float),
            "close": np.asarray(bars.closes, dtype=float),
            "volume": np.asarray(bars.volumes, dtype=float),
        },
        index=idx,
    )


def as_utc_ts(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def iso_or_blank(ts: pd.Timestamp | None) -> str:
    return "" if ts is None else ts.isoformat()


def pct_return(series: pd.Series) -> float:
    clean = series.dropna()
    if len(clean) < 2 or clean.iloc[0] == 0:
        return 0.0
    return float((clean.iloc[-1] / clean.iloc[0] - 1.0) * 100.0)


def max_drawdown_pct(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    dd = np.where(peak > 0, (peak - arr) / peak, 0.0)
    return float(np.max(dd) * 100.0)


def top_win_share(rs: np.ndarray, n: int) -> float:
    wins = np.asarray([x for x in rs if x > 0], dtype=float)
    if wins.size == 0:
        return 0.0
    return float(np.sum(np.sort(wins)[-n:]) / max(np.sum(wins), 1e-9))


def percentiles(values: list[float]) -> dict[str, float]:
    vals = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    if vals.size == 0:
        return {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    return {f"p{p}": float(np.percentile(vals, p)) for p in (10, 25, 50, 75, 90)}


def percentile_rank(values: list[float], value: float) -> float:
    vals = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals <= value))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(normalize_jsonable(payload), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(normalize_jsonable(rows))


def normalize_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [normalize_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return 0.0
    return value


if __name__ == "__main__":
    main()
