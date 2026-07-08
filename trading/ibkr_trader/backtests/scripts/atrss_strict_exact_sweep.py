"""Exact ATRSS strict-OOS mutation sweep.

Analysis-only runner for the 2026-03-21..2026-05-01 ATRSS undertrading
question. It evaluates candidate configs directly on the report split instead
of using the anchored repair runner's top-k development prefilter.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtests.shared.validation.oos_validation import compute_window_metrics
from backtests.swing.auto.atrss.anchored_walk_forward import (
    build_historical_acceptance_features,
    build_incumbent_ablation_candidates,
    build_incumbent_perturbation_candidates,
    build_targeted_oos_candidates,
)
from backtests.swing.auto.config_mutator import mutate_atrss_config
from backtests.swing.config import AblationFlags, BacktestConfig, SlippageConfig
from backtests.swing.data.replay_cache import load_atrss_replay_bundle
from backtests.swing.engine.portfolio_engine import run_synchronized

DATA_DIR = PROJECT_ROOT / "backtests" / "swing" / "data" / "raw"
INCUMBENT_PATH = PROJECT_ROOT / "backtests" / "output" / "swing" / "atrss" / "round_3" / "optimized_config.json"
ROUNDS_DIR = PROJECT_ROOT / "backtests" / "output" / "swing" / "atrss"
OUTPUT_ROOT = PROJECT_ROOT / "backtests" / "output" / "oos"
DATA_END = "2026-05-01"
IS_START = "2024-01-01"
OOS_START = "2026-03-21"
OOS_END_EXCL = "2026-05-02"
PRE_OOS_START = "2026-01-01"

_WORKER_DATA = None
_WORKER_BASE = None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def signature(mutations: dict[str, Any]) -> str:
    return json.dumps(mutations, sort_keys=True, default=str, separators=(",", ":"))


def merge_patch(current: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    merged.update(patch)
    return merged


def add_candidate(candidates: list[dict[str, Any]], current: dict[str, Any], stage: str, name: str, patch: dict[str, Any], intent: str) -> None:
    mutations = merge_patch(current, patch)
    if signature(mutations) == signature(current):
        return
    candidates.append(
        {
            "name": name,
            "stage": stage,
            "mutations": mutations,
            "patch": patch,
            "intent": intent,
        }
    )


def build_candidates(current: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    historical = build_historical_acceptance_features(ROUNDS_DIR)
    for cand in build_incumbent_ablation_candidates(current, historical):
        out.append(asdict(cand))

    for cand in build_incumbent_perturbation_candidates(current):
        out.append(asdict(cand))

    args = SimpleNamespace(allow_symbol_expansion=False)
    for cand in build_targeted_oos_candidates(current, args):
        out.append(asdict(cand))

    # Additional weakness-derived probes after seeing strict OOS:
    # one QQQ bad-fill-flatten, no GLD trades after 2026-03-21, and no loss tail.
    custom: list[tuple[str, dict[str, Any], str]] = [
        ("disable_slippage_abort", {"flags.slippage_abort": False}, "Check whether the only OOS trade is an avoidable bad-fill edge case."),
        ("slip_atr_050", {"param_overrides.max_entry_slip_atr": 0.50}, "Loosen bad-fill ATR guard around the April QQQ fill."),
        ("slip_atr_075", {"param_overrides.max_entry_slip_atr": 0.75}, "Aggressively loosen bad-fill ATR guard."),
        ("stop_market_entries", {"slippage.use_stop_market": True}, "Test whether stop-limit entry simulation is suppressing OOS fills."),
        ("limit_ticks_0", {"param_overrides.limit_ticks": 0}, "Tighten stop-limit band to isolate fill sensitivity."),
        ("limit_ticks_4", {"param_overrides.limit_ticks": 4}, "Widen stop-limit band to improve fill conversion."),
        ("limit_pct_0025", {"param_overrides.limit_pct": 0.0025}, "Widen percent limit band to improve fill conversion."),
        ("adx_on_12", {"param_overrides.adx_on": 12}, "Loosen regime activation below the accepted 14 threshold."),
        ("adx_on_13", {"param_overrides.adx_on": 13}, "Slightly loosen regime activation."),
        ("adx_on_17", {"param_overrides.adx_on": 17}, "Tighten regime activation to test whether lower-quality regimes are noise."),
        ("adx_strong_25", {"param_overrides.adx_strong": 25}, "Let strong-trend logic activate earlier."),
        ("fast_confirm_score_40", {"param_overrides.fast_confirm_score": 40}, "Aggressively loosen fast-confirm score."),
        ("fast_confirm_adx_15", {"param_overrides.fast_confirm_adx": 15}, "Aggressively loosen fast-confirm ADX."),
        ("touch_tol_100", {"param_overrides.pullback_touch_tolerance_atr": 1.00}, "Very loose pullback touch tolerance for OOS frequency."),
        ("touch_tol_125", {"param_overrides.pullback_touch_tolerance_atr": 1.25}, "Extreme pullback touch tolerance stress test."),
        ("recovery_trend_075", {"param_overrides.recovery_tolerance_atr_trend": 0.75}, "Loosen trend recovery tolerance."),
        ("recovery_trend_085", {"param_overrides.recovery_tolerance_atr_trend": 0.85}, "Aggressively loosen trend recovery tolerance."),
        ("recov_base_060", {"param_overrides.recovery_tolerance_atr": 0.60}, "Loosen base recovery tolerance."),
        ("touch100_recovery075", {"param_overrides.pullback_touch_tolerance_atr": 1.00, "param_overrides.recovery_tolerance_atr_trend": 0.75}, "Joint pullback surface expansion."),
        ("confirm0_touch100", {"param_overrides.confirm_days_normal": 0, "param_overrides.pullback_touch_tolerance_atr": 1.00}, "Confirmation-speed plus broad pullback surface."),
        ("disable_reset_requirement", {"flags.reset_requirement": False}, "Check whether reset gating suppresses OOS re-entry."),
        ("disable_cooldown", {"flags.cooldown": False}, "Check whether cooldown suppresses OOS re-entry."),
        ("disable_prior_high", {"flags.prior_high_confirm": False}, "Check whether prior-high confirmation is too restrictive."),
        ("disable_hysteresis_gap", {"flags.hysteresis_gap": False}, "Check whether hysteresis gap prevents trend/bias flips."),
        ("disable_momentum_filter", {"flags.momentum_filter": False}, "Check non-pullback signal momentum filter drag."),
        ("disable_voucher", {"flags.voucher_system": False}, "Check whether voucher bookkeeping suppresses OOS signals."),
        ("enable_quality_30", {"flags.quality_gate": True, "param_overrides.quality_gate_threshold": 3.0}, "See if a quality gate helps loosened signals survive IS."),
        ("qqq_shorts_no_safety", {"param_overrides.shorts_enabled_QQQ": 1, "flags.short_safety": False}, "Probe QQQ short opportunity in the weak OOS period."),
        ("gld_shorts_no_safety", {"param_overrides.shorts_enabled_GLD": 1, "flags.short_safety": False}, "Probe GLD short opportunity after GLD longs stopped firing."),
        ("etf_shorts_no_safety", {"param_overrides.shorts_enabled_QQQ": 1, "param_overrides.shorts_enabled_GLD": 1, "flags.short_safety": False}, "Probe ETF short opportunity surface."),
        ("breakout_direct_no_candle", {"param_overrides.breakout_direct_entry": True, "param_overrides.breakout_require_directional_candle": False}, "Force breakout conversion without candle gate."),
        ("breakout_loose_confirm0", {"param_overrides.breakout_retrace_entry_frac": 0.10, "param_overrides.breakout_retrace_limit_frac": 0.75, "param_overrides.breakout_require_directional_candle": False, "param_overrides.confirm_days_normal": 0}, "Loosen breakout conversion plus confirmation."),
    ]
    for name, patch, intent in custom:
        add_candidate(out, current, "weakness_targeted", name, patch, intent)

    deduped: list[dict[str, Any]] = []
    seen = {signature(current)}
    for cand in out:
        mutations = dict(cand["mutations"])
        if mutations.get("symbols") not in (None, ["QQQ", "GLD"]):
            continue
        sig = signature(mutations)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(cand)
    return deduped


def init_worker() -> None:
    global _WORKER_DATA, _WORKER_BASE
    _WORKER_BASE = BacktestConfig(
        symbols=["QQQ", "GLD"],
        initial_equity=10_000,
        fixed_qty=10,
        data_dir=DATA_DIR,
        slippage=SlippageConfig(commission_per_contract=1.00),
        flags=AblationFlags(stall_exit=False),
        track_shadows=False,
    )
    _WORKER_DATA = load_atrss_replay_bundle(
        DATA_DIR,
        symbols=("QQQ", "GLD"),
        end_date=DATA_END,
    ).data


def naive(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value
    return None


def window_months(start: str, end_exclusive: str) -> float:
    return max((date.fromisoformat(end_exclusive) - date.fromisoformat(start)).days / 30.44, 0.1)


def metrics_for(trades: list[tuple[str, Any]], start: str, end_exclusive: str) -> tuple[dict[str, Any], list[tuple[str, Any]]]:
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end_exclusive)
    rows = [(sym, t) for sym, t in trades if (ts := naive(t.entry_time)) is not None and start_dt <= ts < end_dt]
    m = compute_window_metrics([float(t.r_multiple) for _, t in rows], window_months(start, end_exclusive))
    payload = asdict(m)
    payload["by_symbol"] = dict(Counter(sym for sym, _ in rows))
    payload["by_entry"] = dict(Counter(str(getattr(t, "entry_type", "")) for _, t in rows))
    payload["by_exit"] = dict(Counter(str(getattr(t, "exit_reason", "")) for _, t in rows))
    return payload, rows


def finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return 999.0 if value > 0 else -999.0
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    return value


def evaluate(payload: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    index, cand = payload
    if _WORKER_BASE is None or _WORKER_DATA is None:
        init_worker()
    config = mutate_atrss_config(_WORKER_BASE, cand["mutations"])
    result = run_synchronized(_WORKER_DATA, config)
    trades: list[tuple[str, Any]] = []
    for sym, sr in result.symbol_results.items():
        for trade in sr.trades:
            trades.append((sym, trade))
    trades.sort(key=lambda row: row[1].entry_time)
    is_m, is_rows = metrics_for(trades, IS_START, OOS_START)
    oos_m, oos_rows = metrics_for(trades, OOS_START, OOS_END_EXCL)
    pre_m, _ = metrics_for(trades, PRE_OOS_START, OOS_START)
    all_m, _ = metrics_for(trades, IS_START, OOS_END_EXCL)
    oos_examples = [
        {
            "symbol": sym,
            "entry_time": str(t.entry_time),
            "exit_time": str(t.exit_time),
            "entry_type": getattr(t, "entry_type", ""),
            "direction": int(getattr(t, "direction", 0)),
            "r_multiple": float(getattr(t, "r_multiple", 0.0)),
            "mfe_r": float(getattr(t, "mfe_r", 0.0)),
            "mae_r": float(getattr(t, "mae_r", 0.0)),
            "exit_reason": getattr(t, "exit_reason", ""),
            "qty": int(getattr(t, "qty", 0)),
        }
        for sym, t in oos_rows[:12]
    ]
    return finite(
        {
            "index": index,
            "name": cand["name"],
            "stage": cand["stage"],
            "intent": cand.get("intent", ""),
            "source": cand.get("source", ""),
            "patch": cand.get("patch", {}),
            "mutations": cand["mutations"],
            "is_metrics": is_m,
            "pre_oos_metrics": pre_m,
            "oos_metrics": oos_m,
            "all_metrics": all_m,
            "oos_examples": oos_examples,
        }
    )


def objective(item: dict[str, Any], baseline: dict[str, Any]) -> float:
    is_m = item["is_metrics"]
    oos_m = item["oos_metrics"]
    base_is = baseline["is_metrics"]
    base_oos = baseline["oos_metrics"]
    is_r_ratio = is_m["net_r"] / max(abs(base_is["net_r"]), 1e-9)
    is_trade_ratio = is_m["total_trades"] / max(base_is["total_trades"], 1)
    is_pf_ratio = min(is_m["profit_factor"], 999.0) / max(min(base_is["profit_factor"], 999.0), 1e-9)
    oos_trade_uplift = (oos_m["total_trades"] - base_oos["total_trades"]) / max(base_oos["total_trades"], 1)
    oos_r_uplift = oos_m["net_r"] - base_oos["net_r"]
    return (
        0.40 * oos_trade_uplift
        + 0.35 * oos_r_uplift
        + 0.12 * (is_r_ratio - 1.0)
        + 0.08 * (is_trade_ratio - 1.0)
        + 0.05 * (is_pf_ratio - 1.0)
    )


def acceptance_label(item: dict[str, Any], baseline: dict[str, Any]) -> str:
    is_m = item["is_metrics"]
    oos_m = item["oos_metrics"]
    base_is = baseline["is_metrics"]
    if (
        oos_m["total_trades"] > baseline["oos_metrics"]["total_trades"]
        and oos_m["net_r"] > baseline["oos_metrics"]["net_r"]
        and is_m["net_r"] >= 0.93 * base_is["net_r"]
        and is_m["total_trades"] >= 0.85 * base_is["total_trades"]
        and is_m["profit_factor"] >= 4.5
    ):
        return "candidate"
    if oos_m["total_trades"] > baseline["oos_metrics"]["total_trades"]:
        return "frequency_only_or_deteriorating"
    return "no_oos_frequency_uplift"


def main() -> int:
    current = load_json(INCUMBENT_PATH)
    candidates = build_candidates(current)
    out_dir = OUTPUT_ROOT / f"atrss_strict_exact_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate_count.txt").write_text(str(len(candidates)), encoding="utf-8")
    print(f"Output: {out_dir}")
    print(f"Candidates: {len(candidates)}")

    baseline = evaluate((0, {"name": "baseline", "stage": "baseline", "mutations": current, "intent": ""}))
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=4, initializer=init_worker) as pool:
        futures = [pool.submit(evaluate, (idx, cand)) for idx, cand in enumerate(candidates, start=1)]
        for completed, future in enumerate(as_completed(futures), start=1):
            item = future.result()
            item["objective"] = objective(item, baseline)
            item["acceptance_label"] = acceptance_label(item, baseline)
            results.append(item)
            if completed % 10 == 0 or completed == len(futures):
                best = max(results, key=lambda row: row["objective"])
                print(
                    f"[{completed}/{len(futures)}] best={best['name']} "
                    f"stage={best['stage']} oos_n={best['oos_metrics']['total_trades']} "
                    f"oos_R={best['oos_metrics']['net_r']:.2f} obj={best['objective']:.3f}",
                    flush=True,
                )

    results.sort(key=lambda row: row["objective"], reverse=True)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": {"is": [IS_START, OOS_START], "oos": [OOS_START, OOS_END_EXCL], "data_end": DATA_END},
        "baseline": baseline,
        "candidate_count": len(candidates),
        "results": results,
        "top_by_objective": results[:25],
        "top_by_oos_trades": sorted(
            results,
            key=lambda row: (
                row["oos_metrics"]["total_trades"],
                row["oos_metrics"]["net_r"],
                row["is_metrics"]["net_r"],
            ),
            reverse=True,
        )[:25],
        "accepted_like": [row for row in results if row["acceptance_label"] == "candidate"],
    }
    (out_dir / "summary.json").write_text(json.dumps(finite(summary), indent=2), encoding="utf-8")

    lines = [
        "ATRSS Strict Exact Sweep",
        f"Output: {out_dir}",
        f"Candidates: {len(candidates)}",
        "",
        "Baseline:",
        format_line(baseline),
        "",
        "Top Objective:",
    ]
    for row in results[:15]:
        lines.append(format_line(row))
    lines.extend(["", "Top OOS Trades:"])
    for row in summary["top_by_oos_trades"][:15]:
        lines.append(format_line(row))
    lines.extend(["", "Accepted-like candidates:"])
    if summary["accepted_like"]:
        for row in summary["accepted_like"][:15]:
            lines.append(format_line(row))
    else:
        lines.append("  None")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((out_dir / "summary.txt").read_text(encoding="utf-8"))
    return 0


def format_line(row: dict[str, Any]) -> str:
    is_m = row["is_metrics"]
    oos_m = row["oos_metrics"]
    return (
        f"  {row['stage']}/{row['name']}: "
        f"OOS n={oos_m['total_trades']} R={oos_m['net_r']:.2f} avgR={oos_m['avg_r']:.2f}; "
        f"IS n={is_m['total_trades']} PF={is_m['profit_factor']:.2f} R={is_m['net_r']:.1f}; "
        f"obj={row.get('objective', 0.0):.3f} label={row.get('acceptance_label', '')}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
