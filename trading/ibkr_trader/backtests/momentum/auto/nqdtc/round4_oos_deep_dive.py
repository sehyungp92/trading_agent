"""Round-4 NQDTC OOS deep dive and weakness-targeted repair probes."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate repository root.")


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.shared.validation.oos_validation import (  # noqa: E402
    BACKTEST_START,
    BACKTEST_START_DATE,
    OOS_CUTOFF,
    OOS_CUTOFF_DATE,
    WindowMetrics,
    _assess,
    _compute_oos_months,
    _get_entry_time,
    _get_r_multiple,
    _window_months,
    compute_window_metrics,
)
from backtests.swing.auto.incumbent_repair import (  # noqa: E402
    CandidateEvaluation,
    FoldMetrics,
    RepairCandidate,
    StrategyRun,
    acceptance_reasons,
    build_fold_metrics,
    score_candidate,
    serialize,
)
from backtests.swing.auto.oos_repair_diagnostics import evaluate_strategy  # noqa: E402


CURRENT_CONFIG = ROOT / "backtests/output/momentum/nqdtc/round_4/optimized_config.json"


def main() -> None:
    args = parse_args()
    logging.getLogger("strategies.momentum.nqdtc.box").setLevel(logging.WARNING)
    logging.getLogger("backtests.momentum.engine.nqdtc_engine").setLevel(logging.WARNING)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    current = read_json(Path(args.config_path))
    baseline = evaluate_strategy("nqdtc", current, args.data_end)
    full = run_full_replay(current, args.data_end)
    baseline_diagnostics = diagnose_replay(full, args.data_end)

    candidates = build_weakness_targeted_candidates(current)
    progress_path = output_dir / "targeted_candidate_progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    evaluations = evaluate_candidates(
        baseline=baseline,
        candidates=candidates,
        data_end=args.data_end,
        max_workers=args.max_workers,
        progress_path=progress_path,
    )

    ranked = sorted(evaluations, key=lambda ev: ranking_tuple(ev, baseline), reverse=True)
    oos_net = sorted(
        evaluations,
        key=lambda ev: (
            ev.run.oos_metrics.net_r,
            ev.run.oos_metrics.total_trades,
            ev.run.is_metrics.net_r,
        ),
        reverse=True,
    )
    frequency = sorted(
        evaluations,
        key=lambda ev: (
            ev.run.oos_metrics.total_trades,
            ev.run.oos_metrics.net_r,
            ev.run.is_metrics.net_r,
        ),
        reverse=True,
    )
    balanced = [ev for ev in ranked if is_balanced_uplift(ev, baseline)]
    positive_preserved = [ev for ev in ranked if is_positive_oos_preserved(ev, baseline)]

    summary = {
        "run_spec": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(Path(args.config_path).resolve()),
            "data_end": args.data_end,
            "oos_cutoff": OOS_CUTOFF_DATE.isoformat(),
            "max_workers": args.max_workers,
            "candidate_count": len(candidates),
            "elapsed_seconds": round(time.time() - started, 2),
            "selection_oos_note": (
                "The OOS window is used for diagnosis/selection in this run; "
                "validate any promotion on a fresh holdout/forward window."
            ),
        },
        "baseline": serialize(baseline),
        "baseline_diagnostics": baseline_diagnostics,
        "targeted_candidate_count": len(candidates),
        "balanced_uplifts": [serialize(ev) for ev in balanced[: args.top_n]],
        "positive_oos_preserved_is": [serialize(ev) for ev in positive_preserved[: args.top_n]],
        "ranked": [serialize(ev) for ev in ranked[: args.top_n]],
        "oos_net_leaders": [serialize(ev) for ev in oos_net[: args.top_n]],
        "frequency_leaders": [serialize(ev) for ev in frequency[: args.top_n]],
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.txt").write_text(format_report(summary), encoding="utf-8")
    write_json(output_dir / "baseline_trade_diagnostics.json", baseline_diagnostics)
    print(f"Output: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default=str(CURRENT_CONFIG))
    parser.add_argument("--data-end", default="2026-05-01")
    parser.add_argument(
        "--output-dir",
        default="backtests/output/momentum/nqdtc/round_4/oos_deep_dive_20260524",
    )
    parser.add_argument("--max-workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--top-n", type=int, default=40)
    return parser.parse_args()


def run_full_replay(mutations: dict[str, Any], data_end: str) -> dict[str, Any]:
    from backtests.momentum.auto.config_mutator import mutate_nqdtc_config
    from backtests.momentum.auto.nqdtc.worker import load_worker_data
    from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
    from backtests.momentum.data.replay_cache import replay_engine_kwargs
    from backtests.momentum.engine.nqdtc_engine import NQDTCEngine
    from backtests.swing.auto.incumbent_repair import finalize_config

    data_dir = ROOT / "backtests/momentum/data/raw"
    base_config = NQDTCBacktestConfig(
        initial_equity=10_000,
        data_dir=data_dir,
        fixed_qty=10,
        track_signals=True,
        track_shadows=True,
        scoring_mode=False,
        max_dd_abort=0.0,
    )
    config = finalize_config(mutate_nqdtc_config(base_config, mutations), data_end)
    bundle = load_worker_data("NQ", data_dir)
    engine = NQDTCEngine("MNQ", config)
    result = engine.run(**replay_engine_kwargs(bundle))
    shadow_results = []
    if engine.shadow_tracker is not None:
        for item in engine.shadow_tracker.results:
            shadow_results.append(shadow_to_dict(item))
    return {
        "trades": list(result.trades),
        "signals": list(result.signal_events),
        "shadow_results": shadow_results,
        "engine_counters": {
            "breakouts_evaluated": result.breakouts_evaluated,
            "breakouts_qualified": result.breakouts_qualified,
            "entries_placed": result.entries_placed,
            "entries_filled": result.entries_filled,
            "gates_blocked": result.gates_blocked,
        },
        "shadow_summary": result.shadow_summary,
    }


def diagnose_replay(full: dict[str, Any], data_end: str) -> dict[str, Any]:
    end = datetime.combine(date.fromisoformat(data_end) + timedelta(days=1), datetime.min.time())
    trades = [
        trade for trade in full["trades"]
        if BACKTEST_START <= _get_entry_time(trade) < end
    ]
    is_trades = [t for t in trades if _get_entry_time(t) < OOS_CUTOFF]
    oos_trades = [t for t in trades if _get_entry_time(t) >= OOS_CUTOFF]
    signals = [
        sig for sig in full["signals"]
        if BACKTEST_START <= normalize_dt(sig.timestamp) < end
    ]
    is_signals = [s for s in signals if normalize_dt(s.timestamp) < OOS_CUTOFF]
    oos_signals = [s for s in signals if normalize_dt(s.timestamp) >= OOS_CUTOFF]
    oos_shadow = [
        s for s in full["shadow_results"]
        if OOS_CUTOFF <= datetime.fromisoformat(s["time"]) < end
    ]
    return {
        "periods": {
            "is": metrics_for_trades(is_trades, _window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE)),
            "oos": metrics_for_trades(oos_trades, _compute_oos_months(data_end)),
        },
        "oos_trades": [trade_to_dict(t) for t in sorted(oos_trades, key=_get_entry_time)],
        "is_group_summaries": {
            "session_direction": group_trade_summary(is_trades, lambda t: f"{t.session}_{dir_label(t.direction)}"),
            "entry_subtype": group_trade_summary(is_trades, lambda t: str(t.entry_subtype)),
            "exit_reason": group_trade_summary(is_trades, lambda t: str(t.exit_reason)),
            "regime": group_trade_summary(is_trades, lambda t: str(t.composite_regime)),
        },
        "oos_group_summaries": {
            "session_direction": group_trade_summary(oos_trades, lambda t: f"{t.session}_{dir_label(t.direction)}"),
            "entry_subtype": group_trade_summary(oos_trades, lambda t: str(t.entry_subtype)),
            "exit_reason": group_trade_summary(oos_trades, lambda t: str(t.exit_reason)),
            "regime": group_trade_summary(oos_trades, lambda t: str(t.composite_regime)),
            "ny_hour": group_trade_summary(oos_trades, lambda t: f"{ny_hour(_get_entry_time(t)):02d}"),
        },
        "loss_concentration": loss_concentration(oos_trades),
        "signal_summary": {
            "is": signal_summary(is_signals),
            "oos": signal_summary(oos_signals),
        },
        "oos_shadow_by_filter": shadow_summary(oos_shadow),
        "engine_counters": full["engine_counters"],
        "shadow_summary_text": full["shadow_summary"],
    }


def metrics_for_trades(trades: list[Any], months: float) -> dict[str, Any]:
    return serialize(compute_window_metrics([float(_get_r_multiple(t)) for t in trades], months))


def trade_to_dict(trade: Any) -> dict[str, Any]:
    entry = _get_entry_time(trade)
    exit_time = normalize_dt(getattr(trade, "exit_time", None))
    return {
        "entry_time": entry.isoformat(),
        "exit_time": exit_time.isoformat() if exit_time else None,
        "ny_hour": ny_hour(entry),
        "direction": dir_label(getattr(trade, "direction", 0)),
        "session": getattr(trade, "session", ""),
        "entry_subtype": getattr(trade, "entry_subtype", ""),
        "composite_regime": getattr(trade, "composite_regime", ""),
        "chop_mode": getattr(trade, "chop_mode", ""),
        "exit_reason": getattr(trade, "exit_reason", ""),
        "r_multiple": round(float(getattr(trade, "r_multiple", 0.0)), 6),
        "mfe_r": round(float(getattr(trade, "mfe_r", 0.0)), 6),
        "mae_r": round(float(getattr(trade, "mae_r", 0.0)), 6),
        "score_at_entry": round(float(getattr(trade, "score_at_entry", 0.0)), 6),
        "displacement_at_entry": round(float(getattr(trade, "displacement_at_entry", 0.0)), 6),
        "rvol_at_entry": round(float(getattr(trade, "rvol_at_entry", 0.0)), 6),
        "quality_mult": round(float(getattr(trade, "quality_mult", 0.0)), 6),
        "box_width": round(float(getattr(trade, "box_width", 0.0)), 6),
        "bars_held_30m": int(getattr(trade, "bars_held_30m", 0) or 0),
        "tp1_hit": bool(getattr(trade, "tp1_hit", False)),
        "tp2_hit": bool(getattr(trade, "tp2_hit", False)),
    }


def group_trade_summary(trades: list[Any], key_fn) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for trade in trades:
        groups.setdefault(str(key_fn(trade)), []).append(float(_get_r_multiple(trade)))
    return {
        key: summarize_rs(rs)
        for key, rs in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    }


def summarize_rs(rs: list[float]) -> dict[str, Any]:
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss > 0 else ("inf" if gross_win > 0 else 0.0)
    return {
        "trades": len(rs),
        "wins": len(wins),
        "win_rate": len(wins) / len(rs) if rs else 0.0,
        "net_r": round(sum(rs), 6),
        "avg_r": round(sum(rs) / len(rs), 6) if rs else 0.0,
        "profit_factor": pf if isinstance(pf, str) else round(pf, 6),
        "min_r": round(min(rs), 6) if rs else 0.0,
        "max_r": round(max(rs), 6) if rs else 0.0,
    }


def loss_concentration(trades: list[Any]) -> dict[str, Any]:
    rs = [float(_get_r_multiple(t)) for t in trades]
    if not rs:
        return {}
    losses = sorted([r for r in rs if r < 0])
    winners = sorted([r for r in rs if r > 0], reverse=True)
    net = sum(rs)
    return {
        "trade_count": len(rs),
        "net_r": round(net, 6),
        "loss_count": len(losses),
        "winner_count": len(winners),
        "largest_loss_r": round(losses[0], 6) if losses else 0.0,
        "largest_win_r": round(winners[0], 6) if winners else 0.0,
        "net_ex_largest_loss_r": round(net - losses[0], 6) if losses else round(net, 6),
        "net_ex_two_largest_losses_r": round(net - sum(losses[:2]), 6) if len(losses) >= 2 else round(net, 6),
        "net_ex_largest_win_r": round(net - winners[0], 6) if winners else round(net, 6),
        "losses": [round(r, 6) for r in losses],
        "winners": [round(r, 6) for r in winners],
    }


def signal_summary(signals: list[Any]) -> dict[str, Any]:
    first_blocks: dict[str, int] = {}
    regimes: dict[str, int] = {}
    sessions: dict[str, int] = {}
    passed = 0
    for sig in signals:
        if getattr(sig, "passed_all", False):
            passed += 1
        reason = getattr(sig, "first_block_reason", "") or "passed_all"
        first_blocks[reason] = first_blocks.get(reason, 0) + 1
        regime = getattr(sig, "composite_regime", "")
        regimes[regime] = regimes.get(regime, 0) + 1
        session = f"{getattr(sig, 'session', '')}_{dir_label(getattr(sig, 'direction', 0))}"
        sessions[session] = sessions.get(session, 0) + 1
    return {
        "evaluated": len(signals),
        "passed_all": passed,
        "first_block_reason": dict(sorted(first_blocks.items(), key=lambda item: (-item[1], item[0]))),
        "composite_regime": dict(sorted(regimes.items(), key=lambda item: (-item[1], item[0]))),
        "session_direction": dict(sorted(sessions.items(), key=lambda item: (-item[1], item[0]))),
    }


def shadow_to_dict(item: Any) -> dict[str, Any]:
    cand = item.candidate
    return {
        "time": normalize_dt(cand.time).isoformat(),
        "filter_name": cand.filter_name,
        "session": cand.session,
        "direction": dir_label(cand.direction),
        "composite_regime": cand.composite_regime,
        "filled": bool(item.filled),
        "r_multiple": round(float(item.r_multiple), 6),
        "mfe_r": round(float(item.mfe_r), 6),
        "mae_r": round(float(item.mae_r), 6),
        "reached_tp1": bool(item.reached_tp1),
        "reached_tp2": bool(item.reached_tp2),
        "exit_reason": item.exit_reason,
    }


def shadow_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(item["filter_name"], []).append(item)
    out: dict[str, Any] = {}
    for name, rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        filled = [r for r in rows if r["filled"]]
        rs = [float(r["r_multiple"]) for r in filled]
        out[name] = {
            "rejections": len(rows),
            "filled": len(filled),
            **summarize_rs(rs),
            "tp1_rate": sum(1 for r in filled if r["reached_tp1"]) / len(filled) if filled else 0.0,
            "tp2_rate": sum(1 for r in filled if r["reached_tp2"]) / len(filled) if filled else 0.0,
        }
    return out


def build_weakness_targeted_candidates(current: dict[str, Any]) -> list[RepairCandidate]:
    candidates: list[RepairCandidate] = []

    def add(name: str, patch: dict[str, Any], source: str = "weakness_targeted") -> None:
        merged = dict(current)
        changed = False
        for key, value in patch.items():
            if value == "__DROP__":
                if key in merged:
                    merged.pop(key)
                    changed = True
            elif merged.get(key) != value:
                merged[key] = value
                changed = True
        if changed:
            candidates.append(
                RepairCandidate(
                    name=name,
                    stage="weakness_targeted",
                    mutations=merged,
                    intent="Weakness-targeted NQDTC round-4 OOS repair probe.",
                    source=source,
                )
            )

    c_offsets = [0.248, 0.252, 0.264, 0.276, 0.288, 0.300, 0.320]
    for offset in c_offsets:
        label = fmt_value(offset)
        add(f"c_offset_{label}", {"param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": offset}, "c_offset")
        for tp1 in [1.30, 1.35, 1.40, 1.50, 1.60]:
            add(
                f"c_offset_{label}_tp1_{fmt_value(tp1)}",
                {
                    "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": offset,
                    "param_overrides.TP1_R": tp1,
                },
                "c_offset_tp",
            )
        for tp2 in [2.025, 2.475, 2.70]:
            add(
                f"c_offset_{label}_tp2_{fmt_value(tp2)}",
                {
                    "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": offset,
                    "param_overrides.TP2_R": tp2,
                },
                "c_offset_tp",
            )
        for pct in [0.30, 0.36, 0.40, 0.50, 0.55]:
            add(
                f"c_offset_{label}_tp1pct_{fmt_value(pct)}",
                {
                    "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": offset,
                    "param_overrides.TP1_PARTIAL_PCT": pct,
                },
                "c_offset_tp",
            )
        for ttl in [10, 11, 13, 14]:
            add(
                f"c_offset_{label}_a_ttl_{ttl}",
                {
                    "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": offset,
                    "param_overrides.A_TTL_5M_BARS": ttl,
                },
                "c_offset_a_ttl",
            )

    for mult in [2.25, 2.50, 2.75, 3.00, 3.25, 3.50]:
        add(
            f"neutral_open_nrm_{fmt_value(mult)}",
            {
                "param_overrides.BLOCK_NEUTRAL_REGIME": False,
                "param_overrides.SCORE_NON_RANGE_MULT": mult,
            },
            "neutral_recovery",
        )
        for offset in [0.252, 0.264, 0.288, 0.300]:
            add(
                f"neutral_open_nrm_{fmt_value(mult)}_c_{fmt_value(offset)}",
                {
                    "param_overrides.BLOCK_NEUTRAL_REGIME": False,
                    "param_overrides.SCORE_NON_RANGE_MULT": mult,
                    "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": offset,
                },
                "neutral_recovery_c_offset",
            )
            for tp1 in [1.35, 1.60]:
                add(
                    f"neutral_open_nrm_{fmt_value(mult)}_c_{fmt_value(offset)}_tp1_{fmt_value(tp1)}",
                    {
                        "param_overrides.BLOCK_NEUTRAL_REGIME": False,
                        "param_overrides.SCORE_NON_RANGE_MULT": mult,
                        "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": offset,
                        "param_overrides.TP1_R": tp1,
                    },
                    "neutral_recovery_c_tp",
                )

    for mult in [2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00]:
        for cooldown in [30, 45, 60, 75]:
            add(
                f"aligned_open_nrm_{fmt_value(mult)}_cd{cooldown}_c252",
                {
                    "param_overrides.BLOCK_ALIGNED_REGIME": False,
                    "param_overrides.SCORE_NON_RANGE_MULT": mult,
                    "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": cooldown,
                    "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": 0.252,
                },
                "aligned_frequency_guarded",
            )

    for width in [75, 100, 125, 150, 175]:
        add(
            f"min_box_{width}_c252",
            {
                "param_overrides.MIN_BOX_WIDTH": width,
                "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": 0.252,
            },
            "bad_trade_filter",
        )
    for hour_flag in ["flags.block_04_et", "flags.block_06_et", "flags.block_12_et"]:
        add(
            f"open_{hour_flag.replace('flags.', '')}_c252",
            {hour_flag: False, "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": 0.252},
            "hour_probe_c_offset",
        )
    for gap in [35, 40, 45, 50, 55, 60]:
        add(
            f"cooldown_{gap}_c252",
            {
                "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": gap,
                "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": 0.252,
            },
            "cooldown_c_offset",
        )
    add(
        "disable_a_keep_c252",
        {
            "param_overrides.A_ENTRY_ENABLED": False,
            "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": 0.252,
        },
        "a_entry_check",
    )
    add(
        "a_box200_ttl13_c252",
        {
            "param_overrides.A_MAX_BOX_WIDTH": 200.0,
            "param_overrides.A_TTL_5M_BARS": 13,
            "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": 0.252,
        },
        "a_entry_check",
    )
    add(
        "a_score3_c252",
        {
            "param_overrides.A_MIN_SCORE": 3.0,
            "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": 0.252,
        },
        "a_entry_check",
    )

    return dedupe(candidates, current)


def evaluate_candidates(
    *,
    baseline: StrategyRun,
    candidates: list[RepairCandidate],
    data_end: str,
    max_workers: int,
    progress_path: Path,
) -> list[CandidateEvaluation]:
    evaluations: list[CandidateEvaluation] = []
    total = len(candidates)
    payload = serialize(baseline)
    if max_workers <= 1:
        for idx, candidate in enumerate(candidates, start=1):
            ev = evaluate_one(candidate, payload, data_end)
            evaluations.append(ev)
            append_progress(progress_path, idx, total, ev)
        return evaluations
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(evaluate_one, candidate, payload, data_end) for candidate in candidates]
        for idx, future in enumerate(as_completed(futures), start=1):
            ev = future.result()
            evaluations.append(ev)
            append_progress(progress_path, idx, total, ev)
            print(
                f"[deep-dive] {idx}/{total} {ev.candidate.name} "
                f"OOS={ev.run.oos_metrics.total_trades} {ev.run.oos_metrics.net_r:+.2f}R "
                f"IS={ev.run.is_metrics.total_trades} {ev.run.is_metrics.net_r:+.1f}R "
                f"passed={ev.passed}",
                flush=True,
            )
    return evaluations


def evaluate_one(candidate: RepairCandidate, baseline_payload: dict[str, Any], data_end: str) -> CandidateEvaluation:
    baseline = strategy_run_from_payload(baseline_payload)
    try:
        run = evaluate_strategy("nqdtc", candidate.mutations, data_end)
        ev = score_candidate(candidate, baseline, run)
        ev.deltas["balanced_rank"] = ranking_tuple(ev, baseline)[0]
        return ev
    except Exception as exc:
        run = StrategyRun(
            strategy="nqdtc",
            mutations=dict(candidate.mutations),
            is_metrics=WindowMetrics(),
            oos_metrics=WindowMetrics(),
            fold_metrics=[],
            assessment="ERROR",
            action="Error",
            error=str(exc),
        )
        return CandidateEvaluation(candidate=candidate, run=run, objective_delta=-999.0, passed=False, reasons=[str(exc)])


def strategy_run_from_payload(payload: dict[str, Any]) -> StrategyRun:
    def wm(item: dict[str, Any]) -> WindowMetrics:
        return WindowMetrics(**item)

    return StrategyRun(
        strategy=payload["strategy"],
        mutations=payload["mutations"],
        is_metrics=wm(payload["is_metrics"]),
        oos_metrics=wm(payload["oos_metrics"]),
        fold_metrics=[
            FoldMetrics(name=f["name"], start=f["start"], end=f["end"], metrics=wm(f["metrics"]))
            for f in payload["fold_metrics"]
        ],
        assessment=payload["assessment"],
        action=payload["action"],
        error=payload.get("error", ""),
    )


def append_progress(path: Path, completed: int, total: int, ev: CandidateEvaluation) -> None:
    payload = {
        "completed": completed,
        "total": total,
        "candidate": ev.candidate.name,
        "source": ev.candidate.source,
        "objective_delta": ev.objective_delta,
        "balanced_rank": ranking_tuple(ev, None)[0],
        "passed": ev.passed,
        "reasons": ev.reasons,
        "is_metrics": serialize(ev.run.is_metrics),
        "oos_metrics": serialize(ev.run.oos_metrics),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(serialize(payload), default=str) + "\n")


def ranking_tuple(ev: CandidateEvaluation, baseline: StrategyRun | None) -> tuple[float, float, float, float]:
    if baseline is None:
        return (ev.objective_delta, ev.run.oos_metrics.net_r, ev.run.oos_metrics.total_trades, ev.run.is_metrics.net_r)
    oos_net_delta = ev.run.oos_metrics.net_r - baseline.oos_metrics.net_r
    oos_trade_delta = ev.run.oos_metrics.total_trades - baseline.oos_metrics.total_trades
    is_net_ratio = safe_ratio(ev.run.is_metrics.net_r, baseline.is_metrics.net_r)
    preservation_penalty = max(0.0, 0.95 - is_net_ratio) * 4.0
    reason_penalty = 0.15 * len(ev.reasons)
    rank = (
        0.45 * oos_net_delta
        + 0.18 * oos_trade_delta
        + 0.22 * (ev.run.is_metrics.net_r - baseline.is_metrics.net_r) / max(abs(baseline.is_metrics.net_r), 5.0)
        + 0.15 * (ev.run.is_metrics.total_trades - baseline.is_metrics.total_trades) / max(baseline.is_metrics.total_trades, 10)
        - preservation_penalty
        - reason_penalty
    )
    return (rank, ev.run.oos_metrics.net_r, ev.run.oos_metrics.total_trades, ev.run.is_metrics.net_r)


def is_balanced_uplift(ev: CandidateEvaluation, baseline: StrategyRun) -> bool:
    return (
        ev.run.oos_metrics.net_r > baseline.oos_metrics.net_r
        and ev.run.oos_metrics.total_trades >= baseline.oos_metrics.total_trades
        and ev.run.is_metrics.net_r >= baseline.is_metrics.net_r * 0.95
        and ev.run.is_metrics.total_trades >= int(baseline.is_metrics.total_trades * 0.90)
        and ev.run.is_metrics.avg_r >= baseline.is_metrics.avg_r - 0.10
    )


def is_positive_oos_preserved(ev: CandidateEvaluation, baseline: StrategyRun) -> bool:
    return (
        ev.run.oos_metrics.net_r > 0
        and ev.run.is_metrics.net_r >= baseline.is_metrics.net_r * 0.90
        and ev.run.is_metrics.total_trades >= int(baseline.is_metrics.total_trades * 0.85)
        and ev.run.is_metrics.avg_r >= baseline.is_metrics.avg_r - 0.15
    )


def format_report(summary: dict[str, Any]) -> str:
    base = summary["baseline"]
    diag = summary["baseline_diagnostics"]
    lines = [
        "NQDTC Round 4 OOS Deep Dive",
        "=" * 100,
        summary["run_spec"]["selection_oos_note"],
        f"Data end: {summary['run_spec']['data_end']}; OOS starts {summary['run_spec']['oos_cutoff']}",
        "",
        (
            "Baseline IS: "
            f"trades={base['is_metrics']['total_trades']} "
            f"netR={base['is_metrics']['net_r']:.2f} "
            f"avgR={base['is_metrics']['avg_r']:.3f} "
            f"PF={fmt(base['is_metrics']['profit_factor'])}"
        ),
        (
            "Baseline OOS: "
            f"trades={base['oos_metrics']['total_trades']} "
            f"netR={base['oos_metrics']['net_r']:.2f} "
            f"avgR={base['oos_metrics']['avg_r']:.3f} "
            f"PF={fmt(base['oos_metrics']['profit_factor'])}"
        ),
        "",
        "OOS trades:",
    ]
    for row in diag["oos_trades"]:
        lines.append(
            f"  {row['entry_time']} {row['session']} {row['direction']} {row['entry_subtype']} "
            f"{row['composite_regime']} hour={row['ny_hour']} R={row['r_multiple']:+.3f} "
            f"MFE={row['mfe_r']:.2f} exit={row['exit_reason']}"
        )
    lc = diag["loss_concentration"]
    lines.extend([
        "",
        (
            "Loss concentration: "
            f"netR={lc.get('net_r', 0):+.2f}, losses={lc.get('losses', [])}, "
            f"net ex worst loss={lc.get('net_ex_largest_loss_r', 0):+.2f}, "
            f"net ex two worst={lc.get('net_ex_two_largest_losses_r', 0):+.2f}"
        ),
        "",
        "OOS group summaries:",
    ])
    for group, values in diag["oos_group_summaries"].items():
        lines.append(f"  {group}: {json.dumps(values, default=str)}")
    lines.extend([
        "",
        "OOS signal blocks:",
        f"  {json.dumps(diag['signal_summary']['oos'], default=str)}",
        "",
        "OOS shadow by first block:",
        f"  {json.dumps(diag['oos_shadow_by_filter'], default=str)}",
        "",
        f"Additional targeted candidates evaluated: {summary['targeted_candidate_count']}",
    ])
    for key, title in [
        ("balanced_uplifts", "Balanced uplifts"),
        ("positive_oos_preserved_is", "Positive OOS with IS preserved"),
        ("oos_net_leaders", "OOS-net leaders"),
        ("frequency_leaders", "Frequency leaders"),
    ]:
        lines.append("")
        lines.append(title + ":")
        items = summary.get(key, [])[:12]
        if not items:
            lines.append("  None")
            continue
        for item in items:
            c = item["candidate"]
            run = item["run"]
            oos = run["oos_metrics"]
            is_m = run["is_metrics"]
            lines.append(
                f"  {c['name']}: passed={item['passed']} obj={item['objective_delta']:+.2%}, "
                f"OOS trades={oos['total_trades']} netR={oos['net_r']:+.2f} avgR={oos['avg_r']:+.3f}, "
                f"IS trades={is_m['total_trades']} netR={is_m['net_r']:+.1f} avgR={is_m['avg_r']:+.3f}, "
                f"reasons={item.get('reasons', [])}"
            )
    return "\n".join(lines) + "\n"


def dedupe(candidates: list[RepairCandidate], current: dict[str, Any]) -> list[RepairCandidate]:
    current_sig = signature(current)
    seen: set[str] = set()
    out: list[RepairCandidate] = []
    for item in candidates:
        sig = signature(item.mutations)
        if sig == current_sig or sig in seen:
            continue
        seen.add(sig)
        out.append(item)
    return out


def normalize_dt(value: Any) -> datetime:
    if isinstance(value, np.datetime64):
        value = value.astype("datetime64[us]").astype(datetime)
    if value is None:
        return value
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def ny_hour(value: datetime) -> int:
    from zoneinfo import ZoneInfo

    dt = value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ZoneInfo("America/New_York")).hour


def dir_label(direction: Any) -> str:
    val = int(direction)
    if val > 0:
        return "LONG"
    if val < 0:
        return "SHORT"
    return "FLAT"


def fmt(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{float(value):.2f}"


def fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}".replace(".", "p")
    return str(value).replace(".", "p")


def safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def signature(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialize(value), indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
