"""Diagnostic matrix for NQDTC trade-count bottlenecks.

This runner intentionally evaluates broad diagnostic toggles before those ideas
are promoted into phased-auto candidates. It uses the same full replay engine
and can exclude the holdout window.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parents[4]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from backtests.momentum.auto.config_mutator import mutate_nqdtc_config
from backtests.momentum.auto.nqdtc.scoring import extract_nqdtc_metrics
from backtests.momentum.auto.nqdtc.worker import load_worker_data
from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
from backtests.momentum.data.replay_cache import replay_engine_kwargs
from backtests.momentum.engine.nqdtc_engine import NQDTCEngine

_WORKER_DATA = None
_WORKER_BASE: dict[str, Any] = {}


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_mutations(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if isinstance(payload.get("mutations"), dict):
            return dict(payload["mutations"])
        if isinstance(payload.get("cumulative_mutations"), dict):
            return dict(payload["cumulative_mutations"])
        return dict(payload)
    raise TypeError(f"Unexpected config payload in {path}")


def _init_worker(data_dir: str, equity: float, end_date: str | None, base_json: str) -> None:
    global _WORKER_DATA, _WORKER_BASE
    _WORKER_DATA = load_worker_data("NQ", Path(data_dir))
    _WORKER_BASE = {
        "data_dir": data_dir,
        "equity": equity,
        "end_date": end_date,
        "base": json.loads(base_json),
    }


def _evaluate(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    name, mutations = item
    base = dict(_WORKER_BASE["base"])
    base.update(mutations)
    config = mutate_nqdtc_config(
        NQDTCBacktestConfig(
            initial_equity=float(_WORKER_BASE["equity"]),
            data_dir=Path(_WORKER_BASE["data_dir"]),
            fixed_qty=10,
            end_date=_parse_dt(_WORKER_BASE["end_date"]),
            track_signals=True,
            track_shadows=False,
            scoring_mode=True,
            max_dd_abort=0.50,
        ),
        base,
    )
    started = time.time()
    result = NQDTCEngine("MNQ", config).run(**replay_engine_kwargs(_WORKER_DATA))
    metrics = extract_nqdtc_metrics(
        result.trades,
        list(result.equity_curve),
        list(result.timestamps),
        float(_WORKER_BASE["equity"]),
    )
    trades = result.trades
    signals = result.signal_events
    return {
        "name": name,
        "mutations": mutations,
        "elapsed_seconds": round(time.time() - started, 2),
        "metrics": asdict(metrics),
        "funnel": {
            "breakouts_evaluated": result.breakouts_evaluated,
            "breakouts_qualified": result.breakouts_qualified,
            "entries_placed": result.entries_placed,
            "entries_filled": result.entries_filled,
            "trades": len(trades),
            "qualified_to_entry_placed": _ratio(result.entries_placed, result.breakouts_qualified),
            "placed_to_filled": _ratio(result.entries_filled, result.entries_placed),
            "qualified_to_trade": _ratio(len(trades), result.breakouts_qualified),
        },
        "entry_subtypes": dict(Counter(getattr(t, "entry_subtype", "") for t in trades)),
        "exit_reasons": dict(Counter(getattr(t, "exit_reason", "") for t in trades)),
        "signal_blocks": dict(Counter(evt.first_block_reason or "passed" for evt in signals)),
        "signal_passed": sum(1 for evt in signals if evt.passed_all),
        "sessions": dict(Counter(getattr(t, "session", "") + "_" + ("LONG" if getattr(t, "direction", 0) == 1 else "SHORT") for t in trades)),
    }


def _ratio(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def _context_filter() -> dict[str, Any]:
    return {
        "param_overrides.WEAK_SCORE_BAND_FILTER_ENABLED": True,
        "param_overrides.WEAK_SCORE_BAND_MAX_BOX_WIDTH": 225.0,
        "param_overrides.WEAK_SCORE_BAND_MIN_RVOL": 1.75,
        "param_overrides.WIDE_BOX_SCORE_FILTER_ENABLED": True,
        "param_overrides.WIDE_BOX_MIN_WIDTH": 275.0,
        "param_overrides.WIDE_BOX_MIN_SCORE": 3.0,
        "param_overrides.WIDE_BOX_MIN_RVOL": 1.75,
    }


def _broad_variants() -> list[tuple[str, dict[str, Any]]]:
    variants: list[tuple[str, dict[str, Any]]] = [
        ("baseline", {}),
        ("cooldown_off", {"param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 0, "flags.loss_streak_cooldown": False}),
        ("time_blocks_off", {
            "flags.block_04_et": False,
            "flags.block_05_et": False,
            "flags.block_06_et": False,
            "flags.block_09_et": False,
            "flags.block_12_et": False,
        }),
        ("eth_shorts_on", {"flags.block_eth_shorts": False}),
        ("regime_blocks_off", {
            "param_overrides.BLOCK_NEUTRAL_REGIME": False,
            "param_overrides.BLOCK_ALIGNED_REGIME": False,
            "param_overrides.BLOCK_CAUTION_REGIME": False,
            "param_overrides.SCORE_NON_RANGE_MULT": 1.0,
        }),
        ("regime_reopen_score3", {
            "param_overrides.BLOCK_NEUTRAL_REGIME": False,
            "param_overrides.BLOCK_ALIGNED_REGIME": False,
            "param_overrides.BLOCK_CAUTION_REGIME": False,
            "param_overrides.SCORE_NON_RANGE_MULT": 3.0,
        }),
        ("score_off", {"flags.score_threshold": False}),
        ("quality_off", {"flags.breakout_quality_reject": False}),
        ("box_min_0", {"param_overrides.MIN_BOX_WIDTH": 0}),
        ("max_stop_off", {"flags.max_stop_width": False}),
        ("c_offset_0_35", {"param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": 0.35}),
        ("c_offset_0_50", {"param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": 0.50}),
        ("c_ttl_12", {"param_overrides.A_TTL_5M_BARS": 12}),
        ("c_ttl_18", {"param_overrides.A_TTL_5M_BARS": 18}),
        ("c_hold_tolerance_0_18", {"param_overrides.C_ENTRY_OFFSET_ATR": 0.18}),
        ("c_hold_tolerance_0_25", {"param_overrides.C_ENTRY_OFFSET_ATR": 0.25}),
        ("c_offset_0_35_ttl_12", {
            "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": 0.35,
            "param_overrides.A_TTL_5M_BARS": 12,
        }),
        ("c_offset_0_50_ttl_18", {
            "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": 0.50,
            "param_overrides.A_TTL_5M_BARS": 18,
        }),
        ("a_latch_only", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": False,
            "flags.entry_a_latch": True,
        }),
        ("a_both_ttl12", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": True,
            "flags.entry_a_latch": True,
            "param_overrides.A_TTL_5M_BARS": 12,
        }),
        ("b_range_neutral_p80", {
            "param_overrides.B_ALLOW_RANGE": True,
            "param_overrides.B_ALLOW_NEUTRAL": True,
            "param_overrides.B_MIN_DISP_Q": 0.80,
        }),
        ("b_range_depth_0_10", {
            "param_overrides.B_ALLOW_RANGE": True,
            "param_overrides.B_MIN_DISP_Q": 0.80,
            "param_overrides.B_SWEEP_DEPTH_ATR": 0.10,
        }),
        ("b_all_depth_0_05", {
            "param_overrides.B_ALLOW_RANGE": True,
            "param_overrides.B_ALLOW_NEUTRAL": True,
            "param_overrides.B_ALLOW_CAUTION": True,
            "param_overrides.B_MIN_DISP_Q": 0.70,
            "param_overrides.B_SWEEP_DEPTH_ATR": 0.05,
            "param_overrides.RESCUE_MAX_SLIP_ATR": 0.05,
        }),
        ("c_cont_on_mfe0", {
            "flags.entry_c_continuation": True,
            "param_overrides.C_CONT_ENTRY_ENABLED": True,
            "param_overrides.C_CONT_MFE_GATE_R": 0.0,
            "param_overrides.BLOCK_CONT_ALIGNED": False,
        }),
        ("all_entries_open", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": True,
            "flags.entry_a_latch": True,
            "flags.entry_c_continuation": True,
            "param_overrides.C_CONT_ENTRY_ENABLED": True,
            "param_overrides.C_CONT_MFE_GATE_R": 0.0,
            "param_overrides.BLOCK_CONT_ALIGNED": False,
            "param_overrides.B_ALLOW_RANGE": True,
            "param_overrides.B_ALLOW_NEUTRAL": True,
            "param_overrides.B_ALLOW_CAUTION": True,
            "param_overrides.B_MIN_DISP_Q": 0.70,
            "param_overrides.B_SWEEP_DEPTH_ATR": 0.05,
        }),
        ("all_frequency_open", {
            "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 0,
            "flags.loss_streak_cooldown": False,
            "flags.block_04_et": False,
            "flags.block_05_et": False,
            "flags.block_06_et": False,
            "flags.block_09_et": False,
            "flags.block_12_et": False,
            "flags.block_eth_shorts": False,
            "param_overrides.MIN_BOX_WIDTH": 0,
            "flags.max_stop_width": False,
        }),
        ("all_gates_open", {
            "flags.score_threshold": False,
            "flags.breakout_quality_reject": False,
            "param_overrides.MIN_BOX_WIDTH": 0,
            "flags.max_stop_width": False,
            "param_overrides.BLOCK_NEUTRAL_REGIME": False,
            "param_overrides.BLOCK_ALIGNED_REGIME": False,
            "param_overrides.BLOCK_CAUTION_REGIME": False,
            "param_overrides.SCORE_NON_RANGE_MULT": 1.0,
            "flags.block_eth_shorts": False,
        }),
        ("all_gates_entries_open", {
            "flags.score_threshold": False,
            "flags.breakout_quality_reject": False,
            "param_overrides.MIN_BOX_WIDTH": 0,
            "flags.max_stop_width": False,
            "param_overrides.BLOCK_NEUTRAL_REGIME": False,
            "param_overrides.BLOCK_ALIGNED_REGIME": False,
            "param_overrides.BLOCK_CAUTION_REGIME": False,
            "param_overrides.SCORE_NON_RANGE_MULT": 1.0,
            "flags.block_eth_shorts": False,
            "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 0,
            "flags.loss_streak_cooldown": False,
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": True,
            "flags.entry_a_latch": True,
            "flags.entry_c_continuation": True,
            "param_overrides.C_CONT_ENTRY_ENABLED": True,
            "param_overrides.C_CONT_MFE_GATE_R": 0.0,
            "param_overrides.BLOCK_CONT_ALIGNED": False,
            "param_overrides.B_ALLOW_RANGE": True,
            "param_overrides.B_ALLOW_NEUTRAL": True,
            "param_overrides.B_ALLOW_CAUTION": True,
            "param_overrides.B_MIN_DISP_Q": 0.70,
            "param_overrides.B_SWEEP_DEPTH_ATR": 0.05,
            "param_overrides.A_TTL_5M_BARS": 12,
            "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": 0.35,
        }),
    ]
    return variants


def _targeted_variants() -> list[tuple[str, dict[str, Any]]]:
    """Second-pass ideas promoted by the broad bottleneck scan."""
    context = _context_filter()
    variants: list[tuple[str, dict[str, Any]]] = [
        ("baseline", {}),
        ("box_min_0", {"param_overrides.MIN_BOX_WIDTH": 0}),
        ("box_min_50", {"param_overrides.MIN_BOX_WIDTH": 50}),
        ("box_min_75", {"param_overrides.MIN_BOX_WIDTH": 75}),
        ("box_min_100", {"param_overrides.MIN_BOX_WIDTH": 100}),
        ("box_min_125", {"param_overrides.MIN_BOX_WIDTH": 125}),
        ("box_min_0_context", {**context, "param_overrides.MIN_BOX_WIDTH": 0}),
        ("box_min_75_context", {**context, "param_overrides.MIN_BOX_WIDTH": 75}),
        ("allow_04", {"flags.block_04_et": False}),
        ("allow_05", {"flags.block_05_et": False}),
        ("allow_09", {"flags.block_09_et": False}),
        ("allow_12", {"flags.block_12_et": False}),
        ("allow_04_12", {"flags.block_04_et": False, "flags.block_12_et": False}),
        ("allow_05_12", {"flags.block_05_et": False, "flags.block_12_et": False}),
        ("time_blocks_off", {
            "flags.block_04_et": False,
            "flags.block_05_et": False,
            "flags.block_06_et": False,
            "flags.block_09_et": False,
            "flags.block_12_et": False,
        }),
        ("box_min_0_allow_12", {
            "param_overrides.MIN_BOX_WIDTH": 0,
            "flags.block_12_et": False,
        }),
        ("eth_shorts_on", {"flags.block_eth_shorts": False}),
        ("eth_shorts_half", {
            "flags.block_eth_shorts": False,
            "param_overrides.ETH_SHORT_SIZE_MULT": 0.50,
        }),
        ("box_min_0_eth_shorts", {
            "param_overrides.MIN_BOX_WIDTH": 0,
            "flags.block_eth_shorts": False,
        }),
        ("a_latch_stop_175", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": False,
            "flags.entry_a_latch": True,
            "param_overrides.MAX_STOP_WIDTH_PTS": 175.0,
        }),
        ("a_latch_ttl12_stop_175", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": False,
            "flags.entry_a_latch": True,
            "param_overrides.A_TTL_5M_BARS": 12,
            "param_overrides.MAX_STOP_WIDTH_PTS": 175.0,
        }),
        ("a_both_ttl6_stop_175", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": True,
            "flags.entry_a_latch": True,
            "param_overrides.A_TTL_5M_BARS": 6,
            "param_overrides.MAX_STOP_WIDTH_PTS": 175.0,
        }),
        ("a_both_ttl12_stop_175", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": True,
            "flags.entry_a_latch": True,
            "param_overrides.A_TTL_5M_BARS": 12,
            "param_overrides.MAX_STOP_WIDTH_PTS": 175.0,
        }),
        ("a_retest_only_ttl12", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": True,
            "flags.entry_a_latch": False,
            "param_overrides.A_TTL_5M_BARS": 12,
        }),
        ("a_retest_box_225", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": True,
            "flags.entry_a_latch": False,
            "param_overrides.A_TTL_5M_BARS": 12,
            "param_overrides.A_MAX_BOX_WIDTH": 225.0,
        }),
        ("a_retest_box_225_no_weak", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": True,
            "flags.entry_a_latch": False,
            "param_overrides.A_TTL_5M_BARS": 12,
            "param_overrides.A_MAX_BOX_WIDTH": 225.0,
            "param_overrides.A_BLOCK_WEAK_SCORE_BAND": True,
        }),
        ("a_retest_box_225_score3", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": True,
            "flags.entry_a_latch": False,
            "param_overrides.A_TTL_5M_BARS": 12,
            "param_overrides.A_MAX_BOX_WIDTH": 225.0,
            "param_overrides.A_MIN_SCORE": 3.0,
        }),
        ("a_retest_box_225_box_min_50", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": True,
            "flags.entry_a_latch": False,
            "param_overrides.A_TTL_5M_BARS": 12,
            "param_overrides.A_MAX_BOX_WIDTH": 225.0,
            "param_overrides.MIN_BOX_WIDTH": 50,
        }),
        ("box_min_50_allow_05", {
            "param_overrides.MIN_BOX_WIDTH": 50,
            "flags.block_05_et": False,
        }),
        ("a_retest_box_225_box_min_50_allow_05", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": True,
            "flags.entry_a_latch": False,
            "param_overrides.A_TTL_5M_BARS": 12,
            "param_overrides.A_MAX_BOX_WIDTH": 225.0,
            "param_overrides.MIN_BOX_WIDTH": 50,
            "flags.block_05_et": False,
        }),
        ("a_retest_box_225_box_min_75", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": True,
            "flags.entry_a_latch": False,
            "param_overrides.A_TTL_5M_BARS": 12,
            "param_overrides.A_MAX_BOX_WIDTH": 225.0,
            "param_overrides.MIN_BOX_WIDTH": 75,
        }),
        ("a_both_box_225_no_weak", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": True,
            "flags.entry_a_latch": True,
            "param_overrides.A_TTL_5M_BARS": 12,
            "param_overrides.A_MAX_BOX_WIDTH": 225.0,
            "param_overrides.A_BLOCK_WEAK_SCORE_BAND": True,
        }),
        ("a_both_ttl12_context", {
            **context,
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": True,
            "flags.entry_a_latch": True,
            "param_overrides.A_TTL_5M_BARS": 12,
        }),
        ("c_cont_mfe_0_50", {
            "flags.entry_c_continuation": True,
            "param_overrides.C_CONT_ENTRY_ENABLED": True,
            "param_overrides.C_CONT_MFE_GATE_R": 0.50,
            "param_overrides.BLOCK_CONT_ALIGNED": False,
        }),
        ("c_cont_mfe_1_00", {
            "flags.entry_c_continuation": True,
            "param_overrides.C_CONT_ENTRY_ENABLED": True,
            "param_overrides.C_CONT_MFE_GATE_R": 1.00,
            "param_overrides.BLOCK_CONT_ALIGNED": False,
        }),
        ("c_cont_context_mfe_0_50", {
            **context,
            "flags.entry_c_continuation": True,
            "param_overrides.C_CONT_ENTRY_ENABLED": True,
            "param_overrides.C_CONT_MFE_GATE_R": 0.50,
            "param_overrides.BLOCK_CONT_ALIGNED": False,
        }),
    ]
    return variants


def _a_context_variants() -> list[tuple[str, dict[str, Any]]]:
    selected = {
        "baseline",
        "box_min_50",
        "allow_05",
        "a_retest_only_ttl12",
        "a_retest_box_225",
        "a_retest_box_225_no_weak",
        "a_retest_box_225_score3",
        "a_retest_box_225_box_min_50",
        "box_min_50_allow_05",
        "a_retest_box_225_box_min_50_allow_05",
        "a_retest_box_225_box_min_75",
        "a_both_box_225_no_weak",
    }
    return [(name, muts) for name, muts in _targeted_variants() if name in selected]


def _variants(profile: str) -> list[tuple[str, dict[str, Any]]]:
    if profile == "broad":
        return _broad_variants()
    if profile == "targeted":
        return _targeted_variants()
    if profile == "a-context":
        return _a_context_variants()
    if profile == "all":
        seen: set[str] = set()
        merged: list[tuple[str, dict[str, Any]]] = []
        for name, mutations in [*_broad_variants(), *_targeted_variants()]:
            if name in seen:
                continue
            seen.add(name)
            merged.append((name, mutations))
        return merged
    raise ValueError(f"Unknown profile: {profile}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="backtests/output/momentum/nqdtc/round_3/optimized_config.json")
    parser.add_argument("--data-dir", default="backtests/momentum/data/raw")
    parser.add_argument("--output", default="backtests/output/momentum/nqdtc/round_3/trade_count_diagnostics.json")
    parser.add_argument("--end-date", default="2026-03-21")
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--profile", choices=["broad", "targeted", "a-context", "all"], default="broad")
    args = parser.parse_args()

    base = _load_mutations(Path(args.config))
    variants = _variants(args.profile)
    started = time.time()
    results: list[dict[str, Any]] = []
    print(f"[nqdtc-count] evaluating {len(variants)} variants with {args.max_workers} workers", flush=True)
    with ProcessPoolExecutor(
        max_workers=args.max_workers,
        initializer=_init_worker,
        initargs=(args.data_dir, args.equity, args.end_date, json.dumps(base, sort_keys=True)),
    ) as pool:
        futures = {pool.submit(_evaluate, item): item[0] for item in variants}
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            results.append(result)
            m = result["metrics"]
            f = result["funnel"]
            print(
                f"[nqdtc-count] {i}/{len(variants)} {result['name']} "
                f"trades={m['total_trades']} placed={f['entries_placed']} filled={f['entries_filled']} "
                f"PF={m['profit_factor']:.2f} net={m['net_return_pct']:.1f}% avgR={m['avg_r']:.3f}",
                flush=True,
            )

    by_trades = sorted(results, key=lambda row: (row["metrics"]["total_trades"], row["metrics"]["net_return_pct"]), reverse=True)
    by_net = sorted(results, key=lambda row: (row["metrics"]["net_return_pct"], row["metrics"]["total_trades"]), reverse=True)
    summary = {
        "run_spec": {
            "config": str(Path(args.config).resolve()),
            "data_dir": args.data_dir,
            "end_date": args.end_date,
            "equity": args.equity,
            "max_workers": args.max_workers,
            "profile": args.profile,
            "elapsed_seconds": round(time.time() - started, 2),
        },
        "results": sorted(results, key=lambda row: row["name"]),
        "top_by_trades": by_trades[:10],
        "top_by_net_return": by_net[:10],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[nqdtc-count] wrote {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
