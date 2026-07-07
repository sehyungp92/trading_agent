"""Round-5 TPC OOS ablation, perturbation, and repair runner.

This script is intentionally diagnostic. It treats the six-month OOS window as
selection OOS, not as a fresh untouched holdout.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.swing.auto.tpc.plugin import _extract_tpc_metrics
from backtests.swing.config_tpc import TPCBacktestConfig
from backtests.swing.data.replay_cache import load_tpc_replay_bundle
from backtests.swing.engine.tpc_engine import run_tpc_independent
from strategies.swing.tpc.config import SYMBOL_CONFIGS

MISSING = "__TPC_REPAIR_MISSING__"

DEFAULT_OUTPUT_ROOT = ROOT / "backtests" / "output" / "swing" / "tpc" / "round_5"
DEFAULT_CONFIG_PATH = DEFAULT_OUTPUT_ROOT / "optimized_config.json"
DEFAULT_STAGE1_PATH = DEFAULT_OUTPUT_ROOT / "oos_ablation_perturbation.jsonl"
DEFAULT_TRAIN_END = "2025-11-01"
DATA_DIR = ROOT / "backtests" / "swing" / "data" / "raw"
INITIAL_EQUITY = 100_000.0

_TRAIN_DATA: dict[str, dict[str, Any]] | None = None
_FULL_DATA: dict[str, dict[str, Any]] | None = None
_OOS_WARMUP_15M: int = 1
_WORKER_DATA_DIR: Path | None = None
_WORKER_TRAIN_END: str = DEFAULT_TRAIN_END
_TRAIN_INDICATOR_CACHE: dict[Any, Any] = {}
_OOS_INDICATOR_CACHE: dict[Any, Any] = {}


@dataclass(frozen=True)
class Candidate:
    name: str
    stage: str
    mutations: dict[str, Any]
    source: str = ""
    intent: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--stage1-oos", type=Path, default=DEFAULT_STAGE1_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    parser.add_argument("--max-workers", type=int, default=max(1, min(6, (mp.cpu_count() or 2) - 1)))
    parser.add_argument("--shortlist", type=int, default=36)
    parser.add_argument("--targeted-limit", type=int, default=160)
    parser.add_argument("--skip-targeted-oos", action="store_true")
    args = parser.parse_args()

    started = time.time()
    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / f"oos_repair_extensive_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    incumbent = read_json(args.config)
    history = build_mutation_history(DEFAULT_OUTPUT_ROOT.parent)
    stage1_rows = load_stage1_rows(args.stage1_oos)

    print(f"[tpc] loaded round-5 config with {len(incumbent)} active overrides", flush=True)
    print(f"[tpc] loaded {len(stage1_rows)} prior OOS ablation/perturbation rows", flush=True)

    baseline_oos = stage1_rows[0]["oos"] if stage1_rows else {}
    weakness = diagnose_oos_weakness(baseline_oos)
    targeted_candidates = build_targeted_candidates(incumbent, stage1_rows, history)[: args.targeted_limit]
    print(f"[tpc] built {len(targeted_candidates)} targeted second-stage candidates", flush=True)

    targeted_oos_rows: list[dict[str, Any]] = []
    if not args.skip_targeted_oos and targeted_candidates:
        targeted_oos_rows = evaluate_oos_candidates(
            targeted_candidates,
            baseline_oos=baseline_oos,
            data_dir=DATA_DIR,
            train_end=args.train_end,
            max_workers=args.max_workers,
            output_path=output_dir / "targeted_oos_progress.jsonl",
        )
    elif args.skip_targeted_oos:
        print("[tpc] skipping targeted OOS evaluation by request", flush=True)

    shortlist_candidates = select_shortlist(
        incumbent,
        stage1_rows=stage1_rows,
        targeted_oos_rows=targeted_oos_rows,
        targeted_candidates=targeted_candidates,
        limit=args.shortlist,
    )
    print(f"[tpc] validating {len(shortlist_candidates)} shortlisted candidates on train+OOS", flush=True)
    validation_rows = evaluate_train_oos_candidates(
        shortlist_candidates,
        data_dir=DATA_DIR,
        train_end=args.train_end,
        max_workers=args.max_workers,
        output_path=output_dir / "train_oos_validation_progress.jsonl",
    )

    baseline_validation = first_by_name(validation_rows, "BASE_R5")
    if baseline_validation is None:
        baseline_validation = evaluate_train_oos_candidates(
            [Candidate("BASE_R5", "baseline", dict(incumbent), intent="Round-5 incumbent")],
            data_dir=DATA_DIR,
            train_end=args.train_end,
            max_workers=1,
            output_path=output_dir / "baseline_validation.jsonl",
        )[0]
        validation_rows.append(baseline_validation)

    scored = [score_validation(row, baseline_validation) for row in validation_rows]
    scored_sorted = sorted(scored, key=lambda row: row["objective"], reverse=True)
    best = next(
        (row for row in scored_sorted if row["name"] != "BASE_R5" and row.get("passed_repair_gate")),
        next((row for row in scored_sorted if row["name"] != "BASE_R5"), scored_sorted[0] if scored_sorted else None),
    )
    recommended = dict(best["mutations"]) if best else dict(incumbent)

    write_json(output_dir / "run_spec.json", {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_end": args.train_end,
        "data_end": infer_data_end(DATA_DIR),
        "config": str(args.config.resolve()),
        "stage1_oos": str(args.stage1_oos.resolve()),
        "max_workers": args.max_workers,
        "shortlist": args.shortlist,
        "targeted_limit": args.targeted_limit,
        "selection_oos_note": "The OOS window was used for diagnosis/selection; validate on fresh data before promotion.",
    })
    write_json(output_dir / "mutation_history.json", history)
    write_json(output_dir / "oos_weakness.json", weakness)
    write_json(output_dir / "targeted_oos_results.json", targeted_oos_rows)
    write_json(output_dir / "train_oos_validation.json", scored_sorted)
    write_json(output_dir / "recommended_config.json", recommended)
    write_csv(output_dir / "stage1_oos_summary.csv", summarize_stage1_rows(stage1_rows))
    write_csv(output_dir / "train_oos_validation.csv", flatten_validation_rows(scored_sorted))

    report = format_report(
        baseline=baseline_validation,
        best=best,
        scored=scored_sorted,
        stage1_rows=stage1_rows,
        targeted_oos_rows=targeted_oos_rows,
        weakness=weakness,
        elapsed_seconds=time.time() - started,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"[tpc] output: {output_dir.resolve()}", flush=True)


def build_mutation_history(round_root: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for round_num in range(1, 6):
        state_path = round_root / f"round_{round_num}" / "phase_state.json"
        if not state_path.exists():
            continue
        state = read_json(state_path)
        for phase_key, result in sorted(state.get("phase_results", {}).items(), key=lambda item: int(item[0])):
            new_mutations = dict(result.get("new_mutations") or {})
            if not new_mutations:
                continue
            previous_values = {key: current.get(key, MISSING) for key in new_mutations}
            current.update(new_mutations)
            events.append({
                "round": round_num,
                "phase": int(phase_key),
                "kept_features": list(result.get("kept_features") or []),
                "name": f"round{round_num}_phase{phase_key}::" + "+".join(result.get("kept_features") or ["phase_mutation"]),
                "mutations": normalize_jsonable(new_mutations),
                "previous_values": normalize_jsonable(previous_values),
            })
    return events


def load_stage1_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    raw = path.read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8")
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def diagnose_oos_weakness(oos: dict[str, Any]) -> dict[str, Any]:
    cohorts = dict(oos.get("cohorts") or {})
    long = cohorts.get("LONG", {})
    short = cohorts.get("SHORT", {})
    qqq = cohorts.get("QQQ", {})
    gld = cohorts.get("GLD", {})
    return {
        "headline": {
            "trades": oos.get("total_trades", 0.0),
            "net_return_pct": oos.get("net_return_pct", 0.0),
            "avg_r": oos.get("avg_r", 0.0),
            "win_rate": oos.get("win_rate", 0.0),
            "low_mfe_loss_rate": oos.get("low_mfe_loss_rate", 0.0),
            "max_dd_pct": oos.get("max_dd_pct", 0.0),
        },
        "cohort_findings": {
            "long_pnl": long.get("pnl", 0.0),
            "long_avg_r": long.get("avg_r", 0.0),
            "long_win_rate": long.get("win_rate", 0.0),
            "short_pnl": short.get("pnl", 0.0),
            "short_avg_r": short.get("avg_r", 0.0),
            "qqq_pnl": qqq.get("pnl", 0.0),
            "qqq_avg_r": qqq.get("avg_r", 0.0),
            "gld_pnl": gld.get("pnl", 0.0),
            "gld_avg_r": gld.get("avg_r", 0.0),
        },
        "interpretation": [
            "OOS loss is broad across the long book, not a single-tail-event problem.",
            "QQQ OOS sample has zero winning trades; GLD contributes more trades and more dollars lost.",
            "Shorts are comparatively resilient; blanket long removal is diagnostic, not a deployable repair because it crushes in-sample supply.",
            "The highest-value repair family should filter weak 4h trend quality before entry, then check whether exits/sizing merely mitigate dollars.",
        ],
    }


def build_targeted_candidates(
    incumbent: dict[str, Any],
    stage1_rows: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> list[Candidate]:
    candidates: list[Candidate] = []
    add = candidates.append

    def merged(name: str, stage: str, extra: dict[str, Any], *, source: str = "", intent: str = "") -> None:
        muts = dict(incumbent)
        muts.update(extra)
        add(Candidate(name=name, stage=stage, mutations=muts, source=source, intent=intent))

    add(Candidate("BASE_R5", "baseline", dict(incumbent), intent="Round-5 incumbent"))

    for key in sorted(incumbent):
        muts = dict(incumbent)
        muts.pop(key, None)
        add(Candidate(f"active_remove::{key}", "active_key_ablation", muts, source=key, intent="Remove one active final override."))

    for event in history:
        muts = dict(incumbent)
        touched = False
        for key, previous in event["previous_values"].items():
            if previous == MISSING:
                if key in muts:
                    muts.pop(key, None)
                    touched = True
            else:
                if muts.get(key) != previous:
                    muts[key] = previous
                    touched = True
        if touched:
            add(Candidate(f"rollback::{event['name']}", "historical_rollback", muts, source=event["name"], intent="Roll back one accepted historical mutation group."))

    # Trend-quality repair grid. These are deliberately mostly one-to-three-key
    # changes so we can see which filter is doing the work.
    for slope in (0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
        merged(f"trend_ma100_{slope:.2f}", "targeted_trend", {"all.min_ma100_slope_atr_4h": slope}, intent="Require persistent 4h MA100 slope.")
        merged(
            f"trend_ma100_{slope:.2f}_di",
            "targeted_trend",
            {"all.min_ma100_slope_atr_4h": slope, "all.require_di_alignment": True},
            intent="Require persistent slope plus DI agreement.",
        )
        for adx in (12.0, 15.0, 18.0, 20.0):
            merged(
                f"trend_ma100_{slope:.2f}_di_adx{int(adx)}",
                "targeted_trend",
                {"all.min_ma100_slope_atr_4h": slope, "all.require_di_alignment": True, "all.min_adx_4h": adx},
                intent="Require slope, DI agreement, and minimum ADX.",
            )
    for symbol in ("QQQ", "GLD"):
        for slope in (0.04, 0.06, 0.08):
            merged(f"{symbol.lower()}_ma100_{slope:.2f}", "targeted_symbol_trend", {f"{symbol}.min_ma100_slope_atr_4h": slope})
            merged(
                f"{symbol.lower()}_ma100_{slope:.2f}_di",
                "targeted_symbol_trend",
                {f"{symbol}.min_ma100_slope_atr_4h": slope, f"{symbol}.require_di_alignment": True},
            )
        merged(f"{symbol.lower()}_longs_off_probe", "diagnostic_direction", {f"{symbol}.longs_enabled": False})

    # Session, value, and confirmation repair probes.
    merged("session_no_11am_hour", "targeted_session", {"all.avoid_windows_et": ((11, 0, 12, 0),)})
    merged("session_no_1030_12", "targeted_session", {"all.avoid_windows_et": ((10, 30, 12, 0),)})
    merged("gld_no_8am_window", "targeted_session", {"GLD.primary_windows_et": ((9, 30, 11, 30), (13, 0, 16, 0))})
    merged("all_value_hits2", "targeted_signal_quality", {"all.type_a_value_hits_min": 2})
    merged("all_value_hits3", "targeted_signal_quality", {"all.type_a_value_hits_min": 3})
    merged("all_confirm2", "targeted_signal_quality", {"all.confirmation_required": 2})
    merged("all_require_structure", "targeted_signal_quality", {"all.require_structure_confirmation": True})
    merged("all_require_vwap", "targeted_signal_quality", {"all.require_vwap_confirmation": True})
    merged("type_c_off", "targeted_supply", {"all.type_c_enabled": False})
    merged("type_c_aplus_score15", "targeted_supply", {"all.type_c_requires_a_plus": True, "all.second_entry_score_min": 15})
    merged("type_c_score16_wait8", "targeted_supply", {"all.second_entry_score_min": 16, "all.second_entry_max_wait_bars_15m": 8})

    # Exit/risk repair probes. These help separate true expectancy repair from
    # merely reducing dollar exposure to bad OOS trades.
    for stop_r in (0.35, 0.40, 0.50, 0.60):
        merged(f"t1_stop_{int(stop_r*100):03d}", "targeted_exit", {"all.t1_stop_r": stop_r})
    for ladder_name, ladder in {
        "floor_light": ((1.0, 0.10), (1.5, 0.40), (2.25, 0.90)),
        "floor_balanced": ((0.75, 0.10), (1.50, 0.50), (2.25, 1.10), (3.25, 1.80)),
        "floor_fast": ((0.60, 0.00), (1.10, 0.30), (1.60, 0.70), (2.25, 1.15)),
    }.items():
        merged(ladder_name, "targeted_exit", {"all.profit_floor_ladder": ladder})
    merged("addon_off", "targeted_addon", {"all.addon_enabled": False})
    merged("addon_smaller_score18", "targeted_addon", {"all.addon_size_mult": 0.15, "all.addon_min_score": 18})
    merged("addon_after_175_score18", "targeted_addon", {"all.addon_trigger_r": 1.75, "all.addon_min_score": 18})
    for max_risk in (0.015, 0.0175, 0.020):
        merged(
            f"risk_stack_{max_risk:.4f}",
            "targeted_risk",
            {
                "all.max_risk_pct": max_risk,
                "all.risk_a_pct": max_risk * 0.64,
                "all.risk_a_plus_pct": max_risk,
                "all.risk_b_pct": max_risk * 0.40,
            },
            intent="Scale dollars without changing signal expectancy.",
        )

    # Combinations seeded by the prior OOS sweep leaders.
    leader_names = {
        str(row.get("name"))
        for row in sorted(stage1_rows, key=lambda row: float(row.get("delta_net", -999.0)), reverse=True)[:24]
    }
    if any("ma100" in name for name in leader_names):
        trend_bases = [
            ("ma100_006", {"all.min_ma100_slope_atr_4h": 0.06}),
            ("ma100_006_di", {"all.min_ma100_slope_atr_4h": 0.06, "all.require_di_alignment": True}),
            ("ma100_006_di_adx15", {"all.min_ma100_slope_atr_4h": 0.06, "all.require_di_alignment": True, "all.min_adx_4h": 15.0}),
            ("ma100_004_di_adx15", {"all.min_ma100_slope_atr_4h": 0.04, "all.require_di_alignment": True, "all.min_adx_4h": 15.0}),
        ]
        overlays = [
            ("no11", {"all.avoid_windows_et": ((11, 0, 12, 0),)}),
            ("gld_no8", {"GLD.primary_windows_et": ((9, 30, 11, 30), (13, 0, 16, 0))}),
            ("t1stop04", {"all.t1_stop_r": 0.40}),
            ("floor_light", {"all.profit_floor_ladder": ((1.0, 0.10), (1.5, 0.40), (2.25, 0.90))}),
            ("addon_off", {"all.addon_enabled": False}),
            ("valuehits2", {"all.type_a_value_hits_min": 2}),
            ("risk020", {"all.max_risk_pct": 0.020, "all.risk_a_pct": 0.013, "all.risk_a_plus_pct": 0.020, "all.risk_b_pct": 0.008}),
        ]
        for base_name, base_muts in trend_bases:
            for overlay_name, overlay_muts in overlays:
                merged(f"combo_{base_name}_{overlay_name}", "targeted_combo", {**base_muts, **overlay_muts})
            for i, (overlay_a, muts_a) in enumerate(overlays):
                for overlay_b, muts_b in overlays[i + 1:]:
                    if len(candidates) > 260:
                        break
                    merged(f"combo_{base_name}_{overlay_a}_{overlay_b}", "targeted_combo", {**base_muts, **muts_a, **muts_b})

    return dedupe_candidates(candidates)


def evaluate_oos_candidates(
    candidates: list[Candidate],
    *,
    baseline_oos: dict[str, Any],
    data_dir: Path,
    train_end: str,
    max_workers: int,
    output_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    output_path.write_text("", encoding="utf-8")
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(str(data_dir), train_end),
    ) as pool:
        futures = {pool.submit(_score_oos_worker, candidate): candidate for candidate in candidates}
        for idx, fut in enumerate(as_completed(futures), start=1):
            candidate = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:  # pragma: no cover - diagnostic runner
                row = {
                    "name": candidate.name,
                    "stage": candidate.stage,
                    "source": candidate.source,
                    "intent": candidate.intent,
                    "mutations": normalize_jsonable(candidate.mutations),
                    "error": repr(exc),
                    "oos": {},
                }
            row["oos_delta_net"] = float(row.get("oos", {}).get("net_return_pct", 0.0)) - float(baseline_oos.get("net_return_pct", 0.0))
            row["oos_delta_trades"] = float(row.get("oos", {}).get("total_trades", 0.0)) - float(baseline_oos.get("total_trades", 0.0))
            rows.append(row)
            append_jsonl(output_path, row)
            if idx % 10 == 0 or idx == len(candidates):
                best = max(rows, key=lambda item: item.get("oos_delta_net", -999.0))
                print(
                    f"[tpc] targeted OOS {idx}/{len(candidates)} best={best['name']} "
                    f"delta={best.get('oos_delta_net', 0.0):+.2f}%",
                    flush=True,
                )
    return sorted(rows, key=lambda item: item.get("oos_delta_net", -999.0), reverse=True)


def evaluate_train_oos_candidates(
    candidates: list[Candidate],
    *,
    data_dir: Path,
    train_end: str,
    max_workers: int,
    output_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    output_path.write_text("", encoding="utf-8")
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(str(data_dir), train_end),
    ) as pool:
        futures = {pool.submit(_score_train_oos_worker, candidate): candidate for candidate in candidates}
        for idx, fut in enumerate(as_completed(futures), start=1):
            candidate = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:  # pragma: no cover - diagnostic runner
                row = {
                    "name": candidate.name,
                    "stage": candidate.stage,
                    "source": candidate.source,
                    "intent": candidate.intent,
                    "mutations": normalize_jsonable(candidate.mutations),
                    "error": repr(exc),
                    "train": {},
                    "oos": {},
                }
            rows.append(row)
            append_jsonl(output_path, row)
            if idx % 5 == 0 or idx == len(candidates):
                print(f"[tpc] train+OOS validation {idx}/{len(candidates)}", flush=True)
    return rows


def _init_worker(data_dir_str: str, train_end: str) -> None:
    global _TRAIN_DATA, _FULL_DATA, _OOS_WARMUP_15M, _WORKER_DATA_DIR, _WORKER_TRAIN_END
    global _TRAIN_INDICATOR_CACHE, _OOS_INDICATOR_CACHE
    _WORKER_DATA_DIR = Path(data_dir_str)
    _WORKER_TRAIN_END = train_end
    _TRAIN_DATA = load_tpc_replay_bundle(_WORKER_DATA_DIR, end_date=train_end).data
    full = load_tpc_replay_bundle(_WORKER_DATA_DIR, end_date=None)
    _FULL_DATA = full.data
    _OOS_WARMUP_15M = infer_holdout_warmup(_FULL_DATA, train_end)
    _TRAIN_INDICATOR_CACHE = {}
    _OOS_INDICATOR_CACHE = {}


def _score_oos_worker(candidate: Candidate) -> dict[str, Any]:
    if _FULL_DATA is None or _WORKER_DATA_DIR is None:
        raise RuntimeError("worker not initialised")
    cfg = TPCBacktestConfig(initial_equity=INITIAL_EQUITY, data_dir=_WORKER_DATA_DIR)
    muts = dict(candidate.mutations)
    muts["warmup_15m"] = _OOS_WARMUP_15M
    cfg = cfg.with_overrides(muts)
    result = run_tpc_independent(_FULL_DATA, cfg, indicator_cache=_OOS_INDICATOR_CACHE)
    return {
        "name": candidate.name,
        "stage": candidate.stage,
        "source": candidate.source,
        "intent": candidate.intent,
        "mutations": normalize_jsonable(candidate.mutations),
        "oos": metrics_with_cohorts(result),
    }


def _score_train_oos_worker(candidate: Candidate) -> dict[str, Any]:
    if _TRAIN_DATA is None or _FULL_DATA is None or _WORKER_DATA_DIR is None:
        raise RuntimeError("worker not initialised")
    cfg_train = TPCBacktestConfig(initial_equity=INITIAL_EQUITY, data_dir=_WORKER_DATA_DIR).with_overrides(candidate.mutations)
    train_result = run_tpc_independent(_TRAIN_DATA, cfg_train, indicator_cache=_TRAIN_INDICATOR_CACHE)
    train = metrics_with_cohorts(train_result)

    muts = dict(candidate.mutations)
    muts["warmup_15m"] = _OOS_WARMUP_15M
    cfg_oos = TPCBacktestConfig(initial_equity=INITIAL_EQUITY, data_dir=_WORKER_DATA_DIR).with_overrides(muts)
    oos_result = run_tpc_independent(_FULL_DATA, cfg_oos, indicator_cache=_OOS_INDICATOR_CACHE)
    oos = metrics_with_cohorts(oos_result)
    return {
        "name": candidate.name,
        "stage": candidate.stage,
        "source": candidate.source,
        "intent": candidate.intent,
        "mutations": normalize_jsonable(candidate.mutations),
        "train": train,
        "oos": oos,
    }


def infer_holdout_warmup(data: dict[str, dict[str, Any]], train_end: str) -> int:
    if not data:
        return 1
    primary = max(data, key=lambda symbol: len(data[symbol]["bars_15m"].closes))
    times = pd.DatetimeIndex(data[primary]["bars_15m"].times)
    if times.tz is None:
        times = times.tz_localize("UTC")
    else:
        times = times.tz_convert("UTC")
    cutoff = pd.Timestamp(train_end)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    return int(np.searchsorted(times.values, cutoff.to_datetime64(), side="left"))


def metrics_with_cohorts(result: Any) -> dict[str, Any]:
    metrics = dict(_extract_tpc_metrics(result, INITIAL_EQUITY))
    trades = list(getattr(result, "trades", []))
    cohorts: dict[str, dict[str, float]] = {}
    cohort_specs: dict[str, list[Any]] = {
        "LONG": [t for t in trades if int(getattr(t, "direction", 0) or 0) > 0],
        "SHORT": [t for t in trades if int(getattr(t, "direction", 0) or 0) < 0],
    }
    for symbol in sorted({str(getattr(t, "symbol", "") or "") for t in trades} | {"QQQ", "GLD"}):
        cohort_specs[symbol] = [t for t in trades if str(getattr(t, "symbol", "") or "") == symbol]
    for name, items in cohort_specs.items():
        rs = np.asarray([float(getattr(t, "r_multiple", 0.0) or 0.0) for t in items], dtype=float)
        pnls = np.asarray([float(getattr(t, "pnl_dollars", 0.0) or 0.0) for t in items], dtype=float)
        mfes = np.asarray([float(getattr(t, "mfe_r", 0.0) or 0.0) for t in items], dtype=float)
        cohorts[name] = {
            "n": float(len(items)),
            "pnl": float(np.sum(pnls)) if pnls.size else 0.0,
            "avg_r": float(np.mean(rs)) if rs.size else 0.0,
            "win_rate": float(np.mean(rs > 0.0)) if rs.size else 0.0,
            "low_mfe_loss_rate": float(np.mean((mfes < 1.0) & (rs <= 0.0))) if rs.size else 0.0,
        }
    metrics["cohorts"] = cohorts
    metrics["worst_trades"] = [
        {
            "symbol": str(getattr(t, "symbol", "") or ""),
            "direction": "LONG" if int(getattr(t, "direction", 0) or 0) > 0 else "SHORT",
            "entry_time": str(getattr(t, "entry_time", "") or ""),
            "exit_time": str(getattr(t, "exit_time", "") or ""),
            "r": float(getattr(t, "r_multiple", 0.0) or 0.0),
            "pnl": float(getattr(t, "pnl_dollars", 0.0) or 0.0),
            "mfe_r": float(getattr(t, "mfe_r", 0.0) or 0.0),
            "mae_r": float(getattr(t, "mae_r", 0.0) or 0.0),
            "entry_type": str(getattr(t, "entry_type", "") or ""),
            "exit_reason": str(getattr(t, "exit_reason", "") or ""),
            "score": float(getattr(t, "score_entry", 0.0) or 0.0),
            "grade": str(getattr(t, "regime_entry", "") or ""),
        }
        for t in sorted(trades, key=lambda tr: float(getattr(tr, "pnl_dollars", 0.0) or 0.0))[:10]
    ]
    metrics["loss_concentration"] = loss_concentration(trades)
    return normalize_jsonable(metrics)


def loss_concentration(trades: list[Any]) -> dict[str, float]:
    losses = sorted([abs(float(getattr(t, "pnl_dollars", 0.0) or 0.0)) for t in trades if float(getattr(t, "pnl_dollars", 0.0) or 0.0) < 0], reverse=True)
    total_loss = sum(losses)
    return {
        "loss_count": float(len(losses)),
        "top1_loss_share": losses[0] / total_loss if total_loss > 0 and losses else 0.0,
        "top3_loss_share": sum(losses[:3]) / total_loss if total_loss > 0 else 0.0,
        "top5_loss_share": sum(losses[:5]) / total_loss if total_loss > 0 else 0.0,
    }


def select_shortlist(
    incumbent: dict[str, Any],
    *,
    stage1_rows: list[dict[str, Any]],
    targeted_oos_rows: list[dict[str, Any]],
    targeted_candidates: list[Candidate],
    limit: int,
) -> list[Candidate]:
    by_name = {candidate.name: candidate for candidate in targeted_candidates}
    selected: list[Candidate] = [Candidate("BASE_R5", "baseline", dict(incumbent), intent="Round-5 incumbent")]

    def add_candidate(row: dict[str, Any]) -> None:
        name = str(row.get("name", ""))
        if not name or name == "BASE_R5":
            return
        if name in by_name:
            selected.append(by_name[name])
            return
        muts = row.get("mutations")
        if isinstance(muts, dict):
            # Stage-1 rows store deltas for some candidates. Reconstruct only
            # rows that carried full candidate mutations.
            if any(value == "<removed>" for value in muts.values()):
                full = dict(incumbent)
                for key, value in muts.items():
                    if value == "<removed>":
                        full.pop(key, None)
                    else:
                        full[key] = value
                selected.append(Candidate(name, str(row.get("kind", "stage1")), full, intent="Prior stage-1 OOS leader"))
            else:
                full = dict(incumbent)
                full.update(muts)
                selected.append(Candidate(name, str(row.get("kind", "stage1")), full, intent="Prior stage-1 OOS leader"))

    for row in sorted(stage1_rows, key=lambda item: float(item.get("delta_net", item.get("oos_delta_net", -999.0))), reverse=True)[: max(24, limit // 2)]:
        add_candidate(row)
    for row in sorted(targeted_oos_rows, key=lambda item: float(item.get("oos_delta_net", -999.0)), reverse=True)[: max(24, limit)]:
        add_candidate(row)

    # Include balanced train candidates from the existing targeted validation if present.
    prior_train = DEFAULT_OUTPUT_ROOT / "targeted_combo_validation.jsonl"
    if prior_train.exists():
        for row in load_stage1_rows(prior_train):
            name = str(row.get("name", ""))
            if name in by_name:
                selected.append(by_name[name])

    return dedupe_candidates(selected)[:limit]


def score_validation(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    train = row.get("train", {}) or {}
    oos = row.get("oos", {}) or {}
    base_train = baseline.get("train", {}) or {}
    base_oos = baseline.get("oos", {}) or {}
    train_net = float(train.get("net_return_pct", 0.0))
    train_trades = float(train.get("total_trades", 0.0))
    oos_net = float(oos.get("net_return_pct", 0.0))
    oos_trades = float(oos.get("total_trades", 0.0))
    oos_avg_r = float(oos.get("avg_r", 0.0))
    oos_pf = float(oos.get("dollar_profit_factor", 0.0))
    oos_dd = float(oos.get("max_dd_pct", 0.0))
    base_train_net = max(float(base_train.get("net_return_pct", 0.0)), 1e-9)
    base_train_trades = max(float(base_train.get("total_trades", 0.0)), 1e-9)
    base_oos_net = float(base_oos.get("net_return_pct", 0.0))
    base_oos_trades = max(float(base_oos.get("total_trades", 0.0)), 1e-9)

    train_retention = train_net / base_train_net
    trade_retention = train_trades / base_train_trades
    oos_trade_retention = oos_trades / base_oos_trades
    objective = (
        0.38 * scale(oos_net, base_oos_net, 4.0)
        + 0.20 * scale(oos_avg_r, -0.55, 0.25)
        + 0.12 * scale(oos_pf, 0.30, 1.15)
        + 0.12 * scale(oos_trade_retention, 0.35, 0.90)
        + 0.10 * scale(train_retention, 0.35, 1.15)
        + 0.08 * scale(trade_retention, 0.35, 1.00)
        - 0.10 * scale(oos_dd, 8.0, 28.0)
    )
    passed = (
        oos_net > base_oos_net + 10.0
        and oos_avg_r > -0.30
        and oos_trades >= 10
        and train_retention >= 0.55
        and trade_retention >= 0.45
        and oos_dd <= 15.0
    )
    out = dict(row)
    out.update({
        "objective": objective,
        "passed_repair_gate": passed,
        "delta": {
            "train_net_return_pct": train_net - float(base_train.get("net_return_pct", 0.0)),
            "train_total_trades": train_trades - float(base_train.get("total_trades", 0.0)),
            "oos_net_return_pct": oos_net - base_oos_net,
            "oos_total_trades": oos_trades - float(base_oos.get("total_trades", 0.0)),
            "oos_avg_r": oos_avg_r - float(base_oos.get("avg_r", 0.0)),
            "oos_win_rate": float(oos.get("win_rate", 0.0)) - float(base_oos.get("win_rate", 0.0)),
            "oos_max_dd_pct": oos_dd - float(base_oos.get("max_dd_pct", 0.0)),
        },
        "retention": {
            "train_net": train_retention,
            "train_trades": trade_retention,
            "oos_trades": oos_trade_retention,
        },
    })
    return out


def scale(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def summarize_stage1_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        oos = row.get("oos", {}) or {}
        out.append({
            "name": row.get("name", ""),
            "kind": row.get("kind", row.get("stage", "")),
            "delta_net": row.get("delta_net", row.get("oos_delta_net", "")),
            "oos_net_return_pct": oos.get("net_return_pct", ""),
            "oos_total_trades": oos.get("total_trades", ""),
            "oos_avg_r": oos.get("avg_r", ""),
            "oos_win_rate": oos.get("win_rate", ""),
            "oos_dollar_profit_factor": oos.get("dollar_profit_factor", ""),
            "oos_max_dd_pct": oos.get("max_dd_pct", ""),
        })
    return out


def flatten_validation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = []
    for row in rows:
        train = row.get("train", {}) or {}
        oos = row.get("oos", {}) or {}
        flat.append({
            "name": row.get("name", ""),
            "stage": row.get("stage", ""),
            "objective": row.get("objective", 0.0),
            "passed_repair_gate": row.get("passed_repair_gate", False),
            "train_net_return_pct": train.get("net_return_pct", ""),
            "train_total_trades": train.get("total_trades", ""),
            "train_avg_r": train.get("avg_r", ""),
            "train_dollar_profit_factor": train.get("dollar_profit_factor", ""),
            "oos_net_return_pct": oos.get("net_return_pct", ""),
            "oos_total_trades": oos.get("total_trades", ""),
            "oos_avg_r": oos.get("avg_r", ""),
            "oos_win_rate": oos.get("win_rate", ""),
            "oos_dollar_profit_factor": oos.get("dollar_profit_factor", ""),
            "oos_max_dd_pct": oos.get("max_dd_pct", ""),
            "delta_oos_net_return_pct": row.get("delta", {}).get("oos_net_return_pct", ""),
            "delta_train_net_return_pct": row.get("delta", {}).get("train_net_return_pct", ""),
            "train_net_retention": row.get("retention", {}).get("train_net", ""),
            "train_trade_retention": row.get("retention", {}).get("train_trades", ""),
        })
    return flat


def format_report(
    *,
    baseline: dict[str, Any],
    best: dict[str, Any] | None,
    scored: list[dict[str, Any]],
    stage1_rows: list[dict[str, Any]],
    targeted_oos_rows: list[dict[str, Any]],
    weakness: dict[str, Any],
    elapsed_seconds: float,
) -> str:
    base_train = baseline.get("train", {}) or {}
    base_oos = baseline.get("oos", {}) or {}
    top_stage1 = sorted(stage1_rows, key=lambda row: float(row.get("delta_net", row.get("oos_delta_net", -999.0))), reverse=True)[:8]
    top_targeted = sorted(targeted_oos_rows, key=lambda row: float(row.get("oos_delta_net", -999.0)), reverse=True)[:8]
    top_validated = [row for row in scored if row.get("name") != "BASE_R5"][:10]

    lines = [
        "# TPC Round 5 OOS Repair Report",
        "",
        f"Elapsed minutes: {elapsed_seconds / 60.0:.1f}",
        "",
        "## Baseline",
        f"Train: net {base_train.get('net_return_pct', 0.0):+.2f}%, trades {base_train.get('total_trades', 0.0):.0f}, avgR {base_train.get('avg_r', 0.0):+.3f}, $PF {base_train.get('dollar_profit_factor', 0.0):.2f}.",
        f"OOS: net {base_oos.get('net_return_pct', 0.0):+.2f}%, trades {base_oos.get('total_trades', 0.0):.0f}, avgR {base_oos.get('avg_r', 0.0):+.3f}, win rate {base_oos.get('win_rate', 0.0):.1%}, $PF {base_oos.get('dollar_profit_factor', 0.0):.2f}, DD {base_oos.get('max_dd_pct', 0.0):.2f}%.",
        "",
        "## OOS Weakness",
    ]
    for item in weakness.get("interpretation", []):
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## Prior Ablation/Perturbation OOS Leaders",
        "| Candidate | OOS Net | Delta | Trades | AvgR | Win |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in top_stage1:
        oos = row.get("oos", {}) or {}
        lines.append(
            f"| {row.get('name', '')} | {oos.get('net_return_pct', 0.0):+.2f}% | "
            f"{row.get('delta_net', row.get('oos_delta_net', 0.0)):+.2f}% | "
            f"{oos.get('total_trades', 0.0):.0f} | {oos.get('avg_r', 0.0):+.3f} | {oos.get('win_rate', 0.0):.1%} |"
        )
    if top_targeted:
        lines.extend([
            "",
            "## New Targeted OOS Leaders",
            "| Candidate | OOS Net | Delta | Trades | AvgR | Win |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in top_targeted:
            oos = row.get("oos", {}) or {}
            lines.append(
                f"| {row.get('name', '')} | {oos.get('net_return_pct', 0.0):+.2f}% | "
                f"{row.get('oos_delta_net', 0.0):+.2f}% | {oos.get('total_trades', 0.0):.0f} | "
                f"{oos.get('avg_r', 0.0):+.3f} | {oos.get('win_rate', 0.0):.1%} |"
            )
    lines.extend([
        "",
        "## Train+OOS Validation Leaders",
        "| Candidate | Gate | Train Net | Train Trades | OOS Net | OOS Trades | OOS AvgR | OOS DD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in top_validated:
        train = row.get("train", {}) or {}
        oos = row.get("oos", {}) or {}
        lines.append(
            f"| {row.get('name', '')} | {str(row.get('passed_repair_gate', False))} | "
            f"{train.get('net_return_pct', 0.0):+.2f}% | {train.get('total_trades', 0.0):.0f} | "
            f"{oos.get('net_return_pct', 0.0):+.2f}% | {oos.get('total_trades', 0.0):.0f} | "
            f"{oos.get('avg_r', 0.0):+.3f} | {oos.get('max_dd_pct', 0.0):.2f}% |"
        )
    if best:
        train = best.get("train", {}) or {}
        oos = best.get("oos", {}) or {}
        lines.extend([
            "",
            "## Recommended Repair",
            f"Selected candidate: `{best.get('name', '')}`.",
            f"Train: net {train.get('net_return_pct', 0.0):+.2f}%, trades {train.get('total_trades', 0.0):.0f}, avgR {train.get('avg_r', 0.0):+.3f}.",
            f"OOS: net {oos.get('net_return_pct', 0.0):+.2f}%, trades {oos.get('total_trades', 0.0):.0f}, avgR {oos.get('avg_r', 0.0):+.3f}, win {oos.get('win_rate', 0.0):.1%}.",
            "This is a selection-OOS repair, so it should be promoted only after a fresh holdout or paper-trade validation.",
        ])
    return "\n".join(lines) + "\n"


def first_by_name(rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("name") == name:
            return row
    return None


def dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    out: list[Candidate] = []
    for candidate in candidates:
        sig = json.dumps(normalize_jsonable(candidate.mutations), sort_keys=True, separators=(",", ":"))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(candidate)
    return out


def normalize_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize_jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [normalize_jsonable(v) for v in value]
    if isinstance(value, list):
        return [normalize_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(normalize_jsonable(payload), indent=2, sort_keys=False), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(normalize_jsonable(payload), sort_keys=False) + "\n")


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
        writer.writerows(rows)


def infer_data_end(data_dir: Path) -> str:
    ends = []
    for symbol in SYMBOL_CONFIGS:
        path = data_dir / f"{symbol}_15m.parquet"
        if path.exists():
            df = pd.read_parquet(path, columns=[])
            ends.append(str(pd.to_datetime(df.index).max()))
    return min(ends) if ends else ""


if __name__ == "__main__":
    main()
