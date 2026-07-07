"""Incremental baseline optimizer for ALCB T2 strategy (Phase 3).

Applies experiments one-by-one in strategic order, keeping only those that
improve the composite score.  Phase 3 scoring includes a trade-frequency
component (15% weight) and rejects configs with <25 trades.

Experiment ordering:
  1. Signal expansion (trade count multipliers) — pass the 25-trade floor
  2. Alpha amplification (quality per trade)
  3. Graduated sizing (replace binary filters)
  4. Position management & fine-tuning
  5. Compound interactions (multi-feature)

Usage:
    cd bots/ibkr_trading
    PYTHONUNBUFFERED=1 python -u -m backtests.stock.auto.output.incremental_optimize
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

BOT_ROOT = Path(__file__).resolve().parents[4]
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from backtests.stock.auto.config_mutator import mutate_alcb_config
from backtests.stock.auto.scoring import composite_score, extract_metrics
from backtests.stock.config_alcb import ALCBBacktestConfig
from backtests.stock.engine.alcb_engine import ALCBIntradayEngine
from backtests.stock.engine.research_replay import ResearchReplayEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = BOT_ROOT / "backtests" / "stock" / "data" / "raw"
OUTPUT_DIR = BOT_ROOT / "backtests" / "stock" / "auto" / "output"
INITIAL_EQUITY = 10_000.0
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"

BASELINE_OVERRIDES: dict = {
    "base_risk_fraction": 0.015,
    "min_adv_usd": 5_000_000.0,
    "heat_cap_r": 10.0,
    "min_containment": 0.70,
    "max_squeeze_metric": 1.30,
    "breakout_tolerance_pct": 0.10,
}

# ---------------------------------------------------------------------------
# Experiments — ordered by expected impact
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    # ====== Group 1: Signal Expansion (maximize trade count) ======

    {"step": 1, "id": "p3_no_binary_filters",
     "description": "Remove weekday/sector binary blocks",
     "mutations": {
         "param_overrides.entry_blocked_weekdays": (),
         "param_overrides.sector_blocked_list": (),
     }},

    {"step": 2, "id": "p3_abl_direct_breakout",
     "description": "Direct breakout entry (gap breakouts that never retest)",
     "mutations": {"ablation.use_direct_breakout": True}},

    {"step": 3, "id": "p3_abl_entry_b",
     "description": "Sweep & reclaim entries",
     "mutations": {"ablation.use_entry_b": True}},

    {"step": 4, "id": "p3_abl_entry_c",
     "description": "Continuation entries",
     "mutations": {"ablation.use_entry_c": True}},

    {"step": 5, "id": "p3_abl_entry_bc",
     "description": "Both entry B + C together",
     "mutations": {
         "ablation.use_entry_b": True,
         "ablation.use_entry_c": True,
     }},

    {"step": 6, "id": "p3_abl_4h_boxes",
     "description": "4h timeframe box detection",
     "mutations": {"ablation.use_4h_boxes": True}},

    {"step": 7, "id": "p3_expand_sel_50",
     "description": "50 long candidates (from 20)",
     "mutations": {"param_overrides.selection_long_count": 50}},

    {"step": 8, "id": "p3_expand_sel_100_50",
     "description": "100 long + 50 short candidates",
     "mutations": {
         "param_overrides.selection_long_count": 100,
         "param_overrides.selection_short_count": 50,
     }},

    {"step": 9, "id": "p3_max_positions_8",
     "description": "Max 8 concurrent positions (from 5)",
     "mutations": {"param_overrides.max_positions": 8}},

    {"step": 10, "id": "p3_max_positions_10",
     "description": "Max 10 concurrent positions",
     "mutations": {"param_overrides.max_positions": 10}},

    {"step": 11, "id": "p3_sector_limit_3",
     "description": "3 positions per sector (from 2)",
     "mutations": {"param_overrides.max_positions_per_sector": 3}},

    {"step": 12, "id": "p3_runner_timeout_30",
     "description": "Runner timeout 30 bdays",
     "mutations": {
         "ablation.use_runner_timeout": True,
         "param_overrides.runner_max_bdays": 30,
     }},

    {"step": 13, "id": "p3_runner_timeout_20",
     "description": "Runner timeout 20 bdays",
     "mutations": {
         "ablation.use_runner_timeout": True,
         "param_overrides.runner_max_bdays": 20,
     }},

    # ====== Group 2: Alpha Amplification ======

    {"step": 14, "id": "p3_fix_intraday_scoring",
     "description": "Wire real intraday_evidence_score (was hardcoded 0)",
     "mutations": {"ablation.use_intraday_scoring": True}},

    {"step": 15, "id": "p3_fix_intraday_conviction",
     "description": "Conviction sizing WITH working intraday scores",
     "mutations": {
         "ablation.use_intraday_scoring": True,
         "ablation.use_conviction_sizing": True,
     }},

    {"step": 16, "id": "p3_fix_intraday_conviction_boxht",
     "description": "Full alpha stack: intraday + conviction + box height",
     "mutations": {
         "ablation.use_intraday_scoring": True,
         "ablation.use_conviction_sizing": True,
         "ablation.use_box_height_targets": True,
     }},

    {"step": 17, "id": "p3_abl_box_height_targets",
     "description": "Box-height adaptive targets alone",
     "mutations": {"ablation.use_box_height_targets": True}},

    {"step": 18, "id": "p3_abl_regime_sizing",
     "description": "Wire regime_mult as position sizing factor",
     "mutations": {"ablation.use_regime_sizing": True}},

    {"step": 19, "id": "p3_int_full_alpha_stack",
     "description": "Complete alpha: intraday + conviction + boxht + regime",
     "mutations": {
         "ablation.use_intraday_scoring": True,
         "ablation.use_conviction_sizing": True,
         "ablation.use_box_height_targets": True,
         "ablation.use_regime_sizing": True,
     }},

    # ====== Group 3: Graduated Sizing (replace binary filters) ======

    {"step": 20, "id": "p3_sizing_all_050",
     "description": "Half-size Wed/Thu + Tech (instead of blocking)",
     "mutations": {
         "ablation.use_sizing_multipliers": True,
         "param_overrides.weekday_sizing_mults": (1.0, 1.0, 0.5, 0.5, 1.0),
         "param_overrides.sector_sizing_penalty": 0.5,
     }},

    {"step": 21, "id": "p3_sizing_wed_thu_050",
     "description": "Half-size Wed/Thu only",
     "mutations": {
         "ablation.use_sizing_multipliers": True,
         "param_overrides.weekday_sizing_mults": (1.0, 1.0, 0.5, 0.5, 1.0),
     }},

    {"step": 22, "id": "p3_sizing_tech_050",
     "description": "Half-size Tech sector only",
     "mutations": {
         "ablation.use_sizing_multipliers": True,
         "param_overrides.sector_sizing_penalty": 0.5,
     }},

    # ====== Group 4: Phase 2 Single-Param Tuning ======

    {"step": 23, "id": "abl_long_only",
     "description": "Long-only mode",
     "mutations": {"ablation.use_long_only": True}},

    {"step": 24, "id": "sweep_base_risk_0.01",
     "description": "Base risk 0.01",
     "mutations": {"param_overrides.base_risk_fraction": 0.01}},

    {"step": 25, "id": "sweep_base_risk_0.012",
     "description": "Base risk 0.012",
     "mutations": {"param_overrides.base_risk_fraction": 0.012}},

    {"step": 26, "id": "sweep_stale_exit_12",
     "description": "Stale exit 12 days",
     "mutations": {"param_overrides.stale_exit_days": 12}},

    {"step": 27, "id": "sweep_atr_stop_1.25",
     "description": "ATR stop mult 1.25",
     "mutations": {"param_overrides.atr_stop_mult_std": 1.25}},

    {"step": 28, "id": "sweep_tp1_neutral_0.75",
     "description": "TP1 neutral 0.75R",
     "mutations": {"param_overrides.tp1_neutral_r": 0.75}},

    {"step": 29, "id": "int_wider_entry_agg",
     "description": "Wider entry funnel (contain=0.55, tol=0.20, pos=8)",
     "mutations": {
         "param_overrides.min_containment": 0.55,
         "param_overrides.breakout_tolerance_pct": 0.20,
         "param_overrides.max_positions": 8,
         "param_overrides.max_squeeze_metric": 1.30,
     }},

    {"step": 30, "id": "int_wider_entry_mod",
     "description": "Moderate entry funnel (contain=0.65, tol=0.15, pos=7)",
     "mutations": {
         "param_overrides.min_containment": 0.65,
         "param_overrides.breakout_tolerance_pct": 0.15,
         "param_overrides.max_positions": 7,
     }},

    # ====== Group 5: Conviction Tuning ======

    {"step": 31, "id": "p3_conviction_floor_070",
     "description": "Conviction floor 0.7 (from 0.5)",
     "mutations": {
         "ablation.use_intraday_scoring": True,
         "ablation.use_conviction_sizing": True,
         "param_overrides.conviction_size_floor": 0.7,
     }},

    {"step": 32, "id": "p3_conviction_ceil_150",
     "description": "Conviction ceil 1.5 (from 1.3)",
     "mutations": {
         "ablation.use_intraday_scoring": True,
         "ablation.use_conviction_sizing": True,
         "param_overrides.conviction_size_ceil": 1.5,
     }},

    {"step": 33, "id": "p3_good_box_premium_150",
     "description": "GOOD box premium 1.5 (from 1.2)",
     "mutations": {
         "ablation.use_intraday_scoring": True,
         "ablation.use_conviction_sizing": True,
         "param_overrides.good_box_size_premium": 1.5,
     }},

    {"step": 34, "id": "p3_min_intraday_3",
     "description": "Require intraday score >= 3",
     "mutations": {
         "ablation.use_intraday_scoring": True,
         "param_overrides.min_intraday_score": 3,
     }},

    {"step": 35, "id": "p3_boxht_tp1_075",
     "description": "Box-height TP1 mult 0.75",
     "mutations": {
         "ablation.use_box_height_targets": True,
         "param_overrides.box_height_tp1_mult": 0.75,
     }},

    {"step": 36, "id": "p3_boxht_tp2_150",
     "description": "Box-height TP2 mult 1.5",
     "mutations": {
         "ablation.use_box_height_targets": True,
         "param_overrides.box_height_tp2_mult": 1.5,
     }},

    # ====== Group 6: Compound Interactions ======

    {"step": 37, "id": "p3_int_expansion_pack",
     "description": "Entry B+C + direct breakout + sel 100 + pos 8",
     "mutations": {
         "ablation.use_entry_b": True,
         "ablation.use_entry_c": True,
         "ablation.use_direct_breakout": True,
         "param_overrides.selection_long_count": 100,
         "param_overrides.max_positions": 8,
     }},

    {"step": 38, "id": "p3_int_balanced",
     "description": "Balanced: intraday + conviction + boxht + entry_b + runner",
     "mutations": {
         "ablation.use_intraday_scoring": True,
         "ablation.use_conviction_sizing": True,
         "ablation.use_box_height_targets": True,
         "ablation.use_entry_b": True,
         "param_overrides.selection_long_count": 50,
         "param_overrides.max_positions_per_sector": 3,
         "ablation.use_runner_timeout": True,
         "param_overrides.runner_max_bdays": 30,
     }},

    {"step": 39, "id": "p3_int_alpha_expansion",
     "description": "Full alpha stack + expansion pack",
     "mutations": {
         "ablation.use_intraday_scoring": True,
         "ablation.use_conviction_sizing": True,
         "ablation.use_box_height_targets": True,
         "ablation.use_regime_sizing": True,
         "ablation.use_entry_b": True,
         "ablation.use_entry_c": True,
         "ablation.use_direct_breakout": True,
         "param_overrides.selection_long_count": 100,
         "param_overrides.max_positions": 8,
     }},

    {"step": 40, "id": "p3_int_alpha_expansion_smart",
     "description": "Full overhaul: alpha + expansion + smart filters",
     "mutations": {
         "ablation.use_intraday_scoring": True,
         "ablation.use_conviction_sizing": True,
         "ablation.use_box_height_targets": True,
         "ablation.use_regime_sizing": True,
         "ablation.use_entry_b": True,
         "ablation.use_entry_c": True,
         "ablation.use_direct_breakout": True,
         "ablation.use_sizing_multipliers": True,
         "ablation.use_runner_timeout": True,
         "param_overrides.selection_long_count": 100,
         "param_overrides.max_positions": 8,
         "param_overrides.runner_max_bdays": 30,
         "param_overrides.max_positions_per_sector": 3,
     }},

    {"step": 41, "id": "p3_int_long_only_alpha_expansion",
     "description": "Long-only + full alpha + expansion + smart filters",
     "mutations": {
         "ablation.use_long_only": True,
         "ablation.use_intraday_scoring": True,
         "ablation.use_conviction_sizing": True,
         "ablation.use_box_height_targets": True,
         "ablation.use_regime_sizing": True,
         "ablation.use_entry_b": True,
         "ablation.use_entry_c": True,
         "ablation.use_direct_breakout": True,
         "ablation.use_sizing_multipliers": True,
         "ablation.use_runner_timeout": True,
         "param_overrides.selection_long_count": 100,
         "param_overrides.max_positions": 8,
         "param_overrides.runner_max_bdays": 30,
         "param_overrides.max_positions_per_sector": 3,
     }},

    {"step": 42, "id": "p3_int_4h_alpha",
     "description": "4h boxes + full alpha stack + entry BC + direct breakout",
     "mutations": {
         "ablation.use_4h_boxes": True,
         "ablation.use_intraday_scoring": True,
         "ablation.use_conviction_sizing": True,
         "ablation.use_box_height_targets": True,
         "ablation.use_regime_sizing": True,
         "ablation.use_entry_b": True,
         "ablation.use_entry_c": True,
         "ablation.use_direct_breakout": True,
     }},

    {"step": 43, "id": "p3_int_smart_filters",
     "description": "Sizing multipliers + sector limit 3 + runner timeout 30",
     "mutations": {
         "ablation.use_sizing_multipliers": True,
         "param_overrides.max_positions_per_sector": 3,
         "ablation.use_runner_timeout": True,
         "param_overrides.runner_max_bdays": 30,
     }},

    # ====== Group 7: Direct Breakout Size Variants ======

    {"step": 44, "id": "p3_abl_direct_bk_025",
     "description": "Direct breakout at 0.25x size",
     "mutations": {
         "ablation.use_direct_breakout": True,
         "param_overrides.direct_breakout_size_mult": 0.25,
     }},

    {"step": 45, "id": "p3_abl_direct_bk_075",
     "description": "Direct breakout at 0.75x size",
     "mutations": {
         "ablation.use_direct_breakout": True,
         "param_overrides.direct_breakout_size_mult": 0.75,
     }},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_config(mutations: dict) -> ALCBBacktestConfig:
    """Build ALCBBacktestConfig with baseline defaults + accumulated mutations."""
    base = ALCBBacktestConfig(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_equity=INITIAL_EQUITY,
        tier=2,
        data_dir=DATA_DIR,
        param_overrides=dict(BASELINE_OVERRIDES),
    )
    if mutations:
        return mutate_alcb_config(base, mutations)
    return base


def run_and_score(config: ALCBBacktestConfig, replay: ResearchReplayEngine):
    """Run engine and return (CompositeScore, PerformanceMetrics, trades)."""
    engine = ALCBIntradayEngine(config, replay)
    result = engine.run()
    metrics = extract_metrics(
        result.trades, result.equity_curve, result.timestamps, INITIAL_EQUITY,
    )
    score = composite_score(metrics, INITIAL_EQUITY)
    return score, metrics, result.trades


def resolve_optimal_config(mutations: dict) -> dict:
    """Extract the full optimal config from accumulated dot-notation mutations."""
    params = dict(BASELINE_OVERRIDES)
    ablation: dict = {}
    for key, val in mutations.items():
        if key.startswith("param_overrides."):
            v = list(val) if isinstance(val, tuple) else val
            params[key.split(".", 1)[1]] = v
        elif key.startswith("ablation."):
            ablation[key.split(".", 1)[1]] = val
        else:
            params[key] = val
    result: dict = {"param_overrides": params}
    if ablation:
        result["ablation"] = ablation
    return result


def fmt_row(step, eid, score, delta, trades, pf, wr, dd, net, freq, status, desc):
    """Format a TSV row."""
    return (
        f"{step}\t{eid}\t{score:.6f}\t{delta:+.2f}\t{trades}\t"
        f"{pf:.2f}\t{wr:.1f}%\t{dd:.1f}%\t{net:.2f}\t{freq:.2f}\t{status}\t{desc}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    print("=" * 70)
    print("ALCB Phase 3 Incremental Baseline Optimization")
    print("=" * 70)
    print(f"  Equity: ${INITIAL_EQUITY:,.0f}  |  Period: {START_DATE} -> {END_DATE}")
    print(f"  Experiments: {len(EXPERIMENTS)}  |  Tier: 2 (30m intraday)")
    print(f"  Scoring: 25% Calmar + 20% PF + 20% InvDD + 20% Net + 15% Freq")
    print(f"  Hard reject: <25 trades, >35% DD, PF<0.8")

    # ------------------------------------------------------------------
    # 1. Load data once
    # ------------------------------------------------------------------
    print("\n[1/3] Loading data...", flush=True)
    t_start = time.time()
    replay = ResearchReplayEngine(data_dir=DATA_DIR)
    replay.load_all_data()
    print(f"  Data loaded in {time.time() - t_start:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # 2. Run baseline
    # ------------------------------------------------------------------
    print("\n[2/3] Running baseline...", flush=True)
    t0 = time.time()
    config = build_config({})
    baseline_score, baseline_metrics, baseline_trades = run_and_score(config, replay)
    bl_time = time.time() - t0

    print(
        f"  Baseline: score={baseline_score.total:.4f}  trades={len(baseline_trades)}  "
        f"WR={baseline_metrics.win_rate * 100:.1f}%  "
        f"PF={baseline_metrics.profit_factor:.2f}  "
        f"DD={baseline_metrics.max_drawdown_pct * 100:.1f}%  "
        f"net=${baseline_metrics.net_profit:.2f}  "
        f"wr={baseline_score.wr_component:.2f}  ({bl_time:.1f}s)",
        flush=True,
    )
    if baseline_score.rejected:
        print(f"  WARNING: Baseline REJECTED — {baseline_score.reject_reason}", flush=True)

    rows: list[str] = []
    rows.append(
        fmt_row(
            0, "baseline", baseline_score.total, 0.0,
            len(baseline_trades), baseline_metrics.profit_factor,
            baseline_metrics.win_rate * 100,
            baseline_metrics.max_drawdown_pct * 100, baseline_metrics.net_profit,
            baseline_score.wr_component,
            "baseline", "Harness default config",
        )
    )

    # ------------------------------------------------------------------
    # 3. Incremental keep-or-revert loop
    # ------------------------------------------------------------------
    print(f"\n[3/3] Running {len(EXPERIMENTS)} experiments incrementally...", flush=True)
    print("-" * 90, flush=True)

    running_mutations: dict = {}
    running_best = baseline_score.total
    kept: list[str] = []
    discarded: list[str] = []

    for exp in EXPERIMENTS:
        step = exp["step"]
        eid = exp["id"]
        desc = exp["description"]

        snapshot = dict(running_mutations)
        running_mutations.update(exp["mutations"])

        t0 = time.time()
        config = build_config(running_mutations)
        score, metrics, trades = run_and_score(config, replay)
        elapsed = time.time() - t0

        delta = (
            (score.total - running_best) / running_best * 100
            if running_best > 0
            else (score.total * 100 if score.total > 0 else 0.0)
        )

        if score.total > running_best:
            status = "keep"
            running_best = score.total
            kept.append(eid)
            marker = "KEEP"
        else:
            status = "discard"
            running_mutations = snapshot
            discarded.append(eid)
            marker = "DISCARD"

        rej = " [REJECTED]" if score.rejected else ""
        print(
            f"  Step {step:>2}: {eid:<40s} "
            f"score={score.total:.4f} ({delta:+6.2f}%)  "
            f"trades={len(trades):>3}  WR={metrics.win_rate * 100:>5.1f}%  "
            f"PF={metrics.profit_factor:>5.2f}  "
            f"DD={metrics.max_drawdown_pct * 100:>4.1f}%  "
            f"net=${metrics.net_profit:>8.2f}  "
            f"[{marker}]{rej}  ({elapsed:.1f}s)",
            flush=True,
        )

        rows.append(
            fmt_row(
                step, eid, score.total, delta,
                len(trades), metrics.profit_factor,
                metrics.win_rate * 100,
                metrics.max_drawdown_pct * 100, metrics.net_profit,
                score.wr_component,
                status, desc,
            )
        )

    # ------------------------------------------------------------------
    # 4. Final validation run with optimal config
    # ------------------------------------------------------------------
    print("-" * 90, flush=True)
    print("\nRunning final validation with optimal config...", flush=True)
    t0 = time.time()
    config = build_config(running_mutations)
    final_score, final_metrics, final_trades = run_and_score(config, replay)
    final_time = time.time() - t0

    final_delta = (
        (final_score.total - baseline_score.total) / baseline_score.total * 100
        if baseline_score.total > 0
        else 0.0
    )

    print(
        f"  Final:    score={final_score.total:.4f} ({final_delta:+.2f}% vs baseline)  "
        f"trades={len(final_trades)}  WR={final_metrics.win_rate * 100:.1f}%  "
        f"PF={final_metrics.profit_factor:.2f}  "
        f"DD={final_metrics.max_drawdown_pct * 100:.1f}%  "
        f"net=${final_metrics.net_profit:.2f}  ({final_time:.1f}s)",
        flush=True,
    )

    rows.append(
        fmt_row(
            "final", "optimal", final_score.total, final_delta,
            len(final_trades), final_metrics.profit_factor,
            final_metrics.win_rate * 100,
            final_metrics.max_drawdown_pct * 100, final_metrics.net_profit,
            final_score.wr_component,
            "final", "All kept mutations combined",
        )
    )

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # incremental_results.tsv
    tsv_path = OUTPUT_DIR / "incremental_results.tsv"
    header = (
        "step\texperiment_id\tscore\tdelta_pct\ttrades\tpf\twin_rate\t"
        "max_dd\tnet_profit\twr_comp\tstatus\tdescription"
    )
    tsv_path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"\n  Saved: {tsv_path}", flush=True)

    # optimal_config.json
    optimal = resolve_optimal_config(running_mutations)
    json_path = OUTPUT_DIR / "optimal_config.json"
    json_path.write_text(
        json.dumps(optimal, indent=2, sort_keys=True, default=list) + "\n",
        encoding="utf-8",
    )
    print(f"  Saved: {json_path}", flush=True)

    # optimal_baseline.txt (summary)
    txt_path = OUTPUT_DIR / "optimal_baseline.txt"
    lines = [
        "ALCB Phase 3 Incremental Optimization Results",
        "=" * 50,
        f"Period: {START_DATE} -> {END_DATE}",
        f"Equity: ${INITIAL_EQUITY:,.0f}",
        f"Scoring: 10% Calmar + 20% PF + 10% InvDD + 30% Net + 15% WR + 15% Edge",
        "",
        "Baseline Config:",
        f"  {json.dumps(dict(BASELINE_OVERRIDES), indent=2)}",
        "",
        "Optimal Config:",
        f"  {json.dumps(optimal, indent=2, default=list)}",
        "",
        f"Baseline Score: {baseline_score.total:.6f}",
        f"Optimal Score:  {final_score.total:.6f}",
        f"Improvement:    {final_delta:+.2f}%",
        "",
        f"Trades: {len(baseline_trades)} -> {len(final_trades)}",
        f"Win Rate: {baseline_metrics.win_rate * 100:.1f}% -> {final_metrics.win_rate * 100:.1f}%",
        f"PF:     {baseline_metrics.profit_factor:.2f} -> {final_metrics.profit_factor:.2f}",
        f"Max DD: {baseline_metrics.max_drawdown_pct * 100:.1f}% -> {final_metrics.max_drawdown_pct * 100:.1f}%",
        f"Net:    ${baseline_metrics.net_profit:.2f} -> ${final_metrics.net_profit:.2f}",
        f"WR:     {baseline_score.wr_component:.2f} -> {final_score.wr_component:.2f}",
        "",
        "Kept:",
    ]
    for eid in kept:
        lines.append(f"  + {eid}")
    lines.append("")
    lines.append("Discarded:")
    for eid in discarded:
        lines.append(f"  - {eid}")
    lines.append("")

    total_time = time.time() - t_start
    lines.append(f"Total time: {total_time:.0f}s ({total_time / 60:.1f}min)")

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved: {txt_path}", flush=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  Kept: {len(kept)}/{len(EXPERIMENTS)} experiments")
    print(f"  Score: {baseline_score.total:.4f} -> {final_score.total:.4f} ({final_delta:+.2f}%)")
    print(f"  Trades: {len(baseline_trades)} -> {len(final_trades)}")
    print(f"  Total time: {total_time:.0f}s ({total_time / 60:.1f}min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
