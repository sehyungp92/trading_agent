"""IARIC Tier B Optimization Sweep.

Tests surgical Tier B parameter adjustments to improve overall strategy quality.
Runs baseline + 10 independent lever variants + 1 best-combo = 12 runs.

Levers:
  1. max_positions_tier_b: 3, 4, 6 (default 5)
  2. pb_v2_signal_floor_tier_b: 78, 80 (default 0 = use global 75)
  3. regime_b_carry_mult: 0.0, 0.3 (default 0.6)
  4. t2_regime_b_sizing_mult: 0.5, 0.7 (default 1.0)
  5. pb_delayed_confirm_enabled: False (default True)

Usage::
    python -m backtests.stock.auto.iaric.run_tier_b_sweep
"""
from __future__ import annotations

import io
import json
import sys
import time as _time
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from backtests.stock.analysis.iaric_pullback_diagnostics import pullback_full_diagnostic
from backtests.stock.auto.config_mutator import mutate_iaric_config
from backtests.stock.config_iaric import IARICBacktestConfig
from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine
from backtests.stock.engine.research_replay import ResearchReplayEngine

# ── Constants ──────────────────────────────────────────────────────────────────
PHASE_STATE_PATH = Path("backtests/stock/auto/iaric/output_multiphase/phase_state.json")
DATA_DIR = Path("backtests/stock/data/raw")
OUTPUT_DIR = Path("backtests/stock/auto/iaric/output_tier_b_sweep")
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"
INITIAL_EQUITY = 10_000.0

# ── Variant definitions ───────────────────────────────────────────────────────
# Each tuple: (name, mutations_dict)
# Top-level keys (e.g. max_positions_tier_b) are applied directly to IARICBacktestConfig.
# Dotted keys (param_overrides.X) are merged into config.param_overrides.
VARIANTS: list[tuple[str, dict]] = [
    # Baseline (no extra mutations)
    ("baseline", {}),
    # Lever 1: Position cap
    ("max_pos_b_3", {"max_positions_tier_b": 3}),
    ("max_pos_b_4", {"max_positions_tier_b": 4}),
    ("max_pos_b_6", {"max_positions_tier_b": 6}),
    # Lever 2: Tier-B signal floor
    ("floor_b_78", {"param_overrides.pb_v2_signal_floor_tier_b": 78.0}),
    ("floor_b_80", {"param_overrides.pb_v2_signal_floor_tier_b": 80.0}),
    # Lever 3: Carry restriction
    ("carry_b_0.0", {"param_overrides.regime_b_carry_mult": 0.0}),
    ("carry_b_0.3", {"param_overrides.regime_b_carry_mult": 0.3}),
    # Lever 4: Sizing discount
    ("sizing_b_0.5", {"param_overrides.t2_regime_b_sizing_mult": 0.5}),
    ("sizing_b_0.7", {"param_overrides.t2_regime_b_sizing_mult": 0.7}),
    # Lever 5: Disable DELAYED_CONFIRM
    ("no_delayed", {"param_overrides.pb_delayed_confirm_enabled": False}),
]

# Lever groups for best-combo selection (name prefix -> lever label)
LEVER_GROUPS = {
    "max_pos_b": "Lever 1: max_positions_tier_b",
    "floor_b": "Lever 2: pb_v2_signal_floor_tier_b",
    "carry_b": "Lever 3: regime_b_carry_mult",
    "sizing_b": "Lever 4: t2_regime_b_sizing_mult",
    "no_delayed": "Lever 5: pb_delayed_confirm_enabled",
}


def _extract_metrics(trades: list) -> dict:
    """Extract comparison metrics from trade list with tier breakdown."""
    empty = {
        "trades": 0, "wr": "0.0%", "mean_r": 0.0, "total_r": 0.0,
        "pnl": 0.0, "pf": 0.0,
        "a_trades": 0, "a_mean_r": 0.0, "a_total_r": 0.0, "a_pnl": 0.0,
        "b_trades": 0, "b_mean_r": 0.0, "b_total_r": 0.0, "b_pnl": 0.0,
        "b_wr": "0.0%",
        "routes": {},
    }
    if not trades:
        return empty

    wins = [t for t in trades if t.r_multiple > 0]
    total_r = sum(t.r_multiple for t in trades)
    gross_win = sum(t.r_multiple for t in wins)
    gross_loss = abs(sum(t.r_multiple for t in trades if t.r_multiple <= 0))
    pnl = sum(t.pnl_net for t in trades)

    # Tier breakdown
    tier_a = [t for t in trades if t.regime_tier == "A"]
    tier_b = [t for t in trades if t.regime_tier == "B"]
    tier_a_r = sum(t.r_multiple for t in tier_a)
    tier_b_r = sum(t.r_multiple for t in tier_b)
    tier_b_wins = [t for t in tier_b if t.r_multiple > 0]

    # Route breakdown (from metadata)
    route_map: dict[str, list] = {}
    for t in trades:
        route = t.metadata.get("entry_route_family", "") or t.entry_type or "UNKNOWN"
        route_map.setdefault(route, []).append(t)

    routes = {}
    for route, rtrades in sorted(route_map.items()):
        r_sum = sum(t.r_multiple for t in rtrades)
        r_wins = [t for t in rtrades if t.r_multiple > 0]
        routes[route] = {
            "trades": len(rtrades),
            "wr": f"{len(r_wins)/len(rtrades)*100:.1f}%" if rtrades else "0.0%",
            "mean_r": round(r_sum / len(rtrades), 4) if rtrades else 0.0,
            "total_r": round(r_sum, 2),
        }

    return {
        "trades": len(trades),
        "wr": f"{len(wins)/len(trades)*100:.1f}%",
        "mean_r": round(total_r / len(trades), 4),
        "total_r": round(total_r, 2),
        "pnl": round(pnl, 2),
        "pf": round(gross_win / max(gross_loss, 0.01), 2),
        "a_trades": len(tier_a),
        "a_mean_r": round(tier_a_r / max(len(tier_a), 1), 4),
        "a_total_r": round(tier_a_r, 2),
        "a_pnl": round(sum(t.pnl_net for t in tier_a), 2),
        "b_trades": len(tier_b),
        "b_mean_r": round(tier_b_r / max(len(tier_b), 1), 4),
        "b_total_r": round(tier_b_r, 2),
        "b_pnl": round(sum(t.pnl_net for t in tier_b), 2),
        "b_wr": f"{len(tier_b_wins)/max(len(tier_b),1)*100:.1f}%",
        "routes": routes,
    }


def _select_best_combo(results: list[tuple[str, dict]]) -> dict:
    """Select best value for each lever by total_r improvement over baseline."""
    baseline_r = results[0][1]["total_r"]  # first entry is baseline
    best_per_lever: dict[str, tuple[str, dict, float]] = {}

    for name, metrics in results[1:]:  # skip baseline
        # Find which lever group this variant belongs to
        lever_key = None
        for prefix in LEVER_GROUPS:
            if name.startswith(prefix):
                lever_key = prefix
                break
        if lever_key is None:
            continue

        improvement = metrics["total_r"] - baseline_r
        if lever_key not in best_per_lever or improvement > best_per_lever[lever_key][2]:
            best_per_lever[lever_key] = (name, metrics, improvement)

    # Build combo mutations: only include levers that improved over baseline
    combo_muts: dict = {}
    selections: list[str] = []
    for lever_key, (name, _metrics, improvement) in best_per_lever.items():
        if improvement <= 0:
            selections.append(f"  {LEVER_GROUPS[lever_key]}: SKIP (best={name}, delta={improvement:+.2f}R)")
            continue
        # Find the variant mutations
        for vname, vmuts in VARIANTS:
            if vname == name:
                combo_muts.update(vmuts)
                selections.append(f"  {LEVER_GROUPS[lever_key]}: {name} (delta={improvement:+.2f}R)")
                break

    return {"mutations": combo_muts, "selections": selections}


def _print_overall_table(results: list[tuple[str, dict]]) -> str:
    """Format overall comparison table."""
    lines = []
    header = f"{'Variant':<25} {'Trades':>6} {'WR':>7} {'MeanR':>8} {'TotalR':>8} {'PnL':>10} {'PF':>6}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, m in results:
        lines.append(
            f"{name:<25} {m['trades']:>6} {m['wr']:>7} {m['mean_r']:>8.4f} "
            f"{m['total_r']:>8.2f} {m['pnl']:>10.2f} {m['pf']:>6.2f}"
        )
    return "\n".join(lines)


def _print_tier_table(results: list[tuple[str, dict]]) -> str:
    """Format tier breakdown table."""
    lines = []
    header = (
        f"{'Variant':<25} {'A_Trades':>8} {'A_MeanR':>8} {'A_TotalR':>9} {'A_PnL':>10} "
        f"{'B_Trades':>8} {'B_WR':>7} {'B_MeanR':>8} {'B_TotalR':>9} {'B_PnL':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name, m in results:
        lines.append(
            f"{name:<25} {m['a_trades']:>8} {m['a_mean_r']:>8.4f} {m['a_total_r']:>9.2f} {m['a_pnl']:>10.2f} "
            f"{m['b_trades']:>8} {m['b_wr']:>7} {m['b_mean_r']:>8.4f} {m['b_total_r']:>9.2f} {m['b_pnl']:>10.2f}"
        )
    return "\n".join(lines)


def _print_route_table(results: list[tuple[str, dict]]) -> str:
    """Format route breakdown for baseline and key variants."""
    lines = []
    # Collect all routes seen
    all_routes: set[str] = set()
    for _name, m in results:
        all_routes.update(m["routes"].keys())
    sorted_routes = sorted(all_routes)

    for name, m in results:
        lines.append(f"\n  {name}:")
        rh = f"    {'Route':<25} {'Trades':>6} {'WR':>7} {'MeanR':>8} {'TotalR':>8}"
        lines.append(rh)
        lines.append("    " + "-" * (len(rh) - 4))
        for route in sorted_routes:
            rd = m["routes"].get(route, {"trades": 0, "wr": "0.0%", "mean_r": 0.0, "total_r": 0.0})
            lines.append(
                f"    {route:<25} {rd['trades']:>6} {rd['wr']:>7} {rd['mean_r']:>8} {rd['total_r']:>8}"
            )
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load R4 baseline mutations
    phase_state = json.loads(PHASE_STATE_PATH.read_text(encoding="utf-8"))
    base_mutations = phase_state["cumulative_mutations"]
    print(f"Loaded {len(base_mutations)} R4 baseline mutations from {PHASE_STATE_PATH}")
    print(f"Date range: {START_DATE} -- {END_DATE}")
    print(f"Initial equity: ${INITIAL_EQUITY:,.0f}")
    print()

    # Load replay data once
    replay = ResearchReplayEngine(data_dir=DATA_DIR)
    print("Loading bar data...")
    replay.load_all_data()
    print("Bar data loaded.\n")

    base_config = IARICBacktestConfig(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_equity=INITIAL_EQUITY,
        tier=3,
        data_dir=DATA_DIR,
    )

    # ── Phase 1: Run all independent variants ─────────────────────────────────
    results: list[tuple[str, dict]] = []
    baseline_result = None

    for name, variant_muts in VARIANTS:
        print(f"{'='*60}")
        print(f"Running variant: {name}")
        print(f"{'='*60}")

        # Merge R4 base + variant mutations
        muts = {**base_mutations, **variant_muts}
        config = mutate_iaric_config(base_config, muts)

        t0 = _time.time()
        collect_diag = (name == "baseline")
        engine = IARICPullbackEngine(config, replay, collect_diagnostics=collect_diag)
        result = engine.run()
        elapsed = _time.time() - t0

        metrics = _extract_metrics(result.trades)
        results.append((name, metrics))
        print(f"  Completed in {elapsed:.0f}s -- {metrics['trades']} trades, "
              f"total_r={metrics['total_r']}, PF={metrics['pf']}")

        # Save baseline diagnostics
        if name == "baseline":
            baseline_result = result
            diag = pullback_full_diagnostic(
                result.trades,
                replay=replay,
                daily_selections=result.daily_selections,
                candidate_ledger=result.candidate_ledger,
                funnel_counters=result.funnel_counters,
                rejection_log=result.rejection_log,
                shadow_outcomes=result.shadow_outcomes,
                selection_attribution=result.selection_attribution,
                fsm_log=result.fsm_log,
            )
            diag_path = OUTPUT_DIR / "tier_b_sweep_baseline_diagnostics.txt"
            diag_path.write_text(diag, encoding="utf-8")
            print(f"  Baseline diagnostics saved to {diag_path}")
            del baseline_result  # free diagnostic-heavy result before remaining runs

    # ── Phase 2: Best-combo selection and run ─────────────────────────────────
    print(f"\n\n{'='*80}")
    print("BEST-COMBO SELECTION")
    print(f"{'='*80}\n")

    combo_info = _select_best_combo(results)
    for sel in combo_info["selections"]:
        print(sel)

    if combo_info["mutations"]:
        print(f"\nRunning best-combo variant with {len(combo_info['mutations'])} mutations...")
        combo_muts = {**base_mutations, **combo_info["mutations"]}
        combo_config = mutate_iaric_config(base_config, combo_muts)

        t0 = _time.time()
        engine = IARICPullbackEngine(combo_config, replay, collect_diagnostics=True)
        combo_result = engine.run()
        elapsed = _time.time() - t0

        combo_metrics = _extract_metrics(combo_result.trades)
        results.append(("BEST_COMBO", combo_metrics))
        print(f"  Completed in {elapsed:.0f}s -- {combo_metrics['trades']} trades, "
              f"total_r={combo_metrics['total_r']}, PF={combo_metrics['pf']}")

        # Save combo diagnostics
        diag = pullback_full_diagnostic(
            combo_result.trades,
            replay=replay,
            daily_selections=combo_result.daily_selections,
            candidate_ledger=combo_result.candidate_ledger,
            funnel_counters=combo_result.funnel_counters,
            rejection_log=combo_result.rejection_log,
            shadow_outcomes=combo_result.shadow_outcomes,
            selection_attribution=combo_result.selection_attribution,
            fsm_log=combo_result.fsm_log,
        )
        diag_path = OUTPUT_DIR / "tier_b_sweep_best_combo_diagnostics.txt"
        diag_path.write_text(diag, encoding="utf-8")
        print(f"  Best-combo diagnostics saved to {diag_path}")
    else:
        print("\nNo lever improved over baseline -- skipping combo run.")

    # ── Summary tables ────────────────────────────────────────────────────────
    summary_lines = []
    summary_lines.append(f"{'='*80}")
    summary_lines.append("IARIC TIER B OPTIMIZATION SWEEP -- SUMMARY")
    summary_lines.append(f"{'='*80}")
    summary_lines.append(f"Date range: {START_DATE} -- {END_DATE}")
    summary_lines.append(f"Initial equity: ${INITIAL_EQUITY:,.0f}")
    summary_lines.append(f"R4 baseline mutations: {len(base_mutations)}")
    summary_lines.append("")

    summary_lines.append("TABLE 1: OVERALL COMPARISON")
    summary_lines.append("")
    summary_lines.append(_print_overall_table(results))

    summary_lines.append("")
    summary_lines.append("")
    summary_lines.append("TABLE 2: TIER BREAKDOWN")
    summary_lines.append("")
    summary_lines.append(_print_tier_table(results))

    summary_lines.append("")
    summary_lines.append("")
    summary_lines.append("TABLE 3: ROUTE BREAKDOWN (baseline + best variants)")
    # Show baseline, best-combo, and any variant with > baseline total_r
    baseline_r = results[0][1]["total_r"]
    route_entries = [results[0]]  # always show baseline
    for name, m in results[1:]:
        if m["total_r"] > baseline_r or name == "BEST_COMBO":
            route_entries.append((name, m))
    if len(route_entries) == 1:
        # No improvements -- show top 3 by total_r for comparison
        sorted_by_r = sorted(results[1:], key=lambda x: x[1]["total_r"], reverse=True)
        route_entries.extend(sorted_by_r[:3])
    summary_lines.append(_print_route_table(route_entries))

    if combo_info["mutations"]:
        summary_lines.append("")
        summary_lines.append("")
        summary_lines.append("BEST-COMBO SELECTIONS:")
        for sel in combo_info["selections"]:
            summary_lines.append(sel)
        summary_lines.append("")
        summary_lines.append(f"Combo mutations: {json.dumps(combo_info['mutations'], indent=2)}")

    summary_text = "\n".join(summary_lines)
    print(f"\n\n{summary_text}")

    # Save summary
    summary_path = OUTPUT_DIR / "tier_b_sweep_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
