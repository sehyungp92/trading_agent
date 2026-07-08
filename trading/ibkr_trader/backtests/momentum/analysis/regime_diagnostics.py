from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Iterable

import numpy as np

from strategies.scalp._shared.nq_contract import spec_for

MODULE_SPECS = {
    "second_wind": ("nq_1", "Second-Wind Continuation", "AM trend + compression + PM squeeze fire"),
    "structural_expansion": ("nq_2", "Structural Expansion", "clean 15m IB acceptance with VWAP alignment"),
    "liquidity_reversion": ("nq_3", "Liquidity Reversion", "failed sweep + reclaim/loss back toward value"),
}


def generate_regime_diagnostics(
    trades: list[Any],
    metrics: dict[str, float],
    *,
    signal_events: list[Any] | None = None,
    equity_curve: Iterable[float] | None = None,
    timestamps: Iterable[Any] | None = None,
    title: str = "NQ_REGIME diagnostics",
) -> str:
    events = list(signal_events or [])
    sections = [
        _header(title, trades, metrics),
        _strength_weakness_snapshot(trades, metrics, events),
        _component_edge_scorecard(trades, metrics),
        _routing_quality(events, metrics),
        _regime_confidence_pressure(events),
        _candidate_opportunity_funnel(trades, events),
        _structural_expansion_diagnostics(trades),
        _liquidity_reversion_diagnostics(trades),
        _second_wind_diagnostics(trades),
        _setup_quality_breakdowns(trades),
        _risk_room_quality(trades),
        _direction_asymmetry_by_module(trades),
        _winner_loser_entry_profile(trades),
        _exit_and_mfe_mae(trades),
        _excursion_data_integrity(trades),
        _exit_efficiency_by_module(trades),
        _mfe_threshold_retention(trades),
        _profit_floor_sensitivity(trades),
        _hold_duration_analysis(trades),
        _time_breakdowns(trades),
        _monthly_pnl(trades),
        _year_module_stability(trades),
        _trade_gap_analysis(trades),
        _daily_sequence_diagnostics(trades),
        _trade_autocorrelation(trades),
        _rolling_expectancy(trades),
        _r_distribution(trades),
        _drawdown_profile(equity_curve, timestamps),
        _closed_trade_drawdown_anatomy(trades),
        _accounting_reconciliation(trades, metrics, equity_curve),
        _baseline_readiness(metrics),
    ]
    return "\n\n".join(section for section in sections if section)


def _header(title: str, trades: list[Any], metrics: dict[str, float]) -> str:
    modules = Counter(_module(trade) for trade in trades)
    module_text = ", ".join(f"{key}={value}" for key, value in sorted(modules.items())) or "none"
    lines = [
        title,
        "=" * len(title),
        "Baseline profile: strict spec baseline, all three modules enabled, completed 5m replay, 15m context only at completed boundaries.",
        f"Trades: {len(trades)} ({module_text})",
        f"Net: ${metrics.get('net_profit', 0.0):+.2f} | PF={metrics.get('profit_factor', 0.0):.2f} | "
        f"WR={metrics.get('win_rate', 0.0):.1%} | avgR={metrics.get('avg_r', 0.0):+.3f} | "
        f"totalR={metrics.get('total_r', 0.0):+.2f} | DD={metrics.get('max_drawdown_pct', 0.0):.1%}",
        f"Frequency: {metrics.get('trades_per_month', 0.0):.2f}/month | expectancy=${metrics.get('expectancy_dollar', 0.0):+.2f}",
    ]
    return "\n".join(lines)


def _strength_weakness_snapshot(trades: list[Any], metrics: dict[str, float], events: list[Any]) -> str:
    lines = ["=== Strength / Weakness Snapshot ==="]
    if not trades:
        lines.append("  No trades.")
        return "\n".join(lines)
    module_groups = {module: [trade for trade in trades if _module(trade) == module] for module in MODULE_SPECS}
    active = [(module, group, _stats(group)) for module, group in module_groups.items() if group]
    best_module = max(active, key=lambda item: item[2]["avg_r"]) if active else None
    worst_module = min(active, key=lambda item: item[2]["avg_r"]) if active else None
    exit_groups: dict[str, list[Any]] = defaultdict(list)
    for trade in trades:
        exit_groups[_attr(trade, "exit_reason") or "unknown"].append(trade)
    eligible_exits = [(reason, group, _stats(group)) for reason, group in exit_groups.items() if len(group) >= 2]
    best_exit = max(eligible_exits, key=lambda item: item[2]["avg_r"]) if eligible_exits else None
    worst_exit = min(eligible_exits, key=lambda item: item[2]["avg_r"]) if eligible_exits else None
    winners = sorted([trade for trade in trades if _num(trade, "pnl_dollars") > 0], key=lambda trade: _num(trade, "pnl_dollars"), reverse=True)
    gross_win = sum(_num(trade, "pnl_dollars") for trade in winners)
    top5_share = sum(_num(trade, "pnl_dollars") for trade in winners[:5]) / gross_win if gross_win > 0 else 0.0
    strengths = []
    if best_module is not None:
        spec_id, label, _ = MODULE_SPECS[best_module[0]]
        strengths.append(f"{spec_id} {label}: N={len(best_module[1])}, avgR={best_module[2]['avg_r']:+.3f}, PF={best_module[2]['pf']:.2f}")
    if best_exit is not None:
        strengths.append(f"Best exit cohort: {best_exit[0]} N={len(best_exit[1])}, avgR={best_exit[2]['avg_r']:+.3f}")
    year_strength = _yearly_strength_summary(trades)
    if year_strength:
        strengths.append(year_strength)
    strengths.append(f"Top 5 winners contribute {top5_share:.0%} of gross wins")
    weaknesses = []
    if worst_module is not None:
        spec_id, label, _ = MODULE_SPECS[worst_module[0]]
        weaknesses.append(f"{spec_id} {label}: N={len(worst_module[1])}, avgR={worst_module[2]['avg_r']:+.3f}, PF={worst_module[2]['pf']:.2f}")
    if worst_exit is not None:
        weaknesses.append(f"Weakest exit cohort: {worst_exit[0]} N={len(worst_exit[1])}, avgR={worst_exit[2]['avg_r']:+.3f}")
    leakage = _mfe_leakage_summary(trades)
    if leakage:
        weaknesses.append(leakage)
    bottleneck = _top_candidate_bottleneck(events)
    if bottleneck:
        weaknesses.append(bottleneck)
    weaknesses.append(
        f"Module coverage {metrics.get('module_coverage', 0.0):.0%}; min module trades {metrics.get('min_module_trades', 0.0):.0f}"
    )
    lines.append("  Strengths")
    for item in strengths[:4]:
        lines.append(f"    - {item}")
    lines.append("  Weaknesses")
    for item in weaknesses[:5]:
        lines.append(f"    - {item}")
    return "\n".join(lines)


def _component_edge_scorecard(trades: list[Any], metrics: dict[str, float]) -> str:
    lines = ["=== Component Edge Scorecard ==="]
    lines.append("  Each row is cohort-pure: only trades actually routed to that module are counted.")
    header = f"  {'Spec':6s} {'Component':24s} {'N':>4s} {'WR':>6s} {'PF':>6s} {'AvgR':>8s} {'TotR':>8s} {'MFE':>7s} {'MAE':>7s} {'EdgeRead'}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for module, (spec_id, label, edge) in MODULE_SPECS.items():
        group = [trade for trade in trades if _module(trade) == module]
        stats = _stats(group)
        verdict = _edge_verdict(group, stats)
        lines.append(
            f"  {spec_id:6s} {label:24s} {len(group):4d} {stats['wr']:5.0%} {stats['pf']:6.2f} "
            f"{stats['avg_r']:+8.3f} {stats['total_r']:+8.2f} {stats['avg_mfe']:+7.2f} {stats['avg_mae']:+7.2f} {verdict}"
        )
        lines.append(f"         Edge thesis: {edge}.")
    lines.append(
        f"  Module coverage: {metrics.get('module_coverage', 0.0):.0%}; "
        f"minimum module trades: {metrics.get('min_module_trades', 0.0):.0f}."
    )
    return "\n".join(lines)


def _routing_quality(events: list[Any], metrics: dict[str, float]) -> str:
    lines = ["=== Regime Routing Quality ==="]
    routing = _routing_events(events)
    if not routing:
        lines.append("  No routing events available.")
        return "\n".join(lines)
    reasons = Counter(str(getattr(event, "details", {}).get("reason", "unknown")) for event in routing)
    selected = sum(1 for event in routing if getattr(event, "details", {}).get("selected"))
    blocked_total = sum(float(getattr(event, "details", {}).get("blocked", 0) or 0) for event in routing)
    lines.append(f"  Routing decisions: {len(routing)}")
    lines.append(f"  Selected rate: {selected / len(routing):.1%}")
    lines.append(f"  Avg blocked candidates/decision: {blocked_total / len(routing):.2f}")
    lines.append(f"  News vetoes: {metrics.get('news_veto_events', 0):.0f}; size blocks: {metrics.get('entry_blocked_by_size', 0):.0f}; session blocks: {metrics.get('entry_blocked_by_session', 0):.0f}.")
    lines.append("  Reason distribution:")
    for reason, count in reasons.most_common(10):
        lines.append(f"    {reason:32s} {count:5d} ({count / len(routing):5.1%})")
    return "\n".join(lines)


def _regime_confidence_pressure(events: list[Any]) -> str:
    routing = _routing_events(events)
    lines = ["=== Regime Confidence And Threshold Pressure ==="]
    if not routing:
        lines.append("  No routing events available.")
        return "\n".join(lines)
    selected = [event for event in routing if getattr(event, "details", {}).get("selected")]
    skipped = [event for event in routing if not getattr(event, "details", {}).get("selected")]
    lines.append("  Confidence/margin describe the classifier state before execution quality is considered.")
    _event_numeric_summary(lines, selected, "Selected decisions")
    _event_numeric_summary(lines, skipped, "No-trade decisions")
    low_conf = sum(1 for event in routing if _event_num(event, "confidence") < 0.65)
    thin_margin = sum(1 for event in routing if _event_num(event, "margin") < 0.15)
    lines.append(
        f"  Strict-threshold pressure: confidence<0.65 on {low_conf}/{len(routing)} ({low_conf / len(routing):.1%}); "
        f"margin<0.15 on {thin_margin}/{len(routing)} ({thin_margin / len(routing):.1%})."
    )
    buckets: dict[str, list[Any]] = defaultdict(list)
    for event in routing:
        buckets[str(getattr(event, "details", {}).get("regime", "unknown"))].append(event)
    header = f"  {'Regime':24s} {'N':>6s} {'Sel%':>7s} {'AvgConf':>8s} {'AvgMargin':>9s}"
    lines.append(header)
    for regime, group in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0]))[:8]:
        picked = sum(1 for event in group if getattr(event, "details", {}).get("selected"))
        conf = [_event_num(event, "confidence") for event in group]
        margin = [_event_num(event, "margin") for event in group]
        lines.append(
            f"  {regime[:24]:24s} {len(group):6d} {picked / len(group):6.1%} "
            f"{_mean(conf):8.3f} {_mean(margin):9.3f}"
        )
    return "\n".join(lines)


def _candidate_opportunity_funnel(trades: list[Any], events: list[Any]) -> str:
    routing = _routing_events(events)
    lines = ["=== Candidate Opportunity Funnel ==="]
    if not routing:
        lines.append("  No routing events available.")
        return "\n".join(lines)
    if not any(_candidate_inventory(event) for event in routing):
        lines.append("  Candidate inventory details are not available in these routing events.")
        return "\n".join(lines)
    rows = {
        module: {
            "generated": 0,
            "valid": 0,
            "selected": 0,
            "blocked": 0,
            "scores": [],
            "rooms": [],
            "stops": [],
            "blocks": Counter(),
            "vetoes": Counter(),
        }
        for module in MODULE_SPECS
    }
    for event in routing:
        details = getattr(event, "details", {})
        selected_module = str(details.get("selected_module", ""))
        if details.get("selected") and selected_module in rows:
            rows[selected_module]["selected"] += 1
        for item in _candidate_inventory(event):
            module = str(item.get("module", ""))
            if module not in rows:
                continue
            rows[module]["generated"] += 1
            rows[module]["valid"] += int(bool(item.get("valid")))
            rows[module]["scores"].append(float(item.get("score", 0.0) or 0.0))
            rows[module]["rooms"].append(float(item.get("target_room_r", 0.0) or 0.0))
            rows[module]["stops"].append(float(item.get("stop_distance_points", 0.0) or 0.0))
            rows[module]["vetoes"].update(str(veto) for veto in item.get("vetoes", ()) or ())
        for item in _blocked_inventory(event):
            module = str(item.get("module", ""))
            if module not in rows:
                continue
            rows[module]["blocked"] += 1
            rows[module]["blocks"][str(item.get("block_reason", "unknown"))] += 1
    trade_counts = Counter(_module(trade) for trade in trades)
    lines.append("  Candidate counts are pre-fill opportunities; executed trades are filled/closed outcomes.")
    header = (
        f"  {'Spec':6s} {'Module':22s} {'Cand':>6s} {'Valid':>6s} {'Sel':>6s} {'Exec':>6s} "
        f"{'Sel/Cand':>9s} {'Exec/Sel':>9s} {'AvgScore':>9s} {'AvgRoom':>8s} {'Top Block / Veto'}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for module, (spec_id, label, _) in MODULE_SPECS.items():
        row = rows[module]
        generated = int(row["generated"])
        selected = int(row["selected"])
        executed = int(trade_counts.get(module, 0))
        top_block = _top_counter(row["blocks"])
        top_veto = _top_counter(row["vetoes"])
        lines.append(
            f"  {spec_id:6s} {label[:22]:22s} {generated:6d} {int(row['valid']):6d} {selected:6d} {executed:6d} "
            f"{_safe_pct(selected, generated):8.1%} {_safe_pct(executed, selected):8.1%} "
            f"{_mean(row['scores']):9.2f} {_mean(row['rooms']):8.2f} {top_block} / {top_veto}"
        )
    for module, (_, label, _) in MODULE_SPECS.items():
        row = rows[module]
        if int(row["generated"]) == 0:
            lines.append(f"  Watch: {label} produced no candidates; optimise upstream formation gates before routing or exits.")
        elif int(row["valid"]) == 0:
            lines.append(f"  Watch: {label} produced candidates but no valid candidates; top veto is {_top_counter(row['vetoes'])}.")
        elif int(row["selected"]) > 0 and _safe_pct(int(trade_counts.get(module, 0)), int(row["selected"])) < 0.25:
            lines.append(f"  Watch: {label} has low selected-to-executed conversion; inspect entry model, order TTL, and retest distance.")
    return "\n".join(lines)


def _structural_expansion_diagnostics(trades: list[Any]) -> str:
    group = [trade for trade in trades if _module(trade) == "structural_expansion"]
    lines = ["=== nq_2 Structural Expansion Diagnostics ==="]
    if not group:
        lines.append("  No structural expansion trades. Check whether IB acceptance, room-to-level, or regime margin gates are too restrictive for the sample.")
        return "\n".join(lines)
    lines.append("  Intended edge: confirmed 15m acceptance outside the 30m IB, not wick-only breakout chasing.")
    _bucket_table(lines, group, "IB type", lambda trade: _attr(trade, "ib_type") or "unknown")
    _bucket_table(lines, group, "Entry model", lambda trade: _attr(trade, "entry_model") or "unknown")
    _numeric_summary(lines, group, "Score", lambda trade: _num(trade, "setup_score"))
    _numeric_summary(lines, group, "Target room R", lambda trade: _num(trade, "target_room_r"))
    _numeric_summary(lines, group, "Stop distance pts", lambda trade: _num(trade, "stop_distance_points"))
    return "\n".join(lines)


def _liquidity_reversion_diagnostics(trades: list[Any]) -> str:
    group = [trade for trade in trades if _module(trade) == "liquidity_reversion"]
    lines = ["=== nq_3 Liquidity Reversion Diagnostics ==="]
    if not group:
        lines.append("  No liquidity reversion trades. Check sweep penetration, delayed reclaim, value-trap factor, stop-cap, and VWAP-room gates.")
        return "\n".join(lines)
    lines.append("  Intended edge: failed sweep after liquidity is taken, then reclaim/loss back toward VWAP or midpoint.")
    _bucket_table(lines, group, "Value factors", lambda trade: str(int(_num(trade, "value_factors"))))
    _bucket_table(lines, group, "Penetration bucket", lambda trade: _bucket(_num(trade, "penetration"), (4, 8, 12, 15), "pt"))
    _numeric_summary(lines, group, "VWAP room R", lambda trade: _num(trade, "vwap_room_r"))
    _numeric_summary(lines, group, "Penetration pts", lambda trade: _num(trade, "penetration"))
    _numeric_summary(lines, group, "Stop distance pts", lambda trade: _num(trade, "stop_distance_points"))
    return "\n".join(lines)


def _second_wind_diagnostics(trades: list[Any]) -> str:
    group = [trade for trade in trades if _module(trade) == "second_wind"]
    lines = ["=== nq_1 Second-Wind Continuation Diagnostics ==="]
    if not group:
        lines.append("  No second-wind trades. Check PM continuation regime confidence, squeeze duration, trigger close, and 45-point stop cap.")
        return "\n".join(lines)
    lines.append("  Intended edge: established AM trend, lunch compression, then PM squeeze release in trend direction.")
    _bucket_table(lines, group, "Squeeze duration", lambda trade: _bucket(_num(trade, "squeeze_duration"), (3, 5, 8, 10), "bars"))
    _bucket_table(lines, group, "Entry model", lambda trade: _attr(trade, "entry_model") or "unknown")
    _numeric_summary(lines, group, "Squeeze range pts", lambda trade: _num(trade, "squeeze_range"))
    _numeric_summary(lines, group, "Volume multiple", lambda trade: _num(trade, "volume_multiple"))
    _numeric_summary(lines, group, "Target room R", lambda trade: _num(trade, "target_room_r"))
    return "\n".join(lines)


def _setup_quality_breakdowns(trades: list[Any]) -> str:
    if not trades:
        return "=== Setup Quality Breakdowns ===\n  No trades."
    lines = ["=== Setup Quality Breakdowns ==="]
    _bucket_table(lines, trades, "Grade", lambda trade: _attr(trade, "grade") or "unknown")
    _bucket_table(lines, trades, "Setup type", lambda trade: _attr(trade, "setup_type") or "unknown")
    _bucket_table(lines, trades, "Entry model", lambda trade: _attr(trade, "entry_model") or "unknown")
    _bucket_table(lines, trades, "Side", lambda trade: _attr(trade, "side") or "unknown")
    _numeric_summary(lines, trades, "Setup score", lambda trade: _num(trade, "setup_score"))
    return "\n".join(lines)


def _risk_room_quality(trades: list[Any]) -> str:
    if not trades:
        return "=== Risk Room And Stop Quality ===\n  No trades."
    lines = ["=== Risk Room And Stop Quality ==="]
    lines.append("  These buckets test whether the baseline is paid for stricter room/stop filters.")
    _bucket_table(lines, trades, "Target room R", lambda trade: _bucket(_num(trade, "target_room_r"), (1.5, 2.0, 3.0, 5.0), "R"))
    _bucket_table(lines, trades, "Stop distance", lambda trade: _bucket(_num(trade, "stop_distance_points"), (8, 12, 15, 20, 45), "pt"))
    _bucket_table(lines, trades, "Initial target R", lambda trade: _bucket(_initial_target_r(trade), (0.75, 1.0, 1.5, 2.0), "R"))
    return "\n".join(lines)


def _direction_asymmetry_by_module(trades: list[Any]) -> str:
    if not trades:
        return "=== Direction Asymmetry By Module ===\n  No trades."
    lines = ["=== Direction Asymmetry By Module ==="]
    longs = [trade for trade in trades if _side_label(trade) == "BUY"]
    shorts = [trade for trade in trades if _side_label(trade) == "SELL"]
    _cohort_line(lines, "Long / BUY", longs)
    _cohort_line(lines, "Short / SELL", shorts)
    if longs and shorts:
        lines.append(f"  Directional edge (long minus short avgR): {_stats(longs)['avg_r'] - _stats(shorts)['avg_r']:+.3f}R")
    header = f"  {'Module':22s} {'L_N':>5s} {'L_WR':>6s} {'L_AvgR':>8s} {'S_N':>5s} {'S_WR':>6s} {'S_AvgR':>8s} {'Edge':>8s}"
    lines.append("\n  Module x direction:")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for module, (_, label, _) in MODULE_SPECS.items():
        group = [trade for trade in trades if _module(trade) == module]
        module_longs = [trade for trade in group if _side_label(trade) == "BUY"]
        module_shorts = [trade for trade in group if _side_label(trade) == "SELL"]
        long_stats = _stats(module_longs)
        short_stats = _stats(module_shorts)
        edge = long_stats["avg_r"] - short_stats["avg_r"]
        lines.append(
            f"  {label[:22]:22s} {len(module_longs):5d} {long_stats['wr']:5.0%} {long_stats['avg_r']:+8.3f} "
            f"{len(module_shorts):5d} {short_stats['wr']:5.0%} {short_stats['avg_r']:+8.3f} {edge:+8.3f}"
        )
    return "\n".join(lines)


def _winner_loser_entry_profile(trades: list[Any]) -> str:
    winners = [trade for trade in trades if _num(trade, "r_multiple") > 0]
    losers = [trade for trade in trades if _num(trade, "r_multiple") <= 0]
    lines = ["=== Winner / Loser Entry Profile ==="]
    if not winners or not losers:
        lines.append("  Need both winners and losers.")
        return "\n".join(lines)
    lines.append(f"  Winners: {len(winners)}; losers: {len(losers)}.")
    fields = [
        ("setup_score", lambda trade: _num(trade, "setup_score")),
        ("target_room_r", lambda trade: _num(trade, "target_room_r")),
        ("initial_target_r", _initial_target_r),
        ("stop_distance", lambda trade: _num(trade, "stop_distance_points")),
        ("vwap_room_r", lambda trade: _num(trade, "vwap_room_r")),
        ("penetration", lambda trade: _num(trade, "penetration")),
        ("value_factors", lambda trade: _num(trade, "value_factors")),
        ("mfe_r", lambda trade: _num(trade, "mfe_r")),
        ("mae_r", lambda trade: _num(trade, "mae_r")),
    ]
    header = f"  {'Metric':18s} {'WinMean':>9s} {'LossMean':>9s} {'Delta':>9s} {'Signal'}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for name, getter in fields:
        win_values = [float(getter(trade)) for trade in winners]
        loss_values = [float(getter(trade)) for trade in losers]
        if not any(abs(value) > 1e-12 for value in win_values + loss_values):
            continue
        win_mean = _mean(win_values)
        loss_mean = _mean(loss_values)
        delta = win_mean - loss_mean
        pooled = float(np.sqrt((np.var(win_values) + np.var(loss_values)) / 2.0)) if len(win_values) > 1 and len(loss_values) > 1 else 0.0
        effect = delta / pooled if pooled > 0 else 0.0
        signal = "STRONG" if abs(effect) > 0.5 else "WEAK" if abs(effect) > 0.2 else ""
        lines.append(f"  {name:18s} {win_mean:9.3f} {loss_mean:9.3f} {delta:+9.3f} {signal}")
    lines.append("\n  Winner/loser module mix:")
    for module, (_, label, _) in MODULE_SPECS.items():
        win_n = sum(1 for trade in winners if _module(trade) == module)
        loss_n = sum(1 for trade in losers if _module(trade) == module)
        total = win_n + loss_n
        if total:
            lines.append(f"    {label[:24]:24s} wins={win_n:3d} losses={loss_n:3d} WR={win_n / total:5.1%}")
    return "\n".join(lines)


def _exit_and_mfe_mae(trades: list[Any]) -> str:
    if not trades:
        return "=== Exit, MFE, MAE, Capture ===\n  No trades."
    lines = ["=== Exit, MFE, MAE, Capture ==="]
    _bucket_table(lines, trades, "Exit reason", lambda trade: _attr(trade, "exit_reason") or "unknown")
    _numeric_summary(lines, trades, "MFE R", lambda trade: _num(trade, "mfe_r"))
    _numeric_summary(lines, trades, "MAE R", lambda trade: _num(trade, "mae_r"))
    winners = [trade for trade in trades if _num(trade, "r_multiple") > 0 and _num(trade, "mfe_r") > 0]
    if winners:
        capture = [_num(trade, "r_multiple") / _num(trade, "mfe_r") for trade in winners if _num(trade, "mfe_r") > 0]
        lines.append(f"  Winner capture ratio: mean={np.mean(capture):.2f}, median={np.median(capture):.2f}, n={len(capture)}")
    losers = [trade for trade in trades if _num(trade, "r_multiple") <= 0]
    if losers:
        went_positive = sum(1 for trade in losers if _num(trade, "mfe_r") >= 0.5)
        lines.append(f"  Losers reaching at least +0.5R MFE: {went_positive}/{len(losers)} ({went_positive / len(losers):.1%})")
    return "\n".join(lines)


def _excursion_data_integrity(trades: list[Any]) -> str:
    lines = ["=== Excursion Data Integrity ==="]
    if not trades:
        lines.append("  No trades.")
        return "\n".join(lines)
    lines.append("  Validates recorded price MFE/MAE before leaning on capture and giveback reads.")
    lines.append("  Net final R is compared with commission/tolerance adjustment because MAE/MFE are price excursions.")
    tolerance = 0.05
    cohorts = [("All trades", trades)] + [
        (label[:22], [trade for trade in trades if _module(trade) == module])
        for module, (_, label, _) in MODULE_SPECS.items()
    ]
    header = f"  {'Cohort':24s} {'N':>4s} {'R>MFE':>7s} {'R<MAE':>7s} {'Tol/Comm':>8s} {'Win MFE<=0':>10s} {'Avg +Gap':>9s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    all_above = 0
    all_below = 0
    for label, group in cohorts:
        if not group:
            lines.append(f"  {label:24s} {0:4d} {0:7d} {0:7d} {0:8d} {0:10d} {0.0:+9.3f}")
            continue
        above_mfe = [
            max(0.0, _num(trade, "r_multiple") - _num(trade, "mfe_r"))
            for trade in group
            if _num(trade, "r_multiple") > _num(trade, "mfe_r") + tolerance + _commission_r(trade)
        ]
        raw_below_mae = [
            trade for trade in group if _num(trade, "r_multiple") < _num(trade, "mae_r") - tolerance
        ]
        below_mae = [
            trade for trade in group if _num(trade, "r_multiple") < _num(trade, "mae_r") - tolerance - _commission_r(trade)
        ]
        commission_or_tolerance_only = max(0, len(raw_below_mae) - len(below_mae))
        winner_zero_mfe = [
            trade for trade in group if _num(trade, "r_multiple") > 0 and _num(trade, "mfe_r") <= tolerance
        ]
        if label == "All trades":
            all_above = len(above_mfe)
            all_below = len(below_mae)
        lines.append(
            f"  {label:24s} {len(group):4d} {len(above_mfe):7d} {len(below_mae):7d} "
            f"{commission_or_tolerance_only:8d} {len(winner_zero_mfe):10d} {_mean(above_mfe):+9.3f}"
        )
    if all_above or all_below:
        lines.append(
            f"  Watch: {all_above + all_below} trades have final R outside recorded excursion bounds; "
            "inspect exit-bar excursion capture before using threshold diagnostics."
        )
    else:
        lines.append("  Excursion bounds are internally consistent within tolerance.")
    return "\n".join(lines)


def _exit_efficiency_by_module(trades: list[Any]) -> str:
    lines = ["=== Module Exit Efficiency And Edge Leakage ==="]
    if not trades:
        lines.append("  No trades.")
        return "\n".join(lines)
    lines.append("  Positive capture clips negative final R to zero; giveback is MFE minus final R.")
    header = f"  {'Spec':6s} {'Module':22s} {'N':>4s} {'AvgR':>8s} {'MFE':>7s} {'MAE':>7s} {'Cap':>7s} {'Give':>7s} {'+0.5R Losers':>13s} {'Stop%':>7s} {'TopExit'}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for module, (spec_id, label, _) in MODULE_SPECS.items():
        group = [trade for trade in trades if _module(trade) == module]
        if not group:
            lines.append(f"  {spec_id:6s} {label[:22]:22s} {0:4d} {'+0.000':>8s} {'+0.00':>7s} {'+0.00':>7s} {'0%':>7s} {'+0.00':>7s} {'0/0':>13s} {'0%':>7s} none")
            continue
        mfe_positive = [trade for trade in group if _num(trade, "mfe_r") > 0]
        capture = [
            max(_num(trade, "r_multiple"), 0.0) / _num(trade, "mfe_r")
            for trade in mfe_positive
            if _num(trade, "mfe_r") > 0
        ]
        giveback = [_num(trade, "mfe_r") - _num(trade, "r_multiple") for trade in mfe_positive]
        losers = [trade for trade in group if _num(trade, "r_multiple") <= 0]
        positive_losers = sum(1 for trade in losers if _num(trade, "mfe_r") >= 0.5)
        stopouts = sum(1 for trade in group if _attr(trade, "exit_reason") == "stop")
        stats = _stats(group)
        top_exit = Counter(_attr(trade, "exit_reason") or "unknown" for trade in group).most_common(1)[0][0]
        lines.append(
            f"  {spec_id:6s} {label[:22]:22s} {len(group):4d} {stats['avg_r']:+8.3f} {stats['avg_mfe']:+7.2f} "
            f"{stats['avg_mae']:+7.2f} {_mean(capture):6.1%} {_mean(giveback):+7.2f} "
            f"{positive_losers}/{len(losers):<9d} {_safe_pct(stopouts, len(group)):6.1%} {top_exit}"
        )
    return "\n".join(lines)


def _mfe_threshold_retention(trades: list[Any]) -> str:
    lines = ["=== MFE Threshold Retention ==="]
    if not trades:
        lines.append("  No trades.")
        return "\n".join(lines)
    lines.append("  Shows trades that reached useful open-profit thresholds but still gave back edge.")
    lines.append("  Effective MFE is max(recorded MFE, final R) to avoid exit-bar undercounts.")
    cohorts = [("All trades", trades)] + [
        (label[:22], [trade for trade in trades if _module(trade) == module])
        for module, (_, label, _) in MODULE_SPECS.items()
    ]
    header = f"  {'Trigger':>7s} {'Cohort':24s} {'Hit':>4s} {'Hit%':>6s} {'Fin<=0':>8s} {'Stop%':>7s} {'AvgR':>8s} {'Give':>7s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    leakage_rows: list[tuple[float, str, int, int, float]] = []
    for threshold in (0.5, 1.0, 2.0):
        for label, group in cohorts:
            hit = [trade for trade in group if _effective_mfe_r(trade) >= threshold]
            finished_non_positive = sum(1 for trade in hit if _num(trade, "r_multiple") <= 0)
            stopouts = sum(1 for trade in hit if _attr(trade, "exit_reason") == "stop")
            giveback = [_effective_mfe_r(trade) - _num(trade, "r_multiple") for trade in hit]
            if label != "All trades" and len(hit) >= 3:
                leakage_rows.append((threshold, label, finished_non_positive, len(hit), _mean(giveback)))
            lines.append(
                f"  {threshold:+6.1f}R {label:24s} {len(hit):4d} {_safe_pct(len(hit), len(group)):5.0%} "
                f"{finished_non_positive:3d}/{len(hit):<3d} {_safe_pct(stopouts, len(hit)):6.1%} "
                f"{_mean(_num(trade, 'r_multiple') for trade in hit):+8.3f} {_mean(giveback):+7.2f}"
            )
    if leakage_rows:
        threshold, label, leaked, hit_count, giveback = max(
            leakage_rows,
            key=lambda item: (_safe_pct(item[2], item[3]), item[4], item[3]),
        )
        lines.append(
            f"  Largest leakage read: {label} after +{threshold:.1f}R MFE "
            f"finished <=0R on {leaked}/{hit_count} hits with avg giveback {giveback:+.2f}R."
        )
    return "\n".join(lines)


def _profit_floor_sensitivity(trades: list[Any]) -> str:
    lines = ["=== Profit Floor Sensitivity From Recorded MFE ==="]
    if not trades:
        lines.append("  No trades.")
        return "\n".join(lines)
    lines.append("  Idealized floor scenarios use recorded/effective MFE only; replay is still required before adopting a rule.")
    base_total = sum(_num(trade, "r_multiple") for trade in trades)
    scenarios = [
        (0.50, 0.00),
        (0.50, 0.25),
        (0.75, 0.25),
        (1.00, 0.25),
        (1.00, 0.50),
        (2.00, 1.00),
    ]
    header = f"  {'Trigger':>7s} {'Lock':>7s} {'Touched':>7s} {'Raised':>6s} {'LossCuts':>8s} {'AvgR':>8s} {'TotR':>8s} {'DeltaR':>8s} {'Top Lift'}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for trigger, lock in scenarios:
        adjusted: list[float] = []
        raised = 0
        loss_cuts = 0
        deltas_by_module: defaultdict[str, float] = defaultdict(float)
        touched = 0
        for trade in trades:
            current = _num(trade, "r_multiple")
            updated = current
            if _effective_mfe_r(trade) >= trigger:
                touched += 1
                updated = max(current, lock)
            delta = updated - current
            if delta > 1e-9:
                raised += 1
                deltas_by_module[_module(trade)] += delta
                if current <= 0:
                    loss_cuts += 1
            adjusted.append(updated)
        total = sum(adjusted)
        top_lift = _format_top_module_delta(deltas_by_module)
        lines.append(
            f"  {trigger:+6.2f}R {lock:+6.2f}R {touched:7d} {raised:6d} {loss_cuts:8d} "
            f"{_safe_pct(total, len(trades)):+8.3f} {total:+8.2f} {total - base_total:+8.2f} {top_lift}"
        )
    return "\n".join(lines)


def _hold_duration_analysis(trades: list[Any]) -> str:
    timed = [(trade, minutes) for trade in trades if (minutes := _hold_minutes(trade)) is not None]
    lines = ["=== Hold Duration Efficiency ==="]
    if not timed:
        lines.append("  No completed timestamped trades.")
        return "\n".join(lines)
    durations = np.asarray([minutes for _, minutes in timed], dtype=float)
    rs = np.asarray([_num(trade, "r_multiple") for trade, _ in timed], dtype=float)
    corr = 0.0
    if len(durations) >= 3 and float(np.std(durations)) > 0 and float(np.std(rs)) > 0:
        corr = float(np.corrcoef(durations, rs)[0, 1])
        if np.isnan(corr):
            corr = 0.0
    lines.append(
        f"  Hold minutes: mean={np.mean(durations):.1f}, median={np.median(durations):.1f}, "
        f"p75={np.percentile(durations, 75):.1f}, p90={np.percentile(durations, 90):.1f}, R-corr={corr:+.2f}"
    )
    buckets = [
        ("<= 15 min", lambda value: value <= 15),
        ("<= 30 min", lambda value: 15 < value <= 30),
        ("<= 60 min", lambda value: 30 < value <= 60),
        ("<= 120 min", lambda value: 60 < value <= 120),
        ("> 120 min", lambda value: value > 120),
    ]
    groups: list[tuple[str, list[Any]]] = []
    for label, predicate in buckets:
        group = [trade for trade, minutes in timed if predicate(minutes)]
        if group:
            groups.append((label, group))
    header = f"  {'Bucket':12s} {'N':>4s} {'WR':>6s} {'PF':>6s} {'AvgR':>8s} {'MFE':>7s} {'MAE':>7s} {'PnL':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for label, group in groups:
        stats = _stats(group)
        pnl = sum(_num(trade, "pnl_dollars") for trade in group)
        lines.append(
            f"  {label:12s} {len(group):4d} {stats['wr']:5.0%} {stats['pf']:6.2f} "
            f"{stats['avg_r']:+8.3f} {stats['avg_mfe']:+7.2f} {stats['avg_mae']:+7.2f} {pnl:+10.2f}"
        )
    eligible = [(label, group, _stats(group)) for label, group in groups if len(group) >= 3]
    if eligible:
        best = max(eligible, key=lambda item: item[2]["avg_r"])
        worst = min(eligible, key=lambda item: item[2]["avg_r"])
        lines.append(
            f"  Best sampled hold bucket: {best[0]} avgR={best[2]['avg_r']:+.3f}; "
            f"weakest: {worst[0]} avgR={worst[2]['avg_r']:+.3f}."
        )
    return "\n".join(lines)


def _time_breakdowns(trades: list[Any]) -> str:
    if not trades:
        return "=== Time Breakdowns ===\n  No trades."
    lines = ["=== Time Breakdowns ==="]
    _bucket_table(lines, trades, "Entry hour ET/UTC stored", lambda trade: str(_entry_time(trade).hour) if _entry_time(trade) else "unknown")
    _bucket_table(lines, trades, "Day of week", lambda trade: _entry_time(trade).strftime("%a") if _entry_time(trade) else "unknown")
    _bucket_table(lines, trades, "Month", lambda trade: _entry_time(trade).strftime("%Y-%m") if _entry_time(trade) else "unknown", limit=12)
    return "\n".join(lines)


def _monthly_pnl(trades: list[Any]) -> str:
    dated = _dated_trades(trades)
    lines = ["=== Monthly P&L Consistency ==="]
    if not dated:
        lines.append("  No timestamped trades.")
        return "\n".join(lines)
    by_month: dict[str, list[Any]] = defaultdict(list)
    for trade in dated:
        entry = _entry_time(trade)
        if entry is not None:
            by_month[entry.strftime("%Y-%m")].append(trade)
    cum_pnl = 0.0
    monthly_rows: list[tuple[str, list[Any], float]] = []
    header = f"  {'Month':8s} {'N':>4s} {'WR':>6s} {'AvgR':>8s} {'NetPnL':>10s} {'CumPnL':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for month in sorted(by_month):
        group = by_month[month]
        stats = _stats(group)
        month_pnl = sum(_num(trade, "pnl_dollars") for trade in group)
        cum_pnl += month_pnl
        monthly_rows.append((month, group, month_pnl))
        lines.append(
            f"  {month:8s} {len(group):4d} {stats['wr']:5.0%} {stats['avg_r']:+8.3f} {month_pnl:+10.2f} {cum_pnl:+10.2f}"
        )
    winning = sum(1 for _, _, pnl in monthly_rows if pnl > 0)
    best = max(monthly_rows, key=lambda item: item[2])
    worst = min(monthly_rows, key=lambda item: item[2])
    lines.append(
        f"  Winning months: {winning}/{len(monthly_rows)} ({winning / len(monthly_rows):.1%}); "
        f"best={best[0]} ${best[2]:+.2f}; worst={worst[0]} ${worst[2]:+.2f}."
    )
    return "\n".join(lines)


def _year_module_stability(trades: list[Any]) -> str:
    dated = _dated_trades(trades)
    lines = ["=== Year And Module Stability ==="]
    if not dated:
        lines.append("  No timestamped trades.")
        return "\n".join(lines)
    by_year: dict[str, list[Any]] = defaultdict(list)
    for trade in dated:
        entry = _entry_time(trade)
        if entry is not None:
            by_year[entry.strftime("%Y")].append(trade)
    header = f"  {'Year':6s} {'N':>4s} {'WR':>6s} {'PF':>6s} {'AvgR':>8s} {'TotR':>8s} {'Best module':>24s} {'Weak module':>24s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for year in sorted(by_year):
        group = by_year[year]
        stats = _stats(group)
        module_stats = [
            (label, module_group, _stats(module_group))
            for module, (_, label, _) in MODULE_SPECS.items()
            if (module_group := [trade for trade in group if _module(trade) == module])
        ]
        best = max(module_stats, key=lambda item: item[2]["avg_r"]) if module_stats else None
        weak = min(module_stats, key=lambda item: item[2]["avg_r"]) if module_stats else None
        best_text = f"{best[0][:14]} {len(best[1])}/{best[2]['avg_r']:+.2f}R" if best else "none"
        weak_text = f"{weak[0][:14]} {len(weak[1])}/{weak[2]['avg_r']:+.2f}R" if weak else "none"
        lines.append(
            f"  {year:6s} {len(group):4d} {stats['wr']:5.0%} {stats['pf']:6.2f} "
            f"{stats['avg_r']:+8.3f} {stats['total_r']:+8.2f} {best_text:>24s} {weak_text:>24s}"
        )
    lines.append("\n  Year x module cells show N/avgR:")
    cell_header = f"  {'Year':6s} {'Structural':>18s} {'Liquidity':>18s} {'SecondWind':>18s}"
    lines.append(cell_header)
    lines.append("  " + "-" * (len(cell_header) - 2))
    for year in sorted(by_year):
        group = by_year[year]
        cells = []
        for module in ("structural_expansion", "liquidity_reversion", "second_wind"):
            module_group = [trade for trade in group if _module(trade) == module]
            cells.append(f"{len(module_group)}/{_stats(module_group)['avg_r']:+.2f}R" if module_group else "-")
        lines.append(f"  {year:6s} {cells[0]:>18s} {cells[1]:>18s} {cells[2]:>18s}")
    positive_years = sum(1 for group in by_year.values() if sum(_num(trade, "r_multiple") for trade in group) > 0)
    lines.append(f"  Positive years: {positive_years}/{len(by_year)} ({positive_years / len(by_year):.1%}).")
    return "\n".join(lines)


def _trade_gap_analysis(trades: list[Any]) -> str:
    dated = _dated_trades(trades)
    lines = ["=== Trade Gap And Opportunity Density ==="]
    if len(dated) < 2:
        lines.append("  Need at least two timestamped trades.")
        return "\n".join(lines)
    gaps: list[tuple[float, Any, Any]] = []
    for prior, current in zip(dated, dated[1:]):
        prior_anchor = _exit_time(prior) or _entry_time(prior)
        current_anchor = _entry_time(current)
        if prior_anchor is None or current_anchor is None:
            continue
        gap_hours = max((current_anchor - prior_anchor).total_seconds() / 3600.0, 0.0)
        gaps.append((gap_hours, prior, current))
    if not gaps:
        lines.append("  No usable trade gaps.")
        return "\n".join(lines)
    values = np.asarray([gap for gap, _, _ in gaps], dtype=float)
    lines.append(
        f"  Gap hours: mean={np.mean(values):.1f}, median={np.median(values):.1f}, "
        f"p75={np.percentile(values, 75):.1f}, p90={np.percentile(values, 90):.1f}, max={np.max(values):.1f}"
    )
    dry_spells = sorted((item for item in gaps if item[0] >= 24 * 5), key=lambda item: item[0], reverse=True)[:5]
    if dry_spells:
        lines.append("  Largest dry spells:")
        for gap_hours, prior, current in dry_spells:
            prior_dt = _exit_time(prior) or _entry_time(prior)
            current_dt = _entry_time(current)
            lines.append(
                f"    {gap_hours / 24.0:5.1f} days from {_date_text(prior_dt)} to {_date_text(current_dt)}; "
                f"next={_module(current)} { _num(current, 'r_multiple'):+.2f}R"
            )
    by_week = Counter(_week_key(_entry_time(trade)) for trade in dated if _entry_time(trade) is not None)
    week_keys = sorted(key for key in by_week if key)
    if week_keys:
        start = datetime.fromisoformat(week_keys[0])
        end = datetime.fromisoformat(week_keys[-1])
        all_weeks: list[str] = []
        cursor = start
        while cursor <= end:
            all_weeks.append(cursor.date().isoformat())
            cursor += timedelta(days=7)
        counts = [by_week.get(key, 0) for key in all_weeks]
        zero_weeks = sum(1 for count in counts if count == 0)
        longest_zero = _longest_zero_streak(counts)
        lines.append(
            f"  Weekly density: active_weeks={sum(1 for count in counts if count > 0)}/{len(counts)}, "
            f"zero_weeks={zero_weeks}, longest_zero_week_run={longest_zero}, "
            f"mean_trades_per_active_week={_safe_pct(len(dated), max(sum(1 for count in counts if count > 0), 1)):.2f}"
        )
    return "\n".join(lines)


def _daily_sequence_diagnostics(trades: list[Any]) -> str:
    timed = [trade for trade in trades if _entry_time(trade) is not None]
    lines = ["=== Daily Sequencing And After-Loss Behavior ==="]
    if not timed:
        lines.append("  No timestamped trades.")
        return "\n".join(lines)
    by_day: dict[str, list[Any]] = defaultdict(list)
    for trade in sorted(timed, key=lambda item: _entry_time(item) or datetime.min):
        entry = _entry_time(trade)
        if entry is not None:
            by_day[entry.date().isoformat()].append(trade)
    first_trades: list[Any] = []
    second_trades: list[Any] = []
    after_win: list[Any] = []
    after_loss: list[Any] = []
    for day_trades in by_day.values():
        day_trades.sort(key=lambda item: _entry_time(item) or datetime.min)
        if day_trades:
            first_trades.append(day_trades[0])
        if len(day_trades) >= 2:
            second_trades.append(day_trades[1])
        for idx in range(1, len(day_trades)):
            if _num(day_trades[idx - 1], "r_multiple") > 0:
                after_win.append(day_trades[idx])
            else:
                after_loss.append(day_trades[idx])
    days_with_second = sum(1 for items in by_day.values() if len(items) >= 2)
    lines.append(f"  Trading days: {len(by_day)}; days with a second trade: {days_with_second} ({days_with_second / len(by_day):.1%}).")
    _cohort_line(lines, "First trade of day", first_trades)
    _cohort_line(lines, "Second trade of day", second_trades)
    _cohort_line(lines, "Trade after prior win", after_win)
    _cohort_line(lines, "Trade after prior loss", after_loss)
    return "\n".join(lines)


def _trade_autocorrelation(trades: list[Any]) -> str:
    dated = _dated_trades(trades)
    lines = ["=== Trade Autocorrelation And Loss Clustering ==="]
    if len(dated) < 3:
        lines.append("  Need at least three timestamped trades.")
        return "\n".join(lines)
    rs = np.asarray([_num(trade, "r_multiple") for trade in dated], dtype=float)
    lag_corr = 0.0
    if len(rs) >= 3 and float(np.std(rs[:-1])) > 0 and float(np.std(rs[1:])) > 0:
        lag_corr = float(np.corrcoef(rs[:-1], rs[1:])[0, 1])
        if np.isnan(lag_corr):
            lag_corr = 0.0
    lines.append(f"  Lag-1 R autocorrelation: {lag_corr:+.2f}")
    transitions = Counter()
    for prior, current in zip(rs, rs[1:]):
        transitions[("W" if prior > 0 else "L", "W" if current > 0 else "L")] += 1
    after_win = transitions[("W", "W")] + transitions[("W", "L")]
    after_loss = transitions[("L", "W")] + transitions[("L", "L")]
    lines.append(
        f"  After win: next win={_safe_pct(transitions[('W', 'W')], after_win):.1%} "
        f"({transitions[('W', 'W')]}/{after_win}); "
        f"after loss: next win={_safe_pct(transitions[('L', 'W')], after_loss):.1%} "
        f"({transitions[('L', 'W')]}/{after_loss})."
    )
    after_streak: dict[int, list[Any]] = defaultdict(list)
    loss_run = 0
    for trade in dated:
        if loss_run in (1, 2, 3):
            after_streak[loss_run].append(trade)
        if _num(trade, "r_multiple") > 0:
            loss_run = 0
        else:
            loss_run += 1
            if loss_run > 3:
                loss_run = 3
    for run_length in (1, 2, 3):
        _cohort_line(lines, f"After {run_length} prior losses", after_streak.get(run_length, []))
    return "\n".join(lines)


def _rolling_expectancy(trades: list[Any], window: int = 20) -> str:
    dated = _dated_trades(trades)
    lines = [f"=== Rolling Expectancy ({window}-Trade Windows) ==="]
    if len(dated) < window:
        lines.append(f"  Need at least {window} timestamped trades; have {len(dated)}.")
        return "\n".join(lines)
    windows: list[tuple[int, int, float, float]] = []
    rs = [_num(trade, "r_multiple") for trade in dated]
    pnls = [_num(trade, "pnl_dollars") for trade in dated]
    for start in range(0, len(dated) - window + 1):
        end = start + window
        windows.append((start, end - 1, float(np.mean(rs[start:end])), float(sum(pnls[start:end]))))
    averages = np.asarray([item[2] for item in windows], dtype=float)
    negative = int(np.sum(averages < 0))
    first_half = _mean(rs[: len(rs) // 2])
    second_half = _mean(rs[len(rs) // 2 :])
    current = windows[-1]
    best = max(windows, key=lambda item: item[2])
    worst = min(windows, key=lambda item: item[2])
    lines.append(
        f"  Rolling avgR: min={np.min(averages):+.3f}, max={np.max(averages):+.3f}, "
        f"current={current[2]:+.3f}; negative_windows={negative}/{len(windows)} ({negative / len(windows):.1%})."
    )
    lines.append(f"  Sample drift: first_half_avgR={first_half:+.3f}, second_half_avgR={second_half:+.3f}.")
    lines.append(
        f"  Best window: {_date_text(_entry_time(dated[best[0]]))} to {_date_text(_entry_time(dated[best[1]]))}, "
        f"avgR={best[2]:+.3f}, pnl=${best[3]:+.2f}."
    )
    lines.append(
        f"  Worst window: {_date_text(_entry_time(dated[worst[0]]))} to {_date_text(_entry_time(dated[worst[1]]))}, "
        f"avgR={worst[2]:+.3f}, pnl=${worst[3]:+.2f}."
    )
    return "\n".join(lines)


def _r_distribution(trades: list[Any]) -> str:
    if not trades:
        return "=== R Distribution And Streaks ===\n  No trades."
    rs = np.asarray([_num(trade, "r_multiple") for trade in trades], dtype=float)
    lines = ["=== R Distribution And Streaks ==="]
    lines.append(f"  R: mean={np.mean(rs):+.3f}, median={np.median(rs):+.3f}, p25={np.percentile(rs, 25):+.3f}, p75={np.percentile(rs, 75):+.3f}, p90={np.percentile(rs, 90):+.3f}")
    for threshold in (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
        count = int(np.sum(rs >= threshold))
        lines.append(f"  >= {threshold:+.1f}R: {count:4d} ({count / len(rs):5.1%})")
    max_win, max_loss = _streaks(rs)
    lines.append(f"  Max win streak: {max_win}; max loss streak: {max_loss}.")
    return "\n".join(lines)


def _drawdown_profile(equity_curve: Iterable[float] | None, timestamps: Iterable[Any] | None) -> str:
    if equity_curve is None:
        return "=== Drawdown Profile ===\n  No equity curve data available."
    equity = np.asarray(list(equity_curve), dtype=float)
    if len(equity) == 0:
        return "=== Drawdown Profile ===\n  No equity curve data available."
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / np.maximum(peaks, 1.0)
    trough = int(np.argmin(dd))
    ts_list = list(timestamps) if timestamps is not None else []
    trough_ts = ts_list[trough] if trough < len(ts_list) else "n/a"
    underwater = int(np.sum(dd < 0))
    return "\n".join(
        [
            "=== Drawdown Profile ===",
            f"  Max DD: {float(abs(np.min(dd))):.2%} at {trough_ts}",
            f"  Bars underwater: {underwater}/{len(equity)} ({underwater / len(equity):.1%})",
            f"  Ending equity: ${float(equity[-1]):,.2f}",
        ]
    )


def _closed_trade_drawdown_anatomy(trades: list[Any]) -> str:
    dated = sorted(_dated_trades(trades), key=lambda trade: _exit_time(trade) or _entry_time(trade) or datetime.min)
    lines = ["=== Closed-Trade Drawdown Anatomy ==="]
    if not dated:
        lines.append("  No timestamped trades.")
        return "\n".join(lines)
    episodes: list[dict[str, Any]] = []
    peak = 0.0
    equity = 0.0
    episode: dict[str, Any] | None = None
    for idx, trade in enumerate(dated):
        equity += _num(trade, "pnl_dollars")
        if equity >= peak:
            if episode is not None:
                episode["end_idx"] = idx
                episode["end_equity"] = equity
                episodes.append(episode)
                episode = None
            peak = equity
            continue
        dd = peak - equity
        if episode is None:
            episode = {
                "start_idx": idx,
                "trough_idx": idx,
                "peak": peak,
                "max_dd": dd,
            }
        elif dd > float(episode["max_dd"]):
            episode["trough_idx"] = idx
            episode["max_dd"] = dd
    if episode is not None:
        episode["end_idx"] = len(dated) - 1
        episode["end_equity"] = equity
        episodes.append(episode)
    if not episodes:
        lines.append("  Closed-trade equity never fell below a prior peak.")
        return "\n".join(lines)
    top = sorted(episodes, key=lambda item: float(item["max_dd"]), reverse=True)[:3]
    header = f"  {'Rank':>4s} {'Start':10s} {'Trough':10s} {'End':10s} {'Trades':>6s} {'DD$':>10s} {'NetR':>8s} {'Worst Trade'}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for rank, item in enumerate(top, start=1):
        start = int(item["start_idx"])
        trough = int(item["trough_idx"])
        end = int(item["end_idx"])
        episode_trades = dated[start : end + 1]
        dd_trades = dated[start : trough + 1]
        worst = min(dd_trades, key=lambda trade: _num(trade, "pnl_dollars"))
        net_r_drawdown = max(0.0, -sum(_num(trade, "r_multiple") for trade in dd_trades))
        lines.append(
            f"  {rank:4d} {_date_text(_entry_time(dated[start])):10s} {_date_text(_entry_time(dated[trough])):10s} "
            f"{_date_text(_entry_time(dated[end])):10s} {len(episode_trades):6d} "
            f"{float(item['max_dd']):10.2f} {net_r_drawdown:8.2f} {_module(worst)} {_num(worst, 'r_multiple'):+.2f}R"
        )
        mix = Counter(_module(trade) for trade in dd_trades)
        lines.append(f"       Module mix to trough: {', '.join(f'{key}={value}' for key, value in sorted(mix.items()))}")
    return "\n".join(lines)


def _accounting_reconciliation(
    trades: list[Any],
    metrics: dict[str, float],
    equity_curve: Iterable[float] | None,
) -> str:
    lines = ["=== Accounting Reconciliation ==="]
    trade_pnl = sum(_num(trade, "pnl_dollars") for trade in trades)
    metric_pnl = float(metrics.get("net_profit", 0.0) or 0.0)
    commission = sum(_num(trade, "commission") for trade in trades)
    lines.append(f"  Trade ledger net: ${trade_pnl:+.2f}; metrics net: ${metric_pnl:+.2f}; gap=${trade_pnl - metric_pnl:+.4f}.")
    if equity_curve is None:
        lines.append("  No equity curve supplied for equity-to-metrics reconciliation.")
    else:
        equity = np.asarray(list(equity_curve), dtype=float)
        if len(equity):
            inferred_initial = float(equity[-1] - metric_pnl)
            equity_delta = float(equity[-1] - inferred_initial)
            lines.append(
                f"  Equity final: ${float(equity[-1]):,.2f}; inferred initial=${inferred_initial:,.2f}; "
                f"equity_delta=${equity_delta:+.2f}; equity-metric gap=${equity_delta - metric_pnl:+.4f}."
            )
        else:
            lines.append("  Empty equity curve supplied.")
    lines.append(f"  Total recorded commission: ${commission:.2f}; avg/trade=${_safe_pct(commission, len(trades)):.2f}.")
    return "\n".join(lines)


def _baseline_readiness(metrics: dict[str, float]) -> str:
    checks = [
        ("minimum trade sample", metrics.get("total_trades", 0.0) >= 30),
        ("positive expectancy", metrics.get("avg_r", 0.0) > 0.0),
        ("PF above 1.10", metrics.get("profit_factor", 0.0) >= 1.10),
        ("DD below 25%", metrics.get("max_drawdown_pct", 0.0) <= 0.25),
        ("at least two modules active", metrics.get("module_coverage", 0.0) >= 2 / 3),
    ]
    lines = ["=== Baseline Readiness For Phased Auto-Optimization ==="]
    for label, passed in checks:
        lines.append(f"  {'PASS' if passed else 'WATCH'}  {label}")
    lines.append("  Interpretation: phase 1 should isolate module edges; later phases should improve routing, quality gates, risk, exits, and final frequency without diluting component expectancy.")
    return "\n".join(lines)


def _bucket_table(lines: list[str], trades: list[Any], label: str, key_fn, *, limit: int = 10) -> None:
    buckets: dict[str, list[Any]] = defaultdict(list)
    for trade in trades:
        buckets[str(key_fn(trade) or "unknown")].append(trade)
    lines.append(f"\n  {label}:")
    header = f"    {'Bucket':28s} {'N':>4s} {'WR':>6s} {'PF':>6s} {'AvgR':>8s} {'TotR':>8s}"
    lines.append(header)
    for key, group in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0]))[:limit]:
        stats = _stats(group)
        lines.append(f"    {key[:28]:28s} {len(group):4d} {stats['wr']:5.0%} {stats['pf']:6.2f} {stats['avg_r']:+8.3f} {stats['total_r']:+8.2f}")


def _cohort_line(lines: list[str], label: str, trades: list[Any]) -> None:
    stats = _stats(trades)
    lines.append(
        f"  {label:24s} N={len(trades):3d} WR={stats['wr']:5.0%} PF={stats['pf']:5.2f} "
        f"AvgR={stats['avg_r']:+7.3f} TotR={stats['total_r']:+7.2f}"
    )


def _numeric_summary(lines: list[str], trades: list[Any], label: str, value_fn) -> None:
    values = np.asarray([float(value_fn(trade)) for trade in trades if value_fn(trade) is not None], dtype=float)
    if len(values) == 0:
        return
    lines.append(f"  {label}: mean={np.mean(values):.3f}, median={np.median(values):.3f}, p25={np.percentile(values, 25):.3f}, p75={np.percentile(values, 75):.3f}")


def _stats(trades: list[Any]) -> dict[str, float]:
    if not trades:
        return {"wr": 0.0, "pf": 0.0, "avg_r": 0.0, "total_r": 0.0, "avg_mfe": 0.0, "avg_mae": 0.0}
    pnl = [_num(trade, "pnl_dollars") for trade in trades]
    rs = [_num(trade, "r_multiple") for trade in trades]
    gross_win = sum(value for value in pnl if value > 0)
    gross_loss = abs(sum(value for value in pnl if value < 0))
    return {
        "wr": sum(1 for value in pnl if value > 0) / len(trades),
        "pf": gross_win / gross_loss if gross_loss > 0 else (10.0 if gross_win > 0 else 0.0),
        "avg_r": float(np.mean(rs)) if rs else 0.0,
        "total_r": float(sum(rs)),
        "avg_mfe": float(np.mean([_num(trade, "mfe_r") for trade in trades])),
        "avg_mae": float(np.mean([_num(trade, "mae_r") for trade in trades])),
    }


def _edge_verdict(group: list[Any], stats: dict[str, float]) -> str:
    if not group:
        return "NO_SAMPLE"
    if len(group) < 10:
        return "LOW_SAMPLE"
    if stats["avg_r"] > 0.15 and stats["pf"] >= 1.25:
        return "EDGE_PRESENT"
    if stats["avg_mfe"] > 1.0 and stats["avg_r"] <= 0:
        return "ENTRY_OK_EXIT_WEAK"
    return "NEEDS_OPT"


def _streaks(rs: np.ndarray) -> tuple[int, int]:
    max_win = max_loss = win = loss = 0
    for value in rs:
        if value > 0:
            win += 1
            loss = 0
            max_win = max(max_win, win)
        else:
            loss += 1
            win = 0
            max_loss = max(max_loss, loss)
    return max_win, max_loss


def _module(trade: Any) -> str:
    return str(getattr(trade, "module", "") or "unknown")


def _attr(trade: Any, name: str) -> str:
    return str(getattr(trade, name, "") or "")


def _num(trade: Any, name: str) -> float:
    value = getattr(trade, name, 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _commission_r(trade: Any) -> float:
    risk_points = abs(_num(trade, "entry_price") - _num(trade, "initial_stop"))
    qty = max(1.0, _num(trade, "qty"))
    commission = abs(_num(trade, "commission"))
    if risk_points <= 0 or commission <= 0:
        return 0.0
    try:
        point_value = spec_for(_attr(trade, "symbol") or "MNQ").point_value
    except ValueError:
        return 0.0
    risk_dollars = risk_points * qty * point_value
    return commission / risk_dollars if risk_dollars > 0 else 0.0


def _entry_time(trade: Any) -> datetime | None:
    value = getattr(trade, "entry_time", None)
    return value if isinstance(value, datetime) else None


def _exit_time(trade: Any) -> datetime | None:
    value = getattr(trade, "exit_time", None)
    return value if isinstance(value, datetime) else None


def _hold_minutes(trade: Any) -> float | None:
    entry = _entry_time(trade)
    exit_time = _exit_time(trade)
    if entry is None or exit_time is None:
        return None
    return max((exit_time - entry).total_seconds() / 60.0, 0.0)


def _side_label(trade: Any) -> str:
    raw = _attr(trade, "side").upper()
    if raw in {"BUY", "LONG"} or raw.endswith(".LONG"):
        return "BUY"
    if raw in {"SELL", "SHORT"} or raw.endswith(".SHORT"):
        return "SELL"
    return raw or "unknown"


def _dated_trades(trades: list[Any]) -> list[Any]:
    return sorted(
        [trade for trade in trades if _entry_time(trade) is not None],
        key=lambda trade: _entry_time(trade) or datetime.min,
    )


def _date_text(value: datetime | None) -> str:
    return value.date().isoformat() if isinstance(value, datetime) else "n/a"


def _week_key(value: datetime | None) -> str:
    if value is None:
        return ""
    return (value - timedelta(days=value.weekday())).date().isoformat()


def _longest_zero_streak(values: list[int]) -> int:
    longest = current = 0
    for value in values:
        if value == 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _bucket(value: float, thresholds: tuple[float, ...], suffix: str) -> str:
    for threshold in thresholds:
        if value <= threshold:
            return f"<= {threshold:g} {suffix}"
    return f"> {thresholds[-1]:g} {suffix}"


def _routing_events(events: list[Any]) -> list[Any]:
    return [event for event in events if getattr(event, "code", "") == "ROUTING_DECISION"]


def _candidate_inventory(event: Any) -> list[dict[str, Any]]:
    inventory = getattr(event, "details", {}).get("candidate_inventory", [])
    return [dict(item) for item in inventory if isinstance(item, dict)]


def _blocked_inventory(event: Any) -> list[dict[str, Any]]:
    blocked = getattr(event, "details", {}).get("blocked_candidates", [])
    return [dict(item) for item in blocked if isinstance(item, dict)]


def _event_num(event: Any, key: str) -> float:
    value = getattr(event, "details", {}).get(key, 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _event_numeric_summary(lines: list[str], events: list[Any], label: str) -> None:
    if not events:
        lines.append(f"  {label}: n=0")
        return
    confidences = [_event_num(event, "confidence") for event in events]
    margins = [_event_num(event, "margin") for event in events]
    lines.append(
        f"  {label}: n={len(events)}, avg_conf={_mean(confidences):.3f}, "
        f"p25_margin={float(np.percentile(margins, 25)):.3f}, avg_margin={_mean(margins):.3f}"
    )


def _safe_pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return float(np.mean(items)) if items else 0.0


def _top_counter(counter: Counter) -> str:
    if not counter:
        return "none"
    key, count = counter.most_common(1)[0]
    return f"{key}:{count}"


def _effective_mfe_r(trade: Any) -> float:
    return max(_num(trade, "mfe_r"), _num(trade, "r_multiple"), 0.0)


def _format_top_module_delta(deltas: dict[str, float]) -> str:
    if not deltas:
        return "none"
    module, delta = max(deltas.items(), key=lambda item: item[1])
    _, label, _ = MODULE_SPECS.get(module, (module, module, ""))
    return f"{label[:18]} {delta:+.2f}R"


def _yearly_strength_summary(trades: list[Any]) -> str:
    dated = _dated_trades(trades)
    by_year: dict[str, list[Any]] = defaultdict(list)
    for trade in dated:
        entry = _entry_time(trade)
        if entry is not None:
            by_year[entry.strftime("%Y")].append(trade)
    if len(by_year) < 2:
        return ""
    positive = sum(1 for group in by_year.values() if sum(_num(trade, "r_multiple") for trade in group) > 0)
    return f"Positive sampled years {positive}/{len(by_year)}"


def _mfe_leakage_summary(trades: list[Any]) -> str:
    rows: list[tuple[float, str, int, int, float]] = []
    for threshold in (0.5, 1.0, 2.0):
        for module, (_, label, _) in MODULE_SPECS.items():
            group = [trade for trade in trades if _module(trade) == module]
            hit = [trade for trade in group if _effective_mfe_r(trade) >= threshold]
            if len(hit) < 3:
                continue
            leaked = sum(1 for trade in hit if _num(trade, "r_multiple") <= 0)
            giveback = _mean(_effective_mfe_r(trade) - _num(trade, "r_multiple") for trade in hit)
            rows.append((threshold, label, leaked, len(hit), giveback))
    if not rows:
        return ""
    threshold, label, leaked, hit_count, giveback = max(
        rows,
        key=lambda item: (_safe_pct(item[2], item[3]), item[4], item[3]),
    )
    if leaked == 0:
        return ""
    return (
        f"{label} leakage after +{threshold:.1f}R: "
        f"{leaked}/{hit_count} finish <=0R, avg giveback {giveback:+.2f}R"
    )


def _top_candidate_bottleneck(events: list[Any]) -> str:
    rows = {
        module: {
            "generated": 0,
            "valid": 0,
            "selected": 0,
            "vetoes": Counter(),
        }
        for module in MODULE_SPECS
    }
    for event in _routing_events(events):
        details = getattr(event, "details", {})
        selected_module = str(details.get("selected_module", ""))
        if details.get("selected") and selected_module in rows:
            rows[selected_module]["selected"] += 1
        for item in _candidate_inventory(event):
            module = str(item.get("module", ""))
            if module not in rows:
                continue
            rows[module]["generated"] += 1
            rows[module]["valid"] += int(bool(item.get("valid")))
            rows[module]["vetoes"].update(str(veto) for veto in item.get("vetoes", ()) or ())
    zero_valid = [
        (module, data)
        for module, data in rows.items()
        if int(data["generated"]) > 0 and int(data["valid"]) == 0
    ]
    if zero_valid:
        module, data = max(zero_valid, key=lambda item: int(item[1]["generated"]))
        _, label, _ = MODULE_SPECS[module]
        return f"{label} generated {int(data['generated'])} candidates but 0 valid; top veto {_top_counter(data['vetoes'])}"
    generated = [(module, data) for module, data in rows.items() if int(data["generated"]) > 0]
    if not generated:
        return ""
    module, data = min(generated, key=lambda item: _safe_pct(int(item[1]["selected"]), int(item[1]["generated"])))
    selected_rate = _safe_pct(int(data["selected"]), int(data["generated"]))
    if selected_rate < 0.05:
        _, label, _ = MODULE_SPECS[module]
        return f"{label} selected only {int(data['selected'])}/{int(data['generated'])} candidates ({selected_rate:.1%})"
    return ""


def _initial_target_r(trade: Any) -> float:
    risk = abs(_num(trade, "entry_price") - _num(trade, "initial_stop"))
    if risk <= 0:
        return 0.0
    return abs(_num(trade, "initial_target") - _num(trade, "entry_price")) / risk
