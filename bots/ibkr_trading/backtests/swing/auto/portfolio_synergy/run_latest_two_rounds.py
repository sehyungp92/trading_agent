"""Rerun two swing portfolio synergy optimisation passes from latest strategy configs.

The current swing family is ATRSS + AKC Helix + TPC + overlay.  This runner
starts from each strategy's latest optimized artifact, applies the current live
portfolio stance as the seed, runs a broad greedy pass, then runs a refinement
greedy pass from the broad winner. Candidate scoring uses the same static
initial strategy-risk return basis that is reported in the round artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from backtests.swing.auto.config_mutator import mutate_unified_config
from backtests.swing.auto.greedy_optimize import run_greedy, save_result
from backtests.swing.config_unified import UnifiedBacktestConfig
from backtests.swing.engine.unified_portfolio_engine import load_unified_data, run_unified


LATEST_CONFIGS = {
    "ATRSS": ROOT / "backtests" / "output" / "swing" / "atrss" / "round_4" / "optimized_config.json",
    "AKC_HELIX": ROOT / "backtests" / "output" / "swing" / "helix" / "round_5" / "optimized_config.json",
    "TPC": ROOT / "backtests" / "output" / "swing" / "tpc" / "round_8" / "optimized_config.json",
}

OPTIMIZATION_RETURN_BASIS = "static_initial_strategy_risk"
OPTIMIZATION_SCORE_PROFILE = "portfolio_synergy_alpha_frequency"
OPTIMIZATION_SCORING_KWARGS = {
    "max_drawdown_hard_pct": 0.15,
    "drawdown_score_scale_pct": 0.15,
    "drawdown_penalty_start_pct": 0.12,
    "drawdown_penalty_full_pct": 0.15,
    "drawdown_penalty_weight": 0.30,
    "drawdown_comfort_pct": 0.12,
    "alpha_return_target_pct": 350.0,
    "min_profit_factor": 1.75,
    "min_trades": 120,
    "required_strategies": ("ATRSS", "AKC_HELIX", "TPC"),
    "min_required_strategy_trades": 10,
}


LIVE_SEED_MUTATIONS: dict[str, Any] = {
    "heat_cap_R": 5.0,
    "portfolio_daily_stop_R": 4.5,
    "atrss.unit_risk_pct": 0.016,
    "atrss.max_heat_R": 1.85,
    "atrss.daily_stop_R": 2.25,
    "helix.unit_risk_pct": 0.009,
    "helix.max_heat_R": 1.50,
    "helix.daily_stop_R": 2.5,
    "tpc.unit_risk_pct": 0.005,
    "tpc.max_heat_R": 1.00,
    "tpc.daily_stop_R": 2.0,
}


ROUND_1_CANDIDATES: list[tuple[str, dict[str, Any]]] = [
    (
        "risk_units_80",
        {
            "atrss.unit_risk_pct": 0.0128,
            "helix.unit_risk_pct": 0.0072,
            "tpc.unit_risk_pct": 0.0040,
        },
    ),
    (
        "risk_units_70",
        {
            "atrss.unit_risk_pct": 0.0112,
            "helix.unit_risk_pct": 0.0063,
            "tpc.unit_risk_pct": 0.0035,
        },
    ),
    (
        "risk_units_60",
        {
            "atrss.unit_risk_pct": 0.0096,
            "helix.unit_risk_pct": 0.0054,
            "tpc.unit_risk_pct": 0.0030,
        },
    ),
    (
        "risk_units_70_heat4",
        {
            "heat_cap_R": 4.0,
            "atrss.unit_risk_pct": 0.0112,
            "helix.unit_risk_pct": 0.0063,
            "tpc.unit_risk_pct": 0.0035,
        },
    ),
    (
        "strict_drawdown_throttle",
        {
            "dynamic_risk_enabled": True,
            "drawdown_risk_tiers": (
                (0.04, 0.80),
                (0.06, 0.60),
                (0.08, 0.40),
                (0.10, 0.25),
                (0.12, 0.00),
            ),
        },
    ),
    (
        "risk_units_80_strict_throttle",
        {
            "dynamic_risk_enabled": True,
            "atrss.unit_risk_pct": 0.0128,
            "helix.unit_risk_pct": 0.0072,
            "tpc.unit_risk_pct": 0.0040,
            "drawdown_risk_tiers": (
                (0.04, 0.80),
                (0.06, 0.60),
                (0.08, 0.40),
                (0.10, 0.25),
                (0.12, 0.00),
            ),
        },
    ),
    (
        "risk_units_80_stricter_throttle",
        {
            "dynamic_risk_enabled": True,
            "atrss.unit_risk_pct": 0.0128,
            "helix.unit_risk_pct": 0.0072,
            "tpc.unit_risk_pct": 0.0040,
            "drawdown_risk_tiers": (
                (0.035, 0.75),
                (0.055, 0.55),
                (0.075, 0.35),
                (0.095, 0.15),
                (0.11, 0.00),
            ),
        },
    ),
    (
        "risk_units_75_stricter_throttle",
        {
            "dynamic_risk_enabled": True,
            "atrss.unit_risk_pct": 0.0120,
            "helix.unit_risk_pct": 0.00675,
            "tpc.unit_risk_pct": 0.00375,
            "drawdown_risk_tiers": (
                (0.035, 0.75),
                (0.055, 0.55),
                (0.075, 0.35),
                (0.095, 0.15),
                (0.11, 0.00),
            ),
        },
    ),
    ("portfolio_heat3_5", {"heat_cap_R": 3.5}),
    ("portfolio_heat4_0", {"heat_cap_R": 4.0}),
    ("helix_unit70", {"helix.unit_risk_pct": 0.0063}),
    ("helix_unit60", {"helix.unit_risk_pct": 0.0054}),
    ("helix_no_addons", {"helix_flags.disable_add_ons": True}),
    ("helix_add_frac_0_75", {"helix_param.ADD_RISK_FRAC": 0.75}),
    ("helix_add_frac_0_50", {"helix_param.ADD_RISK_FRAC": 0.50}),
    (
        "helix_later_smaller_adds",
        {
            "helix_param.ADD_RISK_FRAC": 0.75,
            "helix_param.ADD_1H_R": 1.25,
            "helix_param.ADD_4H_R": 0.75,
        },
    ),
    (
        "risk80_helix_add_frac_0_50",
        {
            "atrss.unit_risk_pct": 0.0128,
            "helix.unit_risk_pct": 0.0072,
            "tpc.unit_risk_pct": 0.0040,
            "helix_param.ADD_RISK_FRAC": 0.50,
        },
    ),
    ("no_idle_priority_reservation", {"reserve_idle_higher_priority": False}),
    ("open_tpc_heat6_5", {"heat_cap_R": 6.5, "tpc.max_heat_R": 3.0}),
    ("open_tpc_heat7_5", {"heat_cap_R": 7.5, "tpc.max_heat_R": 4.0, "portfolio_daily_stop_R": 6.0}),
    ("tpc_disabled", {"tpc_symbols": []}),
    (
        "old_round2_overlay_control",
        {
            "heat_cap_R": 3.75,
            "portfolio_daily_stop_R": 3.25,
            "dynamic_risk_enabled": True,
            "overlay_max_pct": 0.55,
            "symbol_risk_multipliers": {"ATRSS:QQQ": 0.85, "ATRSS:GLD": 0.85},
        },
    ),
    ("overlay_70", {"overlay_max_pct": 0.70}),
    ("overlay_55", {"overlay_max_pct": 0.55}),
    ("multi_overlay", {"overlay_mode": "multi"}),
    ("dynamic_risk", {"dynamic_risk_enabled": True}),
    ("helix_heat2_0", {"helix.max_heat_R": 2.0}),
    ("daily_stop6_0", {"portfolio_daily_stop_R": 6.0}),
]


ROUND_2_CANDIDATES: list[tuple[str, dict[str, Any]]] = [
    ("no_idle_priority_reservation", {"reserve_idle_higher_priority": False}),
    ("risk_units_90", {"atrss.unit_risk_pct": 0.0144, "helix.unit_risk_pct": 0.0081, "tpc.unit_risk_pct": 0.0045}),
    ("risk_units_75", {"atrss.unit_risk_pct": 0.0120, "helix.unit_risk_pct": 0.00675, "tpc.unit_risk_pct": 0.00375}),
    ("risk_units_65", {"atrss.unit_risk_pct": 0.0104, "helix.unit_risk_pct": 0.00585, "tpc.unit_risk_pct": 0.00325}),
    ("helix_heat1_2", {"helix.max_heat_R": 1.2}),
    ("atrss_heat1_5", {"atrss.max_heat_R": 1.5}),
    ("portfolio_heat3_5", {"heat_cap_R": 3.5}),
    ("portfolio_heat4_0", {"heat_cap_R": 4.0}),
    ("helix_unit70", {"helix.unit_risk_pct": 0.0063}),
    ("helix_unit60", {"helix.unit_risk_pct": 0.0054}),
    ("helix_no_addons", {"helix_flags.disable_add_ons": True}),
    ("helix_add_frac_0_75", {"helix_param.ADD_RISK_FRAC": 0.75}),
    ("helix_add_frac_0_50", {"helix_param.ADD_RISK_FRAC": 0.50}),
    (
        "helix_later_smaller_adds",
        {
            "helix_param.ADD_RISK_FRAC": 0.75,
            "helix_param.ADD_1H_R": 1.25,
            "helix_param.ADD_4H_R": 0.75,
        },
    ),
    ("tpc_heat2_5", {"tpc.max_heat_R": 2.5}),
    ("tpc_heat3_5", {"tpc.max_heat_R": 3.5}),
    ("tpc_heat5_0", {"tpc.max_heat_R": 5.0}),
    ("portfolio_heat6_0", {"heat_cap_R": 6.0}),
    ("portfolio_heat7_0", {"heat_cap_R": 7.0}),
    ("portfolio_heat8_5", {"heat_cap_R": 8.5}),
    ("daily_stop5_0", {"portfolio_daily_stop_R": 5.0}),
    ("daily_stop7_0", {"portfolio_daily_stop_R": 7.0}),
    ("overlay_75", {"overlay_max_pct": 0.75}),
    ("overlay_60", {"overlay_max_pct": 0.60}),
    ("overlay_off", {"overlay_enabled": False}),
    (
        "tpc_mid_source_risk",
        {
            "tpc_param.all.max_risk_pct": 0.015,
            "tpc_param.all.risk_a_plus_pct": 0.015,
            "tpc_param.all.risk_a_pct": 0.010,
            "tpc_param.all.risk_b_pct": 0.006,
        },
    ),
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def latest_strategy_mutations() -> dict[str, Any]:
    mutations = dict(LIVE_SEED_MUTATIONS)

    atrss = _load_json(LATEST_CONFIGS["ATRSS"])
    for key, value in atrss.items():
        if key.startswith("flags."):
            mutations[f"atrss_flags.{key.split('.', 1)[1]}"] = value
        elif key.startswith("param_overrides."):
            mutations[f"atrss_param.{key.split('.', 1)[1]}"] = value
        elif key == "fixed_qty" and value is not None:
            mutations["fixed_qty"] = value

    helix = _load_json(LATEST_CONFIGS["AKC_HELIX"])
    for key, value in helix.items():
        if key.startswith("flags."):
            mutations[f"helix_flags.{key.split('.', 1)[1]}"] = value
        elif key.startswith("param_overrides."):
            mutations[f"helix_param.{key.split('.', 1)[1]}"] = value

    tpc = _load_json(LATEST_CONFIGS["TPC"])
    for key, value in tpc.items():
        mutations[f"tpc_param.{key}"] = value

    return mutations


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _collect_trades(result) -> list[Any]:
    trades: list[Any] = []
    for attr in ("atrss_trades", "helix_trades", "tpc_trades"):
        trades.extend(getattr(result, attr, []) or [])
    return trades


def _sharpe(equity: np.ndarray) -> float:
    if len(equity) < 3:
        return 0.0
    returns = np.diff(equity) / equity[:-1]
    if len(returns) < 2 or float(np.std(returns)) == 0.0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 7))


def _max_dd_pct(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(abs(np.min((equity - peak) / peak)))


def _profit_factor(trades: list[Any]) -> float:
    def _net_pnl(t: Any) -> float:
        if hasattr(t, "net_pnl_dollars"):
            return float(getattr(t, "net_pnl_dollars", 0.0) or 0.0)
        return float(getattr(t, "pnl_dollars", 0.0) or 0.0) - float(getattr(t, "commission", 0.0) or 0.0)

    pnls = [_net_pnl(t) for t in trades]
    wins = sum(pnl for pnl in pnls if pnl > 0)
    losses = abs(sum(pnl for pnl in pnls if pnl < 0))
    return wins / losses if losses > 0 else float("inf")


def _block_reason_summary(result) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for event in getattr(result, "heat_rejections", []) or []:
        strategy = str(event.get("strategy", "UNKNOWN") or "UNKNOWN")
        reason = str(event.get("reason", "unknown") or "unknown")
        by_reason = summary.setdefault(strategy, {})
        by_reason[reason] = by_reason.get(reason, 0) + 1
    return summary


_STRATEGY_SLOT_ATTRS = {
    "ATRSS": "atrss",
    "AKC_HELIX": "helix",
    "TPC": "tpc",
}


def _initial_unit_risk_dollars(
    config: UnifiedBacktestConfig,
    strategy_id: str,
    initial_equity: float,
) -> float:
    slot_attr = _STRATEGY_SLOT_ATTRS.get(strategy_id)
    if slot_attr is None:
        return 0.0
    slot = getattr(config, slot_attr, None)
    unit_risk_pct = float(getattr(slot, "unit_risk_pct", 0.0) or 0.0)
    return float(initial_equity) * unit_risk_pct


def _static_initial_risk_return(
    config: UnifiedBacktestConfig,
    strategy_results: dict[str, Any],
    initial_equity: float,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Return PnL using each strategy's initial unit risk instead of compounded dollars.

    Individual strategy reports are easier to compare in R/initial-equity terms
    than in a compounded family equity curve.  This keeps the portfolio
    headline from being dominated by later large-dollar trades after the
    family account has already grown.
    """
    total = 0.0
    by_strategy: dict[str, dict[str, float]] = {}
    for sid, sr in strategy_results.items():
        total_r = float(getattr(sr, "total_r", 0.0) or 0.0)
        unit_risk = _initial_unit_risk_dollars(config, sid, initial_equity)
        pnl = total_r * unit_risk
        total += pnl
        by_strategy[sid] = {
            "initial_unit_risk_dollars": unit_risk,
            "static_risk_pnl": pnl,
            "static_risk_return_pct": (pnl / initial_equity * 100.0) if initial_equity else 0.0,
        }
    return total, by_strategy


def _evaluate(
    data,
    mutations: dict[str, Any],
    *,
    equity: float,
    data_dir: Path,
) -> tuple[UnifiedBacktestConfig, Any, dict[str, Any]]:
    config = UnifiedBacktestConfig(initial_equity=equity, data_dir=data_dir)
    config = mutate_unified_config(config, mutations)
    result = run_unified(data, config)
    eq = np.asarray(result.combined_equity, dtype=float)
    trades = _collect_trades(result)
    static_risk_pnl, static_risk_by_strategy = _static_initial_risk_return(
        config,
        result.strategy_results,
        equity,
    )
    strategy_summary = {}
    for sid, sr in result.strategy_results.items():
        strategy_summary[sid] = {
            "trades": sr.total_trades,
            "pnl": sr.total_pnl,
            "compounded_pnl": sr.total_pnl,
            "total_r": sr.total_r,
            "entry_signals_fired": sr.entry_signals_fired,
            "entry_requests": getattr(sr, "entry_requests", sr.entry_signals_fired),
            "entries_accepted": sr.entries_accepted_by_portfolio,
            "entries_blocked": sr.entries_blocked_by_heat,
            "suppressed_entry_retries": getattr(sr, "suppressed_entry_retries", 0),
            **static_risk_by_strategy.get(sid, {}),
        }
    final_equity = float(eq[-1]) if len(eq) else equity
    compounded_mtm_net_pnl = final_equity - equity
    static_risk_return_pct = (static_risk_pnl / equity * 100.0) if equity else 0.0
    compounded_mtm_return_pct = (compounded_mtm_net_pnl / equity * 100.0) if equity else 0.0
    metrics = {
        "initial_equity": equity,
        "final_equity": final_equity,
        "return_basis": "static_initial_strategy_risk",
        "net_pnl": static_risk_pnl,
        "net_return_pct": static_risk_return_pct,
        "static_risk_net_pnl": static_risk_pnl,
        "static_risk_net_return_pct": static_risk_return_pct,
        "compounded_mtm_final_equity": final_equity,
        "compounded_mtm_net_pnl": compounded_mtm_net_pnl,
        "compounded_mtm_net_return_pct": compounded_mtm_return_pct,
        "total_trades": len(trades),
        "profit_factor": _profit_factor(trades),
        "max_drawdown_pct": _max_dd_pct(eq) * 100,
        "sharpe": _sharpe(eq),
        "overlay_pnl": float(getattr(result, "overlay_pnl", 0.0) or 0.0),
        "overlay_commission": float(getattr(result, "overlay_commission", 0.0) or 0.0),
        "heat_avg_R": result.heat_stats.avg_heat_pct,
        "heat_max_R": result.heat_stats.max_heat_pct,
        "heat_pct_time_at_cap": result.heat_stats.pct_time_at_cap,
        "coordination_tightens": result.coordination_tighten_count,
        "coordination_boosts": result.coordination_boost_count,
        "portfolio_daily_stop_activations": result.portfolio_daily_stop_activations,
        "portfolio_rule_events": len(getattr(result, "portfolio_rule_events", []) or []),
        "portfolio_rule_block_events": sum(
            1
            for event in (getattr(result, "portfolio_rule_events", []) or [])
            if event.get("result") == "blocked"
        ),
        "portfolio_rule_sizing_events": sum(
            1
            for event in (getattr(result, "portfolio_rule_events", []) or [])
            if event.get("result") == "sized"
        ),
        "block_reason_summary": _block_reason_summary(result),
        "strategy_summary": strategy_summary,
    }
    return config, result, metrics


def _write_round_artifacts(
    round_dir: Path,
    *,
    round_num: int,
    base_mutations: dict[str, Any],
    candidates: list[tuple[str, dict[str, Any]]],
    greedy_result,
    final_mutations: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    round_dir.mkdir(parents=True, exist_ok=True)
    greedy_result_path = round_dir / "greedy_result.json"
    save_result(greedy_result, greedy_result_path)
    greedy_payload = json.loads(greedy_result_path.read_text())
    greedy_payload["return_basis"] = metrics["return_basis"]
    greedy_payload["final_return_pct"] = metrics["net_return_pct"] / 100.0
    greedy_payload["compounded_mtm_final_return_pct"] = (
        metrics["compounded_mtm_net_return_pct"] / 100.0
    )
    greedy_result_path.write_text(json.dumps(greedy_payload, indent=2, default=_json_default))
    (round_dir / "candidate_space.json").write_text(json.dumps(
        [{"name": name, "mutations": muts} for name, muts in candidates],
        indent=2,
        default=_json_default,
    ))
    (round_dir / "optimized_config.json").write_text(json.dumps(final_mutations, indent=2, default=_json_default))
    summary = {
        "family": "swing",
        "strategy": "portfolio_synergy",
        "round": round_num,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_strategy_configs": {sid: str(path) for sid, path in LATEST_CONFIGS.items()},
        "base_mutation_count": len(base_mutations),
        "candidate_count": len(candidates),
        "kept_features": greedy_result.kept_features,
        "base_score": greedy_result.base_score,
        "final_score": greedy_result.final_score,
        "optimization_return_basis": OPTIMIZATION_RETURN_BASIS,
        "optimization_score_profile": OPTIMIZATION_SCORE_PROFILE,
        "optimization_scoring_kwargs": OPTIMIZATION_SCORING_KWARGS,
        "improvement_pct": (
            (greedy_result.final_score - greedy_result.base_score) / greedy_result.base_score
            if greedy_result.base_score > 0 else 0.0
        ),
        "final_mutations": final_mutations,
        "final_metrics": metrics,
    }
    (round_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))

    lines = [
        f"SWING PORTFOLIO SYNERGY ROUND {round_num} EVALUATION",
        "=" * 70,
        f"Base score:  {greedy_result.base_score:.4f}",
        f"Final score: {greedy_result.final_score:.4f}",
        f"Score basis: {OPTIMIZATION_RETURN_BASIS}",
        (
            "DD scoring: 12% penalty start, 15% full penalty, "
            "15% hard reject"
        ),
        f"Kept:        {', '.join(greedy_result.kept_features) if greedy_result.kept_features else '(none)'}",
        "",
        "Final Metrics",
        f"  Comparable net return: {metrics['net_return_pct']:+.2f}% ({metrics['return_basis']})",
        f"  Comparable net PnL:    ${metrics['net_pnl']:+,.2f}",
        f"  Compounded MTM return: {metrics['compounded_mtm_net_return_pct']:+.2f}%",
        f"  Final MTM equity:      ${metrics['compounded_mtm_final_equity']:,.2f}",
        f"  Compounded MTM PnL:    ${metrics['compounded_mtm_net_pnl']:+,.2f}",
        f"  Max DD:                {metrics['max_drawdown_pct']:.2f}%",
        f"  Sharpe:                {metrics['sharpe']:.2f}",
        f"  PF:                    {metrics['profit_factor']:.2f}",
        f"  Trades:                {metrics['total_trades']}",
        f"  Overlay PnL:           ${metrics['overlay_pnl']:+,.2f}",
        f"  Max heat:              {metrics['heat_max_R']:.2f}R",
        "",
        "Per Strategy",
    ]
    for sid, strat in metrics["strategy_summary"].items():
        lines.append(
            f"  {sid:<10} trades={strat['trades']:>4} "
            f"static_pnl=${strat.get('static_risk_pnl', 0.0):+,.2f} "
            f"actual_pnl=${strat['pnl']:+,.2f} "
            f"signals={strat.get('entry_signals_fired', 0)} "
            f"requests={strat.get('entry_requests', strat.get('entry_signals_fired', 0))} "
            f"blocked={strat['entries_blocked']}"
        )
    if metrics.get("block_reason_summary"):
        lines.extend(["", "Block Reasons"])
        for sid, reasons in metrics["block_reason_summary"].items():
            reason_text = ", ".join(f"{reason}: {count}" for reason, count in sorted(reasons.items()))
            lines.append(f"  {sid:<10} {reason_text}")
    lines.extend(["", "Final Mutations"])
    for key, value in sorted(final_mutations.items()):
        lines.append(f"  {key}: {value}")
    (round_dir / "round_evaluation.txt").write_text("\n".join(lines) + "\n")


def run_two_rounds(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data_dir)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    latest = latest_strategy_mutations()
    (output_root / "source_strategy_config_mutations.json").write_text(json.dumps(
        latest,
        indent=2,
        default=_json_default,
    ))

    print("Loading unified data...", flush=True)
    t0 = time.time()
    seed_config = mutate_unified_config(
        UnifiedBacktestConfig(initial_equity=args.equity, data_dir=data_dir),
        latest,
    )
    data = load_unified_data(seed_config)
    print(f"Data loaded in {time.time() - t0:.1f}s", flush=True)

    print("\n=== ROUND 1: broad synergy pass ===", flush=True)
    round1 = run_greedy(
        data=data,
        candidates=ROUND_1_CANDIDATES,
        initial_equity=args.equity,
        base_mutations=latest,
        data_dir=data_dir,
        max_workers=args.max_workers,
        return_basis=OPTIMIZATION_RETURN_BASIS,
        scoring_kwargs=OPTIMIZATION_SCORING_KWARGS,
        score_profile=OPTIMIZATION_SCORE_PROFILE,
        verbose=True,
    )
    round1_mutations = {**latest, **round1.final_mutations}
    _, _, round1_metrics = _evaluate(data, round1_mutations, equity=args.equity, data_dir=data_dir)
    _write_round_artifacts(
        output_root / "round_1",
        round_num=1,
        base_mutations=latest,
        candidates=ROUND_1_CANDIDATES,
        greedy_result=round1,
        final_mutations=round1_mutations,
        metrics=round1_metrics,
    )

    print("\n=== ROUND 2: refinement pass ===", flush=True)
    round2 = run_greedy(
        data=data,
        candidates=ROUND_2_CANDIDATES,
        initial_equity=args.equity,
        base_mutations=round1_mutations,
        data_dir=data_dir,
        max_workers=args.max_workers,
        return_basis=OPTIMIZATION_RETURN_BASIS,
        scoring_kwargs=OPTIMIZATION_SCORING_KWARGS,
        score_profile=OPTIMIZATION_SCORE_PROFILE,
        verbose=True,
    )
    round2_mutations = {**round1_mutations, **round2.final_mutations}
    _, _, round2_metrics = _evaluate(data, round2_mutations, equity=args.equity, data_dir=data_dir)
    _write_round_artifacts(
        output_root / "round_2",
        round_num=2,
        base_mutations=round1_mutations,
        candidates=ROUND_2_CANDIDATES,
        greedy_result=round2,
        final_mutations=round2_mutations,
        metrics=round2_metrics,
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "equity": args.equity,
        "data_dir": str(data_dir),
        "optimization_return_basis": OPTIMIZATION_RETURN_BASIS,
        "optimization_score_profile": OPTIMIZATION_SCORE_PROFILE,
        "optimization_scoring_kwargs": OPTIMIZATION_SCORING_KWARGS,
        "rounds": [
            {"round": 1, "path": str(output_root / "round_1"), "metrics": round1_metrics},
            {"round": 2, "path": str(output_root / "round_2"), "metrics": round2_metrics},
        ],
        "winner": {
            "round": 2,
            "optimized_config": str(output_root / "round_2" / "optimized_config.json"),
            "run_summary": str(output_root / "round_2" / "run_summary.json"),
            "metrics": round2_metrics,
        },
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default))
    print(f"\nTwo-round synergy rerun complete: {output_root}", flush=True)
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equity", type=float, default=25_000.0)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "backtests" / "swing" / "data" / "raw")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "backtests" / "output" / "swing" / "portfolio_synergy" / "latest_tpc_rerun_20260507",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    run_two_rounds(parse_args())


if __name__ == "__main__":
    main()
