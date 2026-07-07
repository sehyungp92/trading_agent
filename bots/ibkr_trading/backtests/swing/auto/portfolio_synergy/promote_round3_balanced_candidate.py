"""Promote the selected balanced swing portfolio candidate to round 3.

The script replays round 2 and the promoted round 3 config through the unified
portfolio engine, writes full diagnostics into the canonical round folders, and
refreshes the round manager manifest.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from backtests.shared.auto.provenance import AutoRunProvenance, build_phase_auto_provenance
from backtests.shared.auto.round_manager import RoundManager, canonicalize_metrics
from backtests.swing.auto.portfolio_synergy.run_latest_two_rounds import _json_default
from backtests.swing.auto.portfolio_synergy.run_phase_auto_from_latest import (
    LIVE_PORTFOLIO_RULE_CHECKER,
    LIVE_STRATEGY_COORDINATOR,
    LIVE_SWING_RISK_ADAPTER,
    OPTIMIZATION_RETURN_BASIS,
    PHASE_SCORING_KWARGS,
    PHASE_SOURCE_CONFIGS,
    PHASE_SOURCE_DIAGNOSTICS,
    PORTFOLIO_REPLAY_SCOPE,
    REPLAY_ARCHITECTURE,
    RISK_STANCE,
    SCORE_PROFILE,
    STARTING_EQUITY,
    THIN_LAYER_ASSESSMENT,
    PortfolioSynergyPhasePlugin,
)


DEFAULT_DATA_DIR = ROOT / "backtests" / "swing" / "data" / "raw"
OUTPUT_ROOT = ROOT / "backtests" / "output" / "swing" / "portfolio_synergy"
ROUND_2_CONFIG = OUTPUT_ROOT / "round_2" / "optimized_config.json"
OOS_REPORT = OUTPUT_ROOT / "oos" / "candidate_oos_compare_20260509_151131.txt"

CANDIDATE_NAME = "balanced_509_77_atrss_71"
CANDIDATE_SOURCE_NAME = "focused_helix_tpc_heavier"
CANDIDATE_DELTAS: dict[str, Any] = {
    "atrss.unit_risk_pct": 0.0165,
    "atrss.max_heat_R": 2.15,
    "helix.unit_risk_pct": 0.0130,
    "helix.max_heat_R": 2.10,
    "tpc.unit_risk_pct": 0.0050,
    "tpc.max_heat_R": 4.00,
    "tpc_param.all.max_risk_pct": 0.020,
    "tpc_param.all.risk_a_plus_pct": 0.020,
    "tpc_param.all.risk_a_pct": 0.012,
    "tpc_param.all.risk_b_pct": 0.009,
}

SOURCE_SMOKE_METRICS = {
    "net_return_pct": 509.77,
    "max_drawdown_pct": 8.56,
    "total_trades": 716,
    "profit_factor": 3.35,
    "atrss_static_pnl_share_pct": 71.19,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _strategy_shares(metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    summary = metrics.get("strategy_summary", {}) or {}
    positive_total = sum(
        max(_safe_float(item.get("static_risk_pnl", 0.0)), 0.0)
        for item in summary.values()
    )
    shares: dict[str, dict[str, float]] = {}
    for sid, item in summary.items():
        static_pnl = _safe_float(item.get("static_risk_pnl", 0.0))
        shares[sid] = {
            "trades": _safe_float(item.get("trades", 0.0)),
            "signals": _safe_float(item.get("entry_signals_fired", 0.0)),
            "requests": _safe_float(item.get("entry_requests", item.get("entry_signals_fired", 0.0))),
            "accepted": _safe_float(item.get("entries_accepted", 0.0)),
            "blocked": _safe_float(item.get("entries_blocked", 0.0)),
            "suppressed": _safe_float(item.get("suppressed_entry_retries", 0.0)),
            "total_r": _safe_float(item.get("total_r", 0.0)),
            "unit_risk": _safe_float(item.get("initial_unit_risk_dollars", 0.0)),
            "static_pnl": static_pnl,
            "share_pct": (static_pnl / positive_total * 100.0) if positive_total > 0.0 else 0.0,
        }
    return shares


def _config_snapshot(mutations: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "heat_cap_R",
        "portfolio_daily_stop_R",
        "dynamic_risk_enabled",
        "drawdown_risk_tiers",
        "overlay_enabled",
        "overlay_max_pct",
        "atrss.unit_risk_pct",
        "atrss.max_heat_R",
        "helix.unit_risk_pct",
        "helix.max_heat_R",
        "tpc.unit_risk_pct",
        "tpc.max_heat_R",
        "tpc_param.all.max_risk_pct",
        "tpc_param.all.risk_a_plus_pct",
        "tpc_param.all.risk_a_pct",
        "tpc_param.all.risk_b_pct",
    )
    return {key: mutations[key] for key in keys if key in mutations}


def _format_report(
    *,
    round_num: int,
    label: str,
    metrics: dict[str, Any],
    mutations: dict[str, Any],
    config_source: str,
    candidate_deltas: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
) -> str:
    shares = _strategy_shares(metrics)
    lines = [
        f"SWING PORTFOLIO SYNERGY ROUND {round_num} FINAL DIAGNOSTICS",
        "=" * 70,
        f"Label: {label}",
        f"Initial equity: ${metrics.get('initial_equity', STARTING_EQUITY):,.0f}",
        f"Risk stance: {RISK_STANCE}",
        f"Score profile: {SCORE_PROFILE}",
        f"Return basis: {metrics.get('return_basis', OPTIMIZATION_RETURN_BASIS)}",
        f"Config source: {config_source}",
        f"Replay architecture: {metrics.get('replay_architecture', REPLAY_ARCHITECTURE)}",
        f"Replay source fingerprint: {metrics.get('replay_source_fingerprint', '')}",
        "",
        "Headline",
        f"  Comparable net return: {metrics.get('net_return_pct', 0.0):+.2f}%",
        f"  Comparable net PnL:    ${metrics.get('net_pnl', 0.0):+,.2f}",
        f"  Compounded MTM return: {metrics.get('compounded_mtm_net_return_pct', 0.0):+.2f}%",
        f"  Final MTM equity:      ${metrics.get('compounded_mtm_final_equity', 0.0):,.2f}",
        f"  Max DD:                {metrics.get('max_drawdown_pct', 0.0):.2f}%",
        f"  Sharpe:                {metrics.get('sharpe', 0.0):.2f}",
        f"  PF:                    {metrics.get('profit_factor', 0.0):.2f}",
        f"  Trades:                {metrics.get('total_trades', 0)}",
        f"  Overlay PnL:           ${metrics.get('overlay_pnl', 0.0):+,.2f}",
        f"  Max heat:              {metrics.get('heat_max_R', 0.0):.2f}R",
        f"  Daily stops:           {metrics.get('portfolio_daily_stop_activations', 0)}",
        f"  Portfolio rule blocks: {metrics.get('portfolio_rule_block_events', 0)}",
        f"  Portfolio size events: {metrics.get('portfolio_rule_sizing_events', 0)}",
        "",
        "Per Strategy",
    ]
    for sid, strat in (metrics.get("strategy_summary", {}) or {}).items():
        share = shares.get(sid, {}).get("share_pct", 0.0)
        lines.append(
            f"  {sid:<10} trades={strat.get('trades', 0):>4} "
            f"signals={strat.get('entry_signals_fired', 0):>4} "
            f"requests={strat.get('entry_requests', strat.get('entry_signals_fired', 0)):>4} "
            f"accepted={strat.get('entries_accepted', 0):>4} "
            f"blocked={strat.get('entries_blocked', 0):>4} "
            f"suppressed={strat.get('suppressed_entry_retries', 0):>4} "
            f"total_R={strat.get('total_r', 0.0):>8.2f} "
            f"static_pnl=${strat.get('static_risk_pnl', 0.0):+,.2f} "
            f"share={share:5.2f}%"
        )

    if comparison:
        lines.extend(["", "Round 2 Comparison"])
        for key, label_text in (
            ("net_return_pct", "Comparable return"),
            ("max_drawdown_pct", "Max DD"),
            ("profit_factor", "PF"),
            ("total_trades", "Trades"),
        ):
            base = _safe_float(comparison["baseline"].get(key, 0.0))
            current = _safe_float(metrics.get(key, 0.0))
            lines.append(f"  {label_text:<18} round2={base:,.2f} round{round_num}={current:,.2f} delta={current - base:+,.2f}")
        base_share = _safe_float(comparison["baseline_shares"].get("ATRSS", {}).get("share_pct", 0.0))
        current_share = _safe_float(shares.get("ATRSS", {}).get("share_pct", 0.0))
        lines.append(f"  ATRSS PnL share    round2={base_share:.2f}% round{round_num}={current_share:.2f}% delta={current_share - base_share:+.2f}pp")

    if metrics.get("block_reason_summary"):
        lines.extend(["", "Block Reasons"])
        for sid, reasons in metrics["block_reason_summary"].items():
            reason_text = ", ".join(f"{reason}: {count}" for reason, count in sorted(reasons.items()))
            lines.append(f"  {sid:<10} {reason_text}")

    if candidate_deltas:
        lines.extend(["", "Promoted Candidate Deltas"])
        for key, value in sorted(candidate_deltas.items()):
            lines.append(f"  {key}: {value}")

    lines.extend(["", "Key Config"])
    for key, value in _config_snapshot(mutations).items():
        lines.append(f"  {key}: {value}")

    lines.extend(
        [
            "",
            "Existing Safeguards",
            f"  Swing heat adapter: {metrics.get('live_swing_risk_adapter', LIVE_SWING_RISK_ADAPTER)}",
            f"  Portfolio rule checker: {metrics.get('live_portfolio_rule_checker', LIVE_PORTFOLIO_RULE_CHECKER)}",
            f"  Strategy coordinator: {metrics.get('live_strategy_coordinator', LIVE_STRATEGY_COORDINATOR)}",
            "  Risk basis: static initial strategy-risk returns for selection, MTM equity for drawdown.",
            f"  Replay scope: {metrics.get('portfolio_replay_scope', PORTFOLIO_REPLAY_SCOPE)}",
            "  Thin-layer status: portfolio risk is thin over live risk/coordination; TPC source replay is thin; ATRSS and Helix still rely on migrated shared-core components plus source-engine decision paths.",
        ]
    )
    return "\n".join(lines) + "\n"


def _diagnostics_summary(
    *,
    round_num: int,
    label: str,
    metrics: dict[str, Any],
    mutations: dict[str, Any],
    config_source: str,
    data_dir: Path,
    candidate_deltas: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "family": "swing",
        "strategy": "portfolio_synergy",
        "round": round_num,
        "label": label,
        "generated_at_utc": _utc_now(),
        "initial_equity": metrics.get("initial_equity", STARTING_EQUITY),
        "data_dir": str(data_dir),
        "config_source": config_source,
        "candidate_name": CANDIDATE_NAME if candidate_deltas else None,
        "candidate_source_name": CANDIDATE_SOURCE_NAME if candidate_deltas else None,
        "source_smoke_metrics": SOURCE_SMOKE_METRICS if candidate_deltas else None,
        "oos_report": str(OOS_REPORT) if OOS_REPORT.exists() else None,
        "headline_metrics": canonicalize_metrics(metrics),
        "strategy_static_pnl_shares": _strategy_shares(metrics),
        "final_metrics": metrics,
        "optimized_config": mutations,
        "candidate_deltas": dict(candidate_deltas or {}),
        "thin_layer_assessment": THIN_LAYER_ASSESSMENT,
    }


def _build_provenance(
    *,
    round_num: int,
    data_dir: Path,
    config_source: str,
    candidate_deltas: dict[str, Any] | None,
) -> AutoRunProvenance:
    source_artifacts = {
        f"{sid}:optimized_config": path
        for sid, path in PHASE_SOURCE_CONFIGS.items()
    }
    source_artifacts.update(
        {
            f"{sid}:diagnostics": path
            for sid, path in PHASE_SOURCE_DIAGNOSTICS.items()
        }
    )
    source_artifacts["baseline_config"] = Path(config_source)
    if OOS_REPORT.exists():
        source_artifacts["oos_candidate_report"] = OOS_REPORT

    return build_phase_auto_provenance(
        "portfolio_synergy",
        repo_root=ROOT,
        code_dirs=(Path(__file__).resolve().parent,),
        code_paths=(
            Path(__file__).resolve(),
            ROOT / "backtests/swing/engine/unified_portfolio_engine.py",
            ROOT / "strategies/swing/overlay/engine.py",
            ROOT / "libs/oms/risk/portfolio_rules.py",
            ROOT / "libs/oms/risk/swing_portfolio_adapter.py",
        ),
        data_dir=data_dir,
        source_artifacts=source_artifacts,
        selection_context={
            "round": round_num,
            "candidate_name": CANDIDATE_NAME if candidate_deltas else None,
            "candidate_source_name": CANDIDATE_SOURCE_NAME if candidate_deltas else None,
            "candidate_deltas": dict(candidate_deltas or {}),
            "risk_stance": RISK_STANCE,
            "score_profile": SCORE_PROFILE,
            "return_basis": OPTIMIZATION_RETURN_BASIS,
            "replay_architecture": REPLAY_ARCHITECTURE,
            "phase_scoring_kwargs": PHASE_SCORING_KWARGS,
        },
    )


def _write_round_outputs(
    *,
    manager: RoundManager,
    round_num: int,
    round_dir: Path,
    label: str,
    metrics: dict[str, Any],
    mutations: dict[str, Any],
    data_dir: Path,
    config_source: str,
    completed_phases: list[int],
    candidate_deltas: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
) -> dict[str, Path]:
    diagnostics_text = _format_report(
        round_num=round_num,
        label=label,
        metrics=metrics,
        mutations=mutations,
        config_source=config_source,
        candidate_deltas=candidate_deltas,
        comparison=comparison,
    )
    diagnostics_path = manager.diagnostics_path(round_dir)
    diagnostics_path.write_text(diagnostics_text, encoding="utf-8")

    summary = _diagnostics_summary(
        round_num=round_num,
        label=label,
        metrics=metrics,
        mutations=mutations,
        config_source=config_source,
        data_dir=data_dir,
        candidate_deltas=candidate_deltas,
    )
    provenance = _build_provenance(
        round_num=round_num,
        data_dir=data_dir,
        config_source=config_source,
        candidate_deltas=candidate_deltas,
    )
    summary["provenance"] = provenance.to_dict()
    summary["provenance_status"] = "complete"
    diagnostics_summary_path = manager.diagnostics_summary_path(round_dir)
    _write_json(diagnostics_summary_path, summary)

    manager.write_optimized_config(round_dir, mutations)
    manager.write_run_summary(
        round_dir,
        mutations,
        metrics,
        completed_phases,
        round_num=round_num,
        source_diagnostics=diagnostics_path,
        provenance=provenance,
        provenance_status="complete",
    )
    manager.append_to_manifest(round_num, mutations, metrics, provenance=provenance, provenance_status="complete")
    return {
        "diagnostics": diagnostics_path,
        "diagnostics_summary": diagnostics_summary_path,
        "optimized_config": manager.optimized_config_path(round_dir),
        "run_summary": manager.run_summary_path(round_dir),
        "manifest": manager.manifest_path,
    }


def promote(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    round2_config_path = Path(args.round2_config)
    if not round2_config_path.is_absolute():
        round2_config_path = ROOT / round2_config_path

    round2_mutations = _load_json(round2_config_path)
    round3_mutations = dict(round2_mutations)
    round3_mutations.update(CANDIDATE_DELTAS)

    manager = RoundManager("swing", "portfolio_synergy")
    round2_dir = manager.get_round_dir(2)
    round3_dir = manager.get_round_dir(3)

    plugin = PortfolioSynergyPhasePlugin(
        data_dir,
        initial_equity=float(args.equity),
        max_workers=1,
        initial_mutations=round2_mutations,
        base_source=str(round2_config_path),
    )

    print("Replaying round 2 optimized config...", flush=True)
    round2_metrics = plugin.compute_final_metrics(round2_mutations)
    print(
        f"round_2: return={round2_metrics['net_return_pct']:.2f}% "
        f"dd={round2_metrics['max_drawdown_pct']:.2f}% "
        f"pf={round2_metrics['profit_factor']:.2f} "
        f"trades={round2_metrics['total_trades']}",
        flush=True,
    )

    print(f"Replaying {CANDIDATE_NAME} as round 3...", flush=True)
    round3_metrics = plugin.compute_final_metrics(round3_mutations)
    print(
        f"round_3: return={round3_metrics['net_return_pct']:.2f}% "
        f"dd={round3_metrics['max_drawdown_pct']:.2f}% "
        f"pf={round3_metrics['profit_factor']:.2f} "
        f"trades={round3_metrics['total_trades']}",
        flush=True,
    )

    round2_outputs = _write_round_outputs(
        manager=manager,
        round_num=2,
        round_dir=round2_dir,
        label="round_2 optimized config diagnostic replay",
        metrics=round2_metrics,
        mutations=round2_mutations,
        data_dir=data_dir,
        config_source=str(round2_config_path),
        completed_phases=[1, 2, 3, 4, 5],
    )
    round3_provenance = _build_provenance(
        round_num=3,
        data_dir=data_dir,
        config_source=str(round2_config_path),
        candidate_deltas=CANDIDATE_DELTAS,
    )

    manager.write_run_spec(
        round3_dir,
        3,
        strategy_name="portfolio_synergy",
        description=f"Round 3 promotion of {CANDIDATE_NAME} from balance/OOS candidate smoke tests.",
        scoring_weights={
            "alpha_quality": 0.30,
            "frequency_quality": 0.24,
            "drawdown_quality": 0.16,
            "pf_quality": 0.09,
            "balance_quality": 0.10,
            "capture_quality": 0.07,
            "robustness_quality": 0.04,
        },
        baseline_mutations=round2_mutations,
        baseline_source=round2_config_path,
        execution_context={
            "candidate_name": CANDIDATE_NAME,
            "candidate_source_name": CANDIDATE_SOURCE_NAME,
            "candidate_deltas": CANDIDATE_DELTAS,
            "source_smoke_metrics": SOURCE_SMOKE_METRICS,
            "oos_report": str(OOS_REPORT) if OOS_REPORT.exists() else None,
            "replay_architecture": REPLAY_ARCHITECTURE,
            "risk_stance": RISK_STANCE,
            "return_basis": OPTIMIZATION_RETURN_BASIS,
        },
        provenance=round3_provenance,
        provenance_status="complete",
        overwrite=True,
    )

    comparison = {
        "baseline": round2_metrics,
        "baseline_shares": _strategy_shares(round2_metrics),
    }
    round3_outputs = _write_round_outputs(
        manager=manager,
        round_num=3,
        round_dir=round3_dir,
        label=f"{CANDIDATE_NAME} promoted optimized config",
        metrics=round3_metrics,
        mutations=round3_mutations,
        data_dir=data_dir,
        config_source=f"{round2_config_path} + {CANDIDATE_NAME}",
        completed_phases=[1],
        candidate_deltas=CANDIDATE_DELTAS,
        comparison=comparison,
    )

    evaluation_lines = [
        "SWING PORTFOLIO SYNERGY ROUND 3 PROMOTION EVALUATION",
        "=" * 70,
        f"Candidate: {CANDIDATE_NAME}",
        f"Source smoke candidate: {CANDIDATE_SOURCE_NAME}",
        f"OOS comparison: {OOS_REPORT if OOS_REPORT.exists() else '(missing)'}",
        "",
        "Round 2 vs Promoted Round 3",
        f"  Return: {round2_metrics['net_return_pct']:+.2f}% -> {round3_metrics['net_return_pct']:+.2f}%",
        f"  Max DD: {round2_metrics['max_drawdown_pct']:.2f}% -> {round3_metrics['max_drawdown_pct']:.2f}%",
        f"  PF: {round2_metrics['profit_factor']:.2f} -> {round3_metrics['profit_factor']:.2f}",
        f"  Trades: {round2_metrics['total_trades']} -> {round3_metrics['total_trades']}",
        "",
        "Strategy Shares",
    ]
    for sid, share in _strategy_shares(round3_metrics).items():
        evaluation_lines.append(
            f"  {sid:<10} share={share['share_pct']:.2f}% trades={int(share['trades'])} "
            f"static_pnl=${share['static_pnl']:+,.2f}"
        )
    manager.evaluation_path(round3_dir).write_text("\n".join(evaluation_lines) + "\n", encoding="utf-8")

    return {
        "round_2": {
            "metrics": canonicalize_metrics(round2_metrics),
            "strategy_shares": _strategy_shares(round2_metrics),
            "outputs": {key: str(value) for key, value in round2_outputs.items()},
        },
        "round_3": {
            "metrics": canonicalize_metrics(round3_metrics),
            "strategy_shares": _strategy_shares(round3_metrics),
            "outputs": {key: str(value) for key, value in round3_outputs.items()},
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equity", type=float, default=STARTING_EQUITY)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--round2-config", default=ROUND_2_CONFIG)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    result = promote(parse_args(argv))
    print(json.dumps(result, indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
