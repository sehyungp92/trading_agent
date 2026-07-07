from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from .phase_candidates import (
    INITIAL_EQUITY,
    PHASE_FOCUS,
    PHASE_GATES,
    RISK_STANCE,
    ROUND_NAME,
    ROUND_TARGETS,
    SCORE_WEIGHTS,
    SEED_PORTFOLIO_CONFIG,
    STRATEGY_ORDER,
    get_phase_candidates,
    phase_summary,
)


@dataclass(frozen=True)
class LatestRoundSource:
    key: str
    strategy_id: str
    round_name: str
    summary_path: str
    diagnostics_path: str


@dataclass(frozen=True)
class StrategyAssessment:
    key: str
    strategy_id: str
    round_name: str
    total_trades: float
    trades_per_month: float
    avg_r: float
    total_r_per_month: float
    profit_factor: float
    max_drawdown_pct: float
    win_rate: float
    sharpe: float
    primary_read: str
    risk_read: str
    allocation_bias: str
    summary_path: str
    diagnostics_path: str
    source_hashes: dict[str, str]


LATEST_ROUNDS: tuple[LatestRoundSource, ...] = (
    LatestRoundSource(
        "iaric",
        "IARIC_V5R1",
        "round_1",
        "backtests/output/stock/iaric/round_1/run_summary.json",
        "backtests/output/stock/iaric/round_1/round_final_diagnostics.txt",
    ),
    LatestRoundSource(
        "alcb",
        "ALCB_R3",
        "round_2",
        "backtests/output/stock/alcb/round_2/run_summary.json",
        "backtests/output/stock/alcb/round_2/round_final_diagnostics.txt",
    ),
)

STRATEGY_READS = {
    "iaric": (
        "Latest active IARIC artifact from the manifest supplies the pullback/frequency sleeve.",
        "Current-code parity metrics must drive sizing; avoid inheriting stale archived expectations.",
        "primary_overweight",
    ),
    "alcb": (
        "Latest active ALCB artifact from the manifest supplies the intraday momentum sleeve.",
        "Use as a sized complement with explicit heat and single-strategy share controls.",
        "alpha_complement",
    ),
}


def build_round_design(repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[4]
    latest_rounds = _latest_round_sources(root)
    assessments = _load_strategy_assessments(root, latest_rounds)
    assessment_map = {item.strategy_id: item for item in assessments}
    ordered = [assessment_map[strategy_id] for strategy_id in STRATEGY_ORDER]

    isolated_trades_per_month = sum(item.trades_per_month for item in ordered)
    isolated_total_r_per_month = sum(item.total_r_per_month for item in ordered)
    expected_selected_trades_per_month = isolated_trades_per_month * 0.88
    expected_selected_r_per_month = isolated_total_r_per_month * 0.86

    return {
        "family": "stock",
        "strategy": "portfolio_synergy",
        "round": ROUND_NAME,
        "status": "designed_not_executed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Two-strategy stock phase-auto design using the latest active ALCB and "
            "IARIC diagnostics from their rounds manifests."
        ),
        "risk_stance": RISK_STANCE,
        "initial_equity": INITIAL_EQUITY,
        "starting_equity_assumption": (
            "$25,000 is used as the controlled-aggressive starter book: it is the practical "
            "US equities threshold for frequent intraday stock trading while still making "
            "50-90 bp unit-risk choices meaningful."
        ),
        "diagnostic_sources": [asdict(source) for source in latest_rounds],
        "diagnostic_assessments": [asdict(item) for item in ordered],
        "diagnostic_portfolio_baseline": {
            "isolated_trades_per_month": isolated_trades_per_month,
            "isolated_total_r_per_month_proxy": isolated_total_r_per_month,
            "expected_selected_trades_per_month": expected_selected_trades_per_month,
            "expected_selected_total_r_per_month_proxy": expected_selected_r_per_month,
            "design_target_trades_per_month": ROUND_TARGETS["min_active_trades_per_month"],
            "design_target_total_r_per_month": ROUND_TARGETS["min_total_r_per_month"],
            "design_target_max_drawdown_pct": ROUND_TARGETS["target_max_drawdown_pct"],
            "design_hard_max_drawdown_pct": ROUND_TARGETS["hard_max_drawdown_pct"],
        },
        "seed_portfolio_config": _jsonable(SEED_PORTFOLIO_CONFIG),
        "scoring_weights": SCORE_WEIGHTS,
        "round_targets": ROUND_TARGETS,
        "phase_design": [
            {
                "phase": phase,
                "focus": PHASE_FOCUS[phase],
                "gate": PHASE_GATES[phase],
                "candidates": _jsonable(get_phase_candidates(phase)),
            }
            for phase in sorted(PHASE_FOCUS)
        ],
        "phase_summary": phase_summary(),
        "acceptance_policy": {
            "score_component_limit": 7,
            "min_delta": 0.003,
            "max_rounds_per_phase": 8,
            "hard_rejects": {
                "reject_if_max_drawdown_pct_gt": ROUND_TARGETS["hard_max_drawdown_pct"],
                "reject_if_active_trades_per_month_lt": ROUND_TARGETS["min_active_trades_per_month"] * 0.90,
                "reject_if_positive_alpha_block_rate_gt": 0.22,
                "reject_if_only_one_strategy_active": True,
            },
        },
        "overfit_controls": [
            "Candidate ranking uses live-known trade metadata and never future R outcomes.",
            "Blocked positive-alpha rate is a diagnostic penalty, not a direct hindsight selector.",
            "Final gates require all four temporal slices to be positive.",
            "No single strategy may exceed 70-72% of selected trades or risk in the final blend.",
        ],
    }


def render_markdown(design: dict[str, Any]) -> str:
    baseline = design["diagnostic_portfolio_baseline"]
    lines = [
        "# Stock Portfolio Synergy Round 1 Design",
        "",
        f"Status: {design['status']}",
        f"Generated: {design['generated_at_utc']}",
        f"Initial equity: ${design['initial_equity']:,.0f}",
        f"Risk stance: {design['risk_stance']}",
        "",
        "## Diagnostic Read",
        "",
        "| Strategy | Round | Trades | Trades/Mo | Avg R | R/Mo Proxy | PF | Max DD | Allocation Read |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in design["diagnostic_assessments"]:
        lines.append(
            "| {strategy_id} | {round_name} | {total_trades:.0f} | {trades_per_month:.2f} | "
            "{avg_r:.3f} | {total_r_per_month:.2f} | {profit_factor:.2f} | {max_drawdown_pct:.1%} | "
            "{allocation_bias} |".format(**item)
        )
    lines.extend(
        [
            "",
            "## Portfolio Baseline",
            "",
            f"Isolated frequency proxy: {baseline['isolated_trades_per_month']:.2f} trades/month.",
            f"Expected selected frequency target: {baseline['expected_selected_trades_per_month']:.2f} trades/month.",
            f"Design gate: at least {baseline['design_target_trades_per_month']:.2f} trades/month, "
            f"at least {baseline['design_target_total_r_per_month']:.2f} R/month proxy, "
            f"target DD {baseline['design_target_max_drawdown_pct']:.1%}, "
            f"hard DD {baseline['design_hard_max_drawdown_pct']:.1%}.",
            "",
            "## Seed Allocation",
            "",
            "| Strategy | Unit Risk | Max Heat | Daily Stop | Max Conc | Priority | Role |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    allocations = design["seed_portfolio_config"]["strategy_allocations"]
    for strategy_id in STRATEGY_ORDER:
        alloc = allocations[strategy_id]
        lines.append(
            f"| {strategy_id} | {alloc['unit_risk_pct']:.2%} | {alloc['max_heat_R']:.2f}R | "
            f"{alloc['daily_stop_R']:.2f}R | {alloc['max_concurrent']} | {alloc['priority']} | {alloc['role']} |"
        )
    lines.extend(["", "## Phase Auto Design", ""])
    for phase in design["phase_design"]:
        lines.append(
            f"{phase['phase']}. {phase['focus']} - {len(phase['candidates'])} candidates; "
            f"gate {phase['gate']}"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Both stock strategies start enabled from their latest optimized rounds.",
            "- Scoring has seven components and weights alpha/frequency ahead of, but not instead of, drawdown.",
            "- Candidate ranking uses live-known quality proxies; future blocked-trade R is only used for diagnostics.",
            "- Hard DD is 10%, target DD is 8%, with drawdown tiers cutting risk before the hard ceiling.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_round_design(output_dir: Path, repo_root: Path | None = None) -> dict[str, Path]:
    design = build_round_design(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "run_spec": output_dir / "run_spec.json",
        "seed_config": output_dir / "seed_portfolio_config.json",
        "candidate_space": output_dir / "candidate_space.json",
        "assessment": output_dir / "diagnostic_assessment.json",
        "evaluation": output_dir / "round_evaluation.txt",
    }
    paths["run_spec"].write_text(json.dumps(design, indent=2), encoding="utf-8")
    paths["seed_config"].write_text(
        json.dumps(design["seed_portfolio_config"], indent=2),
        encoding="utf-8",
    )
    paths["candidate_space"].write_text(
        json.dumps(design["phase_design"], indent=2),
        encoding="utf-8",
    )
    paths["assessment"].write_text(
        json.dumps(design["diagnostic_assessments"], indent=2),
        encoding="utf-8",
    )
    paths["evaluation"].write_text(render_markdown(design), encoding="utf-8")
    return paths


def _latest_round_sources(repo_root: Path) -> tuple[LatestRoundSource, ...]:
    sources: list[LatestRoundSource] = []
    for source in LATEST_ROUNDS:
        round_num = _active_manifest_round(repo_root / "backtests" / "output" / "stock" / source.key)
        if round_num is None:
            sources.append(source)
            continue
        round_name = f"round_{round_num}"
        sources.append(
            LatestRoundSource(
                source.key,
                source.strategy_id,
                round_name,
                f"backtests/output/stock/{source.key}/{round_name}/run_summary.json",
                f"backtests/output/stock/{source.key}/{round_name}/round_final_diagnostics.txt",
            )
        )
    return tuple(sources)


def _active_manifest_round(strategy_dir: Path) -> int | None:
    manifest_path = strategy_dir / "rounds_manifest.json"
    if not manifest_path.exists():
        return None
    rounds = _load_json(manifest_path).get("rounds", [])
    active = [
        int(entry["round"])
        for entry in rounds
        if not entry.get("archived")
        and entry.get("round") is not None
        and (strategy_dir / f"round_{int(entry['round'])}" / "run_summary.json").exists()
    ]
    return max(active) if active else None


def _load_strategy_assessments(repo_root: Path, sources: tuple[LatestRoundSource, ...]) -> list[StrategyAssessment]:
    assessments: list[StrategyAssessment] = []
    for source in sources:
        summary_path = repo_root / source.summary_path
        diagnostics_path = repo_root / source.diagnostics_path
        metrics = _load_json(summary_path).get("final_metrics", {})
        total_trades = _metric(metrics, "total_trades")
        trades_per_month = _metric(metrics, "trades_per_month")
        avg_r = _metric(metrics, "avg_r", "expectancy")
        total_r_per_month = trades_per_month * avg_r
        primary_read, risk_read, allocation_bias = STRATEGY_READS[source.key]
        assessments.append(
            StrategyAssessment(
                key=source.key,
                strategy_id=source.strategy_id,
                round_name=source.round_name,
                total_trades=total_trades,
                trades_per_month=trades_per_month,
                avg_r=avg_r,
                total_r_per_month=total_r_per_month,
                profit_factor=_metric(metrics, "profit_factor"),
                max_drawdown_pct=_metric(metrics, "max_drawdown_pct"),
                win_rate=_metric(metrics, "win_rate"),
                sharpe=_metric(metrics, "sharpe"),
                primary_read=primary_read,
                risk_read=risk_read,
                allocation_bias=allocation_bias,
                summary_path=source.summary_path,
                diagnostics_path=source.diagnostics_path,
                source_hashes={
                    "summary": _file_sha256(summary_path),
                    "diagnostics": _file_sha256(diagnostics_path),
                },
            )
        )
    return assessments


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(metrics: dict[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = metrics.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
    return default


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
