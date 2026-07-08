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
    sharpe: float | None
    primary_read: str
    risk_read: str
    allocation_bias: str
    summary_path: str
    diagnostics_path: str
    source_hashes: dict[str, str]


LATEST_ROUNDS: tuple[LatestRoundSource, ...] = (
    LatestRoundSource(
        "nqdtc",
        "NQDTC_v2.1",
        "round_3",
        "backtests/output/momentum/nqdtc/round_3/run_summary.json",
        "backtests/output/momentum/nqdtc/round_3/round_final_diagnostics.txt",
    ),
    LatestRoundSource(
        "vdubus",
        "VdubusNQ_v4",
        "round_3",
        "backtests/output/momentum/vdubus/round_3/run_summary.json",
        "backtests/output/momentum/vdubus/round_3/round_final_diagnostics.txt",
    ),
    LatestRoundSource(
        "downturn",
        "DownturnDominator_v1",
        "round_3",
        "backtests/output/momentum/downturn/round_3/run_summary.json",
        "backtests/output/momentum/downturn/round_3/round_final_diagnostics.txt",
    ),
    LatestRoundSource(
        "nq_regime",
        "NQ_REGIME",
        "round_5",
        "backtests/output/momentum/nq_regime/round_5/run_summary.json",
        "backtests/output/momentum/nq_regime/round_5/round_final_diagnostics.txt",
    ),
)

AVG_R_FALLBACKS = {
    "nqdtc": 0.511,
    "vdubus": 0.440,
    "downturn": 0.425,
    "nq_regime": 1.173,
}

STRATEGY_READS = {
    "nqdtc": (
        "Stable range-regime contributor: 100 trades, 58% WR, PF 2.14, robust return remains positive ex-largest winner.",
        "Latest phase accepted no new candidates, so treat it as confirmation/frequency support rather than primary alpha.",
        "medium_diversifier",
    ),
    "vdubus": (
        "Major frequency/return engine with 198 trades, 6.05 trades/month, PF 2.78, and strong long-hold payoff.",
        "Fast-death and stale-exit diagnostics must be controlled before the optimizer grants more heat.",
        "overweight_with_guards",
    ),
    "downturn": (
        "Low-DD correction/range ballast: 7.14% DD, PF 2.88, useful correction PnL.",
        "Aligned/emerging bear buckets were negative and all trades are fade-driven, so regime specialization matters.",
        "medium_ballast",
    ),
    "nq_regime": (
        "Cleanest latest profile: 497 trades, 10.47 trades/month, PF 7.87, 3.2% DD, all modules active.",
        "Structural Expansion is the weak module and selected only 2.1% of candidates; keep coverage gates.",
        "primary_overweight",
    ),
}


def build_round_design(repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[4]
    assessments = _load_strategy_assessments(root)
    assessment_map = {item.strategy_id: item for item in assessments}
    ordered_assessments = [assessment_map[strategy_id] for strategy_id in STRATEGY_ORDER]

    isolated_trades_per_month = sum(item.trades_per_month for item in ordered_assessments)
    isolated_total_r_per_month = sum(item.total_r_per_month for item in ordered_assessments)
    expected_selected_trades_per_month = isolated_trades_per_month * 0.88
    expected_selected_r_per_month = isolated_total_r_per_month * 0.78

    generated_at = datetime.now(timezone.utc).isoformat()
    return {
        "family": "momentum",
        "strategy": "portfolio_synergy",
        "round": ROUND_NAME,
        "status": "designed_not_executed",
        "generated_at_utc": generated_at,
        "description": (
            "Four-strategy phase-auto round design using the latest NQDTC, Vdubus, "
            "Downturn, and NQ_REGIME diagnostics."
        ),
        "risk_stance": RISK_STANCE,
        "initial_equity": INITIAL_EQUITY,
        "starting_equity_assumption": (
            "$50,000 is used as the controlled-aggressive starter book: large enough for active MNQ engines "
            "to trade without $10k-style fragility, small enough that risk percent choices stay meaningful."
        ),
        "diagnostic_sources": [asdict(source) for source in LATEST_ROUNDS],
        "diagnostic_assessments": [asdict(item) for item in ordered_assessments],
        "diagnostic_portfolio_baseline": {
            "isolated_trades_per_month": isolated_trades_per_month,
            "isolated_total_r_per_month_proxy": isolated_total_r_per_month,
            "expected_selected_trades_per_month": expected_selected_trades_per_month,
            "expected_selected_total_r_per_month_proxy": expected_selected_r_per_month,
            "design_target_trades_per_month": ROUND_TARGETS["min_trades_per_month"],
            "design_target_total_r_per_month": ROUND_TARGETS["min_total_r_per_month"],
            "design_max_drawdown_pct": ROUND_TARGETS["max_drawdown_pct"],
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
            "max_rounds_per_phase": 10,
            "hard_rejects": {
                "no_strategy_disable_without_positive_ablation": True,
                "reject_if_max_drawdown_pct_gt": ROUND_TARGETS["max_drawdown_pct"],
                "reject_if_trades_per_month_lt": ROUND_TARGETS["min_trades_per_month"],
                "reject_if_profit_factor_lt": ROUND_TARGETS["min_profit_factor"],
            },
        },
        "replay_notes": [
            "This artifact is a phase-auto design seed, not an optimized result.",
            "The execution round should evaluate synchronized trade streams before adopting live risk changes.",
            "Candidates that improve standalone return but reduce all-four coverage should be rejected.",
        ],
    }


def render_markdown(design: dict[str, Any]) -> str:
    baseline = design["diagnostic_portfolio_baseline"]
    lines = [
        "# Momentum Portfolio Synergy Round 1 Design",
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
            f"max DD no worse than {baseline['design_max_drawdown_pct']:.1%}.",
            "",
            "## Seed Allocation",
            "",
            "| Strategy | Risk % | Daily Stop | Max Conc | Priority | Role |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    allocations = design["seed_portfolio_config"]["strategy_allocations"]
    for strategy_id in STRATEGY_ORDER:
        alloc = allocations[strategy_id]
        lines.append(
            f"| {strategy_id} | {alloc['base_risk_pct']:.2%} | {alloc['daily_stop_R']:.2f}R | "
            f"{alloc['max_concurrent']} | {alloc['priority']} | {alloc['role']} |"
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
            "- All four momentum strategies start enabled.",
            "- No single strategy may exceed 40% of selected risk contribution.",
            "- The optimizer is allowed to be aggressive on frequency, but DD hard-rejects above 18%.",
            "- Live adoption requires synchronized portfolio replay and completed-bar/fee parity checks.",
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
        json.dumps(design["seed_portfolio_config"], indent=2), encoding="utf-8"
    )
    paths["candidate_space"].write_text(
        json.dumps(design["phase_design"], indent=2), encoding="utf-8"
    )
    paths["assessment"].write_text(
        json.dumps(design["diagnostic_assessments"], indent=2), encoding="utf-8"
    )
    paths["evaluation"].write_text(render_markdown(design), encoding="utf-8")
    return paths


def _load_strategy_assessments(repo_root: Path) -> list[StrategyAssessment]:
    raw: list[tuple[LatestRoundSource, dict[str, Any]]] = []
    observed_months: list[float] = []
    for source in LATEST_ROUNDS:
        summary_path = repo_root / source.summary_path
        metrics = _load_json(summary_path).get("final_metrics", {})
        raw.append((source, metrics))
        total_trades = _metric(metrics, "total_trades")
        trades_per_month = _metric(metrics, "trades_per_month")
        if total_trades > 0 and trades_per_month > 0:
            observed_months.append(total_trades / trades_per_month)

    backtest_months = max(observed_months) if observed_months else 48.0
    assessments: list[StrategyAssessment] = []
    for source, metrics in raw:
        total_trades = _metric(metrics, "total_trades")
        trades_per_month = _metric(metrics, "trades_per_month")
        if trades_per_month <= 0 and backtest_months > 0:
            trades_per_month = total_trades / backtest_months
        avg_r = _metric(metrics, "avg_r", default=AVG_R_FALLBACKS[source.key])
        total_r_per_month = _metric(metrics, "total_r_per_month")
        if total_r_per_month <= 0:
            total_r_per_month = trades_per_month * avg_r
        primary_read, risk_read, allocation_bias = STRATEGY_READS[source.key]
        summary_path = repo_root / source.summary_path
        diagnostics_path = repo_root / source.diagnostics_path
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
                max_drawdown_pct=_metric(metrics, "max_drawdown_pct", "max_dd_pct"),
                win_rate=_metric(metrics, "win_rate"),
                sharpe=_optional_metric(metrics, "sharpe"),
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


def _optional_metric(metrics: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = metrics.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


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
